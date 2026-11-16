"""Run the Hugging Face MiMo-V2 remote code in fp32 on CPU over a checkpoint
directory, with weights dequantised by `dequant`.

The model is built on the meta device and each decoder layer's weights are
materialised just before that layer runs and dropped right after, so a
truncated real checkpoint (hundreds of experts per layer) fits in memory.
Everything numerical is the remote code's own: only weight placement happens
here.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import dequant
from .remote import import_remote_code


class Checkpoint:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.raw_config = json.loads((self.dir / "config.json").read_text())
        self._handles = {}
        self._where = {}
        self.metadata = {}
        for p in sorted(self.dir.glob("*.safetensors")):
            h = safe_open(str(p), framework="pt")
            self._handles[p.name] = h
            self.metadata.update(h.metadata() or {})
            for k in h.keys():
                self._where[k] = p.name
        idx = self.dir / "model.safetensors.index.json"
        if idx.exists():
            self.metadata.update(json.loads(idx.read_text()).get("metadata", {}))

    def get(self, name: str) -> torch.Tensor:
        return self._handles[self._where[name]].get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self._where

    @property
    def qkv_chunks(self) -> int:
        if "tp_size" in self.metadata:
            return int(self.metadata["tp_size"])
        return int(self.raw_config["num_key_value_heads"])


def hf_config(raw: dict, keep: list[int] | None = None, mtp_block: bool = False):
    configuration, _ = import_remote_code()
    cfg = dict(raw)
    for k in ("vision_config", "audio_config", "processor_config", "quantization_config"):
        cfg.pop(k, None)
    cfg.pop("auto_map", None)
    if mtp_block:
        cfg["num_hidden_layers"] = 1
        cfg["hybrid_layer_pattern"] = [1]
        cfg["moe_layer_freq"] = [0]
    elif keep is not None:
        cfg["num_hidden_layers"] = len(keep)
        cfg["hybrid_layer_pattern"] = [raw["hybrid_layer_pattern"][i] for i in keep]
        cfg["moe_layer_freq"] = [raw["moe_layer_freq"][i] for i in keep]
    config = configuration.MiMoV2Config(**cfg)
    config._attn_implementation = "eager"
    return config


def _set_param(root: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    mod_path, _, pname = name.rpartition(".")
    mod = root.get_submodule(mod_path) if mod_path else root
    old = mod._parameters[pname]
    if old is not None and tuple(old.shape) != tuple(value.shape):
        raise ValueError(f"{name}: shape {tuple(value.shape)} != model {tuple(old.shape)}")
    mod._parameters[pname] = torch.nn.Parameter(value.contiguous(), requires_grad=False)


def _to_meta(mod: torch.nn.Module) -> None:
    for sub in mod.modules():
        for pname, p in list(sub._parameters.items()):
            if p is not None:
                sub._parameters[pname] = torch.nn.Parameter(
                    torch.empty(p.shape, device="meta"), requires_grad=False
                )


def _fresh_rotary(model_core, config) -> None:
    _, modeling = import_remote_code()
    model_core.rotary_emb = modeling.MiMoV2RotaryEmbedding(config=config, is_swa=False)
    model_core.swa_rotary_emb = modeling.MiMoV2RotaryEmbedding(config=config, is_swa=True)


def layer_state(ckpt: Checkpoint, raw: dict, src_prefix: str, is_swa: bool, is_moe: bool) -> dict:
    """fp32 parameters of one decoder layer, keyed relative to the layer module."""
    g = ckpt.get
    if is_swa:
        n_q = raw.get("swa_num_attention_heads", raw["num_attention_heads"])
        n_kv = raw.get("swa_num_key_value_heads", raw["num_key_value_heads"])
        hd = raw.get("swa_head_dim", raw["head_dim"])
        vhd = raw.get("swa_v_head_dim", raw["v_head_dim"])
    else:
        n_q, n_kv, hd, vhd = (
            raw["num_attention_heads"],
            raw["num_key_value_heads"],
            raw["head_dim"],
            raw["v_head_dim"],
        )
    w = g(f"{src_prefix}.self_attn.qkv_proj.weight")
    s_name = f"{src_prefix}.self_attn.qkv_proj.weight_scale_inv"
    s = g(s_name) if ckpt.has(s_name) else None
    st = {
        "self_attn.qkv_proj.weight": dequant.fused_qkv(w, s, n_q, n_kv, hd, vhd, ckpt.qkv_chunks),
        "self_attn.o_proj.weight": dequant.linear(g, f"{src_prefix}.self_attn.o_proj"),
        "input_layernorm.weight": g(f"{src_prefix}.input_layernorm.weight").float(),
    }
    sink = f"{src_prefix}.self_attn.attention_sink_bias"
    if ckpt.has(sink):
        st["self_attn.attention_sink_bias"] = g(sink).float()
    post = f"{src_prefix}.post_attention_layernorm.weight"
    if not ckpt.has(post):
        post = f"{src_prefix}.pre_mlp_layernorm.weight"
    st["post_attention_layernorm.weight"] = g(post).float()
    if is_moe:
        st["mlp.gate.weight"] = g(f"{src_prefix}.mlp.gate.weight").float()
        st["mlp.gate.e_score_correction_bias"] = g(
            f"{src_prefix}.mlp.gate.e_score_correction_bias"
        ).float()
        for e in range(raw["n_routed_experts"]):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                st[f"mlp.experts.{e}.{proj}.weight"] = dequant.linear(
                    g, f"{src_prefix}.mlp.experts.{e}.{proj}"
                )
    else:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            st[f"mlp.{proj}.weight"] = dequant.linear(g, f"{src_prefix}.mlp.{proj}")
    return st


def run_main(ckpt: Checkpoint, tokens: list[int], keep: list[int] | None = None) -> dict:
    """Logits for every position, the residual stream after every layer, and
    the final hidden state before and after the final norm."""
    _, modeling = import_remote_code()
    raw = ckpt.raw_config
    keep = list(range(raw["num_hidden_layers"])) if keep is None else keep
    config = hf_config(raw, keep)
    with torch.device("meta"):
        model = modeling.MiMoV2ForCausalLM(config)
    model.eval()
    _fresh_rotary(model.model, config)
    _set_param(model, "model.embed_tokens.weight", ckpt.get("model.embed_tokens.weight").float())
    _set_param(model, "model.norm.weight", ckpt.get("model.norm.weight").float())
    _set_param(model, "lm_head.weight", ckpt.get("lm_head.weight").float())

    captured: dict[str, torch.Tensor] = {}
    for new_idx, src in enumerate(keep):
        layer = model.model.layers[new_idx]
        is_swa = raw["hybrid_layer_pattern"][src] == 1
        is_moe = bool(raw["moe_layer_freq"][src])

        def pre(mod, args, src=src, is_swa=is_swa, is_moe=is_moe):
            for k, v in layer_state(ckpt, raw, f"model.layers.{src}", is_swa, is_moe).items():
                _set_param(mod, k, v)

        def post(mod, args, out, new_idx=new_idx):
            captured[f"hidden.{new_idx}"] = out[0].detach().clone()
            _to_meta(mod)

        layer.register_forward_pre_hook(pre)
        layer.register_forward_hook(post)

    model.model.norm.register_forward_hook(
        lambda m, a, out: captured.__setitem__("hidden_normed", out[0].detach().clone())
    )
    ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    captured["logits"] = out.logits[0].detach().clone()
    captured["hidden"] = captured[f"hidden.{len(keep) - 1}"]
    return captured


def run_mtp(ckpt: Checkpoint, k: int, tokens: list[int], prev_hidden: torch.Tensor) -> dict:
    """MTP layer k over rows (token, previous hidden state) at positions 0..n.

    Composed from the remote code's own modules: two MiMoV2RMSNorms and the
    `eh_proj` linear in front of a one-layer MiMoV2Model whose single layer is
    sliding-window with a dense MLP (the MTP block) and whose final norm is the
    MTP layer's `final_layernorm`; logits from the shared `lm_head`.
    """
    _, modeling = import_remote_code()
    raw = ckpt.raw_config
    g = ckpt.get
    p = f"model.mtp.layers.{k}"
    config = hf_config(raw, mtp_block=True)
    eps = raw["layernorm_epsilon"]
    h = raw["hidden_size"]

    enorm = modeling.MiMoV2RMSNorm(h, eps=eps)
    hnorm = modeling.MiMoV2RMSNorm(h, eps=eps)
    enorm.weight.data = g(f"{p}.enorm.weight").float()
    hnorm.weight.data = g(f"{p}.hnorm.weight").float()
    eh = g(f"{p}.eh_proj.weight").float()

    with torch.device("meta"):
        block = modeling.MiMoV2Model(config)
    block.eval()
    _fresh_rotary(block, config)
    for name, v in layer_state(ckpt, raw, p, is_swa=True, is_moe=False).items():
        _set_param(block, f"layers.0.{name}", v)
    _set_param(block, "norm.weight", g(f"{p}.final_layernorm.weight").float())

    captured = {}
    block.layers[0].register_forward_hook(
        lambda m, a, out: captured.__setitem__("hidden", out[0].detach().clone())
    )
    embed = g("model.embed_tokens.weight").float()
    lm_head = g("lm_head.weight").float()
    with torch.no_grad():
        e = enorm(embed[torch.tensor(tokens)])
        hp = hnorm(prev_hidden.float())
        x = F.linear(torch.cat([e, hp], dim=-1), eh)
        n = len(tokens)
        out = block(
            inputs_embeds=x[None],
            position_ids=torch.arange(n)[None],
            use_cache=False,
        )
        normed = out.last_hidden_state[0]
        logits = F.linear(normed, lm_head)
    return {
        "hidden": captured["hidden"],
        "hidden_normed": normed.clone(),
        "logits": logits,
    }
