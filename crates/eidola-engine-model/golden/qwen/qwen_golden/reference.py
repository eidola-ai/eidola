"""The released model code as the oracle: Transformers' `qwen3_5` modules,
built from a checkpoint directory, run on a CPU with the unspecified
operations pinned (`pinned_ops`).

The text model is `Qwen3_5ForCausalLM` exactly as Transformers v5.23.0
defines it. For an FP8 checkpoint (Qwen3.8-27B-FP8) its projections are
Transformers' own `FP8Linear` modules holding the checkpoint's E4M3 weights and
block scales, and only `fp8_linear`, the hub kernel those modules call, is
replaced by the pinned emulation; for a BF16 one (Qwen3.5) they are the model
code's own `nn.Linear`s, whose `F.linear` is pinned like every other. A tied
head (`tie_word_embeddings`) is the embedding's tensor. The weights are
assigned directly from the safetensors files (Transformers' FP8 quantiser,
which needs a GPU, is not involved).

Transformers has no drafter (`_keys_to_ignore_on_load_unexpected = ["^mtp.*"]`).
`Mtp` rebuilds the one vLLM serves (`vllm/model_executor/models/qwen3_5_mtp.py`
at v0.33.0, `Qwen3_5MultiTokenPredictor`) from Transformers' own classes:

- `pre_fc_norm_embedding` of the token's embedding and `pre_fc_norm_hidden` of
  the hidden state (lines 146-160: two `Qwen3_5RMSNorm`s), concatenated
  embedding first and projected by the BF16 `fc` (`torch.cat([inputs_embeds,
  hidden_states])`, then `self.fc`, lines 161-163);
- one decoder layer of the full-attention kind (`Qwen3_5DecoderLayer(...,
  layer_type="full_attention")`, lines 116-125), here Transformers'
  `Qwen3_5DecoderLayer` built from the text config with that one layer type,
  its RoPE the text model's at the row's position;
- the drafter's own norm (`self.norm`, line 187), then the text model's head.

vLLM fuses each residual add into the next norm; Transformers' layer rounds
the stream to BF16 after each add. The oracle is Transformers' layer.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn

from . import st

PREFIX = "model.language_model."
FP8_PROJECTIONS = (
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)  # fmt: skip


def text_config(directory: Path, layers: list[int] | None = None):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    raw = json.loads((directory / "config.json").read_text())
    raw.pop("quantization_config", None)
    cfg = Qwen3_5Config.from_dict(raw).text_config
    if layers is not None:
        assert layers == list(range(len(layers))), "a truncation keeps the first layers"
        cfg.layer_types = cfg.layer_types[: len(layers)]
        cfg.num_hidden_layers = len(layers)
    cfg._attn_implementation = "eager"
    cfg.dtype = torch.bfloat16
    return cfg


def is_fp8(directory: Path) -> bool:
    """Whether the checkpoint stores its projections in FP8 (a
    `quantization_config`); otherwise they are BF16 `nn.Linear`s as built."""
    return "quantization_config" in json.loads((directory / "config.json").read_text())


def _fp8_modules(module: nn.Module) -> None:
    from transformers.integrations.finegrained_fp8 import FP8Linear

    for name, child in list(module.named_modules()):
        for proj in FP8_PROJECTIONS:
            if name.endswith(proj) and isinstance(child, nn.Linear):
                parent = module.get_submodule(name.rsplit(".", 1)[0])
                fp8 = FP8Linear(child.in_features, child.out_features, block_size=(128, 128))
                setattr(parent, name.rsplit(".", 1)[1], fp8)


class Mtp(nn.Module):
    """vLLM's drafter composition from Transformers' modules (see the module
    docstring). Parameter names are the checkpoint's under `mtp.`."""

    def __init__(self, cfg):
        super().__init__()
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

        h, eps = cfg.hidden_size, cfg.rms_norm_eps
        layer_cfg = copy.deepcopy(cfg)
        layer_cfg.layer_types = ["full_attention"]
        layer_cfg.num_hidden_layers = 1
        self.pre_fc_norm_embedding = mq.Qwen3_5RMSNorm(h, eps=eps)
        self.pre_fc_norm_hidden = mq.Qwen3_5RMSNorm(h, eps=eps)
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.layers = nn.ModuleList([mq.Qwen3_5DecoderLayer(layer_cfg, 0)])
        self.norm = mq.Qwen3_5RMSNorm(h, eps=eps)

    @torch.no_grad()
    def forward(self, model, tokens: torch.Tensor, hidden: torch.Tensor, positions: torch.Tensor):
        """Rows at `positions` (consecutive, from 0), each the embedding of
        `tokens[r]` with `hidden[r]`; returns (layer output, normed, logits)."""
        device = model.lm_head.weight.device
        tokens, hidden, positions = tokens.to(device), hidden.to(device), positions.to(device)
        embeds = model.model.embed_tokens(tokens[None])
        x = torch.cat([self.pre_fc_norm_embedding(embeds), self.pre_fc_norm_hidden(hidden[None])], dim=-1)
        x = self.fc(x)
        cos_sin = model.model.rotary_emb(x, positions[None, None, :].expand(3, 1, -1))
        out = self.layers[0](x, position_embeddings=cos_sin, attention_mask=None, position_ids=positions[None, :])
        normed = self.norm(out)
        return out[0].cpu(), normed[0].cpu(), model.lm_head(normed)[0].cpu()


def load(directory: Path, layers: list[int] | None = None, mtp: bool = True):
    """(model, mtp or None) from a checkpoint directory. `pinned_ops.install()`
    must have run."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    cfg = text_config(directory, layers)
    fp8 = is_fp8(directory)
    with torch.device("meta"):
        model = mq.Qwen3_5ForCausalLM(cfg)
        drafter = Mtp(cfg) if mtp else None
        if fp8:
            _fp8_modules(model)
            if drafter is not None:
                _fp8_modules(drafter)
    entries: dict[str, st.Entry] = {}
    for file in sorted(directory.glob("*.safetensors")):
        if file.name != "golden.safetensors":
            entries.update(st.read_file(file))
    keep = set(range(cfg.num_hidden_layers))
    sd, mtp_sd = {}, {}
    for name, e in entries.items():
        if name.startswith(PREFIX):
            local = "model." + name[len(PREFIX) :]
            if local.startswith("model.layers.") and int(local.split(".")[2]) not in keep:
                continue
            sd[local] = e.tensor()
        elif name == "lm_head.weight":
            sd[name] = e.tensor()
        elif name.startswith("mtp."):
            mtp_sd[name[4:]] = e.tensor()
    for d in (sd, mtp_sd):
        for k in list(d):
            if k.endswith("weight_scale_inv"):
                d[k] = d[k].float()
            elif d[k].dtype == torch.float32:
                # `from_pretrained` builds the model in the config's dtype and loads every
                # parameter into it (Qwen3.5 stores `A_log` and the gated norm's weight in
                # F32), as vLLM's loader does: rounded to BF16, ties to even.
                d[k] = d[k].to(cfg.dtype)
    if cfg.tie_word_embeddings:
        # A tied checkpoint has no head of its own: the model's head is the
        # embedding's tensor.
        assert "lm_head.weight" not in sd, "a tied checkpoint with a head of its own"
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    model.load_state_dict(sd, strict=True, assign=True)
    # Non-persistent buffers were built on the meta device: rebuild the RoPE
    # frequencies (under the pins) on the CPU.
    model.model.rotary_emb = mq.Qwen3_5TextRotaryEmbedding(config=cfg)
    tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    assert tied == bool(cfg.tie_word_embeddings), (tied, cfg.tie_word_embeddings)
    model.eval()
    if drafter is not None:
        drafter.load_state_dict(mtp_sd, strict=True, assign=True)
        drafter.eval()
    for m in (model, drafter):
        if m is None:
            continue
        for n, p in list(m.named_parameters()) + list(m.named_buffers()):
            assert p.device.type == "cpu", f"{n} was never assigned"
    return model, drafter


class Recorder:
    """Every decoder layer's output (the residual stream it leaves)."""

    def __init__(self, model):
        self.streams: list[list[torch.Tensor]] = [[] for _ in model.model.layers]
        self.hooks = [
            layer.register_forward_hook(lambda _m, _i, out, k=k: self.streams[k].append(out.detach().clone()))
            for k, layer in enumerate(model.model.layers)
        ]

    def take(self) -> list[torch.Tensor]:
        out = [torch.cat(s, dim=1)[0] for s in self.streams]
        for s in self.streams:
            s.clear()
        return out

    def close(self) -> None:
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def prefill(model, tokens: list[int], device: str = "cpu"):
    """The whole sequence as one call with a cache: (logits, normed, streams,
    cache). Every DeltaNet token runs the recurrence (the pins)."""
    rec = Recorder(model)
    ids = torch.tensor([tokens], device=device)
    out = model.model(input_ids=ids, use_cache=True)
    normed = out.last_hidden_state
    logits = model.lm_head(normed)
    streams = rec.take()
    rec.close()
    return logits[0].cpu(), normed[0].cpu(), [s.cpu() for s in streams], out.past_key_values


@torch.no_grad()
def stepwise(model, tokens: list[int], first: int, device: str = "cpu"):
    """A call of the first `first` tokens, then one token per call with the
    cache: the logits of every position, and the final cache."""
    out = model.model(input_ids=torch.tensor([tokens[:first]], device=device), use_cache=True)
    rows = [model.lm_head(out.last_hidden_state)[0].cpu()]
    cache = out.past_key_values
    for t in tokens[first:]:
        out = model.model(input_ids=torch.tensor([[t]], device=device), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        rows.append(model.lm_head(out.last_hidden_state)[0].cpu())
    return torch.cat(rows), cache


def states(model, cache) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Per DeltaNet layer (by position in the model): the last `kernel - 1` raw
    rows of its convolution input `[kernel - 1, channels]` and its recurrent
    state `[value heads, key dim, value dim]`."""
    out = {}
    for i, kind in enumerate(model.config.layer_types):
        if kind != "linear_attention":
            continue
        conv = cache.get_conv_state(i)[0]  # [channels, kernel]
        kernel = model.config.linear_conv_kernel_dim
        out[i] = (conv[:, -(kernel - 1) :].T.contiguous().cpu(), cache.get_recurrent_state(i)[0].float().cpu())
    return out
