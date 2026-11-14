"""Run the reference `model.py` / `engram.py` on a CPU over the emulated kernels.

The reference files are loaded from a directory given by the caller and are not
modified. What this module supplies around them:

- `kernel` is `kernel_shim` (the reference's is tilelang CUDA);
- `image_processor` and `vision` are empty stand-ins: the text path never
  constructs them, and the real ones need Pillow;
- the library arithmetic the reference leaves unspecified is pinned by
  `pinned_ops.install()`, which the caller runs first;
- weights are placed the way the reference's `convert.py` and `load_model` do
  for one tensor-parallel rank;
- forward hooks record every layer's output.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from . import kernel_shim, st

HF_TO_ARGS = {
    "vocab_size": "vocab_size",
    "hidden_size": "dim",
    "moe_intermediate_size": "moe_inter_dim",
    "num_hidden_layers": "n_layers",
    "num_attention_heads": "n_heads",
    "head_dim": "head_dim",
    "qk_rope_head_dim": "rope_head_dim",
    "q_lora_rank": "q_lora_rank",
    "o_lora_rank": "o_lora_rank",
    "o_groups": "o_groups",
    "swiglu_limit": "swiglu_limit",
    "rms_norm_eps": "norm_eps",
    "rope_theta": "rope_theta",
    "n_routed_experts": "n_routed_experts",
    "n_shared_experts": "n_shared_experts",
    "num_experts_per_tok": "n_activated_experts",
    "scoring_func": "score_func",
    "norm_topk_prob": "norm_topk_prob",
    "routed_scaling_factor": "route_scale",
    "sliding_window": "window_size",
    "compress_ratios": "compress_ratios",
    "compress_rope_theta": "compress_rope_theta",
    "kv_source_layer_ids": "kv_source_layers",
    "index_source_layer_ids": "index_source_layers",
    "index_n_heads": "index_n_heads",
    "index_head_dim": "index_head_dim",
    "index_topk": "index_topk",
    "candidate_source_layer_id": "candidate_source_layer",
    "candidate_topk_blocks": "candidate_topk_blocks",
    "candidate_block_size": "candidate_block_size",
    "hc_mult": "hc_mult",
    "hc_sinkhorn_iters": "hc_sinkhorn_iters",
    "hc_eps": "hc_eps",
    "engram_layer_ids": "engram_layer_ids",
    "engram_num_embeddings": "engram_num_embeddings",
    "engram_max_ngram_size": "engram_max_ngram_size",
    "engram_vocab_size": "engram_vocab_size",
    "engram_n_heads": "engram_n_heads",
    "engram_head_dim": "engram_head_dim",
    "engram_pad_token_id": "engram_pad_id",
    "engram_compressed_vocab_size": "engram_compressed_vocab_size",
    "num_nextn_predict_layers": "n_mtp_layers",
    "dspark_block_size": "dspark_block_size",
    "dspark_noise_token_id": "dspark_noise_token_id",
    "dspark_target_layer_ids": "dspark_target_layer_ids",
    "dspark_markov_rank": "dspark_markov_rank",
    "dspark_n_routed_experts": "dspark_n_routed_experts",
    "dspark_num_experts_per_tok": "dspark_n_activated_experts",
}


def args_from_hf_config(config: dict) -> dict:
    """The reference's `ModelArgs` fields for the text model of an HF `config.json`."""
    text = config["text_config"]
    args = {ours: text[theirs] for theirs, ours in HF_TO_ARGS.items()}
    yarn = text["rope_scaling"]
    args.update(
        original_seq_len=yarn["original_max_position_embeddings"],
        rope_factor=yarn["factor"],
        beta_fast=yarn["beta_fast"],
        beta_slow=yarn["beta_slow"],
        dtype="fp8",
        expert_dtype="fp4",
    )
    return args


def load_reference_modules(reference_dir: Path, kernel=kernel_shim):
    """Import the reference's `engram` and `model` modules from `reference_dir`."""
    image_processor = types.ModuleType("image_processor")
    image_processor.IMAGE, image_processor.IMAGE_END = 0, 1
    image_processor.IMAGE_NEW_LINE, image_processor.IMAGE_START = 2, 3
    vision = types.ModuleType("vision")
    vision.Aligner = vision.ViT = None
    sys.modules.update(kernel=kernel, image_processor=image_processor, vision=vision)
    for name in ("engram", "model"):
        spec = importlib.util.spec_from_file_location(name, Path(reference_dir) / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules["model"]


def _dequant_wo_a(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """`convert.py`: wo_a is stored FP8 with block scales and used as BF16."""
    bo, bi = weight.size(0) // scale.size(0), weight.size(1) // scale.size(1)
    w = weight.float().unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)) * scale.float()[:, None, :, None]
    return w.flatten(2, 3).flatten(0, 1).bfloat16()


def place_weights(model, ckpt: dict[str, st.Entry], layer_map=None, routed_experts=None, engram_tables=None):
    """Assign checkpoint tensors to the reference model's parameters.

    `layer_map[i]` is the checkpoint layer that model layer `i` is.
    `routed_experts = (backbone, drafter)` keeps only the first N routed
    experts (each router's rows are cut to match). `engram_tables` maps a
    model layer to a compact `(weight, scale)` pair replacing the stored table."""
    used = set()
    for name, param in model.named_parameters():
        parts = name.split(".")
        if layer_map is not None and parts[0] == "layers":
            parts[1] = str(layer_map[int(parts[1])])
        model_name, name = name, ".".join(parts)
        if engram_tables is not None and ".engram.embed." in name:
            parts = model_name.split(".")
            weight, scale = engram_tables[int(parts[1])]
            value = weight if parts[-1] == "weight" else scale
        elif name.endswith("wo_a.weight"):
            scale_name = name[: -len("weight")] + "scale"
            value = _dequant_wo_a(ckpt[name].tensor(), ckpt[scale_name].tensor())
            used.update((name, scale_name))
        else:
            value = ckpt[name].tensor()
            used.add(name)
            if value.dtype == torch.int8:
                value = value.view(torch.float4_e2m1fn_x2)
            if routed_experts is not None and parts[2:4] == ["ffn", "gate"]:
                value = value[: routed_experts[parts[0] == "mtp"]]
        if value.dtype != param.dtype:
            assert value.dtype == torch.bfloat16 and param.dtype == torch.float32, (name, value.dtype, param.dtype)
            value = value.float()
        assert value.shape == param.shape, (name, tuple(value.shape), tuple(param.shape))
        param.data = value.contiguous()
    return used


class Reference:
    """One reference model, driven the way the released `generate.py` drives it:
    one prefill chunk at position 0, then one token per call."""

    def __init__(self, module, args: dict, tokenizer, max_seq_len: int):
        torch.set_default_dtype(torch.bfloat16)
        self.module = module
        self.args = self.module.ModelArgs(**{**args, "max_batch_size": 1, "max_seq_len": max_seq_len, "temperature": 0.0})
        self.model = self.module.Transformer(self.args, tokenizer)
        self.records: dict[str, list[torch.Tensor]] = collections.defaultdict(list)
        for i, layer in enumerate(self.model.layers):
            layer.register_forward_hook(lambda _m, _i, out, i=i: self.records[f"layer.{i}"].append(out[0][0].clone()))
            layer.attn.register_forward_hook(lambda _m, _i, out, i=i: self.records[f"attn.{i}"].append(out[0].clone()))
            layer.ffn.register_forward_hook(lambda _m, _i, out, i=i: self.records[f"ffn.{i}"].append(out[0].clone()))
        self.model.norm.register_forward_hook(lambda _m, _i, out: self.records["normed"].append(out[0].clone()))
        if self.model.engram_hash is not None:
            self.model.engram_hash.register_forward_hook(lambda _m, _i, out: self.records["engram_rows"].append(out[0].clone()))

    def token_map(self) -> torch.Tensor:
        return self.model.engram_hash.token_map.to(torch.int32)

    def run(self, tokens: list[int], prefill: int, drafter: bool = False) -> dict[str, torch.Tensor]:
        """Feed `tokens[:prefill]` as one chunk, then the rest one at a time.

        Returns per-position tensors: `layer.{i}` (the residual streams after
        block i), `attn.{i}` / `ffn.{i}` (sub-block outputs), `normed`, `logits`,
        `main_hidden`, `engram_rows`. With `drafter`, also the draft head's
        greedy chain at every position from 1 on: at position p the drafter is
        given the token at p + 1 (the first token again at the last position)."""
        self.records.clear()
        ids = torch.tensor([tokens], dtype=torch.long)
        n = len(tokens)
        main_hidden, drafts = [], collections.defaultdict(list)
        steps = [(0, prefill)] + [(p, p + 1) for p in range(prefill, n)]
        for start, end in steps:
            _, _, hidden = self.model.forward(ids[:, start:end], start)
            if hidden is not None:
                main_hidden.append(hidden[0].clone())
            if drafter:
                assert end - start == 1 or start == 0
                given = torch.tensor([tokens[end % n]], dtype=torch.long)
                result = self.model.forward_spec(given, hidden, start)
                if result is not None:
                    out_ids, logits, confidence = result
                    drafts["draft_tokens"].append(out_ids[0, 1:].to(torch.int32))
                    drafts["draft_logits"].append(logits[0].float())
                    drafts["draft_confidence"].append(confidence[0].float())
        out = {k: torch.cat(v, dim=0) for k, v in self.records.items()}
        if main_hidden:
            out["main_hidden"] = torch.cat(main_hidden, dim=0)
        head = self.model.head.weight
        # One row at a time, so a position's logits are the same function of its
        # normed state whichever call pattern produced it.
        out["logits"] = torch.stack([F.linear(row.float(), head) for row in out["normed"]])
        out.update({k: torch.stack(v) for k, v in drafts.items()})
        return out


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
