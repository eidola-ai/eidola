"""The committed synthetic fixture: a Qwen3.8-shaped model at small sizes.

Three Gated DeltaNet layers then one gated full-attention layer (the released
model's pattern, `full_attention_interval` 4), and one MTP layer, in the
checkpoint's storage formats: FP8 E4M3 projections with a BF16 scale per
128×128 block, everything else BF16.

| | Released | Fixture |
|---|---|---|
| hidden, intermediate | 5120, 17408 | 128, 256 |
| DeltaNet key heads, value heads | 16, 48 | 2, 6 (the same 1:3 grouping) |
| DeltaNet key dim, value dim | 128, 128 | 128, 64 (unequal, so a transposed state is caught) |
| attention query heads, KV heads, head dim | 24, 4, 256 | 4, 2, 128 |
| rotary width, `mrope_section` | 64, [11, 11, 10] | 32, [6, 5, 5] |
| vocabulary | 248320 | 512 |

Some `dt_bias` values sit near softplus's threshold of 20, and the norm weights
are zero-centred random values (the gated norm's plain ones around 0.85), so
each of the quirks the forward must get right changes its output.
"""

from __future__ import annotations

import numpy as np
import torch

from . import pinned_ops, st

SEED = 20261116
SEQ_LEN = 80
# The stepwise golden: one call of this many tokens, then one token per call.
STEP_FIRST = 24

VOCAB = 512
IMAGE_TOKEN = VOCAB - 2
VIDEO_TOKEN = VOCAB - 1

TEXT_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_output_gate": True,
    "bos_token_id": 500,
    "dtype": "bfloat16",
    "eos_token_id": 500,
    "full_attention_interval": 4,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 128,
    "initializer_range": 0.02,
    "intermediate_size": 256,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 6,
    "linear_value_head_dim": 64,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 4096,
    "model_type": "qwen3_5_text",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "num_key_value_heads": 2,
    "output_gate_type": "swish",
    "pad_token_id": None,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [6, 5, 5],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": VOCAB,
}

CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "image_token_id": IMAGE_TOKEN,
    "language_model_only": False,
    "model_type": "qwen3_5",
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "modules_to_not_convert": [],
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    },
    "text_config": TEXT_CONFIG,
    "tie_word_embeddings": False,
    "transformers_version": "5.23.0",
    "video_token_id": VIDEO_TOKEN,
    "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 1,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 64,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 128,
        "model_type": "qwen3_5",
        "num_heads": 2,
        "num_position_embeddings": 64,
        "out_hidden_size": 128,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 509,
    "vision_start_token_id": 508,
}


class Gen:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def normal(self, shape, std) -> torch.Tensor:
        return torch.from_numpy(self.rng.standard_normal(shape).astype(np.float32) * np.float32(std))

    def uniform(self, shape, lo, hi) -> torch.Tensor:
        return torch.from_numpy(self.rng.uniform(lo, hi, shape).astype(np.float32))

    def bf16(self, t: torch.Tensor) -> tuple:
        return st.entry(t.to(torch.bfloat16))

    def fp8(self, out: dict, name: str, rows: int, cols: int) -> None:
        """`name.weight` E4M3 and `name.weight_scale_inv` BF16 from a random
        weight of standard deviation `cols^-½`, per 128×128 block."""
        w = self.normal((rows, cols), cols**-0.5)
        blocks = w.view(rows // 128, 128, cols // 128, 128)
        scale = (blocks.abs().amax(dim=(1, 3)) / 448.0).to(torch.bfloat16).float()
        codes = pinned_ops.round_e4m3(blocks / scale[:, None, :, None]).view(rows, cols)
        out[f"{name}.weight"] = st.entry(codes.to(torch.float8_e4m3fn), "F8_E4M3")
        out[f"{name}.weight_scale_inv"] = st.entry(scale.to(torch.bfloat16))


def _mlp(g: Gen, out: dict, p: str) -> None:
    h, i = TEXT_CONFIG["hidden_size"], TEXT_CONFIG["intermediate_size"]
    g.fp8(out, f"{p}.mlp.gate_proj", i, h)
    g.fp8(out, f"{p}.mlp.up_proj", i, h)
    g.fp8(out, f"{p}.mlp.down_proj", h, i)


def _norms(g: Gen, out: dict, p: str) -> None:
    h = TEXT_CONFIG["hidden_size"]
    out[f"{p}.input_layernorm.weight"] = g.bf16(g.normal((h,), 0.15))
    out[f"{p}.post_attention_layernorm.weight"] = g.bf16(g.normal((h,), 0.15))


def _attention(g: Gen, out: dict, p: str) -> None:
    c = TEXT_CONFIG
    h, d, nq, nkv = c["hidden_size"], c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
    g.fp8(out, f"{p}.self_attn.q_proj", 2 * nq * d, h)
    g.fp8(out, f"{p}.self_attn.k_proj", nkv * d, h)
    g.fp8(out, f"{p}.self_attn.v_proj", nkv * d, h)
    g.fp8(out, f"{p}.self_attn.o_proj", h, nq * d)
    out[f"{p}.self_attn.q_norm.weight"] = g.bf16(g.normal((d,), 0.2))
    out[f"{p}.self_attn.k_norm.weight"] = g.bf16(g.normal((d,), 0.2))


def _deltanet(g: Gen, out: dict, p: str) -> None:
    c = TEXT_CONFIG
    h = c["hidden_size"]
    nk, nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
    kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
    conv_dim = 2 * nk * kd + nv * vd
    q = f"{p}.linear_attn"
    g.fp8(out, f"{q}.in_proj_qkv", conv_dim, h)
    g.fp8(out, f"{q}.in_proj_z", nv * vd, h)
    out[f"{q}.in_proj_b.weight"] = g.bf16(g.normal((nv, h), h**-0.5))
    out[f"{q}.in_proj_a.weight"] = g.bf16(g.normal((nv, h), 2.0 * h**-0.5))
    out[f"{q}.conv1d.weight"] = g.bf16(g.normal((conv_dim, 1, 4), 0.4))
    out[f"{q}.A_log"] = g.bf16(g.uniform((nv,), -5.0, -1.0))
    # Two heads straddle softplus's threshold: a + dt_bias crosses 20.
    dt = g.uniform((nv,), -3.0, 3.0)
    dt[1], dt[4] = 19.25, 20.5
    out[f"{q}.dt_bias"] = g.bf16(dt)
    out[f"{q}.norm.weight"] = g.bf16(g.uniform((vd,), 0.7, 1.0))
    g.fp8(out, f"{q}.out_proj", h, nv * vd)


def weights() -> dict:
    g = Gen(SEED)
    c = TEXT_CONFIG
    h = c["hidden_size"]
    out: dict = {}
    out["model.language_model.embed_tokens.weight"] = g.bf16(g.normal((VOCAB, h), 1.0))
    out["lm_head.weight"] = g.bf16(g.normal((VOCAB, h), h**-0.5))
    out["model.language_model.norm.weight"] = g.bf16(g.normal((h,), 0.15))
    for i, kind in enumerate(c["layer_types"]):
        p = f"model.language_model.layers.{i}"
        _norms(g, out, p)
        (_deltanet if kind == "linear_attention" else _attention)(g, out, p)
        _mlp(g, out, p)
    out["mtp.pre_fc_norm_embedding.weight"] = g.bf16(g.normal((h,), 0.15))
    out["mtp.pre_fc_norm_hidden.weight"] = g.bf16(g.normal((h,), 0.15))
    out["mtp.fc.weight"] = g.bf16(g.normal((h, 2 * h), (2 * h) ** -0.5))
    out["mtp.norm.weight"] = g.bf16(g.normal((h,), 0.15))
    _norms(g, out, "mtp.layers.0")
    _attention(g, out, "mtp.layers.0")
    _mlp(g, out, "mtp.layers.0")
    return out


def tokens() -> list[int]:
    rng = np.random.default_rng(SEED + 1)
    out: list[int] = []
    while len(out) < SEQ_LEN:
        t = int(rng.integers(0, VOCAB))
        if t not in (IMAGE_TOKEN, VIDEO_TOKEN):
            out.append(t)
    return out
