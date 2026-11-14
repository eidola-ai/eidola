"""A tiny MiMo-V2-shaped checkpoint with random weights in the real storage
formats, plus golden outputs from the HF remote code.

Shape choices exercise every loader path at a size small enough to commit:
global and sliding-window layers with different KV-head counts, sinks on the
sliding layers only, a dense first layer, routed MXFP4 experts with
sigmoid + correction-bias routing, FP8 linears with partial edge tiles, and
a fused QKV pre-sharded rank-major over two chunks whose global-layer chunk
(352 rows = 2.75 tiles) is padded in its scale grid, as the real Flash
global layers are.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save

from . import hf_run

SEED = 20261007
WINDOW = 8
SEQ_LEN = 40
FP8_MAX = 448.0

CONFIG = {
    "architectures": ["MiMoV2ForCausalLM"],
    "model_type": "mimo_v2",
    "attention_projection_layout": "fused_qkv",
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "hybrid_layer_pattern": [0, 1, 1, 0],
    "moe_layer_freq": [0, 1, 1, 1],
    "vocab_size": 256,
    "intermediate_size": 192,
    "moe_intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 96,
    "v_head_dim": 64,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 4,
    "swa_head_dim": 96,
    "swa_v_head_dim": 64,
    "rope_theta": 10000000.0,
    "swa_rope_theta": 10000.0,
    "partial_rotary_factor": 0.334,
    "rope_parameters": {
        "partial_rotary_factor": 0.334,
        "rope_theta": 10000000.0,
        "rope_type": "default",
        "type": "default",
    },
    "sliding_window": WINDOW,
    "sliding_window_size": WINDOW,
    "attention_chunk_size": 128,
    "add_full_attention_sink_bias": False,
    "add_swa_attention_sink_bias": True,
    "attention_value_scale": 0.707,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hybrid_block_size": None,
    "n_routed_experts": 8,
    "num_experts_per_tok": 3,
    "n_shared_experts": None,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": None,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "moe_router_dtype": "bfloat16",
    "layernorm_epsilon": 1e-06,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "max_position_embeddings": 4096,
    "num_nextn_predict_layers": 2,
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "store_dtype": "mxfp4",
        "mxfp4_block_size": 32,
        "weight_block_size": [128, 128],
        "ignored_layers": [f"model.layers.{i}.self_attn.o_proj" for i in range(4)],
    },
    "dtype": "bfloat16",
}
QKV_CHUNKS = 2


class Gen:
    def __init__(self, seed: int):
        self.g = torch.Generator().manual_seed(seed)

    def normal(self, *shape, std=1.0, mean=0.0):
        return torch.randn(*shape, generator=self.g) * std + mean

    def uniform(self, *shape, lo=0.0, hi=1.0):
        return torch.rand(*shape, generator=self.g) * (hi - lo) + lo

    def randint(self, lo, hi, *shape):
        return torch.randint(lo, hi, shape, generator=self.g)


def bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16)


def quantize_fp8(w: torch.Tensor, gen: Gen) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-scaled FP8 with deliberately non-power-of-two F32 scales."""
    r, c = w.shape
    br, bc = (r + 127) // 128, (c + 127) // 128
    scale = torch.empty(br, bc)
    q = torch.empty(r, c, dtype=torch.float8_e4m3fn)
    for i in range(br):
        for j in range(bc):
            tile = w[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128]
            s = tile.abs().max().item() / FP8_MAX * gen.uniform(1, lo=1.05, hi=1.6).item()
            scale[i, j] = s
            q[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = (tile / s).to(
                torch.float8_e4m3fn
            )
    return q, scale


def quantize_fp8_sharded(w: torch.Tensor, chunks: int, gen: Gen):
    """Quantise each rank-major chunk on its own tile grid (per-chunk scale rows)."""
    per = w.shape[0] // chunks
    qs, ss = [], []
    for c in range(chunks):
        q, s = quantize_fp8(w[c * per : (c + 1) * per], gen)
        qs.append(q)
        ss.append(s)
    return torch.cat(qs, 0), torch.cat(ss, 0)


def mxfp4_random(rows: int, cols: int, gen: Gen):
    packed = gen.randint(0, 256, rows, cols // 2).to(torch.uint8)
    scale = gen.randint(119, 123, rows, cols // 32).to(torch.uint8)
    return packed, scale


def attention_tensors(prefix: str, gen: Gen, is_swa: bool, cfg: dict) -> dict:
    h = cfg["hidden_size"]
    nq = cfg["swa_num_attention_heads"] if is_swa else cfg["num_attention_heads"]
    nkv = cfg["swa_num_key_value_heads"] if is_swa else cfg["num_key_value_heads"]
    hd, vhd = cfg["head_dim"], cfg["v_head_dim"]
    rows = nq * hd + nkv * (hd + vhd)
    # Random rows are generated directly in the checkpoint's rank-major order.
    w = gen.normal(rows, h, std=0.06)
    q, s = quantize_fp8_sharded(w, QKV_CHUNKS, gen)
    t = {
        f"{prefix}.self_attn.qkv_proj.weight": q,
        f"{prefix}.self_attn.qkv_proj.weight_scale_inv": s,
        f"{prefix}.self_attn.o_proj.weight": bf16(gen.normal(h, nq * vhd, std=0.06)),
        f"{prefix}.input_layernorm.weight": bf16(gen.normal(h, std=0.1, mean=1.0)),
    }
    if is_swa:
        t[f"{prefix}.self_attn.attention_sink_bias"] = bf16(gen.normal(nq, std=0.5, mean=0.8))
    return t


def dense_mlp(prefix: str, gen: Gen, cfg: dict) -> dict:
    h, i = cfg["hidden_size"], cfg["intermediate_size"]
    t = {}
    for name, shape in (("gate_proj", (i, h)), ("up_proj", (i, h)), ("down_proj", (h, i))):
        q, s = quantize_fp8(gen.normal(*shape, std=0.06), gen)
        t[f"{prefix}.mlp.{name}.weight"] = q
        t[f"{prefix}.mlp.{name}.weight_scale_inv"] = s
    return t


def build_checkpoint(cfg: dict) -> dict:
    gen = Gen(SEED)
    h, v = cfg["hidden_size"], cfg["vocab_size"]
    t = {
        "model.embed_tokens.weight": bf16(gen.normal(v, h, std=0.5)),
        "lm_head.weight": bf16(gen.normal(v, h, std=0.08)),
        "model.norm.weight": bf16(gen.normal(h, std=0.1, mean=1.0)),
    }
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        is_swa = cfg["hybrid_layer_pattern"][i] == 1
        t.update(attention_tensors(p, gen, is_swa, cfg))
        t[f"{p}.post_attention_layernorm.weight"] = bf16(gen.normal(h, std=0.1, mean=1.0))
        if cfg["moe_layer_freq"][i]:
            e, mi = cfg["n_routed_experts"], cfg["moe_intermediate_size"]
            t[f"{p}.mlp.gate.weight"] = bf16(gen.normal(e, h, std=0.06))
            t[f"{p}.mlp.gate.e_score_correction_bias"] = gen.normal(e, std=0.15, mean=1.9)
            for x in range(e):
                for name, (r, c) in (
                    ("gate_proj", (mi, h)),
                    ("up_proj", (mi, h)),
                    ("down_proj", (h, mi)),
                ):
                    packed, scale = mxfp4_random(r, c, gen)
                    t[f"{p}.mlp.experts.{x}.{name}.weight"] = packed
                    t[f"{p}.mlp.experts.{x}.{name}.weight_scale"] = scale
        else:
            t.update(dense_mlp(p, gen, cfg))
    for k in range(cfg["num_nextn_predict_layers"]):
        p = f"model.mtp.layers.{k}"
        t.update(attention_tensors(p, gen, True, cfg))
        t[f"{p}.pre_mlp_layernorm.weight"] = bf16(gen.normal(h, std=0.1, mean=1.0))
        t.update(dense_mlp(p, gen, cfg))
        t[f"{p}.enorm.weight"] = bf16(gen.normal(h, std=0.1, mean=1.0))
        t[f"{p}.hnorm.weight"] = bf16(gen.normal(h, std=0.1, mean=1.0))
        t[f"{p}.final_layernorm.weight"] = bf16(gen.normal(h, std=0.1, mean=1.0))
        t[f"{p}.eh_proj.weight"] = bf16(gen.normal(h, 2 * h, std=0.06))
    return t


def generate(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    tensors = build_checkpoint(cfg)
    save_deterministic(
        {k: v.contiguous() for k, v in sorted(tensors.items())},
        out_dir / "model.safetensors",
        metadata={"tp_size": str(QKV_CHUNKS)},
    )

    ckpt = hf_run.Checkpoint(out_dir)
    gen = Gen(SEED + 1)
    tokens = gen.randint(0, cfg["vocab_size"], SEQ_LEN).tolist()
    main = hf_run.run_main(ckpt, tokens)

    golden = {
        "tokens": torch.tensor(tokens, dtype=torch.int64),
        "logits": main["logits"],
        "hidden_normed": main["hidden_normed"],
    }
    for i in range(cfg["num_hidden_layers"]):
        golden[f"hidden.{i}"] = main[f"hidden.{i}"]

    # MTP chain: layer k row i embeds token i+k+1 and reads the previous
    # depth's pre-norm hidden state at row i, at position i.
    prev = main["hidden"]
    for k in range(cfg["num_nextn_predict_layers"]):
        n = SEQ_LEN - k - 1
        mtp_tokens = tokens[k + 1 : k + 1 + n]
        prev = prev[:n]
        out = hf_run.run_mtp(ckpt, k, mtp_tokens, prev)
        golden[f"mtp.{k}.tokens"] = torch.tensor(mtp_tokens, dtype=torch.int64)
        golden[f"mtp.{k}.prev_hidden"] = prev.clone()
        golden[f"mtp.{k}.hidden"] = out["hidden"]
        golden[f"mtp.{k}.hidden_normed"] = out["hidden_normed"]
        golden[f"mtp.{k}.logits"] = out["logits"]
        prev = out["hidden"]

    save_deterministic(
        {k: v.contiguous() for k, v in golden.items()},
        out_dir / "golden.safetensors",
        metadata={"generator": "hf-remote-code fp32 cpu", "seq_len": str(SEQ_LEN)},
    )


def save_deterministic(tensors: dict, path: Path, metadata: dict[str, str]) -> None:
    """`safetensors.torch.save_file`, with the header's `__metadata__` keys in
    sorted order. The library keeps metadata in a hash map, so with more than
    one key its serialized order (and the file's bytes) varies run to run even
    though every tensor is identical; sorting makes the fixture a function of
    its contents."""
    data = save(tensors, metadata=metadata)
    n = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + n])
    if "__metadata__" in header:
        header["__metadata__"] = dict(sorted(header["__metadata__"].items()))
    text = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    if len(text) > n:
        raise ValueError("re-serialized header grew")
    text += b" " * (n - len(text))
    path.write_bytes(data[:8] + text + data[8 + n :])
