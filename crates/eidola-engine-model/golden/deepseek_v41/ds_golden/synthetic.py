"""The committed synthetic fixture: a DeepSeek-V4.1-shaped model at small sizes.

Every mechanism of the text forward is present, at dimensions where each one
does real work within a 96-token sequence:

| Layer | compress ratio | Role |
|---|---|---|
| 0 | 0 | sliding window only |
| 1 | 0 | sliding window only, Engram |
| 2 | 2 | global-KV source (pooled pairs) with its own indexer, Engram |
| 3 | 2 | reuses layer 2's KV and top-k |
| 4 | 1 | global-KV source with its own indexer; candidate-block source |
| 5 | 1 | reuses layer 4's KV and top-k |
| 6 | 1 | own indexer inside layer 4's candidate blocks |
| 7 | 1 | reuses layer 4's KV and layer 6's top-k |
| drafter 0-2 | 0 | draft blocks; target layers 2, 6, 7 |

The window is 48 and the top-k 24, so a query's attention list reaches 72
entries (two 64-wide softmax blocks), the window slides from position 48, the
ratio-1 top-k prunes from position 24, the ratio-2 top-k from position 49, and
the candidate pool (8 blocks of 4) prunes from position 32.

Layer 2 is a drafter target with an Engram module, so the drafter's input is
seen to be the stream after Engram.

Weights are seeded random values in the checkpoint's storage formats.
"""

from __future__ import annotations

import numpy as np
import torch

from . import kernel_shim as ks
from . import st

SEED = 20251017
SEQ_LEN = 96

TEXT_CONFIG = {
    "model_type": "deepseek_v41_text",
    "vocab_size": 96,
    "hidden_size": 64,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 32,
    "qk_rope_head_dim": 8,
    "q_lora_rank": 32,
    "o_lora_rank": 32,
    "o_groups": 2,
    "hidden_act": "silu",
    "swiglu_limit": 2.0,
    "rms_norm_eps": 1e-20,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "use_cache": True,
    "tie_word_embeddings": False,
    "max_position_embeddings": 1024,
    "rope_theta": 10000,
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": 4,
        "beta_fast": 4,
        "beta_slow": 1,
        "original_max_position_embeddings": 64,
    },
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.5,
    "sliding_window": 48,
    "compress_ratios": [0, 0, 2, 2, 1, 1, 1, 1, 0, 0, 0],
    "compress_rope_theta": 1000,
    "kv_source_layer_ids": [2, 4],
    "index_source_layer_ids": [2, 4, 6],
    "index_n_heads": 2,
    "index_head_dim": 32,
    "index_topk": 24,
    "candidate_source_layer_id": 4,
    "candidate_topk_blocks": 8,
    "candidate_block_size": 4,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-06,
    "engram_layer_ids": [1, 2],
    "engram_num_embeddings": [384, 552],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 50,
    "engram_n_heads": 2,
    "engram_head_dim": 32,
    "engram_pad_token_id": 2,
    "engram_compressed_vocab_size": 0,  # filled in from the token map
    "num_nextn_predict_layers": 3,
    "dspark_block_size": 4,
    "dspark_noise_token_id": 90,
    "dspark_target_layer_ids": [2, 6, 7],
    "dspark_markov_rank": 16,
    "dspark_n_routed_experts": 4,
    "dspark_num_experts_per_tok": 2,
}

CONFIG = {
    "architectures": ["DeepseekV41ForCausalLM"],
    "model_type": "deepseek_v41",
    "dtype": "bfloat16",
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 2,
    "image_token_id": 95,
    "quantization_config": {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    },
    "text_config": TEXT_CONFIG,
}


class SyntheticTokenizer:
    """Stands in for the tokenizer the reference builds its compressed
    vocabulary from: token texts that collide after normalisation (case,
    leading space), two undecodable byte tokens, and a lone space."""

    def __init__(self, vocab: int):
        self.texts = []
        for i in range(vocab):
            word = f"{'W' if i % 2 else 'w'}{(i // 2) % 40}"
            self.texts.append(f" {word}" if i % 3 == 0 else word)
        self.texts[7] = "�"
        self.texts[50] = "x�"
        self.texts[11] = " "
        self.backend_tokenizer = self

    def __len__(self):
        return len(self.texts)

    def decode(self, ids, skip_special_tokens=False):
        return self.texts[ids[0]]

    def id_to_token(self, token_id):
        return f"<raw {token_id}>"


def _pow2_ceil(amax: torch.Tensor, limit: float) -> torch.Tensor:
    return torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-30) / limit)))


def _e8m0_bytes(scale: torch.Tensor) -> torch.Tensor:
    return (torch.log2(scale).round().to(torch.int64) + 127).to(torch.uint8)


def fp8_weight(w: torch.Tensor, row_block: int = 32) -> dict[str, tuple]:
    """FP8 E4M3 codes with one E8M0 scale per `row_block` x 32 tile."""
    out, cols = w.shape
    assert cols % 32 == 0 and out % row_block == 0
    tiles = w.view(out // row_block, row_block, cols // 32, 32)
    scale = _pow2_ceil(tiles.abs().amax(dim=(1, 3)), 448.0)
    value, code = ks.round_to_codes((tiles / scale[:, None, :, None]).clamp(-448, 448), ks.E4M3_TABLE)
    code = (code + 128 * torch.signbit(value).to(code.dtype)).to(torch.uint8).view(out, cols)
    return {"weight": st.entry(code, "F8_E4M3"), "scale": st.entry(_e8m0_bytes(scale), "F8_E8M0")}


def fp4_weight(w: torch.Tensor) -> dict[str, tuple]:
    """Packed E2M1 codes (element 2i in the low nibble) with one E8M0 scale per
    row per 32 columns."""
    out, cols = w.shape
    blocks = w.view(out, cols // 32, 32)
    scale = _pow2_ceil(blocks.abs().amax(-1), 6.0)
    value, code = ks.round_to_codes((blocks / scale[..., None]).clamp(-6, 6), ks.E2M1_TABLE)
    code = (code + 8 * torch.signbit(value).to(code.dtype)).view(out, cols // 2, 2)
    packed = (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8)
    return {"weight": st.entry(packed, "I8"), "scale": st.entry(_e8m0_bytes(scale), "F8_E8M0")}


def build_weights(config: dict) -> dict[str, tuple]:
    c = config["text_config"]
    g = torch.Generator().manual_seed(SEED)
    d, hc = c["hidden_size"], c["hc_mult"]
    tensors: dict[str, tuple] = {}

    def randn(*shape, std=1.0):
        return torch.randn(*shape, generator=g, dtype=torch.float32) * std

    def bf16(name, t):
        tensors[name] = st.entry(t.to(torch.bfloat16))

    def f32(name, t):
        tensors[name] = st.entry(t.to(torch.float32))

    def put(prefix, parts):
        for key, value in parts.items():
            tensors[f"{prefix}.{key}"] = value

    def gain(name, n):
        bf16(name, 1.0 + randn(n, std=0.15))

    def block(p, n_experts, backbone_layer=None):
        inter, qr, hd = c["moe_intermediate_size"], c["q_lora_rank"], c["head_dim"]
        heads, groups, orank = c["num_attention_heads"], c["o_groups"], c["o_lora_rank"]
        gain(f"{p}.attn_norm.weight", d)
        gain(f"{p}.ffn_norm.weight", d)
        put(f"{p}.attn.wq_a", fp8_weight(randn(qr, d, std=d**-0.5)))
        gain(f"{p}.attn.q_norm.weight", qr)
        put(f"{p}.attn.wq_b", fp8_weight(randn(heads * hd, qr, std=1.5 * qr**-0.5)))
        put(f"{p}.attn.wkv", fp8_weight(randn(hd, d, std=d**-0.5)))
        gain(f"{p}.attn.kv_norm.weight", hd)
        f32(f"{p}.attn.attn_sink", randn(heads))
        put(f"{p}.attn.wo_a", fp8_weight(randn(groups * orank, heads * hd // groups, std=(heads * hd // groups) ** -0.5)))
        put(f"{p}.attn.wo_b", fp8_weight(randn(d, groups * orank, std=(groups * orank) ** -0.5)))
        bf16(f"{p}.ffn.gate.weight", randn(n_experts, d, std=2 * d**-0.5))
        f32(f"{p}.ffn.gate.bias", randn(n_experts, std=0.2))
        f32(f"{p}.ffn.gate.bias_vl", randn(n_experts, std=0.2))
        for expert, quant in [("shared_experts", fp8_weight)] + [(f"experts.{e}", fp4_weight) for e in range(n_experts)]:
            put(f"{p}.ffn.{expert}.w1", quant(randn(inter, d, std=2 * d**-0.5)))
            put(f"{p}.ffn.{expert}.w3", quant(randn(inter, d, std=2 * d**-0.5)))
            put(f"{p}.ffn.{expert}.w2", quant(randn(d, inter, std=inter**-0.5)))
        for sub in ("attn", "ffn"):
            f32(f"{p}.hc_{sub}_fn", randn((2 + hc) * hc, hc * d, std=0.1))
            f32(f"{p}.hc_{sub}_base", randn((2 + hc) * hc, std=0.5))
            f32(f"{p}.hc_{sub}_scale", 0.5 + torch.rand(3, generator=g, dtype=torch.float32))
        if backbone_layer is None:
            return
        layer = backbone_layer
        ratio = c["compress_ratios"][layer]
        if layer in c["kv_source_layer_ids"]:
            bf16(f"{p}.attn.compressor.wkv.weight", randn(hd, d, std=d**-0.5))
            if ratio > 1:
                bf16(f"{p}.attn.compressor.wgate.weight", randn(hd, d, std=2 * d**-0.5))
            gain(f"{p}.attn.compressor.norm.weight", hd)
            bf16(f"{p}.attn.indexer.wk.weight", randn(c["index_head_dim"], hd, std=hd**-0.5))
            gain(f"{p}.attn.indexer.k_norm.weight", c["index_head_dim"])
        if layer in c["index_source_layer_ids"]:
            rows = c["index_n_heads"] * c["index_head_dim"]
            put(f"{p}.attn.indexer.wq_b", fp8_weight(randn(rows, qr, std=1.5 * qr**-0.5)))
            bf16(f"{p}.attn.indexer.weights_proj.weight", randn(c["index_n_heads"], d, std=d**-0.5))
        if layer in c["engram_layer_ids"]:
            rows = c["engram_num_embeddings"][c["engram_layer_ids"].index(layer)]
            cols = (c["engram_max_ngram_size"] - 1) * c["engram_n_heads"] * c["engram_head_dim"]
            row_scale = torch.exp2(torch.randint(-3, 2, (rows, 1), generator=g).float())
            table = fp8_weight(randn(rows, c["engram_head_dim"]) * row_scale, row_block=1)
            put(f"{p}.engram.embed", table)
            put(f"{p}.engram.wkv", fp8_weight(randn(d * (hc + 1), cols, std=cols**-0.5)))
            bf16(f"{p}.engram.q_weight", 1.0 + randn(hc, d, std=0.3))
            bf16(f"{p}.engram.k_weight", 1.0 + randn(hc, d, std=0.3))

    bf16("embed.weight", randn(c["vocab_size"], d))
    bf16("head.weight", randn(c["vocab_size"], d, std=3 * d**-0.5))
    gain("norm.weight", d)
    for layer in range(c["num_hidden_layers"]):
        block(f"layers.{layer}", c["n_routed_experts"], layer)
    n_mtp = c["num_nextn_predict_layers"]
    for stage in range(n_mtp):
        block(f"mtp.{stage}", c["dspark_n_routed_experts"])
    targets = len(c["dspark_target_layer_ids"])
    put("mtp.0.main_proj", fp8_weight(randn(d, d * targets, std=(d * targets) ** -0.5)))
    gain("mtp.0.main_norm.weight", d)
    last, rank = f"mtp.{n_mtp - 1}", c["dspark_markov_rank"]
    gain(f"{last}.norm.weight", d)
    bf16(f"{last}.markov_head.embed.weight", randn(c["vocab_size"], rank))
    bf16(f"{last}.markov_head.head.weight", randn(c["vocab_size"], rank, std=rank**-0.5))
    bf16(f"{last}.confidence_head.proj.weight", randn(1, d + rank, std=(d + rank) ** -0.5))
    return tensors


def tokens(config: dict) -> list[int]:
    """A seeded sequence that repeats some n-grams (so Engram rows recur) and
    never contains the image token."""
    rng = np.random.default_rng(SEED)
    vocab = config["text_config"]["vocab_size"]
    seq = rng.integers(0, vocab - 1, size=SEQ_LEN).tolist()
    seq[40:48] = seq[8:16]
    return seq
