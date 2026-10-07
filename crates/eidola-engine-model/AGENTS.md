# eidola-engine-model — Agent Development Guide

This crate is the model definition for the MiMo-V2.6 family (Flash-MOPD and Pro-MOPD), text only. It covers four things:

- turning `config.json` into a typed per-layer table;
- loading the checkpoint from memory-mapped safetensors;
- dequantising every stored format exactly to f32;
- a dense f32 **reference forward** for the main model and the MTP draft layers.

The reference forward is the golden oracle. Every faster executor (paged CPU, GPU) is diffed against it, so its job is to be right and legible. It is not meant to be fast. It must not depend on `eidola-engine`; the dependency points the other way.

## Layout

| Module | Role |
|---|---|
| `config` | `RawConfig` (serde) → `ModelConfig`. Builds the per-layer `LayerSpec` table (attention geometry, RoPE, window, sinks, FFN kind) from the config's own fields. Unsupported features are rejected. Every stored float (RoPE thetas, `layernorm_epsilon`, `attention_value_scale`, `routed_scaling_factor`, the softmax scale) is validated after conversion to f32, as finite and positive (`positive_f32`), so `rope_theta: 1e40` cannot become infinity. `truncated()` keeps a subset of checkpoint layers. |
| `safetensors` | `WeightSet` covers every `*.safetensors` file in a directory. Files are mmapped and tensors read lazily, and a subset of shards or tensors is fine. Index and `__metadata__` values (`tp_size`) are read. One sha256 manifest covers every file the loader reads, shards plus `config.json` and the index (`SEMANTIC_FILES`), whose bytes are read once and only used as hashed (`WeightSet::model_config`). |
| `numeric` | Exact BF16, FP8 e4m3fn, E2M1 and E8M0 decoders. |
| `dequant` | FP8 block-scaled linears, MXFP4 experts, and the rank-major pre-sharded fused QKV. |
| `weights` | `ModelWeights::load`. Dense tensors are dequantised once. Routed experts stay mapped and are dequantised per forward call (`ModelWeights::expert`). Optionally folds the value scale into `o_proj`. |
| `tensor` | `Matrix` plus the kernels: `dot`, `linear`, `rms_norm`, `silu` and `sigmoid`. |
| `attention` | RoPE tables, partial rotate-half, window visibility, and single-query attention with a sink. |
| `forward` | `ReferenceModel::forward` (logits at chosen positions, plus optional per-layer residual streams), `mtp_forward`, `route` and `swiglu`. |
| `compare` | Top-1 agreement, KL divergence and max-abs-diff for golden checks, and for certifying faster executors. Fail-closed: any NaN or infinity on either side is a `NonFinite` error, never a metric (a NaN off the winning logit would otherwise pass with perfect top-1, zero KL and zero max-abs-diff). |
| `golden/` | Dev-only Python that produces goldens from the HF remote code. See `golden/README.md`. |

## Numerics contract

- Everything is f32, and stored values decode exactly. BF16, FP8 times an F32 scale, and E2M1 times a power of two are all representable.
- Each kernel computes one output row at a time in a fixed summation order:
  - `dot` uses 16 interleaved partial sums, then a halving tree, then the tail.
  - Attention normalises over keys in ascending position, then adds the sink term.
  - MoE accumulates `w_e · expert_e(x)` in ascending expert index.

  So a token's result is bit-identical whatever else is in the batch, and a prefix run reproduces the prefix rows exactly (`prefix_rows_are_bit_identical`).

  An executor that wants bit equality with the oracle (a paged CPU executor, say) calls these same public kernels: `attention::attend`, `forward::route`, `tensor::dot` and friends. It must not reimplement them.
- RoPE tables are computed in f32 the way the HF reference does: `1 / θ^(2i/d)`, then `· pos`, then `sin`/`cos`.

## Architecture quirks: each one is a silent-wrong-output trap

Each item names the test that fails if it regresses. In the synthetic-fixture tests, each was checked by mutating the code and watching `main_model_matches_hf_remote_code` or `mtp_layers_match_hf_modules` fail.

1. **The layer pattern is irregular and comes from the config.** `hybrid_layer_pattern` gives 0 (global) or 1 (sliding window), and `moe_layer_freq` gives 0 (dense) or 1 (MoE). Flash's global layers are 0, 5, 11, …, 47 and Pro's are 0, 7, 15, …, 62, 69. Never hard-code a ratio. Test: `tests/config.rs`.
2. **KV heads differ by layer type.** Flash has 4 on global layers and 8 on sliding ones, set through the `swa_*` overrides. GQA maps query head `h` to KV head `h / (nq / nkv)`, so groups are contiguous.
3. **Partial NeoX RoPE.** The rotary width is `int(head_dim × partial_rotary_factor)`, a truncation in double precision: `int(192 × 0.334) = 64`. Rotate-half applies to dims `[0, 64)` of each 192-dim Q/K head, and the rest pass through. The HF code reads the factor from `rope_parameters` for the tables and from the top level for the split, so a config where the two disagree is rejected.
4. **θ differs by layer type.** Global layers use `rope_theta` (1e7) and sliding layers use `swa_rope_theta` (1e4).
5. **The sliding window is 128 and includes the query itself.** Key `k` is visible from query `q` iff `q - 127 ≤ k ≤ q`, which is `kv_idx > q_idx - window` in the HF mask.
6. **Sinks are on sliding layers only** (`add_swa_attention_sink_bias`). There is one learned logit per query head. It is appended as an extra softmax column, so it joins the max and the normaliser, and is dropped after normalisation. A checkpoint that carries sinks on a layer the config does not enable them for is refused at load.
7. **`attention_value_scale`** is 0.707 for Flash and 0.612 for Pro. V is multiplied by it before attention. Attention output is linear in V, so the loader folds it into the `o_proj` columns by default (`LoadOptions::fold_value_scale`). The unfolded path is kept for testing. `folded_value_scale_equals_scaling_v` checks the two agree to f32 rounding and that they really are different code paths.

   **Caveat for GPU engines:** the fold is exact only while `o_proj` stays in f32. Rounding `0.707 · W` back to BF16 is lossy, because the scale is not a power of two. Apply the scale as an epilogue scalar instead.
8. **`o_proj` is BF16.** It sits in the FP8 `ignored_layers`. Every other attention and dense-FFN linear is FP8.
9. **The FP8 scales are arbitrary F32**, not powers of two. The weight is `q · weight_scale_inv[r / 128][c / 128]`, and edge tiles may be partial.
10. **The fused QKV is pre-sharded rank-major.**
    - **Layout.** The checkpoint stores `[Q_c | K_c | V_c]` for chunks `c = 0..tp_size`, and the forward wants `[Q | K | V]`.
    - **Scale tiling.** The FP8 scale grid is tiled per chunk. Flash global chunks are 3392 rows (26.5 tiles), so each chunk owns 27 scale rows and the scale grid is `[108, 32]`, not `[106, 32]`. Row `r` uses scale row `(r / R)·27 + (r mod R)/128`, where `R` is the chunk's row count, and not `r / 128`.
    - **Chunk count.** It comes from the checkpoint's `tp_size` metadata, falling back to the config's top-level `num_key_value_heads`.
    - **Oracles.** This matches vLLM's `_shard_fp8_qkv_proj` and llama.cpp's `_tp_aware_qkv_dequant`.
    - **Tests.** `dequant::tests::flash_global_padded_chunks_use_per_chunk_scale_rows` checks the real Flash geometry. The synthetic fixture has a padded chunk of 352 rows (2.75 tiles), and a "continuous grid" mutation fails it.
11. **MXFP4 experts.** Each U8 byte packs two E2M1 codes along K: element `2i` in the low nibble and `2i+1` in the high nibble. The scales are U8 E8M0 (`2^(s-127)`), one per 32 K elements. Gate and up are `[2048, H/2]`; down is `[H, 1024]`.
12. **Router.** The router weight is stored as BF16 (`moe_router_dtype`) and used in f32, as the HF reference does:
    1. Scores are `sigmoid(x·Wᵀ)`.
    2. Selection is top-k on `score + e_score_correction_bias`.
    3. Weights are the unbiased scores at the selected experts.
    4. Weights are normalised by `Σ + 1e-20`.
    5. They are multiplied by `routed_scaling_factor`, which is null in the config and therefore 1.0.

    `n_group == topk_group` makes group routing a no-op; anything else is rejected. Ties in the biased score go to the lower expert index. Using biased weights, or unbiased selection, fails the goldens.
13. **Layer 0 has a dense SwiGLU FFN** (intermediate 16384), and every other layer is MoE, with no shared expert.
14. **Vocab padding.** `vocab_size` (152,576) is the padded row count of `embed_tokens` and `lm_head`. The tokenizer defines 151,675 ids (151,643 base plus 32 added tokens); the last 901 rows are padding that still get logits here. This crate does not know the tokenizer, so it computes every row. The serving core makes padded ids unsampleable: executors are configured with the tokenizer's vocabulary size (`ModelSpec::sampleable_vocab_size`) and sample only over that prefix of each row (`eidola-engine/AGENTS.md`).
15. **Ignored config fields.** `attention_chunk_size` (128) is present but unused by every reference implementation, so it is ignored. Vision, audio and processor configs, and their tensors, are never read.
16. **MTP layers** live at `model.mtp.layers.{i}`, and the HF reference ignores them. Each layer computes `eh_proj([enorm(embed(t)) ‖ hnorm(h_prev)])`, then a sliding-window attention block (SWA geometry, sinks, FP8 QKV), then a dense SwiGLU (`intermediate_size`), then its own `final_layernorm`, then the shared `lm_head`. The post-attention norm is stored as `pre_mlp_layernorm`.
    - **Layer count.** Flash declares `num_nextn_predict_layers: 3`. Pro leaves it null but ships the weights, so the loader takes as many as are present.
    - **Chaining is open.** `mtp_forward` takes `h_prev` explicitly and returns both the pre-norm and normed hidden states, because the oracles do not agree on what feeds the next depth. llama.cpp chains the pre-norm state (and feeds the main model's pre-norm state into depth 0). vLLM and SGLang as of this writing pass the normed output. Which state, which token and which position feed each row is a scheduler decision. Settle it against a serving oracle before relying on acceptance rates.

## What is verified against what

- **Committed synthetic fixture** (`tests/golden_tiny.rs`, `tests/fixtures/tiny/`, about 4.7 MB).
  - The model is 4 layers at hidden size 256:
    - global plus dense;
    - sliding-window MoE twice;
    - global MoE;
    - two MTP layers.
  - All weights are random, in the real storage formats.
  - Goldens come from the HF remote code at a pinned revision, run in fp32 on CPU, with weights dequantised by `golden/mimo_golden/dequant.py`.
  - Tolerance is a max absolute difference of `2e-5 ×` the reference's scale, on every layer's residual stream, the normed hidden state and the logits. The two sides differ only in summation order.
  - Top-1 must agree at every position, and KL must be below 1e-9.
- **Local truncated real Flash-MOPD** (`tests/real_flash.rs`, ignored).
  - The model is checkpoint layers 0, 1, 2 and 5, the embeddings and head, and MTP layer 0, all fetched by HTTP range reads.
  - We are compared against the HF remote code over 363 tokens of real text (prose plus code).
  - Acceptance is top-1 ≥ 99.9 % and mean KL < 1e-3.
  - Measured at checkpoint revision `2479e2d0`:
    - main logits: top-1 363/363, KL mean 1.6e-6 and max 5.7e-4;
    - MTP layer 0: top-1 362/362, KL mean 8e-11;
    - per-layer residual streams agree to about 1e-6.

    The one large position is a router near-tie. At position 323 the 8th and 9th biased scores of checkpoint layer 2 differ by 6e-8 (one f32 ulp), so the selection flips on rounding. When a single position disagrees, first check whether that position's 8th and 9th biased router scores are within an ulp.
  - llama.cpp is a second, independent oracle, with its own QKV de-interleave, MXFP4 repack and attention, run on CPU with an f32 KV cache. On the same truncation and tokens it gives top-1 98.9 % (4 of 363), KL mean 1.2e-4 and max 9.5e-3 against both HF and us. Every top-1 disagreement sits on a small top-2 margin (0.01–0.13, against a median of 0.61). That is consistent with llama.cpp quantising activations to Q8 for MXFP4 matmuls, not with a semantic difference.
  - Record new numbers here whenever numerics change.

## Regenerating goldens

See `golden/README.md`. Python is dev-only and never a build input. `cargo test` needs neither Python nor the network.

## Not here

- The DFlash block-diffusion drafter (`dflash/` in the model repos). Its HF reference omits the sinks and value scale that the vLLM and SGLang ports apply, so it needs its own oracle decision.
- Multimodal encoders.
- The split (non-fused) QKV layout, which is rejected by the config.
- Any performance work.
