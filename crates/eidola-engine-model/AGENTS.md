# eidola-engine-model — Agent Development Guide

This crate holds the engine's model definitions, text only: the MiMo-V2.6 family (Flash-MOPD and Pro-MOPD) at the crate root, and DeepSeek-V4.1 under `src/deepseek/` ([DeepSeek-V4.1](#deepseek-v41) below). The two share the weight set, the matrix kernels and the comparison metrics, and nothing else: MiMo's modules do not know the second architecture exists.

For MiMo it covers four things:

- turning `config.json` into a typed per-layer table;
- loading the checkpoint from memory-mapped safetensors;
- dequantising every stored format exactly to f32;
- a dense f32 **reference forward** for the main model and the MTP draft layers.

The reference forward is the golden oracle. Every faster executor (paged CPU, GPU) is diffed against it, so its job is to be right and legible. It is not meant to be fast. It must not depend on `eidola-engine`; the dependency points the other way.

## Layout

| Module | Role |
|---|---|
| `config` | `RawConfig` (serde) → `ModelConfig`. Builds the per-layer `LayerSpec` table (attention geometry, RoPE, window, sinks, FFN kind) from the config's own fields. Unsupported features are rejected. Every stored float (RoPE thetas, `layernorm_epsilon`, `attention_value_scale`, `routed_scaling_factor`, the softmax scale) is validated after conversion to f32, as finite and positive (`positive_f32`), so `rope_theta: 1e40` cannot become infinity. `truncated()` keeps a subset of checkpoint layers. Every `quantization_config` key is read (an unknown one is refused) and `QuantSpec` keeps what an executor may need to pin, `activation_scheme` included, though the f32 reference does not use it. Likewise `MoeSpec::router_dtype` (`moe_router_dtype`) and `attention_chunk_size` are kept as declared rather than dropped after parsing. |
| `deepseek` | The second architecture, self-contained. See [DeepSeek-V4.1](#deepseek-v41). |
| `safetensors` | `WeightSet` covers every `*.safetensors` file in a directory. Files are mmapped and tensors read lazily, and a subset of shards or tensors is fine. Index and `__metadata__` values (`tp_size`) are read. One sha256 manifest covers every file the loader reads, shards plus `config.json` and the index (`SEMANTIC_FILES`), whose bytes are read once and only used as hashed (`WeightSet::model_config`). The manifest is keyed by file name, so a name that is not UTF-8 is refused (before the file is opened), never converted lossily. |
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
    - **What feeds each layer is the caller's.** `mtp_forward` takes `h_prev` explicitly and returns both the pre-norm and normed hidden states, so any convention can be built on it. The vendor's serving path feeds every layer the main model's normed state (layer `k` at anchor `x`: `h_x` with `t_{x+k+1}`, RoPE `x`, predicting `t_{x+k+2}`), never another layer's output; llama.cpp chains the pre-norm state instead. The executors' convention and its sources are in `eidola-engine-cpu/AGENTS.md` → MTP row layout; normed versus pre-norm is still settled by acceptance on the GPU (`eidola-engine-cuda/AGENTS.md` → Drafting).

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

## DeepSeek-V4.1

`src/deepseek/` is the text model of DeepSeek-V4.1-Flash: 40 layers over a residual stream carried as four copies (hyper-connections), low-rank attention with one shared 512-dim KV per position, a 128-token sliding window plus up to 512 selected entries of a shared global KV, a mixture of FP4 experts, two Engram n-gram tables, and a three-block draft head. The oracle is the released inference code's full forward at one tensor-parallel rank: every layer on every token (`inference/model.py` of `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `08c8657f`). The reference keeps activations in BF16 and quantises them to FP8 or FP4 at run time; that precision model is the trained function, and it is reproduced here rounding point by rounding point. Nothing is f32-idealised.

### Layout

| Module | Role |
|---|---|
| `config` | Either config shape (the checkpoint's `config.json` with its nested `text_config`, or the flat `inference/config.json`) → `DsConfig`. Every key of both is read and an unknown key is refused, at every nesting level. Floats are validated after conversion to f32. The per-layer table (`DsLayerSpec`) comes from the config's own lists: compress ratio, RoPE variant, the layer whose global KV a layer reads, the layer whose top-k it uses, candidate source and users, Engram slot. `truncated()` keeps a subset of layers and refuses one that is not a model (a consumer without its KV source, a re-indexing layer without the candidate source); dropping a drafter target layer drops the drafter. `with_routed_experts()` uses only the first N routed experts. |
| `numeric` | Every rounding point (`bf16_round`, `round_e4m3`, `round_e2m1`, `round_scale`) and every transcendental (`det_exp`, `det_ln`, `det_sin_cos` and the f32 wrappers `exp32`, `sigmoid32`, `silu32`, `softplus32`, `pow32`, `rsqrt32`), plus `pinned_sum`. |
| `quant` | Run-time activation quantisation: `act_quant` (FP8 codes and block scales for a GEMM), `fake_quant_fp8`, `fake_quant_fp4_pow2`, `fake_quant_fp4_e4m3`. |
| `linear` | `lin8` and `lin4` (the quantised GEMMs over `Fp8Linear` / `Fp4Linear`), `linear_bf16` and `linear_f32` (the products the reference leaves to PyTorch). |
| `rope` | The frequency table, with YaRN on compressed layers, and `rotate` / `unrotate` of a vector's last `rope_dim` elements. |
| `mhc` | Hyper-connection coefficients (`mixes`, with the Sinkhorn steps), `collapse` and `expand`. |
| `attention` | `sparse_attend` (the online-softmax attention over a slot list), `top_k`, `index_score`, `candidate_mask`. |
| `moe` | `route`, `swiglu_hidden`, `shared_expert`, `routed_expert`. |
| `engram` | The hash layout (primes, offsets, multipliers), `EngramHasher`, the `EngramTable` seam with `MappedEngramTable` (rows read lazily from the mapped checkpoint) and `SparseEngramTable` (a test's handful of rows), and `inject`. |
| `weights` | `DsWeights::load`. Dense tensors are decoded once; routed experts stay mapped and are decoded per call; Engram rows are read by index. Vision tensors and `gate.bias_vl` are never read. |
| `forward` | `DsModel`: `embed`, `layer_forward` (one layer over a contiguous range of positions), `head`, and on top of them `forward`, `encode` / `decode` and `bank_stub`. Block-level kernels: `sub_layer_input`, `query`, `window_kv`, `attention_output`, `moe_forward`, `stream_mean`. |
| `drafter` | `drafter_observe`, `drafter_forward`, `draft_logits`, `draft_confidence`, `draft_greedy`, `ring_order`. |

### The forward is one per-layer step

`DsModel::layer_forward(layer, state, rows, opts)` runs one layer over a contiguous range of positions. A layer's output at a position depends on its input at that position, its own window KV for the positions before, and the global entries the position can see. So the same step serves every way of running the model, and they agree bit for bit:

- **The full forward** (`forward`) and any chunking of it, one token per call included.
- **The two halves** (`encode`, `decode`). What crosses is a `Boundary`: the `Rows` entering the first decoder layer (the residual copies, the collapse weights the next sub-layer applies, and the Engram rows later layers may need) and, exactly when that layer is a KV source, its own global bank over every position so far (`BoundaryBank`, which names the layer), which the encoder builds with `bank_stub` from the rows it hands over. The decoder is given the bank and never runs that layer's compressor. Nothing of an encoder layer's bank crosses, so a split is supported only where no decoder layer reads one (`DsModel::check_split`): before layers 1, 2, 8, 14 and 20 of the released model. `encode` refuses any other split before its state changes, and `decode` refuses a boundary made for another split. Test: `every_split_point_is_exact_or_refused_up_front`.
- **A triangular replay.** `LayerOptions::emit_from` makes the rows before it contribute only their window KV. Decoder layer `j` of `d` then emits `t - (d-1-j)·(w-1) ..= t` and takes the `w - 1` rows before those for KV only, and the last position's logits equal the full forward's.
- **A bounded replay.** `LayerOptions::window_start = s` makes a query at `i` see window keys in `[max(s, i - w + 1), i]`. 0 is the reference.

A row that needs a window KV or a global entry the state does not hold is an error, never a silently shorter window.

### Numerics contract

Where the reference's source determines the bits, they are reproduced. Where it leaves them to a compiler, a GPU or a library, one rule is fixed here, chosen so that a token's result is the same whatever else is in the batch and however the sequence is chunked. An executor that wants bit equality calls the public kernels above and does not reimplement them.

- **BF16 between kernels.** Every quantised-linear output, norm, rotation, attention output, expert sum and stream update is rounded to BF16 (nearest, ties to even) where the reference stores it in a BF16 tensor.
- **Casts to FP8 and FP4** round to the nearest code, ties to the even code, after the kernel's own clamp (so they saturate). The kernel source does not state the rounding.
- **Activation scales** follow the kernel's bit arithmetic exactly: `2^ceil(log2(amax / 448))` from the f32 product's exponent and mantissa, with the kernel's floors (`1e-4` for FP8, `6·2^-126` and `6·2^-9` for FP4). The global KV's scale is an E4M3 value, not a power of two, so `amax / scale` can exceed 6 and the cast saturates.
- **Quantised GEMMs.** A 32-wide block dot is a sum of products of small multiples of powers of two. It is computed exactly and rounded once to f32, so it is independent of order and identical on any executor. Blocks accumulate in ascending order in f32 as `(dot · activation scale) · weight scale`, as the kernel does.
- **Plain products and sums** (`F.linear`, `einsum`, `sum`, `mean` in the reference: the router, the head, `wo_a`, the mHC projection, the compressor, index scores, norm statistics) are f32 in the crate's pinned order (`tensor::dot`: 16 interleaved partial sums, a halving tree, the tail; under 16 terms that is left to right). A BF16 result is rounded once.
- **Attention** (`sparse_attend`). One canonical slot list per query: the window's `w` slots in ascending position, right-aligned (a position before the sequence start, or before `window_start`, is an empty slot), then the selected global entries in ascending position. The list is processed in blocks of 64 slots, as the kernel does; an empty slot keeps its place. Per block the score is a pinned-order dot, the normaliser gains the block's weights summed left to right, and the value sum gains each entry in slot order with its weight rounded to BF16 first (the kernel's own asymmetry). This is the reference's layout for one token per call, exactly. **The reference's prefill pass lays the list out differently**: the window part has `min(chunk, w)` slots with a not-yet-full window left-aligned, and the global entries start at slot `min(chunk, w)`. Entries then fall into other 64-slot blocks wherever a query's window is not full and its list crosses a block boundary (at the released window of 128: positions 64–126 of a prefill, and every position of a chunk under 128 tokens that has global entries), and since the weights are rounded to BF16 against each block's running maximum the outputs differ at the 2^-9 level, not the last bit. The reference's own prefill and decode passes therefore disagree with each other, and its prefill depends on the chunk length. The canonical layout is the one that depends on neither. `LayerOptions::reference_prefill` reproduces the prefill layout for a given chunk length, for measurement only: it is how the goldens of a reference prefill are matched exactly.
- **Every top-k** (indexer, candidate blocks, router) uses one total order: higher score first, then lower index. Index scores are BF16, so equal scores are ordinary, and `torch.topk` leaves their order unspecified.
- **Selection never leaves the candidate pool.** The reference fills a short selection with entries outside the pool; a config whose pool could hold fewer entries than the top-k is refused instead.
- **Transcendentals** come from one place, built from correctly rounded `+ - * /` and `sqrt` in f64 and rounded to f32: `det_exp` (the same function as `eidola_engine::sampling::det_exp`), `det_ln`, `det_sin_cos`. No platform `exp`, `log`, `sin` or `cos` is called, so goldens do not depend on the host. Their bits are pinned by `transcendentals_are_pinned`.
- **Nothing is fused.** The RoPE rotation is `a·c − b·s`, `a·s + b·c` in f32; mHC is `m·scale + base`.
- **RoPE tables** are data computed once: `1 / base^(2i/d)` through `pow32`, the YaRN ramp in f32, the angle as the f32 product `position · freq`, then `det_sin_cos`.
- **Engram multipliers are recomputed, not stored.** They are a function of the config alone (NumPy's `default_rng(10007 · layer).integers`, reimplemented: SeedSequence, PCG64, Lemire's bounded draw). A stored constant would be one more input to keep in step with the config, and the synthetic fixture needs other values than the released model. The released model's eight values are pinned by `released_hash_multipliers`. NumPy guarantees the bit generator's stream but not `integers`' across versions; the pin is what holds the values. The compressed-vocabulary map does depend on the tokenizer, so the caller provides it (`DsModel::new`); this crate has no tokenizer.

### Architecture quirks: each one is a silent-wrong-output trap

Each item names the test that fails if it regresses. Every one was checked by mutating the code and watching the named test fail. `fixture` means the exact-equality tests in `tests/deepseek_tiny.rs` (`stepwise_forward_matches_the_reference_exactly` and `sub_layer_outputs_match_the_reference_exactly`, the second of which names the first layer and sub-layer that differ).

1. **The layer pattern comes from the config.** Layers 0–1 are window-only; 2–19 read a ratio-2 global KV from sources 2, 8 and 14; 20–39 read layer 20's ratio-1 KV; 24, 28, 32 and 36 re-index inside layer 20's candidate blocks and the layers after each reuse its selection. Test: `released_layer_table`.
2. **A layer with a compress ratio rotates everything with the compressed table** (θ 160000 with YaRN): queries, window KV, global KV, index keys and queries. Ratio-0 layers and the drafter use θ 10000 without YaRN. Test: `released_layer_table`, `yarn_ramp_of_the_released_config`.
3. **RoPE rotates the last 64 elements** of a head, adjacent elements paired, and the attention output has the query's rotation undone. Test: `rotation_touches_only_the_tail_and_inverts`, fixture.
4. **Window KV is quantised after rotation**, over the whole vector. Test: fixture.
5. **A global entry stands at its group's first position**, and index keys come from the un-rotated latent. A ratio-2 entry is a softmax-gated pool of two tokens in f32, rounded to BF16 before its norm; a ratio-1 entry is a plain BF16 projection. Test: fixture.
6. **The global KV is FP4 with an E4M3 scale per 16; index keys and queries are FP4 with a power-of-two scale per 32.** Test: `fp4_e4m3_scale_can_saturate`, fixture.
7. **The candidate source selects over everything it sees** and only publishes the blocks; the later index sources select inside them. The block holding the query's newest entry is always kept. Test: fixture, `candidate_mask_pins_the_newest_block`.
8. **Attention weights are rounded to BF16 in the value sum and not in the normaliser**, in blocks of 64 slots; the sink joins the normaliser unscaled and contributes no value. Test: fixture, `blocks_follow_slot_positions`.
9. **Each sub-layer applies the collapse weights the sub-layer before it computed**, and its own `post` and `comb`; `comb[i][j]` mixes old copy `i` into new copy `j`; "20 Sinkhorn iterations" is one softmax-and-column step plus 19 row-and-column steps. Test: fixture.
10. **Router:** `sqrt(softplus(·))` scores; the bias steers selection only; weights are the unbiased scores, normalised with `+ 1e-20` and scaled by 1.5. Test: fixture.
11. **The routing weight multiplies the SwiGLU hidden activation before the down-projection quantises it.** The gate branch is clamped from above only, the up branch on both sides. Experts accumulate in ascending index, the shared expert last. Test: fixture.
12. **FP4 experts** pack element `2i` in the low nibble. Test: `fp4_unpacks_low_nibble_first`.
13. **FP8 scales are powers of two per 32×32 tile and are multiplied** (the tensor is named `.scale`); `wo_a` is stored FP8 and used as BF16. Engram rows have one scale per row per 32 columns.
14. **Engram:** lookbacks before the sequence start read the pad id; the hash is a running XOR; one value is added to all four copies, each with its own gate from a signed-square-root dot product. Test: `engram_rows_match_the_reference_hash`, fixture.
15. **The drafter reads the mean of the copies entering its target layers, after Engram** (not the normed input). Test: `stepwise_forward_matches_the_reference_exactly` (the fixture's layer 2 is a target with an Engram module).
16. **Draft slots attend to every draft slot, later ones included, and to the window in ring-slot order**, not position order; confidence reads the state before the drafter's final norm. Test: `drafter_matches_the_reference_exactly`, `window_is_listed_by_ring_slot`. The fixture's window (48) is too narrow for the ring order to change a 64-slot block, so the unit test carries that one.
17. **Vocabulary and image tokens.** Logits are computed for every row of the head. A token equal to `image_token_id` is refused: the vision path is not implemented.

One rule the fixture cannot distinguish: a block dot computed exactly, as against accumulated in f32. The fixture's codes are small enough for the two to agree; the real-weights test is what holds it.

### What is verified against what

The goldens are the released model code run over an **emulated kernel layer** (the reference's kernels are tilelang CUDA kernels) with the library arithmetic it leaves unspecified pinned to the rules above. `golden/deepseek_v41/README.md` says exactly what is emulated and pinned. Under those rules the reference is a function of the tokens alone and this forward equals it bit for bit; how far the real kernels and a GPU's arithmetic are from the rules is measured by `golden/deepseek_v41/compare_real_kernels.py`, which has not yet been run on a GPU.

- **Committed synthetic fixture** (`tests/deepseek_tiny.rs`, `tests/fixtures/deepseek_tiny/`, about 2.7 MB). Eight layers at hidden size 64 covering every mechanism, a three-block drafter, 96 tokens; the window (48), top-k (24) and candidate pool (8 blocks of 4) are small enough that the window slides, both top-ks prune, and an attention list spans two 64-slot blocks.
  - **Exact** (zero elements differ): every layer's streams, each sub-layer's output, the final normed state, the logits, the Engram rows, the drafter's input, and the drafter's tokens, logits and confidences, against the reference run one token per call.
  - **Exact** against the reference run as one prefill chunk, and equal for any chunking (`one_chunk_equals_stepwise_and_the_reference_prefill`). The fixture's window is under one 64-slot block, so a prefill chunk at least a window long has the canonical layout's blocks; at the released window it does not (below).
  - **Exact** split halves at every supported split point, with every other one refused up front, and an exact triangular replay, against the full forward (`every_split_point_is_exact_or_refused_up_front`, `split_halves_equal_the_full_forward`, `triangular_decoder_replay_equals_the_full_forward`).
  - **The short-prefill layout.** Against a 40-token reference prefill (shorter than the window), the forward is exact with that layout reproduced, and with the canonical layout 23 of 40 positions differ: top-1 40/40, KL mean 8.2e-3, max 2.2e-1. The first 16 positions are equal; the layouts part once a query has more than 16 global entries.
  - **Sensitivity to the unpinned bits.** The same reference with PyTorch's own CPU arithmetic and `topk` (nothing pinned but the kernels) gives top-1 90.6 %, KL mean 7.4e-2 against the pinned goldens. A random-weight model amplifies a one-ulp difference through top-k and routing choices; this number says how much the fixture depends on the rules, not how far the forward is from the reference.
- **Local truncated real DeepSeek-V4.1-Flash-1017** (`tests/deepseek_real.rs`, ignored; revision `08c8657f`).
  - The model is checkpoint layers 0, 1, 2, 3, 14, 15, 20, 21, 24, 37, 38 and 39 with the first 32 of each layer's 384 routed experts, the embedding and head, both Engram projections with only the table rows the tokens touch (`SparseEngramTable`), and the three draft blocks with their first 16 experts: about 14 GB fetched by HTTP range reads. Every tensor is the checkpoint's own.
  - 256 tokens of prose and code. Under that length the ratio-1 top-512 selects every entry, so index scores and candidate blocks decide nothing here; the fixture is what exercises pruning.
  - Against the reference with the kernels emulated and the arithmetic pinned, measured:
    - one prefill chunk, its layout reproduced: **0 of 33,095,680 logits differ**, and 0 elements of any of the 12 layers' streams (top-1 256/256, KL 0);
    - a window-long prefill then one token per call: 0 of 28,958,720 logits differ (224 positions);
    - drafter: 480 of 480 draft tokens equal, logits and confidences equal.
  - **The canonical layout against the reference's prefill layout**, same weights and rules, only the 64-slot blocks differing: top-1 165/256 (64.5 %), KL mean 0.28, max 3.5. Against the reference's decode steps that follow its window-long prefill (whose window KV those steps read): top-1 43/96, KL mean 0.25.
  - **Against the reference with PyTorch's own CPU arithmetic and `topk`** (kernels still emulated): top-1 91/256 (35.5 %), KL mean 0.82, max 5.1; the streams' relative error grows from 1.6e-3 after layer 0 to 0.77 after the last layer. Reported by the test, not asserted.
  - **What those two numbers mean.** The statistical target MiMo's real test meets (top-1 ≥ 99.9 %, mean KL < 1e-3) is met here only against the pinned reference, where agreement is exact. This truncation is not a stable function of its unspecified bits: two runs of the reference's own code that differ only in summation order, or only in attention block layout, disagree with each other as much as with us. FP8 and FP4 activations make a one-ulp difference a whole quantisation step at a rounding boundary, the router chooses 6 of 32 kept experts where the model was trained with 384, and layers 4–13, 16–19, 22–23 and 25–36 are missing, so the later layers see inputs they were not trained on. Whether the full model is as sensitive is not known from this test. It needs the full checkpoint on GPUs (`golden/deepseek_v41/compare_real_kernels.py` measures the fixture there, not the real model).
- **Local whole early layers of the real checkpoint** (`tests/deepseek_real_prefix.rs`, ignored; revision `08c8657f`).
  - Checkpoint layers 0–3 whole, all 384 routed experts each, with the Engram rows the tokens touch: about 32 GB. These are the full model's first four layers, so the residual streams after each are the full model's and so is their sensitivity to the bits the reference leaves unspecified. Four layers have no meaningful logits; the measurement is the streams and each token's routed experts. 555 tokens of prose and code. The ratio-2 global KV holds at most 277 entries here, under the top-k of 512, so the indexer never prunes and index scores decide nothing.
  - Against the pinned reference: **0 elements differ** in any layer's streams and every token's routed experts are the same, for one prefill chunk of 555 tokens (its layout reproduced) and for 96 tokens one per call (the canonical layout). Asserted.
  - Against the same reference code with nothing pinned but the kernels (PyTorch's own CPU arithmetic and `topk`), and between the reference's own two layouts (measured with this forward, which equals the pinned reference in both):

    | After layer | Unpinned, prefill: relative error | elements differing | tokens routed the same | Unpinned, 96 decode steps: relative error | Prefill vs decode layout: relative error | elements differing | tokens routed the same |
    |---|---|---|---|---|---|---|---|
    | 0 | 2.9e-3 | 21 % | 554 / 555 | 4.7e-4 | 2.3e-3 | 8 % | 550 / 555 |
    | 1 | 4.3e-3 | 63 % | 503 / 555 | 1.6e-3 | 2.6e-3 | 22 % | 538 / 555 |
    | 2 | 7.1e-3 | 76 % | 476 / 555 | 3.1e-3 | 5.0e-3 | 61 % | 509 / 555 |
    | 3 | 1.1e-2 | 84 % | 440 / 555 | 5.6e-3 | 8.7e-3 | 70 % | 473 / 555 |

    The largest single difference is 0.25 after layer 3 (0.14 between the layouts). The layouts first part at position 64, where a prefill's left-aligned window first crosses a 64-slot block.
  - **Reading.** The instability of the 12-layer truncation is not an artefact of that truncation. With every layer and every expert in place, a run of the reference's own code that differs only in summation order and library rounding is 0.3 % away after one layer and 1.1 % after four, growing by about 1.6× per layer, and by then one token in five goes to a different set of experts. The mechanism is the precision model itself: a last-bit difference before an FP8 or FP4 quantiser becomes a whole code step at a rounding boundary, the dense layer after it spreads that step over every output, and the router's top-6 turns small differences into different experts. The reference without pins is therefore not one function, and "top-1 ≥ 99.9 % against it" is a property of how tolerant the remaining 36 layers and the head are of percent-level stream differences, which four layers cannot show and which the 12-layer truncation (0.77 relative error at its end, top-1 35 %) suggests is not a given. Exact agreement with the reference under fixed rules is what this crate can and does assert; the unpinned numbers are measurements of the model.
  - Record new numbers here whenever numerics change.

### Not here

- Vision: the tower, the image router bias, image positions.
- An acceptance procedure for drafts: the released code has none.
- A replay that resumes the encoder half at a position above 0 with cached global KV (a KV-source layer must see every position from 0, or be given its bank).
- Tensor-parallel ranks above one, which change the reference's rounding.
- Any performance work.
