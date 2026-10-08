# eidola-engine-cuda

The CUDA executor for the MiMo-V2.6 engine: the serving core's `Executor` on one Blackwell GPU, running the kernels `eidola-engine-kernels` builds ahead of time, through the CUDA driver API. It is held to the seam contract (`eidola-engine/src/executor.rs`, normative) and diffed against the CPU reference executor and the model crate's f32 reference forward. AGPL-3.0-only. Workspace conventions live in the root `AGENTS.md`.

| Module | Contents |
|---|---|
| `executor.rs` | `CudaExecutor`, `CudaExecutorConfig`: the seam (maintenance, tables, forward, sampling), contract checks. |
| `model.rs` | `GpuModel` (the batched target forward), `Kernels` (every loaded kernel), KV grouping of layers, RoPE tables. |
| `weights.rs` | The checkpoint on the device in kernel layouts (no dequantization). |
| `kv.rs` | `KvStore`: KV pools per group, device block tables with a host mirror, per-slot state, `Zero`/`Copy`/`ResetSlot`. |
| `gemm.rs` | CUTLASS FP8-blockwise and BF16 GEMMs: their `Params` built in Rust. |
| `moe_gemm.rs` | DeepGEMM FP8 × MXFP4 grouped GEMM (masked and contiguous): its TMA descriptors built in Rust. |
| `attention.rs` | FlashInfer FA2 sink attention: `PagedParams` mirror and the per-step work list. |
| `sampler.rs` | The device sampler and chain acceptance. |
| `engine_ops.rs` | Wrappers for our own kernels (`engine_ops.cu`). |
| `support.rs` | `check_supported`: the one model configuration the kernels can run; `Unsupported` names the field. |
| `tma.rs` | `cuTensorMapEncodeTiled` specs, CuTe's descriptor fix-up. |
| `device.rs`, `module.rs`, `launch.rs`, `ops.rs`, `bf16.rs` | Device, verified image loading and launch, argument lists, the RMSNorm smoke kernel and DeepGEMM's grid, BF16 bits. |

## Host boundary

- **Driver API only.** `cudarc` with `driver` + `dynamic-loading`, bindings pinned to CUDA 13.0 (`cuda-13000`, the R580 driver's API). No CUDA runtime, no NVRTC, nothing linked at build time: the crate builds and its tests run on a machine without CUDA. Do not enable `cudarc`'s `nvrtc` feature; `module.rs` calls `cuModuleLoadData` directly.
- **No host C++.** Every upstream kernel's launch parameters are built in Rust:
  - **CUTLASS** (`gemm.rs`): the 2048-byte `GemmUniversal::Params` (problem shape, mainloop TMA descriptors and, for the blockwise kernel, scale-factor pointers and layouts, epilogue alpha and output descriptor, the cluster-launch-control scheduler's tile counts, divisors and raster order, hardware info), field by field. Descriptors are encoded by the driver and post-processed exactly as CuTe does (bit 21 of word 1 cleared unless the first 128 KiB are dense). Pinned by `tests/cutlass_params.rs` against CUTLASS's own host path: every byte CUTLASS defines must match.
  - **DeepGEMM** (`moe_gemm.rs`): the five 2-D descriptors its host code builds (L2 promotion 256 B, packed-FP4 B as `16U4_ALIGN16B`), the `Identity` epilogue arguments, grid = the instance's SM count.
  - **FlashInfer** (`attention.rs`): `PagedParams` as a `repr(C)` struct with compile-time size and offset checks against the generated C++ (296 bytes), CCCL's `fast_mod_div` reproduced for `uint_fastdiv`, and the non-split work list (one CTA per request × query tile × KV head). Sliding layers list only the pages some query can see.
- **Images only through `KernelDir`**, size and SHA-256 checked against the compiled-in manifest. `KernelDir::new` takes only a directory: no other manifest can be supplied, so nothing but the compiled-in hashes can approve bytes.
- **Every driver call binds the context first.** The current context is per thread and the executor may run on a thread other than the one that loaded it. `Kernel::launch` binds its module's context, and the paths that make raw driver calls bind before them: descriptor encoding in the GEMM launches and KV maintenance. A plain launch happened to work without a current context on R580, but descriptor encoding fails with `CUDA_ERROR_INVALID_CONTEXT` (`tests/smoke.rs` launches all three from a fresh thread).
- **Every launch wrapper checks its kernel's limits on the host** and returns an error instead of launching. The checks cover divisibility (128- and 512-wide quantization groups), row and stride coverage, at most 256 experts and 8 per token, whole GQA groups, a vocabulary of at most 2^20 for both sampling and acceptance, and whole rows in every array (`tests/preconditions.rs`). Device-resident indices (block ids, logit rows) are the executor's to keep in range; it builds them from the host mirror. CC 10.0 runs `sm_100a`, 10.3 runs `sm_103a`, any other 10.x runs `sm_100f`; a foreign architecture's cubin is refused, never JIT-compiled. `CudaExecutorConfig::image` can force the family image.

## The dev-only oracle

`oracle/cutlass_params.cu` is a host program that `#include`s the kernels crate's two CUTLASS translation units, so its types are the AOT kernels' types, runs CUTLASS's `to_underlying_arguments` / `get_grid_shape` / `can_implement` for a list of shapes and fake addresses, logs every `cuTensorMapEncodeTiled` call, and prints the Params bytes with a mask of the bytes CUTLASS defines (padding is found by building the struct three times over different stack contents). Its output is `tests/data/cutlass_params.jsonl`. It is compiled with any CUDA 13 toolkit and the pinned CUTLASS tree on a GPU host (descriptor encoding needs a context); the command is in its header. Never part of the engine. Regenerate it when the CUTLASS pin or either GEMM instantiation changes.

## Supported configuration

**The executor supports exactly the configuration space a real checkpoint exercises and the tests cover: no permitted-but-untested degrees of freedom.** That is MiMo-V2.6-Flash's configuration, field for field, for the whole model or any subset of its layers. `check_supported` runs before any device memory is touched and refuses everything else with `CudaError::Unsupported`, naming the field. Shapes are template arguments or compile-time constants in the kernels, so another shape would read or write out of bounds rather than fail. The scalars the kernels take at run time are held to the tested values too.

| Field | Required | Why |
|---|---|---|
| hidden size | 4096 | DeepGEMM instances, UE8M0 words |
| vocabulary | 152,576 | the BF16 `lm_head` GEMM |
| dense intermediate | 16,384 | the FP8 GEMMs, f32-scale SwiGLU |
| every layer: QK / V head dims, rotated dims | 192 / 128 / 64 | FA2 instances, `eidola_qkv_rope_kv`, RoPE tables |
| every layer: query heads | 64 | one attention plan per KV group serves all its layers |
| global layers: KV heads, RoPE θ, sinks | 4, 1e7, none | as Flash |
| sliding layers: KV heads, window, RoPE θ, sinks | 8, 128, 1e4, present | as Flash |
| experts, expert intermediate | 256, 2048 | DeepGEMM's group count and instances; router and placement arrays |
| experts per token | exactly 8 | the router kernel always renormalizes, which the reference skips for one expert |
| `norm_topk_prob`, routing scale | true, 1.0 | the router kernel |
| epsilon, value scale | 1e-6, 0.707 | the tested values |
| context window (`max_position_embeddings`) | 1,048,576 | as Flash |
| `max_model_len` | at most the context window | positions past it are outside what the checkpoint declares |
| sampleable vocabulary | at most 2^20, and no more than the head | `sampling.cu`'s chunks |

Each field, changed alone, is refused (`support::tests`). Pro differs in its hidden size (6144), query heads (128), global KV heads (8), experts (384), value scale and epsilon; serving it is the multi-GPU executor's scope, not a widening of this one. Every f32 vector loaded (norms, sinks, router bias) must have exactly the model's width; the BF16 and FP8 tensors' shapes are checked likewise.

**The device is part of the configuration.** `check_device` holds every device-side requirement a retained kernel bakes in, and refuses the rest with `CudaError::Unsupported`, naming the field:

| Field | Required | Why |
|---|---|---|
| compute capability | 10.x, and a forced image this device runs (`sm_100f` on any 10.x; `sm_100a` / `sm_103a` only on their own) | the images are Blackwell datacenter code |
| opt-in shared memory per block | at least `REQUIRED_SMEM_PER_BLOCK` (231,424 bytes) | the largest launch: CUTLASS's BF16 GEMM (`o_proj`, `lm_head`), dynamic plus static |
| SM count (when any layer has experts) | at least 148 | DeepGEMM's persistent instances are compiled for 148 SMs |
| cluster launch (when any layer has experts) | supported | DeepGEMM launches 2-CTA clusters, the only kernels that do |

A dense-only subset of layers needs neither of the last two. `support::tests` refuses a synthetic device missing each requirement alone; `tests/smoke.rs` reads every kernel's launch contract and static shared memory from the loaded images, and fails if the largest is not exactly `REQUIRED_SMEM_PER_BLOCK` or if anything but DeepGEMM launches a cluster. The launch-time guards (`ops::deepgemm_grid`, the driver's own refusal) stay as defense in depth.

**Construction order is structural.** `CudaExecutor::new` checks, in order:

1. the configuration, the sampleable vocabulary and the device;
2. the KV geometry (`GroupGeometry::validate`: every product in checked `usize`, a block within the 32-bit page stride, block ids within the `i32` tables);
3. it loads and verifies every kernel image.

Only then does it move weights. `ModelWeights::load` takes the loaded `Kernels`, so no weight can reach the device first, and a wrong device, image or kernel directory fails in about a second instead of after the checkpoint has filled the GPU (`tests/construction.rs`). Scratch sizes are computed with checked arithmetic, all of them before any allocation.

## The executor

- **Seam.** One step: maintenance in order, then table updates in order (`KvStore::apply`, memsets and device-to-device copies on the stream, dirty table rows uploaded), the forward over every host token, sampling on the device over `sampleable_vocab_size`. Block tables live on the device and change only through the seam; the host mirror is what plans attention and checks the contract.
- **Contract checks**, as in the CPU reference executor, before anything is launched: a write into a block mapped by more than one table entry, a read through an unmapped entry, positions disagreeing with `context_len`, any drafts (speculation is not implemented yet: `max_draft_tokens` is 0), and drafts at position 0 above all, panic.
- **Memory.** One pool per KV group (global, sliding; as the CPU executor groups them). Block `b` holds every layer's K `[block, kv heads, 192]` then V `[block, kv heads, 128]`, BF16, contiguous, so `Zero` and `Copy` are single ranges and paged attention addresses a layer with page stride = the whole block. Per-slot model state exists (`state_width`) and is scrubbed by `ResetSlot`; it is empty until a drafter needs it.
- **Forward** (`model.rs`): f32 residual stream; per layer RMSNorm → FP8 (f32 per-128 scales) → fused-QKV FP8 GEMM (BF16) → RoPE (host-computed tables, bit-identical to the reference's) + KV write → FA2 attention → `o_proj` BF16 GEMM with the value scale as alpha (f32) → residual; RMSNorm → dense SwiGLU (FP8 GEMMs) or experts: router top-k in f32, expert-major placement (masked layout, 128 rows per expert, for batches of at most 128 tokens; contiguous otherwise, sized for the worst case so nothing is read back), FP8 with power-of-two per-128 scales (DeepGEMM's recipe), two grouped GEMMs, weighted combine in ascending expert order → residual. Head: final RMSNorm of the wanted rows → BF16 → `lm_head` BF16 GEMM, f32 logits.
- **Weights** (`weights.rs`): copied as stored; only scale grids are re-laid-out. The fused QKV stays rank-major with each chunk padded to whole 128-row tiles, which turns the checkpoint's per-chunk scale tiling into an ordinary grid; dense gate/up are stacked; experts' gate/up rows are stacked per expert and their E8M0 scales packed into DeepGEMM's SFB words.
- **Chain acceptance takes a host plan.** `Sampler::chain_accept` takes each row's drafts and distribution rows as host values (`AcceptRow`), bounds every index the kernel derives from them (k drafts plus the bonus token within the row's output, target and draft rows within the distributions and within `u32`, draft ids within the vocabulary), and only then uploads them into `AcceptInputs`, which nothing else writes. A drafter that produces its drafts on the device would need a kernel-side bound and status flag for the ids instead.
- **No drafting yet.** `max_draft_tokens` is 0. The device sampler's draft draws and chain acceptance exist and are bit-exact; the drafter rows (MTP layers with boundary taps in a drafter KV group, the draft chain inside one step) are the next piece.
- **Eager.** Inputs are uploaded per step and attention work lists planned on the host; the step structure (fixed-size scratch, plans as device buffers) is meant to become graph replay later.

## Numerics

Bit equality with the f32 reference is not a goal for the kernels; the GPU's divergence must be explained by its quantization recipe, and is measured, not assumed:

- **Sampler**: bit-identical to `eidola-engine::sampling` (probabilities and tokens, every filter, every stream, chain acceptance) given equal logits.
- **Kernels** (synthetic inputs vs host references over the same quantized operands): RMSNorm 1 BF16 ulp; FP8 GEMM BF16 output rounding (3.9e-3 relative); BF16 GEMM ≤ 5.6 f32 ulp of Σ|terms|; DeepGEMM 3.9e-3 relative (BF16 out); attention ≤ 2.7e-3 absolute on values in [-1, 1] (FA2 rounds probabilities to BF16).
- **Model** (truncated real checkpoint, layers 0/1/2/5, 256 real tokens): residual stream relative L2 2.5e-3 → 1.2e-2 across the four layers; logits top-1 96.9 %, KL mean 8.6e-4. The f32 reference with the executor's quantization points emulated (`tests/common/qref.rs`) diverges from the f32 reference by the same amount (top-1 96.9 %, KL 8.9e-4), and the GPU sits within KL 2.2e-4 of that emulation: the divergence is the activation-quantization recipe. Chunked and unchunked prefill agree to KL 2.6e-5; decode steps agree with the reference (8/8 top-1, KL 4e-4).
- **Full Flash** (all 48 layers, the same 256 tokens): logits top-1 99.6 %, KL mean 5.8e-3 against f32; the quantization emulation alone gives 99.2 % / 4.7e-3, and the GPU is within KL 1.1e-3 of it. Against the vendor serving path (SGLang TP1 with DeepGEMM MoE and FA4 attention, same box, same token ids) on the model's own continuations: GPU vs f32 98.9 % top-1, KL 2.1e-3; SGLang vs f32 99.2 %, KL 4.6e-3; GPU vs SGLang 98.8 %, KL ≈ 5e-3, so no further from SGLang than SGLang is from f32. Quality is equal: GSM8K (first 200, no thinking, greedy) 193/200 for both, a 40-case tool-call suite 39/40 for both.
- **Not batch-invariant.** A row's logits depend slightly on what else is in the step (attention's query tile is chosen per step; the expert layout depends on the batch): max |Δlogit| 0.052 measured for a decode row alone vs beside a 30-token prefill. Resuming from copied blocks in a fresh slot reproduces the original logits bit for bit. Batch-invariant kernels (deterministic serving across batch compositions) are an open question for later.

## Tests

`cargo test -p eidola-engine-cuda` runs everywhere: on a machine without a device or without `EIDOLA_ENGINE_KERNELS_DIR` the GPU tests print why and pass. Every kernel correctness test runs on the device's exact cubin and on the `sm_100f` family cubin (the binary a B200-class part would run).

| File | What it proves |
|---|---|
| `smoke.rs` | Images load, every entry and launch contract resolves, foreign images are refused, RMSNorm. |
| `kv.rs` | Zero/Copy touch exactly their blocks, `ResetSlot` empties a slot's tables and scrubs only its state, shared writes and unmapped reads panic. |
| `sampler.rs` | Probabilities and tokens bit for bit, draws over 512 positions, chain acceptance, non-finite logits reported. |
| `cutlass_params.rs` | The Rust Params equal CUTLASS's (recorded descriptors without a GPU, driver-encoded with one). |
| `gemm.rs`, `moe_gemm.rs`, `attention.rs` | Each upstream kernel against a host reference. |
| `engine.rs` | The serving core's `Engine` over this executor on the truncated checkpoint: chunked prefill, prefix hits and salt isolation, preemption under KV pressure, cancellation, a randomized mix; KV invariants and "every free block is zero on the device or has a zero queued" after every step; greedy outputs within 1.5 logits of the f32 reference's argmax. |
| `real_flash.rs`, `executor.rs` | The truncated checkpoint (`EIDOLA_MIMO_LAYERS=all` runs `real_flash.rs` on all 48 layers; `EIDOLA_IMAGES=exact` skips the family image): layer diffs and logits (unchunked, chunked, decode) against the f32 reference and the quantization emulation; seam behaviour (zero, copy-and-resume, batch effect, contract panics). Need `EIDOLA_MIMO_DIR` (checkpoint directory, whole or truncated) and, for `real_flash.rs`, `EIDOLA_MIMO_GOLDEN` (the model crate's `hf-golden.safetensors`, for real token ids). |

```sh
EIDOLA_ENGINE_KERNELS_DIR=<build output> EIDOLA_MIMO_DIR=<checkpoint> EIDOLA_MIMO_GOLDEN=<golden> \
  cargo test --release -p eidola-engine-cuda -- --nocapture --test-threads 1
cargo run --release -p eidola-engine-cuda --example smoke -- <kernel build output>
```

`examples/eval.rs` drives a whole checkpoint through the engine for quality comparisons with another engine on identical token ids: `render` (chat template and tokenizer from `eidola-engine-chat`), `generate` (greedy, to EOS), `logprobs` (top-20 per position, from the GPU or, with `--reference`, the f32 forward), `score` (GSM8K answers; tool calls parsed with the engine's own parser) and `compare` (top-1, top-20 KL). Its module docs give the file formats; datasets and the other engine are the operator's.

## Measured on the B300 (driver 580.173.02, CUDA 13.0 driver API)

- `NVIDIA B300 SXM6 AC`, CC 10.3, **148 SMs**, 232,448 bytes opt-in shared memory per block, 287.4 GB device memory.
- The CUDA 13.2-built cubins load on the 13.0 driver (SASS only; minor-version compatibility).
- 256-token prefill of the 4-layer truncation, eager: ~100 ms. Sampler with top-p: ~24 ms for 110 rows.

## DeepGEMM SM count

DeepGEMM's persistent scheduler takes the SM count as a template argument and uses it as the grid stride, so an instance is correct only when launched with exactly that many blocks on a part with at least that many SMs. The built instances use 148, which equals the B300's count. `ops::deepgemm_grid` launches with the instance's count (`DEEPGEMM_INSTANCE_SMS`, checked against the template argument in every manifest symbol by a unit test) and refuses a device with fewer SMs; a part with more runs correctly on 148 of them. A part with a different count that needs every SM gets its own instance in the kernel build.
