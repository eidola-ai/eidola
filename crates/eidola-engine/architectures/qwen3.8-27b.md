# Qwen3.8-27B

`Qwen/Qwen3.8-27B-FP8`, text only, on one Blackwell GPU. The architecture is `eidola_common::architecture::Architecture::Qwen38Dense27B`.

**Status: in progress.** The reference forward and the chat format are implemented and verified; the serving core carries the recurrent state (below); the CPU executor reproduces the forward bit for bit through the real scheduler, drafting included; the CUDA executor and its kernels are built (decode graphs per rung, FA2 at head dimension 256 in the fast profile, the FP8 GEMM swapped at decode sizes, sampling and chain acceptance on the device), equal to the CPU executor on host-compiled kernel bodies eagerly and on the padded rung path, not yet run on a GPU; and the inference node serves the architecture (below). The variant is implemented: its `Shape` is known, so a catalog row and a deployment may name it. None is committed.

## The model

- **A dense hybrid of 64 layers.** Three Gated DeltaNet layers then one gated full-attention layer, sixteen times over; every layer has a SwiGLU feed-forward.
- **Gated DeltaNet.** Linear attention with a fixed-size recurrent state per sequence: 16 query/key heads and 48 value heads of 128 dimensions, a causal depthwise convolution of width 4 over the projected queries, keys and values, a per-head decay and write strength, and an output RMS norm gated by SiLU. Its state is an FP32 128 × 128 matrix per value head, about 151 MB a sequence over the 48 layers, whatever the context length.
- **Gated full attention.** 24 query heads and 4 KV heads of 256 dimensions, per-head RMS norms on queries and keys, rotary position on the first 64 dimensions of each head, and a sigmoid gate on the attention output. KV is 64 KiB a position.
- **One MTP drafting layer** of the full-attention kind, sharing the embedding and the output head.
- **Precision.** FP8 E4M3 weights with a BF16 scale per 128 × 128 block for the large projections; the embedding, the head, the norms, the DeltaNet's small projections, its convolution and the drafter's input projection are BF16. Activations are quantised to FP8 per token and 128 columns at runtime.
- **Not served: vision.** The checkpoint carries a vision encoder, which the text path never reads.

## How it will be verified

Each level is held to the one before it, as for the other architectures:

1. **The reference forward** (`eidola-engine-model`), an f32 forward with the released model's rounding points, against the released modelling code (Transformers' `qwen3_5`) with its unspecified operations pinned, and a health check on the real checkpoint: a next-token accuracy floor over the checkpoint's own licence text and a known one-word answer. **Done on the CPU:** exact on a synthetic fixture, on the first four real layers and on the whole released model; healthy (accuracy 0.998 on the licence text, as vLLM measured, and "Paris"). Still to run on a GPU: the reference as released there, against which ours must be no further than the reference is from itself, and the drafter against vLLM's own (`eidola-engine-model/AGENTS.md` → Qwen3.8).
2. **The CPU executor** (`eidola-engine-cpu`) equal to that forward bit for bit through the real scheduler: chunk sizes, interleavings, prefix hits at recurrent-state snapshots, preemption, the host tier, and drafts accepted and rejected. **Done on the committed fixture**: every row's logits, every drafter and chain row, every draw and every stored state exact (`eidola-engine-cpu/AGENTS.md` → Qwen3.8).
3. **The CUDA executor** (`eidola-engine-cuda`) held to the CPU executor, in two profiles. The **exact** profile (the verification mode) equals the CPU executor bit for bit; the **fast** profile (the serving mode: the FP8 and BF16 linears on tensor cores, the DeltaNet's `a` and `b` projections as one small-N GEMM over their stacked weight, the gates in f32) is held to reproducibility between calls (greedy tokens and logits the same whatever the batch, the chunking, decode or prefill, a cache hit or not, and the draft width), asserted, and its distance from the exact profile measured. **Done without a GPU**: every kernel body equal to the model crate, and the exact profile equal to the CPU executor through the real scheduler (logits, tokens, state images byte for byte, encryption, the host tier) on those bodies compiled for the host, eagerly and on a decode rung's padded launch sequence, and invariant on both; the fast profile's plans packed on every path (`eidola-engine-cuda/AGENTS.md` → Qwen3.8). Prepared for a GPU: the same tests on the device with graphs replaying, the fast profile's invariance across paths on a fixture with the released 256-wide heads, the released model's comparison, health, generation, and serving speed at the reference stack's workloads (`scripts/qwen_*_session.sh`).

The Gated DeltaNet can be computed token by token or in chunks of tokens, and the two orders round differently. The contract is the token recurrence: every token, in a prefill, a decode step or a draft check, runs one step from the state the token before it left, so a layer's state and output depend only on the token prefix, and a state saved after any token resumes exactly. The reference forward exposes the step as public kernels the executors call.

## The recurrent state in the serving core

The DeltaNet layers' state is a pool of state buffers beside the KV groups (`eidola-engine/AGENTS.md` → Recurrent state): a sequence keeps a base buffer at its last block boundary and its executor slot the inputs after it; drafts never reach a stored state; and the prefix cache keeps snapshots only at a sequence's branch, regeneration and continuation points, with every guarantee its KV blocks get. `mock::qwen_like_spec` is the shape the serving core's tests run (`tests/recurrent_state.rs`).

## The node

`eidola-server-engine` serves it as it serves the other two (`eidola-server-engine/AGENTS.md` → Qwen3.8-27B): the model's own template at its defaults (thinking on, the template's default effort, every past turn's reasoning kept, which is what keeps a new turn's prompt an extension of the last one's), no continuation, every tokenizer id sampleable. On `cuda` it runs the CUDA executor in the **fast** profile (FA2 attention) with the live state strategy and decode graphs as configured, drafting up to three tokens from the MTP layer. Its device memory is a state pool of `1 + S × (2 + P)` buffers of about 157 MB (`S` seats, `P` = 3 retention points with the prefix cache) beside one count of KV blocks for the target's and the drafter's groups; its reserve counts each seat (about 193 MB) and the executor's fixed step scratch by its own allocations: about 404 KB per step row, FA2's partial rows (about 12 bytes per step row and context position) and the step tables, whose page lists grow with the context. `eidola_common::architecture::QWEN38_DENSE_27B` and its `HybridDims` carry the numbers, and the node's tests hold them to the executor.

## Where the code lives

| Crate | Qwen3.8 code |
|---|---|
| `eidola-common` | `architecture.rs` (`Qwen38Dense27B`, `QWEN38_DENSE_27B`, `QWEN38_DIMS` and `HybridDims`); `engine_deployment.rs` (`recurrent_*`, and the recurrent arms of `cuda_kv_min_bytes`, `cuda_kv_host_min_bytes`, `cuda_device_reserve_bytes` and `check_resources`); `weights_manifest.rs` (`required_files`) |
| `eidola-engine` | the seam's recurrent state (`spec.rs` `StateSpec`, `executor.rs` → Recurrent state) and its snapshots in the prefix cache (`kv.rs`); `mock::qwen_like_spec` and `mock/recurrent.rs`; `tests/recurrent_state.rs` |
| `eidola-engine-model` | `src/qwen/`; `tests/qwen_*` (`qwen_tiny`, `qwen_config`, `qwen_real`, `qwen_health`, `qwen_upstream/`); `tests/fixtures/qwen_tiny/`, `tests/fixtures/qwen_tiny256/`; `tests/data/qwen3.8-27b.config.json`; `golden/qwen/` (pinned reference tooling, upstream hashes, health check, the whole-model session) |
| `eidola-engine-chat` | `src/qwen/` (the template, options and validation); with MiMo, the shared `src/jinja.rs`, `src/reasoning.rs` and `src/tool_call/xml.rs`; `tests/qwen_*`; `tests/fixtures/qwen/`; `dev/qwen_*` |
| `eidola-engine-cpu` | `src/qwen/` (`QwenCpuExecutor`, its state buffers and tails, `QwenDenseOracle`); `tests/qwen_engine.rs`, `tests/qwen_memory.rs`, `tests/qwen_common/` |
| `eidola-engine-kernels` | `csrc/qwen_abi.cuh`, `csrc/qwen_ops.cuh`, `csrc/qwen_ops.cu`, `csrc/qwen_ops_div.cu`, `csrc/qwen_deltanet.cu`, `csrc/qwen_fast.cuh`, `csrc/qwen_fast.cu`, `csrc/cutlass_fp8_blockwise_swap.cu`, `csrc/flashinfer_fa2_sink_paged_d256.cu` and their `kernels.json` entries; `tests/manifest.rs` (`qwen_*`) |
| `eidola-engine-cuda` | `src/qwen/` (`QwenCudaExecutor`, its step plan and program, its devices, kernel families and FA2 at d256); `oracle/qwen_launch_host.cc`; `tests/qwen_*` and `tests/qwen_common/`; `examples/qwen_whole.rs`, `examples/qwen_kernel_bench.rs`, `examples/qwen_serve_bench.rs`; `scripts/qwen_*` |
| `eidola-server-engine` | the Qwen arms in `lib.rs` (`qwen_cpu_config`), `cuda.rs` (`prepare_qwen`, `qwen_config`, `derive_qwen_pools`, `derive_qwen_host_pools`), `model.rs` (`Served::Qwen`), `pipeline.rs` (`render_qwen`, `qwen_options`); `tests/qwen.rs`, the Qwen tests in `tests/cuda.rs`, and the dev model in `tests/common/fixture.rs` |
| `models/`, `deploy/engine/` | a catalog row and a pinned one-GPU deployment, once the engine serves it |

## Outstanding work

- the reference forward against the reference as released on a GPU, and its drafter against vLLM's (prepared: `golden/qwen/full_session.sh`);
- the CUDA executor on a GPU (the prepared sessions, graphs and FA2 included), and tuning from the profiles. Built and held without a GPU, to measure on one (`qwen_perf_session.sh`'s `perf_tiny` and `perf_kernels`): the fast profile's FP8 GEMM swapped at decode sizes (its instances' agreement on a row's bits asserted on the device before the dispatch by row count may stand), the norms' loads issued at once, the small-N `a`/`b` projection, the gates in f32 and the quantiser at four rows a CTA. Then FA2 decode with shorter key chunks, the recurrence's decode, device-side page lists, a fused staging kernel and piecewise capture of prefill steps;
- the catalog row and the deployment;
- the CPU reference executor refuses to embed a sampled vision placeholder, which ends its engine, where the CUDA executor embeds its row;
- speed measured against a reference serving stack on the same GPU.
