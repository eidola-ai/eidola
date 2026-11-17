# Qwen3.8-27B

`Qwen/Qwen3.8-27B-FP8`, text only, on one Blackwell GPU. The architecture is `eidola_common::architecture::Architecture::Qwen38Dense27B`.

**Status: in progress.** The reference forward is implemented and verified on the CPU, and the serving core carries the recurrent state (below) and its mock runs it; nothing else is implemented. The variant is a stub: its shape is `unimplemented!`, and no catalog row or deployment may name it.

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
2. **The CPU executor** (`eidola-engine-cpu`) equal to that forward bit for bit through the real scheduler: chunk sizes, interleavings, prefix hits at recurrent-state snapshots, preemption, and drafts accepted and rejected.
3. **The CUDA executor** (`eidola-engine-cuda`) held to the CPU executor, with greedy output identical whatever shares the batch.

The Gated DeltaNet can be computed token by token or in chunks of tokens, and the two orders round differently. The contract is the token recurrence: every token, in a prefill, a decode step or a draft check, runs one step from the state the token before it left, so a layer's state and output depend only on the token prefix, and a state saved after any token resumes exactly. The reference forward exposes the step as public kernels the executors call.

## The recurrent state in the serving core

The DeltaNet layers' state is a pool of state buffers beside the KV groups (`eidola-engine/AGENTS.md` → Recurrent state): a sequence keeps a base buffer at its last block boundary and its executor slot the inputs after it; drafts never reach a stored state; and the prefix cache keeps snapshots only at a sequence's branch, regeneration and continuation points, with every guarantee its KV blocks get. `mock::qwen_like_spec` is the shape the serving core's tests run (`tests/recurrent_state.rs`).

## Where the code will live

| Crate | Qwen3.8 code |
|---|---|
| `eidola-common` | `architecture.rs` (`Qwen38Dense27B`, a stub); then its `Shape`, `engine_deployment` (one device) and `weights_manifest` (required files) |
| `eidola-engine` | the seam's recurrent state (`spec.rs` `StateSpec`, `executor.rs` → Recurrent state) and its snapshots in the prefix cache (`kv.rs`); `mock::qwen_like_spec` and `mock/recurrent.rs`; `tests/recurrent_state.rs` |
| `eidola-engine-model` | `src/qwen/`; `tests/qwen_*` (`qwen_tiny`, `qwen_config`, `qwen_real`, `qwen_health`, `qwen_upstream/`); `tests/fixtures/qwen_tiny/`; `tests/data/qwen3.8-27b.config.json`; `golden/qwen/` (pinned reference tooling, upstream hashes, health check, the whole-model session) |
| `eidola-engine-chat` | `src/qwen/` (the template, `<think>` reasoning and `<tool_call>` parsers); `tests/qwen_*`; `tests/fixtures/qwen/`; `dev/qwen_*` |
| `eidola-engine-cpu` | `src/qwen/`; `tests/qwen_*` |
| `eidola-engine-kernels` | `csrc/qwen_*` and their `kernels.json` entries; `oracle/qwen_*`; `tests/qwen_*` |
| `eidola-engine-cuda` | `src/qwen/`; `tests/qwen_*`; `examples/qwen_*`; `scripts/qwen_*` |
| `eidola-server-engine` | the Qwen arms in `lib.rs`, `cuda.rs`, `model.rs`, `pipeline.rs`; `tests/qwen*.rs` |
| `models/`, `deploy/engine/` | a catalog row and a pinned one-GPU deployment, once the engine serves it |

## Outstanding work

- the reference forward against the reference as released on a GPU, and its drafter against vLLM's (prepared: `golden/qwen/full_session.sh`);
- the chat template, rendered byte for byte, and the reasoning and tool-call parsers;
- MTP drafting: the drafter's rows and the canonical rewrite of its KV at accepted positions (the serving core already keeps the recurrent state across rejected drafts);
- the CPU executor;
- ahead-of-time kernels for the DeltaNet recurrence, the short convolution and gated attention, and the CUDA executor;
- the architecture's `Shape`, the node, the catalog row and the deployment;
- speed measured against a reference serving stack on the same GPU.
