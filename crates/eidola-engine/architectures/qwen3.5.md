# Qwen3.5-2B and Qwen3.5-0.8B

`Qwen/Qwen3.5-2B` and `Qwen/Qwen3.5-0.8B` (Apache-2.0, BF16), text only, on the CPU executor. The architectures are `eidola_common::architecture::Architecture::Qwen35Dense2B` and `Qwen35Dense08B`, one file for both: they share every line of code, differing only in their dimensions.

**Status: served on the CPU executor, as a GPU-free proof of concept.** The reference forward, the chat format, the CPU executor and the node are implemented and verified; there is no CUDA executor for them (a `cuda` node is refused before the device), no catalog row and no engine deployment.

## The model

Qwen3.8-27B's architecture (`qwen3.8-27b.md` → The model; Transformers' `qwen3_5`), smaller, in BF16, with a tied head:

| | Qwen3.5-2B | Qwen3.5-0.8B | Qwen3.8-27B |
|---|---|---|---|
| layers (DeltaNet + attention) | 24 (18 + 6) | 24 (18 + 6) | 64 (48 + 16) |
| hidden, intermediate | 2,048, 6,144 | 1,024, 3,584 | 5,120, 17,408 |
| attention query and KV heads (of 256) | 8, 2 | 8, 2 | 24, 4 |
| DeltaNet key and value heads (of 128) | 16, 16 | 16, 16 | 16, 48 |
| weights | BF16 | BF16 | FP8 E4M3, 128×128 blocks |
| head | tied to the embedding | tied | its own |
| MTP layers | 1 (shipped) | 1 (shipped) | 1 |

The same: three DeltaNet layers then one attention layer, the gated attention (head 256, sigmoid output gate, rotary on the first 64 dimensions, interleaved M-RoPE `[11, 11, 10]`), the DeltaNet's convolution of width 4 and f32 recurrent state, the 248,320-row vocabulary, 262,144 positions. The KV is 12 KiB a position over the six attention layers; the recurrent state is 18 layers of 16 × 128 × 128 f32, about 19 MB a sequence. The checkpoints store `A_log` and the DeltaNet's gated-norm weight in F32, held in BF16 as the released stacks hold them (`eidola-engine-model/AGENTS.md` → Qwen3.5). The vision tower is in the checkpoint and never read.

## How it is verified

1. **The reference forward** (`eidola-engine-model`) against the released modelling code with its unspecified operations pinned, as Qwen3.8's: exact on two committed BF16 fixtures (one tied) and on the whole of both released models; healthy on both (`eidola-engine-model/AGENTS.md` → Qwen3.5).
2. **The CPU executor** (`eidola-engine-cpu`) equal to that forward bit for bit through the real scheduler: every Qwen test on the BF16 tied fixture as on Qwen3.8's, and on the released checkpoints (`tests/qwen_real.rs`; `eidola-engine-cpu/AGENTS.md` → Qwen3.8).
3. **The chat format** (`eidola-engine-chat` → Qwen3.5): its own template and tokenizer, pinned, byte-exact against `apply_chat_template` over its own corpus.
4. **The node** (`eidola-server-engine` → Qwen3.5): its dev model served end to end, and the released 2B booted on the CPU executor answering correctly.

## The prefix across requests

The template writes no reasoning block for an assistant turn before the last user message, so a new user message's prompt does not extend the last request's: it shares the user turn's first request up to its generation header. Within a tool loop the next prompt does extend the last one. What that means for a client's declared cached prefix, and the one refusal it brings, is `eidola-engine-chat/AGENTS.md` → Qwen3.5 → What that means for a declared cached prefix.

## Where the code lives

The Qwen code of every crate (`qwen3.8-27b.md` → Where the code will live), extended rather than copied:

| Crate | Qwen3.5 code |
|---|---|
| `eidola-common` | `architecture.rs` (`Qwen35Dense2B`, `Qwen35Dense08B`, `QWEN35_2B_DIMS`, `QWEN35_08B_DIMS` and their `Shape`s); `weights_manifest` (`QWEN3_5_REQUIRED_FILES`) |
| `eidola-engine-model` | `src/qwen/config.rs` (`WeightFormat`, the tied head, the Qwen3.5 keys), `linear.rs` (`Linear`, `linear_rows`), `weights.rs` (the tied head, F32 small tensors held in BF16); `tests/fixtures/qwen_tiny_bf16/`, `tests/fixtures/qwen_tiny_bf16_tied/`; `tests/data/qwen3.5-*.config.json`; `golden/qwen/` (`--qwen35`, `--model`, `upstream-qwen3.5-*-sha256.json`) |
| `eidola-engine-chat` | `ChatFormat::Qwen3_5`; `src/qwen/template.rs` (`TemplateKind`, Python's `split`, `lstrip`, `rstrip` and `str()`); `OutputConfig::qwen3_5`; the tokenizer's pin and end-of-sequence rule; `tests/fixtures/qwen35/`; the Qwen tests over both corpora |
| `eidola-engine-cpu` | `src/qwen/executor.rs` (`linear_rows`); `tests/qwen_engine.rs` and `tests/qwen_memory.rs` on both fixtures; `tests/qwen_real.rs` |
| `eidola-engine-cuda` | `qwen::check_supported` (refusing BF16 and a tied head) |
| `eidola-server-engine` | `model.rs` (`qwen_architecture`, `qwen_dims`, the Qwen3.5 arm of the loader), `pipeline.rs` (`render_qwen` by template kind), `cuda.rs` (the refusal); `tests/qwen35.rs`, `qwen35_is_refused_before_the_device` in `tests/cuda.rs`, the dev model in `tests/common/fixture.rs` |

## Outstanding work

- a CUDA executor for BF16 projections and a tied head, if these are to be served on a GPU;
- a catalog row and a deployment;
- a client rule declaring a new user turn's cached prefix from the turn's first request, for a template that drops earlier reasoning;
- the reference as released on a GPU, and the drafter against vLLM's own.
