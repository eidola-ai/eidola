# MiMo-V2.6-Flash

`XiaomiMiMo/MiMo-V2.6-Flash` on one Blackwell GPU. The architecture is `eidola_common::architecture::Architecture::MimoV26Flash`.

**Status: in the catalog** (`models/mimo/mimo-v2.6-flash.toml`) and pinned to a committed one-B300 deployment (`deploy/engine/mimo-v2.6-flash/b300x1/`) that is not running.

## The model

- A mixture-of-experts model with global-attention and sliding-window layers and attention sinks.
- FP8 dense weights and MXFP4 experts.
- Up to three MTP drafting layers.
- Each crate's `AGENTS.md` describes its part in detail.

## How it is verified

1. **The reference forward** (`eidola-engine-model`) against the Hugging Face model code, on synthetic fixtures and a truncation of the real weights. Its acceptance is top-1 ≥ 99.9 % and mean KL < 1e-3; it is exact on the fixture.
2. **The CPU executor** (`eidola-engine-cpu`) reproduces that forward bit for bit through the real scheduler.
3. **The CUDA executor** (`eidola-engine-cuda`) runs tensor-core kernels (CUTLASS, DeepGEMM, FlashInfer FA2) and is held to the CPU executor within a documented quantisation tolerance. Its greedy output is batch-invariant and identical at every draft width.

## Where the code lives

| Crate | MiMo-V2.6 code |
|---|---|
| `eidola-common` | `architecture.rs` (`MimoV26Flash`); `engine_deployment` (one-device grammar) |
| `eidola-engine` | the seam's MiMo shapes (global, sliding and drafter groups, MTP and block drafting); the MiMo-shaped mock in `mock.rs` (`mimo_like_spec`) |
| `eidola-engine-model` | `src/mimo/`; `tests/mimo_*`; `tests/fixtures/mimo_tiny/`; `tests/data/`; `golden/mimo/` |
| `eidola-engine-chat` | `src/mimo/` (the Jinja template, `<think>` reasoning and `<tool_call>` parsers); `tests/mimo_*`; `tests/fixtures/mimo/`; `dev/mimo_*` |
| `eidola-engine-cpu` | `src/mimo/`; `tests/mimo_*` |
| `eidola-engine-kernels` | `csrc/` (outside `ds_*`), the CUTLASS, DeepGEMM and FlashInfer instantiations and `engine_ops.cu` |
| `eidola-engine-cuda` | `src/mimo/`; `oracle/cutlass_params.cu`; `tests/mimo_*`; `tests/data/cutlass_params.jsonl`; `examples/mimo_*` |
| `eidola-server-engine` | the MiMo arms in `lib.rs`, `cuda.rs`, `model.rs`, `pipeline.rs` |
| `models/mimo/`, `deploy/engine/mimo-v2.6-flash/` | the catalog row and the pinned deployment |

## Outstanding work

Each crate's `AGENTS.md` lists the open items; the deployment is committed but not running.
