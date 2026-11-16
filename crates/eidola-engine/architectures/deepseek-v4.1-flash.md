# DeepSeek-V4.1-Flash

`deepseek-ai/DeepSeek-V4.1-Flash-1017`, text only, on two Blackwell GPUs. The architecture is `eidola_common::architecture::Architecture::DeepSeekV41Flash`.

**Status: implemented and verified for correctness; not deployed, not in the catalog, and not tuned for speed.** The engine computes the released model exactly, on the CPU and on two B300s, and its quality is that of the reference serving stacks. It is 25–70× slower than vLLM on the same hardware. Speed work is deferred until demand justifies it; the work it needs is listed under [Outstanding work](#outstanding-work). If the architecture is still unused when that decision is taken, the [Where the code lives](#where-the-code-lives) list is also the list to prune.

## The model

- **One 40-layer stack.** Layers 0–19 are the "encoder" and run on device 0; layers 20–39 are the "decoder" and run on device 1.
- **Residual streams.** mHC: four residual streams mixed by Sinkhorn.
- **Attention.** Low-rank, with one 512-dim KV head and a 128-token sliding window, plus CSA2 global KV: ratio-2 pooled sources at layers 2, 8 and 14, a ratio-1 source at layer 20 with a candidate pool, index keys and a top-512 selection.
- **Engram.** Hashed 2/3/4-gram lookups at layers 1 and 14.
- **MoE.** 384 routed FP4 experts and one shared expert.
- **Drafter.** DSpark: three blocks and five draft slots.
- **Precision.** BF16 activations with runtime FP8/FP4 activation quantisation.
- **Size.** The checkpoint is 510 GB. Its weights hash, by this repository's manifest, is `dc8555f4d34008c1301e0ffb943b68196ef2a4e4ec8ed04fdfa6ad45d733d4d5`.
- **Device split.** Device 0 holds the embedding, layers 0–19, the layer-1 Engram table and a layer-20 stub (about 251 GB). Device 1 holds layers 20–39, the head, the layer-14 Engram table and the drafter (about 260 GB).

**Decisions that differ from the reference stacks:**

- **Exact triangular decoder replay.** On a prefix-cache hit, decoder layer `20 + j` recomputes positions `t − 128 − (19 − j)·127 … t`. That is bit-identical to the full forward, so cached and uncached requests give the same output. Shipped stacks replay a 128–140-token rectangle instead and say it is approximate.
- **Deterministic rules where the reference leaves the result unspecified:** exact 32-wide block dot products rounded once, total-order top-k (score, then lower index), and one canonical attention slot layout.
- **Agreement with the reference** is judged three ways:
  - exact against the reference with those operations pinned;
  - no further from the reference as it runs on a GPU than the reference is from itself;
  - a health check that both must pass (a public reply reproduced token for token, and a next-token accuracy floor).

  The reference's own execution paths agree on the top token only about 94 % of the time at full depth, so a fixed 99.9 % bar is not attainable even by the reference.

## How it is verified

Each level is held bit for bit to the one before it:

1. **The reference forward** (`eidola-engine-model`) is exact against DeepSeek's released code with the unspecified operations pinned. That holds on a synthetic fixture, on a 12-layer truncation of the real weights and on the first four real layers whole.
2. **At full depth on the released model** (555 tokens), ours against the reference on a GPU agrees on the top token 94.1–95.7 % of the time with KL 0.023–0.024. The reference against itself gives 94.4 %. Both pass the health check.
3. **The CPU executor** (`eidola-engine-cpu`) equals that dense forward bit for bit through the real scheduler. That covers chunking, prefix hits, preemption, drafting at widths 0–5, and window pools far smaller than a conversation.
4. **The CUDA executor** (`eidola-engine-cuda`) equals the CPU executor bit for bit:
   - on tiny models, including drafting, the host tier and scrubs;
   - on the released weights one half at a time;
   - on the whole released model on one GPU with staged experts (development only);
   - resident on two B300s in one forward;
   - with the exact tensor-core ("banded") GEMMs as well as the scalar ones.
5. **Through the node, on two B300s**, greedy tokens equal the CPU executor's at draft widths 0 and 3.
6. **Against vLLM 0.33 (TP2) on the same machine and checkpoint**, quality is within noise: next-token accuracy 0.659 / 0.666 against vLLM's 0.663 / 0.657, and 94–96 % top-1 agreement.

Each crate's `AGENTS.md` lists its suites and the commands that run the GPU and checkpoint-scale ones. The session scripts are `crates/eidola-engine-cuda/scripts/deepseek_gpu_session.sh` (one GPU) and `deepseek_two_gpu_session.sh` (two GPUs, the node and the vLLM yardstick).

## Where the code lives

| Crate | DeepSeek-V4.1 code |
|---|---|
| `eidola-common` | `architecture.rs` (`DeepSeekV41Flash`, `DEEPSEEK_V41_FLASH`: device split, KV groups, step scratch); `engine_deployment` (two-device grammar, per-device KV minimums and reserve); `weights_manifest` (required files) |
| `eidola-engine` | the seam's DeepSeek shapes (`spec.rs`: pooled groups, `coverage: Chain`, `window_chain`, `token_history`); `mock/chain.rs`; `tests/window_chain.rs` |
| `eidola-engine-model` | `src/deepseek/`; `tests/deepseek_*`; `tests/fixtures/deepseek_tiny/`; `golden/deepseek/` (pinned reference tooling, upstream hashes, health check) |
| `eidola-engine-chat` | `src/deepseek/` (encoder, output parser, Engram vocabulary map); `tests/deepseek_*`; `tests/fixtures/deepseek/`; `dev/deepseek_*` |
| `eidola-engine-cpu` | `src/deepseek/`; `tests/deepseek_*` |
| `eidola-engine-kernels` | `csrc/ds_*.cu{,h}` and `kernels.json` entries; `oracle/ds_*.cc`; `tests/deepseek_*`; `tests/data/ds_*` |
| `eidola-engine-cuda` | `src/deepseek/`; `oracle/ds_launch_host.cc`; `tests/deepseek_*`; `examples/deepseek_*` (halves, whole model staged, two GPUs, GEMM bench); `scripts/deepseek_*` |
| `eidola-server-engine` | the DeepSeek arms in `lib.rs`, `cuda.rs`, `model.rs`, `pipeline.rs`; `tests/deepseek*.rs`; `examples/deepseek_generate/` |

## Outstanding work

**Speed.** Measured on 2× B300 with 1,024 tokens in and 256 out:

| | Ours | vLLM 0.33 |
|---|---|---|
| Concurrency 1, decode | 8.2 tok/s | 213 tok/s |
| Concurrency 1, time to first token | 5.1 s | 0.25 s |
| Concurrency 32, aggregate output | 32.5 tok/s | 2,339 tok/s |

The time goes to, roughly in order:

1. **Scalar exact attention and indexer kernels.** Tensor-core forms are needed, held to a per-kernel tolerance against the exact form, which stays the verification mode.
2. **The GEMMs.** The banded tensor-core GEMMs are exact and 10–30× faster than the scalar ones, but decode is latency-bound at 1–12 % of memory bandwidth (profiled with `ncu`). It needs a decode-specific structure.
3. **No CUDA decode graphs.** Graphs are refused for this architecture today.
4. **The two devices alternate instead of overlapping.** Pipelining needs the executor seam split into submit and collect.

Also outstanding:

- **Deployment.** No catalog row, no `deploy/engine/` deployment, and no pin. Before any of these:
  - choose a price and a maximum output length;
  - build and measure the node image;
  - cover two GPUs in confidential mode (encrypted NVLink or host staging, and attestation of both GPUs).
- **Decisions open for a product:**
  - the node always renders thinking on at effort `high`, because the request subset carries no thinking switch or `reasoning_effort`;
  - answers are never continuable, while the client continues bounded output;
  - draft ids and boundary activations cross devices through the VM's host memory.
- **Not exercised on rented hardware:** the executor's memory scenarios on two physical GPUs (the containers' memory-lock limit was 8 MiB).
- **Not implemented:** vision.
