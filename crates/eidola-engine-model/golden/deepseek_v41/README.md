# DeepSeek-V4.1 golden generation (dev-only)

Python tooling that produces the reference outputs the DeepSeek-V4.1 forward (`src/deepseek/`) is checked against. It is never a build input: `cargo test` reads only the committed fixture under `tests/fixtures/deepseek_tiny/` and needs neither Python nor the network.

## What the goldens are

**The committed goldens come from the released reference model code run over an emulated kernel layer, with its library arithmetic pinned. They do not come from the reference's own kernels**, which do not compile at the fixture's dimensions.

The reference is the inference code published with the checkpoint: `inference/model.py` and `inference/engram.py` of `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `08c8657f02291e11b6f6b63e0e16014f8e748518`. Those two files run unmodified. Nothing from that repository is vendored here: every script takes the directory holding them as `--reference-dir`.

Three things stand between that code and a CPU, and each is supplied by `ds_golden/`:

1. **The kernels.** The reference's `inference/kernel.py` is a set of tilelang CUDA kernels. `ds_golden/kernel_shim.py` has the same function signatures and computes what the kernel source specifies, in plain PyTorch on a CPU. Where the kernel source does not determine the bits (the accumulation order inside its GEMMs and reductions, the rounding of a float to FP8 or FP4, `exp`, `sigmoid`), the shim takes the rule this crate fixes (`AGENTS.md` → DeepSeek-V4.1 → Numerics contract).
2. **Library arithmetic the reference leaves unspecified.** `ds_golden/pinned_ops.py` replaces, in the `torch` namespace the reference reads them from, exactly the operations whose bits depend on the library or the call shape: the summation order of `F.linear`, `einsum`, `sum` and `mean`; the last bit of `exp`, `log`, `sin` and `cos`; whether a complex multiply is fused; and which of several equal values `topk` returns. Its module docstring lists each operation and its rule.
3. **Modules the text path never uses.** `image_processor` and `vision` are empty stand-ins (the real ones need Pillow).

With the kernels emulated and the arithmetic pinned, the reference's output is a function of the tokens and of how they are split into its prefill chunk and decode steps (its prefill and decode passes lay attention out in different 64-slot blocks), and the Rust forward reproduces it **bit for bit** for either. That exactness is what the committed tests assert. It is agreement between this crate and the reference's model code under this crate's rules. How far the real kernels and a GPU's own arithmetic are from those rules is a separate measurement, made once on a GPU (below, and `AGENTS.md`).

No workaround was needed for dtypes: PyTorch 2.10 on a CPU allocates, views, indexes and decodes `float8_e4m3fn`, `float8_e8m0fnu` and `float4_e2m1fn_x2` tensors, which is all the reference does with them outside the kernels.

## Setup

[uv](https://docs.astral.sh/uv/) and Python 3.12. This directory is its own project, with versions pinned in `pyproject.toml` and locked in `uv.lock` (torch from PyTorch's CPU-only index). It is separate from the MiMo tooling one level up, whose pins it does not move.

```sh
cd crates/eidola-engine-model/golden/deepseek_v41
uv sync
```

Get the reference code (`inference/model.py`, `inference/engram.py`, and `inference/kernel.py` for the GPU comparison) from the repository and revision above into a directory of its own, and pass that directory as `--reference-dir`.

## Synthetic fixture (committed)

```sh
uv run python make_synthetic.py --reference-dir <dir>
```

This rewrites `tests/fixtures/deepseek_tiny/`:

- `config.json`: the checkpoint's config shape at small sizes.
- `model.safetensors`: seeded random weights in the real storage formats. `ds_golden/synthetic.py` explains the shape choices: every mechanism is present and does real work within the 96-token sequence.
- `golden.safetensors`:
  - `tokens`, and `engram.token_map` (the compressed-vocabulary map the reference built from the fixture's stand-in tokenizer);
  - `step.*`: one token per call, the reference's decode pattern. Every layer's streams, each sub-layer's output, the final normed state, the logits, the Engram table rows, the drafter's input, and the drafter's greedy chain at every position from 1 on;
  - `prefill.*`: the whole sequence as one chunk. Streams, normed state and logits;
  - `short.logits`: a 40-token prefill chunk, shorter than the window, where the reference's prefill layout and the canonical one put entries in different blocks.

Output is deterministic down to the file bytes. After regenerating, run `cargo test -p eidola-engine-model`.

`--native-arithmetic` and `--native-topk` leave PyTorch's own arithmetic or `topk` in place, for measuring what the pinning changes. Write those runs elsewhere with `--out`; they are not fixtures.

## Truncated real checkpoint (local, not CI)

```sh
uv run python fetch_truncated.py --dir <checkpoint dir>                       # about 14 GB of HTTP range reads
uv run python real_golden.py --dir <checkpoint dir> --reference-dir <dir> --tokens 256     # pinned; about 15 minutes
uv run python real_golden.py --dir <checkpoint dir> --reference-dir <dir> --tokens 256 --native-arithmetic
cd ../../../.. && EIDOLA_DEEPSEEK_TRUNCATED_DIR=<checkpoint dir> \
    cargo test --release -p eidola-engine-model --test deepseek_real -- --ignored --nocapture
```

`fetch_truncated.py` fetches `config.json`, `tokenizer.json` and the tensors `ds_golden/truncated.py` keeps, re-packed byte for byte into one local file per upstream shard, and records every file's sha256 in `local-sha256.json`, which the Rust test verifies before loading. The truncation is:

- checkpoint layers 0, 1, 2, 3, 14, 15, 20, 21, 24, 37, 38 and 39, which cover every layer kind and the drafter's three target layers;
- the first 32 of each backbone layer's 384 routed experts and the first 16 of each draft block's 128. The router's rows are cut to match, so routing chooses among the kept experts. Every tensor is the checkpoint's own; the model is smaller than the released one, not an approximation of it;
- the embedding, the head, the final norm, both Engram projections and the three draft blocks.

The two Engram tables (about 98 GB each) are not fetched. `real_golden.py` hashes its tokens with the reference's own code, fetches only the rows they touch (two range reads per row) into `<dir>/golden/engram-rows.safetensors`, and gives the reference and the Rust test a table of just those rows. The reference's table class runs unmodified on the compact table; a forward hook maps its row ids. The host rate-limits that many small requests (about 24,000 for 256 tokens took 50 minutes), so the fetch backs off, appends rows to a partial file as they arrive and resumes from it. The recorded numbers use `--tokens 256` for that reason; without it the passage is 487 tokens.

`real_golden.py` tokenises a fixed passage of prose and code and writes `<dir>/golden/reference-pinned.safetensors`: the logits and every layer's streams for the passage as one prefill chunk, then a one-window prefill followed by one token per call with the drafter. With `--native-arithmetic` it writes `reference-native.safetensors` from the same model with PyTorch's own arithmetic and `topk`: the reference with its unspecified bits left to the library, which nothing bit-unequal to it reproduces. The Rust test reports that comparison and does not assert it (`AGENTS.md` → What is verified against what).

## Whole early layers (local, not CI)

The 12-layer truncation above is not a stable function of the bits the reference leaves unspecified (`AGENTS.md` → What is verified against what), and it cannot say whether that is the model or the truncation: it skips most layers and routes among 32 of 384 experts. The `prefix` truncation is the model's first four layers whole, every routed expert, so the residual streams after each layer are the full model's, and so is their sensitivity.

```sh
uv run python fetch_truncated.py --dir <prefix dir> --truncation prefix --reuse <checkpoint dir>   # about 32 GB
uv run python real_golden.py --dir <prefix dir> --reference-dir <dir> --truncation prefix --rows-from <checkpoint dir>
uv run python real_golden.py --dir <prefix dir> --reference-dir <dir> --truncation prefix --native-arithmetic
cd ../../../.. && EIDOLA_DEEPSEEK_PREFIX_DIR=<prefix dir> \
    cargo test --release -p eidola-engine-model --test deepseek_real_prefix -- --ignored --nocapture
```

`--reuse` copies the tensors another fetched truncation already holds instead of fetching them again, and `--rows-from` does the same for Engram rows. Four layers have no meaningful logits, so the goldens are every layer's streams and every token's routed experts for the whole passage (555 tokens) as one prefill chunk, plus the first 96 tokens one per call. The test equals the pinned golden exactly and prints, per layer, the relative error, the largest difference, the share of elements and positions that differ and the share of tokens whose routed experts are the same, for every other `golden/reference-*.safetensors` present and for the reference's prefill layout against its decode layout.

## The GPU session

`gpu_session.sh` is a whole session on a rented Linux machine with one NVIDIA GPU: it installs what it needs, runs the reference's real kernels, and leaves every result in one directory. Its first part has been run once (one B300, Ubuntu 24.04, torch 2.10.0 with CUDA 13 wheels, tilelang 0.1.12); the measured times below are from that run and the results are recorded in `AGENTS.md` → What is verified against what. The steps over the whole checkpoint have not been run.

On a fresh Ubuntu 24.04 machine with the NVIDIA driver and CUDA installed (root only for `apt-get install -y git curl build-essential`):

```sh
git clone <this repository> eidola && cd eidola/crates/eidola-engine-model/golden/deepseek_v41
./gpu_session.sh all          # setup, probes, kernels, fetch, real, summary
./gpu_session.sh rust         # our forward against the goldens, on that machine
```

or one step at a time, in this order:

| Step | What it does | Wall-clock |
|---|---|---|
| `setup` | Installs uv, a Python 3.12 venv with torch (CUDA wheels from `TORCH_INDEX`), tilelang 0.1.12, numpy, sympy, tokenizers; fetches the reference's `model.py`, `engram.py`, `kernel.py` at the pinned revision; checks that PyTorch sees the GPU and can hold the FP8 / FP4 dtypes on it | measured 13 s |
| `probes` | `gpu_probes.py`: the open numerics questions, on the real kernels (below) | measured 5 s |
| `kernels` | `compare_real_kernels.py --inputs generated`: every kernel once, real against emulated, on seeded inputs at the released model's dimensions | not yet run; under a minute expected |
| `fetch` | Both truncations by range reads into `SPREAD_DIR` and `PREFIX_DIR` (skipped per shard when present) | measured 318 s and 627 s |
| `real` | The reference with its real kernels over both truncations, unpinned (exactly what the released code computes on a GPU) and pinned, writing `reference-gpu-native.safetensors` and `reference-gpu-pinned.safetensors`. The unpinned runs also give every kernel call to the emulation and write `kernels-<truncation>.json` (not yet run with that comparison; it adds about a CPU run of the model) | measured, 12 layers: 89 s unpinned, 222 s pinned; whole layers 0–3: 17 s and 69 s |
| `summary` | `summary.txt` from the logs | seconds |
| `rust` | Makes the CPU goldens our forward must equal (`reference-pinned`, emulated kernels and pinned arithmetic) for both truncations if they are not there, installs the Rust toolchain of `rust-toolchain.toml`, and runs both ignored tests | not yet run in this form; estimate 30–60 min for the two CPU goldens on 24 vCPUs, 5–10 min install and build, 10–15 min of tests |

**The committed fixture cannot run on the real kernels.** tilelang 0.1.12 refuses to compile `sparse_attn` at the fixture's head dimension of 32 (`AssertionError: warp_col_tiles must be greater than 8, got 4`). So there is no end-to-end comparison of the real kernels against the committed goldens. What there is instead: the probes, the per-kernel comparison on generated inputs at dimensions the kernels accept (`kernels`), the per-kernel comparison on real activations (`real`), and the truncation goldens made with the real kernels under pinned arithmetic (`reference-gpu-pinned`), in which every difference from our forward is a kernel difference. `compare_real_kernels.py --inputs fixture` remains for `--kernels shim --device cpu`, which is how the script is checked without a GPU.

**Engram rows are the one slow part, and are avoidable.** A run needs two small range reads per table row, the host rate-limits them (about 25 minutes per 12,000 requests), and the two truncations need about 24,000 and 12,000 requests. Upload `<dir>/golden/engram-rows.safetensors` of each truncation from a machine that has them into `$SPREAD_DIR/golden/` and `$PREFIX_DIR/golden/` before `real`, and none are fetched. The whole pre-fetched directories can be uploaded the same way (`SPREAD_DIR=… PREFIX_DIR=…`), which skips `fetch` too.

A kernel output or a golden value that is not finite fails its step: the scripts count such values on their own, never as a zero difference, and exit non-zero after writing what they have. Everything is resumable: a step that finished leaves `$OUT/done/<step>` and is skipped on a rerun, and the fetches resume by shard and by row.

Results, all under `$OUT` (default `~/eidola-gpu/out`):

- `probes.json`, `kernels-generated.json`, `kernels-spread.json`, `kernels-prefix.json`;
- `goldens/spread/`, `goldens/prefix/`, `goldens/full/`: the `reference-*.safetensors` made there and the `engram-rows.safetensors` they were made with;
- `logs/<step>.log` and `summary.txt`.

`TORCH_INDEX` defaults to PyTorch's CUDA 13 wheel index and `TORCH_VERSION` to 2.10.0; set them if the machine's driver needs another build. `setup` fails if PyTorch does not see the GPU.

### What the probes answer

`gpu_probes.py` runs the real kernels on inputs built to decide what their source leaves open, and records what they did beside what the emulation does. The answers of the first session are in `AGENTS.md`:

- **float → FP8 cast**: all 252 exact ties between adjacent E4M3 codes (both signs), through the in-place path and the code path.
- **float → FP4 cast**: all 14 exact ties, with the power-of-two scale and with the E4M3 scale; the E4M3 scale's own rounding when `amax / 6` is an exact tie; and saturation when `amax / scale` exceeds 6.
- **`torch.topk` on CUDA over equal BF16 scores**: whether the lowest indices, the highest, or neither win among the entries tied at the threshold, and whether it is repeatable.
- **`sparse_attn` and block layout**: the same entries in the reference's decode layout, its prefill layout, packed, and with empty slots moved without changing any 64-slot block.

### Comparing GPU-made goldens with our forward

Put `goldens/spread/reference-gpu-*.safetensors` into `<12-layer dir>/golden/` and `goldens/prefix/reference-gpu-*.safetensors` into `<prefix dir>/golden/` and run the two ignored tests (on the GPU machine itself, `./gpu_session.sh rust` does it). Each test compares every `golden/reference-*.safetensors` it finds with our forward and with each other. `reference-pinned`, the CPU golden under our rules, is asserted exact; every other one is printed. **Without `reference-pinned` a test asserts nothing** beyond completing with finite values, and says so: it takes its tokens from whichever golden is there. Measured on a 16-core Apple-silicon Mac after the build, with four goldens in each directory: about 4½ minutes (`deepseek_real`) and about 80 seconds (`deepseek_real_prefix`), under 40 GB of memory. The 12-layer goldens must be made with the same `--tokens` as the pinned one they sit beside (256, the script's default).

## The whole model, one half at a time

The checkpoint (about 510 GB) does not fit one GPU, and the truncations cannot say how the final logits of the whole model respond to the bits the reference leaves open. `full_halves.py` runs the reference over all 40 layers on one GPU: the unmodified `model.py` built for layers 0–19, run, its boundary recorded (the residual copies and collapse weights leaving layer 19), freed; then built for layers 20–39 with the head and teacher-forced from that boundary by a forward pre-hook on its first layer (`ds_golden/halves.py`). Layer 20 is the KV source every later layer reads, so nothing else crosses. The Engram tables are never loaded: only the rows the tokens touch are read from their shards.

`check_halves.py` checks the driver on the synthetic fixture with the emulated kernels: the two halves reproduce the whole model bit for bit, in every layer's streams, the final normed state and the logits, for one prefill chunk, for one token per call and for a prefill chunk followed by single tokens. `full_halves.py --first-layers 4 --split 2` runs the same command line over the first four real layers, where its streams equal `real_golden.py`'s exactly and, with `EIDOLA_DEEPSEEK_FULL_FIRST_LAYERS=4` and `EIDOLA_DEEPSEEK_FULL_GOLDEN_DIR`, the Rust test equals its logits exactly in both layouts: the whole chain has run on real weights at four layers, not at forty.

It records final logits at every position for three call patterns over the same 555 tokens: `prefill` (one chunk), `decode` (one token per call) and `reshaped` (a prefill chunk of all but the last 64 tokens, then single tokens: with a chunk at least a window long the two layouts coincide past it, so against `prefill` only the shape of the calls differs; with a shorter sequence the chunk has a layout of its own, its length is recorded in the golden, and the test runs that layout). `tests/deepseek_real_full.rs` (`EIDOLA_DEEPSEEK_FULL_DIR`) runs our forward over the whole 40 layers twice, in the reference's prefill layout and in the canonical one, and prints top-1 and KL of ours against each pattern, of the reference's patterns against each other, and of every pair of goldens.

For a machine with the whole checkpoint on a volume, one GPU of at least 160 GB and about 340 GB of host memory:

```sh
FULL_DIR=<volume>/checkpoint ./gpu_session.sh full     # setup, full-fetch, full-real, full-pinned, full-rust, summary
```

| Step | What it does | Wall-clock (estimate) | Needs |
|---|---|---|---|
| `full-fetch` | The checkpoint as published into `FULL_DIR`, shard by shard, resumable; skipped per shard when present | 1–3 h for 510 GB | 510 GB of disk |
| `full-real` | `full_halves.py --kernels real --native-arithmetic`: both halves on the GPU, three patterns, `reference-gpu-native.safetensors` | 30–45 min: each half is 150 GB to read from the volume (5–10 min), then about a minute for the prefill patterns and up to 8 for the 555 single-token calls, by the measured cost of the truncations | 150 GB of GPU memory at a time; a 3 GB golden |
| `full-pinned` | The same on the CPU with emulated kernels and pinned arithmetic, one prefill chunk only: the golden our forward must equal exactly at full depth | 45–90 min on 24 vCPUs, scaled from 90 s for four layers here | about 170 GB of host memory at a time |
| `full-rust` | Our forward over the whole model, twice, against every golden | 30–60 min on 24 vCPUs after a 5–10 min build: here one layer over 555 tokens costs about 9 s, so 40 layers twice are about 12 minutes plus the head, and each run reads every expert of every layer from the volume (about 290 GB) | about 35 GB of memory plus page cache |

None of these four steps has been run. The estimates are scaled from the measured truncation runs and are guesses for the parts that depend on the volume's speed.
