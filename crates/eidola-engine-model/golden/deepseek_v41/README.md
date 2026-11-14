# DeepSeek-V4.1 golden generation (dev-only)

Python tooling that produces the reference outputs the DeepSeek-V4.1 forward (`src/deepseek/`) is checked against. It is never a build input: `cargo test` reads only the committed fixture under `tests/fixtures/deepseek_tiny/` and needs neither Python nor the network.

## What the goldens are

**The goldens come from the released reference model code run over an emulated kernel layer, with its library arithmetic pinned. They do not come from the reference's own kernels.**

The reference is the inference code published with the checkpoint: `inference/model.py` and `inference/engram.py` of `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `08c8657f02291e11b6f6b63e0e16014f8e748518`. Those two files run unmodified. Nothing from that repository is vendored here: every script takes the directory holding them as `--reference-dir`.

Three things stand between that code and a CPU, and each is supplied by `ds_golden/`:

1. **The kernels.** The reference's `inference/kernel.py` is a set of tilelang CUDA kernels. `ds_golden/kernel_shim.py` has the same function signatures and computes what the kernel source specifies, in plain PyTorch on a CPU. Where the kernel source does not determine the bits (the accumulation order inside its GEMMs and reductions, the rounding of a float to FP8 or FP4, `exp`, `sigmoid`), the shim takes the rule this crate fixes (`AGENTS.md` → DeepSeek-V4.1 → Numerics contract).
2. **Library arithmetic the reference leaves unspecified.** `ds_golden/pinned_ops.py` replaces, in the `torch` namespace the reference reads them from, exactly the operations whose bits depend on the library or the call shape: the summation order of `F.linear`, `einsum`, `sum` and `mean`; the last bit of `exp`, `log`, `sin` and `cos`; whether a complex multiply is fused; and which of several equal values `topk` returns. Its module docstring lists each operation and its rule.
3. **Modules the text path never uses.** `image_processor` and `vision` are empty stand-ins (the real ones need Pillow).

With the kernels emulated and the arithmetic pinned, the reference's output is a function of the tokens and of how they are split into its prefill chunk and decode steps (its prefill and decode passes lay attention out in different 64-slot blocks), and the Rust forward reproduces it **bit for bit** for either. That exactness is what the committed tests assert. It is agreement between this crate and the reference's model code under this crate's rules. How far the real kernels and a GPU's own arithmetic are from those rules is a separate measurement (below).

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

`gpu_session.sh` is the whole session on a rented Linux machine with one NVIDIA GPU: it installs what it needs, runs the real kernels, and leaves every result in one directory. It has **not been run**: nothing here has a CUDA device. Its step runner (skip a finished step, stop loudly on a failure) and both Python entry points were exercised on a CPU against the emulation.

On a fresh Ubuntu 24.04 machine with the NVIDIA driver and CUDA installed (root is not needed beyond `apt-get install -y git curl build-essential`):

```sh
git clone <this repository> eidola && cd eidola/crates/eidola-engine-model/golden/deepseek_v41
./gpu_session.sh all          # setup, probes, fixture, fetch, real, summary
./gpu_session.sh rust         # optional: our forward against the goldens, on that machine
```

or one step at a time, in this order:

| Step | What it does | Wall-clock (estimate) |
|---|---|---|
| `setup` | Installs uv, a Python 3.12 venv with torch (CUDA wheels from `TORCH_INDEX`), tilelang 0.1.12, numpy, sympy, tokenizers; fetches the reference's `model.py`, `engram.py`, `kernel.py` at the pinned revision; checks that PyTorch sees the GPU and can hold the FP8 / FP4 dtypes on it | 5–10 min |
| `probes` | `gpu_probes.py`: the open numerics questions, on the real kernels (below) | 2–5 min (kernel JIT) |
| `fixture` | `compare_real_kernels.py` on the committed fixture, with the GPU's own arithmetic and with ours pinned: per-kernel real-against-emulation differences, and end to end against the committed goldens | 5–15 min each of two |
| `fetch` | Both truncations by range reads into `SPREAD_DIR` and `PREFIX_DIR` (skipped per shard when present) | 10–20 min for 46 GB on a datacentre link |
| `real` | The reference with its real kernels over both truncations, unpinned (exactly what the released code computes on a GPU) and pinned, writing `reference-gpu-native.safetensors` and `reference-gpu-pinned.safetensors` | unpinned: 2–5 min each; pinned: 15–40 min each (the pinned arithmetic is Python loops) |
| `summary` | `summary.txt` from the logs | seconds |
| `rust` | Installs the Rust toolchain of `rust-toolchain.toml` and runs both ignored tests there | 5–10 min install and build, then about 5 and 2 min on 24 vCPUs |

**Engram rows are the one slow part, and are avoidable.** A run needs two small range reads per table row, the host rate-limits them (about 25 minutes per 12,000 requests from here), and the two truncations need about 24,000 and 12,000 requests. Upload `<dir>/golden/engram-rows.safetensors` of each truncation from a machine that has them into `$SPREAD_DIR/golden/` and `$PREFIX_DIR/golden/` before `real`, and none are fetched. The whole pre-fetched directories can be uploaded the same way (`SPREAD_DIR=… PREFIX_DIR=…`), which skips `fetch` too. With the rows uploaded the session is about 1½–2½ hours; the estimates above are guesses until it has been run once, and `real`'s pinned half is the least certain. A kernel output or a golden value that is not finite fails its step: the scripts count such values on their own, never as a zero difference, and exit non-zero after writing what they have. Everything is resumable: a step that finished leaves `$OUT/done/<step>` and is skipped on a rerun, and the fetches resume by shard and by row.

Results, all under `$OUT` (default `~/eidola-gpu/out`):

- `probes.json`, `fixture-native.json`, `fixture-pinned.json`;
- `goldens/spread/` and `goldens/prefix/`: `reference-gpu-native.safetensors`, `reference-gpu-pinned.safetensors` and the `engram-rows.safetensors` they were made with;
- `logs/<step>.log` and `summary.txt`.

`TORCH_INDEX` defaults to PyTorch's CUDA 13 wheel index and `TORCH_VERSION` to 2.10.0; set them if the machine's driver needs another build. `setup` fails if PyTorch does not see the GPU.

### What the probes answer

`gpu_probes.py` runs the real kernels on inputs built to decide what their source leaves open, and records what they did beside what the emulation does:

- **float → FP8 cast**: all 252 exact ties between adjacent E4M3 codes (both signs), through the in-place path and the code path. Ours: ties to the even code.
- **float → FP4 cast**: all 14 exact ties, with the power-of-two scale and with the E4M3 scale; the E4M3 scale's own rounding when `amax / 6` is an exact tie; and saturation when `amax / scale` exceeds 6.
- **`torch.topk` on CUDA over equal BF16 scores**: whether the lowest indices, the highest, or neither win among the entries tied at the threshold, and whether it is repeatable. Ours: lower index first.
- **`sparse_attn` and block layout**: the same entries in the reference's decode layout, its prefill layout, packed, and with empty slots moved without changing any 64-slot block. The emulation says the first three differ from each other and the last equals the decode layout.

### Comparing the GPU-made goldens with our forward

Copy `goldens/spread/reference-gpu-*.safetensors` into `<12-layer dir>/golden/` and `goldens/prefix/reference-gpu-*.safetensors` into `<prefix dir>/golden/` on a machine that has those directories (the GPU machine itself, with `./gpu_session.sh rust`, or this one), and run the two ignored tests as above. Each test compares every `golden/reference-*.safetensors` it finds: `reference-pinned` is asserted exact, and the others (`native`, `gpu-native`, `gpu-pinned`) are printed. `reference-gpu-pinned` is the one to read first: the real kernels under our arithmetic, so every difference in it is a kernel difference. Measured on a 16-core Apple-silicon Mac after the build: about 3 minutes (`deepseek_real`, three forwards of 256 tokens over 12 layers) and about 70 seconds (`deepseek_real_prefix`, two forwards of 555 tokens over 4 layers), with under 40 GB of memory. On 24 x86 vCPUs expect the same order, perhaps twice as long. The 12-layer goldens must be made with the same `--tokens` as the pinned one they sit beside (256, the script's default).
