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

## The real kernels (GPU, one command)

The emulation's distance from the reference's real kernels is measured, not assumed. On a CUDA machine with tilelang 0.1.12:

```sh
uv run --no-project --with torch --with tilelang==0.1.12 --with numpy --with sympy --with tokenizers \
    python compare_real_kernels.py --reference-dir <dir> --json real-kernels.json
```

- **Input:** the committed synthetic fixture (`--fixture` to point elsewhere) and the reference directory, which here must also hold the upstream `kernel.py`.
- **What runs:** the reference model on the GPU with its real kernels. Every kernel call is also given to the emulation on the same inputs.
- **Output:** per kernel, the number of calls, elements, elements that differ and the largest difference; end to end, top-1 agreement, KL and the per-layer stream differences against the committed goldens, for the decode and the prefill pattern. `--json` writes the same numbers.
- **Variants:** by default PyTorch's arithmetic and `topk` are the GPU's own (the released code as it runs). `--pinned` pins them as the goldens do, so that only the kernels differ.

`--kernels shim --device cpu` runs the script against the emulation itself, where every difference is zero: that is how the script is checked without a GPU. It has not been run against the real kernels.
