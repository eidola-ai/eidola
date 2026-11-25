# Qwen3.8 and Qwen3.5 golden generation (dev-only)

Python tooling that produces the reference outputs the Qwen forward (`src/qwen/`) is checked against, and checks the released checkpoints' health: Qwen3.8-27B-FP8, and Qwen3.5-2B and -0.8B (below, Qwen3.5). It is never a build input: `cargo test` reads only the committed fixtures under `tests/fixtures/qwen_tiny/`, `qwen_tiny256/`, `qwen_tiny_bf16/` and `qwen_tiny_bf16_tied/` and needs neither Python nor the network.

## What the goldens are

The reference is the released modelling code: `transformers/models/qwen3_5/modeling_qwen3_5.py` of Transformers v5.23.0 (commit `52b080b6416a7897e2af90febeb4dfad22b84d19`), installed from PyPI at that version and run unmodified on a CPU. `qwen_golden/reference.py` builds `Qwen3_5ForCausalLM` from a checkpoint directory: its FP8 projections are Transformers' own `FP8Linear` modules holding the checkpoint's E4M3 weights and BF16 block scales, assigned directly from the safetensors files (Transformers' FP8 quantiser, which needs a GPU, is not involved).

The model code states most of its own rounding points: the BF16 residual stream, the norms' single rounding, β as the sigmoid of a BF16 tensor (so β is BF16), the gated norm's three roundings, RoPE's table cast to BF16 and its per-operation BF16 arithmetic, the head's BF16 logits. Those are followed as written. What it leaves to PyTorch, the C library or a kernel is replaced by `qwen_golden/pinned_ops.py`, in the namespaces the model code reads it from; its module docstring lists each operation and its rule:

- **The FP8 GEMM.** `FP8Linear` calls a CUDA-only Triton kernel from the Hub (`kernels-community/finegrained-fp8`, read at revision `0fc209e04af4db8fb02cc9416d44a9a28d4b2c71`: `_fp8_act_quant_kernel` and `w8a8_block_fp8_matmul_kernel`). The emulation follows its source: per token and 128 columns `s = amax / 448` in f32, codes `(x / s)` cast to E4M3, blocks accumulated in ascending order as `acc + (dot · s) · weight_scale`, BF16 out. What the source leaves open is fixed: the cast rounds to nearest, ties to the even code, saturating (a NaN, from an all-zero block, casts to +0), and each 128-term block dot is exact, rounded once to f32.
- **Library arithmetic.** The summation order of `F.linear`, `sum` and `mean`; the depthwise `conv1d`'s accumulation; `rsqrt`; the last bit of `exp`, `sigmoid`, `silu`, `softplus`, `sin`, `cos` and `scalar ** tensor` (the engine's deterministic functions, built from correctly rounded `+ - * /`).
- **The attention core.** The eager attention function is replaced by an f32 attention with one BF16 rounding of the output (scores in the pinned order over the causally visible keys, `exp` of the score minus the maximum, the normaliser summed left to right, `Σ (w / Σw) · v` in ascending position).
- **The DeltaNet form.** The model code runs a chunked form (`torch_chunk_gated_delta_rule`) for any call of more than one token and the token recurrence (`torch_recurrent_gated_delta_rule`) for one token with a cache. Pinned, every call runs the model code's own recurrence, so a layer's state and output depend only on the token prefix. `pinned_ops.check_environment()` refuses to run when `flash-linear-attention`, the Hub `kernels` package or `causal_conv1d` is importable, since any of them would replace the model code's PyTorch paths.

With these in place the reference is a function of the tokens alone: its one-token calls equal its one call bit for bit (checked by `make_synthetic.py` and `real_golden.py`), and the Rust forward reproduces it bit for bit. That exactness is what the committed tests assert. How far the reference as released on a GPU is from it is a separate measurement (`full_session.sh`, below).

### The drafter's oracle

Transformers has no MTP code (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`). `reference.Mtp` rebuilds the drafter vLLM serves, `Qwen3_5MultiTokenPredictor` in `vllm/model_executor/models/qwen3_5_mtp.py` at vLLM v0.33.0, from Transformers' own classes:

1. `pre_fc_norm_embedding` of the next token's embedding (the shared `embed_tokens`) and `pre_fc_norm_hidden` of the hidden state, both `Qwen3_5RMSNorm` (lines 146-160 there);
2. concatenated embedding first and projected by the BF16 `fc` (`torch.cat([inputs_embeds, hidden_states], dim=-1)`, `self.fc`, lines 161-163);
3. one decoder layer of the full-attention kind (`Qwen3_5DecoderLayer(..., layer_type="full_attention")`, lines 116-125): here Transformers' `Qwen3_5DecoderLayer` built from the text config with that one layer type, its RoPE the text model's at the row's position;
4. the drafter's norm (`self.norm`, line 187), then the text model's head (vLLM's proposer shares it).

vLLM fuses each residual add into the next norm; Transformers' layer rounds the stream to BF16 after each add, and the oracle is Transformers' layer. The goldens feed row `i` the token at `i + 1` and the target's normed final state at `i` (vLLM's target returns post-norm states). On a GPU the drafter is still to be compared with vLLM's own.

## Setup

[uv](https://docs.astral.sh/uv/) and Python 3.12. This directory is its own project, with versions pinned in `pyproject.toml` and locked in `uv.lock` (torch from PyTorch's CPU-only index). It is separate from the other architectures' tooling.

```sh
cd crates/eidola-engine-model/golden/qwen
uv sync
```

## Synthetic fixture (committed)

```sh
uv run python make_synthetic.py
```

rewrites `tests/fixtures/qwen_tiny/` (about 3 MB):

- `config.json`: the checkpoint's config shape at small sizes;
- `model.safetensors`: seeded random weights in the real storage formats; `qwen_golden/synthetic.py` gives the dimensions and why each was chosen (three DeltaNet layers, one attention layer, one MTP layer; DeltaNet key and value dims unequal so a transposed state is caught);
- `golden.safetensors`: `tokens` (80); `prefill.*`, the sequence as one call (logits, the normed final state, every layer's stream, every DeltaNet layer's convolution history and recurrent state after the last token); `step.logits`, a call of 24 tokens then one token per call; `mtp.*`, the drafter at every position but the last.

Output is deterministic down to the file bytes. After regenerating, run `cargo test -p eidola-engine-model`.

```sh
uv run python make_synthetic.py --head-dim 256
```

rewrites `tests/fixtures/qwen_tiny256/` (about 5 MB): the same model with the released attention geometry (256-wide heads, 12 query heads over 2 KV heads, a rotary width of 64; `synthetic.head_dim_256`), the files and goldens as above. The CUDA executor's fast profile runs on it, since its attention kernels are compiled for 256-wide heads; `eidola-engine-cuda`'s `tests/qwen_host.rs` holds the model crate's forward to its goldens bit for bit.

## Real weights (local, not CI)

```sh
uv run python fetch_checkpoint.py --dir <dir> --truncation prefix      # about 7 GB; `full` is the whole 31 GB
uv run python real_golden.py --dir <dir>                               # pinned; about 4 minutes
uv run python real_golden.py --dir <dir> --mode native                 # PyTorch's own arithmetic; about 2 minutes
cd ../../../.. && EIDOLA_QWEN_DIR=<dir> \
    cargo test --release -p eidola-engine-model --test qwen_real -- --ignored --nocapture
```

The `prefix` truncation is whole files of the checkpoint: `layers-0` to `layers-3` (three DeltaNet layers and one attention layer), `mtp.safetensors`, and `outside.safetensors` (the embedding, the head, the final norm, and the vision tower, which is never read), with `config.json`, the tokenizer, the chat template and `LICENSE`. Each shard holds whole layers, so a truncation needs no range reads and no re-packing.

`real_golden.py` tokenises a fixed passage of prose and code (`qwen_golden/passage.py`, 454 tokens) with the checkpoint's own tokenizer and writes `<dir>/golden/reference-<mode>.safetensors` with the same contents as the fixture's golden (`step.logits` is a call of 64 tokens then one token per call, as far as 96). `--layers 64` runs the whole model over the whole checkpoint. Modes:

- `pinned` (CPU): the goldens our forward equals bit for bit.
- `native` (CPU): nothing pinned but the FP8 GEMM, whose kernel is CUDA-only: PyTorch's own arithmetic, its default (`sdpa`) attention, and the model code's chunked DeltaNet for the one call and its recurrence for one-token calls.
- `released` (CUDA, `--device cuda`): nothing pinned or emulated. Transformers dispatches the FP8 GEMM to its own kernels and the DeltaNet to `flash-linear-attention`'s when it is installed (`--torch-deltanet` hides it).

`tests/qwen_real.rs` asserts exact equality with `reference-pinned`, and reports every other golden (top-1, KL, per-layer stream relative error), each golden's one-token calls against its one call (for `native`, the recurrence against the chunked form), and the goldens against each other.

## Recorded hashes

`upstream-sha256.json` records, for the pinned revision of `Qwen/Qwen3.8-27B-FP8`, the size and SHA-256 of every file used here. A weight shard's and `tokenizer.json`'s SHA-256 is its Git LFS object id in the repository's tree listing (`https://huggingface.co/api/models/Qwen/Qwen3.8-27B-FP8/tree/<revision>`); a small file's was computed from bytes whose Git blob id (also recorded) equals the listing's.

Nothing is loaded that has not been checked against it. `fetch_checkpoint.py` downloads each file whole, from its first byte, under a name nothing loads, hashes it as it streams, and renames it only when size and SHA-256 match; a failed attempt is deleted and the next starts from the first byte again (a partial download is never resumed), and a file already present is checked again rather than trusted. `real_golden.py`, `health_check.py` and the Rust tests check every file before they load. A size check is not enough: a resumed download once left a shard of another model at the right length with the wrong bytes, and a reference and a forward that read it agreed with each other about a model that was not the released one.

## Health: is this the model

`health_check.py` checks the files, and the reference's run of them, against facts outside this tooling (`qwen_golden/health.py`):

- next-token accuracy over the first 1,024 tokens of the checkpoint's own `LICENSE` (the Apache-2.0 text) of at least 0.95; the released model served by vLLM measured 0.998 on these tokens;
- a greedy reply to "What is the capital of France? Answer with one word.", rendered by the checkpoint's own chat template with thinking off (`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`), containing "Paris";
- next-token accuracy over the goldens' passage, reported.

Measured on the hash-verified checkpoint with the pinned reference on a CPU, all 64 layers: accuracy 0.998 over the licence text, 0.614 over the passage, and the reply `Paris<|im_end|>`. Our forward gives the same three (`AGENTS.md` → Qwen3.8 → What is verified against what).

It writes `<dir>/golden/health.safetensors`: the token sequences (`--tokens-only` writes only these), the reference's most likely token at every position and its reply, and which mode it ran in. `tests/qwen_health.rs` holds our forward to the same floor and reply on the whole checkpoint, and, when the health file came from the pinned reference, to the reference's choices exactly.

## The whole model, and the GPU comparison

`full_session.sh` runs everything over the whole checkpoint and leaves comparable outputs under `$OUT`. On a CPU machine (`DEVICE=cpu`, the default) it runs our forward, the pinned and native references and the health check; on a Linux machine with one NVIDIA GPU (`DEVICE=cuda`) it adds the reference as released on the GPU (FLA and Transformers' FP8 dispatch; then the model code's own PyTorch DeltaNet) and runs the health check there.

```sh
CKPT=<dir> ./full_session.sh all                 # or one step at a time
DRY_RUN=1 CKPT=<dir> ./full_session.sh all       # print every command, run none
```

| Step | What it does | Wall-clock (M4 Max, 16 cores) |
|---|---|---|
| `setup` | `uv sync`; on a GPU machine a second venv with CUDA torch, Transformers 5.23.0, `kernels` 0.18.3 and `flash-linear-attention` 0.6.0; rustup if there is no `cargo` | |
| `fetch` | The whole checkpoint, each file checked before it is named | 12 minutes for 31 GB at four streams |
| `tokens` | The health check's token sequences | seconds |
| `pinned` | `real_golden.py --layers 64 --mode pinned` | 12.5 minutes, 38 GB peak RSS |
| `native` | `real_golden.py --layers 64 --mode native`, eight one-token calls (PyTorch's BF16 depthwise convolution is slow on a CPU) | 18 minutes while another job shared the machine, most of it the one-token calls |
| `released`, `released-torch` | GPU only: the reference as released, with FLA's DeltaNet and then with the model code's own | not yet run; minutes each |
| `health` | The reference's health: pinned on a CPU, as released on a GPU | 15 minutes and 37 GB peak RSS pinned on the CPU |
| `rust` | `tests/qwen_real.rs` at 64 layers against every golden, and `tests/qwen_health.rs` | 570 s (35 GB peak RSS) and 890 s (34 GB) |
| `summary` | `summary.txt` from the logs; fails if any step that ran has no done marker | seconds |

On a GPU machine only `setup`, `fetch`, `tokens`, `released`, `released-torch` and `health` need the GPU: run those there, copy `<CKPT>/golden/` back, and run `rust` (and, if not done yet, `pinned` and `native`) on a CPU machine with the checkpoint.

Every step is resumable: a finished step leaves `$OUT/done/<step>` and is skipped on a rerun. A step that fails exits non-zero, leaves no marker and stops the session; its output is in `$OUT/logs/<step>.log`. The released runs' goldens land beside the pinned one in `<CKPT>/golden/`, which is what the `rust` step compares.

## Qwen3.5

The same tooling runs Qwen3.5-2B and -0.8B (`Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc`, `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`). Their reference is the same pinned Transformers code. Three things differ in how it is built (`qwen_golden/reference.py`):

- **BF16 projections.** A checkpoint without a `quantization_config` keeps the model code's own `nn.Linear`s, whose `F.linear` is pinned (`pinned_ops.linear`); nothing of the FP8 emulation runs.
- **The head tied to the embedding**, the model's `lm_head` the embedding's tensor (asserted), and the checkpoint carries no `lm_head.weight`.
- **Every F32 parameter loaded into BF16.** The checkpoints store `A_log` and the gated norm's weight in F32. `from_pretrained` builds the model in the config's dtype and copies every parameter into it, and vLLM's loader does the same, so the reference rounds them to BF16 (ties to even) where Qwen3.8's tooling assigns the stored tensors as they are (all of its small tensors are BF16 already).

### Synthetic fixtures (committed)

```sh
uv run python make_synthetic.py --qwen35            # tests/fixtures/qwen_tiny_bf16/
uv run python make_synthetic.py --qwen35 --tied     # tests/fixtures/qwen_tiny_bf16_tied/
```

`synthetic.qwen35`: the checkpoint's config keys (no `quantization_config`, `language_model_only`, `output_gate_type`, top-level `partial_rotary_factor`, `bos_token_id` or `pad_token_id`; `mlp_only_layers: []`), BF16 projections, `A_log` and the gated-norm weight in F32, the released groupings (six value heads over six key heads, eight query heads over two KV heads) at the fixture's other sizes, and with `--tied` no `lm_head.weight` and the embedding drawn at the head's scale. The random draws are the same as `qwen_tiny`'s, so these change only where the format does. Regenerating every fixture with the extended tooling leaves `qwen_tiny` and `qwen_tiny256` byte for byte as they were.

### Real weights (local, not CI)

```sh
uv run python fetch_checkpoint.py --model qwen3.5-2b --dir <dir>      # 4.5 GB; 0.8b: 1.7 GB
uv run python real_golden.py --model qwen3.5-2b --dir <dir>           # the whole model, pinned
uv run python real_golden.py --model qwen3.5-2b --dir <dir> --mode native
uv run python health_check.py --model qwen3.5-2b --dir <dir>
cd ../../../.. && EIDOLA_QWEN_DIR=<dir> EIDOLA_QWEN_LAYERS=24 \
    cargo test --release -p eidola-engine-model --test qwen_real -- --ignored --nocapture
EIDOLA_QWEN_DIR=<dir> cargo test --release -p eidola-engine-model --test qwen_health -- --ignored --nocapture
EIDOLA_QWEN_DIR=<dir> cargo test --release -p eidola-engine-cpu --test qwen_real -- --ignored --nocapture
```

Each checkpoint is one shard, fetched whole (`--truncation full`, the only one), and `--layers` defaults to the whole model. Its files are recorded in `upstream-qwen3.5-2b-sha256.json` and `upstream-qwen3.5-0.8b-sha256.json` as Qwen3.8's are (`--model` selects the list; `qwen_golden/upstream.py`), and the Rust tests pick the list whose `config.json` the directory holds. On an M4 Max: `real_golden.py` pinned 246 s for the 2B and 90 s for the 0.8B, `native` about 5 and 3 minutes (most of it the one-token calls), `health_check.py` about 6 and 3 minutes; the measurements are in `eidola-engine-model/AGENTS.md` → Qwen3.5.
