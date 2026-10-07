# Golden generation (dev-only)

Python tooling that produces the reference outputs `eidola-engine-model` is checked against. It is never a build input: `cargo test` reads only the committed fixture under `tests/fixtures/` and needs neither Python nor the network.

The oracle is the Hugging Face remote code for MiMo-V2 (`modeling_mimo_v2.py` / `configuration_mimo_v2.py` from `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` at revision `2479e2d0029eca9a34cc7e7f55a121925f81908e`), downloaded at run time and checked against the sha256 digests in `mimo_golden/remote.py`. It runs in fp32 on CPU with eager attention. Checkpoint tensors are dequantised by `mimo_golden/dequant.py`, written independently of the Rust loader. Beyond placing weights, nothing in the remote code is changed. One call is bridged: the remote code calls transformers' `create_causal_mask` / `create_sliding_window_causal_mask` with the older keyword names (`input_embeds`, `cache_position`), and `mimo_golden/remote.py` renames or drops just those keywords for the pinned transformers. The committed fixture regenerates byte for byte through the bridge (transformers 5.19.0, torch 2.9.1, safetensors 0.8.0), as it does without it on the earlier 5.3.0 / 2.9.0 / 0.6.2 pins.

## Setup

[uv](https://docs.astral.sh/uv/) and Python 3.12. Versions are pinned in `pyproject.toml` and locked in `uv.lock`:

```sh
cd crates/eidola-engine-model/golden
uv sync
```

Downloads (remote code, tokenizer, checkpoint subsets) are cached under `~/.cache/eidola/engine/`, or `$EIDOLA_ENGINE_CACHE` when set.

## Synthetic fixture (committed)

```sh
uv run python make_synthetic.py
```

This rewrites `tests/fixtures/tiny/`:

- `config.json`
- `model.safetensors`: a 4-layer MiMo-shaped model with seeded random weights in the real storage formats.
- `golden.safetensors`: the remote code's logits, the residual stream after every layer, and the final normed hidden state. It also holds two chained MTP layers' inputs and outputs.

`mimo_golden/synthetic.py` explains the shape choices. Output is deterministic for a given torch version, down to the file bytes: the safetensors library keeps header metadata in a hash map, so `save_deterministic` rewrites it with sorted keys (without that, `golden.safetensors` changed bytes run to run with identical tensors). After regenerating, run `cargo test -p eidola-engine-model`.

The remote code ignores the MTP weights, so the MTP goldens are built from the remote code's own modules:

- two `MiMoV2RMSNorm`s and the `eh_proj` linear;
- a one-layer `MiMoV2Model`, configured as sliding-window with a dense MLP;
- the MTP layer's `final_layernorm`;
- the shared `lm_head`.

This follows the published structure of the MTP layer: `eh_proj([enorm(embed) ‖ hnorm(h_prev)])`, then the block, then the final norm.

## Truncated real checkpoint (local, not CI)

```sh
uv run python fetch_truncated.py      # ~15 GB of HTTP range reads
uv run python real_golden.py          # HF forward; ~35 GB peak RAM
cd ../../.. && cargo test --release -p eidola-engine-model --test real_flash -- --ignored --nocapture
```

`fetch_truncated.py` fetches the following, and nothing else:

- `config.json`;
- the tensors of layers 0, 1, 2 and 5:
  - 0 is global attention with a dense FFN;
  - 1 and 2 are sliding-window MoE;
  - 5 is global MoE, whose fused-QKV scale grid has the padded chunks;
- the embeddings, the final norm and the head;
- the three MTP layers.

Tensors are re-packed byte for byte into one local file per upstream shard. `local-sha256.json` records their digests and those of `config.json` and the index (every file the loader reads), and the Rust test verifies the cache against it before loading. The kept layers are run as a 4-layer model, with every layer keeping its own type.

`real_golden.py` tokenises a fixed public-domain passage (363 tokens). It writes `golden/hf-golden.safetensors` into the checkpoint directory, with:

- the tokens;
- logits at every position;
- every layer's output;
- MTP layer 0 over the shifted sequence.

Use `--layers` (and `EIDOLA_MIMO_LAYERS` on the Rust side) to choose another truncation. Use `EIDOLA_MIMO_TRUNCATED_DIR` to point the test at a different directory.
