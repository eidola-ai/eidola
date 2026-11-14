"""Run the reference over the whole checkpoint, one half at a time.

    python full_halves.py --dir <whole checkpoint dir> --reference-dir <dir> [--kernels real]

The checkpoint (about 510 GB) does not fit one GPU; each half of the model
does (about 150 GB). `ds_golden/halves.py` builds the unmodified `model.py`
for layers 0-19, runs it and records the boundary, frees it, then builds
layers 20-39 with the head and teacher-forces them from that boundary.
`check_halves.py` shows on the synthetic fixture that the two halves are the
whole model, bit for bit.

`--dir` holds the checkpoint as published: every `model-*.safetensors` shard,
`config.json` and `tokenizer.json`. The two Engram tables (about 98 GB each)
are not loaded: only the rows the tokens touch are read from their shards.

Three call patterns over the same tokens, each recorded as final logits at
every position (`<pattern>.logits`):

- `prefill`: one prefill chunk. Every layer's streams and every token's routed
  experts are recorded too.
- `decode`: one token per call, the decode layout at every position. Every
  layer's streams are recorded too.
- `reshaped`: a prefill chunk of all but the last 64 tokens, then one token
  per call. When that chunk is at least a window long, the prefill and decode
  layouts coincide past it and the chunk has `prefill`'s layout, so against
  `prefill` this differs in the shape of the calls and nothing else. With
  fewer tokens the chunk has its own layout; its length is recorded
  (`reshaped.first`) so that a comparison reproduces it.

Writes `<dir>/golden/reference-<tag>.safetensors`, tagged as `real_golden.py`
tags its files (`gpu-native`, `gpu-pinned`, `native`, `pinned`), for
`tests/deepseek_real_full.rs`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from ds_golden import halves, pinned_ops, st  # noqa: E402
from ds_golden.reference import args_from_hf_config, load_reference_modules  # noqa: E402
from ds_golden import truncated  # noqa: E402
from real_golden import TEXT, TokenizerShim, log  # noqa: E402
from real_golden import _rows_tensor as rows_tensor  # noqa: E402
from real_golden import _stored_rows as stored_rows  # noqa: E402


def local_rows(directory: Path, ckpt, layer: int, ids: list[int], dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows `ids` of an Engram table: from the mapped checkpoint when it holds
    the table, else from the rows `real_golden.py` fetched for this directory."""
    parts = []
    if f"layers.{layer}.engram.embed.weight" in ckpt:
        for kind, width in (("weight", dim), ("scale", dim // 32)):
            data = ckpt[f"layers.{layer}.engram.embed.{kind}"].data
            parts.append(b"".join(bytes(data[r * width : (r + 1) * width]) for r in ids))
    else:
        stored = stored_rows(directory / "golden" / "engram-rows.safetensors", layer)
        missing = [r for r in ids if r not in stored]
        assert not missing, f"Engram layer {layer}: {len(missing)} rows are neither in the checkpoint nor fetched"
        parts = [b"".join(stored[r][i] for r in ids) for i in range(2)]
    width = dim // 32
    weight = rows_tensor([parts[0][i * dim : (i + 1) * dim] for i in range(len(ids))], dim)
    scale = rows_tensor([parts[1][i * width : (i + 1) * width] for i in range(len(ids))], width)
    return weight.view(torch.float8_e4m3fn), scale.view(torch.float8_e8m0fnu)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--kernels", choices=("shim", "real"), default="shim")
    parser.add_argument("--device", default=None, help="default: cuda with the real kernels, else cpu")
    parser.add_argument("--native-arithmetic", action="store_true", help="leave PyTorch's own arithmetic and top-k as they are")
    parser.add_argument("--split", type=int, default=20, help="the first decoder layer; must be the KV source the later layers read")
    parser.add_argument("--tokens", type=int, default=0, help="use only the first N tokens")
    parser.add_argument("--patterns", default="prefill,decode,reshaped")
    parser.add_argument("--first-layers", type=int, default=0, help="treat the first N layers as the whole model: for checking this script on a partial checkpoint")
    parser.add_argument("--out", type=Path, help="write the golden here instead of <dir>/golden/reference-<tag>.safetensors")
    opts = parser.parse_args()
    (opts.dir / "golden").mkdir(exist_ok=True)
    device = opts.device or ("cuda" if opts.kernels == "real" else "cpu")
    mode = "native" if opts.native_arithmetic else "pinned"
    tag = mode if opts.kernels == "shim" else f"gpu-{mode}"
    began = time.time()

    kernel = []
    if opts.kernels == "real":
        spec = importlib.util.spec_from_file_location("real_kernel", opts.reference_dir / "kernel.py")
        real = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real)
        kernel = [real]
    pinned_ops.install(arithmetic=not opts.native_arithmetic, topk=not opts.native_arithmetic)
    module = load_reference_modules(opts.reference_dir, *kernel)
    engram = sys.modules["engram"]
    config = json.loads((opts.dir / "config.json").read_text())
    full = args_from_hf_config(config)
    # Logits, not drafts: the draft blocks are left out.
    full.update(n_mtp_layers=0, dspark_block_size=0, dspark_target_layer_ids=[], compress_ratios=full["compress_ratios"][: full["n_layers"]])
    if opts.first_layers:
        full = truncated.truncate_args(full, truncated.Truncation(tuple(range(opts.first_layers)), full["n_routed_experts"], 0))
    tokenizer = TokenizerShim(Tokenizer.from_file(str(opts.dir / "tokenizer.json")))
    tokens = [config["bos_token_id"]] + tokenizer.backend_tokenizer.encode(TEXT, add_special_tokens=False).ids
    if opts.tokens:
        tokens = tokens[: opts.tokens]
    n = len(tokens)
    log(f"{n} tokens; split {opts.split}; kernels {opts.kernels}; arithmetic {mode}; device {device}")

    ckpt = st.read_dir(opts.dir)
    # The Engram hash of the checkpoint's own layer ids (a prefix keeps them).
    hash_args = types.SimpleNamespace(**full, max_batch_size=1, max_seq_len=n)
    hasher = engram.NgramHashState(hash_args, engram.EngramLayout.from_args(hash_args), tokenizer)
    rows = hasher(torch.tensor([tokens], device="cpu"), 0)[0].cpu()  # [n, engram layers, lookups]
    unique = [sorted(set(rows[:, s].flatten().tolist())) for s in range(rows.size(1))]
    tables = {l: local_rows(opts.dir, ckpt, l, u, full["engram_head_dim"]) for l, u in zip(full["engram_layer_ids"], unique)}
    compact = torch.stack([torch.searchsorted(torch.tensor(u), rows[:, s].contiguous()) for s, u in enumerate(unique)], dim=1)
    log(f"{sum(len(u) for u in unique)} Engram rows read")

    reshaped_first = max(n - 64, 1)
    known = {"prefill": halves.prefill(n), "decode": halves.steps(n), "reshaped": halves.steps(n, first=reshaped_first)}
    patterns = {name: known[name] for name in opts.patterns.split(",")}
    out = halves.run_halves(module, full, ckpt, tokens, opts.split, patterns, tokenizer, device, tables, compact, log=log)

    golden = {
        "tokens": st.entry(torch.tensor(tokens, dtype=torch.int32)),
        "engram.token_map": st.entry(hasher.token_map.cpu().to(torch.int32)),
    }
    if "reshaped" in out:
        # The length of that pattern's prefill chunk, which decides its layout.
        golden["reshaped.first"] = st.entry(torch.tensor([reshaped_first], dtype=torch.int32))
    bad = []
    for name, tensors in out.items():
        keep = [k for k in tensors if k == "logits" or (name != "reshaped" and k.startswith("layer.")) or (name == "prefill" and k.startswith("route."))]
        for key in keep:
            value = tensors[key].cpu()
            if value.is_floating_point():
                count = value.numel() - int(torch.count_nonzero(torch.isfinite(value.float())))
                if count:
                    bad.append(f"{name}.{key}: {count} of {value.numel()} values not finite")
            golden[f"{name}.{key}"] = st.entry(value)
    target = opts.out or opts.dir / "golden" / f"reference-{tag}.safetensors"
    st.write_file(target, golden)
    log(f"wrote {target.name} after {time.time() - began:.0f} s")
    if bad:
        sys.exit(f"FAILED, reference-{tag}.safetensors holds non-finite values:\n  " + "\n  ".join(bad))


if __name__ == "__main__":
    main()
