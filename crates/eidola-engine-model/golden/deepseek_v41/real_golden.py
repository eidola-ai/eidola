"""Run the reference over a truncated real checkpoint and record goldens.

    uv run python real_golden.py --dir <checkpoint dir> --reference-dir <dir holding model.py and engram.py>

Writes `<dir>/golden/reference-<tag>.safetensors`. The tag is `pinned`
(emulated kernels, pinned arithmetic), `native` (`--native-arithmetic`:
emulated kernels, PyTorch's own arithmetic and top-k), or, with
`--kernels real` on a CUDA machine, `gpu-native` / `gpu-pinned`: the
reference's own tilelang kernels (`--reference-dir` must then hold `kernel.py`).

What is recorded depends on the truncation (`--truncation`, the one
`fetch_truncated.py` fetched into `--dir`):

- `spread`: the logits and every layer's streams for the text as one prefill
  chunk, then a one-window prefill followed by `--step-tokens` tokens one per
  call, with the drafter.
- `prefix`: every layer's streams and every token's routed experts for the
  text as one prefill chunk (four layers have no meaningful logits), then the
  first `--step-tokens` tokens one per call.

The Engram rows the tokens touch are fetched by HTTP range reads into
`<dir>/golden/engram-rows.safetensors`; rows already in that file, or in the
same file of a directory given with `--rows-from`, are not fetched again.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from ds_golden import kernel_compare, pinned_ops, st, truncated  # noqa: E402
from ds_golden.reference import Reference, args_from_hf_config, load_reference_modules, place_weights  # noqa: E402

TEXT = '''The keeper of the northern light kept two logbooks. In the first he wrote what the service required: the hour the lamp was lit, the hour it was put out, the state of the sea, the direction of the wind, and the quantity of oil consumed during the night. In the second, which nobody had asked him to keep, he wrote down everything else. He recorded the colour of the water at dawn, the birds that struck the glass in fog, and the names of the vessels he could identify as they passed, together with a guess at their cargo and their destination. Over thirty years the second book grew to eleven volumes, and it was the second book, not the first, that the harbour board asked to see when the old charts were finally redrawn.

It turned out that he had been measuring the tide against a mark of his own, cut into the rock below the landing, and that his figures disagreed with the published tables by a small but steady amount. Nobody had noticed, because nobody else had looked at the same rock twice a day for thirty years. The surveyors who came to check his mark found it sound, corrected the tables, and left him a brass plate, which he never fixed to anything.

A short program in the same spirit reads a list of observations and reports how far each one lies from the running mean:

```python
def drift(observations):
    total, count = 0.0, 0
    report = []
    for value in observations:
        count += 1
        total += value
        mean = total / count
        report.append((count, value, round(value - mean, 3)))
    return report


if __name__ == "__main__":
    readings = [4.12, 4.15, 4.11, 4.19, 4.22, 4.18, 4.25]
    for index, value, offset in drift(readings):
        sign = "+" if offset >= 0 else "-"
        print(f"{index:>2}  {value:.2f}  {sign}{abs(offset):.3f}")
```

Run on the keeper's figures for a single month, it prints a column of offsets that climbs slowly and never comes back, which is exactly what the surveyors saw when they laid his second logbook beside their own.

The board kept both sets of books after that. The official log went to the archive in the capital, bound in blue cloth, and was never opened again. The eleven private volumes stayed in the harbour office, on the shelf above the tide gauge, where the pilots consulted them before every spring tide until the paper wore through at the corners.
'''


class TokenizerShim:
    """What the reference's Engram map needs of a tokenizer: its length and the
    raw `decode` / `id_to_token`."""

    def __init__(self, tokenizer: Tokenizer):
        self.backend_tokenizer = tokenizer
        self.size = tokenizer.get_vocab_size(with_added_tokens=True)

    def __len__(self):
        return self.size


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def _rows_tensor(blobs: list[bytes], width: int) -> torch.Tensor:
    """`[len(blobs), width]` bytes; an explicitly shaped empty tensor for no
    rows (`torch.frombuffer` refuses an empty buffer)."""
    if not blobs:
        return torch.zeros((0, width), dtype=torch.uint8)
    return torch.frombuffer(bytearray(b"".join(blobs)), dtype=torch.uint8).view(len(blobs), width)


def _stored_rows(path: Path, layer: int) -> dict[int, tuple[bytes, bytes]]:
    if not path.exists():
        return {}
    have = st.read_file(path)
    if f"layers.{layer}.rows" not in have:
        return {}
    ids = have[f"layers.{layer}.rows"].tensor().tolist()
    if not ids:
        return {}
    weight = have[f"layers.{layer}.weight"].tensor().view(torch.uint8)
    scale = have[f"layers.{layer}.scale"].tensor().view(torch.uint8)
    return {r: (bytes(weight[i].tolist()), bytes(scale[i].tolist())) for i, r in enumerate(ids)}


def fetch_engram_rows(directory: Path, layers: list[int], rows: list[list[int]], dim: int, sources: list[Path]) -> dict[int, tuple]:
    """Table rows `rows[i]` (sorted, unique) of Engram layer `layers[i]`, as
    compact `(weight, scale)` tables in that order.

    The store file keeps every row ever fetched for this directory and is
    rewritten after each batch, so an interrupted fetch resumes."""
    path = directory / "golden" / "engram-rows.safetensors"
    store = {layer: _stored_rows(path, layer) for layer in layers}
    for source in sources:
        for layer in layers:
            for row, value in _stored_rows(source / "golden" / "engram-rows.safetensors", layer).items():
                store[layer].setdefault(row, value)

    def save():
        tensors = {}
        for layer in layers:
            # A layer whose rows have not been fetched yet is written as
            # explicitly shaped empty tensors, and read back as no rows.
            ids = sorted(store[layer])
            tensors[f"layers.{layer}.rows"] = st.entry(torch.tensor(ids, dtype=torch.int64))
            tensors[f"layers.{layer}.weight"] = st.entry(_rows_tensor([store[layer][r][0] for r in ids], dim), "F8_E4M3")
            tensors[f"layers.{layer}.scale"] = st.entry(_rows_tensor([store[layer][r][1] for r in ids], dim // 32), "F8_E8M0")
        st.write_file(path.with_suffix(".partial"), tensors)
        path.with_suffix(".partial").replace(path)

    index = None
    for layer, ids in zip(layers, rows):
        missing = [r for r in ids if r not in store[layer]]
        if not missing:
            continue
        index = index or json.loads(truncated.get("model.safetensors.index.json"))["weight_map"]
        where = {}
        for kind in ("weight", "scale"):
            name = f"layers.{layer}.engram.embed.{kind}"
            header, base = truncated.remote_header(index[name])
            where[kind] = (index[name], base + header[name]["data_offsets"][0])

        def one(row):
            out = []
            for kind, width in (("weight", dim), ("scale", dim // 32)):
                shard, start = where[kind]
                out.append(truncated.get(shard, start + row * width, start + (row + 1) * width))
            return row, tuple(out)

        with concurrent.futures.ThreadPoolExecutor(6) as pool:
            for at in range(0, len(missing), 256):
                for row, value in pool.map(one, missing[at : at + 256]):
                    store[layer][row] = value
                save()
                log(f"Engram layer {layer}: {min(at + 256, len(missing))} of {len(missing)} missing rows fetched")
    save()
    out = {}
    for layer, ids in zip(layers, rows):
        weight = _rows_tensor([store[layer][r][0] for r in ids], dim)
        scale = _rows_tensor([store[layer][r][1] for r in ids], dim // 32)
        out[layer] = (weight.view(torch.float8_e4m3fn), scale.view(torch.float8_e8m0fnu))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--truncation", choices=sorted(truncated.TRUNCATIONS), default="spread")
    parser.add_argument("--tokens", type=int, default=0, help="use only the first N tokens")
    parser.add_argument("--step-tokens", type=int, default=96)
    parser.add_argument("--native-arithmetic", action="store_true", help="leave PyTorch's own arithmetic and top-k as they are")
    parser.add_argument("--kernels", choices=("shim", "real"), default="shim")
    parser.add_argument("--device", default=None, help="default: cuda with the real kernels, else cpu")
    parser.add_argument("--kernel-stats", type=Path, help="with the real kernels: also give every kernel call to the emulation and write the per-kernel differences here")
    parser.add_argument("--rows-from", type=Path, action="append", default=[], help="another checkpoint dir whose fetched Engram rows may be reused")
    opts = parser.parse_args()
    (opts.dir / "golden").mkdir(exist_ok=True)
    t = truncated.TRUNCATIONS[opts.truncation]
    device = opts.device or ("cuda" if opts.kernels == "real" else "cpu")
    mode = "native" if opts.native_arithmetic else "pinned"
    tag = mode if opts.kernels == "shim" else f"gpu-{mode}"
    began = time.time()

    kernel = None
    if opts.kernels == "real":
        spec = importlib.util.spec_from_file_location("real_kernel", opts.reference_dir / "kernel.py")
        kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernel)
    kernel_stats = None
    if opts.kernel_stats:
        assert kernel is not None, "--kernel-stats compares the real kernels with the emulation"
        kernel, kernel_stats = kernel_compare.instrumented(kernel)
    pinned_ops.install(arithmetic=not opts.native_arithmetic, topk=not opts.native_arithmetic)
    module = load_reference_modules(opts.reference_dir, *([kernel] if kernel else []))
    engram = sys.modules["engram"]
    config = json.loads((opts.dir / "config.json").read_text())
    full = args_from_hf_config(config)
    tokenizer = Tokenizer.from_file(str(opts.dir / "tokenizer.json"))
    shim = TokenizerShim(tokenizer)
    tokens = [config["bos_token_id"]] + tokenizer.encode(TEXT, add_special_tokens=False).ids
    if opts.tokens:
        tokens = tokens[: opts.tokens]
    n = len(tokens)
    log(f"{n} tokens; truncation {opts.truncation}; kernels {opts.kernels}; arithmetic {mode}; device {device}")

    # Table rows from the reference's own hash over the original layer ids.
    hash_args = types.SimpleNamespace(**full, max_batch_size=1, max_seq_len=n)
    layout = engram.EngramLayout.from_args(hash_args)
    hasher = engram.NgramHashState(hash_args, layout, shim)
    rows = hasher(torch.tensor([tokens]), 0)[0]  # [n, engram layers, lookups]
    kept = [l for l in full["engram_layer_ids"] if l in t.layers]
    slots = [full["engram_layer_ids"].index(l) for l in kept]
    unique = [sorted(set(rows[:, s].flatten().tolist())) for s in slots]
    tables = fetch_engram_rows(opts.dir, kept, unique, full["engram_head_dim"], opts.rows_from)
    compact = torch.stack(
        [torch.searchsorted(torch.tensor(u), rows[:, s].contiguous()) for u, s in zip(unique, slots)], dim=1
    )

    args = truncated.truncate_args(full, t)
    args["engram_num_embeddings"] = [len(u) for u in unique]
    reference = Reference(module, args, shim, max_seq_len=n + 8, device=device)
    reference.place(
        st.read_dir(opts.dir),
        layer_map=t.layers,
        routed_experts=(t.backbone_experts, t.drafter_experts),
        engram_tables={t.layers.index(l): tables[l] for l in kept},
    )
    # The truncated model's own hash would use the renumbered layer ids.
    compact = compact.to(device)
    reference.model.engram_hash.register_forward_hook(
        lambda _m, inputs, _out: compact[inputs[1] : inputs[1] + inputs[0].size(1)].unsqueeze(0)
    )
    log(f"loaded in {time.time() - began:.0f} s")

    layers = range(args["n_layers"])
    golden = {
        "tokens": st.entry(torch.tensor(tokens, dtype=torch.int32)),
        "engram.token_map": st.entry(hasher.token_map.cpu().to(torch.int32)),
    }
    spread = t.drafter
    out = reference.run(tokens, prefill=n, logits=spread)
    for i in layers:
        golden[f"prefill.layer.{i}"] = st.entry(out[f"layer.{i}"])
        golden[f"prefill.route.{i}"] = st.entry(out[f"route.{i}"])
    if spread:
        golden["prefill.logits"] = st.entry(out["logits"])
        golden["prefill.main_hidden"] = st.entry(out["main_hidden"])
    log(f"prefill done at {time.time() - began:.0f} s")
    if opts.step_tokens and spread and n > args["window_size"]:
        # A prefill chunk of one window, then one token per call: every
        # position then has the layout of the decode pattern.
        window = args["window_size"]
        k = min(window + opts.step_tokens, n)
        step = reference.run(tokens[:k], prefill=min(window, k), drafter=True)
        golden["step.logits"] = st.entry(step["logits"])
        for i in layers:
            for part in ("layer", "attn", "ffn"):
                golden[f"step.{part}.{i}"] = st.entry(step[f"{part}.{i}"])
        golden["step.draft_tokens"] = st.entry(step["draft_tokens"])
        golden["step.draft_confidence"] = st.entry(step["draft_confidence"])
        golden["step.draft_logits"] = st.entry(step["draft_logits"][:16])
        log(f"steps done at {time.time() - began:.0f} s")
    elif opts.step_tokens and not spread:
        # One token per call from the first: the reference's decode layout at
        # every position (no global entry exists at position 0 here).
        k = min(opts.step_tokens, n)
        step = reference.run(tokens[:k], prefill=1, logits=False)
        for i in layers:
            golden[f"step.layer.{i}"] = st.entry(step[f"layer.{i}"])
        log(f"steps done at {time.time() - began:.0f} s")
    st.write_file(opts.dir / "golden" / f"reference-{tag}.safetensors", golden)
    log(f"wrote reference-{tag}.safetensors")
    if kernel_stats is not None:
        kernel_compare.print_stats(kernel_stats)
        opts.kernel_stats.write_text(json.dumps({"truncation": opts.truncation, "tokens": n, "arithmetic": mode, "kernels": kernel_stats}, indent=2) + "\n")
        if kernel_compare.failures(kernel_stats):
            sys.exit("FAILED, non-finite kernel outputs:\n  " + "\n  ".join(kernel_compare.failures(kernel_stats)))
    # Fail closed: a golden with a value that is not finite is a failed run,
    # written so that it can be inspected.
    bad = []
    for name, (dtype, shape, data) in sorted(golden.items()):
        if dtype in ("BF16", "F32") and data:
            values = torch.frombuffer(bytearray(data), dtype=st.TORCH_DTYPE[dtype]).float()
            count = values.numel() - int(torch.count_nonzero(torch.isfinite(values)))
            if count:
                bad.append(f"{name}: {count} of {values.numel()} values not finite")
    if bad:
        sys.exit(f"FAILED, reference-{tag}.safetensors holds non-finite values:\n  " + "\n  ".join(bad))


if __name__ == "__main__":
    main()
