"""Run the reference over a truncated real checkpoint and record goldens.

    uv run python real_golden.py --dir <checkpoint dir> --reference-dir <dir holding model.py and engram.py>

Writes `<dir>/golden/reference-pinned.safetensors` (or `-native` with
`--native-arithmetic`): the tokens, the Engram compressed-vocabulary map, the
logits and every layer's streams for the whole text as one prefill chunk, and,
in the pinned run, a prefill chunk of one window followed by `--step-tokens`
tokens one per call, with the drafter. The Engram rows the tokens touch are fetched by
HTTP range reads into `<dir>/golden/engram-rows.safetensors`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from ds_golden import pinned_ops, st, truncated  # noqa: E402
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
'''


class TokenizerShim:
    """What the reference's Engram map needs of a tokenizer: its length and the
    raw `decode` / `id_to_token`."""

    def __init__(self, tokenizer: Tokenizer):
        self.backend_tokenizer = tokenizer
        self.size = tokenizer.get_vocab_size(with_added_tokens=True)

    def __len__(self):
        return self.size


def fetch_engram_rows(directory: Path, layers: list[int], rows: list[list[int]], dim: int) -> dict[int, tuple]:
    """Table rows `rows[i]` (sorted, unique) of Engram layer `layers[i]`."""
    path = directory / "golden" / "engram-rows.safetensors"
    if path.exists():
        have = st.read_file(path)
        if all(have[f"layers.{l}.rows"].tensor().tolist() == r for l, r in zip(layers, rows)):
            return {l: (have[f"layers.{l}.weight"].tensor(), have[f"layers.{l}.scale"].tensor()) for l in layers}
    index = json.loads(truncated.get("model.safetensors.index.json"))["weight_map"]
    out, tensors = {}, {}
    for layer, ids in zip(layers, rows):
        parts = {}
        for kind, width in (("weight", dim), ("scale", dim // 32)):
            name = f"layers.{layer}.engram.embed.{kind}"
            header, base = truncated.remote_header(index[name])
            start = base + header[name]["data_offsets"][0]
            # Rows are appended in id order, so an interrupted fetch resumes
            # from the file's length. The id list is part of the file's name.
            digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
            partial = directory / "golden" / f"engram-{layer}-{kind}-{digest}.partial"
            done = partial.stat().st_size // width if partial.exists() else 0
            with open(partial, "ab") as f, concurrent.futures.ThreadPoolExecutor(6) as pool:
                for at in range(done, len(ids), 256):
                    batch = ids[at : at + 256]
                    chunks = pool.map(lambda r: truncated.get(index[name], start + r * width, start + (r + 1) * width), batch)
                    f.write(b"".join(chunks))
                    f.flush()
                    print(f"layer {layer} {kind}: {at + len(batch)} of {len(ids)} rows", flush=True)
            parts[kind] = torch.frombuffer(bytearray(partial.read_bytes()), dtype=torch.uint8).view(len(ids), width)
        weight, scale = parts["weight"].view(torch.float8_e4m3fn), parts["scale"].view(torch.float8_e8m0fnu)
        out[layer] = (weight, scale)
        tensors[f"layers.{layer}.rows"] = st.entry(torch.tensor(ids, dtype=torch.int64))
        tensors[f"layers.{layer}.weight"] = st.entry(weight)
        tensors[f"layers.{layer}.scale"] = st.entry(scale)
    st.write_file(path, tensors)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=0, help="use only the first N tokens")
    parser.add_argument("--step-tokens", type=int, default=96)
    parser.add_argument("--native-arithmetic", action="store_true", help="leave PyTorch's own arithmetic and top-k as they are")
    opts = parser.parse_args()
    (opts.dir / "golden").mkdir(exist_ok=True)
    began = time.time()

    pinned_ops.install(arithmetic=not opts.native_arithmetic, topk=not opts.native_arithmetic)
    module = load_reference_modules(opts.reference_dir)
    engram = sys.modules["engram"]
    config = json.loads((opts.dir / "config.json").read_text())
    full = args_from_hf_config(config)
    tokenizer = Tokenizer.from_file(str(opts.dir / "tokenizer.json"))
    shim = TokenizerShim(tokenizer)
    tokens = [config["bos_token_id"]] + tokenizer.encode(TEXT, add_special_tokens=False).ids
    if opts.tokens:
        tokens = tokens[: opts.tokens]
    n = len(tokens)
    print(f"{n} tokens", flush=True)

    # Table rows from the reference's own hash over the original layer ids.
    hash_args = types.SimpleNamespace(**full, max_batch_size=1, max_seq_len=n)
    layout = engram.EngramLayout.from_args(hash_args)
    hasher = engram.NgramHashState(hash_args, layout, shim)
    rows = hasher(torch.tensor([tokens]), 0)[0]  # [n, engram layers, lookups]
    kept = [l for l in full["engram_layer_ids"] if l in truncated.LAYERS]
    slots = [full["engram_layer_ids"].index(l) for l in kept]
    unique = [sorted(set(rows[:, s].flatten().tolist())) for s in slots]
    tables = fetch_engram_rows(opts.dir, kept, unique, full["engram_head_dim"])
    compact = torch.stack(
        [torch.searchsorted(torch.tensor(u), rows[:, s].contiguous()) for u, s in zip(unique, slots)], dim=1
    )

    args = truncated.truncate_args(full)
    args["engram_num_embeddings"] = [len(u) for u in unique]
    reference = Reference(module, args, shim, max_seq_len=n + 8)
    model_tables = {truncated.LAYERS.index(l): tables[l] for l in kept}
    place_weights(
        reference.model,
        st.read_dir(opts.dir),
        layer_map=truncated.LAYERS,
        routed_experts=(truncated.BACKBONE_EXPERTS, truncated.DRAFTER_EXPERTS),
        engram_tables=model_tables,
    )
    # The truncated model's own hash would use the renumbered layer ids.
    reference.model.engram_hash.register_forward_hook(
        lambda _m, inputs, _out: compact[inputs[1] : inputs[1] + inputs[0].size(1)].unsqueeze(0)
    )
    print(f"loaded in {time.time() - began:.0f} s", flush=True)

    golden = {
        "tokens": st.entry(torch.tensor(tokens, dtype=torch.int32)),
        "engram.token_map": st.entry(hasher.token_map.to(torch.int32)),
    }
    out = reference.run(tokens, prefill=n)
    for i in range(args["n_layers"]):
        golden[f"prefill.layer.{i}"] = st.entry(out[f"layer.{i}"])
    golden["prefill.logits"] = st.entry(out["logits"])
    golden["prefill.main_hidden"] = st.entry(out["main_hidden"])
    print(f"prefill done at {time.time() - began:.0f} s", flush=True)
    if not opts.native_arithmetic and opts.step_tokens:
        # A prefill chunk of one window, then one token per call: every
        # position then has the layout of the decode pattern.
        window = args["window_size"]
        k = min(window + opts.step_tokens, n)
        step = reference.run(tokens[:k], prefill=min(window, k), drafter=True)
        golden["step.logits"] = st.entry(step["logits"])
        for i in range(args["n_layers"]):
            for part in ("layer", "attn", "ffn"):
                golden[f"step.{part}.{i}"] = st.entry(step[f"{part}.{i}"])
        golden["step.draft_tokens"] = st.entry(step["draft_tokens"])
        golden["step.draft_confidence"] = st.entry(step["draft_confidence"])
        golden["step.draft_logits"] = st.entry(step["draft_logits"][:16])
        print(f"steps done at {time.time() - began:.0f} s", flush=True)
    mode = "native" if opts.native_arithmetic else "pinned"
    st.write_file(opts.dir / "golden" / f"reference-{mode}.safetensors", golden)


if __name__ == "__main__":
    main()
