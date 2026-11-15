"""Is this the model? The reference over the whole checkpoint, against the world.

    python health_check.py --dir <whole checkpoint dir> --reference-dir <dir> --encoding-dir <dir> [--kernels real]

Agreement with the reference proves a forward computes what `model.py`
computes on the bytes both were given. It proves nothing about those bytes: a
damaged embedding table gives a reference and a forward that agree to the last
bit and a model that cannot continue a sentence. This check is the other
half, and it fails loudly:

- every checkpoint file is first checked against the recorded upstream hashes
  (`upstream-sha256.json`);
- the `hello` request of `ds_golden/health.py` must be answered, token for
  token, with the publicly reported reply: at every position of it the most
  likely token is the reply's, except at the one position the report gives as
  a near three-way tie, where it must be one of the three reported;
- next-token accuracy over the story as a chat reply must reach its floor.

`--encoding-dir` holds the repository's `encoding/encoding.py`, the published
prompt format, unmodified. The model runs one half at a time
(`ds_golden/halves.py`), both halves loaded once for all sequences.

Writes `<dir>/golden/health.safetensors` for `tests/deepseek_real_full.rs`:
each sequence's tokens (`<name>.tokens`), where its scored reply starts
(`<name>.start`) and the reference's most likely next token at every position
(`<name>.argmax`), the tie's position and its three tokens (`hello.fork`,
`hello.fork_tokens`), with the Engram token map.

`--first-layers N` treats the first N layers as the model, for checking this
script on a partial checkpoint: it asserts nothing about the model, skips the
upstream hashes (a truncation's re-packs are checked by `fetch_truncated.py`)
and fetches the Engram rows it needs.
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

from ds_golden import halves, health, pinned_ops, st, truncated, upstream  # noqa: E402
from ds_golden.reference import args_from_hf_config, load_reference_modules  # noqa: E402
from full_halves import local_rows  # noqa: E402
from real_golden import TokenizerShim, fetch_engram_rows, log  # noqa: E402


def load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sequences(encoding, tokenizer: Tokenizer, bos: int) -> dict[str, tuple[list[int], int]]:
    """`name -> (tokens, start)`: a rendered chat request followed by the
    reply, scored from the reply's first token."""

    def chat(messages, reply):
        prompt = encoding.encode_messages([{"role": r, "content": c} for r, c in messages], thinking_mode="chat")
        head = tokenizer.encode(prompt, add_special_tokens=False).ids
        assert head[0] == bos and head.count(bos) == 1, "the rendered prompt starts with one beginning-of-sequence token"
        return head + tokenizer.encode(reply, add_special_tokens=False).ids, len(head)

    return {"hello": chat(health.HELLO_MESSAGES, health.HELLO_REPLY), "story": chat(health.STORY_MESSAGES, health.STORY)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--encoding-dir", type=Path, required=True)
    parser.add_argument("--kernels", choices=("shim", "real"), default="shim")
    parser.add_argument("--device", default=None, help="default: cuda with the real kernels, else cpu")
    parser.add_argument("--native-arithmetic", action="store_true", help="leave PyTorch's own arithmetic and top-k as they are")
    parser.add_argument("--split", type=int, default=20, help="the first decoder layer; must be the KV source the later layers read")
    parser.add_argument("--sequences", default="hello,story")
    parser.add_argument("--first-layers", type=int, default=0, help="treat the first N layers as the whole model; asserts nothing about the model")
    parser.add_argument("--json", type=Path, help="write the measurements here")
    parser.add_argument("--out", type=Path, help="write the golden here instead of <dir>/golden/health.safetensors")
    opts = parser.parse_args()
    (opts.dir / "golden").mkdir(exist_ok=True)
    device = opts.device or ("cuda" if opts.kernels == "real" else "cpu")
    began = time.time()

    if not opts.first_layers:
        try:
            names = upstream.check_dir(opts.dir)
        except upstream.Mismatch as error:
            sys.exit(f"REFUSED {error}")
        log(f"{len(names)} checkpoint files match their recorded hashes")

    kernel = [load_by_path("real_kernel", opts.reference_dir / "kernel.py")] if opts.kernels == "real" else []
    pinned_ops.install(arithmetic=not opts.native_arithmetic, topk=not opts.native_arithmetic)
    module = load_reference_modules(opts.reference_dir, *kernel)
    engram = sys.modules["engram"]
    encoding = load_by_path("published_encoding", opts.encoding_dir / "encoding.py")
    config = json.loads((opts.dir / "config.json").read_text())
    full = args_from_hf_config(config)
    full.update(n_mtp_layers=0, dspark_block_size=0, dspark_target_layer_ids=[], compress_ratios=full["compress_ratios"][: full["n_layers"]])
    split = opts.split
    if opts.first_layers:
        full = truncated.truncate_args(full, truncated.Truncation(tuple(range(opts.first_layers)), full["n_routed_experts"], 0))
    layers = full["n_layers"]
    backend = Tokenizer.from_file(str(opts.dir / "tokenizer.json"))
    tokenizer = TokenizerShim(backend)
    seqs = sequences(encoding, backend, config["bos_token_id"])
    seqs = {name: seqs[name] for name in opts.sequences.split(",")}
    longest = max(len(tokens) for tokens, _ in seqs.values())
    for name, (tokens, start) in seqs.items():
        log(f"{name}: {len(tokens)} tokens, scored from {start}")

    ckpt = st.read_dir(opts.dir)
    # One compact Engram table per layer for all sequences: the union of the
    # rows they touch, and each sequence's row ids within it.
    hash_args = types.SimpleNamespace(**full, max_batch_size=1, max_seq_len=longest)
    hasher = engram.NgramHashState(hash_args, engram.EngramLayout.from_args(hash_args), tokenizer)
    hashed = {name: hasher(torch.tensor([tokens], device="cpu"), 0)[0].cpu() for name, (tokens, _) in seqs.items()}
    slots = next(iter(hashed.values())).size(1)
    unique = [sorted({r for rows in hashed.values() for r in rows[:, s].flatten().tolist()}) for s in range(slots)]
    kept = full["engram_layer_ids"]
    if all(f"layers.{l}.engram.embed.weight" in ckpt for l in kept):
        tables = {l: local_rows(opts.dir, ckpt, l, u, full["engram_head_dim"]) for l, u in zip(kept, unique)}
    else:
        tables = fetch_engram_rows(opts.dir, kept, unique, full["engram_head_dim"], [])
    compact = {
        name: torch.stack([torch.searchsorted(torch.tensor(u), rows[:, s].contiguous()) for s, u in enumerate(unique)], dim=1)
        for name, rows in hashed.items()
    }
    log(f"{sum(len(u) for u in unique)} Engram rows read")

    # A loaded half must not carry anything from one sequence into the next:
    # the first sequence is run again last and must come out the same.
    order = list(seqs) + [next(iter(seqs))]
    encoder = halves.Half(module, full, ckpt, range(0, split), tokenizer, longest + 8, device, tables, None)
    log(f"encoder half (layers 0-{split - 1}) loaded")
    boundaries = []
    for name in order:
        tokens = seqs[name][0]
        encoder.engram_rows = compact[name]
        e = encoder.run(len(tokens), halves.prefill(len(tokens)), tokens=tokens)
        boundaries.append((e["boundary.streams"].clone(), e["boundary.pre_mix"].clone()))
    assert torch.equal(boundaries[0][0], boundaries[-1][0]) and torch.equal(boundaries[0][1], boundaries[-1][1]), "the encoder half carried state between sequences"
    encoder.free()
    decoder = halves.Half(module, full, ckpt, range(split, layers), tokenizer, longest + 8, device)
    log(f"decoder half (layers {split}-{layers - 1}) loaded")
    logits = {}
    for name, boundary in zip(order, boundaries):
        n = len(seqs[name][0])
        got = decoder.run(n, halves.prefill(n), boundary=boundary, logits=True)["logits"].detach().float().cpu()
        assert name not in logits or torch.equal(logits[name], got), "the decoder half carried state between sequences"
        logits[name] = got
    decoder.free()

    golden = {"engram.token_map": st.entry(hasher.token_map.cpu().to(torch.int32))}
    report, failures = {}, []
    word = lambda t: backend.id_to_token(int(t))  # noqa: E731
    for name, (tokens, start) in seqs.items():
        argmax = logits[name].argmax(-1)
        golden[f"{name}.tokens"] = st.entry(torch.tensor(tokens, dtype=torch.int32))
        golden[f"{name}.start"] = st.entry(torch.tensor([start], dtype=torch.int32))
        golden[f"{name}.argmax"] = st.entry(argmax.to(torch.int32))
        hits = [int(argmax[i]) == tokens[i + 1] for i in range(start - 1, len(tokens) - 1)]
        accuracy = sum(hits) / len(hits)
        spans = [round(sum(hits[a : a + 128]) / len(hits[a : a + 128]), 3) for a in range(0, len(hits), 128)]
        report[name] = {"tokens": len(tokens), "scored": len(hits), "accuracy": round(accuracy, 4), "by_128_positions": spans}
        log(f"{name}: next-token accuracy {accuracy:.3f} over {len(hits)} positions; by 128-position span {spans}")
        if name == "hello":
            lp = logits[name].log_softmax(-1)
            for i in range(start - 1, len(tokens) - 1):
                top = lp[i].topk(3)
                log(f"   want {word(tokens[i + 1])!r} ({float(lp[i, tokens[i + 1]]):.2f}); most likely " + ", ".join(f"{word(t)!r} {float(v):.2f}" for v, t in zip(top.values, top.indices)))
            # The reported position is a near three-way tie (about -2.4 each in
            # the report), so there the most likely token must be one of the
            # three reported; everywhere else it must be the reply's.
            fork = start - 1 + len(backend.encode(health.HELLO_FORK, add_special_tokens=False).ids)
            reported = [backend.encode(t, add_special_tokens=False).ids for t in health.HELLO_FORK_TOP3]
            assert all(len(ids) == 1 for ids in reported) and tokens[fork + 1] == reported[0][0]
            reported = [ids[0] for ids in reported]
            golden[f"{name}.fork"] = st.entry(torch.tensor([fork], dtype=torch.int32))
            golden[f"{name}.fork_tokens"] = st.entry(torch.tensor(reported, dtype=torch.int32))
            wrong = [i for i in range(start - 1, len(tokens) - 1) if i != fork and int(argmax[i]) != tokens[i + 1]]
            if wrong:
                failures.append(f"hello: {len(wrong)} tokens of the reported reply are not the most likely token (first at position {wrong[0]})")
            top3 = [int(t) for t in lp[fork].topk(3).indices]
            report[name]["after_how_can"] = {backend.decode([t]): round(float(lp[fork, t]), 3) for t in top3}
            if top3[0] not in reported:
                failures.append(f"hello: the most likely token after {health.HELLO_FORK!r} is {backend.decode([top3[0]])!r}, not one of the reported {list(health.HELLO_FORK_TOP3)}")
            log(f"   after {health.HELLO_FORK!r}: {report[name]['after_how_can']}; reported {list(health.HELLO_FORK_TOP3)} at about -2.4 each")
        if name == "story" and accuracy < health.STORY_FLOOR:
            failures.append(f"story: next-token accuracy {accuracy:.3f} is under the floor of {health.STORY_FLOOR}")
    if opts.json:
        opts.json.write_text(json.dumps(report, indent=1) + "\n")
    target = opts.out or opts.dir / "golden" / "health.safetensors"
    st.write_file(target, golden)
    log(f"wrote {target.name} after {time.time() - began:.0f} s")
    if opts.first_layers:
        log(f"the first {opts.first_layers} layers are not the model: nothing was asserted")
    elif failures:
        sys.exit("UNHEALTHY: this is not the released model.\n  " + "\n  ".join(failures))
    else:
        log("healthy: the reported reply is reproduced and the accuracy floor is met")


if __name__ == "__main__":
    main()
