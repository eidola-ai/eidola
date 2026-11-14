"""Check the half-at-a-time driver on the committed synthetic fixture.

    uv run python check_halves.py --reference-dir <dir holding model.py and engram.py>

The reference is run whole and then as two halves (`ds_golden/halves.py`), on
the emulated kernels with the arithmetic pinned, for one prefill chunk, for
one token per call and for a prefill chunk followed by single tokens. Every
layer's streams, the final normed state and the logits must be equal bit for
bit, or the script exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from ds_golden import halves, pinned_ops, st, synthetic  # noqa: E402
from ds_golden.reference import Reference, args_from_hf_config, load_reference_modules  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "deepseek_tiny"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--split", type=int, default=4, help="the fixture's ratio-1 KV source, as layer 20 is in the released model")
    opts = parser.parse_args()
    pinned_ops.install()
    module = load_reference_modules(opts.reference_dir)
    config = json.loads((opts.fixture / "config.json").read_text())
    args = args_from_hf_config(config)
    # The fixture's drafter reads layers on both sides of the split; the
    # halves produce logits, not drafts.
    args.update(n_mtp_layers=0, dspark_block_size=0, dspark_target_layer_ids=[], compress_ratios=args["compress_ratios"][: args["n_layers"]])
    ckpt = st.read_file(opts.fixture / "model.safetensors")
    tokenizer = synthetic.SyntheticTokenizer(config["text_config"]["vocab_size"])
    tokens = synthetic.tokens(config)
    n = len(tokens)
    patterns = {"prefill": halves.prefill(n), "decode": halves.steps(n), "reshaped": halves.steps(n, first=n - 20)}

    whole = Reference(module, args, tokenizer, max_seq_len=n + 8)
    ckpt_whole = {k: v for k, v in ckpt.items() if not k.startswith("mtp.")}
    whole.place(ckpt_whole)
    want = {}
    for name, calls in patterns.items():
        whole.records.clear()
        ids = torch.tensor([tokens], dtype=torch.long)
        for start, end in calls:
            whole.model.forward(ids[:, start:end], start)
        want[name] = {k: torch.cat(v, dim=0) for k, v in whole.records.items()}

    got = halves.run_halves(module, args, ckpt_whole, tokens, opts.split, patterns, tokenizer, "cpu")
    failures = 0
    for name in patterns:
        compared = 0
        for key, expected in want[name].items():
            if key.startswith(("attn.", "ffn.", "engram_rows")):
                continue
            compared += 1
            if not torch.equal(got[name][key], expected):
                failures += 1
                print(f"{name}: {key} differs")
        head = whole.model.head.weight
        logits = torch.stack([torch.nn.functional.linear(row.float(), head) for row in want[name]["normed"]])
        same = torch.equal(got[name]["logits"], logits)
        failures += not same
        print(f"{name}: {compared} tensors compared, logits {'equal' if same else 'DIFFER'}")
    if failures:
        sys.exit(f"FAILED: {failures} tensors differ between the whole model and its halves")
    print("the halves reproduce the whole model exactly")


if __name__ == "__main__":
    main()
