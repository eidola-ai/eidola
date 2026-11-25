"""Rewrite the committed synthetic fixture and its goldens.

    uv run python make_synthetic.py [--out <dir>] [--head-dim 256]

Writes `config.json`, `model.safetensors` (seeded random weights in the
checkpoint's storage formats, `qwen_golden/synthetic.py`) and
`golden.safetensors`: the released model code's outputs, with the unspecified
operations pinned (`qwen_golden/pinned_ops.py`), for a fixed token sequence.
The bytes are a function of the code alone.

`--head-dim 256` writes the released attention geometry at the fixture's other
sizes (`synthetic.head_dim_256`) into `tests/fixtures/qwen_tiny256` by default:
the fixture the CUDA executor's fast profile runs, whose attention kernels are
compiled for 256-wide heads.

`--qwen35` writes a Qwen3.5 checkpoint's shape (`synthetic.qwen35`: BF16
projections, Qwen3.5's config keys and groupings) into
`tests/fixtures/qwen_tiny_bf16`, and with `--tied` its head tied to the
embedding into `tests/fixtures/qwen_tiny_bf16_tied`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from qwen_golden import pinned_ops, reference, st, synthetic  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent.parent / "tests" / "fixtures" / "qwen_tiny"
D256_OUT = HERE.parent.parent / "tests" / "fixtures" / "qwen_tiny256"
BF16_OUT = HERE.parent.parent / "tests" / "fixtures" / "qwen_tiny_bf16"
TIED_OUT = HERE.parent.parent / "tests" / "fixtures" / "qwen_tiny_bf16_tied"


def bf16(t: torch.Tensor) -> tuple:
    assert t.dtype == torch.bfloat16, t.dtype
    return st.entry(t.contiguous())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--head-dim", type=int, choices=[128, 256], default=128)
    ap.add_argument("--qwen35", action="store_true", help="a Qwen3.5 checkpoint's shape: BF16 projections")
    ap.add_argument("--tied", action="store_true", help="with --qwen35: the head tied to the embedding")
    a = ap.parse_args()
    assert not (a.tied and not a.qwen35), "--tied is a Qwen3.5 shape"
    assert not (a.qwen35 and a.head_dim == 256), "the Qwen3.5 fixture keeps the 128-wide heads"
    if a.head_dim == 256:
        synthetic.head_dim_256()
    if a.qwen35:
        synthetic.qwen35(a.tied)
    default = TIED_OUT if a.tied else BF16_OUT if a.qwen35 else D256_OUT if a.head_dim == 256 else DEFAULT_OUT
    a.out = a.out or default
    torch.set_num_threads(1)
    pinned_ops.install()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps(synthetic.CONFIG, indent=2) + "\n")
    st.write_file(a.out / "model.safetensors", synthetic.weights())

    model, mtp = reference.load(a.out)
    tokens = synthetic.tokens()
    golden: dict = {"tokens": st.entry(torch.tensor(tokens, dtype=torch.int32))}

    logits, normed, streams, cache = reference.prefill(model, tokens)
    golden["prefill.logits"] = bf16(logits)
    golden["prefill.normed"] = bf16(normed)
    for i, s in enumerate(streams):
        golden[f"prefill.stream.{i}"] = bf16(s)
    one_call = reference.states(model, cache)
    for i, (conv, rec) in one_call.items():
        golden[f"state.{i}.conv"] = bf16(conv)
        golden[f"state.{i}.recurrent"] = st.entry(rec)

    step_logits, step_cache = reference.stepwise(model, tokens, synthetic.STEP_FIRST)
    golden["step.logits"] = bf16(step_logits)
    same = torch.equal(step_logits, logits)
    for i, (conv, rec) in reference.states(model, step_cache).items():
        same &= torch.equal(conv, one_call[i][0]) and torch.equal(rec, one_call[i][1])
    print(f"stepwise equals one call: {same}")

    positions = torch.arange(len(tokens) - 1)
    hidden, mnormed, mlogits = mtp(model, torch.tensor(tokens[1:]), normed[:-1], positions)
    golden["mtp.hidden"] = bf16(hidden)
    golden["mtp.normed"] = bf16(mnormed)
    golden["mtp.logits"] = bf16(mlogits)

    st.write_file(a.out / "golden.safetensors", golden)
    for name in ("config.json", "model.safetensors", "golden.safetensors"):
        data = (a.out / name).read_bytes()
        print(f"{name}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
