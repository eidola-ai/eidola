"""Rewrite the committed synthetic fixture and its goldens.

    uv run python make_synthetic.py [--out <dir>]

Writes `config.json`, `model.safetensors` (seeded random weights in the
checkpoint's storage formats, `qwen_golden/synthetic.py`) and
`golden.safetensors`: the released model code's outputs, with the unspecified
operations pinned (`qwen_golden/pinned_ops.py`), for a fixed token sequence.
The bytes are a function of the code alone.
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


def bf16(t: torch.Tensor) -> tuple:
    assert t.dtype == torch.bfloat16, t.dtype
    return st.entry(t.contiguous())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
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
