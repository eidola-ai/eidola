"""Health: is this the released model?

    uv run python health_check.py --dir <checkpoint dir> [--tokens-only] [--mode pinned|native|released] [--device cpu|cuda]

Agreement with the reference shows that a forward computes what the
modelling code computes on the files both were given; it says nothing about
the files. This checks them, and the reference's run of them, against facts
outside this tooling (`qwen_golden/health.py`):

- every checkpoint file is the recorded upstream one (`upstream-sha256.json`);
- next-token accuracy over the first 1,024 tokens of the checkpoint's own
  `LICENSE` (the Apache-2.0 text) is at least 0.95 (the released model served
  by vLLM measured 0.998);
- the greedy reply to "What is the capital of France? Answer with one word.",
  rendered by the checkpoint's own chat template with thinking off, contains
  "Paris";
- next-token accuracy over the goldens' passage (prose and code) is reported.

Writes `<dir>/golden/health.safetensors`: the token sequences (so the Rust test
`tests/qwen_health.rs` holds our forward to the same checks), and, unless
`--tokens-only`, the reference's most likely token at every position and its
greedy reply. Exits non-zero when a check fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from qwen_golden import health, pinned_ops, reference, st, upstream  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--tokens-only", action="store_true")
    ap.add_argument("--mode", choices=["pinned", "native", "released"], default="pinned",
                    help="as in real_golden.py: pinned and native on the CPU, released on a GPU")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layers", type=int, default=64, help="fewer only to exercise the script on a truncation")
    a = ap.parse_args()
    upstream.check_dir(a.dir, "full" if a.layers == 64 else "prefix")
    log("checkpoint files match the recorded hashes")
    seqs = health.sequences(a.dir)
    out = {k: st.entry(torch.tensor(v, dtype=torch.int32)) for k, v in seqs.items()}
    for k, v in seqs.items():
        log(f"{k}: {len(v)} tokens")
    if a.tokens_only:
        health.write(a.dir, out)
        return 0

    if a.mode == "pinned":
        assert a.device == "cpu", "the pins run on the CPU"
        pinned_ops.install()
    elif a.mode == "native":
        assert a.device == "cpu"
        pinned_ops.check_environment()
        from transformers.integrations import finegrained_fp8

        finegrained_fp8.fp8_linear = pinned_ops.fp8_linear
    model, _ = reference.load(a.dir, layers=list(range(a.layers)), mtp=False)
    if a.mode != "pinned":
        model.config._attn_implementation = "sdpa"
    model.to(a.device)
    log("model loaded")
    ok = True
    for name, floor in (("license", health.LICENSE_FLOOR), ("passage", None)):
        tokens = seqs[f"{name}.tokens"]
        logits, *_ = reference.prefill(model, tokens, device=a.device)
        top1 = logits.float().argmax(-1).to(torch.int32).cpu()
        out[f"{name}.reference_top1"] = st.entry(top1)
        acc = health.accuracy(top1.tolist(), tokens)
        verdict = "" if floor is None else (" (floor %.2f: %s)" % (floor, "ok" if acc >= floor else "FAILED"))
        log(f"{name}: next-token accuracy {acc:.3f}{verdict}")
        ok &= floor is None or acc >= floor
    reply = health.greedy(model, seqs["capital.prompt"], seqs["capital.stop"], device=a.device)
    out["capital.reference_reply"] = st.entry(torch.tensor(reply, dtype=torch.int32))
    text = health.decode(a.dir, reply)
    found = health.ANSWER in text
    log(f"capital: reply {text!r} ({'ok' if found else 'FAILED'})")
    ok &= found
    out["reference.mode"] = st.entry(torch.tensor([["pinned", "native", "released"].index(a.mode)], dtype=torch.int32))
    health.write(a.dir, out)
    if not ok:
        log("FAILED: the model is not healthy")
        return 1
    log("healthy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
