"""Goldens of the released model code over real weights: a truncation of the
checkpoint (layers 0-3, the drafter, the embedding and head), or the whole
model.

    uv run python real_golden.py --dir <checkpoint dir> [--model qwen3.8-27b|qwen3.5-2b|qwen3.5-0.8b] \
        [--layers N] [--mode pinned|native|released] [--device cpu|cuda]

Every file the run reads is first checked against the model's recorded list
(`upstream-sha256.json` for Qwen3.8-27B-FP8, `upstream-qwen3.5-*-sha256.json`
for Qwen3.5). `--layers` keeps checkpoint layers 0..N-1 (default 4 for
Qwen3.8-27B-FP8, the whole model for Qwen3.5). The
passage (`qwen_golden/passage.py`) is tokenised with the checkpoint's own
tokenizer, no special tokens. Writes `<dir>/golden/reference-<name>.safetensors`:

- `pinned` (the default; CPU): the unspecified operations pinned
  (`qwen_golden/pinned_ops.py`). Our forward equals it bit for bit.
- `native` (CPU): PyTorch's own arithmetic, its default attention and the
  model code's chunked DeltaNet prefill; only the FP8 GEMM is emulated (its
  kernel is CUDA-only; a BF16 checkpoint has none). Name `native`.
- `released` (CUDA): nothing pinned or emulated. Transformers dispatches the FP8
  GEMM to its own kernels and, when `flash-linear-attention` is installed, the
  DeltaNet to FLA's. Name `gpu-released`, or `--name`.

Contents: `tokens`; `prefill.*` (the passage as one call: logits, the normed
final state, every layer's stream, every DeltaNet layer's states after the
last token); `step.logits` (a call of the first `--step-first` tokens, then one
token per call, as far as `--step-tokens`); `mtp.*` (the drafter at every
position but the last, fed the next token and the target's normed state).
A golden holding a value that is not finite is refused (exit 1).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from qwen_golden import pinned_ops, reference, st, upstream  # noqa: E402
from qwen_golden.passage import TEXT  # noqa: E402


def bf16(t: torch.Tensor) -> tuple:
    assert t.dtype == torch.bfloat16, t.dtype
    return st.entry(t.contiguous())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--model", choices=sorted(upstream.MODELS), default=upstream.DEFAULT_MODEL)
    ap.add_argument("--layers", type=int, help="checkpoint layers 0..N-1 (default: 4 for Qwen3.8, all for Qwen3.5)")
    ap.add_argument("--mode", choices=["pinned", "native", "released"], default="pinned")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", help="the golden's name (default by mode)")
    ap.add_argument("--step-first", type=int, default=64)
    ap.add_argument("--step-tokens", type=int, default=96)
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--torch-deltanet", action="store_true",
                    help="hide flash-linear-attention, so the model code runs its own PyTorch DeltaNet")
    a = ap.parse_args()
    if a.torch_deltanet:
        sys.modules["fla"] = None  # an import of it now fails, and Transformers falls back
    began = time.time()
    upstream.use(a.model)
    upstream.check_file(a.dir / "config.json")  # read below, before the rest is checked
    if a.layers is None:
        a.layers = 4 if "prefix" in upstream.TRUNCATIONS else reference.text_config(a.dir).num_hidden_layers
    upstream.check_dir(a.dir, upstream.truncation_for(a.layers))
    print(f"[{time.time() - began:.0f}s] checkpoint files match the recorded hashes", flush=True)
    name = a.name or {"pinned": "pinned", "native": "native", "released": "gpu-released"}[a.mode]
    if a.mode == "pinned":
        assert a.device == "cpu", "the pins run on the CPU"
        pinned_ops.install()
    elif a.mode == "native":
        assert a.device == "cpu", "native is the CPU run; released is the GPU one"
        pinned_ops.check_environment()
        from transformers.integrations import finegrained_fp8

        finegrained_fp8.fp8_linear = pinned_ops.fp8_linear
    tokenizer = Tokenizer.from_file(str(a.dir / "tokenizer.json"))
    tokens = tokenizer.encode(TEXT, add_special_tokens=False).ids
    print(f"{len(tokens)} tokens", flush=True)

    model, mtp = reference.load(a.dir, layers=list(range(a.layers)), mtp=not a.no_mtp)
    if a.mode != "pinned":
        model.config._attn_implementation = "sdpa"
        if mtp is not None:
            for layer in mtp.layers:
                layer.self_attn.config._attn_implementation = "sdpa"
    model.to(a.device)
    if mtp is not None:
        mtp.to(a.device)
    print(f"[{time.time() - began:.0f}s] model loaded", flush=True)
    golden: dict = {"tokens": st.entry(torch.tensor(tokens, dtype=torch.int32))}
    logits, normed, streams, cache = reference.prefill(model, tokens, device=a.device)
    golden["prefill.logits"] = bf16(logits)
    golden["prefill.normed"] = bf16(normed)
    for i, s in enumerate(streams):
        golden[f"prefill.stream.{i}"] = bf16(s)
    one_call = reference.states(model, cache)
    for i, (conv, rec) in one_call.items():
        golden[f"state.{i}.conv"] = bf16(conv)
        golden[f"state.{i}.recurrent"] = st.entry(rec)
    print(f"[{time.time() - began:.0f}s] prefill done", flush=True)

    n_step = min(a.step_tokens, len(tokens))
    step_logits, _ = reference.stepwise(model, tokens[:n_step], a.step_first, device=a.device)
    golden["step.logits"] = bf16(step_logits)
    golden["step.first"] = st.entry(torch.tensor([a.step_first], dtype=torch.int32))
    same = torch.equal(step_logits, logits[:n_step])
    print(f"[{time.time() - began:.0f}s] stepwise done; equal to one call: {same}", flush=True)

    if mtp is not None:
        hidden, mnormed, mlogits = mtp(model, torch.tensor(tokens[1:]), normed[:-1], torch.arange(len(tokens) - 1))
        golden["mtp.hidden"] = bf16(hidden)
        golden["mtp.normed"] = bf16(mnormed)
        golden["mtp.logits"] = bf16(mlogits)
        print(f"[{time.time() - began:.0f}s] drafter done", flush=True)

    for k, (_, _, data) in golden.items():
        t = st.Entry(*golden[k][:2], memoryview(data)).tensor()
        if t.is_floating_point() and not bool(torch.isfinite(t.float()).all()):
            print(f"FAILED: {k} holds values that are not finite", flush=True)
            return 1
    out = a.dir / "golden" / f"reference-{name}.safetensors"
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    st.write_file(tmp, golden)
    tmp.replace(out)
    print(f"[{time.time() - began:.0f}s] wrote {out}", flush=True)
    if a.mode == "pinned" and not same:
        print("FAILED: the pinned reference's one-token calls differ from its one call", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
