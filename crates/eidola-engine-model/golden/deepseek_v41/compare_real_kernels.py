"""Measure the emulated kernel layer against the reference's real kernels.

The committed goldens come from the reference model code over `kernel_shim`
with its library arithmetic pinned. This script runs the same synthetic
fixture through the reference's own `kernel.py` (tilelang CUDA kernels) and
reports, in one pass:

- per kernel: every call's output from the real kernel against the emulation
  on the same inputs (calls, elements, elements that differ, largest
  difference);
- end to end: logits and per-layer streams against the committed goldens.

On a CUDA machine with tilelang 0.1.12 (one command; `--reference-dir` holds
the upstream `model.py`, `engram.py` and `kernel.py`):

    uv run --no-project --with torch --with tilelang==0.1.12 --with numpy \
        --with sympy --with tokenizers python compare_real_kernels.py \
        --reference-dir <upstream inference dir>

By default PyTorch's own arithmetic and `topk` are left as the GPU computes
them, which is the released code as it runs. `--pinned` pins them as the
goldens do, so that only the kernels differ. `--kernels shim --device cpu`
runs the script against the emulation itself (every difference is then zero);
it is how the script is checked on a machine without CUDA.

Output: a table on stdout and, with `--json <path>`, the same numbers as JSON.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from ds_golden import kernel_shim, pinned_ops, st, synthetic  # noqa: E402
from ds_golden.reference import Reference, args_from_hf_config, load_reference_modules  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "deepseek_tiny"
KERNELS = ("act_quant", "fp4_act_quant", "fp4_gemm", "fp8_gemm", "hc_split_sinkhorn", "sparse_attn")


def _cpu(x):
    return x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x


def _values(out) -> list[torch.Tensor]:
    """Outputs as comparable f32 tensors (FP8 / scale tensors decoded)."""
    outs = out if isinstance(out, tuple) else (out,)
    return [o.detach().cpu().view(torch.uint8).float() if o.dtype == torch.float4_e2m1fn_x2 else o.detach().cpu().float() for o in outs]


def instrumented(real) -> tuple[types.ModuleType, dict]:
    """A `kernel` module that runs the real kernel and the emulation on every
    call, returns the real result, and tallies the differences."""
    stats = {k: {"calls": 0, "elements": 0, "differing": 0, "max_abs_diff": 0.0} for k in KERNELS}
    module = types.ModuleType("kernel")

    def wrap(name):
        def call(*args, **kwargs):
            cpu_args = [_cpu(a) for a in args]
            cpu_kwargs = {k: _cpu(v) for k, v in kwargs.items()}
            # FP8 weights carry their scale as an attribute, which a clone drops.
            for a, c in zip(args, cpu_args):
                if isinstance(a, torch.Tensor) and hasattr(a, "scale"):
                    c.scale = _cpu(a.scale)
            with torch.device("cpu"):
                want = _values(getattr(kernel_shim, name)(*cpu_args, **cpu_kwargs))
            out = getattr(real, name)(*args, **kwargs)
            s = stats[name]
            s["calls"] += 1
            for got, expected in zip(_values(out), want):
                diff = (got - expected).abs()
                diff = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
                s["elements"] += got.numel()
                s["differing"] += int(torch.count_nonzero(diff))
                s["max_abs_diff"] = max(s["max_abs_diff"], float(diff.max()) if diff.numel() else 0.0)
            return out

        return call

    for name in KERNELS:
        setattr(module, name, wrap(name))
    return module, stats


def agreement(want: torch.Tensor, got: torch.Tensor) -> dict:
    # Plain Python sums: `Tensor.sum` / `mean` may be pinned to f32 rules.
    lw, lg = torch.log_softmax(want.double(), -1), torch.log_softmax(got.double(), -1)
    kl = [math.fsum(row) for row in (lw.exp() * (lw - lg)).tolist()]
    same = torch.count_nonzero(want.argmax(-1) == got.argmax(-1))
    return {
        "rows": want.size(0),
        "top1": int(same) / want.size(0),
        "kl_mean": math.fsum(kl) / len(kl),
        "kl_max": max(kl),
        "max_abs_diff": float((want - got).abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--kernels", choices=("real", "shim"), default="real")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pinned", action="store_true", help="pin PyTorch's arithmetic and top-k as the goldens do")
    parser.add_argument("--json", type=Path)
    opts = parser.parse_args()

    if opts.kernels == "real":
        spec = importlib.util.spec_from_file_location("real_kernel", opts.reference_dir / "kernel.py")
        real = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real)
    else:
        real = kernel_shim
    kernel, stats = instrumented(real)
    if opts.pinned:
        pinned_ops.install()
    module = load_reference_modules(opts.reference_dir, kernel)

    config = json.loads((opts.fixture / "config.json").read_text())
    golden = {k: v.tensor() for k, v in st.read_file(opts.fixture / "golden.safetensors").items()}
    tokens = golden["tokens"].tolist()
    tokenizer = synthetic.SyntheticTokenizer(config["text_config"]["vocab_size"])
    reference = Reference(module, args_from_hf_config(config), tokenizer, max_seq_len=128, device=opts.device)
    reference.place(st.read_file(opts.fixture / "model.safetensors"))

    report = {"kernels": stats, "end_to_end": {}}
    for pattern, prefill in (("step", 1), ("prefill", len(tokens))):
        print(f"{time.strftime('%H:%M:%S')} running the {pattern} pattern", flush=True)
        out = {k: v.cpu() for k, v in reference.run(tokens, prefill=prefill).items()}
        entry = {"logits": agreement(golden[f"{pattern}.logits"], out["logits"].float())}
        for i in range(config["text_config"]["num_hidden_layers"]):
            want, got = golden[f"{pattern}.layer.{i}"].float(), out[f"layer.{i}"].float()
            entry[f"layer.{i}"] = {
                "differing": int(torch.count_nonzero(want != got)),
                "elements": want.numel(),
                "max_abs_diff": float((want - got).abs().max()),
            }
        report["end_to_end"][pattern] = entry

    print(f"kernels: {opts.kernels}; arithmetic: {'pinned' if opts.pinned else 'native'}; device: {opts.device}")
    print(f"{'kernel':<20}{'calls':>8}{'elements':>12}{'differing':>12}{'max |diff|':>14}")
    for name, s in stats.items():
        print(f"{name:<20}{s['calls']:>8}{s['elements']:>12}{s['differing']:>12}{s['max_abs_diff']:>14.6g}")
    for pattern, entry in report["end_to_end"].items():
        a = entry["logits"]
        print(f"{pattern}: top-1 {a['top1']:.4%}  KL mean {a['kl_mean']:.3e} max {a['kl_max']:.3e}  max |dlogit| {a['max_abs_diff']:.3e}")
        for key, value in entry.items():
            if key != "logits":
                print(f"  {key}: {value['differing']} of {value['elements']} differ, max |diff| {value['max_abs_diff']:.4g}")
    if opts.json:
        opts.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
