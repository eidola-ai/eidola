"""Measure the emulated kernel layer against the reference's real kernels.

The committed goldens come from the reference model code over `kernel_shim`
with its library arithmetic pinned. This script gives the same inputs to the
reference's own `kernel.py` (tilelang CUDA kernels) and to the emulation, and
reports per kernel the calls, elements, elements that differ and the largest
difference.

    python compare_real_kernels.py --reference-dir <dir holding model.py, engram.py, kernel.py> --json out.json

Two sources of inputs (`--inputs`):

- `generated` (the default): one call of every kernel on seeded inputs at the
  released model's dimensions. This is what runs on a GPU.
- `fixture`: the reference model run over the committed synthetic fixture,
  every kernel call compared, plus the end-to-end logits and streams against
  the committed goldens. **The real kernels cannot run it**: tilelang 0.1.12
  refuses to compile `sparse_attn` at the fixture's head dimension of 32
  (`warp_col_tiles must be greater than 8, got 4`). It is kept for
  `--kernels shim --device cpu`, where every difference is zero: that is how
  this script is checked on a machine without CUDA. `--pinned` pins PyTorch's
  arithmetic and top-k as the goldens do.

The real kernels on real activations are measured by `real_golden.py
--kernels real --kernel-stats <json>`, which wraps the same comparison around
a whole run over a truncated checkpoint.

A value that is not finite, from either side, is counted on its own and makes
the script exit non-zero after the report is written: it is never a zero
difference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from ds_golden import kernel_shim, pinned_ops, st, synthetic  # noqa: E402
from ds_golden.kernel_compare import failures as kernel_failures  # noqa: E402
from ds_golden.kernel_compare import generated_calls, instrumented, print_stats  # noqa: E402
from ds_golden.reference import Reference, args_from_hf_config, load_reference_modules  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "deepseek_tiny"

def non_finite(*tensors: torch.Tensor) -> int:
    return sum(t.numel() - int(torch.count_nonzero(torch.isfinite(t))) for t in tensors)


def agreement(want: torch.Tensor, got: torch.Tensor) -> dict:
    """Top-1 and KL of `got` against `want`. With a non-finite logit on either
    side there is no agreement to report: only the count."""
    bad = non_finite(want, got)
    if bad:
        return {"rows": want.size(0), "non_finite": bad}
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
        "non_finite": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--inputs", choices=("generated", "fixture"), default="generated")
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
    report = {"kernels": stats, "end_to_end": {}, "inputs": opts.inputs}
    failures = []
    print(f"kernels: {opts.kernels}; inputs: {opts.inputs}; arithmetic: {'pinned' if opts.pinned else 'native'}; device: {opts.device}")
    if opts.inputs == "generated":
        torch.set_default_dtype(torch.bfloat16)
        generated_calls(kernel, opts.device)
    else:
        if opts.pinned:
            pinned_ops.install()
        module = load_reference_modules(opts.reference_dir, kernel)
        config = json.loads((opts.fixture / "config.json").read_text())
        golden = {k: v.tensor() for k, v in st.read_file(opts.fixture / "golden.safetensors").items()}
        tokens = golden["tokens"].tolist()
        tokenizer = synthetic.SyntheticTokenizer(config["text_config"]["vocab_size"])
        reference = Reference(module, args_from_hf_config(config), tokenizer, max_seq_len=128, device=opts.device)
        reference.place(st.read_file(opts.fixture / "model.safetensors"))
        for pattern, prefill in (("step", 1), ("prefill", len(tokens))):
            print(f"{time.strftime('%H:%M:%S')} running the {pattern} pattern", flush=True)
            out = {k: v.cpu() for k, v in reference.run(tokens, prefill=prefill).items()}
            entry = {"logits": agreement(golden[f"{pattern}.logits"], out["logits"].float())}
            for i in range(config["text_config"]["num_hidden_layers"]):
                want, got = golden[f"{pattern}.layer.{i}"].float(), out[f"layer.{i}"].float()
                finite = torch.isfinite(want) & torch.isfinite(got)
                entry[f"layer.{i}"] = {
                    "differing": int(torch.count_nonzero(want != got)),
                    "elements": want.numel(),
                    "non_finite": non_finite(want, got),
                    "max_abs_diff": float(torch.where(finite, (want - got).abs(), torch.zeros_like(want)).max()),
                }
            report["end_to_end"][pattern] = entry

    print_stats(stats)
    failures += kernel_failures(stats)
    for pattern, entry in report["end_to_end"].items():
        a = entry["logits"]
        if a["non_finite"]:
            print(f"{pattern}: {a['non_finite']} non-finite logits; no agreement to report")
        else:
            print(f"{pattern}: top-1 {a['top1']:.4%}  KL mean {a['kl_mean']:.3e} max {a['kl_max']:.3e}  max |dlogit| {a['max_abs_diff']:.3e}")
        for key, value in entry.items():
            if value["non_finite"]:
                failures.append(f"{pattern} {key}: {value['non_finite']} non-finite elements")
            if key != "logits":
                print(f"  {key}: {value['differing']} of {value['elements']} differ ({value['non_finite']} non-finite), max |diff| {value['max_abs_diff']:.4g}")
    report["failures"] = failures
    if opts.json:
        opts.json.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        sys.exit("FAILED, non-finite values:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
