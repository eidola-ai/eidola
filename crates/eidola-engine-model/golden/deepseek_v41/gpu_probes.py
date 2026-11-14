"""Small probes of the reference's real kernels, answering what their source
leaves open. Each compares the real kernel (`--reference-dir/kernel.py`, on
CUDA) with the emulation (`ds_golden/kernel_shim.py`, on the CPU) and records
what the real kernel did:

1. float -> FP8 cast: every exact tie between adjacent E4M3 codes, both signs.
2. float -> FP4 cast: every exact tie between adjacent E2M1 codes, with the
   power-of-two scale and with the E4M3 scale; the E4M3 scale's own rounding on
   a tie; saturation when `amax / scale` exceeds 6.
3. `torch.topk` on CUDA over equal BF16 scores: which indices win.
4. `sparse_attn`: whether the result depends on which 64-slot block an entry
   falls in, and on empty slots that do not move a block boundary.

    python gpu_probes.py --reference-dir <dir> --json probes.json

`--kernels shim --device cpu` runs the probes against the emulation itself
(how the script is checked without a GPU).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from ds_golden import kernel_shim as ks  # noqa: E402

BF16 = torch.bfloat16


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def classify(value: float, lo: float, hi: float, lo_code: int) -> str:
    """How a tie between codes `lo` and `hi` was resolved."""
    if value == lo:
        return "even" if lo_code % 2 == 0 else "odd"
    if value == hi:
        return "odd" if lo_code % 2 == 0 else "even"
    return f"neither ({value})"


def tally(results: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["resolved"]] = counts.get(r["resolved"], 0) + 1
    return counts


def tie_rows(table: list[float], top: float, width: int) -> tuple[torch.Tensor, list[tuple[int, float, float]]]:
    """Rows of `width` values whose first element is `top` (fixing the block
    scale at 1) and whose others are exact ties between adjacent codes."""
    ties = [(i, table[i], table[i + 1]) for i in range(len(table) - 1)]
    values = [(lo + hi) / 2 for _, lo, hi in ties]
    values = values + [-v for v in values]
    rows = []
    for at in range(0, len(values), width - 1):
        chunk = values[at : at + width - 1]
        rows.append([top] + chunk + [0.0] * (width - 1 - len(chunk)))
    return torch.tensor(rows, dtype=torch.float32), ties


def read_ties(out: torch.Tensor, ties, width: int) -> list[dict]:
    flat = [v for row in out.float().cpu().tolist() for v in row[1:]]
    results = []
    for sign, offset in ((1.0, 0), (-1.0, len(ties))):
        for j, (code, lo, hi) in enumerate(ties):
            got = flat[offset + j] * sign
            results.append({"tie": sign * (lo + hi) / 2, "between": [lo, hi], "got": sign * got, "resolved": classify(got, lo, hi, code)})
    return results


def probe_casts(kernel, device) -> dict:
    report = {}
    e4m3 = ks.E4M3_TABLE.tolist()
    x, ties = tie_rows(e4m3, 448.0, 32)
    assert torch.equal(x.to(BF16).float(), x), "an FP8 tie is not a BF16 value"
    out = kernel.act_quant(x.to(BF16).to(device).clone(), 32, "ue8m0", torch.float8_e8m0fnu, True)
    fp8 = read_ties(out, ties, 32)
    report["fp8_ties"] = {"count": len(fp8), "resolved": tally(fp8), "not_even": [r for r in fp8 if r["resolved"] != "even"][:40]}
    codes, _ = kernel.act_quant(x.to(BF16).to(device).clone(), 32, "ue8m0", torch.float8_e8m0fnu)
    fp8_codes = read_ties(codes.float(), ties, 32)
    report["fp8_ties_code_path"] = {"resolved": tally(fp8_codes)}

    e2m1 = ks.E2M1_TABLE.tolist()
    for name, block, dtype in (("fp4_pow2_scale", 32, torch.float8_e8m0fnu), ("fp4_e4m3_scale", 16, torch.float8_e4m3fn)):
        x, ties = tie_rows(e2m1, 6.0, block)
        out = kernel.fp4_act_quant(x.to(BF16).to(device).clone(), block, True, scale_dtype=dtype)
        fp4 = read_ties(out, ties, block)
        report[f"{name}_ties"] = {"count": len(fp4), "resolved": tally(fp4), "detail": fp4}
    # amax / 6 on an exact tie between two E4M3 scales (1.0 and 1.125), and a
    # scale that rounds down so that amax / scale exceeds 6 and saturates.
    for name, amax in (("scale_tie_amax_6.375", 6.375), ("saturation_amax_7", 7.0)):
        x = torch.zeros(1, 16)
        x[0, 0], x[0, 1], x[0, 2] = amax, 1.0, -amax
        out = kernel.fp4_act_quant(x.to(BF16).to(device).clone(), 16, True, scale_dtype=torch.float8_e4m3fn)
        with torch.device("cpu"):
            want = ks.fp4_act_quant(x.to(BF16).clone(), 16, True, scale_dtype=torch.float8_e4m3fn)
        report[name] = {"input": x[0, :3].tolist(), "real": out.float().cpu()[0, :3].tolist(), "emulation": want.float()[0, :3].tolist()}
    return report


def probe_topk(device) -> dict:
    report = {}
    for n, k in ((8, 3), (64, 16), (1000, 512), (5000, 512), (20000, 512)):
        # Groups of equal values: value i // 7, so 7 candidates tie everywhere.
        scores = (torch.arange(n, device=device) // 7).to(BF16).flip(0).contiguous()
        entry = {}
        for sorted_flag in (False, True):
            picks = [scores.view(1, n).topk(k, dim=-1, sorted=sorted_flag).indices[0].sort().values.cpu().tolist() for _ in range(3)]
            threshold = scores.float().topk(k).values.min().item()
            tied = [i for i in range(n) if scores[i].item() == threshold]
            chosen = [i for i in picks[0] if i in set(tied)]
            entry[f"sorted={sorted_flag}"] = {
                "repeatable": picks[0] == picks[1] == picks[2],
                "tied_at_threshold": len(tied),
                "chosen_among_tied": len(chosen),
                "lowest_indices_win": chosen == tied[: len(chosen)],
                "highest_indices_win": chosen == tied[len(tied) - len(chosen) :],
                "chosen": chosen[:16],
                "tied": tied[:16],
            }
        report[f"n={n},k={k}"] = entry
    return report


def probe_attention(kernel, device) -> dict:
    g = torch.Generator().manual_seed(7)
    heads, d, window, n_window, n_global = 64, 512, 128, 100, 50
    q = torch.randn(1, 1, heads, d, generator=g).to(BF16)
    kv = torch.randn(1, n_window + n_global, d, generator=g).to(BF16)
    sink = torch.randn(heads, generator=g)
    valid = list(range(n_window))
    glob = list(range(n_window, n_window + n_global))
    pad = [-1] * (window - n_window)
    layouts = {
        "window right-aligned (decode)": pad + valid + glob,
        "window left-aligned (prefill)": valid + pad + glob,
        "packed, no empty slots": valid + glob,
        # The same entries in the same 64-slot blocks as the decode layout, with
        # the empty slots moved inside the first block.
        "decode blocks, empty slots moved": valid[:20] + pad + valid[20:] + glob,
    }
    outs, report = {}, {"layouts": {}}
    for name, idx in layouts.items():
        idxs = torch.tensor([[idx]], dtype=torch.int32)
        real = kernel.sparse_attn(q.to(device), kv.to(device), sink.to(device), idxs.to(device), d**-0.5).float().cpu()
        with torch.device("cpu"):
            want = ks.sparse_attn(q.clone(), kv, sink, idxs, d**-0.5).float()
        outs[name] = real
        diff = (real - want).abs()
        report["layouts"][name] = {"elements": real.numel(), "differ_from_emulation": int(torch.count_nonzero(diff)), "max_abs_diff": float(diff.max())}
    base = outs["window right-aligned (decode)"]
    report["real_kernel_against_its_decode_layout"] = {
        name: {"differing": int(torch.count_nonzero(out != base)), "max_abs_diff": float((out - base).abs().max())}
        for name, out in outs.items()
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--kernels", choices=("real", "shim"), default="real")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path)
    opts = parser.parse_args()
    torch.set_default_dtype(BF16)
    if opts.kernels == "real":
        spec = importlib.util.spec_from_file_location("real_kernel", opts.reference_dir / "kernel.py")
        kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernel)
    else:
        kernel = ks
    report = {"kernels": opts.kernels, "device": opts.device, "torch": torch.__version__}
    for name, probe in (("casts", lambda: probe_casts(kernel, opts.device)), ("topk", lambda: probe_topk(opts.device)), ("attention", lambda: probe_attention(kernel, opts.device))):
        log(f"probe: {name}")
        report[name] = probe()
    c = report["casts"]
    print(f"FP8 ties ({c['fp8_ties']['count']}): {c['fp8_ties']['resolved']}; code path: {c['fp8_ties_code_path']['resolved']}")
    for key in ("fp4_pow2_scale_ties", "fp4_e4m3_scale_ties"):
        print(f"{key} ({c[key]['count']}): {c[key]['resolved']}")
    for key in ("scale_tie_amax_6.375", "saturation_amax_7"):
        print(f"{key}: real {c[key]['real']} emulation {c[key]['emulation']}")
    for key, entry in report["topk"].items():
        e = entry["sorted=False"]
        print(f"topk {key}: repeatable {e['repeatable']}, lowest indices win {e['lowest_indices_win']}, highest win {e['highest_indices_win']}")
    for name, entry in report["attention"]["layouts"].items():
        versus = report["attention"]["real_kernel_against_its_decode_layout"][name]
        print(f"sparse_attn, {name}: {entry['differ_from_emulation']} of {entry['elements']} differ from the emulation (max {entry['max_abs_diff']:.3g}); {versus['differing']} differ from the real decode layout")
    if opts.json:
        opts.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
