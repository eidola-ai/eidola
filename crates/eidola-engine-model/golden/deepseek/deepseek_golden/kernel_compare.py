"""Run a kernel module and the emulation side by side.

`instrumented(real)` is a `kernel` module that gives every call to the real
kernel and, on the same inputs, to `kernel_shim` on the CPU, returns the real
result, and tallies per kernel how many elements differ and by how much. A
value that is not finite on either side is a failure of its own, never a zero
difference.

`generated_calls` drives each kernel once with seeded inputs at the released
model's dimensions. The real kernels do not compile at every size (tilelang
0.1.12 refuses `sparse_attn` at the synthetic fixture's head dimension of 32),
so a per-kernel comparison uses inputs made here, not the fixture.
"""

from __future__ import annotations

import types

import torch

from . import kernel_shim

KERNELS = ("act_quant", "fp4_act_quant", "fp4_gemm", "fp8_gemm", "hc_split_sinkhorn", "sparse_attn")
BF16 = torch.bfloat16

# Host copies of model parameters, by the parameter's identity: the emulation
# caches decoded weights by tensor identity, so a weight must reach it as the
# same object on every call.
_host_parameters: dict[int, torch.Tensor] = {}


def _host(x):
    if not isinstance(x, torch.Tensor):
        return x
    if isinstance(x, torch.nn.Parameter):
        copy = _host_parameters.get(id(x))
        if copy is None:
            copy = _host_parameters[id(x)] = x.detach().cpu()
        return copy
    return x.detach().cpu().clone()


def _values(out) -> list[torch.Tensor]:
    """Outputs as comparable f32 tensors (FP8 / scale tensors decoded)."""
    outs = out if isinstance(out, tuple) else (out,)
    return [o.detach().cpu().view(torch.uint8).float() if o.dtype == torch.float4_e2m1fn_x2 else o.detach().cpu().float() for o in outs]


def instrumented(real) -> tuple[types.ModuleType, dict]:
    stats = {k: {"calls": 0, "elements": 0, "differing": 0, "non_finite": 0, "max_abs_diff": 0.0} for k in KERNELS}
    module = types.ModuleType("kernel")

    def wrap(name):
        def call(*args, **kwargs):
            host_args = [_host(a) for a in args]
            host_kwargs = {k: _host(v) for k, v in kwargs.items()}
            with torch.device("cpu"):
                want = _values(getattr(kernel_shim, name)(*host_args, **host_kwargs))
            out = getattr(real, name)(*args, **kwargs)
            s = stats[name]
            s["calls"] += 1
            for got, expected in zip(_values(out), want):
                finite = torch.isfinite(got) & torch.isfinite(expected)
                diff = torch.where(finite, (got - expected).abs(), torch.zeros_like(got))
                bad = got.numel() - int(torch.count_nonzero(finite))
                s["elements"] += got.numel()
                s["non_finite"] += bad
                s["differing"] += int(torch.count_nonzero(diff)) + bad
                s["max_abs_diff"] = max(s["max_abs_diff"], float(diff.max()) if diff.numel() else 0.0)
            return out

        return call

    for name in KERNELS:
        setattr(module, name, wrap(name))
    return module, stats


def failures(stats: dict) -> list[str]:
    return [f"{name}: {s['non_finite']} non-finite elements" for name, s in stats.items() if s["non_finite"]]


def print_stats(stats: dict) -> None:
    print(f"{'kernel':<20}{'calls':>8}{'elements':>12}{'differing':>12}{'non-finite':>12}{'max |diff|':>14}")
    for name, s in stats.items():
        print(f"{name:<20}{s['calls']:>8}{s['elements']:>12}{s['differing']:>12}{s['non_finite']:>12}{s['max_abs_diff']:>14.6g}")


def generated_calls(kernel, device: str, seed: int = 20251017) -> None:
    """One call of every kernel on seeded inputs at the released dimensions:
    hidden 5120, 64 heads of 512, expert width 2304, 4 residual copies."""
    g = torch.Generator().manual_seed(seed)

    def randn(*shape, std=1.0):
        return torch.randn(*shape, generator=g, dtype=torch.float32) * std

    def rows(n, width):
        # Rows of very different magnitude, so block scales vary.
        return (randn(n, width) * torch.exp2(randn(n, 1, std=3.0).round())).to(BF16)

    def code_bytes(*shape):
        b = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)
        return torch.where((b & 0x7F) == 0x7F, b & 0x80, b)  # no NaN code

    def scales(*shape):
        return torch.randint(118, 126, shape, generator=g, dtype=torch.uint8).view(torch.float8_e8m0fnu)

    x = rows(32, 5120)
    kernel.act_quant(x.clone().to(device), 32, "ue8m0", torch.float8_e8m0fnu, True)
    codes, act_scales = kernel.act_quant(x.clone().to(device), 32, "ue8m0", torch.float8_e8m0fnu)
    kernel.fp4_act_quant(rows(32, 4096).to(device), 32, True)
    kernel.fp4_act_quant(rows(32, 512).to(device), 16, True, scale_dtype=torch.float8_e4m3fn)

    w8 = code_bytes(1024, 5120).view(torch.float8_e4m3fn).to(device)
    kernel.fp8_gemm(codes, act_scales, w8, scales(32, 160).to(device), torch.float8_e8m0fnu, block_size=32)
    w4 = torch.randint(0, 256, (2304, 2560), generator=g, dtype=torch.uint8).view(torch.float4_e2m1fn_x2).to(device)
    kernel.fp4_gemm(codes, act_scales, w4, scales(2304, 160).to(device), torch.float8_e8m0fnu, act_block_size=32)

    kernel.hc_split_sinkhorn(randn(1, 64, 24, std=2.0).to(device), (0.5 + torch.rand(3, generator=g, dtype=torch.float32)).to(device), randn(24, std=0.5).to(device), 4, 20, 1e-6)

    heads, d, queries, bank = 64, 512, 8, 700
    idxs = torch.full((1, queries, 640), -1, dtype=torch.int32)
    for m in range(queries):
        # A window that fills up, then global entries, with empty slots between.
        window = torch.arange(max(0, 16 * (m + 1) - 128), 16 * (m + 1))
        idxs[0, m, 128 - window.numel() : 128] = window
        chosen = torch.randperm(bank - 128, generator=g)[: 60 * m].sort().values + 128
        idxs[0, m, 128 : 128 + chosen.numel()] = chosen
    kernel.sparse_attn(randn(1, queries, heads, d).to(BF16).to(device), randn(1, bank, d).to(BF16).to(device), randn(heads).to(device), idxs.to(device), d**-0.5)
