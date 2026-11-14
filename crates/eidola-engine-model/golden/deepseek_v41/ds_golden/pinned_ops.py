"""The reference's library arithmetic, pinned to the engine's rules.

`model.py` leaves a number of results to PyTorch and the C library: the
summation order of `F.linear`, `einsum`, `sum` and `mean`, the last bit of
`exp`, `log`, `sin` and `cos`, whether a complex multiply is fused, and which of
several equal values `topk` returns. Those bits differ between a CPU and a GPU
and between call shapes, and a one-ulp difference is enough to flip a BF16
rounding or a top-k choice downstream.

`install()` replaces exactly those operations, in the `torch` namespace the
reference reads them from, with the rule `eidola-engine-model` fixes for each.
The reference's model code itself is not touched. With the operations pinned,
its output is a function of the tokens alone, and the Rust forward reproduces
it bit for bit.

| Operation in `model.py` | Rule |
|---|---|
| `F.linear`, `torch.einsum` (BF16 or f32) | f32 dot products in the pinned order; a BF16 result is rounded once |
| `Tensor.sum`, `torch.sum`, `Tensor.mean` | f32, the pinned order over the reduced dim; a BF16 result is rounded once |
| `Tensor.softmax` | subtract the max, `exp`, divide by the pinned-order sum |
| `torch.sigmoid`, `F.silu`, `F.softplus` | f64 from `det_exp` / `det_ln`, rounded to f32 |
| `scalar ** tensor`, `torch.polar` (RoPE table) | f64 `det_exp(e * det_ln(base))`, `det_sin_cos`, rounded to f32 |
| complex multiply (RoPE) | `ac - bs`, `as + bc` in f32, nothing fused |
| `Tensor.topk` | higher value first, then lower index |
"""

from __future__ import annotations

import collections

import torch
import torch.nn.functional as F

from . import kernel_shim as ks

F32, F64 = torch.float32, torch.float64

LN2_HI = ks._f64_bits(0x3FE62E42FEE00000)
LN2_LO = ks._f64_bits(0x3DEA39EF35793C76)
SQRT2 = 1.4142135623730951


def det_ln(x: torch.Tensor) -> torch.Tensor:
    """Natural log of a positive normal f64, from + - * / alone.

    x = m 2^e with m in (sqrt(1/2), sqrt 2]; s = (m - 1) / (m + 1);
    ln m = 2 s (1 + s^2/3 + ... + s^22/23); ln x = e ln2_hi + (ln m + e ln2_lo)."""
    x = x.to(F64)
    bits = x.view(torch.int64)
    e = ((bits >> 52) & 0x7FF) - 1023
    m = ((bits & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000).view(F64)
    big = m > SQRT2
    m = torch.where(big, m * 0.5, m)
    e = (e + big.to(torch.int64)).to(F64)
    s = (m - 1.0) / (m + 1.0)
    z = s * s
    p = torch.full_like(z, 1.0 / 23.0)
    for k in range(10, -1, -1):
        p = p * z + 1.0 / (2 * k + 1)
    return e * LN2_HI + (2.0 * s * p + e * LN2_LO)


def softplus32(x: torch.Tensor) -> torch.Tensor:
    """`x` above 20, else `ln(1 + e^x)` (as `e - e^2/2` while `e^x < 2^-20`)."""
    e = ks.det_exp(x)
    small = e < 2.0**-20
    big = det_ln(torch.where(small, torch.ones_like(e), 1.0 + e))
    v = torch.where(small, e - 0.5 * e * e, big).to(F32)
    return torch.where(x > 20.0, x, v)


def silu32(x: torch.Tensor) -> torch.Tensor:
    x64 = x.to(F64)
    return (x64 / (1.0 + ks.det_exp(-x64))).to(F32)


_INV_PIO2 = 6.36619772367581382433e-01
_PIO2 = (1.57079632673412561417e00, 6.07710050630396597660e-11, 2.02226624871116645580e-21)
_SIN = (-1.66666666666666324348e-01, 8.33333333332248946124e-03, -1.98412698298579493134e-04,
        2.75573137070700676789e-06, -2.50507602534068634195e-08, 1.58969099521155010221e-10)  # fmt: skip
_COS = (4.16666666666666019037e-02, -1.38888888888741095749e-03, 2.48015872894767294178e-05,
        -2.75573143513906633035e-07, 2.08757232129817482790e-09, -1.13596475577881948265e-11)  # fmt: skip


def det_sin_cos(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """sin and cos of an f64 up to about 1.6e6 in magnitude, from + - * alone.

    n = round-half-away(x 2/pi); r = ((x - n p1) - n p2) - n p3 with pi/2 split
    into 33-bit pieces (each n p is exact); degree-13 / degree-14 polynomials in
    r; then the quadrant n mod 4."""
    x = x.to(F64)
    t = x * _INV_PIO2
    n = torch.copysign(torch.floor(t.abs() + 0.5), t)
    r = ((x - n * _PIO2[0]) - n * _PIO2[1]) - n * _PIO2[2]
    z = r * r
    ps = torch.full_like(z, _SIN[5])
    pc = torch.full_like(z, _COS[5])
    for k in range(4, -1, -1):
        ps = ps * z + _SIN[k]
        pc = pc * z + _COS[k]
    sin = r + (r * z) * ps
    cos = (1.0 - 0.5 * z) + (z * z) * pc
    q = n.to(torch.int64) % 4
    s = torch.where(q == 0, sin, torch.where(q == 1, cos, torch.where(q == 2, -sin, -cos)))
    c = torch.where(q == 0, cos, torch.where(q == 1, -sin, torch.where(q == 2, -cos, sin)))
    return s, c


def pow32(base: float, exponent: torch.Tensor) -> torch.Tensor:
    ln_base = det_ln(torch.tensor(float(base), dtype=F64))
    return ks.det_exp(exponent.to(F64) * ln_base).to(F32)


# Rows handled per pass by `linear`, to bound the [rows, out, 16] working set.
_LINEAR_ELEMENTS = 1 << 22


def linear(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    assert bias is None
    assert x.dtype == weight.dtype and x.dtype in (F32, torch.bfloat16), (x.dtype, weight.dtype)
    k = x.size(-1)
    xs, w = x.reshape(-1, k).float(), weight.float()
    out = torch.empty(xs.size(0), w.size(0), dtype=F32)
    step = max(1, _LINEAR_ELEMENTS // w.size(0))
    for m0 in range(0, xs.size(0), step):
        out[m0 : m0 + step] = ks.pinned_dot(xs[m0 : m0 + step, None, :], w[None, :, :])
    return out.to(x.dtype).view(*x.shape[:-1], w.size(0))


def einsum(equation: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dtype == b.dtype
    if equation == "bsgd,grd->bsgr":
        out = torch.stack([linear(a[:, :, g], b[g]) for g in range(b.size(0))], dim=2)
    elif equation == "bshd,btd->bsht":
        assert a.size(0) == 1
        out = linear(a[0], b[0]).unsqueeze(0)
    else:
        raise NotImplementedError(equation)
    return out


def _reduce(x: torch.Tensor, dim: int, keepdim: bool, mean: bool) -> torch.Tensor:
    assert x.dtype in (F32, torch.bfloat16), x.dtype
    v = ks.pinned_sum(x.float().movedim(dim, -1))
    if mean:
        v = v / x.size(dim)
    if keepdim:
        v = v.unsqueeze(dim)
    return v.to(x.dtype)


def tensor_sum(self, dim=None, keepdim=False):
    return _reduce(self, dim, keepdim, mean=False)


def tensor_mean(self, dim=None, keepdim=False):
    return _reduce(self, dim, keepdim, mean=True)


def tensor_softmax(self, dim):
    assert self.dtype == F32
    e = ks.exp32(self - self.amax(dim, keepdim=True))
    return e / _reduce(e, dim, True, mean=False)


def tensor_rpow(self, base):
    assert self.dtype == F32
    return pow32(base, self)


def polar(magnitude: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    assert angle.dtype == F32
    sin, cos = det_sin_cos(angle)
    return torch.complex(magnitude * cos.to(F32), magnitude * sin.to(F32))


_native_mul = torch.Tensor.__mul__


def tensor_mul(self, other):
    if isinstance(other, torch.Tensor) and self.is_complex() and other.is_complex():
        a, b, c, s = self.real, self.imag, other.real, other.imag
        return torch.complex(a * c - b * s, a * s + b * c)
    return _native_mul(self, other)


_Topk = collections.namedtuple("topk", ["values", "indices"])


def tensor_topk(self, k, dim=-1, largest=True, sorted=True):
    assert largest
    order = torch.sort(self.float(), dim=dim, descending=True, stable=True).indices.narrow(dim, 0, k)
    return _Topk(self.gather(dim, order), order)


def install(arithmetic: bool = True, topk: bool = True) -> None:
    """Pin the operations listed in the module docstring, process-wide."""
    if topk:
        torch.Tensor.topk = tensor_topk
    if not arithmetic:
        return
    F.linear = linear
    F.softplus = softplus32
    F.silu = silu32
    torch.einsum = einsum
    torch.sum = tensor_sum
    torch.sigmoid = ks.sigmoid32
    torch.polar = polar
    torch.Tensor.sum = tensor_sum
    torch.Tensor.mean = tensor_mean
    torch.Tensor.softmax = tensor_softmax
    torch.Tensor.__rpow__ = tensor_rpow
    torch.Tensor.__mul__ = tensor_mul
