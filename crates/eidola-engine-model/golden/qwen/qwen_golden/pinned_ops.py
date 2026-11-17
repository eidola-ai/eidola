"""The released model code's unspecified operations, pinned to the engine's rules.

Transformers' `qwen3_5` modelling code (v5.23.0) states most of its rounding
points itself: which tensors are BF16, where a norm casts, that β is the
sigmoid of a BF16 tensor. What it leaves to PyTorch, the C library or a kernel
is replaced here, in the namespaces the model code reads it from, with the rule
`eidola-engine-model` fixes. The model code itself is not touched. With these
installed its output is a function of the tokens alone, and the Rust forward
reproduces it bit for bit.

| Operation | Rule |
|---|---|
| `F.linear` (BF16) | f32 dot products in the pinned order (16 interleaved partial sums, a halving tree, the tail); the result rounded once to BF16 |
| `fp8_linear` (the hub Triton kernel) | per token and 128 columns `s = amax / 448`, codes `e4m3(x / s)` nearest-even and saturating (a NaN gives +0); each 128-term block dot exact, rounded once to f32; blocks ascending as `acc + (dot · s) · w_scale` in f32; BF16 out |
| `F.conv1d` (depthwise) | per channel the products in tap order, oldest first, summed left to right in f32; rounded once to the input dtype |
| `Tensor.sum`, `Tensor.mean` (float, over one dim) | f32 in the pinned order; a mean divides that sum by the count |
| `torch.rsqrt` | `1 / sqrt(x)`, two correctly rounded f32 operations |
| `Tensor.exp`, `sigmoid`, `F.silu`, `F.softplus` | f64 from `det_exp` / `det_ln`, rounded to f32 (then to the input dtype); softplus is `x` above 20 |
| `Tensor.cos`, `Tensor.sin` | `det_sin_cos` of the f32 argument in f64, rounded to f32 |
| `scalar ** tensor` (the RoPE frequencies) | `det_exp(e · det_ln(base))`, rounded to f32 |
| `eager_attention_forward` | per query and head: scores are pinned-order dots times the scale over the causally visible keys; minus their maximum, `exp`, normaliser summed left to right, output `Σ (w / Σw) · v` in ascending position; rounded once to BF16 |
| `torch_chunk_gated_delta_rule` | replaced by the model code's own `torch_recurrent_gated_delta_rule`, so every token, prefill included, runs the token recurrence |

`check_environment()` refuses to run when a package that would replace the
model code's own PyTorch paths (FLA, the hub `kernels`, `causal_conv1d`) is
importable.
"""

from __future__ import annotations

import importlib.util
import struct

import torch
import torch.nn.functional as F

F32, F64, BF16 = torch.float32, torch.float64, torch.bfloat16

_native = {
    "sum": torch.Tensor.sum,
    "mean": torch.Tensor.mean,
    "exp": torch.Tensor.exp,
    "sigmoid": torch.Tensor.sigmoid,
    "cos": torch.Tensor.cos,
    "sin": torch.Tensor.sin,
    "rsqrt": torch.rsqrt,
    "sqrt": torch.sqrt,
    "linear": F.linear,
    "conv1d": F.conv1d,
}


def _f64_bits(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


_EXP_COEFFS = [
    0x3FF0000000000000, 0x3FF0000000000000, 0x3FE0000000000000, 0x3FC5555555555555,
    0x3FA5555555555555, 0x3F81111111111111, 0x3F56C16C16C16C17, 0x3F2A01A01A01A01A,
    0x3EFA01A01A01A01A, 0x3EC71DE3A556C734, 0x3E927E4FB7789F5C, 0x3E5AE64567F544E4,
    0x3E21EED8EFF8D898, 0x3DE6124613A86D09,
]  # fmt: skip


def det_exp(x: torch.Tensor) -> torch.Tensor:
    """`e^x` in f64 from correctly rounded + - * only (the engine's `det_exp`)."""
    x = x.to(F64)
    inv_ln2, ln2_hi, ln2_lo = (_f64_bits(b) for b in (0x3FF71547652B82FE, 0x3FE62E42FEE00000, 0x3DEA39EF35793C76))
    tiny = x < -708.0
    xs = torch.where(tiny, torch.zeros_like(x), x)
    t = xs * inv_ln2
    n = torch.copysign(torch.floor(t.abs() + 0.5), t)
    r = (xs - n * ln2_hi) - n * ln2_lo
    p = torch.full_like(r, _f64_bits(_EXP_COEFFS[13]))
    for c in reversed(_EXP_COEFFS[:13]):
        p = p * r + _f64_bits(c)
    out = torch.ldexp(p, n.to(torch.int64))
    out = torch.where(tiny, torch.zeros_like(out), out)
    return torch.where(x > 709.782712893384, torch.full_like(out, torch.inf), out)


LN2_HI = _f64_bits(0x3FE62E42FEE00000)
LN2_LO = _f64_bits(0x3DEA39EF35793C76)
SQRT2 = 1.4142135623730951


def det_ln(x: torch.Tensor) -> torch.Tensor:
    """Natural log of a positive normal f64, from + - * / alone."""
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


_INV_PIO2 = _f64_bits(0x3FE45F306DC9C883)
_PIO2 = (_f64_bits(0x3FF921FB54400000), _f64_bits(0x3DD0B4611A600000), _f64_bits(0x3BA3198A2E000000))
_SIN = [_f64_bits(b) for b in (0xBFC5555555555549, 0x3F8111111110F8A6, 0xBF2A01A019C161D5,
                               0x3EC71DE357B1FE7D, 0xBE5AE5E68A2B9CEB, 0x3DE5D93A5ACFD57C)]  # fmt: skip
_COS = [_f64_bits(b) for b in (0x3FA555555555554C, 0xBF56C16C16C15177, 0x3EFA01A019CB1590,
                               0xBE927E4F809C52AD, 0x3E21EE9EBDB4B1C4, 0xBDA8FAE9BE8838D4)]  # fmt: skip


def det_sin_cos(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """sin and cos of an f64 up to about 1.6e6 in magnitude, from + - * alone."""
    x = x.to(F64)
    assert bool((x.abs() < 1.6e6).all()), "det_sin_cos argument out of range"
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


def _pinned(d, shape, terms):
    full = d // 16 * 16
    acc = torch.zeros(*shape, 16, dtype=F32)
    for lo in range(0, full, 16):
        acc = acc + terms(lo, lo + 16)
    w = 8
    while w:
        acc = acc[..., :w] + acc[..., w : 2 * w]
        w //= 2
    tail = torch.zeros(shape, dtype=F32)
    if full < d:
        rest = terms(full, d)
        for j in range(d - full):
            tail = tail + rest[..., j]
    return acc[..., 0] + tail


def pinned_sum(x: torch.Tensor) -> torch.Tensor:
    """f32 sum over the last dim in the engine's order."""
    x = x.to(F32)
    return _pinned(x.size(-1), x.shape[:-1], lambda lo, hi: x[..., lo:hi])


def pinned_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """f32 dot product over the last dim (broadcasting), in the pinned order."""
    a, b = a.to(F32), b.to(F32)
    shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    return _pinned(a.size(-1), shape, lambda lo, hi: a[..., lo:hi] * b[..., lo:hi])


def exp32(x: torch.Tensor) -> torch.Tensor:
    return det_exp(x).to(F32)


def _float_out(v: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """An f32 result, then the input's dtype (one more rounding for BF16)."""
    return v.to(F32).to(like.dtype)


def tensor_exp(self):
    assert self.is_floating_point()
    return _float_out(det_exp(self), self)


def sigmoid(self):
    return _float_out(1.0 / (1.0 + det_exp(-self.to(F64))), self)


def silu(x, inplace=False):
    assert not inplace
    x64 = x.to(F64)
    return _float_out(x64 / (1.0 + det_exp(-x64)), x)


def softplus(x, beta=1.0, threshold=20.0):
    assert beta == 1.0 and threshold == 20.0 and x.dtype == F32
    e = det_exp(x)
    small = e < 2.0**-20
    big = det_ln(torch.where(small, torch.ones_like(e), 1.0 + e))
    v = torch.where(small, e - 0.5 * e * e, big).to(F32)
    return torch.where(x > 20.0, x, v)


def rsqrt(x):
    assert x.dtype == F32
    return 1.0 / _native["sqrt"](x)


def tensor_cos(self):
    assert self.dtype == F32
    return det_sin_cos(self)[1].to(F32)


def tensor_sin(self):
    assert self.dtype == F32
    return det_sin_cos(self)[0].to(F32)


def pow32(base: float, exponent: torch.Tensor) -> torch.Tensor:
    ln_base = det_ln(torch.tensor(float(base), dtype=F64))
    return det_exp(exponent.to(F64) * ln_base).to(F32)


def tensor_rpow(self, base):
    assert self.dtype == F32 and isinstance(base, (int, float))
    return pow32(base, self)


def _reduce(x, args, kwargs, mean, native):
    dim = args[0] if args else kwargs.get("dim")
    if not x.is_floating_point():
        return native(x, *args, **kwargs)
    keepdim = args[1] if len(args) > 1 else kwargs.get("keepdim", False)
    assert isinstance(dim, int) and len(args) + len(kwargs) <= 2 and set(kwargs) <= {"dim", "keepdim"}, (
        "a float reduction the pins do not cover"
    )
    assert x.dtype in (F32, BF16), x.dtype
    v = pinned_sum(x.movedim(dim, -1))
    if mean:
        v = v / x.size(dim)
    if keepdim:
        v = v.unsqueeze(dim)
    return v.to(x.dtype)


def tensor_sum(self, *args, **kwargs):
    return _reduce(self, args, kwargs, False, _native["sum"])


def tensor_mean(self, *args, **kwargs):
    return _reduce(self, args, kwargs, True, _native["mean"])


# Rows handled per pass by `linear`, to bound the [rows, out, 16] working set.
_LINEAR_ELEMENTS = 1 << 22


def linear(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    assert bias is None
    assert x.dtype == weight.dtype and x.dtype in (F32, BF16), (x.dtype, weight.dtype)
    k = x.size(-1)
    xs, w = x.reshape(-1, k).float(), weight.float()
    out = torch.empty(xs.size(0), w.size(0), dtype=F32)
    step = max(1, _LINEAR_ELEMENTS // w.size(0))
    for m0 in range(0, xs.size(0), step):
        out[m0 : m0 + step] = pinned_dot(xs[m0 : m0 + step, None, :], w[None, :, :])
    return out.to(x.dtype).view(*x.shape[:-1], w.size(0))


def _e4m3_value(code: int) -> float:
    exp, man = (code >> 3) & 0xF, code & 7
    return man / 8 * 2.0**-6 if exp == 0 else (1 + man / 8) * 2.0 ** (exp - 7)


# Non-negative code points in code order, so the index parity is the code parity.
E4M3_TABLE = torch.tensor([_e4m3_value(c) for c in range(127)], dtype=F32)


def round_e4m3(v: torch.Tensor) -> torch.Tensor:
    """f32 -> nearest E4M3 value, ties to the even code, saturating at 448; a
    NaN gives +0."""
    nan = torch.isnan(v)
    v = torch.where(nan, torch.zeros_like(v), v).clamp(-448.0, 448.0)
    a = v.abs().contiguous()
    table = E4M3_TABLE
    hi = torch.searchsorted(table, a).clamp(max=table.numel() - 1)
    lo = (hi - 1).clamp(min=0)
    d_lo, d_hi = a - table[lo], table[hi] - a
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    q = torch.copysign(table[torch.where(pick_hi, hi, lo)], v)
    return torch.where(nan, torch.zeros_like(q), q)


BLOCK = 128


def act_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[M, K] -> E4M3 values [M, K] (f32, not multiplied back) and scales [M, K/128]."""
    m, k = x.shape
    z = x.float().view(m, k // BLOCK, BLOCK)
    s = z.abs().amax(-1) / torch.tensor(448.0, dtype=F32)
    q = round_e4m3(z / s.unsqueeze(-1))
    return q.view(m, k), s


def fp8_linear(input, weight, weight_scale_inv, block_size=None, bias=None, activation_scale=None, allow_deepgemm=True):
    assert bias is None and activation_scale is None and tuple(block_size) == (BLOCK, BLOCK)
    assert weight.dtype == torch.float8_e4m3fn and input.dtype == BF16
    n, k = weight.shape
    x = input.reshape(-1, k)
    q, sa = act_quant(x)
    w = weight.float().to(F64).view(n, k // BLOCK, BLOCK)
    sw = weight_scale_inv.float().repeat_interleave(BLOCK, dim=0)[:n]  # [n, k/128]
    acc = torch.zeros(x.size(0), n, dtype=F32)
    qv = q.to(F64).view(-1, k // BLOCK, BLOCK)
    for kb in range(k // BLOCK):
        dot = (qv[:, kb] @ w[:, kb].T).to(F32)
        acc = acc + (dot * sa[:, kb, None]) * sw[None, :, kb]
    return acc.to(BF16).view(*input.shape[:-1], n)


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    b, c, length = input.shape
    assert bias is None and stride == 1 and dilation == 1 and groups == c and weight.shape[:2] == (c, 1)
    kw = weight.size(-1)
    x = torch.nn.functional.pad(input.float(), (padding, padding))
    w = weight.float()[:, 0, :]
    out_len = length + 2 * padding - kw + 1
    acc = w[None, :, 0, None] * x[:, :, 0:out_len]
    for j in range(1, kw):
        acc = acc + w[None, :, j, None] * x[:, :, j : j + out_len]
    return acc.to(input.dtype)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """[B, H, Lq, D] queries over [B, KVH, Lk, D] keys and values: causal by
    position (query `r` sees keys up to `Lk - Lq + r`), output [B, Lq, H, D]."""
    b, h, lq, d = query.shape
    lk = key.size(2)
    rep = h // key.size(1)
    allowed = torch.arange(lk)[None, :] <= (lk - lq + torch.arange(lq))[:, None]  # [Lq, Lk]
    if attention_mask is not None:
        given = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
        assert bool((given.expand(b, 1, lq, lk)[:, 0] == allowed).all()), "the model's mask is not causal by position"
    scale = torch.tensor(scaling, dtype=F32)
    out = torch.empty(b, lq, h, d, dtype=query.dtype)
    kf, vf = key.float(), value.float()
    for head in range(h):
        g = head // rep
        s = pinned_dot(query[:, head, :, None, :].float(), kf[:, g, None, :, :]) * scale  # [B, Lq, Lk]
        s = torch.where(allowed, s, torch.full_like(s, -torch.inf))
        m = s.amax(-1, keepdim=True)
        e = exp32(s - m)
        den = torch.zeros(b, lq, dtype=F32)
        for j in range(lk):
            den = den + e[:, :, j]
        acc = torch.zeros(b, lq, d, dtype=F32)
        for j in range(lk):
            p = e[:, :, j] / den
            acc = acc + p[:, :, None] * vf[:, g, None, j, :]
        out[:, :, head] = acc.to(query.dtype)
    return out, None


def check_environment() -> None:
    for name in ("fla", "kernels", "causal_conv1d"):
        if importlib.util.find_spec(name) is not None:
            raise SystemExit(f"{name} is importable: it would replace the model code's own PyTorch paths")


def install() -> None:
    """Pin the operations listed in the module docstring, process-wide. Call
    before building a model: RoPE frequencies are computed at construction."""
    check_environment()
    from transformers.integrations import finegrained_fp8
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    F.linear = linear
    F.conv1d = conv1d
    F.silu = silu
    F.softplus = softplus
    torch.sigmoid = sigmoid
    torch.rsqrt = rsqrt
    torch.Tensor.sigmoid = sigmoid
    torch.Tensor.exp = tensor_exp
    torch.Tensor.cos = tensor_cos
    torch.Tensor.sin = tensor_sin
    torch.Tensor.sum = tensor_sum
    torch.Tensor.mean = tensor_mean
    torch.Tensor.__rpow__ = tensor_rpow
    finegrained_fp8.fp8_linear = fp8_linear
    mq.eager_attention_forward = eager_attention_forward
    recurrent = mq.torch_recurrent_gated_delta_rule

    def chunk_as_recurrent(query, key, value, g, beta, chunk_size=64, initial_state=None,
                           output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        return recurrent(query, key, value, g=g, beta=beta, initial_state=initial_state,
                         output_final_state=output_final_state,
                         use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)

    mq.torch_chunk_gated_delta_rule = chunk_as_recurrent
