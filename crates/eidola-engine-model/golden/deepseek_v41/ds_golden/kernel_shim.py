"""Pure-torch CPU emulation of the reference's `inference/kernel.py`.

The reference kernels are tilelang CUDA kernels. This module has the same
function signatures and computes what the kernel source specifies, so the
unmodified reference `model.py` runs on top of it on a CPU.

Where the kernel source does not determine the bits (accumulation order inside
`T.gemm` and the reductions, float -> FP8/FP4 cast rounding, `exp`, `sigmoid`),
this module takes the rule `eidola-engine-model` fixes, so the Rust kernels
reproduce these functions bit for bit:

- a float -> FP8 / FP4 cast rounds to nearest, ties to the even code, after
  the kernel's own clamp;
- the 32-wide block dot products of the quantised GEMMs are exact (they are
  sums of small integers) and rounded once to f32; blocks accumulate in
  ascending order as the kernel does;
- attention scores are f32 dot products in the engine's pinned order (16
  interleaved partial sums, a halving tree, then the tail); every other sum is
  left to right;
- `exp` is the engine's `det_exp` evaluated in f64 and rounded to f32, and
  `sigmoid(x)` is `1 / (1 + det_exp(-x))` in f64 rounded to f32;
- no multiply-add is fused.

`compare_real_kernels.py` measures how far the real kernels are from this.
"""

from __future__ import annotations

import struct

import torch

F32 = torch.float32
F64 = torch.float64


def _f32(x: float) -> torch.Tensor:
    return torch.tensor(x, dtype=F32)


def _e4m3_value(code: int) -> float:
    exp, man = (code >> 3) & 0xF, code & 7
    return man / 8 * 2.0**-6 if exp == 0 else (1 + man / 8) * 2.0 ** (exp - 7)


# Non-negative code points in code order, so the index parity is the code parity.
E4M3_TABLE = torch.tensor([_e4m3_value(c) for c in range(127)], dtype=F32)
E2M1_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=F32)
E2M1_SIGNED = torch.cat([E2M1_TABLE, -E2M1_TABLE])


def round_to_codes(v: torch.Tensor, table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest code point of `table` to each `|v|` (ties to the even code),
    returned as (signed value, code index). `v` is f32 within the table's range."""
    a = v.abs().contiguous()
    hi = torch.searchsorted(table, a).clamp(max=table.numel() - 1)
    lo = (hi - 1).clamp(min=0)
    d_lo, d_hi = a - table[lo], table[hi] - a
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    code = torch.where(pick_hi, hi, lo)
    return torch.copysign(table[code], v), code


def round_scale(amax: torch.Tensor, inv: float) -> torch.Tensor:
    """`fast_round_scale`: 2^ceil(log2(amax * inv)) by the kernel's bit arithmetic."""
    bits = (amax * _f32(inv)).view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    e = exp - 127 + (man != 0).to(torch.int32)
    return ((e + 127) << 23).view(F32)


def _e8m0(scale: torch.Tensor) -> torch.Tensor:
    """A power-of-two f32 as a float8_e8m0fnu tensor."""
    exp = ((scale.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    return exp.view(torch.float8_e8m0fnu)


def _to_scale_dtype(s: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float8_e8m0fnu:
        return _e8m0(s)
    if dtype == torch.float8_e4m3fn:
        return s.to(torch.float8_e4m3fn)  # already an E4M3 value
    return s.to(dtype)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=F32, inplace=False):
    n = x.size(-1)
    assert n % block_size == 0
    z = x.contiguous().float().view(-1, n // block_size, block_size)
    amax = z.abs().amax(-1).clamp_min(_f32(1e-4))
    s = round_scale(amax, 1 / 448.0) if scale_fmt is not None else amax * _f32(1 / 448.0)
    q, _ = round_to_codes((z / s.unsqueeze(-1)).clamp(-448.0, 448.0), E4M3_TABLE)
    if inplace:
        x.copy_((q * s.unsqueeze(-1)).view(x.shape).to(x.dtype))
        return x
    y = q.view(x.shape).to(torch.float8_e4m3fn)
    return y, _to_scale_dtype(s.view(*x.shape[:-1], n // block_size), scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    n = x.size(-1)
    assert n % block_size == 0
    z = x.contiguous().float().view(-1, n // block_size, block_size)
    amax = z.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(_f32(6 * 2.0**-9))
        s, _ = round_to_codes((amax / _f32(6.0)).clamp(max=448.0), E4M3_TABLE)
    else:
        amax = amax.clamp_min(_f32(6 * 2.0**-126))
        s = round_scale(amax, 1.0 / 6.0)
    q, code = round_to_codes((z / s.unsqueeze(-1)).clamp(-6.0, 6.0), E2M1_TABLE)
    if inplace:
        x.copy_((q * s.unsqueeze(-1)).view(x.shape).to(x.dtype))
        return x
    code = (code + 8 * torch.signbit(q).to(code.dtype)).view(-1, n // 2, 2)
    packed = (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8)
    y = packed.view(*x.shape[:-1], n // 2).view(torch.float4_e2m1fn_x2)
    return y, _to_scale_dtype(s.view(*x.shape[:-1], n // block_size), scale_dtype)


# Dense FP8 weights decoded once to f32, keyed by the weight tensor's identity.
# A weight is a model parameter that lives as long as the process (the list
# holds it), so an entry never goes stale.
_dense_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_keepalive: list[torch.Tensor] = []


def clear_cache() -> None:
    """Forget the decoded weights of a model that is being freed."""
    _dense_cache.clear()
    _keepalive.clear()


def _block_gemm(a, a_s, w, sb_rows, block):
    """Sum over K blocks, ascending, of (exact block dot -> f32) * sa * sb, in f32.

    `w` is the decoded weight [N, K/block, block] in f32. A block dot is a sum of
    32 products of small multiples of powers of two: exact in f64 in any order."""
    k = a.size(-1)
    m = a.numel() // k
    nb = k // block
    av = a.float().view(m, nb, block).to(F64)
    sa = a_s.float().reshape(m, nb)
    acc = torch.zeros(m, w.size(0), dtype=F32)
    for kb in range(nb):
        dot = (av[:, kb] @ w[:, kb].to(F64).T).to(F32)
        acc += dot * sa[:, kb, None] * sb_rows[None, :, kb]
    return acc


def fp8_gemm(a, a_s, b, b_s, scale_dtype=F32, block_size=128):
    assert block_size in (32, 128)
    k = a.size(-1)
    n = b.size(0)
    assert k % block_size == 0
    assert b_s.shape == ((n + block_size - 1) // block_size, k // block_size)
    cached = _dense_cache.get(id(b))
    if cached is None:
        w = b.float().view(n, k // block_size, block_size)
        sb_rows = b_s.float().repeat_interleave(block_size, dim=0)[:n].contiguous()
        cached = _dense_cache[id(b)] = (w, sb_rows)
        _keepalive.append(b)
    w, sb_rows = cached
    acc = _block_gemm(a, a_s, w, sb_rows, block_size)
    return acc.to(torch.get_default_dtype()).view(*a.shape[:-1], n)


_NIBBLES = torch.tensor(
    [[E2M1_SIGNED[b & 0xF].item(), E2M1_SIGNED[b >> 4].item()] for b in range(256)], dtype=F32
)


def unpack_fp4(b: torch.Tensor) -> torch.Tensor:
    """Packed float4_e2m1fn_x2 [N, K/2] -> f32 values [N, K]; element 2i is the low nibble."""
    return _NIBBLES[b.contiguous().view(torch.uint8).to(torch.int64)].flatten(-2)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=F32, act_block_size=128):
    k = a.size(-1)
    n = b.size(0)
    assert act_block_size == 32, "only 32-wide activation blocks are emulated"
    assert k % act_block_size == 0
    assert b_s.shape == (n, k // 32)
    w = unpack_fp4(b).view(n, k // 32, 32)
    acc = _block_gemm(a, a_s, w, b_s.float(), 32)
    return acc.to(torch.get_default_dtype()).view(*a.shape[:-1], n)


_EXP_COEFFS = [
    0x3FF0000000000000, 0x3FF0000000000000, 0x3FE0000000000000, 0x3FC5555555555555,
    0x3FA5555555555555, 0x3F81111111111111, 0x3F56C16C16C16C17, 0x3F2A01A01A01A01A,
    0x3EFA01A01A01A01A, 0x3EC71DE3A556C734, 0x3E927E4FB7789F5C, 0x3E5AE64567F544E4,
    0x3E21EED8EFF8D898, 0x3DE6124613A86D09,
]  # fmt: skip


def _f64_bits(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def det_exp(x: torch.Tensor) -> torch.Tensor:
    """`e^x` in f64 from correctly rounded + - * only (the engine's `det_exp`).

    n = round-half-away(x / ln 2); r = (x - n ln2_hi) - n ln2_lo; a degree-13
    Horner polynomial; then a power-of-two scale. Arguments below -708 return 0,
    which every caller rounds to f32 anyway."""
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


def exp32(x: torch.Tensor) -> torch.Tensor:
    return det_exp(x).to(F32)


def sigmoid32(x: torch.Tensor) -> torch.Tensor:
    return (1.0 / (1.0 + det_exp(-x.to(F64)))).to(F32)


def pinned_sum(x: torch.Tensor) -> torch.Tensor:
    """f32 sum over the last dim in the engine's order: 16 interleaved partial
    sums over the full 16-wide chunks, a halving tree over them, then the tail
    left to right. Under 16 elements that is a plain left-to-right sum."""
    return _pinned(x.size(-1), x.shape[:-1], lambda lo, hi: x[..., lo:hi])


def pinned_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """f32 dot product over the last dim (broadcasting), products summed in
    [`pinned_sum`] order."""
    shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    return _pinned(a.size(-1), shape, lambda lo, hi: a[..., lo:hi] * b[..., lo:hi])


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


ATTN_BLOCK = 64
# Queries handled per pass, to bound the gathered [queries, 64, d] working set.
_ATTN_QUERY_CHUNK = 32


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    assert b == 1, "the emulation runs one sequence"
    out = torch.empty_like(q)
    scale = _f32(softmax_scale)
    kvf = kv[0].float()
    sink = attn_sink.float()
    for m0 in range(0, m, _ATTN_QUERY_CHUNK):
        qf = q[0, m0 : m0 + _ATTN_QUERY_CHUNK].float()
        idx_all = topk_idxs[0, m0 : m0 + _ATTN_QUERY_CHUNK].to(torch.int64)
        mc = qf.size(0)
        smax = torch.full((mc, h), -1e30, dtype=F32)
        den = torch.zeros(mc, h, dtype=F32)
        acc = torch.zeros(mc, h, d, dtype=F32)
        for t0 in range(0, idx_all.size(1), ATTN_BLOCK):
            idx = idx_all[:, t0 : t0 + ATTN_BLOCK]
            valid = idx != -1
            if not valid.any():
                continue
            rows = kvf[idx.clamp(min=0)]  # [mc, blk, d]
            s = pinned_dot(qf[:, :, None, :], rows[:, None, :, :]) * scale  # [mc, h, blk]
            s = torch.where(valid[:, None, :], s, torch.full_like(s, -torch.inf))
            prev = smax
            smax = torch.maximum(smax, s.amax(-1))
            r = exp32(prev - smax)
            p = exp32(s - smax.unsqueeze(-1))
            block_sum = torch.zeros(mc, h, dtype=F32)
            acc = acc * r.unsqueeze(-1)
            p_cast = p.to(torch.bfloat16).float()
            for j in range(idx.size(1)):
                block_sum = block_sum + p[:, :, j]
                acc = acc + p_cast[:, :, j, None] * rows[:, None, j, :]
            den = den * r + block_sum
        den = den + exp32(sink[None, :] - smax)
        out[0, m0 : m0 + mc] = (acc / den.unsqueeze(-1)).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    b, s, _ = mixes.size()
    hc = hc_mult
    mx = mixes.float().view(-1, (2 + hc) * hc)
    eps = _f32(eps)
    pre = sigmoid32(mx[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * sigmoid32(mx[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mx[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).view(-1, hc, hc)

    def row_sum(c):
        acc = torch.zeros_like(c[:, :, 0])
        for k in range(hc):
            acc = acc + c[:, :, k]
        return acc.unsqueeze(-1)

    def col_sum(c):
        acc = torch.zeros_like(c[:, 0, :])
        for j in range(hc):
            acc = acc + c[:, j, :]
        return acc.unsqueeze(-2)

    comb = exp32(comb - comb.amax(-1, keepdim=True))
    comb = comb / row_sum(comb) + eps
    comb = comb / (col_sum(comb) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (row_sum(comb) + eps)
        comb = comb / (col_sum(comb) + eps)
    return pre.view(b, s, hc), post.view(b, s, hc), comb.view(b, s, hc, hc)
