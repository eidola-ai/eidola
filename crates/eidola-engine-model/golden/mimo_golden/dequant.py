"""Checkpoint tensors -> fp32 torch tensors, written independently of the Rust
loader (vectorised gathers instead of per-row index maps), so the two can
check each other.
"""

from __future__ import annotations

import torch

BLOCK = 128

E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def fp8_block(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """FP8 e4m3 [R, C] with [ceil(R/128), ceil(C/128)] F32 scales."""
    r, c = w.shape
    assert scale_inv.shape == ((r + BLOCK - 1) // BLOCK, (c + BLOCK - 1) // BLOCK), (
        w.shape,
        scale_inv.shape,
    )
    s = scale_inv.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[:r, :c]
    return w.float() * s


def fused_qkv(
    w: torch.Tensor,
    scale_inv: torch.Tensor | None,
    n_q: int,
    n_kv: int,
    head_dim: int,
    v_head_dim: int,
    chunks: int,
) -> torch.Tensor:
    """Rank-major pre-sharded fused QKV -> unsharded [Q | K | V].

    The checkpoint stores `chunks` slices [Q_c | K_c | V_c]; the FP8 scale grid
    is tiled per slice (ceil(rows_per_slice / 128) scale rows each, the last
    one partial when the slice is not a multiple of 128 rows).
    """
    total = w.shape[0]
    assert total == n_q * head_dim + n_kv * (head_dim + v_head_dim)
    per = total // chunks
    q_per = n_q * head_dim // chunks
    k_per = n_kv * head_dim // chunks
    if scale_inv is None:
        deq = w.float()
    else:
        bpc = (per + BLOCK - 1) // BLOCK
        assert scale_inv.shape[0] == chunks * bpc, (scale_inv.shape, chunks, bpc)
        parts = []
        for c in range(chunks):
            wc = w[c * per : (c + 1) * per]
            sc = scale_inv[c * bpc : (c + 1) * bpc]
            parts.append(fp8_block(wc, sc))
        deq = torch.cat(parts, 0)
    qs, ks, vs = [], [], []
    for c in range(chunks):
        base = c * per
        qs.append(deq[base : base + q_per])
        ks.append(deq[base + q_per : base + q_per + k_per])
        vs.append(deq[base + q_per + k_per : base + per])
    return torch.cat(qs + ks + vs, 0)


def mxfp4(packed: torch.Tensor, scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Packed E2M1 [R, C/2] (element 2i low nibble) + E8M0 [R, C/block]."""
    r, half = packed.shape
    p = packed.to(torch.int64)
    codes = torch.stack((p & 0x0F, p >> 4), dim=-1).reshape(r, half * 2)
    vals = E2M1[codes]
    exp = scale.to(torch.int64) - 127
    s = torch.pow(torch.tensor(2.0, dtype=torch.float64), exp.double()).float()
    return vals * s.repeat_interleave(block, 1)


def linear(get, prefix: str) -> torch.Tensor:
    """`{prefix}.weight`, dequantised if it carries FP8 or MXFP4 scales."""
    w = get(f"{prefix}.weight")
    if w.dtype == torch.float8_e4m3fn:
        return fp8_block(w, get(f"{prefix}.weight_scale_inv"))
    if w.dtype == torch.uint8:
        return mxfp4(w, get(f"{prefix}.weight_scale"))
    return w.float()
