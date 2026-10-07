//! Exact dequantisation of MiMo's stored weight formats to f32.
//!
//! - **FP8 block-scaled linears**: e4m3 weights, one F32 `weight_scale_inv`
//!   per `[block_rows, block_cols]` tile, `w = q · scale` (the scales are
//!   arbitrary F32, not powers of two). Edge tiles may be partial.
//! - **MXFP4 experts**: two E2M1 codes per byte along K (element `2i` in the
//!   low nibble, `2i+1` in the high nibble), one E8M0 scale per 32 K elements.
//! - **Rank-major pre-sharded fused QKV**: see [`deinterleave_qkv`].

use rayon::prelude::*;

use crate::config::AttentionSpec;
use crate::error::{Error, Result};
use crate::numeric::{E2M1_VALUES, e8m0_to_f32, fp8_e4m3_to_f32};
use crate::tensor::Matrix;

/// Dequantise an FP8 `[rows, cols]` weight whose scale tile for row `r` is
/// row `scale_row_of(r)` of the `[*, ceil(cols / block_cols)]` scale grid.
fn fp8_dequant_with(
    name: &str,
    q: &[u8],
    rows: usize,
    cols: usize,
    scales: &[f32],
    block_cols: usize,
    scale_row_of: impl Fn(usize) -> usize + Sync,
) -> Result<Matrix> {
    if q.len() != rows * cols {
        return Err(Error::layout(
            name,
            format!("{} bytes for {rows}×{cols}", q.len()),
        ));
    }
    let scale_cols = cols.div_ceil(block_cols);
    if !scales.len().is_multiple_of(scale_cols) {
        return Err(Error::layout(name, "scale grid width mismatch"));
    }
    let scale_rows = scales.len() / scale_cols;
    let mut out = Matrix::zeros(rows, cols);
    let mut err = None;
    for r in 0..rows {
        if scale_row_of(r) >= scale_rows {
            err = Some(r);
            break;
        }
    }
    if let Some(r) = err {
        return Err(Error::layout(
            name,
            format!("row {r} maps past the {scale_rows}-row scale grid"),
        ));
    }
    out.data
        .par_chunks_mut(cols.max(1))
        .enumerate()
        .for_each(|(r, orow)| {
            let srow = &scales[scale_row_of(r) * scale_cols..][..scale_cols];
            let qrow = &q[r * cols..(r + 1) * cols];
            for (c, (o, &b)) in orow.iter_mut().zip(qrow).enumerate() {
                *o = fp8_e4m3_to_f32(b) * srow[c / block_cols];
            }
        });
    Ok(out)
}

/// Plain block-scaled FP8: scale shape must be
/// `[ceil(rows / block_rows), ceil(cols / block_cols)]`.
pub fn fp8_block_dequant(
    name: &str,
    q: &[u8],
    rows: usize,
    cols: usize,
    scales: &[f32],
    scale_shape: [usize; 2],
    block: [usize; 2],
) -> Result<Matrix> {
    let want = [rows.div_ceil(block[0]), cols.div_ceil(block[1])];
    if scale_shape != want {
        return Err(Error::layout(
            name,
            format!(
                "scale shape {scale_shape:?}, expected {want:?} for {rows}×{cols} in {block:?} blocks"
            ),
        ));
    }
    fp8_dequant_with(name, q, rows, cols, scales, block[1], |r| r / block[0])
}

/// MXFP4 `[rows, cols]` from packed `[rows, cols/2]` codes and
/// `[rows, cols/32]` E8M0 scales.
pub fn mxfp4_dequant(
    name: &str,
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    block: usize,
) -> Result<Matrix> {
    if !cols.is_multiple_of(block) || !block.is_multiple_of(2) {
        return Err(Error::layout(
            name,
            format!("{cols} columns in blocks of {block}"),
        ));
    }
    if packed.len() != rows * cols / 2 || scales.len() != rows * cols / block {
        return Err(Error::layout(
            name,
            format!(
                "{} packed bytes and {} scales for {rows}×{cols}",
                packed.len(),
                scales.len()
            ),
        ));
    }
    let mut out = Matrix::zeros(rows, cols);
    out.data
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(r, orow)| {
            let prow = &packed[r * cols / 2..(r + 1) * cols / 2];
            let srow = &scales[r * cols / block..(r + 1) * cols / block];
            for (i, &byte) in prow.iter().enumerate() {
                let s = e8m0_to_f32(srow[2 * i / block]);
                orow[2 * i] = E2M1_VALUES[(byte & 0x0f) as usize] * s;
                orow[2 * i + 1] = E2M1_VALUES[(byte >> 4) as usize] * s;
            }
        });
    Ok(out)
}

/// How the fused QKV rows of one checkpoint are laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QkvSharding {
    /// Number of rank-major chunks (`tp_size` the checkpoint was saved for).
    pub chunks: usize,
}

/// Rows of the fused `qkv_proj` in the unsharded `[Q | K | V]` order the
/// forward uses, from a checkpoint saved pre-sharded for `chunks` ranks.
///
/// The checkpoint stores, for chunk `c = 0..chunks`, that rank's slice
/// `[Q_c | K_c | V_c]`: `Q_c` holds query heads
/// `c·(nq/chunks) .. (c+1)·(nq/chunks)`, `K_c` and `V_c` the KV heads
/// `c·(nkv/chunks) ..`, each head's rows contiguous. The FP8 scale grid is
/// tiled **per chunk**: chunk `c` owns scale rows
/// `c·ceil(R/b) .. (c+1)·ceil(R/b)` where `R` is the chunk's row count, so
/// when `R` is not a multiple of the block (MiMo-V2.6-Flash global layers:
/// `R = 3392 = 26.5 × 128`) the last tile of every chunk is a partial tile
/// with phantom rows, and row `r` of chunk `c` uses scale row
/// `c·ceil(R/b) + (r mod R) / b` — not `r / b`. A grid with exactly
/// `ceil(total/b)` rows (one continuous tiling) is also accepted, and the
/// per-chunk reading wins when both row counts coincide. This is the layout
/// vLLM's `_shard_fp8_qkv_proj` and llama.cpp's `_tp_aware_qkv_dequant`
/// read.
///
/// `scales` is `None` for an unquantised (float) fused QKV, which is
/// de-interleaved the same way.
pub fn deinterleave_qkv(
    name: &str,
    weight: QkvWeight<'_>,
    hidden: usize,
    attn: &AttentionSpec,
    sharding: QkvSharding,
    block: [usize; 2],
) -> Result<Matrix> {
    let rows = attn.qkv_rows();
    let n = sharding.chunks;
    if n == 0 || !attn.num_q_heads.is_multiple_of(n) || !attn.num_kv_heads.is_multiple_of(n) {
        return Err(Error::layout(
            name,
            format!(
                "{n} chunks do not divide {} query / {} KV heads",
                attn.num_q_heads, attn.num_kv_heads
            ),
        ));
    }
    let per = rows / n;
    let q_per = attn.q_size() / n;
    let k_per = attn.k_size() / n;
    let qh_per = attn.num_q_heads / n;
    let kvh_per = attn.num_kv_heads / n;

    // Source row (in the checkpoint) of each destination row.
    let mut src = Vec::with_capacity(rows);
    for h in 0..attn.num_q_heads {
        let (c, j) = (h / qh_per, h % qh_per);
        src.extend((0..attn.head_dim_qk).map(|d| c * per + j * attn.head_dim_qk + d));
    }
    for h in 0..attn.num_kv_heads {
        let (c, j) = (h / kvh_per, h % kvh_per);
        src.extend((0..attn.head_dim_qk).map(|d| c * per + q_per + j * attn.head_dim_qk + d));
    }
    for h in 0..attn.num_kv_heads {
        let (c, j) = (h / kvh_per, h % kvh_per);
        src.extend((0..attn.head_dim_v).map(|d| c * per + q_per + k_per + j * attn.head_dim_v + d));
    }
    debug_assert_eq!(src.len(), rows);

    let sharded = match weight {
        QkvWeight::Float(data) => {
            if data.len() != rows * hidden {
                return Err(Error::layout(name, "float qkv size mismatch"));
            }
            Matrix::from_vec(rows, hidden, data)
        }
        QkvWeight::Fp8 {
            q,
            scales,
            scale_shape,
        } => {
            let col_blocks = hidden.div_ceil(block[1]);
            if scale_shape[1] != col_blocks {
                return Err(Error::layout(
                    name,
                    format!("scale shape {scale_shape:?}: {col_blocks} column blocks expected"),
                ));
            }
            let chunk_scale_rows = per.div_ceil(block[0]);
            let br = block[0];
            if scale_shape[0] == n * chunk_scale_rows {
                fp8_dequant_with(name, q, rows, hidden, scales, block[1], |r| {
                    (r / per) * chunk_scale_rows + (r % per) / br
                })?
            } else if scale_shape[0] == rows.div_ceil(br) {
                fp8_dequant_with(name, q, rows, hidden, scales, block[1], |r| r / br)?
            } else {
                return Err(Error::layout(
                    name,
                    format!(
                        "scale has {} rows; expected {} ({n} chunks of {chunk_scale_rows}) or {}",
                        scale_shape[0],
                        n * chunk_scale_rows,
                        rows.div_ceil(br)
                    ),
                ));
            }
        }
    };
    Ok(sharded.select_rows(&src))
}

/// The stored form of a fused QKV weight.
pub enum QkvWeight<'a> {
    Float(Vec<f32>),
    Fp8 {
        q: &'a [u8],
        scales: &'a [f32],
        scale_shape: [usize; 2],
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AttentionKind;

    fn flash_global() -> AttentionSpec {
        AttentionSpec {
            kind: AttentionKind::Global,
            num_q_heads: 64,
            num_kv_heads: 4,
            head_dim_qk: 192,
            head_dim_v: 128,
            rope_dim: 64,
            rope_theta: 1e7,
            has_sinks: false,
            softmax_scale: 192f32.powf(-0.5),
        }
    }

    /// Encode a row/column identity into an FP8 value plus scale so the
    /// dequantised value tells us which source row and which scale tile were
    /// used.
    #[test]
    fn flash_global_padded_chunks_use_per_chunk_scale_rows() {
        let attn = flash_global();
        let hidden = 256; // two column blocks keep the test small
        let rows = attn.qkv_rows();
        assert_eq!(rows, 4 * 3392);
        let per = 3392;
        let chunk_scale_rows = 27; // 26.5 blocks, padded
        // Every weight byte is 1.0 (0x38); the scale encodes (tile row, tile col).
        let q = vec![0x38u8; rows * hidden];
        let scale_shape = [4 * chunk_scale_rows, 2];
        let scales: Vec<f32> = (0..scale_shape[0] * 2)
            .map(|i| (i / 2) as f32 * 10.0 + (i % 2) as f32 + 1.0)
            .collect();
        let m = deinterleave_qkv(
            "t",
            QkvWeight::Fp8 {
                q: &q,
                scales: &scales,
                scale_shape,
            },
            hidden,
            &attn,
            QkvSharding { chunks: 4 },
            [128, 128],
        )
        .unwrap();
        let expect = |dst_row: usize, src_row: usize| {
            let srow = (src_row / per) * chunk_scale_rows + (src_row % per) / 128;
            assert_eq!(m.row(dst_row)[0], srow as f32 * 10.0 + 1.0, "dst {dst_row}");
            assert_eq!(
                m.row(dst_row)[200],
                srow as f32 * 10.0 + 2.0,
                "dst {dst_row}"
            );
        };
        // Q head 17 lives in chunk 1 at local head 1.
        expect(17 * 192 + 5, per + 192 + 5);
        // K head 3 (chunk 3): after chunk 3's 16 Q heads.
        expect(64 * 192 + 3 * 192 + 7, 3 * per + 16 * 192 + 7);
        // V head 2, last row: chunk 2, after Q (3072) and K (192).
        expect(
            64 * 192 + 4 * 192 + 2 * 128 + 127,
            2 * per + 3072 + 192 + 127,
        );
        // The last row of chunk 0 sits in the half-filled tile 26.
        expect(64 * 192 + 4 * 192 + 127, 3072 + 192 + 127);
        assert_eq!(3072 + 192 + 127, 3391);
    }

    #[test]
    fn mxfp4_nibble_order_and_scale() {
        // Row of 32 elements: byte 0 = 0x21 → element 0 code 1 (0.5), element 1 code 2 (1.0).
        let mut packed = vec![0u8; 16];
        packed[0] = 0x21;
        packed[15] = 0xf7; // element 30 = 6.0, element 31 = -6.0
        let scales = vec![128u8]; // 2.0
        let m = mxfp4_dequant("t", &packed, &scales, 1, 32, 32).unwrap();
        assert_eq!(m.row(0)[0], 1.0);
        assert_eq!(m.row(0)[1], 2.0);
        assert_eq!(m.row(0)[30], 12.0);
        assert_eq!(m.row(0)[31], -12.0);
    }

    #[test]
    fn fp8_partial_edge_blocks() {
        // 3×5 weight in 2×2 blocks → 2×3 scale grid.
        let q = vec![0x38u8; 15];
        let scales: Vec<f32> = (0..6).map(|i| i as f32 + 1.0).collect();
        let m = fp8_block_dequant("t", &q, 3, 5, &scales, [2, 3], [2, 2]).unwrap();
        assert_eq!(m.row(0), &[1.0, 1.0, 2.0, 2.0, 3.0]);
        assert_eq!(m.row(2), &[4.0, 4.0, 5.0, 5.0, 6.0]);
        assert!(fp8_block_dequant("t", &q, 3, 5, &scales, [3, 2], [2, 2]).is_err());
    }
}
