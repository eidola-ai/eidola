//! The released models' linear layers: FP8 block-scaled GEMMs with dynamic
//! per-token activation quantisation, and plain BF16 products. A projection
//! is a [`Linear`] of the checkpoint's format: FP8 in Qwen3.8-27B-FP8, BF16 in
//! Qwen3.5's checkpoints, and [`linear_rows`] runs either.
//!
//! Weights stay in the memory-mapped checkpoint and are decoded one output
//! row at a time as a product reads them, so a model whose f32 weights would
//! not fit in memory still runs.
//!
//! **FP8 GEMM** (every FP8 weight). Per token and per 128 consecutive input
//! columns, the activation scale is `s = amax / 448` in f32 and each element
//! becomes the E4M3 value of `x / s` (nearest, ties to even, saturating at
//! ±448; an all-zero block has codes and scale 0). A 128-term block dot of
//! E4M3 codes is a sum of multiples of 2^-18 each below 2^18, exact in f64 in
//! any order; it is rounded once to f32. Blocks accumulate in ascending order
//! in f32 as `(dot · s) · weight_scale`, and the sum is rounded to BF16. That
//! is the released Triton kernel's arithmetic
//! (`kernels-community/finegrained-fp8`: `_fp8_act_quant_kernel` and
//! `w8a8_block_fp8_matmul_kernel`) with what its source leaves open (the
//! accumulation inside a block, the cast's rounding) fixed.
//!
//! **BF16 product** (every BF16 weight, and every projection of a BF16
//! checkpoint): f32 dot products of the BF16 input and weight in the crate's
//! pinned order ([`crate::tensor::dot`]), each rounded once to BF16. That is
//! `F.linear` of two BF16 tensors with its summation order fixed.

use std::sync::Arc;

use rayon::prelude::*;

use super::config::{QUANT_BLOCK, WeightFormat};
use super::numeric::{bf16_round, round_e4m3};
use crate::error::{Error, Result};
use crate::numeric::{bf16_to_f32, fp8_e4m3_to_f32};
use crate::safetensors::{Dtype, WeightSet};
use crate::tensor::{Matrix, dot};

/// Bytes of one stored tensor: in a mapped checkpoint, or owned.
#[derive(Clone)]
pub enum Stored {
    Mapped { store: Arc<WeightSet>, name: String },
    Owned(Arc<Vec<u8>>),
}

impl Stored {
    /// A tensor of `store`, checked for dtype and shape.
    pub fn mapped(
        store: &Arc<WeightSet>,
        name: &str,
        dtype: Dtype,
        shape: &[usize],
    ) -> Result<Self> {
        store.get(name)?.expect(name, dtype, shape)?;
        Ok(Stored::Mapped {
            store: store.clone(),
            name: name.to_string(),
        })
    }

    pub fn bytes(&self) -> &[u8] {
        match self {
            Stored::Mapped { store, name } => {
                store
                    .get(name)
                    .expect("a mapped tensor was checked when it was opened")
                    .data
            }
            Stored::Owned(b) => b,
        }
    }
}

/// A BF16 weight `[rows, cols]` read in place.
#[derive(Clone)]
pub struct Bf16Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Stored,
}

impl std::fmt::Debug for Bf16Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bf16Matrix({}×{})", self.rows, self.cols)
    }
}

impl Bf16Matrix {
    pub fn open(store: &Arc<WeightSet>, name: &str, rows: usize, cols: usize) -> Result<Self> {
        Ok(Bf16Matrix {
            rows,
            cols,
            data: Stored::mapped(store, name, Dtype::Bf16, &[rows, cols])?,
        })
    }

    /// Row `r`, decoded exactly to f32.
    pub fn row_into(&self, r: usize, out: &mut [f32]) {
        assert!(r < self.rows, "row {r} of {}", self.rows);
        assert_eq!(out.len(), self.cols);
        let bytes = &self.data.bytes()[r * self.cols * 2..(r + 1) * self.cols * 2];
        for (o, b) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
            *o = bf16_to_f32(u16::from_le_bytes(*b));
        }
    }

    pub fn row(&self, r: usize) -> Vec<f32> {
        let mut out = vec![0.0; self.cols];
        self.row_into(r, &mut out);
        out
    }

    /// The whole matrix in f32 (for small weights).
    pub fn to_matrix(&self) -> Matrix {
        let mut m = Matrix::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            self.row_into(r, m.row_mut(r));
        }
        m
    }
}

/// An FP8 E4M3 weight `[rows, cols]` with one BF16 scale per 128×128 block.
#[derive(Clone)]
pub struct Fp8Linear {
    pub rows: usize,
    pub cols: usize,
    pub codes: Stored,
    /// `[rows / 128, cols / 128]`, row-major, widened exactly from BF16.
    pub scales: Vec<f32>,
}

impl std::fmt::Debug for Fp8Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fp8Linear({}×{})", self.rows, self.cols)
    }
}

impl Fp8Linear {
    /// `<prefix>.weight` (F8_E4M3) and `<prefix>.weight_scale_inv` (BF16).
    pub fn open(store: &Arc<WeightSet>, prefix: &str, rows: usize, cols: usize) -> Result<Self> {
        if !rows.is_multiple_of(QUANT_BLOCK) || !cols.is_multiple_of(QUANT_BLOCK) {
            return Err(Error::layout(
                prefix,
                format!("{rows}×{cols} is not whole 128×128 blocks"),
            ));
        }
        let codes = Stored::mapped(
            store,
            &format!("{prefix}.weight"),
            Dtype::F8E4M3,
            &[rows, cols],
        )?;
        let sname = format!("{prefix}.weight_scale_inv");
        let s = store.get(&sname)?;
        s.expect(
            &sname,
            Dtype::Bf16,
            &[rows / QUANT_BLOCK, cols / QUANT_BLOCK],
        )?;
        let scales = s.to_f32(&sname)?;
        if let Some(bad) = scales.iter().find(|v| !v.is_finite()) {
            return Err(Error::layout(&sname, format!("scale {bad} is not finite")));
        }
        Ok(Fp8Linear {
            rows,
            cols,
            codes,
            scales,
        })
    }

    /// From codes and scales in memory (tests).
    pub fn from_parts(rows: usize, cols: usize, codes: Vec<u8>, scales: Vec<f32>) -> Self {
        assert!(rows.is_multiple_of(QUANT_BLOCK) && cols.is_multiple_of(QUANT_BLOCK));
        assert_eq!(codes.len(), rows * cols);
        assert_eq!(scales.len(), (rows / QUANT_BLOCK) * (cols / QUANT_BLOCK));
        Fp8Linear {
            rows,
            cols,
            codes: Stored::Owned(Arc::new(codes)),
            scales,
        }
    }

    /// Row `r`'s E4M3 values (not multiplied by their scales), widened to f64.
    fn row_codes(&self, r: usize, out: &mut [f64]) {
        let bytes = &self.codes.bytes()[r * self.cols..(r + 1) * self.cols];
        for (o, &b) in out.iter_mut().zip(bytes) {
            *o = E4M3_F64[b as usize];
        }
    }

    /// The weight as real values `code · scale`, exact in f32 (for tests).
    pub fn dequantised(&self) -> Matrix {
        let kb = self.cols / QUANT_BLOCK;
        let codes = self.codes.bytes();
        let mut m = Matrix::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                let s = self.scales[(r / QUANT_BLOCK) * kb + c / QUANT_BLOCK];
                m.data[r * self.cols + c] = fp8_e4m3_to_f32(codes[r * self.cols + c]) * s;
            }
        }
        m
    }
}

/// A projection as the checkpoint stores it: `<prefix>.weight` FP8 with its
/// block scales, or BF16.
#[derive(Clone, Debug)]
pub enum Linear {
    Fp8(Fp8Linear),
    Bf16(Bf16Matrix),
}

impl Linear {
    /// `<prefix>.weight` (and, for FP8, `<prefix>.weight_scale_inv`) as
    /// `format` stores it, `[rows, cols]`.
    pub fn open(
        store: &Arc<WeightSet>,
        prefix: &str,
        rows: usize,
        cols: usize,
        format: WeightFormat,
    ) -> Result<Self> {
        Ok(match format {
            WeightFormat::Fp8Block128 => Linear::Fp8(Fp8Linear::open(store, prefix, rows, cols)?),
            WeightFormat::Bf16 => Linear::Bf16(Bf16Matrix::open(
                store,
                &format!("{prefix}.weight"),
                rows,
                cols,
            )?),
        })
    }

    pub fn rows(&self) -> usize {
        match self {
            Linear::Fp8(w) => w.rows,
            Linear::Bf16(w) => w.rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Linear::Fp8(w) => w.cols,
            Linear::Bf16(w) => w.cols,
        }
    }

    /// The FP8 weight, for an executor that runs only those.
    pub fn as_fp8(&self) -> Option<&Fp8Linear> {
        match self {
            Linear::Fp8(w) => Some(w),
            Linear::Bf16(_) => None,
        }
    }
}

const E4M3_F64: [f64; 256] = {
    let mut t = [0.0f64; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = fp8_e4m3_to_f32(i as u8) as f64;
        i += 1;
    }
    t
};

/// The FP8 activation quantiser for one token: per 128-wide block,
/// `scale = amax / 448` and `code = e4m3(x / scale)`. A block of zeros has
/// scale 0 and codes 0 (`0 / 0` is a NaN, which casts to +0).
pub fn act_quant(x: &[f32], codes: &mut [f32], scales: &mut [f32]) {
    assert!(x.len().is_multiple_of(QUANT_BLOCK), "width {}", x.len());
    assert_eq!(codes.len(), x.len());
    assert_eq!(scales.len(), x.len() / QUANT_BLOCK);
    for ((xb, cb), s) in x
        .chunks_exact(QUANT_BLOCK)
        .zip(codes.chunks_exact_mut(QUANT_BLOCK))
        .zip(scales.iter_mut())
    {
        let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        *s = amax / 448.0;
        for (c, &v) in cb.iter_mut().zip(xb) {
            *c = round_e4m3(v / *s);
        }
    }
}

/// Exact dot product of two 128-wide blocks of E4M3 values, rounded once to
/// f32. Every partial sum is an integer multiple of 2^-18 below 2^25, exact in
/// f64 whatever the order.
#[inline]
pub fn fp8_block_dot(a: &[f64], w: &[f64]) -> f32 {
    debug_assert_eq!(a.len(), QUANT_BLOCK);
    let mut s = [0.0f64; 8];
    for (ac, wc) in a.chunks_exact(8).zip(w.chunks_exact(8)) {
        for j in 0..8 {
            s[j] += ac[j] * wc[j];
        }
    }
    (((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]))) as f32
}

/// Quantised activations of a batch of tokens.
pub struct Quantised {
    pub cols: usize,
    /// `[tokens, cols]`: the E4M3 values, widened to f64.
    pub codes: Vec<f64>,
    /// `[tokens, cols / 128]`.
    pub scales: Vec<f32>,
}

pub fn quantise_rows(x: &Matrix) -> Quantised {
    let kb = x.cols / QUANT_BLOCK;
    let mut codes = vec![0.0f64; x.rows * x.cols];
    let mut scales = vec![0.0f32; x.rows * kb];
    codes
        .par_chunks_mut(x.cols.max(1))
        .zip(scales.par_chunks_mut(kb.max(1)))
        .enumerate()
        .for_each(|(t, (c, s))| {
            let mut c32 = vec![0.0f32; x.cols];
            act_quant(x.row(t), &mut c32, s);
            for (o, v) in c.iter_mut().zip(c32) {
                *o = v as f64;
            }
        });
    Quantised {
        cols: x.cols,
        codes,
        scales,
    }
}

/// One output element: blocks in ascending order, `acc += (dot · s_a) · s_w`.
#[inline]
pub fn fp8_row_dot(a: &[f64], sa: &[f32], w: &[f64], sw: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (k, (ab, wb)) in a
        .chunks_exact(QUANT_BLOCK)
        .zip(w.chunks_exact(QUANT_BLOCK))
        .enumerate()
    {
        acc += (fp8_block_dot(ab, wb) * sa[k]) * sw[k];
    }
    acc
}

/// Output rows a parallel task computes; each weight row is decoded once per
/// call.
const ROW_BLOCK: usize = 64;

/// Every output of `rows_of(n)` for `n` in `0..out_rows`, for every token,
/// computed by `f(n, row_buffer) -> [token values]` in parallel over blocks of
/// output rows and assembled token-major.
fn by_output_row<B: Send>(
    tokens: usize,
    out_rows: usize,
    make: impl Fn() -> B + Sync,
    f: impl Fn(usize, &mut B, &mut [f32]) + Sync,
) -> Matrix {
    let blocks: Vec<Vec<f32>> = (0..out_rows.div_ceil(ROW_BLOCK))
        .into_par_iter()
        .map(|b| {
            let n0 = b * ROW_BLOCK;
            let n1 = (n0 + ROW_BLOCK).min(out_rows);
            let mut buf = make();
            let mut part = vec![0.0f32; (n1 - n0) * tokens];
            for n in n0..n1 {
                f(
                    n,
                    &mut buf,
                    &mut part[(n - n0) * tokens..(n - n0 + 1) * tokens],
                );
            }
            part
        })
        .collect();
    let mut y = Matrix::zeros(tokens, out_rows);
    for (b, part) in blocks.iter().enumerate() {
        let n0 = b * ROW_BLOCK;
        for i in 0..part.len() / tokens.max(1) {
            for t in 0..tokens {
                y.data[t * out_rows + n0 + i] = part[i * tokens + t];
            }
        }
    }
    y
}

/// The FP8 linear over every row of `x` (BF16 values), rounded to BF16.
pub fn lin8_rows(x: &Matrix, w: &Fp8Linear) -> Matrix {
    assert_eq!(x.cols, w.cols, "FP8 linear input width");
    let q = quantise_rows(x);
    let kb = w.cols / QUANT_BLOCK;
    by_output_row(
        x.rows,
        w.rows,
        || vec![0.0f64; w.cols],
        |n, row, out| {
            w.row_codes(n, row);
            let sw = &w.scales[(n / QUANT_BLOCK) * kb..(n / QUANT_BLOCK + 1) * kb];
            for (t, o) in out.iter_mut().enumerate() {
                let a = &q.codes[t * w.cols..(t + 1) * w.cols];
                let sa = &q.scales[t * kb..(t + 1) * kb];
                *o = bf16_round(fp8_row_dot(a, sa, row, sw));
            }
        },
    )
}

/// The projection `w` over every row of `x` (BF16 values), in its stored
/// format's arithmetic: [`lin8_rows`] or [`linear_bf16_rows`].
pub fn linear_rows(x: &Matrix, w: &Linear) -> Matrix {
    match w {
        Linear::Fp8(w) => lin8_rows(x, w),
        Linear::Bf16(w) => linear_bf16_rows(x, w),
    }
}

/// A BF16 weight on BF16 input: f32 dot products in the pinned order
/// ([`crate::tensor::dot`]), each rounded once to BF16.
pub fn linear_bf16_rows(x: &Matrix, w: &Bf16Matrix) -> Matrix {
    assert_eq!(x.cols, w.cols, "BF16 linear input width");
    by_output_row(
        x.rows,
        w.rows,
        || vec![0.0f32; w.cols],
        |n, row, out| {
            w.row_into(n, row);
            for (t, o) in out.iter_mut().enumerate() {
                *o = bf16_round(dot(x.row(t), row));
            }
        },
    )
}

/// [`linear_bf16_rows`] for a weight already in f32.
pub fn linear_bf16_dense(x: &Matrix, w: &Matrix) -> Matrix {
    assert_eq!(x.cols, w.cols, "BF16 linear input width");
    by_output_row(
        x.rows,
        w.rows,
        || (),
        |n, _, out| {
            for (t, o) in out.iter_mut().enumerate() {
                *o = bf16_round(dot(x.row(t), w.row(n)));
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The quantised GEMM against the same sum done with scaled real values in
    /// f64: equal up to the f32 accumulation and the final BF16 rounding.
    #[test]
    fn lin8_matches_a_scaled_reference() {
        let (rows, cols) = (256usize, 384usize);
        let codes: Vec<u8> = (0..rows * cols)
            .map(|i| ((i * 37 + 11) % 126) as u8 | (((i / 7) % 2) as u8) << 7)
            .collect();
        let scales: Vec<f32> = (0..2 * 3)
            .map(|i| bf16_round(1e-3 * (1.0 + i as f32 * 0.37)))
            .collect();
        let w = Fp8Linear::from_parts(rows, cols, codes, scales);
        let x = Matrix::from_vec(
            2,
            cols,
            (0..2 * cols)
                .map(|i| bf16_round(((i * 13 % 29) as f32 - 14.0) * 0.37))
                .collect(),
        );
        let y = lin8_rows(&x, &w);
        let dense = w.dequantised();
        for t in 0..2 {
            let mut c = vec![0.0; cols];
            let mut s = vec![0.0; cols / QUANT_BLOCK];
            act_quant(x.row(t), &mut c, &mut s);
            for n in 0..rows {
                let want: f64 = (0..cols)
                    .map(|j| {
                        c[j] as f64 * s[j / QUANT_BLOCK] as f64 * dense.data[n * cols + j] as f64
                    })
                    .sum();
                let got = y.data[t * rows + n] as f64;
                assert!(
                    (got - want).abs() <= want.abs() * 2f64.powi(-7) + 1e-30,
                    "token {t} row {n}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn a_zero_block_contributes_nothing() {
        let mut x = vec![0.0f32; 256];
        x[200] = 3.0;
        let (mut c, mut s) = (vec![0.0; 256], vec![0.0; 2]);
        act_quant(&x, &mut c, &mut s);
        assert_eq!(s, [0.0, 3.0 / 448.0]);
        assert!(c[..128].iter().all(|v| v.to_bits() == 0));
        assert_eq!(c[200], 448.0);
    }
}
