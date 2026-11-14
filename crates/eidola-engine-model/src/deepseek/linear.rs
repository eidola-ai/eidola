//! The reference's linear layers: quantised GEMMs (`fp8_gemm_kernel`,
//! `fp4_gemm_kernel`) and the plain BF16 / f32 products it leaves to PyTorch.
//!
//! A quantised GEMM multiplies FP8 activation codes by FP8 or FP4 weight
//! codes in 32-wide blocks along the input, applying the two block scales per
//! block. A block's dot product is a sum of 32 products of small multiples of
//! powers of two, so it is computed exactly and rounded once to f32, whatever
//! the order; blocks then accumulate in ascending order in f32.

use rayon::prelude::*;

use super::numeric::bf16_round;
use super::quant::{FP8_BLOCK, act_quant};
use crate::numeric::{E2M1_VALUES, e8m0_to_f32, fp8_e4m3_to_f32};
use crate::tensor::{Matrix, dot};

/// An FP8 weight: E4M3 values `[rows, cols]` (not multiplied by their scales)
/// and one power-of-two scale per 32×32 tile.
#[derive(Clone)]
pub struct Fp8Linear {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
    /// `[ceil(rows / 32), cols / 32]`, row-major.
    pub scales: Vec<f32>,
}

impl std::fmt::Debug for Fp8Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fp8Linear({}×{})", self.rows, self.cols)
    }
}

impl Fp8Linear {
    /// From the stored bytes: `codes` is `[rows, cols]` E4M3 and `scales` is
    /// `[ceil(rows / 32), cols / 32]` E8M0.
    pub fn from_bytes(rows: usize, cols: usize, codes: &[u8], scales: &[u8]) -> Self {
        assert!(cols.is_multiple_of(FP8_BLOCK));
        assert_eq!(codes.len(), rows * cols);
        assert_eq!(scales.len(), rows.div_ceil(32) * (cols / 32));
        Fp8Linear {
            rows,
            cols,
            values: codes.iter().map(|&b| fp8_e4m3_to_f32(b)).collect(),
            scales: scales.iter().map(|&b| e8m0_to_f32(b)).collect(),
        }
    }

    /// The weight as real values, each rounded to BF16: the form the reference
    /// keeps `wo_a` in.
    pub fn dequantised_bf16(&self) -> Matrix {
        let kb = self.cols / 32;
        let mut out = Matrix::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                let s = self.scales[(r / 32) * kb + c / 32];
                out.data[r * self.cols + c] = bf16_round(self.values[r * self.cols + c] * s);
            }
        }
        out
    }
}

/// Exact dot product of two 32-wide blocks of FP8 values, rounded once to
/// f32. Every partial sum is an integer multiple of 2^-18 below 2^23, exact in
/// f64 in any order.
#[inline]
fn fp8_block_dot(a: &[f64], w: &[f32]) -> f32 {
    let mut s = [0.0f64; 4];
    for j in 0..FP8_BLOCK {
        s[j & 3] += a[j] * w[j] as f64;
    }
    ((s[0] + s[1]) + (s[2] + s[3])) as f32
}

/// Activation codes of one token, widened for the exact block products.
struct Codes {
    wide: Vec<f64>,
    scales: Vec<f32>,
}

fn quantise(x: &[f32]) -> Codes {
    let mut codes = vec![0.0f32; x.len()];
    let mut scales = vec![0.0f32; x.len() / FP8_BLOCK];
    act_quant(x, &mut codes, &mut scales);
    Codes {
        wide: codes.iter().map(|&c| c as f64).collect(),
        scales,
    }
}

fn fp8_gemm_row(codes: &Codes, w: &Fp8Linear, n: usize) -> f32 {
    let kb = w.cols / FP8_BLOCK;
    let row = &w.values[n * w.cols..(n + 1) * w.cols];
    let wscale = &w.scales[(n / 32) * kb..(n / 32 + 1) * kb];
    let mut acc = 0.0f32;
    for k in 0..kb {
        let d = fp8_block_dot(
            &codes.wide[k * FP8_BLOCK..(k + 1) * FP8_BLOCK],
            &row[k * FP8_BLOCK..(k + 1) * FP8_BLOCK],
        );
        acc += d * codes.scales[k] * wscale[k];
    }
    acc
}

/// `lin8`: FP8-quantise `x` (32-wide blocks), multiply by an FP8 weight, round
/// to BF16. One token; `out` has `w.rows` elements.
pub fn lin8(x: &[f32], w: &Fp8Linear, out: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "lin8 input width");
    assert_eq!(out.len(), w.rows);
    let codes = quantise(x);
    out.par_iter_mut()
        .enumerate()
        .with_min_len(64)
        .for_each(|(n, o)| *o = bf16_round(fp8_gemm_row(&codes, w, n)));
}

/// [`lin8`] over every row of `x`.
pub fn lin8_rows(x: &Matrix, w: &Fp8Linear) -> Matrix {
    let mut out = Matrix::zeros(x.rows, w.rows);
    for r in 0..x.rows {
        lin8(x.row(r), w, out.row_mut(r));
    }
    out
}

/// An FP4 expert weight decoded for the exact block products: twice the E2M1
/// value of every element (an integer in `-12..=12`), and the E8M0 scale of
/// each row's 32-wide blocks.
pub struct Fp4Linear {
    pub rows: usize,
    pub cols: usize,
    doubled: Vec<i8>,
    /// `[rows, cols / 32]`.
    scales: Vec<f32>,
}

impl Fp4Linear {
    /// From the stored bytes: `packed` is `[rows, cols / 2]` with element `2i`
    /// in the low nibble of byte `i`, and `scales` is `[rows, cols / 32]` E8M0.
    pub fn from_bytes(rows: usize, cols: usize, packed: &[u8], scales: &[u8]) -> Self {
        assert!(cols.is_multiple_of(32));
        assert_eq!(packed.len(), rows * cols / 2);
        assert_eq!(scales.len(), rows * cols / 32);
        let mut doubled = Vec::with_capacity(rows * cols);
        for &b in packed {
            doubled.push((2.0 * E2M1_VALUES[(b & 0x0f) as usize]) as i8);
            doubled.push((2.0 * E2M1_VALUES[(b >> 4) as usize]) as i8);
        }
        Fp4Linear {
            rows,
            cols,
            doubled,
            scales: scales.iter().map(|&b| e8m0_to_f32(b)).collect(),
        }
    }
}

/// `lin4`: FP8-quantise `x`, multiply by an FP4 weight, round to BF16.
///
/// A block dot is `Σ a·w` with `a` a multiple of 2^-9 up to 448 and `w` a
/// multiple of ½ up to 6: in units of 2^-10 it is an integer below 2^27,
/// summed exactly in i32 and rounded once to f32.
pub fn lin4(x: &[f32], w: &Fp4Linear, out: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "lin4 input width");
    assert_eq!(out.len(), w.rows);
    let codes = quantise(x);
    let ints: Vec<i32> = codes.wide.iter().map(|&c| (c * 512.0) as i32).collect();
    let kb = w.cols / 32;
    debug_assert_eq!(ints.len(), kb * 32);
    out.par_iter_mut()
        .enumerate()
        .with_min_len(64)
        .for_each(|(n, o)| {
            let row = &w.doubled[n * w.cols..(n + 1) * w.cols];
            let wscale = &w.scales[n * kb..(n + 1) * kb];
            let mut acc = 0.0f32;
            for (k, (a, w)) in ints.chunks_exact(32).zip(row.chunks_exact(32)).enumerate() {
                let sum: i32 = a.iter().zip(w).map(|(a, w)| a * *w as i32).sum();
                let d = sum as f32 * (1.0 / 1024.0);
                acc += d * codes.scales[k] * wscale[k];
            }
            *o = bf16_round(acc);
        });
}

/// A plain product with a BF16 weight on BF16 input: f32 dot products in the
/// pinned order, each rounded once to BF16.
pub fn linear_bf16(x: &[f32], w: &Matrix, out: &mut [f32]) {
    linear_f32(x, w, out);
    for o in out {
        *o = bf16_round(*o);
    }
}

/// A plain f32 product: `out[n] = dot(x, w.row(n))` in the pinned order.
pub fn linear_f32(x: &[f32], w: &Matrix, out: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "linear input width");
    assert_eq!(out.len(), w.rows);
    out.par_iter_mut()
        .enumerate()
        .with_min_len(64)
        .for_each(|(n, o)| *o = dot(x, w.row(n)));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quantised GEMM against the same sum done with scaled real values in
    /// f64: equal up to the f32 accumulation.
    #[test]
    fn lin8_matches_a_scaled_reference() {
        let (rows, cols) = (40usize, 64usize);
        let codes: Vec<u8> = (0..rows * cols)
            .map(|i| ((i * 37 + 11) % 127) as u8 | (((i / 7) % 2) as u8) << 7)
            .collect();
        let scales: Vec<u8> = (0..rows.div_ceil(32) * (cols / 32))
            .map(|i| 120 + (i % 5) as u8)
            .collect();
        let w = Fp8Linear::from_bytes(rows, cols, &codes, &scales);
        let x: Vec<f32> = (0..cols)
            .map(|i| bf16_round(((i * 13 % 29) as f32 - 14.0) * 0.37))
            .collect();
        let mut out = vec![0.0; rows];
        lin8(&x, &w, &mut out);
        let (mut c, mut s) = (vec![0.0; cols], vec![0.0; cols / 32]);
        act_quant(&x, &mut c, &mut s);
        for (n, &got) in out.iter().enumerate() {
            let mut want = 0.0f64;
            for j in 0..cols {
                let ws = w.scales[(n / 32) * (cols / 32) + j / 32] as f64;
                want += c[j] as f64 * s[j / 32] as f64 * w.values[n * cols + j] as f64 * ws;
            }
            assert!(
                (got as f64 - want).abs() <= want.abs() * 2f64.powi(-7) + 1e-30,
                "row {n}"
            );
        }
    }

    #[test]
    fn fp4_unpacks_low_nibble_first() {
        // Byte 0x93: low nibble 3 (1.5) is element 0, high nibble 9 (-0.5) is element 1.
        let mut packed = vec![0u8; 16];
        packed[0] = 0x93;
        let w = Fp4Linear::from_bytes(1, 32, &packed, &[128]);
        assert_eq!(&w.doubled[..3], &[3, -1, 0]);
        let mut x = vec![0.0f32; 32];
        x[0] = 1.0;
        x[1] = 2.0;
        let mut out = [0.0f32];
        lin4(&x, &w, &mut out);
        // (1·1.5 + 2·(-0.5)) · 2^(128-127)
        assert_eq!(out[0], 1.0);
    }
}
