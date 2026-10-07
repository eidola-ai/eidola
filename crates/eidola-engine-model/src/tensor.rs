//! A row-major f32 matrix and the handful of kernels the reference uses.
//!
//! Every kernel works one output row at a time with a fixed summation order,
//! so a token's result never depends on which other tokens share the batch.
//! Executors that want bit-identical numerics call these same functions.

use rayon::prelude::*;

/// Row-major `rows × cols` f32 matrix.
#[derive(Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl std::fmt::Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matrix({}×{})", self.rows, self.cols)
    }
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "matrix data length");
        Matrix { rows, cols, data }
    }

    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// A new matrix made of the given rows, in order.
    pub fn select_rows(&self, rows: &[usize]) -> Matrix {
        let mut out = Matrix::zeros(rows.len(), self.cols);
        for (i, &r) in rows.iter().enumerate() {
            out.row_mut(i).copy_from_slice(self.row(r));
        }
        out
    }
}

const LANES: usize = 16;

/// Dot product with a fixed order: 16 interleaved partial sums, a halving
/// tree over them, then the tail.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; LANES];
    let (ca, ra) = a.as_chunks::<LANES>();
    let (cb, rb) = b.as_chunks::<LANES>();
    for (x, y) in ca.iter().zip(cb) {
        for j in 0..LANES {
            acc[j] += x[j] * y[j];
        }
    }
    let mut w = LANES / 2;
    while w > 0 {
        for j in 0..w {
            acc[j] += acc[j + w];
        }
        w /= 2;
    }
    let mut tail = 0.0f32;
    for (x, y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    acc[0] + tail
}

/// `out[n] = dot(x, w.row(n))` — a linear layer without bias for one token.
pub fn linear_row(x: &[f32], w: &Matrix, out: &mut [f32]) {
    debug_assert_eq!(x.len(), w.cols);
    debug_assert_eq!(out.len(), w.rows);
    for (n, o) in out.iter_mut().enumerate() {
        *o = dot(x, w.row(n));
    }
}

/// `y = x · wᵀ` for every row of `x`. Parallel over blocks of `w`'s rows so
/// each weight row is read once per call; per-element arithmetic is exactly
/// [`linear_row`]'s.
pub fn linear(x: &Matrix, w: &Matrix) -> Matrix {
    assert_eq!(
        x.cols, w.cols,
        "linear: input width {} vs weight {}",
        x.cols, w.cols
    );
    const BLOCK: usize = 64;
    let t = x.rows;
    let blocks: Vec<Vec<f32>> = (0..w.rows.div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| {
            let n0 = b * BLOCK;
            let n1 = (n0 + BLOCK).min(w.rows);
            let mut part = vec![0.0f32; (n1 - n0) * t];
            for n in n0..n1 {
                let wr = w.row(n);
                for r in 0..t {
                    part[(n - n0) * t + r] = dot(x.row(r), wr);
                }
            }
            part
        })
        .collect();
    let mut y = Matrix::zeros(t, w.rows);
    for (b, part) in blocks.iter().enumerate() {
        let n0 = b * BLOCK;
        let width = part.len() / t.max(1);
        for i in 0..width {
            for r in 0..t {
                y.data[r * w.rows + n0 + i] = part[i * t + r];
            }
        }
    }
    y
}

/// RMSNorm: `weight * (x / sqrt(mean(x²) + eps))`, with the mean of squares
/// summed in [`dot`] order.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    let var = dot(x, x) / x.len() as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for ((o, &v), &w) in out.iter_mut().zip(x).zip(weight) {
        *o = w * (v * inv);
    }
}

pub fn rms_norm_rows(x: &Matrix, weight: &[f32], eps: f32) -> Matrix {
    let mut out = Matrix::zeros(x.rows, x.cols);
    out.data
        .par_chunks_mut(x.cols.max(1))
        .enumerate()
        .for_each(|(r, o)| rms_norm(x.row(r), weight, eps, o));
    out
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// In-place `a += b`.
pub fn add_assign(a: &mut Matrix, b: &Matrix) {
    assert_eq!((a.rows, a.cols), (b.rows, b.cols));
    for (x, y) in a.data.iter_mut().zip(&b.data) {
        *x += *y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_naive_on_integers() {
        let a: Vec<f32> = (0..37).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..37).map(|i| (i % 5) as f32).collect();
        let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert_eq!(dot(&a, &b), naive);
    }

    #[test]
    fn linear_equals_linear_row() {
        let x = Matrix::from_vec(3, 70, (0..210).map(|i| (i as f32 * 0.37).sin()).collect());
        let w = Matrix::from_vec(
            130,
            70,
            (0..130 * 70).map(|i| (i as f32 * 0.11).cos()).collect(),
        );
        let y = linear(&x, &w);
        for r in 0..3 {
            let mut row = vec![0.0; 130];
            linear_row(x.row(r), &w, &mut row);
            assert_eq!(y.row(r), &row[..]);
        }
    }
}
