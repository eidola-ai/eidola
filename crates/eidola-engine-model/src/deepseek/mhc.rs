//! Hyper-connections: the residual stream is `hc` parallel copies, collapsed
//! into one sub-layer input and expanded back with coefficients computed from
//! the stream itself (`inference/model.py` `Block.hc_mixes` / `hc_pre` /
//! `hc_post`, `inference/kernel.py` `hc_split_sinkhorn_kernel`).

use super::numeric::{bf16_round, exp32, rsqrt32, sigmoid32};
use crate::tensor::{Matrix, dot};

/// The learned projection behind one sub-layer's coefficients.
#[derive(Debug, Clone)]
pub struct MhcWeights {
    /// `[(2 + hc) · hc, hc · dim]`.
    pub projection: Matrix,
    /// `[(2 + hc) · hc]`.
    pub base: Vec<f32>,
    /// One scale each for the pre, post and comb groups.
    pub scale: [f32; 3],
}

/// The coefficients one sub-layer computes from the stream entering it.
#[derive(Debug, Clone, PartialEq)]
pub struct Mixes {
    /// Collapse weights. A sub-layer applies the `pre` computed by the
    /// sub-layer before it.
    pub pre: Vec<f32>,
    /// How much of the sub-layer output each new copy takes.
    pub post: Vec<f32>,
    /// `[hc, hc]` row-major, made near doubly stochastic by Sinkhorn steps:
    /// new copy `j` mixes old copy `i` through `comb[i][j]`.
    pub comb: Vec<f32>,
}

/// The one-hot collapse the first sub-layer of a stack uses.
pub fn identity_pre(hc: usize) -> Vec<f32> {
    let mut pre = vec![0.0; hc];
    pre[0] = 1.0;
    pre
}

/// Coefficients from one token's stream `x` (`hc · dim`, copy-major).
///
/// `m = (W·x) · rsqrt(mean(x²) + norm_eps)`, then per group an affine map:
/// `pre = sigmoid(·) + eps`, `post = 2·sigmoid(·)`, and `comb` a row softmax
/// plus `eps`, a column normalisation, then `iters - 1` row-and-column
/// normalisations (each divides by `sum + eps`). Sums run left to right.
pub fn mixes(w: &MhcWeights, x: &[f32], hc: usize, norm_eps: f32, eps: f32, iters: usize) -> Mixes {
    let n = (2 + hc) * hc;
    assert_eq!(w.projection.rows, n);
    assert_eq!(x.len(), w.projection.cols);
    let rs = rsqrt32(dot(x, x) / x.len() as f32 + norm_eps);
    let m: Vec<f32> = (0..n).map(|j| dot(x, w.projection.row(j)) * rs).collect();

    let pre = (0..hc)
        .map(|j| sigmoid32(m[j] * w.scale[0] + w.base[j]) + eps)
        .collect();
    let post = (0..hc)
        .map(|j| 2.0 * sigmoid32(m[hc + j] * w.scale[1] + w.base[hc + j]))
        .collect();
    let mut comb: Vec<f32> = (2 * hc..n).map(|j| m[j] * w.scale[2] + w.base[j]).collect();

    let rows = |c: &mut [f32], f: &dyn Fn(f32, f32) -> f32| {
        for row in c.chunks_exact_mut(hc) {
            let sum = row.iter().fold(0.0f32, |a, v| a + v);
            for v in row {
                *v = f(*v, sum);
            }
        }
    };
    let cols = |c: &mut [f32]| {
        for k in 0..hc {
            let sum = (0..hc).fold(0.0f32, |a, j| a + c[j * hc + k]);
            for j in 0..hc {
                c[j * hc + k] /= sum + eps;
            }
        }
    };
    for row in comb.chunks_exact_mut(hc) {
        let max = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
        for v in row {
            *v = exp32(*v - max);
        }
    }
    rows(&mut comb, &|v, sum| v / sum + eps);
    cols(&mut comb);
    for _ in 1..iters {
        rows(&mut comb, &|v, sum| v / (sum + eps));
        cols(&mut comb);
    }
    Mixes { pre, post, comb }
}

/// Collapse the copies into one sub-layer input: `bf16(Σ_j pre[j]·x[j])`,
/// summed over copies left to right.
pub fn collapse(x: &[f32], pre: &[f32], out: &mut [f32]) {
    let dim = out.len();
    assert_eq!(x.len(), pre.len() * dim);
    for (d, o) in out.iter_mut().enumerate() {
        let mut s = 0.0f32;
        for (j, &p) in pre.iter().enumerate() {
            s += p * x[j * dim + d];
        }
        *o = bf16_round(s);
    }
}

/// Expand a sub-layer output `y` back into the copies, in place:
/// `x'[j] = bf16(post[j]·y + Σ_i comb[i][j]·x[i])`.
pub fn expand(x: &mut [f32], y: &[f32], mixes: &Mixes) {
    let (hc, dim) = (mixes.post.len(), y.len());
    assert_eq!(x.len(), hc * dim);
    let old = x.to_vec();
    for j in 0..hc {
        for d in 0..dim {
            let mut s = 0.0f32;
            for i in 0..hc {
                s += mixes.comb[i * hc + j] * old[i * dim + d];
            }
            x[j * dim + d] = bf16_round(mixes.post[j] * y[d] + s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comb_is_nearly_doubly_stochastic() {
        let hc = 4;
        let n = (2 + hc) * hc;
        let dim = 8;
        let w = MhcWeights {
            projection: Matrix::from_vec(
                n,
                hc * dim,
                (0..n * hc * dim)
                    .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.05)
                    .collect(),
            ),
            base: (0..n).map(|i| (i as f32 * 0.3).sin()).collect(),
            scale: [0.7, 0.9, 1.1],
        };
        let x: Vec<f32> = (0..hc * dim).map(|i| (i as f32 * 0.41).cos()).collect();
        let m = mixes(&w, &x, hc, 1e-20, 1e-6, 20);
        for k in 0..hc {
            let col: f32 = (0..hc).map(|j| m.comb[j * hc + k]).sum();
            let row: f32 = (0..hc).map(|j| m.comb[k * hc + j]).sum();
            assert!(
                (col - 1.0).abs() < 1e-4 && (row - 1.0).abs() < 1e-3,
                "{col} {row}"
            );
        }
        assert!(m.pre.iter().all(|p| *p > 0.0 && *p < 1.0 + 1e-5));
        assert!(m.post.iter().all(|p| *p > 0.0 && *p < 2.0));
    }
}
