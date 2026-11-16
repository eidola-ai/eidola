//! Rotary position embedding: the frequency table (with YaRN on compressed
//! layers) and the rotation of the last `rope_dim` elements of a vector.
//!
//! The reference builds `1 / base^(2i/d)` in f32, multiplies by the position in
//! f32, and takes `cos` / `sin` (`inference/model.py`, `precompute_freqs_cis`).
//! Adjacent elements pair up as one complex number.

use super::numeric::{bf16_round, det_ln, det_sin_cos, pow32};

/// YaRN frequency rescaling for sequences longer than the training context.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Yarn {
    pub original_seq_len: usize,
    pub factor: f32,
    pub beta_fast: f64,
    pub beta_slow: f64,
}

/// The per-pair frequencies of one RoPE variant.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeTable {
    pub freqs: Vec<f32>,
}

impl RopeTable {
    /// `rope_dim` rotated elements (so `rope_dim / 2` pairs) at base `theta`.
    pub fn new(rope_dim: usize, theta: f64, yarn: Option<Yarn>) -> Self {
        let pairs = rope_dim / 2;
        let mut freqs: Vec<f32> = (0..pairs)
            .map(|i| 1.0 / pow32(theta, (2 * i) as f32 / rope_dim as f32))
            .collect();
        if let Some(y) = yarn {
            // The pair whose wavelength completes `rotations` turns over the
            // training context.
            let corrected = |rotations: f64| {
                rope_dim as f64
                    * det_ln(y.original_seq_len as f64 / (rotations * 2.0 * std::f64::consts::PI))
                    / (2.0 * det_ln(theta))
            };
            let low = corrected(y.beta_fast).floor().max(0.0);
            let high = corrected(y.beta_slow).ceil().min((rope_dim - 1) as f64);
            let span = (high - low).max(1e-3) as f32;
            for (i, f) in freqs.iter_mut().enumerate() {
                let ramp = ((i as f32 - low as f32) / span).clamp(0.0, 1.0);
                let smooth = 1.0 - ramp;
                *f = *f / y.factor * (1.0 - smooth) + *f * smooth;
            }
        }
        RopeTable { freqs }
    }

    /// `(cos, sin)` of every pair's angle at `position`. The angle is the f32
    /// product `position · freq`.
    pub fn cis(&self, position: usize) -> Vec<(f32, f32)> {
        self.freqs
            .iter()
            .map(|&f| {
                let (sin, cos) = det_sin_cos((position as f32 * f) as f64);
                (cos as f32, sin as f32)
            })
            .collect()
    }
}

/// Rotate the last `2 · cis.len()` elements of `x` in place, rounding each to
/// BF16: `(a, b) → (a·c − b·s, a·s + b·c)`, nothing fused.
pub fn rotate(x: &mut [f32], cis: &[(f32, f32)]) {
    let start = x.len() - 2 * cis.len();
    for (pair, &(c, s)) in x[start..].chunks_exact_mut(2).zip(cis) {
        let (a, b) = (pair[0], pair[1]);
        pair[0] = bf16_round(a * c - b * s);
        pair[1] = bf16_round(a * s + b * c);
    }
}

/// The inverse rotation (multiplication by the conjugate):
/// `(a, b) → (a·c + b·s, b·c − a·s)`.
pub fn unrotate(x: &mut [f32], cis: &[(f32, f32)]) {
    let start = x.len() - 2 * cis.len();
    for (pair, &(c, s)) in x[start..].chunks_exact_mut(2).zip(cis) {
        let (a, b) = (pair[0], pair[1]);
        pair[0] = bf16_round(a * c + b * s);
        pair[1] = bf16_round(b * c - a * s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DeepSeek-V4.1-Flash's compressed-layer table: base 160000 with YaRN
    /// (original length 65536, factor 16, β 32 and 1) keeps pairs 0-15,
    /// divides pairs 25-31 by 16 and interpolates between.
    #[test]
    fn yarn_ramp_of_the_released_config() {
        let yarn = Yarn {
            original_seq_len: 65536,
            factor: 16.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let plain = RopeTable::new(64, 160000.0, None);
        let scaled = RopeTable::new(64, 160000.0, Some(yarn));
        for i in 0..=15 {
            assert_eq!(scaled.freqs[i], plain.freqs[i], "pair {i}");
        }
        for i in 25..32 {
            assert_eq!(scaled.freqs[i], plain.freqs[i] / 16.0, "pair {i}");
        }
        for i in 16..25 {
            assert!(scaled.freqs[i] < plain.freqs[i] && scaled.freqs[i] > plain.freqs[i] / 16.0);
        }
        assert_eq!(plain.freqs[0], 1.0);
    }

    #[test]
    fn rotation_touches_only_the_tail_and_inverts() {
        let table = RopeTable::new(4, 10000.0, None);
        let cis = table.cis(3);
        let mut x = [1.0f32, 2.0, 3.0, 4.0, 0.5, -0.25];
        let before = x;
        rotate(&mut x, &cis);
        assert_eq!(&x[..2], &before[..2]);
        assert_ne!(&x[2..], &before[2..]);
        unrotate(&mut x, &cis);
        for (a, b) in x.iter().zip(before) {
            assert!((a - b).abs() < 0.05);
        }
    }
}
