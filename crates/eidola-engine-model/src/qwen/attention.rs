//! Gated full attention: RoPE, the attention core and the output gate.
//!
//! **RoPE.** The released model's M-RoPE gives text the same position in all
//! three streams, so for text it is ordinary partial NeoX rotation: the first
//! `rotary_dim` dimensions of each query and key head, pairs `(j, j + d/2)`.
//! The tables are `inv_freq_i = 1 / θ^(2i/d)` (f32), the angle the f32 product
//! `position · inv_freq_i`, its cosine and sine rounded to BF16, duplicated
//! over the two halves. The rotation rounds every operation to BF16 as the
//! reference's BF16 tensors do: `x' = bf16(bf16(x·cos) + bf16(rot(x)·sin))`
//! with `rot(x) = (−x[d/2..], x[..d/2])`.
//!
//! **Attention core** (what the reference leaves to a kernel, fixed here):
//! per query and head, scores are f32 dot products in the pinned order times
//! the softmax scale, over every visible key in ascending position; the
//! maximum is subtracted, weights are `exp` of that, the normaliser is their
//! sum left to right, and the output is `Σ (w / Σw) · v` in ascending
//! position, rounded once to BF16.

use super::config::AttentionSpec;
use super::numeric::{bf16_round, det_sin_cos, exp32, pow32};
use crate::tensor::dot;

/// `inv_freq_i = 1 / θ^(2i/d)` for `i < d/2`, `d` the rotary width.
pub fn inv_freq(spec: &AttentionSpec) -> Vec<f32> {
    let d = spec.rotary_dim;
    (0..d / 2)
        .map(|i| 1.0 / pow32(spec.rope_theta, (2 * i) as f32 / d as f32))
        .collect()
}

/// `(cos, sin)` at `position`, each `rotary_dim` BF16 values (the half table
/// repeated).
pub fn rope_cos_sin(inv_freq: &[f32], position: usize) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let mut cos = vec![0.0f32; 2 * half];
    let mut sin = vec![0.0f32; 2 * half];
    for (i, &f) in inv_freq.iter().enumerate() {
        let angle = position as f32 * f;
        let (s, c) = det_sin_cos(angle as f64);
        let (s, c) = (bf16_round(s as f32), bf16_round(c as f32));
        cos[i] = c;
        cos[i + half] = c;
        sin[i] = s;
        sin[i + half] = s;
    }
    (cos, sin)
}

/// Rotate the first `cos.len()` elements of one head in place.
pub fn apply_rope(head: &mut [f32], cos: &[f32], sin: &[f32]) {
    let d = cos.len();
    let half = d / 2;
    let x: Vec<f32> = head[..d].to_vec();
    for j in 0..d {
        let rot = if j < half { -x[j + half] } else { x[j - half] };
        head[j] = bf16_round(bf16_round(x[j] * cos[j]) + bf16_round(rot * sin[j]));
    }
}

/// The attention core for one query head over `keys` and `values` (the
/// visible positions of its KV head, ascending), into `out` (BF16 values).
pub fn attend(q: &[f32], keys: &[&[f32]], values: &[&[f32]], scale: f32, out: &mut [f32]) {
    debug_assert_eq!(keys.len(), values.len());
    let mut w: Vec<f32> = keys.iter().map(|k| dot(q, k) * scale).collect();
    let m = w.iter().fold(f32::NEG_INFINITY, |m, &s| m.max(s));
    let mut denom = 0.0f32;
    for s in &mut w {
        *s = exp32(*s - m);
        denom += *s;
    }
    out.fill(0.0);
    for (p, v) in w.iter().zip(values) {
        let p = p / denom;
        for (o, x) in out.iter_mut().zip(v.iter()) {
            *o += p * x;
        }
    }
    for o in out {
        *o = bf16_round(*o);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> AttentionSpec {
        AttentionSpec {
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            rotary_dim: 4,
            rope_theta: 1e7,
            softmax_scale: 0.5,
        }
    }

    #[test]
    fn position_zero_is_the_identity_and_the_tail_passes_through() {
        let f = inv_freq(&spec());
        assert_eq!(f[0], 1.0);
        let (c, s) = rope_cos_sin(&f, 0);
        let mut h = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        apply_rope(&mut h, &c, &s);
        assert_eq!(h, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let (c, s) = rope_cos_sin(&f, 3);
        apply_rope(&mut h, &c, &s);
        assert_eq!(&h[4..], &[5.0, 6.0, 7.0, 8.0]);
        // Dimension 0 pairs with dimension 2 (half the rotary width), not 1.
        let mut a = [1.0, 0.0, 0.0, 0.0];
        apply_rope(&mut a, &c, &s);
        assert_eq!(a[1], 0.0);
        assert_eq!(a[2], bf16_round(s[2]));
    }

    #[test]
    fn one_key_returns_its_value() {
        let mut out = [0.0f32; 2];
        attend(&[1.0, 0.0], &[&[3.0, 1.0]], &[&[2.0, -4.0]], 0.5, &mut out);
        assert_eq!(out, [2.0, -4.0]);
    }
}
