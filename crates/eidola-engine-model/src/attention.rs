//! Partial NeoX RoPE and single-query attention with GQA, sliding windows and
//! learned sinks.

use crate::config::AttentionSpec;
use crate::tensor::dot;

/// `cos`/`sin` for one position: `rope_dim` entries each, the second half a
/// copy of the first (`cat(freqs, freqs)`), computed in f32 as the HF
/// reference does: `inv_freq[i] = 1 / θ^(2i / rope_dim)`, angle
/// `= inv_freq[i] · position`.
pub fn rope_cos_sin(spec: &AttentionSpec, position: usize) -> (Vec<f32>, Vec<f32>) {
    let dim = spec.rope_dim;
    let half = dim / 2;
    let mut cos = vec![0.0f32; dim];
    let mut sin = vec![0.0f32; dim];
    let pos = position as f32;
    for i in 0..half {
        let exponent = (2 * i) as f32 / dim as f32;
        let inv_freq = 1.0f32 / spec.rope_theta.powf(exponent);
        let angle = inv_freq * pos;
        let (s, c) = angle.sin_cos();
        cos[i] = c;
        cos[i + half] = c;
        sin[i] = s;
        sin[i + half] = s;
    }
    (cos, sin)
}

/// Rotate-half RoPE on `head[..cos.len()]`; the remaining dims pass through.
pub fn apply_rope(head: &mut [f32], cos: &[f32], sin: &[f32]) {
    let dim = cos.len();
    let half = dim / 2;
    for i in 0..half {
        let x1 = head[i];
        let x2 = head[i + half];
        head[i] = x1 * cos[i] + (-x2) * sin[i];
        head[i + half] = x2 * cos[i + half] + x1 * sin[i + half];
    }
}

/// Whether a query at `q_pos` may attend to a key at `k_pos`.
#[inline]
pub fn visible(spec: &AttentionSpec, q_pos: usize, k_pos: usize) -> bool {
    k_pos <= q_pos && spec.window().is_none_or(|w| q_pos - k_pos < w)
}

/// Attention for one query head over its visible keys, given in ascending
/// position order. `sink` is the head's learned sink logit: it joins the
/// softmax normaliser and contributes no value.
///
/// Order of operations: scaled scores, max over scores and sink, `exp(s - m)`,
/// normaliser summed over keys in order then the sink term, probabilities
/// `p / Σ`, output accumulated over keys in order.
pub fn attend(
    spec: &AttentionSpec,
    q: &[f32],
    keys: &[&[f32]],
    values: &[&[f32]],
    sink: Option<f32>,
    out: &mut [f32],
) {
    debug_assert_eq!(keys.len(), values.len());
    let mut scores: Vec<f32> = keys
        .iter()
        .map(|k| dot(q, k) * spec.softmax_scale)
        .collect();
    let mut m = f32::NEG_INFINITY;
    for &s in &scores {
        m = m.max(s);
    }
    if let Some(s) = sink {
        m = m.max(s);
    }
    let mut denom = 0.0f32;
    for s in &mut scores {
        *s = (*s - m).exp();
        denom += *s;
    }
    if let Some(s) = sink {
        denom += (s - m).exp();
    }
    out.fill(0.0);
    for (p, v) in scores.iter().zip(values) {
        let p = p / denom;
        for (o, x) in out.iter_mut().zip(v.iter()) {
            *o += p * x;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AttentionKind;

    fn spec(window: Option<usize>) -> AttentionSpec {
        AttentionSpec {
            kind: match window {
                None => AttentionKind::Global,
                Some(window) => AttentionKind::Sliding { window },
            },
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim_qk: 4,
            head_dim_v: 2,
            rope_dim: 2,
            rope_theta: 10000.0,
            has_sinks: true,
            softmax_scale: 0.5,
        }
    }

    #[test]
    fn window_includes_self_and_window_minus_one_predecessors() {
        let s = spec(Some(3));
        assert!(visible(&s, 5, 5));
        assert!(visible(&s, 5, 3));
        assert!(!visible(&s, 5, 2));
        assert!(!visible(&s, 5, 6));
        assert!(visible(&spec(None), 5, 0));
    }

    #[test]
    fn sink_only_enlarges_the_denominator() {
        let s = spec(None);
        let q = [1.0, 0.0, 0.0, 0.0];
        let k = [1.0, 0.0, 0.0, 0.0];
        let v = [2.0, -4.0];
        let mut out = [0.0; 2];
        attend(&s, &q, &[&k], &[&v], None, &mut out);
        assert_eq!(out, [2.0, -4.0]);
        // One key with score 0.5 and a sink of 0.5: half the mass goes to the sink.
        attend(&s, &q, &[&k], &[&v], Some(0.5), &mut out);
        assert_eq!(out, [1.0, -2.0]);
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let s = spec(None);
        let (c, si) = rope_cos_sin(&s, 0);
        let mut h = [1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut h, &c, &si);
        assert_eq!(h, [1.0, 2.0, 3.0, 4.0]);
    }
}
