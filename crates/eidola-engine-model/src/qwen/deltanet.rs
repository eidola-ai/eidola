//! The Gated DeltaNet, one token at a time.
//!
//! The token recurrence is the definition: every token, in a prefill, a decode
//! step or a draft check, runs the same step on the state the token before it
//! left. A layer's state and output therefore depend only on the token prefix,
//! and a state saved after any token resumes exactly. The kernels here take a
//! state in and leave the next one in its place; an executor that wants bit
//! equality with this oracle calls them and does not reimplement them.
//!
//! Per token, for the BF16 input `x` of the layer's mixer
//! ([`DeltaNetWeights`] holds the weights):
//!
//! 1. `mixed = fp8(x · W_qkv)`, `z = fp8(x · W_z)`, `b = bf16(x · W_b)`,
//!    `a = bf16(x · W_a)` ([`super::linear`]).
//! 2. [`conv_step`]: a causal depthwise convolution of width 4 over `mixed`
//!    and the three raw rows before it (zeros before the sequence start),
//!    then SiLU.
//! 3. The convolved row splits into queries (key heads × key dim), keys and
//!    values (value heads × value dim); value head `h` reads key head
//!    `h / (value heads / key heads)`.
//! 4. [`gates`]: `β = bf16(σ(b))` and `α = exp(-exp(A_log) · softplus(a + dt_bias))`.
//! 5. [`l2_normalise`] queries and keys; queries are then divided by
//!    `sqrt(key dim)` ([`scale_query`]).
//! 6. [`recurrent_step`] per value head; the output is rounded to BF16.
//! 7. [`gated_norm`] per value head with `silu(z)`, then the output
//!    projection.

use rayon::prelude::*;

use super::config::DeltaNetSpec;
use super::numeric::{bf16_round, exp32, pinned_sum, rsqrt32, sigmoid32, silu32, softplus32};

/// The epsilon of the query and key L2 norms (inside the root).
pub const L2_EPS: f32 = 1e-6;

/// One layer's state between tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct DeltaNetState {
    /// The last `conv_kernel - 1` raw `mixed` rows (before the convolution),
    /// oldest first, `[conv_kernel - 1, conv_dim]`. Zeros at the sequence
    /// start.
    pub conv: Vec<f32>,
    /// Per value head a `[key dim, value dim]` f32 matrix `S`, row-major:
    /// `S[h][i][j]` at `(h · key_dim + i) · value_dim + j`.
    pub recurrent: Vec<f32>,
}

impl DeltaNetState {
    pub fn new(spec: &DeltaNetSpec) -> Self {
        DeltaNetState {
            conv: vec![0.0; (spec.conv_kernel - 1) * spec.conv_dim()],
            recurrent: vec![0.0; spec.state_len()],
        }
    }
}

/// The causal depthwise convolution for one token, then SiLU.
///
/// `history` holds the `K - 1` raw rows before this one, oldest first, and is
/// shifted to end with `mixed`. `weight` is `[channels, K]`, tap `K - 1`
/// multiplying the current token. Per channel the products are summed in tap
/// order, oldest first, in f32 (each product of two BF16 values is exact), and
/// rounded to BF16; then `out = bf16(silu(that))`.
pub fn conv_step(history: &mut [f32], mixed: &[f32], weight: &[f32], out: &mut [f32]) {
    let c = mixed.len();
    let k = weight.len() / c;
    assert_eq!(weight.len(), c * k);
    assert_eq!(history.len(), (k - 1) * c);
    assert_eq!(out.len(), c);
    for ch in 0..c {
        let w = &weight[ch * k..(ch + 1) * k];
        let mut acc = w[0] * if k == 1 { mixed[ch] } else { history[ch] };
        for (j, &wj) in w.iter().enumerate().skip(1) {
            let v = if j == k - 1 {
                mixed[ch]
            } else {
                history[j * c + ch]
            };
            acc += wj * v;
        }
        out[ch] = bf16_round(silu32(bf16_round(acc)));
    }
    if k > 1 {
        history.copy_within(c.., 0);
        history[(k - 2) * c..].copy_from_slice(mixed);
    }
}

/// `(β, α)` of one value head: `β = bf16(σ(b))`, and with
/// `g = -exp(A_log) · softplus(a + dt_bias)` (f32), `α = exp(g)`.
#[inline]
pub fn gates(a: f32, b: f32, a_log: f32, dt_bias: f32) -> (f32, f32) {
    let beta = bf16_round(sigmoid32(b));
    let g = -exp32(a_log) * softplus32(a + dt_bias);
    (beta, exp32(g))
}

/// `x · rsqrt(Σx² + 1e-6)` in place, the sum in the pinned order.
pub fn l2_normalise(x: &mut [f32]) {
    let sq: Vec<f32> = x.iter().map(|v| v * v).collect();
    let r = rsqrt32(pinned_sum(&sq) + L2_EPS);
    for v in x {
        *v *= r;
    }
}

/// Queries are divided by `sqrt(key dim)`, rounded to f32.
pub fn scale_query(q: &mut [f32]) {
    let d = (q.len() as f64).sqrt() as f32;
    for v in q {
        *v /= d;
    }
}

/// One step of one value head's recurrence, in f32:
///
/// ```text
/// S ← α·S;  r_j = Σ_i S_ij k_i;  δ_j = (v_j − r_j)·β;  S_ij ← S_ij + k_i δ_j;  o_j = Σ_i S_ij q_i
/// ```
///
/// Both sums run over the key dimension `i` in the pinned order
/// ([`pinned_sum`]). `state` is `[key dim, value dim]`; `out` gets `o`, not
/// yet rounded.
pub fn recurrent_step(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: f32,
    beta: f32,
    out: &mut [f32],
) {
    let dk = k.len();
    let dv = v.len();
    assert_eq!(state.len(), dk * dv);
    assert_eq!(q.len(), dk);
    assert_eq!(out.len(), dv);
    for s in state.iter_mut() {
        *s *= alpha;
    }
    let mut col = vec![0.0f32; dk];
    let mut delta = vec![0.0f32; dv];
    for j in 0..dv {
        for i in 0..dk {
            col[i] = state[i * dv + j] * k[i];
        }
        delta[j] = (v[j] - pinned_sum(&col)) * beta;
    }
    for i in 0..dk {
        for j in 0..dv {
            state[i * dv + j] += k[i] * delta[j];
        }
    }
    for j in 0..dv {
        for i in 0..dk {
            col[i] = state[i * dv + j] * q[i];
        }
        out[j] = pinned_sum(&col);
    }
}

/// The gated RMS norm of one value head:
/// `bf16( bf16( w · bf16(o · rsqrt(mean(o²) + eps)) ) · silu(z) )`, with
/// `o` the head's BF16 output, the plain (not zero-centred) weight `w`, and
/// `silu` in f32.
pub fn gated_norm(o: &[f32], z: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let sq: Vec<f32> = o.iter().map(|v| v * v).collect();
    let r = rsqrt32(pinned_sum(&sq) / o.len() as f32 + eps);
    for (((dst, &x), &g), &w) in out.iter_mut().zip(o).zip(z).zip(weight) {
        let n = bf16_round(w * bf16_round(x * r));
        *dst = bf16_round(n * silu32(g));
    }
}

/// The per-token part of the mixer, after the projections and before the
/// output projection: from `mixed`, `z`, `a` and `b` of one token to the
/// gated-norm output (`[value heads · value dim]`), advancing `state`.
#[allow(clippy::too_many_arguments)]
pub fn token_core(
    spec: &DeltaNetSpec,
    state: &mut DeltaNetState,
    mixed: &[f32],
    z: &[f32],
    a: &[f32],
    b: &[f32],
    w: &CoreWeights<'_>,
    eps: f32,
) -> Vec<f32> {
    let cd = spec.conv_dim();
    assert_eq!(mixed.len(), cd);
    let mut c = vec![0.0f32; cd];
    conv_step(&mut state.conv, mixed, w.conv, &mut c);
    let (kd, vd) = (spec.k_head_dim, spec.v_head_dim);
    let key_dim = spec.key_dim();
    let mut q = c[..key_dim].to_vec();
    let mut k = c[key_dim..2 * key_dim].to_vec();
    let v = &c[2 * key_dim..];
    for (qh, kh) in q.chunks_exact_mut(kd).zip(k.chunks_exact_mut(kd)) {
        l2_normalise(qh);
        l2_normalise(kh);
        scale_query(qh);
    }
    let group = spec.group();
    let mut out = vec![0.0f32; spec.value_dim()];
    state
        .recurrent
        .par_chunks_mut(kd * vd)
        .zip(out.par_chunks_mut(vd))
        .enumerate()
        .for_each(|(h, (s, o))| {
            let kh = h / group;
            let (beta, alpha) = gates(a[h], b[h], w.a_log[h], w.dt_bias[h]);
            let mut raw = vec![0.0f32; vd];
            recurrent_step(
                s,
                &q[kh * kd..(kh + 1) * kd],
                &k[kh * kd..(kh + 1) * kd],
                &v[h * vd..(h + 1) * vd],
                alpha,
                beta,
                &mut raw,
            );
            for r in raw.iter_mut() {
                *r = bf16_round(*r);
            }
            gated_norm(&raw, &z[h * vd..(h + 1) * vd], w.norm, eps, o);
        });
    out
}

/// The small per-layer tensors [`token_core`] reads.
pub struct CoreWeights<'a> {
    /// `[conv_dim, conv_kernel]`.
    pub conv: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    /// The gated norm's plain weight, `[value dim]`.
    pub norm: &'a [f32],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_reads_zeros_before_the_start_and_the_newest_tap_last() {
        // Two channels, width 4: channel 0 weights [1, 2, 4, 8], channel 1 [0, 0, 0, 1].
        let w = [1.0, 2.0, 4.0, 8.0, 0.0, 0.0, 0.0, 1.0];
        let mut hist = vec![0.0f32; 6];
        let mut out = [0.0f32; 2];
        conv_step(&mut hist, &[1.0, 3.0], &w, &mut out);
        assert_eq!(out, [bf16_round(silu32(8.0)), bf16_round(silu32(3.0))]);
        conv_step(&mut hist, &[0.5, -1.0], &w, &mut out);
        // Channel 0: 4·1 + 8·0.5 = 8.
        assert_eq!(out[0], bf16_round(silu32(8.0)));
        assert_eq!(hist, [0.0, 0.0, 1.0, 3.0, 0.5, -1.0]);
    }

    #[test]
    fn a_decay_of_one_and_beta_of_one_writes_the_value() {
        // With S = 0, α = 1, β = 1 and a unit key, S becomes k vᵀ and o = (q·k) v.
        let (dk, dv) = (4, 3);
        let mut s = vec![0.0f32; dk * dv];
        let k = [1.0, 0.0, 0.0, 0.0];
        let q = [0.5, 0.0, 0.0, 0.0];
        let v = [2.0, -1.0, 4.0];
        let mut o = [0.0f32; 3];
        recurrent_step(&mut s, &q, &k, &v, 1.0, 1.0, &mut o);
        assert_eq!(o, [1.0, -0.5, 2.0]);
        // A second write of the same value with the same key changes nothing.
        recurrent_step(&mut s, &q, &k, &v, 1.0, 1.0, &mut o);
        assert_eq!(o, [1.0, -0.5, 2.0]);
        // Decay is applied before the read: α = ½ halves S, and δ restores it.
        recurrent_step(&mut s, &q, &k, &v, 0.5, 1.0, &mut o);
        assert_eq!(o, [1.0, -0.5, 2.0]);
        recurrent_step(&mut s, &q, &k, &v, 0.5, 0.0, &mut o);
        assert_eq!(o, [0.5, -0.25, 1.0]);
    }
}
