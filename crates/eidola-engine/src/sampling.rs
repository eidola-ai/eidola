//! CPU reference sampling: the exact semantics every executor's device sampler implements.
//!
//! # Definition
//!
//! Given a logit row `l` (finite or `-inf`) and [`SamplingParams`]:
//!
//! 0. **Sampleable vocabulary**: `l` is the row's first `sampleable_vocab_size` entries
//!    ([`Logits`]). A model's head can be wider than its tokenizer (MiMo's is padded to
//!    152,576 rows for 151,675 tokens); padded ids have no token and are outside every
//!    distribution below, so no sample, draft, acceptance or residual draw can return
//!    one. Every function here takes [`Logits`] or rows derived from it, so the limit is
//!    applied by construction rather than by each caller remembering to mask.
//! 1. **Greedy** (`temperature == 0`): the distribution is one-hot at `argmax l`, ties to
//!    the lowest token id. No random draw is consumed.
//! 2. Otherwise `z = l / temperature`, `p = softmax(z)` (max-subtracted).
//! 3. **Top-k** (`top_k > 0`): order tokens by `(p descending, id ascending)`; keep the
//!    first `top_k`.
//! 4. **Top-p** (`top_p < 1`): renormalize the kept set; in the same order keep the
//!    shortest prefix whose cumulative mass is `>= top_p` (always at least one token).
//! 5. **Min-p** (`min_p > 0`): keep tokens with `p >= min_p * max(p)`.
//! 6. Renormalize the kept set; everything else has probability exactly 0.
//! 7. **Draw**: `u = uniform(seed, position, stream)`; the token is the smallest id `i`
//!    (in vocabulary order) with `Σ_{j<=i} p_j > u`. If rounding leaves no such id, the
//!    largest id with `p > 0` is chosen.
//!
//! `position` is the absolute sequence position of the token being chosen, so a draw
//! depends only on `(seed, position, stream)` — never on batch composition, step
//! boundaries, preemption, or chunking. That is what makes recompute-based preemption and
//! chunked prefill reproduce an uninterrupted run exactly.
//!
//! # Precision
//!
//! This reference accumulates in `f64` in vocabulary order. A device sampler reducing in
//! `f32` in parallel produces the same token except when `u` lies within rounding distance
//! of a CDF boundary; that is the only permitted divergence.
//!
//! # Speculative acceptance
//!
//! [`chain_accept`] is chain speculative sampling over `k` drafts: draft `d_i` drawn from
//! the drafter's processed distribution `q_i` is accepted iff `u_i * q_i(d_i) < p_i(d_i)`
//! (i.e. with probability `min(1, p/q)`); on the first rejection the replacement is drawn
//! from `normalize(max(0, p_i - q_i))`; if all `k` are accepted a bonus token is drawn
//! from `p_k`. Output tokens are distributed exactly as non-speculative sampling from `p`.

/// Per-sequence sampling parameters (device-side arrays carry the same five fields).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    /// 0 means greedy.
    pub temperature: f32,
    /// 0 disables.
    pub top_k: u32,
    /// 1.0 disables.
    pub top_p: f32,
    /// 0.0 disables.
    pub min_p: f32,
    /// Counter-based RNG seed.
    pub seed: u64,
}

impl SamplingParams {
    /// Greedy decoding.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
        }
    }

    /// Plain temperature sampling with `seed`.
    pub fn random(temperature: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed,
        }
    }

    /// A fresh seed from the operating system's CSPRNG, for requests that did not ask for
    /// reproducibility. The seed is per request and fixed for its lifetime, so preemption
    /// never changes its output.
    pub fn random_seed() -> u64 {
        crate::secret::random_u64()
    }

    /// Whether this is greedy decoding.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// Random streams; each consumer of randomness at a position uses its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum Stream {
    /// Ordinary sampling and the speculative bonus token.
    Sample = 0,
    /// Speculative acceptance test.
    Accept = 1,
    /// Speculative replacement draw from the residual.
    Residual = 2,
    /// The drafter's own proposal draw.
    Draft = 3,
}

/// SplitMix64 finalizer.
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The counter-based random word for `(seed, position, stream)`:
/// `mix64(mix64(seed) ^ mix64(position * 0x9E3779B97F4A7C15 + stream))`.
pub fn random_word(seed: u64, position: u64, stream: Stream) -> u64 {
    mix64(
        mix64(seed)
            ^ mix64(
                position
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(stream as u64),
            ),
    )
}

/// A uniform draw in `[0, 1)` with 24 bits of precision (exactly representable in `f32`).
pub fn uniform(seed: u64, position: u64, stream: Stream) -> f64 {
    (random_word(seed, position, stream) >> 40) as f64 * (1.0 / (1u64 << 24) as f64)
}

/// A logit row restricted to the sampleable vocabulary (step 0 of the definition).
///
/// The only constructors state the limit, so every sampler input has padded ids removed;
/// every id a sampler returns indexes this prefix.
#[derive(Clone, Copy, Debug)]
pub struct Logits<'a>(&'a [f32]);

impl<'a> Logits<'a> {
    /// The first `sampleable` entries of `row`. Panics unless `0 < sampleable <=
    /// row.len()`.
    pub fn new(row: &'a [f32], sampleable: u32) -> Self {
        let n = sampleable as usize;
        assert!(
            n > 0 && n <= row.len(),
            "sampleable vocabulary {sampleable} outside a logit row of {}",
            row.len()
        );
        Self(&row[..n])
    }

    /// The sampleable logits.
    pub fn as_slice(&self) -> &'a [f32] {
        self.0
    }

    /// The sampleable vocabulary size.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Always false: a sampleable vocabulary is never empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Lowest-id argmax.
pub fn argmax(logits: Logits<'_>) -> u32 {
    let logits = logits.as_slice();
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// The processed distribution (steps 1–6 above) over the sampleable vocabulary.
pub fn processed_probs(logits: Logits<'_>, params: &SamplingParams) -> Vec<f64> {
    let n = logits.len();
    let mut p = vec![0.0f64; n];
    if params.is_greedy() {
        p[argmax(logits) as usize] = 1.0;
        return p;
    }
    let logits = logits.as_slice();
    let t = params.temperature as f64;
    let max = logits
        .iter()
        .fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64 / t));
    let mut sum = 0.0;
    for (pi, &l) in p.iter_mut().zip(logits) {
        *pi = ((l as f64 / t) - max).exp();
        sum += *pi;
    }
    for pi in &mut p {
        *pi /= sum;
    }
    let filtering = params.top_k > 0 || params.top_p < 1.0 || params.min_p > 0.0;
    if !filtering {
        return p;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    let mut keep = n;
    if params.top_k > 0 {
        keep = keep.min(params.top_k as usize);
    }
    if params.top_p < 1.0 {
        let mass: f64 = order[..keep].iter().map(|&i| p[i]).sum();
        let target = params.top_p as f64 * mass;
        let mut cum = 0.0;
        let mut cut = keep;
        for (rank, &i) in order[..keep].iter().enumerate() {
            cum += p[i];
            if cum >= target {
                cut = rank + 1;
                break;
            }
        }
        keep = cut.max(1);
    }
    if params.min_p > 0.0 {
        let floor = params.min_p as f64 * p[order[0]];
        let cut = order[..keep].iter().take_while(|&&i| p[i] >= floor).count();
        keep = cut.max(1);
    }
    let mut out = vec![0.0f64; n];
    let mass: f64 = order[..keep].iter().map(|&i| p[i]).sum();
    for &i in &order[..keep] {
        out[i] = p[i] / mass;
    }
    out
}

/// Inverse-CDF draw in vocabulary order (step 7).
pub fn sample_from(probs: &[f64], u: f64) -> u32 {
    let total: f64 = probs.iter().sum();
    let target = u * total;
    let mut cum = 0.0;
    let mut last_nonzero = 0usize;
    for (i, &p) in probs.iter().enumerate() {
        if p > 0.0 {
            last_nonzero = i;
            cum += p;
            if cum > target {
                return i as u32;
            }
        }
    }
    last_nonzero as u32
}

/// Samples the token at sequence `position` from `logits`.
pub fn sample(logits: Logits<'_>, params: &SamplingParams, position: u64) -> u32 {
    if params.is_greedy() {
        return argmax(logits);
    }
    let p = processed_probs(logits, params);
    sample_from(&p, uniform(params.seed, position, Stream::Sample))
}

/// Chain speculative acceptance.
///
/// * `target[i]` is the processed target distribution for the token at
///   `first_position + i` (`k + 1` rows; row `k` serves the bonus token).
/// * `draft[i]` is the processed drafter distribution the draft `drafts[i]` was drawn from.
///
/// Every row comes from [`processed_probs`] over the same sampleable vocabulary, so every
/// accepted draft, residual draw and bonus token is a sampleable id.
///
/// Returns the accepted drafts followed by exactly one replacement or bonus token
/// (`1..=k+1` tokens).
pub fn chain_accept(
    target: &[Vec<f64>],
    draft: &[Vec<f64>],
    drafts: &[u32],
    params: &SamplingParams,
    first_position: u64,
) -> Vec<u32> {
    let k = drafts.len();
    assert_eq!(target.len(), k + 1, "target rows must be k + 1");
    assert_eq!(draft.len(), k, "draft rows must be k");
    let n = target[0].len();
    assert!(
        target.iter().chain(draft).all(|r| r.len() == n),
        "target and draft rows must cover the same sampleable vocabulary"
    );
    assert!(
        drafts.iter().all(|&d| (d as usize) < n),
        "draft outside the vocabulary"
    );
    let mut out = Vec::with_capacity(k + 1);
    for i in 0..k {
        let pos = first_position + i as u64;
        let d = drafts[i] as usize;
        let (p, q) = (&target[i], &draft[i]);
        let accepted = if params.is_greedy() {
            p[d] > 0.0
        } else {
            uniform(params.seed, pos, Stream::Accept) * q[d] < p[d]
        };
        if accepted {
            out.push(drafts[i]);
            continue;
        }
        let token = if params.is_greedy() {
            argmax_f64(p)
        } else {
            let residual: Vec<f64> = p.iter().zip(q).map(|(a, b)| (a - b).max(0.0)).collect();
            let row = if residual.iter().sum::<f64>() > 0.0 {
                &residual
            } else {
                p
            };
            sample_from(row, uniform(params.seed, pos, Stream::Residual))
        };
        out.push(token);
        return out;
    }
    let pos = first_position + k as u64;
    let bonus = if params.is_greedy() {
        argmax_f64(&target[k])
    } else {
        sample_from(&target[k], uniform(params.seed, pos, Stream::Sample))
    };
    out.push(bonus);
    out
}

fn argmax_f64(p: &[f64]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in p.iter().enumerate() {
        if v > p[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_ties_go_low() {
        assert_eq!(argmax(full(&[1.0, 3.0, 3.0, 2.0])), 1);
        assert_eq!(
            sample(full(&[1.0, 3.0, 3.0]), &SamplingParams::greedy(), 7),
            1
        );
    }

    fn full(row: &[f32]) -> Logits<'_> {
        Logits::new(row, row.len() as u32)
    }

    /// Padded ids are outside every distribution: the padded entries here would win
    /// greedy decoding and dominate sampling, yet nothing returns one.
    #[test]
    fn padded_ids_are_never_sampled() {
        let row = [0.0f32, 1.0, 0.5, 40.0, 50.0];
        let logits = Logits::new(&row, 3);
        assert_eq!(argmax(full(&row)), 4, "unmasked, the padding wins");
        assert_eq!(argmax(logits), 1);
        assert_eq!(sample(logits, &SamplingParams::greedy(), 0), 1);
        let params = SamplingParams::random(1.0, 9);
        let probs = processed_probs(logits, &params);
        assert_eq!(probs.len(), 3);
        for pos in 0..2000 {
            assert!(sample(logits, &params, pos) < 3);
        }
        let draft = processed_probs(Logits::new(&[3.0, 0.0, 0.0, 9.0], 3), &params);
        for pos in 0..500 {
            let out = chain_accept(
                &[probs.clone(), probs.clone()],
                std::slice::from_ref(&draft),
                &[0],
                &params,
                pos,
            );
            assert!(out.iter().all(|&t| t < 3), "{out:?}");
        }
    }

    #[test]
    #[should_panic(expected = "sampleable vocabulary")]
    fn empty_sampleable_vocabulary_is_refused() {
        Logits::new(&[1.0], 0);
    }

    #[test]
    fn uniform_is_deterministic_and_in_range() {
        for pos in 0..1000 {
            let u = uniform(42, pos, Stream::Sample);
            assert!((0.0..1.0).contains(&u));
            assert_eq!(u, uniform(42, pos, Stream::Sample));
            assert_ne!(u, uniform(42, pos, Stream::Accept));
        }
    }

    #[test]
    fn filters_compose_as_documented() {
        let logits = [0.0f32, 1.0, 2.0, 3.0, 3.0];
        let mut p = SamplingParams::random(1.0, 0);
        p.top_k = 2;
        let probs = processed_probs(full(&logits), &p);
        assert!(probs[3] > 0.0 && probs[4] > 0.0);
        assert_eq!(probs[..3], [0.0, 0.0, 0.0]);
        assert!((probs[3] - 0.5).abs() < 1e-12);

        let mut p = SamplingParams::random(1.0, 0);
        p.top_p = 0.5;
        let probs = processed_probs(full(&logits), &p);
        // Tokens 3 and 4 tie; 3 ranks first. Its mass alone is < 0.5, so 4 joins.
        assert!(probs[3] > 0.0 && probs[4] > 0.0 && probs[2] == 0.0);

        let mut p = SamplingParams::random(1.0, 0);
        p.min_p = 0.3;
        let probs = processed_probs(full(&logits), &p);
        // exp(-1) = 0.37 >= 0.3 keeps token 2; exp(-2) = 0.135 drops token 1.
        assert!(probs[2] > 0.0 && probs[1] == 0.0);
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn inverse_cdf_respects_boundaries() {
        let p = [0.25, 0.0, 0.5, 0.25];
        assert_eq!(sample_from(&p, 0.0), 0);
        assert_eq!(sample_from(&p, 0.2499), 0);
        assert_eq!(sample_from(&p, 0.25), 2);
        assert_eq!(sample_from(&p, 0.7499), 2);
        assert_eq!(sample_from(&p, 0.75), 3);
        assert_eq!(sample_from(&p, 0.999_999), 3);
    }

    #[test]
    fn greedy_chain_accept_matches_argmax() {
        let one_hot = |i: usize| {
            let mut v = vec![0.0; 4];
            v[i] = 1.0;
            v
        };
        let target = vec![one_hot(1), one_hot(2), one_hot(3)];
        let draft = vec![one_hot(1), one_hot(0)];
        let out = chain_accept(&target, &draft, &[1, 0], &SamplingParams::greedy(), 10);
        assert_eq!(out, vec![1, 2]);
        let draft = vec![one_hot(1), one_hot(2)];
        let out = chain_accept(&target, &draft, &[1, 2], &SamplingParams::greedy(), 10);
        assert_eq!(out, vec![1, 2, 3]);
    }

    /// Exact check of the acceptance rule's marginal for one step: P(token) under chain
    /// acceptance equals p, computed analytically from the rule (no sampling noise).
    #[test]
    fn one_step_acceptance_marginal_is_exact() {
        let p = [0.1, 0.4, 0.3, 0.2];
        let q = [0.4, 0.1, 0.3, 0.2];
        let mut marginal = [0.0f64; 4];
        let residual: Vec<f64> = p
            .iter()
            .zip(&q)
            .map(|(a, b): (&f64, &f64)| (a - b).max(0.0))
            .collect();
        let rsum: f64 = residual.iter().sum();
        for d in 0..4 {
            let acc = (p[d] / q[d]).min(1.0);
            marginal[d] += q[d] * acc;
            for t in 0..4 {
                marginal[t] += q[d] * (1.0 - acc) * residual[t] / rsum;
            }
        }
        for t in 0..4 {
            assert!((marginal[t] - p[t]).abs() < 1e-12);
        }
    }
}
