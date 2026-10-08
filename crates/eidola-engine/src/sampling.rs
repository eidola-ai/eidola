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
//! 2. Otherwise `z = l / temperature`, `p = softmax(z)` (max-subtracted, exponentials
//!    from [`det_exp`]).
//! 3. **Top-k** (`top_k > 0`): order tokens by `(p descending, id ascending)`; keep the
//!    first `top_k`.
//! 4. **Top-p** (`top_p < 1`): renormalize the kept set; in the same order keep the
//!    shortest prefix whose mass is `>= top_p` (always at least one token).
//! 5. **Min-p** (`min_p > 0`): keep tokens with `p >= min_p * max(p)`.
//! 6. Renormalize the kept set; everything else has probability exactly 0.
//! 7. **Draw**: `u = uniform(seed, position, stream)`; the token is the smallest id `i`
//!    (in vocabulary order) with `Σ_{j<=i} p_j > u · Σ_j p_j`. If rounding leaves no such
//!    id, the largest id with `p > 0` is chosen.
//!
//! `position` is the absolute sequence position of the token being chosen, so a draw
//! depends only on `(seed, position, stream)` — never on batch composition, step
//! boundaries, preemption, or chunking. That is what makes recompute-based preemption and
//! chunked prefill reproduce an uninterrupted run exactly.
//!
//! # Precision: pinned arithmetic
//!
//! The arithmetic is `f64` and fixed down to the last bit, so that a parallel device
//! sampler reproduces every probability, and therefore every token, exactly:
//!
//! * **Exponentials** come from [`det_exp`], built from correctly rounded `+ - * /` only
//!   (no FMA contraction), never from the platform's `exp`, whose last bit differs between
//!   C libraries (macOS and glibc disagree on about 0.2 % of softmax-range arguments).
//! * **Every sum of a set of probabilities** is [`pinned_sum`] in vocabulary order: each
//!   chunk of [`SUM_CHUNK`] consecutive ids summed left to right, then the chunk sums left
//!   to right. A mass "in rank order" (top-p's prefixes) is the pinned sum of that set in
//!   vocabulary order. Because adding a non-negative term never lowers a rounded partial
//!   sum, these masses grow with the set, so "the shortest prefix whose mass is `>= x`" is
//!   well defined.
//! * **Cumulative sums** for the draw follow the same chunking: the prefix through id `i`
//!   in chunk `c` is `(sum of the chunk sums before c) + (left-to-right sum of chunk c
//!   through i)`, and its last value is exactly the pinned total.
//!
//! Max, comparisons, `f32 → f64` conversion and the divisions are exact or correctly
//! rounded on every IEEE platform. A device sampler that performs these same operations in
//! this same grouping returns the same token as this reference, bit for bit, given equal
//! logits.
//!
//! # Speculative acceptance
//!
//! [`chain_accept`] is chain speculative sampling over `k` drafts: draft `d_i` drawn from
//! the drafter's processed distribution `q_i` is accepted iff `u_i * q_i(d_i) < p_i(d_i)`
//! (i.e. with probability `min(1, p/q)`); on the first rejection the replacement is drawn
//! from `normalize(max(0, p_i - q_i))`; if all `k` are accepted a bonus token is drawn
//! from `p_k`. Output tokens are distributed exactly as non-speculative sampling from `p`.

/// Per-sequence sampling parameters (device-side arrays carry the same five fields).
///
/// Always valid: the fields are private and every constructor checks them, so a value the
/// sampler cannot define (a NaN temperature, `top_p = 0`, …) is unrepresentable rather
/// than silently turned into wrong tokens. The ranges:
///
/// * `temperature`: finite and `>= 0`; 0 means greedy.
/// * `top_k`: any; 0 disables.
/// * `top_p`: in `(0, 1]`; 1 disables.
/// * `min_p`: in `[0, 1]`; 0 disables.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    seed: u64,
}

/// Sampling parameters outside the ranges [`SamplingParams`] documents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidSampling(pub String);

impl std::fmt::Display for InvalidSampling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid sampling parameters: {}", self.0)
    }
}

impl std::error::Error for InvalidSampling {}

impl SamplingParams {
    /// Checked construction.
    pub fn new(
        temperature: f32,
        top_k: u32,
        top_p: f32,
        min_p: f32,
        seed: u64,
    ) -> Result<Self, InvalidSampling> {
        let bad = |m: String| Err(InvalidSampling(m));
        if !(temperature.is_finite() && temperature >= 0.0) {
            return bad(format!("temperature {temperature} must be finite and >= 0"));
        }
        if !(top_p > 0.0 && top_p <= 1.0) {
            return bad(format!("top_p {top_p} must be in (0, 1]"));
        }
        if !(0.0..=1.0).contains(&min_p) {
            return bad(format!("min_p {min_p} must be in [0, 1]"));
        }
        Ok(Self {
            temperature,
            top_k,
            top_p,
            min_p,
            seed,
        })
    }

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
    pub fn random(temperature: f32, seed: u64) -> Result<Self, InvalidSampling> {
        Self::new(temperature, 0, 1.0, 0.0, seed)
    }

    /// The same parameters with another seed (every seed is valid).
    pub fn with_seed(self, seed: u64) -> Self {
        Self { seed, ..self }
    }

    /// 0 means greedy.
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// 0 disables.
    pub fn top_k(&self) -> u32 {
        self.top_k
    }

    /// 1 disables.
    pub fn top_p(&self) -> f32 {
        self.top_p
    }

    /// 0 disables.
    pub fn min_p(&self) -> f32 {
        self.min_p
    }

    /// Counter-based RNG seed.
    pub fn seed(&self) -> u64 {
        self.seed
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

/// Ids per chunk of a [`pinned_sum`].
pub const SUM_CHUNK: usize = 1024;

/// The sum of `x` in the pinned order: each [`SUM_CHUNK`] of consecutive entries summed
/// left to right from 0, then the chunk sums left to right from 0.
pub fn pinned_sum(x: &[f64]) -> f64 {
    let mut total = 0.0;
    for chunk in x.chunks(SUM_CHUNK) {
        let mut s = 0.0;
        for &v in chunk {
            s += v;
        }
        total += s;
    }
    total
}

/// [`pinned_sum`] of the entries `keep` selects (the others count as 0).
fn pinned_sum_where(x: &[f64], keep: impl Fn(usize) -> bool) -> f64 {
    let mut total = 0.0;
    for (c, chunk) in x.chunks(SUM_CHUNK).enumerate() {
        let mut s = 0.0;
        for (j, &v) in chunk.iter().enumerate() {
            if keep(c * SUM_CHUNK + j) {
                s += v;
            }
        }
        total += s;
    }
    total
}

/// Coefficients `1 / k!` (correctly rounded) of the degree-13 Taylor polynomial of `e^r`.
const EXP_COEFFS: [u64; 14] = [
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3fe0000000000000,
    0x3fc5555555555555,
    0x3fa5555555555555,
    0x3f81111111111111,
    0x3f56c16c16c16c17,
    0x3f2a01a01a01a01a,
    0x3efa01a01a01a01a,
    0x3ec71de3a556c734,
    0x3e927e4fb7789f5c,
    0x3e5ae64567f544e4,
    0x3e21eed8eff8d898,
    0x3de6124613a86d09,
];

/// `e^x` from correctly rounded `+ - * /` alone, so every IEEE platform (and a device
/// kernel that avoids FMA contraction) computes the same bits. Accurate to a few ulp:
///
/// 1. `n = round(x / ln 2)` (half away from zero), `r = (x - n·ln2_hi) - n·ln2_lo`
///    (Cody–Waite; `ln2_hi` has 32 significant bits, so `n·ln2_hi` is exact);
/// 2. `e^r` by Horner over [`EXP_COEFFS`] (`|r| <= 0.35`, truncation below 1e-17);
/// 3. scaled by `2^n` with power-of-two multiplications (two of them below `2^-1022`,
///    where the second rounds once into the subnormals).
pub fn det_exp(x: f64) -> f64 {
    const INV_LN2: f64 = f64::from_bits(0x3ff71547652b82fe);
    const LN2_HI: f64 = f64::from_bits(0x3fe62e42fee00000);
    const LN2_LO: f64 = f64::from_bits(0x3dea39ef35793c76);
    if x.is_nan() {
        return x;
    }
    if x > 709.782712893384 {
        return f64::INFINITY;
    }
    if x < -745.2 {
        return 0.0;
    }
    let n = (x * INV_LN2).round();
    let r = (x - n * LN2_HI) - n * LN2_LO;
    let mut p = f64::from_bits(EXP_COEFFS[13]);
    for k in (0..13).rev() {
        p = p * r + f64::from_bits(EXP_COEFFS[k]);
    }
    let n = n as i64;
    let pow2 = |e: i64| f64::from_bits(((e + 1023) as u64) << 52);
    if n > 1023 {
        p * pow2(1023) * pow2(n - 1023)
    } else if n < -1022 {
        p * pow2(-600) * pow2(n + 600)
    } else {
        p * pow2(n)
    }
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
    for (pi, &l) in p.iter_mut().zip(logits) {
        *pi = det_exp((l as f64 / t) - max);
    }
    let sum = pinned_sum(&p);
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
    // rank[i]: position of id i in `order`.
    let mut rank = vec![0usize; n];
    for (r, &i) in order.iter().enumerate() {
        rank[i] = r;
    }
    if params.top_p < 1.0 {
        let mass = pinned_sum_where(&p, |i| rank[i] < keep);
        let target = params.top_p as f64 * mass;
        // The mass of the first `r` ranks grows with `r`: binary search for the
        // shortest prefix reaching the target (the whole kept set always does).
        let (mut lo, mut hi) = (1usize, keep);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if pinned_sum_where(&p, |i| rank[i] < mid) >= target {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        keep = lo.max(1);
    }
    if params.min_p > 0.0 {
        let floor = params.min_p as f64 * p[order[0]];
        let cut = order[..keep].iter().take_while(|&&i| p[i] >= floor).count();
        keep = cut.max(1);
    }
    let mut out = vec![0.0f64; n];
    let mass = pinned_sum_where(&p, |i| rank[i] < keep);
    for &i in &order[..keep] {
        out[i] = p[i] / mass;
    }
    out
}

/// Inverse-CDF draw in vocabulary order (step 7), with the pinned chunked prefix sums.
pub fn sample_from(probs: &[f64], u: f64) -> u32 {
    let target = u * pinned_sum(probs);
    let mut base = 0.0;
    let mut last_nonzero = 0usize;
    for (c, chunk) in probs.chunks(SUM_CHUNK).enumerate() {
        let mut chunk_sum = 0.0;
        for &p in chunk {
            chunk_sum += p;
        }
        if let Some(j) = chunk.iter().rposition(|&p| p > 0.0) {
            last_nonzero = c * SUM_CHUNK + j;
        }
        if base + chunk_sum > target {
            let mut cum = 0.0;
            for (j, &p) in chunk.iter().enumerate() {
                cum += p;
                if p > 0.0 && base + cum > target {
                    return (c * SUM_CHUNK + j) as u32;
                }
            }
        }
        base += chunk_sum;
    }
    last_nonzero as u32
}

/// Samples the token at sequence `position` from `logits`.
pub fn sample(logits: Logits<'_>, params: &SamplingParams, position: u64) -> u32 {
    if params.is_greedy() {
        return argmax(logits);
    }
    let p = processed_probs(logits, params);
    sample_from(&p, uniform(params.seed(), position, Stream::Sample))
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
            uniform(params.seed(), pos, Stream::Accept) * q[d] < p[d]
        };
        if accepted {
            out.push(drafts[i]);
            continue;
        }
        let token = if params.is_greedy() {
            argmax_f64(p)
        } else {
            let residual: Vec<f64> = p.iter().zip(q).map(|(a, b)| (a - b).max(0.0)).collect();
            let row = if pinned_sum(&residual) > 0.0 {
                &residual
            } else {
                p
            };
            sample_from(row, uniform(params.seed(), pos, Stream::Residual))
        };
        out.push(token);
        return out;
    }
    let pos = first_position + k as u64;
    let bonus = if params.is_greedy() {
        argmax_f64(&target[k])
    } else {
        sample_from(&target[k], uniform(params.seed(), pos, Stream::Sample))
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
        let params = SamplingParams::random(1.0, 9).unwrap();
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

    /// Values the sampler cannot define are unrepresentable: a NaN temperature used to
    /// fall through softmax to token 0.
    #[test]
    fn invalid_parameters_are_refused() {
        for t in [f32::NAN, f32::INFINITY, -0.5] {
            assert!(SamplingParams::random(t, 0).is_err(), "temperature {t}");
        }
        for top_p in [0.0, -0.1, 1.5, f32::NAN] {
            assert!(
                SamplingParams::new(1.0, 0, top_p, 0.0, 0).is_err(),
                "top_p {top_p}"
            );
        }
        for min_p in [-0.1, 1.5, f32::NAN] {
            assert!(
                SamplingParams::new(1.0, 0, 1.0, min_p, 0).is_err(),
                "min_p {min_p}"
            );
        }
        assert!(
            SamplingParams::new(0.0, 0, 1.0, 0.0, 0)
                .unwrap()
                .is_greedy()
        );
        assert!(SamplingParams::new(0.7, 8, 0.9, 1.0, 3).is_ok());
    }

    /// `det_exp` stays within a few ulp of the platform `exp` across the range softmax
    /// uses (and beyond, through the subnormals), and its bits are pinned: the goldens
    /// below were computed once and must hold on every platform.
    #[test]
    fn det_exp_is_accurate_and_pinned() {
        let ulps = |a: f64, b: f64| (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs();
        let mut s = 1u64;
        let mut worst = 0;
        for _ in 0..200_000 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = -((s >> 11) as f64 / (1u64 << 53) as f64) * 745.0 + 5.0;
            let (a, b) = (det_exp(x), x.exp());
            if b > f64::MIN_POSITIVE {
                worst = worst.max(ulps(a, b));
            } else {
                assert!((a - b).abs() <= f64::from_bits(2), "{x}: {a:e} vs {b:e}");
            }
        }
        assert!(worst <= 2, "worst {worst} ulp");
        assert_eq!(det_exp(0.0), 1.0);
        assert_eq!(det_exp(f64::NEG_INFINITY), 0.0);
        assert_eq!(det_exp(-746.0), 0.0);
        assert_eq!(det_exp(710.0), f64::INFINITY);
        assert!(det_exp(f64::NAN).is_nan());
        for (x, bits) in DET_EXP_GOLDEN {
            assert_eq!(det_exp(x).to_bits(), bits, "det_exp({x})");
        }
    }

    /// Computed independently (Python's IEEE doubles, same operations).
    const DET_EXP_GOLDEN: [(f64, u64); 6] = [
        (-1.0, 0x3fd78b56362cef38),
        (-0.5, 0x3fe368b2fc6f960a),
        (-13.37, 0x3eba31ad57ba12f4),
        (-700.25, 0x00caf5fe9a485c8e),
        (-740.0, 0x0000000000000055),
        (3.0, 0x403415e5bf6fb106),
    ];

    #[test]
    fn pinned_sum_chunks() {
        let x: Vec<f64> = (0..3000).map(|i| 1.0 / (i as f64 + 1.0)).collect();
        let mut want = 0.0;
        for c in x.chunks(SUM_CHUNK) {
            want += c.iter().fold(0.0, |a, &b| a + b);
        }
        assert_eq!(pinned_sum(&x).to_bits(), want.to_bits());
        assert_eq!(pinned_sum_where(&x, |_| true).to_bits(), want.to_bits());
        // A draw lands on the id whose pinned prefix first exceeds the target.
        let mut p = vec![0.0; 2 * SUM_CHUNK + 5];
        p[3] = 0.25;
        p[SUM_CHUNK + 7] = 0.5;
        p[2 * SUM_CHUNK + 1] = 0.25;
        assert_eq!(sample_from(&p, 0.0), 3);
        assert_eq!(sample_from(&p, 0.25), SUM_CHUNK as u32 + 7);
        assert_eq!(sample_from(&p, 0.75), 2 * SUM_CHUNK as u32 + 1);
        assert_eq!(sample_from(&p, 0.999), 2 * SUM_CHUNK as u32 + 1);
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
        let mut p = SamplingParams::random(1.0, 0).unwrap();
        p.top_k = 2;
        let probs = processed_probs(full(&logits), &p);
        assert!(probs[3] > 0.0 && probs[4] > 0.0);
        assert_eq!(probs[..3], [0.0, 0.0, 0.0]);
        assert!((probs[3] - 0.5).abs() < 1e-12);

        let mut p = SamplingParams::random(1.0, 0).unwrap();
        p.top_p = 0.5;
        let probs = processed_probs(full(&logits), &p);
        // Tokens 3 and 4 tie; 3 ranks first. Its mass alone is < 0.5, so 4 joins.
        assert!(probs[3] > 0.0 && probs[4] > 0.0 && probs[2] == 0.0);

        let mut p = SamplingParams::random(1.0, 0).unwrap();
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
