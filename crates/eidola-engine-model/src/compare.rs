//! Logit and hidden-state comparison metrics for golden checks.
//!
//! **Fail-closed.** These metrics certify executors and kernels against the
//! reference, so a value they cannot measure is a failure, never a number that
//! looks good. Every function checks that every value on both sides is finite
//! before computing anything and returns [`NonFinite`] otherwise. With finite
//! inputs every metric is finite, so no `max`, `min` or clamp below can swallow a
//! NaN: an f32/f64 `max` drops a NaN operand, and a NaN KL clamped by `.max(0.0)`
//! would read as perfect agreement.

/// Which side of a comparison held a value that is not finite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Reference,
    Candidate,
}

/// A NaN or infinity in a compared row: the comparison is refused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NonFinite {
    pub side: Side,
    /// Row index (0 for single-row functions).
    pub row: usize,
    /// Index within the row.
    pub index: usize,
    pub value: f32,
}

impl std::fmt::Display for NonFinite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} row {} index {} is {} (comparisons require finite values)",
            self.side, self.row, self.index, self.value
        )
    }
}

impl std::error::Error for NonFinite {}

fn check_finite(x: &[f32], side: Side, row: usize) -> Result<(), NonFinite> {
    match x.iter().position(|v| !v.is_finite()) {
        None => Ok(()),
        Some(index) => Err(NonFinite {
            side,
            row,
            index,
            value: x[index],
        }),
    }
}

fn check_pair(reference: &[f32], candidate: &[f32], row: usize) -> Result<(), NonFinite> {
    assert_eq!(
        reference.len(),
        candidate.len(),
        "compared rows differ in length"
    );
    check_finite(reference, Side::Reference, row)?;
    check_finite(candidate, Side::Candidate, row)
}

/// `log_softmax` in f64 of a finite row (the caller has checked it).
fn log_softmax_finite(logits: &[f32]) -> Vec<f64> {
    let m = logits
        .iter()
        .fold(f64::NEG_INFINITY, |m, &x| m.max(x as f64));
    let lse = logits
        .iter()
        .map(|&x| (x as f64 - m).exp())
        .sum::<f64>()
        .ln()
        + m;
    logits.iter().map(|&x| x as f64 - lse).collect()
}

/// `log_softmax` in f64. Refuses a non-finite row.
pub fn log_softmax(logits: &[f32]) -> Result<Vec<f64>, NonFinite> {
    check_finite(logits, Side::Reference, 0)?;
    Ok(log_softmax_finite(logits))
}

/// `KL(reference ‖ candidate)` in nats. Refuses non-finite rows. Rounding can
/// make the sum of a near-identical pair slightly negative; only that finite
/// value is clamped to 0.
pub fn kl_divergence(reference: &[f32], candidate: &[f32]) -> Result<f64, NonFinite> {
    check_pair(reference, candidate, 0)?;
    Ok(kl_finite(reference, candidate))
}

fn kl_finite(reference: &[f32], candidate: &[f32]) -> f64 {
    let p = log_softmax_finite(reference);
    let q = log_softmax_finite(candidate);
    let kl: f64 = p
        .iter()
        .zip(&q)
        .map(|(&lp, &lq)| lp.exp() * (lp - lq))
        .sum();
    assert!(kl.is_finite(), "KL of finite rows is finite");
    kl.max(0.0)
}

/// Lowest-index argmax. Refuses a non-finite row (a NaN never compares greater,
/// so it could hide anywhere but the winning position).
pub fn argmax(x: &[f32]) -> Result<usize, NonFinite> {
    check_finite(x, Side::Reference, 0)?;
    Ok(argmax_finite(x))
}

fn argmax_finite(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}

/// Largest absolute difference and the reference's largest magnitude. Refuses
/// non-finite values on either side.
pub fn max_abs_diff(reference: &[f32], candidate: &[f32]) -> Result<(f32, f32), NonFinite> {
    check_pair(reference, candidate, 0)?;
    Ok(max_abs_diff_finite(reference, candidate))
}

fn max_abs_diff_finite(reference: &[f32], candidate: &[f32]) -> (f32, f32) {
    let mut d = 0.0f32;
    let mut m = 0.0f32;
    for (&a, &b) in reference.iter().zip(candidate) {
        d = d.max((a - b).abs());
        m = m.max(a.abs());
    }
    (d, m)
}

/// Summary over many rows of logits.
#[derive(Debug, Clone, Default)]
pub struct LogitAgreement {
    pub rows: usize,
    pub top1_matches: usize,
    pub kl_mean: f64,
    pub kl_max: f64,
    pub max_abs_diff: f32,
}

impl LogitAgreement {
    pub fn top1_rate(&self) -> f64 {
        self.top1_matches as f64 / self.rows.max(1) as f64
    }
}

impl std::fmt::Display for LogitAgreement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rows {} top-1 {}/{} ({:.4}%) KL mean {:.3e} max {:.3e} max|Δlogit| {:.3e}",
            self.rows,
            self.top1_matches,
            self.rows,
            100.0 * self.top1_rate(),
            self.kl_mean,
            self.kl_max,
            self.max_abs_diff
        )
    }
}

/// Compare row-major `[rows, vocab]` logits. Any non-finite value in any row of
/// either side refuses the whole comparison, naming the first one.
pub fn compare_logits(
    reference: &[f32],
    candidate: &[f32],
    vocab: usize,
) -> Result<LogitAgreement, NonFinite> {
    assert_eq!(reference.len(), candidate.len());
    assert!(vocab > 0 && reference.len() % vocab == 0, "whole rows");
    let mut a = LogitAgreement::default();
    for (row, (r, c)) in reference
        .chunks_exact(vocab)
        .zip(candidate.chunks_exact(vocab))
        .enumerate()
    {
        check_pair(r, c, row)?;
        a.rows += 1;
        if argmax_finite(r) == argmax_finite(c) {
            a.top1_matches += 1;
        }
        let kl = kl_finite(r, c);
        a.kl_mean += kl;
        a.kl_max = a.kl_max.max(kl);
        a.max_abs_diff = a.max_abs_diff.max(max_abs_diff_finite(r, c).0);
    }
    a.kl_mean /= a.rows.max(1) as f64;
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A NaN or infinity anywhere, including off the winning logit where argmax
    /// still agrees and an unchecked KL or max |Δ| would read as perfect, fails
    /// every metric.
    #[test]
    fn non_finite_values_fail_every_metric() {
        let reference = [0.0f32, 5.0, 1.0, 2.0];
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut candidate = reference;
            candidate[2] = bad;
            let err = compare_logits(&reference, &candidate, 4).unwrap_err();
            assert_eq!((err.side, err.row, err.index), (Side::Candidate, 0, 2));
            assert!(kl_divergence(&reference, &candidate).is_err());
            assert!(max_abs_diff(&reference, &candidate).is_err());
            assert!(argmax(&candidate).is_err());
            // On the reference side too, and in a later row.
            let two = [reference, reference].concat();
            let mut bad_two = two.clone();
            bad_two[4 + 3] = bad;
            let err = compare_logits(&bad_two, &two, 4).unwrap_err();
            assert_eq!((err.side, err.row, err.index), (Side::Reference, 1, 3));
        }
        let ok = compare_logits(&reference, &reference, 4).unwrap();
        assert_eq!((ok.rows, ok.top1_matches, ok.kl_max), (1, 1, 0.0));
    }
}
