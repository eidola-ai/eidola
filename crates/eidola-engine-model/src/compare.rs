//! Logit and hidden-state comparison metrics for golden checks.

/// `log_softmax` in f64.
pub fn log_softmax(logits: &[f32]) -> Vec<f64> {
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

/// `KL(reference ‖ candidate)` in nats.
pub fn kl_divergence(reference: &[f32], candidate: &[f32]) -> f64 {
    let p = log_softmax(reference);
    let q = log_softmax(candidate);
    p.iter()
        .zip(&q)
        .map(|(&lp, &lq)| lp.exp() * (lp - lq))
        .sum::<f64>()
        .max(0.0)
}

pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}

/// Largest absolute difference and the reference's largest magnitude.
pub fn max_abs_diff(reference: &[f32], candidate: &[f32]) -> (f32, f32) {
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

/// Compare row-major `[rows, vocab]` logits.
pub fn compare_logits(reference: &[f32], candidate: &[f32], vocab: usize) -> LogitAgreement {
    assert_eq!(reference.len(), candidate.len());
    let mut a = LogitAgreement::default();
    for (r, c) in reference
        .chunks_exact(vocab)
        .zip(candidate.chunks_exact(vocab))
    {
        a.rows += 1;
        if argmax(r) == argmax(c) {
            a.top1_matches += 1;
        }
        let kl = kl_divergence(r, c);
        a.kl_mean += kl;
        a.kl_max = a.kl_max.max(kl);
        a.max_abs_diff = a.max_abs_diff.max(max_abs_diff(r, c).0);
    }
    a.kl_mean /= a.rows.max(1) as f64;
    a
}
