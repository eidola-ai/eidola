//! Drafting acceptance per depth, tallied from what each row was asked to
//! draft. Shared by `examples/eval.rs` and `tests/drafting.rs`, each of which
//! includes this file by path, so it is the one definition both report.
//!
//! Depth `d`'s rate is conditional: draft `d` accepted among the rows that
//! drafted at least `d` tokens in a step and had their first `d - 1`
//! accepted. A row that drafted nothing (a prefill end, a step that ran short
//! of KV) or fewer than `d` tokens (trimmed near the model length) is outside
//! depth `d`'s denominator, not a rejection there.

use eidola_engine::executor::{Executor, ExecutorError, SeqEntry, StepInput, StepOutput};
use eidola_engine::spec::ModelSpec;

/// Per-row drafting outcomes of every step, from the step's entries and what
/// the executor produced (before the engine cuts a row at EOS, so the totals
/// are the engine's `Stats::drafted` and `accepted`).
#[derive(Debug, Default, PartialEq)]
pub struct DraftTally {
    /// Sampled decode rows (one host token) by draft width, `0 ..= k`.
    pub widths: Vec<u64>,
    /// Sampled rows with more than one host token (prefill ends): never drafted.
    pub prefill: u64,
    /// Per depth `d` (index `d - 1`): rows that drafted at least `d` tokens
    /// and had their first `d - 1` accepted.
    pub reached: Vec<u64>,
    /// Per depth `d`: those of `reached` whose draft `d` was accepted.
    pub accepted: Vec<u64>,
    /// Drafts proposed over every drafting row.
    pub drafted_total: u64,
    /// Drafts accepted over every drafting row.
    pub accepted_total: u64,
}

impl DraftTally {
    pub fn new(depths: usize) -> Self {
        Self {
            widths: vec![0; depths + 1],
            reached: vec![0; depths],
            accepted: vec![0; depths],
            ..Self::default()
        }
    }

    /// One row of a step: its entry, and the tokens it produced.
    pub fn record(&mut self, e: &SeqEntry, produced: usize) {
        if !e.sample {
            return;
        }
        if e.num_tokens > 1 {
            self.prefill += 1;
            return;
        }
        let width = e.num_drafts as usize;
        self.widths[width] += 1;
        let accepted = produced.saturating_sub(1);
        self.drafted_total += width as u64;
        self.accepted_total += accepted as u64;
        for d in 1..=width {
            if accepted < d - 1 {
                break;
            }
            self.reached[d - 1] += 1;
            if accepted >= d {
                self.accepted[d - 1] += 1;
            }
        }
    }

    /// Conditional acceptance of each depth (0 where no row reached it).
    pub fn rates(&self) -> Vec<f64> {
        self.reached
            .iter()
            .zip(&self.accepted)
            .map(|(&r, &a)| if r == 0 { 0.0 } else { a as f64 / r as f64 })
            .collect()
    }
}

/// An executor that tallies every step's rows ([`DraftTally`]) over its
/// spec's draft depth.
pub struct Tallying<E> {
    inner: E,
    tally: DraftTally,
}

impl<E: Executor> Tallying<E> {
    pub fn new(inner: E) -> Self {
        let depths = inner.spec().max_draft_tokens as usize;
        Self {
            inner,
            tally: DraftTally::new(depths),
        }
    }

    #[allow(
        dead_code,
        reason = "the drafting tests reach their executor through it; the eval example does not"
    )]
    pub fn inner(&self) -> &E {
        &self.inner
    }

    pub fn tally(&self) -> &DraftTally {
        &self.tally
    }
}

impl<E: Executor> Executor for Tallying<E> {
    fn spec(&self) -> &ModelSpec {
        self.inner.spec()
    }

    fn execute(&mut self, step: &StepInput) -> Result<StepOutput, ExecutorError> {
        let out = self.inner.execute(step)?;
        for (i, e) in step.seqs.iter().enumerate() {
            self.tally.record(e, out.num_tokens[i] as usize);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidola_engine::sampling::SamplingParams;

    fn row(num_tokens: u32, num_drafts: u32, sample: bool) -> SeqEntry {
        SeqEntry {
            slot: 0,
            token_start: 0,
            num_tokens,
            context_len: 16,
            num_drafts,
            sample,
            sampling: SamplingParams::greedy(),
        }
    }

    /// Depth `d`'s denominator is the rows that drafted at least `d` tokens
    /// and had the first `d - 1` accepted: rows that drafted nothing, or fewer
    /// than `d`, are not rejections at `d`.
    #[test]
    fn acceptance_is_conditional_on_the_drafted_width() {
        let mut t = DraftTally::new(3);
        // Full width: all accepted, two accepted, none accepted.
        t.record(&row(1, 3, true), 4);
        t.record(&row(1, 3, true), 3);
        t.record(&row(1, 3, true), 1);
        // Narrowed to width 1 (accepted) and width 0, a prefill end, and a
        // prefill chunk that does not sample.
        t.record(&row(1, 1, true), 2);
        t.record(&row(1, 0, true), 1);
        t.record(&row(64, 0, true), 1);
        t.record(&row(64, 0, false), 0);
        assert_eq!(t.widths, vec![1, 1, 0, 3]);
        assert_eq!(t.prefill, 1);
        assert_eq!(t.reached, vec![4, 2, 2]);
        assert_eq!(t.accepted, vec![3, 2, 1]);
        assert_eq!((t.accepted_total, t.drafted_total), (6, 10));
        assert_eq!(t.rates(), vec![0.75, 1.0, 0.5]);
        // No row reached a depth: 0, not NaN.
        assert_eq!(DraftTally::new(2).rates(), vec![0.0, 0.0]);
    }
}
