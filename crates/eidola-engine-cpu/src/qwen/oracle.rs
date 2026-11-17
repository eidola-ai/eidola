//! The dense oracle the Qwen3.8 executor is diffed against: the model crate's reference
//! forward over a whole token sequence from the zero state, the MTP layer over every
//! position in the executor's row layout, and the draft chain.
//!
//! **Drafter row layout.** The drafter's row at position `s` (for `s >= 1`) consumes the
//! token at `s` and the target's normed final state at `s - 1`, attends with RoPE
//! position `s - 1`, and predicts the token at `s + 1`: the model crate's MTP row `s - 1`
//! (`mtp_forward` row `i` combines the token at `i + 1` with the state at `i`). Its KV at
//! `s` therefore depends only on tokens `0 ..= s`. The **draft chain** after the last host
//! position `p` is the row at `p` (draft 1), then the same layer recursing on its own
//! normed output: step `i` consumes draft `i` at `p + i` and step `i - 1`'s normed
//! output, at RoPE position `p + i - 1`, and predicts draft `i + 1`. Only the row at `p`
//! is canonical; once drafts are accepted the executor rewrites their rows from the
//! target's states.

use eidola_engine::sampling::{self, Logits, SamplingParams};
use eidola_engine_model::qwen::{
    DeltaNetState, KvCache, LayerState, LogitRows, QwenForwardOptions, QwenModel,
};
use eidola_engine_model::{Matrix, Result};

/// Dense logits for one token sequence.
#[derive(Debug)]
pub struct QwenDenseRun {
    /// Target logits at every position.
    pub logits: Matrix,
    /// The target's normed final state at every position.
    pub normed: Matrix,
    /// The drafter's logits at positions `1 ..` (row `i` is position `i + 1`); empty
    /// without a drafter.
    pub drafter_logits: Matrix,
}

impl QwenDenseRun {
    /// Target logits at `pos`.
    pub fn target(&self, pos: u32) -> &[f32] {
        self.logits.row(pos as usize)
    }

    /// The drafter's logits at position `s`, if that row exists.
    pub fn drafter(&self, s: u32) -> Option<&[f32]> {
        let i = (s as usize).checked_sub(1)?;
        (i < self.drafter_logits.rows).then(|| self.drafter_logits.row(i))
    }
}

/// The dense oracle over a model.
#[derive(Clone, Copy)]
pub struct QwenDenseOracle<'a> {
    /// The model.
    pub model: &'a QwenModel,
    /// Whether the drafter's rows are computed.
    pub drafting: bool,
    /// Token ids sampling may return; the head's rows past it are padding.
    pub sampleable_vocab_size: u32,
}

impl QwenDenseOracle<'_> {
    /// Target logits and normed states at every position of `tokens`, and every drafter
    /// row.
    pub fn run(&self, tokens: &[u32]) -> Result<QwenDenseRun> {
        let m = self.model;
        let mut state = m.new_state();
        let out = m.forward(
            &mut state,
            tokens,
            &QwenForwardOptions {
                logits: LogitRows::All,
                keep_streams: false,
            },
        )?;
        let drafter_logits = if self.drafting && tokens.len() >= 2 {
            let rows: Vec<usize> = (0..tokens.len() - 1).collect();
            let mut cache = KvCache::default();
            m.mtp_forward(
                0,
                &mut cache,
                0,
                &tokens[1..],
                &out.normed.select_rows(&rows),
                &LogitRows::All,
            )?
            .logits
        } else {
            Matrix::zeros(0, m.config().vocab_size)
        };
        Ok(QwenDenseRun {
            logits: out.logits,
            normed: out.normed,
            drafter_logits,
        })
    }

    /// The draft chain's logits after the last host position `p` of `tokens` (which
    /// holds at least `p + 1` tokens): step 0 is the drafter's row at `p`, and step `i`
    /// consumes `drafts[i - 1]` and step `i - 1`'s normed output. One row per element of
    /// `drafts`, plus one: the logits each draft was drawn from, and the next.
    pub fn chain(
        &self,
        run: &QwenDenseRun,
        tokens: &[u32],
        p: u32,
        drafts: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        let m = self.model;
        let p = p as usize;
        assert!(p >= 1, "no draft chain at position 0");
        let rows: Vec<usize> = (0..p).collect();
        let mut cache = KvCache::default();
        let first = m.mtp_forward(
            0,
            &mut cache,
            0,
            &tokens[1..=p],
            &run.normed.select_rows(&rows),
            &LogitRows::Last,
        )?;
        let mut out = vec![first.logits.row(0).to_vec()];
        let mut prev = first.normed.select_rows(&[p - 1]);
        for (i, &d) in drafts.iter().enumerate() {
            let step = m.mtp_forward(0, &mut cache, p + i, &[d], &prev, &LogitRows::Last)?;
            out.push(step.logits.row(0).to_vec());
            prev = step.normed;
        }
        Ok(out)
    }

    /// The DeltaNet layers' states at `len`: what a state buffer stored at that boundary
    /// holds, layer by layer.
    pub fn states_at(&self, tokens: &[u32], len: usize) -> Result<Vec<DeltaNetState>> {
        let mut state = self.model.new_state();
        self.model
            .forward(&mut state, &tokens[..len], &QwenForwardOptions::default())?;
        Ok(state
            .layers
            .into_iter()
            .filter_map(|l| match l {
                LayerState::DeltaNet(s) => Some(s),
                LayerState::Attention(_) => None,
            })
            .collect())
    }
}

/// Whether `output` is exactly what non-speculative generation from the dense logits
/// produces after `prompt` (each token sampled at its position from the logits of its
/// prefix), given the dense logits over `prompt ++ output`. Returns the first position
/// that disagrees.
pub fn qwen_check_generation(
    dense: &QwenDenseRun,
    sampleable_vocab_size: u32,
    prompt_len: usize,
    output: &[u32],
    params: &SamplingParams,
) -> std::result::Result<(), usize> {
    for (i, &t) in output.iter().enumerate() {
        let q = (prompt_len + i - 1) as u32;
        let logits = Logits::new(dense.target(q), sampleable_vocab_size);
        if sampling::sample(logits, params, q as u64 + 1) != t {
            return Err(q as usize + 1);
        }
    }
    Ok(())
}
