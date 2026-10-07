//! The dense oracle the executor is diffed against: the model crate's reference forward
//! over a whole token sequence, plus the MTP draft chain laid out the way the executor
//! runs it.
//!
//! **Drafter row layout.** MTP depth `d`'s row at *slot* `s` consumes the token at `s`
//! and chain level `d` at `s - 1` (level 0 is the main model's hidden state, level `d` is
//! depth `d - 1`'s output), attends with RoPE position `s - 1`, and predicts the token at
//! `s + 1`. Rows exist for `s >= d + 1`. Depth `d`'s KV at slot `s` therefore depends only
//! on tokens `0 ..= s`, which is what lets drafter blocks be prefix-cached like target
//! blocks. The draft for position `p + 1 + i` is depth `i`'s prediction at slot `p + i`.

use eidola_engine::sampling::{self, SamplingParams};
use eidola_engine_model::{ForwardOptions, LogitsAt, Matrix, ReferenceModel, Result};

use crate::executor::MtpHidden;

/// Dense logits and drafter outputs for one token sequence.
#[derive(Debug)]
pub struct DenseRun {
    /// Target logits at every position.
    pub logits: Matrix,
    /// Per depth: logits for slots `d + 1 ..` (row `i` is slot `d + 1 + i`).
    pub drafter_logits: Vec<Matrix>,
}

impl DenseRun {
    /// Target logits at `pos`.
    pub fn target(&self, pos: u32) -> &[f32] {
        self.logits.row(pos as usize)
    }

    /// Depth `depth`'s logits at `slot`, if that row exists.
    pub fn drafter(&self, depth: usize, slot: u32) -> Option<&[f32]> {
        let m = self.drafter_logits.get(depth)?;
        let i = (slot as usize).checked_sub(depth + 1)?;
        (i < m.rows).then(|| m.row(i))
    }
}

/// The dense oracle over a model and a drafter configuration.
#[derive(Clone, Copy)]
pub struct DenseOracle<'a> {
    /// The model.
    pub model: &'a ReferenceModel,
    /// MTP layer per depth.
    pub mtp_depths: &'a [usize],
    /// Hidden-state chaining.
    pub mtp_hidden: MtpHidden,
}

impl DenseOracle<'_> {
    /// Target logits at every position of `tokens` and every drafter row.
    pub fn run(&self, tokens: &[u32]) -> Result<DenseRun> {
        let fwd = self.model.forward(
            tokens,
            &ForwardOptions {
                logits: LogitsAt::All,
                capture_layers: false,
            },
        )?;
        let pick = |hidden: &Matrix, normed: &Matrix| match self.mtp_hidden {
            MtpHidden::Normed => normed.clone(),
            MtpHidden::PreNorm => hidden.clone(),
        };
        // Level `d` for slots `d ..` (row `i` is slot `d + i`).
        let mut level = pick(&fwd.hidden, &fwd.hidden_normed);
        let mut drafter_logits = Vec::new();
        let len = tokens.len();
        for (d, &layer) in self.mtp_depths.iter().enumerate() {
            if len < d + 2 {
                break;
            }
            // Slots `d + 1 .. len`, continuing from level `d` at slots `d .. len - 1`.
            let slots: Vec<usize> = (d + 1..len).collect();
            let prev = level.select_rows(&(0..slots.len()).collect::<Vec<_>>());
            let positions: Vec<usize> = slots.iter().map(|s| s - 1).collect();
            let toks: Vec<u32> = slots.iter().map(|&s| tokens[s]).collect();
            let out = self
                .model
                .mtp_forward(layer, &toks, &prev, &positions, &LogitsAt::All)?;
            drafter_logits.push(out.logits);
            level = pick(&out.hidden, &out.hidden_normed);
        }
        Ok(DenseRun {
            logits: fwd.logits,
            drafter_logits,
        })
    }
}

/// Non-speculative generation from dense logits: what every engine run must reproduce
/// exactly when speculation is off or greedy. One dense forward per generated token, so
/// keep it to short checks; [`check_generation`] verifies a finished output with one.
pub fn dense_generate(
    model: &ReferenceModel,
    prompt: &[u32],
    params: &SamplingParams,
    max_tokens: u32,
    stop: &[u32],
) -> Result<Vec<u32>> {
    let mut tokens = prompt.to_vec();
    let mut out = Vec::new();
    while (out.len() as u32) < max_tokens {
        let fwd = model.forward(
            &tokens,
            &ForwardOptions {
                logits: LogitsAt::Last,
                capture_layers: false,
            },
        )?;
        let q = tokens.len() as u64 - 1;
        let t = sampling::sample(fwd.logits.row(0), params, q + 1);
        tokens.push(t);
        out.push(t);
        if stop.contains(&t) {
            break;
        }
    }
    Ok(out)
}

/// Whether `output` is exactly what non-speculative generation from the dense logits
/// produces after `prompt` (each token sampled at its position from the logits of its
/// prefix), given the dense logits over `prompt ++ output`. Returns the first position
/// that disagrees.
pub fn check_generation(
    dense: &DenseRun,
    prompt_len: usize,
    output: &[u32],
    params: &SamplingParams,
) -> std::result::Result<(), usize> {
    for (i, &t) in output.iter().enumerate() {
        let q = (prompt_len + i - 1) as u32;
        if sampling::sample(dense.target(q), params, q as u64 + 1) != t {
            return Err(q as usize + 1);
        }
    }
    Ok(())
}
