//! The dense oracle [`super::DsCpuExecutor`] is diffed against: the model crate's full
//! forward (every layer at every position, one sequence, no paging, no split) and its
//! drafter, laid out in the executor's rows.
//!
//! **Drafter row layout.** A row whose last host position is `p` drafts from the
//! drafter's context KV at the positions before `p` (each a function of the target's
//! state at that position) and the token at `p`: draft slot `i` stands at `p + i`,
//! takes the token at `p` (slot 0) or the noise token, and predicts the token at
//! `p + 1 + i`; its logits add a Markov term from the token before it in the chain (the
//! token at `p` for slot 0, the draft for `p + i` otherwise). In the reference's
//! indexing that is a draft pass at main position `p - 1` given the token chosen for
//! `p`. The reference's entry point never drafts from main position 0; the pass is the
//! same function there, and a row at `p = 1` runs it.

use eidola_engine::sampling::{self, Logits, SamplingParams};
use eidola_engine_model::Result;
use eidola_engine_model::deepseek::config::RopeKind;
use eidola_engine_model::deepseek::drafter::{DraftCore, ring_order};
use eidola_engine_model::deepseek::forward::{DsForwardOptions, DsModel, window_kv};
use eidola_engine_model::tensor::Matrix;

/// Dense logits and drafter context for one token sequence.
#[derive(Debug)]
pub struct DsDenseRun {
    pub tokens: Vec<u32>,
    /// Target logits at every position.
    pub logits: Matrix,
    /// Per draft block: its context KV at every position (`[positions · head_dim]`).
    pub drafter_kv: Vec<Vec<f32>>,
}

impl DsDenseRun {
    /// Target logits at `pos`.
    pub fn target(&self, pos: u32) -> &[f32] {
        self.logits.row(pos as usize)
    }
}

/// The dense oracle over a model.
#[derive(Clone, Copy)]
pub struct DsDenseOracle<'a> {
    pub model: &'a DsModel,
    /// Token ids sampling may return; the head's rows past it are padding.
    pub sampleable_vocab_size: u32,
}

impl DsDenseOracle<'_> {
    /// The full forward over `tokens`, and the drafter's context KV at every position.
    pub fn run(&self, tokens: &[u32]) -> Result<DsDenseRun> {
        let m = self.model;
        let c = m.config();
        let out = m.forward(&mut m.new_state(), tokens, &DsForwardOptions::default())?;
        let mut drafter_kv = Vec::new();
        if let (Some(w), Some(hidden)) = (&m.weights.drafter, &out.main_hidden) {
            drafter_kv = vec![Vec::new(); w.blocks.len()];
            let mut state = m.new_drafter_state();
            for pos in 0..tokens.len() {
                let main_x = m.drafter_observe(&mut state, hidden.row(pos))?;
                let cis = m.rope(RopeKind::Plain).cis(pos);
                for (block, kv) in w.blocks.iter().zip(&mut drafter_kv) {
                    kv.extend(window_kv(c, &block.attention, &main_x, &cis));
                }
            }
        }
        Ok(DsDenseRun {
            tokens: tokens.to_vec(),
            logits: out.logits,
            drafter_kv,
        })
    }

    /// The draft pass of a row whose last host position is `p >= 1`.
    pub fn draft(&self, run: &DsDenseRun, p: u32) -> Result<DraftCore> {
        let c = self.model.config();
        let (hd, main) = (c.attention.head_dim, p as usize - 1);
        let context: Vec<Vec<&[f32]>> = run
            .drafter_kv
            .iter()
            .map(|kv| {
                ring_order(main, c.attention.window)
                    .into_iter()
                    .map(|pos| &kv[pos * hd..(pos + 1) * hd])
                    .collect()
            })
            .collect();
        self.model
            .drafter_slots(main, &context, run.tokens[p as usize])
    }
}

/// Non-speculative generation from dense logits: what every engine run must reproduce
/// exactly when speculation is off or greedy.
pub fn ds_dense_generate(
    model: &DsModel,
    sampleable_vocab_size: u32,
    prompt: &[u32],
    params: &SamplingParams,
    max_tokens: u32,
    stop: &[u32],
) -> Result<Vec<u32>> {
    let opts = DsForwardOptions::default();
    let mut state = model.new_state();
    let mut last = model.forward(&mut state, prompt, &opts)?.logits;
    let mut q = prompt.len() as u64 - 1;
    let mut out = Vec::new();
    while (out.len() as u32) < max_tokens {
        let logits = Logits::new(last.row(last.rows - 1), sampleable_vocab_size);
        let t = sampling::sample(logits, params, q + 1);
        out.push(t);
        if stop.contains(&t) {
            break;
        }
        last = model.forward(&mut state, &[t], &opts)?.logits;
        q += 1;
    }
    Ok(out)
}

/// Whether `output` is exactly what non-speculative generation from the dense logits
/// produces after the prompt, given the dense run over `prompt ++ output`. Returns the
/// first position that disagrees.
pub fn ds_check_generation(
    dense: &DsDenseRun,
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
