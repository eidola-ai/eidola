//! The DSpark draft head (`inference/model.py` `Transformer.forward_spec`,
//! `DSparkBlock`, `DSparkAttention`, `DSparkMarkovHead`,
//! `DSparkConfidenceHead`).
//!
//! Given the main model's hidden state at position `p` and the token chosen
//! for `p + 1`, the drafter computes `slots` draft positions in one pass:
//! slot `i` stands at position `p + 1 + i` and predicts the token at
//! `p + 2 + i`. Slot 0's input is the given token, the others a noise token.
//! Each slot's logits get a Markov term from the token drafted before it, so
//! the draft is a chain. The released code defines no acceptance procedure;
//! none is here.

use rayon::prelude::*;

use super::config::RopeKind;
use super::forward::{DsModel, attention_output, moe_forward, query, sub_layer_input, window_kv};
use super::linear::{lin8, linear_f32};
use super::mhc;
use super::weights::rms;
use crate::error::{Error, Result};
use crate::tensor::{Matrix, dot};

/// The drafter's per-sequence state: each draft block's window KV of every
/// main position so far (an executor keeps the last `window`).
#[derive(Debug, Clone, Default)]
pub struct DrafterState {
    /// Per block, `[positions, head_dim]`.
    window: Vec<Vec<f32>>,
    positions: usize,
}

impl DrafterState {
    pub fn positions(&self) -> usize {
        self.positions
    }
}

/// One drafting pass, before the Markov chain is chosen.
#[derive(Debug, Clone)]
pub struct DraftCore {
    /// `[slots, vocab]`: the backbone head over the drafter's normed states.
    pub base_logits: Matrix,
    /// `[slots, dim]`: the collapsed states before the drafter's final norm,
    /// which the confidence head reads.
    pub hidden: Matrix,
}

/// A greedy draft chain.
#[derive(Debug, Clone)]
pub struct Draft {
    /// The drafted tokens: `tokens[i]` is slot `i`'s prediction.
    pub tokens: Vec<u32>,
    /// `[slots, vocab]`: each slot's logits including its Markov term.
    pub logits: Matrix,
    /// One raw confidence logit per slot.
    pub confidence: Vec<f32>,
}

/// The main positions a draft pass at position `p` attends to, in the order
/// the reference lists them: by slot of a ring of `window` slots, where slot
/// `r` holds the newest position congruent to `r`. Once the ring has wrapped
/// this is not position order, and the order decides which 64-wide attention
/// block an entry falls in.
pub fn ring_order(p: usize, window: usize) -> Vec<usize> {
    (0..window.min(p + 1))
        .map(|r| p - (p - r) % window)
        .collect()
}

impl DsModel {
    pub fn new_drafter_state(&self) -> DrafterState {
        let blocks = self.config().drafter.as_ref().map_or(0, |d| d.blocks);
        DrafterState {
            window: vec![Vec::new(); blocks],
            positions: 0,
        }
    }

    /// Record the next main position without drafting: its `main_hidden`
    /// (`dim · target layers`) becomes one window KV per draft block. Returns
    /// the projected state the blocks read.
    pub fn drafter_observe(
        &self,
        state: &mut DrafterState,
        main_hidden: &[f32],
    ) -> Result<Vec<f32>> {
        let c = self.config();
        let w = self
            .weights
            .drafter
            .as_ref()
            .ok_or_else(|| Error::Input("the drafter is not loaded".into()))?;
        if main_hidden.len() != w.main_proj.cols {
            return Err(Error::Input(format!(
                "main_hidden has {} elements, expected {}",
                main_hidden.len(),
                w.main_proj.cols
            )));
        }
        let mut main_x = vec![0.0; c.dim];
        lin8(main_hidden, &w.main_proj, &mut main_x);
        let main_x = rms(&main_x, &w.main_norm, c.norm_eps);
        let cis = self.rope(RopeKind::Plain).cis(state.positions);
        for (block, ring) in w.blocks.iter().zip(&mut state.window) {
            ring.extend_from_slice(&window_kv(c, &block.attention, &main_x, &cis));
        }
        state.positions += 1;
        Ok(main_x)
    }

    /// Record main position `p` (the next one) and draft from it, given the
    /// token chosen for `p + 1`. Position 0 cannot draft, as in the
    /// reference: record it with [`DsModel::drafter_observe`].
    pub fn drafter_forward(
        &self,
        state: &mut DrafterState,
        main_hidden: &[f32],
        token: u32,
    ) -> Result<DraftCore> {
        let c = self.config();
        if c.drafter.is_none() {
            return Err(Error::Input("this model has no drafter".into()));
        }
        if state.positions == 0 {
            return Err(Error::Input(
                "the drafter cannot draft from position 0".into(),
            ));
        }
        if token as usize >= c.vocab_size {
            return Err(Error::Input(format!(
                "token {token} is outside the vocabulary"
            )));
        }
        self.drafter_observe(state, main_hidden)?;
        let p = state.positions - 1;
        let (window, hd) = (c.attention.window, c.attention.head_dim);
        let context: Vec<Vec<&[f32]>> = state
            .window
            .iter()
            .map(|ring| {
                ring_order(p, window)
                    .into_iter()
                    .map(|pos| &ring[pos * hd..(pos + 1) * hd])
                    .collect()
            })
            .collect();
        self.drafter_slots(p, &context, token)
    }

    /// The draft pass itself, given each draft block's window KV: `context[b]`
    /// lists block `b`'s KV of the main positions a pass at main position `p`
    /// attends to, in [`ring_order`]. This is [`DsModel::drafter_forward`]
    /// for a caller that keeps the window KV itself.
    pub fn drafter_slots(
        &self,
        p: usize,
        context: &[Vec<&[f32]>],
        token: u32,
    ) -> Result<DraftCore> {
        let c = self.config();
        let spec = c
            .drafter
            .as_ref()
            .ok_or_else(|| Error::Input("this model has no drafter".into()))?;
        let w = self
            .weights
            .drafter
            .as_ref()
            .ok_or_else(|| Error::Input("the drafter is not loaded".into()))?;
        if token as usize >= c.vocab_size {
            return Err(Error::Input(format!(
                "token {token} is outside the vocabulary"
            )));
        }
        if context.len() != w.blocks.len() {
            return Err(Error::Input(format!(
                "{} draft blocks' window KV given, the drafter has {}",
                context.len(),
                w.blocks.len()
            )));
        }
        let slots = spec.slots;
        let rope = self.rope(RopeKind::Plain);

        let mut xs: Vec<Vec<f32>> = (0..slots)
            .map(|i| {
                let id = if i == 0 { token } else { spec.noise_token };
                self.weights.embed.row(id as usize).repeat(c.hc_mult)
            })
            .collect();
        let mut pres: Vec<Vec<f32>> = vec![mhc::identity_pre(c.hc_mult); slots];
        for (block, ring) in w.blocks.iter().zip(context) {
            let prepared: Vec<_> = (0..slots)
                .into_par_iter()
                .map(|i| {
                    let cis = rope.cis(p + 1 + i);
                    let (mixes, u) =
                        sub_layer_input(c, &block.hc_attn, &xs[i], &pres[i], &block.attn_norm);
                    let (_, q) = query(c, &block.attention, &u, &cis);
                    let kv = window_kv(c, &block.attention, &u, &cis);
                    (mixes, q, kv, cis)
                })
                .collect();
            // The window in ring-slot order, then every draft slot: each draft
            // position attends to all of them, later ones included.
            let mut entries: Vec<Option<&[f32]>> = ring.iter().map(|kv| Some(*kv)).collect();
            entries.extend(prepared.iter().map(|t| Some(t.2.as_slice())));
            let attended: Vec<_> = (0..slots)
                .into_par_iter()
                .map(|i| {
                    let (mixes, q, _, cis) = &prepared[i];
                    let y = attention_output(c, &block.attention, q, &entries, cis);
                    let mut x = xs[i].clone();
                    mhc::expand(&mut x, &y, mixes);
                    let (ffn_mixes, u) =
                        sub_layer_input(c, &block.hc_ffn, &x, &mixes.pre, &block.ffn_norm);
                    (x, ffn_mixes, u)
                })
                .collect();
            let inputs: Vec<&[f32]> = attended.iter().map(|t| t.2.as_slice()).collect();
            let (ffn, _) = moe_forward(&self.weights, &block.moe, &spec.moe, &inputs)?;
            for (i, (x, mixes, _)) in attended.into_iter().enumerate() {
                let mut x = x;
                mhc::expand(&mut x, ffn.row(i), &mixes);
                xs[i] = x;
                pres[i] = mixes.pre;
            }
        }
        let mut hidden = Matrix::zeros(slots, c.dim);
        let mut base_logits = Matrix::zeros(slots, c.vocab_size);
        for i in 0..slots {
            mhc::collapse(&xs[i], &pres[i], hidden.row_mut(i));
            let normed = rms(hidden.row(i), &w.norm, c.norm_eps);
            linear_f32(&normed, &self.weights.head, base_logits.row_mut(i));
        }
        Ok(DraftCore {
            base_logits,
            hidden,
        })
    }

    /// Slot `slot`'s logits given the token before it in the chain (the given
    /// token for slot 0, else the token drafted by slot `slot - 1`): the base
    /// logits plus the Markov head over that token's Markov embedding.
    pub fn draft_logits(&self, core: &DraftCore, slot: usize, previous: u32) -> Vec<f32> {
        let w = self
            .weights
            .drafter
            .as_ref()
            .expect("a draft core implies a drafter");
        let embed = w.markov_embed.row(previous as usize);
        let mut bias = vec![0.0; self.config().vocab_size];
        linear_f32(embed, &w.markov_head, &mut bias);
        core.base_logits
            .row(slot)
            .iter()
            .zip(&bias)
            .map(|(b, m)| b + m)
            .collect()
    }

    /// Slot `slot`'s raw confidence logit: a linear map of its hidden state
    /// and the Markov embedding of the token before it in the chain.
    pub fn draft_confidence(&self, core: &DraftCore, slot: usize, previous: u32) -> f32 {
        let w = self
            .weights
            .drafter
            .as_ref()
            .expect("a draft core implies a drafter");
        let mut input = core.hidden.row(slot).to_vec();
        input.extend_from_slice(w.markov_embed.row(previous as usize));
        dot(&input, &w.confidence)
    }

    /// The chain the reference draws at temperature 0: each slot takes the
    /// first maximal logit.
    pub fn draft_greedy(&self, core: &DraftCore, token: u32) -> Draft {
        let slots = core.hidden.rows;
        let mut logits = Matrix::zeros(slots, self.config().vocab_size);
        let (mut tokens, mut confidence) = (Vec::new(), Vec::new());
        let mut previous = token;
        for i in 0..slots {
            let row = self.draft_logits(core, i, previous);
            confidence.push(self.draft_confidence(core, i, previous));
            let mut best = 0;
            for (v, &l) in row.iter().enumerate() {
                if l > row[best] {
                    best = v;
                }
            }
            logits.row_mut(i).copy_from_slice(&row);
            previous = best as u32;
            tokens.push(previous);
        }
        Draft {
            tokens,
            logits,
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ring_order;

    #[test]
    fn window_is_listed_by_ring_slot() {
        assert_eq!(ring_order(2, 4), [0, 1, 2]);
        assert_eq!(ring_order(3, 4), [0, 1, 2, 3]);
        // Position 4 overwrote slot 0, position 5 slot 1.
        assert_eq!(ring_order(5, 4), [4, 5, 2, 3]);
        assert_eq!(ring_order(8, 4), [8, 5, 6, 7]);
    }
}
