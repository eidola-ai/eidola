//! The dense reference forward of DeepSeek-V4.1 (`inference/model.py`
//! `Transformer.forward` at one tensor-parallel rank).
//!
//! The model is evaluated one layer at a time over a contiguous range of
//! positions ([`DsModel::layer_forward`]). A layer's output at a position is a
//! function of its input at that position, its own window KV for the
//! positions before it, and the global KV entries the position can see;
//! nothing depends on how positions are grouped into calls. The full forward,
//! the encoder / decoder halves and a triangular replay are all sequences of
//! the same per-layer step.

use rayon::prelude::*;

use super::attention::{candidate_mask, index_score, sparse_attend, top_k};
use super::config::{DsConfig, DsLayerSpec, DsMoeSpec, RopeKind};
use super::engram::{EngramHasher, inject};
use super::linear::{lin8, linear_bf16, linear_f32};
use super::mhc::{self, Mixes};
use super::moe::{DsRouting, route, routed_expert, shared_expert};
use super::numeric::{bf16_round, exp32};
use super::quant::{fake_quant_fp4_e4m3, fake_quant_fp4_pow2, fake_quant_fp8};
use super::rope::{RopeTable, Yarn, rotate, unrotate};
use super::weights::{
    CompressorWeights, DsAttentionWeights, DsMoeWeights, DsWeights, IndexerWeights, rms,
};
use crate::error::{Error, Result};
use crate::tensor::{Matrix, dot};

/// The top-k selection an index source publishes for the layers that reuse it.
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    /// The layer (index in the layer table) whose global KV the entries index.
    pub kv_source: usize,
    /// Per row: the selected entries, ascending.
    pub entries: Vec<Vec<u32>>,
}

/// The candidate blocks the candidate source publishes: per row, one flag per
/// entry the row can see.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidates {
    pub kv_source: usize,
    pub masks: Vec<Vec<bool>>,
}

/// What one layer hands the next for a contiguous range of positions.
#[derive(Debug, Clone, PartialEq)]
pub struct Rows {
    /// The position of row 0.
    pub start: usize,
    /// `[n, hc · dim]`: the residual copies (BF16 values), copy-major.
    pub streams: Matrix,
    /// `[n, hc]`: the collapse weights the next sub-layer applies.
    pub pre_mix: Matrix,
    /// Per Engram layer of the config: `n · lookups` table rows.
    pub engram_rows: Vec<Vec<u64>>,
    pub selection: Option<Selection>,
    pub candidates: Option<Candidates>,
}

impl Rows {
    pub fn len(&self) -> usize {
        self.streams.rows
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn end(&self) -> usize {
        self.start + self.len()
    }

    /// The rows for positions `from..self.end()`.
    pub fn from_position(&self, from: usize) -> Rows {
        assert!(
            from >= self.start && from <= self.end(),
            "position {from} outside {}..{}",
            self.start,
            self.end()
        );
        let (skip, n) = (from - self.start, self.len());
        let rows: Vec<usize> = (skip..n).collect();
        Rows {
            start: from,
            streams: self.streams.select_rows(&rows),
            pre_mix: self.pre_mix.select_rows(&rows),
            engram_rows: self
                .engram_rows
                .iter()
                .map(|r| r[r.len() / n.max(1) * skip..].to_vec())
                .collect(),
            selection: self.selection.as_ref().map(|s| Selection {
                kv_source: s.kv_source,
                entries: s.entries[skip..].to_vec(),
            }),
            candidates: self.candidates.as_ref().map(|c| Candidates {
                kv_source: c.kv_source,
                masks: c.masks[skip..].to_vec(),
            }),
        }
    }
}

/// A KV-source layer's global bank: one entry per `ratio` tokens, each the
/// attention KV (RoPE'd, fake-FP4 with E4M3 scales) and the index key
/// (RoPE'd, fake-FP4 with power-of-two scales).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GlobalBank {
    pub entries: usize,
    /// `[entries, head_dim]`.
    pub kv: Vec<f32>,
    /// `[entries, index_head_dim]`.
    pub index_keys: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
struct LayerState {
    started: bool,
    /// Position of the first window KV held.
    first: usize,
    /// `[positions, head_dim]`.
    window: Vec<f32>,
    bank: Option<GlobalBank>,
    /// The bank was supplied from outside; this layer does not extend it.
    bank_supplied: bool,
    /// The compressor's incomplete group: per waiting token, its f32 KV and
    /// gate score.
    pending: Vec<(Vec<f32>, Vec<f32>)>,
    /// Tokens the compressor has consumed.
    compressed: usize,
}

/// Everything one sequence carries between calls. The dense reference keeps
/// every position's window KV; an executor keeps the last `window - 1`.
#[derive(Debug, Clone)]
pub struct SequenceState {
    layers: Vec<LayerState>,
    /// The tokens embedded so far (the Engram lookback reads the last few).
    tokens: Vec<u32>,
}

impl SequenceState {
    /// The global bank layer `layer` has built (or was given).
    pub fn bank(&self, layer: usize) -> Option<&GlobalBank> {
        self.layers[layer].bank.as_ref()
    }

    /// Give `layer` its global bank from outside: the layer then reads it and
    /// does not run its own compressor. This is how a decoder half receives
    /// the bank the encoder's stub computed.
    pub fn supply_bank(&mut self, layer: usize, bank: GlobalBank) {
        self.layers[layer].bank = Some(bank);
        self.layers[layer].bank_supplied = true;
    }

    /// Positions embedded so far.
    pub fn position(&self) -> usize {
        self.tokens.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LayerOptions {
    /// Rows before this position only contribute their window KV (and, in a
    /// KV source, their global entries); rows from it on are computed in full.
    pub emit_from: usize,
    /// A query at `i` sees window keys in `[max(window_start, i - window + 1), i]`.
    /// 0 is the reference.
    pub window_start: usize,
    /// Measurement only. `Some(s)` lays the attention list out as the
    /// reference does for a prefill chunk of `s` tokens from position 0: the
    /// window part has `min(s, window)` slots holding the positions from
    /// `max(0, i - window + 1)` on in order, so a window that is not yet full
    /// is left-aligned (the slots after the query are empty) and the global
    /// entries start at slot `min(s, window)`. `None` is the canonical layout
    /// (the window right-aligned in `window` slots), which is the reference's
    /// for one token per call. The two put entries in different 64-slot
    /// blocks wherever a query's window is not full and its list crosses a
    /// block boundary.
    pub reference_prefill: Option<usize>,
}

/// One layer's result for the emitted rows.
#[derive(Debug, Clone)]
pub struct LayerOutput {
    pub rows: Rows,
    /// `[n, dim]`: the mean of the copies entering the block (after Engram),
    /// which the drafter reads at its target layers.
    pub entering_mean: Matrix,
    /// `[n, dim]`: the attention sub-layer's output.
    pub attention: Matrix,
    /// `[n, dim]`: the feed-forward sub-layer's output.
    pub ffn: Matrix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DsForwardOptions {
    /// Keep every layer's output streams.
    pub taps: bool,
    /// Skip the output head.
    pub no_logits: bool,
    pub window_start: usize,
    /// See [`LayerOptions::reference_prefill`].
    pub reference_prefill: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct DsForwardOutput {
    /// `[n, vocab]`, empty when the head was skipped.
    pub logits: Matrix,
    /// `[n, dim]`: the final normed state the head reads.
    pub normed: Matrix,
    /// Per layer run, `[n, hc · dim]`, when asked for.
    pub taps: Vec<Matrix>,
    /// `[n, dim · target layers]` when the config has a drafter and every
    /// target layer was run.
    pub main_hidden: Option<Matrix>,
    /// The rows after the last layer run.
    pub rows: Rows,
}

/// A global bank crossing the boundary, with the layer it belongs to.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryBank {
    /// The KV-source layer (index in the layer table) whose bank this is:
    /// always the first decoder layer.
    pub layer: usize,
    pub bank: GlobalBank,
}

/// The values that cross from the encoder half to the decoder half.
///
/// A split is supported only where no decoder layer reads a global bank an
/// encoder layer built ([`DsModel::check_split`]), so the one bank that can
/// cross is the first decoder layer's own, which the encoder computes from
/// the rows it hands over.
#[derive(Debug, Clone, PartialEq)]
pub struct Boundary {
    /// The streams and collapse weights entering the first decoder layer,
    /// with the Engram rows the decoder may still need.
    pub rows: Rows,
    /// The first decoder layer's global bank over every position so far;
    /// present exactly when that layer is a KV source.
    pub bank: Option<BoundaryBank>,
}

pub struct DsModel {
    pub weights: DsWeights,
    hasher: Option<EngramHasher>,
    plain: RopeTable,
    compressed: RopeTable,
}

struct Prepared {
    x: Vec<f32>,
    mixes: Mixes,
    u: Vec<f32>,
    window_kv: Vec<f32>,
    mean: Vec<f32>,
}

impl DsModel {
    /// `token_map` is the Engram compressed-vocabulary map (see
    /// [`EngramHasher::new`]); required when a kept layer has an Engram module.
    pub fn new(weights: DsWeights, token_map: Option<Vec<u32>>) -> Result<Self> {
        let c = &weights.config;
        let uses_engram = c.layers.iter().any(|l| l.engram.is_some());
        let hasher = match (&c.engram, token_map) {
            (Some(spec), Some(map)) => Some(EngramHasher::new(spec, c.vocab_size, map)?),
            (Some(_), None) if uses_engram => {
                return Err(Error::Input(
                    "this model has Engram layers and needs the compressed-vocabulary map".into(),
                ));
            }
            _ => None,
        };
        let r = &c.rope;
        let yarn = (r.original_seq_len > 0).then_some(Yarn {
            original_seq_len: r.original_seq_len,
            factor: r.factor,
            beta_fast: r.beta_fast,
            beta_slow: r.beta_slow,
        });
        Ok(DsModel {
            plain: RopeTable::new(c.attention.rope_dim, r.theta, None),
            compressed: RopeTable::new(c.attention.rope_dim, r.compress_theta, yarn),
            hasher,
            weights,
        })
    }

    pub fn config(&self) -> &DsConfig {
        &self.weights.config
    }

    pub fn new_state(&self) -> SequenceState {
        SequenceState {
            layers: vec![LayerState::default(); self.config().layers.len()],
            tokens: Vec::new(),
        }
    }

    /// The rotation table of a RoPE variant.
    pub fn rope(&self, kind: RopeKind) -> &RopeTable {
        match kind {
            RopeKind::Plain => &self.plain,
            RopeKind::Compressed => &self.compressed,
        }
    }

    /// The table rows Engram layer `slot` (index in the config's Engram
    /// layers) reads for the position whose token is the last of `history`:
    /// the sequence from its start, or at least its last `max_ngram` tokens.
    pub fn engram_rows(&self, slot: usize, history: &[u32], position: usize) -> Result<Vec<u64>> {
        match (&self.config().engram, &self.hasher) {
            (Some(spec), Some(hasher)) => Ok(hasher.rows(&spec.layers[slot], history, position)),
            _ => Err(Error::Input(
                "this model has no Engram hasher (no compressed-vocabulary map was given)".into(),
            )),
        }
    }

    /// Embed the next tokens of a sequence: each copy of the stream starts as
    /// the token's embedding, with the one-hot collapse. Also resolves the
    /// Engram table rows, which depend on the tokens before.
    pub fn embed(&self, state: &mut SequenceState, tokens: &[u32]) -> Result<Rows> {
        let c = self.config();
        let start = state.tokens.len();
        let mut streams = Matrix::zeros(tokens.len(), c.hc_mult * c.dim);
        let mut pre_mix = Matrix::zeros(tokens.len(), c.hc_mult);
        for (r, &t) in tokens.iter().enumerate() {
            if t as usize >= c.vocab_size {
                return Err(Error::Input(format!(
                    "token {t} is outside the vocabulary of {}",
                    c.vocab_size
                )));
            }
            if c.image_token_id == Some(t) {
                return Err(Error::Input(format!(
                    "token {t} is an image position; the vision path is not implemented"
                )));
            }
            for copy in streams.row_mut(r).chunks_exact_mut(c.dim) {
                copy.copy_from_slice(self.weights.embed.row(t as usize));
            }
            pre_mix
                .row_mut(r)
                .copy_from_slice(&mhc::identity_pre(c.hc_mult));
        }
        state.tokens.extend_from_slice(tokens);
        let engram_rows = match (&c.engram, &self.hasher) {
            (Some(spec), Some(hasher)) => spec
                .layers
                .iter()
                .map(|layer| {
                    (0..tokens.len())
                        .flat_map(|r| hasher.rows(layer, &state.tokens[..start + r + 1], start + r))
                        .collect()
                })
                .collect(),
            _ => Vec::new(),
        };
        Ok(Rows {
            start,
            streams,
            pre_mix,
            engram_rows,
            selection: None,
            candidates: None,
        })
    }

    /// The first half of a block for one row: Engram, the attention
    /// sub-layer's coefficients and input, and this layer's window KV.
    fn prepare(&self, layer: usize, rows: &Rows, r: usize) -> Result<Prepared> {
        let c = self.config();
        let (spec, w) = (&c.layers[layer], &self.weights.layers[layer]);
        let position = rows.start + r;
        let mut x = rows.streams.row(r).to_vec();
        if let (Some(slot), Some(e)) = (spec.engram, &w.engram) {
            let lookups = c.engram.as_ref().unwrap().lookups();
            let table_rows = rows.engram_rows.get(slot).ok_or_else(|| {
                Error::Input("these rows carry no Engram table rows (no compressed-vocabulary map was given)".into())
            })?;
            inject(
                e,
                &table_rows[r * lookups..(r + 1) * lookups],
                &mut x,
                c.hc_mult,
                c.norm_eps,
            )?;
        }
        let mean = stream_mean(&x, c.hc_mult);
        let (mixes, u) = sub_layer_input(c, &w.hc_attn, &x, rows.pre_mix.row(r), &w.attn_norm);
        let window_kv = window_kv(c, &w.attention, &u, &self.rope(spec.rope).cis(position));
        Ok(Prepared {
            x,
            mixes,
            u,
            window_kv,
            mean,
        })
    }

    /// Append the global entries a KV source derives from one position's
    /// attention input, when that position completes a group.
    fn compress(&self, spec: &DsLayerSpec, cw: &CompressorWeights, st: &mut LayerState, u: &[f32]) {
        let c = self.config();
        let (hd, ratio) = (c.attention.head_dim, spec.compress_ratio);
        let position = st.compressed;
        st.compressed += 1;
        let latent = match &cw.wgate {
            None => {
                let mut kv = vec![0.0; hd];
                linear_bf16(u, &cw.wkv, &mut kv);
                rms(&kv, &cw.norm, c.norm_eps)
            }
            Some(wgate) => {
                let (mut kv, mut score) = (vec![0.0; hd], vec![0.0; hd]);
                linear_f32(u, &cw.wkv, &mut kv);
                linear_f32(u, wgate, &mut score);
                st.pending.push((kv, score));
                if st.pending.len() < ratio {
                    return;
                }
                let group = std::mem::take(&mut st.pending);
                pooled_latent(c, cw, &group)
            }
        };
        // An entry stands at the first position of its group.
        let cis = self.rope(spec.rope).cis(position + 1 - ratio);
        let mut key = vec![0.0; c.indexer.head_dim];
        linear_bf16(&latent, &cw.wk, &mut key);
        let mut key = rms(&key, &cw.k_norm, c.norm_eps);
        rotate(&mut key, &cis);
        fake_quant_fp4_pow2(&mut key);
        let mut kv = latent;
        rotate(&mut kv, &cis);
        fake_quant_fp4_e4m3(&mut kv);
        let bank = st.bank.get_or_insert_with(GlobalBank::default);
        bank.entries += 1;
        bank.kv.extend_from_slice(&kv);
        bank.index_keys.extend_from_slice(&key);
    }

    /// The global bank the layer `layer` would build from `rows`, without
    /// running the layer: the encoder half's stub for the first decoder
    /// layer. `state` accumulates across calls like any layer's.
    pub fn bank_stub(
        &self,
        layer: usize,
        state: &mut SequenceState,
        rows: &Rows,
    ) -> Result<GlobalBank> {
        let c = self.config();
        let (spec, w) = (&c.layers[layer], &self.weights.layers[layer]);
        let cw = w
            .attention
            .compressor
            .as_ref()
            .ok_or_else(|| Error::Input(format!("layer {layer} is not a KV source")))?;
        if state.layers[layer].compressed != rows.start {
            return Err(Error::Input(format!(
                "layer {layer}'s global KV covers {} positions; the rows start at {}",
                state.layers[layer].compressed, rows.start
            )));
        }
        let inputs: Vec<Vec<f32>> = (0..rows.len())
            .into_par_iter()
            .map(|r| {
                let mut x = rows.streams.row(r).to_vec();
                if let (Some(slot), Some(e)) = (spec.engram, &w.engram) {
                    let lookups = c.engram.as_ref().unwrap().lookups();
                    inject(
                        e,
                        &rows.engram_rows[slot][r * lookups..(r + 1) * lookups],
                        &mut x,
                        c.hc_mult,
                        c.norm_eps,
                    )?;
                }
                Ok(sub_layer_input(c, &w.hc_attn, &x, rows.pre_mix.row(r), &w.attn_norm).1)
            })
            .collect::<Result<_>>()?;
        let st = &mut state.layers[layer];
        for u in &inputs {
            self.compress(spec, cw, st, u);
        }
        Ok(st.bank.clone().unwrap_or_default())
    }

    /// Run one layer over a contiguous range of positions.
    ///
    /// `rows` continues where this layer's state stops (or starts it, at any
    /// position). Every row adds its window KV to the state; rows from
    /// `opts.emit_from` on are computed in full and returned. A returned row
    /// needs the layer's window KV back to `max(window_start, i - window + 1)`
    /// and, on a layer with a compress ratio, the global bank up to its
    /// position; a missing one is an error, never a silently shorter window.
    pub fn layer_forward(
        &self,
        layer: usize,
        state: &mut SequenceState,
        rows: &Rows,
        opts: LayerOptions,
    ) -> Result<LayerOutput> {
        let c = self.config();
        let (spec, w) = (&c.layers[layer], &self.weights.layers[layer]);
        let a = &c.attention;
        let hd = a.head_dim;
        let n = rows.len();
        {
            let st = &mut state.layers[layer];
            if !st.started {
                st.started = true;
                st.first = rows.start;
            } else if st.first + st.window.len() / hd != rows.start {
                return Err(Error::Input(format!(
                    "layer {layer} holds window KV up to position {}; the rows start at {}",
                    st.first + st.window.len() / hd,
                    rows.start
                )));
            }
        }
        let prepared: Vec<Prepared> = (0..n)
            .into_par_iter()
            .map(|r| self.prepare(layer, rows, r))
            .collect::<Result<_>>()?;
        {
            let st = &mut state.layers[layer];
            for p in &prepared {
                st.window.extend_from_slice(&p.window_kv);
            }
            if let (Some(cw), false) = (&w.attention.compressor, st.bank_supplied) {
                if st.compressed != rows.start {
                    return Err(Error::Input(format!(
                        "layer {layer} builds a global KV and must see every position from {}; the rows start at {}",
                        st.compressed, rows.start
                    )));
                }
                for p in &prepared {
                    self.compress(spec, cw, st, &p.u);
                }
            }
        }

        let skip = opts.emit_from.saturating_sub(rows.start).min(n);
        let emitted = n - skip;
        let st = &state.layers[layer];
        // A source that has not completed a group yet has an empty bank.
        let empty = GlobalBank::default();
        let bank = spec
            .kv_source
            .map(|src| state.layers[src].bank.as_ref().unwrap_or(&empty));
        let inherited = rows
            .selection
            .as_ref()
            .filter(|s| Some(s.kv_source) == spec.kv_source);
        let pool = rows
            .candidates
            .as_ref()
            .filter(|s| Some(s.kv_source) == spec.kv_source);

        struct Attended {
            x: Vec<f32>,
            attention: Vec<f32>,
            mixes: Mixes,
            u: Vec<f32>,
            selection: Vec<u32>,
            candidates: Option<Vec<bool>>,
        }
        let attended: Vec<Attended> = (skip..n)
            .into_par_iter()
            .map(|r| {
                let p = &prepared[r];
                let position = rows.start + r;
                let cis = self.rope(spec.rope).cis(position);
                let (qr, q) = query(c, &w.attention, &p.u, &cis);

                let mut slots: Vec<Option<&[f32]>> = Vec::with_capacity(a.window + c.indexer.top_k);
                let window_slots = opts.reference_prefill.map_or(a.window, |chunk| chunk.min(a.window));
                for i in 0..window_slots {
                    let pos = match opts.reference_prefill {
                        None => (position + i + 1).checked_sub(a.window),
                        Some(_) => Some((position + 1).saturating_sub(a.window) + i).filter(|pos| *pos <= position),
                    }
                    .filter(|pos| *pos >= opts.window_start);
                    slots.push(match pos {
                        None => None,
                        Some(pos) if pos < st.first => {
                            return Err(Error::Input(format!(
                                "layer {layer} at position {position} needs its window KV from position {pos}; it holds them from {}",
                                st.first
                            )));
                        }
                        Some(pos) => {
                            Some(&st.window[(pos - st.first) * hd..(pos - st.first + 1) * hd])
                        }
                    });
                }
                let (mut selection, mut candidates) = (Vec::new(), None);
                if let Some(bank) = bank {
                    let visible = (position + 1) / spec.compress_ratio;
                    if bank.entries < visible {
                        return Err(Error::Input(format!(
                            "layer {layer} at position {position} sees {visible} global entries; the bank holds {}",
                            bank.entries
                        )));
                    }
                    if let Some(ix) = &w.attention.indexer {
                        if visible > 0 {
                            let ihd = c.indexer.head_dim;
                            let (iq, weights) = index_query(c, ix, &qr, &p.u, &cis);
                            let scores: Vec<f32> = (0..visible)
                                .map(|t| index_score(&iq, &bank.index_keys[t * ihd..(t + 1) * ihd], &weights))
                                .collect();
                            // The candidate source publishes the blocks and
                            // selects over everything it can see itself.
                            let mut mask = None;
                            if spec.is_candidate_source {
                                let k = c.candidates.as_ref().unwrap();
                                candidates =
                                    Some(candidate_mask(&scores, visible, k.top_blocks, k.block_size));
                            } else if spec.uses_candidates {
                                let pool = pool.ok_or_else(|| {
                                    Error::Input(format!(
                                        "layer {layer} selects inside candidate blocks its rows do not carry"
                                    ))
                                })?;
                                mask = Some(pool.masks[r].clone());
                            }
                            let allowed = (0..visible).filter(|t| mask.as_ref().is_none_or(|m| m[*t]));
                            let mut chosen = top_k(&scores, allowed, c.indexer.top_k);
                            chosen.sort_unstable();
                            selection = chosen.into_iter().map(|t| t as u32).collect();
                        }
                    } else {
                        let inherited = inherited.ok_or_else(|| {
                            Error::Input(format!("layer {layer} reuses a top-k selection its rows do not carry"))
                        })?;
                        selection = inherited.entries[r].clone();
                    }
                    for &t in &selection {
                        slots.push(Some(&bank.kv[t as usize * hd..(t as usize + 1) * hd]));
                    }
                }
                let attention = attention_output(c, &w.attention, &q, &slots, &cis);
                let mut x = p.x.clone();
                mhc::expand(&mut x, &attention, &p.mixes);
                let (mixes, u) = sub_layer_input(c, &w.hc_ffn, &x, &p.mixes.pre, &w.ffn_norm);
                Ok(Attended { x, attention, mixes, u, selection, candidates })
            })
            .collect::<Result<_>>()?;

        let inputs: Vec<&[f32]> = attended.iter().map(|t| t.u.as_slice()).collect();
        let ffn = moe_forward(&self.weights, &w.moe, &c.moe, &inputs)?;

        let mut out = Rows {
            start: rows.start + skip,
            streams: Matrix::zeros(emitted, c.hc_mult * c.dim),
            pre_mix: Matrix::zeros(emitted, c.hc_mult),
            engram_rows: rows.from_position(rows.start + skip).engram_rows,
            selection: None,
            candidates: None,
        };
        let mut entering_mean = Matrix::zeros(emitted, c.dim);
        let mut attention = Matrix::zeros(emitted, c.dim);
        for (i, t) in attended.iter().enumerate() {
            let mut x = t.x.clone();
            mhc::expand(&mut x, ffn.row(i), &t.mixes);
            out.streams.row_mut(i).copy_from_slice(&x);
            out.pre_mix.row_mut(i).copy_from_slice(&t.mixes.pre);
            entering_mean
                .row_mut(i)
                .copy_from_slice(&prepared[skip + i].mean);
            attention.row_mut(i).copy_from_slice(&t.attention);
        }
        // A selection stays with the rows until a layer over another bank
        // replaces it; the same for the candidate blocks.
        if let Some(kv_source) = spec.kv_source {
            out.selection = Some(Selection {
                kv_source,
                entries: attended.iter().map(|t| t.selection.clone()).collect(),
            });
            out.candidates = if spec.is_candidate_source {
                Some(Candidates {
                    kv_source,
                    masks: attended
                        .iter()
                        .map(|t| t.candidates.clone().unwrap_or_default())
                        .collect(),
                })
            } else {
                pool.map(|p| Candidates {
                    kv_source: p.kv_source,
                    masks: p.masks[skip..].to_vec(),
                })
            };
        } else {
            let tail = rows.from_position(rows.start + skip);
            out.selection = tail.selection;
            out.candidates = tail.candidates;
        }
        Ok(LayerOutput {
            rows: out,
            entering_mean,
            attention,
            ffn,
        })
    }

    /// Collapse the final streams, apply the final norm and the output head.
    /// Returns `(normed, logits)`; `logits` is empty unless `head` is set.
    pub fn head(&self, rows: &Rows, head: bool) -> (Matrix, Matrix) {
        let c = self.config();
        let n = rows.len();
        let mut normed = Matrix::zeros(n, c.dim);
        let mut logits = Matrix::zeros(if head { n } else { 0 }, c.vocab_size);
        for r in 0..n {
            let mut h = vec![0.0; c.dim];
            mhc::collapse(rows.streams.row(r), rows.pre_mix.row(r), &mut h);
            normed
                .row_mut(r)
                .copy_from_slice(&rms(&h, &self.weights.final_norm, c.norm_eps));
            if head {
                linear_f32(normed.row(r), &self.weights.head, logits.row_mut(r));
            }
        }
        (normed, logits)
    }

    /// Run layers `layers` over `rows` and, when the range reaches the last
    /// layer, the head.
    pub fn forward_layers(
        &self,
        state: &mut SequenceState,
        mut rows: Rows,
        layers: std::ops::Range<usize>,
        opts: &DsForwardOptions,
    ) -> Result<DsForwardOutput> {
        let c = self.config();
        let n_layers = c.layers.len();
        let targets = c
            .drafter
            .as_ref()
            .map(|d| d.target_layers.clone())
            .unwrap_or_default();
        let mut hidden: Vec<Matrix> = Vec::new();
        let mut taps = Vec::new();
        let last = layers.end == n_layers;
        for layer in layers {
            let out = self.layer_forward(
                layer,
                state,
                &rows,
                LayerOptions {
                    emit_from: rows.start,
                    window_start: opts.window_start,
                    reference_prefill: opts.reference_prefill,
                },
            )?;
            if targets.contains(&layer) {
                hidden.push(out.entering_mean);
            }
            rows = out.rows;
            if opts.taps {
                taps.push(rows.streams.clone());
            }
        }
        let main_hidden = (!targets.is_empty() && hidden.len() == targets.len()).then(|| {
            let mut m = Matrix::zeros(rows.len(), c.dim * targets.len());
            for (t, part) in hidden.iter().enumerate() {
                for r in 0..rows.len() {
                    m.row_mut(r)[t * c.dim..(t + 1) * c.dim].copy_from_slice(part.row(r));
                }
            }
            m
        });
        let (normed, logits) = if last {
            self.head(&rows, !opts.no_logits)
        } else {
            (Matrix::zeros(0, c.dim), Matrix::zeros(0, c.vocab_size))
        };
        Ok(DsForwardOutput {
            logits,
            normed,
            taps,
            main_hidden,
            rows,
        })
    }

    /// The full forward over the next tokens of a sequence: every layer on
    /// every token. Feeding a sequence in any chunking gives the same rows.
    pub fn forward(
        &self,
        state: &mut SequenceState,
        tokens: &[u32],
        opts: &DsForwardOptions,
    ) -> Result<DsForwardOutput> {
        let rows = self.embed(state, tokens)?;
        self.forward_layers(state, rows, 0..self.config().layers.len(), opts)
    }

    /// Whether the model can be split into layers `0..split` and `split..`.
    ///
    /// Both halves need a layer, and no decoder layer may read a global bank
    /// that an encoder layer built: the boundary carries the first decoder
    /// layer's own bank and nothing of the encoder's. In the released model
    /// that allows a split before layers 1, 2, 8, 14 and 20.
    pub fn check_split(&self, split: usize) -> Result<()> {
        let layers = &self.config().layers;
        if split == 0 || split >= layers.len() {
            return Err(Error::Input(format!(
                "split {split} leaves a half without layers"
            )));
        }
        match layers[split..]
            .iter()
            .find(|l| l.kv_source.is_some_and(|src| src < split))
        {
            Some(l) => Err(Error::Input(format!(
                "split {split} is not supported: layer {} reads the global KV of layer {}, which would be on the encoder side",
                l.index,
                l.kv_source.unwrap_or_default()
            ))),
            None => Ok(()),
        }
    }

    /// The encoder half: the embedding and layers `0..split`, plus the stub
    /// that builds layer `split`'s global bank when it is a KV source. An
    /// unsupported split ([`DsModel::check_split`]) is refused before the
    /// state changes.
    pub fn encode(
        &self,
        state: &mut SequenceState,
        tokens: &[u32],
        split: usize,
        opts: &DsForwardOptions,
    ) -> Result<Boundary> {
        self.check_split(split)?;
        let rows = self.embed(state, tokens)?;
        let rows = self.forward_layers(state, rows, 0..split, opts)?.rows;
        let bank = match self.config().layers[split].is_kv_source {
            true => Some(BoundaryBank {
                layer: split,
                bank: self.bank_stub(split, state, &rows)?,
            }),
            false => None,
        };
        Ok(Boundary { rows, bank })
    }

    /// The decoder half: layers `split..` and the head, over the boundary's
    /// rows. `state` is the decoder's own. A boundary that does not fit the
    /// split is refused before the state changes.
    pub fn decode(
        &self,
        state: &mut SequenceState,
        boundary: &Boundary,
        split: usize,
        opts: &DsForwardOptions,
    ) -> Result<DsForwardOutput> {
        self.check_split(split)?;
        let source = self.config().layers[split].is_kv_source;
        match &boundary.bank {
            Some(b) if b.layer != split || !source => {
                return Err(Error::Input(format!(
                    "the boundary carries layer {}'s global KV; a split at {split} {}",
                    b.layer,
                    if source {
                        "needs that layer's own"
                    } else {
                        "carries none"
                    }
                )));
            }
            None if source => {
                return Err(Error::Input(format!(
                    "the boundary carries no global KV; layer {split} needs its own from the encoder"
                )));
            }
            Some(b) => state.supply_bank(split, b.bank.clone()),
            None => {}
        }
        self.forward_layers(
            state,
            boundary.rows.clone(),
            split..self.config().layers.len(),
            opts,
        )
    }
}

/// A pooled global entry's latent from its group's tokens, each as its f32
/// `(kv, gate score)`: per channel the softmax of the scores weights the KVs,
/// in f32 and in token order, the sum is rounded to BF16, and the result is
/// normed.
pub fn pooled_latent(
    c: &DsConfig,
    cw: &CompressorWeights,
    group: &[(Vec<f32>, Vec<f32>)],
) -> Vec<f32> {
    let hd = c.attention.head_dim;
    let pooled: Vec<f32> = (0..hd)
        .map(|ch| {
            let max = group.iter().fold(f32::NEG_INFINITY, |m, g| m.max(g.1[ch]));
            let e: Vec<f32> = group.iter().map(|g| exp32(g.1[ch] - max)).collect();
            let sum = e.iter().fold(0.0f32, |a, v| a + v);
            let mut pooled = 0.0f32;
            for (g, e) in group.iter().zip(&e) {
                pooled += g.0[ch] * (e / sum);
            }
            bf16_round(pooled)
        })
        .collect();
    rms(&pooled, &cw.norm, c.norm_eps)
}

/// An index source's query for one position: the per-head index queries
/// (projected from the query latent `qr`, rotated, fake-FP4) and the per-head
/// weights (from the attention input `u`) that [`index_score`] combines.
pub fn index_query(
    c: &DsConfig,
    ix: &IndexerWeights,
    qr: &[f32],
    u: &[f32],
    cis: &[(f32, f32)],
) -> (Vec<f32>, Vec<f32>) {
    let mut iq = vec![0.0; ix.wq_b.rows];
    lin8(qr, &ix.wq_b, &mut iq);
    for head in iq.chunks_exact_mut(c.indexer.head_dim) {
        rotate(head, cis);
    }
    fake_quant_fp4_pow2(&mut iq);
    let weights = (0..c.indexer.n_heads)
        .map(|h| bf16_round(bf16_round(dot(u, ix.weights_proj.row(h))) * c.indexer.weight_scale))
        .collect();
    (iq, weights)
}

/// The mean of the residual copies, as a BF16 vector.
pub fn stream_mean(x: &[f32], hc: usize) -> Vec<f32> {
    let dim = x.len() / hc;
    (0..dim)
        .map(|d| {
            let mut s = 0.0f32;
            for j in 0..hc {
                s += x[j * dim + d];
            }
            bf16_round(s / hc as f32)
        })
        .collect()
}

/// A sub-layer's own coefficients (from the stream entering it) and its
/// input: the stream collapsed with the previous sub-layer's `pre`, normed.
pub fn sub_layer_input(
    c: &DsConfig,
    w: &mhc::MhcWeights,
    x: &[f32],
    pre: &[f32],
    norm: &[f32],
) -> (Mixes, Vec<f32>) {
    let mixes = mhc::mixes(w, x, c.hc_mult, c.norm_eps, c.hc_eps, c.hc_sinkhorn_iters);
    let mut h = vec![0.0; c.dim];
    mhc::collapse(x, pre, &mut h);
    (mixes, rms(&h, norm, c.norm_eps))
}

/// The query latent and the rotated per-head queries of one position.
pub fn query(
    c: &DsConfig,
    w: &DsAttentionWeights,
    u: &[f32],
    cis: &[(f32, f32)],
) -> (Vec<f32>, Vec<f32>) {
    let mut latent = vec![0.0; w.wq_a.rows];
    lin8(u, &w.wq_a, &mut latent);
    let latent = rms(&latent, &w.q_norm, c.norm_eps);
    let mut q = vec![0.0; w.wq_b.rows];
    lin8(&latent, &w.wq_b, &mut q);
    for head in q.chunks_exact_mut(c.attention.head_dim) {
        rotate(head, cis);
    }
    (latent, q)
}

/// A position's window KV: projected, normed, rotated, then fake-FP8 over the
/// whole vector.
pub fn window_kv(c: &DsConfig, w: &DsAttentionWeights, u: &[f32], cis: &[(f32, f32)]) -> Vec<f32> {
    let mut kv = vec![0.0; w.wkv.rows];
    lin8(u, &w.wkv, &mut kv);
    let mut kv = rms(&kv, &w.kv_norm, c.norm_eps);
    rotate(&mut kv, cis);
    fake_quant_fp8(&mut kv);
    kv
}

/// Attention over `slots` and the output projection: the query's rotation is
/// undone on each head's output, the result is fake-FP8 quantised, projected
/// group by group through `wo_a` (BF16) and through `wo_b`.
pub fn attention_output(
    c: &DsConfig,
    w: &DsAttentionWeights,
    q: &[f32],
    slots: &[Option<&[f32]>],
    cis: &[(f32, f32)],
) -> Vec<f32> {
    let a = &c.attention;
    let mut o = vec![0.0; q.len()];
    sparse_attend(q, slots, &w.sink, a.softmax_scale, &mut o);
    for head in o.chunks_exact_mut(a.head_dim) {
        unrotate(head, cis);
    }
    fake_quant_fp8(&mut o);
    let width = o.len() / a.o_groups;
    let z: Vec<f32> = (0..a.o_groups * a.o_lora_rank)
        .map(|row| {
            let g = row / a.o_lora_rank;
            bf16_round(dot(&o[g * width..(g + 1) * width], w.wo_a.row(row)))
        })
        .collect();
    let mut y = vec![0.0; w.wo_b.rows];
    lin8(&z, &w.wo_b, &mut y);
    y
}

/// The feed-forward sub-layer for a set of tokens: per token, the routed
/// experts' outputs accumulate in f32 in ascending expert index, the shared
/// expert is added last, and the sum is rounded to BF16.
pub fn moe_forward(
    weights: &DsWeights,
    w: &DsMoeWeights,
    spec: &DsMoeSpec,
    inputs: &[&[f32]],
) -> Result<Matrix> {
    let dim = weights.config.dim;
    let routing: Vec<DsRouting> = inputs
        .par_iter()
        .map(|x| route(spec, &w.router, &w.bias, x))
        .collect();
    let mut y = Matrix::zeros(inputs.len(), dim);
    for e in 0..spec.n_routed {
        let users: Vec<(usize, f32)> = routing
            .iter()
            .enumerate()
            .filter_map(|(t, r)| {
                r.experts
                    .iter()
                    .position(|x| *x == e)
                    .map(|i| (t, r.weights[i]))
            })
            .collect();
        if users.is_empty() {
            continue;
        }
        let expert = weights.expert(w, spec, e)?;
        let outs: Vec<Vec<f32>> = users
            .par_iter()
            .map(|&(t, weight)| routed_expert(&expert, inputs[t], spec.swiglu_limit, weight))
            .collect();
        for (&(t, _), out) in users.iter().zip(&outs) {
            for (a, b) in y.row_mut(t).iter_mut().zip(out) {
                *a += *b;
            }
        }
    }
    let shared: Vec<Vec<f32>> = inputs
        .par_iter()
        .map(|x| shared_expert(&w.shared, x, spec.swiglu_limit))
        .collect();
    for (t, out) in shared.iter().enumerate() {
        for (a, b) in y.row_mut(t).iter_mut().zip(out) {
            *a = bf16_round(*a + *b);
        }
    }
    Ok(y)
}
