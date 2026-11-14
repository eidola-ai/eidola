//! One model layer over a set of positions, with every KV read and written through a
//! device's paged memory. It composes the model crate's kernels in the order
//! `DsModel::layer_forward` does, so each value equals the dense forward's bit for bit;
//! the attention list is the canonical one (the window right-aligned in `window` slots,
//! then the selected global entries, ascending).

use eidola_engine::executor::{ExecutorError, Slot};
use eidola_engine_model::deepseek::DsConfig;
use eidola_engine_model::deepseek::attention::{candidate_mask, index_score, top_k};
use eidola_engine_model::deepseek::engram::inject_gathered;
use eidola_engine_model::deepseek::forward::{
    DsModel, attention_output, index_query, moe_forward, pooled_latent, query, stream_mean,
    sub_layer_input,
};
use eidola_engine_model::deepseek::linear::{lin8, linear_bf16, linear_f32};
use eidola_engine_model::deepseek::mhc::{self, Mixes};
use eidola_engine_model::deepseek::rope::rotate;
use eidola_engine_model::deepseek::weights::{CompressorWeights, DsAttentionWeights, rms};
use rayon::prelude::*;

use super::memory::Memory;
use super::packed::{Fp4E4m3, Fp4Pow2, Fp8};

/// What a position carries from one layer to the next.
#[derive(Clone, Debug)]
pub(crate) struct Flow {
    /// The residual copies (`hc · dim`).
    pub x: Vec<f32>,
    /// The collapse weights the next sub-layer applies.
    pub pre: Vec<f32>,
    /// The selection an index source published, with the bank it indexes.
    pub selection: Option<(usize, Vec<u32>)>,
    /// The candidate blocks the candidate source published.
    pub candidates: Option<(usize, Vec<bool>)>,
}

/// One position of a layer step.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerItem {
    pub slot: Slot,
    pub pos: u32,
    pub token: u32,
    /// The layer computes this position: it writes its window KV (and, a source, its
    /// global entries).
    pub active: bool,
    /// The layer's output at this position is consumed.
    pub emit: bool,
}

/// Where a layer's KV lives on its device.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerPlan {
    /// Index in the model's layer table.
    pub layer: usize,
    /// The window KV's `(group, layer within the group)`.
    pub window: (usize, usize),
    /// The global bank the layer reads: `(group, layer within the group)`.
    pub reads_bank: Option<(usize, usize)>,
    /// The global bank the layer builds, for a KV source on this device.
    pub builds_bank: Option<(usize, usize)>,
    /// Record the mean of the copies entering the layer (a drafter target).
    pub tap: bool,
}

/// A pooled source's open entries per slot: the `(kv, gate score)` of the positions
/// computed since an entry last closed, kept for the last `cap` positions so the entry
/// open after any accepted length of a speculative step is still there. Fixed buffers,
/// overwritten in place and scrubbed by a slot reset.
#[derive(Debug)]
pub(crate) struct OpenEntries {
    cap: usize,
    width: usize,
    /// `[slot][source][pos % cap]`.
    tags: Vec<Vec<Vec<u32>>>,
    /// `[slot][source][pos % cap]`: `kv` then `score`.
    values: Vec<Vec<Vec<Vec<f32>>>>,
}

impl OpenEntries {
    pub fn new(slots: usize, sources: usize, cap: usize, width: usize) -> Self {
        Self {
            cap,
            width,
            tags: vec![vec![vec![0; cap]; sources]; slots],
            values: vec![vec![vec![vec![0.0; 2 * width]; cap]; sources]; slots],
        }
    }

    fn put(&mut self, slot: Slot, source: usize, pos: u32, kv: &[f32], score: &[f32]) {
        let at = pos as usize % self.cap;
        let buf = &mut self.values[slot as usize][source][at];
        buf[..self.width].copy_from_slice(kv);
        buf[self.width..].copy_from_slice(score);
        self.tags[slot as usize][source][at] = pos + 1;
    }

    fn get(&self, slot: Slot, source: usize, pos: u32) -> (Vec<f32>, Vec<f32>) {
        let at = pos as usize % self.cap;
        assert_eq!(
            self.tags[slot as usize][source][at],
            pos + 1,
            "slot {slot}: the open global entry's state at position {pos} is not held"
        );
        let buf = &self.values[slot as usize][source][at];
        (buf[..self.width].to_vec(), buf[self.width..].to_vec())
    }

    pub fn scrub(&mut self, slot: Slot) {
        for source in &mut self.tags[slot as usize] {
            source.fill(0);
        }
        for source in &mut self.values[slot as usize] {
            for buf in source {
                buf.fill(0.0);
            }
        }
    }

    pub fn is_zero(&self, slot: Slot) -> bool {
        self.tags[slot as usize].iter().flatten().all(|t| *t == 0)
            && self.values[slot as usize]
                .iter()
                .flatten()
                .flatten()
                .all(|v| v.to_bits() == 0)
    }
}

/// A position's window KV, packed: projected, normed, rotated, FP8.
pub(crate) fn window_cell(
    c: &DsConfig,
    w: &DsAttentionWeights,
    u: &[f32],
    cis: &[(f32, f32)],
) -> Vec<u8> {
    let mut kv = vec![0.0; w.wkv.rows];
    lin8(u, &w.wkv, &mut kv);
    let mut kv = rms(&kv, &w.kv_norm, c.norm_eps);
    rotate(&mut kv, cis);
    let mut cell = vec![0u8; Fp8::bytes(kv.len())];
    Fp8::pack(&kv, &mut cell);
    cell
}

/// Bytes of a global entry: the KV, then the index key.
pub(crate) fn bank_cell_bytes(c: &DsConfig) -> usize {
    Fp4E4m3::bytes(c.attention.head_dim) + Fp4Pow2::bytes(c.indexer.head_dim)
}

/// A global entry from its latent, packed: the index key from the un-rotated latent
/// (projected, normed, rotated, FP4 with power-of-two scales), and the latent itself
/// rotated (FP4 with E4M3 scales). `cis` is the rotation at the entry's first position.
pub(crate) fn bank_cell(
    c: &DsConfig,
    cw: &CompressorWeights,
    latent: Vec<f32>,
    cis: &[(f32, f32)],
) -> Vec<u8> {
    let mut key = vec![0.0; c.indexer.head_dim];
    linear_bf16(&latent, &cw.wk, &mut key);
    let mut key = rms(&key, &cw.k_norm, c.norm_eps);
    rotate(&mut key, cis);
    let mut kv = latent;
    rotate(&mut kv, cis);
    let split = Fp4E4m3::bytes(kv.len());
    let mut cell = vec![0u8; bank_cell_bytes(c)];
    Fp4E4m3::pack(&kv, &mut cell[..split]);
    Fp4Pow2::pack(&key, &mut cell[split..]);
    cell
}

/// A per-position (ratio 1) global entry's latent: a BF16 projection, normed.
pub(crate) fn plain_latent(c: &DsConfig, cw: &CompressorWeights, u: &[f32]) -> Vec<f32> {
    let mut kv = vec![0.0; c.attention.head_dim];
    linear_bf16(u, &cw.wkv, &mut kv);
    rms(&kv, &cw.norm, c.norm_eps)
}

pub(crate) fn unpack_window(c: &DsConfig, bytes: &[u8]) -> Vec<f32> {
    let mut kv = vec![0.0; c.attention.head_dim];
    Fp8::unpack(bytes, &mut kv);
    kv
}

fn unpack_bank_kv(c: &DsConfig, bytes: &[u8]) -> Vec<f32> {
    let mut kv = vec![0.0; c.attention.head_dim];
    Fp4E4m3::unpack(bytes, &mut kv);
    kv
}

fn unpack_bank_key(c: &DsConfig, bytes: &[u8]) -> Vec<f32> {
    let mut key = vec![0.0; c.indexer.head_dim];
    Fp4Pow2::unpack(&bytes[Fp4E4m3::bytes(c.attention.head_dim)..], &mut key);
    key
}

struct Prepared {
    x: Vec<f32>,
    mixes: Mixes,
    u: Vec<f32>,
    cell: Vec<u8>,
    mean: Option<Vec<f32>>,
}

struct Attended {
    x: Vec<f32>,
    mixes: Mixes,
    u: Vec<f32>,
    selection: Vec<u32>,
    candidates: Option<Vec<bool>>,
}

fn err(e: eidola_engine_model::Error) -> ExecutorError {
    ExecutorError(e.to_string())
}

/// Runs one layer. `flows[i]` enters as item `i`'s input (for an active item) and
/// leaves as its output (for an emitting item) or `None`. `gathered[i]` is item `i`'s
/// gathered Engram rows, for a layer with an Engram module. `taps[i]` gains the mean of
/// the copies entering the layer, when the plan asks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_layer(
    model: &DsModel,
    plan: &LayerPlan,
    mem: &mut Memory,
    open: &mut OpenEntries,
    items: &[LayerItem],
    flows: &mut [Option<Flow>],
    gathered: Option<&[Vec<f32>]>,
    taps: &mut [Vec<Vec<f32>>],
) -> Result<(), ExecutorError> {
    let c = model.config();
    let (spec, w) = (&c.layers[plan.layer], &model.weights.layers[plan.layer]);
    let rope = model.rope(spec.rope);

    let prepared: Vec<Option<Prepared>> = items
        .par_iter()
        .enumerate()
        .map(|(i, it)| {
            if !it.active {
                return None;
            }
            let f = flows[i]
                .as_ref()
                .unwrap_or_else(|| panic!("layer {}: no input at position {}", plan.layer, it.pos));
            let mut x = f.x.clone();
            if let Some(e) = &w.engram {
                let rows = gathered.expect("an Engram layer is given its gathered rows");
                inject_gathered(e, &rows[i], &mut x, c.hc_mult, c.norm_eps);
            }
            let mean = plan.tap.then(|| stream_mean(&x, c.hc_mult));
            let (mixes, u) = sub_layer_input(c, &w.hc_attn, &x, &f.pre, &w.attn_norm);
            let cell = window_cell(c, &w.attention, &u, &rope.cis(it.pos as usize));
            Some(Prepared {
                x,
                mixes,
                u,
                cell,
                mean,
            })
        })
        .collect();

    // Every write of the step lands before any read: a position attends to the ones
    // before it in the same row.
    for (i, (it, p)) in items.iter().zip(&prepared).enumerate() {
        let Some(p) = p else { continue };
        mem.write(
            it.slot,
            plan.window.0,
            plan.window.1,
            it.pos,
            it.token,
            &p.cell,
        );
        if let Some(mean) = &p.mean {
            taps[i].push(mean.clone());
        }
        let Some((g, source)) = plan.builds_bank else {
            continue;
        };
        let cw = w
            .attention
            .compressor
            .as_ref()
            .expect("a KV source has a compressor");
        let ratio = spec.compress_ratio as u32;
        let latent = match &cw.wgate {
            None => Some(plain_latent(c, cw, &p.u)),
            Some(wgate) => {
                let hd = c.attention.head_dim;
                let (mut kv, mut score) = (vec![0.0; hd], vec![0.0; hd]);
                linear_f32(&p.u, &cw.wkv, &mut kv);
                linear_f32(&p.u, wgate, &mut score);
                open.put(it.slot, source, it.pos, &kv, &score);
                (it.pos + 1).is_multiple_of(ratio).then(|| {
                    let group: Vec<_> = (it.pos + 1 - ratio..=it.pos)
                        .map(|q| open.get(it.slot, source, q))
                        .collect();
                    pooled_latent(c, cw, &group)
                })
            }
        };
        if let Some(latent) = latent {
            let cis = rope.cis((it.pos + 1 - ratio) as usize);
            mem.write(
                it.slot,
                g,
                source,
                it.pos,
                it.token,
                &bank_cell(c, cw, latent, &cis),
            );
        }
    }

    let (a, window) = (&c.attention, c.attention.window as u32);
    let mem = &*mem;
    let flows_in = &*flows;
    let attended: Vec<Option<Result<Attended, ExecutorError>>> = items
        .par_iter()
        .enumerate()
        .map(|(i, it)| {
            if !it.emit {
                return None;
            }
            let p = prepared[i].as_ref().unwrap_or_else(|| {
                panic!(
                    "layer {}: output at position {} it did not compute",
                    plan.layer, it.pos
                )
            });
            let f = flows_in[i].as_ref().expect("an active item has an input");
            Some((|| {
                let cis = rope.cis(it.pos as usize);
                let (qr, q) = query(c, &w.attention, &p.u, &cis);
                let mut entries: Vec<Option<Vec<f32>>> = (0..window)
                    .map(|i| {
                        (it.pos + i + 1).checked_sub(window).map(|at| {
                            let bytes = mem.read(it.slot, plan.window.0, plan.window.1, at);
                            unpack_window(c, &bytes)
                        })
                    })
                    .collect();
                let (mut selection, mut candidates) = (Vec::new(), None);
                if let Some((g, layer)) = plan.reads_bank {
                    let ratio = mem.groups[g].ratio;
                    let visible = ((it.pos + 1) / ratio) as usize;
                    let entry_at = |t: usize| t as u32 * ratio + ratio - 1;
                    if let Some(ix) = &w.attention.indexer {
                        if visible > 0 {
                            let (iq, weights) = index_query(c, ix, &qr, &p.u, &cis);
                            let scores: Vec<f32> = (0..visible)
                                .map(|t| {
                                    let bytes = mem.read(it.slot, g, layer, entry_at(t));
                                    index_score(&iq, &unpack_bank_key(c, &bytes), &weights)
                                })
                                .collect();
                            let mut mask = None;
                            if spec.is_candidate_source {
                                let k = c.candidates.as_ref().expect("a candidate source");
                                candidates = Some(candidate_mask(
                                    &scores,
                                    visible,
                                    k.top_blocks,
                                    k.block_size,
                                ));
                            } else if spec.uses_candidates {
                                let pool = f
                                    .candidates
                                    .as_ref()
                                    .filter(|p| Some(p.0) == spec.kv_source)
                                    .ok_or_else(|| {
                                        ExecutorError(format!(
                                            "layer {} selects inside candidate blocks its \
                                             input does not carry",
                                            plan.layer
                                        ))
                                    })?;
                                mask = Some(&pool.1);
                            }
                            let allowed =
                                (0..visible).filter(|t| mask.is_none_or(|m: &Vec<bool>| m[*t]));
                            let mut chosen = top_k(&scores, allowed, c.indexer.top_k);
                            chosen.sort_unstable();
                            selection = chosen.into_iter().map(|t| t as u32).collect();
                        }
                    } else {
                        selection = f
                            .selection
                            .as_ref()
                            .filter(|s| Some(s.0) == spec.kv_source)
                            .ok_or_else(|| {
                                ExecutorError(format!(
                                    "layer {} reuses a top-k selection its input does not carry",
                                    plan.layer
                                ))
                            })?
                            .1
                            .clone();
                    }
                    for &t in &selection {
                        let bytes = mem.read(it.slot, g, layer, entry_at(t as usize));
                        entries.push(Some(unpack_bank_kv(c, &bytes)));
                    }
                }
                let slots: Vec<Option<&[f32]>> = entries.iter().map(|e| e.as_deref()).collect();
                let y = attention_output(c, &w.attention, &q, &slots, &cis);
                let mut x = p.x.clone();
                mhc::expand(&mut x, &y, &p.mixes);
                let (mixes, u) = sub_layer_input(c, &w.hc_ffn, &x, &p.mixes.pre, &w.ffn_norm);
                Ok(Attended {
                    x,
                    mixes,
                    u,
                    selection,
                    candidates,
                })
            })())
        })
        .collect();
    let _ = a;

    let mut done = Vec::with_capacity(items.len());
    for t in attended {
        done.push(t.transpose()?);
    }
    let emitting: Vec<usize> = (0..items.len()).filter(|i| done[*i].is_some()).collect();
    let inputs: Vec<&[f32]> = emitting
        .iter()
        .map(|i| done[*i].as_ref().unwrap().u.as_slice())
        .collect();
    let ffn = moe_forward(&model.weights, &w.moe, &c.moe, &inputs).map_err(err)?;
    let mut row = 0;
    for (i, t) in done.into_iter().enumerate() {
        let Some(t) = t else {
            flows[i] = None;
            continue;
        };
        let mut x = t.x;
        mhc::expand(&mut x, ffn.row(row), &t.mixes);
        row += 1;
        let before = flows[i].take().expect("an emitting item had an input");
        let (selection, candidates) = match spec.kv_source {
            Some(src) => (
                Some((src, t.selection)),
                if spec.is_candidate_source {
                    Some((src, t.candidates.unwrap_or_default()))
                } else {
                    before.candidates.filter(|p| p.0 == src)
                },
            ),
            None => (before.selection, before.candidates),
        };
        flows[i] = Some(Flow {
            x,
            pre: t.mixes.pre,
            selection,
            candidates,
        });
    }
    Ok(())
}
