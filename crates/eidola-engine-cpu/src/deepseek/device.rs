//! The two logical devices and what crosses between them.
//!
//! Device 0 ([`Encoder`]) holds the embedding, the layers before the split with their
//! KV (the pooled global banks and the encoder's windows), the first Engram table, and
//! the first decoder layer's bank stub (its attention-input norm, compressor and index
//! key projection). Device 1 ([`Decoder`]) holds the layers from the split on with
//! their KV (that layer's per-position bank, the chain's windows, the drafter's
//! context), the head, the drafter with its own copy of the embedding, and the later
//! Engram tables. Each struct owns only its own pools and reaches only its own layers,
//! so neither can read the other's memory; what one needs of the other is a value of
//! [`ToEncoder`] or [`ToDecoder`], built once per step and checked against the set the
//! step declares.

use std::sync::Arc;

use eidola_engine::executor::{ExecutorError, Slot};
use eidola_engine::spec::ModelSpec;
use eidola_engine_model::deepseek::config::RopeKind;
use eidola_engine_model::deepseek::drafter::{DraftCore, ring_order};
use eidola_engine_model::deepseek::engram::gather;
use eidola_engine_model::deepseek::forward::{DsModel, Rows, sub_layer_input};
use eidola_engine_model::deepseek::linear::lin8;
use eidola_engine_model::deepseek::mhc;
use eidola_engine_model::deepseek::weights::rms;
use eidola_engine_model::tensor::Matrix;

use super::DsCpuExecutorConfig;
use super::layer::{
    Flow, LayerItem, LayerPlan, OpenEntries, bank_cell, bank_cell_bytes, plain_latent, run_layer,
    unpack_window, window_cell,
};
use super::memory::{GroupMem, Memory};
use super::packed::Fp8;
use super::pool::BytePool;

/// The model's shape as the executor splits it.
#[derive(Clone, Debug)]
pub(crate) struct Shape {
    /// The first decoder layer: a per-position KV source.
    pub split: usize,
    pub layers: usize,
    /// The encoder's KV-source layers, ascending.
    pub sources: Vec<usize>,
    /// Their pooling ratio.
    pub ratio: u32,
    /// Token ids before a position its Engram lookups read.
    pub history: usize,
    pub drafter_blocks: usize,
    /// The drafter's target layers on each side of the split, ascending.
    pub encoder_targets: Vec<usize>,
    pub decoder_targets: Vec<usize>,
    pub drafting: bool,
    /// Per Engram layer of the config: the device holding its table.
    pub engram_device: Vec<u32>,
}

impl Shape {
    pub fn of(model: &DsModel, draft_width: u32) -> Self {
        let c = model.config();
        let split = c
            .layers
            .iter()
            .position(|l| l.is_kv_source && l.compress_ratio == 1)
            .expect("a model with a per-position global KV source to split before");
        model
            .check_split(split)
            .unwrap_or_else(|e| panic!("the model cannot be split before layer {split}: {e}"));
        let stub = &model.weights.layers[split];
        assert!(
            stub.engram.is_none()
                && stub
                    .attention
                    .compressor
                    .as_ref()
                    .is_some_and(|cw| cw.wgate.is_none()),
            "the first decoder layer is an ungated KV source without an Engram module"
        );
        let sources: Vec<usize> = (0..split).filter(|l| c.layers[*l].is_kv_source).collect();
        assert!(!sources.is_empty(), "the encoder has a pooled KV source");
        let ratio = c.layers[sources[0]].compress_ratio as u32;
        for l in &c.layers[..split] {
            assert!(
                !l.is_kv_source || l.compress_ratio as u32 == ratio,
                "the encoder's KV sources pool at one ratio"
            );
            assert!(l.kv_source.is_none_or(|s| sources.contains(&s)));
        }
        for (i, l) in c.layers.iter().enumerate().skip(split) {
            assert!(
                l.engram.is_none() && l.kv_source.is_none_or(|s| s == split),
                "decoder layer {i} reads only the split layer's bank and no Engram table"
            );
        }
        let drafting = draft_width > 0;
        let (mut blocks, mut targets) = (0, Vec::new());
        if drafting {
            let (d, w) = match (&c.drafter, &model.weights.drafter) {
                (Some(d), Some(w)) => (d, w),
                _ => panic!("drafts were asked of a model with no drafter loaded"),
            };
            assert!(
                draft_width as usize <= d.slots,
                "draft width {draft_width} exceeds the drafter's {} slots",
                d.slots
            );
            assert!(d.target_layers.is_sorted(), "drafter target layers ascend");
            blocks = w.blocks.len();
            targets = d.target_layers.clone();
        }
        let engram = c.engram.as_ref();
        Self {
            split,
            layers: c.layers.len(),
            sources,
            ratio,
            history: engram.map_or(0, |e| e.max_ngram - 1),
            drafter_blocks: blocks,
            encoder_targets: targets.iter().copied().filter(|t| *t < split).collect(),
            decoder_targets: targets.iter().copied().filter(|t| *t >= split).collect(),
            drafting,
            // The first table beside the encoder; the others beside the decoder, whose
            // device has the room. Only gathered rows cross.
            engram_device: (0..engram.map_or(0, |e| e.layers.len()))
                .map(|slot| u32::from(slot > 0))
                .collect(),
        }
    }

    pub fn depth(&self) -> usize {
        self.layers - self.split
    }
}

/// One position of a step.
#[derive(Clone, Debug)]
pub(crate) struct Item {
    pub row: usize,
    pub slot: Slot,
    pub pos: u32,
    pub token: u32,
    /// The token at `pos` and the ids before it an Engram lookup reads, oldest first.
    pub lookback: Vec<u32>,
}

/// What the decoder needs of a position beyond its bank entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Need {
    /// The chain's bottom layer computes it: its boundary row crosses.
    pub boundary: bool,
    /// The chain's top layer computes it: the drafter's context is written there.
    pub taps: bool,
}

/// Device 1 → device 0: the Engram rows gathered from the tables device 1 holds.
#[derive(Debug, Default)]
pub struct ToEncoder {
    /// Per Engram layer of the config, per position of the step: the gathered rows
    /// (empty for a table device 0 holds itself).
    pub gathered: Vec<Vec<Vec<f32>>>,
}

/// One global entry of the first decoder layer's bank, as stored: the packed KV, then
/// the packed index key.
#[derive(Debug)]
pub struct BankEntry {
    pub slot: Slot,
    pub pos: u32,
    pub token: u32,
    pub bytes: Vec<u8>,
}

/// The state entering the first decoder layer at one position.
#[derive(Debug)]
pub struct BoundaryRow {
    pub slot: Slot,
    pub pos: u32,
    /// The residual copies.
    pub streams: Vec<f32>,
    /// The collapse weights the next sub-layer applies.
    pub pre_mix: Vec<f32>,
    /// The means entering the drafter's target layers on device 0, where the drafter's
    /// context is written and the model has such layers (the released model has none).
    pub taps: Option<Vec<f32>>,
}

/// Device 0 → device 1.
#[derive(Debug, Default)]
pub struct ToDecoder {
    /// A bank entry for every position of the step.
    pub entries: Vec<BankEntry>,
    /// A boundary row for every position the chain's bottom layer computes.
    pub rows: Vec<BoundaryRow>,
    encoder_taps: bool,
}

impl ToDecoder {
    /// Panics unless this is exactly what the step declares: an entry per position, a
    /// boundary row per position the chain computes, taps where the drafter's context
    /// is written.
    pub(crate) fn check(&self, items: &[Item], needs: &[Need]) {
        let at = |it: &Item| (it.slot, it.pos);
        let entries: Vec<_> = self.entries.iter().map(|e| (e.slot, e.pos)).collect();
        assert_eq!(entries, items.iter().map(at).collect::<Vec<_>>());
        let want: Vec<_> = items
            .iter()
            .zip(needs)
            .filter(|(_, n)| n.boundary)
            .map(|(it, n)| (it.slot, it.pos, n.taps && self.encoder_taps))
            .collect();
        let rows: Vec<_> = self
            .rows
            .iter()
            .map(|r| (r.slot, r.pos, r.taps.is_some()))
            .collect();
        assert_eq!(rows, want, "the boundary rows are those the chain computes");
    }
}

/// What crossed between the devices in one step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepTransfer {
    /// Device 1 → 0: the drafted token ids, per drafting row's slot.
    pub drafts: Vec<(Slot, Vec<u32>)>,
    /// Device 1 → 0: gathered Engram rows, as `(slot, position, Engram layer)`.
    pub gathered: Vec<(Slot, u32, usize)>,
    /// Device 0 → 1: bank entries, as `(slot, position)`.
    pub bank_entries: Vec<(Slot, u32)>,
    /// Bytes of one bank entry.
    pub bank_entry_bytes: usize,
    /// Device 0 → 1: boundary rows, as `(slot, position)`.
    pub boundary_rows: Vec<(Slot, u32)>,
    /// Device 0 → 1: boundary rows that carried drafter taps.
    pub taps: Vec<(Slot, u32)>,
}

impl StepTransfer {
    pub(crate) fn note_to_encoder(&mut self, items: &[Item], to: &ToEncoder) {
        for (layer, rows) in to.gathered.iter().enumerate() {
            if !rows.is_empty() {
                self.gathered
                    .extend(items.iter().map(|it| (it.slot, it.pos, layer)));
            }
        }
    }

    pub(crate) fn note_to_decoder(&mut self, to: &ToDecoder) {
        self.bank_entries = to.entries.iter().map(|e| (e.slot, e.pos)).collect();
        self.bank_entry_bytes = to.entries.first().map_or(0, |e| e.bytes.len());
        self.boundary_rows = to.rows.iter().map(|r| (r.slot, r.pos)).collect();
        self.taps = to
            .rows
            .iter()
            .filter(|r| r.taps.is_some())
            .map(|r| (r.slot, r.pos))
            .collect();
    }
}

fn err(e: eidola_engine_model::Error) -> ExecutorError {
    ExecutorError(e.to_string())
}

fn group_mem(
    spec: &ModelSpec,
    cfg: &DsCpuExecutorConfig,
    global: u32,
    cell_bytes: usize,
) -> GroupMem {
    let g = &spec.kv_groups[global as usize];
    let cells = (spec.block_size / g.ratio * g.num_layers) as usize;
    let pool = BytePool::new(cell_bytes, cells, cfg.num_blocks as usize);
    let host = (cfg.host_blocks > 0).then(|| {
        let mut host = pool.like(cfg.host_blocks as usize);
        host.lock().unwrap_or_else(|e| {
            panic!(
                "the host tier's KV ({} bytes in group {}) cannot be locked into memory: {e}",
                host.block_bytes() * host.num_blocks(),
                g.name
            )
        });
        host
    });
    GroupMem::new(
        global,
        g.ratio,
        pool,
        host,
        cfg.num_state_slots as usize,
        spec.max_blocks_per_seq() as usize,
    )
}

/// Engram rows of the tables `device` holds, per Engram layer and item.
fn gather_for(
    model: &DsModel,
    shape: &Shape,
    device: u32,
    items: &[Item],
) -> Result<Vec<Vec<Vec<f32>>>, ExecutorError> {
    let c = model.config();
    (0..shape.engram_device.len())
        .map(|slot| {
            if shape.engram_device[slot] != device {
                return Ok(Vec::new());
            }
            // A truncated model may have dropped the layer.
            let Some(layer) = c.layers.iter().position(|l| l.engram == Some(slot)) else {
                return Ok(Vec::new());
            };
            let Some(e) = &model.weights.layers[layer].engram else {
                return Ok(Vec::new());
            };
            items
                .iter()
                .map(|it| {
                    let rows = model
                        .engram_rows(slot, &it.lookback, it.pos as usize)
                        .map_err(err)?;
                    gather(e.table.as_ref(), &rows).map_err(err)
                })
                .collect()
        })
        .collect()
}

/// Per position: the mean entering each drafter target layer run so far.
type Taps = Vec<Vec<Vec<f32>>>;

const BANK: usize = 0;
const WINDOW: usize = 1;
const DRAFTER: usize = 2;

/// Device 0. Its model handle is used only for the layers before the split, the
/// embedding, the split layer's bank stub, and the Engram tables assigned to it.
pub(crate) struct Encoder {
    model: Arc<DsModel>,
    shape: Shape,
    pub mem: Memory,
    /// The pooled sources' open entries: this device's per-slot model state.
    pub open: OpenEntries,
}

impl Encoder {
    pub fn new(
        model: Arc<DsModel>,
        shape: &Shape,
        spec: &ModelSpec,
        cfg: &DsCpuExecutorConfig,
    ) -> Self {
        let c = model.config();
        let hd = c.attention.head_dim;
        let mem = Memory {
            block_size: spec.block_size,
            groups: vec![
                group_mem(spec, cfg, 0, bank_cell_bytes(c)),
                group_mem(spec, cfg, 1, Fp8::bytes(hd)),
            ],
        };
        Self {
            open: OpenEntries::new(
                cfg.num_state_slots as usize,
                shape.sources.len(),
                cfg.draft_width as usize + shape.ratio as usize + 1,
                hd,
            ),
            shape: shape.clone(),
            mem,
            model,
        }
    }

    /// The encoder over the step's positions: every layer before the split at every
    /// position, then the bank stub. Returns what crosses to device 1.
    pub fn forward(
        &mut self,
        items: &[Item],
        needs: &[Need],
        from_decoder: &ToEncoder,
    ) -> Result<ToDecoder, ExecutorError> {
        let (flows, taps) = self.layers(items, from_decoder)?;
        let model = self.model.clone();
        let c = model.config();
        // The stub: the split layer's attention input, compressed to its bank entry.
        let (spec, w) = (
            &c.layers[self.shape.split],
            &model.weights.layers[self.shape.split],
        );
        let cw = w
            .attention
            .compressor
            .as_ref()
            .expect("checked at construction");
        let rope = model.rope(spec.rope);
        let encoder_taps = self.shape.drafting && !self.shape.encoder_targets.is_empty();
        let mut out = ToDecoder {
            encoder_taps,
            ..ToDecoder::default()
        };
        for (i, (it, f)) in items.iter().zip(flows).enumerate() {
            let u = sub_layer_input(c, &w.hc_attn, &f.x, &f.pre, &w.attn_norm).1;
            let cell = bank_cell(c, cw, plain_latent(c, cw, &u), &rope.cis(it.pos as usize));
            out.entries.push(BankEntry {
                slot: it.slot,
                pos: it.pos,
                token: it.token,
                bytes: cell,
            });
            if needs[i].boundary {
                out.rows.push(BoundaryRow {
                    slot: it.slot,
                    pos: it.pos,
                    streams: f.x,
                    pre_mix: f.pre,
                    taps: (needs[i].taps && encoder_taps).then(|| taps[i].concat()),
                });
            }
        }
        Ok(out)
    }

    /// Every layer before the split at every position of the step: the state leaving
    /// the last of them, and the means entering the drafter's target layers here.
    fn layers(
        &mut self,
        items: &[Item],
        from_decoder: &ToEncoder,
    ) -> Result<(Vec<Flow>, Taps), ExecutorError> {
        let model = self.model.clone();
        let c = model.config();
        let mut flows = Vec::with_capacity(items.len());
        for it in items {
            if it.token as usize >= c.vocab_size || c.image_token_id == Some(it.token) {
                return Err(ExecutorError(format!(
                    "token {} cannot be embedded (vocabulary {}, image token {:?})",
                    it.token, c.vocab_size, c.image_token_id
                )));
            }
            flows.push(Some(Flow {
                x: model.weights.embed.row(it.token as usize).repeat(c.hc_mult),
                pre: mhc::identity_pre(c.hc_mult),
                selection: None,
                candidates: None,
            }));
        }
        let own = gather_for(&model, &self.shape, 0, items)?;
        let layer_items: Vec<LayerItem> = items
            .iter()
            .map(|it| LayerItem {
                slot: it.slot,
                pos: it.pos,
                token: it.token,
                active: true,
                emit: true,
            })
            .collect();
        let mut taps: Vec<Vec<Vec<f32>>> = vec![Vec::new(); items.len()];
        let source = |l: usize| {
            let at = self.shape.sources.iter().position(|s| *s == l);
            (BANK, at.expect("a KV source of the encoder"))
        };
        for l in 0..self.shape.split {
            let spec = &c.layers[l];
            let plan = LayerPlan {
                layer: l,
                window: (WINDOW, l),
                reads_bank: spec.kv_source.map(source),
                builds_bank: spec.is_kv_source.then(|| source(l)),
                tap: self.shape.encoder_targets.contains(&l),
            };
            let gathered = spec.engram.map(|slot| {
                if self.shape.engram_device[slot] == 0 {
                    own[slot].as_slice()
                } else {
                    from_decoder.gathered[slot].as_slice()
                }
            });
            run_layer(
                &model,
                &plan,
                &mut self.mem,
                &mut self.open,
                &layer_items,
                &mut flows,
                gathered,
                &mut taps,
            )?;
        }
        let flows = flows
            .into_iter()
            .map(|f| f.expect("the encoder emits every position"))
            .collect();
        Ok((flows, taps))
    }
}

/// Device 1. Its model handle is used only for the layers from the split on, the head,
/// the drafter, the embedding rows the drafter reads, and the Engram tables assigned
/// to it.
pub(crate) struct Decoder {
    model: Arc<DsModel>,
    shape: Shape,
    pub mem: Memory,
    /// Nothing on this device pools; `run_layer` is handed this empty state.
    no_open: OpenEntries,
    /// Layer-positions the chain computed.
    pub chain_work: u64,
}

impl Decoder {
    pub fn new(
        model: Arc<DsModel>,
        shape: &Shape,
        spec: &ModelSpec,
        cfg: &DsCpuExecutorConfig,
    ) -> Self {
        let c = model.config();
        let window = Fp8::bytes(c.attention.head_dim);
        let mut groups = vec![
            group_mem(spec, cfg, 2, bank_cell_bytes(c)),
            group_mem(spec, cfg, 3, window),
        ];
        if shape.drafting {
            groups.push(group_mem(spec, cfg, 4, window));
        }
        Self {
            no_open: OpenEntries::new(0, 0, 1, 0),
            chain_work: 0,
            mem: Memory {
                block_size: spec.block_size,
                groups,
            },
            shape: shape.clone(),
            model,
        }
    }

    /// The Engram rows of the tables this device holds, for device 0.
    pub fn gather(&self, items: &[Item]) -> Result<ToEncoder, ExecutorError> {
        Ok(ToEncoder {
            gathered: gather_for(&self.model, &self.shape, 1, items)?,
        })
    }

    /// The drafter's pass for a row whose last host position is `p`, holding `token`:
    /// each draft block attends to its context KV at the positions before `p`, in ring
    /// order, and to every draft slot.
    pub fn draft(&self, slot: Slot, p: u32, token: u32) -> Result<DraftCore, ExecutorError> {
        let c = self.model.config();
        let main = p as usize - 1;
        let context: Vec<Vec<Vec<f32>>> = (0..self.shape.drafter_blocks)
            .map(|b| {
                ring_order(main, c.attention.window)
                    .into_iter()
                    .map(|pos| unpack_window(c, &self.mem.read(slot, DRAFTER, b, pos as u32)))
                    .collect()
            })
            .collect();
        let refs: Vec<Vec<&[f32]>> = context
            .iter()
            .map(|b| b.iter().map(|kv| kv.as_slice()).collect())
            .collect();
        self.model.drafter_slots(main, &refs, token).map_err(err)
    }

    /// Draft slot `slot`'s logits given the token before it in the chain.
    pub fn draft_logits(&self, core: &DraftCore, slot: usize, previous: u32) -> Vec<f32> {
        self.model.draft_logits(core, slot, previous)
    }

    /// The tokens at positions `0..n`, read back from the bank's tags.
    pub fn prefix_tokens(&self, slot: Slot, n: u32) -> Vec<u32> {
        (0..n)
            .map(|pos| self.mem.token_at(slot, BANK, pos))
            .collect()
    }

    /// The decoder over the step's positions: the bank entries stored, the chain run
    /// by the row rule (`starts[row][j]` is where chain layer `j` starts in that row),
    /// the drafter's context written where the top layer computes, and the logits of
    /// the positions flagged in `top`.
    pub fn forward(
        &mut self,
        items: &[Item],
        needs: &[Need],
        starts: &[Vec<u32>],
        top: &[bool],
        crossing: ToDecoder,
    ) -> Result<Vec<Option<Vec<f32>>>, ExecutorError> {
        let model = self.model.clone();
        let c = model.config();
        for e in &crossing.entries {
            self.mem.write(e.slot, BANK, 0, e.pos, e.token, &e.bytes);
        }
        let mut flows: Vec<Option<Flow>> = vec![None; items.len()];
        let mut encoder_taps: Vec<Option<Vec<f32>>> = vec![None; items.len()];
        let mut rows = crossing.rows.into_iter();
        for (i, need) in needs.iter().enumerate() {
            if need.boundary {
                let r = rows.next().expect("checked: a row per computed position");
                encoder_taps[i] = r.taps;
                flows[i] = Some(Flow {
                    x: r.streams,
                    pre: r.pre_mix,
                    selection: None,
                    candidates: None,
                });
            }
        }
        let depth = self.shape.depth();
        let mut taps: Vec<Vec<Vec<f32>>> = vec![Vec::new(); items.len()];
        for j in 0..depth {
            let l = self.shape.split + j;
            let layer_items: Vec<LayerItem> = items
                .iter()
                .enumerate()
                .map(|(i, it)| LayerItem {
                    slot: it.slot,
                    pos: it.pos,
                    token: it.token,
                    active: it.pos >= starts[it.row][j],
                    emit: match starts[it.row].get(j + 1) {
                        Some(next) => it.pos >= *next,
                        None => top[i],
                    },
                })
                .collect();
            self.chain_work += layer_items.iter().filter(|it| it.active).count() as u64;
            let plan = LayerPlan {
                layer: l,
                window: (WINDOW, j),
                reads_bank: c.layers[l].kv_source.map(|_| (BANK, 0)),
                builds_bank: None,
                tap: self.shape.decoder_targets.contains(&l),
            };
            run_layer(
                &model,
                &plan,
                &mut self.mem,
                &mut self.no_open,
                &layer_items,
                &mut flows,
                None,
                &mut taps,
            )?;
        }

        if self.shape.drafting {
            let w = model
                .weights
                .drafter
                .as_ref()
                .expect("checked at construction");
            let rope = model.rope(RopeKind::Plain);
            for (i, it) in items.iter().enumerate() {
                if !needs[i].taps {
                    continue;
                }
                let mut hidden = encoder_taps[i].take().unwrap_or_default();
                assert_eq!(taps[i].len(), self.shape.decoder_targets.len());
                hidden.extend(taps[i].concat());
                let mut main_x = vec![0.0; c.dim];
                lin8(&hidden, &w.main_proj, &mut main_x);
                let main_x = rms(&main_x, &w.main_norm, c.norm_eps);
                let cis = rope.cis(it.pos as usize);
                for (b, block) in w.blocks.iter().enumerate() {
                    let cell = window_cell(c, &block.attention, &main_x, &cis);
                    self.mem.write(it.slot, DRAFTER, b, it.pos, it.token, &cell);
                }
            }
        }

        let wanted: Vec<usize> = (0..items.len()).filter(|i| top[*i]).collect();
        let mut out = vec![None; items.len()];
        if wanted.is_empty() {
            return Ok(out);
        }
        let mut streams = Matrix::zeros(wanted.len(), c.hc_mult * c.dim);
        let mut pre_mix = Matrix::zeros(wanted.len(), c.hc_mult);
        for (r, i) in wanted.iter().enumerate() {
            let f = flows[*i]
                .as_ref()
                .expect("the top layer emitted this position");
            streams.row_mut(r).copy_from_slice(&f.x);
            pre_mix.row_mut(r).copy_from_slice(&f.pre);
        }
        let (_, logits) = model.head(
            &Rows {
                start: 0,
                streams,
                pre_mix,
                engram_rows: Vec::new(),
                selection: None,
                candidates: None,
            },
            true,
        );
        for (r, i) in wanted.iter().enumerate() {
            out[*i] = Some(logits.row(r).to_vec());
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use eidola_engine::spec::{AttentionKind, Coverage, KvGroupSpec, KvRole};
    use eidola_engine_model::WeightSet;
    use eidola_engine_model::deepseek::engram::{EngramTable, SparseEngramTable};
    use eidola_engine_model::deepseek::forward::DsForwardOptions;
    use eidola_engine_model::deepseek::{DsLoadOptions, DsWeights, config_of};

    use super::*;

    /// Device 0 over whole layers 0-3 of the released checkpoint: the embedding, the
    /// first Engram module, two window-only layers and the first pooled KV source with
    /// the layer that reuses its selection, every routed expert. The KV goes through
    /// the pools in the released packed sizes (528 bytes of window KV a position, 356 of
    /// a global entry), in chunks, and the state leaving layer 3 equals the model
    /// crate's dense forward bit for bit.
    ///
    /// The truncation holds no layer to split before, so the whole executor cannot run
    /// on it; this is the half that can.
    ///
    ///     EIDOLA_DEEPSEEK_PREFIX_DIR=<dir> cargo test --release -p eidola-engine-cpu \
    ///         --lib -- --ignored --nocapture released_encoder
    #[test]
    #[ignore = "needs a local prefix of the released checkpoint"]
    fn released_encoder_layers_page_exactly() {
        let dir = PathBuf::from(
            std::env::var("EIDOLA_DEEPSEEK_PREFIX_DIR")
                .expect("EIDOLA_DEEPSEEK_PREFIX_DIR names the checkpoint prefix"),
        );
        let open = |name: &str| WeightSet::open_files(&[dir.join(name)]).unwrap();
        let ints = |set: &WeightSet, name: &str| -> Vec<i64> {
            set.get(name).unwrap().to_i64(name).unwrap()
        };
        let pinned = open("golden/reference-pinned.safetensors");
        let tokens: Vec<u32> = ints(&pinned, "tokens").iter().map(|t| *t as u32).collect();
        let token_map = ints(&pinned, "engram.token_map")
            .iter()
            .map(|t| *t as u32)
            .collect();
        let store = Arc::new(WeightSet::open_dir(&dir).unwrap());
        let config = config_of(&store).unwrap().truncated(&[0, 1, 2, 3]).unwrap();
        let rows = open("golden/engram-rows.safetensors");
        let spec = config.engram.clone().unwrap();
        let first = &spec.layers[0];
        let p = format!("layers.{}", first.source_layer);
        let ids: Vec<u64> = ints(&rows, &format!("{p}.rows"))
            .iter()
            .map(|r| *r as u64)
            .collect();
        let table = SparseEngramTable::from_stored(
            first.num_rows,
            spec.head_dim,
            &ids,
            rows.get(&format!("{p}.weight")).unwrap().data,
            rows.get(&format!("{p}.scale")).unwrap().data,
        )
        .unwrap();
        let weights = DsWeights::load(
            store,
            config,
            &DsLoadOptions {
                skip_drafter: true,
                engram_tables: vec![(first.source_layer, Arc::new(table) as Arc<dyn EngramTable>)],
            },
        )
        .unwrap();
        let model = Arc::new(DsModel::new(weights, Some(token_map)).unwrap());
        let c = model.config();
        let n = tokens.len();
        eprintln!("{n} tokens, {} layers, dim {}", c.layers.len(), c.dim);

        let block_size = 32u32;
        let blocks = n as u32 / block_size + 2;
        let group = |attention, layers, ratio| KvGroupSpec {
            name: String::new(),
            role: KvRole::Target,
            attention,
            num_layers: layers,
            num_kv_heads: 1,
            head_dim_qk: c.attention.head_dim as u32,
            head_dim_v: c.attention.head_dim as u32,
            num_blocks: blocks,
            host_blocks: 0,
            ratio,
            device: 0,
            coverage: Coverage::All,
        };
        let window = c.attention.window as u32;
        let spec = ModelSpec {
            vocab_size: c.vocab_size as u32,
            sampleable_vocab_size: c.vocab_size as u32,
            block_size,
            max_model_len: blocks * block_size,
            kv_groups: vec![
                group(AttentionKind::Full, 1, 2),
                group(AttentionKind::Sliding { window }, 4, 1),
            ],
            max_draft_tokens: 0,
            num_state_slots: 1,
            buckets: Vec::new(),
            window_chain: None,
            token_history: 3,
        };
        let cfg = DsCpuExecutorConfig {
            block_size,
            num_blocks: blocks,
            host_blocks: 0,
            num_state_slots: 1,
            max_model_len: blocks * block_size,
            buckets: Vec::new(),
            draft_width: 0,
            sampleable_vocab_size: c.vocab_size as u32,
            record: false,
        };
        let shape = Shape {
            split: 4,
            layers: 4,
            sources: vec![2],
            ratio: 2,
            history: 3,
            drafter_blocks: 0,
            encoder_targets: Vec::new(),
            decoder_targets: Vec::new(),
            drafting: false,
            engram_device: vec![0, 1],
        };
        let mut encoder = Encoder::new(model.clone(), &shape, &spec, &cfg);
        assert_eq!(encoder.mem.groups[WINDOW].pool.block_bytes(), 4 * 32 * 528);
        assert_eq!(encoder.mem.groups[BANK].pool.block_bytes(), 16 * 356);
        for g in [BANK, WINDOW] {
            for index in 0..blocks - 1 {
                encoder.mem.set_table(g, 0, index, index + 1);
            }
        }

        let dense = model
            .forward(
                &mut model.new_state(),
                &tokens,
                &DsForwardOptions {
                    no_logits: true,
                    ..Default::default()
                },
            )
            .unwrap()
            .rows;
        let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let mut at = 0;
        for chunk in [1usize, 37, 64, 5, 129].into_iter().cycle() {
            if at == n {
                break;
            }
            let end = (at + chunk).min(n);
            let items: Vec<Item> = (at..end)
                .map(|pos| Item {
                    row: 0,
                    slot: 0,
                    pos: pos as u32,
                    token: tokens[pos],
                    lookback: tokens[(pos + 1).saturating_sub(4)..=pos].to_vec(),
                })
                .collect();
            let (flows, _) = encoder.layers(&items, &ToEncoder::default()).unwrap();
            for (it, f) in items.iter().zip(&flows) {
                let pos = it.pos as usize;
                assert_eq!(bits(&f.x), bits(dense.streams.row(pos)), "streams at {pos}");
                assert_eq!(
                    bits(&f.pre),
                    bits(dense.pre_mix.row(pos)),
                    "collapse at {pos}"
                );
            }
            at = end;
        }
        eprintln!("{n} positions equal the dense forward after layer 3");
    }
}
