//! One half's forward: a model layer as the kernel launches of `csrc/ds_abi.cuh`, in the
//! order `DsModel::layer_forward` and the CPU reference executor's `run_layer` compose
//! the model crate's functions, over rows in the planner's order ([`super::plan`]).
//!
//! A layer runs its Engram injection, its attention input and its KV writes over the
//! rows it computes (`active`), and everything from the queries on over the rows whose
//! output is consumed (`emit`), both prefixes. Every KV write of the step is queued
//! before any read of the layer. A fake-quantised vector exists only as codes and
//! scales; BF16 between launches is 16 bits.
//!
//! Scratch is allocated once, for the largest step, and overwritten in place.

use eidola_engine::executor::Slot;
use eidola_engine_model::deepseek::DsConfig;
use eidola_engine_model::deepseek::config::{DsMoeSpec, RopeKind};

use super::abi::*;
use super::half::{Buffer, Device, Kind};
use super::kv::{KvStore, Plane};
use super::plan::{Member, PoolPlan};
use super::support::DsShape;
use super::weights::{Fp8, Half, HalfWeights, LayerWeights, Mhc};
use crate::{CudaError, Result, narrow};

/// This half's bank group and window group, as indices into its [`KvStore`].
pub const BANK: usize = 0;
pub const WINDOW: usize = 1;

/// Rows each expert's run starts on a multiple of.
pub const ROW_ALIGN: u32 = 1;

/// The configuration's numbers as the kernels take them.
#[derive(Clone, Debug, PartialEq)]
pub struct Dims {
    pub dim: u32,
    pub copies: u32,
    pub mix_words: u32,
    pub head_dim: u32,
    pub heads: u32,
    pub rope_dim: u32,
    pub q_rank: u32,
    pub o_groups: u32,
    pub o_rank: u32,
    pub index_heads: u32,
    pub index_dim: u32,
    pub inter: u32,
    pub experts: u32,
    pub active: u32,
    pub window: u32,
    pub top_k: u32,
    /// Words of a row's attention slot list: the window, then the selection.
    pub slot_stride: u32,
    pub vocab: u32,
    /// Table rows an Engram lookup gathers, and their total width.
    pub lookups: u32,
    pub engram_width: u32,
    /// Index scores a row can hold, and words of its candidate mask.
    pub score_stride: u32,
    pub mask_words: u32,
    pub candidate_blocks: u32,
    pub candidate_entries: u32,
    pub positions: u32,
    /// The backbone's feed-forward.
    pub moe: MoeDims,
    /// The drafter, when the executor drafts.
    pub draft: Option<DraftDims>,
}

/// A feed-forward's routing numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeDims {
    pub experts: u32,
    pub active: u32,
    pub inter: u32,
    pub normalize: bool,
    pub route_scale: f32,
    pub limit: f32,
}

impl MoeDims {
    fn new(m: &DsMoeSpec) -> Result<MoeDims> {
        Ok(MoeDims {
            experts: narrow(m.n_routed, "experts")?,
            active: narrow(m.n_active, "experts per token")?,
            inter: narrow(m.inter_dim, "expert width")?,
            normalize: m.norm_topk_prob && m.n_active > 1,
            route_scale: m.route_scale,
            limit: m.swiglu_limit,
        })
    }
}

/// The drafter's numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftDims {
    /// Slots a pass computes.
    pub slots: u32,
    /// Drafted tokens per step.
    pub width: u32,
    pub blocks: u32,
    pub moe: MoeDims,
    pub markov_rank: u32,
    /// Target layers on this half, and on the encoder (whose taps come first).
    pub targets: u32,
    pub encoder_targets: u32,
    /// The token of every slot after the first.
    pub noise_token: u32,
}

impl Dims {
    pub fn new(c: &DsConfig, shape: &DsShape, half: Half, max_model_len: u32) -> Result<Dims> {
        let n = |x: usize, what: &str| narrow::<u32, usize>(x, what);
        let (a, ix, m) = (&c.attention, &c.indexer, &c.moe);
        let ratio = match half {
            Half::Encoder => shape.ratio,
            Half::Decoder => 1,
        };
        let score_stride = max_model_len / ratio + 1;
        let (candidate_blocks, candidate_entries) = match &c.candidates {
            Some(k) => (
                n(k.top_blocks, "candidate blocks")?,
                n(k.block_size, "block")?,
            ),
            None => (0, 1),
        };
        let copies = n(c.hc_mult, "copies")?;
        let lookups = n(c.engram.as_ref().map_or(0, |e| e.lookups()), "lookups")?;
        let window = n(a.window, "window")?;
        let top_k = n(ix.top_k, "top k")?;
        Ok(Dims {
            dim: n(c.dim, "dim")?,
            copies,
            mix_words: mix_words(copies),
            head_dim: n(a.head_dim, "head dim")?,
            heads: n(a.n_heads, "heads")?,
            rope_dim: n(a.rope_dim, "rope dim")?,
            q_rank: n(a.q_lora_rank, "query rank")?,
            o_groups: n(a.o_groups, "output groups")?,
            o_rank: n(a.o_lora_rank, "output rank")?,
            index_heads: n(ix.n_heads, "index heads")?,
            index_dim: n(ix.head_dim, "index head dim")?,
            inter: n(m.inter_dim, "expert width")?,
            experts: n(m.n_routed, "experts")?,
            active: n(m.n_active, "experts per token")?,
            window,
            top_k,
            slot_stride: window
                .checked_add(top_k)
                .ok_or_else(|| CudaError::new("slot list width"))?,
            vocab: n(c.vocab_size, "vocabulary")?,
            lookups,
            engram_width: lookups * n(c.engram.as_ref().map_or(0, |e| e.head_dim), "engram")?,
            score_stride,
            mask_words: score_stride.div_ceil(candidate_entries).div_ceil(32),
            candidate_blocks,
            candidate_entries,
            positions: max_model_len,
            moe: MoeDims::new(m)?,
            draft: match (&c.drafter, shape.drafting()) {
                (Some(dr), true) => Some(DraftDims {
                    slots: n(dr.slots, "draft slots")?,
                    width: shape.draft_width,
                    blocks: n(dr.blocks, "draft blocks")?,
                    moe: MoeDims::new(&dr.moe)?,
                    markov_rank: n(dr.markov_rank, "markov rank")?,
                    targets: n(
                        match half {
                            Half::Encoder => shape.encoder_targets.len(),
                            Half::Decoder => dr.target_layers.len(),
                        },
                        "target layers",
                    )?,
                    encoder_targets: n(shape.encoder_targets.len(), "target layers")?,
                    noise_token: dr.noise_token,
                }),
                _ => None,
            },
        })
    }
}

macro_rules! scratch {
    ($($name:ident),* $(,)?) => {
        /// A half's step scratch: one buffer per array a launch reads or writes.
        pub struct Scratch {
            $(pub $name: Buffer,)*
            /// Per Engram layer of the config: the gathered rows' codes and scales.
            pub gathered: Vec<(Buffer, Buffer)>,
        }

        impl Scratch {
            /// Every buffer.
            pub fn buffers(&self) -> Vec<&Buffer> {
                let mut all = vec![$(&self.$name),*];
                for (codes, scales) in &self.gathered {
                    all.extend([codes, scales]);
                }
                all
            }
        }
    };
}

scratch! {
    x, mix_a, mix_b, u, ucodes, uscales, kv, latent, lcodes, lscales, qraw, q, slots, attn,
    ocodes, oscales, z, zcodes, zscales, y, iq, iq_codes, iq_scales, head_dots, scores, mask,
    visible, table_row, positions, tokens, wtargets, proj_kv, proj_score, pooled, key,
    pool_first, pool_second, pool_pos, pool_targets, rlogits, ids, rweights, row_of,
    row_weight, group_end, place_scratch, lay_codes, lay_scales, gate_up, hid_codes, hid_scales, routed,
    sgate, sup, shcodes, shscales, shared, status, eids, ecodes, escales, ekv, bkv_codes,
    bkv_scales, bkey_codes, bkey_scales, logits, probs, srows, stokens,
}

/// Rows of per-slot state behind the projection scratch: an open entry per kept
/// position of each pooled source of each slot.
fn state_rows(slots: u32, sources: usize, cap: u32) -> usize {
    slots as usize * sources * cap as usize
}

impl Scratch {
    /// Bytes of scratch (per-slot state aside) a half needs for steps of `rows`
    /// positions and `samples` sampling rows.
    #[allow(clippy::too_many_arguments)]
    fn new(
        dev: &dyn Device,
        d: &Dims,
        half: Half,
        rows: u32,
        samples: u32,
        sampleable: u32,
        state: usize,
        slots: usize,
    ) -> Result<Scratch> {
        let r = rows as usize;
        let u = |x: u32| x as usize;
        // The widest of the backbone's and the drafter's feed-forwards.
        let widest = |f: fn(&MoeDims) -> u32| f(&d.moe).max(d.draft.map_or(0, |x| f(&x.moe)));
        let (experts, active) = (widest(|m| m.experts), widest(|m| m.active));
        let layout = u(super::plan::layout_rows(rows, active, experts, ROW_ALIGN)
            .ok_or_else(|| CudaError::new("expert layout rows"))?);
        let (dim, hd, q) = (u(d.dim), u(d.head_dim), u(d.q_rank));
        let (qw, rank) = (u(d.heads) * hd, u(d.o_groups * d.o_rank));
        let (iw, inter) = (u(d.index_heads * d.index_dim), u(widest(|m| m.inter)));
        let ew = u(d.engram_width);
        let (enc, dec) = (half == Half::Encoder, half == Half::Decoder);
        let t = if dec { u(samples) } else { 0 };
        let a = |name: &'static str, bytes: usize| Buffer::new(dev, name, Kind::Scratch, bytes);
        let only = |on: bool, bytes: usize| if on { bytes } else { 0 };
        Ok(Scratch {
            x: a("x", r * u(d.copies) * dim * 2)?,
            mix_a: a("mix_a", r * u(d.mix_words) * 4)?,
            mix_b: a("mix_b", r * u(d.mix_words) * 4)?,
            u: a("u", r * dim * 2)?,
            ucodes: a("ucodes", r * dim)?,
            uscales: a("uscales", r * dim / 32)?,
            kv: a("kv", r * hd * 2)?,
            latent: a("latent", r * q * 2)?,
            lcodes: a("lcodes", r * q)?,
            lscales: a("lscales", r * q / 32)?,
            qraw: a("qraw", r * qw * 2)?,
            q: a("q", r * qw * 2)?,
            slots: a("slots", r * u(d.slot_stride) * 4)?,
            attn: a("attn", r * qw * 2)?,
            ocodes: a("ocodes", r * qw)?,
            oscales: a("oscales", r * qw / 32)?,
            z: a("z", r * rank * 2)?,
            zcodes: a("zcodes", r * rank)?,
            zscales: a("zscales", r * rank / 32)?,
            y: a("y", r * dim * 2)?,
            iq: a("iq", r * iw * 2)?,
            iq_codes: a("iq_codes", r * iw / 2)?,
            iq_scales: a("iq_scales", r * iw / 32)?,
            head_dots: a("head_dots", r * u(d.index_heads) * 2)?,
            scores: a("scores", r * u(d.score_stride) * 2)?,
            mask: a("mask", r * u(d.mask_words) * 4)?,
            visible: a("visible", r * 4)?,
            table_row: a("table_row", r * 4)?,
            positions: a("positions", r * 4)?,
            tokens: a("tokens", r * 4)?,
            wtargets: a("wtargets", r * 4)?,
            proj_kv: Buffer::new(dev, "proj_kv", Kind::State, only(enc, (r + state) * hd * 4))?,
            proj_score: Buffer::new(
                dev,
                "proj_score",
                Kind::State,
                only(enc, (r + state) * hd * 4),
            )?,
            pooled: a("pooled", only(enc, r * hd * 2))?,
            key: a("key", only(enc, r * u(d.index_dim) * 2))?,
            pool_first: a("pool_first", only(enc, r * 4))?,
            pool_second: a("pool_second", only(enc, r * 4))?,
            pool_pos: a("pool_pos", only(enc, r * 4))?,
            pool_targets: a("pool_targets", only(enc, r * 4))?,
            rlogits: a("rlogits", r * u(experts) * 4)?,
            ids: a("ids", r * u(active) * 4)?,
            rweights: a("rweights", r * u(active) * 4)?,
            row_of: a("row_of", r * u(active) * 4)?,
            row_weight: a("row_weight", layout * 4)?,
            group_end: a("group_end", u(experts) * 4)?,
            place_scratch: a("place_scratch", u(experts) * 4)?,
            lay_codes: a("lay_codes", layout * dim)?,
            lay_scales: a("lay_scales", layout * dim / 32)?,
            gate_up: a("gate_up", layout * 2 * inter * 2)?,
            hid_codes: a("hid_codes", layout * inter)?,
            hid_scales: a("hid_scales", layout * inter / 32)?,
            routed: a("routed", layout * dim * 2)?,
            sgate: a("sgate", r * inter * 2)?,
            sup: a("sup", r * inter * 2)?,
            shcodes: a("shcodes", r * inter)?,
            shscales: a("shscales", r * inter / 32)?,
            shared: a("shared", r * dim * 2)?,
            status: a("status", 8)?,
            eids: a("eids", r * u(d.lookups) * 4)?,
            ecodes: a("ecodes", only(enc, r * ew))?,
            escales: a("escales", only(enc, r * ew / 32))?,
            ekv: a("ekv", only(enc, r * (u(d.copies) + 1) * dim * 2))?,
            bkv_codes: a("bkv_codes", only(enc, r * hd / 2))?,
            bkv_scales: a("bkv_scales", only(enc, r * hd / 16))?,
            bkey_codes: a("bkey_codes", only(enc, r * u(d.index_dim) / 2))?,
            bkey_scales: a("bkey_scales", only(enc, r * u(d.index_dim) / 32))?,
            logits: a("logits", t * u(d.vocab) * 4)?,
            probs: a("probs", t * u(sampleable) * 8)?,
            srows: a("srows", t * 32)?,
            stokens: a("stokens", t * 4)?,
            gathered: (0..slots)
                .map(|_| Ok((a("gcodes", r * ew)?, a("gscales", r * ew / 32)?)))
                .collect::<Result<_>>()?,
        })
    }
}

/// A half's memory by kind, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryBytes {
    pub weights: u64,
    pub kv: u64,
    pub tables: u64,
    /// Per-slot model state: the pooled sources' open entries.
    pub state: u64,
    pub scratch: u64,
}

impl MemoryBytes {
    pub fn total(&self) -> u64 {
        self.weights + self.kv + self.tables + self.state + self.scratch
    }
}

/// Where a rotation's packed rows go.
pub enum Dst<'a> {
    /// Entries of a pool plane, named by a `[rows]` array of targets.
    Plane(KvPlane, &'a Buffer),
    /// Dense codes and scales, a row each.
    Dense(&'a Buffer, &'a Buffer),
}

/// How a layer runs in one step.
pub struct LayerRun<'a> {
    /// Rows the layer computes.
    pub active: u32,
    /// Rows whose output is consumed.
    pub emit: u32,
    /// The layer's place in its window group.
    pub window_layer: u32,
    /// The bank layer it reads.
    pub bank_layer: Option<u32>,
    /// For a pooled source: its bank layer and the step's pooling.
    pub build: Option<(u32, &'a PoolPlan)>,
    /// For a drafter target layer: its place among the half's taps (the mean of the
    /// copies entering the layer is kept there for the rows the layer computes).
    pub tap: Option<u32>,
}

/// One half: its device, weights, KV, scratch and per-slot state.
pub struct HalfNet {
    pub half: Half,
    pub dev: Box<dyn Device>,
    pub w: HalfWeights,
    pub kv: KvStore,
    pub s: Scratch,
    pub d: Dims,
    c: DsConfig,
    /// Rows the scratch holds.
    pub max_rows: u32,
    sources: usize,
    /// Positions of open-entry state kept per source and slot.
    pub cap: u32,
    /// The step's positions in device row order, for the position tags.
    pub items: Vec<(Slot, u32)>,
    /// `[slot][source][pos % cap]`: the position held, plus one.
    open_tags: Vec<u32>,
    /// The drafter's scratch, when the executor drafts.
    pub ds: Option<DraftScratch>,
}

/// The scratch drafting adds to a half. The encoder keeps only the taps of its target
/// layers.
pub struct DraftScratch {
    /// BF16 `[rows, targets · dim]`: the means entering the target layers.
    pub hidden: Buffer,
    /// The taps quantised for the projection.
    pub hcodes: Buffer,
    pub hscales: Buffer,
    /// BF16 `[rows, dim]`: the projected state the draft blocks' context is written from.
    pub mainx: Buffer,
    /// `[rows]`: each row's entry in the drafter's group.
    pub targets: Buffer,
    /// A pass's own KV: FP8 codes `[rows, head_dim]`, then scales `[rows, head_dim / 32]`.
    pub kv: Buffer,
    /// A pass's slot lists, `[rows, window + slots]`.
    pub slots: Buffer,
    /// f32 `[rows, vocab]`: the pass's base logits.
    pub base: Buffer,
    /// f32 `[drafting rows, vocab]`: one slot's logits, its Markov term added.
    pub logits: Buffer,
    /// f64 `[rows, sampleable]`: each drafted slot's distribution.
    pub probs: Buffer,
    /// `[rows]`: the drafted tokens, slot-major.
    pub tokens: Buffer,
    /// `[drafting rows]`: the token before slot 0 in the chain.
    pub previous: Buffer,
    /// `[drafting rows]` sample rows of one slot, and of the acceptance.
    pub rows: Buffer,
    pub accept_rows: Buffer,
    pub target_row: Buffer,
    pub draft_row: Buffer,
    pub num_drafts: Buffer,
    /// f64 `[drafting rows, sampleable]`.
    pub residual: Buffer,
    /// `[drafting rows, width + 1]` produced tokens, and their counts.
    pub out: Buffer,
    pub counts: Buffer,
}

impl DraftScratch {
    fn new(
        dev: &dyn Device,
        d: &Dims,
        draft: &DraftDims,
        half: Half,
        rows: u32,
        seqs: u32,
        sampleable: u32,
    ) -> Result<DraftScratch> {
        let u = |x: u32| x as usize;
        let (r, dec) = (u(rows), half == Half::Decoder);
        // A pass computes every slot of every drafting row.
        let pass = if dec { u(seqs) * u(draft.slots) } else { 0 };
        let seqs = if dec { u(seqs) } else { 0 };
        let taps = u(draft.targets) * u(d.dim);
        let a = |name: &'static str, bytes: usize| Buffer::new(dev, name, Kind::Scratch, bytes);
        let only = |bytes: usize| if dec { bytes } else { 0 };
        let (hd, n) = (u(d.head_dim), u(sampleable));
        Ok(DraftScratch {
            hidden: a("draft hidden", r * taps * 2)?,
            hcodes: a("draft hcodes", only(r * taps))?,
            hscales: a("draft hscales", only(r * taps / 32))?,
            mainx: a("draft mainx", only(r * u(d.dim) * 2))?,
            targets: a("draft targets", only(r * 4))?,
            kv: a("draft kv", pass * (hd + hd / 32))?,
            slots: a("draft slots", pass * u(d.window + draft.slots) * 4)?,
            base: a("draft base", pass * u(d.vocab) * 4)?,
            logits: a("draft logits", seqs * u(d.vocab) * 4)?,
            probs: a("draft probs", pass * n * 8)?,
            tokens: a("draft tokens", pass * 4)?,
            previous: a("draft previous", seqs * 4)?,
            rows: a("draft rows", seqs * 32)?,
            accept_rows: a("accept rows", seqs * 32)?,
            target_row: a("accept target", seqs * 4)?,
            draft_row: a("accept draft", seqs * 4)?,
            num_drafts: a("accept counts in", seqs * 4)?,
            residual: a("accept residual", seqs * n * 8)?,
            out: a("accept out", seqs * u(draft.width + 1) * 4)?,
            counts: a("accept counts", seqs * 4)?,
        })
    }

    fn buffers(&self) -> Vec<&Buffer> {
        vec![
            &self.hidden,
            &self.hcodes,
            &self.hscales,
            &self.mainx,
            &self.targets,
            &self.kv,
            &self.slots,
            &self.base,
            &self.logits,
            &self.probs,
            &self.tokens,
            &self.previous,
            &self.rows,
            &self.accept_rows,
            &self.target_row,
            &self.draft_row,
            &self.num_drafts,
            &self.residual,
            &self.out,
            &self.counts,
        ]
    }
}

impl HalfNet {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        half: Half,
        dev: Box<dyn Device>,
        w: HalfWeights,
        kv: KvStore,
        c: &DsConfig,
        shape: &DsShape,
        max_model_len: u32,
        max_rows: u32,
        samples: u32,
        seqs: u32,
        sampleable: u32,
        cap: u32,
    ) -> Result<HalfNet> {
        let d = Dims::new(c, shape, half, max_model_len)?;
        let ds =
            match &d.draft {
                Some(draft) if draft.targets > 0 || half == Half::Decoder => Some(
                    DraftScratch::new(dev.as_ref(), &d, draft, half, max_rows, seqs, sampleable)?,
                ),
                _ => None,
            };
        let sources = match half {
            Half::Encoder => shape.sources.len(),
            Half::Decoder => 0,
        };
        let state = state_rows(kv.slots, sources, cap);
        let s = Scratch::new(
            dev.as_ref(),
            &d,
            half,
            max_rows,
            samples,
            sampleable,
            state,
            shape.engram_device.len(),
        )?;
        Ok(HalfNet {
            half,
            w,
            s,
            d,
            c: c.clone(),
            max_rows,
            sources,
            cap,
            open_tags: vec![0; state],
            items: Vec::new(),
            ds,
            kv,
            dev,
        })
    }

    /// Every scratch buffer of the half.
    pub fn buffers(&self) -> Vec<&Buffer> {
        let mut all = self.s.buffers();
        if let Some(ds) = &self.ds {
            all.extend(ds.buffers());
        }
        all
    }

    /// The drafter's scratch and numbers; an error when the half does not draft.
    pub fn drafting(&self) -> Result<(&DraftScratch, DraftDims)> {
        match (&self.ds, self.d.draft) {
            (Some(ds), Some(draft)) => Ok((ds, draft)),
            _ => Err(CudaError::new("this half does not draft")),
        }
    }

    /// The half's memory by kind.
    pub fn memory(&self) -> MemoryBytes {
        let mut m = MemoryBytes {
            weights: self.w.bytes.total(),
            ..MemoryBytes::default()
        };
        for g in &self.kv.groups {
            m.kv += g.pool.bytes as u64;
            m.tables += g.tables.bytes as u64;
        }
        let state = (self.open_tags.len() * self.d.head_dim as usize * 4) as u64;
        for b in self.buffers() {
            m.scratch += b.bytes as u64;
        }
        // The projection buffers hold the step's rows, then the per-slot state.
        m.scratch -= 2 * state;
        m.state = 2 * state;
        m
    }

    fn state_row(&self, slot: Slot, source: u32, pos: u32) -> usize {
        (slot as usize * self.sources + source as usize) * self.cap as usize
            + (pos % self.cap) as usize
    }

    /// Scrubs a slot's open entries.
    pub fn reset_state(&mut self, slot: Slot) -> Result<()> {
        if self.sources == 0 {
            return Ok(());
        }
        let per = self.sources * self.cap as usize;
        let first = slot as usize * per;
        self.open_tags[first..first + per].fill(0);
        let row = self.d.head_dim as usize * 4;
        let at = (self.max_rows as usize + first) * row;
        for buf in [&self.s.proj_kv, &self.s.proj_score] {
            // SAFETY: the slot's rows of the buffer's state part.
            unsafe { self.dev.zero(buf.at(at, per * row), per * row) }?;
        }
        Ok(())
    }

    /// Whether a slot's open entries are zero on the device and unrecorded on the host.
    pub fn state_is_zero(&self, slot: Slot) -> Result<bool> {
        if self.sources == 0 {
            return Ok(true);
        }
        let per = self.sources * self.cap as usize;
        let first = slot as usize * per;
        let row = self.d.head_dim as usize * 4;
        let at = (self.max_rows as usize + first) * row;
        let mut zero = self.open_tags[first..first + per].iter().all(|t| *t == 0);
        for buf in [&self.s.proj_kv, &self.s.proj_score] {
            let bytes: Vec<u8> = buf.read(self.dev.as_ref(), at, per * row)?;
            zero &= bytes.iter().all(|b| *b == 0);
        }
        Ok(zero)
    }

    /// Zeroes every scratch buffer in place (the step's rows of the projection buffers,
    /// not the per-slot state behind them).
    pub fn scrub_scratch(&self) -> Result<()> {
        let dev = self.dev.as_ref();
        let head = self.max_rows as usize * self.d.head_dim as usize * 4;
        for b in self.buffers() {
            let bytes = match b.kind {
                Kind::State => head.min(b.bytes),
                _ => b.bytes,
            };
            // SAFETY: a prefix of the buffer.
            unsafe { dev.zero(b.addr, bytes) }?;
        }
        Ok(())
    }

    /// Whether every scratch buffer is zero (the step's rows of the projection buffers).
    pub fn scratch_is_zero(&self) -> Result<bool> {
        let dev = self.dev.as_ref();
        let head = self.max_rows as usize * self.d.head_dim as usize * 4;
        for b in self.buffers() {
            let bytes = match b.kind {
                Kind::State => head.min(b.bytes),
                _ => b.bytes,
            };
            if b.read::<u8>(dev, 0, bytes)?.iter().any(|x| *x != 0) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Zeroes everything that is not a weight: pools, tables, state, scratch and the
    /// host tier.
    pub fn scrub_all(&mut self) -> Result<()> {
        let dev = self.dev.as_ref();
        for g in &self.kv.groups {
            g.pool.clear(dev)?;
            g.tables.clear(dev)?;
        }
        for b in self.buffers() {
            b.clear(dev)?;
        }
        self.kv.scrub_host(dev)?;
        dev.synchronize()
    }

    /// With position tags: `layer` of group `g` now holds the positions of the step's
    /// first `rows` rows.
    pub fn tag_rows(&mut self, g: usize, layer: u32, rows: u32) {
        if !self.kv.tagged() {
            return;
        }
        for i in 0..rows as usize {
            let (slot, pos) = self.items[i];
            let entry = self.kv.entry(g, slot, pos);
            self.kv.tag(g, layer, entry, pos);
        }
    }

    /// With position tags: the window positions `pos + 1 - window ..= pos` of `layer` of
    /// group `g` hold what a step computed, for a read at `pos` of `slot`.
    pub fn check_window(&self, g: usize, layer: u32, slot: Slot, pos: u32) -> Result<()> {
        for p in (pos + 1).saturating_sub(self.d.window)..=pos {
            self.kv.check_tag(g, layer, self.kv.entry(g, slot, p), p)?;
        }
        Ok(())
    }

    /// With position tags: every entry the first `rows` rows of the step read at a layer
    /// (its window and, with `bank_layer`, every visible entry of that bank layer) holds
    /// what a step computed.
    fn check_reads(&self, run: &LayerRun<'_>, rows: u32) -> Result<()> {
        if !self.kv.tagged() {
            return Ok(());
        }
        let ratio = self.kv.groups[BANK].geom.ratio;
        for &(slot, pos) in &self.items[..rows as usize] {
            self.check_window(WINDOW, run.window_layer, slot, pos)?;
            let Some(bl) = run.bank_layer else { continue };
            for e in 0..(pos + 1) / ratio {
                let close = (e + 1) * ratio - 1;
                self.kv
                    .check_tag(BANK, bl, self.kv.entry(BANK, slot, close), close)?;
            }
        }
        Ok(())
    }

    fn lin8(
        &self,
        codes: &Buffer,
        scales: &Buffer,
        w: &Fp8,
        out: &Buffer,
        rows: u32,
    ) -> Result<()> {
        // SAFETY: the scratch rows hold `rows` rows of `w.k` codes; `out` holds `rows`
        // rows of `w.n` values (`Scratch::new` sizes each for the largest step).
        unsafe {
            self.dev.kernels().lin8(&Lin8Params {
                codes: codes.addr,
                act_scales: scales.addr,
                w_codes: w.codes,
                w_scales: w.scales,
                out: out.addr,
                rows,
                n: w.n,
                k: w.k,
                reserved: 0,
            })
        }
    }

    fn quant(
        &self,
        x: &Buffer,
        codes: &Buffer,
        scales: &Buffer,
        rows: u32,
        width: u32,
    ) -> Result<()> {
        // SAFETY: as `lin8`.
        unsafe {
            self.dev.kernels().quant_fp8(&QuantFp8Params {
                x: x.addr,
                x_scales: 0,
                codes: codes.addr,
                scales: scales.addr,
                rows,
                width,
                x_format: X_BF16,
                reserved: 0,
            })
        }
    }

    fn rms(&self, x: &Buffer, weight: u64, rows: u32, width: u32) -> Result<()> {
        // SAFETY: as `lin8`; the norm runs in place.
        unsafe {
            self.dev.kernels().rms_norm(&RmsNormParams {
                x: x.addr,
                weight,
                out: x.addr,
                rows,
                width,
                eps: self.c.norm_eps,
                reserved: 0,
            })
        }
    }

    fn mix(&self, w: &Mhc, out: &Buffer, rows: u32) -> Result<()> {
        // SAFETY: as `lin8`.
        unsafe {
            self.dev.kernels().mhc_mix(&MhcMixParams {
                x: self.s.x.addr,
                projection: w.projection,
                base: w.base,
                out: out.addr,
                rows,
                dim: self.d.dim,
                copies: self.d.copies,
                iters: narrow(self.c.hc_sinkhorn_iters, "sinkhorn steps")?,
                scale: w.scale,
                norm_eps: self.c.norm_eps,
                eps: self.c.hc_eps,
                reserved: 0,
            })
        }
    }

    /// The stream collapsed with the `pre` of `mixes`' rows, into `u`.
    pub fn collapse(&self, mixes: &Buffer, rows: u32) -> Result<()> {
        // SAFETY: as `lin8`.
        unsafe {
            self.dev.kernels().collapse(&CollapseParams {
                x: self.s.x.addr,
                pre: mixes.addr,
                out: self.s.u.addr,
                rows,
                dim: self.d.dim,
                copies: self.d.copies,
                pre_stride: self.d.mix_words,
            })
        }
    }

    fn expand(&self, mixes: &Buffer, rows: u32) -> Result<()> {
        // SAFETY: as `lin8`; the expansion runs in place.
        unsafe {
            self.dev.kernels().expand(&ExpandParams {
                x: self.s.x.addr,
                y: self.s.y.addr,
                mixes: mixes.addr,
                out: self.s.x.addr,
                rows,
                dim: self.d.dim,
                copies: self.d.copies,
                mix_stride: self.d.mix_words,
            })
        }
    }

    /// A plain product of `rows` rows of `k` BF16 values with a BF16 weight.
    #[allow(clippy::too_many_arguments)]
    pub fn linear(
        &self,
        x: &Buffer,
        w: u64,
        out: &Buffer,
        rows: u32,
        n: u32,
        k: u32,
        out_dtype: u32,
    ) -> Result<()> {
        // SAFETY: as `lin8`.
        unsafe {
            self.dev.kernels().linear(&LinearParams {
                x: x.addr,
                x_rows: 0,
                w,
                addend: 0,
                out: out.addr,
                rows,
                n,
                k,
                x_dtype: DT_BF16,
                w_dtype: DT_BF16,
                out_dtype,
            })
        }
    }

    /// Rotation and a fake quantisation of `rows` rows of `x`, into `plane` at the
    /// `targets` entries or densely into `codes` and `scales`.
    #[allow(clippy::too_many_arguments)]
    pub fn rot_quant(
        &self,
        x: &Buffer,
        cis: u64,
        positions: &Buffer,
        dst: Dst<'_>,
        rows: u32,
        heads: u32,
        head_dim: u32,
        rotation: u32,
        format: u32,
    ) -> Result<()> {
        let mut p = RotQuantParams {
            x: x.addr,
            cis,
            positions: positions.addr,
            rows,
            heads,
            head_dim,
            rope_dim: self.d.rope_dim,
            rotation,
            format,
            cis_positions: self.d.positions,
            ..RotQuantParams::default()
        };
        match dst {
            Dst::Plane(plane, targets) => {
                p.dst = plane;
                p.targets = targets.addr;
            }
            Dst::Dense(codes, scales) => {
                p.dst_codes = codes.addr;
                p.dst_scales = scales.addr;
            }
        }
        // SAFETY: as `lin8`; every position is below the tables' rows and every target
        // an entry of the plane's pool (the step's arrays are built from the mirror).
        unsafe { self.dev.kernels().rot_quant(&p) }
    }

    fn plane(&self, g: usize, layer: u32, plane: Plane) -> KvPlane {
        let gs = &self.kv.groups[g];
        gs.geom.plane(gs.pool.addr, layer, plane)
    }

    /// One model layer over the step's rows.
    pub fn layer(&mut self, l: usize, run: &LayerRun<'_>) -> Result<()> {
        let lw = *self.w.layer(l);
        let spec = self.c.layers[l].clone();
        let cis = self.w.rope(spec.rope);
        let (na, ne) = (run.active, run.emit);
        if na == 0 {
            return Ok(());
        }
        let d = self.d.clone();
        let status = self.s.status.addr;

        if let Some(e) = &lw.engram {
            let (codes, scales) = &self.s.gathered[e.slot];
            // SAFETY: the gathered rows of this step, `na` of them.
            unsafe {
                self.dev.kernels().quant_fp8(&QuantFp8Params {
                    x: codes.addr,
                    x_scales: scales.addr,
                    codes: self.s.ecodes.addr,
                    scales: self.s.escales.addr,
                    rows: na,
                    width: d.engram_width,
                    x_format: X_FP8,
                    reserved: 0,
                })?;
            }
            self.lin8(&self.s.ecodes, &self.s.escales, &e.wkv, &self.s.ekv, na)?;
            // SAFETY: as `lin8`; the injection runs in place.
            unsafe {
                self.dev.kernels().engram_inject(&EngramInjectParams {
                    x: self.s.x.addr,
                    kv: self.s.ekv.addr,
                    gate_weight: e.gate_weight,
                    out: self.s.x.addr,
                    rows: na,
                    dim: d.dim,
                    copies: d.copies,
                    norm_eps: self.c.norm_eps,
                })?;
            }
        }

        if let Some(column) = run.tap {
            let (ds, draft) = self.drafting()?;
            let stride = draft.targets * d.dim;
            // SAFETY: as `lin8`; a row's taps are `targets` vectors side by side.
            unsafe {
                self.dev.kernels().stream_mean(&StreamMeanParams {
                    x: self.s.x.addr,
                    out: ds.hidden.at((column * d.dim) as usize * 2, 0),
                    rows: na,
                    dim: d.dim,
                    copies: d.copies,
                    out_stride: stride,
                })?;
            }
        }

        // The attention input, and the window KV written from it.
        self.mix(&lw.hc_attn, &self.s.mix_a, na)?;
        self.collapse(&self.s.mix_b, na)?;
        self.rms(&self.s.u, lw.attn_norm, na, d.dim)?;
        self.quant(&self.s.u, &self.s.ucodes, &self.s.uscales, na, d.dim)?;
        self.lin8(&self.s.ucodes, &self.s.uscales, &lw.wkv, &self.s.kv, na)?;
        self.rms(&self.s.kv, lw.kv_norm, na, d.head_dim)?;
        self.rot_quant(
            &self.s.kv,
            cis,
            &self.s.positions,
            Dst::Plane(
                self.plane(WINDOW, run.window_layer, Plane::Window),
                &self.s.wtargets,
            ),
            na,
            1,
            d.head_dim,
            ROT_ROTATE,
            KV_FP8,
        )?;

        self.tag_rows(WINDOW, run.window_layer, na);

        if let Some((source, pool)) = run.build {
            self.build_bank(&lw, source, pool, cis, na)?;
        }
        if ne == 0 {
            return Ok(());
        }
        self.check_reads(run, ne)?;

        // The queries.
        self.lin8(
            &self.s.ucodes,
            &self.s.uscales,
            &lw.wq_a,
            &self.s.latent,
            ne,
        )?;
        self.rms(&self.s.latent, lw.q_norm, ne, d.q_rank)?;
        self.quant(
            &self.s.latent,
            &self.s.lcodes,
            &self.s.lscales,
            ne,
            d.q_rank,
        )?;
        self.lin8(&self.s.lcodes, &self.s.lscales, &lw.wq_b, &self.s.qraw, ne)?;
        // SAFETY: as `rot_quant`: the rotation alone, BF16 rows out.
        unsafe {
            self.dev.kernels().rot_quant(&RotQuantParams {
                x: self.s.qraw.addr,
                cis,
                positions: self.s.positions.addr,
                dst_codes: self.s.q.addr,
                rows: ne,
                heads: d.heads,
                head_dim: d.head_dim,
                rope_dim: d.rope_dim,
                rotation: ROT_ROTATE,
                format: KV_BF16,
                cis_positions: d.positions,
                ..RotQuantParams::default()
            })?;
        }

        // The selection, for a layer that runs its indexer.
        let window = self.plane(WINDOW, run.window_layer, Plane::Window);
        let mut bank = window;
        let mut b_slots = 0;
        if let Some(bl) = run.bank_layer {
            bank = self.plane(BANK, bl, Plane::BankKv);
            b_slots = d.top_k;
            let tables = self.kv.groups[BANK].tables.addr;
            if let Some(ix) = &lw.indexer {
                self.lin8(&self.s.lcodes, &self.s.lscales, &ix.wq_b, &self.s.iq, ne)?;
                self.rot_quant(
                    &self.s.iq,
                    cis,
                    &self.s.positions,
                    Dst::Dense(&self.s.iq_codes, &self.s.iq_scales),
                    ne,
                    d.index_heads,
                    d.index_dim,
                    ROT_ROTATE,
                    KV_FP4_E8M0,
                )?;
                self.linear(
                    &self.s.u,
                    ix.weights_proj,
                    &self.s.head_dots,
                    ne,
                    d.index_heads,
                    d.dim,
                    DT_BF16,
                )?;
                let entries = self.kv.groups[BANK].geom.entries();
                // SAFETY: as `lin8`; every row's visible entries lie in mapped blocks
                // (checked against the mirror when the step's arrays were built).
                unsafe {
                    self.dev.kernels().index_score(&IndexScoreParams {
                        q_codes: self.s.iq_codes.addr,
                        q_scales: self.s.iq_scales.addr,
                        head_dots: self.s.head_dots.addr,
                        tables,
                        table_row: self.s.table_row.addr,
                        visible: self.s.visible.addr,
                        scores: self.s.scores.addr,
                        status,
                        keys: self.plane(BANK, bl, Plane::BankKey),
                        rows: ne,
                        heads: d.index_heads,
                        dim: d.index_dim,
                        table_stride: self.kv.stride,
                        score_stride: d.score_stride,
                        weight_scale: self.c.indexer.weight_scale,
                    })?;
                    self.dev.kernels().select(&SelectParams {
                        scores: self.s.scores.addr,
                        visible: self.s.visible.addr,
                        mask_in: match spec.uses_candidates {
                            true => self.s.mask.addr,
                            false => 0,
                        },
                        mask_out: match spec.is_candidate_source {
                            true => self.s.mask.addr,
                            false => 0,
                        },
                        tables,
                        table_row: self.s.table_row.addr,
                        slots: self.s.slots.addr,
                        status,
                        rows: ne,
                        score_stride: d.score_stride,
                        mask_words: d.mask_words,
                        table_stride: self.kv.stride,
                        slot_stride: d.slot_stride,
                        slot_first: d.window,
                        top_k: d.top_k,
                        top_blocks: d.candidate_blocks,
                        block_entries: d.candidate_entries,
                        entries_per_block: entries,
                    })?;
                }
            }
        }

        // Attention and the output projection.
        // SAFETY: as `lin8`; the slot lists name entries of the two planes' pools.
        unsafe {
            self.dev.kernels().attention(&AttnParams {
                q: self.s.q.addr,
                slots: self.s.slots.addr,
                sink: lw.sink,
                out: self.s.attn.addr,
                a: window,
                b: bank,
                rows: ne,
                heads: d.heads,
                head_dim: d.head_dim,
                a_slots: d.window,
                b_slots,
                slot_stride: d.slot_stride,
                scale: self.c.attention.softmax_scale,
                layout: ATTN_FIXED,
            })?;
        }
        self.rot_quant(
            &self.s.attn,
            cis,
            &self.s.positions,
            Dst::Dense(&self.s.ocodes, &self.s.oscales),
            ne,
            d.heads,
            d.head_dim,
            ROT_UNROTATE,
            KV_FP8,
        )?;
        // SAFETY: as `lin8`.
        unsafe {
            self.dev.kernels().grouped_dot(&GroupedDotParams {
                codes: self.s.ocodes.addr,
                act_scales: self.s.oscales.addr,
                w_codes: lw.wo_a.codes,
                w_scales: lw.wo_a.scales,
                out: self.s.z.addr,
                rows: ne,
                groups: d.o_groups,
                rank: d.o_rank,
                k: lw.wo_a.k,
            })?;
        }
        let rank = d.o_groups * d.o_rank;
        self.quant(&self.s.z, &self.s.zcodes, &self.s.zscales, ne, rank)?;
        self.lin8(&self.s.zcodes, &self.s.zscales, &lw.wo_b, &self.s.y, ne)?;
        self.expand(&self.s.mix_a, ne)?;

        // The feed-forward.
        self.mix(&lw.hc_ffn, &self.s.mix_b, ne)?;
        self.collapse(&self.s.mix_a, ne)?;
        self.rms(&self.s.u, lw.ffn_norm, ne, d.dim)?;
        self.quant(&self.s.u, &self.s.ucodes, &self.s.uscales, ne, d.dim)?;
        self.moe(&lw, ne, &d.moe)?;
        self.expand(&self.s.mix_b, ne)
    }

    /// The feed-forward of `rows` rows from their quantised input, into `y`.
    fn moe(&self, lw: &LayerWeights, rows: u32, m: &MoeDims) -> Result<()> {
        let (d, s, k) = (&self.d, &self.s, self.dev.kernels());
        let pairs = rows
            .checked_mul(m.active)
            .ok_or_else(|| CudaError::new("routed pairs"))?;
        self.linear(&s.u, lw.router, &s.rlogits, rows, m.experts, d.dim, DT_F32)?;
        // SAFETY: as `lin8`; the layout buffers hold every pair of the largest step.
        unsafe {
            k.route(&RouteParams {
                logits: s.rlogits.addr,
                bias: lw.bias,
                ids: s.ids.addr,
                weights: s.rweights.addr,
                rows,
                experts: m.experts,
                active: m.active,
                normalize: u32::from(m.normalize),
                route_scale: m.route_scale,
                reserved: 0,
            })?;
            k.place(&PlaceParams {
                ids: s.ids.addr,
                weights: s.rweights.addr,
                row_of: s.row_of.addr,
                row_weight: s.row_weight.addr,
                group_end: s.group_end.addr,
                scratch: s.place_scratch.addr,
                rows,
                active: m.active,
                groups: m.experts,
                row_align: ROW_ALIGN,
                layout_rows: pairs,
                reserved: 0,
            })?;
            for (src, dst, row_bytes) in [
                (&s.ucodes, &s.lay_codes, d.dim),
                (&s.uscales, &s.lay_scales, d.dim / 32),
            ] {
                k.place_rows(&PlaceRowsParams {
                    src: src.addr,
                    dst: dst.addr,
                    row_of: s.row_of.addr,
                    status: s.status.addr,
                    rows,
                    active: m.active,
                    row_bytes,
                    layout_rows: pairs,
                })?;
            }
            let lin4 = |codes: &Buffer, scales: &Buffer, w: (u64, u64), out: &Buffer, n, kk| {
                k.lin4(&Lin4Params {
                    codes: codes.addr,
                    act_scales: scales.addr,
                    w_packed: w.0,
                    w_scales: w.1,
                    group_end: s.group_end.addr,
                    out: out.addr,
                    layout_rows: pairs,
                    n,
                    k: kk,
                    groups: m.experts,
                    row_align: ROW_ALIGN,
                    reserved: 0,
                })
            };
            let e = &lw.experts;
            lin4(
                &s.lay_codes,
                &s.lay_scales,
                (e.up_packed, e.up_scales),
                &s.gate_up,
                2 * m.inter,
                d.dim,
            )?;
            k.swiglu(&SwigluParams {
                gate: s.gate_up.addr,
                up: s.gate_up.at(m.inter as usize * 2, 0),
                weight: s.row_weight.addr,
                codes: s.hid_codes.addr,
                scales: s.hid_scales.addr,
                rows: pairs,
                width: m.inter,
                in_stride: 2 * m.inter,
                limit: m.limit,
            })?;
            lin4(
                &s.hid_codes,
                &s.hid_scales,
                (e.down_packed, e.down_scales),
                &s.routed,
                d.dim,
                m.inter,
            )?;
        }
        // The shared expert.
        self.lin8(&s.ucodes, &s.uscales, &lw.w1, &s.sgate, rows)?;
        self.lin8(&s.ucodes, &s.uscales, &lw.w3, &s.sup, rows)?;
        // SAFETY: as `lin8`.
        unsafe {
            k.swiglu(&SwigluParams {
                gate: s.sgate.addr,
                up: s.sup.addr,
                weight: 0,
                codes: s.shcodes.addr,
                scales: s.shscales.addr,
                rows,
                width: m.inter,
                in_stride: m.inter,
                limit: m.limit,
            })?;
        }
        self.lin8(&s.shcodes, &s.shscales, &lw.w2, &s.shared, rows)?;
        // SAFETY: as `lin8`.
        unsafe {
            k.combine(&CombineParams {
                routed: s.routed.addr,
                shared: s.shared.addr,
                row_of: s.row_of.addr,
                out: s.y.addr,
                rows,
                active: m.active,
                dim: d.dim,
                reserved: 0,
            })
        }
    }

    /// A pooled source's bank entries from the attention input of `rows` rows: the
    /// projections, the pairs that close (a member may be the slot's open entry), their
    /// index keys and KV written to the bank, and the projections the slot keeps.
    fn build_bank(
        &mut self,
        lw: &LayerWeights,
        source: u32,
        pool: &PoolPlan,
        cis: u64,
        rows: u32,
    ) -> Result<()> {
        let cw = lw
            .compressor
            .ok_or_else(|| CudaError::new("a KV source without a compressor"))?;
        let wgate = cw
            .wgate
            .ok_or_else(|| CudaError::new("a pooled KV source without a gate"))?;
        let d = self.d.clone();
        self.linear(
            &self.s.u,
            cw.wkv,
            &self.s.proj_kv,
            rows,
            d.head_dim,
            d.dim,
            DT_F32,
        )?;
        self.linear(
            &self.s.u,
            wgate,
            &self.s.proj_score,
            rows,
            d.head_dim,
            d.dim,
            DT_F32,
        )?;

        let entries: u32 = narrow(pool.closes.len(), "pooled entries")?;
        if entries > 0 {
            let mut first = Vec::with_capacity(pool.first.len());
            for m in &pool.first {
                first.push(match *m {
                    Member::Row(r) => r,
                    Member::State { slot, pos } => {
                        let at = self.state_row(slot, source, pos);
                        assert_eq!(
                            self.open_tags[at],
                            pos + 1,
                            "slot {slot}: the open global entry's state at position {pos} is \
                             not held"
                        );
                        self.max_rows + narrow::<u32, usize>(at, "state row")?
                    }
                });
            }
            let (mut targets, mut positions) = (Vec::new(), Vec::new());
            for (slot, pos) in &pool.closes {
                let target = self.kv.target(BANK, *slot, *pos);
                self.kv.tag(BANK, source, target, *pos);
                targets.push(target);
                positions.push(*pos - 1);
            }
            let dev = self.dev.as_ref();
            self.s.pool_first.write(dev, 0, &first)?;
            self.s.pool_second.write(dev, 0, &pool.second)?;
            self.s.pool_pos.write(dev, 0, &positions)?;
            self.s.pool_targets.write(dev, 0, &targets)?;
            // SAFETY: as `lin8`; the member rows are rows of this step or of the state
            // behind them.
            unsafe {
                dev.kernels().pool(&PoolParams {
                    kv: self.s.proj_kv.addr,
                    score: self.s.proj_score.addr,
                    first: self.s.pool_first.addr,
                    second: self.s.pool_second.addr,
                    out: self.s.pooled.addr,
                    entries,
                    width: d.head_dim,
                })?;
            }
            self.rms(&self.s.pooled, cw.norm, entries, d.head_dim)?;
            self.linear(
                &self.s.pooled,
                cw.wk,
                &self.s.key,
                entries,
                d.index_dim,
                d.head_dim,
                DT_BF16,
            )?;
            self.rms(&self.s.key, cw.k_norm, entries, d.index_dim)?;
            self.rot_quant(
                &self.s.key,
                cis,
                &self.s.pool_pos,
                Dst::Plane(
                    self.plane(BANK, source, Plane::BankKey),
                    &self.s.pool_targets,
                ),
                entries,
                1,
                d.index_dim,
                ROT_ROTATE,
                KV_FP4_E8M0,
            )?;
            self.rot_quant(
                &self.s.pooled,
                cis,
                &self.s.pool_pos,
                Dst::Plane(
                    self.plane(BANK, source, Plane::BankKv),
                    &self.s.pool_targets,
                ),
                entries,
                1,
                d.head_dim,
                ROT_ROTATE,
                KV_FP4_E4M3,
            )?;
        }
        // Queued after the pooling, which reads the state these overwrite.
        let row = d.head_dim as usize * 4;
        for (r, slot, pos) in &pool.keep {
            let at = self.state_row(*slot, source, *pos);
            self.open_tags[at] = pos + 1;
            let to = (self.max_rows as usize + at) * row;
            for buf in [&self.s.proj_kv, &self.s.proj_score] {
                // SAFETY: a row of the step and a row of the state, which do not overlap.
                unsafe {
                    self.dev
                        .copy(buf.at(to, row), buf.at(*r as usize * row, row), row)?;
                }
            }
        }
        Ok(())
    }

    /// The step's status word, after everything queued.
    pub fn status(&self) -> Result<u32> {
        Ok(self.s.status.read::<u32>(self.dev.as_ref(), 0, 1)?[0])
    }
}

/// The drafter, on the decoder half.
impl HalfNet {
    /// A pass's own KV as a plane: entry `e` is the pass's row `e`.
    fn draft_plane(&self, ds: &DraftScratch, rows: u32) -> KvPlane {
        let hd = self.d.head_dim;
        KvPlane {
            base: ds.kv.addr,
            block_bytes: ds.kv.bytes as u64,
            codes_offset: 0,
            scales_offset: u64::from(rows) * u64::from(hd),
            entries_per_block: rows,
            format: KV_FP8,
            width: hd,
            reserved: 0,
        }
    }

    /// The draft pass over `rows` rows (every slot of every drafting row, slot-major),
    /// from their tokens (`tokens` scratch), positions (`positions` scratch) and slot
    /// lists (the drafter scratch's): the draft blocks, each row attending to its
    /// sequence's context KV in ring order and to every slot of its own pass, then the
    /// drafter's norm and the head. Leaves the base logits in the drafter's scratch.
    pub fn draft_pass(&mut self, rows: u32) -> Result<()> {
        let d = self.d.clone();
        let (_, draft) = self.drafting()?;
        let w = self
            .w
            .drafter
            .clone()
            .ok_or_else(|| CudaError::new("the drafter is not loaded"))?;
        let cis = self.w.rope(RopeKind::Plain);
        let (head, _) = self.w.head.expect("the decoder holds the head");
        let ds = self.ds.as_ref().expect("checked: the half drafts");
        let k = self.dev.kernels();
        // SAFETY: `rows` rows of every scratch array (sized for a pass of the largest
        // step); tokens were checked against the vocabulary on the host.
        unsafe {
            k.embed(&EmbedParams {
                table: w.embed,
                tokens: self.s.tokens.addr,
                out: self.s.x.addr,
                status: self.s.status.addr,
                rows,
                dim: d.dim,
                copies: d.copies,
                vocab: d.vocab,
            })?;
        }
        let plane = self.draft_plane(ds, rows);
        let group = self.kv.groups.len() - 1;
        for (b, lw) in w.blocks.iter().enumerate() {
            self.mix(&lw.hc_attn, &self.s.mix_a, rows)?;
            self.collapse(&self.s.mix_b, rows)?;
            self.rms(&self.s.u, lw.attn_norm, rows, d.dim)?;
            self.quant(&self.s.u, &self.s.ucodes, &self.s.uscales, rows, d.dim)?;
            // The queries.
            self.lin8(
                &self.s.ucodes,
                &self.s.uscales,
                &lw.wq_a,
                &self.s.latent,
                rows,
            )?;
            self.rms(&self.s.latent, lw.q_norm, rows, d.q_rank)?;
            self.quant(
                &self.s.latent,
                &self.s.lcodes,
                &self.s.lscales,
                rows,
                d.q_rank,
            )?;
            self.lin8(
                &self.s.lcodes,
                &self.s.lscales,
                &lw.wq_b,
                &self.s.qraw,
                rows,
            )?;
            // SAFETY: as `rot_quant`.
            unsafe {
                k.rot_quant(&RotQuantParams {
                    x: self.s.qraw.addr,
                    cis,
                    positions: self.s.positions.addr,
                    dst_codes: self.s.q.addr,
                    rows,
                    heads: d.heads,
                    head_dim: d.head_dim,
                    rope_dim: d.rope_dim,
                    rotation: ROT_ROTATE,
                    format: KV_BF16,
                    cis_positions: d.positions,
                    ..RotQuantParams::default()
                })?;
            }
            // Each slot's own KV, into the pass's scratch plane.
            self.lin8(&self.s.ucodes, &self.s.uscales, &lw.wkv, &self.s.kv, rows)?;
            self.rms(&self.s.kv, lw.kv_norm, rows, d.head_dim)?;
            // SAFETY: as `rot_quant`: dense rows, the plane's codes then its scales.
            unsafe {
                k.rot_quant(&RotQuantParams {
                    x: self.s.kv.addr,
                    cis,
                    positions: self.s.positions.addr,
                    dst_codes: plane.base,
                    dst_scales: plane.base + plane.scales_offset,
                    rows,
                    heads: 1,
                    head_dim: d.head_dim,
                    rope_dim: d.rope_dim,
                    rotation: ROT_ROTATE,
                    format: KV_FP8,
                    cis_positions: d.positions,
                    ..RotQuantParams::default()
                })?;
                // The context in ring order, then the pass's slots, packed.
                k.attention(&AttnParams {
                    q: self.s.q.addr,
                    slots: ds.slots.addr,
                    sink: lw.sink,
                    out: self.s.attn.addr,
                    a: self.plane(group, narrow(b, "draft block")?, Plane::Window),
                    b: plane,
                    rows,
                    heads: d.heads,
                    head_dim: d.head_dim,
                    a_slots: d.window,
                    b_slots: draft.slots,
                    slot_stride: d.window + draft.slots,
                    scale: self.c.attention.softmax_scale,
                    layout: ATTN_PACKED,
                })?;
            }
            self.rot_quant(
                &self.s.attn,
                cis,
                &self.s.positions,
                Dst::Dense(&self.s.ocodes, &self.s.oscales),
                rows,
                d.heads,
                d.head_dim,
                ROT_UNROTATE,
                KV_FP8,
            )?;
            // SAFETY: as `lin8`.
            unsafe {
                k.grouped_dot(&GroupedDotParams {
                    codes: self.s.ocodes.addr,
                    act_scales: self.s.oscales.addr,
                    w_codes: lw.wo_a.codes,
                    w_scales: lw.wo_a.scales,
                    out: self.s.z.addr,
                    rows,
                    groups: d.o_groups,
                    rank: d.o_rank,
                    k: lw.wo_a.k,
                })?;
            }
            let rank = d.o_groups * d.o_rank;
            self.quant(&self.s.z, &self.s.zcodes, &self.s.zscales, rows, rank)?;
            self.lin8(&self.s.zcodes, &self.s.zscales, &lw.wo_b, &self.s.y, rows)?;
            self.expand(&self.s.mix_a, rows)?;
            self.mix(&lw.hc_ffn, &self.s.mix_b, rows)?;
            self.collapse(&self.s.mix_a, rows)?;
            self.rms(&self.s.u, lw.ffn_norm, rows, d.dim)?;
            self.quant(&self.s.u, &self.s.ucodes, &self.s.uscales, rows, d.dim)?;
            self.moe(lw, rows, &draft.moe)?;
            self.expand(&self.s.mix_b, rows)?;
        }
        self.collapse(&self.s.mix_b, rows)?;
        self.rms(&self.s.u, w.norm, rows, d.dim)?;
        // SAFETY: `rows` normed rows and `rows` rows of base logits.
        unsafe {
            k.head(&LinearParams {
                x: self.s.u.addr,
                w: head,
                out: ds.base.addr,
                rows,
                n: d.vocab,
                k: d.dim,
                x_dtype: DT_BF16,
                w_dtype: DT_BF16,
                out_dtype: DT_F32,
                ..LinearParams::default()
            })
        }
    }

    /// One slot's logits for the first `rows` drafting rows: the base logits of its
    /// pass rows (from row `base_row`) plus the Markov head over the embedding of each
    /// row's previous token (`previous`, `[rows]` ids on the device).
    pub fn draft_logits(&self, rows: u32, base_row: u32, previous: u64) -> Result<()> {
        let (ds, draft) = self.drafting()?;
        let w = self
            .w
            .drafter
            .as_ref()
            .ok_or_else(|| CudaError::new("the drafter is not loaded"))?;
        let vocab = self.d.vocab as usize;
        // SAFETY: `rows` rows of base logits from `base_row` and of output; a previous
        // token is a sampleable id (the sampler's, or checked on the host).
        unsafe {
            self.dev.kernels().head(&LinearParams {
                x: w.markov_embed,
                x_rows: previous,
                w: w.markov_head,
                addend: ds.base.at(base_row as usize * vocab * 4, 0),
                out: ds.logits.addr,
                rows,
                n: self.d.vocab,
                k: draft.markov_rank,
                x_dtype: DT_BF16,
                w_dtype: DT_BF16,
                out_dtype: DT_F32,
            })
        }
    }

    /// The drafter's context KV of the first `rows` rows, from their taps (the drafter
    /// scratch's `hidden`): the projection and its norm, then each draft block's window
    /// KV, written at the rows' entries of the drafter's group (the scratch's
    /// `targets`).
    pub fn drafter_context(&mut self, rows: u32) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let d = self.d.clone();
        let (ds, draft) = self.drafting()?;
        let w = self
            .w
            .drafter
            .clone()
            .ok_or_else(|| CudaError::new("the drafter is not loaded"))?;
        let cis = self.w.rope(RopeKind::Plain);
        let group = self.kv.groups.len() - 1;
        self.quant(
            &ds.hidden,
            &ds.hcodes,
            &ds.hscales,
            rows,
            draft.targets * d.dim,
        )?;
        self.lin8(&ds.hcodes, &ds.hscales, &w.main_proj, &ds.mainx, rows)?;
        self.rms(&ds.mainx, w.main_norm, rows, d.dim)?;
        self.quant(&ds.mainx, &self.s.ucodes, &self.s.uscales, rows, d.dim)?;
        for (b, lw) in w.blocks.iter().enumerate() {
            self.lin8(&self.s.ucodes, &self.s.uscales, &lw.wkv, &self.s.kv, rows)?;
            self.rms(&self.s.kv, lw.kv_norm, rows, d.head_dim)?;
            self.rot_quant(
                &self.s.kv,
                cis,
                &self.s.positions,
                Dst::Plane(
                    self.plane(group, narrow(b, "draft block")?, Plane::Window),
                    &ds.targets,
                ),
                rows,
                1,
                d.head_dim,
                ROT_ROTATE,
                KV_FP8,
            )?;
        }
        Ok(())
    }
}
