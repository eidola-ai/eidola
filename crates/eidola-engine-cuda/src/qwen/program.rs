//! The step program: every launch of a step, over the executor's fixed scratch and the
//! step table ([`super::plan`]), with no allocation, no host copy and no read-back inside
//! it. The eager path runs it over a table laid out for the step itself; a decode rung
//! runs it over the rung's layout, directly or as the graph captured from it. Either way
//! every per-step value is table data read through fixed addresses, so the captured
//! launches serve every step of the rung.
//!
//! Order: the status word cleared; the drafter at each drafting row's host position and
//! each chain depth, each depth's drafts drawn on the device into the draft lanes; the
//! target, once, over every row's host tokens followed by its drafts, and its head; the
//! drafter over the host positions of rows that do not draft; plain rows drawn, drafting
//! rows' target distributions and chain acceptance into the read-back; the drafter's
//! canonical KV at the drafts' positions; the normed rows into the slots' kept states and
//! the drafter's taps.

use eidola_engine::sampling::Stream;

use super::super::abi::*;
use super::super::attention::{Fa2Args, HEAD_DIM, PARTIAL_BYTES_PER_HEAD};
use super::super::device::QwenDevice;
use super::super::kernels::{AcceptLaunch, DN_SHORT_RUN, SampleArgs, fp8_splits, split_width};
use super::super::plan::{AttnAt, DrafterAt, Fa2At, ItemsAt, Layout, SAMPLE_WORDS, TargetAt};
use super::super::sampler::{accept_work_bytes, sample_work_bytes};
use super::super::weights::{AttentionW, DeltaNetW, LayerW, MixerW, ab_rows};
use super::{Alloc, DRAFTER, QwenCudaExecutor, TARGET};
use crate::mimo::attention::{AttnLayer, Partials, PlanView};
use crate::{CudaError, Result};

/// The capacities the scratch is sized for.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Caps {
    /// Rows of the target's forward: host tokens and drafts.
    pub rows: usize,
    /// Staged DeltaNet positions of one forward.
    pub staged: usize,
    /// Staged conv rows.
    pub conv_rows: usize,
    /// Rows of a drafter forward.
    pub drafter: usize,
    /// Head rows of one launch, target logit rows, and the drafter's logit rows (one
    /// depth's drafting rows).
    pub head: usize,
    pub logits_t: usize,
    pub logits_d: usize,
    /// Drafting rows and draft depth: lanes and draft distributions are `K · D`.
    pub drafting: usize,
    pub depth: usize,
    /// Plain draws and target distributions.
    pub plain: usize,
    pub targets: usize,
    /// Pad rows of each copy width.
    pub pad_conv: usize,
    pub pad_ab: usize,
    pub pad_o: usize,
    pub pad_h: usize,
    pub pad_t: usize,
    /// Words of the largest table and read-back.
    pub table: usize,
    pub readback: usize,
    /// The exact attention's score stride, and FA2's partial rows.
    pub score_stride: usize,
    pub partial_rows: usize,
}

/// Every buffer the program writes besides the seam's state: fixed for the executor's
/// life (the graphs captured their addresses).
pub(super) struct Scratch {
    pub caps: Caps,
    pub table: Alloc,
    pub readback: Alloc,
    pub h: Alloc,
    pub x: Alloc,
    pub mix: Alloc,
    pub m: Alloc,
    pub normed: Alloc,
    pub codes: Alloc,
    pub scales: Alloc,
    /// The split FP8 GEMMs' partial sums (fast profile only).
    pub parts: Alloc,
    /// The MLP's stacked gate and up outputs: `[rows, 2 · inter]`, gate's then up's.
    pub gate_up: Alloc,
    pub hmid: Alloc,
    /// The DeltaNet's stacked input projections: `[rows, conv_dim + value_dim]`, the
    /// convolution's inputs then `z`.
    pub qkvz: Alloc,
    pub aa: Alloc,
    pub bb: Alloc,
    pub nrm: Alloc,
    pub conv_in: Alloc,
    pub new_f32: Alloc,
    pub ab_new: Alloc,
    pub ab: Alloc,
    pub dq: Alloc,
    pub dk: Alloc,
    pub dv: Alloc,
    pub beta: Alloc,
    pub alpha: Alloc,
    pub o: Alloc,
    pub o_rows: Alloc,
    pub zero_rows: Alloc,
    /// The attention's stacked projections: `[rows, 2 · q_width + 2 · kv_width]`, the
    /// queries and gates, then keys, then values.
    pub qkv: Alloc,
    pub q: Alloc,
    pub kr: Alloc,
    pub attn: Alloc,
    pub y: Alloc,
    pub scores: Alloc,
    pub part_v: Alloc,
    pub part_s: Alloc,
    pub sink: Alloc,
    pub d_tokens: Alloc,
    pub d_e: Alloc,
    pub d_hid: Alloc,
    pub d_cat: Alloc,
    pub d_h: Alloc,
    pub d_normed: Alloc,
    pub head_in: Alloc,
    pub head_tmp: Alloc,
    pub logits_t: Alloc,
    pub logits_d: Alloc,
    pub p_plain: Alloc,
    pub p_target: Alloc,
    pub q_draft: Alloc,
    pub accept_scratch: Alloc,
    pub sample_work: Alloc,
    pub lanes: Alloc,
    pub pad_c: Alloc,
    pub pad_a: Alloc,
    pub pad_o: Alloc,
    pub pad_h: Alloc,
    pub pad_t: Alloc,
}

/// Bytes of the fast BF16 linears' f32 scratch: the head's rows of the vocabulary, the
/// drafter's input projection's rows of the hidden size, or a forward's rows of the
/// DeltaNet's stacked `a` and `b` projections.
fn head_tmp(c: &Caps, d: &super::Dims) -> usize {
    (pad4(c.head) * d.vocab as usize)
        .max(pad4(c.drafter) * d.hidden as usize)
        .max(pad4(c.rows.max(c.drafter)) * ab_rows(d.v_heads) as usize)
        * 4
}

/// f32 values a row of the split FP8 GEMMs' partial sums takes: the most of any FP8
/// linear of the model ([`split_width`]; 0 when none splits).
pub(super) fn parts_width(d: &super::Dims) -> usize {
    let (h, cd, vd) = (d.hidden, d.conv_dim(), d.value_dim());
    let (qw, kvw) = (d.q_heads * d.head_dim, d.kv_width());
    [
        (cd + vd, h),
        (h, vd),
        (2 * qw + 2 * kvw, h),
        (h, qw),
        (2 * d.inter, h),
        (h, d.inter),
    ]
    .into_iter()
    .map(|(n, k)| split_width(n, k))
    .max()
    .unwrap_or(0)
}

/// One group of a row-copy launch: its items, its width in words, and its buffers
/// (address, rows).
type CopyGroupAt<'a> = (&'a ItemsAt, usize, &'a [(u64, u64)]);

/// The most rows one sample launch takes: the plain draws, the target distributions, or
/// one depth's drafts.
fn sample_rows(c: &Caps) -> usize {
    c.plain.max(c.targets).max(c.drafting)
}

/// `rows` rounded up to the quantiser's and the fast GEMM's 4.
fn pad4(rows: usize) -> usize {
    rows.next_multiple_of(4).max(4)
}

impl<D: QwenDevice> QwenCudaExecutor<D> {
    /// The program's scratch for `c`.
    pub(super) fn alloc_scratch(
        dev: &D,
        d: &super::Dims,
        k_max: usize,
        c: Caps,
    ) -> Result<Scratch> {
        let a = |bytes: usize| Alloc::new(dev, bytes);
        let (hid, cd, vd, vh) = (
            d.hidden as usize,
            d.conv_dim() as usize,
            d.value_dim() as usize,
            d.v_heads as usize,
        );
        let qw = (d.q_heads * d.head_dim) as usize;
        let kvw = d.kv_width() as usize;
        let rows = pad4(c.rows.max(c.drafter));
        let vocab = d.vocab as usize;
        let n = vocab;
        let fast = dev.profile() == super::QwenProfile::Fast;
        let hq = d.q_heads as usize;
        let sink = a(hq * 4)?;
        // SAFETY: `hq` f32 of the allocation just made.
        unsafe {
            let inf: Vec<u8> = (0..hq)
                .flat_map(|_| f32::NEG_INFINITY.to_le_bytes())
                .collect();
            dev.upload(sink.addr, &inf)?;
        }
        Ok(Scratch {
            table: a(c.table.max(64) * 4)?,
            readback: a(c.readback * 4)?,
            h: a(rows * hid * 2)?,
            x: a(rows * hid * 2)?,
            mix: a(rows * hid * 2)?,
            m: a(rows * hid * 2)?,
            normed: a(pad4(c.rows) * hid * 2)?,
            codes: a(rows * k_max)?,
            scales: a(k_max / 128 * rows * 4)?,
            parts: a(if fast {
                (rows * parts_width(d) * 4).max(8)
            } else {
                8
            })?,
            gate_up: a(rows * 2 * d.inter as usize * 2)?,
            hmid: a(rows * d.inter as usize * 2)?,
            qkvz: a(rows * (cd + vd) * 2)?,
            aa: a(rows * vh * 2)?,
            bb: a(rows * vh * 2)?,
            nrm: a(rows * vd * 2)?,
            conv_in: a(c.conv_rows.max(4) * cd * 4)?,
            new_f32: a(rows * cd * 4)?,
            ab_new: a(rows * 2 * vh * 4)?,
            ab: a(c.staged.max(1) * 2 * vh * 4)?,
            dq: a(c.staged.max(1) * d.key_dim() as usize * 4)?,
            dk: a(c.staged.max(1) * d.key_dim() as usize * 4)?,
            dv: a(c.staged.max(1) * vd * 4)?,
            beta: a(c.staged.max(1) * vh * 4)?,
            alpha: a(c.staged.max(1) * vh * 4)?,
            o: a(c.staged.max(1) * vd * 4)?,
            o_rows: a(rows * vd * 4)?,
            zero_rows: a(3 * cd * 4)?,
            qkv: a(rows * (2 * qw + 2 * kvw) * 2)?,
            q: a(rows * qw * 2)?,
            kr: a(rows * kvw * 2)?,
            attn: a(rows * qw * 2)?,
            y: a(rows * qw * 2)?,
            scores: a(if fast {
                8
            } else {
                rows * hq * c.score_stride * 4
            })?,
            part_v: a(if fast {
                c.partial_rows * hq * HEAD_DIM as usize * 2
            } else {
                8
            })?,
            part_s: a(if fast { c.partial_rows * hq * 4 } else { 8 })?,
            sink,
            d_tokens: a(c.drafter.max(1) * 4)?,
            d_e: a(rows * hid * 2)?,
            d_hid: a(rows * hid * 2)?,
            d_cat: a(rows * 2 * hid * 2)?,
            d_h: a(rows * hid * 2)?,
            d_normed: a(rows * hid * 2)?,
            head_in: a(pad4(c.head) * hid * 2)?,
            head_tmp: a(if fast { head_tmp(&c, d) } else { 8 })?,
            logits_t: a(c.logits_t.max(1) * vocab * 4)?,
            logits_d: a(c.logits_d.max(1) * vocab * 4)?,
            p_plain: a(c.plain.max(1) * n * 8)?,
            p_target: a(c.targets.max(1) * n * 8)?,
            q_draft: a((c.drafting * c.depth).max(1) * n * 8)?,
            accept_scratch: a(accept_work_bytes(c.drafting))?,
            sample_work: a(sample_work_bytes(sample_rows(&c)))?,
            lanes: a((c.drafting * c.depth).max(1) * 4)?,
            pad_c: a(c.pad_conv.max(1) * cd * 4)?,
            pad_a: a(c.pad_ab.max(1) * 2 * vh * 4)?,
            pad_o: a(c.pad_o.max(1) * vd * 4)?,
            pad_h: a(c.pad_h.max(1) * hid * 2)?,
            pad_t: a(c.pad_t.max(1) * 4)?,
            caps: c,
        })
    }

    /// Every scratch range the program writes, for the idle scrub.
    pub(super) fn scratch_ranges(&self) -> Vec<(u64, usize)> {
        let s = &self.scratch;
        let c = &s.caps;
        let d = &self.dims;
        let (hid, cd, vd, vh) = (
            d.hidden as usize,
            d.conv_dim() as usize,
            d.value_dim() as usize,
            d.v_heads as usize,
        );
        let qw = (d.q_heads * d.head_dim) as usize;
        let kvw = d.kv_width() as usize;
        let rows = pad4(c.rows.max(c.drafter));
        let vocab = d.vocab as usize;
        let hq = d.q_heads as usize;
        let fast = self.dev.profile() == super::QwenProfile::Fast;
        let kmax = self.k_max();
        vec![
            (s.table.addr, c.table.max(64) * 4),
            (s.readback.addr, c.readback * 4),
            (s.h.addr, rows * hid * 2),
            (s.x.addr, rows * hid * 2),
            (s.mix.addr, rows * hid * 2),
            (s.m.addr, rows * hid * 2),
            (s.normed.addr, pad4(c.rows) * hid * 2),
            (s.codes.addr, rows * kmax),
            (s.scales.addr, kmax / 128 * rows * 4),
            (
                s.parts.addr,
                if fast {
                    (rows * parts_width(d) * 4).max(8)
                } else {
                    8
                },
            ),
            (s.gate_up.addr, rows * 2 * d.inter as usize * 2),
            (s.hmid.addr, rows * d.inter as usize * 2),
            (s.qkvz.addr, rows * (cd + vd) * 2),
            (s.aa.addr, rows * vh * 2),
            (s.bb.addr, rows * vh * 2),
            (s.nrm.addr, rows * vd * 2),
            (s.conv_in.addr, c.conv_rows.max(4) * cd * 4),
            (s.new_f32.addr, rows * cd * 4),
            (s.ab_new.addr, rows * 2 * vh * 4),
            (s.ab.addr, c.staged.max(1) * 2 * vh * 4),
            (s.dq.addr, c.staged.max(1) * d.key_dim() as usize * 4),
            (s.dk.addr, c.staged.max(1) * d.key_dim() as usize * 4),
            (s.dv.addr, c.staged.max(1) * vd * 4),
            (s.beta.addr, c.staged.max(1) * vh * 4),
            (s.alpha.addr, c.staged.max(1) * vh * 4),
            (s.o.addr, c.staged.max(1) * vd * 4),
            (s.o_rows.addr, rows * vd * 4),
            (s.qkv.addr, rows * (2 * qw + 2 * kvw) * 2),
            (s.q.addr, rows * qw * 2),
            (s.kr.addr, rows * kvw * 2),
            (s.attn.addr, rows * qw * 2),
            (s.y.addr, rows * qw * 2),
            (
                s.scores.addr,
                if fast {
                    8
                } else {
                    rows * hq * c.score_stride * 4
                },
            ),
            (
                s.part_v.addr,
                if fast {
                    c.partial_rows * hq * HEAD_DIM as usize * 2
                } else {
                    8
                },
            ),
            (
                s.part_s.addr,
                if fast { c.partial_rows * hq * 4 } else { 8 },
            ),
            (s.d_tokens.addr, c.drafter.max(1) * 4),
            (s.d_e.addr, rows * hid * 2),
            (s.d_hid.addr, rows * hid * 2),
            (s.d_cat.addr, rows * 2 * hid * 2),
            (s.d_h.addr, rows * hid * 2),
            (s.d_normed.addr, rows * hid * 2),
            (s.head_in.addr, pad4(c.head) * hid * 2),
            (s.head_tmp.addr, if fast { head_tmp(c, d) } else { 8 }),
            (s.logits_t.addr, c.logits_t.max(1) * vocab * 4),
            (s.logits_d.addr, c.logits_d.max(1) * vocab * 4),
            (s.p_plain.addr, c.plain.max(1) * vocab * 8),
            (s.p_target.addr, c.targets.max(1) * vocab * 8),
            (s.q_draft.addr, (c.drafting * c.depth).max(1) * vocab * 8),
            (s.accept_scratch.addr, accept_work_bytes(c.drafting)),
            (s.sample_work.addr, sample_work_bytes(sample_rows(c))),
            (s.lanes.addr, (c.drafting * c.depth).max(1) * 4),
            (s.pad_c.addr, c.pad_conv.max(1) * cd * 4),
            (s.pad_a.addr, c.pad_ab.max(1) * 2 * vh * 4),
            (s.pad_o.addr, c.pad_o.max(1) * vd * 4),
            (s.pad_h.addr, c.pad_h.max(1) * hid * 2),
            (s.pad_t.addr, c.pad_t.max(1) * 4),
        ]
    }

    /// The widest K of an FP8 linear.
    pub(super) fn k_max(&self) -> usize {
        let d = &self.dims;
        [d.hidden, d.inter, d.q_heads * d.head_dim, d.value_dim()]
            .into_iter()
            .max()
            .unwrap_or(0) as usize
    }

    fn tw(&self, a: super::super::plan::Arr) -> u64 {
        self.table_base.get() + 4 * a.at as u64
    }

    /// One row-copy launch over `bufs` (address, rows) of `width` words.
    fn copies(&self, at: &ItemsAt, width: usize, bufs: &[(u64, u64)]) -> Result<()> {
        self.copy_groups(&[(at, width, bufs)])
    }

    /// One row-copy launch over up to [`COPY_GROUPS`] groups, each its items over its
    /// buffers (address, rows) of its width in words.
    fn copy_groups(&self, groups: &[CopyGroupAt<'_>]) -> Result<()> {
        let mut p = CopyParams {
            status: self.status_addr(),
            ..CopyParams::default()
        };
        assert!(groups.len() <= COPY_GROUPS);
        for &(at, width, bufs) in groups {
            if at.items == 0 {
                continue;
            }
            let g = &mut p.group[p.groups as usize];
            *g = CopyGroup {
                src_buf: self.tw(at.src_buf),
                src_row: self.tw(at.src_row),
                dst_buf: self.tw(at.dst_buf),
                dst_row: self.tw(at.dst_row),
                width: u32::try_from(width).map_err(|_| CudaError::new("copy width"))?,
                items: u32::try_from(at.items).map_err(|_| CudaError::new("copy items"))?,
                ..CopyGroup::default()
            };
            assert!(bufs.len() <= COPY_BUFFERS);
            for (i, &(addr, rows)) in bufs.iter().enumerate() {
                g.buf[i] = addr;
                g.rows[i] = rows;
            }
            p.groups += 1;
        }
        if p.groups == 0 {
            return Ok(());
        }
        // SAFETY: every buffer holds `rows · width` words; the items are bounded by the
        // kernel against them.
        unsafe { self.dev.copy_rows(&p) }
    }

    /// The read-back's status word: its first.
    pub(super) fn status_addr(&self) -> u64 {
        self.scratch.readback.addr
    }

    fn words(bytes: usize, row: usize) -> u64 {
        (bytes / row) as u64
    }

    /// The conv-width buffers at DeltaNet layer `dn`.
    fn conv_bufs(&self, dn: usize) -> [(u64, u64); 7] {
        let s = &self.scratch;
        let cd4 = self.dims.conv_dim() as usize * 4;
        let off = dn * self.layer_bytes();
        let states = self.cfg.state_buffers as usize * self.image_stride;
        let live = self.slot_count() * self.image_stride;
        let tail = self.tail_layer_rows() * cd4;
        [
            (
                self.states.addr + off as u64,
                Self::words(states - off, cd4),
            ),
            (self.live.addr + off as u64, Self::words(live - off, cd4)),
            (
                self.tails_conv.addr + (dn * tail) as u64,
                self.tail_layer_rows() as u64,
            ),
            (s.conv_in.addr, s.caps.conv_rows.max(4) as u64),
            (s.new_f32.addr, pad4(s.caps.rows.max(s.caps.drafter)) as u64),
            (s.zero_rows.addr, 3),
            (s.pad_c.addr, s.caps.pad_conv as u64),
        ]
    }

    fn ab_bufs(&self, dn: usize) -> [(u64, u64); 4] {
        let s = &self.scratch;
        let row = 2 * self.dims.v_heads as usize * 4;
        [
            (
                self.tails_ab.addr + (dn * self.tail_layer_rows() * row) as u64,
                self.tail_layer_rows() as u64,
            ),
            (s.ab.addr, s.caps.staged.max(1) as u64),
            (s.ab_new.addr, pad4(s.caps.rows.max(s.caps.drafter)) as u64),
            (s.pad_a.addr, s.caps.pad_ab as u64),
        ]
    }

    fn hid_bufs(&self, dst: u64, dst_rows: usize) -> [(u64, u64); 6] {
        let s = &self.scratch;
        let c = &s.caps;
        let (taps, tap_rows) = match self.pools.get(DRAFTER).and_then(|p| p.taps.as_ref()) {
            Some(t) => (t.addr, self.cfg.num_blocks as u64 + 1),
            None => (0, 0),
        };
        let k1 = self.cfg.draft_tokens as usize + 1;
        [
            (self.hiddens.addr, (self.slot_count() * k1) as u64),
            (taps, tap_rows),
            (s.normed.addr, pad4(c.rows) as u64),
            (s.d_normed.addr, pad4(c.rows.max(c.drafter)) as u64),
            (dst, dst_rows as u64),
            (s.pad_h.addr, c.pad_h as u64),
        ]
    }

    fn tok_bufs(&self, l: &Layout, dst: u64, dst_rows: usize) -> [(u64, u64); 4] {
        let s = &self.scratch;
        [
            (self.tw(l.target.tokens), l.target.tokens.len as u64),
            (s.lanes.addr, (s.caps.drafting * s.caps.depth) as u64),
            (dst, dst_rows as u64),
            (s.pad_t.addr, s.caps.pad_t as u64),
        ]
    }

    /// The whole step over the uploaded table (`l`). Nothing in it reads back.
    pub(super) fn program(&self, l: &Layout) -> Result<()> {
        let s = &self.scratch;
        // SAFETY: the read-back's status word.
        unsafe { self.dev.zero(self.status_addr(), 4)? };
        let hsz = self.dims.hidden as usize;
        for (j, d) in l.lead.iter().enumerate() {
            self.drafter_forward(l, d)?;
            if d.head.items > 0 {
                self.copies(
                    &d.head,
                    hsz / 2,
                    &self.hid_bufs(s.head_in.addr, pad4(s.caps.head)),
                )?;
                self.head(d.head.items, s.logits_d.addr)?;
            }
            let draws = d.draws.len / SAMPLE_WORDS;
            if draws > 0 {
                let off = j * l.drafting as usize;
                self.sample(SampleArgs {
                    logits: s.logits_d.addr,
                    logits_stride: u64::from(self.dims.vocab),
                    n: self.spec.sampleable_vocab_size,
                    rows: self.tw(d.draws),
                    num_rows: n32(draws),
                    draw: Some(Stream::Draft),
                    probs: s.q_draft.addr + (off * self.n_sample() * 8) as u64,
                    tokens: s.lanes.addr + (off * 4) as u64,
                    status: self.status_addr(),
                    work: s.sample_work.addr,
                })?;
            }
        }
        self.target_forward(l, &l.target, s.normed.addr)?;
        self.copies(
            &l.head,
            hsz / 2,
            &self.hid_bufs(s.head_in.addr, pad4(s.caps.head)),
        )?;
        self.head(l.head.items, s.logits_t.addr)?;
        if let Some(d) = &l.trail {
            self.drafter_forward(l, d)?;
        }
        let plain = l.plain.len / SAMPLE_WORDS;
        self.sample(SampleArgs {
            logits: s.logits_t.addr,
            logits_stride: u64::from(self.dims.vocab),
            n: self.spec.sampleable_vocab_size,
            rows: self.tw(l.plain),
            num_rows: n32(plain),
            draw: Some(Stream::Sample),
            probs: s.p_plain.addr,
            tokens: s.readback.addr + 4,
            status: self.status_addr(),
            work: s.sample_work.addr,
        })?;
        if l.accepts > 0 {
            self.sample(SampleArgs {
                logits: s.logits_t.addr,
                logits_stride: u64::from(self.dims.vocab),
                n: self.spec.sampleable_vocab_size,
                rows: self.tw(l.targets),
                num_rows: n32(l.targets.len / SAMPLE_WORDS),
                draw: None,
                probs: s.p_target.addr,
                tokens: 0,
                status: self.status_addr(),
                work: s.sample_work.addr,
            })?;
            let k1 = l.width as usize + 1;
            let out = s.readback.addr + 4 + (plain * 4) as u64;
            let a = AcceptLaunch {
                target: s.p_target.addr,
                draft: s.q_draft.addr,
                n: self.spec.sampleable_vocab_size,
                rows: self.tw(l.accept_rows),
                target_row: self.tw(l.accept_target),
                draft_row: self.tw(l.accept_draft),
                draft_step: l.drafting,
                num_drafts: self.tw(l.accept_drafts),
                drafts: s.lanes.addr,
                stride: n32(k1),
                scratch: s.accept_scratch.addr,
                out,
                counts: out + (l.accepts * k1 * 4) as u64,
                status: self.status_addr(),
                num_rows: n32(l.accepts),
            };
            // SAFETY: the planner's rows, inside the distributions and lanes.
            unsafe { self.dev.accept(&a)? };
        }
        if let Some(r) = &l.rewrite {
            self.drafter_forward(l, r)?;
        }
        let taps = self
            .pools
            .get(DRAFTER)
            .and_then(|p| p.taps.as_ref())
            .map_or(0, |t| t.addr);
        if l.keep.items > 0 {
            self.copies(
                &l.keep,
                hsz / 2,
                &self.hid_bufs(taps, self.cfg.num_blocks as usize + 1),
            )?;
        }
        Ok(())
    }

    fn n_sample(&self) -> usize {
        self.spec.sampleable_vocab_size as usize
    }

    fn sample(&self, a: SampleArgs) -> Result<()> {
        if a.num_rows == 0 {
            return Ok(());
        }
        // SAFETY: the planner's rows read logit rows the heads wrote; the distributions
        // and tokens are the scratch sized for the capacities.
        unsafe { self.dev.sample(&a) }
    }

    /// The head over `rows` rows of the head's input into `out` (f32 logits).
    fn head(&self, rows: usize, out: u64) -> Result<()> {
        let d = &self.dims;
        let fast =
            (self.dev.profile() == super::QwenProfile::Fast).then_some(self.scratch.head_tmp.addr);
        let p = LinearParams {
            x: self.scratch.head_in.addr,
            w: self.weights.head,
            out,
            rows: n32(rows),
            n: d.vocab,
            k: d.hidden,
            x_stride: d.hidden,
            out_stride: d.vocab,
            out_f32: 1,
        };
        // SAFETY: `rows` gathered rows; `out` holds `rows · vocab` f32.
        unsafe { self.dev.linear(&p, fast) }
    }

    /// The residual norm of `rows` rows of `x` (`n` wide) into `out`, the residual add's
    /// input `add`; with `quant`, its normed rows also quantised into the codes and scales
    /// scratch, as [`Self::quant`] would, for the FP8 GEMM that reads them
    /// ([`Self::gemm8`]).
    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        x: u64,
        add: Residual,
        w: u64,
        out: u64,
        rows: usize,
        n: u32,
        quant: bool,
    ) -> Result<()> {
        let mut p = NormParams {
            x,
            weight: w,
            out,
            rows: n32(rows),
            units: 1,
            n,
            x_stride: n,
            add_stride: n,
            out_stride: n,
            eps: self.dims.eps,
            ..NormParams::default()
        };
        match add {
            Residual::None => {}
            Residual::Bf16(a) => p.add = a,
            Residual::Parts { splits, part_rows } => {
                p.add_parts = self.scratch.parts.addr;
                p.add_splits = splits;
                p.add_part_rows = part_rows;
            }
        }
        if quant {
            p.codes = self.scratch.codes.addr;
            p.scales = self.scratch.scales.addr;
            p.m_pad = n32(pad4(rows));
        }
        // SAFETY: `rows · n` BF16 values at each; the partials are the parts scratch's,
        // the codes and scales the quantiser's scratch, which hold the capacity's rows.
        unsafe { self.dev.norm(&p) }
    }

    /// The FP8 linear of `rows` rows of `x` by `w` into `out`: the quantiser, then
    /// [`Self::gemm8`].
    fn lin8(&self, x: u64, rows: usize, w: &super::super::weights::Fp8W, out: u64) -> Result<()> {
        self.quant(x, rows, w.k)?;
        self.gemm8(rows, w, out)
    }

    /// The FP8 linear of `rows` rows of `x` by `w` whose output only the next norm's
    /// residual add reads: in the fast profile a split shape's partial sums, left in the
    /// parts scratch for that norm to add; otherwise its BF16 rows in `out`.
    fn lin8_residual(
        &self,
        x: u64,
        rows: usize,
        w: &super::super::weights::Fp8W,
        out: u64,
    ) -> Result<Residual> {
        let splits = fp8_splits(w.n, w.k);
        if self.dev.profile() == super::QwenProfile::Exact || splits < 2 {
            self.lin8(x, rows, w, out)?;
            return Ok(Residual::Bf16(out));
        }
        self.quant(x, rows, w.k)?;
        let l = self.lin8_params(rows, w, out);
        // SAFETY: the codes and scales of `m_pad` rows; the parts scratch holds
        // `parts_width` f32 a row.
        unsafe { self.dev.lin8_parts(&l, self.scratch.parts.addr)? };
        Ok(Residual::Parts {
            splits,
            part_rows: l.m_pad,
        })
    }

    /// `rows` rows of `x`, `k` wide, quantised into the codes and scales scratch.
    fn quant(&self, x: u64, rows: usize, k: u32) -> Result<()> {
        let q = QuantParams {
            x,
            codes: self.scratch.codes.addr,
            scales: self.scratch.scales.addr,
            rows: n32(rows),
            k,
            x_stride: k,
            m_pad: n32(pad4(rows)),
        };
        // SAFETY: the quantiser's scratch holds the widest K at the capacity's rows.
        unsafe { self.dev.quant(&q) }
    }

    /// The FP8 GEMM of the `rows` rows [`Self::quant`] left by `w` into `out`, the split
    /// shapes' partial sums through the parts scratch.
    fn gemm8(&self, rows: usize, w: &super::super::weights::Fp8W, out: u64) -> Result<()> {
        let l = self.lin8_params(rows, w, out);
        let parts =
            (self.dev.profile() == super::QwenProfile::Fast).then_some(self.scratch.parts.addr);
        // SAFETY: the codes and scales of `m_pad` rows; `out` holds `m_pad` rows (the fast
        // forms write the padding rows); the parts scratch holds `parts_width` f32 a row.
        unsafe { self.dev.lin8(&l, parts) }
    }

    /// The FP8 linear of the quantiser's scratch by `w` into `out`.
    fn lin8_params(&self, rows: usize, w: &super::super::weights::Fp8W, out: u64) -> Lin8Params {
        let m_pad = n32(pad4(rows));
        Lin8Params {
            a: self.scratch.codes.addr,
            a_scales: self.scratch.scales.addr,
            w: w.codes,
            w_scales: w.scales,
            out,
            rows: n32(rows),
            n: w.n,
            k: w.k,
            m_pad,
            out_stride: w.n,
            reserved: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        x: u64,
        rows: usize,
        w: u64,
        n: u32,
        k: u32,
        out: u64,
        large: bool,
    ) -> Result<()> {
        let p = LinearParams {
            x,
            w,
            out,
            rows: n32(rows),
            n,
            k,
            x_stride: k,
            out_stride: n,
            out_f32: 0,
        };
        let fast = (large && self.dev.profile() == super::QwenProfile::Fast)
            .then_some(self.scratch.head_tmp.addr);
        // SAFETY: the caller's buffers; the f32 scratch holds the head's rows, more than
        // the drafter's input projection's.
        unsafe { self.dev.linear(&p, fast) }
    }

    /// The DeltaNet's `a` and `b` projections of `rows` rows. Exact: each by the exact
    /// linear to BF16 (`aa`, `bb`), widened into `ab_new` by [`Self::deltanet`]. Fast: the
    /// stacked weight by the BF16 GEMM into the f32 scratch, then its first `2 · v_heads`
    /// columns rounded to BF16 values straight into `ab_new` (`a`'s then `b`'s, the
    /// widening's layout).
    fn ab_projections(&self, x: u64, rows: usize, w: &DeltaNetW) -> Result<()> {
        let s = &self.scratch;
        let d = &self.dims;
        if self.dev.profile() == super::QwenProfile::Exact {
            self.linear(x, rows, w.a, d.v_heads, d.hidden, s.aa.addr, false)?;
            return self.linear(x, rows, w.b, d.v_heads, d.hidden, s.bb.addr, false);
        }
        let g = LinearParams {
            x,
            w: w.ab,
            out: s.head_tmp.addr,
            rows: n32(rows),
            n: w.ab_n,
            k: d.hidden,
            x_stride: d.hidden,
            out_stride: w.ab_n,
            out_f32: 1,
        };
        let r = RoundParams {
            x: s.head_tmp.addr,
            out: s.ab_new.addr,
            rows: n32(rows),
            width: 2 * d.v_heads,
            x_stride: w.ab_n,
            out_stride: 2 * d.v_heads,
            out_f32: 1,
            reserved: 0,
        };
        // SAFETY: `rows` rows of `x`; the f32 scratch holds `rows · ab_n` (`head_tmp`),
        // `ab_new` `rows · 2 · v_heads`.
        unsafe {
            self.dev.gemm_bf16(&g)?;
            self.dev.round(&r)
        }
    }

    fn widen(
        &self,
        x: u64,
        x_stride: u32,
        out: u64,
        out_stride: u32,
        rows: usize,
        width: u32,
    ) -> Result<()> {
        let p = WidenParams {
            x,
            out,
            rows: n32(rows),
            width,
            x_stride,
            out_stride,
        };
        // SAFETY: the caller's buffers.
        unsafe { self.dev.widen(&p) }
    }

    /// The MLP of `rows` rows whose input the norm before it quantised (`x` normed), its
    /// down projection for the next norm's residual add ([`Self::lin8_residual`], `m`).
    fn mlp(&self, l: &LayerW, rows: usize, m: u64) -> Result<Residual> {
        let s = &self.scratch;
        let inter = self.dims.inter;
        self.gemm8(rows, &l.gate_up, s.gate_up.addr)?;
        let p = GateParams {
            a: s.gate_up.addr + u64::from(inter) * 2,
            g: s.gate_up.addr,
            out: s.hmid.addr,
            rows: n32(rows),
            width: inter,
            a_stride: 2 * inter,
            g_stride: 2 * inter,
            out_stride: inter,
            unit: inter,
            g_unit_stride: 0,
            kind: GATE_SWIGLU,
        };
        // SAFETY: `rows · inter` values in each.
        unsafe { self.dev.gate(&p)? };
        self.lin8_residual(s.hmid.addr, rows, &l.down, m)
    }

    /// One DeltaNet layer's staging, recurrence and keeping, over the pass's runs.
    fn deltanet(
        &self,
        at: &TargetAt,
        dn: usize,
        w: &super::super::weights::DeltaNetW,
        new: bool,
    ) -> Result<()> {
        let s = &self.scratch;
        let d = &self.dims;
        let (cd, vh, vd) = (
            d.conv_dim() as usize,
            d.v_heads as usize,
            d.value_dim() as usize,
        );
        let t = at.t();
        if new && t > 0 {
            let two = n32(2 * vh);
            self.widen(
                s.qkvz.addr,
                d.conv_dim() + d.value_dim(),
                s.new_f32.addr,
                d.conv_dim(),
                t,
                d.conv_dim(),
            )?;
            if self.dev.profile() == super::QwenProfile::Exact {
                self.widen(s.aa.addr, d.v_heads, s.ab_new.addr, two, t, d.v_heads)?;
                self.widen(
                    s.bb.addr,
                    d.v_heads,
                    s.ab_new.addr + (vh * 4) as u64,
                    two,
                    t,
                    d.v_heads,
                )?;
            }
        }
        let (conv, ab) = (self.conv_bufs(dn), self.ab_bufs(dn));
        self.copy_groups(&[(&at.stage_conv, cd, &conv), (&at.stage_ab, 2 * vh, &ab)])?;
        let prep = DnPrepParams {
            conv_in: s.conv_in.addr,
            window: self.tw(at.window),
            ab: s.ab.addr,
            conv_w: w.conv,
            exp_a_log: w.exp_a_log,
            dt_bias: w.dt_bias,
            q: s.dq.addr,
            k: s.dk.addr,
            v: s.dv.addr,
            beta: s.beta.addr,
            alpha: s.alpha.addr,
            positions: n32(at.staged()),
            k_heads: d.k_heads,
            v_heads: d.v_heads,
            k_dim: d.k_dim,
            v_dim: d.v_dim,
            reserved: 0,
        };
        let rec = DnRecParams {
            seqs: self.tw(at.seqs),
            q: s.dq.addr,
            k: s.dk.addr,
            v: s.dv.addr,
            beta: s.beta.addr,
            alpha: s.alpha.addr,
            out: s.o.addr,
            state_offset: (dn * self.layer_bytes() + 3 * cd * 4) as u64,
            seq_count: n32(at.runs),
            k_heads: d.k_heads,
            v_heads: d.v_heads,
            k_dim: d.k_dim,
            v_dim: d.v_dim,
            reserved: 0,
        };
        // SAFETY: the staged arrays hold the shape's positions; every state address in the
        // sequences is a live image or a state buffer.
        // The short form when the runs are a few positions each (a decode step, a draft
        // check): both compute the same bits, so the choice is the launch's alone.
        let short = at.staged() <= DN_SHORT_RUN * at.runs;
        unsafe {
            self.dev.dn_prep(&prep)?;
            if short {
                self.dev.dn_rec_short(&rec)?;
            } else {
                self.dev.dn_rec(&rec)?;
            }
        }
        let o = [
            (s.o.addr, at.staged().max(1) as u64),
            (s.o_rows.addr, pad4(s.caps.rows.max(s.caps.drafter)) as u64),
            (s.pad_o.addr, s.caps.pad_o as u64),
        ];
        let none = ItemsAt::default();
        self.copy_groups(&[
            (&at.post_conv, cd, &conv),
            (&at.post_ab, 2 * vh, &ab),
            (if t > 0 { &at.o_rows } else { &none }, vd, &o),
        ])?;
        Ok(())
    }

    /// A fold alone over every DeltaNet layer (`SaveState`).
    pub(super) fn fold_program(&self, at: &TargetAt) -> Result<()> {
        let mut dn = 0;
        for lw in &self.weights.layers {
            if let MixerW::DeltaNet(w) = &lw.mixer {
                self.deltanet(at, dn, w, false)?;
                dn += 1;
            }
        }
        Ok(())
    }

    /// The target over a pass's rows into `normed`.
    fn target_forward(&self, l: &Layout, at: &TargetAt, normed: u64) -> Result<()> {
        let s = &self.scratch;
        let d = self.dims.clone();
        let t = at.t();
        if at.token_copies.items > 0 {
            let dst = self.tw(at.tokens);
            self.copies(&at.token_copies, 1, &self.tok_bufs(l, dst, t))?;
        }
        self.embed(self.tw(at.tokens), t, s.h.addr)?;
        let (mut dn, mut layer) = (0usize, 0u32);
        let mut prev = Residual::None;
        for lw in &self.weights.layers {
            self.norm(s.h.addr, prev, lw.input_norm, s.x.addr, t, d.hidden, true)?;
            let mix = match &lw.mixer {
                MixerW::DeltaNet(w) => {
                    self.gemm8(t, &w.qkvz, s.qkvz.addr)?;
                    self.ab_projections(s.x.addr, t, w)?;
                    self.deltanet(at, dn, w, true)?;
                    let p = DnNormParams {
                        o: s.o_rows.addr,
                        z: s.qkvz.addr + u64::from(d.conv_dim()) * 2,
                        weight: w.norm,
                        out: s.nrm.addr,
                        rows: n32(t),
                        v_heads: d.v_heads,
                        v_dim: d.v_dim,
                        z_stride: d.conv_dim() + d.value_dim(),
                        eps: d.eps,
                        reserved: 0,
                    };
                    // SAFETY: `t` rows of each.
                    unsafe { self.dev.dn_norm(&p)? };
                    dn += 1;
                    self.lin8_residual(s.nrm.addr, t, &w.out, s.mix.addr)?
                }
                MixerW::Attention(w) => {
                    layer += 1;
                    self.attention_mix(w, t, TARGET, layer - 1, 0, &at.attn, s.mix.addr, false)?
                }
            };
            self.norm(s.h.addr, mix, lw.post_norm, s.x.addr, t, d.hidden, true)?;
            prev = self.mlp(lw, t, s.m.addr)?;
        }
        self.norm(
            s.h.addr,
            prev,
            self.weights.final_norm,
            normed,
            t,
            d.hidden,
            false,
        )
    }

    fn embed(&self, tokens: u64, rows: usize, out: u64) -> Result<()> {
        let p = EmbedParams {
            tokens,
            table: self.weights.embed,
            out,
            status: self.status_addr(),
            rows: n32(rows),
            width: self.dims.hidden,
            vocab: self.dims.vocab,
            out_stride: self.dims.hidden,
        };
        // SAFETY: `rows` tokens and output rows.
        unsafe { self.dev.embed(&p) }
    }

    /// One attention layer's mixer over `rows` normed rows, which the norm before it
    /// quantised, its output projection for the next norm's residual add
    /// ([`Self::lin8_residual`], `mix`); with `kv_only`, only the KV write.
    #[allow(clippy::too_many_arguments)]
    fn attention_mix(
        &self,
        w: &AttentionW,
        rows: usize,
        group: usize,
        layer: u32,
        key_first: u32,
        at: &AttnAt,
        mix: u64,
        kv_only: bool,
    ) -> Result<Residual> {
        let s = &self.scratch;
        let d = &self.dims;
        let (hq, hd, kvw) = (d.q_heads, d.head_dim, d.kv_width());
        let qw = hq * hd;
        let r = n32(rows);
        // One GEMM over the stacked projections, the passes that write KV alone too, so
        // every pass computes a position's keys and values by the same GEMM.
        self.gemm8(rows, &w.qkv, s.qkv.addr)?;
        let stride = 2 * qw + 2 * kvw;
        let (kk, vv) = (
            s.qkv.addr + u64::from(2 * qw) * 2,
            s.qkv.addr + u64::from(2 * qw + kvw) * 2,
        );
        let rope = self.tw(at.rope);
        let rot =
            |src: u64, units: u32, stride: u32, unit_stride: u32, out: u64, wt: u64| NormParams {
                x: src,
                weight: wt,
                out,
                rope_cos: self.weights.rope_cos,
                rope_sin: self.weights.rope_sin,
                positions: rope,
                rows: r,
                units,
                n: hd,
                x_stride: stride,
                x_unit_stride: unit_stride,
                out_stride: units * hd,
                out_unit_stride: hd,
                rotary: d.rotary,
                eps: d.eps,
                ..NormParams::default()
            };
        let pool = &self.pools[group];
        let kvp = KvWriteParams {
            k: s.kr.addr,
            v: vv,
            pool: pool.data.addr,
            targets: self.tw(at.targets),
            rows: r,
            width: kvw,
            k_stride: kvw,
            v_stride: stride,
            layer,
            layers: pool.layers,
            block_size: self.cfg.block_size,
            reserved: 0,
        };
        // SAFETY: every buffer holds the capacity's rows; the targets are blocks the rows'
        // slots alone map, or the pad block.
        unsafe {
            if !kv_only {
                self.dev
                    .norm(&rot(s.qkv.addr, hq, stride, 2 * hd, s.q.addr, w.q_norm))?;
            }
            self.dev
                .norm(&rot(kk, d.kv_heads, stride, hd, s.kr.addr, w.k_norm))?;
            self.dev.kv_write(&kvp)?;
        }
        if kv_only {
            return Ok(Residual::None);
        }
        match at.fa2.as_slice() {
            [] => {
                let ap = AttnParams {
                    q: s.q.addr,
                    pool: pool.data.addr,
                    tables: self.tw(at.tables),
                    positions: self.tw(at.query),
                    out: s.attn.addr,
                    scores: s.scores.addr,
                    rows: r,
                    q_heads: hq,
                    kv_heads: d.kv_heads,
                    head_dim: hd,
                    q_stride: qw,
                    q_head_stride: hd,
                    out_stride: qw,
                    table_stride: n32(at.table_width),
                    layer,
                    layers: pool.layers,
                    block_size: self.cfg.block_size,
                    score_stride: n32(s.caps.score_stride),
                    key_first,
                    scale: d.scale,
                };
                // SAFETY: the tables cover every visible position; the scores hold the
                // capacity's rows at the fixed stride.
                unsafe { self.dev.attention(&ap)? };
            }
            // FA2 pass by pass: each merges its own rows from its first one on.
            passes => {
                for f in passes {
                    self.fa2(f, group, layer)?;
                }
            }
        }
        let gp = GateParams {
            a: s.attn.addr,
            g: s.qkv.addr + u64::from(hd) * 2,
            out: s.y.addr,
            rows: r,
            width: qw,
            a_stride: qw,
            g_stride: stride,
            out_stride: qw,
            unit: hd,
            g_unit_stride: 2 * hd,
            kind: GATE_SIGMOID,
        };
        // SAFETY: `rows` rows of each.
        unsafe { self.dev.gate(&gp)? };
        self.lin8_residual(s.y.addr, rows, &w.o, mix)
    }

    fn fa2(&self, f: &Fa2At, group: usize, layer: u32) -> Result<()> {
        let s = &self.scratch;
        let d = &self.dims;
        let pool = &self.pools[group];
        let w = u64::from(d.kv_width());
        let bs = u64::from(self.cfg.block_size);
        let el = |e: u64| u32::try_from(e).map_err(|_| CudaError::new("a page stride past u32"));
        let layer_at = if group == TARGET {
            let k_base = pool.data.addr + u64::from(layer) * 2 * bs * w * 2;
            AttnLayer {
                k_base,
                v_base: k_base + bs * w * 2,
                k_page_stride: el(u64::from(pool.layers) * 2 * bs * w)?,
                v_page_stride: el(u64::from(pool.layers) * 2 * bs * w)?,
                num_kv_heads: d.kv_heads,
                page_size: self.cfg.block_size,
                window_left: -1,
                sink: s.sink.addr,
            }
        } else {
            AttnLayer {
                k_base: pool.data.addr,
                v_base: pool.data.addr + bs * w * 2,
                k_page_stride: el(w)?,
                v_page_stride: el(w)?,
                num_kv_heads: d.kv_heads,
                page_size: 1,
                window_left: -1,
                sink: s.sink.addr,
            }
        };
        let view = PlanView {
            shape: f.caps.shape,
            out_row: f.out_row,
            q_indptr: self.tw(f.q_indptr),
            indices: self.tw(f.indices),
            indptr: self.tw(f.indptr),
            last_page_len: self.tw(f.last_page_len),
            request_indices: self.tw(f.request_indices),
            qo_tile_indices: self.tw(f.qo_tile_indices),
            kv_tile_indices: self.tw(f.kv_tile_indices),
            o_indptr: self.tw(f.o_indptr),
            merge_indptr: self.tw(f.merge_indptr),
            work_items: self.tw(f.counts),
            kv_chunk_size: self.tw(f.counts) + 4,
        };
        let a = Fa2Args {
            view,
            layer: layer_at,
            num_qo_heads: d.q_heads,
            q: s.q.addr,
            o: s.attn.addr,
            partials: Partials {
                v: s.part_v.addr,
                s: s.part_s.addr,
                rows: n32(s.caps.partial_rows),
                heads: d.q_heads,
            },
            scale: d.scale,
        };
        debug_assert_eq!(PARTIAL_BYTES_PER_HEAD, HEAD_DIM as usize * 2 + 4);
        // SAFETY: the table holds a plan padded to this reservation; the pool covers its
        // pages (the pad block included).
        unsafe { self.dev.fa2(&a) }
    }

    /// The drafter's layer over a pass's rows: its KV written at each row's position and,
    /// unless KV only, its normed rows.
    fn drafter_forward(&self, l: &Layout, at: &DrafterAt) -> Result<()> {
        let mtp = self.weights.mtp.as_ref().expect("a drafter");
        let s = &self.scratch;
        let d = self.dims.clone();
        let n = at.attn.rows;
        if n == 0 {
            return Ok(());
        }
        let hsz = d.hidden as usize;
        self.copies(
            &at.token_copies,
            1,
            &self.tok_bufs(l, s.d_tokens.addr, s.caps.drafter),
        )?;
        self.copies(
            &at.hidden_copies,
            hsz / 2,
            &self.hid_bufs(s.d_hid.addr, pad4(s.caps.drafter)),
        )?;
        self.embed(s.d_tokens.addr, n, s.d_e.addr)?;
        let half = |src: u64, w: u64, off: u64| NormParams {
            x: src,
            weight: w,
            out: s.d_cat.addr + off,
            rows: n32(n),
            units: 1,
            n: d.hidden,
            x_stride: d.hidden,
            out_stride: 2 * d.hidden,
            eps: d.eps,
            ..NormParams::default()
        };
        // SAFETY: `n` rows of each.
        unsafe {
            self.dev
                .norm(&half(s.d_e.addr, mtp.pre_fc_norm_embedding, 0))?;
            self.dev.norm(&half(
                s.d_hid.addr,
                mtp.pre_fc_norm_hidden,
                (hsz * 2) as u64,
            ))?;
        }
        self.linear(
            s.d_cat.addr,
            n,
            mtp.fc,
            d.hidden,
            2 * d.hidden,
            s.d_h.addr,
            true,
        )?;
        let lw = &mtp.layer;
        let MixerW::Attention(w) = &lw.mixer else {
            unreachable!("the drafter's layer is an attention layer")
        };
        self.norm(
            s.d_h.addr,
            Residual::None,
            lw.input_norm,
            s.x.addr,
            n,
            d.hidden,
            true,
        )?;
        let mix = self.attention_mix(w, n, DRAFTER, 0, 1, &at.attn, s.mix.addr, at.kv_only)?;
        if at.kv_only {
            return Ok(());
        }
        self.norm(s.d_h.addr, mix, lw.post_norm, s.x.addr, n, d.hidden, true)?;
        let m = self.mlp(lw, n, s.m.addr)?;
        self.norm(s.d_h.addr, m, mtp.norm, s.d_normed.addr, n, d.hidden, false)
    }
}

/// What a residual norm adds: nothing, BF16 rows, or the split FP8 GEMM's partial sums
/// in the parts scratch.
#[derive(Clone, Copy, Debug)]
enum Residual {
    None,
    Bf16(u64),
    Parts { splits: u32, part_rows: u32 },
}

fn n32(x: usize) -> u32 {
    u32::try_from(x).expect("a count within the kernels' 32-bit range")
}
