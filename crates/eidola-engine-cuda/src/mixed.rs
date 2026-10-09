//! CUDA-graph replay of mixed steps.
//!
//! A step that is not pure decode (prefill chunks or extends, alone or beside
//! decode rows) runs as one captured graph per rung of a ladder of **token
//! counts** ([`mixed_ladder`]) when every target group's eager work list
//! would use the 128-row query tile ([`MIXED_TILE`]: on Flash, a step whose
//! longest row has at least nine host tokens). The step is padded to its
//! rung's tokens and everything it varies is data in one step table, filled
//! by one host-to-device copy ([`MixedLayout`], [`pack`]), exactly as for
//! the decode graphs ([`crate::graph`]); the launches are the eager forward's
//! ([`GpuModel::launch`]) over the rung's token count, then the head over a
//! fixed number of logit rows and the sampler over as many sample rows.
//!
//! **Attention's launch shape is fixed by the rung.** A step's eager work
//! list varies in its number of requests and work items with the step's
//! composition; the graph launches the 128-row tile over a fixed number of
//! work items per group ([`work_cap`]) and a fixed request count (the rung's
//! tokens: every request has at least one query row), and marks the work
//! items the step does not use in FlashInfer's `block_valid_mask`, which the
//! kernel reads before anything else and returns on (the mechanism
//! FlashInfer's own CUDA-graph mode uses). Request slots past the step's
//! requests repeat the last `q_indptr` and `indptr` entries, so the kernel's
//! page bound (`indptr[batch_size]`) is the step's page count. The real
//! requests' work items come first, in the eager plan's order and at the
//! eager plan's tile (a step whose eager tile differs is not replayed), so
//! every real CTA computes what it computes eagerly.
//!
//! **Padding is inert by construction**, as in [`crate::graph`]: a padding
//! token is [`PAD_TOKEN`] at [`PAD_POSITION`], one request of one query over
//! the one position of the pad block it wrote ([`pad_request`]), so it
//! writes and reads only the pad blocks; every other launch is row-local;
//! padding logit rows are token row 0 and padding sample rows are greedy over
//! real logits row 0 ([`pad_sample`]), never returned.
//!
//! **Numerics.** Each real token runs the kernels its eager step runs, over
//! the same operands: the same attention tile and work items (above); the
//! expert layout is the masked one exactly when the eager step's would be
//! (the ladder keeps 128 as a rung, [`mixed_ladder`]); the router's two
//! forms agree bit for bit; and every other launch computes a row
//! independently of the row count (row-wise kernels, GEMM rows whose
//! reduction order does not depend on M, expert rows, the combine per
//! token). `tests/graphs.rs` checks tokens and logits bit for bit across
//! replay, direct and eager on mixed steps of every kind it builds.

use cudarc::driver::{CudaGraph, CudaSlice, sys};
use eidola_engine::sampling::Stream;
use eidola_engine::spec::Bucket;

use crate::attention::{AttnRequest, HostPlan, PlanView, tile_for};
use crate::graph::{
    GroupSlots, PAD_POSITION, PAD_TOKEN, PackedStep, ProgramArgs, free_memory, pad_request,
    pad_sample, sample_row_words, split_readback,
};
use crate::launch::dptr;
use crate::model::Indirect;
use crate::sampler::{STATUS_NON_FINITE, SampleLaunch, SampleRow};
use crate::{CudaError, Gpu, Result};

/// Rungs are multiples of this many tokens.
pub const MIXED_GRAIN: u32 = 16;
/// The largest rung: a step of more tokens runs eagerly. Past a few hundred
/// tokens a step's kernels are long enough that the host's launches keep
/// ahead of the device, so replay saves little, while padding costs device
/// work in proportion.
pub const MIXED_MAX_TOKENS: u32 = 512;
/// The query tile a replayed mixed step's work lists use, in every group.
pub const MIXED_TILE: u32 = 128;
/// Words per sample row in the step table.
const SAMPLE_ROW_WORDS: usize = 8;

/// The rungs: every multiple of [`MIXED_GRAIN`] below the last bucket's token
/// capacity capped at [`MIXED_MAX_TOKENS`], then that cap itself; ascending.
///
/// Padding is under one grain past the first rung. 128 (the masked expert
/// layout's limit) is a rung whenever the cap exceeds it, so a step of at
/// most 128 tokens always gets a rung of at most 128, and the padded step
/// takes the expert layout the eager step would.
pub fn mixed_ladder(buckets: &[Bucket]) -> Vec<u32> {
    let top = buckets
        .last()
        .map_or(0, |b| b.max_tokens)
        .min(MIXED_MAX_TOKENS);
    let mut rungs: Vec<u32> = (1..)
        .map(|i| i * MIXED_GRAIN)
        .take_while(|&r| r < top)
        .collect();
    if top > 0 {
        rungs.push(top);
    }
    rungs
}

/// The rung a step of `tokens` host tokens replays on, if any: when every
/// target group's eager work list takes the 128-row tile (`tiles`, one per
/// group, from [`tile_for`]) and a rung holds the tokens.
pub fn mixed_rung(ladder: &[u32], tokens: usize, tiles: &[u32]) -> Option<usize> {
    if tiles.is_empty() || tiles.iter().any(|&t| t != MIXED_TILE) {
        return None;
    }
    crate::graph::rung_for(ladder, tokens)
}

/// Work items a group's list may need in a step of `tokens` tokens (padding
/// included), at the 128-row tile: a request of `q` queries takes `ceil(q ·
/// g / 128)` items, at most `q` when the GQA group size `g` is at most 128,
/// so a list has at most one item per token. `None` for a larger group.
pub fn work_cap(tokens: u32, group_size: u32) -> Option<usize> {
    (group_size <= MIXED_TILE).then_some(tokens as usize)
}

/// Pages a group's list may hold in a step of `tokens` tokens with at most
/// `seats` real requests: a request of `q` queries lists at most its decode
/// bound plus `ceil(q / page_size)` pages (its window, or its whole
/// sequence, extended by its own queries), so `r ≤ min(tokens, seats)` real
/// requests over `t` tokens list at most `r (max_pages + 1) + ceil(t /
/// page_size)`, and the `tokens - t` padding requests one page each:
/// `min(tokens, seats) (max_pages + 1) + tokens` bounds the sum.
pub fn pages_cap(tokens: u32, seats: u32, slots: &GroupSlots) -> usize {
    tokens.min(seats) as usize * (slots.max_pages as usize + 1) + tokens as usize
}

/// One group's arrays in a mixed step table: word offsets and fixed counts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedRegion {
    pub kv_block: usize,
    pub kv_slot: usize,
    /// `tokens + 1` words: one request slot per token.
    pub q_indptr: usize,
    pub indptr: usize,
    pub last_page_len: usize,
    pub request_indices: usize,
    pub qo_tile_indices: usize,
    pub kv_tile_indices: usize,
    /// `block_valid_mask`: one byte per work item, four to a word.
    pub valid: usize,
    /// Work items every launch of this group runs ([`work_cap`]).
    pub work_items: usize,
    pub indices: usize,
    /// Words reserved for `indices` ([`pages_cap`]).
    pub indices_cap: usize,
}

/// The step table of one mixed rung: where each array sits, in 32-bit words
/// from the table's start, laid out as [`crate::graph::DecodeLayout`]'s (the
/// fixed-size arrays first, sample rows at an even word, then each group's
/// page list, the smallest reservation first), so a step uploads one prefix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedLayout {
    /// Token rows every launch covers: the rung.
    pub tokens: u32,
    /// Logit rows the head computes and sample rows the sampler draws:
    /// `min(tokens, seats)`, the most rows a step can sample.
    pub logits: u32,
    pub token_ids: usize,
    pub positions: usize,
    pub logit_rows: usize,
    /// One word, always zero (FlashInfer reads it only when splitting KV).
    pub kv_chunk_size: usize,
    pub sample_rows: usize,
    pub groups: Vec<MixedRegion>,
    /// The table's size.
    pub words: usize,
}

impl MixedLayout {
    pub fn new(tokens: u32, seats: u32, groups: &[GroupSlots]) -> Result<MixedLayout> {
        let overflow = || CudaError::new(format!("mixed step table for {tokens} tokens overflows"));
        let t = tokens as usize;
        let logits = tokens.min(seats);
        let n = logits as usize;
        let mut at = 0usize;
        let mut place = |len: usize, align: usize| -> Result<usize> {
            let off = at.next_multiple_of(align);
            at = off.checked_add(len).ok_or_else(overflow)?;
            Ok(off)
        };
        let token_ids = place(t, 1)?;
        let positions = place(t, 1)?;
        let logit_rows = place(n, 1)?;
        let kv_chunk_size = place(1, 1)?;
        let mut fixed = Vec::with_capacity(groups.len());
        for g in groups {
            let w = work_cap(tokens, g.group_size).ok_or_else(|| {
                CudaError::new(format!(
                    "mixed step table: GQA groups of {} past the {MIXED_TILE}-row tile",
                    g.group_size
                ))
            })?;
            fixed.push((
                [
                    place(t, 1)?,
                    place(t, 1)?,
                    place(t + 1, 1)?,
                    place(t + 1, 1)?,
                    place(t, 1)?,
                    place(w, 1)?,
                    place(w, 1)?,
                    place(w, 1)?,
                    place(w.div_ceil(4), 1)?,
                ],
                w,
            ));
        }
        let sample_rows = place(n.checked_mul(SAMPLE_ROW_WORDS).ok_or_else(overflow)?, 2)?;
        let caps: Vec<usize> = groups.iter().map(|g| pages_cap(tokens, seats, g)).collect();
        let mut order: Vec<usize> = (0..groups.len()).collect();
        order.sort_by_key(|&g| (caps[g], g));
        let mut indices = vec![0; groups.len()];
        for g in order {
            indices[g] = place(caps[g], 1)?;
        }
        let groups = fixed
            .into_iter()
            .enumerate()
            .map(|(g, (f, w))| MixedRegion {
                kv_block: f[0],
                kv_slot: f[1],
                q_indptr: f[2],
                indptr: f[3],
                last_page_len: f[4],
                request_indices: f[5],
                qo_tile_indices: f[6],
                kv_tile_indices: f[7],
                valid: f[8],
                work_items: w,
                indices: indices[g],
                indices_cap: caps[g],
            })
            .collect();
        Ok(MixedLayout {
            tokens,
            logits,
            token_ids,
            positions,
            logit_rows,
            kv_chunk_size,
            sample_rows,
            groups,
            words: at,
        })
    }

    /// Words a step must upload: the table through the last-placed group's
    /// `used` page entries (`used[g]` per group, each within its
    /// reservation).
    pub fn upload_words(&self, used: &[usize]) -> usize {
        assert_eq!(used.len(), self.groups.len(), "pages used per group");
        let last = self
            .groups
            .iter()
            .enumerate()
            .max_by_key(|(_, r)| r.indices)
            .map(|(g, _)| g);
        match last {
            None => self.sample_rows + self.logits as usize * SAMPLE_ROW_WORDS,
            Some(g) => {
                assert!(
                    used[g] <= self.groups[g].indices_cap,
                    "pages past the reservation"
                );
                self.groups[g].indices + used[g]
            }
        }
    }

    /// Every array as `(offset, words)`, for checks.
    pub fn regions(&self) -> Vec<(usize, usize)> {
        let t = self.tokens as usize;
        let n = self.logits as usize;
        let mut v = vec![
            (self.token_ids, t),
            (self.positions, t),
            (self.logit_rows, n),
            (self.kv_chunk_size, 1),
            (self.sample_rows, n * SAMPLE_ROW_WORDS),
        ];
        for g in &self.groups {
            v.extend([
                (g.kv_block, t),
                (g.kv_slot, t),
                (g.q_indptr, t + 1),
                (g.indptr, t + 1),
                (g.last_page_len, t),
                (g.request_indices, g.work_items),
                (g.qo_tile_indices, g.work_items),
                (g.kv_tile_indices, g.work_items),
                (g.valid, g.work_items.div_ceil(4)),
                (g.indices, g.indices_cap),
            ]);
        }
        v
    }
}

/// A mixed step's real tokens, as the executor gathered and checked them
/// (the same values its eager path uploads).
#[derive(Clone, Copy, Debug)]
pub struct MixedRows<'a> {
    pub tokens: &'a [u32],
    pub positions: &'a [u32],
    /// Per group, per token: the block and offset its KV goes to.
    pub kv_targets: &'a [Vec<(u32, u32)>],
    /// Per group, per row: its attention request, covering the tokens in
    /// order.
    pub requests: &'a [Vec<AttnRequest>],
    /// The token rows whose logits are sampled, in sample order.
    pub logit_rows: &'a [u32],
    /// Sample row `j` reads logits row `j`.
    pub samples: &'a [SampleRow],
}

/// The step table for `real`'s tokens padded to `layout.tokens`: real tokens
/// first, as given, then padding tokens ([`PAD_TOKEN`] at [`PAD_POSITION`],
/// KV and pages in each group's pad block, one request each); real logit and
/// sample rows, then padding ones (token row 0; [`pad_sample`]). Every
/// group's work list is [`HostPlan::with_tile`]'s at [`MIXED_TILE`] over the
/// real and padding requests, so it passes the eager path's checks and its
/// real work items are the eager plan's; the step's eager tile must be
/// [`MIXED_TILE`] (an all-padding step, as captured, has none). Work items
/// past the list are masked off, request slots past it repeat its last
/// offsets. Refuses (changing nothing) a step the rung cannot run.
pub fn pack(
    layout: &MixedLayout,
    groups: &[GroupSlots],
    real: &MixedRows<'_>,
) -> Result<PackedStep> {
    let bad = |what: String| Err(CudaError::new(format!("mixed step: {what}")));
    let t_cap = layout.tokens as usize;
    let n_cap = layout.logits as usize;
    let t = real.tokens.len();
    let n = real.samples.len();
    if t > t_cap || n > n_cap {
        return bad(format!(
            "{t} tokens and {n} sample rows past the rung's {t_cap} and {n_cap}"
        ));
    }
    if real.positions.len() != t || real.logit_rows.len() != n {
        return bad(format!(
            "{t} tokens, {} positions, {} logit rows for {n} sample rows",
            real.positions.len(),
            real.logit_rows.len()
        ));
    }
    if let Some((j, s)) = real
        .samples
        .iter()
        .enumerate()
        .find(|(j, s)| s.logit_row as usize != *j)
    {
        return bad(format!("sample row {j} reads logits row {}", s.logit_row));
    }
    if let Some(&r) = real.logit_rows.iter().find(|&&r| r as usize >= t) {
        return bad(format!("logit row {r} of {t} tokens"));
    }
    let g_count = groups.len();
    if layout.groups.len() != g_count
        || real.kv_targets.len() != g_count
        || real.requests.len() != g_count
    {
        return bad(format!(
            "{} groups in the layout, {} slots, {} target lists, {} request lists",
            layout.groups.len(),
            g_count,
            real.kv_targets.len(),
            real.requests.len()
        ));
    }
    let mut w = vec![0u32; layout.words];
    let put = |w: &mut [u32], at: usize, xs: &mut dyn Iterator<Item = u32>| {
        for (i, x) in xs.enumerate() {
            w[at + i] = x;
        }
    };
    let word = |x: i32| u32::try_from(x).expect("HostPlan indices are non-negative");
    let pad = t..t_cap;
    put(
        &mut w,
        layout.token_ids,
        &mut real
            .tokens
            .iter()
            .copied()
            .chain(pad.clone().map(|_| PAD_TOKEN)),
    );
    put(
        &mut w,
        layout.positions,
        &mut real
            .positions
            .iter()
            .copied()
            .chain(pad.clone().map(|_| PAD_POSITION)),
    );
    put(
        &mut w,
        layout.logit_rows,
        &mut real.logit_rows.iter().copied().chain((n..n_cap).map(|_| 0)),
    );
    for (i, s) in real
        .samples
        .iter()
        .copied()
        .chain((n..n_cap).map(|_| pad_sample()))
        .enumerate()
    {
        let at = layout.sample_rows + i * SAMPLE_ROW_WORDS;
        w[at..at + SAMPLE_ROW_WORDS].copy_from_slice(&sample_row_words(&s));
    }
    let mut used = Vec::with_capacity(g_count);
    for (g, (slots, region)) in groups.iter().zip(&layout.groups).enumerate() {
        let targets = &real.kv_targets[g];
        let requests = &real.requests[g];
        if targets.len() != t {
            return bad(format!(
                "group {g}: {} KV targets for {t} tokens",
                targets.len()
            ));
        }
        let covered = requests
            .last()
            .map_or(0, |r| (r.q_start + r.qo_len) as usize);
        if covered != t {
            return bad(format!("group {g}: requests cover {covered} of {t} tokens"));
        }
        if t > 0 && tile_for(requests, slots.group_size) != MIXED_TILE {
            return bad(format!(
                "group {g}: the eager plan's tile is {}, not {MIXED_TILE}",
                tile_for(requests, slots.group_size)
            ));
        }
        let mut padded = requests.clone();
        padded.extend(
            pad.clone()
                .map(|i| pad_request(u32::try_from(i).expect("tokens is a u32"), slots.pad_block)),
        );
        let plan = HostPlan::with_tile(&padded, slots.group_size, slots.page_size, MIXED_TILE)?;
        let items = plan.work_items();
        if items > region.work_items || plan.num_requests() > t_cap {
            return bad(format!(
                "group {g}: {items} work items and {} requests past the rung's {} and {t_cap}",
                plan.num_requests(),
                region.work_items
            ));
        }
        if plan.indices.len() > region.indices_cap {
            return bad(format!(
                "group {g}: {} pages past the reservation of {}",
                plan.indices.len(),
                region.indices_cap
            ));
        }
        put(
            &mut w,
            region.kv_block,
            &mut targets
                .iter()
                .map(|x| x.0)
                .chain(pad.clone().map(|_| slots.pad_block)),
        );
        put(
            &mut w,
            region.kv_slot,
            &mut targets
                .iter()
                .map(|x| x.1)
                .chain(pad.clone().map(|_| PAD_POSITION % slots.page_size)),
        );
        // Request slots past the step's requests repeat its last offsets.
        let tail = |xs: &[i32]| {
            let last = word(*xs.last().expect("an indptr starts at 0"));
            xs.iter()
                .map(|&x| word(x))
                .chain(std::iter::repeat(last))
                .take(t_cap + 1)
                .collect::<Vec<u32>>()
        };
        put(
            &mut w,
            region.q_indptr,
            &mut tail(&plan.q_indptr).into_iter(),
        );
        put(&mut w, region.indptr, &mut tail(&plan.indptr).into_iter());
        for (at, xs) in [
            (region.last_page_len, &plan.last_page_len),
            (region.request_indices, &plan.request_indices),
            (region.qo_tile_indices, &plan.qo_tile_indices),
            (region.kv_tile_indices, &plan.kv_tile_indices),
            (region.indices, &plan.indices),
        ] {
            put(&mut w, at, &mut xs.iter().map(|&x| word(x)));
        }
        // One byte per work item, little-endian in its word: 1 for the
        // step's items, 0 for the rest.
        for i in 0..items {
            w[region.valid + i / 4] |= 1 << (8 * (i % 4));
        }
        used.push(plan.indices.len());
    }
    Ok(PackedStep {
        upload: layout.upload_words(&used),
        words: w,
    })
}

/// The captured mixed-step graphs and the buffers they read and write.
pub struct MixedGraphs {
    ladder: Vec<u32>,
    slots: Vec<GroupSlots>,
    layouts: Vec<MixedLayout>,
    /// The step table, sized for the largest rung.
    table: CudaSlice<u32>,
    /// The largest rung's sampled tokens, then the status word.
    readback: CudaSlice<u32>,
    graphs: Vec<CudaGraph>,
    capture_bytes: Vec<i64>,
}

impl std::fmt::Debug for MixedGraphs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MixedGraphs")
            .field("ladder", &self.ladder)
            .field("captured", &self.graphs.len())
            .finish_non_exhaustive()
    }
}

impl MixedGraphs {
    /// Allocate the step table and read-back for `ladder` over `slots`, with
    /// at most `seats` rows a step; nothing is captured yet.
    pub(crate) fn new(
        gpu: &Gpu,
        ladder: Vec<u32>,
        seats: u32,
        slots: Vec<GroupSlots>,
    ) -> Result<MixedGraphs> {
        let layouts = ladder
            .iter()
            .map(|&t| MixedLayout::new(t, seats, &slots))
            .collect::<Result<Vec<_>>>()?;
        let words = layouts.iter().map(|l| l.words).max().unwrap_or(1);
        let logits = layouts.iter().map(|l| l.logits as usize).max().unwrap_or(0);
        let s = gpu.stream();
        Ok(MixedGraphs {
            table: s.alloc_zeros(words.max(1))?,
            readback: s.alloc_zeros(logits + 1)?,
            ladder,
            slots,
            layouts,
            graphs: Vec::new(),
            capture_bytes: Vec::new(),
        })
    }

    pub fn ladder(&self) -> &[u32] {
        &self.ladder
    }

    pub fn slots(&self) -> &[GroupSlots] {
        &self.slots
    }

    pub fn layout(&self, rung: usize) -> &MixedLayout {
        &self.layouts[rung]
    }

    /// Device memory each rung's capture took, as the driver's free memory
    /// moved (approximate: the driver allocates in its own granules).
    pub fn capture_bytes(&self) -> &[i64] {
        &self.capture_bytes
    }

    /// Bytes of the step table and read-back.
    pub fn table_bytes(&self) -> usize {
        4 * (self.table.len() + self.readback.len())
    }

    /// Copy the uploaded prefix of a packed step into the step table.
    pub(crate) fn upload(&mut self, gpu: &Gpu, packed: &PackedStep) -> Result<()> {
        let n = packed.upload;
        if n > self.table.len() || n > packed.words.len() {
            return Err(CudaError::new("mixed step table overflows"));
        }
        gpu.stream()
            .memcpy_htod(&packed.words[..n], &mut self.table.slice_mut(..n))?;
        Ok(())
    }

    /// The padded step's launches for rung `rung`: zero the padded
    /// activations' rows, the forward over every token row with its logit
    /// rows, then sampling into the read-back. Reads only the step table,
    /// fixed buffers and the KV pools; copies nothing from the host,
    /// allocates nothing and reads nothing back.
    ///
    /// # Safety
    ///
    /// The step table must hold a [`pack`]ed step for this rung's layout when
    /// the launches run, and `a` must be the model, pools and sampler scratch
    /// the executor sized this rung for.
    unsafe fn program(&mut self, gpu: &Gpu, rung: usize, a: &mut ProgramArgs<'_>) -> Result<()> {
        let layout = &self.layouts[rung];
        let t = layout.tokens as usize;
        let n = layout.logits as usize;
        let s = gpu.stream();
        let base = dptr(&self.table, s);
        let at = |off: usize| base + 4 * off as u64;
        a.model.zero_padding_rows(gpu, t)?;
        let src = Indirect {
            tokens: t,
            token_ids: at(layout.token_ids),
            positions: at(layout.positions),
            kv_block: layout.groups.iter().map(|r| at(r.kv_block)).collect(),
            kv_slot: layout.groups.iter().map(|r| at(r.kv_slot)).collect(),
            plans: layout
                .groups
                .iter()
                .zip(&self.slots)
                .map(|(r, g)| -> Result<PlanView> {
                    Ok(PlanView {
                        tile: MIXED_TILE,
                        page_size: g.page_size,
                        group_size: g.group_size,
                        work_items: crate::narrow(r.work_items, "mixed work items")?,
                        num_requests: layout.tokens,
                        q_indptr: at(r.q_indptr),
                        indices: at(r.indices),
                        indptr: at(r.indptr),
                        last_page_len: at(r.last_page_len),
                        request_indices: at(r.request_indices),
                        qo_tile_indices: at(r.qo_tile_indices),
                        kv_tile_indices: at(r.kv_tile_indices),
                        kv_chunk_size: at(layout.kv_chunk_size),
                        valid: at(r.valid),
                    })
                })
                .collect::<Result<_>>()?,
            logit_rows: at(layout.logit_rows),
            num_logit_rows: n,
            final_norm: false,
        };
        // SAFETY: the table holds a packed step for this layout (the
        // caller's contract): every value `pack` wrote passed the eager
        // path's checks or is a padding constant inside the pad block; work
        // items past the step's are masked off before they read anything.
        unsafe { a.model.launch(gpu, a.kv, &src, false)? };
        s.memset_zeros(&mut self.readback.slice_mut(n..n + 1))?;
        let rb = dptr(&self.readback, s);
        let logits = a.model.logits();
        if logits.len() < n * a.vocab as usize {
            return Err(CudaError::new("mixed rung past the logits scratch"));
        }
        let launch = SampleLaunch {
            logits: dptr(logits, s),
            logits_stride: u64::from(a.vocab),
            n: a.sampleable,
            rows: at(layout.sample_rows),
            num_rows: layout.logits,
            draw: Some(Stream::Sample),
            probs: a.probs,
            tokens: rb,
            status: rb + 4 * n as u64,
        };
        // SAFETY: the sample rows read logits rows below `n` (real rows their
        // own, padding rows row 0), inside the logits checked above; the
        // read-back holds `n + 1` words; `probs` is the executor's scratch for
        // at least `seats` rows, and `n` is at most `seats`.
        unsafe { a.model.kernels.sampler.launch_sample(gpu, launch) }
    }

    /// Run each rung once directly over an all-padding step (which checks
    /// every launch on this device), capture it, replay the capture once and
    /// keep it; for every rung, in order. Called once, at construction.
    pub(crate) fn capture(&mut self, gpu: &Gpu, a: &mut ProgramArgs<'_>) -> Result<()> {
        let s = gpu.stream().clone();
        let targets = vec![Vec::new(); self.slots.len()];
        let requests = vec![Vec::new(); self.slots.len()];
        let none = MixedRows {
            tokens: &[],
            positions: &[],
            kv_targets: &targets,
            requests: &requests,
            logit_rows: &[],
            samples: &[],
        };
        for rung in 0..self.layouts.len() {
            let packed = pack(&self.layouts[rung], &self.slots, &none)?;
            self.upload(gpu, &packed)?;
            // SAFETY: the table holds the packed all-padding step.
            unsafe { self.program(gpu, rung, a)? };
            self.check_status(gpu, rung)?;
            let before = free_memory()?;
            s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            // SAFETY: as above; nothing runs while capturing.
            let recorded = unsafe { self.program(gpu, rung, a) };
            let graph = s.end_capture(sys::CUgraphInstantiate_flags(0));
            recorded?;
            let graph = graph?.ok_or_else(|| {
                CudaError::new(format!(
                    "mixed rung {}: the capture is empty",
                    self.ladder[rung]
                ))
            })?;
            graph.upload()?;
            graph.launch()?;
            self.check_status(gpu, rung)?;
            self.capture_bytes.push(before - free_memory()?);
            self.graphs.push(graph);
        }
        Ok(())
    }

    /// Launch rung `rung`'s step (the table already uploaded): its graph, or
    /// with `direct` the same launches unrecorded.
    pub(crate) fn run(
        &mut self,
        gpu: &Gpu,
        rung: usize,
        direct: bool,
        a: &mut ProgramArgs<'_>,
    ) -> Result<()> {
        if direct {
            // SAFETY: the executor uploaded a packed step for this rung.
            unsafe { self.program(gpu, rung, a) }
        } else {
            let graph = self
                .graphs
                .get(rung)
                .ok_or_else(|| CudaError::new("no captured mixed graph for this rung"))?;
            Ok(graph.launch()?)
        }
    }

    /// The rung's read-back: its sample rows' tokens and the status word
    /// (one copy, which waits for the step).
    pub(crate) fn readback(&self, gpu: &Gpu, rung: usize) -> Result<Vec<u32>> {
        let n = self.layouts[rung].logits as usize;
        Ok(gpu.stream().clone_dtoh(&self.readback.slice(..n + 1))?)
    }

    fn check_status(&self, gpu: &Gpu, rung: usize) -> Result<()> {
        let n = self.layouts[rung].logits as usize;
        let rb = self.readback(gpu, rung)?;
        let (_, status) = split_readback(&rb, 0, n);
        if status & STATUS_NON_FINITE != 0 {
            return Err(CudaError::new(format!(
                "mixed rung {}: non-finite logits over padding",
                self.ladder[rung]
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::max_decode_pages;
    use eidola_engine::sampling::SamplingParams;
    use eidola_engine::spec::AttentionKind;

    fn bucket(max_seqs: u32, max_tokens: u32) -> Bucket {
        Bucket {
            max_seqs,
            max_tokens,
        }
    }

    #[test]
    fn the_ladder_is_every_grain_to_the_cap() {
        let ladder = mixed_ladder(&[bucket(64, 8192)]);
        assert_eq!(ladder.len(), 32);
        assert_eq!(ladder[..4], [16, 32, 48, 64]);
        assert_eq!(*ladder.last().unwrap(), MIXED_MAX_TOKENS);
        assert!(ladder.iter().all(|r| r.is_multiple_of(MIXED_GRAIN)));
        // The last bucket's tokens cap it, and the cap is a rung.
        assert_eq!(
            mixed_ladder(&[bucket(1, 16), bucket(6, 130)]),
            vec![16, 32, 48, 64, 80, 96, 112, 128, 130]
        );
        assert_eq!(mixed_ladder(&[bucket(8, 40)]), vec![16, 32, 40]);
        assert_eq!(mixed_ladder(&[bucket(8, 16)]), vec![16]);
        assert_eq!(mixed_ladder(&[bucket(8, 9)]), vec![9]);
        assert_eq!(mixed_ladder(&[]), Vec::<u32>::new());
    }

    /// The padded step takes the expert layout the eager one would: masked
    /// exactly when its tokens are at most 128; padding is under one grain;
    /// past the cap nothing replays.
    #[test]
    fn a_rung_is_masked_exactly_when_its_tokens_are() {
        const MASKED: usize = 128;
        for cap in 1..=700u32 {
            let ladder = mixed_ladder(&[bucket(64, cap)]);
            assert!(ladder.windows(2).all(|w| w[0] < w[1]), "{ladder:?}");
            let top = cap.min(MIXED_MAX_TOKENS) as usize;
            assert_eq!(*ladder.last().unwrap() as usize, top);
            for tokens in 1..=top {
                let rung = ladder[mixed_rung(&ladder, tokens, &[128, 128]).unwrap()] as usize;
                assert!(rung >= tokens, "cap {cap}: {tokens} tokens in rung {rung}");
                assert_eq!(
                    tokens <= MASKED,
                    rung <= MASKED,
                    "cap {cap}: {tokens} tokens in rung {rung}"
                );
                assert!(
                    rung - tokens < MIXED_GRAIN as usize,
                    "cap {cap}: {tokens} tokens in rung {rung}"
                );
            }
            assert_eq!(mixed_rung(&ladder, top + 1, &[128, 128]), None);
            assert_eq!(mixed_rung(&ladder, 0, &[128, 128]), None);
        }
    }

    /// Only steps whose every group would tile at 128 eagerly replay.
    #[test]
    fn only_steps_tiled_at_128_replay() {
        let ladder = mixed_ladder(&[bucket(8, 512)]);
        assert_eq!(mixed_rung(&ladder, 40, &[128, 128]), Some(2));
        assert_eq!(mixed_rung(&ladder, 40, &[128, 64]), None);
        assert_eq!(mixed_rung(&ladder, 40, &[16, 16]), None);
        assert_eq!(mixed_rung(&ladder, 40, &[128]), Some(2));
        assert_eq!(mixed_rung(&ladder, 40, &[]), None);
    }

    /// A request's pages: as the executor lists them, from the first
    /// position its first query sees through its last query.
    fn request_pages(kind: AttentionKind, bs: u32, c: u32, q: u32) -> u32 {
        let p = c + q - 1;
        p / bs - kind.first_visible(c) / bs + 1
    }

    /// A request of `q` queries lists at most its decode bound plus
    /// `ceil(q / page_size)` pages, the premise of [`pages_cap`].
    #[test]
    fn a_requests_pages_are_bounded_by_its_queries() {
        for bs in 1..=8u32 {
            for kind in (1..=40u32)
                .map(|window| AttentionKind::Sliding { window })
                .chain([AttentionKind::Full])
            {
                let max_len = 160u32;
                let max_blocks = max_len.div_ceil(bs);
                let bound = max_decode_pages(kind, bs, max_blocks);
                for c in 0..max_len {
                    for q in 1..=max_len - c {
                        let pages = request_pages(kind, bs, c, q);
                        assert!(
                            pages <= bound + q.div_ceil(bs),
                            "bs {bs} {kind:?} c {c} q {q}: {pages} pages"
                        );
                        assert!(pages <= max_blocks);
                    }
                }
            }
        }
    }

    /// Flash's two groups: global (GQA 16) and sliding (GQA 8, window 128),
    /// pages of 16 positions, 1,000 blocks the seam may name (`1..1000`; the
    /// pad block is 1000), sequences of up to `max_blocks` blocks.
    fn flash_slots(max_blocks: u32) -> Vec<GroupSlots> {
        [
            (16, AttentionKind::Full),
            (8, AttentionKind::Sliding { window: 128 }),
        ]
        .into_iter()
        .map(|(group_size, kind)| GroupSlots {
            group_size,
            page_size: 16,
            pad_block: 1000,
            max_pages: max_decode_pages(kind, 16, max_blocks),
        })
        .collect()
    }

    const KINDS: [AttentionKind; 2] = [AttentionKind::Full, AttentionKind::Sliding { window: 128 }];

    /// A step of real rows (`(context, queries, samples)` each) over Flash's
    /// two groups, as the executor gathers it: row `i`'s blocks from `1 +
    /// 20 i`, in both groups.
    struct Step {
        tokens: Vec<u32>,
        positions: Vec<u32>,
        kv_targets: Vec<Vec<(u32, u32)>>,
        requests: Vec<Vec<AttnRequest>>,
        logit_rows: Vec<u32>,
        samples: Vec<SampleRow>,
    }

    impl Step {
        fn of(rows: &[(u32, u32, bool)]) -> Step {
            let bs = 16;
            let mut s = Step {
                tokens: Vec::new(),
                positions: Vec::new(),
                kv_targets: vec![Vec::new(), Vec::new()],
                requests: vec![Vec::new(), Vec::new()],
                logit_rows: Vec::new(),
                samples: Vec::new(),
            };
            let mut q_start = 0u32;
            for (i, &(c, q, sample)) in rows.iter().enumerate() {
                let base = 1 + 20 * u32::try_from(i).unwrap();
                let p = c + q - 1;
                for pos in c..=p {
                    s.tokens.push(100 + pos);
                    s.positions.push(pos);
                }
                for (g, kind) in KINDS.iter().enumerate() {
                    for pos in c..=p {
                        s.kv_targets[g].push((base + pos / bs, pos % bs));
                    }
                    let first = kind.first_visible(c) / bs;
                    s.requests[g].push(AttnRequest {
                        q_start,
                        qo_len: q,
                        pages: (first..=p / bs).map(|b| base + b).collect(),
                        kv_len: p + 1 - first * bs,
                    });
                }
                if sample {
                    let j = u32::try_from(s.samples.len()).unwrap();
                    s.logit_rows.push(q_start + q - 1);
                    s.samples
                        .push(SampleRow::new(&SamplingParams::greedy(), p + 1, j));
                }
                q_start += q;
            }
            s
        }

        fn rows(&self) -> MixedRows<'_> {
            MixedRows {
                tokens: &self.tokens,
                positions: &self.positions,
                kv_targets: &self.kv_targets,
                requests: &self.requests,
                logit_rows: &self.logit_rows,
                samples: &self.samples,
            }
        }
    }

    fn read(w: &[u32], at: usize, n: usize) -> Vec<u32> {
        w[at..at + n].to_vec()
    }

    fn mask(w: &[u32], r: &MixedRegion) -> Vec<u8> {
        (0..r.work_items)
            .map(|i| u8::try_from((w[r.valid + i / 4] >> (8 * (i % 4))) & 0xff).unwrap())
            .collect()
    }

    #[test]
    fn the_layout_is_disjoint_aligned_and_uploads_a_prefix() {
        for (tokens, seats) in [(16u32, 8u32), (40, 64), (128, 64), (130, 3), (512, 64)] {
            let slots = flash_slots(19);
            let l = MixedLayout::new(tokens, seats, &slots).unwrap();
            assert_eq!(l.logits, tokens.min(seats));
            let mut regions = l.regions();
            regions.sort();
            for w in regions.windows(2) {
                assert!(w[0].0 + w[0].1 <= w[1].0, "{tokens}: {regions:?}");
            }
            let (last, len) = *regions.last().unwrap();
            assert_eq!(last + len, l.words);
            assert_eq!(l.sample_rows % 2, 0, "u64 seeds need 8-byte rows");
            for (r, s) in l.groups.iter().zip(&slots) {
                assert_eq!(Some(r.work_items), work_cap(tokens, s.group_size));
                assert_eq!(r.indices_cap, pages_cap(tokens, seats, s));
            }
            // The smaller reservation (sliding) is placed first; the global
            // list's used part ends the upload, and every other region lies
            // inside it.
            assert!(l.groups[1].indices < l.groups[0].indices);
            assert_eq!(l.upload_words(&[0, 3]), l.groups[0].indices);
            assert_eq!(l.upload_words(&[5, 0]), l.groups[0].indices + 5);
            for (off, len) in l.regions() {
                if off != l.groups[0].indices {
                    assert!(off + len <= l.upload_words(&[0, 0]));
                }
            }
        }
    }

    /// Padding tokens touch only the pad blocks and sample real row 0
    /// greedily; the real rows' work list is exactly the eager plan's, the
    /// items past the step's masked off and the request slots past its
    /// requests repeating its last offsets.
    #[test]
    fn padding_is_inert_and_real_rows_keep_the_eager_plan() {
        // A decode row, a 40-token extend past the window, a 12-token prefill
        // chunk that does not sample, a decode row past the window: 54
        // tokens in the 64 rung.
        let step = Step::of(&[
            (17, 1, true),
            (130, 40, true),
            (0, 12, false),
            (150, 1, true),
        ]);
        let t = step.tokens.len();
        assert_eq!(t, 54);
        let slots = flash_slots(19);
        let l = MixedLayout::new(64, 8, &slots).unwrap();
        let packed = pack(&l, &slots, &step.rows()).unwrap();
        let w = &packed.words;
        let tok = read(w, l.token_ids, 64);
        assert_eq!(tok[..t], step.tokens[..]);
        assert!(tok[t..].iter().all(|&x| x == PAD_TOKEN));
        let pos = read(w, l.positions, 64);
        assert_eq!(pos[..t], step.positions[..]);
        assert!(pos[t..].iter().all(|&x| x == PAD_POSITION));
        // Three sampled rows, then padding logit rows at token row 0.
        assert_eq!(l.logits, 8);
        assert_eq!(read(w, l.logit_rows, 8), [0, 40, 53, 0, 0, 0, 0, 0]);
        assert_eq!(step.logit_rows, [0, 40, 53]);
        for i in 0..8 {
            let at = l.sample_rows + i * SAMPLE_ROW_WORDS;
            let want = if i < 3 { step.samples[i] } else { pad_sample() };
            assert_eq!(w[at..at + SAMPLE_ROW_WORDS], sample_row_words(&want));
            if i >= 3 {
                assert_eq!(w[at], 0f32.to_bits(), "row {i} greedy");
                assert_eq!(w[at + 7], 0, "row {i} reads real logits row 0");
            }
        }
        for (g, (s, r)) in slots.iter().zip(&l.groups).enumerate() {
            let blocks = read(w, r.kv_block, 64);
            assert_eq!(
                blocks[..t],
                step.kv_targets[g].iter().map(|x| x.0).collect::<Vec<_>>()[..]
            );
            assert!(blocks[..t].iter().all(|&b| (1..s.pad_block).contains(&b)));
            assert!(blocks[t..].iter().all(|&b| b == s.pad_block));
            let offs = read(w, r.kv_slot, 64);
            assert!(offs[t..].iter().all(|&o| o == 0));
            // The real requests' arrays are the eager plan's, at its tile.
            let eager = HostPlan::new(&step.requests[g], s.group_size, s.page_size).unwrap();
            assert_eq!(eager.tile(), MIXED_TILE);
            let words = |xs: &[i32]| {
                xs.iter()
                    .map(|&x| u32::try_from(x).unwrap())
                    .collect::<Vec<_>>()
            };
            let reqs = eager.num_requests();
            let items = eager.work_items();
            let real_pages = eager.indices.len();
            let q_indptr = read(w, r.q_indptr, 65);
            let indptr = read(w, r.indptr, 65);
            assert_eq!(q_indptr[..=reqs], words(&eager.q_indptr)[..]);
            assert_eq!(indptr[..=reqs], words(&eager.indptr)[..]);
            assert_eq!(read(w, r.indices, real_pages), words(&eager.indices));
            assert_eq!(
                read(w, r.request_indices, items),
                words(&eager.request_indices)
            );
            assert_eq!(
                read(w, r.qo_tile_indices, items),
                words(&eager.qo_tile_indices)
            );
            assert_eq!(
                read(w, r.kv_tile_indices, items),
                words(&eager.kv_tile_indices)
            );
            assert_eq!(read(w, r.last_page_len, reqs), words(&eager.last_page_len));
            // Each padding token: one request of one query over one position
            // of the pad block, one work item.
            let pads = 64 - t;
            assert_eq!(
                read(w, r.indices + real_pages, pads),
                vec![s.pad_block; pads]
            );
            for i in 0..pads {
                assert_eq!(q_indptr[reqs + i + 1], u32::try_from(t + i + 1).unwrap());
                assert_eq!(indptr[reqs + i + 1] - indptr[reqs + i], 1);
                assert_eq!(w[r.last_page_len + reqs + i], 1);
                assert_eq!(
                    w[r.request_indices + items + i],
                    u32::try_from(reqs + i).unwrap()
                );
            }
            // The request slots past the step's (four real, ten padding)
            // repeat its last offsets: every token, every page.
            assert_eq!(reqs + pads, 14);
            assert!(q_indptr[reqs + pads..].iter().all(|&x| x == 64));
            assert!(
                indptr[reqs + pads..]
                    .iter()
                    .all(|&x| x as usize == real_pages + pads)
            );
            // The step's items valid, the rest of the fixed grid masked.
            let m = mask(w, r);
            assert!(m[..items + pads].iter().all(|&b| b == 1), "group {g}");
            assert!(m[items + pads..].iter().all(|&b| b == 0), "group {g}");
            assert!(items + pads < r.work_items);
        }
        assert_eq!(w[l.kv_chunk_size], 0);
        let global_pages = HostPlan::new(&step.requests[0], 16, 16)
            .unwrap()
            .indices
            .len()
            + 64
            - t;
        assert_eq!(packed.upload, l.groups[0].indices + global_pages);

        // An all-padding step (the capture's): every token padding, tiled at
        // 128, no sample row real.
        let none = Step::of(&[]);
        let p = pack(&l, &slots, &none.rows()).unwrap();
        for (s, r) in slots.iter().zip(&l.groups) {
            assert_eq!(read(&p.words, r.kv_block, 64), [s.pad_block; 64]);
            assert_eq!(read(&p.words, r.indices, 64), [s.pad_block; 64]);
            assert_eq!(mask(&p.words, r).iter().filter(|&&b| b == 1).count(), 64);
        }
    }

    /// A step with fewer requests than tokens leaves request slots past its
    /// last: they repeat the final offsets, so `indptr[batch_size]` is the
    /// step's page count.
    #[test]
    fn spare_request_slots_repeat_the_last_offsets() {
        let step = Step::of(&[(0, 20, true)]);
        let slots = flash_slots(19);
        let l = MixedLayout::new(20, 8, &slots).unwrap();
        let p = pack(&l, &slots, &step.rows()).unwrap();
        for (g, (s, r)) in slots.iter().zip(&l.groups).enumerate() {
            let eager = HostPlan::new(&step.requests[g], s.group_size, s.page_size).unwrap();
            let q = read(&p.words, r.q_indptr, 21);
            assert_eq!(q[..2], [0, 20]);
            assert!(q[2..].iter().all(|&x| x == 20), "{q:?}");
            let ip = read(&p.words, r.indptr, 21);
            assert!(
                ip[1..].iter().all(|&x| x as usize == eager.indices.len()),
                "{ip:?}"
            );
        }
    }

    /// Every step a rung can hold fits its fixed work items and page
    /// reservations: random compositions of decode rows, extends and prefill
    /// chunks at random contexts, sampled or not.
    #[test]
    fn every_step_fits_its_rung() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut below = |n: u32| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            u32::try_from((state >> 33) % u64::from(n)).unwrap()
        };
        let max_blocks = 19;
        let slots = flash_slots(max_blocks);
        let ladder = mixed_ladder(&[bucket(8, 512)]);
        let mut fitted = 0;
        for _ in 0..3000 {
            let n_rows = 1 + below(8);
            let mut rows = Vec::new();
            let mut tokens = 0;
            for _ in 0..n_rows {
                let q = if below(3) == 0 { 1 } else { 1 + below(120) };
                let c = below(max_blocks * 16 - q);
                if tokens + q > 512 {
                    break;
                }
                tokens += q;
                rows.push((c, q, below(2) == 0));
            }
            if rows.is_empty() {
                continue;
            }
            let step = Step::of(&rows);
            let tiles: Vec<u32> = slots
                .iter()
                .zip(&step.requests)
                .map(|(s, r)| tile_for(r, s.group_size))
                .collect();
            let Some(rung) = mixed_rung(&ladder, step.tokens.len(), &tiles) else {
                continue;
            };
            let l = MixedLayout::new(ladder[rung], 8, &slots).unwrap();
            pack(&l, &slots, &step.rows()).unwrap_or_else(|e| panic!("{rows:?}: {e}"));
            fitted += 1;
        }
        assert!(fitted > 1000, "{fitted} steps replayable");
        // A long extend filling its rung lists more sliding pages than its
        // decode bound plus one: the reservation's per-token term holds them.
        let long = Step::of(&[(200, 64, true)]);
        let l = MixedLayout::new(64, 1, &slots).unwrap();
        pack(&l, &slots, &long.rows()).unwrap();
        assert!(long.requests[1][0].pages.len() > slots[1].max_pages as usize + 1);
        // A request takes at most one work item per query at the 128-row
        // tile for GQA groups up to 128 (one item per query exactly for one
        // query), and no bound is claimed past that.
        for g in 1..=128u32 {
            for q in 1..=600u32 {
                assert!((q * g).div_ceil(128) <= q, "g {g} q {q}");
            }
            assert_eq!(work_cap(37, g), Some(37));
        }
        assert_eq!(work_cap(37, 129), None);
    }

    #[test]
    fn steps_the_rung_cannot_run_are_refused() {
        let slots = flash_slots(19);
        let step = Step::of(&[(17, 1, true), (130, 40, true)]);
        let l = MixedLayout::new(64, 8, &slots).unwrap();
        pack(&l, &slots, &step.rows()).unwrap();
        let refused = |l: &MixedLayout, s: &Step| {
            pack(l, &slots, &s.rows()).expect_err("refused");
        };
        // More tokens than the rung.
        refused(&MixedLayout::new(32, 8, &slots).unwrap(), &step);
        // More sample rows than the rung's logit rows.
        let many = Step::of(&[(0, 9, true), (0, 1, true), (0, 1, true)]);
        refused(&MixedLayout::new(16, 2, &slots).unwrap(), &many);
        // A step the eager plan tiles below 128 in one group (eight tokens:
        // 128 packed rows in the global group, 64 in the sliding one).
        refused(&l, &Step::of(&[(17, 1, true), (30, 8, true)]));
        // A sample row reading another row's logits.
        let mut s = Step::of(&[(17, 1, true), (130, 40, true)]);
        s.samples[1].logit_row = 0;
        refused(&l, &s);
        // A logit row past the tokens.
        let mut s = Step::of(&[(17, 1, true), (130, 40, true)]);
        s.logit_rows[1] = 41;
        refused(&l, &s);
        // Requests not covering every token.
        let mut s = Step::of(&[(17, 1, true), (130, 40, true)]);
        s.requests[1][1].qo_len = 39;
        refused(&l, &s);
        // A request the eager plan refuses (pages not covering its KV).
        let mut s = Step::of(&[(17, 1, true), (130, 40, true)]);
        s.requests[0][1].kv_len = 2000;
        refused(&l, &s);
        // A missing KV target.
        let mut s = Step::of(&[(17, 1, true), (130, 40, true)]);
        s.kv_targets[1].pop();
        refused(&l, &s);
        // More pages than the reservation.
        let tight = flash_slots(2);
        let lt = MixedLayout::new(16, 1, &tight).unwrap();
        let wide = Step::of(&[(250, 9, true), (280, 7, false)]);
        pack(&lt, &tight, &wide.rows()).expect_err("past the reservation");
    }

    #[test]
    fn padding_tokens_are_never_returned() {
        let rb = [7, 8, 1000, 1000, 0b1];
        let (tokens, status) = split_readback(&rb, 2, 4);
        assert_eq!(tokens, [7, 8]);
        assert_eq!(status, 1);
    }
}
