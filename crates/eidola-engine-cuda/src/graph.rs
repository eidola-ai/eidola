//! CUDA-graph replay of decode steps.
//!
//! A pure-decode step (every row one host token, sampled, no drafts) runs as
//! one captured graph per rung of a fixed ladder of row counts
//! ([`decode_ladder`]). The step is padded to its rung; everything it varies
//! is data in one device buffer, the **step table**, filled by one
//! host-to-device copy before the graph launches ([`DecodeLayout`],
//! [`pack`]); the launches read it through fixed addresses, with fixed
//! counts. Maintenance and table updates run before the copy, as on the eager
//! path. The step's one read-back is the sampled tokens and the status word,
//! one copy after the graph.
//!
//! **Padding rows are inert by construction.** A padding row is token
//! [`PAD_TOKEN`] at position [`PAD_POSITION`]; its KV write and its whole
//! attention read go to the pad block of each group
//! ([`crate::kv::GroupGeometry::pad_block`]), which no table maps and no
//! maintenance reaches, so padding never writes or reads a block the host
//! allocates, shares or caches. Every other launch is row-local (norms,
//! quantization, GEMM rows, the expert layout's rows, the combine), so a
//! padding row reaches no real row's values. Its sample row is greedy over
//! real row 0's logits, so it adds no status the real rows do not already
//! raise, and its token is never returned ([`split_readback`]).
//!
//! **Numerics.** The graph is a recording of the padded launch sequence, so
//! replaying it computes what launching that sequence directly computes,
//! bit for bit. Against the eager (unpadded) path each real row runs the same
//! kernels over the same operands: the attention tile is 16 for every decode
//! row either way (a decode row's packed query run is one row times the GQA
//! group size), the expert layout is the masked one exactly when the eager
//! step's would be (the ladder keeps 128 as a rung, see [`decode_ladder`]),
//! and the GEMMs reduce each output row in an order that does not depend on
//! the number of rows. The GPU tests check token and logit identity.

use cudarc::driver::{CudaGraph, CudaSlice, sys};
use eidola_engine::executor::StepInput;
use eidola_engine::sampling::Stream;
use eidola_engine::spec::{AttentionKind, Bucket};

use crate::attention::{AttnRequest, HostPlan, PlanView};
use crate::kv::KvStore;
use crate::launch::dptr;
use crate::model::{GpuModel, Indirect};
use crate::sampler::{STATUS_NON_FINITE, SampleLaunch, SampleRow};
use crate::{CudaError, Gpu, Result, narrow};

/// Whether decode steps replay captured graphs: the executor's kill switch
/// (`CudaExecutorConfig::graphs`, the inference node's
/// `EIDOLA_ENGINE_CUDA_GRAPHS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaGraphs {
    /// Every step runs eagerly; nothing is captured.
    Off,
    /// Decode graphs are captured at construction, one per rung, and every
    /// pure-decode step that fits a rung replays one.
    On,
}

impl CudaGraphs {
    /// The configuration spelling: exactly `on` or `off`; anything else is
    /// refused.
    pub fn parse(s: &str) -> std::result::Result<CudaGraphs, String> {
        match s {
            "on" => Ok(CudaGraphs::On),
            "off" => Ok(CudaGraphs::Off),
            _ => Err("expected `on` or `off`".into()),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            CudaGraphs::On => "on",
            CudaGraphs::Off => "off",
        }
    }
}

/// How the executor runs a decode step that fits a rung. Serving uses
/// `Replay` with graphs on and `Eager` with them off; the other choices are
/// for tests and measurement on one executor
/// (`CudaExecutor::set_decode_path`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodePath {
    /// The eager path, unpadded (every step, as with graphs off).
    Eager,
    /// The padded launch sequence, launched directly rather than replayed.
    Direct,
    /// The captured graph.
    Replay,
}

/// The token a padding row embeds.
pub const PAD_TOKEN: u32 = 0;
/// The position a padding row sits at (its RoPE row; its KV slot in the pad
/// block).
pub const PAD_POSITION: u32 = 0;
/// The query tile every decode work item uses: a decode row's packed query
/// run is the GQA group size (16 or 8 on Flash), which FlashInfer's rule
/// tiles at 16.
pub const DECODE_TILE: u32 = 16;
/// Words per [`SampleRow`] in the step table.
const SAMPLE_ROW_WORDS: usize = 8;

/// The rungs: powers of two below the largest bucket's row capacity
/// (`min(max_seqs, max_tokens)`, one host token per decode row), each
/// configured bucket's row capacity, and that largest capacity itself;
/// ascending, without repeats.
///
/// Powers of two bound the padding at under half the rung, and padding is
/// cheap where small rungs sit: a decode step is bound by reading weights,
/// a GEMM's M tile is 128 rows, and padding rows are identical, so together
/// they route to the same `top_k` experts. 128 is a rung whenever the
/// capacity exceeds it, so a step of at most 128 rows (the masked expert
/// layout's limit) always gets a rung of at most 128, and the padded step
/// takes the expert layout the eager step would.
pub fn decode_ladder(buckets: &[Bucket]) -> Vec<u32> {
    let rows = |b: &Bucket| b.max_seqs.min(b.max_tokens);
    let cap = buckets.last().map_or(0, rows);
    let mut rungs: Vec<u32> = (0..32)
        .map(|i| 1u32 << i)
        .take_while(|&r| r < cap)
        .collect();
    rungs.extend(buckets.iter().map(rows).filter(|&r| r > 0));
    rungs.sort_unstable();
    rungs.dedup();
    rungs
}

/// The smallest rung holding `rows` rows (ladder ascending), if any.
pub fn rung_for(ladder: &[u32], rows: usize) -> Option<usize> {
    if rows == 0 {
        return None;
    }
    ladder.iter().position(|&r| r as usize >= rows)
}

/// The rows of a step a decode graph can run, or `None`: every row one host
/// token, sampled, with no drafts.
pub fn decode_rows(step: &StepInput) -> Option<usize> {
    let decode = !step.seqs.is_empty()
        && step
            .seqs
            .iter()
            .all(|e| e.num_tokens == 1 && e.sample && e.num_drafts == 0);
    decode.then_some(step.seqs.len())
}

/// The most pages one decode row of a group can list: every block of the
/// longest sequence for full attention; for a window `W`, the blocks
/// `W` consecutive positions can touch, `ceil((W - 1) / B) + 1` (the core's
/// bound), and never more than the sequence's blocks.
pub fn max_decode_pages(attention: AttentionKind, block_size: u32, max_blocks_per_seq: u32) -> u32 {
    match attention {
        AttentionKind::Full => max_blocks_per_seq,
        AttentionKind::Sliding { window } => {
            (window.saturating_sub(1).div_ceil(block_size) + 1).min(max_blocks_per_seq)
        }
    }
}

/// One KV group as the decode graphs see it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupSlots {
    /// Query heads per KV head.
    pub group_size: u32,
    pub page_size: u32,
    /// Where padding rows write and read ([`crate::kv::GroupGeometry::pad_block`]).
    pub pad_block: u32,
    /// The most pages one decode row lists ([`max_decode_pages`]).
    pub max_pages: u32,
}

/// One group's arrays in the step table: word offsets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupRegion {
    pub kv_block: usize,
    pub kv_slot: usize,
    pub q_indptr: usize,
    pub indptr: usize,
    pub last_page_len: usize,
    pub request_indices: usize,
    pub qo_tile_indices: usize,
    pub kv_tile_indices: usize,
    pub indices: usize,
    /// Words reserved for `indices`: rows × the group's page bound.
    pub indices_cap: usize,
}

/// The step table of one rung: where each array sits, in 32-bit words from
/// the table's start. The fixed-size arrays come first (sample rows at an
/// even word, for their `u64` seeds), then each group's page list, the
/// smallest reservation first, so a step's upload is one prefix of the
/// table: everything up to the last group's used pages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeLayout {
    pub rows: u32,
    pub token_ids: usize,
    pub positions: usize,
    pub logit_rows: usize,
    /// One word, always zero (FlashInfer reads it only when splitting KV).
    pub kv_chunk_size: usize,
    pub sample_rows: usize,
    pub groups: Vec<GroupRegion>,
    /// The table's size.
    pub words: usize,
}

impl DecodeLayout {
    pub fn new(rows: u32, groups: &[GroupSlots]) -> Result<DecodeLayout> {
        let overflow = || CudaError::new(format!("decode step table for {rows} rows overflows"));
        let r = rows as usize;
        let mut at = 0usize;
        // Each array starts at a multiple of `align` words.
        let mut place = |len: usize, align: usize| -> Result<usize> {
            let off = at.next_multiple_of(align);
            at = off.checked_add(len).ok_or_else(overflow)?;
            Ok(off)
        };
        let token_ids = place(r, 1)?;
        let positions = place(r, 1)?;
        let logit_rows = place(r, 1)?;
        let kv_chunk_size = place(1, 1)?;
        let mut fixed = Vec::with_capacity(groups.len());
        for _ in groups {
            fixed.push([
                place(r, 1)?,
                place(r, 1)?,
                place(r + 1, 1)?,
                place(r + 1, 1)?,
                place(r, 1)?,
                place(r, 1)?,
                place(r, 1)?,
                place(r, 1)?,
            ]);
        }
        let sample_rows = place(r.checked_mul(SAMPLE_ROW_WORDS).ok_or_else(overflow)?, 2)?;
        let caps: Vec<usize> = groups
            .iter()
            .map(|g| r.checked_mul(g.max_pages as usize).ok_or_else(overflow))
            .collect::<Result<_>>()?;
        let mut order: Vec<usize> = (0..groups.len()).collect();
        order.sort_by_key(|&g| (caps[g], g));
        let mut indices = vec![0; groups.len()];
        for g in order {
            indices[g] = place(caps[g], 1)?;
        }
        let groups = fixed
            .into_iter()
            .enumerate()
            .map(|(g, f)| GroupRegion {
                kv_block: f[0],
                kv_slot: f[1],
                q_indptr: f[2],
                indptr: f[3],
                last_page_len: f[4],
                request_indices: f[5],
                qo_tile_indices: f[6],
                kv_tile_indices: f[7],
                indices: indices[g],
                indices_cap: caps[g],
            })
            .collect();
        Ok(DecodeLayout {
            rows,
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
    /// reservation). Every region before that one is uploaded whole.
    pub fn upload_words(&self, used: &[usize]) -> usize {
        assert_eq!(used.len(), self.groups.len(), "pages used per group");
        let last = self
            .groups
            .iter()
            .enumerate()
            .max_by_key(|(_, r)| r.indices)
            .map(|(g, _)| g);
        match last {
            None => self.sample_rows + self.rows as usize * SAMPLE_ROW_WORDS,
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
        let r = self.rows as usize;
        let mut v = vec![
            (self.token_ids, r),
            (self.positions, r),
            (self.logit_rows, r),
            (self.kv_chunk_size, 1),
            (self.sample_rows, r * SAMPLE_ROW_WORDS),
        ];
        for g in &self.groups {
            v.extend([
                (g.kv_block, r),
                (g.kv_slot, r),
                (g.q_indptr, r + 1),
                (g.indptr, r + 1),
                (g.last_page_len, r),
                (g.request_indices, r),
                (g.qo_tile_indices, r),
                (g.kv_tile_indices, r),
                (g.indices, g.indices_cap),
            ]);
        }
        v
    }
}

/// A decode step's real rows, as the executor gathered and checked them
/// (the same values its eager path uploads).
#[derive(Clone, Copy, Debug)]
pub struct DecodeRows<'a> {
    pub tokens: &'a [u32],
    pub positions: &'a [u32],
    /// Per group, per row: the block and offset its KV goes to.
    pub kv_targets: &'a [Vec<(u32, u32)>],
    /// Per group, per row: its attention request.
    pub requests: &'a [Vec<AttnRequest>],
    /// Per row: its sample row, reading its own logits row.
    pub samples: &'a [SampleRow],
}

/// A packed step table and how much of it to upload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedStep {
    pub words: Vec<u32>,
    /// Words from the start the step must upload.
    pub upload: usize,
}

/// The row a padding row's attention request is: one query over the one
/// position of the pad block it wrote.
pub fn pad_request(row: u32, pad_block: u32) -> AttnRequest {
    AttnRequest {
        q_start: row,
        qo_len: 1,
        pages: vec![pad_block],
        kv_len: 1,
    }
}

/// A padding row's sample row: greedy over real row 0's logits (a padding
/// row's own logits are never sampled).
pub fn pad_sample() -> SampleRow {
    SampleRow {
        logit_row: 0,
        ..SampleRow::default()
    }
}

/// The step table for `real`'s rows padded to `layout.rows`: real rows
/// first, as given; padding rows after them ([`PAD_TOKEN`] at
/// [`PAD_POSITION`], KV and pages in each group's pad block, sample rows from
/// [`pad_sample`]). Every group's work list, padding included, is made by
/// [`HostPlan::new`], so it passes the eager path's checks; it must be one
/// work item per row at [`DECODE_TILE`], within the group's page
/// reservation. Refuses (changing nothing) anything else.
pub fn pack(
    layout: &DecodeLayout,
    groups: &[GroupSlots],
    real: &DecodeRows<'_>,
) -> Result<PackedStep> {
    let bad = |what: String| Err(CudaError::new(format!("decode step: {what}")));
    let rows = layout.rows as usize;
    let n = real.tokens.len();
    if n > rows {
        return bad(format!("{n} rows past the rung's {rows}"));
    }
    if real.positions.len() != n || real.samples.len() != n {
        return bad(format!(
            "{n} tokens, {} positions, {} sample rows",
            real.positions.len(),
            real.samples.len()
        ));
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
    for (i, s) in real.samples.iter().enumerate() {
        if s.logit_row as usize != i {
            return bad(format!("row {i} samples logits row {}", s.logit_row));
        }
    }
    let rows32 = layout.rows;
    let mut w = vec![0u32; layout.words];
    let put = |w: &mut [u32], at: usize, xs: &mut dyn Iterator<Item = u32>| {
        for (i, x) in xs.enumerate() {
            w[at + i] = x;
        }
    };
    let word = |x: i32| u32::try_from(x).expect("HostPlan indices are non-negative");
    let pad = n..rows;
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
    put(&mut w, layout.logit_rows, &mut (0..rows32));
    for (i, s) in real
        .samples
        .iter()
        .copied()
        .chain(pad.clone().map(|_| pad_sample()))
        .enumerate()
    {
        let at = layout.sample_rows + i * SAMPLE_ROW_WORDS;
        w[at..at + SAMPLE_ROW_WORDS].copy_from_slice(&sample_row_words(&s));
    }
    let mut used = Vec::with_capacity(g_count);
    for (g, (slots, region)) in groups.iter().zip(&layout.groups).enumerate() {
        let targets = &real.kv_targets[g];
        let requests = &real.requests[g];
        if targets.len() != n || requests.len() != n {
            return bad(format!(
                "group {g}: {} KV targets and {} requests for {n} rows",
                targets.len(),
                requests.len()
            ));
        }
        if let Some((i, r)) = requests
            .iter()
            .enumerate()
            .find(|(i, r)| r.qo_len != 1 || r.q_start as usize != *i)
        {
            return bad(format!("group {g}: row {i} is not one decode query: {r:?}"));
        }
        let mut padded = requests.clone();
        padded.extend(
            (n..rows)
                .map(|i| pad_request(u32::try_from(i).expect("rows is a u32"), slots.pad_block)),
        );
        let plan = HostPlan::new(&padded, slots.group_size, slots.page_size)?;
        if plan.tile() != DECODE_TILE || plan.work_items() != rows || plan.num_requests() != rows {
            return bad(format!(
                "group {g}: tile {}, {} work items for {rows} rows",
                plan.tile(),
                plan.work_items()
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
                .map(|t| t.0)
                .chain(pad.clone().map(|_| slots.pad_block)),
        );
        put(
            &mut w,
            region.kv_slot,
            &mut targets
                .iter()
                .map(|t| t.1)
                .chain(pad.clone().map(|_| PAD_POSITION % slots.page_size)),
        );
        for (at, xs) in [
            (region.q_indptr, &plan.q_indptr),
            (region.indptr, &plan.indptr),
            (region.last_page_len, &plan.last_page_len),
            (region.request_indices, &plan.request_indices),
            (region.qo_tile_indices, &plan.qo_tile_indices),
            (region.kv_tile_indices, &plan.kv_tile_indices),
            (region.indices, &plan.indices),
        ] {
            put(&mut w, at, &mut xs.iter().map(|&x| word(x)));
        }
        used.push(plan.indices.len());
    }
    Ok(PackedStep {
        upload: layout.upload_words(&used),
        words: w,
    })
}

/// A [`SampleRow`] as the eight 32-bit words of its `repr(C)` layout
/// (little-endian `u64` seed), as the kernel reads it from the step table.
pub fn sample_row_words(r: &SampleRow) -> [u32; SAMPLE_ROW_WORDS] {
    let seed = r.seed.to_le_bytes();
    let half = |b: &[u8]| u32::from_le_bytes(b.try_into().expect("four bytes"));
    [
        r.temperature.to_bits(),
        r.top_k,
        r.top_p.to_bits(),
        r.min_p.to_bits(),
        half(&seed[..4]),
        half(&seed[4..]),
        r.position,
        r.logit_row,
    ]
}

/// The real rows' tokens and the status word, out of a rung's read-back
/// (`rows` tokens, then the status). Padding rows' tokens are never
/// returned.
pub fn split_readback(readback: &[u32], real: usize, rows: usize) -> (&[u32], u32) {
    assert!(
        real <= rows && readback.len() == rows + 1,
        "a rung's read-back"
    );
    (&readback[..real], readback[rows])
}

/// The captured graphs and the buffers they read and write.
pub struct DecodeGraphs {
    ladder: Vec<u32>,
    slots: Vec<GroupSlots>,
    layouts: Vec<DecodeLayout>,
    /// The step table, sized for the largest rung.
    table: CudaSlice<u32>,
    /// `rows` sampled tokens, then the status word.
    readback: CudaSlice<u32>,
    graphs: Vec<CudaGraph>,
    /// Device memory each capture took (free memory before minus after).
    capture_bytes: Vec<i64>,
}

impl std::fmt::Debug for DecodeGraphs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodeGraphs")
            .field("ladder", &self.ladder)
            .field("captured", &self.graphs.len())
            .finish_non_exhaustive()
    }
}

/// What a decode-graph launch needs besides the graphs' own buffers.
pub(crate) struct ProgramArgs<'a> {
    pub model: &'a mut GpuModel,
    pub kv: &'a KvStore,
    /// The sampler's distribution scratch (`rows × sampleable` f64).
    pub probs: u64,
    pub vocab: u32,
    pub sampleable: u32,
}

impl DecodeGraphs {
    /// Allocate the step table and read-back for `ladder` over `slots`;
    /// nothing is captured yet ([`DecodeGraphs::capture`]).
    pub(crate) fn new(gpu: &Gpu, ladder: Vec<u32>, slots: Vec<GroupSlots>) -> Result<DecodeGraphs> {
        let layouts = ladder
            .iter()
            .map(|&r| DecodeLayout::new(r, &slots))
            .collect::<Result<Vec<_>>>()?;
        let words = layouts.iter().map(|l| l.words).max().unwrap_or(1);
        let rows = ladder.last().copied().unwrap_or(0) as usize;
        let s = gpu.stream();
        Ok(DecodeGraphs {
            table: s.alloc_zeros(words.max(1))?,
            readback: s.alloc_zeros(rows + 1)?,
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

    pub fn layout(&self, rung: usize) -> &DecodeLayout {
        &self.layouts[rung]
    }

    /// Device memory each rung's capture took, as the driver's free memory
    /// moved (approximate: the driver allocates in its own granules).
    pub fn capture_bytes(&self) -> &[i64] {
        &self.capture_bytes
    }

    /// Copy the uploaded prefix of a packed step into the step table.
    pub(crate) fn upload(&mut self, gpu: &Gpu, packed: &PackedStep) -> Result<()> {
        let n = packed.upload;
        if n > self.table.len() || n > packed.words.len() {
            return Err(CudaError::new("decode step table overflows"));
        }
        gpu.stream()
            .memcpy_htod(&packed.words[..n], &mut self.table.slice_mut(..n))?;
        Ok(())
    }

    /// The padded step's launches for rung `rung`: zero the padded
    /// activations' rows, the forward over every row, then sampling into the
    /// read-back. Reads only the step table, fixed buffers and the KV pools;
    /// copies nothing from the host, allocates nothing and reads nothing
    /// back, so it can be launched directly or recorded.
    ///
    /// # Safety
    ///
    /// The step table must hold a [`pack`]ed step for this rung's layout when
    /// the launches run, and `a` must be the model, pools and sampler
    /// scratch the executor sized this rung for.
    pub(crate) unsafe fn program(
        &mut self,
        gpu: &Gpu,
        rung: usize,
        a: &mut ProgramArgs<'_>,
    ) -> Result<()> {
        let layout = &self.layouts[rung];
        let rows = layout.rows as usize;
        let rows32 = layout.rows;
        let s = gpu.stream();
        let base = dptr(&self.table, s);
        let at = |off: usize| base + 4 * off as u64;
        a.model.zero_padding_rows(gpu, rows)?;
        let src = Indirect {
            tokens: rows,
            token_ids: at(layout.token_ids),
            positions: at(layout.positions),
            kv_block: layout.groups.iter().map(|r| at(r.kv_block)).collect(),
            kv_slot: layout.groups.iter().map(|r| at(r.kv_slot)).collect(),
            plans: layout
                .groups
                .iter()
                .zip(&self.slots)
                .map(|(r, g)| PlanView {
                    tile: DECODE_TILE,
                    page_size: g.page_size,
                    group_size: g.group_size,
                    work_items: rows32,
                    num_requests: rows32,
                    q_indptr: at(r.q_indptr),
                    indices: at(r.indices),
                    indptr: at(r.indptr),
                    last_page_len: at(r.last_page_len),
                    request_indices: at(r.request_indices),
                    qo_tile_indices: at(r.qo_tile_indices),
                    kv_tile_indices: at(r.kv_tile_indices),
                    kv_chunk_size: at(layout.kv_chunk_size),
                })
                .collect(),
            logit_rows: at(layout.logit_rows),
            num_logit_rows: rows,
        };
        // SAFETY: the table holds a packed step for this layout (the
        // caller's contract): every value `pack` wrote passed the eager
        // path's checks or is a padding constant inside the pad block.
        unsafe { a.model.launch(gpu, a.kv, &src, false)? };
        s.memset_zeros(&mut self.readback.slice_mut(rows..rows + 1))?;
        let rb = dptr(&self.readback, s);
        let logits = a.model.logits();
        if logits.len() < rows * a.vocab as usize {
            return Err(CudaError::new("decode rung past the logits scratch"));
        }
        let launch = SampleLaunch {
            logits: dptr(logits, s),
            logits_stride: u64::from(a.vocab),
            n: a.sampleable,
            rows: at(layout.sample_rows),
            num_rows: rows32,
            draw: Some(Stream::Sample),
            probs: a.probs,
            tokens: rb,
            status: rb + 4 * rows as u64,
        };
        // SAFETY: the sample rows read logits rows below `rows` (real rows
        // their own, padding rows row 0), inside the logits checked above;
        // the read-back holds `rows + 1` words; `probs` is the executor's
        // scratch for at least the largest rung's rows.
        unsafe { a.model.kernels.sampler.launch_sample(gpu, launch) }
    }

    /// Run `rung` once directly over an all-padding step (which checks every
    /// launch on this device), then capture it, replay the capture once and
    /// keep it; for every rung, in order. Called once, at construction.
    pub(crate) fn capture(&mut self, gpu: &Gpu, a: &mut ProgramArgs<'_>) -> Result<()> {
        let s = gpu.stream().clone();
        let targets = vec![Vec::new(); self.slots.len()];
        let requests = vec![Vec::new(); self.slots.len()];
        let none = DecodeRows {
            tokens: &[],
            positions: &[],
            kv_targets: &targets,
            requests: &requests,
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
                CudaError::new(format!("rung {}: the capture is empty", self.ladder[rung]))
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
                .ok_or_else(|| CudaError::new("no captured decode graph for this rung"))?;
            Ok(graph.launch()?)
        }
    }

    /// The rung's read-back: `rows` tokens and the status word (one copy,
    /// which waits for the step).
    pub(crate) fn readback(&self, gpu: &Gpu, rung: usize) -> Result<Vec<u32>> {
        let rows = self.layouts[rung].rows as usize;
        Ok(gpu.stream().clone_dtoh(&self.readback.slice(..rows + 1))?)
    }

    fn check_status(&self, gpu: &Gpu, rung: usize) -> Result<()> {
        let rows = self.layouts[rung].rows as usize;
        let rb = self.readback(gpu, rung)?;
        let (_, status) = split_readback(&rb, 0, rows);
        if status & STATUS_NON_FINITE != 0 {
            return Err(CudaError::new(format!(
                "rung {}: non-finite logits over padding",
                self.ladder[rung]
            )));
        }
        Ok(())
    }
}

fn free_memory() -> Result<i64> {
    let (free, _) = cudarc::driver::result::mem_get_info()?;
    narrow(free, "free device memory")
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidola_engine::executor::SeqEntry;
    use eidola_engine::sampling::SamplingParams;

    #[test]
    fn the_kill_switch_parses_exactly_on_and_off() {
        assert_eq!(CudaGraphs::parse("on"), Ok(CudaGraphs::On));
        assert_eq!(CudaGraphs::parse("off"), Ok(CudaGraphs::Off));
        for bad in ["", "ON", "Off", "true", "1", "on ", " off", "yes", "auto"] {
            assert!(CudaGraphs::parse(bad).is_err(), "{bad:?}");
        }
        for g in [CudaGraphs::On, CudaGraphs::Off] {
            assert_eq!(CudaGraphs::parse(g.as_str()), Ok(g));
        }
    }

    fn bucket(max_seqs: u32, max_tokens: u32) -> Bucket {
        Bucket {
            max_seqs,
            max_tokens,
        }
    }

    #[test]
    fn the_ladder_is_powers_of_two_and_the_buckets() {
        assert_eq!(
            decode_ladder(&[bucket(64, 8192)]),
            vec![1, 2, 4, 8, 16, 32, 64]
        );
        assert_eq!(
            decode_ladder(&[bucket(48, 8192)]),
            vec![1, 2, 4, 8, 16, 32, 48]
        );
        // Rows are bounded by tokens too: one host token per decode row.
        assert_eq!(decode_ladder(&[bucket(64, 12)]), vec![1, 2, 4, 8, 12]);
        assert_eq!(
            decode_ladder(&[bucket(1, 16), bucket(6, 256), bucket(8, 256)]),
            vec![1, 2, 4, 6, 8]
        );
        assert_eq!(decode_ladder(&[bucket(1, 1)]), vec![1]);
        assert_eq!(decode_ladder(&[]), Vec::<u32>::new());
    }

    /// The padded step takes the expert layout the eager one would: masked
    /// exactly when its rows are at most 128.
    #[test]
    fn a_rung_is_masked_exactly_when_its_rows_are() {
        const MASKED: usize = 128;
        for cap in 1..=600u32 {
            let ladder = decode_ladder(&[bucket(cap, 4096)]);
            assert!(ladder.windows(2).all(|w| w[0] < w[1]), "{ladder:?}");
            assert_eq!(*ladder.last().unwrap(), cap);
            for rows in 1..=cap as usize {
                let rung = ladder[rung_for(&ladder, rows).unwrap()] as usize;
                assert!(rung >= rows, "cap {cap}: {rows} rows in rung {rung}");
                assert_eq!(
                    rows <= MASKED,
                    rung <= MASKED,
                    "cap {cap}: {rows} rows in rung {rung}"
                );
                // The smallest rung that fits.
                assert!(
                    ladder
                        .iter()
                        .all(|&r| r as usize >= rung || (r as usize) < rows)
                );
                // Padding stays under half the rung past the first rungs.
                if rung.is_power_of_two() && rung > 1 {
                    assert!(rows > rung / 2, "cap {cap}: {rows} rows in rung {rung}");
                }
            }
            assert_eq!(rung_for(&ladder, cap as usize + 1), None);
            assert_eq!(rung_for(&ladder, 0), None);
        }
    }

    fn entry(num_tokens: u32, sample: bool, num_drafts: u32) -> SeqEntry {
        SeqEntry {
            slot: 0,
            token_start: 0,
            num_tokens,
            context_len: 5,
            num_drafts,
            sample,
            sampling: SamplingParams::greedy(),
        }
    }

    fn step_of(seqs: Vec<SeqEntry>) -> StepInput {
        StepInput {
            bucket: bucket(8, 64),
            maintenance: vec![],
            table_updates: vec![],
            seqs,
            token_ids: vec![],
            positions: vec![],
            return_logits: false,
        }
    }

    #[test]
    fn only_pure_decode_steps_replay() {
        assert_eq!(decode_rows(&step_of(vec![entry(1, true, 0); 3])), Some(3));
        assert_eq!(decode_rows(&step_of(vec![])), None);
        assert_eq!(
            decode_rows(&step_of(vec![entry(1, true, 0), entry(2, true, 0)])),
            None
        );
        assert_eq!(
            decode_rows(&step_of(vec![entry(1, true, 0), entry(1, false, 0)])),
            None
        );
        assert_eq!(decode_rows(&step_of(vec![entry(1, true, 2)])), None);
    }

    /// The page bound holds for every decode position, and is reached.
    #[test]
    fn page_bounds_cover_every_decode_row() {
        for bs in 1..=8u32 {
            for window in 1..=40u32 {
                let kind = AttentionKind::Sliding { window };
                let max_len = 200u32;
                let max_blocks = max_len.div_ceil(bs);
                let bound = max_decode_pages(kind, bs, max_blocks);
                let mut most = 0;
                for p in 0..max_len {
                    let pages = p / bs - kind.first_visible(p) / bs + 1;
                    assert!(
                        pages <= bound,
                        "bs {bs} window {window} p {p}: {pages} > {bound}"
                    );
                    most = most.max(pages);
                }
                assert_eq!(most, bound, "bs {bs} window {window}");
            }
            assert_eq!(max_decode_pages(AttentionKind::Full, bs, 77), 77);
        }
        // Never more than the sequence has.
        assert_eq!(
            max_decode_pages(AttentionKind::Sliding { window: 128 }, 16, 3),
            3
        );
    }

    /// Flash's two groups: global (GQA 16, unbounded pages) and sliding
    /// (GQA 8, window 128), each with 64 blocks the seam may name (`1..64`;
    /// the pad block is 64).
    fn flash_slots(max_blocks: u32) -> Vec<GroupSlots> {
        vec![
            GroupSlots {
                group_size: 16,
                page_size: 16,
                pad_block: 64,
                max_pages: max_decode_pages(AttentionKind::Full, 16, max_blocks),
            },
            GroupSlots {
                group_size: 8,
                page_size: 16,
                pad_block: 64,
                max_pages: max_decode_pages(AttentionKind::Sliding { window: 128 }, 16, max_blocks),
            },
        ]
    }

    #[test]
    fn the_layout_is_disjoint_aligned_and_uploads_a_prefix() {
        for rows in [1u32, 2, 3, 8, 64, 100] {
            let slots = flash_slots(64);
            let l = DecodeLayout::new(rows, &slots).unwrap();
            let mut regions = l.regions();
            regions.sort();
            for w in regions.windows(2) {
                assert!(w[0].0 + w[0].1 <= w[1].0, "rows {rows}: {regions:?}");
            }
            let (last, len) = *regions.last().unwrap();
            assert_eq!(last + len, l.words);
            assert_eq!(l.sample_rows % 2, 0, "u64 seeds need 8-byte rows");
            // The smaller reservation (sliding) is placed first; the global
            // list's used part ends the upload.
            assert!(l.groups[1].indices < l.groups[0].indices);
            assert_eq!(l.groups[0].indices_cap, rows as usize * 64);
            assert_eq!(l.groups[1].indices_cap, rows as usize * 9);
            assert_eq!(l.upload_words(&[0, 9]), l.groups[0].indices);
            assert_eq!(l.upload_words(&[5, 0]), l.groups[0].indices + 5);
            // Every region but the last-placed one lies inside the upload.
            for (off, len) in l.regions() {
                if off != l.groups[0].indices {
                    assert!(off + len <= l.upload_words(&[0, 0]));
                }
            }
        }
    }

    #[test]
    fn sample_rows_pack_as_their_repr_c_bytes() {
        let r = SampleRow {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            seed: 0x0123_4567_89ab_cdef,
            position: 1234,
            logit_row: 5,
        };
        // SAFETY: `SampleRow` is `repr(C)` plain data, 32 bytes, no padding.
        let bytes: [u8; 32] = unsafe { std::mem::transmute(r) };
        let words: Vec<u8> = sample_row_words(&r)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        assert_eq!(words, bytes);
    }

    /// Three real decode rows of Flash's two groups, as the executor would
    /// gather them.
    struct Rows {
        tokens: Vec<u32>,
        positions: Vec<u32>,
        kv_targets: Vec<Vec<(u32, u32)>>,
        requests: Vec<Vec<AttnRequest>>,
        samples: Vec<SampleRow>,
    }

    impl Rows {
        fn three() -> Rows {
            // Contexts 17, 40 and 150 (past the window), each row's blocks
            // 1.., 11.., 21.. in both groups.
            let contexts = [17u32, 40, 150];
            let mut kv_targets = vec![Vec::new(), Vec::new()];
            let mut requests = vec![Vec::new(), Vec::new()];
            for (i, &c) in contexts.iter().enumerate() {
                let i32 = u32::try_from(i).unwrap();
                let base = 1 + 10 * i32;
                let block = |pos: u32| base + pos / 16;
                for (g, window) in [(0, None), (1, Some(128u32))] {
                    kv_targets[g].push((block(c), c % 16));
                    let first = window.map_or(0, |w| (c + 1).saturating_sub(w)) / 16;
                    requests[g].push(AttnRequest {
                        q_start: i32,
                        qo_len: 1,
                        pages: (first..=c / 16).map(|p| base + p).collect(),
                        kv_len: c + 1 - first * 16,
                    });
                }
            }
            let params = SamplingParams::greedy();
            Rows {
                tokens: vec![11, 22, 33],
                positions: contexts.to_vec(),
                kv_targets,
                requests,
                samples: (0..3)
                    .map(|i| SampleRow::new(&params, contexts[i as usize] + 1, i))
                    .collect(),
            }
        }

        fn rows(&self) -> DecodeRows<'_> {
            DecodeRows {
                tokens: &self.tokens,
                positions: &self.positions,
                kv_targets: &self.kv_targets,
                requests: &self.requests,
                samples: &self.samples,
            }
        }
    }

    fn read(w: &[u32], at: usize, n: usize) -> Vec<u32> {
        w[at..at + n].to_vec()
    }

    /// Padding rows touch only the pad block, sample real row 0 greedily,
    /// and leave every real row's values exactly as the eager path's plan
    /// has them.
    #[test]
    fn padding_rows_are_inert() {
        let rows = Rows::three();
        let slots = flash_slots(64);
        let l = DecodeLayout::new(8, &slots).unwrap();
        let packed = pack(&l, &slots, &rows.rows()).unwrap();
        let w = &packed.words;
        assert_eq!(read(w, l.token_ids, 8), [11, 22, 33, 0, 0, 0, 0, 0]);
        assert_eq!(read(w, l.positions, 8), [17, 40, 150, 0, 0, 0, 0, 0]);
        assert_eq!(read(w, l.logit_rows, 8), (0..8).collect::<Vec<_>>());
        for i in 0..8 {
            let at = l.sample_rows + i * SAMPLE_ROW_WORDS;
            let want = if i < 3 { rows.samples[i] } else { pad_sample() };
            assert_eq!(w[at..at + SAMPLE_ROW_WORDS], sample_row_words(&want));
        }
        assert_eq!(pad_sample().temperature, 0.0, "greedy");
        for (g, (s, r)) in slots.iter().zip(&l.groups).enumerate() {
            // KV writes: real rows where the executor put them, padding rows
            // only into the pad block, which the seam can never name.
            let blocks = read(w, r.kv_block, 8);
            assert_eq!(
                blocks[..3],
                rows.kv_targets[g].iter().map(|t| t.0).collect::<Vec<_>>()[..]
            );
            assert!(blocks[3..].iter().all(|&b| b == s.pad_block));
            assert!(blocks[..3].iter().all(|&b| (1..s.pad_block).contains(&b)));
            let offs = read(w, r.kv_slot, 8);
            assert_eq!(
                offs[..3],
                rows.kv_targets[g].iter().map(|t| t.1).collect::<Vec<_>>()[..]
            );
            assert!(offs[3..].iter().all(|&o| o == 0));
            // Attention: the real rows' arrays are exactly the unpadded
            // plan's; each padding row reads one position of the pad block.
            let eager = HostPlan::new(&rows.requests[g], s.group_size, s.page_size).unwrap();
            let words = |xs: &[i32]| {
                xs.iter()
                    .map(|&x| u32::try_from(x).unwrap())
                    .collect::<Vec<_>>()
            };
            let indptr = read(w, r.indptr, 9);
            assert_eq!(indptr[..4], words(&eager.indptr)[..]);
            let real_pages = eager.indices.len();
            assert_eq!(read(w, r.indices, real_pages), words(&eager.indices));
            assert_eq!(read(w, r.indices + real_pages, 5), [s.pad_block; 5]);
            assert_eq!(
                (4..9)
                    .map(|i| indptr[i] - indptr[i - 1])
                    .collect::<Vec<_>>(),
                [1; 5]
            );
            let last = read(w, r.last_page_len, 8);
            assert_eq!(last[..3], words(&eager.last_page_len)[..]);
            assert_eq!(last[3..], [1; 5]);
            assert_eq!(read(w, r.q_indptr, 9), (0..9).collect::<Vec<_>>());
            assert_eq!(read(w, r.request_indices, 8), (0..8).collect::<Vec<_>>());
            assert_eq!(read(w, r.qo_tile_indices, 8), [0; 8]);
            assert_eq!(read(w, r.kv_tile_indices, 8), [0; 8]);
            assert_eq!(eager.tile(), DECODE_TILE);
        }
        assert_eq!(w[l.kv_chunk_size], 0);
        // The upload ends with the global group's used pages.
        let global_pages = HostPlan::new(&rows.requests[0], 16, 16)
            .unwrap()
            .indices
            .len()
            + 5;
        assert_eq!(packed.upload, l.groups[0].indices + global_pages);
        // A full rung has no padding rows; an empty one is all padding.
        let full = DecodeLayout::new(3, &slots).unwrap();
        let p = pack(&full, &slots, &rows.rows()).unwrap();
        assert_eq!(read(&p.words, full.groups[0].kv_block, 3), [2, 13, 30]);
        let (targets, requests) = (vec![Vec::new(); 2], vec![Vec::new(); 2]);
        let none = DecodeRows {
            tokens: &[],
            positions: &[],
            kv_targets: &targets,
            requests: &requests,
            samples: &[],
        };
        let p = pack(&l, &slots, &none).unwrap();
        for (s, r) in slots.iter().zip(&l.groups) {
            assert_eq!(read(&p.words, r.kv_block, 8), [s.pad_block; 8]);
            assert_eq!(read(&p.words, r.indices, 8), [s.pad_block; 8]);
        }
    }

    #[test]
    fn steps_the_rung_cannot_run_are_refused() {
        let slots = flash_slots(64);
        let rows = Rows::three();
        let refused = |l: &DecodeLayout, r: &Rows| {
            pack(l, &slots, &r.rows()).expect_err("refused");
        };
        // More rows than the rung.
        refused(&DecodeLayout::new(2, &slots).unwrap(), &rows);
        let l = DecodeLayout::new(4, &slots).unwrap();
        // A prefill row.
        let mut r = Rows::three();
        r.requests[0][1].qo_len = 2;
        refused(&l, &r);
        // A row sampling another row's logits.
        let mut r = Rows::three();
        r.samples[2].logit_row = 0;
        refused(&l, &r);
        // A request the eager plan refuses (pages not covering its KV).
        let mut r = Rows::three();
        r.requests[1][0].kv_len = 40;
        refused(&l, &r);
        // More pages than the reservation.
        let tight = flash_slots(2);
        let l2 = DecodeLayout::new(4, &tight).unwrap();
        pack(&l2, &tight, &rows.rows()).expect_err("past the reservation");
        // A missing KV target.
        let mut r = Rows::three();
        r.kv_targets[1].pop();
        refused(&l, &r);
    }

    #[test]
    fn padding_tokens_are_never_returned() {
        let rb = [7, 8, 9, 1000, 1000, 0b1];
        let (tokens, status) = split_readback(&rb, 3, 5);
        assert_eq!(tokens, [7, 8, 9]);
        assert_eq!(status, 1);
    }
}
