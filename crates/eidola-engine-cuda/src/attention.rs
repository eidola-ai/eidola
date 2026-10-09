//! Paged attention: FlashInfer's FA2 prefill template with the attention-sink
//! variant (`flashinfer_fa2_sink_paged`), BF16 Q/KV/O, head dims 192/128,
//! causal, with an optional sliding window and per-head sink logits. One
//! kernel family serves prefill, decode and verification.
//!
//! **Batch invariance.** A row's output depends only on its own query, the
//! keys it sees and its position: never on the other rows of its step, how
//! the step was chunked, or which query tile it landed in. Three rules make
//! that hold, each enforced here or in the kernel
//! (`eidola-engine-kernels/csrc/flashinfer_fa2_sink_paged.cu`):
//!
//! - **One reduction shape.** Every instance reduces a row's softmax with one
//!   KV warp over [`KV_TILE`]-key tiles; only the query tile (16, 64 or 128
//!   rows, chosen per step for speed) differs, and it does not change a
//!   row's arithmetic.
//! - **Anchored sliding tiles** ([`Reduction::Anchored`]). A sliding
//!   request's pages start at an anchor, its layer's origin plus a multiple
//!   of [`KV_TILE`] positions ([`Reduction::kv_start`]), and the kernel
//!   starts every query tile's keys at a multiple of [`KV_TILE`] from there,
//!   masking each row's true window per key. A row's tile boundaries are
//!   therefore absolute positions, whatever rows share its query tile.
//! - **A fixed split of global keys** ([`Reduction::Split`]). Global layers'
//!   keys are split at absolute multiples of [`SPLIT_KEYS`], each chunk its
//!   own work item writing a partial state, and the merge kernel combines a
//!   row's partials in chunk order. Requests are cut at multiples of
//!   [`SPLIT_KEYS`] of their query positions ([`HostPlan::new`]), so a row at
//!   position `p` merges exactly chunks `0 ..= p / SPLIT_KEYS`, and every row
//!   of a work item sees at least one key of its chunk. The sink is added to
//!   chunk 0's state only, so the merge carries it into each row once.
//!
//! The kernel takes FlashInfer's `PagedParams` by value inside [`AttnParams`]
//! ([`PagedParams`] is its `repr(C)` mirror, 296 bytes; [`AttnParams`] is
//! checked against the image's launch contract). Every CTA walks the work
//! list from its block index in steps of the grid up to a count it reads
//! from device memory, so a captured launch with a fixed grid serves a step
//! whose global work list is device data (its chunk count grows with the
//! context).

use std::ffi::c_void;
use std::ops::Range;

use cudarc::driver::CudaSlice;
use eidola_engine::spec::AttentionKind;

use crate::launch::dptr;
use crate::module::{Kernel, KernelModule};
use crate::{CudaError, Gpu, Result, narrow};

pub const HEAD_DIM_QK: u32 = 192;
pub const HEAD_DIM_VO: u32 = 128;

/// Keys per KV tile of every instance: one KV warp of four 16-key MMA
/// fragments. Anchors are multiples of it.
pub const KV_TILE: u32 = 64;

/// Global layers' keys are split at absolute multiples of this many
/// positions, each chunk a work item, and merged in chunk order. Fixed: a
/// row's reduction may not depend on the step.
pub const SPLIT_KEYS: u32 = 1024;

/// CTAs per KV head a captured split launch walks its device-sized work list
/// with: Flash's 4 global KV heads make 592, the 16-row instance's resident
/// capacity on 148 SMs (four per SM by shared memory).
pub const PERSISTENT_CTAS: u32 = 148;

/// Blocks of the persistent merge kernel, at most (it strides over rows ×
/// heads): four per SM of a 148-SM part.
const MERGE_BLOCKS: u32 = 148 * 4;

/// Bytes of one merged partial row: the BF16 output of every query head and
/// its f32 log-sum-exp, per query head (`num_qo_heads` of them).
pub const PARTIAL_BYTES_PER_HEAD: usize = HEAD_DIM_VO as usize * 2 + 4;

/// `cuda::fast_mod_div<uint32_t>` (CCCL 13.2) wrapped as FlashInfer's
/// `uint_fastdiv`: `{divisor, multiplier, add, shift}` then the plain divisor.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UintFastdiv {
    divisor: u32,
    multiplier: u32,
    add: u32,
    shift: i32,
    d: u32,
}

impl UintFastdiv {
    pub fn new(d: u32) -> UintFastdiv {
        let divisor = d.max(1);
        let shift = 31 - divisor.leading_zeros();
        let (multiplier, add) = if divisor.is_power_of_two() {
            (0, 0)
        } else {
            let k = 32 + shift;
            let mul = ((1u64 << k) + (1u64 << shift)) / divisor as u64;
            let low = (1u64 << k) / divisor as u64;
            // `divisor` lies strictly between 2^shift and 2^(shift + 1), so
            // `mul` < 2^32.
            let mul32 = u32::try_from(mul).expect("fast_mod_div multiplier fits 32 bits");
            (mul32, u32::from(low == mul))
        };
        UintFastdiv {
            divisor,
            multiplier,
            add,
            shift: i32::try_from(shift).expect("a shift below 32"),
            d,
        }
    }
}

/// `paged_kv_t<nv_bfloat16, int32_t>` with independent K and V strides.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PagedKv {
    pub page_size: UintFastdiv,
    pub num_heads: u32,
    pub head_dim: u32,
    pub batch_size: u32,
    pub stride_page: u32,
    pub stride_n: u32,
    pub stride_h: u32,
    pub v_stride_page: u32,
    pub v_stride_n: u32,
    pub v_stride_h: u32,
    pub k_data: u64,
    pub v_data: u64,
    pub indices: u64,
    pub indptr: u64,
    pub last_page_len: u64,
    pub rope_pos_offset: u64,
}

/// FlashInfer's generated `PagedParams` for the sink variant.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PagedParams {
    pub q: u64,
    pub paged_kv: PagedKv,
    pub q_indptr: u64,
    pub o: u64,
    pub lse: u64,
    pub group_size: UintFastdiv,
    pub sink: u64,
    pub sm_scale: f64,
    pub num_qo_heads: u32,
    pub q_stride_n: i32,
    pub q_stride_h: i32,
    pub k_sf_stride_page: u32,
    pub k_sf_stride_n: u32,
    pub k_sf_stride_h: u32,
    pub v_sf_stride_page: u32,
    pub v_sf_stride_n: u32,
    pub v_sf_stride_h: u32,
    pub window_left: i32,
    pub request_indices: u64,
    pub qo_tile_indices: u64,
    pub kv_tile_indices: u64,
    pub merge_indptr: u64,
    pub o_indptr: u64,
    pub block_valid_mask: u64,
    pub kv_chunk_size_ptr: u64,
    pub max_total_num_rows: u32,
    pub total_num_rows: u64,
    pub padded_batch_size: u32,
    pub partition_kv: bool,
}

/// The kernels' by-value argument (`EidolaAttnParams`): FlashInfer's
/// parameters and the device word holding the number of work items.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AttnParams {
    pub paged: PagedParams,
    pub work_items: u64,
}

const _: () = {
    assert!(std::mem::size_of::<UintFastdiv>() == 20);
    assert!(std::mem::size_of::<PagedKv>() == 104);
    assert!(std::mem::size_of::<PagedParams>() == 296);
    assert!(std::mem::offset_of!(PagedParams, paged_kv) == 8);
    assert!(std::mem::offset_of!(PagedParams, q_indptr) == 112);
    assert!(std::mem::offset_of!(PagedParams, group_size) == 136);
    assert!(std::mem::offset_of!(PagedParams, sink) == 160);
    assert!(std::mem::offset_of!(PagedParams, window_left) == 212);
    assert!(std::mem::offset_of!(PagedParams, kv_chunk_size_ptr) == 264);
    assert!(std::mem::offset_of!(PagedParams, partition_kv) == 292);
    assert!(std::mem::size_of::<AttnParams>() == 304);
    assert!(std::mem::offset_of!(AttnParams, work_items) == 296);
};

/// How a group's attention reduces a row over its keys (see the module
/// docs). Global attention splits; a sliding window anchors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    /// Keys split at absolute multiples of [`SPLIT_KEYS`] and merged; pages
    /// listed from position 0.
    Split,
    /// Pages listed from `origin` plus a multiple of [`KV_TILE`] positions
    /// (and of the page size), the true window masked per key. `origin` is
    /// the first position the layer holds KV for: 0 for a target layer,
    /// `d + 1` for MTP depth `d`.
    Anchored { origin: u32 },
}

/// The least common multiple of [`KV_TILE`] and `page_size`: where anchors
/// may sit in a paged pool.
fn anchor_stride(page_size: u32) -> u64 {
    let (a, b) = (u64::from(KV_TILE), u64::from(page_size.max(1)));
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    a / x * b
}

impl Reduction {
    /// Where a request whose first query first sees position `first` lists
    /// its pages from: 0 for a split; for an anchored group the largest
    /// anchor at or below `first` (and at or above the origin). The anchor
    /// is a page boundary when the origin is one.
    pub fn kv_start(self, first: u32, page_size: u32) -> u32 {
        match self {
            Reduction::Split => 0,
            Reduction::Anchored { origin } => {
                let first = u64::from(first.max(origin));
                let stride = anchor_stride(page_size);
                let o = u64::from(origin);
                u32::try_from(o + (first - o) / stride * stride)
                    .expect("an anchor at or below a u32 position")
            }
        }
    }

    /// The most pages one request of `queries` consecutive queries can
    /// list: every block of the longest sequence for a split; for an
    /// anchored window `W`, the pages from the anchor below the first
    /// query's window through the last query, and never more than the
    /// sequence's blocks.
    pub fn max_pages(
        self,
        window: Option<u32>,
        queries: u32,
        page_size: u32,
        max_blocks_per_seq: u32,
    ) -> u32 {
        match (self, window) {
            (Reduction::Split, _) | (Reduction::Anchored { .. }, None) => max_blocks_per_seq,
            (Reduction::Anchored { .. }, Some(w)) => {
                // The anchor sits at most `stride - 1` below the first
                // query's first visible position, which sits `W - 1` below
                // the first query; the last query is `queries - 1` past it.
                let span =
                    u64::from(w.max(1)) + u64::from(queries.max(1)) + anchor_stride(page_size) - 3;
                let pages = span / u64::from(page_size.max(1)) + 1;
                u32::try_from(pages.min(u64::from(max_blocks_per_seq))).expect("below a u32")
            }
        }
    }

    pub fn is_split(self) -> bool {
        self == Reduction::Split
    }

    /// A target group's reduction: global attention splits, a sliding window
    /// anchors at position 0.
    pub fn target(kind: AttentionKind) -> Reduction {
        match kind {
            AttentionKind::Full => Reduction::Split,
            AttentionKind::Sliding { .. } => Reduction::Anchored { origin: 0 },
        }
    }
}

/// The page an anchored request lists for positions wholly before every
/// one of its queries' windows (between its anchor and its first query's
/// first visible page): the null block (or, in a planar pool, its first
/// row), which nothing writes and which reads as zeros. The kernel masks
/// every such key, so its key and value never reach a row (a masked key's
/// probability is exactly 0, and a zero value adds exactly 0), and the
/// blocks the serving core may already have released behind the window are
/// never read.
pub const MASKED_PAGE: u32 = eidola_engine::spec::NULL_BLOCK;

/// One request (sequence row) of an attention call: its query rows and the
/// KV pages it reads, in position order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttnRequest {
    /// First query row in Q/O.
    pub q_start: u32,
    pub qo_len: u32,
    /// Physical pages covering KV positions `kv_start .. kv_start + kv_len`
    /// ([`MASKED_PAGE`] for those wholly before every query's window).
    pub pages: Vec<u32>,
    /// The position the first listed page starts at ([`Reduction::kv_start`]).
    pub kv_start: u32,
    /// KV positions covered, counted from `kv_start`; the last query sits at
    /// `kv_start + kv_len - 1`.
    pub kv_len: u32,
}

/// The query tile (16, 64 or 128 rows) for a step whose longest packed query
/// run (query rows × GQA group size) is `packed`: FlashInfer's rule. Only
/// speed depends on it.
pub fn tile_for(packed: u64) -> u32 {
    if packed <= 16 {
        16
    } else if packed <= 64 {
        64
    } else {
        128
    }
}

/// One launch's work list as the host derives it, before upload: FlashInfer's
/// arrays, plus (split) where each request's partial rows start and each
/// query row's partials for the merge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostPass {
    /// Query rows of the step this pass computes.
    pub(crate) q_rows: Range<u32>,
    pub(crate) q_indptr: Vec<i32>,
    pub(crate) indices: Vec<i32>,
    pub(crate) indptr: Vec<i32>,
    pub(crate) last_page_len: Vec<i32>,
    pub(crate) request_indices: Vec<i32>,
    pub(crate) qo_tile_indices: Vec<i32>,
    pub(crate) kv_tile_indices: Vec<i32>,
    /// Split: the first partial row of each request (FlashInfer's
    /// `o_indptr`: request `i`'s query `q`, chunk `c` at `o_indptr[i] + q ×
    /// chunks + c`), then the partial rows used. Anchored: `q_indptr`.
    pub(crate) o_indptr: Vec<i32>,
    /// Split: per query row of the pass, its first partial row, then one
    /// past the last (the merge's `indptr`). Anchored: empty.
    pub(crate) merge_indptr: Vec<i32>,
}

impl HostPass {
    pub fn work_items(&self) -> usize {
        self.request_indices.len()
    }

    pub fn num_requests(&self) -> usize {
        self.indptr.len() - 1
    }

    /// Partial rows a split pass writes (0 when anchored).
    pub fn partial_rows(&self, split: bool) -> u32 {
        if split {
            u32::try_from(*self.o_indptr.last().expect("o_indptr starts at 0"))
                .expect("checked by HostPlan::new: indices fit")
        } else {
            0
        }
    }

    /// The pass's launch shape, its arrays exactly as long as they are.
    pub fn shape(&self, plan: &HostPlan) -> Result<PassShape> {
        let split = plan.reduction.is_split();
        let work = narrow(self.work_items(), "attention work items")?;
        Ok(PassShape {
            tile: plan.tile,
            page_size: plan.page_size,
            group_size: plan.group_size,
            split,
            grid: work,
            num_requests: narrow(self.num_requests(), "attention requests")?,
            merge_rows: if split {
                self.q_rows.end - self.q_rows.start
            } else {
                0
            },
            partial_rows: self.partial_rows(split),
        })
    }
}

/// A step's work lists for one group, as the host derives them: one pass,
/// or (a split whose partial rows outgrow the scratch) several, each a
/// contiguous run of query rows. Every index is checked to fit the kernel's
/// `i32` arrays, every request's pages to cover exactly its KV (`(pages - 1)
/// × page_size < kv_len <= pages × page_size`, so the last page holds
/// 1..=page_size positions), and every request's pages to start where the
/// reduction anchors them. Its fields are private: [`HostPlan::new`], which
/// checks every relation the kernel relies on, is the only way to make one,
/// and [`Attention::upload`] takes only this type.
///
/// ```compile_fail
/// // Not constructible outside the crate: every field is private.
/// let _ = eidola_engine_cuda::attention::HostPlan {
///     group_size: 16,
///     page_size: 16,
///     tile: 16,
///     reduction: eidola_engine_cuda::attention::Reduction::Split,
///     q_rows: 1,
///     max_page: None,
///     passes: vec![],
/// };
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct HostPlan {
    /// The GQA group size and page size it was checked for: upload takes
    /// them from here, never from elsewhere.
    pub(crate) group_size: u32,
    pub(crate) page_size: u32,
    pub(crate) tile: u32,
    pub(crate) reduction: Reduction,
    /// Query rows the requests cover.
    pub(crate) q_rows: u32,
    pub(crate) max_page: Option<u32>,
    pub(crate) passes: Vec<HostPass>,
}

/// What a forward checks of a plan before launching over it: the geometry it
/// was made for and the rows and pages it reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanShape {
    pub page_size: u32,
    pub group_size: u32,
    /// Query rows the requests cover.
    pub q_rows: u32,
    /// The largest page id listed, if any.
    pub max_page: Option<u32>,
}

/// One request cut to queries that share their chunks: positions
/// `kv_len - qo_len .. kv_len` of its keys, counted from position 0.
struct Piece {
    q_start: u32,
    qo_len: u32,
    kv_len: u32,
    /// Index of the request it came from.
    from: usize,
}

impl HostPlan {
    /// Check `requests` and lay out their work lists. A split's partial rows
    /// go in passes of at most `partial_rows` rows (the scratch the caller
    /// holds); an anchored plan is one pass.
    pub fn new(
        requests: &[AttnRequest],
        group_size: u32,
        page_size: u32,
        reduction: Reduction,
        partial_rows: u32,
    ) -> Result<HostPlan> {
        let bad =
            |r: &AttnRequest, why: &str| CudaError::new(format!("attention request {r:?}: {why}"));
        let int = |x: u64| i32::try_from(x).ok();
        if group_size == 0 || page_size == 0 {
            return Err(CudaError::new(format!(
                "attention plan: group size {group_size}, page size {page_size}"
            )));
        }
        let mut q_end = 0u64;
        let mut max_page = None;
        for r in requests {
            let (kv, pages, ps) = (r.kv_len as u64, r.pages.len() as u64, page_size as u64);
            let before = pages.checked_sub(1).and_then(|n| n.checked_mul(ps));
            if before.is_none_or(|b| kv <= b || kv - b > ps) {
                return Err(bad(
                    r,
                    "its pages must cover its KV, the last one partly or wholly",
                ));
            }
            // Every query's own position is in its KV.
            if r.qo_len == 0 || r.qo_len > r.kv_len {
                return Err(bad(r, "1..=kv_len queries"));
            }
            if u64::from(r.q_start) != q_end {
                return Err(CudaError::new(
                    "attention requests must cover Q rows in order",
                ));
            }
            q_end = u64::from(r.q_start) + u64::from(r.qo_len);
            int(q_end).ok_or_else(|| bad(r, "Q rows past i32"))?;
            int(u64::from(r.kv_start) + kv).ok_or_else(|| bad(r, "positions past i32"))?;
            if !u64::from(r.kv_start).is_multiple_of(ps) {
                return Err(bad(r, "its pages must start at a page boundary"));
            }
            match reduction {
                Reduction::Split if r.kv_start != 0 => {
                    return Err(bad(r, "a split lists its pages from position 0"));
                }
                Reduction::Anchored { origin }
                    if r.kv_start < origin || !(r.kv_start - origin).is_multiple_of(KV_TILE) =>
                {
                    return Err(bad(
                        r,
                        &format!("anchored pages start at {origin} plus a multiple of {KV_TILE}"),
                    ));
                }
                _ => {}
            }
            for &page in &r.pages {
                int(page as u64).ok_or_else(|| bad(r, "a page past i32"))?;
                max_page = max_page.max(Some(page));
            }
        }
        let packed = requests
            .iter()
            .map(|r| r.qo_len as u64 * group_size as u64)
            .max()
            .unwrap_or(1);
        let tile = tile_for(packed);
        let mut plan = HostPlan {
            group_size,
            page_size,
            tile,
            reduction,
            q_rows: u32::try_from(q_end).expect("checked: fits i32"),
            max_page,
            passes: Vec::new(),
        };
        match reduction {
            Reduction::Anchored { .. } => {
                let pieces: Vec<Piece> = requests
                    .iter()
                    .enumerate()
                    .map(|(i, r)| Piece {
                        q_start: r.q_start,
                        qo_len: r.qo_len,
                        kv_len: r.kv_len,
                        from: i,
                    })
                    .collect();
                plan.passes.push(plan.pass(requests, &pieces)?);
            }
            Reduction::Split => {
                let split = SPLIT_KEYS;
                let mut pass: Vec<Piece> = Vec::new();
                let mut partials = 0u64;
                for (i, r) in requests.iter().enumerate() {
                    // The request's queries sit at positions `first ..
                    // r.kv_len` (its pages start at position 0); cut them at
                    // multiples of the split, then into pieces whose partial
                    // rows fit one pass.
                    let first = r.kv_len - r.qo_len;
                    let mut a = first;
                    while a < r.kv_len {
                        let chunk_end = (a / split + 1).saturating_mul(split).min(r.kv_len);
                        let chunks = u64::from(a / split + 1);
                        if chunks > u64::from(partial_rows) {
                            return Err(bad(
                                r,
                                &format!(
                                    "{chunks} chunks a query, past the {partial_rows} partial rows"
                                ),
                            ));
                        }
                        let most = u64::from(partial_rows) / chunks;
                        let mut b = a;
                        while b < chunk_end {
                            // Room left in this pass, in queries.
                            let room = (u64::from(partial_rows) - partials) / chunks;
                            if room == 0 {
                                plan.passes.push(plan.pass(requests, &pass)?);
                                pass.clear();
                                partials = 0;
                                continue;
                            }
                            let n = u64::from(chunk_end - b).min(room).min(most);
                            let n32 = u32::try_from(n).expect("below a u32 span");
                            pass.push(Piece {
                                q_start: r.q_start + (b - first),
                                qo_len: n32,
                                kv_len: b + n32,
                                from: i,
                            });
                            partials += n * chunks;
                            b += n32;
                        }
                        a = chunk_end;
                    }
                }
                if !pass.is_empty() || plan.passes.is_empty() {
                    plan.passes.push(plan.pass(requests, &pass)?);
                }
            }
        }
        Ok(plan)
    }

    /// One pass over `pieces` (consecutive query rows).
    fn pass(&self, requests: &[AttnRequest], pieces: &[Piece]) -> Result<HostPass> {
        let int = |x: u64, what: &str| {
            i32::try_from(x).map_err(|_| CudaError::new(format!("attention plan: {what} past i32")))
        };
        let split = self.reduction.is_split();
        let start = pieces.first().map_or(0, |p| p.q_start);
        let mut p = HostPass {
            q_rows: start..start,
            q_indptr: vec![int(u64::from(start), "Q rows")?],
            indices: vec![],
            indptr: vec![0],
            last_page_len: vec![],
            request_indices: vec![],
            qo_tile_indices: vec![],
            kv_tile_indices: vec![],
            o_indptr: vec![if split {
                0
            } else {
                int(u64::from(start), "Q rows")?
            }],
            merge_indptr: if split { vec![0] } else { vec![] },
        };
        let ps = u64::from(self.page_size);
        // Split: partial rows written so far, and merged so far.
        let (mut partials, mut merged) = (0u64, 0u64);
        for (i, piece) in pieces.iter().enumerate() {
            let r = &requests[piece.from];
            let pages = u64::from(piece.kv_len).div_ceil(ps);
            let pages_us = usize::try_from(pages).expect("within the request's pages");
            p.indices.extend(
                r.pages[..pages_us]
                    .iter()
                    .map(|&x| i32::try_from(x).expect("checked by HostPlan::new: pages fit i32")),
            );
            p.indptr.push(int(p.indices.len() as u64, "pages")?);
            p.last_page_len
                .push(int(u64::from(piece.kv_len) - (pages - 1) * ps, "a page")?);
            let q_end = u64::from(piece.q_start) + u64::from(piece.qo_len);
            p.q_indptr.push(int(q_end, "Q rows")?);
            p.q_rows.end = u32::try_from(q_end).expect("checked: fits i32");
            let chunks = if split {
                u64::from(piece.kv_len.div_ceil(SPLIT_KEYS))
            } else {
                1
            };
            let tiles = (u64::from(piece.qo_len) * u64::from(self.group_size))
                .div_ceil(u64::from(self.tile));
            for t in 0..tiles {
                for c in 0..chunks {
                    p.request_indices.push(int(i as u64, "requests")?);
                    p.qo_tile_indices.push(int(t, "query tiles")?);
                    p.kv_tile_indices.push(int(c, "KV chunks")?);
                }
            }
            if split {
                partials += u64::from(piece.qo_len) * chunks;
                p.o_indptr.push(int(partials, "partial rows")?);
                for q in 1..=u64::from(piece.qo_len) {
                    p.merge_indptr
                        .push(int(merged + q * chunks, "partial rows")?);
                }
                merged = partials;
            } else {
                p.o_indptr.push(int(q_end, "Q rows")?);
            }
        }
        Ok(p)
    }

    /// The query tile every work item uses (16, 64 or 128).
    pub fn tile(&self) -> u32 {
        self.tile
    }

    pub fn reduction(&self) -> Reduction {
        self.reduction
    }

    pub fn passes(&self) -> &[HostPass] {
        &self.passes
    }

    /// Work items over every pass (CTAs per KV head).
    pub fn work_items(&self) -> usize {
        self.passes.iter().map(HostPass::work_items).sum()
    }

    pub fn shape(&self) -> PlanShape {
        PlanShape {
            page_size: self.page_size,
            group_size: self.group_size,
            q_rows: self.q_rows,
            max_page: self.max_page,
        }
    }
}

/// A pass's launch shape: what a captured launch fixes (its arrays' contents
/// are device data).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassShape {
    pub tile: u32,
    pub page_size: u32,
    pub group_size: u32,
    /// A split (global) pass: partial rows, then the merge.
    pub split: bool,
    /// CTAs per KV head; every CTA walks the work list to its count.
    pub grid: u32,
    /// Requests the arrays hold (`paged_kv.batch_size`), padding included.
    pub num_requests: u32,
    /// Query rows the merge writes (split; 0 otherwise).
    pub merge_rows: u32,
    /// The most partial rows the attention writes (split; 0 otherwise).
    pub partial_rows: u32,
}

/// What a captured rung reserves for one group's pass, and the shape it
/// launches with: every request with `queries` consecutive queries over at
/// most `max_pages` pages each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassCaps {
    pub shape: PassShape,
    /// Work-item words each of the three work arrays reserves.
    pub work_items: u32,
    /// Page words the page list reserves.
    pub pages: u32,
}

impl PassCaps {
    /// The reservation for `rows` requests of `queries` queries each, with
    /// keys up to `max_model_len`, in a group of `group_size` query heads per
    /// KV head and pages of `page_size`, `max_pages` a request.
    pub fn new(
        reduction: Reduction,
        rows: u32,
        queries: u32,
        group_size: u32,
        page_size: u32,
        max_pages: u32,
        max_model_len: u32,
    ) -> Result<PassCaps> {
        let overflow =
            || CudaError::new(format!("attention reservation for {rows} rows overflows"));
        let tile = tile_for(u64::from(queries) * u64::from(group_size));
        let tiles = (u64::from(queries) * u64::from(group_size)).div_ceil(u64::from(tile));
        let rows64 = u64::from(rows);
        let fit = |x: u64| u32::try_from(x).ok().filter(|&v| i32::try_from(v).is_ok());
        let caps = match reduction {
            Reduction::Split => {
                // A request of `queries` queries crosses at most
                // `(queries - 1) / SPLIT_KEYS + 1` chunk boundaries.
                let pieces = u64::from(queries.max(1) - 1).div_ceil(u64::from(SPLIT_KEYS)) + 1;
                let chunks = u64::from(max_model_len.max(1)).div_ceil(u64::from(SPLIT_KEYS));
                let requests = rows64 * pieces;
                let work = requests * tiles * chunks;
                let partials = rows64 * u64::from(queries) * chunks;
                let work32 = fit(work).ok_or_else(overflow)?;
                PassCaps {
                    shape: PassShape {
                        tile,
                        page_size,
                        group_size,
                        split: true,
                        grid: work32.min(PERSISTENT_CTAS),
                        num_requests: fit(requests).ok_or_else(overflow)?,
                        merge_rows: fit(rows64 * u64::from(queries)).ok_or_else(overflow)?,
                        partial_rows: fit(partials).ok_or_else(overflow)?,
                    },
                    work_items: work32,
                    pages: fit(requests * u64::from(max_pages)).ok_or_else(overflow)?,
                }
            }
            Reduction::Anchored { .. } => {
                let work32 = fit(rows64 * tiles).ok_or_else(overflow)?;
                PassCaps {
                    shape: PassShape {
                        tile,
                        page_size,
                        group_size,
                        split: false,
                        grid: work32,
                        num_requests: rows,
                        merge_rows: 0,
                        partial_rows: 0,
                    },
                    work_items: work32,
                    pages: fit(rows64 * u64::from(max_pages)).ok_or_else(overflow)?,
                }
            }
        };
        Ok(caps)
    }

    /// Words of the arrays sized by request count (`q_indptr`, `indptr`,
    /// `o_indptr`: one more than the requests; `last_page_len`: one each).
    pub fn request_words(&self) -> usize {
        self.shape.num_requests as usize + 1
    }

    /// Words of the merge's `indptr` (split; 0 otherwise).
    pub fn merge_words(&self) -> usize {
        if self.shape.split {
            self.shape.merge_rows as usize + 1
        } else {
            0
        }
    }
}

/// A pass's arrays padded to a rung's reservation, as table words: the
/// request arrays extended with empty requests (their indptrs repeat the
/// last entry), the work arrays and pages as long as they are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaddedPass {
    pub q_indptr: Vec<u32>,
    pub indptr: Vec<u32>,
    pub last_page_len: Vec<u32>,
    pub o_indptr: Vec<u32>,
    pub merge_indptr: Vec<u32>,
    pub request_indices: Vec<u32>,
    pub qo_tile_indices: Vec<u32>,
    pub kv_tile_indices: Vec<u32>,
    pub indices: Vec<u32>,
    /// The work-item count the kernel reads.
    pub work_items: u32,
}

impl HostPlan {
    /// This plan as one pass inside `caps`: the same tile, kind and geometry,
    /// one pass, no more requests, work items, pages or partial rows than the
    /// rung reserves, and exactly its merge rows. Refuses anything else.
    pub fn padded(&self, caps: &PassCaps) -> Result<PaddedPass> {
        let bad = |what: String| Err(CudaError::new(format!("attention pass: {what}")));
        let s = &caps.shape;
        let [p] = self.passes.as_slice() else {
            return bad(format!("{} passes for one launch", self.passes.len()));
        };
        let shape = p.shape(self)?;
        if shape.tile != s.tile
            || shape.split != s.split
            || shape.page_size != s.page_size
            || shape.group_size != s.group_size
        {
            return bad(format!("{shape:?} under a reservation for {s:?}"));
        }
        if shape.num_requests > s.num_requests
            || shape.grid > caps.work_items
            || p.indices.len() > caps.pages as usize
            || shape.partial_rows > s.partial_rows
            || shape.merge_rows != s.merge_rows
            || p.q_rows.start != 0
        {
            return bad(format!(
                "{shape:?} with {} pages past a reservation of {caps:?}",
                p.indices.len()
            ));
        }
        let word = |x: i32| u32::try_from(x).expect("HostPlan indices are non-negative");
        let words = |v: &[i32]| v.iter().map(|&x| word(x)).collect::<Vec<u32>>();
        let extend = |v: &[i32], len: usize| {
            let mut w = words(v);
            let last = w.last().copied().unwrap_or(0);
            w.resize(len, last);
            w
        };
        let n = caps.request_words();
        let mut last_page_len = words(&p.last_page_len);
        last_page_len.resize(n - 1, 0);
        Ok(PaddedPass {
            q_indptr: extend(&p.q_indptr, n),
            indptr: extend(&p.indptr, n),
            last_page_len,
            o_indptr: extend(&p.o_indptr, n),
            merge_indptr: words(&p.merge_indptr),
            request_indices: words(&p.request_indices),
            qo_tile_indices: words(&p.qo_tile_indices),
            kv_tile_indices: words(&p.kv_tile_indices),
            indices: words(&p.indices),
            work_items: shape.grid,
        })
    }
}

/// One pass of an uploaded plan: its shape, the first query row its merge
/// writes, and its arrays' word offsets in the plan's buffer (`q_indptr`,
/// `indices`, `indptr`, `last_page_len`, request, query-tile and KV-tile
/// indices, `o_indptr`, the merge's `indptr`, then the work-item count and
/// the KV chunk size).
struct PassAt {
    shape: PassShape,
    out_row: u32,
    at: [usize; 10],
}

/// The per-step work lists for one KV group (shared by all its layers):
/// every pass's arrays in one device buffer, uploaded in one copy.
pub struct AttnPlan {
    shape: PlanShape,
    words: CudaSlice<u32>,
    passes: Vec<PassAt>,
}

impl AttnPlan {
    pub fn page_size(&self) -> u32 {
        self.shape.page_size
    }

    pub fn group_size(&self) -> u32 {
        self.shape.group_size
    }

    /// Query rows the plan's requests cover.
    pub fn q_rows(&self) -> u32 {
        self.shape.q_rows
    }

    /// The largest page id it reads, if any.
    pub fn max_page(&self) -> Option<u32> {
        self.shape.max_page
    }

    pub fn shape(&self) -> PlanShape {
        self.shape
    }

    /// The plan as the launches read it: per pass, its shape and the device
    /// addresses of its arrays.
    pub fn views(&self, gpu: &Gpu) -> Vec<PlanView> {
        let base = dptr(&self.words, gpu.stream());
        let word = |at: usize| base + 4 * at as u64;
        self.passes
            .iter()
            .map(|p| PlanView {
                shape: p.shape,
                out_row: p.out_row,
                q_indptr: word(p.at[0]),
                indices: word(p.at[1]),
                indptr: word(p.at[2]),
                last_page_len: word(p.at[3]),
                request_indices: word(p.at[4]),
                qo_tile_indices: word(p.at[5]),
                kv_tile_indices: word(p.at[6]),
                o_indptr: word(p.at[7]),
                merge_indptr: word(p.at[8]),
                work_items: word(p.at[9]),
                kv_chunk_size: word(p.at[9] + 1),
            })
            .collect()
    }
}

/// A pass as one launch reads it: the launch shape and the device addresses
/// of the arrays the kernels read indirectly. Everything a step varies
/// inside a fixed shape lives behind these addresses, so a launch over a
/// view can be recorded once and replayed with new array contents. Made by
/// [`AttnPlan::views`] (fresh buffers per step) or by the graphs over their
/// step tables.
#[derive(Clone, Copy, Debug)]
pub struct PlanView {
    pub(crate) shape: PassShape,
    /// The first query row the merge writes (split; its rows are the pass's).
    pub(crate) out_row: u32,
    pub(crate) q_indptr: u64,
    pub(crate) indices: u64,
    pub(crate) indptr: u64,
    pub(crate) last_page_len: u64,
    pub(crate) request_indices: u64,
    pub(crate) qo_tile_indices: u64,
    pub(crate) kv_tile_indices: u64,
    pub(crate) o_indptr: u64,
    pub(crate) merge_indptr: u64,
    /// The word holding the work-item count.
    pub(crate) work_items: u64,
    /// The word holding [`SPLIT_KEYS`] (FlashInfer reads it on every launch).
    pub(crate) kv_chunk_size: u64,
}

/// One layer's KV and attention settings.
#[derive(Clone, Copy, Debug)]
pub struct AttnLayer {
    /// Device address of this layer's K inside page 0.
    pub k_base: u64,
    /// Device address of this layer's V inside page 0.
    pub v_base: u64,
    /// Elements from one page's K to the next: a whole block (every layer)
    /// for a blocked group, one position's K for a planar one.
    pub k_page_stride: u32,
    /// Elements from one page's V to the next.
    pub v_page_stride: u32,
    pub num_kv_heads: u32,
    pub page_size: u32,
    /// Keys visible to a query besides itself (`window - 1`), or -1.
    pub window_left: i32,
    /// `f32[num_qo_heads]` sink logits (`-inf` for none).
    pub sink: u64,
}

/// Where a split pass's partial states go between the attention and the
/// merge: `rows` partial rows of every query head's BF16 output and f32
/// log-sum-exp.
#[derive(Clone, Copy, Debug)]
pub struct Partials {
    pub v: u64,
    pub s: u64,
    pub rows: u32,
    /// Query heads a partial row holds.
    pub heads: u32,
}

/// The attention kernels: per reduction, one instance per query tile, and
/// the merge.
pub struct Attention {
    _module: KernelModule,
    /// Split (global) instances at query tiles 16, 64, 128.
    split: [Kernel; 3],
    /// Anchored (sliding) instances at query tiles 16, 64, 128.
    anchored: [Kernel; 3],
    merge: Kernel,
}

/// The merge's launch-contract record.
const MERGE_META: &str = "eidola_fa2_merge_states_bf16_d128_meta";

impl Attention {
    pub fn from_module(module: KernelModule) -> Result<Attention> {
        module.expect_image("flashinfer_fa2_sink_paged")?;
        let get = |s: &str| -> Result<Kernel> {
            let k = module.kernel(s)?;
            if k.meta().params_bytes as usize != std::mem::size_of::<AttnParams>() {
                return Err(CudaError::new(format!(
                    "{s}: its parameters are {} bytes in the image, {} here",
                    k.meta().params_bytes,
                    std::mem::size_of::<AttnParams>()
                )));
            }
            Ok(k)
        };
        let tiles = |prefix: &str| -> Result<[Kernel; 3]> {
            Ok([
                get(&format!("{prefix}_q16"))?,
                get(&format!("{prefix}_q64"))?,
                get(&format!("{prefix}_q128"))?,
            ])
        };
        let merge_symbol = module
            .cubin()
            .entry_for_meta(MERGE_META)
            .ok_or_else(|| CudaError::new("no attention merge entry"))?
            .symbol
            .clone();
        let merge = module.kernel(&merge_symbol)?;
        if merge.meta().params_bytes != 0 {
            return Err(CudaError::new("the attention merge takes an argument list"));
        }
        Ok(Attention {
            split: tiles("eidola_fa2_sink_paged_bf16")?,
            anchored: tiles("eidola_fa2_sink_paged_anchored_bf16")?,
            merge,
            _module: module,
        })
    }

    /// Upload a host plan (see [`Attention::plan`]), with the geometry it
    /// was checked for.
    pub fn upload(&self, gpu: &Gpu, host: HostPlan) -> Result<AttnPlan> {
        let mut words: Vec<u32> = Vec::new();
        let mut put = |v: &[i32]| -> usize {
            let at = words.len();
            words.extend(
                v.iter()
                    .map(|&x| u32::try_from(x).expect("HostPlan indices are non-negative")),
            );
            at
        };
        let mut passes = Vec::with_capacity(host.passes.len());
        for p in &host.passes {
            let shape = p.shape(&host)?;
            let at = [
                put(&p.q_indptr),
                put(&p.indices),
                put(&p.indptr),
                put(&p.last_page_len),
                put(&p.request_indices),
                put(&p.qo_tile_indices),
                put(&p.kv_tile_indices),
                put(&p.o_indptr),
                put(&p.merge_indptr),
                put(&[
                    i32::try_from(shape.grid).expect("checked by HostPlan::new: fits i32"),
                    i32::try_from(SPLIT_KEYS).expect("a small constant"),
                ]),
            ];
            passes.push(PassAt {
                shape,
                out_row: p.q_rows.start,
                at,
            });
        }
        Ok(AttnPlan {
            shape: host.shape(),
            words: gpu.stream().clone_htod(&words)?,
            passes,
        })
    }

    /// Plan one step's attention for a group (see [`HostPlan::new`]).
    pub fn plan(
        &self,
        gpu: &Gpu,
        requests: &[AttnRequest],
        group_size: u32,
        page_size: u32,
        reduction: Reduction,
        partial_rows: u32,
    ) -> Result<AttnPlan> {
        let host = HostPlan::new(requests, group_size, page_size, reduction, partial_rows)?;
        self.upload(gpu, host)
    }

    /// Attention for one layer over every pass of `plan`: `q` is `[rows,
    /// num_qo_heads, 192]`, `o` is `[rows, num_qo_heads, 128]`, both BF16.
    ///
    /// # Safety
    ///
    /// `q`, `o`, the layer's pool and its sinks must cover what the plan's
    /// requests address.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run(
        &self,
        gpu: &Gpu,
        plan: &AttnPlan,
        layer: &AttnLayer,
        num_qo_heads: u32,
        q: u64,
        o: u64,
        partials: &Partials,
    ) -> Result<()> {
        for view in plan.views(gpu) {
            // SAFETY: the caller's contract, over the plan's own buffers.
            unsafe { self.run_view(gpu, &view, layer, num_qo_heads, q, o, partials)? };
        }
        Ok(())
    }

    /// One pass ([`Attention::run`]) over a [`PlanView`]: the attention, then
    /// for a split the merge of every row's partials into `o`.
    ///
    /// # Safety
    ///
    /// As [`Attention::run`], and the view's addresses must hold a pass
    /// [`HostPlan::new`] made for the view's shape (padded to it, for a
    /// captured rung) when the launches run.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn run_view(
        &self,
        gpu: &Gpu,
        view: &PlanView,
        layer: &AttnLayer,
        num_qo_heads: u32,
        q: u64,
        o: u64,
        partials: &Partials,
    ) -> Result<()> {
        let shape = &view.shape;
        if shape.grid == 0 {
            return Ok(());
        }
        if layer.num_kv_heads == 0
            || !num_qo_heads.is_multiple_of(layer.num_kv_heads)
            || layer.page_size == 0
            || layer.k_page_stride == 0
            || layer.v_page_stride == 0
        {
            return Err(CudaError::new(
                "attention: query heads must be whole GQA groups; pages non-empty",
            ));
        }
        if num_qo_heads / layer.num_kv_heads != shape.group_size {
            return Err(CudaError::new(format!(
                "attention: a plan for GQA groups of {} run on groups of {}",
                shape.group_size,
                num_qo_heads / layer.num_kv_heads
            )));
        }
        if layer.page_size != shape.page_size {
            return Err(CudaError::new(format!(
                "attention: a plan for {}-position pages run on {}",
                shape.page_size, layer.page_size
            )));
        }
        // A split is global attention: no window to anchor; an anchored
        // plan never splits.
        if shape.split != (layer.window_left < 0) {
            return Err(CudaError::new(format!(
                "attention: a {} plan run on a layer with window_left {}",
                if shape.split { "split" } else { "anchored" },
                layer.window_left
            )));
        }
        if shape.split && (shape.partial_rows > partials.rows || partials.heads != num_qo_heads) {
            return Err(CudaError::new(format!(
                "attention: {} partial rows of {num_qo_heads} heads past the scratch's {} of {}",
                shape.partial_rows, partials.rows, partials.heads
            )));
        }
        let s = gpu.stream();
        let family = if shape.split {
            &self.split
        } else {
            &self.anchored
        };
        let kernel = match shape.tile {
            16 => &family[0],
            64 => &family[1],
            128 => &family[2],
            t => return Err(CudaError::new(format!("attention: no {t}-row query tile"))),
        };
        let h = layer.num_kv_heads;
        let (out, lse) = if shape.split {
            (partials.v, partials.s)
        } else {
            (o, 0)
        };
        let mut params = AttnParams {
            paged: PagedParams {
                q,
                paged_kv: PagedKv {
                    page_size: UintFastdiv::new(layer.page_size),
                    num_heads: h,
                    head_dim: HEAD_DIM_QK,
                    batch_size: shape.num_requests,
                    stride_page: layer.k_page_stride,
                    stride_n: h * HEAD_DIM_QK,
                    stride_h: HEAD_DIM_QK,
                    v_stride_page: layer.v_page_stride,
                    v_stride_n: h * HEAD_DIM_VO,
                    v_stride_h: HEAD_DIM_VO,
                    k_data: layer.k_base,
                    v_data: layer.v_base,
                    indices: view.indices,
                    indptr: view.indptr,
                    last_page_len: view.last_page_len,
                    rope_pos_offset: 0,
                },
                q_indptr: view.q_indptr,
                o: out,
                lse,
                group_size: UintFastdiv::new(num_qo_heads / h),
                sink: layer.sink,
                sm_scale: 1.0 / (HEAD_DIM_QK as f64).sqrt(),
                num_qo_heads,
                q_stride_n: narrow(num_qo_heads as u64 * HEAD_DIM_QK as u64, "query row stride")?,
                q_stride_h: narrow(HEAD_DIM_QK, "query head stride")?,
                window_left: layer.window_left,
                request_indices: view.request_indices,
                qo_tile_indices: view.qo_tile_indices,
                kv_tile_indices: view.kv_tile_indices,
                o_indptr: view.o_indptr,
                kv_chunk_size_ptr: view.kv_chunk_size,
                padded_batch_size: shape.grid,
                partition_kv: shape.split,
                ..PagedParams::default()
            },
            work_items: view.work_items,
        };
        let mut args = [&mut params as *mut AttnParams as *mut c_void];
        // SAFETY: the single by-value argument is the kernel's
        // EidolaAttnParams (size checked at load); addresses are the
        // caller's.
        unsafe { kernel.launch(s, [shape.grid, 1, h], &mut args)? };
        if !shape.split {
            return Ok(());
        }
        let row_bytes = u64::from(num_qo_heads) * u64::from(HEAD_DIM_VO) * 2;
        let pairs = u64::from(shape.merge_rows) * u64::from(num_qo_heads);
        let blocks: u32 = narrow(pairs.min(u64::from(MERGE_BLOCKS)), "merge blocks")?;
        if blocks == 0 {
            return Ok(());
        }
        // SAFETY: the merge reads `merge_indptr[merge_rows]` partial rows of
        // the scratch the attention launch wrote (within `partials.rows`,
        // checked above) and writes `merge_rows` rows of every query head of
        // `o` from the pass's first row.
        unsafe {
            crate::launch!(
                gpu,
                self.merge,
                [blocks, 1, 1],
                partials.v,
                partials.s,
                view.merge_indptr,
                o + u64::from(view.out_row) * row_bytes,
                0u64,
                shape.merge_rows,
                0u64,
                num_qo_heads
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(q_start: u32, qo_len: u32, pages: usize, kv_len: u32) -> AttnRequest {
        AttnRequest {
            q_start,
            qo_len,
            pages: (1..=u32::try_from(pages).unwrap()).collect(),
            kv_start: 0,
            kv_len,
        }
    }

    const ANCHORED: Reduction = Reduction::Anchored { origin: 0 };

    fn u(x: i32) -> u32 {
        u32::try_from(x).unwrap()
    }

    fn us(x: i32) -> usize {
        usize::try_from(x).unwrap()
    }

    /// A request's pages cover exactly its KV, the last one with 1..=page_size
    /// positions; every index fits the kernel's `i32` arrays.
    #[test]
    fn plans_are_checked_on_the_host() {
        let p = HostPlan::new(&[req(0, 1, 2, 17), req(1, 3, 2, 32)], 16, 16, ANCHORED, 0).unwrap();
        let pass = &p.passes[0];
        assert_eq!(pass.last_page_len, vec![1, 16]);
        assert_eq!(pass.q_indptr, vec![0, 1, 4]);
        assert_eq!(pass.indptr, vec![0, 2, 4]);
        assert_eq!(pass.o_indptr, pass.q_indptr);
        assert!(pass.merge_indptr.is_empty());
        assert_eq!(p.tile, 64);
        // The geometry it was checked for travels with it.
        assert_eq!((p.group_size, p.page_size), (16, 16));
        let p = HostPlan::new(&[req(0, 1, 1, 30)], 8, 32, ANCHORED, 0).unwrap();
        assert_eq!(
            (p.group_size, p.page_size, p.passes[0].last_page_len[0]),
            (8, 32, 30)
        );
        let refused = |rs: &[AttnRequest], group: u32, page: u32| {
            for reduction in [ANCHORED, Reduction::Split] {
                HostPlan::new(rs, group, page, reduction, 1 << 20).expect_err(&format!("{rs:?}"));
            }
        };
        refused(&[req(0, 1, 2, 1)], 16, 16);
        refused(&[req(0, 1, 2, 16)], 16, 16);
        refused(&[req(0, 1, 2, 33)], 16, 16);
        refused(&[req(0, 1, 0, 1)], 16, 16);
        refused(&[req(0, 0, 1, 1)], 16, 16);
        refused(&[req(0, 5, 1, 4)], 16, 16);
        refused(&[req(1, 1, 1, 1)], 16, 16);
        refused(&[req(0, 1, 1, 1), req(2, 1, 1, 1)], 16, 16);
        refused(&[req(0, 1, 1, 1)], 0, 16);
        refused(&[req(0, 1, 1, 1)], 16, 0);
        refused(
            &[AttnRequest {
                pages: vec![1 << 31],
                ..req(0, 1, 1, 1)
            }],
            16,
            16,
        );
        refused(
            &[req(0, 1, 1, 1), req(1, i32::MAX as u32, 1, u32::MAX)],
            16,
            u32::MAX,
        );
        // A page size past i32 leaves a last page whose length cannot be held.
        refused(&[req(0, 1, 1, 1 << 31)], 16, u32::MAX);
        // Pages off a page boundary.
        refused(
            &[AttnRequest {
                kv_start: 8,
                ..req(0, 1, 1, 1)
            }],
            16,
            16,
        );
    }

    /// Anchored pages start at the origin plus a multiple of the KV tile, a
    /// split's at position 0; anything else is refused.
    #[test]
    fn plans_hold_the_anchor() {
        let at = |kv_start: u32| AttnRequest {
            kv_start,
            ..req(0, 1, 1, 1)
        };
        let ok = |r: Reduction, start: u32, page: u32| HostPlan::new(&[at(start)], 8, page, r, 64);
        assert!(ok(ANCHORED, 0, 16).is_ok());
        assert!(ok(ANCHORED, 64, 16).is_ok());
        assert!(ok(ANCHORED, 128, 16).is_ok());
        assert!(ok(ANCHORED, 48, 16).is_err());
        assert!(ok(ANCHORED, 16, 16).is_err());
        let drafter = Reduction::Anchored { origin: 3 };
        assert!(ok(drafter, 3, 1).is_ok());
        assert!(ok(drafter, 67, 1).is_ok());
        assert!(ok(drafter, 0, 1).is_err());
        assert!(ok(drafter, 64, 1).is_err());
        assert!(ok(drafter, 66, 1).is_err());
        assert!(ok(Reduction::Split, 0, 16).is_ok());
        assert!(HostPlan::new(&[at(64)], 8, 16, Reduction::Split, 1).is_err());
    }

    /// The anchor is the largest one at or below the first visible position,
    /// a page boundary, and the origin's phase modulo the KV tile; the page
    /// bound holds every request a window's queries make.
    #[test]
    fn anchors_sit_at_absolute_multiples() {
        for page in [1u32, 2, 4, 8, 16, 32, 48, 64, 128] {
            let stride = anchor_stride(page);
            assert!(
                stride.is_multiple_of(u64::from(KV_TILE)) && stride.is_multiple_of(u64::from(page))
            );
            for origin in [0u32, 1, 2, 3] {
                if !origin.is_multiple_of(page) {
                    continue;
                }
                let r = Reduction::Anchored { origin };
                for first in 0..700u32 {
                    let a = r.kv_start(first, page);
                    let f = first.max(origin);
                    assert!(a <= f && a >= origin, "{page} {origin} {first}: {a}");
                    assert_eq!((a - origin) % KV_TILE, 0);
                    assert_eq!(a % page, 0);
                    assert!(u64::from(f - a) < stride);
                }
                for window in [1u32, 2, 64, 127, 128, 129, 200] {
                    for queries in 1..=5u32 {
                        let bound = r.max_pages(Some(window), queries, page, u32::MAX);
                        for c in 0..900u32 {
                            let first = (c + 1).saturating_sub(window);
                            let last = c + queries - 1;
                            if last < origin {
                                // No KV below the origin, so no query there.
                                continue;
                            }
                            let a = r.kv_start(first, page);
                            let pages = last / page - a / page + 1;
                            assert!(
                                pages <= bound,
                                "page {page} origin {origin} window {window} queries {queries} \
                                 c {c}: {pages} pages past {bound}"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(Reduction::Split.kv_start(500, 16), 0);
        assert_eq!(Reduction::Split.max_pages(None, 4, 16, 77), 77);
    }

    /// Emulates the merge's reads over a split plan: each query row merges
    /// exactly chunks `0 ..= p / SPLIT_KEYS`, in chunk order, each partial
    /// written by exactly one work item of that chunk (so the sink, which the
    /// kernel adds in chunk 0 only, enters each row exactly once); every work
    /// item's rows see at least one key of its chunk; partial rows never
    /// overlap; and passes cover the rows in order within the budget.
    fn check_split(requests: &[AttnRequest], group: u32, page: u32, budget: u32) -> HostPlan {
        let p = HostPlan::new(requests, group, page, Reduction::Split, budget).unwrap();
        // The absolute position of every query row.
        let mut pos_of = Vec::new();
        for r in requests {
            for q in 0..r.qo_len {
                pos_of.push(r.kv_start + r.kv_len - r.qo_len + q);
            }
        }
        let mut next_row = 0u32;
        for pass in &p.passes {
            assert_eq!(pass.q_rows.start, next_row);
            next_row = pass.q_rows.end;
            let partials = pass.partial_rows(true);
            assert!(partials <= budget);
            // Which (query row, chunk) wrote each partial row.
            let mut writer: Vec<Option<(u32, u32)>> = vec![None; partials as usize];
            for w in 0..pass.work_items() {
                let i = us(pass.request_indices[w]);
                let (t, c) = (u(pass.qo_tile_indices[w]), u(pass.kv_tile_indices[w]));
                let q0 = u(pass.q_indptr[i]);
                let qo = u(pass.q_indptr[i + 1]) - q0;
                let pages = u(pass.indptr[i + 1] - pass.indptr[i]);
                let kv = (pages - 1) * page + u(pass.last_page_len[i]);
                let chunks = kv.div_ceil(SPLIT_KEYS);
                assert!(c < chunks);
                // The tile's rows (packed rows / group size).
                let lo = t * p.tile / group;
                let hi = ((t + 1) * p.tile).div_ceil(group).min(qo);
                for q in lo..hi {
                    let pos = kv - qo + q;
                    assert_eq!(pos, pos_of[(q0 + q) as usize], "the piece's positions");
                    // The row sees a key of the chunk.
                    assert!(pos >= c * SPLIT_KEYS, "row {pos} in chunk {c}");
                    let row = u(pass.o_indptr[i]) + q * chunks + c;
                    let slot = &mut writer[row as usize];
                    assert!(slot.is_none() || *slot == Some((q0 + q, c)));
                    *slot = Some((q0 + q, c));
                }
            }
            let rows = pass.q_rows.end - pass.q_rows.start;
            assert_eq!(pass.merge_indptr.len(), (rows + 1) as usize);
            for j in 0..rows {
                let row = pass.q_rows.start + j;
                let pos = pos_of[row as usize];
                let (a, b) = (
                    pass.merge_indptr[j as usize],
                    pass.merge_indptr[j as usize + 1],
                );
                let merged: Vec<(u32, u32)> =
                    (a..b).map(|x| writer[us(x)].expect("written")).collect();
                let want: Vec<(u32, u32)> = (0..=pos / SPLIT_KEYS).map(|c| (row, c)).collect();
                assert_eq!(merged, want, "row {row} at {pos}");
                assert_eq!(merged.iter().filter(|&&(_, c)| c == 0).count(), 1);
            }
        }
        assert_eq!(next_row as usize, pos_of.len());
        p
    }

    fn global(q_start: u32, qo_len: u32, end: u32, page: u32) -> AttnRequest {
        AttnRequest {
            q_start,
            qo_len,
            pages: (0..end.div_ceil(page)).map(|i| i + 1).collect(),
            kv_start: 0,
            kv_len: end,
        }
    }

    /// Every row merges exactly its own chunks whatever request it is in: a
    /// decode row, a verify run across a boundary, chunks of every size
    /// across several, and plans in several passes.
    #[test]
    fn split_rows_merge_exactly_their_chunks() {
        for page in [1u32, 16, 64] {
            for group in [8u32, 16] {
                for end in [1u32, 2, 1023, 1024, 1025, 2047, 2048, 2049, 4101, 9000] {
                    for qo in [1u32, 2, 4, 300, 1024, 2048] {
                        if qo > end {
                            continue;
                        }
                        let rs = [global(0, qo, end, page), global(qo, 1, 1500, page)];
                        check_split(&rs, group, page, 1 << 20);
                        // Budgets that force passes, down to the most chunks a
                        // row has; one less holds no row of those.
                        let chunks = end.div_ceil(SPLIT_KEYS).max(2);
                        for budget in [chunks, chunks + 1, 3 * chunks, 4096] {
                            check_split(&rs, group, page, budget);
                        }
                        assert!(
                            HostPlan::new(&rs, group, page, Reduction::Split, chunks - 1).is_err()
                        );
                    }
                }
            }
        }
    }

    /// A request is cut at multiples of the split of its query positions:
    /// pieces never straddle one, and lose no query.
    #[test]
    fn split_requests_are_cut_at_chunk_boundaries() {
        let p = check_split(&[global(0, 4, 1026, 16)], 16, 16, 1 << 20);
        let pass = &p.passes[0];
        // Queries at 1022, 1023 | 1024, 1025.
        assert_eq!(pass.q_indptr, vec![0, 2, 4]);
        assert_eq!(pass.last_page_len, vec![16, 2]);
        assert_eq!(pass.indptr, vec![0, 64, 129]);
        assert_eq!(pass.merge_indptr, vec![0, 1, 2, 4, 6]);
        assert_eq!(pass.o_indptr, vec![0, 2, 6]);
        // The tile is the step's (four rows of 16 heads), not the pieces'.
        assert_eq!(p.tile, 64);
        // One decode row: one piece, its chunks.
        let p = check_split(&[global(0, 1, 4101, 16)], 16, 16, 1 << 20);
        assert_eq!(p.passes[0].kv_tile_indices, vec![0, 1, 2, 3, 4]);
        assert_eq!(p.passes[0].merge_indptr, vec![0, 5]);
        // Too little scratch for one row's chunks.
        assert!(HostPlan::new(&[global(0, 1, 4101, 16)], 16, 16, Reduction::Split, 4).is_err());
    }

    /// The work list's shape (tile, items per piece and chunk) is
    /// FlashInfer's split layout.
    #[test]
    fn split_work_items_are_flashinfers_layout() {
        let p = check_split(&[global(0, 300, 2100, 16)], 8, 16, 1 << 20);
        let pass = &p.passes[0];
        assert_eq!(p.tile, 128);
        // Queries 1800..2100: 1800..2048 (248 rows, 2 chunks), 2048..2100
        // (52 rows, 3 chunks).
        assert_eq!(pass.q_indptr, vec![0, 248, 300]);
        let tiles = |q: u32| (q * 8).div_ceil(128);
        assert_eq!(pass.work_items(), (tiles(248) * 2 + tiles(52) * 3) as usize);
        assert_eq!(&pass.kv_tile_indices[..4], &[0, 1, 0, 1]);
        assert_eq!(&pass.qo_tile_indices[..4], &[0, 0, 1, 1]);
        assert_eq!(*pass.o_indptr.last().unwrap(), 248 * 2 + 52 * 3);
    }

    /// A rung's reservation holds every step its rows can make, the pass
    /// padded to it keeps the real arrays as a prefix, and a step past it is
    /// refused.
    #[test]
    fn padded_passes_fit_their_reservation() {
        let max_len = 5000u32;
        for queries in [1u32, 2, 4] {
            let rows = 3u32;
            let caps = PassCaps::new(
                Reduction::Split,
                rows,
                queries,
                16,
                16,
                max_len.div_ceil(16),
                max_len,
            )
            .unwrap();
            assert_eq!(caps.shape.merge_rows, rows * queries);
            for ends in [
                [1u32, 2, 3],
                [1024, 1025, 1026],
                [4997, 1023 + queries, 5000],
            ] {
                let mut rs = Vec::new();
                for (i, &end) in ends.iter().enumerate() {
                    let end = end.max(queries);
                    rs.push(global(
                        u32::try_from(i).unwrap() * queries,
                        queries,
                        end,
                        16,
                    ));
                }
                let p = check_split(&rs, 16, 16, caps.shape.partial_rows);
                let padded = p.padded(&caps).unwrap();
                assert_eq!(padded.q_indptr.len(), caps.request_words());
                assert_eq!(padded.merge_indptr.len(), caps.merge_words());
                assert!(padded.work_items <= caps.work_items);
                assert!(padded.indices.len() <= caps.pages as usize);
                let pass = &p.passes[0];
                let real = pass.num_requests();
                assert_eq!(
                    &padded.q_indptr[..=real],
                    &pass.q_indptr.iter().map(|&x| u(x)).collect::<Vec<_>>()[..]
                );
                assert!(padded.q_indptr[real..].iter().all(|&x| x == rows * queries));
            }
        }
        // Anchored: one request a row, exactly its work items.
        let caps = PassCaps::new(ANCHORED, 2, 1, 8, 16, 12, 4096).unwrap();
        assert_eq!(
            (caps.shape.grid, caps.shape.num_requests, caps.shape.tile),
            (2, 2, 16)
        );
        let rs = [req(0, 1, 2, 20), req(1, 1, 1, 3)];
        let p = HostPlan::new(&rs, 8, 16, ANCHORED, 0).unwrap();
        assert_eq!(p.padded(&caps).unwrap().work_items, 2);
        // More rows than reserved.
        let rs3 = [req(0, 1, 1, 1), req(1, 1, 1, 1), req(2, 1, 1, 1)];
        let p = HostPlan::new(&rs3, 8, 16, ANCHORED, 0).unwrap();
        assert!(p.padded(&caps).is_err());
        // Another tile.
        let rs = [req(0, 4, 1, 4), req(4, 1, 1, 1)];
        let p = HostPlan::new(&rs, 8, 16, ANCHORED, 0).unwrap();
        assert!(p.padded(&caps).is_err());
    }
}
