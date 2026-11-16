//! Sparse attention (`ds_attention.cu`): `eidola_ds_attn` over each row's slot
//! list, and `eidola_ds_window_slots`, which writes the window part of the
//! lists. Bit-identical to `eidola_engine_model::deepseek::attention::sparse_attend`
//! (`eidola-engine-kernels/AGENTS.md` → *DeepSeek-V4.1 attention, index and selection*).
//!
//! The launch geometry is a function of the row count and the configuration:
//! a decode step's launches are the same whatever the rows hold.

use eidola_engine_model::deepseek::drafter::ring_order;

use super::abi::{
    ATTN_FIXED, ATTN_PACKED, AttnParams, KV_FP4_E8M0, KvPlane, ORDER_POSITION, ORDER_RING,
    SLOT_EMPTY, WindowSlotsParams,
};
use crate::module::{Kernel, KernelDir, KernelModule};
use crate::{CudaError, Gpu, Result, launch};

/// Threads per block of `eidola_ds_attn`.
pub const ATTN_THREADS: u32 = 512;
/// The widest head the kernel's shared memory holds.
pub const MAX_HEAD_DIM: u32 = 512;
/// The most heads one block computes.
pub const MAX_HEADS_PER_BLOCK: u32 = 8;
/// Threads per block of `eidola_ds_window_slots`: a slot each.
pub const SLOT_THREADS: u32 = 128;
/// The driver's grid limit in y.
const MAX_GRID_Y: u32 = 65_535;
/// Blocks of `eidola_ds_attn` a 148-SM part keeps resident (two an SM, by
/// shared memory).
const RESIDENT_BLOCKS: u32 = 296;

/// Heads per block for a launch of `rows` rows: one while every block of the
/// launch is resident at once (a decode row's heads run side by side), then
/// more, up to [`MAX_HEADS_PER_BLOCK`], so that a block's entries are
/// dequantised once for several heads. No row's bits depend on it.
pub fn heads_per_block(rows: u32, heads: u32) -> u32 {
    let mut group = 1;
    while group < MAX_HEADS_PER_BLOCK
        && u64::from(rows) * u64::from(heads.div_ceil(group)) > u64::from(RESIDENT_BLOCKS)
    {
        group *= 2;
    }
    group
}

/// The grid of `eidola_ds_attn` with `group` heads per block.
pub fn attn_grid(p: &AttnParams, group: u32) -> Result<[u32; 3]> {
    if group == 0 || group > MAX_HEADS_PER_BLOCK {
        return Err(CudaError::new(format!("ds_attn: {group} heads per block")));
    }
    Ok([p.heads.div_ceil(group), p.rows, 1])
}

fn check_plane(name: &str, plane: &KvPlane, slots: u32, width: u32) -> Result<()> {
    if slots == 0 {
        return Ok(());
    }
    if plane.base == 0 || plane.entries_per_block == 0 || plane.format > KV_FP4_E8M0 {
        return Err(CudaError::new(format!("ds_attn: plane {name} is not set")));
    }
    if plane.width != width {
        return Err(CudaError::new(format!(
            "ds_attn: plane {name} holds entries of {}, the heads are {width} wide",
            plane.width
        )));
    }
    Ok(())
}

/// What `eidola_ds_attn` needs of its parameters; the slot words and the
/// planes' contents are the caller's.
pub fn check_attn(p: &AttnParams) -> Result<()> {
    if p.heads == 0
        || p.head_dim == 0
        || !p.head_dim.is_multiple_of(32)
        || p.head_dim > MAX_HEAD_DIM
    {
        return Err(CudaError::new(format!(
            "ds_attn: {} heads of {} (a head is a multiple of 32, at most {MAX_HEAD_DIM})",
            p.heads, p.head_dim
        )));
    }
    if p.rows > MAX_GRID_Y {
        return Err(CudaError::new(format!("ds_attn: {} rows", p.rows)));
    }
    if p.a_slots
        .checked_add(p.b_slots)
        .is_none_or(|n| n > p.slot_stride)
    {
        return Err(CudaError::new(format!(
            "ds_attn: {} + {} slots in rows of {}",
            p.a_slots, p.b_slots, p.slot_stride
        )));
    }
    if p.layout != ATTN_FIXED && p.layout != ATTN_PACKED {
        return Err(CudaError::new(format!("ds_attn: layout {}", p.layout)));
    }
    if p.q == 0 || p.slots == 0 || p.sink == 0 || p.out == 0 {
        return Err(CudaError::new("ds_attn: a null array"));
    }
    if !p.q.is_multiple_of(2)
        || !p.out.is_multiple_of(2)
        || !p.slots.is_multiple_of(4)
        || !p.sink.is_multiple_of(4)
    {
        return Err(CudaError::new("ds_attn: a misaligned array"));
    }
    check_plane("a", &p.a, p.a_slots, p.head_dim)?;
    check_plane("b", &p.b, p.b_slots, p.head_dim)
}

/// The grid of `eidola_ds_window_slots`.
pub fn window_slots_grid(p: &WindowSlotsParams) -> [u32; 3] {
    [p.window.div_ceil(SLOT_THREADS), p.rows, 1]
}

/// What `eidola_ds_window_slots` needs of its parameters; positions and
/// table words are device data (a table word that maps nothing raises
/// `STATUS_BAD_INDEX`).
pub fn check_window_slots(p: &WindowSlotsParams) -> Result<()> {
    if p.window == 0 || p.window > p.slot_stride || p.block_positions == 0 || p.table_stride == 0 {
        return Err(CudaError::new(format!(
            "ds_window_slots: a window of {} in rows of {}, {} positions a block, tables of {}",
            p.window, p.slot_stride, p.block_positions, p.table_stride
        )));
    }
    if p.rows > MAX_GRID_Y {
        return Err(CudaError::new(format!("ds_window_slots: {} rows", p.rows)));
    }
    if p.order != ORDER_POSITION && p.order != ORDER_RING {
        return Err(CudaError::new(format!(
            "ds_window_slots: order {}",
            p.order
        )));
    }
    let arrays = [p.positions, p.tables, p.table_row, p.slots, p.status];
    if arrays.contains(&0) || (p.order == ORDER_POSITION && p.first == 0) {
        return Err(CudaError::new("ds_window_slots: a null array"));
    }
    if arrays
        .iter()
        .chain([&p.first])
        .any(|a| !a.is_multiple_of(4))
    {
        return Err(CudaError::new("ds_window_slots: a misaligned array"));
    }
    Ok(())
}

/// The positions a window lists, slot by slot: the host form of
/// `eidola_ds_window_slots` before the block table. In position order slot
/// `i` is position `pos + i + 1 - window` (the reference's canonical layout:
/// right-aligned, the row's own position last), empty before the sequence
/// start or below `first`; in ring order the slots are
/// `drafter::ring_order(pos, window)`, then empty.
pub fn window_positions(order: u32, window: usize, pos: usize, first: usize) -> Vec<Option<usize>> {
    if order == ORDER_RING {
        let mut ring: Vec<Option<usize>> = ring_order(pos, window).into_iter().map(Some).collect();
        ring.resize(window, None);
        return ring;
    }
    (0..window)
        .map(|i| (pos + i + 1).checked_sub(window).filter(|x| *x >= first))
        .collect()
}

/// The slot word of logical index `x` (a position, or a bank entry) through
/// one block table, as the kernels compute it: `None` when the table maps
/// nothing there (the kernels raise `STATUS_BAD_INDEX` and write
/// [`SLOT_EMPTY`]).
pub fn table_entry(table: &[i32], x: usize, per_block: usize) -> Option<u32> {
    let block = usize::try_from(table[x / per_block]).ok()?;
    u32::try_from(block * per_block + x % per_block).ok()
}

/// The window part of one row's slot list: the host form of
/// `eidola_ds_window_slots`. The second value is whether a listed position
/// had no block.
pub fn window_slots(
    order: u32,
    window: usize,
    pos: usize,
    first: usize,
    table: &[i32],
    block_positions: usize,
) -> (Vec<u32>, bool) {
    let mut unmapped = false;
    let slots = window_positions(order, window, pos, first)
        .into_iter()
        .map(
            |x| match x.map(|x| table_entry(table, x, block_positions)) {
                None => SLOT_EMPTY,
                Some(Some(word)) => word,
                Some(None) => {
                    unmapped = true;
                    SLOT_EMPTY
                }
            },
        )
        .collect();
    (slots, unmapped)
}

/// The loaded attention kernels.
pub struct DsAttention {
    _module: KernelModule,
    attn: Kernel,
    window_slots: Kernel,
}

impl DsAttention {
    pub const IMAGE: &'static str = "ds_attention";

    pub fn load(gpu: &Gpu, dir: &KernelDir) -> Result<DsAttention> {
        DsAttention::from_module(KernelModule::load(gpu, dir, Self::IMAGE)?)
    }

    pub fn from_module(m: KernelModule) -> Result<DsAttention> {
        let attn = m.bound_kernel(Self::IMAGE, "eidola_ds_attn")?;
        let window_slots = m.bound_kernel(Self::IMAGE, "eidola_ds_window_slots")?;
        let contract = |k: &Kernel, threads: u32, params: usize| {
            k.meta().block == [threads, 1, 1] && k.meta().params_bytes as usize == params
        };
        if !contract(&attn, ATTN_THREADS, std::mem::size_of::<AttnParams>())
            || !contract(
                &window_slots,
                SLOT_THREADS,
                std::mem::size_of::<WindowSlotsParams>(),
            )
        {
            return Err(CudaError::new("ds_attention: launch contract"));
        }
        Ok(DsAttention {
            _module: m,
            attn,
            window_slots,
        })
    }

    /// Dynamic shared memory of `eidola_ds_attn`, from its launch contract.
    pub fn attn_smem_bytes(&self) -> u32 {
        self.attn.meta().dynamic_smem_bytes
    }

    /// Sparse attention of rows `0 .. p.rows`.
    ///
    /// # Safety
    ///
    /// Every array `p` names must be a live device allocation of this context
    /// holding what [`AttnParams`] says, until the launch completes; every
    /// slot word must be [`SLOT_EMPTY`] or an entry inside its plane.
    pub unsafe fn attend(&self, gpu: &Gpu, p: &AttnParams) -> Result<()> {
        // SAFETY: the caller's contract.
        unsafe { self.attend_grouped(gpu, p, heads_per_block(p.rows, p.heads)) }
    }

    /// [`DsAttention::attend`] with `group` heads per block, for the tests
    /// and the bench: the output is the same for every `group`.
    ///
    /// # Safety
    ///
    /// As [`DsAttention::attend`].
    pub unsafe fn attend_grouped(&self, gpu: &Gpu, p: &AttnParams, group: u32) -> Result<()> {
        check_attn(p)?;
        let grid = attn_grid(p, group)?;
        if p.rows == 0 {
            return Ok(());
        }
        // SAFETY: `eidola_ds_attn(DsAttnParams)`, checked above; the arrays are the caller's.
        unsafe { launch!(gpu, self.attn, grid, *p) }
    }

    /// Writes slots `0 .. p.window` of every row's list.
    ///
    /// # Safety
    ///
    /// Every array `p` names must be a live device allocation of this context
    /// holding what [`WindowSlotsParams`] says, until the launch completes;
    /// every table row must hold the block of each position its row lists.
    pub unsafe fn window_slots(&self, gpu: &Gpu, p: &WindowSlotsParams) -> Result<()> {
        check_window_slots(p)?;
        if p.rows == 0 {
            return Ok(());
        }
        // SAFETY: `eidola_ds_window_slots(DsWindowSlotsParams)`, checked above.
        unsafe { launch!(gpu, self.window_slots, window_slots_grid(p), *p) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> AttnParams {
        let plane = KvPlane {
            base: 0x1000,
            block_bytes: 4096,
            entries_per_block: 4,
            width: 512,
            ..KvPlane::default()
        };
        AttnParams {
            q: 0x2000,
            slots: 0x3000,
            sink: 0x4000,
            out: 0x5000,
            a: plane,
            b: plane,
            rows: 3,
            heads: 64,
            head_dim: 512,
            a_slots: 128,
            b_slots: 512,
            slot_stride: 640,
            scale: 0.5,
            layout: ATTN_FIXED,
        }
    }

    #[test]
    fn a_decode_row_gets_a_block_per_head_and_a_batch_shares_entries() {
        assert_eq!(heads_per_block(1, 64), 1);
        assert_eq!(heads_per_block(4, 64), 1);
        assert_eq!(heads_per_block(5, 64), 2);
        assert_eq!(heads_per_block(16, 64), 4);
        assert_eq!(heads_per_block(64, 64), 8);
        assert_eq!(heads_per_block(384, 64), 8);
        // A grid covers every head, with no block past the last one.
        for heads in 1..=70u32 {
            for rows in [1, 7, 64, 384] {
                let group = heads_per_block(rows, heads);
                let p = AttnParams {
                    heads,
                    rows,
                    ..params()
                };
                let grid = attn_grid(&p, group).unwrap();
                assert_eq!(grid[1], rows);
                assert!(grid[0] * group >= heads && (grid[0] - 1) * group < heads);
                // The group the kernel derives from the grid is within its shared memory.
                assert!(heads.div_ceil(grid[0]) <= group);
            }
        }
        assert!(attn_grid(&params(), 0).is_err());
        assert!(attn_grid(&params(), 9).is_err());
    }

    #[test]
    fn attention_preconditions() {
        check_attn(&params()).unwrap();
        let bad = |edit: fn(&mut AttnParams)| {
            let mut p = params();
            edit(&mut p);
            check_attn(&p).is_err()
        };
        assert!(bad(|p| p.heads = 0));
        assert!(bad(|p| p.head_dim = 48));
        assert!(bad(|p| {
            p.head_dim = 544;
            p.a.width = 544;
            p.b.width = 544;
        }));
        assert!(bad(|p| p.rows = 65_536));
        assert!(bad(|p| p.slot_stride = 639));
        assert!(bad(|p| p.b_slots = u32::MAX));
        assert!(bad(|p| p.layout = 2));
        assert!(bad(|p| p.q = 0));
        assert!(bad(|p| p.out = 0x5001));
        assert!(bad(|p| p.slots = 0x3002));
        assert!(bad(|p| p.sink = 0));
        assert!(bad(|p| p.a.width = 256));
        assert!(bad(|p| p.b.base = 0));
        assert!(bad(|p| p.b.entries_per_block = 0));
        assert!(bad(|p| p.a.format = 3));
        // A plane no slot names is not read.
        let mut p = params();
        p.b_slots = 0;
        p.b = KvPlane::default();
        check_attn(&p).unwrap();
    }

    fn window() -> WindowSlotsParams {
        WindowSlotsParams {
            positions: 0x100,
            first: 0x200,
            tables: 0x300,
            table_row: 0x400,
            slots: 0x500,
            status: 0x600,
            rows: 5,
            window: 128,
            block_positions: 16,
            table_stride: 64,
            slot_stride: 640,
            order: ORDER_POSITION,
        }
    }

    #[test]
    fn window_slot_preconditions_and_grid() {
        check_window_slots(&window()).unwrap();
        assert_eq!(window_slots_grid(&window()), [1, 5, 1]);
        let wide = WindowSlotsParams {
            window: 129,
            ..window()
        };
        assert_eq!(window_slots_grid(&wide), [2, 5, 1]);
        let bad = |edit: fn(&mut WindowSlotsParams)| {
            let mut p = window();
            edit(&mut p);
            check_window_slots(&p).is_err()
        };
        assert!(bad(|p| p.window = 0));
        assert!(bad(|p| p.window = 641));
        assert!(bad(|p| p.block_positions = 0));
        assert!(bad(|p| p.table_stride = 0));
        assert!(bad(|p| p.order = 2));
        assert!(bad(|p| p.first = 0));
        assert!(bad(|p| p.status = 0));
        assert!(bad(|p| p.slots = 0x502));
        assert!(bad(|p| p.rows = 70_000));
        // Ring order reads no first positions.
        let mut p = window();
        p.order = ORDER_RING;
        p.first = 0;
        check_window_slots(&p).unwrap();
    }

    /// The canonical list of `forward.rs` (the window right-aligned, a position
    /// before the start or below the first one an empty slot) and the drafter's
    /// (`ring_order`).
    #[test]
    fn window_positions_are_the_references() {
        assert_eq!(
            window_positions(ORDER_POSITION, 4, 1, 0),
            [None, None, Some(0), Some(1)]
        );
        assert_eq!(
            window_positions(ORDER_POSITION, 4, 9, 8),
            [None, None, Some(8), Some(9)]
        );
        assert_eq!(
            window_positions(ORDER_RING, 4, 2, 0),
            [Some(0), Some(1), Some(2), None]
        );
        assert_eq!(
            window_positions(ORDER_RING, 4, 5, 0),
            [Some(4), Some(5), Some(2), Some(3)]
        );
        for window in [1usize, 3, 48, 128, 129] {
            for pos in 0..3 * window + 2 {
                let listed: Vec<usize> = window_positions(ORDER_POSITION, window, pos, 0)
                    .into_iter()
                    .flatten()
                    .collect();
                let from = (pos + 1).saturating_sub(window);
                assert_eq!(listed, (from..=pos).collect::<Vec<_>>());
                let ring = window_positions(ORDER_RING, window, pos, 0);
                let listed: Vec<usize> = ring.iter().copied().flatten().collect();
                assert_eq!(listed, ring_order(pos, window));
                // The ring's empty slots are all behind its listed ones.
                assert!(ring[listed.len()..].iter().all(Option::is_none));
            }
        }
    }

    #[test]
    fn slot_words_go_through_the_block_table() {
        let table = [5, -1, 2];
        assert_eq!(table_entry(&table, 3, 4), Some(23));
        assert_eq!(table_entry(&table, 4, 4), None);
        assert_eq!(table_entry(&table, 9, 4), Some(9));
        let (slots, unmapped) = window_slots(ORDER_POSITION, 4, 9, 0, &table, 4);
        assert_eq!(slots, [SLOT_EMPTY, SLOT_EMPTY, 8, 9]);
        assert!(unmapped);
        let (slots, unmapped) = window_slots(ORDER_POSITION, 2, 9, 0, &table, 4);
        assert_eq!(slots, [8, 9]);
        assert!(!unmapped);
    }
}
