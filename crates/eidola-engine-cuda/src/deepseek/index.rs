//! The indexer's scores and the selections over them (`ds_index.cu`):
//! `eidola_ds_index_score` scores every visible entry of each row's bank, and
//! `eidola_ds_select` takes the total-order top-k of them, writes the
//! candidate blocks, and appends the selection to the attention slot lists.
//! Bit-identical to `eidola_engine_model::deepseek::attention`'s
//! `index_score`, `top_k` and `candidate_mask`
//! (`eidola-engine-kernels/AGENTS.md` → *DeepSeek-V4.1 attention, index and selection*).
//!
//! The launch geometry is a function of the row count and the configuration
//! (the score rows' capacity), never of how many entries a row sees.

use eidola_engine_model::deepseek::attention::{candidate_mask, top_k};

use super::abi::{IndexScoreParams, KV_FP4_E8M0, SLOT_EMPTY, SelectParams};
use super::attention::table_entry;
use crate::bf16::to_f32 as bf16_to_f32;
use crate::module::{Kernel, KernelDir, KernelModule};
use crate::{CudaError, Gpu, Result, launch};

/// Threads per block of `eidola_ds_index_score`: a warp per entry, a lane per
/// index head.
pub const SCORE_THREADS: u32 = 256;
/// Entries a block scores per pass.
pub const SCORE_PASS: u32 = 8;
/// The most index heads, and the widest one.
pub const MAX_INDEX_HEADS: u32 = 32;
pub const MAX_INDEX_DIM: u32 = 128;
/// Threads per block of `eidola_ds_select`: one block per row.
pub const SELECT_THREADS: u32 = 512;
/// The driver's grid limits.
const MAX_GRID_X: u32 = i32::MAX as u32;
const MAX_GRID_Y: u32 = 65_535;
/// Blocks of `eidola_ds_index_score` a 148-SM part keeps resident (eight an
/// SM, by threads).
const RESIDENT_SCORE_BLOCKS: u32 = 1184;

/// The grid of `eidola_ds_index_score`: one block row per row, and per row as
/// many blocks as a row's capacity has passes, up to the row's share of the
/// part. A block walks the passes congruent to its index, up to the entries
/// its row sees (device data), so the grid does not depend on them.
pub fn score_grid(p: &IndexScoreParams) -> [u32; 3] {
    let passes = p.score_stride.div_ceil(SCORE_PASS);
    let share = (RESIDENT_SCORE_BLOCKS / p.rows.max(1)).max(1);
    [passes.clamp(1, share), p.rows, 1]
}

/// What `eidola_ds_index_score` needs of its parameters; the visible counts
/// and table words are device data (a count past `score_stride` or a table
/// word that maps nothing raises `STATUS_BAD_INDEX`).
pub fn check_score(p: &IndexScoreParams) -> Result<()> {
    if p.heads == 0
        || p.heads > MAX_INDEX_HEADS
        || p.dim == 0
        || !p.dim.is_multiple_of(32)
        || p.dim > MAX_INDEX_DIM
    {
        return Err(CudaError::new(format!(
            "ds_index_score: {} heads of {} (at most {MAX_INDEX_HEADS} heads; a head is a \
             multiple of 32, at most {MAX_INDEX_DIM})",
            p.heads, p.dim
        )));
    }
    if p.rows > MAX_GRID_Y || p.score_stride == 0 || p.table_stride == 0 {
        return Err(CudaError::new(format!(
            "ds_index_score: {} rows, score rows of {}, tables of {}",
            p.rows, p.score_stride, p.table_stride
        )));
    }
    if p.keys.base == 0
        || p.keys.entries_per_block == 0
        || p.keys.format > KV_FP4_E8M0
        || p.keys.width != p.dim
    {
        return Err(CudaError::new(
            "ds_index_score: the key plane is not set for this head width",
        ));
    }
    let words = [p.tables, p.table_row, p.visible, p.status];
    if words.contains(&0) || [p.q_codes, p.q_scales, p.head_dots, p.scores].contains(&0) {
        return Err(CudaError::new("ds_index_score: a null array"));
    }
    if words.iter().any(|a| !a.is_multiple_of(4))
        || !p.head_dots.is_multiple_of(2)
        || !p.scores.is_multiple_of(2)
    {
        return Err(CudaError::new("ds_index_score: a misaligned array"));
    }
    Ok(())
}

/// Mask words a row of candidate blocks needs for `score_stride` entries.
pub fn mask_words(score_stride: u32, block_entries: u32) -> u32 {
    score_stride.div_ceil(block_entries).div_ceil(32)
}

/// What `eidola_ds_select` needs of its parameters; the visible counts,
/// scores, masks and table words are device data.
pub fn check_select(p: &SelectParams) -> Result<()> {
    if p.rows > MAX_GRID_X || p.score_stride == 0 || p.table_stride == 0 {
        return Err(CudaError::new(format!(
            "ds_select: {} rows, score rows of {}, tables of {}",
            p.rows, p.score_stride, p.table_stride
        )));
    }
    if p.block_entries == 0 || p.entries_per_block == 0 {
        return Err(CudaError::new("ds_select: blocks of no entries"));
    }
    if p.slot_first
        .checked_add(p.top_k)
        .is_none_or(|n| n > p.slot_stride)
    {
        return Err(CudaError::new(format!(
            "ds_select: slots {} + {} in rows of {}",
            p.slot_first, p.top_k, p.slot_stride
        )));
    }
    let masked = p.mask_in != 0 || p.mask_out != 0;
    if masked && p.mask_words < mask_words(p.score_stride, p.block_entries) {
        return Err(CudaError::new(format!(
            "ds_select: {} mask words for {} entries in blocks of {}",
            p.mask_words, p.score_stride, p.block_entries
        )));
    }
    if masked && p.mask_in == p.mask_out {
        return Err(CudaError::new(
            "ds_select: the candidate blocks read and written are one array",
        ));
    }
    let words = [p.visible, p.tables, p.table_row, p.slots, p.status];
    if words.contains(&0) || p.scores == 0 {
        return Err(CudaError::new("ds_select: a null array"));
    }
    if words
        .iter()
        .chain([&p.mask_in, &p.mask_out])
        .any(|a| !a.is_multiple_of(4))
        || !p.scores.is_multiple_of(2)
    {
        return Err(CudaError::new("ds_select: a misaligned array"));
    }
    Ok(())
}

/// The candidate blocks a row keeps, as mask words (bit `b % 32` of word
/// `b / 32`): the host form of `eidola_ds_select`'s `mask_out`, which is
/// `attention::candidate_mask` per block. `scores` are the row's visible
/// entries' BF16 scores.
pub fn candidate_blocks(
    scores: &[u16],
    top_blocks: usize,
    block_entries: usize,
    words: usize,
) -> Vec<u32> {
    let values: Vec<f32> = scores.iter().map(|b| bf16_to_f32(*b)).collect();
    let kept = candidate_mask(&values, values.len(), top_blocks, block_entries);
    let mut mask = vec![0u32; words];
    for b in (0..values.len().div_ceil(block_entries)).filter(|b| kept[b * block_entries]) {
        mask[b / 32] |= 1 << (b % 32);
    }
    mask
}

/// The global part of one row's slot list: the host form of
/// `eidola_ds_select`'s slots. The best `k` of the row's visible entries
/// (inside the candidate blocks of `mask`, when given) under
/// `attention::top_k`'s total order, ascending, as pool entries through the
/// bank's block table, then empty slots. The second value is whether a
/// selected entry had no block.
pub fn global_slots(
    scores: &[u16],
    mask: Option<&[u32]>,
    k: usize,
    block_entries: usize,
    table: &[i32],
    entries_per_block: usize,
) -> (Vec<u32>, bool) {
    let values: Vec<f32> = scores.iter().map(|b| bf16_to_f32(*b)).collect();
    let allowed = (0..values.len()).filter(|t| {
        let b = t / block_entries;
        mask.is_none_or(|m| m.get(b / 32).is_some_and(|w| w >> (b % 32) & 1 == 1))
    });
    let mut chosen = top_k(&values, allowed, k);
    chosen.sort_unstable();
    let mut unmapped = false;
    let mut slots: Vec<u32> = chosen
        .into_iter()
        .map(|t| {
            table_entry(table, t, entries_per_block).unwrap_or_else(|| {
                unmapped = true;
                SLOT_EMPTY
            })
        })
        .collect();
    slots.resize(k, SLOT_EMPTY);
    (slots, unmapped)
}

/// The loaded index kernels.
pub struct DsIndex {
    _module: KernelModule,
    score: Kernel,
    select: Kernel,
}

impl DsIndex {
    pub const IMAGE: &'static str = "ds_index";

    pub fn load(gpu: &Gpu, dir: &KernelDir) -> Result<DsIndex> {
        DsIndex::from_module(KernelModule::load(gpu, dir, Self::IMAGE)?)
    }

    pub fn from_module(m: KernelModule) -> Result<DsIndex> {
        let score = m.bound_kernel(Self::IMAGE, "eidola_ds_index_score")?;
        let select = m.bound_kernel(Self::IMAGE, "eidola_ds_select")?;
        let contract = |k: &Kernel, threads: u32, params: usize| {
            k.meta().block == [threads, 1, 1] && k.meta().params_bytes as usize == params
        };
        if !contract(
            &score,
            SCORE_THREADS,
            std::mem::size_of::<IndexScoreParams>(),
        ) || !contract(&select, SELECT_THREADS, std::mem::size_of::<SelectParams>())
        {
            return Err(CudaError::new("ds_index: launch contract"));
        }
        Ok(DsIndex {
            _module: m,
            score,
            select,
        })
    }

    /// Scores every visible entry of rows `0 .. p.rows`.
    ///
    /// # Safety
    ///
    /// Every array `p` names must be a live device allocation of this context
    /// holding what [`IndexScoreParams`] says, until the launch completes;
    /// every table row must hold `table_stride` words, each negative or a
    /// block of the key plane.
    pub unsafe fn score(&self, gpu: &Gpu, p: &IndexScoreParams) -> Result<()> {
        // SAFETY: the caller's contract.
        unsafe { self.score_on(gpu, p, score_grid(p)[0]) }
    }

    /// [`DsIndex::score`] over a grid `width` blocks wide, for the tests and
    /// the bench: the scores are the same for every width.
    ///
    /// # Safety
    ///
    /// As [`DsIndex::score`].
    pub unsafe fn score_on(&self, gpu: &Gpu, p: &IndexScoreParams, width: u32) -> Result<()> {
        check_score(p)?;
        if width == 0 {
            return Err(CudaError::new("ds_index_score: an empty grid"));
        }
        if p.rows == 0 {
            return Ok(());
        }
        // SAFETY: `eidola_ds_index_score(DsIndexScoreParams)`, checked above.
        unsafe { launch!(gpu, self.score, [width, p.rows, 1], *p) }
    }

    /// Selects for rows `0 .. p.rows`: candidate blocks into `mask_out` when
    /// given, and the top-k into slots `slot_first .. slot_first + top_k`.
    ///
    /// # Safety
    ///
    /// Every array `p` names must be a live device allocation of this context
    /// holding what [`SelectParams`] says, until the launch completes; every
    /// table row must hold `table_stride` words covering `score_stride`
    /// entries.
    pub unsafe fn select(&self, gpu: &Gpu, p: &SelectParams) -> Result<()> {
        check_select(p)?;
        if p.rows == 0 {
            return Ok(());
        }
        // SAFETY: `eidola_ds_select(DsSelectParams)`, checked above.
        unsafe { launch!(gpu, self.select, [p.rows, 1, 1], *p) }
    }
}

#[cfg(test)]
mod tests {
    use super::super::abi::KvPlane;
    use super::*;

    fn score() -> IndexScoreParams {
        IndexScoreParams {
            q_codes: 0x100,
            q_scales: 0x200,
            head_dots: 0x300,
            tables: 0x400,
            table_row: 0x500,
            visible: 0x600,
            scores: 0x700,
            status: 0x800,
            keys: KvPlane {
                base: 0x1000,
                block_bytes: 4096,
                entries_per_block: 16,
                format: KV_FP4_E8M0,
                width: 128,
                ..KvPlane::default()
            },
            rows: 1,
            heads: 32,
            dim: 128,
            table_stride: 64,
            score_stride: 131_072,
            weight_scale: 0.5,
        }
    }

    #[test]
    fn score_grid_is_a_function_of_rows_and_capacity() {
        // One row takes the whole part; rows share it; never an empty grid.
        assert_eq!(score_grid(&score()), [1184, 1, 1]);
        let grid = |rows, score_stride| {
            score_grid(&IndexScoreParams {
                rows,
                score_stride,
                ..score()
            })
        };
        assert_eq!(grid(64, 131_072), [18, 64, 1]);
        assert_eq!(grid(384, 131_072), [3, 384, 1]);
        assert_eq!(grid(2000, 131_072), [1, 2000, 1]);
        // A short capacity needs no more blocks than it has passes.
        assert_eq!(grid(1, 20), [3, 1, 1]);
        assert_eq!(grid(1, 8), [1, 1, 1]);
        assert_eq!(grid(0, 8), [1, 0, 1]);
    }

    #[test]
    fn score_preconditions() {
        check_score(&score()).unwrap();
        let bad = |edit: fn(&mut IndexScoreParams)| {
            let mut p = score();
            edit(&mut p);
            check_score(&p).is_err()
        };
        assert!(bad(|p| p.heads = 33));
        assert!(bad(|p| p.heads = 0));
        assert!(bad(|p| {
            p.dim = 160;
            p.keys.width = 160;
        }));
        assert!(bad(|p| p.dim = 64));
        assert!(bad(|p| p.keys.width = 64));
        assert!(bad(|p| p.keys.base = 0));
        assert!(bad(|p| p.keys.entries_per_block = 0));
        assert!(bad(|p| p.score_stride = 0));
        assert!(bad(|p| p.table_stride = 0));
        assert!(bad(|p| p.rows = 65_536));
        assert!(bad(|p| p.visible = 0));
        assert!(bad(|p| p.scores = 0x701));
        assert!(bad(|p| p.tables = 0x402));
    }

    fn select() -> SelectParams {
        SelectParams {
            scores: 0x100,
            visible: 0x200,
            mask_in: 0,
            mask_out: 0x300,
            tables: 0x400,
            table_row: 0x500,
            slots: 0x600,
            status: 0x700,
            rows: 4,
            score_stride: 131_072,
            mask_words: 512,
            table_stride: 64,
            slot_stride: 640,
            slot_first: 128,
            top_k: 512,
            top_blocks: 2048,
            block_entries: 8,
            entries_per_block: 16,
        }
    }

    #[test]
    fn select_preconditions() {
        check_select(&select()).unwrap();
        assert_eq!(mask_words(131_072, 8), 512);
        assert_eq!(mask_words(131_073, 8), 513);
        assert_eq!(mask_words(1, 8), 1);
        let bad = |edit: fn(&mut SelectParams)| {
            let mut p = select();
            edit(&mut p);
            check_select(&p).is_err()
        };
        assert!(bad(|p| p.mask_words = 511));
        assert!(bad(|p| p.mask_in = 0x300));
        assert!(bad(|p| p.block_entries = 0));
        assert!(bad(|p| p.entries_per_block = 0));
        assert!(bad(|p| p.slot_first = 129));
        assert!(bad(|p| p.top_k = u32::MAX));
        assert!(bad(|p| p.score_stride = 0));
        assert!(bad(|p| p.table_stride = 0));
        assert!(bad(|p| p.slots = 0));
        assert!(bad(|p| p.scores = 0x101));
        assert!(bad(|p| p.mask_out = 0x302));
        // Without candidate blocks the mask words are not looked at.
        let mut p = select();
        p.mask_out = 0;
        p.mask_words = 0;
        check_select(&p).unwrap();
    }

    /// The host forms on the reference's own small examples
    /// (`attention::tests`).
    #[test]
    fn host_forms_follow_the_reference() {
        let bits = |v: &[f32]| -> Vec<u16> {
            v.iter()
                .map(|x| u16::try_from(x.to_bits() >> 16).unwrap())
                .collect()
        };
        // Ties to the lower index; -0 and +0 one score.
        let scores = bits(&[1.0, 3.0, 3.0, -0.0, 0.0, 2.0, 3.0]);
        let table = [3, 0];
        let (slots, unmapped) = global_slots(&scores, None, 4, 2, &table, 4);
        assert_eq!(slots, [13, 14, 1, 2]);
        assert!(!unmapped);
        let (slots, _) = global_slots(&scores[..5], None, 9, 2, &table, 4);
        assert_eq!(slots[..5], [12, 13, 14, 15, 0]);
        assert!(slots[5..].iter().all(|s| *s == SLOT_EMPTY));
        // Blocks of 2 over (5,1) (4,4) (0,0) (3): the newest and the best other.
        let scores = bits(&[5.0, 1.0, 4.0, 4.0, 0.0, 0.0, 3.0]);
        assert_eq!(candidate_blocks(&scores, 2, 2, 2), [0b1001, 0]);
        assert_eq!(candidate_blocks(&scores, 8, 2, 1), [0b1111]);
        // Selection inside the kept blocks only.
        let (slots, _) = global_slots(&scores, Some(&[0b1001]), 3, 2, &[0, 1], 4);
        assert_eq!(slots, [0, 1, 6]);
        let (slots, unmapped) = global_slots(&scores, Some(&[0b1001]), 3, 2, &[0, -1], 4);
        assert_eq!(slots, [0, 1, SLOT_EMPTY]);
        assert!(unmapped);
    }
}
