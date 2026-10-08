//! Device KV memory: one pool per KV group, the per-slot block tables, per-slot
//! model state, and the seam's maintenance operations on them.
//!
//! A group's pool is `num_blocks` blocks plus one pad block (id `num_blocks`,
//! [`GroupGeometry::pad_block`]). A target group is laid out by block
//! ([`KvLayout::Blocked`]): block `b` holds, for each of the group's layers in
//! order, K as `[block_size, kv_heads, head_dim_qk]` then V as
//! `[block_size, kv_heads, head_dim_v]`, BF16, contiguous. A block is therefore
//! one contiguous byte range across every layer: `Zero` is one memset and `Copy`
//! one device-to-device copy. Paged attention addresses a layer's K (or V) with
//! page stride = the whole block and the layer's offset inside it.
//!
//! The drafter group is laid out by layer ([`KvLayout::Planar`]): each layer
//! (MTP depth) has a K plane and a V plane of position rows, row `b *
//! block_size + o` holding position `o` of block `b`, so attention can list
//! exactly the positions a drafter row sees, one position per page (depth `d`
//! has no KV below position `d + 1`, which a block's first page would
//! otherwise expose). Beside the planes, every block holds a **boundary tap**:
//! the drafter's chain levels at the block's last position, `taps` f32 rows
//! of `hidden` (`eidola-engine`'s executor contract). `Zero` and `Copy` of a
//! drafter block touch its rows in every plane and its tap, and nothing else.
//!
//! The pad block is outside every id the seam may name (`1..num_blocks`):
//! no table maps it and no maintenance reaches it. A replayed decode graph
//! points its padding rows' KV writes and attention reads at it, so padding
//! never touches a block the host allocates, shares or caches; it holds
//! only what padding rows wrote (functions of the constant padding token).
//!
//! Block tables live on the device as `[slot][group][index]` `i32` rows (the
//! attention kernels' index type) and change only through table updates and
//! `ResetSlot`. The host keeps an exact mirror, which is how it plans attention
//! and checks the contract: a write into a block mapped by more than one table
//! entry (a shared prefix-cache block) or a read through an unmapped entry is a
//! host bug and panics, as in the CPU reference executor.
//!
//! A step's maintenance and table updates are validated whole before any of
//! them takes effect: [`KvStore::stage`] checks them and computes the mirror
//! they lead to without touching the device or the mirror, and
//! [`KvStore::commit`] then queues the device work and installs that mirror.
//! A malformed step panics or errs in `stage`, leaving everything as it was.

use std::sync::Arc;

use cudarc::driver::sys;
use cudarc::driver::{CudaSlice, CudaStream};
use eidola_engine::executor::{Maintenance, Slot, TableUpdate};
use eidola_engine::spec::NULL_BLOCK;

use crate::launch::dptr;
use crate::{CudaError, Gpu, Result};

/// How a group's pool is laid out (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvLayout {
    /// By block: every layer's K then V of a block, contiguous (target groups).
    Blocked,
    /// By layer: a K plane and a V plane of position rows per layer, and a
    /// boundary tap of `taps` f32 rows of `hidden` per block (the drafter).
    Planar { taps: u32, hidden: u32 },
}

/// The shape of one KV group's pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupGeometry {
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim_qk: u32,
    pub head_dim_v: u32,
    pub block_size: u32,
    pub num_blocks: u32,
    pub layout: KvLayout,
}

/// One contiguous piece of a block: in the group's KV pool (BF16 elements)
/// or in its taps (f32 elements).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockRange {
    Pool { offset: usize, elems: usize },
    Taps { offset: usize, elems: usize },
}

impl BlockRange {
    /// `(byte offset, bytes)` in its buffer.
    pub fn bytes(self) -> (usize, usize) {
        match self {
            BlockRange::Pool { offset, elems } => (offset * 2, elems * 2),
            BlockRange::Taps { offset, elems } => (offset * 4, elems * 4),
        }
    }
}

impl GroupGeometry {
    /// Check that every size derived from the geometry fits: a block's
    /// elements in the 32-bit page stride the attention kernel takes, the
    /// pool's bytes in `usize`, block ids in the device tables' `i32`.
    /// [`KvStore::new`] refuses a geometry that fails; the accessors below
    /// assume one that passed.
    pub fn validate(&self) -> Result<()> {
        let overflow = || CudaError::new(format!("KV geometry {self:?} overflows"));
        let m = |a: usize, b: u32| a.checked_mul(b as usize).ok_or_else(overflow);
        let k = m(
            m(self.block_size as usize, self.num_kv_heads)?,
            self.head_dim_qk,
        )?;
        let v = m(
            m(self.block_size as usize, self.num_kv_heads)?,
            self.head_dim_v,
        )?;
        let block = m(k.checked_add(v).ok_or_else(overflow)?, self.num_layers)?;
        if block == 0 || block > u32::MAX as usize {
            return Err(CudaError::new(format!(
                "KV geometry {self:?}: {block} elements per block, the page stride is 32-bit"
            )));
        }
        block
            .checked_mul(self.pool_blocks())
            .and_then(|e| e.checked_mul(2))
            .ok_or_else(overflow)?;
        if self.num_blocks < 2 || self.num_blocks > i32::MAX as u32 {
            return Err(CudaError::new(format!(
                "KV geometry {self:?}: blocks must be 2..=i32::MAX (the null block and one more; i32 tables)"
            )));
        }
        if let KvLayout::Planar { taps, hidden } = self.layout {
            // Position rows are the attention kernel's `i32` page ids and the
            // KV write's `u32` slot; a block's tap rows are copy-kernel rows.
            let rows = self
                .pool_blocks()
                .checked_mul(self.block_size as usize)
                .ok_or_else(overflow)?;
            let tap_rows = self
                .pool_blocks()
                .checked_mul(taps as usize)
                .ok_or_else(overflow)?;
            if rows > i32::MAX as usize || tap_rows > u32::MAX as usize {
                return Err(CudaError::new(format!(
                    "KV geometry {self:?}: {rows} position rows past the kernels' i32 page ids"
                )));
            }
            if taps == 0 || hidden == 0 {
                return Err(CudaError::new(format!(
                    "KV geometry {self:?}: a drafter block holds a tap"
                )));
            }
            tap_rows
                .checked_mul(hidden as usize)
                .and_then(|e| e.checked_mul(4))
                .ok_or_else(overflow)?;
            (taps as usize)
                .checked_mul(hidden as usize)
                .and_then(|e| e.checked_mul(4))
                .and_then(|t| t.checked_add(block * 2))
                .ok_or_else(overflow)?;
        }
        Ok(())
    }

    /// Position rows each plane holds (planar layout): every position of
    /// every block, the pad block's included.
    pub fn pool_rows(&self) -> usize {
        self.pool_blocks() * self.block_size as usize
    }

    /// BF16 elements of one position's K (every KV head).
    pub fn k_row_elems(&self) -> usize {
        self.num_kv_heads as usize * self.head_dim_qk as usize
    }

    /// BF16 elements of one position's V (every KV head).
    pub fn v_row_elems(&self) -> usize {
        self.num_kv_heads as usize * self.head_dim_v as usize
    }

    /// Element offset of layer `layer`'s K plane (planar layout).
    pub fn plane_k_offset(&self, layer: u32) -> usize {
        layer as usize * self.pool_rows() * (self.k_row_elems() + self.v_row_elems())
    }

    /// Element offset of layer `layer`'s V plane (planar layout).
    pub fn plane_v_offset(&self, layer: u32) -> usize {
        self.plane_k_offset(layer) + self.pool_rows() * self.k_row_elems()
    }

    /// Tap rows per block (0 for a blocked group).
    pub fn taps(&self) -> u32 {
        match self.layout {
            KvLayout::Blocked => 0,
            KvLayout::Planar { taps, .. } => taps,
        }
    }

    /// f32 elements of one block's tap (0 for a blocked group).
    pub fn tap_elems(&self) -> usize {
        match self.layout {
            KvLayout::Blocked => 0,
            KvLayout::Planar { taps, hidden } => taps as usize * hidden as usize,
        }
    }

    /// Every contiguous piece of block `block`: the one range of a blocked
    /// group; the block's rows in each layer's K and V planes, then its tap,
    /// for a planar one.
    pub fn block_ranges(&self, block: u32) -> Vec<BlockRange> {
        let b = block as usize;
        match self.layout {
            KvLayout::Blocked => vec![BlockRange::Pool {
                offset: b * self.block_elems(),
                elems: self.block_elems(),
            }],
            KvLayout::Planar { .. } => {
                let rows = self.block_size as usize;
                let mut v = Vec::with_capacity(2 * self.num_layers as usize + 1);
                for l in 0..self.num_layers {
                    v.push(BlockRange::Pool {
                        offset: self.plane_k_offset(l) + b * rows * self.k_row_elems(),
                        elems: rows * self.k_row_elems(),
                    });
                    v.push(BlockRange::Pool {
                        offset: self.plane_v_offset(l) + b * rows * self.v_row_elems(),
                        elems: rows * self.v_row_elems(),
                    });
                }
                v.push(BlockRange::Taps {
                    offset: b * self.tap_elems(),
                    elems: self.tap_elems(),
                });
                v
            }
        }
    }

    /// The pad block's id: one past the last id the seam may name, so it
    /// still fits the `i32` the attention kernel indexes pages with.
    pub fn pad_block(&self) -> u32 {
        self.num_blocks
    }

    /// Blocks the pool holds: the seam's `num_blocks` and the pad block.
    pub fn pool_blocks(&self) -> usize {
        self.num_blocks as usize + 1
    }

    /// Elements of one layer's K in one block.
    pub fn k_elems(&self) -> usize {
        self.block_size as usize * self.num_kv_heads as usize * self.head_dim_qk as usize
    }

    /// Elements of one layer's V in one block.
    pub fn v_elems(&self) -> usize {
        self.block_size as usize * self.num_kv_heads as usize * self.head_dim_v as usize
    }

    /// Elements of one layer in one block.
    pub fn layer_elems(&self) -> usize {
        self.k_elems() + self.v_elems()
    }

    /// Elements of one block (every layer): the page stride.
    pub fn block_elems(&self) -> usize {
        self.layer_elems() * self.num_layers as usize
    }

    /// Device bytes one block takes: its KV, and its tap for a drafter.
    pub fn block_bytes(&self) -> usize {
        self.block_elems() * 2 + self.tap_elems() * 4
    }

    /// Element offset of layer `layer`'s K inside a block.
    pub fn k_offset(&self, layer: u32) -> usize {
        self.layer_elems() * layer as usize
    }

    /// Element offset of layer `layer`'s V inside a block.
    pub fn v_offset(&self, layer: u32) -> usize {
        self.k_offset(layer) + self.k_elems()
    }
}

/// The host mirror of the block tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableMirror {
    /// `[slot][group][index]`.
    tables: Vec<Vec<Vec<u32>>>,
    /// `[group][block]`: number of table entries mapping it.
    mapped: Vec<Vec<u32>>,
    /// Positions per block, per group.
    block_size: Vec<u32>,
}

impl TableMirror {
    /// Empty tables: `slots` rows of `max_blocks` null entries per group.
    fn new(geometry: &[GroupGeometry], slots: usize, max_blocks: usize) -> TableMirror {
        TableMirror {
            tables: vec![vec![vec![NULL_BLOCK; max_blocks]; geometry.len()]; slots],
            mapped: geometry
                .iter()
                .map(|g| vec![0; g.num_blocks as usize])
                .collect(),
            block_size: geometry.iter().map(|g| g.block_size).collect(),
        }
    }

    fn set(&mut self, slot: usize, group: usize, index: usize, block: u32) {
        let old = std::mem::replace(&mut self.tables[slot][group][index], block);
        if old != NULL_BLOCK {
            self.mapped[group][old as usize] -= 1;
        }
        if block != NULL_BLOCK {
            self.mapped[group][block as usize] += 1;
        }
    }

    /// One table row.
    pub fn table(&self, slot: Slot, group: usize) -> &[u32] {
        &self.tables[slot as usize][group]
    }

    /// The block holding `pos` for `slot` in `group`; panics if unmapped (a
    /// read the host never provided for), or past the table.
    pub fn block_of(&self, slot: Slot, group: usize, pos: u32) -> u32 {
        let row = &self.tables[slot as usize][group];
        let index = (pos / self.block_size[group]) as usize;
        assert!(
            index < row.len(),
            "slot {slot} group {group} position {pos}: past the table"
        );
        let block = row[index];
        assert!(
            block != NULL_BLOCK,
            "slot {slot} group {group} position {pos}: unmapped block"
        );
        block
    }

    /// The block a write at `pos` lands in, which must be mapped by exactly
    /// one table entry: blocks shared through the prefix cache are immutable.
    pub fn writable_block(&self, slot: Slot, group: usize, pos: u32) -> u32 {
        let block = self.block_of(slot, group, pos);
        assert!(
            self.mapped[group][block as usize] == 1,
            "slot {slot} group {group} position {pos}: write to shared block {block}"
        );
        block
    }
}

/// Which buffer of a group a device operation touches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mem {
    Pool(usize),
    Taps(usize),
}

/// Device work a staged step queues.
#[derive(Clone, Copy, Debug)]
enum DeviceOp {
    Zero {
        mem: Mem,
        offset: usize,
        bytes: usize,
    },
    Copy {
        mem: Mem,
        src: usize,
        dst: usize,
        bytes: usize,
    },
    ResetState {
        slot: usize,
    },
}

/// A step's maintenance and table updates, validated, with the mirror they
/// lead to; nothing has happened yet ([`KvStore::commit`] applies it).
#[derive(Debug)]
pub struct Staged {
    generation: u64,
    mirror: TableMirror,
    ops: Vec<DeviceOp>,
    dirty: Vec<(usize, usize)>,
}

impl Staged {
    /// The tables as they will be once committed.
    pub fn mirror(&self) -> &TableMirror {
        &self.mirror
    }
}

fn stage(
    geometry: &[GroupGeometry],
    max_blocks: usize,
    state_width: usize,
    current: &TableMirror,
    generation: u64,
    maintenance: &[Maintenance],
    updates: &[TableUpdate],
) -> Staged {
    let mut mirror = current.clone();
    let mut ops = Vec::new();
    let mut dirty: Vec<(usize, usize)> = Vec::new();
    let slots = mirror.tables.len();
    for m in maintenance {
        match *m {
            Maintenance::Zero { group, block } => {
                let (g, ranges) = block_ranges(geometry, group, block);
                for r in ranges {
                    let (mem, (offset, bytes)) = (mem_of(g, r), r.bytes());
                    ops.push(DeviceOp::Zero { mem, offset, bytes });
                }
            }
            Maintenance::Copy { group, src, dst } => {
                let (g, from) = block_ranges(geometry, group, src);
                let (_, to) = block_ranges(geometry, group, dst);
                for (a, b) in from.into_iter().zip(to) {
                    let ((src, bytes), (dst, _)) = (a.bytes(), b.bytes());
                    ops.push(DeviceOp::Copy {
                        mem: mem_of(g, a),
                        src,
                        dst,
                        bytes,
                    });
                }
            }
            Maintenance::ResetSlot { slot } => {
                let sl = slot as usize;
                assert!(sl < slots, "reset of slot {slot} out of range");
                for g in 0..geometry.len() {
                    for idx in 0..max_blocks {
                        mirror.set(sl, g, idx, NULL_BLOCK);
                    }
                    dirty.push((sl, g));
                }
                if state_width > 0 {
                    ops.push(DeviceOp::ResetState { slot: sl });
                }
            }
        }
    }
    for u in updates {
        let (sl, g) = (u.slot as usize, u.group as usize);
        assert!(sl < slots, "table update for slot {} out of range", u.slot);
        assert!(
            g < geometry.len(),
            "table update for group {} out of range",
            u.group
        );
        assert!(
            u.block < geometry[g].num_blocks,
            "table update to block {} outside group {}",
            u.block,
            u.group
        );
        assert!(
            (u.index as usize) < max_blocks,
            "table index {} past the longest sequence",
            u.index
        );
        mirror.set(sl, g, u.index as usize, u.block);
        dirty.push((sl, g));
    }
    dirty.sort_unstable();
    dirty.dedup();
    Staged {
        generation,
        mirror,
        ops,
        dirty,
    }
}

/// The group index and every piece of a block; panics on a group or block
/// outside the pools.
fn block_ranges(geometry: &[GroupGeometry], group: u32, block: u32) -> (usize, Vec<BlockRange>) {
    let g = group as usize;
    assert!(
        g < geometry.len(),
        "maintenance on group {group} out of range"
    );
    let geom = &geometry[g];
    assert!(
        block < geom.num_blocks,
        "maintenance on block {block} outside group {group}"
    );
    (g, geom.block_ranges(block))
}

fn mem_of(group: usize, r: BlockRange) -> Mem {
    match r {
        BlockRange::Pool { .. } => Mem::Pool(group),
        BlockRange::Taps { .. } => Mem::Taps(group),
    }
}

/// KV pools, block tables and per-slot state.
pub struct KvStore {
    geometry: Vec<GroupGeometry>,
    pools: Vec<CudaSlice<u16>>,
    /// Per group: its blocks' taps (planar groups), else nothing.
    taps: Vec<Option<CudaSlice<f32>>>,
    mirror: TableMirror,
    /// Commits so far: a staged step applies only to the state it was
    /// validated against.
    generation: u64,
    /// Device tables, `[slot][group][index]`.
    tables_dev: CudaSlice<i32>,
    max_blocks: usize,
    /// Per-slot model state, `[slot][state_width]` f32 (drafter hidden states;
    /// empty when nothing is drafted).
    state: CudaSlice<f32>,
    state_width: usize,
}

impl KvStore {
    /// Allocate every pool (zeroed, taps included), `num_slots` empty tables
    /// of `max_blocks` entries per group, and `state_width` floats of zeroed
    /// state per slot.
    pub fn new(
        gpu: &Gpu,
        geometry: Vec<GroupGeometry>,
        num_slots: u32,
        max_blocks: u32,
        state_width: usize,
    ) -> Result<KvStore> {
        let s = gpu.stream();
        let mut pools = Vec::with_capacity(geometry.len());
        let groups = geometry.len();
        let (slots, mb) = (num_slots as usize, max_blocks as usize);
        // Every size is checked before anything is allocated.
        for g in &geometry {
            g.validate()?;
        }
        let overflow = || CudaError::new("KV tables or slot state overflow");
        let table_entries = slots
            .checked_mul(groups)
            .and_then(|x| x.checked_mul(mb))
            .ok_or_else(overflow)?;
        let state_len = slots.checked_mul(state_width).ok_or_else(overflow)?;
        let mut taps = Vec::with_capacity(groups);
        for g in &geometry {
            pools.push(s.alloc_zeros::<u16>(g.block_elems() * g.pool_blocks())?);
            taps.push(match g.layout {
                KvLayout::Blocked => None,
                KvLayout::Planar { .. } => {
                    Some(s.alloc_zeros::<f32>(g.tap_elems() * g.pool_blocks())?)
                }
            });
        }
        Ok(KvStore {
            mirror: TableMirror::new(&geometry, slots, mb),
            generation: 0,
            tables_dev: s.alloc_zeros::<i32>(table_entries.max(1))?,
            state: s.alloc_zeros::<f32>(state_len.max(1))?,
            max_blocks: mb,
            state_width,
            geometry,
            pools,
            taps,
        })
    }

    /// A planar group's taps: `pool_blocks × taps` rows of `hidden` f32.
    pub fn taps(&self, group: usize) -> Option<&CudaSlice<f32>> {
        self.taps[group].as_ref()
    }

    /// f32 per slot of model state.
    pub fn state_width(&self) -> usize {
        self.state_width
    }

    pub fn geometry(&self) -> &[GroupGeometry] {
        &self.geometry
    }

    pub fn pool(&self, group: usize) -> &CudaSlice<u16> {
        &self.pools[group]
    }

    pub fn pool_mut(&mut self, group: usize) -> &mut CudaSlice<u16> {
        &mut self.pools[group]
    }

    pub fn tables_dev(&self) -> &CudaSlice<i32> {
        &self.tables_dev
    }

    pub fn max_blocks(&self) -> usize {
        self.max_blocks
    }

    pub fn state(&self) -> &CudaSlice<f32> {
        &self.state
    }

    /// The host mirror.
    pub fn mirror(&self) -> &TableMirror {
        &self.mirror
    }

    /// The host mirror of one table row.
    pub fn table(&self, slot: Slot, group: usize) -> &[u32] {
        self.mirror.table(slot, group)
    }

    /// Validate a step's maintenance, then its table updates, in order (the
    /// seam's steps 1 and 2), and compute the tables they lead to, without
    /// touching the device or the mirror. A contract violation (a slot,
    /// group, block or index out of range) panics here, before anything has
    /// changed.
    pub fn stage(&self, maintenance: &[Maintenance], updates: &[TableUpdate]) -> Staged {
        stage(
            &self.geometry,
            self.max_blocks,
            self.state_width,
            &self.mirror,
            self.generation,
            maintenance,
            updates,
        )
    }

    /// Queue a staged step's device work on the stream (ahead of the
    /// forward), upload the table rows it changed, and install its mirror.
    pub fn commit(&mut self, gpu: &Gpu, staged: Staged) -> Result<()> {
        assert_eq!(
            staged.generation, self.generation,
            "a staged step applies only to the state it was validated against"
        );
        let s = gpu.stream();
        // The raw driver calls below need this thread's current context.
        gpu.context().bind_to_thread()?;
        for op in &staged.ops {
            match *op {
                DeviceOp::Zero { mem, offset, bytes } => {
                    let base = self.mem_base(s, mem) + offset as u64;
                    // SAFETY: `stage` held the range inside the buffer.
                    unsafe { sys::cuMemsetD8Async(base, 0, bytes, s.cu_stream()) }.result()?;
                }
                DeviceOp::Copy {
                    mem,
                    src,
                    dst,
                    bytes,
                } => {
                    let base = self.mem_base(s, mem);
                    // SAFETY: `stage` held both ranges inside the pool;
                    // distinct blocks do not overlap.
                    unsafe {
                        sys::cuMemcpyDtoDAsync_v2(
                            base + dst as u64,
                            base + src as u64,
                            bytes,
                            s.cu_stream(),
                        )
                    }
                    .result()?;
                }
                DeviceOp::ResetState { slot } => {
                    let w = self.state_width;
                    let base = dptr(&self.state, s) + (slot * w * 4) as u64;
                    // SAFETY: `stage` held the slot inside the state buffer.
                    unsafe { sys::cuMemsetD8Async(base, 0, w * 4, s.cu_stream()) }.result()?;
                }
            }
        }
        self.mirror = staged.mirror;
        self.generation += 1;
        for &(sl, g) in &staged.dirty {
            self.upload_row(s, sl, g)?;
        }
        Ok(())
    }

    /// Stage, then commit: maintenance and table updates, validated whole
    /// before any takes effect.
    pub fn apply(
        &mut self,
        gpu: &Gpu,
        maintenance: &[Maintenance],
        updates: &[TableUpdate],
    ) -> Result<()> {
        let staged = self.stage(maintenance, updates);
        self.commit(gpu, staged)
    }

    fn mem_base(&self, s: &Arc<CudaStream>, mem: Mem) -> u64 {
        match mem {
            Mem::Pool(g) => dptr(&self.pools[g], s),
            Mem::Taps(g) => dptr(
                self.taps[g]
                    .as_ref()
                    .expect("only a planar group has tap ranges"),
                s,
            ),
        }
    }

    fn upload_row(&mut self, s: &Arc<CudaStream>, slot: usize, group: usize) -> Result<()> {
        // Block ids are at most `num_blocks - 1`, which `validate` holds
        // within `i32` (the device tables' type).
        let row: Vec<i32> = self.mirror.tables[slot][group]
            .iter()
            .map(|&b| i32::try_from(b).expect("validated: block ids fit i32"))
            .collect();
        let off = (slot * self.geometry.len() + group) * self.max_blocks;
        let mut view = self.tables_dev.slice_mut(off..off + self.max_blocks);
        s.memcpy_htod(&row, &mut view)?;
        Ok(())
    }

    /// The block holding `pos` for `slot` in `group`; panics if unmapped.
    pub fn block_of(&self, slot: Slot, group: usize, pos: u32) -> u32 {
        self.mirror.block_of(slot, group, pos)
    }

    /// The block a write at `pos` lands in; panics unless exactly one table
    /// entry maps it.
    pub fn writable_block(&self, slot: Slot, group: usize, pos: u32) -> u32 {
        self.mirror.writable_block(slot, group, pos)
    }

    /// Copy one block of a group back to the host (tests and diagnostics):
    /// its pieces in [`GroupGeometry::block_ranges`] order, a tap's f32
    /// words as pairs of 16-bit halves.
    pub fn read_block(&self, gpu: &Gpu, group: usize, block: u32) -> Result<Vec<u16>> {
        let s = gpu.stream();
        let mut out = Vec::with_capacity(self.geometry[group].block_bytes() / 2);
        for r in self.geometry[group].block_ranges(block) {
            match r {
                BlockRange::Pool { offset, elems } => {
                    out.extend(s.clone_dtoh(&self.pools[group].slice(offset..offset + elems))?);
                }
                BlockRange::Taps { offset, elems } => {
                    let taps = self.taps[group].as_ref().expect("a planar group");
                    for x in s.clone_dtoh(&taps.slice(offset..offset + elems))? {
                        let b = x.to_bits();
                        out.extend([(b & 0xffff) as u16, (b >> 16) as u16]);
                    }
                }
            }
        }
        Ok(out)
    }

    /// Overwrite one block from the host (tests), in [`KvStore::read_block`]'s
    /// order.
    pub fn write_block(&mut self, gpu: &Gpu, group: usize, block: u32, data: &[u16]) -> Result<()> {
        assert_eq!(data.len() * 2, self.geometry[group].block_bytes());
        let s = gpu.stream().clone();
        let mut at = 0;
        for r in self.geometry[group].block_ranges(block) {
            match r {
                BlockRange::Pool { offset, elems } => {
                    let mut view = self.pools[group].slice_mut(offset..offset + elems);
                    s.memcpy_htod(&data[at..at + elems], &mut view)?;
                    at += elems;
                }
                BlockRange::Taps { offset, elems } => {
                    let words: Vec<f32> = data[at..at + 2 * elems]
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|h| f32::from_bits(u32::from(h[0]) | (u32::from(h[1]) << 16)))
                        .collect();
                    let taps = self.taps[group].as_mut().expect("a planar group");
                    let mut view = taps.slice_mut(offset..offset + elems);
                    s.memcpy_htod(&words, &mut view)?;
                    at += 2 * elems;
                }
            }
        }
        Ok(())
    }

    /// The device copy of one table row (tests and diagnostics).
    pub fn read_table_row(&self, gpu: &Gpu, slot: Slot, group: usize) -> Result<Vec<i32>> {
        let off = (slot as usize * self.geometry.len() + group) * self.max_blocks;
        let view = self.tables_dev.slice(off..off + self.max_blocks);
        Ok(gpu.stream().clone_dtoh(&view)?)
    }

    /// One slot's model state (tests and diagnostics).
    pub fn read_state(&self, gpu: &Gpu, slot: Slot) -> Result<Vec<f32>> {
        let w = self.state_width;
        let view = self.state.slice(slot as usize * w..(slot as usize + 1) * w);
        Ok(gpu.stream().clone_dtoh(&view)?)
    }

    /// Overwrite one slot's model state (tests).
    pub fn write_state(&mut self, gpu: &Gpu, slot: Slot, data: &[f32]) -> Result<()> {
        let w = self.state_width;
        assert_eq!(data.len(), w);
        let mut view = self
            .state
            .slice_mut(slot as usize * w..(slot as usize + 1) * w);
        gpu.stream().memcpy_htod(data, &mut view)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom(block_size: u32, num_blocks: u32) -> GroupGeometry {
        GroupGeometry {
            num_layers: 9,
            num_kv_heads: 4,
            head_dim_qk: 192,
            head_dim_v: 128,
            block_size,
            num_blocks,
            layout: KvLayout::Blocked,
        }
    }

    /// Flash's drafter group: three MTP depths of 8 KV heads, a tap of three
    /// levels of 4,096 per block.
    fn drafter(block_size: u32, num_blocks: u32) -> GroupGeometry {
        GroupGeometry {
            num_layers: 3,
            num_kv_heads: 8,
            layout: KvLayout::Planar {
                taps: 3,
                hidden: 4096,
            },
            ..geom(block_size, num_blocks)
        }
    }

    /// A drafter block is its rows in every plane and its tap: disjoint,
    /// inside the pool, adding up to its bytes, and every block's ranges
    /// disjoint from every other block's.
    #[test]
    fn planar_blocks_are_their_plane_rows_and_their_tap() {
        let g = drafter(16, 7);
        g.validate().unwrap();
        assert_eq!(g.pool_rows(), 8 * 16);
        assert_eq!(g.k_row_elems(), 8 * 192);
        assert_eq!(g.v_row_elems(), 8 * 128);
        // Per block: 3 layers of 16 positions of 8 x 320 BF16, and the tap.
        assert_eq!(g.block_bytes(), 3 * 16 * 8 * 320 * 2 + 3 * 4096 * 4);
        let pool_elems = g.block_elems() * g.pool_blocks();
        let taps_elems = g.tap_elems() * g.pool_blocks();
        let mut pool = vec![u32::MAX; pool_elems];
        let mut taps = vec![u32::MAX; taps_elems];
        for b in 0..u32::try_from(g.pool_blocks()).unwrap() {
            let ranges = g.block_ranges(b);
            assert_eq!(ranges.len(), 2 * 3 + 1);
            let bytes: usize = ranges.iter().map(|r| r.bytes().1).sum();
            assert_eq!(bytes, g.block_bytes());
            for r in ranges {
                let (mem, offset, elems) = match r {
                    BlockRange::Pool { offset, elems } => (&mut pool, offset, elems),
                    BlockRange::Taps { offset, elems } => (&mut taps, offset, elems),
                };
                for x in &mut mem[offset..offset + elems] {
                    assert_eq!(*x, u32::MAX, "block {b} overlaps another range");
                    *x = b;
                }
            }
        }
        assert!(pool.iter().chain(&taps).all(|&x| x != u32::MAX));
        // Position `o` of block `b`, layer `l`: K row `b * 16 + o` of plane `l`.
        for (l, b, o) in [(0u32, 0usize, 0usize), (1, 3, 5), (2, 7, 15)] {
            let row = b * 16 + o;
            let k = g.plane_k_offset(l) + row * g.k_row_elems();
            let v = g.plane_v_offset(l) + row * g.v_row_elems();
            let b = u32::try_from(b).unwrap();
            assert_eq!(pool[k], b);
            assert_eq!(pool[k + g.k_row_elems() - 1], b);
            assert_eq!(pool[v], b);
        }
        // Position rows are i32 page ids.
        assert!(drafter(16, (i32::MAX as u32) / 16).validate().is_err());
        assert!(drafter(16, (i32::MAX as u32) / 16 - 1).validate().is_ok());
        let no_tap = GroupGeometry {
            layout: KvLayout::Planar {
                taps: 0,
                hidden: 4096,
            },
            ..drafter(16, 7)
        };
        assert!(no_tap.validate().is_err());
    }

    /// Zero and copy of a drafter block touch exactly its plane rows and its
    /// tap.
    #[test]
    fn planar_maintenance_covers_the_whole_block() {
        let geometry = vec![geom(16, 8), drafter(16, 8)];
        let current = TableMirror::new(&geometry, 2, 4);
        let staged = stage(
            &geometry,
            4,
            0,
            &current,
            0,
            &[
                Maintenance::Zero { group: 1, block: 3 },
                Maintenance::Copy {
                    group: 1,
                    src: 2,
                    dst: 5,
                },
                Maintenance::Zero { group: 0, block: 3 },
            ],
            &[],
        );
        let g = &geometry[1];
        let zeros: Vec<(Mem, usize, usize)> = staged
            .ops
            .iter()
            .filter_map(|op| match *op {
                DeviceOp::Zero { mem, offset, bytes } => Some((mem, offset, bytes)),
                _ => None,
            })
            .collect();
        let want: Vec<(Mem, usize, usize)> = g
            .block_ranges(3)
            .into_iter()
            .map(|r| {
                let (o, b) = r.bytes();
                (mem_of(1, r), o, b)
            })
            .chain([(
                Mem::Pool(0),
                3 * geometry[0].block_bytes(),
                geometry[0].block_bytes(),
            )])
            .collect();
        assert_eq!(zeros, want);
        assert!(zeros.iter().any(|z| z.0 == Mem::Taps(1)));
        let copies: Vec<(Mem, usize, usize, usize)> = staged
            .ops
            .iter()
            .filter_map(|op| match *op {
                DeviceOp::Copy {
                    mem,
                    src,
                    dst,
                    bytes,
                } => Some((mem, src, dst, bytes)),
                _ => None,
            })
            .collect();
        assert_eq!(copies.len(), 7);
        for ((mem, src, dst, bytes), (a, b)) in copies
            .into_iter()
            .zip(g.block_ranges(2).into_iter().zip(g.block_ranges(5)))
        {
            assert_eq!((mem, src, bytes), (mem_of(1, a), a.bytes().0, a.bytes().1));
            assert_eq!(dst, b.bytes().0);
        }
    }

    /// Staging computes the tables a step leads to without touching the
    /// current ones; a malformed step panics in staging, leaving them (and,
    /// since nothing was queued, the device) as they were.
    #[test]
    fn staging_validates_before_anything_changes() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let geometry = vec![geom(16, 8), geom(16, 8)];
        let current = TableMirror::new(&geometry, 2, 4);
        let before = current.clone();
        let update = |slot, group, index, block| TableUpdate {
            slot,
            group,
            index,
            block,
        };
        let ok = stage(
            &geometry,
            4,
            0,
            &current,
            7,
            &[Maintenance::Zero { group: 1, block: 3 }],
            &[update(0, 0, 0, 5), update(1, 0, 0, 5), update(0, 1, 1, 2)],
        );
        assert_eq!(current, before, "staging changed the current tables");
        assert_eq!(ok.generation, 7);
        assert_eq!(ok.mirror.table(0, 0), &[5, 0, 0, 0]);
        assert_eq!(ok.mirror.block_of(0, 1, 16), 2);
        assert_eq!(ok.mirror.writable_block(0, 1, 17), 2);
        assert_eq!(ok.dirty, vec![(0, 0), (0, 1), (1, 0)]);
        assert_eq!(ok.ops.len(), 1);
        // Block 5 is mapped twice: a write into it is refused on the staged
        // tables.
        let shared = catch_unwind(AssertUnwindSafe(|| ok.mirror.writable_block(0, 0, 3)));
        assert!(shared.is_err());
        // Each malformed piece, after valid ones, panics in staging.
        let malformed: [(&[Maintenance], &[TableUpdate]); 6] = [
            (&[Maintenance::Zero { group: 2, block: 1 }], &[]),
            (
                &[Maintenance::Copy {
                    group: 0,
                    src: 1,
                    dst: 8,
                }],
                &[],
            ),
            (&[Maintenance::ResetSlot { slot: 2 }], &[]),
            (&[], &[update(0, 0, 0, 1), update(0, 0, 4, 1)]),
            (&[], &[update(0, 0, 0, 1), update(0, 0, 0, 8)]),
            (
                &[Maintenance::Zero { group: 0, block: 1 }],
                &[update(2, 0, 0, 1)],
            ),
        ];
        for (m, u) in malformed {
            let r = catch_unwind(AssertUnwindSafe(|| {
                stage(&geometry, 4, 0, &current, 7, m, u)
            }));
            assert!(r.is_err(), "{m:?} {u:?}");
            assert_eq!(current, before);
        }
    }

    /// Sizes that would wrap (or exceed the 32-bit page stride, or the i32
    /// tables) are refused, not allocated short.
    #[test]
    fn overflowing_geometries_are_refused() {
        geom(16, 4096).validate().unwrap();
        // 16,777,217 positions wrap a u32 product of 4 heads x 64: per layer
        // it would allocate one token.
        assert!(geom(16_777_217, 4).validate().is_err());
        assert!(geom(u32::MAX, u32::MAX).validate().is_err());
        assert!(
            geom(16, u32::MAX).validate().is_err(),
            "past the i32 tables"
        );
        assert!(
            geom(16, 1).validate().is_err(),
            "no block besides the null one"
        );
        assert!(geom(0, 4).validate().is_err(), "empty blocks");
    }
}
