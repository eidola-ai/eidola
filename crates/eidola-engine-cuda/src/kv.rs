//! Device KV memory: one pool per KV group, the per-slot block tables, per-slot
//! model state, and the seam's maintenance operations on them.
//!
//! A group's pool is `num_blocks` blocks; block `b` holds, for each of the
//! group's layers in order, K as `[block_size, kv_heads, head_dim_qk]` then V as
//! `[block_size, kv_heads, head_dim_v]`, BF16, contiguous. A block is therefore
//! one contiguous byte range across every layer: `Zero` is one memset and `Copy`
//! one device-to-device copy. Paged attention addresses a layer's K (or V) with
//! page stride = the whole block and the layer's offset inside it.
//!
//! Block tables live on the device as `[slot][group][index]` `i32` rows (the
//! attention kernels' index type) and change only through table updates and
//! `ResetSlot`. The host keeps an exact mirror, which is how it plans attention
//! and checks the contract: a write into a block mapped by more than one table
//! entry (a shared prefix-cache block) or a read through an unmapped entry is a
//! host bug and panics, as in the CPU reference executor.

use std::sync::Arc;

use cudarc::driver::sys;
use cudarc::driver::{CudaSlice, CudaStream};
use eidola_engine::executor::{Maintenance, Slot, TableUpdate};
use eidola_engine::spec::NULL_BLOCK;

use crate::launch::dptr;
use crate::{CudaError, Gpu, Result};

/// The shape of one KV group's pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupGeometry {
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim_qk: u32,
    pub head_dim_v: u32,
    pub block_size: u32,
    pub num_blocks: u32,
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
        m(m(block, self.num_blocks)?, 2)?;
        if self.num_blocks < 2 || self.num_blocks > i32::MAX as u32 {
            return Err(CudaError::new(format!(
                "KV geometry {self:?}: blocks must be 2..=i32::MAX (the null block and one more; i32 tables)"
            )));
        }
        Ok(())
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

    pub fn block_bytes(&self) -> usize {
        self.block_elems() * 2
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

/// KV pools, block tables and per-slot state.
pub struct KvStore {
    geometry: Vec<GroupGeometry>,
    pools: Vec<CudaSlice<u16>>,
    /// Host mirror, `[slot][group][index]`.
    tables: Vec<Vec<Vec<u32>>>,
    /// `[group][block]`: number of table entries mapping it.
    mapped: Vec<Vec<u32>>,
    /// Device tables, `[slot][group][index]`.
    tables_dev: CudaSlice<i32>,
    max_blocks: usize,
    /// Per-slot model state, `[slot][state_width]` f32 (drafter hidden states;
    /// empty when nothing is drafted).
    state: CudaSlice<f32>,
    state_width: usize,
}

impl KvStore {
    /// Allocate every pool (zeroed), `num_slots` empty tables of `max_blocks`
    /// entries per group, and `state_width` floats of zeroed state per slot.
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
        for g in &geometry {
            pools.push(s.alloc_zeros::<u16>(g.block_elems() * g.num_blocks as usize)?);
        }
        Ok(KvStore {
            tables: vec![vec![vec![NULL_BLOCK; mb]; groups]; slots],
            mapped: geometry
                .iter()
                .map(|g| vec![0; g.num_blocks as usize])
                .collect(),
            tables_dev: s.alloc_zeros::<i32>(table_entries.max(1))?,
            state: s.alloc_zeros::<f32>(state_len.max(1))?,
            max_blocks: mb,
            state_width,
            geometry,
            pools,
        })
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

    /// The host mirror of one table row.
    pub fn table(&self, slot: Slot, group: usize) -> &[u32] {
        &self.tables[slot as usize][group]
    }

    /// Apply a step's maintenance, then its table updates, in order (the seam's
    /// steps 1 and 2). Device work is queued on the stream ahead of the forward.
    pub fn apply(
        &mut self,
        gpu: &Gpu,
        maintenance: &[Maintenance],
        updates: &[TableUpdate],
    ) -> Result<()> {
        let s = gpu.stream();
        // The raw driver calls below need this thread's current context.
        gpu.context().bind_to_thread()?;
        let mut dirty: Vec<(usize, usize)> = Vec::new();
        for m in maintenance {
            match *m {
                Maintenance::Zero { group, block } => {
                    let (g, off, bytes) = self.block_range(group, block);
                    let base = dptr(&self.pools[g], s) + off as u64;
                    // SAFETY: the range lies inside the pool.
                    unsafe { sys::cuMemsetD8Async(base, 0, bytes, s.cu_stream()) }.result()?;
                }
                Maintenance::Copy { group, src, dst } => {
                    let (g, src_off, bytes) = self.block_range(group, src);
                    let (_, dst_off, _) = self.block_range(group, dst);
                    let base = dptr(&self.pools[g], s);
                    // SAFETY: both ranges lie inside the pool; distinct blocks
                    // do not overlap.
                    unsafe {
                        sys::cuMemcpyDtoDAsync_v2(
                            base + dst_off as u64,
                            base + src_off as u64,
                            bytes,
                            s.cu_stream(),
                        )
                    }
                    .result()?;
                }
                Maintenance::ResetSlot { slot } => {
                    let sl = slot as usize;
                    assert!(sl < self.tables.len(), "reset of slot {slot} out of range");
                    for g in 0..self.geometry.len() {
                        for idx in 0..self.max_blocks {
                            self.set(sl, g, idx, NULL_BLOCK);
                        }
                        dirty.push((sl, g));
                    }
                    if self.state_width > 0 {
                        let w = self.state_width;
                        let base = dptr(&self.state, s) + (sl * w * 4) as u64;
                        // SAFETY: inside the state buffer.
                        unsafe { sys::cuMemsetD8Async(base, 0, w * 4, s.cu_stream()) }.result()?;
                    }
                }
            }
        }
        for u in updates {
            let (sl, g) = (u.slot as usize, u.group as usize);
            assert!(
                sl < self.tables.len(),
                "table update for slot {} out of range",
                u.slot
            );
            assert!(
                g < self.geometry.len(),
                "table update for group {} out of range",
                u.group
            );
            assert!(
                u.block < self.geometry[g].num_blocks,
                "table update to block {} outside group {}",
                u.block,
                u.group
            );
            assert!(
                (u.index as usize) < self.max_blocks,
                "table index {} past the longest sequence",
                u.index
            );
            self.set(sl, g, u.index as usize, u.block);
            dirty.push((sl, g));
        }
        dirty.sort_unstable();
        dirty.dedup();
        for (sl, g) in dirty {
            self.upload_row(s, sl, g)?;
        }
        Ok(())
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

    fn upload_row(&mut self, s: &Arc<CudaStream>, slot: usize, group: usize) -> Result<()> {
        // Block ids are at most `num_blocks - 1`, which `validate` holds
        // within `i32` (the device tables' type).
        let row: Vec<i32> = self.tables[slot][group]
            .iter()
            .map(|&b| i32::try_from(b).expect("validated: block ids fit i32"))
            .collect();
        let off = (slot * self.geometry.len() + group) * self.max_blocks;
        let mut view = self.tables_dev.slice_mut(off..off + self.max_blocks);
        s.memcpy_htod(&row, &mut view)?;
        Ok(())
    }

    /// `(group, byte offset, bytes)` of a block; panics on an id outside the pool.
    fn block_range(&self, group: u32, block: u32) -> (usize, usize, usize) {
        let g = group as usize;
        let geom = &self.geometry[g];
        assert!(
            block < geom.num_blocks,
            "maintenance on block {block} outside group {group}"
        );
        let bytes = geom.block_bytes();
        (g, block as usize * bytes, bytes)
    }

    /// The block holding `pos` for `slot` in `group`; panics if unmapped (a
    /// read the host never provided for).
    pub fn block_of(&self, slot: Slot, group: usize, pos: u32) -> u32 {
        let bs = self.geometry[group].block_size;
        let block = self.tables[slot as usize][group][(pos / bs) as usize];
        assert!(
            block != NULL_BLOCK,
            "slot {slot} group {group} position {pos}: unmapped block"
        );
        block
    }

    /// The block a write at `pos` lands in, which must be mapped by exactly one
    /// table entry: blocks shared through the prefix cache are immutable.
    pub fn writable_block(&self, slot: Slot, group: usize, pos: u32) -> u32 {
        let block = self.block_of(slot, group, pos);
        assert!(
            self.mapped[group][block as usize] == 1,
            "slot {slot} group {group} position {pos}: write to shared block {block}"
        );
        block
    }

    /// Copy one block of a group back to the host (tests and diagnostics).
    pub fn read_block(&self, gpu: &Gpu, group: usize, block: u32) -> Result<Vec<u16>> {
        let e = self.geometry[group].block_elems();
        let off = block as usize * e;
        let view = self.pools[group].slice(off..off + e);
        Ok(gpu.stream().clone_dtoh(&view)?)
    }

    /// Overwrite one block from the host (tests).
    pub fn write_block(&mut self, gpu: &Gpu, group: usize, block: u32, data: &[u16]) -> Result<()> {
        let e = self.geometry[group].block_elems();
        assert_eq!(data.len(), e);
        let off = block as usize * e;
        let mut view = self.pools[group].slice_mut(off..off + e);
        gpu.stream().memcpy_htod(data, &mut view)?;
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
