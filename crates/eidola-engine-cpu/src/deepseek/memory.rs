//! One logical device's KV memory: its groups' pools, host-tier pools and per-slot block
//! tables. A device holds only its own groups, so it has no way to name another's.

use eidola_engine::executor::Slot;
use eidola_engine::spec::NULL_BLOCK;

use super::pool::BytePool;
use crate::pool::Tag;

/// One KV group on a device.
#[derive(Debug)]
pub(crate) struct GroupMem {
    /// The group's index in the model spec.
    pub global: u32,
    /// Positions per entry.
    pub ratio: u32,
    pub pool: BytePool,
    pub host: Option<BytePool>,
    /// `[slot][index]`.
    tables: Vec<Vec<u32>>,
    /// Per block: `(slot, index)` mappings.
    mapped: Vec<u32>,
}

impl GroupMem {
    pub fn new(
        global: u32,
        ratio: u32,
        pool: BytePool,
        host: Option<BytePool>,
        slots: usize,
        width: usize,
    ) -> Self {
        Self {
            tables: vec![vec![NULL_BLOCK; width]; slots],
            mapped: vec![0; pool.num_blocks()],
            global,
            ratio,
            pool,
            host,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Memory {
    pub block_size: u32,
    pub groups: Vec<GroupMem>,
}

impl Memory {
    /// The device's group with spec index `global`, if it is one of this device's.
    pub fn local(&self, global: u32) -> Option<usize> {
        self.groups.iter().position(|g| g.global == global)
    }

    pub fn set_table(&mut self, g: usize, slot: Slot, index: u32, block: u32) {
        let gm = &mut self.groups[g];
        assert!(
            (block as usize) < gm.pool.num_blocks(),
            "table update to block {block} outside group {}",
            gm.global
        );
        let old = std::mem::replace(&mut gm.tables[slot as usize][index as usize], block);
        if old != NULL_BLOCK {
            gm.mapped[old as usize] -= 1;
        }
        if block != NULL_BLOCK {
            gm.mapped[block as usize] += 1;
        }
    }

    pub fn reset_slot(&mut self, slot: Slot) {
        for g in 0..self.groups.len() {
            for index in 0..self.groups[g].tables[slot as usize].len() {
                if self.groups[g].tables[slot as usize][index] != NULL_BLOCK {
                    self.set_table(g, slot, index as u32, NULL_BLOCK);
                }
            }
        }
    }

    /// The block and cell of `layer`'s entry completed at `pos`.
    fn locate(&self, slot: Slot, g: usize, layer: usize, pos: u32) -> (u32, usize) {
        let gm = &self.groups[g];
        let first = pos + 1 - gm.ratio;
        let block = gm.tables[slot as usize][(first / self.block_size) as usize];
        assert!(
            block != NULL_BLOCK,
            "slot {slot} group {} position {pos}: unmapped block",
            gm.global
        );
        let entries = (self.block_size / gm.ratio) as usize;
        let entry = (first % self.block_size / gm.ratio) as usize;
        (block, layer * entries + entry)
    }

    /// Stores `layer`'s entry completed at `pos`. The block must be mapped by exactly
    /// this one table entry: blocks shared through the prefix cache are immutable.
    pub fn write(
        &mut self,
        slot: Slot,
        g: usize,
        layer: usize,
        pos: u32,
        token: u32,
        bytes: &[u8],
    ) {
        let (block, cell) = self.locate(slot, g, layer, pos);
        let gm = &mut self.groups[g];
        assert!(
            gm.mapped[block as usize] == 1,
            "slot {slot} group {} position {pos}: write to shared block {block}",
            gm.global
        );
        let tag = Tag {
            pos_plus_one: pos + 1,
            token,
        };
        gm.pool.write(block, cell, tag, bytes);
    }

    /// `layer`'s entry completed at `pos`: its bytes. Panics unless that layer wrote it
    /// for that position.
    pub fn read(&self, slot: Slot, g: usize, layer: usize, pos: u32) -> Vec<u8> {
        let (block, cell) = self.locate(slot, g, layer, pos);
        let (bytes, tag) = self.groups[g].pool.read(block, cell);
        assert_eq!(
            tag.pos_plus_one,
            pos + 1,
            "slot {slot} group {} layer {layer} position {pos}: read of KV that was never \
             computed, zeroed, encrypted or foreign",
            self.groups[g].global
        );
        bytes
    }

    /// The token tagged on `layer` 0's entry completed at `pos`.
    pub fn token_at(&self, slot: Slot, g: usize, pos: u32) -> u32 {
        let (block, cell) = self.locate(slot, g, 0, pos);
        let tag = self.groups[g].pool.tag(block, cell);
        assert_eq!(
            tag.pos_plus_one,
            pos + 1,
            "slot {slot}: position {pos} holds foreign or zeroed KV"
        );
        tag.token
    }
}
