//! One half's KV memory: a pool and per-slot block tables for each of its groups.
//!
//! **A block's image** is the kernel ABI's (`csrc/ds_abi.cuh`): per layer of the group,
//! planes of packed entries back to back, then zero bytes up to a multiple of 16. A
//! window group's layer is codes `[E, width]` then scales `[E, width / 32]`; a bank
//! group's is KV codes `[E, width / 2]`, KV scales `[E, width / 16]`, index-key codes
//! `[E, index / 2]`, index-key scales `[E, index / 32]`, with `E` the block's positions
//! divided by the group's ratio. That whole range is what `Zero` clears, `Copy` moves and
//! the idle cipher XORs, as little-endian words from keystream block 0.
//!
//! The CPU reference executor stores the same bytes cell by cell (an entry's codes then
//! its scales, padded to a word). [`GroupGeometry::cells`] is that permutation, so a
//! device block is compared with the reference's byte for byte; ciphertext is not
//! comparable, because the two images order the keystream differently.
//!
//! Tables live on the device for the kernels that walk a bank (`[slots, stride]` 32-bit
//! words, -1 where nothing is mapped, so a kernel reading through one raises its status
//! bit) and change only through the seam; the host mirror plans every write and read.

use std::collections::BTreeSet;

use eidola_engine::executor::Slot;
use eidola_engine::kv_cipher::KvNonce;
use eidola_engine::secret::KvKey;
use eidola_engine::spec::NULL_BLOCK;

use super::abi::{KV_FP4_E4M3, KV_FP4_E8M0, KV_FP8, KvPlane};
use super::half::{Buffer, Device, Kind};
use crate::{CudaError, Result, narrow};

/// What a group's entries are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    /// Window KV: FP8 codes with a power-of-two scale per 32.
    Window,
    /// A global entry: the FP4 KV with an E4M3 scale per 16, then the FP4 index key
    /// with a power-of-two scale per 32.
    Bank,
}

/// A plane of a group's layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plane {
    Window,
    BankKv,
    BankKey,
}

/// One group's block layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupGeometry {
    /// The group's index in the seam's `ModelSpec`.
    pub global: u32,
    pub cell: Cell,
    pub layers: u32,
    pub ratio: u32,
    /// Positions per block.
    pub block_size: u32,
    /// Elements of a KV entry.
    pub width: u32,
    /// Elements of an index key (bank groups).
    pub index_width: u32,
    /// Blocks, the null block included.
    pub num_blocks: u32,
}

impl GroupGeometry {
    /// Entries of one layer in a block.
    pub fn entries(&self) -> u32 {
        self.block_size / self.ratio
    }

    /// Bytes of one entry: the reference executor's cell before its padding.
    pub fn cell_bytes(&self) -> usize {
        let (w, i) = (self.width as usize, self.index_width as usize);
        match self.cell {
            Cell::Window => w + w / 32,
            Cell::Bank => w / 2 + w / 16 + i / 2 + i / 32,
        }
    }

    /// The entry's sections, in cell order: bytes of each.
    fn sections(&self) -> Vec<usize> {
        let (w, i) = (self.width as usize, self.index_width as usize);
        match self.cell {
            Cell::Window => vec![w, w / 32],
            Cell::Bank => vec![w / 2, w / 16, i / 2, i / 32],
        }
    }

    /// Bytes of one layer's planes in a block.
    pub fn layer_bytes(&self) -> usize {
        self.entries() as usize * self.cell_bytes()
    }

    /// Bytes of a block that hold entries.
    pub fn image_bytes(&self) -> usize {
        self.layers as usize * self.layer_bytes()
    }

    /// Bytes from one block to the next: the image, padded to 16.
    pub fn block_bytes(&self) -> usize {
        self.image_bytes().next_multiple_of(16)
    }

    pub fn pool_bytes(&self) -> usize {
        self.num_blocks as usize * self.block_bytes()
    }

    /// The plane of `layer` in a pool at `base`.
    pub fn plane(&self, base: u64, layer: u32, plane: Plane) -> KvPlane {
        assert!(layer < self.layers, "layer {layer} of {}", self.layers);
        let s = self.sections();
        let e = self.entries() as usize;
        let at = layer as usize * self.layer_bytes();
        let (first, format, width) = match (self.cell, plane) {
            (Cell::Window, Plane::Window) => (0, KV_FP8, self.width),
            (Cell::Bank, Plane::BankKv) => (0, KV_FP4_E4M3, self.width),
            (Cell::Bank, Plane::BankKey) => (2, KV_FP4_E8M0, self.index_width),
            _ => panic!("{plane:?} is not a plane of a {:?} group", self.cell),
        };
        let before: usize = s[..first].iter().sum();
        KvPlane {
            base,
            block_bytes: self.block_bytes() as u64,
            codes_offset: (at + e * before) as u64,
            scales_offset: (at + e * (before + s[first])) as u64,
            entries_per_block: self.entries(),
            format,
            width,
            reserved: 0,
        }
    }

    /// A block's bytes in the reference executor's order, as its image's words: each
    /// layer's entries in turn, an entry's sections back to back, padded with zero bytes
    /// to a whole word.
    pub fn cells(&self, block: &[u8]) -> Vec<u32> {
        assert_eq!(block.len(), self.block_bytes());
        let (s, e) = (self.sections(), self.entries() as usize);
        let cell_words = self.cell_bytes().div_ceil(4);
        let mut out = Vec::with_capacity(self.layers as usize * e * cell_words);
        let mut cell = vec![0u8; cell_words * 4];
        for layer in 0..self.layers as usize {
            let at = layer * self.layer_bytes();
            for o in 0..e {
                let (mut plane, mut filled) = (at, 0);
                for bytes in &s {
                    cell[filled..filled + bytes]
                        .copy_from_slice(&block[plane + o * bytes..plane + (o + 1) * bytes]);
                    filled += bytes;
                    plane += e * bytes;
                }
                out.extend(
                    cell.chunks_exact(4)
                        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])),
                );
            }
        }
        out
    }

    /// Where `count` consecutive entries from offset `first` of `layer` sit in a block:
    /// per section, the byte offset and the bytes of one entry.
    pub fn runs(&self, layer: u32, first: u32) -> Vec<(usize, usize)> {
        let e = self.entries() as usize;
        let mut plane = layer as usize * self.layer_bytes();
        self.sections()
            .into_iter()
            .map(|bytes| {
                let at = plane + first as usize * bytes;
                plane += e * bytes;
                (at, bytes)
            })
            .collect()
    }
}

/// A group's memory.
pub struct GroupStore {
    pub geom: GroupGeometry,
    pub pool: Buffer,
    /// `[slots, stride]` 32-bit words: the block of each logical index, or -1.
    pub tables: Buffer,
    mirror: Vec<u32>,
    /// Table entries naming each block.
    mapped: Vec<u32>,
    dirty: BTreeSet<Slot>,
    encrypted: Vec<bool>,
}

/// One half's groups.
pub struct KvStore {
    pub groups: Vec<GroupStore>,
    pub block_size: u32,
    pub slots: u32,
    /// Words per table: logical blocks of the longest sequence.
    pub stride: u32,
}

impl KvStore {
    pub fn new(
        dev: &dyn Device,
        geometry: Vec<GroupGeometry>,
        slots: u32,
        stride: u32,
    ) -> Result<KvStore> {
        let block_size = geometry.first().map_or(1, |g| g.block_size);
        let words = slots as usize * stride as usize;
        let mut groups = Vec::with_capacity(geometry.len());
        for geom in geometry {
            narrow::<i32, u32>(geom.num_blocks, "blocks of a group")?;
            let pool = Buffer::new(dev, "kv pool", Kind::Kv, geom.pool_bytes())?;
            let tables = Buffer::new(dev, "block tables", Kind::Tables, words * 4)?;
            tables.write(dev, 0, &vec![-1i32; words])?;
            groups.push(GroupStore {
                mirror: vec![NULL_BLOCK; words],
                mapped: vec![0; geom.num_blocks as usize],
                dirty: BTreeSet::new(),
                encrypted: vec![false; geom.num_blocks as usize],
                geom,
                pool,
                tables,
            });
        }
        Ok(KvStore {
            groups,
            block_size,
            slots,
            stride,
        })
    }

    /// This half's index of seam group `global`.
    pub fn local(&self, global: u32) -> Option<usize> {
        self.groups.iter().position(|g| g.geom.global == global)
    }

    fn word(&self, slot: Slot, index: u32) -> usize {
        assert!(
            slot < self.slots && index < self.stride,
            "table entry {slot}/{index} outside {} slots of {} blocks",
            self.slots,
            self.stride
        );
        slot as usize * self.stride as usize + index as usize
    }

    pub fn set_table(&mut self, g: usize, slot: Slot, index: u32, block: u32) {
        let at = self.word(slot, index);
        let gs = &mut self.groups[g];
        assert!(
            block < gs.geom.num_blocks,
            "table update to block {block} outside group {}",
            gs.geom.global
        );
        let old = std::mem::replace(&mut gs.mirror[at], block);
        if old != NULL_BLOCK {
            gs.mapped[old as usize] -= 1;
        }
        if block != NULL_BLOCK {
            gs.mapped[block as usize] += 1;
        }
        gs.dirty.insert(slot);
    }

    pub fn reset_slot(&mut self, slot: Slot) {
        for g in 0..self.groups.len() {
            for index in 0..self.stride {
                if self.groups[g].mirror[self.word(slot, index)] != NULL_BLOCK {
                    self.set_table(g, slot, index, NULL_BLOCK);
                }
            }
        }
    }

    /// Uploads the table rows the seam changed since the last call.
    pub fn flush_tables(&mut self, dev: &dyn Device) -> Result<()> {
        let stride = self.stride as usize;
        for gs in &mut self.groups {
            for slot in std::mem::take(&mut gs.dirty) {
                let at = slot as usize * stride;
                let row: Vec<i32> = gs.mirror[at..at + stride]
                    .iter()
                    .map(|b| match *b {
                        NULL_BLOCK => -1,
                        // Block ids fit `i32`: checked when the store was built.
                        b => i32::try_from(b).expect("block id"),
                    })
                    .collect();
                gs.tables.write(dev, at * 4, &row)?;
            }
        }
        Ok(())
    }

    fn block_range(&self, g: usize, block: u32) -> (u64, usize) {
        let gs = &self.groups[g];
        assert!(
            block < gs.geom.num_blocks,
            "block {block} outside group {}",
            gs.geom.global
        );
        let bytes = gs.geom.block_bytes();
        (gs.pool.at(block as usize * bytes, bytes), bytes)
    }

    pub fn zero(&mut self, dev: &dyn Device, g: usize, block: u32) -> Result<()> {
        let (addr, bytes) = self.block_range(g, block);
        self.groups[g].encrypted[block as usize] = false;
        // SAFETY: one block of the pool.
        unsafe { dev.zero(addr, bytes) }
    }

    pub fn copy(&mut self, dev: &dyn Device, g: usize, src: u32, dst: u32) -> Result<()> {
        if src == dst {
            return Ok(());
        }
        let ((from, bytes), (to, _)) = (self.block_range(g, src), self.block_range(g, dst));
        let gs = &mut self.groups[g];
        gs.encrypted[dst as usize] = gs.encrypted[src as usize];
        // SAFETY: two distinct blocks of the pool.
        unsafe { dev.copy(to, from, bytes) }
    }

    /// Encrypts or decrypts a block in place. Panics on encrypting an encrypted block or
    /// decrypting a plaintext one, as the reference executor does.
    pub fn crypt(
        &mut self,
        dev: &dyn Device,
        g: usize,
        block: u32,
        key: &KvKey,
        nonce: &KvNonce,
        encrypt: bool,
    ) -> Result<()> {
        let (addr, bytes) = self.block_range(g, block);
        let gs = &mut self.groups[g];
        assert_ne!(
            gs.encrypted[block as usize],
            encrypt,
            "group {} block {block}: {} twice",
            gs.geom.global,
            if encrypt { "encrypted" } else { "decrypted" }
        );
        gs.encrypted[block as usize] = encrypt;
        // SAFETY: one block of the pool, a whole number of words at a word boundary.
        unsafe { dev.cipher(addr, narrow(bytes / 4, "words of a block")?, key, nonce) }
    }

    pub fn is_encrypted(&self, g: usize, block: u32) -> bool {
        self.groups[g].encrypted[block as usize]
    }

    /// The block holding `pos` in `slot`'s sequence and the entry's offset in it.
    fn locate(&self, g: usize, slot: Slot, pos: u32) -> (u32, u32) {
        let geom = &self.groups[g].geom;
        let block = self.groups[g].mirror[self.word(slot, pos / self.block_size)];
        assert!(
            block != NULL_BLOCK,
            "slot {slot} group {} position {pos}: unmapped block",
            geom.global
        );
        assert!(
            !self.groups[g].encrypted[block as usize],
            "slot {slot} group {} position {pos}: block {block} is encrypted",
            geom.global
        );
        (block, pos % self.block_size / geom.ratio)
    }

    /// The entry index (`block · entries + offset`) holding position `pos`, for a read.
    /// Panics on an unmapped or encrypted block: a host bug.
    pub fn entry(&self, g: usize, slot: Slot, pos: u32) -> u32 {
        let (block, offset) = self.locate(g, slot, pos);
        block * self.groups[g].geom.entries() + offset
    }

    /// As [`KvStore::entry`], for a write: the block must be mapped exactly once.
    pub fn target(&self, g: usize, slot: Slot, pos: u32) -> u32 {
        let (block, offset) = self.locate(g, slot, pos);
        let gs = &self.groups[g];
        assert!(
            gs.mapped[block as usize] == 1,
            "slot {slot} group {} position {pos}: write to shared block {block}",
            gs.geom.global
        );
        block * gs.geom.entries() + offset
    }

    /// A block's bytes (tests and comparison with the reference executor).
    pub fn read_block(&self, dev: &dyn Device, g: usize, block: u32) -> Result<Vec<u8>> {
        let gs = &self.groups[g];
        let bytes = gs.geom.block_bytes();
        if block >= gs.geom.num_blocks {
            return Err(CudaError::new(format!("block {block} outside the pool")));
        }
        gs.pool.read(dev, block as usize * bytes, bytes)
    }

    /// Every pool's bytes (tests).
    pub fn read_pool(&self, dev: &dyn Device, g: usize) -> Result<Vec<u8>> {
        let gs = &self.groups[g];
        gs.pool.read(dev, 0, gs.geom.pool_bytes())
    }

    /// A slot's table as the device holds it (tests).
    pub fn read_table(&self, dev: &dyn Device, g: usize, slot: Slot) -> Result<Vec<i32>> {
        let stride = self.stride as usize;
        self.groups[g]
            .tables
            .read(dev, slot as usize * stride * 4, stride)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(layers: u32, block: u32, width: u32) -> GroupGeometry {
        GroupGeometry {
            global: 1,
            cell: Cell::Window,
            layers,
            ratio: 1,
            block_size: block,
            width,
            index_width: 0,
            num_blocks: 4,
        }
    }

    fn bank(layers: u32, ratio: u32, block: u32, width: u32, index: u32) -> GroupGeometry {
        GroupGeometry {
            global: 0,
            cell: Cell::Bank,
            layers,
            ratio,
            block_size: block,
            width,
            index_width: index,
            num_blocks: 4,
        }
    }

    /// The released sizes: 528 bytes of window KV a position, 356 of a global entry
    /// (288 of KV and 68 of index key), as the reference executor's cells.
    #[test]
    fn released_cells() {
        let w = window(20, 32, 512);
        assert_eq!(w.cell_bytes(), 528);
        assert_eq!(w.block_bytes(), 20 * 32 * 528);
        let b = bank(3, 2, 32, 512, 128);
        assert_eq!(b.cell_bytes(), 356);
        assert_eq!(b.block_bytes(), 3 * 16 * 356);
        let one = bank(1, 1, 32, 512, 128);
        assert_eq!(one.block_bytes(), 32 * 356);
    }

    #[test]
    fn planes_tile_a_layer_in_the_headers_order() {
        let w = window(3, 8, 64);
        let p = w.plane(1000, 2, Plane::Window);
        assert_eq!(p.base, 1000);
        assert_eq!(p.block_bytes, (3 * 8 * 66usize).next_multiple_of(16) as u64);
        assert_eq!((p.codes_offset, p.scales_offset), (2 * 528, 2 * 528 + 512));
        assert_eq!((p.entries_per_block, p.format, p.width), (8, KV_FP8, 64));

        let b = bank(2, 2, 8, 64, 32);
        // A layer: 4 entries of 32 + 4 + 16 + 1 bytes.
        assert_eq!(b.layer_bytes(), 4 * 53);
        let kv = b.plane(0, 1, Plane::BankKv);
        assert_eq!((kv.codes_offset, kv.scales_offset), (212, 212 + 128));
        assert_eq!(
            (kv.entries_per_block, kv.format, kv.width),
            (4, KV_FP4_E4M3, 64)
        );
        let key = b.plane(0, 1, Plane::BankKey);
        assert_eq!(
            (key.codes_offset, key.scales_offset),
            (212 + 144, 212 + 144 + 64)
        );
        assert_eq!((key.format, key.width), (KV_FP4_E8M0, 32));
        assert_eq!(
            b.runs(1, 3),
            vec![
                (212 + 96, 32),
                (212 + 128 + 12, 4),
                (212 + 144 + 48, 16),
                (212 + 208 + 3, 1)
            ]
        );
    }

    /// The permutation to the reference executor's cells: an entry's sections back to
    /// back, padded to a word, layer-major.
    #[test]
    fn cells_gather_an_entrys_sections() {
        let g = bank(2, 2, 4, 32, 32);
        // Sections of 16, 2, 16 and 1 bytes; 2 entries a layer; cells of 35 bytes.
        assert_eq!(g.cell_bytes(), 35);
        let mut block = vec![0u8; g.block_bytes()];
        for layer in 0..2u32 {
            for o in 0..2u32 {
                for (section, (at, bytes)) in g.runs(layer, o).into_iter().enumerate() {
                    for (i, b) in block[at..at + bytes].iter_mut().enumerate() {
                        assert_eq!(*b, 0, "sections do not overlap");
                        *b = u8::try_from(1 + (layer * 100 + o * 50) as usize + i).unwrap()
                            ^ u8::try_from(section << 6).unwrap();
                    }
                }
            }
        }
        assert!(block[..g.image_bytes()].iter().all(|b| *b != 0));
        let words = g.cells(&block);
        assert_eq!(words.len(), 2 * 2 * 9);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        for layer in 0..2u32 {
            for o in 0..2u32 {
                let cell = &bytes[(layer * 2 + o) as usize * 36..][..36];
                let mut at = 0;
                for (section, len) in [16usize, 2, 16, 1].into_iter().enumerate() {
                    for i in 0..len {
                        let want = u8::try_from(1 + (layer * 100 + o * 50) as usize + i).unwrap()
                            ^ u8::try_from(section << 6).unwrap();
                        assert_eq!(cell[at + i], want, "layer {layer} entry {o}");
                    }
                    at += len;
                }
                assert_eq!(cell[35], 0, "padding");
            }
        }
    }
}
