//! Physical KV memory in the model's packed formats: one pool per KV group.
//!
//! A block holds `layers · entries` **cells** of `cell_bytes` bytes each, layer-major
//! (`cell = layer · entries + entry`), where `entries = block_size / ratio`. A cell is the
//! packed KV of one entry of one layer (the group says which packed form), padded with
//! zero bytes to a whole number of 32-bit words. A block's bytes are contiguous, so
//! zeroing, copying and encrypting it are each one range.
//!
//! Beside the bytes the pool keeps a tag per cell: the position that completed the entry
//! (plus one; 0 means never written since the last zero) and that position's token.
//! Tags are instrumentation: every read checks that the cell was written for the position
//! it expects, so a wrong table, a premature free, a missing window, an unexecuted zero,
//! or a chain layer that never computed that position panics instead of attending to
//! the wrong keys.
//!
//! **A block's image** (what [`BytePool::crypt`] XORs with the keystream, what a
//! transfer to or from the host tier copies, what a zero clears) is two ranges, each
//! from a fresh keystream block: the cells' bytes as little-endian words in cell order,
//! then the tags (`pos + 1`, then the token, per cell).

use eidola_engine::kv_cipher::{KvKeystream, KvNonce};
use eidola_engine::secret::KvKey;

use crate::pool::{PageBuf, Tag};

#[derive(Debug)]
pub(crate) struct BytePool {
    /// Words per cell.
    cell_words: usize,
    cells_per_block: usize,
    num_blocks: usize,
    data: PageBuf<u32>,
    tags: PageBuf<Tag>,
    encrypted: Vec<bool>,
}

impl BytePool {
    pub fn new(cell_bytes: usize, cells_per_block: usize, num_blocks: usize) -> Self {
        let cell_words = cell_bytes.div_ceil(4);
        Self {
            data: PageBuf::zeroed(num_blocks * cells_per_block * cell_words),
            tags: PageBuf::zeroed(num_blocks * cells_per_block),
            encrypted: vec![false; num_blocks],
            cell_words,
            cells_per_block,
            num_blocks,
        }
    }

    /// A pool of the same shape with `num_blocks` blocks.
    pub fn like(&self, num_blocks: usize) -> Self {
        Self::new(self.cell_words * 4, self.cells_per_block, num_blocks)
    }

    /// Locks the pool's memory so it cannot be paged out; dropping it unlocks.
    pub fn lock(&mut self) -> std::io::Result<()> {
        let locked = self.data.lock().and_then(|()| self.tags.lock());
        if locked.is_err() {
            self.data.unlock();
            self.tags.unlock();
        }
        locked
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Bytes of one block's cells (without instrumentation).
    pub fn block_bytes(&self) -> usize {
        self.cells_per_block * self.cell_words * 4
    }

    fn words(&self, block: u32) -> std::ops::Range<usize> {
        let n = self.cells_per_block * self.cell_words;
        block as usize * n..(block as usize + 1) * n
    }

    fn cells(&self, block: u32) -> std::ops::Range<usize> {
        block as usize * self.cells_per_block..(block as usize + 1) * self.cells_per_block
    }

    pub fn write(&mut self, block: u32, cell: usize, tag: Tag, bytes: &[u8]) {
        assert!(cell < self.cells_per_block && bytes.len() <= self.cell_words * 4);
        let at = block as usize * self.cells_per_block + cell;
        let words = &mut self.data[at * self.cell_words..(at + 1) * self.cell_words];
        words.fill(0);
        for (i, b) in bytes.iter().enumerate() {
            words[i / 4] |= u32::from(*b) << (8 * (i % 4));
        }
        self.tags[at] = tag;
    }

    /// A cell's bytes and tag.
    pub fn read(&self, block: u32, cell: usize) -> (Vec<u8>, Tag) {
        let at = block as usize * self.cells_per_block + cell;
        let words = &self.data[at * self.cell_words..(at + 1) * self.cell_words];
        (
            words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            self.tags[at],
        )
    }

    pub fn tag(&self, block: u32, cell: usize) -> Tag {
        self.tags[block as usize * self.cells_per_block + cell]
    }

    pub fn zero(&mut self, block: u32) {
        let (w, c) = (self.words(block), self.cells(block));
        self.data[w].fill(0);
        self.tags[c].fill(Tag::default());
        self.encrypted[block as usize] = false;
    }

    pub fn copy(&mut self, src: u32, dst: u32) {
        let (w, c) = (self.words(src), self.cells(src));
        let (to_w, to_c) = (self.words(dst).start, self.cells(dst).start);
        self.data.copy_within(w, to_w);
        self.tags.copy_within(c, to_c);
        self.encrypted[dst as usize] = self.encrypted[src as usize];
    }

    /// Copies block `src` of `from` over block `dst` of `to`: its whole image and the
    /// encrypted flag.
    pub fn transfer(from: &BytePool, src: u32, to: &mut BytePool, dst: u32) {
        assert_eq!(
            (from.cell_words, from.cells_per_block),
            (to.cell_words, to.cells_per_block),
            "transfer between pools of one shape"
        );
        let (w, c) = (to.words(dst), to.cells(dst));
        to.data[w].copy_from_slice(&from.data[from.words(src)]);
        to.tags[c].copy_from_slice(&from.tags[from.cells(src)]);
        to.encrypted[dst as usize] = from.encrypted[src as usize];
    }

    /// Encrypts or decrypts the block in place: one keystream XOR over its image. Panics
    /// on encrypting an encrypted block or decrypting a plaintext one.
    pub fn crypt(&mut self, block: u32, key: &KvKey, nonce: &KvNonce, encrypt: bool) {
        let b = block as usize;
        assert_ne!(
            self.encrypted[b],
            encrypt,
            "block {block}: {} twice",
            if encrypt { "encrypted" } else { "decrypted" }
        );
        self.encrypted[b] = encrypt;
        let (w, c) = (self.words(block), self.cells(block));
        let mut ks = KvKeystream::new(key, nonce);
        ks.apply_words(&mut self.data[w]);
        let tags = &mut self.tags[c];
        ks.range(2 * tags.len(), |i, k| {
            let t = &mut tags[i / 2];
            if i % 2 == 0 {
                t.pos_plus_one ^= k;
            } else {
                t.token ^= k;
            }
        });
    }

    pub fn is_encrypted(&self, block: u32) -> bool {
        self.encrypted[block as usize]
    }

    /// The block's image as words, in [`BytePool::crypt`]'s order.
    pub fn image(&self, block: u32) -> Vec<u32> {
        let mut out = self.data[self.words(block)].to_vec();
        for t in &self.tags[self.cells(block)] {
            out.extend([t.pos_plus_one, t.token]);
        }
        out
    }

    /// Whether every byte of the block and every tag is zero.
    pub fn block_is_zero(&self, block: u32) -> bool {
        self.data[self.words(block)].iter().all(|w| *w == 0)
            && self.tags[self.cells(block)]
                .iter()
                .all(|t| *t == Tag::default())
    }
}
