//! Physical KV memory: one pool per KV group, laid out by block.
//!
//! A block of a group holds, for each of the group's layers, `block_size` rows of
//! `[K (num_kv_heads · head_dim_qk) | V (num_kv_heads · head_dim_v)]`, contiguous, so
//! zeroing or copying a block is one contiguous range. A drafter group's block also holds
//! one *boundary tap*: the hidden states the drafter continues from, recorded at the
//! block's last position (see the crate documentation).
//!
//! Beside the data the pool keeps a tag per stored row (the position and token it was
//! written for) and per tap. Tags are instrumentation, not model state: every read checks
//! that the row it lands on was written for the position it expects, so a wrong block
//! table, a premature free, a missing window or an unexecuted zero panics instead of
//! silently attending to the wrong keys. Zero and copy treat tags like data.
//!
//! **Encryption** ([`Pool::crypt`]) XORs a block's whole image with the ChaCha20 keystream
//! of `eidola_engine::kv_cipher`, as four ranges in this order, each from a fresh keystream
//! block: the data (every layer's rows, `f32` bits), the tap (`f32` bits), the row tags
//! (`pos + 1` then token, per row) and the tap tag. Tags are encrypted too, so a read of a
//! block left encrypted fails its tag check like a read of a zeroed one, and an idle block
//! holds no token or position in the clear. The pool also records which blocks are
//! encrypted, as instrumentation beside the tags.

use eidola_engine::kv_cipher::{KvKeystream, KvNonce};
use eidola_engine::secret::KvKey;

/// Shape of one KV group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GroupLayout {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim_qk: usize,
    pub head_dim_v: usize,
    /// Floats in one boundary tap (0 for groups without taps).
    pub tap_width: usize,
}

impl GroupLayout {
    pub fn k_width(&self) -> usize {
        self.num_kv_heads * self.head_dim_qk
    }

    pub fn row_width(&self) -> usize {
        self.num_kv_heads * (self.head_dim_qk + self.head_dim_v)
    }
}

/// A row's tag: `pos + 1` (0 means never written since the last zero) and its token.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Tag {
    pub pos_plus_one: u32,
    pub token: u32,
}

#[derive(Debug)]
pub(crate) struct Pool {
    pub layout: GroupLayout,
    block_size: usize,
    num_blocks: usize,
    data: Vec<f32>,
    tags: Vec<Tag>,
    taps: Vec<f32>,
    tap_tags: Vec<u32>,
    /// Per block: whether it is held encrypted (instrumentation).
    encrypted: Vec<bool>,
}

impl Pool {
    pub fn new(layout: GroupLayout, block_size: usize, num_blocks: usize) -> Self {
        let rows = num_blocks * layout.num_layers * block_size;
        Self {
            data: vec![0.0; rows * layout.row_width()],
            tags: vec![Tag::default(); rows],
            taps: vec![0.0; num_blocks * layout.tap_width],
            tap_tags: vec![0; num_blocks],
            encrypted: vec![false; num_blocks],
            layout,
            block_size,
            num_blocks,
        }
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    fn row_index(&self, block: u32, layer: usize, offset: usize) -> usize {
        (block as usize * self.layout.num_layers + layer) * self.block_size + offset
    }

    fn rows_per_block(&self) -> usize {
        self.layout.num_layers * self.block_size
    }

    /// Stores one row's K and V.
    pub fn write(
        &mut self,
        block: u32,
        layer: usize,
        offset: usize,
        tag: Tag,
        k: &[f32],
        v: &[f32],
    ) {
        let w = self.layout.row_width();
        let kw = self.layout.k_width();
        let r = self.row_index(block, layer, offset);
        let row = &mut self.data[r * w..(r + 1) * w];
        row[..kw].copy_from_slice(k);
        row[kw..].copy_from_slice(v);
        self.tags[r] = tag;
    }

    /// One row's `(K, V, tag)`.
    pub fn read(&self, block: u32, layer: usize, offset: usize) -> (&[f32], &[f32], Tag) {
        let w = self.layout.row_width();
        let kw = self.layout.k_width();
        let r = self.row_index(block, layer, offset);
        let row = &self.data[r * w..(r + 1) * w];
        (&row[..kw], &row[kw..], self.tags[r])
    }

    pub fn tag(&self, block: u32, layer: usize, offset: usize) -> Tag {
        self.tags[self.row_index(block, layer, offset)]
    }

    pub fn write_tap(&mut self, block: u32, pos: u32, tap: &[f32]) {
        let tw = self.layout.tap_width;
        self.taps[block as usize * tw..(block as usize + 1) * tw].copy_from_slice(tap);
        self.tap_tags[block as usize] = pos + 1;
    }

    /// The tap of `block` if it was recorded for `pos`.
    pub fn read_tap(&self, block: u32, pos: u32) -> Option<&[f32]> {
        let tw = self.layout.tap_width;
        (self.tap_tags[block as usize] == pos + 1)
            .then(|| &self.taps[block as usize * tw..(block as usize + 1) * tw])
    }

    pub fn zero(&mut self, block: u32) {
        let (rpb, w, tw) = (
            self.rows_per_block(),
            self.layout.row_width(),
            self.layout.tap_width,
        );
        let b = block as usize;
        self.data[b * rpb * w..(b + 1) * rpb * w].fill(0.0);
        self.tags[b * rpb..(b + 1) * rpb].fill(Tag::default());
        self.taps[b * tw..(b + 1) * tw].fill(0.0);
        self.tap_tags[b] = 0;
        self.encrypted[b] = false;
    }

    /// Encrypts (`encrypt`) or decrypts the block in place: one keystream XOR over its
    /// image (see the module docs). Panics on encrypting an encrypted block or decrypting
    /// a plaintext one: the host tracks which blocks are encrypted, so either is a host bug.
    pub fn crypt(&mut self, block: u32, key: &KvKey, nonce: &KvNonce, encrypt: bool) {
        let b = block as usize;
        assert_ne!(
            self.encrypted[b],
            encrypt,
            "block {block}: {} twice",
            if encrypt { "encrypted" } else { "decrypted" }
        );
        self.encrypted[b] = encrypt;
        let (rpb, w, tw) = (
            self.rows_per_block(),
            self.layout.row_width(),
            self.layout.tap_width,
        );
        let mut ks = KvKeystream::new(key, nonce);
        let data = &mut self.data[b * rpb * w..(b + 1) * rpb * w];
        ks.range(data.len(), |i, k| {
            data[i] = f32::from_bits(data[i].to_bits() ^ k)
        });
        let taps = &mut self.taps[b * tw..(b + 1) * tw];
        ks.range(taps.len(), |i, k| {
            taps[i] = f32::from_bits(taps[i].to_bits() ^ k)
        });
        let tags = &mut self.tags[b * rpb..(b + 1) * rpb];
        ks.range(2 * tags.len(), |i, k| {
            let t = &mut tags[i / 2];
            if i % 2 == 0 {
                t.pos_plus_one ^= k;
            } else {
                t.token ^= k;
            }
        });
        let tap_tag = &mut self.tap_tags[b];
        ks.range(1, |_, k| *tap_tag ^= k);
    }

    /// Whether the block is held encrypted.
    pub fn is_encrypted(&self, block: u32) -> bool {
        self.encrypted[block as usize]
    }

    /// The block's image as words, in [`Pool::crypt`]'s order (tests).
    pub fn image(&self, block: u32) -> Vec<u32> {
        let (rpb, w, tw) = (
            self.rows_per_block(),
            self.layout.row_width(),
            self.layout.tap_width,
        );
        let b = block as usize;
        let mut out: Vec<u32> = self.data[b * rpb * w..(b + 1) * rpb * w]
            .iter()
            .map(|x| x.to_bits())
            .collect();
        out.extend(self.taps[b * tw..(b + 1) * tw].iter().map(|x| x.to_bits()));
        for t in &self.tags[b * rpb..(b + 1) * rpb] {
            out.extend([t.pos_plus_one, t.token]);
        }
        out.push(self.tap_tags[b]);
        out
    }

    pub fn copy(&mut self, src: u32, dst: u32) {
        let (rpb, w, tw) = (
            self.rows_per_block(),
            self.layout.row_width(),
            self.layout.tap_width,
        );
        let (s, d) = (src as usize, dst as usize);
        self.data
            .copy_within(s * rpb * w..(s + 1) * rpb * w, d * rpb * w);
        self.tags.copy_within(s * rpb..(s + 1) * rpb, d * rpb);
        self.taps.copy_within(s * tw..(s + 1) * tw, d * tw);
        self.tap_tags[d] = self.tap_tags[s];
        self.encrypted[d] = self.encrypted[s];
    }

    /// Whether every byte of the block (data, tap, and tags) is zero.
    pub fn block_is_zero(&self, block: u32) -> bool {
        let (rpb, w, tw) = (
            self.rows_per_block(),
            self.layout.row_width(),
            self.layout.tap_width,
        );
        let b = block as usize;
        self.data[b * rpb * w..(b + 1) * rpb * w]
            .iter()
            .all(|x| x.to_bits() == 0)
            && self.tags[b * rpb..(b + 1) * rpb]
                .iter()
                .all(|t| *t == Tag::default())
            && self.taps[b * tw..(b + 1) * tw]
                .iter()
                .all(|x| x.to_bits() == 0)
            && self.tap_tags[b] == 0
    }
}
