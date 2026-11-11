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
//!
//! **The host tier** is a second pool per group of the same layout. [`Pool::transfer`]
//! copies a block's whole image (data, tap, tags and the encrypted flag) between a
//! device pool and a host pool; the executor demotes only encrypted blocks and promotes
//! only into zeroed ones, and panics otherwise. A host pool's buffers are locked into
//! memory ([`Pool::lock`], `mlock`).
//!
//! **Every pool's buffers** are page-aligned allocations of whole pages that no other
//! allocation shares, zeroed before they are freed and, if locked, unlocked after that.

use std::alloc::Layout;
use std::ptr::NonNull;

use eidola_engine::kv_cipher::{KvKeystream, KvNonce};
use eidola_engine::secret::KvKey;
use zeroize::Zeroize;

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

/// Plain data whose all-zero bytes are a valid value.
///
/// # Safety
///
/// All-zero bytes must be a valid `Self`.
unsafe trait Zeroable: Copy {}

// SAFETY: zero bytes are `0.0`, `0` and a `Tag` of two zero words.
unsafe impl Zeroable for f32 {}
// SAFETY: as above.
unsafe impl Zeroable for u32 {}
// SAFETY: as above.
unsafe impl Zeroable for Tag {}

/// `len` values in an allocation of their own: page-aligned and whole pages long, zero when
/// made, and zeroed (then unlocked) before it is freed. Since no other allocation shares its
/// pages, locking or unlocking them touches nothing else.
struct PageBuf<T: Zeroable> {
    ptr: NonNull<T>,
    len: usize,
    /// The allocation; `None` for an empty buffer, which owns no memory.
    layout: Option<Layout>,
    locked: bool,
}

// SAFETY: the buffer uniquely owns its allocation, as a `Box<[T]>` does.
unsafe impl<T: Zeroable + Send> Send for PageBuf<T> {}
// SAFETY: shared access is only through `&[T]`.
unsafe impl<T: Zeroable + Sync> Sync for PageBuf<T> {}

impl<T: Zeroable> PageBuf<T> {
    fn zeroed(len: usize) -> Self {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .expect("a pool's size fits in usize");
        if bytes == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len,
                layout: None,
                locked: false,
            };
        }
        let page = rustix::param::page_size();
        let layout = Layout::from_size_align(bytes.next_multiple_of(page), page)
            .expect("a pool's size fits a layout");
        // SAFETY: `layout` has a nonzero size, and its alignment (a page) is at least `T`'s.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr =
            NonNull::new(raw.cast::<T>()).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self {
            ptr,
            len,
            layout: Some(layout),
            locked: false,
        }
    }

    /// The allocation as words: every byte of every page it owns.
    fn words(&mut self) -> &mut [u64] {
        match self.layout {
            // SAFETY: this buffer's own live allocation, page-aligned and whole pages long
            // (so a whole number of aligned `u64`s), borrowed uniquely through `self`.
            Some(l) => unsafe {
                std::slice::from_raw_parts_mut(self.ptr.as_ptr().cast::<u64>(), l.size() / 8)
            },
            None => &mut [],
        }
    }

    fn lock(&mut self) -> std::io::Result<()> {
        if let Some(l) = self.layout {
            // SAFETY: exactly this buffer's allocation, whole pages no other allocation uses.
            unsafe { rustix::mm::mlock(self.ptr.as_ptr().cast(), l.size()) }?;
            self.locked = true;
        }
        Ok(())
    }

    fn unlock(&mut self) {
        if let (true, Some(l)) = (std::mem::take(&mut self.locked), self.layout) {
            // SAFETY: exactly this buffer's allocation, locked by `lock`; its pages are its
            // own, so no other allocation's lock is released. An error leaves them locked
            // until they are freed, which is harmless.
            let _ = unsafe { rustix::mm::munlock(self.ptr.as_ptr().cast(), l.size()) };
        }
    }

    /// Zeroes every byte the allocation owns, then unlocks it.
    fn scrub(&mut self) {
        self.words().zeroize();
        self.unlock();
    }
}

impl<T: Zeroable> Drop for PageBuf<T> {
    fn drop(&mut self) {
        self.scrub();
        if let Some(l) = self.layout {
            // SAFETY: allocated by `zeroed` with this layout and not freed since.
            unsafe { std::alloc::dealloc(self.ptr.as_ptr().cast(), l) };
        }
    }
}

impl<T: Zeroable> std::ops::Deref for PageBuf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: `len` initialised values (zero, a valid `T`, or written since) at `ptr`,
        // or an empty slice at a dangling, aligned pointer.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Zeroable> std::ops::DerefMut for PageBuf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`, borrowed uniquely through `self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Zeroable> std::fmt::Debug for PageBuf<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageBuf").field("len", &self.len).finish()
    }
}

/// A group's blocks. Block contents (data, taps, tags) live in [`PageBuf`]s, so dropping a
/// pool zeroes them, and a host pool locks and unlocks only its own pages.
#[derive(Debug)]
pub(crate) struct Pool {
    pub layout: GroupLayout,
    block_size: usize,
    num_blocks: usize,
    data: PageBuf<f32>,
    tags: PageBuf<Tag>,
    taps: PageBuf<f32>,
    tap_tags: PageBuf<u32>,
    /// Per block: whether it is held encrypted (instrumentation).
    encrypted: Vec<bool>,
}

impl Pool {
    pub fn new(layout: GroupLayout, block_size: usize, num_blocks: usize) -> Self {
        let rows = num_blocks * layout.num_layers * block_size;
        Self {
            data: PageBuf::zeroed(rows * layout.row_width()),
            tags: PageBuf::zeroed(rows),
            taps: PageBuf::zeroed(num_blocks * layout.tap_width),
            tap_tags: PageBuf::zeroed(num_blocks),
            encrypted: vec![false; num_blocks],
            layout,
            block_size,
            num_blocks,
        }
    }

    /// Locks every buffer holding block contents into memory, so none of it can be
    /// paged out to swap; dropping the pool unlocks them. Fails, leaving nothing locked,
    /// if the system refuses.
    pub fn lock(&mut self) -> std::io::Result<()> {
        let locked = self
            .data
            .lock()
            .and_then(|()| self.tags.lock())
            .and_then(|()| self.taps.lock())
            .and_then(|()| self.tap_tags.lock());
        if locked.is_err() {
            self.data.unlock();
            self.tags.unlock();
            self.taps.unlock();
            self.tap_tags.unlock();
        }
        locked
    }

    /// Copies block `src` of `from` over block `dst` of `to`, its whole image: data,
    /// tap, row tags, tap tag, and the encrypted flag. The pools have the same layout.
    pub fn transfer(from: &Pool, src: u32, to: &mut Pool, dst: u32) {
        assert_eq!(
            from.layout, to.layout,
            "transfer between pools of one layout"
        );
        assert_eq!(from.block_size, to.block_size);
        let (rpb, w, tw) = (
            from.rows_per_block(),
            from.layout.row_width(),
            from.layout.tap_width,
        );
        let (s, d) = (src as usize, dst as usize);
        to.data[d * rpb * w..(d + 1) * rpb * w]
            .copy_from_slice(&from.data[s * rpb * w..(s + 1) * rpb * w]);
        to.tags[d * rpb..(d + 1) * rpb].copy_from_slice(&from.tags[s * rpb..(s + 1) * rpb]);
        to.taps[d * tw..(d + 1) * tw].copy_from_slice(&from.taps[s * tw..(s + 1) * tw]);
        to.tap_tags[d] = from.tap_tags[s];
        to.encrypted[d] = from.encrypted[s];
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Bytes one block takes: its data, tap, row tags, tap tag and encrypted flag.
    pub fn block_bytes(&self) -> usize {
        let rows = self.rows_per_block();
        rows * self.layout.row_width() * 4
            + self.layout.tap_width * 4
            + rows * std::mem::size_of::<Tag>()
            + 4
            + 1
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Each buffer owns whole pages outright, so locking or unlocking it cannot reach a
    /// neighbouring allocation's pages.
    #[test]
    fn buffers_own_whole_pages() {
        let page = rustix::param::page_size();
        for len in [1, 3, page / 4, page / 4 + 1, 5 * page] {
            let a = PageBuf::<u32>::zeroed(len);
            let b = PageBuf::<u32>::zeroed(len);
            for buf in [&a, &b] {
                let l = buf.layout.expect("a nonempty buffer owns memory");
                assert_eq!(buf.ptr.as_ptr() as usize % page, 0);
                assert_eq!(l.size() % page, 0);
                assert!(l.size() >= len * 4);
                assert!(buf.iter().all(|&w| w == 0));
            }
        }
        assert!(PageBuf::<f32>::zeroed(0).layout.is_none());
    }

    /// Scrubbing (what dropping does before freeing) zeroes every byte of the pages,
    /// including the tail past the values, and unlocks them.
    #[test]
    fn scrub_zeroes_every_owned_byte_and_unlocks() {
        let mut buf = PageBuf::<u32>::zeroed(10);
        buf.fill(0xA5A5_A5A5);
        *buf.words().last_mut().unwrap() = u64::MAX;
        if buf.lock().is_ok() {
            assert!(buf.locked);
        }
        buf.scrub();
        assert!(!buf.locked);
        assert!(buf.words().iter().all(|&w| w == 0));
    }
}
