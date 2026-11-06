//! The idle-KV cipher on the device: `kv_cipher.cu`'s `eidola_kv_chacha20`, the
//! transliteration of `eidola_engine::kv_cipher` that one launch applies to a list of
//! ranges ([`CipherItem`]).
//!
//! Keys reach the device only inside the item list, which [`KvCipher::apply`] uploads,
//! launches over, and zeroes on the stream right behind the launch; the host copy is
//! scrubbed once the upload has returned (a copy from pageable memory has left it by
//! then). The host list is a [`CipherItems`], never a bare `Vec`: growing a `Vec` frees
//! its old allocation with the keys still in it, so the list scrubs every allocation it
//! leaves (and the store sizes it for the whole step up front). Not covered: an item's
//! transient stack copies while it is built, and the driver's own staging of the upload
//! (under confidential computing, a bounce buffer inside the encrypted VM), as
//! `eidola-engine` leaves the HMAC's transient stack state uncovered.

use cudarc::driver::sys;
use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};
use eidola_engine::kv_cipher::KvNonce;
use eidola_engine::secret::KvKey;
use zeroize::Zeroize;

use crate::launch::dptr;
use crate::module::{Kernel, KernelDir, KernelModule};
use crate::{CudaError, Gpu, Result, launch};

/// `EidolaKvCipherItem`: XOR `words` 32-bit words at `base` with the keystream of
/// (`key`, `nonce`) from block `counter` on.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct CipherItem {
    pub base: u64,
    pub words: u32,
    pub counter: u32,
    pub key: [u32; 8],
    pub nonce: [u32; 3],
    pub reserved: u32,
}

const _: () = assert!(std::mem::size_of::<CipherItem>() == 64);

// SAFETY: plain old data, laid out as the kernel's struct (asserted above).
unsafe impl DeviceRepr for CipherItem {}
// SAFETY: all-zero bytes are a valid (empty) item.
unsafe impl ValidAsZeroBits for CipherItem {}

impl std::fmt::Debug for CipherItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CipherItem")
            .field("base", &self.base)
            .field("words", &self.words)
            .field("counter", &self.counter)
            .finish_non_exhaustive()
    }
}

impl CipherItem {
    /// An item over `words` words at `base`, from keystream block `counter`.
    pub fn new(base: u64, words: u32, counter: u32, key: &KvKey, nonce: &KvNonce) -> Self {
        let mut item = CipherItem {
            base,
            words,
            counter,
            nonce: nonce.words(),
            ..CipherItem::default()
        };
        key.words_into(&mut item.key);
        item
    }

    fn scrub(&mut self) {
        self.key.zeroize();
        self.nonce.zeroize();
    }
}

/// A host list of [`CipherItem`]s, which carry keys, that never leaves a copy of one in
/// memory it gives back: it grows only by moving its items into a larger allocation and
/// scrubbing the old one before freeing it (a plain `Vec` would free the old allocation
/// with the keys in it), and it scrubs its items when cleared or dropped.
#[derive(Debug, Default)]
pub struct CipherItems(Vec<CipherItem>);

impl CipherItems {
    /// An empty list with room for `n` items before it first grows.
    pub fn with_capacity(n: usize) -> Self {
        CipherItems(Vec::with_capacity(n))
    }

    /// Appends an item. `Vec::push` with spare capacity never reallocates, so the only
    /// reallocation is [`regrow`]'s, which scrubs what it leaves behind.
    pub fn push(&mut self, item: CipherItem) {
        if self.0.len() == self.0.capacity() {
            let cap = (2 * self.0.capacity()).max(16);
            let old = std::mem::take(&mut self.0);
            self.0 = regrow(old, cap);
        }
        debug_assert!(self.0.len() < self.0.capacity());
        self.0.push(item);
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_slice(&self) -> &[CipherItem] {
        &self.0
    }

    /// Scrubs every item and empties the list, keeping its allocation.
    pub fn clear(&mut self) {
        for item in &mut self.0 {
            item.scrub();
        }
        self.0.clear();
    }
}

impl Drop for CipherItems {
    fn drop(&mut self) {
        self.clear();
    }
}

impl FromIterator<CipherItem> for CipherItems {
    fn from_iter<I: IntoIterator<Item = CipherItem>>(iter: I) -> Self {
        let mut items = CipherItems::default();
        for item in iter {
            items.push(item);
        }
        items
    }
}

/// Moves `old`'s items into a new allocation of `cap` items and scrubs `old`'s before
/// it is freed.
fn regrow(mut old: Vec<CipherItem>, cap: usize) -> Vec<CipherItem> {
    let new = copy_and_scrub(&mut old, cap);
    drop(old);
    new
}

/// The copy half of [`regrow`]: `old`'s items in a new allocation of `cap`, and `old`
/// left allocated but scrubbed (separate so a test can look at what is freed).
fn copy_and_scrub(old: &mut [CipherItem], cap: usize) -> Vec<CipherItem> {
    let mut new = Vec::with_capacity(cap.max(old.len()));
    new.extend_from_slice(old);
    for item in old.iter_mut() {
        item.scrub();
    }
    new
}

/// The keystream block after a range of `words` words starting at `counter`: every range
/// of a block's image starts at a fresh block (`eidola_engine::kv_cipher`).
pub fn next_counter(counter: u32, words: u32) -> Result<u32> {
    counter
        .checked_add(words.div_ceil(16))
        .ok_or_else(|| CudaError::new("KV block image past the 32-bit ChaCha20 counter"))
}

/// Threads per block of `eidola_kv_chacha20`.
pub const THREADS: u32 = 256;
/// Most block columns per item: enough to cover a large target block (Flash's global
/// blocks are about 16 K keystream blocks) in one pass, with the rest striding.
pub const MAX_GRID_X: u32 = 64;
/// The driver's grid limit in y.
const MAX_GRID_Y: u32 = 65_535;

/// The launch grid for `items`: one block row per item (up to the driver's limit, the
/// kernel strides past it) and enough columns for the largest item's keystream blocks.
pub fn grid(items: &[CipherItem]) -> Result<[u32; 3]> {
    let rows = u32::try_from(items.len())
        .map_err(|_| CudaError::new("kv_chacha20: too many items"))?
        .min(MAX_GRID_Y);
    let widest = items
        .iter()
        .map(|i| i.words.div_ceil(16))
        .max()
        .unwrap_or(0);
    let cols = widest.div_ceil(THREADS).clamp(1, MAX_GRID_X);
    Ok([cols, rows, 1])
}

/// The loaded kernel and its device item list (grown as needed, zero between launches).
pub struct KvCipher {
    _module: KernelModule,
    kernel: Kernel,
    items: Option<CudaSlice<CipherItem>>,
}

impl KvCipher {
    pub const IMAGE: &'static str = "kv_cipher";
    pub const SYMBOL: &'static str = "eidola_kv_chacha20";

    pub fn load(gpu: &Gpu, dir: &KernelDir) -> Result<KvCipher> {
        KvCipher::from_module(KernelModule::load(gpu, dir, Self::IMAGE)?)
    }

    pub fn from_module(module: KernelModule) -> Result<KvCipher> {
        let kernel = module.bound_kernel(Self::IMAGE, Self::SYMBOL)?;
        if kernel.meta().block != [THREADS, 1, 1] || kernel.meta().params_bytes != 0 {
            return Err(CudaError::new("eidola_kv_chacha20: launch contract"));
        }
        Ok(KvCipher {
            _module: module,
            kernel,
            items: None,
        })
    }

    /// XORs every item's range with its keystream, on the stream, and scrubs the items:
    /// `items` is left empty and its key words zeroed, and the device copy is zeroed by a
    /// memset queued right after the launch. Only word alignment is checked here (a
    /// misaligned range is refused before anything launches).
    ///
    /// # Safety
    ///
    /// Each item's `base .. base + 4 * words` must lie inside one live device allocation
    /// of this context, and stay allocated, with nothing else reading or writing it,
    /// until the launch has completed on `gpu`'s stream (stream order covers later work
    /// queued there); no two items' ranges may overlap.
    pub unsafe fn apply(&mut self, gpu: &Gpu, items: &mut CipherItems) -> Result<()> {
        let result = self.launch(gpu, items.as_slice());
        items.clear();
        result
    }

    /// The launch [`KvCipher::apply`] makes; its safety contract is `apply`'s.
    fn launch(&mut self, gpu: &Gpu, items: &[CipherItem]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        if items.iter().any(|i| i.base % 4 != 0) {
            return Err(CudaError::new("kv_chacha20: a range is not word-aligned"));
        }
        let s = gpu.stream();
        let n = items.len();
        if self.items.as_ref().is_none_or(|b| b.len() < n) {
            // The old list is zero (each launch's memset leaves it so) when it is freed.
            self.items = Some(s.alloc_zeros::<CipherItem>(n.next_power_of_two())?);
        }
        let buf = self.items.as_mut().expect("allocated");
        let mut view = buf.slice_mut(..n);
        s.memcpy_htod(items, &mut view)?;
        let base = dptr(buf, s);
        let count = u32::try_from(n).map_err(|_| CudaError::new("kv_chacha20: too many items"))?;
        // SAFETY: `eidola_kv_chacha20(const EidolaKvCipherItem*, uint32_t)`; the list holds
        // `count` items, each naming a range the caller vouches for.
        let launched = unsafe { launch!(gpu, self.kernel, grid(items)?, base, count) };
        // Zero the device copy of the keys whatever the launch did.
        gpu.context().bind_to_thread()?;
        // SAFETY: the first `n` items of the list.
        unsafe { sys::cuMemsetD8Async(base, 0, n * 64, s.cu_stream()) }.result()?;
        launched
    }

    /// The device item list's current contents (tests: zero after every launch).
    pub fn read_items(&self, gpu: &Gpu) -> Result<Vec<CipherItem>> {
        match &self.items {
            Some(b) => Ok(gpu.stream().clone_dtoh(b)?),
            None => Ok(Vec::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(words: u32) -> CipherItem {
        CipherItem {
            words,
            ..CipherItem::default()
        }
    }

    #[test]
    fn grid_covers_the_largest_item_and_every_item_row() {
        assert_eq!(grid(&[item(16)]).unwrap(), [1, 1, 1]);
        assert_eq!(grid(&[item(16 * 256 + 1), item(3)]).unwrap(), [2, 2, 1]);
        assert_eq!(grid(&[item(u32::MAX)]).unwrap()[0], MAX_GRID_X);
        let many = vec![item(1); 70_000];
        assert_eq!(grid(&many).unwrap()[1], 65_535);
    }

    #[test]
    fn counters_advance_by_whole_blocks_and_refuse_overflow() {
        assert_eq!(next_counter(0, 0).unwrap(), 0);
        assert_eq!(next_counter(0, 1).unwrap(), 1);
        assert_eq!(next_counter(3, 32).unwrap(), 5);
        assert_eq!(next_counter(3, 33).unwrap(), 6);
        assert!(next_counter(u32::MAX, 1).is_err());
    }

    /// Growing the list scrubs the allocation it leaves: the old buffer holds no key word
    /// once its items have moved, and the list keeps every item.
    #[test]
    fn growth_scrubs_the_allocation_it_leaves() {
        let key = KvKey::from_bytes([0x5a; 32]);
        let item = |i: u64| CipherItem::new(4 * i, 1, 0, &key, &KvNonce::from_words([1, 2, 3]));
        let mut old: Vec<CipherItem> = (0..5).map(item).collect();
        let new = copy_and_scrub(&mut old, 10);
        assert!(old.iter().all(|i| i.key == [0; 8] && i.nonce == [0; 3]));
        assert_eq!(new.len(), 5);
        assert!(new.iter().all(|i| i.key == [0x5a5a_5a5a; 8]));
        // Through the list: past several growths every item is kept, in order.
        let mut items = CipherItems::with_capacity(1);
        for i in 0..100 {
            items.push(item(i));
        }
        assert_eq!(items.len(), 100);
        assert!(
            items
                .as_slice()
                .iter()
                .enumerate()
                .all(|(i, it)| it.base == 4 * i as u64)
        );
        items.clear();
        assert!(items.is_empty());
    }

    #[test]
    fn items_carry_the_key_words_and_print_none() {
        let key = KvKey::from_bytes(std::array::from_fn(|i| u8::try_from(i).unwrap()));
        let it = CipherItem::new(64, 5, 2, &key, &KvNonce::from_words([1, 2, 3]));
        assert_eq!(it.key[0], 0x0302_0100);
        assert_eq!(it.nonce, [1, 2, 3]);
        let text = format!("{it:?}");
        assert!(!text.contains("key") && !text.contains("nonce"), "{text}");
    }
}
