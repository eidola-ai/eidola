//! The memory every KV pool is built from: [`PageBuf`], page-aligned allocations of whole
//! pages that no other allocation shares, zeroed before they are freed and, if locked,
//! unlocked after that; and [`Tag`], the instrumentation each stored row carries.

use std::alloc::Layout;
use std::ptr::NonNull;

use zeroize::Zeroize;

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
pub(crate) unsafe trait Zeroable: Copy {}

// SAFETY: zero bytes are `0.0`, `0` and a `Tag` of two zero words.
unsafe impl Zeroable for f32 {}
// SAFETY: as above.
unsafe impl Zeroable for u32 {}
// SAFETY: as above.
unsafe impl Zeroable for Tag {}

/// `len` values in an allocation of their own: page-aligned and whole pages long, zero when
/// made, and zeroed (then unlocked) before it is freed. Since no other allocation shares its
/// pages, locking or unlocking them touches nothing else.
pub(crate) struct PageBuf<T: Zeroable> {
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
    pub(crate) fn zeroed(len: usize) -> Self {
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

    pub(crate) fn lock(&mut self) -> std::io::Result<()> {
        if let Some(l) = self.layout {
            // SAFETY: exactly this buffer's allocation, whole pages no other allocation uses.
            unsafe { rustix::mm::mlock(self.ptr.as_ptr().cast(), l.size()) }?;
            self.locked = true;
        }
        Ok(())
    }

    pub(crate) fn unlock(&mut self) {
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
