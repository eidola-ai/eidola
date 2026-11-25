//! The node's global allocator: every heap allocation is zeroed before it is freed.
//!
//! What a step computes on the host (the CPU executor's per-step scratch: token ids and
//! positions, hidden states, attention outputs, logits and distributions, drafts) and what
//! the node handles around it (request bodies, parse trees, rendered prompts, token ids,
//! output text) lives in heap memory that is freed when the step or the request is done.
//! The system allocator hands freed memory back to its free lists, or to the kernel, with
//! its contents in place, so a memory reader of the VM would find them there. [`ZeroOnFree`]
//! wraps the system allocator so that nothing is freed with its contents:
//!
//! - `dealloc` zeroes the whole block before handing it back;
//! - `realloc` never lets the inner allocator move a block (which would free the old one
//!   with its contents): it allocates the new block, copies, and frees the old one through
//!   `dealloc`, zeroed;
//! - `alloc` and `alloc_zeroed` are the inner allocator's, so what a caller reads from a
//!   fresh block is exactly what it would read without this wrapper, and no number the
//!   executors compute can change.
//!
//! The zeroing cannot be removed as a dead store: after the stores, an empty `asm!` block
//! takes the block's address and may read memory through it, so the compiler must keep
//! them, even though the memory is freed next.
//!
//! The binary installs it (`main.rs`); tests and examples run with the default allocator,
//! except `tests/zeroing.rs`, which runs the node under it.

use std::alloc::{GlobalAlloc, Layout};

/// A global allocator over `A` that zeroes every block before freeing it. See the module
/// documentation.
pub struct ZeroOnFree<A>(pub A);

/// Zero `len` bytes at `ptr`, in a way the compiler keeps although the memory is freed
/// next.
///
/// # Safety
///
/// `ptr` must be valid for writes of `len` bytes.
#[inline]
unsafe fn scrub(ptr: *mut u8, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: the caller's contract.
    unsafe { core::ptr::write_bytes(ptr, 0, len) };
    // SAFETY: an empty template; it touches no register but the one it is given, and no
    // stack. Without `nomem` or `readonly`, the compiler must assume it reads the memory
    // `ptr` points to, so the stores above happen before it.
    unsafe { core::arch::asm!("/* {0} */", in(reg) ptr, options(nostack, preserves_flags)) };
}

// SAFETY: every block is allocated and freed by `A`, with the layout it was allocated
// with; this wrapper only writes zeros into a block it is about to free, and copies a
// block's live bytes into a new one on `realloc`.
unsafe impl<A: GlobalAlloc> GlobalAlloc for ZeroOnFree<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        unsafe { self.0.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        unsafe { self.0.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` is a live block of `layout.size()` bytes (the caller's contract).
        unsafe {
            scrub(ptr, layout.size());
            self.0.dealloc(ptr, layout);
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller guarantees `new_size`, rounded up to `layout.align()`, does
        // not overflow `isize`, which is `Layout`'s own requirement.
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        // SAFETY: `new_layout` is non-zero-sized (the caller guarantees `new_size > 0`).
        let new = unsafe { self.0.alloc(new_layout) };
        if !new.is_null() {
            // SAFETY: both blocks are live and distinct, each at least as long as the copy;
            // the old block is then freed, zeroed, with the layout it was allocated with.
            unsafe {
                core::ptr::copy_nonoverlapping(ptr, new, layout.size().min(new_size));
                self.dealloc(ptr, layout);
            }
        }
        // On failure the old block stays live and untouched, as `realloc` promises.
        new
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::System;
    use std::sync::Mutex;

    /// The system allocator, recording what each block held when it was freed and
    /// whether its own `realloc` was ever called.
    struct Inspecting {
        freed: Mutex<Vec<Vec<u8>>>,
        inner_reallocs: Mutex<usize>,
    }

    unsafe impl GlobalAlloc for Inspecting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            let bytes = unsafe { std::slice::from_raw_parts(ptr, layout.size()) }.to_vec();
            self.freed.lock().unwrap().push(bytes);
            unsafe { System.dealloc(ptr, layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            *self.inner_reallocs.lock().unwrap() += 1;
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    fn inspecting() -> ZeroOnFree<Inspecting> {
        ZeroOnFree(Inspecting {
            freed: Mutex::new(Vec::new()),
            inner_reallocs: Mutex::new(0),
        })
    }

    /// A freed block reaches the inner allocator all zero, whatever it held.
    #[test]
    fn a_block_is_zero_when_it_is_freed() {
        let a = inspecting();
        for size in [1usize, 7, 64, 4096, 1 << 20] {
            let layout = Layout::from_size_align(size, 8).unwrap();
            unsafe {
                let p = a.alloc(layout);
                assert!(!p.is_null());
                core::ptr::write_bytes(p, 0xA5, size);
                a.dealloc(p, layout);
            }
        }
        let freed = a.0.freed.lock().unwrap();
        assert_eq!(freed.len(), 5);
        for block in freed.iter() {
            assert!(
                block.iter().all(|&b| b == 0),
                "a {}-byte block",
                block.len()
            );
        }
    }

    /// Growing or shrinking a block keeps its bytes, never lets the inner allocator move
    /// it, and frees the old block zeroed.
    #[test]
    fn a_reallocated_block_keeps_its_bytes_and_leaves_none_behind() {
        let a = inspecting();
        let layout = Layout::from_size_align(256, 16).unwrap();
        unsafe {
            let p = a.alloc(layout);
            for i in 0..256 {
                *p.add(i) = i as u8;
            }
            let grown = a.realloc(p, layout, 4096);
            assert!(!grown.is_null());
            assert_eq!(grown as usize % 16, 0, "the alignment is kept");
            for i in 0..256 {
                assert_eq!(*grown.add(i), i as u8);
            }
            let grown_layout = Layout::from_size_align(4096, 16).unwrap();
            let shrunk = a.realloc(grown, grown_layout, 100);
            for i in 0..100 {
                assert_eq!(*shrunk.add(i), i as u8);
            }
            a.dealloc(shrunk, Layout::from_size_align(100, 16).unwrap());
        }
        assert_eq!(*a.0.inner_reallocs.lock().unwrap(), 0);
        let freed = a.0.freed.lock().unwrap();
        assert_eq!(
            freed.iter().map(Vec::len).collect::<Vec<_>>(),
            [256, 4096, 100]
        );
        assert!(freed.iter().flatten().all(|&b| b == 0));
    }

    /// `alloc_zeroed` is the inner allocator's, zero; an empty block is not written.
    #[test]
    fn zeroed_allocation_passes_through_and_an_empty_block_is_untouched() {
        let a = inspecting();
        let layout = Layout::from_size_align(1 << 16, 64).unwrap();
        unsafe {
            let p = a.alloc_zeroed(layout);
            assert!(
                std::slice::from_raw_parts(p, 1 << 16)
                    .iter()
                    .all(|&b| b == 0)
            );
            a.dealloc(p, layout);
            let empty = Layout::from_size_align(0, 1).unwrap();
            scrub(core::ptr::NonNull::<u8>::dangling().as_ptr(), empty.size());
        }
    }
}
