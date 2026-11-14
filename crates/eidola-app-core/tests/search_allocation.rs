//! Allocation gate for the text matcher (`eidola_app_core::search`).
//!
//! The matcher folds a whole haystack on every case-insensitive search, and a
//! window searches on every keystroke, so per-character heap traffic in the
//! fold is a cost the whole feature pays. This binary installs a counting
//! allocator — hence its own file, since a `#[global_allocator]` is per test
//! binary — and pins the fold's allocation count to *the output*, not to the
//! input's length.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use eidola_app_core::search::{Query, fold_case};

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

// SAFETY: every method forwards to `System` unchanged; the counter is the only
// addition and touches no allocator state.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn counted<T>(body: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let value = body();
    (value, ALLOCATIONS.load(Ordering::Relaxed) - before)
}

/// A post-sized fixture: mixed scripts, so the walk is not all ASCII.
fn haystack() -> String {
    let paragraph = "The mitochondria is the powerhouse of the cell — καλημέρα \
                     κόσμε, 你好世界, Grüße aus München. ";
    paragraph.repeat(60)
}

#[test]
fn folding_does_not_allocate_per_character() {
    let text = haystack();
    let characters = text.chars().count();

    // Warm the allocator's own lazily-initialized state before counting.
    let _ = fold_case(&text);

    let (projection, allocations) = counted(|| fold_case(&text));
    println!("fold_case: {characters} characters, {allocations} allocations");
    assert_eq!(projection.text().chars().count(), characters);
    assert!(
        allocations < characters / 8,
        "folding {characters} characters took {allocations} allocations — the \
         fold is allocating per character rather than per output buffer",
    );
}

#[test]
fn a_case_insensitive_search_does_not_allocate_per_character() {
    let text = haystack();
    let characters = text.chars().count();
    let query = Query::new("powerhouse").expect("non-empty");

    let _ = query.find_in(&text);

    let (hits, allocations) = counted(|| query.find_in(&text));
    println!("find_in: {characters} characters, {allocations} allocations");
    assert_eq!(hits.len(), 60);
    assert!(
        allocations < characters / 8,
        "searching {characters} characters took {allocations} allocations",
    );
}
