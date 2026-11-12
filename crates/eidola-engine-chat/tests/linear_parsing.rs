//! One chunk holding many tool-call blocks is parsed with work linear in its
//! length.
//!
//! The measure is bytes allocated, counted by this binary's allocator: exact
//! and the same on every run. A parser that copies the rest of its buffer at
//! each block allocates quadratically.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use eidola_engine_chat::json;
use eidola_engine_chat::tool_call::ToolCallParser;
use eidola_engine_chat::{PrefixedCallIds, ToolSchemas};

struct Counting;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is forwarded unchanged to the system allocator.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: the caller's contract, passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract, passed on.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATED.fetch_add(new_size, Ordering::Relaxed);
        // SAFETY: the caller's contract, passed on.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Bytes allocated while parsing `n` blocks given in one push.
fn allocated_parsing(n: usize) -> usize {
    let schemas = ToolSchemas::from_tools(
        &json::parse(
            r#"[{"type":"function","function":{"name":"run","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}}}}]"#,
        )
        .unwrap(),
    );
    let text =
        "<tool_call><function=run><parameter=cmd>a</parameter></function></tool_call>\n".repeat(n);
    let mut parser = ToolCallParser::new(schemas, PrefixedCallIds("c".into()));
    let before = ALLOCATED.load(Ordering::Relaxed);
    let delta = parser.push(&text);
    let allocated = ALLOCATED.load(Ordering::Relaxed) - before;
    assert_eq!(delta.calls.len(), n);
    allocated
}

// The only test in this binary, so nothing else allocates while it counts.
#[test]
fn many_blocks_in_one_push_allocate_linearly() {
    let small = allocated_parsing(2_000);
    let large = allocated_parsing(32_000);
    // 16 times the input: linear work is about 16 times the bytes, quadratic
    // about 256 times.
    assert!(large < small * 32, "{small} -> {large} bytes");
    println!("{small} -> {large} bytes ({}x)", large / small);
}
