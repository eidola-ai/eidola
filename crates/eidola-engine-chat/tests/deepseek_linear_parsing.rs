//! One chunk holding many DSML elements is parsed with work linear in its
//! length: many blocks, many calls in one block, many parameters in one call.
//!
//! The measure is bytes allocated, counted by this binary's allocator, as in
//! `linear_parsing.rs`; the time bound catches a scan that allocates nothing.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use eidola_engine_chat::deepseek_v4_1::output::DsmlParser;
use eidola_engine_chat::deepseek_v4_1::tool_schemas;
use eidola_engine_chat::{PrefixedCallIds, json};

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

const INVOKE: &str = "<｜DSML｜ invoke name=\"run\">\n<｜DSML｜ parameter name=\"cmd\" string=\"true\">a</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n";

/// The three shapes at size `n`, each with the number of calls it holds.
fn shapes(n: usize) -> [(String, usize); 3] {
    let blocks = format!("\n\n<｜DSML｜ calls>\n{INVOKE}</｜DSML｜ calls>").repeat(n);
    let calls = format!(
        "\n\n<｜DSML｜ calls>\n{}</｜DSML｜ calls>",
        INVOKE.repeat(n)
    );
    let parameters: String = (0..n)
        .map(|i| {
            format!("<｜DSML｜ parameter name=\"p{i}\" string=\"false\">{i}</｜DSML｜ parameter>\n")
        })
        .collect();
    let parameters = format!(
        "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"run\">\n{parameters}</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    );
    [(blocks, n), (calls, n), (parameters, 1)]
}

/// Bytes allocated and seconds spent parsing `text` given in one push.
fn parsing(text: &str, calls: usize) -> (usize, f64) {
    let schemas =
        tool_schemas(&json::parse(r#"[{"type":"function","function":{"name":"run"}}]"#).unwrap())
            .unwrap();
    let mut parser = DsmlParser::new(schemas, PrefixedCallIds("c".into()));
    let before = ALLOCATED.load(Ordering::Relaxed);
    let start = Instant::now();
    let delta = parser.push(text);
    let seconds = start.elapsed().as_secs_f64();
    let allocated = ALLOCATED.load(Ordering::Relaxed) - before;
    assert_eq!(delta.calls.len(), calls);
    (allocated, seconds)
}

// The only test in this binary, so nothing else allocates while it counts.
#[test]
fn many_elements_in_one_push_parse_linearly() {
    let (small, large) = (shapes(2_000), shapes(32_000));
    for (shape, ((small, small_calls), (large, large_calls))) in
        small.iter().zip(&large).enumerate()
    {
        let (small_bytes, _) = parsing(small, *small_calls);
        let (large_bytes, seconds) = parsing(large, *large_calls);
        // 16 times the input: linear work is about 16 times the bytes,
        // quadratic about 256 times, and seconds of scanning.
        assert!(
            large_bytes < small_bytes * 32,
            "shape {shape}: {small_bytes} -> {large_bytes} bytes"
        );
        assert!(seconds < 2.0, "shape {shape}: {seconds} s");
        println!("shape {shape}: {small_bytes} -> {large_bytes} bytes, {seconds:.3} s");
    }
}
