//! The host memory a request's parse takes, measured.
//!
//! `api::parse_request` holds the body, its strict parse and its order-preserving parse
//! at once; `eidola_common::engine_deployment::read_slot_bytes` is what a deployment's
//! host-memory check budgets for that per read slot, and `ordered_tree_bytes` what an
//! admitted request keeps. A counting global allocator measures the peak over the
//! densest shapes a body within the JSON value cap
//! (`eidola_common::engine_protocol::MAX_REQUEST_JSON_VALUES`) can take, and each must
//! stay within the budget. This binary holds one test, so nothing else allocates while
//! it measures.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use eidola_common::engine_deployment::{ordered_tree_bytes, read_slot_bytes};
use eidola_common::engine_protocol::{MAX_REQUEST_JSON_VALUES, check_request_json};
use eidola_server_engine::api::parse_request;
use eidola_server_engine::error::ApiError;

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(now, Ordering::SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                let now = CURRENT.fetch_add(new_size - layout.size(), Ordering::SeqCst)
                    + (new_size - layout.size());
                PEAK.fetch_max(now, Ordering::SeqCst);
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Ordering::SeqCst);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// A request whose one tool's `parameters` is `payload` (opaque to the strict types).
fn in_tool_schema(payload: &str) -> String {
    format!(
        r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],"tools":[{{"type":"function","function":{{"name":"f","parameters":{payload}}}}}]}}"#
    )
}

/// A request whose `messages` are `items`.
fn as_messages(items: &str) -> String {
    format!(r#"{{"model":"m","messages":[{items}]}}"#)
}

/// `n` copies of `item`, comma-separated.
fn repeated(item: &str, n: usize) -> String {
    vec![item; n].join(",")
}

/// The largest `n` for which `wrap(n)` stays within the value cap.
fn at_cap(wrap: &dyn Fn(usize) -> String) -> String {
    let (mut lo, mut hi) = (1usize, MAX_REQUEST_JSON_VALUES);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if check_request_json(wrap(mid).as_bytes()).is_ok() {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    wrap(lo)
}

struct Measured {
    body: usize,
    values: usize,
    /// Peak bytes while parsing, the body included.
    peak: usize,
    /// Bytes the accepted request keeps.
    kept: usize,
}

fn measure(body: &str) -> Measured {
    let values = check_request_json(body.as_bytes())
        .expect("within the cap")
        .values;
    let before = CURRENT.load(Ordering::SeqCst);
    PEAK.store(before, Ordering::SeqCst);
    let request = parse_request(body.as_bytes(), "m").expect("a valid request");
    let kept = CURRENT.load(Ordering::SeqCst) - before;
    let peak = PEAK.load(Ordering::SeqCst) - before;
    drop(request);
    Measured {
        body: body.len(),
        values,
        peak: peak + body.len(),
        kept,
    }
}

#[test]
fn the_densest_bodies_stay_within_the_read_and_admission_budgets() {
    let shapes: Vec<(&str, String)> = vec![
        (
            "one-member objects in a tool schema",
            at_cap(&|n| in_tool_schema(&format!("[{}]", repeated(r#"{"":0}"#, n)))),
        ),
        (
            "empty objects in a tool schema",
            at_cap(&|n| in_tool_schema(&format!("[{}]", repeated("{}", n)))),
        ),
        (
            "empty arrays in a tool schema",
            at_cap(&|n| in_tool_schema(&format!("[{}]", repeated("[]", n)))),
        ),
        (
            "nested arrays in a tool schema",
            at_cap(&|n| in_tool_schema(&format!("[{}]", repeated("[[[[[[[[0]]]]]]]]", n)))),
        ),
        (
            "single-element arrays nested 59 deep in a tool schema",
            at_cap(&|n| {
                let one = format!("{}0{}", "[".repeat(59), "]".repeat(59));
                in_tool_schema(&format!("[{}]", repeated(&one, n)))
            }),
        ),
        (
            "single-member objects nested 30 deep in a tool schema",
            at_cap(&|n| {
                let one = format!("{}0{}", r#"{"":"#.repeat(30), "}".repeat(30));
                in_tool_schema(&format!("[{}]", repeated(&one, n)))
            }),
        ),
        (
            "numbers in a tool schema",
            at_cap(&|n| in_tool_schema(&format!("[{}]", repeated("0", n)))),
        ),
        (
            "short strings in a tool schema",
            at_cap(&|n| in_tool_schema(&format!("[{}]", repeated(r#""""#, n)))),
        ),
        (
            "a wide object in a tool schema",
            at_cap(&|n| {
                let members: Vec<String> = (0..n).map(|i| format!(r#""{i:x}":0"#)).collect();
                in_tool_schema(&format!("{{{}}}", members.join(",")))
            }),
        ),
        (
            "minimal messages",
            at_cap(&|n| as_messages(&repeated(r#"{"role":"user"}"#, n))),
        ),
        (
            "text parts",
            at_cap(&|n| {
                as_messages(&format!(
                    r#"{{"role":"user","content":[{}]}}"#,
                    repeated(r#"{"type":"text","text":""}"#, n)
                ))
            }),
        ),
        (
            "tool calls",
            at_cap(&|n| {
                as_messages(&format!(
                    r#"{{"role":"assistant","tool_calls":[{}]}}"#,
                    repeated("{}", n)
                ))
            }),
        ),
        (
            "long text",
            as_messages(&format!(
                r#"{{"role":"user","content":"{}"}}"#,
                "a".repeat(eidola_common::engine_protocol::MAX_REQUEST_BODY_BYTES - 64)
            )),
        ),
        (
            "escaped text",
            as_messages(&format!(
                r#"{{"role":"user","content":"{}"}}"#,
                "\\n".repeat((eidola_common::engine_protocol::MAX_REQUEST_BODY_BYTES - 64) / 2)
            )),
        ),
    ];
    // A per-byte budget cannot bound these: the read slot's budget before the value
    // cap was 129 times the body (1 + 96 for the strict tree + 32 for the ordered one),
    // and nested one-member objects exceed it.
    let mut beyond_per_byte = false;
    for (name, body) in &shapes {
        let m = measure(body);
        beyond_per_byte |= m.peak > 129 * m.body;
        let read = read_slot_bytes(m.body as u64, m.values as u64);
        let kept = ordered_tree_bytes(m.body as u64, m.values as u64);
        println!(
            "{name}: {} bytes, {} values; peak {} ({:.1} per value), budget {read}; kept {}, budget {kept}",
            m.body,
            m.values,
            m.peak,
            m.peak as f64 / m.values as f64,
            m.kept,
        );
        assert!(m.peak as u64 <= read, "{name}: peak {} > {read}", m.peak);
        assert!(m.kept as u64 <= kept, "{name}: kept {} > {kept}", m.kept);
    }
    assert!(beyond_per_byte, "no shape exceeds the per-byte budget");

    // One value past the cap is refused before either parse allocates.
    let over = in_tool_schema(&format!("[{}]", repeated("0", MAX_REQUEST_JSON_VALUES)));
    let before = CURRENT.load(Ordering::SeqCst);
    PEAK.store(before, Ordering::SeqCst);
    let err = parse_request(over.as_bytes(), "m").unwrap_err();
    let peak = PEAK.load(Ordering::SeqCst) - before;
    assert!(
        matches!(&err, ApiError::InvalidRequest(m) if m.contains("JSON values")),
        "{err:?}"
    );
    assert!(peak < 4096, "the refusal allocated {peak} bytes");
}
