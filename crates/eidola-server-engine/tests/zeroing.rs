//! The node end to end under the binary's global allocator (`zeroing::ZeroOnFree`), which
//! a test binary installs for itself: the CPU executor computes exactly what it computes
//! without it (the allocator writes only into blocks it is freeing, and hands out the
//! system allocator's blocks unchanged), and `ZEROING_BENCH=1` times a run of requests for
//! the allocator's cost (run it here, and as `tests/server.rs`-style code without the
//! allocator, to compare).

mod common;

use std::time::Instant;

use common::*;
use eidola_server_engine::zeroing::ZeroOnFree;

#[global_allocator]
static ALLOCATOR: ZeroOnFree<std::alloc::System> = ZeroOnFree(std::alloc::System);

/// Greedy completions under the zeroing allocator are the reference executor's: the same
/// request answers the same tokens every time, and as many as asked for.
#[tokio::test(flavor = "multi_thread")]
async fn the_cpu_executor_computes_the_same_under_the_zeroing_allocator() {
    let node = TestNode::start(&[]).await;
    let mut body = request("The quick brown fox jumps over the lazy dog.", 24);
    body["temperature"] = 0.0.into();
    let answer = |c: serde_json::Value| (c["choices"].clone(), c["usage"].clone());
    let first = answer(completion(&node.chat_stream(&body).await));
    for _ in 0..3 {
        assert_eq!(answer(completion(&node.chat_stream(&body).await)), first);
    }
    assert_eq!(first.1["completion_tokens"], 24, "{first:?}");
    if std::env::var_os("ZEROING_BENCH").is_some() {
        let start = Instant::now();
        for _ in 0..20 {
            completion(&node.chat_stream(&body).await);
        }
        eprintln!("20 requests under ZeroOnFree: {:?}", start.elapsed());
    }
}
