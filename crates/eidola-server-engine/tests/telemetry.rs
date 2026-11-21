//! The telemetry snapshot in `/v1/engine/info`: strict in the shared contract's shape,
//! padded like every other answer, and moved by the engine's real activity, over the
//! CPU executor and the dev model.

mod common;

use std::time::Duration;

use common::*;
use eidola_common::engine_protocol::telemetry::{
    INFO_TELEMETRY_MEMBER, NodeTelemetry, PoolOccupancy, quantize_prefix_tokens,
};
use eidola_server_engine::config::env;
use futures_util::StreamExt;
use serde_json::Value;

/// The node's info answer: its padded size, and its snapshot read by the gateway's
/// strict rule.
async fn info(node: &TestNode) -> (usize, NodeTelemetry) {
    let r = node
        .client
        .get(format!("{}/v1/engine/info", node.base))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let declared = r.content_length();
    let bytes = r.bytes().await.unwrap();
    assert_eq!(declared, Some(bytes.len() as u64));
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    let snapshot = NodeTelemetry::from_value(&v[INFO_TELEMETRY_MEMBER])
        .unwrap_or_else(|| panic!("an unreadable snapshot: {}", v[INFO_TELEMETRY_MEMBER]));
    (bytes.len(), snapshot)
}

/// The first snapshot `pred` holds for, read again until it does: the pools are
/// recounted at most once a second.
async fn info_until(node: &TestNode, pred: impl Fn(&NodeTelemetry) -> bool) -> NodeTelemetry {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let (_, snapshot) = info(node).await;
        if pred(&snapshot) {
            return snapshot;
        }
        assert!(std::time::Instant::now() < deadline, "{snapshot:?}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_fresh_node_reports_an_empty_snapshot_padded() {
    let node = TestNode::start(&[]).await;
    let (size, snapshot) = info(&node).await;
    assert!(
        size.is_power_of_two() && size >= eidola_common::stream_shape::MIN_BODY_BUCKET,
        "{size} bytes"
    );
    assert!(
        snapshot.counters.entries().iter().all(|&(_, v)| v == 0),
        "{:?}",
        snapshot.counters
    );
    assert_eq!(
        (
            snapshot.gauges.sequences_waiting,
            snapshot.gauges.sequences_running
        ),
        (0, 0)
    );
    // The MiMo-V2.6 dev model's KV groups (global, sliding, and its MTP layer's), each a
    // pool of KV_BLOCKS less the null block, all free, with no host tier.
    assert!(!snapshot.gauges.pools.is_empty());
    for pool in &snapshot.gauges.pools {
        assert_eq!(
            *pool,
            PoolOccupancy {
                total: 255,
                ..PoolOccupancy::default()
            }
        );
    }
    // Another process would have another boot id; this one keeps its own.
    let (_, again) = info(&node).await;
    assert_eq!(again.boot_id, snapshot.boot_id);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_snapshot_follows_the_engine() {
    let node = TestNode::start(&[(env::MAX_REQUESTS, "1")]).await;
    let (_, before) = info(&node).await;

    // Two requests under one cache key: the second finds the first's prompt cached.
    let mut keyed = request(&"a story about a lighthouse keeper ".repeat(6), 4);
    keyed["cache_key"] = "A".repeat(43).into();
    for _ in 0..2 {
        let (status, v) = node.chat_json(&keyed).await;
        assert_eq!(status, 200, "{v}");
    }
    let s = node.stats();
    assert!(
        s.prefix_hit_tokens > 0,
        "the second request hit the cache: {s:?}"
    );

    // While a long request runs, it is a running sequence, and the only admission slot
    // is taken, so the next request is refused `overloaded`.
    let long = request("Tell me a very long story.", 1000);
    let running = node.chat(&long).send().await.unwrap();
    let mut running = running.bytes_stream();
    running.next().await.unwrap().unwrap();
    node.wait_for(Duration::from_secs(10), |s| s.running == 1)
        .await;
    let during = info_until(&node, |t| t.gauges.pools.iter().any(|p| p.in_use > 0)).await;
    assert_eq!(during.gauges.sequences_running, 1);
    let (status, v) = node.chat_json(&request("hi", 2)).await;
    assert_eq!(status, 503);
    assert_eq!(v["error"]["type"], "overloaded");
    drop(running);
    node.wait_for(Duration::from_secs(10), |s| s.in_engine == 0)
        .await;
    // The count that follows the last request.
    let after = info_until(&node, |t| t.gauges.pools.iter().all(|p| p.in_use == 0)).await;

    let s = node.stats();
    let c = after.counters;
    assert_eq!(after.boot_id, before.boot_id);
    assert_eq!(c.requests_accepted, 3);
    assert_eq!(c.requests_accepted, s.submitted);
    assert_eq!(c.requests_finished, 2);
    assert_eq!(c.requests_cancelled, 1);
    assert_eq!(c.requests_overloaded, 1);
    assert_eq!(c.prefix_lookups, 3);
    assert!(c.steps > 0 && c.step_rows >= c.steps, "{c:?}");
    assert!(c.step_micros > 0, "{c:?}");
    assert!(c.generated_tokens >= 2 * 4, "{c:?}");
    assert!(c.prefill_tokens > 0, "{c:?}");
    assert!(c.drafted_tokens >= c.accepted_drafts, "{c:?}");
    // The prefix cache's token counts leave the node floored; the engine's own are
    // exact.
    assert_eq!(
        c.prefix_hit_tokens,
        quantize_prefix_tokens(s.prefix_hit_tokens)
    );
    assert_eq!(
        c.prefix_miss_tokens,
        quantize_prefix_tokens(s.prefix_miss_tokens)
    );
    assert!(after.counters.covers(&during.counters));
    assert_eq!(
        (
            after.gauges.sequences_waiting,
            after.gauges.sequences_running
        ),
        (0, 0)
    );
    // The keyed requests' blocks stay cached, idle, under their key; the pool size
    // here is small enough to be reported exactly.
    assert!(
        after.gauges.pools.iter().any(|p| p.idle > 0),
        "{:?}",
        after.gauges.pools
    );
    let exact = node.engine.pools();
    assert_eq!(exact.len(), after.gauges.pools.len());
    for (e, q) in exact.iter().zip(&after.gauges.pools) {
        assert_eq!(
            *q,
            PoolOccupancy::quantized(e.total, e.in_use, e.idle, e.host_total, e.host_used)
        );
    }
}
