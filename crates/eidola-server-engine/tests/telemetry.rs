//! The node's own telemetry: what its instruments observe, moved by the engine's real
//! activity over the CPU executor and the dev model, coarsened as `telemetry.rs`
//! states, and exported through an OpenTelemetry meter provider; and the instance id
//! its info answer carries.

mod common;

use std::time::Duration;

use common::*;
use eidola_server_engine::config::env;
use eidola_server_engine::telemetry::{
    PROMPT_TOKEN_QUANTUM, PoolOccupancy, STEP_MICROS_QUANTUM, STEP_QUANTUM, Value, floor_to,
};
use futures_util::StreamExt;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};

/// A count the node observes for `instrument` in the series with `labels`.
fn count(node: &TestNode, instrument: &str, labels: &[(&str, &str)]) -> u64 {
    match node.observed(instrument, labels) {
        Some(Value::Count(n)) => n,
        other => panic!("{instrument} {labels:?}: {other:?}"),
    }
}

/// The first moment `pred` holds for what the node observes: the pools are recounted
/// at most once a second.
async fn observe_until(node: &TestNode, pred: impl Fn(&TestNode) -> bool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !pred(node) {
        assert!(std::time::Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Each pool's quantized occupancy, from the `kv.blocks` gauge's series.
fn pools(node: &TestNode) -> Vec<PoolOccupancy> {
    (0..8)
        .map(|i| i.to_string())
        .take_while(|pool| {
            node.observed("engine.node.kv.blocks", &[("pool", pool)])
                .is_some()
        })
        .map(|pool| {
            let state = |s| {
                count(
                    node,
                    "engine.node.kv.blocks",
                    &[("pool", &pool), ("state", s)],
                )
            };
            let (in_use, idle, free) = (state("in_use"), state("idle"), state("free"));
            PoolOccupancy {
                total: in_use + idle + free,
                in_use,
                idle,
                ..PoolOccupancy::default()
            }
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_fresh_node_observes_every_series_at_zero_and_names_its_instance() {
    let node = TestNode::start(&[]).await;
    let observed = node.telemetry.observe();
    assert!(!observed.is_empty());
    for o in &observed {
        assert_eq!(o.attributes[0].key.as_str(), "model");
        assert_eq!(o.attributes[0].value.to_string(), node.weights);
        if !o.instrument.starts_with("engine.node.kv.") {
            assert!(
                matches!(o.value, Value::Count(0)) || o.value == Value::Seconds(0.0),
                "{o:?}"
            );
        }
    }
    // The MiMo-V2.6 dev model's KV groups (global, sliding, and its MTP layer's), each a
    // pool of KV_BLOCKS less the null block, all free, with no host tier.
    let pools = pools(&node);
    assert!(!pools.is_empty());
    for pool in &pools {
        assert_eq!(
            *pool,
            PoolOccupancy {
                total: 255,
                ..PoolOccupancy::default()
            }
        );
    }
    assert!(node.observed("engine.node.kv.host_blocks", &[]).is_none());

    // The info answer names the process, padded like every other answer, and carries
    // nothing of the node's activity.
    let r = node
        .client
        .get(format!("{}/v1/engine/info", node.base))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let bytes = r.bytes().await.unwrap();
    assert!(
        bytes.len().is_power_of_two()
            && bytes.len() >= eidola_common::stream_shape::MIN_BODY_BUCKET
    );
    let info: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        info["instance_id"],
        eidola_server_engine::telemetry::instance_id()
    );
    assert!(info.get("telemetry").is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn the_counts_follow_the_engine_coarsened() {
    let node = TestNode::start(&[(env::MAX_REQUESTS, "1")]).await;

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
    observe_until(&node, |n| pools(n).iter().any(|p| p.in_use > 0)).await;
    assert_eq!(
        count(&node, "engine.node.sequences", &[("state", "running")]),
        1
    );
    let (status, v) = node.chat_json(&request("hi", 2)).await;
    assert_eq!(status, 503);
    assert_eq!(v["error"]["type"], "overloaded");
    drop(running);
    node.wait_for(Duration::from_secs(10), |s| s.in_engine == 0)
        .await;
    observe_until(&node, |n| pools(n).iter().all(|p| p.in_use == 0)).await;

    let s = node.stats();
    let requests = |outcome| count(&node, "engine.node.requests", &[("outcome", outcome)]);
    assert_eq!(requests("accepted"), 3);
    assert_eq!(requests("accepted"), s.submitted);
    assert_eq!(requests("finished"), 2);
    assert_eq!(requests("cancelled"), 1);
    assert_eq!(requests("overloaded"), 1);
    assert_eq!(count(&node, "engine.node.prefix_cache.lookups", &[]), 3);
    let generated = count(&node, "engine.node.tokens", &[("type", "generated")]);
    assert!(generated >= 2 * 4, "{generated}");
    // The step, step time and prompt counts are observed floored, each to its quantum
    // (at this little work, mostly to zero); the request and token counts are exact.
    assert!(
        s.forward_steps > 0 && s.step_rows >= s.forward_steps && s.step_micros > 0,
        "{s:?}"
    );
    assert!(s.prefill_tokens > 0, "{s:?}");
    assert_eq!(
        count(&node, "engine.node.steps", &[]),
        floor_to(s.forward_steps, STEP_QUANTUM)
    );
    assert_eq!(
        count(&node, "engine.node.step.rows", &[]),
        floor_to(s.step_rows, STEP_QUANTUM)
    );
    assert_eq!(
        node.observed("engine.node.step.time", &[]),
        Some(Value::Seconds(
            floor_to(s.step_micros, STEP_MICROS_QUANTUM) as f64 / 1e6
        ))
    );
    assert_eq!(
        count(&node, "engine.node.tokens", &[("type", "prefill")]),
        floor_to(s.prefill_tokens, PROMPT_TOKEN_QUANTUM)
    );
    for (result, exact) in [("hit", s.prefix_hit_tokens), ("miss", s.prefix_miss_tokens)] {
        assert_eq!(
            count(
                &node,
                "engine.node.prefix_cache.tokens",
                &[("result", result)]
            ),
            floor_to(exact, PROMPT_TOKEN_QUANTUM)
        );
    }
    assert_eq!(
        (
            count(&node, "engine.node.sequences", &[("state", "waiting")]),
            count(&node, "engine.node.sequences", &[("state", "running")])
        ),
        (0, 0)
    );
    // The keyed requests' blocks stay cached, idle, under their key; the pool size here
    // is small enough to be reported exactly.
    let observed = pools(&node);
    assert!(observed.iter().any(|p| p.idle > 0), "{observed:?}");
    let exact = node.engine.pools();
    assert_eq!(exact.len(), observed.len());
    for (e, q) in exact.iter().zip(&observed) {
        let quantized = PoolOccupancy::quantized(e.total, e.in_use, e.idle, 0, 0);
        assert_eq!(
            (q.total, q.in_use, q.idle),
            (quantized.total, quantized.in_use, quantized.idle)
        );
    }
}

/// Registered on a meter provider, the node's instruments export what the node
/// observes, every series labelled by the model, and are read only when the provider
/// collects: nothing is recorded per request.
#[tokio::test(flavor = "multi_thread")]
async fn the_instruments_export_what_the_node_observes() {
    let node = TestNode::start(&[]).await;
    let exporter = opentelemetry_sdk::metrics::InMemoryMetricExporter::default();
    let provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
        .with_reader(
            opentelemetry_sdk::metrics::PeriodicReader::builder(exporter.clone())
                .with_interval(Duration::from_secs(3600))
                .build(),
        )
        .build();
    node.telemetry
        .register(&opentelemetry::metrics::MeterProvider::meter(
            &provider, "test",
        ));

    let (status, v) = node.chat_json(&request("hi", 2)).await;
    assert_eq!(status, 200, "{v}");
    assert!(
        exporter.get_finished_metrics().unwrap().is_empty(),
        "nothing is exported between collections"
    );
    provider.force_flush().unwrap();

    let exported = exporter.get_finished_metrics().unwrap();
    let metrics: Vec<_> = exported
        .iter()
        .flat_map(|r| r.scope_metrics())
        .flat_map(|s| s.metrics())
        .collect();
    let finished = metrics
        .iter()
        .find(|m| m.name() == "engine.node.requests")
        .and_then(|m| match m.data() {
            AggregatedMetrics::U64(MetricData::Sum(sum)) => sum
                .data_points()
                .find(|p| {
                    p.attributes().any(|kv| {
                        kv.key.as_str() == "outcome" && kv.value.to_string() == "finished"
                    })
                })
                .map(|p| {
                    assert!(p.attributes().any(
                        |kv| kv.key.as_str() == "model" && kv.value.to_string() == node.weights
                    ));
                    p.value()
                }),
            _ => None,
        });
    assert_eq!(finished, Some(1));
    for name in [
        "engine.node.steps",
        "engine.node.step.time",
        "engine.node.tokens",
        "engine.node.prefix_cache.tokens",
        "engine.node.sequences",
        "engine.node.kv.blocks",
    ] {
        assert!(
            metrics.iter().any(|m| m.name() == name),
            "{name} was not exported"
        );
    }
    assert!(
        !metrics
            .iter()
            .any(|m| m.name() == "engine.node.kv.host_blocks"),
        "a node without a host tier has no host-tier series"
    );
    provider.shutdown().unwrap();
}
