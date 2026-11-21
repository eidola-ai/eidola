//! What the gateway exports for the engine nodes it routes to.
//!
//! A node has no egress: its measured config allows the gateway's connection
//! and nothing else. So its operational snapshot
//! (`eidola_common::engine_protocol::telemetry`) rides on the probe: the
//! router reads it from the node's `/v1/engine/info` answer on each probe,
//! over the attested, token-authenticated connection the probe already uses,
//! and this module turns it into the gateway's own instruments, under
//! `engine.node.*`, labelled by the model's pinned weights hash and fixed
//! label values only: never an upstream's URL or deployment.
//!
//! - **Counters** are the node's counts since its boot. Each upstream keeps
//!   the last snapshot's counts and boot id ([`NodeState`]), and each probe
//!   adds the growth since then. A changed boot id is a restart: the new
//!   boot's counts so far are all growth, and `engine.node.restarts` counts
//!   it. The first snapshot from an upstream is only a baseline, so a
//!   gateway that starts, or an upstream that placement adds, never counts a
//!   node's history as one probe interval's work.
//! - **Gauges** are each upstream's last snapshot, summed over a model's
//!   upstreams at export. An upstream whose probe fails stops reporting until
//!   one passes, and `engine.node.reporting` says how many did.
//! - **A snapshot that does not parse** is counted
//!   (`engine.node.snapshots{outcome=unparsed}`), never logged, and changes
//!   nothing else: telemetry never decides whether a node takes traffic. A
//!   node built before the snapshot answers without it (`outcome=absent`).
//!
//! The state is per upstream, so it is bounded by placement, and goes with
//! the upstream when placement drops it.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use eidola_common::engine_protocol::telemetry::{
    BOOT_ID_BYTES, INFO_TELEMETRY_MEMBER, MAX_POOLS, NodeCounters, NodeGauges, NodeTelemetry,
    PoolOccupancy,
};
use opentelemetry::KeyValue;
use opentelemetry::metrics::Counter;

use super::EngineRouter;

/// What one upstream's snapshots have left behind.
#[derive(Debug, Default)]
pub(crate) struct NodeState {
    /// The last snapshot's boot id and counts.
    baseline: Option<([u8; BOOT_ID_BYTES], NodeCounters)>,
    /// The last snapshot's levels, while the upstream's probes pass.
    gauges: Option<NodeGauges>,
}

/// What a snapshot adds to the counters.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Growth {
    pub counters: NodeCounters,
    /// The node restarted since the last snapshot.
    pub restarted: bool,
}

impl NodeState {
    /// Takes `snapshot` as the upstream's latest, returning what it adds to
    /// the counters.
    pub(crate) fn observe(&mut self, snapshot: NodeTelemetry) -> Growth {
        let growth = match &self.baseline {
            None => Growth {
                counters: NodeCounters::default(),
                restarted: false,
            },
            Some((boot, earlier)) if *boot == snapshot.boot_id => Growth {
                counters: snapshot.counters.since(earlier),
                restarted: false,
            },
            Some(_) => Growth {
                counters: snapshot.counters,
                restarted: true,
            },
        };
        self.baseline = Some((snapshot.boot_id, snapshot.counters));
        self.gauges = Some(snapshot.gauges);
        growth
    }

    /// The upstream failed its probe: its levels are no longer reported. Its
    /// counts stay as the baseline for its next snapshot.
    pub(crate) fn lapse(&mut self) {
        self.gauges = None;
    }

    pub(crate) fn gauges(&self) -> Option<&NodeGauges> {
        self.gauges.as_ref()
    }
}

/// How the snapshot in a node's info answer read.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Reading {
    Read(Box<NodeTelemetry>),
    /// The answer has no snapshot (a node built before it).
    Absent,
    /// The snapshot is not in the contract's shape.
    Unparsed,
}

impl Reading {
    /// The snapshot in a node's info answer.
    pub(crate) fn of(info: &serde_json::Value) -> Self {
        match info.get(INFO_TELEMETRY_MEMBER) {
            None => Reading::Absent,
            Some(value) => NodeTelemetry::from_value(value)
                .map_or(Reading::Unparsed, |t| Reading::Read(Box::new(t))),
        }
    }

    fn outcome(&self) -> &'static str {
        match self {
            Reading::Read(_) => "read",
            Reading::Absent => "absent",
            Reading::Unparsed => "unparsed",
        }
    }
}

/// A model's levels, summed over its upstreams that reported one.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct ModelGauges {
    /// Upstreams whose last probe passed with a snapshot.
    pub reporting: u64,
    pub sequences_waiting: u64,
    pub sequences_running: u64,
    /// Pool by pool, in each node's pool order.
    pub pools: Vec<PoolOccupancy>,
}

impl ModelGauges {
    pub(crate) fn add(&mut self, gauges: &NodeGauges) {
        self.reporting += 1;
        self.sequences_waiting += gauges.sequences_waiting;
        self.sequences_running += gauges.sequences_running;
        if self.pools.len() < gauges.pools.len() {
            self.pools
                .resize(gauges.pools.len(), PoolOccupancy::default());
        }
        for (sum, pool) in self.pools.iter_mut().zip(&gauges.pools) {
            sum.total += pool.total;
            sum.in_use += pool.in_use;
            sum.idle += pool.idle;
            sum.host_total += pool.host_total;
            sum.host_used += pool.host_used;
        }
    }
}

/// The `pool` label: a pool's index in its node's order, from this fixed
/// list.
const POOL_LABELS: [&str; MAX_POOLS] = ["0", "1", "2", "3", "4", "5", "6", "7"];

/// Where each counter of the snapshot is exported: the instrument, and the
/// fixed label that tells it from the other members it shares one with.
/// `step_micros` is exported apart, as seconds ([`STEP_TIME`]).
type CounterExport = (
    &'static str,
    &'static str,
    Option<(&'static str, &'static str)>,
);

const COUNTER_EXPORTS: &[CounterExport] = &[
    (
        "requests_accepted",
        "engine.node.requests",
        Some(("outcome", "accepted")),
    ),
    (
        "requests_finished",
        "engine.node.requests",
        Some(("outcome", "finished")),
    ),
    (
        "requests_cancelled",
        "engine.node.requests",
        Some(("outcome", "cancelled")),
    ),
    (
        "requests_cut_off",
        "engine.node.requests",
        Some(("outcome", "cut_off")),
    ),
    (
        "requests_rejected",
        "engine.node.requests",
        Some(("outcome", "rejected")),
    ),
    (
        "requests_prefix_refused",
        "engine.node.requests",
        Some(("outcome", "prefix_refused")),
    ),
    (
        "requests_overloaded",
        "engine.node.requests",
        Some(("outcome", "overloaded")),
    ),
    ("steps", "engine.node.steps", None),
    ("step_micros", "engine.node.step.time", None),
    ("step_rows", "engine.node.step.rows", None),
    (
        "prefill_tokens",
        "engine.node.tokens",
        Some(("type", "prefill")),
    ),
    (
        "generated_tokens",
        "engine.node.tokens",
        Some(("type", "generated")),
    ),
    (
        "drafted_tokens",
        "engine.node.drafts",
        Some(("outcome", "proposed")),
    ),
    (
        "accepted_drafts",
        "engine.node.drafts",
        Some(("outcome", "accepted")),
    ),
    ("preemptions", "engine.node.preemptions", None),
    ("prefix_lookups", "engine.node.prefix_cache.lookups", None),
    (
        "prefix_hit_tokens",
        "engine.node.prefix_cache.tokens",
        Some(("result", "hit")),
    ),
    (
        "prefix_miss_tokens",
        "engine.node.prefix_cache.tokens",
        Some(("result", "miss")),
    ),
    (
        "cache_device_evictions",
        "engine.node.prefix_cache.evictions",
        Some(("tier", "device")),
    ),
    (
        "cache_host_evictions",
        "engine.node.prefix_cache.evictions",
        Some(("tier", "host")),
    ),
    (
        "cache_demoted_blocks",
        "engine.node.prefix_cache.transfers",
        Some(("direction", "demoted")),
    ),
    (
        "cache_promoted_blocks",
        "engine.node.prefix_cache.transfers",
        Some(("direction", "promoted")),
    ),
];

/// Each `u64` counter instrument with its description.
const COUNTER_INSTRUMENTS: &[(&str, &str)] = &[
    (
        "engine.node.requests",
        "Requests on the engine nodes, by outcome, from their snapshots",
    ),
    (
        "engine.node.steps",
        "Engine steps that ran at least one sequence",
    ),
    (
        "engine.node.step.rows",
        "Sequences run by engine steps, summed over the steps",
    ),
    (
        "engine.node.tokens",
        "Tokens the engine nodes computed, by type: prefill or generated",
    ),
    (
        "engine.node.drafts",
        "Speculative draft tokens on the engine nodes, proposed and accepted",
    ),
    (
        "engine.node.preemptions",
        "Running sequences preempted to make room",
    ),
    (
        "engine.node.prefix_cache.lookups",
        "Prefix-cache lookups at requests' first admission",
    ),
    (
        "engine.node.prefix_cache.tokens",
        "Prompt tokens the prefix cache held (hit) or did not (miss), in steps of 1024",
    ),
    (
        "engine.node.prefix_cache.evictions",
        "Prefix-cache entries dropped to make room, by tier",
    ),
    (
        "engine.node.prefix_cache.transfers",
        "Prefix-cache blocks moved between the device and the host tier",
    ),
    (
        "engine.node.restarts",
        "Engine node restarts seen by the probe",
    ),
    (
        "engine.node.snapshots",
        "Engine node telemetry snapshots read by the probe, by outcome",
    ),
];

fn meter() -> opentelemetry::metrics::Meter {
    opentelemetry::global::meter("eidola-server-gateway")
}

static COUNTERS: LazyLock<BTreeMap<&'static str, Counter<u64>>> = LazyLock::new(|| {
    COUNTER_INSTRUMENTS
        .iter()
        .map(|&(name, description)| {
            (
                name,
                meter()
                    .u64_counter(name)
                    .with_description(description)
                    .build(),
            )
        })
        .collect()
});

/// Wall time of the engine nodes' steps that ran a sequence, in seconds.
static STEP_TIME: LazyLock<Counter<f64>> = LazyLock::new(|| {
    meter()
        .f64_counter("engine.node.step.time")
        .with_description("Wall time of engine steps that ran at least one sequence")
        .with_unit("s")
        .build()
});

fn count(instrument: &str, value: u64, labels: &[KeyValue]) {
    if let Some(counter) = COUNTERS.get(instrument) {
        counter.add(value, labels);
    }
}

/// Exports what one probe's snapshot read for `model`, a pinned weights hash.
pub(crate) fn record(model: &str, reading: &Reading, growth: Option<&Growth>) {
    let model_label = KeyValue::new("model", model.to_string());
    count(
        "engine.node.snapshots",
        1,
        &[
            model_label.clone(),
            KeyValue::new("outcome", reading.outcome()),
        ],
    );
    let Some(growth) = growth else { return };
    if growth.restarted {
        count(
            "engine.node.restarts",
            1,
            std::slice::from_ref(&model_label),
        );
    }
    for (member, value) in growth.counters.entries() {
        if value == 0 {
            continue;
        }
        if member == "step_micros" {
            STEP_TIME.add(value as f64 / 1e6, std::slice::from_ref(&model_label));
            continue;
        }
        let Some(&(_, instrument, label)) = COUNTER_EXPORTS.iter().find(|(m, ..)| *m == member)
        else {
            continue;
        };
        match label {
            Some((key, value_label)) => count(
                instrument,
                value,
                &[model_label.clone(), KeyValue::new(key, value_label)],
            ),
            None => count(instrument, value, std::slice::from_ref(&model_label)),
        }
    }
}

/// Registers the observable gauges read from the router at each export,
/// for every pinned model (zeros included): `engine.node.reporting`,
/// `engine.node.sequences{state=waiting|running}`,
/// `engine.node.kv.blocks{pool, state=in_use|idle|free}`, and, for a model
/// whose nodes have a host tier, `engine.node.kv.host_blocks{pool,
/// state=used|free}`.
pub fn register(router: EngineRouter) {
    let reporting = router.clone();
    meter()
        .u64_observable_gauge("engine.node.reporting")
        .with_description("Engine upstreams whose last probe read a telemetry snapshot")
        .with_callback(move |gauge| {
            for (model, sums) in reporting.node_gauges() {
                gauge.observe(sums.reporting, &[KeyValue::new("model", model)]);
            }
        })
        .build();
    let sequences = router.clone();
    meter()
        .u64_observable_gauge("engine.node.sequences")
        .with_description("Requests in the engine nodes, waiting for a seat or running")
        .with_callback(move |gauge| {
            for (model, sums) in sequences.node_gauges() {
                for (state, value) in [
                    ("waiting", sums.sequences_waiting),
                    ("running", sums.sequences_running),
                ] {
                    gauge.observe(
                        value,
                        &[
                            KeyValue::new("model", model.clone()),
                            KeyValue::new("state", state),
                        ],
                    );
                }
            }
        })
        .build();
    let blocks = router.clone();
    meter()
        .u64_observable_gauge("engine.node.kv.blocks")
        .with_description("Engine KV pool blocks, by pool and state, rounded on the node")
        .with_callback(move |gauge| {
            for (model, sums) in blocks.node_gauges() {
                for (pool, sum) in POOL_LABELS.iter().zip(&sums.pools) {
                    for (state, value) in [
                        ("in_use", sum.in_use),
                        ("idle", sum.idle),
                        ("free", sum.free()),
                    ] {
                        gauge.observe(
                            value,
                            &[
                                KeyValue::new("model", model.clone()),
                                KeyValue::new("pool", *pool),
                                KeyValue::new("state", state),
                            ],
                        );
                    }
                }
            }
        })
        .build();
    meter()
        .u64_observable_gauge("engine.node.kv.host_blocks")
        .with_description("Engine host-tier blocks, by pool and state, rounded on the node")
        .with_callback(move |gauge| {
            for (model, sums) in router.node_gauges() {
                for (pool, sum) in POOL_LABELS.iter().zip(&sums.pools) {
                    if sum.host_total == 0 {
                        continue;
                    }
                    for (state, value) in [("used", sum.host_used), ("free", sum.host_free())] {
                        gauge.observe(
                            value,
                            &[
                                KeyValue::new("model", model.clone()),
                                KeyValue::new("pool", *pool),
                                KeyValue::new("state", state),
                            ],
                        );
                    }
                }
            }
        })
        .build();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(boot: u8, steps: u64, running: u64) -> NodeTelemetry {
        NodeTelemetry {
            boot_id: [boot; BOOT_ID_BYTES],
            counters: NodeCounters {
                steps,
                generated_tokens: steps * 3,
                ..NodeCounters::default()
            },
            gauges: NodeGauges {
                sequences_waiting: 0,
                sequences_running: running,
                pools: vec![PoolOccupancy {
                    total: 100,
                    in_use: running * 10,
                    idle: 5,
                    host_total: 0,
                    host_used: 0,
                }],
            },
        }
    }

    #[test]
    fn every_counter_is_exported_once() {
        let exported: Vec<&str> = COUNTER_EXPORTS.iter().map(|(m, ..)| *m).collect();
        assert_eq!(exported, NodeCounters::MEMBERS);
        for (_, instrument, _) in COUNTER_EXPORTS {
            assert!(
                *instrument == "engine.node.step.time"
                    || COUNTER_INSTRUMENTS
                        .iter()
                        .any(|(name, _)| name == instrument),
                "{instrument} has no instrument"
            );
        }
        // Members sharing an instrument are told apart by distinct labels.
        for (i, (m, instrument, label)) in COUNTER_EXPORTS.iter().enumerate() {
            for (other, other_instrument, other_label) in &COUNTER_EXPORTS[i + 1..] {
                if instrument == other_instrument {
                    assert!(
                        label.is_some() && label != other_label,
                        "{m} and {other} share {instrument}"
                    );
                }
            }
        }
    }

    /// The first snapshot is a baseline; later ones add their growth; a new
    /// boot id is a restart whose counts so far are all growth.
    #[test]
    fn counters_grow_across_probes_and_restarts() {
        let mut state = NodeState::default();
        let first = state.observe(snapshot(1, 100, 1));
        assert_eq!(first.counters, NodeCounters::default());
        assert!(!first.restarted);

        let second = state.observe(snapshot(1, 130, 2));
        assert!(!second.restarted);
        assert_eq!(second.counters.steps, 30);
        assert_eq!(second.counters.generated_tokens, 90);

        // Restarted: counts start again from zero under a new boot id.
        let third = state.observe(snapshot(2, 7, 0));
        assert!(third.restarted);
        assert_eq!(third.counters.steps, 7);
        assert_eq!(third.counters.generated_tokens, 21);

        let fourth = state.observe(snapshot(2, 10, 0));
        assert!(!fourth.restarted);
        assert_eq!(fourth.counters.steps, 3);

        // A count that went backward under one boot id adds nothing.
        let fifth = state.observe(snapshot(2, 4, 0));
        assert_eq!(fifth.counters, NodeCounters::default());
        let sixth = state.observe(snapshot(2, 6, 0));
        assert_eq!(sixth.counters.steps, 2);
    }

    #[test]
    fn a_lapsed_upstream_stops_reporting_levels_but_keeps_its_baseline() {
        let mut state = NodeState::default();
        state.observe(snapshot(1, 100, 1));
        assert!(state.gauges().is_some());
        state.lapse();
        assert!(state.gauges().is_none());
        let back = state.observe(snapshot(1, 120, 0));
        assert_eq!(back.counters.steps, 20);
        assert!(state.gauges().is_some());
    }

    #[test]
    fn levels_sum_over_a_models_upstreams() {
        let mut sums = ModelGauges::default();
        sums.add(&snapshot(1, 0, 1).gauges);
        let mut other = snapshot(2, 0, 2).gauges;
        other.pools.push(PoolOccupancy {
            total: 8,
            in_use: 0,
            idle: 0,
            host_total: 16,
            host_used: 4,
        });
        sums.add(&other);
        assert_eq!(sums.reporting, 2);
        assert_eq!(sums.sequences_running, 3);
        assert_eq!(sums.pools.len(), 2);
        assert_eq!(
            sums.pools[0],
            PoolOccupancy {
                total: 200,
                in_use: 30,
                idle: 10,
                host_total: 0,
                host_used: 0,
            }
        );
        assert_eq!(sums.pools[0].free(), 160);
        assert_eq!(sums.pools[1].host_free(), 12);
    }

    #[test]
    fn a_snapshot_reads_as_present_absent_or_unparsed() {
        let mut info = serde_json::json!({"model": "m", "weights_sha256": "00"});
        assert_eq!(Reading::of(&info), Reading::Absent);
        info[INFO_TELEMETRY_MEMBER] = snapshot(1, 5, 0).to_value();
        assert_eq!(
            Reading::of(&info),
            Reading::Read(Box::new(snapshot(1, 5, 0)))
        );
        info[INFO_TELEMETRY_MEMBER]["request_ids"] = serde_json::json!([]);
        assert_eq!(Reading::of(&info), Reading::Unparsed);
        info[INFO_TELEMETRY_MEMBER] = serde_json::Value::Null;
        assert_eq!(Reading::of(&info), Reading::Unparsed);
    }
}
