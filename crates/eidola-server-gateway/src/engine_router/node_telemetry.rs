//! What the gateway exports for the engine nodes it routes to.
//!
//! A node has no egress: its measured config allows the gateway's connection
//! and nothing else. So its operational snapshot
//! (`eidola_common::engine_protocol::telemetry`) rides on the probe: the
//! router reads it from the node's `/v1/engine/info` answer on each probe,
//! over the attested, token-authenticated connection the probe already uses,
//! and this module turns it into the gateway's own instruments, under
//! `engine.node.*`.
//!
//! **Every series is one upstream's**, labelled by `model` (the pinned
//! weights hash), `deployment` and `node` ([`NodeLabels`]), plus fixed label
//! values. `deployment` names one of the compiled-in pins; `node` is an
//! opaque id derived from the upstream's normalized base URL. Neither the URL
//! nor the node's boot id is ever a label value. Why a per-node series is
//! admissible, and what it leaves: `crates/eidola-server-engine/AGENTS.md` →
//! Telemetry.
//!
//! - **Counters** are the node's counts since its boot. Each upstream keeps
//!   the last snapshot's counts and boot id ([`NodeState`]), and each probe
//!   adds the growth since then. A changed boot id is a restart: the new
//!   boot's counts so far are all growth, and `engine.node.restarts` counts
//!   it. The first snapshot from an upstream is only a baseline, so a
//!   gateway that starts, or an upstream that placement adds, never counts a
//!   node's history as one probe interval's work.
//! - **Gauges** are each placed upstream's last snapshot, read from the
//!   router at export. `engine.node.reporting` is 1 for an upstream whose last
//!   probe read a snapshot and 0 for one whose did not (a failed probe, or no
//!   snapshot), which then reports no levels until a probe reads one. A model
//!   with no placed upstream has no `engine.node.*` series; `engine.upstreams`
//!   says that.
//! - **A snapshot that does not parse** is counted
//!   (`engine.node.snapshots{outcome=unparsed}`), never logged, and changes
//!   nothing else: telemetry never decides whether a node takes traffic. A
//!   node built before the snapshot answers without it (`outcome=absent`).
//!
//! The state is per upstream, so it is bounded by placement, and goes with
//! the upstream when placement drops it. So do its series: its gauges are no
//! longer observed, and the observable gauges export only what the last
//! callback observed. Its counters are cumulative and synchronous, so the SDK
//! keeps exporting their last totals, flat, until the gateway restarts: a
//! rate over them reads zero, and the series held are bounded by the
//! distinct upstreams placed in the process's life (and by the SDK's
//! per-instrument cardinality limit beyond that).

use std::collections::BTreeMap;
use std::sync::LazyLock;

use eidola_common::engine_deployment::is_safe_component;
use eidola_common::engine_protocol::telemetry::{
    BOOT_ID_BYTES, INFO_TELEMETRY_MEMBER, MAX_POOLS, NodeCounters, NodeGauges, NodeTelemetry,
};
use opentelemetry::KeyValue;
use opentelemetry::metrics::Counter;
use sha2::{Digest, Sha256};

use super::EngineRouter;

/// Where a pinned deployment's config lives, relative to the workspace root:
/// `deploy/engine/<name>/<variant>/tinfoil-config.yml`.
const CONFIG_PREFIX: &str = "deploy/engine/";
const CONFIG_SUFFIX: &str = "/tinfoil-config.yml";

/// How many hex digits of a config hash name a deployment whose config path
/// does not give it a name.
const DEPLOYMENT_HASH_DIGITS: usize = 16;

/// How many hex digits of the SHA-256 of an upstream's base URL make its
/// `node` label.
const NODE_ID_DIGITS: usize = 12;

/// The `deployment` label of a pinned deployment: `<name>/<variant>` from
/// its config's path, which the gateway's build holds to
/// `deploy/engine/<name>/<variant>/tinfoil-config.yml` with both components
/// in `[a-z0-9][a-z0-9._-]*`; otherwise the first sixteen hex digits of its
/// config hash.
pub(crate) fn deployment_label(config: Option<&str>, config_sha256: &str) -> String {
    config
        .and_then(|path| path.strip_prefix(CONFIG_PREFIX))
        .and_then(|rest| rest.strip_suffix(CONFIG_SUFFIX))
        .and_then(|rest| rest.split_once('/'))
        .filter(|(name, variant)| is_safe_component(name) && is_safe_component(variant))
        .map_or_else(
            || {
                config_sha256
                    .get(..DEPLOYMENT_HASH_DIGITS)
                    .unwrap_or(config_sha256)
                    .to_string()
            },
            |(name, variant)| format!("{name}/{variant}"),
        )
}

/// The `node` label of the upstream at `base_url` (normalized, as placement
/// gives it): the first twelve hex digits of its SHA-256. Stable across
/// gateway restarts and gateways, and carrying no hostname; the operator
/// maps it back to a placement row by hashing the row's normalized URL.
pub(crate) fn node_id(base_url: &str) -> String {
    let mut id = hex::encode(Sha256::digest(base_url.as_bytes()));
    id.truncate(NODE_ID_DIGITS);
    id
}

/// What one upstream's `engine.node.*` series are labelled by, besides its
/// model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NodeLabels {
    /// [`deployment_label`] of its placement row's deployment.
    pub deployment: String,
    /// [`node_id`] of its base URL.
    pub node: String,
}

impl NodeLabels {
    pub(crate) fn new(deployment: String, base_url: &str) -> Self {
        Self {
            deployment,
            node: node_id(base_url),
        }
    }

    /// The labels every series of this upstream serving `model` carries.
    fn series(&self, model: &str) -> Vec<KeyValue> {
        vec![
            KeyValue::new("model", model.to_string()),
            KeyValue::new("deployment", self.deployment.clone()),
            KeyValue::new("node", self.node.clone()),
        ]
    }
}

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

/// How much one counter measurement adds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Amount {
    Count(u64),
    /// `engine.node.step.time`, the one counter in seconds.
    Seconds(f64),
}

/// One addition to a counter instrument, with the series it lands in.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CounterAdd {
    pub instrument: &'static str,
    pub amount: Amount,
    pub attributes: Vec<KeyValue>,
}

/// What one probe's snapshot adds to the counters, for the upstream labelled
/// `labels` serving `model`, a pinned weights hash.
pub(crate) fn counter_adds(
    model: &str,
    labels: &NodeLabels,
    reading: &Reading,
    growth: Option<&Growth>,
) -> Vec<CounterAdd> {
    let series = labels.series(model);
    let with = |label: Option<(&'static str, &'static str)>| {
        let mut attributes = series.clone();
        attributes.extend(label.map(|(key, value)| KeyValue::new(key, value)));
        attributes
    };
    let mut adds = vec![CounterAdd {
        instrument: "engine.node.snapshots",
        amount: Amount::Count(1),
        attributes: with(Some(("outcome", reading.outcome()))),
    }];
    let Some(growth) = growth else { return adds };
    if growth.restarted {
        adds.push(CounterAdd {
            instrument: "engine.node.restarts",
            amount: Amount::Count(1),
            attributes: with(None),
        });
    }
    for (member, value) in growth.counters.entries() {
        if value == 0 {
            continue;
        }
        let Some(&(_, instrument, label)) = COUNTER_EXPORTS.iter().find(|(m, ..)| *m == member)
        else {
            continue;
        };
        let amount = if member == "step_micros" {
            Amount::Seconds(value as f64 / 1e6)
        } else {
            Amount::Count(value)
        };
        adds.push(CounterAdd {
            instrument,
            amount,
            attributes: with(label),
        });
    }
    adds
}

/// Exports what one probe's snapshot read for the upstream labelled `labels`
/// serving `model`, a pinned weights hash.
pub(crate) fn record(model: &str, labels: &NodeLabels, reading: &Reading, growth: Option<&Growth>) {
    for add in counter_adds(model, labels, reading, growth) {
        match add.amount {
            Amount::Seconds(seconds) => STEP_TIME.add(seconds, &add.attributes),
            Amount::Count(value) => {
                if let Some(counter) = COUNTERS.get(add.instrument) {
                    counter.add(value, &add.attributes);
                }
            }
        }
    }
}

/// One placed upstream, as the gauges read it at export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NodeReport {
    /// Its model's pinned weights hash.
    pub model: String,
    pub labels: NodeLabels,
    /// Its last snapshot's levels, while its last probe read one.
    pub gauges: Option<NodeGauges>,
}

/// One observation of a gauge instrument.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GaugeObservation {
    pub instrument: &'static str,
    pub value: u64,
    pub attributes: Vec<KeyValue>,
}

/// Every gauge observation for `reports`: `engine.node.reporting`, 1 or 0,
/// for every placed upstream, and the levels of each upstream that reported
/// them.
pub(crate) fn gauge_observations(reports: &[NodeReport]) -> Vec<GaugeObservation> {
    let mut out = Vec::new();
    for report in reports {
        let series = report.labels.series(&report.model);
        let mut observe = |instrument, value, extra: &[(&'static str, &'static str)]| {
            let mut attributes = series.clone();
            attributes.extend(extra.iter().map(|&(k, v)| KeyValue::new(k, v)));
            out.push(GaugeObservation {
                instrument,
                value,
                attributes,
            });
        };
        observe(
            "engine.node.reporting",
            u64::from(report.gauges.is_some()),
            &[],
        );
        let Some(gauges) = &report.gauges else {
            continue;
        };
        for (state, value) in [
            ("waiting", gauges.sequences_waiting),
            ("running", gauges.sequences_running),
        ] {
            observe("engine.node.sequences", value, &[("state", state)]);
        }
        for (pool, occupancy) in POOL_LABELS.iter().zip(&gauges.pools) {
            for (state, value) in [
                ("in_use", occupancy.in_use),
                ("idle", occupancy.idle),
                ("free", occupancy.free()),
            ] {
                observe(
                    "engine.node.kv.blocks",
                    value,
                    &[("pool", pool), ("state", state)],
                );
            }
            if occupancy.host_total == 0 {
                continue;
            }
            for (state, value) in [
                ("used", occupancy.host_used),
                ("free", occupancy.host_free()),
            ] {
                observe(
                    "engine.node.kv.host_blocks",
                    value,
                    &[("pool", pool), ("state", state)],
                );
            }
        }
    }
    out
}

/// Each observable gauge with its description.
const GAUGE_INSTRUMENTS: &[(&str, &str)] = &[
    (
        "engine.node.reporting",
        "Whether the engine upstream's last probe read a telemetry snapshot (1) or not (0)",
    ),
    (
        "engine.node.sequences",
        "Requests in the engine node, waiting for a seat or running",
    ),
    (
        "engine.node.kv.blocks",
        "Engine KV pool blocks, by pool and state, rounded on the node",
    ),
    (
        "engine.node.kv.host_blocks",
        "Engine host-tier blocks, by pool and state, rounded on the node",
    ),
];

/// Registers the observable gauges read from the router at each export
/// ([`gauge_observations`]): `engine.node.reporting` for every placed
/// upstream, and, for each that reported levels,
/// `engine.node.sequences{state=waiting|running}`,
/// `engine.node.kv.blocks{pool, state=in_use|idle|free}`, and, for a pool
/// with a host tier, `engine.node.kv.host_blocks{pool, state=used|free}`.
pub fn register(router: EngineRouter) {
    for &(name, description) in GAUGE_INSTRUMENTS {
        let router = router.clone();
        meter()
            .u64_observable_gauge(name)
            .with_description(description)
            .with_callback(move |gauge| {
                for observation in gauge_observations(&router.node_reports()) {
                    if observation.instrument == name {
                        gauge.observe(observation.value, &observation.attributes);
                    }
                }
            })
            .build();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use eidola_common::engine_protocol::telemetry::PoolOccupancy;

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

    const URL_A: &str = "https://engine-a.example:8443";
    const URL_B: &str = "https://engine-b.example";

    fn labels(url: &str) -> NodeLabels {
        NodeLabels::new("mimo-v2.6-flash/b300x1".into(), url)
    }

    /// Each attribute as `key=value`.
    fn rendered(attributes: &[KeyValue]) -> Vec<String> {
        attributes
            .iter()
            .map(|kv| format!("{}={}", kv.key, kv.value))
            .collect()
    }

    /// No attribute value holds any part of an upstream's URL.
    fn assert_no_url(attributes: &[KeyValue]) {
        for kv in attributes {
            let value = kv.value.to_string();
            for url in [URL_A, URL_B] {
                let host = url.trim_start_matches("https://");
                let host = host.split(':').next().unwrap();
                assert!(
                    !value.contains(host) && !value.contains("https"),
                    "{}={value} carries an upstream's URL",
                    kv.key
                );
            }
        }
    }

    #[test]
    fn a_deployment_is_labelled_by_its_config_paths_name_or_its_hash() {
        let hash = "fde1d9f96fa9f20e8cf375e9315719b9c21018e164aa010faaa10178a55570dd";
        assert_eq!(
            deployment_label(
                Some("deploy/engine/mimo-v2.6-flash/b300x1/tinfoil-config.yml"),
                hash
            ),
            "mimo-v2.6-flash/b300x1"
        );
        for config in [
            None,
            Some("deploy/engine/mimo/tinfoil-config.yml"),
            Some("deploy/engine/mimo/b300x1/extra/tinfoil-config.yml"),
            Some("deploy/engine/Mimo/b300x1/tinfoil-config.yml"),
            Some("elsewhere/mimo/b300x1/tinfoil-config.yml"),
        ] {
            assert_eq!(deployment_label(config, hash), "fde1d9f96fa9f20e");
        }
    }

    /// Every compiled-in deployment has a name from its config path.
    #[test]
    fn every_pinned_deployment_has_a_name() {
        for model in crate::engine_trust::PINNED_MODELS {
            for deployment in crate::engine_trust::accepted_deployments(model.id()).unwrap() {
                let label = deployment_label(Some(&deployment.config), &deployment.config_sha256);
                assert_eq!(
                    format!("{CONFIG_PREFIX}{label}{CONFIG_SUFFIX}"),
                    deployment.config
                );
            }
        }
    }

    #[test]
    fn a_node_is_labelled_by_a_prefix_of_its_urls_hash() {
        let id = node_id(URL_A);
        assert_eq!(id.len(), NODE_ID_DIGITS);
        assert!(id.bytes().all(|b| b.is_ascii_hexdigit()));
        assert!(hex::encode(Sha256::digest(URL_A.as_bytes())).starts_with(&id));
        assert_eq!(id, node_id(URL_A));
        assert_ne!(id, node_id(URL_B));
    }

    /// Every counter a probe adds lands in its upstream's series: the model,
    /// deployment and node labels, and the member's fixed label.
    #[test]
    fn counters_carry_the_upstreams_labels() {
        let labels = labels(URL_A);
        let mut state = NodeState::default();
        state.observe(snapshot(1, 100, 1));
        let growth = state.observe(snapshot(2, 30, 1));
        let reading = Reading::Read(Box::new(snapshot(2, 30, 1)));
        let adds = counter_adds("w", &labels, &reading, Some(&growth));
        let node = format!("node={}", node_id(URL_A));
        let expected = |extra: &[&str]| {
            let mut all = vec![
                "model=w".to_string(),
                "deployment=mimo-v2.6-flash/b300x1".to_string(),
                node.clone(),
            ];
            all.extend(extra.iter().map(|s| s.to_string()));
            all
        };
        let of = |instrument: &str| {
            adds.iter()
                .filter(|a| a.instrument == instrument)
                .map(|a| (a.amount, rendered(&a.attributes)))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            of("engine.node.snapshots"),
            [(Amount::Count(1), expected(&["outcome=read"]))]
        );
        assert_eq!(
            of("engine.node.restarts"),
            [(Amount::Count(1), expected(&[]))]
        );
        assert_eq!(
            of("engine.node.steps"),
            [(Amount::Count(30), expected(&[]))]
        );
        assert_eq!(
            of("engine.node.tokens"),
            [(Amount::Count(90), expected(&["type=generated"]))]
        );
        for add in &adds {
            assert_no_url(&add.attributes);
        }

        // A reading with no snapshot counts only its outcome.
        let adds = counter_adds("w", &labels, &Reading::Absent, None);
        assert_eq!(adds.len(), 1);
        assert_eq!(rendered(&adds[0].attributes), expected(&["outcome=absent"]));
    }

    /// Two nodes of one model report apart: each placed upstream's
    /// `engine.node.reporting` is 1 or 0, and its levels are its own.
    #[test]
    fn levels_are_reported_per_upstream() {
        let mut with_host = snapshot(2, 0, 2).gauges;
        with_host.pools.push(PoolOccupancy {
            total: 8,
            in_use: 0,
            idle: 0,
            host_total: 16,
            host_used: 4,
        });
        let reports = [
            NodeReport {
                model: "w".into(),
                labels: labels(URL_A),
                gauges: Some(snapshot(1, 0, 1).gauges),
            },
            NodeReport {
                model: "w".into(),
                labels: labels(URL_B),
                gauges: Some(with_host),
            },
            NodeReport {
                model: "w".into(),
                labels: NodeLabels::new("other/x1".into(), "https://engine-c.example"),
                gauges: None,
            },
        ];
        let observations = gauge_observations(&reports);
        let find = |instrument: &str, node: &str, extra: &[&str]| {
            let mut want = vec!["model=w".to_string(), format!("node={}", node_id(node))];
            want.extend(extra.iter().map(|s| s.to_string()));
            let found: Vec<u64> = observations
                .iter()
                .filter(|o| {
                    let have = rendered(&o.attributes);
                    o.instrument == instrument
                        && have.len() == want.len() + 1
                        && want.iter().all(|w| have.contains(w))
                })
                .map(|o| o.value)
                .collect();
            assert!(found.len() <= 1, "{instrument} {want:?} observed twice");
            found.first().copied()
        };
        assert_eq!(find("engine.node.reporting", URL_A, &[]), Some(1));
        assert_eq!(find("engine.node.reporting", URL_B, &[]), Some(1));
        assert_eq!(
            find("engine.node.reporting", "https://engine-c.example", &[]),
            Some(0)
        );
        assert_eq!(
            find("engine.node.sequences", URL_A, &["state=running"]),
            Some(1)
        );
        assert_eq!(
            find("engine.node.sequences", URL_B, &["state=running"]),
            Some(2)
        );
        assert_eq!(
            find("engine.node.kv.blocks", URL_A, &["pool=0", "state=in_use"]),
            Some(10)
        );
        assert_eq!(
            find("engine.node.kv.blocks", URL_B, &["pool=0", "state=free"]),
            Some(75)
        );
        assert_eq!(
            find(
                "engine.node.kv.host_blocks",
                URL_B,
                &["pool=1", "state=free"]
            ),
            Some(12)
        );
        // A pool with no host tier, and an upstream that reported no levels,
        // observe none.
        assert_eq!(
            find(
                "engine.node.kv.host_blocks",
                URL_A,
                &["pool=0", "state=free"]
            ),
            None
        );
        let lapsed = node_id("https://engine-c.example");
        assert_eq!(
            observations
                .iter()
                .filter(|o| rendered(&o.attributes).contains(&format!("node={lapsed}")))
                .count(),
            1
        );
        for observation in &observations {
            assert_no_url(&observation.attributes);
            assert!(
                GAUGE_INSTRUMENTS
                    .iter()
                    .any(|(name, _)| *name == observation.instrument)
            );
        }
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
