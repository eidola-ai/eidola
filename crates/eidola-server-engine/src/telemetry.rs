//! The node's own telemetry: metrics and logs over OTLP, through the pipeline Eidola's
//! servers share (`eidola-telemetry`), enabled when `OTEL_EXPORTER_OTLP_ENDPOINT` is set
//! and not empty, read under the environment's prefix like every other variable. The
//! endpoint and its headers are secrets of the deployment, not measured, so a measured
//! configuration names no telemetry vendor.
//!
//! **Metrics are aggregates, read at export.** Every instrument is observable: the SDK
//! reads the engine's counts and levels when it exports, once per
//! [`eidola_telemetry::METRIC_EXPORT_INTERVAL`] on its own timer from when the process
//! started, and at no other time ([`observations`]). Nothing in them is per request, and
//! the values from which a quiet interval's cached prompt length could be read are
//! coarsened before they are observed ([`Counts::coarsened`], [`quantize_occupancy`]).
//! `AGENTS.md` → Telemetry has the reasoning for each.
//!
//! **Logs** are the node's lifecycle (configuration and boot, a fatal stop, shutdown),
//! written to stdout and, when enabled, exported. Nothing is logged per request. No
//! traces are exported.
//!
//! **Labels** are `model`, the weights hash, on every series, plus fixed values (an
//! outcome, a pool's index). The resource carries `service.instance.id`, the node's
//! [`instance_id`], which its info answer also carries, so the gateway's logs about an
//! upstream can name the series it exports.

use std::sync::atomic::Ordering;
use std::sync::{Arc, LazyLock, Weak};

use opentelemetry::KeyValue;
use opentelemetry::metrics::Meter;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::http::AppState;

/// The service name telemetry carries unless `OTEL_SERVICE_NAME` says otherwise.
pub const SERVICE_NAME: &str = "eidola-server-engine";

/// Bytes of the node's [`instance_id`], written as lowercase hex.
pub const INSTANCE_ID_BYTES: usize = 16;

static INSTANCE_ID: LazyLock<String> = LazyLock::new(|| {
    let mut id = [0u8; INSTANCE_ID_BYTES];
    getrandom::fill(&mut id).expect("operating-system randomness unavailable");
    hex::encode(id)
});

/// This process's id: random, lowercase hex, the same for its whole life and never
/// again. It is the telemetry resource's `service.instance.id` and the info answer's
/// `instance_id`.
pub fn instance_id() -> &'static str {
    &INSTANCE_ID
}

// ---------------------------------------------------------------------------
// Coarsening
// ---------------------------------------------------------------------------

/// The step the counts of prompt positions are floored to: those computed, and those
/// the prefix cache found and did not. With the gateway's exact prompt-token totals,
/// any one of them exact would give a lone request's cached length.
pub const PROMPT_TOKEN_QUANTUM: u64 = 1024;

/// The step the step and row counts are floored to. A lone request's prompt takes one
/// step per prefill chunk, and its decode steps follow from the exact generated and
/// draft counts, so exact steps would give its uncached length to within a chunk; 64
/// steps is 64 chunks.
pub const STEP_QUANTUM: u64 = 64;

/// The step the steps' wall time is floored to, in microseconds: one second, many
/// prefill chunks' worth on any production GPU.
pub const STEP_MICROS_QUANTUM: u64 = 1_000_000;

/// The step the prefix cache's block and entry counts are floored to: one entry is one
/// block, so exact counts would give a promoted hit's length, or an evicted or demoted
/// conversation's, to within a block. 64 blocks of 16 tokens is
/// [`PROMPT_TOKEN_QUANTUM`].
pub const BLOCK_QUANTUM: u64 = 64;

/// The steps a pool's occupancy is reported in: each count is rounded to the nearest
/// multiple of `ceil(total / OCCUPANCY_STEPS)` blocks.
pub const OCCUPANCY_STEPS: u64 = 256;

/// The most pools reported. The engine's pools are its KV groups, in the order its
/// architecture declares them, then its recurrent-state pool if it has one; no
/// architecture has more than six.
pub const MAX_POOLS: usize = 8;

/// A cumulative count floored to a multiple of `quantum`. Floored rather than rounded,
/// so the reported value never decreases while the count grows.
pub const fn floor_to(count: u64, quantum: u64) -> u64 {
    count / quantum * quantum
}

/// `count` blocks of a pool of `total`, rounded to the nearest multiple of the pool's
/// step (`ceil(total / OCCUPANCY_STEPS)`, at least one block), and never past `total`.
pub const fn quantize_occupancy(count: u64, total: u64) -> u64 {
    let step = total.div_ceil(OCCUPANCY_STEPS);
    let step = if step == 0 { 1 } else { step };
    let rounded = (count.saturating_add(step / 2) / step) * step;
    if rounded > total { total } else { rounded }
}

/// One pool's occupancy, its counts quantized ([`quantize_occupancy`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolOccupancy {
    /// Blocks on the device, exact (the pool's size, fixed at boot).
    pub total: u64,
    /// Blocks a running sequence maps.
    pub in_use: u64,
    /// Blocks the prefix cache holds that no sequence maps (idle, encrypted).
    pub idle: u64,
    /// Blocks in the host tier, exact; zero without one.
    pub host_total: u64,
    /// Host-tier blocks holding a demoted entry.
    pub host_used: u64,
}

impl PoolOccupancy {
    /// Quantizes exact counts. Each rounds on its own, and the idle count gives way
    /// where the two would round past the pool.
    pub const fn quantized(
        total: u64,
        in_use: u64,
        idle: u64,
        host_total: u64,
        host_used: u64,
    ) -> Self {
        let in_use = quantize_occupancy(in_use, total);
        let idle = quantize_occupancy(idle, total);
        let room = total - in_use;
        Self {
            total,
            in_use,
            idle: if idle > room { room } else { idle },
            host_total,
            host_used: quantize_occupancy(host_used, host_total),
        }
    }

    /// Blocks neither in use nor idle.
    pub const fn free(&self) -> u64 {
        self.total
            .saturating_sub(self.in_use)
            .saturating_sub(self.idle)
    }

    /// Host-tier blocks holding nothing.
    pub const fn host_free(&self) -> u64 {
        self.host_total.saturating_sub(self.host_used)
    }
}

macro_rules! counts {
    ($($(#[$doc:meta])* $name:ident / $quantum:expr,)*) => {
        /// Counts since the process started. Each member states the quantum it is
        /// observed floored to ([`Counts::QUANTA`]), 1 for exact.
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct Counts {
            $($(#[$doc])* pub $name: u64,)*
        }

        impl Counts {
            /// Each member's name and the quantum its value is floored to, 1 for one
            /// observed exactly.
            pub const QUANTA: &'static [(&'static str, u64)] =
                &[$((stringify!($name), $quantum)),*];

            /// Each member's name and value, in [`Counts::QUANTA`] order.
            pub fn entries(&self) -> Vec<(&'static str, u64)> {
                vec![$((stringify!($name), self.$name)),*]
            }

            /// The counts as they are observed: each floored to its quantum
            /// ([`floor_to`]).
            pub const fn coarsened(&self) -> Self {
                Self { $($name: floor_to(self.$name, $quantum),)* }
            }
        }
    };
}

counts! {
    /// Requests the engine accepted.
    requests_accepted / 1,
    /// Requests that finished on their own (an end of sequence, a length).
    requests_finished / 1,
    /// Requests cancelled: a disconnect, a stop sequence, a reader cut off.
    requests_cancelled / 1,
    /// Of those, readers cut off because they stopped draining their output.
    requests_cut_off / 1,
    /// Requests the engine refused at submission.
    requests_rejected / 1,
    /// Requests refused at first admission because the prefix cache did not hold their
    /// declared prefix.
    requests_prefix_refused / 1,
    /// Requests the node refused before reading or admitting them because every read
    /// or admission slot was taken (`overloaded`).
    requests_overloaded / 1,
    /// Requests answered with a failure on the node's side, before any stream opened:
    /// the engine not running, or an internal error.
    requests_failed / 1,
    /// Streams that ended with an error event after they opened: the engine stopped or
    /// failed under them.
    streams_failed / 1,
    /// Engine steps that ran at least one sequence.
    steps / STEP_QUANTUM,
    /// Wall time of those steps, in microseconds.
    step_micros / STEP_MICROS_QUANTUM,
    /// Sequences those steps ran, summed over the steps: over `steps`, the mean batch.
    step_rows / STEP_QUANTUM,
    /// Prompt positions computed: the prompts' uncached parts, and the recomputation
    /// of a preempted sequence.
    prefill_tokens / PROMPT_TOKEN_QUANTUM,
    /// Tokens generated.
    generated_tokens / 1,
    /// Speculative draft tokens proposed.
    drafted_tokens / 1,
    /// Of those, the ones accepted.
    accepted_drafts / 1,
    /// Running sequences preempted to make room.
    preemptions / 1,
    /// Requests admitted for the first time, each with one prefix-cache lookup.
    prefix_lookups / 1,
    /// Prompt tokens those lookups found in the prefix cache.
    prefix_hit_tokens / PROMPT_TOKEN_QUANTUM,
    /// Prompt tokens they did not.
    prefix_miss_tokens / PROMPT_TOKEN_QUANTUM,
    /// Cache entries dropped from the device pools to make room, with no host tier to
    /// take them.
    cache_device_evictions / BLOCK_QUANTUM,
    /// Cache entries dropped from the host tier to make room.
    cache_host_evictions / BLOCK_QUANTUM,
    /// Blocks demoted from the device pools to the host tier.
    cache_demoted_blocks / BLOCK_QUANTUM,
    /// Blocks promoted from the host tier back to the device for a hit.
    cache_promoted_blocks / BLOCK_QUANTUM,
}

/// Levels now.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Levels {
    /// Requests the engine holds that are waiting for a seat.
    pub sequences_waiting: u64,
    /// Requests it is running.
    pub sequences_running: u64,
    /// Its pools, at most [`MAX_POOLS`], quantized.
    pub pools: Vec<PoolOccupancy>,
}

/// The node's counts, coarsened, and its levels, with the pools as last recounted.
pub fn read(state: &AppState) -> (Counts, Levels) {
    let s = state.engine.stats();
    let counts = Counts {
        requests_accepted: s.submitted,
        requests_finished: s.finished,
        requests_cancelled: s.cancelled,
        requests_cut_off: s.cut_off,
        requests_rejected: s.rejected,
        requests_prefix_refused: s.prefix_refused,
        requests_overloaded: state.overloaded.load(Ordering::Relaxed),
        requests_failed: state.failed.load(Ordering::Relaxed),
        streams_failed: state.streams_failed.load(Ordering::Relaxed),
        steps: s.forward_steps,
        step_micros: s.step_micros,
        step_rows: s.step_rows,
        prefill_tokens: s.prefill_tokens,
        generated_tokens: s.generated_tokens,
        drafted_tokens: s.drafted,
        accepted_drafts: s.accepted_drafts,
        preemptions: s.preemptions,
        prefix_lookups: s.admissions,
        prefix_hit_tokens: s.prefix_hit_tokens,
        prefix_miss_tokens: s.prefix_miss_tokens,
        cache_device_evictions: s.device_evictions,
        cache_host_evictions: s.host_evictions,
        cache_demoted_blocks: s.demoted_blocks,
        cache_promoted_blocks: s.promoted_blocks,
    }
    .coarsened();
    let levels = Levels {
        sequences_waiting: s.waiting,
        sequences_running: s.running,
        pools: state
            .engine
            .pools()
            .iter()
            .take(MAX_POOLS)
            .map(|p| PoolOccupancy::quantized(p.total, p.in_use, p.idle, p.host_total, p.host_used))
            .collect(),
    };
    (counts, levels)
}

// ---------------------------------------------------------------------------
// Instruments
// ---------------------------------------------------------------------------

/// Where each count is exported: the instrument, and the fixed label that tells it
/// from the other members it shares one with. `step_micros` is exported as seconds.
type CountExport = (
    &'static str,
    &'static str,
    Option<(&'static str, &'static str)>,
);

const COUNT_EXPORTS: &[CountExport] = &[
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
    (
        "requests_failed",
        "engine.node.requests",
        Some(("outcome", "failed")),
    ),
    (
        "streams_failed",
        "engine.node.requests",
        Some(("outcome", "stream_failed")),
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

/// The one counter observed in seconds.
const STEP_TIME: &str = "engine.node.step.time";

/// Each counter instrument with its description.
const COUNTER_INSTRUMENTS: &[(&str, &str)] = &[
    (
        "engine.node.requests",
        "Requests on the engine node, by outcome",
    ),
    (
        "engine.node.steps",
        "Engine steps that ran at least one sequence, in steps of 64",
    ),
    (
        STEP_TIME,
        "Wall time of engine steps that ran at least one sequence, in whole seconds",
    ),
    (
        "engine.node.step.rows",
        "Sequences run by engine steps, summed over the steps, in steps of 64",
    ),
    (
        "engine.node.tokens",
        "Tokens the engine computed, by type: prefill (in steps of 1024) or generated",
    ),
    (
        "engine.node.drafts",
        "Speculative draft tokens, proposed and accepted",
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
        "Prefix-cache entries dropped to make room, by tier, in steps of 64",
    ),
    (
        "engine.node.prefix_cache.transfers",
        "Prefix-cache blocks moved between the device and the host tier, in steps of 64",
    ),
];

/// Each gauge instrument with its description.
const GAUGE_INSTRUMENTS: &[(&str, &str)] = &[
    (
        "engine.node.sequences",
        "Requests in the engine, waiting for a seat or running",
    ),
    (
        "engine.node.kv.blocks",
        "KV pool blocks, by pool and state, rounded to 1/256 of the pool",
    ),
    (
        "engine.node.kv.host_blocks",
        "Host-tier blocks, by pool and state, rounded to 1/256 of the tier",
    ),
];

/// The `pool` label: a pool's index in the engine's order, from this fixed list.
const POOL_LABELS: [&str; MAX_POOLS] = ["0", "1", "2", "3", "4", "5", "6", "7"];

/// What one observation adds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Count(u64),
    /// `engine.node.step.time`, the one counter in seconds.
    Seconds(f64),
}

/// One observation of one instrument, with the series it lands in.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub instrument: &'static str,
    pub value: Value,
    pub attributes: Vec<KeyValue>,
}

/// Every observation of every instrument for these counts and levels, labelled by
/// `model`. Every series is observed every time, zeros included, so which series an
/// export carries never depends on the engine's activity.
pub fn observations(model: &str, counts: &Counts, levels: &Levels) -> Vec<Observation> {
    let mut out = Vec::new();
    let mut observe = |instrument, value, extra: &[(&'static str, &'static str)]| {
        let mut attributes = vec![KeyValue::new("model", model.to_string())];
        attributes.extend(extra.iter().map(|&(k, v)| KeyValue::new(k, v)));
        out.push(Observation {
            instrument,
            value,
            attributes,
        });
    };
    for (member, count) in counts.entries() {
        let Some(&(_, instrument, label)) = COUNT_EXPORTS.iter().find(|(m, ..)| *m == member)
        else {
            continue;
        };
        let value = if instrument == STEP_TIME {
            Value::Seconds(count as f64 / 1e6)
        } else {
            Value::Count(count)
        };
        observe(instrument, value, label.as_slice());
    }
    for (state, value) in [
        ("waiting", levels.sequences_waiting),
        ("running", levels.sequences_running),
    ] {
        observe(
            "engine.node.sequences",
            Value::Count(value),
            &[("state", state)],
        );
    }
    for (pool, occupancy) in POOL_LABELS.iter().zip(&levels.pools) {
        for (state, value) in [
            ("in_use", occupancy.in_use),
            ("idle", occupancy.idle),
            ("free", occupancy.free()),
        ] {
            observe(
                "engine.node.kv.blocks",
                Value::Count(value),
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
                Value::Count(value),
                &[("pool", pool), ("state", state)],
            );
        }
    }
    out
}

/// What the node's instruments read: its state, held weakly, so an instrument never
/// keeps the engine alive past the router, and observes nothing once it is gone.
#[derive(Clone)]
pub struct Source(Weak<AppState>);

impl Source {
    /// The instruments' source over `state`.
    pub fn new(state: &Arc<AppState>) -> Self {
        Self(Arc::downgrade(state))
    }

    /// This node's observations now, or none once its state is gone.
    pub fn observe(&self) -> Vec<Observation> {
        let Some(state) = self.0.upgrade() else {
            return Vec::new();
        };
        let (counts, levels) = read(&state);
        observations(state.model.weights_hash(), &counts, &levels)
    }

    /// Registers the node's instruments on `meter`, each observable: the SDK reads
    /// them when it exports, and only then ([`observations`]).
    pub fn register(&self, meter: &Meter) {
        for &(name, description) in COUNTER_INSTRUMENTS {
            let source = self.clone();
            if name == STEP_TIME {
                meter
                    .f64_observable_counter(name)
                    .with_description(description)
                    .with_unit("s")
                    .with_callback(move |counter| {
                        for o in source.observe() {
                            if let (true, Value::Seconds(v)) = (o.instrument == name, o.value) {
                                counter.observe(v, &o.attributes);
                            }
                        }
                    })
                    .build();
                continue;
            }
            meter
                .u64_observable_counter(name)
                .with_description(description)
                .with_callback(move |counter| {
                    for o in source.observe() {
                        if let (true, Value::Count(v)) = (o.instrument == name, o.value) {
                            counter.observe(v, &o.attributes);
                        }
                    }
                })
                .build();
        }
        for &(name, description) in GAUGE_INSTRUMENTS {
            let source = self.clone();
            meter
                .u64_observable_gauge(name)
                .with_description(description)
                .with_callback(move |gauge| {
                    for o in source.observe() {
                        if let (true, Value::Count(v)) = (o.instrument == name, o.value) {
                            gauge.observe(v, &o.attributes);
                        }
                    }
                })
                .build();
        }
    }
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

/// The OTLP providers, so they can be flushed and shut down.
pub struct OtelGuard {
    meter_provider: opentelemetry_sdk::metrics::SdkMeterProvider,
    logger_provider: opentelemetry_sdk::logs::SdkLoggerProvider,
}

impl OtelGuard {
    /// The meter the node's instruments register on ([`Source::register`]).
    pub fn meter(&self) -> Meter {
        opentelemetry::metrics::MeterProvider::meter(&self.meter_provider, SERVICE_NAME)
    }

    /// Flushes and shuts down both providers.
    pub fn shutdown(self) {
        if let Err(e) = self.meter_provider.shutdown() {
            eprintln!("otel: meter provider shutdown error: {e}");
        }
        if let Err(e) = self.logger_provider.shutdown() {
            eprintln!("otel: logger provider shutdown error: {e}");
        }
    }
}

/// Installs the log subscriber (stdout, plus the OTLP log bridge when enabled) and
/// returns the providers when OTLP export is enabled. `lookup` answers for one bare
/// variable name: the prefix is read first (`EIDOLA_ENV_PREFIX`), then the telemetry
/// variables under it. A prefix that does not read leaves telemetry off; the boot
/// that follows refuses it and says why.
pub fn init(lookup: impl Fn(&str) -> Option<String>) -> Option<OtelGuard> {
    let otel = eidola_common::env::Env::from_lookup(&lookup)
        .ok()
        .and_then(|env| {
            eidola_telemetry::OtlpConfig::read(|name| env.lookup(name, &lookup), SERVICE_NAME)
        })
        .map(|config| {
            let timeouts = config.timeouts();
            let [metrics_client, logs_client] = eidola_telemetry::otlp_http_clients(
                tls_config(),
                [timeouts.metrics, timeouts.logs],
            );
            let resource = config.resource([KeyValue::new("service.instance.id", instance_id())]);
            OtelGuard {
                meter_provider: config.meter_provider(resource.clone(), metrics_client),
                logger_provider: config.logger_provider(resource, logs_client),
            }
        });

    let log_layer = otel.as_ref().map(|guard| {
        opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new(
            &guard.logger_provider,
        )
    });
    tracing_subscriber::registry()
        .with(log_filter())
        .with(
            tracing_subscriber::fmt::layer()
                .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stdout())),
        )
        .with(log_layer)
        .init();
    otel
}

/// The log filter, fixed in the measured image: `info` for everything, with the
/// telemetry pipeline's own crates and its HTTP client at `warn`, so the exporter's
/// own events do not feed its log bridge.
fn log_filter() -> EnvFilter {
    [
        "opentelemetry=warn",
        "opentelemetry_sdk=warn",
        "opentelemetry_otlp=warn",
        "reqwest=warn",
        "hyper=warn",
        "hyper_util=warn",
    ]
    .into_iter()
    .fold(EnvFilter::new("info"), |filter, directive| {
        filter.add_directive(directive.parse().expect("a valid log directive"))
    })
}

/// The exporters' TLS: the bundled Mozilla roots, under the pure-Rust provider the
/// caller installs ([`install_crypto_provider`]).
fn tls_config() -> rustls::ClientConfig {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth()
}

/// Installs the process's TLS crypto provider (pure Rust, as the gateway's), once.
pub fn install_crypto_provider() {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counts a quiet interval's cached prompt length could be read from, given the
    /// gateway's exact token totals, are exactly these, at these quanta; every other is
    /// exact. A new member states its quantum here on purpose.
    #[test]
    fn the_counts_coarsened_are_exactly_these() {
        let coarse: Vec<(&str, u64)> = Counts::QUANTA
            .iter()
            .copied()
            .filter(|&(_, q)| q != 1)
            .collect();
        assert_eq!(
            coarse,
            [
                ("steps", STEP_QUANTUM),
                ("step_micros", STEP_MICROS_QUANTUM),
                ("step_rows", STEP_QUANTUM),
                ("prefill_tokens", PROMPT_TOKEN_QUANTUM),
                ("prefix_hit_tokens", PROMPT_TOKEN_QUANTUM),
                ("prefix_miss_tokens", PROMPT_TOKEN_QUANTUM),
                ("cache_device_evictions", BLOCK_QUANTUM),
                ("cache_host_evictions", BLOCK_QUANTUM),
                ("cache_demoted_blocks", BLOCK_QUANTUM),
                ("cache_promoted_blocks", BLOCK_QUANTUM),
            ]
        );
        // 64 blocks of the pinned deployments' 16 tokens is the prompt quantum.
        assert_eq!(BLOCK_QUANTUM * 16, PROMPT_TOKEN_QUANTUM);
    }

    #[test]
    fn coarsening_floors_each_count_to_its_own_quantum() {
        assert_eq!(floor_to(0, PROMPT_TOKEN_QUANTUM), 0);
        assert_eq!(floor_to(PROMPT_TOKEN_QUANTUM - 1, PROMPT_TOKEN_QUANTUM), 0);
        assert_eq!(
            floor_to(3 * PROMPT_TOKEN_QUANTUM + 5, PROMPT_TOKEN_QUANTUM),
            3 * PROMPT_TOKEN_QUANTUM
        );
        let exact = Counts {
            steps: STEP_QUANTUM + 5,
            generated_tokens: 7,
            prefill_tokens: 2 * PROMPT_TOKEN_QUANTUM + 1,
            ..Counts::default()
        };
        let c = exact.coarsened();
        assert_eq!(
            (c.steps, c.generated_tokens, c.prefill_tokens),
            (STEP_QUANTUM, 7, 2 * PROMPT_TOKEN_QUANTUM)
        );
    }

    #[test]
    fn occupancy_is_rounded_to_its_pool_steps() {
        // A pool of 1024 blocks reports in steps of 4.
        assert_eq!(quantize_occupancy(0, 1024), 0);
        assert_eq!(quantize_occupancy(1, 1024), 0);
        assert_eq!(quantize_occupancy(2, 1024), 4);
        assert_eq!(quantize_occupancy(5, 1024), 4);
        assert_eq!(quantize_occupancy(1024, 1024), 1024);
        // A pool of 100,000 blocks in steps of 391: one sequence of a few hundred
        // blocks is not told apart from none.
        assert_eq!(quantize_occupancy(150, 100_000), 0);
        assert_eq!(quantize_occupancy(200, 100_000), 391);
        // Never past the pool, and a small pool is exact.
        assert_eq!(quantize_occupancy(999, 1000), 1000);
        assert_eq!(quantize_occupancy(7, 64), 7);
        assert_eq!(quantize_occupancy(0, 0), 0);
        // Quantized counts always fit their pool together.
        for total in [1u64, 63, 64, 255, 256, 257, 1000, 100_000] {
            for in_use in [0, total / 3, total / 2, total.saturating_sub(2), total] {
                let pool = PoolOccupancy::quantized(total, in_use, total - in_use, total, in_use);
                assert!(pool.in_use + pool.idle <= pool.total, "{total} {in_use}");
                assert!(pool.host_used <= pool.host_total);
            }
        }
    }

    /// Every count is exported once, and instruments sharing an export are told apart
    /// by distinct labels.
    #[test]
    fn every_count_is_exported_once() {
        let exported: Vec<&str> = COUNT_EXPORTS.iter().map(|(m, ..)| *m).collect();
        let members: Vec<&str> = Counts::QUANTA.iter().map(|(m, _)| *m).collect();
        assert_eq!(exported, members);
        for (_, instrument, _) in COUNT_EXPORTS {
            assert!(
                COUNTER_INSTRUMENTS
                    .iter()
                    .any(|(name, _)| name == instrument),
                "{instrument} has no instrument"
            );
        }
        for (i, (m, instrument, label)) in COUNT_EXPORTS.iter().enumerate() {
            for (other, other_instrument, other_label) in &COUNT_EXPORTS[i + 1..] {
                if instrument == other_instrument {
                    assert!(
                        label.is_some() && label != other_label,
                        "{m} and {other} share {instrument}"
                    );
                }
            }
        }
    }

    /// Which series are observed depends on the node's shape (its pools, whether it
    /// has a host tier), never on its activity: an idle node and a busy one observe
    /// the same series.
    #[test]
    fn the_series_observed_do_not_follow_activity() {
        let pools = |used| {
            vec![
                PoolOccupancy::quantized(1000, used, used / 2, 0, 0),
                PoolOccupancy::quantized(1000, used, 0, 500, used / 4),
            ]
        };
        let series = |o: &[Observation]| -> Vec<(String, Vec<String>)> {
            o.iter()
                .map(|o| {
                    (
                        o.instrument.to_string(),
                        o.attributes
                            .iter()
                            .map(|kv| format!("{}={}", kv.key, kv.value))
                            .collect(),
                    )
                })
                .collect()
        };
        let idle = observations(
            "m",
            &Counts::default(),
            &Levels {
                pools: pools(0),
                ..Levels::default()
            },
        );
        let busy = observations(
            "m",
            &Counts {
                requests_accepted: 9,
                generated_tokens: 900,
                ..Counts::default()
            },
            &Levels {
                sequences_waiting: 2,
                sequences_running: 3,
                pools: pools(600),
            },
        );
        assert_eq!(series(&idle), series(&busy));
        assert!(
            idle.iter()
                .all(|o| o.attributes[0] == KeyValue::new("model", "m"))
        );
        // The host tier's series exist only for the pool that has one.
        let host: Vec<_> = idle
            .iter()
            .filter(|o| o.instrument == "engine.node.kv.host_blocks")
            .collect();
        assert_eq!(host.len(), 2);
        assert!(
            host.iter()
                .all(|o| o.attributes.contains(&KeyValue::new("pool", "1")))
        );
    }

    #[test]
    fn the_instance_id_is_random_hex_for_the_process() {
        assert_eq!(instance_id().len(), 2 * INSTANCE_ID_BYTES);
        assert!(
            instance_id()
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        );
        assert_eq!(instance_id(), instance_id());
    }

    /// The deployment's secrets carry the variables the pipeline reads.
    #[test]
    fn the_pipeline_reads_the_deployments_telemetry_secrets() {
        let read = std::cell::RefCell::new(Vec::new());
        let config = eidola_telemetry::OtlpConfig::read(
            |name| {
                read.borrow_mut().push(name.to_string());
                (name == eidola_common::engine_deployment::env::OTEL_EXPORTER_OTLP_ENDPOINT)
                    .then(|| "https://otlp.example".to_string())
            },
            SERVICE_NAME,
        );
        assert!(config.is_some());
        let read = read.into_inner();
        for secret in [
            eidola_common::engine_deployment::env::OTEL_EXPORTER_OTLP_ENDPOINT,
            eidola_common::engine_deployment::env::OTEL_EXPORTER_OTLP_HEADERS,
        ] {
            assert!(read.iter().any(|name| name == secret), "{secret}: {read:?}");
        }
    }
}
