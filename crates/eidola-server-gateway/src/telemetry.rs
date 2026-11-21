//! OpenTelemetry initialization for traces, metrics, and logs.
//!
//! Enabled when `OTEL_EXPORTER_OTLP_ENDPOINT` is set and not empty. Ships
//! telemetry directly to Grafana Cloud (or any OTLP-compatible endpoint) via
//! HTTP/protobuf.
//!
//! These variables, read under the environment's prefix
//! (`eidola_common::env`) like every other, configure it:
//! - `OTEL_EXPORTER_OTLP_ENDPOINT` — OTLP endpoint (e.g., `https://otlp-gateway-*.grafana.net/otlp`)
//! - `OTEL_EXPORTER_OTLP_HEADERS` — auth headers (e.g., `Authorization=Basic <base64>`)
//! - `OTEL_SERVICE_NAME` — overrides the default `eidola-server-gateway`
//! - `OTEL_EXPORTER_OTLP_TIMEOUT` and `OTEL_EXPORTER_OTLP_<SIGNAL>_TIMEOUT` —
//!   export timeouts, in milliseconds
//! - `RUST_LOG` — the log filter
//!
//! Everything else the SDK would read from the environment by itself is set
//! explicitly ([`OtlpConfig`]).
//!
//! # Spans are recorded everywhere and exported only on request
//!
//! Tracing runs across the whole server — the HTTP request, each database
//! round trip, each upstream call. Every recorded span reaches
//! [`AggregatingSpanProcessor`], which turns it into histogram observations
//! and drops it. An OTLP batch exporter is also registered, but it drops
//! unsampled spans, and [`ClientDirectedSampler`] leaves a trace unsampled
//! unless the request asked to be traced with a `traceparent`. Ordinary
//! traffic therefore exports no spans at all.
//!
//! The one rule for adding a span anywhere in this crate: **`skip_all`**.
//! `#[instrument]` captures every function argument as a span field by
//! default, which on this codebase would put nullifiers, spend proofs and
//! account ids into span attributes. Aggregation makes that survivable but not
//! acceptable — write `#[instrument(skip_all)]` and add fields explicitly, or
//! do not instrument.

use std::collections::HashMap;
use std::time::Duration;

use eidola_common::env::Env;
use opentelemetry::KeyValue;
use opentelemetry::trace::{Link, SpanKind, TraceContextExt, TraceId, TracerProvider as _};
use opentelemetry_otlp::{Protocol, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{SamplingDecision, SamplingResult, ShouldSample, SpanProcessor};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Sampler that records every span in process and exports only those whose
/// trace was explicitly requested.
///
/// - `RecordOnly` for a trace with no sampled parent. The span is built and
///   run through the processors, so the aggregator sees it, but its sampled
///   flag stays clear and the exporting processor drops it. This is *not*
///   `Drop`, which produces a non-recording span: `on_end` would never fire
///   and every aggregate would read zero.
/// - `RecordAndSample` when the parent context is already sampled. Middleware
///   installs a sampled remote parent for a request carrying a `traceparent`
///   that asks to be sampled; every child span then inherits it, so such a
///   request exports its whole tree rather than a disconnected fragment.
///
/// The sampler never promotes a trace on its own — it propagates the decision
/// middleware made, which keeps that decision in one place.
#[derive(Debug, Clone)]
pub struct ClientDirectedSampler;

impl ShouldSample for ClientDirectedSampler {
    fn should_sample(
        &self,
        parent_context: Option<&opentelemetry::Context>,
        _trace_id: TraceId,
        _name: &str,
        _span_kind: &SpanKind,
        _attributes: &[KeyValue],
        _links: &[Link],
    ) -> SamplingResult {
        let parent_sampled = parent_context
            .map(|cx| cx.span().span_context().is_sampled())
            .unwrap_or(false);

        SamplingResult {
            decision: if parent_sampled {
                SamplingDecision::RecordAndSample
            } else {
                SamplingDecision::RecordOnly
            },
            attributes: Vec::new(),
            trace_state: Default::default(),
        }
    }
}

/// Span processor that converts finished spans into metrics and discards them.
///
/// This is the only processor on the tracer provider, and it exports nothing —
/// `on_end` records to instruments and drops the `SpanData` on return.
#[derive(Debug, Default)]
pub struct AggregatingSpanProcessor;

impl SpanProcessor for AggregatingSpanProcessor {
    fn on_start(&self, _span: &mut opentelemetry_sdk::trace::Span, _cx: &opentelemetry::Context) {}

    fn on_end(&self, span: opentelemetry_sdk::trace::SpanData) {
        let Some(operation) = metrics::span_operation(&span.name) else {
            // The root HTTP span is recorded by middleware instead, with
            // richer attributes than a span name can carry. Aggregating it
            // here as well would double-count the same latency.
            return;
        };

        let duration = span
            .end_time
            .duration_since(span.start_time)
            .unwrap_or_default();

        metrics::SPAN_DURATION.record(
            duration.as_secs_f64(),
            &[
                KeyValue::new("operation", operation),
                KeyValue::new("outcome", span_outcome(&span.status)),
            ],
        );
    }

    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }

    fn shutdown_with_timeout(&self, _timeout: Duration) -> OTelSdkResult {
        Ok(())
    }
}

/// The `outcome` label a finished span aggregates under: `error` when its
/// status is an error, `ok` otherwise.
fn span_outcome(status: &opentelemetry::trace::Status) -> &'static str {
    match status {
        opentelemetry::trace::Status::Error { .. } => "error",
        _ => "ok",
    }
}

/// Marks the current span failed when `result` is an error, and returns it.
///
/// This is how an instrumented function reports failure to the aggregator,
/// which reads it as `outcome=error` on `operation.duration`. The span must
/// declare the field (`fields(otel.status_code)` beside `skip_all`), since
/// `tracing` records only declared fields. `#[instrument(err)]` would set the
/// same status, by emitting an ERROR event: an exported log line for every
/// failed call, timestamped, which is a per-request record whenever the
/// failure is one. This sets the status and emits nothing.
pub fn span_result<T, E>(result: Result<T, E>) -> Result<T, E> {
    if result.is_err() {
        tracing::Span::current().record("otel.status_code", "error");
    }
    result
}

/// Holds OTel providers so they can be shut down gracefully.
pub struct OtelGuard {
    tracer_provider: opentelemetry_sdk::trace::SdkTracerProvider,
    meter_provider: opentelemetry_sdk::metrics::SdkMeterProvider,
    logger_provider: opentelemetry_sdk::logs::SdkLoggerProvider,
}

impl OtelGuard {
    /// Flush and shut down all OTel providers.
    pub fn shutdown(self) {
        if let Err(e) = self.tracer_provider.shutdown() {
            eprintln!("otel: trace provider shutdown error: {e}");
        }
        if let Err(e) = self.meter_provider.shutdown() {
            eprintln!("otel: meter provider shutdown error: {e}");
        }
        if let Err(e) = self.logger_provider.shutdown() {
            eprintln!("otel: logger provider shutdown error: {e}");
        }
    }
}

/// Initialize telemetry: tracing subscriber with fmt layer + optional OTel layers.
///
/// Returns the OTel guard if `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
pub fn init(env: &Env) -> Option<OtelGuard> {
    let otel = OtlpConfig::read(|name| env.var(name)).map(init_otel_providers);

    let fmt_layer = tracing_subscriber::fmt::layer();

    let env_filter = log_filter(&env.var("RUST_LOG").unwrap_or_default());

    let (otel_trace_layer, otel_log_layer) = match &otel {
        Some(guard) => {
            let trace_layer =
                otel_trace_layer(guard.tracer_provider.tracer("eidola-server-gateway"));
            let log_layer = opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new(
                &guard.logger_provider,
            );
            (Some(trace_layer), Some(log_layer))
        }
        None => (None, None),
    };

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(otel_trace_layer)
        .with(otel_log_layer)
        .init();

    otel
}

/// The event and span filter: `EnvFilter::from_default_env`, over the
/// prefixed `RUST_LOG`, with this crate at `info` and `hyper` at `warn`.
fn log_filter(rust_log: &str) -> EnvFilter {
    EnvFilter::builder()
        .with_default_directive(LevelFilter::ERROR.into())
        .parse_lossy(rust_log)
        .add_directive("eidola_server_gateway=info".parse().unwrap())
        .add_directive("hyper=warn".parse().unwrap())
}

/// The OTel trace layer exactly as production runs it.
///
/// Context activation (the `tracing-opentelemetry` default) makes entering
/// a tracing span attach its OpenTelemetry context to the ambient
/// task-local — and the SDK logger stamps that context's trace and span
/// IDs onto every log record it exports, sampled or not. That would give
/// each request's log lines a shared trace id: exactly the "span context
/// by which to group them back into a request" that privacy-guarantees.md
/// §3.3 rules out. Nothing in this server consumes the ambient context
/// (no propagator, no OTel-API spans), so activation buys nothing and
/// stays off. The `exported_logs_carry_no_trace_context` test builds its
/// stack through this same function, so the configuration under test is
/// the configuration deployed.
fn otel_trace_layer<S>(
    tracer: opentelemetry_sdk::trace::SdkTracer,
) -> tracing_opentelemetry::OpenTelemetryLayer<S, opentelemetry_sdk::trace::SdkTracer>
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    tracing_opentelemetry::layer()
        .with_tracer(tracer)
        .with_context_activation(false)
}

/// Everything the OTLP exporters, the SDK's processors and the resource are
/// built from, read by us under the environment's prefix and handed over
/// explicitly.
///
/// Left to themselves, the exporter and SDK builders read the bare `OTEL_*`
/// names: the endpoint, protocol, timeout and temporality, the batch
/// processors' sizes and delays, the metric export interval, span limits, the
/// sampler, and the resource's `OTEL_SERVICE_NAME` and
/// `OTEL_RESOURCE_ATTRIBUTES`. Each is set explicitly in
/// [`init_otel_providers`], which overrides the variable, so no unprefixed
/// value configures any of them. Compression is not settable to "none", and
/// needs no setting: the exporter is built without its gzip and zstd
/// features, so it ignores a compression variable, with a warning.
///
/// One read cannot be closed from outside: the HTTP exporter merges the
/// headers in the bare `OTEL_EXPORTER_OTLP_HEADERS` (or
/// `OTEL_EXPORTER_OTLP_<SIGNAL>_HEADERS`) over the headers it is given, with
/// no setting to stop it. A container receives only the variables its
/// measured config names, so a bare name reaches it only if the config
/// declares one.
struct OtlpConfig {
    /// The base endpoint; each signal's path is appended to it.
    endpoint: String,
    /// `OTEL_EXPORTER_OTLP_HEADERS` as `key=value` pairs, values still
    /// percent-encoded: `with_headers` decodes them, as the exporter decodes
    /// the variable.
    headers: HashMap<String, String>,
    service_name: String,
    /// Per signal: traces, metrics, logs.
    timeouts: [Duration; 3],
}

impl OtlpConfig {
    /// The configuration `lookup` describes, or `None` when it names no
    /// endpoint. `lookup` answers for one variable name, already prefixed.
    fn read(lookup: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let endpoint =
            lookup(opentelemetry_otlp::OTEL_EXPORTER_OTLP_ENDPOINT).filter(|s| !s.is_empty())?;
        let headers = lookup(opentelemetry_otlp::OTEL_EXPORTER_OTLP_HEADERS)
            .map(|raw| otlp_headers(&raw))
            .unwrap_or_default();
        let service_name = lookup("OTEL_SERVICE_NAME")
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "eidola-server-gateway".to_string());
        let timeouts = [
            opentelemetry_otlp::OTEL_EXPORTER_OTLP_TRACES_TIMEOUT,
            opentelemetry_otlp::OTEL_EXPORTER_OTLP_METRICS_TIMEOUT,
            opentelemetry_otlp::OTEL_EXPORTER_OTLP_LOGS_TIMEOUT,
        ]
        .map(|signal_var| otlp_timeout(signal_var, &lookup));
        Some(Self {
            endpoint,
            headers,
            service_name,
            timeouts,
        })
    }

    /// One signal's URL: the base endpoint with the signal's path appended,
    /// as the exporter appends it to `OTEL_EXPORTER_OTLP_ENDPOINT` (a
    /// programmatic endpoint is taken as the whole URL).
    fn signal_endpoint(&self, path: &str) -> String {
        match self.endpoint.strip_suffix('/') {
            Some(base) => format!("{base}{path}"),
            None => format!("{}{path}", self.endpoint),
        }
    }
}

/// `key=value` pairs, comma-separated, as the exporter splits its headers
/// variable: whitespace trimmed, empty keys and values skipped.
fn otlp_headers(raw: &str) -> HashMap<String, String> {
    raw.split_terminator(',')
        .filter_map(|pair| {
            let (key, value) = pair.split_once('=')?;
            let (key, value) = (key.trim(), value.trim());
            (!key.is_empty() && !value.is_empty()).then(|| (key.to_string(), value.to_string()))
        })
        .collect()
}

/// The HTTP clients the OTLP exporters send through, one per signal.
///
/// Left to itself the exporter builds a default `reqwest` 0.13 client, which
/// verifies against the platform trust store — and the server runs `FROM
/// scratch` with none, so that client cannot be built and an HTTPS collector
/// (Grafana Cloud in production) is unreachable. These clients carry the
/// bundled Mozilla roots every other outbound client here uses
/// ([`crate::tls_config`]). A client handed to the exporter does not get the
/// exporter's timeout, so each one carries its signal's ([`otlp_timeout`]),
/// and none reads a proxy from the environment. Built on their own thread, as
/// the exporter builds its default, because a blocking client must not be
/// constructed inside the tokio runtime `main` runs in.
struct OtlpHttpClients {
    traces: reqwest::blocking::Client,
    metrics: reqwest::blocking::Client,
    logs: reqwest::blocking::Client,
}

fn otlp_http_clients(timeouts: [Duration; 3]) -> OtlpHttpClients {
    std::thread::spawn(move || {
        let [traces, metrics, logs] = timeouts.map(|timeout| {
            reqwest::blocking::Client::builder()
                .tls_backend_preconfigured(crate::tls_config())
                .no_proxy()
                .timeout(timeout)
                .build()
                .expect("failed to build OTLP HTTP client")
        });
        OtlpHttpClients {
            traces,
            metrics,
            logs,
        }
    })
    .join()
    .expect("OTLP HTTP client thread panicked")
}

/// The timeout the OTLP exporter would resolve for one signal: the signal's
/// variable, else `OTEL_EXPORTER_OTLP_TIMEOUT`, else the 10 s default —
/// milliseconds, and an unparsable value is skipped. The exporter's resolver
/// is private and reads the bare names, so this mirrors it over `lookup`.
fn otlp_timeout(signal_var: &str, lookup: impl Fn(&str) -> Option<String>) -> Duration {
    [signal_var, opentelemetry_otlp::OTEL_EXPORTER_OTLP_TIMEOUT]
        .into_iter()
        .find_map(|name| lookup(name)?.parse().ok())
        .map(Duration::from_millis)
        .unwrap_or(opentelemetry_otlp::OTEL_EXPORTER_OTLP_TIMEOUT_DEFAULT)
}

/// Create OTel providers for traces, metrics, and logs via OTLP/HTTP.
fn init_otel_providers(config: OtlpConfig) -> OtelGuard {
    let http_clients = otlp_http_clients(config.timeouts);
    let [traces_timeout, metrics_timeout, logs_timeout] = config.timeouts;

    // Built empty, not from `Resource::builder()`, whose detectors read
    // `OTEL_SERVICE_NAME` and `OTEL_RESOURCE_ATTRIBUTES`; the SDK's own
    // attributes (`telemetry.sdk.*`) read nothing.
    let resource = Resource::builder_empty()
        .with_detector(Box::new(
            opentelemetry_sdk::resource::TelemetryResourceDetector,
        ))
        .with_service_name(config.service_name.clone())
        .build();

    // --- Traces ---
    // Two processors, and which one a span reaches is decided by its sampled
    // flag. `AggregatingSpanProcessor` sees every recorded span and turns it
    // into metrics. The batch exporter filters on `is_sampled` before
    // batching, and `ClientDirectedSampler` marks a trace sampled only for a
    // request that asked to be traced, so the export path stays empty for
    // ordinary traffic.
    let trace_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(config.signal_endpoint("/v1/traces"))
        .with_timeout(traces_timeout)
        .with_headers(config.headers.clone())
        .with_http_client(http_clients.traces)
        .build()
        .expect("failed to create OTLP trace exporter");

    // The batch configuration's defaults read `OTEL_BSP_*`; each field that
    // takes effect is set back to the SDK's default here.
    let trace_batch = opentelemetry_sdk::trace::BatchSpanProcessor::builder(trace_exporter)
        .with_batch_config(
            opentelemetry_sdk::trace::BatchConfigBuilder::default()
                .with_max_queue_size(opentelemetry_sdk::trace::OTEL_BSP_MAX_QUEUE_SIZE_DEFAULT)
                .with_scheduled_delay(opentelemetry_sdk::trace::OTEL_BSP_SCHEDULE_DELAY_DEFAULT)
                .with_max_export_batch_size(
                    opentelemetry_sdk::trace::OTEL_BSP_MAX_EXPORT_BATCH_SIZE_DEFAULT,
                )
                .build(),
        )
        .build();

    let tracer_provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_resource(resource.clone())
        .with_sampler(ClientDirectedSampler)
        .with_span_limits(opentelemetry_sdk::trace::SpanLimits::default())
        .with_span_processor(AggregatingSpanProcessor)
        .with_span_processor(trace_batch)
        .build();

    // --- Metrics ---
    let metrics_exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(config.signal_endpoint("/v1/metrics"))
        .with_timeout(metrics_timeout)
        .with_headers(config.headers.clone())
        .with_http_client(http_clients.metrics)
        .with_temporality(opentelemetry_sdk::metrics::Temporality::Cumulative)
        .build()
        .expect("failed to create OTLP metrics exporter");

    let meter_reader = opentelemetry_sdk::metrics::PeriodicReader::builder(metrics_exporter)
        .with_interval(opentelemetry_sdk::metrics::OTEL_METRIC_EXPORT_INTERVAL_DEFAULT)
        .build();

    let meter_provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
        .with_resource(resource.clone())
        .with_reader(meter_reader)
        .build();

    opentelemetry::global::set_meter_provider(meter_provider.clone());

    // --- Logs ---
    let log_exporter = opentelemetry_otlp::LogExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(config.signal_endpoint("/v1/logs"))
        .with_timeout(logs_timeout)
        .with_headers(config.headers)
        .with_http_client(http_clients.logs)
        .build()
        .expect("failed to create OTLP log exporter");

    // As for spans: the defaults read `OTEL_BLRP_*`, so each field is set.
    let log_batch = opentelemetry_sdk::logs::BatchLogProcessor::builder(log_exporter)
        .with_batch_config(
            opentelemetry_sdk::logs::BatchConfigBuilder::default()
                .with_max_queue_size(opentelemetry_sdk::logs::OTEL_BLRP_MAX_QUEUE_SIZE_DEFAULT)
                .with_scheduled_delay(opentelemetry_sdk::logs::OTEL_BLRP_SCHEDULE_DELAY_DEFAULT)
                .with_max_export_batch_size(
                    opentelemetry_sdk::logs::OTEL_BLRP_MAX_EXPORT_BATCH_SIZE_DEFAULT,
                )
                .build(),
        )
        .build();

    let logger_provider = opentelemetry_sdk::logs::SdkLoggerProvider::builder()
        .with_resource(resource)
        .with_log_processor(log_batch)
        .build();

    OtelGuard {
        tracer_provider,
        meter_provider,
        logger_provider,
    }
}

// ---------------------------------------------------------------------------
// Metric instruments
// ---------------------------------------------------------------------------

/// Centralized metric instruments. Using `opentelemetry::global::meter()` ensures
/// these are no-ops when OTel is not configured.
///
/// # Metrics are the inference path's only observability
///
/// Two rules hold for anything added here:
///
/// 1. **No caller-derived label values.** Every attribute must come from a
///    fixed, server-controlled set. `model` qualifies only because
///    `Catalog::resolve` resolves the requested reference to a catalog
///    row's weights hash before it is used. An unbounded label creates a per-request time series,
///    which is a per-request record wearing a metric's clothes.
/// 2. **Prefer a ratio to a duration.** A stream's wall time is length × rate,
///    so it carries the response's size. Dividing by the token count cancels
///    the size term and leaves the performance term — which is the part worth
///    alerting on anyway. `CHAT_OUTPUT_RATE` exists for that reason.
///
/// Every histogram sets its bucket boundaries explicitly, because the SDK
/// default ladder is millisecond-scaled and every duration instrument here
/// records seconds. See `SDK_DEFAULT_BUCKETS`.
pub mod metrics {
    use opentelemetry::KeyValue;
    use opentelemetry::metrics::{Counter, Gauge, Histogram};
    use std::sync::LazyLock;

    fn meter() -> opentelemetry::metrics::Meter {
        opentelemetry::global::meter("eidola-server-gateway")
    }

    // The SDK's default explicit-bucket boundaries, reproduced for reference
    // (`opentelemetry_sdk::metrics::pipeline`). Any histogram built without
    // `.with_boundaries()` gets these:
    //
    //   0, 5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000
    //
    // The ladder is scaled for **milliseconds** while every duration
    // instrument here records **seconds**, so on the defaults a 4-second
    // request and a 5-millisecond one share the first bucket and the whole
    // realistic range collapses into it. Any duration histogram added here
    // must set its own boundaries.

    /// Second-scale buckets for request and inference latencies, spanning
    /// "fast local hop" to "something is badly wrong".
    const LATENCY_BUCKETS: &[f64] = &[
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
    ];

    /// Buckets for output token rate, in tokens/second, spanning a stalled
    /// stream through a fast small model.
    const RATE_BUCKETS: &[f64] = &[1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0];

    /// Buckets for the largest gap between consecutive content chunks, in
    /// seconds. Tight at the low end because that is where a stall shows up.
    const GAP_BUCKETS: &[f64] = &[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 15.0, 60.0];

    /// Every span name the aggregator will emit a metric series for.
    ///
    /// This list *is* the label domain of `SPAN_DURATION`, which is why it
    /// exists. A span name becomes a metric attribute, so deriving the label
    /// from the name directly would let any future span — or any span whose
    /// name interpolates a value — create an unbounded set of time series.
    /// Resolving through a fixed list makes that impossible by construction
    /// rather than by review. Every span this crate opens is either here or
    /// in [`UNTRACKED_SPANS`], which a test checks against the source.
    pub(super) const TRACKED_SPANS: &[&str] = &[
        "account.reconcile",
        "account.reconcile_status",
        "db.find_credential_issuance",
        "db.get_account",
        "db.get_available_balance",
        "db.get_issuer_key",
        "db.get_refund_token",
        "db.get_valid_issuer_keys",
        "db.has_stripe_ledger_history",
        "db.prune_expired_issuances",
        "db.prune_expired_nullifiers",
        "db.record_credential_issuance",
        "db.record_nullifiers",
        "db.replace_refund_token",
        "db.resolve_issuance_off_current_key",
        "db.store_refund_token",
        "engine.chat_stream",
    ];

    /// Spans the aggregator deliberately ignores, each with the reason.
    pub(super) const UNTRACKED_SPANS: &[(&str, &str)] = &[(
        "http.request",
        "the root request span: middleware records its latency itself, with \
         method, route and status attached, and aggregating it here too would \
         double-count it",
    )];

    /// Resolve a span name to its metric label, or `None` for a span the
    /// aggregator should ignore.
    ///
    /// Unknown names collapse to `other` rather than being dropped: a new span
    /// should be visible as *something* so it prompts a `TRACKED_SPANS` entry,
    /// but it must never mint a series of its own.
    pub(super) fn span_operation(name: &str) -> Option<&'static str> {
        if UNTRACKED_SPANS
            .iter()
            .any(|(untracked, _)| *untracked == name)
        {
            return None;
        }
        Some(
            TRACKED_SPANS
                .iter()
                .copied()
                .find(|tracked| *tracked == name)
                .unwrap_or("other"),
        )
    }

    /// Duration of an internal operation, by `operation` and `outcome`.
    ///
    /// Fed by `AggregatingSpanProcessor` from every recorded span. This is the
    /// aggregate substitute for reading a trace: it answers "where does this
    /// workload spend its time" without retaining any individual request.
    pub static SPAN_DURATION: LazyLock<Histogram<f64>> = LazyLock::new(|| {
        meter()
            .f64_histogram("operation.duration")
            .with_description("Duration of an internal operation, aggregated from spans")
            .with_unit("s")
            .with_boundaries(LATENCY_BUCKETS.to_vec())
            .build()
    });

    /// Inbound `traceparent` headers, by `outcome`: `sampled` (honored, the
    /// request's trace is exported), `not_sampled` (the header's sampled flag
    /// was clear, so the request is treated as untraced), or `malformed`.
    ///
    /// Watch the total. A sustained non-zero rate means something is sending
    /// `traceparent` on requests it does not intend to trace — most likely
    /// generic client-side OpenTelemetry auto-instrumentation, which injects
    /// the header on every outbound call.
    pub static TRACEPARENT_RECEIVED: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("traceparent.received")
            .with_description("Inbound traceparent headers by outcome")
            .build()
    });

    /// HTTP request duration in seconds, by method, route and status.
    pub static HTTP_REQUEST_DURATION: LazyLock<Histogram<f64>> = LazyLock::new(|| {
        meter()
            .f64_histogram("http.server.request.duration")
            .with_description("Duration of HTTP server requests")
            .with_unit("s")
            .with_boundaries(LATENCY_BUCKETS.to_vec())
            .build()
    });

    /// Total HTTP requests served.
    pub static HTTP_REQUEST_COUNT: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("http.server.request.count")
            .with_description("Total HTTP server requests")
            .build()
    });

    /// Total tokens processed in chat completions (by model and type):
    /// `prompt` and `completion` as the engine reported them, and
    /// `declared_cached_prompt`, the prompt tokens a request declared cached
    /// and was charged at the cached rate. What an engine reports about its
    /// cache is never counted.
    ///
    /// Deliberately success-only: a disconnected or errored stream reports
    /// no usage (there is nothing trustworthy to count), so this totals
    /// tokens from completed exchanges. It therefore undercounts billed
    /// traffic by exactly the abnormally-terminated streams — a
    /// tokens-vs-credits reconciliation must expect that drift.
    pub static CHAT_TOKENS: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("chat.completion.tokens")
            .with_description("Chat completion tokens processed")
            .build()
    });

    /// Total chat completion requests reaching the handler (by model,
    /// stream, status, reason). `status` is `ok` (2xx response opened),
    /// `error` (a refund-bearing error response built after the credential
    /// was spent), or `rejected` (pre-flight refusal: a request not asking
    /// for a stream, an invalid or duplicate credential). `reason` is `none`
    /// for `ok`, and otherwise the refusal's fixed class
    /// (`ServerError::label`): `bad_request`, `payment_required`,
    /// `unavailable` (no engine took it), an engine's error type resolved
    /// against the known list (`timeout` for the request's deadline), and
    /// so on. The `model` label is always the catalog resolution of the
    /// requested id, with `other` for anything unrecognized — never the
    /// caller's string.
    pub static CHAT_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("chat.completion.requests")
            .with_description("Chat completion requests")
            .build()
    });

    /// Time to first content chunk on a streaming completion, by model.
    ///
    /// The single best indicator of prefill and queueing trouble: it rises
    /// when the upstream is admission-queueing or when prompts get large,
    /// and it is independent of how long the response turns out to be.
    pub static CHAT_TTFT: LazyLock<Histogram<f64>> = LazyLock::new(|| {
        meter()
            .f64_histogram("chat.completion.ttft")
            .with_description("Time from upstream dispatch to first content chunk")
            .with_unit("s")
            .with_boundaries(LATENCY_BUCKETS.to_vec())
            .build()
    });

    /// Wall time of a streaming completion from dispatch to final chunk, by
    /// model.
    pub static CHAT_STREAM_DURATION: LazyLock<Histogram<f64>> = LazyLock::new(|| {
        meter()
            .f64_histogram("chat.completion.stream.duration")
            .with_description("Wall time of a streaming chat completion")
            .with_unit("s")
            .with_boundaries(LATENCY_BUCKETS.to_vec())
            .build()
    });

    /// Sustained output rate of a streaming completion, in tokens per second,
    /// by model.
    ///
    /// The primary performance signal, and the one to alert on. Because it
    /// divides out the response length, a shift here means throughput
    /// genuinely changed rather than that responses happened to get longer —
    /// which also means it carries less about any individual request than the
    /// duration it is derived from.
    pub static CHAT_OUTPUT_RATE: LazyLock<Histogram<f64>> = LazyLock::new(|| {
        meter()
            .f64_histogram("chat.completion.output.rate")
            .with_description("Completion tokens per second of stream wall time")
            .with_unit("{token}/s")
            .with_boundaries(RATE_BUCKETS.to_vec())
            .build()
    });

    /// Largest gap between consecutive content chunks in a stream, by model.
    ///
    /// Catches what an average cannot: a stream that delivers its tokens in
    /// two bursts around a ten-second stall has an unremarkable rate and an
    /// unremarkable duration, and is badly broken.
    pub static CHAT_INTER_TOKEN_GAP_MAX: LazyLock<Histogram<f64>> = LazyLock::new(|| {
        meter()
            .f64_histogram("chat.completion.inter_token.gap.max")
            .with_description("Largest gap between consecutive streamed content chunks")
            .with_unit("s")
            .with_boundaries(GAP_BUCKETS.to_vec())
            .build()
    });

    /// How streams ended, by model and `reason`: `done` (clean completion),
    /// `client_disconnect` (the client hung up mid-stream),
    /// `upstream_error` (the upstream failed after the stream opened), or
    /// `channel_closed` (the upstream channel ended without a Done event).
    ///
    /// This is the classes-of-bugs instrument. A rising `channel_closed` rate
    /// is a gateway or engine defect; a rising `client_disconnect` rate is a
    /// client or network story.
    pub static CHAT_STREAM_OUTCOME: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("chat.completion.stream.outcome")
            .with_description("Stream terminations by reason")
            .build()
    });

    /// Attempts to hand a chat request to an engine, by `model` and
    /// `outcome`: `response` (the engine answered with a stream),
    /// `no_headers` (no response headers in time), `no_response` (no
    /// attested connection, or one that closed under the request),
    /// `misconfigured` (the engine refused the gateway's token, weights or
    /// model; its breaker opens), `overloaded` (at capacity; the next engine
    /// is tried), `refused` (any other answer, returned to the client), and
    /// `none_available` (no engine was left to try, counted once per
    /// request).
    pub static ENGINE_DISPATCH_ATTEMPTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("engine.dispatch.attempts")
            .with_description("Attempts to dispatch a chat request to an engine, by outcome")
            .build()
    });

    /// Engine streams that failed after they opened, by `model` and
    /// `reason`: `silent` (nothing for the idle limit), `read` (the
    /// connection broke), `not_utf8`, `unknown_event`, `unparsed_chunk` (a
    /// chunk this gateway cannot read), `error_event` (the node's own error
    /// frame), or `incomplete` (the body ended without `[DONE]`). Each is
    /// also counted against the engine's health.
    pub static ENGINE_STREAM_FAILURES: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("engine.stream.failures")
            .with_description("Engine streams that failed after opening, by reason")
            .build()
    });

    /// Credential issuance requests, by `outcome`: `issued`, `replayed` (a
    /// retry answered from the stored response), `insufficient` (balance),
    /// `terms_required`, `key_retired`, `key_pending` (the named key is not
    /// this gateway's issuing key yet), `closed` (the account is closing or
    /// closed), `mismatch` (an idempotency key reused for another request),
    /// `invalid` (a malformed request), or `failed` (anything else).
    pub static CREDENTIAL_ISSUANCE: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("credentials.issuance")
            .with_description("Credential issuance requests, by outcome")
            .build()
    });

    /// Refunds settled for spent inputs, by `outcome`: `issued` (the refund
    /// owed), `zero_fallback` (that refund could not be signed, and one
    /// returning nothing was issued instead), or `failed` (not even that).
    /// Any `zero_fallback` or `failed` is an incident.
    pub static CREDENTIAL_REFUNDS: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("credentials.refunds")
            .with_description("Refunds settled for spent inputs, by outcome")
            .build()
    });

    /// Refunds that were issued but could not be stored for recovery. The
    /// refund still reaches the client in its response; one whose response
    /// is lost cannot be recovered.
    pub static REFUND_STORE_FAILURES: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("credentials.refund_store.failures")
            .with_description("Issued refunds that could not be stored for recovery")
            .build()
    });

    /// Refund recoveries whose stored token did not open under the
    /// presented refund key, answered as if nothing were stored.
    pub static REFUND_RECOVERY_UNOPENED: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("credentials.refund_recovery.unopened")
            .with_description("Refund recoveries whose stored token did not open")
            .build()
    });

    /// Request bodies refused before a handler ran, by `payload` (the
    /// expected body type's name, fixed in code) and `class`: `too_large`,
    /// `unreadable`, `read_timeout`, `no_permit` (no large-body permit
    /// free), `too_deep`, `too_many_values`, `data`, `syntax`,
    /// `missing_content_type`, `bytes` or `other`.
    pub static BODY_REJECTED: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("http.server.body.rejected")
            .with_description("Request bodies refused before the handler ran, by class")
            .build()
    });

    /// Count a refused request body. `payload` is a type name: its last
    /// path segment is the label.
    pub fn body_rejected(payload: &'static str, class: &'static str) {
        let payload = payload.rsplit("::").next().unwrap_or(payload);
        BODY_REJECTED.add(
            1,
            &[
                KeyValue::new("payload", payload),
                KeyValue::new("class", class),
            ],
        );
    }

    /// Register the observable gauges read from the engine router at each
    /// export: `engine.upstreams{model, state=routable|unroutable}`, every
    /// pinned model reported, with zeros when it has no upstream; and
    /// `engine.placement.rows{reason=usable|disabled|unpinned_model|
    /// unpinned_deployment|invalid_url|duplicate}`, the last placement
    /// refresh's rows.
    pub fn register_engine_router(router: crate::engine_router::EngineRouter) {
        let placement = router.clone();
        meter()
            .u64_observable_gauge("engine.placement.rows")
            .with_description("Engine placement rows at the last refresh, by reason")
            .with_callback(move |gauge| {
                for (reason, rows) in placement.placement_rows() {
                    gauge.observe(rows, &[KeyValue::new("reason", reason)]);
                }
            })
            .build();
        meter()
            .u64_observable_gauge("engine.upstreams")
            .with_description("Engine upstreams known to this gateway, by model and state")
            .with_callback(move |gauge| {
                for (model, routable, unroutable) in router.upstream_states() {
                    gauge.observe(
                        routable,
                        &[
                            KeyValue::new("model", model.clone()),
                            KeyValue::new("state", "routable"),
                        ],
                    );
                    gauge.observe(
                        unroutable,
                        &[
                            KeyValue::new("model", model),
                            KeyValue::new("state", "unroutable"),
                        ],
                    );
                }
            })
            .build();
    }

    /// Register the observable gauges read from the database pool at each
    /// export: `db.pool.connections{state=in_use|idle}` and
    /// `db.pool.waiting`, the requests waiting for a connection.
    pub fn register_db_pool(pool: deadpool_postgres::Pool) {
        let connections = pool.clone();
        meter()
            .u64_observable_gauge("db.pool.connections")
            .with_description("Database pool connections, by state")
            .with_callback(move |gauge| {
                let status = connections.status();
                let idle = status.available as u64;
                let in_use = status.size.saturating_sub(status.available) as u64;
                gauge.observe(in_use, &[KeyValue::new("state", "in_use")]);
                gauge.observe(idle, &[KeyValue::new("state", "idle")]);
            })
            .build();
        meter()
            .u64_observable_gauge("db.pool.waiting")
            .with_description("Requests waiting for a database pool connection")
            .with_callback(move |gauge| gauge.observe(pool.status().waiting as u64, &[]))
            .build();
    }

    /// Terms-feed polls, by `document` (the fixed list in `terms_feed`) and
    /// `outcome`: `recorded` (a new version), `unchanged`, `conflict` (a
    /// published version's bytes changed without a new version),
    /// `record_failed` (the database write failed) or `fetch_failed`.
    pub static TERMS_FEED_POLLS: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("terms.feed.polls")
            .with_description("Terms-feed polls per document, by outcome")
            .build()
    });

    /// Signed clock drift between the server and the database, in
    /// seconds. Positive values mean the database is ahead of the
    /// server; negative values mean the server is ahead of the
    /// database. Updated after every successful `check_clock_skew`
    /// call (boot + each key rotation iteration). Alert on
    /// `abs(db.clock.skew.seconds) > 5` for sustained periods to catch
    /// NTP/chrony failures before they reach the 10s hard threshold.
    pub static DB_CLOCK_SKEW_SECONDS: LazyLock<Gauge<f64>> = LazyLock::new(|| {
        meter()
            .f64_gauge("db.clock.skew.seconds")
            .with_description(
                "Signed clock drift between server and database (positive = db ahead)",
            )
            .with_unit("s")
            .build()
    });

    /// Count of failed `check_clock_skew` invocations, labeled by
    /// `reason`: `pool` (could not check out a connection), `query`
    /// (the SELECT failed), or `exceeded` (drift exceeded the
    /// threshold). Any non-zero rate is operator-actionable.
    pub static DB_CLOCK_SKEW_CHECK_FAILURES: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("db.clock.skew.check.failures")
            .with_description("Failed database clock skew checks, by reason")
            .build()
    });
}

/// A tracing stack built like production's (the same filter, trace layer and
/// log bridge), recording what reaches the vendor: each finished span's name
/// and the `outcome` the aggregator would give it, and every log record.
#[cfg(test)]
pub(crate) mod capture {
    use std::sync::{Arc, Mutex};

    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::error::OTelSdkResult;
    use opentelemetry_sdk::logs::InMemoryLogExporter;
    use opentelemetry_sdk::trace::SpanProcessor;
    use tracing_subscriber::layer::SubscriberExt;

    #[derive(Debug, Default, Clone)]
    struct Spans(Arc<Mutex<Vec<(String, &'static str)>>>);

    impl SpanProcessor for Spans {
        fn on_start(
            &self,
            _span: &mut opentelemetry_sdk::trace::Span,
            _cx: &opentelemetry::Context,
        ) {
        }

        fn on_end(&self, span: opentelemetry_sdk::trace::SpanData) {
            self.0
                .lock()
                .unwrap()
                .push((span.name.to_string(), super::span_outcome(&span.status)));
        }

        fn force_flush(&self) -> OTelSdkResult {
            Ok(())
        }

        fn shutdown_with_timeout(&self, _timeout: std::time::Duration) -> OTelSdkResult {
            Ok(())
        }
    }

    /// What `run` sent toward the vendor.
    pub(crate) struct Captured {
        /// Each finished span: its name and aggregated `outcome`.
        pub spans: Vec<(String, &'static str)>,
        /// How many log records were emitted past the filter.
        pub logs: usize,
    }

    /// Run `run` under the stack, on this thread.
    pub(crate) fn capture(run: impl FnOnce()) -> Captured {
        let spans = Spans::default();
        let logs = InMemoryLogExporter::default();
        let logger_provider = opentelemetry_sdk::logs::SdkLoggerProvider::builder()
            .with_simple_exporter(logs.clone())
            .build();
        let tracer_provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_sampler(super::ClientDirectedSampler)
            .with_span_processor(spans.clone())
            .build();
        let subscriber = tracing_subscriber::registry()
            .with(super::log_filter(""))
            .with(super::otel_trace_layer(
                tracer_provider.tracer("eidola-server-gateway"),
            ))
            .with(
                opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new(
                    &logger_provider,
                ),
            );
        tracing::subscriber::with_default(subscriber, run);
        let logs = logs.get_emitted_logs().expect("reading emitted logs").len();
        let spans = spans.0.lock().unwrap().clone();
        Captured { spans, logs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tracing::instrument(skip_all, name = "db.get_account", fields(otel.status_code))]
    fn instrumented(fail: bool) -> Result<(), crate::error::ServerError> {
        span_result(if fail {
            Err(crate::error::ServerError::NotFound {
                message: "account not found".to_string(),
            })
        } else {
            Ok(())
        })
    }

    /// A failed call aggregates as `outcome=error` without a log event:
    /// `span_result` sets the span's status, which is what the aggregator
    /// reads, and writes no record the vendor would receive.
    #[test]
    fn a_failed_call_is_an_error_outcome_and_no_log_line() {
        let captured = capture::capture(|| {
            let _ = instrumented(true);
            let _ = instrumented(false);
        });
        assert_eq!(
            captured.spans,
            vec![
                ("db.get_account".to_string(), "error"),
                ("db.get_account".to_string(), "ok"),
            ]
        );
        assert_eq!(captured.logs, 0, "a failed call wrote a log record");
    }

    /// Every span this crate opens is tracked by the aggregator or
    /// deliberately left out of it, with a reason; every `#[instrument]`
    /// names its span, skips its arguments, and reports failure through
    /// `span_result` rather than `err`, which logs every failed call.
    #[test]
    fn every_span_in_the_crate_is_tracked_or_excluded() {
        fn sources(dir: &std::path::Path, out: &mut Vec<(String, String)>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    sources(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let text = std::fs::read_to_string(&path).unwrap();
                    out.push((path.display().to_string(), text));
                }
            }
        }
        fn first_literal(text: &str) -> Option<&str> {
            let rest = text.trim_start().strip_prefix('"')?;
            rest.split_once('"').map(|(name, _)| name)
        }

        let mut files = Vec::new();
        sources(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
            &mut files,
        );
        let untracked: Vec<&str> = metrics::UNTRACKED_SPANS.iter().map(|(n, _)| *n).collect();
        // Spelled in pieces, so this test's own source holds neither needle.
        let attribute_needle = ["#[tracing", "::instrument("].concat();
        let macro_needle = ["_span", "!("].concat();
        let mut names = Vec::new();
        for (path, text) in &files {
            for (at, _) in text.match_indices(&attribute_needle) {
                let attribute = &text[at..at + text[at..].find(")]").unwrap()];
                assert!(attribute.contains("skip_all"), "{path}: {attribute}");
                assert!(
                    !attribute.contains(", err") && !attribute.contains("(err"),
                    "{path}: {attribute} logs every failed call; use span_result"
                );
                let name = attribute
                    .split_once("name = ")
                    .and_then(|(_, rest)| first_literal(rest))
                    .unwrap_or_else(|| panic!("{path}: {attribute} names no span"));
                names.push(name.to_string());
            }
            for (at, mac) in text.match_indices(&macro_needle) {
                if let Some(name) = first_literal(&text[at + mac.len()..]) {
                    names.push(name.to_string());
                }
            }
        }
        assert!(
            names.iter().any(|n| n == "engine.chat_stream"),
            "the scan found the crate's spans"
        );
        for name in &names {
            assert!(
                metrics::TRACKED_SPANS.contains(&name.as_str())
                    || untracked.contains(&name.as_str()),
                "span `{name}` is neither tracked nor excluded"
            );
        }
        for tracked in metrics::TRACKED_SPANS {
            assert!(
                names.iter().any(|n| n == tracked),
                "`{tracked}` is tracked but no span has that name"
            );
        }
    }

    /// The supplied OTLP clients carry the timeout the exporter would have
    /// resolved itself: signal variable over the general one over 10 s, in
    /// milliseconds, skipping a value that does not parse.
    #[test]
    fn otlp_timeout_resolves_like_the_exporter() {
        use opentelemetry_otlp::{
            OTEL_EXPORTER_OTLP_TIMEOUT, OTEL_EXPORTER_OTLP_TIMEOUT_DEFAULT,
            OTEL_EXPORTER_OTLP_TRACES_TIMEOUT,
        };
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| v.to_string())
            }
        };
        let signal = OTEL_EXPORTER_OTLP_TRACES_TIMEOUT;
        assert_eq!(
            otlp_timeout(signal, env(&[])),
            OTEL_EXPORTER_OTLP_TIMEOUT_DEFAULT
        );
        assert_eq!(
            otlp_timeout(signal, env(&[(OTEL_EXPORTER_OTLP_TIMEOUT, "2500")])),
            Duration::from_millis(2500)
        );
        assert_eq!(
            otlp_timeout(
                signal,
                env(&[
                    (OTEL_EXPORTER_OTLP_TRACES_TIMEOUT, "3000"),
                    (OTEL_EXPORTER_OTLP_TIMEOUT, "2500")
                ])
            ),
            Duration::from_millis(3000)
        );
        assert_eq!(
            otlp_timeout(
                signal,
                env(&[
                    (OTEL_EXPORTER_OTLP_TRACES_TIMEOUT, "soon"),
                    (OTEL_EXPORTER_OTLP_TIMEOUT, "2500")
                ])
            ),
            Duration::from_millis(2500)
        );
    }

    fn vars(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name: &str| {
            pairs
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.to_string())
        }
    }

    /// Telemetry is off without an endpoint, and an empty one is none: an
    /// empty endpoint handed to the exporter would make it read the bare
    /// variable instead.
    #[test]
    fn otlp_needs_a_non_empty_endpoint() {
        assert!(OtlpConfig::read(vars(&[])).is_none());
        assert!(OtlpConfig::read(vars(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "")])).is_none());
    }

    /// The values we read are the ones the exporters get: each signal's path
    /// appended to the endpoint as the exporter appends it, the headers split
    /// as it splits them, and the service name defaulted.
    #[test]
    fn otlp_config_is_read_from_the_lookup_alone() {
        let config = OtlpConfig::read(vars(&[
            (
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "https://collector.example/otlp",
            ),
            (
                "OTEL_EXPORTER_OTLP_HEADERS",
                " Authorization = Basic%20abc , empty=, =novalue,X-Scope=tenant",
            ),
            ("OTEL_EXPORTER_OTLP_TIMEOUT", "2500"),
            ("OTEL_EXPORTER_OTLP_LOGS_TIMEOUT", "4000"),
        ]))
        .expect("an endpoint enables telemetry");
        assert_eq!(
            config.signal_endpoint("/v1/traces"),
            "https://collector.example/otlp/v1/traces"
        );
        assert_eq!(
            config.headers,
            HashMap::from([
                ("Authorization".to_string(), "Basic%20abc".to_string()),
                ("X-Scope".to_string(), "tenant".to_string()),
            ])
        );
        assert_eq!(config.service_name, "eidola-server-gateway");
        assert_eq!(
            config.timeouts,
            [
                Duration::from_millis(2500),
                Duration::from_millis(2500),
                Duration::from_millis(4000)
            ]
        );

        let config = OtlpConfig::read(vars(&[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.example/"),
            ("OTEL_SERVICE_NAME", "staging-gateway"),
        ]))
        .unwrap();
        assert_eq!(
            config.signal_endpoint("/v1/logs"),
            "https://collector.example/v1/logs"
        );
        assert_eq!(config.service_name, "staging-gateway");
        assert!(config.headers.is_empty());
    }

    /// With no parent — the shape of every ordinary request — the decision
    /// must be `RecordOnly`. That is load-bearing and easy to break by
    /// reaching for the more obvious `AlwaysOff`, which yields `Drop`: spans
    /// become non-recording, `on_end` never fires, and every aggregate
    /// silently reads zero. It would look like "tracing is off" rather than
    /// "the metrics are broken".
    #[test]
    fn unauthorized_requests_record_without_sampling() {
        let result = ClientDirectedSampler.should_sample(
            None,
            TraceId::from_bytes([1; 16]),
            "http.request",
            &SpanKind::Server,
            &[],
            &[],
        );
        assert_eq!(result.decision, SamplingDecision::RecordOnly);
    }

    /// An unknown span name must land on a shared bucket, never mint a label
    /// of its own — that is what bounds the cardinality of `SPAN_DURATION`.
    #[test]
    fn unknown_span_names_collapse_to_one_bucket() {
        assert_eq!(
            metrics::span_operation("db.record_nullifiers"),
            Some("db.record_nullifiers")
        );
        assert_eq!(metrics::span_operation("something.new"), Some("other"));
        assert_eq!(metrics::span_operation("id-42-suffixed"), Some("other"));
    }

    /// The root HTTP span is recorded by middleware with method/route/status
    /// attached; aggregating it here as well would double-count it.
    #[test]
    fn root_request_span_is_left_to_middleware() {
        assert_eq!(metrics::span_operation("http.request"), None);
    }

    /// Exported log records must carry no trace context — the §3.3 "no span
    /// context by which to group them back into a request" invariant as it
    /// applies to the OTLP log stream. The SDK logger stamps a record's
    /// `trace_context` from OpenTelemetry's *ambient* context whenever one
    /// holds an active span at the log call site, sampled or not — and
    /// `tracing-opentelemetry`'s default context activation attaches that
    /// ambient context on every span entry, which linked each request's
    /// whole log output under one trace id until [`otel_trace_layer`]
    /// turned activation off (Codex review, PR #282). The stack here is
    /// built through that same function; this test holds for ordinary
    /// `RecordOnly` traffic and client-directed traces alike, and fails if
    /// activation is ever re-enabled or an upgrade starts syncing the
    /// ambient context again.
    #[test]
    fn exported_logs_carry_no_trace_context() {
        use opentelemetry::trace::{SpanContext, SpanId, TraceFlags, TraceState};
        use opentelemetry_sdk::logs::InMemoryLogExporter;
        use tracing_opentelemetry::OpenTelemetrySpanExt;

        let exporter = InMemoryLogExporter::default();
        let logger_provider = opentelemetry_sdk::logs::SdkLoggerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let tracer_provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_sampler(ClientDirectedSampler)
            .with_span_processor(AggregatingSpanProcessor)
            .build();

        let subscriber = tracing_subscriber::registry()
            .with(otel_trace_layer(
                tracer_provider.tracer("eidola-server-gateway"),
            ))
            .with(
                opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new(
                    &logger_provider,
                ),
            );

        tracing::subscriber::with_default(subscriber, || {
            // An ordinary request: a recorded, unsampled (`RecordOnly`) span.
            {
                let span = tracing::info_span!("http.request");
                let _entered = span.enter();
                tracing::info!("log inside an ordinary request span");
            }
            // A client-directed trace: sampled remote parent, exactly as
            // middleware installs it. Sampling exports the *spans*; the
            // logs must stay uncorrelated regardless.
            {
                let span = tracing::info_span!("http.request");
                let parent = SpanContext::new(
                    TraceId::from_bytes([7; 16]),
                    SpanId::from_bytes([7; 8]),
                    TraceFlags::SAMPLED,
                    true,
                    TraceState::default(),
                );
                let _ =
                    span.set_parent(opentelemetry::Context::new().with_remote_span_context(parent));
                let _entered = span.enter();
                tracing::info!("log inside a client-directed traced span");
            }
        });

        let logs = exporter.get_emitted_logs().expect("reading emitted logs");
        assert_eq!(logs.len(), 2, "both log events must reach the exporter");
        for log in &logs {
            assert!(
                log.record.trace_context().is_none(),
                "an exported log record carries a trace context"
            );
        }
    }
}
