//! OpenTelemetry initialization for traces, metrics, and logs.
//!
//! Enabled when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Ships telemetry
//! directly to Grafana Cloud (or any OTLP-compatible endpoint) via HTTP/protobuf.
//!
//! Standard OTel env vars control the exporter:
//! - `OTEL_EXPORTER_OTLP_ENDPOINT` — OTLP endpoint (e.g., `https://otlp-gateway-*.grafana.net/otlp`)
//! - `OTEL_EXPORTER_OTLP_HEADERS` — auth headers (e.g., `Authorization=Basic <base64>`)
//! - `OTEL_SERVICE_NAME` — overrides the default `eidola-server`
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

use std::time::Duration;

use opentelemetry::KeyValue;
use opentelemetry::trace::{Link, SpanKind, TraceContextExt, TraceId, TracerProvider as _};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{SamplingDecision, SamplingResult, ShouldSample, SpanProcessor};
use tracing_subscriber::EnvFilter;
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
        let outcome = match span.status {
            opentelemetry::trace::Status::Error { .. } => "error",
            _ => "ok",
        };

        metrics::SPAN_DURATION.record(
            duration.as_secs_f64(),
            &[
                KeyValue::new("operation", operation),
                KeyValue::new("outcome", outcome),
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
pub fn init() -> Option<OtelGuard> {
    let otel = init_otel_providers();

    let fmt_layer = tracing_subscriber::fmt::layer();

    let env_filter = EnvFilter::from_default_env()
        .add_directive("eidola_server=info".parse().unwrap())
        .add_directive("hyper=warn".parse().unwrap());

    let (otel_trace_layer, otel_log_layer) = match &otel {
        Some(guard) => {
            let trace_layer = otel_trace_layer(guard.tracer_provider.tracer("eidola-server"));
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

/// Create OTel providers for traces, metrics, and logs via OTLP/HTTP.
fn init_otel_providers() -> Option<OtelGuard> {
    // Only enable when an endpoint is configured.
    std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok()?;

    let service_name =
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "eidola-server".to_string());

    let resource = Resource::builder()
        .with_attributes([KeyValue::new("service.name", service_name)])
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
        .build()
        .expect("failed to create OTLP trace exporter");

    let tracer_provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_resource(resource.clone())
        .with_sampler(ClientDirectedSampler)
        .with_span_processor(AggregatingSpanProcessor)
        .with_batch_exporter(trace_exporter)
        .build();

    // --- Metrics ---
    let metrics_exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_http()
        .build()
        .expect("failed to create OTLP metrics exporter");

    let meter_reader =
        opentelemetry_sdk::metrics::PeriodicReader::builder(metrics_exporter).build();

    let meter_provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
        .with_resource(resource.clone())
        .with_reader(meter_reader)
        .build();

    opentelemetry::global::set_meter_provider(meter_provider.clone());

    // --- Logs ---
    let log_exporter = opentelemetry_otlp::LogExporter::builder()
        .with_http()
        .build()
        .expect("failed to create OTLP log exporter");

    let logger_provider = opentelemetry_sdk::logs::SdkLoggerProvider::builder()
        .with_resource(resource)
        .with_batch_exporter(log_exporter)
        .build();

    Some(OtelGuard {
        tracer_provider,
        meter_provider,
        logger_provider,
    })
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
///    `lookup_model` resolves the requested id against the known catalog
///    before it is used. An unbounded label creates a per-request time series,
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
    use opentelemetry::metrics::{Counter, Gauge, Histogram};
    use std::sync::LazyLock;

    fn meter() -> opentelemetry::metrics::Meter {
        opentelemetry::global::meter("eidola-server")
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
    /// rather than by review.
    ///
    /// The root `http.request` span is deliberately absent: middleware records
    /// that latency itself, with method, route and status attached, and
    /// aggregating it here too would double-count it.
    const TRACKED_SPANS: &[&str] = &[
        "db.get_account",
        "db.get_available_balance",
        "db.get_issuer_key",
        "db.get_refund_token",
        "db.get_valid_issuer_keys",
        "db.insert_credit_ledger",
        "db.record_nullifier",
        "db.store_refund_token",
        "upstream.chat",
        "upstream.chat_stream",
        "webhook.process",
    ];

    /// Resolve a span name to its metric label, or `None` for a span the
    /// aggregator should ignore.
    ///
    /// Unknown names collapse to `other` rather than being dropped: a new span
    /// should be visible as *something* so it prompts a `TRACKED_SPANS` entry,
    /// but it must never mint a series of its own.
    pub(super) fn span_operation(name: &str) -> Option<&'static str> {
        if name == "http.request" {
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

    /// Stripe webhook processing outcomes, by `event_type` and `reason`.
    ///
    /// The webhook is the one inbound path we cannot opt into a trace: Stripe
    /// originates it, not us. This counter is what makes its failure modes
    /// visible in aggregate — each distinct give-up branch in `webhook.rs`
    /// increments a named reason, so a spike in any one of them is a
    /// dashboard signal rather than a log line someone has to find.
    pub static WEBHOOK_OUTCOME: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("webhook.outcome")
            .with_description("Stripe webhook processing outcomes by reason")
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

    /// Total tokens processed in chat completions (by model and type).
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
    /// stream, status). `status` is `ok` (2xx response opened), `error` (a
    /// refund-bearing error response built after the credential was
    /// spent), or `rejected` (pre-flight refusal: invalid or duplicate
    /// credential, unknown model, insufficient charge). The `model` label
    /// is always the catalog resolution of the requested id, with `other`
    /// for anything unrecognized — never the caller's string.
    pub static CHAT_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("chat.completion.requests")
            .with_description("Chat completion requests")
            .build()
    });

    /// Wall time of a non-streaming chat completion, by model: the full
    /// upstream call, since a blocking request returns nothing until the
    /// generation finishes.
    ///
    /// Deliberately success-only: failed upstream calls land in
    /// `operation.duration{operation="upstream.chat", outcome="error"}`
    /// (the `err`-instrumented span) and in the HTTP histogram by status —
    /// folding them in here would mix "how long does a completion take"
    /// with "how long until an error", which alert thresholds want apart.
    ///
    /// Kept separate from the streaming instruments on purpose. Folding both
    /// into one latency series would blend two unrelated quantities (a
    /// blocking request's duration is dominated by output length, a streaming
    /// request's by prefill) and the blend is not useful.
    pub static CHAT_DURATION: LazyLock<Histogram<f64>> = LazyLock::new(|| {
        meter()
            .f64_histogram("chat.completion.duration")
            .with_description("Wall time of a non-streaming chat completion")
            .with_unit("s")
            .with_boundaries(LATENCY_BUCKETS.to_vec())
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
    /// is a server or upstream defect; a rising `client_disconnect` rate is a
    /// client or network story.
    pub static CHAT_STREAM_OUTCOME: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("chat.completion.stream.outcome")
            .with_description("Stream terminations by reason")
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

    /// Total SEV-SNP attestations observed during outbound enclave
    /// handshakes, labeled by a coarse TCB bucket: `meets_floor`,
    /// `below_floor`, or `rollback_detected`. Includes attestations the
    /// verifier ultimately rejects, so a non-zero `below_floor` rate
    /// signals AMD has published a firmware update we haven't accepted
    /// yet, and any `rollback_detected` is an immediate red flag (a
    /// hypervisor is reporting a TCB lower than the firmware has
    /// committed to).
    pub static SNP_ATTESTATIONS: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("tinfoil.snp.attestations")
            .with_description("SEV-SNP attestations observed by the tinfoil verifier")
            .build()
    });
}

#[cfg(test)]
mod tests {
    use super::*;

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
            metrics::span_operation("db.record_nullifier"),
            Some("db.record_nullifier")
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
            .with(otel_trace_layer(tracer_provider.tracer("eidola-server")))
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
