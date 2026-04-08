//! OpenTelemetry initialization for traces, metrics, and logs.
//!
//! Enabled when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Ships telemetry
//! directly to Grafana Cloud (or any OTLP-compatible endpoint) via HTTP/protobuf.
//!
//! Standard OTel env vars control the exporter:
//! - `OTEL_EXPORTER_OTLP_ENDPOINT` — OTLP endpoint (e.g., `https://otlp-gateway-*.grafana.net/otlp`)
//! - `OTEL_EXPORTER_OTLP_HEADERS` — auth headers (e.g., `Authorization=Basic <base64>`)
//! - `OTEL_SERVICE_NAME` — overrides the default `eidola-server`

use opentelemetry::KeyValue;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::Resource;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

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
            let tracer = guard.tracer_provider.tracer("eidola-server");
            let trace_layer = tracing_opentelemetry::layer().with_tracer(tracer);
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
    let trace_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .build()
        .expect("failed to create OTLP trace exporter");

    let tracer_provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_resource(resource.clone())
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
pub mod metrics {
    use opentelemetry::metrics::{Counter, Histogram};
    use std::sync::LazyLock;

    fn meter() -> opentelemetry::metrics::Meter {
        opentelemetry::global::meter("eidola-server")
    }

    /// HTTP request duration in seconds.
    pub static HTTP_REQUEST_DURATION: LazyLock<Histogram<f64>> = LazyLock::new(|| {
        meter()
            .f64_histogram("http.server.request.duration")
            .with_description("Duration of HTTP server requests")
            .with_unit("s")
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
    pub static CHAT_TOKENS: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("chat.completion.tokens")
            .with_description("Chat completion tokens processed")
            .build()
    });

    /// Total chat completion requests (by model, stream, status).
    pub static CHAT_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("chat.completion.requests")
            .with_description("Chat completion requests")
            .build()
    });

    /// Total TDX attestations observed during outbound enclave handshakes,
    /// labeled by the merged platform + QE TCB status. Includes
    /// attestations the verifier ultimately rejects, so the rate of
    /// non-`up_to_date` observations is the operator-visible signal that
    /// Intel has published a TCB advisory affecting the upstream fleet.
    pub static TDX_ATTESTATIONS: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("tinfoil.tdx.attestations")
            .with_description("TDX attestations observed by the tinfoil verifier")
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
