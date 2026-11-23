//! OpenTelemetry for the receiver and worker: metrics and logs over OTLP,
//! enabled when `OTEL_EXPORTER_OTLP_ENDPOINT` is set and not empty, read
//! under the process's prefix like every other variable
//! (`eidola_telemetry::OtlpConfig` lists the rest). Support commands start
//! none of this and export nothing.
//!
//! No traces are exported. Nothing asks for one — Stripe sends no trace
//! context, and its dashboard already holds the authoritative request log —
//! and a span per job would carry account work's timing for no signal the
//! counters below do not already give.
//!
//! Labels come from fixed sets only (`eidola_billing::metrics` holds the
//! shared billing instruments and their rules); logs are for operator
//! misconfiguration and incidents, once per condition or change of state,
//! never a line per delivery or job.

use eidola_common::env::Env;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Which long-running role this process is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Receiver,
    Worker,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Self::Receiver => "receiver",
            Self::Worker => "worker",
        }
    }

    /// The service name telemetry carries unless `OTEL_SERVICE_NAME` says
    /// otherwise.
    pub fn service_name(self) -> &'static str {
        match self {
            Self::Receiver => "eidola-billing-receiver",
            Self::Worker => "eidola-billing-worker",
        }
    }
}

/// Holds the OTel providers so they can be flushed and shut down.
pub struct OtelGuard {
    meter_provider: opentelemetry_sdk::metrics::SdkMeterProvider,
    logger_provider: opentelemetry_sdk::logs::SdkLoggerProvider,
}

impl OtelGuard {
    /// Flush and shut down both providers.
    pub fn shutdown(self) {
        if let Err(e) = self.meter_provider.shutdown() {
            eprintln!("otel: meter provider shutdown error: {e}");
        }
        if let Err(e) = self.logger_provider.shutdown() {
            eprintln!("otel: logger provider shutdown error: {e}");
        }
    }
}

/// Install the log subscriber (stderr, plus the OTLP log bridge when
/// enabled) and, when enabled, the global meter provider. Returns the guard
/// when OTLP export is enabled.
pub fn init(env: &Env, role: Role) -> Option<OtelGuard> {
    let otel = eidola_telemetry::OtlpConfig::read(|name| env.var(name), role.service_name()).map(
        |config| {
            let timeouts = config.timeouts();
            let [metrics_client, logs_client] = eidola_telemetry::otlp_http_clients(
                eidola_billing::tls_config(),
                [timeouts.metrics, timeouts.logs],
            );
            // Replicas of one role share a service name; the instance keeps
            // their cumulative series apart. It names this process only.
            let resource = config.resource([opentelemetry::KeyValue::new(
                "service.instance.id",
                uuid::Uuid::new_v4().to_string(),
            )]);
            let meter_provider = config.meter_provider(resource.clone(), metrics_client);
            opentelemetry::global::set_meter_provider(meter_provider.clone());
            let logger_provider = config.logger_provider(resource, logs_client);
            OtelGuard {
                meter_provider,
                logger_provider,
            }
        },
    );

    let log_layer = otel.as_ref().map(|guard| {
        opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new(
            &guard.logger_provider,
        )
    });
    tracing_subscriber::registry()
        .with(log_filter(&env.var("RUST_LOG").unwrap_or_default()))
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .with(log_layer)
        .init();

    otel
}

/// The prefixed `RUST_LOG`, defaulting to `error`, with billing's own crates
/// at `info` and `hyper` at `warn`.
fn log_filter(rust_log: &str) -> EnvFilter {
    eidola_telemetry::log_filter(
        rust_log,
        &[
            "eidola_billing=info",
            "eidola_billing_cli=info",
            "hyper=warn",
        ],
    )
}

/// The receiver's instruments. The worker's live in `eidola_billing::metrics`,
/// beside the workflow they count.
pub mod metrics {
    use std::sync::LazyLock;
    use std::time::Duration;

    use opentelemetry::KeyValue;
    use opentelemetry::metrics::{Counter, Histogram};

    fn meter() -> opentelemetry::metrics::Meter {
        opentelemetry::global::meter("eidola-billing-cli")
    }

    /// Second-scale buckets: a delivery is one signature check and one
    /// database write.
    const LATENCY_BUCKETS: &[f64] = &[
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
    ];

    /// Every `outcome` a webhook delivery is counted under.
    pub const WEBHOOK_OUTCOMES: &[&str] = &[
        "stored",
        "duplicate",
        "unsupported_ack",
        "missing_signature",
        "malformed_signature",
        "bad_signature",
        "stale_timestamp",
        "snapshot_refused",
        "malformed",
        "too_large",
        "unreadable",
        "conflict",
        "protocol_refused",
        "db_failed",
    ];

    /// Stripe webhook deliveries, by `event_type` (a supported Stripe event
    /// type, `unsupported` for an authenticated thin event of another type,
    /// or `unknown` for a delivery refused before its type was read) and
    /// `outcome` ([`WEBHOOK_OUTCOMES`]).
    static WEBHOOK_DELIVERIES: LazyLock<Counter<u64>> = LazyLock::new(|| {
        meter()
            .u64_counter("billing.webhook.deliveries")
            .with_description("Stripe webhook deliveries, by event type and outcome")
            .build()
    });

    pub fn webhook_delivery(event_type: &'static str, outcome: &'static str) {
        debug_assert!(WEBHOOK_OUTCOMES.contains(&outcome), "{outcome}");
        WEBHOOK_DELIVERIES.add(
            1,
            &[
                KeyValue::new("event_type", event_type),
                KeyValue::new("outcome", outcome),
            ],
        );
    }

    /// HTTP request duration, by method, route (the matched route, fixed in
    /// code) and status, named as the gateway names its own.
    static HTTP_REQUEST_DURATION: LazyLock<Histogram<f64>> = LazyLock::new(|| {
        meter()
            .f64_histogram("http.server.request.duration")
            .with_description("Duration of HTTP server requests")
            .with_unit("s")
            .with_boundaries(LATENCY_BUCKETS.to_vec())
            .build()
    });

    pub fn http_request(method: &'static str, route: &'static str, status: u16, elapsed: Duration) {
        HTTP_REQUEST_DURATION.record(
            elapsed.as_secs_f64(),
            &[
                KeyValue::new("http.request.method", method),
                KeyValue::new("http.route", route),
                KeyValue::new("http.response.status_code", i64::from(status)),
            ],
        );
    }
}
