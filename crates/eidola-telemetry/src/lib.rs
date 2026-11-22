//! The OTLP export pipeline Eidola's servers share: the gateway, and the
//! billing receiver and worker.
//!
//! Telemetry is enabled when `OTEL_EXPORTER_OTLP_ENDPOINT` is set and not
//! empty, and ships over HTTP/protobuf. These variables, read by the caller
//! under its environment's prefix (`eidola_common::env`) and handed to
//! [`OtlpConfig::read`], configure it:
//! - `OTEL_EXPORTER_OTLP_ENDPOINT` — OTLP endpoint (e.g., `https://otlp-gateway-*.grafana.net/otlp`)
//! - `OTEL_EXPORTER_OTLP_HEADERS` — auth headers (e.g., `Authorization=Basic <base64>`)
//! - `OTEL_SERVICE_NAME` — overrides the caller's default service name
//! - `OTEL_EXPORTER_OTLP_TIMEOUT` and `OTEL_EXPORTER_OTLP_<SIGNAL>_TIMEOUT` —
//!   export timeouts, in milliseconds
//!
//! Everything else the SDK would read from the environment by itself is set
//! explicitly ([`OtlpConfig`]). What a server records — its instruments, its
//! sampler, which signals it exports — stays with that server; this crate
//! only builds the exporters and providers.

use std::collections::HashMap;
use std::time::Duration;

use opentelemetry::KeyValue;
use opentelemetry_otlp::{Protocol, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::Resource;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

/// How often metrics are exported: each export covers this interval, so it is the
/// finest time resolution any exported metric has. An engine node publishes its
/// snapshot at the same epoch (`eidola_common::engine_protocol::telemetry::SNAPSHOT_EPOCH`).
pub const METRIC_EXPORT_INTERVAL: std::time::Duration =
    opentelemetry_sdk::metrics::OTEL_METRIC_EXPORT_INTERVAL_DEFAULT;

/// The event and span filter: `EnvFilter::from_default_env`'s grammar over the
/// prefixed `RUST_LOG`, defaulting to `error`, with `directives` (such as
/// `eidola_server_gateway=info` or `hyper=warn`) added over it.
pub fn log_filter(rust_log: &str, directives: &[&str]) -> EnvFilter {
    directives.iter().fold(
        EnvFilter::builder()
            .with_default_directive(LevelFilter::ERROR.into())
            .parse_lossy(rust_log),
        |filter, directive| filter.add_directive(directive.parse().expect("a valid log directive")),
    )
}

/// Everything the OTLP exporters, the SDK's processors and the resource are
/// built from, read by us under the environment's prefix and handed over
/// explicitly.
///
/// Left to themselves, the exporter and SDK builders read the bare `OTEL_*`
/// names: the endpoint, protocol, timeout and temporality, the batch
/// processors' sizes and delays, the metric export interval, span limits, the
/// sampler, and the resource's `OTEL_SERVICE_NAME` and
/// `OTEL_RESOURCE_ATTRIBUTES`. Each is set explicitly by the builders here
/// (the sampler and span limits by the caller that builds a tracer provider),
/// which overrides the variable, so no unprefixed value configures any of
/// them. Compression is not settable to "none", and needs no setting: the
/// exporter is built without its gzip and zstd features, so it ignores a
/// compression variable, with a warning.
///
/// One read cannot be closed from outside: the HTTP exporter merges the
/// headers in the bare `OTEL_EXPORTER_OTLP_HEADERS` (or
/// `OTEL_EXPORTER_OTLP_<SIGNAL>_HEADERS`) over the headers it is given, with
/// no setting to stop it. A container receives only the variables its
/// deployment config names, so a bare name reaches it only if the config
/// declares one.
pub struct OtlpConfig {
    /// The base endpoint; each signal's path is appended to it.
    endpoint: String,
    /// `OTEL_EXPORTER_OTLP_HEADERS` as `key=value` pairs, values still
    /// percent-encoded: `with_headers` decodes them, as the exporter decodes
    /// the variable.
    headers: HashMap<String, String>,
    service_name: String,
    timeouts: Timeouts,
}

/// Each signal's export timeout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Timeouts {
    pub traces: Duration,
    pub metrics: Duration,
    pub logs: Duration,
}

impl OtlpConfig {
    /// The configuration `lookup` describes, or `None` when it names no
    /// endpoint. `lookup` answers for one variable name, already prefixed;
    /// `default_service_name` stands when `OTEL_SERVICE_NAME` is unset or
    /// empty.
    pub fn read(
        lookup: impl Fn(&str) -> Option<String>,
        default_service_name: &str,
    ) -> Option<Self> {
        let endpoint =
            lookup(opentelemetry_otlp::OTEL_EXPORTER_OTLP_ENDPOINT).filter(|s| !s.is_empty())?;
        let headers = lookup(opentelemetry_otlp::OTEL_EXPORTER_OTLP_HEADERS)
            .map(|raw| otlp_headers(&raw))
            .unwrap_or_default();
        let service_name = lookup("OTEL_SERVICE_NAME")
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| default_service_name.to_string());
        let timeouts = Timeouts {
            traces: otlp_timeout(
                opentelemetry_otlp::OTEL_EXPORTER_OTLP_TRACES_TIMEOUT,
                &lookup,
            ),
            metrics: otlp_timeout(
                opentelemetry_otlp::OTEL_EXPORTER_OTLP_METRICS_TIMEOUT,
                &lookup,
            ),
            logs: otlp_timeout(opentelemetry_otlp::OTEL_EXPORTER_OTLP_LOGS_TIMEOUT, &lookup),
        };
        Some(Self {
            endpoint,
            headers,
            service_name,
            timeouts,
        })
    }

    /// Each signal's export timeout, which the HTTP client sending that
    /// signal must carry ([`otlp_http_clients`]).
    pub fn timeouts(&self) -> Timeouts {
        self.timeouts
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

    /// The resource every provider carries: the service name, the SDK's own
    /// attributes, and `attributes` (such as `service.instance.id`, for a
    /// service that runs as several processes).
    ///
    /// Built empty, not from `Resource::builder()`, whose detectors read
    /// `OTEL_SERVICE_NAME` and `OTEL_RESOURCE_ATTRIBUTES`; the SDK's own
    /// attributes (`telemetry.sdk.*`) read nothing.
    pub fn resource(&self, attributes: impl IntoIterator<Item = KeyValue>) -> Resource {
        Resource::builder_empty()
            .with_detector(Box::new(
                opentelemetry_sdk::resource::TelemetryResourceDetector,
            ))
            .with_service_name(self.service_name.clone())
            .with_attributes(attributes)
            .build()
    }

    /// A batch processor exporting sampled spans over OTLP. It filters on
    /// `is_sampled` before batching, so which spans leave is the caller's
    /// sampler's decision.
    pub fn span_batch_processor(
        &self,
        client: reqwest::blocking::Client,
    ) -> opentelemetry_sdk::trace::BatchSpanProcessor {
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpBinary)
            .with_endpoint(self.signal_endpoint("/v1/traces"))
            .with_timeout(self.timeouts.traces)
            .with_headers(self.headers.clone())
            .with_http_client(client)
            .build()
            .expect("failed to create OTLP trace exporter");

        // The batch configuration's defaults read `OTEL_BSP_*`; each field that
        // takes effect is set back to the SDK's default here.
        opentelemetry_sdk::trace::BatchSpanProcessor::builder(exporter)
            .with_batch_config(
                opentelemetry_sdk::trace::BatchConfigBuilder::default()
                    .with_max_queue_size(opentelemetry_sdk::trace::OTEL_BSP_MAX_QUEUE_SIZE_DEFAULT)
                    .with_scheduled_delay(opentelemetry_sdk::trace::OTEL_BSP_SCHEDULE_DELAY_DEFAULT)
                    .with_max_export_batch_size(
                        opentelemetry_sdk::trace::OTEL_BSP_MAX_EXPORT_BATCH_SIZE_DEFAULT,
                    )
                    .build(),
            )
            .build()
    }

    /// A meter provider exporting cumulative metrics over OTLP at the SDK's
    /// default interval. The caller installs it as the global provider.
    pub fn meter_provider(
        &self,
        resource: Resource,
        client: reqwest::blocking::Client,
    ) -> opentelemetry_sdk::metrics::SdkMeterProvider {
        let exporter = opentelemetry_otlp::MetricExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpBinary)
            .with_endpoint(self.signal_endpoint("/v1/metrics"))
            .with_timeout(self.timeouts.metrics)
            .with_headers(self.headers.clone())
            .with_http_client(client)
            .with_temporality(opentelemetry_sdk::metrics::Temporality::Cumulative)
            .build()
            .expect("failed to create OTLP metrics exporter");

        let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(exporter)
            .with_interval(METRIC_EXPORT_INTERVAL)
            .build();

        opentelemetry_sdk::metrics::SdkMeterProvider::builder()
            .with_resource(resource)
            .with_reader(reader)
            .build()
    }

    /// A logger provider exporting log records over OTLP in batches.
    pub fn logger_provider(
        &self,
        resource: Resource,
        client: reqwest::blocking::Client,
    ) -> opentelemetry_sdk::logs::SdkLoggerProvider {
        let exporter = opentelemetry_otlp::LogExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpBinary)
            .with_endpoint(self.signal_endpoint("/v1/logs"))
            .with_timeout(self.timeouts.logs)
            .with_headers(self.headers.clone())
            .with_http_client(client)
            .build()
            .expect("failed to create OTLP log exporter");

        // As for spans: the defaults read `OTEL_BLRP_*`, so each field is set.
        let batch = opentelemetry_sdk::logs::BatchLogProcessor::builder(exporter)
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

        opentelemetry_sdk::logs::SdkLoggerProvider::builder()
            .with_resource(resource)
            .with_log_processor(batch)
            .build()
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

/// The HTTP clients the OTLP exporters send through, one per timeout given:
/// one per exported signal, each carrying that signal's timeout.
///
/// Left to itself the exporter builds a default `reqwest` 0.13 client, which
/// verifies against the platform trust store — and the servers run `FROM
/// scratch` with none, so that client cannot be built and an HTTPS collector
/// (Grafana Cloud in production) is unreachable. These clients verify with
/// `tls`, the caller's configuration carrying the bundled Mozilla roots its
/// other outbound clients use. A client handed to the exporter does not get
/// the exporter's timeout, so each one carries its signal's
/// ([`OtlpConfig::timeouts`]), and none reads a proxy from the environment.
/// Built on their own thread, as the exporter builds its default, because a
/// blocking client must not be constructed inside the tokio runtime a server
/// runs in.
pub fn otlp_http_clients<const N: usize>(
    tls: rustls::ClientConfig,
    timeouts: [Duration; N],
) -> [reqwest::blocking::Client; N] {
    std::thread::spawn(move || {
        timeouts.map(|timeout| {
            reqwest::blocking::Client::builder()
                .tls_backend_preconfigured(tls.clone())
                .no_proxy()
                .timeout(timeout)
                .build()
                .expect("failed to build OTLP HTTP client")
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The supplied OTLP clients carry the timeout the exporter would have
    /// resolved itself: signal variable over the general one over 10 s, in
    /// milliseconds, skipping a value that does not parse.
    #[test]
    fn otlp_timeout_resolves_like_the_exporter() {
        use opentelemetry_otlp::{
            OTEL_EXPORTER_OTLP_TIMEOUT, OTEL_EXPORTER_OTLP_TIMEOUT_DEFAULT,
            OTEL_EXPORTER_OTLP_TRACES_TIMEOUT,
        };
        let signal = OTEL_EXPORTER_OTLP_TRACES_TIMEOUT;
        assert_eq!(
            otlp_timeout(signal, vars(&[])),
            OTEL_EXPORTER_OTLP_TIMEOUT_DEFAULT
        );
        assert_eq!(
            otlp_timeout(signal, vars(&[(OTEL_EXPORTER_OTLP_TIMEOUT, "2500")])),
            Duration::from_millis(2500)
        );
        assert_eq!(
            otlp_timeout(
                signal,
                vars(&[
                    (OTEL_EXPORTER_OTLP_TRACES_TIMEOUT, "3000"),
                    (OTEL_EXPORTER_OTLP_TIMEOUT, "2500")
                ])
            ),
            Duration::from_millis(3000)
        );
        assert_eq!(
            otlp_timeout(
                signal,
                vars(&[
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
        assert!(OtlpConfig::read(vars(&[]), "service").is_none());
        assert!(
            OtlpConfig::read(vars(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "")]), "service").is_none()
        );
    }

    /// The values we read are the ones the exporters get: each signal's path
    /// appended to the endpoint as the exporter appends it, the headers split
    /// as it splits them, and the service name defaulted.
    #[test]
    fn otlp_config_is_read_from_the_lookup_alone() {
        let config = OtlpConfig::read(
            vars(&[
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
            ]),
            "eidola-server-gateway",
        )
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
            config.timeouts(),
            Timeouts {
                traces: Duration::from_millis(2500),
                metrics: Duration::from_millis(2500),
                logs: Duration::from_millis(4000),
            }
        );

        let config = OtlpConfig::read(
            vars(&[
                ("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.example/"),
                ("OTEL_SERVICE_NAME", "staging-gateway"),
            ]),
            "eidola-server-gateway",
        )
        .unwrap();
        assert_eq!(
            config.signal_endpoint("/v1/logs"),
            "https://collector.example/v1/logs"
        );
        assert_eq!(config.service_name, "staging-gateway");
        assert!(config.headers.is_empty());
    }
}
