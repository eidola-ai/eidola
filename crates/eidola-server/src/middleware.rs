//! HTTP request observability middleware.
//!
//! Opens a tracing span for every request and records HTTP metrics.
//!
//! Spans are aggregated into metrics in process and not exported, unless the
//! request carries a `traceparent` asking to be sampled (see
//! [`crate::telemetry`]).

use std::time::Instant;

use axum::extract::{MatchedPath, Request};
use axum::http::HeaderMap;
use axum::middleware::Next;
use axum::response::Response;
use opentelemetry::KeyValue;
use opentelemetry::trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState};
use tracing::Instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::telemetry::metrics;

/// Axum middleware that opens a request span and records HTTP metrics.
pub async fn observe(matched_path: Option<MatchedPath>, request: Request, next: Next) -> Response {
    let method = request.method().to_string();
    let path = matched_path
        .as_ref()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| request.uri().path().to_string());

    let span = tracing::info_span!(
        "http.request",
        otel.kind = "server",
        http.request.method = %method,
        http.route = %path,
        http.response.status_code = tracing::field::Empty,
    );

    // Must happen before the handler runs: the first child span started under
    // this one causes it to be built, fixing its sampled flag.
    if let Some(parent) = requested_trace_parent(request.headers()) {
        // Fails only when no OpenTelemetry layer is installed — the ordinary
        // state when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset. Nothing is
        // exported in that case anyway, so there is nothing to report.
        let _ = span.set_parent(opentelemetry::Context::new().with_remote_span_context(parent));
    }

    async move {
        let start = Instant::now();
        let response = next.run(request).await;
        let latency = start.elapsed();
        let status = response.status().as_u16();

        tracing::Span::current().record("http.response.status_code", status);

        // `http.route` is the matched path template, never the raw URI, so it
        // cannot carry request-derived values into a label.
        let attrs = [
            KeyValue::new("http.request.method", method),
            KeyValue::new("http.route", path),
            KeyValue::new("http.response.status_code", status as i64),
        ];
        metrics::HTTP_REQUEST_DURATION.record(latency.as_secs_f64(), &attrs);
        metrics::HTTP_REQUEST_COUNT.add(1, &attrs);

        response
    }
    .instrument(span)
    .await
}

/// Build a sampled remote parent from the request's headers, or `None`.
///
/// Returns `Some` only for a well-formed W3C `traceparent` whose sampled flag
/// is set. Nothing else marks a request for export.
///
/// The sampled flag is honored rather than assumed. Per W3C Trace Context it
/// reports whether the caller recorded the request, and generic client-side
/// OpenTelemetry instrumentation injects `traceparent` on every outbound call
/// with the flag reflecting its own sampler. Honoring it means such a client
/// only causes an export when it too is recording.
///
/// The trace id is the caller's. A caller reusing one across requests asserts
/// that those requests are related, which on the anonymous surface is a
/// self-linking primitive — so anything generating this header must mint a
/// fresh random id per attempt, including on retries, where resending the
/// original headers is the obvious implementation and the wrong one.
fn requested_trace_parent(headers: &HeaderMap) -> Option<SpanContext> {
    let value = headers.get("traceparent")?.to_str().ok()?;

    let Some(parent) = parse_traceparent(value) else {
        metrics::TRACEPARENT_RECEIVED.add(1, &[KeyValue::new("outcome", "malformed")]);
        return None;
    };

    if !parent.is_sampled() {
        metrics::TRACEPARENT_RECEIVED.add(1, &[KeyValue::new("outcome", "not_sampled")]);
        return None;
    }

    metrics::TRACEPARENT_RECEIVED.add(1, &[KeyValue::new("outcome", "sampled")]);
    Some(parent)
}

/// Parse a W3C `traceparent` into a remote span context.
///
/// Format: `<version>-<32 hex trace id>-<16 hex span id>-<2 hex flags>`. Only
/// version `00` is accepted, and the sampled bit of the flags is carried
/// through as-is.
///
/// Parsed by hand rather than by installing a `TextMapPropagator`: a global
/// propagator would also extract and propagate context on the server's own
/// outbound calls, which would put the trace id on requests to the upstream,
/// Stripe, and Postgres.
fn parse_traceparent(value: &str) -> Option<SpanContext> {
    let mut parts = value.split('-');
    let (version, trace_id, span_id, flags) =
        (parts.next()?, parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() || version != "00" {
        return None;
    }

    // Exact W3C field shapes, checked before parsing: `from_hex` /
    // `from_str_radix` alone accept variable-length (and uppercase, and
    // `+`-prefixed) values, so without this a header like `00-1-1-01`
    // would opt its request into export against this function's
    // well-formed-only contract.
    let exact_lower_hex = |s: &str, len: usize| {
        s.len() == len && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    };
    if !exact_lower_hex(trace_id, 32) || !exact_lower_hex(span_id, 16) || !exact_lower_hex(flags, 2)
    {
        return None;
    }

    let trace_id = TraceId::from_hex(trace_id).ok()?;
    let span_id = SpanId::from_hex(span_id).ok()?;
    if trace_id == TraceId::INVALID || span_id == SpanId::INVALID {
        return None;
    }

    let flags = u8::from_str_radix(flags, 16).ok()?;

    Some(SpanContext::new(
        trace_id,
        span_id,
        TraceFlags::new(flags) & TraceFlags::SAMPLED,
        true,
        TraceState::default(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_valid_traceparent() {
        let cx = parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
            .expect("valid traceparent");
        assert!(cx.is_sampled());
        assert!(cx.is_remote());
    }

    /// A clear sampled bit is carried through, so the request is not exported.
    #[test]
    fn unsampled_traceparent_is_not_sampled() {
        let cx = parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00")
            .expect("valid traceparent");
        assert!(!cx.is_sampled());
        assert!(requested_trace_parent(&sampled_headers("00")).is_none());
    }

    fn sampled_headers(flags: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            "traceparent",
            format!("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-{flags}")
                .parse()
                .unwrap(),
        );
        headers
    }

    #[test]
    fn a_sampled_traceparent_yields_a_parent() {
        let parent = requested_trace_parent(&sampled_headers("01")).expect("sampled");
        assert!(parent.is_sampled());
    }

    #[test]
    fn rejects_malformed_traceparents() {
        for bad in [
            "",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
            "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
            "00-nothex-00f067aa0ba902b7-01",
            // Wrong field lengths: `u128::from_str_radix` would happily
            // parse these, so the shape check is what rejects them.
            "00-1-1-01",
            "00-4bf92f3577b34da6a3ce929d0e0e47361-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-1",
            // `from_str_radix` accepts a leading `+`; W3C hex has none.
            "00-+bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            // W3C requires lowercase hex.
            "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-0F",
        ] {
            assert!(
                parse_traceparent(bad).is_none(),
                "should have rejected {bad:?}"
            );
        }
    }

    #[test]
    fn no_headers_yields_no_parent() {
        assert!(requested_trace_parent(&HeaderMap::new()).is_none());
    }
}
