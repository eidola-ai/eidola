//! HTTP request observability middleware.
//!
//! Creates a tracing span per request and records HTTP metrics. Classifies
//! routes into confidentiality layers to enforce privacy boundaries:
//!
//! - **unlinked** (`/v1/chat/*`): no identifying information in spans/logs
//! - **linked** (`/v1/account/*`, `/v1/webhooks/*`): account_id permitted
//! - **public** (everything else): minimal logging

use std::time::Instant;

use axum::extract::{MatchedPath, Request};
use axum::middleware::Next;
use axum::response::Response;
use opentelemetry::KeyValue;
use tracing::Instrument;

use crate::telemetry::metrics;

/// Axum middleware that instruments every request with a tracing span and metrics.
pub async fn observe(matched_path: Option<MatchedPath>, request: Request, next: Next) -> Response {
    let method = request.method().to_string();
    let path = matched_path
        .as_ref()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| request.uri().path().to_string());
    let layer = classify_route(&path);

    let span = tracing::info_span!(
        "http.request",
        otel.kind = "server",
        http.request.method = %method,
        http.route = %path,
        http.response.status_code = tracing::field::Empty,
        eidola.layer = layer,
    );

    async move {
        let start = Instant::now();
        let response = next.run(request).await;
        let latency = start.elapsed();
        let status = response.status().as_u16();

        tracing::Span::current().record("http.response.status_code", status);

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

/// Classify a route into its confidentiality layer.
fn classify_route(path: &str) -> &'static str {
    if path.starts_with("/v1/chat/") {
        "unlinked"
    } else if path.starts_with("/v1/account") || path.starts_with("/v1/webhooks/") {
        "linked"
    } else {
        "public"
    }
}
