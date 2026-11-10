//! Stripe webhooks durably enqueue observations. Billing effects belong to
//! eidola-billing, shared with account repair and the private support CLI.

use crate::{AppState, error::ServerError};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use bytes::Bytes;
use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use subtle::ConstantTimeEq;

type HmacSha256 = Hmac<Sha256>;

/// Maximum age for a webhook signature (5 minutes).
const MAX_SIGNATURE_AGE: Duration = Duration::from_secs(300);

// ---------------------------------------------------------------------------
// Signature verification
// ---------------------------------------------------------------------------

/// Verify a Stripe webhook signature against the raw body.
///
/// Public wrapper that uses the current time.
pub fn verify_signature(header: &str, body: &[u8], secret: &str) -> Result<(), &'static str> {
    let now = SystemTime::now();
    verify_signature_at(header, body, secret, now)
}

/// Verify a Stripe webhook signature at a specific time (for deterministic testing).
fn verify_signature_at(
    header: &str,
    body: &[u8],
    secret: &str,
    now: SystemTime,
) -> Result<(), &'static str> {
    let mut timestamp: Option<&str> = None;
    let mut signature: Option<&str> = None;

    for part in header.split(',') {
        if let Some(val) = part.strip_prefix("t=") {
            timestamp = Some(val);
        } else if let Some(val) = part.strip_prefix("v1=") {
            signature = Some(val);
        }
    }

    let timestamp = timestamp.ok_or("missing timestamp")?;
    let signature = signature.ok_or("missing v1 signature")?;

    let ts_secs: u64 = timestamp.parse().map_err(|_| "invalid timestamp")?;
    let event_time = UNIX_EPOCH + Duration::from_secs(ts_secs);
    if now.duration_since(event_time).unwrap_or(Duration::MAX) > MAX_SIGNATURE_AGE {
        return Err("stale timestamp");
    }

    let mut mac =
        HmacSha256::new_from_slice(secret.as_bytes()).map_err(|_| "invalid webhook secret")?;
    mac.update(timestamp.as_bytes());
    mac.update(b".");
    mac.update(body);
    let expected = mac.finalize().into_bytes();

    let provided = hex::decode(signature).map_err(|_| "invalid hex signature")?;

    if expected.as_slice().ct_eq(&provided).into() {
        Ok(())
    } else {
        Err("signature mismatch")
    }
}

#[derive(serde::Deserialize)]
struct StripeEvent {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    data: EventData,
}

#[derive(serde::Deserialize)]
struct EventData {
    object: EventObject,
}

#[derive(serde::Deserialize)]
struct EventObject {
    id: String,
}

use eidola_billing::service::BILLING_EVENT_TYPES;

fn event_type_label(kind: &str) -> &'static str {
    BILLING_EVENT_TYPES
        .iter()
        .copied()
        .find(|item| *item == kind)
        .unwrap_or("other")
}

fn record_webhook_outcome(kind: &str, reason: &'static str) {
    crate::telemetry::metrics::WEBHOOK_OUTCOME.add(
        1,
        &[
            opentelemetry::KeyValue::new("event_type", event_type_label(kind)),
            opentelemetry::KeyValue::new("reason", reason),
        ],
    );
}

/// An acknowledgment means the event is durable, not that Stripe has already
/// been fetched. A gateway crash after this response cannot lose the work.
#[utoipa::path(
    post,
    path = "/v1/webhooks/stripe",
    tag = "Linked",
    responses(
        (status = 200, description = "Event durably received or type ignored"),
        (status = 400, description = "Invalid signature or event", body = crate::types::ErrorResponse),
        (status = 500, description = "Event was not persisted; retry delivery", body = crate::types::ErrorResponse),
        (status = 503, description = "Webhook secret not configured", body = crate::types::ErrorResponse)
    )
)]
#[tracing::instrument(skip_all, name = "webhook.process")]
pub async fn stripe_webhook(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    let Some(secret) = state.stripe_webhook_secret.as_deref() else {
        record_webhook_outcome("", "not_configured");
        return ServerError::ServiceUnavailable("webhook secret is not configured".into())
            .into_response();
    };
    let Some(signature) = headers
        .get("stripe-signature")
        .and_then(|v| v.to_str().ok())
    else {
        record_webhook_outcome("", "missing_signature");
        return bad_request("missing stripe-signature header");
    };
    if verify_signature(signature, &body, secret).is_err() {
        record_webhook_outcome("", "bad_signature");
        return bad_request("invalid webhook signature");
    }
    let Ok(event) = serde_json::from_slice::<StripeEvent>(&body) else {
        record_webhook_outcome("", "malformed_event");
        return bad_request("invalid webhook event");
    };
    if event_type_label(&event.kind) == "other" {
        record_webhook_outcome("", "ignored");
        return StatusCode::OK.into_response();
    }
    if !valid_object_id(&event.id) || !valid_object_id(&event.data.object.id) {
        record_webhook_outcome(&event.kind, "malformed_event");
        return bad_request("invalid webhook object identifier");
    }
    let store = eidola_billing::store::Store::new(state.db_pool.clone());
    match store
        .enqueue_event(&event.id, &event.kind, &event.data.object.id)
        .await
    {
        Ok(_) => {
            record_webhook_outcome(&event.kind, "queued");
            state.billing_wakeup.notify_one();
            StatusCode::OK.into_response()
        }
        Err(_) => {
            record_webhook_outcome(&event.kind, "database_error");
            ServerError::Internal("could not persist billing event".into()).into_response()
        }
    }
}

fn valid_object_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 255 && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

fn bad_request(message: &str) -> axum::response::Response {
    ServerError::BadRequest {
        message: message.into(),
    }
    .into_response()
}

/// Retry durable inbox work, including work accepted by a previous gateway.
/// The timer reads only our queue; it never periodically polls Stripe accounts.
pub fn spawn_billing_worker(state: AppState) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let Some(stripe) = state.stripe.clone() else {
            return;
        };
        let service = eidola_billing::service::BillingService::new(state.db_pool.clone(), stripe);
        loop {
            match service.drain_events(32).await {
                Ok(report) => {
                    if report.failed > 0 {
                        tracing::warn!("billing reconciliation remains pending");
                    }
                    if report.review > 0 {
                        tracing::warn!("billing reconciliation requires support review");
                    }
                    if report.processed == 32 && report.failed == 0 {
                        tokio::task::yield_now().await;
                        continue;
                    }
                }
                Err(_) => tracing::warn!("billing inbox unavailable; will retry"),
            }
            tokio::select! {
                _ = state.billing_wakeup.notified() => {},
                _ = tokio::time::sleep(Duration::from_secs(5)) => {},
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const TEST_SECRET: &str = "whsec_test_secret";
    const TEST_BODY: &[u8] = b"{\"id\":\"evt_test\"}";

    fn make_signature(timestamp: u64, body: &[u8], secret: &str) -> String {
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(timestamp.to_string().as_bytes());
        mac.update(b".");
        mac.update(body);
        let sig = hex::encode(mac.finalize().into_bytes());
        format!("t={},v1={}", timestamp, sig)
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    #[test]
    fn valid_signature() {
        let ts = now_secs();
        let header = make_signature(ts, TEST_BODY, TEST_SECRET);
        let now = UNIX_EPOCH + Duration::from_secs(ts);
        assert!(verify_signature_at(&header, TEST_BODY, TEST_SECRET, now).is_ok());
    }

    #[test]
    fn missing_timestamp() {
        let header = "v1=deadbeef";
        let now = SystemTime::now();
        assert_eq!(
            verify_signature_at(header, TEST_BODY, TEST_SECRET, now),
            Err("missing timestamp")
        );
    }

    #[test]
    fn missing_v1() {
        let ts = now_secs();
        let header = format!("t={}", ts);
        let now = UNIX_EPOCH + Duration::from_secs(ts);
        assert_eq!(
            verify_signature_at(&header, TEST_BODY, TEST_SECRET, now),
            Err("missing v1 signature")
        );
    }

    #[test]
    fn wrong_signature() {
        let ts = now_secs();
        let header = format!("t={},v1={}", ts, "00".repeat(32));
        let now = UNIX_EPOCH + Duration::from_secs(ts);
        assert_eq!(
            verify_signature_at(&header, TEST_BODY, TEST_SECRET, now),
            Err("signature mismatch")
        );
    }

    #[test]
    fn stale_timestamp() {
        let ts = now_secs() - 600;
        let header = make_signature(ts, TEST_BODY, TEST_SECRET);
        let now = UNIX_EPOCH + Duration::from_secs(ts + 600);
        assert_eq!(
            verify_signature_at(&header, TEST_BODY, TEST_SECRET, now),
            Err("stale timestamp")
        );
    }

    #[test]
    fn every_financial_event_is_an_observation() {
        for kind in [
            "refund.created",
            "refund.updated",
            "refund.failed",
            "checkout.session.async_payment_succeeded",
            "charge.dispute.updated",
            "charge.dispute.funds_reinstated",
        ] {
            assert_ne!(event_type_label(kind), "other");
        }
        assert_eq!(event_type_label("unexpected_customer_input"), "other");
    }

    #[test]
    fn an_object_id_cannot_select_an_arbitrary_provider_path() {
        for id in [
            "",
            "../charges/ch_other",
            "re_test?customer=cus_other",
            "re_test/other",
        ] {
            assert!(!valid_object_id(id));
        }
        assert!(valid_object_id("re_test_123"));
    }
}
