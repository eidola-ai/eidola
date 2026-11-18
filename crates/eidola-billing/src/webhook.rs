//! Stripe ingress authenticates raw bytes and retains only event identities.
//! It accepts Stripe's thin event notifications, which name an event and its
//! related object without carrying that object's contents. A notification is
//! a wakeup, never a ledger mutation instruction.

use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use subtle::ConstantTimeEq;

type HmacSha256 = Hmac<Sha256>;

/// Maximum signature age or clock skew in either direction (5 minutes).
const MAX_SIGNATURE_SKEW: Duration = Duration::from_secs(300);

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

    for part in header.split(',').map(str::trim) {
        if let Some(val) = part.strip_prefix("t=") {
            timestamp = Some(val);
        }
    }

    let timestamp = timestamp.ok_or("missing timestamp")?;
    let mut signatures = header
        .split(',')
        .filter_map(|part| part.trim().strip_prefix("v1="))
        .peekable();
    if signatures.peek().is_none() {
        return Err("missing v1 signature");
    }

    let ts_secs: u64 = timestamp.parse().map_err(|_| "invalid timestamp")?;
    let event_time = UNIX_EPOCH
        .checked_add(Duration::from_secs(ts_secs))
        .ok_or("invalid timestamp")?;
    let skew = now
        .duration_since(event_time)
        .unwrap_or_else(|difference| difference.duration());
    if skew > MAX_SIGNATURE_SKEW {
        return Err("stale timestamp");
    }

    let mut mac =
        HmacSha256::new_from_slice(secret.as_bytes()).map_err(|_| "invalid webhook secret")?;
    mac.update(timestamp.as_bytes());
    mac.update(b".");
    mac.update(body);
    let expected = mac.finalize().into_bytes();

    // Stripe sends one v1 signature per active secret during rotation. Check
    // every well-formed candidate without making its position significant.
    let mut matches = subtle::Choice::from(0);
    for signature in signatures {
        let mut provided = [0_u8; 32];
        if hex::decode_to_slice(signature, &mut provided).is_ok() {
            matches |= expected.as_slice().ct_eq(&provided);
        }
    }

    if matches.into() {
        Ok(())
    } else {
        Err("signature mismatch")
    }
}

/// The `object` member of a thin event notification. A snapshot event names
/// `event` here instead, and carries the related object's contents.
const THIN_EVENT_OBJECT: &str = "v2.core.event";

#[derive(serde::Deserialize)]
struct Envelope {
    object: Option<String>,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(serde::Deserialize)]
struct ThinEvent {
    id: String,
    related_object: Option<RelatedObject>,
}

#[derive(serde::Deserialize)]
struct RelatedObject {
    id: String,
}

use crate::service::BILLING_EVENT_TYPES;

fn event_type_label(kind: &str) -> &'static str {
    BILLING_EVENT_TYPES
        .iter()
        .copied()
        .find(|item| *item == kind)
        .unwrap_or("other")
}

/// The only fields persisted from a verified Stripe event: its ID, its type
/// exactly as Stripe names it, and its related object's ID.
#[derive(Debug, PartialEq, Eq)]
pub struct WebhookEvent {
    pub id: String,
    pub kind: String,
    pub object_id: String,
}

/// Parse a previously authenticated thin event notification. A snapshot
/// event is refused; unsupported thin event types are ignored. The caller
/// must verify the signature against these exact bytes first.
pub fn parse_event(body: &[u8]) -> Result<Option<WebhookEvent>, &'static str> {
    let envelope: Envelope = serde_json::from_slice(body).map_err(|_| "invalid webhook event")?;
    if envelope.object.as_deref() != Some(THIN_EVENT_OBJECT) {
        return Err("only thin events are accepted");
    }
    // Unsupported types need not name a related object: some thin event
    // types, such as v1.billing.meter.no_meter_found, have none.
    if event_type_label(&envelope.kind) == "other" {
        return Ok(None);
    }
    let event: ThinEvent = serde_json::from_slice(body).map_err(|_| "invalid webhook event")?;
    let related = event
        .related_object
        .ok_or("webhook event has no related object")?;
    if !valid_object_id(&event.id) || !valid_object_id(&related.id) {
        return Err("invalid webhook object identifier");
    }
    Ok(Some(WebhookEvent {
        id: event.id,
        kind: envelope.kind,
        object_id: related.id,
    }))
}

fn valid_object_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 255 && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
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
    fn secret_rotation_accepts_either_secret_in_either_order() {
        let ts = 1_700_000_000;
        let now = UNIX_EPOCH + Duration::from_secs(ts);
        let old = make_signature(ts, TEST_BODY, TEST_SECRET);
        let new = make_signature(ts, TEST_BODY, "whsec_rotated");
        for (first, second) in [(&old, &new), (&new, &old)] {
            let header = format!("{first}, {}", second.split_once(',').unwrap().1);
            for secret in [TEST_SECRET, "whsec_rotated"] {
                assert_eq!(verify_signature_at(&header, TEST_BODY, secret, now), Ok(()));
            }
            assert_eq!(
                verify_signature_at(&header, TEST_BODY, "whsec_unrelated", now),
                Err("signature mismatch")
            );
        }
    }

    #[test]
    fn malformed_signatures_do_not_hide_a_valid_v1() {
        let ts = 1_700_000_000;
        let now = UNIX_EPOCH + Duration::from_secs(ts);
        let valid = make_signature(ts, TEST_BODY, TEST_SECRET);
        for malformed in [
            String::new(),
            "not-hex".into(),
            "ab".repeat(31),
            "ab".repeat(33),
        ] {
            for header in [
                format!("v1={malformed},{valid}"),
                format!("{valid},v1={malformed}"),
            ] {
                assert_eq!(
                    verify_signature_at(&header, TEST_BODY, TEST_SECRET, now),
                    Ok(())
                );
            }
            assert_eq!(
                verify_signature_at(
                    &format!("t={ts},v1={malformed}"),
                    TEST_BODY,
                    TEST_SECRET,
                    now
                ),
                Err("signature mismatch")
            );
        }
    }

    #[test]
    fn other_signature_versions_do_not_authenticate() {
        let ts = 1_700_000_000;
        let now = UNIX_EPOCH + Duration::from_secs(ts);
        let valid = make_signature(ts, TEST_BODY, TEST_SECRET);
        for version in ["v0=", "v2=", "unknown="] {
            let header = valid.replace("v1=", version);
            assert_eq!(
                verify_signature_at(&header, TEST_BODY, TEST_SECRET, now),
                Err("missing v1 signature")
            );
            assert_eq!(
                verify_signature_at(
                    &format!("{header},v1={}", "00".repeat(32)),
                    TEST_BODY,
                    TEST_SECRET,
                    now
                ),
                Err("signature mismatch")
            );
        }
    }

    #[test]
    fn rotation_still_authenticates_the_exact_body_and_timestamp() {
        let ts = 1_700_000_000;
        let old = make_signature(ts, TEST_BODY, TEST_SECRET);
        let new = make_signature(ts, TEST_BODY, "whsec_rotated");
        let header = format!("{old},{}", new.split_once(',').unwrap().1);
        for secret in [TEST_SECRET, "whsec_rotated"] {
            assert_eq!(
                verify_signature_at(
                    &header,
                    b"{ \"id\": \"evt_test\" }",
                    secret,
                    UNIX_EPOCH + Duration::from_secs(ts)
                ),
                Err("signature mismatch")
            );
            for now in [ts - 301, ts + 301] {
                assert_eq!(
                    verify_signature_at(
                        &header,
                        TEST_BODY,
                        secret,
                        UNIX_EPOCH + Duration::from_secs(now)
                    ),
                    Err("stale timestamp")
                );
            }
        }
    }

    #[test]
    fn unrepresentable_timestamp_is_rejected() {
        let header = make_signature(u64::MAX, TEST_BODY, TEST_SECRET);
        assert_eq!(
            verify_signature_at(&header, TEST_BODY, TEST_SECRET, UNIX_EPOCH),
            Err("invalid timestamp")
        );
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
    fn signature_clock_skew_is_bounded_in_both_directions() {
        let ts = 1_700_000_000;
        let event_time = UNIX_EPOCH + Duration::from_secs(ts);
        let header = make_signature(ts, TEST_BODY, TEST_SECRET);
        for offset in [
            Duration::ZERO,
            Duration::from_secs(1),
            Duration::from_secs(300),
        ] {
            for now in [event_time - offset, event_time + offset] {
                assert_eq!(
                    verify_signature_at(&header, TEST_BODY, TEST_SECRET, now),
                    Ok(())
                );
            }
        }
        for offset in [
            Duration::from_secs(300) + Duration::from_nanos(1),
            Duration::from_secs(301),
            Duration::from_secs(86_400),
        ] {
            for now in [event_time - offset, event_time + offset] {
                assert_eq!(
                    verify_signature_at(&header, TEST_BODY, TEST_SECRET, now),
                    Err("stale timestamp")
                );
            }
        }
    }

    fn thin(value: serde_json::Value) -> Vec<u8> {
        serde_json::to_vec(&value).unwrap()
    }

    #[test]
    fn a_thin_notification_keeps_only_its_event_type_and_related_object_ids() {
        // The shape of Stripe's documented v1.charge.dispute.created payload.
        let body = thin(serde_json::json!({
            "context": null,
            "created": "2025-01-01T00:00:00.000Z",
            "id": "evt_65TKLQpAv1Sm67uODUs16TKLJo0qSQNzPlJ0NZGpF32MVk",
            "object": "v2.core.event",
            "reason": {"type": "request", "request": {
                "id": "req_yK7ktaG8CO3xK5",
                "idempotency_key": "672c7b35-f590-4946-bdcf-1206bcd52530"
            }},
            "related_object": {
                "id": "du_1MtJUT2eZvKYlo2CNaw2HvEv",
                "type": "dispute",
                "url": "/v1/charges/ch_3MmlLrLkdIwHu7ix0snN0B15/dispute"
            },
            "type": "v1.charge.dispute.created",
            "snapshot_event": "evt_1SAxTSNX9ddFk9bgxDuoHH3A",
            "livemode": true
        }));
        assert_eq!(
            parse_event(&body),
            Ok(Some(WebhookEvent {
                id: "evt_65TKLQpAv1Sm67uODUs16TKLJo0qSQNzPlJ0NZGpF32MVk".into(),
                kind: "v1.charge.dispute.created".into(),
                object_id: "du_1MtJUT2eZvKYlo2CNaw2HvEv".into(),
            }))
        );
    }

    #[test]
    fn snapshot_events_are_refused_whatever_their_type() {
        for body in [
            serde_json::json!({"id":"evt_test","object":"event","type":"invoice.paid",
                "data":{"object":{"id":"in_test","customer_email":"never accept this"}}}),
            serde_json::json!({"id":"evt_test","object":"event","type":"v1.invoice.paid",
                "related_object":{"id":"in_test"}}),
            serde_json::json!({"id":"evt_test","object":"event","type":"customer.updated",
                "data":{"object":{"id":"cus_test"}}}),
            serde_json::json!({"id":"evt_test","type":"v1.invoice.paid",
                "related_object":{"id":"in_test"}}),
            serde_json::json!({"id":"evt_test","object":null,"type":"v1.invoice.paid",
                "related_object":{"id":"in_test"}}),
        ] {
            assert_eq!(
                parse_event(&thin(body)),
                Err("only thin events are accepted")
            );
        }
    }

    #[test]
    fn unsupported_thin_events_are_ignored_without_requiring_a_related_object() {
        for body in [
            serde_json::json!({"id":"evt_meter","object":"v2.core.event",
                "type":"v1.billing.meter.no_meter_found","related_object":null}),
            serde_json::json!({"id":"evt_meter","object":"v2.core.event",
                "type":"v1.billing.meter.no_meter_found"}),
            serde_json::json!({"id":"evt_customer","object":"v2.core.event",
                "type":"v1.customer.updated","related_object":{"id":"cus_test"}}),
        ] {
            assert_eq!(parse_event(&thin(body)), Ok(None));
        }
    }

    #[test]
    fn supported_events_still_require_valid_event_and_related_object_ids() {
        for body in [
            serde_json::json!({"object":"v2.core.event","type":"v1.invoice.paid",
                "related_object":{"id":"in_test"}}),
            serde_json::json!({"id":"evt_test","object":"v2.core.event","type":"v1.invoice.paid"}),
            serde_json::json!({"id":"evt_test","object":"v2.core.event","type":"v1.invoice.paid",
                "related_object":null}),
            serde_json::json!({"id":"evt_test","object":"v2.core.event","type":"v1.invoice.paid",
                "related_object":{}}),
            serde_json::json!({"id":"evt/test","object":"v2.core.event","type":"v1.invoice.paid",
                "related_object":{"id":"in_test"}}),
            serde_json::json!({"id":"evt_test","object":"v2.core.event","type":"v1.invoice.paid",
                "related_object":{"id":"in/test"}}),
        ] {
            assert!(parse_event(&thin(body)).is_err());
        }
        let body = br#"{"id":"evt_test","object":"v2.core.event","type":"v1.invoice.paid","related_object":{"id":"in_test","type":"invoice","url":"/v1/invoices/in_test"}}"#;
        assert_eq!(
            parse_event(body),
            Ok(Some(WebhookEvent {
                id: "evt_test".into(),
                kind: "v1.invoice.paid".into(),
                object_id: "in_test".into(),
            }))
        );
    }

    #[test]
    fn every_financial_event_is_an_observation() {
        for kind in [
            "v1.refund.created",
            "v1.refund.updated",
            "v1.refund.failed",
            "v1.checkout.session.async_payment_succeeded",
            "v1.charge.dispute.updated",
            "v1.charge.dispute.funds_reinstated",
        ] {
            assert_ne!(event_type_label(kind), "other");
        }
        // A snapshot type's name is not a thin type's.
        assert_eq!(event_type_label("refund.created"), "other");
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
