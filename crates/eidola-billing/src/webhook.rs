//! Stripe ingress authenticates raw bytes and retains only event identities.
//! Provider payloads are observations, never ledger mutation instructions.

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
    if now.duration_since(event_time).unwrap_or(Duration::MAX) > MAX_SIGNATURE_AGE {
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

use crate::service::BILLING_EVENT_TYPES;

fn event_type_label(kind: &str) -> &'static str {
    BILLING_EVENT_TYPES
        .iter()
        .copied()
        .find(|item| *item == kind)
        .unwrap_or("other")
}

/// The only fields persisted from a verified Stripe event.
#[derive(Debug, PartialEq, Eq)]
pub struct WebhookEvent {
    pub id: String,
    pub kind: String,
    pub object_id: String,
}

/// Parse a previously authenticated payload. Unsupported types are ignored.
/// The caller must verify the signature against these exact bytes first.
pub fn parse_event(body: &[u8]) -> Result<Option<WebhookEvent>, &'static str> {
    let event: StripeEvent = serde_json::from_slice(body).map_err(|_| "invalid webhook event")?;
    if event_type_label(&event.kind) == "other" {
        return Ok(None);
    }
    if !valid_object_id(&event.id) || !valid_object_id(&event.data.object.id) {
        return Err("invalid webhook object identifier");
    }
    Ok(Some(WebhookEvent {
        id: event.id,
        kind: event.kind,
        object_id: event.data.object.id,
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
            for now in [ts - 1, ts + 301] {
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
