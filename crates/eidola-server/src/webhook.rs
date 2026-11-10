//! Stripe webhook handling: signature verification and event dispatch.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use bytes::Bytes;
use deadpool_postgres::Pool;
use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;
use subtle::ConstantTimeEq;
use tracing::{error, info, warn};

use crate::AppState;
use crate::db;
use crate::error::ServerError;
use crate::stripe::StripeClient;

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

// ---------------------------------------------------------------------------
// Event types (minimal deserialization)
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct StripeEvent {
    id: String,
    #[serde(rename = "type")]
    event_type: String,
    /// Unix timestamp (seconds) Stripe stamped on the event when it was
    /// created. Used to anchor any "X seconds from the event" expiry
    /// calculations so the resulting timestamp is a deterministic function
    /// of the event itself, independent of when our handler runs.
    created: i64,
    data: StripeEventData,
}

#[derive(serde::Deserialize)]
struct StripeEventData {
    object: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Webhook outcome
// ---------------------------------------------------------------------------

/// How one delivery ended, carrying the bounded `reason` label the
/// `WEBHOOK_OUTCOME` counter records. Every reason is a code-authored
/// constant — never a value off the event — so the label set stays fixed
/// (privacy-guarantees.md §3.3) while still separating the failure modes:
/// a ledger write is `handled`, a redelivery of a spent event is
/// `duplicate`, a permanently unusable event names what it lacked, and a
/// transient failure names which dependency failed.
enum WebhookOutcome {
    /// Answer 200 so Stripe stops redelivering — the event was applied,
    /// deliberately ignored, or is permanently unusable (retrying cannot
    /// change what the event contains).
    Done(&'static str),
    /// Answer 500 so Stripe redelivers.
    Retryable(&'static str),
}

impl WebhookOutcome {
    fn is_retryable(&self) -> bool {
        matches!(self, Self::Retryable(_))
    }

    fn reason(&self) -> &'static str {
        match self {
            Self::Done(reason) | Self::Retryable(reason) => reason,
        }
    }
}

/// The Stripe event types this handler dispatches on.
///
/// Doubles as the label domain for `WEBHOOK_OUTCOME` and for every log
/// line that mentions an event type: `event_type` arrives from Stripe, so
/// it is caller-derived and must not become a metric label or a logged
/// value unfiltered. Resolving through this list bounds the cardinality;
/// anything outside it — including a new type Stripe starts sending —
/// lands on `other`, which is also the dispatch table's fallthrough.
const HANDLED_EVENT_TYPES: &[&str] = &[
    "charge.dispute.closed",
    "charge.dispute.created",
    "charge.refunded",
    "checkout.session.completed",
    "customer.subscription.deleted",
    "invoice.paid",
];

fn event_type_label(event_type: &str) -> &'static str {
    HANDLED_EVENT_TYPES
        .iter()
        .copied()
        .find(|known| *known == event_type)
        .unwrap_or("other")
}

/// Record how one webhook delivery ended.
///
/// Stripe originates these requests, so we can never opt one into a
/// per-request trace the way we can with our own traffic. This counter is
/// what keeps the failure modes visible in aggregate instead.
fn record_webhook_outcome(event_type: &str, reason: &'static str) {
    crate::telemetry::metrics::WEBHOOK_OUTCOME.add(
        1,
        &[
            opentelemetry::KeyValue::new("event_type", event_type_label(event_type)),
            opentelemetry::KeyValue::new("reason", reason),
        ],
    );
}

// ---------------------------------------------------------------------------
// Main handler
// ---------------------------------------------------------------------------

/// POST /v1/webhooks/stripe
#[utoipa::path(
    post,
    path = "/v1/webhooks/stripe",
    tag = "Linked",
    responses(
        (status = 200, description = "Event received"),
        (status = 400, description = "Signature verification failed", body = crate::types::ErrorResponse),
        (status = 503, description = "Webhook secret not configured", body = crate::types::ErrorResponse)
    )
)]
#[tracing::instrument(skip_all, name = "webhook.process", fields(otel.status_code = tracing::field::Empty))]
pub async fn stripe_webhook(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    // The pre-parse rejections have no event type yet, so they count against
    // the `other` bucket. That is honest: we genuinely do not know what
    // Stripe was trying to tell us.
    let webhook_secret = match &state.stripe_webhook_secret {
        Some(s) => s,
        None => {
            record_webhook_outcome("", "not_configured");
            return ServerError::ServiceUnavailable("webhook secret is not configured".to_string())
                .into_response();
        }
    };

    let sig_header = match headers
        .get("stripe-signature")
        .and_then(|v| v.to_str().ok())
    {
        Some(h) => h.to_string(),
        None => {
            record_webhook_outcome("", "missing_signature");
            return bad_request("missing stripe-signature header");
        }
    };

    if let Err(reason) = verify_signature(&sig_header, &body, webhook_secret) {
        warn!("webhook: signature verification failed: {}", reason);
        record_webhook_outcome("", "bad_signature");
        return bad_request(&format!("signature verification failed: {}", reason));
    }

    let event: StripeEvent = match serde_json::from_slice(&body) {
        Ok(e) => e,
        Err(e) => {
            // serde messages quote offending values from the Stripe-authored
            // body; log only the category summary.
            error!(
                "webhook: failed to parse event: {}",
                crate::error::parse_error_summary(&e)
            );
            record_webhook_outcome("", "unparseable");
            return ok_empty();
        }
    };

    // `event_type` is Stripe-authored, so the log carries its fixed-list
    // resolution, never the verbatim value — and no `event.id`: a Stripe
    // identifier in the log would join our diagnostics to Stripe's own
    // records (privacy-guarantees.md §3.3). The database keeps the id for
    // idempotency; logs stay identifier-free.
    info!("webhook: received {}", event_type_label(&event.event_type));

    let outcome = match event.event_type.as_str() {
        "checkout.session.completed" => {
            handle_checkout_completed(&event, &state.db_pool, &state.stripe).await
        }
        "invoice.paid" => handle_invoice_paid(&event, &state.db_pool, &state.stripe).await,
        "customer.subscription.deleted" => {
            info!("webhook: subscription deleted, no ledger action");
            WebhookOutcome::Done("ignored")
        }
        "charge.refunded" => handle_charge_refunded(&event, &state.db_pool).await,
        "charge.dispute.created" => {
            handle_dispute_created(&event, &state.db_pool, &state.stripe).await
        }
        "charge.dispute.closed" => {
            handle_dispute_closed(&event, &state.db_pool, &state.stripe).await
        }
        _ => {
            // The verbatim type is unrecognized by definition here, so
            // there is nothing loggable beyond the fact of the fallthrough
            // (the receipt log above already recorded the `other` label).
            info!("webhook: ignoring unhandled event type");
            WebhookOutcome::Done("ignored")
        }
    };

    record_webhook_outcome(&event.event_type, outcome.reason());
    if outcome.is_retryable() {
        // This handler returns a response rather than an `Err`, so the
        // `err` instrument sugar every other tracked span uses cannot mark
        // the span — without this record, `operation.duration`'s outcome
        // for webhook.process could never read "error".
        tracing::Span::current().record("otel.status_code", "ERROR");
        return internal_error();
    }
    ok_empty()
}

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

async fn handle_checkout_completed(
    event: &StripeEvent,
    pool: &Pool,
    stripe: &Option<StripeClient>,
) -> WebhookOutcome {
    let obj = &event.data.object;

    // `mode` is Stripe-authored, so the log states the branch taken rather
    // than the verbatim value — the same rule as every identifier below
    // (session, customer, product ids): they steer the code and reach the
    // database where the flow needs them, but never the logs. The success
    // lines here and in the other handlers carry no account id or amount
    // either: the ledger already records both durably, and an exported log
    // stream naming them would hand the telemetry vendor a per-account
    // payment history.
    let mode = obj["mode"].as_str().unwrap_or("");
    if mode != "payment" {
        info!(
            "webhook: checkout mode is not payment, skipping (invoice.paid handles subscriptions)"
        );
        return WebhookOutcome::Done("ignored");
    }

    let session_id = match obj["id"].as_str() {
        Some(id) => id,
        None => {
            error!("webhook: checkout.session.completed missing session id");
            return WebhookOutcome::Done("malformed_event");
        }
    };

    let customer_id = match obj["customer"].as_str() {
        Some(c) => c,
        None => {
            error!("webhook: checkout.session.completed missing customer");
            return WebhookOutcome::Done("malformed_event");
        }
    };

    let account = match db::get_account_by_stripe_customer(pool, customer_id).await {
        Ok(Some(a)) => a,
        Ok(None) => {
            warn!("webhook: orphan customer in checkout.session.completed");
            return WebhookOutcome::Done("orphan_customer");
        }
        Err(e) => {
            error!("webhook: db error looking up customer: {}", e);
            return WebhookOutcome::Retryable("db_error");
        }
    };

    let stripe = match stripe.as_ref() {
        Some(s) => s,
        None => {
            error!("webhook: stripe client not configured, cannot fetch line items");
            return WebhookOutcome::Retryable("stripe_unconfigured");
        }
    };

    let line_items = match stripe.list_checkout_line_items(session_id).await {
        Ok(items) => items,
        Err(e) => {
            error!("webhook: failed to fetch line items for checkout: {}", e);
            return WebhookOutcome::Retryable("stripe_api_error");
        }
    };

    let product_id = match line_items.first().map(|item| item.price.product.as_str()) {
        Some(id) => id,
        None => {
            error!("webhook: no line items for checkout session");
            return WebhookOutcome::Done("no_line_items");
        }
    };

    let product = match stripe.get_product(product_id).await {
        Ok(p) => p,
        Err(e) => {
            error!("webhook: failed to fetch product for checkout: {}", e);
            return WebhookOutcome::Retryable("stripe_api_error");
        }
    };

    let credits = match product
        .metadata
        .get("credits")
        .and_then(|s| s.parse::<i64>().ok())
    {
        Some(c) => c,
        None => {
            error!("webhook: no credits metadata on checkout product");
            return WebhookOutcome::Done("no_credits_metadata");
        }
    };

    // Anchor the 1-year expiry to the Stripe event's `created` timestamp
    // rather than wall-clock at handler time. This makes `expires_at` a
    // deterministic function of the event (idempotent across redeliveries
    // and consistent with what the customer's receipt implies) and matches
    // the pattern used by `handle_invoice_paid` for subscription renewals.
    let event_time = UNIX_EPOCH + Duration::from_secs(event.created.max(0) as u64);
    let expires_at = event_time + Duration::from_secs(365 * 24 * 3600);

    // Payment-mode checkout sessions carry the PaymentIntent; recording it
    // lets a later charge.refunded event match the refund back to this pool.
    let payment_intent = obj["payment_intent"].as_str();

    match db::insert_credit_ledger(
        pool,
        account.id,
        credits,
        "purchase",
        &event.id,
        payment_intent,
        Some(expires_at),
    )
    .await
    {
        Ok(true) => {
            info!("webhook: ledger credited (purchase)");
            WebhookOutcome::Done("handled")
        }
        Ok(false) => {
            info!("webhook: duplicate event, skipping");
            WebhookOutcome::Done("duplicate")
        }
        Err(e) => {
            error!("webhook: failed to insert ledger entry: {}", e);
            WebhookOutcome::Retryable("db_error")
        }
    }
}

async fn handle_invoice_paid(
    event: &StripeEvent,
    pool: &Pool,
    stripe: &Option<StripeClient>,
) -> WebhookOutcome {
    let obj = &event.data.object;

    let billing_reason = obj["billing_reason"].as_str().unwrap_or("");
    if billing_reason != "subscription_create" && billing_reason != "subscription_cycle" {
        info!("webhook: invoice.paid billing_reason is not a subscription charge, skipping");
        return WebhookOutcome::Done("ignored");
    }

    let customer_id = match obj["customer"].as_str() {
        Some(c) => c,
        None => {
            error!("webhook: invoice.paid missing customer");
            return WebhookOutcome::Done("malformed_event");
        }
    };

    let account = match db::get_account_by_stripe_customer(pool, customer_id).await {
        Ok(Some(a)) => a,
        Ok(None) => {
            warn!("webhook: orphan customer in invoice.paid");
            return WebhookOutcome::Done("orphan_customer");
        }
        Err(e) => {
            error!("webhook: db error looking up customer: {}", e);
            return WebhookOutcome::Retryable("db_error");
        }
    };

    let lines = &obj["lines"]["data"];
    let first_line = match lines.as_array().and_then(|arr| arr.first()) {
        Some(line) => line,
        None => {
            error!("webhook: no line items on invoice");
            return WebhookOutcome::Done("no_line_items");
        }
    };

    let product_id = match first_line["pricing"]["price_details"]["product"].as_str() {
        Some(id) => id,
        None => {
            error!("webhook: no product ID on invoice line item");
            return WebhookOutcome::Done("malformed_event");
        }
    };

    let stripe = match stripe.as_ref() {
        Some(s) => s,
        None => {
            error!("webhook: stripe client not configured, cannot fetch product");
            return WebhookOutcome::Retryable("stripe_unconfigured");
        }
    };

    let product = match stripe.get_product(product_id).await {
        Ok(p) => p,
        Err(e) => {
            error!("webhook: failed to fetch product for invoice: {}", e);
            return WebhookOutcome::Retryable("stripe_api_error");
        }
    };

    let credits = match product
        .metadata
        .get("credits")
        .and_then(|s| s.parse::<i64>().ok())
    {
        Some(c) => c,
        None => {
            error!("webhook: no credits metadata on invoice product");
            return WebhookOutcome::Done("no_credits_metadata");
        }
    };

    let period_end = first_line["period"]["end"].as_i64();
    let expires_at = period_end.map(|ts| UNIX_EPOCH + Duration::from_secs(ts as u64));

    // Best-effort: older Stripe API versions expose the PaymentIntent on the
    // invoice object; newer ones moved it under `payments` (not expanded
    // here). When absent, refunds of this charge fall back to pool-order
    // debiting in debit_account.
    let payment_intent = obj["payment_intent"].as_str();

    match db::insert_credit_ledger(
        pool,
        account.id,
        credits,
        "subscription_renewal",
        &event.id,
        payment_intent,
        expires_at,
    )
    .await
    {
        Ok(true) => {
            info!("webhook: ledger credited (subscription_renewal)");
            WebhookOutcome::Done("handled")
        }
        Ok(false) => {
            info!("webhook: duplicate event, skipping");
            WebhookOutcome::Done("duplicate")
        }
        Err(e) => {
            error!("webhook: failed to insert ledger entry: {}", e);
            WebhookOutcome::Retryable("db_error")
        }
    }
}

async fn handle_charge_refunded(event: &StripeEvent, pool: &Pool) -> WebhookOutcome {
    let obj = &event.data.object;

    let customer_id = match obj["customer"].as_str() {
        Some(c) => c,
        None => {
            error!("webhook: charge.refunded missing customer");
            return WebhookOutcome::Done("malformed_event");
        }
    };

    let account = match db::get_account_by_stripe_customer(pool, customer_id).await {
        Ok(Some(a)) => a,
        Ok(None) => {
            warn!("webhook: orphan customer in charge.refunded");
            return WebhookOutcome::Done("orphan_customer");
        }
        Err(e) => {
            error!("webhook: db error looking up customer: {}", e);
            return WebhookOutcome::Retryable("db_error");
        }
    };

    let amount_refunded = match obj["amount_refunded"].as_i64() {
        Some(a) => a,
        None => {
            error!("webhook: charge.refunded missing amount_refunded");
            return WebhookOutcome::Done("malformed_event");
        }
    };

    if cents_to_microdollars(amount_refunded).is_none() {
        error!("webhook: amount overflow in charge.refunded");
        return WebhookOutcome::Done("amount_overflow");
    }

    // `amount_refunded` is the charge's refunded total, and this event fires
    // for every partial refund: `refund_charge` debits only what the
    // charge's earlier refunds have not. The PaymentIntent both tags those
    // debits and, when it links to the original credit entry, matches the
    // refund to that pool (inheriting its expires_at even if it has lapsed,
    // so goodwill refunds of expired credits net to zero) and voids the
    // credits it granted pro rata to the charge `amount` refunded.
    let payment_intent = obj["payment_intent"].as_str();
    // Required, like `amount_refunded`: a Charge always carries `amount`, and
    // defaulting its absence to "unknown" selected a financial meaning (the
    // at-par target) for a payload this code does not understand — a full
    // refund of a $20 purchase that granted 16,000,000 credits would have
    // written -20,000,000.
    let charge_amount = match obj["amount"].as_i64() {
        Some(a) => a,
        None => {
            error!("webhook: charge.refunded missing amount");
            return WebhookOutcome::Done("malformed_event");
        }
    };

    match db::refund_charge(
        pool,
        account.id,
        &event.id,
        payment_intent,
        charge_amount,
        amount_refunded,
    )
    .await
    {
        Ok(db::RefundOutcome::Debited) => {
            info!("webhook: ledger debited (refund)");
            WebhookOutcome::Done("handled")
        }
        Ok(db::RefundOutcome::Duplicate) => {
            info!("webhook: duplicate event, skipping");
            WebhookOutcome::Done("duplicate")
        }
        Ok(db::RefundOutcome::NothingNew) => {
            info!("webhook: charge's refunds already debited, skipping");
            WebhookOutcome::Done("already_refunded")
        }
        Err(e) => {
            error!("webhook: failed to debit refund: {}", e);
            WebhookOutcome::Retryable("db_error")
        }
    }
}

/// The account a `charge.dispute.*` event concerns.
///
/// Stripe's Dispute object has no `customer` field (nor anything like a
/// `charge_object`): it names only the disputed `charge`, as an unexpanded
/// id in webhook payloads. The customer lives on that Charge, so it is
/// fetched — the same shape as the product lookup `invoice.paid` makes,
/// and a failure to reach Stripe is likewise retryable.
async fn dispute_account(
    obj: &serde_json::Value,
    pool: &Pool,
    stripe: &Option<StripeClient>,
    event_label: &'static str,
) -> Result<db::AccountRow, WebhookOutcome> {
    let Some(charge_id) = obj["charge"].as_str() else {
        error!("webhook: {event_label} missing charge");
        return Err(WebhookOutcome::Done("malformed_event"));
    };
    let Some(stripe) = stripe.as_ref() else {
        error!("webhook: stripe client not configured, cannot fetch disputed charge");
        return Err(WebhookOutcome::Retryable("stripe_unconfigured"));
    };
    let customer_id = match stripe.get_charge_customer(charge_id).await {
        Ok(Some(c)) => c,
        Ok(None) => {
            warn!("webhook: disputed charge in {event_label} has no customer");
            return Err(WebhookOutcome::Done("orphan_customer"));
        }
        Err(e) => {
            error!("webhook: failed to fetch disputed charge: {}", e);
            return Err(WebhookOutcome::Retryable("stripe_api_error"));
        }
    };
    match db::get_account_by_stripe_customer(pool, &customer_id).await {
        Ok(Some(a)) => Ok(a),
        Ok(None) => {
            warn!("webhook: orphan customer in {event_label}");
            Err(WebhookOutcome::Done("orphan_customer"))
        }
        Err(e) => {
            error!("webhook: db error looking up customer: {}", e);
            Err(WebhookOutcome::Retryable("db_error"))
        }
    }
}

async fn handle_dispute_created(
    event: &StripeEvent,
    pool: &Pool,
    stripe: &Option<StripeClient>,
) -> WebhookOutcome {
    let obj = &event.data.object;

    let account = match dispute_account(obj, pool, stripe, "charge.dispute.created").await {
        Ok(a) => a,
        Err(outcome) => return outcome,
    };

    // The dispute id is what its `closed` event will reverse by: the
    // clawback row carries it as `memo`.
    let Some(dispute_id) = obj["id"].as_str() else {
        error!("webhook: charge.dispute.created missing dispute id");
        return WebhookOutcome::Done("malformed_event");
    };

    let amount = match obj["amount"].as_i64() {
        Some(a) => a,
        None => {
            error!("webhook: charge.dispute.created missing amount");
            return WebhookOutcome::Done("malformed_event");
        }
    };

    let microdollars = match cents_to_microdollars(amount) {
        Some(d) => d,
        None => {
            error!("webhook: amount overflow in charge.dispute.created");
            return WebhookOutcome::Done("amount_overflow");
        }
    };

    match db::debit_stripe_event(
        pool,
        account.id,
        microdollars,
        "dispute_clawback",
        &event.id,
        Some(dispute_id),
    )
    .await
    {
        Ok(true) => {
            info!("webhook: ledger debited (dispute_clawback)");
            WebhookOutcome::Done("handled")
        }
        Ok(false) => {
            info!("webhook: duplicate event, skipping");
            WebhookOutcome::Done("duplicate")
        }
        Err(e) => {
            error!("webhook: failed to insert ledger entry: {}", e);
            WebhookOutcome::Retryable("db_error")
        }
    }
}

async fn handle_dispute_closed(
    event: &StripeEvent,
    pool: &Pool,
    stripe: &Option<StripeClient>,
) -> WebhookOutcome {
    let obj = &event.data.object;

    let status = obj["status"].as_str().unwrap_or("");
    if status != "won" {
        info!("webhook: dispute closed without a won status, no reversal");
        return WebhookOutcome::Done("ignored");
    }

    let account = match dispute_account(obj, pool, stripe, "charge.dispute.closed").await {
        Ok(a) => a,
        Err(outcome) => return outcome,
    };

    let Some(dispute_id) = obj["id"].as_str() else {
        error!("webhook: charge.dispute.closed missing dispute id");
        return WebhookOutcome::Done("malformed_event");
    };

    // What comes back is what the dispute's clawback took, read off the
    // ledger — not the payload's `amount`. A dispute this server never
    // debited (its `created` event arrived before the handler could resolve
    // the account, or was exhausted) has nothing to reverse; crediting the
    // payload's amount would mint balance the account never lost, and
    // balance converts to unlinkable credentials.
    match db::reverse_dispute_clawback(pool, account.id, dispute_id, &event.id).await {
        Ok(db::DisputeReversal::Credited(_)) => {
            info!("webhook: ledger credited (dispute_reversal)");
            WebhookOutcome::Done("handled")
        }
        Ok(db::DisputeReversal::Duplicate) => {
            info!("webhook: duplicate event, skipping");
            WebhookOutcome::Done("duplicate")
        }
        Ok(db::DisputeReversal::NothingToReverse) => {
            warn!("webhook: dispute won with no clawback on the ledger, nothing to reverse");
            WebhookOutcome::Done("no_clawback_recorded")
        }
        Err(e) => {
            error!("webhook: failed to reverse dispute clawback: {}", e);
            WebhookOutcome::Retryable("db_error")
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cents_to_microdollars(cents: i64) -> Option<i64> {
    cents.checked_mul(10_000)
}

fn internal_error() -> axum::response::Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": "internal error"})),
    )
        .into_response()
}

fn ok_empty() -> axum::response::Response {
    (StatusCode::OK, Json(serde_json::json!({}))).into_response()
}

fn bad_request(msg: &str) -> axum::response::Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": msg})),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Integration tests (require running postgres: just db && just db-reset)
    // -----------------------------------------------------------------------

    fn test_pool() -> Pool {
        let url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
        db::create_pool(&url, None, None).expect("failed to create test pool")
    }

    async fn create_test_account(pool: &Pool, customer_id: &str) -> uuid::Uuid {
        let id = uuid::Uuid::new_v4();
        db::insert_account(pool, id, "test_hash").await.unwrap();
        db::set_stripe_customer_id(pool, id, customer_id)
            .await
            .unwrap();
        id
    }

    fn make_event(id: &str, event_type: &str, object: serde_json::Value) -> StripeEvent {
        StripeEvent {
            id: id.to_string(),
            event_type: event_type.to_string(),
            created: now_secs() as i64,
            data: StripeEventData { object },
        }
    }

    async fn get_balance(pool: &Pool, account_id: uuid::Uuid) -> i64 {
        let client = pool.get().await.unwrap();
        let row = client
            .query_one("SELECT available_balance($1) as balance", &[&account_id])
            .await
            .unwrap();
        row.get("balance")
    }

    async fn count_ledger_entries(pool: &Pool, account_id: uuid::Uuid) -> i64 {
        let client = pool.get().await.unwrap();
        let row = client
            .query_one(
                "SELECT COUNT(*)::bigint as cnt FROM credit_ledger WHERE account_id = $1",
                &[&account_id],
            )
            .await
            .unwrap();
        row.get("cnt")
    }

    fn test_stripe() -> Option<StripeClient> {
        std::env::var("STRIPE_API_KEY").ok().map(StripeClient::new)
    }

    fn test_product_id() -> String {
        std::env::var("TEST_STRIPE_PRODUCT_ID")
            .expect("TEST_STRIPE_PRODUCT_ID must be set for invoice integration tests")
    }

    #[tokio::test]
    #[ignore]
    async fn invoice_paid_credits_account() {
        let pool = test_pool();
        let stripe = test_stripe();
        let product_id = test_product_id();
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(&pool, &cust).await;

        let period_end = now_secs() + 30 * 24 * 3600;
        let event = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "invoice.paid",
            serde_json::json!({
                "customer": cust,
                "billing_reason": "subscription_create",
                "lines": {
                    "data": [{
                        "pricing": {
                            "price_details": { "product": product_id }
                        },
                        "period": { "end": period_end }
                    }]
                }
            }),
        );

        let outcome = handle_invoice_paid(&event, &pool, &stripe).await;

        assert!(!outcome.is_retryable());
        assert_eq!(count_ledger_entries(&pool, account_id).await, 1);
        assert!(get_balance(&pool, account_id).await > 0);
    }

    #[tokio::test]
    #[ignore]
    async fn invoice_paid_skips_irrelevant_billing_reason() {
        let pool = test_pool();
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(&pool, &cust).await;

        let event = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "invoice.paid",
            serde_json::json!({
                "customer": cust,
                "billing_reason": "manual",
                "lines": { "data": [] }
            }),
        );

        let outcome = handle_invoice_paid(&event, &pool, &None).await;

        assert!(!outcome.is_retryable());
        assert_eq!(get_balance(&pool, account_id).await, 0);
        assert_eq!(count_ledger_entries(&pool, account_id).await, 0);
    }

    #[tokio::test]
    #[ignore]
    async fn charge_refunded_debits_account() {
        let pool = test_pool();
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(&pool, &cust).await;

        db::insert_credit_ledger(
            &pool,
            account_id,
            50_000_000,
            "subscription_renewal",
            &format!("evt_seed_{}", uuid::Uuid::new_v4()),
            None,
            None,
        )
        .await
        .unwrap();

        let event = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "charge.refunded",
            serde_json::json!({
                "customer": cust,
                "amount": 1000,
                "amount_refunded": 1000
            }),
        );

        let outcome = handle_charge_refunded(&event, &pool).await;

        assert!(!outcome.is_retryable());
        assert_eq!(get_balance(&pool, account_id).await, 40_000_000);
        assert_eq!(count_ledger_entries(&pool, account_id).await, 2);
    }

    /// Stripe sends `charge.refunded` for every partial refund, each a new
    /// event, and the Charge's `amount_refunded` is its refunded total so
    /// far. Two $10 partial refunds of one charge must void $20, not
    /// $10 + $20.
    #[tokio::test]
    #[ignore]
    async fn second_partial_refund_debits_only_the_new_amount_matched() {
        let pool = test_pool();
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(&pool, &cust).await;
        let pi = format!("pi_test_{}", uuid::Uuid::new_v4());
        let expires = std::time::SystemTime::now() + std::time::Duration::from_secs(86_400);
        db::insert_credit_ledger(
            &pool,
            account_id,
            100_000_000,
            "purchase",
            &format!("evt_seed_{}", uuid::Uuid::new_v4()),
            Some(&pi),
            Some(expires),
        )
        .await
        .unwrap();

        for cumulative_cents in [1000, 2000] {
            let event = make_event(
                &format!("evt_{}", uuid::Uuid::new_v4()),
                "charge.refunded",
                serde_json::json!({
                    "customer": cust,
                    "payment_intent": pi,
                    "amount": 10000,
                    "amount_refunded": cumulative_cents
                }),
            );
            assert!(!handle_charge_refunded(&event, &pool).await.is_retryable());
        }

        assert_eq!(get_balance(&pool, account_id).await, 80_000_000);
    }

    /// The same two partial refunds through the fallback path: a
    /// subscription credit recorded without a PaymentIntent (the pinned API
    /// version's invoices carry none), refunded by a charge that has one.
    #[tokio::test]
    #[ignore]
    async fn second_partial_refund_debits_only_the_new_amount_unmatched() {
        let pool = test_pool();
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(&pool, &cust).await;
        let pi = format!("pi_test_{}", uuid::Uuid::new_v4());
        db::insert_credit_ledger(
            &pool,
            account_id,
            100_000_000,
            "subscription_renewal",
            &format!("evt_seed_{}", uuid::Uuid::new_v4()),
            None,
            None,
        )
        .await
        .unwrap();

        for cumulative_cents in [1000, 2000] {
            let event = make_event(
                &format!("evt_{}", uuid::Uuid::new_v4()),
                "charge.refunded",
                serde_json::json!({
                    "customer": cust,
                    "payment_intent": pi,
                    "amount": 10000,
                    "amount_refunded": cumulative_cents
                }),
            );
            assert!(!handle_charge_refunded(&event, &pool).await.is_retryable());
        }

        assert_eq!(get_balance(&pool, account_id).await, 80_000_000);
    }

    /// A $100 purchase linked to its PaymentIntent, for the refund tests
    /// below. Returns the customer and PaymentIntent ids.
    async fn seed_linked_purchase(pool: &Pool) -> (uuid::Uuid, String, String) {
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(pool, &cust).await;
        let pi = format!("pi_test_{}", uuid::Uuid::new_v4());
        let expires = std::time::SystemTime::now() + std::time::Duration::from_secs(86_400);
        db::insert_credit_ledger(
            pool,
            account_id,
            100_000_000,
            "purchase",
            &format!("evt_seed_{}", uuid::Uuid::new_v4()),
            Some(&pi),
            Some(expires),
        )
        .await
        .unwrap();
        (account_id, cust, pi)
    }

    fn partial_refund_event(cust: &str, pi: &str, cumulative_cents: i64) -> StripeEvent {
        make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "charge.refunded",
            serde_json::json!({
                "customer": cust,
                "payment_intent": pi,
                "amount": 10000,
                "amount_refunded": cumulative_cents
            }),
        )
    }

    /// Stripe does not guarantee delivery order: the second partial
    /// refund's event arriving before the first must still void exactly
    /// what was refunded.
    #[tokio::test]
    #[ignore]
    async fn partial_refunds_delivered_out_of_order() {
        let pool = test_pool();
        let (account_id, cust, pi) = seed_linked_purchase(&pool).await;

        for cumulative_cents in [2000, 1000] {
            let event = partial_refund_event(&cust, &pi, cumulative_cents);
            assert!(!handle_charge_refunded(&event, &pool).await.is_retryable());
        }

        assert_eq!(get_balance(&pool, account_id).await, 80_000_000);
    }

    /// Two partial refunds of one charge processed at once must not both
    /// read the same "already debited" total.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn concurrent_partial_refunds_of_one_charge() {
        const ROUNDS: usize = 20;
        let pool = test_pool();

        let mut wrong = Vec::new();
        for round in 0..ROUNDS {
            let (account_id, cust, pi) = seed_linked_purchase(&pool).await;
            let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(2));
            let tasks: Vec<_> = [1000, 2000]
                .into_iter()
                .map(|cumulative_cents| {
                    let event = partial_refund_event(&cust, &pi, cumulative_cents);
                    let (pool, barrier) = (pool.clone(), barrier.clone());
                    tokio::spawn(async move {
                        barrier.wait().await;
                        handle_charge_refunded(&event, &pool).await
                    })
                })
                .collect();
            for task in tasks {
                assert!(!task.await.unwrap().is_retryable());
            }
            let balance = get_balance(&pool, account_id).await;
            if balance != 80_000_000 {
                wrong.push((round, balance));
            }
        }

        assert!(wrong.is_empty(), "rounds with a wrong balance: {wrong:?}");
    }

    /// Terms §5: a refund voids the credits the refunded money bought. A
    /// product's `credits` metadata need not equal its price in
    /// micro-dollars (terms §4: plans differ in price per unit), so a full
    /// refund of a $20 purchase that granted 16,000,000 credits must void
    /// 16,000,000, not 20,000,000.
    #[tokio::test]
    #[ignore]
    async fn full_refund_voids_the_credits_the_purchase_granted() {
        let pool = test_pool();
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(&pool, &cust).await;
        let pi = format!("pi_test_{}", uuid::Uuid::new_v4());
        let expires = std::time::SystemTime::now() + std::time::Duration::from_secs(86_400);
        db::insert_credit_ledger(
            &pool,
            account_id,
            16_000_000,
            "purchase",
            &format!("evt_seed_{}", uuid::Uuid::new_v4()),
            Some(&pi),
            Some(expires),
        )
        .await
        .unwrap();

        let event = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "charge.refunded",
            serde_json::json!({
                "customer": cust,
                "payment_intent": pi,
                "amount": 2000,
                "amount_refunded": 2000
            }),
        );
        assert!(!handle_charge_refunded(&event, &pool).await.is_retryable());

        assert_eq!(get_balance(&pool, account_id).await, 0);
    }

    /// A refund whose charge carries a PaymentIntent linked to an
    /// already-expired purchase pool nets against that pool: zero effect on
    /// the live balance, and the refund debit inherits the pool's (past)
    /// expires_at instead of becoming a permanent negative entry.
    #[tokio::test]
    #[ignore]
    async fn charge_refunded_matches_expired_original_pool() {
        let pool = test_pool();
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(&pool, &cust).await;
        let pi = format!("pi_test_{}", uuid::Uuid::new_v4());

        let expired = std::time::SystemTime::now() - std::time::Duration::from_secs(86_400);
        db::insert_credit_ledger(
            &pool,
            account_id,
            50_000_000,
            "purchase",
            &format!("evt_seed_{}", uuid::Uuid::new_v4()),
            Some(&pi),
            Some(expired),
        )
        .await
        .unwrap();
        assert_eq!(get_balance(&pool, account_id).await, 0);

        let event = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "charge.refunded",
            serde_json::json!({
                "customer": cust,
                "payment_intent": pi,
                "amount": 5000,
                "amount_refunded": 5000
            }),
        );

        let outcome = handle_charge_refunded(&event, &pool).await;

        assert!(!outcome.is_retryable());
        assert_eq!(get_balance(&pool, account_id).await, 0);
        assert_eq!(count_ledger_entries(&pool, account_id).await, 2);

        let client = pool.get().await.unwrap();
        let row = client
            .query_one(
                "SELECT expires_at FROM credit_ledger \
                 WHERE account_id = $1 AND reason = 'refund'",
                &[&account_id],
            )
            .await
            .unwrap();
        let refund_expiry: Option<std::time::SystemTime> = row.get("expires_at");
        assert!(
            refund_expiry.is_some_and(|t| t < std::time::SystemTime::now()),
            "refund debit must inherit the expired pool's past expires_at"
        );
    }

    /// A refund with no PaymentIntent linkage still avoids the phantom
    /// permanent negative: with no live pools, debit_account's expired-pool
    /// backfill nets it against the expired positive pool.
    #[tokio::test]
    #[ignore]
    async fn charge_refunded_without_intent_backfills_expired_pool() {
        let pool = test_pool();
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(&pool, &cust).await;

        let expired = std::time::SystemTime::now() - std::time::Duration::from_secs(86_400);
        db::insert_credit_ledger(
            &pool,
            account_id,
            50_000_000,
            "purchase",
            &format!("evt_seed_{}", uuid::Uuid::new_v4()),
            None,
            Some(expired),
        )
        .await
        .unwrap();

        let event = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "charge.refunded",
            serde_json::json!({
                "customer": cust,
                "amount": 5000,
                "amount_refunded": 5000
            }),
        );

        let outcome = handle_charge_refunded(&event, &pool).await;

        assert!(!outcome.is_retryable());
        assert_eq!(get_balance(&pool, account_id).await, 0);

        let client = pool.get().await.unwrap();
        let row = client
            .query_one(
                "SELECT expires_at FROM credit_ledger \
                 WHERE account_id = $1 AND reason = 'refund'",
                &[&account_id],
            )
            .await
            .unwrap();
        let refund_expiry: Option<std::time::SystemTime> = row.get("expires_at");
        assert!(
            refund_expiry.is_some(),
            "refund debit must land on the expired pool, not the permanent (NULL) pool"
        );
    }

    #[tokio::test]
    #[ignore]
    async fn dispute_created_claws_back() {
        let pool = test_pool();
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(&pool, &cust).await;

        db::insert_credit_ledger(
            &pool,
            account_id,
            50_000_000,
            "subscription_renewal",
            &format!("evt_seed_{}", uuid::Uuid::new_v4()),
            None,
            None,
        )
        .await
        .unwrap();

        let event = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "charge.dispute.created",
            serde_json::json!({
                "id": "dp_plain_123",
                "customer": cust,
                "amount": 500,
                "charge": "ch_test_123"
            }),
        );

        let stripe = stub_charge_customer(Some(cust.clone())).await;
        let outcome = handle_dispute_created(&event, &pool, &stripe).await;

        assert!(!outcome.is_retryable());
        assert_eq!(get_balance(&pool, account_id).await, 45_000_000);
        assert_eq!(count_ledger_entries(&pool, account_id).await, 2);
    }

    /// A stub Stripe answering `GET /charges/{id}` with `customer`, through
    /// the production client (the pattern of `stripe::tests::stub_stripe`).
    async fn stub_charge_customer(customer: Option<String>) -> Option<StripeClient> {
        use axum::routing::get;
        let app = axum::Router::new().route(
            "/charges/{id}",
            get(
                move |axum::extract::Path(id): axum::extract::Path<String>| async move {
                    Json(serde_json::json!({ "id": id, "object": "charge", "customer": customer }))
                },
            ),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        Some(StripeClient::new("sk_test".to_string()).with_base_url(format!("http://{addr}")))
    }

    /// A dispute that cannot be resolved to its account because Stripe is
    /// unreachable or unconfigured must be redelivered, not dropped as a
    /// 200 — the old `malformed_event` ending was silent and final.
    #[tokio::test]
    async fn dispute_whose_charge_cannot_be_fetched_is_retryable() {
        // Never connected: the lookup fails before the pool is used.
        let pool = db::create_pool("postgres://x@127.0.0.1:1/x?sslmode=disable", None, None)
            .expect("lazy pool");
        let event = make_event(
            "evt_dispute_unreachable",
            "charge.dispute.created",
            stripe_shaped_dispute("needs_response"),
        );
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let unreachable = Some(
            StripeClient::new("sk_test".to_string()).with_base_url("http://127.0.0.1:1".into()),
        );

        assert!(
            handle_dispute_created(&event, &pool, &unreachable)
                .await
                .is_retryable()
        );
        assert!(
            handle_dispute_created(&event, &pool, &None)
                .await
                .is_retryable()
        );
    }

    /// The Dispute object Stripe actually sends (API `2025-03-31.basil`
    /// and today): no `customer`, `charge` as an unexpanded id, and a
    /// `payment_intent`. There is no `charge_object` field.
    fn stripe_shaped_dispute(status: &str) -> serde_json::Value {
        serde_json::json!({
            "id": "dp_test_123",
            "object": "dispute",
            "amount": 500,
            "currency": "usd",
            "charge": "ch_test_123",
            "payment_intent": "pi_test_123",
            "reason": "fraudulent",
            "status": status
        })
    }

    #[tokio::test]
    #[ignore]
    async fn dispute_created_claws_back_a_stripe_shaped_dispute() {
        let pool = test_pool();
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(&pool, &cust).await;
        db::insert_credit_ledger(
            &pool,
            account_id,
            50_000_000,
            "subscription_renewal",
            &format!("evt_seed_{}", uuid::Uuid::new_v4()),
            None,
            None,
        )
        .await
        .unwrap();

        let event = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "charge.dispute.created",
            stripe_shaped_dispute("needs_response"),
        );
        let stripe = stub_charge_customer(Some(cust.clone())).await;
        let outcome = handle_dispute_created(&event, &pool, &stripe).await;

        assert!(!outcome.is_retryable());
        assert_eq!(get_balance(&pool, account_id).await, 45_000_000);
    }

    /// A won dispute gives back what its clawback took — read off the ledger
    /// by dispute id, through the real `created` → `closed` sequence.
    #[tokio::test]
    #[ignore]
    async fn dispute_closed_won_reverses_a_stripe_shaped_dispute() {
        let pool = test_pool();
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(&pool, &cust).await;
        db::insert_credit_ledger(
            &pool,
            account_id,
            50_000_000,
            "subscription_renewal",
            &format!("evt_seed_{}", uuid::Uuid::new_v4()),
            None,
            None,
        )
        .await
        .unwrap();
        let stripe = stub_charge_customer(Some(cust.clone())).await;

        let created = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "charge.dispute.created",
            stripe_shaped_dispute("needs_response"),
        );
        assert!(
            !handle_dispute_created(&created, &pool, &stripe)
                .await
                .is_retryable()
        );
        assert_eq!(get_balance(&pool, account_id).await, 45_000_000);

        let won = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "charge.dispute.closed",
            stripe_shaped_dispute("won"),
        );
        let outcome = handle_dispute_closed(&won, &pool, &stripe).await;
        assert!(!outcome.is_retryable());
        assert_eq!(get_balance(&pool, account_id).await, 50_000_000);

        // Redelivered, it gives nothing back twice.
        assert!(
            !handle_dispute_closed(&won, &pool, &stripe)
                .await
                .is_retryable()
        );
        assert_eq!(get_balance(&pool, account_id).await, 50_000_000);
    }

    /// A dispute this server never debited — its `created` event arrived
    /// before the handler could resolve the account, or was exhausted — has
    /// nothing to reverse. Crediting the payload's amount would mint balance
    /// the account never lost, and balance converts to unlinkable credentials.
    #[tokio::test]
    #[ignore]
    async fn a_won_dispute_with_no_recorded_clawback_credits_nothing() {
        let pool = test_pool();
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(&pool, &cust).await;

        let event = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "charge.dispute.closed",
            stripe_shaped_dispute("won"),
        );
        let stripe = stub_charge_customer(Some(cust.clone())).await;
        let outcome = handle_dispute_closed(&event, &pool, &stripe).await;

        assert!(!outcome.is_retryable(), "redelivery would not help");
        assert_eq!(get_balance(&pool, account_id).await, 0);
        assert_eq!(count_ledger_entries(&pool, account_id).await, 0);
    }

    #[tokio::test]
    #[ignore]
    async fn dispute_closed_lost_no_action() {
        let pool = test_pool();
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(&pool, &cust).await;

        let event = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "charge.dispute.closed",
            serde_json::json!({
                "customer": cust,
                "status": "lost",
                "amount": 500
            }),
        );

        let outcome = handle_dispute_closed(&event, &pool, &None).await;

        assert!(!outcome.is_retryable());
        assert_eq!(get_balance(&pool, account_id).await, 0);
        assert_eq!(count_ledger_entries(&pool, account_id).await, 0);
    }

    #[tokio::test]
    #[ignore]
    async fn duplicate_event_is_idempotent() {
        let pool = test_pool();
        let stripe = test_stripe();
        let product_id = test_product_id();
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(&pool, &cust).await;

        let event_id = format!("evt_{}", uuid::Uuid::new_v4());
        let period_end = now_secs() + 30 * 24 * 3600;
        let event = make_event(
            &event_id,
            "invoice.paid",
            serde_json::json!({
                "customer": cust,
                "billing_reason": "subscription_cycle",
                "lines": {
                    "data": [{
                        "pricing": {
                            "price_details": { "product": product_id }
                        },
                        "period": { "end": period_end }
                    }]
                }
            }),
        );

        let outcome = handle_invoice_paid(&event, &pool, &stripe).await;
        assert!(!outcome.is_retryable());
        let outcome = handle_invoice_paid(&event, &pool, &stripe).await;
        assert!(!outcome.is_retryable());

        assert_eq!(count_ledger_entries(&pool, account_id).await, 1);
    }

    #[tokio::test]
    #[ignore]
    async fn orphan_customer_no_panic() {
        let pool = test_pool();

        let event = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "invoice.paid",
            serde_json::json!({
                "customer": "cus_nonexistent_999",
                "billing_reason": "subscription_create",
                "lines": { "data": [] }
            }),
        );

        let outcome = handle_invoice_paid(&event, &pool, &None).await;
        assert!(!outcome.is_retryable());
    }
}
