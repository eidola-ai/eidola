//! Stripe webhook handling: signature verification and event dispatch.

use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use deadpool_postgres::Pool;
use hmac::{Hmac, Mac};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use sha2::Sha256;
use subtle::ConstantTimeEq;
use tracing::{error, info, warn};

use crate::db;
use crate::stripe::StripeClient;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;
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

    // Check staleness.
    let ts_secs: u64 = timestamp.parse().map_err(|_| "invalid timestamp")?;
    let event_time = UNIX_EPOCH + Duration::from_secs(ts_secs);
    if now.duration_since(event_time).unwrap_or(Duration::MAX) > MAX_SIGNATURE_AGE {
        return Err("stale timestamp");
    }

    // Compute expected HMAC.
    let mut mac =
        HmacSha256::new_from_slice(secret.as_bytes()).map_err(|_| "invalid webhook secret")?;
    mac.update(timestamp.as_bytes());
    mac.update(b".");
    mac.update(body);
    let expected = mac.finalize().into_bytes();

    // Decode the provided hex signature.
    let provided = hex::decode(signature).map_err(|_| "invalid hex signature")?;

    // Constant-time comparison.
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
    data: StripeEventData,
}

#[derive(serde::Deserialize)]
struct StripeEventData {
    object: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Webhook outcome
// ---------------------------------------------------------------------------

/// Signals whether a webhook handler failure is retryable.
///
/// `Handled` → 200 (event processed or permanently unprocessable).
/// `RetryableError` → 500 (transient failure; Stripe will retry).
enum WebhookOutcome {
    Handled,
    RetryableError,
}

impl WebhookOutcome {
    fn is_retryable(&self) -> bool {
        matches!(self, Self::RetryableError)
    }
}

// ---------------------------------------------------------------------------
// Main handler
// ---------------------------------------------------------------------------

/// POST /v1/webhooks/stripe
pub async fn handle_stripe_webhook(
    req: Request<Incoming>,
    pool: &Pool,
    stripe: &Option<StripeClient>,
    webhook_secret: &str,
) -> Response<BoxBody> {
    // Extract signature header before consuming body.
    let sig_header = match req
        .headers()
        .get("stripe-signature")
        .and_then(|v| v.to_str().ok())
    {
        Some(h) => h.to_string(),
        None => return bad_request("missing stripe-signature header"),
    };

    // Read raw body.
    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            error!("webhook: failed to read body: {}", e);
            return bad_request("failed to read request body");
        }
    };

    // Verify signature.
    if let Err(reason) = verify_signature(&sig_header, &body_bytes, webhook_secret) {
        warn!("webhook: signature verification failed: {}", reason);
        return bad_request(&format!("signature verification failed: {}", reason));
    }

    // Parse event.
    let event: StripeEvent = match serde_json::from_slice(&body_bytes) {
        Ok(e) => e,
        Err(e) => {
            error!("webhook: failed to parse event: {}", e);
            return ok_empty();
        }
    };

    info!("webhook: received {} ({})", event.event_type, event.id);

    // Dispatch by event type.
    let outcome = match event.event_type.as_str() {
        "checkout.session.completed" => {
            handle_checkout_completed(&event, pool, stripe).await
        }
        "invoice.paid" => handle_invoice_paid(&event, pool, stripe).await,
        "customer.subscription.deleted" => {
            info!(
                "webhook: subscription deleted (event {}), no ledger action",
                event.id
            );
            WebhookOutcome::Handled
        }
        "charge.refunded" => handle_charge_refunded(&event, pool).await,
        "charge.dispute.created" => handle_dispute_created(&event, pool).await,
        "charge.dispute.closed" => handle_dispute_closed(&event, pool).await,
        _ => {
            info!("webhook: ignoring unhandled event type {}", event.event_type);
            WebhookOutcome::Handled
        }
    };

    if outcome.is_retryable() {
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

    let mode = obj["mode"].as_str().unwrap_or("");
    if mode != "payment" {
        info!(
            "webhook: checkout mode={}, skipping (invoice.paid handles subscriptions)",
            mode
        );
        return WebhookOutcome::Handled;
    }

    let session_id = match obj["id"].as_str() {
        Some(id) => id,
        None => {
            error!("webhook: checkout.session.completed missing session id");
            return WebhookOutcome::Handled;
        }
    };

    let customer_id = match obj["customer"].as_str() {
        Some(c) => c,
        None => {
            error!("webhook: checkout.session.completed missing customer");
            return WebhookOutcome::Handled;
        }
    };

    // Look up account by stripe customer.
    let account = match db::get_account_by_stripe_customer(pool, customer_id).await {
        Ok(Some(a)) => a,
        Ok(None) => {
            warn!(
                "webhook: orphan customer {} in checkout.session.completed",
                customer_id
            );
            return WebhookOutcome::Handled;
        }
        Err(e) => {
            error!("webhook: db error looking up customer {}: {}", customer_id, e);
            return WebhookOutcome::RetryableError;
        }
    };

    // Fetch line items to find the product, then read credits from product metadata.
    let stripe = match stripe.as_ref() {
        Some(s) => s,
        None => {
            error!("webhook: stripe client not configured, cannot fetch line items");
            return WebhookOutcome::RetryableError;
        }
    };

    let line_items = match stripe.list_checkout_line_items(session_id).await {
        Ok(items) => items,
        Err(e) => {
            error!("webhook: failed to fetch line items for {}: {}", session_id, e);
            return WebhookOutcome::RetryableError;
        }
    };

    let product_id = match line_items.first().map(|item| item.price.product.as_str()) {
        Some(id) => id,
        None => {
            error!(
                "webhook: no line items for checkout session {}",
                session_id
            );
            return WebhookOutcome::Handled;
        }
    };

    let product = match stripe.get_product(product_id).await {
        Ok(p) => p,
        Err(e) => {
            error!(
                "webhook: failed to fetch product {} for checkout: {}",
                product_id, e
            );
            return WebhookOutcome::RetryableError;
        }
    };

    let credits = match product.metadata.get("credits").and_then(|s| s.parse::<i64>().ok()) {
        Some(c) => c,
        None => {
            error!(
                "webhook: no credits metadata on product {} for session {}",
                product_id, session_id
            );
            return WebhookOutcome::Handled;
        }
    };

    // Purchases expire after 1 year.
    let expires_at = SystemTime::now() + Duration::from_secs(365 * 24 * 3600);

    match db::insert_credit_ledger(pool, account.id, credits, "purchase", &event.id, Some(expires_at))
        .await
    {
        Ok(true) => info!(
            "webhook: credited {} to account {} (purchase, event {})",
            credits, account.id, event.id
        ),
        Ok(false) => info!("webhook: duplicate event {}, skipping", event.id),
        Err(e) => {
            error!("webhook: failed to insert ledger entry: {}", e);
            return WebhookOutcome::RetryableError;
        }
    }

    WebhookOutcome::Handled
}

async fn handle_invoice_paid(
    event: &StripeEvent,
    pool: &Pool,
    stripe: &Option<StripeClient>,
) -> WebhookOutcome {
    let obj = &event.data.object;

    let billing_reason = obj["billing_reason"].as_str().unwrap_or("");
    if billing_reason != "subscription_create" && billing_reason != "subscription_cycle" {
        info!(
            "webhook: invoice.paid billing_reason={}, skipping",
            billing_reason
        );
        return WebhookOutcome::Handled;
    }

    let customer_id = match obj["customer"].as_str() {
        Some(c) => c,
        None => {
            error!("webhook: invoice.paid missing customer");
            return WebhookOutcome::Handled;
        }
    };

    let account = match db::get_account_by_stripe_customer(pool, customer_id).await {
        Ok(Some(a)) => a,
        Ok(None) => {
            warn!(
                "webhook: orphan customer {} in invoice.paid",
                customer_id
            );
            return WebhookOutcome::Handled;
        }
        Err(e) => {
            error!("webhook: db error looking up customer {}: {}", customer_id, e);
            return WebhookOutcome::RetryableError;
        }
    };

    // Extract the product ID and period from the first line item.
    // Stripe's current API nests it under pricing.price_details.product.
    let lines = &obj["lines"]["data"];
    let first_line = match lines.as_array().and_then(|arr| arr.first()) {
        Some(line) => line,
        None => {
            error!(
                "webhook: no line items on invoice (event {})",
                event.id
            );
            return WebhookOutcome::Handled;
        }
    };

    let product_id = match first_line["pricing"]["price_details"]["product"].as_str() {
        Some(id) => id,
        None => {
            error!(
                "webhook: no product ID on invoice line item (event {})",
                event.id
            );
            return WebhookOutcome::Handled;
        }
    };

    // Fetch the product from Stripe to read credits metadata.
    let stripe = match stripe.as_ref() {
        Some(s) => s,
        None => {
            error!("webhook: stripe client not configured, cannot fetch product");
            return WebhookOutcome::RetryableError;
        }
    };

    let product = match stripe.get_product(product_id).await {
        Ok(p) => p,
        Err(e) => {
            error!(
                "webhook: failed to fetch product {} for invoice: {}",
                product_id, e
            );
            return WebhookOutcome::RetryableError;
        }
    };

    let credits = match product.metadata.get("credits").and_then(|s| s.parse::<i64>().ok()) {
        Some(c) => c,
        None => {
            error!(
                "webhook: no credits metadata on product {} (event {})",
                product_id, event.id
            );
            return WebhookOutcome::Handled;
        }
    };

    // expires_at = period end from the first line.
    let period_end = first_line["period"]["end"].as_i64();

    let expires_at = period_end.map(|ts| UNIX_EPOCH + Duration::from_secs(ts as u64));

    match db::insert_credit_ledger(
        pool,
        account.id,
        credits,
        "subscription_renewal",
        &event.id,
        expires_at,
    )
    .await
    {
        Ok(true) => info!(
            "webhook: credited {} to account {} (subscription_renewal, event {})",
            credits, account.id, event.id
        ),
        Ok(false) => info!("webhook: duplicate event {}, skipping", event.id),
        Err(e) => {
            error!("webhook: failed to insert ledger entry: {}", e);
            return WebhookOutcome::RetryableError;
        }
    }

    WebhookOutcome::Handled
}

async fn handle_charge_refunded(event: &StripeEvent, pool: &Pool) -> WebhookOutcome {
    let obj = &event.data.object;

    let customer_id = match obj["customer"].as_str() {
        Some(c) => c,
        None => {
            error!("webhook: charge.refunded missing customer");
            return WebhookOutcome::Handled;
        }
    };

    let account = match db::get_account_by_stripe_customer(pool, customer_id).await {
        Ok(Some(a)) => a,
        Ok(None) => {
            warn!(
                "webhook: orphan customer {} in charge.refunded",
                customer_id
            );
            return WebhookOutcome::Handled;
        }
        Err(e) => {
            error!("webhook: db error looking up customer {}: {}", customer_id, e);
            return WebhookOutcome::RetryableError;
        }
    };

    let amount_refunded = match obj["amount_refunded"].as_i64() {
        Some(a) => a,
        None => {
            error!("webhook: charge.refunded missing amount_refunded");
            return WebhookOutcome::Handled;
        }
    };

    // Convert cents to micro-dollars (1 cent = 10,000 micro-dollars).
    let delta = match cents_to_microdollars(amount_refunded).and_then(i64::checked_neg) {
        Some(d) => d,
        None => {
            error!("webhook: amount overflow in charge.refunded (event {})", event.id);
            return WebhookOutcome::Handled;
        }
    };

    match db::insert_credit_ledger(pool, account.id, delta, "refund", &event.id, None).await {
        Ok(true) => info!(
            "webhook: debited {} from account {} (refund, event {})",
            delta, account.id, event.id
        ),
        Ok(false) => info!("webhook: duplicate event {}, skipping", event.id),
        Err(e) => {
            error!("webhook: failed to insert ledger entry: {}", e);
            return WebhookOutcome::RetryableError;
        }
    }

    WebhookOutcome::Handled
}

async fn handle_dispute_created(event: &StripeEvent, pool: &Pool) -> WebhookOutcome {
    let obj = &event.data.object;

    let customer_id = match obj["charge"].as_str() {
        Some(_) => {
            // The dispute object has the customer on the charge, but Stripe
            // also embeds it directly on newer API versions.
            obj["customer"]
                .as_str()
                .or_else(|| obj["charge_object"]["customer"].as_str())
        }
        None => obj["customer"].as_str(),
    };

    let customer_id = match customer_id {
        Some(c) => c,
        None => {
            error!("webhook: charge.dispute.created missing customer");
            return WebhookOutcome::Handled;
        }
    };

    let account = match db::get_account_by_stripe_customer(pool, customer_id).await {
        Ok(Some(a)) => a,
        Ok(None) => {
            warn!(
                "webhook: orphan customer {} in charge.dispute.created",
                customer_id
            );
            return WebhookOutcome::Handled;
        }
        Err(e) => {
            error!("webhook: db error looking up customer {}: {}", customer_id, e);
            return WebhookOutcome::RetryableError;
        }
    };

    let amount = match obj["amount"].as_i64() {
        Some(a) => a,
        None => {
            error!("webhook: charge.dispute.created missing amount");
            return WebhookOutcome::Handled;
        }
    };

    // Convert cents to micro-dollars.
    let delta = match cents_to_microdollars(amount).and_then(i64::checked_neg) {
        Some(d) => d,
        None => {
            error!("webhook: amount overflow in charge.dispute.created (event {})", event.id);
            return WebhookOutcome::Handled;
        }
    };

    match db::insert_credit_ledger(pool, account.id, delta, "dispute_clawback", &event.id, None)
        .await
    {
        Ok(true) => info!(
            "webhook: debited {} from account {} (dispute_clawback, event {})",
            delta, account.id, event.id
        ),
        Ok(false) => info!("webhook: duplicate event {}, skipping", event.id),
        Err(e) => {
            error!("webhook: failed to insert ledger entry: {}", e);
            return WebhookOutcome::RetryableError;
        }
    }

    WebhookOutcome::Handled
}

async fn handle_dispute_closed(event: &StripeEvent, pool: &Pool) -> WebhookOutcome {
    let obj = &event.data.object;

    let status = obj["status"].as_str().unwrap_or("");
    if status != "won" {
        info!(
            "webhook: dispute closed with status={}, no reversal",
            status
        );
        return WebhookOutcome::Handled;
    }

    let customer_id = match obj["customer"].as_str() {
        Some(c) => c,
        None => {
            error!("webhook: charge.dispute.closed missing customer");
            return WebhookOutcome::Handled;
        }
    };

    let account = match db::get_account_by_stripe_customer(pool, customer_id).await {
        Ok(Some(a)) => a,
        Ok(None) => {
            warn!(
                "webhook: orphan customer {} in charge.dispute.closed",
                customer_id
            );
            return WebhookOutcome::Handled;
        }
        Err(e) => {
            error!("webhook: db error looking up customer {}: {}", customer_id, e);
            return WebhookOutcome::RetryableError;
        }
    };

    let amount = match obj["amount"].as_i64() {
        Some(a) => a,
        None => {
            error!("webhook: charge.dispute.closed missing amount");
            return WebhookOutcome::Handled;
        }
    };

    // Reversal: positive delta (credits returned).
    let delta = match cents_to_microdollars(amount) {
        Some(d) => d,
        None => {
            error!("webhook: amount overflow in charge.dispute.closed (event {})", event.id);
            return WebhookOutcome::Handled;
        }
    };

    match db::insert_credit_ledger(pool, account.id, delta, "dispute_reversal", &event.id, None)
        .await
    {
        Ok(true) => info!(
            "webhook: credited {} to account {} (dispute_reversal, event {})",
            delta, account.id, event.id
        ),
        Ok(false) => info!("webhook: duplicate event {}, skipping", event.id),
        Err(e) => {
            error!("webhook: failed to insert ledger entry: {}", e);
            return WebhookOutcome::RetryableError;
        }
    }

    WebhookOutcome::Handled
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert Stripe cents to micro-dollars with overflow check.
///
/// Returns `None` if the multiplication would overflow `i64`.
fn cents_to_microdollars(cents: i64) -> Option<i64> {
    cents.checked_mul(10_000)
}

fn internal_error() -> Response<BoxBody> {
    let body = serde_json::json!({"error": "internal error"}).to_string();
    Response::builder()
        .status(StatusCode::INTERNAL_SERVER_ERROR)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

fn ok_empty() -> Response<BoxBody> {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from("{}")).boxed())
        .unwrap()
}

fn bad_request(msg: &str) -> Response<BoxBody> {
    let body = serde_json::json!({"error": msg}).to_string();
    Response::builder()
        .status(StatusCode::BAD_REQUEST)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
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
        let ts = now_secs() - 600; // 10 minutes ago
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
        db::create_pool(&url).expect("failed to create test pool")
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
        std::env::var("STRIPE_API_KEY")
            .ok()
            .map(StripeClient::new)
    }

    /// Return the product ID to use in invoice tests.
    ///
    /// Set `TEST_STRIPE_PRODUCT_ID` to a Stripe product whose metadata
    /// contains `credits`.  The integration tests that exercise the full
    /// invoice flow need this (plus a running Stripe key) to fetch product
    /// metadata.
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

        let period_end = now_secs() + 30 * 24 * 3600; // 30 days from now
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

        // billing_reason="manual" exits before reaching Stripe.
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

        // Pre-credit the account.
        db::insert_credit_ledger(
            &pool,
            account_id,
            50_000_000,
            "subscription_renewal",
            &format!("evt_seed_{}", uuid::Uuid::new_v4()),
            None,
        )
        .await
        .unwrap();

        let event = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "charge.refunded",
            serde_json::json!({
                "customer": cust,
                "amount_refunded": 1000
            }),
        );

        let outcome = handle_charge_refunded(&event, &pool).await;

        assert!(!outcome.is_retryable());
        // 1000 cents = 10,000,000 micro-dollars deducted.
        assert_eq!(get_balance(&pool, account_id).await, 40_000_000);
        assert_eq!(count_ledger_entries(&pool, account_id).await, 2);
    }

    #[tokio::test]
    #[ignore]
    async fn dispute_created_claws_back() {
        let pool = test_pool();
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(&pool, &cust).await;

        // Pre-credit.
        db::insert_credit_ledger(
            &pool,
            account_id,
            50_000_000,
            "subscription_renewal",
            &format!("evt_seed_{}", uuid::Uuid::new_v4()),
            None,
        )
        .await
        .unwrap();

        let event = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "charge.dispute.created",
            serde_json::json!({
                "customer": cust,
                "amount": 500,
                "charge": "ch_test_123"
            }),
        );

        let outcome = handle_dispute_created(&event, &pool).await;

        assert!(!outcome.is_retryable());
        // 500 cents = 5,000,000 micro-dollars clawed back.
        assert_eq!(get_balance(&pool, account_id).await, 45_000_000);
        assert_eq!(count_ledger_entries(&pool, account_id).await, 2);
    }

    #[tokio::test]
    #[ignore]
    async fn dispute_closed_won_reverses() {
        let pool = test_pool();
        let cust = format!("cus_test_{}", uuid::Uuid::new_v4());
        let account_id = create_test_account(&pool, &cust).await;

        let event = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "charge.dispute.closed",
            serde_json::json!({
                "customer": cust,
                "status": "won",
                "amount": 500
            }),
        );

        let outcome = handle_dispute_closed(&event, &pool).await;

        assert!(!outcome.is_retryable());
        // 500 cents = 5,000,000 micro-dollars reversed (positive).
        assert_eq!(get_balance(&pool, account_id).await, 5_000_000);
        assert_eq!(count_ledger_entries(&pool, account_id).await, 1);
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

        let outcome = handle_dispute_closed(&event, &pool).await;

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

        // Orphan customer exits before reaching Stripe.
        let event = make_event(
            &format!("evt_{}", uuid::Uuid::new_v4()),
            "invoice.paid",
            serde_json::json!({
                "customer": "cus_nonexistent_999",
                "billing_reason": "subscription_create",
                "lines": { "data": [] }
            }),
        );

        // Should not panic — handler logs a warning and returns Handled.
        let outcome = handle_invoice_paid(&event, &pool, &None).await;
        assert!(!outcome.is_retryable());
    }
}
