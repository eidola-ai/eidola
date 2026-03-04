//! Account management HTTP handlers and Basic auth.

use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use argon2::Argon2;
use argon2::password_hash::rand_core::OsRng;
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use bytes::Bytes;
use deadpool_postgres::Pool;
use http_body_util::BodyExt;
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use crate::db;
use crate::error::ServerError;
use crate::stripe::{CheckoutParams, StripeClient};

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

/// Authenticate a request using HTTP Basic auth against the account table.
///
/// The username is the account UUID, and the password is the credential secret.
pub async fn authenticate_account(
    req: &Request<Incoming>,
    pool: &Pool,
) -> Result<Uuid, ServerError> {
    let header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| ServerError::Unauthorized {
            message: "missing authorization header".to_string(),
        })?;

    let encoded = header
        .strip_prefix("Basic ")
        .ok_or_else(|| ServerError::Unauthorized {
            message: "expected Basic auth".to_string(),
        })?;

    let decoded = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|_| ServerError::Unauthorized {
            message: "invalid base64 in authorization header".to_string(),
        })?;

    let decoded_str = String::from_utf8(decoded).map_err(|_| ServerError::Unauthorized {
        message: "invalid utf-8 in authorization header".to_string(),
    })?;

    let (account_id_str, secret) =
        decoded_str
            .split_once(':')
            .ok_or_else(|| ServerError::Unauthorized {
                message: "invalid Basic auth format".to_string(),
            })?;

    let account_id = Uuid::parse_str(account_id_str).map_err(|_| ServerError::Unauthorized {
        message: "invalid account_id".to_string(),
    })?;

    let account = db::get_account_by_id(pool, account_id).await.map_err(|e| {
        // Map NotFound to Unauthorized to avoid leaking account existence.
        match e {
            ServerError::NotFound { .. } => ServerError::Unauthorized {
                message: "invalid credentials".to_string(),
            },
            other => other,
        }
    })?;

    let parsed_hash = PasswordHash::new(&account.secret_hash)
        .map_err(|_| ServerError::Internal("corrupt credential hash".to_string()))?;

    Argon2::default()
        .verify_password(secret.as_bytes(), &parsed_hash)
        .map_err(|_| ServerError::Unauthorized {
            message: "invalid credentials".to_string(),
        })?;

    Ok(account_id)
}

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

#[derive(Serialize, ToSchema)]
pub struct CreateAccountResponse {
    pub account_id: Uuid,
    pub secret: String,
    pub created_at: String,
}

#[derive(Serialize, ToSchema)]
pub struct GetAccountResponse {
    pub id: Uuid,
    pub stripe_customer_id: Option<String>,
    pub created_at: String,
}

#[derive(Serialize, ToSchema)]
pub struct SubscriptionResponse {
    pub id: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_period_end: Option<String>,
    pub management_url: String,
}

#[derive(Serialize, ToSchema)]
pub struct CheckoutUrlResponse {
    pub checkout_url: String,
}

#[derive(Deserialize, ToSchema)]
pub struct CheckoutRequest {
    pub price_id: String,
    #[serde(default = "default_success_url")]
    pub success_url: String,
    #[serde(default = "default_cancel_url")]
    pub cancel_url: String,
}

#[derive(Serialize, ToSchema)]
pub struct PriceResponse {
    pub id: String,
    pub product_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub product_description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit_amount: Option<i64>,
    pub currency: String,
    #[serde(rename = "type")]
    pub price_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recurring: Option<RecurringResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lookup_key: Option<String>,
    pub credits: i64,
}

#[derive(Serialize, ToSchema)]
pub struct RecurringResponse {
    pub interval: String,
    pub interval_count: i64,
}

#[derive(Serialize, ToSchema)]
pub struct ListPricesResponse {
    pub data: Vec<PriceResponse>,
}

#[derive(Serialize, ToSchema)]
pub struct BalancesResponse {
    pub available: i64,
    pub pools: Vec<BalancePool>,
}

#[derive(Serialize, ToSchema)]
pub struct BalancePool {
    pub amount: i64,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub struct LedgerResponse {
    pub data: Vec<LedgerEntry>,
    pub has_more: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub struct LedgerEntry {
    pub id: Uuid,
    pub delta: i64,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_epoch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_credits: Option<i64>,
}

fn default_success_url() -> String {
    "https://eidolons.ai/payment/success".to_string()
}

fn default_cancel_url() -> String {
    "https://eidolons.ai/payment/cancel".to_string()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn json_response(status: StatusCode, body: &str) -> Response<BoxBody> {
    use http_body_util::Full;

    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body.to_string())).boxed())
        .unwrap()
}

fn error_response(err: &ServerError) -> Response<BoxBody> {
    let status = err.status_code();
    let body = err.to_error_response();
    json_response(status, &serde_json::to_string(&body).unwrap())
}

/// Format a `SystemTime` as an ISO 8601 string (e.g. "2026-03-02T12:34:56Z").
fn system_time_to_iso(t: SystemTime) -> Result<String, ServerError> {
    let secs = system_time_to_unix_seconds(t).ok_or_else(|| {
        ServerError::Internal("timestamp out of supported range for i64 unix seconds".to_string())
    })?;
    Ok(unix_to_iso(secs))
}

/// Convert a `SystemTime` to signed unix seconds, rounded down to whole seconds.
fn system_time_to_unix_seconds(t: SystemTime) -> Option<i64> {
    match t.duration_since(UNIX_EPOCH) {
        Ok(d) => i64::try_from(d.as_secs()).ok(),
        Err(e) => {
            let d = e.duration();
            let secs = i64::try_from(d.as_secs()).ok()?;
            if d.subsec_nanos() == 0 {
                secs.checked_neg()
            } else {
                secs.checked_add(1)?.checked_neg()
            }
        }
    }
}

/// Format signed unix seconds as an ISO 8601 string.
fn unix_to_iso(secs: i64) -> String {
    let days_since_epoch = secs.div_euclid(86_400);
    let time_of_day = secs.rem_euclid(86_400) as u64;
    let (hour, min, sec) = (
        time_of_day / 3600,
        (time_of_day % 3600) / 60,
        time_of_day % 60,
    );

    // Convert days since 1970-01-01 to (year, month, day) using a civil calendar algorithm.
    // Ref: https://howardhinnant.github.io/date_algorithms.html#civil_from_days
    let z = days_since_epoch + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Encode a ledger pagination cursor from `(created_at, id)`.
fn encode_cursor(created_at: SystemTime, id: Uuid) -> Option<String> {
    let secs = system_time_to_unix_seconds(created_at)?;
    let plain = format!("{}:{}", secs, id);
    Some(URL_SAFE_NO_PAD.encode(plain.as_bytes()))
}

/// Decode a ledger pagination cursor into `(created_at, id)`.
fn decode_cursor(cursor: &str) -> Option<(SystemTime, Uuid)> {
    let bytes = URL_SAFE_NO_PAD.decode(cursor).ok()?;
    let plain = String::from_utf8(bytes).ok()?;
    let (secs_str, id_str) = plain.split_once(':')?;
    let secs: i64 = secs_str.parse().ok()?;
    let id = Uuid::parse_str(id_str).ok()?;
    let ts = if secs >= 0 {
        UNIX_EPOCH + std::time::Duration::from_secs(secs as u64)
    } else {
        UNIX_EPOCH.checked_sub(std::time::Duration::from_secs((-secs) as u64))?
    };
    Some((ts, id))
}

fn require_stripe(stripe: &Option<StripeClient>) -> Result<&StripeClient, ServerError> {
    stripe
        .as_ref()
        .ok_or_else(|| ServerError::ServiceUnavailable("stripe is not configured".to_string()))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /v1/account — create a new account.
pub async fn handle_create_account(pool: &Pool) -> Response<BoxBody> {
    let account_id = Uuid::new_v4();

    // Generate a random credential secret (32 bytes, base64url-no-pad encoded).
    let mut secret_bytes = [0u8; 32];
    argon2::password_hash::rand_core::OsRng.fill_bytes(&mut secret_bytes);
    let secret = URL_SAFE_NO_PAD.encode(secret_bytes);

    // Hash with Argon2id.
    let salt = SaltString::generate(&mut OsRng);
    let hash = match Argon2::default().hash_password(secret.as_bytes(), &salt) {
        Ok(h) => h.to_string(),
        Err(e) => return error_response(&ServerError::Internal(format!("hash error: {}", e))),
    };

    let created_at = match db::insert_account(pool, account_id, &hash).await {
        Ok(ts) => ts,
        Err(e) => return error_response(&e),
    };

    let resp = CreateAccountResponse {
        account_id,
        secret,
        created_at: match system_time_to_iso(created_at) {
            Ok(ts) => ts,
            Err(e) => return error_response(&e),
        },
    };

    json_response(StatusCode::CREATED, &serde_json::to_string(&resp).unwrap())
}

/// GET /v1/account — retrieve account info (authenticated).
pub async fn handle_get_account(pool: &Pool, account_id: Uuid) -> Response<BoxBody> {
    let account = match db::get_account_by_id(pool, account_id).await {
        Ok(a) => a,
        Err(e) => return error_response(&e),
    };

    let resp = GetAccountResponse {
        id: account.id,
        stripe_customer_id: account.stripe_customer_id,
        created_at: match system_time_to_iso(account.created_at) {
            Ok(ts) => ts,
            Err(e) => return error_response(&e),
        },
    };

    json_response(StatusCode::OK, &serde_json::to_string(&resp).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn unix_to_iso_handles_epoch_and_pre_epoch() {
        assert_eq!(unix_to_iso(0), "1970-01-01T00:00:00Z");
        assert_eq!(unix_to_iso(-1), "1969-12-31T23:59:59Z");
        assert_eq!(unix_to_iso(86_400), "1970-01-02T00:00:00Z");
    }

    #[test]
    fn system_time_to_iso_handles_pre_epoch() {
        let t = UNIX_EPOCH
            .checked_sub(Duration::from_secs(1))
            .expect("valid pre-epoch timestamp");
        assert_eq!(
            system_time_to_iso(t).expect("timestamp should format"),
            "1969-12-31T23:59:59Z"
        );
    }

    #[test]
    fn system_time_to_iso_floors_subsecond_pre_epoch() {
        let t = UNIX_EPOCH
            .checked_sub(Duration::from_millis(1))
            .expect("valid pre-epoch timestamp");
        assert_eq!(
            system_time_to_iso(t).expect("timestamp should format"),
            "1969-12-31T23:59:59Z"
        );
    }
}

/// GET /v1/account/subscription — get subscription details (authenticated).
pub async fn handle_get_subscription(
    pool: &Pool,
    stripe: &Option<StripeClient>,
    account_id: Uuid,
) -> Response<BoxBody> {
    let stripe = match require_stripe(stripe) {
        Ok(s) => s,
        Err(e) => return error_response(&e),
    };

    let account = match db::get_account_by_id(pool, account_id).await {
        Ok(a) => a,
        Err(e) => return error_response(&e),
    };

    let customer_id = match account.stripe_customer_id {
        Some(id) => id,
        None => {
            return error_response(&ServerError::NotFound {
                message: "no_stripe_customer".to_string(),
            });
        }
    };

    let subscriptions = match stripe.list_subscriptions(&customer_id).await {
        Ok(subs) => subs,
        Err(e) => return error_response(&e),
    };

    let sub = match subscriptions.into_iter().next() {
        Some(s) => s,
        None => {
            return error_response(&ServerError::NotFound {
                message: "no_subscription".to_string(),
            });
        }
    };

    let management_url = match stripe
        .create_portal_session(&customer_id, "https://eidolons.ai")
        .await
    {
        Ok(url) => url,
        Err(e) => return error_response(&e),
    };

    let resp = SubscriptionResponse {
        id: sub.id,
        status: sub.status,
        current_period_end: sub.current_period_end.map(unix_to_iso),
        management_url,
    };

    json_response(StatusCode::OK, &serde_json::to_string(&resp).unwrap())
}

/// GET /v1/prices — list available prices.
pub async fn handle_list_prices(stripe: &Option<StripeClient>) -> Response<BoxBody> {
    let stripe = match require_stripe(stripe) {
        Ok(s) => s,
        Err(e) => return error_response(&e),
    };

    let prices = match stripe.list_prices().await {
        Ok(p) => p,
        Err(e) => return error_response(&e),
    };

    let data = prices
        .into_iter()
        .filter_map(|p| {
            let credits: i64 = p.product.metadata.get("credits")?.parse().ok()?;
            Some(PriceResponse {
                id: p.id,
                product_name: p.product.name,
                product_description: p.product.description,
                unit_amount: p.unit_amount,
                currency: p.currency,
                price_type: p.price_type,
                recurring: p.recurring.map(|r| RecurringResponse {
                    interval: r.interval,
                    interval_count: r.interval_count,
                }),
                lookup_key: p.lookup_key,
                credits,
            })
        })
        .collect();

    let resp = ListPricesResponse { data };
    json_response(StatusCode::OK, &serde_json::to_string(&resp).unwrap())
}

/// POST /v1/account/checkout — create a checkout session (authenticated).
///
/// Accepts any active Stripe price ID. Automatically determines whether to
/// create a subscription or one-time payment checkout based on the price type.
/// For subscription prices, enforces the one-active-subscription constraint.
pub async fn handle_create_checkout(
    req: Request<Incoming>,
    pool: &Pool,
    stripe: &Option<StripeClient>,
    account_id: Uuid,
) -> Response<BoxBody> {
    let stripe = match require_stripe(stripe) {
        Ok(s) => s,
        Err(e) => return error_response(&e),
    };

    // Read and parse request body.
    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            return error_response(&ServerError::BadRequest {
                message: format!("failed to read request body: {}", e),
            });
        }
    };

    let checkout_req: CheckoutRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return error_response(&ServerError::BadRequest {
                message: format!("invalid request body: {}", e),
            });
        }
    };

    if checkout_req.price_id.is_empty() {
        return error_response(&ServerError::BadRequest {
            message: "price_id must not be empty".to_string(),
        });
    }
    if checkout_req.success_url.is_empty() {
        return error_response(&ServerError::BadRequest {
            message: "success_url must not be empty".to_string(),
        });
    }
    if checkout_req.cancel_url.is_empty() {
        return error_response(&ServerError::BadRequest {
            message: "cancel_url must not be empty".to_string(),
        });
    }

    // Ensure Stripe customer exists.
    let customer_id = match ensure_stripe_customer(pool, stripe, account_id).await {
        Ok(id) => id,
        Err(e) => return error_response(&e),
    };

    // Fetch the price to determine if it's recurring (subscription) or one-time.
    let price = match stripe.get_price(&checkout_req.price_id).await {
        Ok(p) => p,
        Err(e) => return error_response(&e),
    };

    let mode = if price.recurring.is_some() {
        // Enforce one active subscription per customer.
        match stripe.list_subscriptions(&customer_id).await {
            Ok(subs) => {
                if subs.iter().any(|s| {
                    s.status == "active" || s.status == "past_due" || s.status == "trialing"
                }) {
                    return error_response(&ServerError::Conflict {
                        message: "account already has an active subscription".to_string(),
                    });
                }
            }
            Err(e) => return error_response(&e),
        }
        "subscription"
    } else {
        "payment"
    };

    let account_id_str = account_id.to_string();
    let params = CheckoutParams {
        customer_id: &customer_id,
        price_id: &checkout_req.price_id,
        mode,
        success_url: &checkout_req.success_url,
        cancel_url: &checkout_req.cancel_url,
        client_reference_id: Some(&account_id_str),
    };

    let checkout_url = match stripe.create_checkout_session(&params).await {
        Ok(url) => url,
        Err(e) => return error_response(&e),
    };

    let resp = CheckoutUrlResponse { checkout_url };
    json_response(StatusCode::OK, &serde_json::to_string(&resp).unwrap())
}

/// Ensure the account has a Stripe customer ID, creating one if needed.
async fn ensure_stripe_customer(
    pool: &Pool,
    stripe: &StripeClient,
    account_id: Uuid,
) -> Result<String, ServerError> {
    let account = db::get_account_by_id(pool, account_id).await?;

    if let Some(customer_id) = account.stripe_customer_id {
        return Ok(customer_id);
    }

    let customer_id = stripe.create_customer(account_id).await?;
    db::set_stripe_customer_id(pool, account_id, &customer_id).await
}

// Bring the trait into scope for fill_bytes.
use argon2::password_hash::rand_core::RngCore;

/// GET /v1/account/balances — get credit balance breakdown (authenticated).
pub async fn handle_get_balances(pool: &Pool, account_id: Uuid) -> Response<BoxBody> {
    let (total, pools) = match db::get_balance_pools(pool, account_id).await {
        Ok(r) => r,
        Err(e) => return error_response(&e),
    };

    let pools = pools
        .into_iter()
        .map(|p| {
            let source = match p.source_reason.as_deref() {
                Some("subscription_renewal") => "subscription",
                Some("purchase") => "purchase",
                _ => "other",
            };
            BalancePool {
                amount: p.pool_amount,
                source: source.to_string(),
                expires_at: p.expires_at.and_then(|t| system_time_to_iso(t).ok()),
            }
        })
        .collect();

    let resp = BalancesResponse {
        available: total,
        pools,
    };

    json_response(StatusCode::OK, &serde_json::to_string(&resp).unwrap())
}

/// GET /v1/account/ledger — get credit ledger entries (authenticated).
pub async fn handle_get_ledger(
    req: &Request<Incoming>,
    pool: &Pool,
    account_id: Uuid,
) -> Response<BoxBody> {
    // Parse query parameters.
    let query_str = req.uri().query().unwrap_or("");
    let params: Vec<(String, String)> = url_decode_pairs(query_str);

    let reasons: Option<Vec<String>> = params
        .iter()
        .find(|(k, _)| k == "reason")
        .map(|(_, v)| v.split(',').map(|s| s.trim().to_string()).collect());

    let cursor = params
        .iter()
        .find(|(k, _)| k == "cursor")
        .and_then(|(_, v)| decode_cursor(v));

    let limit: i64 = params
        .iter()
        .find(|(k, _)| k == "limit")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(50_i64)
        .clamp(1, 200);

    let rows =
        match db::get_ledger_entries(pool, account_id, reasons.as_deref(), cursor, limit).await {
            Ok(e) => e,
            Err(e) => return error_response(&e),
        };

    let has_more = rows.len() as i64 > limit;
    let rows: Vec<_> = if has_more {
        rows.into_iter().take(limit as usize).collect()
    } else {
        rows
    };

    // Encode cursor from the last entry if there are more pages.
    let next_cursor = if has_more {
        rows.last().and_then(|e| encode_cursor(e.created_at, e.id))
    } else {
        None
    };

    let data: Vec<LedgerEntry> = rows
        .into_iter()
        .map(|e| {
            let is_act = e.reason == "act_issuance";
            LedgerEntry {
                id: e.id,
                delta: e.delta,
                reason: e.reason,
                expires_at: e.expires_at.and_then(|t| system_time_to_iso(t).ok()),
                created_at: system_time_to_iso(e.created_at).unwrap_or_default(),
                token_epoch: if is_act { e.token_epoch } else { None },
                token_credits: if is_act { e.token_credits } else { None },
            }
        })
        .collect();

    let resp = LedgerResponse {
        data,
        has_more,
        cursor: next_cursor,
    };
    json_response(StatusCode::OK, &serde_json::to_string(&resp).unwrap())
}

/// Parse URL-encoded query string into key-value pairs.
fn url_decode_pairs(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|s| !s.is_empty())
        .filter_map(|pair| {
            let (k, v) = pair.split_once('=')?;
            Some((url_decode(k), url_decode(v)))
        })
        .collect()
}

/// Minimal URL percent-decoding.
fn url_decode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.bytes();
    while let Some(b) = chars.next() {
        if b == b'%' {
            let hi = chars.next();
            let lo = chars.next();
            if let (Some(hi), Some(lo)) = (hi, lo) {
                let hex = [hi, lo];
                if let Ok(decoded) = u8::from_str_radix(std::str::from_utf8(&hex).unwrap_or(""), 16)
                {
                    result.push(decoded as char);
                    continue;
                }
            }
            result.push('%');
        } else if b == b'+' {
            result.push(' ');
        } else {
            result.push(b as char);
        }
    }
    result
}
