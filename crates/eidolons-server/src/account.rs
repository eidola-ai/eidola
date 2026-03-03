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
pub struct PurchaseRequest {
    pub price_id: String,
    #[serde(default = "default_success_url")]
    pub success_url: String,
    #[serde(default = "default_cancel_url")]
    pub cancel_url: String,
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

/// POST /v1/account/subscription — create a subscription checkout (authenticated).
pub async fn handle_create_subscription(
    pool: &Pool,
    stripe: &Option<StripeClient>,
    account_id: Uuid,
    subscription_price_id: &Option<String>,
) -> Response<BoxBody> {
    let stripe = match require_stripe(stripe) {
        Ok(s) => s,
        Err(e) => return error_response(&e),
    };

    let price_id = match subscription_price_id {
        Some(id) => id,
        None => {
            return error_response(&ServerError::ServiceUnavailable(
                "subscription price not configured".to_string(),
            ));
        }
    };

    // Ensure Stripe customer exists.
    let customer_id = match ensure_stripe_customer(pool, stripe, account_id).await {
        Ok(id) => id,
        Err(e) => return error_response(&e),
    };

    // Check for existing active/past_due subscriptions.
    match stripe.list_subscriptions(&customer_id).await {
        Ok(subs) => {
            if subs
                .iter()
                .any(|s| s.status == "active" || s.status == "past_due" || s.status == "trialing")
            {
                return error_response(&ServerError::Conflict {
                    message: "account already has an active subscription".to_string(),
                });
            }
        }
        Err(e) => return error_response(&e),
    }

    let params = CheckoutParams {
        customer_id: &customer_id,
        price_id,
        mode: "subscription",
        success_url: "https://eidolons.ai/payment/success",
        cancel_url: "https://eidolons.ai/payment/cancel",
    };

    let checkout_url = match stripe.create_checkout_session(&params).await {
        Ok(url) => url,
        Err(e) => return error_response(&e),
    };

    let resp = CheckoutUrlResponse { checkout_url };
    json_response(StatusCode::OK, &serde_json::to_string(&resp).unwrap())
}

/// POST /v1/account/purchase — create a one-time purchase checkout (authenticated).
pub async fn handle_create_purchase(
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

    let purchase_req: PurchaseRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return error_response(&ServerError::BadRequest {
                message: format!("invalid request body: {}", e),
            });
        }
    };

    // Ensure Stripe customer exists.
    let customer_id = match ensure_stripe_customer(pool, stripe, account_id).await {
        Ok(id) => id,
        Err(e) => return error_response(&e),
    };

    let params = CheckoutParams {
        customer_id: &customer_id,
        price_id: &purchase_req.price_id,
        mode: "payment",
        success_url: &purchase_req.success_url,
        cancel_url: &purchase_req.cancel_url,
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
