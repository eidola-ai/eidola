//! Account management HTTP handlers.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use crate::AppState;
use crate::auth::BasicAuth;
use crate::db;
use crate::error::ServerError;
use crate::helpers::{system_time_to_iso, unix_to_iso};
use crate::stripe::{CheckoutParams, StripeClient};

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
    pub credential_key_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credential_credits: Option<i64>,
}

fn default_success_url() -> String {
    "https://eidola.ai/payment/success".to_string()
}

fn default_cancel_url() -> String {
    "https://eidola.ai/payment/cancel".to_string()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn require_stripe(stripe: &Option<StripeClient>) -> Result<&StripeClient, ServerError> {
    stripe
        .as_ref()
        .ok_or_else(|| ServerError::ServiceUnavailable("stripe is not configured".to_string()))
}

// Bring the trait into scope for fill_bytes.
use argon2::password_hash::rand_core::RngCore;

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /v1/account — create a new account.
#[utoipa::path(
    post,
    path = "/v1/account",
    tag = "Linked",
    responses(
        (status = 201, description = "Account created", body = CreateAccountResponse),
        (status = 500, description = "Internal error", body = crate::types::ErrorResponse)
    )
)]
pub async fn create_account(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ServerError> {
    use argon2::Argon2;
    use argon2::password_hash::{PasswordHasher, SaltString, rand_core::OsRng};

    let account_id = Uuid::new_v4();

    let mut secret_bytes = [0u8; 32];
    argon2::password_hash::rand_core::OsRng.fill_bytes(&mut secret_bytes);
    let secret = URL_SAFE_NO_PAD.encode(secret_bytes);

    let salt = SaltString::generate(&mut OsRng);
    let hash = Argon2::default()
        .hash_password(secret.as_bytes(), &salt)
        .map_err(|e| ServerError::Internal(format!("hash error: {}", e)))?
        .to_string();

    let created_at = db::insert_account(&state.db_pool, account_id, &hash).await?;

    let resp = CreateAccountResponse {
        account_id,
        secret,
        created_at: system_time_to_iso(created_at)?,
    };

    Ok((StatusCode::CREATED, Json(resp)))
}

/// GET /v1/account — retrieve account info (authenticated).
#[utoipa::path(
    get,
    path = "/v1/account",
    tag = "Linked",
    security(("basic" = [])),
    responses(
        (status = 200, description = "Account info", body = GetAccountResponse),
        (status = 401, description = "Invalid credentials", body = crate::types::ErrorResponse)
    )
)]
pub async fn get_account(
    BasicAuth(account_id): BasicAuth,
    State(state): State<AppState>,
) -> Result<Json<GetAccountResponse>, ServerError> {
    let account = db::get_account_by_id(&state.db_pool, account_id).await?;

    Ok(Json(GetAccountResponse {
        id: account.id,
        stripe_customer_id: account.stripe_customer_id,
        created_at: system_time_to_iso(account.created_at)?,
    }))
}

/// GET /v1/account/subscription — get subscription details (authenticated).
#[utoipa::path(
    get,
    path = "/v1/account/subscription",
    tag = "Linked",
    security(("basic" = [])),
    responses(
        (status = 200, description = "Subscription details", body = SubscriptionResponse),
        (status = 401, description = "Invalid credentials", body = crate::types::ErrorResponse),
        (status = 404, description = "No Stripe customer or no subscription", body = crate::types::ErrorResponse),
        (status = 503, description = "Stripe not configured", body = crate::types::ErrorResponse)
    )
)]
pub async fn get_subscription(
    BasicAuth(account_id): BasicAuth,
    State(state): State<AppState>,
) -> Result<Json<SubscriptionResponse>, ServerError> {
    let stripe = require_stripe(&state.stripe)?;
    let account = db::get_account_by_id(&state.db_pool, account_id).await?;

    let customer_id = account
        .stripe_customer_id
        .ok_or_else(|| ServerError::NotFound {
            message: "no_stripe_customer".to_string(),
        })?;

    let subscriptions = stripe.list_subscriptions(&customer_id).await?;

    let sub = subscriptions
        .into_iter()
        .next()
        .ok_or_else(|| ServerError::NotFound {
            message: "no_subscription".to_string(),
        })?;

    let management_url = stripe
        .create_portal_session(&customer_id, "https://eidola.ai")
        .await?;

    Ok(Json(SubscriptionResponse {
        id: sub.id,
        status: sub.status,
        current_period_end: sub.current_period_end.map(unix_to_iso),
        management_url,
    }))
}

/// GET /v1/prices — list available prices.
#[utoipa::path(
    get,
    path = "/v1/prices",
    tag = "Public",
    responses(
        (status = 200, description = "List of available prices", body = ListPricesResponse),
        (status = 503, description = "Stripe not configured", body = crate::types::ErrorResponse)
    )
)]
pub async fn list_prices(
    State(state): State<AppState>,
) -> Result<Json<ListPricesResponse>, ServerError> {
    let stripe = require_stripe(&state.stripe)?;
    let prices = stripe.list_prices().await?;

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

    Ok(Json(ListPricesResponse { data }))
}

/// POST /v1/account/checkout — create a checkout session (authenticated).
///
/// Accepts any active Stripe price ID. Automatically determines whether to
/// create a subscription or one-time payment checkout based on the price type.
/// For subscription prices, enforces the one-active-subscription constraint.
#[utoipa::path(
    post,
    path = "/v1/account/checkout",
    tag = "Linked",
    request_body = CheckoutRequest,
    security(("basic" = [])),
    responses(
        (status = 200, description = "Checkout session created", body = CheckoutUrlResponse),
        (status = 400, description = "Invalid request", body = crate::types::ErrorResponse),
        (status = 401, description = "Invalid credentials", body = crate::types::ErrorResponse),
        (status = 409, description = "Already subscribed", body = crate::types::ErrorResponse),
        (status = 503, description = "Stripe not configured", body = crate::types::ErrorResponse)
    )
)]
pub async fn create_checkout(
    BasicAuth(account_id): BasicAuth,
    State(state): State<AppState>,
    Json(checkout_req): Json<CheckoutRequest>,
) -> Result<Json<CheckoutUrlResponse>, ServerError> {
    let stripe = require_stripe(&state.stripe)?;

    if checkout_req.price_id.is_empty() {
        return Err(ServerError::BadRequest {
            message: "price_id must not be empty".to_string(),
        });
    }
    if checkout_req.success_url.is_empty() {
        return Err(ServerError::BadRequest {
            message: "success_url must not be empty".to_string(),
        });
    }
    if checkout_req.cancel_url.is_empty() {
        return Err(ServerError::BadRequest {
            message: "cancel_url must not be empty".to_string(),
        });
    }

    let customer_id = ensure_stripe_customer(&state.db_pool, stripe, account_id).await?;

    let price = stripe.get_price(&checkout_req.price_id).await?;

    let mode = if price.recurring.is_some() {
        let subs = stripe.list_subscriptions(&customer_id).await?;
        if subs
            .iter()
            .any(|s| s.status == "active" || s.status == "past_due" || s.status == "trialing")
        {
            return Err(ServerError::Conflict {
                message: "account already has an active subscription".to_string(),
            });
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

    let checkout_url = stripe.create_checkout_session(&params).await?;

    Ok(Json(CheckoutUrlResponse { checkout_url }))
}

/// Ensure the account has a Stripe customer ID, creating one if needed.
async fn ensure_stripe_customer(
    pool: &deadpool_postgres::Pool,
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

/// GET /v1/account/balances — get credit balance breakdown (authenticated).
#[utoipa::path(
    get,
    path = "/v1/account/balances",
    tag = "Linked",
    security(("basic" = [])),
    responses(
        (status = 200, description = "Balance breakdown", body = BalancesResponse),
        (status = 401, description = "Invalid credentials", body = crate::types::ErrorResponse)
    )
)]
pub async fn get_balances(
    BasicAuth(account_id): BasicAuth,
    State(state): State<AppState>,
) -> Result<Json<BalancesResponse>, ServerError> {
    let (total, pools) = db::get_balance_pools(&state.db_pool, account_id).await?;

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

    Ok(Json(BalancesResponse {
        available: total,
        pools,
    }))
}

/// GET /v1/account/ledger — get credit ledger entries (authenticated).
#[utoipa::path(
    get,
    path = "/v1/account/ledger",
    tag = "Linked",
    security(("basic" = [])),
    responses(
        (status = 200, description = "Ledger entries", body = LedgerResponse),
        (status = 401, description = "Invalid credentials", body = crate::types::ErrorResponse)
    )
)]
pub async fn get_ledger(
    BasicAuth(account_id): BasicAuth,
    State(state): State<AppState>,
) -> Result<Json<LedgerResponse>, ServerError> {
    let rows = db::get_ledger_entries(&state.db_pool, account_id).await?;

    let data: Vec<LedgerEntry> = rows
        .into_iter()
        .map(|e| {
            let is_credential = e.reason == "credential_issuance";
            LedgerEntry {
                id: e.id,
                delta: e.delta,
                reason: e.reason,
                expires_at: e.expires_at.and_then(|t| system_time_to_iso(t).ok()),
                created_at: system_time_to_iso(e.created_at).unwrap_or_default(),
                credential_key_id: if is_credential {
                    e.credential_key_id.map(|id| id.to_string())
                } else {
                    None
                },
                credential_credits: if is_credential {
                    e.credential_credits
                } else {
                    None
                },
            }
        })
        .collect();

    Ok(Json(LedgerResponse { data }))
}
