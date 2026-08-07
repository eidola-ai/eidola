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

/// A document whose current version must be accepted before purchases or
/// credential issuance. `sha256` identifies the exact text (the markdown
/// source, retrievable at `{url}source.md`), so acceptance binds to
/// precise words rather than a mutable URL; `version` is the document's
/// monotonically increasing front-matter version, which orders releases
/// (accepting version N satisfies any requirement ≤ N).
#[derive(Clone, Serialize, ToSchema)]
pub struct RequiredDocument {
    /// `terms_of_service` or `privacy_policy`.
    pub document: String,
    /// Monotonically increasing document version.
    pub version: i64,
    /// Where the current text is published.
    pub url: String,
    /// Hex-encoded SHA-256 of the exact published document text.
    pub sha256: String,
}

/// The documents (and versions) whose acceptance the server currently
/// requires. Empty when no acceptance gate is configured.
#[derive(Serialize, ToSchema)]
pub struct TermsResponse {
    pub documents: Vec<RequiredDocument>,
}

/// Acceptance of one currently required document version.
#[derive(Deserialize, ToSchema)]
pub struct AcceptTermsRequest {
    /// `terms_of_service` or `privacy_policy`.
    pub document: String,
    /// Hex-encoded SHA-256 of the document text being accepted; must match
    /// the currently required version from `GET /v1/terms`.
    pub sha256: String,
}

/// A purchasable plan.
#[derive(Serialize, ToSchema)]
pub struct PriceResponse {
    pub id: String,
    pub product_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub product_description: Option<String>,
    /// Purchase price in the minor unit of `currency` (e.g. cents for USD).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit_amount: Option<i64>,
    pub currency: String,
    #[serde(rename = "type")]
    pub price_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recurring: Option<RecurringResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lookup_key: Option<String>,
    /// Credits granted by this plan, denominated in micro-USD
    /// (1 credit = $0.000001). Subscription credits expire at the end of
    /// each billing period; one-time purchase credits expire one year
    /// after purchase.
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

/// Credit balance breakdown. All amounts are credits, denominated in
/// micro-USD (1 credit = $0.000001).
#[derive(Serialize, ToSchema)]
pub struct BalancesResponse {
    /// Total spendable credits (expired pools excluded).
    pub available: i64,
    /// Per-pool breakdown, earliest expiry first.
    pub pools: Vec<BalancePool>,
}

/// One balance pool: credits sharing an origin and expiration.
#[derive(Serialize, ToSchema)]
pub struct BalancePool {
    /// Credits remaining in this pool (micro-USD denominated).
    pub amount: i64,
    /// Where the pool came from: `subscription`, `purchase`, or `other`.
    pub source: String,
    /// ISO-8601 instant at which these credits expire; absent for credits
    /// that never expire.
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

/// The required documents the account has not accepted at (or above) their
/// required versions. The `>=` comparison is what makes brief cross-instance
/// requirement skew invisible to users: accepting version 6 satisfies an
/// instance still requiring version 5. Pure so the gate logic is
/// unit-testable without a database.
pub(crate) fn missing_documents<'a>(
    required: &'a [db::RequiredDocumentRow],
    accepted_versions: &[(String, i64)],
) -> Vec<&'a str> {
    required
        .iter()
        .filter(|d| {
            !accepted_versions
                .iter()
                .any(|(doc, version)| *doc == d.document && *version >= d.version)
        })
        .map(|d| d.document.as_str())
        .collect()
}

/// Gate for purchases and credential issuance: the account must have
/// accepted every required document at (or above) the cluster-wide
/// required version (the `required_document` table, advanced by the
/// terms-feed poller and/or the startup env seed). A no-op when the table
/// is empty (gate disabled).
pub(crate) async fn ensure_terms_accepted(
    state: &AppState,
    account_id: Uuid,
) -> Result<(), ServerError> {
    let required = db::get_required_documents(&state.db_pool).await?;
    if required.is_empty() {
        return Ok(());
    }
    let accepted = db::get_accepted_versions(&state.db_pool, account_id).await?;
    let missing = missing_documents(&required, &accepted);
    if missing.is_empty() {
        Ok(())
    } else {
        Err(ServerError::TermsAcceptanceRequired {
            message: format!(
                "acceptance of the current {} is required — fetch GET /v1/terms \
                 and record acceptance via POST /v1/account/terms",
                missing.join(" and ")
            ),
        })
    }
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

/// GET /v1/terms — the document versions whose acceptance is required.
///
/// Public, and necessarily so: a client fetches this *before* it has an
/// account, to learn what it must accept. It takes no credential and returns
/// no account-specific data. (Accepting them is `POST /v1/account/terms`,
/// which is on the linked surface.)
#[utoipa::path(
    get,
    path = "/v1/terms",
    tag = "Public",
    responses(
        (status = 200, description = "Currently required document versions", body = TermsResponse)
    )
)]
pub async fn get_terms(State(state): State<AppState>) -> Result<Json<TermsResponse>, ServerError> {
    let documents = db::get_required_documents(&state.db_pool)
        .await?
        .into_iter()
        .map(|d| RequiredDocument {
            document: d.document,
            version: d.version,
            url: d.url,
            sha256: d.sha256,
        })
        .collect();
    Ok(Json(TermsResponse { documents }))
}

/// POST /v1/account/terms — record acceptance of a current document version
/// (authenticated).
#[utoipa::path(
    post,
    path = "/v1/account/terms",
    tag = "Linked",
    request_body = AcceptTermsRequest,
    security(("basic" = [])),
    responses(
        (status = 204, description = "Acceptance recorded"),
        (status = 401, description = "Invalid credentials", body = crate::types::ErrorResponse),
        (status = 409, description = "Not the currently required version", body = crate::types::ErrorResponse)
    )
)]
pub async fn accept_terms(
    BasicAuth(account_id): BasicAuth,
    State(state): State<AppState>,
    Json(req): Json<AcceptTermsRequest>,
) -> Result<StatusCode, ServerError> {
    let sha256 = req.sha256.to_lowercase();
    let required = db::get_required_documents(&state.db_pool).await?;
    let Some(current) = required
        .iter()
        .find(|d| d.document == req.document && d.sha256 == sha256)
    else {
        return Err(ServerError::Conflict {
            message: format!(
                "{} with hash {} is not the currently required version — \
                 fetch GET /v1/terms for the current documents",
                req.document, req.sha256
            ),
        });
    };

    db::insert_acceptance(
        &state.db_pool,
        account_id,
        &current.document,
        current.version,
        &current.sha256,
    )
    .await?;
    Ok(StatusCode::NO_CONTENT)
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
        (status = 428, description = "Terms acceptance required", body = crate::types::ErrorResponse),
        (status = 503, description = "Stripe not configured", body = crate::types::ErrorResponse)
    )
)]
pub async fn create_checkout(
    BasicAuth(account_id): BasicAuth,
    State(state): State<AppState>,
    Json(checkout_req): Json<CheckoutRequest>,
) -> Result<Json<CheckoutUrlResponse>, ServerError> {
    let stripe = require_stripe(&state.stripe)?;

    ensure_terms_accepted(&state, account_id).await?;

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

    // Conspicuous expiry disclosure at the point of purchase: `submit_note`
    // renders next to Checkout's pay button; `description` lands on the
    // PaymentIntent / Subscription and thus on Stripe's email receipts and
    // invoices. Must stay consistent with the published terms
    // (www/pages/terms.md) and the webhook expiry logic.
    let (submit_note, description) = if mode == "subscription" {
        (
            "Credits granted each billing period expire at the end of that period. \
             Unused, unexpired credits are refundable on request. Details: eidola.ai/terms",
            "Eidola subscription — each period's credits expire at the end of that \
             billing period (see eidola.ai/terms)",
        )
    } else {
        (
            "Credits expire one year after purchase. Unused, unexpired credits are \
             refundable on request. Details: eidola.ai/terms",
            "Eidola credits — expire one year after purchase (see eidola.ai/terms)",
        )
    };

    let account_id_str = account_id.to_string();
    let params = CheckoutParams {
        customer_id: &customer_id,
        price_id: &checkout_req.price_id,
        mode,
        success_url: &checkout_req.success_url,
        cancel_url: &checkout_req.cancel_url,
        client_reference_id: Some(&account_id_str),
        submit_note: Some(submit_note),
        description: Some(description),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(document: &str, version: i64) -> db::RequiredDocumentRow {
        db::RequiredDocumentRow {
            document: document.to_string(),
            version,
            sha256: "a".repeat(64),
            url: format!("https://www.eidola.ai/{document}/"),
        }
    }

    #[test]
    fn no_required_documents_means_nothing_missing() {
        assert!(missing_documents(&[], &[]).is_empty());
    }

    #[test]
    fn unaccepted_documents_are_missing() {
        let required = [doc("terms_of_service", 1)];
        assert_eq!(missing_documents(&required, &[]), vec!["terms_of_service"]);
    }

    #[test]
    fn acceptance_of_current_version_satisfies() {
        let required = [doc("terms_of_service", 2)];
        let accepted = [("terms_of_service".to_string(), 2)];
        assert!(missing_documents(&required, &accepted).is_empty());
    }

    #[test]
    fn acceptance_of_newer_version_satisfies_a_lagging_requirement() {
        // The cross-instance skew case: the user accepted version 6 via a
        // fresh instance; this instance still requires version 5.
        let required = [doc("terms_of_service", 5)];
        let accepted = [("terms_of_service".to_string(), 6)];
        assert!(missing_documents(&required, &accepted).is_empty());
    }

    #[test]
    fn acceptance_of_old_version_does_not_satisfy() {
        let required = [doc("terms_of_service", 3)];
        let accepted = [("terms_of_service".to_string(), 2)];
        assert_eq!(
            missing_documents(&required, &accepted),
            vec!["terms_of_service"]
        );
    }

    #[test]
    fn each_document_is_gated_independently() {
        let required = [doc("terms_of_service", 1), doc("privacy_policy", 1)];
        let accepted = [("terms_of_service".to_string(), 1)];
        assert_eq!(
            missing_documents(&required, &accepted),
            vec!["privacy_policy"]
        );
    }
}
