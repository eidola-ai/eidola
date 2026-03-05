//! Eidolons Server Library
//!
//! This module exposes the server's internal types for testing and reuse.

use std::sync::Arc;

use utoipa::OpenApi;
use utoipa_axum::{router::OpenApiRouter, routes};

pub mod account;
pub mod api_doc;
pub mod attestation;
pub mod auth;
pub mod backend;
pub mod db;
pub mod error;
pub mod handlers;
pub mod helpers;
pub mod response;
pub mod stripe;
pub mod tokens;
pub mod types;
pub mod webhook;

/// Shared application state (Clone via inner Arc).
#[derive(Clone)]
pub struct AppState {
    inner: Arc<AppStateInner>,
}

pub struct AppStateInner {
    pub backend: backend::RedPillBackend,
    pub validator: auth::AnyValidator,
    pub attestation: attestation::AttestationClient,
    pub db_pool: deadpool_postgres::Pool,
    pub stripe: Option<stripe::StripeClient>,
    pub stripe_webhook_secret: Option<String>,
    pub act_master_key: Option<[u8; 32]>,
    pub act_key_cache: tokens::KeyCache,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: backend::RedPillBackend,
        validator: auth::AnyValidator,
        attestation: attestation::AttestationClient,
        db_pool: deadpool_postgres::Pool,
        stripe: Option<stripe::StripeClient>,
        stripe_webhook_secret: Option<String>,
        act_master_key: Option<[u8; 32]>,
        act_key_cache: tokens::KeyCache,
    ) -> Self {
        Self {
            inner: Arc::new(AppStateInner {
                backend,
                validator,
                attestation,
                db_pool,
                stripe,
                stripe_webhook_secret,
                act_master_key,
                act_key_cache,
            }),
        }
    }
}

impl std::ops::Deref for AppState {
    type Target = AppStateInner;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Build the `OpenApiRouter` with all routes registered.
///
/// Used by both the server (`main.rs`) and the OpenAPI spec generator tool.
pub fn build_router() -> OpenApiRouter<AppState> {
    OpenApiRouter::with_openapi(api_doc::ApiDoc::openapi())
        .routes(routes!(handlers::health))
        .routes(routes!(handlers::list_models))
        .routes(routes!(handlers::chat_completions))
        .routes(routes!(account::list_prices))
        .routes(routes!(account::create_account, account::get_account))
        .routes(routes!(account::get_subscription))
        .routes(routes!(account::create_checkout))
        .routes(routes!(account::get_balances))
        .routes(routes!(account::get_ledger))
        .routes(routes!(tokens::list_keys))
        .routes(routes!(tokens::issue_tokens))
        .routes(routes!(webhook::stripe_webhook))
}
