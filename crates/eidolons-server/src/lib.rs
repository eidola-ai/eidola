//! Eidolons Server Library
//!
//! This module exposes the server's internal types for testing and reuse.

use std::sync::Arc;

use utoipa::OpenApi;
use utoipa_axum::{router::OpenApiRouter, routes};

pub mod account;
pub mod api_doc;
pub mod auth;
pub mod backend;
pub mod credentials;
pub mod db;
pub mod error;
pub mod handlers;
pub mod helpers;
pub mod measurements;
pub mod response;
pub mod stripe;
pub mod tls;
pub mod types;
pub mod webhook;

/// Shared application state (Clone via inner Arc).
#[derive(Clone)]
pub struct AppState {
    inner: Arc<AppStateInner>,
}

pub struct AppStateInner {
    pub backend: backend::TinfoilBackend,
    pub db_pool: deadpool_postgres::Pool,
    pub stripe: Option<stripe::StripeClient>,
    pub stripe_webhook_secret: Option<String>,
    pub credential_master_key: [u8; 32],
    pub credential_key_cache: credentials::KeyCache,
    pub epoch_config: helpers::EpochConfig,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: backend::TinfoilBackend,
        db_pool: deadpool_postgres::Pool,
        stripe: Option<stripe::StripeClient>,
        stripe_webhook_secret: Option<String>,
        credential_master_key: [u8; 32],
        credential_key_cache: credentials::KeyCache,
        epoch_config: helpers::EpochConfig,
    ) -> Self {
        Self {
            inner: Arc::new(AppStateInner {
                backend,
                db_pool,
                stripe,
                stripe_webhook_secret,
                credential_master_key,
                credential_key_cache,
                epoch_config,
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
        .routes(routes!(credentials::list_keys))
        .routes(routes!(credentials::issue_credentials))
        .routes(routes!(webhook::stripe_webhook))
}
