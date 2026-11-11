//! Eidola gateway library
//!
//! This module exposes the gateway's internal types for testing and reuse.

use std::sync::Arc;

use utoipa::OpenApi;
use utoipa_axum::{router::OpenApiRouter, routes};

pub mod account;
pub mod api_doc;
pub mod auth;
pub mod backend;
pub mod credentials;
pub mod db;
pub mod engine_router;
pub mod engine_trust;
pub mod error;
pub mod handlers;
pub mod helpers;
pub mod middleware;
pub mod models;
pub mod padding;
pub mod response;
pub mod stripe;
pub mod telemetry;
pub mod terms_feed;
pub mod types;
pub mod webhook;

/// Build a `rustls::ClientConfig` with Mozilla root CA certificates.
///
/// Used to configure reqwest clients when using `rustls-no-provider` (which
/// does not bundle any root certificates).
pub fn tls_config() -> rustls::ClientConfig {
    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth()
}

/// Largest request body a JSON route reads, in bytes; a longer one is
/// refused with 413. Shared with the node's limit
/// (`eidola_common::engine_protocol`), so a body accepted here is never one a
/// node refuses for its size.
pub const MAX_REQUEST_BODY_BYTES: usize =
    eidola_common::engine_protocol::GATEWAY_MAX_REQUEST_BODY_BYTES;

/// Largest request body read without a large-body permit, in bytes, and the
/// largest read at all from a request that does not declare its length or on
/// a route that is not JSON (the router's `DefaultBodyLimit`).
pub const UNGATED_REQUEST_BODY_BYTES: usize = 2 * 1024 * 1024;

/// Memory the gateway lets large request bodies (those past
/// [`UNGATED_REQUEST_BODY_BYTES`]) take while they are read and parsed, before
/// any credential is verified.
pub const LARGE_BODY_MEMORY_BUDGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// How many large request bodies are read and parsed at once: the budget over
/// the most one takes (`eidola_common::engine_deployment::read_slot_bytes` at
/// the largest body and the JSON value cap: the body, its text copied into
/// the parse, and the parse's nodes; measured for the node, which holds one
/// tree more than the gateway does).
pub fn large_body_permits() -> usize {
    let each = eidola_common::engine_deployment::read_slot_bytes(
        MAX_REQUEST_BODY_BYTES as u64,
        eidola_common::engine_protocol::MAX_REQUEST_JSON_VALUES as u64,
    );
    (LARGE_BODY_MEMORY_BUDGET_BYTES / each) as usize
}

/// Shared application state (Clone via inner Arc).
#[derive(Clone)]
pub struct AppState {
    inner: Arc<AppStateInner>,
}

pub struct AppStateInner {
    pub catalog: backend::Catalog,
    /// The router to Eidola-hosted engines; `None` when this build pins none.
    pub engines: Option<engine_router::EngineRouter>,
    pub db_pool: deadpool_postgres::Pool,
    pub stripe: Option<stripe::StripeClient>,
    pub stripe_webhook_secret: Option<String>,
    pub credential_master_key: [u8; 32],
    pub credential_key_cache: credentials::KeyCache,
    /// The spends this process is serving right now (`credentials::LiveSpends`).
    pub live_spends: Arc<credentials::LiveSpends>,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        catalog: backend::Catalog,
        engines: Option<engine_router::EngineRouter>,
        db_pool: deadpool_postgres::Pool,
        stripe: Option<stripe::StripeClient>,
        stripe_webhook_secret: Option<String>,
        credential_master_key: [u8; 32],
        credential_key_cache: credentials::KeyCache,
    ) -> Self {
        Self {
            inner: Arc::new(AppStateInner {
                catalog,
                engines,
                db_pool,
                stripe,
                stripe_webhook_secret,
                credential_master_key,
                credential_key_cache,
                live_spends: Arc::default(),
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
        .routes(routes!(account::get_terms))
        .routes(routes!(account::accept_terms))
        .routes(routes!(account::get_subscription))
        .routes(routes!(account::create_portal_session))
        .routes(routes!(account::create_checkout))
        .routes(routes!(account::get_balances))
        .routes(routes!(account::get_ledger))
        .routes(routes!(credentials::list_keys))
        .routes(routes!(credentials::issue_credentials))
        .routes(routes!(credentials::recover_refund))
        .routes(routes!(webhook::stripe_webhook))
        .layer(axum::extract::DefaultBodyLimit::max(
            UNGATED_REQUEST_BODY_BYTES,
        ))
}
