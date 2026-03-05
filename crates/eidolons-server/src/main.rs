//! Eidolons Server - A privacy-transparent AI proxy.
//!
//! This server accepts requests in OpenAI Chat Completions API format and
//! proxies them to upstream AI providers via RedPill.ai, enriching responses
//! with inline privacy metadata and cryptographic verification.

use std::net::SocketAddr;

use axum::http::StatusCode;
use tokio::net::TcpListener;
use tracing::{error, info, warn};

use eidolons_server::AppState;
use eidolons_server::attestation::AttestationClient;
use eidolons_server::auth::{AnyValidator, NoopValidator};
use eidolons_server::backend::RedPillBackend;
use eidolons_server::stripe::StripeClient;
use eidolons_server::credentials;

/// Server configuration.
struct Config {
    bind_addr: SocketAddr,
    redpill_api_key: String,
    redpill_base_url: Option<String>,
    auth_mode: String,
    database_url: String,
    stripe_api_key: Option<String>,
    stripe_webhook_secret: Option<String>,
    credential_master_key: Option<[u8; 32]>,
}

impl Config {
    fn from_env() -> Result<Self, String> {
        let bind_addr = std::env::var("BIND_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:8080".to_string())
            .parse()
            .map_err(|e| format!("invalid BIND_ADDR: {}", e))?;

        let redpill_api_key = std::env::var("REDPILL_API_KEY")
            .map_err(|_| "REDPILL_API_KEY environment variable is required")?;

        let redpill_base_url = std::env::var("REDPILL_BASE_URL").ok();

        let auth_mode = std::env::var("AUTH_MODE").unwrap_or_else(|_| "none".to_string());

        let database_url = std::env::var("DATABASE_URL")
            .map_err(|_| "DATABASE_URL environment variable is required")?;

        let stripe_api_key = std::env::var("STRIPE_API_KEY")
            .ok()
            .filter(|s| !s.is_empty());

        let stripe_webhook_secret = std::env::var("STRIPE_WEBHOOK_SECRET")
            .ok()
            .filter(|s| !s.is_empty());

        let credential_master_key = std::env::var("CREDENTIAL_MASTER_KEY")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| {
                let bytes = hex::decode(&s)
                    .map_err(|_| "CREDENTIAL_MASTER_KEY must be valid hex".to_string())?;
                if bytes.len() != 32 {
                    return Err(
                        "CREDENTIAL_MASTER_KEY must be exactly 32 bytes (64 hex chars)".to_string()
                    );
                }
                let mut key = [0u8; 32];
                key.copy_from_slice(&bytes);
                Ok(key)
            })
            .transpose()?;

        Ok(Config {
            bind_addr,
            redpill_api_key,
            redpill_base_url,
            auth_mode,
            database_url,
            stripe_api_key,
            stripe_webhook_secret,
            credential_master_key,
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Install the pure-Rust crypto provider for TLS (must be done before any TLS operations)
    rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider())
        .expect("failed to install rustls crypto provider");

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("eidolons_server=info".parse().unwrap())
                .add_directive("hyper=warn".parse().unwrap()),
        )
        .init();

    // Load configuration
    let config = Config::from_env().map_err(|e| {
        error!("Configuration error: {}", e);
        e
    })?;

    info!("Starting Eidolons server on {}", config.bind_addr);
    info!("Auth mode: {}", config.auth_mode);

    // Build the token validator
    let validator = match config.auth_mode.as_str() {
        "none" => {
            warn!("Running with no authentication (development mode)");
            AnyValidator::Noop(NoopValidator)
        }
        other => {
            return Err(format!("unknown AUTH_MODE: {} (expected: none)", other).into());
        }
    };

    // Create database connection pool
    let db_pool = eidolons_server::db::create_pool(&config.database_url).map_err(|e| {
        error!("Database pool error: {}", e);
        e
    })?;

    // Create Stripe client (optional)
    let stripe = config.stripe_api_key.map(StripeClient::new);
    if stripe.is_none() {
        warn!("STRIPE_API_KEY not set — account billing endpoints will return 503");
    }
    if config.stripe_webhook_secret.is_none() {
        warn!("STRIPE_WEBHOOK_SECRET not set — webhook endpoint will return 503");
    }

    // Credential key cache
    let credential_key_cache: credentials::KeyCache = Default::default();
    if config.credential_master_key.is_none() {
        warn!("CREDENTIAL_MASTER_KEY not set — credential issuance endpoints will return 503");
    }

    // Create shared state
    let state = AppState::new(
        RedPillBackend::new(
            config.redpill_api_key.clone(),
            config.redpill_base_url.clone(),
        ),
        validator,
        AttestationClient::new(config.redpill_api_key, config.redpill_base_url),
        db_pool,
        stripe,
        config.stripe_webhook_secret,
        config.credential_master_key,
        credential_key_cache,
    );

    // Pre-warm the current epoch's issuer key if master key is configured.
    if let Some(ref mk) = state.credential_master_key {
        match credentials::ensure_current_epoch_key(&state.credential_key_cache, mk, &state.db_pool).await {
            Ok(epoch) => info!("Issuer key ready for epoch {}", epoch),
            Err(e) => warn!("Failed to pre-warm issuer key: {}", e),
        }
    }

    // Build the router with OpenAPI integration
    let (router, api) = eidolons_server::build_router()
        .with_state(state)
        .split_for_parts();

    // Store the generated OpenAPI spec for the /openapi.json endpoint
    let api_json = api.to_json().expect("OpenAPI spec serialization failed");
    let app = router.route(
        "/openapi.json",
        axum::routing::get(move || {
            let spec = api_json.clone();
            async move { (StatusCode::OK, [("content-type", "application/json")], spec) }
        }),
    );

    // Bind and serve
    let listener = TcpListener::bind(config.bind_addr).await?;
    info!("Listening on http://{}", config.bind_addr);
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_openapi_spec_generation() {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());

        // Build the full router to capture paths from handler annotations.
        let (_, spec) = eidolons_server::build_router().split_for_parts();

        // Verify basic info
        assert_eq!(spec.info.title, "Eidolons API");
        assert_eq!(spec.info.version, "0.1.0");

        // Verify paths exist
        assert!(
            spec.paths.paths.contains_key("/health"),
            "missing /health path"
        );
        assert!(
            spec.paths.paths.contains_key("/v1/models"),
            "missing /v1/models path"
        );
        assert!(
            spec.paths.paths.contains_key("/v1/chat/completions"),
            "missing /v1/chat/completions path"
        );
        assert!(
            spec.paths.paths.contains_key("/v1/account/balances"),
            "missing /v1/account/balances path"
        );
        assert!(
            spec.paths.paths.contains_key("/v1/account/ledger"),
            "missing /v1/account/ledger path"
        );
        assert!(
            spec.paths.paths.contains_key("/v1/webhooks/stripe"),
            "missing /v1/webhooks/stripe path"
        );

        // Verify schemas exist
        let schemas = spec.components.as_ref().unwrap();
        assert!(
            schemas.schemas.contains_key("ChatCompletionRequest"),
            "missing ChatCompletionRequest schema"
        );

        // Verify JSON serialization works
        let json = spec.to_json().expect("failed to serialize to JSON");
        assert!(json.contains("Eidolons API"));
        assert!(json.contains("/v1/chat/completions"));
    }
}
