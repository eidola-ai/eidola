//! Eidolons Server - A privacy-transparent AI proxy.
//!
//! This server accepts requests in OpenAI Chat Completions API format and
//! proxies them to Tinfoil's confidential inference enclaves, enriching
//! responses with inline privacy metadata and cryptographic verification.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::http::StatusCode;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as AutoBuilder;
use tokio::net::TcpListener;
use tower_service::Service;
use tracing::{debug, error, info, warn};

use dstack_sdk::dstack_client::DstackClient;

use eidolons_server::AppState;
use eidolons_server::backend::{self, TinfoilBackend};
use eidolons_server::credentials;
use eidolons_server::helpers::EpochConfig;
use eidolons_server::stripe::StripeClient;
use eidolons_server::tls;

/// Server configuration.
struct Config {
    bind_addr: SocketAddr,
    tinfoil_api_key: String,
    tinfoil_base_url: Option<String>,
    database_url: String,
    stripe_api_key: Option<String>,
    stripe_webhook_secret: Option<String>,
    credential_master_key: [u8; 32],
    pricing_markup: Option<f64>,
    dstack: DstackClient,
    tls_sans: Vec<String>,
}

impl Config {
    async fn load() -> Result<Self, String> {
        let bind_addr = std::env::var("BIND_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:8443".to_string())
            .parse()
            .map_err(|e| format!("invalid BIND_ADDR: {}", e))?;

        let tinfoil_api_key = std::env::var("TINFOIL_API_KEY")
            .map_err(|_| "TINFOIL_API_KEY environment variable is required")?;

        let tinfoil_base_url = std::env::var("TINFOIL_BASE_URL").ok();

        let database_url = std::env::var("DATABASE_URL")
            .map_err(|_| "DATABASE_URL environment variable is required")?;

        let stripe_api_key = std::env::var("STRIPE_API_KEY")
            .ok()
            .filter(|s| !s.is_empty());

        let stripe_webhook_secret = std::env::var("STRIPE_WEBHOOK_SECRET")
            .ok()
            .filter(|s| !s.is_empty());

        let pricing_markup = std::env::var("PRICING_MARKUP")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<f64>()
                    .map_err(|_| "PRICING_MARKUP must be a valid number".to_string())
            })
            .transpose()?;

        let tls_sans = tls::parse_sans();

        // Derive the credential master key from dstack KMS.
        // The SDK auto-discovers the socket at /var/run/dstack/dstack.sock (containers
        // and production). For host-side dev on macOS, DSTACK_SIMULATOR_ENDPOINT can
        // point at the simulator's HTTP port (e.g. http://localhost:8090).
        let dstack = DstackClient::new(
            std::env::var("DSTACK_SIMULATOR_ENDPOINT")
                .ok()
                .as_deref(),
        );
        info!("Deriving credential master key from dstack...");
        let key_response = dstack
            .get_key(Some("eidolons/credential-master-key/v1".to_string()), None)
            .await
            .map_err(|e| format!("failed to derive credential master key: {e}"))?;
        let key_bytes = hex::decode(&key_response.key)
            .map_err(|e| format!("invalid hex in dstack key: {e}"))?;
        let credential_master_key: [u8; 32] = key_bytes
            .get(..32)
            .and_then(|s| <[u8; 32]>::try_from(s).ok())
            .ok_or("dstack key too short (need >= 32 bytes)")?;
        info!("Credential master key derived successfully");

        Ok(Config {
            bind_addr,
            tinfoil_api_key,
            tinfoil_base_url,
            database_url,
            stripe_api_key,
            stripe_webhook_secret,
            credential_master_key,
            pricing_markup,
            dstack,
            tls_sans,
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

    // Load configuration (includes dstack key derivation)
    let config = Config::load().await.map_err(|e| {
        error!("Configuration error: {}", e);
        e
    })?;

    info!("Starting Eidolons server on {}", config.bind_addr);

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

    // Credential key cache and epoch configuration
    let credential_key_cache: credentials::KeyCache = Default::default();
    let epoch_config = EpochConfig::default();

    // Verify Tinfoil enclave attestation and build a TLS-pinned client.
    info!("Verifying Tinfoil enclave attestation...");
    let (pinned_client, verification) = tinfoil_verifier::verify_and_pin(
        backend::ALLOWED_MEASUREMENTS,
        None, // use default ATC endpoint
    )
    .await
    .map_err(|e| {
        error!("Tinfoil attestation verification failed: {e}");
        e
    })?;
    info!(
        "Tinfoil attestation verified — measurement: {}, TLS fingerprint: {}",
        verification.measurement, verification.tls_fingerprint
    );

    // Create shared state
    let state = AppState::new(
        TinfoilBackend::new(
            pinned_client,
            config.tinfoil_api_key.clone(),
            config.tinfoil_base_url.clone(),
            config.pricing_markup,
        ),
        db_pool,
        stripe,
        config.stripe_webhook_secret,
        config.credential_master_key,
        credential_key_cache,
        epoch_config,
    );

    // Provision issuer keys on boot and start periodic rotation task.
    match credentials::ensure_keys(
        &state.credential_key_cache,
        &state.credential_master_key,
        &state.db_pool,
        &state.epoch_config,
    )
    .await
    {
        Ok(key_hash) => info!("Issuer key ready: {}", hex::encode(key_hash)),
        Err(e) => warn!("Failed to provision issuer keys on boot: {}", e),
    }

    credentials::spawn_key_rotation_task(
        state.credential_key_cache.clone(),
        state.credential_master_key,
        state.db_pool.clone(),
        state.epoch_config.clone(),
    );

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

    // Fetch initial RA-TLS certificate from dstack (fatal on failure).
    info!("Fetching RA-TLS certificate from dstack...");
    let initial_cert = tls::fetch_ra_tls_cert(&config.dstack, &config.tls_sans)
        .await
        .map_err(|e| {
            error!("Failed to fetch RA-TLS certificate: {e}");
            e
        })?;
    info!("RA-TLS certificate obtained");

    let resolver = tls::RaTlsCertResolver::new(initial_cert);
    let tls_acceptor = tls::build_tls_acceptor(Arc::clone(&resolver));

    // Spawn background cert rotation (attestation quotes expire).
    tls::spawn_cert_rotation_task(Arc::clone(&resolver), config.dstack, config.tls_sans);

    // Bind and serve with RA-TLS.
    let listener = TcpListener::bind(config.bind_addr).await?;
    info!("Listening on https://{}", config.bind_addr);

    loop {
        let (tcp_stream, remote_addr) = listener.accept().await?;
        let tls_acceptor = tls_acceptor.clone();
        let app = app.clone();

        tokio::spawn(async move {
            let tls_stream = match tls_acceptor.accept(tcp_stream).await {
                Ok(s) => s,
                Err(e) => {
                    debug!("TLS handshake failed from {remote_addr}: {e}");
                    return;
                }
            };

            let io = TokioIo::new(tls_stream);
            let service = hyper::service::service_fn(move |req| {
                let mut app = app.clone();
                async move { app.call(req).await }
            });

            if let Err(e) = AutoBuilder::new(TokioExecutor::new())
                .serve_connection(io, service)
                .await
            {
                debug!("Connection error from {remote_addr}: {e}");
            }
        });
    }
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
