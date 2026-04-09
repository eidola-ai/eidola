//! Eidola Server - A privacy-transparent AI proxy.
//!
//! This server accepts requests in OpenAI Chat Completions API format and
//! proxies them to Tinfoil's confidential inference enclaves, enriching
//! responses with inline privacy metadata and cryptographic verification.

use std::net::SocketAddr;

use axum::http::StatusCode;
use tokio::net::TcpListener;
use tracing::{error, info, warn};

use eidola_server::AppState;
use eidola_server::backend::TinfoilBackend;
use eidola_server::credentials;
use eidola_server::helpers::EpochConfig;
use eidola_server::measurements;
use eidola_server::stripe::StripeClient;
use eidola_server::telemetry;

/// Server configuration.
struct Config {
    bind_addr: SocketAddr,
    tinfoil_api_key: String,
    tinfoil_base_url: Option<String>,
    tinfoil_repo: String,
    database_url: String,
    database_password: Option<String>,
    database_ssl_cert: Option<String>,
    stripe_api_key: Option<String>,
    stripe_webhook_secret: Option<String>,
    credential_master_key: [u8; 32],
    pricing_markup: Option<f64>,
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

        // Source repo used to attest the upstream enclave via the ATC
        // `POST /attestation` endpoint. Must match the GitHub repo whose
        // signed measurements correspond to the running enclave.
        let tinfoil_repo = std::env::var("TINFOIL_REPO")
            .unwrap_or_else(|_| "tinfoilsh/confidential-model-router".to_string());

        let database_url = std::env::var("DATABASE_URL")
            .map_err(|_| "DATABASE_URL environment variable is required")?;

        let database_password = std::env::var("DATABASE_PASSWORD")
            .ok()
            .filter(|s| !s.is_empty());

        let database_ssl_cert = std::env::var("DATABASE_SSL_CERT")
            .ok()
            .filter(|s| !s.is_empty());

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

        let credential_master_key_hex = std::env::var("CREDENTIAL_MASTER_KEY")
            .map_err(|_| "CREDENTIAL_MASTER_KEY environment variable is required")?;
        let key_bytes = hex::decode(&credential_master_key_hex)
            .map_err(|e| format!("invalid CREDENTIAL_MASTER_KEY hex: {e}"))?;
        let credential_master_key: [u8; 32] = key_bytes.try_into().map_err(|_| {
            "CREDENTIAL_MASTER_KEY must be exactly 32 bytes (64 hex chars)".to_string()
        })?;

        // Verify measured secret hashes: if *_HASH env vars are set (committed in
        // tinfoil-config.yml and thus included in the enclave measurement), verify
        // that the corresponding runtime secret matches the hash. This binds
        // injected secrets to the measurement without exposing them in the config.
        verify_measured_secrets(&[
            ("CREDENTIAL_MASTER_KEY", &credential_master_key_hex),
            ("TINFOIL_API_KEY", &tinfoil_api_key),
            (
                "DATABASE_PASSWORD",
                database_password.as_deref().unwrap_or(""),
            ),
            ("STRIPE_API_KEY", stripe_api_key.as_deref().unwrap_or("")),
            (
                "STRIPE_WEBHOOK_SECRET",
                stripe_webhook_secret.as_deref().unwrap_or(""),
            ),
        ])?;

        Ok(Config {
            bind_addr,
            tinfoil_api_key,
            tinfoil_base_url,
            tinfoil_repo,
            database_url,
            database_password,
            database_ssl_cert,
            stripe_api_key,
            stripe_webhook_secret,
            credential_master_key,
            pricing_markup,
        })
    }
}

/// Verify that runtime secrets match their measured hashes.
///
/// For each `(name, value)` pair, checks if `{name}_HASH` is set as an env var.
/// If present, verifies the Argon2id hash matches `value`. If absent, the secret
/// is not measured and no check is performed.
///
/// The `_HASH` env vars should be hardcoded in `tinfoil-config.yml` so they are
/// included in the enclave measurement. This cryptographically binds injected
/// secrets to the measurement without exposing their plaintext in the config.
fn verify_measured_secrets(secrets: &[(&str, &str)]) -> Result<(), String> {
    use argon2::PasswordVerifier;

    for (name, value) in secrets {
        if value.is_empty() {
            continue;
        }
        let hash_var = format!("{name}_HASH");
        let Ok(expected_hash) = std::env::var(&hash_var) else {
            continue;
        };
        if expected_hash.is_empty() {
            continue;
        }

        let parsed = argon2::PasswordHash::new(&expected_hash)
            .map_err(|e| format!("{hash_var}: invalid Argon2 hash: {e}"))?;

        argon2::Argon2::default()
            .verify_password(value.as_bytes(), &parsed)
            .map_err(|_| {
                format!(
                    "{name} does not match measured hash in {hash_var}.\n\
                     The injected secret differs from what was committed in the \
                     enclave configuration. Refusing to start."
                )
            })?;

        tracing::info!("{name} verified against measured hash");
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Install the pure-Rust crypto provider for TLS (must be done before any TLS operations)
    rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider())
        .expect("failed to install rustls crypto provider");

    // Initialize logging + optional OpenTelemetry (when OTEL_EXPORTER_OTLP_ENDPOINT is set).
    let otel_guard = telemetry::init();

    // Load configuration
    let config = Config::load().await.map_err(|e| {
        error!("Configuration error: {}", e);
        e
    })?;

    info!("Starting Eidola server on {}", config.bind_addr);

    // Create database connection pool
    let db_pool = eidola_server::db::create_pool(
        &config.database_url,
        config.database_password.as_deref(),
        config.database_ssl_cert.as_deref(),
    )
    .map_err(|e| {
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

    // Build the attesting client. Verification happens per-handshake inside
    // the connector, so this call performs no network I/O — the first real
    // request through the client is also the first attestation. We make a
    // tiny smoke-test request immediately after construction to fail fast at
    // startup if the upstream is misconfigured.
    info!("Building Tinfoil attesting client...");
    let default_base_url = "https://inference.tinfoil.sh/v1".to_string();
    let inference_base_url = config
        .tinfoil_base_url
        .as_deref()
        .unwrap_or(&default_base_url);
    // Observers wired to the OTel TDX_ATTESTATIONS / SNP_ATTESTATIONS
    // counters so every attestation the verifier sees (including ones
    // it rejects) is surfaced as a labeled metric. The closures run on
    // the TLS handshake hot path inside the connector layer, so we keep
    // them to a single counter increment each with no allocation in the
    // steady state.
    let tdx_observer: tinfoil_verifier::TdxObserver =
        std::sync::Arc::new(|observation: &tinfoil_verifier::TdxTcbObservation| {
            telemetry::metrics::TDX_ATTESTATIONS.add(
                1,
                &[opentelemetry::KeyValue::new(
                    "status",
                    observation.status.as_metric_label(),
                )],
            );
        });
    let snp_observer: tinfoil_verifier::SevSnpObserver =
        std::sync::Arc::new(|observation: &tinfoil_verifier::SevSnpTcbObservation| {
            telemetry::metrics::SNP_ATTESTATIONS.add(
                1,
                &[opentelemetry::KeyValue::new(
                    "bucket",
                    observation.as_metric_label(),
                )],
            );
        });
    let attesting_client =
        tinfoil_verifier::attesting_client(tinfoil_verifier::AttestingClientConfig {
            allowed_measurements: measurements::ALLOWED.as_slice(),
            inference_base_url,
            atc_url: None,
            enclave_repo: Some(&config.tinfoil_repo),
            trusted_ark_der: None,
            trusted_ask_der: None,
            tdx_advisory_allowlist: None,
            tdx_observer: Some(tdx_observer),
            snp_min_tcb: None,
            snp_observer: Some(snp_observer),
        })
        .await
        .map_err(|e| {
            error!("Tinfoil attesting client build failed: {e}");
            e
        })?;

    info!("Smoke-testing Tinfoil enclave attestation via {inference_base_url}/models...");
    attesting_client
        .get(format!("{inference_base_url}/models"))
        .header(
            "authorization",
            format!("Bearer {}", config.tinfoil_api_key),
        )
        .send()
        .await
        .map_err(|e| {
            error!("Tinfoil attestation smoke test failed: {e}");
            e
        })?
        .error_for_status()
        .map_err(|e| {
            error!("Tinfoil attestation smoke test returned non-success: {e}");
            e
        })?;
    info!("Tinfoil attestation smoke test succeeded");

    // Create shared state
    let state = AppState::new(
        TinfoilBackend::new(
            attesting_client,
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

    // Verify that this server's clock agrees with the database clock
    // before doing anything that writes time-anchored state. A skewed
    // node would otherwise create issuer keys with bogus issuance
    // windows that other (correctly-clocked) nodes would never
    // produce, polluting shared state.
    eidola_server::db::check_clock_skew(&state.db_pool, eidola_server::db::MAX_CLOCK_SKEW)
        .await
        .map_err(|e| {
            error!("Database clock skew check failed: {}", e);
            e.to_string()
        })?;

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

    // Keep the database connection pool warm and prevent serverless Postgres
    // (e.g. Neon) from autosuspending the compute during quiet periods.
    eidola_server::db::spawn_keepalive(state.db_pool.clone(), std::time::Duration::from_secs(60));

    // Build the router with OpenAPI integration
    let (router, api) = eidola_server::build_router()
        .with_state(state)
        .split_for_parts();

    // Store the generated OpenAPI spec for the /openapi.json endpoint
    let api_json = api.to_json().expect("OpenAPI spec serialization failed");
    let app = router
        .route(
            "/openapi.json",
            axum::routing::get(move || {
                let spec = api_json.clone();
                async move { (StatusCode::OK, [("content-type", "application/json")], spec) }
            }),
        )
        .layer(axum::middleware::from_fn(
            eidola_server::middleware::observe,
        ));

    let listener = TcpListener::bind(config.bind_addr).await?;
    info!("Listening on http://{}", config.bind_addr);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Flush OTel data before exiting.
    if let Some(guard) = otel_guard {
        guard.shutdown();
    }

    Ok(())
}

/// Wait for SIGINT or SIGTERM for graceful shutdown.
async fn shutdown_signal() {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("received SIGINT, shutting down"),
        _ = terminate => info!("received SIGTERM, shutting down"),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_openapi_spec_generation() {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());

        // Build the full router to capture paths from handler annotations.
        let (_, spec) = eidola_server::build_router().split_for_parts();

        // Verify basic info
        assert_eq!(spec.info.title, "Eidola API");
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
        assert!(json.contains("Eidola API"));
        assert!(json.contains("/v1/chat/completions"));
    }
}
