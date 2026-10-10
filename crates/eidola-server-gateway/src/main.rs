//! Eidola Server - A privacy-transparent AI proxy.
//!
//! This server accepts requests in OpenAI Chat Completions API format and
//! proxies them to Tinfoil's confidential inference enclaves, enriching
//! responses with inline privacy metadata and cryptographic verification.

use std::net::SocketAddr;

use axum::http::StatusCode;
use tokio::net::TcpListener;
use tracing::{error, info, warn};

use eidola_server_gateway::AppState;
use eidola_server_gateway::backend::TinfoilBackend;
use eidola_server_gateway::credentials;
use eidola_server_gateway::helpers::EpochConfig;
use eidola_server_gateway::stripe::StripeClient;
use eidola_server_gateway::telemetry;

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
    /// Static seed for the required-terms gate (dev/test pin; see
    /// `terms_seed` parsing). Applied via the same monotonic upsert the
    /// terms-feed poller uses.
    terms_seed: Vec<eidola_server_gateway::db::RequiredDocumentRow>,
    /// Base URL of the published website to poll for the current legal
    /// document versions (e.g. `https://www.eidola.ai`). None = no polling.
    terms_feed_base_url: Option<String>,
    /// How often the terms feed re-polls.
    terms_refresh: std::time::Duration,
    /// How often `src/upstream_trust` re-checks Tinfoil's latest release.
    upstream_refresh: std::time::Duration,
    /// The bearer token Eidola's inference nodes verify (`ENGINE_TOKEN`).
    /// Required when this build pins an engine deployment.
    engine_token: Option<String>,
    /// How often the engine placement table is re-read.
    engine_placement_refresh: std::time::Duration,
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

        // Source repo whose signed releases drive the runtime measurement
        // allowlist for the upstream enclave.
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

        // Read for measurement verification only — the OTLP exporter
        // consumes this straight from the environment. Grafana routes by
        // these auth headers to a tenant, so leaving the hash unverified
        // would let a different telemetry destination be injected without
        // a measurement change (privacy-guarantees.md §7.4).
        let otel_exporter_otlp_headers = std::env::var("OTEL_EXPORTER_OTLP_HEADERS")
            .ok()
            .filter(|s| !s.is_empty());

        // The gateway's token for Eidola's inference nodes: a secret, measured
        // by its Argon2id hash like the others when `ENGINE_TOKEN_HASH` is set.
        let engine_token = std::env::var("ENGINE_TOKEN").ok().filter(|s| !s.is_empty());
        if engine_token.is_none() && !eidola_server_gateway::engine_trust::PINNED_MODELS.is_empty()
        {
            return Err(
                "ENGINE_TOKEN is required: this build pins Eidola-hosted engine deployments"
                    .to_string(),
            );
        }

        let pricing_markup = std::env::var("PRICING_MARKUP")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<f64>()
                    .map_err(|_| "PRICING_MARKUP must be a valid number".to_string())
            })
            .transpose()?;
        // Refuse to start with a markup below the pricing contract's safe
        // cost factor — see `validate_pricing_markup` for the loss-window
        // rationale (and the future dynamic-factor-via-/models note).
        eidola_server_gateway::backend::validate_pricing_markup(
            pricing_markup.unwrap_or(eidola_server_gateway::backend::DEFAULT_PRICING_MARKUP),
        )?;

        let credential_master_key_hex = std::env::var("CREDENTIAL_MASTER_KEY")
            .map_err(|_| "CREDENTIAL_MASTER_KEY environment variable is required")?;
        // The hex error is discarded, not interpolated: its Display quotes
        // the offending character — a byte of the configured master key.
        let key_bytes = hex::decode(&credential_master_key_hex)
            .map_err(|_| "invalid CREDENTIAL_MASTER_KEY hex".to_string())?;
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
            (
                "OTEL_EXPORTER_OTLP_HEADERS",
                otel_exporter_otlp_headers.as_deref().unwrap_or(""),
            ),
            ("ENGINE_TOKEN", engine_token.as_deref().unwrap_or("")),
        ])?;

        // Terms-acceptance gate. Two independent sources feed the shared
        // `required_document` table (both through the same monotonic
        // upsert — version may only increase):
        //
        //  - TERMS_FEED_BASE_URL: the published website to poll for each
        //    document's exact source (`/terms/source.md`,
        //    `/privacy/source.md`). This is the production source of
        //    truth — legal documents outlive any client/server release,
        //    so their required versions must never be baked into the
        //    measured server config (which would make a terms update
        //    require a coordinated client update).
        //  - TERMS_OF_SERVICE_SHA256 / PRIVACY_POLICY_SHA256 (+ optional
        //    `_VERSION`, default 1, and `_URL`): a static dev/test pin,
        //    seeded once at startup.
        //
        // Neither set = gate disabled.
        let terms_feed_base_url = std::env::var("TERMS_FEED_BASE_URL")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| s.trim_end_matches('/').to_string());

        // Both refresh periods go through the same guard: each drives a
        // detached loop where zero is a live-process failure (a panicked
        // ticker, or a poll with no sleep between passes), so it is refused
        // here instead.
        let terms_refresh =
            eidola_server_gateway::helpers::refresh_secs_from_env("TERMS_REFRESH_SECS", 600)?;
        let upstream_refresh = eidola_server_gateway::helpers::refresh_secs_from_env(
            "TINFOIL_MEASUREMENT_REFRESH_SECS",
            eidola_server_gateway::upstream_trust::DEFAULT_REFRESH_SECS,
        )?;
        let engine_placement_refresh = eidola_server_gateway::helpers::refresh_secs_from_env(
            "ENGINE_PLACEMENT_REFRESH_SECS",
            30,
        )?;

        let mut terms_seed = Vec::new();
        for (document, hash_var, version_var, url_var, default_url) in [
            (
                "terms_of_service",
                "TERMS_OF_SERVICE_SHA256",
                "TERMS_OF_SERVICE_VERSION",
                "TERMS_OF_SERVICE_URL",
                "https://www.eidola.ai/terms/",
            ),
            (
                "privacy_policy",
                "PRIVACY_POLICY_SHA256",
                "PRIVACY_POLICY_VERSION",
                "PRIVACY_POLICY_URL",
                "https://www.eidola.ai/privacy/",
            ),
        ] {
            let Some(sha256) = std::env::var(hash_var).ok().filter(|s| !s.is_empty()) else {
                continue;
            };
            if sha256.len() != 64 || !sha256.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(format!("{hash_var} must be 64 hex chars (a SHA-256)"));
            }
            let version = std::env::var(version_var)
                .ok()
                .filter(|s| !s.is_empty())
                .map(|s| {
                    s.parse::<i64>()
                        .map_err(|_| format!("{version_var} must be a positive integer"))
                })
                .transpose()?
                .unwrap_or(1);
            if version < 1 {
                return Err(format!("{version_var} must be >= 1"));
            }
            let url = std::env::var(url_var)
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| default_url.to_string());
            terms_seed.push(eidola_server_gateway::db::RequiredDocumentRow {
                document: document.to_string(),
                version,
                sha256: sha256.to_lowercase(),
                url,
            });
        }

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
            terms_seed,
            terms_feed_base_url,
            terms_refresh,
            upstream_refresh,
            engine_token,
            engine_placement_refresh,
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
    let db_pool = eidola_server_gateway::db::create_pool(
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
    // The observer feeds the OTel SNP_ATTESTATIONS counter for every report
    // whose signature verifies, including reports the policy then rejects.
    // It runs on the handshake hot path, so it performs one allocation-free
    // counter increment.
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
    // The server runs `FROM scratch` inside an enclave with no system trust
    // store, so trust roots come from the bundled Mozilla list. Tinfoil's
    // production cert chains under public WebPKI. We deliberately do not
    // pull in `rustls-native-certs` here.
    let mut tls_roots = rustls::RootCertStore::empty();
    tls_roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    // The allowed upstream measurements are resolved at *runtime* from
    // Tinfoil's latest router release (see `src/upstream_trust` for why).
    // This factory rebuilds the attesting client whenever that set changes;
    // all the `attesting_client` wiring (TLS roots, TCB policy, telemetry
    // observers) lives here so `upstream_trust` stays
    // free of a telemetry dependency. `attesting_client` does no network
    // I/O, so rebuilding is cheap.
    let client_factory: eidola_server_gateway::upstream_trust::AttestingClientFactory = {
        let inference_base_url = inference_base_url.to_string();
        std::sync::Arc::new(move |allowed: Vec<tinfoil_verifier::EnclaveMeasurement>| {
            let inference_base_url = inference_base_url.clone();
            let tls_roots = tls_roots.clone();
            let snp_observer = snp_observer.clone();
            Box::pin(async move {
                // The router's release records pin SEV-SNP only.
                let allowed: Vec<tinfoil_verifier::AllowedMeasurement> = allowed
                    .iter()
                    .map(tinfoil_verifier::AllowedMeasurement::from)
                    .collect();
                tinfoil_verifier::attesting_client(tinfoil_verifier::AttestingClientConfig {
                    allowed_measurements: allowed.as_slice(),
                    inference_base_url: &inference_base_url,
                    trusted_ark_der: None,
                    trusted_ask_der: None,
                    snp_min_tcb: None,
                    snp_observer: Some(snp_observer),
                    attestation_observer: None,
                    tls_roots,
                })
                .await
                .map_err(|e| e.to_string())
            })
        })
    };

    // Bootstrap the runtime trust: resolve + verify the latest release's
    // measurement, build the initial attesting client, and start the periodic
    // refresh task. There is no static fallback — if the measurement can't be
    // resolved and verified at boot, the server refuses to start.
    info!("Resolving Tinfoil upstream measurement and building attesting client...");
    let upstream = eidola_server_gateway::upstream_trust::UpstreamTrust::bootstrap(
        config.tinfoil_repo.clone(),
        client_factory,
    )
    .await
    .map_err(|e| {
        error!("Upstream trust bootstrap failed: {e}");
        e
    })?;
    let client_cell = upstream.client_cell();
    std::sync::Arc::clone(&upstream).spawn_refresh(config.upstream_refresh);

    // Readiness: attest the enclave once through the current client, failing
    // fast at startup if the upstream is misconfigured or attestation fails.
    info!("Smoke-testing Tinfoil enclave attestation via {inference_base_url}/models...");
    client_cell
        .load()
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

    // The router to Eidola-hosted engines, when this build pins any. Each
    // upstream's client attests against its model's compiled-in pins; the
    // placement table only says where engines are. A first placement read and
    // probe run before the server listens, so a healthy deployment takes
    // traffic from the start; a failure there is not fatal (the refresh task
    // keeps trying, and requests for those models are refused meanwhile).
    let engines = match config.engine_token.clone() {
        Some(token) if !eidola_server_gateway::engine_trust::PINNED_MODELS.is_empty() => {
            use eidola_server_gateway::engine_router::{self, placement};
            let token = eidola_server_gateway::engine_trust::protocol::EngineToken::new(token)
                .map_err(|e| {
                    error!("ENGINE_TOKEN: {e}");
                    e.to_string()
                })?;
            let models = engine_router::EngineModel::compiled().map_err(|e| {
                error!("engine pins: {e}");
                e
            })?;
            let mut tls_roots = rustls::RootCertStore::empty();
            tls_roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
            let router = engine_router::EngineRouter::new(
                models,
                token,
                engine_router::attesting_client_factory(engine_router::AttestationRoots {
                    tls_roots,
                    trusted_ark_der: None,
                    trusted_ask_der: None,
                }),
                engine_router::RouterConfig {
                    placement_refresh: config.engine_placement_refresh,
                    ..Default::default()
                },
            );
            let source: std::sync::Arc<dyn placement::PlacementSource> =
                std::sync::Arc::new(placement::PostgresPlacement(db_pool.clone()));
            info!("Reading engine placement and probing engines...");
            router.refresh(source.as_ref()).await;
            router.spawn(source);
            Some(router)
        }
        _ => None,
    };

    // Create shared state
    let state = AppState::new(
        TinfoilBackend::new(
            client_cell,
            config.tinfoil_api_key.clone(),
            config.tinfoil_base_url.clone(),
            config.pricing_markup,
        ),
        engines,
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
    eidola_server_gateway::db::check_clock_skew(
        &state.db_pool,
        eidola_server_gateway::db::MAX_CLOCK_SKEW,
    )
    .await
    .map_err(|e| {
        error!("Database clock skew check failed: {}", e);
        e.to_string()
    })?;

    // Terms-acceptance gate: seed any static env pins into the shared
    // required_document table (monotonic — a stale pin can't regress a
    // newer polled version), then start the terms-feed poller if a feed
    // URL is configured. Seed failures are fatal (an explicitly pinned
    // gate that silently doesn't apply would be worse than not starting);
    // poller failures are per-tick and logged.
    for doc in &config.terms_seed {
        use eidola_server_gateway::db::RecordRequiredOutcome;
        match eidola_server_gateway::db::record_required_document(&state.db_pool, doc).await {
            Ok(RecordRequiredOutcome::Recorded) => info!(
                "terms gate: {} seeded at version {} ({})",
                doc.document, doc.version, doc.sha256
            ),
            Ok(RecordRequiredOutcome::AlreadyRecorded) => info!(
                "terms gate: {} version {} already on record, seed unchanged",
                doc.document, doc.version
            ),
            Ok(RecordRequiredOutcome::HashConflict { stored_sha256 }) => {
                // An explicit pin that contradicts the recorded history is
                // an operator error worth refusing to start over: either
                // the pin is wrong, or someone changed a document's bytes
                // without a version bump.
                error!(
                    "terms gate: seed for {} version {} has hash {} but that version is \
                     recorded with hash {} — fix the pin or publish a new version",
                    doc.document, doc.version, doc.sha256, stored_sha256
                );
                return Err("terms seed conflicts with recorded document history".into());
            }
            Err(e) => {
                error!("terms gate: seeding {} failed: {}", doc.document, e);
                return Err(e.to_string().into());
            }
        }
    }
    if let Some(base_url) = config.terms_feed_base_url.clone() {
        info!(
            "terms feed: polling {} every {:?}",
            base_url, config.terms_refresh
        );
        eidola_server_gateway::terms_feed::spawn_terms_feed_task(
            state.db_pool.clone(),
            base_url,
            config.terms_refresh,
        );
    }

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
    eidola_server_gateway::db::spawn_keepalive(
        state.db_pool.clone(),
        std::time::Duration::from_secs(60),
    );

    // Build the router with OpenAPI integration
    let (router, api) = eidola_server_gateway::build_router()
        .with_state(state)
        .split_for_parts();

    // Store the generated OpenAPI spec for the /openapi.json endpoint
    let api_json = api.to_json().expect("OpenAPI spec serialization failed");
    let router = router.route(
        "/openapi.json",
        axum::routing::get(move || {
            let spec = api_json.clone();
            async move { (StatusCode::OK, [("content-type", "application/json")], spec) }
        }),
    );
    // Every response shaped, refusals and 404s included (`padding::shape`);
    // observed outside that, so the metrics see what the client got.
    let app = eidola_server_gateway::padding::shape(router).layer(axum::middleware::from_fn(
        eidola_server_gateway::middleware::observe,
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
    /// Written by argon2 0.5.3 (`Argon2::default()`, random salt) for the
    /// secret below. Measured `*_HASH` values in `tinfoil-config.yml` and
    /// every stored account hash were produced by that version, so an argon2
    /// upgrade must keep verifying them — or the enclave refuses to start
    /// (`verify_measured_secrets`) and every account stops authenticating.
    const ARGON2_0_5_HASH: &str = "$argon2id$v=19$m=19456,t=2,p=1$6pErOZUqn/EOSXWM1TCvWQ$iGAfGeBFsWcYRHUv3H6CSSISW1RVT8L3V2DeGpWw14Y";
    const ARGON2_0_5_SECRET: &[u8] = b"argon2-0.5-fixture-secret";

    #[test]
    fn hashes_written_by_argon2_0_5_still_verify() {
        use argon2::PasswordVerifier;
        let parsed = argon2::PasswordHash::new(ARGON2_0_5_HASH).expect("parse 0.5 hash");
        argon2::Argon2::default()
            .verify_password(ARGON2_0_5_SECRET, &parsed)
            .expect("a 0.5-written hash must verify");
        assert!(
            argon2::Argon2::default()
                .verify_password(b"not-the-secret", &parsed)
                .is_err()
        );
    }

    #[test]
    fn new_hashes_keep_the_measured_parameters() {
        use argon2::PasswordHasher;
        // `hash-secret` and account creation both rely on the default
        // parameters; a change here would silently change what new hashes
        // cost and how they read in `tinfoil-config.yml`.
        let hash = argon2::Argon2::default()
            .hash_password(ARGON2_0_5_SECRET)
            .expect("hash")
            .to_string();
        assert!(
            hash.starts_with("$argon2id$v=19$m=19456,t=2,p=1$"),
            "unexpected parameters: {hash}"
        );
    }

    #[test]
    fn test_openapi_spec_generation() {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());

        // Build the full router to capture paths from handler annotations.
        let (_, spec) = eidola_server_gateway::build_router().split_for_parts();

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
