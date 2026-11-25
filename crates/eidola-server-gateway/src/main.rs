//! Eidola gateway - A privacy-transparent AI proxy.
//!
//! The gateway accepts requests in OpenAI Chat Completions API format and
//! routes them to Eidola's own inference engines, in confidential VMs it
//! attests against the deployments pinned in its own build, enriching
//! responses with inline privacy metadata and cryptographic verification.

use std::net::SocketAddr;

use axum::http::StatusCode;
use tokio::net::TcpListener;
use tracing::{error, info, warn};

use eidola_common::env::Env;
use eidola_server_gateway::AppState;
use eidola_server_gateway::backend::Catalog;
use eidola_server_gateway::credentials;
use eidola_server_gateway::stripe::StripeClient;
use eidola_server_gateway::telemetry;

/// Server configuration.
struct Config {
    bind_addr: SocketAddr,
    database_url: String,
    database_password: Option<String>,
    database_ssl_cert: Option<String>,
    stripe_api_key: Option<String>,
    credential_master_key: [u8; 32],
    /// Static seed for the required-terms gate (dev/test pin; see
    /// `terms_seed` parsing). Applied via the same monotonic upsert the
    /// terms-feed poller uses.
    terms_seed: Vec<eidola_server_gateway::db::RequiredDocumentRow>,
    /// Base URL of the published website to poll for the current legal
    /// document versions (e.g. `https://www.eidola.ai`). None = no polling.
    terms_feed_base_url: Option<String>,
    /// How often the terms feed re-polls.
    terms_refresh: std::time::Duration,
    /// The bearer token Eidola's inference nodes verify (`ENGINE_TOKEN`).
    /// Required when this build pins an engine deployment.
    engine_token: Option<String>,
    /// How often the engine placement table is re-read.
    engine_placement_refresh: std::time::Duration,
}

impl Config {
    async fn load(env: &Env) -> Result<Self, String> {
        let bind_addr = env
            .var("BIND_ADDR")
            .unwrap_or_else(|| "127.0.0.1:8443".to_string())
            .parse()
            .map_err(|e| format!("invalid {}: {}", env.name("BIND_ADDR"), e))?;

        let database_url = env.var("DATABASE_URL").ok_or_else(|| {
            format!(
                "{} environment variable is required",
                env.name("DATABASE_URL")
            )
        })?;

        let database_password = env.var("DATABASE_PASSWORD").filter(|s| !s.is_empty());

        let database_ssl_cert = env.var("DATABASE_SSL_CERT").filter(|s| !s.is_empty());

        let stripe_api_key = env.var("STRIPE_API_KEY").filter(|s| !s.is_empty());

        // The gateway's token for Eidola's inference nodes: a secret, like the
        // others. Absent, the gateway starts without an engine router (see
        // `main`).
        let engine_token = env.var("ENGINE_TOKEN").filter(|s| !s.is_empty());

        let master_key_var = env.name("CREDENTIAL_MASTER_KEY");
        let credential_master_key_hex = env
            .var("CREDENTIAL_MASTER_KEY")
            .ok_or_else(|| format!("{master_key_var} environment variable is required"))?;
        // The hex error is discarded, not interpolated: its Display quotes
        // the offending character — a byte of the configured master key.
        let key_bytes = hex::decode(&credential_master_key_hex)
            .map_err(|_| format!("invalid {master_key_var} hex"))?;
        let credential_master_key: [u8; 32] = key_bytes
            .try_into()
            .map_err(|_| format!("{master_key_var} must be exactly 32 bytes (64 hex chars)"))?;

        // Terms-acceptance gate. Two independent sources feed the shared
        // `required_document_version` table (both through the same monotonic
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
        let terms_feed_base_url = env
            .var("TERMS_FEED_BASE_URL")
            .filter(|s| !s.is_empty())
            .map(|s| s.trim_end_matches('/').to_string());

        // Every refresh period goes through the same guard: each drives a
        // detached loop where zero is a live-process failure (a panicked
        // ticker, or a poll with no sleep between passes), so it is refused
        // here instead.
        let terms_refresh =
            eidola_server_gateway::helpers::refresh_secs_from_env(env, "TERMS_REFRESH_SECS", 600)?;
        let engine_placement_refresh = eidola_server_gateway::helpers::refresh_secs_from_env(
            env,
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
            let Some(sha256) = env.var(hash_var).filter(|s| !s.is_empty()) else {
                continue;
            };
            if sha256.len() != 64 || !sha256.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(format!(
                    "{} must be 64 hex chars (a SHA-256)",
                    env.name(hash_var)
                ));
            }
            let version = env
                .var(version_var)
                .filter(|s| !s.is_empty())
                .map(|s| {
                    s.parse::<i64>().map_err(|_| {
                        format!("{} must be a positive integer", env.name(version_var))
                    })
                })
                .transpose()?
                .unwrap_or(1);
            if version < 1 {
                return Err(format!("{} must be >= 1", env.name(version_var)));
            }
            let url = env
                .var(url_var)
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
            database_url,
            database_password,
            database_ssl_cert,
            stripe_api_key,
            credential_master_key,
            terms_seed,
            terms_feed_base_url,
            terms_refresh,
            engine_token,
            engine_placement_refresh,
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Install the pure-Rust crypto provider for TLS (must be done before any TLS operations)
    rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider())
        .expect("failed to install rustls crypto provider");

    // Every variable below is read under this prefix, so it is read first:
    // nothing, not even logging, is configured before it.
    let env = match Env::from_process() {
        Ok(env) => env,
        Err(e) => {
            eprintln!("Configuration error: {e}");
            std::process::exit(1);
        }
    };

    // Initialize logging + optional OpenTelemetry (when OTEL_EXPORTER_OTLP_ENDPOINT is set).
    let otel_guard = telemetry::init(&env);
    info!("Environment variable prefix: {:?}", env.prefix());

    // Load configuration
    let config = Config::load(&env).await.map_err(|e| {
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
    telemetry::metrics::register_db_pool(db_pool.clone());

    // Create Stripe client (optional)
    let stripe = config.stripe_api_key.map(StripeClient::new);
    if stripe.is_none() {
        warn!(
            "{} not set — Stripe checkout, subscription and portal endpoints will return 503",
            env.name("STRIPE_API_KEY")
        );
    }

    // Credential key cache. Key windows come from the public calendar
    // (`eidola_common::issuer_calendar`), not from configuration.
    let credential_key_cache: credentials::KeyCache = Default::default();

    // The router to Eidola-hosted engines, when this build pins any. Each
    // upstream's client attests against its model's compiled-in pins; the
    // placement table only says where engines are. A first placement read and
    // probe run before the server listens, so a healthy deployment takes
    // traffic from the start; a failure there is not fatal (the refresh task
    // keeps trying, and requests for those models are refused meanwhile).
    //
    // Without `ENGINE_TOKEN` there is no router: no node would accept this
    // gateway, so it serves accounts, billing and the catalog, and refuses
    // every inference request as unavailable, with its charge refunded.
    if config.engine_token.is_none()
        && !eidola_server_gateway::engine_trust::PINNED_MODELS.is_empty()
    {
        error!(
            "ENGINE_TOKEN is not set: this build pins engine deployments and cannot reach them; \
             inference requests are refused until the deployment supplies it"
        );
    }
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
            telemetry::metrics::register_engine_router(router.clone());
            Some(router)
        }
        _ => None,
    };

    // The catalog, priced. `PRICING_OVERRIDES` is keyed by weights hash, and
    // an override that names no served model, or does not fit its pricing,
    // refuses the boot rather than being ignored.
    let pricing_overrides = env.var("PRICING_OVERRIDES").filter(|s| !s.is_empty());
    let catalog = Catalog::new(pricing_overrides.as_deref()).map_err(|e| {
        error!("{e}");
        e
    })?;

    // Create shared state
    let state = AppState::new(
        catalog,
        engines,
        db_pool,
        stripe,
        config.credential_master_key,
        credential_key_cache,
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
    // required_document_version table (monotonic — a stale pin can't regress a
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
    } else if config.terms_seed.is_empty() {
        // Nothing will ever turn the gate on in this process. Off is a
        // deployment's choice, but an unset secret looks the same, so it is
        // said once, at startup.
        match eidola_server_gateway::db::get_required_documents(&state.db_pool).await {
            Ok(required) if required.is_empty() => warn!(
                "terms gate is off: neither {} nor a terms seed is set, and no required \
                 document is on record, so accounts are never asked to accept terms",
                env.name("TERMS_FEED_BASE_URL")
            ),
            Ok(_) => {}
            Err(e) => warn!("terms gate: reading the required documents failed: {e}"),
        }
    }

    // Provision issuer keys on boot and start periodic rotation task.
    match credentials::ensure_keys(
        &state.credential_key_cache,
        &state.credential_master_key,
        &state.db_pool,
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
    /// secret below. Every stored account hash was produced by that version,
    /// so an argon2 upgrade must keep verifying them, or every account stops
    /// authenticating.
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
    fn new_hashes_keep_the_default_parameters() {
        use argon2::PasswordHasher;
        // Account creation relies on the default parameters; a change here
        // would silently change what new hashes cost.
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
            !spec.paths.paths.contains_key("/v1/webhooks/stripe"),
            "webhooks belong to the independently deployed billing receiver"
        );
        let repair = spec
            .paths
            .paths
            .get("/v1/account/reconcile")
            .expect("missing account repair path");
        assert!(repair.post.is_some(), "missing queued repair request");
        assert!(repair.get.is_some(), "missing durable repair status");
        assert!(
            repair
                .post
                .as_ref()
                .unwrap()
                .responses
                .responses
                .contains_key("202"),
            "repair must acknowledge its durable request without claiming completion"
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
