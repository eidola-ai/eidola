//! Credential issuance: key management and credential endpoints.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use aes_gcm::aead::Aead;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use anonymous_credit_tokens::{
    IssuanceRequest, Params, PrivateKey, Scalar, credit_to_scalar, scalar_to_credit,
};
use axum::Json;
use axum::extract::State;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rand_core::{OsRng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;
use tracing::{info, warn};
use utoipa::ToSchema;

use crate::AppState;
use crate::auth::{BasicAuth, TokenAuth};
use crate::db;
use crate::error::ServerError;
use crate::helpers::{EpochConfig, system_time_to_iso_lossy};
use crate::response::RefundInfo;

// ---------------------------------------------------------------------------
// Key cache types
// ---------------------------------------------------------------------------

/// A decrypted issuer key held in memory.
pub struct DecryptedIssuerKey {
    pub secret_key: PrivateKey,
    pub params: Params,
    pub key_hash: [u8; 32],
    pub request_context_scalar: Scalar,
    pub issue_from: SystemTime,
    pub issue_until: SystemTime,
    /// End of the acceptance window. A cached key past it is retired: its
    /// credentials are refused and the key is evicted, exactly as if the
    /// database lookup (`accept_until > now()`) had been made.
    pub accept_until: SystemTime,
}

/// Thread-safe cache of decrypted issuer keys, keyed by SHA-256(pkI).
pub type KeyCache = Arc<RwLock<HashMap<[u8; 32], DecryptedIssuerKey>>>;

// ---------------------------------------------------------------------------
// Key encryption / decryption
// ---------------------------------------------------------------------------

/// Derive a per-key AES-256-GCM key from the master key using HKDF-SHA256.
fn derive_key_encryption_key(master_key: &[u8; 32], key_id: &str) -> [u8; 32] {
    let hk = hkdf::Hkdf::<Sha256>::new(Some(key_id.as_bytes()), master_key);
    let mut okm = [0u8; 32];
    hk.expand(b"eidola-issuer-key-v1", &mut okm)
        .expect("32 bytes is a valid HKDF-SHA256 output length");
    okm
}

/// Encrypt a private key with AES-256-GCM.
/// Output format: `nonce (12 bytes) || ciphertext+tag`.
fn encrypt_private_key(
    master_key: &[u8; 32],
    key_id: &str,
    plaintext: &[u8],
) -> Result<Vec<u8>, ServerError> {
    let aes_key = derive_key_encryption_key(master_key, key_id);
    let cipher = Aes256Gcm::new_from_slice(&aes_key)
        .map_err(|e| ServerError::Internal(format!("AES key init failed: {}", e)))?;

    let mut nonce_bytes = [0u8; 12];
    rand_core::OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from(nonce_bytes);

    let ciphertext = cipher
        .encrypt(&nonce, plaintext)
        .map_err(|e| ServerError::Internal(format!("AES encryption failed: {}", e)))?;

    let mut out = Vec::with_capacity(12 + ciphertext.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Decrypt a private key from the `nonce || ciphertext+tag` format.
fn decrypt_private_key(
    master_key: &[u8; 32],
    key_id: &str,
    encrypted: &[u8],
) -> Result<Vec<u8>, ServerError> {
    if encrypted.len() < 12 {
        return Err(ServerError::Internal("encrypted key too short".to_string()));
    }
    let (nonce_bytes, ciphertext) = encrypted.split_at(12);
    let aes_key = derive_key_encryption_key(master_key, key_id);
    let cipher = Aes256Gcm::new_from_slice(&aes_key)
        .map_err(|e| ServerError::Internal(format!("AES key init failed: {}", e)))?;
    let nonce = Nonce::try_from(nonce_bytes)
        .map_err(|e| ServerError::Internal(format!("invalid nonce length: {}", e)))?;

    cipher
        .decrypt(&nonce, ciphertext)
        .map_err(|e| ServerError::Internal(format!("AES decryption failed: {}", e)))
}

// ---------------------------------------------------------------------------
// Domain separator
// ---------------------------------------------------------------------------

/// Domain separator components for credential operations.
///
/// These are baked into all credential issuance and verification via
/// `Params::new()`, which constructs the full domain separator string
/// `ACT-v1:{ORG}:{SERVICE}:{DEPLOYMENT}:{VERSION}`.
///
/// Per the ACT spec (draft-schlesinger-cfrg-act), the version field is an
/// ISO 8601 date indicating when parameters were generated. Critically,
/// the domain separator is orthogonal to key rotation:
///
/// - **Key rotation** (every ~7 days) provides temporal epoch boundaries,
///   nullifier partitioning, and bounds the lifetime of issued credentials.
///   Keys rotate frequently and automatically.
///
/// - **Domain separator** provides cryptographic isolation between different
///   deployments, services, and protocol versions. It changes only on
///   protocol upgrades or deployment changes — never as part of routine
///   key rotation.
///
/// We intentionally do NOT rotate the domain separator on a schedule.
/// Doing so would shrink the anonymity set (which is the intersection of
/// users sharing the same key AND domain separator) without providing any
/// benefit that key rotation doesn't already give. Nullifiers are partitioned
/// by issuer key, not domain separator, so shorter domain separator epochs
/// would not enable earlier pruning. The spec itself frames version changes
/// as exceptional: "When parameters need to be updated (e.g., for security
/// reasons or protocol upgrades), a new version date MUST be used."
///
/// If these values ever need to change, all existing credentials become
/// unspendable under the new parameters — this is by design (cryptographic
/// isolation), but means changes must be coordinated with a migration plan.
const DS_ORG: &str = "eidola";
const DS_SERVICE: &str = "inference";
const DS_DEPLOYMENT: &str = "production";
const DS_VERSION: &str = "2026-03-05";

/// The full domain separator string, for storage in the database.
/// Must match what `Params::new(DS_ORG, DS_SERVICE, DS_DEPLOYMENT, DS_VERSION)`
/// produces internally.
fn domain_separator() -> String {
    format!(
        "ACT-v1:{}:{}:{}:{}",
        DS_ORG, DS_SERVICE, DS_DEPLOYMENT, DS_VERSION
    )
}

// ---------------------------------------------------------------------------
// Issuer key ID and request context (ACT Privacy Pass spec)
// ---------------------------------------------------------------------------

/// ACT token type (provisional, pending IANA assignment).
pub const ACT_TOKEN_TYPE: u16 = 0xE5AD;

/// Issuer name used in TokenChallenge and request_context construction.
const ISSUER_NAME: &str = "eidola";

/// Origin info used in TokenChallenge and request_context construction.
const ORIGIN_INFO: &str = "inference";

/// Compute the issuer key ID as SHA-256(pkI_serialized).
///
/// Per draft-schlesinger-privacypass-act-01, Section 6.
pub fn compute_key_hash(public_key_bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(public_key_bytes).into()
}

/// Compute the request_context bytes.
///
/// Per draft-schlesinger-privacypass-act-01, Section 9.2:
/// `request_context = concat(issuer_name, origin_info, credential_context, issuer_key_id)`
///
/// We use empty `credential_context` (key rotation handles expiration).
pub fn compute_request_context(key_hash: &[u8; 32]) -> Vec<u8> {
    let mut ctx = Vec::new();
    ctx.extend_from_slice(ISSUER_NAME.as_bytes());
    ctx.extend_from_slice(ORIGIN_INFO.as_bytes());
    // credential_context is empty
    ctx.extend_from_slice(key_hash);
    ctx
}

/// Hash request_context bytes to a Scalar for use as the `ctx` parameter
/// in ACT cryptographic operations.
pub fn request_context_to_scalar(request_context: &[u8]) -> Scalar {
    let hash: [u8; 32] = Sha256::digest(request_context).into();
    Scalar::from_bytes_mod_order(hash)
}

/// Serialize a TokenChallenge structure.
///
/// Per draft-schlesinger-privacypass-act-01, Section 7:
/// ```text
/// struct {
///     uint16_t token_type = 0xE5AD;
///     opaque issuer_name<1..2^16-1>;
///     opaque redemption_context<0..32>;
///     opaque origin_info<0..2^16-1>;
///     opaque credential_context<0..32>;
/// } TokenChallenge;
/// ```
///
/// We use empty `redemption_context` and `credential_context` (no per-request
/// freshness needed since we have nullifiers, and key rotation handles expiration).
pub fn serialize_token_challenge() -> Vec<u8> {
    let mut buf = Vec::new();
    // token_type (2 bytes, big-endian)
    buf.extend_from_slice(&ACT_TOKEN_TYPE.to_be_bytes());
    // issuer_name with 2-byte length prefix
    buf.extend_from_slice(&(ISSUER_NAME.len() as u16).to_be_bytes());
    buf.extend_from_slice(ISSUER_NAME.as_bytes());
    // redemption_context with 1-byte length prefix (empty)
    buf.push(0);
    // origin_info with 2-byte length prefix
    buf.extend_from_slice(&(ORIGIN_INFO.len() as u16).to_be_bytes());
    buf.extend_from_slice(ORIGIN_INFO.as_bytes());
    // credential_context with 1-byte length prefix (empty)
    buf.push(0);
    buf
}

/// Compute the expected challenge_digest = SHA-256(TokenChallenge).
pub fn compute_challenge_digest() -> [u8; 32] {
    Sha256::digest(serialize_token_challenge()).into()
}

// ---------------------------------------------------------------------------
// Key lifecycle
// ---------------------------------------------------------------------------

/// Decrypt an issuer key row and insert it into the cache. Returns the key hash.
fn cache_key(
    cache: &mut HashMap<[u8; 32], DecryptedIssuerKey>,
    master_key: &[u8; 32],
    row: &db::IssuerKeyRow,
) -> Result<[u8; 32], ServerError> {
    let key_hash = compute_key_hash(&row.public_key);
    if cache.contains_key(&key_hash) {
        return Ok(key_hash);
    }
    let plaintext = decrypt_private_key(master_key, &row.id, &row.private_key_enc)?;
    let secret_key = PrivateKey::from_cbor(&plaintext).map_err(|e| {
        ServerError::Internal(format!("failed to decode issuer private key: {}", e))
    })?;
    let params = Params::new(DS_ORG, DS_SERVICE, DS_DEPLOYMENT, DS_VERSION);
    let request_context = compute_request_context(&key_hash);
    let request_context_scalar = request_context_to_scalar(&request_context);
    cache.insert(
        key_hash,
        DecryptedIssuerKey {
            secret_key,
            params,
            key_hash,
            request_context_scalar,
            issue_from: row.issue_from,
            issue_until: row.issue_until,
            accept_until: row.accept_until,
        },
    );
    Ok(key_hash)
}

/// Generate a new issuer key pair and build a row ready for insertion.
fn generate_key(
    master_key: &[u8; 32],
    issue_from: SystemTime,
    epoch_config: &EpochConfig,
) -> Result<db::IssuerKeyRow, ServerError> {
    let secret_key = PrivateKey::random(OsRng);
    let public_key_cbor = secret_key
        .public()
        .to_cbor()
        .map_err(|e| ServerError::Internal(format!("failed to encode public key: {}", e)))?;
    let id = hex::encode(compute_key_hash(&public_key_cbor));
    let private_key_cbor = secret_key
        .to_cbor()
        .map_err(|e| ServerError::Internal(format!("failed to encode private key: {}", e)))?;
    let encrypted = encrypt_private_key(master_key, &id, &private_key_cbor)?;

    let (issue_until, accept_until) = epoch_config.boundaries_from(issue_from);

    Ok(db::IssuerKeyRow {
        id,
        private_key_enc: encrypted,
        public_key: public_key_cbor,
        domain_separator: domain_separator(),
        issue_from,
        issue_until,
        accept_until,
    })
}

/// Ensure that both a current key and the next key exist in the database
/// and are cached in memory. Returns the current key's hash (for issuance).
///
/// Key chaining: each new key's `issue_from` = predecessor's `issue_until`.
/// The very first key's `issue_from` = now.
///
/// Race safety: uses a serializable transaction so concurrent server instances
/// cannot create duplicate keys.
pub async fn ensure_keys(
    key_cache: &KeyCache,
    master_key: &[u8; 32],
    pool: &deadpool_postgres::Pool,
    epoch_config: &EpochConfig,
) -> Result<[u8; 32], ServerError> {
    let now = SystemTime::now();

    // Fast path: check if a valid current key is already cached.
    {
        let cache = key_cache.read().await;
        let current = cache
            .values()
            .find(|k| k.issue_from <= now && k.issue_until > now);
        if let Some(k) = current {
            // Also check that a next key exists in cache.
            let has_next = cache.values().any(|k2| k2.issue_from >= k.issue_until);
            if has_next {
                return Ok(k.key_hash);
            }
        }
    }

    // Slow path: check the database and provision any missing keys.
    // We may need to create up to 2 keys (current + next), or just the next one.
    loop {
        let ec = epoch_config.clone();
        let mk = *master_key;

        let result = db::insert_issuer_key_checked(pool, move |latest| {
            let now = SystemTime::now();
            match latest {
                Some(latest_key) => {
                    if latest_key.issue_until <= now {
                        // Latest key's issuance window has passed — need a new current key.
                        // Chain from the latest key's issue_until, but if that's in the past,
                        // start from now to avoid creating already-expired keys.
                        let issue_from = if latest_key.issue_until < now {
                            now
                        } else {
                            latest_key.issue_until
                        };
                        Ok(Some(generate_key(&mk, issue_from, &ec)?))
                    } else if latest_key.issue_from <= now {
                        // Latest key is the current key — need the next key.
                        Ok(Some(generate_key(&mk, latest_key.issue_until, &ec)?))
                    } else {
                        // Latest key is already a future key — nothing to do.
                        Ok(None)
                    }
                }
                None => {
                    // No keys at all — bootstrap with issue_from = now.
                    Ok(Some(generate_key(&mk, now, &ec)?))
                }
            }
        })
        .await;

        match result {
            Ok(Some(inserted)) => {
                info!(
                    "provisioned issuer key {} (issue_from={})",
                    inserted.id,
                    system_time_to_iso_lossy(inserted.issue_from)
                );
                // Cache it and loop to check if we need another key.
                let mut cache = key_cache.write().await;
                cache_key(&mut cache, master_key, &inserted)?;
                continue;
            }
            Ok(None) => {
                // No key needed — break out and load from DB.
                break;
            }
            Err(e) => {
                // Serialization failure means another instance raced us — retry.
                let msg = format!("{e:?}");
                if msg.contains("serialization") || msg.contains("40001") {
                    info!("key provisioning serialization conflict, retrying");
                    continue;
                }
                return Err(e);
            }
        }
    }

    // Load all valid keys from DB into cache and find the current one.
    // Re-capture `now` here: the outer `now` was taken before the
    // provisioning round-trips, and on a high-latency database link the
    // freshly-inserted key's `issue_from` (captured inside the insert
    // closure) can be strictly after the outer `now`, which would make
    // the key look like a future key and trip the "no current issuer
    // key found" error below.
    let now = SystemTime::now();
    let rows = db::get_valid_issuer_keys(pool).await?;
    let mut cache = key_cache.write().await;
    cache.retain(|_, k| k.accept_until > now);
    let mut current_hash = None;
    for row in &rows {
        let kh = cache_key(&mut cache, master_key, row)?;
        if row.issue_from <= now && row.issue_until > now {
            current_hash = Some(kh);
        }
    }

    current_hash.ok_or_else(|| {
        ServerError::Internal("no current issuer key found after provisioning".to_string())
    })
}

/// Load an issuer key for spending verification, checking the cache first
/// then falling back to the database.
pub async fn load_key_for_spending(
    key_cache: &KeyCache,
    master_key: &[u8; 32],
    pool: &deadpool_postgres::Pool,
    key_hash: &[u8; 32],
) -> Result<(), ServerError> {
    // Fast path: already in cache, and not yet retired. A retired key is
    // refused here, not reloaded: the database would refuse it too.
    if let Some(key) = key_cache.read().await.get(key_hash) {
        if key.accept_until > SystemTime::now() {
            return Ok(());
        }
        return Err(ServerError::Unauthorized {
            message: "unknown or expired issuer key".to_string(),
        });
    }

    // Slow path: load from DB by key hash
    let row = db::get_issuer_key_by_hash(pool, key_hash)
        .await?
        .ok_or_else(|| ServerError::Unauthorized {
            message: "unknown or expired issuer key".to_string(),
        })?;

    let mut cache = key_cache.write().await;
    cache_key(&mut cache, master_key, &row)?;
    Ok(())
}

/// Periodic key rotation check interval: 1 hour.
const KEY_CHECK_INTERVAL: Duration = Duration::from_secs(3600);

/// Maximum random jitter added to the check interval.
const KEY_CHECK_JITTER: Duration = Duration::from_secs(300);

/// Spawn a background task that periodically ensures keys are provisioned.
pub fn spawn_key_rotation_task(
    key_cache: KeyCache,
    master_key: [u8; 32],
    pool: deadpool_postgres::Pool,
    epoch_config: EpochConfig,
) {
    tokio::spawn(async move {
        loop {
            // Sleep with jitter.
            let jitter_secs = OsRng.next_u64() % KEY_CHECK_JITTER.as_secs();
            let sleep_dur = KEY_CHECK_INTERVAL + Duration::from_secs(jitter_secs);
            tokio::time::sleep(sleep_dur).await;

            // Re-verify clock agreement before each rotation iteration.
            // Drift can develop during a long process lifetime (NTP
            // hiccups, hypervisor migration, etc.); skipping the
            // iteration is safer than provisioning a key with a bogus
            // issuance window. The next iteration will retry.
            if let Err(e) = db::check_clock_skew(&pool, db::MAX_CLOCK_SKEW).await {
                warn!("skipping key rotation iteration: {}", e);
                continue;
            }

            match ensure_keys(&key_cache, &master_key, &pool, &epoch_config).await {
                Ok(key_hash) => {
                    info!("periodic key check: current key {}", hex::encode(key_hash));
                }
                Err(e) => {
                    warn!("periodic key check failed: {}", e);
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

/// A single issuer public key in the `GET /v1/keys` response.
#[derive(Serialize, ToSchema)]
pub struct IssuerKeyResponse {
    /// Issuer key ID: hex-encoded SHA-256(pkI_serialized).
    pub id: String,
    /// Base64url-encoded CBOR public key (compressed Ristretto255 point).
    pub public_key: String,
    /// Domain separator used for parameter generation.
    pub domain_separator: String,
    /// Start of the issuance window (ISO 8601).
    pub issue_from: String,
    /// End of the issuance window (ISO 8601).
    pub issue_until: String,
    /// End of the acceptance window (ISO 8601).
    pub accept_until: String,
}

/// Response for `GET /v1/keys`.
#[derive(Serialize, ToSchema)]
pub struct ListKeysResponse {
    pub data: Vec<IssuerKeyResponse>,
}

/// Request body for `POST /v1/account/credentials`.
#[derive(Deserialize, ToSchema)]
pub struct IssueCredentialsRequest {
    /// Base64-encoded CBOR `IssuanceRequest`.
    pub issuance_request: String,
    /// Number of credits to issue.
    pub credits: i64,
}

/// Response for `POST /v1/account/credentials`.
#[derive(Serialize, ToSchema)]
pub struct IssueCredentialsResponse {
    /// Base64-encoded CBOR `IssuanceResponse`.
    pub issuance_response: String,
    /// Issuer key identifier: hex-encoded SHA-256(pkI_serialized).
    pub issuer_key_id: String,
    /// Number of credits issued.
    pub credits: i64,
    /// The ledger entry ID for this issuance.
    pub ledger_entry_id: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `GET /v1/keys` — list valid issuer public keys (unauthenticated).
#[utoipa::path(
    get,
    path = "/v1/keys",
    tag = "Public",
    responses(
        (status = 200, description = "Valid issuer public keys", body = ListKeysResponse),
    )
)]
pub async fn list_keys(
    State(state): State<AppState>,
) -> Result<Json<ListKeysResponse>, ServerError> {
    let rows = db::get_valid_issuer_keys(&state.db_pool).await?;

    let data: Vec<IssuerKeyResponse> = rows
        .into_iter()
        .map(|r| IssuerKeyResponse {
            id: r.id.clone(),
            public_key: URL_SAFE_NO_PAD.encode(&r.public_key),
            domain_separator: r.domain_separator,
            issue_from: system_time_to_iso_lossy(r.issue_from),
            issue_until: system_time_to_iso_lossy(r.issue_until),
            accept_until: system_time_to_iso_lossy(r.accept_until),
        })
        .collect();

    Ok(Json(ListKeysResponse { data }))
}

/// `POST /v1/account/credentials` — issue anonymous credentials (authenticated).
#[utoipa::path(
    post,
    path = "/v1/account/credentials",
    tag = "Linked",
    request_body = IssueCredentialsRequest,
    security(("basic" = [])),
    responses(
        (status = 200, description = "Credential issued", body = IssueCredentialsResponse),
        (status = 400, description = "Invalid request", body = crate::types::ErrorResponse),
        (status = 401, description = "Invalid credentials", body = crate::types::ErrorResponse),
        (status = 402, description = "Insufficient credit balance", body = crate::types::ErrorResponse),
        (status = 428, description = "Terms acceptance required", body = crate::types::ErrorResponse),
        (status = 503, description = "Credential issuance not configured", body = crate::types::ErrorResponse),
    )
)]
pub async fn issue_credentials(
    BasicAuth(account_id): BasicAuth,
    State(state): State<AppState>,
    Json(request): Json<IssueCredentialsRequest>,
) -> Result<Json<IssueCredentialsResponse>, ServerError> {
    let master_key = &state.credential_master_key;

    if request.credits <= 0 {
        return Err(ServerError::BadRequest {
            message: "credits must be greater than 0".to_string(),
        });
    }

    // Conversion is a gated operation: the account must have accepted the
    // currently required terms/privacy versions (no-op when unconfigured).
    crate::account::ensure_terms_accepted(&state, account_id).await?;

    // Both decode errors are discarded, not interpolated: their Display
    // quotes the offending byte, its offset, and the input length — all
    // derived from the client-supplied blob (`auth.rs` sets the same
    // pattern for the spend-proof decode).
    let issuance_cbor = URL_SAFE_NO_PAD
        .decode(&request.issuance_request)
        .map_err(|_| ServerError::BadRequest {
            message: "invalid base64 in issuance_request".to_string(),
        })?;

    let issuance_request =
        IssuanceRequest::from_cbor(&issuance_cbor).map_err(|_| ServerError::BadRequest {
            message: "invalid CBOR issuance request".to_string(),
        })?;

    let credit_scalar =
        credit_to_scalar::<128>(request.credits as u128).map_err(|e| ServerError::BadRequest {
            message: format!("invalid credit amount: {}", e),
        })?;

    let key_hash = ensure_keys(
        &state.credential_key_cache,
        master_key,
        &state.db_pool,
        &state.epoch_config,
    )
    .await?;

    let key_id = hex::encode(key_hash);

    let issuance_response = {
        let cache = state.credential_key_cache.read().await;
        let key = cache.get(&key_hash).ok_or_else(|| {
            ServerError::Internal("current key evicted from cache unexpectedly".to_string())
        })?;
        key.secret_key
            .issue::<128>(
                &key.params,
                &issuance_request,
                credit_scalar,
                key.request_context_scalar,
                OsRng,
            )
            .map_err(|e| {
                warn!("credential issuance failed: {}", e);
                ServerError::BadRequest {
                    message: format!("issuance failed: {}", e),
                }
            })?
    };

    let response_cbor = issuance_response
        .to_cbor()
        .map_err(|e| ServerError::Internal(format!("failed to encode issuance response: {}", e)))?;

    let ledger_entry_id =
        match db::insert_credential_issuance(&state.db_pool, account_id, request.credits, &key_id)
            .await?
        {
            Some(id) => id,
            None => {
                let available = match db::get_available_balance(&state.db_pool, account_id).await {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("failed to fetch balance for 402 response: {e}");
                        0
                    }
                };
                return Err(ServerError::PaymentRequired {
                    message: "insufficient balance".to_string(),
                    available,
                });
            }
        };

    // No ledger id and no amount: the ledger row already records both
    // durably, and an exported log line naming them would hand the
    // telemetry vendor a timestamped per-issuance record — the same rule
    // the webhook success logs follow.
    info!("credential issued");

    Ok(Json(IssueCredentialsResponse {
        issuance_response: URL_SAFE_NO_PAD.encode(&response_cbor),
        issuer_key_id: key_id,
        credits: request.credits,
        ledger_entry_id: ledger_entry_id.to_string(),
    }))
}

// ---------------------------------------------------------------------------
// Refund recovery
// ---------------------------------------------------------------------------

/// Response from the refund recovery endpoint.
#[derive(Debug, Serialize, ToSchema)]
pub struct RefundRecoveryResponse {
    /// The refund token.
    pub refund: RefundInfo,
}

/// Recover a refund token for a previously spent ACT.
///
/// If the ACT's nullifier was already recorded (the server received and
/// processed the original request), the stored refund token is returned.
///
/// If the nullifier was NOT recorded (the original request never arrived),
/// the server verifies the spend proof, issues a full refund (no credits
/// consumed), records the nullifier together with it, and returns it.
///
/// The spend proof is verified before anything else, in both cases.
///
/// In both cases, the client gets a refund token it can combine with its
/// local `PreRefund` state to reconstruct a credential.
#[utoipa::path(
    post,
    path = "/v1/credentials/refund",
    tag = "Unlinked",
    responses(
        (status = 200, description = "Refund token recovered or issued", body = RefundRecoveryResponse),
        (status = 401, description = "Invalid spend proof", body = crate::types::ErrorResponse),
        (status = 404, description = "Nullifier found but refund token not yet available", body = crate::types::ErrorResponse),
        (status = 500, description = "Internal error", body = crate::types::ErrorResponse)
    )
)]
pub async fn recover_refund(
    TokenAuth(act): TokenAuth,
    State(state): State<AppState>,
) -> Result<Json<RefundRecoveryResponse>, ServerError> {
    // Verify first, exactly as the chat path does: until the proof checks
    // out the caller is unauthenticated, and a rejected token must leave
    // nothing behind — no nullifier recorded.
    crate::handlers::verify_spend_proof(&state, &act).await?;

    let key_id = hex::encode(act.issuer_key_hash);
    let nullifier_bytes = act.spend_proof.nullifier().as_bytes().to_vec();
    let answer = |refund_cbor: &[u8], key_id: String| {
        Json(RefundRecoveryResponse {
            refund: RefundInfo {
                refund: URL_SAFE_NO_PAD.encode(refund_cbor),
                issuer_key_id: key_id,
            },
        })
    };

    // A recorded spend's row answers through `recorded_refund`, which reads
    // the row, checks the settling mark and re-reads: see its doc for why
    // each step is where it is. A row without a token predates provisional
    // refunds and can never be served.
    let still_in_flight = || {
        Err(ServerError::NotFound {
            message: "nullifier recorded but refund token not yet available — \
                      the original request may still be in flight, retry shortly"
                .to_string(),
        })
    };

    // Check if the nullifier was already recorded (original request arrived).
    match recorded_refund(&state, &key_id, &nullifier_bytes).await? {
        Recorded::Token(refund_cbor) => return Ok(answer(&refund_cbor, key_id)),
        Recorded::InFlight => return still_in_flight(),
        Recorded::Absent => {}
    }

    // The nullifier doesn't exist: the original request never arrived.
    // Prepare a full refund (charge_credits returned in full since no work
    // was done) — computing it releases nothing.
    let charge_credits = scalar_to_credit::<128>(&act.spend_proof.charge()).map_err(|_| {
        ServerError::BadRequest {
            message: "invalid charge amount in spend proof".to_string(),
        }
    })?;

    let refund_cbor = crate::handlers::mint_refund(
        &state,
        &act.spend_proof,
        &act.issuer_key_hash,
        charge_credits,
    )
    .await?;

    // Record the nullifier and the refund in one statement: the nullifier is
    // durable before the refund is released (the forking-attack rule), and
    // no interruption — a dropped connection, a restart — can leave it
    // recorded without the token every retry has to find.
    if db::record_nullifier_with_refund(
        &state.db_pool,
        &key_id,
        &nullifier_bytes,
        &refund_cbor,
        db::RefundStanding::Settled,
    )
    .await?
    {
        return Ok(answer(&refund_cbor, key_id));
    }

    // Recorded by someone else meanwhile: a concurrent recovery (settled) or
    // the chat path (provisional, and in flight in this process).
    match recorded_refund(&state, &key_id, &nullifier_bytes).await? {
        Recorded::Token(refund_cbor) => Ok(answer(&refund_cbor, key_id)),
        Recorded::InFlight | Recorded::Absent => still_in_flight(),
    }
}

/// What a recorded spend answers recovery with.
enum Recorded {
    /// No row: the original request never arrived.
    Absent,
    /// This process can still settle the spend (or the row predates
    /// provisional refunds): "retry shortly".
    InFlight,
    /// A token that is final — the settlement, or a provisional refund
    /// nothing can still replace.
    Token(Vec<u8>),
}

/// Read the spend's row, and decide whether its token may be served.
///
/// A settled token is final. A provisional one is the chat path's zero
/// refund from before dispatch, final only once nothing can still settle
/// the spend — while this process holds the settling mark for it, the
/// settled token is still coming, and a client handed the provisional one
/// would spend its one successor on the full hold. The three steps, in this
/// order, cover both sides of the mark's lifetime:
///
/// 1. **Read the row.** A row that exists was written by a chat that took
///    its mark *before* the INSERT, so the check that follows observes that
///    mark. (Checking before the read would miss a chat that began and
///    inserted in between.)
/// 2. **Check the mark.** Held: in flight. Absent: nothing in this process
///    can still settle the spend — but a settlement may have stored its token
///    and released between the read and this check, so:
/// 3. **Re-read the row.** Settlement stores the token before it releases the
///    mark, so a mark seen absent means this read sees the settled token if
///    there is one; otherwise the provisional token stands.
async fn recorded_refund(
    state: &AppState,
    key_id: &str,
    nullifier: &[u8],
) -> Result<Recorded, ServerError> {
    let Some(stored) = db::get_refund_token(&state.db_pool, key_id, nullifier).await? else {
        return Ok(Recorded::Absent);
    };
    match stored {
        db::StoredRefund {
            refund_token: Some(refund_cbor),
            standing: db::RefundStanding::Settled,
        } => Ok(Recorded::Token(refund_cbor)),
        db::StoredRefund {
            refund_token: Some(provisional),
            standing: db::RefundStanding::Provisional,
        } => {
            if state.settling.is_settling(key_id, nullifier) {
                return Ok(Recorded::InFlight);
            }
            Ok(
                match db::get_refund_token(&state.db_pool, key_id, nullifier).await? {
                    Some(db::StoredRefund {
                        refund_token: Some(latest),
                        ..
                    }) => Recorded::Token(latest),
                    _ => Recorded::Token(provisional),
                },
            )
        }
        db::StoredRefund {
            refund_token: None, ..
        } => Ok(Recorded::InFlight),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let master_key = [42u8; 32];
        let key_id = "550e8400-e29b-41d4-a716-446655440000";
        let plaintext = b"test private key material";
        let encrypted = encrypt_private_key(&master_key, key_id, plaintext).unwrap();
        let decrypted = decrypt_private_key(&master_key, key_id, &encrypted).unwrap();
        assert_eq!(plaintext.as_slice(), &decrypted);
    }

    #[test]
    fn test_decrypt_wrong_key_id_fails() {
        let master_key = [42u8; 32];
        let encrypted = encrypt_private_key(
            &master_key,
            "550e8400-e29b-41d4-a716-446655440000",
            b"secret",
        )
        .unwrap();
        let result = decrypt_private_key(
            &master_key,
            "660e8400-e29b-41d4-a716-446655440000",
            &encrypted,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_domain_separator_is_stable() {
        // The domain separator must never change between key rotations.
        assert_eq!(
            domain_separator(),
            "ACT-v1:eidola:inference:production:2026-03-05"
        );
    }

    #[test]
    fn test_compute_key_hash_deterministic() {
        let pk_bytes = b"test public key bytes";
        let h1 = compute_key_hash(pk_bytes);
        let h2 = compute_key_hash(pk_bytes);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 32);
    }

    #[test]
    fn test_request_context_includes_key_hash() {
        let key_hash = [42u8; 32];
        let ctx = compute_request_context(&key_hash);
        // Should contain issuer_name + origin_info + key_hash
        assert!(ctx.starts_with(b"eidola"));
        assert!(ctx.ends_with(&key_hash));
        assert_eq!(ctx.len(), "eidola".len() + "inference".len() + 32);
    }

    #[test]
    fn test_request_context_scalar_deterministic() {
        let key_hash = [42u8; 32];
        let ctx = compute_request_context(&key_hash);
        let s1 = request_context_to_scalar(&ctx);
        let s2 = request_context_to_scalar(&ctx);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_different_key_hashes_give_different_contexts() {
        let ctx1 = request_context_to_scalar(&compute_request_context(&[1u8; 32]));
        let ctx2 = request_context_to_scalar(&compute_request_context(&[2u8; 32]));
        assert_ne!(ctx1, ctx2);
    }

    #[test]
    fn test_token_challenge_serialization() {
        let challenge = serialize_token_challenge();
        let expected_len = 2 + 2 + ISSUER_NAME.len() + 1 + 2 + ORIGIN_INFO.len() + 1;
        assert_eq!(challenge.len(), expected_len);
        // Starts with 0xE5AD
        assert_eq!(challenge[0], 0xE5);
        assert_eq!(challenge[1], 0xAD);
    }

    #[test]
    fn test_challenge_digest_is_stable() {
        let d1 = compute_challenge_digest();
        let d2 = compute_challenge_digest();
        assert_eq!(d1, d2);
        assert_eq!(d1.len(), 32);
    }

    #[test]
    fn test_params_does_not_panic() {
        // Verify Params::new accepts our domain separator components (no colons).
        let _params = Params::new(DS_ORG, DS_SERVICE, DS_DEPLOYMENT, DS_VERSION);
    }

    /// Two refunds of one spend mint two successors, but the successor's
    /// nullifier is fixed by the spend proof, not by the refund: they share
    /// it, so only one of them can ever be spent. That is what makes handing
    /// out a provisional refund before the final one safe — the client keeps
    /// the better of the two, never both.
    #[test]
    fn every_refund_of_one_spend_mints_the_same_successor_nullifier() {
        use anonymous_credit_tokens::{PreIssuance, Refund};
        let params = Params::new(DS_ORG, DS_SERVICE, DS_DEPLOYMENT, DS_VERSION);
        let key = PrivateKey::random(OsRng);
        let ctx = Scalar::from(42u128);
        let pre = PreIssuance::random(OsRng);
        let request = pre.request(&params, OsRng);
        let response = key
            .issue::<128>(&params, &request, Scalar::from(1_000u128), ctx, OsRng)
            .unwrap();
        let token = pre
            .to_credit_token::<128>(&params, key.public(), &request, &response)
            .unwrap();
        let (spend, pre_refund) = token
            .prove_spend::<128>(&params, Scalar::from(100u128), OsRng)
            .unwrap();

        let successor = |t: u128| {
            let refund: Refund = key.refund(&params, &spend, Scalar::from(t), OsRng).unwrap();
            pre_refund
                .to_credit_token(&params, &spend, &refund, key.public())
                .unwrap()
        };
        let (provisional, settled) = (successor(0), successor(60));
        assert_eq!(provisional.credits(), Scalar::from(900u128));
        assert_eq!(settled.credits(), Scalar::from(960u128));
        let nullifier_of = |t: &anonymous_credit_tokens::CreditToken| {
            t.prove_spend::<128>(&params, Scalar::from(1u128), OsRng)
                .unwrap()
                .0
                .nullifier()
        };
        assert_eq!(nullifier_of(&provisional), nullifier_of(&settled));
    }

    // -----------------------------------------------------------------------
    // Refund recovery (require running postgres: just db && just db-reset)
    // -----------------------------------------------------------------------

    async fn recovery_state() -> (AppState, [u8; 32]) {
        recovery_state_under(test_epoch()).await
    }

    /// The epoch the DB-backed tests mint their keys under: long enough for
    /// any test, short enough to leave the shared development database alone.
    /// A key minted under the default epoch would be advertised, selected as
    /// the newest issuing key and fail `cache_key` (it is sealed under this
    /// module's test master key) for a server pointed at the same database,
    /// for the whole default issuing window and grace period.
    fn test_epoch() -> crate::helpers::EpochConfig {
        crate::helpers::EpochConfig {
            epoch_duration: Duration::from_secs(60),
            grace_period: Duration::from_secs(120),
        }
    }

    async fn recovery_state_under(epoch: crate::helpers::EpochConfig) -> (AppState, [u8; 32]) {
        recovery_state_at(epoch, None).await
    }

    /// `upstream` is the base URL chat requests are forwarded to.
    async fn recovery_state_at(
        epoch: crate::helpers::EpochConfig,
        upstream: Option<String>,
    ) -> (AppState, [u8; 32]) {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
        let pool = db::create_pool(&url, None, None).expect("failed to create test pool");
        let backend = crate::backend::TinfoilBackend::new(
            Arc::new(arc_swap::ArcSwap::from_pointee(reqwest::Client::new())),
            String::new(),
            upstream,
            None,
        );
        let master_key = [7u8; 32];
        let cache: KeyCache = Arc::new(RwLock::new(HashMap::new()));
        // A key of this test's own: the shared test database holds other
        // tests' keys, sealed under other master keys.
        // Serializable, so a concurrent test's insert can make it retry.
        let row = loop {
            let row = generate_key(&master_key, SystemTime::now(), &epoch).unwrap();
            if let Ok(Some(row)) =
                db::insert_issuer_key_checked(&pool, move |_| Ok(Some(row))).await
            {
                break row;
            }
        };
        let key_hash: [u8; 32] = hex::decode(&row.id).unwrap().try_into().unwrap();
        load_key_for_spending(&cache, &master_key, &pool, &key_hash)
            .await
            .expect("load key");
        let state = AppState::new(backend, pool, None, None, master_key, cache, epoch);
        (state, key_hash)
    }

    /// A spend of `charge` from a fresh `credits` credential, presented under
    /// `key_hash`. `signer` issues the credential — the cached key for an
    /// honest spend, any other key for a proof the server must reject.
    async fn spend_under(
        state: &AppState,
        key_hash: [u8; 32],
        signer: Option<&PrivateKey>,
        credits: u128,
        charge: u128,
    ) -> (crate::auth::ActSpend, anonymous_credit_tokens::PreRefund) {
        use anonymous_credit_tokens::PreIssuance;
        let cache = state.credential_key_cache.read().await;
        let key = cache.get(&key_hash).expect("key cached");
        let signer = signer.unwrap_or(&key.secret_key);
        let pre = PreIssuance::random(OsRng);
        let request = pre.request(&key.params, OsRng);
        let response = signer
            .issue::<128>(
                &key.params,
                &request,
                Scalar::from(credits),
                key.request_context_scalar,
                OsRng,
            )
            .expect("issue");
        let token = pre
            .to_credit_token::<128>(&key.params, signer.public(), &request, &response)
            .expect("credit token");
        let (spend_proof, pre_refund) = token
            .prove_spend::<128>(&key.params, Scalar::from(charge), OsRng)
            .expect("prove spend");
        let act = crate::auth::ActSpend {
            challenge_digest: compute_challenge_digest(),
            issuer_key_hash: key_hash,
            spend_proof,
        };
        (act, pre_refund)
    }

    /// The same presented token again (`ActSpend` is not `Clone`).
    fn copy_of(act: &crate::auth::ActSpend) -> crate::auth::ActSpend {
        let cbor = act.spend_proof.to_cbor().unwrap();
        crate::auth::ActSpend {
            challenge_digest: act.challenge_digest,
            issuer_key_hash: act.issuer_key_hash,
            spend_proof: anonymous_credit_tokens::SpendProof::<128>::from_cbor(&cbor).unwrap(),
        }
    }

    async fn nullifier_rows(state: &AppState, act: &crate::auth::ActSpend) -> i64 {
        let client = state.db_pool.get().await.unwrap();
        client
            .query_one(
                "SELECT count(*) FROM nullifier WHERE issuer_key_id = $1 AND value = $2",
                &[
                    &hex::encode(act.issuer_key_hash),
                    &act.spend_proof.nullifier().as_bytes().to_vec(),
                ],
            )
            .await
            .unwrap()
            .get(0)
    }

    /// The endpoint is unlinked and unauthenticated until the proof verifies,
    /// so a rejected proof must leave nothing behind — the chat path's Phase 1
    /// rule ("no nullifier recorded"). Recording first let anyone holding a
    /// key's public hash write permanent rows into the billing database.
    #[tokio::test]
    #[ignore]
    async fn refund_recovery_with_an_invalid_proof_records_nothing() {
        let (state, key_hash) = recovery_state().await;
        let forger = PrivateKey::random(OsRng);
        let (act, _) = spend_under(&state, key_hash, Some(&forger), 1_000, 400).await;

        let result = recover_refund(TokenAuth(copy_of(&act)), State(state.clone())).await;

        assert!(
            matches!(result, Err(ServerError::Unauthorized { .. })),
            "a forged proof must be refused"
        );
        assert_eq!(
            nullifier_rows(&state, &act).await,
            0,
            "a refused proof wrote a nullifier"
        );
    }

    /// Recovery exists for the interrupted request, so an interrupted
    /// *recovery* — a dropped connection, a restart — must be retryable too.
    /// A test-only trigger holds this nullifier's insert open while the call
    /// is abandoned; the statement still commits, and the retry must find a
    /// refund rather than a nullifier with none ("retry shortly", forever).
    #[tokio::test]
    #[ignore]
    async fn refund_recovery_interrupted_after_its_write_can_be_retried() {
        let (state, key_hash) = recovery_state().await;
        let (act, _) = spend_under(&state, key_hash, None, 1_000, 400).await;
        let nullifier_hex = hex::encode(act.spend_proof.nullifier().as_bytes());
        let trigger = format!("review_slow_{}", &nullifier_hex[..16]);
        let client = state.db_pool.get().await.unwrap();
        client
            .batch_execute(&format!(
                "CREATE FUNCTION {trigger}() RETURNS trigger LANGUAGE plpgsql AS $$ \
                 BEGIN IF NEW.value = '\\x{nullifier_hex}'::bytea THEN \
                 PERFORM pg_sleep(1.5); END IF; RETURN NEW; END $$; \
                 CREATE TRIGGER {trigger} BEFORE INSERT ON nullifier \
                 FOR EACH ROW EXECUTE FUNCTION {trigger}();"
            ))
            .await
            .unwrap();

        let abandoned = tokio::time::timeout(
            Duration::from_millis(500),
            recover_refund(TokenAuth(copy_of(&act)), State(state.clone())),
        )
        .await;
        tokio::time::sleep(Duration::from_millis(2_000)).await;
        client
            .batch_execute(&format!(
                "DROP TRIGGER {trigger} ON nullifier; DROP FUNCTION {trigger}();"
            ))
            .await
            .unwrap();
        assert!(
            abandoned.is_err(),
            "the first call must be the interrupted one"
        );

        let retried = recover_refund(TokenAuth(copy_of(&act)), State(state.clone())).await;
        assert!(
            retried.is_ok(),
            "the retry of an interrupted recovery was refused: {:?}",
            retried.err()
        );
    }

    /// The endpoint's purpose: a spend whose request never arrived comes back
    /// whole — the refund restores every credit the spend withheld — and
    /// asking again answers the same token rather than minting a second one.
    #[tokio::test]
    #[ignore]
    async fn refund_recovery_restores_the_whole_credential_once() {
        use anonymous_credit_tokens::Refund;
        let (state, key_hash) = recovery_state().await;
        let (act, pre_refund) = spend_under(&state, key_hash, None, 1_000, 400).await;

        let first = recover_refund(TokenAuth(copy_of(&act)), State(state.clone()))
            .await
            .expect("recovery of an unspent request");
        let again = recover_refund(TokenAuth(copy_of(&act)), State(state.clone()))
            .await
            .expect("repeated recovery");
        assert_eq!(first.0.refund.refund, again.0.refund.refund);

        let refund =
            Refund::from_cbor(&URL_SAFE_NO_PAD.decode(&first.0.refund.refund).unwrap()).unwrap();
        let cache = state.credential_key_cache.read().await;
        let key = cache.get(&key_hash).unwrap();
        let restored = pre_refund
            .to_credit_token::<128>(
                &key.params,
                &act.spend_proof,
                &refund,
                key.secret_key.public(),
            )
            .expect("refund applies");
        assert_eq!(restored.credits(), Scalar::from(1_000u128));
    }

    // -----------------------------------------------------------------------
    // Key retirement (require running postgres: just db && just db-reset)
    // -----------------------------------------------------------------------

    /// A key that issues for one second and is accepted for one more, so a
    /// test can watch it retire in real time.
    async fn short_lived_key() -> (AppState, [u8; 32]) {
        recovery_state_under(crate::helpers::EpochConfig {
            epoch_duration: Duration::from_secs(1),
            grace_period: Duration::from_secs(1),
        })
        .await
    }

    /// Waits past the key's `accept_until`.
    async fn retire() {
        tokio::time::sleep(Duration::from_millis(2_200)).await;
    }

    /// Retirement is the revocation mechanism (docs/server.md): after
    /// `accept_until` a key's credentials are no longer spendable. The
    /// database lookup enforces that, so the cache must too — a key cached
    /// while valid would otherwise stay spendable for the process lifetime.
    #[tokio::test]
    #[ignore]
    async fn a_retired_key_is_refused_by_a_process_that_cached_it() {
        let (state, key_hash) = short_lived_key().await;
        let (act, _) = spend_under(&state, key_hash, None, 1_000, 10).await;
        crate::handlers::verify_spend_proof(&state, &act)
            .await
            .expect("a live key's spend verifies");

        retire().await;

        let (act, _) = spend_under(&state, key_hash, None, 1_000, 10).await;
        let result = crate::handlers::verify_spend_proof(&state, &act).await;
        assert!(
            matches!(result, Err(ServerError::Unauthorized { .. })),
            "a retired key's credential must be refused, got {result:?}"
        );
    }

    /// `prune_expired_nullifiers` forgets a retired key's spends. That is only
    /// safe if nothing still accepts the key: a cached key would accept every
    /// pruned credential a second time.
    #[tokio::test]
    #[ignore]
    async fn pruning_never_reopens_a_spent_credential() {
        let (state, key_hash) = short_lived_key().await;
        let (act, _) = spend_under(&state, key_hash, None, 1_000, 10).await;
        let key_id = hex::encode(key_hash);
        let nullifier = act.spend_proof.nullifier().as_bytes().to_vec();
        crate::handlers::verify_spend_proof(&state, &act)
            .await
            .expect("first spend verifies");
        assert!(
            db::record_nullifier_with_refund(
                &state.db_pool,
                &key_id,
                &nullifier,
                &[],
                db::RefundStanding::Settled,
            )
            .await
            .unwrap()
        );

        retire().await;
        let client = state.db_pool.get().await.unwrap();
        client
            .execute("SELECT prune_expired_nullifiers()", &[])
            .await
            .unwrap();
        assert_eq!(
            nullifier_rows(&state, &act).await,
            0,
            "the spend was pruned"
        );

        let again = copy_of(&act);
        let result = crate::handlers::verify_spend_proof(&state, &again).await;
        assert!(
            matches!(result, Err(ServerError::Unauthorized { .. })),
            "a pruned spend must not verify again, got {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // A spend whose request never finishes (require running postgres)
    // -----------------------------------------------------------------------

    /// An upstream that accepts a request and never answers it.
    async fn upstream_that_never_answers() -> String {
        use tokio::io::AsyncReadExt;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((mut socket, _)) = listener.accept().await {
                let mut request = [0u8; 8192];
                let _ = socket.read(&mut request).await;
                held.push(socket);
            }
        });
        format!("http://{addr}")
    }

    /// The `Authorization` value the client sends for `act`.
    fn authorization(act: &crate::auth::ActSpend) -> String {
        let mut bytes = ACT_TOKEN_TYPE.to_be_bytes().to_vec();
        bytes.extend_from_slice(&act.challenge_digest);
        bytes.extend_from_slice(&act.issuer_key_hash);
        bytes.extend_from_slice(&act.spend_proof.to_cbor().unwrap());
        format!("PrivateToken token=\"{}\"", URL_SAFE_NO_PAD.encode(bytes))
    }

    /// Recording the nullifier spends the whole credential, and the refund
    /// is its successor. A request that never reaches settlement (the client
    /// gives up on a blocking answer, or the process restarts mid-flight)
    /// must still leave something for `POST /v1/credentials/refund` to return.
    /// Otherwise recovery answers "retry shortly" forever, and the credential's
    /// whole value is gone, not just its hold.
    #[tokio::test]
    #[ignore]
    async fn a_spend_whose_request_never_finishes_is_still_recoverable() {
        let upstream = upstream_that_never_answers().await;
        let (state, key_hash) = recovery_state_at(test_epoch(), Some(upstream)).await;
        let (act, pre_refund) = spend_under(&state, key_hash, None, 1_000_000, 100_000).await;

        let (router, _) = crate::build_router()
            .with_state(state.clone())
            .split_for_parts();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

        // A blocking ask the client abandons, as Ctrl-C on `eidola chat` does.
        let abandoned = reqwest::Client::new()
            .post(format!("http://{addr}/v1/chat/completions"))
            .header("authorization", authorization(&act))
            .json(&serde_json::json!({
                "model": "glm-5-3-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 16,
            }))
            .timeout(Duration::from_millis(500))
            .send()
            .await;
        assert!(abandoned.is_err(), "the upstream never answers");
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(
            nullifier_rows(&state, &act).await,
            1,
            "the credential was spent"
        );

        let recovered = recover_refund(TokenAuth(copy_of(&act)), State(state.clone()))
            .await
            .map(|Json(r)| r.refund);
        let refund = match recovered {
            Ok(refund) => refund,
            Err(e) => panic!("the spent credential must be recoverable, got {e:?}"),
        };
        let cache = state.credential_key_cache.read().await;
        let key = cache.get(&key_hash).unwrap();
        let refund = anonymous_credit_tokens::Refund::from_cbor(
            &URL_SAFE_NO_PAD.decode(refund.refund).unwrap(),
        )
        .unwrap();
        let successor = pre_refund
            .to_credit_token(
                &key.params,
                &act.spend_proof,
                &refund,
                key.secret_key.public(),
            )
            .expect("refund applies");
        // Nothing settled the request, so its hold is charged in full (the
        // stream's own rule for a request whose cost is unknown); the rest of
        // the credential comes back.
        assert_eq!(successor.credits(), Scalar::from(900_000u128));
    }

    /// The provisional refund is final only once nothing can still settle
    /// the spend. While this process has the handler, recovery must say
    /// "retry shortly" as before: a client handed the provisional token now
    /// would spend its one successor on the full hold, and the settled
    /// refund would land on a row nobody reads. Once the client has dropped
    /// the request — and hyper the handler with it — the provisional refund
    /// is the right answer.
    #[tokio::test]
    #[ignore]
    async fn recovery_waits_for_a_spend_that_is_still_settling() {
        let upstream = upstream_that_never_answers().await;
        let (state, key_hash) = recovery_state_at(test_epoch(), Some(upstream)).await;
        let (act, pre_refund) = spend_under(&state, key_hash, None, 1_000_000, 100_000).await;
        let key_id = hex::encode(act.issuer_key_hash);
        let nullifier = act.spend_proof.nullifier().as_bytes().to_vec();

        let (router, _) = crate::build_router()
            .with_state(state.clone())
            .split_for_parts();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

        // A blocking ask that stays in flight: the upstream never answers and
        // this client does not give up on its own.
        let authorization = authorization(&act);
        let in_flight = tokio::spawn(async move {
            reqwest::Client::new()
                .post(format!("http://{addr}/v1/chat/completions"))
                .header("authorization", authorization)
                .json(&serde_json::json!({
                    "model": "glm-5-3-flash",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_completion_tokens": 16,
                }))
                .send()
                .await
        });
        for _ in 0..100 {
            if nullifier_rows(&state, &act).await == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(
            nullifier_rows(&state, &act).await,
            1,
            "the credential was spent"
        );
        assert!(
            state.settling.is_settling(&key_id, &nullifier),
            "the handler holds the settling mark while the request is in flight"
        );

        let early = recover_refund(TokenAuth(copy_of(&act)), State(state.clone())).await;
        match early {
            Err(ServerError::NotFound { .. }) => {}
            other => panic!(
                "recovery must wait for a spend that is still settling, got {:?}",
                other.err()
            ),
        }

        // The client gives up, as Ctrl-C does: its connection closes and hyper
        // drops the handler, which releases the mark. Nothing can settle the
        // spend now, so the provisional refund is served.
        in_flight.abort();
        let _ = in_flight.await;
        for _ in 0..100 {
            if !state.settling.is_settling(&key_id, &nullifier) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            !state.settling.is_settling(&key_id, &nullifier),
            "a dropped handler releases its settling mark"
        );

        let refund = recover_refund(TokenAuth(copy_of(&act)), State(state.clone()))
            .await
            .map(|Json(r)| r.refund)
            .unwrap_or_else(|e| panic!("the abandoned spend must be recoverable, got {e:?}"));
        let cache = state.credential_key_cache.read().await;
        let key = cache.get(&key_hash).unwrap();
        let refund = anonymous_credit_tokens::Refund::from_cbor(
            &URL_SAFE_NO_PAD.decode(refund.refund).unwrap(),
        )
        .unwrap();
        let successor = pre_refund
            .to_credit_token(
                &key.params,
                &act.spend_proof,
                &refund,
                key.secret_key.public(),
            )
            .expect("refund applies");
        assert_eq!(successor.credits(), Scalar::from(900_000u128));
    }

    /// An upstream that answers a stream's headers and then goes quiet, so
    /// the server's stream task sits waiting on bytes that never come.
    async fn upstream_that_opens_a_stream_and_stalls() -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((mut socket, _)) = listener.accept().await {
                let mut request = [0u8; 8192];
                let _ = socket.read(&mut request).await;
                let _ = socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                          transfer-encoding: chunked\r\n\r\n",
                    )
                    .await;
                held.push(socket);
            }
        });
        format!("http://{addr}")
    }

    /// A stream task notices a departed client only when it sends the next
    /// chunk — and an upstream that sent its headers and then stalled never
    /// produces one. Left at that, the task holds the settling mark for as
    /// long as the upstream stays quiet, and recovery answers "retry shortly"
    /// for just as long: the mark would have turned the provisional refund
    /// this branch added back into the lost credential it replaced. The task
    /// must notice the departure on its own and settle the spend.
    #[tokio::test]
    #[ignore]
    async fn a_departed_client_releases_a_stream_the_upstream_left_idle() {
        let upstream = upstream_that_opens_a_stream_and_stalls().await;
        let (state, key_hash) = recovery_state_at(test_epoch(), Some(upstream)).await;
        let (act, _) = spend_under(&state, key_hash, None, 1_000_000, 100_000).await;
        let key_id = hex::encode(act.issuer_key_hash);
        let nullifier = act.spend_proof.nullifier().as_bytes().to_vec();

        let (router, _) = crate::build_router()
            .with_state(state.clone())
            .split_for_parts();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

        // The stream opens as soon as the upstream's headers arrive; the
        // spend is recorded and marked by then.
        let response = reqwest::Client::new()
            .post(format!("http://{addr}/v1/chat/completions"))
            .header("authorization", authorization(&act))
            .json(&serde_json::json!({
                "model": "glm-5-3-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 16,
                "stream": true,
            }))
            .send()
            .await
            .expect("the stream opens once the upstream answers its headers");
        assert!(response.status().is_success(), "{}", response.status());
        assert_eq!(nullifier_rows(&state, &act).await, 1);
        assert!(
            state.settling.is_settling(&key_id, &nullifier),
            "the stream task holds the settling mark while the stream is open"
        );

        // The client leaves without reading a byte. Nothing upstream will
        // ever prompt a send, so only the task watching its receiver can
        // notice.
        drop(response);
        for _ in 0..200 {
            if !state.settling.is_settling(&key_id, &nullifier) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            !state.settling.is_settling(&key_id, &nullifier),
            "the stream task releases its mark once the client is gone"
        );

        // The departed client's spend settled with a zero refund, which is
        // what recovery now serves.
        recover_refund(TokenAuth(copy_of(&act)), State(state.clone()))
            .await
            .map(|Json(r)| r.refund)
            .unwrap_or_else(|e| panic!("the abandoned stream must be recoverable, got {e:?}"));
    }
}
