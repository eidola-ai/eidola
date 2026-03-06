//! Credential issuance: key management and credential endpoints.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use aes_gcm::aead::Aead;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use anonymous_credit_tokens::{IssuanceRequest, Params, PrivateKey, Scalar, credit_to_scalar};
use axum::Json;
use axum::extract::State;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rand_core::{OsRng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use tokio::sync::RwLock;
use tracing::{info, warn};
use utoipa::ToSchema;

use uuid::Uuid;

use crate::AppState;
use crate::auth::BasicAuth;
use crate::db;
use crate::error::ServerError;
use crate::helpers::{EpochConfig, system_time_to_iso_lossy};

// ---------------------------------------------------------------------------
// Key cache types
// ---------------------------------------------------------------------------

/// A decrypted issuer key held in memory.
pub struct DecryptedIssuerKey {
    pub secret_key: PrivateKey,
    pub params: Params,
    pub id: Uuid,
    pub issue_from: SystemTime,
    pub issue_until: SystemTime,
}

/// Thread-safe cache of decrypted issuer keys, keyed by UUID.
pub type KeyCache = Arc<RwLock<HashMap<Uuid, DecryptedIssuerKey>>>;

// ---------------------------------------------------------------------------
// Key encryption / decryption
// ---------------------------------------------------------------------------

/// Derive a per-key AES-256-GCM key from the master key using HKDF-SHA256.
fn derive_key_encryption_key(master_key: &[u8; 32], key_id: &str) -> [u8; 32] {
    let hk = hkdf::Hkdf::<Sha256>::new(Some(key_id.as_bytes()), master_key);
    let mut okm = [0u8; 32];
    hk.expand(b"eidolons-issuer-key-v1", &mut okm)
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
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher
        .encrypt(nonce, plaintext)
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
    let nonce = Nonce::from_slice(nonce_bytes);

    cipher
        .decrypt(nonce, ciphertext)
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
const DS_ORG: &str = "eidolons";
const DS_SERVICE: &str = "inference";
const DS_DEPLOYMENT: &str = "production";
const DS_VERSION: &str = "2026-03-05";

/// The full domain separator string, for storage in the database.
/// Must match what `Params::new(DS_ORG, DS_SERVICE, DS_DEPLOYMENT, DS_VERSION)`
/// produces internally.
fn domain_separator() -> String {
    format!("ACT-v1:{}:{}:{}:{}", DS_ORG, DS_SERVICE, DS_DEPLOYMENT, DS_VERSION)
}

// ---------------------------------------------------------------------------
// Key lifecycle
// ---------------------------------------------------------------------------

/// Decrypt an issuer key row and insert it into the cache. Returns the key UUID.
fn cache_key(
    cache: &mut HashMap<Uuid, DecryptedIssuerKey>,
    master_key: &[u8; 32],
    row: &db::IssuerKeyRow,
) -> Result<Uuid, ServerError> {
    if cache.contains_key(&row.id) {
        return Ok(row.id);
    }
    let id_str = row.id.to_string();
    let plaintext = decrypt_private_key(master_key, &id_str, &row.private_key_enc)?;
    let secret_key = PrivateKey::from_cbor(&plaintext).map_err(|e| {
        ServerError::Internal(format!("failed to decode issuer private key: {}", e))
    })?;
    let params = Params::new(DS_ORG, DS_SERVICE, DS_DEPLOYMENT, DS_VERSION);
    cache.insert(
        row.id,
        DecryptedIssuerKey {
            secret_key,
            params,
            id: row.id,
            issue_from: row.issue_from,
            issue_until: row.issue_until,
        },
    );
    Ok(row.id)
}

/// Generate a new issuer key pair and build a row ready for insertion.
fn generate_key(
    master_key: &[u8; 32],
    issue_from: SystemTime,
    epoch_config: &EpochConfig,
) -> Result<db::IssuerKeyRow, ServerError> {
    let key_id = Uuid::new_v4();
    let secret_key = PrivateKey::random(OsRng);
    let public_key_cbor = secret_key
        .public()
        .to_cbor()
        .map_err(|e| ServerError::Internal(format!("failed to encode public key: {}", e)))?;
    let private_key_cbor = secret_key
        .to_cbor()
        .map_err(|e| ServerError::Internal(format!("failed to encode private key: {}", e)))?;
    let id_str = key_id.to_string();
    let encrypted = encrypt_private_key(master_key, &id_str, &private_key_cbor)?;

    let (issue_until, accept_until) = epoch_config.boundaries_from(issue_from);

    Ok(db::IssuerKeyRow {
        id: key_id,
        private_key_enc: encrypted,
        public_key: public_key_cbor,
        domain_separator: domain_separator(),
        issue_from,
        issue_until,
        accept_until,
    })
}

/// Ensure that both a current key and the next key exist in the database
/// and are cached in memory. Returns the current key's UUID (for issuance).
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
) -> Result<Uuid, ServerError> {
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
                return Ok(k.id);
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
                info!("provisioned issuer key {} (issue_from={})",
                    inserted.id,
                    system_time_to_iso_lossy(inserted.issue_from));
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
    let rows = db::get_valid_issuer_keys(pool).await?;
    let mut cache = key_cache.write().await;
    let mut current_id = None;
    for row in &rows {
        cache_key(&mut cache, master_key, row)?;
        if row.issue_from <= now && row.issue_until > now {
            current_id = Some(row.id);
        }
    }

    current_id.ok_or_else(|| {
        ServerError::Internal("no current issuer key found after provisioning".to_string())
    })
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

            match ensure_keys(&key_cache, &master_key, &pool, &epoch_config).await {
                Ok(key_id) => {
                    info!("periodic key check: current key {}", key_id);
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
    /// Unique key identifier (UUID).
    pub id: String,
    /// Base64-encoded CBOR public key (compressed Ristretto255 point).
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
    /// Issuer key identifier (UUID).
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
            id: r.id.to_string(),
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
        (status = 503, description = "Credential issuance not configured", body = crate::types::ErrorResponse),
    )
)]
pub async fn issue_credentials(
    BasicAuth(account_id): BasicAuth,
    State(state): State<AppState>,
    Json(request): Json<IssueCredentialsRequest>,
) -> Result<Json<IssueCredentialsResponse>, ServerError> {
    let master_key = state.credential_master_key.as_ref().ok_or_else(|| {
        ServerError::ServiceUnavailable("credential issuance is not configured".to_string())
    })?;

    if request.credits <= 0 {
        return Err(ServerError::BadRequest {
            message: "credits must be greater than 0".to_string(),
        });
    }

    let issuance_cbor = URL_SAFE_NO_PAD
        .decode(&request.issuance_request)
        .map_err(|e| ServerError::BadRequest {
            message: format!("invalid base64 in issuance_request: {}", e),
        })?;

    let issuance_request =
        IssuanceRequest::from_cbor(&issuance_cbor).map_err(|e| ServerError::BadRequest {
            message: format!("invalid CBOR issuance request: {}", e),
        })?;

    let credit_scalar =
        credit_to_scalar::<128>(request.credits as u128).map_err(|e| ServerError::BadRequest {
            message: format!("invalid credit amount: {}", e),
        })?;

    let key_id = ensure_keys(
        &state.credential_key_cache,
        master_key,
        &state.db_pool,
        &state.epoch_config,
    )
    .await?;

    let ledger_entry_id =
        match db::insert_credential_issuance(&state.db_pool, account_id, request.credits, key_id)
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

    let issuance_response = {
        let cache = state.credential_key_cache.read().await;
        let key = cache.get(&key_id).ok_or_else(|| {
            ServerError::Internal("current key evicted from cache unexpectedly".to_string())
        })?;
        key.secret_key
            .issue::<128>(
                &key.params,
                &issuance_request,
                credit_scalar,
                Scalar::ZERO,
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

    info!(
        "issued credential: account={}, credits={}, ledger_entry={}",
        account_id, request.credits, ledger_entry_id
    );

    Ok(Json(IssueCredentialsResponse {
        issuance_response: URL_SAFE_NO_PAD.encode(&response_cbor),
        issuer_key_id: key_id.to_string(),
        credits: request.credits,
        ledger_entry_id: ledger_entry_id.to_string(),
    }))
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
        assert_eq!(domain_separator(), "ACT-v1:eidolons:inference:production:2026-03-05");
    }

    #[test]
    fn test_params_does_not_panic() {
        // Verify Params::new accepts our domain separator components (no colons).
        let _params = Params::new(DS_ORG, DS_SERVICE, DS_DEPLOYMENT, DS_VERSION);
    }
}
