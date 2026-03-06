//! Credential issuance: key management and credential endpoints.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

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

use crate::AppState;
use crate::auth::BasicAuth;
use crate::db;
use crate::error::ServerError;
use crate::helpers::{current_epoch, epoch_boundaries, system_time_to_iso_lossy};

// ---------------------------------------------------------------------------
// Key cache types
// ---------------------------------------------------------------------------

/// A decrypted issuer key held in memory.
pub struct DecryptedIssuerKey {
    pub secret_key: PrivateKey,
    pub params: Params,
    pub epoch: String,
    pub valid_until: SystemTime,
}

/// Thread-safe cache of decrypted issuer keys, keyed by epoch string.
pub type KeyCache = Arc<RwLock<HashMap<String, DecryptedIssuerKey>>>;

// ---------------------------------------------------------------------------
// Key encryption / decryption
// ---------------------------------------------------------------------------

/// Derive a per-epoch AES-256-GCM key from the master key using HKDF-SHA256.
fn derive_epoch_key(master_key: &[u8; 32], epoch: &str) -> [u8; 32] {
    let hk = hkdf::Hkdf::<Sha256>::new(Some(epoch.as_bytes()), master_key);
    let mut okm = [0u8; 32];
    hk.expand(b"eidolons-issuer-key-v1", &mut okm)
        .expect("32 bytes is a valid HKDF-SHA256 output length");
    okm
}

/// Encrypt a private key with AES-256-GCM.
/// Output format: `nonce (12 bytes) || ciphertext+tag`.
fn encrypt_private_key(
    master_key: &[u8; 32],
    epoch: &str,
    plaintext: &[u8],
) -> Result<Vec<u8>, ServerError> {
    let aes_key = derive_epoch_key(master_key, epoch);
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
    epoch: &str,
    encrypted: &[u8],
) -> Result<Vec<u8>, ServerError> {
    if encrypted.len() < 12 {
        return Err(ServerError::Internal("encrypted key too short".to_string()));
    }
    let (nonce_bytes, ciphertext) = encrypted.split_at(12);
    let aes_key = derive_epoch_key(master_key, epoch);
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

/// Build the domain separator string for an epoch.
fn domain_separator(epoch: &str) -> String {
    format!("ACT-v1:eidolons:inference:production:{}", epoch)
}

// ---------------------------------------------------------------------------
// Key lifecycle
// ---------------------------------------------------------------------------

/// Ensure a key exists for the current epoch. Returns the epoch string.
///
/// 1. Check in-memory cache.
/// 2. Check database (decrypt + cache if found).
/// 3. Generate new key → encrypt → insert → cache.
pub async fn ensure_current_epoch_key(
    key_cache: &KeyCache,
    master_key: &[u8; 32],
    pool: &deadpool_postgres::Pool,
) -> Result<String, ServerError> {
    let epoch = current_epoch();

    // Fast path: already cached.
    {
        let cache = key_cache.read().await;
        if cache.contains_key(&epoch) {
            return Ok(epoch);
        }
    }

    // Check the database.
    if let Some(row) = db::get_issuer_key_by_epoch(pool, &epoch).await? {
        let plaintext = decrypt_private_key(master_key, &epoch, &row.private_key_enc)?;
        let secret_key = PrivateKey::from_cbor(&plaintext).map_err(|e| {
            ServerError::Internal(format!("failed to decode issuer private key: {}", e))
        })?;
        let params = Params::new("eidolons", "inference", "production", &epoch);
        let mut cache = key_cache.write().await;
        cache.entry(epoch.clone()).or_insert(DecryptedIssuerKey {
            secret_key,
            params,
            epoch: epoch.clone(),
            valid_until: row.valid_until,
        });
        return Ok(epoch);
    }

    // Generate a new key pair.
    let secret_key = PrivateKey::random(OsRng);
    let public_key_cbor = secret_key
        .public()
        .to_cbor()
        .map_err(|e| ServerError::Internal(format!("failed to encode public key: {}", e)))?;
    let private_key_cbor = secret_key
        .to_cbor()
        .map_err(|e| ServerError::Internal(format!("failed to encode private key: {}", e)))?;
    let encrypted = encrypt_private_key(master_key, &epoch, &private_key_cbor)?;

    let (valid_from, valid_until, accept_until) = epoch_boundaries(&epoch)?;
    let ds = domain_separator(&epoch);

    let row_to_insert = db::IssuerKeyRow {
        epoch: epoch.clone(),
        private_key_enc: encrypted,
        public_key: public_key_cbor,
        domain_separator: ds,
        valid_from,
        valid_until,
        accept_until,
    };
    let inserted = db::insert_issuer_key(pool, &row_to_insert).await?;

    if !inserted {
        info!(
            "issuer key for epoch {} was created concurrently, loading winner",
            epoch
        );
        let row = db::get_issuer_key_by_epoch(pool, &epoch)
            .await?
            .ok_or_else(|| {
                ServerError::Internal("issuer key vanished after concurrent insert".to_string())
            })?;
        let plaintext = decrypt_private_key(master_key, &epoch, &row.private_key_enc)?;
        let secret_key = PrivateKey::from_cbor(&plaintext).map_err(|e| {
            ServerError::Internal(format!("failed to decode issuer private key: {}", e))
        })?;
        let params = Params::new("eidolons", "inference", "production", &epoch);
        let mut cache = key_cache.write().await;
        cache.entry(epoch.clone()).or_insert(DecryptedIssuerKey {
            secret_key,
            params,
            epoch: epoch.clone(),
            valid_until: row.valid_until,
        });
        return Ok(epoch);
    }

    info!("generated new issuer key for epoch {}", epoch);
    let params = Params::new("eidolons", "inference", "production", &epoch);
    let mut cache = key_cache.write().await;
    cache.entry(epoch.clone()).or_insert(DecryptedIssuerKey {
        secret_key,
        params,
        epoch: epoch.clone(),
        valid_until,
    });

    Ok(epoch)
}

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

/// A single issuer public key in the `GET /v1/keys` response.
#[derive(Serialize, ToSchema)]
pub struct IssuerKeyResponse {
    /// Epoch identifier (YYYY-MM).
    pub epoch: String,
    /// Base64-encoded CBOR public key (compressed Ristretto255 point).
    pub public_key: String,
    /// Domain separator used for parameter generation.
    pub domain_separator: String,
    /// Start of the issuance window (ISO 8601).
    pub valid_from: String,
    /// End of the issuance window (ISO 8601).
    pub valid_until: String,
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
    /// Epoch identifier (YYYY-MM).
    pub epoch: String,
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
            epoch: r.epoch,
            public_key: URL_SAFE_NO_PAD.encode(&r.public_key),
            domain_separator: r.domain_separator,
            valid_from: system_time_to_iso_lossy(r.valid_from),
            valid_until: system_time_to_iso_lossy(r.valid_until),
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

    let epoch =
        ensure_current_epoch_key(&state.credential_key_cache, master_key, &state.db_pool).await?;

    let ledger_entry_id = match db::insert_credential_issuance(
        &state.db_pool,
        account_id,
        request.credits,
        &epoch,
    )
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
        let key = cache.get(&epoch).ok_or_else(|| {
            ServerError::Internal("epoch key evicted from cache unexpectedly".to_string())
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
        epoch,
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
    use crate::helpers::{civil_from_unix, days_from_civil};

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let master_key = [42u8; 32];
        let epoch = "2026-03";
        let plaintext = b"test private key material";
        let encrypted = encrypt_private_key(&master_key, epoch, plaintext).unwrap();
        let decrypted = decrypt_private_key(&master_key, epoch, &encrypted).unwrap();
        assert_eq!(plaintext.as_slice(), &decrypted);
    }

    #[test]
    fn test_decrypt_wrong_epoch_fails() {
        let master_key = [42u8; 32];
        let encrypted = encrypt_private_key(&master_key, "2026-03", b"secret").unwrap();
        let result = decrypt_private_key(&master_key, "2026-04", &encrypted);
        assert!(result.is_err());
    }

    #[test]
    fn test_domain_separator_format() {
        assert_eq!(
            domain_separator("2026-03"),
            "ACT-v1:eidolons:inference:production:2026-03"
        );
    }

    #[test]
    fn test_civil_roundtrip() {
        let days = days_from_civil(2026, 3, 4);
        let secs = days * 86400;
        let (y, m, d) = civil_from_unix(secs);
        assert_eq!((y, m, d), (2026, 3, 4));
    }
}
