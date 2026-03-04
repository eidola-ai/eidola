//! ACT (Anonymous Credit Token) issuance: key management and token endpoints.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aes_gcm::aead::Aead;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use anonymous_credit_tokens::{IssuanceRequest, Params, PrivateKey, Scalar, credit_to_scalar};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use bytes::Bytes;
use hkdf::Hkdf;
use http_body_util::combinators::BoxBody;
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use rand_core::{OsRng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use tokio::sync::RwLock;
use tracing::{info, warn};
use utoipa::ToSchema;
use uuid::Uuid;

use crate::db;
use crate::error::ServerError;

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
    let hk = Hkdf::<Sha256>::new(Some(epoch.as_bytes()), master_key);
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
// Epoch computation (Hinnant civil calendar algorithm)
// ---------------------------------------------------------------------------

/// Decompose a unix timestamp (seconds) into (year, month, day).
fn civil_from_unix(secs: i64) -> (i64, u64, u64) {
    let days_since_epoch = secs.div_euclid(86_400);
    let z = days_since_epoch + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Convert (year, month, day) to days since 1970-01-01 (inverse Hinnant).
fn days_from_civil(y: i64, m: u64, d: u64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400) as u64;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe as i64 - 719468
}

/// Compute the current epoch string "YYYY-MM" from the system clock.
fn current_epoch() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_secs() as i64;
    let (y, m, _) = civil_from_unix(secs);
    format!("{:04}-{:02}", y, m)
}

/// Given a "YYYY-MM" epoch string, compute (valid_from, valid_until, accept_until).
fn epoch_boundaries(epoch: &str) -> Result<(SystemTime, SystemTime, SystemTime), ServerError> {
    let parts: Vec<&str> = epoch.split('-').collect();
    if parts.len() != 2 {
        return Err(ServerError::BadRequest {
            message: "epoch must be YYYY-MM format".to_string(),
        });
    }
    let y: i64 = parts[0].parse().map_err(|_| ServerError::BadRequest {
        message: "invalid epoch year".to_string(),
    })?;
    let m: u64 = parts[1].parse().map_err(|_| ServerError::BadRequest {
        message: "invalid epoch month".to_string(),
    })?;
    if !(1..=12).contains(&m) {
        return Err(ServerError::BadRequest {
            message: "epoch month must be 1-12".to_string(),
        });
    }

    let valid_from_days = days_from_civil(y, m, 1);
    let valid_from = UNIX_EPOCH + Duration::from_secs(valid_from_days as u64 * 86400);

    // Next month
    let (ny, nm) = if m == 12 { (y + 1, 1u64) } else { (y, m + 1) };
    let valid_until_days = days_from_civil(ny, nm, 1);
    let valid_until = UNIX_EPOCH + Duration::from_secs(valid_until_days as u64 * 86400);

    // Grace period: +3 days
    let accept_until = valid_until + Duration::from_secs(3 * 86400);

    Ok((valid_from, valid_until, accept_until))
}

/// Format a SystemTime as ISO 8601.
fn system_time_to_iso(t: SystemTime) -> String {
    let secs = t
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or_else(|e| -(e.duration().as_secs() as i64));
    let (y, m, d) = civil_from_unix(secs);
    let tod = secs.rem_euclid(86_400) as u64;
    let (hour, min, sec) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}Z")
}

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
        // Another instance won the race — load the winning row.
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

/// Request body for `POST /v1/account/tokens`.
#[derive(Deserialize, ToSchema)]
pub struct IssueTokensRequest {
    /// Base64-encoded CBOR `IssuanceRequest` from the client.
    pub issuance_request: String,
    /// Number of micro-dollar credits to load into the token. Must be > 0.
    pub credits: i64,
}

/// Response for `POST /v1/account/tokens`.
#[derive(Serialize, ToSchema)]
pub struct IssueTokensResponse {
    /// Base64-encoded CBOR `IssuanceResponse`.
    pub issuance_response: String,
    /// The epoch of the issuer key used.
    pub epoch: String,
    /// Credits loaded into the token.
    pub credits: i64,
    /// The ledger entry ID for this issuance.
    pub ledger_entry_id: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `GET /v1/keys` — list valid issuer public keys (unauthenticated).
pub async fn handle_list_keys(
    pool: &deadpool_postgres::Pool,
) -> Response<BoxBody<Bytes, Infallible>> {
    let rows = match db::get_valid_issuer_keys(pool).await {
        Ok(r) => r,
        Err(e) => return error_response(&e),
    };

    let data: Vec<IssuerKeyResponse> = rows
        .into_iter()
        .map(|r| IssuerKeyResponse {
            epoch: r.epoch,
            public_key: URL_SAFE_NO_PAD.encode(&r.public_key),
            domain_separator: r.domain_separator,
            valid_from: system_time_to_iso(r.valid_from),
            valid_until: system_time_to_iso(r.valid_until),
            accept_until: system_time_to_iso(r.accept_until),
        })
        .collect();

    let body = serde_json::to_string(&ListKeysResponse { data }).unwrap();
    json_response(StatusCode::OK, &body)
}

/// `POST /v1/account/tokens` — issue anonymous credit tokens (authenticated).
pub async fn handle_issue_tokens(
    req: Request<Incoming>,
    pool: &deadpool_postgres::Pool,
    key_cache: &KeyCache,
    master_key: &[u8; 32],
    account_id: Uuid,
) -> Response<BoxBody<Bytes, Infallible>> {
    // Parse request body.
    let body_bytes = match http_body_util::BodyExt::collect(req.into_body()).await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            return error_response(&ServerError::BadRequest {
                message: format!("failed to read request body: {}", e),
            });
        }
    };
    let request: IssueTokensRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return error_response(&ServerError::BadRequest {
                message: format!("invalid JSON: {}", e),
            });
        }
    };

    // Validate credits.
    if request.credits <= 0 {
        return error_response(&ServerError::BadRequest {
            message: "credits must be greater than 0".to_string(),
        });
    }

    // Decode the issuance request.
    let issuance_cbor = match URL_SAFE_NO_PAD.decode(&request.issuance_request) {
        Ok(b) => b,
        Err(e) => {
            return error_response(&ServerError::BadRequest {
                message: format!("invalid base64 in issuance_request: {}", e),
            });
        }
    };
    let issuance_request = match IssuanceRequest::from_cbor(&issuance_cbor) {
        Ok(r) => r,
        Err(e) => {
            return error_response(&ServerError::BadRequest {
                message: format!("invalid CBOR issuance request: {}", e),
            });
        }
    };

    // Convert credits to Scalar.
    let credit_scalar = match credit_to_scalar::<128>(request.credits as u128) {
        Ok(s) => s,
        Err(e) => {
            return error_response(&ServerError::BadRequest {
                message: format!("invalid credit amount: {}", e),
            });
        }
    };

    // Ensure the current epoch key is ready.
    let epoch = match ensure_current_epoch_key(key_cache, master_key, pool).await {
        Ok(e) => e,
        Err(e) => return error_response(&e),
    };

    // Atomically debit the balance.
    let ledger_entry_id =
        match db::insert_act_issuance(pool, account_id, request.credits, &epoch).await {
            Ok(Some(id)) => id,
            Ok(None) => {
                // Insufficient balance — fetch the actual balance for the error response.
                let available = db::get_available_balance(pool, account_id)
                    .await
                    .unwrap_or(0);
                return error_response(&ServerError::PaymentRequired {
                    message: "insufficient balance".to_string(),
                    available,
                });
            }
            Err(e) => return error_response(&e),
        };

    // Issue the token.
    let issuance_response = {
        let cache = key_cache.read().await;
        let key = match cache.get(&epoch) {
            Some(k) => k,
            None => {
                return error_response(&ServerError::Internal(
                    "epoch key evicted from cache unexpectedly".to_string(),
                ));
            }
        };
        match key.secret_key.issue::<128>(
            &key.params,
            &issuance_request,
            credit_scalar,
            Scalar::ZERO,
            OsRng,
        ) {
            Ok(resp) => resp,
            Err(e) => {
                warn!("ACT issuance failed: {}", e);
                // The debit already happened — in a production system we'd refund.
                // For PoC, log and return 400.
                return error_response(&ServerError::BadRequest {
                    message: format!("issuance failed: {}", e),
                });
            }
        }
    };

    // Encode the response.
    let response_cbor = match issuance_response.to_cbor() {
        Ok(b) => b,
        Err(e) => {
            return error_response(&ServerError::Internal(format!(
                "failed to encode issuance response: {}",
                e
            )));
        }
    };

    let body = serde_json::to_string(&IssueTokensResponse {
        issuance_response: URL_SAFE_NO_PAD.encode(&response_cbor),
        epoch,
        credits: request.credits,
        ledger_entry_id: ledger_entry_id.to_string(),
    })
    .unwrap();

    info!(
        "issued ACT: account={}, credits={}, ledger_entry={}",
        account_id, request.credits, ledger_entry_id
    );
    json_response(StatusCode::OK, &body)
}

// ---------------------------------------------------------------------------
// Helpers (duplicated from main.rs to keep the module self-contained)
// ---------------------------------------------------------------------------

fn json_response(status: StatusCode, body: &str) -> Response<BoxBody<Bytes, Infallible>> {
    use http_body_util::BodyExt;
    use http_body_util::Full;

    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(
            Full::new(Bytes::from(body.to_string()))
                .map_err(|e| match e {})
                .boxed(),
        )
        .unwrap()
}

fn error_response(err: &ServerError) -> Response<BoxBody<Bytes, Infallible>> {
    let body = serde_json::to_string(&err.to_error_response()).unwrap();
    json_response(err.status_code(), &body)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_current_epoch_format() {
        let epoch = current_epoch();
        assert_eq!(epoch.len(), 7);
        assert_eq!(epoch.as_bytes()[4], b'-');
        let year: i64 = epoch[..4].parse().unwrap();
        let month: u64 = epoch[5..].parse().unwrap();
        assert!(year >= 2024);
        assert!((1..=12).contains(&month));
    }

    #[test]
    fn test_epoch_boundaries() {
        let (from, until, accept) = epoch_boundaries("2026-03").unwrap();
        assert_eq!(system_time_to_iso(from), "2026-03-01T00:00:00Z");
        assert_eq!(system_time_to_iso(until), "2026-04-01T00:00:00Z");
        assert_eq!(system_time_to_iso(accept), "2026-04-04T00:00:00Z");
    }

    #[test]
    fn test_epoch_boundaries_december_wraps() {
        let (from, until, accept) = epoch_boundaries("2025-12").unwrap();
        assert_eq!(system_time_to_iso(from), "2025-12-01T00:00:00Z");
        assert_eq!(system_time_to_iso(until), "2026-01-01T00:00:00Z");
        assert_eq!(system_time_to_iso(accept), "2026-01-04T00:00:00Z");
    }

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
        // 2026-03-04 = known date
        let days = days_from_civil(2026, 3, 4);
        let secs = days * 86400;
        let (y, m, d) = civil_from_unix(secs);
        assert_eq!((y, m, d), (2026, 3, 4));
    }

    #[test]
    fn test_invalid_epoch_format() {
        assert!(epoch_boundaries("2026").is_err());
        assert!(epoch_boundaries("2026-13").is_err());
        assert!(epoch_boundaries("2026-00").is_err());
    }
}
