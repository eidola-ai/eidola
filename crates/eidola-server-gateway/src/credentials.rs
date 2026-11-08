//! Credential issuance: key management and credential endpoints.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use aes_gcm::aead::{Aead, Payload};
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
use crate::auth::{BasicAuth, RefundKey, TokenAuth};
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
// Refund-token sealing
// ---------------------------------------------------------------------------

/// HKDF info for the refund-token key, domain-separated from the issuer-key
/// wrapping key (`eidola-issuer-key-v1`) derived from the same master key.
const REFUND_TOKEN_KEY_INFO: &[u8] = b"eidola-refund-token-v1";

/// First byte of every sealed refund token. A CBOR `RefundMsg` (a five-entry
/// map) begins with `0xa5`, so a bare token is told apart by this byte as
/// well as by failing to open.
const SEALED_REFUND_VERSION: u8 = 0x01;

const SEALED_REFUND_NONCE_LEN: usize = 12;

/// A refund token as it is stored in `nullifier.refund_token`: AES-256-GCM
/// under a key that needs both the client's refund recovery key and the
/// enclave's master key, bound to its row.
///
/// The CBOR `RefundMsg` carries the refund amount `t` as a plain scalar, and
/// `t` is the hold minus the actual charge: a function of the request's
/// content lengths. Postgres is outside the enclave and assumed adversarial,
/// so the token is never written there in the clear, and what it is sealed
/// under is not something the gateway keeps: the client's key arrives with
/// the spend and again with a recovery, and lives only in enclave memory in
/// between. The database layer accepts and returns only this type, and
/// [`seal_refund_token`] is the only way to make one from a token.
///
/// Layout: `version (1) || nonce (12) || ciphertext || tag (16)`. The
/// ciphertext is exactly as long as the CBOR, which is fixed-size (four
/// 32-byte scalars and a 32-byte point), so the length says nothing either.
pub struct SealedRefundToken(Vec<u8>);

impl SealedRefundToken {
    /// Wrap bytes read back from the database. Nothing is checked here:
    /// [`open_refund_token`] is what decides whether they are a sealed token.
    pub(crate) fn from_stored(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// The key one spend's refund is sealed under: HKDF-SHA256 over the client's
/// refund recovery key and the master key together, salted with the issuer
/// key id.
///
/// The client's key is what keeps the stored amount from anyone without the
/// client, the gateway's own measured secrets included. The master key is
/// mixed in as well so a client that sends a weak or shared key (a bug, or a
/// third-party implementation) still leaves the stored copy sealed against a
/// reader of the database alone. Each spend's client key is its own, so the
/// random-nonce count under any one AES key is one or two.
fn derive_refund_token_key(
    master_key: &[u8; 32],
    refund_key: &RefundKey,
    issuer_key_id: &str,
) -> zeroize::Zeroizing<[u8; 32]> {
    let mut ikm = zeroize::Zeroizing::new([0u8; 64]);
    ikm[..32].copy_from_slice(refund_key.as_bytes());
    ikm[32..].copy_from_slice(master_key);
    let hk = hkdf::Hkdf::<Sha256>::new(Some(issuer_key_id.as_bytes()), ikm.as_ref());
    let mut okm = zeroize::Zeroizing::new([0u8; 32]);
    hk.expand(REFUND_TOKEN_KEY_INFO, okm.as_mut())
        .expect("32 bytes is a valid HKDF-SHA256 output length");
    okm
}

/// Associated data binding a sealed token to its row: the format version,
/// then the issuer key id (length-prefixed), then the nullifier. A token
/// copied onto another row (another nullifier, or another key's) fails to
/// open there.
fn refund_token_aad(issuer_key_id: &str, nullifier: &[u8]) -> Vec<u8> {
    let id = issuer_key_id.as_bytes();
    let mut aad = Vec::with_capacity(1 + 2 + id.len() + nullifier.len());
    aad.push(SEALED_REFUND_VERSION);
    aad.extend_from_slice(&(id.len() as u16).to_be_bytes());
    aad.extend_from_slice(id);
    aad.extend_from_slice(nullifier);
    aad
}

/// Seal a CBOR refund token for storage on the nullifier row
/// `(issuer_key_id, nullifier)`.
///
/// The nonce is random rather than derived from the nullifier: the row is
/// written once per spend today, but a random nonce stays safe if a path ever
/// writes a row twice, where a derived one would repeat a nonce under a key.
pub fn seal_refund_token(
    master_key: &[u8; 32],
    refund_key: &RefundKey,
    issuer_key_id: &str,
    nullifier: &[u8],
    refund_cbor: &[u8],
) -> Result<SealedRefundToken, ServerError> {
    let key = derive_refund_token_key(master_key, refund_key, issuer_key_id);
    let cipher = Aes256Gcm::new_from_slice(key.as_ref())
        .map_err(|e| ServerError::Internal(format!("AES key init failed: {}", e)))?;

    let mut nonce_bytes = [0u8; SEALED_REFUND_NONCE_LEN];
    rand_core::OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from(nonce_bytes);
    let aad = refund_token_aad(issuer_key_id, nullifier);

    let ciphertext = cipher
        .encrypt(
            &nonce,
            Payload {
                msg: refund_cbor,
                aad: &aad,
            },
        )
        .map_err(|_| ServerError::Internal("refund token sealing failed".to_string()))?;

    let mut out = Vec::with_capacity(1 + SEALED_REFUND_NONCE_LEN + ciphertext.len());
    out.push(SEALED_REFUND_VERSION);
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(SealedRefundToken(out))
}

/// Open a stored refund token for the row `(issuer_key_id, nullifier)`.
///
/// `None` for anything that is not a token sealed for this row under this
/// client key and this master key: a recovery presenting another key, a
/// token moved from another row, a bare token, or tampered bytes. The caller
/// treats that exactly as a row with no token stored.
pub fn open_refund_token(
    master_key: &[u8; 32],
    refund_key: &RefundKey,
    issuer_key_id: &str,
    nullifier: &[u8],
    sealed: &SealedRefundToken,
) -> Option<Vec<u8>> {
    let bytes = sealed.as_bytes();
    let (&version, rest) = bytes.split_first()?;
    if version != SEALED_REFUND_VERSION || rest.len() < SEALED_REFUND_NONCE_LEN {
        return None;
    }
    let (nonce_bytes, ciphertext) = rest.split_at(SEALED_REFUND_NONCE_LEN);
    let nonce = Nonce::try_from(nonce_bytes).ok()?;
    let key = derive_refund_token_key(master_key, refund_key, issuer_key_id);
    let cipher = Aes256Gcm::new_from_slice(key.as_ref()).ok()?;
    let aad = refund_token_aad(issuer_key_id, nullifier);
    cipher
        .decrypt(
            &nonce,
            Payload {
                msg: ciphertext,
                aad: &aad,
            },
        )
        .ok()
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
    // Fast path: already in cache
    if key_cache.read().await.contains_key(key_hash) {
        return Ok(());
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
/// the server verifies the spend proof, records the nullifier, issues a
/// full refund (no credits consumed), stores it, and returns it.
///
/// In both cases, the client gets a refund token it can combine with its
/// local `PreRefund` state to reconstruct a credential.
///
/// The stored token is sealed under the refund recovery key the client sent
/// with the spend, so recovery must present the same key: a different one
/// opens nothing and is answered as no token stored (404).
#[utoipa::path(
    post,
    path = "/v1/credentials/refund",
    tag = "Unlinked",
    params(
        ("x-eidola-refund-key" = String, Header, description = "The refund recovery key sent with the spend: 64 lowercase hex digits")
    ),
    responses(
        (status = 200, description = "Refund token recovered or issued", body = RefundRecoveryResponse),
        (status = 400, description = "Missing or malformed refund recovery key", body = crate::types::ErrorResponse),
        (status = 401, description = "Invalid spend proof", body = crate::types::ErrorResponse),
        (status = 404, description = "Nullifier found but no refund token is available for this key yet", body = crate::types::ErrorResponse),
        (status = 500, description = "Internal error", body = crate::types::ErrorResponse)
    )
)]
pub async fn recover_refund(
    TokenAuth(act): TokenAuth,
    State(state): State<AppState>,
) -> Result<Json<RefundRecoveryResponse>, ServerError> {
    let master_key = &state.credential_master_key;

    // Verify the challenge_digest matches our expected TokenChallenge.
    let expected_digest = compute_challenge_digest();
    if act.challenge_digest != expected_digest {
        return Err(ServerError::Unauthorized {
            message: "invalid challenge_digest in token".to_string(),
        });
    }

    // Load the issuer key.
    load_key_for_spending(
        &state.credential_key_cache,
        master_key,
        &state.db_pool,
        &act.issuer_key_hash,
    )
    .await?;

    let key_id = hex::encode(act.issuer_key_hash);
    let nullifier_bytes = act.spend_proof.nullifier().as_bytes().to_vec();

    // Check if the nullifier was already recorded (original request arrived).
    if let Some(sealed) = db::get_refund_token(&state.db_pool, &key_id, &nullifier_bytes).await? {
        // Nullifier exists and has a stored refund token — return it. A
        // token that does not open for this row under the presented refund
        // key (another key, moved from another row, or altered) is no token:
        // the row exists, so this is the same permanent 404 a client gets
        // when the best-effort store never happened, and the client settles
        // the hold the way it does then.
        let Some(refund_cbor) = open_refund_token(
            master_key,
            &act.refund_key,
            &key_id,
            &nullifier_bytes,
            &sealed,
        ) else {
            warn!("stored refund token did not open; answering as if none were stored");
            return Err(ServerError::NotFound {
                message: "no refund token is stored for this credential".to_string(),
            });
        };
        return Ok(Json(RefundRecoveryResponse {
            refund: RefundInfo {
                refund: URL_SAFE_NO_PAD.encode(&refund_cbor),
                issuer_key_id: key_id,
            },
        }));
    }

    // Either the nullifier doesn't exist (request never arrived) or it exists
    // but the refund token hasn't been stored yet (race with in-flight request).
    // Try to record the nullifier — if it succeeds, we own it.
    let recorded = db::record_nullifier(&state.db_pool, &key_id, &nullifier_bytes).await?;

    if !recorded {
        // Nullifier exists but no refund token stored yet. The original
        // request is likely still in flight. The client should retry shortly.
        return Err(ServerError::NotFound {
            message: "nullifier recorded but refund token not yet available — \
                      the original request may still be in flight, retry shortly"
                .to_string(),
        });
    }

    // Nullifier was not previously recorded — the original request never
    // arrived. Verify the spend proof and issue a full refund.
    {
        let cache = state.credential_key_cache.read().await;
        let key = cache.get(&act.issuer_key_hash).ok_or_else(|| {
            ServerError::Internal("issuer key evicted from cache unexpectedly".to_string())
        })?;

        if act.spend_proof.context() != key.request_context_scalar {
            return Err(ServerError::Unauthorized {
                message: "invalid request_context in spend proof".to_string(),
            });
        }

        key.secret_key
            .refund::<128>(&key.params, &act.spend_proof, Scalar::ZERO, OsRng)
            .map_err(|_| ServerError::Unauthorized {
                message: "invalid spend proof".to_string(),
            })?;
    }

    // Issue a full refund (charge_credits returned in full since no work was done).
    let charge_credits = scalar_to_credit::<128>(&act.spend_proof.charge()).map_err(|_| {
        ServerError::BadRequest {
            message: "invalid charge amount in spend proof".to_string(),
        }
    })?;

    let t = credit_to_scalar::<128>(charge_credits)
        .map_err(|e| ServerError::Internal(format!("invalid refund amount: {e:?}")))?;

    let refund_cbor = {
        let cache = state.credential_key_cache.read().await;
        let key = cache.get(&act.issuer_key_hash).ok_or_else(|| {
            ServerError::Internal("issuer key not in cache for refund".to_string())
        })?;

        let refund = key
            .secret_key
            .refund(&key.params, &act.spend_proof, t, OsRng)
            .map_err(|e| ServerError::Internal(format!("refund issuance failed: {e:?}")))?;

        refund
            .to_cbor()
            .map_err(|e| ServerError::Internal(format!("refund CBOR encoding failed: {e:?}")))?
    };

    // Store the refund token for idempotency, sealed (it carries the amount).
    let sealed = seal_refund_token(
        master_key,
        &act.refund_key,
        &key_id,
        &nullifier_bytes,
        &refund_cbor,
    )?;
    db::store_refund_token(&state.db_pool, &key_id, &nullifier_bytes, &sealed).await?;

    Ok(Json(RefundRecoveryResponse {
        refund: RefundInfo {
            refund: URL_SAFE_NO_PAD.encode(&refund_cbor),
            issuer_key_id: key_id,
        },
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

    // --- Refund-token sealing ---

    use anonymous_credit_tokens::{PreIssuance, PreRefund, Refund, SpendProof};

    /// A real refund for a real spend: a credential of `credits`, spending
    /// `charge` and refunding `t`. Returns what the client holds (the spend
    /// proof, its `PreRefund`) and the issuer's CBOR refund message.
    fn refund_fixture(
        credits: u128,
        charge: u128,
        t: u128,
    ) -> (PrivateKey, Params, SpendProof<128>, PreRefund, Vec<u8>) {
        let params = Params::new(DS_ORG, DS_SERVICE, DS_DEPLOYMENT, DS_VERSION);
        let sk = PrivateKey::random(OsRng);
        let pre = PreIssuance::random(OsRng);
        let request = pre.request(&params, OsRng);
        let response = sk
            .issue::<128>(
                &params,
                &request,
                credit_to_scalar::<128>(credits).unwrap(),
                Scalar::ZERO,
                OsRng,
            )
            .unwrap();
        let token = pre
            .to_credit_token::<128>(&params, sk.public(), &request, &response)
            .unwrap();
        let (proof, prerefund) = token
            .prove_spend::<128>(&params, credit_to_scalar::<128>(charge).unwrap(), OsRng)
            .unwrap();
        let refund = sk
            .refund(&params, &proof, credit_to_scalar::<128>(t).unwrap(), OsRng)
            .unwrap();
        let cbor = refund.to_cbor().unwrap();
        (sk, params, proof, prerefund, cbor)
    }

    const KEY_ID: &str = "4f1c0d5e2a7b9c3d6e8f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d";

    fn contains(haystack: &[u8], needle: &[u8]) -> bool {
        haystack.windows(needle.len()).any(|w| w == needle)
    }

    fn client_key(fill: u8) -> RefundKey {
        RefundKey::from_bytes([fill; 32])
    }

    #[test]
    fn a_sealed_refund_token_opens_to_the_token_the_client_completes() {
        let master_key = [42u8; 32];
        let ck = client_key(7);
        let (sk, params, proof, prerefund, cbor) = refund_fixture(1000, 300, 117);
        let nullifier = proof.nullifier().as_bytes().to_vec();

        let sealed = seal_refund_token(&master_key, &ck, KEY_ID, &nullifier, &cbor).unwrap();
        let opened = open_refund_token(&master_key, &ck, KEY_ID, &nullifier, &sealed).unwrap();
        assert_eq!(opened, cbor);

        // What recovery hands back still completes the client's credential.
        let refund = Refund::from_cbor(&opened).unwrap();
        let next = prerefund
            .to_credit_token(&params, &proof, &refund, sk.public())
            .unwrap();
        assert_eq!(
            scalar_to_credit::<128>(&next.credits()).unwrap(),
            1000 - 300 + 117
        );
    }

    #[test]
    fn a_sealed_refund_token_opens_only_on_its_own_row() {
        let master_key = [42u8; 32];
        let ck = client_key(7);
        let (_, _, proof, _, cbor) = refund_fixture(1000, 300, 117);
        let nullifier = proof.nullifier().as_bytes().to_vec();
        let sealed = seal_refund_token(&master_key, &ck, KEY_ID, &nullifier, &cbor).unwrap();

        // Moved to another nullifier under the same key.
        let (_, _, other_proof, _, _) = refund_fixture(1000, 300, 117);
        let other_nullifier = other_proof.nullifier().as_bytes().to_vec();
        assert!(open_refund_token(&master_key, &ck, KEY_ID, &other_nullifier, &sealed).is_none());

        // Moved to the same nullifier value under another issuer key.
        let other_key = "0".repeat(64);
        assert!(open_refund_token(&master_key, &ck, &other_key, &nullifier, &sealed).is_none());

        // Opened under another master key.
        assert!(open_refund_token(&[43u8; 32], &ck, KEY_ID, &nullifier, &sealed).is_none());

        // Any single altered byte.
        for i in 0..sealed.as_bytes().len() {
            let mut bytes = sealed.as_bytes().to_vec();
            bytes[i] ^= 0x01;
            let altered = SealedRefundToken::from_stored(bytes);
            assert!(
                open_refund_token(&master_key, &ck, KEY_ID, &nullifier, &altered).is_none(),
                "byte {i} altered and the token still opened"
            );
        }

        // A row written before tokens were sealed holds the bare CBOR.
        assert_eq!(cbor[0], 0xa5);
        let legacy = SealedRefundToken::from_stored(cbor.clone());
        assert!(open_refund_token(&master_key, &ck, KEY_ID, &nullifier, &legacy).is_none());
        let empty = SealedRefundToken::from_stored(Vec::new());
        assert!(open_refund_token(&master_key, &ck, KEY_ID, &nullifier, &empty).is_none());
    }

    #[test]
    fn a_sealed_refund_token_carries_no_plaintext_amount() {
        let master_key = [42u8; 32];
        let ck = client_key(7);
        let t: u128 = 0x0123_4567_89ab_cdef_0011_2233_4455_6677;
        let (_, _, proof, _, cbor) = refund_fixture(u128::MAX >> 1, t, t);
        let nullifier = proof.nullifier().as_bytes().to_vec();
        let t_bytes = credit_to_scalar::<128>(t).unwrap().to_bytes();
        // The bare token does carry the amount: that is what is being hidden.
        assert!(contains(&cbor, &t_bytes));

        let sealed = seal_refund_token(&master_key, &ck, KEY_ID, &nullifier, &cbor).unwrap();
        let bytes = sealed.as_bytes();
        assert!(!contains(bytes, &t_bytes));
        // Nor the amount's own bytes, nor any run of the CBOR.
        assert!(!contains(bytes, &t.to_le_bytes()));
        assert!(!cbor.windows(12).any(|w| contains(bytes, w)));

        // And the length is the same whatever the amount.
        let (_, _, p0, _, c0) = refund_fixture(1000, 300, 0);
        let s0 =
            seal_refund_token(&master_key, &ck, KEY_ID, p0.nullifier().as_bytes(), &c0).unwrap();
        assert_eq!(s0.as_bytes().len(), bytes.len());
        assert_eq!(bytes.len(), 1 + 12 + cbor.len() + 16);
    }

    #[test]
    fn the_refund_token_key_is_its_own() {
        let master_key = [42u8; 32];
        let ck = client_key(7);
        // Not the issuer-key wrapping key derived from the same master key...
        assert_ne!(
            *derive_refund_token_key(&master_key, &ck, KEY_ID),
            derive_key_encryption_key(&master_key, KEY_ID)
        );
        // ...and not another issuer key's refund-token key...
        assert_ne!(
            *derive_refund_token_key(&master_key, &ck, KEY_ID),
            *derive_refund_token_key(&master_key, &ck, &"0".repeat(64))
        );
        // ...and not another client's, nor another master key's.
        assert_ne!(
            *derive_refund_token_key(&master_key, &ck, KEY_ID),
            *derive_refund_token_key(&master_key, &client_key(8), KEY_ID)
        );
        assert_ne!(
            *derive_refund_token_key(&master_key, &ck, KEY_ID),
            *derive_refund_token_key(&[43u8; 32], &ck, KEY_ID)
        );
    }

    #[test]
    fn a_sealed_refund_token_opens_only_with_its_clients_key() {
        let master_key = [42u8; 32];
        let ck = client_key(7);
        let (_, _, proof, _, cbor) = refund_fixture(1000, 300, 117);
        let nullifier = proof.nullifier().as_bytes().to_vec();
        let sealed = seal_refund_token(&master_key, &ck, KEY_ID, &nullifier, &cbor).unwrap();

        // A recovery presenting any other key opens nothing.
        for other in [
            client_key(8),
            client_key(0),
            RefundKey::from_bytes(master_key),
        ] {
            assert!(open_refund_token(&master_key, &other, KEY_ID, &nullifier, &sealed).is_none());
        }
        let mut near = [7u8; 32];
        near[31] ^= 0x01;
        assert!(
            open_refund_token(
                &master_key,
                &RefundKey::from_bytes(near),
                KEY_ID,
                &nullifier,
                &sealed
            )
            .is_none()
        );
    }

    /// A spend arrives with its refund recovery key or is refused by the
    /// extractor, before the handler that records the nullifier runs.
    #[tokio::test]
    async fn a_spend_without_its_refund_key_is_refused_before_it_is_taken() {
        use axum::extract::FromRequestParts;
        use eidola_common::refund_key::{REFUND_KEY_HEADER, encode_refund_key};

        let (_, _, proof, _, _) = refund_fixture(1000, 300, 0);
        let mut token = Vec::new();
        token.extend_from_slice(&ACT_TOKEN_TYPE.to_be_bytes());
        token.extend_from_slice(&compute_challenge_digest());
        token.extend_from_slice(&[9u8; 32]);
        token.extend_from_slice(&proof.to_cbor().unwrap());
        let authorization = format!("PrivateToken token=\"{}\"", URL_SAFE_NO_PAD.encode(&token));
        let key = [0x5au8; 32];
        let good = encode_refund_key(&key);
        let good = std::str::from_utf8(&good).unwrap().to_string();

        let extract = |refund_key: Option<String>| {
            let authorization = authorization.clone();
            async move {
                let mut request =
                    axum::http::Request::builder().header("authorization", authorization);
                if let Some(value) = refund_key {
                    request = request.header(REFUND_KEY_HEADER, value);
                }
                let (mut parts, ()) = request.body(()).unwrap().into_parts();
                TokenAuth::from_request_parts(&mut parts, &()).await
            }
        };

        for refused in [
            None,
            Some(String::new()),
            Some(good.to_ascii_uppercase()),
            Some(good[..63].to_string()),
            Some(format!("{good}0")),
        ] {
            match extract(refused).await {
                Err(ServerError::BadRequest { .. }) => {}
                Err(e) => panic!("expected 400, got {e:?}"),
                Ok(_) => panic!("a spend without a well-formed refund key was accepted"),
            }
        }
        let TokenAuth(act) = extract(Some(good)).await.unwrap();
        assert_eq!(act.refund_key.as_bytes(), &key);
        assert_eq!(format!("{:?}", act.refund_key), "RefundKey(<redacted>)");
    }

    /// What an operator holds without the client: the stored row and every
    /// secret the gateway has (the master key, and every key derivable from
    /// it alone). None of it opens the row.
    #[test]
    fn the_stored_row_and_the_gateways_secrets_alone_open_nothing() {
        let master_key = [42u8; 32];
        let ck = client_key(7);
        let (_, _, proof, _, cbor) = refund_fixture(1000, 300, 117);
        let nullifier = proof.nullifier().as_bytes().to_vec();
        let sealed = seal_refund_token(&master_key, &ck, KEY_ID, &nullifier, &cbor).unwrap();
        let bytes = sealed.as_bytes();
        let (nonce, ciphertext) = bytes[1..].split_at(SEALED_REFUND_NONCE_LEN);
        let nonce = Nonce::try_from(nonce).unwrap();
        let aad = refund_token_aad(KEY_ID, &nullifier);

        // Keys an operator could derive from the master key by itself: the
        // master key, the issuer-key wrapping key, the refund-token
        // derivation with the client's half missing, and with it filled by
        // anything the gateway itself has (zeros, its own master key).
        let hk = hkdf::Hkdf::<Sha256>::new(Some(KEY_ID.as_bytes()), &master_key);
        let mut master_only = [0u8; 32];
        hk.expand(REFUND_TOKEN_KEY_INFO, &mut master_only).unwrap();
        let candidates = [
            master_key,
            derive_key_encryption_key(&master_key, KEY_ID),
            master_only,
            *derive_refund_token_key(&master_key, &client_key(0), KEY_ID),
            *derive_refund_token_key(&master_key, &RefundKey::from_bytes(master_key), KEY_ID),
        ];
        for key in candidates {
            let cipher = Aes256Gcm::new_from_slice(&key).unwrap();
            let attempt = cipher.decrypt(
                &nonce,
                Payload {
                    msg: ciphertext,
                    aad: &aad,
                },
            );
            assert!(attempt.is_err());
        }
    }

    /// The recovery endpoint end to end against Postgres: it stores only a
    /// sealed token, returns the same token on the next ask with the same
    /// client key, and answers another client key, a token moved from another
    /// row, or a bare token, as no token at all.
    ///
    /// `DATABASE_URL` and `CREDENTIAL_MASTER_KEY` as `just test-integration`
    /// sets them. Run on a thread with a larger stack: an unoptimized build
    /// holds several 128-bit spend proofs in the handler's future at once,
    /// which overflows a test thread's default 2 MiB.
    #[test]
    #[ignore]
    fn recovery_stores_only_sealed_tokens_and_still_returns_the_refund() {
        std::thread::Builder::new()
            .stack_size(16 << 20)
            .spawn(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(recovery_stores_only_sealed_tokens())
            })
            .unwrap()
            .join()
            .unwrap();
    }

    async fn recovery_stores_only_sealed_tokens() {
        use crate::auth::{ActSpend, TokenAuth};
        use crate::backend::Catalog;

        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let pool = db::create_pool(
            &std::env::var("DATABASE_URL").expect("DATABASE_URL must be set"),
            None,
            None,
        )
        .unwrap();
        let master_key: [u8; 32] =
            hex::decode(std::env::var("CREDENTIAL_MASTER_KEY").unwrap_or_else(|_| "00".repeat(32)))
                .unwrap()
                .try_into()
                .unwrap();
        let cache: KeyCache = Arc::new(RwLock::new(HashMap::new()));
        let epoch = EpochConfig::default();
        let key_hash = ensure_keys(&cache, &master_key, &pool, &epoch)
            .await
            .unwrap();
        let state = AppState::new(
            Catalog::new(None),
            None,
            pool.clone(),
            None,
            None,
            master_key,
            cache.clone(),
            epoch,
        );
        let key_id = hex::encode(key_hash);

        // A client's credential under the current key, and a spend of it.
        let spend = |credits: u128, charge: u128| {
            let cache = cache.clone();
            async move {
                let cache = cache.read().await;
                let key = cache.get(&key_hash).unwrap();
                let pre = PreIssuance::random(OsRng);
                let request = pre.request(&key.params, OsRng);
                let response = key
                    .secret_key
                    .issue::<128>(
                        &key.params,
                        &request,
                        credit_to_scalar::<128>(credits).unwrap(),
                        key.request_context_scalar,
                        OsRng,
                    )
                    .unwrap();
                let token = pre
                    .to_credit_token::<128>(
                        &key.params,
                        key.secret_key.public(),
                        &request,
                        &response,
                    )
                    .unwrap();
                token
                    .prove_spend::<128>(
                        &key.params,
                        credit_to_scalar::<128>(charge).unwrap(),
                        OsRng,
                    )
                    .unwrap()
            }
        };
        let recover = |proof: SpendProof<128>, refund_key: RefundKey| {
            let state = state.clone();
            async move {
                recover_refund(
                    TokenAuth(ActSpend {
                        challenge_digest: compute_challenge_digest(),
                        issuer_key_hash: key_hash,
                        spend_proof: proof,
                        refund_key,
                    }),
                    State(state),
                )
                .await
                .map(|Json(r)| r.refund.refund)
            }
        };
        let client = pool.get().await.unwrap();
        let stored_row = async |nullifier: &[u8]| -> Vec<u8> {
            client
                .query_one(
                    "SELECT refund_token FROM nullifier \
                     WHERE issuer_key_id = $1 AND value = $2",
                    &[&key_id, &nullifier],
                )
                .await
                .unwrap()
                .get::<_, Option<Vec<u8>>>("refund_token")
                .unwrap()
        };

        let (proof, prerefund) = spend(1000, 300).await;
        let nullifier = proof.nullifier().as_bytes().to_vec();
        let ck = client_key(7);

        // The original request never arrived: a full refund is issued...
        let first = recover(proof.clone(), ck.clone()).await.unwrap();
        let first_cbor = URL_SAFE_NO_PAD.decode(&first).unwrap();
        // ...and what Postgres holds is the sealed token, not the CBOR.
        let row = stored_row(&nullifier).await;
        assert_eq!(row[0], SEALED_REFUND_VERSION);
        let t_bytes = credit_to_scalar::<128>(300).unwrap().to_bytes();
        assert!(contains(&first_cbor, &t_bytes));
        assert!(!contains(&row, &t_bytes));
        assert!(!first_cbor.windows(12).any(|w| contains(&row, w)));

        // Asked with another key, nothing comes back, and the row is untouched.
        match recover(proof.clone(), client_key(8)).await {
            Err(ServerError::NotFound { .. }) => {}
            Err(e) => panic!("expected 404, got {e:?}"),
            Ok(_) => panic!("a stored token opened under another client's key"),
        }
        assert_eq!(stored_row(&nullifier).await, row);

        // Asked again with its own key, the stored token comes back and
        // completes the credential.
        let second = recover(proof.clone(), ck.clone()).await.unwrap();
        assert_eq!(second, first);
        let refund = Refund::from_cbor(&URL_SAFE_NO_PAD.decode(&second).unwrap()).unwrap();
        let next = {
            let cache = cache.read().await;
            let key = cache.get(&key_hash).unwrap();
            prerefund
                .to_credit_token(&key.params, &proof, &refund, key.secret_key.public())
                .unwrap()
        };
        assert_eq!(scalar_to_credit::<128>(&next.credits()).unwrap(), 1000);

        // Another spend whose row exists: the first row's sealed token copied
        // onto it, or a plaintext token, recovers nothing.
        let (other, _) = spend(1000, 50).await;
        let other_nullifier = other.nullifier().as_bytes().to_vec();
        assert!(
            db::record_nullifier(&pool, &key_id, &other_nullifier)
                .await
                .unwrap()
        );
        for planted in [row.clone(), first_cbor.clone()] {
            client
                .execute(
                    "UPDATE nullifier SET refund_token = $3 \
                     WHERE issuer_key_id = $1 AND value = $2",
                    &[&key_id, &other_nullifier, &planted],
                )
                .await
                .unwrap();
            match recover(other.clone(), ck.clone()).await {
                Err(ServerError::NotFound { .. }) => {}
                Err(e) => panic!("expected 404, got {e:?}"),
                Ok(_) => panic!("a token planted on another row was returned"),
            }
        }
    }
}
