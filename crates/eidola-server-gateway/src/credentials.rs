//! Credential issuance: key management and credential endpoints.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aes_gcm::aead::{Aead, Payload};
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use anonymous_credit_tokens::{IssuanceRequest, Params, PrivateKey, Scalar, credit_to_scalar};
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
use crate::helpers::system_time_to_iso_lossy;
use crate::response::RefundInfo;
use eidola_common::issuer_calendar::{EpochWindows, ISSUER_CALENDAR, rfc3339_utc};

// ---------------------------------------------------------------------------
// Key cache types
// ---------------------------------------------------------------------------

/// A decrypted issuer key held in memory.
pub struct DecryptedIssuerKey {
    pub secret_key: PrivateKey,
    pub params: Params,
    pub key_hash: [u8; 32],
    /// The key's public half (CBOR), as `GET /v1/keys` serves it.
    pub public_key: Vec<u8>,
    pub request_context_scalar: Scalar,
    /// The calendar epoch this key issues in. Every cached key's windows are
    /// exactly this epoch's (see [`canonical_epoch`]).
    pub epoch: u64,
    pub issue_from: SystemTime,
    pub issue_until: SystemTime,
    /// End of acceptance (exclusive): no spend under this key is accepted
    /// at or after this instant.
    pub accept_until: SystemTime,
}

/// Thread-safe cache of decrypted issuer keys, keyed by SHA-256(pkI).
pub type KeyCache = Arc<RwLock<HashMap<[u8; 32], DecryptedIssuerKey>>>;

// ---------------------------------------------------------------------------
// Key deadlines
// ---------------------------------------------------------------------------

/// How long past a key's `accept_until` the gateway keeps the key, and every
/// nullifier recorded under it.
///
/// A spend accepted just before `accept_until` still has to finish: its
/// refund is signed when the response completes, and the stored refund token
/// answers `POST /v1/credentials/refund`. The key stays usable for that work
/// until `accept_until + RETENTION_PAST_ACCEPT`, and the gateway prunes a
/// key's nullifiers (`db::prune_expired_nullifiers`, called only with this
/// constant) only after that same instant. So no nullifier is pruned while
/// its key could still accept a spend, with this whole margin to spare
/// against clock skew between gateway instances and the database.
///
/// The client stops starting spends `CREDENTIAL_SPEND_MARGIN` (10 min) before
/// `accept_until` and stops recovering at `accept_until`, so its useful
/// window ends inside this one. A refund signed after `accept_until` carries
/// no spendable value (it is a credential under the same, now-refusing key);
/// signing it only keeps a response that began in time well-formed.
pub const RETENTION_PAST_ACCEPT: Duration = Duration::from_secs(3600);

const _: () = assert!(
    RETENTION_PAST_ACCEPT.as_secs() > 2 * db::MAX_CLOCK_SKEW.as_secs(),
    "the retention margin must exceed the clock skew two instances can disagree by"
);

/// What a key is being used for, each with its own deadline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyUse {
    /// Accepting a spend: verifying a proof and recording a fresh nullifier.
    /// Allowed while `now < accept_until`.
    Spend,
    /// Finishing a spend already accepted: signing its refund or returning
    /// its stored refund token. Allowed while `now < retain_until`.
    Refund,
}

/// The instant from which a key does nothing at all, and its nullifiers may
/// be pruned.
pub fn retain_until(accept_until: SystemTime) -> SystemTime {
    accept_until + RETENTION_PAST_ACCEPT
}

/// Whether a key whose acceptance deadline is `accept_until` may be used for
/// `key_use` at `now`.
pub fn key_usable(accept_until: SystemTime, key_use: KeyUse, now: SystemTime) -> bool {
    match key_use {
        KeyUse::Spend => now < accept_until,
        KeyUse::Refund => now < retain_until(accept_until),
    }
}

/// Whether a key's nullifiers may be pruned at `now`: only once nothing can
/// use the key any more. Mirrors the condition in the SQL
/// `prune_expired_nullifiers`, which the gateway calls with
/// `RETENTION_PAST_ACCEPT`.
pub fn nullifiers_prunable(accept_until: SystemTime, now: SystemTime) -> bool {
    now >= retain_until(accept_until)
}

/// The cached key for `key_hash`, if it may be used for `key_use` at `now`.
///
/// Every read of the cache that acts on a presented credential goes through
/// here, so a key past its deadline is refused whether or not it is cached.
pub fn usable_key<'a>(
    cache: &'a HashMap<[u8; 32], DecryptedIssuerKey>,
    key_hash: &[u8; 32],
    key_use: KeyUse,
    now: SystemTime,
) -> Result<&'a DecryptedIssuerKey, ServerError> {
    match cache.get(key_hash) {
        Some(key) if key_usable(key.accept_until, key_use, now) => Ok(key),
        _ => Err(expired_key()),
    }
}

fn expired_key() -> ServerError {
    ServerError::Unauthorized {
        message: "unknown or expired issuer key".to_string(),
    }
}

/// Drop every cached key past its retention deadline. Returns how many were
/// dropped.
pub fn evict_retired_keys(
    cache: &mut HashMap<[u8; 32], DecryptedIssuerKey>,
    now: SystemTime,
) -> usize {
    let before = cache.len();
    cache.retain(|_, key| key_usable(key.accept_until, KeyUse::Refund, now));
    before - cache.len()
}

// ---------------------------------------------------------------------------
// Key windows: the public calendar
// ---------------------------------------------------------------------------

/// Unix seconds of `t`, if it is a whole second at or after the Unix epoch.
/// Every instant on the calendar is; anything else is not a key window.
fn whole_unix_secs(t: SystemTime) -> Option<u64> {
    let d = t.duration_since(UNIX_EPOCH).ok()?;
    (d.subsec_nanos() == 0).then_some(d.as_secs())
}

/// The calendar's windows as stored and served.
fn calendar_windows(
    issue_from: SystemTime,
    issue_until: SystemTime,
    accept_until: SystemTime,
) -> Option<EpochWindows> {
    Some(EpochWindows {
        issue_from: whole_unix_secs(issue_from)?,
        issue_until: whole_unix_secs(issue_until)?,
        accept_until: whole_unix_secs(accept_until)?,
    })
}

/// An instant as `GET /v1/keys` spells it ([`rfc3339_utc`]).
fn wire_instant(t: SystemTime) -> String {
    rfc3339_utc(whole_unix_secs(t).unwrap_or(0))
}

/// The epoch a key with these windows belongs to, or `None` if they are not
/// exactly one epoch's windows on the public calendar
/// ([`ISSUER_CALENDAR`]). A key that answers `None` is never used.
pub fn canonical_epoch(
    issue_from: SystemTime,
    issue_until: SystemTime,
    accept_until: SystemTime,
) -> Option<u64> {
    ISSUER_CALENDAR.epoch_of(calendar_windows(issue_from, issue_until, accept_until)?)
}

/// The epoch `now` falls in.
fn epoch_at(now: SystemTime) -> Result<u64, ServerError> {
    now.duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| ISSUER_CALENDAR.epoch_at(d.as_secs()))
        .ok_or_else(|| ServerError::Internal("clock is before the issuer calendar".to_string()))
}

/// Epoch `epoch`'s three instants.
fn epoch_instants(epoch: u64) -> Result<(SystemTime, SystemTime, SystemTime), ServerError> {
    let w = ISSUER_CALENDAR
        .windows(epoch)
        .ok_or_else(|| ServerError::Internal("issuer epoch out of range".to_string()))?;
    let at = |secs: u64| UNIX_EPOCH + Duration::from_secs(secs);
    Ok((at(w.issue_from), at(w.issue_until), at(w.accept_until)))
}

/// The one refusal for a key record this gateway will not use: windows off
/// the calendar, a foreign domain separator, an id that is not its key's
/// hash, or a private key that does not decrypt under the record as it
/// stands. Logged without anything from the record.
fn refused_key_record() -> ServerError {
    warn!("refusing an issuer key record that does not authenticate");
    ServerError::Internal("issuer key record refused".to_string())
}

// ---------------------------------------------------------------------------
// Key encryption / decryption
// ---------------------------------------------------------------------------

/// HKDF info for the issuer-key wrapping key. `v2` is the format whose
/// associated data binds the record (see [`issuer_key_aad`]).
const ISSUER_KEY_WRAP_INFO: &[u8] = b"eidola-issuer-key-v2";

/// Domain separator at the head of an issuer key's associated data.
const ISSUER_KEY_AAD_LABEL: &[u8] = b"eidola/issuer-key/v2";

/// The associated data an issuer private key is encrypted under: everything
/// the gateway reads from the key's row besides the ciphertext.
///
/// `label ‖ len(id) ‖ id ‖ len(public_key) ‖ public_key ‖ len(ds) ‖ ds ‖
/// issue_from ‖ issue_until ‖ accept_until`, each length and instant a
/// big-endian `u64` (instants in Unix seconds). So a row whose id, public
/// key, domain separator or any of its three windows was edited no longer
/// decrypts, and the key is refused. Postgres holds these columns in the
/// clear; this is what makes them the enclave's statement rather than the
/// database's.
fn issuer_key_aad(
    id: &str,
    public_key: &[u8],
    domain_separator: &str,
    windows: EpochWindows,
) -> Vec<u8> {
    let mut aad = Vec::with_capacity(
        ISSUER_KEY_AAD_LABEL.len()
            + 3 * 8
            + id.len()
            + public_key.len()
            + domain_separator.len()
            + 3 * 8,
    );
    aad.extend_from_slice(ISSUER_KEY_AAD_LABEL);
    for field in [id.as_bytes(), public_key, domain_separator.as_bytes()] {
        aad.extend_from_slice(&(field.len() as u64).to_be_bytes());
        aad.extend_from_slice(field);
    }
    for instant in [
        windows.issue_from,
        windows.issue_until,
        windows.accept_until,
    ] {
        aad.extend_from_slice(&instant.to_be_bytes());
    }
    aad
}

/// Derive a per-key AES-256-GCM key from the master key using HKDF-SHA256.
fn derive_key_encryption_key(master_key: &[u8; 32], key_id: &str) -> [u8; 32] {
    let hk = hkdf::Hkdf::<Sha256>::new(Some(key_id.as_bytes()), master_key);
    let mut okm = [0u8; 32];
    hk.expand(ISSUER_KEY_WRAP_INFO, &mut okm)
        .expect("32 bytes is a valid HKDF-SHA256 output length");
    okm
}

/// Encrypt a private key with AES-256-GCM under associated data `aad`.
/// Output format: `nonce (12 bytes) || ciphertext+tag`.
fn encrypt_private_key(
    master_key: &[u8; 32],
    key_id: &str,
    aad: &[u8],
    plaintext: &[u8],
) -> Result<Vec<u8>, ServerError> {
    let aes_key = derive_key_encryption_key(master_key, key_id);
    let cipher = Aes256Gcm::new_from_slice(&aes_key)
        .map_err(|e| ServerError::Internal(format!("AES key init failed: {}", e)))?;

    let mut nonce_bytes = [0u8; 12];
    rand_core::OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from(nonce_bytes);

    let ciphertext = cipher
        .encrypt(
            &nonce,
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .map_err(|e| ServerError::Internal(format!("AES encryption failed: {}", e)))?;

    let mut out = Vec::with_capacity(12 + ciphertext.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Decrypt a private key from the `nonce || ciphertext+tag` format, under the
/// same associated data it was encrypted with.
fn decrypt_private_key(
    master_key: &[u8; 32],
    key_id: &str,
    aad: &[u8],
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
        .decrypt(
            &nonce,
            Payload {
                msg: ciphertext,
                aad,
            },
        )
        .map_err(|e| ServerError::Internal(format!("AES decryption failed: {}", e)))
}

// ---------------------------------------------------------------------------
// Refund-token sealing
// ---------------------------------------------------------------------------

/// HKDF info for the refund-token key, domain-separated from the issuer-key
/// wrapping key's (`eidola-issuer-key-v2`).
const REFUND_TOKEN_KEY_INFO: &[u8] = b"eidola-refund-token-v1";

/// First byte of every sealed refund token. A CBOR `RefundMsg` (a five-entry
/// map) begins with `0xa5`, so a bare token is told apart by this byte as
/// well as by failing to open.
const SEALED_REFUND_VERSION: u8 = 0x01;

const SEALED_REFUND_NONCE_LEN: usize = 12;

/// A refund token as it is stored in `nullifier.refund_token`: AES-256-GCM
/// under a key derived from the client's refund recovery key alone, bound to
/// its row.
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

/// The key one spend's refund is sealed under: HKDF-SHA256 with the client's
/// refund recovery key as the input key material, salted with the issuer key
/// id.
///
/// The client's key alone: it is what keeps the stored amount from anyone
/// without the client, and nothing the operator holds is mixed in, because
/// an operator-provided secret (`CREDENTIAL_MASTER_KEY`) protects nothing
/// against the operator. Each spend's client key is its own, so the
/// random-nonce count under any one AES key is one or two.
fn derive_refund_token_key(
    refund_key: &RefundKey,
    issuer_key_id: &str,
) -> zeroize::Zeroizing<[u8; 32]> {
    let hk = hkdf::Hkdf::<Sha256>::new(Some(issuer_key_id.as_bytes()), refund_key.as_bytes());
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
    refund_key: &RefundKey,
    issuer_key_id: &str,
    nullifier: &[u8],
    refund_cbor: &[u8],
) -> Result<SealedRefundToken, ServerError> {
    let key = derive_refund_token_key(refund_key, issuer_key_id);
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
/// client key: a recovery presenting another key, a
/// token moved from another row, a bare token, or tampered bytes. The caller
/// treats that exactly as a row with no token stored.
pub fn open_refund_token(
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
    let key = derive_refund_token_key(refund_key, issuer_key_id);
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

/// Seal a spend's refund under its client's refund recovery key and store it
/// on the spend's nullifier row: the one way a refund reaches Postgres, for
/// every path that signs one ([`sign_refund`], [`verify_and_sign_full_refund`]).
pub async fn store_sealed_refund(
    state: &AppState,
    act: &crate::auth::ActSpend,
    issuer_key_id: &str,
    refund_cbor: &[u8],
) -> Result<(), ServerError> {
    let nullifier = act.spend_proof.nullifier().as_bytes().to_vec();
    let sealed = seal_refund_token(&act.refund_key, issuer_key_id, &nullifier, refund_cbor)?;
    db::store_refund_token(&state.db_pool, issuer_key_id, &nullifier, &sealed).await
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

/// Authenticate an issuer key row and insert it into the cache. Returns the
/// key hash.
///
/// The row is used only if its windows are exactly one epoch's on the public
/// calendar, its domain separator is this deployment's, its id is the hash
/// of its public key, and its private key decrypts under associated data
/// binding all of them ([`issuer_key_aad`]). Anything else is refused
/// ([`refused_key_record`]): Postgres can edit a plain column, but not one
/// that is bound into the ciphertext.
fn cache_key(
    cache: &mut HashMap<[u8; 32], DecryptedIssuerKey>,
    master_key: &[u8; 32],
    row: &db::IssuerKeyRow,
) -> Result<[u8; 32], ServerError> {
    let key_hash = compute_key_hash(&row.public_key);
    if cache.contains_key(&key_hash) {
        return Ok(key_hash);
    }
    if row.id != hex::encode(key_hash) || row.domain_separator != domain_separator() {
        return Err(refused_key_record());
    }
    let (Some(windows), Some(epoch)) = (
        calendar_windows(row.issue_from, row.issue_until, row.accept_until),
        canonical_epoch(row.issue_from, row.issue_until, row.accept_until),
    ) else {
        return Err(refused_key_record());
    };
    let aad = issuer_key_aad(&row.id, &row.public_key, &row.domain_separator, windows);
    let plaintext = decrypt_private_key(master_key, &row.id, &aad, &row.private_key_enc)
        .map_err(|_| refused_key_record())?;
    let secret_key = PrivateKey::from_cbor(&plaintext).map_err(|_| refused_key_record())?;
    let params = Params::new(DS_ORG, DS_SERVICE, DS_DEPLOYMENT, DS_VERSION);
    let request_context = compute_request_context(&key_hash);
    let request_context_scalar = request_context_to_scalar(&request_context);
    cache.insert(
        key_hash,
        DecryptedIssuerKey {
            secret_key,
            params,
            key_hash,
            public_key: row.public_key.clone(),
            request_context_scalar,
            epoch,
            issue_from: row.issue_from,
            issue_until: row.issue_until,
            accept_until: row.accept_until,
        },
    );
    Ok(key_hash)
}

/// Generate a new issuer key pair for calendar epoch `epoch` and build a row
/// ready for insertion. Its windows are the epoch's and nothing else, and
/// they are bound into the encryption of its private key.
fn generate_key(master_key: &[u8; 32], epoch: u64) -> Result<db::IssuerKeyRow, ServerError> {
    let secret_key = PrivateKey::random(OsRng);
    let public_key_cbor = secret_key
        .public()
        .to_cbor()
        .map_err(|e| ServerError::Internal(format!("failed to encode public key: {}", e)))?;
    let id = hex::encode(compute_key_hash(&public_key_cbor));
    let private_key_cbor = secret_key
        .to_cbor()
        .map_err(|e| ServerError::Internal(format!("failed to encode private key: {}", e)))?;

    let (issue_from, issue_until, accept_until) = epoch_instants(epoch)?;
    let windows = calendar_windows(issue_from, issue_until, accept_until)
        .ok_or_else(|| ServerError::Internal("issuer epoch out of range".to_string()))?;
    let ds = domain_separator();
    let aad = issuer_key_aad(&id, &public_key_cbor, &ds, windows);
    let encrypted = encrypt_private_key(master_key, &id, &aad, &private_key_cbor)?;

    Ok(db::IssuerKeyRow {
        id,
        private_key_enc: encrypted,
        public_key: public_key_cbor,
        domain_separator: ds,
        issue_from,
        issue_until,
        accept_until,
    })
}

/// The one cached key for `epoch`: `Ok(None)` if there is none, and a refusal
/// if there is more than one. An honest gateway holds exactly one key per
/// epoch, so two can only mean a database that hid one key from some
/// instance until it minted another; issuing under either would put the
/// clients who got it in a smaller set than the epoch's, so neither is used
/// for issuance. (Both still accept spends: each is genuinely this
/// deployment's key, and tokens already issued under it were paid for.)
fn sole_key_for_epoch(
    cache: &HashMap<[u8; 32], DecryptedIssuerKey>,
    epoch: u64,
) -> Result<Option<[u8; 32]>, ServerError> {
    let mut keys = cache.values().filter(|k| k.epoch == epoch);
    match (keys.next(), keys.next()) {
        (None, _) => Ok(None),
        (Some(k), None) => Ok(Some(k.key_hash)),
        (Some(_), Some(_)) => {
            warn!("more than one issuer key for the current epoch; refusing to issue under either");
            Err(ServerError::Internal(
                "more than one issuer key for one epoch".to_string(),
            ))
        }
    }
}

/// Which epoch's key to mint, given the epochs among the current and the
/// next that the database shows a key for (`present`), the current epoch,
/// and the epochs this gateway already holds keys for. `Ok(None)` when both
/// keys exist.
///
/// The current key first, then the next one, so a client can see the next
/// epoch's key a whole epoch before it issues. Never a key for an epoch this
/// gateway already holds one for: a database that stopped showing a key this
/// instance has cached is refused rather than answered with a second key for
/// that epoch, which is how a hidden row would otherwise split the epoch's
/// anonymity set.
fn epoch_to_mint(
    present: &std::collections::HashSet<u64>,
    current: u64,
    held: &std::collections::HashSet<u64>,
) -> Result<Option<u64>, ServerError> {
    let Some(wanted) = [current, current + 1]
        .into_iter()
        .find(|e| !present.contains(e))
    else {
        return Ok(None);
    };
    if held.contains(&wanted) {
        warn!(
            "the database shows no issuer key for an epoch this gateway holds one for; \
             refusing to mint a second"
        );
        return Err(ServerError::Internal(
            "issuer key missing from the database".to_string(),
        ));
    }
    Ok(Some(wanted))
}

/// Ensure that the current epoch's key and the next epoch's key exist in the
/// database and are cached in memory. Returns the current key's hash (for
/// issuance).
///
/// Keys sit on the public calendar ([`ISSUER_CALENDAR`]): epoch `n`'s key is
/// the one whose windows are epoch `n`'s, whenever and by whichever instance
/// it was minted, so a restart or a late rotation cannot shift a boundary.
/// One key per epoch: a gateway never mints a key for an epoch it already
/// holds one for, even when the database no longer shows it.
///
/// Race safety: uses a serializable transaction (and the table's unique
/// `issue_from`) so concurrent server instances cannot create duplicate keys.
pub async fn ensure_keys(
    key_cache: &KeyCache,
    master_key: &[u8; 32],
    pool: &deadpool_postgres::Pool,
) -> Result<[u8; 32], ServerError> {
    let current = epoch_at(SystemTime::now())?;

    // Fast path: the current key and the next are already cached.
    {
        let cache = key_cache.read().await;
        if let Some(hash) = sole_key_for_epoch(&cache, current)?
            && cache.values().any(|k| k.epoch == current + 1)
        {
            return Ok(hash);
        }
    }

    // Slow path: check the database and provision any missing keys.
    // We may need to create up to 2 keys (current + next), or just the next one.
    loop {
        let mk = *master_key;
        let held: std::collections::HashSet<u64> =
            key_cache.read().await.values().map(|k| k.epoch).collect();
        let current = epoch_at(SystemTime::now())?;
        let wanted = [epoch_instants(current)?.0, epoch_instants(current + 1)?.0];

        let result = db::insert_issuer_key_checked(pool, &wanted, move |rows| {
            // A row occupies its epoch's slot (the table's unique
            // `issue_from`) whether or not it authenticates; one that does
            // not is refused when it is loaded, below.
            let present = rows
                .iter()
                .filter_map(|r| {
                    let secs = whole_unix_secs(r.issue_from)?;
                    ISSUER_CALENDAR.epoch_at(secs)
                })
                .collect();
            match epoch_to_mint(&present, current, &held)? {
                Some(epoch) => Ok(Some(generate_key(&mk, epoch)?)),
                None => Ok(None),
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

    // Load all valid keys from DB into cache, authenticating each; a row that
    // does not authenticate is refused (and logged) and the rest still load.
    // Re-read the epoch: the provisioning round-trips may have crossed a
    // boundary.
    let rows = db::get_valid_issuer_keys(pool).await?;
    let mut cache = key_cache.write().await;
    for row in &rows {
        let _ = cache_key(&mut cache, master_key, row);
    }
    let current = epoch_at(SystemTime::now())?;
    sole_key_for_epoch(&cache, current)?.ok_or_else(|| {
        ServerError::Internal("no current issuer key found after provisioning".to_string())
    })
}

/// Make sure the issuer key for `key_hash` is cached and usable for
/// `key_use`, checking the cache first and then the database.
///
/// The deadline is checked on both paths: a cached key past it is refused
/// (and dropped once past retention), never answered from memory. A key
/// read from the database is authenticated ([`cache_key`]) before its
/// deadline is read, so the deadline checked is the one bound to the key.
pub async fn load_key(
    key_cache: &KeyCache,
    master_key: &[u8; 32],
    pool: &deadpool_postgres::Pool,
    key_hash: &[u8; 32],
    key_use: KeyUse,
) -> Result<(), ServerError> {
    let now = SystemTime::now();

    // Fast path: already in cache.
    {
        let cache = key_cache.read().await;
        if let Some(key) = cache.get(key_hash) {
            if key_usable(key.accept_until, key_use, now) {
                return Ok(());
            }
            let retired = !key_usable(key.accept_until, KeyUse::Refund, now);
            drop(cache);
            if retired {
                key_cache.write().await.remove(key_hash);
            }
            return Err(expired_key());
        }
    }

    // Slow path: load from DB by key hash. The query returns any key still
    // retained; the deadline for this use is checked here, on our clock,
    // against the authenticated record.
    let row = db::get_issuer_key_by_hash(pool, key_hash, RETENTION_PAST_ACCEPT)
        .await?
        .ok_or_else(expired_key)?;
    if !key_usable(row.accept_until, KeyUse::Refund, now) {
        return Err(expired_key());
    }

    let mut cache = key_cache.write().await;
    let hash = cache_key(&mut cache, master_key, &row).map_err(|_| expired_key())?;
    let accept_until = cache[&hash].accept_until;
    if !key_usable(accept_until, key_use, now) {
        return Err(expired_key());
    }
    Ok(())
}

/// Verify `act`'s spend proof under `key` and, only if it verifies, sign a
/// refund returning `t` credits.
///
/// This is the gateway's one call into the ACT verifier. The crate
/// (`anonymous-credit-tokens` 0.4) has no verify-only operation:
/// `PrivateKey::refund` verifies the proof and then signs, and the refund it
/// signs commits to `t` (both its signature and its proof of correctness are
/// computed over it), so a refund signed for one `t` cannot be turned into
/// one for another. A caller that does not yet know its final `t` therefore
/// verifies here and discards the refund, and verifies again when it signs:
/// the chat spend does, since its `t` is known only once the response is
/// settled (`handlers::verify_spend_proof`, then [`sign_refund`]). Recovery
/// knows its `t` (the whole charge) before it reads the database, so it
/// verifies once ([`verify_and_sign_full_refund`]).
///
/// Every verification passes through here, which is what the
/// `verification_count` seam counts in tests.
fn verify_and_refund(
    key: &DecryptedIssuerKey,
    act: &crate::auth::ActSpend,
    t: Scalar,
) -> Result<anonymous_credit_tokens::Refund, anonymous_credit_tokens::ErrorCode> {
    #[cfg(test)]
    verification_count::record(&act.spend_proof);
    key.secret_key
        .refund::<128>(&key.params, &act.spend_proof, t, OsRng)
}

/// The refusal for a presented spend whose request context is not `key`'s.
fn check_presented_context(
    key: &DecryptedIssuerKey,
    act: &crate::auth::ActSpend,
) -> Result<(), ServerError> {
    if act.spend_proof.context() != key.request_context_scalar {
        return Err(ServerError::Unauthorized {
            message: "invalid request_context in spend proof".to_string(),
        });
    }
    Ok(())
}

fn invalid_spend_proof() -> ServerError {
    ServerError::Unauthorized {
        message: "invalid spend proof".to_string(),
    }
}

/// Check a presented spend proof against the key it names: its request
/// context must be this key's, and the proof must verify under this key.
///
/// Nothing is recorded and nothing is kept: the refund the verifier signs for
/// `t = 0` is discarded (see [`verify_and_refund`] on why). The chat spend
/// calls this before it records the nullifier, so a proof that does not
/// verify never reaches the database. Recovery, which knows its refund
/// already, verifies with [`verify_and_sign_full_refund`] instead.
pub fn verify_presented_spend(
    key: &DecryptedIssuerKey,
    act: &crate::auth::ActSpend,
) -> Result<(), ServerError> {
    check_presented_context(key, act)?;
    verify_and_refund(key, act, Scalar::ZERO).map_err(|_| invalid_spend_proof())?;
    Ok(())
}

/// A refund signed for a spend, CBOR-encoded, with the id of the key that
/// signed it: the spent credential's own key.
pub struct SignedRefund {
    pub cbor: Vec<u8>,
    pub issuer_key_id: String,
}

impl SignedRefund {
    fn encode(
        act: &crate::auth::ActSpend,
        refund: anonymous_credit_tokens::Refund,
    ) -> Result<Self, ServerError> {
        let cbor = refund
            .to_cbor()
            .map_err(|e| ServerError::Internal(format!("refund CBOR encoding failed: {e:?}")))?;
        Ok(Self {
            cbor,
            issuer_key_id: hex::encode(act.issuer_key_hash),
        })
    }
}

/// Check a presented spend proof as [`verify_presented_spend`] does, and
/// keep the refund that one verification signs: a full refund, returning the
/// whole charge. The proof is verified exactly once.
///
/// The key is the one `act` names, as for [`sign_refund`], and must still be
/// usable for [`KeyUse::Refund`] at `now`. Nothing is recorded: a refund
/// signed here leaves the gateway only if the caller goes on to record the
/// spend's nullifier, so signing it before the database is read changes
/// nothing anyone sees.
pub fn verify_and_sign_full_refund(
    cache: &HashMap<[u8; 32], DecryptedIssuerKey>,
    act: &crate::auth::ActSpend,
    now: SystemTime,
) -> Result<SignedRefund, ServerError> {
    let key = usable_key(cache, &act.issuer_key_hash, KeyUse::Refund, now)?;
    check_presented_context(key, act)?;
    // `t = s`. Any refusal is the proof's: a verified proof's charge fits the
    // credential's 128 bits, and `t = s` is within `0 <= t <= s`.
    let refund =
        verify_and_refund(key, act, act.spend_proof.charge()).map_err(|_| invalid_spend_proof())?;
    SignedRefund::encode(act, refund)
}

/// Sign the refund for a spent credential, returning its CBOR encoding and
/// the id of the key that signed it.
///
/// The key is the one the spent credential names (`act.issuer_key_hash`),
/// never the current issuing key: a refund is a credential under the same
/// key as the one it replaces, so it lapses at that key's `accept_until`, and
/// a spend near the end of an epoch cannot carry value into a later one.
///
/// Signing verifies the proof again ([`verify_and_refund`]): the chat spend's
/// second verification, which the crate does not let it skip.
pub fn sign_refund(
    cache: &HashMap<[u8; 32], DecryptedIssuerKey>,
    act: &crate::auth::ActSpend,
    refund_credits: u128,
    now: SystemTime,
) -> Result<(Vec<u8>, String), ServerError> {
    let t = credit_to_scalar::<128>(refund_credits)
        .map_err(|e| ServerError::Internal(format!("invalid refund amount: {e:?}")))?;
    let key = usable_key(cache, &act.issuer_key_hash, KeyUse::Refund, now)?;
    let refund = verify_and_refund(key, act, t)
        .map_err(|e| ServerError::Internal(format!("refund issuance failed: {e:?}")))?;
    let signed = SignedRefund::encode(act, refund)?;
    Ok((signed.cbor, signed.issuer_key_id))
}

/// How many times each spend proof has been through the verifier, keyed by
/// its nullifier (unique to the proof), so tests running in parallel count
/// their own proofs only. A clone of a spend counts as the same spend.
#[cfg(test)]
pub(crate) mod verification_count {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use anonymous_credit_tokens::SpendProof;

    fn counts() -> &'static Mutex<HashMap<[u8; 32], usize>> {
        static COUNTS: OnceLock<Mutex<HashMap<[u8; 32], usize>>> = OnceLock::new();
        COUNTS.get_or_init(Default::default)
    }

    pub(crate) fn record(proof: &SpendProof<128>) {
        *counts()
            .lock()
            .unwrap()
            .entry(proof.nullifier().to_bytes())
            .or_default() += 1;
    }

    /// How many times `proof` has been verified so far.
    pub(crate) fn of(proof: &SpendProof<128>) -> usize {
        counts()
            .lock()
            .unwrap()
            .get(&proof.nullifier().to_bytes())
            .copied()
            .unwrap_or(0)
    }
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

            let evicted = evict_retired_keys(&mut *key_cache.write().await, SystemTime::now());
            if evicted > 0 {
                info!("evicted {evicted} retired issuer key(s) from the cache");
            }

            match ensure_keys(&key_cache, &master_key, &pool).await {
                Ok(key_hash) => {
                    info!("periodic key check: current key {}", hex::encode(key_hash));
                }
                Err(e) => {
                    warn!("periodic key check failed: {}", e);
                }
            }

            // Same deadline as the eviction above: a key's nullifiers go
            // only once nothing can use the key.
            match db::prune_expired_nullifiers(&pool, RETENTION_PAST_ACCEPT).await {
                Ok(0) => {}
                Ok(n) => info!("pruned {n} nullifier(s) of retired issuer keys"),
                Err(e) => warn!("nullifier pruning failed: {}", e),
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
    // Served from the cache, never straight from Postgres: every cached key
    // was authenticated against its own record when it was loaded, so what a
    // client is told about a key's windows is what the enclave bound to it.
    // A failure here (say, two keys for the current epoch) still lists what
    // is held, so a client can see it and refuse.
    if let Err(e) = ensure_keys(
        &state.credential_key_cache,
        &state.credential_master_key,
        &state.db_pool,
    )
    .await
    {
        warn!("issuer key check before listing failed: {}", e);
    }
    let now = SystemTime::now();
    let cache = state.credential_key_cache.read().await;
    let mut keys: Vec<&DecryptedIssuerKey> = cache
        .values()
        .filter(|k| key_usable(k.accept_until, KeyUse::Spend, now))
        .collect();
    keys.sort_by_key(|k| (k.issue_from, k.key_hash));

    let data: Vec<IssuerKeyResponse> = keys
        .into_iter()
        .map(|k| IssuerKeyResponse {
            id: hex::encode(k.key_hash),
            public_key: URL_SAFE_NO_PAD.encode(&k.public_key),
            domain_separator: domain_separator(),
            // The calendar's one spelling, which the client compares
            // byte for byte. Every cached key's instants are whole seconds.
            issue_from: wire_instant(k.issue_from),
            issue_until: wire_instant(k.issue_until),
            accept_until: wire_instant(k.accept_until),
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

    let key_hash = ensure_keys(&state.credential_key_cache, master_key, &state.db_pool).await?;

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

    // Load the issuer key. Returning a stored refund token finishes a spend
    // already accepted, so it is allowed until the key's retention deadline.
    load_key(
        &state.credential_key_cache,
        master_key,
        &state.db_pool,
        &act.issuer_key_hash,
        KeyUse::Refund,
    )
    .await?;

    // The proof must verify before anything about its nullifier is read or
    // written. Recording an unverified nullifier would let anyone holding a
    // public key id write rows, and answering 200 or 404 for one would let
    // anyone who knows a nullifier ask what became of its spend.
    //
    // The one verification also signs the full refund the never-arrived
    // branch returns (the verifier cannot verify without signing). It leaves
    // the gateway only if that branch records the nullifier; otherwise it is
    // dropped.
    let full_refund = {
        let cache = state.credential_key_cache.read().await;
        verify_and_sign_full_refund(&cache, &act, SystemTime::now())?
    };

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
        let Some(refund_cbor) =
            open_refund_token(&act.refund_key, &key_id, &nullifier_bytes, &sealed)
        else {
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

    // Anything past here may record a fresh nullifier, which is accepting a
    // spend: only before the key's acceptance deadline, like any spend.
    {
        let cache = state.credential_key_cache.read().await;
        usable_key(
            &cache,
            &act.issuer_key_hash,
            KeyUse::Spend,
            SystemTime::now(),
        )?;
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
    // arrived. The full refund (the charge returned in full, since no work
    // was done) was signed when the proof was verified above.
    let SignedRefund {
        cbor: refund_cbor,
        issuer_key_id: key_id,
    } = full_refund;

    // Store the refund token for idempotency, sealed (it carries the amount).
    store_sealed_refund(&state, &act, &key_id, &refund_cbor).await?;

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
        let encrypted = encrypt_private_key(&master_key, key_id, b"aad", plaintext).unwrap();
        let decrypted = decrypt_private_key(&master_key, key_id, b"aad", &encrypted).unwrap();
        assert_eq!(plaintext.as_slice(), &decrypted);
    }

    #[test]
    fn test_decrypt_wrong_key_id_fails() {
        let master_key = [42u8; 32];
        let encrypted = encrypt_private_key(
            &master_key,
            "550e8400-e29b-41d4-a716-446655440000",
            b"aad",
            b"secret",
        )
        .unwrap();
        let result = decrypt_private_key(
            &master_key,
            "660e8400-e29b-41d4-a716-446655440000",
            b"aad",
            &encrypted,
        );
        assert!(result.is_err());
    }

    // --- Key records bound to their windows ---

    const WEEK_SECS: u64 = 7 * 86_400;

    fn current_epoch() -> u64 {
        epoch_at(SystemTime::now()).unwrap()
    }

    /// Every field the associated data covers is load-bearing: a private key
    /// encrypted under one record's fields decrypts under no other's.
    #[test]
    fn an_issuer_key_decrypts_only_under_its_own_record() {
        let master_key = [7u8; 32];
        let row = generate_key(&master_key, current_epoch()).unwrap();
        let windows = calendar_windows(row.issue_from, row.issue_until, row.accept_until).unwrap();
        let aad = |id: &str, pk: &[u8], ds: &str, w: EpochWindows| issuer_key_aad(id, pk, ds, w);
        let open =
            |aad: Vec<u8>| decrypt_private_key(&master_key, &row.id, &aad, &row.private_key_enc);

        assert!(
            open(aad(
                &row.id,
                &row.public_key,
                &row.domain_separator,
                windows
            ))
            .is_ok()
        );

        let mut other_pk = row.public_key.clone();
        other_pk[0] ^= 1;
        let other_id = hex::encode([0xAB; 32]);
        let edits = [
            (
                "id",
                aad(&other_id, &row.public_key, &row.domain_separator, windows),
            ),
            (
                "public key",
                aad(&row.id, &other_pk, &row.domain_separator, windows),
            ),
            (
                "domain separator",
                aad(
                    &row.id,
                    &row.public_key,
                    "ACT-v1:eidola:inference:production:2099-01-01",
                    windows,
                ),
            ),
            (
                "issue_from",
                aad(
                    &row.id,
                    &row.public_key,
                    &row.domain_separator,
                    EpochWindows {
                        issue_from: windows.issue_from + 1,
                        ..windows
                    },
                ),
            ),
            (
                "issue_until",
                aad(
                    &row.id,
                    &row.public_key,
                    &row.domain_separator,
                    EpochWindows {
                        issue_until: windows.issue_until + WEEK_SECS,
                        ..windows
                    },
                ),
            ),
            (
                "accept_until",
                aad(
                    &row.id,
                    &row.public_key,
                    &row.domain_separator,
                    EpochWindows {
                        accept_until: windows.accept_until + WEEK_SECS,
                        ..windows
                    },
                ),
            ),
        ];
        for (field, edited) in edits {
            assert!(open(edited).is_err(), "an edited {field} still decrypts");
        }
        // Without associated data (the old format's shape) it does not open.
        assert!(open(Vec::new()).is_err());
    }

    /// The same, through the path every key takes into use: a row whose
    /// plain columns were edited is refused, including the edit that keeps
    /// the windows on the calendar — relabelling an old key as the current
    /// epoch's.
    #[test]
    fn an_edited_key_row_is_refused() {
        let master_key = [7u8; 32];
        let n = current_epoch();
        let row = generate_key(&master_key, n).unwrap();
        let mut cache = Cache::new();
        let hash = cache_key(&mut cache, &master_key, &row).unwrap();
        assert_eq!(cache[&hash].epoch, n);

        let week = Duration::from_secs(WEEK_SECS);
        let other = generate_key(&master_key, n).unwrap();
        type Edit = fn(&mut db::IssuerKeyRow, &db::IssuerKeyRow, Duration);
        let edits: [(&str, Edit); 8] = [
            ("issue_from", |r, _, _| {
                r.issue_from += Duration::from_secs(1)
            }),
            ("issue_until", |r, _, w| r.issue_until += w),
            ("accept_until", |r, _, w| r.accept_until += w),
            ("a whole epoch later", |r, _, w| {
                r.issue_from += w;
                r.issue_until += w;
                r.accept_until += w;
            }),
            ("a whole epoch earlier", |r, _, w| {
                r.issue_from -= w;
                r.issue_until -= w;
                r.accept_until -= w;
            }),
            ("domain separator", |r, _, _| {
                r.domain_separator = "ACT-v1:eidola:inference:production:2099-01-01".into()
            }),
            ("id and public key", |r, o, _| {
                r.id = o.id.clone();
                r.public_key = o.public_key.clone();
            }),
            ("ciphertext", |r, _, _| {
                let last = r.private_key_enc.len() - 1;
                r.private_key_enc[last] ^= 1;
            }),
        ];
        for (field, edit) in edits {
            let mut edited = db::IssuerKeyRow {
                id: row.id.clone(),
                private_key_enc: row.private_key_enc.clone(),
                public_key: row.public_key.clone(),
                domain_separator: row.domain_separator.clone(),
                issue_from: row.issue_from,
                issue_until: row.issue_until,
                accept_until: row.accept_until,
            };
            edit(&mut edited, &other, week);
            let mut fresh = Cache::new();
            assert!(
                cache_key(&mut fresh, &master_key, &edited).is_err(),
                "a row with an edited {field} was used"
            );
            assert!(fresh.is_empty());
        }
    }

    /// A key whose windows are off the calendar is refused even when its
    /// record authenticates: the gateway uses no key a client could not check.
    #[test]
    fn a_key_off_the_calendar_is_refused_even_when_it_authenticates() {
        let master_key = [7u8; 32];
        let secret_key = PrivateKey::random(OsRng);
        let public_key = secret_key.public().to_cbor().unwrap();
        let id = hex::encode(compute_key_hash(&public_key));
        let (issue_from, _, _) = epoch_instants(current_epoch()).unwrap();
        // The pre-calendar shape: a window starting at an arbitrary instant.
        let issue_from = issue_from + Duration::from_secs(3_600);
        let issue_until = issue_from + Duration::from_secs(WEEK_SECS);
        let accept_until = issue_until + Duration::from_secs(WEEK_SECS);
        let windows = calendar_windows(issue_from, issue_until, accept_until).unwrap();
        let ds = domain_separator();
        let aad = issuer_key_aad(&id, &public_key, &ds, windows);
        let row = db::IssuerKeyRow {
            private_key_enc: encrypt_private_key(
                &master_key,
                &id,
                &aad,
                &secret_key.to_cbor().unwrap(),
            )
            .unwrap(),
            id,
            public_key,
            domain_separator: ds,
            issue_from,
            issue_until,
            accept_until,
        };
        let mut cache = Cache::new();
        assert!(cache_key(&mut cache, &master_key, &row).is_err());
        assert!(cache.is_empty());
    }

    #[test]
    fn generated_keys_sit_exactly_on_the_calendar() {
        let master_key = [7u8; 32];
        let n = current_epoch();
        for epoch in [n, n + 1] {
            let row = generate_key(&master_key, epoch).unwrap();
            let secs = |t: SystemTime| t.duration_since(UNIX_EPOCH).unwrap().as_secs();
            assert_eq!(
                secs(row.issue_from) % WEEK_SECS,
                0,
                "a boundary is a multiple of a week"
            );
            assert_eq!(secs(row.issue_until), secs(row.issue_from) + WEEK_SECS);
            assert_eq!(secs(row.accept_until), secs(row.issue_until) + WEEK_SECS);
            assert_eq!(
                canonical_epoch(row.issue_from, row.issue_until, row.accept_until),
                Some(epoch)
            );
        }
        let now = SystemTime::now();
        let (from, until, _) = epoch_instants(n).unwrap();
        assert!(from <= now && now < until, "the current epoch covers now");
        // Sub-second instants are never on the calendar.
        let (f, u, a) = epoch_instants(n).unwrap();
        let nudge = Duration::from_millis(1);
        assert_eq!(canonical_epoch(f + nudge, u + nudge, a + nudge), None);
    }

    #[test]
    fn a_gateway_mints_the_current_key_then_the_next_and_never_a_second() {
        use std::collections::HashSet;
        let n = 100;
        let none = HashSet::new();
        let shows = |epochs: &[u64]| epochs.iter().copied().collect::<HashSet<u64>>();
        assert_eq!(epoch_to_mint(&shows(&[]), n, &none).unwrap(), Some(n));
        assert_eq!(epoch_to_mint(&shows(&[n + 1]), n, &none).unwrap(), Some(n));
        assert_eq!(epoch_to_mint(&shows(&[n]), n, &none).unwrap(), Some(n + 1));
        assert_eq!(epoch_to_mint(&shows(&[n, n + 1]), n, &none).unwrap(), None);

        // A database that hides the current key (or every key) from an
        // instance that holds it does not get a second key for the epoch.
        let held: HashSet<u64> = [n].into();
        assert!(epoch_to_mint(&shows(&[]), n, &held).is_err());
        assert!(epoch_to_mint(&shows(&[n + 1]), n, &held).is_err());
        let held: HashSet<u64> = [n, n + 1].into();
        assert!(epoch_to_mint(&shows(&[n]), n, &held).is_err());
        assert_eq!(epoch_to_mint(&shows(&[n, n + 1]), n, &held).unwrap(), None);
    }

    #[test]
    fn two_keys_for_one_epoch_issue_under_neither() {
        let master_key = [7u8; 32];
        let n = current_epoch();
        let mut cache = Cache::new();
        assert_eq!(sole_key_for_epoch(&cache, n).unwrap(), None);
        let first = cache_key(
            &mut cache,
            &master_key,
            &generate_key(&master_key, n).unwrap(),
        )
        .unwrap();
        assert_eq!(sole_key_for_epoch(&cache, n).unwrap(), Some(first));
        cache_key(
            &mut cache,
            &master_key,
            &generate_key(&master_key, n + 1).unwrap(),
        )
        .unwrap();
        assert_eq!(sole_key_for_epoch(&cache, n).unwrap(), Some(first));
        cache_key(
            &mut cache,
            &master_key,
            &generate_key(&master_key, n).unwrap(),
        )
        .unwrap();
        assert!(sole_key_for_epoch(&cache, n).is_err());
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

    // --- Key deadlines ---

    use anonymous_credit_tokens::{CreditToken, PreIssuance, PublicKey, scalar_to_credit};

    const HOUR: Duration = Duration::from_secs(3600);
    const DAY: Duration = Duration::from_secs(86400);

    type Cache = HashMap<[u8; 32], DecryptedIssuerKey>;

    /// Cache a fresh key whose acceptance window ends at `accept_until`,
    /// returning its hash. (Keys are built straight into the map: `Params`
    /// is large enough that holding several on an unoptimized test stack
    /// overflows it.)
    fn add_key(cache: &mut Cache, accept_until: SystemTime) -> [u8; 32] {
        let secret_key = PrivateKey::random(OsRng);
        let public_key = secret_key.public().to_cbor().unwrap();
        let key_hash = compute_key_hash(&public_key);
        cache.insert(
            key_hash,
            DecryptedIssuerKey {
                secret_key,
                params: Params::new(DS_ORG, DS_SERVICE, DS_DEPLOYMENT, DS_VERSION),
                key_hash,
                public_key,
                request_context_scalar: request_context_to_scalar(&compute_request_context(
                    &key_hash,
                )),
                // These tests are about deadlines, not the calendar; no two
                // of them ask which key is an epoch's.
                epoch: 0,
                issue_from: accept_until - 14 * DAY,
                issue_until: accept_until - 7 * DAY,
                accept_until,
            },
        );
        key_hash
    }

    /// A credential of `credits` issued under `key`, as the client holds it.
    fn credential(key: &DecryptedIssuerKey, credits: u128) -> CreditToken {
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
        pre.to_credit_token::<128>(&key.params, key.secret_key.public(), &request, &response)
            .unwrap()
    }

    /// Spend `charge` of a fresh credential under `key`, returning the token
    /// the server sees and the client's pre-refund state.
    fn spend(
        key: &DecryptedIssuerKey,
        credits: u128,
        charge: u128,
    ) -> (crate::auth::ActSpend, anonymous_credit_tokens::PreRefund) {
        let token = credential(key, credits);
        let (spend_proof, pre_refund) = token
            .prove_spend::<128>(&key.params, credit_to_scalar::<128>(charge).unwrap(), OsRng)
            .unwrap();
        let act = crate::auth::ActSpend {
            challenge_digest: compute_challenge_digest(),
            issuer_key_hash: key.key_hash,
            spend_proof,
            refund_key: RefundKey::from_bytes([0x11; 32]),
        };
        (act, pre_refund)
    }

    #[test]
    fn a_cached_key_past_accept_until_refuses_a_spend() {
        let accept_until = SystemTime::now() + DAY;
        let mut cache = Cache::new();
        let hash = add_key(&mut cache, accept_until);
        let second = Duration::from_secs(1);

        assert!(usable_key(&cache, &hash, KeyUse::Spend, accept_until - second).is_ok());
        for now in [accept_until, accept_until + second, accept_until + HOUR] {
            assert!(matches!(
                usable_key(&cache, &hash, KeyUse::Spend, now),
                Err(ServerError::Unauthorized { .. })
            ));
        }
    }

    #[test]
    fn a_spend_accepted_in_time_still_gets_its_refund_after_the_deadline() {
        let accept_until = SystemTime::now() + DAY;
        let mut cache = Cache::new();
        let hash = add_key(&mut cache, accept_until);
        let just_after = accept_until + Duration::from_secs(30);
        let retired = retain_until(accept_until);

        assert!(usable_key(&cache, &hash, KeyUse::Spend, just_after).is_err());
        assert!(usable_key(&cache, &hash, KeyUse::Refund, just_after).is_ok());
        assert!(
            usable_key(
                &cache,
                &hash,
                KeyUse::Refund,
                retired - Duration::from_secs(1)
            )
            .is_ok()
        );
        assert!(usable_key(&cache, &hash, KeyUse::Refund, retired).is_err());
    }

    #[tokio::test]
    async fn load_key_refuses_a_cached_key_past_its_deadline_without_the_database() {
        // An unreachable database: reaching the slow path would answer with a
        // pool error, never the refusal asserted here.
        let pool = db::create_pool(
            "postgres://nobody@127.0.0.1:1/none?sslmode=disable",
            None,
            None,
        )
        .unwrap();
        let master_key = [0u8; 32];
        let now = SystemTime::now();

        // Past acceptance, still retained: refused for a spend, kept cached
        // for the refund of a spend already accepted.
        let mut keys = Cache::new();
        let lapsing_hash = add_key(&mut keys, now - Duration::from_secs(60));
        // Past retention: refused for everything, and dropped from the cache.
        let retired_hash = add_key(&mut keys, now - RETENTION_PAST_ACCEPT - HOUR);
        // Live: accepted from the cache.
        let live_hash = add_key(&mut keys, now + DAY);
        let cache: KeyCache = Arc::new(RwLock::new(keys));

        let load = |hash: [u8; 32], key_use| {
            let cache = cache.clone();
            let pool = pool.clone();
            async move { load_key(&cache, &master_key, &pool, &hash, key_use).await }
        };

        assert!(load(live_hash, KeyUse::Spend).await.is_ok());
        assert!(matches!(
            load(lapsing_hash, KeyUse::Spend).await,
            Err(ServerError::Unauthorized { .. })
        ));
        assert!(load(lapsing_hash, KeyUse::Refund).await.is_ok());
        assert!(cache.read().await.contains_key(&lapsing_hash));
        assert!(matches!(
            load(retired_hash, KeyUse::Refund).await,
            Err(ServerError::Unauthorized { .. })
        ));
        assert!(!cache.read().await.contains_key(&retired_hash));
    }

    #[test]
    fn eviction_drops_only_keys_past_retention() {
        let now = SystemTime::now();
        let mut cache = Cache::new();
        let live_hash = add_key(&mut cache, now + DAY);
        let lapsing_hash = add_key(&mut cache, now - Duration::from_secs(60));
        let retired_hash = add_key(&mut cache, now - RETENTION_PAST_ACCEPT);

        assert_eq!(evict_retired_keys(&mut cache, now), 1);
        assert!(cache.contains_key(&live_hash));
        assert!(cache.contains_key(&lapsing_hash));
        assert!(!cache.contains_key(&retired_hash));
    }

    /// The invariant: no nullifier is pruned while its key could still accept
    /// a spend — not on this instance's clock, and not on any instance's
    /// clock within the skew the gateway tolerates against the database's.
    #[test]
    fn no_nullifier_is_pruned_while_its_key_could_accept_a_spend() {
        let accept_until = SystemTime::now() + DAY;
        let skew = db::MAX_CLOCK_SKEW;
        let step = Duration::from_secs(5);
        let mut t = accept_until - HOUR;
        while t < retain_until(accept_until) + HOUR {
            let prunable = nullifiers_prunable(accept_until, t);
            for instance_clock in [t - skew, t, t + skew] {
                if prunable {
                    assert!(
                        !key_usable(accept_until, KeyUse::Spend, instance_clock),
                        "nullifiers prunable while a spend could be accepted"
                    );
                }
            }
            // Pruning starts exactly where every use of the key ends.
            assert_eq!(prunable, !key_usable(accept_until, KeyUse::Refund, t));
            t += step;
        }
        assert!(RETENTION_PAST_ACCEPT > 2 * skew);
    }

    /// The SQL pruning condition is the one `nullifiers_prunable` states,
    /// parameterized by the margin the gateway passes in.
    #[test]
    fn the_schema_prunes_on_the_shared_deadline() {
        let schema = include_str!("../schema/schema.sql");
        assert!(schema.contains(
            "CREATE OR REPLACE FUNCTION prune_expired_nullifiers(p_retain_past_accept INTERVAL)"
        ));
        assert!(schema.contains("WHERE accept_until + p_retain_past_accept < now()"));
        assert!(schema.contains("DROP FUNCTION IF EXISTS prune_expired_nullifiers();"));
    }

    /// A refund is signed under the spent credential's own key, never the
    /// current issuing key, so a spend near the end of an epoch cannot carry
    /// value into a later one.
    #[test]
    fn a_refund_is_signed_under_the_spent_credentials_key() {
        let now = SystemTime::now();
        // The spent credential's key has stopped issuing and lapses in an hour;
        // a newer key is the one currently issuing.
        let mut cache = Cache::new();
        let old_hash = add_key(&mut cache, now + HOUR);
        let current_hash = add_key(&mut cache, now + 8 * DAY);
        let old_public: PublicKey = cache[&old_hash].secret_key.public().clone();
        let current_public: PublicKey = cache[&current_hash].secret_key.public().clone();
        let (act, pre_refund) = spend(&cache[&old_hash], 100, 10);

        for at in [now, now + HOUR + Duration::from_secs(60)] {
            let (refund_cbor, key_id) = sign_refund(&cache, &act, 7, at).unwrap();
            assert_eq!(key_id, hex::encode(old_hash));
            let refund = anonymous_credit_tokens::Refund::from_cbor(&refund_cbor).unwrap();
            let params = Params::new(DS_ORG, DS_SERVICE, DS_DEPLOYMENT, DS_VERSION);
            let token = pre_refund
                .to_credit_token::<128>(&params, &act.spend_proof, &refund, &old_public)
                .expect("refund completes under the spent credential's key");
            assert_eq!(
                scalar_to_credit::<128>(&token.credits()).unwrap(),
                100 - 10 + 7
            );
            assert!(
                pre_refund
                    .to_credit_token::<128>(&params, &act.spend_proof, &refund, &current_public)
                    .is_err(),
                "a refund must not complete under any other key"
            );
        }

        // Past retention the spent credential's key signs nothing, and no
        // other key stands in for it.
        assert!(matches!(
            sign_refund(&cache, &act, 7, retain_until(now + HOUR)),
            Err(ServerError::Unauthorized { .. })
        ));
    }

    /// The database's own filters against the shared deadline: a key retained
    /// past `accept_until` is still found, and pruning spares its nullifiers
    /// until retention ends. Needs `DATABASE_URL` with `schema.sql` applied.
    #[tokio::test]
    #[ignore]
    async fn pruning_and_lookup_honour_the_retention_margin() {
        let url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
        let pool = db::create_pool(&url, None, None).unwrap();
        let master_key = [0u8; 32];
        let now = SystemTime::now();
        // This exercises the database's own filters, which read the plain
        // columns; the rows are placed off the calendar so their deadlines can
        // sit minutes from now (the gateway would never use them).
        let row_ending_at = |accept_until: SystemTime| {
            let mut row = generate_key(&master_key, 0).unwrap();
            row.issue_from = accept_until - 14 * DAY;
            row.issue_until = accept_until - 7 * DAY;
            row.accept_until = accept_until;
            row
        };

        // accept_until passed 10 minutes ago: retained.
        let lapsing = row_ending_at(now - Duration::from_secs(600));
        // accept_until passed past the retention margin: retired.
        let retired = row_ending_at(now - RETENTION_PAST_ACCEPT - HOUR);

        let client = pool.get().await.unwrap();
        for key in [&lapsing, &retired] {
            client
                .execute(
                    "INSERT INTO issuer_key (id, private_key_enc, public_key, domain_separator, \
                     issue_from, issue_until, accept_until) VALUES ($1,$2,$3,$4,$5,$6,$7)",
                    &[
                        &key.id,
                        &key.private_key_enc,
                        &key.public_key,
                        &key.domain_separator,
                        &key.issue_from,
                        &key.issue_until,
                        &key.accept_until,
                    ],
                )
                .await
                .unwrap();
            assert!(
                db::record_nullifier(&pool, &key.id, &[7u8; 32])
                    .await
                    .unwrap()
            );
        }

        let lapsing_hash = compute_key_hash(&lapsing.public_key);
        let retired_hash = compute_key_hash(&retired.public_key);
        assert!(
            db::get_issuer_key_by_hash(&pool, &lapsing_hash, RETENTION_PAST_ACCEPT)
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            db::get_issuer_key_by_hash(&pool, &retired_hash, RETENTION_PAST_ACCEPT)
                .await
                .unwrap()
                .is_none()
        );

        db::prune_expired_nullifiers(&pool, RETENTION_PAST_ACCEPT)
            .await
            .unwrap();
        // The lapsing key's nullifier survives: recording it again is refused
        // as a duplicate. The retired key's is gone.
        assert!(
            !db::record_nullifier(&pool, &lapsing.id, &[7u8; 32])
                .await
                .unwrap()
        );
        assert!(
            db::record_nullifier(&pool, &retired.id, &[7u8; 32])
                .await
                .unwrap()
        );

        // A non-positive margin is refused outright.
        assert!(
            db::prune_expired_nullifiers(&pool, Duration::ZERO)
                .await
                .is_err()
        );
    }

    // --- Refund-token sealing ---

    use anonymous_credit_tokens::{PreRefund, Refund, SpendProof};

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
        let ck = client_key(7);
        let (sk, params, proof, prerefund, cbor) = refund_fixture(1000, 300, 117);
        let nullifier = proof.nullifier().as_bytes().to_vec();

        let sealed = seal_refund_token(&ck, KEY_ID, &nullifier, &cbor).unwrap();
        let opened = open_refund_token(&ck, KEY_ID, &nullifier, &sealed).unwrap();
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
        let ck = client_key(7);
        let (_, _, proof, _, cbor) = refund_fixture(1000, 300, 117);
        let nullifier = proof.nullifier().as_bytes().to_vec();
        let sealed = seal_refund_token(&ck, KEY_ID, &nullifier, &cbor).unwrap();

        // Moved to another nullifier under the same key.
        let (_, _, other_proof, _, _) = refund_fixture(1000, 300, 117);
        let other_nullifier = other_proof.nullifier().as_bytes().to_vec();
        assert!(open_refund_token(&ck, KEY_ID, &other_nullifier, &sealed).is_none());

        // Moved to the same nullifier value under another issuer key.
        let other_key = "0".repeat(64);
        assert!(open_refund_token(&ck, &other_key, &nullifier, &sealed).is_none());

        // Any single altered byte.
        for i in 0..sealed.as_bytes().len() {
            let mut bytes = sealed.as_bytes().to_vec();
            bytes[i] ^= 0x01;
            let altered = SealedRefundToken::from_stored(bytes);
            assert!(
                open_refund_token(&ck, KEY_ID, &nullifier, &altered).is_none(),
                "byte {i} altered and the token still opened"
            );
        }

        // A row written before tokens were sealed holds the bare CBOR.
        assert_eq!(cbor[0], 0xa5);
        let legacy = SealedRefundToken::from_stored(cbor.clone());
        assert!(open_refund_token(&ck, KEY_ID, &nullifier, &legacy).is_none());
        let empty = SealedRefundToken::from_stored(Vec::new());
        assert!(open_refund_token(&ck, KEY_ID, &nullifier, &empty).is_none());
    }

    #[test]
    fn a_sealed_refund_token_carries_no_plaintext_amount() {
        let ck = client_key(7);
        let t: u128 = 0x0123_4567_89ab_cdef_0011_2233_4455_6677;
        let (_, _, proof, _, cbor) = refund_fixture(u128::MAX >> 1, t, t);
        let nullifier = proof.nullifier().as_bytes().to_vec();
        let t_bytes = credit_to_scalar::<128>(t).unwrap().to_bytes();
        // The bare token does carry the amount: that is what is being hidden.
        assert!(contains(&cbor, &t_bytes));

        let sealed = seal_refund_token(&ck, KEY_ID, &nullifier, &cbor).unwrap();
        let bytes = sealed.as_bytes();
        assert!(!contains(bytes, &t_bytes));
        // Nor the amount's own bytes, nor any run of the CBOR.
        assert!(!contains(bytes, &t.to_le_bytes()));
        assert!(!cbor.windows(12).any(|w| contains(bytes, w)));

        // And the length is the same whatever the amount.
        let (_, _, p0, _, c0) = refund_fixture(1000, 300, 0);
        let s0 = seal_refund_token(&ck, KEY_ID, p0.nullifier().as_bytes(), &c0).unwrap();
        assert_eq!(s0.as_bytes().len(), bytes.len());
        assert_eq!(bytes.len(), 1 + 12 + cbor.len() + 16);
    }

    #[test]
    fn the_refund_token_key_is_its_own() {
        let ck = client_key(7);
        // Not the issuer-key wrapping key's derivation over the same bytes...
        assert_ne!(
            *derive_refund_token_key(&ck, KEY_ID),
            derive_key_encryption_key(ck.as_bytes(), KEY_ID)
        );
        // ...and not another issuer key's refund-token key...
        assert_ne!(
            *derive_refund_token_key(&ck, KEY_ID),
            *derive_refund_token_key(&ck, &"0".repeat(64))
        );
        // ...and not another client's.
        assert_ne!(
            *derive_refund_token_key(&ck, KEY_ID),
            *derive_refund_token_key(&client_key(8), KEY_ID)
        );
    }

    #[test]
    fn a_sealed_refund_token_opens_only_with_its_clients_key() {
        let ck = client_key(7);
        let (_, _, proof, _, cbor) = refund_fixture(1000, 300, 117);
        let nullifier = proof.nullifier().as_bytes().to_vec();
        let sealed = seal_refund_token(&ck, KEY_ID, &nullifier, &cbor).unwrap();

        // A recovery presenting any other key opens nothing.
        for other in [client_key(8), client_key(0)] {
            assert!(open_refund_token(&other, KEY_ID, &nullifier, &sealed).is_none());
        }
        let mut near = [7u8; 32];
        near[31] ^= 0x01;
        assert!(
            open_refund_token(&RefundKey::from_bytes(near), KEY_ID, &nullifier, &sealed).is_none()
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
        let authorization = eidola_common::private_token::authorization_value(&token);
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

    /// A spend's token is read only in the spelling RFC 9577 requires and
    /// the client writes: padded base64url. The same Token without its
    /// padding is refused as unauthenticated by the extractor, before the
    /// handler that records the nullifier runs.
    #[tokio::test]
    async fn a_spend_token_is_accepted_only_as_padded_base64url() {
        use axum::extract::FromRequestParts;
        use base64::engine::general_purpose::URL_SAFE;
        use eidola_common::refund_key::{REFUND_KEY_HEADER, encode_refund_key};

        let (_, _, proof, _, _) = refund_fixture(1000, 300, 0);
        let mut token = Vec::new();
        token.extend_from_slice(&ACT_TOKEN_TYPE.to_be_bytes());
        token.extend_from_slice(&compute_challenge_digest());
        token.extend_from_slice(&[9u8; 32]);
        token.extend_from_slice(&proof.to_cbor().unwrap());
        // The fixture's Token needs padding, so the two spellings differ.
        assert_ne!(token.len() % 3, 0);
        let refund_key = encode_refund_key(&[0x5au8; 32]);
        let refund_key = std::str::from_utf8(&refund_key).unwrap().to_string();

        let extract = |authorization: String| {
            let refund_key = refund_key.clone();
            async move {
                let request = axum::http::Request::builder()
                    .header("authorization", authorization)
                    .header(REFUND_KEY_HEADER, refund_key);
                let (mut parts, ()) = request.body(()).unwrap().into_parts();
                TokenAuth::from_request_parts(&mut parts, &()).await
            }
        };

        let padded = format!("PrivateToken token=\"{}\"", URL_SAFE.encode(&token));
        assert_eq!(
            padded,
            eidola_common::private_token::authorization_value(&token)
        );
        let TokenAuth(act) = extract(padded).await.unwrap();
        assert_eq!(act.spend_proof.nullifier(), proof.nullifier());

        let unpadded = format!("PrivateToken token=\"{}\"", URL_SAFE_NO_PAD.encode(&token));
        match extract(unpadded).await {
            Err(ServerError::Unauthorized { .. }) => {}
            Err(e) => panic!("expected 401, got {e:?}"),
            Ok(_) => panic!("an unpadded token was accepted"),
        }
    }

    /// What anyone holds without the client: the stored row (version, nonce,
    /// ciphertext) and the public facts that address it (the issuer key id,
    /// the nullifier). The key is derived from the client's key alone, so
    /// nothing built from those opens the row; only the client's key does.
    #[test]
    fn the_stored_row_alone_without_the_client_key_opens_nothing() {
        let random_key = || {
            let mut bytes = [0u8; 32];
            OsRng.fill_bytes(&mut bytes);
            RefundKey::from_bytes(bytes)
        };
        let ck = random_key();
        let (_, _, proof, _, cbor) = refund_fixture(1000, 300, 117);
        let nullifier = proof.nullifier().as_bytes().to_vec();
        let sealed = seal_refund_token(&ck, KEY_ID, &nullifier, &cbor).unwrap();
        let row = sealed.as_bytes();

        let as_key = |bytes: &[u8]| {
            let mut key = [0u8; 32];
            key.copy_from_slice(&Sha256::digest(bytes));
            RefundKey::from_bytes(key)
        };
        let guesses = [
            random_key(),
            as_key(&nullifier),
            as_key(KEY_ID.as_bytes()),
            as_key(&row[1..1 + SEALED_REFUND_NONCE_LEN]),
            as_key(row),
        ];
        for guess in &guesses {
            assert!(open_refund_token(guess, KEY_ID, &nullifier, &sealed).is_none());
        }
        assert_eq!(
            open_refund_token(&ck, KEY_ID, &nullifier, &sealed).as_deref(),
            Some(cbor.as_slice())
        );
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
        let key_hash = ensure_keys(&cache, &master_key, &pool).await.unwrap();
        let state = AppState::new(
            Catalog::new(None),
            None,
            pool.clone(),
            None,
            None,
            master_key,
            cache.clone(),
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

        // The original request never arrived: a full refund is issued, its
        // proof verified once...
        let first = recover(proof.clone(), ck.clone()).await.unwrap();
        assert_eq!(verification_count::of(&proof), 1);
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
        assert_eq!(verification_count::of(&proof), 2, "one per ask");

        // Asked again with its own key, the stored token comes back and
        // completes the credential.
        let second = recover(proof.clone(), ck.clone()).await.unwrap();
        assert_eq!(second, first);
        assert_eq!(verification_count::of(&proof), 3, "one per ask");
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

    /// Refund recovery verifies the presented proof before it reads or writes
    /// anything about its nullifier. The database here is unreachable, so a
    /// request that got as far as the nullifier row would fail with a pool
    /// error rather than the refusal asserted; a valid proof getting that pool
    /// error is what shows the database is where the check sits in front of.
    #[test]
    fn recovery_refuses_an_unverified_proof_before_the_database() {
        // `Params` is large, and this holds several: an unoptimized test's
        // default stack is too small.
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(recovery_refuses_an_unverified_proof())
            })
            .unwrap()
            .join()
            .unwrap();
    }

    async fn recovery_refuses_an_unverified_proof() {
        let pool = db::create_pool(
            "postgres://nobody@127.0.0.1:1/none?sslmode=disable",
            None,
            None,
        )
        .unwrap();
        let mut keys = Cache::new();
        let hash = add_key(&mut keys, SystemTime::now() + DAY);
        let (valid, _) = spend(&keys[&hash], 100, 10);

        // A credential from another secret key, issued with this key's own
        // request context: its proof passes the context check and fails only
        // the cryptographic verification.
        let forged = {
            let mut other = Cache::new();
            let other_hash = add_key(&mut other, SystemTime::now() + DAY);
            let other_key = other.get_mut(&other_hash).unwrap();
            other_key.request_context_scalar = keys[&hash].request_context_scalar;
            let (mut act, _) = spend(other_key, 100, 10);
            act.issuer_key_hash = hash;
            act
        };
        // A proof under another key's context, presented under this key.
        let wrong_context = {
            let mut other = Cache::new();
            let other_hash = add_key(&mut other, SystemTime::now() + DAY);
            let (mut act, _) = spend(&other[&other_hash], 100, 10);
            act.issuer_key_hash = hash;
            act
        };

        let cache: KeyCache = Arc::new(RwLock::new(keys));
        let state = AppState::new(
            crate::backend::Catalog::new(None),
            None,
            pool,
            None,
            None,
            [0u8; 32],
            cache,
        );
        let recover = |act: crate::auth::ActSpend| {
            let state = state.clone();
            async move { recover_refund(TokenAuth(act), State(state)).await }
        };

        // The forged proof reaches the verifier once; a proof under another
        // context is refused before it.
        for (name, act, verified) in [("forged", forged, 1), ("wrong context", wrong_context, 0)] {
            let proof = act.spend_proof.clone();
            match recover(act).await {
                Err(ServerError::Unauthorized { .. }) => {}
                Err(e) => panic!("{name}: reached the database before verifying: {e}"),
                Ok(_) => panic!("{name}: an unverified proof was answered"),
            }
            assert_eq!(verification_count::of(&proof), verified, "{name}");
        }
        let proof = valid.spend_proof.clone();
        match recover(valid).await {
            Err(ServerError::Unauthorized { .. }) => {
                panic!("a valid proof was refused")
            }
            Err(_) => {}
            Ok(_) => panic!("an unreachable database answered"),
        }
        assert_eq!(verification_count::of(&proof), 1);
    }

    /// **The gateway calls the ACT verifier in one place.** The crate's only
    /// verifier is `PrivateKey::refund`, so every call to it, outside tests,
    /// must be `verify_and_refund`'s: that is what makes the
    /// `verification_count` seam count every verification of a spend proof.
    #[test]
    fn the_verifier_is_called_only_through_its_counting_seam() {
        fn sources(dir: &std::path::Path, out: &mut Vec<(String, String)>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    sources(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let text = std::fs::read_to_string(&path).unwrap();
                    out.push((path.display().to_string(), text));
                }
            }
        }
        let mut files = Vec::new();
        sources(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
            &mut files,
        );
        assert!(files.len() > 10, "the scan reads the crate's sources");
        let mut calls = Vec::new();
        for (path, text) in &files {
            let shipped = text
                .find("\n#[cfg(test)]\nmod tests")
                .map_or(text.as_str(), |end| &text[..end]);
            for pattern in [".refund(", ".refund::<", "::refund(", "::refund::<"] {
                for (at, _) in shipped.match_indices(pattern) {
                    calls.push((path.clone(), at));
                }
            }
        }
        assert_eq!(calls.len(), 1, "{calls:?}");
        let (path, at) = &calls[0];
        assert!(path.ends_with("credentials.rs"), "{path}");
        let source = include_str!("credentials.rs");
        let seam = source.find("fn verify_and_refund(").unwrap();
        let seam_end = seam + source[seam..].find("\n}\n").unwrap();
        assert!(
            (seam..seam_end).contains(at),
            "the call is verify_and_refund's"
        );
        assert!(source[seam..seam_end].contains("verification_count::record("));
    }

    /// **A chat spend's proof is verified at pre-flight and once more when
    /// its refund is signed, and no more**, on the blocking path (the
    /// handler's own `act`) and the streaming one (a clone moved into the
    /// stream). Two is the floor the crate allows: it verifies only while
    /// signing a refund for a fixed amount, and a chat spend's amount is not
    /// known at pre-flight (`verify_and_refund`). Against an unreachable
    /// database: the key is cached, and storing the refund is best-effort.
    #[test]
    fn a_chat_spend_is_verified_at_preflight_and_once_to_sign_its_refund() {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(a_chat_spend_is_verified_twice())
            })
            .unwrap()
            .join()
            .unwrap();
    }

    async fn a_chat_spend_is_verified_twice() {
        let pool = db::create_pool(
            "postgres://nobody@127.0.0.1:1/none?sslmode=disable",
            None,
            None,
        )
        .unwrap();
        let mut keys = Cache::new();
        let hash = add_key(&mut keys, SystemTime::now() + DAY);
        let (blocking, _) = spend(&keys[&hash], 100, 10);
        let (streaming, _) = spend(&keys[&hash], 100, 10);
        let state = AppState::new(
            crate::backend::Catalog::new(None),
            None,
            pool,
            None,
            None,
            [0u8; 32],
            Arc::new(RwLock::new(keys)),
        );

        crate::handlers::verify_spend_proof(&state, &blocking)
            .await
            .unwrap();
        assert_eq!(verification_count::of(&blocking.spend_proof), 1);
        crate::handlers::issue_refund_async(&state, &blocking, 4)
            .await
            .unwrap();
        assert_eq!(verification_count::of(&blocking.spend_proof), 2);

        crate::handlers::verify_spend_proof(&state, &streaming)
            .await
            .unwrap();
        let spent = streaming.clone();
        crate::handlers::issue_refund_async(&state, &spent, 4)
            .await
            .unwrap();
        assert_eq!(verification_count::of(&streaming.spend_proof), 2);
    }

    /// Recovery's one verification signs the refund it returns: the refund
    /// verifies for the client and returns the whole charge.
    #[test]
    fn the_full_refund_signed_by_verification_completes_the_credential() {
        let mut keys = Cache::new();
        let hash = add_key(&mut keys, SystemTime::now() + DAY);
        let key = &keys[&hash];
        let (act, pre_refund) = spend(key, 100, 30);
        let signed = verify_and_sign_full_refund(&keys, &act, SystemTime::now()).unwrap();
        assert_eq!(verification_count::of(&act.spend_proof), 1);
        assert_eq!(signed.issuer_key_id, hex::encode(hash));
        let refund = anonymous_credit_tokens::Refund::from_cbor(&signed.cbor).unwrap();
        let next = pre_refund
            .to_credit_token(
                &key.params,
                &act.spend_proof,
                &refund,
                key.secret_key.public(),
            )
            .unwrap();
        assert_eq!(scalar_to_credit::<128>(&next.credits()).unwrap(), 100);
    }

    /// Two issuances of one account's whole balance cannot both pass the
    /// balance check: the second waits for the first, then sees the balance it
    /// spent. Without that, a client sending issuances in parallel converts its
    /// balance into anonymous credentials once per request. Needs
    /// `DATABASE_URL` with `schema.sql` applied.
    #[tokio::test]
    #[ignore]
    async fn concurrent_issuances_cannot_spend_one_balance_twice() {
        let url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
        let pool = db::create_pool(&url, None, None).unwrap();
        let master_key = [0u8; 32];
        // Any key the ledger can name; a far-future epoch of its own, so it
        // cannot collide with the epoch's real key in a shared database.
        let epoch = 10_000 + u64::from(OsRng.next_u32() % 1_000_000);
        let key = generate_key(&master_key, epoch).unwrap();
        let client = pool.get().await.unwrap();
        client
            .execute(
                "INSERT INTO issuer_key (id, private_key_enc, public_key, domain_separator, \
                 issue_from, issue_until, accept_until) VALUES ($1,$2,$3,$4,$5,$6,$7)",
                &[
                    &key.id,
                    &key.private_key_enc,
                    &key.public_key,
                    &key.domain_separator,
                    &key.issue_from,
                    &key.issue_until,
                    &key.accept_until,
                ],
            )
            .await
            .unwrap();
        let account = uuid::Uuid::new_v4();
        db::insert_account(&pool, account, "unused").await.unwrap();
        let event = format!("evt_test_{}", account.simple());
        assert!(
            db::insert_credit_ledger(&pool, account, 1000, "purchase", &event, None, None)
                .await
                .unwrap()
        );

        // The first issuance takes the whole balance and has not committed.
        let mut first = pool.get().await.unwrap();
        let tx = first.transaction().await.unwrap();
        let ids: Option<Vec<uuid::Uuid>> = tx
            .query_one(
                "SELECT debit_account($1, 1000, 'credential_issuance', NULL, $2, TRUE)",
                &[&account, &key.id],
            )
            .await
            .unwrap()
            .get(0);
        assert!(ids.is_some_and(|ids| !ids.is_empty()));

        // The second must wait for it rather than read the same balance.
        let second = {
            let pool = pool.clone();
            let key_id = key.id.clone();
            tokio::spawn(async move {
                db::insert_credential_issuance(&pool, account, 1000, &key_id).await
            })
        };
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(
            !second.is_finished(),
            "a second debit of the same account ran while the first was open"
        );
        tx.commit().await.unwrap();
        let second = second.await.unwrap().unwrap();
        assert!(
            second.is_none(),
            "the second issuance was granted out of a balance already spent"
        );
        assert_eq!(db::get_available_balance(&pool, account).await.unwrap(), 0);
    }

    /// Keys land on the calendar however often the gateway restarts, one per
    /// epoch, and a row whose windows the database edits stops being a key
    /// until it is put back. Needs `DATABASE_URL` with `schema.sql` applied;
    /// run with `--test-threads=1` beside the other database tests, since it
    /// edits the current key's row for a moment.
    #[tokio::test]
    #[ignore]
    async fn keys_stay_on_the_calendar_across_restarts_and_an_edited_row_is_refused() {
        let url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
        let pool = db::create_pool(&url, None, None).unwrap();
        let master_key: [u8; 32] =
            hex::decode(std::env::var("CREDENTIAL_MASTER_KEY").unwrap_or_else(|_| "00".repeat(32)))
                .unwrap()
                .try_into()
                .unwrap();
        let n = current_epoch();

        // Every restart (a fresh cache) finds the same current key and mints
        // nothing new.
        let first = ensure_keys(&KeyCache::default(), &master_key, &pool)
            .await
            .unwrap();
        for _ in 0..3 {
            let restarted = ensure_keys(&KeyCache::default(), &master_key, &pool)
                .await
                .unwrap();
            assert_eq!(restarted, first, "a restart moved the current key");
        }
        let client = pool.get().await.unwrap();
        for epoch in [n, n + 1] {
            let (from, until, accept) = epoch_instants(epoch).unwrap();
            let rows = client
                .query(
                    "SELECT issue_until, accept_until FROM issuer_key WHERE issue_from = $1",
                    &[&from],
                )
                .await
                .unwrap();
            assert_eq!(rows.len(), 1, "epoch {epoch} has exactly one key");
            assert_eq!(rows[0].get::<_, SystemTime>(0), until);
            assert_eq!(rows[0].get::<_, SystemTime>(1), accept);
        }

        // Stretch the current key's acceptance in the database: a gateway
        // loading it refuses it, for issuance and for spends.
        let id = hex::encode(first);
        let stretch = |sign: &'static str| {
            let pool = pool.clone();
            let id = id.clone();
            async move {
                pool.get()
                    .await
                    .unwrap()
                    .execute(
                        &format!(
                            "UPDATE issuer_key SET accept_until = accept_until {sign} \
                             interval '7 days' WHERE id = $1"
                        ),
                        &[&id],
                    )
                    .await
                    .unwrap()
            }
        };
        assert_eq!(stretch("+").await, 1);
        let cache = KeyCache::default();
        let issuing = ensure_keys(&cache, &master_key, &pool).await;
        let spending = load_key(&cache, &master_key, &pool, &first, KeyUse::Spend).await;
        assert_eq!(stretch("-").await, 1);
        assert!(
            issuing.is_err(),
            "issued under a key whose window was edited"
        );
        assert!(
            spending.is_err(),
            "accepted a spend under a key whose window was edited"
        );
        assert!(!cache.read().await.contains_key(&first));

        // Put back, it authenticates again.
        assert_eq!(
            ensure_keys(&KeyCache::default(), &master_key, &pool)
                .await
                .unwrap(),
            first
        );
    }
}
