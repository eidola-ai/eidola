//! Verify the Sigstore **DSSE / in-toto** GitHub artifact attestation that
//! Tinfoil publishes on each `confidential-model-router` release, and
//! extract the enclave measurement it commits to.
//!
//! This is the runtime analogue of the client updater's
//! `eidola-app-core::updater::ci_sigstore`, but for a *different* bundle
//! shape. The updater verifies a `cosign sign-blob` **`hashedrekord`**
//! bundle over our own `artifact-manifest.json`; here we verify a GitHub
//! **`dsse`** artifact attestation over Tinfoil's `tinfoil-deployment.json`.
//! The low-level primitives are the same and were ported from that module
//! (Fulcio chain walk + identity extraction, RFC 6962 Merkle inclusion,
//! Rekor SET) with app-core's `AppError` swapped for the local
//! [`TrustError`]. The DSSE envelope + in-toto binding is new.
//!
//! ## What a pass proves
//!
//! 1. The DSSE envelope's payload (an in-toto Statement) is signed by a
//!    Fulcio leaf whose chain walks back to a pinned Sigstore Fulcio CA.
//! 2. That leaf's OIDC issuer is GitHub Actions and its SAN identity is a
//!    tag-triggered workflow in the expected repo **at the expected tag**
//!    (we pin the tag exactly — tighter than Tinfoil's own clients, which
//!    accept any `@refs/tags/*`).
//! 3. The Rekor `dsse` entry commits to *this* envelope (payload hash +
//!    signature + signer cert) and sits in the public transparency log
//!    (SET + inclusion proof).
//! 4. The signed Statement's subject digest equals the release's
//!    `tinfoil.hash`, and its predicate is Tinfoil's
//!    `snp-tdx-multiplatform/v1` measurement.
//!
//! ## Known gaps (identical to app-core's, deferred defense-in-depth)
//!
//! - **SCT verification.** The leaf's embedded CT SCT is not checked; the
//!   OIDC-identity match + Fulcio chain walk make it defense-in-depth.
//! - **Rekor checkpoint signature.** The inclusion proof roots out to the
//!   entry's stated `rootHash`; we do not additionally verify the
//!   checkpoint signature over that root. The SET already binds the entry
//!   to a pinned Rekor key.
//!
//! Both are the same gaps `docs/gaps.md` records for the client verifier.
//! This whole module is temporary — it is deleted when we self-host
//! inference and revert to a statically pinned measurement set.

use const_oid::ObjectIdentifier;
use der::asn1::Utf8StringRef;
use der::{Decode, Encode};
use serde::Deserialize;
use sha2::{Digest, Sha256, Sha384};
use signature::hazmat::PrehashVerifier;
use spki::DecodePublicKey;
use x509_cert::Certificate;
use x509_cert::ext::pkix::SubjectAltName;
use x509_cert::ext::pkix::name::GeneralName;

// The Sigstore trusted root the resolver verifies against, embedded at build
// time from `releases/trust/sigstore-trusted-root.json` (see `build.rs`).
// Generates `pub const SIGSTORE_TRUSTED_ROOT_JSON: &str`.
include!(concat!(env!("OUT_DIR"), "/sigstore_root.gen.rs"));

/// GitHub Actions OIDC issuer — the only issuer we accept a release
/// attestation from.
pub const GITHUB_ACTIONS_OIDC_ISSUER: &str = "https://token.actions.githubusercontent.com";

/// The in-toto predicate type Tinfoil signs the enclave measurement under.
pub const TINFOIL_PREDICATE_TYPE: &str = "https://tinfoil.sh/predicate/snp-tdx-multiplatform/v1";

// ===========================================================================
// Error type
// ===========================================================================

/// A verification failure. Wraps a human-readable message; the resolver
/// logs it and keeps the current trusted set.
#[derive(Debug, Clone)]
pub struct TrustError(pub String);

impl std::fmt::Display for TrustError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TrustError {}

fn err(msg: impl Into<String>) -> TrustError {
    TrustError(msg.into())
}

type Result<T> = std::result::Result<T, TrustError>;

// ===========================================================================
// Public API
// ===========================================================================

/// The verified enclave measurement extracted from a release attestation.
/// All measurement fields are lowercase hex (SNP launch digest = 48 bytes,
/// TDX RTMRs = 48 bytes each).
#[derive(Debug, Clone)]
pub struct VerifiedMeasurement {
    pub snp_measurement: String,
    pub rtmr1: String,
    pub rtmr2: String,
    /// sha256 (hex) of `tinfoil-deployment.json` — the attestation subject.
    pub subject_digest_hex: String,
    /// The Fulcio leaf's SAN identity (the signing workflow URI).
    pub ci_identity: String,
    /// Rekor log index for the verified entry.
    pub rekor_log_index: u64,
}

/// Verify a GitHub artifact-attestation Sigstore bundle for
/// `expected_repo` at `expected_tag`, binding it to `expected_digest_hex`
/// (the release's `tinfoil.hash`), and return the measurement carried in
/// the signed in-toto predicate.
pub fn verify_release_attestation(
    bundle_bytes: &[u8],
    expected_repo: &str,
    expected_tag: &str,
    expected_digest_hex: &str,
    trust: &TrustedRoot,
) -> Result<VerifiedMeasurement> {
    let bundle: Bundle = serde_json::from_slice(bundle_bytes)
        .map_err(|e| err(format!("parsing Sigstore bundle as JSON: {e}")))?;

    if !bundle
        .media_type
        .starts_with("application/vnd.dev.sigstore.bundle")
    {
        return Err(err(format!(
            "attestation bundle has unexpected mediaType `{}` (expected `application/vnd.dev.sigstore.bundle.*`)",
            bundle.media_type
        )));
    }

    let envelope = bundle.dsse_envelope.as_ref().ok_or_else(|| {
        err("bundle has no `dsseEnvelope` — GitHub artifact attestations are DSSE")
    })?;

    // ── Fulcio leaf cert (single `certificate`, or first of a chain) ──────
    let vm = &bundle.verification_material;
    let cert_b64 = vm
        .certificate
        .as_ref()
        .map(|c| c.raw_bytes.as_str())
        .or_else(|| {
            vm.x509_certificate_chain
                .as_ref()
                .and_then(|c| c.certificates.first())
                .map(|c| c.raw_bytes.as_str())
        })
        .ok_or_else(|| err("bundle verificationMaterial has no Fulcio leaf certificate"))?;
    let cert_der = b64_std(cert_b64, "certificate.rawBytes")?;

    // ── Chain walk + identity extraction (ported from app-core cert.rs) ───
    let leaf = cert::verify_chain_and_extract(&cert_der, &trust.fulcio_cas)?;

    // ── Identity: GitHub Actions OIDC + repo + exact tag ──────────────────
    if leaf.oidc_issuer != GITHUB_ACTIONS_OIDC_ISSUER {
        return Err(err(format!(
            "leaf cert OIDC issuer `{}` ≠ expected `{GITHUB_ACTIONS_OIDC_ISSUER}`",
            leaf.oidc_issuer
        )));
    }
    // Pin repo AND tag. Tinfoil's own clients accept any `@refs/tags/*`;
    // binding to the tag we resolved from `releases/latest` blocks an
    // authentic-but-different-tag attestation being served for this digest.
    let expected_identity =
        format!("https://github.com/{expected_repo}/.github/workflows/*@refs/tags/{expected_tag}");
    if !cert::glob_matches(&expected_identity, &leaf.san_uri) {
        return Err(err(format!(
            "leaf cert SAN URI `{}` does not match expected identity `{expected_identity}`",
            leaf.san_uri
        )));
    }

    // ── DSSE signature over PAE(payloadType, payload) ─────────────────────
    let payload = b64_std(&envelope.payload, "dsseEnvelope.payload")?;
    let dsse_sig = envelope
        .signatures
        .first()
        .ok_or_else(|| err("dsseEnvelope has no signatures"))?;
    let dsse_sig_bytes = b64_std(&dsse_sig.sig, "dsseEnvelope.signatures[0].sig")?;
    let pae = dsse_pae(&envelope.payload_type, &payload);
    let pae_digest: [u8; 32] = Sha256::digest(&pae).into();
    cert::verify_ecdsa_prehash(
        &leaf.spki_der,
        leaf.leaf_key_alg,
        &pae_digest,
        &dsse_sig_bytes,
    )
    .map_err(|e| err(format!("DSSE envelope signature verification failed: {e}")))?;

    // ── in-toto Statement: type, predicate, subject digest ────────────────
    let statement: InTotoStatement = serde_json::from_slice(&payload)
        .map_err(|e| err(format!("parsing DSSE payload as in-toto Statement: {e}")))?;
    if statement.type_ != "https://in-toto.io/Statement/v1" {
        return Err(err(format!(
            "in-toto Statement _type `{}` is not `https://in-toto.io/Statement/v1`",
            statement.type_
        )));
    }
    if statement.predicate_type != TINFOIL_PREDICATE_TYPE {
        return Err(err(format!(
            "in-toto predicateType `{}` ≠ expected `{TINFOIL_PREDICATE_TYPE}`",
            statement.predicate_type
        )));
    }
    let subject = statement
        .subject
        .first()
        .ok_or_else(|| err("in-toto Statement has no subject"))?;
    let subject_digest = subject.digest.sha256.to_ascii_lowercase();
    let expected_digest = expected_digest_hex.trim().to_ascii_lowercase();
    if subject_digest != expected_digest {
        return Err(err(format!(
            "in-toto subject digest `{subject_digest}` ≠ release tinfoil.hash `{expected_digest}` \
             — attestation is for a different artifact"
        )));
    }

    // ── Rekor entry: exactly one, kind `dsse`, bound to this envelope ─────
    if vm.tlog_entries.len() != 1 {
        return Err(err(format!(
            "bundle has {} tlog entries; a GitHub artifact attestation has exactly 1",
            vm.tlog_entries.len()
        )));
    }
    let entry = &vm.tlog_entries[0];
    let log_index = entry
        .log_index
        .parse::<u64>()
        .map_err(|e| err(format!("tlog entry logIndex `{}`: {e}", entry.log_index)))?;
    let canonical_body = b64_std(&entry.canonicalized_body, "canonicalizedBody")?;

    rekor::verify_dsse_body_binding(&canonical_body, &payload, &dsse_sig_bytes, &cert_der)?;

    // SET + inclusion proof (ported from app-core rekor_verify.rs/merkle.rs).
    let integrated_time = entry.integrated_time.parse::<i64>().map_err(|e| {
        err(format!(
            "tlog integratedTime `{}`: {e}",
            entry.integrated_time
        ))
    })?;
    let log_id = b64_std_array::<32>(&entry.log_id.key_id, "tlogEntry.logId.keyId")?;
    let promise = entry
        .inclusion_promise
        .as_ref()
        .ok_or_else(|| err("tlog entry missing inclusionPromise (SET)"))?;
    let set_bytes = b64_std(&promise.signed_entry_timestamp, "signedEntryTimestamp")?;
    let proof = entry
        .inclusion_proof
        .as_ref()
        .ok_or_else(|| err("tlog entry missing inclusionProof"))?;
    let root_hash = b64_std_array::<32>(&proof.root_hash, "inclusionProof.rootHash")?;
    let tree_size = proof.tree_size.parse::<u64>().map_err(|e| {
        err(format!(
            "inclusionProof.treeSize `{}`: {e}",
            proof.tree_size
        ))
    })?;
    let proof_leaf_index = proof.log_index.parse::<u64>().map_err(|e| {
        err(format!(
            "inclusionProof.logIndex `{}`: {e}",
            proof.log_index
        ))
    })?;
    let proof_hashes = proof
        .hashes
        .iter()
        .map(|h| b64_std_array::<32>(h, "inclusionProof.hashes[]"))
        .collect::<Result<Vec<_>>>()?;

    rekor::verify_set_and_inclusion(
        &canonical_body,
        &entry.canonicalized_body,
        &set_bytes,
        integrated_time,
        log_index,
        &log_id,
        &root_hash,
        &proof_hashes,
        tree_size,
        proof_leaf_index,
        &trust.rekor_keys,
    )?;

    // Rekor's integratedTime must fall within the leaf cert validity window.
    let it_u64 = u64::try_from(integrated_time)
        .map_err(|_| err(format!("integratedTime `{integrated_time}` is negative")))?;
    if it_u64 < leaf.not_before || it_u64 > leaf.not_after {
        return Err(err(format!(
            "Rekor integratedTime {it_u64} is outside leaf cert validity [{}, {}]",
            leaf.not_before, leaf.not_after
        )));
    }

    // ── Extract + validate the measurement from the signed predicate ──────
    let snp = normalize_measurement(&statement.predicate.snp_measurement, "snp_measurement")?;
    let rtmr1 = normalize_measurement(&statement.predicate.tdx_measurement.rtmr1, "rtmr1")?;
    let rtmr2 = normalize_measurement(&statement.predicate.tdx_measurement.rtmr2, "rtmr2")?;

    Ok(VerifiedMeasurement {
        snp_measurement: snp,
        rtmr1,
        rtmr2,
        subject_digest_hex: subject_digest,
        ci_identity: leaf.san_uri,
        rekor_log_index: log_index,
    })
}

/// DSSE Pre-Authentication Encoding (PAEv1):
/// `"DSSEv1" SP LEN(type) SP type SP LEN(payload) SP payload`.
fn dsse_pae(payload_type: &str, payload: &[u8]) -> Vec<u8> {
    let mut pae = Vec::with_capacity(payload.len() + payload_type.len() + 32);
    pae.extend_from_slice(b"DSSEv1 ");
    pae.extend_from_slice(payload_type.len().to_string().as_bytes());
    pae.push(b' ');
    pae.extend_from_slice(payload_type.as_bytes());
    pae.push(b' ');
    pae.extend_from_slice(payload.len().to_string().as_bytes());
    pae.push(b' ');
    pae.extend_from_slice(payload);
    pae
}

fn normalize_measurement(value: &str, field: &str) -> Result<String> {
    let v = value.trim().to_ascii_lowercase();
    if v.len() != 96 || !v.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(err(format!(
            "predicate `{field}` must be 96 lowercase hex chars (48 bytes), got {} chars",
            v.len()
        )));
    }
    Ok(v)
}

// ===========================================================================
// Bundle / in-toto / rekor-body JSON shapes
// ===========================================================================

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Bundle {
    media_type: String,
    verification_material: VerificationMaterial,
    #[serde(default)]
    dsse_envelope: Option<DsseEnvelope>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VerificationMaterial {
    #[serde(default)]
    certificate: Option<RawCert>,
    #[serde(default)]
    x509_certificate_chain: Option<CertChain>,
    tlog_entries: Vec<TlogEntry>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawCert {
    raw_bytes: String,
}

#[derive(Deserialize)]
struct CertChain {
    certificates: Vec<RawCert>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DsseEnvelope {
    payload: String,
    payload_type: String,
    signatures: Vec<DsseSig>,
}

#[derive(Deserialize)]
struct DsseSig {
    sig: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TlogEntry {
    log_index: String,
    log_id: LogId,
    integrated_time: String,
    #[serde(default)]
    inclusion_promise: Option<InclusionPromise>,
    #[serde(default)]
    inclusion_proof: Option<InclusionProof>,
    canonicalized_body: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LogId {
    key_id: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct InclusionPromise {
    signed_entry_timestamp: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct InclusionProof {
    log_index: String,
    root_hash: String,
    tree_size: String,
    #[serde(default)]
    hashes: Vec<String>,
}

#[derive(Deserialize)]
struct InTotoStatement {
    #[serde(rename = "_type")]
    type_: String,
    #[serde(rename = "predicateType")]
    predicate_type: String,
    subject: Vec<InTotoSubject>,
    predicate: TinfoilPredicate,
}

#[derive(Deserialize)]
struct InTotoSubject {
    digest: SubjectDigest,
}

#[derive(Deserialize)]
struct SubjectDigest {
    sha256: String,
}

#[derive(Deserialize)]
struct TinfoilPredicate {
    snp_measurement: String,
    tdx_measurement: TdxPredicate,
}

#[derive(Deserialize)]
struct TdxPredicate {
    rtmr1: String,
    rtmr2: String,
}

// ===========================================================================
// Trusted root (ported from app-core updater/trust.rs, ctlogs dropped)
// ===========================================================================

/// Parsed subset of `sigstore-trusted-root.json` our verifier consumes.
#[derive(Debug, Clone)]
pub struct TrustedRoot {
    pub fulcio_cas: Vec<FulcioCa>,
    pub rekor_keys: Vec<RekorKey>,
}

#[derive(Debug, Clone)]
pub struct FulcioCa {
    pub cert_chain_der: Vec<Vec<u8>>,
    pub valid_from: i64,
    pub valid_until: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct RekorKey {
    pub log_id: [u8; 32],
    pub spki_der: Vec<u8>,
    pub key_details: KeyDetails,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyDetails {
    EcdsaP256Sha256,
    EcdsaP384Sha384,
    Ed25519,
}

impl KeyDetails {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "PKIX_ECDSA_P256_SHA_256" => Ok(KeyDetails::EcdsaP256Sha256),
            "PKIX_ECDSA_P384_SHA_384" => Ok(KeyDetails::EcdsaP384Sha384),
            "PKIX_ED25519" => Ok(KeyDetails::Ed25519),
            other => Err(err(format!(
                "unsupported KeyDetails `{other}` in sigstore trusted root"
            ))),
        }
    }
}

/// Load and parse the embedded [`SIGSTORE_TRUSTED_ROOT_JSON`].
pub fn load_trusted_root() -> Result<TrustedRoot> {
    load_trusted_root_from_str(SIGSTORE_TRUSTED_ROOT_JSON)
}

/// Like [`load_trusted_root`] but with explicit JSON — used by tests.
pub fn load_trusted_root_from_str(json: &str) -> Result<TrustedRoot> {
    let parsed: RawTrustedRoot = serde_json::from_str(json)
        .map_err(|e| err(format!("parsing sigstore-trusted-root.json: {e}")))?;

    let fulcio_cas = parsed
        .certificate_authorities
        .into_iter()
        .map(parse_fulcio_ca)
        .collect::<Result<Vec<_>>>()?;
    let rekor_keys = parsed
        .tlogs
        .into_iter()
        .map(parse_rekor_key)
        .collect::<Result<Vec<_>>>()?;

    if fulcio_cas.is_empty() {
        return Err(err("sigstore-trusted-root.json has no Fulcio CAs"));
    }
    if rekor_keys.is_empty() {
        return Err(err("sigstore-trusted-root.json has no Rekor keys"));
    }
    Ok(TrustedRoot {
        fulcio_cas,
        rekor_keys,
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTrustedRoot {
    #[serde(default)]
    tlogs: Vec<RawTlog>,
    #[serde(default)]
    certificate_authorities: Vec<RawCa>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawCa {
    cert_chain: RawCertChain,
    valid_for: RawValidFor,
}

#[derive(Deserialize)]
struct RawCertChain {
    certificates: Vec<RawCertEntry>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawCertEntry {
    raw_bytes: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTlog {
    public_key: RawPublicKey,
    log_id: RawLogId,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawPublicKey {
    raw_bytes: String,
    key_details: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawLogId {
    key_id: String,
}

#[derive(Deserialize)]
struct RawValidFor {
    start: String,
    #[serde(default)]
    end: Option<String>,
}

fn parse_fulcio_ca(raw: RawCa) -> Result<FulcioCa> {
    let chain = raw
        .cert_chain
        .certificates
        .into_iter()
        .map(|c| b64_std(&c.raw_bytes, "certificate raw_bytes"))
        .collect::<Result<Vec<_>>>()?;
    let valid_from = parse_rfc3339_unix(&raw.valid_for.start)?;
    let valid_until = raw
        .valid_for
        .end
        .as_deref()
        .map(parse_rfc3339_unix)
        .transpose()?;
    Ok(FulcioCa {
        cert_chain_der: chain,
        valid_from,
        valid_until,
    })
}

fn parse_rekor_key(raw: RawTlog) -> Result<RekorKey> {
    let spki_der = b64_std(&raw.public_key.raw_bytes, "tlog public_key raw_bytes")?;
    let log_id = b64_std_array::<32>(&raw.log_id.key_id, "tlog log_id keyId")?;
    let key_details = KeyDetails::parse(&raw.public_key.key_details)?;
    Ok(RekorKey {
        log_id,
        spki_der,
        key_details,
    })
}

/// Parse `YYYY-MM-DDTHH:MM:SS[.fff]Z` to Unix seconds (UTC). Ported from
/// app-core's `trust.rs` (Howard Hinnant days-from-civil) to avoid a
/// date-time crate.
fn parse_rfc3339_unix(s: &str) -> Result<i64> {
    let s = s.trim_end_matches('Z');
    let (date, time) = s
        .split_once('T')
        .ok_or_else(|| err(format!("malformed RFC3339 timestamp `{s}`")))?;
    let mut d = date.split('-');
    let year: i64 = d
        .next()
        .and_then(|x| x.parse().ok())
        .ok_or_else(|| err(format!("bad year in `{s}`")))?;
    let month: i64 = d
        .next()
        .and_then(|x| x.parse().ok())
        .ok_or_else(|| err(format!("bad month in `{s}`")))?;
    let day: i64 = d
        .next()
        .and_then(|x| x.parse().ok())
        .ok_or_else(|| err(format!("bad day in `{s}`")))?;
    let time = time.split('.').next().unwrap_or(time);
    let mut t = time.split(':');
    let hour: i64 = t
        .next()
        .and_then(|x| x.parse().ok())
        .ok_or_else(|| err(format!("bad hour in `{s}`")))?;
    let minute: i64 = t
        .next()
        .and_then(|x| x.parse().ok())
        .ok_or_else(|| err(format!("bad minute in `{s}`")))?;
    let second: i64 = t
        .next()
        .and_then(|x| x.parse().ok())
        .ok_or_else(|| err(format!("bad second in `{s}`")))?;
    let (y, m) = if month <= 2 {
        (year - 1, month + 12)
    } else {
        (year, month)
    };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let doy = (153 * (m - 3) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    Ok(days * 86400 + hour * 3600 + minute * 60 + second)
}

// ===========================================================================
// Fulcio cert chain + identity (ported from app-core ci_sigstore/cert.rs)
// ===========================================================================

mod cert {
    use super::*;

    mod oid {
        use const_oid::ObjectIdentifier;
        pub const EC_PUBLIC_KEY: ObjectIdentifier =
            ObjectIdentifier::new_unwrap("1.2.840.10045.2.1");
        pub const NIST_P256: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.2.840.10045.3.1.7");
        pub const NIST_P384: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.3.132.0.34");
        pub const ECDSA_WITH_SHA256: ObjectIdentifier =
            ObjectIdentifier::new_unwrap("1.2.840.10045.4.3.2");
        pub const ECDSA_WITH_SHA384: ObjectIdentifier =
            ObjectIdentifier::new_unwrap("1.2.840.10045.4.3.3");
        pub const FULCIO_OIDC_ISSUER_V1: ObjectIdentifier =
            ObjectIdentifier::new_unwrap("1.3.6.1.4.1.57264.1.1");
        pub const FULCIO_OIDC_ISSUER_V2: ObjectIdentifier =
            ObjectIdentifier::new_unwrap("1.3.6.1.4.1.57264.1.8");
    }

    #[derive(Debug, Clone)]
    pub struct LeafCertInfo {
        pub spki_der: Vec<u8>,
        pub leaf_key_alg: LeafKeyAlg,
        pub san_uri: String,
        pub oidc_issuer: String,
        pub not_before: u64,
        pub not_after: u64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum LeafKeyAlg {
        EcdsaP256,
        EcdsaP384,
    }

    pub fn verify_chain_and_extract(
        leaf_der: &[u8],
        fulcio_cas: &[FulcioCa],
    ) -> Result<LeafCertInfo> {
        let leaf = Certificate::from_der(leaf_der)
            .map_err(|e| err(format!("parsing Fulcio leaf cert: {e}")))?;

        let not_before = time_to_unix(leaf.tbs_certificate.validity.not_before);
        let not_after = time_to_unix(leaf.tbs_certificate.validity.not_after);

        let ca = fulcio_cas
            .iter()
            .find(|ca| {
                let after_start = (not_before as i64) >= ca.valid_from;
                let before_end = ca.valid_until.is_none_or(|end| (not_before as i64) <= end);
                after_start && before_end
            })
            .ok_or_else(|| {
                err(format!(
                    "no Fulcio CA in the trusted root covers leaf cert notBefore={not_before}; \
                     either the trusted root is stale or the cert is forged"
                ))
            })?;

        if ca.cert_chain_der.is_empty() {
            return Err(err(
                "Fulcio CA has empty cert_chain — cannot verify the leaf",
            ));
        }

        let mut current = leaf.clone();
        for parent_der in &ca.cert_chain_der {
            let parent = Certificate::from_der(parent_der)
                .map_err(|e| err(format!("parsing Fulcio chain cert: {e}")))?;
            verify_cert_signature(&current, &parent)?;
            current = parent;
        }
        verify_cert_signature(&current, &current)?;

        let spki_der = leaf
            .tbs_certificate
            .subject_public_key_info
            .to_der()
            .map_err(|e| err(format!("re-encoding leaf SPKI: {e}")))?;
        let leaf_key_alg = parse_leaf_key_alg(&leaf)?;
        let san_uri = extract_san_uri(&leaf)?;
        let oidc_issuer = extract_oidc_issuer(&leaf)?;

        Ok(LeafCertInfo {
            spki_der,
            leaf_key_alg,
            san_uri,
            oidc_issuer,
            not_before,
            not_after,
        })
    }

    fn parse_leaf_key_alg(cert: &Certificate) -> Result<LeafKeyAlg> {
        let spki = &cert.tbs_certificate.subject_public_key_info;
        if spki.algorithm.oid != oid::EC_PUBLIC_KEY {
            return Err(err(format!(
                "leaf cert pubkey algorithm `{}` is not EC",
                spki.algorithm.oid
            )));
        }
        let curve_oid: ObjectIdentifier = spki
            .algorithm
            .parameters
            .as_ref()
            .and_then(|p| p.decode_as().ok())
            .ok_or_else(|| err("leaf cert SPKI missing EC named-curve parameter"))?;
        if curve_oid == oid::NIST_P256 {
            Ok(LeafKeyAlg::EcdsaP256)
        } else if curve_oid == oid::NIST_P384 {
            Ok(LeafKeyAlg::EcdsaP384)
        } else {
            Err(err(format!(
                "leaf cert uses unsupported EC curve `{curve_oid}`"
            )))
        }
    }

    fn verify_cert_signature(cert: &Certificate, signer: &Certificate) -> Result<()> {
        let tbs_der = cert
            .tbs_certificate
            .to_der()
            .map_err(|e| err(format!("re-encoding TBSCertificate: {e}")))?;
        let sig_bytes = cert
            .signature
            .as_bytes()
            .ok_or_else(|| err("cert signature BitString is not byte-aligned"))?;
        let signer_spki = signer
            .tbs_certificate
            .subject_public_key_info
            .to_der()
            .map_err(|e| err(format!("re-encoding signer SPKI: {e}")))?;

        let alg_oid = &cert.signature_algorithm.oid;
        if *alg_oid == oid::ECDSA_WITH_SHA256 {
            verify_p256(&signer_spki, &Sha256::digest(&tbs_der), sig_bytes)
        } else if *alg_oid == oid::ECDSA_WITH_SHA384 {
            verify_p384(&signer_spki, &Sha384::digest(&tbs_der), sig_bytes)
        } else {
            Err(err(format!(
                "cert is signed with unsupported algorithm `{alg_oid}` \
                 (only ecdsa-with-SHA256 / ecdsa-with-SHA384)"
            )))
        }
    }

    /// Verify an ECDSA signature over an already-computed prehash, using the
    /// leaf's key algorithm. Used both for the DSSE PAE signature (prehash =
    /// sha256(PAE)) and internally for cert-chain links.
    pub fn verify_ecdsa_prehash(
        spki_der: &[u8],
        alg: LeafKeyAlg,
        prehash: &[u8],
        sig_der: &[u8],
    ) -> Result<()> {
        match alg {
            LeafKeyAlg::EcdsaP256 => verify_p256(spki_der, prehash, sig_der),
            LeafKeyAlg::EcdsaP384 => verify_p384(spki_der, prehash, sig_der),
        }
    }

    fn verify_p256(spki_der: &[u8], prehash: &[u8], sig_der: &[u8]) -> Result<()> {
        let key = p256::ecdsa::VerifyingKey::from_public_key_der(spki_der)
            .map_err(|e| err(format!("parsing P-256 pubkey: {e}")))?;
        let sig = p256::ecdsa::Signature::from_der(sig_der)
            .map_err(|e| err(format!("parsing P-256 signature DER: {e}")))?;
        key.verify_prehash(prehash, &sig)
            .map_err(|e| err(format!("ECDSA P-256 verification failed: {e}")))
    }

    fn verify_p384(spki_der: &[u8], prehash: &[u8], sig_der: &[u8]) -> Result<()> {
        let key = p384::ecdsa::VerifyingKey::from_public_key_der(spki_der)
            .map_err(|e| err(format!("parsing P-384 pubkey: {e}")))?;
        let sig = p384::ecdsa::Signature::from_der(sig_der)
            .map_err(|e| err(format!("parsing P-384 signature DER: {e}")))?;
        key.verify_prehash(prehash, &sig)
            .map_err(|e| err(format!("ECDSA P-384 verification failed: {e}")))
    }

    fn extract_san_uri(cert: &Certificate) -> Result<String> {
        let extensions = cert
            .tbs_certificate
            .extensions
            .as_ref()
            .ok_or_else(|| err("leaf cert has no extensions (missing SAN)"))?;
        for ext in extensions {
            if ext.extn_id == const_oid::db::rfc5280::ID_CE_SUBJECT_ALT_NAME {
                let san = SubjectAltName::from_der(ext.extn_value.as_bytes())
                    .map_err(|e| err(format!("parsing SAN extension: {e}")))?;
                for entry in san.0 {
                    if let GeneralName::UniformResourceIdentifier(uri) = entry {
                        return Ok(uri.to_string());
                    }
                }
                return Err(err("leaf cert SAN has no URI entry"));
            }
        }
        Err(err("leaf cert has no SubjectAltName extension"))
    }

    fn extract_oidc_issuer(cert: &Certificate) -> Result<String> {
        let extensions = cert
            .tbs_certificate
            .extensions
            .as_ref()
            .ok_or_else(|| err("leaf cert has no extensions (missing OIDC issuer)"))?;
        let mut v1: Option<String> = None;
        let mut v2: Option<String> = None;
        for ext in extensions {
            if ext.extn_id == oid::FULCIO_OIDC_ISSUER_V2 {
                let utf8 = Utf8StringRef::from_der(ext.extn_value.as_bytes())
                    .map_err(|e| err(format!("parsing Fulcio OIDC issuer v2 extension: {e}")))?;
                v2 = Some(utf8.as_str().to_string());
            } else if ext.extn_id == oid::FULCIO_OIDC_ISSUER_V1 {
                let s = std::str::from_utf8(ext.extn_value.as_bytes()).map_err(|e| {
                    err(format!("Fulcio OIDC issuer v1 extension is not UTF-8: {e}"))
                })?;
                v1 = Some(s.to_string());
            }
        }
        v2.or(v1)
            .ok_or_else(|| err("leaf cert has no Fulcio OIDC issuer extension"))
    }

    fn time_to_unix(t: x509_cert::time::Time) -> u64 {
        t.to_unix_duration().as_secs()
    }

    /// Glob-match `pattern` (literal `*` wildcards) against `s`, anchored on
    /// both ends. Used for the SAN identity check.
    pub fn glob_matches(pattern: &str, s: &str) -> bool {
        let mut regex_str = String::from("^");
        for ch in pattern.chars() {
            match ch {
                '*' => regex_str.push_str(".*"),
                '.' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\' | '^' | '$' | '?' => {
                    regex_str.push('\\');
                    regex_str.push(ch);
                }
                other => regex_str.push(other),
            }
        }
        regex_str.push('$');
        match regex::Regex::new(&regex_str) {
            Ok(re) => re.is_match(s),
            Err(_) => false,
        }
    }
}

// ===========================================================================
// Rekor: dsse body binding + SET + Merkle inclusion
// (ported from app-core ci_sigstore/rekor.rs + rekor_verify.rs + merkle.rs)
// ===========================================================================

mod rekor {
    use super::*;

    #[derive(Deserialize)]
    struct DsseBody {
        #[serde(rename = "apiVersion")]
        api_version: String,
        kind: String,
        spec: DsseSpec,
    }

    #[derive(Deserialize)]
    struct DsseSpec {
        #[serde(rename = "payloadHash")]
        payload_hash: HashObj,
        signatures: Vec<DsseBodySig>,
    }

    #[derive(Deserialize)]
    struct HashObj {
        algorithm: String,
        value: String,
    }

    #[derive(Deserialize)]
    struct DsseBodySig {
        /// Base64 of the DSSE signature bytes.
        signature: String,
        /// Base64 of the PEM-encoded signer certificate.
        verifier: String,
    }

    /// Confirm the Rekor `dsse` entry body commits to *our* envelope: the
    /// payload hash, the signature bytes, and the signer cert.
    pub fn verify_dsse_body_binding(
        canonical_body: &[u8],
        payload: &[u8],
        envelope_sig_bytes: &[u8],
        leaf_cert_der: &[u8],
    ) -> Result<()> {
        let body: DsseBody = serde_json::from_slice(canonical_body)
            .map_err(|e| err(format!("parsing rekor canonicalizedBody as dsse: {e}")))?;
        if body.kind != "dsse" || body.api_version != "0.0.1" {
            return Err(err(format!(
                "tlog entry kind/apiVersion (`{}` / `{}`) is not dsse 0.0.1",
                body.kind, body.api_version
            )));
        }
        if body.spec.payload_hash.algorithm != "sha256" {
            return Err(err(format!(
                "dsse payloadHash algorithm `{}` is not sha256",
                body.spec.payload_hash.algorithm
            )));
        }
        let payload_hash = hex_decode(&body.spec.payload_hash.value)?;
        if payload_hash.as_slice() != Sha256::digest(payload).as_slice() {
            return Err(err(
                "rekor dsse payloadHash ≠ sha256(envelope payload) — entry is about a different payload",
            ));
        }

        let body_sig = body
            .spec
            .signatures
            .first()
            .ok_or_else(|| err("rekor dsse body has no signatures"))?;
        let body_sig_bytes = b64_std(&body_sig.signature, "rekor dsse signature")?;
        if body_sig_bytes != envelope_sig_bytes {
            return Err(err(
                "rekor dsse signature ≠ the envelope's signature — entry is about a different signing",
            ));
        }

        let verifier_pem_bytes = b64_std(&body_sig.verifier, "rekor dsse verifier")?;
        let verifier_pem = std::str::from_utf8(&verifier_pem_bytes)
            .map_err(|e| err(format!("rekor dsse verifier is not UTF-8 PEM: {e}")))?;
        let verifier_der = pem_to_der(verifier_pem)?;
        if verifier_der != leaf_cert_der {
            return Err(err(
                "rekor dsse verifier cert ≠ the bundle's leaf cert — entry references a different signer",
            ));
        }
        Ok(())
    }

    /// Verify Rekor's SET signature over the canonical
    /// `{body, integratedTime, logID, logIndex}` payload, then verify the
    /// inclusion proof of `canonical_body`'s leaf hash up to `root_hash`.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_set_and_inclusion(
        canonical_body: &[u8],
        canonical_body_b64: &str,
        set_bytes: &[u8],
        integrated_time: i64,
        log_index: u64,
        log_id: &[u8; 32],
        root_hash: &[u8; 32],
        proof_hashes: &[[u8; 32]],
        tree_size: u64,
        proof_leaf_index: u64,
        rekor_keys: &[RekorKey],
    ) -> Result<()> {
        let key = rekor_keys
            .iter()
            .find(|k| k.log_id == *log_id)
            .ok_or_else(|| {
                err(format!(
                    "no pinned Rekor key matches the bundle's logId `{}`",
                    hex_encode(log_id)
                ))
            })?;
        // Keys ordered lexicographically by ASCII codepoint: body <
        // integratedTime < logID < logIndex. Must match what Rekor signed.
        let signed_payload = format!(
            r#"{{"body":"{body}","integratedTime":{it},"logID":"{lid}","logIndex":{li}}}"#,
            body = canonical_body_b64,
            it = integrated_time,
            lid = hex_encode(log_id),
            li = log_index,
        );
        verify_set_signature(key, signed_payload.as_bytes(), set_bytes)?;

        let leaf_hash = hash_leaf(canonical_body);
        verify_inclusion_proof(
            proof_leaf_index,
            &leaf_hash,
            tree_size,
            proof_hashes,
            root_hash,
        )
    }

    fn verify_set_signature(key: &RekorKey, message: &[u8], signature: &[u8]) -> Result<()> {
        match key.key_details {
            KeyDetails::EcdsaP256Sha256 => {
                let vk = p256::ecdsa::VerifyingKey::from_public_key_der(&key.spki_der)
                    .map_err(|e| err(format!("parsing pinned Rekor P-256 pubkey: {e}")))?;
                let sig = p256::ecdsa::Signature::from_der(signature)
                    .map_err(|e| err(format!("parsing Rekor SET signature DER (P-256): {e}")))?;
                let prehash = Sha256::digest(message);
                vk.verify_prehash(&prehash, &sig)
                    .map_err(|e| err(format!("Rekor SET signature failed P-256 verify: {e}")))
            }
            KeyDetails::EcdsaP384Sha384 | KeyDetails::Ed25519 => Err(err(
                "the Rekor log matching this entry uses a non-P256 key, which this verifier does \
                 not implement (GitHub's public Rekor is ECDSA-P256)",
            )),
        }
    }

    // ── RFC 6962 Merkle inclusion (ported from app-core merkle.rs) ────────

    fn hash_leaf(leaf: &[u8]) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update([0x00]);
        h.update(leaf);
        h.finalize().into()
    }

    fn hash_children(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update([0x01]);
        h.update(left);
        h.update(right);
        h.finalize().into()
    }

    fn verify_inclusion_proof(
        index: u64,
        leaf_hash: &[u8; 32],
        tree_size: u64,
        proof_hashes: &[[u8; 32]],
        root_hash: &[u8; 32],
    ) -> Result<()> {
        let computed = compute_root_from_proof(index, leaf_hash, tree_size, proof_hashes)?;
        if computed != *root_hash {
            return Err(err(format!(
                "Merkle inclusion proof's recomputed root `{}` ≠ declared rootHash `{}`",
                hex_encode(&computed),
                hex_encode(root_hash)
            )));
        }
        Ok(())
    }

    fn compute_root_from_proof(
        index: u64,
        leaf_hash: &[u8; 32],
        tree_size: u64,
        proof_hashes: &[[u8; 32]],
    ) -> Result<[u8; 32]> {
        if index >= tree_size {
            return Err(err(format!(
                "Merkle inclusion proof: leaf index {index} >= tree size {tree_size}"
            )));
        }
        let (inner, border) = decomp_inclusion_proof(index, tree_size);
        let expected_len = inner + border;
        if proof_hashes.len() as u64 != expected_len {
            return Err(err(format!(
                "Merkle inclusion proof has {} hashes, expected {expected_len}",
                proof_hashes.len()
            )));
        }
        let after_inner = chain_inner(*leaf_hash, &proof_hashes[..inner as usize], index);
        Ok(chain_border_right(
            after_inner,
            &proof_hashes[inner as usize..],
        ))
    }

    fn chain_inner(mut seed: [u8; 32], proof_hashes: &[[u8; 32]], index: u64) -> [u8; 32] {
        for (i, h) in proof_hashes.iter().enumerate() {
            seed = if ((index >> i) & 1) == 0 {
                hash_children(&seed, h)
            } else {
                hash_children(h, &seed)
            };
        }
        seed
    }

    fn chain_border_right(mut seed: [u8; 32], proof_hashes: &[[u8; 32]]) -> [u8; 32] {
        for h in proof_hashes {
            seed = hash_children(h, &seed);
        }
        seed
    }

    fn decomp_inclusion_proof(index: u64, tree_size: u64) -> (u64, u64) {
        let inner = inner_proof_size(index, tree_size);
        let border = (index >> inner).count_ones() as u64;
        (inner, border)
    }

    fn inner_proof_size(index: u64, tree_size: u64) -> u64 {
        u64::BITS as u64 - ((index ^ (tree_size - 1)).leading_zeros() as u64)
    }

    fn pem_to_der(pem: &str) -> Result<Vec<u8>> {
        let trimmed = pem.trim();
        let body = trimmed
            .strip_prefix("-----BEGIN CERTIFICATE-----")
            .and_then(|s| s.strip_suffix("-----END CERTIFICATE-----"))
            .ok_or_else(|| err("expected CERTIFICATE PEM markers on rekor verifier"))?
            .trim();
        let stripped: String = body.chars().filter(|c| !c.is_whitespace()).collect();
        use base64::Engine;
        base64::engine::general_purpose::STANDARD
            .decode(stripped.as_bytes())
            .map_err(|e| err(format!("base64-decoding PEM cert body: {e}")))
    }

    fn hex_decode(s: &str) -> Result<Vec<u8>> {
        if !s.len().is_multiple_of(2) {
            return Err(err(format!("hex string `{s}` has odd length")));
        }
        (0..s.len())
            .step_by(2)
            .map(|i| {
                u8::from_str_radix(&s[i..i + 2], 16)
                    .map_err(|e| err(format!("hex-decoding rekor hash: {e}")))
            })
            .collect()
    }
}

// ===========================================================================
// Small shared encoders
// ===========================================================================

fn b64_std(s: &str, field: &str) -> Result<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(s.as_bytes())
        .map_err(|e| err(format!("base64-decoding `{field}`: {e}")))
}

fn b64_std_array<const N: usize>(s: &str, field: &str) -> Result<[u8; N]> {
    let bytes = b64_std(s, field)?;
    bytes
        .as_slice()
        .try_into()
        .map_err(|_| err(format!("`{field}` is {} bytes, expected {N}", bytes.len())))
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(out, "{b:02x}").unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_pinned_trusted_root() {
        let trust = load_trusted_root().expect("pinned trusted root must parse");
        assert!(!trust.fulcio_cas.is_empty(), "expected ≥1 Fulcio CA");
        assert!(!trust.rekor_keys.is_empty(), "expected ≥1 Rekor key");
    }

    #[test]
    fn dsse_pae_matches_spec() {
        // "DSSEv1 SP LEN SP TYPE SP LEN SP PAYLOAD"
        let pae = dsse_pae("application/vnd.in-toto+json", b"{}");
        assert_eq!(pae, b"DSSEv1 28 application/vnd.in-toto+json 2 {}");
    }

    #[test]
    fn glob_pins_repo_and_tag() {
        let pat = "https://github.com/tinfoilsh/confidential-model-router/.github/workflows/*@refs/tags/v0.0.115";
        assert!(cert::glob_matches(
            pat,
            "https://github.com/tinfoilsh/confidential-model-router/.github/workflows/release.yml@refs/tags/v0.0.115"
        ));
        // Different tag rejected (the tightening over Tinfoil's own clients).
        assert!(!cert::glob_matches(
            pat,
            "https://github.com/tinfoilsh/confidential-model-router/.github/workflows/release.yml@refs/tags/v0.0.114"
        ));
        // Different repo rejected.
        assert!(!cert::glob_matches(
            pat,
            "https://github.com/attacker/confidential-model-router/.github/workflows/release.yml@refs/tags/v0.0.115"
        ));
    }

    #[test]
    fn normalize_measurement_validates_hex_len() {
        assert!(normalize_measurement(&"ab".repeat(48), "snp").is_ok());
        assert!(normalize_measurement("deadbeef", "snp").is_err());
        assert!(normalize_measurement(&"zz".repeat(48), "snp").is_err());
    }

    #[test]
    fn parse_rfc3339_unix_basic() {
        assert_eq!(parse_rfc3339_unix("1970-01-01T00:00:00Z").unwrap(), 0);
        assert_eq!(
            parse_rfc3339_unix("2021-01-12T11:53:27Z").unwrap(),
            1_610_452_407
        );
    }
}
