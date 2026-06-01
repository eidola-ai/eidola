//! Shared **Sigstore Bundle v0.3** JSON deserialization types.
//!
//! Both the CI side (`cosign sign-blob --bundle`) and the human side
//! (`release-tool attest`) emit the same `application/vnd.dev.sigstore.bundle.v0.3+json`
//! container; the only structural difference is which arm of
//! `verificationMaterial.content` they populate (Fulcio leaf cert vs. a
//! key hint that points back to the rekor-body-embedded SSH pubkey). This
//! module owns the protobuf-JSON parse so the two verifiers can share
//! everything downstream of the structural decode.
//!
//! Reference spec:
//! <https://github.com/sigstore/protobuf-specs/blob/main/protos/sigstore_bundle.proto>
//!
//! We deserialize only the subset of fields the verifiers actually
//! consume. The DSSE-envelope variant of `Bundle.content` is intentionally
//! not modelled — every producer in this repo signs raw blobs, not DSSE
//! attestations.

use serde::Deserialize;

/// Top-level `dev.sigstore.bundle.v1.Bundle` (the protobuf message backing
/// the `application/vnd.dev.sigstore.bundle.v0.3+json` media type).
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CosignBundle {
    pub media_type: String,
    pub verification_material: VerificationMaterial,
    #[serde(default)]
    pub message_signature: Option<MessageSignature>,
    // DsseEnvelope variant ignored — neither producer in this repo emits
    // DSSE attestations.
}

/// `Bundle.verificationMaterial` — carries either a Fulcio leaf cert
/// (cosign keyless) or a `PublicKeyIdentifier` hint (our SSH-signed human
/// attestations).
///
/// Per the spec, the two arms are exclusive members of a protobuf `oneof
/// content`; we model them as two `Option`s and let the consumer dispatch
/// on which is `Some` (rejecting the both-or-neither degenerate cases).
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VerificationMaterial {
    /// Set by `cosign sign-blob` (and similar keyless signing flows).
    #[serde(default)]
    pub certificate: Option<RawCert>,
    /// Set by the human attestation flow — carries the SSH pubkey
    /// fingerprint as a `hint`. The actual pubkey bytes come from the
    /// rekor entry body (`spec.signature.publicKey.content`), since
    /// `PublicKeyIdentifier` is a *hint*, not a key.
    #[serde(default)]
    pub public_key: Option<RawPublicKeyHint>,
    pub tlog_entries: Vec<TlogEntry>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RawCert {
    /// Base64-encoded DER of the Fulcio leaf certificate.
    pub raw_bytes: String,
}

#[derive(Deserialize)]
pub(crate) struct RawPublicKeyHint {
    /// Opaque identifier the verifier uses to look up the pubkey. For our
    /// human attestations this is the SSH pubkey's SHA-256 fingerprint
    /// (lowercase hex) — the same value
    /// `compute_ssh_fingerprint` produces and that the trust constants
    /// pin in `TRUSTED_ATTESTANT_FINGERPRINTS`.
    pub hint: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct MessageSignature {
    pub message_digest: MessageDigest,
    /// Base64. The raw signature bytes — ECDSA DER for ECDSA-P256/P384
    /// keys, 64 bytes for Ed25519. Both the CI bundle (cosign + Fulcio)
    /// and the human bundle (cosign + local/PKCS#11/KMS key) carry the
    /// same shape here.
    pub signature: String,
}

#[derive(Deserialize)]
pub(crate) struct MessageDigest {
    pub algorithm: String,
    /// Base64.
    pub digest: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TlogEntry {
    /// Protobuf int64 — sigstore renders as a JSON string.
    pub log_index: String,
    pub log_id: LogId,
    #[allow(dead_code)]
    pub kind_version: KindVersion,
    /// Protobuf int64 — JSON string of seconds since epoch.
    pub integrated_time: String,
    #[serde(default)]
    pub inclusion_promise: Option<InclusionPromise>,
    #[serde(default)]
    pub inclusion_proof: Option<InclusionProof>,
    /// Base64 of the canonical `hashedrekord` v0.0.1 JSON. Both the
    /// CI sigstore path (publicKey arm = Fulcio leaf cert) and the
    /// human attestation path (publicKey arm = PKIX SPKI) ride this
    /// same body shape.
    pub canonicalized_body: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LogId {
    /// Base64 of the 32-byte sha256 log key id.
    pub key_id: String,
}

#[derive(Deserialize)]
pub(crate) struct KindVersion {
    #[allow(dead_code)]
    pub kind: String,
    #[allow(dead_code)]
    pub version: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InclusionPromise {
    /// Base64 of the rekor SET (SignedEntryTimestamp).
    pub signed_entry_timestamp: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InclusionProof {
    /// Protobuf int64 → JSON string.
    pub log_index: String,
    /// Base64 of the 32-byte root hash.
    pub root_hash: String,
    /// Protobuf int64 → JSON string.
    pub tree_size: String,
    /// Base64 of each 32-byte sibling hash.
    #[serde(default)]
    pub hashes: Vec<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub checkpoint: Option<RawCheckpoint>,
}

#[derive(Deserialize)]
pub(crate) struct RawCheckpoint {
    #[allow(dead_code)]
    #[serde(default)]
    pub envelope: String,
}
