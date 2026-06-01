//! Rekor transparency-log verification for a single sigstore-bundle
//! `tlogEntry` of kind `hashedrekord`.
//!
//! Two cryptographic checks:
//!
//! 1. **SignedEntryTimestamp (SET).** Rekor signs an RFC 8785-canonicalized
//!    JSON of `{body, integratedTime, logID, logIndex}` (keys ordered
//!    lexicographically by ASCII codepoint per RFC 8785) with one of its
//!    log keys (we pin them via the embedded Sigstore TrustedRoot). The
//!    SET is the per-entry "yes, I logged this" assertion.
//!
//! 2. **Inclusion proof.** Walks the `proof.hashes` siblings up to the
//!    log's tree root per RFC 6962, confirming the entry sits in a Merkle
//!    tree whose root matches `proof.rootHash`. The Merkle algorithm is
//!    adapted from sigstore-rs (Apache 2.0).
//!
//! Both must pass. SET alone proves "Rekor signed an entry with these
//! contents," but a misbehaving log could sign an entry and never publish
//! it. Inclusion proof anchors the entry to the public tree.
//!
//! On top of those, we verify that the entry's body actually *commits* to
//! the same blob+signature+pubkey the rest of the bundle does — i.e. the
//! transparency log entry is about *our* signature, not some other one.
//!
//! # KNOWN GAP — checkpoint signature verification deferred
//!
//! We don't verify the **checkpoint** signature in this v1. The
//! checkpoint is rekor's signed tree-head and would prove that
//! `rootHash` is the log's *publicly announced* root, not a side-tree
//! the log forked just for us. The SET already requires the Rekor key
//! to vouch for the entry; the checkpoint adds defense-in-depth.
//! Tracked in `releases/TRUST-ROOT.md` under "Known gaps."

use serde::Deserialize;

use crate::error::AppError;
use crate::updater::rekor_verify;
use crate::updater::trust::RekorKey;

/// `hashedrekord` v0.0.1 body shape — the canonical entry rekor stores.
#[derive(Debug, Deserialize)]
struct HashedRekordBody {
    kind: String,
    #[serde(rename = "apiVersion")]
    api_version: String,
    spec: HashedRekordSpec,
}

#[derive(Debug, Deserialize)]
struct HashedRekordSpec {
    data: SpecData,
    signature: SpecSignature,
}

#[derive(Debug, Deserialize)]
struct SpecData {
    hash: SpecHash,
}

#[derive(Debug, Deserialize)]
struct SpecHash {
    algorithm: String,
    /// Hex digest.
    value: String,
}

#[derive(Debug, Deserialize)]
struct SpecSignature {
    /// Base64 of the signature bytes.
    content: String,
    #[serde(rename = "publicKey")]
    public_key: SpecPublicKey,
}

#[derive(Debug, Deserialize)]
struct SpecPublicKey {
    /// Base64 of the PEM-encoded cert (cosign sign-blob keyless mode).
    content: String,
}

/// What we glean from successful Rekor verification.
#[derive(Debug, Clone)]
pub struct VerifiedRekorEntry {
    pub log_index: u64,
}

/// Verify the bundle's tlog entry against `manifest_sha256`, the leaf
/// cert's DER bytes (so we can confirm the entry references *our* cert),
/// and the signature bytes that the messageSignature commits to. Returns
/// the verified `VerifiedRekorEntry` on success.
///
/// Inputs:
/// - `manifest_sha256` — what the leaf signed; the entry's `hash.value`
///   must equal this.
/// - `leaf_cert_der` — the DER-encoded leaf cert. We re-decode the PEM
///   embedded in `signature.publicKey.content` back to DER and compare
///   bytewise — robust to PEM wrap-column / trailing-newline differences.
/// - `bundle_sig_bytes` — the messageSignature bytes; the entry's
///   `signature.content` must base64-decode to these same bytes.
/// - `canonical_body` — base64-decoded `canonicalizedBody`.
/// - `set_bytes` — base64-decoded SignedEntryTimestamp.
/// - `integrated_time` / `log_index` — from the bundle's tlog entry.
/// - `log_id` — 32-byte sha256 identifying which rekor key signed the SET.
/// - `proof_root_hash` / `proof_hashes` — inclusion proof material.
/// - `tree_size` — total leaves in the tree at proof time.
/// - `proof_leaf_index` — this entry's index in the tree.
/// - `rekor_keys` — pinned rekor public keys to try for SET verification.
#[allow(clippy::too_many_arguments)]
pub fn verify_rekor_entry(
    manifest_sha256: &[u8; 32],
    leaf_cert_der: &[u8],
    bundle_sig_bytes: &[u8],
    canonical_body: &[u8],
    set_bytes: &[u8],
    integrated_time: i64,
    log_index: u64,
    log_id: &[u8; 32],
    proof_root_hash: &[u8; 32],
    proof_hashes: &[[u8; 32]],
    tree_size: u64,
    proof_leaf_index: u64,
    rekor_keys: &[RekorKey],
) -> Result<VerifiedRekorEntry, AppError> {
    // ── 1. Body binding: the entry must reference our manifest hash,
    //       our signature bytes, and our leaf cert. ────────────────────
    let body: HashedRekordBody =
        serde_json::from_slice(canonical_body).map_err(|e| AppError::Update {
            message: format!("parsing rekor canonicalizedBody as hashedrekord: {e}"),
        })?;

    if body.kind != "hashedrekord" || body.api_version != "0.0.1" {
        return Err(AppError::Update {
            message: format!(
                "tlog entry has unsupported kind/apiVersion (`{}` / `{}`); expected hashedrekord 0.0.1",
                body.kind, body.api_version,
            ),
        });
    }

    if body.spec.data.hash.algorithm != "sha256" {
        return Err(AppError::Update {
            message: format!(
                "hashedrekord hash algorithm is `{}`, expected `sha256`",
                body.spec.data.hash.algorithm
            ),
        });
    }
    let body_hash_bytes = hex_decode(&body.spec.data.hash.value).map_err(|e| AppError::Update {
        message: format!("hex-decoding rekor body hash: {e}"),
    })?;
    if body_hash_bytes != manifest_sha256 {
        return Err(AppError::Update {
            message: "rekor entry's signed hash does not equal sha256(manifest) — the bundle's \
                      tlog entry is about a different blob"
                .into(),
        });
    }

    let body_sig_bytes =
        base64_std_decode(&body.spec.signature.content, "rekor signature.content")?;
    if body_sig_bytes != bundle_sig_bytes {
        return Err(AppError::Update {
            message:
                "rekor entry's signature does not equal the bundle's messageSignature.signature \
                      — the tlog entry is about a different signing"
                    .into(),
        });
    }

    let body_cert_pem_bytes = base64_std_decode(
        &body.spec.signature.public_key.content,
        "rekor publicKey.content",
    )?;
    let body_cert_pem =
        std::str::from_utf8(&body_cert_pem_bytes).map_err(|e| AppError::Update {
            message: format!("rekor publicKey.content is not UTF-8 PEM: {e}"),
        })?;
    let body_cert_der = pem_to_der(body_cert_pem)?;
    if body_cert_der != leaf_cert_der {
        return Err(AppError::Update {
            message: "rekor entry's embedded cert (DER-decoded) does not equal the bundle's leaf \
                      cert — the tlog entry references a different signing key"
                .into(),
        });
    }

    // ── 2 + 3. SET signature + Merkle inclusion (shared with the
    //          human-attestation path; see `super::rekor_verify`). ──────
    rekor_verify::verify_set_and_inclusion(
        canonical_body,
        set_bytes,
        integrated_time,
        log_index,
        log_id,
        proof_root_hash,
        proof_hashes,
        tree_size,
        proof_leaf_index,
        rekor_keys,
    )?;

    Ok(VerifiedRekorEntry { log_index })
}

// ---------------------------------------------------------------------------
// Small encoders
// ---------------------------------------------------------------------------

/// Decode a single CERTIFICATE PEM to DER bytes. Tolerant of leading /
/// trailing whitespace; rejects any content outside the expected
/// `-----BEGIN CERTIFICATE-----` / `-----END CERTIFICATE-----` block.
fn pem_to_der(pem: &str) -> Result<Vec<u8>, AppError> {
    use base64::Engine;
    let trimmed = pem.trim();
    let body = trimmed
        .strip_prefix("-----BEGIN CERTIFICATE-----")
        .and_then(|s| s.strip_suffix("-----END CERTIFICATE-----"))
        .ok_or_else(|| AppError::Update {
            message: "expected `-----BEGIN CERTIFICATE-----` / `-----END CERTIFICATE-----` markers"
                .into(),
        })?
        .trim();
    let stripped: String = body.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(stripped.as_bytes())
        .map_err(|e| AppError::Update {
            message: format!("base64-decoding PEM cert body: {e}"),
        })
}

fn base64_std_decode(s: &str, field: &str) -> Result<Vec<u8>, AppError> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(s.as_bytes())
        .map_err(|e| AppError::Update {
            message: format!("base64-decoding `{field}`: {e}"),
        })
}

fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err(format!("hex string `{s}` has odd length"));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Merkle proof tests live in `super::super::merkle` now; the proof
    // verification is exercised end-to-end through this module via the
    // `verify_inclusion_proof` call inside `verify_rekor_entry`.

    #[test]
    fn hex_rejects_odd_length() {
        assert!(hex_decode("abc").is_err());
    }
}
