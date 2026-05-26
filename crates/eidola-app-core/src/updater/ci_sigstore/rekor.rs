//! Rekor transparency-log verification for a single sigstore-bundle
//! `tlogEntry` of kind `hashedrekord`.
//!
//! Two cryptographic checks:
//!
//! 1. **SignedEntryTimestamp (SET).** Rekor signs an RFC 8785-canonicalized
//!    JSON of `{body, integratedTime, logIndex, logID}` with one of its
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
//! Note: we don't verify the **checkpoint** signature in this v1. The
//! checkpoint is rekor's signed tree-head and adds defense-in-depth
//! (proves the root we just computed is the tree's published root, not
//! a tree the log forked just for us). It's a follow-up; the SET +
//! inclusion proof gate already requires the rekor key to vouch for the
//! entry.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use signature::hazmat::PrehashVerifier;

use crate::error::AppError;
use crate::updater::trust::{KeyDetails, RekorKey};

/// `hashedrekord` v0.0.1 body shape — the canonical entry rekor stores.
#[derive(Debug, Deserialize)]
struct HashedRekordBody {
    kind: String,
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

    // ── 2. SET signature verification. ────────────────────────────────
    let key = rekor_keys
        .iter()
        .find(|k| k.log_id == *log_id)
        .ok_or_else(|| AppError::Update {
            message: format!(
                "no pinned Rekor key matches the bundle's logId `{}` — \
                 either the trusted root is stale, or the bundle is from a different log",
                hex_encode(log_id)
            ),
        })?;
    let canonical_body_b64 = base64_std_encode(canonical_body);
    let signed_payload = format!(
        r#"{{"body":"{body}","integratedTime":{it},"logIndex":{li},"logID":"{lid}"}}"#,
        body = canonical_body_b64,
        it = integrated_time,
        li = log_index,
        lid = hex_encode(log_id),
    );
    verify_rekor_signature(key, signed_payload.as_bytes(), set_bytes)?;

    // ── 3. Merkle inclusion proof. ────────────────────────────────────
    let leaf_hash = hash_leaf(canonical_body);
    let computed_root =
        compute_root_from_proof(proof_leaf_index, &leaf_hash, tree_size, proof_hashes)?;
    if computed_root != *proof_root_hash {
        return Err(AppError::Update {
            message: format!(
                "rekor inclusion proof's recomputed root `{}` ≠ declared rootHash `{}`",
                hex_encode(&computed_root),
                hex_encode(proof_root_hash)
            ),
        });
    }

    Ok(VerifiedRekorEntry { log_index })
}

fn verify_rekor_signature(
    key: &RekorKey,
    message: &[u8],
    signature: &[u8],
) -> Result<(), AppError> {
    match key.key_details {
        KeyDetails::EcdsaP256Sha256 => {
            use spki::DecodePublicKey;
            let vk =
                p256::ecdsa::VerifyingKey::from_public_key_der(&key.spki_der).map_err(|e| {
                    AppError::Update {
                        message: format!("parsing pinned Rekor P-256 pubkey: {e}"),
                    }
                })?;
            let sig =
                p256::ecdsa::Signature::from_der(signature).map_err(|e| AppError::Update {
                    message: format!("parsing Rekor SET signature DER (P-256): {e}"),
                })?;
            let prehash = Sha256::digest(message);
            vk.verify_prehash(&prehash, &sig)
                .map_err(|e| AppError::Update {
                    message: format!("Rekor SET signature failed P-256 verify: {e}"),
                })
        }
        KeyDetails::Ed25519 => {
            // Ed25519 SPKI pubkeys are SPKI-wrapped raw 32-byte pubkeys.
            // ed25519-dalek can ingest them via the spki adapter.
            use ed25519_dalek::pkcs8::DecodePublicKey;
            let vk =
                ed25519_dalek::VerifyingKey::from_public_key_der(&key.spki_der).map_err(|e| {
                    AppError::Update {
                        message: format!("parsing pinned Rekor Ed25519 pubkey: {e}"),
                    }
                })?;
            if signature.len() != 64 {
                return Err(AppError::Update {
                    message: format!(
                        "Ed25519 SET signature must be 64 bytes, got {}",
                        signature.len()
                    ),
                });
            }
            let mut sig_bytes = [0u8; 64];
            sig_bytes.copy_from_slice(signature);
            let sig = ed25519_dalek::Signature::from_bytes(&sig_bytes);
            vk.verify_strict(message, &sig)
                .map_err(|e| AppError::Update {
                    message: format!("Rekor SET signature failed Ed25519 verify: {e}"),
                })
        }
        KeyDetails::EcdsaP384Sha384 => Err(AppError::Update {
            message: "P-384 Rekor signing key encountered; not yet wired (no Rekor instance \
                      currently uses P-384)"
                .into(),
        }),
    }
}

// ---------------------------------------------------------------------------
// RFC 6962 Merkle inclusion proof verification
//
// The chain_inner / chain_border / decomp_inclusion_proof / inner_proof_size
// algorithm below is adapted from sigstore-rs (Apache 2.0), originally from
// transparency-dev's Merkle implementation. The trimmed version here only
// supports inclusion proofs (we don't need consistency proofs).
//
// Reference:
//   https://github.com/sigstore/sigstore-rs/blob/main/src/crypto/merkle/proof_verification.rs
// ---------------------------------------------------------------------------

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

fn compute_root_from_proof(
    index: u64,
    leaf_hash: &[u8; 32],
    tree_size: u64,
    proof_hashes: &[[u8; 32]],
) -> Result<[u8; 32], AppError> {
    if index >= tree_size {
        return Err(AppError::Update {
            message: format!("inclusion proof: leaf index {index} >= tree size {tree_size}"),
        });
    }
    let (inner, border) = decomp_inclusion_proof(index, tree_size);
    let expected_len = inner + border;
    if proof_hashes.len() as u64 != expected_len {
        return Err(AppError::Update {
            message: format!(
                "inclusion proof has {} hashes, expected {}",
                proof_hashes.len(),
                expected_len
            ),
        });
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

fn base64_std_encode(b: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(b)
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

fn hex_encode(b: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(b.len() * 2);
    for byte in b {
        write!(out, "{byte:02x}").unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 6962 test vectors lifted from transparency-dev's merkle tests.
    #[test]
    fn rfc6962_leaf_hash_empty() {
        let got = hash_leaf(b"");
        // sha256(0x00 || "") = sha256([0x00])
        let expected = {
            let mut h = Sha256::new();
            h.update([0x00]);
            let d: [u8; 32] = h.finalize().into();
            d
        };
        assert_eq!(got, expected);
    }

    #[test]
    fn inclusion_proof_single_leaf_tree() {
        // tree_size=1, index=0, no proof hashes needed → root == leaf hash.
        let leaf = hash_leaf(b"hello");
        let root = compute_root_from_proof(0, &leaf, 1, &[]).unwrap();
        assert_eq!(root, leaf);
    }

    #[test]
    fn inclusion_proof_two_leaf_tree() {
        // tree_size=2: leaf 0 path is one sibling (leaf 1's hash).
        let l0 = hash_leaf(b"a");
        let l1 = hash_leaf(b"b");
        let root = hash_children(&l0, &l1);

        let computed = compute_root_from_proof(0, &l0, 2, &[l1]).unwrap();
        assert_eq!(computed, root);

        let computed = compute_root_from_proof(1, &l1, 2, &[l0]).unwrap();
        assert_eq!(computed, root);
    }

    #[test]
    fn inclusion_proof_rejects_wrong_proof_length() {
        let l0 = hash_leaf(b"a");
        let err = compute_root_from_proof(0, &l0, 2, &[]).unwrap_err();
        assert!(format!("{err}").contains("inclusion proof has"));
    }

    #[test]
    fn inclusion_proof_rejects_index_out_of_range() {
        let l0 = hash_leaf(b"a");
        let err = compute_root_from_proof(5, &l0, 2, &[]).unwrap_err();
        assert!(format!("{err}").contains("leaf index"));
    }

    #[test]
    fn hex_round_trip() {
        let b: &[u8] = &[0x00, 0xab, 0xcd, 0xef, 0xff];
        let h = hex_encode(b);
        assert_eq!(h, "00abcdefff");
        assert_eq!(hex_decode(&h).unwrap(), b);
    }

    #[test]
    fn hex_rejects_odd_length() {
        assert!(hex_decode("abc").is_err());
    }
}
