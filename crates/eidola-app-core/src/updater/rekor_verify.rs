//! Shared Rekor verification: SET signature + Merkle inclusion proof.
//!
//! Both `ci_sigstore` (CI-signed `hashedrekord` with Fulcio cert) and
//! `human_attestation` (engineer-signed `hashedrekord` with PKIX SPKI
//! pubkey) end up at the same final two checks against Rekor:
//!
//! 1. The **SignedEntryTimestamp** binds `{body, integratedTime, logID,
//!    logIndex}` to a pinned Rekor public key — proof that the log
//!    actually accepted this entry.
//! 2. The **inclusion proof** anchors the entry's body hash inside a
//!    Merkle tree whose root matches the entry's stated `rootHash` —
//!    proof that the entry sits inside the same public tree everyone
//!    else verifies against.
//!
//! Body parsing and per-path cross-checks (cert chain on the CI side,
//! pubkey fingerprint on the human side) live in their respective
//! modules; only the path-agnostic SET + inclusion logic is shared here.

use sha2::{Digest, Sha256};
use signature::hazmat::PrehashVerifier;

use crate::error::AppError;

use super::merkle;
use super::trust::{KeyDetails, RekorKey};

/// Verify Rekor's SET signature over the canonical
/// `{body, integratedTime, logID, logIndex}` payload, then verify the
/// inclusion proof of `canonical_body`'s leaf hash up to
/// `proof_root_hash`.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_set_and_inclusion(
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
) -> Result<(), AppError> {
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
    // Keys ordered lexicographically by ASCII codepoint: body (0x62) <
    // integratedTime (0x69) < logID (0x6C 0x6F 0x67 0x49 0x44) <
    // logIndex (0x6C 0x6F 0x67 0x49 0x6E). `D` (0x44) < `n` (0x6E), so
    // logID precedes logIndex. Must match what Rekor actually signed.
    let signed_payload = format!(
        r#"{{"body":"{body}","integratedTime":{it},"logID":"{lid}","logIndex":{li}}}"#,
        body = canonical_body_b64,
        it = integrated_time,
        lid = hex_encode(log_id),
        li = log_index,
    );
    verify_rekor_set_signature(key, signed_payload.as_bytes(), set_bytes)?;

    let leaf_hash = merkle::hash_leaf(canonical_body);
    merkle::verify_inclusion_proof(
        proof_leaf_index,
        &leaf_hash,
        tree_size,
        proof_hashes,
        proof_root_hash,
    )
}

fn verify_rekor_set_signature(
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
// Blob-signature verification — used by the human-attestation path to
// verify the messageSignature against the extracted pubkey. The key type
// is detected from the SPKI's AlgorithmIdentifier OID.
//
// ECDSA paths verify against the prehashed sha256 (cosign sign-blob
// signs sha256(blob)); Ed25519 verifies against the raw message bytes
// (pure Ed25519, not Ed25519ph).
// ---------------------------------------------------------------------------

/// Verify a blob signature against the message using the public key
/// encoded as PKIX SubjectPublicKeyInfo DER. Dispatches on the SPKI's
/// AlgorithmIdentifier OID:
///
/// - `1.2.840.10045.2.1` (id-ecPublicKey) with P-256 curve params →
///   ECDSA over sha256(message), signature as ASN.1 DER.
/// - `1.2.840.10045.2.1` with P-384 curve params → ECDSA over
///   sha256(message), signature as ASN.1 DER.
/// - `1.3.101.112` (id-Ed25519) → Ed25519 over the raw message, 64-byte
///   signature.
pub(super) fn verify_blob_signature_with_spki(
    spki_der: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<(), AppError> {
    use der::Decode;
    let spki = spki::SubjectPublicKeyInfo::<spki::der::Any, spki::der::asn1::BitString>::from_der(
        spki_der,
    )
    .map_err(|e| AppError::Update {
        message: format!("parsing SPKI DER for blob signature verify: {e}"),
    })?;
    let alg_oid = spki.algorithm.oid;
    match alg_oid.as_bytes() {
        // id-ecPublicKey  (1.2.840.10045.2.1)
        [0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01] => {
            let curve_oid_any = spki.algorithm.parameters.ok_or_else(|| AppError::Update {
                message: "ecPublicKey SPKI is missing curve parameters".into(),
            })?;
            let curve_oid = curve_oid_any
                .decode_as::<spki::ObjectIdentifier>()
                .map_err(|e| AppError::Update {
                    message: format!("parsing ec curve OID: {e}"),
                })?;
            match curve_oid.as_bytes() {
                // prime256v1 (P-256) — 1.2.840.10045.3.1.7
                [0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07] => {
                    verify_ecdsa_p256_blob(spki_der, message, signature)
                }
                // secp384r1 (P-384) — 1.3.132.0.34
                [0x2b, 0x81, 0x04, 0x00, 0x22] => {
                    verify_ecdsa_p384_blob(spki_der, message, signature)
                }
                other => Err(AppError::Update {
                    message: format!(
                        "unsupported ECDSA curve OID bytes {other:?}; expected P-256 or P-384"
                    ),
                }),
            }
        }
        // id-Ed25519 (1.3.101.112)
        [0x2b, 0x65, 0x70] => verify_ed25519_blob(spki_der, message, signature),
        other => Err(AppError::Update {
            message: format!(
                "unsupported attestant key algorithm OID bytes {other:?}; \
                 expected ECDSA-P256, ECDSA-P384, or Ed25519"
            ),
        }),
    }
}

fn verify_ecdsa_p256_blob(
    spki_der: &[u8],
    message: &[u8],
    signature_der: &[u8],
) -> Result<(), AppError> {
    use spki::DecodePublicKey;
    let key =
        p256::ecdsa::VerifyingKey::from_public_key_der(spki_der).map_err(|e| AppError::Update {
            message: format!("parsing attestant P-256 pubkey: {e}"),
        })?;
    let sig = p256::ecdsa::Signature::from_der(signature_der).map_err(|e| AppError::Update {
        message: format!("parsing attestant blob signature DER (P-256): {e}"),
    })?;
    let prehash = Sha256::digest(message);
    key.verify_prehash(&prehash, &sig)
        .map_err(|e| AppError::Update {
            message: format!("attestant ECDSA P-256 signature verification failed: {e}"),
        })
}

fn verify_ecdsa_p384_blob(
    spki_der: &[u8],
    message: &[u8],
    signature_der: &[u8],
) -> Result<(), AppError> {
    use spki::DecodePublicKey;
    let key =
        p384::ecdsa::VerifyingKey::from_public_key_der(spki_der).map_err(|e| AppError::Update {
            message: format!("parsing attestant P-384 pubkey: {e}"),
        })?;
    let sig = p384::ecdsa::Signature::from_der(signature_der).map_err(|e| AppError::Update {
        message: format!("parsing attestant blob signature DER (P-384): {e}"),
    })?;
    // cosign's P-384 sign-blob still hashes with sha256, matching the CI side.
    let prehash = Sha256::digest(message);
    key.verify_prehash(&prehash, &sig)
        .map_err(|e| AppError::Update {
            message: format!("attestant ECDSA P-384 signature verification failed: {e}"),
        })
}

fn verify_ed25519_blob(spki_der: &[u8], message: &[u8], signature: &[u8]) -> Result<(), AppError> {
    use ed25519_dalek::pkcs8::DecodePublicKey;
    let key = ed25519_dalek::VerifyingKey::from_public_key_der(spki_der).map_err(|e| {
        AppError::Update {
            message: format!("parsing attestant Ed25519 pubkey: {e}"),
        }
    })?;
    if signature.len() != 64 {
        return Err(AppError::Update {
            message: format!(
                "Ed25519 blob signature must be 64 bytes, got {}",
                signature.len()
            ),
        });
    }
    let mut sig_bytes = [0u8; 64];
    sig_bytes.copy_from_slice(signature);
    let sig = ed25519_dalek::Signature::from_bytes(&sig_bytes);
    key.verify_strict(message, &sig)
        .map_err(|e| AppError::Update {
            message: format!("attestant Ed25519 signature verification failed: {e}"),
        })
}

// ---------------------------------------------------------------------------
// Small encoders
// ---------------------------------------------------------------------------

fn base64_std_encode(b: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(b)
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(out, "{b:02x}").unwrap();
    }
    out
}
