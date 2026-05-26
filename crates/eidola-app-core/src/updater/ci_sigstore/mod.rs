//! Verify the CI-side Sigstore bundle attached to `artifact-manifest.json`.
//!
//! The bundle is produced by `cosign sign-blob --bundle` in
//! `.github/workflows/tinfoil-build.yml`. It carries:
//!
//! - A Fulcio leaf certificate (Fulcio keyless, OIDC-bound). The cert's
//!   SAN extension records the GitHub Actions workflow URL + tag ref; the
//!   issuer extension records the OIDC issuer. The verifier pins both.
//! - The Fulcio chain (intermediate, root) so the leaf can be walked back
//!   to Sigstore's pinned Fulcio CAs.
//! - The raw ECDSA signature (over `sha256(manifest_bytes)`).
//! - The matching Rekor `hashedrekord` entry — `canonicalizedBody` plus
//!   `inclusionProof` and `inclusionPromise.signedEntryTimestamp` (the
//!   SET, a Rekor-key-signed assertion about the entry).
//!
//! # KNOWN GAPS — deferred defense-in-depth
//!
//! Two layers of cryptographic verification that *should* eventually
//! ship are intentionally **not** implemented today; the verifier is
//! still load-bearing without them, but they tighten the screws.
//!
//! 1. **Signed Certificate Timestamp (SCT) verification.** The Fulcio
//!    leaf cert embeds an SCT proving the cert was logged in a public
//!    Certificate Transparency log. Verifying it would catch Fulcio
//!    misissuance (a malicious or compromised Fulcio issuing certs for
//!    identities it shouldn't). Our OIDC-identity match + Fulcio chain
//!    walk make this defense-in-depth, not load-bearing.
//!
//! 2. **Rekor checkpoint signature verification.** The inclusion proof
//!    we compute roots out to `rootHash`; checking the checkpoint
//!    signature would prove that `rootHash` is the log's *publicly
//!    announced* root, not a side-tree the log forked just for us. The
//!    SET already requires the Rekor key to vouch for the entry; the
//!    checkpoint adds independence-from-private-forks. Future work.
//!
//! Both are tracked in `releases/TRUST-ROOT.md` under "Known gaps."

use serde::Deserialize;

use crate::error::AppError;
use crate::trust_root;

use super::trust::TrustedRoot;

mod cert;
mod rekor;

/// Facts the verifier proves and the rest of the pipeline consumes.
#[derive(Debug, Clone)]
pub struct VerifiedCiSignature {
    /// Sha256 of the manifest bytes — proved equal to what the bundle
    /// claims and what the cert+signature commit to.
    pub manifest_sha256: [u8; 32],
    /// The Fulcio cert's SAN URI — must match
    /// [`trust_root::EXPECTED_CI_IDENTITY_PATTERN`] (glob-matched).
    pub ci_identity: String,
    /// The OIDC issuer extension — must equal
    /// [`trust_root::EXPECTED_CI_ISSUER`].
    pub ci_issuer: String,
    /// Rekor log index for the verified entry. Surfaced to the UI so the
    /// user can independently look up the entry on rekor.sigstore.dev.
    pub rekor_log_index: u64,
}

/// Verify the bundle against the manifest bytes and the embedded trust
/// root. Returns the verified facts on success.
///
/// **4c.1 stub** — does structural checks; cryptographic verification is
/// marked with `TODO (4c.2)` below.
pub fn verify_ci_signature(
    manifest_bytes: &[u8],
    bundle_bytes: &[u8],
    trust: &TrustedRoot,
) -> Result<VerifiedCiSignature, AppError> {
    use sha2::{Digest, Sha256};

    let manifest_sha256: [u8; 32] = Sha256::digest(manifest_bytes).into();

    let bundle: CosignBundle =
        serde_json::from_slice(bundle_bytes).map_err(|e| AppError::Update {
            message: format!("parsing CI Sigstore bundle as JSON: {e}"),
        })?;

    if !bundle
        .media_type
        .starts_with("application/vnd.dev.sigstore.bundle")
    {
        return Err(AppError::Update {
            message: format!(
                "CI Sigstore bundle has unexpected mediaType `{}` (expected `application/vnd.dev.sigstore.bundle.*`)",
                bundle.media_type
            ),
        });
    }

    // ── structural: messageSignature ────────────────────────────────────
    let msg_sig = bundle
        .message_signature
        .as_ref()
        .ok_or_else(|| AppError::Update {
            message: "CI Sigstore bundle has no `messageSignature` (only `dsseEnvelope`?) — \
                  this verifier expects cosign sign-blob output, not a DSSE attestation"
                .into(),
        })?;
    if msg_sig.message_digest.algorithm != "SHA2_256" {
        return Err(AppError::Update {
            message: format!(
                "CI Sigstore bundle messageDigest.algorithm is `{}`, expected `SHA2_256`",
                msg_sig.message_digest.algorithm
            ),
        });
    }
    let claimed_digest = base64_std_decode(&msg_sig.message_digest.digest, "messageDigest.digest")?;
    if claimed_digest.len() != 32 {
        return Err(AppError::Update {
            message: format!(
                "messageDigest.digest is {} bytes, expected 32 (sha256)",
                claimed_digest.len()
            ),
        });
    }
    if claimed_digest != manifest_sha256 {
        return Err(AppError::Update {
            message: "manifest sha256 does not match the bundle's claimed messageDigest \
                      — the wrong manifest or bundle file was downloaded"
                .into(),
        });
    }
    // ── structural: verificationMaterial ─────────────────────────────────
    let vm = &bundle.verification_material;
    let cert_b64 = vm
        .certificate
        .as_ref()
        .ok_or_else(|| AppError::Update {
            message: "CI Sigstore bundle has no `verificationMaterial.certificate` — \
                      cosign keyless signing should always emit a Fulcio leaf cert"
                .into(),
        })?
        .raw_bytes
        .as_str();
    let cert_der = base64_std_decode(cert_b64, "certificate.rawBytes")?;
    let signature_bytes = base64_std_decode(&msg_sig.signature, "messageSignature.signature")?;

    // ── structural: tlog entry (Rekor hashedrekord) ──────────────────────
    if vm.tlog_entries.len() != 1 {
        return Err(AppError::Update {
            message: format!(
                "CI Sigstore bundle has {} tlog entries; cosign sign-blob produces exactly 1",
                vm.tlog_entries.len()
            ),
        });
    }
    let entry = &vm.tlog_entries[0];
    let log_index = entry
        .log_index
        .parse::<u64>()
        .map_err(|e| AppError::Update {
            message: format!(
                "tlog entry has unparseable logIndex `{}`: {e}",
                entry.log_index
            ),
        })?;

    let canonical_body = base64_std_decode(&entry.canonicalized_body, "canonicalizedBody")?;

    let inclusion_promise = entry
        .inclusion_promise
        .as_ref()
        .ok_or_else(|| AppError::Update {
            message: "tlog entry missing inclusionPromise (SignedEntryTimestamp) — \
                      required for transparency-log binding"
                .into(),
        })?;
    let set_bytes = base64_std_decode(
        &inclusion_promise.signed_entry_timestamp,
        "signedEntryTimestamp",
    )?;

    let inclusion_proof = entry
        .inclusion_proof
        .as_ref()
        .ok_or_else(|| AppError::Update {
            message: "tlog entry missing inclusionProof — required to bind the entry \
                      to a public log root"
                .into(),
        })?;
    let root_hash_bytes = base64_std_decode(&inclusion_proof.root_hash, "inclusionProof.rootHash")?;
    let root_hash: [u8; 32] =
        root_hash_bytes
            .as_slice()
            .try_into()
            .map_err(|_| AppError::Update {
                message: format!(
                    "inclusionProof.rootHash is {} bytes, expected 32 (sha256)",
                    root_hash_bytes.len()
                ),
            })?;
    let proof_leaf_index =
        inclusion_proof
            .log_index
            .parse::<u64>()
            .map_err(|e| AppError::Update {
                message: format!(
                    "inclusionProof.logIndex `{}` is not a valid u64: {e}",
                    inclusion_proof.log_index
                ),
            })?;
    let tree_size = inclusion_proof
        .tree_size
        .parse::<u64>()
        .map_err(|e| AppError::Update {
            message: format!(
                "inclusionProof.treeSize `{}` is not a valid u64: {e}",
                inclusion_proof.tree_size
            ),
        })?;
    let proof_hashes: Vec<[u8; 32]> = inclusion_proof
        .hashes
        .iter()
        .map(|h| {
            let decoded = base64_std_decode(h, "inclusionProof.hashes[]")?;
            decoded.as_slice().try_into().map_err(|_| AppError::Update {
                message: format!(
                    "inclusionProof.hashes[] entry is {} bytes, expected 32 (sha256)",
                    decoded.len()
                ),
            })
        })
        .collect::<Result<_, AppError>>()?;
    let log_id_bytes = base64_std_decode(&entry.log_id.key_id, "tlogEntry.logId.keyId")?;
    let log_id: [u8; 32] = log_id_bytes
        .as_slice()
        .try_into()
        .map_err(|_| AppError::Update {
            message: format!(
                "tlogEntry.logId.keyId is {} bytes, expected 32 (sha256)",
                log_id_bytes.len()
            ),
        })?;
    let integrated_time = entry
        .integrated_time
        .parse::<i64>()
        .map_err(|e| AppError::Update {
            message: format!(
                "tlogEntry.integratedTime `{}` is not a valid i64: {e}",
                entry.integrated_time
            ),
        })?;

    // ── cryptographic verification ───────────────────────────────────────

    // 1+2 (chain + identity extraction). 2 (SCT) is deferred to 4c.3.
    let leaf_info = cert::verify_chain_and_extract(&cert_der, &trust.fulcio_cas)?;

    // 3 (identity match).
    if !cert::glob_matches(trust_root::EXPECTED_CI_IDENTITY_PATTERN, &leaf_info.san_uri) {
        return Err(AppError::Update {
            message: format!(
                "leaf cert SAN URI `{}` does not match expected pattern `{}` — the bundle was \
                 not signed by our CI workflow",
                leaf_info.san_uri,
                trust_root::EXPECTED_CI_IDENTITY_PATTERN
            ),
        });
    }
    if leaf_info.oidc_issuer != trust_root::EXPECTED_CI_ISSUER {
        return Err(AppError::Update {
            message: format!(
                "leaf cert OIDC issuer `{}` ≠ expected `{}`",
                leaf_info.oidc_issuer,
                trust_root::EXPECTED_CI_ISSUER
            ),
        });
    }

    // 4 (signature over manifest hash).
    cert::verify_blob_signature(
        &leaf_info.spki_der,
        leaf_info.leaf_key_alg,
        &manifest_sha256,
        &signature_bytes,
    )?;

    // Sanity: Rekor's integratedTime must fall within the leaf cert's
    // validity window. Outside it, either the cert was expired when the
    // signature was logged or the rekor entry was backdated — either way,
    // suspicious.
    let it_u64 = u64::try_from(integrated_time).map_err(|_| AppError::Update {
        message: format!("tlogEntry.integratedTime `{integrated_time}` is negative"),
    })?;
    if it_u64 < leaf_info.not_before || it_u64 > leaf_info.not_after {
        return Err(AppError::Update {
            message: format!(
                "Rekor integratedTime {it_u64} is outside leaf cert validity \
                 [{}, {}] — bundle is malformed or backdated",
                leaf_info.not_before, leaf_info.not_after
            ),
        });
    }

    // 5+6+7 (rekor body binding + SET + inclusion proof).
    let verified_rekor = rekor::verify_rekor_entry(
        &manifest_sha256,
        &cert_der,
        &signature_bytes,
        &canonical_body,
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

    Ok(VerifiedCiSignature {
        manifest_sha256,
        ci_identity: leaf_info.san_uri,
        ci_issuer: leaf_info.oidc_issuer,
        rekor_log_index: verified_rekor.log_index,
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

// ---------------------------------------------------------------------------
// Cosign Sigstore Bundle v0.3 JSON shape — the subset we consume
// ---------------------------------------------------------------------------
//
// Schema reference:
// https://github.com/sigstore/protobuf-specs/blob/main/protos/sigstore_bundle.proto

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CosignBundle {
    media_type: String,
    verification_material: VerificationMaterial,
    #[serde(default)]
    message_signature: Option<MessageSignature>,
    // DsseEnvelope variant ignored for now — cosign sign-blob emits
    // messageSignature. If we later sign in-toto attestations, the parser
    // will need to grow a `dsseEnvelope` arm here.
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VerificationMaterial {
    /// Either `certificate` (cosign v2+) or `x509CertificateChain` (older
    /// cosign / sigstore-rs); we only support the new shape today.
    #[serde(default)]
    certificate: Option<RawCert>,
    tlog_entries: Vec<TlogEntry>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawCert {
    raw_bytes: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MessageSignature {
    message_digest: MessageDigest,
    signature: String,
}

#[derive(Deserialize)]
struct MessageDigest {
    algorithm: String,
    digest: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TlogEntry {
    /// Protobuf int64 — sigstore renders as a JSON string.
    log_index: String,
    #[allow(dead_code)]
    log_id: LogId,
    #[allow(dead_code)]
    kind_version: KindVersion,
    /// Protobuf int64 — JSON string of seconds since epoch.
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    key_id: String,
}

#[derive(Deserialize)]
struct KindVersion {
    #[allow(dead_code)]
    kind: String,
    #[allow(dead_code)]
    version: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct InclusionPromise {
    signed_entry_timestamp: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct InclusionProof {
    #[allow(dead_code)]
    log_index: String,
    root_hash: String,
    #[allow(dead_code)]
    tree_size: String,
    #[allow(dead_code)]
    #[serde(default)]
    hashes: Vec<String>,
    #[allow(dead_code)]
    #[serde(default)]
    checkpoint: Option<RawCheckpoint>,
}

#[derive(Deserialize)]
struct RawCheckpoint {
    #[allow(dead_code)]
    #[serde(default)]
    envelope: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_bundle(manifest_digest_b64: &str) -> Vec<u8> {
        format!(
            r#"{{
                "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
                "verificationMaterial": {{
                    "certificate": {{
                        "rawBytes": "AAA="
                    }},
                    "tlogEntries": [{{
                        "logIndex": "12345",
                        "logId": {{ "keyId": "wNI9atQGlz+VWfO6LRygH4QUfY/8W4RFwiT5i5WRgB0=" }},
                        "kindVersion": {{ "kind": "hashedrekord", "version": "0.0.1" }},
                        "integratedTime": "1700000000",
                        "inclusionPromise": {{ "signedEntryTimestamp": "AAA=" }},
                        "inclusionProof": {{
                            "logIndex": "12345",
                            "rootHash": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
                            "treeSize": "67890",
                            "hashes": []
                        }},
                        "canonicalizedBody": "AAA="
                    }}]
                }},
                "messageSignature": {{
                    "messageDigest": {{
                        "algorithm": "SHA2_256",
                        "digest": "{manifest_digest_b64}"
                    }},
                    "signature": "AAA="
                }}
            }}"#
        )
        .into_bytes()
    }

    fn fake_trust() -> TrustedRoot {
        TrustedRoot {
            fulcio_cas: vec![],
            rekor_keys: vec![],
            ctlog_keys: vec![],
        }
    }

    #[test]
    fn structural_check_passes_for_matching_digest() {
        // The structural happy path falls through to real cryptographic
        // verification, which then fails because our minimal_bundle has a
        // placeholder `AAA=` cert that isn't a valid Fulcio leaf. The
        // *useful* unit-testable structural-fail branches are covered by
        // the rejects_* tests below; a full happy-path verification needs
        // a real CI-signed fixture and lives in integration tests once we
        // have a real `v*` tag.
        use base64::Engine;
        use sha2::{Digest, Sha256};

        let manifest = b"hello world";
        let digest = Sha256::digest(manifest);
        let digest_b64 = base64::engine::general_purpose::STANDARD.encode(digest);
        let bundle = minimal_bundle(&digest_b64);

        let err = verify_ci_signature(manifest, &bundle, &fake_trust()).unwrap_err();
        // The error must be from the *crypto* layer (cert parse / chain
        // walk), not from the earlier structural checks — that proves the
        // pipeline got past structural validation.
        let msg = format!("{err}");
        assert!(
            msg.contains("Fulcio leaf cert") || msg.contains("no Fulcio CA"),
            "expected a crypto-layer error, got: {msg}"
        );
    }

    #[test]
    fn rejects_digest_mismatch() {
        let bundle = minimal_bundle("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=");
        let err =
            verify_ci_signature(b"different manifest bytes", &bundle, &fake_trust()).unwrap_err();
        assert!(format!("{err}").contains("does not match"), "got: {err}");
    }

    #[test]
    fn rejects_wrong_media_type() {
        let bundle = br#"{
            "mediaType": "application/json",
            "verificationMaterial": { "tlogEntries": [] },
            "messageSignature": { "messageDigest": { "algorithm": "SHA2_256", "digest": "AAA=" }, "signature": "AAA=" }
        }"#.to_vec();
        let err = verify_ci_signature(b"x", &bundle, &fake_trust()).unwrap_err();
        assert!(format!("{err}").contains("mediaType"), "got: {err}");
    }

    #[test]
    fn rejects_missing_message_signature() {
        let bundle = br#"{
            "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
            "verificationMaterial": { "tlogEntries": [] }
        }"#
        .to_vec();
        let err = verify_ci_signature(b"x", &bundle, &fake_trust()).unwrap_err();
        assert!(format!("{err}").contains("messageSignature"), "got: {err}");
    }

    #[test]
    fn rejects_zero_or_many_tlog_entries() {
        // Zero entries
        let bundle = br#"{
            "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
            "verificationMaterial": {
                "certificate": {"rawBytes": "AAA="},
                "tlogEntries": []
            },
            "messageSignature": {
                "messageDigest": {"algorithm": "SHA2_256", "digest": "uoBfsmA2ZAchAYDOtCQpYJ70eRtX1upI6F1XHKgM4mE="},
                "signature": "AAA="
            }
        }"#
        .to_vec();
        // sha256("zero") = ba807ec6603664072100100ce8ce4258f3878a6577f5e9b51ed8ca5fe89e8b0d
        // base64 of that = uoB+xmA2ZAchAQDOzs... but I use a different manifest to make this fail differently
        // Doesn't matter — the manifest-digest check happens before tlog check, so use a string that's right for some manifest
        // For this test we just check the tlog-count branch:
        let bundle_str = std::str::from_utf8(&bundle).unwrap();
        // Substitute in a digest that matches some chosen manifest to bypass earlier checks
        use base64::Engine;
        use sha2::{Digest, Sha256};
        let mfst = b"x";
        let digest = Sha256::digest(mfst);
        let b64 = base64::engine::general_purpose::STANDARD.encode(digest);
        let updated = bundle_str.replace("uoBfsmA2ZAchAYDOtCQpYJ70eRtX1upI6F1XHKgM4mE=", &b64);
        let err = verify_ci_signature(mfst, updated.as_bytes(), &fake_trust()).unwrap_err();
        assert!(format!("{err}").contains("tlog entries"), "got: {err}");
    }
}
