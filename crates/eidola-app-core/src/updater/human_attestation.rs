//! Verify a human-signed release attestation.
//!
//! Engineers sign the attestation JSON file with their hardware-backed
//! key via `cosign sign-blob --key …`; cosign posts the result to
//! Sigstore Rekor as a `hashedrekord` v0.0.1 entry and writes a
//! **Sigstore Bundle v0.3** companion file alongside the signature.
//! This module reads that bundle, walks every cryptographic binding,
//! and returns the verified facts.
//!
//! The bundle is structurally identical to the CI bundle (cosign
//! produces both) with one difference: the human path carries
//! `verificationMaterial.publicKey` (a hint plus the actual PEM PKIX
//! pubkey embedded in the rekor body) instead of
//! `verificationMaterial.certificate` (a Fulcio leaf cert). Trust is
//! anchored by pinned key fingerprints rather than by a Fulcio chain +
//! OIDC identity match.
//!
//! Pipeline (one attestation):
//!
//! 1. **Bundle parse.** Deserialize as
//!    [`super::sigstore_bundle::CosignBundle`]; require `mediaType` to
//!    start with `application/vnd.dev.sigstore.bundle` and exactly one
//!    tlog entry; require the `publicKey` arm of `verificationMaterial`
//!    (not `certificate`).
//! 2. **messageSignature extraction.** Base64-decode
//!    `messageSignature.signature` to the raw signature bytes (ECDSA
//!    DER for ECDSA-P256/P384 keys, 64 bytes for Ed25519).
//! 3. **Body parse.** Base64-decode the tlog entry's `canonicalizedBody`
//!    into the canonical hashedrekord JSON; require `kind=hashedrekord`,
//!    `apiVersion=0.0.1`, `data.hash.algorithm=sha256`.
//! 4. **Body binding.** The body's `data.hash.value` (hex) must equal
//!    `sha256(attestation_bytes)`. The body's `messageDigest.digest`
//!    must independently equal the same sha256. The body's
//!    `signature.content` (base64) must equal the bundle's
//!    `messageSignature.signature` bytes.
//! 5. **Pubkey extraction + fingerprint check.** Decode
//!    `body.signature.publicKey.content` (base64) to the PEM
//!    `-----BEGIN PUBLIC KEY-----` block, decode to PKIX SubjectPublicKeyInfo
//!    DER, fingerprint = `sha256(spki_der)`. Require it equal a member
//!    of [`trust_root::TRUSTED_ATTESTANT_FINGERPRINTS`]. The bundle's
//!    `verificationMaterial.publicKey.hint` is producer-defined
//!    advisory metadata and is intentionally not cross-checked —
//!    identity is anchored entirely by the rekor body's embedded
//!    SPKI.
//! 6. **Blob signature verify.** Dispatch on the SPKI's
//!    AlgorithmIdentifier OID (ECDSA-P256, ECDSA-P384, Ed25519) and
//!    verify the messageSignature against `attestation_bytes`. ECDSA
//!    paths verify against `sha256(attestation_bytes)`; Ed25519
//!    verifies against the raw bytes. See
//!    [`super::rekor_verify::verify_blob_signature_with_spki`].
//! 7. **SET verify.** Reconstruct the canonical JSON
//!    `{"body":"<b64>","integratedTime":N,"logID":"<hex>","logIndex":N}`,
//!    select the pinned Rekor key by `logID`, ECDSA-P256 or Ed25519
//!    verify the SET signature against it. Shared with the CI path —
//!    see [`super::rekor_verify::verify_set_and_inclusion`].
//! 8. **Merkle inclusion proof.** RFC 6962 walk via
//!    [`super::merkle`]; recomputed root must equal `rootHash`. Shared
//!    with the CI path via [`super::rekor_verify`].
//!
//! See also the known gap in `releases/TRUST-ROOT.md` on Rekor
//! checkpoint signature verification (same caveat as the CI side).

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::error::AppError;
use crate::trust_root;

use super::rekor_verify;
use super::sigstore_bundle::CosignBundle;
use super::trust::TrustedRoot;
use super::{ReleaseIndex, VerifiedAttestation, VerifiedClaim};

// Re-export the attestant-key classifier so external callers (notably
// `release-tool`) can validate a candidate signing key against the same
// algorithm allowlist the updater enforces at verify time — guarantees
// a release can't be published in a shape the updater would later refuse.
pub use rekor_verify::{AttestantKeyAlgorithm, classify_attestant_spki_algorithm};

/// Verify a single human attestation end-to-end: the cosign-emitted
/// signature against the attestant's pinned key fingerprint, the Rekor
/// inclusion proof (this file), the structural cross-checks against
/// the release index, and the character-exact equality of every signed
/// claim against its pinned template rendering. Returns a
/// fully-populated [`VerifiedAttestation`] on success.
pub fn verify_human_attestation(
    attestation_bytes: &[u8],
    bundle_bytes: &[u8],
    release: &ReleaseIndex,
    trust: &TrustedRoot,
) -> Result<VerifiedAttestation, AppError> {
    let attestation_sha256: [u8; 32] = Sha256::digest(attestation_bytes).into();

    // ── 1. Bundle parse (Sigstore Bundle v0.3) ──────────────────────
    let bundle: CosignBundle =
        serde_json::from_slice(bundle_bytes).map_err(|e| AppError::Update {
            message: format!("parsing attestation bundle JSON: {e}"),
        })?;
    if !bundle
        .media_type
        .starts_with("application/vnd.dev.sigstore.bundle")
    {
        return Err(AppError::Update {
            message: format!(
                "attestation bundle has unexpected mediaType `{}` (expected \
                 `application/vnd.dev.sigstore.bundle.*`)",
                bundle.media_type
            ),
        });
    }
    let vm = &bundle.verification_material;
    if vm.tlog_entries.len() != 1 {
        return Err(AppError::Update {
            message: format!(
                "attestation bundle has {} tlog entries, expected exactly 1",
                vm.tlog_entries.len()
            ),
        });
    }
    let entry = &vm.tlog_entries[0];

    // Human attestations use the `publicKey` arm of the `verificationMaterial`
    // oneof (vs. CI's `certificate`). Fail loudly on a CI-shaped bundle
    // arriving here — that's a mismatched-verifier bug. We only require the
    // arm to be present; its `hint` value is producer-defined metadata and
    // intentionally not cross-checked, since identity is anchored entirely
    // by the SPKI embedded in the rekor body matching
    // `TRUSTED_ATTESTANT_FINGERPRINTS` (see step 5 below).
    if vm.public_key.is_none() {
        return Err(AppError::Update {
            message: "attestation bundle has no `verificationMaterial.publicKey` — \
                      human attestations carry the publicKey arm, not a Fulcio cert"
                .into(),
        });
    }
    if vm.certificate.is_some() {
        return Err(AppError::Update {
            message: "attestation bundle carries both `publicKey` and `certificate` in \
                      `verificationMaterial`; the protobuf `oneof content` requires exactly one"
                .into(),
        });
    }

    // ── 2. messageSignature: raw signature bytes ─────────────────────
    let msg_sig = bundle
        .message_signature
        .as_ref()
        .ok_or_else(|| AppError::Update {
            message: "attestation bundle has no `messageSignature` — human attestations carry the \
                  raw signature bytes there"
                .into(),
        })?;
    if msg_sig.message_digest.algorithm != "SHA2_256" {
        return Err(AppError::Update {
            message: format!(
                "attestation bundle messageDigest.algorithm is `{}`, expected `SHA2_256`",
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
    if claimed_digest != attestation_sha256 {
        return Err(AppError::Update {
            message: "attestation sha256 does not match the bundle's claimed messageDigest \
                      — the wrong attestation or bundle file was downloaded"
                .into(),
        });
    }
    let signature_bytes = base64_std_decode(&msg_sig.signature, "messageSignature.signature")?;

    // ── 3. Body parse ────────────────────────────────────────────────
    let body_bytes = base64_std_decode(&entry.canonicalized_body, "tlog canonicalizedBody")?;
    let body: HashedRekordBody =
        serde_json::from_slice(&body_bytes).map_err(|e| AppError::Update {
            message: format!("parsing rekor entry body as hashedrekord: {e}"),
        })?;
    if body.kind != "hashedrekord" || body.api_version != "0.0.1" {
        return Err(AppError::Update {
            message: format!(
                "attestation rekor entry has unsupported kind/apiVersion (`{}` / `{}`); expected hashedrekord 0.0.1",
                body.kind, body.api_version
            ),
        });
    }
    if body.spec.data.hash.algorithm != "sha256" {
        return Err(AppError::Update {
            message: format!(
                "attestation rekor body hash algorithm is `{}`, expected `sha256`",
                body.spec.data.hash.algorithm
            ),
        });
    }

    // ── 4. Body binding to attestation bytes ─────────────────────────
    let body_hash_bytes = hex_decode(&body.spec.data.hash.value).map_err(|e| AppError::Update {
        message: format!("hex-decoding attestation rekor body hash: {e}"),
    })?;
    if body_hash_bytes != attestation_sha256 {
        return Err(AppError::Update {
            message: "attestation rekor body hash does not equal sha256(attestation) — the \
                      tlog entry is about a different file"
                .into(),
        });
    }
    // The body's signature.content must equal the bundle's messageSignature
    // bytes — defends against a bundle that points at a tlog entry for some
    // other signing event.
    let body_sig_bytes = base64_std_decode(
        &body.spec.signature.content,
        "attestation rekor signature.content",
    )?;
    if body_sig_bytes != signature_bytes {
        return Err(AppError::Update {
            message:
                "attestation rekor body signature.content ≠ bundle messageSignature.signature \
                      — the tlog entry is about a different signing"
                    .into(),
        });
    }

    // ── 5. Pubkey + fingerprint ──────────────────────────────────────
    let pubkey_pem_bytes = base64_std_decode(
        &body.spec.signature.public_key.content,
        "attestation rekor publicKey.content",
    )?;
    let spki_der = pem_public_key_to_spki_der(&pubkey_pem_bytes)?;
    let fingerprint_bytes = Sha256::digest(&spki_der);
    let fingerprint_hex = hex_encode(fingerprint_bytes.as_slice());
    if !trust_root::TRUSTED_ATTESTANT_FINGERPRINTS
        .iter()
        .any(|fp| fp.eq_ignore_ascii_case(&fingerprint_hex))
    {
        return Err(AppError::Update {
            message: format!(
                "attestation was signed by attestant key with fingerprint `{fingerprint_hex}` \
                 (sha256 of PKIX SubjectPublicKeyInfo DER), which is NOT in this client's \
                 TRUSTED_ATTESTANT_FINGERPRINTS — the signer is not authorized for this release line"
            ),
        });
    }

    // ── 6. Blob signature verify ─────────────────────────────────────
    rekor_verify::verify_blob_signature_with_spki(&spki_der, attestation_bytes, &signature_bytes)?;

    // ── 7 + 8. Shared SET + Merkle inclusion. ────────────────────────
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
    let log_index: u64 = entry.log_index.parse().map_err(|e| AppError::Update {
        message: format!(
            "tlogEntry.logIndex `{}` is not a valid u64: {e}",
            entry.log_index
        ),
    })?;
    let integrated_time: i64 = entry
        .integrated_time
        .parse()
        .map_err(|e| AppError::Update {
            message: format!(
                "tlogEntry.integratedTime `{}` is not a valid i64: {e}",
                entry.integrated_time
            ),
        })?;
    let inclusion_promise = entry
        .inclusion_promise
        .as_ref()
        .ok_or_else(|| AppError::Update {
            message: "tlog entry missing inclusionPromise (SignedEntryTimestamp) — required for \
                  transparency-log binding"
                .into(),
        })?;
    let set_bytes = base64_std_decode(
        &inclusion_promise.signed_entry_timestamp,
        "inclusionPromise.signedEntryTimestamp",
    )?;
    let inclusion_proof = entry
        .inclusion_proof
        .as_ref()
        .ok_or_else(|| AppError::Update {
            message: "tlog entry missing inclusionProof — required to bind the entry to a public \
                  log root"
                .into(),
        })?;
    let root_hash_bytes = base64_std_decode(&inclusion_proof.root_hash, "inclusionProof.rootHash")?;
    let root_hash: [u8; 32] =
        root_hash_bytes
            .as_slice()
            .try_into()
            .map_err(|_| AppError::Update {
                message: format!(
                    "inclusionProof.rootHash is {} bytes, expected 32",
                    root_hash_bytes.len()
                ),
            })?;
    let proof_leaf_index: u64 =
        inclusion_proof
            .log_index
            .parse()
            .map_err(|e| AppError::Update {
                message: format!(
                    "inclusionProof.logIndex `{}` is not a valid u64: {e}",
                    inclusion_proof.log_index
                ),
            })?;
    let tree_size: u64 = inclusion_proof
        .tree_size
        .parse()
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
                    "inclusionProof.hashes[] entry is {} bytes, expected 32",
                    decoded.len()
                ),
            })
        })
        .collect::<Result<_, AppError>>()?;
    rekor_verify::verify_set_and_inclusion(
        &body_bytes,
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

    // ── 9. Content verification (template equality + cross-checks) ───
    verify_attestation_content(attestation_bytes, release, &fingerprint_hex, log_index)
}

/// Parse the attestation prose, cross-check its top-level fields
/// against the release index, and verify every signed claim is the
/// character-for-character rendering of its pinned template (with
/// declared `cross_checks` resolving to the corresponding `release.x.y`
/// values).
fn verify_attestation_content(
    attestation_bytes: &[u8],
    release: &ReleaseIndex,
    fingerprint_hex: &str,
    rekor_log_index: u64,
) -> Result<VerifiedAttestation, AppError> {
    let attestation: serde_json::Value =
        serde_json::from_slice(attestation_bytes).map_err(|e| AppError::Update {
            message: format!("parsing attestation JSON: {e}"),
        })?;
    let prose: AttestationProse =
        serde_json::from_value(attestation.clone()).map_err(|e| AppError::Update {
            message: format!("attestation JSON does not match schema: {e}"),
        })?;

    // Schema-version gate.
    if !trust_root::SUPPORTED_ATTESTATION_SCHEMA_VERSIONS.contains(&prose.schema_version) {
        return Err(AppError::Update {
            message: format!(
                "attestation schema_version {} not in supported set {:?}",
                prose.schema_version,
                trust_root::SUPPORTED_ATTESTATION_SCHEMA_VERSIONS,
            ),
        });
    }

    // Attestant pubkey fingerprint must match what we observed in the
    // signature — defends against an attestation file that names a
    // different attestant than the actual signer.
    if !prose
        .attestant
        .key_fingerprint_sha256
        .eq_ignore_ascii_case(fingerprint_hex)
    {
        return Err(AppError::Update {
            message: format!(
                "attestation says key_fingerprint_sha256=`{}` but the actual signing key has \
                 fingerprint `{}`",
                prose.attestant.key_fingerprint_sha256, fingerprint_hex
            ),
        });
    }

    // Top-level binding: attestation must pertain to *this* release.
    if prose.release_version != release.version {
        return Err(AppError::Update {
            message: format!(
                "attestation release_version `{}` ≠ release.version `{}`",
                prose.release_version, release.version
            ),
        });
    }
    if !prose.git_commit.eq_ignore_ascii_case(&release.git_commit) {
        return Err(AppError::Update {
            message: format!(
                "attestation git_commit `{}` ≠ release.git_commit `{}`",
                prose.git_commit, release.git_commit
            ),
        });
    }
    match (
        prose.previous_release_git_commit.as_deref(),
        release.previous_release.as_ref(),
    ) {
        (Some(a), Some(r)) if !a.eq_ignore_ascii_case(&r.git_commit) => {
            return Err(AppError::Update {
                message: format!(
                    "attestation previous_release_git_commit `{a}` ≠ release.previous_release.git_commit `{}`",
                    r.git_commit
                ),
            });
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err(AppError::Update {
                message: "attestation and release disagree on whether a `previous_release` exists"
                    .into(),
            });
        }
        _ => {}
    }

    // Load pinned templates from build-time-embedded JSON, render each,
    // and require character-exact match with the signed statement.
    let templates = eidola_attestation::load_from_str(trust_root::ATTESTATION_TEMPLATES_JSON)
        .map_err(|e| AppError::Update {
            message: format!("loading pinned attestation templates: {e}"),
        })?;

    let release_json: serde_json::Value = serde_json::to_value(SerializableRelease::from(release))
        .map_err(|e| AppError::Update {
            message: format!("serializing ReleaseIndex for cross-check roots: {e}"),
        })?;
    let mut roots: std::collections::BTreeMap<&str, &serde_json::Value> =
        std::collections::BTreeMap::new();
    roots.insert("attestation", &attestation);
    roots.insert("release", &release_json);

    // attestant_statement preamble.
    let (rendered_preamble, _) = eidola_attestation::render(
        &templates.attestant_statement_template.template,
        &templates.attestant_statement_template.sources,
        &roots,
    )
    .map_err(|e| AppError::Update {
        message: format!("rendering attestant_statement_template: {e}"),
    })?;
    if rendered_preamble != prose.attestant_statement {
        return Err(AppError::Update {
            message: "signed attestant_statement does not equal the rendered template — \
                 either the templates have drifted or the attestation prose was tampered with"
                .into(),
        });
    }

    // Each claim.
    let mut verified_claims: Vec<VerifiedClaim> = Vec::with_capacity(templates.claims.len());
    for (claim_id, claim_template) in &templates.claims {
        let signed_claim = prose.claims.get(claim_id).ok_or_else(|| AppError::Update {
            message: format!(
                "attestation missing required claim `{claim_id}` (pinned by the template manifest)"
            ),
        })?;

        let (rendered, values) =
            eidola_attestation::render(&claim_template.template, &claim_template.sources, &roots)
                .map_err(|e| AppError::Update {
                message: format!("rendering claim `{claim_id}`: {e}"),
            })?;

        if rendered != signed_claim.statement {
            return Err(AppError::Update {
                message: format!("signed claim `{claim_id}` does not equal its rendered template"),
            });
        }

        // Cross-checks: for each declared mapping, the resolved
        // substitution value (taken from the *attestation* roots) must
        // also equal what's at the corresponding release path.
        for (placeholder, release_path) in &claim_template.cross_checks {
            let attestation_value = values.get(placeholder).ok_or_else(|| AppError::Update {
                message: format!(
                    "claim `{claim_id}` cross_check refers to placeholder `{placeholder}` \
                         that the template doesn't use"
                ),
            })?;
            let release_value = eidola_attestation::resolve_dotted_path(release_path, &roots)
                .map_err(|e| AppError::Update {
                    message: format!(
                        "claim `{claim_id}` cross_check `{placeholder}` → `{release_path}`: {e}"
                    ),
                })?;
            if *attestation_value != release_value {
                return Err(AppError::Update {
                    message: format!(
                        "claim `{claim_id}` cross_check `{placeholder}`: attestation value `{}` \
                         ≠ release[{release_path}] `{}`",
                        attestation_value, release_value
                    ),
                });
            }
        }

        // If the signed claim carries a `fields` object, every (key, value)
        // must equal what we resolved. Catches a coerced engineer who
        // copies a different `fields` value than the rendered statement.
        if let Some(fields) = &signed_claim.fields {
            for (k, v) in fields {
                let resolved = values.get(k).ok_or_else(|| AppError::Update {
                    message: format!(
                        "claim `{claim_id}` declares field `{k}` that the template doesn't use"
                    ),
                })?;
                if v != resolved {
                    return Err(AppError::Update {
                        message: format!(
                            "claim `{claim_id}` field `{k}` value `{v}` ≠ resolved `{resolved}`"
                        ),
                    });
                }
            }
            // Every substituted placeholder must be present in `fields`.
            for k in values.keys() {
                if !fields.contains_key(k) {
                    return Err(AppError::Update {
                        message: format!(
                            "claim `{claim_id}` is missing `fields.{k}` (the template substitutes it)"
                        ),
                    });
                }
            }
        }

        verified_claims.push(VerifiedClaim {
            claim_id: claim_id.clone(),
            statement: rendered,
        });
    }

    // Sanity: attestation can't carry extra claims the template doesn't
    // know about — that's a coerced-extra-claim attack surface.
    for claim_id in prose.claims.keys() {
        if !templates.claims.contains_key(claim_id) {
            return Err(AppError::Update {
                message: format!(
                    "attestation carries claim `{claim_id}` that the pinned template manifest \
                     does not declare"
                ),
            });
        }
    }

    Ok(VerifiedAttestation {
        attestant_id: prose.attestant.id,
        attestant_name: prose.attestant.name,
        jurisdiction: prose.attestant.jurisdiction,
        fingerprint_hex: fingerprint_hex.to_string(),
        rekor_log_index,
        attested_at: prose.attested_at,
        attestant_statement: prose.attestant_statement,
        claims: verified_claims,
    })
}

// ---------------------------------------------------------------------------
// Attestation prose JSON shape — this struct is the source of truth.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct AttestationProse {
    schema_version: u32,
    release_version: String,
    git_commit: String,
    #[serde(default)]
    previous_release_git_commit: Option<String>,
    attestant: AttestantBlock,
    attested_at: String,
    attestant_statement: String,
    claims: std::collections::BTreeMap<String, SignedClaim>,
}

#[derive(Deserialize)]
struct AttestantBlock {
    id: String,
    name: String,
    key_fingerprint_sha256: String,
    jurisdiction: String,
}

#[derive(Deserialize)]
struct SignedClaim {
    statement: String,
    #[serde(default)]
    fields: Option<std::collections::BTreeMap<String, String>>,
}

/// Side-channel shape used to project a [`ReleaseIndex`] into the JSON
/// roots the template renderer walks. Mirrors the on-disk release.json
/// keys so paths like `release.previous_release.git_commit` resolve.
#[derive(serde::Serialize)]
struct SerializableRelease<'a> {
    version: &'a str,
    git_commit: &'a str,
    git_tag: &'a str,
    released_at: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_release: Option<SerializablePrev<'a>>,
}

#[derive(serde::Serialize)]
struct SerializablePrev<'a> {
    version: &'a str,
    git_commit: &'a str,
}

impl<'a> From<&'a ReleaseIndex> for SerializableRelease<'a> {
    fn from(r: &'a ReleaseIndex) -> Self {
        SerializableRelease {
            version: &r.version,
            git_commit: &r.git_commit,
            git_tag: &r.git_tag,
            released_at: &r.released_at,
            previous_release: r.previous_release.as_ref().map(|p| SerializablePrev {
                version: &p.version,
                git_commit: &p.git_commit,
            }),
        }
    }
}

/// Decode a single `-----BEGIN PUBLIC KEY-----` PEM block to its inner
/// PKIX SubjectPublicKeyInfo DER bytes. Tolerant of leading / trailing
/// whitespace; rejects any other PEM type or content outside the
/// expected markers.
fn pem_public_key_to_spki_der(pem_bytes: &[u8]) -> Result<Vec<u8>, AppError> {
    use base64::Engine;
    let pem = std::str::from_utf8(pem_bytes).map_err(|e| AppError::Update {
        message: format!("attestation rekor publicKey.content is not UTF-8 PEM: {e}"),
    })?;
    let trimmed = pem.trim();
    let body = trimmed
        .strip_prefix("-----BEGIN PUBLIC KEY-----")
        .and_then(|s| s.strip_suffix("-----END PUBLIC KEY-----"))
        .ok_or_else(|| AppError::Update {
            message:
                "attestation rekor publicKey.content missing `-----BEGIN PUBLIC KEY-----` markers \
                 — expected PEM PKIX SubjectPublicKeyInfo (cosign sign-blob output), not a cert \
                 or other key format"
                    .into(),
        })?
        .trim();
    let stripped: String = body.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(stripped.as_bytes())
        .map_err(|e| AppError::Update {
            message: format!("base64-decoding PEM PUBLIC KEY body: {e}"),
        })
}

// ---------------------------------------------------------------------------
// `hashedrekord` v0.0.1 body shape (same kind as the CI side, with the
// difference that `publicKey.content` is a PEM `-----BEGIN PUBLIC KEY-----`
// SPKI block here, vs. a PEM Fulcio certificate on the CI side).
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct HashedRekordBody {
    kind: String,
    #[serde(rename = "apiVersion")]
    api_version: String,
    spec: HashedRekordSpec,
}

#[derive(Deserialize)]
struct HashedRekordSpec {
    data: SpecData,
    signature: SpecSignature,
}

#[derive(Deserialize)]
struct SpecData {
    hash: SpecHash,
}

#[derive(Deserialize)]
struct SpecHash {
    algorithm: String,
    value: String,
}

#[derive(Deserialize)]
struct SpecSignature {
    /// Base64 of the raw signature bytes (ECDSA DER for ECDSA-P256/P384,
    /// 64 bytes for Ed25519). Cross-checked against the bundle's
    /// `messageSignature.signature` so a malicious bundle can't point at
    /// a tlog entry for a different signing event.
    content: String,
    #[serde(rename = "publicKey")]
    public_key: SpecPublicKey,
}

#[derive(Deserialize)]
struct SpecPublicKey {
    content: String,
}

// ---------------------------------------------------------------------------
// Encoders / decoders
// ---------------------------------------------------------------------------

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
    fn pem_public_key_rejects_certificate_block() {
        // Rejects a PEM CERTIFICATE block (which the CI side uses) —
        // human attestations carry a `PUBLIC KEY` SPKI block.
        let pem = b"-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n";
        assert!(pem_public_key_to_spki_der(pem).is_err());
    }

    #[test]
    fn pem_public_key_decodes_real_p256_spki() {
        // A real ECDSA-P-256 PEM SPKI taken from a cosign generate-key-pair
        // output. Confirms the parser produces 91-byte DER (typical SPKI
        // length for P-256).
        let pem = b"-----BEGIN PUBLIC KEY-----\n\
                    MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEJsEXjtQe9u/kRQ006UUEXIt4aY7u\n\
                    JI4fqwrk1qBM9GyGPqZYJrflz/dWImo3wdF17ZG3kmfSe/rCiQKL3x/unQ==\n\
                    -----END PUBLIC KEY-----\n";
        let der = pem_public_key_to_spki_der(pem).unwrap();
        assert_eq!(
            der.len(),
            91,
            "P-256 SPKI DER is 91 bytes; got {}",
            der.len()
        );
    }

    #[test]
    fn bundle_rejects_wrong_media_type() {
        let bundle = br#"{
            "mediaType": "application/json",
            "verificationMaterial": {
                "publicKey": { "hint": "deadbeef" },
                "tlogEntries": []
            },
            "messageSignature": {
                "messageDigest": { "algorithm": "SHA2_256", "digest": "AAA=" },
                "signature": "AAA="
            }
        }"#;
        let trust = synthesize_empty_trust();
        let release = synthesize_release();
        let err = verify_human_attestation(b"x", bundle, &release, &trust).unwrap_err();
        assert!(format!("{err}").contains("mediaType"));
    }

    #[test]
    fn bundle_rejects_wrong_log_entry_count() {
        let bundle = br#"{
            "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
            "verificationMaterial": {
                "publicKey": { "hint": "deadbeef" },
                "tlogEntries": []
            },
            "messageSignature": {
                "messageDigest": { "algorithm": "SHA2_256", "digest": "AAA=" },
                "signature": "AAA="
            }
        }"#;
        let trust = synthesize_empty_trust();
        let release = synthesize_release();
        let err = verify_human_attestation(b"x", bundle, &release, &trust).unwrap_err();
        assert!(format!("{err}").contains("tlog entries"));
    }

    #[test]
    fn bundle_rejects_certificate_arm_for_humans() {
        // Sanity: CI-shaped bundles must be rejected by the human verifier.
        let bundle = br#"{
            "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
            "verificationMaterial": {
                "certificate": { "rawBytes": "AAA=" },
                "tlogEntries": [{
                    "logIndex": "1",
                    "logId": { "keyId": "AAA=" },
                    "kindVersion": { "kind": "rekord", "version": "0.0.1" },
                    "integratedTime": "1",
                    "canonicalizedBody": "AAA="
                }]
            },
            "messageSignature": {
                "messageDigest": { "algorithm": "SHA2_256", "digest": "AAA=" },
                "signature": "AAA="
            }
        }"#;
        let trust = synthesize_empty_trust();
        let release = synthesize_release();
        let err = verify_human_attestation(b"x", bundle, &release, &trust).unwrap_err();
        assert!(format!("{err}").contains("publicKey"), "got: {err}");
    }

    fn synthesize_empty_trust() -> TrustedRoot {
        TrustedRoot {
            fulcio_cas: vec![],
            rekor_keys: vec![],
            ctlog_keys: vec![],
        }
    }

    fn synthesize_release() -> ReleaseIndex {
        serde_json::from_str(
            r#"{
                "schema_version": 1,
                "version": "0.1.0",
                "git_commit": "9c3a000000000000000000000000000000000001",
                "git_tag": "v0.1.0",
                "released_at": "2026-05-26T00:00:00Z",
                "artifact_manifest": {
                    "url": "https://example/m.json",
                    "sigstore_bundle_url": "https://example/m.json.sigstore"
                },
                "human_attestations": []
            }"#,
        )
        .unwrap()
    }
}
