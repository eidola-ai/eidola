//! Verify a human-signed release attestation.
//!
//! Engineers sign the attestation JSON file with their hardware-backed
//! SSH key via `ssh-keygen -Y sign` (namespace `eidola-attestation@v1`),
//! then post the resulting signature to Sigstore Rekor as a
//! `hashedrekord` entry. The release-tool saves Rekor's response as the
//! `.bundle.json` companion file. This module reads that bundle, walks
//! every cryptographic binding, and returns the verified facts.
//!
//! Pipeline (one attestation):
//!
//! 1. **Bundle parse.** Extract the rekor log entry from
//!    `{schema_version, rekor_log_entry: {<uuid>: {...}}}`.
//! 2. **Body parse.** Decode the entry's `body` (base64) into the
//!    canonical hashedrekord JSON; require `kind=hashedrekord`,
//!    `apiVersion=0.0.1`, `data.hash.algorithm=sha256`.
//! 3. **Body binding.** The body's `data.hash.value` must equal
//!    `sha256(attestation_bytes)` in hex.
//! 4. **Pubkey extraction + fingerprint check.** Decode
//!    `body.signature.publicKey.content` (base64) into the OpenSSH
//!    pubkey line, parse it with `ssh-key`, compute `sha256(wire-format
//!    pubkey bytes)`, assert the hex is in
//!    [`trust_root::TRUSTED_ATTESTANT_FINGERPRINTS`].
//! 5. **SSH signature verify.** Decode `body.signature.content`
//!    (base64) into the PEM-wrapped SSH signature blob, parse with
//!    `ssh-key`, verify against `attestation_bytes` with the namespace
//!    `eidola-attestation@v1`.
//! 6. **SET verify.** Reconstruct the canonical JSON
//!    `{"body":"<b64>","integratedTime":N,"logIndex":N,"logID":"<hex>"}`,
//!    select the pinned Rekor key by `logID`, ECDSA-P256 or Ed25519
//!    verify the SET signature against it.
//! 7. **Merkle inclusion proof.** RFC 6962 walk via
//!    [`super::merkle`]; recomputed root must equal `rootHash`.
//!
//! See also the known gap in `releases/TRUST-ROOT.md` on Rekor
//! checkpoint signature verification (same caveat as the CI side).

use serde::Deserialize;
use sha2::{Digest, Sha256};
use signature::hazmat::PrehashVerifier;

use crate::error::AppError;
use crate::trust_root;

use super::merkle;
use super::trust::{KeyDetails, RekorKey, TrustedRoot};
use super::{ReleaseIndex, VerifiedAttestation, VerifiedClaim};

pub const SSH_SIG_NAMESPACE: &str = "eidola-attestation@v1";

/// Verify a single human attestation end-to-end: the SSH signature + the
/// Rekor inclusion (this file), the structural cross-checks against the
/// release index, and the character-exact equality of every signed claim
/// against its pinned template rendering. Returns a fully-populated
/// [`VerifiedAttestation`] on success.
pub fn verify_human_attestation(
    attestation_bytes: &[u8],
    bundle_bytes: &[u8],
    release: &ReleaseIndex,
    trust: &TrustedRoot,
) -> Result<VerifiedAttestation, AppError> {
    let attestation_sha256: [u8; 32] = Sha256::digest(attestation_bytes).into();

    // ── 1. Bundle parse ──────────────────────────────────────────────
    let bundle: AttestationBundle =
        serde_json::from_slice(bundle_bytes).map_err(|e| AppError::Update {
            message: format!("parsing attestation bundle JSON: {e}"),
        })?;
    if bundle.schema_version != 1 {
        return Err(AppError::Update {
            message: format!(
                "attestation bundle schema_version {} not supported (expected 1)",
                bundle.schema_version
            ),
        });
    }
    if bundle.rekor_log_entry.len() != 1 {
        return Err(AppError::Update {
            message: format!(
                "attestation bundle has {} rekor log entries, expected exactly 1",
                bundle.rekor_log_entry.len()
            ),
        });
    }
    let entry = bundle.rekor_log_entry.into_values().next().unwrap();

    // ── 2. Body parse ────────────────────────────────────────────────
    let body_bytes = base64_std_decode(&entry.body, "rekor_log_entry.body")?;
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

    // ── 3. Body binding to attestation bytes ─────────────────────────
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

    // ── 4. Pubkey + fingerprint ──────────────────────────────────────
    let pubkey_pem_bytes = base64_std_decode(
        &body.spec.signature.public_key.content,
        "attestation rekor publicKey.content",
    )?;
    let pubkey_text = std::str::from_utf8(&pubkey_pem_bytes).map_err(|e| AppError::Update {
        message: format!("attestation rekor publicKey.content is not UTF-8: {e}"),
    })?;
    let pubkey = parse_openssh_pubkey(pubkey_text)?;
    let fingerprint_hex = ssh_pubkey_fingerprint_hex(&pubkey);
    if !trust_root::TRUSTED_ATTESTANT_FINGERPRINTS
        .iter()
        .any(|fp| fp.eq_ignore_ascii_case(&fingerprint_hex))
    {
        return Err(AppError::Update {
            message: format!(
                "attestation was signed by SSH key with fingerprint `{fingerprint_hex}`, which is \
                 NOT in this client's TRUSTED_ATTESTANT_FINGERPRINTS — \
                 the signer is not authorized for this release line"
            ),
        });
    }

    // ── 5. SSH signature ─────────────────────────────────────────────
    let sig_pem_bytes = base64_std_decode(
        &body.spec.signature.content,
        "attestation rekor signature.content",
    )?;
    let sig_pem = std::str::from_utf8(&sig_pem_bytes).map_err(|e| AppError::Update {
        message: format!("attestation rekor signature.content is not UTF-8 PEM: {e}"),
    })?;
    let ssh_sig = ssh_key::SshSig::from_pem(sig_pem.as_bytes()).map_err(|e| AppError::Update {
        message: format!("parsing SSH signature PEM: {e}"),
    })?;
    // Sanity: signature must carry our namespace.
    if ssh_sig.namespace() != SSH_SIG_NAMESPACE {
        return Err(AppError::Update {
            message: format!(
                "SSH signature namespace is `{}`, expected `{}` — this signature was made for \
                 a different protocol context",
                ssh_sig.namespace(),
                SSH_SIG_NAMESPACE
            ),
        });
    }
    pubkey
        .verify(SSH_SIG_NAMESPACE, attestation_bytes, &ssh_sig)
        .map_err(|e| AppError::Update {
            message: format!("SSH signature failed cryptographic verify: {e}"),
        })?;

    // ── 6. SET signature ─────────────────────────────────────────────
    let log_id_bytes = hex_decode(&entry.log_id).map_err(|e| AppError::Update {
        message: format!("hex-decoding rekor logID: {e}"),
    })?;
    let log_id: [u8; 32] = log_id_bytes
        .as_slice()
        .try_into()
        .map_err(|_| AppError::Update {
            message: format!(
                "rekor logID is {} bytes, expected 32 (sha256)",
                log_id_bytes.len()
            ),
        })?;
    let rekor_key = trust
        .rekor_keys
        .iter()
        .find(|k| k.log_id == log_id)
        .ok_or_else(|| AppError::Update {
            message: format!(
                "no pinned Rekor key matches the bundle's logID `{}` — \
                 trusted root is stale, or this bundle is from a different Rekor instance",
                entry.log_id
            ),
        })?;
    let set_bytes = base64_std_decode(
        &entry.verification.signed_entry_timestamp,
        "verification.signedEntryTimestamp",
    )?;
    // Canonical (RFC 8785) JSON for our known field set. Keys must be in
    // lexicographic order: body < integratedTime < logID < logIndex.
    let signed_payload = format!(
        r#"{{"body":"{body}","integratedTime":{it},"logID":"{lid}","logIndex":{li}}}"#,
        body = entry.body,
        it = entry.integrated_time,
        lid = entry.log_id,
        li = entry.log_index,
    );
    verify_rekor_set(rekor_key, signed_payload.as_bytes(), &set_bytes)?;

    // ── 7. Merkle inclusion proof ────────────────────────────────────
    let root_hash_bytes =
        hex_decode(&entry.verification.inclusion_proof.root_hash).map_err(|e| {
            AppError::Update {
                message: format!("hex-decoding inclusionProof.rootHash: {e}"),
            }
        })?;
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
    let proof_hashes: Vec<[u8; 32]> = entry
        .verification
        .inclusion_proof
        .hashes
        .iter()
        .map(|h| {
            let decoded = hex_decode(h).map_err(|e| AppError::Update {
                message: format!("hex-decoding inclusionProof.hashes[]: {e}"),
            })?;
            decoded.as_slice().try_into().map_err(|_| AppError::Update {
                message: format!(
                    "inclusionProof.hashes[] entry is {} bytes, expected 32",
                    decoded.len()
                ),
            })
        })
        .collect::<Result<_, AppError>>()?;
    let leaf_hash = merkle::hash_leaf(&body_bytes);
    merkle::verify_inclusion_proof(
        entry.verification.inclusion_proof.log_index,
        &leaf_hash,
        entry.verification.inclusion_proof.tree_size,
        &proof_hashes,
        &root_hash,
    )?;

    // ── 8. Content verification (template equality + cross-checks) ───
    verify_attestation_content(
        attestation_bytes,
        release,
        &fingerprint_hex,
        entry.log_index,
    )
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
// Attestation prose JSON shape (matches releases/schema/attestation-v1.schema.json)
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

/// Parse an OpenSSH `.pub`-style line (`ssh-<type> <base64> [comment]`)
/// into a `ssh_key::PublicKey`.
fn parse_openssh_pubkey(text: &str) -> Result<ssh_key::PublicKey, AppError> {
    let line = text.lines().next().unwrap_or("").trim();
    ssh_key::PublicKey::from_openssh(line).map_err(|e| AppError::Update {
        message: format!("parsing OpenSSH pubkey line `{line}`: {e}"),
    })
}

/// Compute the standard SSH SHA-256 fingerprint of `pubkey` as
/// lowercase hex. `ssh-key`'s `fingerprint()` performs `sha256` over the
/// OpenSSH wire-format pubkey bytes — the same algorithm
/// `ssh-keygen -E sha256 -l` uses, which matches our trust-constants
/// pin.
fn ssh_pubkey_fingerprint_hex(pubkey: &ssh_key::PublicKey) -> String {
    let fp = pubkey.fingerprint(ssh_key::HashAlg::Sha256);
    sha256_hex_of(fp.as_bytes())
}

fn sha256_hex_of(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(out, "{b:02x}").unwrap();
    }
    out
}

fn verify_rekor_set(key: &RekorKey, message: &[u8], signature: &[u8]) -> Result<(), AppError> {
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
// Bundle JSON shape (matches what `release-tool attest` writes; the inner
// `rekor_log_entry` is verbatim Rekor v1 API response).
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct AttestationBundle {
    schema_version: u32,
    rekor_log_entry: std::collections::BTreeMap<String, RekorEntry>,
}

#[derive(Deserialize)]
struct RekorEntry {
    body: String,
    #[serde(rename = "integratedTime")]
    integrated_time: i64,
    #[serde(rename = "logID")]
    log_id: String,
    #[serde(rename = "logIndex")]
    log_index: u64,
    verification: RekorVerification,
}

#[derive(Deserialize)]
struct RekorVerification {
    #[serde(rename = "inclusionProof")]
    inclusion_proof: RekorInclusionProof,
    #[serde(rename = "signedEntryTimestamp")]
    signed_entry_timestamp: String,
}

#[derive(Deserialize)]
struct RekorInclusionProof {
    #[serde(rename = "logIndex")]
    log_index: u64,
    #[serde(rename = "rootHash")]
    root_hash: String,
    #[serde(rename = "treeSize")]
    tree_size: u64,
    #[serde(default)]
    hashes: Vec<String>,
    #[allow(dead_code)]
    #[serde(default)]
    checkpoint: Option<String>,
}

// ---------------------------------------------------------------------------
// `hashedrekord` v0.0.1 body shape (same as the CI side).
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

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        write!(out, "{b:02x}").unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssh_fingerprint_matches_shell_pipeline() {
        // Tests that ssh_pubkey_fingerprint_hex matches what
        //   `awk '{print $2}' pub.key | base64 -d | shasum -a 256`
        // produces (the algorithm we documented in TRUST-ROOT.md). For a
        // fixed Ed25519 pubkey line, both should yield the same hex.
        let pubkey_line = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBzlNbqOgZsQuvOSnk6QklRfL/x6AYHpsLwQy7c6KhM/ test@example";
        let pk = parse_openssh_pubkey(pubkey_line).unwrap();
        let computed = ssh_pubkey_fingerprint_hex(&pk);

        // Independent computation: base64-decode the wire field, sha256, hex.
        use base64::Engine;
        let parts: Vec<&str> = pubkey_line.split_whitespace().collect();
        let wire = base64::engine::general_purpose::STANDARD
            .decode(parts[1])
            .unwrap();
        let shell_pipeline = sha256_hex(&wire);

        assert_eq!(
            computed, shell_pipeline,
            "ssh-key's fingerprint(Sha256) must equal sha256(wire-format pubkey)"
        );
    }

    #[test]
    fn parse_openssh_pubkey_rejects_malformed() {
        assert!(parse_openssh_pubkey("not even close").is_err());
        assert!(parse_openssh_pubkey("").is_err());
    }

    #[test]
    fn bundle_rejects_wrong_schema_version() {
        let bundle = br#"{"schema_version": 99, "rekor_log_entry": {}}"#;
        let trust = synthesize_empty_trust();
        let release = synthesize_release();
        let err = verify_human_attestation(b"x", bundle, &release, &trust).unwrap_err();
        assert!(format!("{err}").contains("schema_version"));
    }

    #[test]
    fn bundle_rejects_wrong_log_entry_count() {
        let bundle = br#"{"schema_version": 1, "rekor_log_entry": {}}"#;
        let trust = synthesize_empty_trust();
        let release = synthesize_release();
        let err = verify_human_attestation(b"x", bundle, &release, &trust).unwrap_err();
        assert!(format!("{err}").contains("rekor log entries"));
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
                "human_attestations": [],
                "policy": { "min_human_attestations": 1 }
            }"#,
        )
        .unwrap()
    }
}
