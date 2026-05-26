//! Self-update verifier — fetches `release.json` from the embedded
//! `UPDATE_DISCOVERY_URL`, validates every signed artifact it references
//! against the embedded trust root, and surfaces the attestation prose to
//! the user for approval before swapping the running binary.
//!
//! ## Pipeline
//!
//! ```text
//!   discover    → fetch release.json from GitHub releases API
//!   schema      → release.json.schema_version in SUPPORTED_RELEASE_SCHEMA_VERSIONS
//!   continuity  → release.version > installed; previous_release.git_commit fast-forwards
//!   fetch       → download artifact-manifest.json, both sigstore-side bundles,
//!                 each attestation JSON, each attestation bundle JSON
//!   verify-ci   → tinfoil-rs sigstore verification of artifact-manifest.json's bundle:
//!                 Fulcio cert chain, identity matches EXPECTED_CI_IDENTITY_PATTERN,
//!                 OIDC issuer matches EXPECTED_CI_ISSUER, rekor inclusion + checkpoint
//!   verify-human→ for each human attestation:
//!                 - parse the SSH signature + pubkey from the bundle's rekor entry body
//!                 - sha256(SSH wire-format pubkey) ∈ TRUSTED_ATTESTANT_FINGERPRINTS
//!                 - verify the SSH signature against the attestation JSON bytes
//!                 - verify rekor SET against rekor pubkey + verify inclusion proof
//!   templates   → render each template from the pinned ATTESTATION_TEMPLATES_JSON;
//!                 require character-for-character match against signed `statement`
//!   cross-check → resolved substitution values match the declared release.x.y paths
//!   policy      → at least min_human_attestations human attestations verified
//!   manifest    → fetch artifact-manifest.json's artifacts; verify each hash
//!   present     → return ReleaseSummary to the UI for user approval
//!   install     → (step 5, deferred) download + swap the new binary
//! ```
//!
//! For step 4a this module is the type skeleton + a stub
//! [`check_for_update`] that returns `Unimplemented`. Subsequent steps
//! fill in the stages above one at a time.

use serde::Deserialize;

use crate::error::AppError;

// ---------------------------------------------------------------------------
// release.json types — match releases/schema/release-v1.0.0.schema.json
// ---------------------------------------------------------------------------

/// Top-level shape of a release index. The verifier deserializes this from
/// the bytes fetched at `UPDATE_DISCOVERY_URL`'s `release.json` asset.
#[derive(Debug, Clone, Deserialize)]
pub struct ReleaseIndex {
    pub schema_version: String,
    pub version: String,
    pub git_commit: String,
    pub git_tag: String,
    pub released_at: String,
    #[serde(default)]
    pub previous_release: Option<PreviousRelease>,
    pub artifact_manifest: ArtifactManifestRef,
    pub human_attestations: Vec<HumanAttestationRef>,
    pub policy: ReleasePolicy,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PreviousRelease {
    pub version: String,
    pub git_commit: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArtifactManifestRef {
    pub url: String,
    pub sigstore_bundle_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HumanAttestationRef {
    pub attestant_id: String,
    pub url: String,
    pub bundle_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReleasePolicy {
    pub min_human_attestations: u32,
}

// ---------------------------------------------------------------------------
// Verifier output — what the UI displays to the user before install
// ---------------------------------------------------------------------------

/// What [`check_for_update`] returns once verification succeeds. The UI
/// shows the user the prose in `attestations` and waits for approval
/// before invoking install.
#[derive(Debug, Clone)]
pub struct ReleaseSummary {
    pub version: String,
    pub git_commit: String,
    pub git_tag: String,
    pub released_at: String,
    pub previous_release: Option<PreviousRelease>,
    pub attestations: Vec<VerifiedAttestation>,
}

/// One verified human attestation, ready to render. The verifier guarantees:
/// every `statement` in `claims` is character-for-character equal to the
/// rendered output of its pinned template.
#[derive(Debug, Clone)]
pub struct VerifiedAttestation {
    pub attestant_id: String,
    pub attestant_name: String,
    pub jurisdiction: String,
    pub attested_at: String,
    pub attestant_statement: String,
    pub claims: Vec<VerifiedClaim>,
}

#[derive(Debug, Clone)]
pub struct VerifiedClaim {
    pub claim_id: String,
    pub statement: String,
}

// ---------------------------------------------------------------------------
// Entry point — stubbed for step 4a
// ---------------------------------------------------------------------------

/// Fetch the latest release index, run the verification pipeline, and
/// return a [`ReleaseSummary`] ready for the UI to display.
///
/// Returns `Ok(None)` if no update is available (the latest release matches
/// the currently-installed version). Returns `Err` if the latest release
/// fails verification for any reason — the UI surfaces the error and the
/// installed version stays put.
///
/// **Step 4a stub** — currently returns
/// [`AppError::Internal`]; subsequent steps wire each pipeline stage.
pub async fn check_for_update(
    _installed_version: &str,
    _installed_git_commit: Option<&str>,
) -> Result<Option<ReleaseSummary>, AppError> {
    Err(AppError::Internal {
        message: "self-update verifier is not yet implemented (step 4 in progress)".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The release.json schema is small enough that we can lock in its
    /// shape via a minimal round-trip test now, before any verifier
    /// machinery is in place. Subsequent steps will add tests for the
    /// pipeline stages they implement.
    #[test]
    fn release_index_deserializes_minimal_form() {
        let json = r#"{
            "schema_version": "1.0.0",
            "version": "0.5.0",
            "git_commit": "9c3a000000000000000000000000000000000001",
            "git_tag": "v0.5.0",
            "released_at": "2026-05-20T17:28:00Z",
            "artifact_manifest": {
                "url": "https://example/artifact-manifest.json",
                "sigstore_bundle_url": "https://example/artifact-manifest.json.sigstore"
            },
            "human_attestations": [{
                "attestant_id": "mike-prince",
                "url": "https://example/attestation-mike-prince.json",
                "bundle_url": "https://example/attestation-mike-prince.bundle.json"
            }],
            "policy": { "min_human_attestations": 1 }
        }"#;
        let parsed: ReleaseIndex = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.schema_version, "1.0.0");
        assert_eq!(parsed.version, "0.5.0");
        assert!(parsed.previous_release.is_none());
        assert_eq!(parsed.human_attestations.len(), 1);
        assert_eq!(parsed.human_attestations[0].attestant_id, "mike-prince");
        assert_eq!(parsed.policy.min_human_attestations, 1);
    }

    #[test]
    fn release_index_with_previous_release() {
        let json = r#"{
            "schema_version": "1.0.0",
            "version": "0.5.0",
            "git_commit": "9c3a000000000000000000000000000000000001",
            "git_tag": "v0.5.0",
            "released_at": "2026-05-20T17:28:00Z",
            "previous_release": {
                "version": "0.4.0",
                "git_commit": "5e1f000000000000000000000000000000000002"
            },
            "artifact_manifest": {
                "url": "https://example/m.json",
                "sigstore_bundle_url": "https://example/m.json.sigstore"
            },
            "human_attestations": [{
                "attestant_id": "mike-prince",
                "url": "https://example/a.json",
                "bundle_url": "https://example/a.bundle.json"
            }],
            "policy": { "min_human_attestations": 1 }
        }"#;
        let parsed: ReleaseIndex = serde_json::from_str(json).unwrap();
        let prev = parsed.previous_release.expect("expected previous_release");
        assert_eq!(prev.version, "0.4.0");
    }

    #[tokio::test]
    async fn check_for_update_returns_unimplemented_stub() {
        let result = check_for_update("0.4.0", None).await;
        assert!(result.is_err());
    }
}
