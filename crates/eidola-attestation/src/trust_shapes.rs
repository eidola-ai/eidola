//! Plain `serde` data shapes for the on-disk JSON documents the release /
//! trust system understands. Lives here so the release-tool (signing side)
//! and the client updater (verifier side) deserialize from the same struct
//! definitions and can't drift apart.
//!
//! - [`TrustConstants`] ↔ `releases/trust/trust-constants.json`. Read by
//!   `release-tool` at runtime; the `eidola-app-core` build script reads
//!   the same file at build time and bakes the values into the embedded
//!   trust root, so values shared between the two paths must round-trip
//!   identically.
//! - [`ReleaseIndex`] (+ sub-types) ↔ `release.json` (one per release,
//!   uploaded to the GitHub release). Pure URL index — *no* policy fields
//!   and no hashes. Threshold and identity policy live in the verifier's
//!   embedded trust root, never here, because an attacker who controls
//!   `release.json` must not be able to lower the bar (see
//!   `docs/trust-root.md`); by the same argument the artifact index it
//!   grew at schema 2 says only *where* bytes live, while *what they must
//!   hash to* comes from the CI-signed manifest and the human
//!   attestation.
//!
//! These are intentionally data-only: no methods, no business logic, just
//! `Serialize`/`Deserialize`. Behaviour belongs in the consuming crates.
//!
//! Every shape carries `#[serde(deny_unknown_fields)]`: release documents
//! must not grow fields without a `schema_version` bump, and a document
//! with fields the verifier doesn't understand is rejected rather than
//! silently accepted (docs/privacy-guarantees.md §6.8).

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// trust-constants.json
// ---------------------------------------------------------------------------

/// Top-level shape of `releases/trust/trust-constants.json`. The
/// release-tool reads this from disk; `eidola-app-core/build.rs` reads
/// it at build time and bakes selected fields into the embedded trust
/// root (see `eidola_app_core::trust_root`).
///
/// Fields that one consumer reads but the other doesn't carry
/// `#[allow(dead_code)]` so each consumer compiles cleanly without
/// having to fork the struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustConstants {
    pub schema_version: u32,
    pub trusted_attestant_fingerprints: Vec<String>,
    /// Minimum number of independently-verified human attestations a
    /// release must carry. Lives here (and in the verifier's embedded
    /// trust root) rather than in `release.json` so a forged release
    /// index can't lower the threshold.
    ///
    /// `release-tool` doesn't enforce this — it's strictly a verifier
    /// knob — but the shape is shared, so the field is present on both
    /// sides.
    #[allow(dead_code)]
    pub min_human_attestations: u32,
    // CI identity + issuer pinning lives in the embedded sigstore verifier
    // (`eidola-app-core::trust_root`); the verifier consults those
    // constants directly. Kept here for schema parity with the on-disk
    // `trust-constants.json`.
    #[allow(dead_code)]
    pub expected_ci_identity_pattern: String,
    #[allow(dead_code)]
    pub expected_ci_issuer: String,
    #[allow(dead_code)]
    pub supported_release_schema_versions: Vec<u32>,
    #[allow(dead_code)]
    pub supported_attestation_schema_versions: Vec<u32>,
    #[allow(dead_code)]
    pub update_discovery_url: String,
    #[allow(dead_code)]
    pub server_url_template: String,
    #[allow(dead_code)]
    pub server_url_hash_length: u32,
}

// ---------------------------------------------------------------------------
// release.json
// ---------------------------------------------------------------------------

/// Top-level shape of `release.json` — a pure URL index pointing at the
/// signed artifacts the verifier downloads. The verifier deserializes
/// this from the bytes fetched at `UPDATE_DISCOVERY_URL`'s `release.json`
/// asset; the release-tool serializes the same shape when generating the
/// file.
///
/// Policy (threshold, allowed identities, allowed schema versions) is
/// **deliberately absent**: those live in the verifier's embedded trust
/// root so a forged index can't downgrade them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseIndex {
    pub schema_version: u32,
    pub version: String,
    pub git_commit: String,
    pub git_tag: String,
    pub released_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_release: Option<PreviousRelease>,
    pub artifact_manifest: ArtifactManifestRef,
    pub human_attestations: Vec<HumanAttestationRef>,
    /// Where to download each artifact `artifact-manifest.json` names,
    /// keyed by the same artifact key. Present from schema 2.
    ///
    /// URLs only, and deliberately no hashes: this document is unsigned,
    /// so a hash recorded here would be an attacker-suppliable value the
    /// installer might compare against itself. The hash of an artifact
    /// comes from the CI-signed manifest, and the hash of the detached
    /// Apple material from the human attestation — the same reason
    /// threshold and identity policy are absent above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<std::collections::BTreeMap<String, ArtifactRef>>,
    /// Where to download the detached Apple signature material for this
    /// release, when one was published. Present from schema 2, and
    /// optional even then: a release with no signed macOS installable has
    /// none, and it is not a manifest artifact — its bytes depend on a
    /// key, which is exactly what the manifest may never record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apple_signature_bundle: Option<ArtifactRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreviousRelease {
    pub version: String,
    pub git_commit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifestRef {
    pub url: String,
    pub sigstore_bundle_url: String,
}

/// One downloadable file. A URL and nothing else — see [`ReleaseIndex`]'s
/// `artifacts` field for why no hash lives here.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRef {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HumanAttestationRef {
    pub attestant_id: String,
    pub url: String,
    pub bundle_url: String,
}
