//! Releases described by test evidence, for the self-test and development
//! bundles only (never a release bundle): no manifest that records a mobile
//! bundle can be CI-signed outside a release, so these take a release index
//! and its manifest **as verified**, and from there run the real path — the
//! bundle reader, the approval, the download and every byte check, the
//! shell's store.

use base64::Engine as _;
use eidola_app_core::updater::{ReleaseEvidence, ReleaseIndex, ReleaseSummary};
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// A release index and its manifest, taken as verified.
#[derive(Debug, Deserialize)]
pub struct StagedRelease {
    pub index: ReleaseIndex,
    /// The manifest's bytes, base64.
    pub manifest: String,
}

/// A staged release, as `verify_release` would have returned it (with no
/// attestations: nobody attested a test release).
pub fn staged_evidence(release: &StagedRelease) -> Result<ReleaseEvidence, String> {
    let manifest_bytes = base64::engine::general_purpose::STANDARD
        .decode(&release.manifest)
        .map_err(|e| e.to_string())?;
    let index = &release.index;
    Ok(ReleaseEvidence {
        summary: ReleaseSummary {
            version: index.version.clone(),
            git_commit: index.git_commit.clone(),
            git_tag: index.git_tag.clone(),
            released_at: index.released_at.clone(),
            previous_release: index.previous_release.clone(),
            attestations: vec![],
        },
        manifest_sha256: Sha256::digest(&manifest_bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
        manifest_bytes,
        ci_identity: String::new(),
        ci_rekor_log_index: 0,
    })
}
