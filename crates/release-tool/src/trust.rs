//! Read the workspace's `releases/trust/trust-constants.json` at runtime.
//!
//! The tool intentionally reads from disk (not from a build-time embed)
//! because the release engineer's workflow does not version-lock these
//! values: when a trust-root rotation lands, the engineer pulls the new
//! commit and the next `release-tool verify` reads the new constants
//! without needing a fresh `cargo build`.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct TrustConstants {
    pub schema_version: String,
    pub trusted_attestant_fingerprints: Vec<String>,
    pub expected_ci_identity_pattern: String,
    pub expected_ci_issuer: String,
    #[allow(dead_code)]
    pub supported_release_schema_versions: Vec<String>,
    #[allow(dead_code)]
    pub supported_attestation_schema_versions: Vec<String>,
    #[allow(dead_code)]
    pub update_discovery_url: String,
    #[allow(dead_code)]
    pub server_url_template: String,
    #[allow(dead_code)]
    pub server_url_hash_length: u32,
}

pub fn load(workspace_root: &Path) -> Result<TrustConstants> {
    let path = workspace_root.join("releases/trust/trust-constants.json");
    let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let parsed: TrustConstants =
        serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    if parsed.schema_version != "1.0.0" {
        anyhow::bail!(
            "trust-constants.json schema_version `{}` not supported by this release-tool (expected `1.0.0`)",
            parsed.schema_version
        );
    }
    Ok(parsed)
}
