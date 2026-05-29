//! Read the workspace's `releases/trust/trust-constants.json` at runtime.
//!
//! The tool intentionally reads from disk (not from a build-time embed)
//! because the release engineer's workflow does not version-lock these
//! values: when a trust-root rotation lands, the engineer pulls the new
//! commit and the next `release-tool verify` reads the new constants
//! without needing a fresh `cargo build`.
//!
//! The shared [`TrustConstants`] struct lives in `eidola-attestation` so
//! the verifier (in `eidola-app-core`'s updater) and this tool agree on
//! the on-disk shape byte-for-byte.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

pub use eidola_attestation::TrustConstants;

pub fn load(workspace_root: &Path) -> Result<TrustConstants> {
    let path = workspace_root.join("releases/trust/trust-constants.json");
    let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let parsed: TrustConstants =
        serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    if parsed.schema_version != 1 {
        anyhow::bail!(
            "trust-constants.json schema_version `{}` not supported by this release-tool (expected `1`)",
            parsed.schema_version
        );
    }
    Ok(parsed)
}
