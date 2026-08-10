//! Fetch and validate a release's CI-built `artifact-manifest.json`.
//!
//! One function both `release-tool verify` and `release-tool attest` go
//! through, so the attest path cannot skip the checks the verify path
//! performs: `attest` signs the sha256 of whatever this returns, and the
//! updater rejects the release unless the asset it fetches hashes to that
//! value — so the bytes signed must be the bytes verified, every time,
//! regardless of which subcommand ran first or what changed in between.

use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// Download the tag's `artifact-manifest.json` (+ its Sigstore bundle)
/// from the GitHub release, verify the CI signature, and require byte
/// equality with the committed workspace manifest. Returns the verified
/// asset bytes.
///
/// Byte equality is the required bar, not JSON equivalence: the signed
/// hash is of exact bytes, and clients hash the exact asset they fetch —
/// a formatting-only difference would publish an attestation no client
/// accepts. The canonical comparison below survives only to say *which*
/// kind of divergence a mismatch is.
pub fn fetch_verified_manifest(workspace_root: &Path, repo: &str, tag: &str) -> Result<Vec<u8>> {
    let tmp = tempfile::tempdir().context("creating tempdir")?;
    let manifest_path = tmp.path().join("artifact-manifest.json");
    let bundle_path = tmp.path().join("artifact-manifest.json.sigstore");

    println!("== fetching release assets from GitHub ==");
    download_asset(repo, tag, "artifact-manifest.json", &manifest_path)?;
    download_asset(repo, tag, "artifact-manifest.json.sigstore", &bundle_path)?;

    println!("== verifying Sigstore bundle ==");
    let manifest_bytes =
        fs::read(&manifest_path).with_context(|| format!("reading {}", manifest_path.display()))?;
    let bundle_bytes =
        fs::read(&bundle_path).with_context(|| format!("reading {}", bundle_path.display()))?;
    let trust = eidola_app_core::updater::trust::load()
        .map_err(|e| anyhow::anyhow!("loading sigstore trust root: {e}"))?;
    let verified = eidola_app_core::updater::ci_sigstore::verify_ci_signature(
        &manifest_bytes,
        &bundle_bytes,
        &trust,
    )
    .map_err(|e| anyhow::anyhow!("verifying CI signature: {e}"))?;
    println!("  ✓ CI signature verified");
    println!("      identity: {}", verified.ci_identity);
    println!("      issuer:   {}", verified.ci_issuer);
    println!(
        "      rekor:    https://search.sigstore.dev/?logIndex={}",
        verified.rekor_log_index
    );

    println!("== comparing CI manifest with committed manifest ==");
    let committed_path = workspace_root.join("artifact-manifest.json");
    let committed_bytes = fs::read(&committed_path)
        .with_context(|| format!("reading {}", committed_path.display()))?;
    if committed_bytes != manifest_bytes {
        let kind = if canonical_json(&manifest_path)? == canonical_json(&committed_path)? {
            "the two parse to identical JSON, so the difference is formatting only —\n\
             which still changes the sha256 that `release-attest` signs and clients check"
        } else {
            "the two differ in content, meaning either:\n\
               (a) you forgot to run `just update-manifest` before pushing the tag, or\n\
               (b) the build is not reproducible on your hardware vs CI"
        };
        eprintln!(
            "  ✗ committed `artifact-manifest.json` differs byte-for-byte from the CI-built one!\n\
             {kind}.\n\
             abort, fix, retag, and re-run.\n\
             committed: {}\n\
             ci:        {}",
            committed_path.display(),
            manifest_path.display()
        );
        bail!("manifest mismatch");
    }
    println!("  ✓ committed manifest matches CI byte-for-byte (reproducible)");

    Ok(manifest_bytes)
}

#[derive(Deserialize)]
struct ReleaseAssets {
    assets: Vec<ReleaseAsset>,
}

#[derive(Deserialize)]
struct ReleaseAsset {
    name: String,
}

fn download_asset(repo: &str, tag: &str, asset_name: &str, dest: &Path) -> Result<()> {
    // Preflight: confirm the asset exists, so a 404 surfaces as a clear
    // "asset missing from release" rather than as a confusing
    // `gh release download` error.
    let out = Command::new("gh")
        .args(["release", "view", tag, "--repo", repo, "--json", "assets"])
        .output()
        .context("running `gh release view`")?;
    if !out.status.success() {
        bail!(
            "`gh release view {tag}` failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let assets: ReleaseAssets =
        serde_json::from_slice(&out.stdout).context("parsing gh release view JSON")?;
    if !assets.assets.iter().any(|a| a.name == asset_name) {
        bail!("release `{tag}` has no asset `{asset_name}`");
    }

    // `gh release download` with `--pattern` matching exactly one asset
    // writes it to the location given by `--output`.
    let status = Command::new("gh")
        .args([
            "release",
            "download",
            tag,
            "--repo",
            repo,
            "--pattern",
            asset_name,
            "--output",
            dest.to_str().unwrap(),
            "--clobber",
        ])
        .status()
        .context("running `gh release download`")?;
    if !status.success() {
        bail!("`gh release download {asset_name}` failed");
    }
    Ok(())
}

fn canonical_json(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing {} as JSON", path.display()))?;
    // serde_json::to_string sorts keys when going through a BTreeMap; the
    // simplest cross-platform canonicalization is to re-serialize via a
    // sorted BTreeMap of the parsed value.
    Ok(canonicalize(&value))
}

fn canonicalize(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(map) => {
            let sorted: std::collections::BTreeMap<&String, &serde_json::Value> =
                map.iter().collect();
            let inner: Vec<String> = sorted
                .iter()
                .map(|(k, v)| format!("{}:{}", serde_json::to_string(k).unwrap(), canonicalize(v)))
                .collect();
            format!("{{{}}}", inner.join(","))
        }
        serde_json::Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(canonicalize).collect();
            format!("[{}]", inner.join(","))
        }
        other => serde_json::to_string(other).unwrap(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_is_order_independent() {
        let a = serde_json::from_str::<serde_json::Value>(r#"{"a":1,"b":2}"#).unwrap();
        let b = serde_json::from_str::<serde_json::Value>(r#"{"b":2,"a":1}"#).unwrap();
        assert_eq!(canonicalize(&a), canonicalize(&b));
    }

    #[test]
    fn canonical_json_distinguishes_different_content() {
        let a = serde_json::from_str::<serde_json::Value>(r#"{"a":1}"#).unwrap();
        let b = serde_json::from_str::<serde_json::Value>(r#"{"a":2}"#).unwrap();
        assert_ne!(canonicalize(&a), canonicalize(&b));
    }
}
