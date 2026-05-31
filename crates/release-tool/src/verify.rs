//! `release-tool verify <tag>` — fetch and verify the CI-built artifact
//! manifest, then show the diff against the prior release for human review.
//!
//! Sigstore verification goes through `eidola-app-core`'s pure-Rust
//! verifier — the same code path that ships to users. Keeping a single
//! implementation eliminates the "two versions of the same check drift
//! apart" risk that the previous `cosign verify-blob` shell-out carried.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

pub struct Args {
    pub workspace_root: PathBuf,
    pub repo: String,
    pub tag: String,
}

pub fn run(args: Args) -> Result<()> {
    require_tool("gh")?;
    require_tool("git")?;

    // Resolve the tag to its commit SHA up front. Both the displayed diff
    // and the `release-attest` step act on this SHA, not on the tag name —
    // so a reviewer can always re-run the diff later from the SHAs printed
    // here and get the exact same bytes they signed off on. Using
    // `<tag>^{commit}` instead of bare `<tag>` so an annotated signed tag
    // also resolves to its underlying commit, not the tag object.
    let tag_commit = resolve_to_commit(&args.workspace_root, &args.tag)?;
    let prev = previous_release_tag(&args.workspace_root, &args.tag)
        .ok()
        .map(|prev_tag| -> Result<(String, String)> {
            let prev_commit = resolve_to_commit(&args.workspace_root, &prev_tag)?;
            Ok((prev_tag, prev_commit))
        })
        .transpose()?;

    let tmp = tempfile::tempdir().context("creating tempdir")?;
    let manifest_path = tmp.path().join("artifact-manifest.json");
    let bundle_path = tmp.path().join("artifact-manifest.json.sigstore");

    println!("== fetching release assets from GitHub ==");
    download_asset(
        &args.repo,
        &args.tag,
        "artifact-manifest.json",
        &manifest_path,
    )?;
    download_asset(
        &args.repo,
        &args.tag,
        "artifact-manifest.json.sigstore",
        &bundle_path,
    )?;

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
    let committed_path = args.workspace_root.join("artifact-manifest.json");
    let ci_canonical = canonical_json(&manifest_path)?;
    let committed_canonical = canonical_json(&committed_path)?;
    if ci_canonical != committed_canonical {
        eprintln!(
            "  ✗ committed `artifact-manifest.json` differs from the CI-built one!\n\
             this means either:\n\
               (a) you forgot to run `just update-manifest` before pushing the tag, or\n\
               (b) the build is not reproducible on your hardware vs CI.\n\
             abort, fix, retag, and re-run.\n\
             committed: {}\n\
             ci:        {}",
            committed_path.display(),
            manifest_path.display()
        );
        bail!("manifest mismatch");
    }
    println!("  ✓ committed manifest matches CI (reproducible)");

    if let Some((prev_tag, prev_commit)) = prev.as_ref() {
        println!();
        println!("== diff vs previous release ==");
        println!("  previous: {prev_tag}  →  {prev_commit}");
        println!("  this:     {tag}  →  {tag_commit}", tag = args.tag);
        println!();
        println!(
            "These commits are what `release-attest` will record verbatim in the\n\
             signed attestation; the diff below is between them."
        );
        println!();
        show_git_diff(&args.workspace_root, prev_commit, &tag_commit)?;
    } else {
        println!();
        println!("(no previous release tag found — skipping diff)");
        println!("  this: {tag}  →  {tag_commit}", tag = args.tag);
    }

    println!();
    println!("Verification complete. If you have reviewed the diff and are ready to attest,");
    println!("run: `just release-attest {}`", args.tag);
    Ok(())
}

fn require_tool(name: &str) -> Result<()> {
    let status = Command::new(name)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    match status {
        Ok(s) if s.success() => Ok(()),
        _ => bail!("required tool `{name}` not found on PATH"),
    }
}

#[derive(Deserialize)]
struct ReleaseAssets {
    assets: Vec<ReleaseAsset>,
}

#[derive(Deserialize)]
struct ReleaseAsset {
    name: String,
}

fn download_asset(repo: &str, tag: &str, asset_name: &str, dest: &std::path::Path) -> Result<()> {
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

fn canonical_json(path: &std::path::Path) -> Result<String> {
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

/// Resolve `refname` to the 40-char SHA of the commit it points at.
/// `<refname>^{commit}` peels through annotated tag objects so a signed
/// annotated tag resolves to its underlying commit rather than the tag
/// object SHA. For lightweight tags it's a no-op.
fn resolve_to_commit(workspace_root: &std::path::Path, refname: &str) -> Result<String> {
    let out = Command::new("git")
        .current_dir(workspace_root)
        .args(["rev-parse", &format!("{refname}^{{commit}}")])
        .output()
        .context("running `git rev-parse <ref>^{commit}`")?;
    if !out.status.success() {
        bail!(
            "`git rev-parse {refname}^{{commit}}` failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8(out.stdout)?.trim().to_string())
}

fn previous_release_tag(workspace_root: &std::path::Path, tag: &str) -> Result<String> {
    let out = Command::new("git")
        .current_dir(workspace_root)
        .args(["describe", "--tags", "--abbrev=0", &format!("{tag}^")])
        .output()
        .context("running `git describe --tags --abbrev=0 <tag>^`")?;
    if !out.status.success() {
        bail!(
            "no previous tag reachable from `{tag}` ({})",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8(out.stdout)?.trim().to_string())
}

fn show_git_diff(workspace_root: &std::path::Path, from: &str, to: &str) -> Result<()> {
    // Inherit the engineer's terminal so their pager (less, delta, …) works.
    let status = Command::new("git")
        .current_dir(workspace_root)
        .args(["diff", "--stat", &format!("{from}..{to}")])
        .status()
        .context("running `git diff --stat`")?;
    if !status.success() {
        bail!("git diff failed");
    }
    println!();
    println!(
        "To inspect specific files, run:  git diff {from}..{to} -- <path>\n\
         To inspect everything, run:      git diff {from}..{to}"
    );
    Ok(())
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
