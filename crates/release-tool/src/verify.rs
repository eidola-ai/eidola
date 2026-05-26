//! `release-tool verify <tag>` — fetch and verify the CI-built artifact
//! manifest, then show the diff against the prior release for human review.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::trust;

pub struct Args {
    pub workspace_root: PathBuf,
    pub repo: String,
    pub tag: String,
}

pub fn run(args: Args) -> Result<()> {
    require_tool("gh")?;
    require_tool("cosign")?;
    require_tool("git")?;

    let trust = trust::load(&args.workspace_root)?;
    let prev_tag = previous_release_tag(&args.workspace_root, &args.tag).ok();

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
    let certificate_identity_regexp =
        identity_regex_from_pattern(&trust.expected_ci_identity_pattern);
    let status = Command::new("cosign")
        .args([
            "verify-blob",
            manifest_path.to_str().unwrap(),
            "--bundle",
            bundle_path.to_str().unwrap(),
            "--certificate-identity-regexp",
            &certificate_identity_regexp,
            "--certificate-oidc-issuer",
            &trust.expected_ci_issuer,
        ])
        .status()
        .context("running `cosign verify-blob`")?;
    if !status.success() {
        bail!(
            "cosign verify-blob failed — CI signature on artifact-manifest.json could not be verified \
             against expected identity `{}` / issuer `{}`",
            certificate_identity_regexp,
            trust.expected_ci_issuer
        );
    }
    println!("  ✓ CI signature verified");

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

    if let Some(prev) = prev_tag.as_deref() {
        println!();
        println!("== diff vs previous release tag (`{prev}`) ==");
        show_git_diff(&args.workspace_root, prev, &args.tag)?;
    } else {
        println!();
        println!("(no previous release tag found — skipping diff)");
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

/// Cosign accepts a regex for the certificate identity; `{}` doesn't need
/// escaping (literal `*` in our pin's `@refs/tags/v*` is what we want as a
/// wildcard). Convert the trust-constants glob into a regex.
fn identity_regex_from_pattern(pattern: &str) -> String {
    // Escape everything except `*`, then translate `*` → `.*`. Anchor with ^…$.
    let mut out = String::from("^");
    for ch in pattern.chars() {
        match ch {
            '*' => out.push_str(".*"),
            '.' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\' | '^' | '$' | '?' => {
                out.push('\\');
                out.push(ch);
            }
            other => out.push(other),
        }
    }
    out.push('$');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_regex_escapes_dots_and_wildcards_star() {
        let r = identity_regex_from_pattern(
            "https://github.com/eidola-ai/eidola/.github/workflows/tinfoil-build.yml@refs/tags/v*",
        );
        assert!(r.starts_with('^'));
        assert!(r.ends_with('$'));
        assert!(r.contains(r"\."));
        assert!(r.contains(".*"));
    }

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
