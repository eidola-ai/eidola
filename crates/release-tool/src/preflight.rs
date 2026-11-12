//! The release-checkout invariant, asserted before anything is verified
//! or signed: the remote tag, the local tag, and HEAD all resolve to one
//! commit, and the working tree and index are clean.
//!
//! This is what lets the rest of the tool read files from the workspace
//! and from `<tag>:` interchangeably — under the assertion they are
//! provably the same bytes — and it pins the commit recorded in the
//! attestation to the commit the *pushed* tag names. Without the remote
//! comparison, a moved or stale local tag ref signs another commit's
//! contents, and the manifest byte-check cannot be relied on to notice:
//! two commits differing only in non-build-input files (docs, say)
//! commit identical manifests.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

/// Assert the invariant for `tag` and return the commit it names.
pub fn assert_release_checkout(workspace_root: &Path, repo: &str, tag: &str) -> Result<String> {
    let local = rev_parse(workspace_root, &format!("{tag}^{{commit}}"))?;
    let head = rev_parse(workspace_root, "HEAD")?;
    let remote = remote_tag_commit(repo, tag)?;

    if remote != local {
        bail!(
            "local tag `{tag}` resolves to {local} but the tag on {repo} resolves to {remote}.\n\
             Your local ref is stale or has been moved — fetch the pushed tag\n\
             (`git fetch origin --tags --force`) and re-run."
        );
    }
    if head != local {
        bail!(
            "HEAD is {head} but tag `{tag}` names {local}.\n\
             Check out the tag (`git checkout {tag}`) so every workspace read\n\
             is the tagged bytes, then re-run."
        );
    }

    let status = Command::new("git")
        .current_dir(workspace_root)
        .args(["status", "--porcelain"])
        .output()
        .context("running `git status --porcelain`")?;
    if !status.status.success() {
        bail!("`git status --porcelain` failed");
    }
    if !status.stdout.is_empty() {
        bail!(
            "the working tree or index has changes.\n\
             Verification and signing read files from the checkout, so it must\n\
             be exactly the tagged commit — stash or commit your changes first.\n\
             (`git status` shows what differs.)"
        );
    }

    assert_tag_names_workspace_version(workspace_root, tag)?;

    println!("== release checkout verified ==");
    println!("  {tag} = {local} (remote agrees, HEAD matches, tree clean, version matches)");
    Ok(local)
}

/// The tag and the version the tagged binaries report are one value.
///
/// Every installed client gates updates on `latest > installed`, where
/// `installed` is its compile-time `CARGO_PKG_VERSION` — the workspace
/// version, not the tag. A release whose tag names a different version
/// ships binaries that misplace themselves: tagged below the workspace
/// version, every client built from it reports itself as newer than the
/// releases that follow and is told it is up to date until the workspace
/// version is passed.
fn assert_tag_names_workspace_version(workspace_root: &Path, tag: &str) -> Result<()> {
    let manifest = std::fs::read_to_string(workspace_root.join("Cargo.toml"))
        .context("reading the workspace Cargo.toml")?;
    let built = workspace_package_version(&manifest)
        .context("the workspace Cargo.toml declares no `[workspace.package]` version")?;
    check_tag_version(tag, built)
}

fn check_tag_version(tag: &str, built: &str) -> Result<()> {
    let tagged = tag.strip_prefix('v').unwrap_or(tag);
    if tagged != built {
        bail!(
            "tag `{tag}` names version {tagged}, but the binaries built from it report \
             {built} (`[workspace.package] version` in Cargo.toml).\n\
             Installed clients compare that compiled-in version against the release\n\
             feed, so the two must agree — bump the workspace version in a commit,\n\
             then tag that commit."
        );
    }
    Ok(())
}

/// `version` from the `[workspace.package]` table, read without a TOML
/// parser: the table is ours and its value a plain quoted string.
fn workspace_package_version(manifest: &str) -> Option<&str> {
    let mut in_table = false;
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            in_table = line == "[workspace.package]";
            continue;
        }
        if in_table
            && let Some(rest) = line.strip_prefix("version")
            && let Some(value) = rest.trim_start().strip_prefix('=')
        {
            return value.trim().strip_prefix('"')?.split('"').next();
        }
    }
    None
}

fn rev_parse(workspace_root: &Path, refname: &str) -> Result<String> {
    let out = Command::new("git")
        .current_dir(workspace_root)
        .args(["rev-parse", refname])
        .output()
        .context("running `git rev-parse`")?;
    if !out.status.success() {
        bail!(
            "`git rev-parse {refname}` failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8(out.stdout)?.trim().to_string())
}

/// The commit the tag names on GitHub, with annotated tag objects peeled.
fn remote_tag_commit(repo: &str, tag: &str) -> Result<String> {
    let (obj_type, sha) = gh_ref_object(repo, &format!("repos/{repo}/git/ref/tags/{tag}"))?;
    if obj_type == "commit" {
        return Ok(sha);
    }
    if obj_type == "tag" {
        // Annotated tag: dereference the tag object to its commit.
        let (inner_type, inner_sha) = gh_ref_object(repo, &format!("repos/{repo}/git/tags/{sha}"))?;
        if inner_type != "commit" {
            bail!("tag `{tag}` on {repo} dereferences to a `{inner_type}` object, not a commit");
        }
        return Ok(inner_sha);
    }
    bail!("tag `{tag}` on {repo} points at a `{obj_type}` object, not a commit or tag");
}

fn gh_ref_object(repo: &str, endpoint: &str) -> Result<(String, String)> {
    let out = Command::new("gh")
        .args([
            "api",
            endpoint,
            "--jq",
            ".object.type + \" \" + .object.sha",
        ])
        .output()
        .context("running `gh api` for the remote tag")?;
    if !out.status.success() {
        bail!(
            "`gh api {endpoint}` failed (does the tag exist on {repo}?): {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let text = String::from_utf8(out.stdout)?;
    let mut parts = text.trim().split(' ');
    match (parts.next(), parts.next()) {
        (Some(t), Some(s)) if !t.is_empty() && !s.is_empty() => Ok((t.to_string(), s.to_string())),
        _ => bail!("unexpected `gh api {endpoint}` output"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tag_must_name_the_version_its_binaries_report() {
        // A tag below the workspace version ships clients that report the
        // workspace version, newer than the tag says, and are told they are up
        // to date with every release until that version is passed.
        let err = check_tag_version("v0.0.8", "0.1.0")
            .unwrap_err()
            .to_string();
        assert!(err.contains("report 0.1.0"), "got: {err}");
        check_tag_version("v0.1.1", "0.1.1").expect("a matching tag passes");
    }

    #[test]
    fn the_workspace_version_is_read_from_its_own_table() {
        let manifest = r#"
[package]
version = "9.9.9"

[workspace.package]
edition = "2024"
version = "0.1.0"

[workspace.dependencies]
version = "8.8.8"
"#;
        assert_eq!(workspace_package_version(manifest), Some("0.1.0"));
        assert_eq!(
            workspace_package_version("[workspace]\nmembers = []\n"),
            None
        );
    }

    #[test]
    fn the_committed_workspace_declares_a_version() {
        // The parser reads what cargo reads: this crate inherits the
        // workspace version, so the two must agree.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let manifest = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
        let version = workspace_package_version(&manifest).expect("workspace version");
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }
}
