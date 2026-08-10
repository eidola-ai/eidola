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

    println!("== release checkout verified ==");
    println!("  {tag} = {local} (remote agrees, HEAD matches, tree clean)");
    Ok(local)
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
