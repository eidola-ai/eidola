//! `release-tool verify <tag>` — fetch and verify the CI-built artifact
//! manifest, then show the diff against the prior release for human review.
//!
//! Sigstore verification goes through `eidola-app-core`'s pure-Rust
//! verifier — the same code path that ships to users. Keeping a single
//! implementation eliminates the "two versions of the same check drift
//! apart" risk that the previous `cosign verify-blob` shell-out carried.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};

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

    // Fetch, Sigstore-verify, and byte-compare through the same function
    // `release-attest` is forced through, so the two subcommands can never
    // disagree about what a valid manifest is.
    crate::manifest::fetch_verified_manifest(&args.workspace_root, &args.repo, &args.tag)?;

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
