//! Release engineer's tool. Two subcommands, run in order on a tag that has
//! already been pushed and built by CI:
//!
//! 1. `release-tool verify <tag>` — fetches the CI-built
//!    `artifact-manifest.json` + Sigstore bundle from the GitHub release,
//!    verifies the bundle against the embedded trust root, compares the
//!    CI manifest against the committed manifest byte-for-byte, and opens
//!    the diff against the prior release tag for human review.
//!
//! 2. `release-tool attest <tag>` — interactively walks every claim in
//!    `attestation-templates-v1.json`, rendering each from the engineer's
//!    inputs. Each claim requires typing the word `yes` to affirm; anything
//!    else aborts. On full affirmation, signs the attestation file with the
//!    engineer's hardware-backed SSH key via `ssh-keygen -Y sign`, posts the
//!    resulting signature to Sigstore Rekor as a `hashedrekord` entry, and
//!    uploads the attestation JSON + bundle (rekor inclusion proof) to the
//!    release. Then generates and uploads `release.json` (URL-only index —
//!    see `releases/TRUST-ROOT.md`) and marks the release as latest.
//!
//! CI side uses sigstore + cosign (Fulcio keyless via OIDC). Human side uses
//! SSH signatures + Rekor `hashedrekord`. Both end up in the same Rekor
//! transparency log; the verifier dispatches on signature format.
//!
//! Shells out to `gh`, `ssh-keygen`, and `git`. All must be on PATH. CI
//! signature verification goes through `eidola-app-core`'s pure-Rust
//! verifier — the same code path that ships to users — so `cosign` is no
//! longer required on the engineer's PATH. `ssh-keygen -Y sign`
//! automatically uses `SSH_AUTH_SOCK` to reach agent-held keys (Secretive,
//! 1Password, FIDO2-SK, …), so the engineer does not need the private key
//! on disk.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

mod attest;
mod trust;
mod verify;

#[derive(Parser)]
#[command(
    name = "release-tool",
    about = "Eidola release engineer's verify + attest workflow",
    version
)]
struct Cli {
    /// Override the GitHub repo (default: parsed from `git config remote.origin.url`).
    #[arg(long, global = true, env = "EIDOLA_RELEASE_REPO")]
    repo: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Verify CI's signed artifact-manifest.json and review the diff.
    Verify {
        /// The release tag, e.g. `v0.5.0`.
        tag: String,
    },

    /// Interactively render and affirm each claim, sign with the hardware-
    /// backed SSH key, post to Rekor, upload, and mark the release as
    /// latest.
    Attest {
        /// The release tag, e.g. `v0.5.0`.
        tag: String,

        /// Path to the SSH public key file (`.pub`). The private key must
        /// be reachable via `SSH_AUTH_SOCK` (e.g. Secretive, 1Password,
        /// FIDO2-SK). Read from `EIDOLA_ATTESTANT_SSH_PUBKEY` if set.
        #[arg(long, env = "EIDOLA_ATTESTANT_SSH_PUBKEY")]
        ssh_pubkey: std::path::PathBuf,

        /// Short attestant identifier — used as the filename suffix
        /// (`attestation-<id>.json`) and recorded in `release.json`. The
        /// SSH pubkey's fingerprint is what's actually matched against the
        /// client's `trusted_attestant_fingerprints`.
        #[arg(long, env = "EIDOLA_ATTESTANT_ID")]
        attestant_id: String,

        /// Attestant's full legal name, substituted verbatim into the
        /// "under penalty of perjury" preamble.
        #[arg(long, env = "EIDOLA_ATTESTANT_NAME")]
        attestant_name: String,

        /// Jurisdiction whose perjury laws apply (e.g.
        /// `the State of California, United States`).
        #[arg(long, env = "EIDOLA_ATTESTANT_JURISDICTION")]
        jurisdiction: String,
    },
}

fn main() -> Result<()> {
    // Install the pure-Rust rustls crypto provider, matching the rest of
    // the workspace's TLS choice. Idempotent across re-installs.
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());

    let cli = Cli::parse();
    let workspace_root = workspace_root()?;
    let repo = resolve_repo(cli.repo.as_deref(), &workspace_root)?;

    match cli.command {
        Command::Verify { tag } => verify::run(verify::Args {
            workspace_root,
            repo,
            tag,
        }),
        Command::Attest {
            tag,
            ssh_pubkey,
            attestant_id,
            attestant_name,
            jurisdiction,
        } => attest::run(attest::Args {
            workspace_root,
            repo,
            tag,
            ssh_pubkey,
            attestant_id,
            attestant_name,
            jurisdiction,
        }),
    }
}

fn workspace_root() -> Result<PathBuf> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .context("running `git rev-parse --show-toplevel`")?;
    if !out.status.success() {
        bail!(
            "git rev-parse failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let path = String::from_utf8(out.stdout)
        .context("git output not utf-8")?
        .trim()
        .to_string();
    Ok(PathBuf::from(path))
}

fn resolve_repo(override_repo: Option<&str>, workspace_root: &std::path::Path) -> Result<String> {
    if let Some(r) = override_repo {
        return Ok(r.to_string());
    }
    let out = std::process::Command::new("git")
        .current_dir(workspace_root)
        .args(["config", "--get", "remote.origin.url"])
        .output()
        .context("reading `remote.origin.url`")?;
    if !out.status.success() {
        bail!(
            "no `remote.origin.url` configured; pass --repo owner/name (or set EIDOLA_RELEASE_REPO)"
        );
    }
    let url = String::from_utf8(out.stdout)
        .context("git output not utf-8")?
        .trim()
        .to_string();
    parse_owner_repo(&url).with_context(|| format!("parsing repo from `{url}`"))
}

/// Accept both `git@github.com:owner/repo.git` and
/// `https://github.com/owner/repo.git` (with or without the `.git` suffix)
/// and emit `owner/repo`.
fn parse_owner_repo(url: &str) -> Result<String> {
    let stripped = url.trim().trim_end_matches('/').trim_end_matches(".git");
    let after = stripped
        .strip_prefix("git@github.com:")
        .or_else(|| stripped.strip_prefix("https://github.com/"))
        .or_else(|| stripped.strip_prefix("http://github.com/"))
        .or_else(|| stripped.strip_prefix("ssh://git@github.com/"))
        .ok_or_else(|| anyhow::anyhow!("unrecognized GitHub URL form"))?;
    let parts: Vec<&str> = after.splitn(2, '/').collect();
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        bail!("could not extract owner/repo from `{url}`");
    }
    Ok(format!("{}/{}", parts[0], parts[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ssh_url() {
        assert_eq!(
            parse_owner_repo("git@github.com:eidola-ai/eidola.git").unwrap(),
            "eidola-ai/eidola"
        );
    }

    #[test]
    fn parse_https_url_without_suffix() {
        assert_eq!(
            parse_owner_repo("https://github.com/eidola-ai/eidola").unwrap(),
            "eidola-ai/eidola"
        );
    }

    #[test]
    fn parse_https_url_with_suffix() {
        assert_eq!(
            parse_owner_repo("https://github.com/eidola-ai/eidola.git").unwrap(),
            "eidola-ai/eidola"
        );
    }

    #[test]
    fn parse_unrecognized_url_errors() {
        assert!(parse_owner_repo("https://gitlab.com/foo/bar").is_err());
    }
}
