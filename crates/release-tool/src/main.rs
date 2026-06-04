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
//!    else aborts. On full affirmation, signs the attestation file via
//!    `cosign sign-blob --key <ref>` (where `<ref>` is a local PEM, a
//!    PKCS#11 URI for a YubiKey/SmartCard, or any cosign-supported KMS
//!    URI), which posts the signature to Sigstore Rekor as a
//!    `hashedrekord` v0.0.1 entry and emits a Sigstore Bundle v0.3.
//!    `release-tool` uploads the attestation JSON + bundle + the
//!    `release.json` URL index to the GitHub release and marks it as
//!    latest. See `docs/trust-root.md`.
//!
//! Both CI and engineer sides ride the same Rekor entry shape:
//! `hashedrekord` v0.0.1. They differ only in the publicKey arm — CI
//! has a Fulcio keyless leaf cert; engineer has a PKIX
//! SubjectPublicKeyInfo whose `sha256` fingerprint is pinned by
//! `TRUSTED_ATTESTANT_FINGERPRINTS`. The Rekor v2 transition keeps
//! `hashedrekord` (rekord and the SSH PKI are being retired), so this
//! shape is forward-compatible.
//!
//! Shells out to `gh`, `cosign`, and `git`. All must be on PATH. For
//! local PEM cosign keys the engineer also needs `COSIGN_PASSWORD` in
//! the environment; for PKCS#11 / KMS keys, the device or KMS handles
//! its own auth (PIN, IAM, ...).

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

mod attest;
mod pkcs11;
mod provenance;
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
    /// backed attestant key, post to Rekor, upload, and mark the
    /// release as latest.
    Attest {
        /// The release tag, e.g. `v0.5.0`.
        tag: String,

        /// Cosign key reference. One of: a local PEM path
        /// (`/path/to/cosign.key` — passphrase from `COSIGN_PASSWORD`),
        /// a PKCS#11 URI (for a YubiKey, get a PIN-free one from
        /// `release-tool pkcs11 list` and supply the PIN via
        /// `COSIGN_PKCS11_PIN`), or any KMS URI cosign supports
        /// (`awskms:...`, `gcpkms:...`, `azurekms:...`,
        /// `hashivault:...`). Passed through to `cosign sign-blob --key`
        /// verbatim. Read from `EIDOLA_ATTESTANT_COSIGN_KEY` if set.
        ///
        /// The underlying key must be ECDSA-P256, ECDSA-P384, or
        /// Ed25519 — RSA, ECDSA-P521, and other algorithms are rejected
        /// up front because the updater's verifier only accepts those
        /// three. release-tool fetches the pubkey with
        /// `cosign public-key --key <ref>` and validates the SPKI
        /// algorithm OID before signing.
        #[arg(long, env = "EIDOLA_ATTESTANT_COSIGN_KEY")]
        cosign_key: String,

        /// Short attestant identifier — used as the filename suffix
        /// (`attestation-<id>.json`) and recorded in `release.json`. The
        /// attestant key's fingerprint (`sha256(SPKI DER)`) is what's
        /// actually matched against the client's
        /// `trusted_attestant_fingerprints`.
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

    /// PKCS#11 helpers for hardware attestant keys (YubiKey / SmartCard /
    /// HSM).
    #[command(subcommand)]
    Pkcs11(Pkcs11Command),

    /// Informational hardware-provenance evidence for attestant keys
    /// (NOT consulted by trust evaluation — see
    /// releases/trust/attestant-provenance/README.md).
    #[command(subcommand)]
    Provenance(ProvenanceCommand),
}

#[derive(Subcommand)]
enum ProvenanceCommand {
    /// Verify each committed provenance bundle's attestation certificate
    /// matches the fingerprint its meta.json claims, and report which
    /// fingerprints are currently pinned.
    Check {
        /// Provenance directory (default:
        /// `releases/trust/attestant-provenance`).
        #[arg(long)]
        dir: Option<PathBuf>,
    },

    /// Capture a YubiKey-PIV attestation bundle via `ykman` and fill its
    /// meta.json from the attestation cert. Other key sources fill the same
    /// bundle shape by hand.
    Capture {
        /// Attestant identifier — the bundle subdirectory name.
        #[arg(long, env = "EIDOLA_ATTESTANT_ID")]
        attestant_id: String,

        /// PIV slot holding the signing key.
        #[arg(long, default_value = "9c")]
        slot: String,

        /// Provenance directory (default:
        /// `releases/trust/attestant-provenance`).
        #[arg(long)]
        dir: Option<PathBuf>,
    },

    /// (Re)derive meta.json fields from a bundle's committed attestation
    /// certificate — no device or `ykman` needed. Runs offline over one
    /// bundle (`--attestant-id`) or all of them.
    Enrich {
        /// A single bundle to enrich; omit to enrich every bundle.
        #[arg(long, env = "EIDOLA_ATTESTANT_ID")]
        attestant_id: Option<String>,

        /// Provenance directory (default:
        /// `releases/trust/attestant-provenance`).
        #[arg(long)]
        dir: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum Pkcs11Command {
    /// List signing keys on a PKCS#11 token and print PIN-free cosign
    /// `--key` URIs (no `pin-value`, no `slot-id`). Reads only public
    /// objects, so it never prompts for or emits a PIN.
    List {
        /// Path to the PKCS#11 module (`libykcs11.dylib` / `.so`).
        /// Defaults to probing the well-known install locations.
        #[arg(long, env = "EIDOLA_PKCS11_MODULE")]
        module_path: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    // Install the pure-Rust rustls crypto provider, matching the rest of
    // the workspace's TLS choice. Idempotent across re-installs.
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());

    let cli = Cli::parse();

    match cli.command {
        // PKCS#11 helpers are device-local and need neither a workspace nor
        // a GitHub repo, so resolve those only for the release subcommands.
        Command::Pkcs11(Pkcs11Command::List { module_path }) => {
            pkcs11::run(pkcs11::Args { module_path })
        }
        Command::Provenance(cmd) => {
            // Needs the workspace (for trust-constants.json and the default
            // provenance directory) but not a GitHub repo.
            let workspace_root = workspace_root()?;
            match cmd {
                ProvenanceCommand::Check { dir } => provenance::check(provenance::CheckArgs {
                    workspace_root,
                    dir,
                }),
                ProvenanceCommand::Capture {
                    attestant_id,
                    slot,
                    dir,
                } => provenance::capture(provenance::CaptureArgs {
                    workspace_root,
                    attestant_id,
                    slot,
                    dir,
                }),
                ProvenanceCommand::Enrich { attestant_id, dir } => {
                    provenance::enrich(provenance::EnrichArgs {
                        workspace_root,
                        attestant_id,
                        dir,
                    })
                }
            }
        }
        Command::Verify { tag } => {
            let workspace_root = workspace_root()?;
            let repo = resolve_repo(cli.repo.as_deref(), &workspace_root)?;
            verify::run(verify::Args {
                workspace_root,
                repo,
                tag,
            })
        }
        Command::Attest {
            tag,
            cosign_key,
            attestant_id,
            attestant_name,
            jurisdiction,
        } => {
            let workspace_root = workspace_root()?;
            let repo = resolve_repo(cli.repo.as_deref(), &workspace_root)?;
            attest::run(attest::Args {
                workspace_root,
                repo,
                tag,
                cosign_key,
                attestant_id,
                attestant_name,
                jurisdiction,
            })
        }
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
