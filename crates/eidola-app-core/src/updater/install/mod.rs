//! Turning a verified release into a bundle that is ready to become the
//! installed one.
//!
//! [`verify_release`](super::check_for_update_with) proves a release is
//! authentic. This module is what happens next: it downloads the bytes,
//! proves *they* are the bytes that release names, reconstructs the signed
//! macOS app from them, reads back what the reconstruction claims, and
//! leaves the result in a private staging directory. It never touches the
//! installed application — promotion is a separate decision with its own
//! constraints (see [`PromotionReadiness`]).
//!
//! ## Why the updater is this mechanism's main consumer
//!
//! A signed macOS app ships as two published objects: an unsigned payload
//! that is a pure function of source, and detached signature material that
//! is not. `eidola_apple::apply` composes them. An external auditor runs
//! that composition rarely; the updater runs it on **every update**, which
//! is what keeps it from rotting. A break in Apple's format surfaces here,
//! on the next update, rather than the next time a stranger tries to check
//! us.
//!
//! ## Where each expected value comes from
//!
//! Three documents, three different reasons to believe them, and this
//! module composes rather than re-derives:
//!
//! | Value | Source | Why that source |
//! |---|---|---|
//! | payload sha256 | `artifact-manifest.json` | CI-signed, and a pure function of source — anyone can rebuild it |
//! | envelope sha256 | the human attestation | key-dependent bytes, so they may never enter the manifest |
//! | Team ID, signing identifier, hardened runtime | the human attestation | an identity claim a person put their name to |
//! | URLs | `release.json` | unsigned, so it may say *where* bytes are and never *what they are* |
//!
//! The plan carries those values; it does not fetch them. That keeps this
//! module honest about the one thing it must never do, which is to take a
//! value and the bytes it describes from the same untrusted place.
//!
//! ## No partial writes
//!
//! Every failure removes the staging tree. Reconstruction is explicitly
//! *not* atomic once it begins writing (`crates/eidola-apple/AGENTS.md`),
//! which is exactly why it runs against a private copy: a half-reconstructed
//! bundle is discarded rather than promoted. Nothing outside the staging
//! root is created, moved, or removed by anything in here.

mod archive;

use std::path::{Path, PathBuf};

use eidola_apple::SignatureFacts;
use sha2::{Digest, Sha256};

use super::Fetcher;
use crate::error::AppError;

/// A file to download, and the hash a signed document says it must have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteFile {
    pub url: String,
    /// Lowercase hex sha256, without a `sha256:` prefix.
    pub sha256: String,
}

/// What the reconstructed bundle must claim about its own signature.
///
/// These are compared against [`eidola_apple::inspect`]'s structurally
/// parsed facts. That parse is not Apple authentication and never claims to
/// be: it says what the bundle *says* about itself, and the comparison is
/// what ties those words to a person who signed a statement naming them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedSignature {
    pub team_id: Option<String>,
    pub identifier: String,
    pub hardened_runtime: bool,
}

/// Everything an install must satisfy, gathered before anything is fetched.
#[derive(Debug, Clone)]
pub struct InstallPlan {
    pub version: String,
    /// The directory name the container is expected to hold, e.g.
    /// `Eidola.app`. Named rather than discovered: an installer that
    /// installs "whatever was in the archive" has no way to refuse a
    /// container holding something else.
    pub bundle_name: String,
    /// The unsigned shipping container, with the hash the CI-signed
    /// manifest records for it.
    pub payload: RemoteFile,
    /// The detached signature material, with the hash the human
    /// attestation records. `None` where a platform ships no envelope, in
    /// which case the staged payload is the installable as-is.
    pub envelope: Option<RemoteFile>,
    /// What `inspect` must report once the envelope is applied. Required
    /// whenever `envelope` is set: applying signature material and then
    /// not looking at what it claims would leave the identity unchecked.
    pub expected_signature: Option<ExpectedSignature>,
}

/// A verified bundle waiting in a staging directory.
///
/// Holding one of these means: the payload hashed to what the manifest
/// said, the envelope hashed to what the attestation said, reconstruction
/// succeeded, and the reconstructed bundle claims the identity the
/// attestation named. It does **not** mean anything has been installed.
#[derive(Debug)]
pub struct StagedInstall {
    pub version: String,
    /// The reconstructed bundle, inside the staging root.
    pub bundle: PathBuf,
    /// The staging root this owns. Removing it removes the staged bundle.
    pub staging_root: PathBuf,
    /// What the bundle claims about its signature, read back after
    /// reconstruction. `None` when the plan carried no envelope.
    pub signature: Option<SignatureFacts>,
}

impl StagedInstall {
    /// Discard the staged tree. Called by whoever decides not to promote.
    pub fn discard(self) -> Result<(), InstallError> {
        remove_tree(&self.staging_root)
    }
}

/// How an install can fail, as the categories a caller would act on
/// differently.
///
/// The split that matters is between "try again later" ([`Self::Download`])
/// and everything else. Every other variant means a document and the bytes
/// it describes disagree, which is not a condition retrying improves.
#[derive(Debug, thiserror::Error)]
pub enum InstallError {
    #[error("could not download {label}: {reason}")]
    Download { label: String, reason: String },
    #[error(
        "the downloaded {label} is not what the release records \
         (expected sha256 {expected}, got {actual})"
    )]
    HashMismatch {
        label: String,
        expected: String,
        actual: String,
    },
    #[error("the downloaded {label} is not a container this can unpack: {reason}")]
    Archive { label: String, reason: String },
    #[error("the payload does not contain `{expected}`")]
    BundleMissing { expected: String },
    #[error("reconstructing the signed app failed: {0}")]
    Reconstruct(#[from] eidola_apple::ApplyError),
    #[error("reading the reconstructed app's signature failed: {0}")]
    Inspect(#[from] eidola_apple::InspectError),
    #[error(
        "the reconstructed app claims a different {field} than the attestation names \
         (attested {expected}, found {actual})"
    )]
    SignatureMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("an install plan carrying signature material must also carry the identity to check")]
    PlanIncomplete,
    #[error("staging failed at `{path}`: {reason}")]
    Staging { path: PathBuf, reason: String },
}

impl From<InstallError> for AppError {
    fn from(e: InstallError) -> Self {
        AppError::Update {
            message: e.to_string(),
        }
    }
}

/// Download, verify, reconstruct, and inspect — leaving a staged bundle.
///
/// `staging_root` must not exist; this creates it and owns everything under
/// it. On **any** failure it is removed before returning, so a failed
/// install leaves no tree for a later run to find and mistake for a good
/// one.
pub async fn stage(
    fetcher: &Fetcher,
    plan: &InstallPlan,
    staging_root: &Path,
) -> Result<StagedInstall, InstallError> {
    if plan.envelope.is_some() && plan.expected_signature.is_none() {
        return Err(InstallError::PlanIncomplete);
    }

    match stage_inner(fetcher, plan, staging_root).await {
        Ok(staged) => Ok(staged),
        Err(e) => {
            // Best effort: the error that got us here is the one worth
            // reporting, and a staging root that cannot be removed is
            // still a staging root nothing will promote.
            let _ = remove_tree(staging_root);
            Err(e)
        }
    }
}

async fn stage_inner(
    fetcher: &Fetcher,
    plan: &InstallPlan,
    staging_root: &Path,
) -> Result<StagedInstall, InstallError> {
    if staging_root.exists() {
        return Err(InstallError::Staging {
            path: staging_root.to_path_buf(),
            reason: "already exists; staging directories are created fresh".to_string(),
        });
    }
    create_dir(staging_root)?;

    // ── the payload: bytes a signed manifest already vouched for ────────
    let payload_bytes = fetch(fetcher, &plan.payload, "the update payload").await?;
    verify_hash(&payload_bytes, &plan.payload.sha256, "update payload")?;

    let payload_root = staging_root.join("payload");
    create_dir(&payload_root)?;
    archive::unpack_zip(&payload_bytes, &payload_root, "update payload")?;
    drop(payload_bytes);

    let bundle = payload_root.join(&plan.bundle_name);
    if !bundle.is_dir() {
        return Err(InstallError::BundleMissing {
            expected: plan.bundle_name.clone(),
        });
    }

    // ── the envelope: bytes a signed attestation vouched for ────────────
    let Some(envelope) = plan.envelope.as_ref() else {
        return Ok(StagedInstall {
            version: plan.version.clone(),
            bundle,
            staging_root: staging_root.to_path_buf(),
            signature: None,
        });
    };

    let envelope_bytes = fetch(fetcher, envelope, "the signature material").await?;
    verify_hash(&envelope_bytes, &envelope.sha256, "signature material")?;

    let envelope_root = staging_root.join("envelope");
    create_dir(&envelope_root)?;
    archive::unpack_zip(&envelope_bytes, &envelope_root, "signature material")?;
    drop(envelope_bytes);

    // ── compose, then read back what the composition claims ─────────────
    eidola_apple::apply(&bundle, &envelope_root)?;
    let facts = eidola_apple::inspect(&bundle)?;

    let expected = plan
        .expected_signature
        .as_ref()
        .ok_or(InstallError::PlanIncomplete)?;
    compare_signature(expected, &facts)?;

    Ok(StagedInstall {
        version: plan.version.clone(),
        bundle,
        staging_root: staging_root.to_path_buf(),
        signature: Some(facts),
    })
}

/// The identity check: what a person attested against what the bundle says.
fn compare_signature(
    expected: &ExpectedSignature,
    facts: &SignatureFacts,
) -> Result<(), InstallError> {
    if facts.identifier != expected.identifier {
        return Err(InstallError::SignatureMismatch {
            field: "bundle identifier",
            expected: expected.identifier.clone(),
            actual: facts.identifier.clone(),
        });
    }
    if facts.team_id != expected.team_id {
        return Err(InstallError::SignatureMismatch {
            field: "Team ID",
            expected: expected.team_id.clone().unwrap_or_else(|| "none".into()),
            actual: facts.team_id.clone().unwrap_or_else(|| "none".into()),
        });
    }
    if facts.hardened_runtime != expected.hardened_runtime {
        return Err(InstallError::SignatureMismatch {
            field: "hardened-runtime flag",
            expected: expected.hardened_runtime.to_string(),
            actual: facts.hardened_runtime.to_string(),
        });
    }
    Ok(())
}

async fn fetch(fetcher: &Fetcher, file: &RemoteFile, label: &str) -> Result<Vec<u8>, InstallError> {
    fetcher
        .fetch_url(&file.url, label)
        .await
        .map_err(|e| InstallError::Download {
            label: label.to_string(),
            // `AppError::Update`'s text is already URL-stripped by the
            // fetch layer; a download error must never quote the URL back,
            // since a redirect can put credentials in it.
            reason: match e {
                AppError::Update { message } => message,
                other => other.to_string(),
            },
        })
}

fn verify_hash(bytes: &[u8], expected_hex: &str, label: &str) -> Result<(), InstallError> {
    let actual = hex_lower(&Sha256::digest(bytes));
    let expected = expected_hex.trim().to_ascii_lowercase();
    // One hash over the bytes as downloaded. No normalization, no
    // re-serialization, nothing between the file and the claim.
    if actual != expected {
        return Err(InstallError::HashMismatch {
            label: label.to_string(),
            expected,
            actual,
        });
    }
    Ok(())
}

fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(out, "{byte:02x}").expect("writing to a String cannot fail");
    }
    out
}

fn create_dir(path: &Path) -> Result<(), InstallError> {
    std::fs::create_dir_all(path).map_err(|e| InstallError::Staging {
        path: path.to_path_buf(),
        reason: e.to_string(),
    })
}

fn remove_tree(path: &Path) -> Result<(), InstallError> {
    if !path.exists() {
        return Ok(());
    }
    std::fs::remove_dir_all(path).map_err(|e| InstallError::Staging {
        path: path.to_path_buf(),
        reason: e.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Promotion — deliberately a probe, not an action
// ---------------------------------------------------------------------------

/// Whether the running process could replace `installed` with a staged
/// bundle, without asking anyone for anything.
///
/// This only ever reads. Replacing a running application is a product
/// decision about restart behaviour rather than a mechanical one, and the
/// answer differs by where the app was installed — so the mechanism reports
/// what it found and leaves the decision to its caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromotionReadiness {
    /// The bundle and its parent directory are writable by this process:
    /// a swap is an ordinary rename.
    Ready,
    /// The install location is not writable by this user. Promotion would
    /// need privileges this process does not have and must not acquire on
    /// its own.
    NeedsPrivileges { reason: String },
    /// Nothing is installed at that path.
    NotInstalled,
}

/// Probe the install location. Reads only; nothing is created or moved.
pub fn promotion_readiness(installed: &Path) -> PromotionReadiness {
    if !installed.exists() {
        return PromotionReadiness::NotInstalled;
    }
    // A swap replaces the bundle *within* its parent, so the parent's
    // writability is what decides it — a writable bundle inside a
    // read-only directory cannot be replaced, only edited in place, which
    // is precisely what an update must not do to a running app.
    let Some(parent) = installed.parent() else {
        return PromotionReadiness::NeedsPrivileges {
            reason: "the install location has no parent directory".to_string(),
        };
    };
    for dir in [parent, installed] {
        match std::fs::metadata(dir) {
            Ok(meta) if meta.permissions().readonly() => {
                return PromotionReadiness::NeedsPrivileges {
                    reason: format!("`{}` is not writable by this user", dir.display()),
                };
            }
            Ok(_) => {}
            Err(e) => {
                return PromotionReadiness::NeedsPrivileges {
                    reason: format!("`{}` could not be examined: {e}", dir.display()),
                };
            }
        }
    }
    PromotionReadiness::Ready
}
