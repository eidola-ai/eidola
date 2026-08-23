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
///
/// The fields are private and there is no public constructor, which is the
/// whole point: [`Self::discard`] deletes a tree recursively, so the path
/// it deletes must be one this module created rather than one a caller
/// supplied. Holding a `StagedInstall` *is* the evidence that `stage`
/// created that directory.
#[derive(Debug)]
pub struct StagedInstall {
    version: String,
    bundle: PathBuf,
    staging_root: PathBuf,
    signature: Option<SignatureFacts>,
}

impl StagedInstall {
    /// The release version this bundle is.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// The reconstructed bundle, inside the staging root.
    pub fn bundle(&self) -> &Path {
        &self.bundle
    }

    /// What the bundle claims about its signature, read back after
    /// reconstruction. `None` when the plan carried no envelope.
    pub fn signature(&self) -> Option<&SignatureFacts> {
        self.signature.as_ref()
    }

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

    // The caller chooses where staging happens, so the directory *holding*
    // it is the caller's — creating it here would mean creating something
    // this call never removes, which is the same ownership mistake in the
    // other direction.
    match staging_root.parent() {
        Some(parent) if parent.as_os_str().is_empty() => {}
        Some(parent) if !parent.is_dir() => {
            return Err(InstallError::Staging {
                path: parent.to_path_buf(),
                reason: "does not exist; the directory holding a staging root belongs to the \
                         caller, so this will not create it"
                    .to_string(),
            });
        }
        _ => {}
    }

    // Creating the directory is what earns the right to delete it. The
    // check and the claim are one atomic step: `create_dir` fails if
    // anything is already there, so a staging path that collides with a
    // caller's data is refused *before* any cleanup is armed, and cannot
    // be created by someone else in between.
    std::fs::create_dir(staging_root).map_err(|e| InstallError::Staging {
        path: staging_root.to_path_buf(),
        reason: if e.kind() == std::io::ErrorKind::AlreadyExists {
            "already exists; staging directories are created fresh, and this one is not \
             ours to remove"
                .to_string()
        } else {
            e.to_string()
        },
    })?;

    // From here on the tree is removed by *going out of scope*, not by
    // reaching an error arm. An install can also end by never finishing —
    // a caller that times out or races this against a quit drops the
    // future mid-download, and a cleanup written as an error arm never
    // runs. `Drop` runs then too.
    let guard = StagingGuard::arm(staging_root);
    let staged = stage_inner(fetcher, plan, staging_root).await?;
    Ok(guard.keep(staged))
}

/// Removes the staging tree unless the install got far enough to hand it
/// over.
///
/// The removal is blocking, and it runs on whatever thread drops this —
/// including an async worker. That is deliberate: the tree is one this
/// call created, so the work is bounded, and the alternative to a brief
/// blocking `remove_dir_all` is leaving a half-reconstructed bundle on
/// disk for a later run to find.
struct StagingGuard {
    root: Option<PathBuf>,
}

impl StagingGuard {
    fn arm(root: &Path) -> Self {
        Self {
            root: Some(root.to_path_buf()),
        }
    }

    /// Disarm, because the staged tree now belongs to the returned value.
    fn keep(mut self, staged: StagedInstall) -> StagedInstall {
        self.root = None;
        staged
    }
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if let Some(root) = self.root.take() {
            // Best effort: the reason we are unwinding is the one worth
            // reporting, and a staging root that cannot be removed is
            // still a staging root nothing will promote.
            let _ = std::fs::remove_dir_all(&root);
        }
    }
}

async fn stage_inner(
    fetcher: &Fetcher,
    plan: &InstallPlan,
    staging_root: &Path,
) -> Result<StagedInstall, InstallError> {
    // ── the payload: bytes a signed manifest already vouched for ────────
    let payload_bytes = fetch(fetcher, &plan.payload, "the update payload").await?;
    verify_hash(&payload_bytes, &plan.payload.sha256, "update payload")?;

    let payload_root = staging_root.join("payload");
    create_dir_all(&payload_root)?;
    archive::unpack_zip(&payload_bytes, &payload_root, "update payload")?;
    drop(payload_bytes);

    // A plan names the directory it expects; it does not get to name a
    // *path*. Everything downstream of this — including reconstruction,
    // which writes — would otherwise act on whatever the join reached.
    let bundle_name = archive::single_component(&plan.bundle_name).ok_or_else(|| {
        InstallError::BundleMissing {
            expected: plan.bundle_name.clone(),
        }
    })?;
    let bundle = directory_named(&payload_root, &bundle_name).ok_or_else(|| {
        InstallError::BundleMissing {
            expected: plan.bundle_name.clone(),
        }
    })?;

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
    create_dir_all(&envelope_root)?;
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

/// Find a directory entry whose name is *exactly* `name`.
///
/// `path.is_dir()` answers a different question: it asks the filesystem
/// whether that path resolves, and the filesystems this unpacks onto
/// resolve names that are not the name asked for. Measured on APFS: a
/// case-insensitive volume resolves `eidola.app` for `Eidola.app`, and
/// normalization-insensitivity resolves an NFC spelling for an NFD entry
/// even on a case-sensitive one. Either way the plan's exact-name
/// requirement would go unenforced and the path handed downstream would be
/// an alias. So the entries are read and the names compared, which is the
/// same rule the container's members are already held to.
fn directory_named(parent: &Path, name: &Path) -> Option<PathBuf> {
    let wanted = name.as_os_str();
    for entry in std::fs::read_dir(parent).ok()? {
        let Ok(entry) = entry else { continue };
        if entry.file_name() == wanted {
            return entry.file_type().ok()?.is_dir().then(|| entry.path());
        }
    }
    None
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

/// The most this will hold in memory for one container.
///
/// The URL these bytes come from is **attacker-suppliable by design**:
/// `release.json` is unsigned, which is exactly why the hash that judges
/// them lives in the manifest and the attestation instead. So the fetch
/// has to be robust against a hostile URL and not merely a wrong one — a
/// body is bounded *before* it is buffered, because a hash cannot be
/// checked until the bytes are already here, and the unpack limit is a
/// bound on the tree, not on the wire.
///
/// Today's macOS container is around 40 MB; this leaves an order of
/// magnitude of headroom and still bounds what a hostile server can make
/// this allocate.
const MAX_CONTAINER_BYTES: u64 = 512 * 1024 * 1024;

async fn fetch(fetcher: &Fetcher, file: &RemoteFile, label: &str) -> Result<Vec<u8>, InstallError> {
    fetcher
        .fetch_url_bounded(&file.url, label, MAX_CONTAINER_BYTES)
        .await
        .map_err(|e| InstallError::Download {
            label: label.to_string(),
            // The fetch layer strips URLs from error text (`AppError`'s
            // `request_error_text`), so this passes the message through
            // rather than reformatting it: a redirect can leave a
            // credential in a URL, and a standing failure is rendered.
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

fn create_dir_all(path: &Path) -> Result<(), InstallError> {
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
/// This only ever reads. Replacing a running application is a decision
/// about restart behaviour rather than a mechanical step, and the answer
/// depends on where the app was installed — so the mechanism reports what
/// it found and leaves the decision to its caller.
///
/// macOS-only, because replacing an `.app` in place is: the other
/// platforms install through a package manager that owns this question.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromotionReadiness {
    /// The bundle and the directory holding it are writable by this
    /// process: a swap is an ordinary rename.
    Ready,
    /// Promotion would need privileges this process does not have — and
    /// must not acquire on its own.
    NeedsPrivileges { reason: String },
    /// Nothing is installed at that path.
    NotInstalled,
}

/// Probe the install location. Reads only; nothing is created or moved.
#[cfg(target_os = "macos")]
pub fn promotion_readiness(installed: &Path) -> PromotionReadiness {
    // `exists()` answers false for "no" *and* for "an ancestor is not
    // searchable", which are opposite answers here: the second means a
    // bundle may well be installed and this process cannot get to it.
    match installed.try_exists() {
        Ok(true) => {}
        Ok(false) => return PromotionReadiness::NotInstalled,
        Err(e) => {
            return PromotionReadiness::NeedsPrivileges {
                reason: format!("`{}` could not be examined: {e}", installed.display()),
            };
        }
    }
    // A swap replaces the bundle *within* its parent, so the parent has to
    // be writable too — a writable bundle inside a read-only directory can
    // only be edited in place, which is precisely what an update must not
    // do to a running app.
    let Some(parent) = installed.parent() else {
        return PromotionReadiness::NeedsPrivileges {
            reason: "the install location has no parent directory".to_string(),
        };
    };
    for dir in [parent, installed] {
        if !writable(dir) {
            return PromotionReadiness::NeedsPrivileges {
                reason: format!("`{}` is not writable by this user", dir.display()),
            };
        }
    }
    PromotionReadiness::Ready
}

/// Ask the kernel, not the mode bits.
///
/// `Permissions::readonly()` reports whether *any* write bit is set, which
/// says nothing about whether **this** user may write: a root-owned
/// `/Applications` is mode 0755 and would read as writable to everyone.
/// `access(2)` accounts for ownership, groups and ACLs, which is the
/// question being asked.
#[cfg(target_os = "macos")]
fn writable(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;

    let Ok(c_path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
    // SAFETY: `c_path` is a valid NUL-terminated string that outlives the
    // call, and `access` only reads it.
    unsafe { libc::access(c_path.as_ptr(), libc::W_OK) == 0 }
}

#[cfg(test)]
mod bundle_lookup_tests {
    use super::*;

    #[test]
    fn a_name_the_filesystem_folds_is_still_not_the_name() {
        // APFS resolves an NFC spelling for an NFD entry regardless of
        // case sensitivity, so `is_dir()` says yes to a name that is not
        // the one on disk. Reading the entry and comparing says no, which
        // is the answer a plan naming an exact directory asked for.
        let root = tempfile::tempdir().unwrap();
        let on_disk = "cafe\u{301}.app";
        let asked_for = "caf\u{e9}.app";
        std::fs::create_dir(root.path().join(on_disk)).unwrap();

        assert_eq!(
            directory_named(root.path(), Path::new(on_disk)),
            Some(root.path().join(on_disk)),
            "the name that is really there resolves"
        );
        assert_eq!(
            directory_named(root.path(), Path::new(asked_for)),
            None,
            "a spelling the filesystem folds onto it does not"
        );
    }

    #[test]
    fn a_file_of_the_right_name_is_not_a_bundle() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("Eidola.app"), b"not a directory").unwrap();
        assert_eq!(directory_named(root.path(), Path::new("Eidola.app")), None);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod promotion_tests {
    use super::*;

    #[test]
    fn reports_what_it_found_without_touching_anything() {
        let root = tempfile::tempdir().unwrap();

        let missing = root.path().join("Nothing.app");
        assert_eq!(
            promotion_readiness(&missing),
            PromotionReadiness::NotInstalled
        );

        let installed = root.path().join("Eidola.app");
        std::fs::create_dir(&installed).unwrap();
        assert_eq!(promotion_readiness(&installed), PromotionReadiness::Ready);

        // Root bypasses the permission check entirely, so the negative
        // case only means something as an ordinary user.
        // SAFETY: `geteuid` reads process state and cannot fail.
        if unsafe { libc::geteuid() } != 0 {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o555)).unwrap();
            let verdict = promotion_readiness(&installed);
            std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
            assert!(
                matches!(verdict, PromotionReadiness::NeedsPrivileges { .. }),
                "a bundle whose parent this user cannot write must not read as promotable: \
                 {verdict:?}"
            );
        }

        // Nothing was created or moved by any of the above.
        assert!(installed.is_dir());
        assert!(!missing.exists());
    }

    #[test]
    fn an_unreachable_bundle_is_not_an_absent_one() {
        // SAFETY: `geteuid` reads process state and cannot fail.
        if unsafe { libc::geteuid() } == 0 {
            return; // root searches anything; the case cannot arise.
        }
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        let outer = root.path().join("outer");
        let installed = outer.join("Eidola.app");
        std::fs::create_dir_all(&installed).unwrap();
        std::fs::set_permissions(&outer, std::fs::Permissions::from_mode(0o000)).unwrap();

        let verdict = promotion_readiness(&installed);
        std::fs::set_permissions(&outer, std::fs::Permissions::from_mode(0o755)).unwrap();

        assert!(
            matches!(verdict, PromotionReadiness::NeedsPrivileges { .. }),
            "an ancestor this process cannot search means the answer is unknown, and an \
             unknown answer is not `NotInstalled`: {verdict:?}"
        );
    }
}
