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

use std::os::unix::fs::DirBuilderExt;
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
    #[error(
        "an install plan must carry signature material and the identity to check it against, \
         or neither"
    )]
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
/// it. Its parent must exist — the directory holding a staging root is the
/// caller's, not this module's.
///
/// The path is **resolved** before anything is created, so a relative one
/// cannot re-resolve later and a symlinked ancestor is followed once.
/// [`StagedInstall::bundle`] therefore sits under the resolved root, which
/// may be spelled differently than the path passed here — on macOS a
/// temporary directory under `/var` comes back under `/private/var`. A
/// caller comparing the two should resolve its own path too. On **any** failure it is removed before returning, so a failed
/// install leaves no tree for a later run to find and mistake for a good
/// one.
pub async fn stage(
    fetcher: &Fetcher,
    plan: &InstallPlan,
    staging_root: &Path,
) -> Result<StagedInstall, InstallError> {
    // Each half requires the other. Signature material with no identity to
    // check means applying signatures nobody looks at; an identity with no
    // material means the requirement is never applied *or* checked, and a
    // plan that demands a Team ID would install an app that was never
    // asked for one.
    if plan.envelope.is_some() != plan.expected_signature.is_some() {
        return Err(InstallError::PlanIncomplete);
    }

    // Pinned to one absolute path before anything touches the filesystem.
    // A relative path is not a location, it is a location *plus* whatever
    // the process working directory happens to be — and this function
    // awaits, so that can change underneath it. Every later operation, and
    // the guard that deletes on the way out, would then be naming
    // something else. Resolving the parent also *is* round 2's ownership
    // check: the directory holding a staging root belongs to the caller,
    // so one that cannot be resolved is refused rather than created.
    let staging_root = &absolutize_staging_root(staging_root)?;

    // Creating the directory is what earns the right to delete it. The
    // check and the claim are one atomic step: `create_dir` fails if
    // anything is already there, so a staging path that collides with a
    // caller's data is refused *before* any cleanup is armed, and cannot
    // be created by someone else in between.
    //
    // The mode is stated rather than left to the umask. Under a permissive
    // one the tree would be world-writable, and every hash this module
    // checks would be checkable-then-replaceable: another local user could
    // swap a payload after it was verified and before it was used.
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(staging_root)
        .map_err(|e| InstallError::Staging {
            path: staging_root.clone(),
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

/// One absolute path for the staging root, resolved before any filesystem
/// operation names it.
///
/// The root does not exist yet, so its *parent* is what gets resolved and
/// the final component is joined back on. A parent that cannot be resolved
/// is a parent that does not exist, which is the caller's to create.
fn absolutize_staging_root(staging_root: &Path) -> Result<PathBuf, InstallError> {
    let Some(name) = staging_root.file_name() else {
        return Err(InstallError::Staging {
            path: staging_root.to_path_buf(),
            reason: "is not a directory this could create (it names no final component)"
                .to_string(),
        });
    };

    // `Some("")` is what a single-component relative path reports; that
    // parent is the working directory.
    let parent = match staging_root.parent() {
        Some(parent) if parent.as_os_str().is_empty() => Path::new("."),
        Some(parent) => parent,
        None => Path::new("."),
    };

    let resolved = parent.canonicalize().map_err(|e| InstallError::Staging {
        path: parent.to_path_buf(),
        reason: format!(
            "could not be resolved ({e}); the directory holding a staging root belongs to \
             the caller, so this will not create it"
        ),
    })?;

    Ok(resolved.join(name))
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
    /// A swap is an ordinary rename: same filesystem, and this process may
    /// replace entries in the directory holding the installed bundle.
    Ready,
    /// The staged bundle and the install location are on different
    /// filesystems, so `rename` cannot move one to the other — it would
    /// fail `EXDEV` no matter the permissions. Reported rather than worked
    /// around here: the caller chose where staging lives, so the caller is
    /// who can move it or copy across.
    DifferentFilesystem { staged: PathBuf, installed: PathBuf },
    /// Promotion would need privileges this process does not have — and
    /// must not acquire on its own.
    NeedsPrivileges { reason: String },
    /// Nothing is installed at that path.
    NotInstalled,
}

/// Probe the install location against a staged bundle. Reads only;
/// nothing is created or moved.
///
/// `staged` is the bundle that would be renamed into place — its
/// filesystem is half the question.
#[cfg(target_os = "macos")]
pub fn promotion_readiness(staged: &Path, installed: &Path) -> PromotionReadiness {
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

    // Resolved for the same reason a staging root is: a relative path is
    // not a location. It also settles the parent — `Path::parent` of a
    // single-component path is `Some("")`, and asking the kernel about an
    // empty pathname fails, which would report a perfectly writable
    // directory as needing privileges.
    let installed = &match installed.canonicalize() {
        Ok(path) => path,
        Err(e) => {
            return PromotionReadiness::NeedsPrivileges {
                reason: format!("`{}` could not be resolved: {e}", installed.display()),
            };
        }
    };
    // The swap renames the installed bundle aside and renames the staged
    // one into its place. Both are operations on entries *in the parent
    // directory*, so what they need is write and search there — and
    // nothing at all inside the old bundle, whose contents are never
    // touched. Asking whether the bundle itself is writable answers a
    // question the swap does not ask, and answers it wrongly for the
    // ordinary case of a read-only `.app` in a writable directory.
    let Some(parent) = installed.parent() else {
        return PromotionReadiness::NeedsPrivileges {
            reason: "the install location has no parent directory".to_string(),
        };
    };
    if !renamable_within(parent) {
        return PromotionReadiness::NeedsPrivileges {
            reason: format!(
                "`{}` does not allow this user to replace entries in it",
                parent.display()
            ),
        };
    }

    // A sticky directory — `/tmp` and anything modelled on it — narrows
    // write permission: an entry there may only be renamed or removed by
    // whoever owns the entry, or owns the directory, or is root. Write and
    // search on the parent are necessary and, here, not sufficient.
    match sticky_verdict(parent, installed) {
        Ok(true) => {}
        Ok(false) => {
            return PromotionReadiness::NeedsPrivileges {
                reason: format!(
                    "`{}` is sticky and this user owns neither it nor `{}`",
                    parent.display(),
                    installed.display()
                ),
            };
        }
        Err(e) => {
            return PromotionReadiness::NeedsPrivileges {
                reason: format!("`{}` could not be examined: {e}", parent.display()),
            };
        }
    }

    // `rename` does not cross filesystems, whatever the permissions say.
    match (device_of(staged), device_of(parent)) {
        (Ok(from), Ok(to)) if from != to => {
            return PromotionReadiness::DifferentFilesystem {
                staged: staged.to_path_buf(),
                installed: installed.to_path_buf(),
            };
        }
        (Ok(_), Ok(_)) => {}
        (Err(e), _) | (_, Err(e)) => {
            return PromotionReadiness::NeedsPrivileges {
                reason: format!("the staged and installed paths could not be compared: {e}"),
            };
        }
    }

    PromotionReadiness::Ready
}

/// Which filesystem a path lives on.
#[cfg(target_os = "macos")]
fn device_of(path: &Path) -> std::io::Result<u64> {
    use std::os::unix::fs::MetadataExt;
    Ok(std::fs::metadata(path)?.dev() as u64)
}

/// Whether a sticky parent still permits renaming `entry` out of it.
#[cfg(target_os = "macos")]
fn sticky_verdict(parent: &Path, entry: &Path) -> std::io::Result<bool> {
    use std::os::unix::fs::MetadataExt;

    let parent_meta = std::fs::metadata(parent)?;
    if parent_meta.mode() & 0o1000 == 0 {
        return Ok(true); // not sticky; the earlier access(2) answer stands
    }
    let entry_meta = std::fs::symlink_metadata(entry)?;
    // SAFETY: `geteuid` reads process state and cannot fail.
    let euid = unsafe { libc::geteuid() };
    Ok(sticky_permits_rename(
        parent_meta.uid(),
        entry_meta.uid(),
        euid,
    ))
}

/// The sticky-directory rule itself, as arithmetic rather than as a
/// filesystem: an entry in a sticky directory may be renamed by the
/// owner of the entry, the owner of the directory, or root.
#[cfg(target_os = "macos")]
fn sticky_permits_rename(parent_uid: u32, entry_uid: u32, euid: u32) -> bool {
    euid == 0 || euid == entry_uid || euid == parent_uid
}

/// Can this user replace an entry inside `dir`?
///
/// That is write plus search on the directory: renaming needs to resolve
/// the name and to modify the directory itself. Asked of the kernel rather
/// than of the mode bits, because `Permissions::readonly()` reports
/// whether *any* write bit is set and says nothing about whether **this**
/// user may write — a root-owned `/Applications` is mode 0755 and would
/// read as writable to everyone. `access(2)` accounts for ownership,
/// groups and ACLs.
#[cfg(target_os = "macos")]
fn renamable_within(dir: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;

    let Ok(c_path) = std::ffi::CString::new(dir.as_os_str().as_bytes()) else {
        return false;
    };
    // SAFETY: `c_path` is a valid NUL-terminated string that outlives the
    // call, and `access` only reads it.
    unsafe { libc::access(c_path.as_ptr(), libc::W_OK | libc::X_OK) == 0 }
}

#[cfg(test)]
mod path_pinning_tests {
    use super::*;

    /// Changing the working directory is process-wide, so the tests that
    /// need to are serialized against each other. Everything else in this
    /// crate's tests names absolute paths, which is why a brief change is
    /// safe here at all.
    static CWD: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn a_relative_staging_root_resolves_before_anything_is_created() {
        let _guard = CWD.lock().unwrap_or_else(|e| e.into_inner());
        let workdir = tempfile::tempdir().unwrap();
        let previous = std::env::current_dir().unwrap();
        std::env::set_current_dir(workdir.path()).unwrap();

        let single = absolutize_staging_root(Path::new("staging"));
        let nested = absolutize_staging_root(Path::new("./deeper/staging"));
        std::fs::create_dir("deeper").unwrap();
        let nested_now = absolutize_staging_root(Path::new("./deeper/staging"));

        std::env::set_current_dir(&previous).unwrap();

        let single = single.expect("a single-component relative root has the cwd as its parent");
        assert!(
            single.is_absolute(),
            "a relative staging root re-resolves later unless it is pinned now: {}",
            single.display()
        );
        assert!(single.ends_with("staging"));

        assert!(
            nested.is_err(),
            "a parent that does not exist is the caller's to create"
        );
        assert!(
            nested_now.expect("the parent exists now").is_absolute(),
            "a nested relative root pins too"
        );
    }

    #[test]
    fn a_single_component_install_path_is_not_a_privilege_problem() {
        // `Path::parent` of a single-component path is `Some("")`, and the
        // kernel rejects an empty pathname — which reported a perfectly
        // writable directory as needing privileges.
        assert_eq!(
            Path::new("Eidola.app").parent().map(Path::to_path_buf),
            Some(PathBuf::new()),
            "this is the shape the probe has to survive"
        );

        let _guard = CWD.lock().unwrap_or_else(|e| e.into_inner());
        let workdir = tempfile::tempdir().unwrap();
        let previous = std::env::current_dir().unwrap();
        std::env::set_current_dir(workdir.path()).unwrap();
        std::fs::create_dir("Eidola.app").unwrap();

        let verdict = promotion_readiness(Path::new("Eidola.app"), Path::new("Eidola.app"));

        std::env::set_current_dir(&previous).unwrap();
        assert_eq!(verdict, PromotionReadiness::Ready, "{verdict:?}");
    }
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
            promotion_readiness(&missing, &missing),
            PromotionReadiness::NotInstalled
        );

        let installed = root.path().join("Eidola.app");
        std::fs::create_dir(&installed).unwrap();
        assert_eq!(
            promotion_readiness(&installed, &installed),
            PromotionReadiness::Ready
        );

        // Root bypasses the permission check entirely, so the negative
        // case only means something as an ordinary user.
        // SAFETY: `geteuid` reads process state and cannot fail.
        if unsafe { libc::geteuid() } != 0 {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o555)).unwrap();
            let verdict = promotion_readiness(&installed, &installed);
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

    /// `rename` does not cross filesystems, whatever the permissions say,
    /// so a staged bundle on a different volume from the install location
    /// cannot be promoted by renaming — it fails `EXDEV`.
    ///
    /// Needs a second writable filesystem to mean anything. Verified
    /// against a RAM disk during development: the rename returns
    /// `CrossesDevices`, this probe reports `DifferentFilesystem`, and a
    /// probe that asked only about permissions said `Ready`.
    #[test]
    fn a_staged_bundle_on_another_filesystem_cannot_be_renamed_into_place() {
        let Some(elsewhere) = another_writable_filesystem() else {
            eprintln!(
                "skipped: no second writable filesystem is mounted, so the cross-device \
                 case cannot be exercised here"
            );
            return;
        };

        let staged = elsewhere.join("eidola-promotion-probe.app");
        if std::fs::create_dir_all(&staged).is_err() {
            eprintln!("skipped: could not stage on {}", elsewhere.display());
            return;
        }

        let here = tempfile::tempdir().unwrap();
        let installed = here.path().join("Eidola.app");
        std::fs::create_dir(&installed).unwrap();

        let verdict = promotion_readiness(&staged, &installed);
        let _ = std::fs::remove_dir_all(&staged);

        assert!(
            matches!(verdict, PromotionReadiness::DifferentFilesystem { .. }),
            "a staged bundle on another filesystem is not promotable by rename: {verdict:?}"
        );
    }

    /// A mounted filesystem that is not the one temporary directories live
    /// on, and that this user can write to.
    fn another_writable_filesystem() -> Option<PathBuf> {
        use std::os::unix::fs::MetadataExt;

        let here_dev = std::fs::metadata(std::env::temp_dir()).ok()?.dev();
        for entry in std::fs::read_dir("/Volumes").ok()? {
            let Ok(entry) = entry else { continue };
            let path = entry.path();
            let Ok(meta) = std::fs::metadata(&path) else {
                continue;
            };
            if !meta.is_dir() || meta.dev() == here_dev {
                continue;
            }
            if renamable_within(&path) {
                return Some(path);
            }
        }
        None
    }

    #[test]
    fn the_sticky_rule_is_the_kernel_s_rule() {
        // An entry in a sticky directory may be renamed by whoever owns
        // the entry, whoever owns the directory, or root — and by nobody
        // else, however writable the directory is.
        assert!(sticky_permits_rename(0, 501, 501), "the entry's owner may");
        assert!(
            sticky_permits_rename(501, 0, 501),
            "the directory's owner may"
        );
        assert!(sticky_permits_rename(0, 0, 0), "root may");
        assert!(
            !sticky_permits_rename(0, 0, 501),
            "a user who owns neither may not, whatever the mode bits say"
        );
    }

    #[test]
    fn a_sticky_directory_this_user_owns_still_permits_promotion() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        let parent = root.path().join("sticky");
        let installed = parent.join("Eidola.app");
        std::fs::create_dir_all(&installed).unwrap();
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o1777)).unwrap();

        assert_eq!(
            promotion_readiness(&installed, &installed),
            PromotionReadiness::Ready,
            "the sticky rule is about ownership, and this user owns both"
        );
    }

    #[test]
    fn a_read_only_bundle_in_a_writable_directory_is_promotable() {
        // The swap renames entries in the parent; it never writes inside
        // the old bundle. Requiring the bundle to be writable refuses the
        // ordinary case — an `.app` whose contents are read-only, which is
        // what a reconstructed one looks like.
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        let installed = root.path().join("Eidola.app");
        std::fs::create_dir(&installed).unwrap();
        std::fs::write(installed.join("Info.plist"), b"x").unwrap();
        std::fs::set_permissions(&installed, std::fs::Permissions::from_mode(0o555)).unwrap();

        let verdict = promotion_readiness(&installed, &installed);
        std::fs::set_permissions(&installed, std::fs::Permissions::from_mode(0o755)).unwrap();

        assert_eq!(
            verdict,
            PromotionReadiness::Ready,
            "a read-only bundle under a writable directory can still be renamed aside"
        );
    }

    #[test]
    fn a_writable_bundle_in_a_read_only_directory_is_not() {
        // SAFETY: `geteuid` reads process state and cannot fail.
        if unsafe { libc::geteuid() } == 0 {
            return; // root renames anything.
        }
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        let parent = root.path().join("Applications");
        let installed = parent.join("Eidola.app");
        std::fs::create_dir_all(&installed).unwrap();
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o555)).unwrap();

        let verdict = promotion_readiness(&installed, &installed);
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o755)).unwrap();

        assert!(
            matches!(verdict, PromotionReadiness::NeedsPrivileges { .. }),
            "nothing can be renamed into a directory this user cannot write: {verdict:?}"
        );
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

        let verdict = promotion_readiness(&installed, &installed);
        std::fs::set_permissions(&outer, std::fs::Permissions::from_mode(0o755)).unwrap();

        assert!(
            matches!(verdict, PromotionReadiness::NeedsPrivileges { .. }),
            "an ancestor this process cannot search means the answer is unknown, and an \
             unknown answer is not `NotInstalled`: {verdict:?}"
        );
    }
}
