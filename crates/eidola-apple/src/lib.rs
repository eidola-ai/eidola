//! Cross-platform detached Apple code-signature reconstruction and inspection.

mod apply;
mod codesign;
mod detach;
mod format;
mod fs_guard;
mod inspect;
mod macho;

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub use format::{
    CodeSignatureFacts, LinkeditFacts, MachOEntry, MachOKind, PlacementRecord, SliceFacts,
};
pub use fs_guard::GuardError;

/// Structurally parsed claims from the main executable's embedded signature.
///
/// These facts are not authenticated Apple identity or trust results. Callers
/// must compose them with an independently authenticated release statement.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SignatureFacts {
    pub team_id: Option<String>,
    pub identifier: String,
    pub hardened_runtime: bool,
    pub entitlements_sha256: Option<String>,
    pub has_notarization_ticket: bool,
}

/// A failure while reconstructing a bundle in place.
#[derive(Debug, thiserror::Error)]
pub enum ApplyError {
    #[error("unsafe bundle or detached path: {0}")]
    UnsafePath(#[from] fs_guard::GuardError),
    #[error("failed to read `{path}`: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse placement record `{path}`: {source}")]
    RecordJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid placement record `{path}`: {reason}")]
    InvalidRecord { path: PathBuf, reason: String },
    #[error(
        "`{path}` is not the build this signature was detached from (expected {expected}, got {actual})"
    )]
    WrongInput {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("unsigned bundle input `{path}` is missing")]
    UnsignedInputMissing { path: PathBuf },
    #[error("unsigned bundle contains unexpected input `{path}`")]
    UnsignedInputUnexpected { path: PathBuf },
    #[error(
        "unsigned bundle input `{path}` does not match its record (expected {expected}, got {actual})"
    )]
    UnsignedInputHash {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("invalid Mach-O `{path}`: {reason}")]
    InvalidMachO { path: PathBuf, reason: String },
    #[error(
        "`{path}` slice {arch} carries no LC_CODE_SIGNATURE; placement rewrites that load command, it does not insert one"
    )]
    UnsignedSlice { path: PathBuf, arch: String },
    #[error("`{path}` slice {arch} cannot use the recorded placement: {reason}")]
    Placement {
        path: PathBuf,
        arch: String,
        reason: String,
    },
    #[error("detached signature for `{path}` slice {arch} is invalid: {reason}")]
    DetachedSignature {
        path: PathBuf,
        arch: String,
        reason: String,
    },
    #[error(
        "reconstructed `{path}` does not match the signed output (expected {expected_len} bytes hashing {expected_hash}, got {actual_len} bytes hashing {actual_hash})"
    )]
    OutputMismatch {
        path: PathBuf,
        expected_len: u64,
        expected_hash: String,
        actual_len: u64,
        actual_hash: String,
    },
    #[error("detached file `{path}` does not match its record (expected {expected}, got {actual})")]
    DetachedFileHash {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("detached material contains unexpected file `{path}`")]
    DetachedInputUnexpected { path: PathBuf },
    #[error("detached material path `{path}` is invalid: {reason}")]
    DetachedInputInvalid { path: PathBuf, reason: String },
    #[error("plain-file mutation target `{path}` is incompatible: {reason}")]
    PlainFileTarget { path: PathBuf, reason: String },
    #[error("failed to write `{path}`: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// A failure while detaching a signed bundle.
#[derive(Debug, thiserror::Error)]
pub enum DetachError {
    #[error("unsafe bundle or detached path: {0}")]
    UnsafePath(#[from] fs_guard::GuardError),
    #[error("failed to read `{path}`: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid Mach-O `{path}`: {reason}")]
    InvalidMachO { path: PathBuf, reason: String },
    #[error("`{path}` slice {arch} carries no LC_CODE_SIGNATURE")]
    UnsignedSlice { path: PathBuf, arch: String },
    #[error("cannot detach `{path}` slice {arch}: {reason}")]
    InvalidSignature {
        path: PathBuf,
        arch: String,
        reason: String,
    },
    #[error("unsigned input `{path}` slice {arch} is incompatible: {reason}")]
    IncompatibleInput {
        path: PathBuf,
        arch: String,
        reason: String,
    },
    #[error("invalid detach destination `{path}`: {reason}")]
    InvalidDestination { path: PathBuf, reason: String },
    #[error("failed to write `{path}`: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to encode placement record: {0}")]
    RecordJson(#[from] serde_json::Error),
}

/// A failure while reading claims from a bundle's embedded signature.
#[derive(Debug, thiserror::Error)]
pub enum InspectError {
    #[error("unsafe bundle path: {0}")]
    UnsafePath(#[from] fs_guard::GuardError),
    #[error("failed to read `{path}`: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse `{path}`: {reason}")]
    Plist { path: PathBuf, reason: String },
    #[error("invalid Mach-O `{path}`: {reason}")]
    InvalidMachO { path: PathBuf, reason: String },
    #[error("`{path}` slice {arch} carries no LC_CODE_SIGNATURE")]
    UnsignedSlice { path: PathBuf, arch: String },
    #[error("invalid embedded signature in `{path}` slice {arch}: {reason}")]
    InvalidSignature {
        path: PathBuf,
        arch: String,
        reason: String,
    },
    #[error("signature claims disagree between slices of `{path}`: {reason}")]
    SliceMismatch { path: PathBuf, reason: String },
}

/// Reconstruct a signed bundle in place from a detached signature directory.
///
/// Both roots must be privately staged and not concurrently modified. Static
/// symbolic links are refused, but this API does not defend against a
/// same-privilege process racing filesystem validation. Input validation
/// finishes before mutation preparation. Preparation may make mutation targets
/// and parent directories writable or create empty signing directories; if it
/// fails, file contents have not changed.
pub fn apply(unsigned_bundle: &Path, detached: &Path) -> Result<(), ApplyError> {
    apply::apply(unsigned_bundle, detached)
}

/// Lift a signed bundle's embedded signatures into the detached layout.
///
/// All roots must be privately staged and not concurrently modified.
pub fn detach(
    signed_bundle: &Path,
    unsigned_bundle: &Path,
    output_dir: &Path,
) -> Result<PathBuf, DetachError> {
    detach::detach(signed_bundle, unsigned_bundle, output_dir)
}

/// Structurally parse embedded signing claims without invoking platform tools.
///
/// This does not authenticate an Apple identity or establish Apple trust.
pub fn inspect(bundle: &Path) -> Result<SignatureFacts, InspectError> {
    inspect::inspect(bundle)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}
