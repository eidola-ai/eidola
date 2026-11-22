//! Which release is installed, as the updater recorded it.
//!
//! No build knows the commit it was released from. The release commit
//! records the binaries' hashes in `artifact-manifest.json`, so a binary
//! that embedded that commit could never reach a fixed point. Continuity
//! (`super::gate_release`) needs the installed commit all the same, so the
//! updater writes it down when it installs a release, and reads it back on
//! every check.
//!
//! **The record is keyed by the bytes it describes.** Each entry names the
//! SHA-256 of the install's main executable
//! ([`super::install_kind::InstallKind::main_executable`]) as the updater
//! installed it, and it is believed only while that file still hashes to
//! that value. Anything that replaces the executable behind the updater's
//! back, such as a person installing a download over the top or a package
//! manager downgrading it, changes the hash, and the install then reads as a
//! fresh one: no commit known, continuity skipped, exactly as before the
//! first update. The record cannot vouch for bytes it was not written for.
//!
//! The record lives in the profile's data directory, because that is the
//! one place every user can write to. An install can be shared by several
//! profiles (`/Applications` serves every account on the Mac), and each
//! profile's record speaks only for the bytes it saw installed, so a profile
//! whose record is stale reads the install as fresh rather than wrongly.
//!
//! The file is unsigned and in the user's own directory. Anything that can
//! write it can also replace the executable, so it is not a trust input in
//! its own right: it can only switch continuity on (by naming a commit) or
//! off (by naming none), and the release it then admits still has to pass
//! every signature check.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::Digest;

use crate::error::AppError;

/// How many installs one profile remembers. A profile is normally used by
/// one install, sometimes two (an app bundle and a static CLI on the same
/// machine, or a second copy of the app); older entries are dropped first.
const MAX_ENTRIES: usize = 8;

const RECORD_FILE: &str = "installed-release.json";

/// One install the updater wrote.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledRelease {
    /// Lowercase hex SHA-256 of the install's main executable, as installed.
    pub executable_sha256: String,
    /// The release's version, as its `release.json` named it.
    pub version: String,
    /// The commit the release was cut from.
    pub git_commit: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Record {
    /// Most recent first.
    installs: Vec<InstalledRelease>,
}

pub fn record_path(data_dir: &Path) -> PathBuf {
    data_dir.join(RECORD_FILE)
}

fn load(data_dir: &Path) -> Record {
    std::fs::read(record_path(data_dir))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

/// The commit of the release installed at `executable_sha256`, if the
/// updater installed exactly those bytes as `version`.
///
/// The version has to agree as well as the hash: the bytes carry their own
/// version (compiled in), so an entry naming another version for them was
/// written wrongly and says nothing.
pub fn installed_commit(data_dir: &Path, executable_sha256: &str, version: &str) -> Option<String> {
    load(data_dir)
        .installs
        .into_iter()
        .find(|e| e.executable_sha256.eq_ignore_ascii_case(executable_sha256))
        .filter(|e| e.version == version)
        .map(|e| e.git_commit)
}

/// Record that `entry` is now installed, replacing any earlier entry for the
/// same bytes. Written to a sibling file and renamed into place, so a
/// reader never sees half a record.
pub fn record_install(data_dir: &Path, entry: InstalledRelease) -> Result<(), AppError> {
    let mut record = load(data_dir);
    record.installs.retain(|e| {
        !e.executable_sha256
            .eq_ignore_ascii_case(&entry.executable_sha256)
    });
    record.installs.insert(0, entry);
    record.installs.truncate(MAX_ENTRIES);

    let fail = |what: &str, e: &dyn std::fmt::Display| AppError::Update {
        message: format!("{what} the installed-release record: {e}"),
    };
    std::fs::create_dir_all(data_dir).map_err(|e| fail("creating the directory for", &e))?;
    let bytes = serde_json::to_vec_pretty(&record).map_err(|e| fail("serializing", &e))?;
    let path = record_path(data_dir);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, bytes).map_err(|e| fail("writing", &e))?;
    std::fs::rename(&tmp, &path).map_err(|e| fail("replacing", &e))
}

/// Lowercase hex SHA-256 of the file at `path`, streamed.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(super::human_attestation::hex_encode(&hasher.finalize()))
}

/// The commit of the release this process belongs to, when the updater
/// installed it: [`super::install_kind::detect`] names the install's main
/// executable, its hash looks the record up, and this build's own version
/// has to match the entry.
///
/// Hashing an executable reads tens of megabytes, so the hash is taken once
/// per process and per path. Blocking; call it off any async executor's
/// worker threads.
#[cfg(feature = "desktop")]
pub fn running_install_commit(data_dir: &Path) -> Option<String> {
    let program = super::install_kind::running_program();
    let main = super::install_kind::detect(program).main_executable()?;
    let sha256 = cached_sha256(&main)?;
    installed_commit(data_dir, &sha256, env!("CARGO_PKG_VERSION"))
}

/// The hash of `path`, taken at most once per process. A file replaced
/// while this process runs keeps its first answer, which is the right one:
/// the question is what *this* process is, and a replacement is a different
/// release that the next launch will hash.
#[cfg(feature = "desktop")]
fn cached_sha256(path: &Path) -> Option<String> {
    static CACHE: std::sync::Mutex<Vec<(PathBuf, Option<String>)>> =
        std::sync::Mutex::new(Vec::new());
    let mut cache = CACHE.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((_, hash)) = cache.iter().find(|(p, _)| p == path) {
        return hash.clone();
    }
    let hash = sha256_file(path).ok();
    cache.push((path.to_path_buf(), hash.clone()));
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(sha: &str, version: &str, commit: &str) -> InstalledRelease {
        InstalledRelease {
            executable_sha256: sha.into(),
            version: version.into(),
            git_commit: commit.into(),
        }
    }

    #[test]
    fn nothing_recorded_is_a_fresh_install() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(installed_commit(dir.path(), "aa", "0.1.0"), None);
    }

    #[test]
    fn the_record_speaks_for_the_bytes_it_was_written_for() {
        let dir = tempfile::tempdir().unwrap();
        record_install(dir.path(), entry("aa", "0.1.0", "c1")).unwrap();
        assert_eq!(
            installed_commit(dir.path(), "AA", "0.1.0").as_deref(),
            Some("c1")
        );
        // A manual install over the top: other bytes, so a fresh install.
        assert_eq!(installed_commit(dir.path(), "bb", "0.1.0"), None);
    }

    #[test]
    fn an_entry_naming_another_version_for_the_bytes_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        record_install(dir.path(), entry("aa", "0.1.0", "c1")).unwrap();
        assert_eq!(installed_commit(dir.path(), "aa", "0.1.1"), None);
    }

    #[test]
    fn a_new_install_of_the_same_bytes_replaces_the_old_entry() {
        let dir = tempfile::tempdir().unwrap();
        record_install(dir.path(), entry("aa", "0.1.0", "c1")).unwrap();
        record_install(dir.path(), entry("bb", "0.1.1", "c2")).unwrap();
        record_install(dir.path(), entry("aa", "0.1.0", "c3")).unwrap();
        assert_eq!(
            installed_commit(dir.path(), "aa", "0.1.0").as_deref(),
            Some("c3")
        );
        assert_eq!(
            installed_commit(dir.path(), "bb", "0.1.1").as_deref(),
            Some("c2")
        );
        assert_eq!(load(dir.path()).installs.len(), 2);
    }

    #[test]
    fn the_record_keeps_the_most_recent_installs() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..MAX_ENTRIES + 2 {
            record_install(dir.path(), entry(&format!("{i:02x}"), "0.1.0", "c")).unwrap();
        }
        assert_eq!(load(dir.path()).installs.len(), MAX_ENTRIES);
        assert_eq!(installed_commit(dir.path(), "00", "0.1.0"), None);
        assert!(
            installed_commit(dir.path(), &format!("{:02x}", MAX_ENTRIES + 1), "0.1.0").is_some()
        );
    }

    #[test]
    fn an_unreadable_record_is_a_fresh_install() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(record_path(dir.path()), b"{ not json").unwrap();
        assert_eq!(installed_commit(dir.path(), "aa", "0.1.0"), None);
        // And the next install writes over it.
        record_install(dir.path(), entry("aa", "0.1.0", "c1")).unwrap();
        assert_eq!(
            installed_commit(dir.path(), "aa", "0.1.0").as_deref(),
            Some("c1")
        );
    }

    #[test]
    fn a_file_hashes_as_sha256() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("exe");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            sha256_file(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
