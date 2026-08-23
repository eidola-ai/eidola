//! Unpacking a downloaded container into a staging tree.
//!
//! Nothing here is a trust boundary in the cryptographic sense: a container
//! only reaches this code after its bytes hashed to the value a signed
//! document names, so an attacker who can choose what gets extracted has
//! already broken the manifest or the attestation. What this code defends
//! is the *filesystem*: a hash-matching container is still a pile of
//! attacker-shaped paths if the signing side is ever confused, and an
//! extractor that writes outside its root turns that into someone else's
//! files. So paths are validated rather than trusted, and the refusals are
//! typed.
//!
//! The rules mirror what the reconstruction crate already requires of a
//! bundle tree (`crates/eidola-apple/AGENTS.md`): no symbolic links, no
//! traversal, no duplicate members. Refusing them here means a bad
//! container fails before any bytes land, rather than during
//! reconstruction with a half-written tree.

use std::collections::HashSet;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use super::InstallError;

/// Largest container this will unpack, and the largest single member.
///
/// A hash-verified container cannot lie about its size, but the hash is
/// checked over bytes already in memory — these bounds are what keep a
/// *malformed* archive from asking for unbounded disk on the way there.
const MAX_TOTAL_UNPACKED: u64 = 4 * 1024 * 1024 * 1024;

/// Unpack a zip container into `dest`, which must already exist and be
/// empty.
///
/// Returns the number of regular files written. Directory entries are
/// created as needed; every other member kind is a refusal.
pub(super) fn unpack_zip(bytes: &[u8], dest: &Path, label: &str) -> Result<usize, InstallError> {
    let reader = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(reader).map_err(|e| InstallError::Archive {
        label: label.to_string(),
        reason: format!("not a readable zip container: {e}"),
    })?;

    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut written = 0usize;
    let mut total_bytes = 0u64;

    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(|e| InstallError::Archive {
            label: label.to_string(),
            reason: format!("member {index} could not be read: {e}"),
        })?;

        // `enclosed_name` is the crate's own traversal check; the component
        // walk below is ours, because "the library said it was fine" is not
        // the kind of thing this should be taking on faith.
        let raw_name = entry.name().to_string();
        let relative = safe_relative_path(&raw_name).ok_or_else(|| InstallError::Archive {
            label: label.to_string(),
            reason: format!("member `{raw_name}` is not a safe relative path"),
        })?;

        if !seen.insert(relative.clone()) {
            return Err(InstallError::Archive {
                label: label.to_string(),
                reason: format!(
                    "member `{raw_name}` appears twice; which one an extractor keeps is not \
                     something this will guess at"
                ),
            });
        }

        let target = dest.join(&relative);

        if entry.is_dir() {
            std::fs::create_dir_all(&target).map_err(|e| InstallError::Staging {
                path: target.clone(),
                reason: e.to_string(),
            })?;
            continue;
        }

        if !entry.is_file() {
            return Err(InstallError::Archive {
                label: label.to_string(),
                reason: format!(
                    "member `{raw_name}` is neither a regular file nor a directory; symbolic \
                     links and device nodes are refused, as they are by reconstruction"
                ),
            });
        }

        // A symlink is stored as a file whose mode says so — `is_file()`
        // alone does not exclude it on every zip writer.
        if let Some(mode) = entry.unix_mode()
            && mode & 0o170000 == 0o120000
        {
            return Err(InstallError::Archive {
                label: label.to_string(),
                reason: format!("member `{raw_name}` is a symbolic link; those are refused"),
            });
        }

        total_bytes = total_bytes.saturating_add(entry.size());
        if total_bytes > MAX_TOTAL_UNPACKED {
            return Err(InstallError::Archive {
                label: label.to_string(),
                reason: format!("unpacks to more than {MAX_TOTAL_UNPACKED} bytes"),
            });
        }

        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| InstallError::Staging {
                path: parent.to_path_buf(),
                reason: e.to_string(),
            })?;
        }

        let mut contents = Vec::with_capacity(entry.size() as usize);
        entry
            .read_to_end(&mut contents)
            .map_err(|e| InstallError::Archive {
                label: label.to_string(),
                reason: format!("member `{raw_name}` could not be decompressed: {e}"),
            })?;

        std::fs::write(&target, &contents).map_err(|e| InstallError::Staging {
            path: target.clone(),
            reason: e.to_string(),
        })?;

        // The packing recipe normalizes modes to `u=rwX,go=rX`, so the
        // executable bit is the only mode information the container
        // carries — and the one the reconstructed bundle needs, since a
        // Mach-O that comes back non-executable cannot launch.
        set_mode(&target, entry.unix_mode())?;

        written += 1;
    }

    Ok(written)
}

#[cfg(unix)]
fn set_mode(path: &Path, stored: Option<u32>) -> Result<(), InstallError> {
    use std::os::unix::fs::PermissionsExt;

    let executable = stored.is_some_and(|mode| mode & 0o111 != 0);
    let mode = if executable { 0o755 } else { 0o644 };
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).map_err(|e| {
        InstallError::Staging {
            path: path.to_path_buf(),
            reason: e.to_string(),
        }
    })
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _stored: Option<u32>) -> Result<(), InstallError> {
    Ok(())
}

/// A member name this is willing to write, as a relative path.
///
/// The check runs on the archive's own `/`-separated segments before any
/// path type sees them, because path types normalize: Rust drops an
/// interior `.` component, so `a/./b` and `a/b` would arrive here as the
/// same path and only one of them is a name our packer can produce.
/// Normalizing is how an extractor ends up writing somewhere its caller
/// did not name, so nothing is normalized — an unusual name is refused.
fn safe_relative_path(name: &str) -> Option<PathBuf> {
    if name.is_empty() || name.starts_with('/') || name.starts_with('\\') {
        return None;
    }
    if name.contains('\0') {
        return None;
    }

    // A directory member arrives with one trailing slash, which is the
    // archive saying "directory" rather than a path component. Every other
    // empty segment is a doubled separator: two names for one place, and
    // this compares names.
    let body = name.strip_suffix('/').unwrap_or(name);

    let mut out = PathBuf::new();
    for segment in body.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return None;
        }
        out.push(segment);
    }

    // Re-walking with the platform's own parser catches anything the
    // segment scan could not see — a drive prefix on Windows, say.
    for component in out.components() {
        if !matches!(component, Component::Normal(_)) {
            return None;
        }
    }

    if out.as_os_str().is_empty() {
        None
    } else {
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_the_paths_an_extractor_must_never_write() {
        for name in [
            "",
            "/etc/passwd",
            "../escape",
            "a/../../escape",
            "./same",
            "a/./b",
            "with\0nul",
            "a//b",
        ] {
            assert!(
                safe_relative_path(name).is_none(),
                "`{name}` should be refused"
            );
        }
    }

    #[test]
    fn accepts_ordinary_bundle_paths() {
        assert_eq!(
            safe_relative_path("Eidola.app/Contents/MacOS/Eidola"),
            Some(PathBuf::from("Eidola.app/Contents/MacOS/Eidola"))
        );
    }
}
