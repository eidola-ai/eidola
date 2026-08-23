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

    let mut seen: HashSet<String> = HashSet::new();
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

        if !seen.insert(collision_key(&relative)) {
            return Err(InstallError::Archive {
                label: label.to_string(),
                reason: format!(
                    "member `{raw_name}` names the same file as an earlier member; which one \
                     an extractor keeps is not something this will guess at"
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

        // `entry.size()` is the container's own claim about the member.
        // It is worth refusing early on, but it is not what the budget is
        // spent against — that is counted below, on bytes actually
        // written, so a member that understates itself cannot overrun the
        // limit by lying.
        if total_bytes.saturating_add(entry.size()) > MAX_TOTAL_UNPACKED {
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

        // Copied to disk in fixed-size pieces rather than decompressed
        // whole. A member is compressed data until it is not: one that
        // expands to gigabytes would otherwise be allocated entire, while
        // the container is still held, and the process would be killed
        // instead of returning a refusal. The budget is spent here, on
        // bytes that actually arrived.
        let written_bytes =
            copy_member(&mut entry, &target, &mut total_bytes).inspect_err(|_| {
                // A partially written member is not left where a later step
                // could mistake it for a whole one. The staging tree is
                // removed on any failure too; this keeps the invariant local.
                let _ = std::fs::remove_file(&target);
            })?;
        if written_bytes > entry.size() {
            let _ = std::fs::remove_file(&target);
            return Err(InstallError::Archive {
                label: label.to_string(),
                reason: format!("member `{raw_name}` decompressed to more than it declared"),
            });
        }

        // The packing recipe normalizes modes to `u=rwX,go=rX`, so the
        // executable bit is the only mode information the container
        // carries — and the one the reconstructed bundle needs, since a
        // Mach-O that comes back non-executable cannot launch.
        set_mode(&target, entry.unix_mode())?;

        written += 1;
    }

    Ok(written)
}

/// Stream one member onto disk, spending `budget_used` as it goes.
///
/// Returns the number of bytes written.
fn copy_member(
    entry: &mut zip::read::ZipFile<'_, std::io::Cursor<&[u8]>>,
    target: &Path,
    budget_used: &mut u64,
) -> Result<u64, InstallError> {
    let staging_error = |path: &Path, e: std::io::Error| InstallError::Staging {
        path: path.to_path_buf(),
        reason: e.to_string(),
    };

    let mut file = std::fs::File::create(target).map_err(|e| staging_error(target, e))?;
    let mut buffer = vec![0u8; 64 * 1024];
    let mut written = 0u64;

    loop {
        let read = entry.read(&mut buffer).map_err(|e| InstallError::Archive {
            label: target.display().to_string(),
            reason: format!("could not be decompressed: {e}"),
        })?;
        if read == 0 {
            break;
        }
        *budget_used = budget_used.saturating_add(read as u64);
        if *budget_used > MAX_TOTAL_UNPACKED {
            return Err(InstallError::Archive {
                label: target.display().to_string(),
                reason: format!("unpacks to more than {MAX_TOTAL_UNPACKED} bytes"),
            });
        }
        std::io::Write::write_all(&mut file, &buffer[..read])
            .map_err(|e| staging_error(target, e))?;
        written += read as u64;
    }

    Ok(written)
}

/// The name two members would have to share to be one file.
///
/// A `PathBuf` comparison is not that test on the filesystems this
/// unpacks onto. Measured on APFS rather than recalled:
///
/// * a case-insensitive volume — the macOS default — makes `A.app/x` and
///   `a.app/x` one file;
/// * APFS is normalization-insensitive *independently of case*, so NFC
///   `café.txt` and its NFD spelling are one entry on a case-sensitive
///   volume too, and the second write wins.
///
/// Case is folded here. Normalization is not: comparing it faithfully
/// needs a Unicode normalization table, and guessing at it would be the
/// same mistake as guessing at an escape. Instead [`safe_relative_path`]
/// refuses non-ASCII names outright, which is what makes ASCII folding a
/// complete answer rather than a partial one — and costs nothing, because
/// the packing recipe produces ASCII bundle paths. A payload that ever
/// needs a non-ASCII name needs that table and a decision, not this
/// function quietly widening.
fn collision_key(relative: &Path) -> String {
    relative.to_string_lossy().to_ascii_lowercase()
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

    // Non-ASCII is refused rather than normalized — see `collision_key`.
    if !name.is_ascii() {
        return None;
    }

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

/// A name that must be exactly one directory component — the bundle
/// directory an install expects to find inside a container.
///
/// Same refusals as a member name, then one more: a single component. A
/// plan naming `../..` or an absolute path would otherwise send everything
/// downstream — including reconstruction, which *writes* — outside the
/// staging tree this module promises never to leave.
pub(super) fn single_component(name: &str) -> Option<PathBuf> {
    let path = safe_relative_path(name)?;
    let mut components = path.components();
    let first = components.next()?;
    if components.next().is_some() {
        return None;
    }
    match first {
        Component::Normal(_) => Some(path),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A member that decompresses to more than the container's own
    /// central directory claims. The budget is spent on bytes that
    /// arrive, so understating a member cannot buy extra room; a check
    /// against the declared size alone would let it through.
    #[test]
    fn refuses_a_member_that_decompresses_past_its_declared_size() {
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut writer = zip::ZipWriter::new(&mut cursor);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated)
                .unix_permissions(0o644);
            writer.start_file("big.bin".to_string(), options).unwrap();
            std::io::Write::write_all(&mut writer, &vec![0u8; 256 * 1024]).unwrap();
            writer.finish().unwrap();
        }
        let mut bytes = cursor.into_inner();

        // Rewrite every recorded uncompressed size to a small lie. The
        // reader believes the container; the extractor counts.
        let truthful = (256u32 * 1024).to_le_bytes();
        let lie = 16u32.to_le_bytes();
        let mut patched = 0;
        for i in 0..bytes.len().saturating_sub(4) {
            if bytes[i..i + 4] == truthful {
                bytes[i..i + 4].copy_from_slice(&lie);
                patched += 1;
            }
        }
        assert!(patched >= 1, "the fixture should record its size somewhere");

        let dest = tempfile::tempdir().unwrap();
        let outcome = unpack_zip(&bytes, dest.path(), "a payload");
        assert!(
            outcome.is_err(),
            "a member that outgrows what the container declared must be refused"
        );
        assert!(
            !dest.path().join("big.bin").exists(),
            "a refused member must not be left half-written"
        );
    }

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
    fn refuses_names_it_cannot_compare_faithfully() {
        // APFS folds these onto one file; this refuses rather than guess
        // which spelling the archive meant.
        assert!(safe_relative_path("caf\u{e9}.txt").is_none());
        assert!(safe_relative_path("cafe\u{301}.txt").is_none());
    }

    #[test]
    fn collision_key_sees_what_the_filesystem_sees() {
        assert_eq!(
            collision_key(Path::new("A.app/Contents/Info.plist")),
            collision_key(Path::new("a.app/CONTENTS/info.plist"))
        );
    }

    #[test]
    fn a_bundle_name_is_one_component_or_nothing() {
        assert_eq!(
            single_component("Eidola.app"),
            Some(PathBuf::from("Eidola.app"))
        );
        for name in ["", "..", "../escape", "/Applications/Evil.app", "a/b", "."] {
            assert!(
                single_component(name).is_none(),
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
