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

/// The most members a container may hold.
///
/// A byte budget does not bound member *count*: a million empty files cost
/// almost nothing to store and a great deal to create, in inodes and in
/// the bookkeeping every step after this does per file. A real bundle is
/// hundreds of entries; this is generous by an order of magnitude and
/// still a bound.
const MAX_MEMBERS: usize = 50_000;

/// Unpack a zip container into `dest`, which must already exist and be
/// empty.
///
/// Returns the number of regular files written. Directory entries are
/// created as needed; every other member kind is a refusal.
pub(super) fn unpack_zip(bytes: &[u8], dest: &Path, label: &str) -> Result<usize, InstallError> {
    // Read the count the container claims before handing it to a parser
    // that will believe it. `ZipArchive::new` walks the whole central
    // directory eagerly and retains a record per member, so a container
    // claiming millions of them costs memory before `len()` can be
    // consulted — and the crate exposes no cheaper way to ask.
    //
    // This reads one fixed-layout record at the end of the file rather
    // than parsing the archive: it can only refuse early, never accept
    // something the parser would reject. When it cannot tell — a ZIP64
    // count, or an end record it cannot find — it says so and the parser
    // decides, which is no worse than not looking.
    if let Some(claimed) = claimed_member_count(bytes)
        && claimed > MAX_MEMBERS as u64
    {
        return Err(InstallError::Archive {
            label: label.to_string(),
            reason: format!(
                "claims {claimed} members, more than the {MAX_MEMBERS} this will unpack"
            ),
        });
    }

    let reader = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(reader).map_err(|e| InstallError::Archive {
        label: label.to_string(),
        reason: format!("not a readable zip container: {e}"),
    })?;

    if archive.len() > MAX_MEMBERS {
        return Err(InstallError::Archive {
            label: label.to_string(),
            reason: format!(
                "holds {} members, more than the {MAX_MEMBERS} this will unpack",
                archive.len()
            ),
        });
    }

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

        // What a member *claims to be* is read before anything dispatches
        // on what it looks like. A name ending in `/` makes a member a
        // directory to every zip reader, so a symlink mode behind a
        // trailing slash would otherwise reach the directory branch and
        // never meet the check written to refuse it.
        if let Some(mode) = entry.unix_mode() {
            let kind = mode & 0o170000;
            if kind == 0o120000 {
                return Err(InstallError::Archive {
                    label: label.to_string(),
                    reason: format!("member `{raw_name}` is a symbolic link; those are refused"),
                });
            }
            if kind != 0 && kind != 0o040000 && kind != 0o100000 {
                return Err(InstallError::Archive {
                    label: label.to_string(),
                    reason: format!(
                        "member `{raw_name}` is neither a regular file nor a directory; only \
                         those are unpacked, as reconstruction requires"
                    ),
                });
            }
            // A member says what it is twice: in its name, where a trailing
            // slash means directory, and in its mode. Reading the mode
            // first stops a symlink hiding behind a slash, but the two can
            // still disagree — a directory mode on a plain name, a file
            // mode on a name ending in `/` — and then whichever one the
            // code happens to dispatch on decides what lands. They have to
            // agree, or the member is not something this can unpack.
            let says_directory = kind == 0o040000;
            let named_directory = entry.is_dir();
            if kind != 0 && says_directory != named_directory {
                return Err(InstallError::Archive {
                    label: label.to_string(),
                    reason: format!(
                        "member `{raw_name}` names {} and its mode says {}; a member that \
                         cannot say what it is once is refused",
                        if named_directory {
                            "a directory"
                        } else {
                            "a file"
                        },
                        if says_directory {
                            "a directory"
                        } else {
                            "a file"
                        }
                    ),
                });
            }
        }

        if entry.is_dir() {
            std::fs::create_dir_all(&target).map_err(|e| InstallError::Staging {
                path: target.clone(),
                reason: e.to_string(),
            })?;
            // Directories get the same exact mode set the files do. Left
            // to `create_dir_all` they inherit the umask, so a staging run
            // under 077 produces 0700 directories — and an app promoted
            // from that tree is unreadable to anyone but the installing
            // user, which is not what the packer recorded.
            set_directory_mode(&target)?;
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

    // Directories get the same exact mode set the files do, in one pass at
    // the end so implicitly created parents are covered as surely as
    // explicit members. Left to `create_dir_all` they inherit the umask,
    // so a staging run under 077 produces 0700 directories — and an app
    // promoted from that tree is unreadable to anyone but the installing
    // user, which is not the tree the packer recorded.
    normalize_modes(dest)?;
    Ok(written)
}

/// `u=rwX,go=rX` over a whole tree — the rule the packing recipe applies,
/// which is to say: directories and anything already executable become
/// 0755, every other file 0644.
///
/// Applied to the unpacked tree, and again after reconstruction, because
/// reconstruction creates paths with ordinary creates and those carry the
/// umask. What a staged tree looks like should not depend on which step
/// wrote each path.
pub(super) fn normalize_modes(root: &Path) -> Result<(), InstallError> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).map_err(|e| InstallError::Staging {
            path: dir.clone(),
            reason: e.to_string(),
        })?;
        for entry in entries {
            let entry = entry.map_err(|e| InstallError::Staging {
                path: dir.clone(),
                reason: e.to_string(),
            })?;
            let path = entry.path();
            let kind = entry.file_type().map_err(|e| InstallError::Staging {
                path: path.clone(),
                reason: e.to_string(),
            })?;
            if kind.is_dir() {
                stack.push(path);
            } else if kind.is_file() {
                set_file_mode(&path)?;
            }
            // Anything else was refused on the way in and is not created
            // by reconstruction; leaving it alone is the honest default.
        }
        set_directory_mode(&dir)?;
    }
    Ok(())
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

/// How many members a container's end-of-central-directory record says it
/// holds, when that can be read without parsing anything else.
///
/// The record is the last 22 bytes plus an optional trailing comment, so
/// it is found by scanning back for its signature. `None` means "cannot
/// tell": no record found, or a ZIP64 count that lives in a different
/// record. Nothing is concluded from `None` — it is not evidence of
/// anything, and the authoritative check still runs after the parse.
fn claimed_member_count(bytes: &[u8]) -> Option<u64> {
    const SIGNATURE: [u8; 4] = [b'P', b'K', 5, 6];
    const RECORD_LEN: usize = 22;
    // A zip comment is at most 64 KiB, so the record starts no earlier.
    const MAX_COMMENT: usize = u16::MAX as usize;

    if bytes.len() < RECORD_LEN {
        return None;
    }
    let earliest = bytes.len().saturating_sub(RECORD_LEN + MAX_COMMENT);
    let start = (earliest..=bytes.len() - RECORD_LEN)
        .rev()
        .find(|&i| bytes[i..i + 4] == SIGNATURE)?;

    let total = u16::from_le_bytes([bytes[start + 10], bytes[start + 11]]);
    if total == u16::MAX {
        // The real count is in the ZIP64 record; reading that is parsing.
        return None;
    }
    Some(u64::from(total))
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

/// `u=rwX,go=rX` for a directory — the same rule the packing recipe
/// applies, so the tree that comes out matches the tree that went in.
/// 0755 if it is already executable, 0644 otherwise — `X` in the packer's
/// `u=rwX,go=rX`, which keys off the executable bit rather than inventing
/// one.
#[cfg(unix)]
fn set_file_mode(path: &Path) -> Result<(), InstallError> {
    use std::os::unix::fs::PermissionsExt;

    let mode = std::fs::metadata(path)
        .map_err(|e| InstallError::Staging {
            path: path.to_path_buf(),
            reason: e.to_string(),
        })?
        .permissions()
        .mode();
    let normalized = if mode & 0o111 != 0 { 0o755 } else { 0o644 };
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(normalized)).map_err(|e| {
        InstallError::Staging {
            path: path.to_path_buf(),
            reason: e.to_string(),
        }
    })
}

#[cfg(not(unix))]
fn set_file_mode(_path: &Path) -> Result<(), InstallError> {
    Ok(())
}

#[cfg(unix)]
fn set_directory_mode(path: &Path) -> Result<(), InstallError> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).map_err(|e| {
        InstallError::Staging {
            path: path.to_path_buf(),
            reason: e.to_string(),
        }
    })
}

#[cfg(not(unix))]
fn set_directory_mode(_path: &Path) -> Result<(), InstallError> {
    Ok(())
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

    /// A member whose name ends in `/` — which makes it a directory to
    /// every zip reader — while its mode says symbolic link. Dispatching
    /// on what a member looks like before reading what it claims to be
    /// sends this down the directory branch, past the check written to
    /// refuse it.
    #[test]
    fn refuses_a_symlink_mode_hidden_behind_a_trailing_slash() {
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut writer = zip::ZipWriter::new(&mut cursor);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored)
                .unix_permissions(0o777);
            writer.start_file("app/link/".to_string(), options).unwrap();
            writer.finish().unwrap();
        }
        let mut bytes = cursor.into_inner();

        // The writer masks the file-type bits off, so the symlink mode is
        // patched into the external attributes, where the mode lives in
        // the high half.
        let regular = (0o100777u32 << 16).to_le_bytes();
        let symlink = (0o120777u32 << 16).to_le_bytes();
        let mut patched = 0;
        for i in 0..bytes.len().saturating_sub(4) {
            if bytes[i..i + 4] == regular {
                bytes[i..i + 4].copy_from_slice(&symlink);
                patched += 1;
            }
        }
        assert!(patched >= 1, "the fixture should record a mode somewhere");

        let dest = tempfile::tempdir().unwrap();
        let error = unpack_zip(&bytes, dest.path(), "a payload")
            .expect_err("a symlink is refused however it is spelled");
        assert!(
            matches!(&error, InstallError::Archive { reason, .. } if reason.contains("symbolic link")),
            "{error}"
        );
    }

    /// A member says what it is twice — in its name and in its mode — and
    /// a container where the two disagree is one where whichever the code
    /// dispatches on decides what lands.
    #[test]
    fn refuses_a_member_whose_name_and_mode_disagree() {
        // A file-shaped name carrying a directory mode needs the mode
        // patched in; a directory-shaped name carrying a file mode needs
        // nothing, because the writer stores `S_IFREG` regardless of the
        // trailing slash. Both are read back before use, so the fixture
        // proves it really holds the disagreement it is testing.
        for (name, patch) in [
            ("app/oddity", Some((0o100755u32, 0o040755u32))),
            ("app/oddity/", None),
        ] {
            let mut cursor = std::io::Cursor::new(Vec::new());
            {
                let mut writer = zip::ZipWriter::new(&mut cursor);
                let options = zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored)
                    .unix_permissions(0o755);
                writer.start_file(name.to_string(), options).unwrap();
                writer.finish().unwrap();
            }
            let mut bytes = cursor.into_inner();

            if let Some((from, to)) = patch {
                let from_bytes = (from << 16).to_le_bytes();
                let to_bytes = (to << 16).to_le_bytes();
                let mut patched = 0;
                for i in 0..bytes.len().saturating_sub(4) {
                    if bytes[i..i + 4] == from_bytes {
                        bytes[i..i + 4].copy_from_slice(&to_bytes);
                        patched += 1;
                    }
                }
                assert!(patched >= 1, "`{name}`: the fixture should record a mode");
            }

            // The fixture is only worth anything if the two really differ.
            {
                let mut archive = zip::ZipArchive::new(std::io::Cursor::new(&bytes[..])).unwrap();
                let entry = archive.by_index(0).unwrap();
                let mode_says_dir = entry.unix_mode().unwrap() & 0o170000 == 0o040000;
                assert_ne!(
                    mode_says_dir,
                    entry.is_dir(),
                    "`{name}`: this fixture is supposed to disagree with itself"
                );
            }

            let dest = tempfile::tempdir().unwrap();
            let error = unpack_zip(&bytes, dest.path(), "a payload")
                .expect_err("a member that cannot say what it is once is refused");
            assert!(
                matches!(&error, InstallError::Archive { reason, .. }
                    if reason.contains("cannot say what it is once")),
                "`{name}`: {error}"
            );
        }
    }

    /// The count a container claims is readable without parsing it, and
    /// an absurd one is refused before a parser retains a record per
    /// member.
    #[test]
    fn reads_the_claimed_member_count_without_parsing() {
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut writer = zip::ZipWriter::new(&mut cursor);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored)
                .unix_permissions(0o644);
            for i in 0..3 {
                writer.start_file(format!("app/{i}"), options).unwrap();
            }
            writer.finish().unwrap();
        }
        let mut bytes = cursor.into_inner();
        assert_eq!(claimed_member_count(&bytes), Some(3));

        // Claim far more than the ceiling; the refusal must come before
        // the parse, so a container that never had those members is still
        // refused on what it said about itself.
        let signature = [b'P', b'K', 5u8, 6u8];
        let start = (0..=bytes.len() - 22)
            .rev()
            .find(|&i| bytes[i..i + 4] == signature)
            .expect("the fixture has an end record");
        let absurd = (MAX_MEMBERS as u16).saturating_add(1);
        bytes[start + 10..start + 12].copy_from_slice(&absurd.to_le_bytes());

        let dest = tempfile::tempdir().unwrap();
        let error = unpack_zip(&bytes, dest.path(), "a payload")
            .expect_err("a container claiming more members than this unpacks is refused");
        assert!(
            matches!(&error, InstallError::Archive { reason, .. } if reason.contains("claims")),
            "{error}"
        );
    }

    /// A byte budget does not bound member count.
    #[test]
    fn refuses_a_container_with_more_members_than_a_bundle_has() {
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut writer = zip::ZipWriter::new(&mut cursor);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored)
                .unix_permissions(0o644);
            for i in 0..(MAX_MEMBERS + 1) {
                writer.start_file(format!("app/{i}"), options).unwrap();
            }
            writer.finish().unwrap();
        }
        let bytes = cursor.into_inner();

        let dest = tempfile::tempdir().unwrap();
        let error = unpack_zip(&bytes, dest.path(), "a payload")
            .expect_err("a container of empty members is still a container to refuse");
        assert!(
            matches!(&error, InstallError::Archive { reason, .. } if reason.contains("members")),
            "{error}"
        );
    }

    /// Directories carry the packer's mode set, not the caller's umask.
    #[test]
    fn normalizes_directory_modes_whatever_the_umask() {
        use std::os::unix::fs::PermissionsExt;

        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut writer = zip::ZipWriter::new(&mut cursor);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated)
                .unix_permissions(0o644);
            // One explicit directory member, and one implied by a nested
            // file — both must come out the same.
            writer
                .add_directory("app/Contents".to_string(), options)
                .unwrap();
            writer
                .start_file("app/Contents/Resources/data.bin".to_string(), options)
                .unwrap();
            std::io::Write::write_all(&mut writer, b"x").unwrap();
            writer.finish().unwrap();
        }
        let bytes = cursor.into_inner();

        let dest = tempfile::tempdir().unwrap();
        // SAFETY: `umask` reads and sets process state; it cannot fail.
        let previous = unsafe { libc::umask(0o077) };
        let outcome = unpack_zip(&bytes, dest.path(), "a payload");
        unsafe { libc::umask(previous) };
        outcome.expect("the container is well formed");

        for dir in ["app", "app/Contents", "app/Contents/Resources"] {
            let mode = std::fs::metadata(dest.path().join(dir))
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o755, "`{dir}` should carry the packer's mode set");
        }
    }

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
