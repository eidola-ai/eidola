//! Reading and writing a document's bytes.
//!
//! The editor's buffer is LF-only markdown with no byte-order mark, but the
//! files people keep are not always that. A note written on Windows uses CRLF,
//! some tools write a UTF-8 BOM, and most files end in one newline the editor
//! would otherwise show as a trailing empty paragraph. [`decode`] normalises
//! all three away and records what it removed in a [`Format`]; [`encode`]
//! puts exactly that back, so opening and saving an untouched file reproduces
//! its bytes.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// The on-disk shape a document was read in, restored on save.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Format {
    pub line_ending: LineEnding,
    /// The file opened with a UTF-8 byte-order mark.
    pub bom: bool,
    /// The file ended in a line ending. A new document gets one, the POSIX
    /// convention for a text file.
    pub trailing_newline: bool,
}

impl Default for Format {
    fn default() -> Self {
        Self {
            line_ending: LineEnding::Lf,
            bom: false,
            trailing_newline: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineEnding {
    Lf,
    CrLf,
    /// A lone carriage return, the classic Mac convention.
    Cr,
}

#[derive(Debug)]
pub enum DecodeError {
    /// The bytes are not UTF-8, so this is not a text file the editor can
    /// hold without losing something.
    NotUtf8,
}

/// Decode a file's bytes into editor text plus the format they came in.
///
/// The line ending is the file's *first* one: a file mixing both is saved
/// back with the one it starts with, which is the conservative reading of
/// what its author meant. A lone CR (classic Mac) is a line break too, and
/// is what such a file is saved back with.
pub fn decode(bytes: &[u8]) -> Result<(String, Format), DecodeError> {
    let (bytes, bom) = match bytes.strip_prefix(b"\xEF\xBB\xBF") {
        Some(rest) => (rest, true),
        None => (bytes, false),
    };
    let raw = std::str::from_utf8(bytes).map_err(|_| DecodeError::NotUtf8)?;

    let line_ending = match raw.find(['\r', '\n']) {
        Some(i) if raw[i..].starts_with("\r\n") => LineEnding::CrLf,
        Some(i) if raw[i..].starts_with('\r') => LineEnding::Cr,
        _ => LineEnding::Lf,
    };
    let mut text = raw.replace("\r\n", "\n").replace('\r', "\n");
    let trailing_newline = text.ends_with('\n');
    if trailing_newline {
        text.pop();
    }
    Ok((
        text,
        Format {
            line_ending,
            bom,
            trailing_newline,
        },
    ))
}

/// Encode editor text back into the bytes of a file in `format`.
pub fn encode(text: &str, format: Format) -> Vec<u8> {
    let mut body = text.to_string();
    // `decode` took exactly one line ending off the end, so exactly one goes
    // back — even when the text already ends in one (a file ending in a
    // blank line), and when it is all the file held. A file that was empty
    // records none and stays empty.
    if format.trailing_newline {
        body.push('\n');
    }
    match format.line_ending {
        LineEnding::Lf => {}
        LineEnding::CrLf => body = body.replace('\n', "\r\n"),
        LineEnding::Cr => body = body.replace('\n', "\r"),
    }
    let mut out = Vec::with_capacity(body.len() + 3);
    if format.bom {
        out.extend_from_slice(b"\xEF\xBB\xBF");
    }
    out.extend_from_slice(body.as_bytes());
    out
}

/// Write `bytes` to `path` without ever leaving it half-written: the new
/// contents go to a sibling temporary file that is renamed over the target,
/// so a crash or a full disk leaves either the old file or the new one.
///
/// An existing file's permissions and group carry over to its replacement,
/// and the temporary file has both before it holds a byte, so a private note
/// is never readable through its sibling, nor by a group the note was not
/// shared with. A save that cannot keep the group is refused. Access control
/// lists are not carried across: a save is refused, before the temporary file
/// holds a byte, when the note has one or the temporary file was given one
/// (inherited from its folder), since who could read the replacement would
/// then be more than its mode bits and group say. A symlink is written
/// *through* rather than replaced, so a note symlinked into a vault stays a
/// symlink — even when the file it points at has gone, which the write then
/// recreates.
pub fn write_atomically(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let target = resolve_links(path);
    let existing_meta = fs::metadata(&target).ok();
    let existing = existing_meta.as_ref().map(|meta| meta.permissions());
    if existing_meta.is_some() {
        refuse_acl(acl::on_path(&target), AclSource::Note)?;
    }
    let dir = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "document".into());
    let (temp, mut file) = create_temp(dir, &name, existing.as_ref())?;

    let result = (|| {
        refuse_acl(acl::on_file(&file), AclSource::Folder)?;
        if let Some(meta) = &existing_meta {
            keep_group(&file, meta)?;
        }
        file.write_all(bytes)?;
        file.sync_all()?;
        if let Some(permissions) = existing {
            fs::set_permissions(&temp, permissions)?;
        }
        fs::rename(&temp, &target)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

/// A new temporary file beside the target. Unique per write, and created
/// exclusively, so two saves in flight never share, truncate or rename each
/// other's; created with the target's permissions, so it is never more
/// readable than the file it will replace.
fn create_temp(
    dir: &Path,
    name: &str,
    permissions: Option<&fs::Permissions>,
) -> io::Result<(PathBuf, fs::File)> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    loop {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let temp = dir.join(format!(
            ".{}.{}.{n}.tmp",
            temp_stem(name),
            std::process::id()
        ));
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        if let Some(permissions) = permissions {
            use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
            options.mode(permissions.mode() & 0o777);
        }
        #[cfg(not(unix))]
        let _ = permissions;
        match options.open(&temp) {
            Ok(file) => return Ok((temp, file)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}

/// Give the still-empty temporary file the group of the file it will replace.
/// A new file takes its group from the directory or the process, which need
/// not be the target's, and the mode's group bits would then grant the note
/// to a different set of people.
#[cfg(unix)]
fn keep_group(temp: &fs::File, target: &fs::Metadata) -> io::Result<()> {
    use std::os::unix::fs::MetadataExt;
    let gid = target.gid();
    if temp.metadata()?.gid() == gid {
        return Ok(());
    }
    std::os::unix::fs::fchown(temp, None, Some(gid)).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("the file's group ({gid}) could not be kept on its replacement: {e}"),
        )
    })
}

#[cfg(not(unix))]
fn keep_group(_temp: &fs::File, _target: &fs::Metadata) -> io::Result<()> {
    Ok(())
}

/// Where an access control list that refuses a save was found.
#[derive(Clone, Copy)]
enum AclSource {
    /// On the note being replaced.
    Note,
    /// On the fresh temporary file, which can only have inherited it from
    /// the folder.
    Folder,
}

/// Turn the answer of an ACL probe into the save's verdict: no list goes
/// ahead; a list, or a probe that could not answer, refuses.
fn refuse_acl(found: io::Result<bool>, source: AclSource) -> io::Result<()> {
    let what = match source {
        AclSource::Note => "This note has an access control list",
        AclSource::Folder => "This note's folder gives new files an access control list",
    };
    match found {
        Ok(false) => Ok(()),
        Ok(true) => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "{what}, which this editor does not carry across a save, so the save was \
                 refused rather than change who can read the note."
            ),
        )),
        Err(e) => Err(io::Error::new(
            e.kind(),
            format!(
                "Whether an access control list applies to this note could not be checked, \
                 so the save was refused: {e}"
            ),
        )),
    }
}

/// Whether a file carries an access control list beyond its mode bits. Each
/// answers `Ok(false)` only when the system says there is none, or that the
/// filesystem cannot hold one; any other failure is an error, never a pass.
mod acl {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    use std::ffi::CString;
    use std::fs;
    use std::io;
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    use std::os::unix::{ffi::OsStrExt, io::AsRawFd};
    use std::path::Path;

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn c_path(path: &Path) -> io::Result<CString> {
        CString::new(path.as_os_str().as_bytes())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
    }

    // Linux: a POSIX access ACL is the `system.posix_acl_access` xattr, which
    // the kernel stores only when the list says more than the mode bits do. A
    // folder's default ACL reaches a new file as exactly that xattr.
    #[cfg(target_os = "linux")]
    const ACCESS_ACL: &std::ffi::CStr = c"system.posix_acl_access";

    #[cfg(target_os = "linux")]
    fn xattr_answer(size: libc::ssize_t) -> io::Result<bool> {
        if size >= 0 {
            return Ok(true);
        }
        let e = io::Error::last_os_error();
        // ENODATA: no such xattr. EOPNOTSUPP (the same number as ENOTSUP on
        // Linux): the filesystem holds no xattrs or no ACLs.
        match e.raw_os_error() {
            Some(libc::ENODATA) | Some(libc::EOPNOTSUPP) => Ok(false),
            _ => Err(e),
        }
    }

    #[cfg(target_os = "linux")]
    pub fn on_path(path: &Path) -> io::Result<bool> {
        let path = c_path(path)?;
        // SAFETY: both strings are NUL-terminated and outlive the call; a
        // null buffer of size 0 asks only for the value's size.
        let size =
            unsafe { libc::getxattr(path.as_ptr(), ACCESS_ACL.as_ptr(), std::ptr::null_mut(), 0) };
        xattr_answer(size)
    }

    #[cfg(target_os = "linux")]
    pub fn on_file(file: &fs::File) -> io::Result<bool> {
        // SAFETY: the descriptor is open for the call's duration (borrowed
        // from `file`); the name is NUL-terminated; a null buffer of size 0
        // asks only for the value's size.
        let size = unsafe {
            libc::fgetxattr(
                file.as_raw_fd(),
                ACCESS_ACL.as_ptr(),
                std::ptr::null_mut(),
                0,
            )
        };
        xattr_answer(size)
    }

    // macOS: extended ACLs, from <sys/acl.h>, which the `libc` crate does
    // not bind. A file without one answers NULL with ENOENT.
    #[cfg(target_os = "macos")]
    mod sys {
        use std::ffi::{c_char, c_int, c_void};

        pub const ACL_TYPE_EXTENDED: c_int = 0x0000_0100;
        pub const ACL_FIRST_ENTRY: c_int = 0;

        unsafe extern "C" {
            pub fn acl_get_fd_np(fd: c_int, kind: c_int) -> *mut c_void;
            pub fn acl_get_file(path: *const c_char, kind: c_int) -> *mut c_void;
            pub fn acl_get_entry(
                acl: *mut c_void,
                entry_id: c_int,
                entry: *mut *mut c_void,
            ) -> c_int;
            pub fn acl_free(obj: *mut c_void) -> c_int;
        }
    }

    /// Whether `acl`, as `acl_get_*` returned it, holds at least one entry.
    /// Takes ownership: the list is freed here.
    #[cfg(target_os = "macos")]
    fn acl_answer(acl: *mut std::ffi::c_void) -> io::Result<bool> {
        if acl.is_null() {
            let e = io::Error::last_os_error();
            return match e.raw_os_error() {
                // ENOENT: no extended ACL. ENOTSUP: the filesystem holds none.
                Some(libc::ENOENT) | Some(libc::ENOTSUP) => Ok(false),
                _ => Err(e),
            };
        }
        let mut entry = std::ptr::null_mut();
        // SAFETY: `acl` is a live list from `acl_get_*` and `entry` a valid
        // out-pointer; the entry is only counted, never used after the free.
        let first = unsafe { sys::acl_get_entry(acl, sys::ACL_FIRST_ENTRY, &mut entry) };
        let answer = if first == 0 {
            Ok(true)
        } else {
            let e = io::Error::last_os_error();
            // EINVAL: there is no first entry, an empty list.
            match e.raw_os_error() {
                Some(libc::EINVAL) => Ok(false),
                _ => Err(e),
            }
        };
        // SAFETY: `acl` came from `acl_get_*`, is freed once, and is not used
        // afterwards.
        unsafe { sys::acl_free(acl) };
        answer
    }

    #[cfg(target_os = "macos")]
    pub fn on_path(path: &Path) -> io::Result<bool> {
        // By path rather than descriptor: a note whose ACL denies its owner
        // reading cannot be opened to ask.
        let path = c_path(path)?;
        // SAFETY: the path is NUL-terminated and outlives the call; the
        // returned list is owned by `acl_answer`.
        acl_answer(unsafe { sys::acl_get_file(path.as_ptr(), sys::ACL_TYPE_EXTENDED) })
    }

    #[cfg(target_os = "macos")]
    pub fn on_file(file: &fs::File) -> io::Result<bool> {
        // SAFETY: the descriptor is open for the call's duration (borrowed
        // from `file`); the returned list is owned by `acl_answer`.
        acl_answer(unsafe { sys::acl_get_fd_np(file.as_raw_fd(), sys::ACL_TYPE_EXTENDED) })
    }

    // Elsewhere nothing is checked, as `keep_group` keeps nothing off Unix.
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub fn on_path(_path: &Path) -> io::Result<bool> {
        Ok(false)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub fn on_file(_file: &fs::File) -> io::Result<bool> {
        Ok(false)
    }
}

/// The document's name as it appears in a temporary file's: enough to tell
/// whose it is, cut on a character boundary so the whole temporary name stays
/// well inside the 255-byte limit most filesystems put on a name, however
/// long the document's own is.
fn temp_stem(name: &str) -> &str {
    const MAX: usize = 128;
    if name.len() <= MAX {
        return name;
    }
    let mut end = MAX;
    while !name.is_char_boundary(end) {
        end -= 1;
    }
    &name[..end]
}

/// The file `path` names, through any symlinks. `canonicalize` cannot see past
/// a link whose target is missing, so such a chain is followed by hand rather
/// than treating the link itself as the file to replace.
fn resolve_links(path: &Path) -> PathBuf {
    if let Ok(resolved) = fs::canonicalize(path) {
        return resolved;
    }
    let mut current = path.to_path_buf();
    // The same bound the kernel puts on a chain of links.
    for _ in 0..40 {
        let Ok(next) = fs::read_link(&current) else {
            break;
        };
        current = match current.parent() {
            Some(dir) => dir.join(next),
            None => next,
        };
    }
    current
}

/// The path a Save As lands on: what the panel returned, with `.md` added
/// when the reader typed a bare name. A name that already carries an
/// extension — `.txt`, `.markdown` — is theirs to keep.
pub fn with_default_extension(path: PathBuf) -> PathBuf {
    if path.extension().is_some() {
        path
    } else {
        path.with_extension("md")
    }
}

/// The name a window and its title band show for a document.
pub fn display_name(path: Option<&Path>, untitled: usize) -> String {
    match path.and_then(|p| p.file_name()) {
        Some(name) => name.to_string_lossy().into_owned(),
        None if untitled <= 1 => "Untitled".into(),
        None => format!("Untitled {untitled}"),
    }
}

/// A modification stamp for noticing that another program changed the file.
pub fn modified(path: &Path) -> Option<std::time::SystemTime> {
    fs::metadata(path).and_then(|m| m.modified()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(bytes: &[u8]) -> Vec<u8> {
        let (text, format) = decode(bytes).unwrap();
        encode(&text, format)
    }

    #[test]
    fn an_untouched_file_saves_back_byte_for_byte() {
        for bytes in [
            &b"# Title\n\nBody\n"[..],
            b"# Title\n\nBody",
            b"# Title\r\n\r\nBody\r\n",
            b"\xEF\xBB\xBF# Title\nBody\n",
            b"# Title\n\n",
            b"",
        ] {
            assert_eq!(
                round_trip(bytes),
                bytes,
                "{:?}",
                String::from_utf8_lossy(bytes)
            );
        }
    }

    #[test]
    fn the_editor_never_sees_a_carriage_return_a_bom_or_the_final_newline() {
        let (text, format) = decode(b"\xEF\xBB\xBFone\r\ntwo\r\n").unwrap();
        assert_eq!(text, "one\ntwo");
        assert_eq!(
            format,
            Format {
                line_ending: LineEnding::CrLf,
                bom: true,
                trailing_newline: true,
            }
        );
    }

    #[test]
    fn a_file_of_one_line_ending_saves_back_as_itself() {
        for bytes in [&b"\n"[..], b"\r\n", b"\r", b""] {
            assert_eq!(round_trip(bytes), bytes, "{bytes:?}");
        }
    }

    #[test]
    fn a_classic_mac_file_saves_back_with_carriage_returns() {
        assert_eq!(round_trip(b"title\rbody\r"), b"title\rbody\r");
        let (text, format) = decode(b"title\rbody").unwrap();
        assert_eq!(text, "title\nbody");
        assert_eq!(format.line_ending, LineEnding::Cr);
    }

    #[test]
    fn a_document_with_the_longest_name_a_filesystem_allows_still_saves() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(format!("{}.md", "a".repeat(250)));
        assert_eq!(path.file_name().unwrap().len(), 253);
        std::fs::write(&path, "old\n").unwrap();
        write_atomically(&path, b"new\n").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new\n");
    }

    #[test]
    fn a_long_name_is_cut_on_a_character_boundary() {
        let name = "é".repeat(100);
        let stem = temp_stem(&name);
        assert!(stem.len() <= 128 && stem.chars().all(|c| c == 'é'));
        assert_eq!(temp_stem("short.md"), "short.md");
    }

    #[test]
    fn edits_are_saved_in_the_files_own_line_ending() {
        let (_, format) = decode(b"a\r\nb\r\n").unwrap();
        assert_eq!(encode("a\nb\nc", format), b"a\r\nb\r\nc\r\n");
    }

    #[test]
    fn a_new_document_ends_in_a_newline() {
        assert_eq!(encode("hello", Format::default()), b"hello\n");
        assert_eq!(encode("", Format::default()), b"\n");
    }

    #[test]
    fn bytes_that_are_not_utf8_are_refused() {
        assert!(matches!(decode(b"\xff\xfe"), Err(DecodeError::NotUtf8)));
    }

    #[test]
    fn a_bare_save_name_gets_the_markdown_extension() {
        assert_eq!(
            with_default_extension("/n/Ideas".into()),
            PathBuf::from("/n/Ideas.md")
        );
        assert_eq!(
            with_default_extension("/n/Ideas.txt".into()),
            PathBuf::from("/n/Ideas.txt")
        );
    }

    #[cfg(unix)]
    #[test]
    fn an_atomic_write_replaces_the_file_and_keeps_its_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("note.md");
        fs::write(&path, "old").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();

        write_atomically(&path, b"new").unwrap();

        assert_eq!(fs::read(&path).unwrap(), b"new");
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        let leftovers: Vec<_> = fs::read_dir(dir.path()).unwrap().collect();
        assert_eq!(leftovers.len(), 1, "no temporary file is left behind");
    }

    #[cfg(unix)]
    /// A group this user belongs to other than the one a new file in `dir`
    /// gets, if there is one: the only kind an unprivileged test can give a
    /// file.
    #[cfg(unix)]
    fn another_group(dir: &Path) -> Option<u32> {
        use std::os::unix::fs::MetadataExt;
        let probe = dir.join("probe");
        fs::write(&probe, b"").unwrap();
        let default = fs::metadata(&probe).unwrap().gid();
        let out = std::process::Command::new("id").arg("-G").output().ok()?;
        String::from_utf8(out.stdout)
            .ok()?
            .split_whitespace()
            .filter_map(|g| g.parse::<u32>().ok())
            .filter(|g| *g != default)
            .find(|g| std::os::unix::fs::chown(&probe, None, Some(*g)).is_ok())
    }

    #[cfg(unix)]
    #[test]
    fn an_atomic_write_keeps_the_files_group() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let dir = tempfile::tempdir().unwrap();
        let Some(group) = another_group(dir.path()) else {
            eprintln!("skipped: this user has one group, so no file can differ from the default");
            return;
        };
        let path = dir.path().join("shared.md");
        fs::write(&path, b"old").unwrap();
        std::os::unix::fs::chown(&path, None, Some(group)).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();

        write_atomically(&path, b"new").unwrap();

        let meta = fs::metadata(&path).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new");
        assert_eq!(meta.gid(), group);
        assert_eq!(meta.permissions().mode() & 0o777, 0o640);
    }

    #[test]
    fn overlapping_atomic_writes_each_land_whole() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.md");
        for _ in 0..20 {
            let writers: Vec<_> = [b"first\n".repeat(4096), b"second\n".repeat(4096)]
                .into_iter()
                .map(|bytes| {
                    let path = path.clone();
                    std::thread::spawn(move || write_atomically(&path, &bytes))
                })
                .collect();
            for writer in writers {
                writer
                    .join()
                    .unwrap()
                    .expect("neither write is spoiled by the other");
            }
            let text = std::fs::read_to_string(&path).unwrap();
            assert!(
                text == "first\n".repeat(4096) || text == "second\n".repeat(4096),
                "the file is one write or the other, never a mix"
            );
        }
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn a_private_file_is_never_readable_through_its_temporary_copy() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("private.md");
        std::fs::write(&path, "secret\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        let existing = std::fs::metadata(&path).unwrap().permissions();
        let (temp, _file) = create_temp(dir.path(), "private.md", Some(&existing)).unwrap();
        assert_eq!(
            std::fs::metadata(&temp).unwrap().permissions().mode() & 0o777,
            0o600,
            "private before a byte is written"
        );
        std::fs::remove_file(&temp).unwrap();

        write_atomically(&path, b"still secret\n").unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }

    /// Run an ACL tool on `target`, giving another principal read access:
    /// `inherited` puts it on a folder for new files to inherit, otherwise on
    /// the file itself. `Err` says why the system could not, which skips a
    /// test only on Linux, where `setfacl` or ACL support may be missing; on
    /// macOS `chmod +a` needs no privileges and always applies.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn grant_read(target: &Path, inherited: bool) -> Result<(), String> {
        let mut command = if cfg!(target_os = "macos") {
            let mut c = std::process::Command::new("chmod");
            let ace = if inherited {
                "everyone allow read,file_inherit"
            } else {
                "everyone allow read"
            };
            c.args(["+a", ace]);
            c
        } else {
            let mut c = std::process::Command::new("setfacl");
            if inherited {
                c.arg("-d");
            }
            c.args(["-m", "u:nobody:r"]);
            c
        };
        let out = command
            .arg(target)
            .output()
            .map_err(|e| format!("{command:?} could not run: {e}"))?;
        if out.status.success() {
            Ok(())
        } else {
            Err(format!(
                "{command:?} failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ))
        }
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn acl_applied(result: Result<(), String>) -> bool {
        match result {
            Ok(()) => true,
            Err(why) if cfg!(target_os = "linux") => {
                eprintln!("skipped: this system cannot set an ACL here ({why})");
                false
            }
            Err(why) => panic!("{why}"),
        }
    }

    /// A refused save leaves the note as it was and nothing beside it.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn assert_refused(dir: &Path, path: &Path, result: io::Result<()>, because: &str) {
        let error = result.expect_err("the save is refused");
        assert!(
            error.to_string().contains(because),
            "the refusal says why: {error}"
        );
        assert_eq!(
            fs::read(path).unwrap(),
            b"secret\n",
            "the note is untouched"
        );
        let names: Vec<_> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(
            names,
            [path.file_name().unwrap()],
            "no temporary file is left"
        );
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn a_save_into_a_folder_whose_acl_new_files_inherit_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("private.md");
        fs::write(&path, "secret\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
        if !acl_applied(grant_read(dir.path(), true)) {
            return;
        }

        let result = write_atomically(&path, b"new\n");

        assert_refused(dir.path(), &path, result, "folder gives new files");
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn a_save_over_a_note_with_an_acl_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.md");
        fs::write(&path, "secret\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
        if !acl_applied(grant_read(&path, false)) {
            return;
        }

        let result = write_atomically(&path, b"new\n");

        assert_refused(dir.path(), &path, result, "note has an access control list");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_whose_file_has_gone_is_written_through_not_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let vault = dir.path().join("vault");
        std::fs::create_dir(&vault).unwrap();
        let real = vault.join("note.md");
        let link = dir.path().join("note.md");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        write_atomically(&link, b"back\n").unwrap();

        assert!(link.symlink_metadata().unwrap().file_type().is_symlink());
        assert_eq!(std::fs::read(&real).unwrap(), b"back\n");
    }

    #[test]
    fn an_atomic_write_through_a_symlink_keeps_the_link() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real.md");
        let link = dir.path().join("link.md");
        fs::write(&real, "old").unwrap();
        std::os::unix::fs::symlink(&real, &link).unwrap();

        write_atomically(&link, b"new").unwrap();

        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read(&real).unwrap(), b"new");
    }

    #[test]
    fn untitled_documents_are_numbered_after_the_first() {
        assert_eq!(display_name(None, 1), "Untitled");
        assert_eq!(display_name(None, 3), "Untitled 3");
        assert_eq!(display_name(Some(Path::new("/a/b.md")), 1), "b.md");
    }
}
