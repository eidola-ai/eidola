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
/// An existing file's permissions carry over to its replacement, and the
/// temporary file has them before it holds a byte, so a private note is never
/// readable through its sibling. A symlink is written *through* rather than
/// replaced, so a note symlinked into a vault stays a symlink — even when the
/// file it points at has gone, which the write then recreates.
pub fn write_atomically(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let target = resolve_links(path);
    let existing = fs::metadata(&target).ok().map(|meta| meta.permissions());
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
