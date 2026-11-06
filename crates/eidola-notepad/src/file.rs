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
/// what its author meant. A lone CR (classic Mac) is read as a line break.
pub fn decode(bytes: &[u8]) -> Result<(String, Format), DecodeError> {
    let (bytes, bom) = match bytes.strip_prefix(b"\xEF\xBB\xBF") {
        Some(rest) => (rest, true),
        None => (bytes, false),
    };
    let raw = std::str::from_utf8(bytes).map_err(|_| DecodeError::NotUtf8)?;

    let line_ending = match raw.find(['\r', '\n']) {
        Some(i) if raw[i..].starts_with("\r\n") => LineEnding::CrLf,
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
    // blank line). An empty document stays an empty file.
    if format.trailing_newline && !body.is_empty() {
        body.push('\n');
    }
    if format.line_ending == LineEnding::CrLf {
        body = body.replace('\n', "\r\n");
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
/// An existing file's permissions carry over to its replacement. A symlink is
/// written *through* rather than replaced, so a note symlinked into a vault
/// stays a symlink.
pub fn write_atomically(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let target = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let dir = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "document".into());
    // Unique per write, and created exclusively: two saves in flight never
    // share, truncate or rename each other's temporary file.
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let (temp, mut file) = loop {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let temp = dir.join(format!(".{name}.{}.{n}.tmp", std::process::id()));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
        {
            Ok(file) => break (temp, file),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    };

    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        if let Ok(meta) = fs::metadata(&target) {
            fs::set_permissions(&temp, meta.permissions())?;
        }
        fs::rename(&temp, &target)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
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
    fn edits_are_saved_in_the_files_own_line_ending() {
        let (_, format) = decode(b"a\r\nb\r\n").unwrap();
        assert_eq!(encode("a\nb\nc", format), b"a\r\nb\r\nc\r\n");
    }

    #[test]
    fn a_new_document_ends_in_a_newline() {
        assert_eq!(encode("hello", Format::default()), b"hello\n");
        assert_eq!(encode("", Format::default()), b"");
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
