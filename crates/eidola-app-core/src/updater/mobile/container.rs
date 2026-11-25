//! The container a bundle is published in: a POSIX `ustar` archive of its
//! files, read in memory.
//!
//! The reader runs only over bytes whose hash the signed manifest already
//! matched, and what it returns is checked file by file against the
//! manifest's listing before anything reaches the shell. It is strict all the
//! same, because a reader that is generous about its input is a reader whose
//! output depends on more than the input: every member is a regular file
//! (no directories, links, devices or extension headers), named by a bundle
//! path, at most once, with a valid header checksum, and the archive ends in
//! zero blocks and nothing else.
//!
//! The emitter's side of the format: `ustar` (magic `ustar\0`, version `00`),
//! regular files only, in path order, each name in the header's `name` field
//! or split across `prefix` and `name`, followed by at least two zero blocks.
//! Mode, owner and times are not read: the shell stores bytes, and the
//! container's own hash already pins every one of them.

use std::collections::BTreeMap;

use super::row::is_bundle_path;

const BLOCK: usize = 512;

/// The files of a container, by path, borrowing its bytes.
pub type Members<'a> = BTreeMap<String, &'a [u8]>;

/// Read every member of a `ustar` container. The message says what is
/// wrong, for a refusal.
pub fn read_ustar(bytes: &[u8]) -> Result<Members<'_>, String> {
    if !bytes.len().is_multiple_of(BLOCK) {
        return Err(format!(
            "the container is {} bytes, not a whole number of {BLOCK}-byte blocks",
            bytes.len()
        ));
    }
    let mut members = Members::new();
    let mut at = 0usize;
    loop {
        let Some(header) = bytes.get(at..at + BLOCK) else {
            return Err("the container ends without its end-of-archive blocks".into());
        };
        if header.iter().all(|b| *b == 0) {
            // The end of the archive: two zero blocks at least, and nothing
            // but zeros after them (a record pads with zero blocks).
            let rest = &bytes[at..];
            if rest.len() < 2 * BLOCK {
                return Err("the container ends with one zero block, not two".into());
            }
            if rest.iter().any(|b| *b != 0) {
                return Err("the container holds bytes after its end-of-archive blocks".into());
            }
            return Ok(members);
        }
        let (path, size) = read_header(header, at)?;
        let start = at + BLOCK;
        let end = start
            .checked_add(size)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| format!("`{path}` runs past the end of the container"))?;
        let padded = end.div_ceil(BLOCK) * BLOCK;
        if bytes[end..padded].iter().any(|b| *b != 0) {
            return Err(format!(
                "`{path}` is padded with something other than zeros"
            ));
        }
        if members.insert(path.clone(), &bytes[start..end]).is_some() {
            return Err(format!("`{path}` appears twice in the container"));
        }
        at = padded;
    }
}

/// The path and length a header declares, refusing everything but a
/// regular file named by a bundle path.
fn read_header(header: &[u8], at: usize) -> Result<(String, usize), String> {
    let declared = octal(&header[148..156]).ok_or_else(|| {
        format!("the header at byte {at} has a checksum that is not an octal number")
    })?;
    // The checksum counts its own field as eight spaces.
    let computed: u64 = header
        .iter()
        .enumerate()
        .map(|(i, b)| if (148..156).contains(&i) { b' ' } else { *b })
        .map(u64::from)
        .sum();
    if declared != computed {
        return Err(format!("the header at byte {at} fails its checksum"));
    }
    if &header[257..263] != b"ustar\0" || &header[263..265] != b"00" {
        return Err(format!(
            "the header at byte {at} is not a POSIX ustar header"
        ));
    }
    let name = field(&header[0..100])?;
    let prefix = field(&header[345..500])?;
    let path = if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}/{name}")
    };
    match header[156] {
        b'0' | 0 => {}
        other => {
            return Err(format!(
                "`{path}` is not a regular file (type `{}`); a bundle container holds \
                 regular files only",
                char::from(other).escape_default()
            ));
        }
    }
    if !is_bundle_path(&path) {
        return Err(format!("`{path}` is not a bundle path"));
    }
    let size = octal(&header[124..136])
        .ok_or_else(|| format!("`{path}` has a size that is not an octal number"))?;
    let size = usize::try_from(size).map_err(|_| format!("`{path}` is too large"))?;
    Ok((path, size))
}

/// A NUL-terminated (or full-width) text field, as UTF-8. Anything after the
/// first NUL must be NUL too.
fn field(raw: &[u8]) -> Result<&str, String> {
    let end = raw.iter().position(|b| *b == 0).unwrap_or(raw.len());
    if raw[end..].iter().any(|b| *b != 0) {
        return Err("a header name holds bytes after its terminator".into());
    }
    std::str::from_utf8(&raw[..end]).map_err(|_| "a header name is not UTF-8".into())
}

/// An octal number, padded with spaces or NULs on either side (tar writers
/// differ in which). `None` for anything else, an empty field included.
fn octal(raw: &[u8]) -> Option<u64> {
    let text = std::str::from_utf8(raw).ok()?;
    let digits = text.trim_matches(|c| c == ' ' || c == '\0');
    if digits.is_empty() || !digits.bytes().all(|b| (b'0'..=b'7').contains(&b)) {
        return None;
    }
    u64::from_str_radix(digits, 8).ok()
}

/// Write a `ustar` container of `files`, in the order given: the format the
/// reader accepts, for tests and fixtures here and the release tooling that
/// will publish one. Fixed metadata (mode 0644, owner 0, time 0), so the
/// bytes are a function of the files alone.
pub fn write_ustar(files: &[(&str, &[u8])]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for (path, bytes) in files {
        let mut header = [0u8; BLOCK];
        let (prefix, name) = split_name(path)?;
        header[..name.len()].copy_from_slice(name.as_bytes());
        header[345..345 + prefix.len()].copy_from_slice(prefix.as_bytes());
        put_octal(&mut header[100..108], 0o644);
        put_octal(&mut header[108..116], 0);
        put_octal(&mut header[116..124], 0);
        put_octal(&mut header[124..136], bytes.len() as u64);
        put_octal(&mut header[136..148], 0);
        header[156] = b'0';
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");
        header[148..156].fill(b' ');
        let sum: u64 = header.iter().map(|b| u64::from(*b)).sum();
        // Six digits, a NUL and a space: the form `tar` itself writes.
        let checksum = format!("{sum:06o}\0 ");
        header[148..156].copy_from_slice(checksum.as_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(bytes);
        out.resize(out.len().div_ceil(BLOCK) * BLOCK, 0);
    }
    out.resize(out.len() + 2 * BLOCK, 0);
    Ok(out)
}

/// Split a path into the header's `prefix` and `name`: the whole path in
/// `name` when it fits, else at the last `/` that leaves `prefix` within its
/// field (the split Python's `tarfile` makes), with `name` then within its.
fn split_name(path: &str) -> Result<(&str, &str), String> {
    if path.len() <= 100 {
        return Ok(("", path));
    }
    path.rmatch_indices('/')
        .map(|(i, _)| (&path[..i], &path[i + 1..]))
        .find(|(prefix, _)| prefix.len() <= 155)
        .filter(|(_, name)| !name.is_empty() && name.len() <= 100)
        .ok_or_else(|| format!("`{path}` is too long for a ustar header"))
}

/// Zero-padded octal filling the field but its last byte, which is NUL.
fn put_octal(field: &mut [u8], value: u64) {
    let width = field.len() - 1;
    let text = format!("{value:0width$o}");
    field[..width].copy_from_slice(text.as_bytes());
    field[width] = 0;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tar(files: &[(&str, &[u8])]) -> Vec<u8> {
        write_ustar(files).unwrap()
    }

    #[test]
    fn a_written_container_reads_back() {
        let long = format!("{}/{}", "d".repeat(120), "file.js");
        let files: &[(&str, &[u8])] = &[
            ("index.html", b"<!doctype html>"),
            ("empty", b""),
            ("exact", &[7u8; 512]),
            (&long, b"long path"),
        ];
        let bytes = tar(files);
        let members = read_ustar(&bytes).unwrap();
        assert_eq!(members.len(), 4);
        for (path, data) in files {
            assert_eq!(members[*path], *data, "{path}");
        }
    }

    #[test]
    fn trailing_record_padding_is_zeros_and_allowed() {
        let mut bytes = tar(&[("index.html", b"x")]);
        bytes.resize(10240, 0);
        assert_eq!(read_ustar(&bytes).unwrap().len(), 1);
    }

    fn header_of(bytes: &mut [u8]) -> &mut [u8] {
        &mut bytes[..BLOCK]
    }

    fn reseal(header: &mut [u8]) {
        header[148..156].fill(b' ');
        let sum: u64 = header.iter().map(|b| u64::from(*b)).sum();
        header[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
    }

    type Mutation = (&'static str, Box<dyn Fn(&mut Vec<u8>)>);

    #[test]
    fn everything_but_a_plain_ustar_of_regular_files_is_refused() {
        let base = tar(&[("index.html", b"<!doctype html>")]);
        let cases: Vec<Mutation> = vec![
            ("a bad checksum", Box::new(|b| b[0] = b'j')),
            (
                "a directory",
                Box::new(|b| {
                    let h = header_of(b);
                    h[156] = b'5';
                    reseal(h);
                }),
            ),
            (
                "a symbolic link",
                Box::new(|b| {
                    let h = header_of(b);
                    h[156] = b'2';
                    reseal(h);
                }),
            ),
            (
                "a pax header",
                Box::new(|b| {
                    let h = header_of(b);
                    h[156] = b'x';
                    reseal(h);
                }),
            ),
            (
                "GNU magic",
                Box::new(|b| {
                    let h = header_of(b);
                    h[257..265].copy_from_slice(b"ustar  \0");
                    reseal(h);
                }),
            ),
            (
                "a traversing name",
                Box::new(|b| {
                    let h = header_of(b);
                    h[..13].copy_from_slice(b"../index.html");
                    reseal(h);
                }),
            ),
            (
                "a hidden name",
                Box::new(|b| {
                    let h = header_of(b);
                    h[0] = b'.';
                    reseal(h);
                }),
            ),
            (
                "a size past the end",
                Box::new(|b| {
                    let h = header_of(b);
                    h[124..136].copy_from_slice(b"00000077777\0");
                    reseal(h);
                }),
            ),
            (
                "a size that is not octal",
                Box::new(|b| {
                    let h = header_of(b);
                    h[124..136].copy_from_slice(b"0000000001x\0");
                    reseal(h);
                }),
            ),
            (
                "junk after the end",
                Box::new(|b| {
                    let last = b.len() - 1;
                    b[last] = 1;
                }),
            ),
            (
                "one end block",
                Box::new(|b| {
                    let len = b.len();
                    b.truncate(len - BLOCK);
                }),
            ),
            (
                "no end blocks",
                Box::new(|b| {
                    let len = b.len();
                    b.truncate(len - 2 * BLOCK);
                }),
            ),
            ("a partial block", Box::new(|b| b.push(0))),
            ("nonzero padding", Box::new(|b| b[BLOCK + 20] = 1)),
            (
                "bytes after a name's terminator",
                Box::new(|b| {
                    let h = header_of(b);
                    h[50] = b'x';
                    reseal(h);
                }),
            ),
        ];
        for (what, mutate) in cases {
            let mut bytes = base.clone();
            mutate(&mut bytes);
            assert!(read_ustar(&bytes).is_err(), "{what} must be refused");
        }
    }

    #[test]
    fn the_writer_s_bytes_are_the_release_tooling_s() {
        // `mobile/tools/bundle_container.py` over a directory holding these
        // three files (`index.html` = `<!doctype html>`, `app.wasm` = 600
        // `w`s, `snippets/x/bridge.js` = `abc`) writes a container hashing
        // to this; the two writers are one format.
        use sha2::Digest as _;
        let wasm = [b'w'; 600];
        let bytes = tar(&[
            ("app.wasm", &wasm),
            ("index.html", b"<!doctype html>"),
            ("snippets/x/bridge.js", b"abc"),
        ]);
        assert_eq!(
            crate::updater::human_attestation::hex_encode(&sha2::Sha256::digest(&bytes)),
            "c52c9de2e5e1f39a6aa7b705a4a29fd39b0f2bf41b9e74ab007dfdeaff7a8843"
        );
    }

    #[test]
    fn a_repeated_member_is_refused() {
        let bytes = tar(&[("index.html", b"a"), ("index.html", b"b")]);
        let err = read_ustar(&bytes).unwrap_err();
        assert!(err.contains("twice"), "{err}");
    }

    #[test]
    fn the_reader_accepts_what_tar_writes_for_these_fields() {
        // GNU and BSD tar pad numbers with leading zeros and end them with a
        // NUL or a space; both read.
        let mut bytes = tar(&[("index.html", b"abc")]);
        let h = header_of(&mut bytes);
        h[124..136].copy_from_slice(b"00000000003 ");
        reseal(h);
        assert_eq!(read_ustar(&bytes).unwrap()["index.html"], b"abc");
    }
}
