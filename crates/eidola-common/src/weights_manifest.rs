//! The weights hash's manifest: which files of a weights directory it covers,
//! how each is named, and the canonical text that is hashed.
//!
//! One spelling for the node (`eidola-server-engine`, which hashes the
//! directory it serves) and the gateway's build (which checks each catalog
//! row's per-file manifest against the row's hash). The hashing itself is
//! each caller's: this crate takes no hash dependency.
//!
//! The **manifest** is one line per file, `<sha256, lowercase hex>␠␠<file
//! name>\n` (exactly `sha256sum`'s output), sorted by file name in byte
//! order. The weights hash is the SHA-256 of the manifest's bytes, in
//! lowercase hex. `docs/model-identity.md` specifies it for reimplementation.

/// The files the hash covers besides the shards (`*.safetensors`): the
/// model's semantic files (`config.json`, and the shard index when present)
/// and the chat artifacts (tokenizer, template, generation defaults).
pub const REQUIRED_FILES: [&str; 4] = [
    "config.json",
    "tokenizer.json",
    "chat_template.jinja",
    "generation_config.json",
];

/// Covered when present.
pub const OPTIONAL_FILES: [&str; 1] = ["model.safetensors.index.json"];

/// The shards' suffix.
pub const SHARD_SUFFIX: &str = ".safetensors";

/// Whether `name` is a safe manifest name: `[A-Za-z0-9_][A-Za-z0-9._-]*`.
///
/// Such a name reproduces byte for byte through `sha256sum` (no escaping),
/// `xargs` (no whitespace or quotes), an option parser (no leading `-`) and
/// the shell's `*` (no leading `.`), and holds no path separator.
pub fn is_safe_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// Whether the hash covers a file named `name` when the directory holds it:
/// a shard, a required file, or an optional one.
pub fn is_covered(name: &str) -> bool {
    (name.len() > SHARD_SUFFIX.len() && name.ends_with(SHARD_SUFFIX))
        || REQUIRED_FILES.contains(&name)
        || OPTIONAL_FILES.contains(&name)
}

/// Whether `digest` is a SHA-256 in lowercase hex.
pub fn is_lower_hex_sha256(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Why a set of files is not a weights manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestError(pub String);

impl core::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ManifestError {}

/// The canonical manifest of a **complete** set of covered files, given as
/// `(file name, sha256 hex)` in any order: every name safe and covered, none
/// repeated, every required file present, at least one shard, every digest
/// 64 lowercase hex digits.
pub fn canonical_manifest<'a>(
    files: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<String, ManifestError> {
    let mut sorted: Vec<(&str, &str)> = files.into_iter().collect();
    sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    for pair in sorted.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(ManifestError(format!("{} is listed twice", pair[0].0)));
        }
    }
    for (name, digest) in &sorted {
        if !is_safe_name(name) {
            return Err(ManifestError(format!(
                "{name:?} is not a safe file name ([A-Za-z0-9_][A-Za-z0-9._-]*)"
            )));
        }
        if !is_covered(name) {
            return Err(ManifestError(format!(
                "{name} is not a file the weights hash covers (a *{SHARD_SUFFIX} shard, or one \
                 of {REQUIRED_FILES:?} {OPTIONAL_FILES:?})"
            )));
        }
        if !is_lower_hex_sha256(digest) {
            return Err(ManifestError(format!(
                "{name}: its sha256 must be 64 lowercase hex digits"
            )));
        }
    }
    for required in REQUIRED_FILES {
        if !sorted.iter().any(|(name, _)| *name == required) {
            return Err(ManifestError(format!("{required} is missing")));
        }
    }
    if !sorted.iter().any(|(name, _)| name.ends_with(SHARD_SUFFIX)) {
        return Err(ManifestError(format!("no *{SHARD_SUFFIX} shard is listed")));
    }
    Ok(manifest_text(sorted))
}

/// The manifest text of `(file name, sha256 hex)` pairs already sorted by
/// name, with no check of what they are: the node's spelling of a directory
/// it has already listed and checked.
pub fn manifest_text<'a>(sorted: impl IntoIterator<Item = (&'a str, &'a str)>) -> String {
    let mut out = String::new();
    for (name, digest) in sorted {
        out.push_str(&digest.to_ascii_lowercase());
        out.push_str("  ");
        out.push_str(name);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn complete() -> Vec<(&'static str, &'static str)> {
        vec![
            ("tokenizer.json", A),
            ("model-00002-of-00002.safetensors", A),
            ("config.json", A),
            ("generation_config.json", A),
            ("chat_template.jinja", A),
            ("model-00001-of-00002.safetensors", A),
        ]
    }

    #[test]
    fn the_manifest_is_sha256sums_output_sorted_by_name() {
        let text = canonical_manifest(complete()).unwrap();
        let names: Vec<&str> = text.lines().map(|l| &l[66..]).collect();
        assert_eq!(
            names,
            [
                "chat_template.jinja",
                "config.json",
                "generation_config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "tokenizer.json",
            ]
        );
        assert!(text.starts_with(&format!("{A}  chat_template.jinja\n")));
        assert!(text.ends_with('\n'));
    }

    #[test]
    fn an_incomplete_or_foreign_set_is_refused() {
        let without = |name: &str| {
            complete()
                .into_iter()
                .filter(|(n, _)| *n != name)
                .collect::<Vec<_>>()
        };
        for required in REQUIRED_FILES {
            assert!(canonical_manifest(without(required)).is_err(), "{required}");
        }
        let no_shards: Vec<_> = complete()
            .into_iter()
            .filter(|(n, _)| !n.ends_with(SHARD_SUFFIX))
            .collect();
        assert!(canonical_manifest(no_shards).is_err());

        for extra in [
            ("README.md", A),
            ("sub/model.safetensors", A),
            (".hidden.safetensors", A),
            ("-x.safetensors", A),
            (".safetensors", A),
            ("model-00001-of-00002.safetensors", A),
            ("model.safetensors", "AAAA"),
        ] {
            let mut files = complete();
            files.push(extra);
            assert!(canonical_manifest(files).is_err(), "{extra:?}");
        }

        let mut with_index = complete();
        with_index.push(("model.safetensors.index.json", A));
        assert!(canonical_manifest(with_index).is_ok());
    }
}
