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
//!
//! **The covered names are the same for every architecture** ([`is_covered`]:
//! the shards and [`COVERED_FILES`]), and the hash covers every covered file
//! the directory holds, so one offline procedure computes it for any model.
//! What differs is which of them must be there: the files an architecture's
//! node reads ([`required_files`]). A covered file it does not read (a
//! template a compiled-in prompt encoder makes unnecessary, a shard of tensors
//! a text-only model never reads) is hashed all the same.

use crate::architecture::Architecture;

/// The files the hash covers besides the shards (`*.safetensors`) when the
/// directory holds them: the model's semantic files (`config.json`, the
/// shard index) and the chat artifacts (tokenizer, template, generation
/// defaults).
pub const COVERED_FILES: [&str; 5] = [
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "chat_template.jinja",
    "generation_config.json",
];

/// The files a MiMo-V2.6 or Qwen3.8 manifest must list besides its shards:
/// the semantic file and the chat artifacts its node reads
/// ([`required_files`]).
pub const REQUIRED_FILES: [&str; 4] = [
    "config.json",
    "tokenizer.json",
    "chat_template.jinja",
    "generation_config.json",
];

/// Covered when present, for MiMo-V2.6: the covered files outside
/// [`REQUIRED_FILES`].
pub const OPTIONAL_FILES: [&str; 1] = ["model.safetensors.index.json"];

/// The files a DeepSeek-V4.1 manifest must list besides its shards: the
/// configuration and the tokenizer. Its prompt encoder is compiled in, so
/// its node reads no template, and its sampling defaults are a
/// `generation_config.json` when the directory holds one (the released
/// checkpoint does not).
pub const DEEPSEEK_V4_1_REQUIRED_FILES: [&str; 2] = ["config.json", "tokenizer.json"];

/// The files a Qwen3.5 manifest must list besides its shards: the
/// configuration (which also names its end-of-sequence id), the tokenizer
/// and its own template. Its sampling defaults are a `generation_config.json`
/// when the directory holds one (the released checkpoints do not).
pub const QWEN3_5_REQUIRED_FILES: [&str; 3] =
    ["config.json", "tokenizer.json", "chat_template.jinja"];

/// The covered files a manifest of `architecture` must list besides its
/// shards: those its node reads whatever else the directory holds.
pub fn required_files(architecture: Architecture) -> &'static [&'static str] {
    match architecture {
        // Qwen3.8 reads what MiMo-V2.6 does: its own template, its tokenizer,
        // and `generation_config.json` for its end-of-sequence ids and
        // sampling defaults.
        Architecture::MimoV26Flash | Architecture::MimoV26Pro | Architecture::Qwen38Dense27B => {
            &REQUIRED_FILES
        }
        Architecture::DeepSeekV41Flash => &DEEPSEEK_V4_1_REQUIRED_FILES,
        Architecture::Qwen35Dense2B | Architecture::Qwen35Dense08B => &QWEN3_5_REQUIRED_FILES,
    }
}

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
/// a shard, or one of [`COVERED_FILES`].
pub fn is_covered(name: &str) -> bool {
    (name.len() > SHARD_SUFFIX.len() && name.ends_with(SHARD_SUFFIX))
        || COVERED_FILES.contains(&name)
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

/// The canonical manifest of a **complete** set of a MiMo-V2.6 model's
/// covered files ([`canonical_manifest_for`] with
/// [`Architecture::MimoV26Flash`]).
pub fn canonical_manifest<'a>(
    files: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<String, ManifestError> {
    canonical_manifest_for(Architecture::MimoV26Flash, files)
}

/// The canonical manifest of a **complete** set of covered files of a model
/// of `architecture`, given as `(file name, sha256 hex)` in any order: every
/// name safe and covered, none repeated, every file [`required_files`] names
/// present, at least one shard, every digest 64 lowercase hex digits.
pub fn canonical_manifest_for<'a>(
    architecture: Architecture,
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
                 of {COVERED_FILES:?})"
            )));
        }
        if !is_lower_hex_sha256(digest) {
            return Err(ManifestError(format!(
                "{name}: its sha256 must be 64 lowercase hex digits"
            )));
        }
    }
    for &required in required_files(architecture) {
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
            ("tokenizer_config.json", A),
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

    /// A text-only DeepSeek-V4.1 directory, as published: shards (the
    /// vision tower's among them), the index, the configuration, the
    /// tokenizer and the model's own template, which its node does not read
    /// but hashes like any covered file; no generation defaults. Files the
    /// hash never covers (`tokenizer_config.json`, the reference code) are
    /// left out of the listing.
    fn deepseek() -> Vec<(&'static str, &'static str)> {
        vec![
            ("model-00001-of-00048.safetensors", A),
            ("model-00048-of-00048.safetensors", A),
            ("model.safetensors.index.json", A),
            ("config.json", A),
            ("tokenizer.json", A),
            ("chat_template.jinja", A),
        ]
    }

    #[test]
    fn each_architecture_requires_the_files_its_node_reads() {
        let ds = Architecture::DeepSeekV41Flash;
        assert_eq!(required_files(ds), ["config.json", "tokenizer.json"]);
        assert_eq!(required_files(Architecture::MimoV26Flash), REQUIRED_FILES);
        for qwen35 in [Architecture::Qwen35Dense2B, Architecture::Qwen35Dense08B] {
            assert_eq!(
                required_files(qwen35),
                ["config.json", "tokenizer.json", "chat_template.jinja"]
            );
        }
        for arch in Architecture::ALL {
            for name in required_files(arch) {
                assert!(is_covered(name), "{arch}: {name}");
            }
        }
        let text = canonical_manifest_for(ds, deepseek()).unwrap();
        assert_eq!(text.lines().count(), 6);
        // Without the template it is still complete; without the tokenizer
        // or the configuration it is not.
        let without = |name: &str| {
            deepseek()
                .into_iter()
                .filter(|(n, _)| *n != name)
                .collect::<Vec<_>>()
        };
        assert!(canonical_manifest_for(ds, without("chat_template.jinja")).is_ok());
        assert!(canonical_manifest_for(ds, without("model.safetensors.index.json")).is_ok());
        for required in ["config.json", "tokenizer.json"] {
            assert!(canonical_manifest_for(ds, without(required)).is_err());
        }
        // The same listing is not a MiMo manifest: it has no generation
        // defaults.
        assert!(canonical_manifest(deepseek()).is_err());
        let mut with_defaults = deepseek();
        with_defaults.push(("generation_config.json", A));
        assert!(canonical_manifest_for(ds, with_defaults).is_ok());
    }

    /// The manifest of a directory is the same text whichever architecture
    /// reads it: only the required set differs.
    #[test]
    fn the_spelling_does_not_depend_on_the_architecture() {
        assert_eq!(
            canonical_manifest(complete()).unwrap(),
            canonical_manifest_for(Architecture::DeepSeekV41Flash, complete()).unwrap()
        );
    }

    /// A synthetic text-only DeepSeek directory: the manifest of every
    /// covered file it holds is the canonical one, and the files the hash
    /// does not cover are not in it.
    #[test]
    fn a_synthetic_deepseek_directory_hashes_its_covered_files() {
        let dir = std::env::temp_dir().join(format!("eidola-ds-manifest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("inference")).unwrap();
        for name in [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "README.md",
            "inference/model.py",
        ] {
            std::fs::write(dir.join(name), name).unwrap();
        }
        let mut listed: Vec<(String, String)> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|name| is_covered(name))
            .map(|name| (name, A.to_string()))
            .collect();
        listed.sort();
        let text = canonical_manifest_for(
            Architecture::DeepSeekV41Flash,
            listed.iter().map(|(n, d)| (n.as_str(), d.as_str())),
        )
        .unwrap();
        let names: Vec<&str> = text.lines().map(|l| &l[66..]).collect();
        assert_eq!(
            names,
            [
                "chat_template.jinja",
                "config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json",
                "tokenizer.json",
            ]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
