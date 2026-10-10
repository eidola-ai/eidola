//! The served model: its files, their identity (the **weights hash**), and the loaded
//! reference model, tokenizer, template and generation defaults.
//!
//! # The weights hash
//!
//! A node's weights hash identifies every file it reads from the weights directory:
//!
//! * every `*.safetensors` file in the directory (the shards);
//! * `config.json`, and `model.safetensors.index.json` when present (the model crate's
//!   semantic files, which decide how the shards are read);
//! * `tokenizer.json`, `chat_template.jinja` and `generation_config.json` (the chat
//!   artifacts, which decide the prompt, the EOS ids and the default sampling).
//!
//! The **manifest** is one line per file, `<sha256 hex, lowercase>  <file name>\n` (two
//! spaces, as `sha256sum` prints), sorted by file name in byte order. The weights hash is
//! the SHA-256 of the manifest's bytes, in lowercase hex. Offline, in the directory:
//!
//! ```sh
//! LC_ALL=C ls *.safetensors config.json model.safetensors.index.json \
//!     tokenizer.json chat_template.jinja generation_config.json 2>/dev/null \
//!   | LC_ALL=C sort | xargs sha256sum | sha256sum
//! ```
//!
//! Every name in the manifest is a **safe name**: ASCII letters, digits, `.`, `_` and `-`,
//! not starting with `.` or `-` (`model-00001-of-00002.safetensors`). A shard named
//! otherwise is refused, so the procedure above reproduces the manifest byte for byte:
//! `sha256sum` escapes names holding a backslash or newline, `xargs` splits on whitespace
//! and quotes, a leading `-` reads as an option, and the shell's `*` skips a leading `.`.
//! Any name in the directory that is not UTF-8 is refused too.
//!
//! Every byte the node uses is the byte it hashed: the shards are memory-mapped once and
//! both hashed and loaded from that mapping, and the semantic and chat files are read once
//! and parsed from the bytes that were hashed. The hash is checked **before** any weight
//! is dequantised or copied to a device.
//!
//! What is loaded depends on the executor: the CPU executor runs the f32 reference model,
//! dequantised here; the CUDA executor copies the checkpoint to the device as stored, from
//! the same verified mapping ([`LoadedModel::store`]), so no reference model is built.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use eidola_engine::sampling::SamplingParams;
use eidola_engine_chat::template::CHAT_TEMPLATE_FILE;
use eidola_engine_chat::tokenizer::{GENERATION_CONFIG_FILE, TOKENIZER_FILE};
use eidola_engine_chat::{ChatTemplate, MimoTokenizer};
use eidola_engine_model::safetensors::{SEMANTIC_FILES, WeightSet};
use eidola_engine_model::{LoadOptions, ModelConfig, ModelWeights, ReferenceModel};
use sha2::{Digest, Sha256};

use crate::config::{ExecutorKind, WeightsStorage};

/// The chat artifacts the weights hash covers besides the model crate's files.
pub const CHAT_FILES: [&str; 3] = [TOKENIZER_FILE, CHAT_TEMPLATE_FILE, GENERATION_CONFIG_FILE];

/// Why the model could not be loaded. Every message is authored here from file names and
/// fixed text; none carries request data (there is none at boot).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelError(pub String);

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "model refused: {}", self.0)
    }
}

impl std::error::Error for ModelError {}

/// The canonical manifest text for `file name → sha256 hex` (see the module docs).
pub fn canonical_manifest(files: &BTreeMap<String, String>) -> Result<String, ModelError> {
    let mut out = String::new();
    for (name, digest) in files {
        if !is_safe_name(name) {
            return Err(ModelError(UNSAFE_NAME.into()));
        }
        out.push_str(&digest.to_ascii_lowercase());
        out.push_str("  ");
        out.push_str(name);
        out.push('\n');
    }
    Ok(out)
}

/// Whether `name` is a safe manifest name (module docs): `[A-Za-z0-9_][A-Za-z0-9._-]*`.
pub fn is_safe_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

const UNSAFE_NAME: &str = "a weights file name has characters other than ASCII letters, digits, \
     `.`, `_` and `-`, or starts with `.` or `-`";

/// SHA-256 (lowercase hex) of the canonical manifest.
pub fn weights_hash(files: &BTreeMap<String, String>) -> Result<String, ModelError> {
    Ok(hex::encode(Sha256::digest(
        canonical_manifest(files)?.as_bytes(),
    )))
}

/// Sampling defaults from `generation_config.json` (applied when a request leaves a
/// parameter out), as vLLM applies a model's generation config.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationDefaults {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
}

impl GenerationDefaults {
    fn parse(bytes: &[u8]) -> Result<Self, ModelError> {
        let bad = |m: &str| ModelError(format!("{GENERATION_CONFIG_FILE}: {m}"));
        let v: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|_| bad("not valid JSON"))?;
        let float = |key: &str, default: f32| -> Result<f32, ModelError> {
            match v.get(key) {
                None | Some(serde_json::Value::Null) => Ok(default),
                Some(x) => x
                    .as_f64()
                    .map(|f| f as f32)
                    .ok_or_else(|| bad(&format!("{key} must be a number"))),
            }
        };
        let top_k = match v.get("top_k") {
            None | Some(serde_json::Value::Null) => 0,
            Some(x) => x
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| bad("top_k must be a non-negative integer"))?,
        };
        let d = GenerationDefaults {
            temperature: float("temperature", 1.0)?,
            top_p: float("top_p", 1.0)?,
            top_k,
        };
        SamplingParams::new(d.temperature, d.top_k, d.top_p, 0.0, 0)
            .map_err(|e| bad(&e.to_string()))?;
        Ok(d)
    }
}

/// The model this node serves, verified against its expected weights hash and loaded.
///
/// Only [`LoadedModel::load`] constructs one, and it checks the hash first, so holding a
/// `LoadedModel` means holding weights whose identity was verified.
pub struct LoadedModel {
    weights_hash: String,
    storage: WeightsStorage,
    config: ModelConfig,
    /// The verified shards, memory-mapped.
    store: Arc<WeightSet>,
    /// The f32 reference model, loaded for the CPU executor only.
    reference: Option<Arc<ReferenceModel>>,
    tokenizer: MimoTokenizer,
    template: ChatTemplate,
    defaults: GenerationDefaults,
}

impl std::fmt::Debug for LoadedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedModel")
            .field("weights_hash", &self.weights_hash)
            .finish_non_exhaustive()
    }
}

impl LoadedModel {
    /// Opens `dir`, checks its storage against `storage`, computes its weights hash,
    /// refuses unless it equals `expected` (lowercase hex), and only then parses the
    /// configuration and chat artifacts and, for the CPU executor, loads the reference
    /// model (the CUDA executor loads the shards itself, from [`LoadedModel::store`]).
    pub fn load(
        dir: &Path,
        expected: &str,
        storage: WeightsStorage,
        executor: ExecutorKind,
    ) -> Result<Self, ModelError> {
        // Storage first, before any weights file is opened, mapped or parsed: the
        // directory, then (from its listing, which reads only names) every file the node
        // will read.
        //
        // The directory is resolved once (`PinnedDir`), and every check, listing and open
        // below goes through that resolution, never through `dir` again: a symlink on the
        // configured path retargeted after the check cannot redirect what is opened.
        let pinned = PinnedDir::open(dir)?;
        let dir = pinned.path();
        let verified = storage == WeightsStorage::VerifiedReadonly;
        if verified {
            crate::storage::require_immutable(dir, "the weights directory")?;
        }
        let inventory = Inventory::list(dir)?;
        if verified {
            for name in inventory.all() {
                require_regular_file(dir, name)?;
                crate::storage::require_immutable(&dir.join(name), name)?;
            }
        }
        let store = WeightSet::open_dir(dir)
            .map_err(|e| ModelError(format!("cannot open the weights: {e}")))?;
        let mut opened: Vec<&str> = store.file_names();
        opened.sort_unstable();
        if opened
            != inventory
                .shards
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        {
            return Err(ModelError(
                "the weights directory changed while it was being opened".into(),
            ));
        }
        let mut manifest = store.sha256_manifest();
        if !manifest.contains_key("config.json") {
            return Err(ModelError("config.json is missing".into()));
        }
        let mut chat: BTreeMap<&str, Vec<u8>> = BTreeMap::new();
        for name in CHAT_FILES {
            let path = dir.join(name);
            let bytes = std::fs::read(&path)
                .map_err(|e| ModelError(format!("cannot read {name}: {}", e.kind())))?;
            manifest.insert(name.to_string(), hex::encode(Sha256::digest(&bytes)));
            chat.insert(name, bytes);
        }
        let actual = weights_hash(&manifest)?;
        if actual != expected {
            return Err(ModelError(format!(
                "the weights hash is {actual}, but the configuration expects {expected}"
            )));
        }

        let config = store
            .model_config()
            .map_err(|e| ModelError(format!("config.json: {e}")))?;
        let store = Arc::new(store);
        let reference = match executor {
            ExecutorKind::Cpu => {
                let weights =
                    ModelWeights::load(store.clone(), config.clone(), &LoadOptions::default())
                        .map_err(|e| ModelError(format!("cannot load the weights: {e}")))?;
                Some(Arc::new(ReferenceModel::new(weights)))
            }
            #[cfg(feature = "cuda")]
            ExecutorKind::Cuda => None,
        };

        let tokenizer =
            MimoTokenizer::from_bytes(&chat[TOKENIZER_FILE], &chat[GENERATION_CONFIG_FILE])
                .map_err(|e| ModelError(e.to_string()))?;
        let template_source = String::from_utf8(chat.remove(CHAT_TEMPLATE_FILE).expect("read"))
            .map_err(|_| ModelError(format!("{CHAT_TEMPLATE_FILE} is not UTF-8")))?;
        let template =
            ChatTemplate::from_source(template_source).map_err(|e| ModelError(e.to_string()))?;
        let defaults = GenerationDefaults::parse(&chat[GENERATION_CONFIG_FILE])?;

        let vocab = tokenizer.vocab_size();
        if vocab > config.vocab_size {
            return Err(ModelError(format!(
                "the tokenizer defines {vocab} ids but the model's head has only {} rows",
                config.vocab_size
            )));
        }
        Ok(LoadedModel {
            weights_hash: actual,
            storage,
            config,
            store,
            reference,
            tokenizer,
            template,
            defaults,
        })
    }

    /// The verified weights hash (lowercase hex).
    pub fn weights_hash(&self) -> &str {
        &self.weights_hash
    }

    /// The storage the weights were checked against.
    pub fn storage(&self) -> WeightsStorage {
        self.storage
    }

    /// The model's configuration (`config.json`, verified).
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The verified shards, memory-mapped: the bytes the weights hash covers.
    pub fn store(&self) -> &Arc<WeightSet> {
        &self.store
    }

    /// The f32 reference model (weights and numerics), when loaded for the CPU executor.
    pub fn reference(&self) -> Option<&Arc<ReferenceModel>> {
        self.reference.as_ref()
    }

    pub fn tokenizer(&self) -> &MimoTokenizer {
        &self.tokenizer
    }

    pub fn template(&self) -> &ChatTemplate {
        &self.template
    }

    pub fn defaults(&self) -> GenerationDefaults {
        self.defaults
    }
}

/// The weights directory, resolved once. Every later check, listing and open goes through
/// [`PinnedDir::path`]:
///
/// * on Linux, `/proc/self/fd/<n>` of an `O_PATH` descriptor opened on the configured path
///   (no read access, nothing in it opened): the kernel resolves that magic link to the
///   directory the descriptor holds, whatever the configured path (or any directory above
///   it) points to later;
/// * elsewhere (development only), the canonical path at the time of opening, free of
///   symlinks.
struct PinnedDir {
    path: std::path::PathBuf,
    #[cfg(target_os = "linux")]
    _fd: std::os::fd::OwnedFd,
}

impl PinnedDir {
    fn open(dir: &Path) -> Result<Self, ModelError> {
        let refused = |e: std::io::Error| {
            ModelError(format!("cannot open the weights directory: {}", e.kind()))
        };
        #[cfg(target_os = "linux")]
        {
            use rustix::fs::{Mode, OFlags};
            use std::os::fd::AsRawFd;
            let fd = rustix::fs::open(
                dir,
                OFlags::PATH | OFlags::DIRECTORY | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(|e| refused(e.into()))?;
            let path = std::path::PathBuf::from(format!("/proc/self/fd/{}", fd.as_raw_fd()));
            Ok(PinnedDir { path, _fd: fd })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let path = std::fs::canonicalize(dir).map_err(refused)?;
            Ok(PinnedDir { path })
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

/// Under `verified-readonly`, refuses an entry the node will read unless it is a regular
/// file (`lstat`, not followed). A symlink's target is resolved again at every open,
/// through paths outside the pinned directory that its read-only mount does not cover;
/// a regular file in a directory on a read-only superblock cannot be replaced.
fn require_regular_file(dir: &Path, name: &str) -> Result<(), ModelError> {
    let meta = std::fs::symlink_metadata(dir.join(name))
        .map_err(|e| ModelError(format!("cannot inspect {name}: {}", e.kind())))?;
    if meta.file_type().is_file() {
        Ok(())
    } else {
        Err(ModelError(format!(
            "{name} is not a regular file; verified-readonly weights may not be symlinks"
        )))
    }
}

/// The files the node will read from a weights directory, from its listing alone.
struct Inventory {
    /// `*.safetensors`, sorted by name.
    shards: Vec<String>,
    /// The semantic and chat files present.
    others: Vec<String>,
}

impl Inventory {
    /// Lists `dir`. Every entry's name must be UTF-8 (the manifest is keyed by exact
    /// names, and the offline `sha256sum` procedure works on bytes): any other name is
    /// refused here, before anything is hashed.
    fn list(dir: &Path) -> Result<Self, ModelError> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| ModelError(format!("cannot list the weights directory: {}", e.kind())))?;
        let mut names = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| {
                ModelError(format!("cannot list the weights directory: {}", e.kind()))
            })?;
            names.push(utf8_name(&entry.file_name())?.to_string());
        }
        let mut shards: Vec<String> = names
            .iter()
            .filter(|n| n.ends_with(".safetensors"))
            .cloned()
            .collect();
        shards.sort_unstable();
        // Refused from the listing, before anything is hashed.
        if !shards.iter().all(|n| is_safe_name(n)) {
            return Err(ModelError(UNSAFE_NAME.into()));
        }
        if shards.is_empty() {
            return Err(ModelError(
                "no *.safetensors files in the weights directory".into(),
            ));
        }
        let others = SEMANTIC_FILES
            .iter()
            .chain(CHAT_FILES.iter())
            .filter(|n| names.iter().any(|m| m == *n))
            .map(|n| n.to_string())
            .collect();
        Ok(Inventory { shards, others })
    }

    fn all(&self) -> impl Iterator<Item = &str> {
        self.shards.iter().chain(&self.others).map(String::as_str)
    }
}

/// A directory entry's name, refused unless it is UTF-8.
fn utf8_name(name: &std::ffi::OsStr) -> Result<&str, ModelError> {
    name.to_str().ok_or_else(|| {
        ModelError("the weights directory holds a file whose name is not UTF-8".into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_is_sha256sum_layout_sorted_by_name() {
        let mut m = BTreeMap::new();
        m.insert("b.safetensors".to_string(), "AB".repeat(32));
        m.insert("a.json".to_string(), "01".repeat(32));
        let text = canonical_manifest(&m).unwrap();
        assert_eq!(
            text,
            format!(
                "{}  a.json\n{}  b.safetensors\n",
                "01".repeat(32),
                "ab".repeat(32)
            )
        );
        assert_eq!(
            weights_hash(&m).unwrap(),
            hex::encode(Sha256::digest(text.as_bytes()))
        );
        m.insert("bad\nname".into(), "00".repeat(32));
        assert!(canonical_manifest(&m).is_err());
    }

    /// Only names the offline `sha256sum` procedure reproduces byte for byte.
    #[test]
    fn manifest_names_are_safe() {
        for ok in [
            "model-00001-of-00002.safetensors",
            "config.json",
            "model.safetensors.index.json",
            "_x.safetensors",
        ] {
            assert!(is_safe_name(ok), "{ok}");
        }
        for bad in [
            "a\\b.safetensors",
            "a b.safetensors",
            "a\tb.safetensors",
            "a\nb.safetensors",
            "a\"b.safetensors",
            "a'b.safetensors",
            "-a.safetensors",
            ".a.safetensors",
            "é.safetensors",
            "",
        ] {
            assert!(!is_safe_name(bad), "{bad:?}");
            let m = BTreeMap::from([(bad.to_string(), "00".repeat(32))]);
            assert!(canonical_manifest(&m).is_err(), "{bad:?}");
        }
        // And from the listing, before anything is hashed.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("ok.safetensors"), b"x").unwrap();
        Inventory::list(tmp.path()).unwrap();
        std::fs::write(tmp.path().join("a\\b.safetensors"), b"x").unwrap();
        let e = Inventory::list(tmp.path()).err().unwrap();
        assert!(e.to_string().contains("ASCII"), "{e}");
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_names_are_refused() {
        use std::os::unix::ffi::OsStrExt;
        let e = utf8_name(std::ffi::OsStr::from_bytes(b"\x80.safetensors")).unwrap_err();
        assert!(e.to_string().contains("not UTF-8"), "{e}");
        assert_eq!(
            utf8_name(std::ffi::OsStr::new("a.safetensors")).unwrap(),
            "a.safetensors"
        );
    }

    /// The directory is resolved once: retargeting a symlink on the configured path
    /// afterwards does not change what is listed or opened.
    #[cfg(unix)]
    #[test]
    fn the_weights_directory_is_resolved_once() {
        let tmp = tempfile::tempdir().unwrap();
        let (a, b, link) = (
            tmp.path().join("a"),
            tmp.path().join("b"),
            tmp.path().join("w"),
        );
        std::fs::create_dir(&a).unwrap();
        std::fs::create_dir(&b).unwrap();
        std::fs::write(a.join("a.safetensors"), b"a").unwrap();
        std::fs::write(b.join("b.safetensors"), b"b").unwrap();
        std::os::unix::fs::symlink(&a, &link).unwrap();
        let pinned = PinnedDir::open(&link).unwrap();
        std::fs::remove_file(&link).unwrap();
        std::os::unix::fs::symlink(&b, &link).unwrap();
        assert_eq!(
            Inventory::list(pinned.path()).unwrap().shards,
            ["a.safetensors"]
        );
        assert_eq!(
            std::fs::read(pinned.path().join("a.safetensors")).unwrap(),
            b"a"
        );
    }

    /// Where read-only mounts are available (`storage.rs`'s env-gated tests), the pinned
    /// path gets the same verdicts as the configured one; skipped otherwise.
    #[test]
    fn a_pinned_directory_gets_the_storage_verdict_of_its_mount() {
        if let Some(dir) = std::env::var_os("EIDOLA_TEST_READ_ONLY_DIR") {
            let pinned = PinnedDir::open(Path::new(&dir)).unwrap();
            crate::storage::require_immutable(pinned.path(), "dir").unwrap();
        }
        if let Some(dir) = std::env::var_os("EIDOLA_TEST_READ_ONLY_BIND_DIR") {
            let pinned = PinnedDir::open(Path::new(&dir)).unwrap();
            assert!(crate::storage::require_immutable(pinned.path(), "dir").is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn verified_entries_must_be_regular_files() {
        let tmp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("real.safetensors"), b"x").unwrap();
        std::fs::write(outside.path().join("elsewhere"), b"x").unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("elsewhere"),
            tmp.path().join("link.safetensors"),
        )
        .unwrap();
        std::fs::create_dir(tmp.path().join("dir.safetensors")).unwrap();
        require_regular_file(tmp.path(), "real.safetensors").unwrap();
        let e = require_regular_file(tmp.path(), "link.safetensors").unwrap_err();
        assert!(e.to_string().contains("not a regular file"), "{e}");
        assert!(require_regular_file(tmp.path(), "dir.safetensors").is_err());
    }

    #[test]
    fn generation_defaults_are_validated() {
        let d = GenerationDefaults::parse(br#"{"temperature": 0.6, "top_p": 0.95}"#).unwrap();
        assert_eq!((d.temperature, d.top_p, d.top_k), (0.6, 0.95, 0));
        let d = GenerationDefaults::parse(b"{}").unwrap();
        assert_eq!((d.temperature, d.top_p, d.top_k), (1.0, 1.0, 0));
        assert!(GenerationDefaults::parse(br#"{"top_p": 0}"#).is_err());
        assert!(GenerationDefaults::parse(br#"{"temperature": "hot"}"#).is_err());
    }
}
