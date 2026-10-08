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
//! A file name containing a newline or that is not UTF-8 is refused, so the manifest is
//! unambiguous.
//!
//! Every byte the node uses is the byte it hashed: the shards are memory-mapped once and
//! both hashed and loaded from that mapping, and the semantic and chat files are read once
//! and parsed from the bytes that were hashed. The hash is checked **before** any weight
//! is dequantised.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use eidola_engine::sampling::SamplingParams;
use eidola_engine_chat::template::CHAT_TEMPLATE_FILE;
use eidola_engine_chat::tokenizer::{GENERATION_CONFIG_FILE, TOKENIZER_FILE};
use eidola_engine_chat::{ChatTemplate, MimoTokenizer};
use eidola_engine_model::safetensors::WeightSet;
use eidola_engine_model::{LoadOptions, ModelWeights, ReferenceModel};
use sha2::{Digest, Sha256};

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
        if name.contains('\n') {
            return Err(ModelError("a weights file name contains a newline".into()));
        }
        out.push_str(&digest.to_ascii_lowercase());
        out.push_str("  ");
        out.push_str(name);
        out.push('\n');
    }
    Ok(out)
}

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
    model: Arc<ReferenceModel>,
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
    /// Opens `dir`, computes its weights hash, refuses unless it equals `expected`
    /// (lowercase hex), and only then loads the weights and chat artifacts.
    pub fn load(dir: &Path, expected: &str) -> Result<Self, ModelError> {
        let store = WeightSet::open_dir(dir)
            .map_err(|e| ModelError(format!("cannot open the weights: {e}")))?;
        if store.file_names().is_empty() {
            return Err(ModelError(
                "no *.safetensors files in the weights directory".into(),
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
        let weights = ModelWeights::load(Arc::new(store), config, &LoadOptions::default())
            .map_err(|e| ModelError(format!("cannot load the weights: {e}")))?;
        let model = Arc::new(ReferenceModel::new(weights));

        let tokenizer =
            MimoTokenizer::from_bytes(&chat[TOKENIZER_FILE], &chat[GENERATION_CONFIG_FILE])
                .map_err(|e| ModelError(e.to_string()))?;
        let template_source = String::from_utf8(chat.remove(CHAT_TEMPLATE_FILE).expect("read"))
            .map_err(|_| ModelError(format!("{CHAT_TEMPLATE_FILE} is not UTF-8")))?;
        let template =
            ChatTemplate::from_source(template_source).map_err(|e| ModelError(e.to_string()))?;
        let defaults = GenerationDefaults::parse(&chat[GENERATION_CONFIG_FILE])?;

        let vocab = tokenizer.vocab_size();
        if vocab > model.weights.config.vocab_size {
            return Err(ModelError(format!(
                "the tokenizer defines {vocab} ids but the model's head has only {} rows",
                model.weights.config.vocab_size
            )));
        }
        Ok(LoadedModel {
            weights_hash: actual,
            model,
            tokenizer,
            template,
            defaults,
        })
    }

    /// The verified weights hash (lowercase hex).
    pub fn weights_hash(&self) -> &str {
        &self.weights_hash
    }

    /// The reference model (weights and numerics).
    pub fn model(&self) -> &Arc<ReferenceModel> {
        &self.model
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
