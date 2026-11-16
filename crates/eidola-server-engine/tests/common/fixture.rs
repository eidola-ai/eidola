//! A servable synthetic model directory, derived in Rust from the committed fixtures.
//!
//! The model crate's synthetic MiMo fixture has a 256-row head, but the node serves
//! through the real (pinned) MiMo tokenizer and template, whose ids reach 151,674. This
//! derives a model from the fixture with `embed_tokens` and `lm_head` widened to the real
//! padded head (152,576 rows): the fixture's 256 rows first, then deterministic
//! pseudo-random rows at the same scale. Every other tensor is copied byte for byte. The
//! chat artifacts are the chat crate's pinned fixture copies.
//!
//! The DeepSeek-V4.1 dev model is derived the same way from the model crate's synthetic
//! DeepSeek fixture (`tests/fixtures/deepseek_tiny`): its four vocabulary-shaped tensors
//! (the embedding, the head, and the drafter's Markov embedding and head) widened from 96
//! rows to the real tokenizer's 129,280; `config.json` with that vocabulary and the real
//! image token id (129,264), and the Engram compressed vocabulary of 99,092 classes the
//! node derives from that tokenizer; and the chat crate's pinned DeepSeek-V4.1
//! `tokenizer.json`. Like the released checkpoint, its shards carry no Engram map. Every
//! other tensor is the fixture's, byte for byte.
//!
//! The output is gibberish (random weights); tests assert structure and token-level
//! equality, never semantics.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Rows of the widened head: MiMo-V2.6's padded vocabulary.
pub const WIDE_VOCAB: usize = 152_576;

const WIDENED: [&str; 2] = ["model.embed_tokens.weight", "lm_head.weight"];

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .to_path_buf()
}

/// Bumped when the derivation changes in a way this file's own bytes would not show.
const GENERATOR_VERSION: &str = "eidola-server-engine dev model v1";

/// Rows of the DeepSeek-V4.1 dev model's widened head: the released vocabulary.
pub const DEEPSEEK_VOCAB: usize = 129_280;

/// Classes of the Engram compressed vocabulary the pinned tokenizer derives.
pub const DEEPSEEK_COMPRESSED_VOCAB: usize = 99_092;

/// The released checkpoint's image token, which the text model never embeds.
pub const DEEPSEEK_IMAGE_TOKEN: u32 = 129_264;

const DEEPSEEK_WIDENED: [&str; 4] = [
    "embed.weight",
    "head.weight",
    "mtp.2.markov_head.embed.weight",
    "mtp.2.markov_head.head.weight",
];

/// Recorded inside a built directory: the fingerprint of the inputs it was built from.
/// (Not a file the node reads, so it is outside the weights hash.)
pub const FINGERPRINT_FILE: &str = ".fixture-fingerprint";

/// Every input file of the DeepSeek-V4.1 derivation.
fn deepseek_input_files() -> Vec<PathBuf> {
    let tiny = crates_dir().join("eidola-engine-model/tests/fixtures/deepseek_tiny");
    let chat = crates_dir().join("eidola-engine-chat/tests/fixtures/deepseek");
    vec![
        tiny.join("config.json"),
        tiny.join("model.safetensors"),
        chat.join("tokenizer.json.gz"),
    ]
}

/// Every input file of the derivation.
fn input_files() -> Vec<PathBuf> {
    let tiny = crates_dir().join("eidola-engine-model/tests/fixtures/mimo_tiny");
    let chat = crates_dir().join("eidola-engine-chat/tests/fixtures/mimo");
    vec![
        tiny.join("config.json"),
        tiny.join("model.safetensors"),
        chat.join("tokenizer.json.gz"),
        chat.join("chat_template.jinja"),
        chat.join("generation_config.json"),
    ]
}

/// SHA-256 over the generator version, this file's source (the derivation itself), and
/// every input file's name and bytes, each length-prefixed.
pub fn inputs_fingerprint() -> std::io::Result<String> {
    fingerprint_of(GENERATOR_VERSION, &input_files())
}

fn fingerprint_of(version: &str, inputs: &[PathBuf]) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    let mut field = |bytes: &[u8]| {
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
    };
    field(version.as_bytes());
    field(include_bytes!("fixture.rs"));
    for path in inputs {
        field(path.file_name().unwrap().as_encoded_bytes());
        field(&std::fs::read(path)?);
    }
    Ok(hex::encode(h.finalize()))
}

/// Makes sure `out` holds the derived model built from the current inputs. A directory
/// whose recorded fingerprint differs (or is missing) is rebuilt: the new model is written
/// into a sibling directory, fingerprint last, and swapped into place by renames, so `out`
/// never holds a partial or mismatched model. Rebuilds are serialized across processes by
/// an exclusive lock on a sibling file, and the fingerprint is rechecked under it, so a
/// builder never moves aside a current model another process has just published.
pub fn ensure_dev_model(out: &Path) -> std::io::Result<()> {
    ensure_built(out, &inputs_fingerprint()?, write_dev_model)
}

/// [`ensure_dev_model`] for the DeepSeek-V4.1 dev model.
pub fn ensure_deepseek_dev_model(out: &Path) -> std::io::Result<()> {
    let fingerprint = fingerprint_of(
        "eidola-server-engine deepseek dev model v1",
        &deepseek_input_files(),
    )?;
    ensure_built(out, &fingerprint, write_deepseek_dev_model)
}

fn ensure_built(
    out: &Path,
    fingerprint: &str,
    write: fn(&Path) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let current = |dir: &Path| {
        std::fs::read_to_string(dir.join(FINGERPRINT_FILE)).is_ok_and(|f| f == fingerprint)
    };
    if current(out) {
        return Ok(());
    }
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let lock = std::fs::File::create(out.with_extension("lock"))?;
    lock.lock()?;
    if current(out) {
        return Ok(());
    }
    let pid = std::process::id();
    let partial = out.with_extension(format!("partial-{pid}"));
    if partial.exists() {
        std::fs::remove_dir_all(&partial)?;
    }
    write(&partial)?;
    std::fs::write(partial.join(FINGERPRINT_FILE), fingerprint)?;
    if out.exists() {
        let stale = out.with_extension(format!("stale-{pid}"));
        if std::fs::rename(out, &stale).is_ok() {
            std::fs::remove_dir_all(&stale)?;
        }
    }
    std::fs::rename(&partial, out)
}

/// Writes the derived model into `out`.
pub fn write_dev_model(out: &Path) -> std::io::Result<()> {
    let tiny = crates_dir().join("eidola-engine-model/tests/fixtures/mimo_tiny");
    let chat = crates_dir().join("eidola-engine-chat/tests/fixtures/mimo");
    std::fs::create_dir_all(out)?;

    // config.json: the fixture's, with the widened head.
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(tiny.join("config.json"))?)?;
    config["vocab_size"] = WIDE_VOCAB.into();
    std::fs::write(out.join("config.json"), canonical_json(&config, true))?;

    // model.safetensors: every tensor copied, the two vocabulary matrices widened.
    let file = widened_safetensors(&tiny.join("model.safetensors"), &WIDENED, WIDE_VOCAB)?;
    std::fs::write(out.join("model.safetensors"), file)?;

    // The pinned chat artifacts.
    let mut tokenizer = Vec::new();
    flate2::read::GzDecoder::new(std::fs::File::open(chat.join("tokenizer.json.gz"))?)
        .read_to_end(&mut tokenizer)?;
    std::fs::write(out.join("tokenizer.json"), tokenizer)?;
    for name in ["chat_template.jinja", "generation_config.json"] {
        std::fs::copy(chat.join(name), out.join(name))?;
    }
    Ok(())
}

/// Writes the DeepSeek-V4.1 dev model into `out` (module docs).
pub fn write_deepseek_dev_model(out: &Path) -> std::io::Result<()> {
    let tiny = crates_dir().join("eidola-engine-model/tests/fixtures/deepseek_tiny");
    let chat = crates_dir().join("eidola-engine-chat/tests/fixtures/deepseek");
    std::fs::create_dir_all(out)?;

    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(tiny.join("config.json"))?)?;
    config["text_config"]["vocab_size"] = DEEPSEEK_VOCAB.into();
    config["image_token_id"] = DEEPSEEK_IMAGE_TOKEN.into();
    config["text_config"]["engram_compressed_vocab_size"] = DEEPSEEK_COMPRESSED_VOCAB.into();
    std::fs::write(out.join("config.json"), canonical_json(&config, true))?;

    let file = widened_safetensors(
        &tiny.join("model.safetensors"),
        &DEEPSEEK_WIDENED,
        DEEPSEEK_VOCAB,
    )?;
    std::fs::write(out.join("model.safetensors"), file)?;

    let mut tokenizer = Vec::new();
    flate2::read::GzDecoder::new(std::fs::File::open(chat.join("tokenizer.json.gz"))?)
        .read_to_end(&mut tokenizer)?;
    std::fs::write(out.join("tokenizer.json"), tokenizer)?;
    Ok(())
}

/// The safetensors file at `path` with the BF16 matrices named in `widened` grown to
/// `rows` rows (the original rows first, then deterministic random rows at the scale of
/// the original), every other tensor copied.
fn widened_safetensors(path: &Path, widened: &[&str], rows_to: usize) -> std::io::Result<Vec<u8>> {
    let bytes = std::fs::read(path)?;
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + header_len])?;
    let data = &bytes[8 + header_len..];
    let mut names: Vec<&String> = header.keys().filter(|k| *k != "__metadata__").collect();
    names.sort();
    let mut new_header = serde_json::Map::new();
    if let Some(m) = header.get("__metadata__") {
        new_header.insert("__metadata__".into(), m.clone());
    }
    let mut new_data: Vec<u8> = Vec::new();
    for name in names {
        let entry = &header[name.as_str()];
        let offsets = entry["data_offsets"].as_array().unwrap();
        let (start, end) = (
            offsets[0].as_u64().unwrap() as usize,
            offsets[1].as_u64().unwrap() as usize,
        );
        let mut tensor = data[start..end].to_vec();
        let mut shape: Vec<u64> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap())
            .collect();
        if widened.contains(&name.as_str()) {
            assert_eq!(entry["dtype"], "BF16");
            let (rows, cols) = (shape[0] as usize, shape[1] as usize);
            let scale = tensor
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16(u16::from_le_bytes(*b)).abs())
                .fold(0.0f32, f32::max);
            let mut rng = SplitMix(0xE1D0_1A00 ^ name.len() as u64);
            for _ in rows * cols..rows_to * cols {
                let u = (rng.next() >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
                let x = (2.0 * u - 1.0) * scale;
                tensor.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
            }
            shape[0] = rows_to as u64;
        }
        let begin = new_data.len();
        new_data.extend_from_slice(&tensor);
        new_header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": entry["dtype"],
                "shape": shape,
                "data_offsets": [begin, new_data.len()],
            }),
        );
    }
    let mut header_bytes = canonical_json(&serde_json::Value::Object(new_header), false);
    while !header_bytes.len().is_multiple_of(8) {
        header_bytes.push(b' ');
    }
    let mut file = Vec::with_capacity(8 + header_bytes.len() + new_data.len());
    file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file.extend_from_slice(&header_bytes);
    file.extend_from_slice(&new_data);
    Ok(file)
}

/// `value` serialized with every object's keys sorted, at every depth.
///
/// `serde_json::Map` keeps insertion order when anything else in the build enables
/// serde_json's `preserve_order` feature, and sorts otherwise; a `Value` serialized as is
/// would make the dev model's bytes, and its weights hash, depend on what else was built.
pub fn canonical_json(value: &serde_json::Value, pretty: bool) -> Vec<u8> {
    if pretty {
        serde_json::to_vec_pretty(&SortedKeys(value))
    } else {
        serde_json::to_vec(&SortedKeys(value))
    }
    .expect("a JSON value serializes")
}

struct SortedKeys<'a>(&'a serde_json::Value);

impl serde::Serialize for SortedKeys<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{SerializeMap, SerializeSeq};
        use serde_json::Value;
        match self.0 {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                let mut out = serializer.serialize_map(Some(keys.len()))?;
                for key in keys {
                    out.serialize_entry(key, &SortedKeys(&map[key]))?;
                }
                out.end()
            }
            Value::Array(items) => {
                let mut out = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    out.serialize_element(&SortedKeys(item))?;
                }
                out.end()
            }
            other => other.serialize(serializer),
        }
    }
}

/// The weights hash of `dir`, computed independently of the crate: `sha256sum`-style
/// lines for every hashed file, sorted by name, hashed.
pub fn weights_hash_of(dir: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut files = BTreeMap::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        let hashed = name.ends_with(".safetensors")
            || [
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "chat_template.jinja",
                "generation_config.json",
            ]
            .contains(&name.as_str());
        if hashed {
            let digest = Sha256::digest(std::fs::read(dir.join(&name)).unwrap());
            files.insert(name, hex::encode(digest));
        }
    }
    let manifest: String = files
        .iter()
        .map(|(name, digest)| format!("{digest}  {name}\n"))
        .collect();
    hex::encode(Sha256::digest(manifest.as_bytes()))
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}
