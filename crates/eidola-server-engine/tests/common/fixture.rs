//! A servable synthetic model directory, derived in Rust from the committed fixtures.
//!
//! The model crate's synthetic MiMo fixture has a 256-row head, but the node serves
//! through the real (pinned) MiMo tokenizer and template, whose ids reach 151,674. This
//! derives a model from the fixture with `embed_tokens` and `lm_head` widened to the real
//! padded head (152,576 rows): the fixture's 256 rows first, then deterministic
//! pseudo-random rows at the same scale. Every other tensor is copied byte for byte. The
//! chat artifacts are the chat crate's pinned fixture copies.
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

/// Recorded inside a built directory: the fingerprint of the inputs it was built from.
/// (Not a file the node reads, so it is outside the weights hash.)
pub const FINGERPRINT_FILE: &str = ".fixture-fingerprint";

/// Every input file of the derivation.
fn input_files() -> Vec<PathBuf> {
    let tiny = crates_dir().join("eidola-engine-model/tests/fixtures/tiny");
    let chat = crates_dir().join("eidola-engine-chat/tests/fixtures");
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
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    let mut field = |bytes: &[u8]| {
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
    };
    field(GENERATOR_VERSION.as_bytes());
    field(include_bytes!("fixture.rs"));
    for path in input_files() {
        field(path.file_name().unwrap().as_encoded_bytes());
        field(&std::fs::read(&path)?);
    }
    Ok(hex::encode(h.finalize()))
}

/// Makes sure `out` holds the derived model built from the current inputs. A directory
/// whose recorded fingerprint differs (or is missing) is rebuilt: the new model is written
/// into a sibling directory, fingerprint last, and swapped into place by renames, so `out`
/// never holds a partial or mismatched model.
pub fn ensure_dev_model(out: &Path) -> std::io::Result<()> {
    let fingerprint = inputs_fingerprint()?;
    let current = |dir: &Path| {
        std::fs::read_to_string(dir.join(FINGERPRINT_FILE)).is_ok_and(|f| f == fingerprint)
    };
    if current(out) {
        return Ok(());
    }
    let pid = std::process::id();
    let partial = out.with_extension(format!("partial-{pid}"));
    if partial.exists() {
        std::fs::remove_dir_all(&partial)?;
    }
    write_dev_model(&partial)?;
    std::fs::write(partial.join(FINGERPRINT_FILE), &fingerprint)?;
    if out.exists() {
        let stale = out.with_extension(format!("stale-{pid}"));
        if std::fs::rename(out, &stale).is_ok() {
            std::fs::remove_dir_all(&stale)?;
        }
    }
    match std::fs::rename(&partial, out) {
        // Another process put a current model in place first.
        Err(_) if current(out) => std::fs::remove_dir_all(&partial),
        r => r,
    }
}

/// Writes the derived model into `out`.
pub fn write_dev_model(out: &Path) -> std::io::Result<()> {
    let tiny = crates_dir().join("eidola-engine-model/tests/fixtures/tiny");
    let chat = crates_dir().join("eidola-engine-chat/tests/fixtures");
    std::fs::create_dir_all(out)?;

    // config.json: the fixture's, with the widened head.
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(tiny.join("config.json"))?)?;
    config["vocab_size"] = WIDE_VOCAB.into();
    std::fs::write(out.join("config.json"), serde_json::to_vec_pretty(&config)?)?;

    // model.safetensors: every tensor copied, the two vocabulary matrices widened.
    let bytes = std::fs::read(tiny.join("model.safetensors"))?;
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
        if WIDENED.contains(&name.as_str()) {
            assert_eq!(entry["dtype"], "BF16");
            let (rows, cols) = (shape[0] as usize, shape[1] as usize);
            let scale = tensor
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16(u16::from_le_bytes(*b)).abs())
                .fold(0.0f32, f32::max);
            let mut rng = SplitMix(0xE1D0_1A00 ^ name.len() as u64);
            for _ in rows * cols..WIDE_VOCAB * cols {
                let u = (rng.next() >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
                let x = (2.0 * u - 1.0) * scale;
                tensor.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
            }
            shape[0] = WIDE_VOCAB as u64;
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
    let mut header_bytes = serde_json::to_vec(&serde_json::Value::Object(new_header))?;
    while header_bytes.len() % 8 != 0 {
        header_bytes.push(b' ');
    }
    let mut file = Vec::with_capacity(8 + header_bytes.len() + new_data.len());
    file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file.extend_from_slice(&header_bytes);
    file.extend_from_slice(&new_data);
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
