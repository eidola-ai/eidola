//! Tokenizer and incremental detokenizer.
//!
//! Encoding is the Hugging Face `tokenizers` crate built with its pure-Rust
//! regex backend (`fancy-regex`); the Python reference build uses Oniguruma,
//! so equivalence is a tested property of the pinned `tokenizer.json`, not an
//! assumption (see the crate's AGENTS.md).
//!
//! Decoding does not go through `tokenizers`: every token id is mapped once to
//! the exact bytes the byte-level decoder would contribute for it, and the
//! [`Detokenizer`] turns a stream of those bytes into text with the same
//! lossy UTF-8 rule `tokenizers` applies to a whole sequence. Streaming output
//! concatenated therefore equals `tokenizer.decode(ids)` exactly, while never
//! emitting half a character.

use std::collections::HashMap;
use std::path::Path;

use crate::error::ChatError;
use crate::template::sha256_hex;

/// File name of the tokenizer inside a model directory.
pub const TOKENIZER_FILE: &str = "tokenizer.json";
/// File name of the generation config inside a model directory.
pub const GENERATION_CONFIG_FILE: &str = "generation_config.json";

/// SHA-256 of the MiMo-V2.6 `tokenizer.json` (identical in Flash and Pro).
pub const MIMO_V2_6_TOKENIZER_SHA256: &str =
    "ff15eb925890d6b71b5160de4b846fbd13178438ab463b38ecc953e8cd1dcb3e";

/// Every `tokenizer.json` this crate will load.
pub const PINNED_TOKENIZER_SHA256: &[&str] = &[MIMO_V2_6_TOKENIZER_SHA256];

/// The MiMo tokenizer plus the per-token byte table used for decoding.
pub struct MimoTokenizer {
    inner: tokenizers::Tokenizer,
    /// Bytes each id contributes to decoded text; every id below its length is a
    /// token.
    token_bytes: Vec<Box<[u8]>>,
    /// Ids of added tokens flagged `special` (skipped when decoding).
    special: Vec<bool>,
    eos: Vec<u32>,
    sha256: String,
}

impl std::fmt::Debug for MimoTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MimoTokenizer")
            .field("sha256", &self.sha256)
            .field("vocab_size", &self.token_bytes.len())
            .field("eos", &self.eos)
            .finish_non_exhaustive()
    }
}

impl MimoTokenizer {
    /// Loads `tokenizer.json` and `generation_config.json` from a model
    /// directory. The tokenizer must match a pinned hash.
    pub fn from_model_dir(dir: &Path) -> Result<MimoTokenizer, ChatError> {
        let read = |name: &str| {
            let path = dir.join(name);
            std::fs::read(&path).map_err(|e| ChatError::Io {
                path: path.display().to_string(),
                message: e.to_string(),
            })
        };
        let tokenizer_json = read(TOKENIZER_FILE)?;
        let generation_config = read(GENERATION_CONFIG_FILE)?;
        MimoTokenizer::from_bytes(&tokenizer_json, &generation_config)
    }

    /// Builds the tokenizer from file contents, refusing an unpinned
    /// `tokenizer.json`.
    pub fn from_bytes(
        tokenizer_json: &[u8],
        generation_config_json: &[u8],
    ) -> Result<MimoTokenizer, ChatError> {
        let sha256 = sha256_hex(tokenizer_json);
        if !PINNED_TOKENIZER_SHA256.contains(&sha256.as_str()) {
            return Err(ChatError::UnpinnedArtifact {
                what: "tokenizer.json",
                sha256,
            });
        }
        MimoTokenizer::build(tokenizer_json, generation_config_json, sha256)
    }

    /// Like [`MimoTokenizer::from_bytes`] without the pin, for synthetic test
    /// tokenizers.
    #[cfg(test)]
    pub(crate) fn from_unpinned_bytes(
        tokenizer_json: &[u8],
        generation_config_json: &[u8],
    ) -> Result<MimoTokenizer, ChatError> {
        let sha256 = sha256_hex(tokenizer_json);
        MimoTokenizer::build(tokenizer_json, generation_config_json, sha256)
    }

    fn build(
        tokenizer_json: &[u8],
        generation_config_json: &[u8],
        sha256: String,
    ) -> Result<MimoTokenizer, ChatError> {
        let inner = tokenizers::Tokenizer::from_bytes(tokenizer_json)
            .map_err(|e| ChatError::InvalidArtifact(format!("tokenizer.json: {e}")))?;
        let eos = parse_eos(generation_config_json)?;

        let char_bytes = byte_level_decoder_table();
        let vocab = inner.get_vocab(true);
        let size = vocab.values().copied().max().map_or(0, |m| m as usize + 1);
        let mut token_bytes: Vec<Box<[u8]>> = Vec::with_capacity(size);
        for id in 0..size as u32 {
            // `id_to_token` consults the added vocabulary first, exactly as
            // `Tokenizer::decode` does. The ids must be dense: `vocab_size` is the
            // sampleable vocabulary the engine is configured with, so a hole would be
            // an id the model may emit that has no token.
            let token = inner.id_to_token(id).ok_or_else(|| {
                ChatError::InvalidArtifact(format!("tokenizer.json: id {id} has no token"))
            })?;
            token_bytes.push(byte_level_token_bytes(&token, &char_bytes).into());
        }
        let mut special = vec![false; size];
        for (id, added) in inner.get_added_tokens_decoder() {
            if added.special && (id as usize) < size {
                special[id as usize] = true;
            }
        }
        for &id in &eos {
            if id as usize >= size {
                return Err(ChatError::InvalidArtifact(format!(
                    "generation_config.json: eos id {id} is outside the vocabulary"
                )));
            }
        }
        Ok(MimoTokenizer {
            inner,
            token_bytes,
            special,
            eos,
            sha256,
        })
    }

    /// Hex SHA-256 of the loaded `tokenizer.json`.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Encodes a rendered prompt. Added tokens (`<|im_start|>`, `<think>`,
    /// `<tool_call>`, …) appearing in the text are matched as single tokens;
    /// no BOS or other special tokens are inserted.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, ChatError> {
        self.inner
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|e| ChatError::InvalidInput(format!("tokenization failed: {e}")))
    }

    /// Decodes a complete sequence, identical to `tokenizers`'
    /// `decode(ids, skip_special_tokens)` for ids the tokenizer defines. An id it does
    /// not define is an error ([`ChatError::UnknownToken`]), where `tokenizers` would
    /// silently drop it.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, ChatError> {
        let mut detok = Detokenizer::new(skip_special_tokens);
        let mut out = String::new();
        for &id in ids {
            out.push_str(&detok.push(self, id)?);
        }
        out.push_str(&detok.finish());
        Ok(out)
    }

    /// The bytes token `id` contributes to decoded text, or `None` for an id the
    /// tokenizer does not define (`id >= vocab_size()`): a padded logit row has no
    /// bytes, and must not be mistaken for a token that decodes to nothing.
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.token_bytes.get(id as usize).map(|b| &**b)
    }

    /// Whether `id` is an added token flagged special.
    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(false)
    }

    /// End-of-sequence ids from `generation_config.json`.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }

    /// Whether `id` ends generation.
    pub fn is_eos(&self, id: u32) -> bool {
        self.eos.contains(&id)
    }

    /// The id of a token string (vocabulary or added token).
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// The number of token ids, `0..vocab_size()`, every one of them defined (base
    /// vocabulary plus added tokens; 151,675 for MiMo-V2.6). This is the sampleable
    /// vocabulary the engine must be configured with (`ModelSpec::sampleable_vocab_size`):
    /// the model's head is padded past it, and those rows are not tokens.
    pub fn vocab_size(&self) -> usize {
        self.token_bytes.len()
    }
}

fn parse_eos(generation_config_json: &[u8]) -> Result<Vec<u32>, ChatError> {
    let invalid = |m: &str| ChatError::InvalidArtifact(format!("generation_config.json: {m}"));
    let config: serde_json::Value =
        serde_json::from_slice(generation_config_json).map_err(|e| invalid(&e.to_string()))?;
    let as_id = |v: &serde_json::Value| {
        v.as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| invalid("eos_token_id entries must be token ids"))
    };
    match config.get("eos_token_id") {
        Some(serde_json::Value::Array(ids)) if !ids.is_empty() => ids.iter().map(as_id).collect(),
        Some(v @ serde_json::Value::Number(_)) => Ok(vec![as_id(v)?]),
        _ => Err(invalid("missing eos_token_id")),
    }
}

/// The GPT-2 byte-level alphabet, inverted: each printable stand-in character
/// maps back to its byte.
fn byte_level_decoder_table() -> HashMap<char, u8> {
    let mut printable: Vec<u32> = ('!' as u32..='~' as u32).collect();
    printable.extend('¡' as u32..='¬' as u32);
    printable.extend('®' as u32..='ÿ' as u32);
    let mut table = HashMap::with_capacity(256);
    let mut extra = 0;
    for byte in 0..=255u32 {
        let c = if printable.contains(&byte) {
            byte
        } else {
            extra += 1;
            255 + extra
        };
        table.insert(char::from_u32(c).expect("valid char"), byte as u8);
    }
    table
}

/// `tokenizers`' byte-level decoder rule for one token: map every character
/// through the alphabet, or, if any character is outside it, take the token's
/// own UTF-8 bytes.
fn byte_level_token_bytes(token: &str, table: &HashMap<char, u8>) -> Vec<u8> {
    token
        .chars()
        .map(|c| table.get(&c).copied())
        .collect::<Option<Vec<u8>>>()
        .unwrap_or_else(|| token.as_bytes().to_vec())
}

/// Incremental, UTF-8-safe detokenizer.
///
/// Bytes that do not yet form a complete character are held back until the
/// next token completes them. Invalid sequences become U+FFFD using the same
/// maximal-subpart rule as `String::from_utf8_lossy` (which `tokenizers`
/// applies to whole sequences), so the concatenated output is independent of
/// how the ids were split into calls.
#[derive(Clone, Debug, Default)]
pub struct Detokenizer {
    pending: Vec<u8>,
    skip_special_tokens: bool,
}

impl Detokenizer {
    pub fn new(skip_special_tokens: bool) -> Detokenizer {
        Detokenizer {
            pending: Vec::new(),
            skip_special_tokens,
        }
    }

    /// Feeds one token and returns the text it completes (possibly empty). An id the
    /// tokenizer does not define is refused, leaving the detokenizer unchanged.
    pub fn push(&mut self, tokenizer: &MimoTokenizer, id: u32) -> Result<String, ChatError> {
        let bytes = tokenizer
            .token_bytes(id)
            .ok_or(ChatError::UnknownToken(id))?;
        if self.skip_special_tokens && tokenizer.is_special(id) {
            return Ok(String::new());
        }
        Ok(self.push_bytes(bytes))
    }

    /// Feeds raw bytes and returns the text they complete.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        let mut consumed = 0;
        loop {
            match std::str::from_utf8(&self.pending[consumed..]) {
                Ok(valid) => {
                    out.push_str(valid);
                    consumed = self.pending.len();
                    break;
                }
                Err(e) => {
                    let valid_end = consumed + e.valid_up_to();
                    out.push_str(
                        std::str::from_utf8(&self.pending[consumed..valid_end])
                            .expect("prefix validated"),
                    );
                    match e.error_len() {
                        Some(bad) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            consumed = valid_end + bad;
                        }
                        None => {
                            // An incomplete but so-far-valid sequence: wait.
                            consumed = valid_end;
                            break;
                        }
                    }
                }
            }
        }
        self.pending.drain(..consumed);
        out
    }

    /// Ends the stream: held-back bytes that never completed a character
    /// become U+FFFD.
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }

    /// Number of bytes currently held back (at most 3).
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use proptest::prelude::*;

    /// A miniature byte-level BPE tokenizer with MiMo's pre-tokenizer,
    /// decoder, and added tokens, built in memory.
    pub(crate) fn synthetic_tokenizer_json() -> String {
        let table = byte_level_decoder_table();
        let mut by_byte: Vec<(u8, char)> = table.iter().map(|(c, b)| (*b, *c)).collect();
        by_byte.sort();
        let mut vocab = serde_json::Map::new();
        for (byte, c) in &by_byte {
            vocab.insert(c.to_string(), (*byte as u64).into());
        }
        // Merges producing multi-byte tokens, including one that ends inside
        // a UTF-8 sequence ("ä½" = the first two bytes of 你).
        let merges = [("h", "e"), ("he", "l"), ("ä", "½"), ("Ġ", "w")];
        let mut next = 256u64;
        let mut merge_list = Vec::new();
        for (a, b) in merges {
            vocab.insert(format!("{a}{b}"), next.into());
            merge_list.push(serde_json::json!([a, b]));
            next += 1;
        }
        let added = [
            ("<|endoftext|>", true),
            ("<|im_start|>", true),
            ("<|im_end|>", true),
            ("<think>", false),
            ("</think>", false),
            ("<tool_call>", false),
            ("</tool_call>", false),
        ];
        let added_tokens: Vec<_> = added
            .iter()
            .enumerate()
            .map(|(i, (content, special))| {
                serde_json::json!({
                    "id": next + i as u64, "content": content, "single_word": false,
                    "lstrip": false, "rstrip": false, "normalized": false, "special": special
                })
            })
            .collect();
        let byte_level = serde_json::json!({
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false
        });
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added_tokens,
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
                byte_level.clone()
            ]},
            "post_processor": byte_level.clone(),
            "decoder": byte_level,
            "model": {
                "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
                "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false,
                "ignore_merges": false, "vocab": vocab, "merges": merge_list
            }
        })
        .to_string()
    }

    pub(crate) fn synthetic() -> MimoTokenizer {
        MimoTokenizer::from_unpinned_bytes(
            synthetic_tokenizer_json().as_bytes(),
            br#"{"eos_token_id": [260, 262]}"#,
        )
        .unwrap()
    }

    #[test]
    fn rejects_unpinned_tokenizer() {
        let err = MimoTokenizer::from_bytes(synthetic_tokenizer_json().as_bytes(), b"{}");
        assert!(matches!(err, Err(ChatError::UnpinnedArtifact { .. })));
    }

    /// Ids past the vocabulary (a padded logit row) are refused, never decoded as
    /// nothing, and the detokenizer is left as it was.
    #[test]
    fn out_of_vocabulary_ids_are_refused() {
        let t = synthetic();
        let past = t.vocab_size() as u32;
        assert_eq!(t.token_bytes(past), None);
        assert_eq!(
            t.token_bytes(past - 1).map(<[u8]>::len).map(|n| n > 0),
            Some(true)
        );
        assert_eq!(
            t.decode(&[0, past], false),
            Err(ChatError::UnknownToken(past))
        );
        let mut detok = Detokenizer::new(false);
        // The first two bytes of 你, held back as an incomplete character.
        let partial = t.token_to_id("ä½").unwrap();
        assert_eq!(detok.push(&t, partial).unwrap(), "");
        assert_eq!(detok.pending_len(), 2);
        assert_eq!(detok.push(&t, past), Err(ChatError::UnknownToken(past)));
        assert_eq!(
            detok.push(&t, past + 1000),
            Err(ChatError::UnknownToken(past + 1000))
        );
        assert_eq!(detok.pending_len(), 2);
    }

    #[test]
    fn eos_from_generation_config() {
        let t = synthetic();
        assert_eq!(t.eos_token_ids(), &[260, 262]);
        assert!(t.is_eos(262));
        assert!(!t.is_eos(263));
    }

    #[test]
    fn decode_matches_tokenizers_for_every_single_token() {
        let t = synthetic();
        for id in 0..t.vocab_size() as u32 {
            for skip in [false, true] {
                assert_eq!(
                    t.decode(&[id], skip).unwrap(),
                    t.inner.decode(&[id], skip).unwrap(),
                    "id {id} skip {skip}"
                );
            }
        }
    }

    #[test]
    fn encode_round_trips_through_added_tokens() {
        let t = synthetic();
        let text = "<|im_start|>assistant\n<think>你好 hello</think><tool_call>😀<|im_end|>";
        let ids = t.encode(text).unwrap();
        assert!(ids.contains(&t.token_to_id("<think>").unwrap()));
        assert_eq!(t.decode(&ids, false).unwrap(), text);
        assert_eq!(
            t.decode(&ids, true).unwrap(),
            "assistant\n<think>你好 hello</think><tool_call>😀"
        );
    }

    proptest! {
        #[test]
        fn streaming_decode_equals_whole_decode(ids in proptest::collection::vec(0u32..270, 0..64), skip: bool) {
            let t = synthetic();
            let mut detok = Detokenizer::new(skip);
            let mut streamed = String::new();
            for &id in &ids {
                // An id past the vocabulary is refused and leaves the stream as it
                // was (`tokenizers` drops such ids, so the comparison still holds).
                let Ok(piece) = detok.push(&t, id) else {
                    prop_assert!(id as usize >= t.vocab_size());
                    continue;
                };
                prop_assert!(!piece.is_empty() || detok.pending_len() <= 3);
                streamed.push_str(&piece);
            }
            streamed.push_str(&detok.finish());
            prop_assert_eq!(&streamed, &t.inner.decode(&ids, skip).unwrap());
        }

        #[test]
        fn utf8_stream_equals_lossy_whole(chunks in proptest::collection::vec(proptest::collection::vec(any::<u8>(), 0..6), 0..32)) {
            let mut detok = Detokenizer::new(false);
            let mut streamed = String::new();
            let mut all = Vec::new();
            for chunk in &chunks {
                streamed.push_str(&detok.push_bytes(chunk));
                all.extend_from_slice(chunk);
            }
            streamed.push_str(&detok.finish());
            prop_assert_eq!(streamed, String::from_utf8_lossy(&all).into_owned());
        }
    }
}

/// Comparison against the reference Python tokenizer on the real MiMo
/// `tokenizer.json`. The pinned file is committed gzip-compressed under
/// `tests/fixtures/`; `EIDOLA_MIMO_MODEL_DIR` may name a model directory to
/// use instead. Either way the decompressed bytes must match the pin.
#[cfg(test)]
mod reference_tests {
    use super::*;
    use std::io::Read;
    use std::path::PathBuf;

    pub const MODEL_DIR_ENV: &str = "EIDOLA_MIMO_MODEL_DIR";

    fn fixtures_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    fn real() -> MimoTokenizer {
        if let Some(dir) = std::env::var_os(MODEL_DIR_ENV) {
            return MimoTokenizer::from_model_dir(&PathBuf::from(dir)).expect("pinned tokenizer");
        }
        let compressed = std::fs::read(fixtures_dir().join("tokenizer.json.gz")).unwrap();
        let mut tokenizer_json = Vec::new();
        flate2::read::GzDecoder::new(compressed.as_slice())
            .read_to_end(&mut tokenizer_json)
            .unwrap();
        let generation_config = std::fs::read(fixtures_dir().join(GENERATION_CONFIG_FILE)).unwrap();
        MimoTokenizer::from_bytes(&tokenizer_json, &generation_config).expect("pinned tokenizer")
    }

    fn token_fixtures() -> serde_json::Value {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/template_token_ids.json");
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn ids(v: &serde_json::Value) -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    }

    #[test]
    fn encodes_like_python_tokenizers() {
        let t = real();
        let fixtures = token_fixtures();
        assert_eq!(
            fixtures["meta"]["tokenizer_sha256"].as_str(),
            Some(t.sha256())
        );
        let cases: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/template_cases.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let mut checked = 0;
        let mut failures = Vec::new();
        for case in cases["cases"].as_array().unwrap() {
            let (Some(text), Some(expected)) = (
                case["expected"].as_str(),
                fixtures["token_ids"].get(case["name"].as_str().unwrap()),
            ) else {
                continue;
            };
            let expected = ids(expected);
            if t.encode(text).unwrap() != expected {
                failures.push(case["name"].as_str().unwrap().to_string());
            }
            checked += 1;
        }
        for pair in fixtures["strings"].as_array().unwrap() {
            let text = pair[0].as_str().unwrap();
            let expected = ids(&pair[1]);
            if t.encode(text).unwrap() != expected {
                failures.push(format!("string {text:?}"));
            }
            checked += 1;
        }
        assert!(
            failures.is_empty(),
            "{} mismatches: {failures:#?}",
            failures.len()
        );
        assert!(checked > 600);
    }

    #[test]
    fn decode_table_matches_tokenizers_for_every_id() {
        let t = real();
        assert_eq!(t.eos_token_ids(), &[151643, 151645, 151672]);
        for id in 0..t.vocab_size() as u32 {
            assert_eq!(
                t.decode(&[id], false).unwrap(),
                t.inner.decode(&[id], false).unwrap(),
                "id {id}"
            );
        }
        // Random sequences, which split multi-byte characters across tokens.
        let mut state = 0x2545f4914f6cdd1du64;
        for _ in 0..2000 {
            let len = (state % 24) as usize;
            let seq: Vec<u32> = (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state % t.vocab_size() as u64) as u32
                })
                .collect();
            for skip in [false, true] {
                assert_eq!(
                    t.decode(&seq, skip).unwrap(),
                    t.inner.decode(&seq, skip).unwrap()
                );
            }
        }
    }

    /// The vocabulary is dense, so `vocab_size` is exactly the set of ids the model may
    /// emit: the real tokenizer defines 151,643 base ids plus 32 added tokens, and the
    /// model head's 152,576 rows are padded past them.
    #[test]
    fn real_vocabulary_is_dense_with_added_tokens() {
        let t = real();
        assert_eq!(t.vocab_size(), 151_675);
        assert_eq!(t.token_to_id("<|endoftext|>"), Some(151_643));
        assert_eq!(t.token_to_id("<|mimo_audio_end|>"), Some(151_674));
        assert!(t.token_bytes(151_674).is_some());
        assert_eq!(t.token_bytes(151_675), None);
        assert_eq!(t.token_bytes(152_575), None);
    }
}
