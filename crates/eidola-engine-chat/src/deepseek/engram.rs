//! The Engram compressed-vocabulary map: every token id onto a smaller id
//! space in which tokens that normalise alike share an id. Engram layers hash
//! n-grams of these ids, and the map's size seeds their hash multipliers.
//!
//! The reference derives it from the tokenizer:
//! `build_compressed_token_map` in `inference/engram.py` of
//! `deepseek-ai/DeepSeek-V4.1-Flash-1017` (MIT). For each id in order, the id
//! decoded alone (raw, special tokens kept) is the key, normalised by the
//! `tokenizers` sequence NFKC → NFD → StripAccents → Lowercase →
//! Replace(`[ \t\r\n]+` → ` `) → Replace(`^ $` → U+E000) → Strip →
//! Replace(U+E000 → ` `), or left as it is when normalising empties it; a
//! decode holding U+FFFD (a partial UTF-8 byte token) is keyed by its raw
//! token string instead. Classes are numbered in first-seen order.
//!
//! The derivation runs the same `tokenizers` code the reference runs: the
//! decoder and these normalizers are that crate's, at the version the
//! reference was checked with. What can differ between the two builds is
//! listed in the crate's `AGENTS.md` (DeepSeek-V4.1 → Engram map); the test
//! below holds the result equal to the reference's for every id of the pinned
//! tokenizer.

use std::collections::HashMap;

use tokenizers::normalizers::replace::ReplacePattern;
use tokenizers::normalizers::{
    Lowercase, NFD, NFKC, NormalizerWrapper, Replace, Sequence, Strip, StripAccents,
};
use tokenizers::{NormalizedString, Normalizer};

use crate::{ChatError, ChatFormat, Tokenizer};

/// The private-use character that carries a lone space through `Strip`.
const SENTINEL: &str = "\u{E000}";

/// The compressed vocabulary of a tokenizer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedVocab {
    /// `map[id]` is the class of token `id`; one entry per token id.
    pub map: Vec<u32>,
    /// The number of classes (99,092 for the released tokenizer).
    pub classes: usize,
}

/// The reference's normalizer sequence.
fn normalizer() -> Result<Sequence, ChatError> {
    let replace = |pattern: ReplacePattern, content: &str| {
        Replace::new(pattern, content)
            .map(NormalizerWrapper::from)
            .map_err(|e| ChatError::InvalidArtifact(format!("Engram normalizer: {e}")))
    };
    Ok(Sequence::new(vec![
        NFKC.into(),
        NFD.into(),
        StripAccents.into(),
        Lowercase.into(),
        replace(ReplacePattern::Regex(r"[ \t\r\n]+".into()), " ")?,
        replace(ReplacePattern::Regex(r"^ $".into()), SENTINEL)?,
        Strip::new(true, true).into(),
        replace(ReplacePattern::String(SENTINEL.into()), " ")?,
    ]))
}

/// The compressed vocabulary of a DeepSeek-V4.1 tokenizer, equal to the
/// reference's `build_compressed_token_map` (module docs). Refuses a
/// tokenizer that is not one pinned for the format: the equality is a tested
/// property of those files only.
pub fn compressed_vocab(tokenizer: &Tokenizer) -> Result<CompressedVocab, ChatError> {
    if !ChatFormat::DeepSeekV4_1
        .pinned_tokenizer_sha256()
        .contains(&tokenizer.sha256())
    {
        return Err(ChatError::UnpinnedArtifact {
            what: "tokenizer.json",
            sha256: tokenizer.sha256().to_string(),
        });
    }
    let normalizer = normalizer()?;
    let backend = tokenizer.backend();
    let failed = |id: u32, e: &dyn std::fmt::Display| {
        ChatError::InvalidArtifact(format!("Engram map: token {id}: {e}"))
    };
    let mut classes: HashMap<String, u32> = HashMap::new();
    let mut map = Vec::with_capacity(tokenizer.vocab_size());
    for id in 0..tokenizer.vocab_size() as u32 {
        let text = backend.decode(&[id], false).map_err(|e| failed(id, &e))?;
        let key = if text.contains('\u{FFFD}') {
            backend
                .id_to_token(id)
                .ok_or_else(|| failed(id, &"no token"))?
        } else {
            let mut normalized = NormalizedString::from(text.as_str());
            normalizer
                .normalize(&mut normalized)
                .map_err(|e| failed(id, &e))?;
            if normalized.get().is_empty() {
                text
            } else {
                normalized.get().to_owned()
            }
        };
        let next = classes.len() as u32;
        map.push(*classes.entry(key).or_insert(next));
    }
    Ok(CompressedVocab {
        classes: classes.len(),
        map,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::path::PathBuf;

    fn fixtures() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/deepseek")
    }

    fn deepseek() -> Tokenizer {
        let mut json = Vec::new();
        flate2::read::GzDecoder::new(
            std::fs::read(fixtures().join("tokenizer.json.gz"))
                .unwrap()
                .as_slice(),
        )
        .read_to_end(&mut json)
        .unwrap();
        let config = std::fs::read(fixtures().join("config.json")).unwrap();
        Tokenizer::for_format(ChatFormat::DeepSeekV4_1, &json, &config).unwrap()
    }

    /// The reference's map (`token_map.json`, `token_map.bin.gz`: one u32 per
    /// id, 0 where the id opens the next class and 1 + its class otherwise).
    fn reference() -> (Vec<u32>, usize, serde_json::Value) {
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fixtures().join("token_map.json")).unwrap())
                .unwrap();
        let mut bytes = Vec::new();
        flate2::read::GzDecoder::new(
            std::fs::read(fixtures().join("token_map.bin.gz"))
                .unwrap()
                .as_slice(),
        )
        .read_to_end(&mut bytes)
        .unwrap();
        let mut opened = 0;
        let map = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| match u32::from_le_bytes(*b) {
                0 => {
                    opened += 1;
                    opened - 1
                }
                n => {
                    assert!(n <= opened, "a class used before it opens");
                    n - 1
                }
            })
            .collect();
        (map, opened as usize, meta)
    }

    #[test]
    fn equals_the_reference_for_every_token() {
        let tokenizer = deepseek();
        let (expected, opened, meta) = reference();
        assert_eq!(meta["meta"]["tokenizer_sha256"], tokenizer.sha256());
        assert_eq!(meta["vocab_size"], tokenizer.vocab_size());
        assert_eq!(meta["classes"], 99_092);
        assert_eq!(opened, 99_092);

        let derived = compressed_vocab(&tokenizer).unwrap();
        assert_eq!(derived.classes, 99_092);
        assert_eq!(derived.map.len(), expected.len());
        let wrong: Vec<_> = (0..expected.len())
            .filter(|&id| derived.map[id] != expected[id])
            .take(20)
            .collect();
        assert!(wrong.is_empty(), "ids whose class differs: {wrong:?}");
    }

    fn normalize(text: &str) -> String {
        let mut n = NormalizedString::from(text);
        normalizer().unwrap().normalize(&mut n).unwrap();
        n.get().to_owned()
    }

    /// The sequence on the cases each step decides, as the reference's
    /// `normalize_str` gives them (Python `tokenizers` 0.23.2).
    #[test]
    fn the_normalizer_steps() {
        for (text, expected) in [
            // NFKC folds compatibility forms; StripAccents drops the marks NFD
            // splits off; Lowercase is per character (`char::to_lowercase`, so
            // no final sigma).
            (" The", "the"),
            ("THÉ", "the"),
            ("ﬁ", "fi"),
            ("ＡＢＣ", "abc"),
            ("İ", "i"),
            ("ẞ", "ß"),
            ("Σ", "σ"),
            // Runs of ASCII whitespace become one space, which Strip removes
            // at the ends; a key that is only such whitespace keeps one space.
            ("a \t\r\n b", "a b"),
            ("a\nb", "a b"),
            ("\n\n", " "),
            ("  ", " "),
            // NFKC makes U+00A0 and U+3000 a space first.
            ("\u{A0}", " "),
            ("\u{3000}", " "),
            // Other Unicode White_Space reaches only Strip, which removes it
            // (the reference then keys the token by its decoded text).
            ("\u{2028}", ""),
            ("\u{85}", ""),
            ("\u{1680}", ""),
            ("x\u{2028}", "x"),
            // Not whitespace.
            ("\u{200B}", "\u{200B}"),
            // The sentinel's own character comes back as a space.
            ("\u{E000}", " "),
            (" \u{E000} ", " "),
        ] {
            assert_eq!(normalize(text), expected, "{text:?}");
        }
    }

    #[test]
    fn refuses_another_formats_tokenizer() {
        let mimo = crate::tokenizer::tests::synthetic();
        assert!(matches!(
            compressed_vocab(&mimo),
            Err(ChatError::UnpinnedArtifact { .. })
        ));
    }
}
