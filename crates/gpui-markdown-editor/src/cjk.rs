//! CJK script classification, for emphasis rendering.
//!
//! Markdown `*emphasis*` asks for italic, and no CJK face has one: the
//! delimiters are consumed and the glyphs render unchanged, so the reader
//! is left with *less* information than the raw asterisks carried.
//! Synthesizing an oblique is not an option at this layer and is wrong
//! typography besides — slanted Han reads as a rendering artifact.
//!
//! The convention a native reader recognizes is 着重号: a dot under each
//! emphasized character. That is what the editor paints, so an emphasis
//! run that mixes scripts renders **both** ways — Latin sub-runs stay
//! true italic, CJK sub-runs take dots. This module draws the line
//! between the two, and the narrower line between characters that take a
//! dot and the punctuation that sits between them.

/// True for characters written in a CJK script, **including** the CJK
/// punctuation that sits between them.
///
/// This is the emphasis *splitting* rule: none of these has an italic
/// face, so an italic run must not cover them. It is deliberately wider
/// than [`takes_emphasis_dot`] — a comma inside an emphasized clause
/// belongs to the same sub-run as the words around it, it simply does
/// not receive a dot of its own.
pub fn is_cjk_script(c: char) -> bool {
    matches!(c as u32,
        0x1100..=0x11FF     // Hangul Jamo
        | 0x2E80..=0x2EFF   // CJK Radicals Supplement
        | 0x2F00..=0x2FDF   // Kangxi Radicals
        | 0x2FF0..=0x2FFF   // Ideographic Description Characters
        | 0x3000..=0x303F   // CJK Symbols and Punctuation
        | 0x3040..=0x309F   // Hiragana
        | 0x30A0..=0x30FF   // Katakana
        | 0x3100..=0x312F   // Bopomofo
        | 0x3130..=0x318F   // Hangul Compatibility Jamo
        | 0x31A0..=0x31BF   // Bopomofo Extended
        | 0x31C0..=0x31EF   // CJK Strokes
        | 0x31F0..=0x31FF   // Katakana Phonetic Extensions
        | 0x3200..=0x32FF   // Enclosed CJK Letters and Months
        | 0x3300..=0x33FF   // CJK Compatibility
        | 0x3400..=0x4DBF   // CJK Unified Ideographs Extension A
        | 0x4E00..=0x9FFF   // CJK Unified Ideographs
        | 0xA960..=0xA97F   // Hangul Jamo Extended-A
        | 0xAC00..=0xD7AF   // Hangul Syllables + Jamo Extended-B
        | 0xF900..=0xFAFF   // CJK Compatibility Ideographs
        | 0xFE10..=0xFE1F   // Vertical Forms
        | 0xFE30..=0xFE4F   // CJK Compatibility Forms
        | 0xFE50..=0xFE6F   // Small Form Variants
        | 0xFF00..=0xFFEF   // Halfwidth and Fullwidth Forms
        | 0x20000..=0x3FFFF // CJK Unified Ideographs Extensions B..
    )
}

/// True for a character that takes an emphasis dot of its own.
///
/// Punctuation and spacing inside an emphasized CJK run are skipped —
/// the dots mark the characters being emphasized, not the marks between
/// them. (Fullwidth digits and Latin letters *are* marked: they are
/// characters of the sentence, set on the CJK body.)
pub fn takes_emphasis_dot(c: char) -> bool {
    is_cjk_script(c) && c.is_alphanumeric()
}

/// The maximal CJK-script sub-ranges of `text`, as byte ranges relative
/// to its start.
///
/// This is what splits one emphasized span into its two renderings: the
/// ranges returned take dots, everything between them stays italic.
pub fn cjk_segments(text: &str) -> Vec<std::ops::Range<usize>> {
    let mut out: Vec<std::ops::Range<usize>> = Vec::new();
    for (offset, c) in text.char_indices() {
        if !is_cjk_script(c) {
            continue;
        }
        let end = offset + c.len_utf8();
        match out.last_mut() {
            Some(last) if last.end == offset => last.end = end,
            _ => out.push(offset..end),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn han_kana_and_hangul_are_cjk_script() {
        for c in ['中', '文', 'あ', 'ア', '한', '々', 'ー'] {
            assert!(is_cjk_script(c), "{c} should be CJK script");
        }
    }

    #[test]
    fn latin_and_latin_punctuation_are_not_cjk_script() {
        for c in ['a', 'Z', '0', ',', '.', ' ', '—', 'é'] {
            assert!(!is_cjk_script(c), "{c} should not be CJK script");
        }
    }

    #[test]
    fn cjk_punctuation_splits_with_the_run_but_takes_no_dot() {
        for c in ['，', '。', '、', '「', '」', '　'] {
            assert!(is_cjk_script(c), "{c} belongs to the CJK sub-run");
            assert!(!takes_emphasis_dot(c), "{c} must not take a dot");
        }
    }

    #[test]
    fn ideographs_and_fullwidth_alphanumerics_take_dots() {
        for c in ['中', 'あ', '한', '１', 'Ａ'] {
            assert!(takes_emphasis_dot(c), "{c} should take a dot");
        }
    }

    #[test]
    fn latin_never_takes_a_dot() {
        for c in ['a', '1', '-'] {
            assert!(!takes_emphasis_dot(c));
        }
    }

    #[test]
    fn segments_split_a_mixed_run() {
        // "中文 mixed 测试" — two CJK segments around a Latin one.
        let text = "中文 mixed 测试";
        let segs = cjk_segments(text);
        assert_eq!(segs.len(), 2);
        assert_eq!(&text[segs[0].clone()], "中文");
        assert_eq!(&text[segs[1].clone()], "测试");
    }

    #[test]
    fn segments_keep_cjk_punctuation_inside_one_run() {
        let text = "中文，测试";
        let segs = cjk_segments(text);
        assert_eq!(segs.len(), 1);
        assert_eq!(&text[segs[0].clone()], text);
    }

    #[test]
    fn pure_latin_has_no_segments() {
        assert!(cjk_segments("plain latin text").is_empty());
    }
}
