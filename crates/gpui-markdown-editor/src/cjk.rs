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
    CJK_BLOCKS
        .iter()
        .any(|(lo, hi)| (*lo..=*hi).contains(&(c as u32)))
}

/// The Unicode blocks [`is_cjk_script`] answers for — **one entry per
/// block, at that block's official range**, sorted and non-overlapping.
///
/// Hand-maintained, so keep it auditable: never merge two blocks into one
/// entry and never round an edge, because a wrong edge is silent (the
/// delimiters still vanish and the reader gets neither a dot nor an
/// upright face). `block_edges_match_their_ranges` pins every edge, and
/// the named-character tests below pin the ones review has caught.
const CJK_BLOCKS: &[(u32, u32)] = &[
    (0x1100, 0x11FF),   // Hangul Jamo
    (0x2E80, 0x2EFF),   // CJK Radicals Supplement
    (0x2F00, 0x2FDF),   // Kangxi Radicals
    (0x2FF0, 0x2FFF),   // Ideographic Description Characters
    (0x3000, 0x303F),   // CJK Symbols and Punctuation
    (0x3040, 0x309F),   // Hiragana
    (0x30A0, 0x30FF),   // Katakana
    (0x3100, 0x312F),   // Bopomofo
    (0x3130, 0x318F),   // Hangul Compatibility Jamo
    (0x3190, 0x319F),   // Kanbun
    (0x31A0, 0x31BF),   // Bopomofo Extended
    (0x31C0, 0x31EF),   // CJK Strokes
    (0x31F0, 0x31FF),   // Katakana Phonetic Extensions
    (0x3200, 0x32FF),   // Enclosed CJK Letters and Months
    (0x3300, 0x33FF),   // CJK Compatibility
    (0x3400, 0x4DBF),   // CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   // CJK Unified Ideographs
    (0xA960, 0xA97F),   // Hangul Jamo Extended-A
    (0xAC00, 0xD7AF),   // Hangul Syllables
    (0xD7B0, 0xD7FF),   // Hangul Jamo Extended-B
    (0xF900, 0xFAFF),   // CJK Compatibility Ideographs
    (0xFE10, 0xFE1F),   // Vertical Forms
    (0xFE30, 0xFE4F),   // CJK Compatibility Forms
    (0xFE50, 0xFE6F),   // Small Form Variants
    (0xFF00, 0xFFEF),   // Halfwidth and Fullwidth Forms
    (0x1AFF0, 0x1AFFF), // Kana Extended-B
    (0x1B000, 0x1B0FF), // Kana Supplement
    (0x1B100, 0x1B12F), // Kana Extended-A
    (0x1B130, 0x1B16F), // Small Kana Extension
    (0x20000, 0x3FFFF), // CJK Unified Ideographs Extension B and later
];

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
    fn block_table_is_sorted_and_non_overlapping() {
        for pair in CJK_BLOCKS.windows(2) {
            assert!(
                pair[0].1 < pair[1].0,
                "blocks must be sorted and disjoint: {:04X?} then {:04X?}",
                pair[0],
                pair[1]
            );
        }
        for (lo, hi) in CJK_BLOCKS {
            assert!(lo <= hi, "block U+{lo:04X}..U+{hi:04X} is inverted");
        }
    }

    #[test]
    fn block_edges_match_their_ranges() {
        // Both edges of every block answer yes, and the codepoint just
        // outside each edge answers no unless a *different* block claims
        // it. Widening a range into its neighbour trips this.
        let covered = |cp: u32| CJK_BLOCKS.iter().any(|(lo, hi)| (*lo..=*hi).contains(&cp));
        for (lo, hi) in CJK_BLOCKS {
            for edge in [*lo, *hi] {
                let c = char::from_u32(edge).expect("block edges are scalar values");
                assert!(is_cjk_script(c), "U+{edge:04X} is inside its own block");
            }
            for outside in [lo.checked_sub(1), hi.checked_add(1)].into_iter().flatten() {
                if covered(outside) {
                    continue; // an adjacent block owns it
                }
                let Some(c) = char::from_u32(outside) else {
                    continue; // surrogate half — never a `char`
                };
                assert!(
                    !is_cjk_script(c),
                    "U+{outside:04X} sits outside every block but classified as CJK"
                );
            }
        }
    }

    #[test]
    fn hangul_jamo_extended_b_is_cjk() {
        // The Hangul Syllables *block* ends at U+D7AF (its last assigned
        // character is U+D7A3), and Hangul Jamo Extended-B starts at
        // U+D7B0 — a separate block that a single range ending at D7AF
        // silently excluded.
        for c in ['\u{D7B0}', '\u{D7CB}', '\u{D7FB}', '\u{D7FF}'] {
            assert!(
                is_cjk_script(c),
                "U+{:04X} is Hangul Jamo Extended-B",
                c as u32
            );
        }
        // The jamo are letters, so they take dots like any other CJK word
        // character.
        assert!(takes_emphasis_dot('\u{D7B0}'));
        // Both neighbours of the pair of blocks stay outside.
        assert!(
            is_cjk_script('\u{AC00}'),
            "Hangul Syllables still start at U+AC00"
        );
        assert!(!is_cjk_script('\u{ABFF}'));
    }

    #[test]
    fn kanbun_annotation_marks_are_cjk() {
        // U+3190..=U+319F sat in the gap between Hangul Compatibility
        // Jamo and Bopomofo Extended. They are annotation marks, not word
        // characters, so they join the sub-run without taking a dot —
        // the same rule CJK punctuation follows.
        for c in ['\u{3190}', '\u{319F}'] {
            assert!(is_cjk_script(c), "U+{:04X} is Kanbun", c as u32);
            assert!(!takes_emphasis_dot(c));
        }
    }

    #[test]
    fn supplementary_plane_kana_is_cjk() {
        for c in ['\u{1AFF0}', '\u{1B000}', '\u{1B132}'] {
            assert!(
                is_cjk_script(c),
                "U+{:04X} is supplementary-plane kana",
                c as u32
            );
            assert!(takes_emphasis_dot(c));
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
