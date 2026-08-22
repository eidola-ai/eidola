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
/// upright face). Merging is the same failure one level up — a range
/// spanning several blocks hides its interior boundaries from
/// `block_edges_match_their_ranges`, which can only audit the edges it
/// is told about. Named-character tests below pin the ones review has
/// caught.
///
/// **Enclosed Ideographic Supplement (`U+1F200..=U+1F2FF`) is
/// deliberately absent.** Every character in it is `General_Category=So`
/// with `Script=Common` (the single exception, U+1F200, is Hiragana), so
/// none is a character of a sentence and none could take a dot under
/// [`takes_emphasis_dot`]; they are squared/circled *symbols* that render
/// from the emoji face. This is a script classifier, and Common-script
/// symbols are not a script. Its BMP near-namesake, Enclosed CJK Letters
/// and Months, *is* listed: that block carries real Hangul- and
/// Katakana-script members set on the CJK body.
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
    (0x16FE0, 0x16FFF), // Ideographic Symbols and Punctuation
    (0x1AFF0, 0x1AFFF), // Kana Extended-B
    (0x1B000, 0x1B0FF), // Kana Supplement
    (0x1B100, 0x1B12F), // Kana Extended-A
    (0x1B130, 0x1B16F), // Small Kana Extension
    (0x20000, 0x2A6DF), // CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F), // CJK Unified Ideographs Extension C
    (0x2B740, 0x2B81F), // CJK Unified Ideographs Extension D
    (0x2B820, 0x2CEAF), // CJK Unified Ideographs Extension E
    (0x2CEB0, 0x2EBEF), // CJK Unified Ideographs Extension F
    (0x2EBF0, 0x2EE5F), // CJK Unified Ideographs Extension I
    (0x2F800, 0x2FA1F), // CJK Compatibility Ideographs Supplement
    (0x30000, 0x3134F), // CJK Unified Ideographs Extension G
    (0x31350, 0x323AF), // CJK Unified Ideographs Extension H
    (0x323B0, 0x3347F), // CJK Unified Ideographs Extension J
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

/// True for a variation selector — `U+FE00..=U+FE0F` (Variation
/// Selectors) or `U+E0100..=U+E01EF` (Variation Selectors Supplement).
///
/// **Deliberately not a [`CJK_BLOCKS`] entry**: a selector is not a
/// script, it is a modifier of whatever script precedes it. It takes the
/// classification of its base character, which is what
/// [`cjk_segments`] implements — and it never takes a dot of its own,
/// which falls out of [`takes_emphasis_dot`] unchanged, since a selector
/// is a nonspacing mark rather than an alphanumeric.
pub fn is_variation_selector(c: char) -> bool {
    matches!(c, '\u{FE00}'..='\u{FE0F}' | '\u{E0100}'..='\u{E01EF}')
}

/// The maximal CJK-script sub-ranges of `text`, as byte ranges relative
/// to its start.
///
/// This is what splits one emphasized span into its two renderings: the
/// ranges returned take dots, everything between them stays italic.
///
/// **A segment runs through the variation selectors attached to it.** An
/// ideographic variation sequence (`葛` + `U+E0100`) is one shaped
/// cluster, and gpui shapes each `TextRun` separately: leaving the
/// selector to the italic run beside the base split the cluster across
/// two runs, where the selector can no longer select the base's glyph
/// variant — the ideograph the reader sees changes, silently, which is
/// the same class of failure the dots exist to prevent. A selector with
/// no CJK character immediately before it belongs to whatever *is*
/// before it, so it stays out.
pub fn cjk_segments(text: &str) -> Vec<std::ops::Range<usize>> {
    let mut out: Vec<std::ops::Range<usize>> = Vec::new();
    for (offset, c) in text.char_indices() {
        let attached = is_variation_selector(c)
            && out
                .last()
                .is_some_and(|last: &std::ops::Range<usize>| last.end == offset);
        if !is_cjk_script(c) && !attached {
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
    fn ideographic_symbols_and_punctuation_is_cjk() {
        // U+16FE0..=U+16FFF sat in the gap between Halfwidth and
        // Fullwidth Forms and the supplementary kana blocks. Most of it
        // is `Script=Han` — the Old Chinese marks, the Vietnamese
        // alternate reading marks, the small ER forms — and the
        // iteration marks are the supplementary-plane analogue of
        // U+3005, which the table has always claimed.
        for c in ['\u{16FE0}', '\u{16FE3}', '\u{16FFF}'] {
            assert!(
                is_cjk_script(c),
                "U+{:04X} is Ideographic Symbols and Punctuation",
                c as u32
            );
        }
        // An iteration mark stands for the character it repeats, so it
        // is marked like one; the hook mark is punctuation and is not.
        assert!(
            takes_emphasis_dot('\u{16FE3}'),
            "OLD CHINESE ITERATION MARK"
        );
        assert!(!takes_emphasis_dot('\u{16FE2}'), "OLD CHINESE HOOK MARK");
    }

    #[test]
    fn enclosed_ideographic_supplement_stays_out() {
        // Excluded on the merits, not by oversight — see the table's
        // doc comment. `So` symbols with `Script=Common`, none of them
        // alphanumeric, so listing the block could never change a dot;
        // it would only pull emoji-face symbols into the CJK sub-run.
        for c in ['\u{1F210}', '\u{1F250}', '\u{1F265}'] {
            assert!(
                !is_cjk_script(c),
                "U+{:04X} is an enclosed symbol",
                c as u32
            );
        }
    }

    #[test]
    fn the_ideograph_extensions_are_listed_block_by_block() {
        // One merged `0x20000..=0x3FFFF` entry hid every boundary above
        // U+20000 from `block_edges_match_their_ranges` and swallowed
        // the unassigned gaps between the extensions.
        for (name, lo, hi) in [
            ("Extension B", 0x20000u32, 0x2A6DFu32),
            ("Extension C", 0x2A700, 0x2B73F),
            ("Extension D", 0x2B740, 0x2B81F),
            ("Extension E", 0x2B820, 0x2CEAF),
            ("Extension F", 0x2CEB0, 0x2EBEF),
            ("Extension I", 0x2EBF0, 0x2EE5F),
            ("Compatibility Ideographs Supplement", 0x2F800, 0x2FA1F),
            ("Extension G", 0x30000, 0x3134F),
            ("Extension H", 0x31350, 0x323AF),
            ("Extension J", 0x323B0, 0x3347F),
        ] {
            for edge in [lo, hi] {
                let c = char::from_u32(edge).expect("scalar value");
                assert!(is_cjk_script(c), "U+{edge:04X} is in CJK {name}");
            }
        }
        // The gaps the merged range used to claim.
        for gap in [0x2A6E0u32, 0x2EE60, 0x2FA20, 0x33480, 0x3FFFF] {
            let c = char::from_u32(gap).expect("scalar value");
            assert!(
                !is_cjk_script(c),
                "U+{gap:04X} belongs to no CJK block and must not classify"
            );
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
    fn segments_run_through_attached_variation_selectors() {
        // 葛 + VARIATION SELECTOR-17 is one ideographic variation
        // sequence; the selector must not be handed to the italic run.
        let text = "\u{845B}\u{E0100}\u{57CE}";
        let segs = cjk_segments(text);
        assert_eq!(segs.len(), 1);
        assert_eq!(&text[segs[0].clone()], text);

        // The BMP selectors attach the same way.
        let text = "\u{3297}\u{FE0F} latin";
        let segs = cjk_segments(text);
        assert_eq!(&text[segs[0].clone()], "\u{3297}\u{FE0F}");
    }

    #[test]
    fn a_variation_selector_without_a_cjk_base_stays_out() {
        // It modifies whatever precedes it, and nothing here does.
        assert!(cjk_segments("a\u{FE0F}").is_empty());
        assert!(cjk_segments("\u{E0100}").is_empty());
        // A selector separated from the CJK run by a Latin character
        // belongs to that character, not to the run.
        let text = "\u{4E2D}a\u{FE0F}";
        let segs = cjk_segments(text);
        assert_eq!(segs.len(), 1);
        assert_eq!(&text[segs[0].clone()], "\u{4E2D}");
    }

    #[test]
    fn a_variation_selector_takes_no_dot_of_its_own() {
        for c in ['\u{FE00}', '\u{FE0F}', '\u{E0100}', '\u{E01EF}'] {
            assert!(is_variation_selector(c));
            assert!(
                !takes_emphasis_dot(c),
                "U+{:04X} marks nothing of its own",
                c as u32
            );
        }
        assert!(
            !is_variation_selector('\u{FE10}'),
            "Vertical Forms starts here"
        );
    }

    #[test]
    fn pure_latin_has_no_segments() {
        assert!(cjk_segments("plain latin text").is_empty());
    }
}
