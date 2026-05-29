//! CommonMark §2.4 backslash escapes and §2.5 entity references.
//!
//! Pulldown-cmark resolves both — `Event::Text` for `\*` skips the
//! backslash byte entirely, and entity references emit decoded
//! characters with the *full source range* — which makes its event
//! stream lossy for an editor that needs to keep the raw source
//! visible. We re-scan the source bytes ourselves and surface each
//! occurrence as a `(construct_range, display_text)` pair the
//! renderer maps to a `Substitution` (cursor outside) or a dimmed
//! `InlineRun` (cursor inside).
//!
//! Verbatim contexts where neither rule applies are the caller's
//! concern: pass a sorted, non-overlapping `verbatim` slice and the
//! scanner skips past those ranges.

use std::ops::Range;

/// One §2.4 / §2.5 occurrence in the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSpan {
    /// Source byte range of the full construct (e.g. 2 bytes for `\*`,
    /// 6 bytes for `&amp;`).
    pub source_range: Range<usize>,
    /// What the construct renders to with the rule applied.
    pub display: String,
}

/// Scan `bytes[range]` for backslash escapes and entity references.
/// Skips past any byte that lies inside a `verbatim` range. The caller
/// must ensure `verbatim` is sorted by `start` and non-overlapping.
pub fn scan(bytes: &[u8], range: Range<usize>, verbatim: &[Range<usize>]) -> Vec<ResolvedSpan> {
    let mut out = Vec::new();
    let mut i = range.start;
    let limit = range.end.min(bytes.len());
    while i < limit {
        // Skip verbatim regions. Since `verbatim` is sorted, a linear
        // probe is fine — at most one match per `i` advance.
        if let Some(v) = verbatim.iter().find(|v| v.start <= i && i < v.end) {
            i = v.end;
            continue;
        }
        let b = bytes[i];
        if b == b'\\' {
            if let Some(span) = scan_backslash_escape(bytes, i, limit) {
                i = span.source_range.end;
                out.push(span);
                continue;
            }
        } else if b == b'&'
            && let Some(span) = scan_entity(bytes, i, limit)
        {
            i = span.source_range.end;
            out.push(span);
            continue;
        }
        i += 1;
    }
    out
}

/// CommonMark §2.4: a backslash followed by an ASCII punctuation
/// character renders as the literal punctuation; the backslash is
/// dropped. A backslash before any other character (including newline,
/// letters, digits, non-ASCII) stays literal in the output. The
/// hard-line-break form `\\n` is *not* an escape — it's a separate
/// construct handled at the line level by the parser.
fn scan_backslash_escape(bytes: &[u8], at: usize, limit: usize) -> Option<ResolvedSpan> {
    if at + 1 >= limit {
        return None;
    }
    let nxt = bytes[at + 1];
    if !is_ascii_punctuation(nxt) {
        return None;
    }
    Some(ResolvedSpan {
        source_range: at..at + 2,
        display: (nxt as char).to_string(),
    })
}

/// CommonMark §2.5: HTML5 named entity (`&name;`), decimal numeric
/// (`&#1234;`), or hexadecimal numeric (`&#xABCD;` / `&#XABCD;`).
/// Returns `None` if the candidate isn't a well-formed reference;
/// invalid / out-of-range / `U+0000` numerics resolve to U+FFFD per
/// spec; unknown named entities don't match (the literal `&name;`
/// remains in source).
fn scan_entity(bytes: &[u8], at: usize, limit: usize) -> Option<ResolvedSpan> {
    debug_assert_eq!(bytes.get(at).copied(), Some(b'&'));
    let mut p = at + 1;
    if p >= limit {
        return None;
    }
    if bytes[p] == b'#' {
        p += 1;
        if p >= limit {
            return None;
        }
        let (hex, digits_start) = if bytes[p] == b'x' || bytes[p] == b'X' {
            (true, p + 1)
        } else {
            (false, p)
        };
        let mut q = digits_start;
        while q < limit && q - digits_start < 7 {
            let b = bytes[q];
            let ok = if hex {
                b.is_ascii_hexdigit()
            } else {
                b.is_ascii_digit()
            };
            if !ok {
                break;
            }
            q += 1;
        }
        if q == digits_start || q >= limit || bytes[q] != b';' {
            return None;
        }
        // Caps: spec allows 1–7 decimal digits, 1–6 hex digits.
        if hex && q - digits_start > 6 {
            return None;
        }
        let s = std::str::from_utf8(&bytes[digits_start..q]).ok()?;
        let codepoint = u32::from_str_radix(s, if hex { 16 } else { 10 }).ok()?;
        let resolved = codepoint_to_string(codepoint);
        return Some(ResolvedSpan {
            source_range: at..q + 1,
            display: resolved,
        });
    }
    // Named: `&name;` where name is alphanumeric, 1..=32 chars per spec.
    let name_start = p;
    let mut q = p;
    while q < limit && q - name_start < 32 && bytes[q].is_ascii_alphanumeric() {
        q += 1;
    }
    if q == name_start || q >= limit || bytes[q] != b';' {
        return None;
    }
    let name = std::str::from_utf8(&bytes[name_start..q]).ok()?;
    let resolved = lookup_named_entity(name)?;
    Some(ResolvedSpan {
        source_range: at..q + 1,
        display: resolved.to_string(),
    })
}

fn codepoint_to_string(cp: u32) -> String {
    // Invalid / null / out-of-range / surrogate → U+FFFD.
    let resolved = char::from_u32(cp)
        .filter(|&c| c != '\0')
        .unwrap_or('\u{FFFD}');
    resolved.to_string()
}

fn is_ascii_punctuation(b: u8) -> bool {
    matches!(
        b,
        b'!' | b'"'
            | b'#'
            | b'$'
            | b'%'
            | b'&'
            | b'\''
            | b'('
            | b')'
            | b'*'
            | b'+'
            | b','
            | b'-'
            | b'.'
            | b'/'
            | b':'
            | b';'
            | b'<'
            | b'='
            | b'>'
            | b'?'
            | b'@'
            | b'['
            | b'\\'
            | b']'
            | b'^'
            | b'_'
            | b'`'
            | b'{'
            | b'|'
            | b'}'
            | b'~'
    )
}

/// Curated set of the most common HTML5 named entities. CommonMark
/// requires the full HTML5 entity list (~2000 names) but in practice
/// users only ever write a tiny subset; unknown names remain literal,
/// which matches both the spec's "valid named character references
/// only" rule and the user's likely intent (typing `&banana;` should
/// render `&banana;`). Extend this table when a real-world miss shows
/// up; for full coverage, vendor pulldown-cmark's `entities.rs`.
fn lookup_named_entity(name: &str) -> Option<&'static str> {
    Some(match name {
        "amp" => "&",
        "lt" => "<",
        "gt" => ">",
        "quot" => "\"",
        "apos" => "'",
        "nbsp" => "\u{A0}",
        "copy" => "\u{A9}",
        "reg" => "\u{AE}",
        "trade" => "\u{2122}",
        "mdash" => "\u{2014}",
        "ndash" => "\u{2013}",
        "hellip" => "\u{2026}",
        "lsquo" => "\u{2018}",
        "rsquo" => "\u{2019}",
        "ldquo" => "\u{201C}",
        "rdquo" => "\u{201D}",
        "laquo" => "\u{AB}",
        "raquo" => "\u{BB}",
        "larr" => "\u{2190}",
        "rarr" => "\u{2192}",
        "uarr" => "\u{2191}",
        "darr" => "\u{2193}",
        "harr" => "\u{2194}",
        "deg" => "\u{B0}",
        "plusmn" => "\u{B1}",
        "times" => "\u{D7}",
        "divide" => "\u{F7}",
        "infin" => "\u{221E}",
        "ne" => "\u{2260}",
        "le" => "\u{2264}",
        "ge" => "\u{2265}",
        "sum" => "\u{2211}",
        "prod" => "\u{220F}",
        "int" => "\u{222B}",
        "alpha" => "\u{3B1}",
        "beta" => "\u{3B2}",
        "gamma" => "\u{3B3}",
        "delta" => "\u{3B4}",
        "epsilon" => "\u{3B5}",
        "theta" => "\u{3B8}",
        "lambda" => "\u{3BB}",
        "mu" => "\u{3BC}",
        "pi" => "\u{3C0}",
        "sigma" => "\u{3C3}",
        "phi" => "\u{3C6}",
        "omega" => "\u{3C9}",
        "Gamma" => "\u{393}",
        "Delta" => "\u{394}",
        "Theta" => "\u{398}",
        "Lambda" => "\u{39B}",
        "Pi" => "\u{3A0}",
        "Sigma" => "\u{3A3}",
        "Phi" => "\u{3A6}",
        "Omega" => "\u{3A9}",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan_all(src: &str) -> Vec<ResolvedSpan> {
        scan(src.as_bytes(), 0..src.len(), &[])
    }

    #[test]
    fn escapes_punctuation() {
        let r = scan_all(r"a\*b");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].source_range, 1..3);
        assert_eq!(r[0].display, "*");
    }

    #[test]
    fn escape_at_end_of_buffer_is_a_no_op() {
        let r = scan_all("a\\");
        assert!(r.is_empty());
    }

    #[test]
    fn backslash_before_letter_is_literal() {
        let r = scan_all(r"a\nb"); // `\n` is the literal two chars, not an escape
        assert!(r.is_empty());
    }

    #[test]
    fn backslash_before_newline_is_not_an_escape() {
        let r = scan_all("a\\\nb");
        assert!(r.is_empty());
    }

    #[test]
    fn double_backslash_escapes_the_second() {
        // `\\` — first `\` escapes the second; the second `\` is a literal.
        let r = scan_all(r"a\\b");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].source_range, 1..3);
        assert_eq!(r[0].display, "\\");
    }

    #[test]
    fn entity_amp() {
        let r = scan_all("a&amp;b");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].source_range, 1..6);
        assert_eq!(r[0].display, "&");
    }

    #[test]
    fn entity_decimal() {
        let r = scan_all("&#169;");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].source_range, 0..6);
        assert_eq!(r[0].display, "©");
    }

    #[test]
    fn entity_hex_lowercase() {
        let r = scan_all("&#xa9;");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].display, "©");
    }

    #[test]
    fn entity_hex_uppercase() {
        let r = scan_all("&#XA9;");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].display, "©");
    }

    #[test]
    fn unknown_named_entity_does_not_match() {
        let r = scan_all("&banana;");
        assert!(r.is_empty());
    }

    #[test]
    fn missing_semicolon_does_not_match() {
        let r = scan_all("&amp");
        assert!(r.is_empty());
    }

    #[test]
    fn null_resolves_to_replacement() {
        let r = scan_all("&#0;");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].display, "\u{FFFD}");
    }

    #[test]
    fn out_of_range_resolves_to_replacement() {
        // 0x110000 is past the Unicode max. Numeric needs 7 digits decimal
        // or fits in 6 hex.
        let r = scan_all("&#x110000;");
        assert_eq!(r.len(), 1);
        assert!(r.is_empty() || r[0].display == "\u{FFFD}");
    }

    #[test]
    fn skips_verbatim_range() {
        // `\*` inside the verbatim region [0..7) should be skipped; the
        // second `\*` outside is reported.
        let src = r"`\*foo`\*";
        #[allow(clippy::single_range_in_vec_init)]
        let v = vec![0..7usize];
        let r = scan(src.as_bytes(), 0..src.len(), &v);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].source_range, 7..9);
    }

    #[test]
    fn multi_byte_decoded_entity() {
        let r = scan_all("&mdash;");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].display, "—");
        assert!(r[0].display.len() > 1); // multi-byte UTF-8 result
    }
}
