//! Embed-marker recognition — the shared structural contract between the
//! `gpui-markdown-editor` embed-block plugin and `eidola-app-core`'s
//! upstream quote expansion.
//!
//! A post body refers to a quoted reference by an `{{ embed N }}` marker (N =
//! the `action_antecedent` ordinal). The editor renders a marker as an atomic
//! quote block only when it is a **sole top-level paragraph**; everywhere
//! else — inline, inside a blockquote/list, inside fenced or indented code,
//! escaped — the marker is deliberately literal text. The upstream expansion
//! must make the *same* decision, or the UI and the wire disagree: a marker
//! the author visibly "defused" by fencing it would silently expand and leak
//! the referenced passage into the request. This module is that decision,
//! kept in a zero-dependency crate both sides can consume.
//!
//! Two layers:
//!
//! * [`parse_embed_marker`] — the **lexical** rule for one candidate line:
//!   `"{{" WS* "embed" WS+ DIGITS WS* "}}"` (WS = space/tab; DIGITS = a
//!   non-negative decimal `u64`, leading zeros allowed). Canonical spelling
//!   `{{ embed N }}`.
//! * [`embed_marker_spans`] — the **structural** rule over a whole document:
//!   a marker line counts only when it stands as its own top-level
//!   paragraph. Implemented as a line scanner (this crate has no markdown
//!   parser): blank-line-delimited, any space/tab indent (the editor's
//!   parser has indented code blocks disabled, so an indented line is an
//!   ordinary paragraph — the corpus lockstep test pinned this), no
//!   container prefix (a `>`/list marker fails the lexical rule anyway),
//!   and not inside a fenced code block (``` / ~~~) or a block-level
//!   `$$ … $$` math region — the two constructs whose *interiors* can
//!   contain blank-line-delimited marker lines that are still literal
//!   content.
//!
//! The editor cannot depend on this crate (it stays free of Eidola-specific
//! symbols so other gpui apps can embed it), so it carries its own
//! parser-driven recognition (`gpui_markdown_editor::embed`). The two are
//! held in lockstep by (a) identical lexical test cases pinned in all three
//! crates and (b) a **corpus lockstep test** in `crates/eidola-gui`
//! (`tests/embed_lockstep.rs`, the one crate that sees both sides) asserting
//! this scanner and the editor's parser-driven `embed_blocks` recognize the
//! exact same marker spans across a corpus of tricky documents. Change one
//! side, run that test.

/// One recognized embed-marker paragraph: the byte range of the marker
/// **line** (leading indent through the closing `}}`, no trailing newline)
/// and its parsed ordinal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedMarkerSpan {
    pub start: usize,
    pub end: usize,
    pub ordinal: u64,
}

/// Parse a candidate marker line: the **entire** input, modulo leading and
/// trailing spaces/tabs, must match `"{{" WS* "embed" WS+ DIGITS WS* "}}"`.
/// Returns the ordinal. Lexical only — callers that need the structural
/// (own-top-level-paragraph) rule use [`embed_marker_spans`].
pub fn parse_embed_marker(line: &str) -> Option<u64> {
    const WS: [char; 2] = [' ', '\t'];
    let s = line.trim_matches(WS);
    let inner = s.strip_prefix("{{")?.strip_suffix("}}")?;
    let inner = inner.trim_matches(WS);
    let digits = inner.strip_prefix("embed")?;
    if !digits.starts_with(WS) {
        return None;
    }
    let digits = digits.trim_matches(WS);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u64>().ok()
}

/// True when `line` is blank (empty or spaces/tabs only).
fn is_blank(line: &str) -> bool {
    line.bytes().all(|b| b == b' ' || b == b'\t')
}

/// Leading-space count, treating a tab as "too much" (a leading tab opens
/// indented code in CommonMark's 4-column model).
fn leading_spaces(line: &str) -> Option<usize> {
    let mut n = 0;
    for b in line.bytes() {
        match b {
            b' ' => n += 1,
            b'\t' => return None,
            _ => break,
        }
    }
    Some(n)
}

/// Fence-region state for the scanner: inside a ``` / ~~~ fenced code block
/// or a block-level `$$ … $$` math region.
enum Region {
    None,
    /// Fenced code: the fence char and the opener's run length (the closer
    /// needs the same char, at least that many, nothing else on the line).
    Fence(u8, usize),
    /// Block-level display math (`$$` opener line; closed by a `$$`-only
    /// line or EOF).
    Math,
}

/// Try to read a fence opener (` ``` `/`~~~`, up to 3 leading spaces) from a
/// line; returns the fence char and run length.
fn fence_opener(line: &str) -> Option<(u8, usize)> {
    let indent = leading_spaces(line)?;
    if indent > 3 {
        return None;
    }
    let rest = &line.as_bytes()[indent..];
    let c = *rest.first()?;
    if c != b'`' && c != b'~' {
        return None;
    }
    let run = rest.iter().take_while(|&&b| b == c).count();
    if run < 3 {
        return None;
    }
    // A backtick fence's info string may not contain backticks.
    if c == b'`' && rest[run..].contains(&b'`') {
        return None;
    }
    Some((c, run))
}

/// Is `line` a closer for a fence opened with `(c, len)`?
fn is_fence_closer(line: &str, c: u8, len: usize) -> bool {
    let Some(indent) = leading_spaces(line) else {
        return false;
    };
    if indent > 3 {
        return false;
    }
    let rest = &line.as_bytes()[indent..];
    let run = rest.iter().take_while(|&&b| b == c).count();
    run >= len && rest[run..].iter().all(|&b| b == b' ' || b == b'\t')
}

/// Scan a document for **structurally recognized** embed markers: lines that
/// pass [`parse_embed_marker`], stand blank-line-delimited as their own
/// top-level paragraph, and sit outside any fenced-code or block-math
/// region. Returns spans in document order.
pub fn embed_marker_spans(text: &str) -> Vec<EmbedMarkerSpan> {
    if !text.contains("{{") {
        return Vec::new();
    }
    // Collect line ranges (without trailing newlines).
    let mut lines: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for (i, b) in text.bytes().enumerate() {
        if b == b'\n' {
            lines.push((start, i));
            start = i + 1;
        }
    }
    lines.push((start, text.len()));

    let mut out = Vec::new();
    let mut region = Region::None;
    for (idx, &(ls, le)) in lines.iter().enumerate() {
        let line = &text[ls..le];
        match region {
            Region::Fence(c, len) => {
                if is_fence_closer(line, c, len) {
                    region = Region::None;
                }
                continue;
            }
            Region::Math => {
                if line.trim_matches([' ', '\t']) == "$$" {
                    region = Region::None;
                }
                continue;
            }
            Region::None => {}
        }
        if let Some((c, len)) = fence_opener(line) {
            region = Region::Fence(c, len);
            continue;
        }
        // Block-level display math opener: a `$$`-only line (the canonical
        // `$$\n…\n$$` shape; a one-line `$$x$$` is a self-contained
        // construct, not a region opener).
        if line.trim_matches([' ', '\t']) == "$$" {
            region = Region::Math;
            continue;
        }
        // Candidate marker line: lexical match (any space/tab indent — the
        // editor's parser has indented code disabled, so indentation does
        // not defuse a marker) and blank-line/document boundaries on both
        // sides.
        let Some(ordinal) = parse_embed_marker(line) else {
            continue;
        };
        let prev_ok = idx == 0 || is_blank(&text[lines[idx - 1].0..lines[idx - 1].1]);
        let next_ok = idx + 1 == lines.len() || is_blank(&text[lines[idx + 1].0..lines[idx + 1].1]);
        if prev_ok && next_ok {
            out.push(EmbedMarkerSpan {
                start: ls,
                end: le,
                ordinal,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ordinals(text: &str) -> Vec<u64> {
        embed_marker_spans(text)
            .into_iter()
            .map(|s| s.ordinal)
            .collect()
    }

    /// The lexical rule — the same cases pinned in
    /// `gpui-markdown-editor::embed` and consumed via this crate by
    /// app-core. Change one, change all.
    #[test]
    fn lexical_rules() {
        assert_eq!(parse_embed_marker("{{ embed 0 }}"), Some(0));
        assert_eq!(parse_embed_marker("{{embed 12}}"), Some(12));
        assert_eq!(parse_embed_marker("{{\tembed\t7\t}}"), Some(7));
        assert_eq!(parse_embed_marker("  {{ embed 3 }}  "), Some(3));
        assert_eq!(parse_embed_marker("{{ embed 007 }}"), Some(7));
        assert_eq!(parse_embed_marker("{{ embed0 }}"), None);
        assert_eq!(parse_embed_marker("{{ embed -1 }}"), None);
        assert_eq!(parse_embed_marker("{{ embed +1 }}"), None);
        assert_eq!(parse_embed_marker("{{ embed 0x1 }}"), None);
        assert_eq!(parse_embed_marker("{{ embed }}"), None);
        assert_eq!(parse_embed_marker("{ embed 0 }"), None);
        assert_eq!(parse_embed_marker("{{ embed 0 }} tail"), None);
        assert_eq!(parse_embed_marker("x {{ embed 0 }}"), None);
        assert_eq!(parse_embed_marker("\\{{ embed 0 }}"), None);
    }

    #[test]
    fn sole_paragraph_markers_recognized() {
        assert_eq!(ordinals("{{ embed 1 }}"), vec![1]);
        assert_eq!(ordinals("a\n\n{{ embed 1 }}\n\nb"), vec![1]);
        assert_eq!(ordinals("{{ embed 1 }}\n\n{{ embed 2 }}"), vec![1, 2]);
        // Any indent is still a paragraph: the editor's parser has indented
        // code blocks disabled, so indentation does not defuse a marker
        // (pinned by the eidola-gui corpus lockstep test).
        assert_eq!(ordinals("a\n\n   {{ embed 1 }}"), vec![1]);
        assert_eq!(ordinals("a\n\n    {{ embed 1 }}"), vec![1]);
        assert_eq!(ordinals("a\n\n\t{{ embed 1 }}"), vec![1]);
    }

    #[test]
    fn non_paragraph_contexts_are_literal() {
        // Inline / same-paragraph adjacency (soft break, no blank line).
        assert_eq!(ordinals("see {{ embed 1 }} here"), Vec::<u64>::new());
        assert_eq!(ordinals("text\n{{ embed 1 }}"), Vec::<u64>::new());
        assert_eq!(ordinals("{{ embed 1 }}\ntext"), Vec::<u64>::new());
        // Container prefixes fail the lexical rule.
        assert_eq!(ordinals("> {{ embed 1 }}"), Vec::<u64>::new());
        assert_eq!(ordinals("- {{ embed 1 }}"), Vec::<u64>::new());
        // Escaped opener.
        assert_eq!(ordinals("\\{{ embed 1 }}"), Vec::<u64>::new());
        // Setext-heading shape (next line non-blank).
        assert_eq!(ordinals("{{ embed 1 }}\n===="), Vec::<u64>::new());
    }

    #[test]
    fn fenced_and_math_regions_are_literal() {
        // The load-bearing case: blank-line-delimited marker INSIDE a fence.
        assert_eq!(ordinals("```\n\n{{ embed 1 }}\n\n```"), Vec::<u64>::new());
        assert_eq!(ordinals("~~~\n\n{{ embed 1 }}\n\n~~~"), Vec::<u64>::new());
        // Unterminated fence swallows the rest of the document.
        assert_eq!(ordinals("```\n\n{{ embed 1 }}"), Vec::<u64>::new());
        // Longer closer required: a shorter run doesn't close.
        assert_eq!(
            ordinals("````\n```\n\n{{ embed 1 }}\n\n````"),
            Vec::<u64>::new()
        );
        // Block math region.
        assert_eq!(ordinals("$$\n\n{{ embed 1 }}\n\n$$"), Vec::<u64>::new());
        // After a closed fence, recognition resumes.
        assert_eq!(ordinals("```\ncode\n```\n\n{{ embed 1 }}"), vec![1]);
    }

    #[test]
    fn spans_cover_the_marker_line() {
        let text = "a\n\n  {{ embed 4 }}\n\nb";
        let spans = embed_marker_spans(text);
        assert_eq!(spans.len(), 1);
        assert_eq!(&text[spans[0].start..spans[0].end], "  {{ embed 4 }}");
        assert_eq!(spans[0].ordinal, 4);
    }
}
