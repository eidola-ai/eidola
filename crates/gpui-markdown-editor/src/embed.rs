//! Embed blocks — the editor's opaque block-plugin mechanism.
//!
//! The host supplies a map of **ordinals → markdown content** ([`EmbedMap`]).
//! A paragraph whose entire content is the marker `{{ embed N }}` — and whose
//! ordinal `N` is present in the map — renders as a single **atomic** block
//! element: the mapped markdown shown read-only inside a quiet quote-like
//! container. The editor never learns what the content *means* (a quote, a
//! transclusion, …) — that is the host's business; the ordinal is the shared
//! key between the buffer text, the map, and the host's own bookkeeping.
//!
//! # Lexical rules
//!
//! A marker is recognized when a **top-level paragraph's entire content**
//! (leading/trailing spaces and tabs tolerated) matches:
//!
//! ```text
//! "{{"  WS*  "embed"  WS+  DIGITS  WS*  "}}"
//! ```
//!
//! where `WS` is a space or tab and `DIGITS` is a non-negative decimal
//! integer parsed as `u64` (leading zeros allowed: `007` = 7). The canonical
//! spelling — what hosts should write into the buffer — is `{{ embed N }}`.
//!
//! Everything else is plain text, which doubles as the escaping story:
//!
//! * an **unmapped** ordinal (`{{ embed 9 }}` with no entry 9) renders as
//!   literal text — honest degradation, and how a marker survives before its
//!   reference exists;
//! * an **inline** occurrence (`see {{ embed 1 }} here`) is literal — only a
//!   sole-paragraph marker is a block;
//! * a marker inside a blockquote / list / code fence is literal (v1 embeds
//!   are top-level only, like tables);
//! * to type the literal text of a *mapped* marker, break the pattern —
//!   CommonMark's backslash escape on the opener (`\{{ embed 1 }}`) renders
//!   as `{{ embed 1 }}` via the escape substitution pass and never matches
//!   (the matcher reads raw source bytes, which start with `\`).
//!
//! **Round-trip:** the buffer always contains the plain marker text — the map
//! is render-time state only, and `value()` / copy / persistence see clean
//! markdown. Deleting the block (backspace/delete at its boundary, or a
//! selection over it) deletes the marker string; re-typing a mapped marker
//! re-materializes the block ("re-embed by typing").
//!
//! # Atomicity
//!
//! Positions strictly inside a mapped marker are **forbidden caret
//! positions** (the hidden-chrome machinery tables and list indents use):
//! arrows hop over the block in one step, clicks snap to its edges, a
//! selection endpoint never rests inside, and backspace/delete at the
//! trailing/leading edge removes the whole marker in one step.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

/// The host-supplied map of embed ordinals → markdown content. Cheap to
/// clone (shared `Arc`); an empty map (the default) disables the plugin.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmbedMap(Option<Arc<BTreeMap<u64, String>>>);

impl EmbedMap {
    /// Build a map from `(ordinal, markdown)` pairs.
    pub fn new(entries: impl IntoIterator<Item = (u64, String)>) -> Self {
        let map: BTreeMap<u64, String> = entries.into_iter().collect();
        if map.is_empty() {
            Self(None)
        } else {
            Self(Some(Arc::new(map)))
        }
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    pub fn contains(&self, ordinal: u64) -> bool {
        self.0.as_ref().is_some_and(|m| m.contains_key(&ordinal))
    }

    pub fn get(&self, ordinal: u64) -> Option<&str> {
        self.0.as_ref()?.get(&ordinal).map(String::as_str)
    }
}

/// The canonical marker text for an ordinal — what hosts insert into the
/// buffer: `{{ embed N }}`.
pub fn embed_marker(ordinal: u64) -> String {
    format!("{{{{ embed {ordinal} }}}}")
}

/// Parse a candidate marker: the **entire** input (modulo leading/trailing
/// spaces/tabs) must match the lexical rule in the module docs. Returns the
/// ordinal. This rule is duplicated (deliberately, with lockstep tests) in
/// `eidola-app-core`'s `parse_embed_marker` — app-core expands markers into
/// quoted text for upstream models and cannot depend on this gpui crate.
pub fn parse_embed_text(s: &str) -> Option<u64> {
    const WS: [char; 2] = [' ', '\t'];
    let s = s.trim_matches(WS);
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

/// One recognized embed block: the marker's byte range in the buffer (the
/// paragraph's content extent, no trailing newline) and its mapped ordinal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedBlock {
    pub range: Range<usize>,
    pub ordinal: u64,
}

/// Scan the document for mapped embed blocks: **top-level** paragraphs whose
/// entire content parses as a marker with an ordinal present in `embeds`.
/// Parser-driven (a paragraph is whatever pulldown says is a paragraph), with
/// a cheap `{{` prefilter for the common no-embed case.
pub fn embed_blocks(markdown: &str, embeds: &EmbedMap) -> Vec<EmbedBlock> {
    if embeds.is_empty() || !markdown.contains("{{") {
        return Vec::new();
    }
    let tree = crate::parser::parse(markdown);
    let mut out = Vec::new();
    for node in &tree {
        if !matches!(node.kind, crate::syntax::NodeKind::Paragraph) {
            continue;
        }
        // Trim the trailing newline pulldown folds into a paragraph's range so
        // the block range is exactly the marker text.
        let mut range = node.range.clone();
        while range.end > range.start && markdown.as_bytes()[range.end - 1] == b'\n' {
            range.end -= 1;
        }
        let Some(slice) = markdown.get(range.clone()) else {
            continue;
        };
        // The raw source must start with `{{` after space/tab trim — an
        // escaped opener (`\{{ …`) therefore never matches even though it
        // *renders* as the literal braces.
        if let Some(ordinal) = parse_embed_text(slice)
            && embeds.contains(ordinal)
        {
            out.push(EmbedBlock { range, ordinal });
        }
    }
    out
}

/// The mapped embed block whose range strictly contains `p`, if any —
/// `start < p < end`. Boundary positions are ordinary caret positions.
pub fn embed_interior_at(markdown: &str, embeds: &EmbedMap, p: usize) -> Option<EmbedBlock> {
    embed_blocks(markdown, embeds)
        .into_iter()
        .find(|b| p > b.range.start && p < b.range.end)
}

/// The mapped embed block ending exactly at `p` (backspace-as-unit), if any.
pub fn embed_ending_at(markdown: &str, embeds: &EmbedMap, p: usize) -> Option<EmbedBlock> {
    embed_blocks(markdown, embeds)
        .into_iter()
        .find(|b| b.range.end == p)
}

/// The mapped embed block starting exactly at `p` (delete-forward-as-unit),
/// if any.
pub fn embed_starting_at(markdown: &str, embeds: &EmbedMap, p: usize) -> Option<EmbedBlock> {
    embed_blocks(markdown, embeds)
        .into_iter()
        .find(|b| b.range.start == p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(ordinals: &[u64]) -> EmbedMap {
        EmbedMap::new(ordinals.iter().map(|&n| (n, format!("content {n}"))))
    }

    /// The lexical rule — kept in lockstep with app-core's
    /// `parse_embed_marker` (same cases pinned there).
    #[test]
    fn parse_embed_text_lexical_rules() {
        assert_eq!(parse_embed_text("{{ embed 0 }}"), Some(0));
        assert_eq!(parse_embed_text("{{embed 12}}"), Some(12));
        assert_eq!(parse_embed_text("{{\tembed\t7\t}}"), Some(7));
        assert_eq!(parse_embed_text("  {{ embed 3 }}  "), Some(3));
        assert_eq!(parse_embed_text("{{ embed 007 }}"), Some(7));
        assert_eq!(parse_embed_text("{{ embed0 }}"), None);
        assert_eq!(parse_embed_text("{{ embed -1 }}"), None);
        assert_eq!(parse_embed_text("{{ embed +1 }}"), None);
        assert_eq!(parse_embed_text("{{ embed 0x1 }}"), None);
        assert_eq!(parse_embed_text("{{ embed }}"), None);
        assert_eq!(parse_embed_text("{ embed 0 }"), None);
        assert_eq!(parse_embed_text("{{ embed 0 }} tail"), None);
        assert_eq!(parse_embed_text("x {{ embed 0 }}"), None);
        assert_eq!(parse_embed_text("\\{{ embed 0 }}"), None);
    }

    #[test]
    fn canonical_marker_round_trips() {
        assert_eq!(embed_marker(4), "{{ embed 4 }}");
        assert_eq!(parse_embed_text(&embed_marker(4)), Some(4));
    }

    #[test]
    fn embed_blocks_finds_sole_paragraph_mapped_markers() {
        let src = "before\n\n{{ embed 1 }}\n\nafter";
        let blocks = embed_blocks(src, &map(&[1]));
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].ordinal, 1);
        assert_eq!(&src[blocks[0].range.clone()], "{{ embed 1 }}");
    }

    #[test]
    fn unmapped_inline_nested_and_escaped_markers_are_plain_text() {
        let m = map(&[1]);
        // Unmapped ordinal.
        assert!(embed_blocks("{{ embed 9 }}", &m).is_empty());
        // Inline occurrence (not the sole paragraph content).
        assert!(embed_blocks("see {{ embed 1 }} here", &m).is_empty());
        // Inside a blockquote / list (v1: top-level only).
        assert!(embed_blocks("> {{ embed 1 }}", &m).is_empty());
        assert!(embed_blocks("- {{ embed 1 }}", &m).is_empty());
        // Inside a fenced code block.
        assert!(embed_blocks("```\n{{ embed 1 }}\n```", &m).is_empty());
        // Escaped opener — renders as literal braces, never a block.
        assert!(embed_blocks("\\{{ embed 1 }}", &m).is_empty());
        // Empty map disables everything.
        assert!(embed_blocks("{{ embed 1 }}", &EmbedMap::default()).is_empty());
    }

    #[test]
    fn interior_and_edge_queries() {
        let src = "a\n\n{{ embed 1 }}\n\nb";
        let m = map(&[1]);
        let block = &embed_blocks(src, &m)[0];
        let (start, end) = (block.range.start, block.range.end);
        assert!(embed_interior_at(src, &m, start).is_none());
        assert!(embed_interior_at(src, &m, start + 1).is_some());
        assert!(embed_interior_at(src, &m, end - 1).is_some());
        assert!(embed_interior_at(src, &m, end).is_none());
        assert_eq!(embed_ending_at(src, &m, end).as_ref(), Some(block));
        assert_eq!(embed_starting_at(src, &m, start).as_ref(), Some(block));
        assert!(embed_ending_at(src, &m, start).is_none());
        assert!(embed_starting_at(src, &m, end).is_none());
    }
}
