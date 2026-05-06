//! Pure transformation: `(EditorState, &[SyntaxNode]) -> RenderSpec`.
//!
//! Centralizes the cursor-aware delimiter visibility rule from
//! `apps/macos/Packages/MarkdownEditor/AGENTS.md`:
//!
//! > Delimiters hide when the cursor is outside the construct. Delimiters
//! > reveal (dimmed) when the cursor — or an active selection — enters the
//! > construct.
//!
//! Every construct with explicit delimiters routes through
//! `apply_delimiter_visibility`.
//!
//! # Empty-paragraph injection
//!
//! pulldown-cmark collapses any number of blank lines between blocks into
//! a single paragraph break, so a source like `"a\n\n\n\nb"` parses to
//! exactly two paragraphs with no record of the extra newlines. Our
//! editor's invariant (see `update::enforce_invariants`) keeps every
//! user-typed `\n` in the buffer, but if the renderer just walked the
//! parser's output the extras would be invisible — pressing Enter many
//! times would change source but not visuals.
//!
//! After collecting the parser's blocks, `inject_empty_paragraphs` walks
//! every free-zone of newlines (leading, between blocks, trailing) and
//! emits one synthetic empty `Paragraph` block per "extra" newline so the
//! cursor has visible empty rows to land on.
//!
//! Formulas (pairs model — see `update.rs` for the rationale): every
//! paragraph-structural unit in source is `\n\n` (a pair). Each empty
//! paragraph is one such pair. Inter-block gaps include one extra pair
//! for the `\n\n` that *is* the paragraph break separating two real
//! paragraphs.
//!
//! With each block's `source_range` trimmed to its content extent
//! (pulldown-cmark folds one trailing `\n` into a paragraph's range —
//! we strip it so the folded `\n` lives in the trailing free zone with
//! the rest of the user-visible structure):
//!
//! - Leading run of `L` newlines: `L / 2` empty paragraphs (each pair
//!   is one row above the first real block).
//! - Inter-block run of `M` newlines: `max(0, (M − 2) / 2)` empties
//!   (one pair is the paragraph break separator; the rest are empties
//!   between the two real paragraphs).
//! - Trailing run of `T` newlines: `T / 2` empty paragraphs.
//!
//! Each synthetic block's `source_range` spans 2 bytes (one `\n` pair),
//! so a cursor anywhere in the pair hit-tests into it and clicks in
//! the empty row land at a real source position.

use std::ops::Range;

use crate::render_spec::{
    BlockKind, Container, InlineRun, InlineStyle, ListItemKind, MarkerOverlay, RenderBlock,
    RenderSpec,
};
use crate::state::{EditorState, Selection};
use crate::syntax::{ListKind, NodeKind, SyntaxNode};

pub fn render(state: &EditorState, tree: &[SyntaxNode]) -> RenderSpec {
    let cursor = CursorRange::from(&state.selection);
    let mut real_blocks = Vec::new();
    for node in tree {
        render_node(node, &state.markdown, cursor, &[], &mut real_blocks);
    }
    let blocks = inject_empty_paragraphs(&state.markdown, real_blocks);
    RenderSpec { blocks }
}

#[derive(Debug, Clone, Copy)]
struct CursorRange {
    start: usize,
    end: usize,
}

impl CursorRange {
    fn from(sel: &Selection) -> Self {
        Self {
            start: sel.lower_bound(),
            end: sel.upper_bound(),
        }
    }

    /// Cursor or selection range overlaps the construct's bounds. The
    /// boundary-equality clause keeps a collapsed cursor sitting on the edge
    /// of the construct treated as "inside" — same rule as the Swift editor.
    fn overlaps(self, range: &Range<usize>) -> bool {
        if self.start < range.end && self.end > range.start {
            return true;
        }
        if self.start == self.end && (self.start == range.start || self.start == range.end) {
            return true;
        }
        false
    }
}

fn render_node(
    node: &SyntaxNode,
    source: &str,
    cursor: CursorRange,
    containers: &[Container],
    out: &mut Vec<RenderBlock>,
) {
    match &node.kind {
        NodeKind::Paragraph => render_paragraph(node, cursor, containers, out),
        NodeKind::Heading { .. } => render_heading(node, cursor, containers, out),
        NodeKind::CodeBlock { .. } => render_code_block(node, cursor, containers, out),
        NodeKind::BlockQuote { prefix_ranges } => {
            render_blockquote(node, prefix_ranges, source, cursor, containers, out)
        }
        NodeKind::List { kind } => render_list(node, *kind, source, cursor, containers, out),
        // Anything else at top level — nothing to do yet.
        _ => {}
    }
}

fn render_paragraph(
    node: &SyntaxNode,
    cursor: CursorRange,
    containers: &[Container],
    out: &mut Vec<RenderBlock>,
) {
    let mut block = RenderBlock::new(node.range.clone(), BlockKind::Paragraph);
    block.containers = containers.to_vec();
    collect_inlines(node, cursor, &mut block);
    out.push(block);
}

/// Render a blockquote container.
///
/// We don't emit a `RenderBlock` for the blockquote itself — every leaf
/// block inside (paragraph, heading, code block, or a nested
/// blockquote's leaves) carries a `Container::BlockQuote` entry on its
/// `containers` chain. The element layer reads that chain to apply
/// cumulative left-indent and paint per-level borders.
///
/// **Pair-aware marker distribution.** The `\n[prefix]\n[prefix]`
/// structural pair (the depth-D analog of `\n\n`) collapses to one
/// paragraph_gap visually — the marker line in the middle is *not* a
/// separate row. So among the per-line markers this blockquote
/// reports, we distinguish three cases:
///
/// 1. **Content-line markers** (claimed by a parsed leaf): attach to
///    that leaf as hidden / dimmed.
/// 2. **First-prefix-of-pair markers** (the marker-only line in the
///    middle of `\n[prefix]\n[prefix]`): skip entirely — the line
///    has no rendered row.
/// 3. **Second-prefix-of-pair markers** that aren't claimed by any
///    parsed leaf (the trailing-empty case post-Enter, or an extra
///    empty between two paragraphs): emit a synthetic empty
///    Paragraph leaf so the cursor can land on that empty row.
///
/// The toggle bit `expect_middle` distinguishes (2) from (3) inside
/// the same loop: it starts true (the first unclaimed marker after
/// content, or after the BQ's start, is the *middle* of a pair) and
/// flips on each unclaimed marker. A claimed marker resets it back
/// to `true` (a parsed paragraph effectively occupies the same role
/// the second-of-pair would).
fn render_blockquote(
    node: &SyntaxNode,
    prefix_ranges: &[Range<usize>],
    source: &str,
    cursor: CursorRange,
    containers: &[Container],
    out: &mut Vec<RenderBlock>,
) {
    let cursor_inside = cursor.overlaps(&node.range);
    // The level of *this* blockquote's prefix markers is the number
    // of containers wrapping it (outer blockquotes / list items, etc.)
    // — the element layer uses it to look up the matching border bar
    // when painting overlay markers.
    let level = containers.len();
    let mut child_chain = containers.to_vec();
    child_chain.push(Container::BlockQuote { cursor_inside });

    let start = out.len();
    for child in &node.children {
        render_node(child, source, cursor, &child_chain, out);
    }

    let bytes = source.as_bytes();
    // `deferred` holds the most recent unclaimed prefix that we
    // tentatively assumed is the *middle* of a `[prefix]\n[prefix]`
    // structural pair (no row of its own). It either gets dropped
    // when its partner comes in — the partner takes the synthetic —
    // or, if no partner ever shows up, we promote the deferred
    // prefix itself to a synthetic at end-of-loop. The promotion
    // keeps a freshly-typed lone `> ` parsing-as-blockquote
    // *rendering* as a blockquote immediately, instead of waiting
    // for the user to type a paired marker.
    let mut deferred: Option<Range<usize>> = None;
    for prefix in prefix_ranges {
        if let Some(leaf) = find_leaf_for_prefix(&mut out[start..], prefix) {
            // Pulldown ranges most leaves to start *after* the line's
            // marker; extend so the marker falls inside the leaf and
            // the element layer can hide / overlay it.
            if prefix.start < leaf.source_range.start {
                leaf.source_range.start = prefix.start;
            }
            attach_marker(leaf, prefix, cursor_inside, level);
            deferred = None;
        } else if is_after_hard_break(bytes, prefix.start)
            && let Some(leaf) = find_leaf_ending_at(&mut out[start..], prefix.start)
        {
            // Hard-break continuation: the prefix sits at the start
            // of a line that follows a `  \n` / `\\\n` hard break.
            // It's a continuation of the previous paragraph (same
            // visual paragraph, new visual line), not the middle of
            // a structural pair. Pulldown often excludes the
            // dangling marker line from the paragraph's range when
            // there's no content after it yet (the post-Shift+Enter
            // transient), so we extend the leaf forward to swallow
            // it. `deferred` isn't touched — this is content, not a
            // pair half.
            leaf.source_range.end = source_line_end(bytes, prefix.end);
            attach_marker(leaf, prefix, cursor_inside, level);
        } else if deferred.is_none() {
            // First unclaimed prefix — defer. Treated as middle of a
            // pair *if* a partner shows up below; otherwise promoted
            // to a synthetic by the post-loop fixup.
            deferred = Some(prefix.clone());
        } else {
            // Partner for a deferred middle — the *current* prefix
            // is the second-of-pair and gets a synthetic empty leaf
            // so the cursor has a visible row to land on (post-
            // Enter trailing, or an extra empty between two
            // paragraphs). The synthetic shares this BQ's container
            // chain, so an outer blockquote's later distribution
            // attaches *its* marker here too.
            let line_end = source_line_end(bytes, prefix.end);
            let mut synth = RenderBlock::new(prefix.start..line_end, BlockKind::Paragraph);
            synth.containers = child_chain.clone();
            attach_marker(&mut synth, prefix, cursor_inside, level);
            out.push(synth);
            deferred = None;
        }
    }
    // Unpaired trailing prefix: a parsed-as-blockquote line with no
    // partner above (no real content claimed it) and no partner
    // below (no following marker). Without this fixup, a lone `> `
    // at end-of-doc would render as a plain paragraph until the
    // user typed a second marker — visually contradicting the
    // already-blockquote parse.
    if let Some(prefix) = deferred {
        let line_end = source_line_end(bytes, prefix.end);
        let mut synth = RenderBlock::new(prefix.start..line_end, BlockKind::Paragraph);
        synth.containers = child_chain.clone();
        attach_marker(&mut synth, &prefix, cursor_inside, level);
        out.push(synth);
    }

    // Synthetics are appended in `prefix_ranges` order (source order)
    // but may now sit *after* a parsed leaf that occurs later in
    // source — sort so subsequent passes (outer-blockquote
    // distribution, `inject_empty_paragraphs`, the editor's per-block
    // index) see blocks in source order.
    out[start..].sort_by_key(|b| b.source_range.start);
}

/// Render a list container.
///
/// Lists themselves contribute no chrome — only their items do — but
/// they're the place we can see all sibling items and assign each its
/// number / bullet info. We walk every direct `ListItem` child and
/// emit one paragraph leaf per item, carrying a `Container::ListItem`
/// in its chain.
///
/// Scope (MVP):
///   - Tight single-paragraph items.
///   - Loose / multi-paragraph items render their content but don't
///     yet preserve the inter-paragraph empty rows specifically for
///     lists — `inject_empty_paragraphs` handles those generically
///     downstream.
///   - Nested lists (a `List` child inside an `Item`) are not yet
///     wired; they parse but don't render specially. The outer
///     item's leaf still appears with its marker.
fn render_list(
    node: &SyntaxNode,
    kind: ListKind,
    source: &str,
    cursor: CursorRange,
    containers: &[Container],
    out: &mut Vec<RenderBlock>,
) {
    // For ordered lists pulldown gives us only the *list's* start
    // number; we increment per item to derive each item's own number.
    let mut next_number = match kind {
        ListKind::Ordered { start } => Some(start),
        ListKind::Unordered => None,
    };
    // Compute the widest marker text in this list once. Every sibling
    // item carries the same string on its `Container::ListItem` chain
    // entry; the element layer measures it to size the marker column
    // so all items align at the same content edge regardless of
    // their own marker's width.
    let max_marker_text = compute_list_max_marker_text(node, kind);
    for child in &node.children {
        let NodeKind::ListItem { marker_range } = &child.kind else {
            continue;
        };
        let item_kind = match &kind {
            ListKind::Unordered => ListItemKind::Unordered(marker_char(source, marker_range)),
            ListKind::Ordered { .. } => {
                let n = next_number.unwrap_or(1);
                next_number = Some(n + 1);
                ListItemKind::Ordered { number: n }
            }
        };
        render_list_item(
            child,
            marker_range,
            item_kind,
            &max_marker_text,
            source,
            cursor,
            containers,
            out,
        );
    }
}

/// Widest marker *text* anywhere in this list.
///
/// For unordered lists every marker is two bytes (`- `, `* `, or `+ `);
/// they shape to nearly identical pixel widths. We canonicalize to
/// `"- "` so the indent computation is stable regardless of which
/// bullet char a particular item uses.
///
/// For ordered lists the widest marker is whichever item has the most
/// digits — `start + child_count - 1`. We format it back with the
/// `". "` suffix the parser uses today; once `)` markers are
/// supported the canonicalization will need to round-trip the actual
/// punctuation.
fn compute_list_max_marker_text(node: &SyntaxNode, kind: ListKind) -> String {
    match kind {
        ListKind::Unordered => "- ".to_string(),
        ListKind::Ordered { start } => {
            let count = node
                .children
                .iter()
                .filter(|c| matches!(c.kind, NodeKind::ListItem { .. }))
                .count() as u64;
            let max_n = if count <= 1 { start } else { start + count - 1 };
            format!("{}. ", max_n)
        }
    }
}

fn marker_char(source: &str, marker_range: &Range<usize>) -> u8 {
    let bytes = source.as_bytes();
    // Skip leading indent spaces; the first non-space byte is the
    // bullet.
    let mut p = marker_range.start;
    while p < marker_range.end && bytes[p] == b' ' {
        p += 1;
    }
    bytes.get(p).copied().unwrap_or(b'-')
}

/// Render one list item, walking its children in source order and
/// partitioning them into:
///
/// * **Inline runs** — contiguous spans of inline children (`Text`,
///   `SoftBreak`, `Strong`, …). Each run becomes one paragraph leaf.
///   For tight items pulldown emits these directly under the item;
///   for loose items they're wrapped in a `Paragraph` (handled
///   below).
/// * **Block-level children** — `Paragraph` (loose-item content),
///   `List` (nested list), `BlockQuote` (nested BQ), `CodeBlock`
///   (fenced code inside an item), `Heading`. Each emits its own
///   leaves via the standard recursion. Nested lists pick up
///   another `Container::ListItem` on top of this item's chain
///   entry — the recursive `render_node` call passes the chain
///   through.
///
/// The first leaf the item emits has its source range extended back
/// to the item's start so the marker (`- ` / `1. `) sits inside the
/// leaf for hide-and-overlay treatment. Subsequent leaves extend
/// back over their leading indent so the same hide-and-overlay
/// applies to continuation indent. The marker bytes are always
/// hidden from the shaped line (so content shapes at column 0
/// regardless of marker width or cursor position) and the marker
/// glyph is painted as a `MarkerOverlay` in the item's indent strip
/// — analogous to the blockquote `>` overlay treatment.
///
/// After all leaves are emitted, this item's own contribution to
/// the cumulative continuation indent is hidden on every leaf
/// inside its source range (direct + recursed). Each enclosing list
/// item's call hides *its* contribution on top, so cumulatively
/// the entire `total_marker_widths` worth of leading bytes
/// disappears from the shaped line — the visual indent is then
/// produced entirely by the container chain's left padding.
#[allow(clippy::too_many_arguments)]
fn render_list_item(
    node: &SyntaxNode,
    marker_range: &Range<usize>,
    item_kind: ListItemKind,
    list_max_marker_text: &str,
    source: &str,
    cursor: CursorRange,
    containers: &[Container],
    out: &mut Vec<RenderBlock>,
) {
    let cursor_inside = cursor.overlaps(&node.range);
    let item_marker_byte_len = marker_range.end - marker_range.start;
    let mut chain = containers.to_vec();
    chain.push(Container::ListItem {
        cursor_inside,
        kind: item_kind,
        marker_byte_len: item_marker_byte_len,
        list_max_marker_text: list_max_marker_text.to_string(),
    });
    // Index of *this* item's `Container::ListItem` entry in the chain
    // — used as the marker overlay's `level` so the element layer
    // paints into this level's indent strip.
    let item_level = chain.len() - 1;
    let bytes = source.as_bytes();
    let leaves_start_idx = out.len();

    // Walk children in source order, accumulating inline children
    // until we hit a block-level child. At each block boundary
    // (and at the end of the children list) flush the accumulated
    // inline run as one paragraph leaf, then either render the
    // block child as its own leaf (Paragraph) or recurse
    // (List / BlockQuote / CodeBlock / Heading).
    let mut emitted_first = false;
    let mut inline_run: Vec<&SyntaxNode> = Vec::new();
    let emit_inline_run =
        |group: &mut Vec<&SyntaxNode>, emitted_first: &mut bool, out: &mut Vec<RenderBlock>| {
            if group.is_empty() {
                return;
            }
            let start = group.first().unwrap().range.start;
            let end = group.last().unwrap().range.end;
            let mut range = start..end;
            extend_leading_range(&mut range, *emitted_first, node.range.start, bytes);
            let mut block = RenderBlock::new(range, BlockKind::Paragraph);
            block.containers = chain.clone();
            for child in group.iter() {
                walk_inline(child, cursor, InlineStyle::default(), &mut block);
            }
            out.push(block);
            *emitted_first = true;
            group.clear();
        };
    let emit_paragraph_child =
        |para: &SyntaxNode, emitted_first: &mut bool, out: &mut Vec<RenderBlock>| {
            let mut range = para.range.clone();
            extend_leading_range(&mut range, *emitted_first, node.range.start, bytes);
            let mut block = RenderBlock::new(range, BlockKind::Paragraph);
            block.containers = chain.clone();
            collect_inlines(para, cursor, &mut block);
            out.push(block);
            *emitted_first = true;
        };

    for child in &node.children {
        if is_block_level(&child.kind) {
            emit_inline_run(&mut inline_run, &mut emitted_first, out);
            match &child.kind {
                NodeKind::Paragraph => {
                    emit_paragraph_child(child, &mut emitted_first, out);
                }
                _ => {
                    // Nested list / blockquote / code block /
                    // heading — recurse with this item's container
                    // chain so any leaves the recursion emits
                    // carry it.
                    let before = out.len();
                    render_node(child, source, cursor, &chain, out);
                    if out.len() > before {
                        emitted_first = true;
                    }
                }
            }
        } else {
            inline_run.push(child);
        }
    }
    emit_inline_run(&mut inline_run, &mut emitted_first, out);

    // Empty item (no children at all, or the recursion emitted
    // nothing) — emit a single empty leaf so the item's source
    // range is claimed by something. Otherwise the cursor /
    // hit-test math has no block to anchor on.
    if !emitted_first {
        let mut block = RenderBlock::new(node.range.clone(), BlockKind::Paragraph);
        block.containers = chain;
        out.push(block);
    }

    // Identify the leaf that owns the marker line — i.e., the leaf
    // whose source_range starts at or before the item's start.
    // After `extend_leading_range` runs for the first emitted leaf,
    // it'll be the one with `source_range.start == node.range.start`.
    // For an empty item the synthetic leaf above also satisfies this.
    let first_leaf_idx = out
        .iter()
        .enumerate()
        .skip(leaves_start_idx)
        .find(|(_, leaf)| leaf.source_range.start <= node.range.start)
        .map(|(i, _)| i);

    if let Some(idx) = first_leaf_idx {
        let first_leaf = &mut out[idx];
        // Hide the marker bytes (and any leading ancestor indent on
        // this line) from the shaped first line — the marker is
        // rendered as an overlay glyph in the indent strip instead.
        let line_end = source_line_end(bytes, first_leaf.source_range.start);
        // `source_line_end` returns the offset *past* the trailing
        // `\n`; trim it so we don't accidentally hide the newline.
        let line_end = if line_end > 0 && bytes.get(line_end - 1) == Some(&b'\n') {
            line_end - 1
        } else {
            line_end
        };
        let hide_end = marker_range.end.min(line_end);
        if hide_end > first_leaf.source_range.start {
            first_leaf
                .hidden_ranges
                .push(first_leaf.source_range.start..hide_end);
        }
        // Push the marker overlay so the element layer paints the
        // marker text in this item's indent strip. The element layer
        // resolves the marker text to display from `containers[level]`
        // (kind + cursor_inside).
        first_leaf.marker_overlays.push(MarkerOverlay {
            source_range: marker_range.clone(),
            level: item_level,
        });
    }

    // Hide *this* item's contribution to leading indent on every
    // line of every leaf the item produced (direct + recursed). Each
    // enclosing list item adds its own contribution on top, so the
    // cumulative `total_marker_widths` of leading whitespace is
    // hidden across the chain.
    for leaf in &mut out[leaves_start_idx..] {
        hide_item_continuation_indent(leaf, item_marker_byte_len, bytes);
    }
}

/// Hide up to `item_marker_byte_len` leading-space bytes on every
/// line in `leaf.source_range`. Marker chars on the first leaf's
/// first line are hidden separately by `render_list_item` as a bulk
/// range that may overlap this hide — `build_display_line` collapses
/// overlapping hidden ranges, so the redundancy is benign.
fn hide_item_continuation_indent(
    leaf: &mut RenderBlock,
    item_marker_byte_len: usize,
    bytes: &[u8],
) {
    if item_marker_byte_len == 0 {
        return;
    }
    let leaf_range = leaf.source_range.clone();
    let mut p = leaf_range.start;
    while p < leaf_range.end {
        let line_end_inclusive_nl = source_line_end(bytes, p);
        let line_content_end =
            if line_end_inclusive_nl > 0 && bytes.get(line_end_inclusive_nl - 1) == Some(&b'\n') {
                line_end_inclusive_nl - 1
            } else {
                line_end_inclusive_nl
            }
            .min(leaf_range.end);

        let mut q = p;
        let limit = (p + item_marker_byte_len).min(line_content_end);
        while q < limit && bytes[q] == b' ' {
            q += 1;
        }
        if q > p {
            leaf.hidden_ranges.push(p..q);
        }
        p = line_end_inclusive_nl.min(leaf_range.end);
        if p < leaf_range.end && bytes[p] == b'\n' {
            p += 1;
        } else if p == line_end_inclusive_nl && p == line_content_end {
            // No trailing newline (last line) and we've consumed the
            // whole line — stop the loop.
            break;
        }
    }
}

fn is_block_level(kind: &NodeKind) -> bool {
    matches!(
        kind,
        NodeKind::Paragraph
            | NodeKind::Heading { .. }
            | NodeKind::CodeBlock { .. }
            | NodeKind::BlockQuote { .. }
            | NodeKind::List { .. }
    )
}

/// For the *first* leaf of a list item, extend its source range
/// back to the item's start so the marker shapes into the leaf.
/// For *subsequent* leaves, extend back over leading spaces on the
/// line so the indent shapes with the content rather than vanishing.
fn extend_leading_range(
    range: &mut Range<usize>,
    already_emitted_first: bool,
    item_start: usize,
    bytes: &[u8],
) {
    if !already_emitted_first {
        range.start = item_start;
    } else {
        while range.start > 0 && bytes[range.start - 1] == b' ' {
            range.start -= 1;
        }
    }
}

/// True if `pos` is the byte index right after a hard-break `\n`
/// (`  \n` or `\\\n`). Used by `render_blockquote` to recognize that
/// a marker at `pos` introduces a paragraph-continuation line, not
/// the middle of a structural pair.
fn is_after_hard_break(bytes: &[u8], pos: usize) -> bool {
    if pos == 0 {
        return false;
    }
    let nl = pos - 1;
    if bytes.get(nl) != Some(&b'\n') {
        return false;
    }
    let preceded_by_two_spaces = nl >= 2 && bytes[nl - 1] == b' ' && bytes[nl - 2] == b' ';
    let preceded_by_backslash = nl >= 1 && bytes[nl - 1] == b'\\';
    preceded_by_two_spaces || preceded_by_backslash
}

/// Find the latest leaf in `slice` whose source range ends *at* `pos`
/// (i.e., the leaf right before a hard-break continuation marker
/// that needs to extend it forward).
fn find_leaf_ending_at(slice: &mut [RenderBlock], pos: usize) -> Option<&mut RenderBlock> {
    slice
        .iter_mut()
        .rev()
        .find(|leaf| leaf.source_range.end == pos)
}

/// Attach one blockquote prefix marker to `leaf`. The marker is
/// *always* hidden from the shaped line (so content position never
/// shifts as the cursor moves in or out of the construct). When the
/// cursor is inside, an entry is also recorded in `marker_overlays`
/// at this blockquote's `level` — the element layer paints those
/// glyphs on top of the corresponding border bar so the user still
/// sees the raw `>` markers when focused, just shifted onto the
/// container decoration instead of inlined into the text.
fn attach_marker(leaf: &mut RenderBlock, prefix: &Range<usize>, cursor_inside: bool, level: usize) {
    leaf.hidden_ranges.push(prefix.clone());
    if cursor_inside {
        leaf.marker_overlays
            .push(crate::render_spec::MarkerOverlay {
                source_range: prefix.clone(),
                level,
            });
    }
}

/// Byte offset just past the end of the source line containing `pos`,
/// inclusive of any trailing `\n`. Mirrors how pulldown-cmark ranges
/// parsed paragraphs (the trailing `\n` is part of the leaf's range,
/// then trimmed by `inject_empty_paragraphs`).
fn source_line_end(bytes: &[u8], pos: usize) -> usize {
    let mut q = pos;
    while q < bytes.len() && bytes[q] != b'\n' {
        q += 1;
    }
    if q < bytes.len() { q + 1 } else { q }
}

/// Find the leaf in `slice` whose source range covers the byte at
/// `prefix.end` — i.e. the first byte of content the prefix introduces.
/// Pulldown ranges most leaves so they *start* exactly at that byte, so
/// the predicate `start <= prefix.end < end` matches both (a) a
/// multi-line leaf that contains the prefix mid-range and (b) the
/// boundary case where the leaf begins right where the prefix ends. A
/// stand-alone `>` line that isn't part of a parsed leaf is dropped —
/// same convention the renderer already uses for bytes no leaf claims.
fn find_leaf_for_prefix<'a>(
    slice: &'a mut [RenderBlock],
    prefix: &Range<usize>,
) -> Option<&'a mut RenderBlock> {
    let target = prefix.end;
    slice
        .iter_mut()
        .find(|b| b.source_range.start <= target && target < b.source_range.end)
}

fn render_code_block(
    node: &SyntaxNode,
    cursor: CursorRange,
    containers: &[Container],
    out: &mut Vec<RenderBlock>,
) {
    let (lang, delimiter_ranges, info_string_range) = match &node.kind {
        NodeKind::CodeBlock {
            lang,
            delimiter_ranges,
            info_string_range,
            ..
        } => (
            lang.clone(),
            delimiter_ranges.clone(),
            info_string_range.clone(),
        ),
        _ => return,
    };

    let mut block = RenderBlock::new(node.range.clone(), BlockKind::CodeBlock { lang });
    block.containers = containers.to_vec();
    let cursor_inside = cursor.overlaps(&node.range);

    // Fence chars (` ``` ` / `~~~`) — hide-when-outside, dim-when-inside.
    apply_delimiter_visibility(&delimiter_ranges, cursor_inside, &mut block);

    // Info string (the language tag after the opening fence). It's
    // visible when the cursor is outside the construct (so a reader
    // can still see the language at a glance) but dimmed when the
    // cursor is inside (consistent with the fence treatment).
    if let Some(info) = info_string_range.as_ref()
        && cursor_inside
    {
        block.inlines.push(InlineRun {
            source_range: info.clone(),
            style: InlineStyle::dimmed(),
        });
    }

    // Mark fence rows for layout. The opener line covers the fence
    // chars *plus* any info string; the closer line is just its
    // fence chars. The element layer uses these to keep fence rows
    // pinned (no horizontal scroll), reserve vertical space for
    // them, and paint them outside the content mask.
    let opener_line_end = info_string_range
        .as_ref()
        .map(|r| r.end)
        .unwrap_or_else(|| delimiter_ranges[0].end);
    block
        .delimiter_lines
        .push(delimiter_ranges[0].start..opener_line_end);
    if let Some(closer) = delimiter_ranges.get(1) {
        block.delimiter_lines.push(closer.clone());
    }

    // No inline children — code-block content is literal source bytes,
    // shaped in mono font by the element layer.
    out.push(block);
}

fn render_heading(
    node: &SyntaxNode,
    cursor: CursorRange,
    containers: &[Container],
    out: &mut Vec<RenderBlock>,
) {
    let (level, content_range, delimiter_ranges) = match &node.kind {
        NodeKind::Heading {
            level,
            content_range,
            delimiter_ranges,
        } => (*level, content_range.clone(), delimiter_ranges.clone()),
        _ => return,
    };

    let mut block = RenderBlock::new(node.range.clone(), BlockKind::Heading { level });
    block.containers = containers.to_vec();
    let cursor_inside = cursor.overlaps(&node.range);
    apply_delimiter_visibility(&delimiter_ranges, cursor_inside, &mut block);
    collect_inlines_in_range(node, &content_range, cursor, &mut block);

    out.push(block);
}

fn collect_inlines(node: &SyntaxNode, cursor: CursorRange, block: &mut RenderBlock) {
    collect_inlines_in_range(node, &node.range, cursor, block);
}

fn collect_inlines_in_range(
    node: &SyntaxNode,
    bound: &Range<usize>,
    cursor: CursorRange,
    block: &mut RenderBlock,
) {
    for child in &node.children {
        if !ranges_overlap(&child.range, bound) {
            continue;
        }
        walk_inline(child, cursor, InlineStyle::default(), block);
    }
}

fn ranges_overlap(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.start < b.end && a.end > b.start
}

fn walk_inline(node: &SyntaxNode, cursor: CursorRange, base: InlineStyle, block: &mut RenderBlock) {
    match &node.kind {
        NodeKind::Strong {
            delimiter_ranges,
            content_range,
        } => {
            let inside = cursor.overlaps(&node.range);
            apply_delimiter_visibility(delimiter_ranges, inside, block);
            walk_styled_children(
                node,
                content_range,
                cursor,
                base.merge(InlineStyle::bold()),
                block,
            );
        }
        NodeKind::Emphasis {
            delimiter_ranges,
            content_range,
        } => {
            let inside = cursor.overlaps(&node.range);
            apply_delimiter_visibility(delimiter_ranges, inside, block);
            walk_styled_children(
                node,
                content_range,
                cursor,
                base.merge(InlineStyle::italic()),
                block,
            );
        }
        NodeKind::Strikethrough {
            delimiter_ranges,
            content_range,
        } => {
            let inside = cursor.overlaps(&node.range);
            apply_delimiter_visibility(delimiter_ranges, inside, block);
            let mut style = base.clone();
            style.strikethrough = true;
            walk_styled_children(node, content_range, cursor, style, block);
        }
        NodeKind::Text => {
            if !base.is_default() {
                block.inlines.push(InlineRun {
                    source_range: node.range.clone(),
                    style: base,
                });
            }
        }
        NodeKind::SoftBreak | NodeKind::HardBreak => {
            // Nothing to emit.
        }
        _ => {
            for child in &node.children {
                walk_inline(child, cursor, base.clone(), block);
            }
        }
    }
}

fn walk_styled_children(
    node: &SyntaxNode,
    bound: &Range<usize>,
    cursor: CursorRange,
    style: InlineStyle,
    block: &mut RenderBlock,
) {
    if node.children.is_empty() {
        if !style.is_default() {
            block.inlines.push(InlineRun {
                source_range: bound.clone(),
                style,
            });
        }
        return;
    }
    for child in &node.children {
        if !ranges_overlap(&child.range, bound) {
            continue;
        }
        walk_inline(child, cursor, style.clone(), block);
    }
}

fn apply_delimiter_visibility(
    delimiters: &[Range<usize>],
    cursor_inside: bool,
    block: &mut RenderBlock,
) {
    for d in delimiters {
        if d.is_empty() {
            continue;
        }
        if cursor_inside {
            block.inlines.push(InlineRun {
                source_range: d.clone(),
                style: InlineStyle::dimmed(),
            });
        } else {
            block.hidden_ranges.push(d.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// Empty-paragraph injection (see module docs for the formulas).
// ---------------------------------------------------------------------------

fn inject_empty_paragraphs(source: &str, real_blocks: Vec<RenderBlock>) -> Vec<RenderBlock> {
    let bytes = source.as_bytes();

    // Special case: no real blocks at all. The content-bearing formulas
    // assume at least one parsed block to anchor empties around; here
    // there isn't. Two things go wrong if we just return an empty `Vec`:
    //
    //   1. No `BlockElement::paint` runs for this frame, so no
    //      `window.handle_input` registers the `EntityInputHandler` and
    //      typed text has nowhere to route.
    //   2. There's no shaped line to paint a cursor against, so the user
    //      can't see where they are and can't click to place the cursor.
    //
    // Pairs-model layout for content-empty docs: emit `N/2` empty
    // paragraph blocks (each spanning a `\n\n` pair) followed by one
    // zero-byte anchor block at the doc end. Pressing Enter `N` times
    // from an empty doc produces `N` pairs and shows `N + 1` rows —
    // typewriter intuition.
    if real_blocks.is_empty() {
        let n_newlines = bytes.iter().filter(|&&b| b == b'\n').count();
        let n_pairs = n_newlines / 2;
        let mut out = Vec::with_capacity(n_pairs + 1);
        for i in 0..n_pairs {
            out.push(empty_paragraph_pair(2 * i));
        }
        let anchor_start = 2 * n_pairs;
        out.push(RenderBlock::new(
            anchor_start..bytes.len(),
            BlockKind::Paragraph,
        ));
        return out;
    }

    // Trim each real block's source_range to the block's content extent.
    // pulldown-cmark folds one trailing `\n` into a paragraph's parser
    // range; without trimming, the trailing `\n` would be invisible to
    // the gap arithmetic (see the bug report: pressing Enter once at end
    // of paragraph produced no visible change). After trimming, the gap
    // arithmetic is uniform: every `\n` outside any block range is a
    // free-zone newline, and the trim is a no-op for blocks the parser
    // ranged tightly.
    let mut real_blocks = real_blocks;
    for block in &mut real_blocks {
        let trimmed_start = block_content_start(bytes, &block.source_range);
        let trimmed_end = block_content_end_excl(bytes, &block.source_range).max(trimmed_start);
        block.source_range = trimmed_start..trimmed_end;
    }

    let mut out: Vec<RenderBlock> = Vec::with_capacity(real_blocks.len() * 2);

    // Leading gap. Each pair of `\n`s is one leading empty paragraph.
    let first_content = real_blocks[0].source_range.start;
    let leading_count = (0..first_content).filter(|&p| bytes[p] == b'\n').count();
    let leading_empties = leading_count / 2;
    for i in 0..leading_empties {
        let start = 2 * i;
        out.push(empty_paragraph_pair(start));
    }

    // Real blocks, with inter-block empties before each. One pair is the
    // paragraph-break separator itself; the rest are empties.
    for (i, block) in real_blocks.iter().enumerate() {
        if i > 0 {
            let prev_end = real_blocks[i - 1].source_range.end;
            let next_start = block.source_range.start;
            let gap_count = (prev_end..next_start)
                .filter(|&p| bytes[p] == b'\n')
                .count();
            let inter_empties = gap_count.saturating_sub(2) / 2;
            // Each empty's pair starts at offset 1 (mod 2) inside the
            // gap. The first `\n` of the gap is "owned" by the previous
            // paragraph's boundary; the second is the first byte of the
            // first empty's pair, etc. This layout keeps every cursor
            // position from `prev_end` through `next_start` covered by
            // either a block boundary or strict-in-empty hit.
            for k in 0..inter_empties {
                let start = prev_end + 2 * k + 1;
                out.push(empty_paragraph_pair(start));
            }
        }
        out.push(block.clone());
    }

    // Trailing gap. Same offset-by-1 layout as the inter-block case so
    // the cursor's natural resting position (the boundary between this
    // empty's pair and the next pair) falls strictly inside the empty's
    // range. With offset 0 — the layout the leading case still uses —
    // the empty's start coincides with the previous block's end, so
    // typing at "the cursor on row N" position would extend the
    // previous paragraph rather than create new content for row N.
    //
    // The shift by 1 pushes each trailing pair forward by one byte. The
    // *last* pair would extend one byte past the document, so we clamp
    // it: the final empty's range becomes a 1-byte slice over its lone
    // `\n`. Visually it still renders as one empty row (the
    // all-newlines fast-path in `shape_block_lines` emits a single line
    // for any block whose text is purely `\n`s); functionally, the
    // cursor at end-of-doc is `range.end` (allowed) and the last block
    // claims it via the end-clause in `block_claims_cursor`.
    let last_end = real_blocks
        .last()
        .expect("checked non-empty above")
        .source_range
        .end;
    let trailing_count = (last_end..bytes.len())
        .filter(|&p| bytes[p] == b'\n')
        .count();
    let trailing_empties = trailing_count / 2;
    for i in 0..trailing_empties {
        let start = last_end + 2 * i + 1;
        let end = (start + 2).min(bytes.len());
        out.push(RenderBlock::new(start..end, BlockKind::Paragraph));
    }

    out
}

/// Position of the first non-`\n` byte in `range`. (Falls back to the end
/// if the range is all newlines, but in practice block ranges from
/// pulldown-cmark always contain at least one content character.)
fn block_content_start(bytes: &[u8], range: &Range<usize>) -> usize {
    let mut p = range.start;
    while p < range.end && bytes[p] == b'\n' {
        p += 1;
    }
    p
}

/// One past the last non-`\n` byte in `range`, *except* `\n`s that are
/// hard breaks (`  \n` or `\\\n`). Hard breaks are in-paragraph line
/// breaks — part of the block's content — so they stay inside the
/// block's range. Only "structural" trailing `\n`s (the kind pulldown-
/// cmark folds in as a paragraph terminator) get trimmed.
fn block_content_end_excl(bytes: &[u8], range: &Range<usize>) -> usize {
    let mut p = range.end;
    while p > range.start && bytes[p - 1] == b'\n' {
        let preceded_by_two_spaces = p >= 3 && bytes[p - 2] == b' ' && bytes[p - 3] == b' ';
        let preceded_by_backslash = p >= 2 && bytes[p - 2] == b'\\';
        if preceded_by_two_spaces || preceded_by_backslash {
            break;
        }
        p -= 1;
    }
    p
}

/// Synthetic empty paragraph block spanning the pair of `\n`s starting
/// at `start` (so its range is `[start, start + 2)`).
fn empty_paragraph_pair(start: usize) -> RenderBlock {
    RenderBlock::new(start..start + 2, BlockKind::Paragraph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn render_with_cursor(src: &str, cursor: usize) -> RenderSpec {
        let state = EditorState {
            markdown: src.into(),
            selection: Selection::Cursor(cursor),
        };
        let tree = parse(src);
        render(&state, &tree)
    }

    fn find_block(spec: &RenderSpec, predicate: impl Fn(&RenderBlock) -> bool) -> &RenderBlock {
        spec.blocks
            .iter()
            .find(|b| predicate(b))
            .expect("expected matching block")
    }

    #[test]
    fn heading_hides_prefix_when_cursor_outside() {
        let src = "# Hello\n\npara";
        let spec = render_with_cursor(src, src.len());
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Heading { .. }));
        assert!(block.has_hidden_range(0..2));
        assert!(!block.has_dimmed_range(0..2));
    }

    #[test]
    fn heading_dims_prefix_when_cursor_inside() {
        let src = "# Hello\n";
        let spec = render_with_cursor(src, 4);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Heading { .. }));
        assert!(block.has_dimmed_range(0..2));
        assert!(!block.has_hidden_range(0..2));
    }

    #[test]
    fn heading_level_is_recorded() {
        let spec = render_with_cursor("### Hi", 0);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Heading { .. }));
        match block.kind {
            BlockKind::Heading { level } => assert_eq!(level, 3),
            _ => panic!(),
        }
    }

    #[test]
    fn bold_hides_asterisks_when_cursor_outside() {
        let src = "a **bold** b";
        let spec = render_with_cursor(src, 0);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert!(block.has_hidden_range(2..4));
        assert!(block.has_hidden_range(8..10));
    }

    #[test]
    fn bold_dims_asterisks_when_cursor_inside() {
        let src = "a **bold** b";
        let spec = render_with_cursor(src, 5);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert!(block.has_dimmed_range(2..4));
        assert!(block.has_dimmed_range(8..10));
    }

    #[test]
    fn bold_emits_bold_inline_run_for_content() {
        let spec = render_with_cursor("**bold**", 0);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert!(
            block
                .inlines
                .iter()
                .any(|r| r.source_range == (2..6) && r.style.bold)
        );
    }

    #[test]
    fn italic_hides_when_cursor_outside() {
        let spec = render_with_cursor("*x*", 999);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert!(block.has_hidden_range(0..1));
        assert!(block.has_hidden_range(2..3));
    }

    #[test]
    fn italic_dims_when_cursor_inside() {
        let spec = render_with_cursor("*x*", 1);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert!(block.has_dimmed_range(0..1));
        assert!(block.has_dimmed_range(2..3));
    }

    #[test]
    fn strikethrough_hides_tildes_when_cursor_outside() {
        let spec = render_with_cursor("~~gone~~", 999);
        let block = &spec.blocks[0];
        assert!(block.has_hidden_range(0..2));
        assert!(block.has_hidden_range(6..8));
    }

    #[test]
    fn strikethrough_dims_when_cursor_inside_and_emits_run() {
        let spec = render_with_cursor("~~gone~~", 4);
        let block = &spec.blocks[0];
        assert!(block.has_dimmed_range(0..2));
        assert!(block.has_dimmed_range(6..8));
        assert!(
            block
                .inlines
                .iter()
                .any(|r| r.source_range == (2..6) && r.style.strikethrough)
        );
    }

    #[test]
    fn nested_emphasis_inside_strong_marks_both_styles() {
        let spec = render_with_cursor("***x***", 999);
        let block = &spec.blocks[0];
        assert!(block.inlines.iter().any(|r| r.style.bold && r.style.italic));
    }

    #[test]
    fn selection_overlapping_construct_dims_delimiters() {
        let state = EditorState {
            markdown: "**bold**".into(),
            selection: Selection::Range { anchor: 1, head: 6 },
        };
        let tree = parse(&state.markdown);
        let spec = render(&state, &tree);
        let block = &spec.blocks[0];
        assert!(block.has_dimmed_range(0..2));
        assert!(block.has_dimmed_range(6..8));
    }

    #[test]
    fn cursor_at_construct_boundary_treated_as_inside() {
        let spec = render_with_cursor("**b**", 0);
        let block = &spec.blocks[0];
        assert!(block.has_dimmed_range(0..2));
    }

    #[test]
    fn empty_document_yields_one_anchor_block() {
        // Without at least one block, no `BlockElement::paint` runs, so
        // the input handler never registers and the cursor never paints
        // — that's the "deleted everything, can't type anymore" bug. The
        // injector emits a single zero-byte anchor for "" so the editor
        // is always usable.
        let spec = render_with_cursor("", 0);
        assert_eq!(spec.blocks.len(), 1);
        assert_eq!(spec.blocks[0].source_range, 0..0);
        assert!(matches!(spec.blocks[0].kind, BlockKind::Paragraph));
    }

    #[test]
    fn paragraph_with_no_inline_constructs_has_no_inline_runs() {
        let spec = render_with_cursor("plain text", 999);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert!(block.inlines.is_empty());
        assert!(block.hidden_ranges.is_empty());
    }

    // ---- Empty-paragraph injection ----------------------------------------

    /// Count "empty paragraph" blocks in a spec — synthetic blocks
    /// produced by `inject_empty_paragraphs`. Most synthetic empties
    /// span 2 `\n` bytes (a full pair), but the *last* trailing empty
    /// is clamped to doc length and may span just 1 `\n`. Either way
    /// the block's text is purely `\n`s, which is what we check (so a
    /// real 2-char paragraph like `"p1"` doesn't get miscounted).
    fn count_empty_blocks(spec: &RenderSpec, src: &str) -> usize {
        let bytes = src.as_bytes();
        spec.blocks
            .iter()
            .filter(|b| {
                if !matches!(b.kind, BlockKind::Paragraph) {
                    return false;
                }
                if b.source_range.end <= b.source_range.start {
                    return false;
                }
                (b.source_range.start..b.source_range.end)
                    .all(|i| bytes.get(i).copied() == Some(b'\n'))
            })
            .count()
    }

    #[test]
    fn paragraph_break_alone_produces_no_empty_blocks() {
        // `\n\n` between two paragraphs is exactly the paragraph break
        // separator — no empty paragraph block.
        let src = "p1\n\np2";
        let spec = render_with_cursor(src, 0);
        assert_eq!(count_empty_blocks(&spec, src), 0);
        assert_eq!(spec.blocks.len(), 2);
    }

    #[test]
    fn extra_inter_block_pair_emits_one_empty_paragraph() {
        // 4 `\n`s between content = paragraph break (1 pair) + 1 empty
        // paragraph (1 pair).
        let src = "p1\n\n\n\np2";
        let spec = render_with_cursor(src, 0);
        assert_eq!(count_empty_blocks(&spec, src), 1);
        assert_eq!(spec.blocks.len(), 3);
    }

    #[test]
    fn user_example_six_newlines_emits_two_empty_paragraphs() {
        // 6 `\n`s = paragraph break + 2 empty paragraphs (the user's
        // intent of "paragraph 1, two empty rows, paragraph 2" expressed
        // in the pairs model).
        let src = "paragraph 1\n\n\n\n\n\nparagraph 2";
        let spec = render_with_cursor(src, 0);
        assert_eq!(count_empty_blocks(&spec, src), 2);
        assert_eq!(spec.blocks.len(), 4);
    }

    fn synthetic_empties(spec: &RenderSpec, src: &str) -> Vec<RenderBlock> {
        spec.blocks
            .iter()
            .filter(|b| {
                matches!(b.kind, BlockKind::Paragraph)
                    && b.source_range.end - b.source_range.start == 2
                    && src.as_bytes()[b.source_range.start] == b'\n'
                    && src.as_bytes()[b.source_range.start + 1] == b'\n'
            })
            .cloned()
            .collect()
    }

    #[test]
    fn inter_block_empties_each_span_two_newlines() {
        // 6 `\n`s between content = 2 empties at offset 1 inside the
        // inter-block gap (positions 3..5 and 5..7 for `p1\n\n\n\n\n\np2`
        // with p1 trimmed to 0..2).
        let src = "p1\n\n\n\n\n\np2";
        let spec = render_with_cursor(src, 0);
        let empties = synthetic_empties(&spec, src);
        assert_eq!(empties.len(), 2);
        for b in &empties {
            assert_eq!(b.source_range.end - b.source_range.start, 2);
            assert_eq!(src.as_bytes()[b.source_range.start], b'\n');
            assert_eq!(src.as_bytes()[b.source_range.start + 1], b'\n');
        }
        let positions: Vec<_> = empties.iter().map(|b| b.source_range.start).collect();
        let mut sorted = positions.clone();
        sorted.sort();
        assert_eq!(positions, sorted);
    }

    #[test]
    fn leading_newline_pair_emits_one_empty() {
        // `\n\np1` has 2 leading `\n`s = 1 leading empty paragraph
        // above the first real block.
        let src = "\n\np1";
        let spec = render_with_cursor(src, 0);
        assert_eq!(count_empty_blocks(&spec, src), 1);
    }

    #[test]
    fn leading_two_pairs_emit_two_empties() {
        // Four leading `\n`s = 2 empties (two Enters at the start).
        let src = "\n\n\n\np1";
        let spec = render_with_cursor(src, 0);
        assert_eq!(count_empty_blocks(&spec, src), 2);
    }

    #[test]
    fn single_leading_newline_emits_no_empty_in_pairs_model() {
        // Anomalous: a single leading `\n` doesn't form a complete pair,
        // so it doesn't render as a leading empty paragraph. In normal
        // user flow this state is unreachable (Enter inserts `\n\n`);
        // it's only producible via paste of pre-existing text.
        let src = "\np1";
        let spec = render_with_cursor(src, 0);
        assert_eq!(count_empty_blocks(&spec, src), 0);
    }

    #[test]
    fn trailing_newline_pairs_emit_empties() {
        // Each trailing pair = 1 trailing empty.
        let src = "p1\n\n";
        assert_eq!(count_empty_blocks(&render_with_cursor(src, 0), src), 1);
        let src = "p1\n\n\n\n";
        assert_eq!(count_empty_blocks(&render_with_cursor(src, 0), src), 2);
        let src = "p1\n\n\n\n\n\n";
        assert_eq!(count_empty_blocks(&render_with_cursor(src, 0), src), 3);
    }

    #[test]
    fn single_trailing_newline_emits_no_empty_in_pairs_model() {
        // Anomalous: 1 trailing `\n` is half a pair. In the pairs model
        // user flow, Enter inserts `\n\n`, so this state would only come
        // from paste of `paragraph 1\n` etc. The renderer drops it
        // (`T / 2 = 0`); `enforce_invariants` doesn't promote it (run is
        // exactly 1 at the document edge, allowed by the soft-break rule).
        let src = "paragraph 1\n";
        let spec = render_with_cursor(src, 0);
        assert_eq!(count_empty_blocks(&spec, src), 0);
    }

    #[test]
    fn enter_at_end_of_paragraph_renders_one_trailing_empty() {
        // The user-flow regression: Enter from end of `paragraph 1`
        // produces `paragraph 1\n\n` (pairs model). Render must show a
        // visible empty trailing row. Trailing empties use the same
        // offset-by-1 layout as inter-block empties; the last empty is
        // clamped to doc length, giving a 1-byte range that still
        // shapes to one visible row.
        let src = "paragraph 1\n\n";
        let spec = render_with_cursor(src, 13);
        // Real "paragraph 1" + 1 trailing synthetic empty.
        assert_eq!(spec.blocks.len(), 2);
        let trailing = spec
            .blocks
            .iter()
            .find(|b| b.source_range == (12..13))
            .expect("synthetic empty owning the clamped trailing pair");
        assert!(matches!(trailing.kind, BlockKind::Paragraph));
    }

    #[test]
    fn trailing_hard_break_keeps_block_range_intact() {
        // Pressing Shift+Enter at the end of "paragraph 1" produces
        // "paragraph 1  \n" — the trailing `\n` is part of an
        // in-paragraph hard break, not a paragraph terminator. The
        // block's range must still cover it (so the element layer can
        // render the implicit empty trailing line within the same
        // paragraph), and *no* trailing empty paragraph block should be
        // emitted (that would produce an extra paragraph_gap).
        let spec = render_with_cursor("paragraph 1  \n", 0);
        assert_eq!(spec.blocks.len(), 1);
        let block = &spec.blocks[0];
        assert_eq!(block.source_range, 0..14);
        assert!(matches!(block.kind, BlockKind::Paragraph));
    }

    #[test]
    fn trailing_backslash_hard_break_kept_in_block_range() {
        let spec = render_with_cursor("paragraph 1\\\n", 0);
        assert_eq!(spec.blocks.len(), 1);
        assert_eq!(spec.blocks[0].source_range, 0..13);
    }

    #[test]
    fn paragraph_block_range_is_trimmed_of_trailing_newline() {
        // Verify the trim happens — without it, the trailing-empty range
        // would overlap the paragraph's range and the cursor at the
        // boundary would land on the paragraph instead of the empty.
        let spec = render_with_cursor("paragraph 1\n", 0);
        let real = spec
            .blocks
            .iter()
            .find(|b| !b.inlines.is_empty() || b.source_range.end - b.source_range.start > 1)
            .or_else(|| {
                // Fallback: real paragraph has range 0..11 (trimmed).
                spec.blocks
                    .iter()
                    .find(|b| b.source_range.start == 0 && b.source_range.end == 11)
            })
            .expect("real paragraph block");
        assert_eq!(real.source_range, 0..11);
    }

    #[test]
    fn empties_around_a_heading() {
        // Headings use the same inter-block formula as paragraphs.
        // 6 `\n`s between heading content and body content = paragraph
        // break + 2 empty paragraphs.
        let src = "# title\n\n\n\n\n\nbody";
        let spec = render_with_cursor(src, 0);
        assert_eq!(count_empty_blocks(&spec, src), 2);
        assert_eq!(spec.blocks.len(), 4);
    }

    #[test]
    fn content_empty_doc_emits_one_block_per_pair_plus_anchor() {
        // Pairs-model layout for content-empty docs: `N/2` synthetic
        // empty paragraph blocks (each `\n\n`) plus one zero-byte
        // anchor block at the doc end. Pressing Enter `N` times in an
        // empty doc gives `N` pairs and `N + 1` visible rows.
        //
        // `\n\n\n\n` has 4 `\n`s = 2 pairs. Two empty pair blocks
        // [0..2) and [2..4), plus a trailing anchor [4..4).
        let spec = render_with_cursor("\n\n\n\n", 0);
        let ranges: Vec<_> = spec.blocks.iter().map(|b| b.source_range.clone()).collect();
        assert_eq!(ranges, vec![0..2, 2..4, 4..4]);
        assert!(
            spec.blocks
                .iter()
                .all(|b| matches!(b.kind, BlockKind::Paragraph))
        );
    }

    #[test]
    fn one_enter_from_empty_doc_emits_two_blocks() {
        // Source `\n\n` is "Enter once in an empty doc": one pair → one
        // visible empty above + one anchor at doc end.
        let spec = render_with_cursor("\n\n", 2);
        assert_eq!(spec.blocks.len(), 2);
        assert_eq!(spec.blocks[0].source_range, 0..2);
        assert_eq!(spec.blocks[1].source_range, 2..2);
    }

    #[test]
    fn anomalous_lone_newline_emits_one_anchor_block() {
        // Anomalous: a single `\n` doesn't form a pair. `enforce_invariants`
        // leaves a single doc-edge `\n` alone, and the renderer treats
        // the whole thing as one anchor block (the all-newlines
        // fast-path in `shape_block_lines` still produces one visible
        // row).
        let spec = render_with_cursor("\n", 0);
        assert_eq!(spec.blocks.len(), 1);
        assert_eq!(spec.blocks[0].source_range, 0..1);
    }

    // ---- Fenced code blocks ----------------------------------------------

    #[test]
    fn fenced_code_block_emits_one_block_with_lang() {
        let src = "```rust\nlet x = 1;\n```\n";
        let spec = render_with_cursor(src, 0);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::CodeBlock { .. }));
        match &block.kind {
            BlockKind::CodeBlock { lang } => assert_eq!(lang.as_deref(), Some("rust")),
            _ => unreachable!(),
        }
    }

    #[test]
    fn fenced_code_block_hides_fences_when_cursor_outside() {
        // Cursor placed at a paragraph below the code block.
        let src = "```rust\nlet x = 1;\n```\n\npara";
        let cursor = src.find("para").unwrap() + 1;
        let spec = render_with_cursor(src, cursor);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::CodeBlock { .. }));
        // Opener fence "```" = 0..3, closer "```" = 19..22 — only the
        // fence chars are hidden, so the info string ("rust" at 3..7)
        // stays visible.
        assert!(block.has_hidden_range(0..3));
        assert!(block.has_hidden_range(19..22));
        // Info string is NOT dimmed when cursor is outside; it
        // renders with normal styling.
        assert!(!block.has_dimmed_range(3..7));
        // And it's NOT hidden either.
        assert!(!block.has_hidden_range(3..7));
    }

    #[test]
    fn fenced_code_block_dims_fences_when_cursor_inside() {
        let src = "```rust\nlet x = 1;\n```";
        // Cursor inside the content.
        let spec = render_with_cursor(src, 10);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::CodeBlock { .. }));
        // Opener fence + info string + closer all dim when cursor is
        // inside the construct.
        assert!(block.has_dimmed_range(0..3));
        assert!(block.has_dimmed_range(3..7));
        assert!(block.has_dimmed_range(19..22));
        assert!(!block.has_hidden_range(0..3));
    }

    #[test]
    fn fenced_code_block_marks_fence_rows_for_layout() {
        // `delimiter_lines` lists whole fence-row ranges so the
        // element layer can pin them outside horizontal scroll
        // regardless of cursor position.
        let src = "```rust\nlet x = 1;\n```";
        let spec = render_with_cursor(src, 999);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::CodeBlock { .. }));
        // Opener line covers fence + info string (0..7); closer
        // is just its fence chars (19..22).
        assert!(block.delimiter_lines.contains(&(0..7)));
        assert!(block.delimiter_lines.contains(&(19..22)));
    }

    #[test]
    fn fenced_code_block_has_no_inline_runs_for_pseudo_markdown() {
        // `**` inside a code block is literal, not a Strong delimiter —
        // there should be no bold inline run, no hidden range for the
        // asterisks.
        let src = "```\n**not bold**\n```";
        let spec = render_with_cursor(src, 0);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::CodeBlock { .. }));
        assert!(
            block
                .inlines
                .iter()
                .all(|r| !r.style.bold && !r.style.italic),
            "code-block content must not be styled by inline markdown",
        );
    }

    // ---- Blockquotes -----------------------------------------------------

    #[test]
    fn blockquote_paragraph_emits_one_renderblock_with_container() {
        // `> hi` parses as BlockQuote(Paragraph(hi)). The renderer
        // emits one RenderBlock for the paragraph carrying a single
        // BlockQuote container. The trailing "\n\nbody" puts the
        // cursor truly outside the blockquote (boundary equality
        // would otherwise count an end-of-blockquote cursor as
        // "inside").
        let src = "> hi\n\nbody";
        let spec = render_with_cursor(src, src.len());
        let blocks: Vec<_> = spec
            .blocks
            .iter()
            .filter(|b| !b.containers.is_empty())
            .collect();
        assert_eq!(blocks.len(), 1);
        let block = blocks[0];
        assert!(matches!(block.kind, BlockKind::Paragraph));
        assert_eq!(block.containers.len(), 1);
        assert!(matches!(
            block.containers[0],
            Container::BlockQuote {
                cursor_inside: false
            }
        ));
    }

    #[test]
    fn blockquote_hides_marker_when_cursor_outside() {
        // Paragraph at parser-range 2..5 ("hi\n") is extended to 0..5
        // by the renderer so the marker `> ` at 0..2 falls inside its
        // source_range — hidden because the cursor is outside.
        let src = "> hi\n\nplain";
        let spec = render_with_cursor(src, src.len());
        let bq = spec
            .blocks
            .iter()
            .find(|b| !b.containers.is_empty())
            .expect("blockquote leaf");
        assert!(bq.has_hidden_range(0..2));
        assert_eq!(bq.source_range.start, 0);
    }

    #[test]
    fn blockquote_marker_overlays_when_cursor_inside() {
        // Markers are *always* hidden from the shaped line so content
        // doesn't shift horizontally between focus / blur. When the
        // cursor is inside the construct, the marker is also recorded
        // as a `MarkerOverlay` at this blockquote's level so the
        // element layer can paint the `>` glyph on top of the border
        // bar.
        let src = "> hi\nbody";
        // Cursor inside "hi" at byte 3.
        let spec = render_with_cursor(src, 3);
        let bq = spec
            .blocks
            .iter()
            .find(|b| !b.containers.is_empty())
            .expect("blockquote leaf");
        assert!(bq.has_hidden_range(0..2));
        assert!(bq.has_marker_overlay(0..2, 0));
        assert!(matches!(
            bq.containers[0],
            Container::BlockQuote {
                cursor_inside: true
            }
        ));
    }

    #[test]
    fn blockquote_with_two_lines_attaches_both_markers_to_one_paragraph() {
        // Two soft-broken lines parse as one paragraph; both line
        // markers should attach to it.
        let src = "> first\n> second\n";
        let spec = render_with_cursor(src, 0);
        let bq = spec
            .blocks
            .iter()
            .find(|b| !b.containers.is_empty())
            .expect("blockquote leaf");
        // Cursor at 0 — boundary equality treats this as "inside",
        // so markers always hide from the line and overlay onto the
        // bar.
        assert!(bq.has_hidden_range(0..2));
        assert!(bq.has_hidden_range(8..10));
        assert!(bq.has_marker_overlay(0..2, 0));
        assert!(bq.has_marker_overlay(8..10, 0));
        assert_eq!(bq.source_range.start, 0);
    }

    #[test]
    fn nested_blockquote_emits_two_containers() {
        let src = "> > deep\n\nbody";
        let spec = render_with_cursor(src, src.len());
        let bq = spec
            .blocks
            .iter()
            .find(|b| !b.containers.is_empty())
            .expect("blockquote leaf");
        assert_eq!(bq.containers.len(), 2);
        // Both levels are blockquotes (not just one nested under
        // another category).
        assert!(
            bq.containers
                .iter()
                .all(|c| matches!(c, Container::BlockQuote { .. }))
        );
        // Both markers attach to the same paragraph leaf — outer
        // marker at 0..2, inner marker at 2..4.
        assert!(bq.has_hidden_range(0..2));
        assert!(bq.has_hidden_range(2..4));
        assert_eq!(bq.source_range.start, 0);
    }

    #[test]
    fn nested_blockquote_overlays_each_level_when_cursor_inside() {
        // Two-deep nest with cursor inside both levels: outer marker
        // (level 0) and inner marker (level 1) each become an overlay
        // at their own bar's column. There is no positional ambiguity
        // in `> > deep` — any cursor inside the source range is
        // inside both nested blockquotes.
        let src = "> > deep\n";
        let spec = render_with_cursor(src, 6);
        let bq = spec
            .blocks
            .iter()
            .find(|b| !b.containers.is_empty())
            .expect("blockquote leaf");
        assert!(bq.has_hidden_range(0..2));
        assert!(bq.has_hidden_range(2..4));
        assert!(bq.has_marker_overlay(0..2, 0));
        assert!(bq.has_marker_overlay(2..4, 1));
    }

    #[test]
    fn blockquote_around_heading_keeps_heading_kind() {
        // Cursor must be truly outside (not at the trailing-`\n`
        // boundary) so the heading's `# ` delimiter hides rather than
        // dims.
        let src = "> # title\n\nbody";
        let spec = render_with_cursor(src, src.len());
        let bq = spec
            .blocks
            .iter()
            .find(|b| !b.containers.is_empty())
            .expect("blockquote leaf");
        assert!(matches!(bq.kind, BlockKind::Heading { level: 1 }));
        assert!(bq.has_hidden_range(2..4)); // "# "
    }

    #[test]
    fn unpaired_trailing_marker_renders_as_blockquote() {
        // `paragraph\n\n> ` parses as a paragraph followed by a
        // blockquote whose only marker is the lone trailing `> ` —
        // no second-of-pair partner. The synthetic must still be
        // emitted with `containers = [BlockQuote { … }]` so the
        // element layer paints the bar immediately, instead of
        // waiting for the user to type a character. Regression
        // against a bug where the deferred prefix was dropped on
        // loop exit and the trailing line rendered as a plain
        // paragraph.
        let src = "paragraph\n\n> ";
        let spec = render_with_cursor(src, src.len());
        let bq_leaves: Vec<_> = spec
            .blocks
            .iter()
            .filter(|b| !b.containers.is_empty())
            .collect();
        assert_eq!(bq_leaves.len(), 1, "expected one BQ leaf for the lone `> `");
        let synth = bq_leaves[0];
        assert!(matches!(synth.kind, BlockKind::Paragraph));
        assert!(matches!(synth.containers[0], Container::BlockQuote { .. }));
        assert!(synth.source_range.contains(&11));
    }

    #[test]
    fn trailing_pair_emits_one_synthetic_for_cursor() {
        // The state right after pressing Enter inside `> hi`: the
        // trailing 6-byte pair `\n> \n> ` collapses to *one*
        // paragraph_gap visually — no row for the middle marker
        // line — but a synthetic leaf is still needed at the
        // second-of-pair so the cursor has somewhere to land.
        let src = "> hi\n> \n> ";
        let spec = render_with_cursor(src, src.len());
        let bq_leaves: Vec<_> = spec
            .blocks
            .iter()
            .filter(|b| !b.containers.is_empty())
            .collect();
        // Real para + 1 synthetic trailing leaf = 2 leaves.
        assert_eq!(bq_leaves.len(), 2);
        assert!(
            bq_leaves.iter().all(|b| b.containers.len() == 1
                && matches!(b.containers[0], Container::BlockQuote { .. }))
        );
        // The trailing synthetic ends at the buffer's end so a
        // boundary cursor lands on it.
        let trailing = bq_leaves.last().unwrap();
        assert_eq!(trailing.source_range.end, src.len());
    }

    #[test]
    fn middle_of_pair_marker_does_not_render_a_row() {
        // The middle marker line of a structural pair (the byte run
        // between the pair's two `\n`s) collapses to whitespace — no
        // leaf claims those bytes, so the element layer has no row to
        // paint there.
        let src = "> hi\n> \n> ";
        let spec = render_with_cursor(src, src.len());
        // Bytes 5..7 are the *middle* marker line `> ` (between the
        // pair's two `\n`s at bytes 4 and 7). No leaf should contain
        // byte 5.
        let claims = spec
            .blocks
            .iter()
            .filter(|b| b.source_range.start <= 5 && 5 < b.source_range.end)
            .count();
        assert_eq!(
            claims, 0,
            "middle-of-pair marker bytes must not be inside any rendered leaf",
        );
    }

    #[test]
    fn trailing_synthetic_overlays_its_marker_when_cursor_inside() {
        // Cursor on the trailing synthetic — its marker overlays
        // onto the bar. The middle marker line has no rendered row,
        // so its marker has nothing to overlay against.
        let src = "> hi\n> \n> ";
        let spec = render_with_cursor(src, src.len());
        let trailing = spec
            .blocks
            .iter()
            .rfind(|b| !b.containers.is_empty())
            .expect("trailing synthetic");
        assert!(
            !trailing.marker_overlays.is_empty(),
            "trailing synthetic must carry an overlay marker",
        );
    }

    #[test]
    fn nested_blockquote_synthetic_carries_outer_marker_too() {
        // Depth-2 trailing state. The outer blockquote's distribution
        // pass should attach its marker to the synthetic the inner
        // pass emitted, so each empty marker line carries *both*
        // levels' markers.
        let src = "> > hi\n> > \n> > ";
        let spec = render_with_cursor(src, src.len());
        let last = spec
            .blocks
            .iter()
            .rfind(|b| !b.containers.is_empty())
            .expect("trailing leaf");
        assert_eq!(last.containers.len(), 2);
        // The trailing line spans bytes 12..16 (`> > `). The outer
        // marker is at 12..14 (level 0), the inner at 14..16 (level
        // 1) — both hidden from the line and recorded as overlays
        // since the cursor at end-of-doc sits on both constructs'
        // boundaries (treated as inside).
        assert!(last.has_hidden_range(12..14));
        assert!(last.has_hidden_range(14..16));
        assert!(last.has_marker_overlay(12..14, 0));
        assert!(last.has_marker_overlay(14..16, 1));
        assert_eq!(last.source_range.start, 12);
    }

    #[test]
    fn blockquote_around_heading_pads_marker_when_cursor_outside() {
        let src = "> # title\n\nbody";
        let spec = render_with_cursor(src, src.len());
        let bq = spec
            .blocks
            .iter()
            .find(|b| !b.containers.is_empty())
            .expect("blockquote leaf");
        // `> ` at 0..2 hides because the cursor is outside the
        // blockquote (in "body").
        assert!(bq.has_hidden_range(0..2));
    }

    #[test]
    fn cursor_inside_an_empty_paragraph_block_lands_on_its_range() {
        // For `p1\n\n\n\np2` (4 `\n`s mid-content) the empty paragraph
        // sits at offset 1 inside the gap: with p1 trimmed to 0..2, the
        // gap is [2, 6) and the empty's range is [3, 5).
        let src = "p1\n\n\n\np2";
        let spec = render_with_cursor(src, 0);
        let empties = synthetic_empties(&spec, src);
        assert_eq!(empties.len(), 1);
        assert_eq!(empties[0].source_range, 3..5);
    }

    // ---- Lists -----------------------------------------------------------

    #[test]
    fn unordered_list_emits_one_paragraph_leaf_per_item() {
        let src = "- foo\n- bar\n";
        let spec = render_with_cursor(src, 0);
        let items: Vec<_> = spec
            .blocks
            .iter()
            .filter(|b| !b.containers.is_empty())
            .collect();
        assert_eq!(items.len(), 2);
        for b in &items {
            assert_eq!(b.containers.len(), 1);
            assert!(matches!(
                b.containers[0],
                Container::ListItem {
                    kind: ListItemKind::Unordered(b'-'),
                    ..
                }
            ));
            assert!(matches!(b.kind, BlockKind::Paragraph));
        }
    }

    #[test]
    fn ordered_list_assigns_per_item_numbers() {
        let src = "1. one\n2. two\n3. three\n";
        let spec = render_with_cursor(src, 0);
        let nums: Vec<u64> = spec
            .blocks
            .iter()
            .filter_map(|b| match b.containers.first() {
                Some(Container::ListItem {
                    kind: ListItemKind::Ordered { number },
                    ..
                }) => Some(*number),
                _ => None,
            })
            .collect();
        assert_eq!(nums, vec![1, 2, 3]);
    }

    #[test]
    fn ordered_list_starting_at_ten_preserves_offset() {
        // The list's start number is whatever pulldown reports;
        // per-item numbers count up from there.
        let src = "10. ten\n11. eleven\n";
        let spec = render_with_cursor(src, 0);
        let nums: Vec<u64> = spec
            .blocks
            .iter()
            .filter_map(|b| match b.containers.first() {
                Some(Container::ListItem {
                    kind: ListItemKind::Ordered { number },
                    ..
                }) => Some(*number),
                _ => None,
            })
            .collect();
        assert_eq!(nums, vec![10, 11]);
    }

    #[test]
    fn list_item_marker_is_inside_leaf_source_range() {
        // The leaf's source_range starts at the item's range start —
        // the marker bytes are inside the leaf so future hide / dim
        // / substitute treatments can address them.
        let src = "- foo\n";
        let spec = render_with_cursor(src, 0);
        let item = spec
            .blocks
            .iter()
            .find(|b| !b.containers.is_empty())
            .expect("list item leaf");
        assert_eq!(item.source_range.start, 0);
    }

    #[test]
    fn cursor_inside_list_item_flips_container_flag() {
        let src = "- foo\n- bar\n";
        // Cursor on first item.
        let spec = render_with_cursor(src, 3);
        let first = spec
            .blocks
            .iter()
            .find(|b| !b.containers.is_empty())
            .unwrap();
        assert!(matches!(
            first.containers[0],
            Container::ListItem {
                cursor_inside: true,
                ..
            }
        ));
        // Cursor on second item.
        let spec = render_with_cursor(src, 9);
        let second_inside_count = spec
            .blocks
            .iter()
            .filter_map(|b| match b.containers.first() {
                Some(Container::ListItem { cursor_inside, .. }) => Some(*cursor_inside),
                _ => None,
            })
            .filter(|c| *c)
            .count();
        assert_eq!(second_inside_count, 1);
    }
}
