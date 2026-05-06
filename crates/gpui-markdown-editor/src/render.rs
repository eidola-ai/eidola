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

use crate::render_spec::{BlockKind, Container, InlineRun, InlineStyle, RenderBlock, RenderSpec};
use crate::state::{EditorState, Selection};
use crate::syntax::{NodeKind, SyntaxNode};

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
        // Anything else at top level — nothing to do yet. (Future phases add
        // handling for lists, etc.)
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
    let mut child_chain = containers.to_vec();
    child_chain.push(Container::BlockQuote { cursor_inside });

    let start = out.len();
    for child in &node.children {
        render_node(child, source, cursor, &child_chain, out);
    }

    let bytes = source.as_bytes();
    let mut expect_middle = true;
    for prefix in prefix_ranges {
        if let Some(leaf) = find_leaf_for_prefix(&mut out[start..], prefix) {
            // Pulldown ranges most leaves to start *after* the line's
            // marker; extend so the marker falls inside the leaf and
            // the element layer can hide / dim it.
            if prefix.start < leaf.source_range.start {
                leaf.source_range.start = prefix.start;
            }
            attach_marker(leaf, prefix, cursor_inside);
            expect_middle = true;
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
            // it. The toggle isn't touched — this is content, not a
            // pair half.
            leaf.source_range.end = source_line_end(bytes, prefix.end);
            attach_marker(leaf, prefix, cursor_inside);
        } else if expect_middle {
            // First prefix of a structural pair — the marker line is
            // the collapsed paragraph-break separator. No row, no
            // synthetic. The bytes still exist in source so the
            // forbidden-position rule keeps the cursor out of the
            // pair interior.
            expect_middle = false;
        } else {
            // Second prefix of a pair with no parsed-paragraph
            // partner — emit a synthetic empty leaf so the cursor
            // has a visible row to land on (post-Enter trailing, or
            // an extra empty between two paragraphs). The synthetic
            // shares this BQ's container chain, so an outer
            // blockquote's later distribution attaches *its* marker
            // here too.
            let line_end = source_line_end(bytes, prefix.end);
            let mut synth = RenderBlock::new(prefix.start..line_end, BlockKind::Paragraph);
            synth.containers = child_chain.clone();
            attach_marker(&mut synth, prefix, cursor_inside);
            out.push(synth);
            expect_middle = true;
        }
    }

    // Synthetics are appended in `prefix_ranges` order (source order)
    // but may now sit *after* a parsed leaf that occurs later in
    // source — sort so subsequent passes (outer-blockquote
    // distribution, `inject_empty_paragraphs`, the editor's per-block
    // index) see blocks in source order.
    out[start..].sort_by_key(|b| b.source_range.start);
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

fn attach_marker(leaf: &mut RenderBlock, prefix: &Range<usize>, cursor_inside: bool) {
    if cursor_inside {
        leaf.inlines.push(InlineRun {
            source_range: prefix.clone(),
            style: InlineStyle::dimmed(),
        });
    } else {
        leaf.hidden_ranges.push(prefix.clone());
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
    fn blockquote_dims_marker_when_cursor_inside() {
        let src = "> hi\nbody";
        // Cursor inside "hi" at byte 3.
        let spec = render_with_cursor(src, 3);
        let bq = spec
            .blocks
            .iter()
            .find(|b| !b.containers.is_empty())
            .expect("blockquote leaf");
        assert!(bq.has_dimmed_range(0..2));
        assert!(!bq.has_hidden_range(0..2));
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
        // so markers dim rather than hide.
        assert!(bq.has_dimmed_range(0..2));
        assert!(bq.has_dimmed_range(8..10));
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
    fn nested_blockquote_dims_only_levels_the_cursor_is_inside() {
        // Two-deep nest with cursor inside both levels: both markers
        // dim. There is no positional ambiguity in `> > deep` — any
        // cursor inside the source range is inside both nested
        // blockquotes.
        let src = "> > deep\n";
        let spec = render_with_cursor(src, 6);
        let bq = spec
            .blocks
            .iter()
            .find(|b| !b.containers.is_empty())
            .expect("blockquote leaf");
        assert!(bq.has_dimmed_range(0..2));
        assert!(bq.has_dimmed_range(2..4));
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
    fn trailing_synthetic_dims_its_marker_when_cursor_inside() {
        // Cursor on the trailing synthetic — its marker dims. The
        // middle marker line has no rendered row, so its marker has
        // nothing to dim against.
        let src = "> hi\n> \n> ";
        let spec = render_with_cursor(src, src.len());
        let trailing = spec
            .blocks
            .iter()
            .rfind(|b| !b.containers.is_empty())
            .expect("trailing synthetic");
        assert!(
            trailing.inlines.iter().any(|r| r.style.dimmed),
            "trailing synthetic must carry a dimmed marker run",
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
        // The trailing line spans bytes 12..16 (`> > `). The inner
        // marker is at 14..16 and the outer at 12..14 — both should be
        // recorded as dimmed inline runs (cursor at end-of-doc is on
        // the construct's boundary, treated as inside).
        assert!(last.has_dimmed_range(12..14));
        assert!(last.has_dimmed_range(14..16));
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
}
