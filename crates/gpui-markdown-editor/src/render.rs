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

use crate::render_spec::{BlockKind, InlineRun, InlineStyle, RenderBlock, RenderSpec};
use crate::state::{EditorState, Selection};
use crate::syntax::{NodeKind, SyntaxNode};

pub fn render(state: &EditorState, tree: &[SyntaxNode]) -> RenderSpec {
    let cursor = CursorRange::from(&state.selection);
    let mut real_blocks = Vec::new();
    for node in tree {
        render_node(node, cursor, &mut real_blocks);
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

fn render_node(node: &SyntaxNode, cursor: CursorRange, out: &mut Vec<RenderBlock>) {
    match &node.kind {
        NodeKind::Paragraph => render_paragraph(node, cursor, out),
        NodeKind::Heading { .. } => render_heading(node, cursor, out),
        // Anything else at top level — nothing to do yet. (Future phases add
        // handling for lists, blockquotes, code blocks, etc.)
        _ => {}
    }
}

fn render_paragraph(node: &SyntaxNode, cursor: CursorRange, out: &mut Vec<RenderBlock>) {
    let mut block = RenderBlock::new(node.range.clone(), BlockKind::Paragraph);
    collect_inlines(node, cursor, &mut block);
    out.push(block);
}

fn render_heading(node: &SyntaxNode, cursor: CursorRange, out: &mut Vec<RenderBlock>) {
    let (level, content_range, delimiter_ranges) = match &node.kind {
        NodeKind::Heading {
            level,
            content_range,
            delimiter_ranges,
        } => (*level, content_range.clone(), delimiter_ranges.clone()),
        _ => return,
    };

    let mut block = RenderBlock::new(node.range.clone(), BlockKind::Heading { level });
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
    // assume there's at least one parsed block to anchor empties around;
    // here there isn't. Two things go wrong if we just return an empty
    // `Vec`:
    //
    //   1. No `BlockElement::paint` runs for this frame, so no
    //      `window.handle_input` registers the `EntityInputHandler` and
    //      typed text has nowhere to route.
    //   2. There's no shaped line to paint a cursor against, so the user
    //      can't see where they are and can't click to place the cursor.
    //
    // Emit one synthetic block per *line* in the source (lines bounded
    // by `\n`, plus one trailing line after the last `\n`) so every
    // byte position from 0 to `len` has a block to anchor against and
    // the cursor follows typewriter intuition: pressing Enter `N` times
    // in an empty doc shows `N + 1` visible rows.
    if real_blocks.is_empty() {
        let mut out = Vec::with_capacity(bytes.len() + 1);
        let mut line_start = 0;
        for (p, &b) in bytes.iter().enumerate() {
            if b == b'\n' {
                out.push(RenderBlock::new(line_start..p + 1, BlockKind::Paragraph));
                line_start = p + 1;
            }
        }
        // Trailing line (after the last `\n`, or the only line if there
        // were no `\n`s at all).
        out.push(RenderBlock::new(
            line_start..bytes.len(),
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

    // Trailing gap. Each pair is one trailing empty paragraph.
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
        let start = last_end + 2 * i;
        out.push(empty_paragraph_pair(start));
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
    /// produced by `inject_empty_paragraphs`. In the pairs model each
    /// such block spans exactly two `\n` bytes (we check the bytes
    /// themselves, not just the range size, so a real 2-char paragraph
    /// like `"p1"` doesn't get miscounted).
    fn count_empty_blocks(spec: &RenderSpec, src: &str) -> usize {
        let bytes = src.as_bytes();
        spec.blocks
            .iter()
            .filter(|b| {
                matches!(b.kind, BlockKind::Paragraph)
                    && b.source_range.end - b.source_range.start == 2
                    && (b.source_range.start..b.source_range.end)
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
        // visible empty trailing row.
        let src = "paragraph 1\n\n";
        let spec = render_with_cursor(src, 13);
        assert_eq!(count_empty_blocks(&spec, src), 1);
        let trailing = spec
            .blocks
            .iter()
            .find(|b| b.source_range == (11..13))
            .expect("synthetic empty owning the trailing pair");
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
    fn doc_of_only_newlines_emits_one_block_per_line() {
        // For a content-empty doc we don't have a parsed block to anchor
        // empties around, so the formula doesn't apply — emit one block
        // per line so every byte position has a cursor anchor and the
        // visual count matches typewriter intuition.
        //
        // `\n\n\n` has 3 `\n`s = 4 lines. Three of those lines are bounded
        // by a `\n` (ranges 0..1, 1..2, 2..3); the fourth is the trailing
        // line after the last `\n` (range 3..3, zero bytes).
        let spec = render_with_cursor("\n\n\n", 0);
        assert_eq!(spec.blocks.len(), 4);
        let ranges: Vec<_> = spec.blocks.iter().map(|b| b.source_range.clone()).collect();
        assert_eq!(ranges, vec![0..1, 1..2, 2..3, 3..3]);
        assert!(
            spec.blocks
                .iter()
                .all(|b| matches!(b.kind, BlockKind::Paragraph))
        );
    }

    #[test]
    fn single_newline_emits_two_blocks() {
        // Source `\n` is "Enter once in an empty doc" — typewriter
        // intuition says the user is now on line 2 with line 1 empty
        // above. Both byte positions (0 and 1) need cursor anchors.
        let spec = render_with_cursor("\n", 1);
        assert_eq!(spec.blocks.len(), 2);
        assert_eq!(spec.blocks[0].source_range, 0..1);
        assert_eq!(spec.blocks[1].source_range, 1..1);
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
