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
//! Formulas: pulldown-cmark folds one trailing `\n` into a paragraph's
//! parser range. We *trim* each real block's `source_range` to its
//! content extent before computing gaps — the folded `\n` then lives
//! in the trailing free zone where the user expects to see it (one Enter
//! at end of doc → one trailing empty row).
//!
//! With trimming applied, the formulas reduce to "every `\n` in a free
//! zone is one empty paragraph, except the two `\n`s that structurally
//! separate two real paragraphs":
//!
//! - Leading run of `L` newlines: `L` empty paragraphs.
//! - Inter-block run of `M` newlines: `max(0, M - 2)` empties (the two
//!   structural newlines that *are* the paragraph break in CommonMark).
//! - Trailing run of `T` newlines: `T` empty paragraphs.
//!
//! Each synthetic block's `source_range` is exactly one `\n` byte, so a
//! cursor at that offset hit-tests into it and a click in the empty row
//! lands the cursor at a real source position.

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

    // Leading gap. No structural cost: there's no preceding paragraph
    // for the leading `\n`s to be separating from, so each one is an
    // empty paragraph above the first real block.
    let first_content = real_blocks[0].source_range.start;
    let leading: Vec<usize> = (0..first_content).filter(|&p| bytes[p] == b'\n').collect();
    for &p in &leading {
        out.push(empty_paragraph_block(p));
    }

    // Real blocks, with inter-block empties before each. The first two
    // `\n`s are the paragraph break itself (CommonMark's `\n\n`
    // separator); the rest are empties.
    for (i, block) in real_blocks.iter().enumerate() {
        if i > 0 {
            let prev_end = real_blocks[i - 1].source_range.end;
            let next_start = block.source_range.start;
            let positions: Vec<usize> = (prev_end..next_start)
                .filter(|&p| bytes[p] == b'\n')
                .collect();
            let empties = positions.len().saturating_sub(2);
            // Skip 1 (prev's terminator). Take `empties`. The remaining
            // trailing position is the next's separator.
            for &p in positions.iter().skip(1).take(empties) {
                out.push(empty_paragraph_block(p));
            }
        }
        out.push(block.clone());
    }

    // Trailing gap. Symmetric to leading — no following paragraph to
    // separate from, so every trailing `\n` is an empty paragraph.
    let last_end = real_blocks
        .last()
        .expect("checked non-empty above")
        .source_range
        .end;
    let trailing: Vec<usize> = (last_end..bytes.len())
        .filter(|&p| bytes[p] == b'\n')
        .collect();
    for &p in &trailing {
        out.push(empty_paragraph_block(p));
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

fn empty_paragraph_block(newline_position: usize) -> RenderBlock {
    RenderBlock::new(newline_position..newline_position + 1, BlockKind::Paragraph)
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

    /// Counts how many top-level blocks are "empty" — `Paragraph` kind
    /// with neither inline runs nor hidden ranges. The synthetic blocks
    /// emitted by `inject_empty_paragraphs` are the only blocks in our
    /// minimal grammar that satisfy that.
    fn count_empty_blocks(spec: &RenderSpec) -> usize {
        spec.blocks
            .iter()
            .filter(|b| {
                matches!(b.kind, BlockKind::Paragraph)
                    && b.inlines.is_empty()
                    && b.hidden_ranges.is_empty()
                    && b.source_range.end - b.source_range.start == 1
            })
            .count()
    }

    #[test]
    fn paragraph_break_alone_produces_no_empty_blocks() {
        // `\n\n` between two paragraphs is just the structural separator.
        let spec = render_with_cursor("p1\n\np2", 0);
        assert_eq!(count_empty_blocks(&spec), 0);
        assert_eq!(spec.blocks.len(), 2);
    }

    #[test]
    fn extra_inter_block_newline_emits_one_empty_paragraph() {
        // 3 newlines between content = paragraph break + 1 empty paragraph.
        let spec = render_with_cursor("p1\n\n\np2", 0);
        assert_eq!(count_empty_blocks(&spec), 1);
        assert_eq!(spec.blocks.len(), 3);
    }

    #[test]
    fn user_example_four_newlines_emits_two_empty_paragraphs() {
        // The user's reported case: `paragraph 1\n\n\n\nparagraph 2` (4
        // newlines between content) should render as p1 + two visible
        // empty rows + p2.
        let spec = render_with_cursor("paragraph 1\n\n\n\nparagraph 2", 0);
        assert_eq!(count_empty_blocks(&spec), 2);
        assert_eq!(spec.blocks.len(), 4);
    }

    fn synthetic_empties(spec: &RenderSpec, src: &str) -> Vec<RenderBlock> {
        spec.blocks
            .iter()
            .filter(|b| {
                matches!(b.kind, BlockKind::Paragraph)
                    && b.source_range.end - b.source_range.start == 1
                    && src.as_bytes()[b.source_range.start] == b'\n'
            })
            .cloned()
            .collect()
    }

    #[test]
    fn empty_block_source_ranges_are_each_one_newline_in_the_gap() {
        let src = "p1\n\n\n\np2";
        let spec = render_with_cursor(src, 0);
        let empties = synthetic_empties(&spec, src);
        assert_eq!(empties.len(), 2);
        // Block ranges should be inside the inter-block newline run and
        // each cover exactly one `\n`.
        for b in &empties {
            let r = &b.source_range;
            assert_eq!(r.end - r.start, 1, "empty block range {:?}", r);
            assert_eq!(
                src.as_bytes()[r.start],
                b'\n',
                "empty block at non-newline byte"
            );
        }
        // ...and the byte positions are strictly increasing.
        let positions: Vec<_> = empties.iter().map(|b| b.source_range.start).collect();
        let mut sorted = positions.clone();
        sorted.sort();
        assert_eq!(positions, sorted);
    }

    #[test]
    fn leading_newlines_emit_empty_paragraphs() {
        // Each leading `\n` is one empty paragraph above the first real
        // block. `\n\np1` → 2 leading empties (matches typewriter: 2
        // Enters then content puts the cursor on line 3).
        let spec = render_with_cursor("\n\np1", 0);
        assert_eq!(count_empty_blocks(&spec), 2);
        assert_eq!(spec.blocks.len(), 3);
    }

    #[test]
    fn single_leading_newline_emits_one_empty() {
        // One leading `\n` is one empty paragraph: the user pressed Enter
        // once at the start of an empty doc, then typed content; the
        // empty above is the line they were on before they typed.
        let spec = render_with_cursor("\np1", 0);
        assert_eq!(count_empty_blocks(&spec), 1);
    }

    #[test]
    fn trailing_newlines_emit_empty_paragraphs() {
        // Each trailing `\n` is one empty paragraph below the last real
        // block. Matches typewriter: every Enter past end-of-content
        // leaves the cursor one line further down.
        let spec = render_with_cursor("p1\n\n", 0);
        assert_eq!(count_empty_blocks(&spec), 2);
        let spec = render_with_cursor("p1\n\n\n", 0);
        assert_eq!(count_empty_blocks(&spec), 3);
    }

    #[test]
    fn single_trailing_newline_emits_one_empty_paragraph() {
        // Regression for the reported bug: pressing Enter once at the
        // end of "paragraph 1" used to produce no visible change because
        // pulldown-cmark folds the trailing `\n` into the paragraph's
        // parser range. With trim + the new formula, that `\n` lives in
        // the trailing free zone and renders as one empty paragraph
        // (with the cursor on it after the keystroke).
        let spec = render_with_cursor("paragraph 1\n", 0);
        assert_eq!(count_empty_blocks(&spec), 1);
        // The empty owns the trailing `\n` byte.
        let trailing = spec
            .blocks
            .iter()
            .filter(|b| {
                matches!(b.kind, BlockKind::Paragraph)
                    && b.source_range.start == 11
                    && b.source_range.end == 12
            })
            .count();
        assert_eq!(trailing, 1);
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
        // Headings also trigger the formula.
        let spec = render_with_cursor("# title\n\n\n\nbody", 0);
        // 4 inter-content newlines → 2 empties between heading and body.
        assert_eq!(count_empty_blocks(&spec), 2);
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
        // For `p1\n\n\np2`, the gap has 3 newlines at bytes 2, 3, 4. The
        // formula is `M - 2 = 1` empty paragraph, placed at the middle
        // `\n` (the one *after* the prev's terminator and *before* the
        // next's separator).
        let src = "p1\n\n\np2";
        let spec = render_with_cursor(src, 0);
        let empties = synthetic_empties(&spec, src);
        assert_eq!(empties.len(), 1);
        // The empty owns byte 3 (positions in the gap are [2, 3, 4];
        // skip 1 — the prev's terminator at 2 — take 1 — the empty at 3;
        // the remaining 4 is the next's separator).
        assert_eq!(empties[0].source_range, 3..4);
    }
}
