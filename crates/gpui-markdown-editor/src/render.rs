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

use std::ops::Range;

use crate::render_spec::{BlockKind, InlineRun, InlineStyle, RenderBlock, RenderSpec};
use crate::state::{EditorState, Selection};
use crate::syntax::{NodeKind, SyntaxNode};

pub fn render(state: &EditorState, tree: &[SyntaxNode]) -> RenderSpec {
    let cursor = CursorRange::from(&state.selection);
    let mut blocks = Vec::new();
    for node in tree {
        render_node(node, cursor, &mut blocks);
    }
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
    fn empty_document_yields_empty_spec() {
        let spec = render_with_cursor("", 0);
        assert!(spec.blocks.is_empty());
    }

    #[test]
    fn paragraph_with_no_inline_constructs_has_no_inline_runs() {
        let spec = render_with_cursor("plain text", 999);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert!(block.inlines.is_empty());
        assert!(block.hidden_ranges.is_empty());
    }
}
