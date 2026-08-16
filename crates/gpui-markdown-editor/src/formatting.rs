//! Parser-driven inline formatting toggles.
//!
//! A toggle edits semantic inline content, never raw block chrome. The parser
//! divides the document into independent formatting islands (paragraphs,
//! headings, tight-list text, and table cells); selections crossing blocks are
//! handled once per island. Within an island, strong/emphasis, strikethrough,
//! and link ancestry forms an inline-context signature: target delimiters close
//! before that context's chrome and reopen after it instead of crossing another
//! construct's delimiters. Existing target delimiters in the touched connected
//! component are removed and re-emitted around the desired coverage, which is
//! what makes partial apply/remove possible without nesting redundant markers.

use std::ops::Range;

use unicode_segmentation::UnicodeSegmentation;

use crate::embed;
use crate::parser::parse;
use crate::state::{EditorState, Selection};
use crate::syntax::{NodeKind, SyntaxNode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InlineFormat {
    Strong,
    Emphasis,
}

impl InlineFormat {
    fn delimiter(self) -> &'static str {
        match self {
            Self::Strong => "**",
            Self::Emphasis => "*",
        }
    }

    fn matches(self, kind: &NodeKind) -> bool {
        matches!(
            (self, kind),
            (Self::Strong, NodeKind::Strong { .. }) | (Self::Emphasis, NodeKind::Emphasis { .. })
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum InlineContextKind {
    Strong,
    Emphasis,
    Strikethrough,
    Link,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InlineContext {
    kind: InlineContextKind,
    range: Range<usize>,
}

#[derive(Debug, Clone)]
struct Atom {
    range: Range<usize>,
    styled: bool,
    opaque: bool,
    context: Vec<InlineContext>,
}

#[derive(Debug, Clone)]
struct FormatNode {
    range: Range<usize>,
    content: Range<usize>,
    delimiters: Vec<Range<usize>>,
}

#[derive(Debug, Clone)]
struct Island {
    range: Range<usize>,
    atoms: Vec<Atom>,
    formats: Vec<FormatNode>,
}

#[derive(Debug, Clone)]
struct Edit {
    range: Range<usize>,
    replacement: &'static str,
}

pub(crate) fn toggle(state: EditorState, format: InlineFormat) -> EditorState {
    let tree = parse(&state.markdown);
    let mut islands = Vec::new();
    collect_islands(&tree, &state.markdown, &state.embeds, format, &mut islands);

    let selections = selected_ranges(&state, &islands);
    if selections.is_empty() {
        return state;
    }

    // Mixed target/plain content means “apply”; only an entirely styled
    // semantic selection means “remove”. Delimiters and whitespace do not vote.
    let remove = selections.iter().all(|(island_index, selected)| {
        islands[*island_index]
            .atoms
            .iter()
            .filter(|atom| overlaps(&atom.range, selected))
            .all(|atom| atom.styled)
    });

    let mut edits = Vec::new();
    let mut expectations = Vec::new();
    for (island_index, selected) in &selections {
        plan_island(
            &state.markdown,
            &islands[*island_index],
            selected.clone(),
            remove,
            format,
            &mut edits,
            &mut expectations,
        );
    }
    if edits.is_empty() {
        return state;
    }
    edits.sort_by(|a, b| {
        a.range
            .start
            .cmp(&b.range.start)
            .then_with(|| a.range.end.cmp(&b.range.end))
    });
    edits.dedup_by(|a, b| a.range == b.range && a.replacement == b.replacement);
    if edits.windows(2).any(|w| w[0].range.end > w[1].range.start) {
        return state;
    }

    let markdown = apply_edits(&state.markdown, &edits);
    let next_tree = parse(&markdown);
    if block_fingerprint(&tree) != block_fingerprint(&next_tree)
        || protected_inline_fingerprint(&state.markdown, &tree)
            != protected_inline_fingerprint(&markdown, &next_tree)
        || !expectations_hold(&next_tree, &edits, &expectations, format)
    {
        return state;
    }

    let selection = remap_selection(state.selection, &edits);
    EditorState {
        markdown,
        selection,
        embeds: state.embeds,
    }
}

fn collect_islands(
    nodes: &[SyntaxNode],
    markdown: &str,
    embeds: &embed::EmbedMap,
    format: InlineFormat,
    out: &mut Vec<Island>,
) {
    for node in nodes {
        match &node.kind {
            NodeKind::Paragraph => {
                let mapped_embed = embed::embed_blocks(markdown, embeds)
                    .iter()
                    .any(|block| block.range == node.range);
                let sole_display_math = node.children.len() == 1
                    && matches!(node.children[0].kind, NodeKind::DisplayMath { .. });
                if !mapped_embed && !sole_display_math {
                    push_inline_groups(node, None, format, out);
                }
                collect_block_children(&node.children, markdown, embeds, format, out);
            }
            NodeKind::Heading { content_range, .. } => {
                push_inline_groups(node, Some(content_range.clone()), format, out);
                collect_block_children(&node.children, markdown, embeds, format, out);
            }
            NodeKind::TableCell => {
                push_inline_groups(node, Some(node.range.clone()), format, out);
            }
            NodeKind::ListItem { .. } => {
                // Tight list items carry inline children directly. Nested lists
                // and loose-item paragraphs recurse as separate islands.
                push_inline_groups(node, None, format, out);
                collect_block_children(&node.children, markdown, embeds, format, out);
            }
            NodeKind::CodeBlock { .. } | NodeKind::DisplayMath { .. } | NodeKind::ThematicBreak => {
            }
            _ => collect_islands(&node.children, markdown, embeds, format, out),
        }
    }
}

fn collect_block_children(
    nodes: &[SyntaxNode],
    markdown: &str,
    embeds: &embed::EmbedMap,
    format: InlineFormat,
    out: &mut Vec<Island>,
) {
    for child in nodes {
        if is_block_node(&child.kind) {
            collect_islands(std::slice::from_ref(child), markdown, embeds, format, out);
        } else {
            collect_block_children(&child.children, markdown, embeds, format, out);
        }
    }
}

fn is_block_node(kind: &NodeKind) -> bool {
    matches!(
        kind,
        NodeKind::Paragraph
            | NodeKind::Heading { .. }
            | NodeKind::CodeBlock { .. }
            | NodeKind::BlockQuote { .. }
            | NodeKind::List { .. }
            | NodeKind::ListItem { .. }
            | NodeKind::Table { .. }
            | NodeKind::TableRow { .. }
            | NodeKind::TableCell
            | NodeKind::ThematicBreak
    )
}

fn is_inline_node(kind: &NodeKind) -> bool {
    matches!(
        kind,
        NodeKind::Text
            | NodeKind::Strong { .. }
            | NodeKind::Emphasis { .. }
            | NodeKind::Strikethrough { .. }
            | NodeKind::InlineCode { .. }
            | NodeKind::InlineMath { .. }
            | NodeKind::Image { .. }
            | NodeKind::Link { .. }
            | NodeKind::SoftBreak
            | NodeKind::HardBreak
    )
}

fn push_inline_groups(
    node: &SyntaxNode,
    clamp: Option<Range<usize>>,
    format: InlineFormat,
    out: &mut Vec<Island>,
) {
    let mut group: Vec<&SyntaxNode> = Vec::new();
    let flush = |group: &mut Vec<&SyntaxNode>, out: &mut Vec<Island>| {
        if group.is_empty() {
            return;
        }
        let group_start = group
            .iter()
            .map(|child| child.range.start)
            .min()
            .unwrap_or(0);
        let group_end = group.iter().map(|child| child.range.end).max().unwrap_or(0);
        let mut atoms = Vec::new();
        let mut formats = Vec::new();
        for child in group.drain(..) {
            collect_inline(
                child,
                false,
                format,
                &mut Vec::new(),
                &mut atoms,
                &mut formats,
            );
        }
        atoms.retain(|atom| {
            clamp
                .as_ref()
                .is_none_or(|range| overlaps(&atom.range, range))
        });
        if atoms.is_empty() {
            return;
        }
        let mut start = group_start;
        let mut end = group_end;
        if let Some(range) = &clamp {
            start = start.max(range.start);
            end = end.min(range.end);
        }
        if start < end {
            out.push(Island {
                range: start..end,
                atoms,
                formats,
            });
        }
    };

    for child in &node.children {
        if is_inline_node(&child.kind) {
            group.push(child);
        } else {
            flush(&mut group, out);
        }
    }
    flush(&mut group, out);
}

fn collect_inline(
    node: &SyntaxNode,
    inherited: bool,
    format: InlineFormat,
    context: &mut Vec<InlineContext>,
    atoms: &mut Vec<Atom>,
    formats: &mut Vec<FormatNode>,
) {
    let styled = inherited || format.matches(&node.kind);
    let pushed_context = if let Some(entry) = inline_context(node, format) {
        context.push(entry);
        true
    } else {
        false
    };
    if format.matches(&node.kind) {
        let (content, delimiters) = match &node.kind {
            NodeKind::Strong {
                content_range,
                delimiter_ranges,
            }
            | NodeKind::Emphasis {
                content_range,
                delimiter_ranges,
            } => (content_range.clone(), delimiter_ranges.clone()),
            _ => unreachable!(),
        };
        formats.push(FormatNode {
            range: node.range.clone(),
            content,
            delimiters,
        });
    }

    match &node.kind {
        NodeKind::Text => atoms.push(Atom {
            range: node.range.clone(),
            styled,
            opaque: false,
            context: context.clone(),
        }),
        NodeKind::InlineCode { .. } | NodeKind::InlineMath { .. } | NodeKind::Image { .. } => {
            atoms.push(Atom {
                range: node.range.clone(),
                styled,
                opaque: true,
                context: context.clone(),
            });
        }
        NodeKind::DisplayMath { .. } => {}
        _ => {
            for child in &node.children {
                collect_inline(child, styled, format, context, atoms, formats);
            }
        }
    }
    if pushed_context {
        context.pop();
    }
}

fn inline_context(node: &SyntaxNode, format: InlineFormat) -> Option<InlineContext> {
    let kind = match &node.kind {
        NodeKind::Strong { .. } if format != InlineFormat::Strong => InlineContextKind::Strong,
        NodeKind::Emphasis { .. } if format != InlineFormat::Emphasis => {
            InlineContextKind::Emphasis
        }
        NodeKind::Strikethrough { .. } => InlineContextKind::Strikethrough,
        NodeKind::Link { .. } => InlineContextKind::Link,
        _ => return None,
    };
    Some(InlineContext {
        kind,
        range: node.range.clone(),
    })
}

fn selected_ranges(state: &EditorState, islands: &[Island]) -> Vec<(usize, Range<usize>)> {
    if state.selection.is_collapsed() {
        let cursor = state.selection.head();
        for (index, island) in islands.iter().enumerate() {
            for atom in &island.atoms {
                if !atom.opaque && cursor >= atom.range.start && cursor <= atom.range.end {
                    if let Some(range) = word_range_at(&state.markdown, &atom.range, cursor) {
                        return vec![(index, range)];
                    }
                }
            }
        }
        return Vec::new();
    }

    let selected = state.selection.selection_range();
    islands
        .iter()
        .enumerate()
        .filter_map(|(index, island)| {
            let qualified: Vec<&Atom> = island
                .atoms
                .iter()
                .filter(|atom| {
                    overlaps(&atom.range, &selected)
                        && (!atom.opaque
                            || (selected.start <= atom.range.start
                                && selected.end >= atom.range.end))
                })
                .collect();
            let first = qualified.first()?;
            let last = qualified.last()?;
            let start = if selected.start <= island.range.start {
                island.range.start
            } else {
                selected.start.max(first.range.start)
            };
            let end = if selected.end >= island.range.end {
                island.range.end
            } else {
                selected.end.min(last.range.end)
            };
            trim_whitespace(&state.markdown, start..end).map(|range| (index, range))
        })
        .collect()
}

fn word_range_at(markdown: &str, atom: &Range<usize>, cursor: usize) -> Option<Range<usize>> {
    let source = &markdown[atom.clone()];
    let local = cursor.saturating_sub(atom.start).min(source.len());
    let mut previous = None;
    for (offset, segment) in source.split_word_bound_indices() {
        let range = offset..offset + segment.len();
        let is_word = segment.chars().any(char::is_alphanumeric);
        if is_word && range.start <= local && local < range.end {
            return Some(atom.start + range.start..atom.start + range.end);
        }
        if is_word && range.end == local {
            previous = Some(range.clone());
        }
        if range.start > local {
            break;
        }
    }
    previous.map(|range| atom.start + range.start..atom.start + range.end)
}

fn trim_whitespace(markdown: &str, range: Range<usize>) -> Option<Range<usize>> {
    if range.start >= range.end {
        return None;
    }
    let slice = &markdown[range.clone()];
    let leading = slice.len() - slice.trim_start_matches(char::is_whitespace).len();
    let trailing = slice.len() - slice.trim_end_matches(char::is_whitespace).len();
    let start = range.start + leading;
    let end = range.end.saturating_sub(trailing);
    (start < end).then_some(start..end)
}

fn split_at_inline_contexts(
    markdown: &str,
    island: &Island,
    ranges: &[Range<usize>],
) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    for desired in ranges {
        let mut group_context: Option<&[InlineContext]> = None;
        let mut group_range: Option<Range<usize>> = None;
        for atom in island
            .atoms
            .iter()
            .filter(|atom| overlaps(&atom.range, desired))
        {
            let part = atom.range.start.max(desired.start)..atom.range.end.min(desired.end);
            if part.start >= part.end {
                continue;
            }
            if group_context.is_some_and(|context| context != atom.context.as_slice()) {
                if let Some(range) = group_range
                    .take()
                    .and_then(|range| trim_whitespace(markdown, range))
                {
                    out.push(range);
                }
            }
            if group_context.is_none_or(|context| context != atom.context.as_slice()) {
                group_context = Some(atom.context.as_slice());
                group_range = Some(part);
            } else if let Some(range) = &mut group_range {
                range.end = part.end;
            }
        }
        if let Some(range) = group_range.and_then(|range| trim_whitespace(markdown, range)) {
            out.push(range);
        }
    }
    out
}

fn plan_island(
    markdown: &str,
    island: &Island,
    selected: Range<usize>,
    remove: bool,
    format: InlineFormat,
    edits: &mut Vec<Edit>,
    expectations: &mut Vec<(usize, bool)>,
) {
    let mut component = selected.clone();
    let mut members = Vec::new();
    loop {
        let mut changed = false;
        for (index, node) in island.formats.iter().enumerate() {
            if members.contains(&index) {
                continue;
            }
            if touches(&node.content, &component) || overlaps(&node.range, &component) {
                component.start = component.start.min(node.content.start);
                component.end = component.end.max(node.content.end);
                members.push(index);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut coverage: Vec<Range<usize>> = members
        .iter()
        .map(|index| island.formats[*index].content.clone())
        .collect();
    coverage = merge_ranges(coverage);
    let desired = if remove {
        subtract_ranges(&coverage, &selected)
    } else {
        coverage.push(selected.clone());
        merge_ranges(coverage)
    };
    let desired = split_at_inline_contexts(markdown, island, &desired);

    for index in &members {
        for delimiter in &island.formats[*index].delimiters {
            edits.push(Edit {
                range: delimiter.clone(),
                replacement: "",
            });
        }
    }
    for range in &desired {
        edits.push(Edit {
            range: range.start..range.start,
            replacement: format.delimiter(),
        });
        edits.push(Edit {
            range: range.end..range.end,
            replacement: format.delimiter(),
        });
    }

    // Verify every semantic atom in the affected component: selected atoms
    // receive the command's result, while styled content outside a partial
    // removal must remain styled after the component is split.
    for atom in &island.atoms {
        if !overlaps(&atom.range, &component) {
            continue;
        }
        if let Some(sample) = first_substantive_byte(markdown, atom, &selected) {
            let expected = if overlaps(&atom.range, &selected) {
                !remove
            } else {
                atom.styled
            };
            expectations.push((sample, expected));
        }
    }
}

fn first_substantive_byte(markdown: &str, atom: &Atom, selected: &Range<usize>) -> Option<usize> {
    let start = atom.range.start.max(selected.start);
    let end = atom.range.end.min(selected.end);
    if start >= end {
        return None;
    }
    markdown[start..end]
        .char_indices()
        .find(|(_, ch)| !ch.is_whitespace())
        .map(|(offset, _)| start + offset)
}

fn merge_ranges(mut ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    ranges.sort_by_key(|range| range.start);
    let mut out: Vec<Range<usize>> = Vec::new();
    for range in ranges {
        if let Some(last) = out.last_mut()
            && range.start <= last.end
        {
            last.end = last.end.max(range.end);
        } else {
            out.push(range);
        }
    }
    out
}

fn subtract_ranges(ranges: &[Range<usize>], removed: &Range<usize>) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    for range in ranges {
        if !overlaps(range, removed) {
            out.push(range.clone());
            continue;
        }
        if range.start < removed.start {
            out.push(range.start..removed.start.min(range.end));
        }
        if removed.end < range.end {
            out.push(removed.end.max(range.start)..range.end);
        }
    }
    out.retain(|range| range.start < range.end);
    out
}

fn apply_edits(markdown: &str, edits: &[Edit]) -> String {
    let delta: isize = edits
        .iter()
        .map(|edit| edit.replacement.len() as isize - (edit.range.end - edit.range.start) as isize)
        .sum();
    let mut out = String::with_capacity((markdown.len() as isize + delta).max(0) as usize);
    let mut last = 0;
    for edit in edits {
        out.push_str(&markdown[last..edit.range.start]);
        out.push_str(edit.replacement);
        last = edit.range.end;
    }
    out.push_str(&markdown[last..]);
    out
}

fn remap_selection(selection: Selection, edits: &[Edit]) -> Selection {
    match selection {
        Selection::Cursor(offset) => Selection::Cursor(map_offset(offset, edits, true)),
        Selection::Range { anchor, head } => {
            let lo = anchor.min(head);
            let hi = anchor.max(head);
            let mapped_lo = map_offset(lo, edits, true);
            let mapped_hi = map_offset(hi, edits, false);
            if anchor <= head {
                Selection::Range {
                    anchor: mapped_lo,
                    head: mapped_hi,
                }
            } else {
                Selection::Range {
                    anchor: mapped_hi,
                    head: mapped_lo,
                }
            }
        }
    }
}

fn map_offset(offset: usize, edits: &[Edit], after_insertions: bool) -> usize {
    let mut shift = 0isize;
    for edit in edits {
        let insertion_here = edit.range.is_empty() && edit.range.start == offset;
        if edit.range.end < offset
            || (edit.range.end == offset && (!insertion_here || after_insertions))
        {
            shift += edit.replacement.len() as isize - (edit.range.end - edit.range.start) as isize;
        } else if edit.range.start < offset && offset < edit.range.end {
            return ((edit.range.start as isize + shift) + edit.replacement.len() as isize).max(0)
                as usize;
        } else {
            break;
        }
    }
    ((offset as isize) + shift).max(0) as usize
}

fn expectations_hold(
    tree: &[SyntaxNode],
    edits: &[Edit],
    expectations: &[(usize, bool)],
    format: InlineFormat,
) -> bool {
    expectations.iter().all(|(offset, expected)| {
        let mapped = map_offset(*offset, edits, true);
        style_at(tree, mapped, format, false) == *expected
    })
}

fn style_at(nodes: &[SyntaxNode], offset: usize, format: InlineFormat, inherited: bool) -> bool {
    for node in nodes {
        if offset < node.range.start || offset >= node.range.end {
            continue;
        }
        let styled = inherited || format.matches(&node.kind);
        if matches!(
            node.kind,
            NodeKind::Text
                | NodeKind::InlineCode { .. }
                | NodeKind::InlineMath { .. }
                | NodeKind::Image { .. }
        ) {
            return styled;
        }
        if style_at(&node.children, offset, format, styled) {
            return true;
        }
    }
    false
}

fn protected_inline_fingerprint(markdown: &str, nodes: &[SyntaxNode]) -> Vec<String> {
    let mut out = Vec::new();
    protected_inline_walk(markdown, nodes, &mut out);
    out
}

fn protected_inline_walk(markdown: &str, nodes: &[SyntaxNode], out: &mut Vec<String>) {
    for node in nodes {
        match &node.kind {
            NodeKind::Link { dest_url, .. } => out.push(format!("link:{dest_url}")),
            NodeKind::Image { dest_url, .. } => out.push(format!("image:{dest_url}")),
            NodeKind::InlineCode { content_range, .. } => {
                out.push(format!("code:{}", &markdown[content_range.clone()]));
            }
            NodeKind::InlineMath { content_range, .. } => {
                out.push(format!("math:{}", &markdown[content_range.clone()]));
            }
            NodeKind::DisplayMath { content_range, .. } => {
                out.push(format!("display-math:{}", &markdown[content_range.clone()]));
            }
            NodeKind::CodeBlock {
                content_range,
                lang,
                ..
            } => out.push(format!(
                "code-block:{lang:?}:{}",
                &markdown[content_range.clone()]
            )),
            _ => {}
        }
        protected_inline_walk(markdown, &node.children, out);
    }
}

fn block_fingerprint(nodes: &[SyntaxNode]) -> Vec<String> {
    let mut out = Vec::new();
    fingerprint_walk(nodes, &mut out);
    out
}

fn fingerprint_walk(nodes: &[SyntaxNode], out: &mut Vec<String>) {
    for node in nodes {
        match &node.kind {
            NodeKind::Paragraph => out.push("paragraph".into()),
            NodeKind::Heading { level, .. } => out.push(format!("heading:{level}")),
            NodeKind::CodeBlock { .. } => out.push("code".into()),
            NodeKind::BlockQuote { .. } => out.push("blockquote".into()),
            NodeKind::List { kind } => out.push(format!("list:{kind:?}")),
            NodeKind::ListItem { .. } => out.push("item".into()),
            NodeKind::Table { alignments } => out.push(format!("table:{}", alignments.len())),
            NodeKind::TableRow { is_header } => out.push(format!("row:{is_header}")),
            NodeKind::TableCell => out.push("cell".into()),
            NodeKind::ThematicBreak => out.push("rule".into()),
            _ => {}
        }
        fingerprint_walk(&node.children, out);
    }
}

fn overlaps(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

fn touches(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.start <= b.end && b.start <= a.end
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(markdown: &str, selection: Selection, format: InlineFormat) -> EditorState {
        toggle(
            EditorState {
                markdown: markdown.into(),
                selection,
                ..Default::default()
            },
            format,
        )
    }

    #[test]
    fn collapsed_caret_toggles_the_word_and_stays_collapsed() {
        let state = apply("one two", Selection::Cursor(5), InlineFormat::Strong);
        assert_eq!(state.markdown, "one **two**");
        assert_eq!(state.selection, Selection::Cursor(7));
    }

    #[test]
    fn whole_and_partial_target_spans_are_removed_semantically() {
        let whole = apply("**one two**", Selection::range(0, 11), InlineFormat::Strong);
        assert_eq!(whole.markdown, "one two");

        let middle = apply("**abcdef**", Selection::range(4, 6), InlineFormat::Strong);
        assert_eq!(middle.markdown, "**ab**cd**ef**");
    }

    #[test]
    fn mixed_selection_extends_and_merges_the_existing_style() {
        let state = apply("**abc** def", Selection::range(3, 11), InlineFormat::Strong);
        assert_eq!(state.markdown, "**abc def**");
    }

    fn apply_marked(markdown: &str, format: InlineFormat) -> EditorState {
        let start = markdown.find('⟦').expect("selection start");
        let end = markdown.find('⟧').expect("selection end");
        let mut plain = markdown.to_string();
        plain.remove(end);
        plain.remove(start);
        apply(
            &plain,
            Selection::range(start, end - '⟦'.len_utf8()),
            format,
        )
    }

    #[test]
    fn opposite_inline_style_is_preserved_and_forms_its_own_context() {
        let state = apply("*one* two", Selection::range(0, 9), InlineFormat::Strong);
        assert_eq!(state.markdown, "***one*** **two**");
        let tree = parse(&state.markdown);
        assert!(style_at(&tree, 3, InlineFormat::Strong, false));
        assert!(style_at(&tree, 3, InlineFormat::Emphasis, false));
    }

    #[test]
    fn applying_italic_splits_at_a_strong_boundary() {
        let state = apply_marked(
            "Alpha **bravo ⟦charley** delta⟧ echo.",
            InlineFormat::Emphasis,
        );
        assert_eq!(state.markdown, "Alpha **bravo *charley*** *delta* echo.");
    }

    #[test]
    fn applying_strong_splits_at_an_emphasis_boundary() {
        let state = apply_marked("Alpha *bravo ⟦charley* delta⟧ echo.", InlineFormat::Strong);
        assert_eq!(state.markdown, "Alpha *bravo **charley*** **delta** echo.");
    }

    #[test]
    fn partial_removal_preserves_the_opposite_style_without_crossing_it() {
        let state = apply_marked("*alpha **⟦bravo⟧** charley*", InlineFormat::Emphasis);
        assert_eq!(state.markdown, "*alpha* **bravo** *charley*");
    }

    #[test]
    fn block_boundaries_are_split_without_touching_their_chrome() {
        let markdown = "# head\n\n- one\n- two\n\n| a | b |\n| --- | --- |\n| c | d |";
        let state = apply(
            markdown,
            Selection::range(0, markdown.len()),
            InlineFormat::Strong,
        );
        assert_eq!(
            state.markdown,
            "# **head**\n\n- **one**\n- **two**\n\n| **a** | **b** |\n| --- | --- |\n| **c** | **d** |"
        );
    }

    #[test]
    fn links_keep_formatting_inside_their_text_context() {
        let markdown = "[one](https://example.test) two";
        let state = apply(
            markdown,
            Selection::range(0, markdown.len()),
            InlineFormat::Strong,
        );
        assert_eq!(state.markdown, "[**one**](https://example.test) **two**");
    }

    #[test]
    fn a_selection_leaving_a_link_splits_at_the_link_boundary() {
        let state = apply_marked(
            "Alpha [bravo ⟦charley](https://example.test) delta⟧ echo.",
            InlineFormat::Strong,
        );
        assert_eq!(
            state.markdown,
            "Alpha [bravo **charley**](https://example.test) **delta** echo."
        );
    }

    #[test]
    fn nested_link_and_strong_contexts_split_independently() {
        let state = apply_marked(
            "[**bravo ⟦charley** delta](https://example.test) echo⟧",
            InlineFormat::Emphasis,
        );
        assert_eq!(
            state.markdown,
            "[**bravo *charley*** *delta*](https://example.test) *echo*"
        );
    }

    #[test]
    fn strikethrough_is_an_inline_boundary_too() {
        let state = apply_marked("~~alpha ⟦bravo~~ charley⟧", InlineFormat::Emphasis);
        assert_eq!(state.markdown, "~~alpha *bravo*~~ *charley*");
    }

    #[test]
    fn removing_an_outer_style_from_link_text_keeps_the_link_intact() {
        let state = apply_marked(
            "*alpha [⟦bravo⟧](https://example.test) charley*",
            InlineFormat::Emphasis,
        );
        assert_eq!(
            state.markdown,
            "*alpha* [bravo](https://example.test) *charley*"
        );
    }

    #[test]
    fn opaque_inline_atoms_are_only_formatted_as_whole_constructs() {
        let partial = apply("`code`", Selection::range(2, 4), InlineFormat::Strong);
        assert_eq!(partial.markdown, "`code`");

        let whole = apply("`code`", Selection::range(0, 6), InlineFormat::Strong);
        assert_eq!(whole.markdown, "**`code`**");
    }

    #[test]
    fn fences_and_display_math_are_inert_while_surrounding_prose_formats() {
        let markdown = "before\n\n```\ncode\n```\n\n$$\nmath\n$$\n\nafter";
        let state = apply(
            markdown,
            Selection::range(0, markdown.len()),
            InlineFormat::Emphasis,
        );
        assert_eq!(
            state.markdown,
            "*before*\n\n```\ncode\n```\n\n$$\nmath\n$$\n\n*after*"
        );
    }

    #[test]
    fn removing_format_that_would_create_a_list_is_refused() {
        let state = apply("**- item**", Selection::range(0, 10), InlineFormat::Strong);
        assert_eq!(state.markdown, "**- item**");
    }
}
