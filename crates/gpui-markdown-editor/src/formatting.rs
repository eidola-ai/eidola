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
use crate::escapes;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AtomEligibility {
    /// Ordinary text can be formatted over any selected subrange.
    Partial,
    /// Code and math can only be formatted when selected as a whole construct.
    Whole,
    /// A resolved escape/entity is one semantic glyph and formats atomically.
    Atomic,
    /// Formatting has no visual meaning for an image; it is a hard boundary.
    Never,
}

#[derive(Debug, Clone)]
struct Atom {
    range: Range<usize>,
    styled: bool,
    eligibility: AtomEligibility,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct InlineStyles {
    strong: bool,
    emphasis: bool,
    strikethrough: bool,
}

#[derive(Debug, Clone, Copy)]
struct StyleExpectation {
    offset: usize,
    styles: InlineStyles,
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
            .filter(|atom| atom_participates(atom, selected))
            .filter(|atom| atom_has_substantive_overlap(&state.markdown, atom, selected))
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
    let selection = remap_selection(state.selection, &edits);
    let candidate = crate::update::enforce_invariants(EditorState {
        markdown: markdown.clone(),
        selection,
        embeds: state.embeds.clone(),
    });
    // Formatting is validated against the same canonical buffer the editable
    // update pipeline will commit. Conservatively refuse candidates that need
    // an additional invariant rewrite: offset lineage would otherwise be lost,
    // and passes such as marker-space injection can change block semantics.
    if candidate.markdown != markdown {
        return state;
    }

    let next_tree = parse(&candidate.markdown);
    if block_fingerprint(&tree) != block_fingerprint(&next_tree)
        || protected_inline_fingerprint(&state.markdown, &tree)
            != protected_inline_fingerprint(&candidate.markdown, &next_tree)
        || resolved_span_fingerprint(&state.markdown, &tree)
            != resolved_span_fingerprint(&candidate.markdown, &next_tree)
        || embed_fingerprint(&state.markdown, &state.embeds)
            != embed_fingerprint(&candidate.markdown, &state.embeds)
        || !opposite_styles_hold(&state.markdown, &tree, &next_tree, &edits, format)
        || !expectations_hold(&next_tree, &edits, &expectations)
    {
        return state;
    }

    candidate
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
                    push_inline_groups(node, markdown, None, format, out);
                }
                collect_block_children(&node.children, markdown, embeds, format, out);
            }
            NodeKind::Heading { content_range, .. } => {
                push_inline_groups(node, markdown, Some(content_range.clone()), format, out);
                collect_block_children(&node.children, markdown, embeds, format, out);
            }
            NodeKind::TableCell => {
                push_inline_groups(node, markdown, Some(node.range.clone()), format, out);
            }
            NodeKind::ListItem { .. } => {
                // Tight list items carry inline children directly. Nested lists
                // and loose-item paragraphs recurse as separate islands.
                push_inline_groups(node, markdown, None, format, out);
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
    markdown: &str,
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
        let resolved = escapes::scan(markdown.as_bytes(), group_start..group_end, &[]);
        let mut atoms = Vec::new();
        let mut formats = Vec::new();
        for child in group.drain(..) {
            collect_inline(
                child,
                &resolved,
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
    resolved: &[escapes::ResolvedSpan],
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
        NodeKind::Text => collect_text_atoms(node, resolved, styled, context, atoms),
        NodeKind::InlineCode { .. } | NodeKind::InlineMath { .. } => atoms.push(Atom {
            range: node.range.clone(),
            styled,
            eligibility: AtomEligibility::Whole,
            context: context.clone(),
        }),
        NodeKind::Image { .. } => atoms.push(Atom {
            range: node.range.clone(),
            styled,
            eligibility: AtomEligibility::Never,
            context: context.clone(),
        }),
        NodeKind::DisplayMath { .. } => {}
        _ => {
            for child in &node.children {
                collect_inline(child, resolved, styled, format, context, atoms, formats);
            }
        }
    }
    if pushed_context {
        context.pop();
    }
}

fn collect_text_atoms(
    node: &SyntaxNode,
    resolved: &[escapes::ResolvedSpan],
    styled: bool,
    context: &[InlineContext],
    atoms: &mut Vec<Atom>,
) {
    let mut start = node.range.start;
    for span in resolved
        .iter()
        .filter(|span| overlaps(&span.source_range, &node.range))
    {
        // Pulldown can split one raw escape across adjacent Text events. The
        // first event owns the full atomic span; later overlapping events skip
        // the portion already represented by that atom.
        if span.source_range.start < node.range.start {
            start = start.max(span.source_range.end);
            continue;
        }
        if start < span.source_range.start {
            atoms.push(Atom {
                range: start..span.source_range.start,
                styled,
                eligibility: AtomEligibility::Partial,
                context: context.to_vec(),
            });
        }
        atoms.push(Atom {
            range: span.source_range.clone(),
            styled,
            eligibility: AtomEligibility::Atomic,
            context: context.to_vec(),
        });
        start = span.source_range.end;
    }
    if start < node.range.end {
        atoms.push(Atom {
            range: start..node.range.end,
            styled,
            eligibility: AtomEligibility::Partial,
            context: context.to_vec(),
        });
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
                if cursor < atom.range.start || cursor > atom.range.end {
                    continue;
                }
                match atom.eligibility {
                    AtomEligibility::Partial => {
                        if let Some(range) = word_range_at(&state.markdown, &atom.range, cursor) {
                            return vec![(index, range)];
                        }
                    }
                    AtomEligibility::Atomic => return vec![(index, atom.range.clone())],
                    AtomEligibility::Whole | AtomEligibility::Never => {}
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
                .filter(|atom| atom_participates(atom, &selected))
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

fn atom_participates(atom: &Atom, selected: &Range<usize>) -> bool {
    if !overlaps(&atom.range, selected) {
        return false;
    }
    match atom.eligibility {
        AtomEligibility::Partial => true,
        AtomEligibility::Whole | AtomEligibility::Atomic => {
            selected.start <= atom.range.start && selected.end >= atom.range.end
        }
        AtomEligibility::Never => false,
    }
}

fn atom_has_substantive_overlap(markdown: &str, atom: &Atom, selected: &Range<usize>) -> bool {
    let start = atom.range.start.max(selected.start);
    let end = atom.range.end.min(selected.end);
    start < end && markdown[start..end].chars().any(|ch| !ch.is_whitespace())
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
        let mut group_context: Option<Vec<InlineContext>> = None;
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

            // Existing style over an inert/partially-selected atom must survive
            // a neighboring edit, but a new style must never bridge across it.
            let include = match atom.eligibility {
                AtomEligibility::Partial => true,
                AtomEligibility::Whole | AtomEligibility::Atomic => {
                    atom.styled
                        || (desired.start <= atom.range.start && desired.end >= atom.range.end)
                }
                AtomEligibility::Never => atom.styled,
            };
            if !include {
                if let Some(range) = group_range
                    .take()
                    .and_then(|range| trim_whitespace(markdown, range))
                {
                    out.push(range);
                }
                group_context = None;
                continue;
            }

            if group_context
                .as_ref()
                .is_some_and(|context| context.as_slice() != atom.context.as_slice())
                && let Some(range) = group_range
                    .take()
                    .and_then(|range| trim_whitespace(markdown, range))
            {
                out.push(range);
            }
            if group_context
                .as_ref()
                .is_none_or(|context| context.as_slice() != atom.context.as_slice())
            {
                group_context = Some(atom.context.clone());
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
    expectations: &mut Vec<StyleExpectation>,
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

    // Verify every substantive source character in the affected component,
    // including both unselected sides of a partially touched atom. The target
    // style changes only where this command can act; all opposite inline
    // styles must remain exactly as parsed before the edit.
    for atom in &island.atoms {
        if !overlaps(&atom.range, &component) {
            continue;
        }
        let start = atom.range.start;
        let end = atom.range.end;
        let original = styles_for_atom(atom, format);
        let selected_atom = atom_participates(atom, &selected);
        for (relative, ch) in markdown[start..end].char_indices() {
            if ch.is_whitespace() {
                continue;
            }
            let offset = start + relative;
            let mut styles = original;
            if selected_atom && selected.start <= offset && offset < selected.end {
                match format {
                    InlineFormat::Strong => styles.strong = !remove,
                    InlineFormat::Emphasis => styles.emphasis = !remove,
                }
            }
            expectations.push(StyleExpectation { offset, styles });
        }
    }
}

fn styles_for_atom(atom: &Atom, format: InlineFormat) -> InlineStyles {
    let mut styles = InlineStyles::default();
    match format {
        InlineFormat::Strong => styles.strong = atom.styled,
        InlineFormat::Emphasis => styles.emphasis = atom.styled,
    }
    for context in &atom.context {
        match context.kind {
            InlineContextKind::Strong => styles.strong = true,
            InlineContextKind::Emphasis => styles.emphasis = true,
            InlineContextKind::Strikethrough => styles.strikethrough = true,
            InlineContextKind::Link => {}
        }
    }
    styles
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

fn opposite_styles_hold(
    markdown: &str,
    original: &[SyntaxNode],
    candidate: &[SyntaxNode],
    edits: &[Edit],
    target: InlineFormat,
) -> bool {
    fn visit(
        markdown: &str,
        nodes: &[SyntaxNode],
        original: &[SyntaxNode],
        candidate: &[SyntaxNode],
        edits: &[Edit],
        target: InlineFormat,
    ) -> bool {
        for node in nodes {
            if matches!(
                node.kind,
                NodeKind::Text
                    | NodeKind::InlineCode { .. }
                    | NodeKind::InlineMath { .. }
                    | NodeKind::Image { .. }
            ) {
                for (relative, _) in markdown[node.range.clone()].char_indices() {
                    let offset = node.range.start + relative;
                    let Some(before) = inline_styles_at(original, offset, InlineStyles::default())
                    else {
                        return false;
                    };
                    let mapped = map_offset(offset, edits, true);
                    let Some(after) = inline_styles_at(candidate, mapped, InlineStyles::default())
                    else {
                        return false;
                    };
                    let unchanged = match target {
                        InlineFormat::Strong => before.emphasis == after.emphasis,
                        InlineFormat::Emphasis => before.strong == after.strong,
                    } && before.strikethrough == after.strikethrough;
                    if !unchanged {
                        return false;
                    }
                }
            } else if !visit(markdown, &node.children, original, candidate, edits, target) {
                return false;
            }
        }
        true
    }

    visit(markdown, original, original, candidate, edits, target)
}

fn expectations_hold(
    tree: &[SyntaxNode],
    edits: &[Edit],
    expectations: &[StyleExpectation],
) -> bool {
    expectations.iter().all(|expectation| {
        let mapped = map_offset(expectation.offset, edits, true);
        inline_styles_at(tree, mapped, InlineStyles::default()) == Some(expectation.styles)
    })
}

fn inline_styles_at(
    nodes: &[SyntaxNode],
    offset: usize,
    inherited: InlineStyles,
) -> Option<InlineStyles> {
    for node in nodes {
        if offset < node.range.start || offset >= node.range.end {
            continue;
        }
        let mut styles = inherited;
        match node.kind {
            NodeKind::Strong { .. } => styles.strong = true,
            NodeKind::Emphasis { .. } => styles.emphasis = true,
            NodeKind::Strikethrough { .. } => styles.strikethrough = true,
            _ => {}
        }
        if matches!(
            node.kind,
            NodeKind::Text
                | NodeKind::InlineCode { .. }
                | NodeKind::InlineMath { .. }
                | NodeKind::Image { .. }
        ) {
            return Some(styles);
        }
        if let Some(styles) = inline_styles_at(&node.children, offset, styles) {
            return Some(styles);
        }
    }
    None
}

#[cfg(test)]
fn style_at(nodes: &[SyntaxNode], offset: usize, format: InlineFormat, inherited: bool) -> bool {
    let mut styles = InlineStyles::default();
    match format {
        InlineFormat::Strong => styles.strong = inherited,
        InlineFormat::Emphasis => styles.emphasis = inherited,
    }
    inline_styles_at(nodes, offset, styles).is_some_and(|styles| match format {
        InlineFormat::Strong => styles.strong,
        InlineFormat::Emphasis => styles.emphasis,
    })
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

fn resolved_span_fingerprint(markdown: &str, tree: &[SyntaxNode]) -> Vec<(String, String)> {
    let verbatim = formatting_verbatim_ranges(tree);
    escapes::scan(markdown.as_bytes(), 0..markdown.len(), &verbatim)
        .into_iter()
        .map(|span| (markdown[span.source_range].to_string(), span.display))
        .collect()
}

fn formatting_verbatim_ranges(tree: &[SyntaxNode]) -> Vec<Range<usize>> {
    fn walk(nodes: &[SyntaxNode], out: &mut Vec<Range<usize>>) {
        for node in nodes {
            match &node.kind {
                NodeKind::CodeBlock { .. }
                | NodeKind::InlineCode { .. }
                | NodeKind::InlineMath { .. }
                | NodeKind::DisplayMath { .. } => out.push(node.range.clone()),
                NodeKind::Link {
                    delimiter_ranges, ..
                }
                | NodeKind::Image {
                    delimiter_ranges, ..
                } => {
                    if let Some(destination) = delimiter_ranges.get(1) {
                        out.push(destination.clone());
                    }
                }
                _ => {}
            }
            walk(&node.children, out);
        }
    }

    let mut ranges = Vec::new();
    walk(tree, &mut ranges);
    ranges.sort_by_key(|range| range.start);
    let mut merged: Vec<Range<usize>> = Vec::new();
    for range in ranges {
        if let Some(last) = merged.last_mut()
            && range.start <= last.end
        {
            last.end = last.end.max(range.end);
        } else {
            merged.push(range);
        }
    }
    merged
}

fn embed_fingerprint(markdown: &str, embeds: &embed::EmbedMap) -> Vec<(u64, String)> {
    embed::embed_blocks(markdown, embeds)
        .into_iter()
        .map(|block| (block.ordinal, markdown[block.range].to_string()))
        .collect()
}

fn block_fingerprint(nodes: &[SyntaxNode]) -> Vec<String> {
    let mut out = Vec::new();
    fingerprint_walk(nodes, &mut out);
    out
}

fn fingerprint_walk(nodes: &[SyntaxNode], out: &mut Vec<String>) {
    for node in nodes {
        match &node.kind {
            NodeKind::Paragraph => {
                if let Some(url) = sole_image_url(node) {
                    out.push(format!("paragraph:sole-image:{url}"));
                } else {
                    out.push("paragraph".into());
                }
            }
            NodeKind::Heading { level, .. } => out.push(format!("heading:{level}")),
            NodeKind::CodeBlock { .. } => out.push("code".into()),
            NodeKind::BlockQuote { .. } => out.push("blockquote".into()),
            NodeKind::List { kind } => out.push(format!("list:{kind:?}")),
            NodeKind::ListItem { task, .. } => out.push(format!("item:task={task:?}")),
            NodeKind::Table { alignments } => out.push(format!("table:{}", alignments.len())),
            NodeKind::TableRow { is_header } => out.push(format!("row:{is_header}")),
            NodeKind::TableCell => out.push("cell".into()),
            NodeKind::ThematicBreak => out.push("rule".into()),
            _ => {}
        }
        fingerprint_walk(&node.children, out);
    }
}

fn sole_image_url(node: &SyntaxNode) -> Option<&str> {
    let mut image = None;
    for child in &node.children {
        match &child.kind {
            NodeKind::Image { dest_url, .. } if image.is_none() => image = Some(dest_url.as_str()),
            NodeKind::SoftBreak | NodeKind::HardBreak => {}
            _ => return None,
        }
    }
    image
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
            "Alpha **bravo ⟦charlie** delta⟧ echo.",
            InlineFormat::Emphasis,
        );
        assert_eq!(state.markdown, "Alpha **bravo *charlie*** *delta* echo.");
    }

    #[test]
    fn applying_strong_splits_at_an_emphasis_boundary() {
        let state = apply_marked("Alpha *bravo ⟦charlie* delta⟧ echo.", InlineFormat::Strong);
        assert_eq!(state.markdown, "Alpha *bravo **charlie*** **delta** echo.");
    }

    #[test]
    fn partial_removal_preserves_the_opposite_style_without_crossing_it() {
        let state = apply_marked("*alpha **⟦bravo⟧** charlie*", InlineFormat::Emphasis);
        assert_eq!(state.markdown, "*alpha* **bravo** *charlie*");
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
            "Alpha [bravo ⟦charlie](https://example.test) delta⟧ echo.",
            InlineFormat::Strong,
        );
        assert_eq!(
            state.markdown,
            "Alpha [bravo **charlie**](https://example.test) **delta** echo."
        );
    }

    #[test]
    fn nested_link_and_strong_contexts_split_independently() {
        let state = apply_marked(
            "[**bravo ⟦charlie** delta](https://example.test) echo⟧",
            InlineFormat::Emphasis,
        );
        assert_eq!(
            state.markdown,
            "[**bravo *charlie*** *delta*](https://example.test) *echo*"
        );
    }

    #[test]
    fn strikethrough_is_an_inline_boundary_too() {
        let state = apply_marked("~~alpha ⟦bravo~~ charlie⟧", InlineFormat::Emphasis);
        assert_eq!(state.markdown, "~~alpha *bravo*~~ *charlie*");
    }

    #[test]
    fn removing_an_outer_style_from_link_text_keeps_the_link_intact() {
        let state = apply_marked(
            "*alpha [⟦bravo⟧](https://example.test) charlie*",
            InlineFormat::Emphasis,
        );
        assert_eq!(
            state.markdown,
            "*alpha* [bravo](https://example.test) *charlie*"
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

    #[test]
    fn partial_removal_rejects_ambiguous_delimiters_that_change_unselected_style() {
        let state = apply_marked("***a⟦b⟧c***", InlineFormat::Strong);
        assert_eq!(state.markdown, "***abc***");
    }

    #[test]
    fn whitespace_between_styled_spans_does_not_vote_against_removal() {
        let markdown = "**abc** **def**";
        let state = apply(
            markdown,
            Selection::range(0, markdown.len()),
            InlineFormat::Strong,
        );
        assert_eq!(state.markdown, "abc def");
    }

    #[test]
    fn removing_format_cannot_materialize_task_list_syntax() {
        for markdown in ["- **[ ] foo**", "- **[x] foo**"] {
            let state = apply(
                markdown,
                Selection::range(0, markdown.len()),
                InlineFormat::Strong,
            );
            assert_eq!(state.markdown, markdown);
        }
    }

    #[test]
    fn removing_format_cannot_materialize_a_mapped_embed() {
        let markdown = "**{{ embed 1 }}**";
        let state = toggle(
            EditorState {
                markdown: markdown.into(),
                selection: Selection::range(0, markdown.len()),
                embeds: embed::EmbedMap::new([(1, "embedded content".into())]),
            },
            InlineFormat::Strong,
        );
        assert_eq!(state.markdown, markdown);
        assert!(embed::embed_blocks(&state.markdown, &state.embeds).is_empty());
    }

    #[test]
    fn post_format_canonicalization_cannot_materialize_a_list() {
        for markdown in ["**-foo**", "**+foo**"] {
            let state = apply(
                markdown,
                Selection::range(0, markdown.len()),
                InlineFormat::Strong,
            );
            assert_eq!(state.markdown, markdown);
        }
    }

    #[test]
    fn resolved_spans_are_atomic_formatting_units() {
        let caret = apply("&amp;", Selection::Cursor(2), InlineFormat::Strong);
        assert_eq!(caret.markdown, "**&amp;**");

        let whole = apply("&#38;", Selection::range(0, 5), InlineFormat::Emphasis);
        assert_eq!(whole.markdown, "*&#38;*");

        let escaped = apply(r"\#", Selection::Cursor(1), InlineFormat::Strong);
        assert_eq!(escaped.markdown, r"\#");

        let partial_escape = apply(r"\#", Selection::range(1, 2), InlineFormat::Strong);
        assert_eq!(partial_escape.markdown, r"\#");

        let partial = apply("&amp;", Selection::range(1, 4), InlineFormat::Strong);
        assert_eq!(partial.markdown, "&amp;");
    }

    #[test]
    fn target_formatting_cannot_materialize_an_opposite_inline_style() {
        let state = apply("~~a~", Selection::range(0, 1), InlineFormat::Strong);
        assert_eq!(state.markdown, "~~a~");
    }

    #[test]
    fn images_are_inert_boundaries_for_inline_formatting() {
        let image = "![alt](url)";
        let standalone = apply(
            image,
            Selection::range(0, image.len()),
            InlineFormat::Strong,
        );
        assert_eq!(standalone.markdown, image);

        let mixed = "before ![alt](url) after";
        let state = apply(
            mixed,
            Selection::range(0, mixed.len()),
            InlineFormat::Strong,
        );
        assert_eq!(state.markdown, "**before** ![alt](url) **after**");
    }
}
