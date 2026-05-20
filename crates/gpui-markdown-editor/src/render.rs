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

/// Build a [`RenderSpec`] for the editor's current state.
///
/// The render walker is structured as a **pipeline**, not a tree walk —
/// the recursive walk produces a flat `Vec<RenderBlock>`, and several
/// post-passes run in a specific order to refine the spec. The order
/// is load-bearing; reorder only with care. New passes that fix
/// follow-on bugs should slot into this list with a clear rationale.
///
/// ```text
/// 1. recursive walk (`render_node` → render_paragraph / render_blockquote /
///    render_list / render_list_item / render_code_block / render_heading).
///    Emits one leaf per parsed paragraph / heading / code block, with
///    per-leaf hidden ranges for own-container chrome (LI marker, BQ
///    prefix) but NOT for chain-aware alternation hiding.
///
/// 2. inject_empty_paragraphs — emit synthetic empty Paragraph leaves
///    for trailing positions and inter-block paragraph breaks that
///    pulldown didn't claim (post-Enter transient, end-of-buffer
///    cursor row, etc.). Each synth's chain comes from
///    `chain_for_position` — the same chain query the cursor walker
///    uses, so render and analysis agree.
///
/// 3. merge_hard_break_continuations — when pulldown's parse splits at
///    a `  \n` hard break followed by a trailing line of pure
///    chain-continuation prefix (no further content), the recursive
///    walk + inject_empty_paragraphs produce two adjacent Paragraph
///    blocks separated by paragraph_gap. The same source with one
///    extra byte of content parses as ONE paragraph (hard-break
///    continuation). Merge the split case so the visual matches.
///
/// 4. hide_chain_continuation_prefix (per-block) — the per-item and
///    per-BQ hides done by the recursive walk catch their own
///    contributions, but miss bytes that sit *between* container
///    kinds in alternating chains like `[LI, BQ, LI]` (e.g. the
///    trailing LI continuation indent after the last BQ marker). One
///    chain-driven pass at the end catches all alternations
///    uniformly. Uses `chain_continuation_prefix(chain)` to compare
///    bytes line-by-line against the canonical prefix; matching
///    spans are added to `hidden_ranges`.
///
/// 5. merge_hidden_ranges (per-block) — multiple hide passes (own
///    container, alternating chain, code-block fence, …) can produce
///    overlapping or duplicate entries. Normalize each block's
///    `hidden_ranges` into a sorted, non-overlapping list so the
///    element layer's shaping doesn't pay for duplicate work.
/// ```
///
/// **Invariant.** Every chain-aware decision in the pipeline goes
/// through `analysis::enclosing_containers_at` /
/// `analysis::chain_continuation_prefix` / `analysis::chain_pair_shape`.
/// Reaching for raw `\n` boundaries or hand-built `"> "` strings is a
/// bug; we've fixed several of those by migrating to the canonical
/// helpers.
pub fn render(state: &EditorState, tree: &[SyntaxNode]) -> RenderSpec {
    let cursor = CursorRange::from(&state.selection);
    let mut real_blocks = Vec::new();
    for node in tree {
        render_node(node, tree, &state.markdown, cursor, &[], &mut real_blocks);
    }
    let mut blocks = inject_empty_paragraphs(&state.markdown, tree, cursor, real_blocks);
    let bytes = state.markdown.as_bytes();
    merge_hard_break_continuations(&mut blocks, bytes);
    let verbatim = collect_verbatim_ranges(tree);
    for block in &mut blocks {
        hide_chain_continuation_prefix(block, bytes);
        apply_escapes_and_entities(block, &state.markdown, &verbatim, cursor);
        merge_hidden_ranges(&mut block.hidden_ranges);
    }
    RenderSpec { blocks }
}

/// Collect every byte range in which CommonMark §2.4 backslash escapes
/// and §2.5 entity references **do not** apply: fenced/indented code
/// blocks, inline code spans, and link destinations (the URL portion
/// inside `](url)`). Output is sorted by `start` and non-overlapping
/// — `escapes::scan` skips past these in a single pass.
///
/// Autolinks and raw HTML are also verbatim contexts in the spec; we
/// don't model them as first-class constructs yet, so they fall through
/// to the scanner. That's a minor visual issue (e.g. an autolink URL
/// containing `&copy;` would show `©` instead of `&copy;`), which we
/// can revisit when those constructs land.
fn collect_verbatim_ranges(tree: &[SyntaxNode]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    fn walk(node: &SyntaxNode, out: &mut Vec<Range<usize>>) {
        match &node.kind {
            NodeKind::CodeBlock { .. } => {
                out.push(node.range.clone());
            }
            NodeKind::InlineCode { .. } => {
                out.push(node.range.clone());
            }
            // Math content uses LaTeX-level escapes (`\$`, `\\`,
            // `\frac`, …), not CommonMark §2.4 escapes. Treating the
            // entire `$...$` / `$$...$$` construct as verbatim
            // prevents the markdown scanner from seeing `\$` and
            // re-rendering it as a literal `$` (which would then
            // collide with math-delimiter parsing in subtle ways).
            NodeKind::InlineMath { .. } | NodeKind::DisplayMath { .. } => {
                out.push(node.range.clone());
            }
            NodeKind::Link {
                delimiter_ranges, ..
            } => {
                // The closing delimiter range covers `](url)`. Everything
                // from the closing `]` through the trailing `)` is the
                // destination — escapes / entities apply per spec, but
                // the bytes are hidden when cursor is outside and we
                // want to show the raw markdown when inside, so skip
                // the scanner here. Children of the link (the link
                // text) still get scanned via tree recursion below.
                if let Some(closer) = delimiter_ranges.get(1) {
                    out.push(closer.clone());
                }
            }
            NodeKind::Image {
                delimiter_ranges, ..
            } => {
                // Same reasoning as for `Link` — the destination
                // portion `](url)` is dim-or-hidden depending on the
                // cursor, never the target of an escape substitution.
                // The alt-text children get walked by the regular
                // recursion below.
                if let Some(closer) = delimiter_ranges.get(1) {
                    out.push(closer.clone());
                }
            }
            _ => {}
        }
        for child in &node.children {
            walk(child, out);
        }
    }
    for node in tree {
        walk(node, &mut out);
    }
    out.sort_by_key(|r| r.start);
    // Merge overlaps so the linear-probe in `escapes::scan` stays O(n).
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(out.len());
    for r in out {
        if let Some(last) = merged.last_mut()
            && r.start <= last.end
        {
            last.end = last.end.max(r.end);
        } else {
            merged.push(r);
        }
    }
    merged
}

/// Apply CommonMark §2.4 backslash escapes and §2.5 entity references
/// to a single rendered block. Each occurrence becomes either:
///
/// * **Cursor outside the construct** — a `Substitution` mapping the
///   construct's source bytes to the resolved display string, plus a
///   `hidden_range` over the construct so the raw bytes don't shape
///   into the line. The substitution's display bytes all map back to
///   `source_range.start`, so a click on the resolved glyph lands at
///   the start of the original construct.
/// * **Cursor inside the construct** — a dimmed `InlineRun` over the
///   construct's source range. The raw bytes shape normally; the
///   delimiter color signals "you're editing this construct" exactly
///   like the cursor-on-emphasis behavior.
///
/// Block kinds that are entirely verbatim (`CodeBlock`, `ThematicBreak`)
/// short-circuit. Inside other blocks, `verbatim` carries the inline
/// verbatim ranges (inline code spans, link destinations) — the scanner
/// skips bytes covered by those.
fn apply_escapes_and_entities(
    block: &mut RenderBlock,
    source: &str,
    verbatim: &[Range<usize>],
    cursor: CursorRange,
) {
    if matches!(
        block.kind,
        BlockKind::CodeBlock { .. } | BlockKind::ThematicBreak
    ) {
        return;
    }
    let bytes = source.as_bytes();
    for span in crate::escapes::scan(bytes, block.source_range.clone(), verbatim) {
        let cons = span.source_range.clone();
        if cursor.overlaps(&cons) {
            // Reveal raw bytes; mark them dimmed so the user sees the
            // construct they're editing in the delimiter color.
            block.inlines.push(InlineRun {
                source_range: cons,
                style: InlineStyle::dimmed(),
            });
        } else {
            // Hide raw bytes; substitute the resolved display.
            block.hidden_ranges.push(cons.clone());
            block.substitutions.push(crate::render_spec::Substitution {
                source_range: cons,
                display: span.display,
            });
        }
    }
}

/// Merge adjacent paragraph blocks where the boundary is a hard break
/// continuation: the previous block ends with `  \n` and the next block
/// is one or more lines of pure chain-continuation prefix (no actual
/// content). Pulldown's parse will fold a continuation-line that has
/// any content into the previous paragraph as a hard-break
/// continuation, but a line of pure prefix bytes ends up as a separate
/// (synthetic empty) block — visually a `paragraph_gap` separates them.
/// Merging restores parity with the with-content case so the user
/// sees the cursor row as the next line of the same paragraph.
///
/// Conditions (all must hold):
/// - Both blocks are `Paragraph`.
/// - Their source ranges are contiguous (`prev.end == next.start`).
/// - Their container chains are identical (same containers, same
///   ordering, same `cursor_inside`).
/// - The chain is non-empty — top-level paragraphs follow the regular
///   pairs model and are kept separate.
/// - The previous block's source ends with `  \n` (a hard break).
/// - Every line of the next block's source equals the canonical chain
///   continuation prefix exactly (no other content on the trailing line).
fn merge_hard_break_continuations(blocks: &mut Vec<RenderBlock>, bytes: &[u8]) {
    let mut i = 0;
    while i + 1 < blocks.len() {
        if should_merge_hard_break_continuation(&blocks[i], &blocks[i + 1], bytes) {
            let next = blocks.remove(i + 1);
            let prev = &mut blocks[i];
            prev.source_range.end = next.source_range.end;
            prev.hidden_ranges.extend(next.hidden_ranges);
            prev.inlines.extend(next.inlines);
            prev.marker_overlays.extend(next.marker_overlays);
            prev.delimiter_lines.extend(next.delimiter_lines);
            prev.substitutions.extend(next.substitutions);
            // Don't advance: try to merge again with the new neighbor
            // in case multiple trailing prefix-only lines were emitted
            // as separate blocks.
        } else {
            i += 1;
        }
    }
}

fn should_merge_hard_break_continuation(
    prev: &RenderBlock,
    next: &RenderBlock,
    bytes: &[u8],
) -> bool {
    if !matches!(prev.kind, BlockKind::Paragraph) || !matches!(next.kind, BlockKind::Paragraph) {
        return false;
    }
    if prev.source_range.end != next.source_range.start {
        return false;
    }
    if prev.containers.is_empty() {
        return false;
    }
    if prev.containers != next.containers {
        return false;
    }
    let end = prev.source_range.end;
    if end < 3 || &bytes[end - 3..end] != b"  \n" {
        return false;
    }
    // Build the canonical chain prefix once; each line of `next` must
    // equal it exactly, with no content beyond.
    let prefix = crate::render_spec::containers_continuation_prefix(&next.containers);
    let prefix_bytes = prefix.as_bytes();
    if prefix_bytes.is_empty() {
        return false;
    }
    let next_bytes = &bytes[next.source_range.start..next.source_range.end];
    let mut p = 0;
    while p < next_bytes.len() {
        let line_end = next_bytes[p..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|i| p + i)
            .unwrap_or(next_bytes.len());
        let line = &next_bytes[p..line_end];
        if line != prefix_bytes {
            return false;
        }
        p = if line_end < next_bytes.len() {
            line_end + 1
        } else {
            line_end
        };
    }
    true
}

/// Sort `ranges` by start and merge any overlapping or touching
/// entries. Multiple passes — per-item hide, BQ marker attach,
/// chain-driven prefix hide — can each contribute overlapping
/// ranges that shape down to the same display bytes; the element
/// layer handles overlap correctly, but a normalized list is easier
/// to inspect (and lets summing-style assertions in tests reflect
/// actual hidden byte count without double-counting).
fn merge_hidden_ranges(ranges: &mut Vec<Range<usize>>) {
    if ranges.len() <= 1 {
        return;
    }
    ranges.sort_by_key(|r| r.start);
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for r in ranges.drain(..) {
        if let Some(last) = merged.last_mut()
            && r.start <= last.end
        {
            last.end = last.end.max(r.end);
        } else {
            merged.push(r);
        }
    }
    *ranges = merged;
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

    /// Container-aware variant of [`overlaps`]. Pulldown's range for a
    /// `List` / `ListItem` / `BlockQuote` often extends past the last
    /// content line into trailing structural separators (the `\n\n`
    /// pair following the construct, or further empty rows). Without
    /// trimming, a cursor parked on the post-construct empty row
    /// would still match by boundary equality and the renderer would
    /// flag the construct as "focused" — bullets stay raw, blockquote
    /// markers dim into view, etc. The trimming rule is the same one
    /// `analysis::effective_node_end` uses for the cursor walker, so
    /// chain queries and render decisions agree.
    fn overlaps_node(self, node: &SyntaxNode, source: &str) -> bool {
        let bytes = source.as_bytes();
        let effective_end = crate::analysis::effective_node_end(node, bytes);
        let trimmed = node.range.start..effective_end;
        self.overlaps(&trimmed)
    }
}

fn render_node(
    node: &SyntaxNode,
    tree: &[SyntaxNode],
    source: &str,
    cursor: CursorRange,
    containers: &[Container],
    out: &mut Vec<RenderBlock>,
) {
    match &node.kind {
        NodeKind::Paragraph => render_paragraph(node, source, cursor, containers, out),
        NodeKind::Heading { .. } => render_heading(node, cursor, containers, out),
        NodeKind::CodeBlock { .. } => render_code_block(node, source, cursor, containers, out),
        NodeKind::BlockQuote { prefix_ranges } => {
            render_blockquote(node, prefix_ranges, tree, source, cursor, containers, out)
        }
        NodeKind::List { kind } => render_list(node, *kind, tree, source, cursor, containers, out),
        NodeKind::ThematicBreak => render_thematic_break(node, source, cursor, containers, out),
        // Top-level DisplayMath — pulldown emits `$$..$$` constructs
        // that sit on their own lines (block form, after the
        // `pulldown-cmark` fork's `parse_display_math_block`) directly
        // at block scope without a wrapping paragraph. This is the
        // analog of how fenced code blocks arrive: a top-level leaf,
        // not nested in a paragraph. The same `BlockKind::DisplayMath`
        // emission rule applies — `emit_display_math_block` is shared
        // with the paragraph-promoted single-line case (`$$x$$`
        // standalone) so both flavors render identically.
        NodeKind::DisplayMath { .. } => {
            emit_display_math_block(node, source, cursor, containers, out)
        }
        // Anything else at top level — nothing to do yet.
        _ => {}
    }
}

/// Emit a `BlockKind::DisplayMath` block for `math` (a `NodeKind::DisplayMath`
/// node). Shared between two call sites:
///
/// 1. `render_paragraph`'s `sole_display_math_child` promotion — a
///    paragraph whose only content-bearing child is a single inline
///    `$$..$$` construct (e.g. `$$x^2$$` standalone on its own line in
///    source, but parsed by pulldown as inline math inside a paragraph).
/// 2. `render_node`'s top-level dispatch — the `pulldown-cmark` fork's
///    block-level `$$..$$` construct, which arrives without a paragraph
///    wrapper.
///
/// Both produce the same render output: the `BlockKind::DisplayMath`
/// block kind, the inclusive-overlap `edit_mode` test, dim-delimiter +
/// mono-content shape in edit mode, full-range hide in display mode.
fn emit_display_math_block(
    math: &SyntaxNode,
    source: &str,
    cursor: CursorRange,
    containers: &[Container],
    out: &mut Vec<RenderBlock>,
) {
    let (content_range, delimiter_ranges) = match &math.kind {
        NodeKind::DisplayMath {
            content_range,
            delimiter_ranges,
        } => (content_range.clone(), delimiter_ranges.clone()),
        // The two call sites (top-level `render_node` dispatch and
        // `sole_display_math_child` promotion in `render_paragraph`)
        // both already pattern-match the kind before calling — a
        // non-DisplayMath here would be a caller bug, not a recoverable
        // state.
        _ => unreachable!("emit_display_math_block called on non-DisplayMath kind"),
    };
    let math_range = math.range.clone();
    // Inclusive overlap: a cursor *touching* either `$$` fence (at
    // `math.start` or `math.end`) counts as inside, so the user sees
    // and can edit the source. Click hit-testing on the typeset
    // overlay collapses to `math.start` (every source byte is hidden,
    // so the shaped line is zero-width and every click maps to display
    // column 0); inclusive overlap is what makes that click flip into
    // edit mode. The cost is that arrow-keying past the math also
    // flashes the source momentarily — the documented "always
    // navigable" mode.
    //
    // `math.range` is the *trimmed* range from `math_kind` (see
    // `parser.rs`) — the trailing `\n` the fork's
    // `parse_display_math_block` includes past the closer line has
    // been stripped — so this overlap test agrees with the closer
    // delimiter's last byte rather than firing on the byte after.
    let edit_mode = cursor.overlaps(&math_range);
    let mut block = RenderBlock::new(
        math_range.clone(),
        BlockKind::DisplayMath {
            content_range: content_range.clone(),
            edit_mode,
        },
    );
    block.containers = containers.to_vec();

    // Register delimiter lines
    let bytes = source.as_bytes();
    block
        .delimiter_lines
        .push(line_start_offset(bytes, delimiter_ranges[0].start)..delimiter_ranges[0].end);
    if let Some(closer) = delimiter_ranges.get(1)
        && !closer.is_empty()
    {
        block
            .delimiter_lines
            .push(line_start_offset(bytes, closer.start)..closer.end);
    }

    if edit_mode {
        // Edit mode: every byte shapes — `$$` delimiters dim, inner
        // LaTeX shapes in mono. Delimiters MUST stay shaped (not
        // hidden) so click-hit-test, selection geometry, and arrow
        // navigation can land on them. Hiding the bytes would also
        // collapse them out of the display line so the user couldn't
        // see what they're editing.
        for d in &delimiter_ranges {
            if !d.is_empty() {
                block.inlines.push(InlineRun {
                    source_range: d.clone(),
                    style: InlineStyle::dimmed(),
                });
            }
        }
        if content_range.start < content_range.end {
            block.inlines.push(InlineRun {
                source_range: content_range,
                style: InlineStyle::code(),
            });
        }
    } else {
        // Display mode: hide the *entire* math range so no source text
        // shapes underneath the typeset math overlay. The element
        // layer's request_layout + prepaint produce the math's height
        // directly; the (hidden-only) shaped lines collapse to one
        // empty fallback line that anchors cursor-at-boundary
        // hit-testing.
        block.hidden_ranges.push(math_range);
    }
    out.push(block);
}

/// Render a thematic break (`---`, `***`, `___`).
///
/// The block kind is its own variant so the element layer can paint a
/// horizontal rule decoration. The source bytes follow the standard
/// cursor rule — hidden when the cursor is outside the construct,
/// dimmed when the cursor is on the line so the user can see and
/// edit the raw markdown.
///
/// The delimiter range is pre-trimmed of any trailing `\n` so it
/// matches the shaped-line bounds the element layer compares against:
/// `inject_empty_paragraphs` trims the block's `source_range` of
/// trailing `\n`s downstream, and the hidden-range lookup in
/// `build_display_line` requires `r.end <= line_logical_end`. Without
/// the trim, a hide range that includes the `\n` silently fails the
/// match and the `---` characters render as plain text.
fn render_thematic_break(
    node: &SyntaxNode,
    source: &str,
    cursor: CursorRange,
    containers: &[Container],
    out: &mut Vec<RenderBlock>,
) {
    let mut block = RenderBlock::new(node.range.clone(), BlockKind::ThematicBreak);
    block.containers = containers.to_vec();
    let cursor_inside = cursor.overlaps(&node.range);
    let bytes = source.as_bytes();
    let trimmed_end = block_content_end_excl(bytes, &node.range).max(node.range.start);
    let delim_range = node.range.start..trimmed_end;
    apply_delimiter_visibility(&[delim_range], cursor_inside, &mut block);
    out.push(block);
}

fn render_paragraph(
    node: &SyntaxNode,
    source: &str,
    cursor: CursorRange,
    containers: &[Container],
    out: &mut Vec<RenderBlock>,
) {
    // Promote a paragraph whose sole content-bearing child is a
    // `DisplayMath` event into a `BlockKind::DisplayMath` block —
    // matches GitHub-flavored rendering where a `$$...$$` standalone
    // construct sits on its own row regardless of whether the source
    // happens to land it on its own line. Mixed paragraphs (text +
    // display math + text) keep the inline rendering path. Block-level
    // `$$\n...\n$$` constructs arrive at top level (no paragraph
    // wrapper) and dispatch through `render_node`; both call sites
    // share `emit_display_math_block`.
    if let Some(math) = sole_display_math_child(node) {
        emit_display_math_block(math, source, cursor, containers, out);
        return;
    }

    // Promote a paragraph whose sole content-bearing child is an
    // `Image` event into a `BlockKind::Image` block — matches the
    // standard markdown convention where a `![alt](url)` standalone
    // construct sits on its own row. Mixed paragraphs (text + image
    // + text) keep the inline rendering path and the image becomes
    // an inline overlay alongside the surrounding text.
    if let Some(image) = sole_image_child(node) {
        let (alt_range, dest_url, image_range, delimiter_ranges) = match &image.kind {
            NodeKind::Image {
                alt_range,
                dest_url,
                delimiter_ranges,
            } => (
                alt_range.clone(),
                dest_url.clone(),
                image.range.clone(),
                delimiter_ranges.clone(),
            ),
            _ => unreachable!("sole_image_child guard"),
        };
        // Inclusive overlap (same rule as DisplayMath): a cursor
        // touching either delimiter — at `image.start` or `image.end`
        // — flips to edit mode, which is the only path that paints
        // raw bytes for click hit-testing.
        let edit_mode = cursor.overlaps(&image_range);
        let mut block = RenderBlock::new(
            image_range.clone(),
            BlockKind::Image {
                alt_range: alt_range.clone(),
                dest_url,
                edit_mode,
            },
        );
        block.containers = containers.to_vec();
        if edit_mode {
            // Edit mode: every byte shapes — `![` and `](url)`
            // delimiters dim, alt text shapes normally. Mirrors the
            // display-math edit-mode path so cursor navigation
            // and selection work over the raw bytes.
            for d in &delimiter_ranges {
                if !d.is_empty() {
                    block.inlines.push(InlineRun {
                        source_range: d.clone(),
                        style: InlineStyle::dimmed(),
                    });
                }
            }
            // Walk the image's children so any inline styling inside
            // the alt text composes (e.g. `![*emphasis*](u)`).
            collect_inlines_in_range(image, &alt_range, cursor, &mut block);
        } else {
            // Display mode: pre-stage the *fallback* shape inline
            // runs (dim delimiters + visible alt text on the raw
            // source bytes). The element layer will push a hide
            // over `image_range` on a successful or in-flight load
            // — hiding suppresses these runs from shaping. On a
            // failed load, no hide is added, so the user sees the
            // dim delimiters + alt text and can correct the URL.
            // This mirrors the inline-image model: render emits the
            // construct without committing to a hide, element
            // commits to a hide only when it has a paintable image.
            for d in &delimiter_ranges {
                if !d.is_empty() {
                    block.inlines.push(InlineRun {
                        source_range: d.clone(),
                        style: InlineStyle::dimmed(),
                    });
                }
            }
            collect_inlines_in_range(image, &alt_range, cursor, &mut block);
        }
        out.push(block);
        return;
    }

    let mut block = RenderBlock::new(node.range.clone(), BlockKind::Paragraph);
    block.containers = containers.to_vec();
    collect_inlines(node, cursor, &mut block);
    out.push(block);
}

/// Returns the lone content-bearing `DisplayMath` child of `node` if
/// the paragraph contains exactly one (modulo whitespace-only `Text`
/// nodes from leading / trailing source whitespace pulldown collapses
/// into Text events). Returns `None` for mixed paragraphs.
fn sole_display_math_child(node: &SyntaxNode) -> Option<&SyntaxNode> {
    let mut math: Option<&SyntaxNode> = None;
    for child in &node.children {
        match &child.kind {
            NodeKind::DisplayMath { .. } => {
                if math.is_some() {
                    return None;
                }
                math = Some(child);
            }
            NodeKind::SoftBreak | NodeKind::HardBreak => {
                // OK — pulldown sometimes emits these around math.
            }
            _ => {
                return None;
            }
        }
    }
    math
}

/// Returns the lone content-bearing `Image` child of `node` if the
/// paragraph contains exactly one (modulo soft/hard breaks pulldown
/// emits around standalone images). Mirrors [`sole_display_math_child`]
/// — same promotion rule for image blocks.
fn sole_image_child(node: &SyntaxNode) -> Option<&SyntaxNode> {
    let mut img: Option<&SyntaxNode> = None;
    for child in &node.children {
        match &child.kind {
            NodeKind::Image { .. } => {
                if img.is_some() {
                    return None;
                }
                img = Some(child);
            }
            NodeKind::SoftBreak | NodeKind::HardBreak => {}
            _ => return None,
        }
    }
    img
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
    tree: &[SyntaxNode],
    source: &str,
    cursor: CursorRange,
    containers: &[Container],
    out: &mut Vec<RenderBlock>,
) {
    let cursor_inside = cursor.overlaps_node(node, source);
    // The level of *this* blockquote's prefix markers is the number
    // of containers wrapping it (outer blockquotes / list items, etc.)
    // — the element layer uses it to look up the matching border bar
    // when painting overlay markers.
    let level = containers.len();
    let mut child_chain = containers.to_vec();
    child_chain.push(Container::BlockQuote { cursor_inside });

    let start = out.len();
    for child in &node.children {
        render_node(child, tree, source, cursor, &child_chain, out);
    }

    let bytes = source.as_bytes();
    // **Same-line opening marker.** The parser's
    // `blockquote_prefix_ranges` walks each line from `line_start`
    // looking for `>` past `outer_depth` indent / `>` segments — but
    // when this BQ opens *mid-line* inside an LI (e.g. `1. > foo`),
    // the line begins with the LI marker (`1. `) which the prefix
    // walker doesn't recognize, so the opening `> ` is silently
    // dropped from `prefix_ranges`. Without it the marker bytes
    // never reach `attach_marker` / `hidden_ranges` and the `>`
    // shows as plain text in the shaped line. Detect the case
    // (BQ.range.start has `>` and no existing prefix already covers
    // it) and prepend a synthetic opening prefix so the regular
    // attach loop hides it like any other line's marker.
    let mut synth_prefix: Option<Range<usize>> = None;
    if node.range.start < bytes.len()
        && bytes[node.range.start] == b'>'
        && !prefix_ranges.iter().any(|r| r.start == node.range.start)
    {
        let mut q = node.range.start + 1;
        if q < bytes.len() && bytes[q] == b' ' {
            q += 1;
        }
        synth_prefix = Some(node.range.start..q);
    }
    let owned: Vec<Range<usize>>;
    let prefix_ranges: &[Range<usize>] = if let Some(synth) = synth_prefix {
        owned = std::iter::once(synth)
            .chain(prefix_ranges.iter().cloned())
            .collect();
        &owned
    } else {
        prefix_ranges
    };

    // `deferred` holds the most recent unclaimed prefix that we
    // tentatively assumed is the *middle* of a `[prefix]\n[prefix]`
    // structural pair (no row of its own). It either gets dropped
    // when its partner comes in — the partner takes the synthetic —
    // or, if no partner ever shows up, we promote the deferred
    // prefix itself to a synthetic at end-of-loop. The promotion
    // keeps a freshly-typed lone `> ` parsing-as-blockquote
    // *rendering* as a blockquote immediately, instead of waiting
    // for the user to type a paired marker.
    //
    // Track indices of synthetics emitted by this BQ so we can run a
    // chain-aware hide pass on them after all markers are attached.
    let mut synth_indices: Vec<usize> = Vec::new();
    let mut deferred: Option<Range<usize>> = None;
    for prefix in prefix_ranges {
        if let Some(leaf) = find_leaf_for_prefix(&mut out[start..], prefix, source) {
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
            // paragraphs). Chain comes from `chain_for_position`
            // (the canonical analysis-side query) so we pick up
            // *any* container that encloses the byte — including a
            // deeper inner list-item that `child_chain`, built up
            // walking-down from this BQ, doesn't see.
            let line_end = source_line_end(bytes, prefix.end);
            let mut synth = RenderBlock::new(prefix.start..line_end, BlockKind::Paragraph);
            synth.containers = chain_for_position(tree, source, cursor, prefix.start);
            attach_marker(&mut synth, prefix, cursor_inside, level);
            synth_indices.push(out.len());
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
        synth.containers = chain_for_position(tree, source, cursor, prefix.start);
        attach_marker(&mut synth, &prefix, cursor_inside, level);
        synth_indices.push(out.len());
        out.push(synth);
    }

    // Extend each synth's range back to the start of its line so the
    // canonical-prefix hide pass at the top level (`render`'s final
    // loop, `hide_chain_continuation_prefix`) can claim every byte
    // of the chain prefix on this line — outer LI indents and outer
    // BQ markers that precede this BQ's emitting prefix on the same
    // source line.
    for &idx in &synth_indices {
        extend_synth_to_line_start(&mut out[idx], bytes);
    }

    // Synthetics are appended in `prefix_ranges` order (source order)
    // but may now sit *after* a parsed leaf that occurs later in
    // source — sort so subsequent passes (outer-blockquote
    // distribution, `inject_empty_paragraphs`, the editor's per-block
    // index) see blocks in source order.
    out[start..].sort_by_key(|b| b.source_range.start);
}

/// Walk back the synth's `source_range.start` to the start of its
/// first line (the byte right after the previous `\n`, or buffer
/// start). This pulls in *all* of the canonical continuation-prefix
/// bytes that lead into the synth — leading LI indent, outer BQ
/// markers, intermediate LI indent — so the chain-aware hide pass
/// can mask the full prefix on this line. Without this the synth's
/// range starts at the BQ marker that emitted it and never claims
/// the bytes that precede that marker on the line, so any portion
/// of the canonical prefix sitting before the synth's emitting BQ
/// renders as visible text.
fn extend_synth_to_line_start(leaf: &mut RenderBlock, bytes: &[u8]) {
    let mut p = leaf.source_range.start;
    while p > 0 && bytes[p - 1] != b'\n' {
        p -= 1;
    }
    leaf.source_range.start = p;
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
    tree: &[SyntaxNode],
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
        let NodeKind::ListItem { marker_range, task } = &child.kind else {
            continue;
        };
        let item_kind = match &kind {
            ListKind::Unordered => {
                ListItemKind::Unordered(marker_char(source, marker_range), *task)
            }
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
            tree,
            source,
            cursor,
            containers,
            out,
        );
    }
}

/// Widest marker *text* anywhere in this list.
///
/// For unordered lists with no task items, every marker is two bytes
/// (`- `, `* `, or `+ `); they shape to nearly identical pixel
/// widths. We canonicalize to `"- "` so the indent computation is
/// stable regardless of which bullet char a particular item uses.
///
/// **Task list items** (`- [ ] todo` / `- [x] done`) carry an extra
/// `[ ] ` (4 bytes) of GFM chrome on top of the bullet. When the
/// cursor is inside such an item the overlay renders the *raw*
/// chrome bytes so the user can edit them directly; the indent
/// column has to be wide enough to fit that raw form, otherwise the
/// content edge would shift between focus states. We canonicalize
/// to `"- [ ] "` whenever the list contains *any* task item, so
/// every sibling — task or plain — aligns at the same content
/// column. (The non-task siblings still paint with a `• ` bullet
/// overlay; the wider indent just leaves extra space to its left.)
///
/// For ordered lists the widest marker is whichever item has the most
/// digits — `start + child_count - 1`. We format it back with the
/// `". "` suffix the parser uses today; once `)` markers are
/// supported the canonicalization will need to round-trip the actual
/// punctuation.
fn compute_list_max_marker_text(node: &SyntaxNode, kind: ListKind) -> String {
    match kind {
        ListKind::Unordered => {
            let has_task = node
                .children
                .iter()
                .any(|c| matches!(c.kind, NodeKind::ListItem { task: Some(_), .. }));
            if has_task {
                "- [ ] ".to_string()
            } else {
                "- ".to_string()
            }
        }
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
    tree: &[SyntaxNode],
    source: &str,
    cursor: CursorRange,
    containers: &[Container],
    out: &mut Vec<RenderBlock>,
) {
    let cursor_inside = cursor.overlaps_node(node, source);
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
                    let was_first = !emitted_first;
                    render_node(child, tree, source, cursor, &chain, out);
                    if out.len() > before {
                        // When this is the *first* leaf the item
                        // emits, extend the leftmost recursed leaf's
                        // source range back to the item's start so
                        // the LI marker bytes (`- ` / `1. `) sit
                        // inside it. Without this the marker-chrome
                        // lookup below (which finds the leaf whose
                        // `source_range.start <= node.range.start`)
                        // misses — the recursion's leaf starts past
                        // the marker, e.g. `1. > foo` parses as
                        // LI→BQ→Paragraph and the BQ-paragraph leaf
                        // starts at the BQ marker (byte 3), so the
                        // outer LI's marker overlay never lands on
                        // any leaf. Mirrors what
                        // `extend_leading_range(_, false, …)` does
                        // for the Paragraph-child path above.
                        if was_first
                            && let Some(idx) =
                                (before..out.len()).min_by_key(|&i| out[i].source_range.start)
                            && out[idx].source_range.start > node.range.start
                        {
                            out[idx].source_range.start = node.range.start;
                        }
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
        // Task list items (`- [ ] todo` / `- [x] done`): hide the
        // GFM task marker bytes that sit immediately after the
        // bullet so the rendered first line reads `todo` / `done`
        // and the marker chrome (checkbox glyph or raw `- [ ]`)
        // paints as an overlay in the indent strip. We don't extend
        // `marker_range` itself because that would inflate the
        // continuation-indent requirement (a task item with a
        // multi-paragraph body uses 2-space continuation matching
        // the bullet, not 6 spaces). The overlay's `source_range`
        // *does* cover the full chrome (bullet + brackets) so the
        // cursor can be placed inside the brackets via overlay-aware
        // cursor positioning when the user navigates there.
        let is_task = matches!(item_kind, ListItemKind::Unordered(_, Some(_)));
        let mut overlay_range = marker_range.clone();
        if is_task {
            let task_start = marker_range.end;
            let task_end = (task_start + 4).min(line_end);
            if task_end > task_start && task_start < bytes.len() && bytes[task_start] == b'[' {
                first_leaf.hidden_ranges.push(task_start..task_end);
                overlay_range.end = task_end;
            }
        }
        // Push the marker overlay so the element layer paints the
        // marker text in this item's indent strip. The element
        // layer resolves the marker text to display from
        // `containers[level]` (kind + cursor_inside) and can place
        // the cursor inside the overlay when source position is
        // within `overlay_range`.
        first_leaf.marker_overlays.push(MarkerOverlay {
            source_range: overlay_range,
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
            | NodeKind::DisplayMath { .. }
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
///
/// **Same-line fallback for nested-prefix lines.** When this BQ's prefix
/// sits at the head of a line whose remainder is owned by a *deeper*
/// nested leaf (e.g. outer `> ` followed by list-item indent followed
/// by inner `> content` — the leaf for the inner BQ paragraph starts
/// past the outer prefix), the strict `start <= prefix.end < end`
/// check would miss it because the inner leaf's `start` sits *after*
/// `prefix.end`. We then look for a leaf that begins later on the
/// same source line (no intervening `\n`) and attach the marker to
/// it — otherwise the deferred-pair logic would emit a phantom synth
/// leaf overlapping the real inner leaf.
fn find_leaf_for_prefix<'a>(
    slice: &'a mut [RenderBlock],
    prefix: &Range<usize>,
    source: &str,
) -> Option<&'a mut RenderBlock> {
    let bytes = source.as_bytes();
    let target = prefix.end;
    // Compute the line bound: the smallest byte position at-or-after
    // `target` that is a `\n` (or `bytes.len()` if none). Any leaf
    // whose `source_range.start` lies in `target..line_end` is on the
    // same source line as the prefix.
    let mut line_end = target;
    while line_end < bytes.len() && bytes[line_end] != b'\n' {
        line_end += 1;
    }
    // Prefer the strict containment match; if none exists, fall back
    // to the leftmost leaf that *starts* somewhere on the same line.
    let mut found_idx: Option<usize> = None;
    for (i, b) in slice.iter().enumerate() {
        if b.source_range.start <= target && target < b.source_range.end {
            return Some(&mut slice[i]);
        }
        if b.source_range.start > target && b.source_range.start <= line_end {
            // Pick the leftmost same-line leaf so the marker attaches
            // to the leaf whose content begins closest to the prefix.
            match found_idx {
                None => found_idx = Some(i),
                Some(prev) if b.source_range.start < slice[prev].source_range.start => {
                    found_idx = Some(i);
                }
                _ => {}
            }
        }
    }
    if let Some(i) = found_idx {
        return Some(&mut slice[i]);
    }
    // **Unterminated-fence boundary.** When the BQ contains an
    // unterminated fenced code block, the CodeBlock leaf's
    // `source_range.end` sits at `bytes.len()`. The strict
    // `target < end` test above misses any prefix on the *last* body
    // line (where `prefix.end == bytes.len() == leaf.source_range.end`),
    // and without this match the deferred-pair logic emits a phantom
    // empty paragraph at the prefix's range — overlapping the
    // CodeBlock leaf at the same bytes (see
    // `bugs.md::render_walker_emits_phantom_paragraph_inside_unterminated_fenced_code`).
    // Detect "prefix lies inside an unterminated verbatim region"
    // (fenced code or block-level `$$..$$` math) via the verbatim
    // predicate and attach the marker to the existing leaf instead.
    if crate::analysis::is_in_verbatim_region(source, prefix.start) {
        for (i, b) in slice.iter().enumerate() {
            if matches!(
                b.kind,
                BlockKind::CodeBlock { .. } | BlockKind::DisplayMath { .. }
            ) && b.source_range.start <= prefix.start
                && prefix.start <= b.source_range.end
            {
                return Some(&mut slice[i]);
            }
        }
    }
    None
}

fn render_code_block(
    node: &SyntaxNode,
    source: &str,
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

    // Mark fence rows for layout. The element layer's
    // `line_is_fully_in_a_delimiter` check requires the
    // `delimiter_lines` entry to *cover* the shaped line's logical
    // range — which starts at the byte right after the previous
    // `\n`, not at the fence chars themselves. Inside a blockquote
    // or a list item the line begins with the chain prefix (`> `,
    // `   `, …) so the entry has to extend back past those bytes
    // for the fence row to register as a delimiter.
    //
    // The opener row covers from line-start through the info
    // string's end (or the opener fence chars' end if there's no
    // info string). The closer row covers from line-start through
    // the closer fence chars' end.
    let bytes = source.as_bytes();
    let opener_line_end = info_string_range
        .as_ref()
        .map(|r| r.end)
        .unwrap_or_else(|| delimiter_ranges[0].end);
    block
        .delimiter_lines
        .push(line_start_offset(bytes, delimiter_ranges[0].start)..opener_line_end);
    if let Some(closer) = delimiter_ranges.get(1) {
        block
            .delimiter_lines
            .push(line_start_offset(bytes, closer.start)..closer.end);
    }

    // No inline children — code-block content is literal source bytes,
    // shaped in mono font by the element layer.
    out.push(block);
}

/// Byte position of the start of the line containing `pos` — the byte
/// right after the previous `\n`, or 0 if `pos` is on the first line.
fn line_start_offset(bytes: &[u8], pos: usize) -> usize {
    let mut s = pos.min(bytes.len());
    while s > 0 && bytes[s - 1] != b'\n' {
        s -= 1;
    }
    s
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
        NodeKind::InlineCode {
            delimiter_ranges,
            content_range,
        } => {
            // Pulldown's `Event::Code` is a leaf event — there are
            // no `Text` children, so we can't rely on
            // `walk_styled_children` to emit the run. Build it
            // directly: hide / dim the backtick delimiters per the
            // cursor rule, then emit one `code`-styled inline run
            // covering the content range. The element layer reads
            // `InlineStyle::code` and switches to the mono font with
            // a faint background fill.
            let inside = cursor.overlaps(&node.range);
            apply_delimiter_visibility(delimiter_ranges, inside, block);
            if content_range.start < content_range.end {
                let mut style = base.clone().merge(InlineStyle::code());
                if inside {
                    // No-op: the content of an inline code span is
                    // *not* dimmed when the cursor is inside — the
                    // delimiters are. The content stays in the
                    // normal text color so the user can still read
                    // the code while editing it.
                    let _ = &mut style;
                }
                block.inlines.push(InlineRun {
                    source_range: content_range.clone(),
                    style,
                });
            }
        }
        NodeKind::Link {
            delimiter_ranges,
            text_range,
            ..
        } => {
            let inside = cursor.overlaps(&node.range);
            apply_delimiter_visibility(delimiter_ranges, inside, block);
            // Walk children for the link text so nested styles
            // (`[**bold link**](url)`) compose. The merged `link`
            // bit drives color + underline at paint time. If a link
            // has no children (e.g. an autolink with no text node)
            // emit a single run covering `text_range` so the URL
            // text still picks up link styling.
            let style = base.clone().merge(InlineStyle::link());
            if node.children.is_empty() {
                if text_range.start < text_range.end {
                    block.inlines.push(InlineRun {
                        source_range: text_range.clone(),
                        style,
                    });
                }
            } else {
                walk_styled_children(node, text_range, cursor, style, block);
            }
        }
        NodeKind::Image {
            delimiter_ranges,
            alt_range,
            dest_url,
        } => {
            // Image rendering mirrors inline math:
            //
            // **Cursor outside**: hide every byte and queue an
            // `ImageOverlay`. The element layer loads the image,
            // reserves horizontal space via a width-matched
            // substitution, and paints the image at the
            // substitution's display position.
            //
            // **Cursor inside**: fall back to dim delimiters + the
            // raw alt text shaped in the normal color so the user
            // can read and edit the markdown directly. (Alt text is
            // user-authored prose — unlike math's LaTeX content, it
            // wants normal styling, not mono.)
            let inside = cursor.overlaps(&node.range);
            if inside {
                apply_delimiter_visibility(delimiter_ranges, true, block);
                // Recurse into alt-text children so nested styling
                // (`![**bold alt**](u)`) composes.
                walk_styled_children(node, alt_range, cursor, base.clone(), block);
            } else {
                block.image_overlays.push(crate::render_spec::ImageOverlay {
                    source_range: node.range.clone(),
                    alt_range: alt_range.clone(),
                    dest_url: dest_url.clone(),
                });
            }
        }
        NodeKind::InlineMath {
            delimiter_ranges,
            content_range,
        }
        | NodeKind::DisplayMath {
            delimiter_ranges,
            content_range,
        } => {
            // Inline math (and inline-positioned display math —
            // `$$..$$` that didn't promote to a block because the
            // host paragraph has surrounding text). Two paths:
            //
            // **Cursor outside** the construct: hide every byte and
            // emit a `MathOverlay`. The element layer typesets,
            // substitutes a width-matched run of non-breaking
            // spaces so the line reserves horizontal space, and
            // paints the typeset math at the substitution's
            // display position. The result composes with surrounding
            // text on the same shaped line.
            //
            // **Cursor inside** the construct: fall back to
            // dim-delimiter / mono-content shaping so the user can
            // read and edit the raw LaTeX directly. No overlay —
            // `MathOverlay` is for the typeset rendering only.
            let inside = cursor.overlaps(&node.range);
            if inside {
                apply_delimiter_visibility(delimiter_ranges, true, block);
                if content_range.start < content_range.end {
                    let style = base.clone().merge(InlineStyle::code());
                    block.inlines.push(InlineRun {
                        source_range: content_range.clone(),
                        style,
                    });
                }
            } else {
                // Cursor outside — queue a typeset overlay and
                // *don't* add a hidden range here. Suppressing the
                // source bytes is the element layer's job, and it
                // does it differently per typeset outcome:
                //
                //   - **Success**: the NBSP `Substitution` it adds
                //     replaces the source bytes by definition, so no
                //     separate hidden range is needed.
                //   - **Failure**: the source bytes shape as
                //     fallback (dim delimiters + mono content)
                //     so the user sees the raw LaTeX they need to
                //     fix.
                //
                // Routing the hide through the element layer keeps
                // the failure path from leaving a blank gap.
                let display_style = matches!(node.kind, NodeKind::DisplayMath { .. });
                block.math_overlays.push(crate::render_spec::MathOverlay {
                    source_range: node.range.clone(),
                    content_range: content_range.clone(),
                    display_style,
                });
            }
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

fn inject_empty_paragraphs(
    source: &str,
    tree: &[SyntaxNode],
    cursor: CursorRange,
    real_blocks: Vec<RenderBlock>,
) -> Vec<RenderBlock> {
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
        let mut trimmed_end = block_content_end_excl(bytes, &block.source_range).max(trimmed_start);
        // Extend past any trailing horizontal whitespace on the
        // block's last content line. Pulldown's leaf range typically
        // excludes trailing spaces (CommonMark "ignored" trailing
        // whitespace); without this extension, a cursor parked at
        // the byte right after a typed trailing space falls outside
        // every block's range — no row claims it, and the caret
        // never paints.
        //
        // Two guards:
        //   1. Only extend when `trimmed_end` is *not* immediately
        //      preceded by a `\n`. If it is, the trim already
        //      walked past the line terminator (or the block ended
        //      with a hard break that kept its `\n` inside the
        //      range), and any further bytes belong to the *next*
        //      line — extending across a `\n` would reach into
        //      another block's territory.
        //   2. Stop at the next `\n` so we never cross a line
        //      boundary in the forward direction either.
        let already_past_nl = trimmed_end > 0 && bytes[trimmed_end - 1] == b'\n';
        if !already_past_nl {
            while trimmed_end < bytes.len()
                && (bytes[trimmed_end] == b' ' || bytes[trimmed_end] == b'\t')
            {
                trimmed_end += 1;
            }
        }
        block.source_range = trimmed_start..trimmed_end;
    }

    let mut out: Vec<RenderBlock> = Vec::with_capacity(real_blocks.len() * 2);

    // Leading gap. Each pair of `\n`s is one leading empty paragraph.
    let first_content = real_blocks[0].source_range.start;
    let leading_count = (0..first_content).filter(|&p| bytes[p] == b'\n').count();
    let leading_empties = leading_count / 2;
    for i in 0..leading_empties {
        let start = 2 * i;
        let mut synth = empty_paragraph_pair(start);
        // Query the chain at the second byte of the synth's pair (the
        // cursor's natural resting position — the first `\n` of a
        // `\n\n` pair is forbidden and snaps forward). This is past
        // any container's `range.end` boundary that coincides with
        // `start`, so a synth sitting in a top-level paragraph break
        // *outside* a just-closed container reports an empty chain
        // instead of inheriting the closed container's chain via a
        // boundary-equality match in `pick_chain_target_for_position`.
        let probe = (start + 1).min(bytes.len());
        synth.containers = chain_for_position(tree, source, cursor, probe);
        out.push(synth);
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
                let mut synth = empty_paragraph_pair(start);
                // See note on `probe` in the leading-gap loop. The
                // synth's first byte is the second-of-pair `\n` in the
                // structural break — at the boundary of a just-closed
                // container its chain query risks a `position ==
                // node.range.end` match. Query one byte deeper so the
                // chain reports "outside the closed container" when
                // the synth is structurally past it.
                let probe = (start + 1).min(bytes.len());
                synth.containers = chain_for_position(tree, source, cursor, probe);
                out.push(synth);
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
    //
    // The *last* trailing pair additionally extends through any
    // remaining trailing whitespace (e.g. the continuation indent that
    // sits after `\n\n` inside a list item: `1. one\n\n   `). Without
    // that, the cursor at end-of-buffer would have no block claiming
    // its byte and the synthetic's chain would not match the cursor
    // walker's chain at the same position.
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
        let end = if i + 1 == trailing_empties {
            bytes.len()
        } else {
            (start + 2).min(bytes.len())
        };
        // Suppress synth-paragraph injection when the trailing position
        // falls inside any verbatim region (fenced code or block-level
        // `$$..$$` math). The CodeBlock / DisplayMath leaf already
        // covers those bytes — a synth here would emit a phantom
        // Paragraph leaf overlapping it. See
        // `bugs.md::render_walker_emits_phantom_…`.
        if crate::analysis::is_in_verbatim_region(source, start) {
            continue;
        }
        let mut synth = RenderBlock::new(start..end, BlockKind::Paragraph);
        // See note on `probe` in the leading / inter-block loops above.
        // Query past the leading boundary `\n` so a trailing synth that
        // sits past a just-closed container reports an empty chain.
        let probe = (start + 1).min(bytes.len());
        synth.containers = chain_for_position(tree, source, cursor, probe);
        hide_synth_continuation_indent(&mut synth, bytes);
        out.push(synth);
    }

    out
}

/// Hide every byte of the canonical chain continuation prefix on every
/// line of a synthetic empty-paragraph leaf. The chain prefix
/// interleaves LI continuation indents with BQ markers (outermost
/// first); a flat "leading-LI-sum" hide misses the LI indent bytes that
/// sit *between* BQ markers in alternating chains like
/// `[LI, BQ, LI, BQ]` — those bytes render as visible text and push
/// the cursor past the chain's content edge.
///
/// Mirrors the parsed-leaf rendering: the real render walker hides BQ
/// markers via `attach_marker` and per-LI continuation indents via
/// `hide_item_continuation_indent`. Synth leaves don't have any of
/// that scaffolding — they're emitted whole-cloth by
/// `inject_empty_paragraphs` — so we hide the full
/// `chain_continuation_prefix` byte sequence directly. See
/// `bugs.md::trailing_li_continuation_indent_visible_and_eatable` and
/// `bugs.md::alternating_chain_synth_leaf_visible_indent`.
fn hide_synth_continuation_indent(leaf: &mut RenderBlock, bytes: &[u8]) {
    hide_chain_continuation_prefix(leaf, bytes);
}

/// Hide the canonical [`crate::analysis::chain_continuation_prefix`]
/// byte sequence on every line of `leaf`'s source range.
///
/// This is the unified continuation-indent hide that subsumes the
/// per-item `hide_item_continuation_indent` for cases where the chain
/// alternates LI / BQ entries. The per-item pass only finds *leading*
/// space runs, so on a line shaped `   > ` + `   ` + content (LI / BQ
/// / LI prefix) it covers only the first `   ` and misses the trailing
/// LI indent after the BQ marker. By comparing each line's bytes
/// against the *full* chain-prefix string built from
/// `chain_continuation_prefix`, we hide every byte of the prefix —
/// leading LI indent, BQ markers, and the LI indent that sits
/// *between* / *after* BQ markers — in one pass.
///
/// Lines whose bytes don't match the canonical prefix are left alone
/// (they're not continuation lines — could be the line carrying the
/// item's marker or a code-block content line that happens to live
/// inside the leaf range).
fn hide_chain_continuation_prefix(leaf: &mut RenderBlock, bytes: &[u8]) {
    if leaf.containers.is_empty() {
        return;
    }
    // Build the canonical chain prefix once. We use the analysis-side
    // helper so render-side and source-side prefix shape can never
    // disagree.
    let analysis_chain: Vec<crate::analysis::EnclosingContainer> = leaf
        .containers
        .iter()
        .map(|c| match c {
            Container::BlockQuote { .. } => crate::analysis::EnclosingContainer::BlockQuote {
                range: 0..0, // range isn't read by chain_continuation_prefix
            },
            Container::ListItem {
                marker_byte_len, ..
            } => crate::analysis::EnclosingContainer::ListItem(crate::analysis::ListItemContext {
                list_kind: crate::syntax::ListKind::Unordered,
                item_index: 0,
                item_range: 0..0,
                // Synthesize a marker_range whose width equals
                // marker_byte_len; only the width matters here.
                marker_range: 0..*marker_byte_len,
                task: None,
            }),
        })
        .collect();
    let prefix = crate::analysis::chain_continuation_prefix(&analysis_chain);
    let prefix_bytes = prefix.as_bytes();
    if prefix_bytes.is_empty() {
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

        // Match the canonical prefix byte-for-byte against the line.
        // For LI segments (runs of spaces) and BQ segments (`> `) the
        // prefix bytes are deterministic, so a literal byte-equal
        // check is sufficient and avoids the per-segment loop's
        // off-by-one footguns.
        if line_content_end - p >= prefix_bytes.len()
            && bytes[p..p + prefix_bytes.len()] == *prefix_bytes
        {
            leaf.hidden_ranges.push(p..p + prefix_bytes.len());
        }

        p = line_end_inclusive_nl.min(leaf_range.end);
        if p < leaf_range.end && bytes[p] == b'\n' {
            p += 1;
        } else if p == line_end_inclusive_nl && p == line_content_end {
            break;
        }
    }
}

/// **The canonical "container chain at byte X" query for the render
/// walker.** Both synth-leaf emission sites (`inject_empty_paragraphs`
/// for trailing / inter-block synths, and `render_blockquote` for
/// deferred-pair synths) go through this helper, which delegates to
/// [`crate::analysis::enclosing_containers_at`] so render-side and
/// cursor-side queries can never disagree.
///
/// New synth-leaf emission paths (e.g. for code blocks in deeper
/// nesting cases) should go through this function — never construct
/// a synth's `containers` chain by hand or reuse a "built-up-while-
/// walking" chain. The walking chain misses containers whose ranges
/// were extended through trailing pair shapes by `effective_node_end`,
/// which is the exact case synths land in.
///
/// Translation from analysis-side `EnclosingContainer` to render-spec
/// `Container`:
///   - `BlockQuote { range }` → `Container::BlockQuote { cursor_inside }`
///     where `cursor_inside = cursor.overlaps(&range)`.
///   - `ListItem(ctx)` → `Container::ListItem { cursor_inside, kind,
///     marker_byte_len, list_max_marker_text }`. `kind` and
///     `list_max_marker_text` need the parent list (sibling count for
///     ordered, marker char for unordered) — we look up the smallest
///     `List` node in `tree` that owns this item via its
///     `marker_range`.
fn chain_for_position(
    tree: &[SyntaxNode],
    source: &str,
    cursor: CursorRange,
    position: usize,
) -> Vec<Container> {
    let chain = crate::analysis::enclosing_containers_at(source, position);
    translate_enclosing_chain(tree, source, cursor, chain)
}

fn translate_enclosing_chain(
    tree: &[SyntaxNode],
    source: &str,
    cursor: CursorRange,
    chain: crate::analysis::EnclosingChain,
) -> Vec<Container> {
    let bytes = source.as_bytes();
    chain
        .into_iter()
        .map(|c| match c {
            crate::analysis::EnclosingContainer::BlockQuote { range } => {
                let trimmed_end = crate::analysis::effective_range_end(&range, bytes);
                Container::BlockQuote {
                    cursor_inside: cursor.overlaps(&(range.start..trimmed_end)),
                }
            }
            crate::analysis::EnclosingContainer::ListItem(ctx) => {
                let (max_marker_text, item_kind) = list_context_for_item(tree, source, &ctx);
                let trimmed_end = crate::analysis::effective_range_end(&ctx.item_range, bytes);
                Container::ListItem {
                    cursor_inside: cursor.overlaps(&(ctx.item_range.start..trimmed_end)),
                    kind: item_kind,
                    marker_byte_len: ctx.marker_width(),
                    list_max_marker_text: max_marker_text,
                }
            }
        })
        .collect()
}

/// Find the smallest `List` in `tree` that contains a `ListItem` whose
/// `marker_range` matches `ctx.marker_range`, and return its
/// `list_max_marker_text` plus this item's `ListItemKind`. Falls back
/// to a single-item synthesis if no match is found (the analysis chain
/// guarantees the item exists, so this fallback is paranoia).
fn list_context_for_item(
    tree: &[SyntaxNode],
    source: &str,
    ctx: &crate::analysis::ListItemContext,
) -> (String, ListItemKind) {
    if let Some((list, kind, task)) = find_owning_list(tree, &ctx.marker_range) {
        let max_marker_text = compute_list_max_marker_text(list, kind);
        let item_kind = match kind {
            ListKind::Unordered => {
                ListItemKind::Unordered(marker_char(source, &ctx.marker_range), task)
            }
            ListKind::Ordered { start } => ListItemKind::Ordered {
                number: start + ctx.item_index as u64,
            },
        };
        (max_marker_text, item_kind)
    } else {
        let max_marker_text = match ctx.list_kind {
            ListKind::Unordered => "- ".to_string(),
            ListKind::Ordered { start } => format!("{}. ", start),
        };
        let item_kind = match ctx.list_kind {
            ListKind::Unordered => {
                ListItemKind::Unordered(marker_char(source, &ctx.marker_range), None)
            }
            ListKind::Ordered { start } => ListItemKind::Ordered {
                number: start + ctx.item_index as u64,
            },
        };
        (max_marker_text, item_kind)
    }
}

fn find_owning_list<'a>(
    nodes: &'a [SyntaxNode],
    item_marker_range: &Range<usize>,
) -> Option<(&'a SyntaxNode, ListKind, Option<bool>)> {
    for node in nodes {
        if let NodeKind::List { kind } = &node.kind {
            for child in &node.children {
                if let NodeKind::ListItem { marker_range, task } = &child.kind
                    && marker_range == item_marker_range
                {
                    return Some((node, *kind, *task));
                }
            }
        }
        if let Some(found) = find_owning_list(&node.children, item_marker_range) {
            return Some(found);
        }
    }
    None
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
                    kind: ListItemKind::Unordered(b'-', None),
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
    fn cursor_in_paragraph_after_list_does_not_mark_last_item_inside() {
        // Pulldown's List/Item ranges sometimes extend past the
        // last content line into trailing structural separators —
        // the cursor parked in a post-list empty paragraph would
        // then match the item's range via boundary equality and
        // render the item as if focused. The item's `cursor_inside`
        // must be false for any cursor strictly past the trimmed
        // content extent.
        let src = "- foo\n\n";
        // Cursor at byte 7 (end of buffer, in the trailing empty
        // paragraph row).
        let spec = render_with_cursor(src, 7);
        let item = spec
            .blocks
            .iter()
            .find(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
            .expect("list item leaf");
        match item.containers.first().unwrap() {
            Container::ListItem { cursor_inside, .. } => {
                assert!(
                    !*cursor_inside,
                    "cursor in trailing empty paragraph should NOT mark the last item as inside",
                );
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn cursor_two_paragraphs_after_list_does_not_mark_last_item_inside() {
        let src = "- foo\n\n\n\n";
        // Cursor at end of buffer — multiple empty paragraphs after
        // the list.
        let spec = render_with_cursor(src, 9);
        let item = spec
            .blocks
            .iter()
            .find(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
            .expect("list item leaf");
        match item.containers.first().unwrap() {
            Container::ListItem { cursor_inside, .. } => {
                assert!(
                    !*cursor_inside,
                    "cursor far past the list should not mark the last item as inside",
                );
            }
            _ => unreachable!(),
        }
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

    // ---- Inline code ----------------------------------------------------

    #[test]
    fn inline_code_hides_backticks_when_cursor_outside() {
        let src = "see `foo()` end";
        let spec = render_with_cursor(src, 0);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        // Backticks at 4..5 and 10..11.
        assert!(block.has_hidden_range(4..5));
        assert!(block.has_hidden_range(10..11));
        // Content range 5..10 carries a code-styled inline run.
        assert!(
            block
                .inlines
                .iter()
                .any(|r| r.source_range == (5..10) && r.style.code),
            "expected code-styled inline run for the span content",
        );
    }

    #[test]
    fn inline_code_dims_backticks_when_cursor_inside() {
        let src = "x `code` y";
        // Cursor inside "code".
        let spec = render_with_cursor(src, 4);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert!(block.has_dimmed_range(2..3));
        assert!(block.has_dimmed_range(7..8));
    }

    // ---- Thematic break -------------------------------------------------

    #[test]
    fn thematic_break_emits_its_own_block_kind() {
        let src = "above\n\n---\n\nbelow";
        let spec = render_with_cursor(src, 0);
        let rule = spec
            .blocks
            .iter()
            .find(|b| matches!(b.kind, BlockKind::ThematicBreak))
            .expect("thematic break block");
        // Source range covers `---` (the trailing `\n` is trimmed by
        // `inject_empty_paragraphs`).
        assert!(rule.source_range.end - rule.source_range.start >= 3);
    }

    #[test]
    fn thematic_break_hides_source_when_cursor_outside() {
        let src = "above\n\n---\n\nbelow";
        // Cursor in "below".
        let cursor = src.find("below").unwrap() + 1;
        let spec = render_with_cursor(src, cursor);
        let rule = spec
            .blocks
            .iter()
            .find(|b| matches!(b.kind, BlockKind::ThematicBreak))
            .expect("thematic break block");
        let r = rule.source_range.clone();
        assert!(rule.has_hidden_range(r));
    }

    #[test]
    fn thematic_break_dims_source_when_cursor_inside() {
        let src = "---\n";
        // Cursor at byte 1 (between `-` and `-`).
        let spec = render_with_cursor(src, 1);
        let rule = spec
            .blocks
            .iter()
            .find(|b| matches!(b.kind, BlockKind::ThematicBreak))
            .expect("thematic break block");
        // The dimmed inline run covers the trimmed range (no trailing
        // newline). Verify the run exists with the dimmed style.
        assert!(
            rule.inlines.iter().any(|r| r.style.dimmed),
            "expected dimmed inline run when cursor is on the rule",
        );
    }

    // ---- Links ----------------------------------------------------------

    #[test]
    fn link_hides_brackets_and_url_when_cursor_outside() {
        let src = "see [docs](u) end";
        // Cursor at byte 0 — outside the link span.
        let spec = render_with_cursor(src, 0);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        // `[` at 4..5; `](u)` at 9..13.
        assert!(block.has_hidden_range(4..5));
        assert!(block.has_hidden_range(9..13));
    }

    #[test]
    fn link_emits_link_styled_inline_run_for_text() {
        let src = "[docs](u)";
        let spec = render_with_cursor(src, 99);
        let block = &spec.blocks[0];
        assert!(
            block.inlines.iter().any(|r| r.style.link),
            "expected link-styled inline run for link text",
        );
    }

    #[test]
    fn link_dims_brackets_when_cursor_inside() {
        let src = "[docs](u)";
        // Cursor inside "docs".
        let spec = render_with_cursor(src, 3);
        let block = &spec.blocks[0];
        assert!(block.has_dimmed_range(0..1));
        assert!(block.has_dimmed_range(5..9));
    }

    // ---- Task list items -----------------------------------------------

    #[test]
    fn task_item_carries_task_state_in_container() {
        let src = "- [ ] todo\n- [x] done\n";
        let spec = render_with_cursor(src, 0);
        let kinds: Vec<_> = spec
            .blocks
            .iter()
            .filter_map(|b| match b.containers.first() {
                Some(Container::ListItem { kind, .. }) => Some(*kind),
                _ => None,
            })
            .collect();
        assert_eq!(kinds.len(), 2);
        assert!(matches!(
            kinds[0],
            ListItemKind::Unordered(b'-', Some(false))
        ));
        assert!(matches!(
            kinds[1],
            ListItemKind::Unordered(b'-', Some(true))
        ));
    }

    #[test]
    fn task_item_hides_full_marker_when_cursor_outside() {
        // Cursor outside the item: bullet `- ` and task marker
        // `[ ] ` are both hidden so the line reads `todo` and a
        // checkbox glyph paints in the indent strip.
        let src = "- [ ] todo\n\nbody";
        let cursor = src.find("body").unwrap() + 1;
        let spec = render_with_cursor(src, cursor);
        let item = spec
            .blocks
            .iter()
            .find(|b| !b.containers.is_empty())
            .expect("list item leaf");
        // After `merge_hidden_ranges` collapses adjacent hides,
        // 0..2 (bullet) and 2..6 (`[ ] `) become 0..6.
        assert!(item.has_hidden_range(0..6));
        // And a marker overlay is queued for the indent strip.
        assert!(
            !item.marker_overlays.is_empty(),
            "expected a checkbox overlay when cursor is outside",
        );
    }

    #[test]
    fn mixed_task_and_plain_items_track_per_item_task_state() {
        // GFM allows intermingling task and non-task items in the
        // same list. Our parse + render keeps each item's state
        // independent — no auto-promotion or demotion.
        let src = "- [ ] todo\n- plain\n- [x] done\n";
        let spec = render_with_cursor(src, 0);
        let kinds: Vec<_> = spec
            .blocks
            .iter()
            .filter_map(|b| match b.containers.first() {
                Some(Container::ListItem { kind, .. }) => Some(*kind),
                _ => None,
            })
            .collect();
        assert_eq!(kinds.len(), 3);
        assert!(matches!(
            kinds[0],
            ListItemKind::Unordered(b'-', Some(false))
        ));
        assert!(matches!(kinds[1], ListItemKind::Unordered(b'-', None)));
        assert!(matches!(
            kinds[2],
            ListItemKind::Unordered(b'-', Some(true))
        ));
    }

    #[test]
    fn mixed_list_uses_widest_marker_text_for_indent() {
        // When any item in the list is a task, every sibling
        // (task or plain) gets the wider `"- [ ] "` indent so all
        // marker overlays right-align at the same content edge.
        let src = "- [ ] task\n- plain\n";
        let spec = render_with_cursor(src, 0);
        let plain_item = spec
            .blocks
            .iter()
            .find(|b| {
                matches!(
                    b.containers.first(),
                    Some(Container::ListItem {
                        kind: ListItemKind::Unordered(_, None),
                        ..
                    })
                )
            })
            .expect("plain item leaf");
        match plain_item.containers.first().unwrap() {
            Container::ListItem {
                list_max_marker_text,
                ..
            } => assert_eq!(
                list_max_marker_text, "- [ ] ",
                "plain items in mixed lists carry the wider indent",
            ),
            _ => unreachable!(),
        }
    }

    #[test]
    fn task_item_overlay_covers_full_chrome_when_cursor_inside() {
        // Cursor inside the task item: the marker chrome (bullet +
        // `[ ]`) is hidden from the shaped line and the marker
        // overlay's source_range covers the full chrome (0..6) so
        // the element layer's overlay-cursor path can place the
        // caret inside the brackets when the user navigates there.
        let src = "- [ ] todo\n";
        // Cursor at byte 8 — inside "todo".
        let spec = render_with_cursor(src, 8);
        let item = spec
            .blocks
            .iter()
            .find(|b| !b.containers.is_empty())
            .expect("list item leaf");
        assert!(
            item.has_hidden_range(0..6),
            "marker chrome stays hidden so content lands at the same column regardless of focus",
        );
        let overlay = item
            .marker_overlays
            .first()
            .expect("expected a marker overlay for the task item");
        assert_eq!(
            overlay.source_range,
            0..6,
            "overlay should cover the full chrome (bullet + brackets)",
        );
    }

    // ---- Backslash escapes / HTML entities ------------------------------

    fn first_substitution_for(block: &RenderBlock, range: Range<usize>) -> Option<&str> {
        block
            .substitutions
            .iter()
            .find(|s| s.source_range == range)
            .map(|s| s.display.as_str())
    }

    #[test]
    fn backslash_escape_substitutes_punctuation_when_cursor_outside() {
        let src = r"a\*b";
        let spec = render_with_cursor(src, 0);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert!(block.has_hidden_range(1..3));
        assert_eq!(first_substitution_for(block, 1..3), Some("*"));
        assert!(!block.has_dimmed_range(1..3));
    }

    #[test]
    fn backslash_escape_dims_when_cursor_inside() {
        // Cursor between the backslash and the `*` (byte 2) sits
        // inside the construct's [1..3) range — reveal raw bytes.
        let src = r"a\*b";
        let spec = render_with_cursor(src, 2);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert!(block.has_dimmed_range(1..3));
        assert!(first_substitution_for(block, 1..3).is_none());
    }

    #[test]
    fn backslash_escape_inside_inline_code_is_not_processed() {
        // `\*` inside backticks is verbatim per CommonMark §6.1; the
        // scanner must skip the inline code span entirely.
        let src = r"a `\*` b";
        let spec = render_with_cursor(src, 0);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        // No substitution for the in-code escape.
        assert!(first_substitution_for(block, 3..5).is_none());
    }

    #[test]
    fn backslash_escape_inside_fenced_code_is_not_processed() {
        let src = "```\nfn foo() { let x = r\"\\*\"; }\n```";
        let spec = render_with_cursor(src, 0);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::CodeBlock { .. }));
        // No substitutions on a code block at all.
        assert!(block.substitutions.is_empty());
    }

    #[test]
    fn entity_amp_substitutes_when_cursor_outside() {
        let src = "x &amp; y";
        let spec = render_with_cursor(src, 0);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert!(block.has_hidden_range(2..7));
        assert_eq!(first_substitution_for(block, 2..7), Some("&"));
    }

    #[test]
    fn entity_dims_when_cursor_inside() {
        let src = "x &amp; y";
        let spec = render_with_cursor(src, 4);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert!(block.has_dimmed_range(2..7));
        assert!(first_substitution_for(block, 2..7).is_none());
    }

    #[test]
    fn numeric_decimal_entity_substitutes() {
        let src = "&#169;";
        let spec = render_with_cursor(src, 999);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert_eq!(first_substitution_for(block, 0..6), Some("©"));
    }

    #[test]
    fn unknown_named_entity_does_not_substitute() {
        let src = "&banana;";
        let spec = render_with_cursor(src, 999);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert!(block.substitutions.iter().all(|s| s.source_range != (0..8)));
    }

    // ---- Math -----------------------------------------------------------

    #[test]
    fn inline_math_outside_cursor_emits_math_overlay_only() {
        // Cursor outside the construct: render emits *only* the
        // overlay (no hidden range, no fallback inline runs). The
        // element layer chooses what to do at paint time —
        // substitution + paint on typeset success, or
        // dim/mono fallback runs on failure. Routing the source-
        // hide through the element layer is what enables the
        // failed-overlay fallback to show raw LaTeX instead of a
        // blank gap.
        let src = "x $a^2$ y";
        let spec = render_with_cursor(src, 0);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        let overlay = block
            .math_overlays
            .iter()
            .find(|o| o.source_range == (2..7))
            .expect("math overlay emitted for cursor-outside inline math");
        assert_eq!(overlay.content_range, 3..6);
        assert!(!overlay.display_style, "$..$ is text-style, not display");
        // Render must not pre-hide — that's the element layer's
        // call.
        assert!(
            !block
                .hidden_ranges
                .iter()
                .any(|r| r.start == 2 && r.end == 7),
            "render must not add the math.range hide; element does it on success"
        );
        // And no competing inline runs for the math source — the
        // element layer only adds those on typeset failure.
        assert!(
            !block
                .inlines
                .iter()
                .any(|r| r.source_range == (3..6) && r.style.code),
            "no inner-LaTeX code run from render when overlay is in effect"
        );
    }

    #[test]
    fn inline_math_dims_delimiters_when_cursor_inside() {
        let src = "x $a^2$ y";
        let spec = render_with_cursor(src, 4);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert!(block.has_dimmed_range(2..3));
        assert!(block.has_dimmed_range(6..7));
        // Cursor inside falls back to the dim/mono path: no overlay
        // should be emitted (the source is shaping normally).
        assert!(
            block.math_overlays.is_empty(),
            "no overlay emitted when cursor is inside the math construct"
        );
        // Inner LaTeX shapes as code-styled mono text, as before.
        assert!(
            block
                .inlines
                .iter()
                .any(|r| r.source_range == (3..6) && r.style.code),
            "inner LaTeX shapes mono in cursor-inside fallback"
        );
    }

    #[test]
    fn display_math_renders_with_double_dollar_delimiters() {
        // `$$a + b$$` standalone promotes to a DisplayMath block —
        // both `$$` pairs hide unconditionally so the typeset math
        // can paint without delimiter chrome shaping into the line.
        let src = "$$a + b$$";
        let spec = render_with_cursor(src, 999);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::DisplayMath { .. }));
        assert!(block.has_hidden_range(0..2));
        assert!(block.has_hidden_range(7..9));
    }

    #[test]
    fn display_math_promotes_paragraph_with_single_math_child() {
        let src = "$$x^2$$";
        let spec = render_with_cursor(src, 999);
        match spec.blocks[0].kind {
            BlockKind::DisplayMath {
                ref content_range,
                edit_mode,
            } => {
                assert_eq!(content_range.clone(), 2..5);
                assert!(!edit_mode, "cursor outside the math should be display mode");
            }
            ref other => panic!("expected DisplayMath block, got {other:?}"),
        }
    }

    #[test]
    fn display_math_does_not_promote_mixed_paragraph() {
        // `before $$math$$ after` keeps the inline path: the
        // paragraph has Text + DisplayMath + Text children, which
        // is *not* a sole-DisplayMath promotion case.
        let src = "before $$x^2$$ after";
        let spec = render_with_cursor(src, 0);
        let has_display_math_block = spec
            .blocks
            .iter()
            .any(|b| matches!(b.kind, BlockKind::DisplayMath { .. }));
        assert!(
            !has_display_math_block,
            "mixed paragraph should keep inline rendering"
        );
    }

    #[test]
    fn display_math_enters_edit_mode_when_cursor_strictly_inside() {
        // Cursor at byte 4 sits strictly inside `$$x^2$$` (between
        // 2 and 5) → edit_mode = true.
        let src = "$$x^2$$";
        let spec = render_with_cursor(src, 4);
        match spec.blocks[0].kind {
            BlockKind::DisplayMath { edit_mode, .. } => assert!(edit_mode),
            ref other => panic!("expected DisplayMath, got {other:?}"),
        }
    }

    #[test]
    fn display_math_enters_edit_mode_when_cursor_touches_boundary() {
        // A cursor touching either `$$` fence — at `math.start` or
        // `math.end` — counts as inside (inclusive overlap). This
        // is what makes click-on-typeset-math switch to edit mode:
        // every source byte is hidden in display mode, so the
        // shaped line is zero-width and every click on the math
        // hits display column 0 → math.start. Inclusive overlap
        // turns that click into an edit-mode entry.
        let src = "$$x^2$$";
        for cursor in [0usize, 7] {
            let spec = render_with_cursor(src, cursor);
            match spec.blocks[0].kind {
                BlockKind::DisplayMath { edit_mode, .. } => {
                    assert!(edit_mode, "cursor at boundary {cursor} should be edit mode")
                }
                ref other => panic!("expected DisplayMath, got {other:?}"),
            }
        }
    }

    #[test]
    fn display_math_returns_to_display_mode_when_cursor_is_outside() {
        // Cursor on a *different* byte (past the math) puts the
        // math back into display mode. With math at 0..7 in
        // `$$x^2$$\n\nrest`, cursor at byte 9 (inside `rest`) is
        // outside the math.
        let src = "$$x^2$$\n\nrest";
        let spec = render_with_cursor(src, 11);
        let math_block = spec
            .blocks
            .iter()
            .find(|b| matches!(b.kind, BlockKind::DisplayMath { .. }))
            .expect("math block");
        match math_block.kind {
            BlockKind::DisplayMath { edit_mode, .. } => assert!(!edit_mode),
            ref other => panic!("expected DisplayMath, got {other:?}"),
        }
    }

    #[test]
    fn display_math_edit_mode_keeps_fences_visible_for_navigation() {
        // Edit mode (cursor strictly inside) MUST leave the `$$`
        // delimiter bytes shaping into the line — hiding them would
        // collapse 4 bytes out of the display, breaking click
        // hit-testing and selection geometry over the fences.
        let src = "$$x^2$$";
        let spec = render_with_cursor(src, 4);
        let block = &spec.blocks[0];
        assert!(
            !block.has_hidden_range(0..2),
            "opening `$$` must shape (dimmed, not hidden) in edit mode for navigation"
        );
        assert!(
            !block.has_hidden_range(5..7),
            "closing `$$` must shape (dimmed, not hidden) in edit mode for navigation"
        );
        assert!(block.has_dimmed_range(0..2));
        assert!(block.has_dimmed_range(5..7));
    }

    #[test]
    fn display_math_display_mode_hides_entire_source_range() {
        // Display mode (cursor outside): every byte of the
        // construct hides so the typeset overlay doesn't paint on
        // top of un-hidden source text. Pre-fix this regression
        // saw `x^2` shape underneath the rendered math.
        let src = "$$x^2$$";
        let spec = render_with_cursor(src, 999);
        let block = &spec.blocks[0];
        assert!(
            block.has_hidden_range(0..7),
            "full math range hidden in display mode"
        );
        // Inner content range specifically — a sub-range of the
        // full hide.
        assert!(block.has_hidden_range(2..5));
    }

    // ---- Images ---------------------------------------------------------

    #[test]
    fn inline_image_outside_cursor_emits_image_overlay_only() {
        // Cursor outside the construct: render emits the overlay
        // record only (the element layer does the substitution +
        // paint). No hidden range, no fallback inline runs.
        let src = "x ![logo](u.png) y";
        let spec = render_with_cursor(src, 0);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        let overlay = block
            .image_overlays
            .iter()
            .find(|o| o.source_range == (2..16))
            .expect("image overlay emitted for cursor-outside inline image");
        assert_eq!(overlay.alt_range, 4..8);
        assert_eq!(overlay.dest_url, "u.png");
        // Render must not pre-hide — that's the element layer's
        // call (substitution covers the bytes on success; fallback
        // shows the raw source on failure).
        assert!(
            !block
                .hidden_ranges
                .iter()
                .any(|r| r.start == 2 && r.end == 16),
            "render must not add the image.range hide; element does it on success"
        );
    }

    #[test]
    fn inline_image_dims_delimiters_when_cursor_inside() {
        // Cursor inside the construct → fallback: dim `![` and
        // `](url)`, alt text shapes in normal style so the user
        // can edit. No overlay emitted (the typeset/load path is
        // bypassed).
        let src = "x ![logo](u.png) y";
        // Cursor at byte 6 sits inside "logo" (the alt text).
        let spec = render_with_cursor(src, 6);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert!(block.has_dimmed_range(2..4)); // `![`
        assert!(block.has_dimmed_range(8..16)); // `](u.png)`
        assert!(
            block.image_overlays.is_empty(),
            "no overlay emitted when cursor is inside the image construct"
        );
    }

    #[test]
    fn image_promotes_paragraph_with_single_image_child() {
        // `![alt](url)` standalone promotes to an Image block —
        // mirrors the DisplayMath promotion rule.
        let src = "![logo](u.png)";
        let spec = render_with_cursor(src, 999);
        match &spec.blocks[0].kind {
            BlockKind::Image {
                alt_range,
                dest_url,
                edit_mode,
            } => {
                assert_eq!(alt_range.clone(), 2..6);
                assert_eq!(dest_url, "u.png");
                assert!(
                    !*edit_mode,
                    "cursor outside the construct should be display mode"
                );
            }
            other => panic!("expected Image block, got {other:?}"),
        }
    }

    #[test]
    fn image_does_not_promote_mixed_paragraph() {
        // `before ![alt](u) after` keeps the inline path — the
        // paragraph has Text + Image + Text children, not a sole
        // image. Same rule as DisplayMath's mixed-paragraph case.
        let src = "before ![logo](u.png) after";
        let spec = render_with_cursor(src, 0);
        let has_image_block = spec
            .blocks
            .iter()
            .any(|b| matches!(b.kind, BlockKind::Image { .. }));
        assert!(
            !has_image_block,
            "mixed paragraph should keep inline rendering"
        );
    }

    #[test]
    fn image_enters_edit_mode_when_cursor_strictly_inside() {
        // Cursor at byte 4 sits strictly inside `![logo](u.png)`
        // (between 0 and 14) → edit_mode = true.
        let src = "![logo](u.png)";
        let spec = render_with_cursor(src, 4);
        match spec.blocks[0].kind {
            BlockKind::Image { edit_mode, .. } => assert!(edit_mode),
            ref other => panic!("expected Image block, got {other:?}"),
        }
    }

    #[test]
    fn image_enters_edit_mode_when_cursor_touches_boundary() {
        // Cursor at either edge of the construct (byte 0 or byte
        // 14 in `![logo](u.png)`) counts as inside. Same inclusive-
        // overlap rule as DisplayMath, for the same click-to-edit
        // affordance.
        let src = "![logo](u.png)";
        for cursor in [0usize, 14] {
            let spec = render_with_cursor(src, cursor);
            match spec.blocks[0].kind {
                BlockKind::Image { edit_mode, .. } => {
                    assert!(edit_mode, "cursor at boundary {cursor} should be edit mode")
                }
                ref other => panic!("expected Image block, got {other:?}"),
            }
        }
    }

    #[test]
    fn image_returns_to_display_mode_when_cursor_outside() {
        let src = "![logo](u.png)\n\nrest";
        let spec = render_with_cursor(src, 18); // cursor inside "rest"
        let image_block = spec
            .blocks
            .iter()
            .find(|b| matches!(b.kind, BlockKind::Image { .. }))
            .expect("image block");
        match image_block.kind {
            BlockKind::Image { edit_mode, .. } => assert!(!edit_mode),
            ref other => panic!("expected Image block, got {other:?}"),
        }
    }

    #[test]
    fn image_edit_mode_keeps_delimiters_visible_for_navigation() {
        // Edit mode must leave `![` and `](url)` shaping into the
        // line — hiding them would collapse bytes out of the
        // display, breaking click hit-testing over the markdown
        // source. Same invariant as DisplayMath edit mode.
        let src = "![logo](u.png)";
        let spec = render_with_cursor(src, 4);
        let block = &spec.blocks[0];
        assert!(
            !block.has_hidden_range(0..2),
            "`![` must shape (dimmed, not hidden) in edit mode for navigation"
        );
        assert!(
            !block.has_hidden_range(6..14),
            "`](url)` must shape (dimmed, not hidden) in edit mode for navigation"
        );
        assert!(block.has_dimmed_range(0..2));
        assert!(block.has_dimmed_range(6..14));
    }

    #[test]
    fn image_display_mode_pre_stages_fallback_runs() {
        // Display mode (cursor outside): render pre-stages the
        // fallback shape (dim delimiters + visible alt text) on the
        // raw bytes but does NOT hide them — the element layer
        // commits a hide only when it has a paintable image. This
        // way, a failed load falls through to the dim-delim + alt-
        // text shape (the same visual treatment as cursor-inside
        // edit mode) so the user can see what went wrong, rather
        // than staring at a blank row.
        let src = "![logo](u.png)";
        let spec = render_with_cursor(src, 999);
        let block = &spec.blocks[0];
        assert!(
            !block.has_hidden_range(0..14),
            "render must not pre-hide; the element layer owns the hide on success/loading"
        );
        assert!(block.has_dimmed_range(0..2));
        assert!(block.has_dimmed_range(6..14));
    }

    #[test]
    fn escapes_do_not_apply_inside_image_url() {
        // The destination portion `](url)` is verbatim — a `\` in
        // the URL should not trigger a CommonMark escape
        // substitution. (Same treatment as link destinations.)
        let src = r"![alt](foo\bar)";
        let spec = render_with_cursor(src, 999);
        let block = &spec.blocks[0];
        // `\b` is not a CommonMark escape (b isn't punctuation), so
        // the scanner wouldn't substitute anyway — but the
        // verbatim coverage guards future-us against picking the
        // wrong punctuation.
        assert!(
            !block
                .substitutions
                .iter()
                .any(|s| s.source_range.start >= 6 && s.source_range.end <= 14),
            "no markdown-level substitution inside image destination"
        );
    }

    #[test]
    fn escapes_do_not_apply_inside_math_source() {
        // `$\frac{a}{b}$` — `\f` is not a CommonMark escape (`f` isn't
        // ASCII punctuation), so the scanner wouldn't substitute it
        // anyway. Test the case that matters: `\$` inside math should
        // NOT become a literal `$` substitution because math content
        // is verbatim from CommonMark's perspective.
        let src = r"$a\$b$";
        let spec = render_with_cursor(src, 999);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        // The `\$` at bytes 2..4 inside `$a\$b$` should not produce
        // a substitution. (Pulldown's math-delimiter parser sees the
        // construct as `$a\$b$`, with `\$` escaped at the LaTeX
        // level — that's RaTeX's concern, not ours.)
        assert!(
            !block.substitutions.iter().any(|s| s.source_range == (2..4)),
            "no markdown-level substitution inside math content"
        );
    }

    #[test]
    fn escape_inside_strong_does_not_break_emphasis_styling() {
        // `xx **a\*b**` keeps the Strong span intact while the inner
        // `\*` renders as a literal `*`. Cursor on the leading `xx`
        // (byte 1) sits *outside* the Strong (Strong range is 3..11)
        // so its delimiters hide; the `\*` at bytes 6..8 is also
        // outside the cursor's overlap so it substitutes.
        let src = r"xx **a\*b**";
        let spec = render_with_cursor(src, 1);
        let block = find_block(&spec, |b| matches!(b.kind, BlockKind::Paragraph));
        assert!(block.has_hidden_range(3..5));
        assert!(block.has_hidden_range(9..11));
        assert_eq!(first_substitution_for(block, 6..8), Some("*"));
    }
}
