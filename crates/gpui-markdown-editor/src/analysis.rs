//! Structural analysis of the markdown buffer.
//!
//! This module is the single source of truth for the byte-level facts
//! the rest of the editor consults: where fenced code blocks live,
//! where structural paragraph-break pairs live, which positions are
//! forbidden cursor landings, and what container prefix (currently:
//! blockquote markers) introduces the line at a given offset.
//!
//! `update.rs` queries these primitives to enforce buffer invariants;
//! `editor.rs` queries them to route Enter / Shift+Enter through a
//! context-aware insertion path; the renderer queries them transitively
//! through `update::enforce_invariants`. Keeping the primitives in one
//! file means a future construct (lists) extends one place — the soft-
//! break rule, the forbidden-position rule, and the active-prefix rule
//! all read the same scan.
//!
//! # Fence-containment via pulldown
//!
//! "Is byte X inside a fenced code block?" is the dominant containment
//! question in this module — it gates the soft-break rule, the
//! forbidden-position predicate, every cursor-driven Enter / Backspace
//! / Delete branch, the auto-close edit, and the render walker's
//! synth-paragraph suppression. We answer it by walking pulldown's
//! parse tree (`crate::parser::parse`), not by scanning the raw bytes:
//! the byte scanner's `count_line_markers` only tolerates 3 spaces
//! between consecutive `>` markers, so a fence sitting inside an
//! `[LI, LI, BQ]` chain (with 4+ spaces of LI indent before the inner
//! `> `) is invisible to a byte scan but visible to pulldown. Routing
//! every fence query through pulldown gives a single source of truth
//! across all chain depths.
//!
//! # Pairs model recap
//!
//! `\n[prefix]\n[prefix]` — the depth-D analog of `\n\n` — is the
//! atomic structural paragraph break. `[prefix]` is `> ` repeated D
//! times. The two halves of a pair are visually one row (a
//! paragraph_gap) and the byte right between them is a forbidden
//! cursor position. See `update.rs` module docs for the full rationale.

use std::ops::Range;

// ---------------------------------------------------------------------------
// Fenced code block scanning
// ---------------------------------------------------------------------------

/// One fenced code block's structural extent, paired with whether the
/// block has a clean closer.
///
/// Returned by [`fenced_code_blocks`]. The `range` always covers the
/// opening fence line, the inner content, and the closing fence line
/// (with its trailing `\n`) or end-of-source for an unterminated block;
/// `terminated` distinguishes the two so the cursor query in
/// [`is_in_fenced_code`] knows whether the byte sitting at `range.end`
/// is "after the closer" (terminated → outside) or "at EOF inside the
/// still-open construct" (unterminated → inside).
///
/// `opener_fence_char` and `opener_fence_len` describe the opening
/// fence run — used by [`auto_close_fence_edit`] to synthesize a
/// matching closer on Enter inside an unterminated block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FencedCodeBlock {
    pub range: Range<usize>,
    pub terminated: bool,
    pub opener_fence_char: u8,
    pub opener_fence_len: usize,
}

/// Find every fenced code block's full byte range — inclusive of the
/// opening fence line, the inner content, and the closing fence line
/// (or end-of-source for an unterminated block) — paired with whether
/// the block was actually closed.
///
/// **Implementation:** walks pulldown's parse tree
/// (`crate::parser::parse`). Routing through the parser instead of a
/// byte scanner is the single source of truth for "is byte X inside a
/// fence" across all chain depths — see the module docs for why a
/// raw byte scan diverges in `[LI, LI, BQ]` chains.
///
/// Pulldown ranges a `CodeBlock` from the opener fence's first byte
/// through the closing fence line's terminating `\n` (or end-of-source
/// for an unterminated block); we project that range and the
/// delimiter count (1 for unterminated, 2 for terminated) directly.
pub fn fenced_code_blocks(markdown: &str) -> Vec<FencedCodeBlock> {
    let tree = crate::parser::parse(markdown);
    fenced_code_blocks_in_tree(&tree, markdown.as_bytes())
}

/// Variant of [`fenced_code_blocks`] for callers that already hold a
/// parse tree (typically a multi-pass `enforce_invariants` step
/// sharing one parse across passes — see `update::enforce_invariants`).
pub fn fenced_code_blocks_in_tree(
    tree: &[crate::syntax::SyntaxNode],
    bytes: &[u8],
) -> Vec<FencedCodeBlock> {
    let mut out = Vec::new();
    collect_fenced_code_blocks(tree, bytes, &mut out);
    out
}

fn collect_fenced_code_blocks(
    nodes: &[crate::syntax::SyntaxNode],
    bytes: &[u8],
    out: &mut Vec<FencedCodeBlock>,
) {
    for node in nodes {
        if let crate::syntax::NodeKind::CodeBlock {
            delimiter_ranges, ..
        } = &node.kind
        {
            // Pulldown also emits `CodeBlock` for indented code blocks.
            // We only model fenced blocks here — `parser.rs` collapses
            // indented blocks to `Paragraph`, so any `CodeBlock` node
            // we see is fenced. Read the opener char from
            // `delimiter_ranges[0]`'s first byte (`` ` `` or `~`).
            let (fence_char, fence_len) = delimiter_ranges
                .first()
                .and_then(|r| {
                    let len = r.end.saturating_sub(r.start);
                    bytes.get(r.start).copied().map(|c| (c, len))
                })
                .unwrap_or((b'`', 0));
            out.push(FencedCodeBlock {
                range: node.range.clone(),
                terminated: delimiter_ranges.len() == 2,
                opener_fence_char: fence_char,
                opener_fence_len: fence_len,
            });
        }
        collect_fenced_code_blocks(&node.children, bytes, out);
    }
}

/// Range-only projection of [`fenced_code_blocks`]. Most callers only
/// need the spans (e.g. to skip over code content while scanning for
/// soft-break candidates); cursor-position queries that need to
/// distinguish "at the end of an unterminated block" from "after a
/// clean closer" should use [`is_in_fenced_code`] (or call the
/// state-aware version directly).
pub fn fenced_code_ranges(markdown: &str) -> Vec<Range<usize>> {
    fenced_code_blocks(markdown)
        .into_iter()
        .map(|b| b.range)
        .collect()
}

/// Variant of [`fenced_code_ranges`] for callers that already hold a
/// parse tree.
pub fn fenced_code_ranges_in_tree(
    tree: &[crate::syntax::SyntaxNode],
    bytes: &[u8],
) -> Vec<Range<usize>> {
    fenced_code_blocks_in_tree(tree, bytes)
        .into_iter()
        .map(|b| b.range)
        .collect()
}

/// `true` if byte index `p` falls inside any fenced code block (opener,
/// content, or closer). The cursor at the EOF position of an
/// **unterminated** fence is treated as inside the construct — there's
/// no closer to sit "after", and the user typing more content there is
/// still composing inside the code block. For terminated blocks the
/// byte right after the closer's trailing `\n` (i.e. exactly at
/// `range.end`) is *outside*, so a paragraph immediately following a
/// closed fence doesn't get classified as code.
pub fn is_in_fenced_code(markdown: &str, p: usize) -> bool {
    fenced_code_blocks(markdown)
        .iter()
        .any(|b| p >= b.range.start && (p < b.range.end || (!b.terminated && p == b.range.end)))
}

/// Variant of [`is_in_fenced_code`] for callers that already have a
/// `Vec<FencedCodeBlock>` (or `Vec<Range<usize>>` via [`fenced_code_ranges`])
/// in hand — avoids re-parsing per query.
pub fn is_in_fenced_code_blocks(blocks: &[FencedCodeBlock], p: usize) -> bool {
    blocks
        .iter()
        .any(|b| p >= b.range.start && (p < b.range.end || (!b.terminated && p == b.range.end)))
}

pub(crate) fn is_in_ranges(p: usize, ranges: &[Range<usize>]) -> bool {
    ranges.iter().any(|r| p >= r.start && p < r.end)
}

// ---------------------------------------------------------------------------
// Forbidden-position detection
// ---------------------------------------------------------------------------

/// Is byte index `p` a forbidden cursor position?
///
/// Two flavors of forbidden:
///
/// 1. **Structural-pair interior.** `p` sits strictly inside a
///    `\n\n` run (top-level) or its depth-D blockquote analog
///    `\n[> …]\n[> …]`. These pair-shaped runs collapse to one
///    `paragraph_gap` visually so the bytes between the two
///    boundary `\n`s have nowhere to land. Inside a fenced code
///    block the same byte pattern is just a blank line of code, so
///    those are exempt. Hard breaks (`  \n` / `\\\n`) are
///    in-paragraph content and exempt regardless of code-block
///    context.
///
/// 2. **List-indent interior.** `p` sits strictly inside the
///    leading hidden bytes of a list-item line — either the
///    cumulative ancestor indent + this item's marker on a marker
///    line, or the cumulative ancestor indent on a continuation
///    line. The renderer hides those bytes from the shaped line so
///    a cursor strictly inside them has no visible position to
///    land on; treating them as forbidden makes arrow-key
///    navigation skip over the indent in one step instead of
///    pausing at invisible byte positions.
pub fn is_forbidden_position(markdown: &str, p: usize) -> bool {
    let bytes = markdown.as_bytes();
    // Compute fence membership once and reuse it across the two
    // exemption checks below — each `is_in_fenced_code` call re-parses
    // the buffer.
    let in_fence = is_in_fenced_code(markdown, p);
    if is_paragraph_break_interior(bytes, p) && !in_fence {
        return true;
    }
    if is_list_indent_interior(markdown, p) {
        return true;
    }
    if is_chain_pair_interior(markdown, p) && !in_fence {
        return true;
    }
    false
}

/// Chain-aware analog of [`is_paragraph_break_interior`]: is byte `p`
/// strictly inside the canonical pair shape for the chain at `p`?
///
/// The canonical pair shape is
/// `\n{blank_prefix}\n{content_prefix}` — see [`chain_pair_shape`] for
/// the three-branch derivation. `is_paragraph_break_interior` only
/// walks BQ markers (`>` and ` `), so it misses pairs whose prefix
/// interleaves a list-item continuation indent with BQ markers (e.g.
/// `\n   > \n   > ` for a depth-1 BQ pair inside a `1. `-prefixed
/// item, or `\n   > \n   >    ` for an alternating chain ending in
/// LI). This helper closes that gap by computing the chain at `p`,
/// building the canonical chain-aware pair shape, and matching it
/// exactly.
///
/// Pure-LI chains (no BQ anywhere) collapse to the depth-0 shape
/// `\n\n[prefix]` and are already handled by the BQ-only walker plus
/// [`is_list_indent_interior`]; this helper only fires for chains
/// that contain at least one BQ.
pub fn is_chain_pair_interior(markdown: &str, p: usize) -> bool {
    if p == 0 || p > markdown.len() {
        return false;
    }
    let chain = enclosing_containers_at(markdown, p);
    if !chain
        .iter()
        .any(|c| matches!(c, EnclosingContainer::BlockQuote { .. }))
    {
        return false;
    }
    let (blank_prefix, content_prefix) = chain_pair_shape(&chain);
    let blank_bytes = blank_prefix.as_bytes();
    let content_bytes = content_prefix.as_bytes();
    let pair_len = 2 + blank_bytes.len() + content_bytes.len();
    if pair_len < 2 {
        return false;
    }
    let bytes = markdown.as_bytes();
    // Try every candidate pair start `s` such that `s < p < s +
    // pair_len`. Each candidate position is the index of the *first*
    // `\n` of the pair. The pair shape: bytes[s] == '\n', then
    // `blank_prefix`, then '\n', then `content_prefix`.
    let lo = p.saturating_sub(pair_len.saturating_sub(1));
    for s in lo..p {
        if s + pair_len > bytes.len() {
            break;
        }
        if bytes[s] != b'\n' {
            continue;
        }
        // Reject hard-break `\n` (`  \n` / `\\\n`) — those aren't
        // structural pair `\n`s.
        if s >= 2 && bytes[s - 1] == b' ' && bytes[s - 2] == b' ' {
            continue;
        }
        if s >= 1 && bytes[s - 1] == b'\\' {
            continue;
        }
        if &bytes[s + 1..s + 1 + blank_bytes.len()] != blank_bytes {
            continue;
        }
        if bytes[s + 1 + blank_bytes.len()] != b'\n' {
            continue;
        }
        if &bytes[s + 2 + blank_bytes.len()..s + pair_len] != content_bytes {
            continue;
        }
        // Found a valid pair from s to s + pair_len. Allowed
        // positions: the pair's start (s) and end (s + pair_len).
        // Every other interior byte is forbidden.
        let offset = p - s;
        if offset > 0 && offset < pair_len {
            return true;
        }
    }
    false
}

/// Is byte index `p` inside hidden list-item indent bytes? On any
/// line in a list item's source range, the leading
/// `cumulative_marker_byte_len` bytes are hidden by the renderer
/// (cumulative = sum of every enclosing item's marker length). On
/// the marker line of an item the same span ends at the marker's
/// end (the marker chars are part of the hidden span).
///
/// The hidden span `[line_start, line_start + cumulative)` collapses
/// to a single visible content edge in the rendered line. Both the
/// "real beginning of the line" (byte right after `\n`) and the
/// "after-marker" content edge map to the same display position, so
/// landing at any of those source bytes — including `line_start` —
/// is visually indistinguishable. Forbidding the entire hidden span
/// (excluding only the *content edge* at `hidden_end`) keeps each
/// visible cursor column owned by exactly one source position, so
/// arrow-key navigation across line boundaries doesn't pause at an
/// invisible byte offset and `SetSelection` snaps to the unique
/// content edge.
///
/// Doc-start byte 0 is exempt — every doc edge is allowed by
/// convention (matching the `is_paragraph_break_interior` rule)
/// and forbidding it would leave `nearest_allowed_position` with
/// no allowed predecessor to fall back to.
pub fn is_list_indent_interior(markdown: &str, p: usize) -> bool {
    if p == 0 {
        return false;
    }
    let bytes = markdown.as_bytes();
    let line_start = line_start_offset(bytes, p);
    let total = list_line_indent_at(markdown, line_start);
    if total == 0 {
        return false;
    }
    // Clamp to end of line so we don't spill past a `\n` in
    // unusual cases (e.g. a line shorter than the cumulative
    // indent).
    let line_end = {
        let mut q = p.max(line_start);
        while q < bytes.len() && bytes[q] != b'\n' {
            q += 1;
        }
        q
    };
    let hidden_end = (line_start + total).min(line_end);
    p < hidden_end
}

/// Sum of marker_byte_lens for every list item whose source range
/// *overlaps* the line at `line_start` — i.e., every item whose
/// hide-continuation-indent pass would touch this line during
/// rendering. This is the right summation for "what's hidden at the
/// start of this line":
///
/// * On a continuation line of an item, the item's range covers the
///   line, so the item contributes its marker_byte_len of leading
///   spaces.
/// * On the marker line of a nested item, the inner item's range
///   starts at the marker (pulldown convention) — it overlaps the
///   line via the marker bytes, so its marker_byte_len is added on
///   top of any outer ancestor's continuation contribution.
/// * Across a sibling-item boundary (e.g. byte right after the `\n`
///   between two siblings), the *previous* sibling's range ends at
///   `line_start` and so does *not* overlap the line — only the
///   next sibling's marker contributes.
fn list_line_indent_at(markdown: &str, line_start: usize) -> usize {
    let bytes = markdown.as_bytes();
    let mut line_end = line_start;
    while line_end < bytes.len() && bytes[line_end] != b'\n' {
        line_end += 1;
    }
    sum_overlapping_item_marker_widths(markdown, line_start, line_end)
}

/// Sum of marker widths for every list item whose source range
/// overlaps `[start, end)`. Walks the parse tree once. This is the
/// non-cursor analog of the LI-only piece of
/// [`chain_continuation_prefix_bytes`] — it answers "how much hidden
/// indent overlaps this byte range?" rather than "how much indent
/// encloses this cursor?".
fn sum_overlapping_item_marker_widths(markdown: &str, start: usize, end: usize) -> usize {
    fn walk(nodes: &[crate::syntax::SyntaxNode], start: usize, end: usize, total: &mut usize) {
        for node in nodes {
            if node.range.start >= end || node.range.end <= start {
                continue;
            }
            if let crate::syntax::NodeKind::ListItem { marker_range } = &node.kind {
                *total += marker_range.end - marker_range.start;
            }
            walk(&node.children, start, end, total);
        }
    }
    let mut total = 0;
    walk(&crate::parser::parse(markdown), start, end, &mut total);
    total
}

/// Pure structural test for "p sits strictly inside a structural pair."
///
/// The structural pair is the depth-D generalization of `\n\n`: it is
/// `\n[prefix]\n[prefix]` where `[prefix]` is `> ` repeated D times
/// (D >= 0; D == 0 collapses to plain `\n\n`). Top-level paragraph
/// breaks are pairs at depth 0; blockquote-internal paragraph breaks
/// are pairs at depth >= 1. Within a contiguous run of consecutive
/// pairs the byte sequence alternates `\n` / `[prefix]` / `\n` /
/// `[prefix]` …; allowed cursor positions are at multiples of one
/// pair length (the run boundaries and the seams between adjacent
/// pairs). Every other interior position is forbidden.
///
/// We detect by walking outward from `p` over the contiguous run of
/// only `\n`, `>`, and ` ` bytes — that's the maximal region a pair
/// can occupy. If both walks bracket the run with content (or
/// buffer edges) and the run holds an even count of `\n`s laid out as
/// equally-sized pair-shaped slices, `p` is forbidden iff its offset
/// from the run start isn't a clean multiple of the pair length.
///
/// Hard breaks (`  \n` / `\\\n`) bound the run early — `bytes[q-1]`
/// of `  ` or `\\` doesn't satisfy the `' '/'>'/'\n'` predicate, so
/// the walk stops before counting them.
pub fn is_paragraph_break_interior(bytes: &[u8], p: usize) -> bool {
    if p == 0 || p > bytes.len() {
        return false;
    }

    // The run reads `[partial-prefix]? \n [prefix] \n [prefix] \n …`
    // around `p`. All `[prefix]`s in a single pair must have the
    // *same* marker count — once one prefix sets the depth, the
    // walks (back and forward) cap consumption to that count so a
    // greedy reach into adjacent content (a stray `>` end-of-buffer,
    // a deeper nested BQ start, a content trailing space) doesn't
    // corrupt pair-length math.
    //
    // Algorithm:
    //   1. Back-walk: consume a `[partial-prefix]` left of `p` (the
    //      markers `p` sits inside), then *require* a structural
    //      `\n`. If none, `p` is in content — return false.
    //   2. Repeat back-walk: consume another `[prefix] \n`. The
    //      first such full prefix sets the depth; later prefixes
    //      must match. If a marker count differs, stop (mixed-depth
    //      run isn't a single pair structure).
    //   3. Forward-walk: consume the rest of the partial-prefix
    //      forward, then alternating `\n [prefix]` segments. Cap
    //      each prefix to the established depth.
    //   4. Check the run is `\n[prefix]\n[prefix]…` with even `\n`
    //      count and pair_len consistent with a depth-D pair.

    let mut q = p;
    let initial_partial_back = walk_back_markers(bytes, &mut q, usize::MAX);
    if !walk_back_required_newline(bytes, &mut q) {
        return false;
    }
    let mut run_start = q;
    let mut depth: Option<usize> = None;
    loop {
        let probe = q;
        let cap = depth.unwrap_or(usize::MAX);
        let count = walk_back_markers(bytes, &mut q, cap);
        if count == 0 {
            break;
        }
        if !walk_back_required_newline(bytes, &mut q) {
            // Markers consumed without a preceding `\n` — content.
            // `q` may be partially walked but `run_start` wasn't
            // updated, so the run already excludes these bytes.
            let _ = probe;
            break;
        }
        match depth {
            Some(d) if d != count => break, // mixed-depth — stop
            None => depth = Some(count),
            _ => {}
        }
        run_start = q;
    }

    let mut q = p;
    // Complete the partial-prefix forward.
    let initial_partial_fwd = if let Some(d) = depth {
        let needed = d.saturating_sub(initial_partial_back);
        walk_forward_markers(bytes, &mut q, needed)
    } else {
        // No depth from backward yet — first thing we see (forward
        // partial-prefix or first full prefix below) sets it.
        walk_forward_markers(bytes, &mut q, usize::MAX)
    };
    let total_partial = initial_partial_back + initial_partial_fwd;
    if let Some(d) = depth {
        if total_partial != d {
            // The partial-prefix at `p` doesn't fit the established
            // depth — `p` is in content adjacent to the run, not
            // inside it.
            return false;
        }
    } else if total_partial > 0 {
        depth = Some(total_partial);
    }

    let mut run_end = q;
    while q < bytes.len() && bytes[q] == b'\n' {
        let preceded_by_two_spaces = q >= 2 && bytes[q - 1] == b' ' && bytes[q - 2] == b' ';
        let preceded_by_backslash = q >= 1 && bytes[q - 1] == b'\\';
        if preceded_by_two_spaces || preceded_by_backslash {
            break;
        }
        q += 1;
        let cap = depth.unwrap_or(usize::MAX);
        let consumed = walk_forward_markers(bytes, &mut q, cap);
        match depth {
            Some(d) if consumed != d => {
                // Next prefix doesn't match depth — the run ends
                // before this `\n`. (q has already been advanced
                // past the `\n` and a partial prefix, but run_end
                // wasn't yet committed for this iteration.)
                break;
            }
            None => depth = Some(consumed),
            _ => {}
        }
        run_end = q;
    }

    let run_len = run_end - run_start;
    if run_len < 2 {
        return false;
    }
    let nl_count = bytes[run_start..run_end]
        .iter()
        .filter(|&&b| b == b'\n')
        .count();
    if nl_count < 2 || !nl_count.is_multiple_of(2) {
        return false;
    }
    let pair_count = nl_count / 2;
    if !run_len.is_multiple_of(pair_count) {
        return false;
    }
    let pair_len = run_len / pair_count;
    // Each pair is `\n[prefix]\n[prefix]` of length `2 + 4D`, so
    // `(pair_len - 2)` must be a multiple of 4 (two prefixes × two
    // bytes per `> ` marker).
    if pair_len < 2 || !(pair_len - 2).is_multiple_of(4) {
        return false;
    }
    let offset = p - run_start;
    offset > 0 && offset < run_len && !offset.is_multiple_of(pair_len)
}

// ---------------------------------------------------------------------------
// Allowed-position snapping
// ---------------------------------------------------------------------------

pub fn next_allowed_position(markdown: &str, mut p: usize) -> usize {
    let len = markdown.len();
    while p < len && is_forbidden_position(markdown, p) {
        p += 1;
    }
    p
}

pub fn prev_allowed_position(markdown: &str, mut p: usize) -> usize {
    while p > 0 && is_forbidden_position(markdown, p) {
        p -= 1;
    }
    p
}

/// Snap `p` to the closest allowed position. Forward wins ties. This is
/// the idempotent variant of the snap rule — used by `set_selection`
/// (mouse clicks, host API), where running the same input twice must
/// produce the same output.
///
/// Two special cases override the source-byte-distance metric:
///
/// 1. **List-indent forbidden positions** all collapse to the same
///    visible content edge of the *current* line. The next allowed
///    byte forward sits at that content edge; the prev allowed byte
///    backward sits on the *previous* visual line. Snap forward so
///    a click on the marker area lands at the current line's
///    content rather than hopping back across the line boundary.
///
/// 2. **Doc-edge degeneracy** — `prev_allowed_position` bottoms out
///    at byte 0 even if 0 is forbidden, and `next_allowed_position`
///    similarly at `markdown.len()`. We re-check each candidate and
///    drop it if still forbidden; if both are unavailable, return
///    `p` (degenerate buffer where every position is forbidden).
pub fn nearest_allowed_position(markdown: &str, p: usize) -> usize {
    if !is_forbidden_position(markdown, p) {
        return p;
    }
    if is_list_indent_interior(markdown, p) {
        let next = next_allowed_position(markdown, p);
        if !is_forbidden_position(markdown, next) {
            return next;
        }
    }
    let next = next_allowed_position(markdown, p);
    let prev = prev_allowed_position(markdown, p);
    let prev_ok = !is_forbidden_position(markdown, prev);
    let next_ok = !is_forbidden_position(markdown, next);
    match (prev_ok, next_ok) {
        (true, true) => {
            if next.saturating_sub(p) <= p.saturating_sub(prev) {
                next
            } else {
                prev
            }
        }
        (true, false) => prev,
        (false, true) => next,
        (false, false) => p,
    }
}

// ---------------------------------------------------------------------------
// Soft-break detection
// ---------------------------------------------------------------------------

/// Is the `\n` at byte index `p` a soft break (a lone newline that would
/// be ambiguous in CommonMark)?
///
/// A soft break is the `\n` *between two lines of paragraph content in
/// the same container scope* — exactly the kind CommonMark would render
/// as a space inside a paragraph. The editor's invariant promotes such
/// `\n`s into the depth-D pair `\n[prefix]\n[prefix]` so the resulting
/// rendering matches the chat transcript pixel-for-pixel.
///
/// The byte detector implements that semantic with five exemption rules.
/// Anything `\n` falling into one of these is *not* a soft break — it's
/// already a structural separator and the buffer carries it verbatim:
///
/// 1. **Document edge.** Leading or trailing single `\n` is harmless
///    whitespace; rewriting it would corrupt pasted content.
/// 2. **Adjacent `\n`.** Already inside a paragraph-break run.
/// 3. **Hard break** (`  \n` / `\\\n`) — deliberate in-paragraph break.
/// 4. **Pair-interior.** The `\n` is one of the two `\n`s of a complete
///    depth-D pair `\n[prefix]\n[prefix]`; the pair detector flags both
///    halves, and we probe `p` and `p + 1` to recognize either.
/// 5. **Marker-only-line adjacency.** Either the line above or the
///    line below is "marker-only" — every byte after its container
///    markers is whitespace. Marker-only lines are *paragraph
///    terminators*, not paragraph content, so the `\n`s on either side
///    of one are structural stitching, never soft breaks. This subsumes
///    most of the depth-change cases (the deeper/shallower marker
///    line is itself marker-only) and is what actually makes
///    `enforce_invariants` idempotent across runs of mixed-depth blank
///    lines: without it, a `\n` between two same-depth blank lines that
///    are interrupted by a different-depth blank elsewhere in the run
///    breaks the pair detector, gets misclassified as a soft break, and
///    each `enforce_invariants` call splices in another `[prefix]\n`
///    that further fragments the run — the cascading-line bug.
///
/// Genuine *lazy paragraph continuations* (line_a has BQ markers,
/// line_b has none but has content) still promote: line_b isn't
/// marker-only, so the rule above doesn't fire, and the editor restores
/// the dropped BQ scope by inserting the missing prefix.
///
/// This rule generalizes to lists by extending `is_marker_only_line` to
/// recognize list-marker continuation prefixes; the same five exemption
/// classes apply unchanged.
pub fn is_soft_break(bytes: &[u8], p: usize) -> bool {
    if bytes[p] != b'\n' {
        return false;
    }
    if p == 0 || p + 1 >= bytes.len() {
        return false;
    }
    if bytes[p - 1] == b'\n' || bytes[p + 1] == b'\n' {
        return false;
    }
    if bytes[p - 1] == b'\\' {
        return false;
    }
    if p >= 2 && bytes[p - 1] == b' ' && bytes[p - 2] == b' ' {
        return false;
    }
    if is_paragraph_break_interior(bytes, p) || is_paragraph_break_interior(bytes, p + 1) {
        return false;
    }
    if is_marker_only_line_ending_at(bytes, p) || is_marker_only_line_starting_at(bytes, p + 1) {
        return false;
    }
    true
}

/// Whether the line ending at `line_end_excl` (i.e. whose `\n` is at
/// `line_end_excl` or whose end-of-buffer is at `line_end_excl`) is
/// "marker-only" — has at least one BQ marker and only whitespace after
/// the markers. Used by the soft-break detector to recognize structural
/// separator lines (which are paragraph terminators, not paragraph
/// content).
pub fn is_marker_only_line_ending_at(bytes: &[u8], line_end_excl: usize) -> bool {
    let mut start = line_end_excl;
    while start > 0 && bytes[start - 1] != b'\n' {
        start -= 1;
    }
    is_marker_only_range(bytes, start, line_end_excl)
}

/// Forward analog of [`is_marker_only_line_ending_at`].
pub fn is_marker_only_line_starting_at(bytes: &[u8], line_start: usize) -> bool {
    let mut end = line_start;
    while end < bytes.len() && bytes[end] != b'\n' {
        end += 1;
    }
    is_marker_only_range(bytes, line_start, end)
}

fn is_marker_only_range(bytes: &[u8], line_start: usize, line_end_excl: usize) -> bool {
    let (markers, after) = count_line_markers(bytes, line_start);
    if markers == 0 {
        return false;
    }
    bytes[after..line_end_excl]
        .iter()
        .all(|&b| b == b' ' || b == b'\t')
}

// ---------------------------------------------------------------------------
// Marker walks (blockquote prefix ` > `)
// ---------------------------------------------------------------------------

/// Walk `q` back over up to `cap` blockquote markers (`> ` or bare `>`),
/// returning the count consumed.
pub fn walk_back_markers(bytes: &[u8], q: &mut usize, cap: usize) -> usize {
    let mut count = 0;
    while count < cap {
        if *q >= 2 && bytes[*q - 1] == b' ' && bytes[*q - 2] == b'>' {
            *q -= 2;
            count += 1;
        } else if *q >= 1 && bytes[*q - 1] == b'>' {
            *q -= 1;
            count += 1;
        } else {
            return count;
        }
    }
    count
}

/// Walk `q` forward over up to `cap` blockquote markers (`> ` or bare
/// `>`), returning the count consumed. Handles the in-marker case where
/// `q` sits between a `>` and its trailing `' '`: the trailing space is
/// consumed *without counting*, since whoever's walking from the other
/// side has already accounted for that marker.
pub fn walk_forward_markers(bytes: &[u8], q: &mut usize, cap: usize) -> usize {
    if *q < bytes.len() && bytes[*q] == b' ' && *q >= 1 && bytes[*q - 1] == b'>' {
        *q += 1;
    }
    let mut count = 0;
    while count < cap && *q < bytes.len() && bytes[*q] == b'>' {
        *q += 1;
        if *q < bytes.len() && bytes[*q] == b' ' {
            *q += 1;
        }
        count += 1;
    }
    count
}

/// Walk `q` back over a single structural `\n`. Returns true on success
/// (and updates `q`); returns false (and leaves `q` unchanged) if `q`
/// isn't preceded by a `\n` or the `\n` is part of a hard break.
pub fn walk_back_required_newline(bytes: &[u8], q: &mut usize) -> bool {
    if *q == 0 || bytes[*q - 1] != b'\n' {
        return false;
    }
    let nl = *q - 1;
    let preceded_by_two_spaces = nl >= 2 && bytes[nl - 1] == b' ' && bytes[nl - 2] == b' ';
    let preceded_by_backslash = nl >= 1 && bytes[nl - 1] == b'\\';
    if preceded_by_two_spaces || preceded_by_backslash {
        return false;
    }
    *q -= 1;
    true
}

/// Walk `q` forward over a single structural `\n` (not the `\n` of a
/// hard break `  \n` / `\\\n`). Returns true on success; leaves `q`
/// unmodified on failure.
pub fn walk_forward_required_newline(bytes: &[u8], q: &mut usize) -> bool {
    if *q >= bytes.len() || bytes[*q] != b'\n' {
        return false;
    }
    let preceded_by_two_spaces = *q >= 2 && bytes[*q - 1] == b' ' && bytes[*q - 2] == b' ';
    let preceded_by_backslash = *q >= 1 && bytes[*q - 1] == b'\\';
    if preceded_by_two_spaces || preceded_by_backslash {
        return false;
    }
    *q += 1;
    true
}

/// Count the leading blockquote markers on the line that contains
/// `line_start`. Each marker is `>` optionally followed by a single
/// space, optionally preceded by up to 3 CommonMark-permitted spaces
/// of indent. Returns the marker count and the byte offset right
/// after the last marker.
pub fn count_line_markers(bytes: &[u8], line_start: usize) -> (usize, usize) {
    let mut q = line_start;
    let mut markers = 0;
    loop {
        let mut indent = 0;
        while q < bytes.len() && bytes[q] == b' ' && indent < 3 {
            q += 1;
            indent += 1;
        }
        if q < bytes.len() && bytes[q] == b'>' {
            q += 1;
            if q < bytes.len() && bytes[q] == b' ' {
                q += 1;
            }
            markers += 1;
            continue;
        }
        return (markers, q);
    }
}

/// Depth (count of leading `> ` markers) of the source line ending at
/// `line_end_excl` — i.e. the line whose final byte is at
/// `line_end_excl - 1` (and whose `\n` terminator, if present, sits at
/// `line_end_excl`). Walks back to the previous `\n` (or buffer start)
/// to find the line's start, then counts markers forward.
pub fn line_depth_ending_at(bytes: &[u8], line_end_excl: usize) -> usize {
    let mut s = line_end_excl;
    while s > 0 && bytes[s - 1] != b'\n' {
        s -= 1;
    }
    let (markers, _) = count_line_markers(bytes, s);
    markers
}

// ---------------------------------------------------------------------------
// Pair detectors at boundaries (used by atomic Backspace/Delete)
// ---------------------------------------------------------------------------

/// If `cursor` sits at the start of a paragraph that is *embedded
/// within* a blockquote (preceded by another BQ-prefixed line), return
/// the two byte ranges that Backspace should remove to decrease the
/// paragraph's nesting by one — one leading `> ` from the line at
/// the cursor *and* one leading `> ` from the prefix line directly
/// above it (the `[prefix]\n[prefix]` pair-half pattern).
///
/// Returned as `(above, current)` in source order, so callers can
/// process them right-to-left to keep offsets stable across a
/// two-stage splice.
///
/// **Why both halves outdent together.** The pair invariant the rest
/// of the editor relies on is that any structural paragraph break is
/// a clean `\n[prefix]\n[prefix]` of *equal-depth* prefixes. Popping a
/// marker only from the cursor's line would leave an asymmetric pair
/// — the line above one level deeper than the line at the cursor —
/// which the soft-break detector and pair-interior detector both have
/// to special-case to avoid corrupting on the next event. By
/// outdenting both halves of the pair in one keystroke, the result is
/// either a clean depth-(D-1) pair (when both started at depth D) or
/// a clean depth-0 paragraph break (when the second half had only one
/// marker to pop). The buffer never enters an asymmetric state, no
/// soft break is introduced, and `enforce_invariants` is a no-op on
/// the result.
///
/// The trigger condition is the source pattern
/// `\n[markers ≥ 1]\n[markers ≥ 1]` ending right at `cursor` — both
/// the line at the cursor and the line above it carry at least one
/// BQ marker. The two prefix lengths do *not* have to match: an
/// asymmetric state that snuck in via paste or a future programmatic
/// edit still outdents cleanly, with each side losing one marker.
///
/// Cases the detector deliberately *doesn't* fire on:
///
/// - The first paragraph of a top-level BQ that follows non-BQ
///   content (`para\n\n> bq`). The line above is content at depth 0
///   — outdenting would erase the user's BQ structure for what should
///   feel like a normal Backspace at the boundary. Falls through to
///   grapheme delete instead, matching the pre-change behavior.
/// - Top-level paragraph break `\n\n`. No marker to pop; the depth-0
///   atomic pair delete path takes over and merges the paragraphs.
///
/// Generalizes to lists: when list containers land, replace
/// `walk_back_markers` with a "walk back over the active continuation
/// prefix of the line ending at `cursor`" and the same outdent rule
/// applies — pop the deepest container marker from each half of the
/// pair.
pub fn bq_paragraph_outdent(bytes: &[u8], cursor: usize) -> Option<(Range<usize>, Range<usize>)> {
    let mut q = cursor;
    let markers1 = walk_back_markers(bytes, &mut q, usize::MAX);
    if markers1 == 0 {
        return None;
    }
    let prefix_below_start = q;
    if !walk_back_required_newline(bytes, &mut q) {
        return None;
    }
    let markers2 = walk_back_markers(bytes, &mut q, usize::MAX);
    if markers2 == 0 {
        return None;
    }
    let prefix_above_start = q;
    if !walk_back_required_newline(bytes, &mut q) {
        return None;
    }
    Some((
        first_marker_range(bytes, prefix_above_start),
        first_marker_range(bytes, prefix_below_start),
    ))
}

/// Byte range of the *first* blockquote marker on the line beginning
/// at `start`. Handles both the canonical `> ` form (2 bytes) and a
/// bare `>` (1 byte; appears when a marker sits right before `\n` and
/// `normalize_blockquote_prefixes` has nothing to pad).
fn first_marker_range(bytes: &[u8], start: usize) -> Range<usize> {
    let mut p = start;
    if p < bytes.len() && bytes[p] == b'>' {
        p += 1;
        if p < bytes.len() && bytes[p] == b' ' {
            p += 1;
        }
    }
    start..p
}

/// If `cursor` sits at the end of a depth-D structural pair (`\n` +
/// `> ` × D + `\n` + `> ` × D), return the pair's start byte.
///
/// The detector walks backward symmetrically: prefix → `\n` → prefix
/// → `\n`, requiring the two prefixes to be the same length so an
/// uneven structure doesn't trigger an atomic delete. Hard-break
/// `\n`s (`  \n` / `\\\n`) are not pair `\n`s — the back-walk rejects
/// them.
pub fn pair_at_end(bytes: &[u8], cursor: usize) -> Option<usize> {
    let mut q = cursor;
    let markers1 = walk_back_markers(bytes, &mut q, usize::MAX);
    if !walk_back_required_newline(bytes, &mut q) {
        return None;
    }
    let markers2 = walk_back_markers(bytes, &mut q, markers1);
    if markers2 != markers1 {
        return None;
    }
    if !walk_back_required_newline(bytes, &mut q) {
        return None;
    }
    Some(q)
}

/// Forward analog of [`pair_at_end`]: if `cursor` sits at the start of
/// a depth-D structural pair, return the pair's end byte.
pub fn pair_at_start(bytes: &[u8], cursor: usize) -> Option<usize> {
    let mut q = cursor;
    if !walk_forward_required_newline(bytes, &mut q) {
        return None;
    }
    let markers1 = walk_forward_markers(bytes, &mut q, usize::MAX);
    if !walk_forward_required_newline(bytes, &mut q) {
        return None;
    }
    let markers2 = walk_forward_markers(bytes, &mut q, markers1);
    if markers2 != markers1 {
        return None;
    }
    Some(q)
}

// ---------------------------------------------------------------------------
// Active container context — the unified `enclosing_containers_at` walker
//
// `enclosing_containers_at(markdown, cursor)` is the single source of
// truth for "which containers does this cursor sit inside?". It walks
// the parse tree once and returns an outermost-first chain. Every
// other cursor-context query in this module derives from the chain:
// blockquote depth, the `> ` continuation prefix, the innermost list
// item, the outer-list indent, the total-list indent, and the
// "cursor sits at the marker end" check are all small extractors over
// a single chain.
//
// Why one walker beats six. Before consolidation, every consumer
// re-walked the parse tree on its own — `blockquote_depth_at` for
// depth, `innermost_list_item_at` for the item context, `enclosing_
// items_at` / `outer_list_indent_at` / `total_list_indent_at` for
// indent arithmetic, etc. Each walk had its own boundary-equality
// semantics; small drift between them produced inconsistent results
// at depth (a deeper-than-expected list item would mismatch the
// blockquote prefix it should pair with). One walk, one boundary
// rule, six extractors.
// ---------------------------------------------------------------------------

/// One container entry in [`EnclosingChain`]. Outermost-first; every
/// container-aware rule in the editor reads its slice off this chain
/// instead of re-walking the parse tree.
///
/// Adding a new container kind (e.g. table cell, definition list)
/// means a new variant here, a new branch in `walk_chain`, and a new
/// extractor for whatever derived shape the consumer needs — the
/// callers (Enter routing, depth gestures, hard-break continuation,
/// canonicalization passes) keep their shape unchanged.
#[derive(Debug, Clone)]
pub enum EnclosingContainer {
    /// One blockquote level. Ranges are the BQ node's range from the
    /// parse tree. The depth at the cursor is just the count of these
    /// in the chain, so the inner data is intentionally minimal.
    BlockQuote { range: Range<usize> },
    /// Cursor-located list item. Carries the same data
    /// [`innermost_list_item_at`] used to return; nested list items
    /// produce successive `ListItem` entries in the chain.
    ListItem(ListItemContext),
}

/// Cursor-location context for one list item. Used both as a chain
/// entry (an item enclosing the cursor) and as the standalone result
/// of [`innermost_list_item_at`]. The carry-over is intentional:
/// the same shape works for "deepest item containing the cursor"
/// and "every item containing the cursor."
#[derive(Debug, Clone)]
pub struct ListItemContext {
    pub list_kind: crate::syntax::ListKind,
    /// Zero-based position of this item within its list — used to
    /// compute the next ordered item's number (`start + index + 1`).
    pub item_index: usize,
    /// Source range of the whole item.
    pub item_range: Range<usize>,
    /// Source range of the marker bytes (e.g. `- ` or `1. `).
    pub marker_range: Range<usize>,
}

impl ListItemContext {
    /// Byte-width of the item's marker (e.g. 2 for `- `, 3 for `1. `,
    /// 4 for `10. `). Used in indent arithmetic.
    pub fn marker_width(&self) -> usize {
        self.marker_range.end - self.marker_range.start
    }

    /// Is this item part of an ordered list?
    pub fn is_ordered(&self) -> bool {
        matches!(self.list_kind, crate::syntax::ListKind::Ordered { .. })
    }

    /// Marker text for the *next* item the user creates by pressing
    /// Enter at the end of this one. For unordered, repeat the
    /// bullet char from this item; for ordered, increment.
    fn next_marker_text(&self, markdown: &str) -> String {
        match self.list_kind {
            crate::syntax::ListKind::Unordered => {
                // Find this list's first item to read the bullet
                // char actually used in source; default to `-` if
                // we can't find it.
                let bullet = bullet_for_item_index(markdown, self.item_index).unwrap_or(b'-');
                format!("{} ", bullet as char)
            }
            crate::syntax::ListKind::Ordered { start } => {
                // `start` is the parsed list-start; `item_index`
                // counts items from zero, so the *next* item's
                // number is `start + item_index + 1`.
                format!("{}. ", start + self.item_index as u64 + 1)
            }
        }
    }
}

/// Outermost-first chain of containers enclosing a cursor.
pub type EnclosingChain = Vec<EnclosingContainer>;

/// Walk the parse tree once, building the outermost-first chain of
/// containers that enclose `cursor`.
///
/// # Two cooperating boundary rules
///
/// ## 1. Strict containment beats range-end equality
///
/// If two siblings both match `cursor` — one strictly (`cursor <
/// range.end`) and one only at its `range.end` boundary — pick the
/// strict one. This is the rule that makes
///
/// ```text
/// 1. one
///                <- cursor here, at the start of the next paragraph
/// two
/// ```
///
/// land *inside* the paragraph rather than "still inside item 1
/// because item 1's range.end happens to equal the cursor byte".
///
/// At a list-item boundary (cursor exactly between two siblings,
/// e.g. byte 7 in `1. one\n2. two`), the same preference picks the
/// later sibling — item 1 strictly contains the cursor while item 0
/// only matches by boundary. This matches the post-Enter caret
/// intent.
///
/// ## 2. Container ranges are trimmed of trailing structural separators
///
/// Pulldown's range for a List, ListItem, or BlockQuote often
/// includes the structural `\n\n` (or longer) separator that
/// follows the construct. Per the [pairs model](`crate::analysis`),
/// each `\n\n` pair is one "after the construct" structural unit —
/// not part of the construct's content.
///
/// Without trimming, a cursor parked on the post-construct empty
/// row (e.g. after pressing Enter twice from `1. asdf` to leave the
/// list and land on a blank line) would still match the list's
/// raw range at its end, and Enter / depth-change actions would
/// route through the now-departed list. With trimming, that cursor
/// is correctly seen as *outside* every container.
///
/// The trim removes any *complete* trailing pair (2+ consecutive
/// `\n`s) from the range's end. A single trailing `\n` is the line
/// terminator that's part of the last content line — it stays.
///
/// # Pure boundary fallback
///
/// If no sibling strictly contains `cursor` (e.g. cursor at end of
/// buffer with no trailing pair), pick the last sibling whose
/// trimmed `range.end == cursor` so end-of-buffer Enter still
/// routes through the surrounding container.
pub fn enclosing_containers_at(markdown: &str, cursor: usize) -> EnclosingChain {
    let tree = crate::parser::parse(markdown);
    enclosing_containers_at_in_tree(&tree, markdown.as_bytes(), cursor)
}

/// Variant of [`enclosing_containers_at`] for callers that already
/// hold a parse tree. Hot in `update::promote_soft_breaks`, which
/// otherwise re-parses the buffer once per byte inside fenced code
/// content (the chain query is the per-byte source-of-truth for
/// "what continuation prefix should this line carry").
pub fn enclosing_containers_at_in_tree(
    tree: &[crate::syntax::SyntaxNode],
    bytes: &[u8],
    cursor: usize,
) -> EnclosingChain {
    let mut chain = Vec::new();
    walk_chain(tree, cursor, bytes, &mut chain);
    chain
}

/// `node`'s range with any trailing structural-separator `\n\n`
/// pair trimmed off. Only applied to container kinds (List,
/// ListItem, BlockQuote); for everything else returns the raw end
/// unchanged.
///
/// Trimming rule: count consecutive trailing `\n`s. If 2 or more,
/// drop them (they're "after the construct" structural units in
/// the pairs model). A single trailing `\n` is the last content
/// line's terminator and stays.
fn effective_node_end(node: &crate::syntax::SyntaxNode, bytes: &[u8]) -> usize {
    let raw_end = node.range.end;
    let trims = matches!(
        node.kind,
        crate::syntax::NodeKind::List { .. }
            | crate::syntax::NodeKind::ListItem { .. }
            | crate::syntax::NodeKind::BlockQuote { .. }
    );
    if !trims {
        return raw_end;
    }
    let start = node.range.start;
    let mut p = raw_end;
    let mut trailing = 0;
    while p > start && bytes[p - 1] == b'\n' {
        p -= 1;
        trailing += 1;
    }
    if trailing >= 2 {
        // Drop the entire trailing-`\n` run. Whether it's exactly
        // 2 (one pair) or more (additional empty paragraphs), all
        // of it is structural separator outside the construct.
        p
    } else {
        raw_end
    }
}

/// Pick the sibling at this level that owns `cursor`, applying the
/// "strict containment beats boundary equality" preference. Returns
/// `None` when no sibling matches.
fn pick_chain_target<'a>(
    nodes: &'a [crate::syntax::SyntaxNode],
    cursor: usize,
    bytes: &[u8],
) -> Option<&'a crate::syntax::SyntaxNode> {
    let mut target: Option<&crate::syntax::SyntaxNode> = None;
    let mut target_is_strict = false;
    for node in nodes {
        let effective_end = effective_node_end(node, bytes);
        if cursor < node.range.start || cursor > effective_end {
            continue;
        }
        let is_strict = cursor < effective_end;
        if is_strict || !target_is_strict {
            target = Some(node);
            target_is_strict = is_strict || target_is_strict;
        }
    }
    target
}

fn walk_chain(
    nodes: &[crate::syntax::SyntaxNode],
    cursor: usize,
    bytes: &[u8],
    out: &mut EnclosingChain,
) {
    let Some(node) = pick_chain_target(nodes, cursor, bytes) else {
        return;
    };
    match &node.kind {
        crate::syntax::NodeKind::BlockQuote { .. } => {
            out.push(EnclosingContainer::BlockQuote {
                range: node.range.clone(),
            });
            walk_chain(&node.children, cursor, bytes, out);
        }
        crate::syntax::NodeKind::List { kind } => {
            // Pick the right item using the same trim + strict-over-
            // boundary preference as the top-level walk. We need the
            // item's positional index in the list (for
            // `next_marker_text` numbering), so we open-code rather
            // than reusing `pick_chain_target`.
            let mut item_target: Option<(usize, &crate::syntax::SyntaxNode, Range<usize>)> = None;
            let mut item_is_strict = false;
            let mut idx = 0;
            for child in &node.children {
                if let crate::syntax::NodeKind::ListItem { marker_range } = &child.kind {
                    let effective_end = effective_node_end(child, bytes);
                    if cursor >= child.range.start && cursor <= effective_end {
                        let is_strict = cursor < effective_end;
                        if is_strict || !item_is_strict {
                            item_target = Some((idx, child, marker_range.clone()));
                            item_is_strict = is_strict || item_is_strict;
                        }
                    }
                    idx += 1;
                }
            }
            if let Some((item_idx, item, marker_range)) = item_target {
                out.push(EnclosingContainer::ListItem(ListItemContext {
                    list_kind: *kind,
                    item_index: item_idx,
                    item_range: item.range.clone(),
                    marker_range,
                }));
                walk_chain(&item.children, cursor, bytes, out);
            }
        }
        _ => {
            walk_chain(&node.children, cursor, bytes, out);
        }
    }
}

/// Number of `BlockQuote` entries in `chain` — the depth at which
/// `cursor` sits.
pub fn chain_blockquote_depth(chain: &[EnclosingContainer]) -> usize {
    chain
        .iter()
        .filter(|c| matches!(c, EnclosingContainer::BlockQuote { .. }))
        .count()
}

// ---------------------------------------------------------------------------
// Chain prefix builders — the canonical entry points
// ---------------------------------------------------------------------------
//
// These helpers turn an `EnclosingChain` into the bytes that introduce a
// continuation line for that chain. **Use these — don't compute prefixes
// locally.** Reaching for raw `\n` boundaries or hand-built `"> "` strings
// in a chain-aware context is a bug; we've fixed several of those by
// migrating to these helpers.
//
// Canonical entry points and when to use each:
//
// - [`chain_continuation_prefix`] — full per-line continuation prefix,
//   interleaving LI indents and BQ markers in chain order. Use whenever
//   you need the bytes that appear at the start of a continuation line
//   for the cursor's chain (Enter inserts, Shift+Enter inserts, soft-break
//   promotion, paragraph-break-pair shapes, render's chain-aware hide pass).
//
// - [`chain_continuation_prefix_bytes`] — byte-length of the same string
//   without allocating. Use for "how many bytes of hidden continuation
//   prefix introduce this line".
//
// - [`chain_outer_prefix_bytes`] — byte-length of the prefix contributed
//   by every container *above* the innermost. Use to compute "where does
//   the active container's content begin on this line, relative to
//   line_start" — i.e. the offset to insert / strip indent at without
//   disturbing outer markers (Tab indent insertion, Shift+Tab dedent
//   strip).
//
// - [`chain_pair_shape`] — the `(blank_prefix, content_prefix)`
//   representation of the canonical paragraph-break pair for the chain.
//   Use whenever you emit or recognize a structural pair: BQ-outdent
//   transform, atomic pair-delete, forbidden-position predicate.
//
// All four helpers agree by construction. If a future call site needs a
// *new* shape variant, add it here with the same naming pattern; don't
// duplicate the chain-walking logic in callers.

/// The full per-line continuation prefix for `chain`, walking
/// outermost-first and emitting one segment per container in chain
/// order:
///
/// - `ListItem`: `marker_width` spaces (the continuation indent that
///   keeps a continuation line aligned with this item's content edge).
/// - `BlockQuote`: `"> "` (the marker that introduces a line inside
///   this blockquote).
///
/// So a chain `[LI(2), BQ, LI(2), BQ]` produces `"  >   > "` —
/// outer-LI indent, outer BQ marker, inner-LI indent, inner BQ
/// marker. This is the canonical "scope continuation prefix" shape the
/// editor needs everywhere it inserts a new line inside an arbitrary
/// container chain (Enter, Shift+Enter, soft-break promotion,
/// pair-promotion). The renderer's per-leaf decoration loop emits the
/// same alternation pixel-for-pixel; this helper produces the source
/// counterpart.
///
/// Pairs with [`chain_continuation_prefix_bytes`] for callers that want
/// the byte count without the string.
pub fn chain_continuation_prefix(chain: &[EnclosingContainer]) -> String {
    let mut out = String::new();
    for c in chain {
        match c {
            EnclosingContainer::ListItem(ctx) => {
                for _ in 0..ctx.marker_width() {
                    out.push(' ');
                }
            }
            EnclosingContainer::BlockQuote { .. } => out.push_str("> "),
        }
    }
    out
}

/// Byte length of [`chain_continuation_prefix`] without building the
/// string. The two functions agree by construction: one space per LI
/// `marker_width` byte, two bytes per BQ marker.
pub fn chain_continuation_prefix_bytes(chain: &[EnclosingContainer]) -> usize {
    chain
        .iter()
        .map(|c| match c {
            EnclosingContainer::ListItem(ctx) => ctx.marker_width(),
            EnclosingContainer::BlockQuote { .. } => 2,
        })
        .sum()
}

/// Canonical paragraph-break-pair shape for `chain` as `(blank_prefix,
/// content_prefix)`. The pair is always
/// `\n{blank_prefix}\n{content_prefix}` — three branches collapse into
/// one representation:
///
/// 1. **Chain ends in BQ** → `blank_prefix == content_prefix == full
///    chain prefix`. Symmetric depth-D pair `\n[full]\n[full]`.
/// 2. **Chain has BQ but trails with LIs** (e.g. `[LI, BQ, LI]`) →
///    `blank_prefix` is the chain prefix *through the last BQ entry*
///    (BQs require their `> ` marker on blank lines; LIs after the
///    last BQ contribute no blank-line prefix because LIs accept
///    blank lines without their continuation indent).
///    `content_prefix` is the full chain prefix. Asymmetric pair
///    `\n[blank]\n[content]`.
/// 3. **Chain has no BQ** (pure LI chain or empty) → `blank_prefix`
///    is empty (LIs accept a blank line at column 0); `content_prefix`
///    is the full chain prefix. Pair `\n\n[content]`.
///
/// The single `(blank, content)` representation lets every call site
/// (BQ-outdent transform, chain-aware pair detector, forbidden-position
/// predicate) emit / recognize the canonical shape without branching on
/// chain shape.
pub fn chain_pair_shape(chain: &[EnclosingContainer]) -> (String, String) {
    let content_prefix = chain_continuation_prefix(chain);
    // Index of the last BQ entry in chain, if any.
    let last_bq = chain
        .iter()
        .rposition(|c| matches!(c, EnclosingContainer::BlockQuote { .. }));
    let blank_prefix = match last_bq {
        Some(i) => chain_continuation_prefix(&chain[..=i]),
        None => String::new(),
    };
    (blank_prefix, content_prefix)
}

/// Innermost list item in `chain`, if any.
pub fn chain_innermost_list_item(chain: &[EnclosingContainer]) -> Option<&ListItemContext> {
    chain.iter().rev().find_map(|c| match c {
        EnclosingContainer::ListItem(ctx) => Some(ctx),
        _ => None,
    })
}

/// `true` when the innermost list-item in `chain` has another
/// list-item as its *immediate* (one step out) enclosing container —
/// i.e. it's a normal nested list-item like `- a\n  - b`. `false`
/// when the innermost LI has no enclosing LI, *or* when its immediate
/// enclosing container is a blockquote (e.g. `- > - inner` where the
/// inner LI's parent in the chain is the BQ, not the outer LI).
///
/// Used by Shift+Tab dedent to decide between "strip parent
/// marker_width spaces" and "drop the marker entirely". Items
/// directly wrapped by a BQ have no parent-LI marker width to
/// subtract, so they fall through to the drop-marker branch.
pub fn innermost_li_immediate_parent_is_list_item(chain: &[EnclosingContainer]) -> bool {
    let mut innermost_pos: Option<usize> = None;
    for (i, c) in chain.iter().enumerate() {
        if matches!(c, EnclosingContainer::ListItem(_)) {
            innermost_pos = Some(i);
        }
    }
    let Some(pos) = innermost_pos else {
        return false;
    };
    if pos == 0 {
        return false;
    }
    matches!(chain[pos - 1], EnclosingContainer::ListItem(_))
}

/// Byte-length of the *active container prefix* that introduces a line
/// inside the innermost entry of `chain` — i.e. the byte count
/// contributed by every container above the innermost. Thin wrapper
/// over [`chain_continuation_prefix_bytes`] applied to the
/// outer-only slice.
///
/// Callers that want to insert / remove content "inside" the innermost
/// container without disturbing the outer scope's prefix use
/// `line_start + chain_outer_prefix_bytes(...)` as their insertion
/// point.
pub fn chain_outer_prefix_bytes(chain: &[EnclosingContainer]) -> usize {
    if chain.is_empty() {
        return 0;
    }
    let last = chain.len() - 1;
    chain_continuation_prefix_bytes(&chain[..last])
}

/// Chain-aware variant of [`pair_at_end`]: if `cursor` sits at the end
/// of a structural pair for `chain`, return the pair's start byte.
///
/// Two pair shapes, picked by the chain's innermost container:
///
/// 1. **Symmetric `\n[prefix]\n[prefix]`** (chain ends in `BlockQuote`):
///    the depth-D pair shape used inside any chain that ends in a BQ
///    scope. `[prefix]` is [`chain_continuation_prefix`] of the full
///    chain — alternating LI indents and `> ` BQ markers. A depth-1 BQ
///    pair inside a `1. ` item appears as `\n   > \n   > ` (12 bytes)
///    and is recognized as one atomic structural pair.
/// 2. **Asymmetric `\n\n[prefix]`** (chain ends in `ListItem`, or is
///    empty with a non-empty prefix): the canonical paragraph-break
///    shape inside an LI without a deeper BQ scope. The break itself
///    is a top-level `\n\n`; the LI's continuation indent re-enters
///    the item on the new row. For a chain `[LI]` of `1. ` width
///    (3 bytes) the shape is `\n\n   ` (5 bytes). The forbidden-
///    position predicate already recognizes these bytes as
///    pair-interior; this branch ensures the delete path agrees so
///    Backspace removes the whole shape in one keystroke instead of
///    eating one indent space per press.
///
/// Path A in `bugs.md::backspace_on_empty_bq_paragraph_in_li_eats_hidden_chars`
/// — extending the atomic-pair-delete predicate to be chain-aware
/// rather than introducing a new BQ-outdent gesture, so a single
/// Backspace at the end of the trailing pair shape removes all
/// `2 + chain_prefix * 2` bytes (or `2 + chain_prefix` bytes in the
/// asymmetric LI-trailing form) and leaves the previous paragraph as
/// the trailing block of the surrounding scope.
pub fn pair_at_end_for_chain(
    bytes: &[u8],
    cursor: usize,
    chain: &[EnclosingContainer],
) -> Option<usize> {
    let (blank_prefix, content_prefix) = chain_pair_shape(chain);
    let blank_bytes = blank_prefix.as_bytes();
    let content_bytes = content_prefix.as_bytes();
    // Reject the trivial pair shape `\n\n` with both prefixes empty —
    // that's the depth-0 top-level break already handled by
    // [`pair_at_end`].
    if blank_bytes.is_empty() && content_bytes.is_empty() {
        return None;
    }
    let mut q = cursor;
    // Walk back over `[content_prefix] \n [blank_prefix] \n` (in
    // reverse: content_prefix → \n → blank_prefix → \n).
    if !walk_back_exact(bytes, &mut q, content_bytes) {
        return None;
    }
    if !walk_back_required_newline(bytes, &mut q) {
        return None;
    }
    if !walk_back_exact(bytes, &mut q, blank_bytes) {
        return None;
    }
    if !walk_back_required_newline(bytes, &mut q) {
        return None;
    }
    Some(q)
}

/// Forward analog of [`pair_at_end_for_chain`].
pub fn pair_at_start_for_chain(
    bytes: &[u8],
    cursor: usize,
    chain: &[EnclosingContainer],
) -> Option<usize> {
    let (blank_prefix, content_prefix) = chain_pair_shape(chain);
    let blank_bytes = blank_prefix.as_bytes();
    let content_bytes = content_prefix.as_bytes();
    if blank_bytes.is_empty() && content_bytes.is_empty() {
        return None;
    }
    let mut q = cursor;
    if !walk_forward_required_newline(bytes, &mut q) {
        return None;
    }
    if !walk_forward_exact(bytes, &mut q, blank_bytes) {
        return None;
    }
    if !walk_forward_required_newline(bytes, &mut q) {
        return None;
    }
    if !walk_forward_exact(bytes, &mut q, content_bytes) {
        return None;
    }
    Some(q)
}

/// Match `expected` exactly at the bytes ending at `*q`, advancing `q`
/// backward past those bytes on success. Empty `expected` succeeds
/// vacuously without moving `q`.
fn walk_back_exact(bytes: &[u8], q: &mut usize, expected: &[u8]) -> bool {
    if expected.is_empty() {
        return true;
    }
    if *q < expected.len() {
        return false;
    }
    let start = *q - expected.len();
    if &bytes[start..*q] != expected {
        return false;
    }
    *q = start;
    true
}

/// Forward analog of [`walk_back_exact`].
fn walk_forward_exact(bytes: &[u8], q: &mut usize, expected: &[u8]) -> bool {
    if expected.is_empty() {
        return true;
    }
    if *q + expected.len() > bytes.len() {
        return false;
    }
    let end = *q + expected.len();
    if &bytes[*q..end] != expected {
        return false;
    }
    *q = end;
    true
}

/// Deepest blockquote nesting that contains `cursor`. Boundary equality
/// (`cursor == range.end`) treats the post-construct caret as still
/// inside, matching the delimiter-visibility rule the renderer uses.
pub fn blockquote_depth_at(markdown: &str, cursor: usize) -> usize {
    chain_blockquote_depth(&enclosing_containers_at(markdown, cursor))
}

// ---------------------------------------------------------------------------
// List ranges and item context
// ---------------------------------------------------------------------------

/// Byte ranges of every list's *interior* — used to exempt
/// list-internal `\n` bytes from soft-break promotion. Inside a list
/// pulldown handles line structure (item separators, continuation
/// indent, hard-break-with-indent inside items) and the buffer's own
/// `\n` discipline doesn't apply for those bytes.
///
/// We deliberately *exclude* the trailing `\n` of every list range:
/// that newline is the boundary with the next top-level block, and
/// the editor-wide rule says any structural-block boundary uses
/// `\n\n`. Without this trim, `- item\nparagraph` would have its
/// boundary `\n` exempt, leaving the soft-break rule unable to
/// enforce the `\n\n` separator between list and paragraph.
///
/// Implementation: walks the parsed tree. Pulldown is fast enough
/// at our buffer sizes that calling it inside `enforce_invariants`
/// per-update isn't visible.
pub fn list_content_ranges(markdown: &str) -> Vec<Range<usize>> {
    let tree = crate::parser::parse(markdown);
    list_content_ranges_in_tree(&tree, markdown.as_bytes())
}

/// Variant of [`list_content_ranges`] for callers that already hold
/// a parse tree.
pub fn list_content_ranges_in_tree(
    tree: &[crate::syntax::SyntaxNode],
    bytes: &[u8],
) -> Vec<Range<usize>> {
    fn collect(nodes: &[crate::syntax::SyntaxNode], bytes: &[u8], out: &mut Vec<Range<usize>>) {
        for n in nodes {
            if matches!(n.kind, crate::syntax::NodeKind::List { .. }) {
                let mut end = n.range.end;
                // Trim every trailing `\n` so the boundary with the
                // next block is not exempt. Pulldown ranges
                // typically include either one trailing `\n`
                // (tight) or two (loose with paragraph break before
                // the next block); both should be promotable to a
                // canonical `\n\n` boundary by the soft-break rule.
                while end > n.range.start && bytes[end - 1] == b'\n' {
                    end -= 1;
                }
                out.push(n.range.start..end);
            }
            collect(&n.children, bytes, out);
        }
    }
    let mut out = Vec::new();
    collect(tree, bytes, &mut out);
    out
}

/// `Some(item)` when `cursor` falls inside a list item — the
/// *innermost* one. Thin wrapper that pulls the deepest list-item
/// entry off the chain returned by [`enclosing_containers_at`].
/// Boundary equality treats end-of-item as still inside (so Enter at
/// the end of `- foo` produces a new item rather than escaping the
/// list).
fn innermost_list_item_at(markdown: &str, cursor: usize) -> Option<ListItemContext> {
    chain_innermost_list_item(&enclosing_containers_at(markdown, cursor)).cloned()
}

/// Walk the parse tree to find the unordered-list bullet character
/// for the item at `item_index` of any list. Used as a fallback when
/// we want the same bullet style the rest of the list uses; we
/// don't currently thread the marker char through `ListItemContext`.
fn bullet_for_item_index(markdown: &str, item_index: usize) -> Option<u8> {
    fn walk(nodes: &[crate::syntax::SyntaxNode], target: usize, source: &str) -> Option<u8> {
        for n in nodes {
            if let crate::syntax::NodeKind::List {
                kind: crate::syntax::ListKind::Unordered,
            } = n.kind
            {
                let mut idx = 0;
                for child in &n.children {
                    if let crate::syntax::NodeKind::ListItem { marker_range } = &child.kind {
                        if idx == target {
                            let bytes = source.as_bytes();
                            for &b in &bytes[marker_range.clone()] {
                                if b == b'-' || b == b'*' || b == b'+' {
                                    return Some(b);
                                }
                            }
                            return None;
                        }
                        idx += 1;
                    }
                }
            }
            if let Some(b) = walk(&n.children, target, source) {
                return Some(b);
            }
        }
        None
    }
    walk(&crate::parser::parse(markdown), item_index, markdown)
}

// ---------------------------------------------------------------------------
// Public Enter / Shift+Enter insertions
// ---------------------------------------------------------------------------

/// Source string to insert when the user presses Enter at `cursor`.
/// Encapsulates the routing across all container kinds we support so
/// keyboard, IME, paste-derived, and programmatic dispatch all share
/// one rule.
///
/// Routing (innermost wins):
///
/// - Inside a fenced code block: `\n` + the chain's continuation
///   prefix. Code uses `\n` as a line separator (promoting to
///   `\n\n` would visually duplicate every keystroke), but the new
///   row still has to carry the outer BQ / LI continuation bytes
///   so the code body stays inside its enclosing scope.
///   Auto-close-fence (see [`auto_close_fence_edit`]) intercepts
///   the *unterminated* case before this function fires, so by the
///   time we reach the in-fence branch the fence has a clean
///   closer below.
/// - Inside a list item: `\n` + the active blockquote continuation
///   prefix (if any) + the *outer* list-items' accumulated indent +
///   the innermost item's next marker. The new line lands as a
///   sibling at the cursor's nesting depth.
/// - Inside a blockquote (without an enclosed list): the depth-D
///   paragraph-break pair `\n[prefix]\n[prefix]` — its two halves
///   render as a single paragraph_gap visually.
/// - Top level: `\n\n`, the top-level paragraph break.
pub fn enter_insertion(markdown: &str, cursor: usize) -> String {
    if is_in_fenced_code(markdown, cursor) {
        let chain = enclosing_containers_at(markdown, cursor);
        let prefix = chain_continuation_prefix(&chain);
        return format!("\n{prefix}");
    }
    let chain = enclosing_containers_at(markdown, cursor);
    // Route on the *innermost* (last) container, not "any LI in the
    // chain". For a chain like `[LI, BQ, LI, BQ]` the cursor's true
    // innermost is the BQ — Enter should produce a depth-D
    // chain-pair, not a sibling list-item, because the cursor isn't
    // at the LI's content edge.
    match chain.last() {
        Some(EnclosingContainer::ListItem(item)) => {
            // New sibling at the innermost LI's depth: continuation
            // prefix of the *outer* chain (everything above the
            // innermost LI) followed by the new item's marker.
            let outer = &chain[..chain.len() - 1];
            let outer_prefix = chain_continuation_prefix(outer);
            format!("\n{outer_prefix}{}", item.next_marker_text(markdown))
        }
        Some(EnclosingContainer::BlockQuote { .. }) => {
            // Depth-D paragraph-break pair, with both halves carrying
            // the full chain-aware continuation prefix so list-item
            // ancestors keep contributing their indent.
            let prefix = chain_continuation_prefix(&chain);
            format!("\n{prefix}\n{prefix}")
        }
        None => "\n\n".to_string(),
    }
}

/// `Some(edit)` when the user pressed Enter at the end of an empty
/// blockquote paragraph row, asking to leave the innermost BQ scope.
///
/// The shape: cursor sits at the end of a chain-aware pair
/// (`\n{blank}\n{content}`, see [`pair_at_end_for_chain`]) whose chain
/// ends in `BlockQuote`. The edit replaces the pair with the
/// reduced-chain pair shape so the trailing row drops one BQ marker;
/// when the chain has no remaining BQs the new shape is the top-level
/// `\n\n`.
///
/// This is the Enter analog of the chain-aware BQ-outdent gesture on
/// Backspace (the matching code lives in `update::delete_backward` and
/// uses the same detector + replacement). Without this gesture every
/// Enter on an empty `> ` row just appends another `\n> \n> ` pair, so
/// the user has no Enter-only path out of a blockquote — Backspace
/// would be the only way. Mirrors [`empty_item_exit_edit`] for list
/// items: empty-LI Enter and empty-BQ Enter both decrease nesting
/// depth by one.
pub fn empty_bq_paragraph_exit_edit(markdown: &str, cursor: usize) -> Option<DepthDecreaseEdit> {
    let bytes = markdown.as_bytes();
    let chain = enclosing_containers_at(markdown, cursor);
    if !matches!(chain.last(), Some(EnclosingContainer::BlockQuote { .. })) {
        return None;
    }
    let pair_start = pair_at_end_for_chain(bytes, cursor, &chain)?;
    let new_chain = &chain[..chain.len() - 1];
    let (blank_prefix, content_prefix) = chain_pair_shape(new_chain);
    let replacement = format!("\n{blank_prefix}\n{content_prefix}");
    let new_cursor = pair_start + replacement.len();
    Some(DepthDecreaseEdit {
        range: pair_start..cursor,
        replacement,
        cursor: new_cursor,
    })
}

/// `Some(edit)` when the user pressed Enter inside an *unterminated*
/// fenced code block. The edit injects a matching closer below the
/// cursor so the buffer never carries an unterminated fence after a
/// keystroke, and places the cursor on a body row between the existing
/// opener and the new closer.
///
/// Why intercept Enter rather than wait for the user to type the
/// closer? Without a closer, pulldown sees the construct as extending
/// to EOF; every subsequent keystroke is "still inside an open code
/// block", which forces every other rule (BQ-prefix normalize,
/// soft-break promote, render-walker synth-paragraph injection) to
/// special-case the unterminated state. Auto-closing the fence at the
/// first natural Enter eliminates that whole class of edge cases.
///
/// The closer matches the opener's fence char (`` ` `` or `~`) and at
/// least its length; both the body row and the closer row carry the
/// cursor's chain continuation prefix so the new bytes stay inside
/// every enclosing BQ / LI scope.
pub fn auto_close_fence_edit(markdown: &str, cursor: usize) -> Option<DepthDecreaseEdit> {
    let blocks = fenced_code_blocks(markdown);
    let block = blocks.iter().find(|b| {
        cursor >= b.range.start
            && (cursor < b.range.end || (!b.terminated && cursor == b.range.end))
    })?;
    if block.terminated || block.opener_fence_len == 0 {
        return None;
    }

    let closer: String =
        std::iter::repeat_n(block.opener_fence_char as char, block.opener_fence_len).collect();

    // The new body / closer rows take the cursor's chain so
    // alternating chains pick up every interleaved indent + marker
    // segment, not just the BQ markers.
    let chain = enclosing_containers_at(markdown, cursor);
    let prefix = chain_continuation_prefix(&chain);

    let body_row = format!("\n{prefix}");
    let closer_row = format!("\n{prefix}{closer}");
    let replacement = format!("{body_row}{closer_row}");
    let new_cursor = cursor + body_row.len();

    Some(DepthDecreaseEdit {
        range: cursor..cursor,
        replacement,
        cursor: new_cursor,
    })
}

// ---------------------------------------------------------------------------
// Depth-change edits (Tab / Shift+Tab)
// ---------------------------------------------------------------------------

/// Edits that increase the nesting level of the list item containing
/// `cursor` by one. Returns `None` if the cursor isn't inside a list
/// item or the item has no previous sibling at the same depth (since
/// there's nothing to nest under).
///
/// Two flavors of edit, applied together:
///
/// 1. **Indent insertion** — insert `previous_sibling_marker_width`
///    spaces at the start of every line of the cursor's item
///    (marker line + any continuations). Pulldown's re-parse
///    classifies the now-deeper-indented item under the previous
///    sibling.
/// 2. **Ordered marker rewrite** — for ordered items, *also*
///    rewrite the item's marker to `1. `. CommonMark forbids an
///    ordered list with start > 1 from interrupting (in this
///    context: opening as a nested list inside another item's
///    content), so without this rewrite the post-Tab source like
///    `1. one\n   2. two` parses as continuation text rather than
///    a nested list. The renumbering pass downstream handles the
///    case where the new nested item joins an existing nested
///    list with prior siblings (rewriting `1.` back to `2.`,
///    etc.).
pub fn list_item_indent_edits(markdown: &str, cursor: usize) -> Option<Vec<SourceEdit>> {
    let chain = enclosing_containers_at(markdown, cursor);
    let innermost = chain_innermost_list_item(&chain)?;
    let prev_marker_width = previous_sibling_marker_width(markdown, &innermost.item_range)?;
    if prev_marker_width == 0 {
        return None;
    }
    let bytes = markdown.as_bytes();
    let line_starts = item_line_starts(bytes, &innermost.item_range);
    let pad = " ".repeat(prev_marker_width);
    // Step past the active container-prefix bytes contributed by the
    // chain entries *above the innermost LI* before inserting indent.
    // For an item inside a blockquote-inside-LI chain like
    // `[LI(outer), LI(inner)]` (cursor in inner LI), that's the outer
    // LI's marker_width — so the insertion lands at the inner LI's
    // own marker position, not before the outer LI's indent. Locating
    // the LI's position in the chain (rather than slicing by
    // `chain.len() - 1`) handles cursors whose innermost is something
    // other than an LI (e.g. a BQ inside the inner LI), so we always
    // insert at the right column for the LI being nested.
    let innermost_li_pos = chain
        .iter()
        .enumerate()
        .filter_map(|(i, c)| matches!(c, EnclosingContainer::ListItem(_)).then_some(i))
        .next_back()?;
    let outer_skip = chain_continuation_prefix_bytes(&chain[..innermost_li_pos]);
    let mut edits = SourceEditList::new();
    for ls in line_starts {
        // Insertion point = line_start + outer_skip. We use
        // `list_item_strip_range`'s `start` to share the offset
        // calculation with the Shift+Tab dedent path; the strip
        // range itself is unused for insertions.
        let insert_at = list_item_strip_range(bytes, ls, outer_skip, 0).start;
        edits.push(SourceEdit {
            range: insert_at..insert_at,
            replacement: pad.clone(),
        });
    }

    if innermost.is_ordered() {
        edits.push(SourceEdit {
            range: innermost.marker_range.clone(),
            replacement: "1. ".to_string(),
        });
    }

    // The builder sorts and resolves overlap; the pad-insert at
    // line_starts[0] and the marker-rewrite both sit at the item's
    // first byte, and the builder's stable sort places insertions
    // (zero-length ranges) before replacements at the same start.
    Some(edits.finish())
}

/// Edits that decrease the nesting level of the list item
/// containing `cursor` by one. This is the shared core of the
/// "decrease depth" gestures — Tab's symmetric counterpart, used
/// by Shift+Tab and Backspace-at-start-of-item.
///
/// - **Top-level item** (no parent list item): become a paragraph
///   in the surrounding scope. The edit replaces the line's
///   leading separator + the marker bytes with a clean paragraph
///   break:
///   - `\n\n` at the document's top level (or `""` when the item
///     is the first thing in the buffer — no leading separator
///     needed).
///   - `\n[bq_prefix]\n[bq_prefix]` when the item lives inside a
///     blockquote — the depth-D pair shape that keeps the
///     resulting paragraph in the BQ scope.
///
///   The item's content past the marker is preserved unchanged.
///   Without the explicit `\n\n`, dropping just the marker would
///   leave a lazy-continuation source like `1. one\nfoo` that
///   the canonicalizer then re-promotes to `1. one  \n   foo`,
///   which is *not* what "the item became a paragraph" should
///   look like.
/// - **Nested item**: remove `parent_marker_width` leading spaces
///   from the start of every line inside the item, so the item
///   becomes a sibling of its former parent.
///
/// In *both* cases, continuation lines past the marker line are
/// also stripped of `marker_width` leading spaces — the item's own
/// marker no longer exists, so its continuation indent shouldn't
/// either. Without this strip, a Backspace at the start of a
/// top-level item that has nested children would leave the
/// children's lines stranded with leading whitespace that no
/// longer corresponds to any container, which pulldown then
/// re-parses as either lazy continuations or shallower-nested
/// structures with arbitrary leftover indent.
///
/// Returns `None` outside of a list.
pub fn list_item_dedent_edits(markdown: &str, cursor: usize) -> Option<Vec<SourceEdit>> {
    let chain = enclosing_containers_at(markdown, cursor);
    let items: Vec<&ListItemContext> = chain
        .iter()
        .filter_map(|c| match c {
            EnclosingContainer::ListItem(ctx) => Some(ctx),
            _ => None,
        })
        .collect();
    let innermost = *items.last()?;
    let bytes = markdown.as_bytes();

    // Treat as "drop the marker" whenever the innermost LI's
    // *immediate* enclosing container in the chain isn't another
    // list-item. That covers two distinct shapes:
    //   - A truly top-level item (chain ends in just `[..., LI]`
    //     with nothing or only BQs above) — the existing top-level
    //     dedent semantics.
    //   - A list-item directly wrapped by a blockquote inside a
    //     deeper container chain (`[..., outer-LI, BQ, inner-LI]`).
    //     Here the inner-LI has no list-item parent to "fall back
    //     to" via indent-stripping, but its surrounding BQ scope
    //     already carries the line prefix; we just drop the marker.
    if !innermost_li_immediate_parent_is_list_item(&chain) {
        // Top-level dedent: replace the leading separator + marker
        // with the surrounding scope's canonical paragraph break.
        // The continuation prefix is taken at the marker_range.start
        // byte, not the original cursor — the marker line's enclosing
        // chain may differ from the cursor's if the cursor is in a
        // continuation line of a nested construct. We use the
        // chain-aware [`chain_continuation_prefix`] so list-item
        // ancestors contribute their indent (mirrors the
        // [`enter_insertion`] LI branch's `outer = chain - innermost
        // LI` pattern). The innermost LI itself is dropped from the
        // chain since it's the one being removed by this dedent.
        let marker_chain = enclosing_containers_at(markdown, innermost.marker_range.start);
        let outer_chain: &[EnclosingContainer] = match marker_chain.last() {
            Some(EnclosingContainer::ListItem(_)) => &marker_chain[..marker_chain.len() - 1],
            _ => &marker_chain[..],
        };
        let scope_prefix = chain_continuation_prefix(outer_chain);
        let line_start = line_start_offset(bytes, innermost.marker_range.start);
        let preceded_by_newline = line_start > 0 && bytes[line_start - 1] == b'\n';

        let (range, replacement) = if !preceded_by_newline {
            // Marker sits at the buffer start *or* mid-line within
            // an outer container's prefix bytes (e.g. `- > - > inner`
            // — the inner LI's marker is on the same line as the
            // outer LI's marker, after its `- > ` preamble). No
            // leading separator to absorb; just drop the marker
            // bytes themselves and leave the outer scope intact.
            (innermost.marker_range.clone(), String::new())
        } else {
            // Fresh line preceded by `\n`. Replace the separator +
            // marker with the surrounding scope's paragraph break.
            let prev_sep_start = line_start - 1;
            let rep = if scope_prefix.is_empty() {
                "\n\n".to_string()
            } else {
                format!("\n{scope_prefix}\n{scope_prefix}")
            };
            (prev_sep_start..innermost.marker_range.end, rep)
        };

        let mut edits = vec![SourceEdit { range, replacement }];
        // Strip this-item's marker_width from each continuation
        // line, so any nested-child / continuation indent that
        // belonged to the now-defunct item doesn't survive as
        // orphaned leading whitespace.
        push_continuation_indent_strips(
            bytes,
            &innermost.item_range,
            innermost.marker_width(),
            &mut edits,
        );
        return Some(edits);
    }

    // Nested — remove the immediate parent's marker-width worth of
    // leading spaces from each line of the item.
    let parent = items[items.len() - 2];
    let strip = parent.marker_width();
    if strip == 0 {
        return None;
    }
    // Locate the parent LI's actual position in the chain (not
    // `chain.len() - 2`). When the chain trails with a non-LI entry
    // (e.g. `[outer-LI, inner-LI, BQ]` for Shift+Tab on a row inside
    // a BQ nested in two LIs), the trailing two entries are
    // *inner-LI, BQ* — not parent-LI, innermost-LI. Slicing by
    // `chain.len() - 2` would include the parent LI itself in
    // `above_parent_chain`, double-counting its marker_width and
    // pushing the strip walker past the line terminator into adjacent
    // lines (producing overlapping byte ranges that panic
    // `apply_edits`).
    let parent_pos = chain
        .iter()
        .enumerate()
        .filter_map(|(i, c)| matches!(c, EnclosingContainer::ListItem(_)).then_some(i))
        .nth(items.len() - 2)?;
    let above_parent_chain = &chain[..parent_pos];
    let above_skip = chain_continuation_prefix_bytes(above_parent_chain);
    let line_starts = item_line_starts(bytes, &innermost.item_range);
    let mut edits = SourceEditList::new();
    for ls in line_starts {
        let r = list_item_strip_range(bytes, ls, above_skip, strip);
        if !r.is_empty() {
            edits.push(SourceEdit {
                range: r,
                replacement: String::new(),
            });
        }
    }
    let edits = edits.finish();
    if edits.is_empty() {
        return None;
    }
    Some(edits)
}

/// Range of bytes to strip (Shift+Tab) or to insert at (Tab — use the
/// returned range's `start` as the insertion point) on a single line
/// of a list item.
///
/// Tab and Shift+Tab share this primitive so they can never disagree:
/// the indent insertion point and the dedent strip start are the same
/// byte (`line_start + outer_skip`). The strip walker is bounded to
/// the line's terminator (`\n` or end of buffer) so a strip on an
/// empty / shorter-than-expected line cannot leak into the next
/// line's bytes — which is what produced the overlapping-edit panic
/// in `[LI, LI, BQ]` chains before refactor C.
///
/// Inputs:
/// - `line_start` is the byte right after the previous `\n` (or 0).
/// - `outer_skip` is the byte count contributed by every container
///   *above* the parent LI for Shift+Tab (chain entries at depth
///   shallower than the parent), or the chain's outer prefix for
///   Tab. Both equal "the offset where the parent's prefix begins on
///   this line".
/// - `max_strip` is the maximum number of leading spaces to consume
///   past `outer_skip` (= `parent.marker_width()` for Shift+Tab,
///   = `previous_sibling.marker_width()` for Tab).
///
/// The returned range may be empty when:
/// - The line is shorter than `outer_skip` (the line carries less
///   prefix than expected — typically a blank line in the item's
///   range).
/// - The bytes past `outer_skip` aren't spaces (the line carries
///   different content — typically the marker line, where the bytes
///   past outer_skip are the marker itself).
pub fn list_item_strip_range(
    bytes: &[u8],
    line_start: usize,
    outer_skip: usize,
    max_strip: usize,
) -> Range<usize> {
    let line_end = {
        let mut e = line_start;
        while e < bytes.len() && bytes[e] != b'\n' {
            e += 1;
        }
        e
    };
    let strip_start = line_start + outer_skip;
    if strip_start > line_end {
        return strip_start..strip_start;
    }
    let mut end = strip_start;
    let mut removed = 0;
    while end < line_end && bytes[end] == b' ' && removed < max_strip {
        end += 1;
        removed += 1;
    }
    strip_start..end
}

/// Append edits that strip up to `strip` leading-space bytes from
/// each *continuation* line of `item_range` — i.e. every line past
/// the first. The first line carries the marker (handled by the
/// caller); only continuation lines need their leading indent
/// dropped to reflect the new structural depth.
///
/// Edits are appended in source order; combined with the dedent's
/// preceding edit (which sits at or before `item_range.start`),
/// the full edit list is non-overlapping and sorted.
fn push_continuation_indent_strips(
    bytes: &[u8],
    item_range: &Range<usize>,
    strip: usize,
    out: &mut Vec<SourceEdit>,
) {
    if strip == 0 {
        return;
    }
    let line_starts = item_line_starts(bytes, item_range);
    for ls in line_starts.into_iter().skip(1) {
        let mut end = ls;
        let mut removed = 0;
        while end < bytes.len() && bytes[end] == b' ' && removed < strip {
            end += 1;
            removed += 1;
        }
        if end > ls {
            out.push(SourceEdit {
                range: ls..end,
                replacement: String::new(),
            });
        }
    }
}

/// Marker width of the list item *immediately preceding* the item
/// at `item_start` within the same list. `None` when the item is
/// the first of its list (no previous sibling) or `cursor`-derived
/// `item_start` doesn't match any item.
fn previous_sibling_marker_width(markdown: &str, item_range: &Range<usize>) -> Option<usize> {
    fn walk(nodes: &[crate::syntax::SyntaxNode], target_start: usize) -> Option<usize> {
        for node in nodes {
            if let crate::syntax::NodeKind::List { .. } = &node.kind {
                let mut prev_width: Option<usize> = None;
                for child in &node.children {
                    if let crate::syntax::NodeKind::ListItem { marker_range } = &child.kind {
                        if child.range.start == target_start {
                            return prev_width;
                        }
                        prev_width = Some(marker_range.end - marker_range.start);
                    }
                }
                // Not found at this list's level — try children's
                // nested lists.
                for child in &node.children {
                    if let Some(w) = walk(&child.children, target_start) {
                        return Some(w);
                    }
                }
            } else if let Some(w) = walk(&node.children, target_start) {
                return Some(w);
            }
        }
        None
    }
    walk(&crate::parser::parse(markdown), item_range.start)
}

/// Byte offsets of every line within the item (the line containing
/// the marker plus every continuation line). Each line_start is the
/// byte right after the previous `\n` (or the buffer start for the
/// very first line).
fn item_line_starts(bytes: &[u8], item_range: &Range<usize>) -> Vec<usize> {
    let mut first_line_start = item_range.start;
    while first_line_start > 0 && bytes[first_line_start - 1] != b'\n' {
        first_line_start -= 1;
    }
    let mut starts = vec![first_line_start];
    let mut p = item_range.start;
    while p < item_range.end {
        if bytes[p] == b'\n' && p + 1 < item_range.end {
            starts.push(p + 1);
        }
        p += 1;
    }
    starts
}

/// Source string to insert when the user presses Shift+Enter at
/// `cursor`. Inside a blockquote, the continuation line carries the
/// blockquote prefix. Inside a list item, the continuation carries
/// `marker_width` spaces of indent so the next line stays inside the
/// item per CommonMark's continuation rule (and so the
/// no-lazy-continuation invariant holds without a separate
/// promotion pass).
pub fn line_break_insertion(markdown: &str, cursor: usize) -> String {
    let chain = enclosing_containers_at(markdown, cursor);
    if chain.is_empty() {
        return "  \n".to_string();
    }
    // The continuation line carries the full chain-aware prefix —
    // each LI ancestor contributes its `marker_width` of continuation
    // indent and each BQ contributes its `> ` marker, in chain order.
    // The result is identical to the renderer's per-leaf prefix for
    // the cursor's row, so the hard break stays inside every scope
    // the cursor was in.
    let prefix = chain_continuation_prefix(&chain);
    format!("  \n{prefix}")
}

// ---------------------------------------------------------------------------
// Empty-item / depth-decrease detection
// ---------------------------------------------------------------------------

/// Edit prescribed by an "exit list" (empty-item Enter or
/// backspace-at-start-of-item) intent.
///
/// `range_to_replace` is the source slice the editor should splice
/// out; `replacement` is what to put in its place; `cursor_offset`
/// (relative to `replacement.start()`) is where to land the cursor
/// after the splice.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthDecreaseEdit {
    pub range: Range<usize>,
    pub replacement: String,
    /// Where the cursor lands, in the *post-splice* buffer, given as
    /// an absolute byte offset.
    pub cursor: usize,
}

/// `Some(edit)` when the user pressed Enter on an *empty* list item
/// at `cursor` (cursor at end of an item line whose content beyond
/// the marker is empty).
///
/// The edit decreases the item's nesting depth by one — analogous to
/// blockquote outdent. For a depth-1 (top-level) empty item the
/// result is a paragraph break with the item line removed; for a
/// nested item the marker bytes are dropped and the line becomes a
/// continuation of the parent item.
pub fn empty_item_exit_edit(markdown: &str, cursor: usize) -> Option<DepthDecreaseEdit> {
    let item = innermost_list_item_at(markdown, cursor)?;
    if !item_is_empty_at_cursor(markdown, &item, cursor) {
        return None;
    }
    Some(build_depth_decrease_edit(markdown, &item))
}

/// True when `cursor` sits at the byte right after the marker of
/// the cursor's innermost list item — the point that triggers
/// "Backspace decreases the item's depth" routing.
pub fn cursor_at_item_marker_end(markdown: &str, cursor: usize) -> bool {
    innermost_list_item_at(markdown, cursor)
        .map(|item| cursor == item.marker_range.end)
        .unwrap_or(false)
}

fn item_is_empty_at_cursor(markdown: &str, item: &ListItemContext, cursor: usize) -> bool {
    if cursor != item.item_range.end && !cursor_at_end_of_first_line(markdown, item, cursor) {
        return false;
    }
    let bytes = markdown.as_bytes();
    let content = &bytes[item.marker_range.end..item.item_range.end];
    content
        .iter()
        .all(|&b| b == b' ' || b == b'\t' || b == b'\n')
}

fn cursor_at_end_of_first_line(markdown: &str, item: &ListItemContext, cursor: usize) -> bool {
    let bytes = markdown.as_bytes();
    let mut p = item.marker_range.end;
    while p < item.item_range.end && bytes[p] != b'\n' {
        p += 1;
    }
    cursor == p
}

/// The empty-Enter edit: replace the item's line + leading
/// separator with the surrounding scope's canonical paragraph
/// break, so the user lands at a fresh empty line "outside" the
/// item.
///
/// Distinct from the Backspace / Shift+Tab dedent path — this
/// drops the item's *content extent* (`item_range.end`), which
/// is correct for empty items (nothing to preserve), whereas
/// dedent uses `marker_range.end` so a non-empty item's content
/// survives.
fn build_depth_decrease_edit(markdown: &str, item: &ListItemContext) -> DepthDecreaseEdit {
    let bytes = markdown.as_bytes();
    // Use the chain-aware [`chain_continuation_prefix`] so list-item
    // ancestors contribute their indent (mirrors the
    // [`enter_insertion`] LI branch's `outer = chain - innermost LI`
    // pattern, and the same fix applied in
    // [`list_item_dedent_edits`]). A BQ-only prefix would drop the
    // outer LI indent and let the BQ visually escape an enclosing
    // list when this item is dropped.
    let marker_chain = enclosing_containers_at(markdown, item.marker_range.start);
    let outer_chain: &[EnclosingContainer] = match marker_chain.last() {
        Some(EnclosingContainer::ListItem(_)) => &marker_chain[..marker_chain.len() - 1],
        _ => &marker_chain[..],
    };
    let scope_prefix = chain_continuation_prefix(outer_chain);
    let line_start = line_start_offset(bytes, item.marker_range.start);
    let prev_sep_start = if line_start > 0 && bytes[line_start - 1] == b'\n' {
        line_start - 1
    } else {
        line_start
    };
    let replacement = if prev_sep_start == 0 {
        String::new()
    } else if scope_prefix.is_empty() {
        "\n\n".to_string()
    } else {
        format!("\n{scope_prefix}\n{scope_prefix}")
    };
    let cursor = prev_sep_start + replacement.len();
    DepthDecreaseEdit {
        range: prev_sep_start..item.item_range.end,
        replacement,
        cursor,
    }
}

fn line_start_offset(bytes: &[u8], pos: usize) -> usize {
    let mut p = pos;
    while p > 0 && bytes[p - 1] != b'\n' {
        p -= 1;
    }
    p
}

// ---------------------------------------------------------------------------
// List canonicalization
// ---------------------------------------------------------------------------

/// One byte-range edit produced by `list_normalization_edits`. Edits
/// are emitted in *original* (pre-edit) coordinates and applied
/// sweep-style: the apply step at the call site walks them in order
/// and produces the new buffer plus a cursor remap.
#[derive(Debug, Clone)]
pub struct SourceEdit {
    pub range: Range<usize>,
    pub replacement: String,
}

/// Accumulates [`SourceEdit`]s during construction, then emits a
/// sorted, non-overlapping list ready for `apply_edits`. Producers
/// push edits in arbitrary order; [`finish`](Self::finish) sorts
/// them by `(range.start, range.end)` and resolves any overlap
/// between adjacent edits deterministically — adjacent strips with
/// empty replacements merge into one larger strip, and strictly
/// overlapping non-strip edits drop the later edit (matching the
/// "longer / earlier wins" rule).
///
/// `apply_edits`'s panic on overlap is a defensive invariant; this
/// builder is the friendlier failure mode that resolves the overlap
/// rather than crashing the editor when a producer emits a malformed
/// edit list (the headline case from refactor C, where two strip
/// edits overlapped by one byte in `[LI, LI, BQ]` chains).
#[derive(Debug, Default)]
pub struct SourceEditList {
    edits: Vec<SourceEdit>,
}

impl SourceEditList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, edit: SourceEdit) {
        // Drop true no-ops (zero-length range with empty replacement).
        if edit.range.start == edit.range.end && edit.replacement.is_empty() {
            return;
        }
        self.edits.push(edit);
    }

    pub fn extend(&mut self, iter: impl IntoIterator<Item = SourceEdit>) {
        for e in iter {
            self.push(e);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    /// Sort edits by `(range.start, range.end)` and merge overlapping
    /// strips. Insertions (zero-length ranges) precede replacements
    /// at the same start; consecutive strip edits (empty replacement,
    /// overlapping or adjacent ranges) collapse into one combined
    /// strip. The result satisfies the
    /// `prev.range.end <= next.range.start` invariant `apply_edits`
    /// asserts.
    pub fn finish(mut self) -> Vec<SourceEdit> {
        if self.edits.is_empty() {
            return self.edits;
        }
        self.edits
            .sort_by_key(|e| (e.range.start, e.range.end, !e.replacement.is_empty()));
        let mut out: Vec<SourceEdit> = Vec::with_capacity(self.edits.len());
        for e in self.edits {
            match out.last_mut() {
                Some(prev) if prev.range.end > e.range.start => {
                    // Overlap. Two cases collapse here:
                    //
                    // 1. Both edits are pure strips (empty
                    //    replacement). Merging extends `prev.range`
                    //    forward to `max(prev.end, e.end)`. This is
                    //    the symmetric Tab/Shift+Tab path: when two
                    //    strip producers race on the same line in a
                    //    deeply-nested chain, one combined strip is
                    //    the right semantics.
                    // 2. Otherwise, prefer the longer edit (the one
                    //    whose range covers more bytes). On a tie,
                    //    keep the existing `prev` (earlier in input
                    //    order — typically the one closer to the
                    //    cursor's intended action).
                    if prev.replacement.is_empty() && e.replacement.is_empty() {
                        prev.range.end = prev.range.end.max(e.range.end);
                    } else {
                        let prev_len = prev.range.end - prev.range.start;
                        let e_len = e.range.end - e.range.start;
                        if e_len > prev_len {
                            *prev = e;
                        }
                        // else: drop `e`.
                    }
                }
                _ => out.push(e),
            }
        }
        out
    }
}

/// Detect every pair of *consecutive* hard breaks in the buffer and
/// emit edits that drop both pairs of trailing `  ` (the hard-break
/// markers) — leaving two adjacent `\n`s where the user typed a
/// double Shift+Enter.
///
/// Runs *before* the list / soft-break passes so the resulting
/// `\n\n` is treated as a paragraph break by everything downstream:
/// at top level it's a paragraph separator; inside a blockquote
/// the depth-D pair shape regenerates around it; inside a list
/// item it splits the item into multiple paragraphs.
///
/// "Consecutive" means: a hard break (`  \n`) followed by another
/// hard break (`  \n`) with only whitespace, `>` markers, or tabs
/// between them. Any actual content between disqualifies the pair.
/// (Backslash hard breaks `\\\n` are recognized symmetrically.)
///
/// `cursors` is the live selection's endpoints. When a cursor sits
/// in the inter-hard-break region of a candidate pair we skip the
/// collapse — that's the false-positive shape produced by a single
/// Shift+Enter inside a list item, where `line_break_insertion`'s
/// continuation indent (e.g. `  \n   ` for a marker_width-3 item)
/// abuts the *existing* line's `\n` and mimics a second
/// hard-break trailing. The two-hard-breaks pattern produced by
/// actually typing Shift+Enter twice lands the cursor *past* the
/// second `\n`, so the legitimate case still fires.
pub fn consecutive_hard_break_edits(markdown: &str, cursors: &[usize]) -> Vec<SourceEdit> {
    let bytes = markdown.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'\n' {
            i += 1;
            continue;
        }
        let Some(first_trailing) = hard_break_trailing_at(bytes, i) else {
            i += 1;
            continue;
        };
        // Walk forward over container-continuation bytes and find
        // the next `\n`. If it's also a hard break with only
        // continuation between, the pair is consecutive.
        let mut j = i + 1;
        while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t' || bytes[j] == b'>') {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'\n' {
            i += 1;
            continue;
        }
        let Some(second_trailing) = hard_break_trailing_at(bytes, j) else {
            i += 1;
            continue;
        };

        // Skip if any cursor sits in the inter-hard-break region —
        // that's the typing-flow shape after a single Shift+Enter,
        // where `line_break_insertion`'s continuation indent abuts
        // a pre-existing line break and the buffer just *looks*
        // like two consecutive hard breaks. After two real
        // Shift+Enters the cursor lands past the second `\n`, so
        // the legitimate collapse still fires.
        if cursors.iter().any(|&c| c > i && c <= j) {
            i += 1;
            continue;
        }

        // Found two consecutive hard breaks — emit deletes for both
        // trailing-marker runs. The `\n`s themselves stay (they
        // become the paragraph break) and any container-continuation
        // bytes between them stay too (BQ markers / list indent
        // belong to the surrounding scope, not the hard breaks).
        out.push(SourceEdit {
            range: first_trailing,
            replacement: String::new(),
        });
        out.push(SourceEdit {
            range: second_trailing,
            replacement: String::new(),
        });
        i = j + 1;
    }
    out
}

/// If the `\n` at `nl` is the terminator of a hard break, return the
/// byte range of the *canonical* trailing-marker characters (`  ` or
/// `\\`) that precede it. Otherwise `None`.
///
/// Non-greedy on purpose: we drop only the two `  ` (or one `\\`)
/// that *are* the hard-break marker, never spaces beyond that.
/// Greedy walking would swallow BQ-marker trailing space (`> ` →
/// `>`) and over-collapse list-item indent into the wrong canonical
/// shape.
fn hard_break_trailing_at(bytes: &[u8], nl: usize) -> Option<Range<usize>> {
    if nl >= 2 && bytes[nl - 1] == b' ' && bytes[nl - 2] == b' ' {
        return Some((nl - 2)..nl);
    }
    if nl >= 1 && bytes[nl - 1] == b'\\' {
        return Some((nl - 1)..nl);
    }
    None
}

/// Canonicalize list source — runs as part of `enforce_invariants`,
/// before the soft-break / blockquote-prefix passes.
///
/// What this enforces, per item:
///
/// 1. **Tight separation between items.** A `\n\n` between two
///    items at the same depth collapses to `\n` (loose lists are
///    rendered tight in our editor; the chat renderer's
///    loose-list spacing is the documented divergence cost).
/// 2. **No lazy continuations.** A continuation line missing its
///    indent gets its indent prepended. The previous line's `\n`
///    is rewritten to `  \n` (hard break) so the line break is
///    explicit in source — this closes the soft-break-as-space
///    fidelity gap inside list items.
/// 3. **Indent matches marker width.** Continuation lines (and
///    nested lists) carry exactly `marker_width` leading spaces.
///    Editing `9.` → `10.` re-aligns every continuation by
///    +1 space; editing back re-aligns by −1.
///
/// `cursors` is the live selection's endpoints (anchor + head, or
/// just the cursor offset) — threaded through so per-rule helpers can
/// gate "don't yank the source out from under the user mid-typing"
/// guards. A pass with no cursor input passes `&[]` and behaves as if
/// the cursor were unrelated.
///
/// Returns the edits in source order. The caller applies them and
/// remaps the cursor.
pub fn list_normalization_edits(markdown: &str, cursors: &[usize]) -> Vec<SourceEdit> {
    let tree = crate::parser::parse(markdown);
    list_normalization_edits_in_tree(&tree, markdown.as_bytes(), cursors)
}

/// Variant of [`list_normalization_edits`] for callers that already
/// hold a parse tree.
pub fn list_normalization_edits_in_tree(
    tree: &[crate::syntax::SyntaxNode],
    bytes: &[u8],
    cursors: &[usize],
) -> Vec<SourceEdit> {
    let mut edits = Vec::new();
    walk_normalize_lists(tree, bytes, 0, cursors, &mut edits);
    edits.sort_by_key(|e| e.range.start);
    edits
}

fn walk_normalize_lists(
    nodes: &[crate::syntax::SyntaxNode],
    bytes: &[u8],
    ancestor_indent: usize,
    cursors: &[usize],
    out: &mut Vec<SourceEdit>,
) {
    for n in nodes {
        match &n.kind {
            crate::syntax::NodeKind::List { kind } => {
                let effective_start = effective_list_start(n, *kind);
                normalize_list_node(n, bytes, ancestor_indent, effective_start, cursors, out);
                // Recurse into each item's children, threading the
                // accumulated indent forward: a nested list (or any
                // nested construct) under this item starts its
                // continuations at `ancestor_indent +
                // this_item_target_marker_width`. We use the
                // *post-renumber* marker width for ordered items —
                // changing item `9.` to `10.` widens the marker by
                // one byte, and the nested content's target indent
                // grows accordingly.
                let mut item_idx: u64 = 0;
                for child in &n.children {
                    if let crate::syntax::NodeKind::ListItem { marker_range } = &child.kind {
                        let target_width = match kind {
                            crate::syntax::ListKind::Unordered => {
                                marker_range.end - marker_range.start
                            }
                            crate::syntax::ListKind::Ordered { .. } => {
                                format!("{}. ", effective_start + item_idx).len()
                            }
                        };
                        let inner = ancestor_indent + target_width;
                        walk_normalize_lists(&child.children, bytes, inner, cursors, out);
                        item_idx += 1;
                    }
                }
            }
            _ => walk_normalize_lists(&n.children, bytes, ancestor_indent, cursors, out),
        }
    }
}

/// What number this list's first item should canonically carry.
///
/// For an unordered list, the answer is unused (returns 0). For an
/// ordered list, it's the parsed `start` *unless* the list is a
/// "split orphan" — a single-item list whose `start` is `> 1`. Such
/// lists almost always result from edits that remove one item from
/// the middle of a longer list (Backspace dedent, empty-Enter exit,
/// forward-delete merge): pulldown re-parses the trailing items as
/// a fresh list whose `start` carries the original number, leaving
/// the user with `1. one\n\n3. three` (their list "split" into a
/// `1.` part and a leftover `3.` part).
///
/// The user expectation is that splits restart at 1, so we
/// renumber. Multi-item lists with `start > 1` are preserved —
/// those are far more likely to be intentional (typed multi-item
/// at start>1, or pasted content the user hasn't yet decided how
/// to renumber).
///
/// The cost: a user who *types* `5. foo` and pauses sees their
/// `5` flip to `1` after the post-pass runs. We accept this — it
/// would only matter if the user then typed enough siblings to
/// resemble a `5..N` sequence, and the `1..N` sequence they get
/// is more often what they actually want.
fn effective_list_start(list: &crate::syntax::SyntaxNode, kind: crate::syntax::ListKind) -> u64 {
    match kind {
        crate::syntax::ListKind::Unordered => 0,
        crate::syntax::ListKind::Ordered { start } => {
            let item_count = list
                .children
                .iter()
                .filter(|c| matches!(c.kind, crate::syntax::NodeKind::ListItem { .. }))
                .count();
            if start > 1 && item_count <= 1 {
                1
            } else {
                start
            }
        }
    }
}

fn normalize_list_node(
    list: &crate::syntax::SyntaxNode,
    bytes: &[u8],
    ancestor_indent: usize,
    effective_start: u64,
    cursors: &[usize],
    out: &mut Vec<SourceEdit>,
) {
    let list_kind = match &list.kind {
        crate::syntax::NodeKind::List { kind } => *kind,
        _ => return,
    };
    let items: Vec<&crate::syntax::SyntaxNode> = list
        .children
        .iter()
        .filter(|c| matches!(c.kind, crate::syntax::NodeKind::ListItem { .. }))
        .collect();

    // Per-item canonicalization. Inter-item tightening (a `\n\n+`
    // run between two items collapsing to `\n`) is handled inside
    // `normalize_item` for non-last items: pulldown folds the
    // trailing run into each item's range, so the tighten edit
    // belongs to the *preceding* item.
    let last_idx = items.len().saturating_sub(1);
    for (idx, item) in items.iter().enumerate() {
        let crate::syntax::NodeKind::ListItem { marker_range } = &item.kind else {
            continue;
        };
        // For ordered lists we re-number every item from
        // `effective_start`. Renumbering may change the marker's
        // byte width, which then drives the continuation-indent
        // target — both in the same pass below.
        let target_marker = match list_kind {
            crate::syntax::ListKind::Unordered => None,
            crate::syntax::ListKind::Ordered { .. } => {
                Some(format!("{}. ", effective_start + idx as u64))
            }
        };
        normalize_item(
            item,
            marker_range,
            target_marker.as_deref(),
            ancestor_indent,
            bytes,
            idx == last_idx,
            cursors,
            out,
        );
    }
}

/// One list item's canonicalization. Decomposed into named per-rule
/// helpers so each rule is independently auditable when debugging
/// nesting cases — the original 150-line monolith made it hard to
/// answer "which rule emitted this edit?" mid-stress-test.
///
/// The rules, applied in order, are:
///
/// 1. [`renumber_ordered_marker_if_needed`] — rewrite the marker
///    text to the canonical `<n>. ` for ordered items.
/// 2. [`normalize_marker_content_spacing`] — drop extra spaces
///    between marker and content (cursor-aware).
/// 3. [`tighten_trailing_separator`] — collapse a loose `\n\n+`
///    run between two items to a single `\n` (non-last items only),
///    and report the canonicalization region's effective end.
/// 4. [`walk_item_content_lines`] — per continuation line, promote
///    soft breaks to hard breaks and normalize leading indent to
///    `target_marker_width + ancestor_indent`.
#[allow(clippy::too_many_arguments)]
fn normalize_item(
    item: &crate::syntax::SyntaxNode,
    marker_range: &Range<usize>,
    target_marker: Option<&str>,
    ancestor_indent: usize,
    bytes: &[u8],
    is_last_item: bool,
    cursors: &[usize],
    out: &mut Vec<SourceEdit>,
) {
    renumber_ordered_marker_if_needed(marker_range, target_marker, bytes, out);
    normalize_marker_content_spacing(item, marker_range, bytes, cursors, out);

    // Continuation lines for this item should carry
    // `ancestor_indent + target_marker_width` leading spaces — the
    // sum of every enclosing list-item's marker (using the
    // *post-renumber* width for ordered items).
    let target_marker_width = target_marker
        .map(|s| s.len())
        .unwrap_or_else(|| marker_range.end - marker_range.start);
    let target_indent = ancestor_indent + target_marker_width;
    let indent_string = " ".repeat(target_indent);

    let item_end = tighten_trailing_separator(item, is_last_item, bytes, out);

    walk_item_content_lines(
        item,
        item_end,
        target_indent,
        &indent_string,
        bytes,
        cursors,
        out,
    );
}

/// Rule 1 — emit a renumber edit if the parsed marker doesn't match
/// the canonical `<idx+start>. ` text the caller computed for this
/// position in the list. Only fires for ordered items (unordered
/// passes `target_marker == None`).
///
/// The renumber may also change the marker's *byte width*, which the
/// caller accounts for when computing this item's
/// continuation-line indent target.
fn renumber_ordered_marker_if_needed(
    marker_range: &Range<usize>,
    target_marker: Option<&str>,
    bytes: &[u8],
    out: &mut Vec<SourceEdit>,
) {
    if let Some(target) = target_marker
        && bytes[marker_range.clone()] != *target.as_bytes()
    {
        out.push(SourceEdit {
            range: marker_range.clone(),
            replacement: target.to_string(),
        });
    }
}

/// Rule 2 — strip *extra* spaces between the marker and the start of
/// content on the marker line. The marker already includes one
/// trailing space, so anything past `marker_range.end` and before
/// the first non-space byte is over-spacing that breaks pixel
/// fidelity with the chat renderer.
///
/// Two guards skip the strip:
///
/// - **Whitespace-only marker line** (`extra_end >= line_end`). The
///   line is just `marker + trailing spaces` — typical mid-edit
///   transient (Enter or Tab just landed the cursor there). Yanking
///   the trailing spaces out from under the cursor would feel wrong.
/// - **Cursor sits in the gap.** If any cursor is at
///   `[marker_range.end, extra_end]`, the user is actively typing in
///   that span and stripping would jerk the cursor to an
///   unexpected position. The legitimate "extra space, fix it"
///   cases (cursor at content-start past the gap) still trigger the
///   strip.
fn normalize_marker_content_spacing(
    item: &crate::syntax::SyntaxNode,
    marker_range: &Range<usize>,
    bytes: &[u8],
    cursors: &[usize],
    out: &mut Vec<SourceEdit>,
) {
    let mut line_end = marker_range.end;
    while line_end < item.range.end && bytes[line_end] != b'\n' {
        line_end += 1;
    }
    let mut extra_end = marker_range.end;
    while extra_end < line_end && bytes[extra_end] == b' ' {
        extra_end += 1;
    }
    if extra_end <= marker_range.end || extra_end >= line_end {
        return;
    }
    // Cursor-in-gap guard. Inclusive on both sides — a cursor at
    // marker_range.end is "right after the marker, before the gap"
    // and a cursor at extra_end is "right at the content edge"; in
    // either case stripping shifts the cursor relative to its
    // surrounding text and would feel wrong mid-typing.
    if cursors
        .iter()
        .any(|&c| c >= marker_range.end && c <= extra_end)
    {
        return;
    }
    out.push(SourceEdit {
        range: marker_range.end..extra_end,
        replacement: String::new(),
    });
}

/// Rule 3 — trim the trailing `\n` run from the canonicalization
/// region and, for non-last items, tighten a loose `\n\n+` separator
/// between this item and the next to a single `\n`.
///
/// Pulldown folds 1 or more trailing newlines into each Item's
/// range — the count distinguishes tight vs loose. For non-last
/// items, all but one of those newlines is "loose padding" we
/// collapse. For the last item, the trailing run is *boundary*
/// with the surrounding structure (next top-level block, EOF, the
/// `\n\n` from empty-item-Enter exit, …) — leave it alone so those
/// upstream rules can do their job.
///
/// Returns the canonicalization region's end byte (the original
/// `item.range.end` minus the trailing-newline run), which the
/// content-line walker uses as its scan boundary.
fn tighten_trailing_separator(
    item: &crate::syntax::SyntaxNode,
    is_last_item: bool,
    bytes: &[u8],
    out: &mut Vec<SourceEdit>,
) -> usize {
    let item_start = item.range.start;
    let raw_end = item.range.end;
    let mut item_end = raw_end;
    while item_end > item_start && bytes[item_end - 1] == b'\n' {
        item_end -= 1;
    }
    let trailing_nls = raw_end - item_end;
    if !is_last_item && trailing_nls > 1 {
        // Tighten: keep one `\n`, drop the rest.
        out.push(SourceEdit {
            range: (item_end + 1)..raw_end,
            replacement: String::new(),
        });
    }
    item_end
}

/// Rule 4 — walk every line of the item beyond the marker line and,
/// for content lines, enforce:
///
/// - The preceding `\n` is a hard break (`  \n` or `\\\n`). A bare
///   `\n` in mid-paragraph would render as a soft-break-as-space in
///   the chat renderer, breaking pixel fidelity inside items.
/// - Leading whitespace equals `indent_string` — `target_marker_width`
///   spaces, so continuation content aligns with the marker line's
///   content edge.
///
/// Skipped line classes (each gets its own check):
///
/// - **Blank lines.** Either transient (cursor parked on a fresh
///   hard-break continuation) or one half of a `\n\n` paragraph
///   break. Neither needs hard-break-or-indent enforcement.
/// - **Blockquote-scope continuation lines.** Pulldown's Item
///   ranges sometimes swallow `> ` lines from an enclosing
///   blockquote; those bytes belong to the outer container.
/// - **Lines inside a nested block child** (nested list, nested BQ,
///   nested code block). The nested construct owns its own
///   normalization at its own indent target.
/// - **Lines that *open* a nested construct** (first non-space byte
///   is a list marker, `>`, fence). The construct's own normalize
///   pass handles them at its own indent target; promoting the
///   preceding `\n` to a hard break here would split the construct
///   off the parent.
fn walk_item_content_lines(
    item: &crate::syntax::SyntaxNode,
    item_end: usize,
    target_indent: usize,
    indent_string: &str,
    bytes: &[u8],
    cursors: &[usize],
    out: &mut Vec<SourceEdit>,
) {
    let item_start = item.range.start;
    let mut line_starts: Vec<usize> = Vec::new();
    {
        let mut r = item_start;
        line_starts.push(r);
        while r < item_end {
            if bytes[r] == b'\n' && r + 1 < item_end {
                line_starts.push(r + 1);
            }
            r += 1;
        }
    }

    for line_start in line_starts.iter().copied().skip(1) {
        let line_end = next_line_end(bytes, line_start, item_end);

        let mut indent_end = line_start;
        while indent_end < line_end && bytes[indent_end] == b' ' {
            indent_end += 1;
        }
        if indent_end >= line_end {
            // Blank line. If it carries leftover whitespace (the
            // residue of typing in a multi-paragraph item — e.g.
            // `1. one\n   \n   A` — the indent was matched on the
            // continuation `A` line, but the blank line between
            // paragraphs picks up the same indent and stays
            // there), strip it so the source is the strictly-
            // canonical `\n\n` paragraph break.
            //
            // Two skip conditions:
            //
            // - **Cursor guard.** The same blank-with-whitespace
            //   pattern is also the transient post-Enter / post-Tab
            //   shape, where the cursor is parked on a freshly-
            //   created empty continuation. Stripping the indent
            //   out from under the cursor would yank it back to
            //   column zero mid-typing — wrong.
            //
            // - **Nested-child guard.** A blank line that sits
            //   inside one of this item's nested block children
            //   (e.g. the loose-list separator between two siblings
            //   of a nested list) belongs to the nested construct's
            //   normalization, not this item's. Without this guard
            //   both the outer item's walk *and* the nested
            //   construct's walk emit identical strip edits at the
            //   same byte range, violating `apply_edits`'s
            //   non-overlap invariant — which surfaces as a slice
            //   panic in production.
            if line_end > line_start
                && !line_inside_nested_block_child(item, line_start)
                && !cursors.iter().any(|&c| c >= line_start && c <= line_end)
            {
                out.push(SourceEdit {
                    range: line_start..line_end,
                    replacement: String::new(),
                });
            }
            continue;
        }
        // Blockquote-scope continuation. CommonMark allows up to
        // 3 spaces of indent before `>`.
        if indent_end < line_end && bytes[indent_end] == b'>' && (indent_end - line_start) <= 3 {
            continue;
        }
        if line_inside_nested_block_child(item, line_start) {
            continue;
        }

        // Count `\n`s in the run immediately preceding this line.
        // run = 1: soft break / lazy continuation → promote to hard
        // break. run ≥ 2: paragraph break (or paragraph break +
        // empty paragraphs) — the `\n\n` IS the structural break.
        let mut nl_count = 0;
        let mut effective_prev_nl = line_start;
        while effective_prev_nl > item_start && bytes[effective_prev_nl - 1] == b'\n' {
            effective_prev_nl -= 1;
            nl_count += 1;
        }
        let is_paragraph_break = nl_count >= 2;

        if line_starts_nested_construct(bytes, line_start, line_end, target_indent) {
            continue;
        }

        if !is_paragraph_break && !is_already_hard_break(bytes, effective_prev_nl) {
            out.push(SourceEdit {
                range: effective_prev_nl..effective_prev_nl,
                replacement: "  ".to_string(),
            });
        }

        let current_indent = indent_end - line_start;
        if current_indent != target_indent {
            out.push(SourceEdit {
                range: line_start..indent_end,
                replacement: indent_string.to_string(),
            });
        }
    }
}

/// True when `line_start` falls inside the source range of any of
/// `item`'s nested block children — `List`, `BlockQuote`, or
/// `CodeBlock`. Those children own their own normalization (lists
/// recurse via `walk_normalize_lists`; the others are leaves whose
/// internal whitespace shouldn't be touched), so the parent item's
/// per-line walk skips them.
fn line_inside_nested_block_child(item: &crate::syntax::SyntaxNode, line_start: usize) -> bool {
    item.children.iter().any(|c| {
        matches!(
            c.kind,
            crate::syntax::NodeKind::List { .. }
                | crate::syntax::NodeKind::BlockQuote { .. }
                | crate::syntax::NodeKind::CodeBlock { .. }
        ) && line_start >= c.range.start
            && line_start < c.range.end
    })
}

fn next_line_end(bytes: &[u8], line_start: usize, item_end: usize) -> usize {
    let mut e = line_start;
    while e < item_end && bytes[e] != b'\n' {
        e += 1;
    }
    e
}

fn is_already_hard_break(bytes: &[u8], nl: usize) -> bool {
    if nl >= 2 && bytes[nl - 1] == b' ' && bytes[nl - 2] == b' ' {
        return true;
    }
    if nl >= 1 && bytes[nl - 1] == b'\\' {
        return true;
    }
    false
}

/// Heuristic: does the line at `line_start` open a nested
/// construct (another list, a blockquote, a fenced code block) at
/// the right indent for our parent item? Used to skip hard-break
/// promotion before such lines (CommonMark would parse the hard
/// break as ending the paragraph, which would split the construct
/// off the parent).
fn line_starts_nested_construct(
    bytes: &[u8],
    line_start: usize,
    line_end: usize,
    parent_marker_width: usize,
) -> bool {
    // Look past leading indent.
    let mut p = line_start;
    let mut indent = 0;
    while p < line_end && bytes[p] == b' ' && indent <= parent_marker_width + 3 {
        p += 1;
        indent += 1;
    }
    if p >= line_end {
        return false;
    }
    let c = bytes[p];
    // List markers.
    if c == b'-' || c == b'*' || c == b'+' {
        // Followed by space?
        if p + 1 < line_end && bytes[p + 1] == b' ' {
            return true;
        }
    }
    if c.is_ascii_digit() {
        let mut q = p;
        while q < line_end && bytes[q].is_ascii_digit() {
            q += 1;
        }
        if q < line_end
            && (bytes[q] == b'.' || bytes[q] == b')')
            && q + 1 < line_end
            && bytes[q + 1] == b' '
        {
            return true;
        }
    }
    // Blockquote.
    if c == b'>' {
        return true;
    }
    // Fenced code block opener.
    if c == b'`' || c == b'~' {
        let mut q = p;
        while q < line_end && bytes[q] == c {
            q += 1;
        }
        if q - p >= 3 {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Tests — primitive-level only. Behavioral tests for the rules that
// consume these primitives live in `update.rs` (forbidden-position
// snapping, soft-break promotion) and `tests/behavior.rs`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paragraph_break_interior_is_forbidden() {
        let src = "p1\n\np2";
        assert!(is_forbidden_position(src, 3));
        assert!(!is_forbidden_position(src, 0));
        assert!(!is_forbidden_position(src, 2));
        assert!(!is_forbidden_position(src, 4));
        assert!(!is_forbidden_position(src, 6));
    }

    #[test]
    fn between_two_pairs_is_allowed() {
        let src = "p1\n\n\n\np2";
        assert!(is_forbidden_position(src, 3));
        assert!(!is_forbidden_position(src, 4));
        assert!(is_forbidden_position(src, 5));
    }

    #[test]
    fn six_newline_run_alternates_forbidden_and_allowed() {
        let src = "p1\n\n\n\n\n\np2";
        assert!(is_forbidden_position(src, 3));
        assert!(!is_forbidden_position(src, 4));
        assert!(is_forbidden_position(src, 5));
        assert!(!is_forbidden_position(src, 6));
        assert!(is_forbidden_position(src, 7));
        assert!(!is_forbidden_position(src, 8));
    }

    #[test]
    fn fenced_code_blanks_are_not_forbidden() {
        let src = "```\n\n\n```\n";
        for p in 0..src.len() {
            assert!(
                !is_forbidden_position(src, p),
                "byte {p} unexpectedly forbidden inside code block",
            );
        }
    }

    // ---- List indent forbidden positions -------------------------------

    #[test]
    fn list_marker_interior_is_forbidden() {
        // `- foo`: bytes 0..2 are the hidden marker. Cursor at the
        // marker chars (1) is forbidden; the doc-start edge (0) is
        // exempt and the content edge (2) is the unique landing for
        // the line.
        let src = "- foo";
        assert!(!is_forbidden_position(src, 0));
        assert!(is_forbidden_position(src, 1));
        assert!(!is_forbidden_position(src, 2));
        assert!(!is_forbidden_position(src, 3));
    }

    #[test]
    fn ordered_list_marker_interior_is_forbidden() {
        // `1. foo`: bytes 0..3 are the hidden marker (`1. `). Doc
        // start (0) allowed; marker chars (1, 2) forbidden; content
        // edge (3) allowed.
        let src = "1. foo";
        assert!(!is_forbidden_position(src, 0));
        assert!(is_forbidden_position(src, 1));
        assert!(is_forbidden_position(src, 2));
        assert!(!is_forbidden_position(src, 3));
    }

    #[test]
    fn nested_list_outer_indent_plus_inner_marker_form_one_run() {
        // `- outer\n  - nested`: on the inner item's marker line,
        // bytes 8..10 are outer-continuation indent and bytes
        // 10..12 are the inner marker. Together [8, 12) is the
        // hidden run. The "real beginning of the line" (8), the
        // ancestor-indent interior (9), and the inner marker chars
        // (10, 11) all collapse to the same visible position —
        // forbidden. Only the content edge (12) is allowed.
        let src = "- outer\n  - nested";
        assert!(is_forbidden_position(src, 8));
        assert!(is_forbidden_position(src, 9));
        assert!(is_forbidden_position(src, 10));
        assert!(is_forbidden_position(src, 11));
        assert!(!is_forbidden_position(src, 12));
    }

    #[test]
    fn list_continuation_indent_interior_is_forbidden() {
        // `- foo\n  bar`: bytes 6..8 are the hidden continuation
        // indent. The line-start byte (6) and the indent interior
        // (7) are both forbidden — they share the visible content
        // edge with byte 8.
        let src = "- foo\n  bar";
        assert!(is_forbidden_position(src, 6));
        assert!(is_forbidden_position(src, 7));
        assert!(!is_forbidden_position(src, 8));
    }

    #[test]
    fn sibling_list_item_line_start_is_forbidden() {
        // `1. one\n2. two`: byte 7 is the *real beginning* of the
        // second item's line (right after `\n` at byte 6). The
        // marker `2. ` runs 7..10. All four bytes [7, 10) collapse
        // to the same visible content edge; forbidden. Byte 10 is
        // the unique allowed landing for the line.
        let src = "1. one\n2. two";
        assert!(!is_forbidden_position(src, 6)); // end of line 1
        assert!(is_forbidden_position(src, 7));
        assert!(is_forbidden_position(src, 8));
        assert!(is_forbidden_position(src, 9));
        assert!(!is_forbidden_position(src, 10));
    }

    #[test]
    fn cursor_outside_any_list_is_unaffected() {
        // `plain` — no list, no list-indent forbidden positions
        // should ever fire.
        let src = "plain text";
        for p in 0..=src.len() {
            assert!(!is_forbidden_position(src, p));
        }
    }

    #[test]
    fn pair_at_end_finds_depth_0_pair() {
        let bytes = b"p1\n\np2";
        // cursor at byte 4 = start of p2; the pair `\n\n` ends at 4
        // and starts at 2.
        assert_eq!(pair_at_end(bytes, 4), Some(2));
    }

    #[test]
    fn pair_at_end_finds_depth_1_pair() {
        let bytes = b"> hi\n> \n> ";
        // cursor at end of buffer; the depth-1 pair `\n> \n> ` ends
        // there and starts at byte 4.
        assert_eq!(pair_at_end(bytes, bytes.len()), Some(4));
    }

    // ---- Enter / Shift+Enter routing -----------------------------------

    #[test]
    fn enter_at_top_level_inserts_paragraph_break() {
        assert_eq!(enter_insertion("hello", 5), "\n\n");
    }

    #[test]
    fn enter_inside_blockquote_inserts_pair() {
        assert_eq!(enter_insertion("> hi", 4), "\n> \n> ");
    }

    #[test]
    fn enter_inside_unordered_list_inserts_next_marker() {
        // After "- foo" (end-of-buffer cursor), Enter should produce
        // a new bullet item below.
        assert_eq!(enter_insertion("- foo", 5), "\n- ");
    }

    #[test]
    fn enter_inside_ordered_list_increments_number() {
        assert_eq!(enter_insertion("1. foo", 6), "\n2. ");
    }

    #[test]
    fn enter_inside_ordered_list_starting_at_ten_yields_eleven() {
        // The list starts at 10; the second item should be 11. Enter
        // at the end of the second item produces 12.
        assert_eq!(enter_insertion("10. ten\n11. eleven", 18), "\n12. ");
    }

    #[test]
    fn enter_in_fenced_code_inserts_single_newline() {
        let src = "```\nx\n```";
        assert_eq!(enter_insertion(src, 5), "\n");
    }

    #[test]
    fn enter_inside_list_with_star_marker_repeats_star() {
        assert_eq!(enter_insertion("* foo", 5), "\n* ");
    }

    #[test]
    fn enter_inside_list_inside_blockquote_emits_combined_prefix() {
        // `> - foo`: cursor in foo. Enter should emit `\n> - ` so the
        // new item stays inside the blockquote *and* starts a new
        // bullet.
        assert_eq!(enter_insertion("> - foo", 7), "\n> - ");
    }

    // ---- Enclosing chain ----------------------------------------------

    #[test]
    fn chain_at_top_level_is_empty() {
        assert!(enclosing_containers_at("plain text", 4).is_empty());
    }

    #[test]
    fn chain_inside_blockquote_records_one_bq_level() {
        let chain = enclosing_containers_at("> hi", 4);
        assert_eq!(chain.len(), 1);
        assert!(matches!(chain[0], EnclosingContainer::BlockQuote { .. }));
        assert_eq!(chain_blockquote_depth(&chain), 1);
    }

    #[test]
    fn chain_inside_nested_blockquote_records_two_bq_levels() {
        let chain = enclosing_containers_at("> > deep", 8);
        assert_eq!(chain_blockquote_depth(&chain), 2);
    }

    #[test]
    fn chain_inside_list_item_carries_item_context() {
        let chain = enclosing_containers_at("- foo", 5);
        assert_eq!(chain.len(), 1);
        let inner = chain_innermost_list_item(&chain).unwrap();
        assert_eq!(inner.item_index, 0);
        assert_eq!(inner.marker_width(), 2);
        assert!(!inner.is_ordered());
    }

    #[test]
    fn chain_at_sibling_boundary_picks_later_item() {
        // Cursor at byte 7 sits at end-of-item-0 *and* start-of-item-1
        // by inclusive boundary equality. Pick item 1 (the post-Enter
        // caret semantics).
        let chain = enclosing_containers_at("1. one\n2. two", 7);
        let inner = chain_innermost_list_item(&chain).unwrap();
        assert_eq!(inner.item_index, 1);
    }

    #[test]
    fn chain_continuation_prefix_for_nested_list() {
        // `- outer\n  - inner`: cursor inside inner. The full chain
        // continuation prefix is 4 spaces (outer 2 + inner 2); the
        // outer-only slice (used to position a new sibling at the
        // inner item's depth) is 2 spaces.
        let src = "- outer\n  - inner";
        let chain = enclosing_containers_at(src, src.len());
        assert_eq!(chain_continuation_prefix(&chain), "    ");
        assert_eq!(chain_continuation_prefix_bytes(&chain), 4);
        assert_eq!(chain_outer_prefix_bytes(&chain), 2);
        assert_eq!(chain_continuation_prefix(&chain[..chain.len() - 1]), "  ");
    }

    #[test]
    fn chain_continuation_prefix_alternates_li_and_bq() {
        // `- > - > one`: chain is `[LI(2), BQ, LI(2), BQ]`. Full
        // continuation prefix interleaves LI indents and BQ markers
        // outermost-first.
        let src = "- > - > one";
        let chain = enclosing_containers_at(src, src.len());
        assert_eq!(chain_continuation_prefix(&chain), "  >   > ");
        assert_eq!(chain_continuation_prefix_bytes(&chain), 8);
    }

    #[test]
    fn chain_in_list_inside_blockquote_records_both() {
        // `> - foo` cursor at byte 7: chain should be [BQ, ListItem].
        let chain = enclosing_containers_at("> - foo", 7);
        assert_eq!(chain.len(), 2);
        assert!(matches!(chain[0], EnclosingContainer::BlockQuote { .. }));
        assert!(matches!(chain[1], EnclosingContainer::ListItem(_)));
    }

    #[test]
    fn line_break_top_level_is_plain_hard_break() {
        assert_eq!(line_break_insertion("ab", 2), "  \n");
    }

    #[test]
    fn line_break_inside_blockquote_carries_marker() {
        assert_eq!(line_break_insertion("> hi", 4), "  \n> ");
    }

    #[test]
    fn line_break_inside_list_carries_indent() {
        // Continuation lines inside a list item must carry indent
        // matching the marker width — `- foo` has a 2-byte marker, so
        // a hard break inside the item inserts `  \n` followed by 2
        // spaces of indent. Ordered items inherit a wider marker
        // (`1. ` is 3 bytes) and therefore a wider indent.
        assert_eq!(line_break_insertion("- foo", 5), "  \n  ");
        assert_eq!(line_break_insertion("1. foo", 6), "  \n   ");
        assert_eq!(line_break_insertion("10. tens", 8), "  \n    ");
    }

    // ---- List ranges ---------------------------------------------------

    #[test]
    fn list_content_ranges_excludes_trailing_newline() {
        // The trailing `\n` of a list is the boundary with the next
        // top-level block; excluding it from the exempt range lets
        // `promote_soft_breaks` enforce the canonical `\n\n`
        // separator there.
        let ranges = list_content_ranges("- foo\n- bar\n");
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 0..11);
    }

    #[test]
    fn list_content_ranges_empty_outside_list() {
        assert!(list_content_ranges("just paragraphs").is_empty());
    }

    /// `fenced_code_blocks` reads ranges off pulldown's parse tree.
    /// Walk a battery of fence shapes — terminated, unterminated,
    /// language tags, nested in BQ — and check the projected ranges
    /// agree with what we read from a fresh `parse()` call. The two
    /// must agree by construction (we compute one from the other), but
    /// the test pins the projection so a future change to pulldown's
    /// `CodeBlock` range semantics doesn't silently shift fence
    /// containment.
    #[test]
    fn fenced_code_blocks_match_parse_tree() {
        use crate::parser::parse;
        use crate::syntax::NodeKind;

        let cases = [
            "```\nx\n```\n",
            "```rust\nfn main() {}\n```\n",
            "~~~js\nlet x = 1;\n~~~\n",
            "```\nunterminated\n",
            "para\n\n```\ncode\n```\n\nafter\n",
            "```\n```\n",
            "> ```\n> x\n> ```\n",
            "- > ```rust\n  > let x = 1;\n  > ```\n",
        ];
        for src in cases {
            let blocks = fenced_code_blocks(src);
            let tree = parse(src);
            let mut pulldown_ranges: Vec<Range<usize>> = Vec::new();
            collect_code_ranges(&tree, &mut pulldown_ranges);
            assert_eq!(
                blocks.iter().map(|b| b.range.clone()).collect::<Vec<_>>(),
                pulldown_ranges,
                "ranges mismatch for {src:?}",
            );
        }

        fn collect_code_ranges(nodes: &[crate::syntax::SyntaxNode], out: &mut Vec<Range<usize>>) {
            for n in nodes {
                if matches!(n.kind, NodeKind::CodeBlock { .. }) {
                    out.push(n.range.clone());
                }
                collect_code_ranges(&n.children, out);
            }
        }
    }
}
