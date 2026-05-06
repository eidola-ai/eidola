//! Byte-level structural analysis of the markdown buffer.
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
//! # Why a byte scanner instead of pulldown
//!
//! Pulldown produces the same fence ranges via the parser, but the
//! cheapest place to ask "is this byte inside a fence?" is mid-update,
//! where we may run multiple times per keystroke and don't want to
//! re-tokenize the whole document. The byte scanner here is intentionally
//! cheap and only knows about the construct edges that gate the
//! invariant rules. A regression test pins it agreeing with pulldown on
//! fence ranges (`fenced_ranges_agree_with_pulldown`).
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

/// Find every fenced code block's full byte range — inclusive of the
/// opening fence line, the inner content, and the closing fence line
/// (or end-of-source for an unterminated block).
///
/// Used to exempt those bytes from the soft-break and forbidden-pair
/// rules. Both rules are about paragraph-context structure; they don't
/// apply inside a code block where every `\n` is content.
///
/// CommonMark fence rules (loosely): opening fence is at most 3 leading
/// spaces of indent then 3+ `` ` ``s or `~`s; closing fence matches
/// the same character with at least the same length and only
/// whitespace on the rest of the line.
pub fn fenced_code_content_ranges(bytes: &[u8]) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = Vec::new();
    // (fence_char, fence_len, block_start) while inside an open block.
    // `block_start` is the byte index of the opening fence line's
    // first byte; the emitted range runs through the closing fence
    // line's *trailing `\n`* (or end-of-source for unterminated).
    let mut open: Option<(u8, usize, usize)> = None;

    let mut p = 0;
    while p < bytes.len() {
        let line_start = p;
        while p < bytes.len() && bytes[p] != b'\n' {
            p += 1;
        }
        let line_end = p;
        let line_after = if p < bytes.len() { p + 1 } else { p };

        let mut q = line_start;
        let mut indent = 0;
        while q < line_end && indent < 4 && bytes[q] == b' ' {
            q += 1;
            indent += 1;
        }
        let fence_char = if q < line_end && (bytes[q] == b'`' || bytes[q] == b'~') {
            Some(bytes[q])
        } else {
            None
        };

        if let Some(fc) = fence_char
            && indent < 4
        {
            let mut r = q;
            while r < line_end && bytes[r] == fc {
                r += 1;
            }
            let fence_len = r - q;
            if fence_len >= 3 {
                if let Some((open_fc, open_len, block_start)) = open {
                    if fc == open_fc && fence_len >= open_len {
                        let mut s = r;
                        let mut only_ws = true;
                        while s < line_end {
                            if bytes[s] != b' ' && bytes[s] != b'\t' {
                                only_ws = false;
                                break;
                            }
                            s += 1;
                        }
                        if only_ws {
                            // Range covers the whole block, including
                            // the closing fence line *and* its
                            // trailing `\n`.
                            out.push(block_start..line_after);
                            open = None;
                        }
                    }
                } else {
                    open = Some((fc, fence_len, line_start));
                }
            }
        }

        p = line_after;
    }

    if let Some((_, _, block_start)) = open {
        out.push(block_start..bytes.len());
    }

    out
}

/// `true` if byte index `p` falls inside any fenced code block (opener,
/// content, or closer). Convenience wrapper that recomputes ranges; the
/// hot paths in `update.rs` cache the range list themselves.
pub fn is_in_fenced_code(bytes: &[u8], p: usize) -> bool {
    is_in_ranges(p, &fenced_code_content_ranges(bytes))
}

pub(crate) fn is_in_ranges(p: usize, ranges: &[Range<usize>]) -> bool {
    ranges.iter().any(|r| p >= r.start && p < r.end)
}

// ---------------------------------------------------------------------------
// Forbidden-position detection
// ---------------------------------------------------------------------------

/// Is byte index `p` a forbidden cursor position?
///
/// Forbidden positions are pair interiors of *structural* `\n\n` runs
/// — paragraph breaks and synthetic empty paragraphs. Inside a fenced
/// code block, the same byte pattern is just a blank line of code, so
/// every offset is allowed. Hard breaks (`  \n` / `\\\n`) are
/// in-paragraph content and exempt regardless of code-block context.
pub fn is_forbidden_position(bytes: &[u8], p: usize) -> bool {
    if !is_paragraph_break_interior(bytes, p) {
        return false;
    }
    // Inside a fenced code block, `\n\n` is a literal blank line in
    // the user's code, not a structural pair. The single
    // structural-paragraph-break rule doesn't apply there.
    !is_in_ranges(p, &fenced_code_content_ranges(bytes))
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

pub fn next_allowed_position(bytes: &[u8], mut p: usize) -> usize {
    while p < bytes.len() && is_forbidden_position(bytes, p) {
        p += 1;
    }
    p
}

pub fn prev_allowed_position(bytes: &[u8], mut p: usize) -> usize {
    while p > 0 && is_forbidden_position(bytes, p) {
        p -= 1;
    }
    p
}

/// Snap `p` to the closest allowed position. Forward wins ties. This is
/// the idempotent variant of the snap rule — used by `set_selection`
/// (mouse clicks, host API), where running the same input twice must
/// produce the same output.
pub fn nearest_allowed_position(bytes: &[u8], p: usize) -> usize {
    if !is_forbidden_position(bytes, p) {
        return p;
    }
    let next = next_allowed_position(bytes, p);
    let prev = prev_allowed_position(bytes, p);
    if next.saturating_sub(p) <= p.saturating_sub(prev) {
        next
    } else {
        prev
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
// Active container context (depth + prefix)
// ---------------------------------------------------------------------------

/// Deepest blockquote nesting that contains `cursor`. Falls back on the
/// parser because byte-level scanning can't disambiguate lazy
/// continuations (a paragraph line without a `>` marker that pulldown
/// still treats as inside the blockquote).
///
/// Boundary equality (`cursor == range.end`) treats the post-construct
/// caret as still inside, matching the delimiter-visibility rule the
/// renderer uses.
pub fn blockquote_depth_at(markdown: &str, cursor: usize) -> usize {
    fn walk(nodes: &[crate::syntax::SyntaxNode], cursor: usize, depth: usize) -> usize {
        let mut deepest = depth;
        for node in nodes {
            if cursor < node.range.start || cursor > node.range.end {
                continue;
            }
            let new_depth = if matches!(node.kind, crate::syntax::NodeKind::BlockQuote { .. }) {
                depth + 1
            } else {
                depth
            };
            deepest = deepest.max(walk(&node.children, cursor, new_depth));
        }
        deepest
    }
    walk(&crate::parser::parse(markdown), cursor, 0)
}

/// The blockquote-marker prefix that introduces the line at `cursor`
/// — `"> "` repeated D times where D is the blockquote depth, or
/// `""` at top level. Lists don't add to this prefix: a list-item
/// continuation line uses indentation matching the marker's width,
/// not a literal repeated marker, so the prefix string concept
/// genuinely applies only to per-line-prefix containers (today, just
/// blockquotes).
///
/// Used by `enforce_invariants` when promoting a soft break across
/// blockquote lines: the depth-D pair we insert needs this prefix
/// repeated on both halves.
pub fn blockquote_continuation_prefix(markdown: &str, cursor: usize) -> String {
    let depth = blockquote_depth_at(markdown, cursor);
    "> ".repeat(depth)
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
    let tree = crate::parser::parse(markdown);
    let mut out = Vec::new();
    collect(&tree, markdown.as_bytes(), &mut out);
    out
}

/// `Some(item)` when `cursor` falls inside a list item. The returned
/// kind is the *innermost* item's — used by Enter handling to choose
/// the next item's marker. Boundary equality treats end-of-item as
/// still inside (so Enter at the end of `- foo` produces a new item
/// rather than escaping the list).
fn innermost_list_item_at(markdown: &str, cursor: usize) -> Option<ItemContext> {
    fn walk(
        nodes: &[crate::syntax::SyntaxNode],
        cursor: usize,
        in_list: Option<crate::syntax::ListKind>,
        item_index_in_list: usize,
        deepest: &mut Option<ItemContext>,
    ) {
        for node in nodes {
            if cursor < node.range.start || cursor > node.range.end {
                continue;
            }
            match &node.kind {
                crate::syntax::NodeKind::List { kind } => {
                    let mut idx = 0;
                    for child in &node.children {
                        if matches!(child.kind, crate::syntax::NodeKind::ListItem { .. })
                            && cursor >= child.range.start
                            && cursor <= child.range.end
                        {
                            walk(
                                std::slice::from_ref(child),
                                cursor,
                                Some(*kind),
                                idx,
                                deepest,
                            );
                            idx += 1;
                        } else if matches!(child.kind, crate::syntax::NodeKind::ListItem { .. }) {
                            idx += 1;
                        }
                    }
                }
                crate::syntax::NodeKind::ListItem { marker_range } => {
                    if let Some(list_kind) = in_list {
                        *deepest = Some(ItemContext {
                            list_kind,
                            item_index: item_index_in_list,
                            item_range: node.range.clone(),
                            marker_range: marker_range.clone(),
                        });
                    }
                    walk(&node.children, cursor, None, 0, deepest);
                }
                _ => {
                    walk(&node.children, cursor, in_list, item_index_in_list, deepest);
                }
            }
        }
    }

    let tree = crate::parser::parse(markdown);
    let mut deepest = None;
    walk(&tree, cursor, None, 0, &mut deepest);
    deepest
}

#[derive(Debug, Clone)]
struct ItemContext {
    list_kind: crate::syntax::ListKind,
    /// Zero-based position of this item within its list — used to
    /// compute the next ordered item's number (`start + index + 1`).
    item_index: usize,
    /// Source range of the whole item.
    item_range: Range<usize>,
    /// Source range of the marker bytes (e.g. `- ` or `1. `).
    marker_range: Range<usize>,
}

impl ItemContext {
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

/// Walk the parse tree to find the unordered-list bullet character
/// for the item at `item_index` of any list. Used as a fallback when
/// we want the same bullet style the rest of the list uses; we
/// don't currently thread the marker char through `ItemContext`.
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
/// - Inside a fenced code block: a literal `\n` (code uses `\n` as
///   a line separator; promoting to `\n\n` would visually duplicate
///   every keystroke).
/// - Inside a list item: `\n` + the active blockquote continuation
///   prefix (if any) + the *outer* list-items' accumulated indent +
///   the innermost item's next marker. The new line lands as a
///   sibling at the cursor's nesting depth.
/// - Inside a blockquote (without an enclosed list): the depth-D
///   paragraph-break pair `\n[prefix]\n[prefix]` — its two halves
///   render as a single paragraph_gap visually.
/// - Top level: `\n\n`, the top-level paragraph break.
pub fn enter_insertion(markdown: &str, cursor: usize) -> String {
    let bytes = markdown.as_bytes();
    if is_in_fenced_code(bytes, cursor) {
        return "\n".to_string();
    }
    let bq_prefix = blockquote_continuation_prefix(markdown, cursor);
    if let Some(item) = innermost_list_item_at(markdown, cursor) {
        let outer_indent = " ".repeat(outer_list_indent_at(markdown, cursor));
        return format!(
            "\n{bq_prefix}{outer_indent}{}",
            item.next_marker_text(markdown)
        );
    }
    if !bq_prefix.is_empty() {
        return format!("\n{bq_prefix}\n{bq_prefix}");
    }
    "\n\n".to_string()
}

/// Each list item that encloses `cursor`, outermost first. Used by
/// the depth-change helpers to find the cursor's innermost item, its
/// parent, and any enclosing chain.
#[derive(Debug, Clone)]
struct ItemSpan {
    range: Range<usize>,
    marker_range: Range<usize>,
}

fn enclosing_items_at(markdown: &str, cursor: usize) -> Vec<ItemSpan> {
    fn walk(nodes: &[crate::syntax::SyntaxNode], cursor: usize, out: &mut Vec<ItemSpan>) {
        for node in nodes {
            if cursor < node.range.start || cursor > node.range.end {
                continue;
            }
            if let crate::syntax::NodeKind::ListItem { marker_range } = &node.kind {
                out.push(ItemSpan {
                    range: node.range.clone(),
                    marker_range: marker_range.clone(),
                });
            }
            walk(&node.children, cursor, out);
        }
    }
    let mut out = Vec::new();
    walk(&crate::parser::parse(markdown), cursor, &mut out);
    out
}

/// Sum of marker widths of every list item *enclosing* the cursor's
/// innermost item (the innermost is excluded — the new sibling's
/// marker takes its place). Used by `enter_insertion` to build the
/// indent that puts the new sibling at the same depth.
fn outer_list_indent_at(markdown: &str, cursor: usize) -> usize {
    let items = enclosing_items_at(markdown, cursor);
    if items.len() < 2 {
        0
    } else {
        items[..items.len() - 1]
            .iter()
            .map(|i| i.marker_range.end - i.marker_range.start)
            .sum()
    }
}

/// Sum of marker widths of *every* list item enclosing `cursor`
/// (including the innermost). Used by `line_break_insertion` for
/// the continuation-line indent of a hard break inside an item.
fn total_list_indent_at(markdown: &str, cursor: usize) -> usize {
    enclosing_items_at(markdown, cursor)
        .iter()
        .map(|i| i.marker_range.end - i.marker_range.start)
        .sum()
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
    let items = enclosing_items_at(markdown, cursor);
    let innermost = items.last()?;
    let prev_marker_width = previous_sibling_marker_width(markdown, &innermost.range)?;
    if prev_marker_width == 0 {
        return None;
    }
    let bytes = markdown.as_bytes();
    let line_starts = item_line_starts(bytes, &innermost.range);
    let pad = " ".repeat(prev_marker_width);
    let mut edits: Vec<SourceEdit> = line_starts
        .into_iter()
        .map(|ls| SourceEdit {
            range: ls..ls,
            replacement: pad.clone(),
        })
        .collect();

    if containing_list_is_ordered(markdown, innermost) {
        edits.push(SourceEdit {
            range: innermost.marker_range.clone(),
            replacement: "1. ".to_string(),
        });
    }

    Some(edits)
}

/// True when the item starting at `item.range.start` belongs to an
/// ordered list. Used by Tab to decide whether to rewrite the
/// marker to `1. ` alongside the indent insertion.
fn containing_list_is_ordered(markdown: &str, item: &ItemSpan) -> bool {
    fn walk(nodes: &[crate::syntax::SyntaxNode], target_start: usize) -> Option<bool> {
        for node in nodes {
            if let crate::syntax::NodeKind::List { kind } = &node.kind {
                for child in &node.children {
                    if let crate::syntax::NodeKind::ListItem { .. } = &child.kind
                        && child.range.start == target_start
                    {
                        return Some(matches!(kind, crate::syntax::ListKind::Ordered { .. }));
                    }
                }
                for child in &node.children {
                    if let Some(b) = walk(&child.children, target_start) {
                        return Some(b);
                    }
                }
            } else if let Some(b) = walk(&node.children, target_start) {
                return Some(b);
            }
        }
        None
    }
    walk(&crate::parser::parse(markdown), item.range.start).unwrap_or(false)
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
/// Returns `None` outside of a list.
pub fn list_item_dedent_edits(markdown: &str, cursor: usize) -> Option<Vec<SourceEdit>> {
    let items = enclosing_items_at(markdown, cursor);
    let innermost = items.last()?;
    let bytes = markdown.as_bytes();

    if items.len() == 1 {
        // Top-level dedent: replace the leading separator + marker
        // with the surrounding scope's canonical paragraph break.
        let bq_prefix = blockquote_continuation_prefix(markdown, innermost.marker_range.start);
        let line_start = line_start_offset(bytes, innermost.marker_range.start);
        let prev_sep_start = if line_start > 0 && bytes[line_start - 1] == b'\n' {
            line_start - 1
        } else {
            line_start
        };
        let replacement = if prev_sep_start == 0 {
            String::new()
        } else if bq_prefix.is_empty() {
            "\n\n".to_string()
        } else {
            format!("\n{bq_prefix}\n{bq_prefix}")
        };
        return Some(vec![SourceEdit {
            range: prev_sep_start..innermost.marker_range.end,
            replacement,
        }]);
    }

    // Nested — remove the immediate parent's marker-width worth of
    // leading spaces from each line of the item.
    let parent = &items[items.len() - 2];
    let strip = parent.marker_range.end - parent.marker_range.start;
    if strip == 0 {
        return None;
    }
    let line_starts = item_line_starts(bytes, &innermost.range);
    let mut edits = Vec::new();
    for ls in line_starts {
        // Only remove bytes that are actually leading spaces — if
        // an upstream edit (or paste anomaly) left a line with less
        // indent than expected, we don't drop content.
        let mut end = ls;
        let mut removed = 0;
        while end < bytes.len() && bytes[end] == b' ' && removed < strip {
            end += 1;
            removed += 1;
        }
        if end > ls {
            edits.push(SourceEdit {
                range: ls..end,
                replacement: String::new(),
            });
        }
    }
    if edits.is_empty() {
        return None;
    }
    Some(edits)
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
    let bq_prefix = blockquote_continuation_prefix(markdown, cursor);
    if innermost_list_item_at(markdown, cursor).is_some() {
        // Hard-break continuation inside a list item lines up with
        // *this item's* content column — that's the sum of every
        // enclosing list-item's marker width.
        let indent = " ".repeat(total_list_indent_at(markdown, cursor));
        return format!("  \n{bq_prefix}{indent}");
    }
    if !bq_prefix.is_empty() {
        return format!("  \n{bq_prefix}");
    }
    "  \n".to_string()
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

fn item_is_empty_at_cursor(markdown: &str, item: &ItemContext, cursor: usize) -> bool {
    if cursor != item.item_range.end && !cursor_at_end_of_first_line(markdown, item, cursor) {
        return false;
    }
    let bytes = markdown.as_bytes();
    let content = &bytes[item.marker_range.end..item.item_range.end];
    content
        .iter()
        .all(|&b| b == b' ' || b == b'\t' || b == b'\n')
}

fn cursor_at_end_of_first_line(markdown: &str, item: &ItemContext, cursor: usize) -> bool {
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
fn build_depth_decrease_edit(markdown: &str, item: &ItemContext) -> DepthDecreaseEdit {
    let bytes = markdown.as_bytes();
    let bq_prefix = blockquote_continuation_prefix(markdown, item.marker_range.start);
    let line_start = line_start_offset(bytes, item.marker_range.start);
    let prev_sep_start = if line_start > 0 && bytes[line_start - 1] == b'\n' {
        line_start - 1
    } else {
        line_start
    };
    let replacement = if prev_sep_start == 0 {
        String::new()
    } else if bq_prefix.is_empty() {
        "\n\n".to_string()
    } else {
        format!("\n{bq_prefix}\n{bq_prefix}")
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
/// Returns the edits in source order. The caller applies them and
/// remaps the cursor.
pub fn list_normalization_edits(markdown: &str) -> Vec<SourceEdit> {
    let tree = crate::parser::parse(markdown);
    let bytes = markdown.as_bytes();
    let mut edits = Vec::new();
    walk_normalize_lists(&tree, bytes, 0, &mut edits);
    edits.sort_by_key(|e| e.range.start);
    edits
}

fn walk_normalize_lists(
    nodes: &[crate::syntax::SyntaxNode],
    bytes: &[u8],
    ancestor_indent: usize,
    out: &mut Vec<SourceEdit>,
) {
    for n in nodes {
        match &n.kind {
            crate::syntax::NodeKind::List { kind } => {
                normalize_list_node(n, bytes, ancestor_indent, out);
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
                            crate::syntax::ListKind::Ordered { start } => {
                                format!("{}. ", start + item_idx).len()
                            }
                        };
                        let inner = ancestor_indent + target_width;
                        walk_normalize_lists(&child.children, bytes, inner, out);
                        item_idx += 1;
                    }
                }
            }
            _ => walk_normalize_lists(&n.children, bytes, ancestor_indent, out),
        }
    }
}

fn normalize_list_node(
    list: &crate::syntax::SyntaxNode,
    bytes: &[u8],
    ancestor_indent: usize,
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
        // For ordered lists we re-number every item from the list's
        // start. The target marker text is `<n>.` + ` ` (sticking
        // with the `.` form pulldown gave us; we don't try to
        // preserve `)` if some items used it). Renumbering may
        // change the marker's byte width, which then drives the
        // continuation-indent target — both in the same pass below.
        let target_marker = match list_kind {
            crate::syntax::ListKind::Unordered => None,
            crate::syntax::ListKind::Ordered { start } => Some(format!("{}. ", start + idx as u64)),
        };
        normalize_item(
            item,
            marker_range,
            target_marker.as_deref(),
            ancestor_indent,
            bytes,
            idx == last_idx,
            out,
        );
    }
}

fn normalize_item(
    item: &crate::syntax::SyntaxNode,
    marker_range: &Range<usize>,
    target_marker: Option<&str>,
    ancestor_indent: usize,
    bytes: &[u8],
    is_last_item: bool,
    out: &mut Vec<SourceEdit>,
) {
    // If this item is in an ordered list, the caller passes the
    // canonical marker text (e.g. `1. `, `2. `, `10. `). When it
    // differs from what's in source, emit a renumber edit. The
    // renumber may also change the marker's *byte width*, which we
    // need to account for when computing this item's
    // continuation-line indent target — the new target is
    // `ancestor_indent + new_marker_width` regardless of what the
    // old width was.
    if let Some(target) = target_marker
        && bytes[marker_range.clone()] != *target.as_bytes()
    {
        out.push(SourceEdit {
            range: marker_range.clone(),
            replacement: target.to_string(),
        });
    }

    // Normalize the gap between marker and content. The marker
    // already includes one trailing space, so any spaces beyond
    // `marker_range.end` and before the first non-space byte are
    // *extra* — typically pasted source or an accidental
    // double-space. Strip them down to zero (the canonical one
    // space lives inside the marker itself).
    //
    // We only strip when *content* follows on the same line. A
    // marker line that's all whitespace (just the marker plus
    // trailing spaces) is a transient mid-edit state — Enter or
    // Tab just landed the cursor there — and yanking trailing
    // spaces out from under the cursor would feel wrong.
    {
        let mut line_end = marker_range.end;
        while line_end < item.range.end && bytes[line_end] != b'\n' {
            line_end += 1;
        }
        let mut extra_end = marker_range.end;
        while extra_end < line_end && bytes[extra_end] == b' ' {
            extra_end += 1;
        }
        if extra_end > marker_range.end && extra_end < line_end {
            out.push(SourceEdit {
                range: marker_range.end..extra_end,
                replacement: String::new(),
            });
        }
    }

    // Continuation lines for this item should carry
    // `ancestor_indent + target_marker_width` leading spaces — the
    // sum of every enclosing list-item's marker (using the
    // *post-renumber* width for ordered items).
    let target_marker_width = target_marker
        .map(|s| s.len())
        .unwrap_or_else(|| marker_range.end - marker_range.start);
    let target_indent = ancestor_indent + target_marker_width;
    let indent_string = " ".repeat(target_indent);
    let item_start = item.range.start;
    // Trim the trailing `\n` run from the canonicalization region.
    // Pulldown folds 1 or more trailing newlines into each Item's
    // range — the count distinguishes tight vs loose. For
    // non-last items, all but one of those newlines is "loose
    // padding" that we collapse to keep lists tight. For the last
    // item, the trailing run is *boundary* with the surrounding
    // structure (next top-level block, EOF, the `\n\n` we just
    // inserted via empty-item-Enter exit, …) — leave it alone so
    // those upstream rules can do their job.
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

    // The item's first line — from `item_start` (where the marker
    // sits) to the first `\n` — is left as-is. Pulldown sometimes
    // includes leading indent ahead of the marker for nested items;
    // we trust that.
    let mut p = item_start;
    while p < item_end && bytes[p] != b'\n' {
        p += 1;
    }

    // Content lines beyond the first. For each non-empty content
    // line we want: previous `\n` → `  \n` (hard break) AND leading
    // spaces == marker_width. A blank line within the item collapses
    // — pulldown ranges already strip the trailing folded newlines
    // for tight lists, but loose-list items can hold internal
    // `\n\n+` runs. Collapse those to a single `\n`.

    // We deliberately do *not* collapse `\n\n+` runs inside an item.
    // Multi-paragraph items are now first-class: a `\n\n` run is a
    // paragraph break inside the item (or, in longer runs, a
    // paragraph break plus empty paragraphs — same pairs model as
    // top level). Inter-item tightening for non-last items is
    // handled above by the trailing-trim block.
    let _ = p;

    // Walk content lines. For each line beyond the first that has
    // *actual content* (something past leading spaces), enforce two
    // things: the preceding line ends with `  \n` (hard break — for
    // plain-text continuations only; nested constructs use a soft
    // boundary) and the leading whitespace equals `indent_string`.
    //
    // Empty / blank continuation lines are skipped — their existence
    // is either transient (the cursor parked on a fresh hard-break
    // continuation, no content yet) or one half of a `\n\n`
    // paragraph break we leave alone.
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

        // Skip blank lines — they're handled by the collapse pass
        // (or, if trailing, intentionally preserved as transient
        // post-Shift+Enter cursor real estate).
        let mut indent_end = line_start;
        while indent_end < line_end && bytes[indent_end] == b' ' {
            indent_end += 1;
        }
        if indent_end >= line_end {
            continue;
        }

        // Skip blockquote-scope continuation lines. Pulldown's Item
        // ranges sometimes swallow `> ` continuation lines from an
        // enclosing blockquote (notably the post-Enter transient
        // `> - foo\n> \n> ` shape). Those bytes are outer-container
        // markers, not item content — adding `marker_width` of indent
        // in front of them would corrupt the BQ scope.
        //
        // CommonMark allows up to 3 spaces of indent before a `>`
        // marker, so we accept the shape even when our heuristic
        // sees a small leading-space run.
        if indent_end < line_end && bytes[indent_end] == b'>' && (indent_end - line_start) <= 3 {
            continue;
        }

        // Skip lines that belong to a *nested* block child of this
        // item — a nested list, a nested blockquote, or a nested
        // code block. Those constructs are normalized by their own
        // recursive `normalize_item` (or are inert content like
        // code blocks), and the parent item shouldn't try to
        // overwrite their indent or insert hard breaks before
        // their lines. Without this guard, the parent's
        // `target_indent` (parent's marker_width) would dedent the
        // nested content back to the parent's column.
        if line_inside_nested_block_child(item, line_start) {
            continue;
        }

        // Count `\n`s in the run immediately preceding this line.
        // - run = 1 (one `\n`): a soft break or lazy continuation
        //   that we promote to a hard break.
        // - run ≥ 2: a paragraph break (or paragraph break + empty
        //   paragraphs), which leaves this line introducing a fresh
        //   paragraph inside the item. No hard-break treatment —
        //   the `\n\n` IS the structural break.
        let mut nl_count = 0;
        let mut effective_prev_nl = line_start;
        while effective_prev_nl > item_start && bytes[effective_prev_nl - 1] == b'\n' {
            effective_prev_nl -= 1;
            nl_count += 1;
        }
        let is_paragraph_break = nl_count >= 2;

        // A line that *opens* a nested construct (its first
        // non-space byte is another list marker, a `>`, a fence,
        // …) gets handled by that construct's own normalize pass.
        // We skip both hard-break promotion AND indent
        // normalization for it: the nested construct may live at
        // a different indent than this item's continuations
        // (e.g. its own `target_indent = ancestor + this_marker +
        // its_own_marker`).
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
                replacement: indent_string.clone(),
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
        let bytes = b"p1\n\np2";
        assert!(is_forbidden_position(bytes, 3));
        assert!(!is_forbidden_position(bytes, 0));
        assert!(!is_forbidden_position(bytes, 2));
        assert!(!is_forbidden_position(bytes, 4));
        assert!(!is_forbidden_position(bytes, 6));
    }

    #[test]
    fn between_two_pairs_is_allowed() {
        let bytes = b"p1\n\n\n\np2";
        assert!(is_forbidden_position(bytes, 3));
        assert!(!is_forbidden_position(bytes, 4));
        assert!(is_forbidden_position(bytes, 5));
    }

    #[test]
    fn six_newline_run_alternates_forbidden_and_allowed() {
        let bytes = b"p1\n\n\n\n\n\np2";
        assert!(is_forbidden_position(bytes, 3));
        assert!(!is_forbidden_position(bytes, 4));
        assert!(is_forbidden_position(bytes, 5));
        assert!(!is_forbidden_position(bytes, 6));
        assert!(is_forbidden_position(bytes, 7));
        assert!(!is_forbidden_position(bytes, 8));
    }

    #[test]
    fn fenced_code_blanks_are_not_forbidden() {
        let bytes = b"```\n\n\n```\n";
        for p in 0..bytes.len() {
            assert!(
                !is_forbidden_position(bytes, p),
                "byte {p} unexpectedly forbidden inside code block",
            );
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

    #[test]
    fn blockquote_continuation_prefix_top_level() {
        assert_eq!(blockquote_continuation_prefix("hello", 2), "");
    }

    #[test]
    fn blockquote_continuation_prefix_depth_1() {
        assert_eq!(blockquote_continuation_prefix("> hi", 4), "> ");
    }

    #[test]
    fn blockquote_continuation_prefix_depth_2() {
        assert_eq!(blockquote_continuation_prefix("> > deep", 8), "> > ");
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

    /// Regression for Point 4 in the architecture review: the byte
    /// scanner here and pulldown-cmark must agree on which bytes are
    /// inside a fenced code block.
    ///
    /// Direction tested: every byte pulldown attributes to a code
    /// block, the scanner also covers. The reverse can diverge by one
    /// trailing `\n` — the scanner's range extends through the
    /// closing fence line's terminating `\n` so the soft-break /
    /// forbidden-pair rules treat the byte right after the closer as
    /// not-content (no spurious paragraph break gets injected after a
    /// code block); pulldown's range stops one byte earlier. That gap
    /// is intentional and documented in `fenced_code_content_ranges`.
    #[test]
    fn fenced_ranges_agree_with_pulldown() {
        use crate::parser::parse;
        use crate::syntax::NodeKind;

        let cases = [
            "```\nx\n```\n",
            "```rust\nfn main() {}\n```\n",
            "~~~js\nlet x = 1;\n~~~\n",
            "```\nunterminated\n",
            "para\n\n```\ncode\n```\n\nafter\n",
            "```\n```\n",
        ];
        for src in cases {
            let bytes = src.as_bytes();
            let scanner = fenced_code_content_ranges(bytes);
            let tree = parse(src);
            let mut pulldown_ranges: Vec<Range<usize>> = Vec::new();
            collect_code_ranges(&tree, &mut pulldown_ranges);
            for p in 0..bytes.len() {
                if is_in_ranges(p, &pulldown_ranges) {
                    assert!(
                        is_in_ranges(p, &scanner),
                        "byte {p} in {src:?}: pulldown says code, scanner disagrees",
                    );
                }
            }
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
