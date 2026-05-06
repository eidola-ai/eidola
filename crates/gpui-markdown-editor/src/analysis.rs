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
/// Soft breaks are mid-content `\n`s that aren't part of a complete
/// structural pair. The depth-D pair `\n[prefix]\n[prefix]` is the
/// generalization of `\n\n`: a `\n` is exempt if it's one of the two
/// `\n`s in such a pair (so the `\n` between two adjacent BQ-content
/// paragraphs survives, but a stray `\n` across two BQ lines —
/// CommonMark's "lazy continuation" — is still promoted).
pub fn is_soft_break(bytes: &[u8], p: usize) -> bool {
    if bytes[p] != b'\n' {
        return false;
    }
    // Edge of document — single leading or trailing `\n` is harmless
    // whitespace in CommonMark, and changing it would surprise users
    // who pasted content that ends in `\n`.
    if p == 0 || p + 1 >= bytes.len() {
        return false;
    }
    // Already part of a paragraph break run.
    if bytes[p - 1] == b'\n' || bytes[p + 1] == b'\n' {
        return false;
    }
    // Hard breaks (`\\\n` / `  \n`).
    if bytes[p - 1] == b'\\' {
        return false;
    }
    if p >= 2 && bytes[p - 1] == b' ' && bytes[p - 2] == b' ' {
        return false;
    }
    // Already part of a structural depth-D pair. The pair-interior
    // detector also classifies the byte right after either `\n` of
    // the pair as interior; piggyback on it to recognize "this `\n`
    // belongs to a pair" by probing both adjacent positions.
    if is_paragraph_break_interior(bytes, p) || is_paragraph_break_interior(bytes, p + 1) {
        return false;
    }
    true
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

/// The active container prefix that introduces the line at `cursor` —
/// the literal string a continuation line would need to repeat to stay
/// inside every container that wraps `cursor`. For top-level content
/// this is `""`; for a depth-D blockquote it's `"> "` repeated D
/// times. Future list-item containers will extend the produced string
/// with their indent prefix.
///
/// Used by Enter / Shift+Enter to emit a newline that keeps the new
/// paragraph or hard-break continuation in the same container scope,
/// and by `enforce_invariants` to detect / repair lazy continuations.
pub fn active_container_prefix(markdown: &str, cursor: usize) -> String {
    let depth = blockquote_depth_at(markdown, cursor);
    "> ".repeat(depth)
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
    fn active_container_prefix_top_level() {
        assert_eq!(active_container_prefix("hello", 2), "");
    }

    #[test]
    fn active_container_prefix_blockquote_depth_1() {
        assert_eq!(active_container_prefix("> hi", 4), "> ");
    }

    #[test]
    fn active_container_prefix_blockquote_depth_2() {
        assert_eq!(active_container_prefix("> > deep", 8), "> > ");
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
