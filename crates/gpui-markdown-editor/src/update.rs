//! Pure state transitions: `update(state, event) -> state`.
//!
//! # Invariants the buffer always satisfies
//!
//! Every state we produce passes through `enforce_invariants` at the end of
//! `update`, so callers don't have to think about these — they just hold:
//!
//! 1. **No soft breaks.** A `\n` in the middle of content is *always* part
//!    of a paragraph break (`\n\n+` run), a hard line break (`  \n` /
//!    `\\\n`), or sits at the document edge (leading / trailing). A lone
//!    mid-content `\n` would render as a space in CommonMark and as a line
//!    break in the editor — that ambiguity breaks pixel fidelity with the
//!    chat transcript renderer, so it's not allowed to exist. If any code
//!    path produces one (typed text, paste, deletion that leaves one
//!    behind), `enforce_invariants` rewrites it into `\n\n`.
//!
//! 2. **Selection offsets sit on UTF-8 char boundaries.** `set_selection`
//!    and `move_` both snap; promotion only inserts ASCII `\n` so it
//!    can't break this.
//!
//! 3. **No cursor inside a structural `\n\n` pair.** The interior byte of
//!    a paragraph-break / empty-paragraph pair is unreachable visually
//!    (the pair renders as one row, not two) and typing there would split
//!    the pair into a stray odd-length run. After every transition,
//!    `avoid_forbidden_positions` snaps any cursor or selection endpoint
//!    that lands on the interior of a structural pair *away* from the
//!    pre-event position: forward if the cursor moved forward (or didn't
//!    move), backward if it moved back. So Right at the end of `p1` in
//!    `p1\n\np2` skips from byte 2 straight to 4 (start of p2), and Left
//!    from byte 4 jumps back to 2. Hard-break `\n`s are exempt — they're
//!    in-paragraph content, not part of a structural pair.
//!
//! # Pairs model
//!
//! The buffer treats `\n\n` as the atomic structural unit. Each Enter
//! inserts exactly `\n\n`; each empty paragraph is rendered from a pair
//! of `\n`s; smart-delete removes pairs. With this discipline the source
//! always carries an even count of `\n`s in any structural run, and the
//! "typing on a trailing empty loses a row" asymmetry disappears for
//! free: typing X at the end of `p1\n\n\n\n` (2 trailing empties)
//! produces `p1\n\n\n\nX`, which the renderer reads as paragraph break +
//! 1 empty + X — same row count as before typing.
//!
//! Hard breaks (`  \n`, `\\\n`) are exempt from the pairs discipline —
//! they're a deliberate single `\n` in mid-paragraph.
//!
//! # Implication for editing actions
//!
//! Most actions don't have to know about either invariant — they just
//! produce whatever markdown they think makes sense, and the post-pass
//! cleans up soft breaks. The only special case is `delete_backward` /
//! `delete_forward`: a generic "delete one byte" at a paragraph break
//! would leave a soft break that the post-pass would immediately
//! re-promote, so the keypress would feel like a no-op. Both handlers
//! detect "I'm in a `\n` run" and delete a *pair* (collapsing the break
//! to merge paragraphs, or removing one empty paragraph from a longer
//! run).

use unicode_segmentation::UnicodeSegmentation;

use crate::analysis::{
    self, count_line_markers, fenced_code_content_ranges, is_forbidden_position, is_in_ranges,
    is_soft_break, line_depth_ending_at, nearest_allowed_position, next_allowed_position,
    pair_at_end, pair_at_start, prev_allowed_position,
};
use crate::event::EditorEvent;
use crate::state::{EditorState, Selection};

pub fn update(state: EditorState, event: EditorEvent) -> EditorState {
    let prev_anchor = match state.selection {
        Selection::Cursor(p) => p,
        Selection::Range { anchor, .. } => anchor,
    };
    let prev_head = state.selection.head();

    let next = match event {
        EditorEvent::InsertText(text) => insert_text(state, &text),
        EditorEvent::InsertNewline => insert_newline(state),
        EditorEvent::InsertLineBreak => insert_line_break(state),

        EditorEvent::DeleteBackward => delete_backward(state),
        EditorEvent::DeleteForward => delete_forward(state),

        EditorEvent::SetSelection(sel) => set_selection(state, sel),

        EditorEvent::MoveLeft => move_(state, Move::Left, false),
        EditorEvent::MoveRight => move_(state, Move::Right, false),
        EditorEvent::MoveUp => move_(state, Move::Up, false),
        EditorEvent::MoveDown => move_(state, Move::Down, false),
        EditorEvent::MoveLineStart => move_(state, Move::LineStart, false),
        EditorEvent::MoveLineEnd => move_(state, Move::LineEnd, false),
        EditorEvent::MoveDocumentStart => move_(state, Move::DocStart, false),
        EditorEvent::MoveDocumentEnd => move_(state, Move::DocEnd, false),

        EditorEvent::ExtendLeft => move_(state, Move::Left, true),
        EditorEvent::ExtendRight => move_(state, Move::Right, true),
        EditorEvent::ExtendUp => move_(state, Move::Up, true),
        EditorEvent::ExtendDown => move_(state, Move::Down, true),
        EditorEvent::ExtendLineStart => move_(state, Move::LineStart, true),
        EditorEvent::ExtendLineEnd => move_(state, Move::LineEnd, true),
        EditorEvent::ExtendDocumentStart => move_(state, Move::DocStart, true),
        EditorEvent::ExtendDocumentEnd => move_(state, Move::DocEnd, true),
    };
    let next = enforce_invariants(next);
    avoid_forbidden_positions(next, prev_anchor, prev_head)
}

/// Promote any lone, mid-content `\n` into `\n\n` so the buffer never
/// contains a soft break, then normalize every blockquote `>` marker to
/// `> `. Idempotent and cheap on already-clean states.
pub fn enforce_invariants(state: EditorState) -> EditorState {
    let state = promote_soft_breaks(state);
    normalize_blockquote_prefixes(state)
}

fn promote_soft_breaks(state: EditorState) -> EditorState {
    let bytes = state.markdown.as_bytes();
    // Two classes of bytes are exempt from the soft-break promotion
    // rule:
    //
    //   * Fenced code-block content. A single mid-content `\n` inside
    //     ```/~~~ fences is a literal line separator, not the
    //     ambiguous CommonMark soft break.
    //   * List ranges. Inside a list pulldown handles line structure
    //     (item separators, marker continuation, indented paragraphs,
    //     lazy continuations). Promoting `\n` to `\n\n` between two
    //     items would split the list.
    let code_ranges = fenced_code_content_ranges(bytes);
    let list_ranges = analysis::list_content_ranges(&state.markdown);

    // Each entry: (insertion_position, inserted_string). Computed in a
    // single forward scan over the *original* buffer so offsets line
    // up with the input — the apply step later remaps cursor offsets
    // exactly once.
    let mut inserts: Vec<(usize, String)> = Vec::new();
    for p in 0..bytes.len() {
        if is_in_ranges(p, &code_ranges) || is_in_ranges(p, &list_ranges) {
            continue;
        }
        if !is_soft_break(bytes, p) {
            continue;
        }

        // Promotion shape: turn the stray `\n` at p into a complete
        // depth-D pair `\n[prefix]\n[prefix]`. D is the depth of the
        // line ending at `p`; whatever markers already exist on the
        // *next* line are kept in place so we never duplicate them.
        let depth = line_depth_ending_at(bytes, p);
        let (existing_markers, _) = count_line_markers(bytes, p + 1);
        let prefix = "> ".repeat(depth);

        let (insert_at, inserted) = if existing_markers >= depth {
            // The next line already opens with at least the line-
            // before's depth. Splice in `[prefix]\n` right *after*
            // the existing `\n`; the existing markers naturally
            // become the second `[prefix]` of the pair.
            (p + 1, format!("{prefix}\n"))
        } else {
            // Lazy continuation (next line lacks at least one of the
            // markers we'd expect). Insert the full `[prefix]\n
            // [prefix]` so the continuation line gains the missing
            // markers and the pair structure is complete.
            (p + 1, format!("{prefix}\n{prefix}"))
        };

        if inserted.is_empty() {
            continue; // depth 0 + existing_markers >= 0 → nothing to do
        }
        inserts.push((insert_at, inserted));
    }

    if inserts.is_empty() {
        return state;
    }

    // Apply inserts in order. Each `inserts[i].0` is in *original*
    // coordinates, so we rebuild the buffer by interleaving original
    // slices with inserted strings.
    let total_added: usize = inserts.iter().map(|(_, s)| s.len()).sum();
    let mut new_md = String::with_capacity(state.markdown.len() + total_added);
    let mut last = 0;
    for (pos, ins) in &inserts {
        new_md.push_str(&state.markdown[last..*pos]);
        new_md.push_str(ins);
        last = *pos;
    }
    new_md.push_str(&state.markdown[last..]);

    let map = |off: usize| -> usize {
        let mut shift = 0;
        for (pos, ins) in &inserts {
            if *pos < off {
                shift += ins.len();
            } else if *pos == off {
                // Convention: an offset exactly at an insertion point
                // shifts forward, so the cursor stays "with" the
                // content it sat against. (For Enter-from-soft-break
                // the cursor is typically left-of the broken `\n`,
                // so this branch rarely fires; it's the safe default.)
                shift += ins.len();
            }
        }
        off + shift
    };
    let new_sel = match state.selection {
        Selection::Cursor(p) => Selection::Cursor(map(p)),
        Selection::Range { anchor, head } => Selection::Range {
            anchor: map(anchor),
            head: map(head),
        },
    };

    EditorState {
        markdown: new_md,
        selection: new_sel,
    }
}

/// Insert a space after every blockquote `>` marker that isn't already
/// followed by one — *unless* the cursor (or selection anchor / head)
/// is exactly the byte right after the `>`. Mid-typing the user might
/// have just pressed `>` and intends to type a space themselves; we
/// don't second-guess them. Once the cursor moves away, the next
/// `update` call's post-pass normalizes the marker so the parsed
/// shape stays predictable for the renderer.
///
/// Skips `>` bytes that fall inside fenced code-block content — those
/// are literal `>` characters, not blockquote markers.
fn normalize_blockquote_prefixes(state: EditorState) -> EditorState {
    let bytes = state.markdown.as_bytes();
    let code_ranges = fenced_code_content_ranges(bytes);
    let (cursor_head, cursor_anchor) = match state.selection {
        Selection::Cursor(p) => (p, p),
        Selection::Range { anchor, head } => (head, anchor),
    };

    let mut insert_at: Vec<usize> = Vec::new();
    let mut p = 0;
    while p < bytes.len() {
        let line_start = p;
        let mut line_end = p;
        while line_end < bytes.len() && bytes[line_end] != b'\n' {
            line_end += 1;
        }
        let mut q = line_start;
        loop {
            let mut indent = 0;
            while q < line_end && bytes[q] == b' ' && indent < 3 {
                q += 1;
                indent += 1;
            }
            if q < line_end && bytes[q] == b'>' && !is_in_ranges(q, &code_ranges) {
                let after_gt = q + 1;
                let needs_space =
                    after_gt < line_end && bytes[after_gt] != b' ' && bytes[after_gt] != b'\n';
                if needs_space && after_gt != cursor_head && after_gt != cursor_anchor {
                    insert_at.push(after_gt);
                }
                q = after_gt;
                if q < line_end && bytes[q] == b' ' {
                    q += 1;
                }
                continue;
            }
            break;
        }
        p = line_end + 1;
    }

    if insert_at.is_empty() {
        return state;
    }

    let mut new_md = String::with_capacity(state.markdown.len() + insert_at.len());
    let mut last = 0;
    for &pos in &insert_at {
        new_md.push_str(&state.markdown[last..pos]);
        new_md.push(' ');
        last = pos;
    }
    new_md.push_str(&state.markdown[last..]);

    let map = |off: usize| -> usize { off + insert_at.iter().filter(|&&p| p < off).count() };
    let new_sel = match state.selection {
        Selection::Cursor(p) => Selection::Cursor(map(p)),
        Selection::Range { anchor, head } => Selection::Range {
            anchor: map(anchor),
            head: map(head),
        },
    };

    EditorState {
        markdown: new_md,
        selection: new_sel,
    }
}

/// Snap any cursor or selection endpoint that landed at a forbidden
/// position (interior of a structural `\n\n` pair) away from where it
/// came from. Forward if the cursor moved forward (or didn't move),
/// backward if it moved back. See module docs for the rationale.
fn avoid_forbidden_positions(
    state: EditorState,
    prev_anchor: usize,
    prev_head: usize,
) -> EditorState {
    let bytes = state.markdown.as_bytes();
    let new_sel = match state.selection {
        Selection::Cursor(p) => Selection::Cursor(snap_off_forbidden(bytes, p, prev_head)),
        Selection::Range { anchor, head } => {
            let a = snap_off_forbidden(bytes, anchor, prev_anchor);
            let h = snap_off_forbidden(bytes, head, prev_head);
            if a == h {
                Selection::Cursor(h)
            } else {
                Selection::Range { anchor: a, head: h }
            }
        }
    };
    EditorState {
        selection: new_sel,
        ..state
    }
}

fn snap_off_forbidden(bytes: &[u8], pos: usize, prev: usize) -> usize {
    if !is_forbidden_position(bytes, pos) {
        return pos;
    }
    if pos < prev {
        prev_allowed_position(bytes, pos)
    } else {
        next_allowed_position(bytes, pos)
    }
}

/// Re-exported for behavior tests that look up depth via `update::blockquote_depth_at`.
/// New callers should import directly from [`crate::analysis`].
pub use crate::analysis::blockquote_depth_at;

// `EditorEvent::InsertNewline` / `InsertLineBreak` route through
// `analysis::enter_insertion` / `line_break_insertion`, which know about
// every container kind and emit the right source string for the cursor's
// position. The shell stays a router so keyboard, IME, paste-derived, and
// programmatic dispatch all share one rule.

fn insert_newline(state: EditorState) -> EditorState {
    let insertion = analysis::enter_insertion(&state.markdown, state.selection.head());
    insert_text(state, &insertion)
}

fn insert_line_break(state: EditorState) -> EditorState {
    let insertion = analysis::line_break_insertion(&state.markdown, state.selection.head());
    insert_text(state, &insertion)
}

fn clamp(pos: usize, len: usize) -> usize {
    pos.min(len)
}

fn delete_selection(state: &EditorState) -> (String, usize) {
    let len = state.markdown.len();
    match state.selection {
        Selection::Cursor(p) => (state.markdown.clone(), clamp(p, len)),
        Selection::Range { anchor, head } => {
            let start = clamp(anchor.min(head), len);
            let end = clamp(anchor.max(head), len);
            let mut out = String::with_capacity(len - (end - start));
            out.push_str(&state.markdown[..start]);
            out.push_str(&state.markdown[end..]);
            (out, start)
        }
    }
}

fn insert_text(state: EditorState, text: &str) -> EditorState {
    let (mut buf, cursor) = delete_selection(&state);
    buf.insert_str(cursor, text);
    EditorState {
        markdown: buf,
        selection: Selection::Cursor(cursor + text.len()),
    }
}

fn delete_backward(state: EditorState) -> EditorState {
    if !state.selection.is_collapsed() {
        let (buf, cursor) = delete_selection(&state);
        return EditorState {
            markdown: buf,
            selection: Selection::Cursor(cursor),
        };
    }
    let cursor = state.selection.head();
    if cursor == 0 {
        return state;
    }

    // At a structural boundary, Backspace has two specialized rules.
    // Both first snap forward over any pair interior — that's where a
    // direct cursor placement (e.g. a click on the visually-collapsed
    // paragraph_gap, or a programmatic SetSelection) might land — then
    // inspect the structure ending at the snapped position. Inside
    // fenced code-block content, `\n`s are literal line separators —
    // fall through to the regular grapheme-delete path instead.
    let bytes = state.markdown.as_bytes();
    if !analysis::is_in_fenced_code(bytes, cursor) {
        let snapped = next_allowed_position(bytes, cursor);
        // Blockquote outdent: at the start of a non-first paragraph
        // inside a BQ, Backspace pops one level of nesting from
        // *both halves* of the preceding `\n[prefix]\n[prefix]` pair
        // instead of merging the paragraph with the previous one.
        // Outdenting both halves keeps the pair invariant (depth-D
        // → depth-(D-1) is still a clean balanced pair, and a
        // depth-1 pair becomes a depth-0 `\n\n` break) so the result
        // is never an asymmetric pair the rest of the rules would
        // have to special-case. When the paragraph reaches depth 0
        // the outdent detector returns `None` and the next Backspace
        // falls through to the depth-0 atomic pair delete below,
        // which merges into the previous paragraph as before.
        if let Some((above, below)) = analysis::bq_paragraph_outdent(bytes, snapped) {
            // Apply right-to-left so the earlier range's offsets
            // don't need to be remapped.
            let after_below = splice(&state.markdown, cursor, below.start, below.end);
            return splice(
                &after_below.markdown,
                after_below.selection.head(),
                above.start,
                above.end,
            );
        }
        // Atomic pair delete at depth 0: when the cursor sits at a
        // top-level `\n\n` paragraph break, Backspace removes the
        // whole pair in one step, merging the two paragraphs.
        if let Some(pair_start) = pair_at_end(bytes, snapped) {
            return splice(&state.markdown, cursor, pair_start, snapped);
        }
    }

    let prev = prev_grapheme_offset(&state.markdown, cursor);
    splice(&state.markdown, cursor, prev, cursor)
}

fn delete_forward(state: EditorState) -> EditorState {
    if !state.selection.is_collapsed() {
        let (buf, cursor) = delete_selection(&state);
        return EditorState {
            markdown: buf,
            selection: Selection::Cursor(cursor),
        };
    }
    let cursor = state.selection.head();
    if cursor >= state.markdown.len() {
        return state;
    }

    let bytes = state.markdown.as_bytes();
    if !analysis::is_in_fenced_code(bytes, cursor) {
        let snapped = prev_allowed_position(bytes, cursor);
        if let Some(pair_end) = pair_at_start(bytes, snapped) {
            return splice(&state.markdown, cursor, snapped, pair_end);
        }
    }

    let next = next_grapheme_offset(&state.markdown, cursor);
    splice(&state.markdown, cursor, cursor, next)
}

/// Splice out `[del_start, del_end)` from `markdown` and re-anchor the
/// cursor relative to the deletion. Caller-supplied `cursor` is the
/// current head; the returned state collapses any range selection to a
/// cursor (which is fine for our delete callers — they all operate on
/// collapsed selections after the range-delete-and-return early exit).
fn splice(markdown: &str, cursor: usize, del_start: usize, del_end: usize) -> EditorState {
    let mut buf = String::with_capacity(markdown.len() - (del_end - del_start));
    buf.push_str(&markdown[..del_start]);
    buf.push_str(&markdown[del_end..]);
    let new_cursor = if cursor <= del_start {
        cursor
    } else if cursor < del_end {
        del_start
    } else {
        cursor - (del_end - del_start)
    };
    EditorState {
        markdown: buf,
        selection: Selection::Cursor(new_cursor),
    }
}

fn set_selection(state: EditorState, sel: Selection) -> EditorState {
    let len = state.markdown.len();
    let normalized = match sel {
        Selection::Cursor(p) => Selection::Cursor(clamp(p, len)),
        Selection::Range { anchor, head } => {
            let a = clamp(anchor, len);
            let h = clamp(head, len);
            if a == h {
                Selection::Cursor(h)
            } else {
                Selection::Range { anchor: a, head: h }
            }
        }
    };
    let on_boundaries = snap_selection_to_char_boundaries(&state.markdown, normalized);
    // Mouse clicks and host SetSelection calls feed offsets that may
    // land on a forbidden pair interior — most often a synthetic empty
    // paragraph between two real blocks, whose `display_to_source`
    // anchors at the block's source-range start (which is the pair
    // interior). Using the prev-comparison rule here would flip the
    // cursor between two adjacent allowed positions every time a
    // `MouseMoveEvent` re-feeds the same offset, because each call
    // sees the previously-snapped position as `prev`. Nearest-allowed
    // is idempotent: same input → same output.
    let bytes = state.markdown.as_bytes();
    let final_sel = match on_boundaries {
        Selection::Cursor(p) => Selection::Cursor(nearest_allowed_position(bytes, p)),
        Selection::Range { anchor, head } => {
            let a = nearest_allowed_position(bytes, anchor);
            let h = nearest_allowed_position(bytes, head);
            if a == h {
                Selection::Cursor(h)
            } else {
                Selection::Range { anchor: a, head: h }
            }
        }
    };
    EditorState {
        selection: final_sel,
        ..state
    }
}

#[derive(Debug, Clone, Copy)]
enum Move {
    Left,
    Right,
    Up,
    Down,
    LineStart,
    LineEnd,
    DocStart,
    DocEnd,
}

fn move_(state: EditorState, direction: Move, extending: bool) -> EditorState {
    let head = state.selection.head();
    let new_head = match direction {
        Move::Left => prev_grapheme_offset(&state.markdown, head),
        Move::Right => next_grapheme_offset(&state.markdown, head),
        Move::Up => move_vertical(&state.markdown, head, -1),
        Move::Down => move_vertical(&state.markdown, head, 1),
        Move::LineStart => line_start_offset(&state.markdown, head),
        Move::LineEnd => line_end_offset(&state.markdown, head),
        Move::DocStart => 0,
        Move::DocEnd => state.markdown.len(),
    };

    let new_sel = if extending {
        let anchor = match state.selection {
            Selection::Cursor(p) => p,
            Selection::Range { anchor, .. } => anchor,
        };
        if anchor == new_head {
            Selection::Cursor(new_head)
        } else {
            Selection::Range {
                anchor,
                head: new_head,
            }
        }
    } else if !state.selection.is_collapsed() {
        // Collapse to the appropriate edge (the natural direction the user
        // is moving toward) instead of jumping past the selection.
        match direction {
            Move::Left | Move::Up | Move::LineStart | Move::DocStart => {
                Selection::Cursor(state.selection.lower_bound())
            }
            Move::Right | Move::Down | Move::LineEnd | Move::DocEnd => {
                Selection::Cursor(state.selection.upper_bound())
            }
        }
    } else {
        Selection::Cursor(new_head)
    };

    EditorState {
        selection: snap_selection_to_char_boundaries(&state.markdown, new_sel),
        ..state
    }
}

fn snap_to_char_boundary(text: &str, pos: usize) -> usize {
    let len = text.len();
    if pos >= len {
        return len;
    }
    let mut p = pos;
    while p > 0 && !text.is_char_boundary(p) {
        p -= 1;
    }
    p
}

fn snap_selection_to_char_boundaries(text: &str, sel: Selection) -> Selection {
    match sel {
        Selection::Cursor(p) => Selection::Cursor(snap_to_char_boundary(text, p)),
        Selection::Range { anchor, head } => Selection::Range {
            anchor: snap_to_char_boundary(text, anchor),
            head: snap_to_char_boundary(text, head),
        },
    }
}

fn prev_grapheme_offset(text: &str, pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    let mut last = 0usize;
    for (idx, _) in text[..pos.min(text.len())].grapheme_indices(true) {
        last = idx;
    }
    last
}

fn next_grapheme_offset(text: &str, pos: usize) -> usize {
    let len = text.len();
    if pos >= len {
        return len;
    }
    let mut iter = text[pos..].grapheme_indices(true);
    iter.next();
    if let Some((idx, _)) = iter.next() {
        pos + idx
    } else {
        len
    }
}

fn line_start_offset(text: &str, pos: usize) -> usize {
    let bytes = text.as_bytes();
    let pos = pos.min(text.len());
    let mut start = pos;
    while start > 0 && bytes[start - 1] != b'\n' {
        start -= 1;
    }
    start
}

fn line_end_offset(text: &str, pos: usize) -> usize {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut end = pos.min(len);
    while end < len && bytes[end] != b'\n' {
        end += 1;
    }
    end
}

/// Vertical navigation, aware of phantom `\n`-bounded segments.
///
/// `line_start_offset` slices the buffer at every `\n`, so a structural
/// `\n\n` pair shows up as two adjacent zero-length "lines" with the
/// pair interior as the start of the second. That position is forbidden
/// (no visible row to land on), and `text.split('\n')` doesn't know it.
/// Skipping any segment whose start is forbidden makes Up/Down move
/// from one *visible* line to the next visible line — across paragraph
/// breaks (which contribute zero visible rows) and synthetic empty
/// paragraphs (one visible row each).
fn move_vertical(text: &str, pos: usize, direction: i32) -> usize {
    let bytes = text.as_bytes();
    let line_start = line_start_offset(text, pos);
    let column = pos - line_start;
    if direction < 0 {
        let mut probe = line_start;
        loop {
            if probe == 0 {
                return 0;
            }
            let prev_line_end = probe - 1;
            let prev_line_start = line_start_offset(text, prev_line_end);
            if is_forbidden_position(bytes, prev_line_start) {
                probe = prev_line_start;
                continue;
            }
            let prev_line_len = prev_line_end - prev_line_start;
            let target = prev_line_start + column.min(prev_line_len);
            return snap_to_char_boundary(text, target);
        }
    } else {
        let mut probe = line_end_offset(text, pos);
        loop {
            if probe >= text.len() {
                return text.len();
            }
            let next_line_start = probe + 1;
            if is_forbidden_position(bytes, next_line_start) {
                probe = line_end_offset(text, next_line_start);
                continue;
            }
            let next_line_end = line_end_offset(text, next_line_start);
            let next_line_len = next_line_end - next_line_start;
            let target = next_line_start + column.min(next_line_len);
            return snap_to_char_boundary(text, target);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(s: &str, cursor: usize) -> EditorState {
        EditorState {
            markdown: s.into(),
            selection: Selection::Cursor(cursor),
        }
    }

    #[test]
    fn insert_text_at_cursor() {
        let s = update(st("ab", 1), EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "aXb");
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn insert_newline_promotes_to_paragraph_break() {
        // Enter in the middle of a paragraph creates a paragraph break, not a
        // soft break — the post-pass enforces the invariant.
        let s = update(st("abc", 1), EditorEvent::InsertNewline);
        assert_eq!(s.markdown, "a\n\nbc");
        // Cursor was at 2 in "a\nbc"; the inserted second `\n` shifts it to 3
        // so it stays "right after the break, before 'b'".
        assert_eq!(s.selection, Selection::Cursor(3));
    }

    #[test]
    fn insert_text_replaces_selection() {
        let initial = EditorState {
            markdown: "abcdef".into(),
            selection: Selection::range(1, 4),
        };
        let s = update(initial, EditorEvent::InsertText("XX".into()));
        assert_eq!(s.markdown, "aXXef");
        assert_eq!(s.selection, Selection::Cursor(3));
    }

    #[test]
    fn delete_backward_removes_one_grapheme() {
        let s = update(st("abc", 2), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "ac");
        assert_eq!(s.selection, Selection::Cursor(1));
    }

    #[test]
    fn delete_backward_handles_multibyte() {
        let s = update(st("héllo", 3), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "hllo");
        assert_eq!(s.selection, Selection::Cursor(1));
    }

    #[test]
    fn delete_forward_removes_one_grapheme() {
        let s = update(st("abc", 1), EditorEvent::DeleteForward);
        assert_eq!(s.markdown, "ac");
        assert_eq!(s.selection, Selection::Cursor(1));
    }

    #[test]
    fn delete_at_start_is_a_noop() {
        let s = update(st("abc", 0), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "abc");
        assert_eq!(s.selection, Selection::Cursor(0));
    }

    #[test]
    fn move_left_steps_one_grapheme() {
        let s = update(st("ab", 2), EditorEvent::MoveLeft);
        assert_eq!(s.selection, Selection::Cursor(1));
    }

    #[test]
    fn move_right_at_end_clamps() {
        let s = update(st("ab", 2), EditorEvent::MoveRight);
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn move_left_with_selection_collapses_to_lower_bound() {
        let initial = EditorState {
            markdown: "abcdef".into(),
            selection: Selection::range(1, 4),
        };
        let s = update(initial, EditorEvent::MoveLeft);
        assert_eq!(s.selection, Selection::Cursor(1));
    }

    #[test]
    fn extend_right_grows_selection() {
        let s = update(st("abcd", 1), EditorEvent::ExtendRight);
        assert_eq!(s.selection, Selection::range(1, 2));
    }

    #[test]
    fn line_start_and_end() {
        // Use already-normalized markdown so move geometry isn't perturbed
        // by the post-pass promoting a soft break.
        let text = "abc\n\ndef";
        let s = update(st(text, 6), EditorEvent::MoveLineStart);
        assert_eq!(s.selection, Selection::Cursor(5));
        let s = update(st(text, 6), EditorEvent::MoveLineEnd);
        assert_eq!(s.selection, Selection::Cursor(8));
    }

    #[test]
    fn move_up_preserves_column_when_possible() {
        // Two soft-wrapped lines coupled by a hard break. MoveUp from
        // column 2 of "ghij" lands on column 2 of "abcdef" → 'c' at offset
        // 2. (Paragraph-break-separated lines are tested separately —
        // crossing the empty inter-paragraph line is its own concern.)
        let text = "abcdef  \nghij";
        let s = update(st(text, 11), EditorEvent::MoveUp);
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn document_start_and_end() {
        let text = "abc\n\ndef";
        let s = update(st(text, 6), EditorEvent::MoveDocumentStart);
        assert_eq!(s.selection, Selection::Cursor(0));
        let s = update(st(text, 0), EditorEvent::MoveDocumentEnd);
        assert_eq!(s.selection, Selection::Cursor(8));
    }

    #[test]
    fn set_selection_clamps_oob() {
        let s = update(
            st("abc", 0),
            EditorEvent::SetSelection(Selection::Cursor(99)),
        );
        assert_eq!(s.selection, Selection::Cursor(3));
    }
}

// ---------------------------------------------------------------------------
// Invariant tests — the no-soft-breaks rule + the smart-delete UX layered on
// top of it. If you're tempted to add a special-case check somewhere in the
// editor, write it as a test here first; the post-pass should be doing the
// work.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod invariant_tests {
    use super::*;

    fn st(s: &str, cursor: usize) -> EditorState {
        EditorState {
            markdown: s.into(),
            selection: Selection::Cursor(cursor),
        }
    }

    fn assert_no_soft_breaks(md: &str) {
        let bytes = md.as_bytes();
        for p in 0..bytes.len() {
            assert!(
                !is_soft_break(bytes, p),
                "soft break at byte {} in {:?}",
                p,
                md
            );
        }
    }

    // ---- Direct enforce_invariants tests --------------------------------------

    #[test]
    fn lone_newline_between_words_is_promoted() {
        let s = enforce_invariants(st("ab\ncd", 0));
        assert_eq!(s.markdown, "ab\n\ncd");
        assert_no_soft_breaks(&s.markdown);
    }

    #[test]
    fn paragraph_break_already_in_source_is_unchanged() {
        let original = st("ab\n\ncd", 4);
        let s = enforce_invariants(original.clone());
        assert_eq!(s, original);
    }

    #[test]
    fn long_newline_run_is_unchanged() {
        // Paragraph break + 2 empty paragraphs.
        let original = st("ab\n\n\n\ncd", 0);
        let s = enforce_invariants(original.clone());
        assert_eq!(s, original);
    }

    #[test]
    fn leading_newline_left_alone() {
        // CommonMark trims leading whitespace; a single leading `\n` is
        // benign and shouldn't surprise the user with a structural change.
        let s = enforce_invariants(st("\nab", 0));
        assert_eq!(s.markdown, "\nab");
    }

    #[test]
    fn trailing_newline_left_alone() {
        let s = enforce_invariants(st("ab\n", 3));
        assert_eq!(s.markdown, "ab\n");
    }

    #[test]
    fn hard_break_with_trailing_spaces_is_left_alone() {
        let s = enforce_invariants(st("ab  \ncd", 0));
        assert_eq!(s.markdown, "ab  \ncd");
    }

    #[test]
    fn hard_break_with_backslash_is_left_alone() {
        let s = enforce_invariants(st("ab\\\ncd", 0));
        assert_eq!(s.markdown, "ab\\\ncd");
    }

    #[test]
    fn multiple_lone_newlines_all_promoted_in_one_pass() {
        let s = enforce_invariants(st("a\nb\nc\nd", 0));
        assert_eq!(s.markdown, "a\n\nb\n\nc\n\nd");
        assert_no_soft_breaks(&s.markdown);
    }

    #[test]
    fn cursor_at_promotion_site_shifts_with_following_content() {
        // Cursor was right before 'c' in "ab\ncd"; after promotion it
        // should still be right before 'c' (not stranded mid-break).
        let s = enforce_invariants(st("ab\ncd", 3));
        assert_eq!(s.markdown, "ab\n\ncd");
        assert_eq!(s.selection, Selection::Cursor(4));
    }

    #[test]
    fn cursor_before_promotion_site_unchanged() {
        // Cursor before the soft break shouldn't move.
        let s = enforce_invariants(st("ab\ncd", 1));
        assert_eq!(s.selection, Selection::Cursor(1));
    }

    #[test]
    fn range_selection_endpoints_both_remapped() {
        let s = enforce_invariants(EditorState {
            markdown: "ab\ncd".into(),
            selection: Selection::range(0, 5),
        });
        assert_eq!(s.markdown, "ab\n\ncd");
        assert_eq!(s.selection, Selection::range(0, 6));
    }

    // ---- Insert / paste flows --------------------------------------------------

    #[test]
    fn second_enter_at_paragraph_boundary_adds_another_pair() {
        // Pairs model: each Enter inserts `\n\n`. Two Enters mid-content
        // gives 4 `\n`s between content = paragraph break + 1 empty.
        let s1 = update(st("abcd", 2), EditorEvent::InsertNewline);
        assert_eq!(s1.markdown, "ab\n\ncd");
        assert_eq!(s1.selection, Selection::Cursor(4));

        let s2 = update(s1, EditorEvent::InsertNewline);
        assert_eq!(s2.markdown, "ab\n\n\n\ncd");
        assert_no_soft_breaks(&s2.markdown);
    }

    #[test]
    fn paste_with_lone_newlines_normalizes_to_paragraphs() {
        let s = update(
            st("", 0),
            EditorEvent::InsertText("line1\nline2\nline3".into()),
        );
        assert_eq!(s.markdown, "line1\n\nline2\n\nline3");
        assert_no_soft_breaks(&s.markdown);
    }

    #[test]
    fn paste_with_existing_paragraph_breaks_preserved() {
        let s = update(st("", 0), EditorEvent::InsertText("a\n\nb\n\nc".into()));
        assert_eq!(s.markdown, "a\n\nb\n\nc");
    }

    #[test]
    fn paste_with_mixed_content_normalized() {
        let s = update(st("", 0), EditorEvent::InsertText("a\nb\n\nc\nd".into()));
        assert_eq!(s.markdown, "a\n\nb\n\nc\n\nd");
        assert_no_soft_breaks(&s.markdown);
    }

    #[test]
    fn paste_preserving_trailing_newline() {
        // README-style content often ends with a `\n`. Don't surprise the
        // user with an extra empty paragraph at the bottom.
        let s = update(st("", 0), EditorEvent::InsertText("hello\n".into()));
        assert_eq!(s.markdown, "hello\n");
    }

    #[test]
    fn shift_enter_emits_hard_break_in_middle() {
        let s = update(st("abcd", 2), EditorEvent::InsertLineBreak);
        assert_eq!(s.markdown, "ab  \ncd");
        assert_no_soft_breaks(&s.markdown);
    }

    // ---- Pairs model: Enter inserts `\n\n`, typing on trailing empty
    //      preserves visible empty count without any prepend trick ----------

    #[test]
    fn enter_inserts_two_newlines_at_end_of_paragraph() {
        let s = update(st("p1", 2), EditorEvent::InsertNewline);
        assert_eq!(s.markdown, "p1\n\n");
        assert_eq!(s.selection, Selection::Cursor(4));
    }

    #[test]
    fn second_enter_at_end_of_paragraph_grows_to_two_pairs() {
        let s = update(st("p1", 2), EditorEvent::InsertNewline);
        let s = update(s, EditorEvent::InsertNewline);
        // Two Enters from end: source is two `\n\n` units → 2 trailing
        // empties when rendered (`T / 2 = 2`).
        assert_eq!(s.markdown, "p1\n\n\n\n");
        assert_eq!(s.selection, Selection::Cursor(6));
    }

    #[test]
    fn typing_at_end_of_two_enter_run_preserves_visible_empty() {
        // 2 Enters from "p1" → `p1\n\n\n\n` (3 rows: p1 + 2 empties).
        // Typing X → `p1\n\n\n\nX`, which renders as p1 + 1 inter empty
        // + X = 3 rows. Visible-row count preserved without any
        // editor-side prepend.
        let mut s = st("p1", 2);
        s = update(s, EditorEvent::InsertNewline);
        s = update(s, EditorEvent::InsertNewline);
        s = update(s, EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "p1\n\n\n\nX");
    }

    #[test]
    fn typing_at_single_trailing_newline_promotes_via_enforce() {
        // A *single* trailing `\n` (anomalous in the pairs model — not
        // produced by Enter — but possible via paste of "p1\n").
        // Typing X gives `p1\nX`; the soft `\n` is then promoted to
        // `\n\n` by `enforce_invariants`. The user sees 0 empties → 0
        // empties. Stable.
        let s = update(st("p1\n", 3), EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "p1\n\nX");
    }

    #[test]
    fn typing_after_trailing_hard_break_fills_the_continuation_line() {
        // After Shift+Enter from end of "ab", source is "ab  \n" with a
        // hard break. The renderer shows that as a paragraph with two
        // visible lines (content + empty trailing). Typing X just fills
        // the continuation line — no extra paragraph injected.
        let s = update(st("ab  \n", 5), EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "ab  \nX");
    }

    #[test]
    fn enter_at_trailing_empty_adds_one_more_pair() {
        // Pressing Enter from a trailing-empty position extends the run
        // by one pair (one more visible empty).
        let s = update(st("p1\n\n", 4), EditorEvent::InsertNewline);
        assert_eq!(s.markdown, "p1\n\n\n\n");
    }

    #[test]
    fn shift_enter_inserts_a_hard_break_not_a_pair() {
        // Shift+Enter is the *line-break* keystroke (in-paragraph), not
        // a paragraph break, so it stays at one `\n`.
        let s = update(st("p1\n\n", 4), EditorEvent::InsertLineBreak);
        assert_eq!(s.markdown, "p1\n\n  \n");
    }

    #[test]
    fn typing_at_end_of_doc_without_trailing_newlines_appends_directly() {
        let s = update(st("p1", 2), EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "p1X");
    }

    #[test]
    fn typing_in_doc_of_only_newlines_does_not_promote() {
        // No content before the trailing run, so this is leading-only.
        // Inserting X gives `\n\nX` — the leading pair is L=2 → 1 leading
        // empty above X.
        let s = update(st("\n\n", 2), EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "\n\nX");
    }

    // ---- Backspace at paragraph boundaries ------------------------------------

    #[test]
    fn backspace_at_start_of_second_paragraph_merges() {
        // Cursor right before 'c' in "ab\n\ncd" — pressing backspace at the
        // start of the "cd" paragraph collapses the break.
        let s = update(st("ab\n\ncd", 4), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "abcd");
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn backspace_inside_two_newline_run_merges() {
        // Cursor sat between the two `\n`s of a paragraph break (a position
        // a click could land on). Backspace still merges the paragraphs in
        // one keystroke rather than feeling like a no-op.
        let s = update(st("ab\n\ncd", 3), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "abcd");
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn backspace_through_odd_run_normalizes_via_pair_delete_plus_promotion() {
        // Source `ab\n\n\ncd` is anomalous in the pairs model (3 `\n`s,
        // odd). Backspace removes a pair, leaving `ab\ncd` — one stray
        // `\n` mid-content. `enforce_invariants` promotes that to
        // `\n\n`. Net effect: ends up as a regular paragraph break.
        let s = update(st("ab\n\n\ncd", 5), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "ab\n\ncd");
        assert_no_soft_breaks(&s.markdown);
    }

    #[test]
    fn backspace_through_empty_paragraphs_drops_one_pair_at_a_time() {
        // Pairs model: source `ab\n\n\n\ncd` is paragraph break + 1 empty.
        // Each backspace deletes a pair (= one paragraph "unit"). First
        // press removes the empty; second press collapses the break.
        let s = update(st("ab\n\n\n\ncd", 6), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "ab\n\ncd");
        let s = update(s, EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "abcd");
    }

    #[test]
    fn backspace_through_three_empty_paragraphs_drops_one_pair_at_a_time() {
        // Source `ab\n\n\n\n\n\ncd` is paragraph break + 2 empties.
        let s = update(st("ab\n\n\n\n\n\ncd", 8), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "ab\n\n\n\ncd");
        let s = update(s, EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "ab\n\ncd");
        let s = update(s, EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "abcd");
    }

    #[test]
    fn backspace_at_trailing_newline_drops_one() {
        let s = update(st("ab\n", 3), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "ab");
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn backspace_at_leading_paragraph_break_removes_a_pair() {
        // Pairs model: a leading run of 2 `\n`s is one leading empty.
        // Backspace at the start of the first content paragraph removes
        // the entire pair (the empty above it).
        let s = update(st("\n\nab", 2), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "ab");
        assert_no_soft_breaks(&s.markdown);
    }

    // ---- Delete-forward symmetric -------------------------------------------

    #[test]
    fn delete_forward_at_end_of_first_paragraph_merges() {
        let s = update(st("ab\n\ncd", 2), EditorEvent::DeleteForward);
        assert_eq!(s.markdown, "abcd");
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn delete_forward_inside_two_newline_run_merges() {
        let s = update(st("ab\n\ncd", 3), EditorEvent::DeleteForward);
        assert_eq!(s.markdown, "abcd");
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn delete_forward_through_empty_paragraph_drops_one() {
        let s = update(st("ab\n\n\ncd", 2), EditorEvent::DeleteForward);
        assert_eq!(s.markdown, "ab\n\ncd");
        // Cursor stayed in place; deletion happened to the right.
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    // ---- Range selection deletion -------------------------------------------

    #[test]
    fn selection_replacing_paragraph_break_with_text() {
        let initial = EditorState {
            markdown: "ab\n\ncd".into(),
            selection: Selection::range(1, 5),
        };
        let s = update(initial, EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "aXd");
    }

    #[test]
    fn selection_deleting_one_newline_of_break_repromotes() {
        // The user selected exactly one `\n` of a `\n\n` break and pressed
        // delete. The post-pass promotes the lone `\n` back, so the doc is
        // unchanged but the cursor is somewhere sensible. (UX could be
        // smarter here, but consistency-of-source is the load-bearing
        // property; we won't ship a doc with a soft break in it.)
        let initial = EditorState {
            markdown: "ab\n\ncd".into(),
            selection: Selection::range(2, 3),
        };
        let s = update(initial, EditorEvent::DeleteForward);
        assert_eq!(s.markdown, "ab\n\ncd");
        assert_no_soft_breaks(&s.markdown);
    }

    // ---- Hard breaks --------------------------------------------------------

    #[test]
    fn hard_break_with_typed_content_after_stays_intact() {
        let s1 = update(st("ab", 2), EditorEvent::InsertLineBreak);
        assert_eq!(s1.markdown, "ab  \n");
        let s2 = update(s1, EditorEvent::InsertText("X".into()));
        assert_eq!(s2.markdown, "ab  \nX");
        assert_no_soft_breaks(&s2.markdown);
    }

    #[test]
    fn deleting_one_trailing_space_of_hard_break_promotes_to_paragraph_break() {
        // The user backspaced over one of the trailing spaces, so the
        // sequence is no longer a hard break. The lone `\n` becomes a soft
        // break, and the post-pass turns the break into a paragraph break.
        // This is a deliberate, documented behavior: we never silently
        // ship a soft break.
        let s = update(st("ab  \nX", 4), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "ab \n\nX");
        assert_no_soft_breaks(&s.markdown);
    }

    // ---- Movement doesn't introduce soft breaks ----------------------------

    #[test]
    fn navigation_events_preserve_already_clean_markdown() {
        let starts = [
            "ab\n\ncd",
            "ab\n\n\ncd",
            "ab  \ncd",
            "ab\\\ncd",
            "\n\nab",
            "ab\n",
        ];
        for src in starts {
            for evt in [
                EditorEvent::MoveLeft,
                EditorEvent::MoveRight,
                EditorEvent::MoveLineStart,
                EditorEvent::MoveLineEnd,
                EditorEvent::MoveDocumentStart,
                EditorEvent::MoveDocumentEnd,
            ] {
                let s = update(st(src, src.len() / 2), evt);
                assert_eq!(s.markdown, src, "navigation altered markdown {src:?}");
                assert_no_soft_breaks(&s.markdown);
            }
        }
    }

    // ---- Stress / fuzz-ish ---------------------------------------------------

    #[test]
    fn typing_pause_typing_preserves_invariant() {
        // Insert "a" → Enter → "b" → Enter → "c". Each step's output is
        // re-validated.
        let mut s = st("", 0);
        let steps: &[EditorEvent] = &[
            EditorEvent::InsertText("a".into()),
            EditorEvent::InsertNewline,
            EditorEvent::InsertText("b".into()),
            EditorEvent::InsertNewline,
            EditorEvent::InsertText("c".into()),
        ];
        for evt in steps {
            s = update(s, evt.clone());
            assert_no_soft_breaks(&s.markdown);
        }
        assert_eq!(s.markdown, "a\n\nb\n\nc");
    }

    #[test]
    fn alternating_insert_and_delete_preserves_invariant() {
        let mut s = st("hello world", 5);
        let steps = [
            EditorEvent::InsertNewline,
            EditorEvent::DeleteBackward,
            EditorEvent::InsertNewline,
            EditorEvent::InsertNewline,
            EditorEvent::DeleteBackward,
            EditorEvent::DeleteForward,
        ];
        for evt in steps {
            s = update(s, evt);
            assert_no_soft_breaks(&s.markdown);
        }
    }
}

// ---------------------------------------------------------------------------
// Forbidden-position invariant — the cursor must never sit inside a
// structural `\n\n` pair. Pair interiors are visually unreachable (the
// pair renders as one row), and typing there would split the pair into a
// stray odd-length newline run. Every state transition in `update()` runs
// the post-pass that snaps offending cursors away, with the snap
// direction determined by where the cursor came from.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod forbidden_position_tests {
    use super::*;

    fn st(s: &str, cursor: usize) -> EditorState {
        EditorState {
            markdown: s.into(),
            selection: Selection::Cursor(cursor),
        }
    }

    fn assert_no_forbidden(state: &EditorState) {
        let bytes = state.markdown.as_bytes();
        let positions: Vec<usize> = match state.selection {
            Selection::Cursor(p) => vec![p],
            Selection::Range { anchor, head } => vec![anchor, head],
        };
        for p in positions {
            assert!(
                !is_forbidden_position(bytes, p),
                "selection endpoint {p} is forbidden in {:?}",
                state.markdown
            );
        }
    }

    // ---- The detector itself -----------------------------------------

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
        // `p1\n\n\n\np2`: 4-newline run = 2 pairs. Position 4 is the
        // boundary between the two pairs (allowed), positions 3 and 5
        // are pair interiors (forbidden).
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
        assert!(!is_forbidden_position(bytes, 8)); // start of p2
    }

    #[test]
    fn leading_pair_interior_is_forbidden() {
        let bytes = b"\n\nab";
        assert!(is_forbidden_position(bytes, 1));
        assert!(!is_forbidden_position(bytes, 0));
        assert!(!is_forbidden_position(bytes, 2));
    }

    #[test]
    fn hard_break_alone_has_no_forbidden_positions() {
        // `ab  \n` — single hard break, no structural pair.
        let bytes = b"ab  \n";
        for p in 0..=bytes.len() {
            assert!(
                !is_forbidden_position(bytes, p),
                "p={p} unexpectedly forbidden"
            );
        }
    }

    #[test]
    fn hard_break_followed_by_structural_pair_only_marks_pair_interior() {
        // `ab  \n\n\ncd` — hard break (\n at 4) + 2 structural \n's
        // (positions 5, 6). The structural pair is [5, 7). Position 6
        // is the pair interior (forbidden); the hard-break \n doesn't
        // pull position 5 into being forbidden because the back-walk
        // stops at the hard-break terminator.
        let bytes = b"ab  \n\n\ncd";
        assert!(!is_forbidden_position(bytes, 5));
        assert!(is_forbidden_position(bytes, 6));
        assert!(!is_forbidden_position(bytes, 7));
    }

    #[test]
    fn backslash_hard_break_treated_same() {
        // `ab\\\n\n\ncd` — backslash + \n is a hard break; same shape.
        let bytes = b"ab\\\n\n\ncd";
        // bytes: a(0) b(1) \\(2) \n(3) \n(4) \n(5) c(6) d(7)
        // Hard-break \n at 3 (preceded by \\ at 2). Structural pair
        // [4, 6); interior is 5.
        assert!(!is_forbidden_position(bytes, 4));
        assert!(is_forbidden_position(bytes, 5));
        assert!(!is_forbidden_position(bytes, 6));
    }

    #[test]
    fn doc_edges_are_never_forbidden() {
        let bytes = b"\n\n";
        assert!(!is_forbidden_position(bytes, 0));
        assert!(!is_forbidden_position(bytes, bytes.len()));
    }

    // ---- Movement directional snapping --------------------------------

    #[test]
    fn right_arrow_skips_paragraph_break_interior() {
        let s = update(st("p1\n\np2", 2), EditorEvent::MoveRight);
        assert_eq!(s.selection, Selection::Cursor(4));
        assert_no_forbidden(&s);
    }

    #[test]
    fn left_arrow_skips_paragraph_break_interior() {
        let s = update(st("p1\n\np2", 4), EditorEvent::MoveLeft);
        assert_eq!(s.selection, Selection::Cursor(2));
        assert_no_forbidden(&s);
    }

    #[test]
    fn right_arrow_through_extra_pair_lands_at_inter_pair_boundary() {
        // `p1\n\n\n\np2` — Right from end of p1 (byte 2) skips byte 3
        // (forbidden) and lands at byte 4 (between the two pairs,
        // allowed; visually on the empty row).
        let s = update(st("p1\n\n\n\np2", 2), EditorEvent::MoveRight);
        assert_eq!(s.selection, Selection::Cursor(4));
        assert_no_forbidden(&s);
    }

    #[test]
    fn left_arrow_through_extra_pair_lands_at_inter_pair_boundary() {
        let s = update(st("p1\n\n\n\np2", 6), EditorEvent::MoveLeft);
        assert_eq!(s.selection, Selection::Cursor(4));
        assert_no_forbidden(&s);
    }

    #[test]
    fn right_arrow_off_inter_pair_boundary_lands_at_next_real_block() {
        // From the inter-pair boundary (byte 4), Right would land at
        // byte 5 (forbidden, interior of pair 2). Snap forward → 6
        // (start of p2).
        let s = update(st("p1\n\n\n\np2", 4), EditorEvent::MoveRight);
        assert_eq!(s.selection, Selection::Cursor(6));
        assert_no_forbidden(&s);
    }

    #[test]
    fn left_arrow_off_inter_pair_boundary_lands_at_prev_real_block() {
        let s = update(st("p1\n\n\n\np2", 4), EditorEvent::MoveLeft);
        assert_eq!(s.selection, Selection::Cursor(2));
        assert_no_forbidden(&s);
    }

    #[test]
    fn down_from_first_paragraph_skips_break_to_second() {
        // `p1\n\np2` MoveDown from byte 0: target byte 3 (forbidden).
        // Snap forward → byte 4 (start of p2).
        let s = update(st("p1\n\np2", 0), EditorEvent::MoveDown);
        assert_eq!(s.selection, Selection::Cursor(4));
        assert_no_forbidden(&s);
    }

    #[test]
    fn up_from_second_paragraph_skips_break_to_first() {
        // `move_vertical` skips the phantom (forbidden) line between
        // paragraphs and preserves column 0, so Up from start of p2
        // lands at start of p1, not at end-of-p1 like a simple
        // post-snap would produce.
        let s = update(st("p1\n\np2", 4), EditorEvent::MoveUp);
        assert_eq!(s.selection, Selection::Cursor(0));
        assert_no_forbidden(&s);
    }

    #[test]
    fn down_from_first_paragraph_lands_on_visible_empty_row() {
        // `p1\n\n\n\np2` MoveDown from p1@0: target → 3 (forbidden) →
        // snap forward to 4. Byte 4 is on the empty row (synthetic
        // empty has range 3..5).
        let s = update(st("p1\n\n\n\np2", 0), EditorEvent::MoveDown);
        assert_eq!(s.selection, Selection::Cursor(4));
        assert_no_forbidden(&s);
    }

    #[test]
    fn up_from_p2_in_long_run_lands_on_empty_row() {
        let s = update(st("p1\n\n\n\np2", 6), EditorEvent::MoveUp);
        assert_eq!(s.selection, Selection::Cursor(4));
        assert_no_forbidden(&s);
    }

    // ---- ExtendLeft / ExtendRight (range selections) ------------------

    #[test]
    fn extend_right_skips_forbidden_interior() {
        // From cursor at byte 2 in `p1\n\np2`, Shift+Right should extend
        // the selection past the forbidden byte 3 to end at byte 4.
        let s = update(st("p1\n\np2", 2), EditorEvent::ExtendRight);
        assert_eq!(s.selection, Selection::range(2, 4));
        assert_no_forbidden(&s);
    }

    #[test]
    fn extend_left_skips_forbidden_interior() {
        let s = update(st("p1\n\np2", 4), EditorEvent::ExtendLeft);
        assert_eq!(s.selection, Selection::range(4, 2));
        assert_no_forbidden(&s);
    }

    // ---- SetSelection (the host API) ----------------------------------
    //
    // SetSelection uses *nearest-allowed* (idempotent), not prev-comparison.
    // That keeps mouse-drag stable: every `mouse_move` fires another
    // `SetSelection` with the same offset, and each must produce the
    // same result regardless of where the previously-snapped cursor
    // ended up. See the comment at the top of `set_selection`.

    #[test]
    fn set_selection_to_forbidden_snaps_to_nearest_allowed() {
        // pos=3 in `p1\n\np2`: equidistant between allowed 2 and 4.
        // Forward wins ties, matching the post-pass default. Same
        // result regardless of where the cursor was before — that's
        // the load-bearing idempotence property.
        for prev in [0, 1, 2, 3, 4, 5, 6] {
            let s = update(
                st("p1\n\np2", prev),
                EditorEvent::SetSelection(Selection::Cursor(3)),
            );
            assert_eq!(
                s.selection,
                Selection::Cursor(4),
                "prev={prev} gave {:?}",
                s.selection
            );
        }
    }

    #[test]
    fn repeated_set_selection_at_forbidden_offset_is_stable() {
        // Regression for the mouse-drag flicker: a `MouseMoveEvent`
        // re-fires `SetSelection` at the same offset on every frame.
        // Under prev-comparison the cursor would oscillate between the
        // two adjacent allowed positions (2 and 4) because each call
        // saw the previous snap as `prev`. Nearest-allowed makes it
        // converge in one step.
        let mut s = st("p1\n\np2", 0);
        for _ in 0..5 {
            s = update(s, EditorEvent::SetSelection(Selection::Cursor(3)));
            assert_eq!(s.selection, Selection::Cursor(4));
        }
    }

    #[test]
    fn set_selection_inside_inter_block_pair_lands_on_visible_empty_row() {
        // The user-reported case: clicking on a synthetic empty row
        // between paragraphs. The renderer's `display_to_source` for
        // the empty maps clicks to the pair interior (forbidden);
        // nearest-allowed snaps to the inter-pair boundary, which is
        // visually the same row.
        let s = update(
            st("p1\n\n\n\np2", 0),
            EditorEvent::SetSelection(Selection::Cursor(3)),
        );
        // Interior of pair 1; equidistant between 2 (end of p1) and 4
        // (visible empty row). Forward wins → 4.
        assert_eq!(s.selection, Selection::Cursor(4));
    }

    // ---- Range endpoints snap independently ---------------------------

    #[test]
    fn range_anchor_and_head_snap_independently() {
        // Both endpoints land on forbidden interiors of pair 1 / pair
        // 2. nearest-allowed snaps each independently — anchor 3 is
        // tied between 2 and 4 (forward → 4); head 5 is tied between
        // 4 and 6 (forward → 6). Result: a real range, not a collapse.
        let initial = EditorState {
            markdown: "p1\n\n\n\np2".into(),
            selection: Selection::Cursor(0),
        };
        let s = update(
            initial,
            EditorEvent::SetSelection(Selection::Range { anchor: 3, head: 5 }),
        );
        assert_eq!(s.selection, Selection::range(4, 6));
    }

    // ---- Hard breaks pass through unchanged --------------------------

    #[test]
    fn navigation_around_hard_break_is_unaffected() {
        // `ab  \nX` cursor right after the hard-break \n. Right arrow
        // moves to byte 6 (after X). No forbidden positions in this doc.
        let s = update(st("ab  \nX", 5), EditorEvent::MoveRight);
        assert_eq!(s.selection, Selection::Cursor(6));
        assert_no_forbidden(&s);
    }

    // ---- Empty-row navigation round-trip ------------------------------

    // ---- Trailing-empty typing semantics ------------------------------
    //
    // The trailing-empty layout shifts each pair by 1 inside the gap so
    // the cursor's resting position when on row N is the typing position
    // that creates content for row N. With offset 0, "cursor on empty
    // row" and "cursor at end of paragraph" share a source offset, and
    // typing extends the paragraph instead.

    #[test]
    fn enter_at_end_of_paragraph_then_type_creates_new_paragraph_for_empty_row() {
        // The user-flow regression. Enter from end of `paragraph` puts
        // the cursor on the empty row below. Typing X must create a new
        // "X" paragraph on that row, not extend "paragraph" to
        // "paragraphX".
        let mut s = st("paragraph", 9);
        s = update(s, EditorEvent::InsertNewline);
        assert_eq!(s.markdown, "paragraph\n\n");
        assert_eq!(s.selection, Selection::Cursor(11));
        s = update(s, EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "paragraph\n\nX");
    }

    #[test]
    fn right_arrow_from_end_of_paragraph_into_trailing_empty_then_type() {
        // Cursor at end of paragraph, Right lands on the empty row, X
        // creates a new paragraph for it.
        let mut s = st("paragraph\n\n\n\n\n\n", 9);
        s = update(s, EditorEvent::MoveRight);
        assert_eq!(s.selection, Selection::Cursor(11));
        s = update(s, EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "paragraph\n\nX\n\n\n\n");
        assert_no_forbidden(&s);
    }

    #[test]
    fn click_offset_at_trailing_empty_anchor_snaps_to_typing_position() {
        // Trailing empties have `display_to_source = [block.start]`,
        // and `block.start` is the *forbidden* interior of the
        // structural pair. Clicking on the empty row therefore feeds
        // `SetSelection(block.start)` through `update`, which uses
        // nearest-allowed and lands at the empty's strict-interior
        // typing position.
        let s = update(
            st("paragraph\n\n\n\n\n\n", 0),
            EditorEvent::SetSelection(Selection::Cursor(10)),
        );
        // Empty 1 is range 10..12; click anchor is 10 (forbidden).
        // Nearest allowed: prev=9 (end of paragraph) and next=11
        // (typing position). Forward wins → 11.
        assert_eq!(s.selection, Selection::Cursor(11));
    }

    #[test]
    fn arrow_round_trip_stays_clear_of_forbidden() {
        // Walk Right from the start of `p1\n\n\n\n\n\np2` and back
        // again; every step must produce an allowed cursor.
        let mut s = st("p1\n\n\n\n\n\np2", 0);
        let path = [
            EditorEvent::MoveRight, // 0 → 1
            EditorEvent::MoveRight, // 1 → 2
            EditorEvent::MoveRight, // 2 → skip 3 → 4
            EditorEvent::MoveRight, // 4 → skip 5 → 6
            EditorEvent::MoveRight, // 6 → skip 7 → 8 (start of p2)
            EditorEvent::MoveRight, // 8 → 9
            EditorEvent::MoveRight, // 9 → 10
        ];
        let expected = [1usize, 2, 4, 6, 8, 9, 10];
        for (evt, want) in path.into_iter().zip(expected) {
            s = update(s, evt);
            assert_eq!(s.selection, Selection::Cursor(want));
            assert_no_forbidden(&s);
        }
        // Walk back to the start.
        let back = [
            EditorEvent::MoveLeft, // 10 → 9
            EditorEvent::MoveLeft, // 9 → 8
            EditorEvent::MoveLeft, // 8 → skip 7 → 6
            EditorEvent::MoveLeft, // 6 → skip 5 → 4
            EditorEvent::MoveLeft, // 4 → skip 3 → 2
            EditorEvent::MoveLeft, // 2 → 1
            EditorEvent::MoveLeft, // 1 → 0
        ];
        let expected = [9usize, 8, 6, 4, 2, 1, 0];
        for (evt, want) in back.into_iter().zip(expected) {
            s = update(s, evt);
            assert_eq!(s.selection, Selection::Cursor(want));
            assert_no_forbidden(&s);
        }
    }
}
