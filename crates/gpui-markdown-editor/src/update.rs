//! Pure state transitions: `update(state, event) -> state`.
//!
//! Minimum-viable scope. No markdown-aware post-processing yet (no setext
//! normalization, no list renumbering, no smart-prefix on Enter). Those land
//! as we add the constructs that need them.

use unicode_segmentation::UnicodeSegmentation;

use crate::event::EditorEvent;
use crate::state::{EditorState, Selection};

pub fn update(state: EditorState, event: EditorEvent) -> EditorState {
    match event {
        EditorEvent::InsertText(text) => insert_text(state, &text),
        EditorEvent::InsertNewline => insert_text(state, "\n"),
        EditorEvent::InsertLineBreak => insert_text(state, "  \n"),

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
    }
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
    let prev = prev_grapheme_offset(&state.markdown, cursor);
    let mut buf = String::with_capacity(state.markdown.len() - (cursor - prev));
    buf.push_str(&state.markdown[..prev]);
    buf.push_str(&state.markdown[cursor..]);
    EditorState {
        markdown: buf,
        selection: Selection::Cursor(prev),
    }
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
    let next = next_grapheme_offset(&state.markdown, cursor);
    let mut buf = String::with_capacity(state.markdown.len() - (next - cursor));
    buf.push_str(&state.markdown[..cursor]);
    buf.push_str(&state.markdown[next..]);
    EditorState {
        markdown: buf,
        selection: Selection::Cursor(cursor),
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
    EditorState {
        selection: snap_selection_to_char_boundaries(&state.markdown, normalized),
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

fn move_vertical(text: &str, pos: usize, direction: i32) -> usize {
    let line_start = line_start_offset(text, pos);
    let column = pos - line_start;
    if direction < 0 {
        if line_start == 0 {
            return 0;
        }
        let prev_line_end = line_start - 1;
        let prev_line_start = line_start_offset(text, prev_line_end);
        let prev_line_len = prev_line_end - prev_line_start;
        let target = prev_line_start + column.min(prev_line_len);
        snap_to_char_boundary(text, target)
    } else {
        let line_end = line_end_offset(text, pos);
        if line_end >= text.len() {
            return text.len();
        }
        let next_line_start = line_end + 1;
        let next_line_end = line_end_offset(text, next_line_start);
        let next_line_len = next_line_end - next_line_start;
        let target = next_line_start + column.min(next_line_len);
        snap_to_char_boundary(text, target)
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
    fn insert_newline_inserts_lf() {
        let s = update(st("abc", 1), EditorEvent::InsertNewline);
        assert_eq!(s.markdown, "a\nbc");
        assert_eq!(s.selection, Selection::Cursor(2));
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
        let text = "abc\ndef";
        let s = update(st(text, 5), EditorEvent::MoveLineStart);
        assert_eq!(s.selection, Selection::Cursor(4));
        let s = update(st(text, 5), EditorEvent::MoveLineEnd);
        assert_eq!(s.selection, Selection::Cursor(7));
    }

    #[test]
    fn move_up_preserves_column_when_possible() {
        let text = "abcdef\nghij";
        let s = update(st(text, 9), EditorEvent::MoveUp); // on 'i' → expect 'c'
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn document_start_and_end() {
        let s = update(st("abc\ndef", 5), EditorEvent::MoveDocumentStart);
        assert_eq!(s.selection, Selection::Cursor(0));
        let s = update(st("abc\ndef", 0), EditorEvent::MoveDocumentEnd);
        assert_eq!(s.selection, Selection::Cursor(7));
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
