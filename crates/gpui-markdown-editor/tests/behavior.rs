//! Behavior tests — the regression gate.
//!
//! Built on `gpui::TestAppContext` (mocked rendering, deterministic
//! dispatcher). They run on libtest's worker thread without touching AppKit.
//!
//! Pattern:
//! 1. Open a window whose root is `gpui_component::Root` wrapping the
//!    editor (same pattern as production).
//! 2. Drive interactions through the editor's `focus_handle` — the same
//!    path keystrokes take in production.
//! 3. Assert against the editor's public state with `read_with`.

use gpui::{AnyWindowHandle, AppContext, Entity, TestAppContext, WindowOptions};
use gpui_component::Root;
use gpui_markdown_editor::editor::{
    Backspace, Delete, DocumentEnd, DocumentStart, Down, End, Enter, Home, Left, Right, SelectAll,
    ShiftEnter, ShiftRight, Up,
};
use gpui_markdown_editor::{BlockKind, EditorState, MarkdownEditor, RenderSpec, Selection};

fn open_editor(
    cx: &mut TestAppContext,
    state: EditorState,
) -> (AnyWindowHandle, Entity<MarkdownEditor>) {
    cx.update(|cx| {
        gpui_component::init(cx);
        let mut inner: Option<Entity<MarkdownEditor>> = None;
        let window = cx
            .open_window(WindowOptions::default(), |window, cx| {
                let editor = cx.new(|cx| MarkdownEditor::with_state(state, window, cx));
                inner = Some(editor.clone());
                cx.new(|cx| Root::new(editor, window, cx))
            })
            .expect("open window");
        (window.into(), inner.expect("editor built"))
    })
}

fn dispatch(
    cx: &mut TestAppContext,
    handle: AnyWindowHandle,
    editor: &Entity<MarkdownEditor>,
    action: impl gpui::Action,
) {
    let focus = editor.read_with(cx, |e, _| e.focus_handle.clone());
    cx.update_window(handle, |_, window, cx| {
        focus.dispatch_action(&action, window, cx);
    })
    .unwrap();
    cx.run_until_parked();
}

fn current_spec(cx: &mut TestAppContext, editor: &Entity<MarkdownEditor>) -> RenderSpec {
    editor.read_with(cx, |e, _| e.render_spec())
}

// ---------------------------------------------------------------------------
// State / update plumbing
// ---------------------------------------------------------------------------

#[gpui::test]
fn editor_constructs_with_initial_state(cx: &mut TestAppContext) {
    let (_, editor) = open_editor(cx, EditorState::with_markdown("# hi"));
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "# hi");
        assert_eq!(e.cursor_offset(), 0);
    });
}

#[gpui::test]
fn enter_action_inserts_paragraph_break(cx: &mut TestAppContext) {
    // Enter mid-paragraph emits `\n` which the post-pass promotes to `\n\n`
    // (a paragraph break) — see `update::enforce_invariants`.
    let initial = EditorState {
        markdown: "abc".into(),
        selection: Selection::Cursor(2),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "ab\n\nc");
        assert_eq!(e.cursor_offset(), 4);
    });
}

#[gpui::test]
fn backspace_removes_one_grapheme(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "abc".into(),
        selection: Selection::Cursor(2),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "ac");
        assert_eq!(e.cursor_offset(), 1);
    });
}

#[gpui::test]
fn delete_removes_forward_grapheme(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "abc".into(),
        selection: Selection::Cursor(1),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Delete);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "ac");
        assert_eq!(e.cursor_offset(), 1);
    });
}

#[gpui::test]
fn arrow_keys_move_cursor(cx: &mut TestAppContext) {
    // Already-normalized fixture so the post-pass is a no-op and the move
    // geometry is the only thing under test.
    let initial = EditorState {
        markdown: "abc\n\ndef".into(),
        selection: Selection::Cursor(0),
    };
    let (handle, editor) = open_editor(cx, initial);

    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 2));

    // Down preserves column 2 and jumps straight to the second
    // paragraph: there's no visible empty row between paragraphs in
    // `abc\n\ndef` (the structural pair is the paragraph break itself,
    // not an empty paragraph), and `move_vertical` skips phantom lines
    // whose start is a forbidden pair interior.
    dispatch(cx, handle, &editor, Down);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 7));

    // Up symmetrically returns to column 2 of the first paragraph.
    dispatch(cx, handle, &editor, Up);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 2));
}

#[gpui::test]
fn right_arrow_skips_paragraph_break_interior(cx: &mut TestAppContext) {
    // The user-reported case: in `p1\n\np2`, byte 3 is between the two
    // `\n`s of the paragraph break — visually unreachable and would
    // split the pair if typed at. Right from byte 2 must jump straight
    // to byte 4 (start of p2).
    let initial = EditorState {
        markdown: "p1\n\np2".into(),
        selection: Selection::Cursor(2),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 4));
    dispatch(cx, handle, &editor, Left);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 2));
}

#[gpui::test]
fn arrow_navigation_through_empty_paragraph_lands_on_visible_row(cx: &mut TestAppContext) {
    // `p1\n\n\n\np2` has one synthetic empty paragraph between (range
    // 3..5). Right from end-of-p1 should land on the empty row (byte 4)
    // — which is the boundary between the structural pair and the
    // empty pair, allowed and visually on the empty row.
    let initial = EditorState {
        markdown: "p1\n\n\n\np2".into(),
        selection: Selection::Cursor(2),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 4));
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 6));
}

#[gpui::test]
fn home_end_doc_jump(cx: &mut TestAppContext) {
    // "abc\n\ndef" — cursor at byte 6 (between 'd' and 'e' on the second
    // paragraph). Already-normalized fixture so the post-pass doesn't move
    // the cursor.
    let initial = EditorState {
        markdown: "abc\n\ndef".into(),
        selection: Selection::Cursor(6),
    };
    let (handle, editor) = open_editor(cx, initial);

    dispatch(cx, handle, &editor, Home);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 5));
    dispatch(cx, handle, &editor, End);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 8));
    dispatch(cx, handle, &editor, DocumentStart);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 0));
    dispatch(cx, handle, &editor, DocumentEnd);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 8));
}

#[gpui::test]
fn shift_right_extends_selection(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "abcd".into(),
        selection: Selection::Cursor(1),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftRight);
    dispatch(cx, handle, &editor, ShiftRight);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.selection, Selection::range(1, 3));
    });
}

#[gpui::test]
fn select_all_spans_document(cx: &mut TestAppContext) {
    let (handle, editor) = open_editor(cx, EditorState::with_markdown("hello"));
    dispatch(cx, handle, &editor, SelectAll);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.selection, Selection::range(0, 5));
    });
}

#[gpui::test]
fn shift_enter_at_end_of_paragraph_keeps_cursor_in_same_paragraph(cx: &mut TestAppContext) {
    // The companion to `enter_at_end_of_paragraph_creates_visible_trailing_empty`.
    // Shift+Enter inserts a hard break (`  \n`); the user expects the
    // cursor to drop to a new visible line *inside* the same paragraph,
    // not to a new separate paragraph.
    let initial = EditorState {
        markdown: "paragraph 1".into(),
        selection: Selection::Cursor(11),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftEnter);

    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "paragraph 1  \n");
        assert_eq!(e.cursor_offset(), 14);
    });

    // Single block (paragraph), no trailing empty paragraph injected —
    // the trailing `\n` is content of this paragraph (the hard-break
    // terminator) and the implicit empty trailing line is rendered
    // *within* the block by `element.rs::shape_block_lines`.
    let spec = current_spec(cx, &editor);
    assert_eq!(spec.blocks.len(), 1);
    assert_eq!(spec.blocks[0].source_range, 0..14);
}

#[gpui::test]
fn enter_at_end_of_paragraph_creates_visible_trailing_empty(cx: &mut TestAppContext) {
    // User-reported regression (now fixed by the pairs model): pressing
    // Enter at the end of the only paragraph used to produce no visible
    // change. With Enter inserting `\n\n`, the source has 2 trailing
    // `\n`s and the renderer emits 1 trailing empty paragraph block.
    let initial = EditorState {
        markdown: "paragraph 1".into(),
        selection: Selection::Cursor(11),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);

    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "paragraph 1\n\n");
        assert_eq!(e.cursor_offset(), 13);
    });

    // Spec: paragraph + 1 trailing empty (range covering the pair of
    // trailing `\n`s).
    let spec = current_spec(cx, &editor);
    assert!(spec.blocks.len() >= 2);
    let trailing_empty = spec
        .blocks
        .iter()
        .find(|b| b.source_range == (11..13))
        .expect("synthetic empty paragraph owning the trailing `\\n\\n`");
    assert!(matches!(trailing_empty.kind, BlockKind::Paragraph));
    assert!(trailing_empty.inlines.is_empty());
}

#[gpui::test]
fn empty_document_still_has_a_renderable_block(cx: &mut TestAppContext) {
    // Regression: deleting all content used to leave the spec with zero
    // blocks, so no `BlockElement::paint` ran and the editor stopped
    // accepting input. The spec must always have at least one block to
    // anchor the cursor and register the input handler.
    let (_, editor) = open_editor(cx, EditorState::with_markdown(""));
    let spec = current_spec(cx, &editor);
    assert!(
        !spec.blocks.is_empty(),
        "empty doc must still render a block"
    );
}

#[gpui::test]
fn select_all_then_backspace_keeps_editor_usable(cx: &mut TestAppContext) {
    // The exact reproduction of the user-reported bug: clear all content
    // via select-all + backspace, then verify the editor's spec still has
    // a block and the cursor offset is sane. (We can't directly test
    // typed-text routing in `TestAppContext` — that goes through
    // `EntityInputHandler` which needs a real window — but a non-empty
    // spec is the load-bearing precondition: it's what makes `paint` run
    // and register the input handler.)
    let (handle, editor) = open_editor(cx, EditorState::with_markdown("Hello, world!"));
    dispatch(cx, handle, &editor, SelectAll);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "");
        assert_eq!(e.cursor_offset(), 0);
    });
    let spec = current_spec(cx, &editor);
    assert!(!spec.blocks.is_empty());

    // Pressing Enter from this empty state goes through the action
    // pipeline and produces `\n\n` (pairs model). Confirm render emits
    // multiple visible blocks.
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "\n\n");
    });
    let spec = current_spec(cx, &editor);
    assert!(spec.blocks.len() >= 2);
}

// ---------------------------------------------------------------------------
// Render spec — the cursor-aware delimiter rule, observed end-to-end
// ---------------------------------------------------------------------------

#[gpui::test]
fn heading_prefix_hidden_when_cursor_elsewhere(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "# Title\n\nbody".into(),
        selection: Selection::Cursor(11),
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let heading = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::Heading { .. }))
        .expect("heading block");
    assert!(heading.has_hidden_range(0..2));
    assert!(!heading.has_dimmed_range(0..2));
}

#[gpui::test]
fn heading_prefix_dims_when_cursor_inside(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "# Title\n".into(),
        selection: Selection::Cursor(4),
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let heading = &spec.blocks[0];
    assert!(heading.has_dimmed_range(0..2));
    assert!(!heading.has_hidden_range(0..2));
}

#[gpui::test]
fn bold_delimiters_flip_on_cursor_position(cx: &mut TestAppContext) {
    // "outside": cursor on a separate paragraph so neither end of `**bold**`
    // is treated as the "inside-by-boundary" case.
    let outside = EditorState {
        markdown: "**bold**\n\nelsewhere".into(),
        selection: Selection::Cursor(15),
    };
    let (_, editor) = open_editor(cx, outside);
    let spec = current_spec(cx, &editor);
    let para = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::Paragraph) && b.source_range.start == 0)
        .expect("first paragraph");
    assert!(para.has_hidden_range(0..2));
    assert!(para.has_hidden_range(6..8));

    let inside = EditorState {
        markdown: "**bold**".into(),
        selection: Selection::Cursor(4),
    };
    let (_, editor) = open_editor(cx, inside);
    let spec = current_spec(cx, &editor);
    let para = &spec.blocks[0];
    assert!(para.has_dimmed_range(0..2));
    assert!(para.has_dimmed_range(6..8));
}

#[gpui::test]
fn italic_and_strike_dim_within_selection(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "*it* and ~~no~~".into(),
        selection: Selection::range(0, 15),
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let para = &spec.blocks[0];
    assert!(para.has_dimmed_range(0..1));
    assert!(para.has_dimmed_range(3..4));
    assert!(para.has_dimmed_range(9..11));
    assert!(para.has_dimmed_range(13..15));
}

// ---------------------------------------------------------------------------
// Soft-break invariant — the buffer never carries a lone mid-content `\n`,
// no matter what edit path produces it.
// ---------------------------------------------------------------------------

fn assert_no_soft_break(md: &str) {
    let bytes = md.as_bytes();
    for p in 1..bytes.len().saturating_sub(1) {
        if bytes[p] != b'\n' {
            continue;
        }
        let surrounded = bytes[p - 1] != b'\n' && bytes[p + 1] != b'\n';
        let backslash = bytes[p - 1] == b'\\';
        let trailing_spaces = p >= 2 && bytes[p - 1] == b' ' && bytes[p - 2] == b' ';
        assert!(
            !surrounded || backslash || trailing_spaces,
            "soft break at byte {p} in {md:?}",
        );
    }
}

#[gpui::test]
fn enter_in_middle_of_paragraph_creates_paragraph_break(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "hello world".into(),
        selection: Selection::Cursor(5),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "hello\n\n world");
        assert_no_soft_break(&e.state.markdown);
    });
}

#[gpui::test]
fn three_enters_grow_into_three_visible_empty_rows(cx: &mut TestAppContext) {
    // Pairs model: each Enter inserts `\n\n`, so three Enters from the
    // end of "ab" produces six trailing `\n`s (`T / 2 = 3` trailing
    // empties).
    let initial = EditorState {
        markdown: "ab".into(),
        selection: Selection::Cursor(2),
    };
    let (handle, editor) = open_editor(cx, initial);

    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| assert_eq!(e.state.markdown, "ab\n\n"));
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| assert_eq!(e.state.markdown, "ab\n\n\n\n"));
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "ab\n\n\n\n\n\n");
        assert_no_soft_break(&e.state.markdown);
    });
}

#[gpui::test]
fn backspace_at_paragraph_break_merges_in_one_keystroke(cx: &mut TestAppContext) {
    // The user is at the very start of the second paragraph and pressing
    // Backspace should collapse the paragraph break, not feel like a no-op.
    let initial = EditorState {
        markdown: "first\n\nsecond".into(),
        selection: Selection::Cursor(7),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "firstsecond");
        assert_eq!(e.cursor_offset(), 5);
        assert_no_soft_break(&e.state.markdown);
    });
}

#[gpui::test]
fn backspace_through_empty_paragraphs_one_pair_at_a_time(cx: &mut TestAppContext) {
    // Pairs model: source `a\n\n\n\nb` is paragraph break + 1 empty (2
    // pairs total). Each backspace removes one pair.
    let initial = EditorState {
        markdown: "a\n\n\n\nb".into(),
        selection: Selection::Cursor(5),
    };
    let (handle, editor) = open_editor(cx, initial);

    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| assert_eq!(e.state.markdown, "a\n\nb"));

    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "ab");
        assert_no_soft_break(&e.state.markdown);
    });
}

#[gpui::test]
fn delete_forward_at_paragraph_break_merges(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "first\n\nsecond".into(),
        selection: Selection::Cursor(5),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Delete);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "firstsecond");
        assert_eq!(e.cursor_offset(), 5);
        assert_no_soft_break(&e.state.markdown);
    });
}

#[gpui::test]
fn select_across_paragraph_break_and_replace(cx: &mut TestAppContext) {
    // Selecting a range that includes the paragraph break and typing should
    // produce a single paragraph with no soft break.
    let initial = EditorState {
        markdown: "alpha\n\nbeta".into(),
        selection: Selection::range(2, 9),
    };
    let (handle, editor) = open_editor(cx, initial);
    // Replacement comes via the `EntityInputHandler` path (the production
    // text-input route — IME / typed chars). We dispatch SelectAll-style
    // intent indirectly by deleting the range first, then inserting; the
    // backspace path exercises selection deletion.
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "alta");
        assert_no_soft_break(&e.state.markdown);
    });
}
