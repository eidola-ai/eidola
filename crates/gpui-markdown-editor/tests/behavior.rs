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
    Backspace, Delete, DocumentEnd, DocumentStart, Down, End, Enter, Home, Right, SelectAll,
    ShiftRight, Up,
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
fn enter_action_inserts_newline(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "abc".into(),
        selection: Selection::Cursor(2),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "ab\nc");
        assert_eq!(e.cursor_offset(), 3);
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
    let initial = EditorState {
        markdown: "abc\ndef".into(),
        selection: Selection::Cursor(0),
    };
    let (handle, editor) = open_editor(cx, initial);

    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 2));

    dispatch(cx, handle, &editor, Down);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 6));

    dispatch(cx, handle, &editor, Up);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 2));
}

#[gpui::test]
fn home_end_doc_jump(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "abc\ndef".into(),
        selection: Selection::Cursor(5),
    };
    let (handle, editor) = open_editor(cx, initial);

    dispatch(cx, handle, &editor, Home);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 4));
    dispatch(cx, handle, &editor, End);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 7));
    dispatch(cx, handle, &editor, DocumentStart);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 0));
    dispatch(cx, handle, &editor, DocumentEnd);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 7));
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
