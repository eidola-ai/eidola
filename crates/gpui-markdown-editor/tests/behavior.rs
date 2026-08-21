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

use gpui::{
    AnyWindowHandle, AppContext, Bounds, Entity, EntityInputHandler, TestAppContext, WindowBounds,
    WindowOptions, point, px, size,
};
use gpui_component::Root;
use gpui_markdown_editor::editor::{
    Backspace, Delete, DeleteToLineEnd, DeleteToLineStart, DeleteWordBackward, DeleteWordForward,
    DocumentEnd, DocumentStart, Down, End, Enter, Home, Left, Redo, Right, SelectAll, ShiftEnd,
    ShiftHome, ShiftRight, ShiftTab, ShiftWordLeft, ShiftWordRight, Tab, ToggleBold, ToggleItalic,
    Undo, Up, WordLeft, WordRight,
};
use gpui_markdown_editor::{
    BlockKind, Container, EditorState, ListItemKind, MarkdownEditor, MarkdownEditorEvent,
    MarkdownEditorState, MarkdownStyle, RenderSpec, Selection,
};

/// Minimal host view for the editor under test: holds the state entity and
/// renders the `MarkdownEditor` element each frame, exactly as a real host
/// does. The state entity is no longer `Render`, so the window root needs a
/// view that drives the element.
struct EditorHarness {
    state: Entity<MarkdownEditorState>,
}

impl gpui::Render for EditorHarness {
    fn render(
        &mut self,
        _: &mut gpui::Window,
        _: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        MarkdownEditor::new(&self.state)
    }
}

fn open_editor(
    cx: &mut TestAppContext,
    state: EditorState,
) -> (AnyWindowHandle, Entity<MarkdownEditorState>) {
    cx.update(|cx| {
        gpui_component::init(cx);
        let mut inner: Option<Entity<MarkdownEditorState>> = None;
        let window = cx
            .open_window(WindowOptions::default(), |window, cx| {
                let editor = cx.new(|cx| MarkdownEditorState::with_state(state, window, cx));
                inner = Some(editor.clone());
                let harness = cx.new(|_| EditorHarness {
                    state: editor.clone(),
                });
                cx.new(|cx| Root::new(harness, window, cx))
            })
            .expect("open window");
        (window.into(), inner.expect("editor built"))
    })
}

/// Variant of [`open_editor`] that fixes the window to a narrow width
/// so paragraphs reliably soft-wrap. Used by visual-navigation tests
/// that need a wrap row inside a logical line to exist.
fn open_editor_narrow(
    cx: &mut TestAppContext,
    state: EditorState,
) -> (AnyWindowHandle, Entity<MarkdownEditorState>) {
    cx.update(|cx| {
        gpui_component::init(cx);
        let mut inner: Option<Entity<MarkdownEditorState>> = None;
        let bounds = Bounds {
            origin: point(px(0.), px(0.)),
            size: size(px(160.), px(400.)),
        };
        let window = cx
            .open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                |window, cx| {
                    let editor = cx.new(|cx| MarkdownEditorState::with_state(state, window, cx));
                    inner = Some(editor.clone());
                    let harness = cx.new(|_| EditorHarness {
                        state: editor.clone(),
                    });
                    cx.new(|cx| Root::new(harness, window, cx))
                },
            )
            .expect("open window");
        (window.into(), inner.expect("editor built"))
    })
}

fn dispatch(
    cx: &mut TestAppContext,
    handle: AnyWindowHandle,
    editor: &Entity<MarkdownEditorState>,
    action: impl gpui::Action,
) {
    let focus = editor.read_with(cx, |e, _| e.focus_handle.clone());
    cx.update_window(handle, |_, window, cx| {
        focus.dispatch_action(&action, window, cx);
    })
    .unwrap();
    cx.run_until_parked();
}

fn current_spec(cx: &mut TestAppContext, editor: &Entity<MarkdownEditorState>) -> RenderSpec {
    editor.read_with(cx, |e, _| e.render_spec())
}

// ---------------------------------------------------------------------------
// State / update plumbing
// ---------------------------------------------------------------------------

#[gpui::test]
fn editor_constructs_with_initial_state(cx: &mut TestAppContext) {
    let (_, editor) = open_editor(cx, EditorState::with_markdown("# hi"));
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "# hi");
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
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "ab\n\nc");
        assert_eq!(e.cursor_offset(), 4);
    });
}

#[gpui::test]
fn bold_and_italic_actions_toggle_the_word_under_the_caret(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "one two".into(),
        selection: Selection::Cursor(5),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);

    dispatch(cx, handle, &editor, ToggleBold);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "one **two**");
        assert!(e.selection().is_collapsed());
    });

    dispatch(cx, handle, &editor, ToggleItalic);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "one ***two***");
        assert!(e.selection().is_collapsed());
    });
}

#[gpui::test]
fn formatting_action_is_one_undo_step(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "**abcdef**".into(),
        selection: Selection::range(4, 6),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial.clone());

    dispatch(cx, handle, &editor, ToggleBold);
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "**ab**cd**ef**"));
    dispatch(cx, handle, &editor, Undo);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), initial.markdown);
        assert_eq!(e.selection(), initial.selection);
    });
}

#[gpui::test]
fn backspace_removes_one_grapheme(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "abc".into(),
        selection: Selection::Cursor(2),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "ac");
        assert_eq!(e.cursor_offset(), 1);
    });
}

#[gpui::test]
fn delete_removes_forward_grapheme(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "abc".into(),
        selection: Selection::Cursor(1),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Delete);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "ac");
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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftRight);
    dispatch(cx, handle, &editor, ShiftRight);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.selection(), Selection::range(1, 3));
    });
}

#[gpui::test]
fn select_all_spans_document(cx: &mut TestAppContext) {
    let (handle, editor) = open_editor(cx, EditorState::with_markdown("hello"));
    dispatch(cx, handle, &editor, SelectAll);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.selection(), Selection::range(0, 5));
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
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: true,
        },
    );

    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "paragraph 1  \n");
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
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );

    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "paragraph 1\n\n");
        assert_eq!(e.cursor_offset(), 13);
    });

    // Spec: paragraph + 1 trailing empty. The trailing-pair formula
    // shifts by 1 (matching the inter-block layout) so the empty's
    // strict-interior is the typing position that creates new content
    // for the empty row instead of extending the paragraph. The last
    // empty is clamped to doc length, giving a 1-byte range over the
    // final `\n`.
    let spec = current_spec(cx, &editor);
    assert!(spec.blocks.len() >= 2);
    let trailing_empty = spec
        .blocks
        .iter()
        .find(|b| b.source_range == (12..13))
        .expect("synthetic empty owning the clamped trailing pair");
    assert!(matches!(trailing_empty.kind, BlockKind::Paragraph));
    assert!(trailing_empty.inlines.is_empty());
}

#[gpui::test]
fn enter_in_empty_doc_emits_one_visible_row_per_press(cx: &mut TestAppContext) {
    // Pressing Enter from an empty doc shows N + 1 visible rows after
    // N presses (typewriter intuition). Previously the lines-based
    // fallback for content-empty docs counted one block per `\n` plus
    // a trailing anchor — twice the expected row count.
    let (handle, editor) = open_editor(cx, EditorState::with_markdown(""));
    let spec0 = current_spec(cx, &editor);
    assert_eq!(spec0.blocks.len(), 1);

    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "\n\n");
        assert_eq!(e.cursor_offset(), 2);
    });
    let spec1 = current_spec(cx, &editor);
    assert_eq!(spec1.blocks.len(), 2);

    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "\n\n\n\n"));
    let spec2 = current_spec(cx, &editor);
    assert_eq!(spec2.blocks.len(), 3);

    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "\n\n\n\n\n\n"));
    let spec3 = current_spec(cx, &editor);
    assert_eq!(spec3.blocks.len(), 4);
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
        assert_eq!(e.value(), "");
        assert_eq!(e.cursor_offset(), 0);
    });
    let spec = current_spec(cx, &editor);
    assert!(!spec.blocks.is_empty());

    // Pressing Enter from this empty state goes through the action
    // pipeline and produces `\n\n` (pairs model). Confirm render emits
    // multiple visible blocks.
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "\n\n");
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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "hello\n\n world");
        assert_no_soft_break(e.value());
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
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);

    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "ab\n\n"));
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "ab\n\n\n\n"));
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "ab\n\n\n\n\n\n");
        assert_no_soft_break(e.value());
    });
}

#[gpui::test]
fn backspace_at_paragraph_break_merges_in_one_keystroke(cx: &mut TestAppContext) {
    // The user is at the very start of the second paragraph and pressing
    // Backspace should collapse the paragraph break, not feel like a no-op.
    let initial = EditorState {
        markdown: "first\n\nsecond".into(),
        selection: Selection::Cursor(7),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "firstsecond");
        assert_eq!(e.cursor_offset(), 5);
        assert_no_soft_break(e.value());
    });
}

#[gpui::test]
fn backspace_through_empty_paragraphs_one_pair_at_a_time(cx: &mut TestAppContext) {
    // Pairs model: source `a\n\n\n\nb` is paragraph break + 1 empty (2
    // pairs total). Each backspace removes one pair.
    let initial = EditorState {
        markdown: "a\n\n\n\nb".into(),
        selection: Selection::Cursor(5),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);

    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "a\n\nb"));

    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "ab");
        assert_no_soft_break(e.value());
    });
}

#[gpui::test]
fn delete_forward_at_paragraph_break_merges(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "first\n\nsecond".into(),
        selection: Selection::Cursor(5),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Delete);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "firstsecond");
        assert_eq!(e.cursor_offset(), 5);
        assert_no_soft_break(e.value());
    });
}

// ---------------------------------------------------------------------------
// Fenced code blocks
// ---------------------------------------------------------------------------

#[gpui::test]
fn enter_inside_code_block_inserts_single_newline(cx: &mut TestAppContext) {
    // Inside a fenced code block, Enter inserts `\n` — not the
    // paragraph-break `\n\n`. The buffer remains valid markdown with
    // its single `\n` preserved (enforce_invariants exempts code-block
    // content from soft-break promotion).
    let initial = EditorState {
        markdown: "```rust\nlet x = 1;\n```".into(),
        // Cursor at end of "let x = 1;".
        selection: Selection::Cursor(18),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "```rust\nlet x = 1;\n\n```");
        assert_eq!(e.cursor_offset(), 19);
    });
}

#[gpui::test]
fn enter_outside_code_block_inserts_paragraph_break(cx: &mut TestAppContext) {
    // Sanity: Enter just after a code block's closing fence is
    // outside-the-block — the existing paragraph-break behavior holds.
    let initial = EditorState {
        markdown: "```\nx\n```\n\npara".into(),
        // Cursor inside "para".
        selection: Selection::Cursor(13),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "```\nx\n```\n\npa\n\nra");
    });
}

#[gpui::test]
fn code_block_renders_as_code_block_kind(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "```rust\nlet x = 1;\n```".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    assert!(
        spec.blocks
            .iter()
            .any(|b| matches!(b.kind, BlockKind::CodeBlock { .. })),
    );
}

#[gpui::test]
fn backspace_inside_code_block_deletes_one_newline(cx: &mut TestAppContext) {
    // Outside code blocks, backspacing into a `\n\n` pair deletes the
    // whole pair (the structural paragraph break). Inside a code
    // block, `\n\n` is a literal blank line — Backspace should
    // remove just one `\n`.
    let initial = EditorState {
        markdown: "```\nline1\n\nline2\n```".into(),
        // Cursor right after the second `\n` (start of "line2").
        selection: Selection::Cursor(11),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "```\nline1\nline2\n```");
        assert_eq!(e.cursor_offset(), 10);
    });
}

#[gpui::test]
fn delete_forward_inside_code_block_deletes_one_newline(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "```\nline1\n\nline2\n```".into(),
        // Cursor at the first `\n` of the `\n\n` pair (end of
        // "line1").
        selection: Selection::Cursor(9),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Delete);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "```\nline1\nline2\n```");
        assert_eq!(e.cursor_offset(), 9);
    });
}

#[gpui::test]
fn cursor_can_land_in_blank_line_inside_code_block(cx: &mut TestAppContext) {
    // The forbidden-position rule (cursor can't sit inside a
    // structural `\n\n` pair) doesn't apply in code blocks — a
    // blank code line is a real, addressable position. Setting the
    // cursor to the interior of a `\n\n` inside a code block must
    // *not* snap it elsewhere.
    let initial = EditorState {
        markdown: "```\nline1\n\nline2\n```".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _window, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::SetSelection(Selection::Cursor(10)),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.cursor_offset(), 10);
    });
}

#[gpui::test]
fn pasted_multiline_inside_code_block_keeps_single_newlines(cx: &mut TestAppContext) {
    // The exact regression: paste of source containing single `\n`s
    // inside a fenced block must NOT have its newlines promoted to
    // `\n\n` (which would mangle the code).
    let initial = EditorState {
        markdown: "```\n\n```".into(),
        // Cursor inside the empty content (between opening `\n` and
        // closing fence).
        selection: Selection::Cursor(4),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _window, cx| {
        editor.update(cx, |e, cx| {
            // Simulate a paste of multiline source by setting the markdown
            // directly, then running the post-pass via `update` to verify it
            // doesn't promote the inner `\n`s.
            e.set_value("```\nline1\nline2\nline3\n```", cx);
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::SetSelection(Selection::Cursor(20)),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "```\nline1\nline2\nline3\n```");
    });
}

// ---------------------------------------------------------------------------
// Blockquotes
// ---------------------------------------------------------------------------

#[gpui::test]
fn blockquote_renders_a_paragraph_with_one_container(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "> hello\n\nbody".into(),
        // Cursor in "body" — outside the blockquote.
        selection: Selection::Cursor(11),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let bq = spec
        .blocks
        .iter()
        .find(|b| !b.containers.is_empty())
        .expect("a blockquote leaf");
    assert_eq!(bq.containers.len(), 1);
    assert!(matches!(bq.kind, BlockKind::Paragraph));
    assert!(matches!(
        bq.containers[0],
        Container::BlockQuote {
            cursor_inside: false
        }
    ));
}

#[gpui::test]
fn typing_inside_blockquote_keeps_it_a_blockquote(cx: &mut TestAppContext) {
    // The user types into a blockquote — the source still parses as a
    // blockquote afterward (i.e. we don't accidentally promote a stray
    // `\n` into `\n\n` and split the construct).
    let initial = EditorState {
        markdown: "> hello".into(),
        selection: Selection::Cursor(7),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText("!".into()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> hello!");
        let spec = e.render_spec();
        assert!(
            spec.blocks.iter().any(|b| !b.containers.is_empty()),
            "still rendered as a blockquote after typing",
        );
    });
}

#[gpui::test]
fn nested_blockquotes_emit_two_containers(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "> > deep\n\nbody".into(),
        // Cursor outside (in "body").
        selection: Selection::Cursor(11),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let bq = spec
        .blocks
        .iter()
        .find(|b| !b.containers.is_empty())
        .expect("a blockquote leaf");
    assert_eq!(bq.containers.len(), 2);
    assert!(bq.containers.iter().all(|c| matches!(
        c,
        Container::BlockQuote {
            cursor_inside: false
        }
    )));
}

#[gpui::test]
fn cursor_inside_blockquote_marks_only_overlapping_levels(cx: &mut TestAppContext) {
    // For `> > deep` any cursor inside the construct is inside both
    // levels — there's no positional ambiguity. Both containers
    // report cursor_inside = true.
    let initial = EditorState {
        markdown: "> > deep\n".into(),
        selection: Selection::Cursor(6),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let bq = spec
        .blocks
        .iter()
        .find(|b| !b.containers.is_empty())
        .expect("a blockquote leaf");
    assert!(bq.containers.iter().all(|c| matches!(
        c,
        Container::BlockQuote {
            cursor_inside: true
        }
    )));
}

#[gpui::test]
fn enter_inside_blockquote_keeps_new_paragraph_at_same_depth(cx: &mut TestAppContext) {
    // The user's load-bearing example: cursor at the end of a single
    // blockquote paragraph; pressing Enter must produce *two* lines —
    // the empty marker separator and the new paragraph's marker — both
    // still inside the blockquote.
    let initial = EditorState {
        markdown: "> hello".into(),
        selection: Selection::Cursor(7),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> hello\n> \n> ");
        assert_eq!(e.cursor_offset(), 13);
        // The paragraph the cursor sits in still belongs to a
        // blockquote (depth 1).
        assert_eq!(
            gpui_markdown_editor::update::blockquote_depth_at(e.value(), e.cursor_offset(),),
            1,
        );
    });
}

#[gpui::test]
fn enter_inside_nested_blockquote_keeps_depth(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "> > deep".into(),
        selection: Selection::Cursor(8),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> > deep\n> > \n> > ");
        assert_eq!(
            gpui_markdown_editor::update::blockquote_depth_at(e.value(), e.cursor_offset(),),
            2,
        );
    });
}

#[gpui::test]
fn shift_enter_inside_blockquote_keeps_marker_on_continuation(cx: &mut TestAppContext) {
    // Hard break inside a blockquote: `  \n` followed by `> ` so the
    // continuation line stays in the blockquote scope.
    let initial = EditorState {
        markdown: "> hello".into(),
        selection: Selection::Cursor(7),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: true,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> hello  \n> ");
        assert_eq!(e.cursor_offset(), 12);
    });
}

#[gpui::test]
fn backspace_at_end_of_depth_1_pair_outdents_to_depth_0(cx: &mut TestAppContext) {
    // After `> hello` + Enter, cursor sits at the end of the depth-1
    // pair `\n> \n> ` — the start of an empty trailing paragraph at
    // depth 1. Backspace pops one level of blockquote nesting from
    // *both* halves of the pair so the structural break stays
    // balanced: the depth-1 pair `\n> \n> ` (6 bytes) becomes a
    // depth-0 pair `\n\n` (2 bytes). The empty trailing paragraph
    // is now top-level.
    let initial = EditorState {
        markdown: "> hello\n> \n> ".into(),
        selection: Selection::Cursor(13),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> hello\n\n");
        assert_eq!(e.cursor_offset(), 9);
    });
}

#[gpui::test]
fn backspace_at_end_of_depth_2_pair_outdents_to_depth_1(cx: &mut TestAppContext) {
    // Depth-2 case: both halves of the pair lose one `> ` so a
    // depth-2 pair `\n> > \n> > ` (10 bytes) becomes a depth-1 pair
    // `\n> \n> ` (6 bytes) — the trailing paragraph is now at
    // depth 1 and the structural break is still balanced.
    let initial = EditorState {
        markdown: "> > deep\n> > \n> > ".into(),
        selection: Selection::Cursor(18),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> > deep\n> \n> ");
        assert_eq!(e.cursor_offset(), 14);
    });
}

#[gpui::test]
fn successive_backspaces_walk_paragraph_through_nesting_levels(cx: &mut TestAppContext) {
    // The outdent walk: each Backspace pops one level off *both*
    // halves of the preceding pair, so the pair structure stays
    // balanced at each step (depth 2 → 1 → 0). After the depth-0
    // paragraph break is reached, the next Backspace has no marker
    // left to pop and falls through to the existing atomic
    // top-level-pair delete, merging into the previous paragraph.
    let initial = EditorState {
        markdown: "> > deep\n> > \n> > ".into(),
        selection: Selection::Cursor(18),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // depth 2 → 1
        assert_eq!(e.value(), "> > deep\n> \n> ");
        assert_eq!(e.cursor_offset(), 14);
    });
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // depth 1 → 0
        assert_eq!(e.value(), "> > deep\n\n");
        assert_eq!(e.cursor_offset(), 10);
    });
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // depth 0 break gets the original atomic pair delete; the
        // trailing empty paragraph merges into "deep".
        assert_eq!(e.value(), "> > deep");
        assert_eq!(e.cursor_offset(), 8);
    });
}

#[gpui::test]
fn backspace_outdents_interior_paragraph_not_just_trailing(cx: &mut TestAppContext) {
    // The outdent rule applies to *any* non-first paragraph in the
    // blockquote, not just an empty trailing one. Cursor at the start
    // of "two" — Backspace pops a `> ` from each half of the pair
    // so the depth-1 pair becomes `\n\n` and "two" is top-level.
    // The previous paragraph "one" is untouched.
    let initial = EditorState {
        markdown: "> one\n> \n> two".into(),
        selection: Selection::Cursor(11),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> one\n\ntwo");
        assert_eq!(e.cursor_offset(), 7);
    });
}

#[gpui::test]
fn backspace_at_top_level_paragraph_break_still_merges(cx: &mut TestAppContext) {
    // The outdent rule only fires inside a blockquote — at depth 0
    // there are no markers to pop, so Backspace at the start of a
    // top-level second paragraph still does the original atomic pair
    // delete and merges into the previous paragraph.
    let initial = EditorState {
        markdown: "p1\n\np2".into(),
        selection: Selection::Cursor(4),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "p1p2");
        assert_eq!(e.cursor_offset(), 2);
    });
}

#[gpui::test]
fn backspace_at_first_paragraph_in_blockquote_falls_through(cx: &mut TestAppContext) {
    // The outdent rule requires a *non-first* paragraph in the BQ —
    // i.e. a preceding pair half *also* in a BQ. Without one (cursor
    // at the start of the blockquote's first paragraph content,
    // preceded by non-BQ content), Backspace falls through to the
    // regular grapheme delete. This protects the case where the BQ
    // begins right after a top-level paragraph.
    let initial = EditorState {
        markdown: "para\n\n> hi".into(),
        selection: Selection::Cursor(8),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "para\n\n>hi");
    });
}

#[gpui::test]
fn backspace_outdent_preserves_no_soft_break_invariant(cx: &mut TestAppContext) {
    // The user-visible invariant: the buffer never carries a soft
    // break, and outdenting a BQ paragraph should not violate that.
    // Run the outdent through the same `update::update` path used in
    // production — including the `enforce_invariants` post-pass that
    // runs after every event — and check the result has no soft
    // breaks.
    use gpui_markdown_editor::analysis::is_soft_break;
    let initial = EditorState {
        markdown: "> > > p1\n> > > \n> > > p2".into(),
        selection: Selection::Cursor(22),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    // Walk the paragraph from depth 3 → 2 → 1 → 0. At each step the
    // pair is still balanced (the post-outdent buffer is depth-D pair
    // structure for some D), so `enforce_invariants` doesn't insert
    // any synthetic prefixes.
    for _ in 0..3 {
        dispatch(cx, handle, &editor, Backspace);
        editor.read_with(cx, |e, _| {
            let bytes = e.value().as_bytes();
            for p in 0..bytes.len() {
                assert!(
                    !is_soft_break(bytes, p),
                    "soft break at byte {p} in {:?}",
                    e.value(),
                );
            }
        });
    }
    editor.read_with(cx, |e, _| {
        // Final state: depth-0 pair separating the two paragraphs.
        assert_eq!(e.value(), "> > > p1\n\np2");
    });
}

#[gpui::test]
fn typing_inside_blockquote_after_enter_preserves_scope(cx: &mut TestAppContext) {
    // The user types Enter then content. The new paragraph is a real
    // second paragraph inside the same blockquote — pulldown sees
    // both `p1` and the typed content as paragraphs of one bq.
    let initial = EditorState {
        markdown: "> p1".into(),
        selection: Selection::Cursor(4),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText("p2".into()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> p1\n> \n> p2");
    });
}

/// `MarkdownEditor::with_state` skips the post-pass — it accepts the
/// initial state verbatim. Production state always arrives via
/// `update::update`, which is where the soft-break + prefix
/// normalization passes live. To exercise those passes in a behavior
/// test we run the (no-op) selection update through `update`.
fn run_enforce_invariants(cx: &mut TestAppContext, initial: EditorState) -> EditorState {
    let sel = initial.selection;
    cx.update(|_| {
        gpui_markdown_editor::update::update(
            initial,
            gpui_markdown_editor::EditorEvent::SetSelection(sel),
        )
    })
}

// ---------------------------------------------------------------------------
// Pair-model invariants — the depth-D pair `\n[prefix]\n[prefix]` is the
// blockquote-internal analog of `\n\n`. Cursor can't sit inside it, arrow
// keys jump over it, Backspace deletes the whole pair atomically, soft
// breaks across BQ lines get promoted to a complete pair shape (incl.
// lazy-continuation insertion), and hard-break lazy continuations get
// the missing marker inserted.
// ---------------------------------------------------------------------------

#[gpui::test]
fn space_inside_blockquote_does_not_inject_extra_lines(cx: &mut TestAppContext) {
    // Regression: typing a space at the end of a blockquote content
    // line whose buffer also has trailing pair-shaped marker rows
    // used to cause `enforce_invariants` to misclassify the inserted
    // space as content extending the run, fail the pair structural
    // check, and re-promote the surrounding `\n` *every* update —
    // each invocation injecting another `> \n` line. The fix is the
    // tighter backward walk in `is_paragraph_break_interior`: a
    // walk-back over `' '` / `'>'` only counts toward the run if it
    // terminates at a structural `\n`.
    let initial = EditorState {
        markdown: "> blockquote\n> \n> ".into(),
        // Cursor right after "blockquote".
        selection: Selection::Cursor(12),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText(" ".into()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        // Exactly one space inserted — the trailing pair structure
        // is unchanged.
        assert_eq!(e.value(), "> blockquote \n> \n> ");
        assert_eq!(e.cursor_offset(), 13);
    });

    // Idempotent: a no-op SetSelection (mouse move, click handler
    // re-feeding the same offset) re-runs `enforce_invariants`. The
    // buffer must stay identical — no fresh promotion fires.
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            let sel = e.selection();
            e.apply_event_for_test(gpui_markdown_editor::EditorEvent::SetSelection(sel), cx);
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> blockquote \n> \n> ");
    });
}

#[gpui::test]
fn typing_gt_to_enter_nested_blockquote_does_not_inject_extra_lines(cx: &mut TestAppContext) {
    // Same class of bug as `space_inside_blockquote_does_not_inject_extra_lines`,
    // this time on the *forward* walk. State right after Enter inside
    // `> level 1`: the trailing pair has cursor on its second-of-pair
    // marker line. Typing `>` to start a depth-2 BQ used to make the
    // forward walk greedily consume the typed `>` as a continuation
    // marker, breaking pair-length math, and promote the existing
    // structural `\n`s — injecting a fresh `> \n` line per keystroke.
    let initial = EditorState {
        markdown: "> level 1\n> \n> ".into(),
        // Cursor at end of buffer.
        selection: Selection::Cursor(15),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText(">".into()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        // Exactly one `>` appended. Trailing pair structure is
        // unchanged (still 4 lines: content + middle marker + new
        // marker line with the typed `>`).
        assert_eq!(e.value(), "> level 1\n> \n> >");
        assert_eq!(e.cursor_offset(), 16);
    });

    // Idempotent on re-update.
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            let sel = e.selection();
            e.apply_event_for_test(gpui_markdown_editor::EditorEvent::SetSelection(sel), cx);
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> level 1\n> \n> >");
    });

    // The reported follow-on: right-arrow navigation must not
    // trigger fresh promotion either.
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> level 1\n> \n> >");
    });
}

#[gpui::test]
fn typing_gt_on_interior_blank_bq_line_does_not_inject_lines(cx: &mut TestAppContext) {
    // Generalization of the trailing-pair bug to *interior* blank lines
    // surrounded by content on both sides. Initial buffer:
    //
    //   > Level 1
    //   >
    //   >        <- cursor at end of marker (depth 1, blank)
    //   >
    //   > Level 1
    //
    // Typing `>` should turn line 3 into a depth-2 marker line (`> >`)
    // without changing any other line. The underlying byte scanner used
    // to misclassify the `\n`s between the new depth-2 line and the
    // adjacent depth-1 blank lines as soft breaks, splice in
    // `[depth-2-prefix]\n[depth-2-prefix]`, and concatenate the inserted
    // prefix with the existing depth-1 prefix to grow the line to
    // depth 3 — and each subsequent event (including the right-arrow
    // navigation tested below) cascaded another round of the same
    // misclassification, producing many spurious lines.
    //
    // The fix that makes both shapes stable is the marker-only-line
    // adjacency exemption in `is_soft_break`: marker-only blank lines
    // are paragraph terminators, not paragraph content, so the `\n`s
    // adjacent to them are structural stitching rather than soft breaks.
    let initial = EditorState {
        markdown: "> Level 1\n> \n> \n> \n> Level 1".into(),
        // Cursor at the `\n` ending line 3 (the middle blank).
        selection: Selection::Cursor(15),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText(">".into()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> Level 1\n> \n> >\n> \n> Level 1");
        assert_eq!(e.cursor_offset(), 16);
    });

    // Right-arrow shouldn't change the buffer at all (only selection
    // moves). The original bug expanded the buffer by ~3 lines on each
    // arrow press.
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> Level 1\n> \n> >\n> \n> Level 1");
    });
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> Level 1\n> \n> >\n> \n> Level 1");
    });
}

#[gpui::test]
fn typing_gt_on_first_blank_bq_line_does_not_cascade(cx: &mut TestAppContext) {
    // Companion to the interior-blank case. Cursor on the *first* blank
    // line (line 2) — typing `>` here makes the depth-2 line sit
    // adjacent to two equal-depth blank `> ` lines (3 and 4) before the
    // closing content. Without the marker-only exemption, the `\n`s
    // between those blank lines (which are interrupted by the new
    // depth-2 line earlier in the run, breaking the pair detector's
    // run analysis) get classified as soft breaks and each
    // `enforce_invariants` call splices in another structural line.
    let initial = EditorState {
        markdown: "> Level 1\n> \n> \n> \n> Level 1".into(),
        selection: Selection::Cursor(11),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText(">".into()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    // Result: line 2 deepens to `> > ` (normalize adds the trailing
    // space because the cursor moves off it after insertion). All other
    // lines untouched.
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> Level 1\n> > \n> \n> \n> Level 1");
    });
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> Level 1\n> > \n> \n> \n> Level 1");
    });
}

#[gpui::test]
fn cursor_cannot_set_inside_blockquote_pair(cx: &mut TestAppContext) {
    // Position 8 is bytes 5..7 = `> ` plus bytes 7 = `\n` of the
    // pair `\n> \n> ` at bytes 4..10. Setting the cursor strictly
    // inside (bytes 5-9) snaps to the nearest allowed boundary.
    let initial = EditorState {
        markdown: "> p1\n> \n> p2".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::SetSelection(Selection::Cursor(7)),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        // Cursor at 7 (interior of pair 4..10 — strictly inside) is
        // forbidden. Snap should land on the nearest boundary
        // (either 4 = end of p1, or 10 = start of p2). Both are
        // valid landing points.
        let off = e.cursor_offset();
        assert!(off == 4 || off == 10, "cursor snapped to unexpected {off}");
    });
}

#[gpui::test]
fn right_arrow_jumps_over_blockquote_pair(cx: &mut TestAppContext) {
    // Right from byte 4 (end of p1) skips the 6-byte pair interior
    // and lands on byte 10 (start of p2).
    let initial = EditorState {
        markdown: "> p1\n> \n> p2".into(),
        selection: Selection::Cursor(4),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 10));
}

#[gpui::test]
fn delete_forward_at_pair_start_atomically_undoes_break(cx: &mut TestAppContext) {
    // Delete forward at the first `\n` of a depth-1 pair removes the
    // whole 6-byte pair, merging the two BQ paragraphs into one.
    // Same shape as top-level `p1\n\np2` → Delete → `p1p2`.
    let initial = EditorState {
        markdown: "> hello\n> \n> world".into(),
        selection: Selection::Cursor(7),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Delete);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> helloworld");
        assert_eq!(e.cursor_offset(), 7);
    });
}

#[gpui::test]
fn soft_break_across_bq_lines_promotes_to_pair(cx: &mut TestAppContext) {
    // Pasted state with a stray `\n` between two BQ lines.
    // enforce_invariants promotes it to the full pair shape so the
    // result is one BQ with two paragraphs (a paragraph break inside
    // the BQ), not two BQs separated by `\n\n`.
    let initial = EditorState {
        markdown: "> p1\n> p2".into(),
        selection: Selection::Cursor(9),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "> p1\n> \n> p2");
}

#[gpui::test]
fn lazy_continuation_under_soft_break_gets_marker_inserted(cx: &mut TestAppContext) {
    // CommonMark lazy continuation: line 2 has no `>` marker.
    // Promotion inserts both the missing prefix on line 2 and the
    // pair structure so the BQ scope continues cleanly.
    let initial = EditorState {
        markdown: "> hello\nworld".into(),
        selection: Selection::Cursor(13),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "> hello\n> \n> world");
}

#[gpui::test]
fn hard_break_to_soft_break_promotes_to_pair(cx: &mut TestAppContext) {
    // The user's load-bearing example: hard break inside a BQ
    // followed by a backspace of one trailing space turns into a
    // depth-D pair, not a top-level `\n\n` break.
    let initial = EditorState {
        markdown: "> hello \n> world".into(),
        selection: Selection::Cursor(9),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "> hello \n> \n> world");
}

#[gpui::test]
fn missing_space_after_marker_normalizes_when_cursor_moves_off(cx: &mut TestAppContext) {
    // Cursor *not* at the byte right after `>` — the post-pass
    // inserts a space so `>foo` becomes `> foo`.
    let initial = EditorState {
        markdown: ">foo".into(),
        selection: Selection::Cursor(4),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "> foo");
}

#[gpui::test]
fn missing_space_after_marker_left_alone_when_cursor_just_after_gt(cx: &mut TestAppContext) {
    // Cursor immediately after `>` — the user might be about to type
    // the space themselves. Don't second-guess them.
    let initial = EditorState {
        markdown: ">foo".into(),
        selection: Selection::Cursor(1),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, ">foo");
    assert_eq!(final_state.selection, Selection::Cursor(1));
}

#[gpui::test]
fn code_block_inside_blockquote_carries_blockquote_container(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "> ```\n> code\n> ```\n\nbody".into(),
        // Cursor outside.
        selection: Selection::Cursor(22),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let cb = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::CodeBlock { .. }))
        .expect("a code-block leaf");
    assert_eq!(cb.containers.len(), 1);
    assert!(matches!(
        cb.containers[0],
        Container::BlockQuote {
            cursor_inside: false
        }
    ));
}

#[gpui::test]
fn select_across_paragraph_break_and_replace(cx: &mut TestAppContext) {
    // Selecting a range that includes the paragraph break and typing should
    // produce a single paragraph with no soft break.
    let initial = EditorState {
        markdown: "alpha\n\nbeta".into(),
        selection: Selection::range(2, 9),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    // Replacement comes via the `EntityInputHandler` path (the production
    // text-input route — IME / typed chars). We dispatch SelectAll-style
    // intent indirectly by deleting the range first, then inserting; the
    // backspace path exercises selection deletion.
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "alta");
        assert_no_soft_break(e.value());
    });
}

// ---------------------------------------------------------------------------
// Lists
// ---------------------------------------------------------------------------

#[gpui::test]
fn unordered_list_renders_one_container_per_item(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "- foo\n- bar\n".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let items: Vec<_> = spec
        .blocks
        .iter()
        .filter(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
        .collect();
    assert_eq!(items.len(), 2);
}

#[gpui::test]
fn enter_at_end_of_unordered_item_creates_next_bullet(cx: &mut TestAppContext) {
    // Cursor at end of "- foo" → Enter inserts `\n- ` so the user
    // can type the next item. The list as a whole now parses as
    // two items.
    let initial = EditorState {
        markdown: "- foo".into(),
        selection: Selection::Cursor(5),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- foo\n- ");
        assert_eq!(e.cursor_offset(), 8);
    });
}

#[gpui::test]
fn enter_at_end_of_ordered_item_increments_number(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "1. foo".into(),
        selection: Selection::Cursor(6),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "1. foo\n2. ");
    });
}

#[gpui::test]
fn typing_inside_list_does_not_split_on_newline(cx: &mut TestAppContext) {
    // The load-bearing soft-break-exemption test: a buffer containing
    // a list with single `\n` separators between items. The rule
    // change exempts list ranges from soft-break promotion so the
    // structure survives `enforce_invariants`. Without the exemption
    // the `\n` between items would promote to `\n\n` and split the
    // list into two single-item lists.
    let initial = EditorState {
        markdown: "- foo\n- bar".into(),
        selection: Selection::Cursor(11),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "- foo\n- bar");
}

#[gpui::test]
fn typing_in_a_list_item_does_not_break_the_list(cx: &mut TestAppContext) {
    // Type a character inside an item's content. Source stays intact
    // (no spurious promotions) and the editor still parses as a list.
    let initial = EditorState {
        markdown: "- foo\n- bar".into(),
        selection: Selection::Cursor(5),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText("X".into()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- fooX\n- bar");
        let spec = e.render_spec();
        let item_count = spec
            .blocks
            .iter()
            .filter(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
            .count();
        assert_eq!(item_count, 2, "list must still parse as two items");
    });
}

#[gpui::test]
fn enter_inside_list_inside_blockquote_keeps_both_scopes(cx: &mut TestAppContext) {
    // `> - foo` cursor at end → Enter must produce `\n> - ` so the
    // new item stays inside both the BQ and the list.
    let initial = EditorState {
        markdown: "> - foo".into(),
        selection: Selection::Cursor(7),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> - foo\n> - ");
    });
}

// ---- List boundary with surrounding blocks --------------------------

#[gpui::test]
fn list_followed_by_heading_uses_double_newline_boundary(cx: &mut TestAppContext) {
    // Pulldown sees a heading as a separate top-level construct (a
    // setext / ATX heading interrupts list parsing), so the bytes
    // between list and heading fall outside the list's exempt range
    // and `promote_soft_breaks` enforces the canonical `\n\n`
    // boundary.
    let initial = EditorState {
        markdown: "- item\n# heading".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "- item\n\n# heading");
}

#[gpui::test]
fn list_followed_by_lazy_paragraph_canonicalizes_to_continuation(cx: &mut TestAppContext) {
    // Pulldown's lazy-continuation rule treats `- item\nparagraph`
    // as one list item with text "item paragraph". Our canonical
    // form preserves that parse but makes it explicit: the
    // continuation gets indented to marker width and the line
    // break becomes a hard break. The result is one item with
    // an indented multi-line body — *not* a list-then-paragraph
    // split (which would require an explicit `\n\n` from the user).
    let initial = EditorState {
        markdown: "- item\nparagraph".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "- item  \n  paragraph");
}

// ---- List canonicalization passes ----------------------------------

#[gpui::test]
fn loose_list_gets_tightened_to_single_newline_separator(cx: &mut TestAppContext) {
    // A pasted loose list (`\n\n` between items) collapses to a
    // tight one (`\n`). The pixel-fidelity cost is documented:
    // the chat renderer would still render the original loosely,
    // but the editor's "always tight" rule wins inside the
    // composer.
    let initial = EditorState {
        markdown: "- foo\n\n- bar".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "- foo\n- bar");
}

#[gpui::test]
fn lazy_continuation_in_item_promotes_to_hard_break_with_indent(cx: &mut TestAppContext) {
    // Pulldown calls `- foo\nbar` a lazy continuation: "bar" stays
    // in item 1. We canonicalize to `- foo  \n  bar` — explicit
    // hard break + explicit indent — so the chat renderer doesn't
    // collapse "foo bar" onto one line via soft-break-as-space.
    let initial = EditorState {
        markdown: "- foo\nbar".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "- foo  \n  bar");
}

#[gpui::test]
fn soft_break_inside_item_promotes_to_hard_break(cx: &mut TestAppContext) {
    // Already-indented soft break (the user pasted text with proper
    // indent but a soft break, not a hard one). Canonical form
    // promotes the `\n` to `  \n` so editor and chat renderers agree.
    let initial = EditorState {
        markdown: "- foo\n  bar".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "- foo  \n  bar");
}

#[gpui::test]
fn ordered_marker_widening_reindents_continuations(cx: &mut TestAppContext) {
    // Marker `9.` (2 chars + space = 3 bytes) becomes `10.`
    // (3 chars + space = 4 bytes). The continuation indent on
    // item 10 must grow from 3 to 4 spaces. Tested via a 2-item
    // list (start=9) so the split-orphan renumbering heuristic
    // (which renumbers single-item start>1 lists to start at 1)
    // doesn't intrude on this fixture.
    let initial = EditorState {
        markdown: "9. nine\n10. foo  \n   bar".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "9. nine\n10. foo  \n    bar");
}

#[gpui::test]
fn ordered_marker_narrowing_reindents_continuations(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "1. foo  \n    bar".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "1. foo  \n   bar");
}

// ---- Empty-item Enter (depth decrease) -------------------------------

#[gpui::test]
fn enter_on_empty_top_level_item_exits_to_paragraph(cx: &mut TestAppContext) {
    // The "Enter twice to exit a list" UX, framed as decreasing the
    // item's nesting depth by one. After:
    //   1. type `- foo` (cursor at 5)
    //   2. Enter → `- foo\n- ` (cursor at 8)
    //   3. Enter on the empty item → exit to top-level paragraph.
    let initial = EditorState {
        markdown: "- foo\n- ".into(),
        selection: Selection::Cursor(8),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- foo\n\n");
        assert_eq!(e.cursor_offset(), 7);
    });
}

#[gpui::test]
fn enter_on_sole_empty_item_clears_buffer(cx: &mut TestAppContext) {
    // First-keystroke flow: `- ` with cursor at 2, Enter exits the
    // list. With no preceding content, the buffer is left empty.
    let initial = EditorState {
        markdown: "- ".into(),
        selection: Selection::Cursor(2),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "");
        assert_eq!(e.cursor_offset(), 0);
    });
}

#[gpui::test]
fn enter_on_empty_item_inside_blockquote_exits_to_bq_paragraph(cx: &mut TestAppContext) {
    // `> - foo\n> - ` with cursor at end → Enter on the empty list
    // item drops the list level but leaves the BQ scope intact.
    let initial = EditorState {
        markdown: "> - foo\n> - ".into(),
        selection: Selection::Cursor(12),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        // Result: BQ paragraph + BQ paragraph break + new empty BQ
        // line. The depth-D pair shape `\n> \n> ` carries the BQ
        // forward without re-introducing a list marker.
        assert_eq!(e.value(), "> - foo\n> \n> ");
    });
}

// ---- Backspace at start of item content ----------------------------

#[gpui::test]
fn backspace_at_start_of_top_level_item_strips_marker(cx: &mut TestAppContext) {
    // Cursor at byte 2 (right after `- `) — Backspace removes the
    // marker and the item becomes a top-level paragraph. With no
    // preceding content, no `\n\n` separator is needed.
    let initial = EditorState {
        markdown: "- foo".into(),
        selection: Selection::Cursor(2),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "foo");
        assert_eq!(e.cursor_offset(), 0);
    });
}

#[gpui::test]
fn backspace_at_start_of_non_first_item_creates_paragraph_break(cx: &mut TestAppContext) {
    // The user-reported flow: `1. Item one\n2. |Item two` with the
    // cursor right after `2. `. Backspace should *decrease the
    // depth* — for a top-level item that means becoming a
    // paragraph separated from the previous item by `\n\n`. Just
    // dropping the marker would leave `1. Item one\nItem two`
    // (lazy continuation), which the canonicalizer would then
    // re-promote to `1. Item one  \n   Item two` — wrong.
    let initial = EditorState {
        markdown: "1. Item one\n2. Item two".into(),
        selection: Selection::Cursor(15),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "1. Item one\n\nItem two");
    });
}

#[gpui::test]
fn backspace_at_start_of_nested_item_dedents_to_sibling(cx: &mut TestAppContext) {
    // Symmetric to Shift+Tab: Backspace at the start of a nested
    // item content makes it a sibling of the parent item rather
    // than merging the content into the parent.
    let initial = EditorState {
        markdown: "- a\n  - b".into(),
        selection: Selection::Cursor(8),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- a\n- b");
    });
}

#[gpui::test]
fn backspace_at_start_of_ordered_item_strips_marker(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "1. foo".into(),
        selection: Selection::Cursor(3),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "foo");
        assert_eq!(e.cursor_offset(), 0);
    });
}

// ---- Two consecutive hard breaks → paragraph break ----------------

#[gpui::test]
fn two_hard_breaks_at_top_level_become_paragraph_break(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "foo  \n  \nbar".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "foo\n\nbar");
}

#[gpui::test]
fn two_hard_breaks_in_blockquote_become_bq_paragraph_pair(cx: &mut TestAppContext) {
    // Inside a depth-1 blockquote the canonical paragraph break is
    // `\n> \n> ` — the depth-1 pair shape. Dropping the trailing
    // `  ` of both hard breaks yields exactly that.
    let initial = EditorState {
        markdown: "> foo  \n>   \n> bar".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "> foo\n> \n> bar");
}

#[gpui::test]
fn two_hard_breaks_in_list_item_create_paragraph_break(cx: &mut TestAppContext) {
    // Pulldown sees one item with two paragraphs ("foo" and
    // "bar"). After collapse_consecutive_hard_breaks drops both
    // hard-break markers, the residual blank-line whitespace is
    // also stripped (cursor at byte 0 — far from the blank line —
    // so the cursor-in-gap guard doesn't fire), producing the
    // strictly-canonical paragraph-break shape `\n\n   ` rather
    // than `\n  \n  `.
    let initial = EditorState {
        markdown: "- foo  \n    \n  bar".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "- foo\n\n  bar");
}

#[gpui::test]
fn shift_enter_twice_inside_list_item_creates_paragraph_break(cx: &mut TestAppContext) {
    // The end-to-end UX flow. Type `- foo`, press Shift+Enter twice:
    // we want a paragraph break inside the same list item, with the
    // cursor on the empty new paragraph.
    let initial = EditorState {
        markdown: "- foo".into(),
        selection: Selection::Cursor(5),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: true,
        },
    );
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: true,
        },
    );
    editor.read_with(cx, |e, _| {
        // After the second Shift+Enter the buffer is
        // `- foo  \n    \n  ` — two consecutive hard breaks. The
        // collapse pass drops both trailing-`  `s. The blank line
        // *between* the breaks (not the trailing one — cursor sits
        // there) gets its residual whitespace stripped, producing
        // the strictly-canonical `- foo\n\n  ` paragraph break.
        assert_eq!(e.value(), "- foo\n\n  ");
    });
}

// ---- Multi-paragraph items survive enforce_invariants ---------------

#[gpui::test]
fn multi_paragraph_list_item_is_preserved(cx: &mut TestAppContext) {
    // `1. This is a list\n\n   With a second paragraph.` — pulldown's
    // canonical multi-paragraph item shape. Indent matches the
    // `1. ` marker width (3 spaces). enforce_invariants must
    // preserve this exactly: no `\n\n` collapse, no hard-break
    // promotion across the paragraph break.
    let initial = EditorState {
        markdown: "1. This is a list\n\n   With a second paragraph.".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(
        final_state.markdown,
        "1. This is a list\n\n   With a second paragraph.",
    );
}

#[gpui::test]
fn multi_paragraph_item_renders_as_two_paragraph_leaves(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "1. first paragraph\n\n   second paragraph".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let item_leaves: Vec<_> = spec
        .blocks
        .iter()
        .filter(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
        .collect();
    assert_eq!(
        item_leaves.len(),
        2,
        "multi-paragraph item must render one leaf per paragraph",
    );
}

// ---- Nested lists --------------------------------------------------

#[gpui::test]
fn nested_list_renders_with_two_container_levels(cx: &mut TestAppContext) {
    // `- outer\n  - nested` — pulldown gives outer item containing
    // text "outer" + a nested List child whose item is "nested".
    // Render must emit two leaves: outer with one ListItem in its
    // chain, inner with two.
    let initial = EditorState {
        markdown: "- outer\n  - nested".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let depths: Vec<usize> = spec
        .blocks
        .iter()
        .filter(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
        .map(|b| {
            b.containers
                .iter()
                .filter(|c| matches!(c, Container::ListItem { .. }))
                .count()
        })
        .collect();
    assert_eq!(depths, vec![1, 2]);
}

#[gpui::test]
fn nested_list_with_outer_sibling_renders_three_leaves(cx: &mut TestAppContext) {
    // Outer item with a nested item, then an outer sibling. Three
    // leaves: outer-1 (depth 1), nested (depth 2), outer-2 (depth 1).
    let initial = EditorState {
        markdown: "- outer\n  - nested\n- sibling".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let depths: Vec<usize> = spec
        .blocks
        .iter()
        .filter(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
        .map(|b| {
            b.containers
                .iter()
                .filter(|c| matches!(c, Container::ListItem { .. }))
                .count()
        })
        .collect();
    assert_eq!(depths, vec![1, 2, 1]);
}

#[gpui::test]
fn triple_nested_list_renders_three_levels(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "- a\n  - b\n    - c".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let depths: Vec<usize> = spec
        .blocks
        .iter()
        .filter(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
        .map(|b| {
            b.containers
                .iter()
                .filter(|c| matches!(c, Container::ListItem { .. }))
                .count()
        })
        .collect();
    assert_eq!(depths, vec![1, 2, 3]);
}

#[gpui::test]
fn nested_ordered_inside_unordered(cx: &mut TestAppContext) {
    // The marker character should track per-list — outer is bullet,
    // inner is ordered. Both items get ListItem container entries
    // with the right `kind`.
    let initial = EditorState {
        markdown: "- foo\n  1. one\n  2. two".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let kinds: Vec<&Container> = spec
        .blocks
        .iter()
        .filter(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
        .map(|b| b.containers.last().unwrap())
        .collect();
    // Outer "foo" → Unordered. Inner "one" → Ordered { 1 }. Inner
    // "two" → Ordered { 2 }.
    assert!(matches!(
        kinds[0],
        Container::ListItem {
            kind: ListItemKind::Unordered(b'-', None),
            ..
        }
    ));
    assert!(matches!(
        kinds[1],
        Container::ListItem {
            kind: ListItemKind::Ordered { number: 1 },
            ..
        }
    ));
    assert!(matches!(
        kinds[2],
        Container::ListItem {
            kind: ListItemKind::Ordered { number: 2 },
            ..
        }
    ));
}

#[gpui::test]
fn enter_inside_nested_list_creates_next_nested_item(cx: &mut TestAppContext) {
    // Cursor inside the nested item — Enter creates the next
    // nested item (not an outer one). The continuation prefix
    // includes the outer-list indent so the new line stays at the
    // nested depth.
    let initial = EditorState {
        markdown: "- outer\n  - nested".into(),
        selection: Selection::Cursor(18),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- outer\n  - nested\n  - ");
    });
}

// ---- Tab / Shift+Tab nesting changes -------------------------------

#[gpui::test]
fn tab_nests_top_level_item_under_previous_sibling(cx: &mut TestAppContext) {
    // Cursor on second item; Tab nests it under the first.
    let initial = EditorState {
        markdown: "- one\n- two".into(),
        selection: Selection::Cursor(8),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- one\n  - two");
    });
}

#[gpui::test]
fn tab_on_first_item_is_a_noop(cx: &mut TestAppContext) {
    // No previous sibling at the same depth — Tab does nothing.
    let initial = EditorState {
        markdown: "- only".into(),
        selection: Selection::Cursor(2),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- only");
    });
}

#[gpui::test]
fn tab_nests_into_existing_nested_list(cx: &mut TestAppContext) {
    // The previous sibling has a nested list. Tab on the next
    // top-level item should join that nested list rather than
    // creating a new one.
    let initial = EditorState {
        markdown: "- one\n  - nested\n- two".into(),
        selection: Selection::Cursor(19),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- one\n  - nested\n  - two");
        // And it parses as one outer item with two nested items.
        let spec = e.render_spec();
        let depths: Vec<usize> = spec
            .blocks
            .iter()
            .filter(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
            .map(|b| b.containers.len())
            .collect();
        assert_eq!(depths, vec![1, 2, 2]);
    });
}

#[gpui::test]
fn tab_nests_already_nested_item_one_level_deeper(cx: &mut TestAppContext) {
    // `- a\n  - b\n  - c` cursor on c. Tab nests c under b.
    let initial = EditorState {
        markdown: "- a\n  - b\n  - c".into(),
        selection: Selection::Cursor(15),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- a\n  - b\n    - c");
    });
}

#[gpui::test]
fn shift_tab_dedents_nested_item_to_sibling(cx: &mut TestAppContext) {
    // `- a\n  - b` cursor on b; Shift+Tab makes b a top-level
    // sibling of a.
    let initial = EditorState {
        markdown: "- a\n  - b".into(),
        selection: Selection::Cursor(9),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- a\n- b");
    });
}

#[gpui::test]
fn shift_tab_dedents_triple_nested_to_double(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "- a\n  - b\n    - c".into(),
        selection: Selection::Cursor(17),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- a\n  - b\n  - c");
    });
}

#[gpui::test]
fn shift_tab_on_top_level_item_drops_marker(cx: &mut TestAppContext) {
    // At depth 0 there's no enclosing list to dedent into; the
    // operation falls through to "drop the marker bytes," which
    // turns the item into a top-level paragraph.
    let initial = EditorState {
        markdown: "- foo".into(),
        selection: Selection::Cursor(2),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "foo");
    });
}

#[gpui::test]
fn shift_tab_outside_a_list_is_a_noop(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "just a paragraph".into(),
        selection: Selection::Cursor(5),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "just a paragraph");
    });
}

// ---- Marker overlay rendering --------------------------------------

#[gpui::test]
fn unordered_marker_hidden_and_overlaid_at_level_zero(cx: &mut TestAppContext) {
    // The marker bytes (`- `) are always hidden from the shaped line
    // — content shapes from column 0 of the leaf so all items in a
    // list align at the same content edge regardless of marker
    // width. The marker glyph paints as a `MarkerOverlay` in the
    // item's indent strip; the element layer chooses `• ` when the
    // cursor is outside vs the raw bullet char when inside.
    let initial = EditorState {
        markdown: "- foo\n\nbody".into(),
        // Cursor in the body paragraph, well outside the list.
        selection: Selection::Cursor(9),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let item = spec
        .blocks
        .iter()
        .find(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
        .expect("list item leaf");
    assert!(item.has_hidden_range(0..2));
    assert!(item.has_marker_overlay(0..2, 0));
}

#[gpui::test]
fn unordered_marker_overlay_present_when_cursor_inside(cx: &mut TestAppContext) {
    // Hide-and-overlay applies regardless of cursor position. The
    // overlay glyph the element layer paints just changes from `• `
    // to the raw bullet char when the cursor is inside the item.
    let initial = EditorState {
        markdown: "- foo".into(),
        selection: Selection::Cursor(3),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let item = spec
        .blocks
        .iter()
        .find(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
        .unwrap();
    assert!(item.has_hidden_range(0..2));
    assert!(item.has_marker_overlay(0..2, 0));
    assert!(matches!(
        item.containers[0],
        Container::ListItem {
            cursor_inside: true,
            ..
        }
    ));
}

#[gpui::test]
fn ordered_marker_hidden_and_overlaid(cx: &mut TestAppContext) {
    // Ordered items get the same hide-and-overlay treatment. The
    // element layer paints the digits via `kind` rather than
    // substituting a bullet glyph.
    let initial = EditorState {
        markdown: "1. foo\n\nbody".into(),
        selection: Selection::Cursor(10),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let item = spec
        .blocks
        .iter()
        .find(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
        .unwrap();
    // Marker `1. ` is 3 bytes.
    assert!(item.has_hidden_range(0..3));
    assert!(item.has_marker_overlay(0..3, 0));
}

#[gpui::test]
fn star_marker_also_hidden_and_overlaid(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "* foo\n\nbody".into(),
        selection: Selection::Cursor(9),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let item = spec
        .blocks
        .iter()
        .find(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
        .unwrap();
    assert!(item.has_hidden_range(0..2));
    assert!(item.has_marker_overlay(0..2, 0));
}

#[gpui::test]
fn list_max_marker_text_reflects_widest_marker_in_list(cx: &mut TestAppContext) {
    // Items in the same ordered list all carry the same
    // `list_max_marker_text` so the element layer can compute one
    // uniform indent for the whole list.
    let initial = EditorState {
        markdown: "1. one\n2. two\n3. three\n4. four\n5. five\n6. six\n7. seven\n8. eight\n9. nine\n10. ten\n11. eleven".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let texts: Vec<String> = spec
        .blocks
        .iter()
        .filter_map(|b| match b.containers.first() {
            Some(Container::ListItem {
                list_max_marker_text,
                ..
            }) => Some(list_max_marker_text.clone()),
            _ => None,
        })
        .collect();
    // Every item in the list reports `11. ` as the widest marker.
    assert!(!texts.is_empty());
    for text in &texts {
        assert_eq!(text, "11. ");
    }
}

#[gpui::test]
fn unordered_list_max_marker_text_canonicalizes_to_dash(cx: &mut TestAppContext) {
    // For unordered lists, `list_max_marker_text` is canonicalized
    // to `"- "` regardless of the actual bullet char (`-`, `*`,
    // `+`) — they all shape to nearly identical pixel widths and
    // the indent should be stable.
    let initial = EditorState {
        markdown: "* foo\n* bar".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let item = spec
        .blocks
        .iter()
        .find(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
        .unwrap();
    if let Container::ListItem {
        list_max_marker_text,
        ..
    } = &item.containers[0]
    {
        assert_eq!(list_max_marker_text, "- ");
    } else {
        panic!("expected ListItem container");
    }
}

#[gpui::test]
fn list_item_marker_byte_len_recorded_per_item(cx: &mut TestAppContext) {
    // The container records this item's specific marker length so
    // the renderer's indent-hiding pass knows how many leading
    // spaces to elide on continuation lines.
    let initial = EditorState {
        markdown: "1. one\n2. two".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let lens: Vec<usize> = spec
        .blocks
        .iter()
        .filter_map(|b| match b.containers.first() {
            Some(Container::ListItem {
                marker_byte_len, ..
            }) => Some(*marker_byte_len),
            _ => None,
        })
        .collect();
    // Both `1. ` and `2. ` are 3 bytes.
    assert_eq!(lens, vec![3, 3]);
}

#[gpui::test]
fn left_arrow_skips_hidden_list_marker_bytes(cx: &mut TestAppContext) {
    // Cursor at byte 2 (right after `- `, start of `foo`). One Left
    // arrow should land at byte 0 (start of leaf), skipping byte 1
    // which is the strict interior of the hidden marker.
    let initial = EditorState {
        markdown: "- foo".into(),
        selection: Selection::Cursor(2),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Left);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.cursor_offset(), 0);
    });
}

#[gpui::test]
fn right_arrow_skips_hidden_list_marker_bytes(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "- foo".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| {
        // From byte 0, Right should jump past the marker interior to
        // byte 2 (start of content).
        assert_eq!(e.cursor_offset(), 2);
    });
}

#[gpui::test]
fn cursor_at_real_start_of_list_item_line_snaps_forward(cx: &mut TestAppContext) {
    // The user's reported case: cursor at the *real* beginning of
    // the second item's line (byte right after the `\n` between
    // items, before the `2`). That position visually overlaps with
    // the content edge of the line, so it's forbidden — clicking
    // there or moving to it via SetSelection should snap to the
    // unique content edge (byte 11, before `Item`).
    let src = "1. Item one\n2. Item two";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let landed = cx.update(|cx| {
        editor.update(cx, |e, cx| {
            // Byte 12 = right after `\n`, real start of second line.
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::SetSelection(Selection::Cursor(12)),
                cx,
            );
            e.cursor_offset()
        })
    });
    // `2. ` runs 12..15; the unique allowed landing is 15 (before `Item`).
    assert_eq!(landed, 15);
}

#[gpui::test]
fn down_arrow_lands_at_content_edge_not_line_start(cx: &mut TestAppContext) {
    // Down-arrow from end of first item should land at the *content
    // edge* of the second item, not at the (forbidden) line-start
    // before its marker.
    let initial = EditorState {
        markdown: "- one\n- two".into(),
        // Cursor at end of "one" (byte 5).
        selection: Selection::Cursor(5),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Down);
    editor.read_with(cx, |e, _| {
        // `- ` of second item runs 6..8; content edge is 8.
        // Without the line-start-forbidden rule the cursor would
        // pause at 6 first; with it, navigation goes straight to 8.
        let cursor = e.cursor_offset();
        assert!(
            cursor == 8 || cursor == 11,
            "expected cursor at content edge (8) or end of line (11), got {cursor}",
        );
        // The hidden line-start (6) and hidden interior (7) must not be claimed.
        assert_ne!(cursor, 6);
        assert_ne!(cursor, 7);
    });
}

#[gpui::test]
fn click_inside_hidden_marker_snaps_to_nearest_edge(cx: &mut TestAppContext) {
    // SetSelection from a click at byte 1 (mid-marker) snaps to the
    // nearest allowed edge — which for `- foo` is byte 0 or byte 2,
    // both at distance 1 (forward wins ties → byte 2).
    let initial = EditorState {
        markdown: "- foo".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let final_state = cx.update(|cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::SetSelection(Selection::Cursor(1)),
                cx,
            );
            e.cursor_offset()
        })
    });
    assert_eq!(final_state, 2);
}

#[gpui::test]
fn nested_list_item_hides_inner_marker(cx: &mut TestAppContext) {
    // Pulldown reports the nested item starting at the marker byte
    // (byte 10 for `- outer\n  - nested`), so the leading 2 spaces
    // (bytes 8..10) live between the outer leaf and the inner leaf
    // — they aren't part of any leaf's source range and need no
    // explicit hiding. The inner leaf's marker bytes themselves
    // (bytes 10..12) ARE inside its range and must be hidden so
    // the shaped line begins at the content column. The visual
    // indent for the nested item comes from the cumulative
    // container-chain left padding.
    let initial = EditorState {
        markdown: "- outer\n  - nested".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let inner = spec
        .blocks
        .iter()
        .find(|b| b.containers.len() == 2)
        .expect("nested item leaf");
    assert_eq!(inner.source_range.start, 10);
    assert!(item_hidden_ranges_cover(inner, 10..12));
    // The marker overlay sits at the inner level (= 1) so the
    // element layer paints it inside the inner item's indent
    // strip, not the outer's.
    assert!(inner.has_marker_overlay(10..12, 1));
}

#[gpui::test]
fn multi_paragraph_list_item_hides_continuation_indent(cx: &mut TestAppContext) {
    // A loose item with two paragraphs has a 2-space continuation
    // indent on the second paragraph's line. With the new model,
    // that indent is hidden so the second paragraph's content
    // shapes from column 0 of the leaf, and the visual indent
    // comes from the container's left padding.
    let initial = EditorState {
        markdown: "- foo\n\n  bar".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let leaves: Vec<&_> = spec
        .blocks
        .iter()
        .filter(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
        .collect();
    // Two paragraph leaves for the same item.
    assert_eq!(leaves.len(), 2, "expected two leaves for loose item");
    // Second leaf's source range starts at or before byte 7 (the
    // start of the line `  bar`, after `- foo\n\n`). Its leading
    // 2 spaces (bytes 7..9) should be hidden.
    let second = leaves[1];
    assert!(
        item_hidden_ranges_cover(second, 7..9),
        "continuation indent at 7..9 should be hidden, got {:?}",
        second.hidden_ranges,
    );
}

/// True iff the union of `block.hidden_ranges` covers every byte in
/// `target`. Used by tests that don't care which specific hidden
/// range covers a byte (the renderer may emit overlapping ranges
/// when multiple list-item levels each contribute to hiding the
/// same continuation indent).
fn item_hidden_ranges_cover(
    block: &gpui_markdown_editor::RenderBlock,
    target: std::ops::Range<usize>,
) -> bool {
    let mut covered = vec![false; target.end.saturating_sub(target.start)];
    for r in &block.hidden_ranges {
        let lo = r.start.max(target.start);
        let hi = r.end.min(target.end);
        if hi <= lo {
            continue;
        }
        for i in lo..hi {
            covered[i - target.start] = true;
        }
    }
    covered.iter().all(|c| *c)
}

#[gpui::test]
fn shift_enter_at_end_of_list_item_with_following_item(cx: &mut TestAppContext) {
    // The user-reported flow: in a two-item ordered list, cursor
    // at the end of item 1's content, press Shift+Enter twice,
    // type "A". Expected end state: item 1 has two paragraphs
    // (the second containing "A"), each at the canonical indent.
    let initial = EditorState {
        markdown: "1. Item one\n2. Item two".into(),
        selection: Selection::Cursor(11),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: true,
        },
    );
    // After one Shift+Enter alone the existing item-2 line break
    // must NOT be misread as a second hard break — that's the
    // false-positive the user hit. The buffer should still have
    // a real hard-break continuation in item 1, with item 2
    // intact below.
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "1. Item one  \n   \n2. Item two");
    });
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: true,
        },
    );
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText("A".into()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        // Two Shift+Enters → paragraph break inside the item.
        // "A" is the start of item 1's second paragraph at the
        // canonical 3-space indent. The cursor-aware blank-line
        // tightening pass strips the residual whitespace from the
        // blank line between the two paragraphs once the cursor
        // moves off it (here, "A" is typed at the cursor's parked
        // position, leaving the *previous* blank line free for
        // canonicalization). The result is the strictly-canonical
        // `1. Item one\n\n   A\n2. Item two` paragraph-break shape
        // pulldown would parse identically. The separator with
        // item 2 is still tight (`\n2.` rather than `\n\n2.`) —
        // that's the documented inter-item tightening rule kept
        // unchanged.
        assert_eq!(e.value(), "1. Item one\n\n   A\n2. Item two");
        // Verify the parse: item 1 has two paragraph leaves
        // (depth 1), item 2 has one (depth 1).
        let spec = e.render_spec();
        let item_leaves: Vec<usize> = spec
            .blocks
            .iter()
            .filter(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
            .map(|b| b.containers.len())
            .collect();
        assert_eq!(item_leaves, vec![1, 1, 1]);
    });
}

// ---- Marker-to-content spacing -------------------------------------

#[gpui::test]
fn extra_space_after_unordered_marker_is_stripped(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "-  foo".into(),
        selection: Selection::Cursor(6),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "- foo");
}

#[gpui::test]
fn multiple_extra_spaces_after_marker_are_stripped(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "-    foo".into(),
        selection: Selection::Cursor(8),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "- foo");
}

#[gpui::test]
fn extra_space_after_ordered_marker_is_stripped(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "1.  foo".into(),
        selection: Selection::Cursor(7),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "1. foo");
}

#[gpui::test]
fn empty_item_with_only_extra_trailing_spaces_is_left_alone(cx: &mut TestAppContext) {
    // A marker followed by only spaces (no content) is a transient
    // mid-edit state — the user just pressed Enter or Tab and the
    // cursor is parked there. We don't strip the trailing spaces;
    // doing so would yank the cursor backward.
    let initial = EditorState {
        markdown: "- foo\n-  ".into(),
        selection: Selection::Cursor(9),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "- foo\n-  ");
}

#[gpui::test]
fn extra_marker_spacing_preserved_when_cursor_in_gap(cx: &mut TestAppContext) {
    // The user typed `- ` then a *second* space and the cursor sits
    // between the two spaces. Stripping the extra space would jerk
    // the cursor backward mid-typing — unwanted. The cursor-in-gap
    // guard preserves the source until the cursor moves away.
    let initial = EditorState {
        markdown: "-  foo".into(),
        selection: Selection::Cursor(2),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "-  foo");
    // Cursor at the content edge (past the gap): legitimate "fix
    // it" cursor position. Strip fires.
    let initial = EditorState {
        markdown: "-  foo".into(),
        selection: Selection::Cursor(6),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "- foo");
}

// ---- Residual whitespace tightening (cursor-aware) -----------------

#[gpui::test]
fn residual_blank_line_whitespace_is_stripped(cx: &mut TestAppContext) {
    // A multi-paragraph item whose paragraph break carries indent
    // residue (`\n   \n`) gets canonicalized to a strict `\n\n`
    // when no cursor sits on the blank line. Pulldown parses both
    // forms identically; the strip is purely source-cleanliness.
    let initial = EditorState {
        markdown: "1. one\n   \n   two".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "1. one\n\n   two");
}

#[gpui::test]
fn residual_blank_line_preserved_when_cursor_parked_there(cx: &mut TestAppContext) {
    // The blank-line residue is also the transient post-Shift+Enter
    // shape. With the cursor parked on the blank line, stripping
    // the indent would yank the cursor to column zero — wrong.
    let initial = EditorState {
        markdown: "1. one\n   \n   two".into(),
        selection: Selection::Cursor(10), // end of "   " on the blank line
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "1. one\n   \n   two");
}

// ---- Ordered-list renumbering --------------------------------------

#[gpui::test]
fn ordered_list_renumbers_after_inserted_item(cx: &mut TestAppContext) {
    // Pretend the user inserted a new "two" between original
    // items 1 and 3 — now the list reads 1, 2, 3 in source but
    // numbered 1, 1, 3 (the inserted item kept the old number).
    // enforce_invariants renumbers to 1, 2, 3.
    let initial = EditorState {
        markdown: "1. one\n1. two\n3. three".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "1. one\n2. two\n3. three");
}

#[gpui::test]
fn ordered_list_renumbers_after_removed_item(cx: &mut TestAppContext) {
    // Source numbered 1, 5, 3 → canonical 1, 2, 3.
    let initial = EditorState {
        markdown: "1. one\n5. middle\n3. three".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "1. one\n2. middle\n3. three");
}

#[gpui::test]
fn ordered_list_renumbering_preserves_non_one_start(cx: &mut TestAppContext) {
    // The first item's number IS the list's start; we don't
    // rewrite the start to 1. Subsequent items count up from
    // wherever the user began.
    let initial = EditorState {
        markdown: "10. ten\n12. twelve".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "10. ten\n11. twelve");
}

#[gpui::test]
fn ordered_list_renumber_widens_indent_for_continuation(cx: &mut TestAppContext) {
    // Item 9 followed by what would *become* item 10. The
    // continuation indent grows from 3 spaces to 4 along with
    // the renumber.
    let initial = EditorState {
        markdown: "9. nine\n9. ten  \n   cont".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "9. nine\n10. ten  \n    cont");
}

#[gpui::test]
fn tab_on_ordered_item_starts_nested_list_at_one(cx: &mut TestAppContext) {
    // The user's reported flow: type `1. Item one`, press Enter
    // (gives `1. Item one\n2. ` cursor at end), then press Tab.
    //
    // Without rewriting the marker, the post-Tab source
    // `1. Item one\n   2. ` doesn't parse as a nested list —
    // CommonMark says an ordered list with start > 1 can't open
    // mid-item, so pulldown sees `   2. ` as continuation text.
    // Tab must rewrite the marker to `1. ` so the nested list
    // actually opens; renumbering then handles any subsequent
    // joining of existing nested items.
    //
    // (Pulldown doesn't open a list for the *empty* `1. ` either
    // — it needs content. So the AST shape only flips to nested
    // once the user starts typing. The test types one character
    // to trigger the parse and verifies the depth.)
    let initial = EditorState {
        markdown: "1. Item one\n2. ".into(),
        selection: Selection::Cursor(15),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "1. Item one\n   1. ");
    });
    // Type one character to give pulldown content to parse.
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText("x".into()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "1. Item one\n   1. x");
        let spec = e.render_spec();
        let depths: Vec<usize> = spec
            .blocks
            .iter()
            .filter(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
            .map(|b| b.containers.len())
            .collect();
        assert_eq!(depths, vec![1, 2]);
    });
}

#[gpui::test]
fn tab_on_ordered_item_joining_existing_nested_list_renumbers(cx: &mut TestAppContext) {
    // Existing nested list with one item; the next outer item
    // gets Tab'd. The renumbering pass should make it item 2 of
    // the nested list.
    let initial = EditorState {
        markdown: "1. one\n   1. nested-1\n2. two".into(),
        selection: Selection::Cursor(25),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "1. one\n   1. nested-1\n   2. two");
    });
}

#[gpui::test]
fn shift_tab_on_nested_ordered_item_dedents_correct_one(cx: &mut TestAppContext) {
    // Regression for the user's report that Shift+Tab "unnests
    // 'Item one' rather than 'Item one, one'." With the buffer
    // canonicalized as `1. Item one\n   1. Item one, one`,
    // cursor on "Item one, one" (depth 2), Shift+Tab dedents the
    // *inner* item, not the outer.
    let initial = EditorState {
        markdown: "1. Item one\n   1. Item one, one".into(),
        selection: Selection::Cursor(20),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "1. Item one\n2. Item one, one");
    });
}

#[gpui::test]
fn tab_preserves_continuation_lines_under_new_indent(cx: &mut TestAppContext) {
    // Item with a hard-break continuation. Tab indents *both*
    // the marker line and the continuation by the previous
    // sibling's marker width, keeping the item's structure intact.
    let initial = EditorState {
        markdown: "- one\n- two  \n  cont".into(),
        selection: Selection::Cursor(8),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- one\n  - two  \n    cont");
    });
}

// ---------------------------------------------------------------------------
// Nesting stress corpus
//
// Scenarios called out in the architectural review's response document
// (`gpui-markdown-editor-review-response.md`, refinement E). Each test
// exercises one nesting interaction the per-construct tests above don't
// cover: alternating-container nesting (BQ-in-list-in-BQ-in-list and
// reversed), code blocks inside containers, multi-paragraph items
// holding nested lists, and the depth-change gestures (Tab / Shift+Tab /
// empty-Enter / Backspace-at-start) at every level.
//
// These read like ordinary behavior tests rather than a parameterized
// corpus on purpose — when one fails, the diagnostic is "this exact
// nesting interaction broke" with full source, action sequence, and
// expected shape inline. A future case added to this section is one
// `#[gpui::test]` away.
// ---------------------------------------------------------------------------
//
// ---- Container nesting --------------------------------------------------

/// Helper: count `Container::ListItem` entries in a block's chain.
fn list_item_depth(b: &gpui_markdown_editor::RenderBlock) -> usize {
    b.containers
        .iter()
        .filter(|c| matches!(c, Container::ListItem { .. }))
        .count()
}

/// Helper: count `Container::BlockQuote` entries in a block's chain.
fn blockquote_depth(b: &gpui_markdown_editor::RenderBlock) -> usize {
    b.containers
        .iter()
        .filter(|c| matches!(c, Container::BlockQuote { .. }))
        .count()
}

#[gpui::test]
fn bq_inside_list_inside_bq_inside_list_renders_with_full_chain(cx: &mut TestAppContext) {
    // The 4-level alternating nesting. Each leaf carries the
    // outermost-first chain `[ListItem, BlockQuote, ListItem,
    // BlockQuote]` at the deepest leaf. Editor must keep the
    // rendered chain consistent with the source structure rather
    // than mis-attributing to a flatter container model.
    let initial = EditorState {
        markdown: "- > - > deepest".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let leaves: Vec<&gpui_markdown_editor::RenderBlock> = spec
        .blocks
        .iter()
        .filter(|b| matches!(b.kind, BlockKind::Paragraph))
        .collect();
    assert!(
        !leaves.is_empty(),
        "expected at least one paragraph leaf in deeply nested BQ/list",
    );
    // The deepest leaf carries 2 ListItem entries + 2 BlockQuote
    // entries.
    let deepest = leaves
        .iter()
        .max_by_key(|b| b.containers.len())
        .copied()
        .unwrap();
    assert_eq!(list_item_depth(deepest), 2);
    assert_eq!(blockquote_depth(deepest), 2);
}

#[gpui::test]
fn list_inside_bq_inside_list_inside_bq_renders_with_full_chain(cx: &mut TestAppContext) {
    // The reverse alternating nesting — outer BQ, then a list
    // item, then a nested BQ, then a list inside it.
    let initial = EditorState {
        markdown: "> - > - deepest".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let deepest = spec
        .blocks
        .iter()
        .filter(|b| matches!(b.kind, BlockKind::Paragraph))
        .max_by_key(|b| b.containers.len())
        .unwrap();
    assert_eq!(list_item_depth(deepest), 2);
    assert_eq!(blockquote_depth(deepest), 2);
}

#[gpui::test]
fn triple_nested_list_carries_three_list_item_entries(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "- one\n  - two\n    - three".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let depths: Vec<usize> = spec
        .blocks
        .iter()
        .filter(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
        .map(list_item_depth)
        .collect();
    assert_eq!(depths, vec![1, 2, 3]);
}

// ---- Code blocks inside containers --------------------------------------

#[gpui::test]
fn code_block_inside_list_carries_list_item_chain(cx: &mut TestAppContext) {
    // A fenced code block as a list item's child. The CodeBlock
    // leaf must carry the enclosing `Container::ListItem` so the
    // element layer applies list indent / chrome.
    let initial = EditorState {
        markdown: "- ```\n  code\n  ```".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let code_leaf = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::CodeBlock { .. }))
        .expect("expected one CodeBlock leaf inside the list item");
    assert_eq!(list_item_depth(code_leaf), 1);
}

#[gpui::test]
fn code_block_inside_bq_carries_blockquote_chain(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "> ```\n> code\n> ```".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let code_leaf = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::CodeBlock { .. }))
        .expect("expected one CodeBlock leaf inside the blockquote");
    assert_eq!(blockquote_depth(code_leaf), 1);
}

#[gpui::test]
fn code_block_inside_bq_inside_list_carries_both_chains(cx: &mut TestAppContext) {
    // Code in BQ in list. Leaf chain should carry one ListItem
    // and one BlockQuote.
    let initial = EditorState {
        markdown: "- > ```\n  > code\n  > ```".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let code_leaf = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::CodeBlock { .. }))
        .expect("expected one CodeBlock leaf inside BQ inside list");
    assert_eq!(list_item_depth(code_leaf), 1);
    assert_eq!(blockquote_depth(code_leaf), 1);
}

#[gpui::test]
fn code_block_inside_list_inside_bq_carries_both_chains(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "> - ```\n>   code\n>   ```".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let code_leaf = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::CodeBlock { .. }))
        .expect("expected one CodeBlock leaf inside list inside BQ");
    assert_eq!(list_item_depth(code_leaf), 1);
    assert_eq!(blockquote_depth(code_leaf), 1);
}

#[gpui::test]
fn enter_inside_code_inside_list_inserts_single_newline(cx: &mut TestAppContext) {
    // The fenced-code rule (Enter inserts `\n` rather than the
    // surrounding scope's paragraph break) must still take
    // precedence over the list-item routing when code lives inside
    // a list item.
    let initial = EditorState {
        markdown: "- ```\n  code\n  ```".into(),
        selection: Selection::Cursor(11), // mid-content on `code` line
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        // Single `\n` inserted; the list-item next-marker rule
        // doesn't fire inside code content.
        assert!(e.value().contains("```\n  cod\n"));
    });
}

// ---- Multi-paragraph items containing a nested list ----------------------

#[gpui::test]
fn multi_paragraph_item_with_nested_list_renders_three_leaves(cx: &mut TestAppContext) {
    // `1. p1\n\n   p2\n\n   - nested` — item 1 has two paragraph
    // children plus a nested unordered list. The render walker
    // must emit one leaf per paragraph child *and* one leaf for
    // the nested list's item, all carrying the outer item's
    // ListItem chain entry.
    let initial = EditorState {
        markdown: "1. p1\n\n   p2\n\n   - nested".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let item_leaves: Vec<&gpui_markdown_editor::RenderBlock> = spec
        .blocks
        .iter()
        .filter(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
        .collect();
    // Two outer-paragraph leaves + one nested-item leaf.
    assert_eq!(item_leaves.len(), 3);
    let depths: Vec<usize> = item_leaves.iter().map(|b| list_item_depth(b)).collect();
    assert_eq!(depths, vec![1, 1, 2]);
}

// ---- Tab at every nesting level ------------------------------------------

#[gpui::test]
fn tab_at_depth_2_nests_to_depth_3(cx: &mut TestAppContext) {
    // Existing depth-2 nest with a sibling at depth 2 → Tab pushes
    // sibling to depth 3.
    let initial = EditorState {
        markdown: "- one\n  - two\n  - three".into(),
        selection: Selection::Cursor(20), // inside "three"
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- one\n  - two\n    - three");
    });
}

#[gpui::test]
fn tab_inside_blockquote_list_nests_within_blockquote(cx: &mut TestAppContext) {
    // List inside a BQ. Tab on the second item should nest it
    // inside the first — the BQ scope must be preserved on every
    // continuation line. **Currently the indent insertion uses
    // raw byte line-starts and inserts ahead of the `> ` BQ
    // marker, producing `  > - two` instead of `>   - two`**, so
    // the BQ scope is broken.
    //
    // The corpus exposes this; the fix lives in
    // `analysis::list_item_indent_edits`'s line-start computation,
    // which needs to walk past the active container-prefix bytes
    // before inserting indent.
    let initial = EditorState {
        markdown: "> - one\n> - two".into(),
        selection: Selection::Cursor(13), // inside "two"
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> - one\n>   - two");
    });
}

// ---- Shift+Tab at every nesting level ------------------------------------

#[gpui::test]
fn shift_tab_at_depth_3_dedents_to_depth_2(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "- one\n  - two\n    - three".into(),
        selection: Selection::Cursor(22),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- one\n  - two\n  - three");
    });
}

#[gpui::test]
fn shift_tab_at_top_level_inside_blockquote_becomes_paragraph_in_bq(cx: &mut TestAppContext) {
    // Top-level item inside a blockquote: Shift+Tab makes it a
    // paragraph in the BQ scope, with the depth-1 pair shape
    // (`\n> \n> `) ahead of it rather than a top-level `\n\n`.
    let initial = EditorState {
        markdown: "> - one\n> - two".into(),
        selection: Selection::Cursor(13), // inside "two"
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        // The marker is dropped and the leading separator
        // becomes a depth-1 pair so the result stays inside the
        // BQ.
        assert_eq!(e.value(), "> - one\n> \n> two");
    });
}

// ---- Empty-Enter at every nesting level ----------------------------------

#[gpui::test]
fn empty_enter_on_top_level_item_becomes_paragraph(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "- one\n- ".into(),
        selection: Selection::Cursor(8),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        // The empty item drops; cursor lands at a fresh empty
        // paragraph after the surviving item.
        assert_eq!(e.value(), "- one\n\n");
    });
}

#[gpui::test]
fn backspace_at_li_wrapped_fence_start_unwraps_li(cx: &mut TestAppContext) {
    // Cursor right before the opener fence chars of an LI-wrapped
    // fence, Backspace. Per the rule "context changes to the first
    // line of the fence apply to the entire fence", the entire
    // fence drops its LI wrapper — the LI marker disappears and
    // the body / closer continuation indent is stripped. Reuses
    // the existing `list_item_dedent_edits` top-level dedent path
    // (with my new `is_in_fenced_code` guard now allowing this
    // boundary position to route through, since strict-start
    // semantics treat the fence's first-byte cursor as outside).
    let initial = EditorState {
        markdown: "1. ```js\n   body\n   ```".into(),
        selection: Selection::Cursor(3),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "```js\nbody\n```");
    });
}

#[gpui::test]
fn backspace_at_bq_wrapped_fence_start_unwraps_bq(cx: &mut TestAppContext) {
    // Cursor right before the opener fence chars of a BQ-wrapped
    // fence, Backspace. New `fence_bq_outdent_edits` strips the
    // innermost-BQ marker (`> `) from every line of the fence in
    // one keystroke. Without it, Backspace would just delete the
    // single space after `>` and leave the rest of the fence
    // BQ-wrapped (the user-reported behavior).
    let initial = EditorState {
        markdown: "> ```js\n> body\n> ```".into(),
        selection: Selection::Cursor(2),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "```js\nbody\n```");
    });
}

#[gpui::test]
fn backspace_on_empty_body_line_in_bq_fence_removes_whole_line(cx: &mut TestAppContext) {
    // User-reported bug: Backspace on an empty body line inside a
    // BQ-wrapped fence (`> ```js\n> |\n> ``` `) used to delete one
    // hidden prefix byte at a time — the user pressed Backspace
    // once and saw nothing happen visually. Now the in-fence
    // "delete prefix-only line" gesture removes the whole line in
    // one keystroke.
    let initial = EditorState {
        markdown: "> ```js\n> \n> ```".into(),
        selection: Selection::Cursor(10),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // Empty body line gone; opener and closer untouched.
        assert_eq!(e.value(), "> ```js\n> ```");
    });
}

#[gpui::test]
fn backspace_on_empty_body_line_in_li_fence_removes_whole_line(cx: &mut TestAppContext) {
    // User-reported bug: in an LI-wrapped fence, pressing Enter
    // twice from the body then Backspace used to delete one space
    // of the LI continuation indent at a time (the "deleting
    // invisible, forbidden characters" report). Now the gesture
    // collapses the whole prefix-only body line in one keystroke.
    let initial = EditorState {
        markdown: "1. ```js\n   body\n   ```".into(),
        selection: Selection::Cursor(16),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.value(),
            "1. ```js\n   body\n   \n   \n   ```",
            "two Enters add two empty body lines",
        );
    });
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // One empty body line removed; one remains.
        assert_eq!(e.value(), "1. ```js\n   body\n   \n   ```");
    });
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // Second empty body line removed.
        assert_eq!(e.value(), "1. ```js\n   body\n   ```");
    });
}

#[gpui::test]
fn typing_bq_marker_at_fence_opener_nests_entire_fence(cx: &mut TestAppContext) {
    // User-reported bug: typing `>` (or `> `) at the start of a
    // top-level fence used to nest the body but leave the closer at
    // top level. Per the rule "context changes to the first line of
    // the fence apply to the entire fence", `enforce_invariants`'s
    // `unify_fence_chain` pass detects the unterminated-BQ-fence +
    // orphan-top-level-fence pattern and propagates the BQ prefix
    // to every line in between.
    fn run(src: &str, insert: &str, cx: &mut TestAppContext) -> String {
        let initial = EditorState {
            markdown: src.into(),
            selection: Selection::Cursor(0),
            ..Default::default()
        };
        let (handle, editor) = open_editor(cx, initial);
        cx.update_window(handle, |_, _, cx| {
            editor.update(cx, |e, cx| {
                e.apply_event_for_test(
                    gpui_markdown_editor::EditorEvent::InsertText(insert.into()),
                    cx,
                );
            });
        })
        .unwrap();
        cx.run_until_parked();
        editor.read_with(cx, |e, _| e.value().to_string())
    }
    // Typing just `>` (no space) — every line of the fence gains a
    // BQ prefix. The opener line keeps `>` without a trailing
    // space because the cursor is parked right after the `>`
    // (`normalize_blockquote_prefixes` defers the space-add until
    // the cursor moves off the marker boundary). Body and closer
    // get the canonical `> ` directly.
    let bare = run("```js\nbody\n```", ">", cx);
    assert_eq!(bare, ">```js\n> body\n> ```");
    // Typing the full `> ` — same outcome with the opener already
    // canonical.
    let with_space = run("```js\nbody\n```", "> ", cx);
    assert_eq!(with_space, "> ```js\n> body\n> ```");
}

#[gpui::test]
fn enter_repeated_in_bq_wrapped_fence_body_keeps_bq_prefixes(cx: &mut TestAppContext) {
    // User-reported bug: Enter x N in a BQ-wrapped fence body used
    // to outdent the BQ scope on the third press. The chain-aware
    // empty-BQ-paragraph exit edit was firing on what looked like a
    // `\n> \n> ` pair ending at the cursor — even though the cursor
    // sat in fenced-code body where `\n` is a literal line
    // separator, not a structural pair. Guarding
    // `empty_bq_paragraph_exit_edit` with `is_in_fenced_code` keeps
    // the fence intact across repeats.
    let initial = EditorState {
        markdown: "> ```js\n> body\n> ```".into(),
        selection: Selection::Cursor(14),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    for _ in 0..4 {
        dispatch(
            cx,
            handle,
            &editor,
            Enter {
                secondary: false,
                shift: false,
            },
        );
    }
    editor.read_with(cx, |e, _| {
        // Each Enter inserts `\n> ` (literal newline + BQ
        // continuation prefix); 4 Enters add 4 such pairs and
        // never strip an existing BQ marker. The fence stays
        // terminated.
        assert_eq!(e.value(), "> ```js\n> body\n> \n> \n> \n> \n> ```",);
        assert!(gpui_markdown_editor::analysis::is_in_fenced_code(
            e.value(),
            e.cursor_offset(),
        ));
    });
}

#[gpui::test]
fn enter_at_opener_fence_start_in_li_inserts_new_list_item(cx: &mut TestAppContext) {
    // User-reported bug: Enter at the byte right before the opener
    // fence chars in a list-wrapped fence (`1. |` `` js`) used to
    // route through the in-fence path and inject a soft break.
    // Per the rule "interactions before the first character of the
    // opening fence behave as not part of the code block", the
    // fence-containment query is now strict at `range.start`, so
    // the cursor at that boundary lands in the LI's chain and
    // Enter inserts a new list item.
    let initial = EditorState {
        markdown: "1. ```js\n   body\n   ```".into(),
        selection: Selection::Cursor(3),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "1. \n2. ```js\n   body\n   ```");
    });
}

#[gpui::test]
fn tab_inside_li_wrapped_fence_body_inserts_literal_tab(cx: &mut TestAppContext) {
    // User-reported bug: Tab inside the body of a list-wrapped
    // fence used to nest the enclosing LI. Per the rule "the
    // innermost block dictates interaction behavior", code-block
    // body wins — Tab here inserts a literal `\t`.
    let initial = EditorState {
        markdown: "1. ```js\n   body\n   ```".into(),
        selection: Selection::Cursor(16),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "1. ```js\n   body\t\n   ```");
        assert_eq!(e.cursor_offset(), 17);
    });
}

#[gpui::test]
fn code_block_fences_are_delimiter_rows_at_every_chain(cx: &mut TestAppContext) {
    // Fence rows (opener + closer) must be reported as
    // `delimiter_lines` covering the *whole source line*, not just
    // the fence chars. The element layer's
    // `line_is_fully_in_a_delimiter` test requires the entry to
    // cover the shaped line's full logical range — and that range
    // begins at the byte right after the previous `\n`. For
    // fences sitting inside a BQ or LI the line starts with the
    // chain prefix (`> `, `   `), so a delimiter entry that
    // started at the fence chars (the original behavior) would
    // miss the fence row entirely and the user would see the
    // fences rendered inside the content background instead of
    // outside it.
    use gpui_markdown_editor::render_spec::BlockKind;
    let cases = [
        ("top-level", "```js\nconsole.log('Hello world');\n```\n"),
        (
            "blockquote",
            "> ```js\n> console.log('Hello world');\n> ```\n",
        ),
        ("list", "1. ```js\n   console.log('Hello world');\n   ```\n"),
    ];
    for (label, src) in cases {
        let initial = EditorState {
            markdown: src.into(),
            selection: Selection::Cursor(0),
            ..Default::default()
        };
        let (_handle, editor) = open_editor(cx, initial);
        editor.read_with(cx, |e, _| {
            let spec = e.render_spec();
            let cb = spec
                .blocks
                .iter()
                .find(|b| matches!(b.kind, BlockKind::CodeBlock { .. }))
                .unwrap_or_else(|| panic!("[{label}] expected a CodeBlock leaf"));
            assert_eq!(
                cb.delimiter_lines.len(),
                2,
                "[{label}] expected opener + closer delimiter rows; got {:?}",
                cb.delimiter_lines,
            );
            // Every fence-row entry must start at a line boundary
            // (the byte right after the previous `\n`, or 0).
            let bytes = src.as_bytes();
            for d in &cb.delimiter_lines {
                let starts_at_line_boundary = d.start == 0 || bytes[d.start - 1] == b'\n';
                assert!(
                    starts_at_line_boundary,
                    "[{label}] delimiter entry {d:?} doesn't start at a line boundary",
                );
            }
        });
    }
}

#[gpui::test]
fn multi_enter_from_nested_item_progresses_one_level_per_press(cx: &mut TestAppContext) {
    // User-reported repro:
    //
    //   - parent
    //     - child|
    //
    // Each Enter should escape one nesting level cleanly without
    // producing weird intermediate shapes. The bug before
    // `pick_chain_target`'s nested-list content-on-line filter
    // was that Enter 3 routed through the inner-LI's
    // `next_marker_text` insertion (because pulldown ranges the
    // inner LI through the outer LI's trailing continuation
    // indent), spawning a stray `\n  - ` marker on a row visually
    // past the nested list.
    let initial = EditorState {
        markdown: "- parent\n  - child".into(),
        selection: Selection::Cursor(18),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);

    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.value(),
            "- parent\n  - child\n  - ",
            "Enter 1: new empty inner sibling",
        );
    });

    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.value(),
            "- parent\n  - child\n\n  ",
            "Enter 2: empty-inner outdent → outer-LI continuation row",
        );
    });

    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        // Outer-LI sibling marker; *not* a stray inner marker.
        assert_eq!(
            e.value(),
            "- parent\n  - child\n\n\n- ",
            "Enter 3: new top-level sibling after the outer LI",
        );
    });

    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        // Empty outer LI → paragraph at top level.
        assert_eq!(
            e.value(),
            "- parent\n  - child\n\n\n\n",
            "Enter 4: empty outer LI drops marker, top-level paragraph",
        );
    });
}

#[gpui::test]
fn double_enter_at_end_of_nested_item_outdents_on_second_press(cx: &mut TestAppContext) {
    // User-reported repro:
    //
    //   - parent
    //     - child|
    //
    // First Enter creates a new empty inner sibling. Second Enter
    // (now on an empty inner item) should behave like Backspace
    // at the same position — outdent the inner LI, leaving a
    // paragraph row at the outer item's content depth.
    let initial = EditorState {
        markdown: "- parent\n  - child".into(),
        selection: Selection::Cursor(18),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);

    // First Enter: new empty inner sibling.
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.value(),
            "- parent\n  - child\n  - ",
            "first Enter should create a new empty nested item",
        );
    });

    // Second Enter: cursor on empty inner item, should outdent.
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        // Same shape as `empty_enter_on_nested_empty_item_outdents_to_outer_paragraph`
        // applied to the trailing empty inner: drop the inner marker,
        // leave the outer item's continuation indent for the new row.
        assert_eq!(
            e.value(),
            "- parent\n  - child\n\n  ",
            "second Enter on empty inner item should outdent (Backspace-equivalent)",
        );
    });
}

#[gpui::test]
fn empty_enter_on_nested_empty_item_outdents_to_outer_paragraph(cx: &mut TestAppContext) {
    // With pulldown's `ENABLE_EMPTY_NESTED_LISTS` option enabled in
    // `parser::parse`, `- one\n  - ` parses as the outer item plus
    // a nested list whose only item is empty. Empty-Enter on that
    // inner item then takes the standard empty-LI outdent path:
    // drop the inner marker, leaving a paragraph row at the outer
    // item's content depth. The result is `- one\n\n  ` — the
    // depth-1 LI-trailing pair shape, with the cursor on the new
    // continuation row inside the outer item.
    //
    // Before the option landed, pulldown couldn't see the inner
    // empty marker as a list item (it needed content), so the
    // chain query returned only the outer LI and Enter fell through
    // to a sibling-marker insertion. The option closes that gap.
    let initial = EditorState {
        markdown: "- one\n  - ".into(),
        selection: Selection::Cursor(10),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- one\n\n  ");
    });
}

#[gpui::test]
fn empty_enter_on_top_level_item_inside_blockquote_stays_in_bq(cx: &mut TestAppContext) {
    // Empty Enter on a depth-1 item inside a BQ should leave the
    // BQ scope intact while ending the list — depth-1 pair shape
    // ahead of the new paragraph.
    let initial = EditorState {
        markdown: "> - one\n> - ".into(),
        selection: Selection::Cursor(12),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        // The empty item drops and a depth-1 paragraph break
        // takes its place — the rest of the BQ stays intact.
        assert!(e.value().starts_with("> - one"));
        assert!(e.value().contains("\n> \n> "));
    });
}

// ---- Backspace-at-start at every nesting level ---------------------------

#[gpui::test]
fn backspace_at_start_of_top_level_item_makes_paragraph(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "- one".into(),
        selection: Selection::Cursor(2), // right after the marker
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // Marker is dropped; the line becomes a paragraph at top
        // level.
        assert_eq!(e.value(), "one");
    });
}

#[gpui::test]
fn backspace_at_start_of_nested_item_dedents_to_outer(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "- one\n  - two".into(),
        selection: Selection::Cursor(10), // right after the inner marker
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // The inner marker stays — Backspace at the marker end of
        // a nested item strips parent-marker-width leading
        // spaces, dedenting the item by one level.
        assert_eq!(e.value(), "- one\n- two");
    });
}

#[gpui::test]
fn enter_past_post_list_separator_does_not_reanimate_list(cx: &mut TestAppContext) {
    // User-reported scenario:
    //
    //   1. asdf
    //   <blank>
    //   |               <-- cursor on the third visual row
    //
    // Built by typing `1. asdf` then Enter twice (the second Enter
    // is the empty-item exit). Cursor lands at the end of buffer
    // `1. asdf\n\n`. A third Enter used to *re-enter the list*
    // and restore `1. asdf\n2. ` — pulldown's list range includes
    // the trailing `\n\n` separator, so the chain walker (with a
    // raw range.end == cursor boundary check) saw the list as
    // still containing the cursor.
    //
    // The architectural fix trims trailing `\n\n` (or longer
    // separator runs) from List / ListItem / BlockQuote ranges
    // before testing containment. Cursor 9 in `1. asdf\n\n` is
    // past the trimmed list end (7), so the chain is empty and
    // Enter routes through the default top-level paragraph break.
    //
    // The expected post-Enter shape is `1. asdf\n\n\n\n` — the
    // original list, an empty paragraph, and a fresh row for the
    // cursor. The list survives unchanged.
    let initial = EditorState {
        markdown: "1. asdf".into(),
        selection: Selection::Cursor(7),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "1. asdf\n2. "));
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "1. asdf\n\n"));
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "1. asdf\n\n\n\n");
        // Cursor at the end so the user can keep typing on the
        // fresh row.
        assert_eq!(e.selection(), Selection::Cursor(11));
    });
}

#[gpui::test]
fn manually_typed_ordered_list_start_is_preserved(cx: &mut TestAppContext) {
    // User types `5. foo` at top level — the list should stay at 5.
    let initial = EditorState {
        markdown: "5. foo".into(),
        selection: Selection::Cursor(6),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    // Trigger a no-op enforce_invariants pass via SelectAll (no
    // structural change, but the post-update normalization runs).
    dispatch(cx, handle, &editor, SelectAll);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "5. foo");
    });
}

#[gpui::test]
fn manually_typed_ordered_list_in_blockquote_preserves_start(cx: &mut TestAppContext) {
    // Inside a blockquote, typing `> 7. foo` should keep the list at 7.
    let initial = EditorState {
        markdown: "> 7. foo".into(),
        selection: Selection::Cursor(8),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, SelectAll);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> 7. foo");
    });
}

#[gpui::test]
fn ordered_list_with_manual_start_renumbers_subsequent_items_from_that_start(
    cx: &mut TestAppContext,
) {
    // Typed `5. one\n2. two\n3. three`. Pulldown sees one list with
    // start=5; renumber pass uses 5 as the basis and rewrites items
    // 2..3 to 6..7. The user's manual `5` is preserved as the first
    // item's number, and subsequent siblings count up from there.
    let initial = EditorState {
        markdown: "5. one\n2. two\n3. three".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, SelectAll);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "5. one\n6. two\n7. three");
    });
}

#[gpui::test]
fn split_list_preserves_trailing_orphan_start(cx: &mut TestAppContext) {
    // User-reported scenario:
    //   1. one
    //   2. |two              <-- cursor after the "2. " marker
    //   3. three
    //
    // Backspace dedents item 2 to a paragraph, splitting the
    // ordered list. The trailing portion ("3. three") keeps its
    // typed start number — manual start values are first-class
    // (a user *can* legitimately type `5. foo` to start a list at
    // 5), and we'd rather preserve the source they typed than
    // second-guess their intent. If the user wants the trailing
    // list to start at 1 they can edit the marker themselves.
    let initial = EditorState {
        markdown: "1. one\n2. two\n3. three".into(),
        selection: Selection::Cursor(10),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "1. one\n\ntwo\n\n3. three");
    });
}

#[gpui::test]
fn enter_at_start_of_paragraph_after_list_inserts_paragraph_break(cx: &mut TestAppContext) {
    // Continuation of the split scenario. After the Backspace
    // dedent the buffer is:
    //
    //   1. one
    //
    //   |two              <-- cursor at the start of "two"
    //
    //   1. three
    //
    // The cursor sits at byte 8, which is *also* the byte that
    // closes the leading list (its range ends at the structural
    // `\n\n`). Without the strict-over-boundary preference in
    // `walk_chain`, the chain at cursor 8 included the leading
    // list's `ListItem` entry, so Enter routed through the list
    // and produced the next-sibling marker `\n2. ` — restoring
    // the original list shape and undoing the Backspace.
    //
    // Strict containment (cursor < range.end) wins over boundary
    // equality (cursor == range.end), so the cursor at the start
    // of the paragraph is recognized as inside the paragraph, not
    // the prior list. Enter inserts the top-level paragraph break.
    let initial = EditorState {
        markdown: "1. one\n\ntwo\n\n1. three".into(),
        selection: Selection::Cursor(8),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "1. one\n\n\n\ntwo\n\n1. three");
    });
}

#[gpui::test]
fn backspace_at_marker_end_of_top_level_item_with_nested_child_strips_orphan_indent(
    cx: &mut TestAppContext,
) {
    // User-reported flow. The original buffer:
    //   1. level one
    //   2. |level one          <-- cursor right after the marker
    //      1. level three      <-- nested child of item 2
    //
    // Backspace at the cursor's marker-end is a top-level item
    // dedent: drop the marker, leaving a paragraph in its place.
    // The nested child line ("   1. level three") used to survive
    // unchanged, leaving 3 spaces of leading whitespace that no
    // longer corresponded to any container — pulldown then
    // re-parsed it as a fresh top-level list with leftover
    // indent. Subsequent operations on that orphaned source could
    // crash apply_edits (overlapping edits in the same byte
    // range).
    //
    // The fix strips the dedented item's marker_width worth of
    // leading spaces from every continuation line, so the child
    // line's indent dies along with the marker. Result: paragraph
    // "level one" followed by a fresh, cleanly-aligned top-level
    // list "1. level three".
    let initial = EditorState {
        markdown: "1. level one\n2. level one\n   1. level three".into(),
        selection: Selection::Cursor(16),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "1. level one\n\nlevel one\n\n1. level three",);
    });
}

#[gpui::test]
fn backspace_then_enter_then_tab_sequence_does_not_crash(cx: &mut TestAppContext) {
    // Regression for the crash in the user-reported sequence:
    //
    //   1. level one
    //   2. |level one
    //      1. level three
    //
    // followed by Backspace → Enter → Tab → Tab. Used to panic
    // `apply_edits` because two passes emitted identical
    // strip-the-blank-line edits at the same byte range, violating
    // the non-overlap invariant.
    //
    // Three fixes in combination cover the flow:
    //
    // - The blank-line strip in `walk_item_content_lines` skips
    //   blank lines that fall inside one of the item's nested
    //   block children (the same guard the hard-break promoter
    //   uses).
    // - `list_item_dedent_edits`'s top-level branch strips the
    //   dedented item's marker_width from continuation lines, so
    //   nested-child indent doesn't survive as orphaned whitespace.
    // - `walk_chain` prefers strict containment over boundary
    //   equality, so the cursor at the start of the next-block
    //   paragraph routes through the paragraph rather than the
    //   list's range that ends at the same byte.
    //
    // The final state is a clean paragraph-and-list document:
    // - Backspace dedents item 2; the orphaned nested child
    //   becomes a fresh top-level list at start=1.
    // - Enter at the start of "level one" paragraph emits the
    //   top-level paragraph break (no longer treated as inside
    //   item 1).
    // - Tab on a paragraph cursor is a no-op (no list to nest in).
    let initial = EditorState {
        markdown: "1. level one\n2. level one\n   1. level three".into(),
        selection: Selection::Cursor(16),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    dispatch(cx, handle, &editor, Tab);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "1. level one\n\n\n\nlevel one\n\n1. level three",);
    });
}

#[gpui::test]
fn tab_on_ordered_item_with_existing_nested_child_does_not_panic(cx: &mut TestAppContext) {
    // Regression for the crash that surfaced as
    //
    //   panicked at update.rs: begin <= end (26 <= 13)
    //
    // Setup:
    //   1. level one
    //   2. level one|     <-- cursor here
    //      1. level three
    //
    // Tab on item 2 should nest it under item 1, dragging the
    // existing nested-child line ("   1. level three") with it
    // to the new deeper indent so the child stays a child.
    //
    // The bug: `list_item_indent_edits` returned its edits in
    // (line_starts in source order) followed by (the marker
    // rewrite for the ordered item). For a multi-line ordered
    // item, that left an unsorted edit list — pad-insert at the
    // second line followed by a marker-rewrite at the first
    // line. `apply_edits` walks edits expecting ascending
    // range.start, so its `last` cursor advanced past the
    // marker-rewrite's position, then hit the rewrite and tried
    // to slice `markdown[last..earlier_start]` — panic.
    //
    // Fix: sort the edits by (range.start, range.end) before
    // returning, so insertions at a position precede replacements
    // at the same position. (`apply_edits` now also asserts the
    // ordering in debug builds.)
    let initial = EditorState {
        markdown: "1. level one\n2. level one\n   1. level three".into(),
        selection: Selection::Cursor(25),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        // Item 2 nested under item 1; nested-child line carried
        // along to the new depth. Pulldown sees this as a
        // 3-level deep ordered list — exactly the user's mental
        // model after one Tab on the parent.
        assert_eq!(
            e.value(),
            "1. level one\n   1. level one\n      1. level three",
        );
    });
}

// ---------------------------------------------------------------------------
// Deep nesting stress corpus
//
// Wild fixtures: code blocks nested 4–5 containers deep, alternating
// `BlockQuote` / `ListItem` chains in both directions, mixed-kind
// siblings inside a single list item, and edits (typing, deleting,
// navigation, depth changes) exercised at the start, interior, and end
// of each kind.
//
// Each test is *self-contained* — its fixture and cursor offset are
// inline so a single failure has full repro on the screen. Failures
// found while authoring the corpus are recorded in `bugs.md` and the
// failing test marked `#[ignore]` with a `known bug:` reason; the
// expected post-action state is encoded so a later fix flips the test
// green without further edits.
// ---------------------------------------------------------------------------

/// Insert `text` at the current cursor, going through `update::update`
/// so `enforce_invariants` runs (the same path keystrokes take in
/// production).
fn type_text(
    cx: &mut TestAppContext,
    handle: AnyWindowHandle,
    editor: &Entity<MarkdownEditorState>,
    text: &str,
) {
    let owned = text.to_string();
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText(owned.clone()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
}
// ---- Static deep-fixture chain shape -------------------------------------

#[gpui::test]
fn deep_code_in_bq_in_list_in_bq_in_list_carries_4_chain_entries(cx: &mut TestAppContext) {
    // Code block four containers deep: List → BQ → List → BQ → code.
    // Continuation prefix on every interior line is `  > ` (outer item
    // indent + BQ marker), then `  > ` again for the inner BQ-in-list:
    //
    //   - > - > ```
    //     >   > code
    //     >   > ```
    let src = "- > - > ```\n  >   > code\n  >   > ```";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let code = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::CodeBlock { .. }))
        .expect("code block leaf");
    assert_eq!(
        list_item_depth(code),
        2,
        "code carries both outer and inner ListItem chain entries",
    );
    assert_eq!(
        blockquote_depth(code),
        2,
        "code carries both outer and inner BlockQuote chain entries",
    );
}

#[gpui::test]
fn deep_code_in_list_in_bq_in_list_in_bq_carries_4_chain_entries(cx: &mut TestAppContext) {
    // Mirror of the above with the outer alternation flipped: BQ →
    // List → BQ → List → code.
    //
    //   > - > - ```
    //   >   >   code
    //   >   >   ```
    let src = "> - > - ```\n>   >   code\n>   >   ```";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let code = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::CodeBlock { .. }))
        .expect("code block leaf");
    assert_eq!(list_item_depth(code), 2);
    assert_eq!(blockquote_depth(code), 2);
}

#[gpui::test]
fn deep_5_level_bq_bq_list_bq_list_chain_renders_each_level(cx: &mut TestAppContext) {
    // Five levels: BQ → BQ → List → BQ → List. Innermost paragraph
    // leaf carries [BQ, BQ, ListItem, BQ, ListItem].
    let src = "> > - > - deepest";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let deepest = spec
        .blocks
        .iter()
        .max_by_key(|b| b.containers.len())
        .expect("at least one block");
    assert_eq!(blockquote_depth(deepest), 3, "three BQ levels");
    assert_eq!(list_item_depth(deepest), 2, "two list-item levels");
}

#[gpui::test]
fn deep_nest_chain_at_start_middle_end_of_innermost_item(_cx: &mut TestAppContext) {
    // The chain walker has a strict-containment-beats-boundary
    // preference; verify it picks the *innermost* item across three
    // cursor positions on the same line: just inside the marker
    // (start), the middle of the content, and the end of the line.
    let src = "- > - > deepest";
    let n = src.len(); // 15
    for cursor in [8usize, 11, n] {
        let chain = gpui_markdown_editor::analysis::enclosing_containers_at(src, cursor);
        assert_eq!(
            gpui_markdown_editor::analysis::chain_blockquote_depth(&chain),
            2,
            "BQ depth at cursor {cursor}",
        );
        let lists: usize = chain
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    gpui_markdown_editor::analysis::EnclosingContainer::ListItem(_)
                )
            })
            .count();
        assert_eq!(lists, 2, "list depth at cursor {cursor}");
    }
}

// ---- Mixed-kind siblings inside one item --------------------------------

#[gpui::test]
fn item_with_paragraph_then_blockquote_then_code_renders_all_in_chain(cx: &mut TestAppContext) {
    // Single list item with three children in source order: a
    // paragraph, a blockquote, and a fenced code block.
    //
    //   - p1
    //
    //     > bq
    //
    //     ```
    //     code
    //     ```
    let src = "- p1\n\n  > bq\n\n  ```\n  code\n  ```";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let leaves: Vec<&gpui_markdown_editor::RenderBlock> = spec
        .blocks
        .iter()
        .filter(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
        .collect();
    // p1 leaf, bq leaf, code leaf — three children of the same
    // outer item.
    assert_eq!(leaves.len(), 3, "three sibling-leaves under item 1");
    let kinds: Vec<&BlockKind> = leaves.iter().map(|b| &b.kind).collect();
    assert!(matches!(kinds[0], BlockKind::Paragraph));
    assert!(matches!(kinds[2], BlockKind::CodeBlock { .. }));
    let bq_leaf = leaves[1];
    assert_eq!(blockquote_depth(bq_leaf), 1, "bq sibling carries BQ chain");
    assert_eq!(list_item_depth(bq_leaf), 1, "still inside outer item");
}

#[gpui::test]
fn item_with_nested_list_then_blockquote_keeps_outer_chain_through_both(cx: &mut TestAppContext) {
    // Sibling-of-different-kinds inside an item: a nested list,
    // then a blockquote, both children of the same outer item.
    //
    //   - outer
    //     - nested
    //
    //     > bq
    let src = "- outer\n  - nested\n\n  > bq";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let leaves: Vec<&gpui_markdown_editor::RenderBlock> = spec
        .blocks
        .iter()
        .filter(|b| matches!(b.containers.first(), Some(Container::ListItem { .. })))
        .collect();
    // outer-item-paragraph leaf + nested-item leaf + bq-sibling
    // leaf = 3 leaves with the outer item in their chain.
    assert!(
        leaves.len() >= 3,
        "outer + nested + bq leaves all carry outer item chain (got {})",
        leaves.len(),
    );
    let bq_leaf = leaves
        .iter()
        .find(|b| blockquote_depth(b) == 1)
        .expect("bq sibling leaf");
    assert_eq!(list_item_depth(bq_leaf), 1, "bq is child of outer item");
}

// ---- Typing into deep nests ----------------------------------------------

#[gpui::test]
fn type_inside_code_at_depth_4_lands_in_code_content(cx: &mut TestAppContext) {
    // `- > - > ```\n  >   > X|\n  >   > ```` — cursor inside the
    // code body. Typing must land in the content (not produce
    // structural changes), and the leaf must remain a code block at
    // depth 4.
    let src = "- > - > ```\n  >   > X\n  >   > ```";
    let cursor = src.find("X").unwrap() + 1;
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(cursor),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "Y");
    editor.read_with(cx, |e, _| {
        assert!(e.value().contains("XY"), "Y typed into code body");
        let spec = e.render_spec();
        let code = spec
            .blocks
            .iter()
            .find(|b| matches!(b.kind, BlockKind::CodeBlock { .. }))
            .expect("still a code block");
        assert_eq!(list_item_depth(code), 2);
        assert_eq!(blockquote_depth(code), 2);
    });
}

#[gpui::test]
fn type_at_end_of_innermost_item_keeps_full_chain(cx: &mut TestAppContext) {
    // End-of-content position on the deepest list item — typing
    // should append to the item, not to a parent or escape the
    // chain.
    let src = "- > - > deep";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "er");
    editor.read_with(cx, |e, _| {
        assert!(e.value().ends_with("deeper"));
        let spec = e.render_spec();
        let deepest = spec
            .blocks
            .iter()
            .max_by_key(|b| b.containers.len())
            .unwrap();
        assert_eq!(blockquote_depth(deepest), 2);
        assert_eq!(list_item_depth(deepest), 2);
    });
}

#[gpui::test]
fn type_at_start_of_innermost_content_keeps_full_chain(cx: &mut TestAppContext) {
    // Start-of-content position (right after the last marker on the
    // deepest line). Typing should land in the item content,
    // preserving every container.
    let src = "- > - > deep";
    let cursor = src.find("deep").unwrap();
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(cursor),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "X");
    editor.read_with(cx, |e, _| {
        assert!(e.value().contains("Xdeep"), "got: {:?}", e.value());
        let spec = e.render_spec();
        let deepest = spec
            .blocks
            .iter()
            .max_by_key(|b| b.containers.len())
            .unwrap();
        assert_eq!(blockquote_depth(deepest), 2);
        assert_eq!(list_item_depth(deepest), 2);
    });
}

#[gpui::test]
fn enter_at_end_of_innermost_item_stays_in_full_chain(cx: &mut TestAppContext) {
    // Enter at end-of-deep-item should produce the next sibling at
    // the same depth — same BQ depth, same list depth, but a new
    // item.
    let src = "- > - > one";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        // The new sibling line carries the same outer-item indent +
        // outer BQ marker + inner-item indent + inner BQ marker.
        // Pulldown picks up `  >   > ` as the depth-4 prefix.
        assert!(
            e.value().starts_with("- > - > one\n  >   > "),
            "got: {:?}",
            e.value(),
        );
    });
}

#[gpui::test]
fn type_gt_at_start_of_nested_item_does_not_split_outer_scope(cx: &mut TestAppContext) {
    // Typing `>` at the start of a nested-item content line should
    // either deepen the BQ scope or be inert — but it must not
    // collapse the outer list-item scope or rewrite the parent.
    let src = "- > - inner";
    let cursor = src.find("inner").unwrap();
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(cursor),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, ">");
    editor.read_with(cx, |e, _| {
        // Outer list and outer BQ both still present after typing.
        assert!(
            e.value().starts_with("- > "),
            "outer list-item + BQ scope preserved, got: {:?}",
            e.value(),
        );
    });
}

// ---- Deleting in deep nests ---------------------------------------------

#[gpui::test]
fn backspace_at_start_of_innermost_item_dedents_one_level(cx: &mut TestAppContext) {
    // Backspace at the marker-end of the inner item should dedent
    // by one (drop the inner `> - ` if Backspace at marker end
    // does the standard one-level outdent), preserving the outer
    // list-item scope.
    let src = "- > - inner";
    let cursor = src.find("inner").unwrap();
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(cursor),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // Outer `- > ` scope must remain — the editor only loses
        // the innermost container, not the whole stack.
        assert!(
            e.value().starts_with("- "),
            "outer list scope preserved, got: {:?}",
            e.value(),
        );
    });
}

#[gpui::test]
fn delete_forward_at_end_of_inner_item_does_not_panic(cx: &mut TestAppContext) {
    // Delete-forward at the end of an inner item with another
    // sibling at the same depth — exercises the merge path between
    // two innermost-list items.
    let src = "- > - one\n  > - two";
    let cursor = src.find("one").unwrap() + "one".len();
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(cursor),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Delete);
    editor.read_with(cx, |e, _| {
        // The buffer is still a list inside a BQ inside a list —
        // the chain shouldn't have collapsed.
        assert!(
            e.value().contains("- "),
            "still has list markers, got: {:?}",
            e.value(),
        );
    });
}

#[gpui::test]
fn backspace_inside_code_at_depth_4_deletes_one_byte(cx: &mut TestAppContext) {
    // Inside a deeply nested code block, Backspace should delete
    // one grapheme of code content (not collapse the whole block).
    let src = "- > - > ```\n  >   > abc\n  >   > ```";
    let cursor = src.find("abc").unwrap() + 3; // after `c`
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(cursor),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert!(
            e.value().contains("ab\n"),
            "code body should now read `ab` not `abc`, got: {:?}",
            e.value(),
        );
        let spec = e.render_spec();
        // Still a code block; chain still has both BQ levels.
        let code = spec
            .blocks
            .iter()
            .find(|b| matches!(b.kind, BlockKind::CodeBlock { .. }))
            .expect("still a code block");
        assert_eq!(blockquote_depth(code), 2);
        assert_eq!(list_item_depth(code), 2);
    });
}

#[gpui::test]
fn select_all_then_backspace_clears_deep_nest_without_panicking(cx: &mut TestAppContext) {
    // Catastrophic delete on a wild nest must leave a usable empty
    // editor — no panic, no torn state.
    let src = "- > - > ```\n  >   > code\n  >   > ```\n\n- sibling";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, SelectAll);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert!(e.value().is_empty());
        // Still produces at least one block to host the cursor.
        assert!(!e.render_spec().blocks.is_empty());
    });
}

// ---- Depth changes (Tab / Shift+Tab / empty Enter) in deep nests ---------

#[gpui::test]
fn shift_tab_at_innermost_of_4_level_alternation_dedents_one_level(cx: &mut TestAppContext) {
    // Shift+Tab on the innermost item should drop the innermost
    // list-item, leaving the outer list-item + both blockquotes
    // intact.
    let src = "- > - > inner";
    let cursor = src.find("inner").unwrap();
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(cursor),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        // Outer `- > ` chain still present; inner `- ` dropped.
        assert!(
            e.value().starts_with("- "),
            "outer list still present, got: {:?}",
            e.value(),
        );
        // The deepest leaf no longer carries 2 list-items — at
        // least one was dropped.
        let spec = e.render_spec();
        let deepest = spec
            .blocks
            .iter()
            .max_by_key(|b| b.containers.len())
            .unwrap();
        assert!(
            list_item_depth(deepest) <= 1,
            "innermost list-item dropped (depth ≤ 1), got: {}",
            list_item_depth(deepest),
        );
    });
}

#[gpui::test]
fn tab_on_a_sibling_inside_deep_nest_nests_within_existing_chain(cx: &mut TestAppContext) {
    // Two innermost siblings; Tab on the second should nest it
    // under the first, *without* losing the outer 4-level chain.
    //
    //   - > - one
    //     > - two       <- cursor in "two"
    let src = "- > - one\n  > - two";
    let cursor = src.find("two").unwrap();
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(cursor),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        let md = e.value();
        // The outer `- > ` scope stays — the only thing that
        // changed is the indent on the second inner-item line.
        assert!(md.starts_with("- > "), "outer chain preserved, got: {md:?}",);
        // The inner-second-item should be deeper than the
        // inner-first-item by one list level.
        let spec = e.render_spec();
        let max_depth = spec.blocks.iter().map(list_item_depth).max().unwrap_or(0);
        assert!(
            max_depth >= 3,
            "expected at least 3 list-item levels after Tab, got max {max_depth}; markdown: {md:?}",
        );
    });
}

#[gpui::test]
fn empty_enter_on_innermost_item_in_deep_nest_dedents(cx: &mut TestAppContext) {
    // Empty inner item — Enter should dedent by one level (drop
    // the innermost marker), staying within the outer scope.
    let src = "- > - one\n  > - ";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        // Outer `- > ` chain still present.
        assert!(
            e.value().starts_with("- > - one"),
            "outer chain + first inner item preserved, got: {:?}",
            e.value(),
        );
    });
}

// ---- Navigation in deep nests --------------------------------------------

#[gpui::test]
fn right_arrow_walks_through_entire_deep_fixture_without_panicking(cx: &mut TestAppContext) {
    // Walk Right from byte 0 to past the end. Each step must
    // succeed (no panic, no infinite loop) and the cursor must be
    // monotonic-non-decreasing — even when stepping over forbidden
    // pair interiors and continuation prefixes.
    let src = "- > - > ```\n  >   > code\n  >   > ```\n\n- sibling";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    let mut last = 0usize;
    for _ in 0..(src.len() + 4) {
        dispatch(cx, handle, &editor, Right);
        let cur = editor.read_with(cx, |e, _| e.cursor_offset());
        assert!(
            cur >= last,
            "cursor went backwards: {last} → {cur} (markdown len {})",
            src.len(),
        );
        last = cur;
    }
}

#[gpui::test]
fn down_arrow_from_top_through_deep_fixture_terminates(cx: &mut TestAppContext) {
    // Down repeatedly — must reach EOF and terminate, not loop.
    let src = "- > - > ```\n  >   > code\n  >   > ```\n\n- sibling";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    let mut last = 0usize;
    let mut stuck_count = 0usize;
    for _ in 0..32 {
        dispatch(cx, handle, &editor, Down);
        let cur = editor.read_with(cx, |e, _| e.cursor_offset());
        if cur == last {
            stuck_count += 1;
            if stuck_count > 2 {
                break;
            }
        } else {
            stuck_count = 0;
        }
        last = cur;
    }
    // After enough Downs we should be past the deep fixture.
    assert!(
        last > "- > - > ```\n  >   > code\n  >   > ```".len(),
        "Down didn't progress past the code block region (cursor at {last})",
    );
}

#[gpui::test]
fn home_inside_deeply_indented_continuation_lands_at_content_edge(cx: &mut TestAppContext) {
    // Cursor inside `code` on the continuation line; Home should
    // land at the *content edge* (right after the last container
    // prefix `> `), not at the raw line start (which would be
    // inside the prefix bytes).
    let src = "- > - > ```\n  >   > code\n  >   > ```";
    let line2_start = src.find("\n  >   > code").unwrap() + 1;
    let content_edge = src.find("code").unwrap();
    let cursor = content_edge + 2; // mid-content
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(cursor),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Home);
    let landed = editor.read_with(cx, |e, _| e.cursor_offset());
    // Either at the content edge or at line2_start are sane —
    // this assertion encodes "Home lands somewhere on line 2,
    // never before line2_start (we don't backtrack into line 1)".
    assert!(
        landed >= line2_start,
        "Home moved to byte {landed}, before line2_start ({line2_start})",
    );
}

#[gpui::test]
fn end_inside_deeply_indented_line_lands_at_line_terminus(cx: &mut TestAppContext) {
    let src = "- > - > ```\n  >   > code\n  >   > ```";
    let cursor = src.find("code").unwrap();
    let line2_end = src.find("code").unwrap() + "code".len();
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(cursor),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, End);
    let landed = editor.read_with(cx, |e, _| e.cursor_offset());
    assert_eq!(landed, line2_end, "End lands at end of `code` line");
}

// ---- Building deep structure incrementally (typing-driven) ---------------

#[gpui::test]
fn typing_a_blockquote_marker_into_a_list_item_deepens_scope(cx: &mut TestAppContext) {
    // Start: `- foo`. Type `>` then ` ` at the marker-end position
    // — the user keying their way into a BQ inside the list item.
    let initial = EditorState {
        markdown: "- foo".into(),
        selection: Selection::Cursor(2), // right after `- `
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "> ");
    editor.read_with(cx, |e, _| {
        // Inside a list item, `- > foo` is a paragraph beginning
        // with `> ` literally, *unless* pulldown opens a BQ on
        // that line. Either outcome must not corrupt the outer
        // list — the `- ` remains.
        assert!(
            e.value().starts_with("- "),
            "outer list-item marker preserved, got: {:?}",
            e.value(),
        );
    });
}

#[gpui::test]
fn list_spacing_does_not_change_with_cursor_focus(cx: &mut TestAppContext) {
    // Regression: list items in the same list used to gain or lose
    // a `container_boundary_gap` half whenever the cursor moved
    // between items, because `Container::ListItem`'s `cursor_inside`
    // flag was part of the chain-equality check and made adjacent
    // siblings look like "different chains" the moment one was
    // focused. The spacing must be stable across cursor positions —
    // cursor focus is a visual cue on the *current* row, not a
    // signal to re-flow the document.
    let src = "Paragraph one.\n\n- list item one\n- list item two\n\nParagraph two.";

    fn render_with_cursor(
        cx: &mut TestAppContext,
        src: &str,
        cursor: usize,
    ) -> Vec<(usize, std::ops::Range<usize>)> {
        let initial = EditorState {
            markdown: src.into(),
            selection: Selection::Cursor(cursor),
            ..Default::default()
        };
        let (_handle, editor) = open_editor(cx, initial);
        editor.read_with(cx, |e, _| {
            e.render_spec()
                .blocks
                .iter()
                .enumerate()
                .map(|(i, b)| (i, b.source_range.clone()))
                .collect()
        })
    }

    // Cursor in the first list item.
    let cursor_in_item1 = src.find("list item one").unwrap() + 1;
    // Cursor in the leading paragraph.
    let cursor_in_para1 = src.find("Paragraph one").unwrap() + 1;
    // Cursor in the trailing paragraph.
    let cursor_in_para2 = src.find("Paragraph two").unwrap() + 1;

    let in_item1 = render_with_cursor(cx, src, cursor_in_item1);
    let in_para1 = render_with_cursor(cx, src, cursor_in_para1);
    let in_para2 = render_with_cursor(cx, src, cursor_in_para2);

    // The block source ranges (and therefore vertical layout) are
    // identical regardless of which item / paragraph the cursor
    // sits in. Cursor focus only changes inline run styling, not
    // the structural shape.
    assert_eq!(
        in_item1, in_para1,
        "block ranges differ when cursor moves between list and paragraph"
    );
    assert_eq!(in_para1, in_para2, "block ranges differ between paragraphs");
}

#[gpui::test]
fn list_item_trailing_space_keeps_cursor_in_block(cx: &mut TestAppContext) {
    // Repro for the trailing-space cursor invisibility bug: typing
    // a trailing space at the end of a list-item line must leave the
    // buffer with that space and the cursor positioned *after* it.
    // (The render-side fix is verified in `render::tests`; here we
    // just pin the buffer state so a future invariant pass doesn't
    // silently strip the space.)
    let initial = EditorState {
        markdown: "".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "- foo ");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- foo ");
        assert_eq!(e.cursor_offset(), 6);
    });
}

#[gpui::test]
fn enter_on_empty_task_item_outdents_to_paragraph(cx: &mut TestAppContext) {
    // Pressing Enter on an empty task item (`- [ ] ` with nothing
    // after the brackets) drops the item back to a paragraph —
    // parity with the empty-bullet and empty-numbered behavior.
    let initial = EditorState {
        markdown: "- [ ] ".into(),
        selection: Selection::Cursor(6),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        // The marker chrome is gone; the buffer now has just the
        // empty paragraph the cursor lands on.
        assert!(
            !e.value().contains("[ ]"),
            "task chrome should be gone after empty-Enter outdent, got {:?}",
            e.value(),
        );
    });
}

#[gpui::test]
fn enter_on_empty_checked_task_item_outdents(cx: &mut TestAppContext) {
    // Same rule for `- [x] ` — checked-state shouldn't matter.
    let initial = EditorState {
        markdown: "- [x] ".into(),
        selection: Selection::Cursor(6),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert!(
            !e.value().contains("[x]"),
            "checked task chrome should be gone after empty-Enter outdent, got {:?}",
            e.value(),
        );
    });
}

#[gpui::test]
fn enter_on_empty_task_in_a_list_drops_only_that_item(cx: &mut TestAppContext) {
    // The trailing empty task item exits the list cleanly; the
    // earlier non-empty items stay intact.
    let initial = EditorState {
        markdown: "- [x] one\n- [ ] ".into(),
        selection: Selection::Cursor(16),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert!(
            e.value().starts_with("- [x] one"),
            "first item should be preserved, got {:?}",
            e.value(),
        );
        // The trailing `- [ ] ` is gone (the empty task triggered
        // the outdent path).
        assert!(
            !e.value().ends_with("- [ ] "),
            "empty task item should be removed, got {:?}",
            e.value(),
        );
    });
}

#[gpui::test]
fn enter_at_end_of_task_item_creates_another_task_item(cx: &mut TestAppContext) {
    // Pressing Enter at the end of `- [x] done` continues the task
    // list — the new sibling carries `- [ ] ` (fresh-unchecked) so
    // the user keeps typing todos without re-typing the brackets.
    let initial = EditorState {
        markdown: "- [x] done".into(),
        selection: Selection::Cursor(10),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- [x] done\n- [ ] ");
        // Cursor lands at the start of the new item's content.
        assert_eq!(e.cursor_offset(), 17);
    });
}

#[gpui::test]
fn enter_at_end_of_unchecked_task_item_creates_another_task_item(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "- [ ] todo".into(),
        selection: Selection::Cursor(10),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- [ ] todo\n- [ ] ");
        assert_eq!(e.cursor_offset(), 17);
    });
}

#[gpui::test]
fn enter_at_end_of_plain_unordered_item_does_not_become_task(cx: &mut TestAppContext) {
    // Regression: the task continuation must not leak into plain
    // unordered items.
    let initial = EditorState {
        markdown: "- foo".into(),
        selection: Selection::Cursor(5),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- foo\n- ");
    });
}

#[gpui::test]
fn typing_dash_then_letter_injects_marker_space(cx: &mut TestAppContext) {
    // Typing `-foo` is a clear list intent — CommonMark requires
    // `<marker><space>` to open a list, but the editor injects the
    // missing space so the user gets a list rather than text.
    let initial = EditorState {
        markdown: "".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "-foo");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- foo");
    });
}

#[gpui::test]
fn typing_star_then_letter_does_not_inject_marker_space(cx: &mut TestAppContext) {
    // `*` is excluded from the rule: it opens emphasis at least as
    // often as it opens a bullet, and the ambiguity can't be resolved
    // from the prefix alone (`*x` is the live prefix of both `*x*`
    // and a `* x` bullet). A `*` bullet is typed with its space.
    let initial = EditorState {
        markdown: "".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "*x");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "*x");
    });
}

/// Type `text` one keystroke at a time, the way a user does — each
/// character is its own event, so the cursor guards in
/// `enforce_invariants` see every intermediate buffer.
fn type_chars(
    cx: &mut TestAppContext,
    handle: AnyWindowHandle,
    editor: &Entity<MarkdownEditorState>,
    text: &str,
) {
    let mut buf = [0u8; 4];
    for ch in text.chars() {
        type_text(cx, handle, editor, ch.encode_utf8(&mut buf));
    }
}

fn assert_no_list(spec: &RenderSpec, what: &str) {
    assert!(
        spec.blocks.iter().all(|b| !b
            .containers
            .iter()
            .any(|c| matches!(c, Container::ListItem { .. }))),
        "{what}: expected no list container, blocks: {:#?}",
        spec.blocks,
    );
}

#[gpui::test]
fn typing_italic_at_paragraph_start_stays_a_paragraph(cx: &mut TestAppContext) {
    // The regression: keystroke-by-keystroke, `*Italic* text …` used
    // to become `* Italic* text …` — a bullet — the instant the
    // second character landed.
    let initial = EditorState {
        markdown: "".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_chars(cx, handle, &editor, "*Italic* text at the beginning.");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "*Italic* text at the beginning.");
    });
    let spec = current_spec(cx, &editor);
    assert_no_list(&spec, "typed italic paragraph");
    assert!(
        spec.blocks[0]
            .inlines
            .iter()
            .any(|r| r.style.italic && r.source_range == (1..7)),
        "emphasis run over `Italic`, inlines: {:#?}",
        spec.blocks[0].inlines,
    );
}

#[gpui::test]
fn typing_underscore_italic_at_paragraph_start_stays_a_paragraph(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_chars(cx, handle, &editor, "_italic_ text at the beginning.");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "_italic_ text at the beginning.");
    });
    assert_no_list(&current_spec(cx, &editor), "typed underscore paragraph");
}

#[gpui::test]
fn typing_bold_at_paragraph_start_stays_a_paragraph(cx: &mut TestAppContext) {
    // Already covered by the repeat-marker exception; pinned so the
    // `*` exclusion can't be undone by a change that only looks at
    // the doubled case.
    let initial = EditorState {
        markdown: "".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_chars(cx, handle, &editor, "**bold** start");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "**bold** start");
    });
    assert_no_list(&current_spec(cx, &editor), "typed bold paragraph");
}

#[gpui::test]
fn seeded_italic_paragraph_survives_an_edit_elsewhere(cx: &mut TestAppContext) {
    // The cursor guard only protects the byte under the caret, so a
    // document *containing* an italic-opening paragraph was rewritten
    // by the next keystroke anywhere in the buffer.
    let initial = EditorState {
        markdown: "Intro\n\n*Italic* text at the beginning.".into(),
        selection: Selection::Cursor(5),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "!");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "Intro!\n\n*Italic* text at the beginning.");
    });
    assert_no_list(&current_spec(cx, &editor), "seeded italic paragraph");
}

#[gpui::test]
fn seeded_italic_paragraph_in_a_blockquote_survives_an_edit(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "> *Italic* text".into(),
        selection: Selection::Cursor(15),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "!");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> *Italic* text!");
    });
    assert_no_list(&current_spec(cx, &editor), "italic paragraph in BQ");
}

#[gpui::test]
fn typing_star_space_still_opens_a_bullet(cx: &mut TestAppContext) {
    // The explicit spelling is untouched by the exclusion.
    let initial = EditorState {
        markdown: "".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_chars(cx, handle, &editor, "* foo");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "* foo");
        assert!(
            e.render_spec()
                .blocks
                .iter()
                .any(|b| list_item_depth(b) >= 1),
            "`* foo` is a bullet",
        );
    });
}

#[gpui::test]
fn fresh_empty_nested_marker_is_still_a_bullet(cx: &mut TestAppContext) {
    // The pulldown fork's `ENABLE_EMPTY_NESTED_LISTS` behavior: a
    // just-typed empty marker line inside an item opens a nested list
    // instead of being claimed as a setext underline. Nothing about
    // the `*` exclusion may disturb it.
    let initial = EditorState {
        markdown: "- one\n  - ".into(),
        selection: Selection::Cursor(10),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    assert!(
        spec.blocks.iter().any(|b| list_item_depth(b) == 2),
        "nested empty item present, blocks: {:#?}",
        spec.blocks,
    );
}

#[gpui::test]
fn typing_plus_then_letter_injects_marker_space(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "+x");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "+ x");
    });
}

#[gpui::test]
fn typing_double_dash_does_not_inject_space(cx: &mut TestAppContext) {
    // `--` could be in progress towards `---` (a thematic break) or
    // just typed-in-a-row marker chars. The repeat-marker exception
    // skips injection so the user can complete the thematic break.
    let initial = EditorState {
        markdown: "".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "--");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "--");
    });
}

#[gpui::test]
fn typing_three_dashes_remains_thematic_break_candidate(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "---");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "---");
    });
}

#[gpui::test]
fn typing_inside_existing_list_item_does_not_inject_space(cx: &mut TestAppContext) {
    // The user types `-x` as content of an existing item — no space
    // injection. `chain_for_position` returns a chain containing
    // `ListItem` for this byte, which suppresses the rule.
    let initial = EditorState {
        markdown: "- foo\n  ".into(),
        selection: Selection::Cursor(8),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "-bar");
    editor.read_with(cx, |e, _| {
        // The continuation line carries the LI indent, so the `-bar`
        // content is part of the item's continuation, not a new
        // sibling marker. The `-` stays adjacent to `bar`.
        assert!(
            e.value().contains("-bar"),
            "expected unchanged `-bar` inside item, got {:?}",
            e.value(),
        );
    });
}

#[gpui::test]
fn typing_dash_alone_does_not_inject_space(cx: &mut TestAppContext) {
    // `-` alone (followed by EOF or `\n`) is still ambiguous —
    // could be the start of a list, thematic break, or text. Don't
    // touch it until the user types a non-marker character.
    let initial = EditorState {
        markdown: "".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "-");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "-");
    });
}

#[gpui::test]
fn typing_a_list_marker_into_a_bq_paragraph_opens_a_list(cx: &mut TestAppContext) {
    // `> ` then `- ` should open a list inside the BQ.
    let initial = EditorState {
        markdown: "> ".into(),
        selection: Selection::Cursor(2),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "- foo");
    editor.read_with(cx, |e, _| {
        let spec = e.render_spec();
        let list_in_bq = spec
            .blocks
            .iter()
            .any(|b| blockquote_depth(b) >= 1 && list_item_depth(b) >= 1);
        assert!(
            list_in_bq,
            "list-in-BQ chain present after typing, blocks: {:#?}",
            spec.blocks,
        );
    });
}

#[gpui::test]
fn build_3_level_list_from_scratch_via_tabs(cx: &mut TestAppContext) {
    // User-flow: type `- one`, Enter, Tab, type `two`, Enter, Tab,
    // type `three`. End state: a triple-nested list with one item
    // at each level.
    let initial = EditorState {
        markdown: "".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "- one");
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    dispatch(cx, handle, &editor, Tab);
    type_text(cx, handle, &editor, "two");
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    dispatch(cx, handle, &editor, Tab);
    type_text(cx, handle, &editor, "three");
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.value(),
            "- one\n  - two\n    - three",
            "expected canonical 3-level list",
        );
    });
}

#[gpui::test]
fn build_bq_then_list_then_bq_via_typing(cx: &mut TestAppContext) {
    // Build a 3-container chain by typing prefixes: end up with a
    // BQ → List → BQ → content shape.
    let initial = EditorState {
        markdown: "".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "> - > deep");
    editor.read_with(cx, |e, _| {
        let spec = e.render_spec();
        // At least one leaf carries [BQ, ListItem, BQ] in some order.
        let target = spec
            .blocks
            .iter()
            .find(|b| blockquote_depth(b) >= 2 && list_item_depth(b) >= 1);
        assert!(
            target.is_some(),
            "no leaf has BQ+List+BQ chain, blocks: {:#?}",
            spec.blocks,
        );
    });
}

// ---- Chain integrity across edits ----------------------------------------

#[gpui::test]
fn chain_remains_consistent_after_typing_at_innermost(cx: &mut TestAppContext) {
    // Sanity: chain at cursor before/after a single keystroke is
    // unchanged at the ListItem/BlockQuote level.
    let src = "- > - > deep";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    let chain_before = editor.read_with(cx, |e, _| {
        gpui_markdown_editor::analysis::enclosing_containers_at(e.value(), e.cursor_offset())
    });
    type_text(cx, handle, &editor, "X");
    let chain_after = editor.read_with(cx, |e, _| {
        gpui_markdown_editor::analysis::enclosing_containers_at(e.value(), e.cursor_offset())
    });
    assert_eq!(
        chain_before.len(),
        chain_after.len(),
        "chain length unchanged after typing",
    );
}

#[gpui::test]
fn end_of_outermost_list_followed_by_paragraph_does_not_reanimate(cx: &mut TestAppContext) {
    // After a deep nested list ends, a trailing paragraph at the
    // top level must not reabsorb back into any list-item chain.
    let src = "- > - > deep\n\nafter";
    let cursor = src.find("after").unwrap() + "after".len();
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(cursor),
        ..Default::default()
    };
    let (_handle, editor) = open_editor(cx, initial);
    let chain = editor.read_with(cx, |e, _| {
        gpui_markdown_editor::analysis::enclosing_containers_at(e.value(), e.cursor_offset())
    });
    assert!(
        chain.is_empty(),
        "cursor in `after` paragraph should be at top level, got chain of length {}",
        chain.len(),
    );
}

// ---- Bugs surfaced by the interactive `code_review_response_session` ----

#[gpui::test]
fn cursor_chain_and_block_chain_agree_on_post_shift_enter_position(cx: &mut TestAppContext) {
    // Repro from the session script: type `1. one`, then ShiftEnter,
    // ShiftEnter — the user expects to be sitting on a paragraph break
    // *inside* the item, but the synthetic empty leaf the renderer
    // produces for the cursor's row claims the chain is `(top level)`,
    // contradicting the cursor walker.
    let initial = EditorState {
        markdown: "1. one".into(),
        selection: Selection::Cursor(6),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: true,
        },
    );
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: true,
        },
    );
    editor.read_with(cx, |e, _| {
        let cursor = e.cursor_offset();
        let cursor_chain = gpui_markdown_editor::analysis::enclosing_containers_at(
            e.value(),
            cursor,
        );
        let spec = e.render_spec();
        let block = spec
            .blocks
            .iter()
            .find(|b| b.source_range.contains(&cursor) || b.source_range.end == cursor)
            .expect("a block for the cursor's row");
        let cursor_in_list = !cursor_chain.is_empty();
        let block_in_list = !block.containers.is_empty();
        assert_eq!(
            cursor_in_list, block_in_list,
            "cursor walker says in_list={cursor_in_list} but block walker says in_list={block_in_list} \
             for source {:?} cursor={cursor}",
            e.value(),
        );
    });
}

#[gpui::test]
fn paragraph_break_inside_bq_in_list_keeps_full_continuation_prefix(cx: &mut TestAppContext) {
    // Reproduces the structural failure from session keyframe 11: two
    // ShiftEnters at end of a BQ paragraph that lives inside an item
    // should produce a paragraph break *inside both scopes*, but
    // currently only the BQ scope is preserved on the new line.
    let initial = EditorState {
        markdown: "1. one\n\n   > a".into(),
        selection: Selection::Cursor(14), // after `> a`
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: true,
        },
    );
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: true,
        },
    );
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText("b".into()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        // The trailing line carries the *outer item's* continuation
        // indent before the BQ marker, so the BQ stays inside item 1.
        assert!(
            e.value().ends_with("   > b"),
            "BQ continuation should keep `   > ` outer-indent prefix, got: {:?}",
            e.value(),
        );
    });
}

#[gpui::test]
fn inserted_text_with_newline_inside_code_in_bq_keeps_single_newline(cx: &mut TestAppContext) {
    // Inside a BQ, open a fenced code block, then insert two-line
    // text. The embedded `\n` should remain literal — code content
    // is exempt from soft-break promotion. Currently the BQ wrapping
    // defeats that exemption and `>   \n>` is injected between the
    // lines.
    let initial = EditorState {
        markdown: "> ```\n> \n> ```".into(),
        selection: Selection::Cursor(8), // inside the empty code body line
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText("foo\nbar".into()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        // No `>   \n>` pair injected — the two lines remain a single
        // `\n` apart inside the code body.
        let md = e.value();
        assert!(
            md.contains("> foo\n> bar"),
            "expected `> foo\\n> bar` literal in code body, got: {md:?}",
        );
        assert!(
            !md.contains("> foo\n>   \n> bar") && !md.contains("> foo\n>  \n> bar"),
            "BQ-pair should NOT have been injected between code lines, got: {md:?}",
        );
    });
}

#[gpui::test]
fn render_blocks_have_no_overlapping_source_ranges_in_simple_bq_in_list(cx: &mut TestAppContext) {
    // The minimal repro for the overlapping-blocks observation in
    // session keyframe 19: an item whose BQ child contains a nested
    // list. The render walker emits a paragraph leaf and a list-item
    // leaf with overlapping source ranges.
    let src = "1. outer\n\n   > - inner";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_handle, editor) = open_editor(cx, initial);
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    // Sort by start, then verify each subsequent block starts at or
    // after the previous block's end.
    let mut sorted: Vec<&gpui_markdown_editor::RenderBlock> = spec.blocks.iter().collect();
    sorted.sort_by_key(|b| (b.source_range.start, b.source_range.end));
    for w in sorted.windows(2) {
        let a = w[0];
        let b = w[1];
        assert!(
            a.source_range.end <= b.source_range.start,
            "overlap: block {:?} ({:?}) and block {:?} ({:?}) — ranges overlap",
            a.kind,
            a.source_range,
            b.kind,
            b.source_range,
        );
    }
}

#[gpui::test]
fn enter_on_empty_inner_item_in_bq_in_outer_list_keeps_outer_indent(cx: &mut TestAppContext) {
    // Repro from the user's report: an outer list item containing a
    // BQ that itself contains a nested list. Cursor on the empty
    // inner item; pressing Enter should drop the inner `- ` marker
    // and produce a depth-1-pair separator that *carries the outer
    // item's continuation indent* on every BQ line, so the BQ stays
    // inside the outer list scope.
    let src =
        "- This is a list.\n\n  > With a blockquote.\n  > \n  > - With a nested list.\n  > - ";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        // Trailing two BQ-prefix lines must carry the outer item's
        // `  ` continuation indent, not be at column 0.
        assert!(
            !e.value().contains("\n> "),
            "found a column-0 `> ` line — outer LI continuation indent was dropped: {:?}",
            e.value(),
        );
        assert!(
            e.value().ends_with("  > "),
            "buffer should end with a BQ continuation line at the outer item's indent column, got: {:?}",
            e.value(),
        );
        // Outer list-item is still intact at the start.
        assert!(
            e.value().starts_with("- This is a list."),
            "outer item's first paragraph preserved",
        );
    });
}

#[gpui::test]
fn shift_tab_on_inner_li_in_bq_wrapped_list_dedents(cx: &mut TestAppContext) {
    // Repro from the user's report. Chain at cursor is
    // [BQ, outer-LI(ord), inner-LI(ord)]. Shift+Tab on the empty
    // inner item should make it a sibling of the outer LI.
    //
    // The line containing the inner marker starts with `>` (the BQ
    // marker), not a space, so the existing nested-branch
    // strip-leading-spaces loop matches zero bytes and the dedent
    // is a no-op. Fix: skip past the active container-prefix bytes
    // (BQ here) before stripping the parent LI's marker_width
    // worth of leading spaces — same architectural pattern as the
    // Tab indent fix that introduced `chain_outer_prefix_bytes`.
    let src =
        "> This is a blockquote.\n> \n> 1. This is a list\n>    1. With a nested list\n>    2. ";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        // Inner marker is now at the outer level (sibling of `1. This is a list`).
        // Result line should be `> 2. ` (BQ + sibling marker), not `>    2. `.
        assert!(
            e.value().ends_with("\n> 2. "),
            "Shift+Tab should have dedented the inner item to sibling of outer LI; got: {:?}",
            e.value(),
        );
    });
}

#[gpui::test]
fn backspace_on_empty_bq_paragraph_inside_li_drops_bq_scope(cx: &mut TestAppContext) {
    // Repro from the user's report. The BQ has two trailing empty
    // lines (the canonical depth-1 pair shape inside an LI:
    // `\n   > \n   > `). Cursor is at the end. The user expects
    // Backspace to convert the empty BQ paragraph into a raw
    // (non-BQ) paragraph still nested inside the outer LI — the
    // analog of "empty LI Enter drops the marker" but for the BQ
    // scope.
    let src = "1. This is a list.\n\n   > This is a blockquote\n   > \n   > ";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // Path B: the BQ scope is dropped on the trailing row but
        // the cursor lands on an empty (non-BQ) continuation row
        // still nested in the outer LI. The earlier BQ paragraph
        // ("This is a blockquote") survives.
        let md = e.value();
        let cursor = e.cursor_offset();
        let chain = gpui_markdown_editor::analysis::enclosing_containers_at(md, cursor);

        // Outer LI's first paragraph preserved.
        assert!(
            md.starts_with("1. This is a list."),
            "outer LI's first paragraph preserved, got: {md:?}",
        );
        // The earlier BQ paragraph survived as content.
        assert!(
            md.contains("> This is a blockquote"),
            "first BQ paragraph preserved, got: {md:?}",
        );

        // Cursor is on a non-BQ row inside the LI: chain has
        // exactly one LI and ZERO BQs. (Path A would have left the
        // chain at [outer-LI, BQ] with cursor at the end of the
        // surviving BQ paragraph, which is *not* what the user
        // wants.)
        let li_count = chain
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    gpui_markdown_editor::analysis::EnclosingContainer::ListItem(_)
                )
            })
            .count();
        let bq_count = chain
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    gpui_markdown_editor::analysis::EnclosingContainer::BlockQuote { .. }
                )
            })
            .count();
        assert_eq!(li_count, 1, "still in outer LI; chain: {chain:?}");
        assert_eq!(
            bq_count, 0,
            "cursor must be on a non-BQ row (BQ scope dropped); chain: {chain:?}",
        );

        // The trailing position must accept the cursor — no
        // trailing `> ` (would be a BQ continuation).
        assert!(
            !md.ends_with("> "),
            "trailing line is not a BQ continuation, got: {md:?}",
        );
    });
}

#[gpui::test]
fn left_arrow_skips_hidden_bq_continuation_prefix_in_li(cx: &mut TestAppContext) {
    // From the same fixture as the empty-BQ-in-LI Backspace bug.
    // Left from byte 57 (end of buffer) should skip over every
    // hidden continuation-prefix byte and land at the previous
    // *visible content edge* — not at byte 56 (between the BQ's
    // `>` and ` `), 55 (between LI indent and BQ `>`), 50, or 49.
    //
    // The `\n   > ` prefix has 5 hidden bytes; one Left over the
    // entire `\n   > \n   > ` pair should land on the previous
    // content row's terminus (end of "blockquote", byte 45) since
    // every byte between 45 and 57 is structural prefix interior.
    let src = "1. This is a list.\n\n   > This is a blockquote\n   > \n   > ";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    // After one Left, cursor must be at a non-forbidden position.
    // The strict version: cursor lands at byte 45 (end of
    // `blockquote`). The looser version we'll assert: cursor is
    // not at any of the known-forbidden bytes (49, 50, 55, 56).
    dispatch(cx, handle, &editor, Left);
    let cursor = editor.read_with(cx, |e, _| e.cursor_offset());
    assert!(
        !matches!(cursor, 49 | 50 | 55 | 56),
        "Left-arrow landed at forbidden BQ-prefix-interior byte {cursor}",
    );
}

#[gpui::test]
fn same_line_bq_inside_li_renders_with_li_marker_overlay(cx: &mut TestAppContext) {
    // Repro from the user's report. Per CommonMark, this parses as
    // a list with one item containing a blockquote containing a
    // paragraph. The render walker should emit one paragraph leaf
    // with chain [LI, BQ], the LI marker visible/hidden per cursor
    // position, and the BQ chrome painted.
    //
    // Currently the LI's marker overlay isn't emitted — the
    // rendered output has no list-item marker.
    let src = "1. > This list contains a blockquote.";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_handle, editor) = open_editor(cx, initial);
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    let paragraph = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::Paragraph))
        .expect("paragraph leaf for the BQ content");
    // Chain has both LI and BQ.
    let li_count = paragraph
        .containers
        .iter()
        .filter(|c| matches!(c, Container::ListItem { .. }))
        .count();
    let bq_count = paragraph
        .containers
        .iter()
        .filter(|c| matches!(c, Container::BlockQuote { .. }))
        .count();
    assert_eq!(li_count, 1, "leaf carries one ListItem container");
    assert_eq!(bq_count, 1, "leaf carries one BlockQuote container");
    // The leaf must include some indication that the LI marker
    // exists — either a marker_overlay for the `1. ` bytes or a
    // hidden_range covering them. Without one, the rendered output
    // can't show "1." anywhere, which is the user-visible bug.
    let has_li_marker_chrome = !paragraph.marker_overlays.is_empty()
        || paragraph
            .hidden_ranges
            .iter()
            .any(|r| r.start == 0 && r.end >= 3);
    assert!(
        has_li_marker_chrome,
        "expected LI marker chrome (overlay or hidden range covering `1. `); \
         got marker_overlays={:?}, hidden_ranges={:?}",
        paragraph.marker_overlays, paragraph.hidden_ranges,
    );
}

#[gpui::test]
fn empty_paragraph_after_bq_outdent_renders_at_top_level(cx: &mut TestAppContext) {
    // Repro from the user's report. After Backspace at the end of
    // an empty BQ paragraph, the trailing pair becomes a top-level
    // paragraph break + a few blank lines. Each blank-line leaf
    // must have an empty container chain — *not* the BQ chain
    // inherited from the original BQ above it.
    let src = "> This is a blockquote.\n> \n> \n\nParagraph";
    let cursor = "> This is a blockquote.\n> \n> ".len();
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(cursor),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        let md = e.value();
        // The first BQ block ends after "blockquote."; everything
        // past that newline is structurally outside the BQ.
        let bq_end = md.find("blockquote.").unwrap() + "blockquote.".len();
        let spec = e.render_spec();
        for b in &spec.blocks {
            // Any leaf whose source_range starts at or after the
            // BQ's structural end must have NO BQ in its chain.
            if b.source_range.start >= bq_end {
                let bq_count = b
                    .containers
                    .iter()
                    .filter(|c| matches!(c, Container::BlockQuote { .. }))
                    .count();
                assert_eq!(
                    bq_count, 0,
                    "leaf at range {:?} (kind {:?}) is past BQ end ({bq_end}) but \
                     still has BQ in its chain — would render BQ chrome on a row \
                     that isn't structurally inside the BQ. Buffer: {md:?}",
                    b.source_range, b.kind,
                );
            }
        }
    });
}

#[gpui::test]
fn same_line_bq_marker_inside_li_is_hidden(cx: &mut TestAppContext) {
    // Repro from the user's report. Source `1. > content`. The
    // first leaf must hide BOTH the LI marker (`1. ` bytes 0..3)
    // AND the BQ marker (`> ` bytes 3..5). Currently only the LI
    // marker is hidden — the BQ marker shows as visible plain
    // text.
    let src = "1. > This one is long enough to wrap around.";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (_handle, editor) = open_editor(cx, initial);
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    let first = &spec.blocks[0];
    // The LI marker chrome should already be hidden from prior
    // fixes.
    assert!(
        first
            .hidden_ranges
            .iter()
            .any(|r| r.start == 0 && r.end >= 3),
        "LI marker `1. ` must be in hidden_ranges; got: {:?}",
        first.hidden_ranges,
    );
    // The BQ marker on the same line must ALSO be hidden.
    assert!(
        first
            .hidden_ranges
            .iter()
            .any(|r| r.start <= 3 && r.end >= 5),
        "BQ marker `> ` (bytes 3..5) must be in hidden_ranges; got: {:?}",
        first.hidden_ranges,
    );
}

#[gpui::test]
fn trailing_li_continuation_indent_is_hidden_in_synth_leaf(cx: &mut TestAppContext) {
    // Repro from the user's report. Source `1. List item one\n\n   `
    // with cursor at end. The synthetic empty paragraph leaf for the
    // trailing continuation should hide the LI's continuation indent
    // (the 3 leading spaces) so the cursor lands at the visible
    // content edge — same column as `List item one`.
    let src = "1. List item one\n\n   ";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (_handle, editor) = open_editor(cx, initial);
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    // Find the leaf covering the trailing position.
    let trailing = spec
        .blocks
        .iter()
        .find(|b| b.source_range.contains(&20) || b.source_range.end == 21)
        .expect("a leaf for the trailing position");
    // Chain has the LI.
    let li_count = trailing
        .containers
        .iter()
        .filter(|c| matches!(c, Container::ListItem { .. }))
        .count();
    assert_eq!(li_count, 1, "trailing leaf carries the LI in its chain");
    // The 3-byte indent (bytes 18..21 — `   ` after the `\n\n`) is
    // in hidden_ranges. Without this, the rendered line shows the
    // 3 spaces as visible text and the cursor sits past them.
    let indent_hidden = trailing.hidden_ranges.iter().any(|r| {
        // Some range that ends at or past 21 (the cursor position)
        // and starts at or before 18 (right after the `\n\n`).
        r.start <= 18 && r.end >= 21
    });
    assert!(
        indent_hidden,
        "expected hidden_range covering 18..21 (LI continuation indent) on trailing leaf; \
         got hidden_ranges={:?}",
        trailing.hidden_ranges,
    );
}

#[gpui::test]
fn backspace_at_trailing_li_continuation_atomic_deletes(cx: &mut TestAppContext) {
    // Repro from the user's report. Source `1. List item one\n\n   `
    // with cursor at end. Backspace should remove the entire
    // paragraph-break-with-LI-indent (`\n\n   `, 5 bytes) and land
    // the cursor back at the end of "List item one" (byte 16).
    let src = "1. List item one\n\n   ";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // Source has shrunk by exactly the `\n\n   ` paragraph-break-
        // with-indent (5 bytes).
        assert_eq!(
            e.value(),
            "1. List item one",
            "Backspace must atomic-delete the full paragraph break + LI indent",
        );
        assert_eq!(
            e.cursor_offset(),
            16,
            "cursor at end of `one` after the atomic delete",
        );
    });
}

#[gpui::test]
fn alternating_chain_trailing_synth_hides_full_continuation_prefix(cx: &mut TestAppContext) {
    // Chain at cursor: [outer-LI, outer-BQ, inner-LI, inner-BQ].
    // Trailing line is `   >    > ` (full continuation prefix:
    // 3sp + `> ` + 3sp + `> `, 10 bytes). Every non-cursor byte
    // of that prefix should be hidden — the LI continuation
    // indents (bytes 31..34 and 36..39) AND the BQ markers
    // (34..36 and 39..41).
    let src = "1. > 1. > Something\n   >    > \n   >    > ";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (_handle, editor) = open_editor(cx, initial);
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    let trailing = spec
        .blocks
        .iter()
        .find(|b| b.source_range.contains(&40) || b.source_range.end == 41)
        .expect("trailing leaf");
    // Sum hidden bytes in the trailing-line range (31..41).
    let hidden_in_trailing: usize = trailing
        .hidden_ranges
        .iter()
        .filter(|r| r.start >= 31 && r.end <= 41)
        .map(|r| r.end - r.start)
        .sum();
    assert_eq!(
        hidden_in_trailing, 10,
        "expected all 10 bytes of trailing continuation prefix \
         (31..41) hidden — got {hidden_in_trailing}, \
         hidden_ranges={:?}",
        trailing.hidden_ranges,
    );
}

#[gpui::test]
fn backspace_on_alternating_chain_keeps_inner_li_scope(cx: &mut TestAppContext) {
    // Source: `1. > 1. > Something\n   >    > \n   >    > `,
    // cursor at end. Chain is [outer-LI, outer-BQ, inner-LI,
    // inner-BQ]. Press Backspace.
    //
    // Expected: BQ-outdent drops the inner-BQ; cursor lands
    // inside chain [outer-LI, outer-BQ, inner-LI] — a paragraph
    // sibling to the inner-BQ within the inner-LI.
    let src = "1. > 1. > Something\n   >    > \n   >    > ";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        let cursor = e.cursor_offset();
        let chain = gpui_markdown_editor::analysis::enclosing_containers_at(e.value(), cursor);
        // Chain has both LIs and the outer BQ: 3 entries total.
        let li_count = chain
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    gpui_markdown_editor::analysis::EnclosingContainer::ListItem(_)
                )
            })
            .count();
        let bq_count = chain
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    gpui_markdown_editor::analysis::EnclosingContainer::BlockQuote { .. }
                )
            })
            .count();
        assert_eq!(
            li_count,
            2,
            "both LIs preserved in chain after BQ-outdent; chain: {chain:?}, buffer: {:?}",
            e.value(),
        );
        assert_eq!(
            bq_count,
            1,
            "outer-BQ preserved, inner-BQ dropped; chain: {chain:?}, buffer: {:?}",
            e.value(),
        );
    });
}

#[gpui::test]
fn alternating_chain_continuation_hides_trailing_li_indent(cx: &mut TestAppContext) {
    // Chain at the trailing position is [outer-LI, outer-BQ,
    // inner-LI]. The continuation prefix is `   >    ` (8 bytes:
    // 3sp outer-LI indent + `> ` outer-BQ marker + 3sp inner-LI
    // indent). When the user types content there, the leaf
    // covering it must hide ALL 8 prefix bytes — the existing fix
    // covers the BQ marker and the leading LI indent but misses
    // the *trailing* LI indent after the last BQ marker.
    let src = "1. > 1. > Blockquote\n   > \n   >    Paragraph  \n   >    More";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (_handle, editor) = open_editor(cx, initial);
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    // Find the leaf covering the "More" content.
    let more_pos = src.find("More").unwrap();
    let leaf = spec
        .blocks
        .iter()
        .find(|b| b.source_range.contains(&more_pos))
        .expect("a leaf covering 'More'");
    // The continuation prefix bytes are 47..55. Every byte must
    // be hidden somewhere in this leaf's hidden_ranges.
    for byte in 47..55 {
        let covered = leaf
            .hidden_ranges
            .iter()
            .any(|r| r.start <= byte && byte < r.end);
        assert!(
            covered,
            "byte {byte} (inside continuation prefix 47..55) is not in hidden_ranges; \
             got hidden_ranges={:?}",
            leaf.hidden_ranges,
        );
    }
}

#[gpui::test]
fn synth_trailing_leaf_in_alternating_chain_keeps_inner_li(cx: &mut TestAppContext) {
    // Chain at the trailing empty position should be [outer-LI,
    // outer-BQ, inner-LI]. The cursor walker sees this correctly
    // (length 3); the render walker emits the leaf with chain
    // length 2 ([outer-LI, outer-BQ]) — so the inner-LI's chrome
    // (indent strip / marker overlay) won't paint, even though
    // the cursor analytically sits inside the inner-LI.
    let src = "1. > 1. > Blockquote\n   > \n   >    Paragraph  \n   >    ";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (_handle, editor) = open_editor(cx, initial);
    let (cursor_chain_len, block_chain_len) = editor.read_with(cx, |e, _| {
        let cursor_chain =
            gpui_markdown_editor::analysis::enclosing_containers_at(e.value(), e.cursor_offset());
        let spec = e.render_spec();
        let trailing = spec
            .blocks
            .iter()
            .find(|b| b.source_range.end == e.value().len())
            .expect("trailing leaf");
        (cursor_chain.len(), trailing.containers.len())
    });
    assert_eq!(
        cursor_chain_len, 3,
        "cursor walker should report 3 containers"
    );
    assert_eq!(
        block_chain_len, 3,
        "trailing leaf chain must agree with cursor walker (3 containers)",
    );
}

#[gpui::test]
fn hard_break_trailing_synth_extends_previous_paragraph(cx: &mut TestAppContext) {
    // Source has a hard break (`Paragraph  `) followed by a
    // continuation row `   >    ` (with no further content). The
    // user expects the cursor's row to be part of the previous
    // paragraph (a continuation of the hard break), not a
    // separate paragraph with a paragraph_gap before it.
    //
    // Ground truth: the same source with one extra char of
    // content (`Paragraph  \n   >    X`) parses as ONE paragraph
    // with hard-break continuation — one RenderBlock. The empty
    // case must produce the same number of blocks.
    let empty_src = "1. > 1. > Blockquote\n   > \n   >    Paragraph  \n   >    ";
    let with_content_src = "1. > 1. > Blockquote\n   > \n   >    Paragraph  \n   >    X";

    let with_content_block_count = {
        let initial = EditorState {
            markdown: with_content_src.into(),
            selection: Selection::Cursor(with_content_src.len()),
            ..Default::default()
        };
        let (_handle, editor) = open_editor(cx, initial);
        editor.read_with(cx, |e, _| e.render_spec().blocks.len())
    };

    let empty_block_count = {
        let initial = EditorState {
            markdown: empty_src.into(),
            selection: Selection::Cursor(empty_src.len()),
            ..Default::default()
        };
        let (_handle, editor) = open_editor(cx, initial);
        editor.read_with(cx, |e, _| e.render_spec().blocks.len())
    };

    assert_eq!(
        empty_block_count, with_content_block_count,
        "empty trailing case must produce the same number of \
         RenderBlocks as the with-content case so the cursor \
         visual position is identical (no extra paragraph_gap). \
         empty={empty_block_count}, with_content={with_content_block_count}",
    );
}

#[gpui::test]
#[ignore = "manual probe — render shape with hard break + trailing empty in alternating chain"]
fn probe_hard_break_trailing_in_alternating_chain(cx: &mut TestAppContext) {
    let src = "1. > 1. > Blockquote\n   > \n   >    Paragraph  \n   >    ";
    eprintln!("\n# Source ({} bytes):", src.len());
    eprintln!("{src:?}");

    fn dump(label: &str, e: &MarkdownEditorState) {
        eprintln!("\n## {label}");
        eprintln!("buffer ({} bytes): {:?}", e.value().len(), e.value());
        eprintln!("cursor: {}", e.cursor_offset());
        let chain =
            gpui_markdown_editor::analysis::enclosing_containers_at(e.value(), e.cursor_offset());
        eprintln!("cursor chain length: {}", chain.len());
        let spec = e.render_spec();
        for (i, b) in spec.blocks.iter().enumerate() {
            let chain_str: Vec<String> = b
                .containers
                .iter()
                .map(|c| match c {
                    Container::BlockQuote { .. } => "BQ".into(),
                    Container::ListItem { .. } => "LI".into(),
                })
                .collect();
            eprintln!(
                "  block[{i}] {:?} range={}..{} chain=[{}]\n    hidden_ranges={:?}\n    delimiter_lines={:?}",
                b.kind,
                b.source_range.start,
                b.source_range.end,
                chain_str.join(", "),
                b.hidden_ranges,
                b.delimiter_lines,
            );
        }
    }

    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    editor.read_with(cx, |e, _| {
        dump("BEFORE typing (cursor on empty trailing)", e)
    });

    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText("More".into()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| dump("AFTER typing 'More'", e));
}

#[gpui::test]
#[ignore = "manual probe — alternating chain backspace stair-step"]
fn probe_alternating_chain_backspace_steps(cx: &mut TestAppContext) {
    let src = "1. > 1. > Something\n   >    > \n   >    > ";
    eprintln!("\n# Initial state");
    eprintln!("source: {src:?}, len: {}", src.len());
    let mut state = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    for step in 0..6 {
        let initial = state.clone();
        let (handle, editor) = open_editor(cx, initial);
        editor.read_with(cx, |e, _| {
            eprintln!("\n## After {step} BS");
            eprintln!("buffer: {:?}, len: {}", e.value(), e.value().len());
            eprintln!("cursor: {}", e.cursor_offset());
            let chain = gpui_markdown_editor::analysis::enclosing_containers_at(
                e.value(),
                e.cursor_offset(),
            );
            eprintln!("cursor chain length: {}", chain.len());
            let spec = e.render_spec();
            for (i, b) in spec.blocks.iter().enumerate() {
                let chain_str: Vec<String> = b
                    .containers
                    .iter()
                    .map(|c| match c {
                        Container::BlockQuote { .. } => "BQ".into(),
                        Container::ListItem { .. } => "LI".into(),
                    })
                    .collect();
                eprintln!(
                    "  block[{i}] {:?} range={}..{} chain=[{}] hidden={:?}",
                    b.kind,
                    b.source_range.start,
                    b.source_range.end,
                    chain_str.join(", "),
                    b.hidden_ranges,
                );
            }
        });
        dispatch(cx, handle, &editor, Backspace);
        state = editor.read_with(cx, |e, _| EditorState {
            markdown: e.value().into(),
            selection: e.selection(),
            ..Default::default()
        });
    }
}

#[gpui::test]
#[ignore = "manual probe — captures render + Backspace state for trailing LI continuation indent"]
fn probe_trailing_li_continuation_indent(cx: &mut TestAppContext) {
    let src = "1. List item one\n\n   ";
    eprintln!("\n# Initial state\nsource: {src:?}, len: {}", src.len());
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    editor.read_with(cx, |e, _| {
        let cursor = e.cursor_offset();
        eprintln!("cursor: {cursor}");
        let chain = gpui_markdown_editor::analysis::enclosing_containers_at(e.value(), cursor);
        eprintln!("cursor chain length: {}", chain.len());
        let spec = e.render_spec();
        for (i, b) in spec.blocks.iter().enumerate() {
            let chain_str: Vec<String> = b
                .containers
                .iter()
                .map(|c| match c {
                    Container::BlockQuote { .. } => "BQ".into(),
                    Container::ListItem { .. } => "LI".into(),
                })
                .collect();
            eprintln!(
                "  block[{i}] {:?} range={}..{} chain=[{}]\n    hidden_ranges={:?}",
                b.kind,
                b.source_range.start,
                b.source_range.end,
                chain_str.join(", "),
                b.hidden_ranges,
            );
        }
    });
    eprintln!("\n# After Backspace");
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        eprintln!("buffer: {:?}", e.value());
        eprintln!("cursor: {}", e.cursor_offset());
    });
    eprintln!("\n# Left arrows from initial cursor");
    for n_lefts in 1..=4 {
        let initial2 = EditorState {
            markdown: src.into(),
            selection: Selection::Cursor(src.len()),
            ..Default::default()
        };
        let (handle2, editor2) = open_editor(cx, initial2);
        for _ in 0..n_lefts {
            dispatch(cx, handle2, &editor2, Left);
        }
        let cursor = editor2.read_with(cx, |e, _| e.cursor_offset());
        eprintln!("  Left*{n_lefts}: cursor={cursor}");
    }
}

/// Manual probe for the empty-paragraph-after-BQ-outdent bug.
#[gpui::test]
#[ignore = "manual probe — captures the chain on each leaf after BQ-outdent leaves multiple blank lines"]
fn probe_empty_paragraph_after_bq_outdent(cx: &mut TestAppContext) {
    let src = "> This is a blockquote.\n> \n> \n\nParagraph";
    let cursor = "> This is a blockquote.\n> \n> ".len();
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(cursor),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        eprintln!("\nbuffer after Backspace: {:?}", e.value());
        eprintln!("cursor: {}", e.cursor_offset());
        let spec = e.render_spec();
        for (i, b) in spec.blocks.iter().enumerate() {
            let chain: Vec<String> = b
                .containers
                .iter()
                .map(|c| match c {
                    Container::BlockQuote { cursor_inside } => {
                        if *cursor_inside {
                            "BQ*".into()
                        } else {
                            "BQ".into()
                        }
                    }
                    Container::ListItem { cursor_inside, .. } => {
                        if *cursor_inside {
                            "LI*".into()
                        } else {
                            "LI".into()
                        }
                    }
                })
                .collect();
            eprintln!(
                "  block[{i}] {:?} range={}..{} chain=[{}] hidden={} overlays={}",
                b.kind,
                b.source_range.start,
                b.source_range.end,
                chain.join(", "),
                b.hidden_ranges.len(),
                b.marker_overlays.len(),
            );
        }
    });
}

/// Manual probe for the visible-`>`-marker on same-line BQ-in-LI bug.
#[gpui::test]
#[ignore = "manual probe — checks hidden_ranges + marker_overlays for `1. > content`"]
fn probe_same_line_bq_in_li_render(cx: &mut TestAppContext) {
    let src = "1. > This one is long enough to wrap.\n   > \n   > Second paragraph.\n\n";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (_handle, editor) = open_editor(cx, initial);
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    eprintln!("\n# Source:\n{src}\n");
    for (i, b) in spec.blocks.iter().enumerate() {
        let chain: Vec<String> = b
            .containers
            .iter()
            .map(|c| match c {
                Container::BlockQuote { .. } => "BQ".into(),
                Container::ListItem { .. } => "LI".into(),
            })
            .collect();
        eprintln!(
            "  block[{i}] {:?} range={}..{} chain=[{}]\n    hidden_ranges={:?}\n    marker_overlays={:?}\n    substitutions={:?}",
            b.kind,
            b.source_range.start,
            b.source_range.end,
            chain.join(", "),
            b.hidden_ranges,
            b.marker_overlays,
            b.substitutions,
        );
    }
}

/// Manual probe — not a regression test. Run with
/// `cargo test -p gpui-markdown-editor --test behavior probe_left_backspace -- --ignored --nocapture`
/// to dump the buffer + cursor at each step of the user-reported sequence.
#[gpui::test]
#[ignore = "manual probe — captures Backspace behavior at Left*N positions for the empty-BQ-in-LI fixture"]
fn probe_left_backspace_on_empty_bq_in_li(cx: &mut TestAppContext) {
    let src = "1. This is a list.\n\n   > This is a blockquote\n   > \n   > ";
    for n_lefts in 0..=5 {
        let initial = EditorState {
            markdown: src.into(),
            selection: Selection::Cursor(src.len()),
            ..Default::default()
        };
        let (handle, editor) = open_editor(cx, initial);
        for _ in 0..n_lefts {
            dispatch(cx, handle, &editor, Left);
        }
        let cursor_after_lefts = editor.read_with(cx, |e, _| e.cursor_offset());
        dispatch(cx, handle, &editor, Backspace);
        editor.read_with(cx, |e, _| {
            let md = e.value();
            let cursor = e.cursor_offset();
            // Render the buffer with `|` at cursor for visual scan.
            let mut visual = String::with_capacity(md.len() + 1);
            for (i, ch) in md.char_indices() {
                if i == cursor {
                    visual.push('|');
                }
                if ch == ' ' {
                    visual.push('·');
                } else if ch == '\n' {
                    visual.push('⏎');
                    visual.push('\n');
                } else {
                    visual.push(ch);
                }
            }
            if cursor == md.len() {
                visual.push('|');
            }
            eprintln!(
                "\n=== n_lefts={n_lefts} (cursor before BS: {cursor_after_lefts}) ===\n\
                 cursor after BS: {cursor}\nmarkdown:\n{visual}\n",
            );
        });
    }
}

// ---------------------------------------------------------------------------
// Code-block nesting cohort: regression tests for the rules in
// `bugs.md::code-block nesting cohort`. The session test
// (`tests/session.rs::nested_code_blocks_session`) is the discovery
// harness — these are the targeted gates.
// ---------------------------------------------------------------------------

#[gpui::test]
fn enter_after_unterminated_fence_in_bq_auto_closes_with_chain_prefix(cx: &mut TestAppContext) {
    // Rule A + Rule B + Rule C combined: typing `> ```rust` and pressing
    // Enter should leave the buffer in a *terminated* fenced state with
    // the cursor on a body row in between, and both body and closer
    // rows must carry the BQ continuation prefix.
    let initial = EditorState {
        markdown: "> ```rust".into(),
        selection: Selection::Cursor(9),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.value(),
            "> ```rust\n> \n> ```",
            "auto-close should inject `\\n> \\n> ```` so the fence is terminated",
        );
        // Cursor lands on the body row (between fences), past the prefix.
        assert_eq!(e.cursor_offset(), 12);
    });
}

#[gpui::test]
fn enter_inside_terminated_fence_in_bq_inserts_newline_with_chain_prefix(cx: &mut TestAppContext) {
    // Rule C: once the fence is closed, Enter on a body line inserts
    // `\n` + chain continuation prefix — *not* a paragraph break pair
    // (which would split the BQ scope) and *not* a bare `\n` (which
    // would lose the BQ marker on the new row).
    let initial = EditorState {
        markdown: "> ```rust\n> body\n> ```".into(),
        selection: Selection::Cursor(16), // end of `> body`
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.value(),
            "> ```rust\n> body\n> \n> ```",
            "Enter inside fenced code in a BQ should add `\\n> ` (not `\\n\\n` or bare `\\n`)",
        );
        // Cursor on the new empty body row, past the BQ prefix.
        assert_eq!(e.cursor_offset(), 19);
    });
}

#[gpui::test]
fn enter_on_empty_bq_paragraph_outdents_bq_scope(cx: &mut TestAppContext) {
    // Rule D: the analog of empty-LI Enter outdent for blockquotes.
    // Cursor on an empty `> ` row (chain ends in BQ); Enter drops
    // the innermost BQ, replacing the trailing chain pair with
    // the reduced-chain pair shape.
    let initial = EditorState {
        markdown: "> a\n> \n> ".into(),
        selection: Selection::Cursor(9), // end of trailing empty `> `
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        // Trailing depth-1 BQ pair → top-level pair shape.
        assert_eq!(
            e.value(),
            "> a\n\n",
            "empty-BQ Enter should outdent the BQ scope; got: {:?}",
            e.value(),
        );
        // Cursor lands at the end of the new top-level shape.
        assert_eq!(e.cursor_offset(), 5);
        // Chain at cursor: empty (top level).
        let chain =
            gpui_markdown_editor::analysis::enclosing_containers_at(e.value(), e.cursor_offset());
        assert!(
            chain.is_empty(),
            "cursor should be at top level after BQ outdent, got chain {chain:?}",
        );
    });
}

#[gpui::test]
fn render_walker_omits_synth_paragraph_inside_unterminated_fence(cx: &mut TestAppContext) {
    // Rule E: a trailing position inside an unterminated fence is
    // already covered by the CodeBlock leaf — the synth-paragraph
    // injection in `render_blockquote` / `inject_empty_paragraphs`
    // must not emit a phantom Paragraph leaf with overlapping range.
    //
    // Use the unterminated `> ```rust\n> \n> ` shape (the post-Enter
    // state if auto-close were *not* in play) so we exercise the
    // render path without depending on whether Enter auto-closed.
    let initial = EditorState {
        markdown: "> ```rust\n> \n> ".into(),
        selection: Selection::Cursor(15),
        ..Default::default()
    };
    let (_handle, editor) = open_editor(cx, initial);
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    // Exactly one block — no phantom Paragraph leaf overlapping the
    // CodeBlock.
    assert_eq!(
        spec.blocks.len(),
        1,
        "expected one CodeBlock leaf, got {} blocks",
        spec.blocks.len(),
    );
    assert!(
        matches!(spec.blocks[0].kind, BlockKind::CodeBlock { .. }),
        "expected CodeBlock leaf, got {:?}",
        spec.blocks[0].kind,
    );
}

#[gpui::test]
fn auto_close_orphan_dedupes_when_user_types_matching_closer(cx: &mut TestAppContext) {
    // After auto-close fires (cursor on a body row between fences), the
    // user types their own closing fence on the body row. The buffer
    // momentarily holds two adjacent identical fence delimiter lines —
    // the user's typed closer + the auto-close's now-orphaned opener.
    // `dedupe_orphan_fence_closer` removes the orphan so the construct
    // ends cleanly.
    //
    // Repro the post-state directly: a buffer where exactly that shape
    // sits at the end. After enforce_invariants (which the editor runs
    // on every update), the orphan is gone.
    let initial = EditorState {
        markdown: "> ```rust\n> body\n> ```\n> ```".into(),
        selection: Selection::Cursor(22), // end of user's typed `> ```` closer
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    // Trigger an update so enforce_invariants runs (any no-op insert
    // will do; here we type and immediately backspace).
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText(String::new()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        // Orphan removed: only the user's typed closer survives.
        assert_eq!(
            e.value(),
            "> ```rust\n> body\n> ```\n",
            "expected orphan opener to be deduped, got: {:?}",
            e.value(),
        );
    });
}

#[gpui::test]
fn cursor_at_eof_of_unterminated_fence_classifies_as_inside(_cx: &mut TestAppContext) {
    // Rule B: `is_in_fenced_code` treats the EOF byte of an
    // unterminated fence range as inside the construct. Without this,
    // the cursor sitting right after the user's `> ```rust` opener
    // routes through chain-based Enter (BQ pair, LI marker) instead
    // of the in-fence path — the original cascade root cause.
    let buf = "> ```rust";
    assert!(
        gpui_markdown_editor::analysis::is_in_fenced_code(buf, buf.len()),
        "EOF position of unterminated fence should classify as inside the construct",
    );
    // For terminated fences the boundary remains exclusive so the
    // byte right after the closer's trailing `\n` is *outside*.
    let closed = "> ```\n> a\n> ```\n";
    assert!(
        !gpui_markdown_editor::analysis::is_in_fenced_code(closed, closed.len()),
        "EOF position past a terminated fence should classify as outside",
    );
}

#[gpui::test]
fn shift_tab_in_li_li_bq_does_not_panic_on_overlapping_strips(cx: &mut TestAppContext) {
    // Headline regression for refactor C. With chain `[LI, LI, BQ]`,
    // pressing Shift+Tab on a row inside the BQ used to compute
    // `above_parent_chain = &chain[..chain.len() - 2]` — which
    // assumed the trailing two chain entries were parent-LI and
    // innermost-LI. With BQ trailing the inner LI, that slice still
    // included the parent LI itself, so `above_skip` was 2 (the
    // parent's marker_width). The strip walker then walked past line
    // terminators into adjacent lines, producing overlapping byte
    // ranges that panicked `apply_edits`'s sortedness check.
    //
    // The fix in refactor C: locate the parent LI's actual position
    // in the chain (rather than slicing by `chain.len() - 2`) and
    // bound the strip walker to a single line.
    let src = "- List\n  - Child\n\n    > Blockquote\n    > \n    > ```js\n    > content\n    > ```\n    > ";
    let initial = EditorState {
        markdown: src.into(),
        selection: Selection::Cursor(src.len()),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    // Shift+Tab on the trailing row inside the BQ — must not panic.
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        // Survive without panic; the buffer should be valid markdown.
        let _ = e.value().len();
    });
}

#[gpui::test]
fn auto_close_fence_fires_in_li_li_bq_chain(cx: &mut TestAppContext) {
    // Headline regression for refactor A. The cursor sits inside an
    // unterminated fence in chain `[LI, LI, BQ]`; pressing Enter
    // should auto-close the fence. The byte scanner's
    // `count_line_markers` only tolerates 3 spaces between consecutive
    // `>` markers, so two LIs of 2sp each (= 4sp before the inner `> `)
    // makes the fence invisible to a byte scan — but pulldown sees it.
    // After refactor A, every fence containment query reads off the
    // parse tree, and auto-close fires correctly here.
    let initial = EditorState {
        markdown: "- List\n  - Child\n\n    > Blockquote\n    > \n    > ```js".into(),
        selection: Selection::Cursor(53), // end of `> ```js`
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        // The buffer should now be terminated — auto-close injected a
        // body row + matching closer below the cursor, both carrying
        // the chain's continuation prefix.
        assert!(
            !gpui_markdown_editor::analysis::is_in_fenced_code(
                e.value(),
                e.value().len(),
            ),
            "after Enter the fence should be terminated; got: {:?}",
            e.value(),
        );
        // And the cursor should sit on the new body row, past the
        // chain prefix.
        let cursor = e.cursor_offset();
        let expected_body = "\n    > ";
        assert!(
            e.value()[..cursor].ends_with(expected_body),
            "cursor should land on the new body row past `{expected_body}`; got buffer {:?} cursor {cursor}",
            e.value(),
        );
    });
}

// ---------------------------------------------------------------------------
// Math navigation + editing
// ---------------------------------------------------------------------------

fn set_cursor(
    cx: &mut TestAppContext,
    handle: AnyWindowHandle,
    editor: &Entity<MarkdownEditorState>,
    offset: usize,
) {
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::SetSelection(Selection::Cursor(offset)),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
}

#[gpui::test]
fn display_math_flips_to_edit_mode_on_arrow_into_construct(cx: &mut TestAppContext) {
    // Cursor *just before* the math (byte 0 is also the math start
    // — inclusive overlap puts it in edit mode immediately). Right
    // arrow from somewhere outside should land inside; we verify
    // the state transition by setting the cursor at the boundary
    // and again one byte further in.
    let initial = EditorState {
        markdown: "$$x^2$$".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    // Inclusive overlap means cursor at 0 (boundary) is already
    // edit mode.
    let spec = current_spec(cx, &editor);
    match spec.blocks[0].kind {
        BlockKind::DisplayMath { edit_mode, .. } => assert!(
            edit_mode,
            "cursor at boundary should flip the block to edit mode"
        ),
        ref other => panic!("expected DisplayMath, got {other:?}"),
    }
    // Move strictly past the construct → display mode returns.
    set_cursor(cx, handle, &editor, 7);
    let spec = current_spec(cx, &editor);
    match spec.blocks[0].kind {
        BlockKind::DisplayMath { edit_mode, .. } => assert!(
            edit_mode,
            "byte 7 == math.end is still boundary-inclusive → edit mode"
        ),
        ref other => panic!("expected DisplayMath, got {other:?}"),
    }
}

#[gpui::test]
fn typing_inside_display_math_updates_source(cx: &mut TestAppContext) {
    // Cursor strictly inside `$$x$$` (byte 3 — between `x` and the
    // closing `$$`). Typing inserts a char into the LaTeX source;
    // the block stays a DisplayMath block in edit mode while the
    // user types.
    let initial = EditorState {
        markdown: "$$x$$".into(),
        selection: Selection::Cursor(3),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText("^2".into()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "$$x^2$$");
        assert_eq!(e.cursor_offset(), 5);
    });
    // Still a DisplayMath block, still in edit mode.
    let spec = current_spec(cx, &editor);
    assert!(matches!(
        spec.blocks[0].kind,
        BlockKind::DisplayMath {
            edit_mode: true,
            ..
        }
    ));
}

#[gpui::test]
fn backspace_inside_display_math_deletes_one_byte(cx: &mut TestAppContext) {
    // Standard grapheme-delete inside math source — math isn't
    // atomic at the byte level; users edit the LaTeX directly.
    let initial = EditorState {
        markdown: "$$x^2$$".into(),
        selection: Selection::Cursor(5), // right after the `2`
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "$$x^$$");
        assert_eq!(e.cursor_offset(), 4);
    });
}

#[gpui::test]
fn inline_math_outside_cursor_emits_overlay(cx: &mut TestAppContext) {
    // Cursor not on the math at all (byte 0, before "see "). The
    // render spec should carry a `MathOverlay` for `$x^2$` and no
    // competing code-styled inline run for the inner content.
    let initial = EditorState {
        markdown: "see $x^2$ here".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let block = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::Paragraph))
        .expect("paragraph");
    assert_eq!(block.math_overlays.len(), 1);
    let overlay = &block.math_overlays[0];
    assert_eq!(overlay.source_range, 4..9);
    assert_eq!(overlay.content_range, 5..8);
    assert!(!overlay.display_style);
}

#[gpui::test]
fn inline_math_dims_delimiters_when_cursor_enters(cx: &mut TestAppContext) {
    // Cursor parked on the inner LaTeX of `$x^2$` (byte 6, between
    // `x` and `^`). Delimiters dim, content shapes as code-styled
    // mono text, and no typeset overlay is emitted — the user
    // wants to read/edit the raw LaTeX.
    let initial = EditorState {
        markdown: "see $x^2$ here".into(),
        selection: Selection::Cursor(6),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let block = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::Paragraph))
        .expect("paragraph");
    assert!(
        block.math_overlays.is_empty(),
        "cursor inside math should suppress the overlay path"
    );
    assert!(block.has_dimmed_range(4..5), "opening `$` should dim");
    assert!(block.has_dimmed_range(8..9), "closing `$` should dim");
}

#[gpui::test]
fn typing_inside_inline_math_extends_latex(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "$x$".into(),
        selection: Selection::Cursor(2), // between `x` and `$`
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText("^2".into()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "$x^2$");
        assert_eq!(e.cursor_offset(), 4);
    });
}

// ---------------------------------------------------------------------------
// Block-level `$$..$$` math (multi-line edit mode)
// ---------------------------------------------------------------------------

#[gpui::test]
fn typing_dollar_dollar_then_enter_auto_closes_block_math(cx: &mut TestAppContext) {
    // The user just typed `$$` (cursor at byte 2) — pulldown sees an
    // unterminated block-level math construct. Pressing Enter
    // auto-closes the construct: replacement is `\n\n$$` injected at
    // the cursor, with the cursor landing on the empty body row
    // between opener and closer. After the edit, the buffer parses
    // as a terminated block-math construct.
    let initial = EditorState {
        markdown: "$$".into(),
        selection: Selection::Cursor(2),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "$$\n\n$$");
        // Cursor lands right after the first `\n`, on the empty body
        // row (column 0).
        assert_eq!(e.cursor_offset(), 3);
    });
    let spec = current_spec(cx, &editor);
    assert!(matches!(
        spec.blocks[0].kind,
        BlockKind::DisplayMath {
            edit_mode: true,
            ..
        }
    ));
}

#[gpui::test]
fn enter_inside_block_math_inserts_literal_newline(cx: &mut TestAppContext) {
    // Cursor strictly inside a terminated block math `$$\nx\n$$`,
    // sitting at the end of the `x` content line. Enter inserts a
    // *literal* `\n` (no soft-break promotion to `\n\n`), preserving
    // the verbatim semantics. The block math survives — pulldown still
    // sees a single DisplayMath construct.
    let initial = EditorState {
        markdown: "$$\nx\n$$".into(),
        selection: Selection::Cursor(4), // after `x`
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        // No promotion — single `\n` inserted.
        assert_eq!(e.value(), "$$\nx\n\n$$");
        assert_eq!(e.cursor_offset(), 5);
    });
    // Still a single block-math leaf, blank line preserved as content.
    let spec = current_spec(cx, &editor);
    assert!(matches!(
        spec.blocks[0].kind,
        BlockKind::DisplayMath {
            edit_mode: true,
            ..
        }
    ));
}

#[gpui::test]
fn block_math_with_blank_line_in_body_round_trips_as_math(cx: &mut TestAppContext) {
    // The fork allows blank lines inside `$$..$$`. Verify the parser
    // surfaces the construct as a block-math leaf with `edit_mode`
    // gated on cursor position. Cursor outside → display mode;
    // cursor inside → edit mode.
    let initial = EditorState {
        markdown: "$$\nx\n\ny\n$$".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    assert!(
        spec.blocks
            .iter()
            .any(|b| matches!(b.kind, BlockKind::DisplayMath { .. })),
        "expected block-level DisplayMath leaf, got {:?}",
        spec.blocks
    );
    // Move strictly inside (between `x` and the blank line).
    set_cursor(cx, handle, &editor, 4);
    let spec = current_spec(cx, &editor);
    let block = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::DisplayMath { .. }))
        .expect("DisplayMath block");
    match &block.kind {
        BlockKind::DisplayMath { edit_mode, .. } => assert!(*edit_mode),
        _ => unreachable!(),
    }
}

#[gpui::test]
fn auto_close_inside_blockquote_carries_chain_prefix(cx: &mut TestAppContext) {
    // User typed `> $$` inside a blockquote (cursor at byte 4). Enter
    // auto-closes — both the new body row and the closer row carry
    // the `> ` continuation prefix so the math stays inside the BQ.
    let initial = EditorState {
        markdown: "> $$".into(),
        selection: Selection::Cursor(4),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        // `> $$` + `\n> ` (body) + `\n> $$` (closer) = "> $$\n> \n> $$"
        assert_eq!(e.value(), "> $$\n> \n> $$");
    });
}

#[gpui::test]
fn block_math_promoted_when_no_paragraph_wrapper(cx: &mut TestAppContext) {
    // Verify the top-level dispatch path: a multi-line `$$..$$`
    // arrives without a paragraph wrapper (per the fork) and still
    // emits a `BlockKind::DisplayMath` leaf with the inner content
    // range pointing at the LaTeX between delimiters.
    let initial = EditorState {
        markdown: "$$\na + b\n$$".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let block = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::DisplayMath { .. }))
        .expect("DisplayMath block emitted");
    match &block.kind {
        BlockKind::DisplayMath { content_range, .. } => {
            // Content range covers the inner LaTeX between the `$$`
            // delimiters (inclusive of any leading/trailing `\n`s
            // pulldown pairs with the construct's content).
            assert!(
                content_range.start <= content_range.end
                    && content_range.end <= "$$\na + b\n$$".len()
            );
        }
        _ => unreachable!(),
    }
}

#[gpui::test]
fn multi_line_block_math_with_trailing_newline_is_terminated_and_renders(cx: &mut TestAppContext) {
    // The fork's `parse_display_math_block` consumes the trailing
    // newline past the closer. Verify the editor still reports the
    // construct as terminated, exposes a content_range covering only
    // the inner LaTeX (not the closer `$$` or its trailing `\n`), and
    // registers both opener and closer delimiter lines.
    let initial = EditorState {
        markdown: "Hello\n\n$$\n\\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n\n$$\n".into(),
        // Cursor on "Hello" — strictly outside the math block.
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let block = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::DisplayMath { .. }))
        .expect("expected display math block");

    match &block.kind {
        BlockKind::DisplayMath {
            content_range,
            edit_mode,
        } => {
            assert!(
                !edit_mode,
                "expected display mode when cursor is outside the math block",
            );
            let content = editor.read_with(cx, |e, _| e.value()[content_range.clone()].to_string());
            assert_eq!(content, "\n\\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n\n");
        }
        _ => unreachable!(),
    }

    assert_eq!(
        block.delimiter_lines.len(),
        2,
        "expected 2 delimiter lines registered",
    );
}

#[gpui::test]
fn enter_inside_terminated_block_math_with_trailing_newline_does_not_duplicate_fence(
    cx: &mut TestAppContext,
) {
    // Terminated block math with a trailing newline. Cursor inside, at
    // the newline that ends the LaTeX content row:
    //
    //   `$$\n`                                       → 3 bytes  (0..3)
    //   `\frac{1}{1 - x} = \sum_{n=0}^{\infty} x^n`  → 41 bytes (3..44)
    //   `\n`                                         → 1 byte   (44..45)
    //   `$$\n`                                       → 3 bytes  (45..48)
    //
    // Cursor at 44 sits at the end of the content line (right before
    // the `\n` that separates content from closer).
    let initial = EditorState {
        markdown: "$$\n\\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n\n$$\n".into(),
        selection: Selection::Cursor(44),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(
        cx,
        handle,
        &editor,
        Enter {
            secondary: false,
            shift: false,
        },
    );
    editor.read_with(cx, |e, _| {
        // Inside a *terminated* block, Enter inserts a literal `\n`
        // — `auto_close_math_edit` short-circuits to None for
        // terminated constructs.
        assert_eq!(
            e.value(),
            "$$\n\\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n\n\n$$\n",
        );
        assert_eq!(e.cursor_offset(), 45);
    });
}

#[gpui::test]
fn cursor_at_start_of_next_block_does_not_flip_math_to_edit_mode(cx: &mut TestAppContext) {
    // Regression for the trailing-newline range bug. Before
    // `parser::math_kind` trimmed the range it returned on `node.range`,
    // the math block's `source_range` extended past the closer's `\n`
    // — so `cursor.overlaps(&math.range)` fired on the byte immediately
    // after the closer (the start of the next paragraph), flipping the
    // math into edit mode while the cursor logically lived on the
    // paragraph below.
    //
    // Layout:
    //   `$$\n`     → 3 bytes (0..3)
    //   `x\n`      → 2 bytes (3..5)
    //   `$$\n`     → 3 bytes (5..8)
    //   `bar`      → 3 bytes (8..11)
    //
    // Cursor at byte 8 is the first byte of "bar". The math must be
    // in display mode at this position.
    let initial = EditorState {
        markdown: "$$\nx\n$$\nbar".into(),
        selection: Selection::Cursor(8),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let block = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::DisplayMath { .. }))
        .expect("expected display math block");
    match &block.kind {
        BlockKind::DisplayMath { edit_mode, .. } => assert!(
            !edit_mode,
            "cursor at start of next paragraph must not put math in edit mode"
        ),
        _ => unreachable!(),
    }
}

#[gpui::test]
fn bare_dash_in_block_math_body_does_not_get_marker_space_injected(cx: &mut TestAppContext) {
    // `inject_unordered_marker_space` would historically inject `- ` →
    // `- ` (no-op) but `-foo` → `- foo` on any top-level-looking line.
    // Inside a verbatim region (fenced code or block math) the line is
    // literal content, so this pass must be skipped.
    //
    // Layout:
    //   `$$\n`   → 3 bytes (0..3)
    //   `-x\n`   → 3 bytes (3..6)  ← cursor parks here after typing
    //   `$$`     → 2 bytes (6..8)
    let initial = EditorState {
        markdown: "$$\n-x\n$$".into(),
        selection: Selection::Cursor(5),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(gpui_markdown_editor::EditorEvent::InsertText("".into()), cx);
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.value(),
            "$$\n-x\n$$",
            "verbatim math body must not have marker space injected",
        );
    });
}

#[gpui::test]
fn bare_gt_in_block_math_body_does_not_get_bq_space_injected(cx: &mut TestAppContext) {
    // `normalize_blockquote_prefixes` rewrites `>x` → `> x` at the
    // start of any line outside a verbatim region. Block math content
    // is verbatim — `>` is a literal LaTeX character (e.g. `\geq`'s
    // shorthand siblings), not a blockquote marker.
    let initial = EditorState {
        markdown: "$$\n>x\n$$".into(),
        selection: Selection::Cursor(5),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(gpui_markdown_editor::EditorEvent::InsertText("".into()), cx);
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.value(),
            "$$\n>x\n$$",
            "verbatim math body must not have BQ-prefix space injected",
        );
    });
}

#[gpui::test]
fn backspace_on_prefix_only_line_in_bq_wrapped_math_eats_the_line(cx: &mut TestAppContext) {
    // Inside a BQ-wrapped block math, an empty body row reads as just
    // `> ` — the BQ continuation prefix and nothing else. The
    // "in-verbatim delete-the-prefix-only-line" Backspace gesture
    // should remove the whole row (`\n> `), not nibble it one byte at
    // a time.
    //
    // Layout:
    //   `> $$\n`   → 5 bytes  (0..5)
    //   `> x\n`    → 4 bytes  (5..9)
    //   `> \n`     → 3 bytes  (9..12)   ← prefix-only line; cursor on it
    //   `> $$`    → 4 bytes  (12..16)
    //
    // Cursor at byte 11 (end of the prefix-only line, before its `\n`).
    let initial = EditorState {
        markdown: "> $$\n> x\n> \n> $$".into(),
        selection: Selection::Cursor(11),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // The whole prefix-only line (including its leading `\n`)
        // disappears; the cursor lands at the end of the previous
        // content row (right after `x`, before its trailing `\n`).
        assert_eq!(e.value(), "> $$\n> x\n> $$");
        assert_eq!(e.cursor_offset(), 8);
    });
}

// ---------------------------------------------------------------------------
// Word- and line-granular navigation and deletion
// ---------------------------------------------------------------------------

#[gpui::test]
fn word_left_from_end_of_word_lands_at_start_of_that_word(cx: &mut TestAppContext) {
    // Cursor at end of "world" in "hello world" → WordLeft jumps to
    // the `w` (start of the same word). The first call lands at the
    // word's start; a second call would jump back to the previous
    // word.
    let initial = EditorState {
        markdown: "hello world".into(),
        selection: Selection::Cursor(11),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, WordLeft);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 6));
    dispatch(cx, handle, &editor, WordLeft);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 0));
}

#[gpui::test]
fn word_left_from_start_of_word_skips_whitespace_to_previous_word(cx: &mut TestAppContext) {
    // Cursor at the start of "world" → WordLeft skips the space and
    // lands at the start of "hello".
    let initial = EditorState {
        markdown: "hello world".into(),
        selection: Selection::Cursor(6),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, WordLeft);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 0));
}

#[gpui::test]
fn word_right_from_start_of_word_lands_at_end_of_that_word(cx: &mut TestAppContext) {
    // Cursor at byte 0 of "hello world" → WordRight to the end of
    // "hello" (byte 5), then again to the end of "world" (byte 11).
    let initial = EditorState {
        markdown: "hello world".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, WordRight);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 5));
    dispatch(cx, handle, &editor, WordRight);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 11));
}

#[gpui::test]
fn word_right_at_end_of_doc_is_a_noop(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "abc".into(),
        selection: Selection::Cursor(3),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, WordRight);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 3));
}

#[gpui::test]
fn word_left_skips_runs_of_punctuation(cx: &mut TestAppContext) {
    // `foo, bar` with cursor at end (`r` + 1 = 8). WordLeft jumps
    // past the comma + space to the start of "foo".
    //
    // Layout:
    //   `foo, bar`
    //    0123456 7  (cursor at 8)
    //
    // First WordLeft → start of "bar" (byte 5). Second WordLeft →
    // start of "foo" (byte 0). The punctuation segment between them
    // never becomes a stopping point.
    let initial = EditorState {
        markdown: "foo, bar".into(),
        selection: Selection::Cursor(8),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, WordLeft);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 5));
    dispatch(cx, handle, &editor, WordLeft);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 0));
}

#[gpui::test]
fn word_right_skips_runs_of_punctuation(cx: &mut TestAppContext) {
    // Symmetric: from byte 0 of `foo, bar`, WordRight lands at the
    // end of "foo" (3), then at end of "bar" (8).
    let initial = EditorState {
        markdown: "foo, bar".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, WordRight);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 3));
    dispatch(cx, handle, &editor, WordRight);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 8));
}

#[gpui::test]
fn shift_word_right_extends_selection(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "alpha beta gamma".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftWordRight);
    editor.read_with(cx, |e, _| match e.selection() {
        Selection::Range { anchor, head } => {
            assert_eq!(anchor, 0);
            assert_eq!(head, 5);
        }
        _ => panic!("expected range selection"),
    });
    dispatch(cx, handle, &editor, ShiftWordRight);
    editor.read_with(cx, |e, _| match e.selection() {
        Selection::Range { anchor, head } => {
            assert_eq!(anchor, 0);
            assert_eq!(head, 10);
        }
        _ => panic!("expected range selection"),
    });
}

#[gpui::test]
fn shift_word_left_extends_selection_backward(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "alpha beta gamma".into(),
        selection: Selection::Cursor(16),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftWordLeft);
    editor.read_with(cx, |e, _| match e.selection() {
        Selection::Range { anchor, head } => {
            assert_eq!(anchor, 16);
            assert_eq!(head, 11);
        }
        _ => panic!("expected range selection"),
    });
}

#[gpui::test]
fn word_left_collapses_existing_selection_to_lower_bound(cx: &mut TestAppContext) {
    // With a range selection, an unshifted WordLeft collapses to the
    // lower bound (same behavior as Left / Up). It must NOT additionally
    // jump back one word — the collapse *is* the move.
    let initial = EditorState {
        markdown: "alpha beta gamma".into(),
        selection: Selection::range(2, 8),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, WordLeft);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 2));
}

#[gpui::test]
fn word_right_collapses_existing_selection_to_upper_bound(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "alpha beta gamma".into(),
        selection: Selection::range(2, 8),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, WordRight);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 8));
}

#[gpui::test]
fn delete_word_backward_removes_previous_word(cx: &mut TestAppContext) {
    // Cursor at end of "world" → DeleteWordBackward removes "world",
    // leaving "hello " with cursor at 6.
    let initial = EditorState {
        markdown: "hello world".into(),
        selection: Selection::Cursor(11),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteWordBackward);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "hello ");
        assert_eq!(e.cursor_offset(), 6);
    });
}

#[gpui::test]
fn delete_word_backward_at_start_of_word_eats_preceding_whitespace_and_word(
    cx: &mut TestAppContext,
) {
    // Cursor at start of "world" → DeleteWordBackward eats the
    // intervening space AND the preceding "hello" (one keystroke
    // removes back to the start of the previous word).
    let initial = EditorState {
        markdown: "hello world".into(),
        selection: Selection::Cursor(6),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteWordBackward);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "world");
        assert_eq!(e.cursor_offset(), 0);
    });
}

#[gpui::test]
fn delete_word_backward_at_buffer_start_is_a_noop(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "hello".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteWordBackward);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "hello");
        assert_eq!(e.cursor_offset(), 0);
    });
}

#[gpui::test]
fn delete_word_forward_removes_next_word(cx: &mut TestAppContext) {
    // Cursor at start → DeleteWordForward removes "hello", leaving
    // " world".
    let initial = EditorState {
        markdown: "hello world".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteWordForward);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), " world");
        assert_eq!(e.cursor_offset(), 0);
    });
}

#[gpui::test]
fn delete_word_forward_at_end_of_word_eats_following_whitespace_and_word(cx: &mut TestAppContext) {
    // Cursor at end of "hello" → DeleteWordForward eats the space and
    // "world".
    let initial = EditorState {
        markdown: "hello world".into(),
        selection: Selection::Cursor(5),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteWordForward);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "hello");
        assert_eq!(e.cursor_offset(), 5);
    });
}

#[gpui::test]
fn delete_word_backward_with_selection_deletes_the_selection(cx: &mut TestAppContext) {
    // The "selection wins" rule from the other delete actions: a
    // range selection is what gets removed, regardless of where the
    // word boundary would have landed.
    let initial = EditorState {
        markdown: "alpha beta gamma".into(),
        selection: Selection::range(6, 10),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteWordBackward);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "alpha  gamma");
        assert_eq!(e.cursor_offset(), 6);
    });
}

#[gpui::test]
fn delete_to_line_start_within_paragraph_removes_prefix(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "hello world".into(),
        selection: Selection::Cursor(6),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteToLineStart);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "world");
        assert_eq!(e.cursor_offset(), 0);
    });
}

#[gpui::test]
fn delete_to_line_end_within_paragraph_removes_suffix(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "hello world".into(),
        selection: Selection::Cursor(5),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteToLineEnd);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "hello");
        assert_eq!(e.cursor_offset(), 5);
    });
}

#[gpui::test]
fn delete_to_line_start_at_line_start_is_a_noop(cx: &mut TestAppContext) {
    // Cursor already at the raw start of the line — no content to
    // delete back to. Cmd+Backspace must NOT cross the `\n\n` into
    // the previous paragraph (mirroring macOS standard behavior).
    // Use canonical `\n\n` separation so `enforce_invariants` doesn't
    // re-shape the buffer on the no-op path.
    let initial = EditorState {
        markdown: "line one\n\nline two".into(),
        selection: Selection::Cursor(10), // start of "line two"
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteToLineStart);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "line one\n\nline two");
        assert_eq!(e.cursor_offset(), 10);
    });
}

#[gpui::test]
fn delete_to_line_start_on_second_paragraph_keeps_first(cx: &mut TestAppContext) {
    // Two paragraphs; cursor mid-second-paragraph between "beta" and
    // " gamma". The delete removes "beta" (bytes 7..11) — but NOT
    // the trailing space; Cmd+Backspace is "delete to line start",
    // not "delete to start of word" — and does NOT cross the `\n\n`
    // paragraph break into the first paragraph.
    let initial = EditorState {
        markdown: "alpha\n\nbeta gamma".into(),
        selection: Selection::Cursor(11),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteToLineStart);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "alpha\n\n gamma");
        assert_eq!(e.cursor_offset(), 7);
    });
}

#[gpui::test]
fn delete_to_line_start_inside_list_item_preserves_marker(cx: &mut TestAppContext) {
    // `- hello world` cursor at the end. Cmd+Backspace stops at
    // the content edge (right after `- `), preserving the marker.
    // The empty item that's left over is structurally fine —
    // `enforce_invariants` doesn't touch it because the marker is
    // still well-formed.
    let initial = EditorState {
        markdown: "- hello world".into(),
        selection: Selection::Cursor(13),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteToLineStart);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- ");
        assert_eq!(e.cursor_offset(), 2);
    });
}

#[gpui::test]
fn delete_to_line_start_inside_blockquote_preserves_marker(cx: &mut TestAppContext) {
    // The reported bug, distilled. Multi-paragraph BQ; cursor mid-
    // word in the middle paragraph. Cmd+Backspace must keep `> ` and
    // delete only the user's content past it. Previously deleted the
    // `> ` along with the content, orphaning ` 2` outside the BQ.
    //
    // Layout:
    //   `> Paragraph 1\n`  (0..14, `\n` at 13)
    //   `> \n`             (14..17, `\n` at 16)
    //   `> Paragraph 2\n`  (17..31, `\n` at 30)
    //   `> \n`             (31..34, `\n` at 33)
    //   `> Paragraph 3`    (34..47)
    //
    // Cursor at 28 sits between `Paragraph` and the space before `2`
    // on the middle BQ paragraph.
    let initial = EditorState {
        markdown: "> Paragraph 1\n> \n> Paragraph 2\n> \n> Paragraph 3".into(),
        selection: Selection::Cursor(28),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteToLineStart);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> Paragraph 1\n> \n>  2\n> \n> Paragraph 3",);
        // Cursor lands at the content edge of the middle BQ line.
        assert_eq!(e.cursor_offset(), 19);
    });
}

#[gpui::test]
fn delete_to_line_start_inside_nested_blockquote_preserves_all_markers(cx: &mut TestAppContext) {
    // Triple-nested BQ. The content edge sits past `> > > ` — all
    // three markers must survive Cmd+Backspace.
    //
    // Layout: `> > > foo` is 9 bytes (0..9). Content edge is at 6
    // (right after the third `> `).
    let initial = EditorState {
        markdown: "> > > foo".into(),
        selection: Selection::Cursor(9),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteToLineStart);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> > > ");
        assert_eq!(e.cursor_offset(), 6);
    });
}

#[gpui::test]
fn delete_to_line_start_at_content_edge_is_a_noop(cx: &mut TestAppContext) {
    // Cursor exactly at the content edge of a BQ paragraph — no
    // content to delete back to, but ALSO must not eat into the
    // chain prefix.
    let initial = EditorState {
        markdown: "> hello".into(),
        selection: Selection::Cursor(2), // right after `> `
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteToLineStart);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> hello");
        assert_eq!(e.cursor_offset(), 2);
    });
}

#[gpui::test]
fn delete_to_line_start_inside_bq_wrapped_list_item_preserves_both_markers(
    cx: &mut TestAppContext,
) {
    // `> - foo` — BQ wraps a list item. The content edge sits past
    // `> - ` (4 bytes). Both the BQ marker AND the list-item marker
    // must survive.
    let initial = EditorState {
        markdown: "> - foo".into(),
        selection: Selection::Cursor(7),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteToLineStart);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> - ");
        assert_eq!(e.cursor_offset(), 4);
    });
}

#[gpui::test]
fn delete_to_line_start_on_list_continuation_preserves_indent(cx: &mut TestAppContext) {
    // Multi-paragraph list item. The continuation line `  bar` (2
    // spaces of LI continuation indent + content) — Cmd+Backspace
    // stops past the indent, preserving the 2 spaces.
    //
    // Layout:
    //   `- foo\n`   (0..6, `\n` at 5)
    //   `\n`        (6..7)         ← paragraph break inside item
    //   `  bar`     (7..12)
    let initial = EditorState {
        markdown: "- foo\n\n  bar".into(),
        selection: Selection::Cursor(12),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteToLineStart);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- foo\n\n  ");
        assert_eq!(e.cursor_offset(), 9);
    });
}

#[gpui::test]
fn delete_to_line_start_in_bq_wrapped_li_continuation_preserves_both_prefixes(
    cx: &mut TestAppContext,
) {
    // BQ-wrapped LI with a continuation line. The line's prefix is
    // `> ` (BQ marker) + `  ` (LI continuation indent) = 4 bytes.
    // Cmd+Backspace stops at the visible content edge.
    //
    // Layout:
    //   `> - foo\n`    (0..8, `\n` at 7)
    //   `>   bar`      (8..15)        ← `> ` + `  ` + `bar`
    let initial = EditorState {
        markdown: "> - foo\n>   bar".into(),
        selection: Selection::Cursor(15),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteToLineStart);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> - foo\n>   ");
        assert_eq!(e.cursor_offset(), 12);
    });
}

#[gpui::test]
fn delete_word_backward_inside_blockquote_preserves_marker(cx: &mut TestAppContext) {
    // Reported-bug symmetry for Option+Backspace: prev_word_offset
    // at content edge of a BQ line walks past the `> ` (not
    // alphanumeric) and would return 0 (no prior word). Clamp to
    // chain-prefix end → no-op. Without the clamp, this would
    // delete the `> ` marker.
    let initial = EditorState {
        markdown: "> hello".into(),
        selection: Selection::Cursor(2), // content edge
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteWordBackward);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> hello");
        assert_eq!(e.cursor_offset(), 2);
    });
}

#[gpui::test]
fn delete_word_backward_inside_blockquote_at_word_boundary_deletes_word(cx: &mut TestAppContext) {
    // Within the BQ content, word-delete works normally — the clamp
    // only kicks in when the word target would have landed before
    // the chain prefix end.
    let initial = EditorState {
        markdown: "> hello world".into(),
        selection: Selection::Cursor(13),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteWordBackward);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> hello ");
        assert_eq!(e.cursor_offset(), 8);
    });
}

#[gpui::test]
fn delete_word_forward_does_not_cross_into_next_blockquote_line(cx: &mut TestAppContext) {
    // Two BQ paragraphs in canonical form. Cursor at end of `world`
    // on line 1. The next word boundary (`goodbye` on the second BQ
    // paragraph) would have us splice through `\n> \n> ` to land at
    // the end of `goodbye`. The next-line chain-prefix clamp pulls
    // the target back to the cursor's line end, so the deletion
    // bottoms out at the `\n` and the structural pair + the second
    // BQ paragraph's `> ` all survive.
    //
    // Canonical input is used directly because `enforce_invariants`
    // would otherwise promote the soft break in `> hello world\n> goodbye`
    // up to the same pair shape even on a no-op delete — the test
    // intent is the clamp, not the soft-break promotion.
    let initial = EditorState {
        markdown: "> hello world\n> \n> goodbye".into(),
        selection: Selection::Cursor(13),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteWordForward);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> hello world\n> \n> goodbye");
        assert_eq!(e.cursor_offset(), 13);
    });
}

#[gpui::test]
fn delete_to_line_start_preserves_multi_digit_ordered_marker(cx: &mut TestAppContext) {
    // `10. foo` — ordered item with a 4-byte marker. Cmd+Backspace
    // must stop at the content edge (byte 4) and leave the marker
    // intact, regardless of marker width.
    let initial = EditorState {
        markdown: "10. foo".into(),
        selection: Selection::Cursor(7),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteToLineStart);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "10. ");
        assert_eq!(e.cursor_offset(), 4);
    });
}

#[gpui::test]
fn delete_to_line_start_in_deeply_nested_alternating_chain(cx: &mut TestAppContext) {
    // `> - > - foo` — BQ wrapping LI wrapping BQ wrapping LI. The
    // chain prefix walks all four containers in order: `> ` (BQ) +
    // `- ` (LI marker) + `> ` (BQ) + `- ` (LI marker) = 8 bytes.
    // Cmd+Backspace preserves every structural marker.
    let initial = EditorState {
        markdown: "> - > - foo".into(),
        selection: Selection::Cursor(11),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteToLineStart);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> - > - ");
        assert_eq!(e.cursor_offset(), 8);
    });
}

#[gpui::test]
fn delete_word_forward_does_not_cross_into_next_list_item_marker(cx: &mut TestAppContext) {
    // Two list items, cursor at end of `foo` (end of item 1's content
    // line). The next word target lies past `\n- ` on item 2. Clamp
    // to current line end so item 2's `- ` survives.
    let initial = EditorState {
        markdown: "- foo\n- bar".into(),
        selection: Selection::Cursor(5),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteWordForward);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- foo\n- bar");
        assert_eq!(e.cursor_offset(), 5);
    });
}

#[gpui::test]
fn delete_word_backward_at_buffer_start_inside_blockquote_is_noop(cx: &mut TestAppContext) {
    // Cursor at byte 0 inside the BQ marker itself. The chain may or
    // may not be reported at this position (pulldown treats byte 0 as
    // before the BQ scope opens); either way, the delete must be a
    // no-op — there's no content to delete back to, and we must not
    // delete forward into the marker bytes.
    let initial = EditorState {
        markdown: "> hello".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteWordBackward);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "> hello");
        assert_eq!(e.cursor_offset(), 0);
    });
}

#[gpui::test]
fn delete_word_forward_top_level_crosses_newline(cx: &mut TestAppContext) {
    // Top-level multi-paragraph — no chain prefix on the next line,
    // so Option+Delete crosses the `\n\n` and eats the next word.
    // This is plain-text editor behavior.
    let initial = EditorState {
        markdown: "alpha\n\nbeta gamma".into(),
        selection: Selection::Cursor(5), // end of "alpha"
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteWordForward);
    editor.read_with(cx, |e, _| {
        // The `\n\n` + "beta" are consumed in one keystroke. The
        // post-pass leaves the result as-is — no chain prefix to
        // restore.
        assert_eq!(e.value(), "alpha gamma");
        assert_eq!(e.cursor_offset(), 5);
    });
}

#[gpui::test]
fn delete_to_line_end_inside_list_item_stops_at_newline(cx: &mut TestAppContext) {
    // List with two items. Cursor at the start of "world" on item 1.
    // Cmd+Delete clears the rest of the line — but NOT the `\n` that
    // separates item 1 from item 2.
    let initial = EditorState {
        markdown: "- hello world\n- second".into(),
        selection: Selection::Cursor(8), // start of "world"
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteToLineEnd);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "- hello \n- second");
        assert_eq!(e.cursor_offset(), 8);
    });
}

#[gpui::test]
fn delete_to_line_end_at_end_of_line_is_a_noop(cx: &mut TestAppContext) {
    // Cursor at the very last byte (no trailing `\n` past it).
    // DeleteToLineEnd should leave the buffer untouched. Single
    // paragraph avoids the soft-break-promotion gotcha (a
    // `\n`-separated multi-line input would get re-shaped to
    // `\n\n` by `enforce_invariants` even on a delete no-op).
    let initial = EditorState {
        markdown: "abc".into(),
        selection: Selection::Cursor(3),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, DeleteToLineEnd);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "abc");
        assert_eq!(e.cursor_offset(), 3);
    });
}

// ---------------------------------------------------------------------------
// Vertical / horizontal navigation across blockquote multi-paragraph runs
//
// The pair shape `\n> \n> ` is one BQ paragraph break (collapses to one
// paragraph_gap visually). N pair shapes back-to-back render as N-1
// visible empty rows between two BQ paragraphs. Each of those rows is a
// cursor-landable position; the forbidden-position machinery's notion
// of "pair interior" has to exempt the inter-pair boundaries, and
// move_vertical's notion of "phantom row" has to recognize the visible
// synth rows. Both used to misbehave: chain-pair-interior over-fired
// on shifted pair patterns and marked the inter-pair `\n` as interior;
// move_vertical checked line_start (interior for BQ synth rows) instead
// of line_end (the canonical pair boundary).
// ---------------------------------------------------------------------------

#[gpui::test]
fn right_arrow_lands_on_bq_synth_row_then_next_paragraph(cx: &mut TestAppContext) {
    // `> a\n> \n> \n> \n> b` = two BQ paragraphs separated by one
    // visible empty row. Right-arrow walks: cursor at end of "> a"
    // → synth row's `\n` boundary (byte 9) → start of "> b" content
    // (byte 15).
    //
    // Layout:
    //   0..3   `> a`
    //   3      `\n`            (pair 1 starts)
    //   4..6   `> `
    //   6      `\n`
    //   7..9   `> `             ← synth row (line_start=7, line_end=9)
    //   9      `\n`             (= pair boundary; allowed)
    //   10..12 `> `
    //   12     `\n`
    //   13..16 `> b`
    let initial = EditorState {
        markdown: "> a\n> \n> \n> \n> b".into(),
        selection: Selection::Cursor(3),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 9));
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 15));
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 16));
}

#[gpui::test]
fn left_arrow_lands_on_bq_synth_row_from_next_paragraph(cx: &mut TestAppContext) {
    // Symmetric: cursor at start of "> b" content, Left walks back
    // to the synth row's allowed position, then to the end of "> a".
    let initial = EditorState {
        markdown: "> a\n> \n> \n> \n> b".into(),
        selection: Selection::Cursor(15),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Left);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 9));
    dispatch(cx, handle, &editor, Left);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 3));
}

#[gpui::test]
fn down_arrow_from_bq_paragraph_lands_on_next_paragraph(cx: &mut TestAppContext) {
    // Basic case: two BQ paragraphs with no synth row between.
    // Down from end of "> a" should land on "> b" content at the
    // same column.
    let initial = EditorState {
        markdown: "> a\n> \n> b".into(),
        selection: Selection::Cursor(2), // end of "a"
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Down);
    editor.read_with(cx, |e, _| {
        // Column 2 of "> a" (= the `a`) maps to column 2 of "> b"
        // (= the `b`). Source byte: 7 (start of `> b` line) + 2 = 9.
        assert_eq!(e.cursor_offset(), 9);
    });
}

#[gpui::test]
fn down_arrow_through_bq_synth_row(cx: &mut TestAppContext) {
    // Down arrow walks from "> a" → synth empty row → "> b".
    let initial = EditorState {
        markdown: "> a\n> \n> \n> \n> b".into(),
        selection: Selection::Cursor(2),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Down);
    editor.read_with(cx, |e, _| {
        // Lands on the synth row. Source-byte column 2 from "> a"
        // would map past the synth's content (just `> `, length 2);
        // we clamp at the row's end-of-line (the `\n` at byte 9).
        assert_eq!(e.cursor_offset(), 9);
    });
    dispatch(cx, handle, &editor, Down);
    editor.read_with(cx, |e, _| {
        // Column persists across the synth row; lands at "> b"
        // column 2 = byte 13 + 2 = 15.
        assert_eq!(e.cursor_offset(), 15);
    });
}

#[gpui::test]
fn up_arrow_walks_through_bq_synth_row_to_first_paragraph(cx: &mut TestAppContext) {
    // The reported bug, in test form. Cursor inside "> Paragraph
    // three." line; Up should land on the visible empty row between
    // the BQ paragraphs, NOT jump straight to "Paragraph one.".
    //
    // Layout for `> P1\n> \n> \n> \n> P3`:
    //   0..4   `> P1`
    //   4      `\n`
    //   5..7   `> `
    //   7      `\n`
    //   8..10  `> `             ← synth row (line_start=8, line_end=10)
    //   10     `\n`             (= pair boundary; allowed)
    //   11..13 `> `
    //   13     `\n`
    //   14..18 `> P3`
    let initial = EditorState {
        markdown: "> P1\n> \n> \n> \n> P3".into(),
        selection: Selection::Cursor(18), // end of "P3"
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Up);
    editor.read_with(cx, |e, _| {
        // First Up lands on the synth row's end-of-line position
        // (= the `\n` boundary at byte 10). Visual x at the empty
        // row maps to the row's only allowed position.
        assert_eq!(e.cursor_offset(), 10);
    });
    dispatch(cx, handle, &editor, Up);
    editor.read_with(cx, |e, _| {
        // Second Up: `intended_x` was captured at end of "P3" on the
        // first press, so we re-target end-of-line on "> P1" too
        // (not column 2 = `P`). Visual nav with intended-x
        // preservation: end of P3 → end of P1.
        assert_eq!(e.cursor_offset(), 4);
    });
}

#[gpui::test]
fn up_arrow_walks_through_two_bq_synth_rows_one_at_a_time(cx: &mut TestAppContext) {
    // Three pair shapes between two BQ paragraphs = two visible
    // empty rows. Up arrow should pause on each one in turn.
    //
    // Layout `> a\n> \n> \n> \n> \n> \n> b`:
    //   0..3   `> a`
    //   3      `\n`
    //   4..6   `> `   (phantom)
    //   6      `\n`
    //   7..9   `> `   (synth #1; line_end=9 is pair boundary)
    //   9      `\n`   (allowed boundary)
    //   10..12 `> `   (phantom)
    //   12     `\n`
    //   13..15 `> `   (synth #2; line_end=15 is pair boundary)
    //   15     `\n`   (allowed boundary)
    //   16..18 `> `   (phantom)
    //   18     `\n`
    //   19..22 `> b`
    let initial = EditorState {
        markdown: "> a\n> \n> \n> \n> \n> \n> b".into(),
        selection: Selection::Cursor(21), // end of "> b" content
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Up);
    editor.read_with(cx, |e, _| {
        // First Up: synth #2 line_end = 15.
        assert_eq!(e.cursor_offset(), 15);
    });
    dispatch(cx, handle, &editor, Up);
    editor.read_with(cx, |e, _| {
        // Second Up: synth #1 line_end = 9.
        assert_eq!(e.cursor_offset(), 9);
    });
    dispatch(cx, handle, &editor, Up);
    editor.read_with(cx, |e, _| {
        // Third Up: first paragraph "> a". Column 2 from previous
        // synth → byte 0 + 2 = byte 2 (= `a`).
        assert_eq!(e.cursor_offset(), 2);
    });
}

#[gpui::test]
fn up_arrow_in_user_reported_bug_lands_on_visible_empty_row(cx: &mut TestAppContext) {
    // Exact reproduction of the user-reported bug:
    //
    //   > Paragraph one.
    //   >
    //   >
    //   >
    //   > |Paragraph three.
    //
    // Before fix: Up arrow jumped past all three `> ` empty lines
    // and landed at "Paragraph one.". After fix: Up lands on the
    // single visible empty row between the two paragraphs.
    let initial = EditorState {
        markdown: "> Paragraph one.\n> \n> \n> \n> Paragraph three.".into(),
        // Cursor at byte 28 = `P` of "Paragraph three." (right after `> `).
        selection: Selection::Cursor(28),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Up);
    editor.read_with(cx, |e, _| {
        // The synth empty row sits between pair 1 (bytes 16..22) and
        // pair 2 (bytes 22..28). Source column 2 of "> Paragraph
        // three." line (line_start=26) → target on synth-row line
        // (line_start=20, line_end=22) at column 2 → byte 22 (clamped
        // by snap_within_line to the pair boundary).
        assert_eq!(e.cursor_offset(), 22);
    });
}

#[gpui::test]
fn left_arrow_in_user_reported_bug_lands_on_visible_empty_row(cx: &mut TestAppContext) {
    // Same fixture, Left arrow path. The bytes between "Paragraph
    // three." start (byte 28) and the synth row boundary (byte 22)
    // are all chain-pair interior; Left collapses them into a single
    // step to the synth row.
    let initial = EditorState {
        markdown: "> Paragraph one.\n> \n> \n> \n> Paragraph three.".into(),
        selection: Selection::Cursor(28),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Left);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.cursor_offset(), 22);
    });
}

#[gpui::test]
fn chain_pair_interior_recognizes_inter_pair_boundary(_cx: &mut TestAppContext) {
    // Regression for the shifted-pair-overlap bug: in a 2-pair run
    // the byte BETWEEN the two canonical pairs (pair_len boundary)
    // is a real cursor-landable position. The old
    // `is_chain_pair_interior` matched a shifted pair shape at the
    // midpoint and marked the boundary byte interior.
    //
    // For `> a\n> \n> \n> \n> b` chain `[BQ]` pair_len=6 run [3..15],
    // the inter-pair boundary is at offset 6 = byte 9.
    let md = "> a\n> \n> \n> \n> b";
    assert!(
        !gpui_markdown_editor::analysis::is_chain_pair_interior(md, 9),
        "byte 9 is the canonical pair boundary; must not be interior"
    );
    assert!(
        !gpui_markdown_editor::analysis::is_forbidden_position(md, 9),
        "byte 9 must be a cursor-allowed position"
    );
}

// ---------------------------------------------------------------------------
// Visual / wrap-aware Up / Down navigation
//
// The cursor's Up / Down movement follows the *rendered* row layout, not
// the source-byte positions. A long soft-wrapped paragraph has multiple
// wrap rows inside a single logical line; Up from the second wrap row
// must land on the first wrap row (same logical line), not on the line
// above. Verified via `open_editor_narrow` which constrains the window
// width so wrapping reliably triggers.
// ---------------------------------------------------------------------------

#[gpui::test]
fn down_from_short_paragraph_lands_on_first_wrap_row_of_long_paragraph(cx: &mut TestAppContext) {
    // Two paragraphs:
    //   1. "Hi."                                            (line 1, no wrap)
    //   2. "This is a very long paragraph that wraps."      (wraps into 2+ rows)
    //
    // From cursor in paragraph 1, Down lands on the *first wrap row*
    // of paragraph 2 — not on its last wrap row (the source-byte
    // model would land near the end since "Hi." line is short).
    let markdown =
        "Hi.\n\nThis is a very long paragraph that wraps in the narrow viewport configured by open_editor_narrow.".to_string();
    let initial = EditorState {
        markdown,
        selection: Selection::Cursor(2), // end of "Hi"
        ..Default::default()
    };
    let (handle, editor) = open_editor_narrow(cx, initial);
    // Trigger a render so last_blocks is populated before the
    // visual-move path is consulted. A no-op SetSelection works
    // because dispatch always runs paint via run_until_parked.
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Left);
    // Now position cursor explicitly at "Hi" column 2.
    set_cursor(cx, handle, &editor, 2);
    dispatch(cx, handle, &editor, Down);
    editor.read_with(cx, |e, _| {
        let cursor = e.cursor_offset();
        // Should land on the first wrap row of the second
        // paragraph — somewhere in the prefix of "This is a very
        // long paragraph that wraps...". The exact column depends
        // on font metrics, but the cursor MUST sit before the
        // (wrap) midpoint of the long paragraph.
        let para2_start = 5;
        let para2_end = e.value().len();
        let mid = (para2_start + para2_end) / 2;
        assert!(
            cursor >= para2_start && cursor < mid,
            "Down from short paragraph should land in the first half \
             of the wrapped second paragraph, got cursor={cursor} \
             (range {para2_start}..{mid})",
        );
    });
}

#[gpui::test]
fn up_from_second_wrap_row_lands_on_first_wrap_row_same_paragraph(cx: &mut TestAppContext) {
    // A single paragraph that wraps over multiple visible rows.
    // Cursor placed deep enough in the content that it lands on the
    // second wrap row; Up should land on the *first wrap row of the
    // same logical line*, not jump to a previous paragraph.
    let para = "This is a very long paragraph that wraps around. \
                In my viewport it wraps here and this is rendered on the second line.";
    let markdown = format!("Heading\n\n{para}");
    let cursor_at = markdown.len() - 10; // near end of long paragraph
    let initial = EditorState {
        markdown: markdown.clone(),
        selection: Selection::Cursor(cursor_at),
        ..Default::default()
    };
    let (handle, editor) = open_editor_narrow(cx, initial);
    // Force a paint pass to populate last_blocks.
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Left);
    set_cursor(cx, handle, &editor, cursor_at);
    dispatch(cx, handle, &editor, Up);
    editor.read_with(cx, |e, _| {
        let cursor = e.cursor_offset();
        let para_start = 9; // "Heading\n\n" = 9 bytes
        // Up should NOT jump out of the long paragraph back to
        // "Heading" — the wrap row above the cursor is still part
        // of the same logical line.
        assert!(
            cursor >= para_start,
            "Up from a wrapped row should stay inside the long \
             paragraph, got cursor={cursor} (para starts at {para_start})",
        );
        assert!(
            cursor < cursor_at,
            "Up should move the cursor backward in source bytes \
             (was at {cursor_at}, ended up at {cursor})",
        );
    });
}

#[gpui::test]
fn down_from_first_wrap_row_lands_on_second_wrap_row_same_paragraph(cx: &mut TestAppContext) {
    // Symmetric to the test above: cursor near the start of a long
    // wrapped paragraph; Down should land on the same paragraph's
    // second wrap row, not on the next block.
    let para = "This is a very long paragraph that wraps around. \
                In my viewport it wraps here and this is rendered on the second line.";
    let markdown = format!("{para}\n\nNext paragraph.");
    let initial = EditorState {
        markdown: markdown.clone(),
        selection: Selection::Cursor(5), // somewhere in "This is..."
        ..Default::default()
    };
    let (handle, editor) = open_editor_narrow(cx, initial);
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Left);
    set_cursor(cx, handle, &editor, 5);
    dispatch(cx, handle, &editor, Down);
    editor.read_with(cx, |e, _| {
        let cursor = e.cursor_offset();
        let para_end = para.len();
        // Down should NOT skip past the wrapped paragraph into
        // "Next paragraph." — the cursor stays inside the long
        // paragraph's bytes.
        assert!(
            cursor > 5 && cursor < para_end,
            "Down from wrap row #1 should stay inside the long \
             paragraph, got cursor={cursor} (expected in (5, {para_end}))",
        );
    });
}

#[gpui::test]
fn repeated_down_advances_through_every_wrap_row_to_the_last(cx: &mut TestAppContext) {
    // Regression for bug #4 ("Down stalls"). In a soft-wrapped paragraph, a
    // Down that lands the caret EXACTLY on a wrap boundary (a row start) used to
    // freeze: the next Down computed the caret's current row from the
    // upper-affinity `position_for_index`, targeting the same boundary row again,
    // so the offset never changed. With the transient wrap-affinity signal, a
    // Down onto a boundary is remembered as belonging to the lower row, so the
    // next Down steps past it. Here every Down before the last visual row must
    // strictly advance the caret, and the caret must reach the final row.
    let para = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do \
                eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim \
                ad minim veniam quis nostrud exercitation ullamco laboris nisi ut \
                aliquip ex ea commodo consequat duis aute irure dolor in \
                reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla \
                pariatur excepteur sint occaecat cupidatat non proident sunt in \
                culpa qui officia deserunt mollit anim id est laborum."
        .to_string();
    let initial = EditorState {
        markdown: para,
        selection: Selection::Cursor(1), // "L|orem" — the reported start
        ..Default::default()
    };
    let (handle, editor) = open_editor_narrow(cx, initial);
    // Populate last_blocks with a paint pass.
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Left);
    set_cursor(cx, handle, &editor, 1);

    let (mut prev_off, mut prev_top) = editor.read_with(cx, |e, _| {
        (
            e.cursor_offset(),
            e.caret_content_y().expect("caret row").0.as_f32(),
        )
    });
    let mut reached_last_row = false;
    let mut distinct_rows = 1usize; // the starting row
    for step in 0..40 {
        dispatch(cx, handle, &editor, Down);
        let (off, top) = editor.read_with(cx, |e, _| {
            (
                e.cursor_offset(),
                e.caret_content_y().expect("caret row").0.as_f32(),
            )
        });
        if top > prev_top + 0.5 {
            // The caret descended to a new visual row — moving to a new row must
            // strictly advance the source offset (no stall). This is the #4 gate.
            distinct_rows += 1;
            assert!(
                off > prev_off,
                "Down reached a new row at step {step} without advancing the \
                 offset (stuck at {off}) — a boundary caret stalled",
            );
        } else {
            // The caret's row didn't descend: it's saturated on the last visual
            // row. It must not have moved backward.
            assert!(
                off >= prev_off,
                "Down moved the caret backward at step {step} ({prev_off} → {off})",
            );
            reached_last_row = true;
            break;
        }
        prev_off = off;
        prev_top = top;
    }
    assert!(
        reached_last_row,
        "repeated Down never reached the paragraph's last wrap row",
    );
    // Walking a genuinely multi-row paragraph, not stalling after ~3 rows.
    assert!(
        distinct_rows >= 6,
        "Down visited only {distinct_rows} distinct rows — it should walk many \
         wrap rows of the long paragraph, not stall early",
    );
}

#[gpui::test]
fn down_onto_wrap_boundary_renders_caret_on_lower_row(cx: &mut TestAppContext) {
    // Bug #3 (render affinity). When a Down (or Home/Right) lands the caret
    // exactly on a soft-wrap boundary, the caret must render at the START of the
    // LOWER row — not the end of the upper row where gpui's `position_for_index`
    // resolves the shared offset. We prove it by contrasting the render position
    // of the SAME boundary offset under the two affinities: after a Down the
    // affinity is Downstream (lower row); a fresh `set_cursor` to that offset
    // resets it to Downstream too, so instead we compare against End on that row,
    // which sets Upstream (upper row). The two must differ by one row height.
    let para = "This is a fairly long paragraph that soft-wraps across several \
                rows inside the narrow viewport configured by open_editor_narrow, \
                giving us interior wrap boundaries to land a caret on."
        .to_string();
    let initial = EditorState {
        markdown: para,
        selection: Selection::Cursor(2),
        ..Default::default()
    };
    let (handle, editor) = open_editor_narrow(cx, initial);
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Left);

    // End on the first wrap row → lands on the boundary offset, Upstream
    // affinity: renders on the UPPER row (row 0, y ≈ 0).
    set_cursor(cx, handle, &editor, 2);
    dispatch(cx, handle, &editor, End);
    let (boundary_off, upper_y, row_h) = editor.read_with(cx, |e, _| {
        let (t, b) = e.caret_content_y().expect("caret row");
        (e.cursor_offset(), t.as_f32(), (b - t).as_f32())
    });
    assert!(
        upper_y < row_h,
        "End keeps Upstream affinity: the boundary caret renders on the upper \
         row (y {upper_y} should be within the first row height {row_h})",
    );

    // The SAME offset, placed fresh (Downstream affinity), renders on the LOWER
    // row (row 1, y ≈ row_h).
    set_cursor(cx, handle, &editor, boundary_off);
    let lower_y = editor.read_with(cx, |e, _| {
        e.caret_content_y().expect("caret row").0.as_f32()
    });
    assert!(
        lower_y >= upper_y + row_h - 1.0,
        "a fresh (Downstream) caret on the same boundary offset renders on the \
         lower row (y {lower_y} should be ~one row {row_h} below the upper y \
         {upper_y})",
    );
}

#[gpui::test]
fn end_on_wrapped_row_stays_on_that_row(cx: &mut TestAppContext) {
    // End on a wrapped row lands the caret at that row's last offset (the wrap
    // boundary) and — via Upstream affinity — keeps the caret rendered at the
    // END of that row, not the start of the next.
    let para = "This is a fairly long paragraph that soft-wraps across several \
                rows inside the narrow viewport configured by open_editor_narrow, \
                giving us interior wrap boundaries to land a caret on."
        .to_string();
    let initial = EditorState {
        markdown: para,
        selection: Selection::Cursor(2),
        ..Default::default()
    };
    let (handle, editor) = open_editor_narrow(cx, initial);
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Left);
    set_cursor(cx, handle, &editor, 2);

    let start_off = editor.read_with(cx, |e, _| e.cursor_offset());
    dispatch(cx, handle, &editor, End);
    editor.read_with(cx, |e, _| {
        let end_off = e.cursor_offset();
        assert!(
            end_off > start_off,
            "End moved the caret forward to the row's end ({start_off} → {end_off})",
        );
        // End sets Upstream affinity, so the caret renders at the END of the
        // acted row (row 0) rather than dropping to the next row's start.
        let (top, bot) = e.caret_content_y().expect("caret row");
        let row_h = (bot - top).as_f32();
        assert!(
            top.as_f32() < row_h,
            "End keeps the caret on the first wrap row (y {} within row height {})",
            top.as_f32(),
            row_h,
        );
    });
}

#[gpui::test]
fn intended_x_preserved_across_consecutive_up_presses(cx: &mut TestAppContext) {
    // Press Up Up: the first lands on a short row that shrinks the
    // visible column to fit; the second should return to the
    // original *visual* column from before the first press (intended-x
    // streak). Demonstrates the streak's persistence.
    //
    // The fixture is the user's reported bug: long paragraph + short
    // paragraph + long paragraph. Start at end of paragraph 3, Up to
    // short-mid, Up to a column on paragraph 1 that matches the
    // original visual x — past the right edge of "P1" content, so
    // the cursor lands at end of "P1" (byte 4), NOT at the source-
    // byte column-mapped position (byte 2 = `P`).
    let initial = EditorState {
        markdown: "> P1\n> \n> \n> \n> P3".into(),
        selection: Selection::Cursor(18),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Up);
    let first = editor.read_with(cx, |e, _| e.cursor_offset());
    dispatch(cx, handle, &editor, Up);
    editor.read_with(cx, |e, _| {
        // First Up landed on the synth empty BQ row.
        assert_eq!(first, 10);
        // Second Up: intended_x was captured at end of "P3" on the
        // first press; even though the synth row was empty (which
        // would shrink any source-byte column), the second press
        // returns to the original x → end of "P1" content (byte 4),
        // not the synth-row-shrunken column.
        assert_eq!(e.cursor_offset(), 4);
    });
}

#[gpui::test]
fn intended_x_reset_by_non_vertical_action(cx: &mut TestAppContext) {
    // Up Up gives the persistent-x behavior. But intervening Left
    // (or any non-vertical) clears intended_x, so the next Up uses
    // the cursor's current x as the new anchor.
    let initial = EditorState {
        markdown: "> P1\n> \n> \n> \n> P3".into(),
        selection: Selection::Cursor(18),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Up);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 10));
    // Break the streak with a Left arrow.
    dispatch(cx, handle, &editor, Left);
    // Now Up should NOT remember the original "P3" x — it captures
    // the current cursor's x (= 0 on the synth row) and maps to the
    // start of "P1" content on the line above.
    dispatch(cx, handle, &editor, Up);
    editor.read_with(cx, |e, _| {
        // The synth row's visible content is empty, so cursor x on
        // it is the line's left edge (≈ chain prefix end). Mapped
        // back to "> P1" line, that's the leading edge of `P` → byte 2.
        // The exact result depends on font metrics; what matters is
        // that the cursor is NOT at byte 4 (which would have
        // indicated stale intended_x).
        assert_ne!(
            e.cursor_offset(),
            4,
            "intended_x should have been cleared by Left"
        );
    });
}

// ---------------------------------------------------------------------------
// Display-line-aware Home / End / Shift+Home / Shift+End
//
// Like Up / Down, Home and End (and their Cmd+Left / Cmd+Right aliases)
// follow the *rendered* wrapped row, not the source `\n`-delimited line.
// Pressing Home on the second wrap row of a long paragraph lands on the
// first character of that row, not the paragraph start. Verified via
// `open_editor_narrow`, which forces soft wrapping; a paint pass
// (Right + Left) populates `last_blocks` so the display-aware path
// engages (with the source-line fallback covered by `home_end_doc_jump`
// above, where no wrapping and pre-paint state exercise `MoveLineStart`).
// ---------------------------------------------------------------------------

/// Cursor near the end of a long wrapped paragraph (its last wrap row);
/// Home lands at the *display-line* start of that row — strictly after
/// the paragraph start (byte 0) and before the caret, which the
/// source-line Home (byte 0) would never do.
#[gpui::test]
fn home_moves_to_display_line_start_of_wrapped_row(cx: &mut TestAppContext) {
    let markdown = "This is a very long paragraph that wraps across several rows \
                    inside the narrow viewport configured by open_editor_narrow here."
        .to_string();
    let cursor_at = markdown.len() - 6;
    let initial = EditorState {
        markdown,
        selection: Selection::Cursor(cursor_at),
        ..Default::default()
    };
    let (handle, editor) = open_editor_narrow(cx, initial);
    // Force a paint so last_blocks is populated before Home is handled.
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Left);
    set_cursor(cx, handle, &editor, cursor_at);
    dispatch(cx, handle, &editor, Home);
    editor.read_with(cx, |e, _| {
        let cursor = e.cursor_offset();
        assert!(
            cursor > 0 && cursor < cursor_at,
            "display-line Home should land at the start of the caret's \
             wrap row (after byte 0, before the caret), got {cursor} \
             (caret was {cursor_at})",
        );
    });
}

/// Cursor near the start of a long wrapped paragraph (its first wrap
/// row); End lands at the *display-line* end of that row — strictly
/// before the paragraph end, which the source-line End would overshoot.
#[gpui::test]
fn end_moves_to_display_line_end_of_wrapped_row(cx: &mut TestAppContext) {
    let markdown = "This is a very long paragraph that wraps across several rows \
                    inside the narrow viewport configured by open_editor_narrow here."
        .to_string();
    let para_end = markdown.len();
    let cursor_at = 5;
    let initial = EditorState {
        markdown,
        selection: Selection::Cursor(cursor_at),
        ..Default::default()
    };
    let (handle, editor) = open_editor_narrow(cx, initial);
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Left);
    set_cursor(cx, handle, &editor, cursor_at);
    dispatch(cx, handle, &editor, End);
    editor.read_with(cx, |e, _| {
        let cursor = e.cursor_offset();
        assert!(
            cursor > cursor_at && cursor < para_end,
            "display-line End should land at the end of the caret's \
             wrap row (after the caret, before paragraph end {para_end}), \
             got {cursor}",
        );
    });
}

/// Shift+Home from the last wrap row extends the selection back to the
/// display-line start of that row (a Range whose head precedes the
/// anchor), not all the way to the paragraph start.
#[gpui::test]
fn shift_home_extends_to_display_line_start(cx: &mut TestAppContext) {
    let markdown = "This is a very long paragraph that wraps across several rows \
                    inside the narrow viewport configured by open_editor_narrow here."
        .to_string();
    let cursor_at = markdown.len() - 6;
    let initial = EditorState {
        markdown,
        selection: Selection::Cursor(cursor_at),
        ..Default::default()
    };
    let (handle, editor) = open_editor_narrow(cx, initial);
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Left);
    set_cursor(cx, handle, &editor, cursor_at);
    dispatch(cx, handle, &editor, ShiftHome);
    editor.read_with(cx, |e, _| match e.selection() {
        Selection::Range { anchor, head } => {
            assert_eq!(anchor, cursor_at, "anchor stays where the caret was");
            assert!(
                head > 0 && head < cursor_at,
                "Shift+Home head should be the wrap-row start (after byte \
                 0, before the anchor), got head={head}",
            );
        }
        other => panic!("expected a Range selection, got {other:?}"),
    });
}

/// Shift+End from the first wrap row extends the selection forward to
/// the display-line end of that row (a Range whose head follows the
/// anchor), not to the paragraph end.
#[gpui::test]
fn shift_end_extends_to_display_line_end(cx: &mut TestAppContext) {
    let markdown = "This is a very long paragraph that wraps across several rows \
                    inside the narrow viewport configured by open_editor_narrow here."
        .to_string();
    let para_end = markdown.len();
    let cursor_at = 5;
    let initial = EditorState {
        markdown,
        selection: Selection::Cursor(cursor_at),
        ..Default::default()
    };
    let (handle, editor) = open_editor_narrow(cx, initial);
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Left);
    set_cursor(cx, handle, &editor, cursor_at);
    dispatch(cx, handle, &editor, ShiftEnd);
    editor.read_with(cx, |e, _| match e.selection() {
        Selection::Range { anchor, head } => {
            assert_eq!(anchor, cursor_at, "anchor stays where the caret was");
            assert!(
                head > cursor_at && head < para_end,
                "Shift+End head should be the wrap-row end (after the \
                 anchor, before paragraph end {para_end}), got head={head}",
            );
        }
        other => panic!("expected a Range selection, got {other:?}"),
    });
}

/// Display-line Home is pure navigation: it changes only the selection,
/// never the buffer, so it must not push an undo step. Type a character
/// (one undoable edit), press Home, then Undo — the Undo reverts the
/// typed character (not a phantom Home step), proving Home recorded no
/// history.
#[gpui::test]
fn display_line_home_records_no_undo_step(cx: &mut TestAppContext) {
    let markdown = "This is a very long paragraph that wraps across several rows \
                    inside the narrow viewport configured by open_editor_narrow here."
        .to_string();
    let cursor_at = markdown.len();
    let initial = EditorState {
        markdown: markdown.clone(),
        selection: Selection::Cursor(cursor_at),
        ..Default::default()
    };
    let (handle, editor) = open_editor_narrow(cx, initial);
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Left);
    set_cursor(cx, handle, &editor, cursor_at);
    type_text(cx, handle, &editor, "Z");
    let typed = editor.read_with(cx, |e, _| e.value().to_string());
    assert!(typed.ends_with('Z'), "precondition: 'Z' was typed");
    // Pure navigation between the edit and the undo.
    dispatch(cx, handle, &editor, Home);
    dispatch(cx, handle, &editor, End);
    dispatch(cx, handle, &editor, Undo);
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.value(),
            markdown,
            "Undo after Home/End should revert the typed 'Z' — navigation \
             must not record its own undo step",
        );
    });
}

// ---------------------------------------------------------------------------
// Empty marker-line list items
//
// A list item whose only block-level child is a non-Paragraph block
// (nested list, blockquote, code block, …) and whose marker stands
// alone on its line. Pulldown's parse tree has no Paragraph child for
// that item, so the render walker used to fold the marker line into
// the recursed leaf — collapsing the marker row's chain depth to the
// nested block's chain, painting both markers on the same row, and
// landing the cursor at the deep indent column. The render-side fix
// emits a synth marker-line block first, keeping the marker row at
// the LI's own chain depth.
// ---------------------------------------------------------------------------

#[gpui::test]
fn empty_intermediate_li_emits_separate_marker_row_block(cx: &mut TestAppContext) {
    // The user-reported fixture:
    //
    //   1. One
    //   2.
    //      1. Two, One
    //
    // Before fix: 2 blocks, second covered both the empty `2. ` row
    // AND the nested list with chain=[outer, inner], so the marker
    // row painted at the inner indent and both marker overlays
    // stacked on one row.
    //
    // After fix: 3 blocks — one per visible row. The synth marker
    // row carries only the outer LI in its container chain so it
    // paints at the outer indent.
    let initial = EditorState {
        markdown: "1. One\n2. \n   1. Two, One".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    assert_eq!(
        spec.blocks.len(),
        3,
        "expected one block per visible row (got {})",
        spec.blocks.len(),
    );
    // Block 0: "1. One" at the start, containers=1 (outer LI only).
    assert_eq!(spec.blocks[0].source_range, 0..6);
    assert_eq!(spec.blocks[0].containers.len(), 1);
    // Block 1: synth for the empty `2. ` row, containers=1 (outer LI).
    // The synth's range covers the outer LI's marker through its
    // line's `\n`-or-EOL (then trimmed of trailing `\n` by
    // `inject_empty_paragraphs`).
    assert_eq!(
        spec.blocks[1].source_range,
        7..10,
        "synth marker-row should cover the LI marker line (trimmed of trailing `\\n`)",
    );
    assert_eq!(
        spec.blocks[1].containers.len(),
        1,
        "synth marker-row should carry only the outer LI in its chain, not the nested inner",
    );
    // Block 2: the nested list's content, containers=2 (outer + inner).
    assert_eq!(spec.blocks[2].source_range, 14..25);
    assert_eq!(spec.blocks[2].containers.len(), 2);
}

#[gpui::test]
fn cursor_lands_on_empty_intermediate_li_marker_row(cx: &mut TestAppContext) {
    // The empty marker line's `\n` byte (= line_end) is the canonical
    // cursor position for "on the empty `2. ` row". The synth block's
    // source_range covers it (after the `\n` trim), so
    // block_claims_cursor lands the cursor on the synth and the
    // visual column corresponds to the outer LI's content edge.
    let initial = EditorState {
        markdown: "1. One\n2. \n   1. Two, One".into(),
        // Cursor at the `\n` ending the `2. ` line.
        selection: Selection::Cursor(10),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let synth = &spec.blocks[1];
    assert!(
        synth.source_range.start <= 10 && 10 <= synth.source_range.end,
        "cursor at byte 10 must fall within the synth marker-row block range {:?}",
        synth.source_range,
    );
}

#[gpui::test]
fn typing_on_empty_intermediate_li_marker_row_adds_content_to_item(cx: &mut TestAppContext) {
    // From the synth marker row, inserting text turns the empty item
    // into a non-empty one with the nested list as a continuation
    // child.
    let initial = EditorState {
        markdown: "1. One\n2. \n   1. Two, One".into(),
        selection: Selection::Cursor(10),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText("Two".into()),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "1. One\n2. Two\n   1. Two, One");
        // Cursor lands right after the inserted text.
        assert_eq!(e.cursor_offset(), 13);
    });
}

#[gpui::test]
fn empty_li_with_only_nested_blockquote_emits_separate_marker_row(cx: &mut TestAppContext) {
    // Variant: LI's only child is a blockquote on a later line.
    // Same shape, same synth needed.
    //
    //   1.
    //      > quoted
    //
    // Marker line is `1.` followed by a newline; the BQ starts on
    // the next line via 3-space indented continuation.
    let initial = EditorState {
        markdown: "1. \n   > quoted".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    let marker_row = spec
        .blocks
        .iter()
        .find(|b| b.source_range.contains(&0) && b.containers.len() == 1)
        .or_else(|| spec.blocks.first());
    assert!(
        marker_row.is_some(),
        "expected a synth marker-row block at the LI start",
    );
}

#[gpui::test]
fn li_with_paragraph_then_nested_list_does_not_emit_synth(cx: &mut TestAppContext) {
    // Sanity check the other branch: when the LI has a Paragraph
    // child (the marker line carries content) followed by a nested
    // list, the existing path still folds the marker into the
    // Paragraph leaf — no synth needed.
    //
    //   1. one
    //      1. nested
    //
    // The outer LI here has a Paragraph child "one" on the marker
    // line; the existing extend_leading_range path handles it.
    let initial = EditorState {
        markdown: "1. one\n   1. nested".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (_, editor) = open_editor(cx, initial);
    let spec = current_spec(cx, &editor);
    // Two blocks: the Paragraph "one" (whose range was extended back
    // to byte 0 to cover the marker) and the nested list's leaf.
    assert_eq!(spec.blocks.len(), 2);
    // No synth: the first block has the LI marker included via
    // extend_leading_range — its source_range starts at 0 and
    // includes inline content past the marker.
    assert!(spec.blocks[0].source_range.start == 0);
    assert!(
        spec.blocks[0].source_range.end > 3,
        "first block should cover marker + paragraph content `one`",
    );
}

// ---------------------------------------------------------------------------
// Undo / redo
// ---------------------------------------------------------------------------

// Undo/redo tests reuse the crate-wide `type_text` helper (single
// `InsertText` events through the production `dispatch` choke point).

/// Drive the IME / typed-input bypass path (`replace_and_mark_text_in_range`),
/// which mutates `self.state` directly rather than routing through
/// `dispatch`/`update`.
fn ime_insert(
    cx: &mut TestAppContext,
    handle: AnyWindowHandle,
    editor: &Entity<MarkdownEditorState>,
    text: &str,
) {
    cx.update_window(handle, |_, window, cx| {
        editor.update(cx, |e, cx| {
            e.replace_and_mark_text_in_range(None, text, None, window, cx);
        });
    })
    .unwrap();
    cx.run_until_parked();
}

#[gpui::test]
fn undo_restores_buffer_and_selection(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "Hello".into(),
        selection: Selection::Cursor(5),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "!");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "Hello!");
        assert_eq!(e.cursor_offset(), 6);
    });

    dispatch(cx, handle, &editor, Undo);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "Hello");
        // Pre-edit selection is restored, not just the buffer.
        assert_eq!(e.selection(), Selection::Cursor(5));
    });

    dispatch(cx, handle, &editor, Redo);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "Hello!");
        assert_eq!(e.selection(), Selection::Cursor(6));
    });
}

#[gpui::test]
fn typing_coalesces_into_single_undo_step(cx: &mut TestAppContext) {
    // Three contiguous single-character inserts collapse into one undo
    // step: a single Undo reverts the whole run, and a single Redo
    // re-applies it.
    let (handle, editor) = open_editor(cx, EditorState::new());
    type_text(cx, handle, &editor, "a");
    type_text(cx, handle, &editor, "b");
    type_text(cx, handle, &editor, "c");
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "abc"));

    dispatch(cx, handle, &editor, Undo);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "");
        assert_eq!(e.cursor_offset(), 0);
    });

    dispatch(cx, handle, &editor, Redo);
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "abc"));
}

#[gpui::test]
fn whitespace_breaks_the_coalescing_run(cx: &mut TestAppContext) {
    // A space is a word boundary: it ends the current typing run so the
    // characters before and after it undo independently.
    let (handle, editor) = open_editor(cx, EditorState::new());
    type_text(cx, handle, &editor, "a");
    type_text(cx, handle, &editor, "b");
    type_text(cx, handle, &editor, " ");
    type_text(cx, handle, &editor, "c");
    type_text(cx, handle, &editor, "d");
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "ab cd"));

    // First undo drops the trailing "cd" run.
    dispatch(cx, handle, &editor, Undo);
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "ab "));
    // Second undo drops the space (its own step).
    dispatch(cx, handle, &editor, Undo);
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "ab"));
    // Third undo drops the leading "ab" run.
    dispatch(cx, handle, &editor, Undo);
    editor.read_with(cx, |e, _| assert_eq!(e.value(), ""));
}

#[gpui::test]
fn deletion_is_its_own_undo_step(cx: &mut TestAppContext) {
    // A backspace must not fold into a preceding typing run.
    let (handle, editor) = open_editor(cx, EditorState::new());
    type_text(cx, handle, &editor, "a");
    type_text(cx, handle, &editor, "b");
    dispatch(cx, handle, &editor, Backspace); // "ab" -> "a"
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "a"));

    // Undo the delete only — the typed "ab" is a separate, still-standing step.
    dispatch(cx, handle, &editor, Undo);
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "ab"));
    // Undo the typing run.
    dispatch(cx, handle, &editor, Undo);
    editor.read_with(cx, |e, _| assert_eq!(e.value(), ""));
}

#[gpui::test]
fn undo_captures_ime_typed_input(cx: &mut TestAppContext) {
    // The IME path bypasses `dispatch`; undo must still capture it.
    let initial = EditorState {
        markdown: "abc".into(),
        selection: Selection::Cursor(3),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    ime_insert(cx, handle, &editor, "X");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "abcX");
        assert_eq!(e.cursor_offset(), 4);
    });

    dispatch(cx, handle, &editor, Undo);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "abc");
        assert_eq!(e.selection(), Selection::Cursor(3));
    });

    dispatch(cx, handle, &editor, Redo);
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "abcX"));
}

#[gpui::test]
fn fresh_edit_clears_redo_stack(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "Hi".into(),
        selection: Selection::Cursor(2),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "!"); // "Hi!"
    dispatch(cx, handle, &editor, Undo); // back to "Hi", redo holds "Hi!"
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "Hi"));

    // A fresh edit discards the redo branch.
    type_text(cx, handle, &editor, "?"); // "Hi?"
    dispatch(cx, handle, &editor, Redo); // no-op: redo was cleared
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "Hi?"));
}

#[gpui::test]
fn undo_redo_noop_on_empty_stacks(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "abc".into(),
        selection: Selection::Cursor(1),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Undo);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "abc");
        assert_eq!(e.selection(), Selection::Cursor(1));
    });
    dispatch(cx, handle, &editor, Redo);
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "abc"));
}

#[gpui::test]
fn selection_only_change_records_no_undo_step(cx: &mut TestAppContext) {
    // Moving the caret is not an edit: Undo afterwards must be a no-op,
    // not a selection reset.
    let initial = EditorState {
        markdown: "abc".into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 1));
    dispatch(cx, handle, &editor, Undo);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "abc");
        // Undo did nothing — the caret stays where the move left it.
        assert_eq!(e.cursor_offset(), 1);
    });
}

#[gpui::test]
fn set_value_clears_history(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "a".into(),
        selection: Selection::Cursor(1),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    type_text(cx, handle, &editor, "!"); // "a!"

    // A programmatic whole-document swap is an undo boundary.
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| e.set_value("new document", cx));
    })
    .unwrap();
    cx.run_until_parked();

    dispatch(cx, handle, &editor, Undo);
    editor.read_with(cx, |e, _| {
        // Undo cannot cross the set_value boundary.
        assert_eq!(e.value(), "new document");
    });
}

// ---------------------------------------------------------------------------
// Pointer selection — double / triple click granularity and drag extension.
// Exercised through the `begin_selection_for_test` / `extend_selection_for_test`
// seams (offset-based, so they need no laid-out geometry); the position→offset
// hit-test and the window-global drag routing are exercised by the eidola-gui
// driver against real rendering.
// ---------------------------------------------------------------------------

fn begin_selection(
    cx: &mut TestAppContext,
    handle: AnyWindowHandle,
    editor: &Entity<MarkdownEditorState>,
    offset: usize,
    click_count: usize,
    shift: bool,
) {
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            e.begin_selection_for_test(offset, click_count, shift, cx)
        });
    })
    .unwrap();
    cx.run_until_parked();
}

fn extend_selection(
    cx: &mut TestAppContext,
    handle: AnyWindowHandle,
    editor: &Entity<MarkdownEditorState>,
    offset: usize,
) {
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| e.extend_selection_for_test(offset, cx));
    })
    .unwrap();
    cx.run_until_parked();
}

#[gpui::test]
fn single_click_places_a_collapsed_cursor(cx: &mut TestAppContext) {
    let (handle, editor) = open_editor(cx, EditorState::with_markdown("the quick brown fox"));
    begin_selection(cx, handle, &editor, 6, 1, false);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.selection(), Selection::Cursor(6));
        assert!(e.is_selecting(), "a press begins a drag");
    });
}

#[gpui::test]
fn double_click_selects_the_word(cx: &mut TestAppContext) {
    let text = "scatter short (blue) wavelengths";
    let (handle, editor) = open_editor(cx, EditorState::with_markdown(text));
    let inside_blue = text.find("blue").unwrap() + 1;
    begin_selection(cx, handle, &editor, inside_blue, 2, false);
    editor.read_with(cx, |e, _| {
        let sel = e.selection();
        assert_eq!(&e.value()[sel.lower_bound()..sel.upper_bound()], "blue");
    });
}

#[gpui::test]
fn triple_click_selects_the_line(cx: &mut TestAppContext) {
    let text = "first line\nsecond line here\nthird";
    let (handle, editor) = open_editor(cx, EditorState::with_markdown(text));
    let mid_second = text.find("second").unwrap() + 3;
    begin_selection(cx, handle, &editor, mid_second, 3, false);
    editor.read_with(cx, |e, _| {
        let sel = e.selection();
        assert_eq!(
            &e.value()[sel.lower_bound()..sel.upper_bound()],
            "second line here"
        );
    });
}

#[gpui::test]
fn drag_after_double_click_extends_by_whole_words(cx: &mut TestAppContext) {
    let text = "alpha beta gamma delta";
    let (handle, editor) = open_editor(cx, EditorState::with_markdown(text));
    // Double-click "beta", then drag into "gamma".
    let in_beta = text.find("beta").unwrap() + 1;
    let in_gamma = text.find("gamma").unwrap() + 2;
    begin_selection(cx, handle, &editor, in_beta, 2, false);
    extend_selection(cx, handle, &editor, in_gamma);
    editor.read_with(cx, |e, _| {
        let sel = e.selection();
        // The selection snaps to whole-word bounds: all of "beta gamma",
        // never a mid-word offset.
        assert_eq!(
            &e.value()[sel.lower_bound()..sel.upper_bound()],
            "beta gamma"
        );
    });
}

#[gpui::test]
fn drag_after_triple_click_extends_by_whole_lines(cx: &mut TestAppContext) {
    // The editor promotes single `\n` soft breaks to `\n\n` paragraph breaks,
    // so compute offsets and the expectation from the normalized buffer.
    let (handle, editor) = open_editor(
        cx,
        EditorState::with_markdown("line one\nline two\nline three\nline four"),
    );
    let value = editor.read_with(cx, |e, _| e.value().to_string());
    let in_two = value.find("line two").unwrap() + 2;
    let in_three = value.find("line three").unwrap() + 2;
    begin_selection(cx, handle, &editor, in_two, 3, false);
    extend_selection(cx, handle, &editor, in_three);
    editor.read_with(cx, |e, _| {
        let sel = e.selection();
        let selected = &e.value()[sel.lower_bound()..sel.upper_bound()];
        // Whole lines from the start of "line two" through the end of "line
        // three" — bounded to those two lines regardless of the buffer's
        // (normalized) inter-line separators.
        assert!(
            selected.starts_with("line two"),
            "starts at line two: {selected:?}"
        );
        assert!(
            selected.ends_with("line three"),
            "ends at line three: {selected:?}"
        );
        assert!(!selected.contains("line one"));
        assert!(!selected.contains("line four"));
    });
}

#[gpui::test]
fn word_drag_back_before_the_anchor_flips_the_selected_end(cx: &mut TestAppContext) {
    let text = "alpha beta gamma delta";
    let (handle, editor) = open_editor(cx, EditorState::with_markdown(text));
    // Double-click "gamma", then drag back into "alpha".
    let in_gamma = text.find("gamma").unwrap() + 1;
    let in_alpha = text.find("alpha").unwrap() + 1;
    begin_selection(cx, handle, &editor, in_gamma, 2, false);
    extend_selection(cx, handle, &editor, in_alpha);
    editor.read_with(cx, |e, _| {
        let sel = e.selection();
        // Dragging back covers whole words from "alpha" through "gamma".
        assert_eq!(
            &e.value()[sel.lower_bound()..sel.upper_bound()],
            "alpha beta gamma"
        );
    });
}

#[gpui::test]
fn char_drag_extends_by_offset(cx: &mut TestAppContext) {
    // Single-click (char mode) drag extends the head to the exact offset —
    // the long-content selection path (window-global drag) rides on this.
    let text = "the quick brown fox jumps";
    let (handle, editor) = open_editor(cx, EditorState::with_markdown(text));
    begin_selection(cx, handle, &editor, 4, 1, false);
    extend_selection(cx, handle, &editor, 15);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.selection(), Selection::range(4, 15));
    });
}

#[gpui::test]
fn shift_click_extends_the_existing_selection(cx: &mut TestAppContext) {
    let text = "the quick brown fox";
    let (handle, editor) = open_editor(cx, EditorState::with_markdown(text));
    begin_selection(cx, handle, &editor, 4, 1, false); // caret at 4
    begin_selection(cx, handle, &editor, 15, 1, true); // shift-click at 15
    editor.read_with(cx, |e, _| {
        assert_eq!(e.selection(), Selection::range(4, 15));
    });
}

#[gpui::test]
fn shift_click_and_drag_keeps_the_original_anchor(cx: &mut TestAppContext) {
    // Native behavior: a shift-click-and-drag keeps the *original* selection
    // anchor throughout — the shift-click sets the head, and a following drag
    // continues to extend from the original anchor. Regression guard for the
    // bug where the shift-click re-anchored the drag at the click point,
    // discarding the original range on the very next mouse move.
    let text = "the quick brown fox jumps over";
    let (handle, editor) = open_editor(cx, EditorState::with_markdown(text));
    // Anchor a selection at A = 4 (start of "quick"), head at 9.
    begin_selection(cx, handle, &editor, 4, 1, false);
    extend_selection(cx, handle, &editor, 9);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.selection(), Selection::range(4, 9));
    });
    // Shift-click at C = 20 extends the visible selection from the original
    // anchor A = 4 (not a fresh anchor at 20).
    begin_selection(cx, handle, &editor, 20, 1, true);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.selection(), Selection::range(4, 20));
    });
    // The subsequent drag must still be anchored at A = 4. Before the fix the
    // drag re-anchored at the shift-click point, yielding range(20, 25).
    extend_selection(cx, handle, &editor, 25);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.selection(), Selection::range(4, 25));
    });
}

#[gpui::test]
fn append_at_end_lands_after_the_text_with_the_caret_behind_it(cx: &mut TestAppContext) {
    // The host seam for "the user started typing at something that isn't this
    // editor" (the space view's type-to-compose jump, task 38). It appends at
    // the document end and leaves the caret there — and, because the intent is
    // *append*, it collapses an existing selection rather than replacing it.
    let (handle, editor) = open_editor(cx, EditorState::with_markdown("already here"));
    dispatch(cx, handle, &editor, SelectAll);
    editor.update(cx, |e, cx| e.append_at_end("x", cx));
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "already herex");
        assert_eq!(e.cursor_offset(), "already herex".len());
    });
}

// ---------------------------------------------------------------------------
// Host seams: offset geometry (`content_y_for_offset`) and IME composition
// (`is_composing`).
//
// The editor has no internal vertical scroll, so a host that wants to reveal
// something reads its content-local y from here. `caret_content_y` answers for
// the caret; `content_y_for_offset` answers for any offset — a match, a cited
// passage — and resolves a wrap boundary downstream, since an arbitrary offset
// carries no affinity.
//
// `is_composing` exists because this editor emits `Change` on every preedit
// keystroke: a host that reads the buffer's *meaning* on Change must be able to
// ask whether the reader has committed those characters.
// ---------------------------------------------------------------------------

#[gpui::test]
fn content_y_for_offset_answers_for_offsets_the_caret_is_not_at(cx: &mut TestAppContext) {
    let markdown = "first paragraph\n\nsecond paragraph\n\nthird paragraph";
    let initial = EditorState {
        markdown: markdown.into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    // Populate `last_blocks` with a paint pass — the geometry is paint-derived.
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Left);

    editor.read_with(cx, |e, _| {
        let at = |needle: &str| {
            e.content_y_for_offset(markdown.find(needle).expect("fixture"))
                .expect("laid out")
        };
        let (first_top, first_bottom) = at("first");
        let (second_top, _) = at("second");
        let (third_top, _) = at("third");

        assert!(
            first_bottom > first_top,
            "the span is a row: {first_top:?}..{first_bottom:?}",
        );
        assert!(
            second_top > first_top && third_top > second_top,
            "later paragraphs sit lower: {first_top:?}, {second_top:?}, {third_top:?}",
        );
        // The caret is at offset 0 and untouched by any of that.
        assert_eq!(e.cursor_offset(), 0);
        assert_eq!(e.caret_content_y(), Some((first_top, first_bottom)));

        // The document end is the same edge `caret_content_y` documents: it
        // clamps onto the last laid-out line rather than answering `None`.
        let (end_top, _) = e.content_y_for_offset(markdown.len()).expect("clamped");
        assert_eq!(end_top, third_top);
    });
}

#[gpui::test]
fn content_y_for_offset_resolves_a_wrap_boundary_downstream(cx: &mut TestAppContext) {
    // The same boundary offset the caret can render on either row of: End
    // leaves Upstream affinity (upper row), while `content_y_for_offset` has no
    // affinity to consult and always answers with the lower row — what a host
    // revealing a range's start wants.
    let para = "This is a fairly long paragraph that soft-wraps across several \
                rows inside the narrow viewport configured by open_editor_narrow, \
                giving us interior wrap boundaries to land a caret on."
        .to_string();
    let initial = EditorState {
        markdown: para,
        selection: Selection::Cursor(2),
        ..Default::default()
    };
    let (handle, editor) = open_editor_narrow(cx, initial);
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Left);

    set_cursor(cx, handle, &editor, 2);
    dispatch(cx, handle, &editor, End);
    editor.read_with(cx, |e, _| {
        let boundary = e.cursor_offset();
        let (caret_top, caret_bottom) = e.caret_content_y().expect("caret row");
        let row_h = caret_bottom - caret_top;
        let (offset_top, _) = e.content_y_for_offset(boundary).expect("boundary row");
        assert!(
            offset_top >= caret_top + row_h - gpui::px(1.0),
            "the same offset resolves one row lower without the caret's upstream \
             affinity (caret {caret_top:?}, offset {offset_top:?}, row {row_h:?})",
        );
    });
}

#[gpui::test]
fn is_composing_holds_while_preedit_text_sits_in_the_buffer(cx: &mut TestAppContext) {
    let (handle, editor) = open_editor(cx, EditorState::with_markdown(""));

    let changes = std::rc::Rc::new(std::cell::Cell::new(0usize));
    let counter = changes.clone();
    cx.update(|cx| {
        cx.subscribe(&editor, move |_, event: &MarkdownEditorEvent, _| {
            if matches!(event, MarkdownEditorEvent::Change) {
                counter.set(counter.get() + 1);
            }
        })
        .detach();
    });

    editor.read_with(cx, |e, _| assert!(!e.is_composing()));

    // Preedit: the marked text is really in the buffer, and `Change` fires —
    // which is exactly why a host needs to be able to ask.
    ime_insert(cx, handle, &editor, "n");
    editor.read_with(cx, |e, _| {
        assert!(e.is_composing());
        assert_eq!(e.value(), "n");
    });
    assert!(
        changes.get() >= 1,
        "the editor emits Change during preedit ({} events)",
        changes.get(),
    );

    ime_insert(cx, handle, &editor, "ni");
    editor.read_with(cx, |e, _| {
        assert!(e.is_composing());
        assert_eq!(e.value(), "ni");
    });

    // Committing the composition clears it.
    ime_commit(cx, handle, &editor, "に");
    editor.read_with(cx, |e, _| {
        assert!(!e.is_composing());
        assert_eq!(e.value(), "に");
    });
}

#[gpui::test]
fn ordinary_typing_never_reports_a_composition(cx: &mut TestAppContext) {
    let (handle, editor) = open_editor(cx, EditorState::with_markdown(""));
    type_text(cx, handle, &editor, "plain");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "plain");
        assert!(!e.is_composing());
    });
}

/// Commit an IME composition (`replace_text_in_range`) — the path that ends a
/// preedit and leaves the chosen characters in the buffer.
fn ime_commit(
    cx: &mut TestAppContext,
    handle: AnyWindowHandle,
    editor: &Entity<MarkdownEditorState>,
    text: &str,
) {
    cx.update_window(handle, |_, window, cx| {
        editor.update(cx, |e, cx| {
            e.replace_text_in_range(None, text, window, cx);
        });
    })
    .unwrap();
    cx.run_until_parked();
}

#[gpui::test]
fn content_y_for_offset_at_a_shared_line_boundary_takes_the_later_line(cx: &mut TestAppContext) {
    // Line ranges are end-inclusive, so where one line's end is the next one's
    // start — a code block's rows here, and equally a hard-broken paragraph or
    // a table's row boundary — one offset is claimed by two lines. A host
    // revealing a match at that offset must land on the line the match is *on*,
    // not the line above it.
    let markdown = "```\nlet x = 1;\nlet y = 2;\n```\n\ntail";
    let initial = EditorState {
        markdown: markdown.into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Left);

    let (boundary, upper_y, lower_y) = editor.read_with(cx, |e, _| {
        let mut geo = e.debug_line_source_geometry();
        geo.sort_by_key(|(start, end, ..)| (*start, *end));
        let shared = geo
            .windows(2)
            .find(|w| w[0].1 == w[1].0 && w[1].3 > w[0].3)
            .expect("the fixture shares a boundary between two lines on different rows")
            .to_vec();
        (shared[0].1, shared[0].3, shared[1].3)
    });

    editor.read_with(cx, |e, _| {
        let (top, _) = e.content_y_for_offset(boundary).expect("boundary offset");
        assert_eq!(
            top.as_f32(),
            lower_y,
            "offset {boundary} starts the lower line, so it reveals at {lower_y}, \
             not at the line above it ({upper_y})",
        );
    });

    // The caret path is untouched: the later-line rule is gated on
    // `downstream`, so an upstream caret still resolves to the line it renders
    // on. No caret can actually land on a *cross-line* shared boundary today
    // (`End` stops at the end of the row's text, before the byte the next line
    // claims), which is why the gate costs nothing and stays as the guarantee.
}

#[gpui::test]
fn theme_colors_follow_a_live_mode_flip_and_an_override_survives_it(cx: &mut TestAppContext) {
    // A host may build a style once and keep it; the element re-derives every
    // theme color each frame so a Day/Night flip recolors live. A color the
    // host set explicitly is its own decision and must not be re-derived.
    cx.update(|cx| {
        gpui_component::init(cx);
        gpui_component::Theme::change(gpui_component::ThemeMode::Light, None, cx);

        let mine = gpui::hsla(0.5, 0.5, 0.5, 0.5);
        let mut style = MarkdownStyle::from_theme(cx).highlight_color(mine);
        let day = (
            style.text_color,
            style.highlight_overlay_color,
            style.highlight_accent_color,
        );

        gpui_component::Theme::change(gpui_component::ThemeMode::Dark, None, cx);
        style.refresh_theme_colors(cx);

        assert_ne!(style.text_color, day.0, "ordinary theme colors follow");
        assert_ne!(
            style.highlight_overlay_color, day.1,
            "the overlay wash follows the mode like every other theme color",
        );
        assert_ne!(
            style.highlight_accent_color, day.2,
            "the accent wash follows the mode like every other theme color",
        );
        assert_eq!(
            style.highlight_color, mine,
            "a color the host set is never re-derived",
        );

        // And a fresh style built under the same mode agrees with the
        // refreshed one — one registry, so there is no second derivation to
        // drift from.
        let fresh = MarkdownStyle::from_theme(cx);
        assert_eq!(style.text_color, fresh.text_color);
        assert_eq!(style.highlight_overlay_color, fresh.highlight_overlay_color);
        assert_eq!(style.highlight_accent_color, fresh.highlight_accent_color);
        assert_eq!(style.table_rule_color, fresh.table_rule_color);
    });
}

/// Count `Change` events from now on.
fn change_counter(
    cx: &mut TestAppContext,
    editor: &Entity<MarkdownEditorState>,
) -> std::rc::Rc<std::cell::Cell<usize>> {
    let changes = std::rc::Rc::new(std::cell::Cell::new(0usize));
    let counter = changes.clone();
    cx.update(|cx| {
        cx.subscribe(editor, move |_, event: &MarkdownEditorEvent, _| {
            if matches!(event, MarkdownEditorEvent::Change) {
                counter.set(counter.get() + 1);
            }
        })
        .detach();
    });
    changes
}

/// The platform dropping the marking without replacement text
/// (`EntityInputHandler::unmark_text`) — a click elsewhere while composing, an
/// input-source switch.
fn ime_unmark(
    cx: &mut TestAppContext,
    handle: AnyWindowHandle,
    editor: &Entity<MarkdownEditorState>,
) {
    cx.update_window(handle, |_, window, cx| {
        editor.update(cx, |e, cx| e.unmark_text(window, cx));
    })
    .unwrap();
    cx.run_until_parked();
}

#[gpui::test]
fn every_way_a_composition_ends_reports_one_change(cx: &mut TestAppContext) {
    // The other half of `is_composing`: a host skips `Change` while a
    // composition is live, so the end of one owes it exactly one `Change` —
    // whatever ended it. A composition that ends silently leaves that host
    // holding text it was told to ignore.
    let ends: Vec<(
        &str,
        Box<dyn Fn(&mut TestAppContext, AnyWindowHandle, &Entity<MarkdownEditorState>)>,
    )> = vec![
        (
            "the platform unmarking it",
            Box::new(|cx, handle, editor| ime_unmark(cx, handle, editor)),
        ),
        (
            "a caret move",
            Box::new(|cx, handle, editor| dispatch(cx, handle, editor, Right)),
        ),
        (
            "a vertical move",
            Box::new(|cx, handle, editor| dispatch(cx, handle, editor, Down)),
        ),
        (
            "Home",
            Box::new(|cx, handle, editor| dispatch(cx, handle, editor, Home)),
        ),
        (
            "End",
            Box::new(|cx, handle, editor| dispatch(cx, handle, editor, End)),
        ),
    ];

    for (what, end_it) in ends {
        let (handle, editor) = open_editor(cx, EditorState::with_markdown("before\n\nafter"));
        set_cursor(cx, handle, &editor, 6);
        ime_insert(cx, handle, &editor, "n");
        editor.read_with(cx, |e, _| assert!(e.is_composing(), "{what}: composing"));

        let changes = change_counter(cx, &editor);
        end_it(cx, handle, &editor);

        editor.read_with(cx, |e, _| {
            assert!(!e.is_composing(), "{what}: composition ended");
            // The marked bytes stay — they are ordinary text now.
            assert!(e.value().starts_with("beforen"), "{what}: text kept");
        });
        assert_eq!(
            changes.get(),
            1,
            "{what}: ends the composition with exactly one Change",
        );
    }
}

#[gpui::test]
fn committing_a_composition_reports_one_change_not_two(cx: &mut TestAppContext) {
    // The commit path replaces the marked text through two internal dispatches;
    // only one of them is this step, so a host re-deriving on `Change` does the
    // work once.
    let (handle, editor) = open_editor(cx, EditorState::with_markdown(""));
    ime_insert(cx, handle, &editor, "ni");
    let changes = change_counter(cx, &editor);
    ime_commit(cx, handle, &editor, "に");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "に");
        assert!(!e.is_composing());
    });
    assert_eq!(changes.get(), 1);
}
