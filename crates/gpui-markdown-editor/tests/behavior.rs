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
    ShiftEnter, ShiftRight, ShiftTab, Tab, Up,
};
use gpui_markdown_editor::{
    BlockKind, Container, EditorState, ListItemKind, MarkdownEditor, RenderSpec, Selection,
};

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

    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "\n\n");
        assert_eq!(e.cursor_offset(), 2);
    });
    let spec1 = current_spec(cx, &editor);
    assert_eq!(spec1.blocks.len(), 2);

    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| assert_eq!(e.state.markdown, "\n\n\n\n"));
    let spec2 = current_spec(cx, &editor);
    assert_eq!(spec2.blocks.len(), 3);

    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| assert_eq!(e.state.markdown, "\n\n\n\n\n\n"));
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "```rust\nlet x = 1;\n\n```");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "```\nx\n```\n\npa\n\nra");
    });
}

#[gpui::test]
fn code_block_renders_as_code_block_kind(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "```rust\nlet x = 1;\n```".into(),
        selection: Selection::Cursor(0),
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "```\nline1\nline2\n```");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Delete);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "```\nline1\nline2\n```");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _window, cx| {
        editor.update(cx, |e, cx| {
            let next = std::mem::take(&mut e.state);
            // Position 10 is the interior of the `\n\n` pair
            // separating "line1" and "line2".
            e.state = gpui_markdown_editor::update::update(
                next,
                gpui_markdown_editor::EditorEvent::SetSelection(Selection::Cursor(10)),
            );
            cx.notify();
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
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _window, cx| {
        editor.update(cx, |e, cx| {
            // Simulate a paste of multiline source by setting the
            // markdown directly, then running the post-pass via
            // `update` to verify it doesn't promote the inner `\n`s.
            e.state.markdown = "```\nline1\nline2\nline3\n```".into();
            e.state.selection = Selection::Cursor(20);
            let next = std::mem::take(&mut e.state);
            e.state = gpui_markdown_editor::update::update(
                next,
                gpui_markdown_editor::EditorEvent::SetSelection(Selection::Cursor(20)),
            );
            cx.notify();
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "```\nline1\nline2\nline3\n```");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            let next = std::mem::take(&mut e.state);
            e.state = gpui_markdown_editor::update::update(
                next,
                gpui_markdown_editor::EditorEvent::InsertText("!".into()),
            );
            cx.notify();
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> hello!");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> hello\n> \n> ");
        assert_eq!(e.cursor_offset(), 13);
        // The paragraph the cursor sits in still belongs to a
        // blockquote (depth 1).
        assert_eq!(
            gpui_markdown_editor::update::blockquote_depth_at(
                &e.state.markdown,
                e.cursor_offset(),
            ),
            1,
        );
    });
}

#[gpui::test]
fn enter_inside_nested_blockquote_keeps_depth(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "> > deep".into(),
        selection: Selection::Cursor(8),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> > deep\n> > \n> > ");
        assert_eq!(
            gpui_markdown_editor::update::blockquote_depth_at(
                &e.state.markdown,
                e.cursor_offset(),
            ),
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftEnter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> hello  \n> ");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> hello\n\n");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> > deep\n> \n> ");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // depth 2 → 1
        assert_eq!(e.state.markdown, "> > deep\n> \n> ");
        assert_eq!(e.cursor_offset(), 14);
    });
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // depth 1 → 0
        assert_eq!(e.state.markdown, "> > deep\n\n");
        assert_eq!(e.cursor_offset(), 10);
    });
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // depth 0 break gets the original atomic pair delete; the
        // trailing empty paragraph merges into "deep".
        assert_eq!(e.state.markdown, "> > deep");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> one\n\ntwo");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "p1p2");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "para\n\n>hi");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    // Walk the paragraph from depth 3 → 2 → 1 → 0. At each step the
    // pair is still balanced (the post-outdent buffer is depth-D pair
    // structure for some D), so `enforce_invariants` doesn't insert
    // any synthetic prefixes.
    for _ in 0..3 {
        dispatch(cx, handle, &editor, Backspace);
        editor.read_with(cx, |e, _| {
            let bytes = e.state.markdown.as_bytes();
            for p in 0..bytes.len() {
                assert!(
                    !is_soft_break(bytes, p),
                    "soft break at byte {p} in {:?}",
                    e.state.markdown,
                );
            }
        });
    }
    editor.read_with(cx, |e, _| {
        // Final state: depth-0 pair separating the two paragraphs.
        assert_eq!(e.state.markdown, "> > > p1\n\np2");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            let next = std::mem::take(&mut e.state);
            e.state = gpui_markdown_editor::update::update(
                next,
                gpui_markdown_editor::EditorEvent::InsertText("p2".into()),
            );
            cx.notify();
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> p1\n> \n> p2");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            let next = std::mem::take(&mut e.state);
            e.state = gpui_markdown_editor::update::update(
                next,
                gpui_markdown_editor::EditorEvent::InsertText(" ".into()),
            );
            cx.notify();
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        // Exactly one space inserted — the trailing pair structure
        // is unchanged.
        assert_eq!(e.state.markdown, "> blockquote \n> \n> ");
        assert_eq!(e.cursor_offset(), 13);
    });

    // Idempotent: a no-op SetSelection (mouse move, click handler
    // re-feeding the same offset) re-runs `enforce_invariants`. The
    // buffer must stay identical — no fresh promotion fires.
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            let sel = e.state.selection;
            let next = std::mem::take(&mut e.state);
            e.state = gpui_markdown_editor::update::update(
                next,
                gpui_markdown_editor::EditorEvent::SetSelection(sel),
            );
            cx.notify();
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> blockquote \n> \n> ");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            let next = std::mem::take(&mut e.state);
            e.state = gpui_markdown_editor::update::update(
                next,
                gpui_markdown_editor::EditorEvent::InsertText(">".into()),
            );
            cx.notify();
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        // Exactly one `>` appended. Trailing pair structure is
        // unchanged (still 4 lines: content + middle marker + new
        // marker line with the typed `>`).
        assert_eq!(e.state.markdown, "> level 1\n> \n> >");
        assert_eq!(e.cursor_offset(), 16);
    });

    // Idempotent on re-update.
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            let sel = e.state.selection;
            let next = std::mem::take(&mut e.state);
            e.state = gpui_markdown_editor::update::update(
                next,
                gpui_markdown_editor::EditorEvent::SetSelection(sel),
            );
            cx.notify();
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> level 1\n> \n> >");
    });

    // The reported follow-on: right-arrow navigation must not
    // trigger fresh promotion either.
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> level 1\n> \n> >");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            let next = std::mem::take(&mut e.state);
            e.state = gpui_markdown_editor::update::update(
                next,
                gpui_markdown_editor::EditorEvent::InsertText(">".into()),
            );
            cx.notify();
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> Level 1\n> \n> >\n> \n> Level 1");
        assert_eq!(e.cursor_offset(), 16);
    });

    // Right-arrow shouldn't change the buffer at all (only selection
    // moves). The original bug expanded the buffer by ~3 lines on each
    // arrow press.
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> Level 1\n> \n> >\n> \n> Level 1");
    });
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> Level 1\n> \n> >\n> \n> Level 1");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            let next = std::mem::take(&mut e.state);
            e.state = gpui_markdown_editor::update::update(
                next,
                gpui_markdown_editor::EditorEvent::InsertText(">".into()),
            );
            cx.notify();
        });
    })
    .unwrap();
    cx.run_until_parked();
    // Result: line 2 deepens to `> > ` (normalize adds the trailing
    // space because the cursor moves off it after insertion). All other
    // lines untouched.
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> Level 1\n> > \n> \n> \n> Level 1");
    });
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Right);
    dispatch(cx, handle, &editor, Right);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> Level 1\n> > \n> \n> \n> Level 1");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            let next = std::mem::take(&mut e.state);
            e.state = gpui_markdown_editor::update::update(
                next,
                gpui_markdown_editor::EditorEvent::SetSelection(Selection::Cursor(7)),
            );
            cx.notify();
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Delete);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> helloworld");
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

// ---------------------------------------------------------------------------
// Lists
// ---------------------------------------------------------------------------

#[gpui::test]
fn unordered_list_renders_one_container_per_item(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "- foo\n- bar\n".into(),
        selection: Selection::Cursor(0),
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "- foo\n- ");
        assert_eq!(e.cursor_offset(), 8);
    });
}

#[gpui::test]
fn enter_at_end_of_ordered_item_increments_number(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "1. foo".into(),
        selection: Selection::Cursor(6),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "1. foo\n2. ");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            let next = std::mem::take(&mut e.state);
            e.state = gpui_markdown_editor::update::update(
                next,
                gpui_markdown_editor::EditorEvent::InsertText("X".into()),
            );
            cx.notify();
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "- fooX\n- bar");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> - foo\n> - ");
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
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "9. nine\n10. foo  \n    bar");
}

#[gpui::test]
fn ordered_marker_narrowing_reindents_continuations(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "1. foo  \n    bar".into(),
        selection: Selection::Cursor(0),
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "- foo\n\n");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        // Result: BQ paragraph + BQ paragraph break + new empty BQ
        // line. The depth-D pair shape `\n> \n> ` carries the BQ
        // forward without re-introducing a list marker.
        assert_eq!(e.state.markdown, "> - foo\n> \n> ");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "foo");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "1. Item one\n\nItem two");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "- a\n- b");
    });
}

#[gpui::test]
fn backspace_at_start_of_ordered_item_strips_marker(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "1. foo".into(),
        selection: Selection::Cursor(3),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "foo");
        assert_eq!(e.cursor_offset(), 0);
    });
}

// ---- Two consecutive hard breaks → paragraph break ----------------

#[gpui::test]
fn two_hard_breaks_at_top_level_become_paragraph_break(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "foo  \n  \nbar".into(),
        selection: Selection::Cursor(0),
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftEnter);
    dispatch(cx, handle, &editor, ShiftEnter);
    editor.read_with(cx, |e, _| {
        // After the second Shift+Enter the buffer is
        // `- foo  \n    \n  ` — two consecutive hard breaks. The
        // collapse pass drops both trailing-`  `s. The blank line
        // *between* the breaks (not the trailing one — cursor sits
        // there) gets its residual whitespace stripped, producing
        // the strictly-canonical `- foo\n\n  ` paragraph break.
        assert_eq!(e.state.markdown, "- foo\n\n  ");
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
            kind: ListItemKind::Unordered(b'-'),
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "- outer\n  - nested\n  - ");
    });
}

// ---- Tab / Shift+Tab nesting changes -------------------------------

#[gpui::test]
fn tab_nests_top_level_item_under_previous_sibling(cx: &mut TestAppContext) {
    // Cursor on second item; Tab nests it under the first.
    let initial = EditorState {
        markdown: "- one\n- two".into(),
        selection: Selection::Cursor(8),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "- one\n  - two");
    });
}

#[gpui::test]
fn tab_on_first_item_is_a_noop(cx: &mut TestAppContext) {
    // No previous sibling at the same depth — Tab does nothing.
    let initial = EditorState {
        markdown: "- only".into(),
        selection: Selection::Cursor(2),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "- only");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "- one\n  - nested\n  - two");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "- a\n  - b\n    - c");
    });
}

#[gpui::test]
fn shift_tab_dedents_nested_item_to_sibling(cx: &mut TestAppContext) {
    // `- a\n  - b` cursor on b; Shift+Tab makes b a top-level
    // sibling of a.
    let initial = EditorState {
        markdown: "- a\n  - b".into(),
        selection: Selection::Cursor(9),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "- a\n- b");
    });
}

#[gpui::test]
fn shift_tab_dedents_triple_nested_to_double(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "- a\n  - b\n    - c".into(),
        selection: Selection::Cursor(17),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "- a\n  - b\n  - c");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "foo");
    });
}

#[gpui::test]
fn shift_tab_outside_a_list_is_a_noop(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "just a paragraph".into(),
        selection: Selection::Cursor(5),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "just a paragraph");
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
    };
    let (_, editor) = open_editor(cx, initial);
    let landed = cx.update(|cx| {
        editor.update(cx, |e, _| {
            let next = std::mem::take(&mut e.state);
            // Byte 12 = right after `\n`, real start of second line.
            let updated = gpui_markdown_editor::update::update(
                next,
                gpui_markdown_editor::EditorEvent::SetSelection(Selection::Cursor(12)),
            );
            e.state = updated;
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
    };
    let (_, editor) = open_editor(cx, initial);
    let final_state = cx.update(|cx| {
        editor.update(cx, |e, _| {
            let next = std::mem::take(&mut e.state);
            let updated = gpui_markdown_editor::update::update(
                next,
                gpui_markdown_editor::EditorEvent::SetSelection(Selection::Cursor(1)),
            );
            e.state = updated;
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftEnter);
    // After one Shift+Enter alone the existing item-2 line break
    // must NOT be misread as a second hard break — that's the
    // false-positive the user hit. The buffer should still have
    // a real hard-break continuation in item 1, with item 2
    // intact below.
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "1. Item one  \n   \n2. Item two");
    });
    dispatch(cx, handle, &editor, ShiftEnter);
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            let next = std::mem::take(&mut e.state);
            e.state = gpui_markdown_editor::update::update(
                next,
                gpui_markdown_editor::EditorEvent::InsertText("A".into()),
            );
            cx.notify();
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
        assert_eq!(e.state.markdown, "1. Item one\n\n   A\n2. Item two");
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
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "- foo");
}

#[gpui::test]
fn multiple_extra_spaces_after_marker_are_stripped(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "-    foo".into(),
        selection: Selection::Cursor(8),
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "- foo");
}

#[gpui::test]
fn extra_space_after_ordered_marker_is_stripped(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "1.  foo".into(),
        selection: Selection::Cursor(7),
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
    };
    let final_state = run_enforce_invariants(cx, initial);
    assert_eq!(final_state.markdown, "-  foo");
    // Cursor at the content edge (past the gap): legitimate "fix
    // it" cursor position. Strip fires.
    let initial = EditorState {
        markdown: "-  foo".into(),
        selection: Selection::Cursor(6),
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "1. Item one\n   1. ");
    });
    // Type one character to give pulldown content to parse.
    cx.update_window(handle, |_, _, cx| {
        editor.update(cx, |e, cx| {
            let next = std::mem::take(&mut e.state);
            e.state = gpui_markdown_editor::update::update(
                next,
                gpui_markdown_editor::EditorEvent::InsertText("x".into()),
            );
            cx.notify();
        });
    })
    .unwrap();
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "1. Item one\n   1. x");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "1. one\n   1. nested-1\n   2. two");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "1. Item one\n2. Item one, one");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "- one\n  - two  \n    cont");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        // Single `\n` inserted; the list-item next-marker rule
        // doesn't fire inside code content.
        assert!(e.state.markdown.contains("```\n  cod\n"));
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "- one\n  - two\n    - three");
    });
}

#[gpui::test]
#[ignore = "known bug: Tab inside a BQ-list inserts indent before the `> ` marker, splitting the BQ scope"]
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "> - one\n>   - two");
    });
}

// ---- Shift+Tab at every nesting level ------------------------------------

#[gpui::test]
fn shift_tab_at_depth_3_dedents_to_depth_2(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "- one\n  - two\n    - three".into(),
        selection: Selection::Cursor(22),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "- one\n  - two\n  - three");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, ShiftTab);
    editor.read_with(cx, |e, _| {
        // The marker is dropped and the leading separator
        // becomes a depth-1 pair so the result stays inside the
        // BQ.
        assert_eq!(e.state.markdown, "> - one\n> \n> two");
    });
}

// ---- Empty-Enter at every nesting level ----------------------------------

#[gpui::test]
fn empty_enter_on_top_level_item_becomes_paragraph(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "- one\n- ".into(),
        selection: Selection::Cursor(8),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        // The empty item drops; cursor lands at a fresh empty
        // paragraph after the surviving item.
        assert_eq!(e.state.markdown, "- one\n\n");
    });
}

#[gpui::test]
fn empty_enter_on_apparent_nested_empty_item_creates_outer_sibling(cx: &mut TestAppContext) {
    // Edge case the corpus surfaces: pulldown doesn't open a
    // nested list for an *empty* `  - ` marker line — it needs
    // content to register as a nested list. So the source
    // `- one\n  - ` with the cursor at the apparent inner-marker
    // position is parsed by pulldown as the *outer* item with
    // continuation content, not as an empty nested item.
    //
    // Empty-Enter therefore can't see "inner empty item to exit"
    // and falls through to `enter_insertion`, which inserts the
    // outer-level next-sibling marker. The result is a new outer
    // sibling, leaving the apparent-nested-marker line in place
    // as continuation text — different from the visual intent
    // ("exit the nested level"), but consistent with what
    // pulldown sees.
    //
    // Documenting the boundary here so future implementations
    // that open a list-for-empty-marker (or that apply a
    // pre-parse heuristic to recognize the inner marker) have a
    // landing pad for the regression test.
    let initial = EditorState {
        markdown: "- one\n  - ".into(),
        selection: Selection::Cursor(10),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "- one\n  - \n- ");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        // The empty item drops and a depth-1 paragraph break
        // takes its place — the rest of the BQ stays intact.
        assert!(e.state.markdown.starts_with("> - one"));
        assert!(e.state.markdown.contains("\n> \n> "));
    });
}

// ---- Backspace-at-start at every nesting level ---------------------------

#[gpui::test]
fn backspace_at_start_of_top_level_item_makes_paragraph(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "- one".into(),
        selection: Selection::Cursor(2), // right after the marker
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // Marker is dropped; the line becomes a paragraph at top
        // level.
        assert_eq!(e.state.markdown, "one");
    });
}

#[gpui::test]
fn backspace_at_start_of_nested_item_dedents_to_outer(cx: &mut TestAppContext) {
    let initial = EditorState {
        markdown: "- one\n  - two".into(),
        selection: Selection::Cursor(10), // right after the inner marker
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        // The inner marker stays — Backspace at the marker end of
        // a nested item strips parent-marker-width leading
        // spaces, dedenting the item by one level.
        assert_eq!(e.state.markdown, "- one\n- two");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| assert_eq!(e.state.markdown, "1. asdf\n2. "));
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| assert_eq!(e.state.markdown, "1. asdf\n\n"));
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "1. asdf\n\n\n\n");
        // Cursor at the end so the user can keep typing on the
        // fresh row.
        assert_eq!(e.state.selection, Selection::Cursor(11));
    });
}

#[gpui::test]
fn split_list_renumbers_trailing_orphan_to_one(cx: &mut TestAppContext) {
    // User-reported scenario:
    //   1. one
    //   2. |two              <-- cursor after the "2. " marker
    //   3. three
    //
    // Backspace dedents item 2 to a paragraph, splitting the
    // ordered list. The trailing portion ("3. three") used to
    // keep its original number — pulldown reports the new
    // standalone list with `start=3` and the source preserves the
    // `3. ` marker, leaving the user with a leftover-numbered
    // orphan list.
    //
    // The split-orphan heuristic in `effective_list_start`
    // recognizes that any single-item ordered list with `start>1`
    // is almost always such a leftover from a list split (no user
    // intentionally types or pastes a one-item list at start>1
    // with the expectation that it stay there once the editor
    // canonicalizes), and renumbers it to start at 1.
    //
    // The reordering of `enforce_invariants` (promote soft breaks
    // *before* normalize_lists) is what makes the heuristic fire
    // on the post-dedent buffer: before the soft break is
    // promoted, pulldown sees `paragraph + lazy continuation`
    // rather than `paragraph + standalone list`, so normalize
    // wouldn't see the trailing list to renumber.
    let initial = EditorState {
        markdown: "1. one\n2. two\n3. three".into(),
        selection: Selection::Cursor(10),
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "1. one\n\ntwo\n\n1. three");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Enter);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.state.markdown, "1. one\n\n\n\ntwo\n\n1. three");
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.state.markdown,
            "1. level one\n\nlevel one\n\n1. level three",
        );
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Backspace);
    dispatch(cx, handle, &editor, Enter);
    dispatch(cx, handle, &editor, Tab);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.state.markdown,
            "1. level one\n\n\n\nlevel one\n\n1. level three",
        );
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
    };
    let (handle, editor) = open_editor(cx, initial);
    dispatch(cx, handle, &editor, Tab);
    editor.read_with(cx, |e, _| {
        // Item 2 nested under item 1; nested-child line carried
        // along to the new depth. Pulldown sees this as a
        // 3-level deep ordered list — exactly the user's mental
        // model after one Tab on the parent.
        assert_eq!(
            e.state.markdown,
            "1. level one\n   1. level one\n      1. level three",
        );
    });
}
