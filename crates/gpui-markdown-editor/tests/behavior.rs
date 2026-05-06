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
use gpui_markdown_editor::{
    BlockKind, Container, EditorState, MarkdownEditor, RenderSpec, Selection,
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
