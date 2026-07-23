//! Keystroke-level table behavior — the regression gate for GFM pipe
//! tables (see `src/table.rs` for the model and `AGENTS.md` for the
//! keystroke map). Every test drives the *real* update pipeline
//! (`update::update`), so each step exercises the full canonicalizer
//! (`enforce_invariants`, including `normalize_tables`) and the
//! forbidden-position snapping exactly as a keystroke would.

use gpui_markdown_editor::update::{update, update_readonly};
use gpui_markdown_editor::{EditorEvent, EditorState, Selection};

fn state(md: &str, cursor: usize) -> EditorState {
    EditorState {
        markdown: md.to_string(),
        selection: Selection::Cursor(cursor),
    }
}

/// Type `text` one character at a time through the update pipeline.
fn type_str(mut s: EditorState, text: &str) -> EditorState {
    for ch in text.chars() {
        s = update(s, EditorEvent::InsertText(ch.to_string()));
    }
    s
}

const CANONICAL: &str = "| a | b |\n| --- | --- |\n| c1 | c2 |\n";

// ---------------------------------------------------------------------------
// Construction — one keystroke at a time
// ---------------------------------------------------------------------------

#[test]
fn typing_a_header_line_then_enter_scaffolds_a_table() {
    let s = type_str(EditorState::new(), "| a | b |");
    assert_eq!(s.markdown, "| a | b |");
    let s = update(s, EditorEvent::InsertNewline);
    assert_eq!(s.markdown, "| a | b |\n| --- | --- |\n|  |  |");
    // Caret in the first body cell.
    let cell_point = s.markdown.rfind("|  |  |").unwrap() + 2;
    assert_eq!(s.selection, Selection::Cursor(cell_point));
}

#[test]
fn full_keyboard_construction_of_a_two_by_two_table() {
    // header → Enter (scaffold) → x → Tab → y → Tab (new row) → …
    let s = type_str(EditorState::new(), "| a | b |");
    let s = update(s, EditorEvent::InsertNewline);
    let s = type_str(s, "x");
    assert_eq!(s.markdown, "| a | b |\n| --- | --- |\n| x |  |");
    let s = update(s, EditorEvent::IncreaseListDepth); // Tab
    let s = type_str(s, "y");
    assert_eq!(s.markdown, "| a | b |\n| --- | --- |\n| x | y |");
    // Tab past the last cell appends a fresh empty row.
    let s = update(s, EditorEvent::IncreaseListDepth);
    assert_eq!(s.markdown, "| a | b |\n| --- | --- |\n| x | y |\n|  |  |");
    let s = type_str(s, "z");
    assert_eq!(s.markdown, "| a | b |\n| --- | --- |\n| x | y |\n| z |  |");
}

#[test]
fn enter_on_empty_last_row_exits_into_a_paragraph() {
    let s = type_str(EditorState::new(), "| a | b |");
    let s = update(s, EditorEvent::InsertNewline); // scaffold; caret in empty body row
    let s = update(s, EditorEvent::InsertNewline); // row is empty → exit
    assert_eq!(s.markdown, "| a | b |\n| --- | --- |\n\n");
    assert_eq!(s.selection, Selection::Cursor(s.markdown.len()));
    // Typing lands in a plain paragraph below the table.
    let s = type_str(s, "after");
    assert!(s.markdown.ends_with("\n\nafter"));
}

#[test]
fn scaffold_does_not_fire_on_plain_text_or_inside_code() {
    // Plain text line: Enter is a paragraph break.
    let s = type_str(EditorState::new(), "plain text");
    let s = update(s, EditorEvent::InsertNewline);
    assert_eq!(s.markdown, "plain text\n\n");

    // Inside a fenced code block, a pipe line is literal source.
    let fence = "```\n| a | b |\n```";
    let cursor = fence.find("b |").unwrap() + 3; // end of the pipe line
    let s = update(state(fence, cursor), EditorEvent::InsertNewline);
    assert!(
        !s.markdown.contains("---"),
        "no delimiter row inside a fence, got {:?}",
        s.markdown
    );
}

// ---------------------------------------------------------------------------
// Tab navigation
// ---------------------------------------------------------------------------

#[test]
fn tab_selects_the_next_cells_content_and_skips_the_delimiter_row() {
    // From header `a` → header `b` (selects the content).
    let s = update(state(CANONICAL, 2), EditorEvent::IncreaseListDepth);
    assert_eq!(s.selection, Selection::Range { anchor: 6, head: 7 });
    // From header `b` → first body cell (the delimiter row is skipped).
    let s = update(s, EditorEvent::IncreaseListDepth);
    let c1 = CANONICAL.find("c1").unwrap();
    assert_eq!(
        s.selection,
        Selection::Range {
            anchor: c1,
            head: c1 + 2
        }
    );
}

#[test]
fn shift_tab_returns_to_the_previous_cell() {
    let c1 = CANONICAL.find("c1").unwrap();
    let s = update(state(CANONICAL, c1), EditorEvent::DecreaseListDepth);
    assert_eq!(s.selection, Selection::Range { anchor: 6, head: 7 });
    // At the very first cell, Shift-Tab is a no-op.
    let s = update(state(CANONICAL, 2), EditorEvent::DecreaseListDepth);
    assert_eq!(s.markdown, CANONICAL);
    assert_eq!(s.selection, Selection::Cursor(2));
}

#[test]
fn tab_with_selected_cell_content_typing_replaces_it() {
    // Tab selects `b`; typing replaces the cell's content.
    let s = update(state(CANONICAL, 2), EditorEvent::IncreaseListDepth);
    let s = type_str(s, "new");
    assert!(s.markdown.starts_with("| a | new |\n"));
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

#[test]
fn enter_inserts_a_row_below_and_header_enter_inserts_first_body_row() {
    // From a body row: new row below it.
    let c1 = CANONICAL.find("c1").unwrap();
    let s = update(state(CANONICAL, c1 + 1), EditorEvent::InsertNewline);
    assert_eq!(
        s.markdown,
        "| a | b |\n| --- | --- |\n| c1 | c2 |\n|  |  |\n"
    );
    // From the header: the new row lands right after the delimiter.
    let s = update(state(CANONICAL, 2), EditorEvent::InsertNewline);
    assert_eq!(
        s.markdown,
        "| a | b |\n| --- | --- |\n|  |  |\n| c1 | c2 |\n"
    );
}

#[test]
fn backspace_in_an_empty_row_deletes_it() {
    let src = "| a | b |\n| --- | --- |\n|  |  |\n";
    let point = src.rfind("|  |  |").unwrap() + 2;
    let s = update(state(src, point), EditorEvent::DeleteBackward);
    assert_eq!(s.markdown, "| a | b |\n| --- | --- |\n");
    // Caret hops to the header's last cell end (the previous
    // navigable row).
    assert_eq!(s.selection, Selection::Cursor(7));
}

#[test]
fn backspace_at_body_first_cell_start_hops_up_without_deleting() {
    let c1 = CANONICAL.find("c1").unwrap();
    let s = update(state(CANONICAL, c1), EditorEvent::DeleteBackward);
    assert_eq!(s.markdown, CANONICAL);
    assert_eq!(s.selection, Selection::Cursor(7));
}

// ---------------------------------------------------------------------------
// Columns
// ---------------------------------------------------------------------------

#[test]
fn typing_pipe_inserts_a_column_through_the_whole_table() {
    // `|` at the end of header cell `a` splits a new column in after
    // it; the delimiter gains `---` in the same event and the body
    // rows are padded by the canonicalizer on the same keystroke.
    let s = update(state(CANONICAL, 3), EditorEvent::InsertText("|".into()));
    assert_eq!(
        s.markdown,
        "| a |  | b |\n| --- | --- | --- |\n| c1 |  | c2 |\n"
    );
    // Caret sits in the fresh empty header cell.
    assert_eq!(s.selection, Selection::Cursor(6));
}

#[test]
fn backspace_at_cell_start_removes_the_column_boundary_table_wide() {
    // The inverse of `typing_pipe_inserts_a_column…`: `|` followed by
    // Backspace round-trips to the original table.
    let s = update(state(CANONICAL, 3), EditorEvent::InsertText("|".into()));
    let s = update(s, EditorEvent::DeleteBackward);
    assert_eq!(s.markdown, CANONICAL);
}

#[test]
fn backslash_pipe_types_a_literal_pipe() {
    let s = update(state(CANONICAL, 3), EditorEvent::InsertText("\\".into()));
    let s = update(s, EditorEvent::InsertText("|".into()));
    assert!(
        s.markdown.starts_with("| a\\| | b |\n"),
        "got {:?}",
        s.markdown
    );
    // Still a 2-column table (the escaped pipe is cell content).
    assert!(s.markdown.contains("| --- | --- |"));
}

#[test]
fn alignment_is_edited_by_typing_colons_on_the_delimiter_row() {
    // Type `:` at the start of the first delimiter cell, then leave
    // the row — the canonicalizer regenerates the row in the compact
    // canonical form for the parsed alignment.
    let dash = CANONICAL.find("---").unwrap();
    let s = update(state(CANONICAL, dash), EditorEvent::InsertText(":".into()));
    assert!(s.markdown.contains(":---"), "got {:?}", s.markdown);
    let s = update(s, EditorEvent::SetSelection(Selection::Cursor(2)));
    assert_eq!(s.markdown, "| a | b |\n| :-- | --- |\n| c1 | c2 |\n");
}

// ---------------------------------------------------------------------------
// Caret motion across cells
// ---------------------------------------------------------------------------

#[test]
fn arrow_right_hops_across_the_cell_chrome() {
    // From `a`'s content end, Right lands at `b`'s content start.
    let s = update(state(CANONICAL, 3), EditorEvent::MoveRight);
    assert_eq!(s.selection, Selection::Cursor(6));
    // And Left hops back.
    let s = update(s, EditorEvent::MoveLeft);
    assert_eq!(s.selection, Selection::Cursor(3));
}

#[test]
fn arrow_right_at_row_end_lands_on_the_next_row() {
    // From header `b`'s end (7), Right walks the trailing ` |`, the
    // newline, and the next row's leading chrome in one hop. The next
    // row is the delimiter row, whose dash cells are editable content.
    let s = update(state(CANONICAL, 7), EditorEvent::MoveRight);
    let dash = CANONICAL.find("---").unwrap();
    assert_eq!(s.selection, Selection::Cursor(dash));
}

#[test]
fn forward_delete_at_cell_end_hops_to_the_next_cell() {
    let s = update(state(CANONICAL, 3), EditorEvent::DeleteForward);
    assert_eq!(s.markdown, CANONICAL);
    assert_eq!(s.selection, Selection::Cursor(6));
}

#[test]
fn word_delete_stays_inside_the_cell() {
    // DeleteWordBackward at `c1`'s end must not cross into the chrome
    // or the previous cell.
    let c1 = CANONICAL.find("c1").unwrap();
    let s = update(state(CANONICAL, c1 + 2), EditorEvent::DeleteWordBackward);
    assert_eq!(s.markdown, "| a | b |\n| --- | --- |\n|  | c2 |\n");
}

// ---------------------------------------------------------------------------
// Canonicalization
// ---------------------------------------------------------------------------

#[test]
fn sloppy_table_normalizes_when_the_caret_is_elsewhere() {
    let sloppy = "| a | b |\n| - | - |\n|c1|   c2   |\nplain after\n";
    // A caret outside the table (in the trailing paragraph)
    // canonicalizes every row.
    let after = sloppy.find("plain").unwrap();
    let s = update(
        state(sloppy, after),
        EditorEvent::SetSelection(Selection::Cursor(after)),
    );
    assert!(
        s.markdown
            .starts_with("| a | b |\n| --- | --- |\n| c1 | c2 |\n"),
        "got {:?}",
        s.markdown
    );
}

#[test]
fn normalization_never_fights_typing_on_the_caret_row() {
    // Typing a space at a cell's content end must survive the same
    // event's canonicalization pass (the caret row is skipped), so
    // multi-word cell content is typeable.
    let s = update(state(CANONICAL, 3), EditorEvent::InsertText(" ".into()));
    let s = update(s, EditorEvent::InsertText("x".into()));
    assert!(
        s.markdown.starts_with("| a x | b |\n"),
        "got {:?}",
        s.markdown
    );
}

#[test]
fn short_rows_are_padded_and_wide_body_rows_grow_the_header() {
    // A body row with fewer cells gains empties; a wider body row
    // grows the header + delimiter (lossless, never truncating).
    let src = "| a | b |\n| --- | --- |\n| c |\n| d | e | f |\nafter\n";
    let after = src.find("after").unwrap();
    let s = update(
        state(src, after),
        EditorEvent::SetSelection(Selection::Cursor(after)),
    );
    assert_eq!(
        s.markdown,
        "| a | b |  |\n| --- | --- | --- |\n| c |  |  |\n| d | e | f |\nafter\n"
    );
}

#[test]
fn canonical_tables_round_trip_untouched_through_every_navigation() {
    let mut s = state(CANONICAL, 0);
    for ev in [
        EditorEvent::SetSelection(Selection::Cursor(2)),
        EditorEvent::MoveRight,
        EditorEvent::MoveDown,
        EditorEvent::MoveLineEnd,
        EditorEvent::MoveDocumentEnd,
        EditorEvent::MoveDocumentStart,
    ] {
        s = update(s, ev);
        assert_eq!(s.markdown, CANONICAL, "buffer changed after {:?}", s);
    }
}

#[test]
fn readonly_table_is_never_rewritten_and_selection_applies() {
    let s = update_readonly(
        state(CANONICAL, 0),
        EditorEvent::SetSelection(Selection::Range {
            anchor: 2,
            head: 30,
        }),
    );
    assert_eq!(s.markdown, CANONICAL);
    // The selection survives (endpoints may snap off chrome, but a
    // real range remains for copy).
    assert!(s.selection.upper_bound() > s.selection.lower_bound());
}

#[test]
fn table_internal_newlines_survive_but_the_boundary_still_promotes() {
    // A paragraph glued directly under the table (no blank line)
    // would be absorbed as a table row by GFM; our canonicalizer
    // leaves internal separators alone. A table followed by a
    // *block* construct promotes the boundary newline to a pair.
    let src = "| a |\n| --- |\n| c |\n# heading";
    let s = update(
        state(src, 0),
        EditorEvent::SetSelection(Selection::Cursor(2)),
    );
    assert_eq!(s.markdown, "| a |\n| --- |\n| c |\n\n# heading");
}

// ---------------------------------------------------------------------------
// Paste
// ---------------------------------------------------------------------------

#[test]
fn paste_into_a_cell_stays_in_the_cell() {
    let s = update(
        state(CANONICAL, 3),
        EditorEvent::Paste {
            text: "multi\nline | piped".into(),
            internal: false,
        },
    );
    assert!(
        s.markdown.starts_with("| amulti line \\| piped | b |\n"),
        "got {:?}",
        s.markdown
    );
    // Still a table with the same structure.
    assert!(s.markdown.contains("| --- | --- |"));
}

// ---------------------------------------------------------------------------
// Render spec — chrome visibility and header emphasis
// ---------------------------------------------------------------------------

mod spec {
    use super::*;
    use gpui_markdown_editor::render::render_readonly;
    use gpui_markdown_editor::{BlockKind, parse, render};

    fn table_block(spec: &gpui_markdown_editor::RenderSpec) -> &gpui_markdown_editor::RenderBlock {
        spec.blocks
            .iter()
            .find(|b| matches!(b.kind, BlockKind::Table { .. }))
            .expect("a table block")
    }

    #[test]
    fn display_mode_hides_chrome_and_the_delimiter_row() {
        // Cursor outside the table.
        let src = format!("{CANONICAL}\nelsewhere");
        let cursor = src.len();
        let state = EditorState {
            markdown: src.clone(),
            selection: Selection::Cursor(cursor),
        };
        let spec = render(&state, &parse(&src));
        let block = table_block(&spec);
        let BlockKind::Table { edit_mode, .. } = &block.kind else {
            unreachable!()
        };
        assert!(!edit_mode);
        // Leading `| `, inter-cell ` | `, trailing ` |` all hidden.
        assert!(block.has_hidden_range(0..2));
        assert!(block.has_hidden_range(3..6));
        assert!(block.has_hidden_range(7..9));
        // The delimiter row is hidden wholesale.
        let delim_start = src.find("| ---").unwrap();
        let delim_end = src.find(" |\n| c1").unwrap() + 2;
        assert!(block.has_hidden_range(delim_start..delim_end));
        // Header cells carry the table-header style.
        assert!(
            block
                .inlines
                .iter()
                .any(|r| r.source_range == (2..3) && r.style.table_header)
        );
    }

    #[test]
    fn edit_mode_dims_chrome_instead_of_hiding() {
        let state = EditorState {
            markdown: CANONICAL.to_string(),
            selection: Selection::Cursor(2),
        };
        let spec = render(&state, &parse(CANONICAL));
        let block = table_block(&spec);
        let BlockKind::Table { edit_mode, .. } = &block.kind else {
            unreachable!()
        };
        assert!(edit_mode);
        assert!(block.hidden_ranges.is_empty());
        assert!(block.has_dimmed_range(0..2));
        assert!(block.has_dimmed_range(3..6));
        // The whole delimiter row dims.
        let delim_start = CANONICAL.find("| ---").unwrap();
        let delim_end = CANONICAL.find(" |\n| c1").unwrap() + 2;
        assert!(block.has_dimmed_range(delim_start..delim_end));
    }

    #[test]
    fn readonly_render_is_display_mode() {
        let state = EditorState {
            markdown: CANONICAL.to_string(),
            selection: Selection::Cursor(2), // even with a caret inside
        };
        let spec = render_readonly(&state, &parse(CANONICAL));
        let block = table_block(&spec);
        let BlockKind::Table { edit_mode, .. } = &block.kind else {
            unreachable!()
        };
        assert!(!edit_mode);
        assert!(block.has_hidden_range(0..2));
    }

    #[test]
    fn cell_inline_styling_composes() {
        let src = "| **bold** | `code` |\n| --- | --- |\n| x | y |\n";
        let state = EditorState {
            markdown: src.to_string(),
            selection: Selection::Cursor(src.len()),
        };
        let spec = render(&state, &parse(src));
        let block = table_block(&spec);
        let bold = src.find("bold").unwrap();
        assert!(
            block
                .inlines
                .iter()
                .any(|r| r.source_range == (bold..bold + 4) && r.style.bold)
        );
        let code = src.find("code").unwrap();
        assert!(
            block
                .inlines
                .iter()
                .any(|r| r.source_range == (code..code + 4) && r.style.code)
        );
    }

    #[test]
    fn nested_table_falls_back_to_raw_source() {
        let src = "> | a | b |\n> | - | - |\n> | c | d |\n";
        let state = EditorState {
            markdown: src.to_string(),
            selection: Selection::Cursor(src.len()),
        };
        let spec = render(&state, &parse(src));
        assert!(
            spec.blocks
                .iter()
                .all(|b| !matches!(b.kind, BlockKind::Table { .. })),
            "nested tables render as raw source, not a grid"
        );
        // …and there is a block covering the table's lines.
        assert!(spec.blocks.iter().any(|b| b.source_range.start <= 2));
    }
}
