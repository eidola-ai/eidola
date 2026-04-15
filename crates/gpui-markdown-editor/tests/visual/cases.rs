//! Snapshot cases. Each case constructs a `MarkdownEditor` in a known state
//! and renders it to PNG. Cursor placement is the load-bearing dimension —
//! every construct gets at least: cursor outside, cursor inside, with
//! selection.

use gpui::{AppContext, Entity, px, size};
use gpui_markdown_editor::{EditorState, MarkdownEditor, Selection};

use super::harness::Snapshots;

const KITCHEN_SINK: &str = "\
# Markdown editor

This is **bold** and *italic* and ~~strikethrough~~ in one line. The
delimiters should hide here because the cursor is below.

## A second-level heading

Mix and match: ***bold italic*** with ~~strike~~ inside.

### A third-level heading

Plain body paragraph at the bottom of the document.
";

pub fn register(s: &mut Snapshots) {
    let win = size(px(720.), px(480.));

    s.add("empty_document", win, |window, cx| {
        cx.new(|cx| MarkdownEditor::new("", window, cx))
    });

    s.add("plain_paragraph", win, |window, cx| {
        cx.new(|cx| MarkdownEditor::new("just a body paragraph.", window, cx))
    });

    // Heading: cursor outside (delimiters hidden).
    s.add("heading_cursor_outside", win, |window, cx| {
        editor_with_cursor(window, cx, "# Hello\n\nbody", "body")
    });

    // Heading: cursor inside (delimiter dimmed).
    s.add("heading_cursor_inside", win, |window, cx| {
        editor_with_cursor(window, cx, "# Hello", "ello")
    });

    // Bold: cursor outside.
    s.add("bold_cursor_outside", win, |window, cx| {
        editor_with_cursor(window, cx, "before **bold** after", "after")
    });

    // Bold: cursor inside.
    s.add("bold_cursor_inside", win, |window, cx| {
        editor_with_cursor(window, cx, "before **bold** after", "old")
    });

    // Italic outside.
    s.add("italic_cursor_outside", win, |window, cx| {
        editor_with_cursor(window, cx, "leading *italic* trailing", "trailing")
    });

    // Italic inside.
    s.add("italic_cursor_inside", win, |window, cx| {
        editor_with_cursor(window, cx, "leading *italic* trailing", "talic")
    });

    // Strikethrough outside.
    s.add("strike_cursor_outside", win, |window, cx| {
        editor_with_cursor(window, cx, "keep ~~drop~~ keep", "keep")
    });

    // Strikethrough inside.
    s.add("strike_cursor_inside", win, |window, cx| {
        editor_with_cursor(window, cx, "keep ~~drop~~ keep", "rop")
    });

    // Combined construct test — the catch-all for interaction bugs.
    s.add(
        "kitchen_sink_cursor_at_top",
        size(px(720.), px(640.)),
        |window, cx| editor_with_cursor(window, cx, KITCHEN_SINK, "Markdown"),
    );

    s.add(
        "kitchen_sink_cursor_in_third_heading",
        size(px(720.), px(640.)),
        |window, cx| editor_with_cursor(window, cx, KITCHEN_SINK, "third-level"),
    );

    s.add(
        "kitchen_sink_cursor_in_bold_italic",
        size(px(720.), px(640.)),
        |window, cx| editor_with_cursor(window, cx, KITCHEN_SINK, "bold italic"),
    );

    // Selection overlapping a construct — delimiters should dim.
    s.add("selection_over_bold", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "before **bold** after".into(),
                selection: Selection::range(0, 21),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });
}

/// Build an editor whose cursor is placed inside `needle` (3 chars in, by
/// default). Panics if `needle` isn't found — keeps the cases honest.
fn editor_with_cursor(
    window: &mut gpui::Window,
    cx: &mut gpui::App,
    markdown: &'static str,
    needle: &'static str,
) -> Entity<MarkdownEditor> {
    let cursor = markdown
        .find(needle)
        .map(|i| i + 3.min(needle.len()))
        .unwrap_or_else(|| panic!("substring {needle:?} not found in test fixture"));
    let state = EditorState {
        markdown: markdown.into(),
        selection: Selection::Cursor(cursor),
    };
    cx.new(|cx| MarkdownEditor::with_state(state, window, cx))
}
