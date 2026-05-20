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

    // Empty paragraph injection: 6 newlines between content should
    // render as paragraph break + 2 visible empty rows in the pairs
    // model (each Enter inserts `\n\n`, so 3 Enters mid-content gives 6
    // `\n`s).
    s.add("empty_paragraphs_between_blocks", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "paragraph 1\n\n\n\n\n\nparagraph 2".into(),
                selection: Selection::Cursor(0),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Same source, cursor on one of the empty rows — confirms the cursor
    // has somewhere visible to land.
    s.add("empty_paragraphs_cursor_in_empty_row", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                // 6 `\n`s = 1 paragraph break + 2 empty paragraphs.
                // Byte 14 is in the middle empty paragraph (range 14..16).
                markdown: "paragraph 1\n\n\n\n\n\nparagraph 2".into(),
                selection: Selection::Cursor(14),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Trailing empty paragraph: pressing Enter at the end of "paragraph 1"
    // produces `paragraph 1\n\n` (pairs model, one Enter = `\n\n`) with
    // the cursor at byte 13. Render shows one trailing empty row
    // anchoring the cursor.
    s.add("trailing_empty_after_one_enter", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "paragraph 1\n\n".into(),
                selection: Selection::Cursor(13),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Fenced code block — cursor outside (fences hidden).
    s.add("code_block_cursor_outside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "Some intro.\n\n```rust\nfn main() {\n    println!(\"hi\");\n}\n```\n\nTrailing prose.".into(),
                // Cursor in trailing prose.
                selection: Selection::Cursor(60),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Fenced code block — cursor inside (fences dimmed).
    s.add("code_block_cursor_inside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "```rust\nfn main() {\n    println!(\"hi\");\n}\n```".into(),
                // Inside content.
                selection: Selection::Cursor(20),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Fenced code block — long line that overflows the visible width
    // and triggers the horizontal scrollbar.
    s.add("code_block_overflow_scrollbar", win, |window, cx| {
        cx.new(|cx| {
            let long = "let x = some_extremely_long_variable_name_that_will_definitely_exceed_the_block_width_at_720_px();";
            let md = format!("```rust\n{long}\n```");
            let state = EditorState {
                markdown: md,
                selection: Selection::Cursor(0),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Blockquote — cursor outside (`> ` markers hidden, content
    // indented behind a left border bar).
    s.add("blockquote_cursor_outside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "Some intro.\n\n> A short quote.\n\nTrailing prose.".into(),
                // Cursor in trailing prose.
                selection: Selection::Cursor(34),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Blockquote — cursor inside (`> ` markers dimmed-visible).
    s.add("blockquote_cursor_inside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "> A short quote.\nfollowing line.".into(),
                // Cursor inside "quote".
                selection: Selection::Cursor(8),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Two-deep nested blockquote — borders stack, both markers hide
    // when cursor outside.
    s.add("nested_blockquotes_outside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "Intro.\n\n> > Deep wisdom here.\n\nBody.".into(),
                selection: Selection::Cursor(33),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Nested-bq sandwich: an outer-only paragraph, a nested
    // paragraph, then another outer-only paragraph. The outer bar
    // (level 0) should remain continuous through *both* boundaries
    // because the outer level is shared on each side; only the
    // inner bar pulls back into the breathing room. Sibling
    // paragraphs above and below the whole construct exercise the
    // paragraph ↔ blockquote boundary too.
    s.add("nested_blockquote_sandwich", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: concat!(
                    "Lead-in paragraph.\n",
                    "\n",
                    "> Outer only.\n",
                    "\n",
                    "> > Nested.\n",
                    "\n",
                    "> Outer only again.\n",
                    "\n",
                    "Trailing prose.",
                )
                .into(),
                selection: Selection::Cursor(0),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Blockquote wrapping a heading — the heading's `# ` *and* the
    // blockquote's `> ` both hide together.
    s.add("blockquote_around_heading", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "> # Quoted heading\n\nBody.".into(),
                selection: Selection::Cursor(22),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Lone trailing `> ` after a regular paragraph — the user just
    // typed `> ` after pressing Enter twice. The block parses as a
    // blockquote and must render as one immediately, with the bar
    // and overlay marker visible at the cursor row.
    s.add("blockquote_lone_trailing_marker", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "paragraph\n\n> ".into(),
                selection: Selection::Cursor(13),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // After Enter inside `> hello` — empty marker line plus a new
    // blockquote line where the cursor sits. Borders span both lines.
    s.add("blockquote_after_enter", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "> hello\n> \n> ".into(),
                selection: Selection::Cursor(13),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Same shape at depth 2 — two stacked borders span all three rows.
    s.add("nested_blockquote_after_enter", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "> > deep\n> > \n> > ".into(),
                selection: Selection::Cursor(18),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Hard break inside a blockquote: `  \n> ` keeps the second visual
    // line in the same paragraph and inside the blockquote.
    s.add("blockquote_hard_break", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "> hello  \n> ".into(),
                selection: Selection::Cursor(12),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Code block inside a blockquote — the code-block bg paints
    // *inside* the blockquote indent, not over the border bar.
    s.add("code_block_inside_blockquote", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "> ```rust\n> let x = 1;\n> ```\n\nBody.".into(),
                selection: Selection::Cursor(31),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Trailing hard break: Shift+Enter at the end produces
    // `"paragraph 1  \n"`. Visually similar to the regular trailing
    // Enter but the empty trailing row sits *inside* the same paragraph
    // (no paragraph_gap between the content row and the empty row),
    // matching CommonMark hard-break semantics.
    s.add("trailing_hard_break", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "paragraph 1  \n".into(),
                selection: Selection::Cursor(14),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // ---- Lists ----------------------------------------------------------

    // Unordered list — bullet glyphs render in the indent strip,
    // content shapes from a uniform left edge.
    s.add("unordered_list_cursor_outside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "- foo\n- bar\n- baz\n\nbody".into(),
                // Cursor outside the list.
                selection: Selection::Cursor(20),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Same source, cursor on one of the items — shows the raw `-`
    // bullet char (vs the `•` shown when outside) so the user has
    // visual feedback they're inside the marker scope.
    s.add("unordered_list_cursor_inside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "- foo\n- bar\n- baz".into(),
                selection: Selection::Cursor(8),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Ordered list spanning a digit-count boundary — items 1-9
    // shape as 2-char markers (`1.`-`9.`) and items 10-11 as
    // 3-char markers (`10.`/`11.`). Every item's content edge
    // aligns at the column of the *widest* marker, so `1. one`
    // shares its content X with `11. eleven`.
    s.add("ordered_list_mixed_width_markers", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: concat!(
                    "1. one\n",
                    "2. two\n",
                    "3. three\n",
                    "4. four\n",
                    "5. five\n",
                    "6. six\n",
                    "7. seven\n",
                    "8. eight\n",
                    "9. nine\n",
                    "10. ten\n",
                    "11. eleven\n",
                )
                .into(),
                // Cursor at end of doc (outside any item, so all
                // markers paint as their digit form).
                selection: Selection::Cursor(0),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Nested list — inner items pick up additional indent from
    // their `Container::ListItem` chain entry. Outer markers sit
    // in their own strip; inner markers in theirs.
    s.add("nested_list", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "- outer\n  - inner one\n  - inner two\n- outer two".into(),
                selection: Selection::Cursor(0),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Triple-nested list — three indent strips stack. `containers_left_indent`
    // sums each level's marker width plus the leading `list_indent`.
    s.add("triple_nested_list", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "- a\n  - b\n    - c".into(),
                selection: Selection::Cursor(0),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Multi-paragraph item — the second paragraph's leading
    // continuation indent is hidden so its content shapes from the
    // same column as the first paragraph.
    s.add("multi_paragraph_list_item", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "- first paragraph\n\n  second paragraph at the same column\n".into(),
                selection: Selection::Cursor(0),
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Display math — cursor outside (rendered typeset LaTeX).
    s.add("display_math_cursor_outside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "Intro paragraph.\n\n$$\n\\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n\n$$\n\nOutro paragraph.".into(),
                selection: Selection::Cursor(0), // Cursor at start of document (outside the math block).
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Display math — cursor inside (raw LaTeX edit mode).
    s.add("display_math_cursor_inside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "Intro paragraph.\n\n$$\n\\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n\n$$\n\nOutro paragraph.".into(),
                selection: Selection::Cursor(25), // Cursor inside the math block.
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Display math inside blockquote — cursor outside (rendered typeset LaTeX).
    s.add("display_math_inside_blockquote_cursor_outside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "> $$\n> \\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n\n> $$\n\nBody paragraph.".into(),
                selection: Selection::Cursor(65), // Cursor on "Body paragraph" (outside blockquote and math).
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Display math inside blockquote — cursor inside (raw LaTeX edit mode).
    s.add("display_math_inside_blockquote_cursor_inside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "> $$\n> \\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n\n> $$\n\nBody paragraph.".into(),
                selection: Selection::Cursor(25), // Cursor inside the math block.
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Display math inside list — cursor outside (rendered typeset LaTeX).
    s.add("display_math_inside_list_cursor_outside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "- Item one\n  $$\n  \\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n\n  $$\n- Item two".into(),
                selection: Selection::Cursor(0), // Cursor at start (outside the math block).
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Display math inside list — cursor inside (raw LaTeX edit mode).
    s.add("display_math_inside_list_cursor_inside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "- Item one\n  $$\n  \\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n\n  $$\n- Item two".into(),
                selection: Selection::Cursor(35), // Cursor inside the math block.
            };
            MarkdownEditor::with_state(state, window, cx)
        })
    });

    // Ordered list with an empty intermediate item that hosts a
    // nested sublist (`2. ` followed by `   1. Two, One`). The empty
    // marker row should sit at the outer LI's indent — same column
    // as `1. One` above — not jump in to the nested list's deeper
    // indent. Cursor outside any item.
    s.add(
        "empty_intermediate_list_item_cursor_outside",
        win,
        |window, cx| {
            cx.new(|cx| {
                let state = EditorState {
                    markdown: "1. One\n2. \n   1. Two, One".into(),
                    selection: Selection::Cursor(0),
                };
                MarkdownEditor::with_state(state, window, cx)
            })
        },
    );

    // Same fixture, cursor on the empty `2. ` row. Verifies that the
    // caret sits at the outer LI's content edge (not the nested
    // list's deeper indent).
    s.add(
        "empty_intermediate_list_item_cursor_on_empty_row",
        win,
        |window, cx| {
            cx.new(|cx| {
                let state = EditorState {
                    markdown: "1. One\n2. \n   1. Two, One".into(),
                    selection: Selection::Cursor(10), // `\n` ending the `2. ` row
                };
                MarkdownEditor::with_state(state, window, cx)
            })
        },
    );
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
