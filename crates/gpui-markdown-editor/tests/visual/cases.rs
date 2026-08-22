//! Snapshot cases. Each case constructs a `MarkdownEditor` in a known state
//! and renders it to PNG. Cursor placement is the load-bearing dimension —
//! every construct gets at least: cursor outside, cursor inside, with
//! selection.

use gpui::{AppContext, Entity, px, size};
use gpui_markdown_editor::{
    EditorState, EmbedMap, HighlightLayer, MarkdownEditor, MarkdownEditorState, Selection,
};

use super::harness::Snapshots;

/// Snapshot host: holds the editor state entity and renders the
/// `MarkdownEditor` element, mirroring a real host. The state entity is not
/// `Render`, so snapshots wrap it in this. Its `new`/`with_state` shadow the
/// old entity constructors so the cases read unchanged.
struct EditorHarness {
    state: Entity<MarkdownEditorState>,
}

impl EditorHarness {
    fn new(
        markdown: impl Into<String>,
        window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) -> Self {
        let state = cx.new(|cx| MarkdownEditorState::new(window, cx).default_value(markdown));
        Self { state }
    }

    fn with_state(
        state: EditorState,
        window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) -> Self {
        let state = cx.new(|cx| MarkdownEditorState::with_state(state, window, cx));
        Self { state }
    }
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

/// Read-only host — the surface embeds are compared against. An embed's
/// mapped content goes through `render_readonly`, so a *disabled* editor over
/// the same markdown is the apples-to-apples control (an editable one would
/// flip whatever construct hosts the cursor into edit mode).
struct ReadonlyHarness {
    state: Entity<MarkdownEditorState>,
}

impl gpui::Render for ReadonlyHarness {
    fn render(
        &mut self,
        _: &mut gpui::Window,
        _: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        MarkdownEditor::new(&self.state).disabled(true)
    }
}

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
        cx.new(|cx| EditorHarness::new("", window, cx))
    });

    s.add("plain_paragraph", win, |window, cx| {
        cx.new(|cx| EditorHarness::new("just a body paragraph.", window, cx))
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

    // CJK emphasis — 着重号 dots stand in for the italic no CJK face
    // has. Cursor outside: delimiters gone, dots under each character.
    s.add("cjk_emphasis_cursor_outside", win, |window, cx| {
        editor_with_cursor(window, cx, "前面 *强调的文字* 后面", "后面")
    });

    // CJK emphasis — cursor inside. Entering reveals the delimiters and
    // leaves the content's emphasis exactly as it was.
    s.add("cjk_emphasis_cursor_inside", win, |window, cx| {
        editor_with_cursor(window, cx, "前面 *强调的文字* 后面", "调的文")
    });

    // One emphasized span carrying both renderings: the Latin word
    // shapes italic, the Han characters take dots, and the CJK comma
    // between them takes none.
    s.add("cjk_emphasis_mixed_run", win, |window, cx| {
        editor_with_cursor(window, cx, "混排 *中文，with latin，测试* 收尾", "收尾")
    });

    // Bold and bold+emphasis on CJK: strong stays a bold face (the CJK
    // fallback ships one), strong+emphasis is bold *and* dotted.
    s.add("cjk_strong_and_strong_emphasis", win, |window, cx| {
        editor_with_cursor(window, cx, "**粗体文字** 与 ***粗体强调*** 收尾", "收尾")
    });

    // A tall inline construct grows the row stride; the inline-code chip
    // beside it must still fill only the text row (the editor draws the
    // chip itself — gpui's background pass takes one number for both the
    // fill height and the row step).
    s.add("inline_code_chip_beside_tall_math", win, |window, cx| {
        editor_with_cursor(
            window,
            cx,
            "lead paragraph\n\nsee $\\frac{\\frac{a}{b}}{\\frac{c}{d}}$ and `code` here\n\ntail paragraph",
            "tail",
        )
    });

    // An emphasized CJK run soft-wrapped at a narrow measure: the dot for
    // the character that *opens* a wrap row belongs on that row, not at
    // the previous row's right edge.
    s.add(
        "cjk_emphasis_wrapped",
        size(px(200.), px(240.)),
        |window, cx| {
            editor_with_cursor(
                window,
                cx,
                "*中文测试强调的文字还有更多内容需要换行* 收尾",
                "收尾",
            )
        },
    );

    // An inline-code span crossing soft wraps at word boundaries: each
    // row's chip ends at that row's last glyph, not at the wrap width
    // (`WrappedLine::width()` reports the configured wrap width for any
    // line that wrapped, which leaves blank trailing space unfilled).
    s.add(
        "inline_code_chip_wrapped",
        size(px(240.), px(240.)),
        |window, cx| {
            editor_with_cursor(
                window,
                cx,
                "lead in `alpha beta gamma delta epsilon zeta eta` and the tail after it",
                "tail",
            )
        },
    );

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
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
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
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
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
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
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
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // Fenced code block — cursor outside (fences hidden).
    s.add("code_block_cursor_outside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "Some intro.\n\n```rust\nfn main() {\n    println!(\"hi\");\n}\n```\n\nTrailing prose.".into(),
                // Cursor in trailing prose.
                selection: Selection::Cursor(60),
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // Fenced code block — cursor inside (fences dimmed).
    s.add("code_block_cursor_inside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "```rust\nfn main() {\n    println!(\"hi\");\n}\n```".into(),
                // Inside content.
                selection: Selection::Cursor(20),
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
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
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
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
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // Blockquote — cursor inside (`> ` markers dimmed-visible).
    s.add("blockquote_cursor_inside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "> A short quote.\nfollowing line.".into(),
                // Cursor inside "quote".
                selection: Selection::Cursor(8),
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // Two-deep nested blockquote — borders stack, both markers hide
    // when cursor outside.
    s.add("nested_blockquotes_outside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "Intro.\n\n> > Deep wisdom here.\n\nBody.".into(),
                selection: Selection::Cursor(33),
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
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
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // Blockquote wrapping a heading — the heading's `# ` *and* the
    // blockquote's `> ` both hide together.
    s.add("blockquote_around_heading", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "> # Quoted heading\n\nBody.".into(),
                selection: Selection::Cursor(22),
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
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
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // After Enter inside `> hello` — empty marker line plus a new
    // blockquote line where the cursor sits. Borders span both lines.
    s.add("blockquote_after_enter", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "> hello\n> \n> ".into(),
                selection: Selection::Cursor(13),
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // Same shape at depth 2 — two stacked borders span all three rows.
    s.add("nested_blockquote_after_enter", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "> > deep\n> > \n> > ".into(),
                selection: Selection::Cursor(18),
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // Hard break inside a blockquote: `  \n> ` keeps the second visual
    // line in the same paragraph and inside the blockquote.
    s.add("blockquote_hard_break", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "> hello  \n> ".into(),
                selection: Selection::Cursor(12),
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // Code block inside a blockquote — the code-block bg paints
    // *inside* the blockquote indent, not over the border bar.
    s.add("code_block_inside_blockquote", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "> ```rust\n> let x = 1;\n> ```\n\nBody.".into(),
                selection: Selection::Cursor(31),
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
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
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
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
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
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
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
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
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
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
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // Triple-nested list — three indent strips stack. `containers_left_indent`
    // sums each level's marker width plus the leading `list_indent`.
    s.add("triple_nested_list", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "- a\n  - b\n    - c".into(),
                selection: Selection::Cursor(0),
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
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
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // Display math — cursor outside (rendered typeset LaTeX).
    s.add("display_math_cursor_outside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "Intro paragraph.\n\n$$\n\\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n\n$$\n\nOutro paragraph.".into(),
                selection: Selection::Cursor(0), // Cursor at start of document (outside the math block).
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // Display math — cursor inside (raw LaTeX edit mode).
    s.add("display_math_cursor_inside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "Intro paragraph.\n\n$$\n\\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n\n$$\n\nOutro paragraph.".into(),
                selection: Selection::Cursor(25), // Cursor inside the math block.
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // Display math inside blockquote — cursor outside (rendered typeset LaTeX).
    s.add("display_math_inside_blockquote_cursor_outside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "> $$\n> \\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n\n> $$\n\nBody paragraph.".into(),
                selection: Selection::Cursor(65), // Cursor on "Body paragraph" (outside blockquote and math).
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // Display math inside blockquote — cursor inside (raw LaTeX edit mode).
    s.add("display_math_inside_blockquote_cursor_inside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "> $$\n> \\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n\n> $$\n\nBody paragraph.".into(),
                selection: Selection::Cursor(25), // Cursor inside the math block.
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // Display math inside list — cursor outside (rendered typeset LaTeX).
    s.add("display_math_inside_list_cursor_outside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "- Item one\n  $$\n  \\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n\n  $$\n- Item two".into(),
                selection: Selection::Cursor(0), // Cursor at start (outside the math block).
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // Display math inside list — cursor inside (raw LaTeX edit mode).
    s.add("display_math_inside_list_cursor_inside", win, |window, cx| {
        cx.new(|cx| {
            let state = EditorState {
                markdown: "- Item one\n  $$\n  \\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n\n  $$\n- Item two".into(),
                selection: Selection::Cursor(35), // Cursor inside the math block.
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // ---- Tables ---------------------------------------------------------

    const TABLE_DOC: &str = "Some intro prose above the table.\n\n\
        | Model | Params | Context |\n\
        | :-- | --: | --: |\n\
        | Gemma 4 E2B | 2B | 32k |\n\
        | Gemma 4 4B | 4B | 128k |\n\
        | Kimi K2 | 1T | 256k |\n\n\
        Trailing prose below the table.";

    // Cursor outside — the rendered grid: pipes + delimiter row
    // hidden, header in the table-header weight, alignment colons
    // honored (numbers right-aligned), hairline rules.
    s.add("table_cursor_outside", win, |window, cx| {
        editor_with_cursor(window, cx, TABLE_DOC, "Trailing")
    });

    // Cursor inside — org-style aligned source: pipes + delimiter
    // row dim into view; the pad substitutions keep the columns
    // aligned while editing.
    s.add("table_cursor_inside", win, |window, cx| {
        editor_with_cursor(window, cx, TABLE_DOC, "Gemma 4 4B")
    });

    // Selection overlapping the table (drag from prose into cells).
    s.add("table_selection_across", win, |window, cx| {
        cx.new(|cx| {
            let start = TABLE_DOC.find("intro").unwrap();
            let end = TABLE_DOC.find("4B |").unwrap();
            let state = EditorState {
                markdown: TABLE_DOC.into(),
                selection: Selection::range(start, end),
                ..Default::default()
            };
            EditorHarness::with_state(state, window, cx)
        })
    });

    // Inline styling inside cells (bold / code / strikethrough) plus
    // an escaped pipe, cursor outside.
    s.add("table_styled_cells", win, |window, cx| {
        editor_with_cursor(
            window,
            cx,
            "| Feature | Status |\n\
             | :-- | :-- |\n\
             | **Bold** and `code` | ~~cut~~ kept |\n\
             | a\\|b literal pipe | plain |\n\n\
             after",
            "after",
        )
    });

    // Wide table — its columns shrink toward min-content and the
    // cells wrap internally (the HTML-auto model), so it fits the
    // 720px measure instead of overflowing.
    s.add("table_wrapped_cells", win, |window, cx| {
        editor_with_cursor(
            window,
            cx,
            "| A rather long header cell one | Header two with more words | Third header column | Fourth column header | Fifth and final header |\n\
             | --- | --- | --- | --- | --- |\n\
             | some content here | more cell content | further content | yet more words | the last cell |\n\n\
             after",
            "after",
        )
    });

    // The same wrapping table in edit mode — pipes render in the
    // gutters between the wrapped cell boxes, the delimiter row
    // stretches, and every chrome byte keeps a true caret position.
    s.add("table_wrapped_cells_edit", win, |window, cx| {
        editor_with_cursor(
            window,
            cx,
            "| Topic | Detail |\n\
             | :-- | :-- |\n\
             | Attestation | Every new TLS handshake re-verifies the enclave measurement against the pinned trust root before any request bytes flow |\n\
             | Refunds | Unspent holds return to the wallet through the recovery endpoint |\n",
            "recovery",
        )
    });

    // Degenerate narrow measure: even min-content columns overflow,
    // so the table floors at min and takes the shared horizontal
    // scroll treatment.
    s.add(
        "table_min_content_overflow_scrollbar",
        size(px(240.), px(480.)),
        |window, cx| {
            editor_with_cursor(
                window,
                cx,
                "| Unbreakable-header-atom-one | Another-unbreakable-atom |\n\
                 | --- | --- |\n\
                 | word | word |\n\n\
                 after",
                "after",
            )
        },
    );

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
                    ..Default::default()
                };
                EditorHarness::with_state(state, window, cx)
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
                    ..Default::default()
                };
                EditorHarness::with_state(state, window, cx)
            })
        },
    );

    register_embed_audit(s);
    register_highlight_wash(s);
}

/// The host-supplied highlight wash (`crate::highlight`) over read-only
/// content — the surface the space view paints quoted-passage references on.
/// The wash has no other coverage in this corpus, so these are the pixels
/// that say whether a change to the highlight plugin moved anything.
fn register_highlight_wash(s: &mut Snapshots) {
    let win = size(px(720.), px(320.));

    // A wash across inline styles, spanning delimiters the reader cannot see.
    s.add("highlight_wash_paragraph", win, |window, cx| {
        highlighted_readonly(
            window,
            cx,
            "A paragraph with **bold** and *italic* words, plus a\n\
             [link to somewhere](https://example.com) after them.\n\n\
             A second paragraph that carries no wash at all.",
            &["with **bold** and", "somewhere"],
        )
    });

    // Two ranges that overlap: the wash merges rather than double-darkening.
    s.add("highlight_wash_overlapping", win, |window, cx| {
        highlighted_readonly(
            window,
            cx,
            "One long sentence whose middle is covered by two overlapping ranges.",
            &["sentence whose middle", "middle is covered by"],
        )
    });

    // Inside a fenced code block — the wash follows the delimiter/content
    // mask split rather than painting over the fence rows.
    s.add("highlight_wash_code_block", win, |window, cx| {
        highlighted_readonly(
            window,
            cx,
            "Before the fence.\n\n\
             ```rust\n\
             fn main() {\n    println!(\"hello\");\n}\n\
             ```\n\n\
             After the fence.",
            &["println!"],
        )
    });

    // Two layers over the same words: the upper layer's wash paints on top of
    // the base one instead of merging with it, and each takes its own color.
    s.add("highlight_wash_layered", win, |window, cx| {
        cx.new(|cx| {
            let markdown = "One sentence whose middle carries a base wash, with a \
                            shorter span inside it singled out on the layer above.";
            let base = markdown.find("whose middle carries").expect("fixture");
            let accent = markdown.find("middle").expect("fixture");
            let state = EditorState {
                markdown: markdown.into(),
                selection: Selection::Cursor(0),
                ..Default::default()
            };
            let state = cx.new(|cx| MarkdownEditorState::with_state(state, window, cx));
            state.update(cx, |e, cx| {
                e.set_highlights(vec![(base..base + "whose middle carries".len(), 0)], cx);
                e.set_highlights_in(
                    HighlightLayer::Accent,
                    vec![(accent..accent + "middle".len(), 1)],
                    cx,
                );
            });
            ReadonlyHarness { state }
        })
    });

    // Inside a table cell — cells are ordinary laid-out lines, so the wash
    // reaches them with no table-specific machinery.
    s.add("highlight_wash_table_cell", win, |window, cx| {
        highlighted_readonly(
            window,
            cx,
            "| Feature | Status |\n\
             | :-- | --: |\n\
             | **Bold** cell | `code` |\n\
             | plain | ~~cut~~ |\n",
            &["Status", "plain"],
        )
    });
}

/// A read-only editor over `markdown` with a highlight range per needle.
/// Panics if a needle is missing — keeps the cases honest, like
/// `editor_with_cursor`.
fn highlighted_readonly(
    window: &mut gpui::Window,
    cx: &mut gpui::App,
    markdown: &'static str,
    needles: &'static [&'static str],
) -> Entity<ReadonlyHarness> {
    let entries: Vec<(std::ops::Range<usize>, u64)> = needles
        .iter()
        .enumerate()
        .map(|(i, needle)| {
            let start = markdown
                .find(needle)
                .unwrap_or_else(|| panic!("substring {needle:?} not found in test fixture"));
            (start..start + needle.len(), i as u64)
        })
        .collect();
    let state = EditorState {
        markdown: markdown.into(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    cx.new(|cx| {
        let state = cx.new(|cx| MarkdownEditorState::with_state(state, window, cx));
        state.update(cx, |e, cx| e.set_highlights(entries, cx));
        ReadonlyHarness { state }
    })
}

/// The embed-fidelity audit: every markdown construct the editor supports,
/// rendered twice — once as ordinary top-level content (the *control*) and
/// once as the mapped content of an embed block. The pair is the evidence:
/// an embed is supposed to look like the control, inset in a quote container.
///
/// One family per corpus so a regression is legible in a single image rather
/// than buried in a kitchen sink.
fn register_embed_audit(s: &mut Snapshots) {
    for (name, corpus, h) in EMBED_AUDIT_CORPORA {
        let win = size(px(720.), px(*h));
        s.add(
            format!("embed_audit_{name}_control"),
            win,
            move |window, cx| {
                cx.new(|cx| {
                    let state = EditorState {
                        markdown: (*corpus).into(),
                        selection: Selection::Cursor(0),
                        ..Default::default()
                    };
                    ReadonlyHarness {
                        state: cx.new(|cx| MarkdownEditorState::with_state(state, window, cx)),
                    }
                })
            },
        );
        s.add(
            format!("embed_audit_{name}_embedded"),
            win,
            move |window, cx| {
                cx.new(|cx| {
                    let state = EditorState {
                        markdown: "{{ embed 1 }}".into(),
                        selection: Selection::Cursor(0),
                        embeds: EmbedMap::new([(1u64, (*corpus).to_string())]),
                    };
                    ReadonlyHarness {
                        state: cx.new(|cx| MarkdownEditorState::with_state(state, window, cx)),
                    }
                })
            },
        );
    }
}

/// `(family, markdown, window height)`.
const EMBED_AUDIT_CORPORA: &[(&str, &str, f32)] = &[
    (
        "headings",
        "# Heading one\n\n\
         Body under one.\n\n\
         ## Heading two\n\n\
         ### Heading three\n\n\
         Trailing paragraph.",
        340.,
    ),
    (
        "unordered_list",
        "- first item\n\
         - second item\n\
         - third item with enough words that it soft-wraps at this measure and continues\n",
        220.,
    ),
    (
        "unordered_list_nested",
        "- outer one\n  - inner one\n  - inner two\n    - deepest\n- outer two\n",
        220.,
    ),
    (
        "ordered_list",
        "8. eight\n9. nine\n10. ten\n11. eleven\n",
        200.,
    ),
    (
        "ordered_list_nested",
        "1. outer one\n   1. inner one\n   2. inner two\n2. outer two\n",
        200.,
    ),
    (
        "task_list",
        "- [x] done item\n- [ ] todo item\n- plain sibling\n",
        180.,
    ),
    (
        "loose_list",
        "- first paragraph of item one\n\n  second paragraph of item one\n\n- item two\n",
        220.,
    ),
    (
        "inline_styles",
        "Plain **bold**, *italic*, ***both***, ~~struck~~, `inline code`, and a\n\
         [link to somewhere](https://example.com) in one paragraph.\n\n\
         Escapes and entities: \\*not italic\\* and &copy; and &#x2014; dash.",
        220.,
    ),
    (
        "hard_break",
        "first line\\\nsecond line after a backslash break\n\n\
         third line  \nfourth line after a two-space break\n",
        200.,
    ),
    (
        "code_fence",
        "Before the fence.\n\n\
         ```rust\n\
         fn main() {\n    println!(\"hello\");\n}\n\
         ```\n\n\
         After the fence.",
        280.,
    ),
    (
        "table",
        "| Feature | Status |\n\
         | :-- | --: |\n\
         | **Bold** cell | `code` |\n\
         | plain | ~~cut~~ |\n",
        220.,
    ),
    (
        "blockquote",
        "> quoted line one\n> quoted line two\n>\n> > nested deeper\n\n\
         after the quote",
        240.,
    ),
    (
        "thematic_break",
        "above the rule\n\n---\n\nbelow the rule",
        200.,
    ),
    (
        "math",
        "Inline $x^2 + y^2$ in a sentence.\n\n$$\n\\frac{a}{b}\n$$\n\nafter the math.",
        260.,
    ),
    // Tall inline math on a *non-final* visual row of a soft-wrapped
    // paragraph — the pixel record of `LaidOutLine::row_stride`, in the
    // live editor and the embed alike. The reservation
    // `compute_math_row_extra` returns lands in the per-row stride, so
    // every visual row of the line is that much further apart: the
    // construct's ink stays off the glyphs of the row beneath it and no
    // reserved space piles up after the logical line (the embedded twin's
    // quote bar ends at the text). The uniform extra leading on the rows
    // that carry no math is the cost of gpui striding a whole
    // `WrappedLine` by one line height.
    (
        "wrapped_math",
        "A paragraph whose tall $\\frac{\\frac{\\frac{a}{b}}{\\frac{c}{d}}}{\\frac{e}{f}}$ \
         appears early on, and which then continues with enough further words that the \
         logical line must soft-wrap several times at this measure — so the tall \
         construct sits on the first visual row while ordinary text keeps flowing \
         onto the rows below it.",
        260.,
    ),
    (
        "list_with_code",
        "- item with a fence:\n\n  ```\n  let x = 1;\n  ```\n\n- plain item\n",
        260.,
    ),
    // 着重号 emphasis dots are painted in their own pass, so the embed —
    // which shares only the *shaping* path — has to re-emit them. This
    // pair is what keeps the two surfaces from drifting.
    (
        "cjk_emphasis",
        "前面 *强调的文字* 后面\n\n\
         混排 *中文，with latin，测试* 收尾\n\n\
         **粗体文字** 与 ***粗体强调*** 收尾\n\n\
         | 表头 | 说明 |\n\
         | --- | --- |\n\
         | *中文* | `code` |\n",
        300.,
    ),
    (
        "quote_with_list",
        "> - quoted bullet one\n> - quoted bullet two\n>\n> tail paragraph\n",
        220.,
    ),
];

/// Build an editor whose cursor is placed inside `needle` (3 chars in, by
/// default). Panics if `needle` isn't found — keeps the cases honest.
fn editor_with_cursor(
    window: &mut gpui::Window,
    cx: &mut gpui::App,
    markdown: &'static str,
    needle: &'static str,
) -> Entity<EditorHarness> {
    let cursor = markdown
        .find(needle)
        .map(|i| i + 3.min(needle.len()))
        .unwrap_or_else(|| panic!("substring {needle:?} not found in test fixture"));
    let state = EditorState {
        markdown: markdown.into(),
        selection: Selection::Cursor(cursor),
        ..Default::default()
    };
    cx.new(|cx| EditorHarness::with_state(state, window, cx))
}
