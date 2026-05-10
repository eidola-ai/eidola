//! Long-form *session* harness — drives a single editor instance through a
//! scripted keystroke sequence and dumps a named "keyframe" at every
//! interesting moment along the way. The output is artifacts on disk, not
//! pass/fail assertions, so the test still runs in CI to keep the script
//! type-checked but its real value is interactive review:
//!
//! ```text
//! cargo test -p gpui-markdown-editor --test session
//! open target/session-artifacts/code-review-response/transcript.md
//! ```
//!
//! Why a session test in addition to the per-construct behavior gate
//! and the visual snapshots?
//!
//! - The behavior tests catch regressions in known-bad inputs but don't
//!   surface *novel* surprises. Real users compose a document
//!   keystroke-by-keystroke; each intermediate state is its own
//!   correctness target.
//! - The visual snapshots check pixel-fidelity but only at hand-picked
//!   moments. Long sessions through deep nesting reach states the
//!   visual cases never enumerate.
//!
//! The artifacts are intentionally textual (not PNG) so they're cheap,
//! deterministic, diffable, and reviewable by either a human or a
//! sub-agent. The harness sketches the visible structure of each
//! keyframe (block kinds, container chain, hidden ranges, marker
//! overlays) plus the markdown source with the cursor rendered as `|`.
//! Visual snapshots can be added on top of this same script later — the
//! script lives in `code_review_response_session()` and is replayable.

use std::path::PathBuf;

use gpui::{AnyWindowHandle, AppContext, Entity, TestAppContext, WindowOptions};
use gpui_component::Root;
use gpui_markdown_editor::editor::{
    Backspace, Delete, Down, End, Enter, Home, Left, Right, ShiftEnter, ShiftLeft, ShiftRight,
    ShiftTab, Tab,
};
use gpui_markdown_editor::{
    BlockKind, Container, EditorEvent, EditorState, MarkdownEditor, RenderSpec, Selection,
};

// ---------------------------------------------------------------------------
// Session driver
// ---------------------------------------------------------------------------

struct Session {
    cx: *mut TestAppContext,
    handle: AnyWindowHandle,
    editor: Entity<MarkdownEditor>,
    out_dir: PathBuf,
    transcript: String,
    step_count: usize,
    last_action: String,
}

impl Session {
    fn new(name: &str, cx: &mut TestAppContext) -> Self {
        let editor_state = EditorState {
            markdown: String::new(),
            selection: Selection::Cursor(0),
        };

        let (handle, editor) = cx.update(|cx| {
            gpui_component::init(cx);
            let mut inner: Option<Entity<MarkdownEditor>> = None;
            let window = cx
                .open_window(WindowOptions::default(), |window, cx| {
                    let editor = cx.new(|cx| MarkdownEditor::with_state(editor_state, window, cx));
                    inner = Some(editor.clone());
                    cx.new(|cx| Root::new(editor, window, cx))
                })
                .expect("open window");
            (window.into(), inner.expect("editor built"))
        });

        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let out_dir = manifest
            .parent()
            .and_then(|p| p.parent())
            .unwrap_or(&manifest)
            .join("target")
            .join("session-artifacts")
            .join(name);
        let _ = std::fs::remove_dir_all(&out_dir);
        std::fs::create_dir_all(&out_dir).expect("create session artifacts dir");

        Self {
            cx: cx as *mut _,
            handle,
            editor,
            out_dir,
            transcript: String::new(),
            step_count: 0,
            last_action: "(initial state)".to_string(),
        }
    }

    fn cx(&mut self) -> &mut TestAppContext {
        // Safety: the harness only runs single-threaded under libtest's
        // worker thread; the borrow is short-lived per method call.
        unsafe { &mut *self.cx }
    }

    fn dispatch_action<A: gpui::Action>(&mut self, action: A, label: &str) {
        let handle = self.handle;
        let editor = self.editor.clone();
        let cx = self.cx();
        let focus = editor.read_with(cx, |e, _| e.focus_handle.clone());
        cx.update_window(handle, |_, window, cx| {
            focus.dispatch_action(&action, window, cx);
        })
        .unwrap();
        cx.run_until_parked();
        self.last_action = label.to_string();
    }

    fn key(&mut self, action: impl gpui::Action, label: &str) -> &mut Self {
        self.dispatch_action(action, label);
        self
    }

    fn type_(&mut self, text: &str) -> &mut Self {
        let owned = text.to_string();
        let handle = self.handle;
        let editor = self.editor.clone();
        let cx = self.cx();
        cx.update_window(handle, |_, _, cx| {
            editor.update(cx, |e, cx| {
                let next = std::mem::take(&mut e.state);
                e.state = gpui_markdown_editor::update::update(
                    next,
                    EditorEvent::InsertText(owned.clone()),
                );
                cx.notify();
            });
        })
        .unwrap();
        cx.run_until_parked();
        self.last_action = format!("type {text:?}");
        self
    }

    /// Place cursor at the end of the first occurrence of `needle`. Panics if
    /// not found — keeps the script honest, since cursor placement is the
    /// load-bearing dimension for what's about to happen.
    fn cursor_after(&mut self, needle: &str) -> &mut Self {
        let editor = self.editor.clone();
        let pos = editor.read_with(self.cx(), |e, _| {
            e.state
                .markdown
                .find(needle)
                .map(|i| i + needle.len())
                .unwrap_or_else(|| panic!("cursor_after: needle {needle:?} not found"))
        });
        self.set_cursor(pos, &format!("cursor_after({needle:?})"));
        self
    }

    fn cursor_before(&mut self, needle: &str) -> &mut Self {
        let editor = self.editor.clone();
        let pos = editor.read_with(self.cx(), |e, _| {
            e.state
                .markdown
                .find(needle)
                .unwrap_or_else(|| panic!("cursor_before: needle {needle:?} not found"))
        });
        self.set_cursor(pos, &format!("cursor_before({needle:?})"));
        self
    }

    fn select_range(&mut self, anchor: usize, head: usize, label: &str) -> &mut Self {
        let handle = self.handle;
        let editor = self.editor.clone();
        let cx = self.cx();
        cx.update_window(handle, |_, _, cx| {
            editor.update(cx, |e, cx| {
                let next = std::mem::take(&mut e.state);
                e.state = gpui_markdown_editor::update::update(
                    next,
                    EditorEvent::SetSelection(Selection::range(anchor, head)),
                );
                cx.notify();
            });
        })
        .unwrap();
        cx.run_until_parked();
        self.last_action = format!("select_range({anchor}..{head}) — {label}");
        self
    }

    /// Select from before the first occurrence of `start_needle` to after the
    /// first occurrence of `end_needle` (search starts after `start_needle`).
    fn select_span(&mut self, start_needle: &str, end_needle: &str) -> &mut Self {
        let editor = self.editor.clone();
        let (anchor, head) = editor.read_with(self.cx(), |e, _| {
            let md = &e.state.markdown;
            let a = md
                .find(start_needle)
                .unwrap_or_else(|| panic!("select_span: start_needle {start_needle:?} not found"));
            let after_start = a + start_needle.len();
            let h = md[after_start..]
                .find(end_needle)
                .map(|i| after_start + i + end_needle.len())
                .unwrap_or_else(|| {
                    panic!("select_span: end_needle {end_needle:?} not found after start")
                });
            (a, h)
        });
        self.select_range(
            anchor,
            head,
            &format!("from {start_needle:?} through {end_needle:?}"),
        )
    }

    fn set_cursor(&mut self, offset: usize, label: &str) -> &mut Self {
        let handle = self.handle;
        let editor = self.editor.clone();
        let cx = self.cx();
        cx.update_window(handle, |_, _, cx| {
            editor.update(cx, |e, cx| {
                let next = std::mem::take(&mut e.state);
                e.state = gpui_markdown_editor::update::update(
                    next,
                    EditorEvent::SetSelection(Selection::Cursor(offset)),
                );
                cx.notify();
            });
        })
        .unwrap();
        cx.run_until_parked();
        self.last_action = format!("set_cursor({offset}) — {label}");
        self
    }

    /// Dump a named keyframe to `<out>/NN-name.md` and append a transcript
    /// section. `note` describes what the user was trying to do at this
    /// moment — the human-language intent for this checkpoint.
    fn keyframe(&mut self, name: &str, note: &str) -> &mut Self {
        self.step_count += 1;
        let n = self.step_count;
        let editor = self.editor.clone();
        let (markdown, cursor, anchor, spec) = editor.read_with(self.cx(), |e, _| {
            (
                e.state.markdown.clone(),
                e.cursor_offset(),
                e.state.selection.anchor(),
                e.render_spec(),
            )
        });

        let chain = gpui_markdown_editor::analysis::enclosing_containers_at(&markdown, cursor);

        let body = format_keyframe(KeyframeArgs {
            n,
            name,
            note,
            last_action: &self.last_action,
            markdown: &markdown,
            cursor,
            anchor,
            spec: &spec,
            chain: &chain,
        });

        let path = self.out_dir.join(format!("{n:02}-{}.md", slugify(name)));
        std::fs::write(&path, &body).expect("write keyframe");

        // Build the transcript by appending each frame.
        if self.transcript.is_empty() {
            self.transcript.push_str("# Session transcript\n\n");
            self.transcript.push_str(
                "Each section is one keyframe — a checkpoint between keystrokes \
                where the user would pause and expect a particular result. The \
                `note` is what the user was trying to do; the rendered shape \
                shows what they got.\n\n",
            );
        }
        self.transcript.push_str(&body);
        self.transcript.push_str("\n---\n\n");

        self
    }

    fn finish(self) {
        let path = self.out_dir.join("transcript.md");
        std::fs::write(&path, &self.transcript).expect("write transcript");
        println!("\n  session artifacts: {}", path.display());
    }
}

fn slugify(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

/// The cursor visualization. Inserts `|` at `cursor` (and `‖` at `anchor`
/// if the selection is a range) so the reader sees both edges of the
/// selection in the source text. Whitespace is annotated for clarity:
/// trailing spaces at line ends become `·` so hard breaks vs. plain
/// soft-wraps are visible.
fn render_with_cursor(markdown: &str, cursor: usize, anchor: usize) -> String {
    let mut out = String::with_capacity(markdown.len() + 4);
    let mut i = 0;
    for ch in markdown.chars() {
        let next = i + ch.len_utf8();
        if i == cursor {
            out.push('|');
        } else if i == anchor {
            out.push('\u{2016}'); // ‖ for the non-head end of a range
        }
        match ch {
            '\n' => {
                // mark trailing whitespace before the newline so hard-break
                // markers are visible.
                if out.ends_with(' ') {
                    let trimmed = out.trim_end_matches(' ').len();
                    let trailing = out.len() - trimmed;
                    if trailing > 0 {
                        out.truncate(trimmed);
                        for _ in 0..trailing {
                            out.push('·');
                        }
                    }
                }
                out.push('\n');
            }
            _ => out.push(ch),
        }
        i = next;
    }
    let n = markdown.len();
    if cursor == n && anchor == n {
        out.push('|');
    } else {
        if cursor == n {
            out.push('|');
        }
        if anchor == n && anchor != cursor {
            out.push('\u{2016}');
        }
    }
    out
}

fn fmt_chain(chain: &[gpui_markdown_editor::analysis::EnclosingContainer]) -> String {
    use gpui_markdown_editor::analysis::EnclosingContainer;
    if chain.is_empty() {
        return "(top level)".to_string();
    }
    chain
        .iter()
        .map(|c| match c {
            EnclosingContainer::BlockQuote { .. } => "BQ".to_string(),
            EnclosingContainer::ListItem(ctx) => {
                let kind = if ctx.is_ordered() { "ord" } else { "unord" };
                format!("LI({kind}, w={})", ctx.marker_width())
            }
        })
        .collect::<Vec<_>>()
        .join(" → ")
}

fn fmt_block_chain(chain: &[Container]) -> String {
    if chain.is_empty() {
        return "(top level)".to_string();
    }
    chain
        .iter()
        .map(|c| match c {
            Container::BlockQuote { cursor_inside } => {
                if *cursor_inside { "BQ*" } else { "BQ" }.to_string()
            }
            Container::ListItem {
                cursor_inside,
                kind,
                ..
            } => {
                let k = match kind {
                    gpui_markdown_editor::ListItemKind::Ordered { .. } => "ord",
                    gpui_markdown_editor::ListItemKind::Unordered(_, Some(true)) => "task✓",
                    gpui_markdown_editor::ListItemKind::Unordered(_, Some(false)) => "task☐",
                    gpui_markdown_editor::ListItemKind::Unordered(_, None) => "unord",
                };
                if *cursor_inside {
                    format!("LI*({k})")
                } else {
                    format!("LI({k})")
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" → ")
}

fn fmt_block_kind(kind: &BlockKind) -> String {
    match kind {
        BlockKind::Paragraph => "Paragraph".to_string(),
        BlockKind::Heading { level } => format!("Heading(h{level})"),
        BlockKind::CodeBlock { lang } => match lang {
            Some(l) if !l.is_empty() => format!("CodeBlock({l})"),
            _ => "CodeBlock".to_string(),
        },
        BlockKind::ThematicBreak => "ThematicBreak".to_string(),
        BlockKind::DisplayMath { edit_mode, .. } => {
            if *edit_mode {
                "DisplayMath(edit)".to_string()
            } else {
                "DisplayMath".to_string()
            }
        }
    }
}

struct KeyframeArgs<'a> {
    n: usize,
    name: &'a str,
    note: &'a str,
    last_action: &'a str,
    markdown: &'a str,
    cursor: usize,
    anchor: usize,
    spec: &'a RenderSpec,
    chain: &'a [gpui_markdown_editor::analysis::EnclosingContainer],
}

fn format_keyframe(args: KeyframeArgs<'_>) -> String {
    let KeyframeArgs {
        n,
        name,
        note,
        last_action,
        markdown,
        cursor,
        anchor,
        spec,
        chain,
    } = args;
    let mut out = String::new();
    out.push_str(&format!("## Keyframe {n:02} — {name}\n\n"));
    out.push_str(&format!("**Intent:** {note}\n\n"));
    out.push_str(&format!("**Last action:** `{last_action}`\n\n"));
    let anchor_note = if anchor != cursor {
        format!(" (selection anchor at {anchor})")
    } else {
        String::new()
    };
    let chain_str = fmt_chain(chain);
    out.push_str(&format!(
        "**Cursor:** byte {cursor}{anchor_note} • chain: {chain_str}\n\n",
    ));

    out.push_str("### Source (with cursor)\n\n");
    out.push_str("```\n");
    out.push_str(&render_with_cursor(markdown, cursor, anchor));
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("```\n\n");

    out.push_str("### Render shape\n\n");
    if spec.blocks.is_empty() {
        out.push_str("_no blocks_\n\n");
    } else {
        for (i, b) in spec.blocks.iter().enumerate() {
            out.push_str(&format!(
                "- block[{i}] **{}** range={}..{}",
                fmt_block_kind(&b.kind),
                b.source_range.start,
                b.source_range.end,
            ));
            out.push_str(&format!(" • chain: {}", fmt_block_chain(&b.containers)));
            if !b.hidden_ranges.is_empty() {
                out.push_str(&format!(" • hidden: {} range(s)", b.hidden_ranges.len()));
            }
            if !b.marker_overlays.is_empty() {
                out.push_str(&format!(" • marker_overlays: {}", b.marker_overlays.len()));
            }
            if !b.substitutions.is_empty() {
                out.push_str(&format!(" • substitutions: {}", b.substitutions.len()));
            }
            out.push('\n');
        }
        out.push('\n');
    }

    out
}

// ---------------------------------------------------------------------------
// The session — a code review response the author writes from scratch.
// ---------------------------------------------------------------------------

#[gpui::test]
fn code_review_response_session(cx: &mut TestAppContext) {
    let mut s = Session::new("code-review-response", cx);

    s.keyframe("00-blank", "Editor opens; nothing typed yet.");

    // Paragraph 1: an opener thanking the reviewers.
    s.type_("Thanks both for the careful review.")
        .keyframe("01-opener", "First paragraph thanking reviewers.");

    // Move to a new paragraph for the response list.
    s.key(Enter, "Enter").keyframe(
        "02-after-opener-enter",
        "Trailing empty row appears for the next paragraph.",
    );

    // Start the response list. The author plans 1. and 2.
    s.type_("1. ")
        .keyframe("03-start-numbered-list", "Author began an ordered list.");

    s.type_("On the migration safety concern:").keyframe(
        "04-first-item-content",
        "First item written; no children yet.",
    );

    // Add the quoted comment as a blockquote child of this item.
    // Standard usage: Enter, ShiftEnter (to stay inside the item but
    // start a new paragraph), then `> ` to open a BQ inside the item.
    s.key(Enter, "Enter").keyframe(
        "05-after-first-item-enter",
        "Enter at end of item should produce a fresh second item — author about \
         to undo and use Shift+Enter twice instead so the next thing belongs to *this* item.",
    );

    // Undo by Backspace until the trailing item-2 marker is gone.
    s.key(Backspace, "Backspace")
        .key(Backspace, "Backspace")
        .key(Backspace, "Backspace")
        .keyframe(
            "06-undo-second-marker",
            "Cleaned up the unwanted '2. ' marker — back to one item.",
        );

    // Two Shift+Enters → paragraph break inside the item (per AGENTS.md).
    s.key(ShiftEnter, "ShiftEnter")
        .key(ShiftEnter, "ShiftEnter")
        .keyframe(
            "07-paragraph-break-inside-item",
            "Two Shift+Enters should produce a paragraph break inside item 1, \
             with the indent matching the item's continuation column.",
        );

    // Author types the blockquote of the reviewer's comment.
    s.type_("> ").keyframe(
        "08-bq-marker-typed",
        "Just typed the BQ marker inside item 1.",
    );

    s.type_("Are we sure the migration is safe under concurrent writes?")
        .keyframe(
            "09-bq-content",
            "Quoted comment in place inside item 1's blockquote child.",
        );

    // Now add the response after the blockquote — another paragraph in
    // the same item.
    s.key(ShiftEnter, "ShiftEnter")
        .key(ShiftEnter, "ShiftEnter")
        .keyframe(
            "10-paragraph-after-bq",
            "Trying to leave the blockquote and continue typing inside item 1 — \
             expecting a paragraph break inside the item, *not* inside the BQ.",
        );

    s.type_("Yes — the backfill uses a default and the schema change is online.")
        .keyframe(
            "11-response-text",
            "Reply paragraph in place after the quoted comment.",
        );

    // Add a fenced code block showing the migration. Code block opens
    // with ` ``` ` plus info string.
    s.key(ShiftEnter, "ShiftEnter")
        .key(ShiftEnter, "ShiftEnter")
        .type_("```sql")
        .keyframe(
            "12-code-fence-opener",
            "Opening a fenced code block inside item 1 — info string is `sql`.",
        );

    s.key(Enter, "Enter")
        .type_("ALTER TABLE users\n  ADD COLUMN tier text NOT NULL DEFAULT 'free';")
        .keyframe(
            "13-code-content",
            "Two-line code body. Inside code, Enter inserts a single `\\n`, not a paragraph break.",
        );

    s.key(Enter, "Enter").type_("```").keyframe(
        "14-code-closer",
        "Closer fence typed; code block now bracketed.",
    );

    // Move out of the code block to add item 2. Enter at the end of the
    // code closer should leave the code block, but we still want to
    // stay inside item 1 OR start item 2 — depends on Enter semantics.
    // The author would expect: Enter twice at the closer end → start
    // item 2 of the outer list.
    s.key(Enter, "Enter").keyframe(
        "15-after-code-closer-enter",
        "After the closer fence, expecting a new paragraph context. \
         Author about to start item 2 of the outer list.",
    );

    // Type the next item marker. If the outer list state is still
    // active, just typing content should re-enter the list. If not,
    // the author types `2. `.
    s.type_("2. ")
        .keyframe("16-second-item-marker", "Started item 2 of the response.");

    s.type_("On the test mocks for the schema-change worker:")
        .keyframe(
            "17-second-item-content",
            "Item 2 content — same shape as item 1.",
        );

    // Quote the second comment.
    s.key(ShiftEnter, "ShiftEnter")
        .key(ShiftEnter, "ShiftEnter")
        .type_("> Don't mock the worker; the test should hit a real DB.")
        .keyframe("18-second-bq", "Reviewer quote in item 2.");

    // Acknowledge with a sub-list of two ordered points.
    s.key(ShiftEnter, "ShiftEnter")
        .key(ShiftEnter, "ShiftEnter")
        .type_("Agreed — splitting the ack into two parts:")
        .keyframe("19-second-item-followup", "Lead-in for the sub-list.");

    s.key(ShiftEnter, "ShiftEnter")
        .key(ShiftEnter, "ShiftEnter")
        .type_("- Switched to a real testcontainers Postgres in the worker tests.")
        .keyframe(
            "20-sub-list-first-bullet",
            "First bullet — author wants a *nested* list inside item 2. \
             Pulldown will currently see this as a top-level continuation \
             unless the indent is right.",
        );

    s.key(Enter, "Enter")
        .type_("Removed the in-memory `MockMigrator` shim entirely.")
        .keyframe(
            "21-sub-list-second-bullet",
            "Second bullet of the sub-list. Enter inside a list item should \
             produce the next bullet at the same depth.",
        );

    // Recompose: copy the SQL code block and reuse it after the second
    // item. The author selects the code block from item 1 then pastes
    // after item 2's bullets.
    s.cursor_after("```").keyframe(
        "22-cursor-on-first-code-block",
        "Cursor parked at end of the first code block's opener — about \
             to range-select the whole code block.",
    );

    // Find the code block's range to select it. Naive approach: select
    // from the opening ` ``` ` to the closing ` ``` `.
    s.select_span("```sql", "DEFAULT 'free';").keyframe(
        "23-select-code-region",
        "Selected the whole code block content (opener + body) for copy/paste.",
    );

    // Simulate copy by reading the selected text, then move cursor
    // and insert it via type_ (which is exactly what paste does
    // through the InsertText pipeline).
    let copied = {
        let editor = s.editor.clone();
        let cx = s.cx();
        editor.read_with(cx, |e, _| match e.state.selection {
            Selection::Range { anchor, head } => {
                let lo = anchor.min(head);
                let hi = anchor.max(head);
                e.state.markdown[lo..hi].to_string()
            }
            Selection::Cursor(_) => String::new(),
        })
    };

    // Cancel the selection, move to the end of the document, paste.
    s.key(Right, "Right (collapse selection)")
        .key(gpui_markdown_editor::editor::DocumentEnd, "DocumentEnd")
        .keyframe(
            "24-end-of-doc",
            "Cursor at end of the document — about to paste the copied code block.",
        );

    s.key(ShiftEnter, "ShiftEnter")
        .key(ShiftEnter, "ShiftEnter")
        .type_(&copied)
        .keyframe(
            "25-pasted-code-block",
            "Pasted the SQL code block. Should land as a child of item 2 \
             (same indent context) since we used Shift+Enter twice.",
        );

    // Add a fence closer.
    s.key(Enter, "Enter").type_("```").keyframe(
        "26-pasted-code-closer",
        "Closer fence appended to the pasted block.",
    );

    // Cleanup pass: walk back to item 1's blockquote and edit a typo.
    s.cursor_before("Are we sure").keyframe(
        "27-cursor-back-to-bq",
        "Cursor returns to the start of the quoted comment — author wants \
             to edit a typo here.",
    );

    s.cursor_after("safe under concurrent writes?").keyframe(
        "28-end-of-bq",
        "Cursor at end of the BQ content — about to add a parenthetical.",
    );

    s.type_(" (esp. mid-failover)").keyframe(
        "29-bq-extended",
        "Extended the BQ content. Should still be one BQ paragraph inside item 1.",
    );

    // Finally: convert the unordered sub-list under item 2 to an
    // ordered list. The author goes back, places the cursor at the
    // first bullet, replaces `- ` with `1. ` for both lines.
    s.cursor_before("- Switched").keyframe(
        "30-cursor-on-first-bullet",
        "Cursor on the first sub-list bullet — about to convert to ordered.",
    );

    s.key(Delete, "Delete")
        .key(Delete, "Delete")
        .type_("1. ")
        .keyframe(
            "31-converted-first-bullet",
            "First bullet replaced with `1. ` marker. Sub-list is now ordered.",
        );

    s.cursor_before("- Removed")
        .key(Delete, "Delete")
        .key(Delete, "Delete")
        .type_("2. ")
        .keyframe(
            "32-converted-second-bullet",
            "Second bullet replaced with `2. ` — sub-list now has two ordered items.",
        );

    // End-of-session navigation: walk Down from doc start to ensure
    // every keyframe along the way is reachable.
    s.set_cursor(0, "back to doc start").keyframe(
        "33-walk-start",
        "Cursor at byte 0; about to Down-arrow through the whole composed document.",
    );

    for i in 0..30 {
        s.key(Down, "Down");
        if i % 5 == 4 {
            s.keyframe(
                &format!("34-walk-step-{i}"),
                "Down-arrow walk through deep nest — the cursor should never \
                 land in a forbidden pair interior or get stuck for >2 \
                 successive presses.",
            );
        }
    }

    s.key(End, "End").keyframe(
        "35-walk-end",
        "End-of-line at the deepest down-walk position — sanity that End \
         lands at the line terminus.",
    );

    s.finish();
}

// Suppress unused-import warnings for actions only referenced via the
// dispatcher above.
#[allow(dead_code)]
fn _used_actions() {
    let _ = (Backspace, Delete, Down, End, Enter, Home, Left, Right);
    let _ = (ShiftEnter, ShiftLeft, ShiftRight, ShiftTab, Tab);
}

// ---------------------------------------------------------------------------
// Nested code blocks — type a small document with code inside a BQ, inside a
// list, inside a BQ-inside-a-list, plus a top-level paragraph; then Backspace
// the whole thing away one keystroke at a time. This is a *discovery* harness:
// the artifacts are what we read to spot misbehavior; assertions are
// intentionally absent so the test doesn't hide failures behind panics.
// ---------------------------------------------------------------------------

impl Session {
    /// Type each character of `text` as its own `InsertText` event. Mirrors a
    /// real keyboard: each char goes through `enforce_invariants` separately,
    /// so order-of-events bugs surface here that would not under a bulk
    /// `type_("...")` paste.
    #[allow(dead_code)]
    fn type_chars(&mut self, text: &str) -> &mut Self {
        for ch in text.chars() {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            self.type_(s);
        }
        self
    }

    /// Type each char one at a time AND keyframe between every char. Use
    /// sparingly — the artifact directory grows quickly.
    fn type_chars_keyframed(&mut self, text: &str, prefix: &str, intent: &str) -> &mut Self {
        let mut acc = String::new();
        for ch in text.chars() {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            self.type_(s);
            acc.push(ch);
            let display: String = acc
                .chars()
                .map(|c| if c == '`' { 'B' } else { c })
                .collect();
            self.keyframe(&format!("{prefix}-{}", slugify(&display)), intent);
        }
        self
    }

    /// Press Backspace once and emit a keyframe. Used by the deletion phase
    /// to surface every interim state.
    fn backspace_keyframed(&mut self, n: usize, intent: &str) -> &mut Self {
        self.key(Backspace, "Backspace");
        self.keyframe(&format!("bs-{n:03}"), intent);
        self
    }
}

#[gpui::test]
fn nested_code_blocks_session(cx: &mut TestAppContext) {
    let mut s = Session::new("nested-code-blocks", cx);

    s.keyframe("00-blank", "Editor opens; nothing typed yet.");

    // ───────────────────────────────────────────────────────────────────
    // SUB-BLOCK 1: code block inside a top-level blockquote.
    //
    //   > ```rust
    //   > let x = 1;
    //   > ```
    //
    // Typing each char of the opening fence as its own event so we can see
    // whether the editor behaves like a fence the moment the third backtick
    // lands, or only after the info string + Enter follow.
    // ───────────────────────────────────────────────────────────────────

    s.type_chars_keyframed("> ", "01-bq-marker", "Opening a blockquote, char by char.");
    s.type_chars_keyframed("```", "02-bq-fence-open", "Typing the opening fence; expect the editor to recognize a code block once the third backtick lands.");
    s.type_chars_keyframed("rust", "03-bq-info", "Typing the language tag.");
    s.key(Enter, "Enter").keyframe(
        "04-bq-after-fence-enter",
        "Enter after the opener+info — without a closing fence, what does the editor do? \
         A naive Enter inserts `\\n\\n` (paragraph break) which would split the BQ; the \
         desired UX may be to inject a closing fence and place the cursor on a body line.",
    );
    s.type_chars_keyframed(
        "let x = 1;",
        "05-bq-body",
        "Typing the body of the code block.",
    );
    s.key(Enter, "Enter").keyframe(
        "06-bq-after-body-enter",
        "Enter inside code body — literal `\\n` + chain prefix; closer below already exists from auto-close.",
    );
    s.type_chars_keyframed("```", "07-bq-fence-close", "Typing the closing fence.");
    s.keyframe(
        "08-bq-block-complete",
        "Code block inside BQ should now be a closed fenced block.",
    );
    s.key(Enter, "Enter").keyframe(
        "09-bq-leave-enter",
        "Enter after the closing fence — does this leave the BQ scope or stay inside it?",
    );

    // Whatever state we're in, get out to top-level cleanly. If still in BQ,
    // pressing Enter again on an empty BQ row should outdent.
    s.key(Enter, "Enter").keyframe(
        "10-back-to-top-level",
        "Second Enter — expect to be at top level by now (any BQ scope dropped).",
    );

    s.type_chars_keyframed("- ", "11-li-marker", "Opening an unordered list item.");
    s.type_chars_keyframed("item", "12-li-content", "Item content.");
    s.key(ShiftEnter, "ShiftEnter").keyframe(
        "13-li-shift-enter-1",
        "First ShiftEnter — hard break inside the item, indent on next line.",
    );
    s.key(ShiftEnter, "ShiftEnter").keyframe(
        "14-li-shift-enter-2",
        "Second ShiftEnter — paragraph break inside the item.",
    );
    s.type_chars_keyframed(
        "```",
        "15-li-fence-open",
        "Typing the opening fence inside the item.",
    );
    s.type_chars_keyframed("rust", "16-li-info", "Language tag.");
    s.key(Enter, "Enter").keyframe(
        "17-li-after-fence-enter",
        "Enter after opener inside an LI — auto-close should fire.",
    );
    s.type_chars_keyframed("let y = 2;", "18-li-body", "Code body inside the LI.");
    s.key(Enter, "Enter").keyframe(
        "19-li-after-body-enter",
        "Enter inside code body — literal `\\n`.",
    );
    s.type_chars_keyframed("```", "20-li-fence-close", "Closing fence.");
    s.keyframe("21-li-block-complete", "Code block inside LI complete.");
    s.key(Enter, "Enter")
        .keyframe("22-li-leave-enter", "Enter after the closing fence.");
    s.key(Enter, "Enter").keyframe(
        "23-back-to-top-level-after-li",
        "Second Enter — expect to be at top level.",
    );

    s.type_chars_keyframed("- ", "24-li2-marker", "Second list — top-level.");
    s.type_chars_keyframed("item", "25-li2-content", "Item content.");
    s.key(ShiftEnter, "ShiftEnter")
        .key(ShiftEnter, "ShiftEnter")
        .keyframe(
            "26-li2-paragraph-break",
            "Paragraph break inside item 1 — about to open a BQ child.",
        );
    s.type_chars_keyframed("> ", "27-li2-bq-marker", "BQ marker inside the LI.");
    s.type_chars_keyframed(
        "```",
        "28-li2-bq-fence-open",
        "Opening fence inside [LI, BQ].",
    );
    s.type_chars_keyframed("rust", "29-li2-bq-info", "Language tag.");
    s.key(Enter, "Enter").keyframe(
        "30-li2-bq-after-fence-enter",
        "Enter after opener inside [LI, BQ] — auto-close should fire.",
    );
    s.type_chars_keyframed("let z = 3;", "31-li2-bq-body", "Body inside [LI, BQ] code.");
    s.key(Enter, "Enter").keyframe(
        "32-li2-bq-after-body-enter",
        "Enter inside [LI, BQ] code body.",
    );
    s.type_chars_keyframed("```", "33-li2-bq-fence-close", "Closing fence in [LI, BQ].");
    s.keyframe(
        "34-li2-bq-block-complete",
        "Code block inside [LI, BQ] complete.",
    );
    s.key(Enter, "Enter").keyframe(
        "35-li2-bq-leave-enter",
        "Enter after closing fence in [LI, BQ].",
    );
    s.key(Enter, "Enter").keyframe(
        "36-back-to-top-level-after-li2",
        "Second Enter — expect to be at top level.",
    );
    s.type_chars_keyframed(
        "Done.",
        "37-trailing-paragraph",
        "Top-level closing paragraph.",
    );
    s.keyframe(
        "38-document-complete",
        "Document fully composed. Ready to delete it from the end.",
    );

    // ───────────────────────────────────────────────────────────────────
    // PREPEND a deeply-nested construct: code block in BQ in BQ in OL in UL.
    //
    // The chain at code body is `[UL_LI, OL_LI, BQ, BQ]` (5 levels of
    // structural nesting once the CodeBlock is included). We navigate
    // back to byte 0 of the existing document and type the construct
    // keystroke-by-keystroke. The existing document follows; the union
    // of the two will exercise re-parse paths under deep nesting.
    // ───────────────────────────────────────────────────────────────────
    s.set_cursor(0, "navigate to doc start for prepend")
        .keyframe(
            "39-cursor-at-doc-start",
            "Cursor at byte 0; about to prepend a deeply-nested construct \
         (code block in BQ in BQ in OL in UL) before the existing document.",
        );

    s.type_chars_keyframed("- ", "40-prep-ul-marker", "Opening the outer UL.");
    s.type_chars_keyframed("out", "41-prep-ul-body", "UL item content.");
    s.key(ShiftEnter, "ShiftEnter")
        .key(ShiftEnter, "ShiftEnter")
        .keyframe(
            "42-prep-ul-paragraph-break",
            "Paragraph break inside the UL item.",
        );

    s.type_chars_keyframed(
        "1. ",
        "43-prep-ol-marker",
        "Opening the nested OL inside the UL.",
    );
    s.type_chars_keyframed("in", "44-prep-ol-body", "OL item content.");
    s.key(ShiftEnter, "ShiftEnter")
        .key(ShiftEnter, "ShiftEnter")
        .keyframe(
            "45-prep-ol-paragraph-break",
            "Paragraph break inside the OL item.",
        );

    s.type_chars_keyframed("> ", "46-prep-bq1-marker", "Opening the outer BQ.");
    s.type_chars_keyframed(
        "> ",
        "47-prep-bq2-marker",
        "Opening the nested BQ inside the outer BQ.",
    );
    s.type_chars_keyframed(
        "```",
        "48-prep-fence-open",
        "Opening fence inside [UL, OL, BQ, BQ].",
    );
    s.type_chars_keyframed("rust", "49-prep-fence-info", "Language tag.");
    s.key(Enter, "Enter").keyframe(
        "50-prep-after-fence-enter",
        "Enter at end of opener — auto-close should fire at chain [UL, OL, BQ, BQ].",
    );
    s.type_chars_keyframed("let n = 5;", "51-prep-body", "Code body.");
    s.key(Enter, "Enter").keyframe(
        "52-prep-after-body-enter",
        "Enter inside body — literal `\\n` plus chain prefix.",
    );
    s.type_chars_keyframed("```", "53-prep-fence-close", "Closing fence (manual).");
    s.keyframe(
        "54-prep-block-complete",
        "Deeply-nested construct prepended; orphan dedupe should keep it clean.",
    );

    let mut step = 0usize;
    let max_steps = 1000usize;
    let initial_len = {
        let editor = s.editor.clone();
        let cx = s.cx();
        editor.read_with(cx, |e, _| e.state.markdown.len())
    };
    let mut min_len_seen = initial_len;
    let mut steps_without_progress = 0usize;
    const NO_PROGRESS_BAIL: usize = 60;
    // Treat a buffer that has grown to 1.5x its starting size as a
    // runaway. Backspace should monotonically shrink the buffer; a
    // sustained net growth means the corrupted-state oscillation
    // (`bugs.md::backspace_oscillates_inside_corrupted_chain`) is
    // doing its work and we'll eventually overflow pulldown's
    // recursion if we keep going. Stop here, write the keyframe, and
    // let the regression review pick up from this state.
    let runaway_threshold = (initial_len * 3) / 2;
    loop {
        if step >= max_steps {
            s.keyframe("99-bs-cap", &format!("Hit {max_steps}-step cap; aborting."));
            break;
        }
        let (markdown, cursor) = {
            let editor = s.editor.clone();
            let cx = s.cx();
            editor.read_with(cx, |e, _| (e.state.markdown.clone(), e.cursor_offset()))
        };
        if markdown.is_empty() {
            s.keyframe("98-bs-empty", "Buffer empty.");
            break;
        }
        if markdown.len() > runaway_threshold {
            s.keyframe(
                "97-bs-runaway",
                &format!(
                    "Bailed: buffer at {} bytes exceeds 1.5x initial ({}).",
                    markdown.len(),
                    initial_len
                ),
            );
            break;
        }
        step += 1;
        s.backspace_keyframed(
            step,
            &format!(
                "Backspace #{step}. cursor was {cursor}, len {}",
                markdown.len()
            ),
        );
        // Stuck detection: if Backspace produced no net change to the
        // buffer or cursor, the press was a no-op. The most common
        // reason at this point is the cursor reaching byte 0 while
        // content still remains — Backspace at byte 0 has nothing to
        // delete. Try jumping back to end-of-doc once and continuing;
        // if even that doesn't make progress, we're truly stuck.
        let (next_md, next_cursor) = {
            let editor = s.editor.clone();
            let cx = s.cx();
            editor.read_with(cx, |e, _| (e.state.markdown.clone(), e.cursor_offset()))
        };
        if next_md == markdown && next_cursor == cursor {
            if cursor == 0 && !markdown.is_empty() {
                s.key(
                    gpui_markdown_editor::editor::DocumentEnd,
                    "DocumentEnd (recover from byte-0 stuck)",
                );
                continue;
            }
            s.keyframe(
                "96-bs-stuck",
                &format!(
                    "Bailed: Backspace #{step} left state unchanged \
                     (cursor {cursor}, len {}).",
                    markdown.len()
                ),
            );
            break;
        }
        // Oscillation / no-progress detection: track the smallest
        // buffer size observed; if Backspace hasn't shrunk past that
        // low-water mark in `NO_PROGRESS_BAIL` consecutive presses,
        // we're cycling between two structurally equivalent states
        // (the canonical case is `\n[prefix]\n` ↔ `\n[prefix]\n[prefix]`
        // where promote_soft_breaks re-injects the half Backspace
        // just removed). Bail with a useful keyframe instead of
        // burning through the 1000-step cap.
        if next_md.len() < min_len_seen {
            min_len_seen = next_md.len();
            steps_without_progress = 0;
        } else {
            steps_without_progress += 1;
            if steps_without_progress >= NO_PROGRESS_BAIL {
                if next_cursor == 0 && !next_md.is_empty() {
                    s.key(
                        gpui_markdown_editor::editor::DocumentEnd,
                        "DocumentEnd (recover from byte-0 oscillation)",
                    );
                    steps_without_progress = 0;
                    continue;
                }
                s.keyframe(
                    "96-bs-stuck",
                    &format!(
                        "Bailed: {NO_PROGRESS_BAIL} consecutive presses \
                         without buffer shrinking past {min_len_seen} \
                         (current len {}, cursor {next_cursor}).",
                        next_md.len()
                    ),
                );
                break;
            }
        }
    }
    s.finish();
}
