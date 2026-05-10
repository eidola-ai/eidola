//! Standalone demo window for `gpui-markdown-editor`. Three-pane layout:
//!
//! - **Editor** (left, flex-grow) — the WYSIWYG editor.
//! - **Source** (middle, fixed width) — the live raw markdown buffer,
//!   selectable so you can copy and inspect whitespace/escape characters.
//! - **AST** (right, fixed width) — `format!("{:#?}", parse(&md))` of the
//!   currently-parsed syntax tree, refreshed on every edit.
//!
//! The two debug panes track the editor live: a `cx.observe` on the editor
//! entity re-renders the parent on every state change.
//!
//! Run with `cargo run -p gpui-markdown-editor --bin demo`.

use gpui::{
    App, AppContext, Bounds, Context, Entity, InteractiveElement, IntoElement, KeyBinding,
    ParentElement, Render, SharedString, StatefulInteractiveElement, Styled, Window, WindowBounds,
    WindowOptions, div, prelude::FluentBuilder, px, rems, size,
};
use gpui_component::{Root, Theme, h_flex, text::TextView, v_flex};
use gpui_component_assets::Assets;
use gpui_markdown_editor::{
    Backspace, Copy, Cut, Delete, DocumentEnd, DocumentStart, Down, End, Enter, Home, Left,
    MarkdownEditor, Paste, Right, SelectAll, ShiftDocumentEnd, ShiftDocumentStart, ShiftDown,
    ShiftEnd, ShiftEnter, ShiftHome, ShiftLeft, ShiftRight, ShiftTab, ShiftUp, Tab, Up, parse,
};

const DEMO_DOCUMENT: &str = "\
# gpui-markdown-editor

A WYSIWYG markdown editor. The first cut covers ATX headings, **bold**,
*italic*, ~~strikethrough~~, `inline code`, and [links](https://example.com).

## Cursor-aware delimiters

When the cursor is outside a construct the delimiters hide; when it's
inside they reveal in a dimmed color. Try clicking around in the
**bold** and *italic* runs above to see them flip in and out.

### Mix and match

You can combine ***bold and italic*** as a triple-asterisk run, or use
~~strikethrough~~ inside a paragraph alongside other styling. Inline
code like `let x = 42;` shapes in the mono font with a faint background.

---

Thematic breaks (`---` / `***` / `___`) render as a thin horizontal
rule. The source bytes hide when the cursor is elsewhere and reveal
(dimmed) when the cursor is on the rule line.

### Lists

- First bullet item.
- A nested case:
  - Inside another bullet.
  - Two-level nesting.
- A third one.

1. Numbered, starting at one.
2. With a nested ordered list:
   1. First sub-item.
   2. Second sub-item.
3. And so on.

### Task lists

- [x] Plan the work
- [x] Implement parsing
- [ ] Implement rendering
- [ ] Polish visuals

### Math

Inline math like $x^2 + y^2 = z^2$ typesets right next to the
prose, with a little extra row height for tall constructs such as
$\\frac{1}{1-x}$ or $\\sqrt{x^2 + y^2}$. Display math sits on its
own row:

$$\\frac{1}{1 - x} = \\sum_{n=0}^{\\infty} x^n$$

Click into the equation above to swap to edit mode and adjust the
LaTeX directly. CommonMark backslash escapes (`\\*`) and HTML
entities (`&copy;`, `&mdash;`) render as literals when the cursor
is elsewhere — try `\\*starred\\*` or 2026 &mdash; it works.
";

fn bind_keys(cx: &mut App) {
    cx.bind_keys([
        // Editing
        KeyBinding::new("backspace", Backspace, None),
        KeyBinding::new("delete", Delete, None),
        KeyBinding::new("enter", Enter, None),
        KeyBinding::new("shift-enter", ShiftEnter, None),
        KeyBinding::new("tab", Tab, None),
        KeyBinding::new("shift-tab", ShiftTab, None),
        // Caret motion
        KeyBinding::new("left", Left, None),
        KeyBinding::new("right", Right, None),
        KeyBinding::new("up", Up, None),
        KeyBinding::new("down", Down, None),
        KeyBinding::new("shift-left", ShiftLeft, None),
        KeyBinding::new("shift-right", ShiftRight, None),
        KeyBinding::new("shift-up", ShiftUp, None),
        KeyBinding::new("shift-down", ShiftDown, None),
        KeyBinding::new("home", Home, None),
        KeyBinding::new("end", End, None),
        KeyBinding::new("cmd-left", Home, None),
        KeyBinding::new("cmd-right", End, None),
        KeyBinding::new("shift-home", ShiftHome, None),
        KeyBinding::new("shift-end", ShiftEnd, None),
        KeyBinding::new("cmd-shift-left", ShiftHome, None),
        KeyBinding::new("cmd-shift-right", ShiftEnd, None),
        KeyBinding::new("cmd-up", DocumentStart, None),
        KeyBinding::new("cmd-down", DocumentEnd, None),
        KeyBinding::new("cmd-shift-up", ShiftDocumentStart, None),
        KeyBinding::new("cmd-shift-down", ShiftDocumentEnd, None),
        // Clipboard
        KeyBinding::new("cmd-a", SelectAll, None),
        KeyBinding::new("cmd-c", Copy, None),
        KeyBinding::new("cmd-x", Cut, None),
        KeyBinding::new("cmd-v", Paste, None),
    ]);
}

/// Top-level demo view. Owns the editor entity and observes it so the
/// debug panes track edits.
struct DemoApp {
    editor: Entity<MarkdownEditor>,
}

impl DemoApp {
    fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let editor = cx.new(|cx| MarkdownEditor::new(DEMO_DOCUMENT, window, cx));
        // Re-render this view whenever the editor's state changes — that's
        // how the source / AST panes track edits live.
        cx.observe(&editor, |_, _, cx| cx.notify()).detach();
        Self { editor }
    }
}

impl Render for DemoApp {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = Theme::global(cx);
        let editor = self.editor.read(cx);
        let md = editor.state.markdown.clone();
        let cursor_label = match editor.state.selection {
            gpui_markdown_editor::Selection::Cursor(p) => format!("cursor: {p}"),
            gpui_markdown_editor::Selection::Range { anchor, head } => {
                format!("selection: anchor={anchor} head={head}")
            }
        };
        let ast = format!("{:#?}", parse(&md));
        let bg = theme.background;
        let fg = theme.foreground;
        let muted = theme.muted_foreground;
        let border = theme.border;

        h_flex()
            .size_full()
            .bg(bg)
            .text_color(fg)
            // Editor pane.
            .child(
                div()
                    .id("editor-pane")
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .overflow_y_scroll()
                    .child(self.editor.clone()),
            )
            .child(div().w(px(1.)).h_full().bg(border))
            // Source pane.
            .child(debug_pane("source", Some(cursor_label), &md, muted, border))
            .child(div().w(px(1.)).h_full().bg(border))
            // AST pane.
            .child(debug_pane("ast", None, &ast, muted, border))
    }
}

/// Side pane showing `content` as a fenced code block (so the literal
/// text — including whitespace — is visible in monospace and selectable
/// for copy / paste).
fn debug_pane(
    label: &'static str,
    subtitle: Option<String>,
    content: &str,
    muted: gpui::Hsla,
    border: gpui::Hsla,
) -> impl IntoElement {
    let id_label = SharedString::from(format!("pane-{label}"));
    let body = SharedString::from(wrap_in_fenced_code_block(content));
    let view_id = SharedString::from(format!("pane-md-{label}"));

    v_flex()
        .w(px(360.))
        .h_full()
        .child(
            v_flex()
                .px_3()
                .py_2()
                .border_b_1()
                .border_color(border)
                .gap_0p5()
                .child(
                    div()
                        .text_xs()
                        .text_color(muted)
                        .child(SharedString::from(label.to_uppercase())),
                )
                .when_some(subtitle, |this, sub| {
                    this.child(
                        div()
                            .text_xs()
                            .text_color(muted)
                            .child(SharedString::from(sub)),
                    )
                }),
        )
        .child(
            div()
                .id(id_label)
                .flex_1()
                .min_h_0()
                .overflow_y_scroll()
                .px_3()
                .pb_3()
                .text_size(rems(0.85))
                .child(TextView::markdown(view_id, body).selectable(true)),
        )
}

/// Wrap `content` in a fenced code block. Picks a fence longer than any
/// run of backticks the content already contains so it round-trips
/// safely even if `content` has its own fenced blocks.
fn wrap_in_fenced_code_block(content: &str) -> String {
    let mut max = 0u32;
    let mut cur = 0u32;
    for c in content.chars() {
        if c == '`' {
            cur += 1;
            max = max.max(cur);
        } else {
            cur = 0;
        }
    }
    let fence: String = "`".repeat((max + 1).max(3) as usize);
    let trail = if content.ends_with('\n') { "" } else { "\n" };
    format!("{fence}\n{content}{trail}{fence}")
}

fn main() {
    gpui_platform::application()
        .with_assets(Assets)
        .run(|cx: &mut App| {
            gpui_component::init(cx);
            Theme::sync_system_appearance(None, cx);

            bind_keys(cx);

            let bounds = Bounds::centered(None, size(px(1280.0), px(800.0)), cx);
            let window = cx
                .open_window(
                    WindowOptions {
                        window_bounds: Some(WindowBounds::Windowed(bounds)),
                        ..Default::default()
                    },
                    |window, cx| {
                        let demo = cx.new(|cx| DemoApp::new(window, cx));
                        cx.new(|cx| Root::new(demo, window, cx))
                    },
                )
                .expect("open window");

            window
                .update(cx, |root, window, cx| {
                    if let Ok(demo) = root.view().clone().downcast::<DemoApp>() {
                        let editor = demo.read(cx).editor.clone();
                        let focus = editor.read(cx).focus_handle.clone();
                        window.focus(&focus, cx);
                    }
                    cx.activate(true);
                })
                .expect("focus editor");
        });
}
