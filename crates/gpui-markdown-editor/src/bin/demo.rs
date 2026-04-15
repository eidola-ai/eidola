//! Standalone demo window for `gpui-markdown-editor`. Useful for blind
//! agent iteration and for eyeballing changes during development.
//!
//! Run with `cargo run -p gpui-markdown-editor --bin demo`.

use gpui::{App, AppContext, Bounds, Focusable, KeyBinding, WindowBounds, WindowOptions, px, size};
use gpui_component::{Root, Theme};
use gpui_component_assets::Assets;
use gpui_markdown_editor::{
    Backspace, Copy, Cut, Delete, DocumentEnd, DocumentStart, Down, End, Enter, Home, Left,
    MarkdownEditor, Paste, Right, SelectAll, ShiftDocumentEnd, ShiftDocumentStart, ShiftDown,
    ShiftEnd, ShiftEnter, ShiftHome, ShiftLeft, ShiftRight, ShiftUp, Up,
};

const DEMO_DOCUMENT: &str = "\
# gpui-markdown-editor

A WYSIWYG markdown editor. The first cut covers ATX headings, **bold**,
*italic*, and ~~strikethrough~~.

## Cursor-aware delimiters

When the cursor is outside a construct the delimiters hide; when it's
inside they reveal in a dimmed color. Try clicking around in the
**bold** and *italic* runs above to see them flip in and out.

### Mix and match

You can combine ***bold and italic*** as a triple-asterisk run, or use
~~strikethrough~~ inside a paragraph alongside other styling.
";

fn bind_keys(cx: &mut App) {
    cx.bind_keys([
        // Editing
        KeyBinding::new("backspace", Backspace, None),
        KeyBinding::new("delete", Delete, None),
        KeyBinding::new("enter", Enter, None),
        KeyBinding::new("shift-enter", ShiftEnter, None),
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

fn main() {
    gpui_platform::application()
        .with_assets(Assets)
        .run(|cx: &mut App| {
            gpui_component::init(cx);
            // Use whichever theme matches the OS appearance.
            Theme::sync_system_appearance(None, cx);

            bind_keys(cx);

            let bounds = Bounds::centered(None, size(px(900.0), px(720.0)), cx);
            let window = cx
                .open_window(
                    WindowOptions {
                        window_bounds: Some(WindowBounds::Windowed(bounds)),
                        ..Default::default()
                    },
                    |window, cx| {
                        let editor = cx.new(|cx| MarkdownEditor::new(DEMO_DOCUMENT, window, cx));
                        cx.new(|cx| Root::new(editor, window, cx))
                    },
                )
                .expect("open window");

            window
                .update(cx, |root, window, cx| {
                    if let Ok(view) = root.view().clone().downcast::<MarkdownEditor>() {
                        window.focus(&view.focus_handle(cx), cx);
                    }
                    cx.activate(true);
                })
                .expect("focus editor");
        });
}
