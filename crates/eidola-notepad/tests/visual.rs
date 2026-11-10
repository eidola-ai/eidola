//! Visual snapshots of a document window — a local debug aid, not a gate.
//!
//! Renders `DocumentView` offscreen through the real Metal renderer in
//! Circadian Day and Night and writes PNGs to `tests/snapshots/` (gitignored:
//! pixels are machine-bound). Run with
//! `EIDOLA_RUN_VISUAL_TESTS=1 cargo test -p eidola-notepad --test visual`.
//! `harness = false`: AppKit needs the main thread.

#[cfg(target_os = "macos")]
fn main() {
    if !matches!(
        std::env::var("EIDOLA_RUN_VISUAL_TESTS").as_deref(),
        Ok("1") | Ok("true")
    ) {
        println!("visual snapshots skipped; set EIDOLA_RUN_VISUAL_TESTS=1 to render them");
        return;
    }
    snapshots::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {}

#[cfg(target_os = "macos")]
mod snapshots {
    use std::path::PathBuf;
    use std::sync::Arc;

    use eidola_notepad::document::{Contents, DocumentView};
    use eidola_notepad::file;
    use gpui::{AppContext, VisualTestAppContext, px, size};
    use gpui_component::{Theme, ThemeMode};
    use gpui_kit_assets::Assets;

    const NOTE: &str = "\
---
title: Morning pages
tags: [notes, journal]
---

# Morning pages

Some thoughts before the day starts. **Bold ideas**, *quiet ones*, and `inline code` all sit together in the same prose.

## A short list

- Walk before coffee
- Read the chapter on *attention*
  - take notes in the margin
- [ ] Call the plumber
- [x] Water the plants

> A quotation sits in its own column, set off by a rule.

![The mark](assets/mark.png)

```rust
fn main() {
    println!(\"hello, notes\");
}
```

Inline math such as $e^{i\\pi} + 1 = 0$ typesets beside the text.
";

    pub fn run() {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let out = manifest.join("tests").join("snapshots");
        std::fs::create_dir_all(&out).expect("snapshot dir");

        // A note on disk beside an image, so the relative image path is
        // exercised the way a real vault would.
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("assets")).unwrap();
        std::fs::copy(
            manifest.join("../../www/static/apple-touch-icon.png"),
            dir.path().join("assets/mark.png"),
        )
        .expect("copy image");
        let note = dir.path().join("Morning pages.md");
        std::fs::write(&note, NOTE).unwrap();
        let (text, format) = file::decode(NOTE.as_bytes()).unwrap();

        let platform = gpui_platform::current_platform(false);
        let mut cx = VisualTestAppContext::with_asset_source(platform, Arc::new(Assets));
        cx.update(|cx| {
            gpui_component::init(cx);
            gpui_component::set_window_border_enabled(false, cx);
            eidola_gui::theme::install(cx);
            gpui_markdown_editor::init(cx);
        });

        for (mode, suffix) in [(ThemeMode::Light, "day"), (ThemeMode::Dark, "night")] {
            cx.update(|cx| Theme::change(mode, None, cx));
            for (name, contents) in [
                (
                    "document",
                    Contents::File {
                        path: note.clone(),
                        text: text.clone(),
                        format,
                    },
                ),
                ("untitled", Contents::Untitled(1)),
            ] {
                let window = cx
                    .open_offscreen_window(size(px(896.), px(840.)), |window, cx| {
                        let view = cx.new(|cx| DocumentView::new(contents, window, cx));
                        cx.new(|cx| eidola_gui::chrome::themed_root(view.into(), window, cx))
                    })
                    .expect("open offscreen window");
                cx.run_until_parked();
                let image = cx
                    .capture_screenshot(window.into())
                    .expect("capture screenshot");
                let path = out.join(format!("{name}-{suffix}.png"));
                image.save(&path).expect("write snapshot");
                println!("  written  {}", path.display());
            }
        }
    }
}
