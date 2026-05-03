//! Snapshot test harness for gpui views.
//!
//! Runs each registered case on a `VisualTestAppContext` (real Metal renderer,
//! offscreen window, deterministic dispatch), captures the resulting image,
//! and compares it byte-for-byte against a golden PNG in `tests/snapshots/`.
//!
//! Behavior:
//! - If the golden does not exist, the new image is written as the golden and
//!   the case is reported as `written`.
//! - If `UPDATE_SNAPSHOTS=1` is set, the golden is overwritten.
//! - Otherwise, a mismatch writes a sibling `<name>.new.png` for review and
//!   the case fails.
//!
//! The runner exits with a non-zero status if any case failed.

use std::path::PathBuf;
use std::sync::Arc;

use gpui::{AppContext, Entity, Pixels, Render, Size, VisualTestAppContext};
use gpui_component::Root;
use gpui_component_assets::Assets;
use image::RgbaImage;

type BuildRoot = Box<dyn FnOnce(&mut gpui::Window, &mut gpui::App) -> Entity<Root>>;

struct Case {
    name: &'static str,
    size: Size<Pixels>,
    build: BuildRoot,
}

pub struct Snapshots {
    cases: Vec<Case>,
    snapshot_dir: PathBuf,
    update: bool,
}

impl Snapshots {
    pub fn new() -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            cases: Vec::new(),
            snapshot_dir: manifest.join("tests").join("snapshots"),
            update: matches!(
                std::env::var("UPDATE_SNAPSHOTS").as_deref(),
                Ok("1") | Ok("true")
            ),
        }
    }

    /// Register a snapshot case. The closure builds the root view of the
    /// window; the harness wraps it in a `Root` and captures the result.
    pub fn add<V, F>(&mut self, name: &'static str, size: Size<Pixels>, build: F)
    where
        V: Render + 'static,
        F: FnOnce(&mut gpui::Window, &mut gpui::App) -> Entity<V> + 'static,
    {
        self.cases.push(Case {
            name,
            size,
            build: Box::new(move |window, cx| {
                let view = build(window, cx);
                cx.new(|cx| Root::new(view, window, cx))
            }),
        });
    }

    /// Run every registered case and exit the process with the appropriate
    /// status.
    pub fn run_or_exit(self) -> ! {
        std::fs::create_dir_all(&self.snapshot_dir).expect("create snapshot dir");

        let platform = gpui_platform::current_platform(false);
        let mut cx = VisualTestAppContext::with_asset_source(platform, Arc::new(Assets));
        cx.update(gpui_component::init);

        let total = self.cases.len();
        let mut written: Vec<&'static str> = Vec::new();
        let mut failed: Vec<&'static str> = Vec::new();
        let mut passed: usize = 0;

        for case in self.cases {
            let path = self.snapshot_dir.join(format!("{}.png", case.name));
            let new_path = self.snapshot_dir.join(format!("{}.new.png", case.name));

            // Always remove a stale .new.png from a previous run.
            let _ = std::fs::remove_file(&new_path);

            let img = render_case(&mut cx, case.size, case.build);

            if !path.exists() || self.update {
                img.save(&path).expect("write golden snapshot");
                written.push(case.name);
                println!("  written  {}", case.name);
                continue;
            }

            let golden = match image::open(&path) {
                Ok(img) => img.to_rgba8(),
                Err(e) => {
                    eprintln!("  fail     {} (cannot read golden: {e})", case.name);
                    failed.push(case.name);
                    continue;
                }
            };

            if images_equal(&img, &golden) {
                passed += 1;
                println!("  ok       {}", case.name);
            } else {
                img.save(&new_path).expect("write .new.png");
                failed.push(case.name);
                println!(
                    "  fail     {}  (review {})",
                    case.name,
                    new_path
                        .strip_prefix(env!("CARGO_MANIFEST_DIR"))
                        .unwrap_or(&new_path)
                        .display()
                );
            }
        }

        println!();
        println!(
            "visual snapshots: {} total, {} ok, {} written, {} failed",
            total,
            passed,
            written.len(),
            failed.len()
        );

        if !failed.is_empty() {
            println!();
            println!("To accept the new output of failed cases, rerun with:");
            println!("  UPDATE_SNAPSHOTS=1 cargo test -p eidola-gui --test visual");
        }

        std::process::exit(if failed.is_empty() { 0 } else { 1 });
    }
}

fn render_case(cx: &mut VisualTestAppContext, size: Size<Pixels>, build: BuildRoot) -> RgbaImage {
    let window = cx
        .open_offscreen_window(size, |window, cx| build(window, cx))
        .expect("open offscreen window");
    cx.run_until_parked();
    cx.capture_screenshot(window.into())
        .expect("capture screenshot")
}

fn images_equal(a: &RgbaImage, b: &RgbaImage) -> bool {
    a.dimensions() == b.dimensions() && a.as_raw() == b.as_raw()
}
