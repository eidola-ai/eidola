//! Snapshot test harness for gpui views.
//!
//! Renders each case **twice** — once with `Circadian Day` (Light) and once
//! with `Circadian Night` (Dark) — and writes `<name>-day.png` /
//! `<name>-night.png` into `tests/snapshots/`. The harness drives theme mode
//! itself rather than reading the OS appearance, so the output is the same
//! whether the developer's machine is in Light or Dark mode.
//!
//! Behavior per (case, mode):
//! - If the golden does not exist, the new image is written as the golden and
//!   the case is reported as `written`.
//! - If `UPDATE_SNAPSHOTS=1` is set, the golden is overwritten.
//! - Otherwise, a mismatch writes a sibling `<name>-<mode>.new.png` for
//!   review and the case fails.
//!
//! The runner exits with a non-zero status if any case failed.

use std::path::PathBuf;
use std::sync::Arc;

use gpui::{AppContext, Entity, Pixels, Render, Size, VisualTestAppContext};
use gpui_component::{Root, Theme, ThemeMode};
use gpui_component_assets::Assets;
use image::RgbaImage;

/// Build closure: must be `Fn` so the harness can invoke it once per mode.
/// Cases capture nothing reusable — they construct fresh entities each call.
type BuildRoot = Box<dyn Fn(&mut gpui::Window, &mut gpui::App) -> Entity<Root>>;

struct Case {
    name: &'static str,
    size: Size<Pixels>,
    build: BuildRoot,
}

const MODES: &[(ThemeMode, &str)] = &[(ThemeMode::Light, "day"), (ThemeMode::Dark, "night")];

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
    /// window; the harness wraps it in a `Root` and captures the result. The
    /// closure must be `Fn`: the harness invokes it once per theme mode, with
    /// fresh entities each time.
    pub fn add<V, F>(&mut self, name: &'static str, size: Size<Pixels>, build: F)
    where
        V: Render + 'static,
        F: Fn(&mut gpui::Window, &mut gpui::App) -> Entity<V> + 'static,
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
        cx.update(|cx| {
            gpui_component::init(cx);
            eidola_gui::theme::install(cx);
        });

        let total = self.cases.len() * MODES.len();
        let mut written: Vec<String> = Vec::new();
        let mut failed: Vec<String> = Vec::new();
        let mut passed: usize = 0;

        for case in self.cases {
            for (mode, suffix) in MODES.iter().copied() {
                let label = format!("{}-{}", case.name, suffix);
                let path = self.snapshot_dir.join(format!("{label}.png"));
                let new_path = self.snapshot_dir.join(format!("{label}.new.png"));

                // Always remove a stale .new.png from a previous run.
                let _ = std::fs::remove_file(&new_path);

                cx.update(|cx| Theme::change(mode, None, cx));
                let img = render_case(&mut cx, case.size, case.build.as_ref());

                if !path.exists() || self.update {
                    img.save(&path).expect("write golden snapshot");
                    written.push(label.clone());
                    println!("  written  {label}");
                    continue;
                }

                let golden = match image::open(&path) {
                    Ok(img) => img.to_rgba8(),
                    Err(e) => {
                        eprintln!("  fail     {label} (cannot read golden: {e})");
                        failed.push(label);
                        continue;
                    }
                };

                if images_equal(&img, &golden) {
                    passed += 1;
                    println!("  ok       {label}");
                } else {
                    img.save(&new_path).expect("write .new.png");
                    failed.push(label.clone());
                    println!(
                        "  fail     {label}  (review {})",
                        new_path
                            .strip_prefix(env!("CARGO_MANIFEST_DIR"))
                            .unwrap_or(&new_path)
                            .display()
                    );
                }
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

fn render_case(
    cx: &mut VisualTestAppContext,
    size: Size<Pixels>,
    build: &dyn Fn(&mut gpui::Window, &mut gpui::App) -> Entity<Root>,
) -> RgbaImage {
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
