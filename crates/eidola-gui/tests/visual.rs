//! Visual snapshot tests — runs as a `harness = false` integration test so
//! `fn main()` executes on the macOS main thread (required by AppKit).
//!
//! Run:
//! - `cargo test -p eidola-gui --test visual`        — verify against goldens
//! - `UPDATE_SNAPSHOTS=1 cargo test -p eidola-gui --test visual` — accept new
//!
//! Snapshots are written to `crates/eidola-gui/tests/snapshots/`.

#[cfg(target_os = "macos")]
mod visual {
    pub mod cases;
    pub mod harness;
}

#[cfg(target_os = "macos")]
fn main() {
    let mut snapshots = visual::harness::Snapshots::new();
    visual::cases::register(&mut snapshots);
    snapshots.run_or_exit();
}

// On non-macOS targets (e.g. CI's Linux clippy/test runner), the visual
// harness can't link: `VisualTestAppContext` is gated on macOS in gpui, and
// the renderer paths target Metal. Compile to an empty test binary so the
// crate still builds on Linux.
#[cfg(not(target_os = "macos"))]
fn main() {}
