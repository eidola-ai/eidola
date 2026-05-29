//! Visual snapshot tests — `harness = false` so `fn main()` runs on the
//! macOS main thread (libtest's worker harness would SIGABRT inside AppKit).
//!
//! Run:
//! - `cargo test -p gpui-markdown-editor --test visual`
//! - `UPDATE_SNAPSHOTS=1 cargo test -p gpui-markdown-editor --test visual`
//!
//! Snapshots are written to `tests/snapshots/`. They're a local debug aid,
//! not a regression gate — see `AGENTS.md` for the rationale.

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
