//! Visual snapshot tests — `harness = false` so `fn main()` runs on the
//! macOS main thread (libtest's worker harness would SIGABRT inside AppKit).
//!
//! Run:
//! - `cargo test -p gpui-markdown-editor --test visual`
//! - `UPDATE_SNAPSHOTS=1 cargo test -p gpui-markdown-editor --test visual`
//!
//! Snapshots are written to `tests/snapshots/`. They're a local debug aid,
//! not a regression gate — see `AGENTS.md` for the rationale.

mod visual {
    pub mod cases;
    pub mod harness;
}

fn main() {
    let mut snapshots = visual::harness::Snapshots::new();
    visual::cases::register(&mut snapshots);
    snapshots.run_or_exit();
}
