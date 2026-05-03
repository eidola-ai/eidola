//! Visual snapshot tests — runs as a `harness = false` integration test so
//! `fn main()` executes on the macOS main thread (required by AppKit).
//!
//! Run:
//! - `cargo test -p eidola-gui --test visual`        — verify against goldens
//! - `UPDATE_SNAPSHOTS=1 cargo test -p eidola-gui --test visual` — accept new
//!
//! Snapshots are written to `apps/gui/tests/snapshots/`.

mod visual {
    pub mod cases;
    pub mod harness;
}

fn main() {
    let mut snapshots = visual::harness::Snapshots::new();
    visual::cases::register(&mut snapshots);
    snapshots.run_or_exit();
}
