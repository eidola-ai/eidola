//! Anchor crate for Rust-based developer tools.
//!
//! This crate has no code of its own. It exists only to pull upstream tool
//! crates (currently [`rumdl`] and [`just`]) into the workspace dep graph
//! so their versions are pinned by `Cargo.lock`. `.envrc` builds each by
//! its own package name and direnv puts `target/debug/` on `PATH`:
//!
//! ```text
//! cargo build --quiet -p rumdl -p just
//! PATH_add target/debug
//! ```
//!
//! so a `cargo` + `direnv` install yields the pinned tools with no separate
//! `brew install just` / `cargo install rumdl` step. CI doesn't install
//! direnv; it emulates `.envrc` instead — building only the tool a step
//! needs and prepending `target/debug` to `PATH`, so the invocation matches
//! the local one (e.g. `cargo build -p rumdl` then `rumdl check .`).
//!
//! Add a tool by adding a `=X.Y.Z` dep in this crate's `Cargo.toml` and a
//! matching `-p <tool>` in `.envrc`. Only crates with a `lib` target can be
//! anchored — a bin-only crate cannot be a `[dependencies]` entry. The
//! package is named `eidola-devtools` (not after any tool) so
//! `cargo build -p <tool>` resolves unambiguously to the registry crate.
