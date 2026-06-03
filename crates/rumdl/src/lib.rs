//! Anchor crate.
//!
//! This crate exists only to pull the upstream [`rumdl`] crate into the
//! workspace dep graph so its version is pinned by `Cargo.lock`. There is
//! no code of our own here — `just check` and CI run the upstream rumdl
//! binary via:
//!
//! ```text
//! cargo build --release -p rumdl --quiet
//! ./target/release/rumdl check .
//! ```
//!
//! Bump the pin by editing `[dependencies] rumdl = "=X.Y.Z"` in this
//! crate's `Cargo.toml`; `Cargo.lock` updates and the next build uses the
//! new version. The package is named `rumdl-pinned` (not `rumdl`) so
//! `cargo build -p rumdl` resolves unambiguously to the registry crate.
