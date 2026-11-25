//! Localization for Eidola's apps: compile-time Fluent, typed accessors, and
//! the rules both apps format by.
//!
//! - [`codegen`] + [`build_script`] (feature `codegen`, for an app's
//!   `build.rs`): every locale's FTL is checked against the contract and
//!   embedded as string literals, with one typed accessor per source message.
//!   **Nothing loads a translation at runtime** — the artifact hash is the trust
//!   root, so a translation read from disk would be an unmeasured input able to
//!   rewrite security-critical text.
//! - [`runtime`] (feature `runtime`): the merged-bundle fallback, formatting,
//!   and locale negotiation (Chinese by script).
//!
//! The generated module calls `super::format`, `super::format_in`,
//! `super::Args` and `super::Arg`, so each app's `i18n` module defines those
//! four over [`runtime`] and `include!`s it.

/// The source locale: the one every accessor is generated from, and the one
/// every other locale falls back to at runtime.
pub const SOURCE_LOCALE: &str = "en";

#[cfg(feature = "codegen")]
pub mod build_script;
#[cfg(feature = "codegen")]
pub mod codegen;
#[cfg(feature = "runtime")]
pub mod runtime;
