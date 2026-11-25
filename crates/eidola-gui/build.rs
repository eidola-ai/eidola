//! Build-time localization codegen.
//!
//! Reads `locales/<tag>/*.ftl`, parses every locale, enforces the codegen
//! contract (`eidola-i18n`'s `codegen`), and writes one typed accessor per
//! English message into `$OUT_DIR/i18n_generated.rs`, which `src/i18n.rs`
//! `include!`s.
//!
//! The FTL itself is emitted as string literals in that generated file, so
//! every shipped string is a build input inside the measured artifact. Nothing
//! reads a translation from disk at runtime — a file that could be edited after
//! the build could rewrite "this connection could not be verified" into
//! "verified", and no signature would break.

fn main() {
    eidola_i18n::build_script::run("i18n_generated.rs");
}
