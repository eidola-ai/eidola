//! Build-time localization codegen: every locale under `locales/` is checked
//! and embedded, with one typed accessor per English message, into
//! `$OUT_DIR/i18n_generated.rs` (`src/ui/i18n.rs` includes it). The same
//! codegen as the desktop app's; nothing reads a translation at runtime.

fn main() {
    eidola_i18n::build_script::run("i18n_generated.rs");
}
