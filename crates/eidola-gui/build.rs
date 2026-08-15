//! Build-time localization codegen.
//!
//! Reads `locales/<tag>/*.ftl`, parses every locale, enforces the source-locale
//! contract (see `build_support/codegen.rs`), and writes one typed accessor per
//! English message into `$OUT_DIR/i18n_generated.rs`, which `src/i18n.rs`
//! `include!`s.
//!
//! The FTL itself is emitted as string literals in that generated file, so
//! every shipped string is a build input inside the measured artifact. Nothing
//! reads a translation from disk at runtime — a file that could be edited after
//! the build could rewrite "this connection could not be verified" into
//! "verified", and no signature would break.

#[path = "build_support/codegen.rs"]
mod codegen;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::{env, fs};

/// The source locale: the one every accessor is generated from, and the one
/// every other locale falls back to at runtime.
const SOURCE_LOCALE: &str = "en";

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let locales_dir = manifest_dir.join("locales");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_support/codegen.rs");
    println!("cargo:rerun-if-changed={}", locales_dir.display());

    let sources = match read_locale_sources(&locales_dir) {
        Ok(sources) => sources,
        Err(e) => fail(&e),
    };

    let mut parsed = Vec::new();
    for (tag, source) in &sources {
        match codegen::parse_locale(tag, source) {
            Ok(locale) => parsed.push(locale),
            Err(e) => fail(&e),
        }
    }

    let Some(en) = parsed.iter().find(|l| l.tag == SOURCE_LOCALE).cloned() else {
        fail(&format!(
            "no `{SOURCE_LOCALE}` locale in {} — the source locale is what every accessor \
             is generated from",
            locales_dir.display()
        ));
    };

    for locale in &parsed {
        if locale.tag == SOURCE_LOCALE {
            continue;
        }
        if let Err(e) = codegen::check_translation(&en, locale) {
            fail(&e);
        }
    }

    let generated = match codegen::generate(&en, &parsed) {
        Ok(generated) => generated,
        Err(e) => fail(&e),
    };

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("i18n_generated.rs");
    if let Err(e) = fs::write(&out, generated) {
        fail(&format!("failed to write {}: {e}", out.display()));
    }
}

/// One entry per locale directory: its tag and its `.ftl` files concatenated in
/// filename order, so a locale may be split across topic files without the
/// embedded bytes depending on directory-iteration order.
fn read_locale_sources(dir: &Path) -> Result<Vec<(String, String)>, String> {
    let entries =
        fs::read_dir(dir).map_err(|e| format!("failed to read {}: {e}", dir.display()))?;

    let mut locales: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("failed to read {}: {e}", dir.display()))?;
        if !entry.path().is_dir() {
            continue;
        }
        let tag = entry
            .file_name()
            .to_str()
            .ok_or_else(|| format!("non-UTF-8 locale directory in {}", dir.display()))?
            .to_string();
        // The directory name *is* the locale's tag at runtime, so it has to be
        // one the runtime and negotiation will both recognize.
        codegen::validate_locale_tag(&tag)?;

        let mut files = BTreeMap::new();
        let inner = fs::read_dir(entry.path())
            .map_err(|e| format!("failed to read {}: {e}", entry.path().display()))?;
        for file in inner {
            let file = file.map_err(|e| format!("failed to read {tag}: {e}"))?;
            let path = file.path();
            if path.extension().and_then(|e| e.to_str()) != Some("ftl") {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| format!("non-UTF-8 FTL filename in {tag}"))?
                .to_string();
            let text = fs::read_to_string(&path)
                .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
            println!("cargo:rerun-if-changed={}", path.display());
            files.insert(name, text);
        }
        if files.is_empty() {
            return Err(format!("locale `{tag}` contains no .ftl files"));
        }
        locales.insert(tag, files);
    }

    if locales.is_empty() {
        return Err(format!("no locale directories in {}", dir.display()));
    }

    Ok(locales
        .into_iter()
        .map(|(tag, files)| {
            let source = files.into_values().collect::<Vec<_>>().join("\n");
            (tag, source)
        })
        .collect())
}

/// Report a localization problem as a build failure. Every rule in the codegen
/// contract lands here: a malformed translation must stop the build, not ship.
fn fail(message: &str) -> ! {
    println!("cargo:warning=localization: {message}");
    panic!("localization: {message}");
}
