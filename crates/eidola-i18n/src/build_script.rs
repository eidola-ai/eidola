//! The build-script half: read an app's `locales/<tag>/*.ftl`, parse every
//! locale, enforce the codegen contract ([`crate::codegen`]), and write one
//! typed accessor per source message into `$OUT_DIR/<file>`, which the app's
//! `i18n` module `include!`s.
//!
//! The FTL itself is emitted as string literals in that generated file, so
//! every shipped string is a build input inside the measured artifact. Nothing
//! reads a translation from disk at runtime — a file that could be edited after
//! the build could rewrite "this connection could not be verified" into
//! "verified", and no signature would break.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::{env, fs};

use crate::SOURCE_LOCALE;
use crate::codegen;

/// The whole build step, for an app's `build.rs`: the locales under
/// `$CARGO_MANIFEST_DIR/locales`, generated into `$OUT_DIR/<out_file>`. Every
/// localization problem is a build failure — a malformed translation must stop
/// the build, not ship.
pub fn run(out_file: &str) {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let locales_dir = manifest_dir.join("locales");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", locales_dir.display());

    let generated = match generate_dir(&locales_dir) {
        Ok(generated) => generated,
        Err(e) => fail(&e),
    };
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join(out_file);
    if let Err(e) = fs::write(&out, generated) {
        fail(&format!("failed to write {}: {e}", out.display()));
    }
}

/// Parse and check every locale under `locales_dir` and return the generated
/// module's source.
pub fn generate_dir(locales_dir: &Path) -> Result<String, String> {
    let sources = read_locale_sources(locales_dir)?;

    let mut parsed = Vec::new();
    for (tag, source) in &sources {
        parsed.push(codegen::parse_locale(tag, source)?);
    }

    let Some(en) = parsed.iter().find(|l| l.tag == SOURCE_LOCALE).cloned() else {
        return Err(format!(
            "no `{SOURCE_LOCALE}` locale in {} — the source locale is what every accessor \
             is generated from",
            locales_dir.display()
        ));
    };

    // Everything the source locale ships must resolve on its own...
    codegen::check_locale(&en, None)?;
    // ...and everything each translation ships must resolve through the merged
    // view the runtime formats it in, and stay consistent with the source.
    for locale in &parsed {
        if locale.tag == SOURCE_LOCALE {
            continue;
        }
        codegen::check_locale(locale, Some(&en))?;
        codegen::check_translation(&en, locale)?;
    }

    codegen::generate(&en, &parsed)
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
