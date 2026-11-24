//! Writes the mobile app bundle's `licences.json`: the app's own licence,
//! every crate the wasm build links, every asset it embeds, and the Android
//! shell's one dependency, each with its licence and the licence texts, for
//! the app's licences screen.
//!
//! ```text
//! mobile-licences ROOT [--features F]... > licences.json
//! ```
//!
//! The crates are those `cargo tree --offline --locked` lists for
//! `eidola-mobile` on `wasm32-unknown-unknown` with the bundle's features
//! (normal edges, procedural macros left out: they run in the compiler and
//! are not in the wasm), each read from `cargo metadata`. A crate's texts are the licence
//! files its package ships (`LICENSE*`, `LICENCE*`, `COPYING*`, `NOTICE*`,
//! `COPYRIGHT*`, `UNLICENSE`, and the manifest's `license-file`); a crate
//! from a git checkout that ships none takes its repository's; any other that
//! ships none gets the SPDX texts of the licences it declares, from
//! `ROOT/LICENSES/`. This repository's own crates take theirs from
//! `ROOT/LICENSES/` too: a crate whose Cargo.toml names a licence under
//! `[package.metadata.eidola] licence` (the identifier `REUSE.toml` declares
//! for it, which Cargo's `license` field cannot carry) takes that file, and the
//! app's own entry is `eidola-mobile`'s. A GPL crate of ours without one is
//! refused: the app is conveyed through app stores, so every copyleft crate it
//! links must carry the app store permission. A crate built from a git source
//! (one of Eidola's forks) must be listed in [`FORKS`] at the commit the
//! lockfile names, and carries the notice of what Eidola changed in it. Then
//! the assets: the Newsreader faces (OFL-1.1), the KaTeX faces (OFL-1.1, with
//! the copyright their own name tables state), the Lucide icons (ISC), and the
//! Kotlin standard library R8 shrinks into the Android APK (Apache-2.0, with
//! the Kotlin distribution's NOTICE, copied into `notices/` for each release
//! `mobile/android/tools.sh` pins).
//!
//! The output is a function of the lockfile and the package sources it
//! names: sorted, every distinct text once, no paths, no times.
//!
//! It reads only what is already downloaded (`--offline`), and `cargo
//! metadata` reads the whole workspace's graph for the wasm target and the
//! host (cargo includes every package the host activates, whatever the
//! target), so the packages must be fetched for both:
//! `cargo fetch --locked --target wasm32-unknown-unknown --target <host>`, as
//! `mobile/build-bundle.sh` and the release image do.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

/// A crate built from one of Eidola's forks: the commit the lockfile must
/// name, and what Eidola changed in the crate at that commit (`None`: the
/// crate is as its upstream published it). Apache-2.0 section 4(b) requires
/// modified files to say so; the notice says it where the app shows the
/// crate's licence.
struct Fork {
    name: &'static str,
    commit: &'static str,
    changes: Option<&'static str>,
}

const GPUI_COMPONENT: &str = "61580c6ed8771a862faaef40a29ac3987884b6c0";
const GPUI_PRE: &str = "3bf5254801e0ed2b8d97228a7775f9dba3e9ec72";
const PULLDOWN_CMARK: &str = "fe50ea3ae3e4f1b3b5ab29f6a615814e0a6305c7";
const TURSO: &str = "eeda62d8d665a8e591635ea49091555af66662ae";

/// Every crate the bundle builds from a git source. A fork that moves fails
/// the build until its entries are reviewed at the new commit.
const FORKS: &[Fork] = &[
    Fork {
        name: "gpui-component",
        commit: GPUI_COMPONENT,
        changes: Some(
            "Eidola, Inc. changed src/lib.rs, src/root.rs and src/window_border.rs of \
             gpui-component 0.10.0 as published: an application-wide switch turns the \
             window border off (set_window_border_enabled, window_border_enabled).",
        ),
    },
    Fork {
        name: "gpui-base",
        commit: GPUI_COMPONENT,
        changes: None,
    },
    Fork {
        name: "gpui-kit-assets",
        commit: GPUI_COMPONENT,
        changes: None,
    },
    Fork {
        name: "gpui-pre-web",
        commit: GPUI_PRE,
        changes: Some(
            "Eidola, Inc. changed src/events.rs of gpui-pre-web 0.3.12 as published: a \
             tap that opens the software keyboard leaves its touchend uncancelled, so \
             WebKit on iOS shows the keyboard, and the compatibility mousedown iOS then \
             emits is cancelled.",
        ),
    },
    Fork {
        name: "pulldown-cmark",
        commit: PULLDOWN_CMARK,
        changes: Some(
            "Eidola, Inc. changed src/firstpass.rs, src/lib.rs and src/scanners.rs of \
             pulldown-cmark at upstream commit 5cc24e8: an option parses empty nested list \
             items (Options::ENABLE_EMPTY_NESTED_LISTS), and $$ lines delimit display math \
             blocks.",
        ),
    },
    Fork {
        name: "turso",
        commit: TURSO,
        changes: None,
    },
    Fork {
        name: "turso_core",
        commit: TURSO,
        changes: Some(
            "Eidola, Inc. changed Cargo.toml, functions/datetime.rs and io/clock.rs of \
             turso_core at upstream commit 20d141cb5: on wasm32-unknown-unknown the clock \
             is read through the web-time crate.",
        ),
    },
    Fork {
        name: "turso_ext",
        commit: TURSO,
        changes: None,
    },
    Fork {
        name: "turso_parser",
        commit: TURSO,
        changes: None,
    },
    Fork {
        name: "turso_sdk_kit",
        commit: TURSO,
        changes: None,
    },
    Fork {
        name: "turso_sync_engine",
        commit: TURSO,
        changes: None,
    },
    Fork {
        name: "turso_sync_sdk_kit",
        commit: TURSO,
        changes: None,
    },
];

/// A component and the texts of its licence.
struct Component {
    name: String,
    version: Option<String>,
    licence: String,
    kind: &'static str,
    texts: Vec<String>,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("mobile-licences: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(
        args.next()
            .ok_or("usage: mobile-licences ROOT [--features F]...")?,
    );
    let mut features = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--features" => features.push(args.next().ok_or("--features needs a value")?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let mut cmd = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    cmd.current_dir(&root).args([
        "metadata",
        "--locked",
        "--offline",
        "--format-version",
        "1",
        "--filter-platform",
        "wasm32-unknown-unknown",
    ]);
    for f in &features {
        cmd.args(["--features", &format!("eidola-mobile/{f}")]);
    }
    let out = cmd.output().map_err(|e| format!("cargo metadata: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo metadata: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let metadata: Value =
        serde_json::from_slice(&out.stdout).map_err(|e| format!("cargo metadata: {e}"))?;

    let packages: BTreeMap<&str, &Value> = metadata["packages"]
        .as_array()
        .ok_or("no packages")?
        .iter()
        .filter_map(|p| Some((p["id"].as_str()?, p)))
        .collect();
    // The crates: `cargo tree`'s normal edges from eidola-mobile for wasm32,
    // with the bundle's features, procedural macros left out. (The metadata's
    // resolve unifies features across the workspace, so it lists more.)
    let mut tree = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    tree.current_dir(&root).args([
        "tree",
        "--locked",
        "--offline",
        "-p",
        "eidola-mobile",
        "--target",
        "wasm32-unknown-unknown",
        "-e",
        "normal,no-proc-macro",
        "--prefix",
        "none",
        "-f",
        "{p}",
    ]);
    for f in &features {
        tree.args(["--features", f]);
    }
    let out = tree.output().map_err(|e| format!("cargo tree: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo tree: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let wanted: BTreeSet<(String, String)> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let mut words = line.split_whitespace();
            let name = words.next()?;
            let version = words.next()?.strip_prefix('v')?;
            Some((name.to_string(), version.to_string()))
        })
        .collect();
    let mut seen: BTreeMap<(String, String), &str> = BTreeMap::new();
    for (id, p) in &packages {
        let key = (
            p["name"].as_str().unwrap_or_default().to_string(),
            p["version"].as_str().unwrap_or_default().to_string(),
        );
        if wanted.contains(&key) {
            seen.entry(key).or_insert(id);
        }
    }
    if let Some(missing) = wanted.iter().find(|k| !seen.contains_key(*k)) {
        return Err(format!(
            "not in cargo metadata: {} {}",
            missing.0, missing.1
        ));
    }

    let mut components = Vec::new();
    let mut lucide_dir = None;
    let mut katex_dir = None;
    let mut app = None;
    for id in seen.values() {
        let p = packages[id];
        let name = p["name"].as_str().unwrap_or_default().to_string();
        let version = p["version"].as_str().unwrap_or_default().to_string();
        let dir = Path::new(p["manifest_path"].as_str().unwrap_or_default())
            .parent()
            .ok_or("a package without a directory")?
            .to_path_buf();
        if name == "gpui-kit-assets" {
            lucide_dir = Some(dir.clone());
        }
        if name == "ratex-katex-fonts" {
            katex_dir = Some(dir.clone());
        }
        let licence = p["license"]
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| {
                if p["license_file"].is_string() {
                    "see its licence file".into()
                } else {
                    "unknown".into()
                }
            });
        let mut texts = Vec::new();
        if p["source"].is_null() {
            // One of ours: the licence REUSE.toml declares for it, if Cargo's
            // field cannot spell it.
            match p["metadata"]["eidola"]["licence"].as_str() {
                Some(id) => {
                    texts.push(read(&root.join("LICENSES").join(format!("{id}.txt")))?);
                    if name == "eidola-mobile" {
                        app = Some(Component {
                            name: "Eidola".into(),
                            version: Some(version.clone()),
                            licence: id.to_string(),
                            kind: "app",
                            texts: texts.clone(),
                        });
                    }
                    components.push(Component {
                        name,
                        version: Some(version),
                        licence: id.to_string(),
                        kind: "crate",
                        texts,
                    });
                    continue;
                }
                None if licence.contains("GPL") => {
                    return Err(format!(
                        "{name} is {licence} without the app store permission \
                         ([package.metadata.eidola] licence), and the app links it"
                    ));
                }
                None => texts = spdx_texts(&root, &licence)?,
            }
        } else {
            let mut names: Vec<String> = fs::read_dir(&dir)
                .map_err(|e| format!("{}: {e}", dir.display()))?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_file())
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|n| is_licence_name(n))
                .collect();
            names.sort();
            if let Some(lf) = p["license_file"].as_str() {
                let path = dir.join(lf);
                let base = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default()
                    .to_string();
                if path.is_file() && !names.contains(&base) {
                    texts.push(read(&path)?);
                }
            }
            for n in &names {
                texts.push(read(&dir.join(n))?);
            }
            let source = p["source"].as_str().unwrap_or_default();
            if let Some(git) = source.strip_prefix("git+") {
                if texts.is_empty() {
                    texts = checkout_texts(&dir)?;
                }
                let (url, commit) = git
                    .split_once('#')
                    .ok_or_else(|| format!("{name}: a git source without a commit: {source}"))?;
                let url = url.split('?').next().unwrap_or(url);
                let fork = FORKS.iter().find(|f| f.name == name).ok_or_else(|| {
                    format!("{name} is built from {url}, which FORKS does not list")
                })?;
                if fork.commit != commit {
                    return Err(format!(
                        "{name} is built from {url} at {commit}, and FORKS describes {}: \
                         review what the fork changes and update its entry",
                        fork.commit
                    ));
                }
                let mut notice = format!("Built from {url} at commit {commit}.");
                if let Some(changes) = fork.changes {
                    notice = format!("{changes}\n\n{notice}");
                }
                texts.insert(0, normalize(&notice));
            }
            if texts.is_empty() {
                texts = spdx_texts(&root, &licence)?;
            }
            // MPL-2.0 section 3.2: say where the Source Code Form is.
            if licence.contains("MPL-2.0") && source.starts_with("registry+") {
                texts.insert(
                    0,
                    normalize(&format!(
                        "The source code of {name} {version}, unmodified, is published at \
                         https://crates.io/crates/{name}/{version}."
                    )),
                );
            }
        }
        if texts.is_empty() {
            return Err(format!("no licence text for {name} {version} ({licence})"));
        }
        components.push(Component {
            name,
            version: Some(version),
            licence,
            kind: "crate",
            texts,
        });
    }

    components.push(app.ok_or("eidola-mobile names no licence of its own")?);

    // The assets the bundle embeds, and the Android shell's one dependency.
    components.push(Component {
        name: "Newsreader".into(),
        version: None,
        licence: "OFL-1.1".into(),
        kind: "font",
        texts: vec![read(
            &root.join("crates/eidola-theme/assets/fonts/OFL.txt"),
        )?],
    });
    let katex = katex_dir
        .ok_or("ratex-katex-fonts is not in the graph: where are the math faces from?")?
        .join("fonts");
    components.push(Component {
        name: "KaTeX fonts".into(),
        version: None,
        licence: "OFL-1.1".into(),
        kind: "font",
        texts: vec![
            font_copyright(&katex)?,
            read(&katex.join("FONT_NOTICE.txt"))?,
            read(&katex.join("OFL.txt"))?,
        ],
    });
    let lucide =
        lucide_dir.ok_or("gpui-kit-assets is not in the graph: where are the icons from?")?;
    components.push(Component {
        name: "Lucide icons".into(),
        version: None,
        licence: "ISC".into(),
        kind: "icons",
        texts: vec![read(&lucide.join("LICENSE-LUCIDE"))?],
    });
    let tools = fs::read_to_string(root.join("mobile/android/tools.sh"))
        .map_err(|e| format!("mobile/android/tools.sh: {e}"))?;
    let kotlin = tools
        .split("kotlin-compiler-")
        .nth(1)
        .and_then(|rest| rest.split(".zip").next())
        .filter(|v| !v.is_empty() && v.chars().all(|c| c.is_ascii_digit() || c == '.'))
        .ok_or("mobile/android/tools.sh names no Kotlin release")?;
    components.push(Component {
        name: "Kotlin standard library".into(),
        version: Some(kotlin.to_string()),
        licence: "Apache-2.0".into(),
        kind: "shell",
        texts: vec![
            read(&root.join(format!(
                "crates/mobile-licences/notices/kotlin-{kotlin}-NOTICE.txt"
            )))
            .map_err(|e| {
                format!("{e}: copy kotlinc/license/NOTICE.txt from the Kotlin {kotlin} release")
            })?,
            read(&root.join("LICENSES/Apache-2.0.txt"))?,
        ],
    });

    let order = |k: &str| match k {
        "app" => 0,
        "crate" => 1,
        "font" => 2,
        "icons" => 3,
        _ => 4,
    };
    components.sort_by(|a, b| {
        (
            order(a.kind),
            a.name.to_lowercase(),
            a.version.clone().unwrap_or_default(),
        )
            .cmp(&(
                order(b.kind),
                b.name.to_lowercase(),
                b.version.clone().unwrap_or_default(),
            ))
    });
    let texts: BTreeSet<&str> = components
        .iter()
        .flat_map(|c| c.texts.iter().map(String::as_str))
        .collect();
    let index: BTreeMap<&str, usize> = texts.iter().enumerate().map(|(i, t)| (*t, i)).collect();

    // Keys written in sorted order, by hand, whatever map the JSON crate uses.
    let mut out = String::from("{\"components\":[");
    for (i, c) in components.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        // In the component's order (a fork's notice first), each once.
        let mut ids: Vec<usize> = Vec::new();
        for t in &c.texts {
            let id = index[t.as_str()];
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        out.push_str(&format!(
            "{{\"kind\":{},\"licence\":{},\"name\":{},\"texts\":[{}],\"version\":{}}}",
            json(c.kind),
            json(&c.licence),
            json(&c.name),
            ids.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
            c.version.as_deref().map_or("null".into(), json),
        ));
    }
    out.push_str("],\"texts\":[");
    out.push_str(&texts.iter().map(|t| json(t)).collect::<Vec<_>>().join(","));
    out.push_str("]}\n");
    print!("{out}");
    Ok(())
}

fn json(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_default()
}

fn is_licence_name(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    [
        "LICENSE",
        "LICENCE",
        "COPYING",
        "NOTICE",
        "COPYRIGHT",
        "UNLICENSE",
    ]
    .iter()
    .any(|stem| {
        upper == *stem
            || upper
                .strip_prefix(stem)
                .is_some_and(|rest| rest.starts_with(['-', '.', '_']))
    })
}

/// A licence text, its line endings and trailing spaces normalized.
fn read(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(normalize(&String::from_utf8_lossy(&bytes)))
}

/// A text's line endings and trailing spaces normalized.
fn normalize(text: &str) -> String {
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let lines: Vec<&str> = text.split('\n').map(str::trim_end).collect();
    format!("{}\n", lines.join("\n").trim())
}

/// The licence files at the root of the git checkout a package sits in (a
/// crate inside a repository's workspace ships none of its own).
fn checkout_texts(dir: &Path) -> Result<Vec<String>, String> {
    let Some(top) = dir.ancestors().find(|d| d.join(".cargo-ok").is_file()) else {
        return Ok(Vec::new());
    };
    let mut names: Vec<String> = fs::read_dir(top)
        .map_err(|e| format!("{}: {e}", top.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| is_licence_name(n))
        .collect();
    names.sort();
    names.iter().map(|n| read(&top.join(n))).collect()
}

/// The distinct copyright notices (OpenType name 0) of the faces in `dir`,
/// which OFL-1.1 lets a font carry in its own metadata.
fn font_copyright(dir: &Path) -> Result<String, String> {
    let mut faces: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "ttf"))
        .collect();
    faces.sort();
    let mut notices = BTreeSet::new();
    for face in &faces {
        let bytes = fs::read(face).map_err(|e| format!("{}: {e}", face.display()))?;
        notices.insert(
            name_record(&bytes, 0)
                .ok_or_else(|| format!("{}: no copyright in its name table", face.display()))?,
        );
    }
    if notices.is_empty() {
        return Err(format!("{}: no faces", dir.display()));
    }
    let lines: Vec<String> = notices.into_iter().collect();
    Ok(normalize(&lines.join("\n")))
}

/// A face's Windows Unicode name record `id`, from its `name` table.
fn name_record(font: &[u8], id: u16) -> Option<String> {
    let u16_at = |o: usize| Some(u16::from_be_bytes(font.get(o..o + 2)?.try_into().ok()?));
    let u32_at = |o: usize| Some(u32::from_be_bytes(font.get(o..o + 4)?.try_into().ok()?));
    let tables = u16_at(4)? as usize;
    let name = (0..tables)
        .map(|i| 12 + 16 * i)
        .find(|&r| font.get(r..r + 4) == Some(b"name"))?;
    let table = u32_at(name + 8)? as usize;
    let count = u16_at(table + 2)? as usize;
    let strings = table + u16_at(table + 4)? as usize;
    (0..count).map(|i| table + 6 + 12 * i).find_map(|r| {
        if u16_at(r)? != 3 || u16_at(r + 6)? != id {
            return None;
        }
        let len = u16_at(r + 8)? as usize;
        let at = strings + u16_at(r + 10)? as usize;
        let units: Vec<u16> = font
            .get(at..at + len)?
            .chunks_exact(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
            .collect();
        String::from_utf16(&units).ok()
    })
}

/// The SPDX texts of the licences an expression names, from `ROOT/LICENSES/`.
fn spdx_texts(root: &Path, expression: &str) -> Result<Vec<String>, String> {
    let ids: BTreeSet<&str> = expression
        .split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '+' | '-')))
        .filter(|t| !t.is_empty() && !matches!(*t, "OR" | "AND" | "WITH"))
        .collect();
    let mut texts = Vec::new();
    for id in ids {
        let file = match id {
            "MIT" => "MIT.txt",
            "Apache-2.0" => "Apache-2.0.txt",
            "GPL-3.0-only" => "GPL-3.0-only.txt",
            _ => continue,
        };
        let path = root.join("LICENSES").join(file);
        if path.is_file() {
            texts.push(read(&path)?);
        }
    }
    Ok(texts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    /// The crates whose Cargo.toml names a licence of ours, and REUSE.toml's
    /// annotation for that licence, list the same crates.
    #[test]
    fn manifests_and_reuse_agree() {
        let out = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
            .current_dir(root())
            .args([
                "metadata",
                "--no-deps",
                "--offline",
                "--format-version",
                "1",
            ])
            .output()
            .unwrap();
        assert!(out.status.success());
        let metadata: Value = serde_json::from_slice(&out.stdout).unwrap();
        let mut declared: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for p in metadata["packages"].as_array().unwrap() {
            if let Some(id) = p["metadata"]["eidola"]["licence"].as_str() {
                let dir = Path::new(p["manifest_path"].as_str().unwrap())
                    .parent()
                    .unwrap()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned();
                declared
                    .entry(id.to_string())
                    .or_default()
                    .insert(format!("crates/{dir}/**"));
            }
        }
        assert!(!declared.is_empty());
        let reuse = fs::read_to_string(root().join("REUSE.toml")).unwrap();
        for (id, crates) in &declared {
            let annotation = reuse
                .split("[[annotations]]")
                .find(|a| a.contains(&format!("SPDX-License-Identifier = \"{id}\"")))
                .unwrap_or_else(|| panic!("REUSE.toml declares nothing {id}"));
            let paths: BTreeSet<String> = annotation
                .split("precedence")
                .next()
                .unwrap()
                .split('"')
                .skip(1)
                .step_by(2)
                .map(str::to_string)
                .collect();
            assert_eq!(&paths, crates, "{id}");
            assert!(root().join("LICENSES").join(format!("{id}.txt")).is_file());
        }
    }

    /// The app store permission's file is the GPL's text, then the permission.
    #[test]
    fn app_store_licence_is_the_gpl_then_the_permission() {
        let gpl = read(&root().join("LICENSES/GPL-3.0-only.txt")).unwrap();
        let combined = read(
            &root().join("LICENSES/LicenseRef-Eidola-GPL-3.0-only-with-app-store-permission.txt"),
        )
        .unwrap();
        let permission = combined.strip_prefix(gpl.trim_end()).unwrap();
        assert!(permission.trim_start().starts_with(
            "Additional permission under GNU GPL version 3 section 7: distribution through app stores"
        ));
    }

    #[test]
    fn a_face_names_its_copyright() {
        let face =
            fs::read(root().join("crates/eidola-theme/assets/fonts/Newsreader16pt-Regular.ttf"))
                .unwrap();
        assert!(name_record(&face, 0).unwrap().contains("Newsreader"));
    }
}
