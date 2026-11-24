//! Writes the mobile app bundle's `licences.json`: every crate the wasm build
//! links, every asset it embeds, and the Android shell's one dependency, each
//! with its licence and the licence texts, for the app's licences screen.
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
//! that ships none gets the SPDX texts of the licences it declares, from
//! `ROOT/LICENSES/`, as this repository's own crates do. Then the assets: the
//! Newsreader faces (OFL-1.1), the Lucide icons (ISC), and the Kotlin
//! standard library R8 shrinks into the Android APK (Apache-2.0).
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
            texts = spdx_texts(&root, &licence)?;
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
            if texts.is_empty() {
                texts = spdx_texts(&root, &licence)?;
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
        texts: vec![read(&root.join("LICENSES/Apache-2.0.txt"))?],
    });

    let order = |k: &str| match k {
        "crate" => 0,
        "font" => 1,
        "icons" => 2,
        _ => 3,
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
        let ids: BTreeSet<usize> = c.texts.iter().map(|t| index[t.as_str()]).collect();
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
    let text = String::from_utf8_lossy(&bytes)
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let lines: Vec<&str> = text.split('\n').map(str::trim_end).collect();
    Ok(format!("{}\n", lines.join("\n").trim()))
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
