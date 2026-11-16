//! The staged-experts path (`deepseek::staged`, dev-only) stays out of every pinned
//! deployment, three ways: the node's release build resolves this crate without the
//! `dev-staged-experts` feature, so the module is not in the measured binary; the
//! token every constructor takes is built nowhere but tests and development examples;
//! and nothing the node's configuration is read from can name it. Runs without the
//! feature, on any machine.

use std::path::{Path, PathBuf};
use std::process::Command;

const FEATURE: &str = "dev-staged-experts";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Every file under `dir` whose name ends with `suffix`.
fn files(dir: &Path, suffix: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            files(&p, suffix, out);
        } else if p.to_string_lossy().ends_with(suffix) {
            out.push(p);
        }
    }
}

/// The node's image builds the node with `--features cuda` and nothing else.
#[test]
fn the_node_image_builds_with_the_cuda_feature_alone() {
    let text = std::fs::read_to_string(root().join("oci/eidola-server-engine/Containerfile"))
        .expect("the node's Containerfile");
    let builds: Vec<&str> = text
        .lines()
        .filter(|l| l.contains("cargo build") && l.contains("-p eidola-server-engine"))
        .collect();
    assert_eq!(builds.len(), 2, "the dependency build and the node's build");
    for line in builds {
        assert!(line.contains("--features cuda"), "{line}");
        let after = line.split("--features cuda").nth(1).unwrap();
        assert!(
            after.trim_start().is_empty()
                || after.trim_start().starts_with('\\')
                || after.trim_start().starts_with("&&"),
            "a feature list longer than `cuda`: {line}"
        );
        assert!(
            !line.contains("--all-features") && !line.contains(FEATURE),
            "{line}"
        );
    }
}

/// No manifest but this crate's own names the feature, so no dependency edge enables it.
#[test]
fn only_this_crate_names_the_feature() {
    let mut manifests = Vec::new();
    files(&root().join("crates"), "Cargo.toml", &mut manifests);
    manifests.push(root().join("Cargo.toml"));
    assert!(manifests.len() > 10, "the workspace's manifests");
    let own = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let mut seen = 0;
    for m in &manifests {
        let text = std::fs::read_to_string(m).unwrap();
        if text.contains(FEATURE) {
            assert_eq!(
                m.canonicalize().unwrap(),
                own.canonicalize().unwrap(),
                "{} names {FEATURE}",
                m.display()
            );
            seen += 1;
        }
    }
    assert_eq!(seen, 1, "this crate defines it");
}

/// Cargo's own resolution of the release build: the node with `cuda` reaches this
/// crate without the feature; the same query with the feature asked for shows it, so
/// the check reads what it means to.
#[test]
fn the_release_resolution_excludes_the_feature() {
    let tree = |features: &str| -> String {
        let out = Command::new(env!("CARGO"))
            .current_dir(root())
            .args([
                "tree",
                "--offline",
                "--locked",
                "-p",
                "eidola-server-engine",
                "--features",
                features,
                "-e",
                "features",
                "-i",
                "eidola-engine-cuda",
                "--target",
                "x86_64-unknown-linux-gnu",
            ])
            .output()
            .expect("cargo tree");
        assert!(
            out.status.success(),
            "cargo tree: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    };
    let release = tree("cuda");
    assert!(release.contains("eidola-engine-cuda"), "{release}");
    assert!(!release.contains(FEATURE), "{release}");
    let asked = tree(&format!("cuda,eidola-engine-cuda/{FEATURE}"));
    assert!(asked.contains(FEATURE), "{asked}");
}

/// The token is built only in tests and examples (and the module's own unit tests), and
/// nothing the node reads its configuration through, nor any deployment file, names the
/// module or the feature.
#[test]
fn nothing_a_node_runs_can_reach_the_module() {
    let token = "DevOnly::for_tests_and_dev_tools";
    let mut sources = Vec::new();
    files(&root().join("crates"), ".rs", &mut sources);
    assert!(sources.len() > 100, "the workspace's sources");
    let staged = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/deepseek/staged.rs");
    let mut uses = 0;
    for f in &sources {
        let text = std::fs::read_to_string(f).unwrap();
        if !text.contains(token) {
            continue;
        }
        let path = f.to_string_lossy().replace('\\', "/");
        let dev_only = path.contains("/tests/") || path.contains("/examples/");
        let own = f.canonicalize().unwrap() == staged.canonicalize().unwrap();
        assert!(dev_only || own, "{path} builds the token");
        uses += usize::from(dev_only);
    }
    assert!(uses > 0, "the tests build it");

    let mut node = Vec::new();
    for dir in [
        "crates/eidola-server-engine/src",
        "crates/eidola-common/src",
        "deploy",
        "oci",
    ] {
        files(&root().join(dir), "", &mut node);
    }
    assert!(node.len() > 20, "the node's sources and deployment files");
    for f in &node {
        let Ok(text) = std::fs::read_to_string(f) else {
            continue;
        };
        for word in [
            FEATURE,
            "dev_staged_experts",
            "deepseek::staged",
            "DevOnly",
            "HostExperts",
        ] {
            assert!(!text.contains(word), "{} names {word}", f.display());
        }
    }
}
