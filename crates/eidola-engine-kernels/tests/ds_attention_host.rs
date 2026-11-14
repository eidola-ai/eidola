//! The DeepSeek attention, index and selection kernel bodies against the Rust
//! reference's vectors, on the host: `oracle/ds_attention_host.cc` (which
//! includes `csrc/ds_attention.cuh` and `csrc/ds_index.cuh`) is compiled with
//! the system C++ compiler and run over `tests/data/ds_attention.vectors`.
//!
//! The compiler is `$CXX`, else `c++`. Without one the tests pass after saying
//! so on stderr: the vectors are held current by `eidola-engine-model`'s
//! `tests/deepseek_attention_vectors.rs`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn crate_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Compile the host check at `opt`, or `None` when there is no compiler.
fn build(opt: &str) -> Option<PathBuf> {
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    if Command::new(&cxx).arg("--version").output().is_err() {
        eprintln!("ds_attention_host: no C++ compiler ({cxx}); the host check did not run");
        return None;
    }
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("ds_attention_host{opt}"));
    let status = Command::new(&cxx)
        // No contraction, as the device build.
        .args([
            "-std=c++17",
            opt,
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .arg("-I")
        .arg(crate_dir().join("csrc"))
        .arg(crate_dir().join("oracle/ds_attention_host.cc"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("run the C++ compiler");
    assert!(
        status.success(),
        "{cxx} failed on oracle/ds_attention_host.cc"
    );
    Some(out)
}

fn run(opt: &str) {
    let Some(binary) = build(opt) else { return };
    let output = Command::new(&binary)
        .arg("vectors")
        .arg(crate_dir().join("tests/data/ds_attention.vectors"))
        .output()
        .expect("run the host check");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Every kind of record was read, and none failed.
    let counts: Vec<u64> = stdout
        .split(|c: char| !c.is_ascii_digit())
        .filter_map(|t| t.parse().ok())
        .collect();
    assert!(
        counts.len() == 5 && counts[..4].iter().all(|n| *n >= 50) && counts[4] == 0,
        "{stdout}"
    );
}

#[test]
fn bodies_match_the_reference_vectors() {
    run("-O2");
}

/// The same without optimisation: the result may not depend on what the
/// compiler folds or keeps in registers.
#[test]
fn bodies_match_the_reference_vectors_unoptimised() {
    run("-O0");
}
