//! The DeepSeek numerics header against the Rust reference's vectors, on the
//! host: `oracle/ds_numeric_host.cc` (which includes `csrc/ds_numeric.cuh` and
//! `csrc/ds_abi.cuh`) is compiled with the system C++ compiler and run over
//! `tests/data/ds_numeric.vectors`.
//!
//! The compiler is `$CXX`, else `c++`. Without one the tests pass after saying
//! so on stderr: the kernel build itself needs none, and the vectors are held
//! current by `eidola-engine-model`'s `tests/deepseek_numeric_vectors.rs`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn crate_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Compile the host check at `opt`, or `None` when there is no compiler.
fn build(opt: &str) -> Option<PathBuf> {
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    if Command::new(&cxx).arg("--version").output().is_err() {
        eprintln!("ds_numeric_host: no C++ compiler ({cxx}); the host check did not run");
        return None;
    }
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("ds_numeric_host{opt}"));
    let status = Command::new(&cxx)
        // No contraction, as the device build: the header's arithmetic is the
        // reference's only operation by operation.
        .args([
            "-std=c++17",
            opt,
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .arg("-pthread")
        .arg("-I")
        .arg(crate_dir().join("csrc"))
        .arg(crate_dir().join("oracle/ds_numeric_host.cc"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("run the C++ compiler");
    assert!(
        status.success(),
        "{cxx} failed on oracle/ds_numeric_host.cc"
    );
    Some(out)
}

fn run(opt: &str, files: &[&str]) {
    let Some(binary) = build(opt) else { return };
    let output = Command::new(&binary)
        .args(files.iter().map(|f| crate_dir().join("tests/data").join(f)))
        .output()
        .expect("run the host check");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // The vector file is not empty, and every kind of record was read.
    let counts: Vec<u64> = stdout
        .split(|c: char| !c.is_ascii_digit())
        .filter_map(|t| t.parse().ok())
        .collect();
    assert!(
        counts.len() >= 5 && counts[0] == 5 && counts[1] > 4000 && counts[2] >= 29,
        "{stdout}"
    );
    if files.len() > 1 {
        assert_eq!(counts[3], 8, "{stdout}");
    }
}

#[test]
fn header_matches_the_reference_vectors() {
    run("-O2", &["ds_numeric.vectors"]);
}

/// The same without optimisation: the result may not depend on what the
/// compiler folds or keeps in registers.
#[test]
fn header_matches_the_reference_vectors_unoptimised() {
    run("-O0", &["ds_numeric.vectors"]);
}

/// Every f32 through each one-argument function, against the reference's
/// digests. About two CPU-minutes.
#[test]
#[ignore = "2^32 inputs per function"]
fn header_matches_the_reference_on_every_f32() {
    run("-O2", &["ds_numeric.vectors", "ds_numeric.exhaustive"]);
}
