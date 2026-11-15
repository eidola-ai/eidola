//! The DeepSeek scalar kernels' bodies against the Rust reference, on the
//! host: `oracle/ds_kernels_host.cc` (which includes `csrc/ds_ops.cuh`) is
//! compiled with the system C++ compiler and run over
//! `tests/data/ds_ops.vectors` and `tests/data/ds_gemm_ref.vectors`.
//!
//! The compiler is `$CXX`, else `c++`. Without one the tests pass after saying
//! so on stderr: the kernel build itself needs none, and the vectors are held
//! current by `eidola-engine-cuda`'s `tests/ds_ops.rs` and
//! `tests/ds_gemm_ref.rs`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn crate_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Compile the host check at `opt`, or `None` when there is no compiler.
fn build(opt: &str) -> Option<PathBuf> {
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    if Command::new(&cxx).arg("--version").output().is_err() {
        eprintln!("ds_kernels_host: no C++ compiler ({cxx}); the host check did not run");
        return None;
    }
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("ds_kernels_host{opt}"));
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
        .arg(crate_dir().join("oracle/ds_kernels_host.cc"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("run the C++ compiler");
    assert!(
        status.success(),
        "{cxx} failed on oracle/ds_kernels_host.cc"
    );
    Some(out)
}

fn run(opt: &str) {
    let Some(binary) = build(opt) else { return };
    let output = Command::new(&binary)
        .args(
            ["ds_ops.vectors", "ds_gemm_ref.vectors"]
                .iter()
                .map(|f| crate_dir().join("tests/data").join(f)),
        )
        .output()
        .expect("run the host check");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Both files were read: two sweeps of each of 23 kernels, and their cases.
    let counts: Vec<u64> = stdout
        .split(|c: char| !c.is_ascii_digit())
        .filter_map(|t| t.parse().ok())
        .collect();
    assert!(
        counts.len() >= 3 && counts[0] == 46 && counts[1] > 10_000 && counts[2] == 0,
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
