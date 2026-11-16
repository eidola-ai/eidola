//! The DeepSeek tensor-core GEMMs' staging, fragment, band and block
//! arithmetic on the host: `oracle/ds_gemm_tc_host.cc` (which includes
//! `csrc/ds_gemm_tc.cuh` and the scalar reference bodies of
//! `csrc/ds_ops.cuh`) is compiled with the system C++ compiler and run. Every
//! banded output must equal the reference's, under an emulated MMA that
//! rounds the exact sum once and under one that adds in a random order with
//! a rounding after every addition.
//!
//! The compiler is `$CXX`, else `c++`. Without one the tests pass after saying
//! so on stderr: the kernel build itself needs none.

use std::path::{Path, PathBuf};
use std::process::Command;

fn crate_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn build(opt: &str) -> Option<PathBuf> {
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    if Command::new(&cxx).arg("--version").output().is_err() {
        eprintln!("ds_gemm_tc_host: no C++ compiler ({cxx}); the host check did not run");
        return None;
    }
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("ds_gemm_tc_host{opt}"));
    let status = Command::new(&cxx)
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
        .arg(crate_dir().join("oracle/ds_gemm_tc_host.cc"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("run the C++ compiler");
    assert!(
        status.success(),
        "{cxx} failed on oracle/ds_gemm_tc_host.cc"
    );
    Some(out)
}

fn run(opt: &str) {
    let Some(binary) = build(opt) else { return };
    let output = Command::new(&binary).output().expect("run the host check");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Both products ran over many outputs, and none of the banded ones differed.
    for product in ["lin8", "lin4"] {
        let line = stdout
            .lines()
            .find(|l| l.starts_with(&format!("{product}:")))
            .unwrap_or_else(|| panic!("{stdout}"));
        let counts: Vec<u64> = line
            .split(|c: char| !c.is_ascii_digit())
            .filter_map(|t| t.parse().ok())
            .collect();
        // `lin8: <outputs> outputs, banded differing <n>, ...`: the 8 of the
        // name is the first number.
        assert!(
            counts.len() >= 3 && counts[1] > 100_000 && counts[2] == 0,
            "{stdout}"
        );
    }
}

#[test]
fn banded_forms_equal_the_reference_bodies() {
    run("-O2");
}

/// The same without optimisation: the result may not depend on what the
/// compiler folds or keeps in registers.
#[test]
fn banded_forms_equal_the_reference_bodies_unoptimised() {
    run("-O0");
}
