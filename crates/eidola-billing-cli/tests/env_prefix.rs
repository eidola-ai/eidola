//! The billing binary reads its environment under `EIDOLA_ENV_PREFIX`.

use std::process::{Command, Output};

fn run(vars: &[(&str, &str)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_eidola-billing"));
    command.env_clear().args([
        "inspect",
        "--account",
        "11111111-1111-4111-8111-111111111111",
    ]);
    for (name, value) in vars {
        command.env(name, value);
    }
    command.output().expect("run eidola-billing")
}

#[test]
fn an_invalid_prefix_refuses_to_start() {
    let output = run(&[("EIDOLA_ENV_PREFIX", "WORKER")]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("EIDOLA_ENV_PREFIX must be"),
        "expected the prefix to be refused:\n{stderr}"
    );
}

#[test]
fn a_prefixed_process_reads_no_bare_name() {
    let output = run(&[
        ("EIDOLA_ENV_PREFIX", "BILLING_WORKER_"),
        ("DATABASE_URL", "postgres://bare@localhost/eidola"),
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("BILLING_WORKER_DATABASE_URL is required"),
        "expected the bare DATABASE_URL to be ignored:\n{stderr}"
    );
}
