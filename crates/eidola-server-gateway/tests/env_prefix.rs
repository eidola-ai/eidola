//! The gateway reads its environment under `EIDOLA_ENV_PREFIX`, end to end.

use std::process::{Command, Output};

fn run(vars: &[(&str, &str)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_eidola-server-gateway"));
    command.env_clear();
    for (name, value) in vars {
        command.env(name, value);
    }
    command.output().expect("run eidola-server-gateway")
}

#[test]
fn an_invalid_prefix_refuses_to_start() {
    let output = run(&[("EIDOLA_ENV_PREFIX", "gateway")]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("EIDOLA_ENV_PREFIX must be"),
        "expected the prefix to be refused:\n{stderr}"
    );
}

#[test]
fn a_prefixed_gateway_reads_no_bare_name() {
    let output = run(&[
        ("EIDOLA_ENV_PREFIX", "GATEWAY_"),
        ("DATABASE_URL", "postgres://bare@localhost/eidola"),
        ("CREDENTIAL_MASTER_KEY", &"00".repeat(32)),
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!output.status.success());
    assert!(
        stdout.contains(r#"Environment variable prefix: "GATEWAY_""#),
        "expected the prefix to be logged:\n{stdout}"
    );
    assert!(
        stdout.contains("GATEWAY_DATABASE_URL environment variable is required"),
        "expected the bare DATABASE_URL to be ignored:\n{stdout}"
    );
}
