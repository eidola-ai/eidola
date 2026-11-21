//! The receiver and worker must start with an HTTPS OTLP endpoint and no
//! system trust store.
//!
//! The image runs `FROM scratch` — there is no CA bundle anywhere in it — and a
//! deployment points `OTEL_EXPORTER_OTLP_ENDPOINT` at an HTTPS collector. Left
//! to their default client, the OTLP exporters verify against the platform
//! store, cannot build a client over an empty one, and panic while telemetry
//! starts, before configuration is even read.
//!
//! The image's missing store is reproduced on any host by pointing
//! `SSL_CERT_FILE`/`SSL_CERT_DIR` at nothing: when either is set,
//! `rustls-native-certs` loads only from them. (macOS's platform verifier does
//! not read them, so there the test passes without exercising the empty store.)

use std::process::Command;

fn start(role: &str) {
    let dir = std::env::temp_dir().join(format!(
        "eidola-billing-otlp-startup-{role}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let empty_bundle = dir.join("empty.pem");
    std::fs::write(&empty_bundle, b"").expect("write empty CA bundle");

    let output = Command::new(env!("CARGO_BIN_EXE_eidola-billing"))
        .arg(role)
        .env_clear()
        .env("SSL_CERT_FILE", &empty_bundle)
        .env("SSL_CERT_DIR", &dir)
        // A closed local port: the exporters are built, and whatever they flush
        // on the way out is refused at once rather than leaving the machine.
        .env("OTEL_EXPORTER_OTLP_ENDPOINT", "https://127.0.0.1:9/otlp")
        .output()
        .expect("run eidola-billing");
    let _ = std::fs::remove_dir_all(&dir);

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panicked"),
        "{role}: telemetry init panicked without a system trust store:\n{stderr}"
    );
    // Telemetry came up, and startup went on to stop at the first missing
    // piece of configuration — the expected exit for an empty environment —
    // which it logged before flushing telemetry and exiting.
    assert!(!output.status.success());
    assert!(
        stderr.contains("DATABASE_URL is required"),
        "{role}: expected startup to reach configuration loading:\n{stderr}"
    );
}

#[test]
fn the_receiver_starts_with_an_https_otlp_endpoint_and_no_system_trust_store() {
    start("serve-webhooks");
}

#[test]
fn the_worker_starts_with_an_https_otlp_endpoint_and_no_system_trust_store() {
    start("work");
}
