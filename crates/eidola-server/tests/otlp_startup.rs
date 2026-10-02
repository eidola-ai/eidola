//! The server must start with an HTTPS OTLP endpoint and no system trust store.
//!
//! Production runs `FROM scratch` — there is no CA bundle anywhere in the image
//! — and `tinfoil-config.yml` points `OTEL_EXPORTER_OTLP_ENDPOINT` at an HTTPS
//! collector. Left to their default client, the OTLP exporters verify against
//! the platform store, cannot build a client over an empty one, and panic
//! inside `telemetry::init()` before configuration is even read.
//!
//! The image's missing store is reproduced on any host by pointing
//! `SSL_CERT_FILE`/`SSL_CERT_DIR` at nothing: when either is set,
//! `rustls-native-certs` loads only from them. (macOS's platform verifier does
//! not read them, so there the test passes without exercising the empty store.)

use std::process::Command;

#[test]
fn starts_with_an_https_otlp_endpoint_and_no_system_trust_store() {
    let dir = std::env::temp_dir().join(format!("eidola-otlp-startup-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let empty_bundle = dir.join("empty.pem");
    std::fs::write(&empty_bundle, b"").expect("write empty CA bundle");

    let output = Command::new(env!("CARGO_BIN_EXE_eidola-server"))
        .env_clear()
        .env("SSL_CERT_FILE", &empty_bundle)
        .env("SSL_CERT_DIR", &dir)
        // A closed local port: the exporters are built, and whatever they flush
        // on the way out is refused at once rather than leaving the machine.
        .env("OTEL_EXPORTER_OTLP_ENDPOINT", "https://127.0.0.1:9/otlp")
        .output()
        .expect("run eidola-server");
    let _ = std::fs::remove_dir_all(&dir);

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stderr.contains("panicked"),
        "telemetry init panicked without a system trust store:\n{stderr}"
    );
    // Telemetry came up, and startup went on to stop at the first missing
    // piece of configuration — the expected exit for an empty environment.
    assert!(
        stdout.contains("Configuration error") || stderr.contains("Configuration error"),
        "expected startup to reach configuration loading:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}
