//! Integration tests against live Tinfoil infrastructure.
//!
//! Run with: `cargo test --package tinfoil-verifier -- --ignored`

/// Full verification flow against the live Tinfoil attestation endpoint.
///
/// Fetches a real attestation bundle from the ATC service, verifies the VCEK
/// chain, validates the report signature and TCB policy, and checks the TLS
/// fingerprint binding. Skips the measurement check (no hardcoded values) but
/// exercises the entire verification pipeline.
#[tokio::test]
#[ignore = "requires network access to atc.tinfoil.sh"]
async fn live_attestation_verification() {
    let _ =
        rustls::crypto::CryptoProvider::install_default(rustls::crypto::ring::default_provider());

    // Fetch the attestation bundle
    let bundle = tinfoil_verifier::bundle::fetch_bundle(None)
        .await
        .expect("failed to fetch attestation bundle");

    eprintln!("Domain: {}", bundle.domain);

    // Decode VCEK and report
    let vcek_der = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &bundle.vcek)
        .expect("failed to decode VCEK");

    let report_bytes =
        tinfoil_verifier::bundle::decode_report(&bundle.enclave_attestation_report.body)
            .expect("failed to decode report");

    eprintln!("Report size: {} bytes", report_bytes.len());

    // Verify chain + report signature
    let report = tinfoil_verifier::sevsnp::verify_attestation(&vcek_der, &report_bytes)
        .expect("attestation verification failed");

    // Verify TCB policy
    tinfoil_verifier::sevsnp::verify_tcb_policy(&report).expect("TCB policy verification failed");

    let measurement = hex::encode(report.measurement);
    eprintln!("Measurement: {measurement}");

    // Verify enclave cert binding
    let tls_fingerprint = &report.report_data[..32];
    tinfoil_verifier::sevsnp::verify_enclave_cert_binding(&bundle.enclave_cert, tls_fingerprint)
        .expect("enclave cert binding verification failed");

    eprintln!("TLS fingerprint: {}", hex::encode(tls_fingerprint));
}

/// Test the full attesting client flow: bootstrap via ATC, then make a request
/// through the attesting client to verify per-connection attestation works.
#[tokio::test]
#[ignore = "requires network access to inference.tinfoil.sh"]
async fn live_attesting_client() {
    let _ =
        rustls::crypto::CryptoProvider::install_default(rustls::crypto::ring::default_provider());

    // First, get the current measurement from ATC (we don't hardcode it in tests)
    let bundle = tinfoil_verifier::bundle::fetch_bundle(None)
        .await
        .expect("failed to fetch attestation bundle");

    let vcek_der = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &bundle.vcek)
        .expect("failed to decode VCEK");
    let report_bytes =
        tinfoil_verifier::bundle::decode_report(&bundle.enclave_attestation_report.body)
            .expect("failed to decode report");
    let report = tinfoil_verifier::sevsnp::verify_attestation(&vcek_der, &report_bytes)
        .expect("attestation verification failed");
    let measurement = hex::encode(report.measurement);

    eprintln!("Using measurement: {measurement}");

    // Build the attesting client
    let (client, verification) =
        tinfoil_verifier::attesting_client(tinfoil_verifier::AttestingClientConfig {
            allowed_measurements: &[measurement.as_str()],
            inference_base_url: "https://inference.tinfoil.sh/v1",
            atc_url: None,
        })
        .await
        .expect("attesting_client failed");

    eprintln!(
        "Verification: measurement={}, fingerprint={}",
        verification.measurement, verification.tls_fingerprint
    );

    // Make a request through the attesting client
    let resp = client
        .get("https://inference.tinfoil.sh/v1/models")
        .send()
        .await
        .expect("request through attesting client failed");

    eprintln!("Response status: {}", resp.status());
    assert!(resp.status().is_success() || resp.status().as_u16() == 401);
}
