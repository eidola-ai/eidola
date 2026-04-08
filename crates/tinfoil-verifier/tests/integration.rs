//! Integration tests against live Tinfoil infrastructure.
//!
//! Run with: `cargo test --package tinfoil-verifier -- --ignored`

const LIVE_ENCLAVE: &str = "inference.tinfoil.sh";
const LIVE_REPO: &str = "tinfoilsh/confidential-model-router";

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
    let bundle = tinfoil_verifier::bundle::fetch_bundle(None, LIVE_ENCLAVE, LIVE_REPO)
        .await
        .expect("failed to fetch attestation bundle");

    eprintln!("Domain: {}", bundle.domain);
    assert_eq!(
        bundle.domain, LIVE_ENCLAVE,
        "ATC POST should bind the bundle to the requested enclave"
    );

    // Decode VCEK and report
    let vcek_der = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &bundle.vcek)
        .expect("failed to decode VCEK");

    let report_bytes =
        tinfoil_verifier::bundle::decode_report_gzipped(&bundle.enclave_attestation_report.body)
            .expect("failed to decode report");

    eprintln!("Report size: {} bytes", report_bytes.len());

    // Verify chain + report signature
    let report = tinfoil_verifier::sevsnp::verify_attestation(&vcek_der, &report_bytes, None, None)
        .expect("attestation verification failed");

    // Verify TCB policy (defaults: AMD recommended floor + rollback check)
    let policy = tinfoil_verifier::SevSnpTcbPolicy::amd_recommended();
    let (_observation, result) = policy.evaluate(&report);
    result.expect("TCB policy verification failed");

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
    let bundle = tinfoil_verifier::bundle::fetch_bundle(None, LIVE_ENCLAVE, LIVE_REPO)
        .await
        .expect("failed to fetch attestation bundle");

    let vcek_der = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &bundle.vcek)
        .expect("failed to decode VCEK");
    let report_bytes =
        tinfoil_verifier::bundle::decode_report_gzipped(&bundle.enclave_attestation_report.body)
            .expect("failed to decode report");
    let report = tinfoil_verifier::sevsnp::verify_attestation(&vcek_der, &report_bytes, None, None)
        .expect("attestation verification failed");
    let measurement = hex::encode(report.measurement);

    eprintln!("Using measurement: {measurement}");

    // The bootstrap path here is SEV-SNP (live ATC bundle), so populate the
    // SNP field with the observed measurement and fill the TDX side with
    // dummy values that will never match — the verifier picks the field that
    // matches the observed platform.
    let allowed = vec![tinfoil_verifier::EnclaveMeasurement {
        snp_measurement: measurement.clone(),
        tdx_measurement: tinfoil_verifier::TdxMeasurement {
            rtmr1: "0".repeat(96),
            rtmr2: "0".repeat(96),
        },
    }];

    // Build the attesting client. Verification happens lazily on the first
    // real request.
    let client = tinfoil_verifier::attesting_client(tinfoil_verifier::AttestingClientConfig {
        allowed_measurements: &allowed,
        inference_base_url: "https://inference.tinfoil.sh/v1",
        atc_url: None,
        enclave_repo: Some(LIVE_REPO),
        trusted_ark_der: None,
        trusted_ask_der: None,
        tdx_advisory_allowlist: None,
        tdx_observer: None,
        snp_min_tcb: None,
        snp_observer: None,
    })
    .await
    .expect("attesting_client failed");

    // Make a request through the attesting client; this is what triggers
    // per-handshake attestation.
    let resp = client
        .get("https://inference.tinfoil.sh/v1/models")
        .send()
        .await
        .expect("request through attesting client failed");

    eprintln!("Response status: {}", resp.status());
    assert!(resp.status().is_success() || resp.status().as_u16() == 401);
}
