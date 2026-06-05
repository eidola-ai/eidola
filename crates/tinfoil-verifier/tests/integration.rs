//! Integration tests against live Tinfoil infrastructure.
//!
//! Run with: `cargo test --package tinfoil-verifier -- --ignored`

const LIVE_ENCLAVE: &str = "inference.tinfoil.sh";
const LIVE_REPO: &str = "tinfoilsh/confidential-model-router";

/// A real fresh attestation document captured from
/// `inference.tinfoil.sh/.well-known/tinfoil-attestation?nonce=<NONCE>`.
/// Used as an offline regression fixture for the new nonce-bound format so the
/// document parser, the `REPORT_DATA` recomputation, the `tls_key_fp` ↔ cert
/// binding, and the ECDSA (P-384 / SHA-256) document-signature verification are
/// all exercised without network access.
const FIXTURE_DOC: &[u8] = include_bytes!("live_attest.json");
const FIXTURE_NONCE: &str = "478a90b1c60e55b4aeef3e7c51fcec0174cba476f6cc5f5f7887f2ec1202ff64";

/// Full connector path against a locally-running `tinfoil-shim-mock`.
///
/// Self-skips unless `MOCK_URL` (e.g. `https://127.0.0.1:18443/v1`) and
/// `MOCK_CERT_DIR` (the shim's `CERT_DIR`, holding `ark.pem`, `ask.pem`,
/// `tls-ca.pem`) are set, so it never runs — or flakes — in CI, but gives a
/// one-command end-to-end check of the nonce flow:
///
/// ```sh
/// LISTEN_ADDR=127.0.0.1:18443 CERT_DIR=/tmp/mockcerts UPSTREAM_URL=http://127.0.0.1:1 \
///   cargo run -p tinfoil-shim-mock &
/// MOCK_URL=https://127.0.0.1:18443/v1 MOCK_CERT_DIR=/tmp/mockcerts \
///   cargo test -p tinfoil-verifier --test integration mock_attesting_client_e2e
/// ```
#[tokio::test]
async fn mock_attesting_client_e2e() {
    let (Ok(base_url), Ok(cert_dir)) = (std::env::var("MOCK_URL"), std::env::var("MOCK_CERT_DIR"))
    else {
        eprintln!("skipping: set MOCK_URL and MOCK_CERT_DIR to run");
        return;
    };

    let _ =
        rustls::crypto::CryptoProvider::install_default(rustls::crypto::ring::default_provider());

    let read_der = |name: &str| {
        let pem = std::fs::read_to_string(format!("{cert_dir}/{name}"))
            .unwrap_or_else(|e| panic!("read {name}: {e}"));
        pem::parse(&pem).expect("parse pem").into_contents()
    };
    let ark_der = read_der("ark.pem");
    let ask_der = read_der("ask.pem");

    // Trust the mock's TLS-CA so the TLS handshake itself succeeds.
    let mut tls_roots = rustls::RootCertStore::empty();
    let ca = read_der("tls-ca.pem");
    tls_roots
        .add(rustls::pki_types::CertificateDer::from(ca))
        .expect("add tls-ca");

    // The mock advertises the all-zero default measurement.
    let allowed = vec![tinfoil_verifier::EnclaveMeasurement {
        snp_measurement: "00".repeat(48),
        tdx_measurement: tinfoil_verifier::TdxMeasurement {
            rtmr1: "0".repeat(96),
            rtmr2: "0".repeat(96),
        },
    }];

    let client = tinfoil_verifier::attesting_client(tinfoil_verifier::AttestingClientConfig {
        allowed_measurements: &allowed,
        inference_base_url: &base_url,
        atc_url: None,
        enclave_repo: None, // mock document self-carries the VCEK; no ATC needed
        trusted_ark_der: Some(&ark_der),
        trusted_ask_der: Some(&ask_der),
        tdx_advisory_allowlist: None,
        tdx_observer: None,
        snp_min_tcb: None,
        snp_observer: None,
        attestation_observer: None,
        tls_roots,
        // The mock binds the TLS session, so exercise the strict path.
        require_channel_binding: true,
    })
    .await
    .expect("attesting_client failed");

    // A successful send means the per-handshake attestation passed (the mock
    // proxies to a dead upstream, so the *response* is a 502 — but reaching it
    // proves the connection was attested).
    let resp = client.get(format!("{base_url}/models")).send().await;
    assert!(
        resp.is_ok(),
        "attestation should pass against the mock: {resp:?}"
    );
    eprintln!("attested OK, upstream status: {}", resp.unwrap().status());
}

/// Offline end-to-end check of the fresh nonce-bound document format against a
/// captured production document. No network, no clock dependence.
#[test]
fn fresh_document_format_offline() {
    let resolved =
        tinfoil_verifier::bundle::parse_document(FIXTURE_DOC).expect("failed to parse document");

    // 1. The echoed nonce is the one embedded in the request URL.
    assert_eq!(
        hex::encode(&resolved.report_data.nonce),
        FIXTURE_NONCE,
        "nonce should round-trip from the request"
    );

    // 2. The document's `tls_key_fp` equals SHA-256 of the embedded cert's SPKI.
    let cert_spki =
        tinfoil_verifier::sevsnp::sha256_spki_from_der(&resolved.certificate_der).unwrap();
    assert_eq!(
        resolved.report_data.tls_key_fp, cert_spki,
        "tls_key_fp must match the embedded certificate's SPKI hash"
    );

    // 3. The recomputed REPORT_DATA matches the hardware report's REPORT_DATA.
    let report = tinfoil_verifier::sevsnp::parse_report(&resolved.report_bytes).unwrap();
    assert_eq!(
        resolved.report_data.expected_report_data(),
        report.report_data,
        "SHA-256(tls_key_fp || hpke_key || nonce) must equal the report's REPORT_DATA"
    );

    // 4. The enclave's ECDSA signature over the document validates (P-384/SHA-256).
    tinfoil_verifier::bundle::verify_document_signature(&resolved)
        .expect("document signature must verify against the embedded certificate");
}

/// Tampering with any signed byte must invalidate the document signature.
#[test]
fn fresh_document_signature_rejects_tampering() {
    // Change a signed field (the nonce) to a different valid value: the
    // document still parses, but the original signature no longer matches.
    let mut doc: serde_json::Value = serde_json::from_slice(FIXTURE_DOC).unwrap();
    let mut nonce = doc["report_data"]["nonce"].as_str().unwrap().to_string();
    // Flip the last hex nibble to a guaranteed-different value.
    nonce.pop();
    nonce.push('0');
    doc["report_data"]["nonce"] = serde_json::Value::String(nonce);
    let tampered = serde_json::to_vec(&doc).unwrap();

    let resolved = tinfoil_verifier::bundle::parse_document(&tampered).unwrap();
    assert!(
        tinfoil_verifier::bundle::verify_document_signature(&resolved).is_err(),
        "a tampered document must fail signature verification"
    );
}

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

    let mut webpki = rustls::RootCertStore::empty();
    webpki.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    // Fetch the attestation bundle
    let bundle = tinfoil_verifier::bundle::fetch_bundle(None, LIVE_ENCLAVE, LIVE_REPO, &webpki)
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

    // Live tests reach out to inference.tinfoil.sh and ATC, both of which
    // chain under public WebPKI.
    let mut webpki = rustls::RootCertStore::empty();
    webpki.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    // First, get the current measurement from ATC (we don't hardcode it in tests)
    let bundle = tinfoil_verifier::bundle::fetch_bundle(None, LIVE_ENCLAVE, LIVE_REPO, &webpki)
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
        attestation_observer: None,
        tls_roots: webpki,
        // Live production enclaves may predate channel binding; don't require.
        require_channel_binding: false,
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
