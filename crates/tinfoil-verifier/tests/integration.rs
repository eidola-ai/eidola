//! Integration and wire-contract tests for the self-contained v3 flow.
//!
//! Run live tests with:
//! `cargo test --package tinfoil-verifier --test integration -- --ignored`

use base64::Engine as _;
use sha2::{Digest, Sha256};

const LIVE_ORIGIN: &str = "https://inference.tinfoil.sh";
const LIVE_ATTESTATION_URL: &str = "https://inference.tinfoil.sh/.well-known/tinfoil-attestation";

fn synthetic_document() -> Vec<u8> {
    let b64 = &base64::engine::general_purpose::STANDARD;
    let nonce = [0x47; 32];
    let tls = [0x19; 32];
    let hpke = [0x45; 32];
    let crypto = serde_json::to_vec(&serde_json::json!({
        "format": "https://tinfoil.sh/crypto-material/v1",
        "items": [
            {
                "id": "tls",
                "format": "https://tinfoil.sh/key/spki-fp-sha256/v1",
                "data": hex::encode(tls),
            },
            {
                "id": "hpke",
                "format": "https://tinfoil.sh/key/x25519-hpke/v1",
                "data": hex::encode(hpke),
            },
        ],
    }))
    .unwrap();
    let devices = br#"{"format":"https://tinfoil.sh/device-evidence/v1","items":[]}"#;
    let crypto_hash: [u8; 32] = Sha256::digest(&crypto).into();
    let device_hash: [u8; 32] = Sha256::digest(devices).into();
    let mut digest = Sha256::new();
    digest.update(b"https://tinfoil.sh/report-data/v1");
    digest.update(nonce);
    digest.update(crypto_hash);
    digest.update(device_hash);
    let mut report_data = [0u8; 64];
    report_data[..32].copy_from_slice(&digest.finalize());
    let mut report = vec![0u8; 1184];
    report[0x50..0x90].copy_from_slice(&report_data);

    serde_json::to_vec(&serde_json::json!({
        "format": "https://tinfoil.sh/predicate/attestation/v3",
        "challenge": {
            "nonce": hex::encode(nonce),
            "report_data": hex::encode(report_data),
            "report_data_algorithm": "https://tinfoil.sh/report-data/v1",
        },
        "cpu_evidence": {
            "format": "https://tinfoil.sh/format/sev-snp-report/v1",
            "report_base64": b64.encode(report),
            "endorsed": {
                "crypto_material_hash": hex::encode(crypto_hash),
                "device_evidence_hash": hex::encode(device_hash),
            },
        },
        "crypto_material": b64.encode(crypto),
        "device_evidence": b64.encode(devices),
        "collateral": [
            {
                "id": "cpu-endorsement",
                "role": "endorsement",
                "format": "https://tinfoil.sh/collateral/amd-vcek/v1",
                "subjects": ["cpu"],
                "data": {
                    "vcek_der_base64": b64.encode([1, 2, 3]),
                    "cert_chain_pem": concat!(
                        "-----BEGIN CERTIFICATE-----\nAQ==\n-----END CERTIFICATE-----\n",
                        "-----BEGIN CERTIFICATE-----\nAg==\n-----END CERTIFICATE-----\n",
                    ),
                },
            },
            {
                "id": "cpu-crl",
                "role": "endorsement",
                "format": "https://tinfoil.sh/collateral/amd-crl/v1",
                "subjects": ["cpu"],
                "data": {"crl_der_base64": b64.encode([4, 5, 6])},
            },
        ],
    }))
    .unwrap()
}

#[test]
fn v3_envelope_recomputes_exact_endorsed_section_hashes() {
    let resolved =
        tinfoil_verifier::bundle::parse_document(&synthetic_document()).expect("valid v3 envelope");
    assert_eq!(resolved.nonce, [0x47; 32]);
    assert_eq!(resolved.tls_key_fp, [0x19; 32]);
    assert_eq!(resolved.hpke_key, [0x45; 32]);
    assert_eq!(&resolved.report_bytes[0x50..0x90], &resolved.report_data);
    assert_eq!(resolved.vcek_der.as_deref(), Some(&[1, 2, 3][..]));
    assert_eq!(resolved.ask_der.as_deref(), Some(&[1][..]));
    assert_eq!(resolved.ark_der.as_deref(), Some(&[2][..]));
    assert_eq!(resolved.crl_der.as_deref(), Some(&[4, 5, 6][..]));
}

#[test]
fn v3_envelope_rejects_tampered_endorsed_section() {
    let mut doc: serde_json::Value = serde_json::from_slice(&synthetic_document()).unwrap();
    let mut crypto = base64::engine::general_purpose::STANDARD
        .decode(doc["crypto_material"].as_str().unwrap())
        .unwrap();
    crypto.push(b' ');
    doc["crypto_material"] =
        serde_json::Value::String(base64::engine::general_purpose::STANDARD.encode(crypto));
    let err = tinfoil_verifier::bundle::parse_document(&serde_json::to_vec(&doc).unwrap())
        .err()
        .expect("tampered section must fail");
    assert!(err.to_string().contains("crypto_material hash"));
}

#[test]
fn v3_envelope_rejects_unknown_members() {
    let mut doc: serde_json::Value = serde_json::from_slice(&synthetic_document()).unwrap();
    doc["signature"] = serde_json::Value::String("obsolete".to_string());
    let err = tinfoil_verifier::bundle::parse_document(&serde_json::to_vec(&doc).unwrap())
        .err()
        .expect("old document signature field must fail");
    assert!(err.to_string().contains("unknown field"));
}

#[test]
fn v3_envelope_rejects_malformed_amd_certificate_chain() {
    let mut doc: serde_json::Value = serde_json::from_slice(&synthetic_document()).unwrap();
    doc["collateral"][0]["data"]["cert_chain_pem"] =
        serde_json::Value::String("not a certificate chain".to_string());
    let err = tinfoil_verifier::bundle::parse_document(&serde_json::to_vec(&doc).unwrap())
        .err()
        .expect("malformed certificate chain must fail");
    assert!(err.to_string().contains("outside CERTIFICATE blocks"));
}

/// Full connector path against a locally running `tinfoil-shim-mock`.
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
        let cert = <x509_cert::Certificate as der::DecodePem>::from_pem(&pem)
            .expect("parse certificate PEM");
        der::Encode::to_der(&cert).expect("encode certificate DER")
    };
    let ark_der = read_der("ark.pem");
    let ask_der = read_der("ask.pem");
    let mut tls_roots = rustls::RootCertStore::empty();
    tls_roots
        .add(rustls::pki_types::CertificateDer::from(read_der(
            "tls-ca.pem",
        )))
        .expect("add tls-ca");

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
        trusted_ark_der: Some(&ark_der),
        trusted_ask_der: Some(&ask_der),
        snp_min_tcb: None,
        snp_observer: None,
        attestation_observer: None,
        tls_roots,
    })
    .await
    .expect("attesting_client failed");

    let response = client.get(format!("{base_url}/models")).send().await;
    assert!(
        response.is_ok(),
        "mock attestation should pass: {response:?}"
    );
}

fn webpki_client() -> reqwest::Client {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let tls = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    reqwest::Client::builder()
        .tls_backend_preconfigured(tls)
        .build()
        .unwrap()
}

#[tokio::test]
#[ignore = "requires network access to inference.tinfoil.sh"]
async fn live_self_contained_document() {
    let _ =
        rustls::crypto::CryptoProvider::install_default(rustls::crypto::ring::default_provider());
    let resolved =
        tinfoil_verifier::bundle::fetch_well_known(&webpki_client(), LIVE_ATTESTATION_URL)
            .await
            .expect("fetch and parse live v3 document");
    assert_eq!(
        resolved.platform,
        tinfoil_verifier::bundle::Platform::SevSnp
    );
    assert!(resolved.vcek_der.is_some());
    assert!(resolved.ask_der.is_some());
    assert!(resolved.ark_der.is_some());
    assert!(resolved.crl_der.is_some());
    let report = tinfoil_verifier::sevsnp::parse_report(&resolved.report_bytes).unwrap();
    assert_eq!(resolved.report_data, report.report_data);
    let (trusted_ark, trusted_ask) =
        tinfoil_verifier::sevsnp::resolve_chain_certs_der(None, None).unwrap();
    assert_eq!(resolved.ark_der.as_deref(), Some(trusted_ark.as_slice()));
    assert_eq!(resolved.ask_der.as_deref(), Some(trusted_ask.as_slice()));
    tinfoil_verifier::sevsnp::verify_report(
        resolved.vcek_der.as_deref().unwrap(),
        &report,
        resolved.ark_der.as_deref(),
        resolved.ask_der.as_deref(),
    )
    .expect("live report signature and AMD chain");
}

#[tokio::test]
#[ignore = "requires network access to inference.tinfoil.sh"]
async fn live_attesting_client() {
    let _ =
        rustls::crypto::CryptoProvider::install_default(rustls::crypto::ring::default_provider());
    let resolved =
        tinfoil_verifier::bundle::fetch_well_known(&webpki_client(), LIVE_ATTESTATION_URL)
            .await
            .expect("fetch live measurement");
    let report = tinfoil_verifier::sevsnp::parse_report(&resolved.report_bytes).unwrap();
    let allowed = vec![tinfoil_verifier::EnclaveMeasurement {
        snp_measurement: hex::encode(report.measurement),
        tdx_measurement: tinfoil_verifier::TdxMeasurement {
            rtmr1: "0".repeat(96),
            rtmr2: "0".repeat(96),
        },
    }];
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let client = tinfoil_verifier::attesting_client(tinfoil_verifier::AttestingClientConfig {
        allowed_measurements: &allowed,
        inference_base_url: LIVE_ORIGIN,
        trusted_ark_der: None,
        trusted_ask_der: None,
        snp_min_tcb: None,
        snp_observer: None,
        attestation_observer: None,
        tls_roots: roots,
    })
    .await
    .expect("attesting client");
    let response = client
        .get(format!("{LIVE_ORIGIN}/v1/models"))
        .send()
        .await
        .expect("request through attested connection");
    assert!(response.status().is_success() || response.status().as_u16() == 401);
}
