//! Dev shim — emulates the Tinfoil Container shim for local development.
//!
//! Generates a mock SEV-SNP attestation chain using the exact same algorithms
//! as production hardware:
//!
//! - **Certificate chain:** ARK → ASK → VCEK, all signed with RSA-PSS SHA-384
//! - **Report signature:** ECDSA P-384 with SHA-384 (VCEK signs the report)
//! - **TLS binding:** report_data[0..32] = SHA-256(TLS cert SPKI)
//!
//! This exercises the complete verification path in `tinfoil-verifier` with
//! no escape hatches — the `sev` crate's `Verifiable` trait verifies every
//! signature in the chain.
//!
//! The ARK key is persisted to `.dev-certs/` so clients can pin the root CA.
//! All other keys are regenerated on each startup.
//!
//! Environment variables:
//!   UPSTREAM_URL    - upstream server URL (default: http://127.0.0.1:8080)
//!   LISTEN_ADDR     - address to bind (default: 0.0.0.0:8443)
//!   DEV_MEASUREMENT - measurement to advertise (default: 48 zero bytes hex)
//!   CERT_DIR        - directory to store persistent certs (default: .dev-certs)

use std::fs;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use arc_swap::ArcSwap;
use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use base64::Engine;
use der::{Decode, Encode, asn1::BitString};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as AutoBuilder;
use rcgen::KeyPair;
use rsa::signature::{RandomizedSigner, SignatureEncoding};
use rustls::server::{ClientHello, ResolvesServerCert};
use rustls::sign::CertifiedKey;
use sha2::{Digest, Sha256};
use spki::{AlgorithmIdentifierOwned, ObjectIdentifier, SubjectPublicKeyInfoOwned};
use tokio::net::TcpListener;
use tower_service::Service;
use tracing::{debug, info};
use x509_cert::Certificate;
use x509_cert::serial_number::SerialNumber;
use x509_cert::time::{Time, Validity};

/// 48 zero bytes = 96 hex chars (matches SEV-SNP measurement size).
const DEFAULT_MEASUREMENT: &str = "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";

/// RSA-PSS OID (1.2.840.113549.1.1.10) — required by the `sev` crate's chain verifier.
const RSA_PSS_OID: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.2.840.113549.1.1.10");

/// RSA key size for the mock CA chain. 2048-bit for fast generation.
const RSA_KEY_BITS: usize = 2048;

// ── Report byte offsets (AMD SEV-SNP ABI, V2 Genoa layout) ──────────────
const OFF_VERSION: usize = 0x000; // u32 LE
const OFF_SIG_ALGO: usize = 0x034; // u32 LE (1 = ECDSA P-384 SHA-384)
const OFF_CURRENT_TCB: usize = 0x038; // 8 bytes (legacy: [bl, tee, 0,0,0,0, snp, ucode])
const OFF_REPORT_DATA: usize = 0x050; // 64 bytes
const OFF_MEASUREMENT: usize = 0x090; // 48 bytes
const OFF_REPORTED_TCB: usize = 0x180; // 8 bytes
const OFF_CHIP_ID: usize = 0x1A0; // 64 bytes
const OFF_COMMITTED_TCB: usize = 0x1E0; // 8 bytes
const OFF_LAUNCH_TCB: usize = 0x1F0; // 8 bytes
const OFF_SIGNATURE: usize = 0x2A0; // 512 bytes (72 R + 72 S + 368 reserved)
const REPORT_SIZE: usize = 0x4A0; // 1184 bytes

fn tls_config() -> rustls::ClientConfig {
    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth()
}

#[derive(Clone)]
struct AppState {
    attestation_json: String,
    upstream_url: String,
    proxy_client: reqwest::Client,
}

// ── Certificate construction ────────────────────────────────────────────

/// RSA-PSS SHA-384 AlgorithmIdentifier (OID only, no params).
/// The `sev` crate only checks the OID and always uses SHA-384 for verification.
fn rsa_pss_alg_id() -> AlgorithmIdentifierOwned {
    AlgorithmIdentifierOwned {
        oid: RSA_PSS_OID,
        parameters: None,
    }
}

/// Build an X.509 certificate signed with RSA-PSS SHA-384.
///
/// The certificate's `signatureAlgorithm` and TBS `signature` fields both use
/// the RSA-PSS OID, matching what the `sev` crate's `Verifiable` trait expects.
fn build_rsa_pss_cert(
    serial: u8,
    issuer: &x509_cert::name::Name,
    subject: &x509_cert::name::Name,
    spki: SubjectPublicKeyInfoOwned,
    signer: &rsa::pss::SigningKey<sha2::Sha384>,
    extensions: Option<x509_cert::ext::Extensions>,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let sig_alg = rsa_pss_alg_id();

    let not_before = Time::UtcTime(der::asn1::UtcTime::from_date_time(der::DateTime::new(
        2024, 1, 1, 0, 0, 0,
    )?)?);
    let not_after = Time::UtcTime(der::asn1::UtcTime::from_date_time(der::DateTime::new(
        2049, 12, 31, 23, 59, 59,
    )?)?);

    let tbs = x509_cert::TbsCertificate {
        version: x509_cert::Version::V3,
        serial_number: SerialNumber::new(&[serial])?,
        signature: sig_alg.clone(),
        issuer: issuer.clone(),
        validity: Validity {
            not_before,
            not_after,
        },
        subject: subject.clone(),
        subject_public_key_info: spki,
        issuer_unique_id: None,
        subject_unique_id: None,
        extensions,
    };

    let tbs_der = tbs.to_der()?;

    let signature: rsa::pss::Signature = signer.sign_with_rng(&mut rand_core::OsRng, &tbs_der);
    let sig_bytes = signature.to_vec();

    let cert = Certificate {
        tbs_certificate: tbs,
        signature_algorithm: sig_alg,
        signature: BitString::from_bytes(&sig_bytes)?,
    };

    Ok(cert.to_der()?)
}

/// Build a Basic Constraints extension (CA:TRUE).
fn ca_extensions() -> Result<x509_cert::ext::Extensions, Box<dyn std::error::Error + Send + Sync>> {
    let bc = x509_cert::ext::pkix::BasicConstraints {
        ca: true,
        path_len_constraint: None,
    };
    let bc_der = bc.to_der()?;
    let ext = x509_cert::ext::Extension {
        extn_id: ObjectIdentifier::new_unwrap("2.5.29.19"),
        critical: true,
        extn_value: der::asn1::OctetString::new(bc_der)?,
    };
    Ok(vec![ext])
}

// ── Report construction ─────────────────────────────────────────────────

/// Write a TcbVersion in legacy Genoa format: [bl, tee, 0, 0, 0, 0, snp, ucode]
fn write_tcb(report: &mut [u8], offset: usize, bl: u8, snp: u8, ucode: u8) {
    report[offset] = bl;
    report[offset + 1] = 0; // tee
    report[offset + 6] = snp;
    report[offset + 7] = ucode;
}

/// Build a mock SEV-SNP attestation report (V2 Genoa layout).
fn build_report(measurement: &[u8], tls_fingerprint: &[u8; 32]) -> [u8; REPORT_SIZE] {
    let mut report = [0u8; REPORT_SIZE];

    // Version = 2 (V2 report)
    report[OFF_VERSION..OFF_VERSION + 4].copy_from_slice(&2u32.to_le_bytes());

    // sig_algo = 1 (ECDSA P-384 SHA-384)
    report[OFF_SIG_ALGO..OFF_SIG_ALGO + 4].copy_from_slice(&1u32.to_le_bytes());

    // TCB values (must pass minimum policy: bl=0x07, snp=0x0E, ucode=0x48)
    let (bl, snp, ucode) = (0x07, 0x0E, 0x48);
    write_tcb(&mut report, OFF_CURRENT_TCB, bl, snp, ucode);
    write_tcb(&mut report, OFF_REPORTED_TCB, bl, snp, ucode);
    write_tcb(&mut report, OFF_COMMITTED_TCB, bl, snp, ucode);
    write_tcb(&mut report, OFF_LAUNCH_TCB, bl, snp, ucode);

    // report_data[0..32] = TLS fingerprint
    report[OFF_REPORT_DATA..OFF_REPORT_DATA + 32].copy_from_slice(tls_fingerprint);

    // measurement (48 bytes)
    let len = measurement.len().min(48);
    report[OFF_MEASUREMENT..OFF_MEASUREMENT + len].copy_from_slice(&measurement[..len]);

    // chip_id — bytes 8..64 must NOT all be zero, otherwise the sev crate
    // detects it as Turin-like and uses a different TcbVersion encoding.
    report[OFF_CHIP_ID] = 0xDE;
    report[OFF_CHIP_ID + 1] = 0xAD;
    report[OFF_CHIP_ID + 8] = 0x01; // byte 8+ nonzero → Genoa detection

    report
}

/// Sign the report body with ECDSA P-384 SHA-384 and write the signature
/// at offset 0x2A0. The `sev` crate expects 72-byte R and S fields containing
/// 48-byte little-endian scalars, zero-padded to 72 bytes.
fn sign_report(report: &mut [u8; REPORT_SIZE], key: &p384::ecdsa::SigningKey) {
    use p384::ecdsa::signature::DigestSigner;
    use sha2::Sha384;

    let digest = Sha384::new_with_prefix(&report[..OFF_SIGNATURE]);
    let sig: p384::ecdsa::Signature = key.sign_digest(digest);
    let (r, s) = sig.split_bytes();

    // R: 48 bytes big-endian → little-endian in first 48 bytes of 72-byte field
    let mut r_le: Vec<u8> = r.to_vec();
    r_le.reverse();
    report[OFF_SIGNATURE..OFF_SIGNATURE + 48].copy_from_slice(&r_le);

    // S: same layout, at offset +72
    let mut s_le: Vec<u8> = s.to_vec();
    s_le.reverse();
    report[OFF_SIGNATURE + 72..OFF_SIGNATURE + 72 + 48].copy_from_slice(&s_le);
}

// ── Key/cert persistence ────────────────────────────────────────────────

/// Load an RSA private key from disk, or generate and persist a new one.
fn load_or_generate_rsa_key(
    path: impl AsRef<Path>,
    label: &str,
) -> Result<rsa::RsaPrivateKey, Box<dyn std::error::Error + Send + Sync>> {
    let path = path.as_ref();
    if path.exists() {
        info!("Loading persistent {label} key from {}", path.display());
        let pem = fs::read_to_string(path)?;
        Ok(<rsa::RsaPrivateKey as rsa::pkcs8::DecodePrivateKey>::from_pkcs8_pem(&pem)?)
    } else {
        info!("Generating new {RSA_KEY_BITS}-bit {label} key...");
        let key = rsa::RsaPrivateKey::new(&mut rand_core::OsRng, RSA_KEY_BITS)?;
        let pem = rsa::pkcs8::EncodePrivateKey::to_pkcs8_pem(&key, rsa::pkcs8::LineEnding::LF)?;
        fs::write(path, pem.as_bytes())?;
        Ok(key)
    }
}

/// Write a DER certificate to disk as PEM (only if the file doesn't exist).
fn write_cert_pem(
    der: &[u8],
    path: impl AsRef<Path>,
    label: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let path = path.as_ref();
    if path.exists() {
        return Ok(());
    }
    let cert = x509_cert::Certificate::from_der(der)?;
    let pem = {
        use der::pem::LineEnding;
        der::EncodePem::to_pem(&cert, LineEnding::LF)?
    };
    fs::write(path, &pem)?;
    info!("{label} cert written to {}", path.display());
    Ok(())
}

// ── Main ────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider())
        .expect("failed to install rustls crypto provider");

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("dev_shim=info".parse().unwrap()),
        )
        .init();

    let upstream_url =
        std::env::var("UPSTREAM_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
    let listen_addr: SocketAddr = std::env::var("LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8443".to_string())
        .parse()
        .expect("invalid LISTEN_ADDR");
    let measurement =
        std::env::var("DEV_MEASUREMENT").unwrap_or_else(|_| DEFAULT_MEASUREMENT.to_string());
    let cert_dir = std::env::var("CERT_DIR").unwrap_or_else(|_| ".dev-certs".to_string());
    let measurement_bytes = hex::decode(&measurement).expect("invalid DEV_MEASUREMENT hex");

    // ── 1. Load or generate persistent RSA keys (ARK + ASK) ────────────

    fs::create_dir_all(&cert_dir)?;
    let ark_rsa_key = load_or_generate_rsa_key(Path::new(&cert_dir).join("ark.key"), "ARK")?;
    let ask_rsa_key = load_or_generate_rsa_key(Path::new(&cert_dir).join("ask.key"), "ASK")?;

    let ark_pss_signer = rsa::pss::SigningKey::<sha2::Sha384>::new(ark_rsa_key.clone());
    let ask_pss_signer = rsa::pss::SigningKey::<sha2::Sha384>::new(ask_rsa_key.clone());

    // ── 2. Build ARK + ASK certs (regenerated from persistent keys) ─────
    //
    // The certs are deterministic given the same keys (modulo PSS salt
    // randomness), but we always regenerate and write them so the PEM files
    // in .dev-certs/ are always consistent with the keys.

    let ark_pkcs8_der = rsa::pkcs8::EncodePrivateKey::to_pkcs8_der(&ark_rsa_key)?;
    let ark_pkcs8_ref =
        rustls::pki_types::PrivatePkcs8KeyDer::from(ark_pkcs8_der.as_bytes().to_vec());
    let ark_rcgen_key =
        KeyPair::from_pkcs8_der_and_sign_algo(&ark_pkcs8_ref, &rcgen::PKCS_RSA_SHA256)?;

    let mut ark_rcgen_params =
        rcgen::CertificateParams::new(vec!["Local Dev Root CA".to_string()])?;
    ark_rcgen_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    let ark_rcgen_issuer = rcgen::CertifiedIssuer::self_signed(ark_rcgen_params, ark_rcgen_key)?;

    // Extract subject and SPKI from the rcgen cert to ensure matching names
    let parsed_rcgen_ark = x509_cert::Certificate::from_der(ark_rcgen_issuer.der().as_ref())?;
    let ark_subject = parsed_rcgen_ark.tbs_certificate.subject.clone();
    let ark_spki = parsed_rcgen_ark
        .tbs_certificate
        .subject_public_key_info
        .clone();

    let ark_pss_der = build_rsa_pss_cert(
        1,
        &ark_subject,
        &ark_subject,
        ark_spki.clone(),
        &ark_pss_signer,
        Some(ca_extensions()?),
    )?;

    let ask_pub_spki_der =
        rsa::pkcs8::EncodePublicKey::to_public_key_der(&ask_rsa_key.to_public_key())?;
    let ask_spki = SubjectPublicKeyInfoOwned::from_der(ask_pub_spki_der.as_ref())?;

    let ask_der = build_rsa_pss_cert(
        2,
        &ark_subject, // issuer = ARK
        &ark_subject, // subject = same (simplicity; sev crate doesn't check names)
        ask_spki,
        &ark_pss_signer, // signed by ARK
        Some(ca_extensions()?),
    )?;

    // Write cert PEM files (for client config)
    write_cert_pem(&ark_pss_der, Path::new(&cert_dir).join("ark.pem"), "ARK")?;
    write_cert_pem(&ask_der, Path::new(&cert_dir).join("ask.pem"), "ASK")?;

    // ── 5. Generate VCEK (P-384 key in RSA-PSS-signed cert) ─────────────

    let vcek_signing_key = p384::ecdsa::SigningKey::random(&mut rand_core::OsRng);
    let vcek_pub_key = p384::PublicKey::from(vcek_signing_key.verifying_key());
    let vcek_pub_spki_der = p384::pkcs8::EncodePublicKey::to_public_key_der(&vcek_pub_key)?;
    let vcek_spki = SubjectPublicKeyInfoOwned::from_der(vcek_pub_spki_der.as_ref())?;

    let vcek_der = build_rsa_pss_cert(
        3,
        &ark_subject, // issuer = ASK (same name)
        &ark_subject, // subject = same
        vcek_spki,
        &ask_pss_signer, // signed by ASK
        None,            // end entity, no CA extension
    )?;

    // ── Self-check: verify chain + report roundtrip ────────────────────

    {
        use sev::certs::snp::{Certificate as SevCert, Chain, Verifiable, ca};

        let sev_ark = SevCert::from_der(&ark_pss_der).expect("self-check: failed to parse ARK");
        let sev_ask = SevCert::from_der(&ask_der).expect("self-check: failed to parse ASK");
        let sev_vcek = SevCert::from_der(&vcek_der).expect("self-check: failed to parse VCEK");

        let chain = Chain {
            ca: ca::Chain {
                ark: sev_ark,
                ask: sev_ask,
            },
            vek: sev_vcek,
        };
        (&chain)
            .verify()
            .expect("self-check: certificate chain verification failed");
        info!("Self-check: certificate chain verified (ARK → ASK → VCEK)");
    }

    // ── 6. Generate TLS leaf certificate ────────────────────────────────

    let tls_alg = &rcgen::PKCS_ECDSA_P256_SHA256;
    let tls_key = KeyPair::generate_for(tls_alg)?;
    let tls_params = rcgen::CertificateParams::new(vec![
        "localhost".to_string(),
        "server".to_string(),
        "shim".to_string(),
    ])?;
    let tls_cert = tls_params.signed_by(&tls_key, &ark_rcgen_issuer)?;

    let cert_der = rustls::pki_types::CertificateDer::from(tls_cert.der().to_vec());
    let key_der = rustls::pki_types::PrivateKeyDer::try_from(tls_key.serialize_der())
        .expect("invalid private key DER");

    // ── 7. Compute TLS fingerprint ──────────────────────────────────────

    let parsed_tls = x509_cert::Certificate::from_der(cert_der.as_ref())?;
    let tls_spki_der = parsed_tls
        .tbs_certificate
        .subject_public_key_info
        .to_der()?;
    let tls_fingerprint: [u8; 32] = Sha256::digest(&tls_spki_der).into();

    info!("TLS fingerprint: {}", hex::encode(tls_fingerprint));
    info!("Measurement: {measurement}");

    // ── 8. Build and sign mock attestation report ────────────────────────

    let mut report = build_report(&measurement_bytes, &tls_fingerprint);
    sign_report(&mut report, &vcek_signing_key);

    // ── 9. Build attestation JSON ───────────────────────────────────────

    let b64 = &base64::engine::general_purpose::STANDARD;

    // V3 attestation: raw report + VCEK. ARK and ASK come from client config.
    let attestation_json = serde_json::to_string(&serde_json::json!({
        "format": "https://tinfoil.sh/predicate/attestation/v3",
        "cpu": {
            "platform": "sev-snp",
            "report": b64.encode(report),
        },
        "vcek": b64.encode(&vcek_der),
    }))?;

    // ── 10. Start HTTPS server ──────────────────────────────────────────

    let state = AppState {
        attestation_json,
        upstream_url: upstream_url.clone(),
        proxy_client: reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .tls_backend_preconfigured(tls_config())
            .build()?,
    };

    let app = Router::new()
        .route("/.well-known/tinfoil-attestation", get(handle_attestation))
        .fallback(handle_proxy)
        .with_state(state);

    let signing_key = rustls::crypto::CryptoProvider::get_default()
        .unwrap()
        .key_provider
        .load_private_key(key_der)
        .expect("failed to load private key");

    let certified_key = CertifiedKey::new(vec![cert_der], signing_key);
    let resolver = DevCertResolver(ArcSwap::new(Arc::new(certified_key)));

    let tls_config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_cert_resolver(Arc::new(resolver));

    let tls_acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(tls_config));

    let listener = TcpListener::bind(listen_addr).await?;
    info!("Dev shim listening on https://{listen_addr}");
    info!("Proxying to {upstream_url}");

    loop {
        let (tcp_stream, remote_addr) = listener.accept().await?;
        let tls_acceptor = tls_acceptor.clone();
        let app = app.clone();

        tokio::spawn(async move {
            let tls_stream = match tls_acceptor.accept(tcp_stream).await {
                Ok(s) => s,
                Err(e) => {
                    debug!("TLS handshake failed from {remote_addr}: {e}");
                    return;
                }
            };

            let io = TokioIo::new(tls_stream);
            let service = hyper::service::service_fn(move |req| {
                let mut app = app.clone();
                async move { app.call(req).await }
            });

            if let Err(e) = AutoBuilder::new(TokioExecutor::new())
                .serve_connection(io, service)
                .await
            {
                debug!("Connection error from {remote_addr}: {e}");
            }
        });
    }
}

// ── TLS cert resolver ───────────────────────────────────────────────────

#[derive(Debug)]
struct DevCertResolver(ArcSwap<CertifiedKey>);

impl ResolvesServerCert for DevCertResolver {
    fn resolve(&self, _client_hello: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        Some(self.0.load_full())
    }
}

// ── HTTP handlers ───────────────────────────────────────────────────────

async fn handle_attestation(State(state): State<AppState>) -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        state.attestation_json,
    )
}

async fn handle_proxy(
    State(state): State<AppState>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let path_and_query = uri.path_and_query().map(|p| p.as_str()).unwrap_or("/");
    let target_url = format!("{}{}", state.upstream_url, path_and_query);

    let body_bytes = match axum::body::to_bytes(body, 10 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("failed to read body: {e}")).into_response();
        }
    };

    let mut req = state.proxy_client.request(method, &target_url);
    for (name, value) in &headers {
        if name == "host" || name == "transfer-encoding" {
            continue;
        }
        req = req.header(name, value);
    }
    req = req.body(body_bytes);

    match req.send().await {
        Ok(resp) => {
            let status = resp.status();
            let resp_headers = resp.headers().clone();
            let resp_body = resp.bytes().await.unwrap_or_default();

            let mut response = (status, resp_body).into_response();
            for (name, value) in &resp_headers {
                response.headers_mut().insert(name, value.clone());
            }
            response
        }
        Err(e) => (StatusCode::BAD_GATEWAY, format!("proxy error: {e}")).into_response(),
    }
}
