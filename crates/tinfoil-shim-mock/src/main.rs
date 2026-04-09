//! Tinfoil shim mock — emulates the Tinfoil Container shim for local development.
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
//! Three persistent identities live in `.dev-certs/`:
//!
//! - **ARK + ASK** (RSA-PSS SHA-384): the SEV-SNP attestation chain root and
//!   intermediate. The `sev` crate's `Verifiable` trait requires PSS, so these
//!   stay PSS. They are *only* consumed by the verifier via the
//!   `--hardware-root-ca` / `--hardware-intermediate-ca` config flags.
//! - **TLS-CA** (RSA PKCS#1 v1.5 SHA-384): a separate self-signed root used
//!   *only* to sign the TLS leaf. Apple's Security framework does not support
//!   RSA-PSS in chain validation (verified empirically), so the TLS path
//!   cannot share an identity with the SEV-SNP chain. Developers trust this
//!   cert in their OS keychain, not the ARK.
//!
//! On boot the shim loads any existing key+cert files unchanged and only
//! generates fresh material for files that are missing — the trust root never
//! flows from the shim's API, only from the filesystem. The VCEK and the TLS
//! leaf are ephemeral and regenerated on every startup.
//!
//! Environment variables:
//!   UPSTREAM_URL    - upstream server URL (default: http://127.0.0.1:8080)
//!   LISTEN_ADDR     - address to bind (default: 0.0.0.0:8443)
//!   DEV_MEASUREMENT - measurement to advertise (default: 48 zero bytes hex)
//!   CERT_DIR        - directory to store persistent certs (default: .dev-certs)

use std::fs;
use std::net::SocketAddr;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
use rsa::sha2 as rsa_sha2;
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

/// DER-encoded RSASSA-PSS-params (RFC 4055 §3.1) for SHA-384 / MGF1-SHA-384 /
/// salt length 48 / default trailerField. Strict X.509 implementations
/// (notably macOS Security framework and OpenSSL) refuse to honor an RSA-PSS
/// AlgorithmIdentifier whose `parameters` field is absent or unparseable, so
/// these explicit bytes are required for the cert to be usable as a trust
/// anchor. The `sev` crate is more lenient and only checks the OID, but it
/// happily verifies a properly-parameterised cert too.
#[rustfmt::skip]
const RSA_PSS_SHA384_PARAMS: &[u8] = &[
    0x30, 0x34,
    // [0] HashAlgorithm = sha-384 (NULL params)
    0xa0, 0x0f,
        0x30, 0x0d,
            0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02,
            0x05, 0x00,
    // [1] MaskGenAlgorithm = mgf1 with sha-384
    0xa1, 0x1c,
        0x30, 0x1a,
            0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x08,
            0x30, 0x0d,
                0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02,
                0x05, 0x00,
    // [2] saltLength = 48
    0xa2, 0x03,
        0x02, 0x01, 0x30,
];

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

/// RSA-PSS SHA-384 AlgorithmIdentifier with proper RFC 4055 parameters.
fn rsa_pss_alg_id() -> Result<AlgorithmIdentifierOwned, der::Error> {
    Ok(AlgorithmIdentifierOwned {
        oid: RSA_PSS_OID,
        parameters: Some(der::Any::from_der(RSA_PSS_SHA384_PARAMS)?),
    })
}

/// PKCS#1 v1.5 SHA-384 OID (sha384WithRSAEncryption).
const RSA_PKCS1_SHA384_OID: ObjectIdentifier =
    ObjectIdentifier::new_unwrap("1.2.840.113549.1.1.12");

/// RSA PKCS#1 v1.5 SHA-384 AlgorithmIdentifier. Per RFC 4055 §5 the
/// `parameters` field MUST be present and encoded as ASN.1 NULL.
fn rsa_pkcs1_sha384_alg_id() -> Result<AlgorithmIdentifierOwned, der::Error> {
    Ok(AlgorithmIdentifierOwned {
        oid: RSA_PKCS1_SHA384_OID,
        parameters: Some(der::Any::from_der(&[0x05, 0x00])?),
    })
}

/// Build a `Validity` window ending `duration` from now, encoded as
/// `UtcTime`. `not_before` is back-dated by one hour so minor clock skew
/// between dev machines doesn't cause "not yet valid" rejections. Valid for
/// any `not_after` before 2050-01-01 (the upper bound of `UtcTime`).
fn validity_from_now(
    duration: Duration,
) -> Result<Validity, Box<dyn std::error::Error + Send + Sync>> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
    let backdate = Duration::from_secs(60 * 60);
    let nb = now.saturating_sub(backdate);
    let na = now + duration;
    let nb_dt = der::DateTime::from_unix_duration(nb)?;
    let na_dt = der::DateTime::from_unix_duration(na)?;
    Ok(Validity {
        not_before: Time::UtcTime(der::asn1::UtcTime::from_date_time(nb_dt)?),
        not_after: Time::UtcTime(der::asn1::UtcTime::from_date_time(na_dt)?),
    })
}

/// Build an X.509 certificate signed with RSA PKCS#1 v1.5 SHA-384. Used for
/// the TLS-CA and TLS leaf certificates because Apple's Security framework
/// does not support RSA-PSS in chain validation.
fn build_rsa_pkcs1_cert(
    serial: u8,
    issuer: &x509_cert::name::Name,
    subject: &x509_cert::name::Name,
    spki: SubjectPublicKeyInfoOwned,
    signer: &rsa::pkcs1v15::SigningKey<rsa_sha2::Sha384>,
    validity: Validity,
    extensions: Option<x509_cert::ext::Extensions>,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    use rsa::signature::Signer;

    let sig_alg = rsa_pkcs1_sha384_alg_id()?;

    let tbs = x509_cert::TbsCertificate {
        version: x509_cert::Version::V3,
        serial_number: SerialNumber::new(&[serial])?,
        signature: sig_alg.clone(),
        issuer: issuer.clone(),
        validity,
        subject: subject.clone(),
        subject_public_key_info: spki,
        issuer_unique_id: None,
        subject_unique_id: None,
        extensions,
    };

    let tbs_der = tbs.to_der()?;
    // PKCS#1 v1.5 is deterministic — no RNG needed.
    let signature: rsa::pkcs1v15::Signature = signer.sign(&tbs_der);
    let sig_bytes = signature.to_vec();

    let cert = Certificate {
        tbs_certificate: tbs,
        signature_algorithm: sig_alg,
        signature: BitString::from_bytes(&sig_bytes)?,
    };

    Ok(cert.to_der()?)
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
    signer: &rsa::pss::SigningKey<rsa_sha2::Sha384>,
    validity: Validity,
    extensions: Option<x509_cert::ext::Extensions>,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let sig_alg = rsa_pss_alg_id()?;

    let tbs = x509_cert::TbsCertificate {
        version: x509_cert::Version::V3,
        serial_number: SerialNumber::new(&[serial])?,
        signature: sig_alg.clone(),
        issuer: issuer.clone(),
        validity,
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

/// Compute a 20-byte key identifier from an SPKI by hashing the
/// `subjectPublicKey` BIT STRING contents. RFC 5280 §4.2.1.2 method 1
/// specifies SHA-1, but the spec also explicitly permits "other methods …
/// for generating unique numbers"; we use truncated SHA-256 so we can stay
/// on the workspace's existing `sha2` dependency. Apple's chain builder
/// matches AKI/SKI by byte equality and doesn't re-hash anything, so any
/// stable function over the SPKI produces compatible identifiers as long
/// as we use the same one everywhere.
fn key_id_from_spki(spki: &SubjectPublicKeyInfoOwned) -> Vec<u8> {
    let pk_bits = spki.subject_public_key.raw_bytes();
    let hash: [u8; 32] = Sha256::digest(pk_bits).into();
    hash[..20].to_vec()
}

/// Build a SubjectKeyIdentifier extension wrapping the given identifier.
fn subject_key_id_ext(
    key_id: &[u8],
) -> Result<x509_cert::ext::Extension, Box<dyn std::error::Error + Send + Sync>> {
    let ski =
        x509_cert::ext::pkix::SubjectKeyIdentifier(der::asn1::OctetString::new(key_id.to_vec())?);
    Ok(x509_cert::ext::Extension {
        extn_id: ObjectIdentifier::new_unwrap("2.5.29.14"),
        critical: false,
        extn_value: der::asn1::OctetString::new(ski.to_der()?)?,
    })
}

/// Build an AuthorityKeyIdentifier extension whose `keyIdentifier` field
/// references the given parent key identifier.
fn authority_key_id_ext(
    issuer_key_id: &[u8],
) -> Result<x509_cert::ext::Extension, Box<dyn std::error::Error + Send + Sync>> {
    let aki = x509_cert::ext::pkix::AuthorityKeyIdentifier {
        key_identifier: Some(der::asn1::OctetString::new(issuer_key_id.to_vec())?),
        authority_cert_issuer: None,
        authority_cert_serial_number: None,
    };
    Ok(x509_cert::ext::Extension {
        extn_id: ObjectIdentifier::new_unwrap("2.5.29.35"),
        critical: false,
        extn_value: der::asn1::OctetString::new(aki.to_der()?)?,
    })
}

/// Build the extensions for a CA cert (ARK / ASK):
///   * BasicConstraints CA:TRUE (critical)
///   * KeyUsage keyCertSign + cRLSign (critical)
///   * SubjectKeyIdentifier from `own_key_id`
///   * AuthorityKeyIdentifier from `issuer_key_id` (for a self-signed root,
///     pass the same value as `own_key_id`)
///
/// macOS's chain builder requires both SKI on the parent and AKI on the
/// child to construct a chain — without them it refuses to fall back to
/// DN matching, even when subject/issuer DNs are byte-identical. RFC 5280
/// §4.2.1.1 / §4.2.1.2 list these as MUST for conforming CAs.
fn ca_extensions(
    own_key_id: &[u8],
    issuer_key_id: &[u8],
) -> Result<x509_cert::ext::Extensions, Box<dyn std::error::Error + Send + Sync>> {
    use x509_cert::ext::pkix::{KeyUsage, KeyUsages};

    let bc = x509_cert::ext::pkix::BasicConstraints {
        ca: true,
        path_len_constraint: None,
    };
    let bc_ext = x509_cert::ext::Extension {
        extn_id: ObjectIdentifier::new_unwrap("2.5.29.19"),
        critical: true,
        extn_value: der::asn1::OctetString::new(bc.to_der()?)?,
    };

    let ku = KeyUsage(KeyUsages::KeyCertSign | KeyUsages::CRLSign);
    let ku_ext = x509_cert::ext::Extension {
        extn_id: ObjectIdentifier::new_unwrap("2.5.29.15"),
        critical: true,
        extn_value: der::asn1::OctetString::new(ku.to_der()?)?,
    };

    Ok(vec![
        bc_ext,
        ku_ext,
        subject_key_id_ext(own_key_id)?,
        authority_key_id_ext(issuer_key_id)?,
    ])
}

/// Build the extensions for an end-entity TLS server cert:
///   * BasicConstraints CA:FALSE (critical)
///   * KeyUsage digitalSignature + keyEncipherment (critical)
///   * ExtendedKeyUsage serverAuth (required by macOS for TLS server certs)
///   * SubjectKeyIdentifier from `own_key_id`
///   * AuthorityKeyIdentifier from `issuer_key_id`
///   * SubjectAltName listing each DNS name
fn tls_leaf_extensions(
    dns_names: &[&str],
    own_key_id: &[u8],
    issuer_key_id: &[u8],
) -> Result<x509_cert::ext::Extensions, Box<dyn std::error::Error + Send + Sync>> {
    use der::asn1::Ia5String;
    use x509_cert::ext::pkix::name::GeneralName;
    use x509_cert::ext::pkix::{KeyUsage, KeyUsages};

    let bc = x509_cert::ext::pkix::BasicConstraints {
        ca: false,
        path_len_constraint: None,
    };
    let bc_ext = x509_cert::ext::Extension {
        extn_id: ObjectIdentifier::new_unwrap("2.5.29.19"),
        critical: true,
        extn_value: der::asn1::OctetString::new(bc.to_der()?)?,
    };

    let ku = KeyUsage(KeyUsages::DigitalSignature | KeyUsages::KeyEncipherment);
    let ku_ext = x509_cert::ext::Extension {
        extn_id: ObjectIdentifier::new_unwrap("2.5.29.15"),
        critical: true,
        extn_value: der::asn1::OctetString::new(ku.to_der()?)?,
    };

    // ExtendedKeyUsage = serverAuth (1.3.6.1.5.5.7.3.1)
    let eku = x509_cert::ext::pkix::ExtendedKeyUsage(vec![ObjectIdentifier::new_unwrap(
        "1.3.6.1.5.5.7.3.1",
    )]);
    let eku_ext = x509_cert::ext::Extension {
        extn_id: ObjectIdentifier::new_unwrap("2.5.29.37"),
        critical: false,
        extn_value: der::asn1::OctetString::new(eku.to_der()?)?,
    };

    let general_names: Vec<GeneralName> = dns_names
        .iter()
        .map(|n| Ok::<_, der::Error>(GeneralName::DnsName(Ia5String::new(*n)?)))
        .collect::<Result<_, _>>()?;
    let san = x509_cert::ext::pkix::SubjectAltName(general_names);
    let san_ext = x509_cert::ext::Extension {
        extn_id: ObjectIdentifier::new_unwrap("2.5.29.17"),
        critical: false,
        extn_value: der::asn1::OctetString::new(san.to_der()?)?,
    };

    Ok(vec![
        bc_ext,
        ku_ext,
        eku_ext,
        subject_key_id_ext(own_key_id)?,
        authority_key_id_ext(issuer_key_id)?,
        san_ext,
    ])
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
    use p384::ecdsa::signature::Signer;

    let sig: p384::ecdsa::Signature = key.sign(&report[..OFF_SIGNATURE]);
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

/// A persisted PSS CA (ARK / ASK): signing key, cert DER, parsed subject,
/// and the key identifier used by downstream certs to populate AKI.
struct PersistedCa {
    signer: rsa::pss::SigningKey<rsa_sha2::Sha384>,
    cert_der: Vec<u8>,
    subject: x509_cert::name::Name,
    key_id: Vec<u8>,
}

/// A persisted PKCS#1 v1.5 CA (TLS-CA): the TLS chain's trust anchor. The
/// cert DER itself is loaded purely to validate consistency with the key on
/// disk and isn't held here — the TLS leaf is sent on the wire alone, with
/// the developer's keychain providing the root.
struct PersistedTlsCa {
    signer: rsa::pkcs1v15::SigningKey<rsa_sha2::Sha384>,
    subject: x509_cert::name::Name,
    key_id: Vec<u8>,
}

/// Encode a DER cert as PEM and write it to `path`.
fn write_cert_pem_atomic(
    der: &[u8],
    path: &Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use der::pem::LineEnding;
    let cert = x509_cert::Certificate::from_der(der)?;
    let pem = der::EncodePem::to_pem(&cert, LineEnding::LF)?;
    fs::write(path, &pem)?;
    Ok(())
}

/// Load `cert_path` if it exists and verify its SPKI matches `pub_spki_der`,
/// otherwise call `build` to create a fresh cert and persist it.
///
/// The on-disk pair `(key_path, cert_path)` is the source of trust: the shim
/// only writes when nothing is there, never overwrites, and refuses to start
/// if a cert is found that doesn't match its companion key (the developer
/// must manually delete `.dev-certs/` to recover).
fn load_or_create_pem<F>(
    label: &str,
    cert_path: &Path,
    pub_spki_der: &[u8],
    build: F,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>>
where
    F: FnOnce() -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>>,
{
    if cert_path.exists() {
        info!(
            "Loading persistent {label} cert from {}",
            cert_path.display()
        );
        let pem = fs::read_to_string(cert_path)?;
        let cert = <x509_cert::Certificate as der::DecodePem>::from_pem(&pem)?;
        let cert_spki_der = cert.tbs_certificate.subject_public_key_info.to_der()?;
        if cert_spki_der != pub_spki_der {
            return Err(format!(
                "{label} cert at {} does not match its companion key on disk; \
                 delete the .dev-certs/ directory to regenerate the chain",
                cert_path.display(),
            )
            .into());
        }
        Ok(cert.to_der()?)
    } else {
        info!("Generating new {label} cert");
        let cert_der = build()?;
        write_cert_pem_atomic(&cert_der, cert_path)?;
        info!("{label} cert written to {}", cert_path.display());
        Ok(cert_der)
    }
}

/// Load or generate the persistent ARK (self-signed root). Bundles the
/// signing key, the resulting cert DER, and parsed metadata for downstream use.
fn load_or_create_ark(
    cert_dir: &Path,
) -> Result<PersistedCa, Box<dyn std::error::Error + Send + Sync>> {
    let key_path = cert_dir.join("ark.key");
    let cert_path = cert_dir.join("ark.pem");

    let key = load_or_generate_rsa_key(&key_path, "ARK")?;
    let signer = rsa::pss::SigningKey::<rsa_sha2::Sha384>::new(key.clone());

    let pub_spki_der = rsa::pkcs8::EncodePublicKey::to_public_key_der(&key.to_public_key())?
        .as_bytes()
        .to_vec();
    let spki = SubjectPublicKeyInfoOwned::from_der(&pub_spki_der)?;

    let subject = x509_cert::name::Name::from_str("CN=Local Dev Root CA")?;
    let key_id = key_id_from_spki(&spki);

    let subject_for_build = subject.clone();
    let spki_for_build = spki.clone();
    let signer_for_build = signer.clone();
    let key_id_for_build = key_id.clone();
    let cert_der = load_or_create_pem("ARK", &cert_path, &pub_spki_der, move || {
        // 10-year window pinned at generation time. Stays inside UtcTime range
        // (≤2049) until late 2039; that's longer than this dev tool will live.
        let validity = validity_from_now(Duration::from_secs(10 * 365 * 24 * 60 * 60))?;
        // Self-signed root: AKI references our own SKI.
        let extensions = ca_extensions(&key_id_for_build, &key_id_for_build)?;
        build_rsa_pss_cert(
            1,
            &subject_for_build,
            &subject_for_build,
            spki_for_build,
            &signer_for_build,
            validity,
            Some(extensions),
        )
    })?;

    Ok(PersistedCa {
        signer,
        cert_der,
        subject,
        key_id,
    })
}

/// Load or generate the persistent ASK (signed by ARK).
fn load_or_create_ask(
    cert_dir: &Path,
    ark: &PersistedCa,
) -> Result<PersistedCa, Box<dyn std::error::Error + Send + Sync>> {
    let key_path = cert_dir.join("ask.key");
    let cert_path = cert_dir.join("ask.pem");

    let key = load_or_generate_rsa_key(&key_path, "ASK")?;
    let signer = rsa::pss::SigningKey::<rsa_sha2::Sha384>::new(key.clone());

    let pub_spki_der = rsa::pkcs8::EncodePublicKey::to_public_key_der(&key.to_public_key())?
        .as_bytes()
        .to_vec();
    let spki = SubjectPublicKeyInfoOwned::from_der(&pub_spki_der)?;

    // ASK uses the same DN as ARK (the `sev` crate's chain verifier doesn't
    // check names, and matching them keeps the chain visually simple).
    let subject = ark.subject.clone();
    let key_id = key_id_from_spki(&spki);

    let ark_subject = ark.subject.clone();
    let subject_for_build = subject.clone();
    let spki_for_build = spki.clone();
    let ark_signer = ark.signer.clone();
    let ark_key_id = ark.key_id.clone();
    let key_id_for_build = key_id.clone();
    let cert_der = load_or_create_pem("ASK", &cert_path, &pub_spki_der, move || {
        let validity = validity_from_now(Duration::from_secs(10 * 365 * 24 * 60 * 60))?;
        let extensions = ca_extensions(&key_id_for_build, &ark_key_id)?;
        build_rsa_pss_cert(
            2,
            &ark_subject,
            &subject_for_build,
            spki_for_build,
            &ark_signer,
            validity,
            Some(extensions),
        )
    })?;

    Ok(PersistedCa {
        signer,
        cert_der,
        subject,
        key_id,
    })
}

/// Load or generate the persistent TLS-CA — a self-signed PKCS#1 v1.5 root
/// used *only* to sign the ephemeral TLS leaf. This is what developers trust
/// in their OS keychain. It is intentionally unrelated to the ARK so the SEV
/// chain (which the `sev` crate requires to be PSS) doesn't drag PSS into
/// the TLS path, where Apple's Security framework refuses to handle it.
fn load_or_create_tls_ca(
    cert_dir: &Path,
) -> Result<PersistedTlsCa, Box<dyn std::error::Error + Send + Sync>> {
    let key_path = cert_dir.join("tls-ca.key");
    let cert_path = cert_dir.join("tls-ca.pem");

    let key = load_or_generate_rsa_key(&key_path, "TLS-CA")?;
    let signer = rsa::pkcs1v15::SigningKey::<rsa_sha2::Sha384>::new(key.clone());

    let pub_spki_der = rsa::pkcs8::EncodePublicKey::to_public_key_der(&key.to_public_key())?
        .as_bytes()
        .to_vec();
    let spki = SubjectPublicKeyInfoOwned::from_der(&pub_spki_der)?;

    let subject = x509_cert::name::Name::from_str("CN=Local Dev TLS Root")?;
    let key_id = key_id_from_spki(&spki);

    let subject_for_build = subject.clone();
    let spki_for_build = spki.clone();
    let signer_for_build = signer.clone();
    let key_id_for_build = key_id.clone();
    let _cert_der = load_or_create_pem("TLS-CA", &cert_path, &pub_spki_der, move || {
        // 10-year window pinned at generation time. Roots are exempt from
        // Apple's 398-day cap on leaves, so a long window is fine.
        let validity = validity_from_now(Duration::from_secs(10 * 365 * 24 * 60 * 60))?;
        let extensions = ca_extensions(&key_id_for_build, &key_id_for_build)?;
        build_rsa_pkcs1_cert(
            10,
            &subject_for_build,
            &subject_for_build,
            spki_for_build,
            &signer_for_build,
            validity,
            Some(extensions),
        )
    })?;

    Ok(PersistedTlsCa {
        signer,
        subject,
        key_id,
    })
}

/// Generate an ephemeral TLS leaf signed by the TLS-CA using RSA PKCS#1 v1.5
/// SHA-384. Returns `(leaf_der, leaf_private_key_der)`. The leaf's SPKI is a
/// fresh ECDSA P-256 key generated for this boot only.
fn build_tls_leaf(
    tls_ca: &PersistedTlsCa,
    dns_names: &[&str],
) -> Result<
    (Vec<u8>, rustls::pki_types::PrivateKeyDer<'static>),
    Box<dyn std::error::Error + Send + Sync>,
> {
    let tls_alg = &rcgen::PKCS_ECDSA_P256_SHA256;
    let tls_key = KeyPair::generate_for(tls_alg)?;
    // rcgen 0.14 only exposes the SPKI as PEM; decode it back to DER for x509-cert.
    let tls_pub_pem = tls_key.public_key_pem();
    let tls_spki = <SubjectPublicKeyInfoOwned as der::DecodePem>::from_pem(&tls_pub_pem)?;

    let leaf_subject = x509_cert::name::Name::from_str("CN=tinfoil-shim-mock")?;
    let leaf_key_id = key_id_from_spki(&tls_spki);

    // 365 days — well under Apple's 398-day cap on TLS server certs.
    let validity = validity_from_now(Duration::from_secs(365 * 24 * 60 * 60))?;
    let cert_der = build_rsa_pkcs1_cert(
        11,
        &tls_ca.subject,
        &leaf_subject,
        tls_spki,
        &tls_ca.signer,
        validity,
        Some(tls_leaf_extensions(
            dns_names,
            &leaf_key_id,
            &tls_ca.key_id,
        )?),
    )?;

    let key_der = rustls::pki_types::PrivateKeyDer::try_from(tls_key.serialize_der())
        .map_err(|e| format!("invalid TLS private key DER: {e}"))?;

    Ok((cert_der, key_der))
}

// ── Main ────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider())
        .expect("failed to install rustls crypto provider");

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("tinfoil_shim_mock=info".parse().unwrap()),
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

    // ── 1. Load or generate the persistent ARK + ASK ────────────────────
    //
    // The on-disk pair under `cert_dir` is the source of trust. If it
    // already exists we load it unchanged so the developer's keychain trust
    // (or `--hardware-root-ca` config) survives restarts. The shim never
    // overwrites existing files.

    fs::create_dir_all(&cert_dir)?;
    let cert_dir_path = Path::new(&cert_dir);
    let ark = load_or_create_ark(cert_dir_path)?;
    let ask = load_or_create_ask(cert_dir_path, &ark)?;
    let tls_ca = load_or_create_tls_ca(cert_dir_path)?;

    // ── 2. Generate the ephemeral VCEK (signed by ASK) ──────────────────

    let vcek_signing_key = p384::ecdsa::SigningKey::random(&mut rand_core::OsRng);
    let vcek_pub_key = p384::PublicKey::from(vcek_signing_key.verifying_key());
    let vcek_pub_spki_der = p384::pkcs8::EncodePublicKey::to_public_key_der(&vcek_pub_key)?;
    let vcek_spki = SubjectPublicKeyInfoOwned::from_der(vcek_pub_spki_der.as_ref())?;

    // VCEK is ephemeral and only used in the SEV chain (the `sev` crate
    // doesn't enforce strict validity policy), so a generous 5-year window
    // from boot is fine. We give it an AKI referencing ASK so chain
    // builders that follow SKI/AKI links work end-to-end.
    let vcek_validity = validity_from_now(Duration::from_secs(5 * 365 * 24 * 60 * 60))?;
    let vcek_extensions = vec![authority_key_id_ext(&ask.key_id)?];
    let vcek_der = build_rsa_pss_cert(
        3,
        &ask.subject,
        &ask.subject,
        vcek_spki,
        &ask.signer,
        vcek_validity,
        Some(vcek_extensions),
    )?;

    // ── 3. Self-check: verify the ARK → ASK → VCEK chain ────────────────

    {
        use sev::certs::snp::{Certificate as SevCert, Chain, Verifiable, ca};

        let sev_ark = SevCert::from_der(&ark.cert_der).expect("self-check: failed to parse ARK");
        let sev_ask = SevCert::from_der(&ask.cert_der).expect("self-check: failed to parse ASK");
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

    // ── 4. Build the ephemeral TLS leaf, signed by TLS-CA (PKCS#1 v1.5) ─
    //
    // The leaf is rooted in the persistent TLS-CA, so a developer who has
    // trusted `tls-ca.pem` in their OS keychain doesn't need to do anything
    // when the shim restarts and rotates the leaf. ARK is *not* used for
    // TLS — Apple's Security framework refuses to handle PSS in chain
    // validation, so we keep the SEV-SNP and TLS chains fully separate.

    let (tls_cert_der, tls_key_der) = build_tls_leaf(&tls_ca, &["localhost", "server", "shim"])?;
    let cert_der = rustls::pki_types::CertificateDer::from(tls_cert_der.clone());
    let key_der = tls_key_der;

    // ── 5. Compute TLS fingerprint ──────────────────────────────────────

    let parsed_tls = x509_cert::Certificate::from_der(&tls_cert_der)?;
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

    // V3 attestation: raw report + VCEK. The mock deliberately does not
    // ship ARK/ASK in the attestation document — those are anchors of
    // trust and must be configured out-of-band on the verifier (via
    // `trusted_ark_der` / `trusted_ask_der`), never sourced from the
    // attested endpoint itself. The on-disk ark.pem / ask.pem files
    // generated above are what the developer feeds to their verifier
    // configuration; this endpoint only emits the dynamic per-boot data.
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
