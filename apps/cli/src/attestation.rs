use std::sync::Arc;

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, Error as TlsError, SignatureScheme};
use sha2::{Digest, Sha512};
use x509_parser::oid_registry::Oid;
use x509_parser::prelude::*;

/// OID for the PHALA_RATLS_ATTESTATION X.509 extension.
/// 1.3.6.1.4.1.62397.1.8
const ATTESTATION_OID: &[u64] = &[1, 3, 6, 1, 4, 1, 62397, 1, 8];

/// Custom TLS certificate verifier that checks:
/// 1. Standard WebPKI chain validation against a pinned CA
/// 2. RA-TLS attestation: report_data binding + compose_hash allowlist
pub struct AttestationVerifier {
    /// Delegates chain validation to this standard verifier.
    inner: Arc<dyn ServerCertVerifier>,
    /// Hex-encoded compose hashes that are trusted.
    trusted_compose_hashes: Vec<String>,
}

impl AttestationVerifier {
    pub fn new(
        inner: Arc<dyn ServerCertVerifier>,
        trusted_compose_hashes: Vec<String>,
    ) -> Self {
        Self {
            inner,
            trusted_compose_hashes,
        }
    }
}

impl std::fmt::Debug for AttestationVerifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttestationVerifier")
            .field("trusted_compose_hashes", &self.trusted_compose_hashes)
            .finish()
    }
}

impl ServerCertVerifier for AttestationVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        ocsp_response: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        // 1. Delegate chain validation to the inner (WebPKI) verifier.
        self.inner
            .verify_server_cert(end_entity, intermediates, server_name, ocsp_response, now)?;

        // 2. Parse the leaf certificate and extract attestation.
        let (_, cert) = X509Certificate::from_der(end_entity.as_ref()).map_err(|e| {
            TlsError::General(format!("failed to parse leaf certificate: {e}"))
        })?;

        let attestation_bytes = extract_attestation_bytes(&cert)?;

        // 3. SCALE-decode the VersionedAttestation.
        let attestation = decode_attestation(&attestation_bytes)?;

        // 4. Verify report_data binds the attestation to this TLS key.
        //    report_data == SHA-512("ratls-cert:" || leaf_public_key_der)
        let pubkey_der = cert.public_key().raw;
        let expected_report_data = {
            let mut hasher = Sha512::new();
            hasher.update(b"ratls-cert:");
            hasher.update(pubkey_der);
            let hash: [u8; 64] = hasher.finalize().into();
            hash
        };
        if attestation.report_data != expected_report_data {
            return Err(TlsError::General(
                "attestation report_data does not bind to the leaf TLS key".into(),
            ));
        }

        // 5. Extract compose_hash from runtime events and check the allowlist.
        let compose_hash = find_compose_hash(&attestation.runtime_events)?;
        let compose_hash_hex = hex_encode(&compose_hash);

        if !self
            .trusted_compose_hashes
            .iter()
            .any(|h| h.eq_ignore_ascii_case(&compose_hash_hex))
        {
            return Err(TlsError::General(format!(
                "compose_hash {compose_hash_hex} is not in the trusted set"
            )));
        }

        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        self.inner.verify_tls12_signature(message, cert, dss)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        self.inner.verify_tls13_signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.inner.supported_verify_schemes()
    }
}

// ---------------------------------------------------------------------------
// Attestation extraction helpers
// ---------------------------------------------------------------------------

/// Extract the raw attestation bytes from the PHALA_RATLS_ATTESTATION extension.
///
/// The extension value in x509-parser is the content of the extnValue OCTET
/// STRING. Inside that, dstack stores a DER-encoded OCTET STRING wrapping the
/// SCALE-encoded `VersionedAttestation`.
fn extract_attestation_bytes(cert: &X509Certificate<'_>) -> Result<Vec<u8>, TlsError> {
    let oid = Oid::from(ATTESTATION_OID)
        .map_err(|_| TlsError::General("invalid attestation OID".into()))?;

    let ext = cert
        .get_extension_unique(&oid)
        .map_err(|e| TlsError::General(format!("error reading attestation extension: {e}")))?
        .ok_or_else(|| {
            TlsError::General("leaf certificate missing PHALA_RATLS_ATTESTATION extension".into())
        })?;

    // The extension value is DER: an OCTET STRING wrapping the payload.
    // Parse the outer OCTET STRING to get the raw SCALE bytes.
    let value = ext.value;
    parse_der_octet_string(value)
}

/// Minimal DER OCTET STRING parser (tag 0x04).
fn parse_der_octet_string(data: &[u8]) -> Result<Vec<u8>, TlsError> {
    if data.is_empty() {
        return Err(TlsError::General("empty DER data".into()));
    }
    if data[0] != 0x04 {
        return Err(TlsError::General(format!(
            "expected DER OCTET STRING (tag 0x04), got 0x{:02x}",
            data[0]
        )));
    }
    let (content, _consumed) = parse_der_length(&data[1..])?;
    Ok(content.to_vec())
}

/// Parse a DER length and return (content_slice, bytes_consumed_for_length).
fn parse_der_length(data: &[u8]) -> Result<(&[u8], usize), TlsError> {
    if data.is_empty() {
        return Err(TlsError::General("truncated DER length".into()));
    }
    let first = data[0];
    if first < 0x80 {
        // Short form
        let len = first as usize;
        if 1 + len > data.len() {
            return Err(TlsError::General("DER content truncated".into()));
        }
        Ok((&data[1..1 + len], 1 + len))
    } else {
        let num_bytes = (first & 0x7f) as usize;
        if num_bytes == 0 || num_bytes > 4 {
            return Err(TlsError::General("unsupported DER length encoding".into()));
        }
        if 1 + num_bytes > data.len() {
            return Err(TlsError::General("truncated DER multi-byte length".into()));
        }
        let mut len: usize = 0;
        for &b in &data[1..1 + num_bytes] {
            len = (len << 8) | b as usize;
        }
        let start = 1 + num_bytes;
        if start + len > data.len() {
            return Err(TlsError::General("DER content truncated".into()));
        }
        Ok((&data[start..start + len], start + len))
    }
}

// ---------------------------------------------------------------------------
// Minimal SCALE decoding of dstack attestation types
//
// These mirror the upstream types at the same git commit (31cfd48) so the
// binary layout is identical. We only derive `Decode` — we never encode.
// ---------------------------------------------------------------------------

use parity_scale_codec::Decode;

/// Mirrors `dstack_attest::attestation::VersionedAttestation`.
#[derive(Decode)]
enum VersionedAttestation {
    V0 { attestation: Attestation },
}

/// Mirrors `dstack_attest::attestation::Attestation<()>`.
#[derive(Decode)]
struct Attestation {
    #[allow(dead_code)]
    quote: AttestationQuote,
    runtime_events: Vec<RuntimeEvent>,
    report_data: [u8; 64],
    #[allow(dead_code)]
    config: String,
    #[allow(dead_code)]
    report: (),
}

/// Mirrors `dstack_attest::attestation::AttestationQuote`.
#[derive(Decode)]
#[allow(dead_code)]
enum AttestationQuote {
    DstackTdx(TdxQuote),
    DstackGcpTdx,
    DstackNitroEnclave,
}

/// Mirrors `dstack_attest::attestation::TdxQuote`.
#[derive(Decode)]
#[allow(dead_code)]
struct TdxQuote {
    quote: Vec<u8>,
    event_log: Vec<TdxEvent>,
}

/// Mirrors `cc_eventlog::TdxEvent`.
#[derive(Decode)]
#[allow(dead_code)]
struct TdxEvent {
    imr: u32,
    event_type: u32,
    digest: Vec<u8>,
    event: String,
    event_payload: Vec<u8>,
}

/// Mirrors `cc_eventlog::RuntimeEvent`.
#[derive(Decode)]
struct RuntimeEvent {
    event: String,
    payload: Vec<u8>,
}

fn decode_attestation(scale_bytes: &[u8]) -> Result<Attestation, TlsError> {
    let versioned =
        VersionedAttestation::decode(&mut &scale_bytes[..]).map_err(|e| {
            TlsError::General(format!("failed to SCALE-decode VersionedAttestation: {e}"))
        })?;
    let VersionedAttestation::V0 { attestation } = versioned;
    Ok(attestation)
}

fn find_compose_hash(events: &[RuntimeEvent]) -> Result<Vec<u8>, TlsError> {
    for event in events {
        if event.event == "system-ready" {
            break;
        }
        if event.event == "compose-hash" {
            return Ok(event.payload.clone());
        }
    }
    Err(TlsError::General(
        "compose-hash event not found in attestation runtime events".into(),
    ))
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
