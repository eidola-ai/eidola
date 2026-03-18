use std::sync::Arc;

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, Error as TlsError, SignatureScheme};
use sha2::{Digest, Sha512};
use x509_parser::oid_registry::Oid;
use x509_parser::prelude::*;

/// OID for the PHALA_RATLS_ATTESTATION X.509 extension (newer format).
/// Contains the SCALE-encoded `VersionedAttestation`.
/// 1.3.6.1.4.1.62397.1.8
const ATTESTATION_OID: &[u64] = &[1, 3, 6, 1, 4, 1, 62397, 1, 8];

/// OID for the legacy TDX quote extension.
/// 1.3.6.1.4.1.62397.1.1
const TDX_QUOTE_OID: &[u64] = &[1, 3, 6, 1, 4, 1, 62397, 1, 1];

/// OID for the legacy TDX event log extension (JSON-encoded).
/// 1.3.6.1.4.1.62397.1.2
const EVENT_LOG_OID: &[u64] = &[1, 3, 6, 1, 4, 1, 62397, 1, 2];

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
    pub fn new(inner: Arc<dyn ServerCertVerifier>, trusted_compose_hashes: Vec<String>) -> Self {
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
        self.inner.verify_server_cert(
            end_entity,
            intermediates,
            server_name,
            ocsp_response,
            now,
        )?;

        // 2. Parse the leaf certificate and extract attestation.
        let (_, cert) = X509Certificate::from_der(end_entity.as_ref())
            .map_err(|e| TlsError::General(format!("failed to parse leaf certificate: {e}")))?;

        // 3. Extract attestation — try the newer .1.8 OID first, fall back to
        //    legacy .1.1 (TDX quote) + .1.2 (event log) for backward compat.
        let attestation = extract_attestation(&cert)?;

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

/// Extract and decode attestation from the certificate.
///
/// Tries the newer PHALA_RATLS_ATTESTATION extension (.1.8) first. If absent,
/// falls back to the legacy TDX quote (.1.1) + event log (.1.2) extensions,
/// matching the backward-compat path in dstack's `ra-tls` crate.
fn extract_attestation(cert: &X509Certificate<'_>) -> Result<Attestation, TlsError> {
    // Try newer .1.8 extension first.
    let att_oid = Oid::from(ATTESTATION_OID)
        .map_err(|_| TlsError::General("invalid attestation OID".into()))?;
    if let Some(ext) = cert
        .get_extension_unique(&att_oid)
        .map_err(|e| TlsError::General(format!("error reading attestation extension: {e}")))?
    {
        let bytes = parse_der_octet_string(ext.value)?;
        return decode_attestation(&bytes);
    }

    // Legacy fallback: build Attestation from TDX quote + JSON event log.
    let quote_oid =
        Oid::from(TDX_QUOTE_OID).map_err(|_| TlsError::General("invalid TDX quote OID".into()))?;
    let quote_ext = cert
        .get_extension_unique(&quote_oid)
        .map_err(|e| TlsError::General(format!("error reading TDX quote extension: {e}")))?
        .ok_or_else(|| {
            TlsError::General(
                "leaf certificate has no attestation: missing both \
                 PHALA_RATLS_ATTESTATION (.1.8) and PHALA_RATLS_TDX_QUOTE (.1.1)"
                    .into(),
            )
        })?;
    let tdx_quote_bytes = parse_der_octet_string(quote_ext.value)?;

    let log_oid =
        Oid::from(EVENT_LOG_OID).map_err(|_| TlsError::General("invalid event log OID".into()))?;
    let log_ext = cert
        .get_extension_unique(&log_oid)
        .map_err(|e| TlsError::General(format!("error reading event log extension: {e}")))?
        .ok_or_else(|| {
            TlsError::General("TDX quote present but event log extension (.1.2) missing".into())
        })?;
    let log_bytes = parse_der_octet_string(log_ext.value)?;

    attestation_from_legacy(&tdx_quote_bytes, &log_bytes)
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
    let versioned = VersionedAttestation::decode(&mut &scale_bytes[..]).map_err(|e| {
        TlsError::General(format!("failed to SCALE-decode VersionedAttestation: {e}"))
    })?;
    let VersionedAttestation::V0 { attestation } = versioned;
    Ok(attestation)
}

/// Event type constant for dstack runtime events (matches cc-eventlog).
const DSTACK_RUNTIME_EVENT_TYPE: u32 = 0x08000001;

/// Byte offset of report_data within a raw TDX quote (version 4/5).
/// Header (48 bytes) + report body fields before report_data (520 bytes).
const TDX_QUOTE_REPORT_DATA_OFFSET: usize = 568;
const TDX_QUOTE_REPORT_DATA_LEN: usize = 64;

/// JSON representation of a TDX event log entry, matching the dstack
/// `cc-eventlog` crate's `TdxEvent` (with `serde_human_bytes` hex encoding).
#[derive(serde::Deserialize)]
struct JsonTdxEvent {
    event_type: u32,
    event: String,
    /// Hex-encoded bytes (serde_human_bytes uses hex for human-readable).
    event_payload: String,
}

/// Build an `Attestation` from legacy cert extensions (TDX quote + JSON event log).
fn attestation_from_legacy(
    tdx_quote_bytes: &[u8],
    event_log_json: &[u8],
) -> Result<Attestation, TlsError> {
    // Extract report_data from the raw TDX quote at a fixed offset.
    let rd_end = TDX_QUOTE_REPORT_DATA_OFFSET + TDX_QUOTE_REPORT_DATA_LEN;
    if tdx_quote_bytes.len() < rd_end {
        return Err(TlsError::General(format!(
            "TDX quote too short ({} bytes, need at least {rd_end})",
            tdx_quote_bytes.len()
        )));
    }
    let mut report_data = [0u8; 64];
    report_data.copy_from_slice(&tdx_quote_bytes[TDX_QUOTE_REPORT_DATA_OFFSET..rd_end]);

    // Parse JSON event log and filter for runtime events.
    let tdx_events: Vec<JsonTdxEvent> = serde_json::from_slice(event_log_json)
        .map_err(|e| TlsError::General(format!("failed to parse TDX event log JSON: {e}")))?;

    let runtime_events: Vec<RuntimeEvent> = tdx_events
        .into_iter()
        .filter(|ev| ev.event_type == DSTACK_RUNTIME_EVENT_TYPE)
        .map(|ev| {
            let payload = hex_decode(&ev.event_payload)?;
            Ok(RuntimeEvent {
                event: ev.event,
                payload,
            })
        })
        .collect::<Result<Vec<_>, TlsError>>()?;

    Ok(Attestation {
        quote: AttestationQuote::DstackTdx(TdxQuote {
            quote: tdx_quote_bytes.to_vec(),
            event_log: vec![],
        }),
        runtime_events,
        report_data,
        config: String::new(),
        report: (),
    })
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

fn hex_decode(s: &str) -> Result<Vec<u8>, TlsError> {
    if s.len() % 2 != 0 {
        return Err(TlsError::General("odd-length hex string".into()));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16)
                .map_err(|e| TlsError::General(format!("invalid hex in event payload: {e}")))
        })
        .collect()
}
