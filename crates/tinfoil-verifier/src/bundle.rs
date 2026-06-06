use der::{Decode, Encode};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use signature::hazmat::PrehashVerifier;
use std::io::Read;

use crate::{Error, sevsnp};

/// Full attestation bundle containing report, VCEK, and enclave certificate.
///
/// Fetched from the Tinfoil attestation transparency service (ATC). ATC is
/// only consulted as a fallback when the enclave's own self-contained v3
/// well-known document is missing required elements.
///
/// Note: this struct intentionally does **not** carry `ark` / `ask` fields
/// even though the wire format includes them. AMD root and intermediate
/// certificates are anchors of trust and must come from a statically known
/// source (the built-in Genoa ARK/ASK in the `sev` crate, or an explicit
/// `trusted_ark_der` / `trusted_ask_der` configured by the caller), never
/// from a third-party service like ATC. Keeping unused-but-deserialized
/// `ark` / `ask` fields here would be a foot-gun: they would sit in the
/// struct waiting for someone to wire them into the trust chain in a later
/// edit and silently re-introduce the very class of bug we're trying to
/// rule out. Serde drops unknown fields silently, so simply omitting them
/// from this struct is enough to make them unreachable from the verifier.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct AttestationBundle {
    pub domain: String,
    #[serde(rename = "enclaveAttestationReport")]
    pub enclave_attestation_report: AtcReport,
    #[serde(rename = "enclaveCert")]
    pub enclave_cert: String,
    pub vcek: String,
    pub digest: String,
    #[serde(rename = "sigstoreBundle")]
    pub sigstore_bundle: Option<serde_json::Value>,
}

/// Inner attestation report element of an ATC bundle. The body is base64-encoded
/// gzip-compressed attestation report bytes; the format string identifies the
/// platform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtcReport {
    pub format: String,
    pub body: String,
}

/// Fresh, nonce-bound attestation document served from
/// `/.well-known/tinfoil-attestation?nonce=<64 hex chars>`.
///
/// The enclave collects a *fresh* hardware report on every request, binding
/// the caller's random `nonce` (plus the TLS key fingerprint and HPKE key)
/// into the report's `REPORT_DATA` field and signing the whole JSON with its
/// TLS leaf private key. This is the format that replaced the old static
/// `?v=3` document and is what makes the attestation replay-proof: a captured
/// document is worthless against a different nonce, and forging a matching
/// report requires the genuine SEV-SNP / TDX hardware.
///
/// Field order mirrors the upstream Go `attestation.Attestation` struct
/// (`tinfoil/internal/attestation/attestation.go`). `cpu.report` is
/// base64-encoded raw report bytes (NOT gzipped, unlike the legacy ATC
/// bundle). `vcek` is **not** part of the upstream fresh document — Tinfoil
/// builds it without one, so we treat it as optional and fall back to ATC
/// when absent. The local shim mock *does* include it so tests need no ATC.
#[derive(Debug, Clone, Deserialize)]
pub struct AttestationDocumentV3 {
    pub format: String,
    pub report_data: ReportDataFields,
    pub cpu: AttestationCPU,
    #[serde(default)]
    pub vcek: Option<String>,
    /// PEM-encoded TLS leaf certificate the enclave signed this document with.
    pub certificate: String,
    /// Base64-encoded ASN.1/DER ECDSA signature over the document with the
    /// `signature` field blanked. Verified against `certificate`'s public key.
    pub signature: String,
}

/// The `report_data` object of a fresh attestation document. Each field is the
/// hex encoding of the corresponding input to the `REPORT_DATA` hash. The GPU
/// and NVSwitch evidence hashes are absent for CPU-only enclaves (the Tinfoil
/// inference router we target).
#[derive(Debug, Clone, Deserialize)]
pub struct ReportDataFields {
    /// SHA-256 of the TLS leaf's SubjectPublicKeyInfo (DER). Equals the
    /// verifier's own `sha256(SPKI(peer_cert))`.
    pub tls_key_fp: String,
    pub hpke_key: String,
    pub nonce: String,
    #[serde(default)]
    pub gpu_evidence_hash: Option<String>,
    #[serde(default)]
    pub nvswitch_evidence_hash: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AttestationCPU {
    pub platform: String,
    pub report: String, // base64-encoded raw report (NOT gzipped)
}

/// TEE platform identified from the attestation document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    SevSnp,
    Tdx,
}

/// Decoded `report_data` inputs, ready to recompute the `REPORT_DATA` hash and
/// cross-check against the hardware report.
#[derive(Debug, Clone)]
pub struct ReportData {
    /// SHA-256 of the TLS leaf SPKI (32 bytes).
    pub tls_key_fp: [u8; 32],
    pub hpke_key: Vec<u8>,
    pub nonce: Vec<u8>,
    /// Empty when the enclave reports no GPU evidence.
    pub gpu_evidence_hash: Vec<u8>,
    /// Empty when the enclave reports no NVSwitch evidence.
    pub nvswitch_evidence_hash: Vec<u8>,
}

impl ReportData {
    /// Recompute the 64-byte `REPORT_DATA` value the hardware should carry:
    ///
    /// ```text
    /// SHA-256(tls_key_fp || hpke_key || nonce || gpu_hash || nvswitch_hash)
    /// ```
    ///
    /// padded to 64 bytes with trailing zeros. Matches upstream
    /// `attestation.ComputeReportData`.
    pub fn expected_report_data(&self) -> [u8; 64] {
        let mut h = Sha256::new();
        h.update(self.tls_key_fp);
        h.update(&self.hpke_key);
        h.update(&self.nonce);
        h.update(&self.gpu_evidence_hash);
        h.update(&self.nvswitch_evidence_hash);
        let digest = h.finalize();
        let mut out = [0u8; 64];
        out[..32].copy_from_slice(&digest);
        out
    }
}

/// Unified attestation data extracted from a fresh well-known document, with
/// everything the per-handshake verifier needs already decoded.
pub struct ResolvedAttestation {
    pub platform: Platform,
    pub report_bytes: Vec<u8>,
    /// Present only when the document self-carries a VCEK (the shim mock); the
    /// production fresh document omits it and the verifier backfills via ATC.
    pub vcek_der: Option<Vec<u8>>,
    /// Decoded `report_data` inputs for the `REPORT_DATA` cross-check.
    pub report_data: ReportData,
    /// DER bytes of the TLS leaf certificate the document was signed with.
    pub certificate_der: Vec<u8>,
    /// Exact bytes the enclave signed (the served document with the
    /// `signature` value blanked). Verified against `signature_der`.
    pub signed_payload: Vec<u8>,
    /// Decoded ASN.1/DER ECDSA signature over `signed_payload`.
    pub signature_der: Vec<u8>,
}

/// ATC bundle format strings identifying the platform of the nested report.
pub(crate) const ATC_SNP_FORMAT: &str = "https://tinfoil.sh/predicate/sev-snp-guest/v2";
pub(crate) const ATC_TDX_FORMAT: &str = "https://tinfoil.sh/predicate/tdx-guest/v2";
pub(crate) const V3_FORMAT: &str = "https://tinfoil.sh/predicate/attestation/v3";

/// Length in bytes of the per-handshake attestation nonce. Fixed by the
/// upstream endpoint, which rejects anything other than exactly 32 bytes
/// (64 hex chars).
pub(crate) const NONCE_LEN: usize = 32;

/// Generate a fresh 32-byte attestation nonce from the OS CSPRNG.
pub(crate) fn random_nonce() -> Result<[u8; NONCE_LEN], Error> {
    let mut nonce = [0u8; NONCE_LEN];
    getrandom::fill(&mut nonce)
        .map_err(|e| Error::Connector(format!("failed to generate attestation nonce: {e}")))?;
    Ok(nonce)
}

/// Named-curve OIDs for the two ECDSA curves the enclave's TLS leaf may use:
/// production is P-384, the local shim mock is P-256.
const OID_P256: spki::ObjectIdentifier = spki::ObjectIdentifier::new_unwrap("1.2.840.10045.3.1.7");
const OID_P384: spki::ObjectIdentifier = spki::ObjectIdentifier::new_unwrap("1.3.132.0.34");

/// Default Tinfoil attestation transparency service URL.
pub const DEFAULT_ATC_URL: &str = "https://atc.tinfoil.sh/attestation";

/// Request body for the ATC `POST /attestation` endpoint.
#[derive(Debug, Serialize)]
struct AtcRequest<'a> {
    #[serde(rename = "enclaveUrl")]
    enclave_url: &'a str,
    repo: &'a str,
}

/// Fetch the full attestation bundle from the Tinfoil ATC service.
///
/// Issues a `POST /attestation` with `{enclaveUrl, repo}` so ATC returns an
/// attestation bundle bound to the specific enclave the caller intends to
/// connect to and the source repository it should match. (The legacy
/// parameterless `GET /attestation` returns whatever the default router is
/// pointing at, which is not necessarily the enclave we'll talk to.)
///
/// The bundle contains the attestation report, VCEK certificate, and enclave
/// TLS certificate — everything needed for verification.
pub async fn fetch_bundle(
    atc_url: Option<&str>,
    enclave_url: &str,
    repo: &str,
    tls_roots: &rustls::RootCertStore,
) -> Result<AttestationBundle, Error> {
    let url = atc_url.unwrap_or(DEFAULT_ATC_URL);
    // Cloning the store is cheap-ish (a Vec of trust anchors). The verifier
    // is the only caller and reuses the same store across all of its
    // outbound HTTPS, so the clone happens at most a few times per process.
    let tls_config = rustls::ClientConfig::builder()
        .with_root_certificates(tls_roots.clone())
        .with_no_client_auth();

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(30))
        .tls_backend_preconfigured(tls_config)
        .build()
        .map_err(|e| Error::Bundle(format!("failed to build HTTP client: {e}")))?;
    let bundle: AttestationBundle = client
        .post(url)
        .json(&AtcRequest { enclave_url, repo })
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let format = &bundle.enclave_attestation_report.format;
    if format != ATC_SNP_FORMAT && format != ATC_TDX_FORMAT {
        return Err(Error::Bundle(format!(
            "unsupported ATC bundle format: {format}",
        )));
    }

    Ok(bundle)
}

/// Fetch and resolve a fresh nonce-bound attestation document out-of-band.
///
/// Generates a random nonce, requests `?nonce=<hex>`, parses the document, and
/// verifies the two integrity properties that don't require the hardware
/// report: the echoed nonce matches what we sent (freshness), and the
/// document's ECDSA signature validates against its embedded TLS certificate.
/// The hardware report verification (VCEK chain, TCB, measurement) and the
/// `REPORT_DATA` cross-check remain the caller's responsibility, since they are
/// platform-specific. The per-handshake connector path does not use this — it
/// drives the request inline over the attested connection (see
/// `attesting_client`) — but it is convenient for out-of-band tooling.
pub async fn fetch_well_known(
    client: &reqwest::Client,
    attestation_url: &str,
) -> Result<ResolvedAttestation, Error> {
    let nonce = random_nonce()?;
    let url = format!("{attestation_url}?nonce={}", hex::encode(nonce));
    let raw = client
        .get(&url)
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    let resolved = parse_document(&raw)?;
    if resolved.report_data.nonce != nonce {
        return Err(Error::NonceMismatch {
            sent: hex::encode(nonce),
            echoed: hex::encode(&resolved.report_data.nonce),
        });
    }
    verify_document_signature(&resolved)?;
    tracing::info!(
        "Fetched fresh attestation document (platform: {:?})",
        resolved.platform
    );
    Ok(resolved)
}

/// Parse the raw JSON bytes of a fresh attestation document into a
/// [`ResolvedAttestation`], decoding every field the verifier needs.
///
/// Takes the raw bytes (not a parsed struct) because the document signature is
/// computed over the exact serialized form with the `signature` value blanked,
/// and reconstructing that byte-for-byte from a typed struct is fragile across
/// upstream field-order or field-addition changes. Instead we blank the
/// `signature` value in place (see [`signed_payload`]), which preserves any
/// field we don't model.
pub fn parse_document(raw: &[u8]) -> Result<ResolvedAttestation, Error> {
    let doc: AttestationDocumentV3 = serde_json::from_slice(raw)
        .map_err(|e| Error::Connector(format!("attestation JSON parse: {e}")))?;
    if doc.format != V3_FORMAT {
        return Err(Error::Bundle(format!(
            "unexpected attestation document format: {}",
            doc.format
        )));
    }
    let platform = match doc.cpu.platform.as_str() {
        "tdx" => Platform::Tdx,
        "sev-snp" => Platform::SevSnp,
        other => {
            return Err(Error::Bundle(format!(
                "unsupported attestation platform: {other}"
            )));
        }
    };

    let report_bytes = sevsnp::decode_base64(&doc.cpu.report)?;

    let tls_key_fp = decode_hex_array::<32>(&doc.report_data.tls_key_fp, "tls_key_fp")?;
    let hpke_key = decode_hex(&doc.report_data.hpke_key, "hpke_key")?;
    let nonce = decode_hex(&doc.report_data.nonce, "nonce")?;
    let gpu_evidence_hash = match &doc.report_data.gpu_evidence_hash {
        Some(s) if !s.is_empty() => decode_hex(s, "gpu_evidence_hash")?,
        _ => Vec::new(),
    };
    let nvswitch_evidence_hash = match &doc.report_data.nvswitch_evidence_hash {
        Some(s) if !s.is_empty() => decode_hex(s, "nvswitch_evidence_hash")?,
        _ => Vec::new(),
    };

    let certificate_der = pem_to_der(&doc.certificate)?;
    let signature_der = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        doc.signature.trim(),
    )
    .map_err(|e| Error::Bundle(format!("attestation signature base64 decode: {e}")))?;
    let signed_payload = signed_payload(raw)?;

    Ok(ResolvedAttestation {
        platform,
        report_bytes,
        vcek_der: doc
            .vcek
            .as_ref()
            .and_then(|s| sevsnp::decode_base64(s).ok()),
        report_data: ReportData {
            tls_key_fp,
            hpke_key,
            nonce,
            gpu_evidence_hash,
            nvswitch_evidence_hash,
        },
        certificate_der,
        signed_payload,
        signature_der,
    })
}

/// Reconstruct the exact bytes the enclave signed: the served document with the
/// `signature` field's *value* blanked.
///
/// Upstream signs `json.Marshal(doc)` with `doc.Signature == ""`, then serves
/// `json.Marshal(doc)` with the real signature. The two byte strings differ
/// only in the signature value, so blanking it in the served bytes reproduces
/// the signed form precisely — and survives any field we don't model, since we
/// never re-serialize. We first trim trailing ASCII whitespace because the
/// upstream handler streams via `json.Encoder`, which appends a newline the
/// `json.Marshal` signing path does not (the shim mock appends nothing).
fn signed_payload(raw: &[u8]) -> Result<Vec<u8>, Error> {
    let body = trim_trailing_ascii_ws(raw);
    let needle = b"\"signature\":\"";
    let key_pos = find_subslice(body, needle)
        .ok_or_else(|| Error::Bundle("attestation document has no signature field".to_string()))?;
    let val_start = key_pos + needle.len();
    let val_end = body[val_start..]
        .iter()
        .position(|&b| b == b'"')
        .map(|rel| val_start + rel)
        .ok_or_else(|| {
            Error::Bundle("attestation signature value is not terminated".to_string())
        })?;
    let mut out = Vec::with_capacity(body.len() - (val_end - val_start));
    out.extend_from_slice(&body[..val_start]);
    out.extend_from_slice(&body[val_end..]);
    Ok(out)
}

/// Verify the document's ECDSA signature over [`ResolvedAttestation::signed_payload`]
/// using the public key in its embedded TLS certificate.
///
/// The enclave signs the SHA-256 prehash of the payload with its TLS leaf key
/// (P-384 in production, P-256 in the shim mock). Note the deliberate
/// SHA-256/P-384 pairing — upstream hashes with SHA-256 regardless of curve, so
/// we verify the prehash rather than letting a `DigestVerifier` impose
/// SHA-384.
pub fn verify_document_signature(resolved: &ResolvedAttestation) -> Result<(), Error> {
    use spki::DecodePublicKey;

    let cert = x509_cert::Certificate::from_der(&resolved.certificate_der)
        .map_err(|e| Error::CertParse(format!("attestation certificate parse: {e}")))?;
    let spki = &cert.tbs_certificate.subject_public_key_info;
    let spki_der = spki
        .to_der()
        .map_err(|e| Error::CertParse(format!("attestation SPKI re-encode: {e}")))?;
    let curve = spki
        .algorithm
        .parameters
        .clone()
        .ok_or_else(|| Error::DocSignature("certificate has no named-curve parameter".to_string()))?
        .decode_as::<spki::ObjectIdentifier>()
        .map_err(|e| Error::DocSignature(format!("curve OID decode: {e}")))?;

    let prehash = Sha256::digest(&resolved.signed_payload);

    if curve == OID_P384 {
        let vk = p384::ecdsa::VerifyingKey::from_public_key_der(&spki_der)
            .map_err(|e| Error::DocSignature(format!("P-384 key decode: {e}")))?;
        let sig = p384::ecdsa::Signature::from_der(&resolved.signature_der)
            .map_err(|e| Error::DocSignature(format!("P-384 signature decode: {e}")))?;
        PrehashVerifier::verify_prehash(&vk, &prehash, &sig)
            .map_err(|e| Error::DocSignature(format!("P-384 signature invalid: {e}")))
    } else if curve == OID_P256 {
        let vk = p256::ecdsa::VerifyingKey::from_public_key_der(&spki_der)
            .map_err(|e| Error::DocSignature(format!("P-256 key decode: {e}")))?;
        let sig = p256::ecdsa::Signature::from_der(&resolved.signature_der)
            .map_err(|e| Error::DocSignature(format!("P-256 signature decode: {e}")))?;
        PrehashVerifier::verify_prehash(&vk, &prehash, &sig)
            .map_err(|e| Error::DocSignature(format!("P-256 signature invalid: {e}")))
    } else {
        Err(Error::DocSignature(format!(
            "unsupported attestation signing curve OID: {curve}"
        )))
    }
}

fn decode_hex(s: &str, field: &str) -> Result<Vec<u8>, Error> {
    hex::decode(s).map_err(|e| Error::Bundle(format!("{field} hex decode: {e}")))
}

fn decode_hex_array<const N: usize>(s: &str, field: &str) -> Result<[u8; N], Error> {
    let v = decode_hex(s, field)?;
    v.try_into()
        .map_err(|_| Error::Bundle(format!("{field} must be {N} bytes")))
}

fn pem_to_der(pem_str: &str) -> Result<Vec<u8>, Error> {
    let parsed =
        pem::parse(pem_str).map_err(|e| Error::CertParse(format!("certificate PEM parse: {e}")))?;
    Ok(parsed.into_contents())
}

fn trim_trailing_ascii_ws(b: &[u8]) -> &[u8] {
    let mut end = b.len();
    while end > 0 && b[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    &b[..end]
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Decode and decompress a gzipped attestation report body (base64 → gzip → raw bytes).
pub fn decode_report_gzipped(body: &str) -> Result<Vec<u8>, Error> {
    let compressed = sevsnp::decode_base64(body)?;
    let mut decoder = GzDecoder::new(&compressed[..]);
    let mut report_bytes = Vec::new();
    decoder
        .read_to_end(&mut report_bytes)
        .map_err(|e| Error::Decompress(e.to_string()))?;
    Ok(report_bytes)
}
