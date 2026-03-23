use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use std::io::Read;

use crate::{Error, sevsnp};

/// Full attestation bundle containing report, VCEK, and enclave certificate.
///
/// Fetched from the Tinfoil attestation transparency service (ATC).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct AttestationBundle {
    pub domain: String,
    #[serde(rename = "enclaveAttestationReport")]
    pub enclave_attestation_report: AttestationDocumentV2,
    #[serde(rename = "enclaveCert")]
    pub enclave_cert: String,
    pub vcek: String,
    pub ark: Option<String>,
    pub ask: Option<String>,
    pub digest: String,
    #[serde(rename = "sigstoreBundle")]
    pub sigstore_bundle: Option<serde_json::Value>,
}

/// V2 attestation document (`/.well-known/tinfoil-attestation` default response).
/// Body is base64-encoded gzip-compressed attestation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationDocumentV2 {
    pub format: String,
    pub body: String,
    pub vcek: Option<String>,
    pub ark: Option<String>,
    pub ask: Option<String>,
    #[serde(rename = "enclaveCert")]
    pub enclave_cert: Option<String>,
}

/// V3 attestation document (`/.well-known/tinfoil-attestation?v=3`).
/// CPU report is base64-encoded raw bytes (not gzipped). VCEK is included
/// when the server self-verifies at boot. ARK/ASK are included for custom
/// chains (dev shim); production uses built-in AMD Genoa certs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationDocumentV3 {
    pub format: String,
    pub cpu: AttestationCPU,
    pub vcek: Option<String>,
    pub ark: Option<String>,
    pub ask: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationCPU {
    pub platform: String,
    pub report: String, // base64-encoded raw report (NOT gzipped)
}

/// Unified attestation data extracted from either v2 or v3 documents.
/// Used by the verification pipeline so the rest of the code doesn't
/// need to care which format was fetched.
pub struct ResolvedAttestation {
    pub report_bytes: Vec<u8>,
    pub vcek_der: Option<Vec<u8>>,
    pub ark_der: Option<Vec<u8>>,
    pub ask_der: Option<Vec<u8>>,
    pub enclave_cert: Option<String>,
}

const V2_FORMAT: &str = "https://tinfoil.sh/predicate/sev-snp-guest/v2";
const V3_FORMAT: &str = "https://tinfoil.sh/predicate/attestation/v3";

/// Default Tinfoil attestation transparency service URL.
pub const DEFAULT_ATC_URL: &str = "https://atc.tinfoil.sh/attestation";

/// Fetch the full attestation bundle from the Tinfoil ATC service.
///
/// The bundle contains the attestation report, VCEK certificate, and enclave
/// TLS certificate — everything needed for verification.
pub async fn fetch_bundle(atc_url: Option<&str>) -> Result<AttestationBundle, Error> {
    let url = atc_url.unwrap_or(DEFAULT_ATC_URL);
    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let tls_config = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(30))
        .tls_backend_preconfigured(tls_config)
        .build()
        .map_err(|e| Error::Bundle(format!("failed to build HTTP client: {e}")))?;
    let bundle: AttestationBundle = client
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    if bundle.enclave_attestation_report.format != V2_FORMAT {
        return Err(Error::Bundle(format!(
            "unsupported attestation format: {}",
            bundle.enclave_attestation_report.format,
        )));
    }

    Ok(bundle)
}

/// Fetch and resolve attestation from a well-known endpoint.
///
/// Tries v3 first (`?v=3`), falls back to v2 if the server doesn't support v3.
/// Returns a unified `ResolvedAttestation` with decoded report bytes and
/// optional certificates.
pub async fn fetch_well_known(
    client: &reqwest::Client,
    attestation_url: &str,
) -> Result<ResolvedAttestation, Error> {
    // Try v3 first
    let v3_url = format!("{attestation_url}?v=3");
    if let Ok(resp) = client.get(&v3_url).send().await
        && resp.status().is_success()
        && let Ok(doc) = resp.json::<AttestationDocumentV3>().await
        && doc.format == V3_FORMAT
        && doc.cpu.platform == "sev-snp"
    {
        tracing::info!("Using v3 attestation document");
        return resolve_v3(&doc);
    }

    // Fall back to v2
    tracing::info!("Falling back to v2 attestation document");
    let doc: AttestationDocumentV2 = client
        .get(attestation_url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    resolve_v2(&doc)
}

/// Resolve a v2 attestation document into unified format.
pub fn resolve_v2(doc: &AttestationDocumentV2) -> Result<ResolvedAttestation, Error> {
    Ok(ResolvedAttestation {
        report_bytes: decode_report_gzipped(&doc.body)?,
        vcek_der: doc
            .vcek
            .as_ref()
            .and_then(|s| sevsnp::decode_base64(s).ok()),
        ark_der: doc.ark.as_ref().and_then(|s| sevsnp::decode_base64(s).ok()),
        ask_der: doc.ask.as_ref().and_then(|s| sevsnp::decode_base64(s).ok()),
        enclave_cert: doc.enclave_cert.clone(),
    })
}

/// Resolve a v3 attestation document into unified format.
pub fn resolve_v3(doc: &AttestationDocumentV3) -> Result<ResolvedAttestation, Error> {
    let report_bytes = sevsnp::decode_base64(&doc.cpu.report)?;
    Ok(ResolvedAttestation {
        report_bytes,
        vcek_der: doc
            .vcek
            .as_ref()
            .and_then(|s| sevsnp::decode_base64(s).ok()),
        ark_der: doc.ark.as_ref().and_then(|s| sevsnp::decode_base64(s).ok()),
        ask_der: doc.ask.as_ref().and_then(|s| sevsnp::decode_base64(s).ok()),
        enclave_cert: None,
    })
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
