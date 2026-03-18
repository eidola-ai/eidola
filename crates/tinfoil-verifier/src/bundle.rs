use base64::Engine;
use flate2::read::GzDecoder;
use serde::Deserialize;
use std::io::Read;

use crate::Error;

/// Full attestation bundle containing report, VCEK, and enclave certificate.
///
/// Fetched from the Tinfoil attestation transparency service (ATC).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct AttestationBundle {
    pub domain: String,
    #[serde(rename = "enclaveAttestationReport")]
    pub enclave_attestation_report: AttestationDocument,
    #[serde(rename = "enclaveCert")]
    pub enclave_cert: String,
    pub vcek: String,
    pub digest: String,
    #[serde(rename = "sigstoreBundle")]
    pub sigstore_bundle: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct AttestationDocument {
    pub format: String,
    pub body: String,
}

const EXPECTED_FORMAT: &str = "https://tinfoil.sh/predicate/sev-snp-guest/v2";

/// Default Tinfoil attestation transparency service URL.
const DEFAULT_ATC_URL: &str = "https://atc.tinfoil.sh/attestation";

/// Fetch the full attestation bundle from the Tinfoil ATC service.
///
/// The bundle contains the attestation report, VCEK certificate, and enclave
/// TLS certificate — everything needed for verification.
pub async fn fetch_bundle(atc_url: Option<&str>) -> Result<AttestationBundle, Error> {
    let url = atc_url.unwrap_or(DEFAULT_ATC_URL);
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| Error::Bundle(format!("failed to build HTTP client: {e}")))?;
    let bundle: AttestationBundle = client
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    if bundle.enclave_attestation_report.format != EXPECTED_FORMAT {
        return Err(Error::Bundle(format!(
            "unsupported attestation format: {}",
            bundle.enclave_attestation_report.format,
        )));
    }

    Ok(bundle)
}

/// Decode and decompress the attestation report body (base64 → gzip → raw bytes).
pub fn decode_report(body: &str) -> Result<Vec<u8>, Error> {
    let compressed = base64::engine::general_purpose::STANDARD.decode(body)?;
    let mut decoder = GzDecoder::new(&compressed[..]);
    let mut report_bytes = Vec::new();
    decoder
        .read_to_end(&mut report_bytes)
        .map_err(|e| Error::Decompress(e.to_string()))?;
    Ok(report_bytes)
}
