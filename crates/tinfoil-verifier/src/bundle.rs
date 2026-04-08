use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
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

/// V3 attestation document (`/.well-known/tinfoil-attestation?v=3`).
/// CPU report is base64-encoded raw bytes (not gzipped). VCEK is included
/// when the server self-verifies at boot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationDocumentV3 {
    pub format: String,
    pub cpu: AttestationCPU,
    pub vcek: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Unified attestation data extracted from a v3 well-known document.
pub struct ResolvedAttestation {
    pub platform: Platform,
    pub report_bytes: Vec<u8>,
    pub vcek_der: Option<Vec<u8>>,
    pub enclave_cert: Option<String>,
}

/// ATC bundle format strings identifying the platform of the nested report.
pub(crate) const ATC_SNP_FORMAT: &str = "https://tinfoil.sh/predicate/sev-snp-guest/v2";
pub(crate) const ATC_TDX_FORMAT: &str = "https://tinfoil.sh/predicate/tdx-guest/v2";
const V3_FORMAT: &str = "https://tinfoil.sh/predicate/attestation/v3";

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
) -> Result<AttestationBundle, Error> {
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

/// Fetch and resolve a v3 attestation document from a well-known endpoint.
pub async fn fetch_well_known(
    client: &reqwest::Client,
    attestation_url: &str,
) -> Result<ResolvedAttestation, Error> {
    let v3_url = format!("{attestation_url}?v=3");
    let doc: AttestationDocumentV3 = client
        .get(&v3_url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    if doc.format != V3_FORMAT {
        return Err(Error::Bundle(format!(
            "unexpected attestation document format: {}",
            doc.format
        )));
    }
    tracing::info!(
        "Fetched v3 attestation document (platform: {})",
        doc.cpu.platform
    );
    resolve_v3(&doc)
}

/// Resolve a v3 attestation document into unified format.
pub fn resolve_v3(doc: &AttestationDocumentV3) -> Result<ResolvedAttestation, Error> {
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
    Ok(ResolvedAttestation {
        platform,
        report_bytes,
        vcek_der: doc
            .vcek
            .as_ref()
            .and_then(|s| sevsnp::decode_base64(s).ok()),
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
