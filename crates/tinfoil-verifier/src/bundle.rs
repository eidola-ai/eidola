//! Strict parsing of Tinfoil's self-contained v3 attestation envelope.
//!
//! Parsing and challenge checking do not authenticate the document by
//! themselves. Authentication happens only after the platform verifier proves
//! that the signed CPU evidence carries [`ResolvedAttestation::report_data`].

use std::collections::HashSet;

use base64::Engine as _;
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};

use crate::Error;

pub const ATTESTATION_V3_FORMAT: &str = "https://tinfoil.sh/predicate/attestation/v3";
pub const REPORT_DATA_V1_ALGORITHM: &str = "https://tinfoil.sh/report-data/v1";
pub const CRYPTO_MATERIAL_V1_FORMAT: &str = "https://tinfoil.sh/crypto-material/v1";
pub const DEVICE_EVIDENCE_V1_FORMAT: &str = "https://tinfoil.sh/device-evidence/v1";
pub const SEV_SNP_REPORT_V1_FORMAT: &str = "https://tinfoil.sh/format/sev-snp-report/v1";
pub const TDX_QUOTE_V1_FORMAT: &str = "https://tinfoil.sh/format/tdx-quote/v1";
pub const KEY_SPKI_FP_SHA256_V1_FORMAT: &str = "https://tinfoil.sh/key/spki-fp-sha256/v1";
pub const KEY_X25519_HPKE_V1_FORMAT: &str = "https://tinfoil.sh/key/x25519-hpke/v1";
pub const COLLATERAL_AMD_VCEK_V1_FORMAT: &str = "https://tinfoil.sh/collateral/amd-vcek/v1";
pub const COLLATERAL_AMD_CRL_V1_FORMAT: &str = "https://tinfoil.sh/collateral/amd-crl/v1";
pub const COLLATERAL_INTEL_PCS_V1_FORMAT: &str = "https://tinfoil.sh/collateral/intel-pcs/v1";
pub const NVIDIA_GPU_EVIDENCE_V1_FORMAT: &str = "https://tinfoil.sh/format/nvidia-gpu-evidence/v1";

/// The collateral set's verification profile. The pinned platform release
/// accepts only this profile from its collateral service, and it is the only
/// one this parser decodes.
pub const COLLATERAL_V3_FORMAT: &str = "https://tinfoil.sh/predicate/attestation-collaterals/v3";
pub const COLLATERAL_IGVM_RUNTIME_V1_FORMAT: &str = "https://tinfoil.sh/collateral/igvm-runtime/v1";
pub const COLLATERAL_SIGSTORE_PLATFORM_V1_FORMAT: &str =
    "https://tinfoil.sh/collateral/sigstore-platform/v1";
pub const COLLATERAL_ARTIFACT_FRESHNESS_V1_FORMAT: &str =
    "https://tinfoil.sh/collateral/artifact-freshness/v1";
pub const COLLATERAL_CONFIG_V1_FORMAT: &str = "https://tinfoil.sh/collateral/config/v1";
pub const COLLATERAL_CONFIG_ENDORSEMENT_V1_FORMAT: &str =
    "https://tinfoil.sh/collateral/config-endorsement/v1";
pub const COLLATERAL_CONFIG_ENDORSEMENT_V2_FORMAT: &str =
    "https://tinfoil.sh/collateral/config-endorsement/v2";
/// Code-provenance collateral of the legacy (v2) profile, refused under v3.
pub const COLLATERAL_SIGSTORE_CODE_V1_FORMAT: &str =
    "https://tinfoil.sh/collateral/sigstore-code/v1";
/// Freshness witnesses of the legacy (v2) profile, refused under v3.
pub const COLLATERAL_SIGSTORE_FRESHNESS_V1_FORMAT: &str =
    "https://tinfoil.sh/collateral/sigstore-freshness/v1";

const ROLE_ENDORSEMENT: &str = "endorsement";
const ROLE_REFERENCE_VALUES: &str = "reference-values";
const SUBJECT_CPU: &str = "cpu";
const CRYPTO_ID_TLS: &str = "tls";
const CRYPTO_ID_HPKE: &str = "hpke";
const COLLATERAL_ID_RUNTIME: &str = "runtime";
const COLLATERAL_ID_RUNTIME_FRESHNESS: &str = "runtime-freshness";
const COLLATERAL_ID_PLATFORM_FRESHNESS: &str = "platform-freshness";
const COLLATERAL_ID_CONFIG: &str = "tinfoil-config";
/// The only repository an `igvm-runtime/v1` entry may name.
const RUNTIME_REPO: &str = "tinfoilsh/cvmimage";

/// Length in bytes of the verifier-chosen challenge nonce.
pub(crate) const NONCE_LEN: usize = 32;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    format: String,
    challenge: Challenge,
    cpu_evidence: CpuEvidence,
    crypto_material: String,
    device_evidence: String,
    collateral_format: String,
    collateral: Vec<CollateralEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Challenge {
    nonce: String,
    report_data: String,
    report_data_algorithm: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CpuEvidence {
    format: String,
    report_base64: String,
    endorsed: EndorsedHashes,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EndorsedHashes {
    crypto_material_hash: String,
    device_evidence_hash: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CryptoMaterialSection {
    format: String,
    items: Vec<CryptoMaterialItem>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CryptoMaterialItem {
    id: String,
    format: String,
    data: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeviceEvidenceSection {
    format: String,
    items: Vec<DeviceEvidenceItem>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeviceEvidenceItem {
    id: String,
    kind: String,
    vendor: String,
    format: String,
    evidence: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CollateralEntry {
    id: String,
    role: String,
    format: String,
    #[serde(default)]
    subjects: Vec<String>,
    data: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AmdVcekCollateral {
    vcek_der_base64: String,
    cert_chain_pem: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AmdCrlCollateral {
    crl_der_base64: String,
}

/// `intel-pcs/v1` collateral: Intel PCS responses captured by the enclave's
/// collateral service, carried so the verifier never contacts Intel.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct IntelPcsCollateral {
    responses: Vec<IntelPcsResponse>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct IntelPcsResponse {
    url: String,
    /// Absent or `null` when the capture kept no headers (Intel's root CA CRL
    /// carries none the verifier reads).
    #[serde(default)]
    headers: Option<std::collections::BTreeMap<String, Vec<String>>>,
    body_base64: String,
}

/// `igvm-runtime/v1` collateral: the platform release manifest the guest's
/// config pins in `cvm-version`, named by tag and SHA-256, with the release's
/// Sigstore provenance bundle.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeCollateral {
    repo: String,
    tag: String,
    digest: String,
    manifest_base64: String,
    sigstore_bundle: serde_json::Value,
}

/// `sigstore-platform/v1` collateral: a platform endorsements release and
/// its Sigstore attestation.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SigstorePlatformCollateral {
    repo: String,
    tag: String,
    digest: String,
    sigstore_bundle: serde_json::Value,
}

/// `artifact-freshness/v1` collateral: the provider's signed approval of the
/// runtime or platform release the entry's ID names.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactFreshnessCollateral {
    sigstore_bundle: serde_json::Value,
}

/// `config/v1`, `config-endorsement/v1` and `config-endorsement/v2`
/// collateral, which share one member set. Which members each format allows
/// or requires is checked in [`decode_config_collateral`]; a member present
/// as `null` counts as present.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigCollateral {
    #[serde(default, deserialize_with = "present")]
    endorsement_ref: Option<serde_json::Value>,
    #[serde(default, deserialize_with = "present")]
    config_base64: Option<serde_json::Value>,
    #[serde(default, deserialize_with = "present")]
    sigstore_bundle: Option<serde_json::Value>,
    #[serde(default, deserialize_with = "present")]
    issuer_certificate_der: Option<serde_json::Value>,
}

/// Deserialize a member that is present, whatever its value (`null`
/// included), as `Some`; `#[serde(default)]` leaves an absent one `None`.
fn present<'de, D>(deserializer: D) -> Result<Option<serde_json::Value>, D::Error>
where
    D: Deserializer<'de>,
{
    serde_json::Value::deserialize(deserializer).map(Some)
}

/// One endorsed device evidence item. Its `evidence` payload is versioned by
/// `format`; only formats a pinned policy names are interpreted.
#[derive(Debug, Clone)]
pub struct DeviceEvidence {
    pub id: String,
    pub kind: String,
    pub vendor: String,
    pub format: String,
    pub evidence: serde_json::Value,
}

/// TEE platform identified by the CPU evidence format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    SevSnp,
    Tdx,
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::SevSnp => "SEV-SNP",
            Self::Tdx => "TDX",
        })
    }
}

/// Fully decoded challenge and CPU evidence needed by the handshake verifier.
pub struct ResolvedAttestation {
    pub platform: Platform,
    pub report_bytes: Vec<u8>,
    /// The recomputed v1 REPORT_DATA that the signed CPU evidence must carry.
    pub report_data: [u8; 64],
    pub nonce: [u8; NONCE_LEN],
    /// SHA-256 of the endorsed TLS SubjectPublicKeyInfo.
    pub tls_key_fp: [u8; 32],
    /// Endorsed X25519 HPKE public key.
    pub hpke_key: [u8; 32],
    /// Present and required for SEV-SNP evidence.
    pub vcek_der: Option<Vec<u8>>,
    /// Present and required for SEV-SNP evidence.
    pub ask_der: Option<Vec<u8>>,
    /// Present and required for SEV-SNP evidence.
    pub ark_der: Option<Vec<u8>>,
    /// Present and required for SEV-SNP evidence.
    pub crl_der: Option<Vec<u8>>,
    /// Present and required for TDX evidence: the captured Intel PCS
    /// responses that authenticate the quote.
    pub intel_pcs: Option<Vec<crate::tdx::PcsResponse>>,
    /// The endorsed device evidence items, in document order.
    pub device_evidence: Vec<DeviceEvidence>,
    /// SHA-256 of the exact config bytes a `tinfoil-config` collateral entry
    /// carries, when it carries them. Unauthenticated: the platform verifier
    /// requires it to equal the config hash the signed report holds
    /// (SEV-SNP HOST_DATA, TDX MRCONFIGID).
    pub config_sha256: Option<[u8; 32]>,
}

/// Generate a fresh 32-byte challenge nonce from the OS CSPRNG.
pub(crate) fn random_nonce() -> Result<[u8; NONCE_LEN], Error> {
    let mut nonce = [0u8; NONCE_LEN];
    getrandom::fill(&mut nonce)
        .map_err(|e| Error::Connector(format!("failed to generate attestation nonce: {e}")))?;
    Ok(nonce)
}

/// Fetch and resolve a v3 document with a fresh challenge.
///
/// This checks the envelope's internal bindings and nonce. The result is not
/// authenticated until its CPU evidence and `report_data` are verified.
#[cfg(not(target_family = "wasm"))]
pub async fn fetch_well_known(
    client: &reqwest::Client,
    attestation_url: &str,
) -> Result<ResolvedAttestation, Error> {
    let nonce = random_nonce()?;
    let separator = if attestation_url.contains('?') {
        '&'
    } else {
        '?'
    };
    let url = format!("{attestation_url}{separator}nonce={}", hex::encode(nonce));
    let raw = client
        .get(&url)
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    let resolved = parse_document(&raw)?;
    if resolved.nonce != nonce {
        return Err(Error::NonceMismatch {
            sent: hex::encode(nonce),
            echoed: hex::encode(resolved.nonce),
        });
    }
    Ok(resolved)
}

/// Strictly parse a self-contained v3 attestation document.
///
/// The parser mirrors Tinfoil's v3 rules: object members are case-sensitive,
/// unknown or duplicate members are rejected, hex is lowercase and
/// fixed-width, base64 is canonical, and item IDs are unique. The two
/// endorsed section hashes are computed over their exact decoded JSON bytes,
/// never over a re-serialization.
pub fn parse_document(raw: &[u8]) -> Result<ResolvedAttestation, Error> {
    parse_document_for(raw, &[Platform::SevSnp, Platform::Tdx])
}

/// [`parse_document`] for a verifier that accepts only `pinned` platforms.
///
/// The CPU evidence format is identified right after the envelope's own
/// JSON is parsed, and a platform outside `pinned` is refused with
/// [`Error::PlatformNotPinned`] before any section is decoded and before any
/// platform-specific payload (the CPU report, its vendor collateral) is
/// touched. A verifier pinned to SEV-SNP never runs the TDX parsers, and the
/// reverse.
pub fn parse_document_for(raw: &[u8], pinned: &[Platform]) -> Result<ResolvedAttestation, Error> {
    reject_duplicate_members(raw, "attestation document")?;
    let doc: Document = serde_json::from_slice(raw)
        .map_err(|e| Error::Bundle(format!("parsing attestation document: {e}")))?;

    require_eq("document format", &doc.format, ATTESTATION_V3_FORMAT)?;
    require_eq(
        "collateral_format",
        &doc.collateral_format,
        COLLATERAL_V3_FORMAT,
    )?;
    let platform = match doc.cpu_evidence.format.as_str() {
        SEV_SNP_REPORT_V1_FORMAT => Platform::SevSnp,
        TDX_QUOTE_V1_FORMAT => Platform::Tdx,
        other => {
            return Err(Error::Bundle(format!(
                "unsupported CPU evidence format: {other}"
            )));
        }
    };
    if !pinned.contains(&platform) {
        return Err(Error::PlatformNotPinned { platform });
    }
    require_eq(
        "report_data_algorithm",
        &doc.challenge.report_data_algorithm,
        REPORT_DATA_V1_ALGORITHM,
    )?;

    let nonce = decode_lower_hex_array::<NONCE_LEN>(&doc.challenge.nonce, "challenge.nonce")?;
    let claimed_report_data =
        decode_lower_hex_array::<64>(&doc.challenge.report_data, "challenge.report_data")?;
    let claimed_crypto_hash = decode_lower_hex_array::<32>(
        &doc.cpu_evidence.endorsed.crypto_material_hash,
        "cpu_evidence.endorsed.crypto_material_hash",
    )?;
    let claimed_device_hash = decode_lower_hex_array::<32>(
        &doc.cpu_evidence.endorsed.device_evidence_hash,
        "cpu_evidence.endorsed.device_evidence_hash",
    )?;

    let crypto_bytes = decode_canonical_base64(&doc.crypto_material, "crypto_material")?;
    let device_bytes = decode_canonical_base64(&doc.device_evidence, "device_evidence")?;
    reject_duplicate_members(&crypto_bytes, "crypto_material")?;
    reject_duplicate_members(&device_bytes, "device_evidence")?;

    let crypto: CryptoMaterialSection = serde_json::from_slice(&crypto_bytes)
        .map_err(|e| Error::Bundle(format!("parsing crypto_material: {e}")))?;
    require_eq(
        "crypto_material format",
        &crypto.format,
        CRYPTO_MATERIAL_V1_FORMAT,
    )?;
    let (tls_key_fp, hpke_key) = parse_crypto_material(&crypto.items)?;

    let device: DeviceEvidenceSection = serde_json::from_slice(&device_bytes)
        .map_err(|e| Error::Bundle(format!("parsing device_evidence: {e}")))?;
    require_eq(
        "device_evidence format",
        &device.format,
        DEVICE_EVIDENCE_V1_FORMAT,
    )?;
    validate_device_evidence(&device.items)?;
    let device_evidence = device
        .items
        .into_iter()
        .map(|item| DeviceEvidence {
            id: item.id,
            kind: item.kind,
            vendor: item.vendor,
            format: item.format,
            evidence: item.evidence,
        })
        .collect();

    let crypto_hash: [u8; 32] = Sha256::digest(&crypto_bytes).into();
    let device_hash: [u8; 32] = Sha256::digest(&device_bytes).into();
    if crypto_hash != claimed_crypto_hash {
        return Err(Error::Bundle(
            "crypto_material hash does not match cpu_evidence endorsement".to_string(),
        ));
    }
    if device_hash != claimed_device_hash {
        return Err(Error::Bundle(
            "device_evidence hash does not match cpu_evidence endorsement".to_string(),
        ));
    }

    let report_data = compute_report_data(&nonce, &crypto_hash, &device_hash);
    if report_data != claimed_report_data {
        return Err(Error::Bundle(
            "challenge report_data does not match the recomputed value".to_string(),
        ));
    }

    let report_bytes = decode_canonical_base64(
        &doc.cpu_evidence.report_base64,
        "cpu_evidence.report_base64",
    )?;
    if report_bytes.is_empty() {
        return Err(Error::Bundle("CPU evidence report is empty".to_string()));
    }

    validate_collateral_entries(&doc.collateral)?;
    let config_sha256 = decode_reference_values(&doc.collateral)?;
    let (vcek_der, ask_der, ark_der, crl_der, intel_pcs) = match platform {
        Platform::SevSnp => {
            let vcek = endorsement_for(&doc.collateral, COLLATERAL_AMD_VCEK_V1_FORMAT, SUBJECT_CPU)
                .ok_or_else(|| {
                    Error::Bundle(
                        "document carries no amd-vcek endorsement collateral for the CPU"
                            .to_string(),
                    )
                })?;
            let crl = endorsement_for(&doc.collateral, COLLATERAL_AMD_CRL_V1_FORMAT, SUBJECT_CPU)
                .ok_or_else(|| {
                Error::Bundle(
                    "document carries no amd-crl endorsement collateral for the CPU".to_string(),
                )
            })?;
            let vcek: AmdVcekCollateral = serde_json::from_value(vcek.data.clone())
                .map_err(|e| Error::Bundle(format!("parsing amd-vcek collateral: {e}")))?;
            let (ask_der, ark_der) = decode_amd_cert_chain(&vcek.cert_chain_pem)?;
            let crl: AmdCrlCollateral = serde_json::from_value(crl.data.clone())
                .map_err(|e| Error::Bundle(format!("parsing amd-crl collateral: {e}")))?;
            let vcek_der = decode_canonical_base64(&vcek.vcek_der_base64, "vcek_der_base64")?;
            let crl_der = decode_canonical_base64(&crl.crl_der_base64, "crl_der_base64")?;
            if vcek_der.is_empty() || crl_der.is_empty() {
                return Err(Error::Bundle(
                    "SEV-SNP endorsement collateral contains empty DER material".to_string(),
                ));
            }
            (
                Some(vcek_der),
                Some(ask_der),
                Some(ark_der),
                Some(crl_der),
                None,
            )
        }
        Platform::Tdx => {
            let pcs = endorsement_for(&doc.collateral, COLLATERAL_INTEL_PCS_V1_FORMAT, SUBJECT_CPU)
                .ok_or_else(|| {
                    Error::Bundle(
                        "document carries no intel-pcs endorsement collateral for the CPU"
                            .to_string(),
                    )
                })?;
            let pcs: IntelPcsCollateral = serde_json::from_value(pcs.data.clone())
                .map_err(|e| Error::Bundle(format!("parsing intel-pcs collateral: {e}")))?;
            let responses = pcs
                .responses
                .into_iter()
                .enumerate()
                .map(|(index, response)| {
                    Ok(crate::tdx::PcsResponse {
                        body: decode_canonical_base64(
                            &response.body_base64,
                            &format!("intel-pcs response {index} body_base64"),
                        )?,
                        url: response.url,
                        headers: response.headers.unwrap_or_default().into_iter().collect(),
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            (None, None, None, None, Some(responses))
        }
    };

    Ok(ResolvedAttestation {
        platform,
        report_bytes,
        report_data,
        nonce,
        tls_key_fp,
        hpke_key,
        vcek_der,
        ask_der,
        ark_der,
        crl_der,
        intel_pcs,
        device_evidence,
        config_sha256,
    })
}

/// Decode the AMD KDS `cert_chain` payload: exactly ASK then ARK, with no
/// non-whitespace material outside those two PEM blocks. The chain is still
/// untrusted transport; the handshake verifier compares both certificates
/// with its configured/pinned AMD chain before using them.
fn decode_amd_cert_chain(chain_pem: &str) -> Result<(Vec<u8>, Vec<u8>), Error> {
    const BEGIN: &str = "-----BEGIN CERTIFICATE-----";
    const END: &str = "-----END CERTIFICATE-----";

    let mut rest = chain_pem.trim();
    let mut certs = Vec::with_capacity(2);
    while !rest.is_empty() {
        if !rest.starts_with(BEGIN) {
            return Err(Error::Bundle(
                "amd-vcek cert_chain_pem carries data outside CERTIFICATE blocks".to_string(),
            ));
        }
        let end = rest.find(END).ok_or_else(|| {
            Error::Bundle("amd-vcek cert_chain_pem has an unterminated certificate".to_string())
        })? + END.len();
        let block = pem::parse(&rest[..end])
            .map_err(|e| Error::Bundle(format!("parsing amd-vcek cert_chain_pem: {e}")))?;
        if block.tag() != "CERTIFICATE" || block.contents().is_empty() {
            return Err(Error::Bundle(
                "amd-vcek cert_chain_pem must contain non-empty CERTIFICATE blocks".to_string(),
            ));
        }
        certs.push(block.into_contents());
        rest = rest[end..].trim();
    }
    if certs.len() != 2 {
        return Err(Error::Bundle(format!(
            "amd-vcek cert_chain_pem must carry exactly ASK and ARK certificates, got {}",
            certs.len()
        )));
    }
    Ok((certs.remove(0), certs.remove(0)))
}

fn compute_report_data(
    nonce: &[u8; 32],
    crypto_material_hash: &[u8; 32],
    device_evidence_hash: &[u8; 32],
) -> [u8; 64] {
    let mut hasher = Sha256::new();
    hasher.update(REPORT_DATA_V1_ALGORITHM.as_bytes());
    hasher.update(nonce);
    hasher.update(crypto_material_hash);
    hasher.update(device_evidence_hash);
    let mut report_data = [0u8; 64];
    report_data[..32].copy_from_slice(&hasher.finalize());
    report_data
}

fn parse_crypto_material(items: &[CryptoMaterialItem]) -> Result<([u8; 32], [u8; 32]), Error> {
    let mut seen = HashSet::new();
    let mut tls_key_fp = None;
    let mut hpke_key = None;
    for item in items {
        if item.id.is_empty() || item.format.is_empty() {
            return Err(Error::Bundle(
                "crypto_material item is incomplete".to_string(),
            ));
        }
        if !seen.insert(&item.id) {
            return Err(Error::Bundle(format!(
                "duplicate crypto_material item id {:?}",
                item.id
            )));
        }
        let data = match item.format.as_str() {
            KEY_SPKI_FP_SHA256_V1_FORMAT | KEY_X25519_HPKE_V1_FORMAT => {
                Some(decode_lower_hex_array::<32>(
                    &item.data,
                    &format!("crypto_material item {:?} data", item.id),
                )?)
            }
            _ => {
                if item.data.is_empty()
                    || !item
                        .data
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                    || item.data.len() % 2 != 0
                {
                    return Err(Error::Bundle(format!(
                        "crypto_material item {:?} data is not non-empty lowercase hex",
                        item.id
                    )));
                }
                None
            }
        };
        match item.id.as_str() {
            CRYPTO_ID_TLS => {
                if item.format != KEY_SPKI_FP_SHA256_V1_FORMAT {
                    return Err(Error::Bundle(
                        "TLS crypto material has an unsupported format".to_string(),
                    ));
                }
                tls_key_fp = data;
            }
            CRYPTO_ID_HPKE => {
                if item.format != KEY_X25519_HPKE_V1_FORMAT {
                    return Err(Error::Bundle(
                        "HPKE crypto material has an unsupported format".to_string(),
                    ));
                }
                hpke_key = data;
            }
            _ => {}
        }
    }
    Ok((
        tls_key_fp.ok_or_else(|| Error::Bundle("crypto_material has no TLS key".to_string()))?,
        hpke_key.ok_or_else(|| Error::Bundle("crypto_material has no HPKE key".to_string()))?,
    ))
}

fn validate_device_evidence(items: &[DeviceEvidenceItem]) -> Result<(), Error> {
    let mut seen = HashSet::new();
    for item in items {
        if item.id.is_empty() {
            return Err(Error::Bundle("device_evidence item has no id".to_string()));
        }
        if !seen.insert(&item.id) {
            return Err(Error::Bundle(format!(
                "duplicate device_evidence item id {:?}",
                item.id
            )));
        }
        // These fields are format-versioned payload metadata. Accessing them
        // here makes it explicit that strict deserialization covered them.
        let _ = (&item.kind, &item.vendor, &item.format, &item.evidence);
    }
    Ok(())
}

fn validate_collateral_entries(entries: &[CollateralEntry]) -> Result<(), Error> {
    let mut seen = HashSet::new();
    for (index, entry) in entries.iter().enumerate() {
        if entry.id.is_empty() || entry.format.is_empty() {
            return Err(Error::Bundle(format!(
                "collateral entry {index} is incomplete"
            )));
        }
        if !seen.insert(&entry.id) {
            return Err(Error::Bundle(format!(
                "duplicate collateral entry id {:?}",
                entry.id
            )));
        }
        if entry.role != ROLE_ENDORSEMENT && entry.role != ROLE_REFERENCE_VALUES {
            return Err(Error::Bundle(format!(
                "collateral entry {:?} has unknown role {:?}",
                entry.id, entry.role
            )));
        }
        if !entry.data.is_object() {
            return Err(Error::Bundle(format!(
                "collateral entry {:?} data is not a JSON object",
                entry.id
            )));
        }
    }
    Ok(())
}

/// Decode the v3 profile's reference-values entries as the provider's own v3
/// decoder does: the reserved IDs `runtime`, `runtime-freshness`,
/// `platform-freshness` and `tinfoil-config` each go with exactly their
/// formats and the `reference-values` role; every entry of a known format is
/// decoded strictly, whether or not anything reads it; and the legacy (v2)
/// code-provenance and Sigstore-freshness formats are refused. An entry of a
/// format this crate does not know is held to the shape checks of
/// [`validate_collateral_entries`] and otherwise not read.
///
/// None of these entries is a trust input. A pin names the launch identity
/// itself, so the provider's release provenance and freshness approvals are
/// not consulted, and no Sigstore bundle here is verified. What a field
/// states that can be checked is checked: a runtime entry's manifest must
/// hash to its digest, and a config entry's bytes must hash to the config
/// hash of the signed report (returned here, compared once the report is
/// authenticated).
fn decode_reference_values(entries: &[CollateralEntry]) -> Result<Option<[u8; 32]>, Error> {
    let mut config_sha256 = None;
    for entry in entries {
        let reference_values = entry.role == ROLE_REFERENCE_VALUES;
        let format = entry.format.as_str();
        let id = entry.id.as_str();
        let config_format = matches!(
            format,
            COLLATERAL_CONFIG_V1_FORMAT
                | COLLATERAL_CONFIG_ENDORSEMENT_V1_FORMAT
                | COLLATERAL_CONFIG_ENDORSEMENT_V2_FORMAT
        );
        if id == COLLATERAL_ID_CONFIG || config_format {
            if id != COLLATERAL_ID_CONFIG || !config_format || !reference_values {
                return Err(Error::Bundle(format!(
                    "conflicting collateral entry for {COLLATERAL_ID_CONFIG:?}"
                )));
            }
            config_sha256 = decode_config_collateral(entry)?;
        } else if id == COLLATERAL_ID_RUNTIME || format == COLLATERAL_IGVM_RUNTIME_V1_FORMAT {
            if id != COLLATERAL_ID_RUNTIME
                || format != COLLATERAL_IGVM_RUNTIME_V1_FORMAT
                || !reference_values
            {
                return Err(Error::Bundle(format!(
                    "conflicting collateral entry for {COLLATERAL_ID_RUNTIME:?}"
                )));
            }
            decode_runtime_collateral(entry)?;
        } else if format == COLLATERAL_ARTIFACT_FRESHNESS_V1_FORMAT
            || id == COLLATERAL_ID_RUNTIME_FRESHNESS
            || id == COLLATERAL_ID_PLATFORM_FRESHNESS
        {
            if (id != COLLATERAL_ID_RUNTIME_FRESHNESS && id != COLLATERAL_ID_PLATFORM_FRESHNESS)
                || format != COLLATERAL_ARTIFACT_FRESHNESS_V1_FORMAT
                || !reference_values
            {
                return Err(Error::Bundle(format!(
                    "conflicting artifact freshness collateral {id:?}"
                )));
            }
            let data: ArtifactFreshnessCollateral = collateral_data(entry)?;
            require_bundle_object(entry, &data.sigstore_bundle)?;
        } else if reference_values && format == COLLATERAL_SIGSTORE_PLATFORM_V1_FORMAT {
            let data: SigstorePlatformCollateral = collateral_data(entry)?;
            decode_lower_hex_array::<32>(&data.digest, &format!("collateral entry {id:?} digest"))?;
            require_bundle_object(entry, &data.sigstore_bundle)?;
            // Repo and tag name the release the bundle attests. The bundle is
            // not verified, so they are only required to be present.
            let _ = (&data.repo, &data.tag);
        } else if reference_values
            && (format == COLLATERAL_SIGSTORE_CODE_V1_FORMAT
                || format == COLLATERAL_SIGSTORE_FRESHNESS_V1_FORMAT)
        {
            return Err(Error::Bundle(format!(
                "collateral entry {id:?} has the legacy format {format:?}, \
                 incompatible with {COLLATERAL_V3_FORMAT:?}"
            )));
        }
    }
    Ok(config_sha256)
}

fn decode_runtime_collateral(entry: &CollateralEntry) -> Result<(), Error> {
    let data: RuntimeCollateral = collateral_data(entry)?;
    if data.repo != RUNTIME_REPO {
        return Err(Error::Bundle(format!(
            "runtime collateral repository {:?} is not {RUNTIME_REPO:?}",
            data.repo
        )));
    }
    if !is_full_semver_tag(&data.tag) {
        return Err(Error::Bundle(format!(
            "runtime collateral tag {:?} is not a full semantic version",
            data.tag
        )));
    }
    let digest = decode_lower_hex_array::<32>(&data.digest, "runtime collateral digest")?;
    let manifest = decode_canonical_base64(&data.manifest_base64, "runtime manifest_base64")?;
    if <[u8; 32]>::from(Sha256::digest(&manifest)) != digest {
        return Err(Error::Bundle(
            "runtime collateral manifest does not hash to its digest".to_string(),
        ));
    }
    require_bundle_object(entry, &data.sigstore_bundle)
}

/// Decode a `tinfoil-config` entry and return the SHA-256 of the config
/// bytes it carries, if any. `config/v1` carries the bytes alone;
/// `config-endorsement/v1` carries them with a `sha256:` reference and a
/// Sigstore bundle; `config-endorsement/v2` may leave the bytes out and may
/// add an issuer certificate.
fn decode_config_collateral(entry: &CollateralEntry) -> Result<Option<[u8; 32]>, Error> {
    let data: ConfigCollateral = collateral_data(entry)?;
    let string = |value: &serde_json::Value, member: &str| -> Result<String, Error> {
        value.as_str().map(str::to_owned).ok_or_else(|| {
            Error::Bundle(format!(
                "tinfoil-config collateral {member} is not a string"
            ))
        })
    };
    if entry.format == COLLATERAL_CONFIG_V1_FORMAT {
        if data.endorsement_ref.is_some() || data.sigstore_bundle.is_some() {
            return Err(Error::Bundle(
                "plain config collateral cannot carry an endorsement".to_string(),
            ));
        }
    } else {
        let reference = data.endorsement_ref.as_ref().ok_or_else(|| {
            Error::Bundle("tinfoil-config collateral has no endorsement_ref".to_string())
        })?;
        let reference = string(reference, "endorsement_ref")?;
        let digest = reference.strip_prefix("sha256:").ok_or_else(|| {
            Error::Bundle("tinfoil-config endorsement_ref is not a SHA-256 reference".to_string())
        })?;
        decode_lower_hex_array::<32>(digest, "tinfoil-config endorsement_ref")?;
        let bundle = data.sigstore_bundle.as_ref().ok_or_else(|| {
            Error::Bundle("tinfoil-config collateral has no sigstore_bundle".to_string())
        })?;
        require_bundle_object(entry, bundle)?;
    }
    if let Some(issuer) = &data.issuer_certificate_der {
        if entry.format != COLLATERAL_CONFIG_ENDORSEMENT_V2_FORMAT {
            return Err(Error::Bundle(
                "issuer certificates require config-endorsement/v2 collateral".to_string(),
            ));
        }
        let issuer = decode_canonical_base64(
            &string(issuer, "issuer_certificate_der")?,
            "tinfoil-config issuer_certificate_der",
        )?;
        if issuer.is_empty() {
            return Err(Error::Bundle(
                "tinfoil-config issuer_certificate_der is empty".to_string(),
            ));
        }
    }
    let config = match &data.config_base64 {
        None if entry.format == COLLATERAL_CONFIG_ENDORSEMENT_V2_FORMAT => return Ok(None),
        None => {
            return Err(Error::Bundle(
                "tinfoil-config collateral has no config_base64".to_string(),
            ));
        }
        Some(config) => decode_canonical_base64(
            &string(config, "config_base64")?,
            "tinfoil-config config_base64",
        )?,
    };
    if config.is_empty() {
        return Err(Error::Bundle(
            "tinfoil-config config_base64 is empty".to_string(),
        ));
    }
    Ok(Some(Sha256::digest(&config).into()))
}

fn collateral_data<T: serde::de::DeserializeOwned>(entry: &CollateralEntry) -> Result<T, Error> {
    serde_json::from_value(entry.data.clone()).map_err(|e| {
        Error::Bundle(format!(
            "parsing {} collateral entry {:?}: {e}",
            entry.format, entry.id
        ))
    })
}

fn require_bundle_object(entry: &CollateralEntry, bundle: &serde_json::Value) -> Result<(), Error> {
    if bundle.is_object() {
        Ok(())
    } else {
        Err(Error::Bundle(format!(
            "collateral entry {:?} sigstore_bundle is not a JSON object",
            entry.id
        )))
    }
}

/// A `v`-prefixed semantic version written out in full, as Go's
/// `golang.org/x/mod/semver` reads one with `Canonical(tag) + Build(tag) ==
/// tag`: `vMAJOR.MINOR.PATCH` with no leading zeros, an optional `-`
/// pre-release of dot-separated identifiers (numeric ones without leading
/// zeros), and an optional `+` build suffix of dot-separated identifiers.
fn is_full_semver_tag(tag: &str) -> bool {
    fn identifiers(part: &str, pre_release: bool) -> bool {
        part.split('.').all(|id| {
            !id.is_empty()
                && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
                && !(pre_release
                    && id.len() > 1
                    && id.starts_with('0')
                    && id.bytes().all(|b| b.is_ascii_digit()))
        })
    }
    let Some(rest) = tag.strip_prefix('v') else {
        return false;
    };
    let (rest, build) = match rest.split_once('+') {
        Some((rest, build)) => (rest, Some(build)),
        None => (rest, None),
    };
    let (core, pre) = match rest.split_once('-') {
        Some((core, pre)) => (core, Some(pre)),
        None => (rest, None),
    };
    let numbers: Vec<&str> = core.split('.').collect();
    numbers.len() == 3
        && numbers.iter().all(|n| {
            !n.is_empty()
                && n.bytes().all(|b| b.is_ascii_digit())
                && (n.len() == 1 || !n.starts_with('0'))
        })
        && pre.is_none_or(|pre| identifiers(pre, true))
        && build.is_none_or(|build| identifiers(build, false))
}

fn endorsement_for<'a>(
    entries: &'a [CollateralEntry],
    format: &str,
    subject: &str,
) -> Option<&'a CollateralEntry> {
    entries.iter().find(|entry| {
        entry.role == ROLE_ENDORSEMENT
            && entry.format == format
            && entry.subjects.iter().any(|candidate| candidate == subject)
    })
}

fn require_eq(field: &str, actual: &str, expected: &str) -> Result<(), Error> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::Bundle(format!(
            "unsupported {field} {actual:?}; expected {expected:?}"
        )))
    }
}

pub(crate) fn decode_lower_hex_array<const N: usize>(
    value: &str,
    field: &str,
) -> Result<[u8; N], Error> {
    if !value
        .bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(Error::Bundle(format!("{field} is not lowercase hex")));
    }
    let decoded = hex::decode(value).map_err(|e| Error::Bundle(format!("{field}: {e}")))?;
    decoded
        .try_into()
        .map_err(|_| Error::Bundle(format!("{field} must be exactly {N} bytes")))
}

fn decode_canonical_base64(value: &str, field: &str) -> Result<Vec<u8>, Error> {
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(value)
        .map_err(|e| Error::Bundle(format!("decoding {field}: {e}")))?;
    if base64::engine::general_purpose::STANDARD.encode(&decoded) != value {
        return Err(Error::Bundle(format!("{field} is not canonical base64")));
    }
    Ok(decoded)
}

/// Reject duplicate member names recursively before typed deserialization.
pub(crate) fn reject_duplicate_members(raw: &[u8], context: &str) -> Result<(), Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(raw);
    StrictJson::deserialize(&mut deserializer)
        .map_err(|e| Error::Bundle(format!("parsing {context}: {e}")))?;
    deserializer
        .end()
        .map_err(|e| Error::Bundle(format!("parsing {context}: {e}")))
}

struct StrictJson;

impl<'de> Deserialize<'de> for StrictJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = StrictJson;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut names = HashSet::new();
        while let Some(name) = map.next_key::<String>()? {
            if !names.insert(name.clone()) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate object member {name:?}"
                )));
            }
            map.next_value::<StrictJson>()?;
        }
        Ok(StrictJson)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while seq.next_element::<StrictJson>()?.is_some() {}
        Ok(StrictJson)
    }

    fn visit_bool<E>(self, _: bool) -> Result<Self::Value, E> {
        Ok(StrictJson)
    }
    fn visit_i64<E>(self, _: i64) -> Result<Self::Value, E> {
        Ok(StrictJson)
    }
    fn visit_u64<E>(self, _: u64) -> Result<Self::Value, E> {
        Ok(StrictJson)
    }
    fn visit_f64<E>(self, _: f64) -> Result<Self::Value, E> {
        Ok(StrictJson)
    }
    fn visit_str<E>(self, _: &str) -> Result<Self::Value, E> {
        Ok(StrictJson)
    }
    fn visit_string<E>(self, _: String) -> Result<Self::Value, E> {
        Ok(StrictJson)
    }
    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(StrictJson)
    }
    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(StrictJson)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_members_are_rejected_recursively() {
        let err = reject_duplicate_members(br#"{"outer":{"x":1,"x":2}}"#, "test")
            .expect_err("duplicate should fail");
        assert!(err.to_string().contains("duplicate object member \"x\""));
    }

    /// A real document of the gateway's platform release; see
    /// `tests/fixtures/tinfoil-gateway/README.md`.
    const GATEWAY_DOCUMENT: &[u8] =
        include_bytes!("../tests/fixtures/tinfoil-gateway/document.json");

    fn gateway_document() -> serde_json::Value {
        serde_json::from_slice(GATEWAY_DOCUMENT).unwrap()
    }

    fn parse(doc: &serde_json::Value) -> Result<ResolvedAttestation, Error> {
        parse_document(&serde_json::to_vec(doc).unwrap())
    }

    fn refused(doc: &serde_json::Value, needle: &str) {
        let err = parse(doc).err().expect(needle).to_string();
        assert!(err.contains(needle), "{needle}: {err}");
    }

    /// A change to one collateral entry.
    type Mutation = fn(&mut serde_json::Value);

    fn entry_mut<'a>(doc: &'a mut serde_json::Value, id: &str) -> &'a mut serde_json::Value {
        doc["collateral"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|entry| entry["id"] == id)
            .unwrap()
    }

    /// The live shape: `collateral_format` names the v3 profile, and the
    /// collateral carries the CPU endorsements plus the runtime, platform
    /// and two freshness reference-values entries.
    #[test]
    fn the_captured_gateway_document_parses() {
        let doc = gateway_document();
        assert_eq!(doc["collateral_format"], COLLATERAL_V3_FORMAT);
        let formats: Vec<&str> = doc["collateral"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["format"].as_str().unwrap())
            .collect();
        assert_eq!(
            formats,
            [
                COLLATERAL_AMD_VCEK_V1_FORMAT,
                COLLATERAL_AMD_CRL_V1_FORMAT,
                COLLATERAL_IGVM_RUNTIME_V1_FORMAT,
                COLLATERAL_ARTIFACT_FRESHNESS_V1_FORMAT,
                COLLATERAL_SIGSTORE_PLATFORM_V1_FORMAT,
                COLLATERAL_ARTIFACT_FRESHNESS_V1_FORMAT,
            ]
        );
        let resolved = parse_document(GATEWAY_DOCUMENT).expect("parses");
        assert_eq!(resolved.platform, Platform::SevSnp);
        assert!(resolved.vcek_der.is_some() && resolved.crl_der.is_some());
        assert_eq!(resolved.config_sha256, None);
    }

    /// The profile is pinned: absent (which the provider reads as the legacy
    /// v2 profile), v2, or anything else is refused.
    #[test]
    fn collateral_format_must_be_the_v3_profile() {
        let mut doc = gateway_document();
        doc.as_object_mut().unwrap().remove("collateral_format");
        refused(&doc, "missing field `collateral_format`");
        for other in [
            "https://tinfoil.sh/predicate/attestation-collaterals/v2",
            "https://tinfoil.sh/predicate/attestation-collaterals/v4",
            "",
        ] {
            doc["collateral_format"] = other.into();
            refused(&doc, "unsupported collateral_format");
        }
    }

    #[test]
    fn runtime_collateral_is_decoded_strictly_and_its_manifest_hashed() {
        let mutations: [(Mutation, &str); 8] = [
            (|d| d["data"]["extra"] = 1.into(), "unknown field `extra`"),
            (
                |d| {
                    d["data"].as_object_mut().unwrap().remove("manifest_base64");
                },
                "missing field `manifest_base64`",
            ),
            (
                |d| d["data"]["repo"] = "tinfoilsh/other".into(),
                "repository",
            ),
            (
                |d| d["data"]["tag"] = "v0.15".into(),
                "full semantic version",
            ),
            (
                |d| d["data"]["tag"] = "0.15.3-rc.1".into(),
                "full semantic version",
            ),
            (
                |d| d["data"]["digest"] = "00".repeat(32).into(),
                "does not hash",
            ),
            (
                |d| d["data"]["sigstore_bundle"] = "{}".into(),
                "not a JSON object",
            ),
            (
                |d| d["role"] = "endorsement".into(),
                "conflicting collateral entry",
            ),
        ];
        for (mutate, needle) in mutations {
            let mut doc = gateway_document();
            mutate(entry_mut(&mut doc, "runtime"));
            refused(&doc, needle);
        }
        // The reserved ID and the format go together.
        let mut doc = gateway_document();
        entry_mut(&mut doc, "runtime")["id"] = "runtime-2".into();
        refused(&doc, "conflicting collateral entry");
        let mut doc = gateway_document();
        entry_mut(&mut doc, "platform")["id"] = "runtime".into();
        doc["collateral"].as_array_mut().unwrap().remove(2);
        refused(&doc, "conflicting collateral entry");
    }

    #[test]
    fn platform_and_freshness_collateral_are_decoded_strictly() {
        let mutations: [(&str, Mutation, &str); 6] = [
            (
                "platform",
                |d| d["data"]["extra"] = 1.into(),
                "unknown field",
            ),
            (
                "platform",
                |d| d["data"]["digest"] = "AB".into(),
                "lowercase hex",
            ),
            (
                "platform",
                |d| d["data"]["sigstore_bundle"] = serde_json::Value::Null,
                "not a JSON object",
            ),
            (
                "runtime-freshness",
                |d| d["data"]["extra"] = 1.into(),
                "unknown field",
            ),
            (
                "platform-freshness",
                |d| d["id"] = "code-freshness".into(),
                "conflicting artifact freshness",
            ),
            (
                "runtime-freshness",
                |d| d["format"] = COLLATERAL_SIGSTORE_FRESHNESS_V1_FORMAT.into(),
                "conflicting artifact freshness",
            ),
        ];
        for (id, mutate, needle) in mutations {
            let mut doc = gateway_document();
            mutate(entry_mut(&mut doc, id));
            refused(&doc, needle);
        }
        // Each reserved freshness ID goes with exactly its format and role:
        // under another format, even one nobody reads, it is refused rather
        // than passed through as an unknown entry.
        for id in ["runtime-freshness", "platform-freshness"] {
            let mut doc = gateway_document();
            let entry = entry_mut(&mut doc, id);
            entry["format"] = "https://example.com/collateral/unknown/v1".into();
            entry["data"] = serde_json::json!({});
            refused(&doc, "conflicting artifact freshness");
            let mut doc = gateway_document();
            entry_mut(&mut doc, id)["role"] = "endorsement".into();
            refused(&doc, "conflicting artifact freshness");
        }
    }

    /// The legacy profile's code provenance and Sigstore freshness have no
    /// place in a v3 collateral set; a format nobody reads is held to the
    /// entry shape only.
    #[test]
    fn legacy_entries_are_refused_and_unknown_formats_only_shape_checked() {
        let push = |entry: serde_json::Value| {
            let mut doc = gateway_document();
            doc["collateral"].as_array_mut().unwrap().push(entry);
            doc
        };
        for format in [
            COLLATERAL_SIGSTORE_CODE_V1_FORMAT,
            COLLATERAL_SIGSTORE_FRESHNESS_V1_FORMAT,
        ] {
            refused(
                &push(serde_json::json!({
                    "id": "code", "role": "reference-values", "format": format,
                    "data": {"sigstore_bundle": {}},
                })),
                "legacy format",
            );
        }
        let unknown = |data: serde_json::Value| {
            push(serde_json::json!({
                "id": "gpu-rim", "role": "endorsement",
                "format": "https://tinfoil.sh/collateral/nvidia-gpu/v1",
                "subjects": ["gpu0"], "data": data,
            }))
        };
        parse(&unknown(serde_json::json!({"anything": [1, 2]}))).expect("shape only");
        refused(&unknown(serde_json::json!([])), "not a JSON object");
    }

    #[test]
    fn config_collateral_follows_its_format() {
        use base64::engine::general_purpose::STANDARD as B64;
        let digest = format!("sha256:{}", "ab".repeat(32));
        let with = |format: &str, data: serde_json::Value| {
            let mut doc = gateway_document();
            doc["collateral"]
                .as_array_mut()
                .unwrap()
                .push(serde_json::json!({
                    "id": "tinfoil-config", "role": "reference-values",
                    "format": format, "data": data,
                }));
            doc
        };
        let config_hash: [u8; 32] = Sha256::digest(b"config").into();
        let carried = |doc| parse(&doc).expect("parses").config_sha256;

        assert_eq!(
            carried(with(
                COLLATERAL_CONFIG_V1_FORMAT,
                serde_json::json!({"config_base64": B64.encode("config")}),
            )),
            Some(config_hash)
        );
        assert_eq!(
            carried(with(
                COLLATERAL_CONFIG_ENDORSEMENT_V1_FORMAT,
                serde_json::json!({
                    "endorsement_ref": digest, "config_base64": B64.encode("config"),
                    "sigstore_bundle": {},
                }),
            )),
            Some(config_hash)
        );
        assert_eq!(
            carried(with(
                COLLATERAL_CONFIG_ENDORSEMENT_V2_FORMAT,
                serde_json::json!({
                    "endorsement_ref": digest, "sigstore_bundle": {},
                    "issuer_certificate_der": B64.encode("root"),
                }),
            )),
            None
        );

        let cases = [
            (
                COLLATERAL_CONFIG_V1_FORMAT,
                serde_json::json!({"config_base64": B64.encode("config"), "endorsement_ref": digest}),
                "cannot carry an endorsement",
            ),
            (
                COLLATERAL_CONFIG_V1_FORMAT,
                serde_json::json!({"config_base64": B64.encode("config"), "sigstore_bundle": null}),
                "cannot carry an endorsement",
            ),
            (
                COLLATERAL_CONFIG_V1_FORMAT,
                serde_json::json!({}),
                "no config_base64",
            ),
            (
                COLLATERAL_CONFIG_V1_FORMAT,
                serde_json::json!({"config_base64": ""}),
                "is empty",
            ),
            (
                COLLATERAL_CONFIG_V1_FORMAT,
                serde_json::json!({"config_base64": B64.encode("config"), "audit": true}),
                "unknown field",
            ),
            (
                COLLATERAL_CONFIG_ENDORSEMENT_V1_FORMAT,
                serde_json::json!({"endorsement_ref": digest, "sigstore_bundle": {}}),
                "no config_base64",
            ),
            (
                COLLATERAL_CONFIG_ENDORSEMENT_V1_FORMAT,
                serde_json::json!({
                    "endorsement_ref": "test-reference", "sigstore_bundle": {},
                    "config_base64": B64.encode("config"),
                }),
                "not a SHA-256 reference",
            ),
            (
                COLLATERAL_CONFIG_ENDORSEMENT_V1_FORMAT,
                serde_json::json!({
                    "endorsement_ref": digest, "sigstore_bundle": {},
                    "config_base64": B64.encode("config"),
                    "issuer_certificate_der": B64.encode("root"),
                }),
                "require config-endorsement/v2",
            ),
            (
                COLLATERAL_CONFIG_ENDORSEMENT_V2_FORMAT,
                serde_json::json!({"endorsement_ref": digest}),
                "no sigstore_bundle",
            ),
        ];
        for (format, data, needle) in cases {
            refused(&with(format, data), needle);
        }

        // The reserved ID and the config formats go together.
        let mut doc = with(
            COLLATERAL_CONFIG_V1_FORMAT,
            serde_json::json!({"config_base64": B64.encode("config")}),
        );
        entry_mut(&mut doc, "tinfoil-config")["id"] = "config".into();
        refused(&doc, "conflicting collateral entry");
    }

    #[test]
    fn full_semver_tags_are_the_ones_go_writes_out_in_full() {
        for tag in [
            "v0.15.3-rc.1",
            "v0.15.3-rc.1+gpu",
            "v0.15.3",
            "v10.0.0-alpha-1.0a",
        ] {
            assert!(is_full_semver_tag(tag), "{tag}");
        }
        for tag in [
            "0.15.3",
            "v0.15",
            "v0",
            "v01.2.3",
            "v1.2.3-01",
            "v1.2.3-",
            "v1.2.3+",
            "v1.2.3-rc..1",
            "v1.2.3-rc_1",
            "V1.2.3",
        ] {
            assert!(!is_full_semver_tag(tag), "{tag}");
        }
    }

    #[test]
    fn report_data_algorithm_is_domain_separated() {
        let nonce = [1; 32];
        let crypto = [2; 32];
        let device = [3; 32];
        let actual = compute_report_data(&nonce, &crypto, &device);
        let mut hasher = Sha256::new();
        hasher.update(REPORT_DATA_V1_ALGORITHM);
        hasher.update(nonce);
        hasher.update(crypto);
        hasher.update(device);
        assert_eq!(&actual[..32], hasher.finalize().as_slice());
        assert_eq!(&actual[32..], &[0; 32]);
    }
}
