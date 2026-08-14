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

const ROLE_ENDORSEMENT: &str = "endorsement";
const ROLE_REFERENCE_VALUES: &str = "reference-values";
const SUBJECT_CPU: &str = "cpu";
const CRYPTO_ID_TLS: &str = "tls";
const CRYPTO_ID_HPKE: &str = "hpke";

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

/// TEE platform identified by the CPU evidence format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    SevSnp,
    Tdx,
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
    reject_duplicate_members(raw, "attestation document")?;
    let doc: Document = serde_json::from_slice(raw)
        .map_err(|e| Error::Bundle(format!("parsing attestation document: {e}")))?;

    require_eq("document format", &doc.format, ATTESTATION_V3_FORMAT)?;
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

    let platform = match doc.cpu_evidence.format.as_str() {
        SEV_SNP_REPORT_V1_FORMAT => Platform::SevSnp,
        TDX_QUOTE_V1_FORMAT => Platform::Tdx,
        other => {
            return Err(Error::Bundle(format!(
                "unsupported CPU evidence format: {other}"
            )));
        }
    };
    let report_bytes = decode_canonical_base64(
        &doc.cpu_evidence.report_base64,
        "cpu_evidence.report_base64",
    )?;
    if report_bytes.is_empty() {
        return Err(Error::Bundle("CPU evidence report is empty".to_string()));
    }

    validate_collateral_entries(&doc.collateral)?;
    let (vcek_der, ask_der, ark_der, crl_der) = match platform {
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
            (Some(vcek_der), Some(ask_der), Some(ark_der), Some(crl_der))
        }
        Platform::Tdx => (None, None, None, None),
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
    }
    Ok(())
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

fn decode_lower_hex_array<const N: usize>(value: &str, field: &str) -> Result<[u8; N], Error> {
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
fn reject_duplicate_members(raw: &[u8], context: &str) -> Result<(), Error> {
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
