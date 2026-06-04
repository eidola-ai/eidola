//! Parse the pinned [`SIGSTORE_TRUSTED_ROOT_JSON`] into typed Fulcio CA +
//! Rekor key + CT log key collections.
//!
//! The verifier uses these to verify the CI's Fulcio cert chain and the
//! Rekor inclusion proofs / SET for both the CI sigstore bundle and the
//! human SSH `rekord` entries. The trust root is a snapshot of
//! Sigstore's upstream public material (`sigstore-trusted-root.json`),
//! refreshed in lockstep with each release.
//!
//! Trust loading is intentionally local (per-call) rather than process-
//! global. Each verification operation passes the parsed [`TrustedRoot`]
//! to the primitives that need it; this keeps the API explicit and lets
//! tests inject fixtures.

use serde::Deserialize;

use crate::error::AppError;
use crate::trust_root::SIGSTORE_TRUSTED_ROOT_JSON;

/// Parsed view of `sigstore-trusted-root.json` — the subset our verifier
/// actually consumes.
#[derive(Debug, Clone)]
pub struct TrustedRoot {
    pub fulcio_cas: Vec<FulcioCa>,
    pub rekor_keys: Vec<RekorKey>,
    pub ctlog_keys: Vec<CtLogKey>,
}

/// A Fulcio CA — the root or intermediate cert plus its validity window.
/// The verifier picks the CA whose validity contains the leaf cert's
/// `notBefore`, then walks the chain.
#[derive(Debug, Clone)]
pub struct FulcioCa {
    /// Cert chain in declared order (typically leaf-most-intermediate
    /// first, root last). Each entry is DER bytes; PEM decoding happens
    /// in the loader.
    pub cert_chain_der: Vec<Vec<u8>>,
    pub valid_from: i64,
    pub valid_until: Option<i64>,
}

/// A Rekor log signing key — used to verify SignedEntryTimestamps and
/// log checkpoints.
#[derive(Debug, Clone)]
pub struct RekorKey {
    /// SHA-256 of the key (the Rekor log ID). Matched against
    /// `tlogEntry.logId.keyId` to pick the right key for the entry.
    pub log_id: [u8; 32],
    /// SubjectPublicKeyInfo bytes (DER).
    pub spki_der: Vec<u8>,
    pub key_details: KeyDetails,
    pub valid_from: i64,
    pub valid_until: Option<i64>,
}

/// A CT log signing key — used to verify embedded SCTs in Fulcio leaf
/// certificates.
#[derive(Debug, Clone)]
pub struct CtLogKey {
    pub log_id: [u8; 32],
    pub spki_der: Vec<u8>,
    pub key_details: KeyDetails,
    pub valid_from: i64,
    pub valid_until: Option<i64>,
}

/// Sigstore's public-key algorithm + curve identifier. The verifier
/// dispatches signature verification based on this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyDetails {
    EcdsaP256Sha256,
    EcdsaP384Sha384,
    Ed25519,
}

impl KeyDetails {
    fn from_str(s: &str) -> Result<Self, AppError> {
        match s {
            "PKIX_ECDSA_P256_SHA_256" => Ok(KeyDetails::EcdsaP256Sha256),
            "PKIX_ECDSA_P384_SHA_384" => Ok(KeyDetails::EcdsaP384Sha384),
            "PKIX_ED25519" => Ok(KeyDetails::Ed25519),
            other => Err(AppError::Update {
                message: format!("unsupported KeyDetails `{other}` in sigstore trusted root"),
            }),
        }
    }
}

/// Load and parse [`SIGSTORE_TRUSTED_ROOT_JSON`] (compile-time-embedded
/// snapshot of Sigstore's public trust material). Any failure here is
/// a build-issue, not a runtime one — the JSON is fixed at compile time.
pub fn load() -> Result<TrustedRoot, AppError> {
    load_from_str(SIGSTORE_TRUSTED_ROOT_JSON)
}

/// Like [`load`] but with explicit JSON input — used by tests.
pub fn load_from_str(json: &str) -> Result<TrustedRoot, AppError> {
    let parsed: RawTrustedRoot = serde_json::from_str(json).map_err(|e| AppError::Update {
        message: format!("parsing sigstore-trusted-root.json: {e}"),
    })?;

    let fulcio_cas = parsed
        .certificate_authorities
        .into_iter()
        .map(parse_fulcio_ca)
        .collect::<Result<Vec<_>, _>>()?;
    let rekor_keys = parsed
        .tlogs
        .into_iter()
        .map(parse_rekor_key)
        .collect::<Result<Vec<_>, _>>()?;
    let ctlog_keys = parsed
        .ctlogs
        .into_iter()
        .map(parse_ctlog_key)
        .collect::<Result<Vec<_>, _>>()?;

    if fulcio_cas.is_empty() {
        return Err(AppError::Update {
            message: "sigstore-trusted-root.json has no Fulcio CAs".into(),
        });
    }
    if rekor_keys.is_empty() {
        return Err(AppError::Update {
            message: "sigstore-trusted-root.json has no Rekor keys".into(),
        });
    }

    Ok(TrustedRoot {
        fulcio_cas,
        rekor_keys,
        ctlog_keys,
    })
}

// ---------------------------------------------------------------------------
// Raw JSON shape — internal, mapped to the typed view by parse_* helpers
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTrustedRoot {
    #[serde(default)]
    tlogs: Vec<RawTlog>,
    #[serde(default)]
    certificate_authorities: Vec<RawCa>,
    #[serde(default)]
    ctlogs: Vec<RawTlog>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawCa {
    cert_chain: RawCertChain,
    valid_for: RawValidFor,
}

#[derive(Deserialize)]
struct RawCertChain {
    certificates: Vec<RawCertEntry>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawCertEntry {
    raw_bytes: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTlog {
    public_key: RawPublicKey,
    log_id: RawLogId,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawPublicKey {
    raw_bytes: String,
    key_details: String,
    #[serde(default)]
    valid_for: Option<RawValidFor>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawLogId {
    key_id: String,
}

#[derive(Deserialize)]
struct RawValidFor {
    start: String,
    #[serde(default)]
    end: Option<String>,
}

fn parse_fulcio_ca(raw: RawCa) -> Result<FulcioCa, AppError> {
    let chain = raw
        .cert_chain
        .certificates
        .into_iter()
        .map(|c| decode_base64(&c.raw_bytes, "certificate raw_bytes"))
        .collect::<Result<Vec<_>, _>>()?;
    let valid_from = parse_rfc3339_unix(&raw.valid_for.start)?;
    let valid_until = raw
        .valid_for
        .end
        .as_deref()
        .map(parse_rfc3339_unix)
        .transpose()?;
    Ok(FulcioCa {
        cert_chain_der: chain,
        valid_from,
        valid_until,
    })
}

fn parse_rekor_key(raw: RawTlog) -> Result<RekorKey, AppError> {
    let spki_der = decode_base64(&raw.public_key.raw_bytes, "tlog public_key raw_bytes")?;
    let key_id = decode_base64(&raw.log_id.key_id, "tlog log_id keyId")?;
    let log_id = key_id.as_slice().try_into().map_err(|_| AppError::Update {
        message: format!(
            "tlog log_id.keyId is {} bytes, expected 32 (sha256)",
            key_id.len()
        ),
    })?;
    let key_details = KeyDetails::from_str(&raw.public_key.key_details)?;
    let (valid_from, valid_until) = parse_optional_valid_for(raw.public_key.valid_for.as_ref())?;
    Ok(RekorKey {
        log_id,
        spki_der,
        key_details,
        valid_from,
        valid_until,
    })
}

fn parse_ctlog_key(raw: RawTlog) -> Result<CtLogKey, AppError> {
    let spki_der = decode_base64(&raw.public_key.raw_bytes, "ctlog public_key raw_bytes")?;
    let key_id = decode_base64(&raw.log_id.key_id, "ctlog log_id keyId")?;
    let log_id = key_id.as_slice().try_into().map_err(|_| AppError::Update {
        message: format!(
            "ctlog log_id.keyId is {} bytes, expected 32 (sha256)",
            key_id.len()
        ),
    })?;
    let key_details = KeyDetails::from_str(&raw.public_key.key_details)?;
    let (valid_from, valid_until) = parse_optional_valid_for(raw.public_key.valid_for.as_ref())?;
    Ok(CtLogKey {
        log_id,
        spki_der,
        key_details,
        valid_from,
        valid_until,
    })
}

fn parse_optional_valid_for(
    valid_for: Option<&RawValidFor>,
) -> Result<(i64, Option<i64>), AppError> {
    match valid_for {
        Some(v) => {
            let start = parse_rfc3339_unix(&v.start)?;
            let end = v.end.as_deref().map(parse_rfc3339_unix).transpose()?;
            Ok((start, end))
        }
        // Missing validFor → treat as valid forever. The sigstore-trusted-
        // root format allows omitting it for keys with no expiry; we don't
        // want to invent a window for those.
        None => Ok((0, None)),
    }
}

fn decode_base64(s: &str, field: &str) -> Result<Vec<u8>, AppError> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(s.as_bytes())
        .map_err(|e| AppError::Update {
            message: format!("base64-decoding `{field}`: {e}"),
        })
}

fn parse_rfc3339_unix(s: &str) -> Result<i64, AppError> {
    // Sigstore's TrustedRoot uses RFC3339 with `Z` suffix. We don't want to
    // pull a full date-time crate just for this; parse the components by
    // hand. Format: `YYYY-MM-DDTHH:MM:SS[.fff]Z`.
    let s = s.trim_end_matches('Z');
    let (date, time) = s.split_once('T').ok_or_else(|| AppError::Update {
        message: format!("malformed RFC3339 timestamp `{s}`"),
    })?;
    let mut date_parts = date.split('-');
    let year: i64 = date_parts
        .next()
        .and_then(|x| x.parse().ok())
        .ok_or_else(|| AppError::Update {
            message: format!("bad year in `{s}`"),
        })?;
    let month: i64 = date_parts
        .next()
        .and_then(|x| x.parse().ok())
        .ok_or_else(|| AppError::Update {
            message: format!("bad month in `{s}`"),
        })?;
    let day: i64 = date_parts
        .next()
        .and_then(|x| x.parse().ok())
        .ok_or_else(|| AppError::Update {
            message: format!("bad day in `{s}`"),
        })?;

    // Drop fractional seconds if present — we only care about whole-second
    // precision here.
    let time = time.split('.').next().unwrap_or(time);
    let mut time_parts = time.split(':');
    let hour: i64 = time_parts
        .next()
        .and_then(|x| x.parse().ok())
        .ok_or_else(|| AppError::Update {
            message: format!("bad hour in `{s}`"),
        })?;
    let minute: i64 = time_parts
        .next()
        .and_then(|x| x.parse().ok())
        .ok_or_else(|| AppError::Update {
            message: format!("bad minute in `{s}`"),
        })?;
    let second: i64 = time_parts
        .next()
        .and_then(|x| x.parse().ok())
        .ok_or_else(|| AppError::Update {
            message: format!("bad second in `{s}`"),
        })?;

    // Days-from-civil — Howard Hinnant's algorithm. Handles leap years
    // correctly and matches `chrono`'s `NaiveDate::from_ymd` for the same
    // (year, month, day). UTC only — fine for our use.
    let (y, m) = if month <= 2 {
        (year - 1, month + 12)
    } else {
        (year, month)
    };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let doy = (153 * (m - 3) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    Ok(days * 86400 + hour * 3600 + minute * 60 + second)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_pinned_trusted_root() {
        let trust = load().expect("the pinned trusted root must parse");
        // Loose assertions — pinned-root content can rotate over time.
        assert!(!trust.fulcio_cas.is_empty(), "expected ≥1 Fulcio CA");
        assert!(!trust.rekor_keys.is_empty(), "expected ≥1 Rekor key");
    }

    #[test]
    fn parse_rfc3339_unix_basic() {
        assert_eq!(parse_rfc3339_unix("1970-01-01T00:00:00Z").unwrap(), 0);
        assert_eq!(parse_rfc3339_unix("1970-01-01T00:00:01Z").unwrap(), 1);
        // Known checkpoint: 2021-01-12T11:53:27Z (matches Rekor's first key
        // validity start, used in production).
        assert_eq!(
            parse_rfc3339_unix("2021-01-12T11:53:27Z").unwrap(),
            1_610_452_407
        );
    }

    #[test]
    fn parse_rfc3339_drops_fractional_seconds() {
        assert_eq!(
            parse_rfc3339_unix("2025-04-08T06:59:43.123Z").unwrap(),
            parse_rfc3339_unix("2025-04-08T06:59:43Z").unwrap()
        );
    }

    #[test]
    fn key_details_rejects_unknown() {
        assert!(KeyDetails::from_str("UNKNOWN_ALG").is_err());
    }

    #[test]
    fn load_from_str_rejects_empty_fulcio() {
        let json = r#"{"tlogs":[{"publicKey":{"rawBytes":"AA==","keyDetails":"PKIX_ECDSA_P256_SHA_256","validFor":{"start":"2021-01-12T11:53:27Z"}},"logId":{"keyId":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="}}],"certificateAuthorities":[],"ctlogs":[]}"#;
        let err = load_from_str(json).unwrap_err();
        assert!(format!("{err}").contains("Fulcio"), "got: {err}");
    }
}
