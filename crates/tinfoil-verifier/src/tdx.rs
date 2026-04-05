//! TDX attestation verification using the `dcap-qvl` crate.
//!
//! Handles Intel TDX Quote V4 parsing, collateral fetching from Intel PCS,
//! and cryptographic verification. Delegates signature verification and TCB
//! evaluation to [`dcap_qvl`].
//!
//! **Limitation:** Collateral fetching requires the TDX quote to contain an
//! embedded PCK certificate chain (certification data type 5). Quotes using
//! encrypted-PPID collateral retrieval (types 6/7) are not supported. This is
//! sufficient for Tinfoil containers, which always embed the PCK chain.

use std::time::{SystemTime, UNIX_EPOCH};

use dcap_qvl::QuoteCollateralV3;
use dcap_qvl::quote::Quote;
use dcap_qvl::verify::rustcrypto::verify as dcap_verify;
use der::Decode;
use x509_cert::ext::pkix::{
    CrlDistributionPoints,
    name::{DistributionPointName, GeneralName},
};

use crate::Error;

const INTEL_PCS_BASE: &str = "https://api.trustedservices.intel.com";

/// Result of a successful TDX quote verification.
pub struct TdxVerification {
    /// RTMR1 (48 bytes).
    pub rtmr1: [u8; 48],
    /// RTMR2 (48 bytes).
    pub rtmr2: [u8; 48],
    /// Full report_data (64 bytes). First 32 bytes = TLS fingerprint.
    pub report_data: [u8; 64],
}

/// Build a measurement string from RTMR1 and RTMR2.
///
/// Returns hex(RTMR1) + hex(RTMR2) — a 192-character hex string that naturally
/// distinguishes from SEV-SNP measurements (96 characters).
pub fn measurement_hex(rtmr1: &[u8; 48], rtmr2: &[u8; 48]) -> String {
    format!("{}{}", hex::encode(rtmr1), hex::encode(rtmr2))
}

/// Verify a TDX Quote V4 against fetched collateral.
///
/// 1. Verifies the quote's ECDSA signature against Intel's root CA
/// 2. Validates TCB policy and TDX module identity
/// 3. Extracts RTMR1, RTMR2, and report_data
pub fn verify_quote(
    raw_quote: &[u8],
    collateral: &QuoteCollateralV3,
) -> Result<TdxVerification, Error> {
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| Error::Tdx(format!("system time error: {e}")))?
        .as_secs();

    let verified = dcap_verify(raw_quote, collateral, now_secs)
        .map_err(|e| Error::Tdx(format!("TDX quote verification failed: {e}")))?;

    let td_report = verified
        .report
        .as_td10()
        .ok_or_else(|| Error::Tdx("expected TDX report, got SGX".to_string()))?;

    Ok(TdxVerification {
        rtmr1: td_report.rt_mr1,
        rtmr2: td_report.rt_mr2,
        report_data: td_report.report_data,
    })
}

/// Fetch TDX attestation collateral from Intel's Provisioning Certification Service.
///
/// Extracts FMSPC and CA type from the quote's embedded PCK certificate chain
/// (certification data type 5), then fetches TCB info, QE identity, PCK CRL,
/// and root CA CRL from Intel PCS.
pub async fn fetch_collateral(
    client: &reqwest::Client,
    raw_quote: &[u8],
) -> Result<QuoteCollateralV3, Error> {
    let quote = Quote::parse(raw_quote)
        .map_err(|e| Error::Tdx(format!("failed to parse TDX quote: {e}")))?;

    let fmspc = quote
        .fmspc()
        .map_err(|e| Error::Tdx(format!("failed to extract FMSPC: {e}")))?;
    let fmspc_hex = hex::encode_upper(fmspc);

    let ca = quote
        .ca()
        .map_err(|e| Error::Tdx(format!("failed to extract CA type: {e}")))?;

    let pck_certificate_chain = quote
        .raw_cert_chain()
        .ok()
        .and_then(|c| std::str::from_utf8(c).ok())
        .map(|c| c.to_string());

    // Fetch all collateral in parallel
    let (tcb_resp, qe_resp, pck_crl_resp) = tokio::try_join!(
        fetch_tcb_info(client, &fmspc_hex),
        fetch_qe_identity(client),
        fetch_pck_crl(client, ca),
    )?;

    // Fetch root CA CRL: try the PCS endpoint first, fall back to the CRL
    // Distribution Point extracted from the root certificate in the QE identity
    // issuer chain (matches dcap-qvl's Intel PCS flow).
    let root_ca_crl = match fetch_root_ca_crl_endpoint(client).await {
        Ok(crl) => crl,
        Err(e) => {
            tracing::debug!("rootcacrl endpoint failed ({e}), trying CDP fallback");
            fetch_root_ca_crl_from_cdp(client, &qe_resp.issuer_chain).await?
        }
    };

    Ok(QuoteCollateralV3 {
        pck_crl_issuer_chain: pck_crl_resp.issuer_chain,
        root_ca_crl,
        pck_crl: pck_crl_resp.body,
        tcb_info_issuer_chain: tcb_resp.issuer_chain,
        tcb_info: tcb_resp.inner_json,
        tcb_info_signature: tcb_resp.signature,
        qe_identity_issuer_chain: qe_resp.issuer_chain,
        qe_identity: qe_resp.inner_json,
        qe_identity_signature: qe_resp.signature,
        pck_certificate_chain,
    })
}

struct SignedJsonResponse {
    issuer_chain: String,
    inner_json: String,
    signature: Vec<u8>,
}

struct CrlResponse {
    issuer_chain: String,
    body: Vec<u8>,
}

/// Fetch TDX TCB info from Intel PCS.
async fn fetch_tcb_info(
    client: &reqwest::Client,
    fmspc_hex: &str,
) -> Result<SignedJsonResponse, Error> {
    let url = format!("{INTEL_PCS_BASE}/tdx/certification/v4/tcb?fmspc={fmspc_hex}");
    let resp = client.get(&url).send().await?.error_for_status()?;

    let issuer_chain = extract_header(&resp, "TCB-Info-Issuer-Chain")
        .or_else(|| extract_header(&resp, "SGX-TCB-Info-Issuer-Chain"))
        .unwrap_or_default();

    let body: serde_json::Value = resp.json().await?;
    extract_signed_json(&body, "tcbInfo").map(|(inner_json, signature)| SignedJsonResponse {
        issuer_chain,
        inner_json,
        signature,
    })
}

/// Fetch TDX QE identity from Intel PCS.
async fn fetch_qe_identity(client: &reqwest::Client) -> Result<SignedJsonResponse, Error> {
    let url = format!("{INTEL_PCS_BASE}/tdx/certification/v4/qe/identity?update=standard");
    let resp = client.get(&url).send().await?.error_for_status()?;

    let issuer_chain =
        extract_header(&resp, "SGX-Enclave-Identity-Issuer-Chain").unwrap_or_default();

    let body: serde_json::Value = resp.json().await?;
    extract_signed_json(&body, "enclaveIdentity").map(|(inner_json, signature)| {
        SignedJsonResponse {
            issuer_chain,
            inner_json,
            signature,
        }
    })
}

/// Fetch PCK CRL from Intel PCS (always uses SGX path).
async fn fetch_pck_crl(client: &reqwest::Client, ca: &str) -> Result<CrlResponse, Error> {
    let url = format!("{INTEL_PCS_BASE}/sgx/certification/v4/pckcrl?ca={ca}&encoding=der");
    let resp = client.get(&url).send().await?.error_for_status()?;

    let issuer_chain = extract_header(&resp, "SGX-PCK-CRL-Issuer-Chain").unwrap_or_default();

    let body = resp.bytes().await?.to_vec();

    Ok(CrlResponse { issuer_chain, body })
}

/// Fetch root CA CRL from the Intel PCS endpoint (hex-encoded DER response).
async fn fetch_root_ca_crl_endpoint(client: &reqwest::Client) -> Result<Vec<u8>, Error> {
    let url = format!("{INTEL_PCS_BASE}/sgx/certification/v4/rootcacrl");
    let text = client
        .get(&url)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;

    let trimmed = text.trim();
    let len = trimmed.len();
    let preview_max = 64;
    let preview: String = trimmed.chars().take(preview_max).collect();

    hex::decode(trimmed).map_err(|e| {
        Error::Tdx(format!(
            "failed to decode root CA CRL hex (len={len}, preview=\"{preview}\"): {e}"
        ))
    })
}

/// Fetch root CA CRL via the CRL Distribution Point from the root certificate
/// in the issuer chain. This is the fallback path matching dcap-qvl's Intel PCS
/// verification flow.
async fn fetch_root_ca_crl_from_cdp(
    client: &reqwest::Client,
    issuer_chain_pem: &str,
) -> Result<Vec<u8>, Error> {
    let cdp_url = extract_root_ca_cdp(issuer_chain_pem)?;
    tracing::info!("Fetching root CA CRL from CDP: {cdp_url}");

    client
        .get(&cdp_url)
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| Error::Tdx(format!("failed to fetch root CA CRL from CDP: {e}")))
}

/// Extract the CRL Distribution Point URL from the last (root) certificate
/// in a PEM chain.
fn extract_root_ca_cdp(pem_chain: &str) -> Result<String, Error> {
    // Find the last certificate in the PEM chain (the root CA)
    let last_cert_pem = pem_chain
        .rmatch_indices("-----BEGIN CERTIFICATE-----")
        .next()
        .map(|(start, _)| &pem_chain[start..])
        .ok_or_else(|| Error::Tdx("no certificate found in issuer chain".to_string()))?;

    let cert_der = crate::sevsnp::decode_base64(last_cert_pem)?;
    let cert = x509_cert::Certificate::from_der(&cert_der)
        .map_err(|e| Error::Tdx(format!("failed to parse root cert: {e}")))?;

    let Some(extensions) = &cert.tbs_certificate.extensions else {
        return Err(Error::Tdx(
            "no CRL Distribution Points extension found in root certificate".to_string(),
        ));
    };

    for ext in extensions {
        if ext.extn_id.to_string() != "2.5.29.31" {
            continue;
        }

        let cdp: CrlDistributionPoints = CrlDistributionPoints::from_der(ext.extn_value.as_bytes())
            .map_err(|e| Error::Tdx(format!("failed to parse CRL Distribution Points: {e}")))?;

        for dist_point in &cdp.0 {
            let Some(dist_point_name) = &dist_point.distribution_point else {
                continue;
            };
            let DistributionPointName::FullName(general_names) = dist_point_name else {
                continue;
            };
            for general_name in general_names {
                let GeneralName::UniformResourceIdentifier(uri) = general_name else {
                    continue;
                };
                return Ok(uri.to_string());
            }
        }
    }

    Err(Error::Tdx(
        "no CRL Distribution Point found in root certificate".to_string(),
    ))
}

/// Extract a URL-decoded header value from an HTTP response.
fn extract_header(resp: &reqwest::Response, name: &str) -> Option<String> {
    resp.headers()
        .get(name)?
        .to_str()
        .ok()
        .and_then(|v| urlencoding::decode(v).ok())
        .map(|v| v.into_owned())
}

/// Extract the inner JSON object and hex-decoded signature from an Intel PCS
/// signed JSON wrapper (e.g. `{"tcbInfo": {...}, "signature": "hex"}`).
fn extract_signed_json(
    body: &serde_json::Value,
    inner_key: &str,
) -> Result<(String, Vec<u8>), Error> {
    let inner = body
        .get(inner_key)
        .ok_or_else(|| Error::Tdx(format!("missing '{inner_key}' in PCS response")))?;
    let inner_json = serde_json::to_string(inner)
        .map_err(|e| Error::Tdx(format!("failed to serialize '{inner_key}': {e}")))?;

    let sig_hex = body
        .get("signature")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::Tdx("missing 'signature' in PCS response".to_string()))?;
    let signature = hex::decode(sig_hex)
        .map_err(|e| Error::Tdx(format!("failed to decode signature hex: {e}")))?;

    Ok((inner_json, signature))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Intel SGX Provisioning Certification Root CA — the certificate our CDP
    // extraction will encounter in production issuer chains.
    const INTEL_SGX_ROOT_CA_PEM: &str = "\
-----BEGIN CERTIFICATE-----
MIICjzCCAjSgAwIBAgIUImUM1lqdNInzg7SVUr9QGzknBqwwCgYIKoZIzj0EAwIw
aDEaMBgGA1UEAwwRSW50ZWwgU0dYIFJvb3QgQ0ExGjAYBgNVBAoMEUludGVsIENv
cnBvcmF0aW9uMRQwEgYDVQQHDAtTYW50YSBDbGFyYTELMAkGA1UECAwCQ0ExCzAJ
BgNVBAYTAlVTMB4XDTE4MDUyMTEwNDUxMFoXDTQ5MTIzMTIzNTk1OVowaDEaMBgG
A1UEAwwRSW50ZWwgU0dYIFJvb3QgQ0ExGjAYBgNVBAoMEUludGVsIENvcnBvcmF0
aW9uMRQwEgYDVQQHDAtTYW50YSBDbGFyYTELMAkGA1UECAwCQ0ExCzAJBgNVBAYT
AlVTMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEC6nEwMDIYZOj/iPWsCzaEKi7
1OiOSLRFhWGjbnBVJfVnkY4u3IjkDYYL0MxO4mqsyYjlBalTVYxFP2sJBK5zlKOB
uzCBuDAfBgNVHSMEGDAWgBQiZQzWWp00ifODtJVSv1AbOScGrDBSBgNVHR8ESzBJ
MEegRaBDhkFodHRwczovL2NlcnRpZmljYXRlcy50cnVzdGVkc2VydmljZXMuaW50
ZWwuY29tL0ludGVsU0dYUm9vdENBLmRlcjAdBgNVHQ4EFgQUImUM1lqdNInzg7SV
Ur9QGzknBqwwDgYDVR0PAQH/BAQDAgEGMBIGA1UdEwEB/wQIMAYBAf8CAQEwCgYI
KoZIzj0EAwIDSQAwRgIhAOW/5QkR+S9CiSDcNoowLuPRLsWGf/Yi7GSX94BgwTwg
AiEA4J0lrHoMs+Xo5o/sX6O9QWxHRAvZUGOdRQ7cvqRXaqI=
-----END CERTIFICATE-----";

    const EXPECTED_INTEL_CDP: &str =
        "https://certificates.trustedservices.intel.com/IntelSGXRootCA.der";

    #[test]
    fn extract_cdp_from_intel_root_ca() {
        let url = extract_root_ca_cdp(INTEL_SGX_ROOT_CA_PEM).expect("should extract CDP");
        assert_eq!(url, EXPECTED_INTEL_CDP);
    }

    #[test]
    fn extract_cdp_from_multi_cert_chain() {
        // Simulate an issuer chain where the root CA is the last certificate.
        // Prepend a dummy "intermediate" PEM block.
        let chain = format!(
            "{}\n{}",
            INTEL_SGX_ROOT_CA_PEM, // intermediate (same cert, doesn't matter)
            INTEL_SGX_ROOT_CA_PEM, // root (last)
        );
        let url = extract_root_ca_cdp(&chain).expect("should extract CDP from last cert");
        assert_eq!(url, EXPECTED_INTEL_CDP);
    }

    #[test]
    fn extract_cdp_empty_chain() {
        let err = extract_root_ca_cdp("").unwrap_err();
        assert!(
            matches!(err, Error::Tdx(ref msg) if msg.contains("no certificate found")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn measurement_hex_concatenates_rtmr1_rtmr2() {
        let rtmr1 = [0xaa; 48];
        let rtmr2 = [0xbb; 48];
        let hex = measurement_hex(&rtmr1, &rtmr2);
        assert_eq!(hex.len(), 192);
        assert!(hex.starts_with(&"aa".repeat(48)));
        assert!(hex.ends_with(&"bb".repeat(48)));
    }
}
