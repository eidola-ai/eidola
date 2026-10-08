//! Intel TDX quote authentication and appraisal.
//!
//! Two steps, kept apart so neither can be applied without the other's
//! inputs being explicit:
//!
//! 1. [`authenticate`] proves the quote came from a genuine Intel TDX
//!    platform: the PCK certificate chain (carried in the quote) verifies to
//!    the pinned Intel SGX root with both Intel CRLs consulted, the TCB Info
//!    and QE Identity collateral carry valid Intel signatures and are inside
//!    their validity windows, the QE report is PCK-signed and binds the
//!    attestation key, the quote is signed by that key, and the platform, QE
//!    and TDX-module TCB levels are matched against Intel's collateral. The
//!    collateral is the document's own captured Intel PCS responses —
//!    untrusted transport authenticated by Intel's signatures — and nothing
//!    is fetched. The cryptographic core is `dcap-qvl`.
//! 2. [`appraise`] compares the authenticated quote with a pinned
//!    [`CompiledTdxPin`]: MRTD and MRCONFIGID (the launch identity), RTMR0–3
//!    all zero, MROWNER/MROWNERCONFIG zero, and the machine policy (MR_SEAM
//!    allowlist, exact TD_ATTRIBUTES and XFAM, TEE_TCB_SVN floor, QE vendor,
//!    collateral evaluation-data floor, optional FMSPC allowlist, and the
//!    PCK certificate's DynamicPlatform / CachedKeys / SMTEnabled flags, each
//!    stated explicitly by the pin), with every Intel TCB status required to
//!    be `UpToDate`.
//!
//! `REPORT_DATA` binding is not here: the handshake verifier compares the
//! quote's signed `REPORT_DATA` with the envelope's recomputation through the
//! same check the SEV-SNP path uses.

use dcap_qvl::configs::RustCryptoConfig;
use dcap_qvl::quote::{Quote, Report, TDReport10};
use dcap_qvl::verify::QuoteVerifier;
use dcap_qvl::{PckCertFlag, QuoteCollateralV3, QuotePolicy, TcbStatus};
use der::Decode as _;
use serde::Deserialize;
use serde_json::value::RawValue;

use crate::Error;
use crate::measurement::{CompiledTdxPin, PckFlag};

/// Intel SGX Provisioning Certification Root CA (DER), the root of every
/// PCK, TCB-signing and CRL chain Intel issues for TDX. Byte-identical to
/// the certificate Intel publishes at
/// `https://certificates.trustedservices.intel.com/Intel_SGX_Provisioning_Certification_RootCA.pem`.
pub(crate) const INTEL_SGX_ROOT_CA_DER: &[u8] = include_bytes!("intel_sgx_root_ca.der");

/// TDX quotes are version 4; the TD report body is TDREPORT 1.0.
const QUOTE_VERSION: u16 = 4;
/// Quote header TEE type for Intel TDX.
const TEE_TYPE_TDX: u32 = 0x81;
/// Quote header (48 bytes) plus a TD 1.0 report body (584 bytes): the bytes
/// the attestation key signs. A little-endian `u32` length then prefixes the
/// signature data.
const SIGNED_LEN: usize = 48 + 584;

/// One Intel PCS HTTP response as the enclave's collateral service captured
/// it. Nothing about it is trusted until Intel's signatures verify; the URL
/// only says which collateral the body claims to be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcsResponse {
    pub url: String,
    pub headers: Vec<(String, Vec<String>)>,
    pub body: Vec<u8>,
}

/// The four Intel PCS resources a TDX quote is verified against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PcsResource {
    PckCrl,
    TcbInfo,
    QeIdentity,
    RootCaCrl,
}

impl PcsResource {
    fn classify(url: &str) -> Result<Self, Error> {
        let parsed = reqwest::Url::parse(url)
            .map_err(|e| Error::Quote(format!("intel-pcs response URL {url:?}: {e}")))?;
        if parsed.scheme() != "https" || parsed.port().is_some() {
            return Err(Error::Quote(format!(
                "intel-pcs response URL {url:?} is not an Intel PCS resource"
            )));
        }
        match (parsed.host_str(), parsed.path()) {
            (Some("api.trustedservices.intel.com"), "/sgx/certification/v4/pckcrl") => {
                Ok(Self::PckCrl)
            }
            (Some("api.trustedservices.intel.com"), "/tdx/certification/v4/tcb") => {
                Ok(Self::TcbInfo)
            }
            (Some("api.trustedservices.intel.com"), "/tdx/certification/v4/qe/identity") => {
                Ok(Self::QeIdentity)
            }
            (Some("certificates.trustedservices.intel.com"), "/IntelSGXRootCA.der") => {
                Ok(Self::RootCaCrl)
            }
            _ => Err(Error::Quote(format!(
                "intel-pcs response URL {url:?} is not an Intel PCS resource this verifier uses"
            ))),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::PckCrl => "PCK CRL",
            Self::TcbInfo => "TCB Info",
            Self::QeIdentity => "QE Identity",
            Self::RootCaCrl => "root CA CRL",
        }
    }
}

/// A TDX quote whose signature chain, collateral and TCB levels verified.
/// Its fields are authenticated; none has been compared with a pin yet.
#[derive(Debug, Clone)]
pub(crate) struct AuthenticatedQuote {
    pub qe_vendor_id: [u8; 16],
    pub report: TDReport10,
    /// FMSPC from the Intel-signed PCK certificate.
    pub fmspc: [u8; 6],
    /// Platform flags from the Intel-signed PCK certificate.
    pub dynamic_platform: PckFlag,
    pub cached_keys: PckFlag,
    pub smt_enabled: PckFlag,
    /// Merged platform/QE/TDX-module status.
    pub tcb_status: TcbStatus,
    pub platform_tcb_status: TcbStatus,
    pub qe_tcb_status: TcbStatus,
    /// `tcbEvaluationDataNumber` of the verified TCB Info.
    pub tcb_info_evaluation_data_number: u32,
    /// `tcbEvaluationDataNumber` of the verified QE Identity.
    pub qe_identity_evaluation_data_number: u32,
}

/// Authenticate a raw TDX quote against Intel's root, using only the
/// captured PCS responses, at `now` (Unix seconds).
pub(crate) fn authenticate(
    quote: &[u8],
    responses: &[PcsResponse],
    intel_root_der: &[u8],
    now: u64,
) -> Result<AuthenticatedQuote, Error> {
    let parsed = parse_quote(quote)?;
    let collateral = assemble_collateral(responses, now)?;
    let tcb_info_json = collateral.tcb_info.clone();

    let claims = QuoteVerifier::<RustCryptoConfig>::new_with_config(intel_root_der.to_vec())
        .verify_with_policy(quote, collateral, now, &QuotePolicy::claims_only(now))
        .map_err(|e| Error::Quote(format!("{e:#}")))?;

    // The verifier re-parses the same bytes; the report it returns is the
    // one its signature check covered.
    let Report::TD10(report) = claims.report else {
        return Err(Error::Quote(
            "verified quote does not carry a TD 1.0 report body".to_string(),
        ));
    };
    if report != parsed {
        return Err(Error::Quote(
            "verified quote body differs from the parsed quote".to_string(),
        ));
    }
    // The claims carry the QE Identity's number and the lower of the two;
    // the TCB Info's own number is read from the bytes Intel's signature
    // just verified, so each document's floor can be checked and named.
    let tcb_info_evaluation_data_number = serde_json::from_str::<serde_json::Value>(&tcb_info_json)
        .ok()
        .and_then(|v| v.get("tcbEvaluationDataNumber")?.as_u64())
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| {
            Error::Quote("verified TCB Info carries no tcbEvaluationDataNumber".to_string())
        })?;
    let qe_identity_evaluation_data_number = claims.qe.tcb_eval_data_number;
    if tcb_info_evaluation_data_number.min(qe_identity_evaluation_data_number)
        != claims.tcb.eval_data_number
    {
        return Err(Error::Quote(
            "collateral evaluation data numbers are inconsistent".to_string(),
        ));
    }
    Ok(AuthenticatedQuote {
        qe_vendor_id: claims.header.qe_vendor_id,
        report,
        fmspc: claims.platform.pck.fmspc,
        dynamic_platform: pck_flag(claims.platform.pck.dynamic_platform),
        cached_keys: pck_flag(claims.platform.pck.cached_keys),
        smt_enabled: pck_flag(claims.platform.pck.smt_enabled),
        tcb_status: claims.tcb.status,
        platform_tcb_status: claims.platform.tcb_level.tcb_status,
        qe_tcb_status: claims.qe.tcb_level.tcb_status,
        tcb_info_evaluation_data_number,
        qe_identity_evaluation_data_number,
    })
}

fn pck_flag(flag: PckCertFlag) -> PckFlag {
    match flag {
        PckCertFlag::True => PckFlag::True,
        PckCertFlag::False => PckFlag::False,
        PckCertFlag::Undefined => PckFlag::Undefined,
    }
}

/// Structural checks the quote library parses past but never constrains:
/// version 4 TDX quotes only, the v4 header's reserved bytes (where earlier
/// versions carried QE/PCE SVNs) zero, and nothing but zero padding after the
/// signature data — any non-zero trailing byte would be unsigned content.
fn parse_quote(quote: &[u8]) -> Result<TDReport10, Error> {
    let parsed = Quote::parse(quote).map_err(|e| Error::Quote(format!("parsing quote: {e}")))?;
    if parsed.header.version != QUOTE_VERSION || parsed.header.tee_type != TEE_TYPE_TDX {
        return Err(Error::Quote(format!(
            "expected a version {QUOTE_VERSION} TDX quote, got version {} TEE type {:#x}",
            parsed.header.version, parsed.header.tee_type
        )));
    }
    if parsed.header.qe_svn != 0 || parsed.header.pce_svn != 0 {
        return Err(Error::Quote(
            "quote header carries non-zero reserved bytes".to_string(),
        ));
    }
    let Report::TD10(report) = parsed.report else {
        return Err(Error::Quote(
            "quote does not carry a TD 1.0 report body".to_string(),
        ));
    };
    let length_bytes: [u8; 4] = quote
        .get(SIGNED_LEN..SIGNED_LEN + 4)
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| Error::Quote("quote is truncated".to_string()))?;
    let end = (SIGNED_LEN + 4)
        .checked_add(u32::from_le_bytes(length_bytes) as usize)
        .filter(|end| *end <= quote.len())
        .ok_or_else(|| Error::Quote("quote signature data is truncated".to_string()))?;
    if quote[end..].iter().any(|b| *b != 0) {
        return Err(Error::Quote(
            "quote carries non-zero bytes after the signature data".to_string(),
        ));
    }
    Ok(report)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SignedTcbInfo<'a> {
    #[serde(rename = "tcbInfo", borrow)]
    tcb_info: &'a RawValue,
    signature: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SignedQeIdentity<'a> {
    #[serde(rename = "enclaveIdentity", borrow)]
    enclave_identity: &'a RawValue,
    signature: String,
}

/// Map the captured responses onto the collateral the quote verifier
/// consumes. Each of the four resources must appear exactly once; anything
/// else is refused. Intel signs the exact bytes of the `tcbInfo` and
/// `enclaveIdentity` members, so those are taken verbatim from the body,
/// never re-serialized.
fn assemble_collateral(responses: &[PcsResponse], now: u64) -> Result<QuoteCollateralV3, Error> {
    let mut found: [Option<&PcsResponse>; 4] = [None; 4];
    for response in responses {
        let resource = PcsResource::classify(&response.url)?;
        let slot = &mut found[resource as usize];
        if slot.is_some() {
            return Err(Error::Quote(format!(
                "intel-pcs collateral carries more than one {} response",
                resource.label()
            )));
        }
        *slot = Some(response);
    }
    let take = |resource: PcsResource| {
        found[resource as usize].ok_or_else(|| {
            Error::Quote(format!(
                "intel-pcs collateral carries no {} response",
                resource.label()
            ))
        })
    };
    let pck_crl = take(PcsResource::PckCrl)?;
    let tcb = take(PcsResource::TcbInfo)?;
    let qe = take(PcsResource::QeIdentity)?;
    let root_crl = take(PcsResource::RootCaCrl)?;

    // The quote library enforces each CRL's nextUpdate but not its
    // thisUpdate; a capture dated in the future must not pass.
    check_crl_window(&pck_crl.body, "PCK CRL", now)?;
    check_crl_window(&root_crl.body, "root CA CRL", now)?;

    crate::bundle::reject_duplicate_members(&tcb.body, "intel-pcs TCB Info")?;
    let signed_tcb: SignedTcbInfo<'_> = serde_json::from_slice(&tcb.body)
        .map_err(|e| Error::Quote(format!("parsing TCB Info response: {e}")))?;
    crate::bundle::reject_duplicate_members(&qe.body, "intel-pcs QE Identity")?;
    let signed_qe: SignedQeIdentity<'_> = serde_json::from_slice(&qe.body)
        .map_err(|e| Error::Quote(format!("parsing QE Identity response: {e}")))?;

    Ok(QuoteCollateralV3 {
        pck_crl_issuer_chain: header(pck_crl, &["SGX-PCK-CRL-Issuer-Chain"])?,
        root_ca_crl: root_crl.body.clone(),
        pck_crl: pck_crl.body.clone(),
        tcb_info_issuer_chain: header(
            tcb,
            &["TCB-Info-Issuer-Chain", "SGX-TCB-Info-Issuer-Chain"],
        )?,
        tcb_info: signed_tcb.tcb_info.get().to_string(),
        tcb_info_signature: decode_signature(&signed_tcb.signature, "TCB Info")?,
        qe_identity_issuer_chain: header(qe, &["SGX-Enclave-Identity-Issuer-Chain"])?,
        qe_identity: signed_qe.enclave_identity.get().to_string(),
        qe_identity_signature: decode_signature(&signed_qe.signature, "QE Identity")?,
        // The PCK chain is the one embedded in the quote (certification
        // data type 5); the collateral never substitutes another.
        pck_certificate_chain: None,
    })
}

/// The single value of the one header among `names` (matched
/// case-insensitively) the response carries, percent-decoded as Intel
/// delivers issuer chains.
fn header(response: &PcsResponse, names: &[&str]) -> Result<String, Error> {
    let mut matches = response
        .headers
        .iter()
        .filter(|(name, _)| names.iter().any(|n| n.eq_ignore_ascii_case(name)));
    let (name, values) = matches.next().ok_or_else(|| {
        Error::Quote(format!(
            "intel-pcs response {:?} lacks the {} header",
            response.url, names[0]
        ))
    })?;
    if matches.next().is_some() {
        return Err(Error::Quote(format!(
            "intel-pcs response {:?} repeats the {} header",
            response.url, names[0]
        )));
    }
    let [value] = values.as_slice() else {
        return Err(Error::Quote(format!(
            "intel-pcs response header {name:?} must carry exactly one value"
        )));
    };
    percent_decode(value)
        .ok_or_else(|| Error::Quote(format!("intel-pcs response header {name:?} is malformed")))
}

fn percent_decode(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hi = (*bytes.get(i + 1)? as char).to_digit(16)?;
            let lo = (*bytes.get(i + 2)? as char).to_digit(16)?;
            out.push((hi * 16 + lo) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

fn decode_signature(value: &str, label: &str) -> Result<Vec<u8>, Error> {
    let signature = hex::decode(value)
        .map_err(|e| Error::Quote(format!("{label} signature is not hex: {e}")))?;
    if signature.len() != 64 {
        return Err(Error::Quote(format!(
            "{label} signature must be a 64-byte ECDSA P-256 r||s value"
        )));
    }
    Ok(signature)
}

fn check_crl_window(der: &[u8], label: &str, now: u64) -> Result<(), Error> {
    let crl = x509_cert::crl::CertificateList::from_der(der)
        .map_err(|e| Error::Quote(format!("parsing {label}: {e}")))?;
    let this_update = crate::sevsnp_crl::time_secs(crl.tbs_cert_list.this_update);
    let next_update = crl
        .tbs_cert_list
        .next_update
        .map(crate::sevsnp_crl::time_secs)
        .ok_or_else(|| Error::Quote(format!("{label} has no nextUpdate")))?;
    if now < this_update || now >= next_update {
        return Err(Error::Quote(format!(
            "{label} is outside its validity window (thisUpdate {this_update}, \
             nextUpdate {next_update}, now {now})"
        )));
    }
    Ok(())
}

/// Compare an authenticated quote with a pin. Every check here is on
/// Intel-signed fields; the order only decides which violation is reported.
pub(crate) fn appraise(quote: &AuthenticatedQuote, pin: &CompiledTdxPin) -> Result<(), Error> {
    let report = &quote.report;
    let violation = |message: String| Err(Error::TdxPolicy(message));

    if report.mr_td != pin.mrtd {
        return violation(format!("MRTD {} is not pinned", hex::encode(report.mr_td)));
    }
    if report.mr_config_id != pin.mrconfigid {
        return violation(format!(
            "MRCONFIGID {} does not match the pinned config",
            hex::encode(report.mr_config_id)
        ));
    }
    for (index, rtmr) in [report.rt_mr0, report.rt_mr1, report.rt_mr2, report.rt_mr3]
        .iter()
        .enumerate()
    {
        if *rtmr != [0u8; 48] {
            return violation(format!(
                "RTMR{index} is {}; the launch extends no runtime register",
                hex::encode(rtmr)
            ));
        }
    }
    if report.mr_owner != [0u8; 48] || report.mr_owner_config != [0u8; 48] {
        return violation("MROWNER/MROWNERCONFIG must be zero".to_string());
    }
    if report.td_attributes != pin.td_attributes {
        return violation(format!(
            "TD_ATTRIBUTES {} is not the pinned {}",
            hex::encode(report.td_attributes),
            hex::encode(pin.td_attributes)
        ));
    }
    if report.xfam != pin.xfam {
        return violation(format!(
            "XFAM {} is not the pinned {}",
            hex::encode(report.xfam),
            hex::encode(pin.xfam)
        ));
    }
    if !pin.mr_seam.contains(&report.mr_seam) {
        return violation(format!(
            "MR_SEAM {} is not an allowed TDX module",
            hex::encode(report.mr_seam)
        ));
    }
    if report
        .tee_tcb_svn
        .iter()
        .zip(&pin.minimum_tee_tcb_svn)
        .any(|(observed, minimum)| observed < minimum)
    {
        return violation(format!(
            "TEE_TCB_SVN {} is below the minimum {}",
            hex::encode(report.tee_tcb_svn),
            hex::encode(pin.minimum_tee_tcb_svn)
        ));
    }
    if quote.qe_vendor_id != pin.qe_vendor_id {
        return violation(format!(
            "QE vendor id {} is not the pinned {}",
            hex::encode(quote.qe_vendor_id),
            hex::encode(pin.qe_vendor_id)
        ));
    }
    for (label, status) in [
        ("platform", quote.platform_tcb_status),
        ("QE", quote.qe_tcb_status),
        ("merged", quote.tcb_status),
    ] {
        if status != TcbStatus::UpToDate {
            return violation(format!("{label} TCB status is {status:?}, not UpToDate"));
        }
    }
    // The floor bounds how old either signed collateral document may be, so
    // it applies to each, and the refusal names the stale one.
    for (document, number) in [
        ("TCB Info", quote.tcb_info_evaluation_data_number),
        ("QE Identity", quote.qe_identity_evaluation_data_number),
    ] {
        if number < pin.minimum_tcb_evaluation_data_number {
            return violation(format!(
                "{document} tcbEvaluationDataNumber {number} is below the minimum {}",
                pin.minimum_tcb_evaluation_data_number
            ));
        }
    }
    if let Some(allowed) = &pin.fmspc
        && !allowed.contains(&quote.fmspc)
    {
        return violation(format!(
            "FMSPC {} is not an allowed platform",
            hex::encode(quote.fmspc)
        ));
    }
    for (label, observed, expected) in [
        (
            "DynamicPlatform",
            quote.dynamic_platform,
            pin.dynamic_platform,
        ),
        ("CachedKeys", quote.cached_keys, pin.cached_keys),
        ("SMTEnabled", quote.smt_enabled, pin.smt_enabled),
    ] {
        if observed != expected {
            return violation(format!(
                "PCK {label} flag is {observed:?}, pinned {expected:?}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    /// SHA-256 of the pinned Intel SGX root certificate.
    const INTEL_SGX_ROOT_CA_SHA256: &str =
        "44a0196b2b99f889b8e149e95b807a350e7424964399e885a7cbb8ccfab674d3";

    #[test]
    fn pinned_intel_root_is_the_published_certificate() {
        assert_eq!(
            hex::encode(Sha256::digest(INTEL_SGX_ROOT_CA_DER)),
            INTEL_SGX_ROOT_CA_SHA256
        );
        let cert = x509_cert::Certificate::from_der(INTEL_SGX_ROOT_CA_DER).unwrap();
        assert!(
            cert.tbs_certificate
                .subject
                .to_string()
                .contains("Intel SGX Root CA")
        );
    }

    #[test]
    fn classifies_only_the_four_pcs_resources() {
        let ok = [
            (
                "https://api.trustedservices.intel.com/sgx/certification/v4/pckcrl?ca=platform&encoding=der",
                PcsResource::PckCrl,
            ),
            (
                "https://api.trustedservices.intel.com/tdx/certification/v4/tcb?fmspc=b0c06f000000&tcbEvaluationDataNumber=19",
                PcsResource::TcbInfo,
            ),
            (
                "https://api.trustedservices.intel.com/tdx/certification/v4/qe/identity",
                PcsResource::QeIdentity,
            ),
            (
                "https://certificates.trustedservices.intel.com/IntelSGXRootCA.der",
                PcsResource::RootCaCrl,
            ),
        ];
        for (url, resource) in ok {
            assert_eq!(PcsResource::classify(url).unwrap(), resource, "{url}");
        }
        for url in [
            "http://api.trustedservices.intel.com/tdx/certification/v4/tcb",
            "https://api.trustedservices.intel.com:8443/tdx/certification/v4/tcb",
            "https://pccs.example.com/tdx/certification/v4/tcb",
            "https://api.trustedservices.intel.com/sgx/certification/v4/tcb",
            "https://api.trustedservices.intel.com/tdx/certification/v4/pckcert",
            "not a url",
        ] {
            assert!(PcsResource::classify(url).is_err(), "{url}");
        }
    }

    #[test]
    fn percent_decoding_is_strict() {
        assert_eq!(percent_decode("a%2Db%0A").as_deref(), Some("a-b\n"));
        assert_eq!(percent_decode("plain+text").as_deref(), Some("plain+text"));
        assert_eq!(percent_decode("bad%2"), None);
        assert_eq!(percent_decode("bad%zz"), None);
        assert_eq!(percent_decode("%FF"), None);
    }

    /// An authenticated quote that satisfies [`pin`].
    pub(crate) fn authenticated() -> AuthenticatedQuote {
        AuthenticatedQuote {
            qe_vendor_id: hex::decode("939a7233f79c4ca9940a0db3957f0607")
                .unwrap()
                .try_into()
                .unwrap(),
            report: TDReport10 {
                tee_tcb_svn: [3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                mr_seam: [0x22; 48],
                mr_signer_seam: [0; 48],
                seam_attributes: [0; 8],
                td_attributes: [0, 0, 0, 0x10, 0, 0, 0, 0],
                xfam: [0xe7, 0x02, 0x06, 0, 0, 0, 0, 0],
                mr_td: [0x11; 48],
                mr_config_id: config_id(),
                mr_owner: [0; 48],
                mr_owner_config: [0; 48],
                rt_mr0: [0; 48],
                rt_mr1: [0; 48],
                rt_mr2: [0; 48],
                rt_mr3: [0; 48],
                report_data: [0; 64],
            },
            fmspc: [0xb0, 0xc0, 0x6f, 0, 0, 0],
            dynamic_platform: PckFlag::True,
            cached_keys: PckFlag::True,
            smt_enabled: PckFlag::True,
            tcb_status: TcbStatus::UpToDate,
            platform_tcb_status: TcbStatus::UpToDate,
            qe_tcb_status: TcbStatus::UpToDate,
            tcb_info_evaluation_data_number: 19,
            qe_identity_evaluation_data_number: 19,
        }
    }

    fn config_id() -> [u8; 48] {
        hex::decode(crate::TdxPin::mrconfigid_for_config(b"config"))
            .unwrap()
            .try_into()
            .unwrap()
    }

    pub(crate) fn pin() -> CompiledTdxPin {
        let compiled = crate::measurement::compile_pins(&[crate::AllowedMeasurement::tdx(
            crate::measurement::tests::tdx_pin(),
        )])
        .unwrap();
        let crate::measurement::CompiledPlatform::Tdx(pin) = &compiled[0].platform else {
            unreachable!()
        };
        CompiledTdxPin {
            minimum_tcb_evaluation_data_number: 19,
            ..(**pin).clone()
        }
    }

    #[test]
    fn appraisal_accepts_a_quote_matching_every_pinned_field() {
        appraise(&authenticated(), &pin()).expect("matching quote");
    }

    /// Every policy field, one mutation each: a quote that differs from the
    /// pin in exactly that field is refused, with that field named.
    #[test]
    fn appraisal_refuses_each_field_that_differs_from_the_pin() {
        type Mutation = Box<dyn Fn(&mut AuthenticatedQuote, &mut CompiledTdxPin)>;
        let cases: Vec<(&str, Mutation)> = vec![
            ("MRTD", Box::new(|q, _| q.report.mr_td[0] ^= 1)),
            ("MRCONFIGID", Box::new(|q, _| q.report.mr_config_id[47] = 1)),
            ("RTMR0", Box::new(|q, _| q.report.rt_mr0[0] = 1)),
            ("RTMR1", Box::new(|q, _| q.report.rt_mr1[0] = 1)),
            ("RTMR2", Box::new(|q, _| q.report.rt_mr2[47] = 1)),
            ("RTMR3", Box::new(|q, _| q.report.rt_mr3[10] = 1)),
            ("MROWNER", Box::new(|q, _| q.report.mr_owner[0] = 1)),
            ("MROWNER", Box::new(|q, _| q.report.mr_owner_config[0] = 1)),
            // DEBUG (bit 0) set on an otherwise identical launch: MRTD does
            // not change, only the attribute pin catches it.
            (
                "TD_ATTRIBUTES",
                Box::new(|q, _| q.report.td_attributes[0] |= 1),
            ),
            // SEPT_VE_DISABLE (bit 28) cleared.
            (
                "TD_ATTRIBUTES",
                Box::new(|q, _| q.report.td_attributes[3] = 0),
            ),
            ("XFAM", Box::new(|q, _| q.report.xfam[0] = 0x03)),
            ("MR_SEAM", Box::new(|q, _| q.report.mr_seam[0] = 0x33)),
            ("TEE_TCB_SVN", Box::new(|q, _| q.report.tee_tcb_svn[0] = 2)),
            ("TEE_TCB_SVN", Box::new(|q, _| q.report.tee_tcb_svn[2] = 1)),
            ("QE vendor", Box::new(|q, _| q.qe_vendor_id[0] = 0)),
            (
                "platform TCB status",
                Box::new(|q, _| q.platform_tcb_status = TcbStatus::OutOfDate),
            ),
            (
                "QE TCB status",
                Box::new(|q, _| q.qe_tcb_status = TcbStatus::SWHardeningNeeded),
            ),
            (
                "merged TCB status",
                Box::new(|q, _| q.tcb_status = TcbStatus::ConfigurationNeeded),
            ),
            (
                "TCB Info tcbEvaluationDataNumber 18",
                Box::new(|q, _| q.tcb_info_evaluation_data_number = 18),
            ),
            // A stale QE Identity under a fresh TCB Info: the floor binds
            // each document, not only the TCB Info.
            (
                "QE Identity tcbEvaluationDataNumber 18",
                Box::new(|q, _| q.qe_identity_evaluation_data_number = 18),
            ),
            (
                "FMSPC",
                Box::new(|_, p| p.fmspc = Some(vec![[0x90, 0xc0, 0x6f, 0, 0, 0]])),
            ),
            // Each platform flag, both directions and the absent case: the
            // pin states the value, so a host that differs either way is
            // refused.
            (
                "DynamicPlatform",
                Box::new(|q, _| q.dynamic_platform = PckFlag::False),
            ),
            (
                "DynamicPlatform",
                Box::new(|_, p| p.dynamic_platform = PckFlag::False),
            ),
            (
                "CachedKeys",
                Box::new(|q, _| q.cached_keys = PckFlag::Undefined),
            ),
            (
                "CachedKeys",
                Box::new(|_, p| p.cached_keys = PckFlag::False),
            ),
            (
                "SMTEnabled",
                Box::new(|q, _| q.smt_enabled = PckFlag::False),
            ),
            (
                "SMTEnabled",
                Box::new(|_, p| p.smt_enabled = PckFlag::False),
            ),
        ];
        for (needle, mutate) in cases {
            let mut quote = authenticated();
            let mut pin = pin();
            mutate(&mut quote, &mut pin);
            let err = appraise(&quote, &pin).expect_err(needle).to_string();
            assert!(err.contains(needle), "{needle}: {err}");
        }
    }

    // A real TDX quote and the Intel collateral it verifies against, from
    // dcap-qvl's test samples (see tests/fixtures/dcap-qvl/README.md). The
    // collateral is re-shaped into the captured-PCS-response form a Tinfoil
    // document carries; Intel's signed bytes are kept verbatim.
    pub(crate) const FIXTURE_QUOTE: &[u8] = include_bytes!("../tests/fixtures/dcap-qvl/tdx_quote");
    /// Inside every validity window of the fixture's collateral.
    pub(crate) const FIXTURE_NOW: u64 = 1_751_500_000;
    pub(crate) const FIXTURE_MRTD: &str = "91eb2b44d141d4ece09f0c75c2c53d247a3c68edd7fafe8a3520c942a604a407de03ae6dc5f87f27428b2538873118b7";

    fn pct(value: &str) -> String {
        value
            .bytes()
            .map(|b| {
                if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
                    (b as char).to_string()
                } else {
                    format!("%{b:02X}")
                }
            })
            .collect()
    }

    pub(crate) fn fixture_responses() -> Vec<PcsResponse> {
        let c: QuoteCollateralV3 = serde_json::from_slice(include_bytes!(
            "../tests/fixtures/dcap-qvl/tdx_quote_collateral.json"
        ))
        .unwrap();
        let chain = |name: &str, pem: &str| vec![(name.to_string(), vec![pct(pem)])];
        vec![
            PcsResponse {
                url: "https://api.trustedservices.intel.com/sgx/certification/v4/pckcrl?ca=platform&encoding=der".to_string(),
                headers: chain("SGX-PCK-CRL-Issuer-Chain", &c.pck_crl_issuer_chain),
                body: c.pck_crl.clone(),
            },
            PcsResponse {
                url: "https://api.trustedservices.intel.com/tdx/certification/v4/tcb?fmspc=b0c06f000000".to_string(),
                headers: chain("Tcb-Info-Issuer-Chain", &c.tcb_info_issuer_chain),
                body: format!(
                    r#"{{"tcbInfo":{},"signature":"{}"}}"#,
                    c.tcb_info,
                    hex::encode(&c.tcb_info_signature)
                )
                .into_bytes(),
            },
            PcsResponse {
                url: "https://api.trustedservices.intel.com/tdx/certification/v4/qe/identity".to_string(),
                headers: chain("SGX-Enclave-Identity-Issuer-Chain", &c.qe_identity_issuer_chain),
                body: format!(
                    r#"{{"enclaveIdentity":{},"signature":"{}"}}"#,
                    c.qe_identity,
                    hex::encode(&c.qe_identity_signature)
                )
                .into_bytes(),
            },
            PcsResponse {
                url: "https://certificates.trustedservices.intel.com/IntelSGXRootCA.der".to_string(),
                headers: vec![],
                body: c.root_ca_crl.clone(),
            },
        ]
    }

    #[test]
    fn real_quote_authenticates_offline_against_captured_collateral() {
        let quote = authenticate(
            FIXTURE_QUOTE,
            &fixture_responses(),
            INTEL_SGX_ROOT_CA_DER,
            FIXTURE_NOW,
        )
        .expect("fixture quote authenticates");
        assert_eq!(hex::encode(quote.report.mr_td), FIXTURE_MRTD);
        assert_eq!(quote.report.mr_config_id, [0; 48]);
        assert_eq!(hex::encode(quote.fmspc), "b0c06f000000");
        assert_eq!(quote.tcb_status, TcbStatus::UpToDate);
        assert_eq!(quote.tcb_info_evaluation_data_number, 17);
        assert_eq!(quote.qe_identity_evaluation_data_number, 17);
        // A multi-package (Platform CA) host with SMT on, as cloud TDX
        // hosts commonly are: all three flags asserted.
        assert_eq!(quote.dynamic_platform, PckFlag::True);
        assert_eq!(quote.cached_keys, PckFlag::True);
        assert_eq!(quote.smt_enabled, PckFlag::True);
        assert_eq!(
            hex::encode(quote.qe_vendor_id),
            "939a7233f79c4ca9940a0db3957f0607"
        );
    }

    /// The real quote fails the IGVM launch model: it was produced under a
    /// measured-boot launch that extends RTMR0. Authentication passes and
    /// appraisal refuses on that register even with every other field
    /// pinned to the quote's own values.
    #[test]
    fn real_quote_with_extended_rtmr0_is_refused_by_appraisal() {
        let quote = authenticate(
            FIXTURE_QUOTE,
            &fixture_responses(),
            INTEL_SGX_ROOT_CA_DER,
            FIXTURE_NOW,
        )
        .unwrap();
        let pin = CompiledTdxPin {
            mrtd: quote.report.mr_td,
            mrconfigid: quote.report.mr_config_id,
            mr_seam: vec![quote.report.mr_seam],
            td_attributes: quote.report.td_attributes,
            xfam: quote.report.xfam,
            minimum_tee_tcb_svn: quote.report.tee_tcb_svn,
            minimum_tcb_evaluation_data_number: 17,
            qe_vendor_id: quote.qe_vendor_id,
            fmspc: Some(vec![quote.fmspc]),
            dynamic_platform: quote.dynamic_platform,
            cached_keys: quote.cached_keys,
            smt_enabled: quote.smt_enabled,
        };
        let err = appraise(&quote, &pin).unwrap_err().to_string();
        assert!(err.contains("RTMR0"), "{err}");
        // With RTMR0 cleared on the (authenticated) copy, the same pin
        // accepts: RTMR0 is the only field keeping this quote out.
        let mut cleared = quote.clone();
        cleared.report.rt_mr0 = [0; 48];
        cleared.report.rt_mr1 = [0; 48];
        cleared.report.rt_mr2 = [0; 48];
        cleared.report.rt_mr3 = [0; 48];
        appraise(&cleared, &pin).expect("all other fields match their own pin");
    }

    /// Every way the collateral or quote can be wrong is refused before any
    /// field is trusted.
    #[test]
    fn tampered_missing_or_stale_collateral_is_refused() {
        type Case = (
            &'static str,
            Box<dyn Fn(&mut Vec<PcsResponse>, &mut Vec<u8>, &mut u64)>,
        );
        fn body<'a>(responses: &'a mut [PcsResponse], url_part: &str) -> &'a mut Vec<u8> {
            &mut responses
                .iter_mut()
                .find(|r| r.url.contains(url_part))
                .unwrap()
                .body
        }
        fn replace(bytes: &mut Vec<u8>, from: &str, to: &str) {
            let text = String::from_utf8(bytes.clone()).unwrap();
            assert!(text.contains(from), "{from}");
            *bytes = text.replacen(from, to, 1).into_bytes();
        }
        let cases: Vec<Case> = vec![
            (
                "Signature is invalid for tcb_info",
                Box::new(|r, _, _| {
                    replace(
                        body(r, "/tcb"),
                        "\"tcbEvaluationDataNumber\":17",
                        "\"tcbEvaluationDataNumber\":18",
                    )
                }),
            ),
            (
                "Signature is invalid for qe_identity",
                Box::new(|r, _, _| {
                    replace(
                        body(r, "qe/identity"),
                        "\"tcbEvaluationDataNumber\":17",
                        "\"tcbEvaluationDataNumber\":18",
                    )
                }),
            ),
            (
                "InvalidCrlSignatureForPublicKey",
                Box::new(|r, _, _| {
                    let b = body(r, "pckcrl");
                    let last = b.len() - 1;
                    b[last] ^= 1;
                }),
            ),
            (
                "root CA CRL",
                Box::new(|r, _, _| {
                    let b = body(r, "IntelSGXRootCA");
                    let last = b.len() - 1;
                    b[last] ^= 1;
                }),
            ),
            (
                "no QE Identity response",
                Box::new(|r, _, _| r.retain(|x| !x.url.contains("qe/identity"))),
            ),
            (
                "more than one TCB Info",
                Box::new(|r, _, _| {
                    let dup = r.iter().find(|x| x.url.contains("/tcb")).unwrap().clone();
                    r.push(dup);
                }),
            ),
            (
                "not an Intel PCS resource",
                Box::new(|r, _, _| {
                    r[0].url = "https://pccs.example.com/sgx/certification/v4/pckcrl".to_string()
                }),
            ),
            (
                "lacks the TCB-Info-Issuer-Chain header",
                Box::new(|r, _, _| {
                    r.iter_mut()
                        .find(|x| x.url.contains("/tcb"))
                        .unwrap()
                        .headers
                        .clear()
                }),
            ),
            (
                "unknown field",
                Box::new(|r, _, _| {
                    replace(body(r, "/tcb"), "{\"tcbInfo\"", "{\"extra\":1,\"tcbInfo\"")
                }),
            ),
            (
                "duplicate object member",
                Box::new(|r, _, _| {
                    replace(
                        body(r, "/tcb"),
                        "\"signature\"",
                        "\"signature\":\"00\",\"signature\"",
                    )
                }),
            ),
            // Collateral expired: past every CRL's nextUpdate.
            (
                "outside its validity window",
                Box::new(|_, _, now| *now = 1_760_000_000),
            ),
            // A capture dated after the verification time.
            (
                "outside its validity window",
                Box::new(|_, _, now| *now = 1_740_000_000),
            ),
            // Inside both CRL windows but before the TCB Info issueDate
            // (2025-06-19T10:16:03Z): Intel's own dating applies too.
            (
                "TCBInfo issue date is in the future",
                Box::new(|_, _, now| *now = 1_750_327_500),
            ),
            // The quote body is covered by the attestation key's signature.
            (
                "ISV enclave report signature is invalid",
                Box::new(|_, q, _| q[48 + 136] ^= 1),
            ),
            ("reserved bytes", Box::new(|_, q, _| q[8] = 1)),
            ("parsing quote", Box::new(|_, q, _| q[0] = 5)),
            // An SGX quote (TEE type 0) does not parse as TDX evidence.
            ("parsing quote", Box::new(|_, q, _| q[4] = 0)),
            (
                "non-zero bytes after the signature data",
                Box::new(|_, q, _| q.extend_from_slice(&[0, 0, 1])),
            ),
            ("parsing quote", Box::new(|_, q, _| q.truncate(700))),
        ];
        for (needle, mutate) in cases {
            let mut responses = fixture_responses();
            let mut quote = FIXTURE_QUOTE.to_vec();
            let mut now = FIXTURE_NOW;
            mutate(&mut responses, &mut quote, &mut now);
            let err = authenticate(&quote, &responses, INTEL_SGX_ROOT_CA_DER, now)
                .expect_err(needle)
                .to_string();
            assert!(err.contains(needle), "{needle}: {err}");
        }
    }

    #[test]
    fn zero_padding_after_the_signature_data_is_accepted() {
        let mut quote = FIXTURE_QUOTE.to_vec();
        quote.extend_from_slice(&[0; 64]);
        authenticate(
            &quote,
            &fixture_responses(),
            INTEL_SGX_ROOT_CA_DER,
            FIXTURE_NOW,
        )
        .expect("zero padding is not content");
    }

    #[test]
    fn a_chain_to_any_other_root_is_refused() {
        let (ark, _) = crate::sevsnp::resolve_chain_certs_der(None, None).unwrap();
        let err = authenticate(FIXTURE_QUOTE, &fixture_responses(), &ark, FIXTURE_NOW)
            .unwrap_err()
            .to_string();
        assert!(err.contains("TDX quote verification failed"), "{err}");
    }

    #[test]
    fn appraisal_accepts_higher_svns_and_a_listed_fmspc() {
        let mut quote = authenticated();
        quote.report.tee_tcb_svn = [4, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        quote.tcb_info_evaluation_data_number = 20;
        quote.qe_identity_evaluation_data_number = 21;
        let mut pin = pin();
        pin.fmspc = Some(vec![[0x90, 0xc0, 0x6f, 0, 0, 0], quote.fmspc]);
        pin.mr_seam.push([0x44; 48]);
        appraise(&quote, &pin).expect("within policy");
    }
}
