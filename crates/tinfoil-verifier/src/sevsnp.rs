//! SEV-SNP attestation verification using the `sev` crate.
//!
//! Delegates certificate chain verification and report signature checking to
//! the [`sev`](https://crates.io/crates/sev) crate (virtee/sev). This crate
//! adds TCB policy enforcement (configurable per-component floors plus a
//! rollback check against `committed_tcb`) and TLS fingerprint
//! cross-checking on top.
//!
//! ## TCB policy: AMD vs Intel
//!
//! AMD does not publish a pull-based TCB recommendation feed equivalent to
//! Intel's `tcb_info` JSON, so the relying party's "minimum TCB SVNs" are
//! operator-set rather than fetched from a remote endpoint. The defaults
//! in [`SevSnpTcbPolicy::amd_recommended`] are picked to match the floor
//! enforced by Google's `go-sev-guest`; operators can tighten them via
//! [`crate::AttestingClientConfig::snp_min_tcb`].
//!
//! ## Rollback protection
//!
//! Each SEV-SNP attestation report carries multiple TCB version fields.
//! The two we care about for policy purposes are:
//!
//! - `reported_tcb`: the TCB version associated with the VCEK that signed
//!   the report. The hypervisor can change this via the firmware's
//!   `SET_TCB_VERSION` command, but only downward (it can't lie upward,
//!   because there is no VCEK that would sign for a higher TCB — a fact
//!   [`verify_report`] enforces by comparing `reported_tcb` and `CHIP_ID`
//!   with the SPL and hwID extensions of the VCEK that signed the report;
//!   the VCEK arrives as untrusted collateral, so the signature alone binds
//!   the report to *some* genuine VCEK, not to this TCB).
//! - `committed_tcb`: a one-way commit by the firmware. Once the firmware
//!   commits to a TCB level, it will not honor any `SET_TCB_VERSION`
//!   request that would drop reported_tcb below it.
//!
//! A malicious hypervisor that wants to make an enclave appear to be
//! running on an older (and known-vulnerable) TCB could call
//! `SET_TCB_VERSION` to drop `reported_tcb` to a value the firmware has
//! *not* committed never to honor. We catch this by requiring
//! `reported_tcb >= committed_tcb` componentwise; the only legitimate way
//! the inequality could fail is a firmware bug or a hypervisor that's
//! actively lying about the TCB level. We classify this as a separate,
//! more severe failure mode than "below operator floor."

use std::io::Cursor;
use std::sync::Arc;

use der::{Decode, Encode};
use sev::certs::snp::{Certificate, Chain, Verifiable, builtin::genoa, ca};
use sev::firmware::guest::AttestationReport;
use sev::firmware::host::TcbVersion;
use sev::parser::Decoder;
use sha2::{Digest, Sha256};

use crate::Error;

/// Exact wire size of a SEV-SNP attestation report (AMD SEV-SNP ABI,
/// `ATTESTATION_REPORT` structure). The `sev` crate reads exactly this many
/// bytes and would silently ignore trailing data, so the length is pinned
/// here to keep the accepted encoding unique.
const REPORT_LEN: usize = 1184;

/// Parse a raw attestation report without verifying its signature.
pub fn parse_report(report_bytes: &[u8]) -> Result<AttestationReport, Error> {
    if report_bytes.len() != REPORT_LEN {
        return Err(Error::Report(format!(
            "attestation report must be exactly {REPORT_LEN} bytes, got {}",
            report_bytes.len()
        )));
    }
    AttestationReport::decode(&mut Cursor::new(report_bytes), ())
        .map_err(|e| Error::Report(format!("failed to parse attestation report: {e}")))
}

/// Structural checks on report fields that are *not* covered by the launch
/// measurement and must therefore be policed by the relying party.
///
/// The guest policy is the sharp one: it is chosen by the hypervisor at
/// launch and enforced by the PSP, but it is **not** folded into the launch
/// digest — a malicious host can relaunch the exact pinned image with
/// `POLICY.DEBUG=1` and obtain an identical measurement while gaining
/// `SNP_DBG_DECRYPT` access to guest memory. `MIGRATE_MA` similarly hands
/// guest state to a migration agent outside the measured image. Both must
/// be off.
///
/// The signer/ID-block fields are hygiene: the report must be VCEK-signed
/// (`SIGNING_KEY=0`), with the chip identity unmasked and no ID-block or
/// author key in play — matching what Tinfoil's own v3 verifier enforces
/// and what its production fleet presents.
///
/// Report version 3 (firmware 1.55+) is required so the CPUID
/// family/model/stepping fields are present and the product generation is
/// bound by the report itself rather than inferred.
pub fn check_report_hygiene(report: &AttestationReport) -> Result<(), Error> {
    if report.version < 3 {
        return Err(Error::Report(format!(
            "attestation report version {} is below the required minimum of 3",
            report.version
        )));
    }
    if report.policy.debug_allowed() {
        return Err(Error::Report(
            "guest policy allows DEBUG: the hypervisor could decrypt guest memory \
             via SNP_DBG_DECRYPT despite a matching launch measurement"
                .to_string(),
        ));
    }
    if report.policy.migrate_ma_allowed() {
        return Err(Error::Report(
            "guest policy allows migration-agent association".to_string(),
        ));
    }
    if report.key_info.mask_chip_key() {
        return Err(Error::Report(
            "report has MASK_CHIP_KEY set; chip identity is masked".to_string(),
        ));
    }
    if report.key_info.signing_key() != 0 {
        return Err(Error::Report(format!(
            "report SIGNING_KEY is {}; only VCEK (0) is accepted",
            report.key_info.signing_key()
        )));
    }
    if report.key_info.author_key_en()
        || report.id_key_digest != [0u8; 48]
        || report.author_key_digest != [0u8; 48]
    {
        return Err(Error::Report(
            "report indicates an ID-block launch (AUTHOR_KEY_EN or a non-zero \
             ID_KEY_DIGEST/AUTHOR_KEY_DIGEST); only plain launches are accepted"
                .to_string(),
        ));
    }
    Ok(())
}

/// Verify a VCEK certificate chain and an already-parsed report's signature.
///
/// 1. Builds the chain: custom or built-in ARK → ASK → VCEK
/// 2. Verifies the chain (ARK self-signed, ARK signs ASK, ASK signs VCEK)
/// 3. Verifies the report's ECDSA-P384 signature against the VCEK
pub fn verify_report(
    vcek_der: &[u8],
    report: &AttestationReport,
    ark_der: Option<&[u8]>,
    ask_der: Option<&[u8]>,
) -> Result<(), Error> {
    let ark = match ark_der {
        Some(der) => Certificate::from_der(der)
            .map_err(|e| Error::CertChain(format!("failed to parse custom ARK: {e}")))?,
        None => {
            genoa::ark().map_err(|e| Error::CertChain(format!("failed to load Genoa ARK: {e}")))?
        }
    };

    let ask = match ask_der {
        Some(der) => Certificate::from_der(der)
            .map_err(|e| Error::CertChain(format!("failed to parse custom ASK: {e}")))?,
        None => {
            genoa::ask().map_err(|e| Error::CertChain(format!("failed to load Genoa ASK: {e}")))?
        }
    };

    let chain = Chain {
        ca: ca::Chain { ark, ask },
        vek: Certificate::from_der(vcek_der)
            .map_err(|e| Error::CertChain(format!("failed to parse VCEK cert: {e}")))?,
    };

    tracing::debug!("Verifying VCEK certificate chain...");
    let verified_vek = (&chain)
        .verify()
        .map_err(|e| Error::CertChain(format!("certificate chain verification failed: {e}")))?;

    tracing::debug!("Verifying attestation report signature...");
    (verified_vek, report)
        .verify()
        .map_err(|e| Error::Signature(format!("report signature verification failed: {e}")))?;

    check_vcek_binding(vcek_der, report)?;

    Ok(())
}

/// AMD KDS VCEK extension OIDs (VCEK/VLEK certificate spec, AMD 57230;
/// the same set `go-sev-guest`'s `kds` package reads).
const OID_BL_SPL: der::asn1::ObjectIdentifier =
    der::asn1::ObjectIdentifier::new_unwrap("1.3.6.1.4.1.3704.1.3.1");
const OID_TEE_SPL: der::asn1::ObjectIdentifier =
    der::asn1::ObjectIdentifier::new_unwrap("1.3.6.1.4.1.3704.1.3.2");
const OID_SNP_SPL: der::asn1::ObjectIdentifier =
    der::asn1::ObjectIdentifier::new_unwrap("1.3.6.1.4.1.3704.1.3.3");
const OID_UCODE_SPL: der::asn1::ObjectIdentifier =
    der::asn1::ObjectIdentifier::new_unwrap("1.3.6.1.4.1.3704.1.3.8");
const OID_HWID: der::asn1::ObjectIdentifier =
    der::asn1::ObjectIdentifier::new_unwrap("1.3.6.1.4.1.3704.1.4");

/// Bind the report to the VCEK that signed it.
///
/// A VCEK is derived per chip *and per TCB*: its certificate's SPL
/// extensions name the TCB it was endorsed for, and its HWID names the
/// chip. The signature alone proves only that *some* genuine VCEK signed
/// the report, and the VCEK arrives as untrusted collateral the document's
/// author chooses — so without this check, a VCEK whose key leaked from a
/// vulnerable TCB could sign a report claiming any `reported_tcb`, and
/// every floor in [`SevSnpTcbPolicy`] would be enforced against a number
/// the attacker wrote. This is what makes the module doc's "there is no
/// VCEK that would sign for a higher TCB" true. Mirrors `go-sev-guest`'s
/// `validateTcb` (`reported == cert`) and its CHIP_ID/HWID check, including
/// its tolerance for the KDS encoding the HWID as untagged raw bytes.
fn check_vcek_binding(vcek_der: &[u8], report: &AttestationReport) -> Result<(), Error> {
    let cert = x509_cert::Certificate::from_der(vcek_der)
        .map_err(|e| Error::CertParse(format!("failed to parse VCEK DER: {e}")))?;
    let extensions = cert.tbs_certificate.extensions.as_deref().unwrap_or(&[]);
    let find = |oid: der::asn1::ObjectIdentifier, name: &str| -> Result<&[u8], Error> {
        let mut found = extensions.iter().filter(|ext| ext.extn_id == oid);
        let ext = found.next().ok_or_else(|| {
            Error::CertChain(format!("VCEK certificate carries no {name} extension"))
        })?;
        if found.next().is_some() {
            return Err(Error::CertChain(format!(
                "VCEK certificate carries duplicate {name} extensions"
            )));
        }
        Ok(ext.extn_value.as_bytes())
    };
    let spl = |oid, name| -> Result<u8, Error> {
        u8::from_der(find(oid, name)?).map_err(|e| {
            Error::CertChain(format!(
                "VCEK {name} extension is not a DER INTEGER 0-255: {e}"
            ))
        })
    };

    let cert_tcb = SevSnpTcbSvns {
        bootloader: spl(OID_BL_SPL, "blSPL")?,
        tee: spl(OID_TEE_SPL, "teeSPL")?,
        snp: spl(OID_SNP_SPL, "snpSPL")?,
        microcode: spl(OID_UCODE_SPL, "ucodeSPL")?,
        fmc: None,
    };
    let reported = SevSnpTcbSvns::from(report.reported_tcb);
    if (
        cert_tcb.bootloader,
        cert_tcb.tee,
        cert_tcb.snp,
        cert_tcb.microcode,
    ) != (
        reported.bootloader,
        reported.tee,
        reported.snp,
        reported.microcode,
    ) {
        return Err(Error::CertChain(format!(
            "report reported_tcb ({reported}) is not the TCB its VCEK was endorsed for \
             ({cert_tcb})"
        )));
    }

    let hwid_ext = find(OID_HWID, "hwID")?;
    let hwid = match hwid_ext.len() {
        64 => hwid_ext,
        _ => der::asn1::OctetStringRef::from_der(hwid_ext)
            .map(|octets| octets.as_bytes())
            .ok()
            .filter(|bytes| bytes.len() == 64)
            .ok_or_else(|| Error::CertChain("VCEK hwID extension is not 64 bytes".to_string()))?,
    };
    // An all-zero CHIP_ID is a masked chip identity (MASK_CHIP_ID); there is
    // nothing to compare, as in go-sev-guest. The TCB binding above still
    // holds.
    if report.chip_id != [0u8; 64] && report.chip_id[..] != hwid[..] {
        return Err(Error::CertChain(
            "report CHIP_ID is not the hwID of the VCEK that signed it".to_string(),
        ));
    }

    Ok(())
}

/// Resolve ARK and ASK as raw DER bytes, falling back to the built-in
/// AMD Genoa certs when no override is supplied.
///
/// Used by the per-handshake connector to obtain the DER bytes it needs
/// for downstream operations the `sev` crate's `verify_report` doesn't
/// itself return — specifically, parsing into `x509-cert::Certificate`
/// for CRL signature verification and serial number extraction.
pub fn resolve_chain_certs_der(
    custom_ark: Option<&[u8]>,
    custom_ask: Option<&[u8]>,
) -> Result<(Vec<u8>, Vec<u8>), Error> {
    let ark = match custom_ark {
        Some(der) => der.to_vec(),
        None => genoa::ark()
            .map_err(|e| Error::CertChain(format!("failed to load Genoa ARK: {e}")))?
            .to_der()
            .map_err(|e| Error::CertChain(format!("failed to DER-encode Genoa ARK: {e}")))?,
    };
    let ask = match custom_ask {
        Some(der) => der.to_vec(),
        None => genoa::ask()
            .map_err(|e| Error::CertChain(format!("failed to load Genoa ASK: {e}")))?
            .to_der()
            .map_err(|e| Error::CertChain(format!("failed to DER-encode Genoa ASK: {e}")))?,
    };
    Ok((ark, ask))
}

/// Extract the raw serial number bytes from a DER-encoded X.509
/// certificate. The result is suitable for direct comparison against
/// the serial numbers in an `x509_cert::crl::CertificateList`.
pub fn cert_serial_from_der(cert_der: &[u8]) -> Result<Vec<u8>, Error> {
    let cert = x509_cert::Certificate::from_der(cert_der)
        .map_err(|e| Error::CertParse(format!("failed to parse cert DER: {e}")))?;
    Ok(cert.tbs_certificate.serial_number.as_bytes().to_vec())
}

/// Per-component TCB SVNs extracted from a SEV-SNP attestation report.
///
/// We re-define this rather than re-exporting [`sev::firmware::host::TcbVersion`]
/// so the public surface of `tinfoil-verifier` does not leak its dependency
/// on the `sev` crate. The field set covers everything we currently
/// inspect; the upstream type also carries an `Option<u8> fmc` field for
/// Turin and newer, which we copy through verbatim for forward
/// compatibility but do not enforce a floor on by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SevSnpTcbSvns {
    pub bootloader: u8,
    pub tee: u8,
    pub snp: u8,
    pub microcode: u8,
    /// Present on Turin and newer; `None` on Genoa and earlier.
    pub fmc: Option<u8>,
}

impl From<TcbVersion> for SevSnpTcbSvns {
    fn from(t: TcbVersion) -> Self {
        Self {
            bootloader: t.bootloader,
            tee: t.tee,
            snp: t.snp,
            microcode: t.microcode,
            fmc: t.fmc,
        }
    }
}

impl std::fmt::Display for SevSnpTcbSvns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "bl={:#04x} tee={:#04x} snp={:#04x} ucode={:#04x}",
            self.bootloader, self.tee, self.snp, self.microcode,
        )?;
        if let Some(fmc) = self.fmc {
            write!(f, " fmc={fmc:#04x}")?;
        }
        Ok(())
    }
}

/// Operator-supplied minimum TCB SVNs the verifier will accept.
///
/// Defaults match the floor enforced by `go-sev-guest` for the AMD Genoa
/// generation and the historical hardcoded constants this module shipped
/// with: `bootloader >= 0x07`, `snp >= 0x0E`, `microcode >= 0x48`. The
/// `tee` (PSP OS version) field was previously not checked at all; the
/// default of `0x00` preserves that behavior, but operators can tighten
/// it through [`crate::AttestingClientConfig::snp_min_tcb`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SevSnpTcbPolicy {
    pub min_bootloader: u8,
    pub min_tee: u8,
    pub min_snp: u8,
    pub min_microcode: u8,
}

impl SevSnpTcbPolicy {
    /// AMD-recommended floor matching the historical hardcoded constants.
    pub fn amd_recommended() -> Self {
        Self {
            min_bootloader: 0x07,
            min_tee: 0x00,
            min_snp: 0x0E,
            min_microcode: 0x48,
        }
    }

    /// Evaluate a parsed attestation report against this policy.
    ///
    /// Returns the [`SevSnpTcbObservation`] (always — even on failure, so
    /// observers can record rejected attestations) and a `Result` that is
    /// `Ok(())` when the report passes, or `Err(Error::TcbPolicy)` when
    /// either the rollback check fails or any SVN is below the configured
    /// floor. Rollback is checked before the floor so the error message
    /// reflects the more severe condition first.
    pub fn evaluate(
        &self,
        report: &AttestationReport,
    ) -> (SevSnpTcbObservation, Result<(), Error>) {
        let reported = SevSnpTcbSvns::from(report.reported_tcb);
        let committed = SevSnpTcbSvns::from(report.committed_tcb);

        let rollback = self.detect_rollback(&reported, &committed);
        let below_floor = self.detect_below_floor(&reported);

        let bucket = if rollback.is_some() {
            BUCKET_ROLLBACK
        } else if below_floor.is_some() {
            BUCKET_BELOW_FLOOR
        } else {
            BUCKET_MEETS_FLOOR
        };

        let observation = SevSnpTcbObservation {
            reported_tcb: reported,
            committed_tcb: committed,
            chip_id: report.chip_id,
            bucket,
        };

        let result = match (rollback, below_floor) {
            (Some(msg), _) => Err(Error::TcbPolicy(msg)),
            (None, Some(msg)) => Err(Error::TcbPolicy(msg)),
            (None, None) => Ok(()),
        };

        (observation, result)
    }

    fn detect_rollback(
        &self,
        reported: &SevSnpTcbSvns,
        committed: &SevSnpTcbSvns,
    ) -> Option<String> {
        if reported.bootloader < committed.bootloader
            || reported.tee < committed.tee
            || reported.snp < committed.snp
            || reported.microcode < committed.microcode
        {
            Some(format!(
                "SEV-SNP reported_tcb ({reported}) is below committed_tcb ({committed}); \
                 possible firmware rollback or hypervisor SET_TCB_VERSION abuse",
            ))
        } else {
            None
        }
    }

    fn detect_below_floor(&self, reported: &SevSnpTcbSvns) -> Option<String> {
        let mut violations: Vec<String> = Vec::new();
        if reported.bootloader < self.min_bootloader {
            violations.push(format!(
                "bootloader {:#04x} < min {:#04x}",
                reported.bootloader, self.min_bootloader
            ));
        }
        if reported.tee < self.min_tee {
            violations.push(format!(
                "tee {:#04x} < min {:#04x}",
                reported.tee, self.min_tee
            ));
        }
        if reported.snp < self.min_snp {
            violations.push(format!(
                "snp {:#04x} < min {:#04x}",
                reported.snp, self.min_snp
            ));
        }
        if reported.microcode < self.min_microcode {
            violations.push(format!(
                "microcode {:#04x} < min {:#04x}",
                reported.microcode, self.min_microcode
            ));
        }
        if violations.is_empty() {
            None
        } else {
            Some(format!(
                "SEV-SNP reported_tcb below operator floor: {}",
                violations.join(", "),
            ))
        }
    }
}

impl Default for SevSnpTcbPolicy {
    fn default() -> Self {
        Self::amd_recommended()
    }
}

// Bucket labels for `SevSnpTcbObservation::as_metric_label`. Stable
// strings — dashboards and alert rules depend on these.
const BUCKET_MEETS_FLOOR: &str = "meets_floor";
const BUCKET_BELOW_FLOOR: &str = "below_floor";
const BUCKET_ROLLBACK: &str = "rollback_detected";

/// Observation surfaced after a SEV-SNP attestation has been
/// signature-verified, before the policy result is propagated.
///
/// Consumers receive this via the optional observer callback on
/// [`crate::AttestingClientConfig`] and can use it to drive metrics,
/// traces, or alerting. The observer fires for *every* attestation that
/// completes signature verification, including ones the policy
/// subsequently rejects, so operators have full visibility into the
/// population of observed TCB levels — not just the ones that made it
/// through.
#[derive(Debug, Clone)]
pub struct SevSnpTcbObservation {
    /// TCB version associated with the VCEK that signed the report.
    pub reported_tcb: SevSnpTcbSvns,
    /// TCB version the firmware has one-way-committed to. The verifier
    /// requires `reported_tcb >= committed_tcb` componentwise.
    pub committed_tcb: SevSnpTcbSvns,
    /// 64-byte chip identifier from the report. High cardinality —
    /// suitable for trace enrichment, *not* as a metric label.
    pub chip_id: [u8; 64],
    bucket: &'static str,
}

impl SevSnpTcbObservation {
    /// Stable lowercase identifier suitable for use as a metric label.
    /// One of `meets_floor`, `below_floor`, or `rollback_detected`.
    pub fn as_metric_label(&self) -> &'static str {
        self.bucket
    }
}

/// Observer callback type. Invoked synchronously inside the connector
/// layer for every SEV-SNP attestation that completes signature
/// verification, regardless of policy outcome. Implementations must be
/// cheap and non-blocking — they run on the TLS handshake hot path.
pub type SevSnpObserver = Arc<dyn Fn(&SevSnpTcbObservation) + Send + Sync>;

/// Compute SHA-256 of the SPKI from a raw DER-encoded certificate.
pub fn sha256_spki_from_der(cert_der: &[u8]) -> Result<[u8; 32], Error> {
    let cert = x509_cert::Certificate::from_der(cert_der)
        .map_err(|e| Error::CertParse(format!("failed to parse cert DER: {e}")))?;
    let spki_der = cert
        .tbs_certificate
        .subject_public_key_info
        .to_der()
        .map_err(|e| Error::CertParse(format!("failed to encode SPKI to DER: {e}")))?;
    Ok(Sha256::digest(&spki_der).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genoa_builtin_certs_load() {
        genoa::ark().expect("failed to load Genoa ARK");
        genoa::ask().expect("failed to load Genoa ASK");
    }

    #[test]
    fn test_genoa_chain_verifies() {
        let ca_chain = ca::Chain {
            ark: genoa::ark().unwrap(),
            ask: genoa::ask().unwrap(),
        };
        (&ca_chain)
            .verify()
            .expect("Genoa CA chain verification failed");
    }

    fn report_with_tcb(reported: SevSnpTcbSvns, committed: SevSnpTcbSvns) -> AttestationReport {
        AttestationReport {
            reported_tcb: TcbVersion {
                bootloader: reported.bootloader,
                tee: reported.tee,
                snp: reported.snp,
                microcode: reported.microcode,
                fmc: reported.fmc,
            },
            committed_tcb: TcbVersion {
                bootloader: committed.bootloader,
                tee: committed.tee,
                snp: committed.snp,
                microcode: committed.microcode,
                fmc: committed.fmc,
            },
            ..Default::default()
        }
    }

    fn svns(bootloader: u8, tee: u8, snp: u8, microcode: u8) -> SevSnpTcbSvns {
        SevSnpTcbSvns {
            bootloader,
            tee,
            snp,
            microcode,
            fmc: None,
        }
    }

    #[test]
    fn policy_accepts_report_at_or_above_floor() {
        let policy = SevSnpTcbPolicy::amd_recommended();
        let reported = svns(0x07, 0x00, 0x0E, 0x48);
        let committed = svns(0x07, 0x00, 0x0E, 0x48);
        let report = report_with_tcb(reported, committed);
        let (obs, result) = policy.evaluate(&report);
        assert!(result.is_ok());
        assert_eq!(obs.as_metric_label(), "meets_floor");
    }

    #[test]
    fn policy_accepts_report_above_floor() {
        let policy = SevSnpTcbPolicy::amd_recommended();
        let reported = svns(0x10, 0x05, 0x20, 0x80);
        let committed = svns(0x10, 0x05, 0x20, 0x80);
        let (obs, result) = policy.evaluate(&report_with_tcb(reported, committed));
        assert!(result.is_ok());
        assert_eq!(obs.as_metric_label(), "meets_floor");
    }

    #[test]
    fn policy_rejects_below_bootloader_floor() {
        let policy = SevSnpTcbPolicy::amd_recommended();
        let reported = svns(0x06, 0x00, 0x0E, 0x48); // bootloader one short
        let committed = svns(0x06, 0x00, 0x0E, 0x48); // committed matches, no rollback
        let (obs, result) = policy.evaluate(&report_with_tcb(reported, committed));
        let err = result.unwrap_err();
        assert!(matches!(err, Error::TcbPolicy(_)));
        assert_eq!(obs.as_metric_label(), "below_floor");
        let Error::TcbPolicy(msg) = err else {
            unreachable!()
        };
        assert!(msg.contains("bootloader"), "got: {msg}");
    }

    #[test]
    fn policy_rejects_below_snp_and_microcode_floor() {
        let policy = SevSnpTcbPolicy::amd_recommended();
        let reported = svns(0x07, 0x00, 0x05, 0x10);
        let committed = reported;
        let (_, result) = policy.evaluate(&report_with_tcb(reported, committed));
        let Error::TcbPolicy(msg) = result.unwrap_err() else {
            panic!("expected TcbPolicy error");
        };
        assert!(msg.contains("snp"), "got: {msg}");
        assert!(msg.contains("microcode"), "got: {msg}");
    }

    #[test]
    fn policy_detects_rollback_even_when_above_floor() {
        let policy = SevSnpTcbPolicy::amd_recommended();
        // reported is above the floor on every component, but the
        // firmware has committed to a higher snp SVN. This is the case
        // a malicious hypervisor SET_TCB_VERSION call would produce.
        let reported = svns(0x10, 0x05, 0x10, 0x80);
        let committed = svns(0x10, 0x05, 0x14, 0x80);
        let (obs, result) = policy.evaluate(&report_with_tcb(reported, committed));
        let Error::TcbPolicy(msg) = result.unwrap_err() else {
            panic!("expected TcbPolicy error");
        };
        assert_eq!(obs.as_metric_label(), "rollback_detected");
        assert!(msg.contains("rollback"), "got: {msg}");
    }

    #[test]
    fn policy_rollback_takes_precedence_over_below_floor() {
        // Both rollback and below-floor: the more severe condition wins
        // and the bucket label reflects rollback.
        let policy = SevSnpTcbPolicy::amd_recommended();
        let reported = svns(0x05, 0x00, 0x05, 0x10);
        let committed = svns(0x07, 0x00, 0x0E, 0x48);
        let (obs, result) = policy.evaluate(&report_with_tcb(reported, committed));
        assert!(result.is_err());
        assert_eq!(obs.as_metric_label(), "rollback_detected");
    }

    #[test]
    fn policy_default_matches_amd_recommended() {
        assert_eq!(
            SevSnpTcbPolicy::default(),
            SevSnpTcbPolicy::amd_recommended()
        );
    }

    /// Minimal parseable version-3 (Genoa) report byte buffer with a
    /// production-shaped guest policy, mutated per test.
    fn v3_report_bytes(mutate: impl FnOnce(&mut [u8; 1184])) -> Vec<u8> {
        let mut b = [0u8; 1184];
        b[0..4].copy_from_slice(&3u32.to_le_bytes()); // version 3
        b[0x08..0x10].copy_from_slice(&0x30000u64.to_le_bytes()); // policy: SMT + reserved-1 bit
        b[0x34..0x38].copy_from_slice(&1u32.to_le_bytes()); // sig_algo: ECDSA P-384
        b[0x188] = 0x19; // CPUID family: Genoa
        b[0x189] = 0x11; // CPUID model
        b[0x18A] = 0x01; // CPUID stepping
        mutate(&mut b);
        b.to_vec()
    }

    #[test]
    fn parse_rejects_wrong_report_length() {
        let ok = v3_report_bytes(|_| {});
        assert!(parse_report(&ok).is_ok());
        let mut long = ok.clone();
        long.push(0);
        assert!(parse_report(&long).is_err(), "trailing byte must fail");
        assert!(parse_report(&ok[..1183]).is_err(), "short report must fail");
    }

    #[test]
    fn hygiene_accepts_production_shaped_report() {
        let report = parse_report(&v3_report_bytes(|_| {})).unwrap();
        check_report_hygiene(&report).expect("clean v3 report must pass");
    }

    #[test]
    fn hygiene_rejects_report_version_below_3() {
        // A version-2 report needs the chip_id heuristic for generation
        // detection instead of the CPUID fields.
        let bytes = v3_report_bytes(|b| {
            b[0..4].copy_from_slice(&2u32.to_le_bytes());
            b[0x188] = 0;
            b[0x189] = 0;
            b[0x18A] = 0;
            b[0x1A0 + 8] = 0x01; // chip_id byte 8+ nonzero → Genoa detection
        });
        let report = parse_report(&bytes).unwrap();
        let err = check_report_hygiene(&report).unwrap_err();
        assert!(err.to_string().contains("version"), "got: {err}");
    }

    #[test]
    fn hygiene_rejects_debug_allowed_policy() {
        let bytes = v3_report_bytes(|b| b[0x0A] |= 0x08); // policy bit 19: DEBUG
        let report = parse_report(&bytes).unwrap();
        let err = check_report_hygiene(&report).unwrap_err();
        assert!(err.to_string().contains("DEBUG"), "got: {err}");
    }

    #[test]
    fn hygiene_rejects_migration_agent_policy() {
        let bytes = v3_report_bytes(|b| b[0x0A] |= 0x04); // policy bit 18: MIGRATE_MA
        let report = parse_report(&bytes).unwrap();
        let err = check_report_hygiene(&report).unwrap_err();
        assert!(err.to_string().contains("migration"), "got: {err}");
    }

    #[test]
    fn hygiene_rejects_masked_chip_key_and_non_vcek_signer() {
        let masked = v3_report_bytes(|b| b[0x48] |= 0x02); // MASK_CHIP_KEY
        let err = check_report_hygiene(&parse_report(&masked).unwrap()).unwrap_err();
        assert!(err.to_string().contains("MASK_CHIP_KEY"), "got: {err}");

        let vlek = v3_report_bytes(|b| b[0x48] |= 0x04); // SIGNING_KEY = 1 (VLEK)
        let err = check_report_hygiene(&parse_report(&vlek).unwrap()).unwrap_err();
        assert!(err.to_string().contains("SIGNING_KEY"), "got: {err}");
    }

    #[test]
    fn hygiene_rejects_id_block_launches() {
        for mutate in [
            (|b: &mut [u8; 1184]| b[0x48] |= 0x01) as fn(&mut [u8; 1184]), // AUTHOR_KEY_EN
            |b| b[0xE0] = 0xAA,                                            // ID_KEY_DIGEST
            |b| b[0x110] = 0xBB,                                           // AUTHOR_KEY_DIGEST
        ] {
            let bytes = v3_report_bytes(mutate);
            let err = check_report_hygiene(&parse_report(&bytes).unwrap()).unwrap_err();
            assert!(err.to_string().contains("ID-block"), "got: {err}");
        }
    }

    #[test]
    fn policy_can_tighten_individual_components() {
        let mut policy = SevSnpTcbPolicy::amd_recommended();
        policy.min_snp = 0x14;
        let reported = svns(0x07, 0x00, 0x10, 0x48); // above old floor, below new
        let (obs, result) = policy.evaluate(&report_with_tcb(reported, reported));
        assert!(result.is_err());
        assert_eq!(obs.as_metric_label(), "below_floor");
    }

    // --- VCEK binding -----------------------------------------------------
    //
    // The binding reads only the VCEK's extensions, so a synthetic VCEK is
    // the built-in Genoa ASK with its extensions replaced. Nothing here is
    // signed; `verify_report` checks the signature before the binding.

    const HWID: [u8; 64] = [0xA5; 64];

    fn kds_extension(
        oid: der::asn1::ObjectIdentifier,
        value: Vec<u8>,
    ) -> x509_cert::ext::Extension {
        x509_cert::ext::Extension {
            extn_id: oid,
            critical: false,
            extn_value: der::asn1::OctetString::new(value).unwrap(),
        }
    }

    fn spl_extension(oid: der::asn1::ObjectIdentifier, spl: u8) -> x509_cert::ext::Extension {
        kds_extension(oid, spl.to_der().unwrap())
    }

    /// A VCEK endorsed for `tcb` on chip `HWID`, with the hwID encoded the
    /// way AMD's KDS writes it: 64 raw bytes, no OCTET STRING tag.
    fn kds_extensions(tcb: SevSnpTcbSvns) -> Vec<x509_cert::ext::Extension> {
        vec![
            spl_extension(OID_BL_SPL, tcb.bootloader),
            spl_extension(OID_TEE_SPL, tcb.tee),
            spl_extension(OID_SNP_SPL, tcb.snp),
            spl_extension(OID_UCODE_SPL, tcb.microcode),
            kds_extension(OID_HWID, HWID.to_vec()),
        ]
    }

    fn vcek_with(extensions: Vec<x509_cert::ext::Extension>) -> Vec<u8> {
        let der = genoa::ask().unwrap().to_der().unwrap();
        let mut cert = x509_cert::Certificate::from_der(&der).unwrap();
        cert.tbs_certificate.extensions = Some(extensions);
        cert.to_der().unwrap()
    }

    fn report_signed_for(tcb: SevSnpTcbSvns, chip_id: [u8; 64]) -> AttestationReport {
        AttestationReport {
            chip_id,
            ..report_with_tcb(tcb, tcb)
        }
    }

    #[test]
    fn vcek_binding_accepts_the_tcb_and_chip_the_vcek_was_endorsed_for() {
        let tcb = svns(0x07, 0x00, 0x0E, 0x48);
        let report = report_signed_for(tcb, HWID);
        check_vcek_binding(&vcek_with(kds_extensions(tcb)), &report).expect("raw hwID");

        // go-sev-guest also tolerates a tagged OCTET STRING hwID.
        let mut tagged = kds_extensions(tcb);
        tagged[4] = kds_extension(
            OID_HWID,
            der::asn1::OctetString::new(HWID.to_vec())
                .unwrap()
                .to_der()
                .unwrap(),
        );
        check_vcek_binding(&vcek_with(tagged), &report).expect("tagged hwID");
    }

    #[test]
    fn vcek_binding_refuses_a_report_claiming_a_tcb_its_vcek_was_not_endorsed_for() {
        // The leaked key's VCEK is for an old microcode; the report claims the
        // patched one, which every floor would otherwise accept.
        let endorsed = svns(0x07, 0x00, 0x0E, 0x0A);
        let claimed = svns(0x07, 0x00, 0x0E, 0x54);
        let vcek = vcek_with(kds_extensions(endorsed));
        let err = check_vcek_binding(&vcek, &report_signed_for(claimed, HWID)).unwrap_err();
        assert!(
            err.to_string()
                .contains("not the TCB its VCEK was endorsed for"),
            "{err}"
        );
        // Each component is compared.
        let bl = svns(0x05, 0x00, 0x0E, 0x0A);
        assert!(check_vcek_binding(&vcek, &report_signed_for(bl, HWID)).is_err());
    }

    #[test]
    fn vcek_binding_refuses_another_chips_vcek() {
        let tcb = svns(0x07, 0x00, 0x0E, 0x48);
        let vcek = vcek_with(kds_extensions(tcb));
        let err = check_vcek_binding(&vcek, &report_signed_for(tcb, [0x5A; 64])).unwrap_err();
        assert!(err.to_string().contains("CHIP_ID"), "{err}");
    }

    #[test]
    fn vcek_binding_tolerates_a_masked_chip_id() {
        // MASK_CHIP_ID zeroes CHIP_ID; there is nothing to compare, as in
        // go-sev-guest. The TCB binding still holds.
        let tcb = svns(0x07, 0x00, 0x0E, 0x48);
        let vcek = vcek_with(kds_extensions(tcb));
        check_vcek_binding(&vcek, &report_signed_for(tcb, [0; 64])).expect("masked chip id");
        let claimed = svns(0x07, 0x00, 0x0E, 0x54);
        assert!(check_vcek_binding(&vcek, &report_signed_for(claimed, [0; 64])).is_err());
    }

    #[test]
    fn vcek_binding_refuses_missing_duplicate_or_malformed_extensions() {
        let tcb = svns(0x07, 0x00, 0x0E, 0x48);
        let report = report_signed_for(tcb, HWID);

        let mut no_hwid = kds_extensions(tcb);
        no_hwid.pop();
        let err = check_vcek_binding(&vcek_with(no_hwid), &report).unwrap_err();
        assert!(err.to_string().contains("no hwID extension"), "{err}");

        let mut twice = kds_extensions(tcb);
        twice.push(spl_extension(OID_UCODE_SPL, tcb.microcode));
        let err = check_vcek_binding(&vcek_with(twice), &report).unwrap_err();
        assert!(err.to_string().contains("duplicate ucodeSPL"), "{err}");

        let mut wide = kds_extensions(tcb);
        wide[0] = kds_extension(OID_BL_SPL, 0x0107u16.to_der().unwrap());
        let err = check_vcek_binding(&vcek_with(wide), &report).unwrap_err();
        assert!(err.to_string().contains("not a DER INTEGER 0-255"), "{err}");

        let mut short_hwid = kds_extensions(tcb);
        short_hwid[4] = kds_extension(OID_HWID, vec![0xA5; 32]);
        let err = check_vcek_binding(&vcek_with(short_hwid), &report).unwrap_err();
        assert!(err.to_string().contains("not 64 bytes"), "{err}");

        // The ASK itself carries no KDS extensions at all.
        let ask = genoa::ask().unwrap().to_der().unwrap();
        assert!(check_vcek_binding(&ask, &report).is_err());
    }
}
