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
//!   because there is no VCEK that would sign for a higher TCB).
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

use base64::Engine;
use der::{Decode, Encode};
use sev::certs::snp::{Certificate, Chain, Verifiable, builtin::genoa, ca};
use sev::firmware::guest::AttestationReport;
use sev::firmware::host::TcbVersion;
use sev::parser::Decoder;
use sha2::{Digest, Sha256};

use crate::Error;

/// Decode base64, ignoring whitespace and PEM headers/footers.
pub fn decode_base64(s: &str) -> Result<Vec<u8>, Error> {
    let clean: String = s
        .lines()
        .filter(|line| !line.trim().starts_with("-----"))
        .flat_map(|line| line.chars().filter(|c| !c.is_whitespace()))
        .collect();

    base64::engine::general_purpose::STANDARD
        .decode(&clean)
        .map_err(|e| Error::CertParse(format!("base64 decode: {e}")))
}

/// Parse a raw attestation report without verifying its signature.
pub fn parse_report(report_bytes: &[u8]) -> Result<AttestationReport, Error> {
    AttestationReport::decode(&mut Cursor::new(report_bytes), ())
        .map_err(|e| Error::Report(format!("failed to parse attestation report: {e}")))
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

    Ok(())
}

/// Verify the VCEK certificate chain and attestation report signature.
///
/// Convenience wrapper that parses the report and verifies in one call.
/// Returns the parsed [`AttestationReport`] on success.
pub fn verify_attestation(
    vcek_der: &[u8],
    report_bytes: &[u8],
    ark_der: Option<&[u8]>,
    ask_der: Option<&[u8]>,
) -> Result<AttestationReport, Error> {
    let report = parse_report(report_bytes)?;
    verify_report(vcek_der, &report, ark_der, ask_der)?;
    Ok(report)
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

/// Verify the enclave certificate's public key fingerprint matches report_data[0..32].
pub fn verify_enclave_cert_binding(
    enclave_cert_b64: &str,
    expected_fingerprint: &[u8],
) -> Result<(), Error> {
    let cert_der = decode_base64(enclave_cert_b64)?;
    let actual = sha256_spki_from_der(&cert_der)?;

    if actual.as_slice() != expected_fingerprint {
        return Err(Error::FingerprintMismatch {
            report_data: hex::encode(expected_fingerprint),
            enclave_cert: hex::encode(actual),
        });
    }

    Ok(())
}

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

    #[test]
    fn policy_can_tighten_individual_components() {
        let mut policy = SevSnpTcbPolicy::amd_recommended();
        policy.min_snp = 0x14;
        let reported = svns(0x07, 0x00, 0x10, 0x48); // above old floor, below new
        let (obs, result) = policy.evaluate(&report_with_tcb(reported, reported));
        assert!(result.is_err());
        assert_eq!(obs.as_metric_label(), "below_floor");
    }
}
