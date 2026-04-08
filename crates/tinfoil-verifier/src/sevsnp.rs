//! SEV-SNP attestation verification using the `sev` crate.
//!
//! Delegates certificate chain verification and report signature checking to
//! the [`sev`](https://crates.io/crates/sev) crate (virtee/sev). This crate
//! only adds TCB policy enforcement and TLS fingerprint cross-checking.

use std::io::Cursor;

use base64::Engine;
use der::{Decode, Encode};
use sev::certs::snp::{Certificate, Chain, Verifiable, builtin::genoa, ca};
use sev::firmware::guest::AttestationReport;
use sev::parser::Decoder;
use sha2::{Digest, Sha256};

use crate::Error;

// TCB policy minimums (matching Go SDK / go-sev-guest)
const MIN_BL_SPL: u8 = 0x07;
const MIN_SNP_SPL: u8 = 0x0E;
const MIN_UCODE_SPL: u8 = 0x48;

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

/// Validate TCB version against minimum policy.
pub fn verify_tcb_policy(report: &AttestationReport) -> Result<(), Error> {
    let tcb = &report.reported_tcb;

    if tcb.bootloader < MIN_BL_SPL {
        return Err(Error::TcbPolicy(format!(
            "bl_spl {:#04x} < minimum {MIN_BL_SPL:#04x}",
            tcb.bootloader
        )));
    }
    if tcb.snp < MIN_SNP_SPL {
        return Err(Error::TcbPolicy(format!(
            "snp_spl {:#04x} < minimum {MIN_SNP_SPL:#04x}",
            tcb.snp
        )));
    }
    if tcb.microcode < MIN_UCODE_SPL {
        return Err(Error::TcbPolicy(format!(
            "ucode_spl {:#04x} < minimum {MIN_UCODE_SPL:#04x}",
            tcb.microcode
        )));
    }

    Ok(())
}

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
}
