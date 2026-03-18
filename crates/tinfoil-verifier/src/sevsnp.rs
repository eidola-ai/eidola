//! SEV-SNP attestation verification using the `sev` crate.
//!
//! Delegates certificate chain verification and report signature checking to
//! the [`sev`](https://crates.io/crates/sev) crate (virtee/sev). This crate
//! only adds TCB policy enforcement and TLS fingerprint cross-checking.

use std::io::Cursor;

use der::{Decode, DecodePem, Encode};
use sev::certs::snp::{Certificate, Chain, Verifiable, builtin::genoa, ca};
use sev::firmware::guest::AttestationReport;
use sev::parser::Decoder;
use sha2::{Digest, Sha256};

use crate::Error;

// TCB policy minimums (matching Go SDK / go-sev-guest)
const MIN_BL_SPL: u8 = 0x07;
const MIN_SNP_SPL: u8 = 0x0E;
const MIN_UCODE_SPL: u8 = 0x48;

/// Verify the VCEK certificate chain and attestation report signature.
///
/// 1. Builds the chain: embedded Genoa ARK → ASK → VCEK (from bundle)
/// 2. Verifies the chain (ARK self-signed, ARK signs ASK, ASK signs VCEK)
/// 3. Parses the raw attestation report
/// 4. Verifies the report's ECDSA-P384 signature against the VCEK
///
/// Returns the parsed [`AttestationReport`] on success.
pub fn verify_attestation(
    vcek_der: &[u8],
    report_bytes: &[u8],
) -> Result<AttestationReport, Error> {
    let chain = Chain {
        ca: ca::Chain {
            ark: genoa::ark()
                .map_err(|e| Error::CertChain(format!("failed to load Genoa ARK: {e}")))?,
            ask: genoa::ask()
                .map_err(|e| Error::CertChain(format!("failed to load Genoa ASK: {e}")))?,
        },
        vek: Certificate::from_der(vcek_der)
            .map_err(|e| Error::CertChain(format!("failed to parse VCEK cert: {e}")))?,
    };

    tracing::debug!("Verifying VCEK certificate chain...");
    let verified_vek = (&chain)
        .verify()
        .map_err(|e| Error::CertChain(format!("certificate chain verification failed: {e}")))?;

    let report = AttestationReport::decode(&mut Cursor::new(report_bytes), ())
        .map_err(|e| Error::Report(format!("failed to parse attestation report: {e}")))?;

    tracing::debug!("Verifying attestation report signature...");
    (verified_vek, &report)
        .verify()
        .map_err(|e| Error::Signature(format!("report signature verification failed: {e}")))?;

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
    enclave_cert_pem: &str,
    expected_fingerprint: &[u8],
) -> Result<(), Error> {
    let actual = sha256_spki_pem(enclave_cert_pem)?;

    if actual.as_slice() != expected_fingerprint {
        return Err(Error::FingerprintMismatch {
            report_data: hex::encode(expected_fingerprint),
            enclave_cert: hex::encode(actual),
        });
    }

    Ok(())
}

/// Compute SHA-256 of the SPKI from a PEM-encoded certificate.
fn sha256_spki_pem(pem_data: &str) -> Result<[u8; 32], Error> {
    let cert = x509_cert::Certificate::from_pem(pem_data)
        .map_err(|e| Error::CertParse(format!("failed to parse enclave cert PEM: {e}")))?;
    sha256_spki_x509(&cert)
}

/// Compute SHA-256 of the SPKI from a parsed x509-cert Certificate.
fn sha256_spki_x509(cert: &x509_cert::Certificate) -> Result<[u8; 32], Error> {
    let spki_der = cert
        .tbs_certificate
        .subject_public_key_info
        .to_der()
        .map_err(|e| Error::CertParse(format!("failed to encode SPKI to DER: {e}")))?;
    Ok(Sha256::digest(&spki_der).into())
}

/// Compute SHA-256 of the SPKI from a raw DER-encoded certificate.
pub fn sha256_spki_from_der(cert_der: &[u8]) -> Result<[u8; 32], Error> {
    let cert = x509_cert::Certificate::from_der(cert_der)
        .map_err(|e| Error::CertParse(format!("failed to parse cert DER: {e}")))?;
    sha256_spki_x509(&cert)
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
