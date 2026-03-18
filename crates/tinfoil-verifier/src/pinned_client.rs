//! TLS certificate pinning via public key fingerprint verification.

use std::sync::Arc;

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{
    CryptoProvider, WebPkiSupportedAlgorithms, verify_tls12_signature, verify_tls13_signature,
};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, Error as TlsError, SignatureScheme};

use crate::sevsnp;

/// A rustls `ServerCertVerifier` that accepts only certificates whose SPKI SHA-256
/// fingerprint matches an expected value (derived from the attestation report).
#[derive(Debug)]
struct FingerprintVerifier {
    expected_fingerprint: [u8; 32],
    supported_algs: WebPkiSupportedAlgorithms,
}

impl FingerprintVerifier {
    fn new(expected_fingerprint: [u8; 32], provider: &Arc<CryptoProvider>) -> Self {
        Self {
            expected_fingerprint,
            supported_algs: provider.signature_verification_algorithms.clone(),
        }
    }
}

impl ServerCertVerifier for FingerprintVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        let actual = sevsnp::sha256_spki_from_der(end_entity.as_ref())
            .map_err(|e| TlsError::General(format!("failed to compute SPKI fingerprint: {e}")))?;
        if actual != self.expected_fingerprint {
            return Err(TlsError::General(format!(
                "TLS public key fingerprint mismatch: expected {}, got {}",
                hex::encode(self.expected_fingerprint),
                hex::encode(actual),
            )));
        }
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        verify_tls12_signature(message, cert, dss, &self.supported_algs)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        verify_tls13_signature(message, cert, dss, &self.supported_algs)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.supported_algs.supported_schemes()
    }
}

/// Build a `reqwest::Client` that pins TLS connections to the given SPKI fingerprint.
pub fn build_pinned_client(
    expected_fingerprint: [u8; 32],
) -> Result<reqwest::Client, crate::Error> {
    let provider = rustls::crypto::CryptoProvider::get_default()
        .ok_or_else(|| crate::Error::Tls("no rustls CryptoProvider installed".into()))?;

    let verifier = FingerprintVerifier::new(expected_fingerprint, provider);

    let tls_config = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(verifier))
        .with_no_client_auth();

    reqwest::Client::builder()
        .use_preconfigured_tls(tls_config)
        .build()
        .map_err(|e| crate::Error::Tls(format!("failed to build pinned HTTP client: {e}")))
}
