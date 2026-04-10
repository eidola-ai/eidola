use thiserror::Error;

use crate::measurement::MatchedMeasurement;

#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to fetch attestation bundle: {0}")]
    Fetch(#[from] reqwest::Error),

    #[error("invalid attestation bundle: {0}")]
    Bundle(String),

    #[error("base64 decode error: {0}")]
    Base64(#[from] base64::DecodeError),

    #[error("decompression error: {0}")]
    Decompress(String),

    #[error("invalid SEV-SNP report: {0}")]
    Report(String),

    #[error("TDX verification error: {0}")]
    Tdx(String),

    #[error("VCEK certificate chain verification failed: {0}")]
    CertChain(String),

    #[error("report signature verification failed: {0}")]
    Signature(String),

    #[error("TCB policy violation: {0}")]
    TcbPolicy(String),

    #[error(
        "measurement mismatch: observed {observed} is not in the allowed list ({allowed_count} entries)"
    )]
    MeasurementMismatch {
        observed: MatchedMeasurement,
        allowed_count: usize,
    },

    #[error("TLS fingerprint mismatch: report_data={report_data}, enclave_cert={enclave_cert}")]
    FingerprintMismatch {
        report_data: String,
        enclave_cert: String,
    },

    #[error("certificate parse error: {0}")]
    CertParse(String),

    #[error("TLS configuration error: {0}")]
    Tls(String),

    /// Catch-all for failures that happen inside the per-handshake attesting
    /// connector layer: HTTP/1.1 framing errors, EOF, missing TLS info on the
    /// freshly-handshaken connection, JSON parse failures on the attestation
    /// document body, and similar.
    #[error("attestation connector error: {0}")]
    Connector(String),

    /// The inline attestation fetch did not complete within the configured
    /// per-handshake deadline. The TLS handshake itself succeeded, but either
    /// the upstream stalled before serving the well-known document or the
    /// HTTP response was being streamed unusually slowly.
    #[error("inline attestation fetch timed out after {seconds}s")]
    AttestationTimeout { seconds: u64 },
}
