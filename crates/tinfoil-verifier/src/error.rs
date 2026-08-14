use thiserror::Error;

use crate::measurement::MatchedMeasurement;

#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to fetch attestation document: {0}")]
    Fetch(#[from] reqwest::Error),

    #[error("invalid attestation document: {0}")]
    Bundle(String),

    #[error("invalid SEV-SNP report: {0}")]
    Report(String),

    #[error(
        "TDX attestation refused: MRTD/RTMR0 policy checks are not implemented, and \
         RTMR1/RTMR2 alone are replayable by guest firmware — only SEV-SNP attestations \
         are accepted (see docs/gaps.md § TDX acceptance)"
    )]
    TdxNotAccepted,

    #[error("VCEK certificate chain verification failed: {0}")]
    CertChain(String),

    #[error("report signature verification failed: {0}")]
    Signature(String),

    #[error("attestation nonce mismatch: sent {sent}, enclave echoed {echoed}")]
    NonceMismatch { sent: String, echoed: String },

    #[error("report_data mismatch: expected {expected}, observed {observed}")]
    ReportDataMismatch { expected: String, observed: String },

    #[error("TCB policy violation: {0}")]
    TcbPolicy(String),

    #[error(
        "measurement mismatch: observed {observed} is not in the allowed list ({allowed_count} entries)"
    )]
    MeasurementMismatch {
        observed: MatchedMeasurement,
        allowed_count: usize,
    },

    #[error("TLS fingerprint mismatch: attested={attested}, peer={peer}")]
    FingerprintMismatch { attested: String, peer: String },

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
