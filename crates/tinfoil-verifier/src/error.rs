use thiserror::Error;

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

    #[error("VCEK certificate chain verification failed: {0}")]
    CertChain(String),

    #[error("report signature verification failed: {0}")]
    Signature(String),

    #[error("TCB policy violation: {0}")]
    TcbPolicy(String),

    #[error("measurement mismatch: got {measurement}, allowed: {allowed:?}")]
    MeasurementMismatch {
        measurement: String,
        allowed: Vec<String>,
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
}
