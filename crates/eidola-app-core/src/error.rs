/// Errors returned by app-core operations.
///
/// Each variant maps to a distinct failure mode so callers (CLI, GUI) can
/// display appropriate feedback without parsing error strings.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    /// A required configuration value is missing (base_url, account, etc.).
    #[error("not configured: {message}")]
    NotConfigured { message: String },

    /// An HTTP request failed at the transport layer.
    #[error("network error: {message}")]
    Network { message: String },

    /// Enclave attestation verification failed.
    #[error("attestation failed: {message}")]
    Attestation { message: String },

    /// The server returned a non-success HTTP status.
    #[error("server error ({status}): {message}")]
    Server { status: u16, message: String },

    /// An anonymous credential operation failed.
    #[error("credential error: {message}")]
    Credential { message: String },

    /// A chat was attempted with no usable credential and no account
    /// configured — onboarding has not begun (or the account was reset).
    /// Distinct from [`AppError::NotConfigured`] so UIs can route to the
    /// account-creation step instead of a generic config error.
    #[error("no account configured — create an account to begin")]
    NoAccount,

    /// The account exists but its available balance cannot cover the
    /// credits required for the attempted operation. Carries both sides
    /// of the comparison so UIs can show honest numbers and route to the
    /// purchase step.
    #[error("insufficient balance: {required} credits required, {available} available")]
    InsufficientBalance { available: i64, required: i64 },

    /// A local database operation failed.
    #[error("database error: {message}")]
    Database { message: String },

    /// Configuration read/write error.
    #[error("config error: {message}")]
    Config { message: String },

    /// An internal runtime or system error.
    #[error("internal error: {message}")]
    Internal { message: String },

    /// A self-update verification step failed.
    ///
    /// Used by [`crate::updater`] to surface fetch/parse/schema/continuity
    /// problems before any cryptographic verification stage runs; the
    /// crypto stages produce [`AppError::Attestation`] instead.
    #[error("update error: {message}")]
    Update { message: String },
}

// ---------------------------------------------------------------------------
// Internal conversion helpers
// ---------------------------------------------------------------------------

impl AppError {
    /// Classify a `reqwest::Error`, surfacing attestation failures explicitly.
    pub(crate) fn from_request(e: reqwest::Error) -> Self {
        let chain = format_error_chain(&e);
        if chain.contains("measurement") && chain.contains("allowed") {
            return AppError::Attestation {
                message: "the server's enclave measurement is not in your \
                          trusted_measurements list. The running server version \
                          is not trusted by this client."
                    .into(),
            };
        }
        if chain.contains("fingerprint") && chain.contains("mismatch") {
            return AppError::Attestation {
                message: "TLS certificate does not match the attested enclave".into(),
            };
        }
        if chain.contains("attestation") {
            return AppError::Attestation {
                message: format!("could not verify enclave attestation: {chain}"),
            };
        }
        AppError::Network {
            message: format!("request failed: {chain}"),
        }
    }

    pub(crate) fn db(e: impl std::fmt::Display) -> Self {
        AppError::Database {
            message: e.to_string(),
        }
    }
}

fn format_error_chain(e: &reqwest::Error) -> String {
    let mut chain = format!("{e}");
    let mut source = std::error::Error::source(e);
    while let Some(err) = source {
        use std::fmt::Write;
        let _ = write!(chain, ": {err}");
        source = err.source();
    }
    chain
}
