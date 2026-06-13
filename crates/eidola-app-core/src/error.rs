/// Errors returned by app-core operations.
///
/// Each variant maps to a distinct failure mode so callers (CLI, GUI) can
/// display appropriate feedback without parsing error strings.
#[derive(Debug, Clone, thiserror::Error)]
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

    /// A chat failed *after* its space was persisted (or, for an existing
    /// space, after the user's turn was committed). Carries the `space_id` so
    /// a blank window's `Space` entity (id=`None` until now) can learn its
    /// persisted id even though no [`ChatResult`] was produced — closing the
    /// id-adoption gap on the failure path.
    ///
    /// `Display` defers to `source` so user-facing messages never regress; the
    /// wrapper is purely a carrier for the id. Errors raised *before* the space
    /// is known/persisted (config, [`AppError::NoAccount`],
    /// [`AppError::InsufficientBalance`]) stay unwrapped.
    #[error("{source}")]
    ChatFailed {
        space_id: String,
        source: Box<AppError>,
    },
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

    /// Unwrap any [`AppError::ChatFailed`] wrapper(s), returning the underlying
    /// error. Callers that route on error *variant* (CLI hint matching, the
    /// GUI's onboarding routing) must look through the wrapper via this helper
    /// so a wrapped `NoAccount` / `InsufficientBalance` still routes correctly.
    /// A non-wrapped error is returned as-is.
    pub fn root(&self) -> &AppError {
        let mut err = self;
        while let AppError::ChatFailed { source, .. } = err {
            err = source;
        }
        err
    }

    /// If this is a [`AppError::ChatFailed`] wrapper, the persisted space id it
    /// carries; otherwise `None`. Lets a blank `Space` adopt its id on failure.
    pub fn chat_space_id(&self) -> Option<&str> {
        match self {
            AppError::ChatFailed { space_id, .. } => Some(space_id),
            _ => None,
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

#[cfg(test)]
mod tests {
    use super::AppError;

    #[test]
    fn chat_failed_display_defers_to_source() {
        let inner = AppError::Server {
            status: 500,
            message: "boom".into(),
        };
        let wrapped = AppError::ChatFailed {
            space_id: "space-1".into(),
            source: Box::new(inner.clone()),
        };
        // The wrapper's Display must be identical to its source's so user-facing
        // messages never regress when an error is wrapped for id-adoption.
        assert_eq!(wrapped.to_string(), inner.to_string());
    }

    #[test]
    fn root_unwraps_through_nested_wrappers() {
        let leaf = AppError::InsufficientBalance {
            available: 1,
            required: 2,
        };
        let wrapped = AppError::ChatFailed {
            space_id: "space-1".into(),
            // Nesting should be collapsed entirely by `root`.
            source: Box::new(AppError::ChatFailed {
                space_id: "space-1".into(),
                source: Box::new(leaf.clone()),
            }),
        };
        assert!(matches!(
            wrapped.root(),
            AppError::InsufficientBalance {
                available: 1,
                required: 2
            }
        ));
        // A non-wrapped error returns itself.
        assert!(matches!(leaf.root(), AppError::InsufficientBalance { .. }));
    }

    #[test]
    fn chat_space_id_only_on_wrapper() {
        let wrapped = AppError::ChatFailed {
            space_id: "space-7".into(),
            source: Box::new(AppError::NoAccount),
        };
        assert_eq!(wrapped.chat_space_id(), Some("space-7"));
        assert_eq!(AppError::NoAccount.chat_space_id(), None);
    }
}
