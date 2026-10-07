//! Error type.

/// Errors from loading chat artifacts or rendering a prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatError {
    /// A model file could not be read.
    Io { path: String, message: String },
    /// A model artifact's hash is not one this crate has been verified against.
    UnpinnedArtifact { what: &'static str, sha256: String },
    /// A model artifact is malformed.
    InvalidArtifact(String),
    /// The request lies outside the accepted chat-input domain.
    InvalidInput(String),
    /// Template evaluation failed (Jinja raised an error).
    Template(String),
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatError::Io { path, message } => write!(f, "cannot read {path}: {message}"),
            ChatError::UnpinnedArtifact { what, sha256 } => {
                write!(f, "{what} with sha256 {sha256} is not pinned")
            }
            ChatError::InvalidArtifact(m) => write!(f, "invalid model artifact: {m}"),
            ChatError::InvalidInput(m) => write!(f, "invalid chat input: {m}"),
            ChatError::Template(m) => write!(f, "chat template error: {m}"),
        }
    }
}

impl std::error::Error for ChatError {}
