use std::path::PathBuf;

/// Everything that can go wrong between a checkpoint on disk and a forward pass.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("i/o error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid JSON in {what}: {source}")]
    Json {
        what: String,
        #[source]
        source: serde_json::Error,
    },

    /// The config asks for something this implementation does not do. Raised
    /// instead of silently computing a different model.
    #[error("unsupported model config: {0}")]
    UnsupportedConfig(String),

    /// The config is internally inconsistent.
    #[error("invalid model config: {0}")]
    InvalidConfig(String),

    #[error("malformed safetensors file {path}: {reason}")]
    Safetensors { path: PathBuf, reason: String },

    #[error("tensor {0} is not present in the loaded weight files")]
    MissingTensor(String),

    #[error("tensor {name}: {reason}")]
    TensorLayout { name: String, reason: String },

    #[error("integrity check failed for {file}: expected sha256 {expected}, got {actual}")]
    Integrity {
        file: String,
        expected: String,
        actual: String,
    },

    #[error("integrity manifest names {0}, which is not among the loaded weight files")]
    IntegrityUnknownFile(String),

    #[error("invalid input: {0}")]
    Input(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn layout(name: &str, reason: impl Into<String>) -> Self {
        Error::TensorLayout {
            name: name.to_string(),
            reason: reason.into(),
        }
    }
}
