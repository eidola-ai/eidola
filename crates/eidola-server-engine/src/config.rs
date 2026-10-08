//! The node's configuration, read once from the environment at boot.
//!
//! Every value here would sit in the node's measured configuration, so none has a
//! default: a missing or malformed variable refuses the boot rather than letting code
//! choose a value the measurement does not show. The one secret (the gateway token) is
//! delivered as an environment variable alongside its Argon2id hash, which is the
//! measured half (see [`crate::auth`]).

use std::net::SocketAddr;
use std::path::PathBuf;

use eidola_common::engine_deployment::Executor;

use crate::auth::GatewayToken;

/// Environment variable names, and the measured-configuration types: the grammar lives in
/// `eidola_common::engine_deployment`, which the gateway's build also runs every pinned
/// deployment's env through, so a pin can only name a configuration this node boots with.
pub use eidola_common::engine_deployment::{CacheConfig, Sizing, WeightsStorage, env};

/// Which executor runs the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutorKind {
    /// The f32 reference executor (`eidola-engine-cpu`).
    Cpu,
    /// The CUDA executor. Parses only in a build with the `cuda` feature; the executor
    /// itself lands separately, and until then boot refuses it.
    #[cfg(feature = "cuda")]
    Cuda,
}

impl ExecutorKind {
    /// The configuration spelling, also reported by the info endpoint.
    pub fn as_str(self) -> &'static str {
        match self {
            ExecutorKind::Cpu => "cpu",
            #[cfg(feature = "cuda")]
            ExecutorKind::Cuda => "cuda",
        }
    }

    fn from_measured(executor: Executor) -> Result<Self, String> {
        match executor {
            Executor::Cpu => Ok(ExecutorKind::Cpu),
            #[cfg(feature = "cuda")]
            Executor::Cuda => Ok(ExecutorKind::Cuda),
            #[cfg(not(feature = "cuda"))]
            Executor::Cuda => {
                Err("this build has no cuda executor (built without the `cuda` feature)".into())
            }
        }
    }
}

/// Everything the node is configured with.
#[derive(Debug)]
pub struct Config {
    pub model_id: String,
    pub weights_dir: PathBuf,
    /// Lowercase hex.
    pub expected_weights_sha256: String,
    pub weights_storage: WeightsStorage,
    pub gateway_token: GatewayToken,
    pub executor: ExecutorKind,
    pub bind_addr: SocketAddr,
    pub sizing: Sizing,
    pub cache: CacheConfig,
}

/// A refused configuration. Names the variable and the problem, never its value (one of
/// them is a secret).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "configuration refused: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

impl Config {
    /// Reads the process environment.
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    /// Reads configuration through `lookup` (the environment, or a map in tests). An
    /// empty value counts as missing.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, ConfigError> {
        let measured = eidola_common::engine_deployment::parse_measured(&lookup)
            .map_err(|e| ConfigError(e.0))?;
        let token = lookup(env::GATEWAY_TOKEN)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| ConfigError(format!("{} is not set", env::GATEWAY_TOKEN)))?;
        let gateway_token =
            GatewayToken::verify(token, &measured.gateway_token_hash).map_err(ConfigError)?;
        let executor = ExecutorKind::from_measured(measured.executor)
            .map_err(|e| ConfigError(format!("{}: {e}", env::EXECUTOR)))?;

        Ok(Config {
            model_id: measured.model_id,
            weights_dir: PathBuf::from(measured.weights_dir),
            expected_weights_sha256: measured.weights_sha256,
            weights_storage: measured.weights_storage,
            gateway_token,
            executor,
            bind_addr: measured.bind_addr,
            sizing: measured.sizing,
            cache: measured.cache,
        })
    }
}

/// Whether `s` is 64 lowercase hex digits.
pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
