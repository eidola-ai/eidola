//! The node's configuration, read once from the environment at boot.
//!
//! Every value here would sit in the node's measured configuration, so none has a
//! default: a missing or malformed variable refuses the boot rather than letting code
//! choose a value the measurement does not show. What the node derives (the CUDA
//! executor's per-group KV block counts) it derives from measured values alone, by a
//! fixed rule, never from the device it finds. The one secret (the gateway token) is
//! delivered as an environment variable alongside its Argon2id hash, which is the
//! measured half (see [`crate::auth`]).

use std::net::SocketAddr;
use std::path::PathBuf;

use eidola_common::engine_deployment::ExecutorSettings;

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
    /// The CUDA executor (`eidola-engine-cuda`). Parses only in a build with the `cuda`
    /// feature.
    #[cfg(feature = "cuda")]
    Cuda,
}

/// The executor and the settings only it takes. A setting of the other executor is
/// refused rather than ignored, so the measured configuration never shows a value that
/// does nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutorConfig {
    Cpu {
        /// Physical blocks per KV group, including the null block.
        kv_blocks: u32,
    },
    #[cfg(feature = "cuda")]
    Cuda {
        /// The kernel build output (images checked against the compiled-in manifest).
        kernels_dir: PathBuf,
        /// Device memory for the KV pools; the node derives the per-group block counts
        /// from it (`crate::cuda::derive_kv_blocks`).
        kv_device_bytes: u64,
        /// Whether decode steps replay captured graphs.
        graphs: eidola_engine_cuda::CudaGraphs,
    },
}

impl ExecutorConfig {
    pub fn kind(&self) -> ExecutorKind {
        match self {
            ExecutorConfig::Cpu { .. } => ExecutorKind::Cpu,
            #[cfg(feature = "cuda")]
            ExecutorConfig::Cuda { .. } => ExecutorKind::Cuda,
        }
    }
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
    pub executor: ExecutorConfig,
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
        // A build without the CUDA executor says so, rather than naming whichever cuda
        // setting it would otherwise find missing.
        #[cfg(not(feature = "cuda"))]
        if lookup(env::EXECUTOR).as_deref() == Some("cuda") {
            return Err(no_cuda_executor());
        }
        let measured = eidola_common::engine_deployment::parse_measured(&lookup)
            .map_err(|e| ConfigError(e.0))?;
        let token = lookup(env::GATEWAY_TOKEN)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| ConfigError(format!("{} is not set", env::GATEWAY_TOKEN)))?;
        let gateway_token =
            GatewayToken::verify(token, &measured.gateway_token_hash).map_err(ConfigError)?;
        let executor = match measured.executor {
            ExecutorSettings::Cpu { kv_blocks } => ExecutorConfig::Cpu { kv_blocks },
            #[cfg(feature = "cuda")]
            ExecutorSettings::Cuda {
                kernels_dir,
                kv_device_bytes,
                graphs,
            } => ExecutorConfig::Cuda {
                kernels_dir: PathBuf::from(kernels_dir),
                kv_device_bytes,
                graphs: match graphs {
                    eidola_common::engine_deployment::CudaGraphs::On => {
                        eidola_engine_cuda::CudaGraphs::On
                    }
                    eidola_common::engine_deployment::CudaGraphs::Off => {
                        eidola_engine_cuda::CudaGraphs::Off
                    }
                },
            },
            #[cfg(not(feature = "cuda"))]
            ExecutorSettings::Cuda { .. } => return Err(no_cuda_executor()),
        };

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

/// The refusal of a build without the CUDA executor.
#[cfg(not(feature = "cuda"))]
fn no_cuda_executor() -> ConfigError {
    ConfigError(format!(
        "{}: this build has no cuda executor (built without the `cuda` feature)",
        env::EXECUTOR
    ))
}
