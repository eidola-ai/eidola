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

use crate::auth::GatewayToken;

/// Environment variable names.
pub mod env {
    /// The one model id this node serves.
    pub const MODEL_ID: &str = "EIDOLA_ENGINE_MODEL_ID";
    /// Directory holding the model's weights and chat artifacts.
    pub const WEIGHTS_DIR: &str = "EIDOLA_ENGINE_WEIGHTS_DIR";
    /// The weights hash the directory must have (64 hex digits).
    pub const WEIGHTS_SHA256: &str = "EIDOLA_ENGINE_WEIGHTS_SHA256";
    /// `verified-readonly` (production: a read-only, kernel-verified mount) or
    /// `dev-writable`.
    pub const WEIGHTS_STORAGE: &str = "EIDOLA_ENGINE_WEIGHTS_STORAGE";
    /// The gateway's bearer token (secret).
    pub const GATEWAY_TOKEN: &str = "GATEWAY_TOKEN";
    /// Argon2id hash of the gateway token (measured).
    pub const GATEWAY_TOKEN_HASH: &str = "GATEWAY_TOKEN_HASH";
    /// `cpu` (or `cuda` in a build with the `cuda` feature).
    pub const EXECUTOR: &str = "EIDOLA_ENGINE_EXECUTOR";
    /// Listen address, `host:port`.
    pub const BIND_ADDR: &str = "EIDOLA_ENGINE_BIND_ADDR";
    /// Positions per KV block.
    pub const KV_BLOCK_SIZE: &str = "EIDOLA_ENGINE_KV_BLOCK_SIZE";
    /// Physical KV blocks per group, including the reserved null block (`cpu` only).
    pub const KV_BLOCKS: &str = "EIDOLA_ENGINE_KV_BLOCKS";
    /// Device memory for the KV pools, in bytes; the per-group block counts are derived
    /// from it (`cuda` only).
    pub const KV_DEVICE_BYTES: &str = "EIDOLA_ENGINE_KV_DEVICE_BYTES";
    /// The kernel build output the CUDA executor loads its images from; every image is
    /// checked against the compiled-in kernel manifest (`cuda` only).
    pub const KERNELS_DIR: &str = "EIDOLA_ENGINE_KERNELS_DIR";
    /// Whether the CUDA executor replays captured graphs for decode steps: `on` or `off`
    /// (`cuda` only).
    pub const CUDA_GRAPHS: &str = "EIDOLA_ENGINE_CUDA_GRAPHS";
    /// Longest sequence (prompt plus completion), in tokens.
    pub const MAX_MODEL_LEN: &str = "EIDOLA_ENGINE_MAX_MODEL_LEN";
    /// Sequences per step (and per-sequence state slots).
    pub const MAX_SEQS: &str = "EIDOLA_ENGINE_MAX_SEQS";
    /// Query tokens per step.
    pub const MAX_BATCHED_TOKENS: &str = "EIDOLA_ENGINE_MAX_BATCHED_TOKENS";
    /// Largest prefill chunk for one sequence in one step.
    pub const MAX_PREFILL_CHUNK: &str = "EIDOLA_ENGINE_MAX_PREFILL_CHUNK";
    /// Speculative draft width `k` (MTP depths); 0 disables speculation.
    pub const DRAFT_TOKENS: &str = "EIDOLA_ENGINE_DRAFT_TOKENS";
    /// Requests admitted to the engine at once (running plus queued).
    pub const MAX_REQUESTS: &str = "EIDOLA_ENGINE_MAX_REQUESTS";
    /// Whether finished keyed requests' KV is kept for reuse (`true` / `false`).
    pub const PREFIX_CACHE: &str = "EIDOLA_ENGINE_PREFIX_CACHE";
    /// Prefix-cache idle lifetime, seconds.
    pub const CACHE_IDLE_TTL_SECS: &str = "EIDOLA_ENGINE_CACHE_IDLE_TTL_SECS";
    /// Prefix-cache maximum age, seconds.
    pub const CACHE_MAX_AGE_SECS: &str = "EIDOLA_ENGINE_CACHE_MAX_AGE_SECS";
}

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

    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "cpu" => Ok(ExecutorKind::Cpu),
            #[cfg(feature = "cuda")]
            "cuda" => Ok(ExecutorKind::Cuda),
            #[cfg(not(feature = "cuda"))]
            "cuda" => {
                Err("this build has no cuda executor (built without the `cuda` feature)".into())
            }
            _ => Err("expected `cpu` or `cuda`".into()),
        }
    }
}

/// What backs the weights directory, as the measured configuration declares it.
///
/// The weights hash is checked once, at boot; the shards stay memory-mapped and routed
/// experts are read from them on later forwards. What binds those later reads to the
/// verified bytes is the storage, not this process: in production the directory is a
/// dm-verity volume, mounted read-only, whose every read the kernel checks against the
/// root hash in the measured configuration. This setting makes that assumption explicit
/// and checked rather than implicit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightsStorage {
    /// Production. Boot refuses, before opening any weights file, unless every weights
    /// file is on read-only storage: a read-only mount everywhere, and on Linux also a
    /// read-only superblock (`crate::storage`).
    VerifiedReadonly,
    /// Development. Boots on any filesystem; reported by `/v1/engine/info` and `/healthz`
    /// so it can never be mistaken for production. A gateway pins the production
    /// configuration, so it never routes to a node configured this way.
    DevWritable,
}

impl WeightsStorage {
    pub fn as_str(self) -> &'static str {
        match self {
            WeightsStorage::VerifiedReadonly => "verified-readonly",
            WeightsStorage::DevWritable => "dev-writable",
        }
    }
}

/// KV memory and scheduler sizing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sizing {
    pub kv_block_size: u32,
    pub max_model_len: u32,
    pub max_seqs: u32,
    pub max_batched_tokens: u32,
    pub max_prefill_chunk: u32,
    pub draft_tokens: u32,
    pub max_requests: u32,
}

/// Prefix-cache lifetime policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheConfig {
    pub enabled: bool,
    pub idle_ttl_secs: u64,
    pub max_age_secs: u64,
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
        let get = |name: &str| -> Result<String, ConfigError> {
            lookup(name)
                .filter(|v| !v.is_empty())
                .ok_or_else(|| ConfigError(format!("{name} is not set")))
        };
        let positive = |name: &str| -> Result<u32, ConfigError> {
            let v = get(name)?;
            match v.parse::<u32>() {
                Ok(n) if n > 0 => Ok(n),
                _ => Err(ConfigError(format!("{name} must be a positive integer"))),
            }
        };
        let non_negative = |name: &str| -> Result<u32, ConfigError> {
            get(name)?
                .parse::<u32>()
                .map_err(|_| ConfigError(format!("{name} must be a non-negative integer")))
        };
        let seconds = |name: &str| -> Result<u64, ConfigError> {
            match get(name)?.parse::<u64>() {
                Ok(n) if n > 0 && n <= u64::MAX / 1000 => Ok(n),
                _ => Err(ConfigError(format!(
                    "{name} must be a positive number of seconds"
                ))),
            }
        };

        let model_id = get(env::MODEL_ID)?;
        if model_id
            .chars()
            .any(|c| c.is_control() || c.is_whitespace())
        {
            return Err(ConfigError(format!(
                "{} must not contain whitespace or control characters",
                env::MODEL_ID
            )));
        }
        let weights_dir = PathBuf::from(get(env::WEIGHTS_DIR)?);
        let expected_weights_sha256 = get(env::WEIGHTS_SHA256)?.to_ascii_lowercase();
        if !is_sha256_hex(&expected_weights_sha256) {
            return Err(ConfigError(format!(
                "{} must be 64 hexadecimal digits",
                env::WEIGHTS_SHA256
            )));
        }
        let weights_storage = match get(env::WEIGHTS_STORAGE)?.as_str() {
            "verified-readonly" => WeightsStorage::VerifiedReadonly,
            "dev-writable" => WeightsStorage::DevWritable,
            _ => {
                return Err(ConfigError(format!(
                    "{} must be `verified-readonly` or `dev-writable`",
                    env::WEIGHTS_STORAGE
                )));
            }
        };
        let token = get(env::GATEWAY_TOKEN)?;
        let token_hash = get(env::GATEWAY_TOKEN_HASH)?;
        let gateway_token = GatewayToken::verify(token, &token_hash).map_err(ConfigError)?;

        let kind = ExecutorKind::parse(&get(env::EXECUTOR)?)
            .map_err(|e| ConfigError(format!("{}: {e}", env::EXECUTOR)))?;
        // Each executor's own settings are required with it and refused without it.
        let refuse_unless = |name: &str, executor: &str| -> Result<(), ConfigError> {
            match lookup(name) {
                Some(v) if !v.is_empty() => Err(ConfigError(format!(
                    "{name} applies only to the {executor} executor"
                ))),
                _ => Ok(()),
            }
        };
        let executor = match kind {
            ExecutorKind::Cpu => {
                refuse_unless(env::KV_DEVICE_BYTES, "cuda")?;
                refuse_unless(env::KERNELS_DIR, "cuda")?;
                refuse_unless(env::CUDA_GRAPHS, "cuda")?;
                let kv_blocks = positive(env::KV_BLOCKS)?;
                if kv_blocks < 2 {
                    return Err(ConfigError(format!(
                        "{} must be at least 2 (block 0 is reserved)",
                        env::KV_BLOCKS
                    )));
                }
                ExecutorConfig::Cpu { kv_blocks }
            }
            #[cfg(feature = "cuda")]
            ExecutorKind::Cuda => {
                refuse_unless(env::KV_BLOCKS, "cpu")?;
                let kv_device_bytes = match get(env::KV_DEVICE_BYTES)?.parse::<u64>() {
                    Ok(n) if n > 0 => n,
                    _ => {
                        return Err(ConfigError(format!(
                            "{} must be a positive number of bytes",
                            env::KV_DEVICE_BYTES
                        )));
                    }
                };
                let graphs = eidola_engine_cuda::CudaGraphs::parse(&get(env::CUDA_GRAPHS)?)
                    .map_err(|e| ConfigError(format!("{}: {e}", env::CUDA_GRAPHS)))?;
                ExecutorConfig::Cuda {
                    kernels_dir: PathBuf::from(get(env::KERNELS_DIR)?),
                    kv_device_bytes,
                    graphs,
                }
            }
        };

        let bind_addr = get(env::BIND_ADDR)?
            .parse::<SocketAddr>()
            .map_err(|_| ConfigError(format!("{} must be host:port", env::BIND_ADDR)))?;

        let sizing = Sizing {
            kv_block_size: positive(env::KV_BLOCK_SIZE)?,
            max_model_len: positive(env::MAX_MODEL_LEN)?,
            max_seqs: positive(env::MAX_SEQS)?,
            max_batched_tokens: positive(env::MAX_BATCHED_TOKENS)?,
            max_prefill_chunk: positive(env::MAX_PREFILL_CHUNK)?,
            draft_tokens: non_negative(env::DRAFT_TOKENS)?,
            max_requests: positive(env::MAX_REQUESTS)?,
        };
        let enabled = match get(env::PREFIX_CACHE)?.as_str() {
            "true" => true,
            "false" => false,
            _ => {
                return Err(ConfigError(format!(
                    "{} must be `true` or `false`",
                    env::PREFIX_CACHE
                )));
            }
        };
        let cache = CacheConfig {
            enabled,
            idle_ttl_secs: seconds(env::CACHE_IDLE_TTL_SECS)?,
            max_age_secs: seconds(env::CACHE_MAX_AGE_SECS)?,
        };
        if cache.idle_ttl_secs > cache.max_age_secs {
            return Err(ConfigError(format!(
                "{} must not exceed {}",
                env::CACHE_IDLE_TTL_SECS,
                env::CACHE_MAX_AGE_SECS
            )));
        }

        Ok(Config {
            model_id,
            weights_dir,
            expected_weights_sha256,
            weights_storage,
            gateway_token,
            executor,
            bind_addr,
            sizing,
            cache,
        })
    }
}

/// Whether `s` is 64 lowercase hex digits.
pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
