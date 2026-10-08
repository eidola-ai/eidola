//! The grammar of an engine deployment's measured configuration, shared by
//! everything that reads one.
//!
//! An Eidola-hosted engine deployment is a `tinfoil-config.yml` whose
//! `eidola-server-engine` container boots from its environment. Three parties
//! read that config and must agree on what it means, so each rule here is the
//! one function they all call:
//!
//! - the node (`eidola-server-engine`), which boots from the values or
//!   refuses to;
//! - the gateway's build (`engine_trust/manifest.rs`), which refuses a pin
//!   for a config the node would refuse, or that it reads differently;
//! - `measure-enclave`, which turns the config into that pin.
//!
//! The Argon2id rule, and with it [`parse_measured`] (the whole measured
//! environment), need the `argon2` crate and sit behind this crate's
//! `argon2` feature: only the node and the gateway enable it, and both
//! already depend on `argon2` themselves (see the crate docs' dependency
//! rule).

/// The node's environment variables. Every one is required (an empty value
/// counts as missing) except where noted.
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
    /// The gateway's bearer token. Secret, so not part of the measured
    /// configuration [`super::parse_measured`] reads.
    pub const GATEWAY_TOKEN: &str = "GATEWAY_TOKEN";
    /// Argon2id hash of the gateway token (measured).
    pub const GATEWAY_TOKEN_HASH: &str = "GATEWAY_TOKEN_HASH";
    /// `cpu` or `cuda`.
    pub const EXECUTOR: &str = "EIDOLA_ENGINE_EXECUTOR";
    /// Listen address, `host:port`.
    pub const BIND_ADDR: &str = "EIDOLA_ENGINE_BIND_ADDR";
    /// Positions per KV block.
    pub const KV_BLOCK_SIZE: &str = "EIDOLA_ENGINE_KV_BLOCK_SIZE";
    /// Physical KV blocks per group, including the reserved null block.
    pub const KV_BLOCKS: &str = "EIDOLA_ENGINE_KV_BLOCKS";
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

    /// Every variable of the measured environment: exactly the names
    /// [`super::parse_measured`] reads. A deployment's env sets these and
    /// nothing else, so no variable a dependency happens to consume
    /// (`TOKIO_WORKER_THREADS`, `RUST_MIN_STACK`, `MALLOC_*`, `LD_*`,
    /// `CUDA_*`, …) can reach the node through its measured config.
    pub const MEASURED: &[&str] = &[
        MODEL_ID,
        WEIGHTS_DIR,
        WEIGHTS_SHA256,
        WEIGHTS_STORAGE,
        GATEWAY_TOKEN_HASH,
        EXECUTOR,
        BIND_ADDR,
        KV_BLOCK_SIZE,
        KV_BLOCKS,
        MAX_MODEL_LEN,
        MAX_SEQS,
        MAX_BATCHED_TOKENS,
        MAX_PREFILL_CHUNK,
        DRAFT_TOKENS,
        MAX_REQUESTS,
        PREFIX_CACHE,
        CACHE_IDLE_TTL_SECS,
        CACHE_MAX_AGE_SECS,
    ];
}

/// Whether a deployment's measured env sets only [`env::MEASURED`] names.
pub fn check_env_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<(), String> {
    match names.into_iter().find(|name| !env::MEASURED.contains(name)) {
        None => Ok(()),
        Some(name) => Err(format!(
            "the engine container's env may not set {name:?}; it sets exactly the node's \
             measured variables"
        )),
    }
}

/// Caps on the node's allocation-driving sizes. The engine serves one model
/// family, MiMo-V2.6-Flash (`eidola-engine-cuda`'s `support::check_supported`
/// refuses every other), so each cap follows from that model and the CUDA
/// executor's own limits, generous but finite.
pub mod caps {
    /// The model's context window (`support::MAX_POSITIONS`, 2^20).
    pub const MAX_MODEL_LEN: u32 = 1 << 20;
    /// Positions per KV block: a block is the attention kernel's page and the
    /// prefix cache's unit of reuse; beyond 1,024 positions paging stops
    /// paying, and the page stride (32-bit) stays far from overflow.
    pub const KV_BLOCK_SIZE: u32 = 1024;
    /// Blocks per KV group: the device block tables are `i32`
    /// (`kv::GroupGeometry::validate`). The device bytes they cost are held
    /// to the attached GPUs separately ([`super::check_resources`]).
    pub const KV_BLOCKS: u32 = i32::MAX as u32;
    /// Sequences per step, which is also the state slots: the sampler's
    /// scratch is rows × the 152,576-entry vocabulary × 4 bytes (625 MB at
    /// 1,024), and each slot holds a block table.
    pub const MAX_SEQS: u32 = 1024;
    /// Query tokens per step: the forward's activation scratch scales with it
    /// (tokens × the 16,384-wide dense intermediate × 2 bytes, 2 GiB at
    /// 65,536).
    pub const MAX_BATCHED_TOKENS: u32 = 65_536;
    /// Largest prefill chunk: bounded by a step's tokens, so the same.
    pub const MAX_PREFILL_CHUNK: u32 = 65_536;
    /// Speculative draft width: MiMo's MTP heads are a handful; the node also
    /// refuses more than the loaded model has.
    pub const DRAFT_TOKENS: u32 = 8;
    /// Admission slots: each may hold a request body of up to
    /// `engine_protocol::MAX_REQUEST_BODY_BYTES` and its parse, held to the
    /// VM's memory separately ([`super::check_resources`]).
    pub const MAX_REQUESTS: u32 = 4096;
}

/// KV-cache bytes per position, every layer of MiMo-V2.6-Flash
/// (`eidola-engine-cuda` `support`): 9 global layers of 4 KV heads and 39
/// sliding layers of 8, each head 192 K + 128 V dims, in bf16.
pub const KV_BYTES_PER_POSITION: u64 = (9 * 4 + 39 * 8) * (192 + 128) * 2;

/// Largest memory of one confidential GPU the platform offers, in bytes: the
/// largest NVIDIA-CC part's HBM, rounded up (288 GiB).
pub const MAX_GPU_MEMORY_BYTES: u64 = 288 << 30;

/// Worst-case bytes of `serde_json::Value` tree per byte of JSON text, for
/// the node's strict parse (`api::parse_request`): the densest text is the
/// smallest non-empty object, `{"":0}` (7 bytes), which allocates a B-tree
/// leaf of eleven 24-byte keys and 32-byte values plus its header, about
/// 640 bytes (≈ 92×), rounded up.
pub const STRICT_TREE_PER_TEXT_BYTE: u64 = 96;

/// Worst-case bytes of the chat crate's order-preserving tree per byte of
/// text (`Json`, 32 bytes a node): an array element `0,` (2 bytes) is one node
/// (16×), and a vector's capacity may be twice its length (32×).
pub const ORDERED_TREE_PER_TEXT_BYTE: u64 = 32;

/// Worst-case bytes a prepared request adds per byte of its body
/// (`pipeline::prepare`): the rendered prompt (at most twice the message
/// text, the template's per-message markup counted) and its tokens (a `u32`
/// per prompt byte at worst, so four bytes each), 2 + 8.
pub const PREPARED_PER_BODY_BYTE: u64 = 10;

/// The node process's own fixed footprint beside its pools (runtime, engine
/// state, the model's host-side metadata): 4 GiB.
pub const PROCESS_HEADROOM_BYTES: u64 = 4 << 30;

/// The host memory a node's configuration commits it to, at worst, from its
/// own code paths (`http::chat`, `api::parse_request`, `pipeline::prepare`):
///
/// - the **read pool**: `MAX_REQUESTS` slots, each a body of up to
///   `MAX_REQUEST_BODY_BYTES` live at once with its strict parse and its
///   order-preserving parse (`parse_request` builds both from the body
///   before the body is dropped): body × (1 + [`STRICT_TREE_PER_TEXT_BYTE`] +
///   [`ORDERED_TREE_PER_TEXT_BYTE`]);
/// - the **admission pool**: `MAX_REQUESTS` admitted requests, each keeping
///   its order-preserving tree while the prompt is rendered and tokenized:
///   body × ([`ORDERED_TREE_PER_TEXT_BYTE`] + [`PREPARED_PER_BODY_BYTE`]).
///   Both pools can be full at once (a slot of each per request in flight);
/// - the block tables: `MAX_SEQS` × ⌈`MAX_MODEL_LEN` / `KV_BLOCK_SIZE`⌉
///   `i32` entries per KV group, two groups;
/// - [`PROCESS_HEADROOM_BYTES`].
pub fn host_memory_bytes(sizing: &Sizing) -> u64 {
    let body = crate::engine_protocol::MAX_REQUEST_BODY_BYTES as u64;
    let requests = u64::from(sizing.max_requests);
    let read_pool = requests * body * (1 + STRICT_TREE_PER_TEXT_BYTE + ORDERED_TREE_PER_TEXT_BYTE);
    let admission_pool = requests * body * (ORDERED_TREE_PER_TEXT_BYTE + PREPARED_PER_BODY_BYTE);
    let blocks_per_seq = u64::from(sizing.max_model_len).div_ceil(u64::from(sizing.kv_block_size));
    let tables = u64::from(sizing.max_seqs) * blocks_per_seq * 4 * 2;
    read_pool + admission_pool + tables + PROCESS_HEADROOM_BYTES
}

/// Whether the deployment's VM and GPUs hold what its configuration
/// allocates, as far as configuration alone fixes it: host memory (`memory`,
/// MiB) holds [`host_memory_bytes`], and GPU memory (`gpus` ×
/// [`MAX_GPU_MEMORY_BYTES`]) holds the KV cache (`KV_BLOCKS` ×
/// `KV_BLOCK_SIZE` × [`KV_BYTES_PER_POSITION`]). The weights' own footprint is
/// the model's and is not estimated here.
pub fn check_resources(sizing: &Sizing, memory_mib: u64, gpus: u64) -> Result<(), String> {
    let host = host_memory_bytes(sizing);
    let host_limit = memory_mib << 20;
    if host > host_limit {
        return Err(format!(
            "the node's request pools, block tables and headroom need {host} bytes, more than \
             the VM's {host_limit} (memory)"
        ));
    }
    let kv = u64::from(sizing.kv_blocks) * u64::from(sizing.kv_block_size) * KV_BYTES_PER_POSITION;
    let device_limit = gpus * MAX_GPU_MEMORY_BYTES;
    if kv > device_limit {
        return Err(format!(
            "the KV cache needs {kv} bytes, more than {gpus} GPUs hold ({device_limit})"
        ));
    }
    Ok(())
}

/// Fewest vCPUs a deployment may give its VM.
pub const MIN_CPUS: u64 = 1;
/// Most vCPUs a deployment may give its VM.
pub const MAX_CPUS: u64 = 256;
/// Smallest VM memory, in MiB.
pub const MIN_MEMORY_MIB: u64 = 8192;
/// Largest VM memory, in MiB.
pub const MAX_MEMORY_MIB: u64 = 2 * 1024 * 1024;

/// Whether the VM's `cpus` and `memory` (MiB) are a shape the platform
/// launches: `cpus` from [`MIN_CPUS`] to [`MAX_CPUS`], and `memory` a power of
/// two from [`MIN_MEMORY_MIB`] (8 GiB) to [`MAX_MEMORY_MIB`] (2 TiB), the sizes
/// the platform provider's deployment accepts. Tinfoil's published checks
/// (`measure-image-action`) require only positive values, so these bounds are
/// ours; the same rule holds for every `cvm-version` an engine pins.
pub fn check_vm_resources(cpus: u64, memory_mib: u64) -> Result<(), String> {
    if !(MIN_CPUS..=MAX_CPUS).contains(&cpus) {
        return Err(format!(
            "cpus must be from {MIN_CPUS} to {MAX_CPUS} (got {cpus})"
        ));
    }
    if !memory_mib.is_power_of_two() || !(MIN_MEMORY_MIB..=MAX_MEMORY_MIB).contains(&memory_mib) {
        return Err(format!(
            "memory must be a power of two from {MIN_MEMORY_MIB} to {MAX_MEMORY_MIB} MiB (got {memory_mib})"
        ));
    }
    Ok(())
}

/// Which executor the configuration selects. Whether a given node build can
/// run it is the node's concern (a `cuda` node build), not the grammar's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Executor {
    /// The f32 reference executor.
    Cpu,
    /// The CUDA executor.
    Cuda,
}

/// What backs the weights directory, as the measured configuration declares
/// it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightsStorage {
    /// Production: every weights file on read-only, kernel-verified storage
    /// (checked by the node before any weights file is opened).
    VerifiedReadonly,
    /// Development: any filesystem, reported as such by the node.
    DevWritable,
}

impl WeightsStorage {
    /// The configuration spelling.
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
    pub kv_blocks: u32,
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

/// A node's measured configuration: every variable of its environment except
/// the secret gateway token, parsed and validated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasuredConfig {
    pub model_id: String,
    pub weights_dir: String,
    /// Lowercase hex.
    pub weights_sha256: String,
    pub weights_storage: WeightsStorage,
    /// A validated Argon2id PHC string.
    pub gateway_token_hash: String,
    pub executor: Executor,
    pub bind_addr: core::net::SocketAddr,
    pub sizing: Sizing,
    pub cache: CacheConfig,
}

/// A refused configuration. Names the variable and the problem, never its
/// value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl core::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "configuration refused: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

/// Parse a node's measured configuration through `lookup` (its environment,
/// or a deployment config's env list). **This is the node's boot grammar**:
/// the node calls it and then verifies its secret token against the hash, so
/// a configuration this accepts is one the node boots with, up to checks
/// against its weights (model length, MTP depth, executor availability) that
/// need the model itself. The gateway's build check runs every pinned
/// deployment's whole env through it.
#[cfg(feature = "argon2")]
pub fn parse_measured(
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<MeasuredConfig, ConfigError> {
    let get = |name: &str| -> Result<String, ConfigError> {
        lookup(name)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| ConfigError(format!("{name} is not set")))
    };
    let positive = |name: &str| -> Result<u32, ConfigError> {
        match get(name)?.parse::<u32>() {
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
        parse_cache_seconds(&get(name)?)
            .ok_or_else(|| ConfigError(format!("{name} must be a positive number of seconds")))
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
    let weights_dir = get(env::WEIGHTS_DIR)?;
    let weights_sha256 = get(env::WEIGHTS_SHA256)?.to_ascii_lowercase();
    if !(weights_sha256.len() == 64
        && weights_sha256
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
    {
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
    let gateway_token_hash = get(env::GATEWAY_TOKEN_HASH)?;
    parse_gateway_token_hash(&gateway_token_hash)
        .map_err(|e| ConfigError(format!("{}: {e}", env::GATEWAY_TOKEN_HASH)))?;
    let executor = match get(env::EXECUTOR)?.as_str() {
        "cpu" => Executor::Cpu,
        "cuda" => Executor::Cuda,
        _ => {
            return Err(ConfigError(format!(
                "{}: expected `cpu` or `cuda`",
                env::EXECUTOR
            )));
        }
    };
    let bind_addr = get(env::BIND_ADDR)?
        .parse::<core::net::SocketAddr>()
        .map_err(|_| ConfigError(format!("{} must be host:port", env::BIND_ADDR)))?;

    let sizing = Sizing {
        kv_block_size: positive(env::KV_BLOCK_SIZE)?,
        kv_blocks: positive(env::KV_BLOCKS)?,
        max_model_len: positive(env::MAX_MODEL_LEN)?,
        max_seqs: positive(env::MAX_SEQS)?,
        max_batched_tokens: positive(env::MAX_BATCHED_TOKENS)?,
        max_prefill_chunk: positive(env::MAX_PREFILL_CHUNK)?,
        draft_tokens: non_negative(env::DRAFT_TOKENS)?,
        max_requests: positive(env::MAX_REQUESTS)?,
    };
    // The scheduler's own boot refusal (`Engine::new`): a decode row costs
    // `1 + draft_tokens` query slots, so a step must hold more than
    // `draft_tokens` of them. The engine's other boot refusals that depend
    // only on these values (a zero seat, prefill chunk or block count) are the
    // positivity rules above; the rest need the model (its position limit and
    // MTP depth) or the node build (its executor).
    // Every allocation-driving value is capped, so a configuration cannot ask
    // the node for an allocation it cannot make (a table of `MAX_SEQS` slots,
    // a scratch of `MAX_BATCHED_TOKENS` rows): refused here, before any
    // weight is loaded. The caps are in [`caps`], each with its reason.
    for (name, value, cap) in [
        (
            env::KV_BLOCK_SIZE,
            sizing.kv_block_size,
            caps::KV_BLOCK_SIZE,
        ),
        (env::KV_BLOCKS, sizing.kv_blocks, caps::KV_BLOCKS),
        (
            env::MAX_MODEL_LEN,
            sizing.max_model_len,
            caps::MAX_MODEL_LEN,
        ),
        (env::MAX_SEQS, sizing.max_seqs, caps::MAX_SEQS),
        (
            env::MAX_BATCHED_TOKENS,
            sizing.max_batched_tokens,
            caps::MAX_BATCHED_TOKENS,
        ),
        (
            env::MAX_PREFILL_CHUNK,
            sizing.max_prefill_chunk,
            caps::MAX_PREFILL_CHUNK,
        ),
        (env::DRAFT_TOKENS, sizing.draft_tokens, caps::DRAFT_TOKENS),
        (env::MAX_REQUESTS, sizing.max_requests, caps::MAX_REQUESTS),
    ] {
        if value > cap {
            return Err(ConfigError(format!("{name} must be at most {cap}")));
        }
    }
    if sizing.max_batched_tokens <= sizing.draft_tokens {
        return Err(ConfigError(format!(
            "{} must exceed {}: a decode row needs 1 + {} query slots",
            env::MAX_BATCHED_TOKENS,
            env::DRAFT_TOKENS,
            env::DRAFT_TOKENS
        )));
    }
    if sizing.kv_blocks < 2 {
        return Err(ConfigError(format!(
            "{} must be at least 2 (block 0 is reserved)",
            env::KV_BLOCKS
        )));
    }

    let enabled = parse_prefix_cache(&get(env::PREFIX_CACHE)?)
        .ok_or_else(|| ConfigError(format!("{} must be `true` or `false`", env::PREFIX_CACHE)))?;
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

    Ok(MeasuredConfig {
        model_id,
        weights_dir,
        weights_sha256,
        weights_storage,
        gateway_token_hash,
        executor,
        bind_addr,
        sizing,
        cache,
    })
}

/// A deployment path component (a model id, a variant):
/// `[a-z0-9][a-z0-9._-]*`. `deploy/engine/<model>/<variant>/` directories
/// are named this way; the gateway's build refuses a pin naming any other,
/// and `measure-enclave` refuses to measure one.
pub fn is_safe_component(s: &str) -> bool {
    let mut bytes = s.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && bytes.all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-')
        })
}

/// Whether the engine container's GPU access matches its executor: the
/// `cuda` executor needs `runtime: nvidia` and every attested GPU
/// (`gpus: all`, so the GPUs the shim attests are the ones the node uses);
/// the `cpu` executor needs neither. `gpus` is the container's selection as
/// written (a number reads as its decimal text).
pub fn check_container_gpu_access(
    executor: Executor,
    runtime: Option<&str>,
    gpus: Option<&str>,
) -> Result<(), String> {
    match (executor, runtime, gpus) {
        (Executor::Cuda, Some("nvidia"), Some("all")) => Ok(()),
        (Executor::Cuda, _, _) => {
            Err("the cuda executor's container needs `runtime: nvidia` and `gpus: all`".into())
        }
        (Executor::Cpu, None, None) => Ok(()),
        (Executor::Cpu, _, _) => {
            Err("the cpu executor's container takes no `runtime` or `gpus`".into())
        }
    }
}

/// Shortest gateway token accepted, in bytes.
pub const GATEWAY_TOKEN_MIN_LEN: usize = 16;
/// Longest gateway token accepted, in bytes.
pub const GATEWAY_TOKEN_MAX_LEN: usize = 1024;

/// The shape of the gateway's bearer token (`GATEWAY_TOKEN`): visible ASCII
/// (no space, no control character), [`GATEWAY_TOKEN_MIN_LEN`] to
/// [`GATEWAY_TOKEN_MAX_LEN`] bytes, so it travels in an `Authorization`
/// header unchanged. The gateway refuses to hold any other token and the node
/// refuses to boot with one.
pub fn is_gateway_token(token: &str) -> bool {
    (GATEWAY_TOKEN_MIN_LEN..=GATEWAY_TOKEN_MAX_LEN).contains(&token.len())
        && token.bytes().all(|b| b.is_ascii_graphic())
}

/// The one secret an engine container takes: everything else it reads is in
/// the measured env. A secret is delivered outside the measurement, so any
/// other one (a log level, say) would change the node's behaviour unmeasured.
pub const SECRETS: &[&str] = &[env::GATEWAY_TOKEN];

/// Whether the engine container's `secrets` are exactly [`SECRETS`].
pub fn check_secrets<'a>(secrets: impl IntoIterator<Item = &'a str>) -> Result<(), String> {
    let secrets: Vec<&str> = secrets.into_iter().collect();
    if secrets == SECRETS {
        Ok(())
    } else {
        Err(format!(
            "the engine container's secrets must be exactly {SECRETS:?}; any other secret \
             reaches the node outside the measurement"
        ))
    }
}

/// Where Tinfoil mounts a model pack granted to a container: read-only, at
/// `/tinfoil/models/<name>`, in the granted containers only (cvmimage
/// `docs/runtime-policy.md`; `ContainerModelsDir` in its boot paths).
pub const MODEL_MOUNT_ROOT: &str = "/tinfoil/models";

/// Whether `mpk` is a modelwrap artifact reference, `rootHash_hashOffset_uuid`
/// (tinfoilsh/modelwrap `modelwrap.go` `ParseRef` at e61ae511): a 64-hex
/// root hash, a decimal hash offset that fits a `u64` (`HashOffsetBytes`), and
/// a lowercase UUID (8-4-4-4-12 hex), joined by exactly two underscores.
pub fn is_artifact_ref(mpk: &str) -> bool {
    let hex = |s: &str, n: usize| {
        s.len() == n && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    };
    let parts: Vec<&str> = mpk.split('_').collect();
    let [root, offset, uuid] = parts.as_slice() else {
        return false;
    };
    let uuid_ok = {
        let groups: Vec<&str> = uuid.split('-').collect();
        groups.len() == 5 && groups.iter().zip([8, 4, 4, 4, 12]).all(|(g, n)| hex(g, n))
    };
    hex(root, 64)
        && !offset.is_empty()
        && offset.bytes().all(|b| b.is_ascii_digit())
        && offset.parse::<u64>().is_ok()
        && uuid_ok
}

/// A model pack as an engine config declares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelPack<'a> {
    pub name: &'a str,
    /// `<owner>/<name>@<revision>`.
    pub repo: Option<&'a str>,
    /// The pinned pack: `<root hash>_<size>_<id>`.
    pub mpk: Option<&'a str>,
}

/// Whether the weights the node reads are the verified model pack: exactly
/// one pack, from `<weights_repo>@<weights_revision>` and pinned by its root
/// hash (`mpk`), granted to the engine container (and only it), with the
/// node's weights directory inside that pack's mount.
pub fn check_weights_pack(
    packs: &[ModelPack<'_>],
    granted: &[&str],
    weights_dir: &str,
    weights_repo: &str,
    weights_revision: &str,
) -> Result<(), String> {
    let [pack] = packs else {
        return Err("an engine deployment declares exactly one model pack, its weights".into());
    };
    let name_ok = {
        let mut bytes = pack.name.bytes();
        bytes.next().is_some_and(|b| b.is_ascii_alphanumeric())
            && pack.name.len() <= 128
            && bytes.all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    };
    if !name_ok {
        return Err(format!(
            "model pack name {:?} is not a valid pack name",
            pack.name
        ));
    }
    let source = format!("{weights_repo}@{weights_revision}");
    if pack.repo != Some(source.as_str()) {
        return Err(format!(
            "the model pack's repo must be {source:?}, the weights' recorded provenance"
        ));
    }
    if !pack.mpk.is_some_and(is_artifact_ref) {
        return Err("the model pack must be pinned by its artifact reference \
             (`mpk: <64 hex root hash>_<offset>_<uuid>`)"
            .into());
    }
    if granted != [pack.name] {
        return Err(format!(
            "the engine container must be granted exactly the weights pack ({:?})",
            pack.name
        ));
    }
    let mount = format!("{MODEL_MOUNT_ROOT}/{}", pack.name);
    let inside = weights_dir == mount
        || weights_dir
            .strip_prefix(&format!("{mount}/"))
            .is_some_and(|rest| {
                rest.split('/')
                    .all(|part| !part.is_empty() && part != "." && part != "..")
            });
    if !inside {
        return Err(format!(
            "EIDOLA_ENGINE_WEIGHTS_DIR must be inside the weights pack's mount, {mount}"
        ));
    }
    Ok(())
}

/// The node routes the shim must expose: the chat route the gateway sends
/// requests to, the info route it checks a node's weights with, and the
/// health route probes read.
pub const NODE_ROUTES: &[&str] = &["/v1/chat/completions", "/v1/engine/info", "/healthz"];

/// Whether a shim path pattern matches a request path, as the shim decides
/// it (cvmimage `tinfoil/cmd/shim/api.go`, `pathMatchesPattern`): an exact
/// path, or a trailing `*` matching on a segment boundary (`/v1/*` matches
/// `/v1/x`; `/v1*` matches `/v1` and `/v1/x`).
pub fn shim_path_matches(pattern: &str, path: &str) -> bool {
    match pattern.strip_suffix('*') {
        None => pattern == path,
        Some(prefix) if prefix.ends_with('/') => path.starts_with(prefix),
        Some(prefix) => {
            path == prefix
                || path
                    .strip_prefix(prefix)
                    .is_some_and(|rest| rest.starts_with('/'))
        }
    }
}

/// Whether the shim's `paths` expose exactly the node's routes: every route
/// in [`NODE_ROUTES`] matched, and every pattern matching at least one of
/// them (a wildcard such as `/*` does). An absent or empty list is refused
/// even though the shim would then forward every path: the exposure should
/// be stated, not defaulted.
pub fn check_shim_paths(paths: &[&str]) -> Result<(), String> {
    if paths.is_empty() {
        return Err("shim.paths must list the node's routes".into());
    }
    for pattern in paths {
        if !pattern.starts_with('/') || !NODE_ROUTES.iter().any(|r| shim_path_matches(pattern, r)) {
            return Err(format!(
                "shim.paths entry {pattern:?} exposes none of the node's routes {NODE_ROUTES:?}"
            ));
        }
    }
    for route in NODE_ROUTES {
        if !paths.iter().any(|p| shim_path_matches(p, route)) {
            return Err(format!("shim.paths must expose {route}"));
        }
    }
    Ok(())
}

/// The GPU counts a confidential NVIDIA deployment may attach (the
/// config's top-level `gpus`): the platform provider's NVIDIA-CC shapes.
pub const CUDA_GPU_COUNTS: &[u64] = &[1, 8];

/// Whether a deployment's GPUs are attested on every handshake.
///
/// The attestation shim collects exactly the config's top-level `gpus` worth
/// of GPU evidence items (Tinfoil's `shimConfig.ExpectedGPUs = config.GPUs`),
/// and a pin's `expected_gpus` requires exactly that many. So:
///
/// - the `cuda` executor needs GPUs, in one of [`CUDA_GPU_COUNTS`], and a pin
///   requiring all of them, or its accelerators would go unattested (or, with
///   no `gpus`, every handshake would fail for want of evidence);
/// - the `cpu` executor attaches none and requires none. A pinned deployment
///   never takes this branch (the deployment check refuses `cpu` first); it
///   stays so the rule is total over the grammar's executors.
///
/// `gpus` and `expected_gpus` are `None` when absent; `0` reads as absent.
pub fn check_gpu_attestation(
    executor: Executor,
    gpus: Option<u64>,
    expected_gpus: Option<u64>,
) -> Result<(), String> {
    let gpus = gpus.filter(|n| *n > 0);
    let expected = expected_gpus.filter(|n| *n > 0);
    match executor {
        Executor::Cuda => match gpus {
            None => Err("the cuda executor needs the config to attach GPUs (`gpus`)".into()),
            Some(n) if !CUDA_GPU_COUNTS.contains(&n) => Err(format!(
                "`gpus: {n}` is not an NVIDIA-CC shape; it must be one of {CUDA_GPU_COUNTS:?}"
            )),
            Some(n) if expected != Some(n) => Err(format!(
                "the config attaches {n} GPUs, so expected_gpus must be {n} (the shim collects \
                 exactly that many evidence items)"
            )),
            Some(_) => Ok(()),
        },
        Executor::Cpu => {
            if gpus.is_some() || expected.is_some() {
                Err(
                    "the cpu executor attaches no GPUs and requires no GPU evidence \
                     (`gpus` and expected_gpus absent or 0)"
                        .into(),
                )
            } else {
                Ok(())
            }
        }
    }
}

#[cfg(feature = "deployment")]
pub mod deployment;
#[cfg(feature = "deployment")]
mod yaml_events;

/// Largest prefix-cache retention bound, in seconds: the engine core keeps
/// retention in milliseconds as a `u64`.
pub const MAX_CACHE_SECONDS: u64 = u64::MAX / 1000;

/// A prefix-cache retention bound (`EIDOLA_ENGINE_CACHE_IDLE_TTL_SECS`,
/// `EIDOLA_ENGINE_CACHE_MAX_AGE_SECS`): a positive whole number of seconds,
/// at most [`MAX_CACHE_SECONDS`].
pub fn parse_cache_seconds(text: &str) -> Option<u64> {
    text.parse::<u64>()
        .ok()
        .filter(|n| *n > 0 && *n <= MAX_CACHE_SECONDS)
}

/// The prefix-cache switch (`EIDOLA_ENGINE_PREFIX_CACHE`): exactly `true` or
/// `false`.
pub fn parse_prefix_cache(text: &str) -> Option<bool> {
    match text {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// The platform release a config's `cvm-version` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CvmVersion<'a> {
    /// The bare version (`0.15.0`; no leading `v`).
    pub version: &'a str,
    /// The release manifest's SHA-256, lowercase hex, when the config pins it
    /// inline (`0.15.0@sha256:<hex>`).
    pub manifest_sha256: Option<&'a str>,
}

/// Why a `cvm-version` value is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CvmVersionError {
    /// The version is empty or starts with `v`.
    NotBare,
    /// Something follows `@` other than `sha256:` and 64 lowercase hex
    /// digits.
    MalformedPin,
}

impl core::fmt::Display for CvmVersionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::NotBare => "cvm-version must be a bare version such as 0.15.0",
            Self::MalformedPin => {
                "malformed cvm-version pin: expected VERSION@sha256:<64 lowercase hex>"
            }
        })
    }
}

/// Parse a config's `cvm-version`: a bare version, optionally followed by
/// exactly one `@sha256:<64 lowercase hex>` release-manifest pin.
pub fn parse_cvm_version(raw: &str) -> Result<CvmVersion<'_>, CvmVersionError> {
    let (version, manifest_sha256) = match raw.split_once('@') {
        None => (raw, None),
        Some((version, pin)) => {
            let hex = pin
                .strip_prefix("sha256:")
                .filter(|h| {
                    h.len() == 64 && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
                })
                .ok_or(CvmVersionError::MalformedPin)?;
            (version, Some(hex))
        }
    };
    if version.is_empty() || version.starts_with('v') {
        return Err(CvmVersionError::NotBare);
    }
    Ok(CvmVersion {
        version,
        manifest_sha256,
    })
}

/// Why a gateway-token hash is refused.
#[cfg(feature = "argon2")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenHashError {
    /// Not a PHC string `argon2` can read.
    NotAPhcString,
    /// A PHC string for an algorithm other than Argon2id.
    NotArgon2id,
    /// No salt, or no hash output, so no token can ever verify against it.
    Incomplete,
    /// Parameters outside the operational profile ([`TOKEN_HASH_M_COST`] and
    /// its siblings), refused before anything is hashed.
    OutsideProfile,
    /// Parameters, version, salt or output length `argon2` refuses to hash
    /// with (a memory cost below its minimum, a salt shorter than eight
    /// bytes, …), so no token can ever verify against it.
    Unusable,
}

#[cfg(feature = "argon2")]
impl core::fmt::Display for TokenHashError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::NotAPhcString => "not a valid Argon2 hash string",
            Self::NotArgon2id => "must be an Argon2id hash",
            Self::Incomplete => "must carry a salt and a hash output",
            Self::Unusable => "has parameters, a salt or an output Argon2 cannot verify with",
            Self::OutsideProfile => {
                "must use the hash-secret profile: Argon2id v=19, m 19456 to 65536 KiB, t 2 to 4, \
                 p 1 to 4, a 32-byte output"
            }
        })
    }
}

/// The Argon2 version a gateway-token hash must use (0x13, `v=19`).
#[cfg(feature = "argon2")]
pub const TOKEN_HASH_VERSION: u32 = 0x13;
/// Memory cost, KiB: from `hash-secret`'s own (`Argon2::default()`, 19,456,
/// OWASP's minimum for Argon2id) up to 64 MiB, a small band for a stronger
/// hash that still costs a boot nothing worth noticing.
#[cfg(feature = "argon2")]
pub const TOKEN_HASH_M_COST: std::ops::RangeInclusive<u32> = 19_456..=65_536;
/// Passes: `hash-secret`'s 2 up to 4.
#[cfg(feature = "argon2")]
pub const TOKEN_HASH_T_COST: std::ops::RangeInclusive<u32> = 2..=4;
/// Lanes: `hash-secret`'s 1 up to 4.
#[cfg(feature = "argon2")]
pub const TOKEN_HASH_P_COST: std::ops::RangeInclusive<u32> = 1..=4;
/// Output length: `hash-secret`'s 32 bytes, exactly.
#[cfg(feature = "argon2")]
pub const TOKEN_HASH_OUTPUT_LEN: usize = 32;

/// Parse the gateway token's measured hash (`GATEWAY_TOKEN_HASH`): an
/// Argon2id PHC string that a token can actually verify against.
///
/// Verification (`PasswordVerifier::verify_password`, which the node runs at
/// boot) needs a salt and an output, and hashes the candidate with the
/// hash's own algorithm, version and parameters; any of those `argon2`
/// refuses makes every token fail. So this runs that same hashing step, with
/// the same `Argon2::default()` the node verifies with, over an empty
/// candidate, and keeps only hashes for which it succeeds: the set this
/// accepts is the set some token verifies against. The comparison itself is
/// the node's, against its secret.
#[cfg(feature = "argon2")]
pub fn parse_gateway_token_hash(hash: &str) -> Result<argon2::PasswordHash, TokenHashError> {
    use argon2::CustomizedPasswordHasher as _;

    let parsed = argon2::PasswordHash::new(hash).map_err(|_| TokenHashError::NotAPhcString)?;
    if parsed.algorithm.as_str() != "argon2id" {
        return Err(TokenHashError::NotArgon2id);
    }
    let (Some(salt), Some(_)) = (&parsed.salt, &parsed.hash) else {
        return Err(TokenHashError::Incomplete);
    };
    let params = argon2::Params::try_from(&parsed).map_err(|_| TokenHashError::Unusable)?;
    // The operational profile, checked before anything is hashed: a hash's
    // parameters decide how much memory and time verifying it costs, so an
    // extreme one would turn the check (and the node's boot) into an
    // allocation of the hash's choosing.
    let in_profile = parsed.version == Some(TOKEN_HASH_VERSION)
        && TOKEN_HASH_M_COST.contains(&params.m_cost())
        && TOKEN_HASH_T_COST.contains(&params.t_cost())
        && TOKEN_HASH_P_COST.contains(&params.p_cost())
        && params.output_len() == Some(TOKEN_HASH_OUTPUT_LEN);
    if !in_profile {
        return Err(TokenHashError::OutsideProfile);
    }
    argon2::Argon2::default()
        .hash_password_customized(
            b"",
            salt,
            Some(parsed.algorithm.as_str()),
            parsed.version,
            params,
        )
        .map_err(|_| TokenHashError::Unusable)?;
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_seconds_are_positive_and_fit_in_milliseconds() {
        assert_eq!(parse_cache_seconds("900"), Some(900));
        assert_eq!(
            parse_cache_seconds(&MAX_CACHE_SECONDS.to_string()),
            Some(MAX_CACHE_SECONDS)
        );
        for bad in [
            "0",
            "-1",
            "",
            " 900",
            "9.5",
            &(MAX_CACHE_SECONDS + 1).to_string(),
        ] {
            assert_eq!(parse_cache_seconds(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn a_deployments_gpus_are_attested_exactly() {
        use Executor::{Cpu, Cuda};
        assert!(check_gpu_attestation(Cuda, Some(8), Some(8)).is_ok());
        assert!(check_gpu_attestation(Cuda, Some(1), Some(1)).is_ok());
        assert!(check_gpu_attestation(Cpu, None, None).is_ok());
        assert!(check_gpu_attestation(Cpu, Some(0), Some(0)).is_ok());
        for (executor, gpus, expected) in [
            (Cuda, None, Some(4)),
            (Cuda, None, None),
            (Cuda, Some(0), Some(8)),
            (Cuda, Some(2), Some(2)),
            (Cuda, Some(8), Some(4)),
            (Cuda, Some(8), None),
            (Cuda, Some(8), Some(0)),
            (Cpu, Some(8), Some(8)),
            (Cpu, None, Some(1)),
        ] {
            assert!(
                check_gpu_attestation(executor, gpus, expected).is_err(),
                "{executor:?} gpus={gpus:?} expected={expected:?}"
            );
        }
    }

    #[test]
    fn resources_fit_the_vm_and_gpus() {
        let sizing = Sizing {
            kv_block_size: 16,
            kv_blocks: 65_536,
            max_model_len: 131_072,
            max_seqs: 64,
            max_batched_tokens: 8192,
            max_prefill_chunk: 4096,
            draft_tokens: 2,
            max_requests: 8,
        };
        assert!(check_resources(&sizing, 65_536, 8).is_ok());
        // Each slot can hold 32 MiB × (129 read + 42 admitted) ≈ 5.3 GiB at
        // worst; eight of them and the headroom exceed 32 GiB.
        assert!(check_resources(&sizing, 32_768, 8).is_err());
        // Near the limit: eleven slots (60,192 MiB of pools, plus headroom
        // and tables) fit 64 GiB; twelve (65,664 MiB of pools alone) do not.
        let near = |n| Sizing {
            max_requests: n,
            ..sizing
        };
        assert!(check_resources(&near(11), 65_536, 8).is_ok());
        assert!(check_resources(&near(12), 65_536, 8).is_err());
        assert!(
            check_resources(
                &Sizing {
                    kv_blocks: 1 << 30,
                    ..sizing
                },
                65_536,
                8
            )
            .is_err()
        );
        assert!(check_resources(&sizing, 65_536, 0).is_err());
        // Block size 1 at the longest model length: 2^20 entries per slot.
        let tables = Sizing {
            kv_block_size: 1,
            max_model_len: caps::MAX_MODEL_LEN,
            max_seqs: caps::MAX_SEQS,
            kv_blocks: 2,
            ..sizing
        };
        assert!(check_resources(&tables, 16_384, 8).is_err());
    }

    #[test]
    fn an_artifact_ref_is_modelwraps() {
        let root = "d".repeat(64);
        let uuid = "3892cd2f-a06e-5aee-8276-93140b9f06ec";
        assert!(is_artifact_ref(&format!("{root}_17419419648_{uuid}")));
        assert!(is_artifact_ref(&format!("{root}_0_{uuid}")));
        for bad in [
            root.clone(),
            format!("{root}_garbage"),
            format!("{root}_x_{uuid}"),
            format!("{root}__{uuid}"),
            format!("{root}_99999999999999999999_{uuid}"),
            format!("{root}_1_{}", uuid.to_uppercase()),
            format!("{root}_1_{uuid}_extra"),
            format!("{}_1_{uuid}", "D".repeat(64)),
            format!("{root}_1_3892cd2fa06e5aee827693140b9f06ec"),
        ] {
            assert!(!is_artifact_ref(&bad), "{bad}");
        }
    }

    #[cfg(feature = "argon2")]
    #[test]
    fn an_extreme_token_hash_is_refused_before_hashing() {
        let good = "$argon2id$v=19$m=19456,t=2,p=1$Mz9P1/uk98yKEflNjzvn5g$unNYT/KTNSNW0JCH9+9OQ2zBApPLxGNZiw746903Q8E";
        assert!(parse_gateway_token_hash(good).is_ok());
        let started = std::time::Instant::now();
        for params in [
            "m=4294967295,t=2,p=1",
            "m=19456,t=4294967295,p=1",
            "m=19456,t=2,p=255",
            "m=8192,t=2,p=1",
            "m=19456,t=1,p=1",
        ] {
            let extreme = good.replace("m=19456,t=2,p=1", params);
            assert_eq!(
                parse_gateway_token_hash(&extreme).unwrap_err(),
                TokenHashError::OutsideProfile,
                "{params}"
            );
        }
        // A 16-byte output and a pre-1.3 version are outside the profile too.
        let short = "$argon2id$v=19$m=19456,t=2,p=1$Mz9P1/uk98yKEflNjzvn5g$aGFzaGhhc2hoYXNoaGFzaA";
        assert_eq!(
            parse_gateway_token_hash(short).unwrap_err(),
            TokenHashError::OutsideProfile
        );
        assert_eq!(
            parse_gateway_token_hash(&good.replace("v=19", "v=16")).unwrap_err(),
            TokenHashError::OutsideProfile
        );
        // Refused without hashing: an m of 4 TiB would take far longer.
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }

    #[test]
    fn path_components_are_safe() {
        for good in ["mimo-v2", "tdx-8x.h200", "0a_b"] {
            assert!(is_safe_component(good), "{good}");
        }
        for bad in [
            "",
            "Upper",
            "has space",
            ".hidden",
            "-dash",
            "a/b",
            "..",
            "é",
        ] {
            assert!(!is_safe_component(bad), "{bad:?}");
        }
    }

    #[test]
    fn container_gpu_access_follows_the_executor() {
        use Executor::{Cpu, Cuda};
        assert!(check_container_gpu_access(Cuda, Some("nvidia"), Some("all")).is_ok());
        assert!(check_container_gpu_access(Cpu, None, None).is_ok());
        for (executor, runtime, gpus) in [
            (Cuda, None, None),
            (Cuda, Some("nvidia"), None),
            (Cuda, Some("nvidia"), Some("1")),
            (Cuda, Some("nvidia"), Some("0,1")),
            (Cuda, Some("runc"), Some("all")),
            (Cpu, Some("nvidia"), None),
            (Cpu, None, Some("all")),
        ] {
            assert!(check_container_gpu_access(executor, runtime, gpus).is_err());
        }
    }

    #[test]
    fn the_env_names_are_exactly_the_measured_ones() {
        assert!(check_env_names(env::MEASURED.iter().copied()).is_ok());
        for extra in [
            "TOKIO_WORKER_THREADS",
            "RUST_MIN_STACK",
            "CUDA_VISIBLE_DEVICES",
            "LD_PRELOAD",
            "GATEWAY_TOKEN",
        ] {
            assert!(check_env_names([env::MODEL_ID, extra]).is_err(), "{extra}");
        }
    }

    #[test]
    fn vm_resources_are_a_launchable_shape() {
        assert!(check_vm_resources(16, 65536).is_ok());
        assert!(check_vm_resources(1, 8192).is_ok());
        assert!(check_vm_resources(MAX_CPUS, MAX_MEMORY_MIB).is_ok());
        for (cpus, memory) in [
            (0, 65536),
            (MAX_CPUS + 1, 65536),
            (16, 1),
            (16, 4096),
            (16, 65535),
            (16, MAX_MEMORY_MIB * 2),
        ] {
            assert!(check_vm_resources(cpus, memory).is_err(), "{cpus} {memory}");
        }
    }

    #[test]
    fn the_gateway_token_shape() {
        assert!(is_gateway_token("dev-gateway-token"));
        assert!(is_gateway_token(&"a".repeat(GATEWAY_TOKEN_MAX_LEN)));
        for bad in [
            "short".to_string(),
            "has a space in it".to_string(),
            "line\nbreak-token-xx".to_string(),
            "non-ascii-tøken-xx".to_string(),
            "a".repeat(GATEWAY_TOKEN_MAX_LEN + 1),
        ] {
            assert!(!is_gateway_token(&bad), "{bad:?}");
        }
    }

    #[test]
    fn the_only_secret_is_the_gateway_token() {
        assert!(check_secrets(["GATEWAY_TOKEN"]).is_ok());
        assert!(check_secrets([]).is_err());
        assert!(check_secrets(["GATEWAY_TOKEN", "RUST_LOG"]).is_err());
        assert!(check_secrets(["GATEWAY_TOKEN", "GATEWAY_TOKEN"]).is_err());
    }

    #[test]
    fn shim_paths_match_as_the_shim_matches() {
        assert!(shim_path_matches(
            "/v1/chat/completions",
            "/v1/chat/completions"
        ));
        assert!(!shim_path_matches(
            "/v1/chat/completions",
            "/v1/chat/completions/x"
        ));
        assert!(shim_path_matches("/*", "/healthz"));
        assert!(shim_path_matches("/v1/*", "/v1/engine/info"));
        assert!(!shim_path_matches("/v1/*", "/v1"));
        assert!(shim_path_matches("/v1*", "/v1"));
        assert!(shim_path_matches("/v1*", "/v1/x"));
        assert!(!shim_path_matches("/v1*", "/v10"));

        assert!(check_shim_paths(&["/*"]).is_ok());
        assert!(check_shim_paths(&["/v1/*", "/healthz"]).is_ok());
        assert!(check_shim_paths(NODE_ROUTES).is_ok());
        assert!(check_shim_paths(&[]).is_err());
        assert!(check_shim_paths(&["/v1/chat/completions"]).is_err());
        assert!(check_shim_paths(&["/*", "/admin"]).is_err());
        assert!(check_shim_paths(&["v1/*", "/healthz"]).is_err());
    }

    #[test]
    fn the_weights_are_the_granted_pinned_pack() {
        let mpk = format!(
            "{}_17419419648_3892cd2f-a06e-5aee-8276-93140b9f06ec",
            "d".repeat(64)
        );
        let pack = ModelPack {
            name: "weights",
            repo: Some("o/m@4444"),
            mpk: Some(&mpk),
        };
        let ok = |packs: &[ModelPack<'_>], granted: &[&str], dir: &str| {
            check_weights_pack(packs, granted, dir, "o/m", "4444")
        };
        assert!(ok(&[pack], &["weights"], "/tinfoil/models/weights").is_ok());
        assert!(ok(&[pack], &["weights"], "/tinfoil/models/weights/original").is_ok());
        assert!(ok(&[], &[], "/tinfoil/models/weights").is_err());
        assert!(ok(&[pack, pack], &["weights"], "/tinfoil/models/weights").is_err());
        assert!(ok(&[pack], &[], "/tinfoil/models/weights").is_err());
        assert!(ok(&[pack], &["other"], "/tinfoil/models/weights").is_err());
        for dir in [
            "/weights",
            "/tinfoil/models/weightsx",
            "/tinfoil/models/weights/../x",
            "/tinfoil/models/weights//x",
        ] {
            assert!(ok(&[pack], &["weights"], dir).is_err(), "{dir}");
        }
        let other_repo = ModelPack {
            repo: Some("o/m@5555"),
            ..pack
        };
        assert!(ok(&[other_repo], &["weights"], "/tinfoil/models/weights").is_err());
        let unpinned = ModelPack { mpk: None, ..pack };
        assert!(ok(&[unpinned], &["weights"], "/tinfoil/models/weights").is_err());
    }

    #[test]
    fn the_prefix_cache_switch_is_exact() {
        assert_eq!(parse_prefix_cache("true"), Some(true));
        assert_eq!(parse_prefix_cache("false"), Some(false));
        for bad in ["True", "1", "yes", ""] {
            assert_eq!(parse_prefix_cache(bad), None);
        }
    }

    #[test]
    fn cvm_version_is_bare_with_at_most_one_inline_pin() {
        let pin = "ab".repeat(32);
        assert_eq!(
            parse_cvm_version("0.15.0"),
            Ok(CvmVersion {
                version: "0.15.0",
                manifest_sha256: None
            })
        );
        assert_eq!(
            parse_cvm_version(&format!("0.15.0@sha256:{pin}"))
                .unwrap()
                .manifest_sha256,
            Some(pin.as_str())
        );
        for (bad, error) in [
            ("".to_string(), CvmVersionError::NotBare),
            ("v0.15.0".to_string(), CvmVersionError::NotBare),
            (format!("@sha256:{pin}"), CvmVersionError::NotBare),
            (
                "0.15.0@sha256:AB".to_string(),
                CvmVersionError::MalformedPin,
            ),
            (
                format!("0.15.0@sha256:{}", pin.to_uppercase()),
                CvmVersionError::MalformedPin,
            ),
            (
                format!("0.15.0@sha512:{pin}"),
                CvmVersionError::MalformedPin,
            ),
            (
                format!("0.15.0@sha256:{pin}@sha256:{pin}"),
                CvmVersionError::MalformedPin,
            ),
            (
                format!("0.15.0@sha256:{pin}x"),
                CvmVersionError::MalformedPin,
            ),
        ] {
            assert_eq!(parse_cvm_version(&bad), Err(error), "{bad:?}");
        }
    }

    #[cfg(feature = "argon2")]
    #[test]
    fn the_token_hash_is_a_whole_argon2id_phc_string() {
        let good = "$argon2id$v=19$m=19456,t=2,p=1$Mz9P1/uk98yKEflNjzvn5g$unNYT/KTNSNW0JCH9+9OQ2zBApPLxGNZiw746903Q8E";
        assert!(parse_gateway_token_hash(good).is_ok());
        assert_eq!(
            parse_gateway_token_hash("$argon2id$garbage").unwrap_err(),
            TokenHashError::NotAPhcString
        );
        assert_eq!(
            parse_gateway_token_hash(&good.replace("argon2id", "argon2i")).unwrap_err(),
            TokenHashError::NotArgon2id
        );
        assert_eq!(
            parse_gateway_token_hash("").unwrap_err(),
            TokenHashError::NotAPhcString
        );
    }

    /// The hashes `parse_gateway_token_hash` refuses are exactly ones no token
    /// verifies against: `verify_password` itself fails for each, even with
    /// the token that would match a well-formed hash.
    #[cfg(feature = "argon2")]
    #[test]
    fn the_token_hash_is_one_a_token_can_verify_against() {
        use argon2::{PasswordHasher, PasswordVerifier};

        let token = b"dev-gateway-token";
        let good = argon2::Argon2::default()
            .hash_password(token)
            .unwrap()
            .to_string();
        let parsed = parse_gateway_token_hash(&good).unwrap();
        argon2::Argon2::default()
            .verify_password(token, &parsed)
            .unwrap();

        let (head, tail) = good.split_once("$m=").unwrap();
        let (_, rest) = tail.split_once('$').unwrap();
        let (salt, output) = rest.split_once('$').unwrap();
        for (bad, error) in [
            (
                format!("{head}$m=1,t=2,p=1${rest}"),
                TokenHashError::Unusable,
            ),
            (
                format!("{head}$m=19456,t=0,p=1${rest}"),
                TokenHashError::Unusable,
            ),
            (
                format!("{head}$m=19456,t=2,p=1$c2FsdA${output}"),
                TokenHashError::NotAPhcString,
            ),
            (
                format!("{head}$m=19456,t=2,p=1"),
                TokenHashError::Incomplete,
            ),
            (
                format!("{head}$m=19456,t=2,p=1${salt}"),
                TokenHashError::Incomplete,
            ),
        ] {
            assert_eq!(parse_gateway_token_hash(&bad).unwrap_err(), error, "{bad}");
            if let Ok(phc) = argon2::PasswordHash::new(&bad) {
                assert!(
                    argon2::Argon2::default()
                        .verify_password(token, &phc)
                        .is_err(),
                    "{bad} verifies, so refusing it is wrong"
                );
            }
        }
    }
}
