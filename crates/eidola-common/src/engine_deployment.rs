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
    let root_hash_ok = pack.mpk.is_some_and(|mpk| {
        mpk.split('_').next().is_some_and(|root| {
            root.len() == 64 && root.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        })
    });
    if !root_hash_ok {
        return Err("the model pack must be pinned by its root hash (`mpk: <64 hex>_…`)".into());
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
/// - the `cpu` executor attaches none and requires none.
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
        })
    }
}

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
        let mpk = format!("{}_17419419648_3892cd2f", "d".repeat(64));
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
        let good = "$argon2id$v=19$m=19456,t=2,p=1$c2FsdHNhbHRzYWx0$aGFzaGhhc2hoYXNoaGFzaA";
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
