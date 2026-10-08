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
}

#[cfg(feature = "argon2")]
impl core::fmt::Display for TokenHashError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::NotAPhcString => "not a valid Argon2 hash string",
            Self::NotArgon2id => "must be an Argon2id hash",
        })
    }
}

/// Parse the gateway token's measured hash (`GATEWAY_TOKEN_HASH`): a PHC
/// string for Argon2id. The node verifies its token against the result.
#[cfg(feature = "argon2")]
pub fn parse_gateway_token_hash(hash: &str) -> Result<argon2::PasswordHash, TokenHashError> {
    let parsed = argon2::PasswordHash::new(hash).map_err(|_| TokenHashError::NotAPhcString)?;
    if parsed.algorithm.as_str() != "argon2id" {
        return Err(TokenHashError::NotArgon2id);
    }
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
}
