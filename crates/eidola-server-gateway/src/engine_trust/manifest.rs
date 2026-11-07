//! The schema of `releases/trust/engine-enclaves.json` and its re-derivation
//! against the committed engine deployments it pins.
//!
//! This file is compiled twice: by `build.rs` (through `#[path]`), which runs
//! [`check`] over the committed file on every gateway build and refuses to
//! build on any disagreement, and by the crate's tests, which run the same
//! function over fixtures. It therefore uses nothing from the crate and only
//! dependencies that are both build- and dev-dependencies (`serde_json`,
//! `sha2`, `eidola-common` with its `deployment` feature, and
//! `tinfoil-verifier`, whose pin compiler is the one the attesting client
//! runs).
//!
//! # What is checked
//!
//! - The file's shape, strictly (the runtime's typed parse first, then every
//!   member's width and form).
//! - Each deployment's `config` names `deploy/engine/<model>/<variant>/
//!   tinfoil-config.yml` for the model it is listed under, and that file
//!   exists and hashes to `config_sha256`.
//! - **The deployment itself** — its config and its sidecar — by
//!   `eidola_common::engine_deployment::deployment::check_deployment`, the one
//!   check `measure-enclave` runs before it emits a pin. Every rule about
//!   what the config may say lives there.
//! - The pin agrees with that deployment: release, weights (hash, repo,
//!   revision), prompt-cache retention, GPU evidence count, TDX policy, and a
//!   TDX MRCONFIGID equal to the config's hash; and the attesting client's
//!   own pin compiler accepts it.
//! - Every deployment of one model agrees on its weights and prompt-cache
//!   policy, since the gateway publishes one of each per model:
//!   `deployment::check_model_agreement`, which `measure-enclave` also runs
//!   before it renders the file.
//! - Every deployment directory committed under `deploy/engine/` is pinned
//!   ([`require_every_deployment_pinned`]), so the file is a function of the
//!   tree in both directions.
//!
//! What is **not** recomputed here: a TDX pin's MRTD and a SEV-SNP launch
//! digest. Both need the platform provider's release images, so they are
//! release-time work (`measure-enclave`'s `measure-engine-enclaves`).

use std::collections::BTreeMap;

use eidola_common::engine_deployment::{CacheConfig, deployment, is_safe_component};
use sha2::{Digest, Sha256};

/// The schema version this gateway reads.
pub const SCHEMA_VERSION: u64 = 1;

/// Where engine deployment configs live, relative to the workspace root.
pub const DEPLOY_ROOT: &str = "deploy/engine";

/// A model the file pins, as checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedModel {
    pub id: String,
    pub weights: CheckedWeights,
    pub prompt_cache: CheckedCachePolicy,
    /// The config path of every accepted deployment, in file order.
    pub deployments: Vec<String>,
}

/// A model's weights: the engine's weights hash and where the bytes came
/// from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedWeights {
    pub sha256: String,
    pub repo: String,
    pub revision: String,
}

/// A model's prompt-cache retention, as its deployments' configs set it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckedCachePolicy {
    pub enabled: bool,
    pub idle_ttl_secs: u64,
    pub max_age_secs: u64,
}

type Object = serde_json::Map<String, serde_json::Value>;

/// Check `releases/trust/engine-enclaves.json` (`json`) against the committed
/// deployments it names, reading each committed file through `read` (a path
/// relative to the workspace root).
///
/// Returns the pinned models in id order. Any disagreement is an error naming
/// the model, the file and the member; nothing is skipped or defaulted.
pub fn check(
    json: &[u8],
    read: &mut dyn FnMut(&str) -> Result<Vec<u8>, String>,
) -> Result<Vec<CheckedModel>, String> {
    // The typed parse the running gateway reads its pins with, first: a file
    // it cannot read (a repeated member at any level, say, which the value
    // reading below would collapse) is refused here.
    super::file::parse(json)?;
    let root: serde_json::Value =
        serde_json::from_slice(json).map_err(|e| format!("engine-enclaves.json: {e}"))?;
    let root = object(&root, "engine-enclaves.json")?;
    exact_keys(
        root,
        &["schema_version", "models"],
        &[],
        "engine-enclaves.json",
    )?;
    let version = root["schema_version"].as_u64();
    if version != Some(SCHEMA_VERSION) {
        return Err(format!(
            "engine-enclaves.json: schema_version must be {SCHEMA_VERSION}"
        ));
    }
    let models = object(&root["models"], "engine-enclaves.json: models")?;

    let mut seen_configs = BTreeMap::new();
    let mut checked = Vec::new();
    for (id, deployments) in models {
        let at = format!("engine-enclaves.json: model {id:?}");
        if !is_safe_component(id) {
            return Err(format!("{at}: a model id must match [a-z0-9][a-z0-9._-]*"));
        }
        let deployments = deployments
            .as_array()
            .filter(|d| !d.is_empty())
            .ok_or_else(|| format!("{at}: must be a non-empty array of deployments"))?;
        let mut configs = Vec::new();
        let mut identities = Vec::new();
        for (index, deployment) in deployments.iter().enumerate() {
            let at = format!("{at}, deployment {index}");
            let (config, weights, prompt_cache) = check_deployment(id, deployment, &at, read)?;
            if let Some(previous) = seen_configs.insert(config.clone(), id.clone()) {
                return Err(format!(
                    "{at}: {config} is already pinned under model {previous:?}"
                ));
            }
            configs.push(config);
            identities.push((
                at,
                deployment::ModelIdentity {
                    weights_sha256: weights.sha256,
                    weights_repo: weights.repo,
                    weights_revision: weights.revision,
                    prompt_cache: CacheConfig {
                        enabled: prompt_cache.enabled,
                        idle_ttl_secs: prompt_cache.idle_ttl_secs,
                        max_age_secs: prompt_cache.max_age_secs,
                    },
                },
            ));
        }
        // The per-model rule `measure-enclave` also runs before it renders.
        let identity = deployment::check_model_agreement(
            id,
            identities.iter().map(|(at, i)| (at.as_str(), i)),
        )?;
        checked.push(CheckedModel {
            id: id.clone(),
            weights: CheckedWeights {
                sha256: identity.weights_sha256,
                repo: identity.weights_repo,
                revision: identity.weights_revision,
            },
            prompt_cache: CheckedCachePolicy {
                enabled: identity.prompt_cache.enabled,
                idle_ttl_secs: identity.prompt_cache.idle_ttl_secs,
                max_age_secs: identity.prompt_cache.max_age_secs,
            },
            deployments: configs,
        });
    }
    Ok(checked)
}

/// Every committed deployment directory must be pinned: `committed` is the
/// config path of every `deploy/engine/<model>/<variant>/` directory in the
/// tree. A deployment committed without its pin is one no gateway built from
/// this tree would accept, which is a release that cannot route to it.
pub fn require_every_deployment_pinned(
    committed: &[String],
    checked: &[CheckedModel],
) -> Result<(), String> {
    for config in committed {
        if !checked.iter().any(|m| m.deployments.contains(config)) {
            return Err(format!(
                "{config} is committed but not pinned in releases/trust/engine-enclaves.json \
                 (measure it with measure-enclave's measure-engine-enclaves)"
            ));
        }
    }
    Ok(())
}

fn check_deployment(
    model_id: &str,
    deployment: &serde_json::Value,
    at: &str,
    read: &mut dyn FnMut(&str) -> Result<Vec<u8>, String>,
) -> Result<(String, CheckedWeights, CheckedCachePolicy), String> {
    let deployment = object(deployment, at)?;
    exact_keys(
        deployment,
        &[
            "config",
            "config_sha256",
            "cvm_version",
            "pin",
            "weights",
            "prompt_cache",
        ],
        &[],
        at,
    )?;

    // The config's path, and the config itself.
    let config_path = string(&deployment["config"], &format!("{at}: config"))?;
    let variant = config_path
        .strip_prefix(&format!("{DEPLOY_ROOT}/{model_id}/"))
        .and_then(|rest| rest.strip_suffix("/tinfoil-config.yml"))
        .filter(|variant| is_safe_component(variant))
        .ok_or_else(|| {
            format!(
                "{at}: config must be {DEPLOY_ROOT}/{model_id}/<variant>/tinfoil-config.yml \
                 with <variant> matching [a-z0-9][a-z0-9._-]*"
            )
        })?;
    let config_bytes = read(config_path)?;
    let config_sha256 = sha256_hex(&config_bytes);
    let pinned_sha256 = hex_string(
        &deployment["config_sha256"],
        32,
        &format!("{at}: config_sha256"),
    )?;
    if pinned_sha256 != config_sha256 {
        return Err(format!(
            "{at}: config_sha256 is {pinned_sha256}, but {config_path} hashes to {config_sha256}"
        ));
    }

    // The deployment itself, by the one check the measurer also runs: every
    // rule about the config and the sidecar lives there.
    let sidecar_path = format!("{DEPLOY_ROOT}/{model_id}/{variant}/deployment.json");
    let sidecar_bytes = read(&sidecar_path)?;
    let checked = deployment::check_deployment(
        model_id,
        config_path,
        &config_bytes,
        &sidecar_path,
        &sidecar_bytes,
    )
    .map_err(|e| format!("{at}: {e}"))?;

    // What follows compares the pin with that deployment.
    let cvm_version = string(&deployment["cvm_version"], &format!("{at}: cvm_version"))?;
    if cvm_version != checked.cvm_version {
        return Err(format!(
            "{at}: cvm_version is {cvm_version:?}, but the config selects {:?}",
            checked.cvm_version
        ));
    }

    let weights_value = object(&deployment["weights"], &format!("{at}: weights"))?;
    exact_keys(
        weights_value,
        &["sha256", "repo", "revision"],
        &[],
        &format!("{at}: weights"),
    )?;
    let weights = CheckedWeights {
        sha256: hex_string(
            &weights_value["sha256"],
            32,
            &format!("{at}: weights.sha256"),
        )?
        .to_owned(),
        repo: string(&weights_value["repo"], &format!("{at}: weights.repo"))?.to_owned(),
        revision: hex_string(
            &weights_value["revision"],
            20,
            &format!("{at}: weights.revision"),
        )?
        .to_owned(),
    };
    if weights.sha256 != checked.measured.weights_sha256 {
        return Err(format!(
            "{at}: EIDOLA_ENGINE_WEIGHTS_SHA256 is {}, but the pin requires {}",
            checked.measured.weights_sha256, weights.sha256
        ));
    }
    if weights.repo != checked.weights_repo || weights.revision != checked.weights_revision {
        return Err(format!(
            "{at}: weights.repo / weights.revision differ from {sidecar_path}"
        ));
    }

    let cache_value = object(&deployment["prompt_cache"], &format!("{at}: prompt_cache"))?;
    exact_keys(
        cache_value,
        &["enabled", "idle_ttl_secs", "max_age_secs"],
        &[],
        &format!("{at}: prompt_cache"),
    )?;
    let prompt_cache = CheckedCachePolicy {
        enabled: cache_value["enabled"]
            .as_bool()
            .ok_or_else(|| format!("{at}: prompt_cache.enabled must be a boolean"))?,
        idle_ttl_secs: cache_value["idle_ttl_secs"]
            .as_u64()
            .ok_or_else(|| format!("{at}: prompt_cache.idle_ttl_secs must be an integer"))?,
        max_age_secs: cache_value["max_age_secs"]
            .as_u64()
            .ok_or_else(|| format!("{at}: prompt_cache.max_age_secs must be an integer"))?,
    };
    let configured = CheckedCachePolicy {
        enabled: checked.measured.cache.enabled,
        idle_ttl_secs: checked.measured.cache.idle_ttl_secs,
        max_age_secs: checked.measured.cache.max_age_secs,
    };
    if configured != prompt_cache {
        return Err(format!(
            "{at}: prompt_cache is {prompt_cache:?}, but the config sets {configured:?}"
        ));
    }

    let pin = object(&deployment["pin"], &format!("{at}: pin"))?;
    exact_keys(
        pin,
        &["platform"],
        &["expected_gpus"],
        &format!("{at}: pin"),
    )?;
    let expected_gpus = match pin.get("expected_gpus") {
        None => None,
        Some(v) => Some(
            v.as_u64()
                .filter(|n| u32::try_from(*n).is_ok())
                .ok_or_else(|| format!("{at}: pin.expected_gpus must be a u32"))?,
        ),
    };
    if expected_gpus != checked.expected_gpus {
        return Err(format!(
            "{at}: pin.expected_gpus differs from {sidecar_path}"
        ));
    }
    let platform = object(&pin["platform"], &format!("{at}: pin.platform"))?;
    match platform
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["tdx"] => {
            let tdx = object(&platform["tdx"], &format!("{at}: pin.platform.tdx"))?;
            exact_keys(
                tdx,
                &["mrtd", "mrconfigid", "policy"],
                &[],
                &format!("{at}: pin.platform.tdx"),
            )?;
            hex_string(&tdx["mrtd"], 48, &format!("{at}: pin.platform.tdx.mrtd"))?;
            let mrconfigid = hex_string(
                &tdx["mrconfigid"],
                48,
                &format!("{at}: pin.platform.tdx.mrconfigid"),
            )?;
            let expected = format!("{config_sha256}{}", "00".repeat(16));
            if mrconfigid != expected {
                return Err(format!(
                    "{at}: pin.platform.tdx.mrconfigid is not the hash of {config_path} \
                     (expected {expected})"
                ));
            }
            object(&tdx["policy"], &format!("{at}: pin.platform.tdx.policy"))?;
            if checked.tdx_policy.as_ref() != Some(&tdx["policy"]) {
                return Err(format!(
                    "{at}: pin.platform.tdx.policy differs from {sidecar_path}'s tdx_policy"
                ));
            }
        }
        ["sev_snp"] => {
            let snp = object(&platform["sev_snp"], &format!("{at}: pin.platform.sev_snp"))?;
            exact_keys(
                snp,
                &["measurement"],
                &[],
                &format!("{at}: pin.platform.sev_snp"),
            )?;
            hex_string(
                &snp["measurement"],
                48,
                &format!("{at}: pin.platform.sev_snp.measurement"),
            )?;
            if checked.tdx_policy.is_some() {
                return Err(format!(
                    "{at}: a SEV-SNP pin, but {sidecar_path} states a tdx_policy"
                ));
            }
        }
        _ => {
            return Err(format!(
                "{at}: pin.platform must have exactly one member, tdx or sev_snp"
            ));
        }
    }

    // The pin as the attesting client will read it: its own type, compiled by
    // its own pin compiler (every field at its width, a TDX policy's safety
    // bits and MR_SEAM allowlist), so a pin that builds is one it accepts.
    let typed: tinfoil_verifier::AllowedMeasurement =
        serde_json::from_value(deployment["pin"].clone())
            .map_err(|e| format!("{at}: pin is not a tinfoil-verifier pin: {e}"))?;
    tinfoil_verifier::validate_pins(std::slice::from_ref(&typed))
        .map_err(|e| format!("{at}: the attesting client refuses this pin: {e}"))?;

    Ok((config_path.to_owned(), weights, prompt_cache))
}

fn object<'a>(value: &'a serde_json::Value, at: &str) -> Result<&'a Object, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{at}: must be an object"))
}

fn string<'a>(value: &'a serde_json::Value, at: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("{at}: must be a string"))
}

/// A string of exactly `bytes` bytes as lowercase hex.
fn hex_string<'a>(value: &'a serde_json::Value, bytes: usize, at: &str) -> Result<&'a str, String> {
    let s = string(value, at)?;
    if is_lower_hex(s, bytes) {
        Ok(s)
    } else {
        Err(format!("{at}: must be {} lowercase hex digits", bytes * 2))
    }
}

/// `required` keys must all be present, `optional` ones may be, and nothing
/// else is allowed.
fn exact_keys(
    object: &Object,
    required: &[&str],
    optional: &[&str],
    at: &str,
) -> Result<(), String> {
    for key in required {
        if !object.contains_key(*key) {
            return Err(format!("{at}: missing member {key:?}"));
        }
    }
    for key in object.keys() {
        if !required.contains(&key.as_str()) && !optional.contains(&key.as_str()) {
            return Err(format!("{at}: unknown member {key:?}"));
        }
    }
    Ok(())
}

fn is_lower_hex(s: &str, bytes: usize) -> bool {
    s.len() == bytes * 2 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
