//! The schema of `releases/trust/engine-enclaves.json` and its re-derivation
//! against the committed engine deployments it pins.
//!
//! This file is compiled twice: by `build.rs` (through `#[path]`), which runs
//! [`check`] over the committed file on every gateway build and refuses to
//! build on any disagreement, and by the crate's tests, which run the same
//! function over fixtures. It therefore uses nothing from the crate and only
//! dependencies that are both build- and dev-dependencies (`serde_json`,
//! `serde_yaml`, `sha2`, `eidola-common` with its `argon2` feature, whose
//! `engine_deployment` module is the grammar the node boots with, and
//! `tinfoil-verifier`, whose pin compiler is the one the attesting client
//! runs).
//!
//! # What is checked
//!
//! Everything a pin states that committed source determines is recomputed
//! here and must agree:
//!
//! - the file's shape, strictly: unknown, missing and mistyped members are
//!   refused at every level, hex is lowercase and of its exact width;
//! - each deployment's `config` names `deploy/engine/<model>/<variant>/
//!   tinfoil-config.yml` for the model it is listed under, and that file
//!   exists and hashes to `config_sha256`;
//! - the config selects the release `cvm_version` records, with the release
//!   manifest pinned inline (`X@sha256:<hex>`);
//! - the config runs exactly one `eidola-server-engine` container, pinned by
//!   digest, serving this model id and these weights, from verified read-only
//!   storage, with the prompt-cache policy the pin records, its gateway token
//!   injected as a secret and only its Argon2id hash measured;
//! - a TDX pin's MRCONFIGID is the config's hash;
//! - the deployment's sidecar (`deployment.json`, beside the config) states
//!   the weights provenance, GPU count and TDX machine policy the pin carries;
//! - every deployment of one model agrees on its weights and prompt-cache
//!   policy, since the gateway publishes one of each per model;
//! - every deployment directory committed under `deploy/engine/` is pinned
//!   ([`require_every_deployment_pinned`]), so the file is a function of the
//!   tree in both directions.
//!
//! What is **not** recomputed here: a TDX pin's MRTD and a SEV-SNP launch
//! digest. Both need the platform provider's release images, so they are
//! release-time work (`measure-enclave`'s `measure-engine-enclaves`).

use std::collections::BTreeMap;

use eidola_common::engine_deployment::{self, allowed_keys};
use sha2::{Digest, Sha256};

/// The schema version this gateway reads.
pub const SCHEMA_VERSION: u64 = 1;

/// The image every engine deployment runs, pinned by digest.
pub const ENGINE_IMAGE_PREFIX: &str = "ghcr.io/eidola-ai/eidola-server-engine@sha256:";

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
        let mut model: Option<CheckedModel> = None;
        for (index, deployment) in deployments.iter().enumerate() {
            let at = format!("{at}, deployment {index}");
            let (config, weights, prompt_cache) = check_deployment(id, deployment, &at, read)?;
            if let Some(previous) = seen_configs.insert(config.clone(), id.clone()) {
                return Err(format!(
                    "{at}: {config} is already pinned under model {previous:?}"
                ));
            }
            match &mut model {
                None => {
                    model = Some(CheckedModel {
                        id: id.clone(),
                        weights,
                        prompt_cache,
                        deployments: vec![config],
                    })
                }
                Some(model) => {
                    if model.weights != weights {
                        return Err(format!(
                            "{at}: weights differ from the model's other deployments; \
                             one model id names one set of weights"
                        ));
                    }
                    if model.prompt_cache != prompt_cache {
                        return Err(format!(
                            "{at}: prompt_cache differs from the model's other deployments; \
                             the gateway publishes one retention policy per model"
                        ));
                    }
                    model.deployments.push(config);
                }
            }
        }
        checked.push(model.expect("a non-empty deployment list"));
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
    let config = EngineConfig::parse(&config_bytes, config_path)?;
    let at_config = format!("{at} ({config_path})");

    // The release it launches on.
    let cvm_version = string(&deployment["cvm_version"], &format!("{at}: cvm_version"))?;
    if cvm_version != config.cvm_version {
        return Err(format!(
            "{at}: cvm_version is {cvm_version:?}, but the config selects {:?}",
            config.cvm_version
        ));
    }

    // Weights, and the config serving them.
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
    if !is_repo(&weights.repo) {
        return Err(format!(
            "{at}: weights.repo must be <owner>/<name> of [A-Za-z0-9._-]"
        ));
    }
    // The whole measured environment, read by the node's own boot grammar:
    // a pin is accepted only for a config the node boots with, read the same
    // way. Its secret token is the one variable not measured.
    let measured = engine_deployment::parse_measured(&|name| config.env.get(name).cloned())
        .map_err(|e| format!("{at_config}: the node would refuse this env: {e}"))?;
    if measured.model_id != model_id {
        return Err(format!(
            "{at_config}: EIDOLA_ENGINE_MODEL_ID is {:?}, but the pin is for {model_id:?}",
            measured.model_id
        ));
    }
    if measured.weights_sha256 != weights.sha256 {
        return Err(format!(
            "{at_config}: EIDOLA_ENGINE_WEIGHTS_SHA256 is {}, but the pin requires {}",
            measured.weights_sha256, weights.sha256
        ));
    }
    if measured.weights_storage != engine_deployment::WeightsStorage::VerifiedReadonly {
        return Err(format!(
            "{at_config}: EIDOLA_ENGINE_WEIGHTS_STORAGE is {:?}, but a pinned deployment \
             must be \"verified-readonly\"",
            measured.weights_storage.as_str()
        ));
    }

    // Prompt-cache retention.
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
    if prompt_cache.idle_ttl_secs > prompt_cache.max_age_secs {
        return Err(format!(
            "{at}: prompt_cache.idle_ttl_secs exceeds max_age_secs, which the node refuses"
        ));
    }
    // The shim forwards to the node's listener: the port must be the one the
    // node binds, on every interface, as the gateway's own deployment does
    // (`BIND_ADDR` 0.0.0.0:8080 behind `upstream-port: 8080`); the shim runs
    // outside the workload container, so a loopback-only node is unreachable.
    if u64::from(measured.bind_addr.port()) != config.shim_upstream_port {
        return Err(format!(
            "{at_config}: shim.upstream-port is {}, but EIDOLA_ENGINE_BIND_ADDR listens on port {}",
            config.shim_upstream_port,
            measured.bind_addr.port()
        ));
    }
    if !measured.bind_addr.ip().is_unspecified() {
        return Err(format!(
            "{at_config}: EIDOLA_ENGINE_BIND_ADDR must listen on every interface \
             (0.0.0.0 or [::]) for the shim to reach it"
        ));
    }

    let configured = CheckedCachePolicy {
        enabled: measured.cache.enabled,
        idle_ttl_secs: measured.cache.idle_ttl_secs,
        max_age_secs: measured.cache.max_age_secs,
    };
    if configured != prompt_cache {
        return Err(format!(
            "{at}: prompt_cache is {prompt_cache:?}, but the config sets {configured:?}"
        ));
    }

    // The gateway token is the one secret: anything else would reach the
    // node outside the measurement.
    engine_deployment::check_secrets(config.secrets.iter().map(String::as_str))
        .map_err(|e| format!("{at_config}: {e}"))?;
    // The shim exposes exactly the node's routes, and the weights the node
    // reads are the pinned, granted model pack.
    engine_deployment::check_shim_paths(
        &config
            .shim_paths
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
    )
    .map_err(|e| format!("{at_config}: {e}"))?;
    let packs: Vec<engine_deployment::ModelPack<'_>> = config
        .packs
        .iter()
        .map(|(name, repo, mpk)| engine_deployment::ModelPack {
            name,
            repo: repo.as_deref(),
            mpk: mpk.as_deref(),
        })
        .collect();
    engine_deployment::check_weights_pack(
        &packs,
        &config
            .granted
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        &measured.weights_dir,
        &weights.repo,
        &weights.revision,
    )
    .map_err(|e| format!("{at_config}: {e}"))?;
    if config.env.contains_key("GATEWAY_TOKEN") {
        return Err(format!(
            "{at_config}: GATEWAY_TOKEN must not be a measured environment value"
        ));
    }
    // The env is exactly the node's measured variables, so nothing a
    // dependency reads (`TOKIO_WORKER_THREADS`, `LD_*`, `CUDA_*`, …) reaches
    // it; and the VM is a shape the platform launches.
    engine_deployment::check_env_names(config.env.keys().map(String::as_str))
        .map_err(|e| format!("{at_config}: {e}"))?;
    engine_deployment::check_vm_resources(config.cpus, config.memory)
        .map_err(|e| format!("{at_config}: {e}"))?;

    // The pin, and the sidecar stating what source cannot derive.
    let sidecar_path = format!("{DEPLOY_ROOT}/{model_id}/{variant}/deployment.json");
    let sidecar = Sidecar::parse(&read(&sidecar_path)?, &sidecar_path)?;
    if sidecar.repo != weights.repo || sidecar.revision != weights.revision {
        return Err(format!(
            "{at}: weights.repo / weights.revision differ from {sidecar_path}"
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
    // The accelerators the prompt runs on are attested on every handshake,
    // by the rule the measurer applies too.
    engine_deployment::check_gpu_attestation(measured.executor, config.gpus, expected_gpus)
        .map_err(|e| format!("{at}: {e}"))?;
    engine_deployment::check_container_gpu_access(
        measured.executor,
        config.runtime.as_deref(),
        config.container_gpus.as_deref(),
    )
    .map_err(|e| format!("{at_config}: {e}"))?;
    if expected_gpus != sidecar.expected_gpus {
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
            if sidecar.tdx_policy.as_ref() != Some(&tdx["policy"]) {
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
            if sidecar.tdx_policy.is_some() {
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

/// The parts of an engine deployment's `tinfoil-config.yml` the pin is
/// checked against.
struct EngineConfig {
    cvm_version: String,
    secrets: Vec<String>,
    env: BTreeMap<String, String>,
    /// The port the attestation shim forwards to (`shim.upstream-port`).
    shim_upstream_port: u64,
    /// The GPUs the deployment attaches (`gpus`), when it states a count.
    gpus: Option<u64>,
    /// The engine container's `runtime`.
    runtime: Option<String>,
    /// The engine container's GPU selection (`gpus`), as written.
    container_gpus: Option<String>,
    /// The model packs, as `(name, repo, mpk)`.
    packs: Vec<(String, Option<String>, Option<String>)>,
    /// The model packs granted to the engine container.
    granted: Vec<String>,
    /// The shim's `paths`.
    shim_paths: Vec<String>,
    /// The VM's vCPUs and memory (MiB).
    cpus: u64,
    memory: u64,
}

/// A YAML list of strings.
fn string_list(value: &serde_yaml::Value, path: &str, what: &str) -> Result<Vec<String>, String> {
    value
        .as_sequence()
        .and_then(|l| {
            l.iter()
                .map(|s| s.as_str().map(str::to_owned))
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| format!("{path}: {what} must be a list of strings"))
}

/// Refuse any key of the mapping `value` that `allowed` does not list.
fn allow_keys(
    value: &serde_yaml::Value,
    allowed: &[&str],
    path: &str,
    what: &str,
) -> Result<(), String> {
    let mapping = value
        .as_mapping()
        .ok_or_else(|| format!("{path}: {what} must be a mapping"))?;
    let mut keys = Vec::with_capacity(mapping.len());
    for key in mapping.keys() {
        keys.push(
            key.as_str()
                .ok_or_else(|| format!("{path}: {what} has a non-string key"))?,
        );
    }
    match engine_deployment::first_disallowed_key(allowed, keys) {
        None => Ok(()),
        Some(key) => Err(format!(
            "{path}: {what} may not set {key:?}; an engine deployment uses only {allowed:?} there"
        )),
    }
}

impl EngineConfig {
    fn parse(bytes: &[u8], path: &str) -> Result<Self, String> {
        let config: serde_yaml::Value =
            serde_yaml::from_slice(bytes).map_err(|e| format!("{path}: {e}"))?;
        // Only the keys an engine deployment needs, at every level that can
        // change what runs (`engine_deployment::allowed_keys`, which the
        // measurer applies too).
        allow_keys(&config, allowed_keys::CONFIG, path, "the config")?;
        if let Some(shim) = config.get("shim") {
            allow_keys(shim, allowed_keys::SHIM, path, "shim")?;
        }
        if let Some(models) = config.get("models") {
            let models = models
                .as_sequence()
                .ok_or_else(|| format!("{path}: models must be a list"))?;
            for (i, model) in models.iter().enumerate() {
                allow_keys(model, allowed_keys::MODEL, path, &format!("models[{i}]"))?;
            }
        }
        let mut packs = Vec::new();
        for (i, model) in config
            .get("models")
            .and_then(serde_yaml::Value::as_sequence)
            .map(Vec::as_slice)
            .unwrap_or(&[])
            .iter()
            .enumerate()
        {
            let field = |key: &str| -> Result<Option<String>, String> {
                match model.get(key) {
                    None => Ok(None),
                    Some(v) => v
                        .as_str()
                        .map(|s| Some(s.to_owned()))
                        .ok_or_else(|| format!("{path}: models[{i}].{key} must be a string")),
                }
            };
            let name = field("name")?.ok_or_else(|| format!("{path}: models[{i}] has no name"))?;
            packs.push((name, field("repo")?, field("mpk")?));
        }
        let shim_paths = match config.get("shim").and_then(|shim| shim.get("paths")) {
            None => Vec::new(),
            Some(paths) => string_list(paths, path, "shim.paths")?,
        };
        let cvm_version = config
            .get("cvm-version")
            .and_then(serde_yaml::Value::as_str)
            .ok_or_else(|| format!("{path}: cvm-version must be a string"))?;
        // `measure-enclave` reads the release through this same grammar.
        let release = engine_deployment::parse_cvm_version(cvm_version)
            .map_err(|e| format!("{path}: {e}"))?;
        if release.manifest_sha256.is_none() {
            return Err(format!(
                "{path}: cvm-version must pin its release manifest inline, \
                 as VERSION@sha256:<64 lowercase hex>"
            ));
        }

        let containers = config
            .get("containers")
            .and_then(serde_yaml::Value::as_sequence)
            .ok_or_else(|| format!("{path}: containers must be a list"))?;
        // Exactly one container, the engine, pinned by full digest. MRCONFIGID
        // covers this file's bytes, not what a tag resolves to, so any image
        // named by a tag (or a companion container at all) would run code the
        // measurement does not fix.
        let engine = match containers.as_slice() {
            [engine]
                if engine
                    .get("image")
                    .and_then(serde_yaml::Value::as_str)
                    .and_then(|image| image.strip_prefix(ENGINE_IMAGE_PREFIX))
                    .is_some_and(|digest| is_lower_hex(digest, 32)) =>
            {
                engine
            }
            _ => {
                return Err(format!(
                    "{path}: an engine deployment runs exactly one container, its image \
                     {ENGINE_IMAGE_PREFIX}<64 lowercase hex>; every other container, and any \
                     image named by a tag or a short digest, escapes the measurement"
                ));
            }
        };

        allow_keys(
            engine,
            allowed_keys::CONTAINER,
            path,
            "the engine container",
        )?;
        let granted = match engine.get("models") {
            None => Vec::new(),
            Some(models) => string_list(models, path, "the engine container's models")?,
        };
        let runtime = match engine.get("runtime") {
            None => None,
            Some(runtime) => Some(
                runtime
                    .as_str()
                    .ok_or_else(|| {
                        format!("{path}: the engine container's runtime must be a string")
                    })?
                    .to_owned(),
            ),
        };
        let container_gpus = match engine.get("gpus") {
            None => None,
            Some(serde_yaml::Value::String(s)) => Some(s.clone()),
            Some(serde_yaml::Value::Number(n)) => Some(n.to_string()),
            Some(_) => {
                return Err(format!(
                    "{path}: the engine container's gpus must be a count or a selection"
                ));
            }
        };

        let secrets = match engine.get("secrets") {
            None | Some(serde_yaml::Value::Null) => Vec::new(),
            Some(list) => list
                .as_sequence()
                .and_then(|l| {
                    l.iter()
                        .map(|s| s.as_str().map(str::to_owned))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or_else(|| {
                    format!("{path}: the engine container's secrets must be a list of names")
                })?,
        };

        let mut env = BTreeMap::new();
        let entries = engine
            .get("env")
            .and_then(serde_yaml::Value::as_sequence)
            .ok_or_else(|| format!("{path}: the engine container's env must be a list"))?;
        for entry in entries {
            let (key, value) = entry
                .as_mapping()
                .filter(|m| m.len() == 1)
                .and_then(|m| m.iter().next())
                .and_then(|(k, v)| Some((k.as_str()?, v.as_str()?)))
                .ok_or_else(|| {
                    format!(
                        "{path}: each env entry must be one NAME: \"value\" pair \
                         with a quoted string value"
                    )
                })?;
            if env.insert(key.to_owned(), value.to_owned()).is_some() {
                return Err(format!("{path}: env sets {key} twice"));
            }
        }

        let shim_upstream_port = config
            .get("shim")
            .and_then(|shim| shim.get("upstream-port"))
            .and_then(serde_yaml::Value::as_u64)
            .ok_or_else(|| format!("{path}: shim.upstream-port must be a port number"))?;
        let whole = |key: &str| -> Result<u64, String> {
            config
                .get(key)
                .and_then(serde_yaml::Value::as_u64)
                .ok_or_else(|| format!("{path}: {key} must be a whole number"))
        };
        let cpus = whole("cpus")?;
        let memory = whole("memory")?;
        let gpus = match config.get("gpus") {
            None => None,
            Some(gpus) => Some(
                gpus.as_u64()
                    .ok_or_else(|| format!("{path}: gpus must be a whole number"))?,
            ),
        };

        Ok(Self {
            cvm_version: cvm_version.to_owned(),
            secrets,
            env,
            shim_upstream_port,
            gpus,
            runtime,
            container_gpus,
            packs,
            granted,
            shim_paths,
            cpus,
            memory,
        })
    }
}

/// A deployment's `deployment.json`: what a pin carries that the config does
/// not determine.
struct Sidecar {
    repo: String,
    revision: String,
    expected_gpus: Option<u64>,
    tdx_policy: Option<serde_json::Value>,
}

impl Sidecar {
    fn parse(bytes: &[u8], path: &str) -> Result<Self, String> {
        let value: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| format!("{path}: {e}"))?;
        let sidecar = object(&value, path)?;
        exact_keys(
            sidecar,
            &["weights"],
            &["expected_gpus", "tdx_policy"],
            path,
        )?;
        let weights = object(&sidecar["weights"], &format!("{path}: weights"))?;
        exact_keys(
            weights,
            &["repo", "revision"],
            &[],
            &format!("{path}: weights"),
        )?;
        let expected_gpus = match sidecar.get("expected_gpus") {
            None => None,
            Some(v) => Some(
                v.as_u64()
                    .ok_or_else(|| format!("{path}: expected_gpus must be an integer"))?,
            ),
        };
        let tdx_policy = match sidecar.get("tdx_policy") {
            None => None,
            Some(policy) => {
                object(policy, &format!("{path}: tdx_policy"))?;
                Some(policy.clone())
            }
        };
        Ok(Self {
            repo: string(&weights["repo"], &format!("{path}: weights.repo"))?.to_owned(),
            revision: string(&weights["revision"], &format!("{path}: weights.revision"))?
                .to_owned(),
            expected_gpus,
            tdx_policy,
        })
    }
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

/// A path component: `[a-z0-9][a-z0-9._-]*`.
fn is_safe_component(s: &str) -> bool {
    let mut bytes = s.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && bytes.all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-')
        })
}

/// `<owner>/<name>`, each non-empty `[A-Za-z0-9._-]` and not `.` or `..`.
fn is_repo(s: &str) -> bool {
    let part = |p: &str| {
        !p.is_empty()
            && p != "."
            && p != ".."
            && p.bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    };
    matches!(s.split_once('/'), Some((owner, name)) if part(owner) && part(name))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
