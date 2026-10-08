//! One engine deployment, checked whole: its committed `tinfoil-config.yml`
//! and its `deployment.json` sidecar.
//!
//! [`check_deployment`] is the single check both readers of a deployment run:
//! the gateway's build, before it accepts a pin for the deployment, and
//! `measure-enclave`, before it emits one. Every rule this module states
//! (the key allow-list, the one digest-pinned container, the measured env,
//! the shim, the weights pack, the VM shape, GPU attestation) is applied here
//! and nowhere else, so the two cannot drift. What is left to each caller is
//! what only it has: the gateway compares the pin it is given with the
//! returned [`Deployment`], and the measurer computes the launch identity
//! from the platform release.
//!
//! Behind this crate's `deployment` feature, which adds `serde_yaml` (and
//! `argon2`) only to the consumers that enable it.

use std::collections::BTreeMap;

use serde_json::Value as Json;
use serde_yaml::Value as Yaml;

use super::{
    MeasuredConfig, ModelPack, WeightsStorage, allowed_keys, check_container_gpu_access,
    check_env_names, check_gpu_attestation, check_secrets, check_shim_paths, check_vm_resources,
    check_weights_pack, env, first_disallowed_key, parse_cvm_version, parse_measured,
};

/// The image every engine deployment runs, pinned by digest.
pub const ENGINE_IMAGE_PREFIX: &str = "ghcr.io/eidola-ai/eidola-server-engine@sha256:";

/// The model-pack layouts (Tinfoil's `ModelSpec.Schema`) a weights pack may
/// declare: the pack's root hash depends on the layout it was built with, so
/// the layout is stated, never defaulted.
pub const PACK_SCHEMAS: &[u64] = &[1, 2];

/// A deployment that passed every rule.
#[derive(Clone, Debug, PartialEq)]
pub struct Deployment {
    /// The config's `cvm-version`, as written (with its inline manifest pin).
    pub cvm_version: String,
    /// The node's measured configuration.
    pub measured: MeasuredConfig,
    /// The weights' provenance: `<owner>/<name>` and a 40-hex revision.
    pub weights_repo: String,
    pub weights_revision: String,
    /// The GPU evidence items a pin must require, if any.
    pub expected_gpus: Option<u64>,
    /// The Intel TDX machine policy a pin carries, when the deployment is TDX.
    pub tdx_policy: Option<Json>,
}

/// Check one deployment of `model_id`: its config (`config_path`, read as
/// `config`) and its sidecar (`sidecar_path`, read as `sidecar`). Errors name
/// the file and the rule, never a secret.
pub fn check_deployment(
    model_id: &str,
    config_path: &str,
    config: &[u8],
    sidecar_path: &str,
    sidecar: &[u8],
) -> Result<Deployment, String> {
    let c = EngineConfig::parse(config, config_path)?;
    let at = config_path;

    // The whole measured environment, read by the node's own boot grammar,
    // and nothing else in it.
    let measured = parse_measured(&|name| c.env.get(name).cloned())
        .map_err(|e| format!("{at}: the node would refuse this env: {e}"))?;
    if c.env.contains_key(env::GATEWAY_TOKEN) {
        return Err(format!(
            "{at}: GATEWAY_TOKEN must not be a measured environment value"
        ));
    }
    check_env_names(c.env.keys().map(String::as_str)).map_err(|e| format!("{at}: {e}"))?;
    if measured.model_id != model_id {
        return Err(format!(
            "{at}: EIDOLA_ENGINE_MODEL_ID is {:?}, but the deployment is for {model_id:?}",
            measured.model_id
        ));
    }
    if measured.weights_storage != WeightsStorage::VerifiedReadonly {
        return Err(format!(
            "{at}: EIDOLA_ENGINE_WEIGHTS_STORAGE is {:?}, but a pinned deployment must be \
             \"verified-readonly\"",
            measured.weights_storage.as_str()
        ));
    }

    // The shim reaches the node and exposes exactly its routes.
    if u64::from(measured.bind_addr.port()) != c.shim_upstream_port {
        return Err(format!(
            "{at}: shim.upstream-port is {}, but EIDOLA_ENGINE_BIND_ADDR listens on port {}",
            c.shim_upstream_port,
            measured.bind_addr.port()
        ));
    }
    if !measured.bind_addr.ip().is_unspecified() {
        return Err(format!(
            "{at}: EIDOLA_ENGINE_BIND_ADDR must listen on every interface (0.0.0.0 or [::]) \
             for the shim to reach it"
        ));
    }
    check_shim_paths(&c.shim_paths.iter().map(String::as_str).collect::<Vec<_>>())
        .map_err(|e| format!("{at}: {e}"))?;

    // The one secret, and the VM shape.
    check_secrets(c.secrets.iter().map(String::as_str)).map_err(|e| format!("{at}: {e}"))?;
    check_vm_resources(c.cpus, c.memory).map_err(|e| format!("{at}: {e}"))?;

    // The sidecar, and the weights it records.
    let side = Sidecar::parse(sidecar, sidecar_path)?;
    let packs: Vec<ModelPack<'_>> = c
        .packs
        .iter()
        .map(|p| ModelPack {
            name: &p.name,
            repo: p.repo.as_deref(),
            mpk: p.mpk.as_deref(),
        })
        .collect();
    check_weights_pack(
        &packs,
        &c.granted.iter().map(String::as_str).collect::<Vec<_>>(),
        &measured.weights_dir,
        &side.repo,
        &side.revision,
    )
    .map_err(|e| format!("{at}: {e}"))?;

    // GPUs: attested exactly, and the container's access matching its executor.
    check_gpu_attestation(measured.executor, c.gpus, side.expected_gpus)
        .map_err(|e| format!("{at}: {e}"))?;
    check_container_gpu_access(
        measured.executor,
        c.runtime.as_deref(),
        c.container_gpus.as_deref(),
    )
    .map_err(|e| format!("{at}: {e}"))?;

    Ok(Deployment {
        cvm_version: c.cvm_version,
        measured,
        weights_repo: side.repo,
        weights_revision: side.revision,
        expected_gpus: side.expected_gpus,
        tdx_policy: side.tdx_policy,
    })
}

struct Pack {
    name: String,
    repo: Option<String>,
    mpk: Option<String>,
}

/// The parts of a deployment's `tinfoil-config.yml` the rules read.
struct EngineConfig {
    cvm_version: String,
    secrets: Vec<String>,
    env: BTreeMap<String, String>,
    shim_upstream_port: u64,
    shim_paths: Vec<String>,
    gpus: Option<u64>,
    runtime: Option<String>,
    container_gpus: Option<String>,
    packs: Vec<Pack>,
    granted: Vec<String>,
    cpus: u64,
    memory: u64,
}

impl EngineConfig {
    fn parse(bytes: &[u8], path: &str) -> Result<Self, String> {
        let config: Yaml = serde_yaml::from_slice(bytes).map_err(|e| format!("{path}: {e}"))?;
        // Only the keys an engine deployment needs, at every level that can
        // change what runs.
        allow_keys(&config, allowed_keys::CONFIG, path, "the config")?;
        if let Some(shim) = config.get("shim") {
            allow_keys(shim, allowed_keys::SHIM, path, "shim")?;
        }

        let mut packs = Vec::new();
        if let Some(models) = config.get("models") {
            let models = models
                .as_sequence()
                .ok_or_else(|| format!("{path}: models must be a list"))?;
            for (i, model) in models.iter().enumerate() {
                let what = format!("models[{i}]");
                allow_keys(model, allowed_keys::MODEL, path, &what)?;
                let field = |key: &str| -> Result<Option<String>, String> {
                    match model.get(key) {
                        None => Ok(None),
                        Some(v) => v
                            .as_str()
                            .map(|s| Some(s.to_owned()))
                            .ok_or_else(|| format!("{path}: {what}.{key} must be a string")),
                    }
                };
                // The layout the pack was built with: stated, and one we know.
                match model.get("schema").and_then(Yaml::as_u64) {
                    Some(n) if PACK_SCHEMAS.contains(&n) => {}
                    _ => {
                        return Err(format!(
                            "{path}: {what}.schema must be one of {PACK_SCHEMAS:?}, the pack \
                             layouts whose root hash it pins"
                        ));
                    }
                }
                packs.push(Pack {
                    name: field("name")?.ok_or_else(|| format!("{path}: {what} has no name"))?,
                    repo: field("repo")?,
                    mpk: field("mpk")?,
                });
            }
        }

        let shim_paths = match config.get("shim").and_then(|shim| shim.get("paths")) {
            None => Vec::new(),
            Some(paths) => string_list(paths, path, "shim.paths")?,
        };
        let cvm_version = config
            .get("cvm-version")
            .and_then(Yaml::as_str)
            .ok_or_else(|| format!("{path}: cvm-version must be a string"))?;
        let release = parse_cvm_version(cvm_version).map_err(|e| format!("{path}: {e}"))?;
        if release.manifest_sha256.is_none() {
            return Err(format!(
                "{path}: cvm-version must pin its release manifest inline, \
                 as VERSION@sha256:<64 lowercase hex>"
            ));
        }

        let containers = config
            .get("containers")
            .and_then(Yaml::as_sequence)
            .ok_or_else(|| format!("{path}: containers must be a list"))?;
        // Exactly one container, the engine, pinned by full digest: MRCONFIGID
        // covers this file's bytes, not what a tag resolves to.
        let engine = match containers.as_slice() {
            [engine]
                if engine
                    .get("image")
                    .and_then(Yaml::as_str)
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
            Some(Yaml::String(s)) => Some(s.clone()),
            Some(Yaml::Number(n)) => Some(n.to_string()),
            Some(_) => {
                return Err(format!(
                    "{path}: the engine container's gpus must be a count or a selection"
                ));
            }
        };
        let secrets = match engine.get("secrets") {
            None | Some(Yaml::Null) => Vec::new(),
            Some(list) => string_list(list, path, "the engine container's secrets")?,
        };

        let mut env = BTreeMap::new();
        let entries = engine
            .get("env")
            .and_then(Yaml::as_sequence)
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
            .and_then(Yaml::as_u64)
            .ok_or_else(|| format!("{path}: shim.upstream-port must be a port number"))?;
        let whole = |key: &str| -> Result<u64, String> {
            config
                .get(key)
                .and_then(Yaml::as_u64)
                .ok_or_else(|| format!("{path}: {key} must be a whole number"))
        };
        let cpus = whole("cpus")?;
        let memory = whole("memory")?;
        let gpus = match config.get("gpus") {
            None => None,
            Some(_) => Some(whole("gpus")?),
        };

        Ok(Self {
            cvm_version: cvm_version.to_owned(),
            secrets,
            env,
            shim_upstream_port,
            shim_paths,
            gpus,
            runtime,
            container_gpus,
            packs,
            granted,
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
    tdx_policy: Option<Json>,
}

impl Sidecar {
    fn parse(bytes: &[u8], path: &str) -> Result<Self, String> {
        let value: Json = serde_json::from_slice(bytes).map_err(|e| format!("{path}: {e}"))?;
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
        let repo = weights["repo"]
            .as_str()
            .filter(|r| is_repo(r))
            .ok_or_else(|| {
                format!("{path}: weights.repo must be <owner>/<name> of [A-Za-z0-9._-]")
            })?;
        let revision = weights["revision"]
            .as_str()
            .filter(|r| is_lower_hex(r, 20))
            .ok_or_else(|| format!("{path}: weights.revision must be 40 lowercase hex digits"))?;
        let expected_gpus = match sidecar.get("expected_gpus") {
            None => None,
            Some(v) => Some(
                v.as_u64()
                    .filter(|n| u32::try_from(*n).is_ok())
                    .ok_or_else(|| format!("{path}: expected_gpus must be a u32"))?,
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
            repo: repo.to_owned(),
            revision: revision.to_owned(),
            expected_gpus,
            tdx_policy,
        })
    }
}

/// A YAML list of strings.
fn string_list(value: &Yaml, path: &str, what: &str) -> Result<Vec<String>, String> {
    value
        .as_sequence()
        .and_then(|l| l.iter().map(|s| s.as_str().map(str::to_owned)).collect())
        .ok_or_else(|| format!("{path}: {what} must be a list of strings"))
}

/// Refuse any key of the mapping `value` that `allowed` does not list.
fn allow_keys(value: &Yaml, allowed: &[&str], path: &str, what: &str) -> Result<(), String> {
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
    match first_disallowed_key(allowed, keys) {
        None => Ok(()),
        Some(key) => Err(format!(
            "{path}: {what} may not set {key:?}; an engine deployment uses only {allowed:?} there"
        )),
    }
}

fn object<'a>(value: &'a Json, at: &str) -> Result<&'a serde_json::Map<String, Json>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{at}: must be an object"))
}

fn exact_keys(
    object: &serde_json::Map<String, Json>,
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
