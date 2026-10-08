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

use super::{
    ExecutorSettings, MeasuredConfig, ModelPack, WeightsStorage, artifact_data_bytes,
    check_container_gpu_access, check_env_names, check_gpu_attestation, check_kernels_dir,
    check_resources, check_secrets, check_shim_paths, check_vm_resources, check_weights_pack, env,
    parse_cvm_version, parse_measured,
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
    // A pinned deployment runs the CUDA executor: the CPU executor is the
    // numerics reference and the development path, and nothing confidential
    // ships on it.
    let ExecutorSettings::Cuda { kernels_dir, .. } = &measured.executor else {
        return Err(format!(
            "{at}: a pinned deployment runs the cuda executor; cpu is the reference and \
             development path"
        ));
    };
    check_kernels_dir(kernels_dir).map_err(|e| format!("{at}: {e}"))?;
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
    check_gpu_attestation(measured.executor.kind(), c.gpus, side.expected_gpus)
        .map_err(|e| format!("{at}: {e}"))?;
    check_container_gpu_access(
        measured.executor.kind(),
        c.runtime.as_deref(),
        c.container_gpus.as_deref(),
    )
    .map_err(|e| format!("{at}: {e}"))?;

    // What the node allocates fits the VM and the GPUs it attaches.
    // `check_weights_pack` held the config to one pack with a whole artifact
    // reference, so its data size bounds the weights.
    let weights_bytes = c
        .packs
        .first()
        .and_then(|p| p.mpk.as_deref())
        .and_then(artifact_data_bytes)
        .ok_or_else(|| format!("{at}: the weights pack has no artifact reference"))?;
    check_resources(
        &measured.sizing,
        &measured.cache,
        &measured.executor,
        weights_bytes,
        c.memory,
        c.gpus.unwrap_or(0),
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

/// Largest config Tinfoil decodes (`tinfoil-config/decode.go`:
/// `MaxConfigBytes = 1 << 20`, checked on the raw bytes before parsing).
pub const MAX_CONFIG_BYTES: usize = 1 << 20;

/// The engine container's name. Tinfoil's validator checks only that names
/// are unique; an engine deployment has one container, and it is this one.
pub const ENGINE_CONTAINER_NAME: &str = "eidola-server-engine";

// The typed mirror of Tinfoil's config schema (`tinfoil-config/types.go`),
// restricted to the fields an engine deployment needs. Every struct refuses
// unknown fields, so a field Tinfoil supports but we do not allow — a
// container's `entrypoint` or `command` (another executable from the pinned
// image), `volumes`, `devices`, `cap_add`, `privileged` (wider reach),
// `networks` or `cvm-network` (egress, inbound ports), the shim's
// `dummy-attestation` (no hardware evidence), a pack's `exec`, `emwp` or
// `key-secret` — fails the parse. Field types follow Tinfoil's: an `int`
// there is an integer here (a quoted number is refused, as yaml.v3 refuses
// it), a `string` a string. Widening the allow-list means adding the typed
// field here and its validation below together.

/// `Config` (types.go), top level.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TinfoilConfig {
    /// The platform release, with its manifest pinned inline.
    #[serde(rename = "cvm-version")]
    cvm_version: String,
    /// VM resources (`int` in types.go).
    cpus: u64,
    memory: u64,
    /// GPUs attached to the VM (`int`; validation.go: 0 to 8).
    #[serde(default)]
    gpus: Option<u64>,
    /// The read-only model packs that carry the weights to verified storage.
    #[serde(default)]
    models: Vec<TinfoilModel>,
    containers: Vec<TinfoilContainer>,
    shim: TinfoilShim,
}

/// `ModelSpec` (types.go): its name, source (`repo`), the pack pinned by root
/// hash (`mpk`) and its layout (`schema`, required here).
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TinfoilModel {
    name: String,
    #[serde(default)]
    repo: Option<String>,
    #[serde(default)]
    mpk: Option<String>,
    #[serde(default)]
    schema: Option<i64>,
}

/// `Container` (types.go): identity, the digest-pinned image, the measured
/// env and the one secret, the weights pack grant, and GPU access.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TinfoilContainer {
    name: String,
    image: String,
    /// One-key `NAME: "value"` mappings. Tinfoil also accepts a bare `NAME`
    /// (inherit from the host) and non-string scalars; neither is measured
    /// text, so neither is accepted here.
    env: Vec<BTreeMap<String, QuotedString>>,
    #[serde(default)]
    secrets: Option<Vec<String>>,
    #[serde(default)]
    models: Vec<String>,
    #[serde(default)]
    runtime: Option<String>,
    #[serde(default)]
    gpus: Option<GpuSelection>,
}

/// A YAML string scalar, and only that: `true`, `16` or `0.5` unquoted are
/// other types, which a plain `String` would silently coerce. An env value is
/// the text the node parses, so it is written as text.
struct QuotedString(String);

impl<'de> serde::Deserialize<'de> for QuotedString {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        match <serde_yaml::Value as serde::Deserialize>::deserialize(d)? {
            serde_yaml::Value::String(s) => Ok(Self(s)),
            _ => Err(serde::de::Error::custom(
                "each env entry must be one NAME: \"value\" pair with a quoted string value",
            )),
        }
    }
}

/// A container's `gpus` (`interface{}` in types.go): a count or a selection.
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum GpuSelection {
    Count(u64),
    Selection(String),
}

/// `ShimConfig` (shim.go), restricted to where it forwards and what it serves.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TinfoilShim {
    #[serde(rename = "upstream-port")]
    upstream_port: u64,
    #[serde(default)]
    paths: Vec<String>,
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

/// An environment name as Tinfoil's validator reads one
/// (validation.go `validEnvironmentName`): 1 to 256 bytes of `[A-Za-z_]`, and
/// digits after the first.
fn is_environment_name(name: &str) -> bool {
    (1..=256).contains(&name.len())
        && name
            .bytes()
            .enumerate()
            .all(|(i, b)| b == b'_' || b.is_ascii_alphabetic() || (i > 0 && b.is_ascii_digit()))
}

impl EngineConfig {
    fn parse(bytes: &[u8], path: &str) -> Result<Self, String> {
        // decode.go: `len(data) > MaxConfigBytes` is refused before parsing.
        if bytes.len() > MAX_CONFIG_BYTES {
            return Err(format!(
                "{path}: {} bytes exceeds Tinfoil's {MAX_CONFIG_BYTES}-byte config limit",
                bytes.len()
            ));
        }
        super::yaml_events::refuse_divergent_constructs(bytes, path)?;
        // One document, every struct strict (decode.go: `KnownFields(true)`,
        // a trailing document refused), duplicate keys refused.
        let config: TinfoilConfig =
            serde_yaml::from_slice(bytes).map_err(|e| format!("{path}: {e}"))?;

        let release = parse_cvm_version(&config.cvm_version).map_err(|e| format!("{path}: {e}"))?;
        if release.manifest_sha256.is_none() {
            return Err(format!(
                "{path}: cvm-version must pin its release manifest inline, \
                 as VERSION@sha256:<64 lowercase hex>"
            ));
        }
        // validation.go `Validate`: `gpus must be between 0 and 8`.
        if config.gpus.is_some_and(|n| n > 8) {
            return Err(format!("{path}: gpus must be between 0 and 8"));
        }
        // shim.go `Validate`: `upstream port is not set` (zero); a port fits
        // in sixteen bits.
        if !(1..=u64::from(u16::MAX)).contains(&config.shim.upstream_port) {
            return Err(format!("{path}: shim.upstream-port must be a port number"));
        }

        let mut packs = Vec::new();
        for (i, model) in config.models.into_iter().enumerate() {
            // validation.go `validateShape`: `schema must be a positive
            // integer`; our rule is stricter: stated, and a layout we know.
            if !model
                .schema
                .and_then(|n| u64::try_from(n).ok())
                .is_some_and(|n| PACK_SCHEMAS.contains(&n))
            {
                return Err(format!(
                    "{path}: models[{i}].schema must be one of {PACK_SCHEMAS:?}, the pack \
                     layouts whose root hash it pins"
                ));
            }
            packs.push(Pack {
                name: model.name,
                repo: model.repo,
                mpk: model.mpk,
            });
        }

        // Exactly one container, the engine, pinned by full digest: MRCONFIGID
        // covers this file's bytes, not what a tag resolves to (validation.go
        // `validateContainerImage` requires a digest; we require ours).
        let mut containers = config.containers;
        let engine = match containers.as_mut_slice() {
            [engine]
                if engine.name == ENGINE_CONTAINER_NAME
                    && engine
                        .image
                        .strip_prefix(ENGINE_IMAGE_PREFIX)
                        .is_some_and(|digest| is_lower_hex(digest, 32)) =>
            {
                containers.pop().expect("one container")
            }
            _ => {
                return Err(format!(
                    "{path}: an engine deployment runs exactly one container, named \
                     {ENGINE_CONTAINER_NAME:?}, its image {ENGINE_IMAGE_PREFIX}<64 lowercase \
                     hex>; every other container, and any image named by a tag or a short \
                     digest, escapes the measurement"
                ));
            }
        };

        // validation.go `validateContainer`: each env entry one key, each
        // name a valid environment name.
        let mut env = BTreeMap::new();
        for entry in engine.env {
            if entry.len() != 1 {
                return Err(format!(
                    "{path}: each env entry must be one NAME: \"value\" pair with a quoted \
                     string value"
                ));
            }
            let (key, value) = entry.into_iter().next().expect("one key");
            if !is_environment_name(&key) {
                return Err(format!(
                    "{path}: env has an invalid environment name {key:?}"
                ));
            }
            if env.contains_key(&key) {
                return Err(format!("{path}: env sets {key} twice"));
            }
            env.insert(key, value.0);
        }
        let secrets = engine.secrets.unwrap_or_default();
        // validation.go `validateContainer`: secrets are environment names.
        if let Some(bad) = secrets.iter().find(|s| !is_environment_name(s)) {
            return Err(format!(
                "{path}: secrets has an invalid environment name {bad:?}"
            ));
        }
        // validation.go `validateModelAccess`: a grant is listed once.
        for (i, name) in engine.models.iter().enumerate() {
            if engine.models[..i].contains(name) {
                return Err(format!(
                    "{path}: the engine container's models grant {name:?} twice"
                ));
            }
        }
        // validation.go `validateContainerPolicy`: runtime is empty or nvidia.
        if engine.runtime.as_deref().is_some_and(|r| r != "nvidia") {
            return Err(format!(
                "{path}: the engine container's runtime must be nvidia"
            ));
        }
        let container_gpus = engine.gpus.map(|g| match g {
            GpuSelection::Count(n) => n.to_string(),
            GpuSelection::Selection(s) => s,
        });

        Ok(Self {
            cvm_version: config.cvm_version,
            secrets,
            env,
            shim_upstream_port: config.shim.upstream_port,
            shim_paths: config.shim.paths,
            gpus: config.gpus,
            runtime: engine.runtime,
            container_gpus,
            packs,
            granted: engine.models,
            cpus: config.cpus,
            memory: config.memory,
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

// The typed shape of `deployment.json`. Every level refuses unknown members,
// and serde's derived struct parse refuses a repeated member (`duplicate
// field`), so a file whose meaning depends on which of two equal keys a
// reader keeps never parses.

/// `deployment.json`, top level.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SidecarFile {
    weights: SidecarWeights,
    #[serde(default, deserialize_with = "present")]
    expected_gpus: Option<u32>,
    #[serde(default, deserialize_with = "present")]
    tdx_policy: Option<TdxPolicy>,
}

/// An optional member that, when present, holds a value: absent is `None`,
/// and `null` is refused rather than read as absent.
fn present<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    T::deserialize(d).map(Some)
}

/// Where the weight files came from.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SidecarWeights {
    repo: String,
    revision: String,
}

/// The machine-side Intel TDX policy, in the attesting client's shape
/// (`tinfoil_verifier::TdxPolicy`, member for member): the shape is held
/// here, and the values (widths, MR_SEAM non-empty, safe attributes) by the
/// client's own pin compiler, which the gateway's build runs over the pin.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct TdxPolicy {
    mr_seam: Vec<String>,
    td_attributes: String,
    xfam: String,
    minimum_tee_tcb_svn: String,
    minimum_tcb_evaluation_data_number: u32,
    qe_vendor_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fmspc: Option<Vec<String>>,
    dynamic_platform: PckFlag,
    cached_keys: PckFlag,
    smt_enabled: PckFlag,
}

/// A PCK certificate platform flag's expected value
/// (`tinfoil_verifier::PckFlag`).
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum PckFlag {
    True,
    False,
    Undefined,
}

impl Sidecar {
    fn parse(bytes: &[u8], path: &str) -> Result<Self, String> {
        let file: SidecarFile =
            serde_json::from_slice(bytes).map_err(|e| format!("{path}: {e}"))?;
        if !is_repo(&file.weights.repo) {
            return Err(format!(
                "{path}: weights.repo must be <owner>/<name> of [A-Za-z0-9._-]"
            ));
        }
        if !is_lower_hex(&file.weights.revision, 20) {
            return Err(format!(
                "{path}: weights.revision must be 40 lowercase hex digits"
            ));
        }
        let tdx_policy = file
            .tdx_policy
            .map(|p| serde_json::to_value(p).map_err(|e| format!("{path}: tdx_policy: {e}")))
            .transpose()?;
        Ok(Self {
            repo: file.weights.repo,
            revision: file.weights.revision,
            expected_gpus: file.expected_gpus.map(u64::from),
            tdx_policy,
        })
    }
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
