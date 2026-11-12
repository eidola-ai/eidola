//! The schema of `releases/trust/engine-enclaves.json` and its re-derivation
//! against the committed engine deployments it pins.
//!
//! This file is compiled twice: by `build.rs` (through `#[path]`), which runs
//! [`check`] over the committed file on every gateway build and refuses to
//! build on any disagreement, and by the crate's tests, which run the same
//! function over fixtures. It therefore uses nothing from the crate and only
//! dependencies that are both build- and dev-dependencies (`serde_json`,
//! `sha2`, `eidola-common` with its `deployment` feature, and
//! `tinfoil-verifier` with its `platform-endorsements` feature, whose pin
//! compiler is the one the attesting client runs).
//!
//! # What is checked
//!
//! - The file's shape, strictly (the runtime's typed parse first, then every
//!   member's width and form).
//! - Each model is listed under its weights hash, and each deployment's
//!   `config` names `deploy/engine/<name>/<variant>/tinfoil-config.yml` (the
//!   `<name>` a label for people, the identity the hash), and that file
//!   exists and hashes to `config_sha256`.
//! - **The deployment itself** — its config and its sidecar — by
//!   `eidola_common::engine_deployment::deployment::check_deployment`, the one
//!   check `measure-enclave` runs before it emits a pin. Every rule about
//!   what the config may say lives there.
//! - The pin agrees with that deployment: release, weights (hash, repo,
//!   revision), prompt-cache retention, GPU evidence count, and a TDX
//!   MRCONFIGID equal to the config's hash; and the attesting client's own
//!   pin compiler accepts it.
//! - **A TDX pin's machine policy is re-derived from the signed platform
//!   endorsements the sidecar names**: the vendored release file and its
//!   Sigstore bundle (`releases/trust/platform-endorsements/<tag>/`) are
//!   verified offline against the committed Sigstore trusted root
//!   (`tinfoil_verifier::platform_endorsements::verify`: the provider's
//!   release workflow at exactly that tag, the file's hash the one the
//!   sidecar records), the named policy and every machine mapped to it are
//!   derived, the sidecar's PCK flags added, and the pin's policy must be
//!   exactly that. `measure-enclave` makes the same two calls over the same
//!   bytes when it emits the pin.
//! - Every deployment listed under a hash serves those weights, and they
//!   agree on provenance, architecture and prompt-cache policy, since the
//!   gateway publishes one of each per hash:
//!   `deployment::check_model_agreement`, which `measure-enclave` also runs
//!   before it renders the file.
//! - Every deployment directory committed under `deploy/engine/` is pinned
//!   ([`require_every_deployment_pinned`]), so the file is a function of the
//!   tree in both directions.
//!
//! - **A TDX pin's MRTD is the platform release's statement of it**: the
//!   release manifest `cvm-version` pins, vendored under
//!   `releases/trust/cvm-releases/` (`deployment::release_manifest_file`), must
//!   hash to that pin, describe that release, state a launch in the measured
//!   model, and state the pin's MRTD (`deployment::check_release_manifest`,
//!   the rule `measure-enclave` reads the manifest by).
//!
//! What is **not** recomputed here: MRTD from the image, and a SEV-SNP launch
//! digest. Both need the platform provider's release images, so they are
//! release-time work (`measure-enclave`'s `measure-engine-enclaves`), and the
//! artifacts chain reruns the first.

use std::collections::BTreeMap;

use eidola_common::engine_deployment::{Architecture, CacheConfig, deployment, is_safe_component};
use sha2::{Digest, Sha256};

/// The schema version this gateway reads.
pub const SCHEMA_VERSION: u64 = 1;

/// Where engine deployment configs live, relative to the workspace root.
pub const DEPLOY_ROOT: &str = "deploy/engine";

/// A model the file pins, as checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedModel {
    /// The weights hash the model is listed under, which is also its
    /// weights' `sha256`.
    pub id: String,
    /// The architecture its deployments' sidecars name.
    pub architecture: Architecture,
    pub weights: CheckedWeights,
    pub prompt_cache: CheckedCachePolicy,
    /// The config path of every accepted deployment, in file order.
    pub deployments: Vec<String>,
    /// The longest sequence every one of its deployments serves: the
    /// smallest `EIDOLA_ENGINE_MAX_MODEL_LEN` among them.
    pub max_model_len: u32,
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
/// Returns the pinned models in weights-hash order. Any disagreement is an error naming
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
        if !is_lower_hex(id, 32) {
            return Err(format!(
                "{at}: a model is listed under its weights hash, 64 lowercase hex digits"
            ));
        }
        let deployments = deployments
            .as_array()
            .filter(|d| !d.is_empty())
            .ok_or_else(|| format!("{at}: must be a non-empty array of deployments"))?;
        let mut configs = Vec::new();
        let mut identities = Vec::new();
        let mut max_model_len = u32::MAX;
        for (index, deployment) in deployments.iter().enumerate() {
            let at = format!("{at}, deployment {index}");
            let (config, weights, prompt_cache, architecture, model_len) =
                check_deployment(deployment, &at, read)?;
            max_model_len = max_model_len.min(model_len);
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
                    architecture,
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
            architecture: identity.architecture,
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
            max_model_len,
        });
    }
    Ok(checked)
}

/// A catalog row's context window is one its deployments serve: `windows` is
/// each row's path, weights hash and `context_length`, and for every hash
/// that is pinned the window must be at most the smallest
/// `EIDOLA_ENGINE_MAX_MODEL_LEN` among its deployments. A wider window would
/// publish a length clients size prompts to and the engine refuses. A row
/// with no pin is the catalog binding's to refuse, not this check's.
pub fn check_context_windows(
    windows: &[(&str, &str, u64)],
    checked: &[CheckedModel],
) -> Result<(), String> {
    for &(path, sha256, context_length) in windows {
        if let Some(model) = checked.iter().find(|m| m.id == sha256)
            && context_length > u64::from(model.max_model_len)
        {
            return Err(format!(
                "{path}: context_length is {context_length}, but a deployment of {sha256} \
                 serves at most {} positions (EIDOLA_ENGINE_MAX_MODEL_LEN)",
                model.max_model_len
            ));
        }
    }
    Ok(())
}

/// Every committed deployment directory must be pinned: `committed` is the
/// config path of every `deploy/engine/<name>/<variant>/` directory in the
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
    deployment: &serde_json::Value,
    at: &str,
    read: &mut dyn FnMut(&str) -> Result<Vec<u8>, String>,
) -> Result<
    (
        String,
        CheckedWeights,
        CheckedCachePolicy,
        Architecture,
        u32,
    ),
    String,
> {
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
    let directory = config_path
        .strip_prefix(&format!("{DEPLOY_ROOT}/"))
        .and_then(|rest| rest.strip_suffix("/tinfoil-config.yml"))
        .and_then(|rest| rest.split_once('/'))
        .filter(|(name, variant)| is_safe_component(name) && is_safe_component(variant))
        .map(|(name, variant)| format!("{DEPLOY_ROOT}/{name}/{variant}"))
        .ok_or_else(|| {
            format!(
                "{at}: config must be {DEPLOY_ROOT}/<name>/<variant>/tinfoil-config.yml with \
                 <name> and <variant> matching [a-z0-9][a-z0-9._-]*"
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
    let sidecar_path = format!("{directory}/deployment.json");
    let sidecar_bytes = read(&sidecar_path)?;
    let checked =
        deployment::check_deployment(config_path, &config_bytes, &sidecar_path, &sidecar_bytes)
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
            let mrtd = hex_string(&tdx["mrtd"], 48, &format!("{at}: pin.platform.tdx.mrtd"))?;
            check_mrtd(
                mrtd,
                &checked.cvm_version,
                &format!("{at} ({config_path})"),
                read,
            )?;
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
            let machines = checked.tdx.as_ref().ok_or_else(|| {
                format!("{at}: a TDX pin, but {sidecar_path} states no tdx machines")
            })?;
            let derived = derive_tdx_policy(machines, read)
                .map_err(|e| format!("{at}: {sidecar_path}: tdx.platform_endorsements: {e}"))?;
            let derived = serde_json::to_value(&derived)
                .map_err(|e| format!("{at}: serializing the derived policy: {e}"))?;
            if tdx["policy"] != derived {
                return Err(format!(
                    "{at}: pin.platform.tdx.policy is not the policy derived from {sidecar_path}'s \
                     platform endorsements ({} {}): expected {derived}",
                    machines.release, machines.policy
                ));
            }
        }
        ["sev_snp"] => {
            let snp = object(&platform["sev_snp"], &format!("{at}: pin.platform.sev_snp"))?;
            exact_keys(
                snp,
                &["measurement", "host_data", "guest_policy"],
                &[],
                &format!("{at}: pin.platform.sev_snp"),
            )?;
            hex_string(
                &snp["measurement"],
                48,
                &format!("{at}: pin.platform.sev_snp.measurement"),
            )?;
            let host_data = hex_string(
                &snp["host_data"],
                32,
                &format!("{at}: pin.platform.sev_snp.host_data"),
            )?;
            if host_data != config_sha256 {
                return Err(format!(
                    "{at}: pin.platform.sev_snp.host_data is not the hash of {config_path} \
                     (expected {config_sha256})"
                ));
            }
            if checked.tdx.is_some() {
                return Err(format!(
                    "{at}: a SEV-SNP pin, but {sidecar_path} states tdx machines"
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

    Ok((
        config_path.to_owned(),
        weights,
        prompt_cache,
        checked.architecture,
        checked.measured.sizing.max_model_len,
    ))
}

/// A TDX pin's MRTD is the one the platform release states for the image
/// the config launches: the release's manifest, vendored under
/// `releases/trust/cvm-releases/`, must be the one `cvm-version` pins, and
/// its `igvm.tdx_launch.mrtd` must be the pin's. `measure-enclave` reads the
/// same manifest by the same rule, and checks the image reproduces it.
fn check_mrtd(
    mrtd: &str,
    cvm_version: &str,
    at: &str,
    read: &mut dyn FnMut(&str) -> Result<Vec<u8>, String>,
) -> Result<(), String> {
    let path = deployment::release_manifest_file(cvm_version).map_err(|e| format!("{at}: {e}"))?;
    let manifest = read(&path).map_err(|e| {
        format!(
            "{at}: {e}: the release manifest cvm-version {cvm_version:?} pins is not vendored \
             (deploy/engine/README.md, Platform release manifests)"
        )
    })?;
    let release = deployment::check_release_manifest(cvm_version, &manifest)
        .map_err(|e| format!("{at}: {path}: {e}"))?;
    if mrtd != release.mrtd {
        return Err(format!(
            "{at}: pin.platform.tdx.mrtd is {mrtd}, but {path} states {} for the image \
             {cvm_version:?} launches (re-measure with measure-enclave's measure-engine-enclaves)",
            release.mrtd
        ));
    }
    Ok(())
}

/// The machine policy `machines` names, from the committed platform
/// endorsements and Sigstore trusted root, verified offline.
fn derive_tdx_policy(
    machines: &deployment::TdxMachines,
    read: &mut dyn FnMut(&str) -> Result<Vec<u8>, String>,
) -> Result<tinfoil_verifier::TdxPolicy, String> {
    use tinfoil_verifier::PckFlag;
    use tinfoil_verifier::platform_endorsements::{PckFlags, verify};
    let flag = |f: deployment::PckFlag| match f {
        deployment::PckFlag::True => PckFlag::True,
        deployment::PckFlag::False => PckFlag::False,
        deployment::PckFlag::Undefined => PckFlag::Undefined,
    };
    let (file, bundle) = deployment::platform_endorsement_files(&machines.release);
    let verified = verify(
        &read(&file)?,
        &read(&bundle)?,
        &read(deployment::SIGSTORE_TRUSTED_ROOT)?,
        &machines.release,
        &machines.sha256,
    )
    .map_err(|e| e.to_string())?;
    verified
        .tdx_policy(
            &machines.policy,
            PckFlags {
                dynamic_platform: flag(machines.dynamic_platform),
                cached_keys: flag(machines.cached_keys),
                smt_enabled: flag(machines.smt_enabled),
            },
        )
        .map_err(|e| e.to_string())
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
