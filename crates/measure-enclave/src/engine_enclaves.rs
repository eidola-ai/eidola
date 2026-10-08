//! `releases/trust/engine-enclaves.json`: the engine deployments a gateway
//! build accepts, computed from the committed deployments.
//!
//! Each Eidola-hosted engine deployment is a directory
//! `deploy/engine/<model-id>/<variant>/` holding:
//!
//! - `tinfoil-config.yml` — the deployment's Tinfoil Container config, deployed
//!   and measured exactly as the gateway's own: the `eidola-server-engine`
//!   image by digest, the weights hash and storage mode, the prompt-cache
//!   retention, the gateway token's Argon2id hash. Its `cvm-version` pins the
//!   platform release manifest inline (`X@sha256:<hex>`).
//! - `deployment.json` — what a pin needs that the config does not determine:
//!   the weights' provenance (`repo`, `revision`), the GPU count the
//!   deployment presents evidence for, and the Intel TDX machine policy
//!   (`tinfoil_verifier::TdxPolicy`) its hosts are held to.
//!
//! [`deployment_entry`] turns one such directory, plus the release manifest
//! and TDX IGVM image its `cvm-version` names, into the file's entry: the
//! TDX launch identity (MRTD recomputed from the image, MRCONFIGID from the
//! config, via [`crate::tdx_igvm::release_pin`]) with the sidecar's policy,
//! and the weights and cache policy the config sets. The gateway's build
//! re-derives everything here except MRTD from the same committed files, so a
//! hand edit that disagrees with them fails that build.
//!
//! Only TDX deployments are measured here: the platform provider deploys
//! engines on its IGVM launch model, which has no SEV-SNP image. The file's
//! schema stays platform-tagged, so a SEV-SNP deployment remains expressible
//! when one exists.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::tdx_igvm;

/// The schema version emitted.
pub const SCHEMA_VERSION: u64 = 1;

/// One deployment's committed files, and the release artifacts its config
/// names.
pub struct DeploymentInputs<'a> {
    /// The config's path relative to the workspace root.
    pub config_path: &'a str,
    pub config: &'a [u8],
    pub sidecar: &'a [u8],
    /// The `tinfoil-inference-<version>-manifest.json` the config pins.
    pub release_manifest: &'a [u8],
    /// The release's `tinfoil-tdx-<version>.igvm`.
    pub igvm: &'a [u8],
}

/// The release artifacts a config needs: `(version, manifest SHA-256)` from
/// its inline `cvm-version` pin.
pub fn required_release(config: &[u8]) -> Result<(String, String)> {
    let release = tdx_igvm::config_release(config)?;
    let sha = release.manifest_sha256.context(
        "an engine deployment's cvm-version must pin its release manifest inline \
         (VERSION@sha256:<hex>)",
    )?;
    Ok((release.version, sha))
}

/// The `engine-enclaves.json` entry for one deployment of `model_id`.
pub fn deployment_entry(model_id: &str, inputs: &DeploymentInputs<'_>) -> Result<Value> {
    let config_yaml: serde_yaml::Value =
        serde_yaml::from_slice(inputs.config).context("parsing tinfoil-config.yml")?;
    let cvm_version = config_yaml
        .get("cvm-version")
        .and_then(serde_yaml::Value::as_str)
        .context("cvm-version must be a string")?
        .to_owned();
    let env = engine_env(&config_yaml)?;
    let var = |name: &str| {
        env.get(name)
            .cloned()
            .with_context(|| format!("the engine container's env does not set {name}"))
    };
    ensure!(
        var("EIDOLA_ENGINE_MODEL_ID")? == model_id,
        "{} serves a model other than {model_id}",
        inputs.config_path
    );
    // The node's own grammar for these values (`eidola_common::engine_deployment`).
    let enabled =
        eidola_common::engine_deployment::parse_prefix_cache(&var("EIDOLA_ENGINE_PREFIX_CACHE")?)
            .context("EIDOLA_ENGINE_PREFIX_CACHE must be true or false")?;
    let seconds = |name: &str| -> Result<u64> {
        eidola_common::engine_deployment::parse_cache_seconds(&var(name)?)
            .with_context(|| format!("{name} must be a positive number of seconds"))
    };

    let sidecar: Value =
        serde_json::from_slice(inputs.sidecar).context("parsing deployment.json")?;
    let tdx_policy = sidecar
        .get("tdx_policy")
        .context("deployment.json states no tdx_policy; only TDX deployments are measured")?;

    let (_, manifest_sha256) = required_release(inputs.config)?;
    let launch = tdx_igvm::release_pin(
        inputs.release_manifest,
        &manifest_sha256,
        inputs.igvm,
        inputs.config,
    )?;

    let mut pin = json!({
        "platform": {
            "tdx": {
                "mrtd": launch.mrtd,
                "mrconfigid": launch.mrconfigid,
                "policy": tdx_policy,
            }
        }
    });
    // The accelerators the prompt runs on are attested on every handshake,
    // by the rule the gateway's build applies too.
    let executor = match var("EIDOLA_ENGINE_EXECUTOR")?.as_str() {
        "cuda" => eidola_common::engine_deployment::Executor::Cuda,
        "cpu" => eidola_common::engine_deployment::Executor::Cpu,
        other => bail!("EIDOLA_ENGINE_EXECUTOR is {other:?}, not cpu or cuda"),
    };
    let gpus = match config_yaml.get("gpus") {
        None => None,
        Some(gpus) => Some(gpus.as_u64().context("gpus must be a whole number")?),
    };
    let expected_gpus = sidecar.get("expected_gpus").and_then(Value::as_u64);
    eidola_common::engine_deployment::check_gpu_attestation(executor, gpus, expected_gpus)
        .map_err(anyhow::Error::msg)?;
    let (runtime, container_gpus) = engine_gpu_access(&config_yaml)?;
    eidola_common::engine_deployment::check_container_gpu_access(
        executor,
        runtime.as_deref(),
        container_gpus.as_deref(),
    )
    .map_err(anyhow::Error::msg)?;
    if let Some(gpus) = sidecar.get("expected_gpus") {
        pin["expected_gpus"] = gpus.clone();
    }

    Ok(json!({
        "config": inputs.config_path,
        "config_sha256": hex::encode(Sha256::digest(inputs.config)),
        "cvm_version": cvm_version,
        "pin": pin,
        "weights": {
            "sha256": var("EIDOLA_ENGINE_WEIGHTS_SHA256")?,
            "repo": sidecar.pointer("/weights/repo").cloned().context("deployment.json has no weights.repo")?,
            "revision": sidecar.pointer("/weights/revision").cloned().context("deployment.json has no weights.revision")?,
        },
        "prompt_cache": {
            "enabled": enabled,
            "idle_ttl_secs": seconds("EIDOLA_ENGINE_CACHE_IDLE_TTL_SECS")?,
            "max_age_secs": seconds("EIDOLA_ENGINE_CACHE_MAX_AGE_SECS")?,
        },
    }))
}

/// The whole file: entries grouped by model id, each model's deployments in
/// config-path order.
pub fn render(entries: BTreeMap<String, Vec<Value>>) -> Result<String> {
    let mut models = serde_json::Map::new();
    for (model, mut deployments) in entries {
        deployments.sort_by(|a, b| a["config"].as_str().cmp(&b["config"].as_str()));
        models.insert(model, Value::Array(deployments));
    }
    let file = json!({ "schema_version": SCHEMA_VERSION, "models": models });
    Ok(serde_json::to_string_pretty(&file)? + "\n")
}

/// The single `eidola-server-engine` container's env, as name → value.
/// Refuse any key of the mapping `value` that `allowed` does not list.
fn allow_keys(value: &serde_yaml::Value, allowed: &[&str], what: &str) -> Result<()> {
    let mapping = value
        .as_mapping()
        .with_context(|| format!("{what} must be a mapping"))?;
    let keys = mapping
        .keys()
        .map(|k| {
            k.as_str()
                .with_context(|| format!("{what} has a non-string key"))
        })
        .collect::<Result<Vec<_>>>()?;
    if let Some(key) = eidola_common::engine_deployment::first_disallowed_key(allowed, keys) {
        bail!("{what} may not set {key:?}; an engine deployment uses only {allowed:?} there");
    }
    Ok(())
}

/// The engine container's `runtime` and GPU selection, as written.
fn engine_gpu_access(config: &serde_yaml::Value) -> Result<(Option<String>, Option<String>)> {
    let engine = &config["containers"][0];
    let runtime = engine
        .get("runtime")
        .map(|r| {
            r.as_str()
                .map(str::to_owned)
                .context("runtime must be a string")
        })
        .transpose()?;
    let gpus = match engine.get("gpus") {
        None => None,
        Some(serde_yaml::Value::String(s)) => Some(s.clone()),
        Some(serde_yaml::Value::Number(n)) => Some(n.to_string()),
        Some(_) => bail!("the engine container's gpus must be a count or a selection"),
    };
    Ok((runtime, gpus))
}

fn engine_env(config: &serde_yaml::Value) -> Result<BTreeMap<String, String>> {
    let containers = config
        .get("containers")
        .and_then(serde_yaml::Value::as_sequence)
        .context("containers must be a list")?;
    // The gateway's build holds the same rule: exactly one container, the
    // engine, pinned by a full digest (a tag escapes the measurement).
    let [engine] = containers.as_slice() else {
        bail!("an engine deployment runs exactly one container, the engine pinned by digest");
    };
    let digest = engine
        .get("image")
        .and_then(serde_yaml::Value::as_str)
        .and_then(|i| i.strip_prefix("ghcr.io/eidola-ai/eidola-server-engine@sha256:"));
    ensure!(
        digest.is_some_and(
            |d| d.len() == 64 && d.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        ),
        "the engine image must be pinned by a 64-hex-digit digest"
    );
    // Only the keys an engine deployment needs, as the gateway's build holds.
    use eidola_common::engine_deployment::allowed_keys;
    allow_keys(config, allowed_keys::CONFIG, "the config")?;
    if let Some(shim) = config.get("shim") {
        allow_keys(shim, allowed_keys::SHIM, "shim")?;
    }
    if let Some(models) = config.get("models") {
        for (i, model) in models
            .as_sequence()
            .context("models must be a list")?
            .iter()
            .enumerate()
        {
            allow_keys(model, allowed_keys::MODEL, &format!("models[{i}]"))?;
        }
    }
    allow_keys(engine, allowed_keys::CONTAINER, "the engine container")?;
    let mut env = BTreeMap::new();
    for entry in engine
        .get("env")
        .and_then(serde_yaml::Value::as_sequence)
        .context("the engine container's env must be a list")?
    {
        let (name, value) = entry
            .as_mapping()
            .filter(|m| m.len() == 1)
            .and_then(|m| m.iter().next())
            .and_then(|(k, v)| Some((k.as_str()?, v.as_str()?)))
            .context("each env entry must be one NAME: \"value\" pair")?;
        ensure!(
            env.insert(name.to_owned(), value.to_owned()).is_none(),
            "env sets {name} twice"
        );
    }
    Ok(env)
}

#[cfg(test)]
mod tests {
    use igvm_defs::IgvmPageDataFlags;

    use super::*;
    use crate::tdx_igvm::tests::{image, manifest, page};

    const PATH: &str = "deploy/engine/fixture-model/tdx-a/tinfoil-config.yml";

    fn config(manifest_sha: &str) -> String {
        format!(
            r#"cvm-version: 0.0.0-test@sha256:{manifest_sha}
cpus: 16
memory: 65536
gpus: 8
containers:
  - name: "eidola-server-engine"
    image: "ghcr.io/eidola-ai/eidola-server-engine@sha256:{digest}"
    runtime: nvidia
    gpus: all
    secrets:
      - GATEWAY_TOKEN
    env:
      - EIDOLA_ENGINE_MODEL_ID: "fixture-model"
      - EIDOLA_ENGINE_WEIGHTS_SHA256: "{weights}"
      - EIDOLA_ENGINE_WEIGHTS_STORAGE: "verified-readonly"
      - EIDOLA_ENGINE_EXECUTOR: "cuda"
      - GATEWAY_TOKEN_HASH: "$argon2id$v=19$m=19456,t=2,p=1$c2FsdHNhbHRzYWx0$aGFzaGhhc2hoYXNoaGFzaA"
      - EIDOLA_ENGINE_PREFIX_CACHE: "true"
      - EIDOLA_ENGINE_CACHE_IDLE_TTL_SECS: "900"
      - EIDOLA_ENGINE_CACHE_MAX_AGE_SECS: "7200"
"#,
            digest = "2".repeat(64),
            weights = "3".repeat(64),
        )
    }

    const SIDECAR: &str = r#"{
        "weights": {"repo": "example/fixture-model", "revision": "4444444444444444444444444444444444444444"},
        "expected_gpus": 8,
        "tdx_policy": {"mr_seam": ["55"], "smt_enabled": "false"}
    }"#;

    #[test]
    fn an_entry_carries_the_launch_identity_and_what_the_config_sets() {
        let igvm = image(vec![page(0x2000, 1, IgvmPageDataFlags::new())]);
        let mrtd = hex::encode(tdx_igvm::mrtd_from_igvm(&igvm).unwrap());
        let release = manifest(&igvm, &mrtd, &"00".repeat(48));
        let release_sha = hex::encode(Sha256::digest(&release));
        let config = config(&release_sha);

        assert_eq!(
            required_release(config.as_bytes()).unwrap(),
            ("0.0.0-test".to_string(), release_sha.clone())
        );
        let entry = deployment_entry(
            "fixture-model",
            &DeploymentInputs {
                config_path: PATH,
                config: config.as_bytes(),
                sidecar: SIDECAR.as_bytes(),
                release_manifest: &release,
                igvm: &igvm,
            },
        )
        .unwrap();

        let config_sha = hex::encode(Sha256::digest(config.as_bytes()));
        assert_eq!(
            entry,
            json!({
                "config": PATH,
                "config_sha256": config_sha,
                "cvm_version": format!("0.0.0-test@sha256:{release_sha}"),
                "pin": {
                    "platform": {"tdx": {
                        "mrtd": mrtd,
                        "mrconfigid": format!("{config_sha}{}", "00".repeat(16)),
                        "policy": {"mr_seam": ["55"], "smt_enabled": "false"},
                    }},
                    "expected_gpus": 8,
                },
                "weights": {
                    "sha256": "3".repeat(64),
                    "repo": "example/fixture-model",
                    "revision": "4".repeat(40),
                },
                "prompt_cache": {"enabled": true, "idle_ttl_secs": 900, "max_age_secs": 7200},
            })
        );

        let rendered =
            render(BTreeMap::from([("fixture-model".to_string(), vec![entry])])).unwrap();
        let parsed: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(parsed["schema_version"], 1);
        assert_eq!(parsed["models"]["fixture-model"][0]["config"], PATH);
    }

    /// The committed file is in this tool's canonical form: re-rendering
    /// what it holds reproduces it byte for byte. With no deployment
    /// committed that is the whole file; with some, the gateway's build
    /// re-derives every value but MRTD from the committed deployments.
    #[test]
    fn the_committed_file_is_in_canonical_form() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../releases/trust/engine-enclaves.json");
        let committed = std::fs::read_to_string(path).unwrap();
        let parsed: Value = serde_json::from_str(&committed).unwrap();
        let models: BTreeMap<String, Vec<Value>> =
            serde_json::from_value(parsed["models"].clone()).unwrap();
        assert_eq!(render(models).unwrap(), committed);
    }

    #[test]
    fn a_deployment_that_cannot_be_measured_is_refused() {
        let igvm = image(vec![page(0x2000, 1, IgvmPageDataFlags::new())]);
        let mrtd = hex::encode(tdx_igvm::mrtd_from_igvm(&igvm).unwrap());
        let release = manifest(&igvm, &mrtd, &"00".repeat(48));
        let release_sha = hex::encode(Sha256::digest(&release));
        let entry = |model: &str, config: &str, sidecar: &str| {
            deployment_entry(
                model,
                &DeploymentInputs {
                    config_path: PATH,
                    config: config.as_bytes(),
                    sidecar: sidecar.as_bytes(),
                    release_manifest: &release,
                    igvm: &igvm,
                },
            )
            .unwrap_err()
            .to_string()
        };
        let good = config(&release_sha);

        assert!(entry("other-model", &good, SIDECAR).contains("serves a model other than"));
        assert!(
            entry("fixture-model", &good, r#"{"weights": {}}"#).contains("only TDX deployments")
        );
        let unpinned = good.replace(&format!("@sha256:{release_sha}"), "");
        assert!(entry("fixture-model", &unpinned, SIDECAR).contains("pin its release manifest"));
        let other_release = good.replace(&release_sha, &"ab".repeat(32));
        assert!(entry("fixture-model", &other_release, SIDECAR).contains("not the pinned"));

        // A CUDA deployment that asks for no GPU evidence, or a different
        // count than it attaches.
        let no_gpus = SIDECAR.replace("\"expected_gpus\": 8,", "");
        assert!(entry("fixture-model", &good, &no_gpus).contains("expected_gpus must be 8"));
        let zero_gpus = SIDECAR.replace("\"expected_gpus\": 8", "\"expected_gpus\": 0");
        assert!(entry("fixture-model", &good, &zero_gpus).contains("expected_gpus must be 8"));
        let unstated = good.replace("gpus: 8\n", "");
        assert!(
            entry("fixture-model", &unstated, SIDECAR).contains("needs the config to attach GPUs")
        );
        let two = good.replace("gpus: 8\n", "gpus: 2\n");
        assert!(entry("fixture-model", &two, SIDECAR).contains("not an NVIDIA-CC shape"));

        // A companion container, or an engine named by a tag.
        let companion =
            format!("{good}  - name: \"sidecar\"\n    image: \"ghcr.io/example/sidecar:latest\"\n");
        assert!(entry("fixture-model", &companion, SIDECAR).contains("exactly one container"));
        // A key an engine deployment does not need.
        let entrypoint = good.replace(
            "    runtime: nvidia\n",
            "    runtime: nvidia\n    entrypoint: [\"/bin/sh\"]\n",
        );
        assert!(
            entry("fixture-model", &entrypoint, SIDECAR).contains("may not set \"entrypoint\"")
        );
        let no_runtime = good.replace("    runtime: nvidia\n", "");
        assert!(entry("fixture-model", &no_runtime, SIDECAR).contains("runtime: nvidia"));
        let tagged = good.replace(&format!("@sha256:{}", "2".repeat(64)), ":v1");
        assert!(entry("fixture-model", &tagged, SIDECAR).contains("64-hex-digit digest"));
    }
}
