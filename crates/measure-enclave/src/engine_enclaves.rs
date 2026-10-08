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

use anyhow::{Context, Result};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use eidola_common::engine_deployment::deployment::{
    ModelIdentity, check_deployment, check_model_agreement,
};

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
///
/// The deployment is checked by `eidola_common::engine_deployment::deployment::
/// check_deployment`, the one check the gateway's build also runs on every
/// pinned deployment, so the generator can only emit a pin the build accepts.
/// What only this tool adds is the launch identity from the platform release.
pub fn deployment_entry(model_id: &str, inputs: &DeploymentInputs<'_>) -> Result<Entry> {
    let sidecar_path = inputs
        .config_path
        .strip_suffix("tinfoil-config.yml")
        .map(|dir| format!("{dir}deployment.json"))
        .context("the config path must end in tinfoil-config.yml")?;
    let checked = check_deployment(
        model_id,
        inputs.config_path,
        inputs.config,
        &sidecar_path,
        inputs.sidecar,
    )
    .map_err(anyhow::Error::msg)?;
    let identity = checked.model_identity();
    let tdx_policy = checked
        .tdx_policy
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
    if let Some(gpus) = checked.expected_gpus {
        pin["expected_gpus"] = gpus.into();
    }
    let cache = checked.measured.cache;
    let json = json!({
        "config": inputs.config_path,
        "config_sha256": hex::encode(Sha256::digest(inputs.config)),
        "cvm_version": checked.cvm_version,
        "pin": pin,
        "weights": {
            "sha256": checked.measured.weights_sha256,
            "repo": checked.weights_repo,
            "revision": checked.weights_revision,
        },
        "prompt_cache": {
            "enabled": cache.enabled,
            "idle_ttl_secs": cache.idle_ttl_secs,
            "max_age_secs": cache.max_age_secs,
        },
    });
    Ok(Entry {
        config_path: inputs.config_path.to_owned(),
        json,
        identity,
    })
}

/// One deployment's entry, with what it must share with its model's other
/// deployments.
#[derive(Debug)]
pub struct Entry {
    pub config_path: String,
    pub json: Value,
    pub identity: ModelIdentity,
}

/// The whole file, from every model's checked entries. Each model's
/// deployments are first held to agree by
/// `eidola_common::engine_deployment::deployment::check_model_agreement`, the
/// per-model rule the gateway's build also runs, so the file rendered is one
/// that build accepts.
pub fn render(entries: BTreeMap<String, Vec<Entry>>) -> Result<String> {
    let mut models = BTreeMap::new();
    for (model, deployments) in entries {
        check_model_agreement(
            &model,
            deployments
                .iter()
                .map(|e| (e.config_path.as_str(), &e.identity)),
        )
        .map_err(anyhow::Error::msg)?;
        models.insert(model, deployments.into_iter().map(|e| e.json).collect());
    }
    render_values(models)
}

/// The file's canonical text: entries grouped by model id, each model's
/// deployments in config-path order.
fn render_values(entries: BTreeMap<String, Vec<Value>>) -> Result<String> {
    let mut models = serde_json::Map::new();
    for (model, mut deployments) in entries {
        deployments.sort_by(|a, b| a["config"].as_str().cmp(&b["config"].as_str()));
        models.insert(model, Value::Array(deployments));
    }
    let file = json!({ "schema_version": SCHEMA_VERSION, "models": models });
    Ok(serde_json::to_string_pretty(&file)? + "\n")
}

/// The single `eidola-server-engine` container's env, as name → value.
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
models:
  - name: "weights"
    repo: "example/fixture-model@4444444444444444444444444444444444444444"
    mpk: "{root}_17419419648_3892cd2f-a06e-5aee-8276-93140b9f06ec"
    schema: 2
shim:
  upstream-port: 8080
  paths:
    - /*
containers:
  - name: "eidola-server-engine"
    image: "ghcr.io/eidola-ai/eidola-server-engine@sha256:{digest}"
    runtime: nvidia
    gpus: all
    models: ["weights"]
    secrets:
      - GATEWAY_TOKEN
    env:
      - EIDOLA_ENGINE_MODEL_ID: "fixture-model"
      - EIDOLA_ENGINE_WEIGHTS_DIR: "/tinfoil/models/weights"
      - EIDOLA_ENGINE_WEIGHTS_SHA256: "{weights}"
      - EIDOLA_ENGINE_WEIGHTS_STORAGE: "verified-readonly"
      - EIDOLA_ENGINE_EXECUTOR: "cuda"
      - EIDOLA_ENGINE_BIND_ADDR: "0.0.0.0:8080"
      - EIDOLA_ENGINE_KV_BLOCK_SIZE: "16"
      - EIDOLA_ENGINE_KV_DEVICE_BYTES: "68719476736"
      - EIDOLA_ENGINE_KERNELS_DIR: "/opt/eidola/kernels"
      - EIDOLA_ENGINE_CUDA_GRAPHS: "on"
      - EIDOLA_ENGINE_MAX_MODEL_LEN: "131072"
      - EIDOLA_ENGINE_MAX_SEQS: "64"
      - EIDOLA_ENGINE_MAX_BATCHED_TOKENS: "8192"
      - EIDOLA_ENGINE_MAX_PREFILL_CHUNK: "4096"
      - EIDOLA_ENGINE_DRAFT_TOKENS: "0"
      - EIDOLA_ENGINE_MAX_REQUESTS: "8"
      - GATEWAY_TOKEN_HASH: "$argon2id$v=19$m=19456,t=2,p=1$Mz9P1/uk98yKEflNjzvn5g$unNYT/KTNSNW0JCH9+9OQ2zBApPLxGNZiw746903Q8E"
      - EIDOLA_ENGINE_PREFIX_CACHE: "true"
      - EIDOLA_ENGINE_CACHE_IDLE_TTL_SECS: "900"
      - EIDOLA_ENGINE_CACHE_MAX_AGE_SECS: "7200"
"#,
            digest = "2".repeat(64),
            weights = "3".repeat(64),
            root = "5".repeat(64),
        )
    }

    const SIDECAR: &str = r#"{
        "weights": {"repo": "example/fixture-model", "revision": "4444444444444444444444444444444444444444"},
        "expected_gpus": 8,
        "tdx_policy": {
            "mr_seam": ["55"],
            "td_attributes": "0000001000000000",
            "xfam": "e702060000000000",
            "minimum_tee_tcb_svn": "06010300000000000000000000000000",
            "minimum_tcb_evaluation_data_number": 17,
            "qe_vendor_id": "939a7233f79c4ca9940a0db3957f0607",
            "dynamic_platform": "true",
            "cached_keys": "true",
            "smt_enabled": "false"
        }
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
            entry.json,
            json!({
                "config": PATH,
                "config_sha256": config_sha,
                "cvm_version": format!("0.0.0-test@sha256:{release_sha}"),
                "pin": {
                    "platform": {"tdx": {
                        "mrtd": mrtd,
                        "mrconfigid": format!("{config_sha}{}", "00".repeat(16)),
                        "policy": serde_json::from_str::<serde_json::Value>(SIDECAR).unwrap()["tdx_policy"],
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

    /// Every variant of a model shares its weights and prompt-cache policy:
    /// the generator refuses to render a file whose variants disagree, by the
    /// same rule the gateway's build holds the file to.
    #[test]
    fn the_variants_of_a_model_agree() {
        let igvm = image(vec![page(0x2000, 1, IgvmPageDataFlags::new())]);
        let mrtd = hex::encode(tdx_igvm::mrtd_from_igvm(&igvm).unwrap());
        let release = manifest(&igvm, &mrtd, &"00".repeat(48));
        let release_sha = hex::encode(Sha256::digest(&release));
        let good = config(&release_sha);
        let entry = |path: &str, config: &str, sidecar: &str| {
            deployment_entry(
                "fixture-model",
                &DeploymentInputs {
                    config_path: path,
                    config: config.as_bytes(),
                    sidecar: sidecar.as_bytes(),
                    release_manifest: &release,
                    igvm: &igvm,
                },
            )
            .unwrap()
        };
        const OTHER: &str = "deploy/engine/fixture-model/tdx-b/tinfoil-config.yml";
        let two = |b: Entry| {
            render(BTreeMap::from([(
                "fixture-model".to_string(),
                vec![entry(PATH, &good, SIDECAR), b],
            )]))
        };

        let rendered = two(entry(OTHER, &good, SIDECAR)).unwrap();
        let parsed: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(parsed["models"]["fixture-model"][1]["config"], OTHER);

        let other_weights = good.replace(&"3".repeat(64), &"6".repeat(64));
        let err = two(entry(OTHER, &other_weights, SIDECAR))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(OTHER) && err.contains("weights differ"),
            "{err}"
        );

        let other_revision = SIDECAR.replace(&"4".repeat(40), &"7".repeat(40));
        let other_pack = good.replace(&"4".repeat(40), &"7".repeat(40));
        let err = two(entry(OTHER, &other_pack, &other_revision))
            .unwrap_err()
            .to_string();
        assert!(err.contains("weights differ"), "{err}");

        let other_cache = good.replace(
            "CACHE_MAX_AGE_SECS: \"7200\"",
            "CACHE_MAX_AGE_SECS: \"3600\"",
        );
        let err = two(entry(OTHER, &other_cache, SIDECAR))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(OTHER) && err.contains("prompt_cache differs"),
            "{err}"
        );
        let no_cache = good.replace("PREFIX_CACHE: \"true\"", "PREFIX_CACHE: \"false\"");
        let err = two(entry(OTHER, &no_cache, SIDECAR))
            .unwrap_err()
            .to_string();
        assert!(err.contains("prompt_cache differs"), "{err}");
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
        assert_eq!(render_values(models).unwrap(), committed);
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

        assert!(entry("other-model", &good, SIDECAR).contains("but the deployment is for"));
        let no_policy = r#"{
            "weights": {"repo": "example/fixture-model", "revision": "4444444444444444444444444444444444444444"},
            "expected_gpus": 8
        }"#;
        assert!(entry("fixture-model", &good, no_policy).contains("only TDX deployments"));
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
        let companion = format!(
            "{good}  - name: \"sidecar\"\n    image: \"ghcr.io/example/sidecar:latest\"\n    env: []\n"
        );
        assert!(entry("fixture-model", &companion, SIDECAR).contains("exactly one container"));
        // A key an engine deployment does not need.
        let entrypoint = good.replace(
            "    runtime: nvidia\n",
            "    runtime: nvidia\n    entrypoint: [\"/bin/sh\"]\n",
        );
        assert!(
            entry("fixture-model", &entrypoint, SIDECAR).contains("unknown field `entrypoint`")
        );
        let no_runtime = good.replace("    runtime: nvidia\n", "");
        assert!(entry("fixture-model", &no_runtime, SIDECAR).contains("runtime: nvidia"));
        // Another secret, a partial shim, an ungranted pack.
        let extra_secret = good.replace(
            "      - GATEWAY_TOKEN\n",
            "      - GATEWAY_TOKEN\n      - RUST_LOG\n",
        );
        assert!(entry("fixture-model", &extra_secret, SIDECAR).contains("secrets must be exactly"));
        let partial = good.replace("    - /*\n", "    - /v1/chat/completions\n");
        assert!(entry("fixture-model", &partial, SIDECAR).contains("shim.paths must expose"));
        let ungranted = good.replace("    models: [\"weights\"]\n", "");
        assert!(
            entry("fixture-model", &ungranted, SIDECAR)
                .contains("granted exactly the weights pack")
        );
        // A variable the node does not read, and a VM shape the platform does
        // not launch.
        let ambient = good.replace(
            "      - EIDOLA_ENGINE_MODEL_ID",
            "      - TOKIO_WORKER_THREADS: \"0\"\n      - EIDOLA_ENGINE_MODEL_ID",
        );
        assert!(
            entry("fixture-model", &ambient, SIDECAR)
                .contains("may not set \"TOKIO_WORKER_THREADS\"")
        );
        let tiny = good.replace("memory: 65536", "memory: 1");
        assert!(entry("fixture-model", &tiny, SIDECAR).contains("memory must be a power of two"));
        let tagged = good.replace(&format!("@sha256:{}", "2".repeat(64)), ":v1");
        assert!(entry("fixture-model", &tagged, SIDECAR).contains("exactly one container"));

        // The generator runs the gateway's whole deployment check: a variable
        // the node requires, missing, is refused here as it is there.
        for name in ["EIDOLA_ENGINE_BIND_ADDR", "EIDOLA_ENGINE_MAX_REQUESTS"] {
            let line = good.lines().find(|l| l.contains(name)).unwrap();
            let missing = good.replace(&format!("{line}\n"), "");
            let err = entry("fixture-model", &missing, SIDECAR);
            assert!(err.contains(&format!("{name} is not set")), "{name}: {err}");
        }
        // And a weights pack whose layout it does not state.
        let no_schema = good.replace("    schema: 2\n", "");
        assert!(entry("fixture-model", &no_schema, SIDECAR).contains("schema must be one of"));
    }
}
