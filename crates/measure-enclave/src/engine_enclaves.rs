//! `releases/trust/engine-enclaves.json`: the engine deployments a gateway
//! build accepts, computed from the committed deployments.
//!
//! Each Eidola-hosted engine deployment is a directory
//! `deploy/engine/<name>/<variant>/` holding (`<name>` a label for people;
//! a deployment is identified by the weights hash it serves):
//!
//! - `tinfoil-config.yml` — the deployment's Tinfoil Container config, deployed
//!   and measured exactly as the gateway's own: the `eidola-server-engine`
//!   image by digest, the weights hash and storage mode, the prompt-cache
//!   retention, the gateway token's Argon2id hash. Its `cvm-version` pins the
//!   platform release manifest inline (`X+gpu@sha256:<hex>`: the release's GPU
//!   image, which the `cuda` executor needs).
//! - `deployment.json` — what a pin needs that the config does not determine:
//!   the weights' architecture and provenance (`repo`, `revision`), the GPU count the
//!   deployment presents evidence for, and the machines its hosts are: one
//!   policy of a vendored platform endorsements release (tag and file hash),
//!   plus the PCK platform flags the endorsements do not carry.
//!
//! [`deployment_entry`] turns one such directory, plus the TDX IGVM image
//! its `cvm-version` names and that release's manifest, vendored under
//! `releases/trust/cvm-releases/`, into the file's entry: the
//! TDX launch identity (MRTD recomputed from the image, MRCONFIGID from the
//! config, via [`crate::tdx_igvm::release_pin`]) with the machine policy
//! derived from the verified endorsements
//! ([`crate::platform_endorsements::tdx_policy`]), and the weights and cache
//! policy the config sets. The gateway's build re-derives everything here
//! from the same committed files, MRTD as the vendored manifest states it
//! (the image itself is not committed), so a hand edit that disagrees with
//! them fails that build.
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
    ModelIdentity, check_deployment, check_model_agreement, release_manifest_file,
};

use tinfoil_verifier::{AllowedMeasurement, PlatformMeasurement, TdxPin};

use crate::platform_endorsements::{self, ReadFile};
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
    /// The release's `tinfoil-tdx-<version>.igvm`.
    pub igvm: &'a [u8],
    /// Reads other committed files (the vendored release manifest and
    /// platform endorsements, the Sigstore trusted root) by path relative to
    /// the workspace root.
    pub read: &'a ReadFile<'a>,
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

/// The `engine-enclaves.json` entry for one deployment, filed under the
/// weights hash it serves.
///
/// The deployment is checked by `eidola_common::engine_deployment::deployment::
/// check_deployment`, the one check the gateway's build also runs on every
/// pinned deployment, so the generator can only emit a pin the build accepts.
/// What only this tool adds is the launch identity from the platform release;
/// the pin is then compiled by the attesting client's own pin compiler
/// (`tinfoil_verifier::validate_pins`), as the gateway's build compiles it.
pub fn deployment_entry(inputs: &DeploymentInputs<'_>) -> Result<Entry> {
    let sidecar_path = inputs
        .config_path
        .strip_suffix("tinfoil-config.yml")
        .map(|dir| format!("{dir}deployment.json"))
        .context("the config path must end in tinfoil-config.yml")?;
    let checked = check_deployment(
        inputs.config_path,
        inputs.config,
        &sidecar_path,
        inputs.sidecar,
    )
    .map_err(anyhow::Error::msg)?;
    let identity = checked.model_identity();
    let machines = checked
        .tdx
        .as_ref()
        .context("deployment.json states no tdx machines; only TDX deployments are measured")?;

    // The vendored manifest, from where the gateway's build reads it;
    // `release_pin` reads it by the build's rule and measures the image
    // against it.
    let manifest_path = release_manifest_file(&checked.cvm_version).map_err(anyhow::Error::msg)?;
    let release_manifest = (inputs.read)(&manifest_path)?;
    let (_, manifest_sha256) = required_release(inputs.config)?;
    let launch = tdx_igvm::release_pin(
        &release_manifest,
        &manifest_sha256,
        inputs.igvm,
        inputs.config,
    )?;

    // The machine policy, from the signed endorsements the sidecar names,
    // verified offline. The pin is then built in the attesting client's own
    // type and compiled by its own pin compiler (the one the gateway's build
    // runs over the file): a policy the client would refuse — DEBUG
    // attributes, a wrong width, no machine — fails here, before the file is
    // rendered.
    let policy = platform_endorsements::tdx_policy(machines, inputs.read)
        .map_err(|e| anyhow::anyhow!("{sidecar_path}: tdx.platform_endorsements: {e:#}"))?;
    let pin = AllowedMeasurement {
        platform: PlatformMeasurement::Tdx(TdxPin {
            mrtd: launch.mrtd,
            mrconfigid: launch.mrconfigid,
            policy,
        }),
        expected_gpus: checked
            .expected_gpus
            .map(u32::try_from)
            .transpose()
            .context("expected_gpus must fit a u32")?,
    };
    tinfoil_verifier::validate_pins(std::slice::from_ref(&pin))
        .map_err(|e| anyhow::anyhow!("the attesting client refuses this pin: {e}"))?;
    let pin = serde_json::to_value(&pin)?;
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

/// The whole file, from every deployment's checked entry, grouped under the
/// weights hash each serves. Each model's deployments are first held to
/// agree by
/// `eidola_common::engine_deployment::deployment::check_model_agreement`, the
/// per-model rule the gateway's build also runs, so the file rendered is one
/// that build accepts.
pub fn render(entries: Vec<Entry>) -> Result<String> {
    let mut grouped: BTreeMap<String, Vec<Entry>> = BTreeMap::new();
    for entry in entries {
        grouped
            .entry(entry.identity.weights_sha256.clone())
            .or_default()
            .push(entry);
    }
    let mut models = BTreeMap::new();
    for (model, deployments) in grouped {
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

/// The file's canonical text: entries grouped by weights hash, each model's
/// deployments in config-path order.
fn render_values(entries: BTreeMap<String, Vec<Value>>) -> Result<String> {
    let mut models = serde_json::Map::new();
    for (model, mut deployments) in entries {
        deployments.sort_by(|a, b| a["config"].as_str().cmp(&b["config"].as_str()));
        models.insert(model, Value::Array(deployments));
    }
    let file = json!({ "schema_version": SCHEMA_VERSION, "models": models });
    Ok(serde_json::to_string_pretty(&SortedKeys(&file))? + "\n")
}

/// Serializes a JSON value with every object's keys in sorted order.
///
/// `serde_json::Map` keeps insertion order when any crate in the build enables
/// serde_json's `preserve_order` feature and sorts otherwise, so a `Value`'s
/// own serialization depends on what else is compiled alongside this crate.
/// The canonical file must not, so the order is fixed here.
struct SortedKeys<'a>(&'a Value);

impl serde::Serialize for SortedKeys<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::{SerializeMap, SerializeSeq};
        match self.0 {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                let mut out = serializer.serialize_map(Some(keys.len()))?;
                for key in keys {
                    out.serialize_entry(key, &SortedKeys(&map[key]))?;
                }
                out.end()
            }
            Value::Array(items) => {
                let mut out = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    out.serialize_element(&SortedKeys(item))?;
                }
                out.end()
            }
            other => other.serialize(serializer),
        }
    }
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
            r#"cvm-version: 0.0.0-test+gpu@sha256:{manifest_sha}
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
      - EIDOLA_ENGINE_MODEL_ID: "{weights}"
      - EIDOLA_ENGINE_WEIGHTS_DIR: "/tinfoil/models/weights"
      - EIDOLA_ENGINE_WEIGHTS_SHA256: "{weights}"
      - EIDOLA_ENGINE_WEIGHTS_STORAGE: "verified-readonly"
      - EIDOLA_ENGINE_EXECUTOR: "cuda"
      - EIDOLA_ENGINE_BIND_ADDR: "0.0.0.0:8080"
      - EIDOLA_ENGINE_KV_BLOCK_SIZE: "16"
      - EIDOLA_ENGINE_KV_DEVICE_BYTES: "68719476736"
      - EIDOLA_ENGINE_KERNELS_DIR: "/opt/eidola/kernels"
      - EIDOLA_ENGINE_CUDA_GRAPHS: "on"
      - EIDOLA_ENGINE_GPU_CC: "required"
      - EIDOLA_ENGINE_MAX_MODEL_LEN: "131072"
      - EIDOLA_ENGINE_MAX_SEQS: "64"
      - EIDOLA_ENGINE_MAX_BATCHED_TOKENS: "8192"
      - EIDOLA_ENGINE_MAX_PREFILL_CHUNK: "4096"
      - EIDOLA_ENGINE_DRAFT_TOKENS: "3"
      - EIDOLA_ENGINE_MAX_REQUESTS: "8"
      - GATEWAY_TOKEN_HASH: "$argon2id$v=19$m=19456,t=2,p=1$Mz9P1/uk98yKEflNjzvn5g$unNYT/KTNSNW0JCH9+9OQ2zBApPLxGNZiw746903Q8E"
      - EIDOLA_ENGINE_PREFIX_CACHE: "true"
      - EIDOLA_ENGINE_CACHE_IDLE_TTL_SECS: "900"
      - EIDOLA_ENGINE_CACHE_MAX_AGE_SECS: "7200"
      - EIDOLA_ENGINE_KV_HOST_BYTES: "34359738368"
"#,
            digest = "2".repeat(64),
            weights = "3".repeat(64),
            root = "5".repeat(64),
        )
    }

    const SIDECAR: &str = r#"{
        "architecture": "MimoV26Flash",
        "weights": {"repo": "example/fixture-model", "revision": "4444444444444444444444444444444444444444"},
        "expected_gpus": 8,
        "tdx": {
            "platform_endorsements": {
                "release": "platform-v0.1.11",
                "sha256": "ba0ccfbe8fb8a92d77a072ffeab362dc4b67faa0c88e7114eee7939ad9dd9548",
                "policy": "tdx-b300-prod"
            },
            "dynamic_platform": "true",
            "cached_keys": "true",
            "smt_enabled": "false"
        }
    }"#;

    /// Reads the committed tree as the tool's binary does, with the
    /// endorsement releases from the test fixtures.
    fn workspace_file(relative: &str) -> Result<Vec<u8>> {
        crate::platform_endorsements::tests::read(relative)
    }

    /// Where a config of [`config`] has its release manifest vendored.
    const MANIFEST: &str =
        "releases/trust/cvm-releases/v0.0.0-test/tinfoil-inference-v0.0.0-test+gpu-manifest.json";

    /// Reads as [`workspace_file`], with `release` vendored as the manifest
    /// of the fixture's release.
    fn vendoring(release: &[u8]) -> impl Fn(&str) -> Result<Vec<u8>> + '_ {
        move |path| {
            if path == MANIFEST {
                Ok(release.to_vec())
            } else {
                workspace_file(path)
            }
        }
    }

    /// The machine policy `SIDECAR` names: `tdx-b300-prod` of the fixture
    /// release `platform-v0.1.11`, with the sidecar's PCK flags.
    fn b300_policy() -> Value {
        json!({
            "mr_seam": ["aef4ed3e686fe6cf44c9a8d1cc63105443b558ac4f40c282abf78271e9a4586c50408c8584c7b43fd21edb736700ba5f"],
            "td_attributes": "0000001000000000",
            "xfam": "e702060000000000",
            "minimum_tee_tcb_svn": "03010200000000000000000000000000",
            "minimum_tcb_evaluation_data_number": 19,
            "qe_vendor_id": "939a7233f79c4ca9940a0db3957f0607",
            "ppids": ["9e2d1e4f1e26f3347d5d7ddd3841f9f4"],
            "dynamic_platform": "true",
            "cached_keys": "true",
            "smt_enabled": "false"
        })
    }

    #[test]
    fn an_entry_carries_the_launch_identity_and_what_the_config_sets() {
        let igvm = image(vec![page(0x2000, 1, IgvmPageDataFlags::new())]);
        let mrtd = hex::encode(tdx_igvm::mrtd_from_igvm(&igvm).unwrap());
        let release = manifest(&igvm, &mrtd, &"00".repeat(48));
        let release_sha = hex::encode(Sha256::digest(&release));
        let config = config(&release_sha);

        assert_eq!(
            required_release(config.as_bytes()).unwrap(),
            ("0.0.0-test+gpu".to_string(), release_sha.clone())
        );
        let entry = deployment_entry(&DeploymentInputs {
            config_path: PATH,
            config: config.as_bytes(),
            sidecar: SIDECAR.as_bytes(),
            igvm: &igvm,
            read: &vendoring(&release),
        })
        .unwrap();

        let config_sha = hex::encode(Sha256::digest(config.as_bytes()));
        assert_eq!(
            entry.json,
            json!({
                "config": PATH,
                "config_sha256": config_sha,
                "cvm_version": format!("0.0.0-test+gpu@sha256:{release_sha}"),
                "pin": {
                    "platform": {"tdx": {
                        "mrtd": mrtd,
                        "mrconfigid": format!("{config_sha}{}", "00".repeat(16)),
                        "policy": b300_policy(),
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

        let rendered = render(vec![entry]).unwrap();
        let parsed: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(parsed["schema_version"], 1);
        assert_eq!(parsed["models"][&"3".repeat(64)][0]["config"], PATH);
    }

    /// Every deployment of one weights hash shares its provenance and
    /// prompt-cache policy:
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
            deployment_entry(&DeploymentInputs {
                config_path: path,
                config: config.as_bytes(),
                sidecar: sidecar.as_bytes(),
                igvm: &igvm,
                read: &vendoring(&release),
            })
            .unwrap()
        };
        const OTHER: &str = "deploy/engine/fixture-model/tdx-b/tinfoil-config.yml";
        let two = |b: Entry| render(vec![entry(PATH, &good, SIDECAR), b]);

        let rendered = two(entry(OTHER, &good, SIDECAR)).unwrap();
        let parsed: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(parsed["models"][&"3".repeat(64)][1]["config"], OTHER);

        // Other weights are another model: filed under their own hash.
        let other_weights = good.replace(&"3".repeat(64), &"6".repeat(64));
        let rendered = two(entry(OTHER, &other_weights, SIDECAR)).unwrap();
        let parsed: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(parsed["models"][&"3".repeat(64)][0]["config"], PATH);
        assert_eq!(parsed["models"][&"6".repeat(64)][0]["config"], OTHER);

        let other_revision = SIDECAR.replace(&"4".repeat(40), &"7".repeat(40));
        let other_pack = good.replace(&"4".repeat(40), &"7".repeat(40));
        let err = two(entry(OTHER, &other_pack, &other_revision))
            .unwrap_err()
            .to_string();
        assert!(err.contains("weights provenance differs"), "{err}");

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
        // Without the cache the host tier must be 0 too.
        let no_cache = good
            .replace("PREFIX_CACHE: \"true\"", "PREFIX_CACHE: \"false\"")
            .replace("KV_HOST_BYTES: \"34359738368\"", "KV_HOST_BYTES: \"0\"");
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
        let entry = |config: &str, sidecar: &str| {
            deployment_entry(&DeploymentInputs {
                config_path: PATH,
                config: config.as_bytes(),
                sidecar: sidecar.as_bytes(),
                igvm: &igvm,
                read: &vendoring(&release),
            })
            .unwrap_err()
            .to_string()
        };
        let good = config(&release_sha);

        // A pinned node is named by its weights hash.
        let named = good.replace(
            &format!("EIDOLA_ENGINE_MODEL_ID: \"{}\"", "3".repeat(64)),
            "EIDOLA_ENGINE_MODEL_ID: \"fixture-model\"",
        );
        assert!(entry(&named, SIDECAR).contains("named by its weights hash"));
        // The sidecar names an architecture the engine implements.
        let pro = SIDECAR.replace("MimoV26Flash", "MimoV26Pro");
        assert!(entry(&good, &pro).contains("not implemented"));
        let unknown = SIDECAR.replace("MimoV26Flash", "Llama");
        assert!(entry(&good, &unknown).contains("not one the engine knows"));
        let no_machines = r#"{
            "architecture": "MimoV26Flash",
            "weights": {"repo": "example/fixture-model", "revision": "4444444444444444444444444444444444444444"},
            "expected_gpus": 8
        }"#;
        assert!(entry(&good, no_machines).contains("only TDX deployments"));
        let unpinned = good.replace(&format!("@sha256:{release_sha}"), "");
        assert!(entry(&unpinned, SIDECAR).contains("pin its release manifest"));
        let other_release = good.replace(&release_sha, &"ab".repeat(32));
        assert!(entry(&other_release, SIDECAR).contains("not the pinned"));
        // The manifest is the vendored one, read where the gateway's build
        // reads it, and must state the MRTD the image measures to.
        let unvendored = deployment_entry(&DeploymentInputs {
            config_path: PATH,
            config: good.as_bytes(),
            sidecar: SIDECAR.as_bytes(),
            igvm: &igvm,
            read: &workspace_file,
        })
        .unwrap_err();
        assert!(
            format!("{unvendored:#}").contains(MANIFEST),
            "{unvendored:#}"
        );
        let wrong_mrtd = manifest(&igvm, &"11".repeat(48), &"00".repeat(48));
        let wrong_config = config(&hex::encode(Sha256::digest(&wrong_mrtd)));
        let err = deployment_entry(&DeploymentInputs {
            config_path: PATH,
            config: wrong_config.as_bytes(),
            sidecar: SIDECAR.as_bytes(),
            igvm: &igvm,
            read: &vendoring(&wrong_mrtd),
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("disagrees with the MRTD release"), "{err}");

        // The machine policy comes only from verified endorsements: a policy
        // the release does not define, a recorded hash that is not the
        // vendored file's, a release that is not vendored, a malformed tag,
        // and a policy written out by hand are each refused before anything
        // is rendered.
        for (name, from, to, needle) in [
            (
                "unknown policy",
                "\"tdx-b300-prod\"",
                "\"tdx-b300-dev\"",
                "define no such policy",
            ),
            (
                "wrong hash",
                "\"ba0ccfbe8fb8a92d77a072ffeab362dc4b67faa0c88e7114eee7939ad9dd9548\"",
                "\"ba0ccfbe8fb8a92d77a072ffeab362dc4b67faa0c88e7114eee7939ad9dd9549\"",
                "but the recorded hash is",
            ),
            (
                "release not vendored",
                "\"platform-v0.1.11\"",
                "\"platform-v0.1.9\"",
                "reading releases/trust/platform-endorsements/platform-v0.1.9/",
            ),
            (
                "malformed tag",
                "\"platform-v0.1.11\"",
                "\"../platform-v0.1.11\"",
                "must be a platform-vX.Y.Z release tag",
            ),
            (
                "hand-written policy",
                "\"dynamic_platform\": \"true\",",
                "\"dynamic_platform\": \"true\", \"mr_seam\": [],",
                "unknown field `mr_seam`",
            ),
        ] {
            let sidecar = SIDECAR.replace(from, to);
            assert_ne!(sidecar, SIDECAR, "{name}: the edit applied");
            let err = entry(&good, &sidecar);
            assert!(err.contains(needle), "{name}: {err}");
        }

        // A CUDA deployment that asks for no GPU evidence, or a different
        // count than it attaches.
        let no_gpus = SIDECAR.replace("\"expected_gpus\": 8,", "");
        assert!(entry(&good, &no_gpus).contains("expected_gpus must be 8"));
        let zero_gpus = SIDECAR.replace("\"expected_gpus\": 8", "\"expected_gpus\": 0");
        assert!(entry(&good, &zero_gpus).contains("expected_gpus must be 8"));
        let unstated = good.replace("gpus: 8\n", "");
        assert!(entry(&unstated, SIDECAR).contains("needs the config to attach GPUs"));
        let nine = good.replace("gpus: 8\n", "gpus: 9\n");
        assert!(entry(&nine, SIDECAR).contains("gpus must be between 0 and 8"));

        // The CPU image under the cuda executor, and the node's GPU
        // confidential-computing check switched off.
        let cpu_image = good.replace("0.0.0-test+gpu@", "0.0.0-test@");
        assert_ne!(cpu_image, good);
        assert!(entry(&cpu_image, SIDECAR).contains("is not a GPU image"));
        let no_cc = good.replace("GPU_CC: \"required\"", "GPU_CC: \"not-required\"");
        assert_ne!(no_cc, good);
        assert!(entry(&no_cc, SIDECAR).contains("EIDOLA_ENGINE_GPU_CC is \"not-required\""));

        // A companion container, or an engine named by a tag.
        let companion = format!(
            "{good}  - name: \"sidecar\"\n    image: \"ghcr.io/example/sidecar:latest\"\n    env: []\n"
        );
        assert!(entry(&companion, SIDECAR).contains("exactly one container"));
        // A key an engine deployment does not need.
        let entrypoint = good.replace(
            "    runtime: nvidia\n",
            "    runtime: nvidia\n    entrypoint: [\"/bin/sh\"]\n",
        );
        assert!(entry(&entrypoint, SIDECAR).contains("unknown field `entrypoint`"));
        let no_runtime = good.replace("    runtime: nvidia\n", "");
        assert!(entry(&no_runtime, SIDECAR).contains("runtime: nvidia"));
        // Another secret, a partial shim, an ungranted pack.
        let extra_secret = good.replace(
            "      - GATEWAY_TOKEN\n",
            "      - GATEWAY_TOKEN\n      - RUST_LOG\n",
        );
        assert!(entry(&extra_secret, SIDECAR).contains("secrets must be exactly"));
        let partial = good.replace("    - /*\n", "    - /v1/chat/completions\n");
        assert!(entry(&partial, SIDECAR).contains("shim.paths must expose"));
        let ungranted = good.replace("    models: [\"weights\"]\n", "");
        assert!(entry(&ungranted, SIDECAR).contains("granted exactly the weights pack"));
        // A variable the node does not read, and a VM shape the platform does
        // not launch.
        let ambient = good.replace(
            "      - EIDOLA_ENGINE_MODEL_ID",
            "      - TOKIO_WORKER_THREADS: \"0\"\n      - EIDOLA_ENGINE_MODEL_ID",
        );
        assert!(entry(&ambient, SIDECAR).contains("may not set \"TOKIO_WORKER_THREADS\""));
        let tiny = good.replace("memory: 65536", "memory: 1");
        assert!(entry(&tiny, SIDECAR).contains("memory must be a whole number of GiB"));
        let tagged = good.replace(&format!("@sha256:{}", "2".repeat(64)), ":v1");
        assert!(entry(&tagged, SIDECAR).contains("exactly one container"));

        // The generator runs the gateway's whole deployment check: a variable
        // the node requires, missing, is refused here as it is there.
        for name in ["EIDOLA_ENGINE_BIND_ADDR", "EIDOLA_ENGINE_MAX_REQUESTS"] {
            let line = good.lines().find(|l| l.contains(name)).unwrap();
            let missing = good.replace(&format!("{line}\n"), "");
            let err = entry(&missing, SIDECAR);
            assert!(err.contains(&format!("{name} is not set")), "{name}: {err}");
        }
        // And a weights pack whose layout it does not state.
        let no_schema = good.replace("    schema: 2\n", "");
        assert!(entry(&no_schema, SIDECAR).contains("schema must be one of"));
    }
}
