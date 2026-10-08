//! The engine-trust check over the committed file and over fixtures.
//!
//! `manifest::check` is what `build.rs` runs on every gateway build; these
//! tests run it over `tests/fixtures/engine-trust/` (a synthetic deployment
//! laid out as a real one is) and over one mutation per rule, so each rule is
//! shown to refuse what it exists to refuse.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::manifest::{self, CheckedCachePolicy, CheckedModel, CheckedWeights};
use super::*;

const MODEL: &str = "fixture-model";
const CONFIG: &str = "deploy/engine/fixture-model/tdx-a/tinfoil-config.yml";
const SIDECAR: &str = "deploy/engine/fixture-model/tdx-a/deployment.json";

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/engine-trust")
}

/// The fixture tree, in memory, so a test can change one file of it.
struct Tree {
    json: serde_json::Value,
    files: BTreeMap<String, Vec<u8>>,
}

impl Tree {
    fn fixture() -> Self {
        let root = fixture_root();
        let read = |p: &str| std::fs::read(root.join(p)).unwrap();
        Self {
            json: serde_json::from_slice(&read("engine-enclaves.json")).unwrap(),
            files: [CONFIG, SIDECAR]
                .into_iter()
                .map(|p| (p.to_string(), read(p)))
                .collect(),
        }
    }

    fn check(&self) -> Result<Vec<CheckedModel>, String> {
        let json = serde_json::to_vec(&self.json).unwrap();
        manifest::check(&json, &mut |path| {
            self.files
                .get(path)
                .cloned()
                .ok_or_else(|| format!("read {path}: no such file"))
        })
    }

    fn deployment(&mut self) -> &mut serde_json::Value {
        &mut self.json["models"][MODEL][0]
    }

    fn config_text(&self) -> String {
        String::from_utf8(self.files[CONFIG].clone()).unwrap()
    }

    /// Replace the config, keeping every hash of it the pin carries in step,
    /// so the rule under test is the only thing that can object.
    fn set_config(&mut self, text: String) {
        use sha2::Digest as _;
        let hash: String = sha2::Sha256::digest(text.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        self.deployment()["config_sha256"] = hash.clone().into();
        self.deployment()["pin"]["platform"]["tdx"]["mrconfigid"] =
            format!("{hash}{}", "00".repeat(16)).into();
        self.files.insert(CONFIG.to_string(), text.into_bytes());
    }

    fn edit_config(&mut self, from: &str, to: &str) {
        let text = self.config_text();
        assert!(text.contains(from), "the fixture config has no {from:?}");
        self.set_config(text.replacen(from, to, 1));
    }

    fn sidecar(&self) -> serde_json::Value {
        serde_json::from_slice(&self.files[SIDECAR]).unwrap()
    }

    fn set_sidecar(&mut self, value: serde_json::Value) {
        self.files
            .insert(SIDECAR.to_string(), serde_json::to_vec(&value).unwrap());
    }
}

#[track_caller]
fn refused(tree: &Tree, expected: &str) {
    match tree.check() {
        Ok(_) => panic!("accepted, but should have been refused with {expected:?}"),
        Err(e) => assert!(e.contains(expected), "refused for another reason: {e}"),
    }
}

// ---------------------------------------------------------------------------
// The committed file
// ---------------------------------------------------------------------------

/// The committed file passes the check `build.rs` ran, and the list compiled
/// into this binary is what that check returns: the generated constants
/// cannot disagree with the file they were generated from.
#[test]
fn the_committed_file_is_what_this_build_embedded() {
    let root = workspace_root();
    let json = std::fs::read(root.join("releases/trust/engine-enclaves.json")).unwrap();
    assert_eq!(json, ENGINE_ENCLAVES_JSON.as_bytes());
    let checked = manifest::check(&json, &mut |p| {
        std::fs::read(root.join(p)).map_err(|e| e.to_string())
    })
    .expect("the committed engine-enclaves.json passes its check");
    assert_eq!(checked.len(), PINNED_MODELS.len());
    for (checked, pinned) in checked.iter().zip(PINNED_MODELS) {
        assert_eq!(checked.id, pinned.id());
        assert_eq!(checked.weights.sha256, pinned.weights().sha256);
        assert_eq!(checked.weights.repo, pinned.weights().repo);
        assert_eq!(checked.weights.revision, pinned.weights().revision);
        assert_eq!(checked.prompt_cache.enabled, pinned.prompt_cache().enabled);
        assert_eq!(
            checked.prompt_cache.idle_ttl_secs,
            pinned.prompt_cache().idle_ttl_secs
        );
        assert_eq!(
            checked.prompt_cache.max_age_secs,
            pinned.prompt_cache().max_age_secs
        );
        assert_eq!(checked.deployments, pinned.deployments());
    }
}

/// Every committed pin is one `tinfoil-verifier` compiles: the shape check
/// at build time does not read a TDX policy's fields, the verifier does.
#[tokio::test]
async fn every_committed_pin_compiles_in_the_verifier() {
    for model in PINNED_MODELS {
        let pins = allowed_measurements(model.id()).unwrap();
        assert_eq!(pins.len(), model.deployments().len(), "{}", model.id());
        verifier_accepts(&pins).await;
    }
    assert!(allowed_measurements("no-such-model").unwrap().is_empty());
}

/// No engine is deployed yet, and that is a valid state: an empty pin set
/// builds and pins nothing.
#[test]
fn an_empty_pin_set_is_valid() {
    let json = br#"{"schema_version": 1, "models": {}}"#;
    let checked = manifest::check(json, &mut |p| Err(format!("read {p}"))).unwrap();
    assert!(checked.is_empty());
}

async fn verifier_accepts(pins: &[AllowedMeasurement]) {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let mut tls_roots = rustls::RootCertStore::empty();
    tls_roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    tinfoil_verifier::attesting_client(tinfoil_verifier::AttestingClientConfig {
        allowed_measurements: pins,
        inference_base_url: "https://engine.invalid/v1",
        tls_roots,
        trusted_ark_der: None,
        trusted_ask_der: None,
        snp_min_tcb: None,
        snp_observer: None,
        attestation_observer: None,
    })
    .await
    .expect("the verifier compiles every pin");
}

// ---------------------------------------------------------------------------
// The fixture: accepted, and read back whole
// ---------------------------------------------------------------------------

#[test]
fn the_fixture_deployment_is_accepted() {
    let checked = Tree::fixture().check().expect("the fixture is valid");
    assert_eq!(
        checked,
        vec![CheckedModel {
            id: MODEL.to_string(),
            weights: CheckedWeights {
                sha256: "3".repeat(64),
                repo: "example/fixture-model".to_string(),
                revision: "4".repeat(40),
            },
            prompt_cache: CheckedCachePolicy {
                enabled: true,
                idle_ttl_secs: 900,
                max_age_secs: 7200,
            },
            deployments: vec![CONFIG.to_string()],
        }]
    );
}

#[tokio::test]
async fn the_fixture_pin_is_a_tdx_pin_the_verifier_compiles() {
    let json = std::fs::read_to_string(fixture_root().join("engine-enclaves.json")).unwrap();
    let pins = allowed_measurements_in(&json, MODEL).unwrap();
    assert_eq!(pins.len(), 1);
    assert_eq!(pins[0].platform(), tinfoil_verifier::Platform::Tdx);
    assert_eq!(pins[0].expected_gpus, Some(8));
    verifier_accepts(&pins).await;
    assert!(
        allowed_measurements_in(&json, "other-model")
            .unwrap()
            .is_empty()
    );
}

// ---------------------------------------------------------------------------
// One mutation per rule
// ---------------------------------------------------------------------------

#[test]
fn the_file_shape_is_strict() {
    let mut tree = Tree::fixture();
    tree.json["schema_version"] = 2.into();
    refused(&tree, "schema_version must be 1");

    let mut tree = Tree::fixture();
    tree.json["extra"] = true.into();
    refused(&tree, "unknown member \"extra\"");

    let mut tree = Tree::fixture();
    tree.deployment()["note"] = "hi".into();
    refused(&tree, "unknown member \"note\"");

    let mut tree = Tree::fixture();
    tree.deployment()
        .as_object_mut()
        .unwrap()
        .remove("prompt_cache");
    refused(&tree, "missing member \"prompt_cache\"");

    let mut tree = Tree::fixture();
    tree.json["models"][MODEL] = serde_json::json!([]);
    refused(&tree, "non-empty array");

    let mut tree = Tree::fixture();
    let upper = tree.deployment()["config_sha256"]
        .as_str()
        .unwrap()
        .to_uppercase();
    tree.deployment()["config_sha256"] = upper.into();
    refused(&tree, "config_sha256: must be 64 lowercase hex digits");
}

#[test]
fn the_config_must_be_this_models_committed_file() {
    let mut tree = Tree::fixture();
    tree.deployment()["config"] = "deploy/engine/other-model/tdx-a/tinfoil-config.yml".into();
    refused(&tree, "config must be deploy/engine/fixture-model/");

    let mut tree = Tree::fixture();
    tree.deployment()["config"] = "deploy/engine/fixture-model/../x/tinfoil-config.yml".into();
    refused(&tree, "config must be deploy/engine/fixture-model/");

    let mut tree = Tree::fixture();
    tree.files.remove(CONFIG);
    refused(&tree, "no such file");

    let mut tree = Tree::fixture();
    let text = tree.config_text() + "\n# edited\n";
    tree.files.insert(CONFIG.to_string(), text.into_bytes());
    refused(
        &tree,
        "but deploy/engine/fixture-model/tdx-a/tinfoil-config.yml hashes to",
    );
}

#[test]
fn a_tdx_pin_carries_the_configs_hash() {
    let mut tree = Tree::fixture();
    tree.deployment()["pin"]["platform"]["tdx"]["mrconfigid"] = "7".repeat(96).into();
    refused(&tree, "mrconfigid is not the hash of");
}

#[test]
fn the_release_is_the_configs_and_pinned_inline() {
    let mut tree = Tree::fixture();
    tree.deployment()["cvm_version"] = "0.15.1@sha256:".to_string().into();
    refused(&tree, "but the config selects");

    let mut tree = Tree::fixture();
    let pinned = format!("0.15.0@sha256:{}", "1".repeat(64));
    tree.edit_config(&pinned, "0.15.0");
    tree.deployment()["cvm_version"] = "0.15.0".into();
    refused(&tree, "must pin its release manifest inline");
}

#[test]
fn the_config_serves_this_model_and_these_weights() {
    let mut tree = Tree::fixture();
    tree.edit_config("\"fixture-model\"", "\"another-model\"");
    refused(&tree, "EIDOLA_ENGINE_MODEL_ID is \"another-model\"");

    let mut tree = Tree::fixture();
    tree.edit_config(&"3".repeat(64), &"8".repeat(64));
    refused(&tree, "EIDOLA_ENGINE_WEIGHTS_SHA256 is");

    let mut tree = Tree::fixture();
    tree.deployment()["weights"]["sha256"] = "8".repeat(64).into();
    refused(&tree, "EIDOLA_ENGINE_WEIGHTS_SHA256 is");

    let mut tree = Tree::fixture();
    tree.edit_config("\"verified-readonly\"", "\"dev-writable\"");
    refused(&tree, "EIDOLA_ENGINE_WEIGHTS_STORAGE is \"dev-writable\"");
}

#[test]
fn the_prompt_cache_policy_is_the_configs() {
    let mut tree = Tree::fixture();
    tree.edit_config(
        "CACHE_IDLE_TTL_SECS: \"900\"",
        "CACHE_IDLE_TTL_SECS: \"600\"",
    );
    refused(&tree, "EIDOLA_ENGINE_CACHE_IDLE_TTL_SECS is \"600\"");

    let mut tree = Tree::fixture();
    tree.deployment()["prompt_cache"]["max_age_secs"] = 3600.into();
    refused(&tree, "EIDOLA_ENGINE_CACHE_MAX_AGE_SECS is \"7200\"");

    let mut tree = Tree::fixture();
    tree.edit_config("PREFIX_CACHE: \"true\"", "PREFIX_CACHE: \"false\"");
    refused(&tree, "EIDOLA_ENGINE_PREFIX_CACHE is \"false\"");

    let mut tree = Tree::fixture();
    tree.deployment()["prompt_cache"]["idle_ttl_secs"] = 9000.into();
    refused(&tree, "idle_ttl_secs exceeds max_age_secs");
}

#[test]
fn the_gateway_token_is_a_secret_and_only_its_hash_is_measured() {
    let mut tree = Tree::fixture();
    tree.edit_config("      - GATEWAY_TOKEN\n", "");
    refused(&tree, "must take GATEWAY_TOKEN as a secret");

    let mut tree = Tree::fixture();
    tree.edit_config(
        "      - EIDOLA_ENGINE_MODEL_ID",
        "      - GATEWAY_TOKEN: \"dev-gateway-token\"\n      - EIDOLA_ENGINE_MODEL_ID",
    );
    refused(
        &tree,
        "GATEWAY_TOKEN must not be a measured environment value",
    );

    let mut tree = Tree::fixture();
    tree.edit_config("$argon2id$", "$argon2i$");
    refused(&tree, "GATEWAY_TOKEN_HASH must be an Argon2id hash");
}

#[test]
fn the_config_runs_one_digest_pinned_engine_with_quoted_values() {
    let mut tree = Tree::fixture();
    tree.edit_config(&format!("@sha256:{}", "2".repeat(64)), ":latest");
    refused(&tree, "must run exactly one container whose image is");

    let mut tree = Tree::fixture();
    let engine = tree.config_text();
    let start = engine.find("  - name:").unwrap();
    let end = engine.find("\nshim:").unwrap();
    let container = engine[start..end].to_string();
    tree.edit_config(&container, &format!("{container}{container}"));
    refused(&tree, "must run exactly one container whose image is");

    let mut tree = Tree::fixture();
    tree.edit_config("PREFIX_CACHE: \"true\"", "PREFIX_CACHE: true");
    refused(&tree, "with a quoted string value");

    let mut tree = Tree::fixture();
    tree.edit_config(
        "      - EIDOLA_ENGINE_EXECUTOR: \"cuda\"\n",
        "      - EIDOLA_ENGINE_EXECUTOR: \"cuda\"\n      - EIDOLA_ENGINE_EXECUTOR: \"cpu\"\n",
    );
    refused(&tree, "env sets EIDOLA_ENGINE_EXECUTOR twice");
}

#[test]
fn the_pin_carries_what_the_sidecar_states() {
    let mut tree = Tree::fixture();
    tree.files.remove(SIDECAR);
    refused(
        &tree,
        "read deploy/engine/fixture-model/tdx-a/deployment.json",
    );

    let mut tree = Tree::fixture();
    let mut sidecar = tree.sidecar();
    sidecar["weights"]["revision"] = "9".repeat(40).into();
    tree.set_sidecar(sidecar);
    refused(&tree, "weights.repo / weights.revision differ from");

    let mut tree = Tree::fixture();
    let mut sidecar = tree.sidecar();
    sidecar["expected_gpus"] = 4.into();
    tree.set_sidecar(sidecar);
    refused(&tree, "pin.expected_gpus differs from");

    let mut tree = Tree::fixture();
    let mut sidecar = tree.sidecar();
    sidecar["tdx_policy"]["smt_enabled"] = "true".into();
    tree.set_sidecar(sidecar);
    refused(&tree, "pin.platform.tdx.policy differs from");

    let mut tree = Tree::fixture();
    let mut sidecar = tree.sidecar();
    sidecar["unexpected"] = 1.into();
    tree.set_sidecar(sidecar);
    refused(&tree, "unknown member \"unexpected\"");
}

#[test]
fn a_pin_names_exactly_one_platform() {
    let mut tree = Tree::fixture();
    tree.deployment()["pin"]["platform"]["sev_snp"] =
        serde_json::json!({"measurement": "a".repeat(96)});
    refused(&tree, "exactly one member, tdx or sev_snp");

    // SEV-SNP stays expressible, and then the sidecar states no TDX policy.
    let mut tree = Tree::fixture();
    tree.deployment()["pin"]["platform"] =
        serde_json::json!({"sev_snp": {"measurement": "a".repeat(96)}});
    refused(&tree, "a SEV-SNP pin, but");
    let mut sidecar = tree.sidecar();
    sidecar.as_object_mut().unwrap().remove("tdx_policy");
    tree.set_sidecar(sidecar);
    tree.check().expect("a SEV-SNP deployment is expressible");
}

#[test]
fn every_deployment_of_a_model_agrees_and_none_is_pinned_twice() {
    let mut tree = Tree::fixture();
    let second = tree.deployment().clone();
    tree.json["models"][MODEL]
        .as_array_mut()
        .unwrap()
        .push(second.clone());
    refused(&tree, "is already pinned under model");

    // A second variant of the same model, with its own files.
    let variant = "deploy/engine/fixture-model/tdx-b";
    let mut tree = Tree::fixture();
    tree.files.insert(
        format!("{variant}/tinfoil-config.yml"),
        tree.files[CONFIG].clone(),
    );
    tree.files.insert(
        format!("{variant}/deployment.json"),
        tree.files[SIDECAR].clone(),
    );
    let mut other = second.clone();
    other["config"] = format!("{variant}/tinfoil-config.yml").into();
    tree.json["models"][MODEL]
        .as_array_mut()
        .unwrap()
        .push(other.clone());
    let checked = tree.check().expect("two agreeing variants");
    assert_eq!(checked[0].deployments.len(), 2);

    let config = String::from_utf8(tree.files[&format!("{variant}/tinfoil-config.yml")].clone())
        .unwrap()
        .replace("\"900\"", "\"600\"");
    tree.files.insert(
        format!("{variant}/tinfoil-config.yml"),
        config.clone().into_bytes(),
    );
    use sha2::Digest as _;
    let hash: String = sha2::Sha256::digest(config.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let entry = &mut tree.json["models"][MODEL][1];
    entry["config_sha256"] = hash.clone().into();
    entry["pin"]["platform"]["tdx"]["mrconfigid"] = format!("{hash}{}", "00".repeat(16)).into();
    entry["prompt_cache"]["idle_ttl_secs"] = 600.into();
    refused(
        &tree,
        "prompt_cache differs from the model's other deployments",
    );
}
