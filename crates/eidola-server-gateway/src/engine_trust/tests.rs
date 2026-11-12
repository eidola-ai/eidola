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
use eidola_common::engine_deployment;

/// The fixture model's weights hash, which it is listed under and its node
/// is named by.
const MODEL: &str = "3333333333333333333333333333333333333333333333333333333333333333";
/// The shape the fixture's resources are checked against.
const FLASH: &engine_deployment::Shape = &eidola_common::architecture::MIMO_V26_FLASH;
const CONFIG: &str = "deploy/engine/fixture-model/tdx-a/tinfoil-config.yml";
const SIDECAR: &str = "deploy/engine/fixture-model/tdx-a/deployment.json";
/// The platform endorsements the fixture sidecar names, as vendored.
const ENDORSEMENTS: &str =
    "releases/trust/platform-endorsements/platform-v0.1.11/platform-endorsements.json";
const ENDORSEMENTS_BUNDLE: &str =
    "releases/trust/platform-endorsements/platform-v0.1.11/attestation.sigstore.json";
const TRUSTED_ROOT: &str = "releases/trust/sigstore-trusted-root.json";
/// The platform release manifest the fixture config pins, vendored: a copy of
/// the real `v0.15.3-rc.1+gpu` manifest, so the fixture's MRTD is that
/// release's.
const RELEASE_MANIFEST: &str =
    "releases/trust/cvm-releases/v0.15.3-rc.1/tinfoil-inference-v0.15.3-rc.1+gpu-manifest.json";
/// That manifest's SHA-256, the fixture config's inline pin.
const RELEASE_SHA256: &str = "1421cbb45dc428f6270d106d8eb68faab3ce7ca02ff2fbb2dc0968605004579f";
/// The MRTD it states, the fixture pin's.
const RELEASE_MRTD: &str = "bf582997bf2145f275db3bc2542d703fdd889c85927b3ee199138298f863631711f41b37d8d347d5aee18036ff02c16c";

/// The weights pack's data size in the fixture's `mpk` (`_17419419648_`).
const FIXTURE_PACK_BYTES: u64 = 17_419_419_648;

/// The fixture config's env, read by the node's own boot grammar.
fn fixture_measured(tree: &Tree) -> engine_deployment::MeasuredConfig {
    let text = tree.config_text();
    let env: BTreeMap<String, String> = text
        .lines()
        .filter_map(|line| {
            let (name, value) = line.trim_start().strip_prefix("- ")?.split_once(": ")?;
            Some((name.to_string(), value.trim_matches('"').to_string()))
        })
        .collect();
    engine_deployment::parse_measured(&|name| env.get(name).cloned())
        .expect("the fixture's env is one the node boots with")
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/engine-trust")
}

/// The fixture tree, in memory, so a test can change one file of it.
#[derive(Clone)]
struct Tree {
    json: serde_json::Value,
    files: BTreeMap<String, Vec<u8>>,
}

impl Tree {
    fn fixture() -> Self {
        let root = fixture_root();
        let read = |p: &str| std::fs::read(root.join(p)).unwrap();
        let mut files: BTreeMap<String, Vec<u8>> = [CONFIG, SIDECAR, RELEASE_MANIFEST]
            .into_iter()
            .map(|p| (p.to_string(), read(p)))
            .collect();
        // The vendored endorsements and the trusted root are the committed
        // ones: the fixture deployment's machine policy is re-derived from
        // the real signed release.
        for path in [ENDORSEMENTS, ENDORSEMENTS_BUNDLE, TRUSTED_ROOT] {
            files.insert(
                path.to_string(),
                std::fs::read(workspace_root().join(path)).unwrap(),
            );
        }
        Self {
            json: serde_json::from_slice(&read("engine-enclaves.json")).unwrap(),
            files,
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

    /// Check the file as the given text, for shapes a `Value` cannot hold.
    fn check_text(&self, json: &str) -> Result<Vec<CheckedModel>, String> {
        manifest::check(json.as_bytes(), &mut |path| {
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

    /// Vendor `manifest` as the release manifest, and pin it in the config
    /// and the pin alike, so only what the manifest states can object.
    fn vendor_manifest(&mut self, manifest: &serde_json::Value) {
        use sha2::Digest as _;
        let bytes = serde_json::to_vec_pretty(manifest).unwrap();
        let hash: String = sha2::Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let pinned = self.deployment()["cvm_version"]
            .as_str()
            .unwrap()
            .to_string();
        let moved = format!("{}@sha256:{hash}", pinned.split_once('@').unwrap().0);
        self.edit_config(&pinned, &moved);
        self.deployment()["cvm_version"] = moved.into();
        self.files.insert(RELEASE_MANIFEST.to_string(), bytes);
    }

    fn release_manifest(&self) -> serde_json::Value {
        serde_json::from_slice(&self.files[RELEASE_MANIFEST]).unwrap()
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
            architecture: engine_deployment::Architecture::MimoV26Flash,
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
            max_model_len: 131_072,
        }]
    );
}

/// A catalog row may not publish a longer context window than its
/// deployments serve: clients size prompts to the published window, and the
/// node refuses a sequence beyond its `EIDOLA_ENGINE_MAX_MODEL_LEN`.
#[test]
fn a_rows_window_is_one_its_deployments_serve() {
    let checked = Tree::fixture().check().expect("the fixture is valid");
    let row = "models/fixture/flash.toml";
    let model = "3".repeat(64);
    manifest::check_context_windows(&[(row, &model, 131_072)], &checked)
        .expect("the deployment's own length");
    manifest::check_context_windows(&[(row, &model, 8_192)], &checked).expect("a shorter window");
    let err = manifest::check_context_windows(&[(row, &model, 131_073)], &checked).unwrap_err();
    assert!(
        err.contains(row) && err.contains("EIDOLA_ENGINE_MAX_MODEL_LEN"),
        "{err}"
    );
    // An unpinned row is the catalog binding's to refuse.
    manifest::check_context_windows(&[(row, &"9".repeat(64), u64::MAX)], &checked)
        .expect("not this check's");
}

#[tokio::test]
async fn the_fixture_pin_is_a_tdx_pin_the_verifier_compiles() {
    let json = std::fs::read_to_string(fixture_root().join("engine-enclaves.json")).unwrap();
    let pins = allowed_measurements_in(&json, MODEL).unwrap();
    assert_eq!(pins.len(), 1);
    assert_eq!(pins[0].platform(), tinfoil_verifier::Platform::Tdx);
    assert_eq!(pins[0].expected_gpus, Some(8));
    verifier_accepts(&pins).await;
    // Each deployment is named by its config's hash, the hash its MRCONFIGID
    // carries.
    let deployments = accepted_deployments_in(&json, MODEL).unwrap();
    assert_eq!(deployments.len(), 1);
    let config = std::fs::read(fixture_root().join(CONFIG)).unwrap();
    assert_eq!(
        deployments[0].config_sha256,
        hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&config))
    );
    assert!(
        allowed_measurements_in(&json, "other-model")
            .unwrap()
            .is_empty()
    );
}

/// A deployment directory committed without a pin is refused; with one, it
/// passes.
#[test]
fn every_committed_deployment_is_pinned() {
    let checked = Tree::fixture().check().unwrap();
    manifest::require_every_deployment_pinned(&[CONFIG.to_string()], &checked).unwrap();
    manifest::require_every_deployment_pinned(&[], &checked).unwrap();
    let stray = "deploy/engine/fixture-model/tdx-b/tinfoil-config.yml".to_string();
    let err = manifest::require_every_deployment_pinned(&[CONFIG.to_string(), stray], &checked)
        .unwrap_err();
    assert!(
        err.contains("tdx-b/tinfoil-config.yml is committed but not pinned"),
        "{err}"
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
    refused(&tree, "unknown field `extra`");

    let mut tree = Tree::fixture();
    tree.deployment()["note"] = "hi".into();
    refused(&tree, "unknown field `note`");

    let mut tree = Tree::fixture();
    tree.deployment()
        .as_object_mut()
        .unwrap()
        .remove("prompt_cache");
    refused(&tree, "missing field `prompt_cache`");

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
    refused(
        &tree,
        "read deploy/engine/other-model/tdx-a/tinfoil-config.yml",
    );

    for bad in [
        "deploy/engine/fixture-model/../x/tinfoil-config.yml",
        "deploy/engine/tdx-a/tinfoil-config.yml",
        "deploy/engine/a/b/c/tinfoil-config.yml",
        "deploy/other/fixture-model/tdx-a/tinfoil-config.yml",
    ] {
        let mut tree = Tree::fixture();
        tree.deployment()["config"] = bad.into();
        refused(&tree, "config must be deploy/engine/<name>/<variant>/");
    }

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
    let pinned = format!("0.15.3-rc.1+gpu@sha256:{RELEASE_SHA256}");
    tree.edit_config(&pinned, "0.15.3-rc.1+gpu");
    tree.deployment()["cvm_version"] = "0.15.3-rc.1+gpu".into();
    refused(&tree, "must pin its release manifest inline");
}

/// A TDX pin's MRTD is the one its platform release states: the vendored
/// manifest the config pins, which must be there, be the pinned bytes,
/// describe the release the config selects and state a launch in the
/// measured model.
#[test]
fn a_tdx_pins_mrtd_is_the_one_its_release_states() {
    let mut tree = Tree::fixture();
    assert_eq!(
        tree.deployment()["pin"]["platform"]["tdx"]["mrtd"],
        RELEASE_MRTD
    );
    // Only the MRTD moved: every other value still re-derives.
    tree.deployment()["pin"]["platform"]["tdx"]["mrtd"] = "6".repeat(96).into();
    refused(
        &tree,
        &format!(
            "pin.platform.tdx.mrtd is {}, but {RELEASE_MANIFEST} states {RELEASE_MRTD}",
            "6".repeat(96)
        ),
    );

    let mut tree = Tree::fixture();
    tree.files.remove(RELEASE_MANIFEST);
    refused(&tree, &format!("read {RELEASE_MANIFEST}: no such file"));
    refused(&tree, "is not vendored");

    // The vendored bytes edited, the config's pin left alone.
    let mut tree = Tree::fixture();
    let mut text = String::from_utf8(tree.files[RELEASE_MANIFEST].clone()).unwrap();
    text.push('\n');
    tree.files
        .insert(RELEASE_MANIFEST.to_string(), text.into_bytes());
    refused(&tree, &format!("not the pinned {RELEASE_SHA256}"));

    // A manifest the config pins, whose statement disagrees: its MRTD, the
    // release it describes, its layout, a measured boot.
    for (name, edit, needle) in [
        (
            "another MRTD",
            &(|m: &mut serde_json::Value| {
                m["igvm"]["tdx_launch"]["mrtd"] = "6".repeat(96).into();
            }) as &dyn Fn(&mut serde_json::Value),
            "but releases/trust/cvm-releases/",
        ),
        (
            "the CPU image's release",
            &|m: &mut serde_json::Value| m["version"] = "v0.15.3-rc.1".into(),
            "but the release manifest is v0.15.3-rc.1",
        ),
        (
            "another layout",
            &|m: &mut serde_json::Value| m["igvm"]["format_version"] = 2.into(),
            "format_version 2 is not supported",
        ),
        (
            "a measured boot",
            &|m: &mut serde_json::Value| {
                m["igvm"]["tdx_launch"]["rtmr1"] = "1".repeat(96).into();
            },
            "non-zero TDX rtmr1",
        ),
    ] {
        let mut tree = Tree::fixture();
        let mut manifest = tree.release_manifest();
        tree.vendor_manifest(&manifest);
        tree.check()
            .unwrap_or_else(|e| panic!("{name}: re-vendored unchanged: {e}"));
        edit(&mut manifest);
        tree.vendor_manifest(&manifest);
        let err = tree.check().expect_err(name);
        assert!(err.contains(needle), "{name}: {err}");
    }
}

/// The cuda executor runs on the release's GPU image (`X+gpu`), whose measured
/// boot attests the GPUs before opening them; the plain CPU image, or any
/// other build tag, is refused (`engine_deployment::check_cvm_image`).
#[test]
fn a_cuda_deployment_runs_on_the_gpu_image() {
    let pinned = format!("0.15.3-rc.1+gpu@sha256:{RELEASE_SHA256}");
    for other in ["0.15.3-rc.1", "0.15.3-rc.1+GPU", "0.15.3-rc.1+gpu2"] {
        let bad = format!("{other}@sha256:{RELEASE_SHA256}");
        let mut tree = Tree::fixture();
        tree.edit_config(&pinned, &bad);
        tree.deployment()["cvm_version"] = bad.clone().into();
        refused(&tree, "is not a GPU image");
    }
}

#[test]
fn the_config_serves_this_model_and_these_weights() {
    // A pinned node is named by its weights hash.
    let mut tree = Tree::fixture();
    tree.edit_config(
        &format!("EIDOLA_ENGINE_MODEL_ID: \"{MODEL}\""),
        "EIDOLA_ENGINE_MODEL_ID: \"fixture-model\"",
    );
    refused(&tree, "a pinned node is named by its weights hash");

    // The file lists the model under that hash.
    let mut tree = Tree::fixture();
    let deployments = tree.json["models"]
        .as_object_mut()
        .unwrap()
        .remove(MODEL)
        .unwrap();
    tree.json["models"]["fixture-model"] = deployments;
    refused(&tree, "listed under its weights hash");

    let mut tree = Tree::fixture();
    let deployments = tree.json["models"]
        .as_object_mut()
        .unwrap()
        .remove(MODEL)
        .unwrap();
    tree.json["models"][&"8".repeat(64)] = deployments;
    refused(&tree, "not 8888");

    // The sidecar names an implemented architecture.
    let mut tree = Tree::fixture();
    let mut sidecar = tree.sidecar();
    sidecar["architecture"] = "MimoV26Pro".into();
    tree.set_sidecar(sidecar);
    refused(&tree, "is not implemented");

    // Other weights, with the node named by them: not the pin's.
    let mut tree = Tree::fixture();
    let text = tree.config_text().replace(&"3".repeat(64), &"8".repeat(64));
    tree.set_config(text);
    refused(&tree, "EIDOLA_ENGINE_WEIGHTS_SHA256 is");

    let mut tree = Tree::fixture();
    tree.deployment()["weights"]["sha256"] = "8".repeat(64).into();
    refused(&tree, "EIDOLA_ENGINE_WEIGHTS_SHA256 is");

    let mut tree = Tree::fixture();
    tree.edit_config("\"verified-readonly\"", "\"dev-writable\"");
    refused(&tree, "EIDOLA_ENGINE_WEIGHTS_STORAGE is \"dev-writable\"");
}

/// The node's own GPU confidential-computing check is on in every pinned
/// deployment: `not-required` boots only unpinned, for development on a GPU
/// without confidential computing.
#[test]
fn a_pinned_deployment_requires_gpu_confidential_computing() {
    let mut tree = Tree::fixture();
    tree.edit_config(
        "EIDOLA_ENGINE_GPU_CC: \"required\"",
        "EIDOLA_ENGINE_GPU_CC: \"not-required\"",
    );
    refused(&tree, "EIDOLA_ENGINE_GPU_CC is \"not-required\"");
}

#[test]
fn the_prompt_cache_policy_is_the_configs() {
    let mut tree = Tree::fixture();
    tree.edit_config(
        "CACHE_IDLE_TTL_SECS: \"900\"",
        "CACHE_IDLE_TTL_SECS: \"600\"",
    );
    refused(
        &tree,
        "but the config sets CheckedCachePolicy { enabled: true, idle_ttl_secs: 600",
    );

    let mut tree = Tree::fixture();
    tree.deployment()["prompt_cache"]["max_age_secs"] = 3600.into();
    refused(
        &tree,
        "but the config sets CheckedCachePolicy { enabled: true, idle_ttl_secs: 900, max_age_secs: 7200",
    );

    let mut tree = Tree::fixture();
    tree.edit_config("PREFIX_CACHE: \"true\"", "PREFIX_CACHE: \"false\"");
    refused(&tree, "EIDOLA_ENGINE_KV_HOST_BYTES must be 0");
    tree.edit_config("KV_HOST_BYTES: \"34359738368\"", "KV_HOST_BYTES: \"0\"");
    refused(
        &tree,
        "but the config sets CheckedCachePolicy { enabled: false",
    );

    let mut tree = Tree::fixture();
    tree.deployment()["prompt_cache"]["idle_ttl_secs"] = 9000.into();
    refused(&tree, "but the config sets");
}

#[test]
fn the_gateway_token_is_a_secret_and_only_its_hash_is_measured() {
    let mut tree = Tree::fixture();
    tree.edit_config("      - GATEWAY_TOKEN\n", "");
    refused(&tree, "secrets must be exactly [\"GATEWAY_TOKEN\"]");

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
    refused(&tree, "GATEWAY_TOKEN_HASH: must be an Argon2id hash");

    // The node parses the whole PHC string, not just its prefix.
    let mut tree = Tree::fixture();
    let hash = "$argon2id$v=19$m=19456,t=2,p=1$Mz9P1/uk98yKEflNjzvn5g$unNYT/KTNSNW0JCH9+9OQ2zBApPLxGNZiw746903Q8E";
    tree.edit_config(hash, "$argon2id$garbage");
    refused(&tree, "GATEWAY_TOKEN_HASH: not a valid Argon2 hash string");

    // A well-formed string no token can verify against, as the node would
    // find at boot: a memory cost below Argon2's minimum, no salt or output.
    let mut tree = Tree::fixture();
    tree.edit_config("m=19456,t=2,p=1", "m=1,t=2,p=1");
    refused(&tree, "GATEWAY_TOKEN_HASH: has parameters");
    let mut tree = Tree::fixture();
    tree.edit_config(hash, "$argon2id$v=19$m=19456,t=2,p=1");
    refused(
        &tree,
        "GATEWAY_TOKEN_HASH: must carry a salt and a hash output",
    );
}

/// A retention bound the node refuses at boot is refused here even when the
/// pin and the config agree on it: zero, and anything that overflows the
/// engine's millisecond clock.
#[test]
fn the_prompt_cache_policy_is_one_the_node_boots_with() {
    let max = eidola_common::engine_deployment::MAX_CACHE_SECONDS;
    for (idle, max_age) in [(0u64, 7200u64), (900, max + 1)] {
        let mut tree = Tree::fixture();
        tree.edit_config(
            "CACHE_IDLE_TTL_SECS: \"900\"",
            &format!("CACHE_IDLE_TTL_SECS: \"{idle}\""),
        );
        tree.edit_config(
            "CACHE_MAX_AGE_SECS: \"7200\"",
            &format!("CACHE_MAX_AGE_SECS: \"{max_age}\""),
        );
        tree.deployment()["prompt_cache"]["idle_ttl_secs"] = idle.into();
        tree.deployment()["prompt_cache"]["max_age_secs"] = max_age.into();
        refused(&tree, "must be a positive number of seconds");
    }

    // The largest bound the node accepts is accepted.
    let mut tree = Tree::fixture();
    tree.edit_config(
        "CACHE_MAX_AGE_SECS: \"7200\"",
        &format!("CACHE_MAX_AGE_SECS: \"{max}\""),
    );
    tree.deployment()["prompt_cache"]["max_age_secs"] = max.into();
    tree.check().expect("the node boots with the largest bound");
}

/// `cvm-version` is read by the grammar `measure-enclave` measures with: a
/// bare version and exactly one inline manifest pin.
#[test]
fn the_release_grammar_is_the_measurers() {
    let pinned = format!("0.15.3-rc.1+gpu@sha256:{RELEASE_SHA256}");
    for bad in [
        format!("v{pinned}"),
        format!("{pinned}@sha256:{RELEASE_SHA256}"),
        format!("0.15.3-rc.1+gpu@sha256:{}", "A".repeat(64)),
    ] {
        let mut tree = Tree::fixture();
        tree.edit_config(&pinned, &bad);
        tree.deployment()["cvm_version"] = bad.clone().into();
        refused(&tree, "cvm-version");
        assert!(
            tree.check().unwrap_err().contains("tinfoil-config.yml: "),
            "{bad}"
        );
    }
}

#[test]
fn the_config_runs_one_digest_pinned_engine_with_quoted_values() {
    let mut tree = Tree::fixture();
    tree.edit_config(&format!("@sha256:{}", "2".repeat(64)), ":latest");
    refused(&tree, "runs exactly one container");

    let mut tree = Tree::fixture();
    let engine = tree.config_text();
    let start = engine.find("  - name: \"eidola-server-engine\"").unwrap();
    let end = engine.find("\nshim:").unwrap();
    let container = engine[start..end].to_string();
    tree.edit_config(&container, &format!("{container}{container}"));
    refused(&tree, "runs exactly one container");

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
    // The sidecar's provenance must be the weights pack's.
    refused(&tree, "the model pack's repo must be");

    // And the pin's must be the sidecar's.
    let mut tree = Tree::fixture();
    tree.deployment()["weights"]["revision"] = "9".repeat(40).into();
    refused(&tree, "weights.repo / weights.revision differ from");

    // The sidecar's GPU count must be the config's; the pin's, the sidecar's.
    let mut tree = Tree::fixture();
    let mut sidecar = tree.sidecar();
    sidecar["expected_gpus"] = 4.into();
    tree.set_sidecar(sidecar);
    refused(&tree, "so expected_gpus must be 8");
    let mut tree = Tree::fixture();
    tree.deployment()["pin"]["expected_gpus"] = 1.into();
    refused(&tree, "pin.expected_gpus differs from");

    // The PCK flags are the sidecar's.
    let mut tree = Tree::fixture();
    let mut sidecar = tree.sidecar();
    sidecar["tdx"]["smt_enabled"] = "true".into();
    tree.set_sidecar(sidecar);
    refused(
        &tree,
        "pin.platform.tdx.policy is not the policy derived from",
    );

    let mut tree = Tree::fixture();
    let mut sidecar = tree.sidecar();
    sidecar["unexpected"] = 1.into();
    tree.set_sidecar(sidecar);
    refused(&tree, "unknown field `unexpected`");
}

/// The sidecar is read with a typed, strict parse at every level: a repeated
/// member (which a plain map would collapse to its last value), an unknown
/// one inside the TDX policy or the weights, and `null` for an optional
/// member are each refused.
#[test]
fn the_sidecar_is_read_strictly_at_every_level() {
    let tree = Tree::fixture();
    let text = String::from_utf8(tree.files[SIDECAR].clone()).unwrap();
    let sidecar = tree.sidecar();
    let machines = serde_json::to_string(&sidecar["tdx"]).unwrap();
    let null_machines = {
        let mut sidecar = sidecar.clone();
        sidecar["tdx"] = serde_json::Value::Null;
        serde_json::to_string_pretty(&sidecar).unwrap()
    };
    let with = |edited: String| {
        assert_ne!(edited, text, "the edit applied");
        let mut tree = Tree::fixture();
        tree.files.insert(SIDECAR.to_string(), edited.into_bytes());
        tree
    };
    for (name, edited, needle) in [
        (
            "tdx",
            text.replacen("\"tdx\":", &format!("\"tdx\": {machines}, \"tdx\":"), 1),
            "duplicate field `tdx`",
        ),
        (
            "an endorsements member",
            text.replacen(
                "\"policy\":",
                "\"policy\": \"tdx-h200-prod\", \"policy\":",
                1,
            ),
            "duplicate field `policy`",
        ),
        (
            "weights.revision",
            text.replacen(
                "\"revision\":",
                &format!("\"revision\": \"{}\", \"revision\":", "4".repeat(40)),
                1,
            ),
            "duplicate field `revision`",
        ),
        // A policy member written by hand: the policy is only ever derived.
        (
            "a hand-written policy member",
            text.replacen(
                "\"dynamic_platform\":",
                "\"xfam\": \"e702060000000000\", \"dynamic_platform\":",
                1,
            ),
            "unknown field `xfam`",
        ),
        (
            "an unknown endorsements member",
            text.replacen("\"policy\":", "\"debug\": \"ok\", \"policy\":", 1),
            "unknown field `debug`",
        ),
        (
            "an unknown weights member",
            text.replacen("\"revision\":", "\"branch\": \"main\", \"revision\":", 1),
            "unknown field `branch`",
        ),
        (
            "a null GPU count",
            text.replacen("\"expected_gpus\": 8", "\"expected_gpus\": null", 1),
            "invalid type: null",
        ),
        ("null machines", null_machines, "invalid type: null"),
    ] {
        let err = with(edited).check().expect_err(name);
        assert!(
            err.contains("deployment.json") && err.contains(needle),
            "{name}: {err}"
        );
    }
}

#[test]
fn a_pin_names_exactly_one_platform() {
    let mut tree = Tree::fixture();
    tree.deployment()["pin"]["platform"]["sev_snp"] =
        serde_json::json!({"measurement": "a".repeat(96)});
    // The typed parse (the runtime's) refuses a second platform member.
    refused(&tree, "engine-enclaves.json: ");

    // SEV-SNP stays expressible, and then the sidecar states no TDX policy.
    // Its HOST_DATA is the config's hash, as MRCONFIGID is on TDX.
    let mut tree = Tree::fixture();
    let config_sha256 = tree.deployment()["config_sha256"].clone();
    let snp = |host_data: serde_json::Value| {
        serde_json::json!({"sev_snp": {
            "measurement": "a".repeat(96),
            "host_data": host_data,
            "guest_policy": "0x30133",
        }})
    };
    tree.deployment()["pin"]["platform"] = snp(config_sha256.clone());
    refused(&tree, "a SEV-SNP pin, but");
    let mut sidecar = tree.sidecar();
    sidecar.as_object_mut().unwrap().remove("tdx");
    tree.set_sidecar(sidecar);
    tree.check().expect("a SEV-SNP deployment is expressible");

    let mut other = tree.clone();
    other.deployment()["pin"]["platform"] = snp("b".repeat(64).into());
    refused(&other, "sev_snp.host_data is not the hash of");
    let mut legacy = tree.clone();
    legacy.deployment()["pin"]["platform"] =
        serde_json::json!({"sev_snp": {"measurement": "a".repeat(96)}});
    // A digest alone, the pre-IGVM pin shape, is no pin: the typed parse
    // refuses it.
    refused(&legacy, "missing field `host_data`");
    let mut debug = tree.clone();
    debug.deployment()["pin"]["platform"]["sev_snp"]["guest_policy"] = "0xb0133".into();
    refused(&debug, "the attesting client refuses this pin");
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

/// Every variable the node requires, omitted or invalid, is refused: the
/// check runs the whole measured env through the node's boot grammar
/// (`eidola_common::engine_deployment::parse_measured`), not a chosen subset.
#[test]
fn every_variable_the_node_requires_is_checked() {
    use eidola_common::engine_deployment::env;
    let required = [
        env::MODEL_ID,
        env::WEIGHTS_DIR,
        env::WEIGHTS_SHA256,
        env::WEIGHTS_STORAGE,
        env::GATEWAY_TOKEN_HASH,
        env::EXECUTOR,
        env::BIND_ADDR,
        env::KV_BLOCK_SIZE,
        env::KV_DEVICE_BYTES,
        env::KERNELS_DIR,
        env::CUDA_GRAPHS,
        env::GPU_CC,
        env::MAX_MODEL_LEN,
        env::MAX_SEQS,
        env::MAX_BATCHED_TOKENS,
        env::MAX_PREFILL_CHUNK,
        env::DRAFT_TOKENS,
        env::MAX_REQUESTS,
        env::PREFIX_CACHE,
        env::CACHE_IDLE_TTL_SECS,
        env::CACHE_MAX_AGE_SECS,
    ];
    let text = Tree::fixture().config_text();
    for name in required {
        let line = text
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("- {name}:")))
            .unwrap_or_else(|| panic!("the fixture sets {name}"))
            .to_string();

        let mut tree = Tree::fixture();
        tree.edit_config(&format!("{line}\n"), "");
        refused(&tree, &format!("{name} is not set"));

        let mut tree = Tree::fixture();
        tree.edit_config(&line, &format!("      - {name}: \"\""));
        refused(&tree, &format!("{name} is not set"));
    }

    for (name, bad) in [
        (env::EXECUTOR, "gpu"),
        (env::BIND_ADDR, "localhost"),
        (env::KV_BLOCK_SIZE, "0"),
        (env::KV_DEVICE_BYTES, "0"),
        (env::KERNELS_DIR, "kernels"),
        (env::CUDA_GRAPHS, "yes"),
        (env::GPU_CC, "optional"),
        // Past MiMo-V2.6-Flash's three MTP layers, the cuda executor's bound.
        (env::DRAFT_TOKENS, "4"),
        (env::MAX_MODEL_LEN, "-1"),
        (env::MAX_SEQS, "0"),
        (env::MAX_BATCHED_TOKENS, "many"),
        (env::MAX_PREFILL_CHUNK, "0"),
        (env::DRAFT_TOKENS, "-1"),
        (env::MAX_REQUESTS, "0"),
        (env::MODEL_ID, "has space"),
        (env::WEIGHTS_SHA256, "abc"),
    ] {
        let line = text
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("- {name}:")))
            .unwrap()
            .to_string();
        let mut tree = Tree::fixture();
        tree.edit_config(&line, &format!("      - {name}: \"{bad}\""));
        refused(
            &tree,
            &format!("the node would refuse this env: configuration refused: {name}"),
        );
    }

    // The cpu executor's block count is refused with cuda, not ignored.
    let mut tree = Tree::fixture();
    tree.edit_config(
        "      - EIDOLA_ENGINE_EXECUTOR: \"cuda\"\n",
        "      - EIDOLA_ENGINE_EXECUTOR: \"cuda\"\n      - EIDOLA_ENGINE_KV_BLOCKS: \"256\"\n",
    );
    refused(
        &tree,
        "configuration refused: EIDOLA_ENGINE_KV_BLOCKS applies only to the cpu executor",
    );
}

/// A TDX pin's machine policy is exactly the one derived from the signed
/// platform endorsements its sidecar names: a pin edited in any member — a
/// lowered floor, an added machine, another TDX module, unsafe attributes,
/// a shape the client cannot read — is refused, as is a pin left behind
/// when the sidecar names another policy.
#[test]
fn a_tdx_policy_other_than_the_derived_one_is_refused() {
    type Edit = fn(&mut serde_json::Value);
    let cases: [(&str, Edit); 8] = [
        ("{}", |p| *p = serde_json::json!({})),
        ("a lowered evaluation-data floor", |p| {
            p["minimum_tcb_evaluation_data_number"] = 17.into()
        }),
        ("a lowered TEE_TCB_SVN floor", |p| {
            p["minimum_tee_tcb_svn"] = "00".repeat(16).into()
        }),
        ("an added machine", |p| {
            p["ppids"]
                .as_array_mut()
                .unwrap()
                .push("00".repeat(16).into())
        }),
        ("no machine", |p| p["ppids"] = serde_json::json!([])),
        ("another TDX module", |p| {
            p["mr_seam"] = serde_json::json!(["55".repeat(48)])
        }),
        // DEBUG set (bit 0) alongside SEPT_VE_DISABLE.
        ("debug attributes", |p| {
            p["td_attributes"] = "0100001000000000".into()
        }),
        ("an FMSPC allowlist", |p| {
            p["fmspc"] = serde_json::json!(["b0c06f000000"])
        }),
    ];
    for (name, edit) in cases {
        let mut tree = Tree::fixture();
        let mut policy = tree.deployment()["pin"]["platform"]["tdx"]["policy"].clone();
        edit(&mut policy);
        tree.deployment()["pin"]["platform"]["tdx"]["policy"] = policy;
        let err = tree.check().expect_err(name);
        assert!(
            err.starts_with("engine-enclaves.json: ")
                || err.contains("pin.platform.tdx.policy is not the policy derived from"),
            "{name}: {err}"
        );
    }

    // The sidecar names another policy of the same release; the pin still
    // carries the old one.
    let mut tree = Tree::fixture();
    let mut sidecar = tree.sidecar();
    sidecar["tdx"]["platform_endorsements"]["policy"] = "tdx-h200-prod".into();
    tree.set_sidecar(sidecar);
    refused(
        &tree,
        "pin.platform.tdx.policy is not the policy derived from",
    );
}

/// The endorsements are verified on every check: a vendored file edited
/// (with the sidecar's hash moved to match), a hash the file does not have,
/// a release that is not vendored, and a bundle that is not the release's
/// are each refused, naming the sidecar.
#[test]
fn the_endorsements_a_pin_is_derived_from_are_verified() {
    let at = "deployment.json: tdx.platform_endorsements: platform endorsements refused";

    let mut tree = Tree::fixture();
    let text = String::from_utf8(tree.files[ENDORSEMENTS].clone()).unwrap();
    let edited = text.replacen(
        "\"minimum_tcb_evaluation_data_number\": 19",
        "\"minimum_tcb_evaluation_data_number\": 17",
        1,
    );
    assert_ne!(edited, text);
    let edited_sha: String = {
        use sha2::Digest as _;
        sha2::Sha256::digest(edited.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    };
    tree.files
        .insert(ENDORSEMENTS.to_string(), edited.into_bytes());
    let mut sidecar = tree.sidecar();
    sidecar["tdx"]["platform_endorsements"]["sha256"] = edited_sha.into();
    tree.set_sidecar(sidecar);
    refused(&tree, "the attestation's subject is");
    refused(&tree, at);

    let mut tree = Tree::fixture();
    let mut sidecar = tree.sidecar();
    sidecar["tdx"]["platform_endorsements"]["sha256"] = "00".repeat(32).into();
    tree.set_sidecar(sidecar);
    refused(&tree, "but the recorded hash is");

    let mut tree = Tree::fixture();
    let mut sidecar = tree.sidecar();
    sidecar["tdx"]["platform_endorsements"]["release"] = "platform-v0.1.9".into();
    tree.set_sidecar(sidecar);
    refused(
        &tree,
        "read releases/trust/platform-endorsements/platform-v0.1.9/platform-endorsements.json",
    );

    // The bundle replaced by another genuine Sigstore bundle (the trusted
    // root's own bytes are not one; any other bundle fails its identity).
    let mut tree = Tree::fixture();
    let mut bundle: serde_json::Value =
        serde_json::from_slice(&tree.files[ENDORSEMENTS_BUNDLE]).unwrap();
    bundle["dsseEnvelope"]["signatures"][0]["sig"] = "MEUCIQ==".into();
    tree.files.insert(
        ENDORSEMENTS_BUNDLE.to_string(),
        serde_json::to_vec(&bundle).unwrap(),
    );
    refused(&tree, "the DSSE signature does not verify");
}

/// Sets `expected_gpus` in the pin and the sidecar alike, so only the GPU rule
/// can object.
fn set_expected_gpus(tree: &mut Tree, gpus: Option<u64>) {
    let mut sidecar = tree.sidecar();
    match gpus {
        Some(n) => {
            tree.deployment()["pin"]["expected_gpus"] = n.into();
            sidecar["expected_gpus"] = n.into();
        }
        None => {
            tree.deployment()["pin"]
                .as_object_mut()
                .unwrap()
                .remove("expected_gpus");
            sidecar.as_object_mut().unwrap().remove("expected_gpus");
        }
    }
    tree.set_sidecar(sidecar);
}

/// A CUDA deployment attaches one to eight GPUs (`gpus`; Tinfoil's validator
/// range, less zero) and its pin requires exactly that many evidence items,
/// which is what the shim collects; a CPU deployment attaches and requires
/// none.
#[test]
fn a_cuda_deployment_attests_its_gpus() {
    for gpus in [None, Some(0), Some(4)] {
        let mut tree = Tree::fixture();
        set_expected_gpus(&mut tree, gpus);
        refused(&tree, "so expected_gpus must be 8");
    }

    // No stated `gpus`: the CVM has none and the shim collects no evidence.
    let mut tree = Tree::fixture();
    tree.edit_config("gpus: 8\n", "");
    refused(&tree, "the cuda executor needs the config to attach GPUs");

    // More GPUs than Tinfoil's validator allows.
    let mut tree = Tree::fixture();
    tree.edit_config("gpus: 8\n", "gpus: 9\n");
    set_expected_gpus(&mut tree, Some(9));
    refused(&tree, "gpus must be between 0 and 8");

    for n in [1, 4] {
        let mut tree = Tree::fixture();
        tree.edit_config("gpus: 8\n", &format!("gpus: {n}\n"));
        set_expected_gpus(&mut tree, Some(n));
        tree.check()
            .expect("a CUDA deployment of any count it attests");
    }

    // A pinned deployment runs the CUDA executor: a CPU deployment is refused
    // even when it is consistent in every other way (no GPUs, no evidence).
    let mut tree = Tree::fixture();
    tree.edit_config("gpus: 8\n", "");
    tree.edit_config("    runtime: nvidia\n", "");
    tree.edit_config("    gpus: all\n", "");
    tree.edit_config("EXECUTOR: \"cuda\"", "EXECUTOR: \"cpu\"");
    tree.edit_config(
        "      - EIDOLA_ENGINE_KV_DEVICE_BYTES: \"68719476736\"\n",
        "      - EIDOLA_ENGINE_KV_BLOCKS: \"4096\"\n",
    );
    tree.edit_config(
        "      - EIDOLA_ENGINE_KERNELS_DIR: \"/opt/eidola/kernels\"\n",
        "",
    );
    tree.edit_config("      - EIDOLA_ENGINE_CUDA_GRAPHS: \"on\"\n", "");
    tree.edit_config("      - EIDOLA_ENGINE_GPU_CC: \"required\"\n", "");
    set_expected_gpus(&mut tree, None);
    refused(&tree, "a pinned deployment runs the cuda executor");
}

/// Every allocation-driving size is capped, and what the node allocates fits
/// the VM's memory and the attached GPUs.
#[test]
fn allocations_are_capped_and_fit_the_deployment() {
    for (name, from, to) in [
        ("EIDOLA_ENGINE_MAX_SEQS", "\"64\"", "\"4294967295\""),
        ("EIDOLA_ENGINE_MAX_SEQS", "\"64\"", "\"1025\""),
        ("EIDOLA_ENGINE_MAX_BATCHED_TOKENS", "\"8192\"", "\"65537\""),
        ("EIDOLA_ENGINE_MAX_PREFILL_CHUNK", "\"4096\"", "\"65537\""),
        ("EIDOLA_ENGINE_MAX_MODEL_LEN", "\"131072\"", "\"1048577\""),
        ("EIDOLA_ENGINE_MAX_REQUESTS", "\"8\"", "\"4097\""),
        ("EIDOLA_ENGINE_DRAFT_TOKENS", "\"3\"", "\"9\""),
        ("EIDOLA_ENGINE_KV_BLOCK_SIZE", "\"16\"", "\"1025\""),
        (
            "EIDOLA_ENGINE_KV_DEVICE_BYTES",
            "\"68719476736\"",
            "\"309237645313\"",
        ),
    ] {
        let mut tree = Tree::fixture();
        tree.edit_config(&format!("{name}: {from}"), &format!("{name}: {to}"));
        refused(&tree, &format!("{name} must be at most"));
    }

    // Host, near the limit: each request slot can hold 645,922,816 bytes at
    // worst (a 32 MiB body of up to 2^18 JSON values with both parses in the
    // read pool, and its tree, prompt and tokens in the admission pool); 99
    // fit 64 GiB beside the headroom and tables, 100 do not (with no host
    // tier).
    let no_tier = |tree: &mut Tree| {
        tree.edit_config("KV_HOST_BYTES: \"34359738368\"", "KV_HOST_BYTES: \"0\"");
    };
    let mut tree = Tree::fixture();
    no_tier(&mut tree);
    tree.edit_config("MAX_REQUESTS: \"8\"", "MAX_REQUESTS: \"99\"");
    tree.check().expect("99 request slots fit 64 GiB");
    let mut tree = Tree::fixture();
    no_tier(&mut tree);
    tree.edit_config("MAX_REQUESTS: \"8\"", "MAX_REQUESTS: \"100\"");
    refused(&tree, "more than the VM's");
    // The prefix cache's host tier is page-locked VM memory beside all that:
    // with the fixture's 8 slots, a tier of exactly the rest of 64 GiB fits,
    // a byte more does not; and it is capped at the largest VM.
    let measured = fixture_measured(&Tree::fixture());
    let rest = (65_536u64 << 20) - engine_deployment::host_memory_bytes(&measured.sizing);
    let mut tree = Tree::fixture();
    tree.edit_config(
        "KV_HOST_BYTES: \"34359738368\"",
        &format!("KV_HOST_BYTES: \"{rest}\""),
    );
    tree.check().expect("the host tier fills the VM exactly");
    let mut tree = Tree::fixture();
    tree.edit_config(
        "KV_HOST_BYTES: \"34359738368\"",
        &format!("KV_HOST_BYTES: \"{}\"", rest + 1),
    );
    refused(&tree, "host tier (EIDOLA_ENGINE_KV_HOST_BYTES)");
    let mut tree = Tree::fixture();
    tree.edit_config(
        "KV_HOST_BYTES: \"34359738368\"",
        &format!(
            "KV_HOST_BYTES: \"{}\"",
            engine_deployment::caps::KV_HOST_BYTES + 1
        ),
    );
    refused(&tree, "EIDOLA_ENGINE_KV_HOST_BYTES must be at most");
    // Device: the KV budget holds one longest sequence, every seat's sliding
    // windows and, drafting three tokens, the drafter pool of as many blocks
    // with their boundary taps, each pool with its pad block, at their
    // smallest (`engine_deployment::cuda_kv_min_bytes`, from the fixture's own
    // measured sizing).
    let measured = fixture_measured(&Tree::fixture());
    assert_eq!(measured.sizing.draft_tokens, 3);
    let kv_min = engine_deployment::cuda_kv_min_bytes(FLASH, &measured.sizing, &measured.cache);
    let mut tree = Tree::fixture();
    tree.edit_config(
        "KV_DEVICE_BYTES: \"68719476736\"",
        &format!("KV_DEVICE_BYTES: \"{kv_min}\""),
    );
    tree.check().expect("the smallest KV budget fits");
    let mut tree = Tree::fixture();
    tree.edit_config(
        "KV_DEVICE_BYTES: \"68719476736\"",
        &format!("KV_DEVICE_BYTES: \"{}\"", kv_min - 1),
    );
    refused(
        &tree,
        &format!("less than the {kv_min} bytes the KV pools need"),
    );
    // The undrafted minimum leaves no room for the drafter pool.
    let undrafted = engine_deployment::Sizing {
        draft_tokens: 0,
        ..measured.sizing
    };
    let undrafted_min = engine_deployment::cuda_kv_min_bytes(FLASH, &undrafted, &measured.cache);
    assert!(undrafted_min < kv_min);
    let mut tree = Tree::fixture();
    tree.edit_config(
        "KV_DEVICE_BYTES: \"68719476736\"",
        &format!("KV_DEVICE_BYTES: \"{undrafted_min}\""),
    );
    refused(
        &tree,
        &format!("less than the {kv_min} bytes the KV pools need"),
    );

    // The device holds the KV budget, the weights (the pack's data size) and
    // the executor's reserve (`engine_deployment::cuda_device_reserve_bytes`
    // for this sizing, the drafter's seat rows, step tables, third group of
    // block tables and wider partial rows included) within the B300's
    // `SUPPORTED_GPU_MEMORY_BYTES`.
    let reserve = engine_deployment::cuda_device_reserve_bytes(FLASH, &measured.sizing);
    let room = engine_deployment::SUPPORTED_GPU_MEMORY_BYTES - FIXTURE_PACK_BYTES - reserve;
    let mut tree = Tree::fixture();
    tree.edit_config(
        "KV_DEVICE_BYTES: \"68719476736\"",
        &format!("KV_DEVICE_BYTES: \"{room}\""),
    );
    tree.check().expect("the budget fills the device exactly");
    let mut tree = Tree::fixture();
    tree.edit_config(
        "KV_DEVICE_BYTES: \"68719476736\"",
        &format!("KV_DEVICE_BYTES: \"{}\"", room + 1),
    );
    refused(&tree, "more than the NVIDIA B300 SXM6 AC's 287428640768");
    // A Flash-sized pack leaves room for 64 GiB, and not for twice that.
    let flash_pack = "_177700003840_";
    let mut tree = Tree::fixture();
    tree.edit_config("_17419419648_", flash_pack);
    tree.check().expect("64 GiB of KV beside Flash's weights");
    let mut tree = Tree::fixture();
    tree.edit_config("_17419419648_", flash_pack);
    tree.edit_config(
        "KV_DEVICE_BYTES: \"68719476736\"",
        "KV_DEVICE_BYTES: \"137438953472\"",
    );
    refused(
        &tree,
        "of device memory, more than the NVIDIA B300 SXM6 AC's",
    );

    // An undrafted cuda deployment fits too: no drafter pool, and the smaller
    // reserve.
    let mut tree = Tree::fixture();
    tree.edit_config("DRAFT_TOKENS: \"3\"", "DRAFT_TOKENS: \"0\"");
    tree.check().expect("a cuda deployment may draft nothing");

    // The kernels ship in the image, never in an attached mount.
    let mut tree = Tree::fixture();
    tree.edit_config(
        "\"/opt/eidola/kernels\"",
        "\"/tinfoil/models/weights/kernels\"",
    );
    refused(
        &tree,
        "EIDOLA_ENGINE_KERNELS_DIR must be a clean absolute path inside the image",
    );
}

/// The shim forwards to the port the node listens on, on every interface.
#[test]
fn the_shim_reaches_the_node() {
    let mut tree = Tree::fixture();
    tree.edit_config("upstream-port: 8080", "upstream-port: 8081");
    refused(
        &tree,
        "shim.upstream-port is 8081, but EIDOLA_ENGINE_BIND_ADDR listens on port 8080",
    );

    let mut tree = Tree::fixture();
    tree.edit_config("\"0.0.0.0:8080\"", "\"127.0.0.1:8080\"");
    refused(&tree, "must listen on every interface");

    let mut tree = Tree::fixture();
    tree.edit_config("  upstream-port: 8080\n", "");
    refused(&tree, "missing field `upstream-port`");

    let mut tree = Tree::fixture();
    tree.edit_config("\"0.0.0.0:8080\"", "\"[::]:8080\"");
    tree.check()
        .expect("the IPv6 unspecified address listens everywhere");
}

/// The engine is the only container, and its image is pinned by a full
/// digest: a companion container, a tag, or a short digest would run code the
/// measured config bytes do not fix.
#[test]
fn every_image_is_the_digest_pinned_engine() {
    let digest = format!("@sha256:{}", "2".repeat(64));
    let text = Tree::fixture().config_text();
    let start = text.find("  - name: \"eidola-server-engine\"").unwrap();
    let end = text.find("\nshim:").unwrap();
    let engine = text[start..end].to_string();

    for companion in [
        "  - name: \"sidecar\"\n    image: \"ghcr.io/example/sidecar:latest\"\n    env: []\n",
        &format!(
            "  - name: \"sidecar\"\n    image: \"ghcr.io/example/sidecar{digest}\"\n    env: []\n"
        ),
    ] {
        let mut tree = Tree::fixture();
        tree.edit_config(&engine, &format!("{engine}{companion}"));
        refused(&tree, "runs exactly one container");
    }

    for image in [
        "ghcr.io/eidola-ai/eidola-server-engine:v1".to_string(),
        "ghcr.io/eidola-ai/eidola-server-engine@sha256:2222".to_string(),
        format!("ghcr.io/eidola-ai/eidola-server-engine{digest}:latest"),
    ] {
        let mut tree = Tree::fixture();
        tree.edit_config(
            &format!("ghcr.io/eidola-ai/eidola-server-engine{digest}"),
            &image,
        );
        refused(&tree, "runs exactly one container");
    }
}

/// A repeated member at any level is refused by the build check, as it is by
/// the runtime's reading of the file: accepted at build means readable at
/// runtime.
#[test]
fn a_repeated_member_anywhere_is_refused() {
    let tree = Tree::fixture();
    let text = serde_json::to_string(&tree.json).unwrap();
    let pin = serde_json::to_string(&tree.json["models"][MODEL][0]["pin"]).unwrap();
    let deployments = serde_json::to_string(&tree.json["models"][MODEL]).unwrap();
    let weights = serde_json::to_string(&tree.json["models"][MODEL][0]["weights"]).unwrap();
    let policy =
        serde_json::to_string(&tree.json["models"][MODEL][0]["pin"]["platform"]["tdx"]["policy"])
            .unwrap();
    assert!(tree.check_text(&text).is_ok(), "the unedited text passes");

    for (name, edited) in [
        (
            "pin",
            text.replacen("\"pin\":", &format!("\"pin\":{pin},\"pin\":"), 1),
        ),
        (
            "model id",
            text.replacen(
                &format!("\"{MODEL}\":"),
                &format!("\"{MODEL}\":{deployments},\"{MODEL}\":"),
                1,
            ),
        ),
        (
            "weights",
            text.replacen(
                "\"weights\":",
                &format!("\"weights\":{weights},\"weights\":"),
                1,
            ),
        ),
        (
            "policy",
            text.replacen(
                "\"policy\":",
                &format!("\"policy\":{policy},\"policy\":"),
                1,
            ),
        ),
        (
            "schema_version",
            text.replacen(
                "\"schema_version\":1",
                "\"schema_version\":1,\"schema_version\":1",
                1,
            ),
        ),
    ] {
        assert_ne!(edited, text, "{name}: the edit applied");
        let err = tree.check_text(&edited).expect_err(name);
        assert!(err.contains("duplicate"), "{name}: {err}");
        assert!(
            file::parse(edited.as_bytes()).is_err(),
            "{name}: the runtime refuses it too"
        );
    }
}

/// Only the keys an engine deployment needs are accepted, at every level that
/// can change what runs or what it reaches: an `entrypoint` or `command`
/// would run another executable from the pinned image, and every other key
/// Tinfoil's schema offers is refused unless listed as needed.
#[test]
fn the_config_uses_only_the_keys_an_engine_needs() {
    let image = format!(
        "    image: \"ghcr.io/eidola-ai/eidola-server-engine@sha256:{}\"\n",
        "2".repeat(64)
    );
    for (extra, key) in [
        ("    entrypoint: [\"/bin/sh\"]\n", "entrypoint"),
        ("    command: [\"--serve-something-else\"]\n", "command"),
        ("    privileged: true\n", "privileged"),
        ("    volumes: [\"/:/host\"]\n", "volumes"),
        ("    no_such_key: 1\n", "no_such_key"),
    ] {
        let mut tree = Tree::fixture();
        tree.edit_config(&image, &format!("{image}{extra}"));
        refused(&tree, &format!("unknown field `{key}`"));
    }

    for (extra, key) in [
        ("networks:\n  egress:\n    egress: open\n", "networks"),
        ("cvm-network:\n  inbound-ports: [22]\n", "cvm-network"),
        ("attested-keys: []\n", "attested-keys"),
    ] {
        let mut tree = Tree::fixture();
        tree.edit_config("gpus: 8\n", &format!("gpus: 8\n{extra}"));
        refused(&tree, &format!("unknown field `{key}`"));
    }

    let mut tree = Tree::fixture();
    tree.edit_config(
        "  upstream-port: 8080\n",
        "  upstream-port: 8080\n  dummy-attestation: true\n",
    );
    refused(&tree, "unknown field `dummy-attestation`");

    let mut tree = Tree::fixture();
    tree.edit_config(
        "  - name: \"weights\"\n",
        "  - name: \"weights\"\n    exec: true\n",
    );
    refused(&tree, "unknown field `exec`");
}

/// The CUDA engine sees every attested GPU through the NVIDIA runtime; the
/// CPU engine takes no GPU access.
#[test]
fn the_engine_container_has_the_gpu_access_its_executor_needs() {
    for (from, to) in [
        ("    gpus: all\n", "    gpus: 1\n"),
        ("    gpus: all\n", ""),
        ("    runtime: nvidia\n", ""),
    ] {
        let mut tree = Tree::fixture();
        tree.edit_config(from, to);
        refused(&tree, "needs `runtime: nvidia` and `gpus: all`");
    }
}

/// The shim exposes exactly the node's routes (chat, info, health), by the
/// shim's own matching: an empty list (which the shim reads as "everything"),
/// a partial one, or a pattern reaching none of the node's routes is refused.
#[test]
fn the_shim_exposes_exactly_the_nodes_routes() {
    for (paths, needle) in [
        ("  paths: []\n", "shim.paths must list the node's routes"),
        (
            "  paths:\n    - /v1/chat/completions\n",
            "shim.paths must expose /v1/engine/info",
        ),
        (
            "  paths:\n    - /*\n    - /admin\n",
            "shim.paths entry \"/admin\"",
        ),
    ] {
        let mut tree = Tree::fixture();
        tree.edit_config("  paths:\n    - /*\n", paths);
        refused(&tree, needle);
    }
    let mut tree = Tree::fixture();
    tree.edit_config("  paths:\n    - /*\n", "");
    refused(&tree, "shim.paths must list the node's routes");

    let mut tree = Tree::fixture();
    tree.edit_config(
        "  paths:\n    - /*\n",
        "  paths:\n    - /v1/*\n    - /healthz\n",
    );
    tree.check().expect("the node's routes, listed by prefix");
}

/// The weights the node reads are the one model pack, pinned and granted to
/// the engine container, mounted where the node looks.
#[test]
fn the_weights_dir_is_the_granted_pinned_pack() {
    let pack = "models:\n  - name: \"weights\"\n";
    let text = Tree::fixture().config_text();
    let start = text.find(pack).unwrap();
    let end = start + text[start..].find("\ncontainers:").unwrap() + 1;
    let block = text[start..end].to_string();

    let mut tree = Tree::fixture();
    tree.edit_config(&block, "");
    tree.edit_config("    models: [\"weights\"]\n", "");
    refused(&tree, "declares exactly one model pack");

    let mut tree = Tree::fixture();
    tree.edit_config("    models: [\"weights\"]\n", "");
    refused(&tree, "must be granted exactly the weights pack");

    let mut tree = Tree::fixture();
    tree.edit_config("    models: [\"weights\"]\n", "    models: [\"other\"]\n");
    refused(&tree, "must be granted exactly the weights pack");

    for dir in [
        "/weights",
        "/tinfoil/models/other",
        "/tinfoil/models/weights/../x",
    ] {
        let mut tree = Tree::fixture();
        tree.edit_config("\"/tinfoil/models/weights\"", &format!("\"{dir}\""));
        refused(
            &tree,
            "EIDOLA_ENGINE_WEIGHTS_DIR must be inside the weights pack's mount",
        );
    }

    let mut tree = Tree::fixture();
    tree.edit_config(
        "@4444444444444444444444444444444444444444",
        &format!("@{}", "9".repeat(40)),
    );
    refused(&tree, "the model pack's repo must be");

    let mut tree = Tree::fixture();
    tree.edit_config("    mpk: \"5555", "    mpk: \"zz");
    refused(&tree, "pinned by its artifact reference");

    let mut tree = Tree::fixture();
    tree.edit_config(
        "\"/tinfoil/models/weights\"",
        "\"/tinfoil/models/weights/original\"",
    );
    tree.check().expect("a weights directory inside the pack");
}

/// Any secret besides the gateway token is refused: it would reach the node
/// outside the measurement.
#[test]
fn the_gateway_token_is_the_only_secret() {
    let mut tree = Tree::fixture();
    tree.edit_config(
        "      - GATEWAY_TOKEN\n",
        "      - GATEWAY_TOKEN\n      - RUST_LOG\n",
    );
    refused(&tree, "secrets must be exactly [\"GATEWAY_TOKEN\"]");
}

/// The env sets exactly the node's measured variables: a name any dependency
/// reads (tokio's worker count, the loader's preload, CUDA's device list) or
/// any other name is refused.
#[test]
fn the_env_is_exactly_the_measured_variables() {
    for name in [
        "TOKIO_WORKER_THREADS",
        "LD_PRELOAD",
        "CUDA_VISIBLE_DEVICES",
        "SOMETHING_ELSE",
    ] {
        let mut tree = Tree::fixture();
        tree.edit_config(
            "      - EIDOLA_ENGINE_MODEL_ID",
            &format!("      - {name}: \"0\"\n      - EIDOLA_ENGINE_MODEL_ID"),
        );
        refused(&tree, &format!("env may not set {name:?}"));
    }
}

/// The VM's cpus and memory are a launchable shape.
#[test]
fn the_vm_is_a_launchable_shape() {
    for (from, to, needle) in [
        (
            "memory: 65536",
            "memory: 1",
            "memory must be a whole number of GiB",
        ),
        (
            "memory: 65536",
            "memory: 65000",
            "memory must be a whole number of GiB",
        ),
        ("cpus: 16", "cpus: 0", "cpus must be from"),
        ("cpus: 16", "cpus: 1000", "cpus must be from"),
    ] {
        let mut tree = Tree::fixture();
        tree.edit_config(from, to);
        refused(&tree, needle);
    }
}

/// A weights pack states the layout it was built with, one of the known
/// schemas: its root hash depends on it.
#[test]
fn the_weights_pack_states_a_known_layout() {
    for to in [
        "",
        "    schema: 3\n",
        "    schema: \"2\"\n",
        "    schema: 0\n",
    ] {
        let mut tree = Tree::fixture();
        tree.edit_config("    schema: 2\n", to);
        refused(&tree, "models[0].schema");
    }
    let mut tree = Tree::fixture();
    tree.edit_config("    schema: 2\n", "    schema: 1\n");
    tree.check().expect("schema 1 is known");
}

/// A deployment directory is named by the path-component grammar the
/// measurer walks with: uppercase or a space is refused.
#[test]
fn a_deployment_path_uses_safe_components() {
    for variant in ["TDX-A", "tdx a"] {
        let mut tree = Tree::fixture();
        tree.deployment()["config"] =
            format!("deploy/engine/fixture-model/{variant}/tinfoil-config.yml").into();
        refused(&tree, "config must be deploy/engine/<name>/<variant>/");
    }
}

/// The engine config is parsed into a typed, unknown-field-refusing mirror of
/// Tinfoil's schema and validated with a port of its validator: one violating
/// value per ported rule, each refused.
#[test]
fn the_config_holds_to_tinfoils_decoder_and_validator() {
    let text = Tree::fixture().config_text();

    // decode.go `MaxConfigBytes`: a config over 1 MiB, here by a comment.
    let mut tree = Tree::fixture();
    tree.set_config(format!("{text}#{}\n", "x".repeat(1 << 20)));
    refused(&tree, "exceeds Tinfoil's 1048576-byte config limit");

    // A `%YAML` directive.
    let mut tree = Tree::fixture();
    tree.set_config(format!("%YAML 1.1\n---\n{text}"));
    refused(&tree, "YAML directives are refused");

    // decode.go: one document (a trailing one is refused).
    let mut tree = Tree::fixture();
    tree.set_config(format!("{text}---\n{text}"));
    refused(&tree, "more than one document");

    let cases: &[(&str, &str, &str)] = &[
        // The container's name: required, a string, and the engine's.
        (
            "  - name: \"eidola-server-engine\"\n",
            "  - image_first: 1\n",
            "unknown field",
        ),
        (
            "  - name: \"eidola-server-engine\"\n",
            "  - name: []\n",
            "containers[0].name",
        ),
        (
            "  - name: \"eidola-server-engine\"\n",
            "  - name: \"\"\n",
            "named \"eidola-server-engine\"",
        ),
        (
            "  - name: \"eidola-server-engine\"\n",
            "  - name: \"other\"\n",
            "named \"eidola-server-engine\"",
        ),
        // validation.go `Validate`: gpus 0 to 8.
        ("gpus: 8\n", "gpus: 9\n", "gpus must be between 0 and 8"),
        // shim.go `Validate`: an upstream port is set; it is a port.
        (
            "upstream-port: 8080",
            "upstream-port: 0",
            "shim.upstream-port must be a port number",
        ),
        (
            "upstream-port: 8080",
            "upstream-port: 70000",
            "shim.upstream-port must be a port number",
        ),
        // validation.go `validEnvironmentName`, on env and on secrets.
        (
            "      - EIDOLA_ENGINE_MODEL_ID",
            "      - 1BAD: \"x\"\n      - EIDOLA_ENGINE_MODEL_ID",
            "invalid environment name \"1BAD\"",
        ),
        (
            "      - GATEWAY_TOKEN\n",
            "      - GATEWAY_TOKEN\n      - bad-name\n",
            "invalid environment name \"bad-name\"",
        ),
        // validation.go `validateModelAccess`: a grant listed once.
        (
            "models: [\"weights\"]",
            "models: [\"weights\", \"weights\"]",
            "grant \"weights\" twice",
        ),
        // validation.go `validateContainerPolicy`: runtime empty or nvidia.
        (
            "    runtime: nvidia\n",
            "    runtime: runc\n",
            "runtime must be nvidia",
        ),
        // validation.go `validateShape`: a pack schema is not negative.
        ("    schema: 2\n", "    schema: -1\n", "models[0].schema"),
        // types.go: `int` fields are integers, not quoted numbers.
        ("cpus: 16", "cpus: \"16\"", "cpus"),
        // An env entry that inherits from the host instead of stating a value.
        (
            "      - EIDOLA_ENGINE_MODEL_ID",
            "      - INHERITED\n      - EIDOLA_ENGINE_MODEL_ID",
            "env",
        ),
        // The YAML decoder: one value per key.
        ("cpus: 16\n", "cpus: 16\ncpus: 32\n", "duplicate"),
        // shim.go `validateYAMLTree`: no aliases (here, anywhere).
        (
            "cpus: 16\n",
            "cpus: &n 16\n",
            "anchors and aliases are refused",
        ),
        (
            "memory: 65536\n",
            "memory: *n\n",
            "anchors and aliases are refused",
        ),
        // Read from the parser's events: an anchor on an explicit key too.
        (
            "cpus: 16\n",
            "? &a cpus\n: 16\n",
            "anchors and aliases are refused",
        ),
        // Constructs yaml.v3 and libyaml can read differently.
        ("cpus: 16\n", "<<: {cpus: 16}\n", "merge keys are refused"),
        (
            "cpus: 16\n",
            "cpus: !!int 16\n",
            "explicit YAML tags are refused",
        ),
        (
            "cvm-version: ",
            "cvm-version: !!str ",
            "explicit YAML tags are refused",
        ),
        (
            "cpus: 16\n",
            "\"cpus\": 16\n",
            "mapping keys must be plain scalars",
        ),
        (
            "cpus: 16\n",
            "{cpus: 16}: 16\ncpus: 16\n",
            "mapping keys must be plain scalars",
        ),
        (
            "cpus: 16\n",
            "? [cpus]\n: 16\ncpus: 16\n",
            "mapping keys must be plain scalars",
        ),
    ];
    for (from, to, needle) in cases {
        let mut tree = Tree::fixture();
        tree.edit_config(from, to);
        refused(&tree, needle);
    }
}

/// The weights pack is pinned by a whole modelwrap artifact reference.
#[test]
fn the_weights_pack_is_pinned_by_a_whole_artifact_reference() {
    let reference = "5555555555555555555555555555555555555555555555555555555555555555_17419419648_3892cd2f-a06e-5aee-8276-93140b9f06ec";
    let root = &reference[..64];
    for bad in [
        root.to_string(),
        format!("{root}_garbage"),
        format!("{root}_x_3892cd2f-a06e-5aee-8276-93140b9f06ec"),
    ] {
        let mut tree = Tree::fixture();
        tree.edit_config(reference, &bad);
        refused(&tree, "pinned by its artifact reference");
    }
}
