//! The reference forward against goldens produced by the Hugging Face MiMo-V2
//! remote code (fp32, CPU) on the committed synthetic checkpoint. Regenerate
//! with `golden/make_synthetic.py` (see `golden/README.md`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use eidola_engine_model::compare::{compare_logits, max_abs_diff};
use eidola_engine_model::safetensors::WeightSet;
use eidola_engine_model::{
    ForwardOptions, LoadOptions, LogitsAt, Matrix, ModelConfig, ModelWeights, ReferenceModel,
};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny")
}

struct Golden(WeightSet);

impl Golden {
    fn open() -> Self {
        Golden(WeightSet::open_files(&[fixture().join("golden.safetensors")]).unwrap())
    }
    fn matrix(&self, name: &str) -> Matrix {
        let t = self.0.get(name).unwrap();
        assert_eq!(t.shape.len(), 2, "{name}");
        Matrix::from_vec(t.shape[0], t.shape[1], t.to_f32(name).unwrap())
    }
    fn tokens(&self, name: &str) -> Vec<u32> {
        let t = self.0.get(name).unwrap();
        t.to_i64(name)
            .unwrap()
            .into_iter()
            .map(|x| x as u32)
            .collect()
    }
}

fn model(opts: &LoadOptions) -> ReferenceModel {
    let dir = fixture();
    let config = ModelConfig::from_file(&dir.join("config.json")).unwrap();
    let store = Arc::new(WeightSet::open_files(&[dir.join("model.safetensors")]).unwrap());
    ReferenceModel::new(ModelWeights::load(store, config, opts).unwrap())
}

/// Element-wise agreement relative to the reference's scale. Both sides are
/// f32; only summation order differs.
fn assert_close(what: &str, reference: &Matrix, ours: &Matrix, rel: f32) {
    assert_eq!(
        (reference.rows, reference.cols),
        (ours.rows, ours.cols),
        "{what}"
    );
    let (d, m) = max_abs_diff(&reference.data, &ours.data);
    assert!(
        d <= rel * m.max(1.0),
        "{what}: max |Δ| {d:e} vs scale {m:e}"
    );
}

const REL: f32 = 2e-5;

#[test]
fn main_model_matches_hf_remote_code() {
    let g = Golden::open();
    let tokens = g.tokens("tokens");
    let m = model(&LoadOptions::default());
    assert_eq!(m.weights.qkv_chunks, 2);
    let out = m
        .forward(
            &tokens,
            &ForwardOptions {
                logits: LogitsAt::All,
                capture_layers: true,
            },
        )
        .unwrap();
    for (i, h) in out.layer_outputs.iter().enumerate() {
        assert_close(
            &format!("hidden.{i}"),
            &g.matrix(&format!("hidden.{i}")),
            h,
            REL,
        );
    }
    assert_close(
        "hidden_normed",
        &g.matrix("hidden_normed"),
        &out.hidden_normed,
        REL,
    );
    let reference = g.matrix("logits");
    assert_close("logits", &reference, &out.logits, REL);
    let a = compare_logits(&reference.data, &out.logits.data, reference.cols);
    assert_eq!(a.top1_matches, a.rows, "{a}");
    assert!(a.kl_max < 1e-9, "{a}");
}

#[test]
fn mtp_layers_match_hf_modules() {
    let g = Golden::open();
    let m = model(&LoadOptions::default());
    assert_eq!(m.weights.mtp.len(), 2);
    for k in 0..2 {
        let tokens = g.tokens(&format!("mtp.{k}.tokens"));
        let prev = g.matrix(&format!("mtp.{k}.prev_hidden"));
        let positions: Vec<usize> = (0..tokens.len()).collect();
        let out = m
            .mtp_forward(k, &tokens, &prev, &positions, &LogitsAt::All)
            .unwrap();
        assert_close(
            &format!("mtp.{k}.hidden"),
            &g.matrix(&format!("mtp.{k}.hidden")),
            &out.hidden,
            REL,
        );
        assert_close(
            &format!("mtp.{k}.hidden_normed"),
            &g.matrix(&format!("mtp.{k}.hidden_normed")),
            &out.hidden_normed,
            REL,
        );
        let reference = g.matrix(&format!("mtp.{k}.logits"));
        assert_close(&format!("mtp.{k}.logits"), &reference, &out.logits, REL);
        let a = compare_logits(&reference.data, &out.logits.data, reference.cols);
        assert_eq!(a.top1_matches, a.rows, "mtp {k}: {a}");
    }
}

#[test]
fn folded_value_scale_equals_scaling_v() {
    let g = Golden::open();
    let tokens = g.tokens("tokens");
    let folded = model(&LoadOptions::default());
    let unfolded = model(&LoadOptions {
        fold_value_scale: false,
        ..LoadOptions::default()
    });
    assert!(folded.weights.value_scale_folded);
    assert!(!unfolded.weights.value_scale_folded);
    let opts = ForwardOptions::default();
    let a = folded.forward(&tokens, &opts).unwrap();
    let b = unfolded.forward(&tokens, &opts).unwrap();
    // Same function; f32 rounding of where the scale is applied differs.
    assert_close("fold vs unfold", &b.logits, &a.logits, 1e-5);
    assert_ne!(
        a.logits.data, b.logits.data,
        "the two paths should not be the same code"
    );
}

/// A token's outputs do not depend on what follows it or on how many rows
/// share the call: running a prefix reproduces the prefix rows bit for bit.
#[test]
fn prefix_rows_are_bit_identical() {
    let g = Golden::open();
    let tokens = g.tokens("tokens");
    let m = model(&LoadOptions::default());
    let full = m.forward(&tokens, &ForwardOptions::default()).unwrap();
    for len in [1, 9, 23] {
        let part = m
            .forward(&tokens[..len], &ForwardOptions::default())
            .unwrap();
        assert_eq!(
            part.logits.data,
            full.logits.data[..len * full.logits.cols],
            "prefix {len}"
        );
    }
}

#[test]
fn truncated_model_keeps_layer_types() {
    let dir = fixture();
    let config = ModelConfig::from_file(&dir.join("config.json")).unwrap();
    let t = config.truncated(&[0, 3]).unwrap();
    assert_eq!(t.num_layers(), 2);
    assert_eq!(t.global_layer_indices(), vec![0, 3]);
    let store = Arc::new(WeightSet::open_files(&[dir.join("model.safetensors")]).unwrap());
    let m = ReferenceModel::new(ModelWeights::load(store, t, &LoadOptions::default()).unwrap());
    let out = m.forward(&[1, 2, 3], &ForwardOptions::default()).unwrap();
    assert_eq!(out.logits.rows, 3);
}

#[test]
fn integrity_manifest_is_enforced() {
    let dir = fixture();
    let store = WeightSet::open_files(&[dir.join("model.safetensors")]).unwrap();
    let mut manifest = store.sha256_manifest();
    store.verify_sha256(&manifest).unwrap();
    let digest = manifest.get_mut("model.safetensors").unwrap();
    *digest = "0".repeat(64);
    assert!(store.verify_sha256(&manifest).is_err());
    let mut extra = store.sha256_manifest();
    extra.insert("other.safetensors".into(), "0".repeat(64));
    assert!(store.verify_sha256(&extra).is_err());
}

/// One manifest covers every file the loader reads: the shards, `config.json`,
/// and the index whose `tp_size` decides how fused QKV rows de-interleave. A
/// changed index or config fails verification, and the configuration is parsed
/// from exactly the bytes that were hashed.
#[test]
fn integrity_manifest_covers_config_and_index() {
    let src = fixture();
    let dir = std::env::temp_dir().join(format!("eidola-model-semantic-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for f in ["config.json", "model.safetensors"] {
        std::fs::copy(src.join(f), dir.join(f)).unwrap();
    }
    let index = |tp: u32| format!(r#"{{"metadata": {{"tp_size": "{tp}"}}, "weight_map": {{}}}}"#);
    std::fs::write(dir.join("model.safetensors.index.json"), index(4)).unwrap();

    let store = WeightSet::open_dir(&dir).unwrap();
    let manifest = store.sha256_manifest();
    assert_eq!(
        manifest.keys().map(String::as_str).collect::<Vec<_>>(),
        [
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json"
        ]
    );
    store.verify_sha256(&manifest).unwrap();
    assert_eq!(store.metadata("tp_size"), Some("4"));
    let config = store.model_config().unwrap();
    assert_eq!(
        config.num_layers(),
        ModelConfig::from_file(&src.join("config.json"))
            .unwrap()
            .num_layers()
    );

    // A stale index (different tp_size) fails the manifest of the original.
    std::fs::write(dir.join("model.safetensors.index.json"), index(8)).unwrap();
    let tampered = WeightSet::open_dir(&dir).unwrap();
    assert!(tampered.verify_sha256(&manifest).is_err());
    std::fs::write(dir.join("model.safetensors.index.json"), index(4)).unwrap();

    // So does an edited config.
    let mut cfg = std::fs::read_to_string(dir.join("config.json")).unwrap();
    cfg.push('\n');
    std::fs::write(dir.join("config.json"), cfg).unwrap();
    let tampered = WeightSet::open_dir(&dir).unwrap();
    assert!(tampered.verify_sha256(&manifest).is_err());

    // A manifest that omits a semantic file is not a complete one.
    let mut shards_only = manifest.clone();
    shards_only.remove("config.json");
    assert!(store.verify_sha256(&shards_only).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}
