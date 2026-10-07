//! Truncated real MiMo-V2.6-Flash-MOPD against the HF remote code. Local only:
//! needs the checkpoint subset and goldens produced by `golden/` (see
//! `golden/README.md`), several GB of RAM, and an optimised build:
//!
//! ```text
//! cargo test --release -p eidola-engine-model --test real_flash -- --ignored --nocapture
//! ```
//!
//! `EIDOLA_MIMO_TRUNCATED_DIR` overrides the checkpoint directory and
//! `EIDOLA_MIMO_LAYERS` the kept layers (default `0,1,2,5`);
//! `EIDOLA_MIMO_DUMP_LOGITS=<file>` also writes our logits as raw f32.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use eidola_engine_model::compare::{compare_logits, max_abs_diff};
use eidola_engine_model::safetensors::WeightSet;
use eidola_engine_model::{
    ForwardOptions, LoadOptions, LogitsAt, Matrix, ModelWeights, ReferenceModel,
};

const REVISION_PREFIX: &str = "2479e2d0029e";

fn layers() -> Vec<usize> {
    std::env::var("EIDOLA_MIMO_LAYERS")
        .unwrap_or_else(|_| "0,1,2,5".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect()
}

fn checkpoint_dir(layers: &[usize]) -> PathBuf {
    if let Ok(d) = std::env::var("EIDOLA_MIMO_TRUNCATED_DIR") {
        return d.into();
    }
    let cache = std::env::var("EIDOLA_ENGINE_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/eidola/engine")
        });
    let tag: Vec<String> = layers.iter().map(|l| l.to_string()).collect();
    cache
        .join("mimo-v2.6-flash-mopd")
        .join(REVISION_PREFIX)
        .join(format!("layers-{}-mtp", tag.join("-")))
}

fn matrix(set: &WeightSet, name: &str) -> Matrix {
    let t = set.get(name).unwrap();
    Matrix::from_vec(t.shape[0], t.shape[1], t.to_f32(name).unwrap())
}

fn tokens(set: &WeightSet, name: &str) -> Vec<u32> {
    let t = set.get(name).unwrap();
    t.to_i64(name)
        .unwrap()
        .into_iter()
        .map(|x| x as u32)
        .collect()
}

#[test]
#[ignore = "needs a locally fetched truncated checkpoint; see golden/README.md"]
fn truncated_flash_matches_hf_remote_code() {
    let layers = layers();
    let dir = checkpoint_dir(&layers);
    let golden = WeightSet::open_files(&[dir.join("golden/hf-golden.safetensors")])
        .expect("golden/hf-golden.safetensors");

    let t0 = Instant::now();
    let store = WeightSet::open_dir(&dir).unwrap();
    let manifest: BTreeMap<String, String> =
        serde_json::from_slice(&std::fs::read(dir.join("local-sha256.json")).unwrap()).unwrap();
    store
        .verify_sha256(&manifest)
        .expect("cached shards match their manifest");
    println!(
        "opened + verified {} files in {:.1?}",
        manifest.len(),
        t0.elapsed()
    );
    let tp = store
        .metadata("tp_size")
        .expect("upstream index records tp_size")
        .to_string();
    let store = Arc::new(store);

    let config = store.model_config().unwrap().truncated(&layers).unwrap();
    let t0 = Instant::now();
    let model =
        ReferenceModel::new(ModelWeights::load(store, config, &LoadOptions::default()).unwrap());
    assert_eq!(model.weights.qkv_chunks.to_string(), tp);
    println!("loaded (tp_size {tp}) in {:.1?}", t0.elapsed());

    let toks = tokens(&golden, "tokens");
    let t0 = Instant::now();
    let out = model
        .forward(
            &toks,
            &ForwardOptions {
                logits: LogitsAt::All,
                capture_layers: true,
            },
        )
        .unwrap();
    println!("forward over {} tokens in {:.1?}", toks.len(), t0.elapsed());

    for (i, h) in out.layer_outputs.iter().enumerate() {
        let r = matrix(&golden, &format!("hidden.{i}"));
        let (d, m) = max_abs_diff(&r.data, &h.data).expect("finite values");
        let worst_row = (0..r.rows)
            .max_by(|&a, &b| {
                max_abs_diff(r.row(a), h.row(a))
                    .expect("finite values")
                    .0
                    .total_cmp(&max_abs_diff(r.row(b), h.row(b)).expect("finite values").0)
            })
            .unwrap();
        println!(
            "layer {i} (checkpoint {}): max |Δ| {d:.3e} (scale {m:.3e}) at position {worst_row}",
            layers[i]
        );
    }
    if let Ok(path) = std::env::var("EIDOLA_MIMO_DUMP_LOGITS") {
        let bytes: Vec<u8> = out
            .logits
            .data
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        std::fs::write(&path, bytes).unwrap();
        println!("wrote our logits to {path}");
    }
    let reference = matrix(&golden, "logits");
    let a =
        compare_logits(&reference.data, &out.logits.data, reference.cols).expect("finite logits");
    println!("main logits: {a}");
    assert!(a.top1_rate() >= 0.999, "{a}");
    assert!(a.kl_mean < 1e-3, "{a}");

    if golden.contains("mtp.0.logits") {
        let mt = tokens(&golden, "mtp.0.tokens");
        let prev = matrix(&golden, "mtp.0.prev_hidden");
        let positions: Vec<usize> = (0..mt.len()).collect();
        let t0 = Instant::now();
        let m = model
            .mtp_forward(0, &mt, &prev, &positions, &LogitsAt::All)
            .unwrap();
        println!("mtp 0 forward in {:.1?}", t0.elapsed());
        let (d, s) = max_abs_diff(&matrix(&golden, "mtp.0.hidden").data, &m.hidden.data)
            .expect("finite values");
        println!("mtp 0 hidden: max |Δ| {d:.3e} (scale {s:.3e})");
        let reference = matrix(&golden, "mtp.0.logits");
        let a =
            compare_logits(&reference.data, &m.logits.data, reference.cols).expect("finite logits");
        println!("mtp 0 logits: {a}");
        assert!(a.top1_rate() >= 0.999, "{a}");
        assert!(a.kl_mean < 1e-3, "{a}");
    }
}
