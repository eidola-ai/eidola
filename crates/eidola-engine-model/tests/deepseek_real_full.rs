//! The whole DeepSeek-V4.1-Flash-1017 model, all 40 layers and every routed
//! expert, against the reference run one half at a time
//! (`golden/deepseek_v41/full_halves.py`). Ignored: needs the whole
//! checkpoint (about 510 GB) and goldens made on a machine that can run it;
//! see `golden/deepseek_v41/README.md`.
//!
//!     EIDOLA_DEEPSEEK_FULL_DIR=<dir> cargo test --release -p eidola-engine-model \
//!         --test deepseek_real_full -- --ignored --nocapture
//!
//! Every `golden/reference-*.safetensors` is compared on final logits, with
//! our forward and with the other goldens. `reference-pinned` must be equalled
//! exactly; the others are reported. Without it nothing is asserted.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use eidola_engine_model::WeightSet;
use eidola_engine_model::compare::compare_logits;
use eidola_engine_model::deepseek::engram::{EngramTable, SparseEngramTable};
use eidola_engine_model::deepseek::forward::DsForwardOptions;
use eidola_engine_model::deepseek::{DsLoadOptions, DsModel, DsWeights, config_of};

fn open(path: &Path) -> WeightSet {
    WeightSet::open_files(&[path.to_path_buf()]).unwrap()
}

fn f32s(set: &WeightSet, name: &str) -> Vec<f32> {
    set.get(name).unwrap().to_f32(name).unwrap()
}

fn relative(got: &[f32], want: &[f32]) -> f64 {
    let (mut err, mut norm) = (0.0f64, 0.0f64);
    for (a, b) in got.iter().zip(want) {
        err += ((a - b) as f64).powi(2);
        norm += (*b as f64).powi(2);
    }
    (err / norm.max(f64::MIN_POSITIVE)).sqrt()
}

#[test]
#[ignore = "needs the whole real checkpoint"]
fn whole_model_against_the_reference_run_in_halves() {
    let dir = PathBuf::from(
        std::env::var("EIDOLA_DEEPSEEK_FULL_DIR")
            .expect("EIDOLA_DEEPSEEK_FULL_DIR names the whole checkpoint"),
    );
    // Checking this test without the whole checkpoint: the first N layers as
    // the model (`full_halves.py --first-layers N`), with Engram rows from
    // `golden/engram-rows.safetensors` and goldens from another directory.
    let first_layers: Option<usize> = std::env::var("EIDOLA_DEEPSEEK_FULL_FIRST_LAYERS")
        .ok()
        .map(|n| n.parse().unwrap());
    let golden_dir = std::env::var("EIDOLA_DEEPSEEK_FULL_GOLDEN_DIR")
        .map_or_else(|_| dir.join("golden"), PathBuf::from);
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&golden_dir)
        .expect("the checkpoint directory has a golden/ directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|f| f.to_str())
                .is_some_and(|f| f.starts_with("reference-") && f.ends_with(".safetensors"))
        })
        .collect();
    paths.sort_by_key(|p| (!p.ends_with("reference-pinned.safetensors"), p.clone()));
    assert!(
        !paths.is_empty(),
        "no golden/reference-*.safetensors to compare against"
    );
    let goldens: Vec<(String, WeightSet)> = paths
        .iter()
        .map(|p| {
            (
                p.file_stem().unwrap().to_string_lossy().into_owned(),
                open(p),
            )
        })
        .collect();
    let ints = |set: &WeightSet, name: &str| set.get(name).unwrap().to_i64(name).unwrap();
    let tokens: Vec<u32> = ints(&goldens[0].1, "tokens")
        .iter()
        .map(|t| *t as u32)
        .collect();
    let token_map = ints(&goldens[0].1, "engram.token_map")
        .iter()
        .map(|t| *t as u32)
        .collect();

    // The checkpoint as published: every layer, every expert, the Engram
    // tables read by row from their shards. The draft blocks are not loaded.
    let store = Arc::new(WeightSet::open_dir(&dir).unwrap());
    let mut config = config_of(&store).unwrap();
    let mut engram_tables: Vec<(usize, Arc<dyn EngramTable>)> = Vec::new();
    if let Some(n) = first_layers {
        config = config.truncated(&(0..n).collect::<Vec<_>>()).unwrap();
        let rows = open(&dir.join("golden/engram-rows.safetensors"));
        let spec = config.engram.clone().unwrap();
        for l in spec.layers.iter().filter(|l| l.source_layer < n) {
            let p = format!("layers.{}", l.source_layer);
            let ids: Vec<u64> = ints(&rows, &format!("{p}.rows"))
                .iter()
                .map(|r| *r as u64)
                .collect();
            let table = SparseEngramTable::from_stored(
                l.num_rows,
                spec.head_dim,
                &ids,
                rows.get(&format!("{p}.weight")).unwrap().data,
                rows.get(&format!("{p}.scale")).unwrap().data,
            )
            .unwrap();
            engram_tables.push((l.source_layer, Arc::new(table)));
        }
    }
    let options = DsLoadOptions {
        skip_drafter: true,
        engram_tables,
    };
    let model = DsModel::new(
        DsWeights::load(store, config, &options).unwrap(),
        Some(token_map),
    )
    .unwrap();
    let c = model.config();
    let (n, v, layers) = (tokens.len(), c.vocab_size, c.layers.len());
    println!(
        "{n} tokens, {layers} layers, {} routed experts per layer",
        c.moe.n_routed
    );

    let run = |reference_prefill: Option<usize>| {
        let began = std::time::Instant::now();
        let opts = DsForwardOptions {
            taps: true,
            reference_prefill,
            ..Default::default()
        };
        let out = model
            .forward(&mut model.new_state(), &tokens, &opts)
            .unwrap();
        println!("  (a forward took {:.0} s)", began.elapsed().as_secs_f64());
        out
    };
    // The reference's one prefill chunk in its own layout, and the canonical
    // layout (the reference's for one token per call).
    let prefill = run(Some(n));
    let canonical = run(None);
    let ours = compare_logits(&prefill.logits.data, &canonical.logits.data, v).unwrap();
    println!("our canonical layout against our reproduction of the prefill layout: {ours}");

    let mut failures = Vec::new();
    for (name, golden) in &goldens {
        let exact_expected = name == "reference-pinned";
        println!(
            "== {name}{}",
            if exact_expected {
                ""
            } else {
                " (measurement only)"
            }
        );
        for (pattern, mine) in [
            ("prefill", &prefill),
            ("reshaped", &prefill),
            ("decode", &canonical),
        ] {
            if !golden.contains(&format!("{pattern}.logits")) {
                continue;
            }
            let want = f32s(golden, &format!("{pattern}.logits"));
            let a = compare_logits(&want, &mine.logits.data, v).unwrap();
            let differing = mine
                .logits
                .data
                .iter()
                .zip(&want)
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "ours against its {pattern} pattern: {a}; {differing} of {} logits differ",
                want.len()
            );
            if exact_expected && differing != 0 {
                failures.push(format!("{name} {pattern}: {differing} logits differ"));
            }
            for l in (0..layers).filter(|l| golden.contains(&format!("{pattern}.layer.{l}"))) {
                let want = f32s(golden, &format!("{pattern}.layer.{l}"));
                println!(
                    "  streams after layer {l:>2}: relative error {:.3e}",
                    relative(&mine.taps[l].data, &want)
                );
            }
        }
        // The reference against itself, within one golden.
        let pair = |a: &str, b: &str| {
            if golden.contains(&format!("{a}.logits")) && golden.contains(&format!("{b}.logits")) {
                let m = compare_logits(
                    &f32s(golden, &format!("{a}.logits")),
                    &f32s(golden, &format!("{b}.logits")),
                    v,
                )
                .unwrap();
                println!("its {b} pattern against its {a} pattern: {m}");
            }
        };
        pair("prefill", "decode");
        pair("prefill", "reshaped");
    }
    for (i, (a_name, a)) in goldens.iter().enumerate() {
        for (b_name, b) in &goldens[i + 1..] {
            let m =
                compare_logits(&f32s(a, "prefill.logits"), &f32s(b, "prefill.logits"), v).unwrap();
            println!("{b_name} against {a_name}, one prefill chunk: {m}");
        }
    }
    if goldens[0].0 != "reference-pinned" {
        println!("no reference-pinned golden present: nothing was asserted beyond finite values");
    }
    assert!(
        failures.is_empty(),
        "the pinned reference must be equalled exactly: {failures:#?}"
    );
}
