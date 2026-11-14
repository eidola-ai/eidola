//! Local check against a truncation of the real DeepSeek-V4.1-Flash-1017
//! checkpoint (ignored: needs the fetched weights and goldens, see
//! `golden/deepseek_v41/README.md`).
//!
//!     EIDOLA_DEEPSEEK_TRUNCATED_DIR=<dir> cargo test --release -p eidola-engine-model \
//!         --test deepseek_real -- --ignored --nocapture

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use eidola_engine_model::WeightSet;
use eidola_engine_model::compare::compare_logits;
use eidola_engine_model::deepseek::engram::{EngramTable, SparseEngramTable};
use eidola_engine_model::deepseek::forward::DsForwardOptions;
use eidola_engine_model::deepseek::{DsLoadOptions, DsModel, DsWeights, config_of};

/// The truncation `golden/deepseek_v41/ds_golden/truncated.py` fetches.
const LAYERS: [usize; 12] = [0, 1, 2, 3, 14, 15, 20, 21, 24, 37, 38, 39];
const BACKBONE_EXPERTS: usize = 32;
const DRAFTER_EXPERTS: usize = 16;

fn open(path: &Path) -> WeightSet {
    WeightSet::open_files(&[path.to_path_buf()]).unwrap()
}

fn f32s(set: &WeightSet, name: &str) -> Vec<f32> {
    set.get(name).unwrap().to_f32(name).unwrap()
}

fn exact(got: &[f32], want: &[f32]) -> usize {
    got.iter().zip(want).filter(|(a, b)| a != b).count()
}

/// Every `golden/reference-*.safetensors`, by name, `reference-pinned` first.
fn goldens(dir: &Path) -> Vec<(String, WeightSet)> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir.join("golden"))
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
    paths
        .iter()
        .map(|p| {
            (
                p.file_stem().unwrap().to_string_lossy().into_owned(),
                open(p),
            )
        })
        .collect()
}

#[test]
#[ignore = "needs a fetched truncation of the real checkpoint"]
fn truncated_real_checkpoint_matches_the_reference() {
    let dir = PathBuf::from(
        std::env::var("EIDOLA_DEEPSEEK_TRUNCATED_DIR")
            .expect("EIDOLA_DEEPSEEK_TRUNCATED_DIR names the fetched checkpoint"),
    );
    let store = Arc::new(WeightSet::open_dir(&dir).unwrap());
    let manifest: BTreeMap<String, String> =
        serde_json::from_slice(&std::fs::read(dir.join("local-sha256.json")).unwrap()).unwrap();
    store
        .verify_sha256(&manifest)
        .expect("fetched files match their manifest");

    // Every golden holds the tokens and the Engram map it was made with.
    let goldens = goldens(&dir);
    let pinned = &goldens[0].1;
    let tokens: Vec<u32> = pinned
        .get("tokens")
        .unwrap()
        .to_i64("tokens")
        .unwrap()
        .iter()
        .map(|t| *t as u32)
        .collect();
    let token_map = pinned
        .get("engram.token_map")
        .unwrap()
        .to_i64("map")
        .unwrap()
        .iter()
        .map(|t| *t as u32)
        .collect();

    let config = config_of(&store)
        .unwrap()
        .truncated(&LAYERS)
        .unwrap()
        .with_routed_experts(BACKBONE_EXPERTS, DRAFTER_EXPERTS)
        .unwrap();
    let rows = open(&dir.join("golden/engram-rows.safetensors"));
    let spec = config.engram.clone().unwrap();
    let engram_tables = spec
        .layers
        .iter()
        .filter(|l| LAYERS.contains(&l.source_layer))
        .map(|l| {
            let p = format!("layers.{}", l.source_layer);
            let ids: Vec<u64> = rows
                .get(&format!("{p}.rows"))
                .unwrap()
                .to_i64("rows")
                .unwrap()
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
            (l.source_layer, Arc::new(table) as Arc<dyn EngramTable>)
        })
        .collect();
    let weights = DsWeights::load(
        store,
        config,
        &DsLoadOptions {
            skip_drafter: false,
            engram_tables,
        },
    )
    .unwrap();
    let model = DsModel::new(weights, Some(token_map)).unwrap();
    let c = model.config();
    let (n, v) = (tokens.len(), c.vocab_size);

    // `layout(position)` is the reference prefill chunk length whose attention
    // layout a position is computed with (`None`: the canonical layout).
    let run = |layout: &dyn Fn(usize) -> Option<usize>| {
        let mut state = model.new_state();
        let (mut logits, mut hidden) = (Vec::with_capacity(n * v), Vec::new());
        let mut taps = vec![Vec::new(); c.layers.len()];
        let window = c.attention.window;
        let mut at = 0;
        while at < n {
            // A chunk never straddles the end of a window-long reference prefill.
            let limit = if at < window { window.min(n) } else { n };
            let end = (at + 48).min(limit);
            let opts = DsForwardOptions {
                taps: true,
                reference_prefill: layout(at),
                ..Default::default()
            };
            let out = model.forward(&mut state, &tokens[at..end], &opts).unwrap();
            logits.extend(out.logits.data);
            hidden.extend(out.main_hidden.unwrap().data);
            for (l, tap) in out.taps.into_iter().enumerate() {
                taps[l].extend(tap.data);
            }
            at = end;
        }
        (logits, hidden, taps)
    };
    let window = c.attention.window;
    let (canonical, ..) = run(&|_| None);
    // The reference's one prefill chunk, in its own layout.
    let (logits, _, taps) = run(&|_| Some(n));
    // A one-window prefill chunk, then one token per call.
    let (stepped, hidden, _) = run(&|at| (at < window).then_some(window));

    let mut failures: Vec<String> = Vec::new();
    println!("{n} tokens, {} layers", c.layers.len());
    let relative = |got: &[f32], want: &[f32]| {
        let (mut err, mut norm) = (0.0f64, 0.0f64);
        for (a, b) in got.iter().zip(want) {
            err += ((a - b) as f64).powi(2);
            norm += (*b as f64).powi(2);
        }
        (err / norm.max(f64::MIN_POSITIVE)).sqrt()
    };
    let width = hidden.len() / n;
    let slots = c.drafter.as_ref().unwrap().slots;
    for (name, golden) in &goldens {
        // Only the reference with emulated kernels and pinned arithmetic is a
        // function this forward equals; every other golden is a measurement.
        let exact_expected = name == "reference-pinned";
        let mut check = |what: &str, a: &eidola_engine_model::compare::LogitAgreement| {
            if exact_expected && (a.top1_rate() < 0.999 || a.kl_mean >= 1e-3) {
                failures.push(format!("{name} {what}: {a}"));
            }
        };
        println!(
            "== {name}{}",
            if exact_expected {
                ""
            } else {
                " (measurement only)"
            }
        );
        let want = f32s(golden, "prefill.logits");
        let a = compare_logits(&want, &logits, v).unwrap();
        println!(
            "one prefill chunk, its layout reproduced: {a}; {} of {} logits differ",
            exact(&logits, &want),
            want.len()
        );
        check("prefill", &a);
        for (l, tap) in taps.iter().enumerate() {
            let want = f32s(golden, &format!("prefill.layer.{l}"));
            println!(
                "  streams after layer {l:>2} (checkpoint layer {:>2}): relative error {:.3e}, {} of {} differ",
                c.layers[l].source_index,
                relative(tap, &want),
                exact(tap, &want),
                want.len()
            );
        }
        let m = compare_logits(&want, &canonical, v).unwrap();
        println!("canonical layout against that prefill chunk: {m}");
        if !golden.contains("step.logits") {
            continue;
        }
        let step = f32s(golden, "step.logits");
        let k = step.len() / v;
        let s = compare_logits(&step, &stepped[..step.len()], v).unwrap();
        println!(
            "a window-long prefill then one token per call: {s}; {} of {} logits differ",
            exact(&stepped[..step.len()], &step),
            step.len()
        );
        check("steps", &s);

        // The drafter, from this forward's own hidden states. The reference
        // seeds it over its prefill chunk (one window) and drafts from every
        // position after it.
        let want_tokens = golden
            .get("step.draft_tokens")
            .unwrap()
            .to_i64("t")
            .unwrap();
        let want_conf = f32s(golden, "step.draft_confidence");
        let want_logits = f32s(golden, "step.draft_logits");
        let first = window.min(k);
        let mut drafter = model.new_drafter_state();
        for p in 0..first {
            model
                .drafter_observe(&mut drafter, &hidden[p * width..(p + 1) * width])
                .unwrap();
        }
        let (mut same_tokens, mut conf_diff, mut draft_logits) = (0, 0.0f32, Vec::new());
        for p in first..k {
            let given = tokens[(p + 1) % k];
            let core = model
                .drafter_forward(&mut drafter, &hidden[p * width..(p + 1) * width], given)
                .unwrap();
            let draft = model.draft_greedy(&core, given);
            let at = (p - first) * slots;
            same_tokens += draft
                .tokens
                .iter()
                .zip(&want_tokens[at..])
                .filter(|(a, b)| **a as i64 == **b)
                .count();
            for (a, b) in draft.confidence.iter().zip(&want_conf[at..]) {
                conf_diff = conf_diff.max((a - b).abs());
            }
            if at * v < want_logits.len() {
                draft_logits.extend(draft.logits.data);
            }
        }
        let d = compare_logits(&want_logits, &draft_logits, v).unwrap();
        println!(
            "drafter: {same_tokens} of {} draft tokens equal, max |Δconfidence| {conf_diff:e}; logits {d}",
            (k - first) * slots
        );
        check("drafter", &d);
    }

    // The reference against itself: every pair of goldens other than the
    // pinned one (which this forward equals, so its rows are the ones above).
    for (i, (a_name, a)) in goldens.iter().enumerate() {
        for (b_name, b) in &goldens[i + 1..] {
            if a_name == "reference-pinned" {
                continue;
            }
            let (x, y) = (f32s(a, "prefill.logits"), f32s(b, "prefill.logits"));
            let m = compare_logits(&x, &y, v).unwrap();
            let last = c.layers.len() - 1;
            let key = format!("prefill.layer.{last}");
            println!(
                "{b_name} against {a_name}, one prefill chunk: {m}; streams after the last layer: relative error {:.3e}",
                relative(&f32s(b, &key), &f32s(a, &key))
            );
        }
    }
    if goldens[0].0 != "reference-pinned" {
        println!("no reference-pinned golden present: nothing was asserted beyond finite values");
    }
    assert!(
        failures.is_empty(),
        "acceptance (top-1 >= 99.9 %, mean KL < 1e-3) missed: {failures:#?}"
    );
}
