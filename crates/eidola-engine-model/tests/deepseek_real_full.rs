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
//! exactly; the others are reported.
//!
//! Agreement with a reference is not health: both read the same files, and on
//! damaged files they agree about a model that is not the released one. So,
//! whatever goldens are present, the test also asserts against the world:
//!
//! - every checkpoint file is the recorded upstream one
//!   (`golden/deepseek_v41/upstream-sha256.json`), before anything loads;
//! - next-token accuracy over the goldens' passage reaches its floor, for our
//!   forward and for every golden;
//! - with `golden/health.safetensors` (`health_check.py`), our forward
//!   answers the `hello` request with the publicly reported reply, token for
//!   token (at the one position reported as a near three-way tie: with one of
//!   the three), and reaches the floor over the story as a chat reply.

use std::path::{Path, PathBuf};
use std::sync::Arc;

mod deepseek_upstream;

use deepseek_upstream::{PASSAGE_FLOOR, STORY_FLOOR, argmax_rows, next_token_accuracy};
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
    // A partial checkpoint here is the fetched first-four-layers truncation.
    deepseek_upstream::verify(&store, &dir, first_layers.map(|_| "prefix"));
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
    // The `reshaped` pattern is a prefill chunk of `first` tokens, then one
    // token per call. When the chunk is at least a window long it has the
    // layout of the whole-sequence prefill and the single tokens after it have
    // full windows, where the two layouts coincide: the prefill run stands in.
    // A shorter chunk has its own layout (`min(chunk, window)` window slots),
    // which only a run of that chunk followed by canonical calls reproduces.
    let reshaped_first = goldens
        .iter()
        .find(|(_, g)| g.contains("reshaped.logits"))
        .map(|(_, g)| match g.contains("reshaped.first") {
            true => ints(g, "reshaped.first")[0] as usize,
            false => n.saturating_sub(64).max(1),
        });
    let reshaped_logits = match reshaped_first {
        Some(first) if first < c.attention.window => {
            println!(
                "  (the reshaped pattern's {first}-token chunk is shorter than the window: running its layout)"
            );
            let began = std::time::Instant::now();
            let mut state = model.new_state();
            let chunk = DsForwardOptions {
                reference_prefill: Some(first),
                ..Default::default()
            };
            let mut logits = model
                .forward(&mut state, &tokens[..first], &chunk)
                .unwrap()
                .logits
                .data;
            logits.extend(
                model
                    .forward(&mut state, &tokens[first..], &Default::default())
                    .unwrap()
                    .logits
                    .data,
            );
            println!("  (a forward took {:.0} s)", began.elapsed().as_secs_f64());
            Some(logits)
        }
        _ => None,
    };
    let reshaped_logits = reshaped_logits.as_ref().unwrap_or(&prefill.logits.data);
    let ours = compare_logits(&prefill.logits.data, &canonical.logits.data, v).unwrap();
    println!("our canonical layout against our reproduction of the prefill layout: {ours}");

    // Health, which agreement cannot show. Only the whole model is held to
    // the floors: its first layers alone are not a language model.
    let whole = first_layers.is_none();
    let mut unhealthy = Vec::new();
    let health = |unhealthy: &mut Vec<String>, what: String, accuracy: f64, floor: f64| {
        let held = if whole {
            ""
        } else {
            ", not asserted of a partial model"
        };
        println!("next-token accuracy of {what}: {accuracy:.3} (floor {floor}{held})");
        if whole && accuracy < floor {
            unhealthy.push(format!(
                "{what}: next-token accuracy {accuracy:.3} is under {floor}"
            ));
        }
    };
    for (what, logits) in [
        ("our canonical layout", &canonical.logits.data),
        ("our prefill layout", &prefill.logits.data),
    ] {
        health(
            &mut unhealthy,
            format!("{what} over the passage"),
            next_token_accuracy(logits, &tokens, v, 0),
            PASSAGE_FLOOR,
        );
    }
    for (name, golden) in &goldens {
        for pattern in ["prefill", "decode", "reshaped"] {
            if golden.contains(&format!("{pattern}.logits")) {
                let logits = f32s(golden, &format!("{pattern}.logits"));
                health(
                    &mut unhealthy,
                    format!("{name}'s {pattern} pattern over the passage"),
                    next_token_accuracy(&logits, &tokens, v, 0),
                    PASSAGE_FLOOR,
                );
            }
        }
    }
    let health_path = golden_dir.join("health.safetensors");
    if whole && health_path.exists() {
        let set = open(&health_path);
        for name in ["hello", "story"] {
            if !set.contains(&format!("{name}.tokens")) {
                continue;
            }
            let sequence: Vec<u32> = ints(&set, &format!("{name}.tokens"))
                .iter()
                .map(|t| *t as u32)
                .collect();
            let start = ints(&set, &format!("{name}.start"))[0] as usize;
            let logits = model
                .forward(&mut model.new_state(), &sequence, &Default::default())
                .unwrap()
                .logits
                .data;
            let mine = argmax_rows(&logits, v);
            let theirs = ints(&set, &format!("{name}.argmax"));
            let same = mine
                .iter()
                .zip(&theirs)
                .filter(|(a, b)| **a as i64 == **b)
                .count();
            println!(
                "{name}: our most likely token equals the reference's at {same} of {} positions",
                mine.len()
            );
            if name == "hello" {
                // The publicly reported reply: every token of it must be our
                // most likely one, except at the position the report gives as
                // a near three-way tie, where ours must be one of the three.
                let fork = ints(&set, "hello.fork")[0] as usize;
                let tied = ints(&set, "hello.fork_tokens");
                let wrong = (start - 1..sequence.len() - 1)
                    .filter(|&i| match i == fork {
                        true => !tied.contains(&(mine[i] as i64)),
                        false => mine[i] != sequence[i + 1],
                    })
                    .count();
                println!(
                    "hello: {wrong} positions of the reported reply are not answered as reported; at the tie our most likely token is {} (reported: {tied:?}, the reply's first)",
                    mine[fork]
                );
                if wrong != 0 {
                    unhealthy.push(format!(
                        "hello: {wrong} positions of the reported reply are not answered as reported"
                    ));
                }
            } else {
                health(
                    &mut unhealthy,
                    format!("our canonical layout over the {name} reply"),
                    next_token_accuracy(&logits, &sequence, v, start - 1),
                    STORY_FLOOR,
                );
            }
        }
    } else if whole {
        println!(
            "no golden/health.safetensors (health_check.py): the reported reply and the chat floor were not checked"
        );
    }

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
        for (pattern, mine, taps) in [
            ("prefill", &prefill.logits.data, Some(&prefill.taps)),
            ("reshaped", reshaped_logits, None),
            ("decode", &canonical.logits.data, Some(&canonical.taps)),
        ] {
            if !golden.contains(&format!("{pattern}.logits")) {
                continue;
            }
            let want = f32s(golden, &format!("{pattern}.logits"));
            let a = compare_logits(&want, mine, v).unwrap();
            let differing = mine.iter().zip(&want).filter(|(a, b)| a != b).count();
            println!(
                "ours against its {pattern} pattern: {a}; {differing} of {} logits differ",
                want.len()
            );
            if exact_expected && differing != 0 {
                failures.push(format!("{name} {pattern}: {differing} logits differ"));
            }
            let Some(taps) = taps else { continue };
            for l in (0..layers).filter(|l| golden.contains(&format!("{pattern}.layer.{l}"))) {
                let want = f32s(golden, &format!("{pattern}.layer.{l}"));
                println!(
                    "  streams after layer {l:>2}: relative error {:.3e}",
                    relative(&taps[l].data, &want)
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
        // Only the shape of the calls differs between these two when the
        // reshaped chunk fills the window; a shorter chunk also has another
        // layout, and the line says which it is.
        if reshaped_first.is_some_and(|first| first < c.attention.window) {
            println!(
                "(next line: the reshaped chunk is shorter than the window, so its layout differs too)"
            );
        }
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
        println!(
            "no reference-pinned golden present: exact agreement was not asserted, only the recorded hashes and health"
        );
    }
    assert!(
        unhealthy.is_empty(),
        "this is not the released model, whatever agrees with it: {unhealthy:#?}"
    );
    assert!(
        failures.is_empty(),
        "the pinned reference must be equalled exactly: {failures:#?}"
    );
}

/// The scoring the health floors rest on: position `i` is right when row
/// `i`'s most likely token is token `i + 1`, counted from `from`, and the
/// last row predicts nothing that is scored.
#[test]
fn next_token_accuracy_scores_each_row_against_the_token_after_it() {
    let tokens = [7u32, 2, 0, 1, 2];
    // Rows predict 2, 0, 2, 2 and (unscored) 1 over a vocabulary of 3.
    #[rustfmt::skip]
    let logits = [
        0.0, 0.1, 0.9,
        0.5, 0.1, 0.2,
        0.0, 0.1, 0.9,
        0.0, 0.1, 0.9,
        0.0, 0.9, 0.1f32,
    ];
    assert_eq!(argmax_rows(&logits, 3), [2, 0, 2, 2, 1]);
    assert_eq!(next_token_accuracy(&logits, &tokens, 3, 0), 0.75);
    assert_eq!(next_token_accuracy(&logits, &tokens, 3, 2), 0.5);
    // Equal logits: the lowest index.
    assert_eq!(argmax_rows(&[0.5, 0.5, 0.1], 3), [0]);
}
