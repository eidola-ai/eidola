//! The Qwen reference forward on real weights (ignored: needs a fetched
//! checkpoint, Qwen3.8-27B-FP8 or Qwen3.5-2B or -0.8B, and its goldens, see
//! `golden/qwen/README.md`).
//!
//!     EIDOLA_QWEN_DIR=<dir> [EIDOLA_QWEN_LAYERS=4] cargo test --release \
//!         -p eidola-engine-model --test qwen_real -- --ignored --nocapture
//!
//! The directory holds the checkpoint's files (all of them, or the truncation
//! `fetch_checkpoint.py --truncation prefix` fetches) and `golden/`, written by
//! `real_golden.py` with the same layer count (`EIDOLA_QWEN_LAYERS`, default
//! 4; 24 is the whole of a Qwen3.5 checkpoint). Every file is checked against
//! the recorded upstream hashes before anything loads. Against
//! `golden/reference-pinned.safetensors` every comparison is exact; every
//! other `golden/reference-*.safetensors` is reported, not asserted.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

mod qwen_upstream;

use eidola_engine_model::WeightSet;
use eidola_engine_model::compare::compare_logits;
use eidola_engine_model::qwen::{
    KvCache, LayerState, LogitRows, QwenForwardOptions, QwenModel, QwenWeights, config_of,
};
use eidola_engine_model::tensor::Matrix;

fn f32s(set: &WeightSet, name: &str) -> Vec<f32> {
    set.get(name).unwrap().to_f32(name).unwrap()
}

fn differing(got: &[f32], want: &[f32]) -> usize {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

/// `‖a − b‖ / ‖b‖` in f64.
fn relative_error(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        num += (*x as f64 - *y as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
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
                WeightSet::open_files(std::slice::from_ref(p)).unwrap(),
            )
        })
        .collect()
}

fn rss_gb() -> f64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<f64>()
                .unwrap_or(0.0)
        })
        .unwrap_or(0.0);
    out / 1024.0 / 1024.0
}

#[test]
#[ignore = "needs the fetched checkpoint and goldens"]
fn real_checkpoint_matches_the_reference() {
    let dir = PathBuf::from(std::env::var("EIDOLA_QWEN_DIR").expect("set EIDOLA_QWEN_DIR"));
    let layers: usize = std::env::var("EIDOLA_QWEN_LAYERS")
        .map(|v| v.parse().expect("EIDOLA_QWEN_LAYERS is a number"))
        .unwrap_or(4);
    let began = Instant::now();
    let store = WeightSet::open_dir(&dir).unwrap();
    qwen_upstream::verify(&store, &dir);
    println!("[{:.0}s] verified", began.elapsed().as_secs_f64());
    let store = Arc::new(store);
    let keep: Vec<usize> = (0..layers).collect();
    let config = config_of(&store).unwrap().truncated(&keep).unwrap();
    let model = QwenModel::new(QwenWeights::load(store, config).unwrap());
    let c = model.config().clone();
    let (v, h) = (c.vocab_size, c.hidden_size);

    let goldens = goldens(&dir);
    let tokens: Vec<u32> = goldens[0]
        .1
        .get("tokens")
        .unwrap()
        .to_i64("tokens")
        .unwrap()
        .into_iter()
        .map(|t| t as u32)
        .collect();
    let n = tokens.len();
    println!("{n} tokens, {} layers", c.layers.len());

    // The passage as one call.
    let mut state = model.new_state();
    let t0 = Instant::now();
    let out = model
        .forward(
            &mut state,
            &tokens,
            &QwenForwardOptions {
                logits: LogitRows::All,
                keep_streams: true,
            },
        )
        .unwrap();
    println!(
        "[{:.0}s] one call: {:.1} s, RSS {:.1} GB",
        began.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64(),
        rss_gb()
    );
    let acc = qwen_upstream::next_token_accuracy(&out.logits.data, &tokens, v);
    println!("next-token accuracy over the passage: {acc:.3}");

    // One token per call from a call of the first tokens, as the goldens do.
    let step_len = goldens[0].1.get("step.logits").unwrap().shape[0];
    let step_first = if goldens[0].1.contains("step.first") {
        goldens[0]
            .1
            .get("step.first")
            .unwrap()
            .to_i64("step.first")
            .unwrap()[0] as usize
    } else {
        64
    };
    let mut s2 = model.new_state();
    let t0 = Instant::now();
    let mut step = model
        .forward(
            &mut s2,
            &tokens[..step_first],
            &QwenForwardOptions {
                logits: LogitRows::All,
                keep_streams: false,
            },
        )
        .unwrap()
        .logits
        .data;
    for t in &tokens[step_first..step_len] {
        let o = model
            .forward(
                &mut s2,
                &[*t],
                &QwenForwardOptions {
                    logits: LogitRows::All,
                    keep_streams: false,
                },
            )
            .unwrap();
        step.extend(o.logits.data);
    }
    println!(
        "[{:.0}s] {} tokens one per call: {:.1} s",
        began.elapsed().as_secs_f64(),
        step_len - step_first,
        t0.elapsed().as_secs_f64()
    );
    assert_eq!(
        differing(&step, &out.logits.data[..step_len * v]),
        0,
        "one token per call differs from one call"
    );

    // The drafter at every position but the last.
    let mtp = (c.mtp_layers > 0).then(|| {
        let hidden = Matrix::from_vec(n - 1, h, out.normed.data[..(n - 1) * h].to_vec());
        model
            .mtp_forward(
                0,
                &mut KvCache::default(),
                0,
                &tokens[1..],
                &hidden,
                &LogitRows::All,
            )
            .unwrap()
    });
    println!("[{:.0}s] drafter done", began.elapsed().as_secs_f64());

    for (name, g) in &goldens {
        let exact = name == "reference-pinned";
        println!("--- against {name}");
        let mut total = 0usize;
        for (i, s) in out.streams.iter().enumerate() {
            let want = f32s(g, &format!("prefill.stream.{i}"));
            let d = differing(&s.data, &want);
            total += d;
            println!(
                "layer {i}: {d} of {} stream elements differ, relative error {:.2e}",
                want.len(),
                relative_error(&s.data, &want)
            );
        }
        for (i, l) in state.layers.iter().enumerate() {
            if let LayerState::DeltaNet(s) = l {
                let conv = differing(&s.conv, &f32s(g, &format!("state.{i}.conv")));
                let want = f32s(g, &format!("state.{i}.recurrent"));
                let rec = differing(&s.recurrent, &want);
                total += conv + rec;
                println!(
                    "layer {i} state: conv {conv} differ, recurrent {rec} of {} differ (relative error {:.2e})",
                    want.len(),
                    relative_error(&s.recurrent, &want)
                );
            }
        }
        let want = f32s(g, "prefill.logits");
        let d = differing(&out.logits.data, &want);
        total += d;
        println!(
            "logits: {d} of {} differ; {}",
            want.len(),
            compare_logits(&want, &out.logits.data, v).unwrap()
        );
        // Goldens may run fewer one-token calls; compare what both ran.
        let want = f32s(g, "step.logits");
        let k = want.len().min(step.len());
        let d = differing(&step[..k], &want[..k]);
        total += d;
        println!(
            "one per call: {d} differ; {}",
            compare_logits(&want[..k], &step[..k], v).unwrap()
        );
        if let Some(m) = &mtp {
            let want = f32s(g, "mtp.logits");
            let d = differing(&m.logits.data, &want)
                + differing(&m.hidden.data, &f32s(g, "mtp.hidden"));
            total += d;
            println!(
                "drafter: {d} differ; {}",
                compare_logits(&want, &m.logits.data, v).unwrap()
            );
        }
        if exact {
            assert_eq!(total, 0, "the forward differs from the pinned reference");
        }
    }
    // A reference's own one-token calls against its own one call: for the
    // native run that is its recurrent decode against its chunked prefill.
    for (name, g) in &goldens {
        let one = f32s(g, "prefill.logits");
        let step = f32s(g, "step.logits");
        let first = if g.contains("step.first") {
            g.get("step.first").unwrap().to_i64("step.first").unwrap()[0] as usize
        } else {
            64
        };
        println!(
            "{name}: its {} one-token calls against its one call: {}",
            step.len() / v - first,
            compare_logits(&one[first * v..step.len()], &step[first * v..], v).unwrap()
        );
    }
    for (i, (a, ga)) in goldens.iter().enumerate() {
        for (b, gb) in &goldens[i + 1..] {
            println!(
                "{a} against {b}: {}",
                compare_logits(&f32s(ga, "prefill.logits"), &f32s(gb, "prefill.logits"), v)
                    .unwrap()
            );
        }
    }
    println!(
        "[{:.0}s] done, RSS {:.1} GB",
        began.elapsed().as_secs_f64(),
        rss_gb()
    );
}
