//! Health of the Qwen reference forward on a whole released checkpoint
//! (ignored: needs the checkpoint, Qwen3.8-27B-FP8's 31 GB or a Qwen3.5
//! one, and the token sequences `golden/qwen/health_check.py` writes, see
//! `golden/qwen/README.md`).
//!
//!     EIDOLA_QWEN_DIR=<dir> cargo test --release -p eidola-engine-model \
//!         --test qwen_health -- --ignored --nocapture
//!
//! Agreement with a reference says nothing about the files both read. This
//! holds our forward, on files checked against the recorded upstream hashes,
//! to facts outside the tooling: next-token accuracy over the checkpoint's own
//! `LICENSE` of at least 0.95, and "Paris" as the greedy reply to "What is the
//! capital of France? Answer with one word." (thinking off). With
//! `EIDOLA_QWEN_LAYERS` below the checkpoint's layer count it runs a
//! truncation and asserts nothing about health.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

mod qwen_upstream;

use eidola_engine_model::WeightSet;
use eidola_engine_model::qwen::{LogitRows, QwenForwardOptions, QwenModel, QwenWeights, config_of};

fn ids(set: &WeightSet, name: &str) -> Vec<u32> {
    set.get(name)
        .unwrap()
        .to_i64(name)
        .unwrap()
        .into_iter()
        .map(|t| t as u32)
        .collect()
}

#[test]
#[ignore = "needs the whole checkpoint"]
fn the_released_model_is_healthy() {
    let dir = PathBuf::from(std::env::var("EIDOLA_QWEN_DIR").expect("set EIDOLA_QWEN_DIR"));
    let began = Instant::now();
    let store = WeightSet::open_dir(&dir).unwrap();
    qwen_upstream::verify(&store, &dir);
    let health = WeightSet::open_files(&[dir.join("golden/health.safetensors")]).unwrap();
    let store = Arc::new(store);
    let layers: usize = std::env::var("EIDOLA_QWEN_LAYERS")
        .map(|v| v.parse().expect("EIDOLA_QWEN_LAYERS is a number"))
        .unwrap_or(config_of(&store).unwrap().checkpoint_layers);
    let keep: Vec<usize> = (0..layers).collect();
    let config = config_of(&store).unwrap().truncated(&keep).unwrap();
    let model = QwenModel::new(QwenWeights::load(store, config).unwrap());
    let v = model.config().vocab_size;
    let whole = layers == model.config().checkpoint_layers;
    // 0: the reference that wrote the health file ran pinned, which our
    // forward equals bit for bit.
    let pinned_reference =
        health.contains("reference.mode") && ids(&health, "reference.mode") == [0];
    println!("[{:.0}s] {layers} layers", began.elapsed().as_secs_f64());

    let all = QwenForwardOptions {
        logits: LogitRows::All,
        keep_streams: false,
    };
    for (name, floor) in [
        ("license", Some(qwen_upstream::LICENSE_FLOOR)),
        ("passage", None),
    ] {
        let tokens = ids(&health, &format!("{name}.tokens"));
        let t0 = Instant::now();
        let out = model
            .forward(&mut model.new_state(), &tokens, &all)
            .unwrap();
        let acc = qwen_upstream::next_token_accuracy(&out.logits.data, &tokens, v);
        println!(
            "[{:.0}s] {name}: {} tokens in {:.0} s, next-token accuracy {acc:.3}",
            began.elapsed().as_secs_f64(),
            tokens.len(),
            t0.elapsed().as_secs_f64()
        );
        let reference = format!("{name}.reference_top1");
        if health.contains(&reference) {
            let ours = qwen_upstream::argmax_rows(&out.logits.data, v);
            let theirs = ids(&health, &reference);
            let same = ours.iter().zip(&theirs).filter(|(a, b)| a == b).count();
            if pinned_reference && whole {
                assert_eq!(
                    same,
                    theirs.len(),
                    "{name}: the pinned reference's choices differ"
                );
            }
            println!(
                "{name}: most likely token equal to the reference's at {same} of {} positions",
                theirs.len()
            );
        }
        if let (true, Some(floor)) = (whole, floor) {
            assert!(
                acc >= floor,
                "{name}: accuracy {acc:.3} under the floor {floor}"
            );
        }
    }

    let prompt = ids(&health, "capital.prompt");
    let stops = ids(&health, "capital.stop");
    let answer = ids(&health, "capital.answer");
    let mut state = model.new_state();
    let mut out = model
        .forward(
            &mut state,
            &prompt,
            &QwenForwardOptions {
                logits: LogitRows::Last,
                keep_streams: false,
            },
        )
        .unwrap();
    let mut reply = Vec::new();
    for _ in 0..16 {
        let t = qwen_upstream::argmax_rows(&out.logits.data, v)[0];
        reply.push(t);
        if stops.contains(&t) {
            break;
        }
        out = model
            .forward(
                &mut state,
                &[t],
                &QwenForwardOptions {
                    logits: LogitRows::Last,
                    keep_streams: false,
                },
            )
            .unwrap();
    }
    println!(
        "[{:.0}s] capital: reply {reply:?} (\"Paris\" is {answer:?})",
        began.elapsed().as_secs_f64()
    );
    if health.contains("capital.reference_reply") {
        let theirs = ids(&health, "capital.reference_reply");
        println!("capital: the reference replied {theirs:?}");
        if pinned_reference && whole {
            assert_eq!(reply, theirs, "the pinned reference replied otherwise");
        }
    }
    if whole {
        assert!(
            reply.starts_with(&answer),
            "the greedy reply {reply:?} does not begin with \"Paris\" {answer:?}"
        );
    }
}
