//! The Qwen CPU executor on a released checkpoint, through the real engine (ignored:
//! needs the fetched checkpoint and its pinned goldens, see the model crate's
//! `golden/qwen/README.md`). Written for Qwen3.5-2B and -0.8B, whose whole models a CPU
//! runs in seconds.
//!
//!     EIDOLA_QWEN_DIR=<dir> cargo test --release -p eidola-engine-cpu \
//!         --test qwen_real -- --ignored --nocapture
//!
//! Every checkpoint file is checked against its recorded upstream hashes (the model
//! crate's `golden/qwen/upstream-*.json`) before anything loads. The prompts are slices
//! of the goldens' passage (`golden/reference-pinned.safetensors`, tokenised by the
//! checkpoint's own tokenizer). Requests run greedy, with drafts from the model's MTP
//! layer, in prefill chunks that cut the prompts mid-block, beside one another and again
//! from the prefix cache; every row's logits, every drafter and chain row, every draw and
//! every stored recurrent state are checked against the model crate's dense forward bit
//! for bit (`qwen_common::Harness::check_all`), and each output against dense greedy
//! generation.

mod qwen_common;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::Bucket;
use eidola_engine_cpu::qwen::QwenCpuExecutorConfig;
use eidola_engine_model::WeightSet;
use eidola_engine_model::qwen::{QwenModel, QwenWeights, config_of};
use qwen_common::*;

/// Every recorded upstream list.
const RECORDED: [&str; 3] = [
    include_str!("../../eidola-engine-model/golden/qwen/upstream-sha256.json"),
    include_str!("../../eidola-engine-model/golden/qwen/upstream-qwen3.5-2b-sha256.json"),
    include_str!("../../eidola-engine-model/golden/qwen/upstream-qwen3.5-0.8b-sha256.json"),
];

/// The released tokenizer's ids (Qwen3.5's): every one sampleable.
const SAMPLEABLE_VOCAB: u32 = 248_070;

/// The checkpoint at `EIDOLA_QWEN_DIR`, every file checked against the recorded list
/// whose `config.json` it holds, and the goldens' passage.
fn load() -> (Arc<QwenModel>, Vec<u32>) {
    let dir = PathBuf::from(std::env::var("EIDOLA_QWEN_DIR").expect("set EIDOLA_QWEN_DIR"));
    let store = WeightSet::open_dir(&dir).unwrap();
    let manifest: BTreeMap<String, String> = store.sha256_manifest();
    let recorded = RECORDED
        .iter()
        .map(|text| serde_json::from_str::<serde_json::Value>(text).unwrap())
        .find(|r| r["files"]["config.json"]["sha256"] == manifest["config.json"].as_str())
        .expect("config.json is a recorded checkpoint's");
    for (name, digest) in &manifest {
        assert_eq!(
            recorded["files"][name]["sha256"].as_str(),
            Some(digest.as_str()),
            "{name} is not the recorded file of {}",
            recorded["repository"]
        );
    }
    println!(
        "{} files of {} match their recorded hashes",
        manifest.len(),
        recorded["repository"]
    );
    let golden = WeightSet::open_files(&[dir.join("golden/reference-pinned.safetensors")]).unwrap();
    let tokens = golden
        .get("tokens")
        .unwrap()
        .to_i64("tokens")
        .unwrap()
        .into_iter()
        .map(|t| t as u32)
        .collect();
    let store = Arc::new(store);
    let config = config_of(&store).unwrap();
    let model = QwenModel::new(QwenWeights::load(store, config).unwrap());
    (Arc::new(model), tokens)
}

fn config(draft_tokens: u32) -> QwenCpuExecutorConfig {
    QwenCpuExecutorConfig {
        block_size: 16,
        num_blocks: 256,
        host_blocks: 0,
        state_buffers: 32,
        host_state_buffers: 0,
        num_state_slots: 4,
        max_model_len: 1024,
        buckets: vec![Bucket {
            max_seqs: 4,
            max_tokens: 64,
        }],
        draft_tokens,
        sampleable_vocab_size: SAMPLEABLE_VOCAB,
        record: true,
    }
}

#[test]
#[ignore = "needs a fetched checkpoint and its goldens"]
fn the_executor_on_a_released_checkpoint_equals_the_dense_forward() {
    let began = Instant::now();
    let (model, passage) = load();
    println!("[{:.0}s] loaded", began.elapsed().as_secs_f64());
    let g = SamplingParams::greedy();
    for drafts in [0, 2] {
        let mut sched = chunked(13, drafts > 0);
        sched.max_batched_tokens = 64;
        sched.max_seqs = 4;
        let mut h = Harness::new(model.clone(), config(drafts), sched);
        // Two prompts beside each other, cut by the chunk mid-block.
        h.submit(request(1, passage[..37].to_vec(), g, 20, keyed(1)));
        h.submit(request(2, passage[100..141].to_vec(), g, 10, keyed(2)));
        h.run();
        h.flush();
        // The first again, from the prefix cache (its regeneration point), and a longer
        // prompt extending it, from its continuation.
        h.run_one(request(3, passage[..37].to_vec(), g, 20, keyed(1)));
        assert_eq!(h.cached[&3], 32);
        assert_eq!(h.outputs[&3], h.outputs[&1]);
        let mut longer = passage[..37].to_vec();
        longer.extend(&h.outputs[&1]);
        longer.extend(&passage[200..205]);
        h.run_one(request(4, longer, g, 6, keyed(1)));
        // Positions 0..=55 hold KV: the continuation point is the boundary at 48.
        assert_eq!(h.cached[&4], 48, "the continuation is a hit");
        let t0 = Instant::now();
        let sum = h.check_all(drafts > 0);
        println!(
            "[{:.0}s] drafts {drafts}: {} rows, {} stored states, {} chain rows checked in {:.0} s; \
             {} of {} drafts accepted; outputs {:?} {:?}",
            began.elapsed().as_secs_f64(),
            sum.records,
            sum.states,
            sum.chain_rows,
            t0.elapsed().as_secs_f64(),
            sum.accepted,
            sum.drafted,
            h.outputs[&1],
            h.outputs[&2],
        );
        assert!(sum.records > 0 && sum.states > 0);
        if drafts > 0 {
            assert!(sum.chain_rows > 0, "nothing drafted");
        }
    }
}
