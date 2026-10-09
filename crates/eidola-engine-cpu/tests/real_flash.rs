//! The engine over the CPU executor on a truncated real MiMo-V2.6-Flash-MOPD checkpoint
//! (checkpoint layers 0, 1, 2 and 5 plus the three MTP layers; see the model crate's
//! `golden/README.md` for how it is fetched). Chunked prefill, speculative decoding with
//! all three MTP depths, and a prefix-cache hit; every computed logit row (target and
//! drafter) is compared bit for bit with the dense reference forward on the same
//! truncation. Local only, slow, and needs tens of GB of RAM:
//!
//! ```text
//! cargo test --release -p eidola-engine-cpu --test real_flash -- --ignored --nocapture
//! ```
//!
//! `EIDOLA_MIMO_TRUNCATED_DIR` overrides the checkpoint directory.

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use common::*;
use eidola_engine::engine::CacheScope;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::Bucket;
use eidola_engine_cpu::{CpuExecutorConfig, MtpHidden};
use eidola_engine_model::safetensors::WeightSet;
use eidola_engine_model::{LoadOptions, ModelWeights, ReferenceModel};

fn checkpoint_dir() -> PathBuf {
    if let Ok(d) = std::env::var("EIDOLA_MIMO_TRUNCATED_DIR") {
        return d.into();
    }
    let cache = std::env::var("EIDOLA_ENGINE_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/eidola/engine")
        });
    cache.join("mimo-v2.6-flash-mopd/2479e2d0029e/layers-0-1-2-5-mtp")
}

#[test]
#[ignore = "needs a locally fetched truncated checkpoint; see the model crate's golden/README.md"]
fn truncated_flash_through_the_engine_matches_the_dense_forward() {
    let dir = checkpoint_dir();
    let golden = WeightSet::open_files(&[dir.join("golden/hf-golden.safetensors")])
        .expect("golden/hf-golden.safetensors");
    let text: Vec<u32> = {
        let t = golden.get("tokens").unwrap();
        t.to_i64("tokens")
            .unwrap()
            .into_iter()
            .map(|x| x as u32)
            .collect()
    };

    let t0 = Instant::now();
    let store = WeightSet::open_dir(&dir).unwrap();
    let config = store
        .model_config()
        .unwrap()
        .truncated(&[0, 1, 2, 5])
        .unwrap();
    let store = Arc::new(store);
    let model = Arc::new(ReferenceModel::new(
        ModelWeights::load(store, config, &LoadOptions::default()).unwrap(),
    ));
    assert_eq!(model.weights.mtp.len(), 3);
    println!("loaded in {:.1?}", t0.elapsed());

    let cfg = CpuExecutorConfig {
        block_size: 16,
        num_blocks: 32,
        num_state_slots: 4,
        max_model_len: 4096,
        buckets: vec![
            Bucket {
                max_seqs: 1,
                max_tokens: 16,
            },
            Bucket {
                max_seqs: 4,
                max_tokens: 64,
            },
        ],
        mtp_depths: vec![0, 1, 2],
        mtp_hidden: MtpHidden::Normed,
        // MiMo-V2.6's tokenizer: 151,643 base ids plus 32 added tokens. The head's
        // remaining 901 rows are padding.
        sampleable_vocab_size: 151_675,
        pad_batches: false,
        draft_step_tokens: 0,
        record: true,
    };
    let mut sched_cfg = sched(true);
    sched_cfg.max_prefill_chunk = 16;
    sched_cfg.max_batched_tokens = 64;
    let mut h = Harness::new(model.clone(), cfg, sched_cfg);

    let g = SamplingParams::greedy();
    let t0 = Instant::now();
    h.submit(request(
        1,
        text[..40].to_vec(),
        g,
        4,
        CacheScope::Keyed(salt(1)),
    ));
    h.run();
    let first = t0.elapsed();
    let steps_first = h.eng.stats().steps;
    // The same conversation, longer: resumes from the cached first 32 positions.
    let t0 = Instant::now();
    h.submit(request(
        2,
        text[..60].to_vec(),
        g,
        6,
        CacheScope::Keyed(salt(1)),
    ));
    h.step();
    let hit = h.eng.cached_prompt_tokens(2).unwrap();
    h.run();
    let second = t0.elapsed();
    assert_eq!(hit, 32, "prefix hit");
    let stats = h.eng.stats();
    println!(
        "request 1 (40-token prompt, 4 tokens): {steps_first} steps in {first:.1?}; \
         request 2 (60-token prompt, hit {hit}, 6 tokens): {} steps in {second:.1?}",
        stats.steps - steps_first
    );
    println!("outputs: {:?} / {:?}", h.outputs[&1], h.outputs[&2]);

    let t0 = Instant::now();
    let sum = h.check_all(true);
    println!(
        "checked bit-exact against the dense forward in {:.1?}: {sum:?}",
        t0.elapsed()
    );
    println!(
        "MTP acceptance on this 4-layer truncation: {}/{} drafts",
        sum.accepted, sum.drafted
    );
    assert!(sum.drafter_rows > 0 && sum.drafted > 0);
}
