//! The Qwen3.8 CPU executor's memory on real bytes: idle state snapshots and KV blocks
//! held as ciphertext and a hit after them exact, recomputed snapshots the same bytes,
//! the host tier round trip, expiry zeroing snapshots, and a randomized workload under
//! the harness's per-step checks (every free block and state buffer zero, every reset
//! slot empty, every idle block and snapshot ciphertext), every row and every stored state
//! against the dense oracle.

mod qwen_common;

use eidola_engine::engine::{CacheScope, FinishReason};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use qwen_common::*;

#[test]
fn idle_snapshots_are_ciphertext_and_a_hit_after_them_is_exact() {
    let g = SamplingParams::greedy();
    let mut h = Harness::fixture(256, 32, 3, sched(true));
    let prompt = TestRng(1).tokens(13, VOCAB);
    let out = h.run_one(request(1, prompt.clone(), g, 7, keyed(1)));
    let kv = h.eng.kv();
    let p = kv.state_pool().unwrap();
    let held = kv.state_snapshots();
    assert_eq!(held.iter().map(|s| s.0).collect::<Vec<_>>(), vec![12, 16]);
    let plain: Vec<_> = h
        .stored
        .iter()
        .filter(|s| s.len == 12 || s.len == 16)
        .map(|s| s.len)
        .collect();
    assert!(!plain.is_empty());
    for &(len, b, _) in &held {
        let exec = h.eng.executor();
        assert!(
            exec.block_is_encrypted(p, b),
            "snapshot at {len} left plaintext"
        );
        // Neither the state's bits nor the position tag are in the clear.
        let image = exec.block_image(p, b);
        assert_ne!(*image.last().unwrap(), len + 1, "a tag in the clear");
    }
    // The same lineage resumes from the regeneration snapshot, decrypting exactly it.
    let mark = h.eng.executor().crypt_log().len();
    let again = h.run_one(request(2, prompt.clone(), g, 7, keyed(1)));
    assert_eq!(again, out);
    assert_eq!(h.cached[&2], 12);
    let regen = held.iter().find(|s| s.0 == 12).unwrap().1;
    let decrypted: Vec<u32> = h.eng.executor().crypt_log()[mark..]
        .iter()
        .filter(|c| c.1 == p && !c.3)
        .map(|c| c.2)
        .collect();
    assert_eq!(decrypted, vec![regen]);
    // Another lineage finds nothing and decrypts no snapshot.
    let mark = h.eng.executor().crypt_log().len();
    h.run_one(request(3, prompt, g, 7, keyed(2)));
    assert_eq!(h.cached[&3], 0);
    assert!(
        !h.eng.executor().crypt_log()[mark..]
            .iter()
            .any(|c| c.1 == p && !c.3 && held.iter().any(|s| s.1 == c.2))
    );
    h.check_all(true);
    assert_eq!(h.eng.kv_keys_held(), 0);
}

/// The state at a boundary is the same bytes whichever request computes it, in whatever
/// chunks: two lineages store equal plaintext states, which encrypt to different
/// ciphertext (another key and another attachment).
#[test]
fn a_recomputed_snapshot_is_the_same_bytes() {
    let g = SamplingParams::greedy();
    let prompt = TestRng(2).tokens(19, VOCAB);
    let mut images = Vec::new();
    for (lineage, chunk) in [(1, 256), (2, 3)] {
        let mut h = Harness::fixture(256, 32, 3, chunked(chunk, true));
        h.run_one(request(1, prompt.clone(), g, 3, keyed(lineage)));
        let s = h
            .stored
            .iter()
            .find(|s| s.len == 16)
            .expect("the state at 16 stored")
            .clone();
        let p = h.eng.kv().state_pool().unwrap();
        let (_, b, _) = *h
            .eng
            .kv()
            .state_snapshots()
            .iter()
            .find(|s| s.0 == 16)
            .unwrap();
        images.push((s, h.eng.executor().block_image(p, b)));
        h.check_all(true);
    }
    assert_eq!(images[0].0.states, images[1].0.states);
    assert_ne!(images[0].1, images[1].1, "ciphertext under two keys");
}

#[test]
fn the_host_tier_round_trips_snapshots() {
    let _guard = memlock_guard();
    let g = SamplingParams::greedy();
    let mut cfg = exec_config(48, 5, 3);
    cfg.host_blocks = 64;
    cfg.host_state_buffers = 8;
    let mut h = Harness::new(fixture(), cfg, sched(true));
    let mut rng = TestRng(3);
    let first = rng.tokens(14, VOCAB);
    let out = h.run_one(request(1, first.clone(), g, 6, keyed(1)));
    for id in 2..7u64 {
        h.run_one(request(id, rng.tokens(14, VOCAB), g, 6, keyed(id as u8)));
    }
    let p = h.eng.kv().state_pool().unwrap();
    let log = h.eng.executor().transfer_log().to_vec();
    assert!(log.iter().any(|t| t.1 == p && t.4), "no snapshot demoted");
    // Every demoted snapshot's host image is ciphertext: its position tag is noise.
    for t in log.iter().filter(|t| t.1 == p && t.4) {
        let image = h.eng.executor().host_block_image(p, t.3);
        let tag = *image.last().unwrap();
        assert!(
            tag > 512,
            "a position tag in the clear on the host tier: {tag}"
        );
    }
    let mark = h.eng.executor().transfer_log().len();
    let again = h.run_one(request(9, first, g, 6, keyed(1)));
    assert_eq!(again, out);
    assert_eq!(h.cached[&9], 12);
    assert!(
        h.eng.executor().transfer_log()[mark..]
            .iter()
            .any(|t| t.1 == p && !t.4),
        "the hit's snapshot was not promoted"
    );
    h.check_all(true);
}

#[test]
fn expired_snapshots_are_zeroed() {
    let mut s = sched(false);
    s.cache = CachePolicy {
        enabled: true,
        idle_ttl_ms: 50,
        max_age_ms: 10_000,
    };
    let mut h = Harness::fixture(256, 32, 0, s);
    h.run_one(request(
        1,
        TestRng(4).tokens(11, VOCAB),
        SamplingParams::greedy(),
        6,
        keyed(1),
    ));
    let held = h.eng.kv().state_snapshots();
    assert!(!held.is_empty());
    let p = h.eng.kv().state_pool().unwrap();
    h.now += 1_000;
    h.eng.sweep(h.now).unwrap();
    h.check();
    assert!(h.eng.kv().state_snapshots().is_empty());
    for &(_, b, _) in &held {
        assert!(h.eng.executor().block_is_zero(p, b));
    }
    h.check_all(false);
}

#[test]
fn a_randomized_workload_keeps_every_invariant() {
    let _guard = memlock_guard();
    for seed in 0..4u64 {
        let tier = seed % 2 == 1;
        let k = if seed % 3 == 2 { 0 } else { 3 };
        let mut cfg = exec_config(40 + seed as u32 * 8, 5 + seed as u32, k);
        if tier {
            cfg.host_blocks = 48;
            cfg.host_state_buffers = 6;
        }
        let mut h = Harness::new(fixture(), cfg, chunked(1 + seed as u32 * 3, k > 0));
        let mut rng = TestRng(seed + 1000);
        let mut prompts: Vec<Vec<u32>> = Vec::new();
        for id in 0..18u64 {
            let mut prompt = rng.var_tokens(1, 18, VOCAB);
            if id > 0 && rng.chance(0.5) {
                let earlier = &prompts[rng.below(prompts.len() as u64) as usize];
                let keep = 1 + rng.below(earlier.len() as u64) as usize;
                let mut p = earlier[..keep].to_vec();
                p.extend(prompt);
                prompt = p;
            }
            prompts.push(prompt.clone());
            let cache = if rng.chance(0.3) {
                CacheScope::Private
            } else {
                keyed(rng.below(3) as u8)
            };
            h.submit(request(
                id,
                prompt,
                params(id),
                1 + rng.below(12) as u32,
                cache,
            ));
            if rng.chance(0.4) {
                h.step();
                h.now += 1;
            }
            if rng.chance(0.06) {
                let victim = rng.below(id + 1);
                if h.eng.cancel(victim, h.now).is_some() {
                    h.finished.insert(victim, FinishReason::Cancelled);
                }
            }
        }
        h.run();
        h.flush();
        h.check_all(k > 0);
        assert_eq!(h.eng.kv_keys_held(), 0);
        for slot in 0..32 {
            assert!(
                h.eng.executor().slot_state_is_zero(slot),
                "seed {seed} slot {slot}"
            );
        }
    }
}
