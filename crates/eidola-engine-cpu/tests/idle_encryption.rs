//! Idle prefix-cache KV in the reference executor's pools: ciphertext while idle (every
//! group, the drafter's boundary taps and the row tags included), different under
//! different keys for the same plaintext, and decrypted on a hit into exactly the bits
//! the dense oracle computes. The harness also checks after every step that idle blocks
//! are encrypted and mapped ones plaintext, so every other suite here (divergent turns at
//! the branch, preemption and resume, the randomized workload) runs through encryption
//! and decryption too.

mod common;

use std::collections::BTreeMap;

use common::*;
use eidola_engine::engine::CacheScope;
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;

type Images = BTreeMap<(u32, u32), Vec<u32>>;

fn images<'a>(h: &Harness, blocks: impl IntoIterator<Item = &'a (u32, u32)>) -> Images {
    blocks
        .into_iter()
        .map(|&(g, b)| ((g, b), h.eng.executor().block_image(g, b)))
        .collect()
}

/// Steps until every request has finished, keeping the latest plaintext image of every
/// block: after a step, a block is plaintext in the pool if a sequence maps it or its
/// encryption is still queued (and nothing writes a block once no sequence maps it).
fn run_recording(h: &mut Harness, plain: &mut Images) {
    while h.eng.unfinished() > 0 {
        h.step();
        h.now += 1;
        let kv = h.eng.kv();
        let crypts = kv.pending_crypts();
        let mut blocks: Vec<(u32, u32)> = kv
            .sequence_blocks()
            .into_iter()
            .filter(|b| !crypts.contains_key(b))
            .collect();
        blocks.extend(crypts.iter().filter(|(_, e)| **e).map(|(b, _)| *b));
        let fresh = images(h, &blocks);
        plain.extend(fresh);
    }
}

/// Words two images share in place.
fn same_words(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x == y).count()
}

#[test]
fn a_hit_after_idle_is_bit_exact_and_idle_kv_is_ciphertext() {
    for spec_on in [false, true] {
        let mut h = Harness::fixture(512, vec![0, 1], sched(spec_on));
        let prompt = TestRng(40).tokens(30, VOCAB);
        let g = SamplingParams::greedy();
        let groups = h.spec.kv_groups.len() as u32;

        h.submit(request(1, prompt.clone(), g, 6, CacheScope::Keyed(salt(1))));
        let mut seen = Images::new();
        run_recording(&mut h, &mut seen);
        let cached = h.eng.kv().cache_blocks();
        let plain: Images = cached.iter().map(|b| (*b, seen[b].clone())).collect();
        h.flush();
        assert_eq!(h.eng.kv_keys_held(), 0);
        let mut seen_groups = std::collections::BTreeSet::new();
        for ((gr, b), p) in &plain {
            assert!(h.eng.executor().block_is_encrypted(*gr, *b));
            let c = h.eng.executor().block_image(*gr, *b);
            assert!(
                same_words(&c, p) * 1000 < c.len(),
                "group {gr} block {b}: idle KV is not ciphertext"
            );
            seen_groups.insert(*gr);
        }
        assert_eq!(seen_groups.len() as u32, groups, "every group is retained");
        let cipher1 = images(&h, &cached);

        // Another lineage, same prompt: a miss, and its idle copy of the same plaintext is
        // other ciphertext.
        h.submit(request(2, prompt.clone(), g, 6, CacheScope::Keyed(salt(2))));
        let mut seen = Images::new();
        run_recording(&mut h, &mut seen);
        let plain2: Images = h
            .eng
            .kv()
            .cache_blocks()
            .difference(&cached)
            .map(|b| (*b, seen[b].clone()))
            .collect();
        h.flush();
        let mut pairs = 0;
        for ((g1, b1), p1) in &plain {
            for ((g2, b2), p2) in &plain2 {
                if g1 == g2 && p1 == p2 {
                    pairs += 1;
                    let c2 = h.eng.executor().block_image(*g2, *b2);
                    assert!(
                        same_words(&cipher1[&(*g1, *b1)], &c2) * 1000 < c2.len(),
                        "equal plaintext under two keys gave equal ciphertext"
                    );
                }
            }
        }
        assert!(pairs >= plain.len() / 2, "the lineages hold the same KV");
        assert_eq!(h.cached[&2], 0);

        // The first lineage again: a hit, decrypted, bit-exact.
        let mark = h.eng.executor().crypt_log().len();
        h.submit(request(3, prompt.clone(), g, 6, CacheScope::Keyed(salt(1))));
        h.step();
        assert_eq!(h.eng.cached_prompt_tokens(3), Some(28));
        assert!(h.eng.executor().crypt_log()[mark..].iter().any(|e| !e.3));
        h.run();
        assert_eq!(h.outputs[&3], h.outputs[&1]);
        assert_eq!(h.outputs[&2], h.outputs[&1]);
        let sum = h.check_all(spec_on);
        eprintln!("speculative {spec_on}: {sum:?}");
        h.flush();
        assert_eq!(h.eng.kv_keys_held(), 0);

        // The same request with no prefix cache at all.
        let mut cfg = sched(spec_on);
        cfg.cache.enabled = false;
        let mut cold = Harness::fixture(512, vec![0, 1], cfg);
        cold.submit(request(1, prompt.clone(), g, 6, CacheScope::Private));
        cold.run();
        assert_eq!(cold.outputs[&1], h.outputs[&3]);
        assert!(cold.eng.executor().crypt_log().is_empty());
    }
}

/// Every cached block's ciphertext, per group, in a canonical order: what the pools hold
/// for a lineage, independent of which physical blocks hold it.
fn sealed_ciphertexts(h: &Harness) -> BTreeMap<u32, Vec<Vec<u32>>> {
    let mut out: BTreeMap<u32, Vec<Vec<u32>>> = BTreeMap::new();
    for (g, b) in h.eng.kv().cache_blocks() {
        assert!(h.eng.executor().block_is_encrypted(g, b));
        out.entry(g)
            .or_default()
            .push(h.eng.executor().block_image(g, b));
    }
    for images in out.values_mut() {
        images.sort();
    }
    out
}

/// A sealed block's nonce is its salted hash and group, so it is safe only if the block's
/// bytes are the same every time it is sealed. One lineage's request runs alone and its
/// idle entries are captured as ciphertext; then they expire and the same request is
/// recomputed beside another sequence (in the same engine, and in one prefilling in
/// chunks that straddle block boundaries), and every block is sealed to the same
/// ciphertext, word for word, in every group.
#[test]
fn a_recomputed_block_seals_to_the_same_ciphertext() {
    for spec_on in [false, true] {
        let prompt = TestRng(41).tokens(30, VOCAB);
        let other = TestRng(42).tokens(45, VOCAB);
        let g = SamplingParams::greedy();
        let mine = |id| request(id, prompt.clone(), g, 6, CacheScope::Keyed(salt(1)));
        let beside = |id| request(id, other.clone(), g, 9, CacheScope::Private);

        let mut h = Harness::fixture(512, vec![0, 1], sched(spec_on));
        h.submit(mine(1));
        h.run();
        h.flush();
        let first = sealed_ciphertexts(&h);
        assert_eq!(
            first.len(),
            h.spec.kv_groups.len(),
            "every group is retained"
        );
        assert!(first.values().all(|v| !v.is_empty()));

        // Expired and zeroed, then recomputed beside a private request that started first.
        h.now += CachePolicy::default().max_age_ms + 1;
        h.eng.sweep(h.now).unwrap();
        h.check();
        assert!(h.eng.kv().cache_blocks().is_empty());
        h.submit(beside(2));
        h.step();
        h.now += 1;
        h.submit(mine(3));
        h.run();
        h.flush();
        assert_eq!(h.cached[&3], 0, "a recompute, not a hit");
        assert_eq!(h.outputs[&3], h.outputs[&1]);
        assert!(
            sealed_ciphertexts(&h) == first,
            "speculative {spec_on}: a block recomputed beside another sequence sealed to \
             other ciphertext"
        );

        // Another engine, prefilling 3 tokens a step (blocks of 4), the other request
        // arriving with it.
        let mut cfg = sched(spec_on);
        cfg.max_prefill_chunk = 3;
        let mut c = Harness::fixture(512, vec![0, 1], cfg);
        c.submit(mine(1));
        c.submit(beside(2));
        c.run();
        c.flush();
        assert_eq!(c.outputs[&1], h.outputs[&1]);
        assert!(
            sealed_ciphertexts(&c) == first,
            "speculative {spec_on}: a block recomputed in other chunks sealed to other \
             ciphertext"
        );
    }
}
