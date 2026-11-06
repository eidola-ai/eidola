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
