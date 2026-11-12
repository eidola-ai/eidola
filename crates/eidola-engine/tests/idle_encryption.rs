//! Idle prefix-cache KV is ciphertext: through the scheduler, KV manager and mock
//! executor, every block an entry holds and no sequence maps is encrypted under its
//! lineage's key, decrypted before a hit reads it, zeroed (never kept encrypted) when its
//! entry goes, and no key outlives the requests that carried it.
//!
//! The harness checks the executor's state against the manager's after every step (idle
//! blocks encrypted, mapped ones plaintext, once queued work runs); the mock fails any
//! read of a block left encrypted, any decrypt under another key, and any decrypt of
//! plaintext or double encryption.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::*;
use eidola_engine::engine::CacheScope;
use eidola_engine::mock::MockConfig;
use eidola_engine::sampling::SamplingParams;

fn keyed(n: u8) -> CacheScope {
    CacheScope::Keyed(salt(n))
}

/// `(group, block)` of the encryptions (`true`) or decryptions executed since `mark`.
fn crypts_since(h: &Harness, mark: usize, encrypt: bool) -> Vec<(u32, u32)> {
    h.eng.executor().crypt_log()[mark..]
        .iter()
        .filter(|e| e.3 == encrypt)
        .map(|&(_, g, b, _)| (g, b))
        .collect()
}

fn as_set(v: &[(u32, u32)]) -> BTreeSet<(u32, u32)> {
    v.iter().copied().collect()
}

fn groups_of(set: &BTreeSet<(u32, u32)>) -> BTreeSet<u32> {
    set.iter().map(|&(g, _)| g).collect()
}

/// A finished keyed request leaves every cached block encrypted, in every group; the next
/// request of the lineage decrypts exactly what it hits and produces the reference output.
#[test]
fn idle_entries_are_ciphertext_and_a_hit_decrypts_them() {
    let mut h = Harness::new(small_spec(512), sched(true), MockConfig::default());
    let prompt = TestRng(1).tokens(37, 32);
    let g = SamplingParams::greedy();
    h.submit(request(1, prompt.clone(), g, 10, keyed(1)));
    h.run();
    assert!(
        h.eng.kv_keys_held() > 0,
        "the release's encryptions hold the key"
    );
    h.flush();
    let cached = h.eng.kv().cache_blocks();
    assert_eq!(h.eng.kv().idle_blocks(), cached);
    assert_eq!(
        groups_of(&cached),
        BTreeSet::from([0, 1, 2]),
        "global, sliding and drafter groups are all retained"
    );
    assert_eq!(as_set(&crypts_since(&h, 0, true)), cached);
    for &(gr, b) in &cached {
        assert!(h.eng.executor().block_is_encrypted(gr, b));
    }
    assert_eq!(h.eng.kv_keys_held(), 0);

    // A regeneration hits 36 positions: blocks 0..9 of the global group and the windows
    // of the regeneration point, every one decrypted before it is read.
    let mark = h.eng.executor().crypt_log().len();
    h.submit(request(2, prompt.clone(), g, 10, keyed(1)));
    h.step();
    assert_eq!(h.eng.cached_prompt_tokens(2), Some(36));
    let decrypted = as_set(&crypts_since(&h, mark, false));
    // What the request still maps of the old entries was decrypted (its first step's
    // commit may already have slid sliding blocks out of its window and re-encrypted them).
    let mapped: BTreeSet<_> = h
        .eng
        .kv()
        .sequence_blocks()
        .intersection(&cached)
        .copied()
        .collect();
    assert!(decrypted.is_subset(&cached) && mapped.is_subset(&decrypted));
    assert_eq!(groups_of(&decrypted), BTreeSet::from([0, 1, 2]));
    h.run();
    assert_eq!(h.outputs[&2], h.outputs[&1]);
    assert_eq!(h.outputs[&2], h.expected(&prompt, &g, 10, &[]));
    h.flush();
    assert_eq!(h.eng.kv().idle_blocks(), h.eng.kv().cache_blocks());
    assert_eq!(h.eng.kv_keys_held(), 0);
}

/// Another lineage with the same prompt misses, never decrypts the first lineage's blocks,
/// and leaves its own encrypted under a different key.
#[test]
fn a_different_key_misses_and_never_decrypts_another_lineage() {
    let mut h = Harness::new(small_spec(512), sched(false), MockConfig::default());
    let prompt = TestRng(2).tokens(37, 32);
    let g = SamplingParams::greedy();
    h.submit(request(1, prompt.clone(), g, 6, keyed(1)));
    h.run();
    h.flush();
    let first = h.eng.kv().cache_blocks();
    let mark = h.eng.executor().crypt_log().len();
    h.submit(request(2, prompt.clone(), g, 6, keyed(2)));
    h.run();
    h.flush();
    assert_eq!(h.cached[&2], 0);
    assert!(
        crypts_since(&h, mark, false).is_empty(),
        "nothing decrypted"
    );
    assert_eq!(h.outputs[&2], h.outputs[&1]);
    let ex = h.eng.executor();
    let key_of = |set: &BTreeSet<(u32, u32)>| -> BTreeSet<u64> {
        set.iter()
            .map(|&(gr, b)| {
                ex.encrypted_under(gr, b)
                    .expect("idle blocks are encrypted")
            })
            .collect()
    };
    let second: BTreeSet<_> = h
        .eng
        .kv()
        .cache_blocks()
        .difference(&first)
        .copied()
        .collect();
    let (a, b) = (key_of(&first), key_of(&second));
    assert_eq!((a.len(), b.len()), (1, 1), "one key per lineage");
    assert_ne!(a, b);
}

/// Requests of one lineage running together share plaintext: each block they hit is
/// decrypted once, and encrypted again only when the last of them lets go of it.
#[test]
fn concurrent_requests_of_a_lineage_share_plaintext() {
    let mut h = Harness::new(small_spec(512), sched(false), MockConfig::default());
    let mut rng = TestRng(3);
    let prefix = rng.tokens(40, 32);
    let g = SamplingParams::greedy();
    h.submit(request(1, prefix.clone(), g, 4, keyed(1)));
    h.run();
    h.flush();
    let mark = h.eng.executor().crypt_log().len();
    let mut prompts = Vec::new();
    for (id, max) in [(2u64, 30u32), (3, 3)] {
        let mut p = prefix.clone();
        p.extend(rng.tokens(5, 32));
        prompts.push(p.clone());
        h.submit(request(id, p, g, max, keyed(1)));
    }
    let mut shared_released = false;
    while h.eng.unfinished() > 0 {
        h.step();
        h.now += 1;
        let log = &h.eng.executor().crypt_log()[mark..];
        let mut decrypts: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        for e in log.iter().filter(|e| !e.3) {
            *decrypts.entry((e.1, e.2)).or_default() += 1;
        }
        if h.finished.len() < 3 {
            assert!(
                decrypts.values().all(|&n| n == 1),
                "a block decrypted twice"
            );
            shared_released |= h.finished.contains_key(&3);
        }
    }
    assert!(
        shared_released,
        "request 3 finishes while request 2 still runs"
    );
    assert_eq!(h.outputs[&2], h.expected(&prompts[0], &g, 30, &[]));
    assert_eq!(h.outputs[&3], h.expected(&prompts[1], &g, 3, &[]));
    h.flush();
}

/// An encrypted entry that expires is zeroed, exactly its blocks, and is no longer held
/// encrypted.
#[test]
fn expiry_of_an_encrypted_entry_zeroes_it() {
    let mut cfg = sched(false);
    cfg.sweep_interval_ms = u64::MAX;
    let ttl = cfg.cache.idle_ttl_ms;
    let mut h = Harness::new(small_spec(512), cfg, MockConfig::default());
    let prompt = TestRng(4).tokens(37, 32);
    h.submit(request(1, prompt, SamplingParams::greedy(), 5, keyed(1)));
    h.run();
    h.flush();
    let cached = h.eng.kv().cache_blocks();
    let finished_at = h.now - 1;
    let zmark = h.eng.executor().zero_log().len();
    let cmark = h.eng.executor().crypt_log().len();
    h.eng.sweep(finished_at + ttl).unwrap();
    let zeroed: BTreeSet<_> = h.eng.executor().zero_log()[zmark..]
        .iter()
        .map(|&(_, g, b)| (g, b))
        .collect();
    assert_eq!(zeroed, cached);
    assert!(
        h.eng.executor().crypt_log()[cmark..].is_empty(),
        "zeroed, not decrypted"
    );
    for (g, b) in cached {
        assert!(h.eng.executor().block_is_zero(g, b));
        assert!(!h.eng.executor().block_is_encrypted(g, b));
    }
    h.check();
}

/// Under pool pressure an encrypted entry is evicted: its blocks are zeroed without being
/// decrypted, and the request that needed the room runs correctly.
#[test]
fn eviction_of_an_encrypted_entry_zeroes_it() {
    // 29 allocatable blocks of 4 positions per group.
    let mut h = Harness::new(small_spec(30), sched(false), MockConfig::default());
    let mut rng = TestRng(5);
    let g = SamplingParams::greedy();
    h.submit(request(1, rng.tokens(40, 32), g, 4, keyed(1)));
    h.run();
    h.flush();
    let first = h.eng.kv().cache_blocks();
    let zmark = h.eng.executor().zero_log().len();
    let cmark = h.eng.executor().crypt_log().len();
    let big = rng.tokens(90, 32);
    h.submit(request(2, big.clone(), g, 4, keyed(2)));
    h.run();
    h.flush();
    assert_eq!(h.outputs[&2], h.expected(&big, &g, 4, &[]));
    let evicted: BTreeSet<_> = first
        .difference(&h.eng.kv().cache_blocks())
        .copied()
        .collect();
    assert!(!evicted.is_empty(), "the workload must evict");
    let zeroed: BTreeSet<_> = h.eng.executor().zero_log()[zmark..]
        .iter()
        .map(|&(_, g, b)| (g, b))
        .collect();
    assert!(evicted.is_subset(&zeroed));
    let decrypted = as_set(&crypts_since(&h, cmark, false));
    assert!(
        decrypted.is_disjoint(&first),
        "no other lineage's block is decrypted"
    );
}

/// An entry detached at max age while in use is never encrypted and kept: its blocks are
/// zeroed when the request using them releases them.
#[test]
fn an_entry_detached_in_use_is_zeroed_not_encrypted() {
    let mut cfg = sched(false);
    cfg.sweep_interval_ms = u64::MAX;
    let max_age = cfg.cache.max_age_ms;
    let mut h = Harness::new(small_spec(512), cfg, MockConfig::default());
    let prompt = TestRng(6).tokens(30, 32);
    let g = SamplingParams::greedy();
    h.submit(request(1, prompt.clone(), g, 3, keyed(1)));
    h.run();
    h.flush();
    let old = h.eng.kv().cache_blocks();
    h.submit(request(2, prompt.clone(), g, 100, keyed(1)));
    h.step();
    assert_eq!(h.eng.cached_prompt_tokens(2), Some(28));
    let held: BTreeSet<_> = h
        .eng
        .kv()
        .sequence_blocks()
        .intersection(&old)
        .copied()
        .collect();
    assert!(!held.is_empty());
    h.now = max_age;
    h.eng.sweep(h.now).unwrap();
    h.check();
    let step_detached = h.eng.executor().steps();
    h.run();
    h.flush();
    assert_eq!(h.outputs[&2], h.expected(&prompt, &g, 100, &[]));
    let ex = h.eng.executor();
    for &(gr, b) in &held {
        let zeroed_at = ex
            .zero_log()
            .iter()
            .find(|&&(s, zg, zb)| s > step_detached && (zg, zb) == (gr, b))
            .map(|e| e.0)
            .expect("a detached block is zeroed at release");
        assert!(
            !ex.crypt_log().iter().any(|&(s, cg, cb, enc)| enc
                && (cg, cb) == (gr, b)
                && s > step_detached
                && s <= zeroed_at),
            "block {gr}/{b} was encrypted after its entry was detached"
        );
    }
}

/// Divergent turns after a shared prefix: the windows retained at the branch, and the
/// drafter's, are encrypted while idle and decrypted for the hit at the branch.
#[test]
fn branch_and_regeneration_windows_are_encrypted() {
    let mut h = Harness::new(small_spec(512), sched(true), MockConfig::default());
    let mut rng = TestRng(7);
    let prefix = rng.tokens(40, 32);
    let g = SamplingParams::greedy();
    let mut prompts = Vec::new();
    for _ in 0..3 {
        let mut p = prefix.clone();
        p.extend(rng.tokens(9, 32));
        prompts.push(p);
    }
    for (i, p) in prompts[..2].iter().enumerate() {
        h.submit(request(i as u64, p.clone(), g, 4, keyed(1)));
        h.run();
        h.flush();
    }
    let cached = h.eng.kv().cache_blocks();
    assert_eq!(h.eng.kv().idle_blocks(), cached);
    let sliding: BTreeSet<_> = cached.iter().filter(|(gr, _)| *gr > 0).copied().collect();
    assert!(groups_of(&sliding) == BTreeSet::from([1, 2]));
    let mark = h.eng.executor().crypt_log().len();
    h.submit(request(2, prompts[2].clone(), g, 4, keyed(1)));
    h.step();
    assert_eq!(
        h.eng.cached_prompt_tokens(2),
        Some(40),
        "a hit at the branch"
    );
    let decrypted = as_set(&crypts_since(&h, mark, false));
    assert_eq!(groups_of(&decrypted), BTreeSet::from([0, 1, 2]));
    assert!(decrypted.is_subset(&cached));
    h.run();
    assert_eq!(h.outputs[&2], h.expected(&prompts[2], &g, 4, &[]));
    h.flush();
}

/// A preempted request's sealed blocks are encrypted while it waits (under its own key,
/// keyed or not) and decrypted when it resumes on them; outputs are unchanged.
#[test]
fn preemption_encrypts_while_waiting_and_resume_decrypts() {
    for private in [false, true] {
        let mut h = Harness::new(small_spec(41), sched(false), MockConfig::default());
        let mut rng = TestRng(8);
        let mut cases = Vec::new();
        for id in 0..24u64 {
            let prompt = rng.var_tokens(5, 25, 32);
            cases.push(prompt.clone());
            let cache = if private {
                CacheScope::Private
            } else {
                keyed(id as u8)
            };
            h.submit(request(id, prompt, SamplingParams::greedy(), 50, cache));
        }
        h.run();
        assert!(h.eng.stats().preemptions > 0);
        let log = h.eng.executor().crypt_log();
        assert!(log.iter().any(|e| e.3) && log.iter().any(|e| !e.3));
        for (id, prompt) in cases.iter().enumerate() {
            assert_eq!(
                h.outputs[&(id as u64)],
                h.expected(prompt, &SamplingParams::greedy(), 50, &[])
            );
        }
        h.flush();
        assert_eq!(h.eng.kv_keys_held(), 0);
        if private {
            assert_eq!(h.eng.kv().cache_entries(), 0);
        }
    }
}

/// Keys live exactly as long as requests do: held while a request is waiting or running
/// and while the step carrying its encryptions is pending, and by nothing afterwards,
/// whether it finished, was cancelled while running, or was cancelled while waiting.
#[test]
fn no_key_outlives_its_requests() {
    let mut h = Harness::new(small_spec(512), sched(false), MockConfig::default());
    let mut rng = TestRng(9);
    let g = SamplingParams::greedy();
    assert_eq!(h.eng.kv_keys_held(), 0);
    h.submit(request(1, rng.tokens(30, 32), g, 5, keyed(1)));
    assert_eq!(h.eng.kv_keys_held(), 1, "the waiting request holds its key");
    h.step();
    assert_eq!(h.eng.kv_keys_held(), 2, "the request and its sequence");
    h.run();
    assert!(
        h.eng.kv_keys_held() > 0,
        "the release's encryptions are pending"
    );
    h.flush();
    assert_eq!(h.eng.kv_keys_held(), 0);

    h.submit(request(2, rng.tokens(30, 32), g, 50, keyed(2)));
    h.step();
    h.step();
    h.eng.cancel(2, h.now).unwrap();
    h.flush();
    assert_eq!(h.eng.kv_keys_held(), 0);

    h.submit(request(3, rng.tokens(30, 32), g, 50, keyed(3)));
    h.eng.cancel(3, h.now).unwrap();
    h.flush();
    assert_eq!(h.eng.kv_keys_held(), 0);
}

/// Randomized keyed and private traffic with small pools, TTL jumps and cancellations:
/// the harness's check (idle blocks encrypted, mapped ones plaintext on the executor)
/// holds after every step, outputs match the reference, and no key is left at the end.
#[test]
fn randomized_traffic_keeps_idle_blocks_encrypted() {
    for seed in 0..6u64 {
        let mut cfg = sched(seed % 2 == 0);
        cfg.cache.idle_ttl_ms = 40;
        let mut h = Harness::new(small_spec(36 + 8 * seed as u32), cfg, MockConfig::default());
        let mut rng = TestRng(100 + seed);
        let heads: Vec<Vec<u32>> = (0..3).map(|_| rng.tokens(12, 32)).collect();
        let mut cases = Vec::new();
        for id in 0..40u64 {
            let mut prompt = heads[rng.below(3) as usize].clone();
            prompt.extend(rng.var_tokens(1, 20, 32));
            let cache = match rng.below(4) {
                0 => CacheScope::Private,
                k => keyed(k as u8),
            };
            let max = 1 + rng.below(12) as u32;
            cases.push((id, prompt.clone(), max));
            h.submit(request(id, prompt, SamplingParams::greedy(), max, cache));
            for _ in 0..rng.below(4) {
                h.step();
                h.now += 1 + rng.below(15);
            }
            if rng.chance(0.1) && !h.finished.contains_key(&id) {
                h.eng.cancel(id, h.now);
                h.finished
                    .insert(id, eidola_engine::engine::FinishReason::Cancelled);
            }
        }
        h.run();
        h.flush();
        for (id, prompt, max) in cases {
            if h.finished[&id] != eidola_engine::engine::FinishReason::Cancelled {
                assert_eq!(
                    h.outputs[&id],
                    h.expected(&prompt, &SamplingParams::greedy(), max, &[]),
                    "seed {seed} request {id}"
                );
            }
        }
        assert_eq!(h.eng.kv_keys_held(), 0);
        assert_eq!(h.eng.kv().idle_blocks(), h.eng.kv().cache_blocks());
    }
}
