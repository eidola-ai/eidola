//! The DeepSeek-V4.1 executor's memory, on real bytes: every group's idle KV is
//! ciphertext over its packed image, a recomputed sealed block seals to the same
//! ciphertext wherever its bytes are a function of the tokens it covers and to another
//! where they are not (the chain-covered groups, whose nonce is per attachment), the
//! host tier round-trips that ciphertext, every freed block and reset slot is zero, and
//! a randomized workload holds the engine's invariants after every step with every row
//! equal to the dense oracle.

mod ds_common;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use ds_common::*;
use eidola_engine::engine::{CacheScope, FinishReason};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::Coverage;

type Images = BTreeMap<(u32, u32), Vec<u32>>;

fn images<'a>(h: &Harness, blocks: impl IntoIterator<Item = &'a (u32, u32)>) -> Images {
    blocks
        .into_iter()
        .map(|&(g, b)| ((g, b), h.eng.executor().block_image(g, b)))
        .collect()
}

/// Steps until every request has finished, keeping the latest plaintext image of every
/// block: after a step, a block is plaintext in the pool if a sequence maps it or its
/// encryption is still queued.
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

fn same_words(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x == y).count()
}

/// Idle KV of every group is ciphertext over the packed image (codes, scales and tags),
/// different under another key, and a hit decrypts it into exactly the bits the dense
/// oracle computes.
#[test]
fn a_hit_after_idle_is_bit_exact_and_idle_kv_is_ciphertext() {
    for spec_on in [false, true] {
        let mut h = Harness::fixture(512, 4, sched(spec_on));
        let prompt = TestRng(40).tokens(210);
        let g = SamplingParams::greedy();
        h.submit(request(1, prompt.clone(), g, 6, keyed(1)));
        let mut seen = Images::new();
        run_recording(&mut h, &mut seen);
        let cached = h.eng.kv().cache_blocks();
        let plain: Images = cached.iter().map(|b| (*b, seen[b].clone())).collect();
        h.flush();
        assert_eq!(h.eng.kv_keys_held(), 0);
        let mut groups = BTreeSet::new();
        for ((gr, b), p) in &plain {
            assert!(h.eng.executor().block_is_encrypted(*gr, *b));
            let c = h.eng.executor().block_image(*gr, *b);
            assert!(
                same_words(&c, p) * 100 < c.len(),
                "group {gr} block {b}: idle KV is not ciphertext"
            );
            groups.insert(*gr);
        }
        assert_eq!(
            groups,
            BTreeSet::from([0, 1, 2, 3, 4]),
            "every group is retained"
        );

        // Another lineage, same prompt: a miss.
        assert_eq!(h.run_one(2, &prompt, g, 6, keyed(2)), 0);
        // The first lineage again: a hit at the regeneration point, decrypted, bit-exact.
        let mark = h.eng.executor().crypt_log().len();
        h.submit(request(3, prompt.clone(), g, 6, keyed(1)));
        h.step();
        assert_eq!(h.eng.cached_prompt_tokens(3), Some(208));
        let decrypted: BTreeSet<u32> = h.eng.executor().crypt_log()[mark..]
            .iter()
            .filter(|e| !e.3)
            .map(|e| e.1)
            .collect();
        assert_eq!(decrypted, BTreeSet::from([0, 1, 2, 3, 4]));
        h.run();
        assert_eq!(h.outputs[&3], h.outputs[&1]);
        assert_eq!(h.outputs[&2], h.outputs[&1]);
        let sum = h.check_all(spec_on);
        eprintln!("speculative {spec_on}: {sum:?}");
        h.flush();
        assert_eq!(h.eng.kv_keys_held(), 0);
    }
}

/// Every cached block's ciphertext, per group, in a canonical order.
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

/// A block of a group whose coverage is `All` is sealed under a nonce fixed by its hash
/// and group, which is safe only if its bytes are the same every time it is sealed: one
/// lineage's request recomputed beside another sequence, and in other prefill chunks,
/// seals every such block (the pooled bank, the encoder's windows, the per-position
/// bank) to the same ciphertext word for word. The chain-covered groups' blocks hold
/// the same plaintext here and still seal to other ciphertext: their nonce is per
/// attachment, because in general their bytes depend on the triangle that was run.
#[test]
fn a_recomputed_block_seals_to_the_same_ciphertext() {
    let prompt = TestRng(41).tokens(210);
    let other = TestRng(42).tokens(150);
    let g = SamplingParams::greedy();
    let mine = |id| request(id, prompt.clone(), g, 6, keyed(1));
    let beside = |id| request(id, other.clone(), g, 9, CacheScope::Private);

    let mut h = Harness::fixture(512, 4, sched(true));
    h.submit(mine(1));
    h.run();
    h.flush();
    let first = sealed_ciphertexts(&h);
    assert_eq!(first.len(), 5, "every group is retained");

    // `same_engine`: the attachments compared were minted by one KV manager.
    let check = |h: &Harness, what: &str, same_engine: bool| {
        let again = sealed_ciphertexts(h);
        for (group, spec) in h.spec.kv_groups.iter().enumerate() {
            let g = group as u32;
            assert_eq!(again[&g].len(), first[&g].len(), "{what}: group {g}");
            match spec.coverage {
                Coverage::All => assert!(
                    again[&g] == first[&g],
                    "{what}: a block of group {g} sealed to other ciphertext"
                ),
                Coverage::Chain if same_engine => assert!(
                    again[&g].iter().all(|c| !first[&g].contains(c)),
                    "{what}: two attachments of group {g} share a keystream"
                ),
                Coverage::Chain => {}
            }
        }
    };

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
    check(&h, "beside another sequence", true);

    // Another engine, prefilling 13 tokens a step (blocks of 8), the other request
    // arriving with it.
    let mut c = Harness::fixture(512, 4, chunked(13, true));
    c.submit(mine(1));
    c.submit(beside(2));
    c.run();
    c.flush();
    assert_eq!(c.outputs[&1], h.outputs[&1]);
    check(&c, "in other chunks", false);
}

/// Device pressure demotes a lineage's idle entries to the host tier as the ciphertext
/// they are, in every group; the hit that promotes them back is the dense oracle's.
#[test]
fn the_host_tier_round_trips_every_group() {
    let _guard = memlock_guard();
    let mut cfg = exec_config(48, 4);
    cfg.host_blocks = 256;
    let mut h = Harness::new(fixture(), cfg, sched(true));
    let mut rng = TestRng(43);
    let g = SamplingParams::greedy();
    let prompt = rng.tokens(120);
    assert_eq!(h.run_one(1, &prompt, g, 6, keyed(1)), 0);
    let first = h.eng.kv().cache_blocks();
    let ciphertext = images(&h, &first);
    let mark = h.eng.executor().transfer_log().len();
    for i in 0..4u64 {
        let other = rng.tokens(120);
        h.run_one(10 + i, &other, g, 5, keyed(100 + i as u8));
    }
    let demoted: Vec<(u32, u32, u32)> = h.eng.executor().transfer_log()[mark..]
        .iter()
        .filter(|t| t.4)
        .map(|t| (t.1, t.2, t.3))
        .collect();
    let blocks: BTreeSet<(u32, u32)> = demoted.iter().map(|d| (d.0, d.1)).collect();
    assert!(first.iter().all(|b| blocks.contains(b)), "all demoted");
    assert_eq!(
        blocks.iter().map(|b| b.0).collect::<BTreeSet<_>>(),
        BTreeSet::from([0, 1, 2, 3, 4])
    );
    // The first entry's blocks arrived on the host word for word.
    let mut compared = 0;
    for (g, b, host) in &demoted {
        if let Some(c) = ciphertext.get(&(*g, *b))
            && h.eng.kv().host_cache_blocks().contains(&(*g, *host))
            && h.eng.executor().host_block_image(*g, *host) == *c
        {
            compared += 1;
        }
    }
    assert!(
        compared > 0,
        "no demoted block of the first entry was found on the host"
    );
    let stats = h.eng.kv().tier_stats();
    assert_eq!((stats.host_drops, stats.device_drops), (0, 0));

    let mark = h.eng.executor().transfer_log().len();
    assert_eq!(h.run_one(2, &prompt, g, 6, keyed(1)), 112);
    let promoted: BTreeSet<u32> = h.eng.executor().transfer_log()[mark..]
        .iter()
        .filter(|t| !t.4)
        .map(|t| t.1)
        .collect();
    assert_eq!(promoted, BTreeSet::from([0, 1, 2, 3, 4]));
    assert_eq!(h.outputs[&2], h.outputs[&1]);
    let sum = h.check_all(true);
    eprintln!("{sum:?} {stats:?}");
}

/// Randomized workload: multi-turn keyed conversations (continuations, regenerations,
/// branches), private requests, seeded sampling, four-slot drafting on or off,
/// cancellations, clock jumps past the idle TTL, pools small enough to force eviction,
/// shedding and preemption in the middle of triangles, and on odd seeds a host tier.
/// After every step: the KV manager's invariants, every free block and free slot zero
/// on real bytes, idle blocks ciphertext and mapped ones plaintext. At the end: every
/// row of every step against the dense oracle, every finished output against dense
/// generation, and after expiry every block of every group zero.
#[test]
fn randomized_workloads_hold_every_invariant() {
    let _guard = memlock_guard();
    let (mut preempted, mut hits, mut triangles) = (0, 0, 0);
    for seed in 0..6u64 {
        let mut rng = TestRng(700 + seed);
        let spec_on = seed % 2 == 0;
        let blocks = 64 + rng.below(48) as u32;
        let mut cfg = sched(spec_on);
        cfg.max_prefill_chunk = 5 + rng.below(90) as u32;
        cfg.max_batched_tokens = 64 + rng.below(200) as u32;
        cfg.cache.idle_ttl_ms = 500;
        cfg.cache.max_age_ms = 2000;
        cfg.sweep_interval_ms = 50;
        let mut exec = exec_config(blocks, 4);
        if seed % 2 == 1 {
            exec.host_blocks = 2 * blocks;
        }
        let mut h = Harness::new(fixture(), exec, cfg);
        let mut convo: HashMap<u8, Vec<u32>> = HashMap::new();
        let mut live: BTreeSet<u64> = BTreeSet::new();
        let mut next_id = 0u64;
        for _round in 0..110 {
            if rng.chance(0.3) && live.len() < 10 {
                let id = next_id;
                next_id += 1;
                let p = if spec_on && rng.chance(0.5) {
                    SamplingParams::greedy()
                } else {
                    params(id)
                };
                let (prompt, cache) = if rng.chance(0.25) {
                    (rng.var_tokens(1, 260), CacheScope::Private)
                } else {
                    let s = rng.below(3) as u8;
                    let prompt = match convo.get(&s) {
                        Some(t) if rng.chance(0.3) => t.clone(),
                        Some(t) if rng.chance(0.3) => {
                            let mut t = t[..1 + rng.below(t.len() as u64) as usize].to_vec();
                            t.extend(rng.var_tokens(1, 20));
                            t
                        }
                        Some(t) => {
                            let mut t = t.clone();
                            t.extend(rng.var_tokens(1, 20));
                            t
                        }
                        None => rng.var_tokens(20, 230),
                    };
                    (prompt, keyed(s + 1))
                };
                if prompt.len() as u32 > blocks * 8 / 2 {
                    continue;
                }
                if let CacheScope::Keyed(_) = cache {
                    convo.insert((id % 3) as u8, prompt.clone());
                }
                live.insert(id);
                h.submit(request(id, prompt, p, 1 + rng.below(16) as u32, cache));
            }
            if rng.chance(0.03) && !live.is_empty() {
                let victim = *live
                    .iter()
                    .nth(rng.below(live.len() as u64) as usize)
                    .unwrap();
                if h.eng.cancel(victim, h.now).is_some() {
                    live.remove(&victim);
                }
            }
            h.step();
            live.retain(|id| !h.finished.contains_key(id));
            h.now += if rng.chance(0.02) {
                600
            } else {
                1 + rng.below(5)
            };
        }
        h.run();
        let stats = h.eng.stats();
        hits += h.cached.values().filter(|c| **c > 0).count();
        preempted += stats.preemptions;
        // A sequence that exhausted the pool alone ended early; its output is a prefix.
        let early = h
            .finished
            .values()
            .filter(|f| **f == FinishReason::KvExhausted)
            .count();
        let sum = h.check_all(spec_on);
        triangles += sum.triangles;
        eprintln!(
            "seed {seed}: blocks {blocks} requests {next_id} {stats:?} {sum:?} early {early} \
             tier {:?}",
            h.eng.kv().tier_stats()
        );
        // Drain: everything expires, and every block of every group and every slot is
        // zero on real bytes.
        h.eng.sweep(h.now + 10_000).unwrap();
        h.check();
        assert_eq!(h.eng.kv().cache_entries(), 0);
        for (g, gs) in h.spec.kv_groups.iter().enumerate() {
            assert_eq!(h.eng.kv().free_blocks(g), gs.num_blocks as usize - 1);
            for b in 1..gs.num_blocks {
                assert!(
                    h.eng.executor().block_is_zero(g as u32, b),
                    "seed {seed}: group {g} block {b} holds bytes after the drain"
                );
            }
        }
        for slot in 0..h.spec.num_state_slots {
            assert!(h.eng.executor().slot_state_is_zero(slot));
        }
    }
    assert!(
        preempted > 0 && hits > 0 && triangles > 0,
        "{preempted} {hits} {triangles}"
    );
}
