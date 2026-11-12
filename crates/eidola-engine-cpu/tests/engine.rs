//! The serving core's engine over the CPU executor on the synthetic fixture. Every row of
//! every step is checked bit for bit against the dense oracle (target logits at every
//! computed position, drafter logits at every drafter row, every draft draw, every
//! produced token), and every finished output against dense generation.

mod common;

use std::collections::{BTreeSet, HashMap};

use common::*;
use eidola_engine::engine::{CacheScope, FinishReason};
use eidola_engine::sampling::SamplingParams;
use eidola_engine_cpu::MtpHidden;
use eidola_engine_model::LoadOptions;

#[test]
fn single_requests_match_the_dense_oracle() {
    for spec_on in [false, true] {
        let mut h = Harness::fixture(256, vec![0, 1], sched(spec_on));
        let mut rng = TestRng(1);
        for id in 0..9 {
            let prompt = rng.var_tokens(1, 30, VOCAB);
            let max = 1 + rng.below(14) as u32;
            h.submit(request(id, prompt, params(id), max, CacheScope::Private));
        }
        h.run();
        for id in 0..9 {
            assert_eq!(h.finished[&id], FinishReason::Length);
        }
        let sum = h.check_all(spec_on);
        eprintln!("speculative {spec_on}: {sum:?}");
        assert!(sum.target_rows > 0 && sum.drafter_rows > 0);
        assert_eq!(sum.drafted > 0, spec_on);
    }
}

#[test]
fn chunked_prefill_is_identical_to_unchunked() {
    let mut rng = TestRng(2);
    let prompts: Vec<Vec<u32>> = (0..4).map(|_| rng.var_tokens(14, 20, VOCAB)).collect();
    for spec_on in [false, true] {
        let mut results = Vec::new();
        for chunk in [1u32, 3, 4, 7, 1000] {
            let mut cfg = sched(spec_on);
            cfg.max_prefill_chunk = chunk;
            cfg.max_batched_tokens = 64.max(chunk.min(256));
            let mut h = Harness::fixture(256, vec![0, 1], cfg);
            for (i, p) in prompts.iter().enumerate() {
                h.submit(request(
                    i as u64,
                    p.clone(),
                    params(i as u64),
                    6,
                    CacheScope::Private,
                ));
            }
            h.run();
            let sum = h.check_all(spec_on);
            eprintln!("speculative {spec_on} chunk {chunk}: {sum:?}");
            results.push(
                (0..prompts.len() as u64)
                    .map(|i| h.outputs[&i].clone())
                    .collect::<Vec<_>>(),
            );
        }
        for r in &results[1..] {
            for (i, out) in r.iter().enumerate() {
                // Seeded speculative sampling is exact only in distribution: the first
                // sample is speculative or not depending on how the prompt was chunked.
                if !spec_on || params(i as u64).is_greedy() {
                    assert_eq!(out, &results[0][i], "request {i}");
                }
            }
        }
    }
}

#[test]
fn greedy_speculation_equals_plain_decoding() {
    let mut rng = TestRng(4);
    let prompts: Vec<Vec<u32>> = (0..8).map(|_| rng.var_tokens(2, 24, VOCAB)).collect();
    let run = |depths: Vec<usize>, spec_on: bool| {
        let mut h = Harness::fixture(256, depths, sched(spec_on));
        for (i, p) in prompts.iter().enumerate() {
            h.submit(request(
                i as u64,
                p.clone(),
                SamplingParams::greedy(),
                24,
                CacheScope::Private,
            ));
        }
        h.run();
        let sum = h.check_all(spec_on);
        let outs: Vec<Vec<u32>> = (0..prompts.len() as u64)
            .map(|i| h.outputs[&i].clone())
            .collect();
        (outs, sum, h.eng.stats())
    };
    let (plain, _, _) = run(vec![0, 1], false);
    // k = 1, 2, and 3 (the fixture ships two MTP layers; the third depth reuses layer 0).
    for depths in [vec![0], vec![0, 1], vec![0, 1, 0]] {
        let (spec, sum, stats) = run(depths.clone(), true);
        assert_eq!(spec, plain, "depths {depths:?}");
        assert_eq!(stats.drafted as usize, sum.drafted);
        assert_eq!(stats.accepted as usize, sum.accepted);
        eprintln!(
            "depths {depths:?}: accepted {}/{} drafts ({:.1} %), per depth {:?}",
            sum.accepted,
            sum.drafted,
            100.0 * sum.accepted as f64 / sum.drafted as f64,
            &sum.per_depth[..depths.len()]
        );
    }
}

#[test]
fn seeded_sampling_with_three_draft_depths_is_exact_per_step() {
    let mut h = Harness::fixture(256, vec![0, 1, 0], sched(true));
    let mut rng = TestRng(5);
    for id in 0..8 {
        let p = SamplingParams::new(
            0.8 + 0.1 * (id % 3) as f32,
            [0, 5, 40][id as usize % 3],
            [1.0, 0.95, 0.8][id as usize % 3],
            0.0,
            77 + id,
        )
        .unwrap();
        h.submit(request(
            id,
            rng.var_tokens(3, 20, VOCAB),
            p,
            16,
            CacheScope::Private,
        ));
    }
    h.run();
    let sum = h.check_all(true);
    eprintln!("{sum:?}");
    assert!(sum.drafted > 0);
}

/// Prefix-cache hits (same salt) resume from cached target and drafter KV, and from the
/// drafter's boundary taps; different salts and private requests never hit.
#[test]
fn prefix_hits_and_salt_isolation() {
    for spec_on in [false, true] {
        let mut h = Harness::fixture(512, vec![0, 1], sched(spec_on));
        let mut rng = TestRng(6);
        let prompt = rng.tokens(30, VOCAB);
        let g = SamplingParams::greedy();
        let mut id = 0;
        let mut run = |h: &mut Harness, prompt: &[u32], p: SamplingParams, cache: CacheScope| {
            id += 1;
            h.submit(request(id, prompt.to_vec(), p, 5, cache));
            h.step();
            let cached = h.eng.cached_prompt_tokens(id).unwrap();
            h.run();
            (id, cached)
        };
        let (first, c) = run(&mut h, &prompt, g, CacheScope::Keyed(salt(1)));
        assert_eq!(c, 0);
        // Same salt, same prompt: the regeneration hit point.
        let (again, c) = run(&mut h, &prompt, g, CacheScope::Keyed(salt(1)));
        assert_eq!(c, 28);
        assert_eq!(h.outputs[&again], h.outputs[&first]);
        // Seeded sampling after a hit.
        let (_, c) = run(&mut h, &prompt, params(1), CacheScope::Keyed(salt(1)));
        assert_eq!(c, 28);
        // Different salt, and no key: nothing.
        assert_eq!(run(&mut h, &prompt, g, CacheScope::Keyed(salt(2))).1, 0);
        assert_eq!(run(&mut h, &prompt, g, CacheScope::Private).1, 0);
        assert_eq!(run(&mut h, &prompt, g, CacheScope::Private).1, 0);
        // A continuation resumes at the previous sequence's last sealed block.
        let mut cont = prompt.clone();
        cont.extend(&h.outputs[&first]);
        cont.extend(rng.tokens(7, VOCAB));
        let (_, c) = run(&mut h, &cont, g, CacheScope::Keyed(salt(1)));
        assert_eq!(c, 32);
        let sum = h.check_all(spec_on);
        eprintln!("speculative {spec_on}: {sum:?}");
    }
}

/// Requests under one key share a prefix spanning several sliding windows (window 8, the
/// drafter's too) and then diverge. The first divergent request recomputes and keeps its
/// window at the branch; every later one resumes there, target and drafter KV and the
/// drafter's boundary tap included, and produces exactly what a cold run produces.
#[test]
fn divergent_turns_after_a_shared_prefix_hit_at_the_branch() {
    for spec_on in [false, true] {
        let mut rng = TestRng(8);
        let prefix = rng.tokens(40, VOCAB);
        let prompts: Vec<Vec<u32>> = (0..5)
            .map(|i| {
                let mut t = prefix.clone();
                t.push(i);
                t.extend(rng.tokens(6, VOCAB));
                t
            })
            .collect();
        // Seeded speculative sampling is exact only in distribution (the first sample is
        // speculative or not depending on where the prefill starts), so it runs greedy.
        let sampling = |i: usize| {
            if spec_on {
                SamplingParams::greedy()
            } else {
                params(i as u64)
            }
        };

        let mut h = Harness::fixture(512, vec![0, 1], sched(spec_on));
        let mut hits = Vec::new();
        for (i, p) in prompts.iter().enumerate() {
            let id = i as u64;
            h.submit(request(
                id,
                p.clone(),
                sampling(i),
                5,
                CacheScope::Keyed(salt(1)),
            ));
            h.step();
            hits.push(h.eng.cached_prompt_tokens(id).unwrap());
            h.run();
        }
        assert_eq!(hits, [0, 0, 40, 40, 40], "speculative {spec_on}");
        let sum = h.check_all(spec_on);
        eprintln!("speculative {spec_on}: {sum:?}");

        // Cold: every prompt alone, nothing cached.
        let mut cold = Harness::fixture(512, vec![0, 1], sched(spec_on));
        for (i, p) in prompts.iter().enumerate() {
            cold.submit(request(
                i as u64,
                p.clone(),
                sampling(i),
                5,
                CacheScope::Private,
            ));
            cold.run();
        }
        for i in 0..prompts.len() as u64 {
            assert_eq!(h.outputs[&i], cold.outputs[&i], "request {i}");
        }
        // Another key never hits the branch.
        h.submit(request(
            9,
            prompts[4].clone(),
            sampling(4),
            5,
            CacheScope::Keyed(salt(2)),
        ));
        h.step();
        assert_eq!(h.eng.cached_prompt_tokens(9), Some(0));
        h.run();
        assert_eq!(h.outputs[&9], cold.outputs[&4]);
    }
}

#[test]
fn preemption_and_resume_reproduce_uninterrupted_outputs() {
    for spec_on in [false, true] {
        // 30 blocks of 4 positions per group: far less than 12 concurrent sequences need.
        let mut h = Harness::fixture(31, vec![0, 1], sched(spec_on));
        let mut rng = TestRng(7);
        for id in 0..12 {
            let prompt = rng.var_tokens(5, 15, VOCAB);
            let p = if spec_on && id % 2 == 0 {
                SamplingParams::greedy()
            } else {
                params(id)
            };
            let cache = if id % 3 == 0 {
                CacheScope::Private
            } else {
                CacheScope::Keyed(salt(id as u8))
            };
            h.submit(request(id, prompt, p, 18, cache));
        }
        h.run();
        let stats = h.eng.stats();
        assert!(stats.preemptions > 0, "workload must force preemption");
        let sum = h.check_all(spec_on);
        eprintln!("speculative {spec_on}: {stats:?} {sum:?}");
    }
}

fn zeros_since(h: &Harness, mark: usize) -> BTreeSet<(u32, u32)> {
    h.eng.executor().zero_log()[mark..]
        .iter()
        .map(|&(_, g, b)| (g, b))
        .collect()
}

#[test]
fn idle_ttl_eviction_zeroes_exactly_the_cached_blocks() {
    let mut cfg = sched(true);
    cfg.sweep_interval_ms = u64::MAX;
    let ttl = cfg.cache.idle_ttl_ms;
    let mut h = Harness::fixture(256, vec![0, 1], cfg);
    let prompt = TestRng(9).tokens(23, VOCAB);
    h.submit(request(
        1,
        prompt,
        SamplingParams::greedy(),
        6,
        CacheScope::Keyed(salt(3)),
    ));
    h.run();
    h.eng.sweep(h.now).unwrap();
    let cached = h.eng.kv().cache_blocks();
    assert!(!cached.is_empty());
    for &(g, b) in &cached {
        assert!(
            !h.eng.executor().block_is_zero(g, b),
            "cached block {g}/{b} holds KV"
        );
    }
    let finished_at = h.now - 1;
    let mark = h.eng.executor().zero_log().len();
    h.eng.sweep(finished_at + ttl - 1).unwrap();
    assert!(zeros_since(&h, mark).is_empty(), "nothing expires early");
    h.eng.sweep(finished_at + ttl).unwrap();
    assert_eq!(
        zeros_since(&h, mark),
        cached,
        "exactly the evicted blocks are zeroed"
    );
    for (g, b) in cached {
        assert!(h.eng.executor().block_is_zero(g, b));
    }
    h.check();
    h.check_all(true);
}

/// TTL eviction while other requests are running: the evicted conversation's blocks are
/// zeroed in the executor's pools, running sequences are unaffected, and a later request
/// under the expired salt recomputes instead of hitting.
#[test]
fn ttl_eviction_mid_workload() {
    let mut cfg = sched(true);
    cfg.cache.idle_ttl_ms = 100;
    cfg.sweep_interval_ms = 10;
    let mut h = Harness::fixture(256, vec![0, 1], cfg);
    let mut rng = TestRng(10);
    let convo = rng.tokens(21, VOCAB);
    h.submit(request(
        1,
        convo.clone(),
        params(1),
        5,
        CacheScope::Keyed(salt(9)),
    ));
    h.run();
    h.eng.sweep(h.now).unwrap();
    let cached = h.eng.kv().cache_blocks();
    assert!(!cached.is_empty());
    let mark = h.eng.executor().zero_log().len();
    // A long request runs across the expiry.
    h.submit(request(
        2,
        rng.tokens(17, VOCAB),
        params(2),
        40,
        CacheScope::Keyed(salt(8)),
    ));
    while h.eng.unfinished() > 0 {
        h.step();
        h.now += 7;
    }
    assert!(h.now > 100, "the workload spans the TTL");
    // Run the zeros the finished request's release queued.
    h.eng.sweep(h.now).unwrap();
    let zeroed = zeros_since(&h, mark);
    let in_use = h.eng.kv().sequence_blocks();
    for b in &cached {
        assert!(zeroed.contains(b), "evicted block {b:?} zeroed");
        // Unless the running request has since been handed the block, it is still zero.
        if !in_use.contains(b) && !h.eng.kv().cache_blocks().contains(b) {
            assert!(h.eng.executor().block_is_zero(b.0, b.1), "{b:?}");
        }
    }
    h.submit(request(3, convo, params(1), 5, CacheScope::Keyed(salt(9))));
    h.step();
    assert_eq!(h.eng.cached_prompt_tokens(3), Some(0));
    h.run();
    assert_eq!(h.outputs[&3], h.outputs[&1]);
    h.check_all(true);
}

#[test]
fn cancellation_releases_and_zeroes_everything() {
    let mut h = Harness::fixture(256, vec![0, 1], sched(true));
    let mut rng = TestRng(13);
    for id in 0..10 {
        let cache = if id % 2 == 0 {
            CacheScope::Keyed(salt(id as u8))
        } else {
            CacheScope::Private
        };
        h.submit(request(id, rng.tokens(12, VOCAB), params(id), 100, cache));
    }
    for _ in 0..4 {
        h.step();
        h.now += 1;
    }
    for id in 0..10 {
        let e = h.eng.cancel(id, h.now);
        assert!(e.is_none_or(|e| e.finish == Some(FinishReason::Cancelled)));
    }
    assert_eq!(h.eng.unfinished(), 0);
    h.eng.sweep(h.now + 100 * 3_600_000).unwrap();
    h.check();
    assert!(h.eng.kv().sequence_blocks().is_empty());
    assert_eq!(h.eng.kv().cache_entries(), 0);
    for (g, gs) in h.spec.kv_groups.iter().enumerate() {
        for b in 1..gs.num_blocks {
            assert!(h.eng.executor().block_is_zero(g as u32, b), "{g}/{b}");
        }
    }
    h.check_all(true);
}

#[test]
fn pre_norm_chaining_is_selectable_and_exact() {
    let m = fixture();
    let prompts: Vec<Vec<u32>> = {
        let mut rng = TestRng(14);
        (0..4).map(|_| rng.var_tokens(4, 16, VOCAB)).collect()
    };
    let mut drafter = Vec::new();
    for mode in [MtpHidden::Normed, MtpHidden::PreNorm] {
        let mut cfg = exec_config(256, vec![0, 1]);
        cfg.mtp_hidden = mode;
        let mut h = Harness::new(m.clone(), cfg, sched(true));
        for (i, p) in prompts.iter().enumerate() {
            h.submit(request(
                i as u64,
                p.clone(),
                SamplingParams::greedy(),
                8,
                CacheScope::Private,
            ));
        }
        h.run();
        drafter.push(
            h.records
                .iter()
                .flat_map(|r| r.drafter_logits.iter().map(|(_, _, l)| l.clone()))
                .collect::<Vec<_>>(),
        );
        h.check_all(true);
    }
    assert_ne!(
        drafter[0], drafter[1],
        "the two chainings are different functions"
    );
}

#[test]
fn unfolded_value_scale_is_exact_too() {
    let m = std::sync::Arc::new(load_fixture(&LoadOptions {
        fold_value_scale: false,
        ..LoadOptions::default()
    }));
    assert!(!m.weights.value_scale_folded);
    let mut h = Harness::new(m, exec_config(256, vec![0, 1]), sched(true));
    let mut rng = TestRng(15);
    for id in 0..4 {
        h.submit(request(
            id,
            rng.var_tokens(3, 20, VOCAB),
            params(id),
            8,
            CacheScope::Private,
        ));
    }
    h.run();
    h.check_all(true);
}

#[test]
fn padded_batches_change_nothing() {
    let m = fixture();
    let mut rng = TestRng(16);
    let prompts: Vec<Vec<u32>> = (0..5).map(|_| rng.var_tokens(2, 20, VOCAB)).collect();
    let mut runs = Vec::new();
    for pad in [false, true] {
        let mut cfg = exec_config(256, vec![0, 1]);
        cfg.pad_batches = pad;
        let mut h = Harness::new(m.clone(), cfg, sched(true));
        for (i, p) in prompts.iter().enumerate() {
            h.submit(request(
                i as u64,
                p.clone(),
                params(i as u64),
                7,
                CacheScope::Private,
            ));
        }
        h.run();
        runs.push(h.records.clone());
        h.check_all(true);
    }
    assert_eq!(runs[0], runs[1]);
}

/// Half the time a shared head plus a random tail, otherwise random.
fn headed(rng: &mut TestRng, head: &[u32]) -> Vec<u32> {
    let mut t = if rng.chance(0.5) {
        head.to_vec()
    } else {
        Vec::new()
    };
    t.extend(rng.var_tokens(1, 14, VOCAB));
    t
}

/// The core's randomized workload over the real executor: multi-turn keyed conversations
/// (continuations and regenerations), private requests, mixed sampling, speculation on or
/// off, cancellations, clock jumps past the idle TTL, and pools small enough to force
/// eviction and preemption. Every step checks the KV manager's invariants and that every
/// free block is zero in the executor's pools or has a zero queued; every row is checked
/// against the dense oracle; hits never exceed what the same salt computed.
#[test]
fn randomized_workloads_match_the_dense_oracle() {
    let mut totals = Checked::default();
    for seed in 0..6u64 {
        let mut rng = TestRng(100 + seed);
        let spec_on = seed % 2 == 0;
        let blocks = 40 + rng.below(40) as u32;
        let mut cfg = sched(spec_on);
        cfg.max_prefill_chunk = 1 + rng.below(20) as u32;
        cfg.max_batched_tokens = 24 + rng.below(60) as u32;
        cfg.cache.idle_ttl_ms = 500;
        cfg.cache.max_age_ms = 2000;
        cfg.sweep_interval_ms = 50;
        let depths = [vec![0], vec![0, 1], vec![0, 1, 0]][seed as usize % 3].clone();
        let mut h = Harness::fixture(blocks, depths, cfg);

        let mut convo: HashMap<u8, Vec<u32>> = HashMap::new();
        let mut history: HashMap<u8, Vec<Vec<u32>>> = HashMap::new();
        let mut live: HashMap<u64, (Vec<u32>, Option<u8>)> = HashMap::new();
        let mut hit_checked = std::collections::HashSet::new();
        let head = rng.tokens(9, VOCAB);
        let mut hits_seen = 0usize;
        let mut cancelled = 0usize;
        let mut next_id = 0u64;

        for _round in 0..120 {
            if rng.chance(0.35) && live.len() < 12 {
                let id = next_id;
                next_id += 1;
                let p = if spec_on && rng.chance(0.5) {
                    SamplingParams::greedy()
                } else {
                    params(id)
                };
                let max = 1 + rng.below(12) as u32;
                let (prompt, key) = if rng.chance(0.25) {
                    (headed(&mut rng, &head), None)
                } else {
                    let s = rng.below(4) as u8;
                    let prompt = match convo.get(&s) {
                        Some(t) if rng.chance(0.3) => t.clone(),
                        Some(t) => {
                            let mut t = t.clone();
                            t.extend(rng.var_tokens(1, 8, VOCAB));
                            t
                        }
                        None => headed(&mut rng, &head),
                    };
                    (prompt, Some(s))
                };
                if prompt.len() as u32 > blocks * 4 / 2 || prompt.len() > 60 {
                    continue;
                }
                let cache = match key {
                    Some(s) => CacheScope::Keyed(salt(s + 1)),
                    None => CacheScope::Private,
                };
                live.insert(id, (prompt.clone(), key));
                h.submit(request(id, prompt, p, max, cache));
            }
            if rng.chance(0.03) && !live.is_empty() {
                let ids: Vec<u64> = live.keys().copied().collect();
                let victim = ids[rng.below(ids.len() as u64) as usize];
                if h.eng.cancel(victim, h.now).is_some() {
                    cancelled += 1;
                    h.finished.insert(victim, FinishReason::Cancelled);
                    let (prompt, key) = live.remove(&victim).unwrap();
                    if let Some(k) = key {
                        let mut t = prompt;
                        t.extend(&h.outputs[&victim]);
                        history.entry(k).or_default().push(t);
                    }
                }
            }
            let before: Vec<u64> = live.keys().copied().collect();
            h.step();
            for id in before {
                let c = h
                    .eng
                    .cached_prompt_tokens(id)
                    .or_else(|| h.cached.get(&id).copied());
                let Some(c) = c else { continue };
                if !hit_checked.insert(id) {
                    continue;
                }
                let (prompt, key) = &live[&id];
                let allowed = match key {
                    None => 0,
                    Some(s) => {
                        let mut sources: Vec<Vec<u32>> =
                            history.get(s).cloned().unwrap_or_default();
                        for (j, (pj, kj)) in &live {
                            if *j != id && kj == key {
                                let mut t = pj.clone();
                                t.extend(&h.outputs[j]);
                                sources.push(t);
                            }
                        }
                        sources
                            .iter()
                            .map(|t| t.iter().zip(prompt).take_while(|(a, b)| a == b).count())
                            .max()
                            .unwrap_or(0)
                    }
                };
                assert!(
                    c as usize <= allowed,
                    "seed {seed}: hit {c} beyond any same-salt prefix {allowed}"
                );
                hits_seen += (c > 0) as usize;
            }
            let done: Vec<u64> = live
                .keys()
                .copied()
                .filter(|id| h.finished.contains_key(id))
                .collect();
            for id in done {
                let (prompt, key) = live.remove(&id).unwrap();
                if let Some(s) = key {
                    let mut t = prompt.clone();
                    t.extend(&h.outputs[&id]);
                    history.entry(s).or_default().push(t.clone());
                    convo.insert(s, t);
                }
            }
            h.now += if rng.chance(0.03) {
                600
            } else {
                1 + rng.below(5)
            };
        }
        h.run();
        assert!(
            hits_seen > 0,
            "seed {seed}: workload produced no cache hits"
        );
        let sum = h.check_all(spec_on);
        eprintln!(
            "seed {seed}: blocks {blocks} requests {next_id} cancelled {cancelled} hits \
             {hits_seen} stats {:?} checked {sum:?}",
            h.eng.stats(),
        );
        totals.records += sum.records;
        totals.target_rows += sum.target_rows;
        totals.drafter_rows += sum.drafter_rows;
        totals.drafted += sum.drafted;
        totals.accepted += sum.accepted;
        // Drain: everything expires, everything is freed and zeroed.
        h.eng.sweep(h.now + 10_000).unwrap();
        h.check();
        assert_eq!(h.eng.kv().cache_entries(), 0);
        for (g, gs) in h.spec.kv_groups.iter().enumerate() {
            assert_eq!(
                h.eng.kv().free_blocks(g),
                gs.num_blocks as usize - 1,
                "seed {seed} group {g} leaked"
            );
            for b in 1..gs.num_blocks {
                assert!(h.eng.executor().block_is_zero(g as u32, b));
            }
        }
    }
    eprintln!("total: {totals:?}");
}

/// Padded head rows never reach a token: with the head's rows past 200 rigged to win,
/// target sampling, drafting (greedy and sampled, two depths) and acceptance still only
/// return ids below the sampleable vocabulary, every row stays bit-exact against the dense
/// oracle under the same limit, and the rig really does win unmasked.
#[test]
fn padded_head_rows_are_never_sampled_drafted_or_accepted() {
    use eidola_engine::sampling::{Logits, argmax};

    const SAMPLEABLE: u32 = 200;
    let m = padded_head_fixture(SAMPLEABLE as usize, 30.0);
    for spec_on in [false, true] {
        let mut cfg = exec_config(256, vec![0, 1]);
        cfg.sampleable_vocab_size = SAMPLEABLE;
        let mut h = Harness::new(m.clone(), cfg, sched(spec_on));
        let mut rng = TestRng(61);
        for id in 0..6 {
            let prompt = rng.var_tokens(2, 12, SAMPLEABLE);
            h.submit(request(id, prompt, params(id), 8, CacheScope::Private));
        }
        h.run();
        for id in 0..6 {
            let out = &h.outputs[&id];
            assert!(out.iter().all(|&t| t < SAMPLEABLE), "request {id}: {out:?}");
        }
        let records = h.records.clone();
        let mut padded_wins = 0;
        for r in &records {
            for (_, l) in &r.target_logits {
                padded_wins += (argmax(Logits::new(l, VOCAB)) >= SAMPLEABLE) as usize;
            }
            for (_, _, l) in &r.drafter_logits {
                padded_wins += (argmax(Logits::new(l, VOCAB)) >= SAMPLEABLE) as usize;
            }
            assert!(r.drafts.iter().all(|&d| d < SAMPLEABLE), "{:?}", r.drafts);
        }
        assert!(
            padded_wins > records.len(),
            "padded rows must win unmasked ({padded_wins})"
        );
        let sum = h.check_all(spec_on);
        assert_eq!(sum.drafted > 0, spec_on);
    }
}
