//! End-to-end simulation of the scheduler, KV manager and prefix cache over the mock
//! executor: outputs against the dense reference, chunking, preemption, speculation,
//! salt isolation, expiry and zeroing.

mod common;

use std::collections::{BTreeSet, HashMap};

use common::*;
use eidola_engine::engine::{CacheScope, FinishReason};
use eidola_engine::mock::{MockConfig, mimo_like_spec};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::AttentionKind;

fn params(i: u64) -> SamplingParams {
    match i % 3 {
        0 => SamplingParams::greedy(),
        1 => SamplingParams::random(1.0, 1000 + i).unwrap(),
        _ => SamplingParams::new(0.7, 8, 0.9, 0.05, 2000 + i).unwrap(),
    }
}

#[test]
fn single_requests_match_the_dense_reference() {
    for spec_on in [false, true] {
        let mut h = Harness::new(small_spec(256), sched(spec_on), MockConfig::default());
        let mut rng = TestRng(1);
        let mut cases = Vec::new();
        for id in 0..12 {
            let prompt = rng.var_tokens(1, 40, 32);
            let p = params(id);
            let max = 1 + rng.below(30) as u32;
            cases.push((id, prompt.clone(), p, max));
            h.submit(request(id, prompt, p, max, CacheScope::Private));
        }
        h.run();
        for (id, prompt, p, max) in cases {
            if spec_on && !p.is_greedy() {
                continue; // equal in distribution only; see the statistical test
            }
            assert_eq!(
                h.outputs[&id],
                h.expected(&prompt, &p, max, &[]),
                "request {id}"
            );
            assert_eq!(h.finished[&id], FinishReason::Length);
        }
    }
}

#[test]
fn chunked_prefill_is_identical_to_unchunked() {
    let mut rng = TestRng(2);
    let prompts: Vec<Vec<u32>> = (0..6).map(|_| rng.var_tokens(30, 70, 32)).collect();
    let mut results = Vec::new();
    for chunk in [1u32, 3, 4, 7, 16, 1000] {
        let mut cfg = sched(false);
        cfg.max_prefill_chunk = chunk;
        cfg.max_batched_tokens = 64.max(chunk.min(256));
        let mut h = Harness::new(small_spec(512), cfg, MockConfig::default());
        for (i, p) in prompts.iter().enumerate() {
            h.submit(request(
                i as u64,
                p.clone(),
                params(i as u64),
                12,
                CacheScope::Private,
            ));
        }
        h.run();
        let mut out: Vec<Vec<u32>> = Vec::new();
        for i in 0..prompts.len() as u64 {
            out.push(h.outputs[&i].clone());
            assert_eq!(
                h.outputs[&i],
                h.expected(&prompts[i as usize], &params(i), 12, &[])
            );
        }
        results.push(out);
    }
    for r in &results[1..] {
        assert_eq!(r, &results[0]);
    }
}

#[test]
fn stop_tokens_and_eos_end_generation() {
    let mut cfg = sched(true);
    cfg.eos_token_ids = vec![5];
    let mut h = Harness::new(small_spec(256), cfg, MockConfig::default());
    let mut rng = TestRng(3);
    let mut cases = Vec::new();
    for id in 0..40 {
        let prompt = rng.var_tokens(3, 10, 32);
        let mut req = request(
            id,
            prompt.clone(),
            SamplingParams::greedy(),
            60,
            CacheScope::Private,
        );
        req.stop_token_ids = vec![7, 9];
        cases.push(prompt);
        h.submit(req);
    }
    h.run();
    let mut stopped = 0;
    for (id, prompt) in cases.iter().enumerate() {
        let exp = h.expected(prompt, &SamplingParams::greedy(), 60, &[7, 9]);
        assert_eq!(h.outputs[&(id as u64)], exp);
        if h.finished[&(id as u64)] == FinishReason::Stop {
            stopped += 1;
            assert!([5, 7, 9].contains(exp.last().unwrap()));
        }
    }
    assert!(stopped > 0);
}

#[test]
fn greedy_speculation_is_exact_at_every_agreement_rate() {
    for agreement in [0.0, 0.3, 0.8, 1.0] {
        let cfg = MockConfig {
            draft_agreement: agreement,
            ..MockConfig::default()
        };
        let mut h = Harness::new(small_spec(256), sched(true), cfg);
        let mut rng = TestRng(4);
        let mut cases = Vec::new();
        for id in 0..16 {
            let prompt = rng.var_tokens(2, 30, 32);
            cases.push(prompt.clone());
            h.submit(request(
                id,
                prompt,
                SamplingParams::greedy(),
                40,
                CacheScope::Private,
            ));
        }
        h.run();
        for (id, prompt) in cases.iter().enumerate() {
            assert_eq!(
                h.outputs[&(id as u64)],
                h.expected(prompt, &SamplingParams::greedy(), 40, &[]),
                "agreement {agreement}"
            );
        }
        let s = h.eng.stats();
        assert!(s.drafted > 0);
        if agreement == 1.0 {
            assert_eq!(
                s.accepted, s.drafted,
                "a perfect drafter is always accepted"
            );
        }
    }
}

/// Speculative sampling must produce the same output distribution as plain sampling.
/// Joint distribution of the first three tokens (vocab 4 ⇒ 64 cells) over 20k seeds each,
/// compared with a two-sample chi-square test at p = 0.001 (critical value 103.4, df 63).
#[test]
fn speculative_sampling_preserves_the_distribution() {
    let spec = mimo_like_spec(4, 4, 6, 10, 256, 32, 3);
    let n = 20_000u64;
    let prompt = vec![1, 2, 3, 0, 1];
    for (agreement, temperature) in [(0.5, 1.0f32), (0.0, 0.8), (0.9, 1.3)] {
        let cfg = MockConfig {
            draft_agreement: agreement,
            logit_spread: 3.0,
            ..MockConfig::default()
        };
        let mut plain = [0u64; 64];
        let mut specd = [0u64; 64];
        let mut h = Harness::new(spec.clone(), sched(true), cfg);
        h.checks = false; // statistics only; invariants are covered elsewhere
        for seed in 0..n {
            let p = SamplingParams::random(temperature, seed).unwrap();
            let r = h.expected(
                &prompt,
                &SamplingParams::random(temperature, seed + 7_777_777).unwrap(),
                3,
                &[],
            );
            plain[(r[0] * 16 + r[1] * 4 + r[2]) as usize] += 1;
            h.submit(request(seed, prompt.clone(), p, 3, CacheScope::Private));
            if h.eng.unfinished() >= 32 {
                while h.eng.unfinished() > 0 {
                    h.step();
                    h.now += 1;
                }
            }
        }
        h.run();
        for seed in 0..n {
            let r = &h.outputs[&seed];
            specd[(r[0] * 16 + r[1] * 4 + r[2]) as usize] += 1;
        }
        let mut chi = 0.0;
        for c in 0..64 {
            let (a, b) = (plain[c] as f64, specd[c] as f64);
            if a + b > 0.0 {
                chi += (a - b) * (a - b) / (a + b);
            }
        }
        let s = h.eng.stats();
        assert!(s.accepted > 0 && s.accepted < s.drafted || agreement == 0.0);
        assert!(chi < 103.4, "agreement {agreement}: chi-square {chi}");
    }
}

#[test]
fn preemption_and_resume_reproduce_uninterrupted_outputs() {
    for spec_on in [false, true] {
        // 40 blocks of 4 tokens per group: far less than 24 concurrent sequences need.
        let mut h = Harness::new(small_spec(41), sched(spec_on), MockConfig::default());
        let mut rng = TestRng(5);
        let mut cases = Vec::new();
        for id in 0..24 {
            let prompt = rng.var_tokens(5, 25, 32);
            let p = if spec_on {
                SamplingParams::greedy()
            } else {
                params(id)
            };
            cases.push((prompt.clone(), p));
            h.submit(request(
                id,
                prompt,
                p,
                50,
                CacheScope::Keyed(salt(id as u8)),
            ));
        }
        h.run();
        assert!(
            h.eng.stats().preemptions > 0,
            "workload must force preemption"
        );
        for (id, (prompt, p)) in cases.iter().enumerate() {
            assert_eq!(
                h.outputs[&(id as u64)],
                h.expected(prompt, p, 50, &[]),
                "request {id}"
            );
        }
    }
}

#[test]
fn salts_isolate_and_equal_salts_share() {
    let mut h = Harness::new(small_spec(512), sched(false), MockConfig::default());
    let mut rng = TestRng(6);
    let prompt = rng.tokens(40, 32);
    let g = SamplingParams::greedy();
    let mut id = 0;
    let mut run = |h: &mut Harness, prompt: &[u32], cache: CacheScope| {
        id += 1;
        h.submit(request(id, prompt.to_vec(), g, 4, cache));
        h.step();
        let cached = h.eng.cached_prompt_tokens(id).unwrap();
        h.run();
        assert_eq!(h.outputs[&id], h.expected(prompt, &g, 4, &[]));
        cached
    };
    assert_eq!(run(&mut h, &prompt, CacheScope::Keyed(salt(1))), 0);
    // Same salt, same prompt: the regeneration hit point (last full block before the
    // final token) is retained.
    assert_eq!(run(&mut h, &prompt, CacheScope::Keyed(salt(1))), 36);
    // Different salt: nothing.
    assert_eq!(run(&mut h, &prompt, CacheScope::Keyed(salt(2))), 0);
    // No key: nothing, and again nothing.
    assert_eq!(run(&mut h, &prompt, CacheScope::Private), 0);
    assert_eq!(run(&mut h, &prompt, CacheScope::Private), 0);
    // Continuation under the original salt: prompt + output + more resumes at the end of
    // the previous sequence.
    let mut cont = prompt.clone();
    cont.extend(&h.outputs[&1]);
    cont.extend(rng.tokens(9, 32));
    let cached = run(&mut h, &cont, CacheScope::Keyed(salt(1)));
    // The previous sequence wrote KV for 40 + 4 - 1 positions (its last token never
    // entered KV); its last sealed boundary is the block boundary at or below that.
    let computed = 40 + 4 - 1;
    assert_eq!(
        cached,
        computed - computed % 4,
        "continuation resumes at the last sealed block"
    );
}

#[test]
fn private_requests_leave_nothing_cached() {
    let mut h = Harness::new(small_spec(512), sched(true), MockConfig::default());
    let mut rng = TestRng(7);
    for id in 0..10 {
        h.submit(request(
            id,
            rng.tokens(30, 32),
            params(id),
            20,
            CacheScope::Private,
        ));
    }
    h.run();
    assert_eq!(h.eng.kv().cache_entries(), 0);
    assert!(h.eng.kv().cache_blocks().is_empty());
}

/// A hit is only taken where every sliding-window group's window is present; a prefix
/// whose full-attention blocks are cached but whose window was not retained is recomputed.
#[test]
fn sliding_window_hits_require_the_window() {
    let mut rng = TestRng(8);
    let base = rng.tokens(40, 32);
    let mut diverging = base[..20].to_vec();
    diverging.extend(rng.tokens(20, 32).iter().map(|t| (t + 1) % 32));
    if diverging[20] == base[20] {
        diverging[20] = (base[20] + 1) % 32;
    }
    let g = SamplingParams::greedy();

    // MiMo-like (full + sliding + drafter): block 5 is not yet a retained hit point (the
    // diverging request keeps it as its branch, for the next one).
    let mut h = Harness::new(small_spec(512), sched(false), MockConfig::default());
    h.submit(request(1, base.clone(), g, 4, CacheScope::Keyed(salt(1))));
    h.run();
    h.submit(request(
        2,
        diverging.clone(),
        g,
        4,
        CacheScope::Keyed(salt(1)),
    ));
    h.step();
    assert_eq!(h.eng.cached_prompt_tokens(2), Some(0));
    h.run();
    assert_eq!(h.outputs[&2], h.expected(&diverging, &g, 4, &[]));

    // Full attention only: the same five shared blocks are a valid hit.
    let mut spec = small_spec(512);
    spec.kv_groups
        .retain(|k| k.attention == AttentionKind::Full);
    let mut h = Harness::new(spec, sched(false), MockConfig::default());
    h.submit(request(1, base, g, 4, CacheScope::Keyed(salt(1))));
    h.run();
    h.submit(request(
        2,
        diverging.clone(),
        g,
        4,
        CacheScope::Keyed(salt(1)),
    ));
    h.step();
    assert_eq!(h.eng.cached_prompt_tokens(2), Some(20));
    h.run();
    assert_eq!(h.outputs[&2], h.expected(&diverging, &g, 4, &[]));
}

/// A shared prefix (a system prompt) under one key, followed by a different turn each
/// time: tails that differ from their first token.
fn branching_prompts(rng: &mut TestRng, prefix: &[u32], n: usize) -> Vec<Vec<u32>> {
    (0..n)
        .map(|i| {
            let mut t = prefix.to_vec();
            t.push(i as u32);
            t.extend(rng.tokens(8, 32));
            t
        })
        .collect()
}

/// Requests that share a long prefix under one key and then diverge hit at the branch:
/// the first divergent request finds the prefix's full-attention blocks but no window
/// there and recomputes, keeping the window it computes at the divergence, and every
/// later one hits. Other keys and private requests never hit it.
#[test]
fn divergent_turns_after_a_shared_prefix_hit_at_the_branch() {
    for spec_on in [false, true] {
        let mut h = Harness::new(small_spec(512), sched(spec_on), MockConfig::default());
        let mut rng = TestRng(20);
        // Ten blocks: several sliding windows (6) and drafter windows (10).
        let prefix = rng.tokens(40, 32);
        let prompts = branching_prompts(&mut rng, &prefix, 6);
        let g = SamplingParams::greedy();
        let mut id = 0;
        let mut run = |h: &mut Harness, prompt: &[u32], cache: CacheScope| {
            id += 1;
            h.submit(request(id, prompt.to_vec(), g, 4, cache));
            h.step();
            let cached = h.eng.cached_prompt_tokens(id).unwrap();
            h.run();
            assert_eq!(
                h.outputs[&id],
                h.expected(prompt, &g, 4, &[]),
                "request {id}"
            );
            cached
        };
        let hits: Vec<u32> = prompts
            .iter()
            .map(|p| run(&mut h, p, CacheScope::Keyed(salt(1))))
            .collect();
        assert_eq!(hits, [0, 0, 40, 40, 40, 40], "speculative {spec_on}");
        // Another key sees none of it, and builds its own branch the same way.
        let hits: Vec<u32> = prompts[..3]
            .iter()
            .map(|p| run(&mut h, p, CacheScope::Keyed(salt(2))))
            .collect();
        assert_eq!(hits, [0, 0, 40]);
        assert_eq!(run(&mut h, &prompts[3], CacheScope::Private), 0);
        assert_eq!(run(&mut h, &prompts[3], CacheScope::Private), 0);
    }
}

/// The branch window is attached as soon as the branching request seals the boundary, so
/// a request arriving while it is still decoding already hits there.
#[test]
fn a_branch_serves_requests_that_arrive_while_it_runs() {
    let mut h = Harness::new(small_spec(512), sched(false), MockConfig::default());
    let mut rng = TestRng(21);
    let prefix = rng.tokens(40, 32);
    let prompts = branching_prompts(&mut rng, &prefix, 3);
    let g = SamplingParams::greedy();
    let key = || CacheScope::Keyed(salt(1));
    h.submit(request(1, prompts[0].clone(), g, 4, key()));
    h.run();
    h.submit(request(2, prompts[1].clone(), g, 60, key()));
    h.step();
    h.now += 1;
    assert_eq!(h.eng.cached_prompt_tokens(2), Some(0));
    h.submit(request(3, prompts[2].clone(), g, 4, key()));
    h.step();
    h.now += 1;
    assert!(
        !h.finished.contains_key(&2),
        "the branching request is still running"
    );
    assert_eq!(h.eng.cached_prompt_tokens(3), Some(40));
    h.run();
    assert_eq!(h.outputs[&2], h.expected(&prompts[1], &g, 60, &[]));
    assert_eq!(h.outputs[&3], h.expected(&prompts[2], &g, 4, &[]));
}

/// Branch windows are ordinary cache blocks: they expire with their entries on the idle
/// TTL and are zeroed exactly like the rest, and an expired branch is never hit.
#[test]
fn branch_windows_expire_and_are_zeroed_like_any_entry() {
    let mut cfg = sched(false);
    cfg.sweep_interval_ms = u64::MAX;
    let ttl = cfg.cache.idle_ttl_ms;
    let mut h = Harness::new(small_spec(512), cfg, MockConfig::default());
    let mut rng = TestRng(22);
    let prefix = rng.tokens(40, 32);
    let prompts = branching_prompts(&mut rng, &prefix, 4);
    let g = SamplingParams::greedy();
    for (i, p) in prompts[..2].iter().enumerate() {
        h.submit(request(
            i as u64,
            p.clone(),
            g,
            4,
            CacheScope::Keyed(salt(1)),
        ));
        h.run();
    }
    h.eng.sweep(h.now).unwrap();
    let cached = h.eng.kv().cache_blocks();
    // Each request keeps its regeneration (block 12) and continuation (block 13) windows:
    // blocks 10..13 of the sliding group (window 6) and 9..13 of the drafter group
    // (window 10), where block 9 is the shared prefix's, held once. The second also keeps
    // the branch at block 10 on the shared entries: blocks 8..10 and 7..10, of which only
    // 7 and 8 of the drafter group were not already held.
    let count = |g: u32| cached.iter().filter(|(gr, _)| *gr == g).count();
    assert_eq!((count(1), count(2)), (3 + 3 + 2, 4 + 3 + 2));
    let finished_at = h.now - 1;
    let mark = h.eng.executor().zero_log().len();
    h.eng.sweep(finished_at + ttl).unwrap();
    assert_eq!(
        zeros_since(&h, mark),
        cached,
        "exactly the cached blocks are zeroed"
    );
    assert_eq!(h.eng.kv().cache_entries(), 0);
    for (gr, b) in cached {
        assert!(h.eng.executor().block_is_zero(gr, b));
    }
    h.now = finished_at + ttl;
    h.submit(request(
        9,
        prompts[2].clone(),
        g,
        4,
        CacheScope::Keyed(salt(1)),
    ));
    h.step();
    assert_eq!(h.eng.cached_prompt_tokens(9), Some(0));
    h.run();
    assert_eq!(h.outputs[&9], h.expected(&prompts[2], &g, 4, &[]));
    h.check();
}

/// Branch retention lives in the same pools as everything else: with room for little
/// more than one request, a stream of divergent turns keeps hitting at the branch while
/// older entries are evicted, and a request needing nearly the whole pool still runs.
#[test]
fn branch_retention_stays_within_the_pool() {
    // 29 allocatable blocks per group; one request needs 14 in the full group.
    let mut h = Harness::new(small_spec(30), sched(false), MockConfig::default());
    let mut rng = TestRng(23);
    let prefix = rng.tokens(40, 32);
    let prompts = branching_prompts(&mut rng, &prefix, 12);
    let g = SamplingParams::greedy();
    let mut hits = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        h.submit(request(
            i as u64,
            p.clone(),
            g,
            4,
            CacheScope::Keyed(salt(1)),
        ));
        h.step();
        hits.push(h.eng.cached_prompt_tokens(i as u64).unwrap());
        h.run();
        assert_eq!(h.outputs[&(i as u64)], h.expected(p, &g, 4, &[]));
    }
    assert_eq!(hits[..2], [0, 0]);
    assert!(hits[2..].iter().all(|&c| c == 40), "{hits:?}");
    assert_eq!(h.eng.stats().preemptions, 0);
    // 108 prompt tokens plus 3 decoded positions: 28 blocks per group.
    let big = rng.tokens(108, 32);
    h.submit(request(100, big.clone(), g, 4, CacheScope::Private));
    h.run();
    assert_eq!(h.outputs[&100], h.expected(&big, &g, 4, &[]));
    h.eng.sweep(h.now + 100 * 3_600_000).unwrap();
    assert_eq!(h.eng.kv().cache_entries(), 0);
    h.check();
}

fn zeros_since(h: &Harness, mark: usize) -> BTreeSet<(u32, u32)> {
    h.eng.executor().zero_log()[mark..]
        .iter()
        .map(|&(_, g, b)| (g, b))
        .collect()
}

#[test]
fn idle_ttl_evicts_and_zeroes_exactly_the_cached_blocks() {
    let mut cfg = sched(false);
    cfg.sweep_interval_ms = u64::MAX;
    let ttl = cfg.cache.idle_ttl_ms;
    let mut h = Harness::new(small_spec(512), cfg, MockConfig::default());
    let mut rng = TestRng(9);
    let prompt = rng.tokens(37, 32);
    let g = SamplingParams::greedy();
    h.submit(request(
        1,
        prompt.clone(),
        g,
        10,
        CacheScope::Keyed(salt(3)),
    ));
    h.run();
    // Flush the zeros from the sequence's own release.
    h.eng.sweep(h.now).unwrap();
    let cached = h.eng.kv().cache_blocks();
    assert!(!cached.is_empty());
    let finished_at = h.now - 1;

    let mark = h.eng.executor().zero_log().len();
    h.eng.sweep(finished_at + ttl - 1).unwrap();
    assert!(zeros_since(&h, mark).is_empty(), "nothing expires early");
    assert_eq!(h.eng.kv().cache_blocks(), cached);

    let mark = h.eng.executor().zero_log().len();
    let steps_before = h.eng.executor().steps();
    h.eng.sweep(finished_at + ttl).unwrap();
    assert_eq!(
        h.eng.executor().steps(),
        steps_before + 1,
        "zeroing runs without traffic"
    );
    assert_eq!(
        zeros_since(&h, mark),
        cached,
        "exactly the evicted blocks are zeroed"
    );
    assert_eq!(h.eng.kv().cache_entries(), 0);
    for (gr, b) in cached {
        assert!(h.eng.executor().block_is_zero(gr, b));
    }
    h.check();
}

#[test]
fn expired_entries_are_never_hit_even_before_a_sweep() {
    let mut cfg = sched(false);
    cfg.sweep_interval_ms = u64::MAX;
    let ttl = cfg.cache.idle_ttl_ms;
    let mut h = Harness::new(small_spec(512), cfg, MockConfig::default());
    let prompt = TestRng(10).tokens(30, 32);
    let g = SamplingParams::greedy();
    h.submit(request(1, prompt.clone(), g, 3, CacheScope::Keyed(salt(4))));
    h.run();
    h.now += ttl;
    h.submit(request(2, prompt.clone(), g, 3, CacheScope::Keyed(salt(4))));
    h.step();
    assert_eq!(h.eng.cached_prompt_tokens(2), Some(0));
    h.run();
    assert_eq!(h.outputs[&2], h.outputs[&1]);
}

#[test]
fn max_age_evicts_even_entries_kept_warm() {
    let mut cfg = sched(false);
    cfg.sweep_interval_ms = u64::MAX;
    let (ttl, max_age) = (cfg.cache.idle_ttl_ms, cfg.cache.max_age_ms);
    let mut h = Harness::new(small_spec(512), cfg, MockConfig::default());
    let prompt = TestRng(11).tokens(30, 32);
    let g = SamplingParams::greedy();
    h.submit(request(1, prompt.clone(), g, 3, CacheScope::Keyed(salt(5))));
    h.run();
    let created = 0;
    // Keep it warm: a regeneration every half TTL, each one a hit.
    let mut id = 2;
    while h.now + ttl / 2 < created + max_age {
        h.now += ttl / 2;
        h.submit(request(
            id,
            prompt.clone(),
            g,
            3,
            CacheScope::Keyed(salt(5)),
        ));
        h.step();
        assert_eq!(h.eng.cached_prompt_tokens(id), Some(28), "warm hit");
        h.run();
        id += 1;
    }
    h.eng.sweep(h.now).unwrap();
    let cached = h.eng.kv().cache_blocks();
    let mark = h.eng.executor().zero_log().len();
    h.eng.sweep(created + max_age).unwrap();
    assert_eq!(zeros_since(&h, mark), cached);
    assert_eq!(h.eng.kv().cache_entries(), 0);
    h.check();
}

#[test]
fn an_entry_in_use_past_max_age_is_detached_and_zeroed_on_release() {
    let mut cfg = sched(false);
    cfg.sweep_interval_ms = u64::MAX;
    let max_age = cfg.cache.max_age_ms;
    let mut h = Harness::new(small_spec(512), cfg, MockConfig::default());
    let prompt = TestRng(12).tokens(30, 32);
    let g = SamplingParams::greedy();
    h.submit(request(1, prompt.clone(), g, 3, CacheScope::Keyed(salt(6))));
    h.run();
    // A long-running continuation holds the old entries past their max age.
    h.submit(request(
        2,
        prompt.clone(),
        g,
        200,
        CacheScope::Keyed(salt(6)),
    ));
    h.step();
    assert!(h.eng.cached_prompt_tokens(2).unwrap() > 0);
    h.now = max_age;
    h.eng.sweep(h.now).unwrap();
    h.check();
    h.run();
    assert_eq!(h.outputs[&2], h.expected(&prompt, &g, 200, &[]));
    // Nothing from before the max age survives as a hit.
    h.submit(request(3, prompt.clone(), g, 3, CacheScope::Keyed(salt(6))));
    h.step();
    let hit = h.eng.cached_prompt_tokens(3).unwrap();
    assert!(
        hit == 0 || hit == 28,
        "only entries re-sealed after detachment may hit"
    );
    h.run();
    h.eng.sweep(h.now + 10 * max_age).unwrap();
    assert_eq!(h.eng.kv().cache_entries(), 0);
    h.check();
}

#[test]
fn cancel_releases_everything() {
    let mut h = Harness::new(small_spec(256), sched(true), MockConfig::default());
    let mut rng = TestRng(13);
    for id in 0..20 {
        let cache = if id % 2 == 0 {
            CacheScope::Keyed(salt(id as u8))
        } else {
            CacheScope::Private
        };
        h.submit(request(id, rng.tokens(20, 32), params(id), 100, cache));
    }
    for _ in 0..5 {
        h.step();
        h.now += 1;
    }
    for id in 0..20 {
        let e = h.eng.cancel(id, h.now);
        assert!(e.is_none_or(|e| e.finish == Some(FinishReason::Cancelled)));
    }
    assert_eq!(h.eng.unfinished(), 0);
    h.check();
    h.eng.sweep(h.now + 100 * 3_600_000).unwrap();
    h.check();
    assert!(h.eng.kv().sequence_blocks().is_empty());
    assert_eq!(h.eng.kv().cache_entries(), 0);
}

#[test]
fn decode_steps_use_the_smallest_fitting_bucket() {
    let mut h = Harness::new(small_spec(256), sched(true), MockConfig::default());
    h.submit(request(
        1,
        vec![1, 2, 3],
        SamplingParams::greedy(),
        20,
        CacheScope::Private,
    ));
    h.run();
    for b in h.eng.executor().buckets_used() {
        assert_eq!((b.max_seqs, b.max_tokens), (1, 16));
    }
}

/// Half the time a shared head plus a random tail, otherwise random.
fn headed(rng: &mut TestRng, head: &[u32]) -> Vec<u32> {
    let mut t = if rng.chance(0.5) {
        head.to_vec()
    } else {
        Vec::new()
    };
    t.extend(rng.var_tokens(1, 30, 32));
    t
}

/// Exact for non-speculative or greedy runs (a prefix when the sequence alone exhausted
/// KV memory and was ended early), length-only otherwise.
fn check_output(h: &Harness, id: u64, prompt: &[u32], p: &SamplingParams, max: u32, spec_on: bool) {
    let out = &h.outputs[&id];
    let early = h.finished.get(&id) == Some(&FinishReason::Length) && (out.len() as u32) < max;
    if !spec_on || p.is_greedy() {
        let exp = h.expected(prompt, p, max, &[]);
        if early {
            assert_eq!(out[..], exp[..out.len()], "request {id}");
        } else {
            assert_eq!(out, &exp, "request {id}");
        }
    } else if !early {
        assert_eq!(out.len() as u32, max);
    }
}

/// Randomized workload: multi-turn keyed conversations (continuations and regenerations),
/// private requests, random sampling, speculation on or off, cancellations, clock jumps
/// past the idle TTL, and small pools that force eviction and preemption. Every step
/// checks the KV invariants (no leak, no double free, exact refcounts) and that every free
/// block is zero or has a zero queued. Every completed request must equal the dense
/// reference; cache hits must never exceed what an earlier request under the same salt
/// computed.
#[test]
fn randomized_workloads_hold_every_invariant() {
    for seed in 0..24u64 {
        let mut rng = TestRng(100 + seed);
        let spec_on = seed % 2 == 0;
        let blocks = 48 + rng.below(80) as u32;
        let mut cfg = sched(spec_on);
        cfg.max_prefill_chunk = 1 + rng.below(40) as u32;
        cfg.max_batched_tokens = 32 + rng.below(100) as u32;
        cfg.cache.idle_ttl_ms = 500;
        cfg.cache.max_age_ms = 2000;
        cfg.sweep_interval_ms = 50;
        let mut h = Harness::new(small_spec(blocks), cfg, MockConfig::default());

        // Conversation state: salt id -> last full transcript.
        let mut convo: HashMap<u8, Vec<u32>> = HashMap::new();
        // salt id -> every token sequence computed under it.
        let mut history: HashMap<u8, Vec<Vec<u32>>> = HashMap::new();
        let mut live: HashMap<u64, (Vec<u32>, SamplingParams, u32, Option<u8>)> = HashMap::new();
        let mut cancelled = BTreeSet::new();
        let mut hit_checked = std::collections::HashSet::new();
        // A shared head (a common system prompt) across salts: only salting keeps apart.
        let head = rng.tokens(13, 32);
        let mut hits_seen = 0usize;
        let mut next_id = 0u64;

        for _round in 0..400 {
            if rng.chance(0.35) && live.len() < 40 {
                let id = next_id;
                next_id += 1;
                let p = if spec_on && rng.chance(0.5) {
                    SamplingParams::greedy()
                } else {
                    params(id)
                };
                let max = 1 + rng.below(24) as u32;
                let (prompt, key) = if rng.chance(0.25) {
                    (headed(&mut rng, &head), None)
                } else {
                    let s = rng.below(5) as u8;
                    let prompt = match convo.get(&s) {
                        Some(t) if rng.chance(0.3) => t.clone(),
                        Some(t) => {
                            let mut t = t.clone();
                            t.extend(rng.var_tokens(1, 12, 32));
                            t
                        }
                        None => headed(&mut rng, &head),
                    };
                    (prompt, Some(s))
                };
                if prompt.len() as u32 > blocks * 4 / 2 {
                    continue;
                }
                let cache = match key {
                    Some(s) => CacheScope::Keyed(salt(s + 1)),
                    None => CacheScope::Private,
                };
                live.insert(id, (prompt.clone(), p, max, key));
                h.submit(request(id, prompt, p, max, cache));
            }
            if rng.chance(0.03) && !live.is_empty() {
                let ids: Vec<u64> = live.keys().copied().collect();
                let victim = ids[rng.below(ids.len() as u64) as usize];
                if h.eng.cancel(victim, h.now).is_some() {
                    cancelled.insert(victim);
                    let (prompt, _, _, key) = live.remove(&victim).unwrap();
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
                let (prompt, _, _, key) = &live[&id];
                // A first-admission hit can only come from blocks computed under the same
                // salt: finished or cancelled transcripts and live same-salt sequences.
                let allowed = match key {
                    None => 0,
                    Some(s) => {
                        let mut sources: Vec<Vec<u32>> =
                            history.get(s).cloned().unwrap_or_default();
                        for (j, (pj, _, _, kj)) in &live {
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
                let (prompt, p, max, key) = live.remove(&id).unwrap();
                check_output(&h, id, &prompt, &p, max, spec_on);
                if let Some(s) = key {
                    let mut t = prompt.clone();
                    t.extend(&h.outputs[&id]);
                    history.entry(s).or_default().push(t.clone());
                    convo.insert(s, t);
                }
            }
            h.now += if rng.chance(0.02) {
                600
            } else {
                1 + rng.below(5)
            };
        }
        h.run();
        for (id, (prompt, p, max, _)) in live {
            check_output(&h, id, &prompt, &p, max, spec_on);
        }
        let hits = hits_seen;
        assert!(hits > 0, "seed {seed}: workload produced no cache hits");
        eprintln!(
            "seed {seed}: blocks {blocks} requests {next_id} stats {:?} cancelled {} zeros {} hits {hits}",
            h.eng.stats(),
            cancelled.len(),
            h.eng.executor().zero_log().len(),
        );
        assert!(h.eng.stats().steps > 0);
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
        assert!(
            h.eng.stats().preemptions > 0 || seed > 3 || blocks > 100,
            "seed {seed}: no pressure"
        );
        let _ = cancelled;
    }
}

/// A configuration under which some sequence could never be stepped is refused at
/// construction. Without the check, a token budget below `1 + k` plans no decode row
/// ever: the request below would sit unfinished forever.
#[test]
fn engine_refuses_configurations_that_cannot_progress() {
    use eidola_engine::engine::{ConfigError, Engine};
    use eidola_engine::mock::MockExecutor;

    let build = |spec: eidola_engine::spec::ModelSpec, sched| {
        Engine::new(MockExecutor::new(spec, MockConfig::default()), sched)
    };
    // k = 3: a decode row costs 4 query tokens.
    let mut tight = sched(true);
    tight.max_batched_tokens = 3;
    assert!(matches!(
        build(small_spec(64), tight.clone()),
        Err(ConfigError::Scheduler(_))
    ));
    // The same budget is fine without speculation, and 4 is fine with it.
    tight.speculative = false;
    assert!(build(small_spec(64), tight.clone()).is_ok());
    tight.speculative = true;
    tight.max_batched_tokens = 4;
    assert!(build(small_spec(64), tight.clone()).is_ok());
    // The largest bucket caps the budget too.
    let mut spec = small_spec(64);
    for b in &mut spec.buckets {
        b.max_tokens = b.max_tokens.min(3);
    }
    assert!(matches!(
        build(spec, sched(true)),
        Err(ConfigError::Scheduler(_))
    ));
    // No seats, no prefill.
    let mut no_seats = sched(true);
    no_seats.max_seqs = 0;
    assert!(matches!(
        build(small_spec(64), no_seats),
        Err(ConfigError::Scheduler(_))
    ));
    let mut no_chunk = sched(true);
    no_chunk.max_prefill_chunk = 0;
    assert!(matches!(
        build(small_spec(64), no_chunk),
        Err(ConfigError::Scheduler(_))
    ));
    // The bucket ladder must grow in both dimensions: with a trade-off shape the last
    // bucket would not bound a step (lexicographic order calls this one sorted).
    let mut tradeoff = small_spec(64);
    tradeoff.buckets = vec![
        eidola_engine::spec::Bucket {
            max_seqs: 1,
            max_tokens: 100,
        },
        eidola_engine::spec::Bucket {
            max_seqs: 2,
            max_tokens: 5,
        },
    ];
    assert!(matches!(
        build(tradeoff, sched(true)),
        Err(ConfigError::Spec(_))
    ));
    let mut equal_steps = small_spec(64);
    equal_steps.buckets = vec![
        eidola_engine::spec::Bucket {
            max_seqs: 1,
            max_tokens: 64,
        },
        eidola_engine::spec::Bucket {
            max_seqs: 8,
            max_tokens: 64,
        },
    ];
    assert!(build(equal_steps, sched(true)).is_ok());
    // An inconsistent spec is an error, not a panic.
    let mut bad = small_spec(64);
    bad.buckets.clear();
    assert!(matches!(build(bad, sched(true)), Err(ConfigError::Spec(_))));
}

/// The smallest accepted budget (exactly one decode row with its drafts) completes
/// requests and matches the dense reference (greedy: speculative sampling is equal to
/// it in distribution only).
#[test]
fn minimal_token_budget_still_finishes() {
    let mut cfg = sched(true);
    cfg.max_batched_tokens = 4;
    let mut h = Harness::new(small_spec(256), cfg, MockConfig::default());
    let mut rng = TestRng(77);
    let mut cases = Vec::new();
    for id in 0..4 {
        let prompt = rng.var_tokens(1, 12, 32);
        let p = SamplingParams::greedy();
        cases.push((id, prompt.clone(), p));
        h.submit(request(id, prompt, p, 6, CacheScope::Private));
    }
    h.run();
    for (id, prompt, p) in cases {
        assert_eq!(
            h.outputs[&id],
            h.expected(&prompt, &p, 6, &[]),
            "request {id}"
        );
    }
}

/// Padded logit rows (past the tokenizer's last token) are never sampled, drafted or
/// accepted, even where they would win: with 20 of 32 rows sampleable, unmasked greedy
/// decoding picks a padded id somewhere in this workload, and the engine never emits one.
#[test]
fn padded_vocabulary_is_never_emitted() {
    use eidola_engine::mock::reference_logits;
    use eidola_engine::sampling::{Logits, argmax};

    const SAMPLEABLE: u32 = 20;
    let mut spec = small_spec(256);
    spec.sampleable_vocab_size = SAMPLEABLE;
    let mut unmasked_padded_wins = 0;
    for spec_on in [false, true] {
        let mut h = Harness::new(spec.clone(), sched(spec_on), MockConfig::default());
        let mut rng = TestRng(91);
        let mut cases = Vec::new();
        for id in 0..12 {
            let prompt = rng.var_tokens(1, 20, SAMPLEABLE);
            let p = params(id);
            cases.push((id, prompt.clone(), p));
            h.submit(request(id, prompt, p, 16, CacheScope::Private));
        }
        h.run();
        for (id, prompt, p) in cases {
            let out = &h.outputs[&id];
            assert!(out.iter().all(|&t| t < SAMPLEABLE), "request {id}: {out:?}");
            if !spec_on || p.is_greedy() {
                assert_eq!(*out, h.expected(&prompt, &p, 16, &[]), "request {id}");
            }
            let mut seq = prompt.clone();
            for &t in out {
                let full = reference_logits(&spec, &h.cfg, &seq, seq.len() as u32 - 1);
                if argmax(Logits::new(&full, spec.vocab_size)) >= SAMPLEABLE {
                    unmasked_padded_wins += 1;
                }
                seq.push(t);
            }
        }
    }
    assert!(
        unmasked_padded_wins > 10,
        "the workload must include positions where a padded row would win ({unmasked_padded_wins})"
    );
}

/// A prompt token outside the sampleable vocabulary is refused at submission, and an
/// executor that returns one fails the step instead of having it committed.
#[test]
fn padded_ids_are_refused_at_both_ends_of_the_seam() {
    use eidola_engine::engine::{Engine, SubmitError};
    use eidola_engine::executor::{Executor, ExecutorError, StepInput, StepOutput};
    use eidola_engine::mock::MockExecutor;
    use eidola_engine::spec::ModelSpec;

    let mut spec = small_spec(256);
    spec.sampleable_vocab_size = 20;
    let mut h = Harness::new(spec, sched(false), MockConfig::default());
    assert_eq!(
        h.eng.submit(request(
            1,
            vec![3, 25],
            SamplingParams::greedy(),
            4,
            CacheScope::Private
        )),
        Err(SubmitError::InvalidToken)
    );

    /// Samples over all 32 rows while claiming only id 0 is sampleable.
    struct Lying(MockExecutor, ModelSpec);
    impl Executor for Lying {
        fn spec(&self) -> &ModelSpec {
            &self.1
        }
        fn execute(&mut self, step: &StepInput) -> Result<StepOutput, ExecutorError> {
            self.0.execute(step)
        }
    }
    let honest = small_spec(256);
    let mut claimed = honest.clone();
    claimed.sampleable_vocab_size = 1;
    let mut eng = Engine::new(
        Lying(MockExecutor::new(honest, MockConfig::default()), claimed),
        sched(false),
    )
    .unwrap();
    eng.submit(request(
        1,
        vec![0, 0, 0],
        SamplingParams::random(1.0, 5).unwrap(),
        8,
        CacheScope::Private,
    ))
    .unwrap();
    let mut failed = false;
    for now in 0..32 {
        match eng.step(now) {
            Ok(events) => {
                for e in events {
                    assert!(e.tokens.iter().all(|&t| t == 0), "{:?}", e.tokens);
                }
            }
            Err(e) => {
                assert!(e.0.contains("sampleable"), "{e}");
                failed = true;
                break;
            }
        }
    }
    assert!(failed, "a padded id must fail the step");
}

/// A row sampling at position 0 (a one-token prompt) has nothing to draft from, so the
/// scheduler reserves no drafts for it. With block size 1, k = 3 and two allocatable
/// blocks, reserving them would make a request that plain decoding serves end with
/// `Length` and no output; and the draft counter would count drafts that never existed.
#[test]
fn no_drafts_are_reserved_at_position_zero() {
    use eidola_engine::mock::mimo_like_spec;

    // Block size 1, 3 blocks per group (2 allocatable), k = 3.
    let spec = mimo_like_spec(32, 1, 6, 10, 3, 4, 3);
    let mut h = Harness::new(spec, sched(true), MockConfig::default());
    let greedy = SamplingParams::greedy();
    h.submit(request(1, vec![7], greedy, 1, CacheScope::Private));
    h.run();
    assert_eq!(h.outputs[&1], h.expected(&[7], &greedy, 1, &[]));
    assert_eq!(h.outputs[&1].len(), 1);
    assert_eq!(h.finished[&1], FinishReason::Length);
    assert_eq!(h.eng.stats().drafted, 0);

    // With room to speculate, only rows past position 0 draft: one-token prompts each
    // contribute exactly k drafts per later decode step and none for their first token.
    let mut h = Harness::new(small_spec(256), sched(true), MockConfig::default());
    for id in 0..4 {
        h.submit(request(
            id,
            vec![id as u32 + 1],
            greedy,
            2,
            CacheScope::Private,
        ));
    }
    h.run();
    let stats = h.eng.stats();
    assert_eq!(stats.drafted, 4 * 3, "{stats:?}");
    for id in 0..4u64 {
        let prompt = [id as u32 + 1];
        assert_eq!(h.outputs[&id], h.expected(&prompt, &greedy, 2, &[]));
    }
}

/// Drafts never cost a sequence its output: when plain decoding fits in KV but the
/// `1 + k` reservation does not, the row drops its drafts instead of finishing with
/// `Length`. Block size 1 and k = 3; both requests used to end early (the second with
/// no output at all).
#[test]
fn kv_pressure_drops_drafts_before_finishing() {
    use eidola_engine::mock::mimo_like_spec;

    let greedy = SamplingParams::greedy();
    // (allocatable blocks + 1, prompt): plain decoding needs prompt + 1 positions.
    for (blocks, prompt) in [(3, vec![7u32]), (4, vec![7, 9])] {
        let spec = mimo_like_spec(32, 1, 6, 10, blocks, 4, 3);
        let mut h = Harness::new(spec, sched(true), MockConfig::default());
        h.submit(request(1, prompt.clone(), greedy, 2, CacheScope::Private));
        h.run();
        assert_eq!(
            h.outputs[&1],
            h.expected(&prompt, &greedy, 2, &[]),
            "{prompt:?}"
        );
        assert_eq!(h.outputs[&1].len(), 2);
        assert_eq!(h.finished[&1], FinishReason::Length);
        assert_eq!(h.eng.stats().drafted, 0);
        assert_eq!(h.eng.stats().preemptions, 0);
    }
    // Where even plain decoding cannot grow, the documented `Length` still applies.
    let spec = mimo_like_spec(32, 1, 6, 10, 3, 4, 3);
    let mut h = Harness::new(spec, sched(true), MockConfig::default());
    h.submit(request(1, vec![7], greedy, 5, CacheScope::Private));
    h.run();
    assert_eq!(h.outputs[&1].len(), 2);
    assert_eq!(h.outputs[&1], h.expected(&[7], &greedy, 2, &[]));
    assert_eq!(h.finished[&1], FinishReason::Length);
}

/// Draft reservations are the first thing given up across the whole step: when the
/// younger of two decoding rows does not fit, the older row's drafts are revoked before
/// anyone is pushed to plain decoding, preempted or finished. Block size 1, k = 3, eight
/// allocatable blocks: both rows fit decoding plainly (3 blocks each), but the older
/// row's `1 + 3` reservation (6 blocks) would leave the younger 2, one short, which
/// used to preempt it.
#[test]
fn draft_reservations_yield_before_any_row_is_preempted() {
    use eidola_engine::mock::mimo_like_spec;

    let greedy = SamplingParams::greedy();
    let spec = mimo_like_spec(32, 1, 6, 10, 9, 4, 3);
    let mut h = Harness::new(spec, sched(true), MockConfig::default());
    let prompts = [vec![7u32, 9], vec![3u32, 5]];
    for (id, p) in prompts.iter().enumerate() {
        h.submit(request(
            id as u64,
            p.clone(),
            greedy,
            2,
            CacheScope::Private,
        ));
    }
    h.run();
    for (id, p) in prompts.iter().enumerate() {
        assert_eq!(h.outputs[&(id as u64)], h.expected(p, &greedy, 2, &[]));
        assert_eq!(h.finished[&(id as u64)], FinishReason::Length);
    }
    let stats = h.eng.stats();
    assert_eq!(stats.preemptions, 0, "{stats:?}");
    assert_eq!(stats.drafted, 0, "{stats:?}");
    assert_eq!(stats.steps, 2, "both rows decode together: {stats:?}");
}

/// Block size 2, sliding window 4, three allocatable blocks in the sliding groups (the
/// full-attention pool is ample): enough for any one window plus the block being
/// written, provided blocks behind the window are recycled.
fn tight_sliding_spec() -> eidola_engine::spec::ModelSpec {
    let mut spec = eidola_engine::mock::mimo_like_spec(32, 2, 4, 4, 64, 4, 3);
    for g in &mut spec.kv_groups {
        if g.attention != AttentionKind::Full {
            g.num_blocks = 4;
        }
    }
    spec
}

/// With caching off, no regeneration window is retained: nothing could ever hit it, and
/// holding the prompt's window would stop the sliding groups recycling (retaining it
/// makes a five-token prompt end with `Length` once decoding needs a fourth block).
#[test]
fn no_regeneration_retention_without_caching() {
    let mut cfg = sched(false);
    cfg.cache.enabled = false;
    let mut h = Harness::new(tight_sliding_spec(), cfg, MockConfig::default());
    let greedy = SamplingParams::greedy();
    let prompt = vec![1u32, 2, 3, 4, 5];
    h.submit(request(1, prompt.clone(), greedy, 8, CacheScope::Private));
    h.run();
    assert_eq!(h.outputs[&1], h.expected(&prompt, &greedy, 8, &[]));
    assert_eq!(h.outputs[&1].len(), 8);
}

/// `Length` means the sequence cannot make its minimum progress even holding every
/// reclaimable block. Under the tight sliding geometry, before concluding that the
/// scheduler sheds drafts, then the prefill chunk (down to one block), then the
/// sequence's own branch and regeneration retention (the windows it attached to cache
/// entries, which only its own release would otherwise give back). Skipping any of these
/// steps ends the case with `Length` early (the 8-token prompt with no output at all).
#[test]
fn optional_reservations_are_shed_before_length() {
    let greedy = SamplingParams::greedy();
    for spec_on in [false, true] {
        for (prompt, chunk) in [
            // Needs the prefill chunk cut: 8 tokens at once is four sliding blocks.
            (vec![1u32, 2, 3, 4, 5, 6, 7, 8], 8),
            // Needs the regeneration retention given back (keyed, caching on): the
            // prompt's window stays attached behind the sliding window once decoding
            // moves on.
            (vec![1u32, 2, 3, 4, 5], 2),
        ] {
            let mut cfg = sched(spec_on);
            cfg.max_prefill_chunk = chunk;
            let mut h = Harness::new(tight_sliding_spec(), cfg, MockConfig::default());
            h.submit(request(
                1,
                prompt.clone(),
                greedy,
                8,
                CacheScope::Keyed(salt(1)),
            ));
            h.run();
            assert_eq!(
                h.outputs[&1],
                h.expected(&prompt, &greedy, 8, &[]),
                "speculative {spec_on}, prompt {prompt:?}"
            );
            assert_eq!(h.outputs[&1].len(), 8);
            assert_eq!(h.eng.stats().preemptions, 0);
        }
    }
}
