//! The Qwen CPU executor through the real engine on the committed fixture: every row's
//! target logits, drafter rows, chain rows and draft draws, every produced token and
//! every stored recurrent state equal to the dense oracle, bit for bit, across chunk
//! sizes, step budgets, interleavings, prefix hits at state snapshots, preemption and
//! drafting. Every test runs on each committed fixture (`qwen_common::FIXTURES`).

mod qwen_common;

use eidola_engine::engine::{CacheScope, FinishReason};
use eidola_engine::sampling::SamplingParams;
use qwen_common::*;

fn requests_match_the_oracle_at_every_chunking() {
    for speculative in [false, true] {
        for chunk in [1, 3, 4, 5, 7, 256] {
            let k = if speculative { 3 } else { 0 };
            let mut h = Harness::fixture(512, 128, k, chunked(chunk, speculative));
            let mut rng = TestRng(chunk as u64 + 100 * speculative as u64);
            for id in 0..6 {
                let prompt = rng.var_tokens(1, 24, VOCAB);
                let max = 1 + rng.below(14) as u32;
                let cache = if id % 2 == 0 {
                    CacheScope::Private
                } else {
                    keyed(id as u8)
                };
                h.submit(request(id, prompt, params(id), max, cache));
            }
            h.run();
            h.flush();
            let sum = h.check_all(speculative);
            assert!(sum.states > 0, "chunk {chunk}: nothing stored");
            if speculative {
                assert!(sum.chain_rows > 0, "chunk {chunk}: nothing drafted");
            }
            assert_eq!(h.eng.kv_keys_held(), 0);
        }
    }
}

fn interleaved_rows_under_any_budget_match_the_oracle() {
    for budget in [8, 13, 64] {
        let mut s = chunked(5, true);
        s.max_batched_tokens = budget;
        let mut h = Harness::fixture(512, 128, 3, s);
        let mut rng = TestRng(budget as u64);
        for id in 0..8u64 {
            let prompt = rng.var_tokens(2, 20, VOCAB);
            h.submit(request(id, prompt, params(id), 12, keyed(id as u8)));
            if id % 3 == 2 {
                h.step();
                h.now += 1;
            }
        }
        h.run();
        h.flush();
        let sum = h.check_all(true);
        assert!(sum.records > 8 && sum.states > 0);
    }
}

/// A request's every logit is the same bits alone and beside others, in another slot.
fn a_request_alone_and_beside_others_computes_the_same_bits() {
    let prompt = TestRng(7).tokens(17, VOCAB);
    let g = SamplingParams::greedy();
    let mut alone = Harness::fixture(512, 64, 3, chunked(6, true));
    alone.run_one(request(1, prompt.clone(), g, 10, keyed(1)));
    let mut crowd = Harness::fixture(512, 64, 3, chunked(6, true));
    let mut rng = TestRng(8);
    for id in 10..14 {
        crowd.submit(request(
            id,
            rng.tokens(13, VOCAB),
            params(id),
            9,
            keyed(id as u8),
        ));
    }
    crowd.step();
    crowd.submit(request(1, prompt, g, 10, keyed(1)));
    crowd.run();
    crowd.flush();
    assert_eq!(alone.outputs[&1], crowd.outputs[&1]);
    let logits = |h: &Harness| {
        let mut v: Vec<(u32, Vec<u32>)> = h
            .records
            .iter()
            .filter(|r| r.tokens.len() >= 17 && r.tokens[..17] == alone.prompts[&1].0[..])
            .flat_map(|r| r.target_logits.iter().map(|(p, l)| (*p, bits(l))))
            .collect();
        v.sort();
        v.dedup();
        v
    };
    assert_eq!(logits(&alone), logits(&crowd));
    assert!(
        crowd
            .records
            .iter()
            .any(|r| r.slot != alone.records[0].slot)
    );
    crowd.check_all(true);
}

fn hits_at_every_snapshot_match_the_oracle() {
    let g = SamplingParams::greedy();
    for speculative in [false, true] {
        let k = if speculative { 3 } else { 0 };
        let mut h = Harness::fixture(512, 64, k, sched(speculative));
        let mut rng = TestRng(9);
        let prompt = rng.tokens(14, VOCAB);
        let out = h.run_one(request(1, prompt.clone(), g, 7, keyed(1)));
        // Regeneration: the last boundary before the final token.
        h.run_one(request(2, prompt.clone(), g, 7, keyed(1)));
        assert_eq!(h.cached[&2], 12);
        assert_eq!(h.outputs[&2], out);
        // Continuation: the prompt, exactly its output, and more.
        let mut next = prompt.clone();
        next.extend(&out);
        next.extend(rng.tokens(3, VOCAB));
        h.run_one(request(3, next, g, 6, keyed(1)));
        assert_eq!(h.cached[&3], (14 + 7 - 1) / 4 * 4);
        // Branch: two requests diverging inside the KV past the snapshots below it; the
        // first recomputes, the second resumes at the branch.
        let mut b = prompt[..10].to_vec();
        b.extend(rng.tokens(6, VOCAB));
        h.run_one(request(4, b, g, 5, keyed(1)));
        assert_eq!(h.cached[&4], 0);
        let mut c = prompt[..10].to_vec();
        c.extend(rng.tokens(5, VOCAB));
        h.run_one(request(5, c, g, 5, keyed(1)));
        assert_eq!(h.cached[&5], 8);
        // A short turn after a hit, run whole.
        let mut short = prompt.clone();
        short.push(rng.below(VOCAB as u64) as u32);
        h.run_one(request(6, short, g, 4, keyed(1)));
        assert_eq!(h.cached[&6], 12);
        let sum = h.check_all(speculative);
        assert!(sum.states > 0);
    }
}

fn preemption_and_resumes_match_the_oracle() {
    for (blocks, buffers) in [(24, 128), (512, 6), (28, 5)] {
        for speculative in [false, true] {
            let k = if speculative { 3 } else { 0 };
            let mut h = Harness::fixture(blocks, buffers, k, chunked(5, speculative));
            let mut rng = TestRng(blocks as u64 * 7 + buffers as u64);
            for id in 0..8u64 {
                let prompt = rng.var_tokens(3, 18, VOCAB);
                let cache = if id % 3 == 0 {
                    CacheScope::Private
                } else {
                    keyed(id as u8 % 3)
                };
                h.submit(request(id, prompt, params(id), 14, cache));
            }
            h.run();
            h.flush();
            let pressure = h.eng.stats().preemptions
                + h.eng.kv().stripped_states()
                + h.eng.kv().saved_states();
            assert!(
                pressure > 0,
                "blocks {blocks} buffers {buffers}: no pressure"
            );
            h.check_all(speculative);
            assert_eq!(h.eng.kv_keys_held(), 0);
        }
    }
}

/// A model whose distributions have few outcomes (the sampleable vocabulary narrowed), so
/// drafts are accepted in every pattern from none to all.
fn narrow(sampleable: u32, buffers: u32) -> Harness {
    let mut cfg = exec_config(512, buffers, 3);
    cfg.sampleable_vocab_size = sampleable;
    Harness::new(fixture(), cfg, sched(true))
}

fn drafts_accepted_and_rejected_in_every_pattern_match_the_oracle() {
    let mut counts = [0usize; 5];
    for sampleable in [2, 3] {
        let mut h = narrow(sampleable, 128);
        let mut rng = TestRng(sampleable as u64);
        for id in 0..10u64 {
            let prompt = rng.var_tokens(2, 12, sampleable);
            h.submit(request(id, prompt, params(id), 24, keyed(id as u8)));
        }
        h.run();
        h.flush();
        let sum = h.check_all(true);
        for (c, n) in counts.iter_mut().zip(sum.accepted_counts) {
            *c += n;
        }
    }
    assert!(
        counts[..4].iter().all(|&n| n > 0),
        "acceptance patterns seen: {counts:?}"
    );
}

/// Greedy output is the same drafted and undrafted, at every acceptance.
fn greedy_drafting_equals_plain_decoding() {
    for sampleable in [2, 3, SAMPLEABLE] {
        let mut outs = Vec::new();
        for speculative in [false, true] {
            let mut cfg = exec_config(512, 128, if speculative { 3 } else { 0 });
            cfg.sampleable_vocab_size = sampleable;
            let mut h = Harness::new(fixture(), cfg, chunked(5, speculative));
            let mut rng = TestRng(11);
            for id in 0..5u64 {
                let prompt = rng.var_tokens(2, 15, sampleable.min(VOCAB));
                h.submit(request(
                    id,
                    prompt,
                    SamplingParams::greedy(),
                    20,
                    keyed(id as u8),
                ));
            }
            h.run();
            let mut o: Vec<_> = h.outputs.clone().into_iter().collect();
            o.sort();
            outs.push(o);
        }
        assert_eq!(outs[0], outs[1], "sampleable {sampleable}");
    }
}

/// A stop token or a length limit inside the drafts a step accepted leaves the device past
/// the end of the turn; the continuation point still holds the turn's own state, saved
/// from the slot, and the next turn resumes there exactly.
fn an_ending_inside_accepted_drafts_saves_the_turns_own_state() {
    let g = SamplingParams::greedy();
    let mut saved = 0;
    for sampleable in [2, 3] {
        for seed in 0..10u64 {
            for stop in [false, true] {
                let mut h = narrow(sampleable, 64);
                let mut rng = TestRng(seed * 31 + sampleable as u64);
                let prompt = rng.var_tokens(5, 9, sampleable);
                let cut = 4 + (seed % 8) as usize;
                let mut first = request(1, prompt.clone(), g, cut as u32, keyed(1));
                if stop {
                    // Stop on the token the free run ends with, if it is new there.
                    let mut free = Harness::new(
                        fixture(),
                        {
                            let mut c = exec_config(512, 64, 0);
                            c.sampleable_vocab_size = sampleable;
                            c
                        },
                        sched(false),
                    );
                    let out = free.run_one(request(1, prompt.clone(), g, cut as u32, keyed(1)));
                    if out[..cut - 1].contains(&out[cut - 1]) {
                        continue;
                    }
                    first.stop_token_ids = vec![out[cut - 1]];
                    first.max_tokens = 40;
                }
                h.run_one(first);
                assert_eq!(h.outputs[&1].len(), cut);
                saved += h.eng.kv().saved_states();
                let mut next = prompt.clone();
                next.extend(&h.outputs[&1]);
                next.extend(rng.tokens(2, sampleable));
                h.run_one(request(2, next, g, 6, keyed(1)));
                let computed = (prompt.len() + cut - 1) as u32;
                assert_eq!(h.cached[&2], computed / 4 * 4, "seed {seed} stop {stop}");
                h.check_all(true);
            }
        }
    }
    assert!(
        saved > 0,
        "no ending fell inside accepted drafts past a boundary"
    );
}

fn shedding_under_pressure_ends_as_the_engine_says() {
    // One sequence alone in a pool too small for its prompt ends without growing.
    let mut h = Harness::fixture(6, 8, 3, chunked(4, true));
    h.submit(request(
        1,
        TestRng(12).tokens(18, VOCAB),
        SamplingParams::greedy(),
        30,
        keyed(1),
    ));
    h.run();
    h.flush();
    assert!(matches!(
        h.finished[&1],
        FinishReason::KvExhausted | FinishReason::Length
    ));
    h.check_all(true);
}

/// A request requiring a cached prefix up to a snapshot is admitted and exact; one
/// requiring a token past the hit is refused before anything runs.
fn a_required_prefix_resumes_at_a_snapshot_or_is_refused() {
    let g = SamplingParams::greedy();
    let mut h = Harness::fixture(256, 32, 3, sched(true));
    let prompt = TestRng(13).tokens(14, VOCAB);
    h.run_one(request(1, prompt.clone(), g, 5, keyed(1)));
    let mut ok = request(2, prompt.clone(), g, 5, keyed(1));
    ok.required_cached_tokens = 12;
    h.run_one(ok);
    assert_eq!(h.cached[&2], 12);
    let mut too_far = request(3, prompt, g, 5, keyed(1));
    too_far.required_cached_tokens = 13;
    h.run_one(too_far);
    assert_eq!(h.finished[&3], FinishReason::CachedPrefixMissing);
    h.check_all(true);
}

on_each_fixture!(
    requests_match_the_oracle_at_every_chunking,
    interleaved_rows_under_any_budget_match_the_oracle,
    a_request_alone_and_beside_others_computes_the_same_bits,
    hits_at_every_snapshot_match_the_oracle,
    preemption_and_resumes_match_the_oracle,
    drafts_accepted_and_rejected_in_every_pattern_match_the_oracle,
    greedy_drafting_equals_plain_decoding,
    an_ending_inside_accepted_drafts_saves_the_turns_own_state,
    shedding_under_pressure_ends_as_the_engine_says,
    a_required_prefix_resumes_at_a_snapshot_or_is_refused,
);
