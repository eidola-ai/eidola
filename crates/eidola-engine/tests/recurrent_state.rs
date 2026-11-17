//! The recurrent state (a Qwen3.8-shaped model: full-attention KV beside a fixed-size
//! state per sequence) through the scheduler, KV manager and prefix cache over the mock,
//! against the dense reference that folds every token from position 0.
//!
//! The mock's state is an order-sensitive fold of tokens and positions, and the logits
//! mix it in, so a skipped, doubled or rejected token changes the output. It fails a step
//! that reads a base holding another position's state, a zeroed or encrypted base, a
//! write into a buffer that holds a state, a fold over positions the slot never kept, a
//! tail past its bound, and a state saved outside the slot's range. The harness checks
//! after every step that free buffers are zero and idle snapshots ciphertext.

mod common;

use std::collections::{BTreeSet, HashSet};

use common::*;
use eidola_engine::engine::{CacheScope, FinishReason, SchedulerConfig};
use eidola_engine::kv::CachePolicy;
use eidola_engine::mock::{MockConfig, qwen_like_spec, recurrent_state, with_host_tier};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::{ModelSpec, NULL_STATE};

const VOCAB: u32 = 32;
const BS: u32 = 4;

/// vocab 32, block 4, `blocks` per KV group, `buffers` state buffers, 32 slots, k = 3.
fn qspec(blocks: u32, buffers: u32) -> ModelSpec {
    qwen_like_spec(VOCAB, BS, blocks, buffers, 32, 3)
}

fn params(i: u64) -> SamplingParams {
    match i % 3 {
        0 => SamplingParams::greedy(),
        1 => SamplingParams::random(1.0, 1000 + i).unwrap(),
        _ => SamplingParams::new(0.7, 8, 0.9, 0.05, 2000 + i).unwrap(),
    }
}

fn chunked(chunk: u32, speculative: bool) -> SchedulerConfig {
    let mut cfg = sched(speculative);
    cfg.max_prefill_chunk = chunk;
    cfg
}

fn keyed(n: u8) -> CacheScope {
    CacheScope::Keyed(salt(n))
}

/// The state snapshots the cache holds, as `(boundary, device buffer, host buffer)`.
fn snapshots(h: &Harness) -> Vec<(u32, u32, u32)> {
    h.eng.kv().state_snapshots()
}

/// Every state image the executor stored is the state the dense reference reaches at its
/// position on one of `sequences`: no stored state ever folded a token it should not have.
fn assert_stored_states_are_states(h: &Harness, sequences: &[Vec<u32>]) {
    let stored = h.eng.executor().stored_states();
    assert!(!stored.is_empty(), "nothing stored");
    let mut want = HashSet::new();
    for s in sequences {
        for len in (BS..=s.len() as u32).step_by(BS as usize) {
            want.insert(recurrent_state(&h.cfg, &s[..len as usize]).image(len));
        }
    }
    for image in stored {
        assert!(
            want.contains(image),
            "a stored state at {} is no prefix's state",
            image[4] - 1
        );
    }
}

/// Runs one request to completion, flushes, and returns its output.
fn run_one(
    h: &mut Harness,
    id: u64,
    prompt: Vec<u32>,
    p: SamplingParams,
    max: u32,
    cache: CacheScope,
) -> Vec<u32> {
    h.submit(request(id, prompt, p, max, cache));
    h.run();
    h.flush();
    h.outputs[&id].clone()
}

#[test]
fn requests_match_the_reference_at_every_chunking() {
    for speculative in [false, true] {
        for chunk in [1, 3, 4, 5, 7, 256] {
            let mut h = Harness::new(
                qspec(512, 128),
                chunked(chunk, speculative),
                MockConfig::default(),
            );
            let mut rng = TestRng(chunk as u64);
            let mut cases = Vec::new();
            for id in 0..12 {
                let prompt = rng.var_tokens(1, 40, VOCAB);
                let p = params(id);
                let max = 1 + rng.below(30) as u32;
                cases.push((id, prompt.clone(), p, max));
                let cache = if id % 2 == 0 {
                    CacheScope::Private
                } else {
                    keyed(id as u8)
                };
                h.submit(request(id, prompt, p, max, cache));
            }
            h.run();
            h.flush();
            let mut sequences = Vec::new();
            for (id, prompt, p, max) in cases {
                let mut seq = prompt.clone();
                seq.extend(&h.outputs[&id]);
                sequences.push(seq);
                if speculative && !p.is_greedy() {
                    continue; // equal in distribution only
                }
                assert_eq!(
                    h.outputs[&id],
                    h.expected(&prompt, &p, max, &[]),
                    "chunk {chunk} speculative {speculative} request {id}"
                );
            }
            assert_stored_states_are_states(&h, &sequences);
            assert_eq!(h.eng.kv_keys_held(), 0);
        }
    }
}

#[test]
fn interleaved_steps_under_any_budget_match_the_reference() {
    for budget in [8, 13, 64] {
        let mut cfg = chunked(5, true);
        cfg.max_batched_tokens = budget;
        let mut h = Harness::new(qspec(512, 128), cfg, MockConfig::default());
        let mut rng = TestRng(budget as u64 + 7);
        let mut cases = Vec::new();
        for id in 0..10 {
            let prompt = rng.var_tokens(2, 30, VOCAB);
            cases.push((id, prompt.clone()));
            h.submit(request(
                id,
                prompt,
                SamplingParams::greedy(),
                20,
                keyed(id as u8),
            ));
            if id % 3 == 2 {
                h.step();
                h.now += 1;
            }
        }
        h.run();
        let g = SamplingParams::greedy();
        let mut sequences = Vec::new();
        for (id, prompt) in cases {
            assert_eq!(
                h.outputs[&id],
                h.expected(&prompt, &g, 20, &[]),
                "budget {budget} {id}"
            );
            let mut seq = prompt;
            seq.extend(&h.outputs[&id]);
            sequences.push(seq);
        }
        assert_stored_states_are_states(&h, &sequences);
    }
}

#[test]
fn a_regeneration_resumes_at_the_last_boundary_before_the_final_token() {
    for speculative in [false, true] {
        for n in 5..=13u32 {
            let mut h = Harness::new(qspec(256, 64), sched(speculative), MockConfig::default());
            let prompt = TestRng(n as u64).tokens(n as usize, VOCAB);
            let g = SamplingParams::greedy();
            let first = run_one(&mut h, 1, prompt.clone(), g, 6, keyed(1));
            let again = run_one(&mut h, 2, prompt.clone(), g, 6, keyed(1));
            assert_eq!(first, again);
            assert_eq!(first, h.expected(&prompt, &g, 6, &[]));
            assert_eq!(h.cached[&2], (n - 1) / BS * BS, "n {n}");
        }
    }
}

#[test]
fn a_continuation_resumes_at_the_last_boundary_it_computed() {
    for speculative in [false, true] {
        for c in 1..=9u32 {
            let mut h = Harness::new(qspec(256, 64), sched(speculative), MockConfig::default());
            let prompt = TestRng(c as u64 + 50).tokens(10, VOCAB);
            let g = SamplingParams::greedy();
            let out = run_one(&mut h, 1, prompt.clone(), g, c, keyed(1));
            let mut next = prompt.clone();
            next.extend(&out);
            next.extend(TestRng(c as u64).tokens(3, VOCAB));
            let got = run_one(&mut h, 2, next.clone(), g, 5, keyed(1));
            assert_eq!(got, h.expected(&next, &g, 5, &[]));
            let computed = 10 + c - 1;
            assert_eq!(h.cached[&2], computed / BS * BS, "c {c}");
        }
    }
}

/// A request diverging from a cached one inside the KV the cache holds, past every
/// snapshot below it, recomputes from the last snapshot (here the start) and keeps one
/// at its branch point, where the next divergent request resumes.
#[test]
fn a_divergence_past_the_snapshots_recomputes_and_keeps_a_branch() {
    let mut h = Harness::new(qspec(256, 64), sched(true), MockConfig::default());
    let mut rng = TestRng(9);
    let shared = rng.tokens(14, VOCAB);
    let g = SamplingParams::greedy();
    let mut a = shared.clone();
    a.extend(rng.tokens(12, VOCAB));
    run_one(&mut h, 1, a, g, 4, keyed(1));
    // The KV matches 3 blocks (12 positions); the snapshots are at 24 and later.
    let mut b = shared.clone();
    b.extend(rng.tokens(5, VOCAB));
    let out_b = run_one(&mut h, 2, b.clone(), g, 4, keyed(1));
    assert_eq!(out_b, h.expected(&b, &g, 4, &[]));
    assert_eq!(
        h.cached[&2], 0,
        "the KV past the last snapshot is not a hit"
    );
    assert!(
        snapshots(&h).iter().any(|s| s.0 == 12),
        "{:?}",
        snapshots(&h)
    );
    let mut c = shared;
    c.extend(rng.tokens(7, VOCAB));
    let out_c = run_one(&mut h, 3, c.clone(), g, 4, keyed(1));
    assert_eq!(out_c, h.expected(&c, &g, 4, &[]));
    assert_eq!(h.cached[&3], 12, "the branch point");
}

/// With drafts accepted in a step, a stop token among them (followed by further
/// accepted drafts) or a length limit inside them leaves the device past the end of the
/// turn; the continuation point still holds the state at the turn's own boundary.
#[test]
fn an_ending_inside_accepted_drafts_keeps_the_turns_own_state() {
    let cfg = MockConfig {
        draft_agreement: 1.0,
        ..MockConfig::default()
    };
    let g = SamplingParams::greedy();
    let mut saved = 0;
    for seed in 0..24u64 {
        for stop in [false, true] {
            let mut h = Harness::new(qspec(256, 64), sched(true), cfg);
            let mut rng = TestRng(seed + 300);
            let prompt = rng.var_tokens(6, 10, VOCAB);
            let free = h.expected(&prompt, &g, 24, &[]);
            let cut = 4 + (seed % 9) as usize;
            let mut first = request(1, prompt.clone(), g, cut as u32, keyed(1));
            if stop {
                if free[..cut - 1].contains(&free[cut - 1]) {
                    continue;
                }
                first.stop_token_ids = vec![free[cut - 1]];
                first.max_tokens = 24;
            }
            h.submit(first);
            h.run();
            h.flush();
            assert_eq!(h.outputs[&1], free[..cut]);
            saved += h.eng.kv().saved_states();
            let mut next = prompt.clone();
            next.extend(&h.outputs[&1]);
            next.extend(rng.tokens(2, VOCAB));
            let got = run_one(&mut h, 2, next.clone(), g, 6, keyed(1));
            assert_eq!(
                got,
                h.expected(&next, &g, 6, &[]),
                "seed {seed} stop {stop}"
            );
            let computed = (prompt.len() + cut - 1) as u32;
            assert_eq!(h.cached[&2], computed / BS * BS, "seed {seed} stop {stop}");
            let mut seqs = vec![next];
            seqs[0].extend(&h.outputs[&2]);
            assert_stored_states_are_states(&h, &seqs);
        }
    }
    assert!(
        saved > 0,
        "no ending fell inside accepted drafts past a boundary"
    );
}

#[test]
fn greedy_drafting_equals_plain_decoding_at_every_agreement() {
    for agreement in [0.0, 0.3, 0.7, 1.0] {
        let cfg = MockConfig {
            draft_agreement: agreement,
            ..MockConfig::default()
        };
        let mut h = Harness::new(qspec(512, 128), chunked(7, true), cfg);
        let mut rng = TestRng(77);
        let mut cases = Vec::new();
        for id in 0..8 {
            let prompt = rng.var_tokens(1, 30, VOCAB);
            cases.push((id, prompt.clone()));
            h.submit(request(
                id,
                prompt,
                SamplingParams::greedy(),
                25,
                keyed(id as u8),
            ));
        }
        h.run();
        for (id, prompt) in cases {
            assert_eq!(
                h.outputs[&id],
                h.expected(&prompt, &SamplingParams::greedy(), 25, &[]),
                "agreement {agreement} request {id}"
            );
        }
        assert!(h.eng.stats().drafted > 0);
    }
}

/// A sequence whose base nothing else holds stores each later boundary in place: one
/// buffer from its first boundary to its end.
#[test]
fn an_exclusive_base_is_written_in_place() {
    let mut h = Harness::new(qspec(256, 8), chunked(3, true), MockConfig::default());
    h.submit(request(
        1,
        TestRng(4).tokens(9, VOCAB),
        SamplingParams::greedy(),
        30,
        CacheScope::Private,
    ));
    let mut bases = BTreeSet::new();
    while h.eng.unfinished() > 0 {
        h.step();
        h.now += 1;
        if let Some(&(b, len)) = h.eng.kv().state_bases().values().next()
            && len > 0
        {
            bases.insert(b);
        }
    }
    assert_eq!(bases.len(), 1, "{bases:?}");
    assert!(h.eng.executor().stored_states().len() > 5);
}

/// A snapshot an entry holds is never written: after its regeneration point is attached,
/// the sequence's next boundary goes to a fresh buffer and the snapshot keeps its state.
#[test]
fn a_held_snapshot_is_never_overwritten() {
    let mut h = Harness::new(qspec(256, 16), chunked(256, false), MockConfig::default());
    let prompt = TestRng(5).tokens(10, VOCAB);
    let g = SamplingParams::greedy();
    h.submit(request(1, prompt.clone(), g, 20, keyed(1)));
    h.step();
    let (r, _, _) = snapshots(&h)[0];
    assert_eq!(r, 8);
    let regen = h.eng.kv().state_snapshots()[0].1;
    h.run();
    h.flush();
    // Decrypt-free check: the regeneration still resumes to the reference's output.
    let got = run_one(&mut h, 2, prompt.clone(), g, 20, keyed(1));
    assert_eq!(got, h.expected(&prompt, &g, 20, &[]));
    assert_eq!(h.cached[&2], 8);
    assert!(snapshots(&h).iter().any(|s| s.1 == regen && s.0 == 8));
}

#[test]
fn idle_snapshots_are_ciphertext_decrypted_only_by_their_lineage() {
    let mut h = Harness::new(qspec(256, 32), sched(true), MockConfig::default());
    let prompt = TestRng(6).tokens(13, VOCAB);
    let g = SamplingParams::greedy();
    run_one(&mut h, 1, prompt.clone(), g, 7, keyed(1));
    let p = h.eng.kv().state_pool().unwrap();
    let held = snapshots(&h);
    assert_eq!(held.len(), 2, "regeneration and continuation: {held:?}");
    for &(_, b, _) in &held {
        assert!(h.eng.executor().block_is_encrypted(p, b));
    }
    // Another lineage misses and decrypts nothing.
    let mark = h.eng.executor().crypt_log().len();
    run_one(&mut h, 2, prompt.clone(), g, 7, keyed(2));
    assert_eq!(h.cached[&2], 0);
    assert!(
        h.eng.executor().crypt_log()[mark..]
            .iter()
            .all(|&(_, grp, b, enc)| !(grp == p && !enc && held.iter().any(|s| s.1 == b))),
        "another lineage decrypted a snapshot"
    );
    // Its own lineage decrypts exactly the snapshot it resumes from.
    let mark = h.eng.executor().crypt_log().len();
    run_one(&mut h, 3, prompt.clone(), g, 7, keyed(1));
    assert_eq!(h.cached[&3], 12);
    let decrypted: Vec<u32> = h.eng.executor().crypt_log()[mark..]
        .iter()
        .filter(|&&(_, grp, _, enc)| grp == p && !enc)
        .map(|&(_, _, b, _)| b)
        .collect();
    let regen = held.iter().find(|s| s.0 == 12).unwrap().1;
    assert_eq!(decrypted, vec![regen]);
    assert_eq!(h.eng.kv_keys_held(), 0);
}

#[test]
fn a_private_request_keeps_no_snapshot() {
    let mut h = Harness::new(qspec(256, 32), sched(true), MockConfig::default());
    run_one(
        &mut h,
        1,
        TestRng(8).tokens(17, VOCAB),
        SamplingParams::greedy(),
        9,
        CacheScope::Private,
    );
    assert!(snapshots(&h).is_empty());
    let p = h.eng.kv().state_pool().unwrap();
    assert!(
        h.eng
            .kv()
            .free_block_set()
            .iter()
            .filter(|b| b.0 == p)
            .count()
            == 31
    );
    assert_eq!(h.eng.kv_keys_held(), 0);
}

#[test]
fn expired_snapshots_are_zeroed() {
    let mut cfg = sched(false);
    cfg.cache = CachePolicy {
        enabled: true,
        idle_ttl_ms: 100,
        max_age_ms: 10_000,
    };
    let mut h = Harness::new(qspec(256, 32), cfg, MockConfig::default());
    run_one(
        &mut h,
        1,
        TestRng(10).tokens(11, VOCAB),
        SamplingParams::greedy(),
        6,
        keyed(1),
    );
    let held = snapshots(&h);
    assert!(!held.is_empty());
    let p = h.eng.kv().state_pool().unwrap();
    h.now += 1000;
    h.eng.sweep(h.now).unwrap();
    h.check();
    assert!(snapshots(&h).is_empty());
    for &(_, b, _) in &held {
        assert!(h.eng.executor().block_is_zero(p, b));
        assert!(
            h.eng
                .executor()
                .zero_log()
                .iter()
                .any(|z| (z.1, z.2) == (p, b))
        );
    }
}

/// A snapshot's nonce comes from its attachment: the same entry's snapshot attached again
/// after the first was purged is encrypted under another nonce.
#[test]
fn no_two_snapshot_attachments_share_a_nonce() {
    let mut cfg = sched(false);
    cfg.cache = CachePolicy {
        enabled: true,
        idle_ttl_ms: 100,
        max_age_ms: 10_000,
    };
    let mut h = Harness::new(qspec(256, 32), cfg, MockConfig::default());
    let prompt = TestRng(11).tokens(11, VOCAB);
    let p = h.eng.kv().state_pool().unwrap();
    let mut nonces = Vec::new();
    for id in 0..2 {
        let mark = h.eng.executor().nonce_log().len();
        run_one(
            &mut h,
            id,
            prompt.clone(),
            SamplingParams::greedy(),
            6,
            keyed(1),
        );
        nonces.extend(
            h.eng.executor().nonce_log()[mark..]
                .iter()
                .filter(|n| n.1 == p)
                .map(|n| n.3),
        );
        h.now += 1000;
        h.eng.sweep(h.now).unwrap();
    }
    assert_eq!(nonces.len(), 4, "two snapshots a run");
    let distinct: BTreeSet<_> = nonces.iter().collect();
    assert_eq!(distinct.len(), nonces.len(), "a snapshot nonce reused");
}

#[test]
fn preemption_and_resumes_under_small_pools_match_the_reference() {
    for (blocks, buffers) in [(40, 128), (256, 5), (40, 4)] {
        for speculative in [false, true] {
            let mut h = Harness::new(
                qspec(blocks, buffers),
                chunked(5, speculative),
                MockConfig::default(),
            );
            let mut rng = TestRng(blocks as u64 * 31 + buffers as u64);
            let mut cases = Vec::new();
            for id in 0..12 {
                let prompt = rng.var_tokens(3, 25, VOCAB);
                cases.push((id, prompt.clone()));
                let cache = if id % 3 == 0 {
                    CacheScope::Private
                } else {
                    keyed(id as u8 % 4)
                };
                h.submit(request(id, prompt, SamplingParams::greedy(), 18, cache));
            }
            h.run();
            h.flush();
            let g = SamplingParams::greedy();
            for (id, prompt) in cases {
                assert_eq!(
                    h.outputs[&id],
                    h.expected(&prompt, &g, 18, &[]),
                    "blocks {blocks} buffers {buffers} speculative {speculative} {id}"
                );
            }
            assert!(
                h.eng.stats().preemptions > 0 || h.eng.kv().stripped_states() > 0,
                "blocks {blocks} buffers {buffers}: no pressure"
            );
            assert_eq!(h.eng.kv_keys_held(), 0);
        }
    }
}

/// `1 + 2S` buffers carry `S` concurrent sequences with no preemption.
#[test]
fn a_pool_of_one_plus_two_per_seat_never_preempts() {
    let seats = 12;
    let mut cfg = chunked(6, true);
    cfg.max_seqs = seats;
    let mut h = Harness::new(qspec(1024, 1 + 2 * seats), cfg, MockConfig::default());
    let mut rng = TestRng(12);
    let g = SamplingParams::greedy();
    let mut cases = Vec::new();
    for round in 0..2u64 {
        for i in 0..seats as u64 {
            let id = round * 100 + i;
            let mut prompt = rng.var_tokens(10, 30, VOCAB);
            if round == 1 {
                // Continuations: each resumes from a shared snapshot and needs a fresh
                // buffer at its first boundary.
                let (_, first): &(u64, Vec<u32>) = &cases[i as usize];
                prompt = first.clone();
                prompt.extend(&h.outputs[&i]);
                prompt.extend(rng.tokens(5, VOCAB));
            }
            cases.push((id, prompt.clone()));
            h.submit(request(id, prompt, g, 16, keyed(i as u8)));
        }
        h.run();
        h.flush();
    }
    for (id, prompt) in cases {
        assert_eq!(h.outputs[&id], h.expected(&prompt, &g, 16, &[]), "{id}");
    }
    assert_eq!(h.eng.stats().preemptions, 0);
}

/// A long conversation through a state pool far smaller than its retained snapshots:
/// each turn resumes from the one before, and the oldest snapshots are taken back.
#[test]
fn a_long_conversation_runs_through_a_small_state_pool() {
    for tier in [false, true] {
        let spec = if tier {
            with_host_tier(qspec(1024, 5), 6)
        } else {
            qspec(1024, 5)
        };
        let mut h = Harness::new(spec, sched(true), MockConfig::default());
        let mut rng = TestRng(13);
        let g = SamplingParams::greedy();
        let mut prompt = rng.tokens(6, VOCAB);
        for turn in 0..40u64 {
            let out = run_one(&mut h, turn, prompt.clone(), g, 6, keyed(1));
            assert_eq!(out, h.expected(&prompt, &g, 6, &[]), "turn {turn}");
            if turn > 0 {
                assert!(
                    h.cached[&turn] >= (prompt.len() as u32 - 10) / BS * BS,
                    "turn {turn}"
                );
            }
            prompt.extend(&out);
            prompt.extend(rng.tokens(3, VOCAB));
        }
        assert!(h.eng.kv().stripped_states() > 0 || h.eng.kv().tier_stats().demoted_blocks > 0);
        if tier {
            assert!(
                h.eng.kv().tier_stats().promoted_blocks > 0
                    || h.eng.kv().tier_stats().demoted_blocks > 0
            );
        }
    }
}

/// Snapshots last used together go branch first, then continuation, then regeneration;
/// older ones before newer.
#[test]
fn snapshots_are_reclaimed_oldest_first_and_regeneration_last() {
    // Three allocatable buffers. A finished keyed request holds its regeneration (8) and
    // continuation (12) snapshots; two private requests then need two buffers.
    let mut h = Harness::new(qspec(256, 4), chunked(256, false), MockConfig::default());
    let g = SamplingParams::greedy();
    run_one(&mut h, 1, TestRng(14).tokens(10, VOCAB), g, 6, keyed(1));
    let held: Vec<u32> = snapshots(&h).iter().map(|s| s.0).collect();
    assert_eq!(held, vec![8, 12]);
    h.submit(request(
        2,
        TestRng(15).tokens(9, VOCAB),
        g,
        3,
        CacheScope::Private,
    ));
    h.submit(request(
        3,
        TestRng(16).tokens(9, VOCAB),
        g,
        3,
        CacheScope::Private,
    ));
    h.step();
    let held: Vec<u32> = snapshots(&h).iter().map(|s| s.0).collect();
    assert_eq!(
        held,
        vec![8],
        "the continuation goes before the regeneration"
    );
    h.run();
    h.flush();

    // A second turn resumes from the first one's continuation: its regeneration is then
    // the oldest snapshot, and goes first.
    let mut h = Harness::new(qspec(256, 4), chunked(256, false), MockConfig::default());
    let prompt = TestRng(17).tokens(10, VOCAB);
    let out = run_one(&mut h, 1, prompt.clone(), g, 6, keyed(1));
    let mut next = prompt;
    next.extend(&out);
    next.extend(TestRng(18).tokens(6, VOCAB));
    h.now += 5;
    let got = run_one(&mut h, 2, next.clone(), g, 3, keyed(1));
    assert_eq!(got, h.expected(&next, &g, 3, &[]));
    assert_eq!(h.cached[&2], 12);
    let held: Vec<u32> = snapshots(&h).iter().map(|s| s.0).collect();
    assert!(
        !held.contains(&8),
        "the first turn's regeneration outlived newer snapshots: {held:?}"
    );
    assert!(held.contains(&12), "{held:?}");
}

#[test]
fn the_host_tier_keeps_snapshots_as_ciphertext() {
    let mut h = Harness::new(
        with_host_tier(qspec(256, 4), 16),
        sched(true),
        MockConfig::default(),
    );
    let g = SamplingParams::greedy();
    let mut rng = TestRng(19);
    let first = rng.tokens(10, VOCAB);
    run_one(&mut h, 1, first.clone(), g, 6, keyed(1));
    // Other lineages push the first one's snapshots to the host tier.
    for id in 2..6 {
        run_one(&mut h, id, rng.tokens(10, VOCAB), g, 6, keyed(id as u8));
    }
    let tier = h.eng.kv().tier_stats();
    assert!(tier.demoted_blocks > 0);
    let p = h.eng.kv().state_pool().unwrap();
    let demoted: Vec<_> = h
        .eng
        .executor()
        .transfer_log()
        .iter()
        .filter(|t| t.1 == p && t.4)
        .collect();
    assert!(!demoted.is_empty(), "no snapshot demoted");
    for &&(_, _, _, host, _) in &demoted {
        let image = h.eng.executor().host_state_image(host);
        assert!(
            !h.eng.executor().stored_states().contains(&image),
            "plaintext state on the host tier"
        );
    }
    // The first lineage's regeneration comes back from the host tier.
    let got = run_one(&mut h, 9, first.clone(), g, 6, keyed(1));
    assert_eq!(got, h.expected(&first, &g, 6, &[]));
    assert_eq!(h.cached[&9], 8);
    assert!(
        h.eng
            .executor()
            .transfer_log()
            .iter()
            .any(|t| t.1 == p && !t.4)
    );
}

#[test]
fn a_randomized_workload_keeps_every_invariant() {
    for seed in 0..6u64 {
        let tier = seed % 2 == 1;
        let base = qspec(48 + seed as u32 * 8, 4 + seed as u32);
        let spec = if tier { with_host_tier(base, 24) } else { base };
        let mut h = Harness::new(
            spec,
            chunked(1 + seed as u32 * 3, seed % 3 != 0),
            MockConfig::default(),
        );
        let mut rng = TestRng(seed + 1000);
        let mut prompts: Vec<(u64, Vec<u32>, SamplingParams, u32)> = Vec::new();
        for id in 0..30u64 {
            let mut prompt = rng.var_tokens(1, 30, VOCAB);
            if id > 0 && rng.chance(0.5) {
                let (_, earlier, _, _) = &prompts[rng.below(prompts.len() as u64) as usize];
                let keep = 1 + rng.below(earlier.len() as u64) as usize;
                let mut p = earlier[..keep].to_vec();
                p.extend(prompt);
                prompt = p;
            }
            let p = params(id);
            let max = 1 + rng.below(20) as u32;
            prompts.push((id, prompt.clone(), p, max));
            let cache = if rng.chance(0.3) {
                CacheScope::Private
            } else {
                keyed(rng.below(3) as u8)
            };
            h.submit(request(id, prompt, p, max, cache));
            if rng.chance(0.4) {
                h.step();
                h.now += 1;
            }
            if rng.chance(0.05)
                && let Some(&(victim, ..)) = prompts.get(rng.below(prompts.len() as u64) as usize)
                && h.eng.cancel(victim, h.now).is_some()
            {
                h.finished.insert(victim, FinishReason::Cancelled);
            }
        }
        h.run();
        h.flush();
        for (id, prompt, p, max) in prompts {
            if h.finished[&id] == FinishReason::Cancelled || (seed % 3 != 0 && !p.is_greedy()) {
                continue;
            }
            assert_eq!(
                h.outputs[&id],
                h.expected(&prompt, &p, max, &[]),
                "seed {seed} {id}"
            );
        }
        assert_eq!(h.eng.kv_keys_held(), 0);
        for slot in 0..32 {
            assert!(h.eng.executor().slot_state_is_clear(slot));
        }
        let _ = NULL_STATE;
    }
}

/// KV pressure demotes an entry's KV blocks and leaves its snapshot on the device; only
/// state pressure moves snapshots.
#[test]
fn kv_pressure_demotes_kv_and_leaves_snapshots() {
    let mut h = Harness::new(
        with_host_tier(qspec(24, 64), 64),
        sched(false),
        MockConfig::default(),
    );
    let g = SamplingParams::greedy();
    let mut rng = TestRng(20);
    for id in 0..6 {
        run_one(&mut h, id, rng.tokens(30, VOCAB), g, 4, keyed(id as u8));
    }
    let p = h.eng.kv().state_pool().unwrap();
    let log = h.eng.executor().transfer_log();
    assert!(log.iter().any(|t| t.1 != p && t.4), "no KV demoted");
    assert!(
        !log.iter().any(|t| t.1 == p),
        "a snapshot moved without state pressure"
    );
}
