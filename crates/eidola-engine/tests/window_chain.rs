//! The window-chain model (an encoder that runs every position, a decoder whose layers
//! run only where a target needs them) through the scheduler, KV manager and prefix
//! cache over the mock executor, against the dense reference that runs every layer at
//! every position. The mock fails a step that reads a chain cell no row computed, so a
//! triangle a row or a layer short, a stale window, or a decoder block shared where it
//! is not valid shows as a failed step or a wrong token.

mod common;

use std::collections::{BTreeSet, HashMap, HashSet};

use common::*;
use eidola_common::engine_protocol::{CACHED_PREFIX_QUANTUM, declarable_cached_tokens};
use eidola_engine::engine::{CacheScope, FinishReason, Request, SchedulerConfig};
use eidola_engine::mock::{MockConfig, deepseek_like_spec, with_host_tier};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::secret::{BootKey, CacheKey, SaltDeriver};
use eidola_engine::spec::{AttentionKind, Coverage, ModelSpec, WindowChain};

/// vocab 32, block 4, pooling ratio 2, window 3, depth 3, 32 slots, k = 5: the top layer
/// starts 3 positions before a target and the bottom one 7.
fn chain_spec(blocks: u32) -> ModelSpec {
    deepseek_like_spec(32, 4, 2, 3, 3, blocks, 32, 5)
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

/// Layer-positions of one triangle that ends at `target` and is not cut by the start of
/// the sequence or by rows that ran before it.
fn triangle(spec: &ModelSpec, target: u32) -> u64 {
    let depth = spec.window_chain.unwrap().depth;
    (0..depth)
        .map(|j| (target + 1 - spec.chain_layer_start(target - spec.chain_top(), j)) as u64)
        .sum()
}

#[test]
fn single_requests_match_the_dense_reference() {
    for spec_on in [false, true] {
        let mut h = Harness::new(chain_spec(256), sched(spec_on), MockConfig::default());
        let mut rng = TestRng(1);
        let mut cases = Vec::new();
        for id in 0..16 {
            let prompt = rng.var_tokens(1, 60, 32);
            let p = params(id);
            let max = 1 + rng.below(30) as u32;
            cases.push((id, prompt.clone(), p, max));
            let cache = if id % 2 == 0 {
                CacheScope::Private
            } else {
                CacheScope::Keyed(salt(id as u8))
            };
            h.submit(request(id, prompt, p, max, cache));
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

/// Prompts below, around and far past the chain's reach, keyed (a regeneration target
/// before the last token) and private, at chunk sizes that cut the triangles everywhere:
/// the same tokens, and the same chain cells computed, whatever the chunking.
#[test]
fn chunked_prefill_is_identical_to_unchunked() {
    let mut rng = TestRng(2);
    let lens = [1usize, 2, 3, 4, 6, 7, 8, 9, 11, 16, 23, 40, 77, 130];
    let prompts: Vec<Vec<u32>> = lens.iter().map(|&n| rng.tokens(n, 32)).collect();
    let mut results = Vec::new();
    for chunk in [1u32, 2, 3, 4, 5, 7, 16, 1000] {
        let mut out = Vec::new();
        for (i, prompt) in prompts.iter().enumerate() {
            for keyed in [false, true] {
                let mut h = Harness::new(
                    chain_spec(512),
                    chunked(chunk, false),
                    MockConfig::default(),
                );
                let cache = if keyed {
                    CacheScope::Keyed(salt(1))
                } else {
                    CacheScope::Private
                };
                let p = params(i as u64);
                h.submit(request(0, prompt.clone(), p, 9, cache));
                h.run();
                assert_eq!(
                    h.outputs[&0],
                    h.expected(prompt, &p, 9, &[]),
                    "chunk {chunk} length {} keyed {keyed}",
                    prompt.len()
                );
                out.push((h.outputs[&0].clone(), h.eng.executor().chain_writes()));
            }
        }
        results.push(out);
    }
    for r in &results[1..] {
        assert_eq!(r, &results[0]);
    }
}

/// A prompt far longer than the chain's reach runs the decoder over one triangle, not
/// over the prompt: exactly the triangle's layer-positions, then every layer at each
/// generated position.
#[test]
fn a_long_prompt_costs_one_triangle() {
    let spec = chain_spec(512);
    let depth = spec.window_chain.unwrap().depth as u64;
    for (len, chunk) in [(200usize, 1000u32), (200, 7), (57, 3)] {
        let mut h = Harness::new(spec.clone(), chunked(chunk, false), MockConfig::default());
        let prompt = TestRng(3).tokens(len, 32);
        let g = SamplingParams::greedy();
        h.submit(request(0, prompt.clone(), g, 5, CacheScope::Private));
        h.run();
        assert_eq!(h.outputs[&0], h.expected(&prompt, &g, 5, &[]));
        // Four generated tokens entered KV; the fifth ended the request.
        assert_eq!(
            h.eng.executor().chain_writes(),
            triangle(&spec, len as u32 - 1) + 4 * depth
        );
    }
}

/// The real model's proportions: 20 layers of window 128 under a drafter reading the 128
/// positions before its row, so the top layer starts 128 before a target and the bottom
/// one 2,541.
#[test]
fn real_proportions_run_one_exact_triangle() {
    let spec = deepseek_like_spec(32, 32, 2, 128, 20, 160, 2, 5);
    assert_eq!(spec.chain_top(), 128);
    assert_eq!(spec.chain_reach(), 2541);
    let mut h = Harness::new(spec.clone(), chunked(512, true), MockConfig::default());
    h.checks = false;
    let len = 3300;
    let prompt = TestRng(4).tokens(len, 32);
    let g = SamplingParams::greedy();
    h.submit(request(0, prompt.clone(), g, 12, CacheScope::Private));
    h.run();
    h.check_now();
    assert_eq!(h.outputs[&0], h.expected(&prompt, &g, 12, &[]));
    assert_eq!(triangle(&spec, len as u32 - 1), 26_710);
    let decode = h.eng.executor().chain_writes() - 26_710;
    assert!(decode > 0 && decode.is_multiple_of(20));
}

fn keyed(n: u8) -> CacheScope {
    CacheScope::Keyed(salt(n))
}

/// Submits a request, steps once, and returns its hit; then runs it to the end, checks
/// its output against the dense reference, and flushes.
fn run_checked(
    h: &mut Harness,
    id: u64,
    prompt: &[u32],
    p: SamplingParams,
    max: u32,
    cache: CacheScope,
) -> u32 {
    h.submit(request(id, prompt.to_vec(), p, max, cache));
    h.step();
    let cached = h
        .eng
        .cached_prompt_tokens(id)
        .or_else(|| h.cached.get(&id).copied())
        .expect("admitted in its first step");
    h.run();
    h.flush();
    assert_eq!(
        h.outputs[&id],
        h.expected(prompt, &p, max, &[]),
        "request {id}"
    );
    cached
}

/// Cached blocks per group.
fn cached_per_group(h: &Harness) -> Vec<usize> {
    let blocks = h.eng.kv().cache_blocks();
    (0..h.spec.kv_groups.len() as u32)
        .map(|g| blocks.iter().filter(|b| b.0 == g).count())
        .collect()
}

/// The three hit points resume with full-depth decoder and drafter windows: a
/// regeneration, a continuation, and (from the third request sharing a prefix on) the
/// branch. Each retained point holds every sliding group's window.
#[test]
fn hits_resume_at_regeneration_continuation_and_branch_points() {
    let g = SamplingParams::greedy();
    for (chunk, spec_on) in [(1000, false), (1000, true), (3, true), (1, true)] {
        let mut h = Harness::new(
            chain_spec(512),
            chunked(chunk, spec_on),
            MockConfig::default(),
        );
        let mut rng = TestRng(5);
        let prompt = rng.tokens(41, 32);
        assert_eq!(run_checked(&mut h, 1, &prompt, g, 6, keyed(1)), 0);
        // Windows of 3 (4 for the drafter) before boundaries 40 and 44: one block per
        // sliding group per point. The banks hold every sealed block.
        assert_eq!(cached_per_group(&h), [11, 2, 11, 2, 2]);
        assert_eq!(run_checked(&mut h, 2, &prompt, g, 6, keyed(1)), 40);
        let mut cont = prompt.clone();
        cont.extend(&h.outputs[&1]);
        cont.extend(rng.tokens(9, 32));
        // 41 + 6 - 1 positions entered KV: the last sealed boundary is 44.
        assert_eq!(run_checked(&mut h, 3, &cont, g, 6, keyed(1)), 44);

        // A short turn after a hit: 6 tokens past boundary 44 is further than the top
        // layer's 3 but within the bottom layer's 7, so every layer runs on from the
        // hit's windows, and the windows before 48 are whole when the request ends there.
        let mut turn = cont[..44].to_vec();
        turn.extend(rng.tokens(6, 32));
        assert_eq!(run_checked(&mut h, 8, &turn, g, 1, keyed(1)), 44);
        turn.extend(&h.outputs[&8]);
        turn.extend(rng.tokens(2, 32));
        assert_eq!(run_checked(&mut h, 9, &turn, g, 1, keyed(1)), 48);

        let base = rng.tokens(40, 32);
        let turn = |rng: &mut TestRng, first: u32| {
            let mut t = base[..20].to_vec();
            t.push(first);
            t.extend(rng.tokens(17, 32));
            t
        };
        assert_eq!(run_checked(&mut h, 4, &base, g, 6, keyed(2)), 0);
        // The first divergence finds the banks without windows, recomputes, and keeps
        // the branch (its triangle ends at the branch); the next one resumes there.
        let first = turn(&mut rng, (base[20] + 1) % 32);
        assert_eq!(run_checked(&mut h, 5, &first, g, 6, keyed(2)), 0);
        let second = turn(&mut rng, (base[20] + 2) % 32);
        assert_eq!(run_checked(&mut h, 6, &second, g, 6, keyed(2)), 20);
        // A branch hit one token before the end: the only row is the drafting target.
        let short = turn(&mut rng, (base[20] + 3) % 32)[..21].to_vec();
        assert_eq!(run_checked(&mut h, 7, &short, g, 6, keyed(2)), 20);
    }
}

/// A sequence that leaves before its chain is valid at full depth on its last sealed
/// window retains no hit point there: its decoder windows would be partial, and a
/// retained point carries every window or none. One that leaves after it does.
#[test]
fn a_point_without_full_depth_windows_is_not_retained() {
    let g = SamplingParams::greedy();
    let prompt = TestRng(6).tokens(30, 32);
    for (steps, hit) in [(5, 0), (9, 28)] {
        let mut h = Harness::new(chain_spec(512), chunked(4, false), MockConfig::default());
        h.submit(request(1, prompt.clone(), g, 40, keyed(1)));
        for _ in 0..steps {
            h.step();
        }
        // After five chunks of four, 20 positions are computed: the bottom chain layer
        // has just reached the regeneration target's triangle and the top has not.
        assert!(h.eng.cancel(1, h.now).is_some());
        h.flush();
        // No sliding group keeps a block for the abandoned point, the encoder's included.
        let c = cached_per_group(&h);
        assert_eq!([c[1] == 0, c[3] == 0, c[4] == 0], [hit == 0; 3]);
        assert_eq!(run_checked(&mut h, 2, &prompt, g, 4, keyed(1)), hit);
    }
}

/// A chain-covered block an entry holds from a shallower triangle is replaced when a
/// deeper one arrives for a point that needs it, so the point becomes a hit.
#[test]
fn a_deeper_block_replaces_a_shallower_one() {
    let g = SamplingParams::greedy();
    // Window 9 over blocks of 4: a point's windows span two (the drafter's, three) blocks.
    let spec = deepseek_like_spec(32, 4, 2, 9, 3, 512, 8, 5);
    let mut h = Harness::new(spec, sched(true), MockConfig::default());
    let mut rng = TestRng(7);
    let prompt = rng.tokens(30, 32);
    // Its regeneration point (28) attaches drafter blocks 4..7 valid from position 18.
    assert_eq!(run_checked(&mut h, 1, &prompt, g, 3, keyed(1)), 0);
    let mut other = prompt[..24].to_vec();
    other.push((prompt[24] + 1) % 32);
    other.extend(rng.tokens(9, 32));
    // Branches at 24, whose drafter window reaches back to 15: block 4 must be valid
    // from 16, and the one held is not.
    assert_eq!(run_checked(&mut h, 2, &other, g, 3, keyed(1)), 0);
    let mut third = prompt[..24].to_vec();
    third.push((prompt[24] + 2) % 32);
    assert_eq!(run_checked(&mut h, 3, &third, g, 3, keyed(1)), 24);
    // The first prompt's own points still hit.
    assert_eq!(run_checked(&mut h, 4, &prompt, g, 3, keyed(1)), 28);
}

#[test]
fn preemption_and_resume_reproduce_uninterrupted_outputs() {
    for (spec_on, chunk) in [(false, 256), (true, 256), (false, 3), (true, 5)] {
        // 40 blocks of 4 tokens per group: far less than 24 concurrent sequences need.
        // Small chunks preempt sequences in the middle of their triangles.
        let mut h = Harness::new(
            chain_spec(41),
            chunked(chunk, spec_on),
            MockConfig::default(),
        );
        let mut rng = TestRng(8);
        let mut cases = Vec::new();
        for id in 0..24 {
            let prompt = rng.var_tokens(5, 25, 32);
            let p = if spec_on {
                SamplingParams::greedy()
            } else {
                params(id)
            };
            cases.push((prompt.clone(), p));
            h.submit(request(id, prompt, p, 50, keyed(id as u8)));
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

/// Block size 2, windows 3 and 4, three allocatable blocks in the sliding groups: enough
/// for any one window plus the block being written, if blocks behind it are recycled.
fn tight_chain_spec() -> ModelSpec {
    let mut spec = deepseek_like_spec(32, 2, 2, 3, 3, 64, 4, 5);
    for g in &mut spec.kv_groups {
        if g.attention != AttentionKind::Full {
            g.num_blocks = 4;
        }
    }
    spec
}

/// Under KV pressure the chain sheds what every model sheds, in the same order: drafts,
/// the prefill chunk, then the sequence's own retention, whose triangles go with it. The
/// triangle its own tokens need is never shed: the output is the reference's.
#[test]
fn optional_reservations_are_shed_before_kv_runs_out() {
    let greedy = SamplingParams::greedy();
    for spec_on in [false, true] {
        for (prompt, chunk) in [
            (vec![1u32, 2, 3, 4, 5, 6, 7, 8], 8),
            (vec![1u32, 2, 3, 4, 5], 2),
            ((0..23).collect(), 8),
        ] {
            let mut h = Harness::new(
                tight_chain_spec(),
                chunked(chunk, spec_on),
                MockConfig::default(),
            );
            h.submit(request(1, prompt.clone(), greedy, 8, keyed(1)));
            h.run();
            assert_eq!(
                h.outputs[&1],
                h.expected(&prompt, &greedy, 8, &[]),
                "speculative {spec_on}, prompt {prompt:?}"
            );
            assert_eq!(h.finished[&1], FinishReason::Length);
            assert_eq!(h.eng.stats().preemptions, 0);
        }
    }
}

/// Where even plain decoding cannot grow, the sequence ends for lack of memory with the
/// tokens it produced; one block more and it drops its drafts instead.
#[test]
fn kv_pressure_drops_drafts_before_finishing() {
    let greedy = SamplingParams::greedy();
    for (blocks, prompt, max, produced, finish) in [
        (4, vec![7u32, 9], 2, 2, FinishReason::Length),
        (3, vec![7u32], 5, 2, FinishReason::KvExhausted),
    ] {
        let spec = deepseek_like_spec(32, 1, 1, 3, 3, blocks, 4, 5);
        let mut h = Harness::new(spec, sched(true), MockConfig::default());
        h.submit(request(1, prompt.clone(), greedy, max, CacheScope::Private));
        h.run();
        assert_eq!(h.outputs[&1], h.expected(&prompt, &greedy, produced, &[]));
        assert_eq!(h.finished[&1], finish);
        assert_eq!(h.eng.stats().drafted, 0);
    }
}

/// Greedy speculation with five-slot block drafts never changes a token, at every
/// agreement rate and wherever the first drafting row falls (a chunk of one makes it the
/// prompt's own last token, whose drafter context reaches furthest back).
#[test]
fn greedy_speculation_is_exact_at_every_agreement_rate() {
    for agreement in [0.0, 0.3, 0.8, 1.0] {
        for chunk in [1, 4, 256] {
            let cfg = MockConfig {
                draft_agreement: agreement,
                ..MockConfig::default()
            };
            let mut h = Harness::new(chain_spec(256), chunked(chunk, true), cfg);
            let mut rng = TestRng(9);
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
            } else if agreement == 0.0 {
                assert!(s.accepted < s.drafted);
            }
        }
    }
}

/// Sampled acceptance over five-slot drafts whose every slot is conditioned on the token
/// drafted before it preserves the output distribution: the joint distribution of the
/// first three tokens (vocab 4, 64 cells) over 20k seeds each, two-sample chi-square at
/// p = 0.001 (critical value 103.4, df 63).
#[test]
fn speculative_sampling_preserves_the_distribution() {
    let spec = deepseek_like_spec(4, 4, 2, 3, 3, 256, 32, 5);
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

/// Checks that no two attachments of a chain-covered block to a cache entry are ever
/// encrypted under one nonce. An attachment is told apart by the harness alone: a
/// (group, block) pair from the step it first appears among the cache's chain-covered
/// blocks until it leaves them (a block given up and attached again within one step
/// counts as one: its bytes did not change).
#[derive(Default)]
struct NonceWatch {
    mark: usize,
    tenure: HashMap<(u32, u32), u64>,
    tenures: u64,
    tenure_of: HashMap<[u32; 3], u64>,
}

impl NonceWatch {
    fn observe(&mut self, h: &Harness) {
        let log = h.eng.executor().nonce_log();
        for &(_, g, b, nonce) in &log[self.mark..] {
            if h.spec.kv_groups[g as usize].coverage != Coverage::Chain {
                continue;
            }
            // Attached since the last observation (a release between two steps) if new.
            let t = *self.tenure.entry((g, b)).or_insert_with(|| {
                self.tenures += 1;
                self.tenures
            });
            assert_eq!(
                *self.tenure_of.entry(nonce).or_insert(t),
                t,
                "two attachments of group {g} share a nonce"
            );
        }
        self.mark = log.len();
        let held: BTreeSet<(u32, u32)> = h
            .eng
            .kv()
            .chain_attachments()
            .iter()
            .map(|a| (a.0, a.1))
            .collect();
        self.tenure.retain(|k, _| held.contains(k));
        for k in held {
            self.tenure.entry(k).or_insert_with(|| {
                self.tenures += 1;
                self.tenures
            });
        }
    }

    fn run(&mut self, h: &mut Harness) {
        while h.eng.unfinished() > 0 {
            h.step();
            h.now += 1;
            self.observe(h);
        }
        h.flush();
        self.observe(h);
    }
}

/// A chain-covered block's bytes depend on the triangle its sequence ran, so the same
/// entry and group can hold different bytes at different times: each attachment is
/// encrypted under its own nonce, across an expiry that re-creates the entry and across
/// a sequence giving its retention up and a later one attaching again.
#[test]
fn no_two_attachments_share_a_nonce() {
    let g = SamplingParams::greedy();
    let mut cfg = sched(true);
    cfg.cache.idle_ttl_ms = 500;
    let mut h = Harness::new(chain_spec(512), cfg, MockConfig::default());
    let mut watch = NonceWatch::default();
    let prompt = TestRng(10).tokens(41, 32);
    for round in 0..4u64 {
        // The same prompt under the same key: the same hashes every round.
        h.submit(request(round, prompt.clone(), g, 6, keyed(1)));
        watch.run(&mut h);
        assert_eq!(h.cached[&round], 0, "the entries expired");
        h.now += 1000;
        h.eng.sweep(h.now).unwrap();
        watch.observe(&h);
        assert_eq!(h.eng.kv().cache_entries(), 0);
    }
    // Two points a round (regeneration and continuation), two chain-covered groups.
    assert_eq!(watch.tenure_of.len(), 16);

    // Retention given up under pressure and attached again by the next request.
    let mut h = Harness::new(tight_chain_spec(), chunked(2, false), MockConfig::default());
    let mut watch = NonceWatch::default();
    let prompt: Vec<u32> = (0..23).collect();
    for round in 0..4u64 {
        h.submit(request(
            round,
            prompt[..5 + 6 * round as usize].to_vec(),
            g,
            8,
            keyed(1),
        ));
        watch.run(&mut h);
        assert_eq!(h.outputs[&round].len(), 8);
    }
    assert!(watch.tenure_of.len() >= 8, "{}", watch.tenure_of.len());
}

/// A finish leaves the cached blocks of every group as ciphertext, the decoder's and
/// drafter's windows among them, and a hit decrypts exactly what it maps.
#[test]
fn idle_blocks_of_every_group_are_ciphertext() {
    let g = SamplingParams::greedy();
    let mut h = Harness::new(chain_spec(512), sched(true), MockConfig::default());
    let prompt = TestRng(11).tokens(41, 32);
    assert_eq!(run_checked(&mut h, 1, &prompt, g, 6, keyed(1)), 0);
    let cached = h.eng.kv().cache_blocks();
    assert_eq!(
        cached.iter().map(|b| b.0).collect::<BTreeSet<_>>(),
        BTreeSet::from([0, 1, 2, 3, 4])
    );
    for &(group, block) in &cached {
        assert!(h.eng.executor().block_is_encrypted(group, block));
    }
    assert_eq!(h.eng.kv_keys_held(), 0);
    // Another lineage decrypts nothing.
    let mark = h.eng.executor().crypt_log().len();
    assert_eq!(run_checked(&mut h, 2, &prompt, g, 6, keyed(2)), 0);
    let decrypted = |h: &Harness, mark: usize| -> BTreeSet<(u32, u32)> {
        h.eng.executor().crypt_log()[mark..]
            .iter()
            .filter(|e| !e.3)
            .map(|e| (e.1, e.2))
            .collect()
    };
    assert!(decrypted(&h, mark).is_disjoint(&cached));
    // The regeneration hit maps every bank block and one window block per sliding group.
    let mark = h.eng.executor().crypt_log().len();
    h.submit(request(3, prompt.clone(), g, 20, keyed(1)));
    h.step();
    assert_eq!(h.eng.cached_prompt_tokens(3), Some(40));
    let d = decrypted(&h, mark);
    assert!(d.is_subset(&cached));
    let per_group: Vec<usize> = (0..5)
        .map(|g| d.iter().filter(|b| b.0 == g).count())
        .collect();
    assert_eq!(per_group, [10, 1, 10, 1, 1]);
    h.run();
    h.flush();
    assert_eq!(h.outputs[&3], h.expected(&prompt, &g, 20, &[]));
    for slot in 0..h.spec.num_state_slots {
        assert!(h.eng.executor().slot_state_is_clear(slot));
    }
}

/// Device pressure demotes a lineage's entries, decoder and drafter windows included, as
/// the ciphertext they are; the hit that brings them back equals an undisturbed one.
#[test]
fn chain_windows_survive_demotion() {
    let g = SamplingParams::greedy();
    let mut h = Harness::new(
        with_host_tier(chain_spec(28), 160),
        sched(true),
        MockConfig::default(),
    );
    let mut rng = TestRng(12);
    let prompt = rng.tokens(41, 32);
    assert_eq!(run_checked(&mut h, 1, &prompt, g, 6, keyed(1)), 0);
    let first = h.eng.kv().cache_blocks();
    let mark = h.eng.executor().transfer_log().len();
    for i in 0..4u64 {
        let other = rng.tokens(37, 32);
        run_checked(&mut h, 10 + i, &other, g, 5, keyed(100 + i as u8));
    }
    let demoted: BTreeSet<(u32, u32)> = h.eng.executor().transfer_log()[mark..]
        .iter()
        .filter(|t| t.4)
        .map(|t| (t.1, t.2))
        .collect();
    assert!(first.iter().all(|b| demoted.contains(b)), "all demoted");
    let stats = h.eng.kv().tier_stats();
    assert_eq!((stats.host_drops, stats.device_drops), (0, 0));
    let mark = h.eng.executor().transfer_log().len();
    assert_eq!(run_checked(&mut h, 2, &prompt, g, 6, keyed(1)), 40);
    let promoted: BTreeSet<u32> = h.eng.executor().transfer_log()[mark..]
        .iter()
        .filter(|t| !t.4)
        .map(|t| t.1)
        .collect();
    assert_eq!(promoted, BTreeSet::from([0, 1, 2, 3, 4]));
    assert_eq!(h.outputs[&2], h.outputs[&1]);
}

fn requiring(id: u64, prompt: Vec<u32>, max: u32, cache: CacheScope, required: u32) -> Request {
    Request {
        required_cached_tokens: required,
        ..request(id, prompt, SamplingParams::greedy(), max, cache)
    }
}

/// The side-effect-free lookup agrees with admission, including where a matched
/// boundary has banks but no full-depth windows: requiring exactly the lookup's answer is
/// admitted with that hit, and requiring one token more is refused with the KV manager
/// and the executor untouched.
#[test]
fn a_required_prefix_is_judged_as_admission_would() {
    let g = SamplingParams::greedy();
    let mut cfg = chunked(4, false);
    cfg.sweep_interval_ms = u64::MAX;
    let mut h = Harness::new(chain_spec(512), cfg, MockConfig::default());
    let mut rng = TestRng(13);
    let done = rng.tokens(41, 32);
    run_checked(&mut h, 1, &done, g, 6, keyed(1));
    // Cancelled in the middle of its triangle: banks sealed to 20, no windows there.
    let cut = rng.tokens(30, 32);
    h.submit(request(2, cut.clone(), g, 40, keyed(1)));
    for _ in 0..5 {
        h.step();
    }
    h.eng.cancel(2, h.now);
    h.eng.sweep(h.now).unwrap();
    h.check();
    let mut extended = done.clone();
    extended.extend(rng.tokens(7, 32));
    let mut continued = done.clone();
    continued.extend(&h.outputs[&1]);
    continued.extend(rng.tokens(3, 32));
    let cases = [
        (done.clone(), 40),
        (extended, 40),
        (continued, 44),
        (cut, 0),
    ];
    let mut id = 10;
    for (prompt, hit) in cases {
        let looked = h.eng.kv().lookup(&salt(1), &prompt, h.now);
        assert_eq!(looked, hit);
        if hit + 1 < prompt.len() as u32 {
            let before = (format!("{:?}", h.eng.kv()), h.eng.executor().steps());
            h.submit(requiring(id, prompt.clone(), 4, keyed(1), hit + 1));
            h.step();
            assert_eq!(h.finished[&id], FinishReason::CachedPrefixMissing);
            let after = (format!("{:?}", h.eng.kv()), h.eng.executor().steps());
            assert_eq!(before, after, "the refused request touched something");
            id += 1;
        }
        h.submit(requiring(id, prompt.clone(), 4, keyed(1), hit));
        h.step();
        assert_eq!(h.eng.cached_prompt_tokens(id), Some(hit));
        // Cancelled before it retains anything the next case could hit.
        h.eng.cancel(id, h.now);
        h.eng.sweep(h.now).unwrap();
        h.check();
        id += 1;
    }
}

/// A lineage per case, so every case meets only its own entries.
fn case_scope(case: u64) -> CacheScope {
    let mut key = [0u8; 32];
    key[..8].copy_from_slice(&case.to_le_bytes());
    CacheScope::Keyed(
        SaltDeriver::with_boot_key(BootKey::from_bytes([7; 32]))
            .lineage(&CacheKey::from_bytes(key)),
    )
}

/// The rule a client declares a cached prefix by lands on a retained point of the chain
/// model too: after a keyed request of prompt `n` that generated `c` tokens, its prompt
/// followed by exactly its output (requiring `declarable_cached_tokens(n + c)`) and its
/// prompt extended by a turn that diverges at once (requiring
/// `declarable_cached_tokens(n)`) are both admitted. Block 16, a chain reaching 118
/// positions back, so every prompt here runs a triangle; every residue of `n` and of
/// `n + c`, with and without drafts, in even and uneven chunks.
#[test]
fn the_declaration_rule_holds_for_the_chain() {
    const VOCAB: u32 = 4096;
    let spec = deepseek_like_spec(VOCAB, 16, 2, 40, 3, 256, 8, 5);
    assert_eq!(spec.chain_reach(), 118);
    let g = SamplingParams::greedy();
    let mut failures = Vec::new();
    let mut residues = BTreeSet::new();
    for (speculative, chunk) in [(false, 256), (true, 256), (true, 100)] {
        for n in 257..273u32 {
            for c in 1..=16u32 {
                let case = u64::from(n) << 32 | u64::from(c) << 8 | u64::from(chunk);
                let mut h = Harness::new(
                    spec.clone(),
                    chunked(chunk, speculative),
                    MockConfig::default(),
                );
                h.checks = false;
                let mut rng = TestRng(case);
                let prompt = rng.tokens(n as usize, VOCAB);
                h.submit(request(1, prompt.clone(), g, c, case_scope(case)));
                h.run();
                h.flush();
                h.check_now();
                let output = h.outputs[&1].clone();
                assert_eq!(output.len() as u32, c);
                residues.insert((speculative, n % 16, (n + c) % 16));
                let mut admit = |h: &mut Harness, id: u64, tokens: Vec<u32>, resumed: u32| {
                    let required = declarable_cached_tokens(u64::from(resumed));
                    assert!(required >= CACHED_PREFIX_QUANTUM);
                    h.submit(requiring(id, tokens, 64, case_scope(case), required));
                    h.step();
                    let hit = match h.finished.get(&id) {
                        None => h.eng.cached_prompt_tokens(id),
                        Some(FinishReason::CachedPrefixMissing) => None,
                        Some(_) => h.cached.get(&id).copied(),
                    };
                    if !hit.is_some_and(|hit| hit >= required) {
                        failures.push(format!(
                            "speculative {speculative} chunk {chunk} n {n} c {c}: \
                             requiring {required}, hit {hit:?}"
                        ));
                    }
                    h.eng.cancel(id, h.now);
                    h.flush();
                    h.now += 1;
                };
                let mut continued = prompt.clone();
                continued.extend(&output);
                admit(&mut h, 2, continued, n + c);
                let mut extended = prompt;
                extended.push((output[0] + 1) % VOCAB);
                extended.extend(rng.tokens(5, VOCAB));
                admit(&mut h, 3, extended, n);
            }
        }
    }
    assert_eq!(residues.len(), 2 * 16 * 16);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The spec's consistency rules for pooling and the chain.
#[test]
fn inconsistent_chain_specs_are_refused() {
    let good = chain_spec(64);
    assert_eq!(good.validate(), Ok(()));
    assert_eq!(
        (good.chain_top(), good.chain_reach(), good.num_devices()),
        (3, 7, 2)
    );
    let broken = |edit: &dyn Fn(&mut ModelSpec)| {
        let mut spec = good.clone();
        edit(&mut spec);
        spec.validate().is_err()
    };
    // A ratio that does not divide the block, so an entry would straddle two blocks.
    assert!(broken(&|s| s.kv_groups[0].ratio = 3));
    assert!(broken(&|s| s.kv_groups[0].ratio = 0));
    // Pooling a sliding window.
    assert!(broken(&|s| s.kv_groups[1].ratio = 2));
    // A chain-covered bank, chain-covered groups without a chain, a chain without its
    // group, and a chain whose window is not its group's.
    assert!(broken(&|s| s.kv_groups[2].coverage = Coverage::Chain));
    assert!(broken(&|s| s.window_chain = None));
    assert!(broken(&|s| s.kv_groups[3].coverage = Coverage::All));
    assert!(broken(&|s| s.window_chain = Some(WindowChain {
        depth: 3,
        window: 5
    })));
    assert!(broken(&|s| s.window_chain = Some(WindowChain {
        depth: 0,
        window: 3
    })));
}

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
    let early = h.finished.get(&id) == Some(&FinishReason::KvExhausted);
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

/// Randomized workload over the chain model: multi-turn keyed conversations
/// (continuations, regenerations, branches off a shared head), private requests, random
/// sampling, five-slot drafting on or off, cancellations, clock jumps past the idle TTL,
/// pools small enough to force eviction and preemption in the middle of triangles, and
/// on every third seed a host tier. Every step checks the KV invariants, that every free
/// block is zero or has a zero queued, that idle blocks are ciphertext, and (without a
/// host tier) that no two attachments share a nonce. Every completed request equals the
/// dense reference, and a hit never exceeds what its lineage computed.
#[test]
fn randomized_workloads_hold_every_invariant() {
    let mut preempted = 0;
    for seed in 0..24u64 {
        let mut rng = TestRng(300 + seed);
        let spec_on = seed % 2 == 0;
        let tiered = seed % 3 == 2;
        let blocks = 48 + rng.below(80) as u32;
        let mut cfg = sched(spec_on);
        cfg.max_prefill_chunk = 1 + rng.below(40) as u32;
        cfg.max_batched_tokens = 32 + rng.below(100) as u32;
        cfg.cache.idle_ttl_ms = 500;
        cfg.cache.max_age_ms = 2000;
        cfg.sweep_interval_ms = 50;
        let spec = if tiered {
            with_host_tier(chain_spec(blocks), 2 * blocks)
        } else {
            chain_spec(blocks)
        };
        let mut h = Harness::new(spec, cfg, MockConfig::default());
        let mut watch = NonceWatch::default();

        let mut convo: HashMap<u8, Vec<u32>> = HashMap::new();
        let mut history: HashMap<u8, Vec<Vec<u32>>> = HashMap::new();
        let mut live: HashMap<u64, (Vec<u32>, SamplingParams, u32, Option<u8>)> = HashMap::new();
        let mut hit_checked = HashSet::new();
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
                        Some(t) if rng.chance(0.3) => {
                            // A branch off an earlier point of the conversation.
                            let mut t = t[..1 + rng.below(t.len() as u64) as usize].to_vec();
                            t.extend(rng.var_tokens(1, 12, 32));
                            t
                        }
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
                    Some(s) => keyed(s + 1),
                    None => CacheScope::Private,
                };
                live.insert(id, (prompt.clone(), p, max, key));
                h.submit(request(id, prompt, p, max, cache));
            }
            if rng.chance(0.03) && !live.is_empty() {
                let ids: Vec<u64> = live.keys().copied().collect();
                let victim = ids[rng.below(ids.len() as u64) as usize];
                if h.eng.cancel(victim, h.now).is_some() {
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
            if !tiered {
                watch.observe(&h);
            }
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
        if tiered {
            h.run();
        } else {
            watch.run(&mut h);
        }
        for (id, (prompt, p, max, _)) in live {
            check_output(&h, id, &prompt, &p, max, spec_on);
        }
        assert!(
            hits_seen > 0,
            "seed {seed}: workload produced no cache hits"
        );
        preempted += h.eng.stats().preemptions;
        eprintln!(
            "seed {seed}: blocks {blocks} requests {next_id} stats {:?} hits {hits_seen} \
             attachments {} tier {:?}",
            h.eng.stats(),
            watch.tenure_of.len(),
            h.eng.kv().tier_stats(),
        );
        // Drain: everything expires, everything is freed and zeroed, no slot keeps state.
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
        for slot in 0..h.spec.num_state_slots {
            assert!(h.eng.executor().slot_state_is_clear(slot));
        }
    }
    assert!(preempted > 0, "no seed forced a preemption");
}
