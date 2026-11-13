//! A required cached prefix (`Request::required_cached_tokens`): checked once, at a
//! request's first admission, by a lookup that changes nothing. A request whose
//! requirement holds runs as any other; one whose requirement does not is refused
//! (`FinishReason::CachedPrefixMissing`) with no executor work, no maintenance and no
//! cache state attributable to it. And the rule a client declares by
//! (`eidola_common::engine_protocol::declarable_cached_tokens`) is one the engine's
//! retention points always satisfy.

mod common;

use common::*;
use eidola_common::engine_protocol::{CACHED_PREFIX_QUANTUM, declarable_cached_tokens};
use eidola_engine::engine::{CacheScope, FinishReason, Request, SchedulerConfig, SubmitError};
use eidola_engine::mock::{MockConfig, mimo_like_spec, with_host_tier};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::secret::{BootKey, CacheKey, SaltDeriver};
use eidola_engine::spec::ModelSpec;

fn keyed(n: u8) -> CacheScope {
    CacheScope::Keyed(salt(n))
}

fn requiring(id: u64, prompt: Vec<u32>, max: u32, cache: CacheScope, required: u32) -> Request {
    Request {
        required_cached_tokens: required,
        ..request(id, prompt, SamplingParams::greedy(), max, cache)
    }
}

/// Everything a refused request could have touched: the KV manager's whole state (its
/// entries, their LRU and retention indexes, block pools, slots and every pending
/// operation) and the executor's work and maintenance logs.
#[derive(Debug, PartialEq, Eq)]
struct Footprint {
    kv: String,
    steps: u64,
    executor_steps: u64,
    zeros: usize,
    crypts: usize,
    transfers: usize,
    host_zeros: usize,
}

fn footprint(h: &Harness) -> Footprint {
    let x = h.eng.executor();
    Footprint {
        kv: format!("{:?}", h.eng.kv()),
        steps: h.eng.stats().steps,
        executor_steps: x.steps(),
        zeros: x.zero_log().len(),
        crypts: x.crypt_log().len(),
        transfers: x.transfer_log().len(),
        host_zeros: x.host_zero_log().len(),
    }
}

/// A scheduler that sweeps only when told to, so expired entries stay in the cache until
/// a lookup or a sweep meets them.
fn manual_sweep() -> SchedulerConfig {
    let mut cfg = sched(false);
    cfg.sweep_interval_ms = u64::MAX;
    cfg
}

/// The 40-token prompt `P` the fixtures cache under key 1 (block 4: its regeneration point
/// is at 36, and after 4 generated tokens its continuation point is at 40), and the
/// 48-token prompt `Q` extending it with a turn that diverges from `P`'s output at once,
/// whose hit under key 1 is therefore 40.
struct Fixture {
    h: Harness,
    p: Vec<u32>,
    q: Vec<u32>,
    finished_at: u64,
}

fn fixture(spec: ModelSpec, cfg: SchedulerConfig) -> Fixture {
    let mut h = Harness::new(spec, cfg, MockConfig::default());
    let mut rng = TestRng(41);
    let p = rng.tokens(40, 32);
    h.submit(request(0, p.clone(), SamplingParams::greedy(), 4, keyed(1)));
    h.run();
    let finished_at = h.now - 1;
    let mut q = p.clone();
    let turn = rng.tokens(8, 32);
    q.extend(&turn);
    if q[40] == h.outputs[&0][0] {
        q[40] = (q[40] + 1) % 32;
    }
    // Run the release's maintenance, and record a sweep, so the next step neither sweeps
    // nor flushes on the refused request's account.
    h.eng.sweep(h.now).unwrap();
    h.check();
    Fixture {
        h,
        p,
        q,
        finished_at,
    }
}

/// Submits `req`, steps once, and asserts it was refused for its prefix with nothing it
/// could have touched changed.
fn assert_refused_untouched(h: &mut Harness, req: Request) {
    let id = req.id;
    let before = footprint(h);
    let refusals = h.eng.stats().prefix_refusals;
    h.submit(req);
    h.step();
    assert_eq!(
        h.finished.get(&id),
        Some(&FinishReason::CachedPrefixMissing)
    );
    assert!(h.outputs[&id].is_empty());
    assert_eq!(h.eng.stats().prefix_refusals, refusals + 1);
    assert_eq!(
        footprint(h),
        before,
        "the refused request touched something"
    );
    assert_eq!(h.eng.unfinished(), 0);
    assert_eq!(h.eng.kv_keys_held(), 0);
}

#[test]
fn a_requirement_the_cache_holds_runs_like_any_request() {
    for required in [4, 36, 40] {
        let Fixture { mut h, q, .. } = fixture(small_spec(512), manual_sweep());
        h.submit(requiring(1, q.clone(), 6, keyed(1), required));
        h.run();
        assert_eq!(h.finished[&1], FinishReason::Length, "required {required}");
        assert_eq!(h.cached[&1], 40, "required {required}");
        assert_eq!(
            h.outputs[&1],
            h.expected(&q, &SamplingParams::greedy(), 6, &[])
        );
        assert_eq!(h.eng.stats().prefix_refusals, 0);
    }
}

#[test]
fn a_requirement_one_token_past_the_hit_is_refused_untouched() {
    let Fixture { mut h, q, .. } = fixture(small_spec(512), manual_sweep());
    assert_refused_untouched(&mut h, requiring(1, q, 6, keyed(1), 41));
}

#[test]
fn a_prefix_never_cached_is_refused_untouched() {
    let Fixture { mut h, q, .. } = fixture(small_spec(512), manual_sweep());
    assert_refused_untouched(&mut h, requiring(1, q, 6, keyed(9), 4));
}

#[test]
fn another_keys_entry_is_refused_untouched() {
    let Fixture { mut h, q, .. } = fixture(small_spec(512), manual_sweep());
    assert_refused_untouched(&mut h, requiring(1, q, 6, keyed(2), 36));
}

/// A request without a key has a fresh lineage, which nothing is cached under.
#[test]
fn a_private_request_is_refused_untouched() {
    let Fixture { mut h, q, .. } = fixture(small_spec(512), manual_sweep());
    assert_refused_untouched(&mut h, requiring(1, q, 6, CacheScope::Private, 4));
}

#[test]
fn a_prefix_cut_short_by_a_divergence_is_refused_untouched() {
    // Full attention only, so a hit needs no retained window and can end at any cached
    // boundary.
    let mut spec = small_spec(512);
    spec.kv_groups
        .retain(|g| g.attention == eidola_engine::spec::AttentionKind::Full);
    let Fixture { mut h, p, .. } = fixture(spec, manual_sweep());
    // Diverges inside block 5: the hit is 20.
    let mut r = p.clone();
    r[22] = (r[22] + 1) % 32;
    assert_refused_untouched(&mut h, requiring(1, r.clone(), 6, keyed(1), 24));
    h.submit(requiring(2, r, 6, keyed(1), 20));
    h.run();
    assert_eq!(h.finished[&2], FinishReason::Length);
}

/// Expired by its idle TTL, before any sweep: the lookup neither hits the entry nor
/// removes it (a sweep does that, later).
#[test]
fn an_entry_past_its_idle_ttl_is_refused_untouched() {
    let cfg = manual_sweep();
    let ttl = cfg.cache.idle_ttl_ms;
    let Fixture {
        mut h,
        q,
        finished_at,
        ..
    } = fixture(small_spec(512), cfg);
    let entries = h.eng.kv().cache_entries();
    h.now = finished_at + ttl;
    assert_refused_untouched(&mut h, requiring(1, q.clone(), 6, keyed(1), 36));
    assert_eq!(
        h.eng.kv().cache_entries(),
        entries,
        "the lookup removed nothing"
    );
    // One millisecond earlier it was a hit.
    let Fixture {
        mut h,
        q,
        finished_at,
        ..
    } = fixture(small_spec(512), manual_sweep());
    h.now = finished_at + ttl - 1;
    h.submit(requiring(1, q, 6, keyed(1), 36));
    h.run();
    assert_eq!(h.finished[&1], FinishReason::Length);
}

/// A refused lookup does not count as a use: the entry still expires on the schedule its
/// last real use set.
#[test]
fn a_refused_lookup_does_not_keep_an_entry_warm() {
    let cfg = manual_sweep();
    let ttl = cfg.cache.idle_ttl_ms;
    let Fixture {
        mut h,
        q,
        finished_at,
        ..
    } = fixture(small_spec(512), cfg);
    h.now = finished_at + ttl - 1;
    // Wants one block more than is held, so it walks the whole cached chain.
    assert_refused_untouched(&mut h, requiring(1, q.clone(), 6, keyed(1), 44));
    h.now = finished_at + ttl;
    h.submit(request(2, q, SamplingParams::greedy(), 6, keyed(1)));
    h.step();
    assert_eq!(h.eng.cached_prompt_tokens(2), Some(0));
}

/// Expired by its max age while its idle TTL still runs (kept warm by regenerations).
#[test]
fn an_entry_past_its_max_age_is_refused_untouched() {
    let cfg = manual_sweep();
    let (ttl, max_age) = (cfg.cache.idle_ttl_ms, cfg.cache.max_age_ms);
    let Fixture { mut h, p, q, .. } = fixture(small_spec(512), cfg);
    let mut id = 10;
    while h.now + ttl / 2 < max_age {
        h.now += ttl / 2;
        h.submit(request(
            id,
            p.clone(),
            SamplingParams::greedy(),
            4,
            keyed(1),
        ));
        h.run();
        assert_eq!(h.cached[&id], 36, "a warm hit");
        id += 1;
    }
    h.eng.sweep(h.now).unwrap();
    h.check();
    // Past the max age of the entries the fixture created in its first steps.
    h.now = max_age + 10;
    assert_refused_untouched(&mut h, requiring(1, q, 6, keyed(1), 36));
}

/// Evicted under pressure by another key's request.
#[test]
fn an_evicted_entry_is_refused_untouched() {
    // 40 blocks a group: P's entries (10 full blocks) do not survive a 120-token prompt.
    let Fixture { mut h, q, .. } = fixture(small_spec(40), manual_sweep());
    let other = TestRng(42).tokens(120, 32);
    h.submit(request(5, other, SamplingParams::greedy(), 4, keyed(2)));
    h.run();
    h.eng.sweep(h.now).unwrap();
    h.check();
    assert_refused_untouched(&mut h, requiring(1, q, 6, keyed(1), 4));
}

/// The refused request leaves the requests stepped with it exactly as they would be
/// without it: outputs, hits and endings.
#[test]
fn a_refusal_leaves_the_rest_of_its_step_alone() {
    let run = |with_refused: bool| {
        let Fixture { mut h, q, .. } = fixture(small_spec(512), manual_sweep());
        let fresh = TestRng(43).tokens(30, 32);
        h.submit(request(1, q.clone(), SamplingParams::greedy(), 6, keyed(1)));
        if with_refused {
            h.submit(requiring(2, q.clone(), 6, keyed(2), 36));
        }
        h.submit(requiring(3, q.clone(), 6, keyed(1), 40));
        h.submit(request(
            4,
            fresh.clone(),
            SamplingParams::greedy(),
            6,
            keyed(3),
        ));
        h.run();
        if with_refused {
            assert_eq!(h.finished[&2], FinishReason::CachedPrefixMissing);
        }
        let g = SamplingParams::greedy();
        assert_eq!(h.outputs[&1], h.expected(&q, &g, 6, &[]));
        assert_eq!(h.outputs[&3], h.expected(&q, &g, 6, &[]));
        assert_eq!(h.outputs[&4], h.expected(&fresh, &g, 6, &[]));
        [1, 3, 4].map(|id| (h.outputs[&id].clone(), h.cached[&id], h.finished[&id]))
    };
    assert_eq!(run(true), run(false));
}

/// The final prompt token is always recomputed, so a requirement reaching it can never be
/// met: refused at submission. One token less is a requirement like any other.
#[test]
fn a_requirement_covering_the_final_token_is_refused_at_submission() {
    let Fixture { mut h, q, .. } = fixture(small_spec(512), manual_sweep());
    let len = q.len() as u32;
    for required in [len, len + 1, u32::MAX] {
        assert_eq!(
            h.eng.submit(requiring(1, q.clone(), 6, keyed(1), required)),
            Err(SubmitError::UnsatisfiablePrefix)
        );
    }
    assert_eq!(h.eng.unfinished(), 0);
    assert_refused_untouched(&mut h, requiring(1, q, 6, keyed(1), len - 1));
}

/// The requirement is the request's price, fixed at its first admission: a request
/// preempted after it and re-admitted when its cached prefix is gone runs on, recomputing.
#[test]
fn a_preempted_request_is_not_checked_again() {
    let cfg = manual_sweep();
    let max_age = cfg.cache.max_age_ms;
    // 36 blocks a group: the older request's growth preempts the newer one.
    let Fixture { mut h, q, .. } = fixture(small_spec(36), cfg);
    let older = TestRng(44).tokens(20, 32);
    let g = SamplingParams::greedy();
    h.submit(request(1, older.clone(), g, 80, keyed(2)));
    h.step();
    h.now += 1;
    h.submit(requiring(2, q.clone(), 40, keyed(1), 36));
    h.step();
    h.now += 1;
    assert_eq!(
        h.eng.cached_prompt_tokens(2),
        Some(40),
        "admitted on its hit"
    );
    while h.eng.stats().preemptions == 0 {
        assert!(
            !h.finished.contains_key(&2),
            "finished before any preemption"
        );
        h.step();
        h.now += 1;
    }
    // Everything cached under key 1 expires while the request waits.
    h.now = max_age + 1000;
    h.run();
    assert_eq!(h.finished[&2], FinishReason::Length);
    assert_eq!(h.outputs[&2], h.expected(&q, &g, 40, &[]));
    assert_eq!(h.outputs[&1], h.expected(&older, &g, 80, &[]));
    assert_eq!(h.eng.stats().prefix_refusals, 0);
}

/// With a host tier, the lookup counts demoted blocks (admission promotes them). When the
/// device cannot take them back because running sequences hold its whole pool, admission
/// shortens the hit; the request is then refused, and what admission mapped is given
/// back.
#[test]
fn a_hit_shortened_while_promoting_is_refused_and_released() {
    let spec = with_host_tier(small_spec(20), 64);
    let Fixture { mut h, q, .. } = fixture(spec, manual_sweep());
    let g = SamplingParams::greedy();
    // 74 tokens: 19 full-attention blocks, every one the device has, which demotes P.
    let big = TestRng(45).tokens(74, 32);
    h.submit(request(1, big.clone(), g, 8, keyed(2)));
    h.step();
    h.now += 1;
    assert!(h.eng.kv().tier_stats().demoted_entries > 0, "P was demoted");
    assert_eq!(
        h.eng.kv().free_blocks(0),
        0,
        "the full-attention pool is held"
    );
    h.submit(requiring(2, q, 6, keyed(1), 36));
    h.run();
    assert_eq!(h.finished[&2], FinishReason::CachedPrefixMissing);
    assert!(h.outputs[&2].is_empty());
    assert_eq!(h.eng.stats().prefix_refusals, 1);
    // The big request runs on as it would alone (out of memory at its fourth token).
    let alone = {
        let mut h2 = Harness::new(
            with_host_tier(small_spec(20), 64),
            manual_sweep(),
            MockConfig::default(),
        );
        h2.submit(request(1, big.clone(), g, 8, keyed(2)));
        h2.run();
        (h2.outputs[&1].clone(), h2.finished[&1])
    };
    assert_eq!((h.outputs[&1].clone(), h.finished[&1]), alone);
    h.flush();
    assert_eq!(h.eng.kv_keys_held(), 0);
}

// ---------------------------------------------------------------------------------------
// The declaration rule.
// ---------------------------------------------------------------------------------------

/// A lineage per case, so every case meets only its own entries.
fn case_scope(case: u64) -> CacheScope {
    let mut key = [0u8; 32];
    key[..8].copy_from_slice(&case.to_le_bytes());
    CacheScope::Keyed(
        SaltDeriver::with_boot_key(BootKey::from_bytes([7; 32]))
            .lineage(&CacheKey::from_bytes(key)),
    )
}

#[derive(Clone, Copy, Debug)]
struct Geometry {
    block: u32,
    window: u32,
    drafter_window: u32,
    chunk: u32,
}

const VOCAB: u32 = 4096;

/// After a keyed request of prompt `n` that generated `c` tokens and ended by `stop` or by
/// its length: (a) a request extending its prompt with a new turn that diverges from its
/// output at once, requiring `declarable_cached_tokens(n)`, and (b) its prompt followed by
/// exactly its output, requiring `declarable_cached_tokens(n + c)`, are both admitted.
/// `None` when a stop ending at `c` cannot be arranged (the token at `c` came earlier
/// too); otherwise the failures, described.
fn rule_case(geo: Geometry, speculative: bool, n: u32, c: u32, stop: bool) -> Option<Vec<String>> {
    let spec = mimo_like_spec(VOCAB, geo.block, geo.window, geo.drafter_window, 256, 8, 3);
    let mut cfg = sched(speculative);
    cfg.max_prefill_chunk = geo.chunk;
    let mut h = Harness::new(spec, cfg, MockConfig::default());
    let case = u64::from(n) << 32 | u64::from(c) << 1 | u64::from(stop);
    let mut rng = TestRng(case ^ u64::from(geo.block) << 48);
    let prompt = rng.tokens(n as usize, VOCAB);
    let g = SamplingParams::greedy();
    let mut first = request(1, prompt.clone(), g, c, case_scope(case));
    if stop {
        let free = h.expected(&prompt, &g, c, &[]);
        let last = free[c as usize - 1];
        if free[..c as usize - 1].contains(&last) {
            return None;
        }
        first.stop_token_ids = vec![last];
        first.max_tokens = c + 8;
    }
    h.submit(first);
    h.run();
    let output = h.outputs[&1].clone();
    let want = if stop {
        FinishReason::Stop
    } else {
        FinishReason::Length
    };
    assert_eq!((output.len() as u32, h.finished[&1]), (c, want));
    h.flush();

    let mut failures = Vec::new();
    let mut admit = |h: &mut Harness, id: u64, tokens: Vec<u32>, resumed: u32, what: &str| {
        let required = declarable_cached_tokens(u64::from(resumed));
        assert!(required >= CACHED_PREFIX_QUANTUM);
        h.submit(requiring(id, tokens, 64, case_scope(case), required));
        h.step();
        // A hit of all but the last token can finish in this one step, drafting.
        let hit = match h.finished.get(&id) {
            None => h.eng.cached_prompt_tokens(id),
            Some(FinishReason::CachedPrefixMissing) => None,
            Some(_) => h.cached.get(&id).copied(),
        };
        match (h.finished.get(&id), hit) {
            (None | Some(FinishReason::Length), Some(hit)) if hit >= required => {}
            (finish, hit) => failures.push(format!(
                "{geo:?} speculative {speculative} n {n} c {c} stop {stop}: {what} \
                 requiring {required}: finish {finish:?}, hit {hit:?}"
            )),
        }
        h.eng.cancel(id, h.now);
        h.flush();
        h.now += 1;
    };
    // (b) first: it can retain nothing below its own hit, so it cannot help (a).
    let mut continued = prompt.clone();
    continued.extend(&output);
    admit(&mut h, 2, continued, n + c, "the continuation");
    let mut extended = prompt;
    let mut turn = rng.tokens(6, VOCAB);
    if turn[0] == output[0] {
        turn[0] = (turn[0] + 1) % VOCAB;
    }
    extended.extend(turn);
    admit(&mut h, 3, extended, n, "the extension");
    Some(failures)
}

fn sweep(geo: Geometry, ns: impl Iterator<Item = u32> + Clone, cs: &[u32]) {
    let (mut ran, mut skipped, mut failures) = (0, 0, Vec::new());
    let (mut n_residues, mut nc_residues) = (
        std::collections::BTreeSet::new(),
        std::collections::BTreeSet::new(),
    );
    for speculative in [false, true] {
        for stop in [false, true] {
            for n in ns.clone() {
                for &c in cs {
                    match rule_case(geo, speculative, n, c, stop) {
                        None => skipped += 1,
                        Some(f) => {
                            ran += 1;
                            n_residues.insert((stop, speculative, n % geo.block));
                            nc_residues.insert((stop, speculative, (n + c) % geo.block));
                            failures.extend(f);
                        }
                    }
                }
            }
        }
    }
    // Every residue of `n` and of `n + c`, for every ending, with and without drafts.
    assert_eq!(n_residues.len() as u32, 4 * geo.block, "{geo:?}");
    assert_eq!(nc_residues.len() as u32, 4 * geo.block, "{geo:?}");
    assert!(skipped * 20 < ran, "{skipped} skipped of {}", ran + skipped);
    assert!(
        failures.is_empty(),
        "{} of {ran} cases failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Two regions of the sweep. Below the second quantum (`n + c <= 512`) both requests
/// require one quantum, which the regeneration point alone covers. Across it (`n <= 512 <
/// n + c`) the continuation requires a quantum more than the extension, beyond the
/// regeneration point: only the continuation point covers that.
fn crossing(ns: std::ops::Range<u32>, cs: &[u32]) -> (std::ops::Range<u32>, Vec<u32>) {
    for n in ns.clone() {
        for &c in cs {
            assert!(
                declarable_cached_tokens(u64::from(n + c)) > declarable_cached_tokens(u64::from(n)),
                "n {n} c {c} does not cross a quantum"
            );
        }
    }
    (ns, cs.to_vec())
}

const BLOCK_16: Geometry = Geometry {
    block: 16,
    window: 40,
    drafter_window: 70,
    chunk: 256,
};

const BLOCK_32: Geometry = Geometry {
    block: 32,
    window: 70,
    drafter_window: 100,
    chunk: 100,
};

#[test]
fn the_declaration_rule_holds_at_block_16() {
    let cs: Vec<u32> = (1..=16).collect();
    sweep(BLOCK_16, 257..273, &cs);
}

#[test]
fn the_declaration_rule_holds_at_block_16_across_a_quantum() {
    let cs: Vec<u32> = (16..32).collect();
    let (ns, cs) = crossing(497..513, &cs);
    sweep(BLOCK_16, ns, &cs);
}

#[test]
fn the_declaration_rule_holds_at_block_32_in_uneven_chunks() {
    sweep(BLOCK_32, 257..289, &[1, 2, 17, 31, 32, 33]);
}

#[test]
fn the_declaration_rule_holds_at_block_32_across_a_quantum() {
    let (ns, cs) = crossing(481..513, &[32, 33, 48, 63]);
    sweep(BLOCK_32, ns, &cs);
}
