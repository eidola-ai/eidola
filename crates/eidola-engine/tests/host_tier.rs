//! The host tier: idle prefix-cache entries evicted from the device pools under pressure
//! are demoted to host memory as the ciphertext they already are, promoted back for a hit
//! and decrypted only then, and kept, zeroed and expired exactly like device entries.
//!
//! The harness checks after every step that free host blocks are zero (or have a zero
//! queued) and held ones hold ciphertext; the mock refuses to demote a block that is not
//! encrypted and to promote into a block that is not zero or from one holding no
//! ciphertext.

mod common;

use std::collections::{BTreeMap, BTreeSet, HashSet};

use common::*;
use eidola_engine::engine::{CacheScope, FinishReason};
use eidola_engine::mock::{MockConfig, kv_value, with_host_tier};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::ModelSpec;

/// A block's stored entries, `(value, position)` per offset.
type Cells = Vec<(u64, u32)>;

fn keyed(n: u8) -> CacheScope {
    CacheScope::Keyed(salt(n))
}

/// `small_spec(device)` with `host` host blocks per group.
fn tiered(device: u32, host: u32) -> ModelSpec {
    with_host_tier(small_spec(device), host)
}

/// Demotions (`true`) or promotions since `mark`, as `(group, block, host)`.
fn transfers_since(h: &Harness, mark: usize, demote: bool) -> Vec<(u32, u32, u32)> {
    h.eng.executor().transfer_log()[mark..]
        .iter()
        .filter(|t| t.4 == demote)
        .map(|&(_, g, b, hb, _)| (g, b, hb))
        .collect()
}

/// Runs one request to completion and flushes.
fn run_one(
    h: &mut Harness,
    id: u64,
    prompt: Vec<u32>,
    params: SamplingParams,
    max: u32,
    cache: CacheScope,
) {
    h.submit(request(id, prompt, params, max, cache));
    h.run();
    h.flush();
}

/// Runs `n` requests of other lineages, one after another, enough to push the first
/// lineage's entries off a small device pool.
fn push_out(h: &mut Harness, rng: &mut TestRng, first_id: u64, n: u64) {
    for i in 0..n {
        let prompt = rng.tokens(37, 32);
        run_one(
            h,
            first_id + i,
            prompt,
            SamplingParams::greedy(),
            5,
            keyed(100 + i as u8),
        );
    }
}

/// Device pressure demotes an idle entry instead of dropping it, in every group, and a
/// later hit finds it whole: the same cached positions as with no pressure, each block it
/// maps promoted and then decrypted, and the reference output.
#[test]
fn device_pressure_demotes_instead_of_dropping() {
    let mut h = Harness::new(tiered(24, 128), sched(true), MockConfig::default());
    let mut rng = TestRng(1);
    let prompt = rng.tokens(37, 32);
    let g = SamplingParams::greedy();
    run_one(&mut h, 1, prompt.clone(), g, 10, keyed(1));
    let first = h.eng.kv().cache_blocks();
    let entries = h.eng.kv().cache_entries();

    let mark = h.eng.executor().transfer_log().len();
    push_out(&mut h, &mut rng, 10, 4);
    let demoted = transfers_since(&h, mark, true);
    let demoted_blocks: BTreeSet<(u32, u32)> = demoted.iter().map(|&(g, b, _)| (g, b)).collect();
    assert!(
        first.iter().all(|b| demoted_blocks.contains(b)),
        "every block of the first entry was demoted"
    );
    assert_eq!(
        demoted.iter().map(|t| t.0).collect::<BTreeSet<_>>(),
        BTreeSet::from([0, 1, 2]),
        "global, sliding and drafter blocks are all demoted"
    );
    let stats = h.eng.kv().tier_stats();
    assert_eq!(stats.host_drops, 0);
    assert_eq!(stats.device_drops, 0, "nothing was dropped");
    assert!(h.eng.kv().cache_entries() >= entries + 4 * 9);

    // The regeneration hits 36 positions, exactly as with no pressure.
    let mark = h.eng.executor().transfer_log().len();
    let cmark = h.eng.executor().crypt_log().len();
    h.submit(request(2, prompt.clone(), g, 10, keyed(1)));
    h.step();
    assert_eq!(h.eng.cached_prompt_tokens(2), Some(36));
    let promoted = transfers_since(&h, mark, false);
    assert!(!promoted.is_empty());
    let decrypted: BTreeSet<(u32, u32)> = h.eng.executor().crypt_log()[cmark..]
        .iter()
        .filter(|e| !e.3)
        .map(|e| (e.1, e.2))
        .collect();
    for &(g, b, _) in &promoted {
        assert!(
            decrypted.contains(&(g, b)),
            "promoted block {g}/{b} decrypted after its copy"
        );
    }
    h.run();
    assert_eq!(h.outputs[&2], h.outputs[&1]);
    assert_eq!(h.outputs[&2], h.expected(&prompt, &g, 10, &[]));
    h.flush();
    assert_eq!(h.eng.kv_keys_held(), 0);
}

/// A demoted block's ciphertext arrives on the host byte for byte, comes back byte for
/// byte, and decrypting it restores the plaintext the prompt determines.
#[test]
fn demote_and_promote_round_trip_the_ciphertext() {
    let cfg = MockConfig::default();
    let mut h = Harness::new(tiered(24, 128), sched(false), cfg);
    let mut rng = TestRng(2);
    let prompt = rng.tokens(37, 32);
    let g = SamplingParams::greedy();
    run_one(&mut h, 1, prompt.clone(), g, 4, keyed(1));
    let before: BTreeMap<(u32, u32), Cells> = h
        .eng
        .kv()
        .cache_blocks()
        .into_iter()
        .map(|(gr, b)| ((gr, b), h.eng.executor().device_block(gr, b).to_vec()))
        .collect();

    let mark = h.eng.executor().transfer_log().len();
    push_out(&mut h, &mut rng, 10, 4);
    let mut on_host: BTreeMap<(u32, u32), (u32, Cells)> = BTreeMap::new();
    for (gr, b, hb) in transfers_since(&h, mark, true) {
        if let Some(cipher) = before.get(&(gr, b))
            && h.eng.kv().host_cache_blocks().contains(&(gr, hb))
            && !on_host.contains_key(&(gr, b))
        {
            assert_eq!(
                h.eng.executor().host_block(gr, hb),
                &cipher[..],
                "ciphertext preserved"
            );
            on_host.insert((gr, b), (hb, cipher.clone()));
        }
    }
    assert!(on_host.keys().any(|&(gr, _)| gr == 0));

    let mark = h.eng.executor().transfer_log().len();
    h.submit(request(2, prompt.clone(), g, 4, keyed(1)));
    h.step();
    let host_to_cipher: BTreeMap<(u32, u32), &Cells> = on_host
        .iter()
        .map(|(&(gr, _), (hb, c))| ((gr, *hb), c))
        .collect();
    let mut checked = 0;
    for (gr, d, hb) in transfers_since(&h, mark, false) {
        if gr != 0 {
            continue; // sliding windows may have slid out and been re-encrypted already
        }
        let cipher = host_to_cipher[&(gr, hb)];
        assert_ne!(
            h.eng.executor().device_block(gr, d),
            &cipher[..],
            "decrypted"
        );
        for &(v, p) in h.eng.executor().device_block(gr, d) {
            assert_eq!(
                v,
                kv_value(&cfg, prompt[p as usize], p, gr),
                "plaintext restored"
            );
        }
        checked += 1;
    }
    assert!(checked > 0);
    h.run();
    assert_eq!(h.outputs[&2], h.outputs[&1]);
    h.flush();
}

/// The outputs of a request whose prefix comes back from the host tier, sampled and
/// greedy, with and without drafting, equal those of the same request hitting the device
/// cache and of one recomputing it with the cache off.
#[test]
fn a_host_hit_equals_a_device_hit_and_a_recompute() {
    for speculative in [false, true] {
        for params in [
            SamplingParams::greedy(),
            SamplingParams::random(0.9, 77).unwrap(),
        ] {
            let mut runs = Vec::new();
            for (spec, cache_on) in [
                (small_spec(1024), true),
                (tiered(24, 256), true),
                (small_spec(1024), false),
            ] {
                let mut cfg = sched(speculative);
                cfg.cache.enabled = cache_on;
                let mut h = Harness::new(spec, cfg, MockConfig::default());
                let mut rng = TestRng(3);
                let prompt = rng.tokens(37, 32);
                run_one(&mut h, 1, prompt.clone(), params, 12, keyed(1));
                push_out(&mut h, &mut rng, 10, 4);
                let mut next = prompt.clone();
                next.extend(&h.outputs[&1]);
                next.extend(rng.tokens(3, 32));
                run_one(&mut h, 2, next.clone(), params, 12, keyed(1));
                runs.push((h.outputs[&2].clone(), h.cached[&2], h.eng.kv().tier_stats()));
            }
            let (device, host, recompute) = (&runs[0], &runs[1], &runs[2]);
            assert_eq!(device.0, host.0);
            assert_eq!(host.0, recompute.0);
            assert!(
                device.1 > 0 && device.1 == host.1,
                "same hit from either tier"
            );
            assert_eq!(recompute.1, 0);
            assert_eq!(device.2.promoted_blocks, 0);
            assert!(
                host.2.promoted_blocks > 0,
                "the hit came from the host tier"
            );
        }
    }
}

/// Every plaintext KV value the workload could produce, in every group.
fn plaintexts(cfg: &MockConfig, prompts: &[Vec<u32>], groups: u32) -> HashSet<u64> {
    let mut out = HashSet::new();
    for p in prompts {
        for (pos, &t) in p.iter().enumerate() {
            for g in 0..groups {
                out.insert(kv_value(cfg, t, pos as u32, g));
            }
        }
    }
    out
}

/// Under randomized traffic with both tiers under pressure, nothing a held host block
/// holds is ever a plaintext KV value, after every step; outputs match the reference.
#[test]
fn nothing_in_the_host_tier_is_plaintext() {
    let cfg = MockConfig::default();
    for seed in 0..4u64 {
        let mut h = Harness::new(tiered(28, 40), sched(seed % 2 == 0), cfg);
        let mut rng = TestRng(200 + seed);
        let heads: Vec<Vec<u32>> = (0..3).map(|_| rng.tokens(12, 32)).collect();
        let mut cases = Vec::new();
        for id in 0..30u64 {
            let mut prompt = heads[rng.below(3) as usize].clone();
            prompt.extend(rng.var_tokens(1, 20, 32));
            let max = 1 + rng.below(10) as u32;
            cases.push((id, prompt.clone(), max));
            h.submit(request(
                id,
                prompt,
                SamplingParams::greedy(),
                max,
                keyed(rng.below(6) as u8),
            ));
            for _ in 0..1 + rng.below(4) {
                h.step();
                h.now += 1;
            }
        }
        h.run();
        h.flush();
        // Every token sequence the run wrote: prompts with their outputs.
        let full: Vec<Vec<u32>> = cases
            .iter()
            .map(|(id, p, _)| {
                let mut t = p.clone();
                t.extend(&h.outputs[id]);
                t
            })
            .collect();
        let plain = plaintexts(&cfg, &full, 3);
        let stats = h.eng.kv().tier_stats();
        assert!(
            stats.demoted_blocks > 0 && stats.promoted_blocks > 0,
            "seed {seed}: {stats:?}"
        );
        for (g, hb) in h.eng.kv().host_cache_blocks() {
            for &(v, _) in h.eng.executor().host_block(g, hb) {
                assert!(
                    !plain.contains(&v),
                    "seed {seed}: plaintext in host block {g}/{hb}"
                );
            }
        }
        for (id, prompt, max) in cases {
            assert_eq!(
                h.outputs[&id],
                h.expected(&prompt, &SamplingParams::greedy(), max, &[]),
                "seed {seed} request {id}"
            );
        }
    }
}

/// When the host tier is full, demoting drops its least recently used entry, zeroing its
/// host blocks: the oldest conversation misses afterwards, the newest demoted one hits.
#[test]
fn host_pressure_drops_the_least_recently_used_and_zeroes_it() {
    let mut h = Harness::new(tiered(24, 24), sched(false), MockConfig::default());
    let mut rng = TestRng(5);
    let g = SamplingParams::greedy();
    let prompts: Vec<Vec<u32>> = (0..8).map(|_| rng.tokens(37, 32)).collect();
    let hz = h.eng.executor().host_zero_log().len();
    for (i, p) in prompts.iter().enumerate() {
        run_one(&mut h, i as u64, p.clone(), g, 4, keyed(i as u8));
        h.now += 10;
    }
    let stats = h.eng.kv().tier_stats();
    assert!(stats.host_drops > 0, "{stats:?}");
    assert!(
        h.eng.executor().host_zero_log().len() > hz,
        "dropped host blocks are zeroed"
    );
    // The oldest was dropped; the most recent demoted one is still there.
    run_one(&mut h, 100, prompts[0].clone(), g, 4, keyed(0));
    assert_eq!(h.cached[&100], 0, "the oldest conversation was dropped");
    assert_eq!(h.outputs[&100], h.outputs[&0]);
    let mark = h.eng.executor().transfer_log().len();
    run_one(&mut h, 101, prompts[5].clone(), g, 4, keyed(5));
    assert_eq!(h.cached[&101], 36, "a recently demoted conversation hits");
    assert!(
        !transfers_since(&h, mark, false).is_empty(),
        "from the host tier"
    );
    assert_eq!(h.outputs[&101], h.outputs[&5]);
}

/// Idle TTL and max age apply to entries on the host as on the device: a sweep removes
/// them and zeroes exactly their host blocks, and a lookup never hits an expired one.
#[test]
fn expiry_applies_in_both_tiers() {
    for max_age in [false, true] {
        let mut cfg = sched(false);
        cfg.sweep_interval_ms = u64::MAX;
        if max_age {
            cfg.cache.max_age_ms = 500;
            cfg.cache.idle_ttl_ms = 400;
        }
        let limit = if max_age {
            cfg.cache.max_age_ms
        } else {
            cfg.cache.idle_ttl_ms
        };
        let mut h = Harness::new(tiered(24, 128), cfg, MockConfig::default());
        let mut rng = TestRng(6);
        let prompt = rng.tokens(37, 32);
        let g = SamplingParams::greedy();
        let start = h.now;
        run_one(&mut h, 1, prompt.clone(), g, 4, keyed(1));
        push_out(&mut h, &mut rng, 10, 4);
        let on_host = h.eng.kv().host_cache_blocks();
        assert!(!on_host.is_empty());
        let hz = h.eng.executor().host_zero_log().len();
        // Long enough for every entry (the first one's and the pressure's).
        h.eng.sweep(start + limit + 1_000).unwrap();
        h.check();
        let zeroed: BTreeSet<(u32, u32)> = h.eng.executor().host_zero_log()[hz..]
            .iter()
            .map(|&(_, g, hb)| (g, hb))
            .collect();
        assert_eq!(
            zeroed, on_host,
            "exactly the expired host blocks are zeroed"
        );
        assert_eq!(h.eng.kv().cache_entries(), 0);
        for &(g, hb) in &on_host {
            assert!(h.eng.executor().host_block_is_zero(g, hb));
        }

        // Lookup-time expiry, no sweep: the demoted entry is past its time when asked.
        let mut cfg = sched(false);
        cfg.sweep_interval_ms = u64::MAX;
        let ttl = cfg.cache.idle_ttl_ms;
        let mut h = Harness::new(tiered(24, 128), cfg, MockConfig::default());
        let mut rng = TestRng(7);
        let prompt = rng.tokens(37, 32);
        run_one(&mut h, 1, prompt.clone(), g, 4, keyed(1));
        push_out(&mut h, &mut rng, 10, 4);
        assert!(!h.eng.kv().host_cache_blocks().is_empty());
        let mark = h.eng.executor().transfer_log().len();
        h.now += ttl + 1_000;
        run_one(&mut h, 2, prompt.clone(), g, 4, keyed(1));
        assert_eq!(h.cached[&2], 0, "an expired host entry is never hit");
        assert!(transfers_since(&h, mark, false).is_empty(), "nor promoted");
    }
}

/// Another lineage with the same prompt never promotes the first lineage's demoted entry,
/// and leaves its host ciphertext untouched.
#[test]
fn another_key_never_promotes_a_demoted_entry() {
    let mut h = Harness::new(tiered(24, 128), sched(false), MockConfig::default());
    let mut rng = TestRng(8);
    let prompt = rng.tokens(37, 32);
    let g = SamplingParams::greedy();
    run_one(&mut h, 1, prompt.clone(), g, 4, keyed(1));
    push_out(&mut h, &mut rng, 10, 4);
    let on_host: BTreeMap<(u32, u32), Cells> = h
        .eng
        .kv()
        .host_cache_blocks()
        .into_iter()
        .map(|(gr, hb)| ((gr, hb), h.eng.executor().host_block(gr, hb).to_vec()))
        .collect();
    let mark = h.eng.executor().transfer_log().len();
    h.submit(request(2, prompt.clone(), g, 4, keyed(2)));
    h.step();
    assert_eq!(h.eng.cached_prompt_tokens(2), Some(0));
    assert!(
        transfers_since(&h, mark, false).is_empty(),
        "nothing promoted"
    );
    h.run();
    h.flush();
    assert_eq!(h.outputs[&2], h.outputs[&1]);
    for ((gr, hb), cipher) in on_host {
        if h.eng.kv().host_cache_blocks().contains(&(gr, hb)) {
            assert_eq!(h.eng.executor().host_block(gr, hb), &cipher[..]);
        }
    }
}

/// Branch and regeneration windows (sliding and drafter groups) survive demotion: a
/// divergent turn after a demoted shared prefix still hits at the branch.
#[test]
fn retained_windows_survive_demotion() {
    let mut h = Harness::new(tiered(32, 256), sched(true), MockConfig::default());
    let mut rng = TestRng(9);
    let g = SamplingParams::greedy();
    let prefix = rng.tokens(24, 32);
    for id in 1..=2u64 {
        let mut p = prefix.clone();
        p.extend(rng.tokens(9, 32));
        run_one(&mut h, id, p, g, 4, keyed(1));
    }
    push_out(&mut h, &mut rng, 10, 5);
    let host_groups: BTreeSet<u32> = h
        .eng
        .kv()
        .host_cache_blocks()
        .iter()
        .map(|&(g, _)| g)
        .collect();
    assert_eq!(host_groups, BTreeSet::from([0, 1, 2]));
    let mut p = prefix.clone();
    p.extend(rng.tokens(9, 32));
    let mark = h.eng.executor().transfer_log().len();
    run_one(&mut h, 3, p.clone(), g, 4, keyed(1));
    assert_eq!(h.cached[&3], 24, "hits at the branch");
    let promoted: BTreeSet<u32> = transfers_since(&h, mark, false)
        .iter()
        .map(|t| t.0)
        .collect();
    assert!(
        promoted.contains(&1) && promoted.contains(&2),
        "{promoted:?}"
    );
    assert_eq!(h.outputs[&3], h.expected(&p, &g, 4, &[]));
}

/// Randomized keyed and private traffic, preemption and cancellation over small device
/// pools and a small host tier: every harness check holds after every step, outputs match
/// the reference, and no key is left.
#[test]
fn randomized_traffic_with_a_host_tier() {
    for seed in 0..6u64 {
        let mut cfg = sched(seed % 2 == 0);
        cfg.cache.idle_ttl_ms = 60;
        let mut h = Harness::new(
            tiered(36 + 4 * seed as u32, 16 + 16 * seed as u32),
            cfg,
            MockConfig::default(),
        );
        let mut rng = TestRng(300 + seed);
        let heads: Vec<Vec<u32>> = (0..3).map(|_| rng.tokens(12, 32)).collect();
        let mut cases = Vec::new();
        for id in 0..40u64 {
            let mut prompt = heads[rng.below(3) as usize].clone();
            prompt.extend(rng.var_tokens(1, 20, 32));
            let cache = match rng.below(5) {
                0 => CacheScope::Private,
                k => keyed(k as u8),
            };
            let max = 1 + rng.below(12) as u32;
            cases.push((id, prompt.clone(), max));
            h.submit(request(id, prompt, SamplingParams::greedy(), max, cache));
            for _ in 0..rng.below(4) {
                h.step();
                h.now += 1 + rng.below(10);
            }
            if rng.chance(0.1) && !h.finished.contains_key(&id) {
                h.eng.cancel(id, h.now);
                h.finished.insert(id, FinishReason::Cancelled);
            }
        }
        h.run();
        h.flush();
        for (id, prompt, max) in cases {
            if h.finished[&id] != FinishReason::Cancelled {
                assert_eq!(
                    h.outputs[&id],
                    h.expected(&prompt, &SamplingParams::greedy(), max, &[]),
                    "seed {seed} request {id}"
                );
            }
        }
        assert_eq!(h.eng.kv_keys_held(), 0);
        assert!(!h.eng.kv().has_pending_maintenance());
        let stats = h.eng.kv().tier_stats();
        assert!(stats.demoted_blocks > 0, "seed {seed}: {stats:?}");
    }
}
