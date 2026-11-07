//! The host tier in the reference executor: idle blocks demoted under device pressure
//! arrive in the host pools as the ciphertext they were, every word of it, and never as
//! plaintext; a hit promotes and decrypts them into exactly the bits the dense oracle
//! computes, through the real scheduler, and gives the same output as a hit on the
//! device and as a recompute with the cache off. The harness checks after every step that
//! free host blocks are zero and held ones ciphertext.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::*;
use eidola_engine::engine::CacheScope;
use eidola_engine::sampling::SamplingParams;

type Images = BTreeMap<(u32, u32), Vec<u32>>;

fn keyed(n: u8) -> CacheScope {
    CacheScope::Keyed(salt(n))
}

/// Host-tier blocks per group in these tests: room for the pushed-out lineages, about
/// 2.6 MB locked (53 KB a block of every group on the fixture), well under an 8 MiB
/// memory-lock limit even at a time.
const HOST: u32 = 48;

/// A harness over the fixture with `device` blocks per group and `host` host-tier blocks.
fn tiered(device: u32, host: u32, spec_on: bool, cache: bool) -> Harness {
    let mut cfg = exec_config(device, vec![0, 1]);
    cfg.host_blocks = host;
    let mut s = sched(spec_on);
    s.cache.enabled = cache;
    Harness::new(fixture(), cfg, s)
}

/// Words two images share in place.
fn same_words(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x == y).count()
}

fn run_one(h: &mut Harness, id: u64, prompt: Vec<u32>, params: SamplingParams, cache: CacheScope) {
    h.submit(request(id, prompt, params, 6, cache));
    h.run();
    h.flush();
}

/// Steps until every request has finished, keeping the latest plaintext image of every
/// block (mapped, or with its encryption still queued).
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
        for (g, b) in blocks {
            plain.insert((g, b), h.eng.executor().block_image(g, b));
        }
    }
}

/// Requests of other lineages until the first one's entries have left the device.
fn push_out(h: &mut Harness, rng: &mut TestRng) {
    for i in 0..5u64 {
        run_one(
            h,
            100 + i,
            rng.tokens(30, VOCAB),
            SamplingParams::greedy(),
            keyed(50 + i as u8),
        );
    }
}

#[test]
fn a_host_hit_is_bit_exact_and_the_host_holds_only_ciphertext() {
    let _locked = memlock_guard();
    for spec_on in [false, true] {
        let mut h = tiered(24, HOST, spec_on, true);
        let mut rng = TestRng(70);
        let prompt = rng.tokens(30, VOCAB);
        let g = SamplingParams::greedy();
        h.submit(request(1, prompt.clone(), g, 6, keyed(1)));
        let mut seen = Images::new();
        run_recording(&mut h, &mut seen);
        h.flush();
        let cached = h.eng.kv().cache_blocks();
        let plain: Images = cached.iter().map(|b| (*b, seen[b].clone())).collect();
        let cipher: Images = cached
            .iter()
            .map(|&(gr, b)| ((gr, b), h.eng.executor().block_image(gr, b)))
            .collect();

        let mark = h.eng.executor().transfer_log().len();
        push_out(&mut h, &mut rng);
        // Each of the first entry's blocks reached the host as its ciphertext, word for
        // word, and nothing on the host is that plaintext.
        let mut on_host = BTreeMap::new();
        for &(_, gr, b, hb, demote) in &h.eng.executor().transfer_log()[mark..] {
            if demote && cipher.contains_key(&(gr, b)) && !on_host.contains_key(&(gr, b)) {
                on_host.insert((gr, b), hb);
            }
        }
        assert_eq!(
            on_host.keys().copied().collect::<BTreeSet<_>>(),
            cached,
            "every block of the first entry was demoted"
        );
        let held = h.eng.kv().host_cache_blocks();
        let groups: BTreeSet<u32> = on_host.keys().map(|&(g, _)| g).collect();
        assert_eq!(groups.len(), h.spec.kv_groups.len(), "every group demoted");
        for (&(gr, b), &hb) in &on_host {
            assert!(held.contains(&(gr, hb)), "still on the host");
            let img = h.eng.executor().host_block_image(gr, hb);
            assert_eq!(img, cipher[&(gr, b)], "group {gr}: ciphertext preserved");
            assert!(
                same_words(&img, &plain[&(gr, b)]) * 1000 < img.len(),
                "group {gr}: plaintext on the host"
            );
        }

        // The hit: promoted, decrypted, bit-exact against the dense oracle.
        let tmark = h.eng.executor().transfer_log().len();
        let cmark = h.eng.executor().crypt_log().len();
        h.submit(request(2, prompt.clone(), g, 6, keyed(1)));
        h.step();
        assert_eq!(h.eng.cached_prompt_tokens(2), Some(28));
        let promoted: Vec<(u32, u32)> = h.eng.executor().transfer_log()[tmark..]
            .iter()
            .filter(|t| !t.4)
            .map(|t| (t.1, t.2))
            .collect();
        assert!(!promoted.is_empty());
        let decrypted: BTreeSet<(u32, u32)> = h.eng.executor().crypt_log()[cmark..]
            .iter()
            .filter(|e| !e.3)
            .map(|e| (e.1, e.2))
            .collect();
        assert!(promoted.iter().all(|b| decrypted.contains(b)));
        h.run();
        assert_eq!(h.outputs[&2], h.outputs[&1]);
        let sum = h.check_all(spec_on);
        eprintln!("speculative {spec_on}: {sum:?}");
        h.flush();
        assert_eq!(h.eng.kv_keys_held(), 0);
        let stats = h.eng.kv().tier_stats();
        assert!(
            stats.promoted_blocks > 0 && stats.device_drops == 0,
            "{stats:?}"
        );
    }
}

/// The same continuation served from the host tier, from the device cache, and with the
/// cache off: byte-identical outputs, sampled and greedy.
#[test]
fn a_host_hit_equals_a_device_hit_and_a_recompute() {
    let _locked = memlock_guard();
    for spec_on in [false, true] {
        for params in [
            SamplingParams::greedy(),
            SamplingParams::random(0.8, 9).unwrap(),
        ] {
            let mut outs = Vec::new();
            for (device, host, cache) in [(512, 0, true), (24, HOST, true), (512, 0, false)] {
                let mut h = tiered(device, host, spec_on, cache);
                let mut rng = TestRng(71);
                let prompt = rng.tokens(30, VOCAB);
                run_one(&mut h, 1, prompt.clone(), params, keyed(1));
                push_out(&mut h, &mut rng);
                let mut next = prompt.clone();
                next.extend(&h.outputs[&1]);
                next.extend(rng.tokens(3, VOCAB));
                run_one(&mut h, 2, next, params, keyed(1));
                outs.push((h.outputs[&2].clone(), h.cached[&2], h.eng.kv().tier_stats()));
                h.check_all(spec_on);
            }
            assert_eq!(outs[0].0, outs[1].0, "device hit vs host hit");
            assert_eq!(outs[1].0, outs[2].0, "host hit vs recompute");
            assert!(outs[0].1 > 0 && outs[0].1 == outs[1].1);
            assert_eq!(outs[2].1, 0);
            assert!(outs[1].2.promoted_blocks > 0, "served from the host tier");
        }
    }
}

/// Randomized traffic over small device pools and a small host tier: both tiers under
/// pressure, every harness check after every step, every row bit-exact.
#[test]
fn randomized_traffic_with_a_host_tier_is_bit_exact() {
    let _locked = memlock_guard();
    for seed in 0..3u64 {
        let spec_on = seed % 2 == 0;
        let mut h = tiered(32, 24, spec_on, true);
        let mut rng = TestRng(400 + seed);
        let heads: Vec<Vec<u32>> = (0..3).map(|_| rng.tokens(10, VOCAB)).collect();
        for id in 0..16u64 {
            let mut prompt = heads[rng.below(3) as usize].clone();
            prompt.extend(rng.var_tokens(1, 14, VOCAB));
            let cache = match rng.below(4) {
                0 => CacheScope::Private,
                k => keyed(k as u8),
            };
            h.submit(request(
                id,
                prompt,
                params(id),
                1 + rng.below(6) as u32,
                cache,
            ));
            for _ in 0..rng.below(3) {
                h.step();
                h.now += 1;
            }
        }
        h.run();
        h.flush();
        h.check_all(spec_on);
        let stats = h.eng.kv().tier_stats();
        assert!(stats.demoted_blocks > 0, "seed {seed}: {stats:?}");
        assert_eq!(h.eng.kv_keys_held(), 0);
    }
}
