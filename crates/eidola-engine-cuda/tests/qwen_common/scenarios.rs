//! The executor's scenarios against the CPU reference executor (`pair`), generic over the
//! device: run on the host-compiled kernel bodies (`tests/qwen_host.rs`) and on a GPU
//! (`tests/qwen_tiny.rs`). Each panics on the first disagreement, naming the step.
#![allow(dead_code)]

use eidola_engine::engine::CacheScope;
use eidola_engine::sampling::SamplingParams;
use eidola_engine_cuda::qwen::{QwenDevice, StateStrategy};

use super::pair::*;

fn small(x: u64) -> usize {
    usize::try_from(x).expect("small")
}

pub fn one_request<D: QwenDevice>(dev: D) {
    let mut h = Harness::new(dev, &Geometry::plain(0), sched(256, false));
    let prompt = TestRng(1).tokens(9, VOCAB);
    let out = h.run_one(request(
        1,
        prompt,
        SamplingParams::greedy(),
        6,
        CacheScope::Private,
    ));
    assert_eq!(out.len(), 6);
    assert!(h.pair().compared_logits >= 6);
}

/// Mixed requests (private and keyed, every sampling kind) at chunk sizes 1, 3, 4, 5 and
/// whole, with and without drafts, under both state strategies.
pub fn chunkings<D: QwenDevice>(make: &dyn Fn() -> D) {
    for strategy in [StateStrategy::Live, StateStrategy::Lagging] {
        for speculative in [false, true] {
            for chunk in [1, 3, 4, 5, 256] {
                let k = if speculative { 3 } else { 0 };
                let g = Geometry {
                    strategy,
                    ..Geometry::plain(k)
                };
                let mut h = Harness::new(make(), &g, sched(chunk, speculative));
                let mut rng = TestRng(u64::from(chunk) + 100 * u64::from(speculative));
                for id in 0..5u64 {
                    let n = 1 + small(rng.below(20));
                    let prompt = rng.tokens(n, VOCAB);
                    let max = 1 + u32::try_from(rng.below(10)).expect("small");
                    let cache = if id % 2 == 0 {
                        CacheScope::Private
                    } else {
                        keyed(u8::try_from(id).expect("small"))
                    };
                    h.submit(request(id, prompt, params(id), max, cache));
                }
                h.run();
                h.flush();
                assert!(
                    h.pair().compared_images > 0,
                    "chunk {chunk}: no state stored"
                );
            }
        }
    }
}

/// Shared prefixes over small pools (hits, preemption, shedding of snapshots), with and
/// without a host tier, then a follow-up turn hitting a snapshot.
pub fn hits_and_pressure<D: QwenDevice>(make: &dyn Fn() -> D) {
    for strategy in [StateStrategy::Live, StateStrategy::Lagging] {
        for tiered in [false, true] {
            let g = Geometry {
                blocks: 40,
                buffers: 12,
                host: if tiered { 64 } else { 0 },
                host_buffers: if tiered { 24 } else { 0 },
                k: 3,
                strategy,
            };
            let mut h = Harness::new(make(), &g, sched(5, true));
            let mut rng = TestRng(7);
            let shared = rng.tokens(14, VOCAB);
            for id in 0..10u64 {
                let mut prompt = shared.clone();
                let n = 1 + small(rng.below(9));
                prompt.extend(rng.tokens(n, VOCAB));
                h.submit(request(
                    id,
                    prompt,
                    params(id),
                    6,
                    keyed(u8::try_from(id % 3).expect("small")),
                ));
                if id % 2 == 1 {
                    h.step();
                    h.now += 1;
                }
            }
            h.run();
            h.flush();
            let first = h.outputs[&0].clone();
            let mut turn = shared.clone();
            turn.extend(first);
            turn.extend(rng.tokens(3, VOCAB));
            h.run_one(request(100, turn, SamplingParams::greedy(), 5, keyed(0)));
        }
    }
}

/// A greedy request alone and beside others: the same tokens.
pub fn alone_and_crowded<D: QwenDevice>(make: &dyn Fn() -> D) {
    let prompt = TestRng(9).tokens(17, VOCAB);
    let mut alone = Harness::new(make(), &Geometry::plain(3), sched(6, true));
    let a = alone.run_one(request(
        1,
        prompt.clone(),
        SamplingParams::greedy(),
        10,
        keyed(1),
    ));
    let mut crowd = Harness::new(make(), &Geometry::plain(3), sched(6, true));
    let mut rng = TestRng(10);
    for id in 10..14u64 {
        let n = 5 + small(rng.below(9));
        crowd.submit(request(
            id,
            rng.tokens(n, VOCAB),
            params(id),
            9,
            keyed(u8::try_from(id).expect("small")),
        ));
    }
    crowd.submit(request(1, prompt, SamplingParams::greedy(), 10, keyed(1)));
    crowd.run();
    crowd.flush();
    assert_eq!(crowd.outputs[&1], a);
}
