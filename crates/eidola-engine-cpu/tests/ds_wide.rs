//! The DeepSeek-V4.1 CPU executor on a model whose window is wider than one 64-slot
//! attention block (the fixture's weights under a window of 80 and a five-slot drafter:
//! see `ds_common::wide`), through the engine, every row against the dense oracle.
//!
//! At this width the place of an empty window slot decides which block an entry falls
//! in, and so does the order of a draft pass's context: the canonical list (the window
//! right-aligned in its slots) and the drafter's ring order are both observable here,
//! which they are not at the committed fixture's window of 48. The oracle is the model
//! crate's forward under the same configuration; it is verified against the reference
//! only at the committed fixture's shape, so these tests hold the executor to the
//! oracle and say nothing about the oracle against the reference.

mod ds_common;

use std::collections::HashMap;

use ds_common::*;
use eidola_engine::engine::{CacheScope, FinishReason};
use eidola_engine::sampling::SamplingParams;

fn wide_harness(blocks: u32, width: u32, sched: eidola_engine::engine::SchedulerConfig) -> Harness {
    Harness::new(wide(), exec_config(blocks, width), sched)
}

fn chunked_case(chunks: &[u32]) {
    let mut rng = TestRng(21);
    // Below one block of window, around the window, and around the reach (317).
    let lens = [63usize, 81, 160, 317, 318, 331];
    let prompts: Vec<Vec<u32>> = lens.iter().map(|&n| rng.tokens(n)).collect();
    for spec_on in [false, true] {
        let mut results = Vec::new();
        for &chunk in chunks {
            let mut h = wide_harness(512, 5, chunked(chunk, spec_on));
            for (i, p) in prompts.iter().enumerate() {
                let cache = if i % 2 == 0 {
                    CacheScope::Private
                } else {
                    keyed(i as u8)
                };
                h.submit(request(i as u64, p.clone(), params(i as u64), 5, cache));
            }
            h.run();
            let work = h.eng.executor().chain_layer_positions();
            let sum = h.check_all(spec_on);
            eprintln!("speculative {spec_on} chunk {chunk}: {sum:?} chain work {work}");
            assert!(sum.triangles > 0);
            let outs: Vec<Vec<u32>> = (0..prompts.len() as u64)
                .map(|i| h.outputs[&i].clone())
                .collect();
            results.push((outs, work));
        }
        for (outs, work) in &results[1..] {
            for (i, out) in outs.iter().enumerate() {
                if !spec_on || params(i as u64).is_greedy() {
                    assert_eq!(out, &results[0].0[i], "request {i}");
                }
            }
            if !spec_on {
                assert_eq!(*work, results[0].1, "the chain's work depends on chunking");
            }
        }
    }
}

/// Chunks that cut the triangles and leave windows partly full at every alignment
/// against the 64-slot blocks: every row equals the dense oracle, and the outputs and
/// the chain's work do not depend on the chunking.
#[test]
fn chunked_prefill_is_identical_to_unchunked() {
    chunked_case(&[13, 80, 1000]);
}

/// The same at every chunk size from one token up.
#[test]
#[ignore = "a sweep: about a minute"]
fn chunked_prefill_sweep() {
    chunked_case(&[1, 2, 5, 31, 63, 64, 65, 79, 81, 127, 1000]);
}

/// A regeneration hit, a continuation, and a short turn after a hit resume from
/// windows two blocks of slots wide read back from packed, decrypted cells.
#[test]
fn a_hit_and_a_short_turn_after_it() {
    let g = SamplingParams::greedy();
    for (chunk, spec_on) in [(1000, true), (29, false)] {
        let mut h = wide_harness(512, 5, chunked(chunk, spec_on));
        let mut rng = TestRng(22);
        let prompt = rng.tokens(421);
        assert_eq!(h.run_one(1, &prompt, g, 7, keyed(1)), 0);
        assert_eq!(h.run_one(2, &prompt, g, 7, keyed(1)), 416);
        let mut cont = prompt.clone();
        cont.extend(&h.outputs[&1]);
        cont.extend(rng.tokens(9));
        // 421 + 7 - 1 positions entered KV: the last sealed boundary is 424.
        assert_eq!(h.run_one(3, &cont, g, 7, keyed(1)), 424);
        let mut turn = cont[..432].to_vec();
        turn.extend(rng.tokens(3));
        assert_eq!(h.run_one(4, &turn, g, 2, keyed(1)), 432);
        let sum = h.check_all(spec_on);
        eprintln!("chunk {chunk} speculative {spec_on}: {sum:?}");
    }
}

/// Preemption on a pool far too small, in chunks that preempt sequences in the middle
/// of their triangles.
#[test]
fn preemption_in_the_middle_of_a_triangle() {
    let mut h = wide_harness(72, 5, chunked(23, true));
    let mut rng = TestRng(23);
    for id in 0..5 {
        let prompt = rng.var_tokens(300, 100);
        h.submit(request(
            id,
            prompt,
            SamplingParams::greedy(),
            12,
            keyed(id as u8),
        ));
    }
    h.run();
    let stats = h.eng.stats();
    assert!(stats.preemptions > 0, "workload must force preemption");
    assert!(h.finished.values().all(|f| *f == FinishReason::Length));
    let sum = h.check_all(true);
    eprintln!("{stats:?} {sum:?}");
    assert!(sum.triangles > 0);
}

/// Greedy speculation never changes a token at any width the five-slot drafter has,
/// across acceptance patterns from none to all five, with draft passes whose context
/// has wrapped its ring (prompts past the window).
#[test]
fn greedy_speculation_equals_plain_decoding_at_every_width() {
    let g = SamplingParams::greedy();
    let mut seen = vec![0usize; 6];
    for sampleable in [VOCAB, 2, 1] {
        let mut rng = TestRng(24);
        let prompts: Vec<Vec<u32>> = [3usize, 70, 95, 140]
            .iter()
            .map(|&n| {
                (0..n)
                    .map(|_| rng.below(sampleable as u64) as u32)
                    .collect()
            })
            .collect();
        let mut plain = None;
        for width in 0..=5u32 {
            let mut cfg = exec_config(256, width);
            cfg.sampleable_vocab_size = sampleable;
            let mut h = Harness::new(wide(), cfg, chunked(64, width > 0));
            for (i, p) in prompts.iter().enumerate() {
                h.submit(request(i as u64, p.clone(), g, 16, CacheScope::Private));
            }
            h.run();
            let mut hist = vec![0usize; width as usize + 1];
            for r in h.records.iter().filter(|r| !r.drafts.is_empty()) {
                hist[r.produced.len() - 1] += 1;
            }
            h.check_all(width > 0);
            let outs: Vec<Vec<u32>> = (0..prompts.len() as u64)
                .map(|i| h.outputs[&i].clone())
                .collect();
            assert_eq!(plain.get_or_insert(outs.clone()), &outs, "width {width}");
            if width == 5 {
                eprintln!("sampleable {sampleable}: accepted {hist:?}");
                for (a, n) in hist.iter().enumerate() {
                    seen[a] += n;
                }
            }
        }
    }
    assert!(
        seen[0] > 0 && seen[5] > 0 && seen[1..5].iter().any(|n| *n > 0),
        "{seen:?}"
    );
}

/// A request's target and draft logits are the same bits alone and beside others in
/// another slot.
#[test]
fn a_request_does_not_depend_on_its_batch() {
    let prompt = TestRng(25).tokens(333);
    let p = SamplingParams::random(0.9, 4242).unwrap();
    type Rows = HashMap<(u32, usize), Vec<u32>>;
    let rows_of = |h: &Harness, out: &[u32]| -> Rows {
        let mut all = prompt.clone();
        all.extend(out);
        let mut rows = Rows::new();
        for r in &h.records {
            let last = (r.context_len + r.num_tokens) as usize;
            if r.tokens[..last] != all[..last.min(all.len())] {
                continue;
            }
            for (pos, l) in &r.target_logits {
                if all.get(..=*pos as usize) == Some(&r.tokens[..=*pos as usize]) {
                    rows.insert((*pos, usize::MAX), bits(l));
                }
            }
            let p = r.context_len + r.num_tokens - 1;
            for (i, l) in r.draft_logits.iter().enumerate() {
                if all.get(p as usize + 1..p as usize + 1 + i) == Some(&r.drafts[..i]) {
                    rows.insert((p, i), bits(l));
                }
            }
        }
        rows
    };
    let mut alone = wide_harness(512, 5, chunked(64, true));
    alone.submit(request(0, prompt.clone(), p, 10, CacheScope::Private));
    alone.run();
    let want = rows_of(&alone, &alone.outputs[&0]);
    assert!(want.len() >= 16, "{}", want.len());

    let mut rng = TestRng(26);
    let mut busy = wide_harness(512, 5, chunked(29, true));
    for id in 1..=3 {
        busy.submit(request(
            id,
            rng.var_tokens(60, 200),
            params(id),
            24,
            CacheScope::Private,
        ));
    }
    for _ in 0..3 {
        busy.step();
    }
    busy.submit(request(0, prompt.clone(), p, 10, CacheScope::Private));
    busy.run();
    assert_eq!(busy.outputs[&0], alone.outputs[&0]);
    assert_eq!(rows_of(&busy, &busy.outputs[&0]), want);
    busy.check_all(true);
}
