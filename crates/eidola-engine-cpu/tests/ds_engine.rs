//! The serving core's engine over the DeepSeek-V4.1 CPU executor on the synthetic
//! fixture. Every row of every step is checked bit for bit against the dense oracle
//! (the target logits it sampled or verified from, every draft slot's logits, every
//! draft draw, every produced token), and every finished output against dense
//! generation. The two logical devices run in every test.

mod ds_common;

use ds_common::*;
use eidola_engine::engine::{CacheScope, FinishReason};

#[test]
fn single_requests_match_the_dense_oracle() {
    for spec_on in [false, true] {
        let mut h = Harness::fixture(256, 4, sched(spec_on));
        let mut rng = TestRng(1);
        for id in 0..8 {
            let prompt = rng.var_tokens(1, 60);
            let max = 1 + rng.below(12) as u32;
            h.submit(request(id, prompt, params(id), max, CacheScope::Private));
        }
        h.run();
        for id in 0..8 {
            assert_eq!(h.finished[&id], FinishReason::Length);
        }
        let sum = h.check_all(spec_on);
        eprintln!("speculative {spec_on}: {sum:?}");
        assert!(sum.target_rows > 0);
        assert_eq!(sum.drafted > 0, spec_on);
    }
}

use std::collections::HashMap;
use std::sync::Arc;

use eidola_engine::engine::Request;
use eidola_engine::sampling::SamplingParams;
use eidola_engine_cpu::deepseek::{DsDenseOracle, DsRowRecord};

/// Layer-positions of one triangle ending at `target`, with the drafter on.
fn triangle(target: u32) -> u64 {
    (0..4)
        .map(|j| (TOP + (3 - j) * (WINDOW - 1) + 1).min(target + 1) as u64)
        .sum()
}

/// Prompts below, around and past the chain's reach (189 positions), keyed (a
/// regeneration target before the last token) and private, at chunk sizes that cut the
/// triangles everywhere: every row equals the dense oracle, and the outputs and the
/// chain's work do not depend on the chunking.
#[test]
fn chunked_prefill_is_identical_to_unchunked() {
    let mut rng = TestRng(2);
    let lens = [1usize, 2, 47, 49, 97, 188, 189, 190, 191, 263];
    let prompts: Vec<Vec<u32>> = lens.iter().map(|&n| rng.tokens(n)).collect();
    for spec_on in [false, true] {
        let mut results = Vec::new();
        for chunk in [1u32, 7, 48, 97, 1000] {
            let mut h = Harness::fixture(512, 4, chunked(chunk, spec_on));
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
                // Seeded speculative sampling is exact only in distribution: the first
                // sample is speculative or not depending on how the prompt was chunked.
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

/// A prompt far past the chain's reach runs the decoder over one triangle, whatever its
/// length: exactly the triangle's layer-positions, then every layer at each generated
/// position.
#[test]
fn a_long_prompt_costs_one_triangle() {
    for (len, chunk) in [(400usize, 1000u32), (400, 37), (230, 5)] {
        let mut h = Harness::fixture(512, 4, chunked(chunk, false));
        let prompt = TestRng(3).tokens(len);
        h.submit(request(
            0,
            prompt,
            SamplingParams::greedy(),
            4,
            CacheScope::Private,
        ));
        h.run();
        assert_eq!(triangle(len as u32 - 1), 49 + 96 + 143 + 190);
        assert_eq!(
            h.eng.executor().chain_layer_positions(),
            triangle(len as u32 - 1) + 3 * 4
        );
        h.check_all(false);
    }
}

fn requiring(id: u64, prompt: Vec<u32>, cache: CacheScope, required: u32) -> Request {
    Request {
        required_cached_tokens: required,
        ..request(id, prompt, SamplingParams::greedy(), 4, cache)
    }
}

/// The three hit points resume with full-depth decoder and drafter windows read from
/// packed, decrypted blocks: a regeneration, a continuation, a short turn after a hit,
/// and (from the third request sharing a prefix on) the branch; a requirement the cache
/// holds is admitted and one token more is refused with nothing touched.
#[test]
fn hits_resume_at_regeneration_continuation_and_branch_points() {
    let g = SamplingParams::greedy();
    for (chunk, spec_on) in [(1000, true), (41, true), (1000, false)] {
        let mut h = Harness::fixture(512, 4, chunked(chunk, spec_on));
        let mut rng = TestRng(5);
        let prompt = rng.tokens(250);
        assert_eq!(h.run_one(1, &prompt, g, 6, keyed(1)), 0);
        assert_eq!(h.run_one(2, &prompt, g, 6, keyed(1)), 248);
        let mut cont = prompt.clone();
        cont.extend(&h.outputs[&1]);
        cont.extend(rng.tokens(9));
        // 250 + 6 - 1 positions entered KV: the last sealed boundary is 248.
        assert_eq!(h.run_one(3, &cont, g, 6, keyed(1)), 248);
        // A short turn after a hit: every layer runs on from the hit's windows.
        let mut turn = cont[..264].to_vec();
        turn.extend(rng.tokens(3));
        assert_eq!(h.run_one(4, &turn, g, 2, keyed(1)), 264);

        let base = rng.tokens(240);
        let turn = |rng: &mut TestRng, first: u32| {
            let mut t = base[..200].to_vec();
            t.push(first);
            t.extend(rng.tokens(30));
            t
        };
        assert_eq!(h.run_one(5, &base, g, 6, keyed(2)), 0);
        // The first divergence finds banks without windows at 200, recomputes, and
        // keeps the branch; the next one resumes there.
        let first = turn(&mut rng, (base[200] + 1) % VOCAB);
        assert_eq!(h.run_one(6, &first, g, 6, keyed(2)), 0);
        let second = turn(&mut rng, (base[200] + 2) % VOCAB);
        assert_eq!(h.run_one(7, &second, g, 6, keyed(2)), 200);
        // A branch hit one token before the end: the only row is the drafting target.
        let short = turn(&mut rng, (base[200] + 3) % VOCAB)[..201].to_vec();
        assert_eq!(h.run_one(8, &short, g, 6, keyed(2)), 200);

        // A required prefix: exactly the hit is admitted; one token more is refused
        // before anything is mapped, decrypted or computed.
        let looked = h.eng.kv().lookup(&salt(2), &second, h.now);
        assert_eq!(looked, 224, "its own regeneration point");
        let before = (format!("{:?}", h.eng.kv()), h.eng.executor().steps());
        h.submit(requiring(9, second.clone(), keyed(2), looked + 1));
        h.step();
        assert_eq!(h.finished[&9], FinishReason::CachedPrefixMissing);
        let after = (format!("{:?}", h.eng.kv()), h.eng.executor().steps());
        assert_eq!(before, after, "the refused request touched something");
        h.submit(requiring(10, second.clone(), keyed(2), looked));
        h.run();
        assert_eq!(h.cached[&10], looked);
        h.finished.remove(&9);

        let sum = h.check_all(spec_on);
        eprintln!("chunk {chunk} speculative {spec_on}: {sum:?}");
    }
}

/// A sequence that leaves in the middle of its triangle retains no hit point at its
/// last sealed boundary (its decoder windows there are partial), and one that leaves
/// after its triangle does.
#[test]
fn a_point_without_full_depth_windows_is_not_retained() {
    let g = SamplingParams::greedy();
    let prompt = TestRng(6).tokens(260);
    // The regeneration target is 255 and the chain's bottom layer starts 189 before it.
    for (steps, hit) in [(10, 0), (40, 256)] {
        let mut h = Harness::fixture(512, 0, chunked(8, false));
        h.submit(request(1, prompt.clone(), g, 40, keyed(1)));
        for _ in 0..steps {
            h.step();
        }
        assert!(h.eng.cancel(1, h.now).is_some());
        h.flush();
        h.records.clear();
        assert_eq!(h.run_one(2, &prompt, g, 4, keyed(1)), hit);
        h.check_all(false);
    }
}

/// Recompute preemption on a pool far too small for the workload, in whole prompts and
/// in chunks that preempt sequences in the middle of their triangles: every row equals
/// the dense oracle and every output dense generation.
#[test]
fn preemption_and_resume_reproduce_uninterrupted_outputs() {
    for (spec_on, chunk) in [(false, 512), (true, 512), (true, 11)] {
        let mut h = Harness::fixture(40, 4, chunked(chunk, spec_on));
        let mut rng = TestRng(8);
        for id in 0..12 {
            let prompt = rng.var_tokens(20, 60);
            let p = if spec_on {
                SamplingParams::greedy()
            } else {
                params(id)
            };
            h.submit(request(id, prompt, p, 24, keyed(id as u8)));
        }
        h.run();
        let stats = h.eng.stats();
        assert!(stats.preemptions > 0, "workload must force preemption");
        let sum = h.check_all(spec_on);
        eprintln!("speculative {spec_on} chunk {chunk}: {stats:?} {sum:?}");
    }
}

/// Under KV pressure a lone sequence sheds drafts, then its prefill chunk, then its own
/// retention, and only then ends for lack of memory: whatever it produced is the dense
/// oracle's, and with a few blocks more it finishes.
#[test]
fn kv_pressure_sheds_before_it_ends_a_sequence() {
    let g = SamplingParams::greedy();
    let prompt = TestRng(9).tokens(44);
    let mut endings = Vec::new();
    for blocks in [8u32, 9, 10, 14] {
        let mut h = Harness::fixture(blocks, 4, chunked(64, true));
        h.submit(request(1, prompt.clone(), g, 24, keyed(1)));
        h.run();
        assert_eq!(h.eng.stats().preemptions, 0);
        h.check_all(true);
        endings.push((h.finished[&1], h.outputs[&1].len(), h.eng.stats().drafted));
    }
    eprintln!("{endings:?}");
    assert_eq!(endings[0].0, FinishReason::KvExhausted);
    assert_eq!((endings[3].0, endings[3].1), (FinishReason::Length, 24));
    assert!(endings[1].2 < endings[3].2, "a tight pool drafts less");
}

fn accept_histogram(records: &[DsRowRecord], width: usize) -> Vec<usize> {
    let mut hist = vec![0; width + 1];
    for r in records.iter().filter(|r| !r.drafts.is_empty()) {
        hist[r.produced.len() - 1] += 1;
    }
    hist
}

/// Greedy speculation never changes a token, at every draft width the drafter has and
/// across acceptance patterns from "every draft rejected" to "every draft accepted".
/// The patterns come from narrowing the sampleable vocabulary (one id: the target and
/// the drafter must agree) and from a drafter whose Markov term is removed.
#[test]
fn greedy_speculation_equals_plain_decoding_at_every_width() {
    let g = SamplingParams::greedy();
    let mut seen = vec![0usize; 5];
    for (sampleable, plain_markov) in [
        (VOCAB, true),
        (VOCAB, false),
        (3, true),
        (2, false),
        (1, true),
    ] {
        let model = if plain_markov {
            fixture()
        } else {
            let mut m = load_fixture();
            m.weights
                .drafter
                .as_mut()
                .unwrap()
                .markov_head
                .data
                .fill(0.0);
            Arc::new(m)
        };
        let mut rng = TestRng(10);
        let prompts: Vec<Vec<u32>> = (0..5)
            .map(|_| {
                let n = 2 + rng.below(70) as usize;
                (0..n)
                    .map(|_| rng.below(sampleable as u64) as u32)
                    .collect()
            })
            .collect();
        let mut plain = None;
        for width in 0..=4u32 {
            let mut cfg = exec_config(256, width);
            cfg.sampleable_vocab_size = sampleable;
            let mut h = Harness::new(model.clone(), cfg, chunked(64, width > 0));
            for (i, p) in prompts.iter().enumerate() {
                h.submit(request(i as u64, p.clone(), g, 20, CacheScope::Private));
            }
            h.run();
            let hist = accept_histogram(&h.records, width as usize);
            h.check_all(width > 0);
            let outs: Vec<Vec<u32>> = (0..prompts.len() as u64)
                .map(|i| h.outputs[&i].clone())
                .collect();
            assert!(outs.iter().flatten().all(|t| *t < sampleable));
            assert_eq!(plain.get_or_insert(outs.clone()), &outs, "width {width}");
            if width == 4 {
                eprintln!("sampleable {sampleable} markov {plain_markov}: accepted {hist:?}");
                for (a, n) in hist.iter().enumerate() {
                    seen[a] += n;
                }
            }
        }
    }
    assert!(
        seen.iter().all(|n| *n > 0),
        "accepted-count coverage {seen:?}"
    );
}

/// Seeded sampled rows with drafting on reproduce the oracle's draws at every step: the
/// draft chain on the draft stream, each slot conditioned on the draft before it, and
/// chain acceptance over the conditional distributions.
#[test]
fn seeded_sampling_is_exact_per_step() {
    for width in [2u32, 4] {
        let mut h = Harness::fixture(256, width, sched(true));
        let mut rng = TestRng(11);
        for id in 0..6 {
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
                rng.var_tokens(3, 50),
                p,
                14,
                CacheScope::Private,
            ));
        }
        h.run();
        let sum = h.check_all(true);
        eprintln!("width {width}: {sum:?}");
        assert!(sum.drafted > 0 && sum.accepted < sum.drafted);
    }
}

/// A request's logits are the same bits alone or beside any other requests, in any
/// slot: every target row and every draft slot's logits of its records.
#[test]
fn a_request_does_not_depend_on_its_batch() {
    let prompt = TestRng(12).tokens(203);
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
                // Only rows on the request's own sequence are comparable.
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
    let mut alone = Harness::fixture(512, 4, chunked(64, true));
    alone.submit(request(0, prompt.clone(), p, 12, CacheScope::Private));
    alone.run();
    let want = rows_of(&alone, &alone.outputs[&0]);
    assert!(want.len() > 20);

    let mut rng = TestRng(13);
    let mut busy = Harness::fixture(512, 4, chunked(29, true));
    for id in 1..=4 {
        busy.submit(request(
            id,
            rng.var_tokens(30, 200),
            params(id),
            30,
            CacheScope::Private,
        ));
    }
    for _ in 0..3 {
        busy.step();
    }
    busy.submit(request(0, prompt.clone(), p, 12, CacheScope::Private));
    busy.run();
    assert_eq!(busy.outputs[&0], alone.outputs[&0]);
    let got = rows_of(&busy, &busy.outputs[&0]);
    assert_eq!(got, want);
    busy.check_all(true);
}

/// What crosses between the two devices in a step is exactly the declared set: from
/// device 0, one packed bank entry for every position and a boundary row for every
/// position the chain's bottom layer computes (with the encoder-side drafter taps where
/// the drafter's context is written); from device 1, the drafted token ids and the rows
/// gathered from the Engram table it holds.
#[test]
fn the_cross_device_transfer_is_exactly_the_declared_set() {
    let mut h = Harness::fixture(512, 4, chunked(53, true));
    let mut rng = TestRng(14);
    for id in 0..3 {
        h.submit(request(
            id,
            rng.var_tokens(150, 150),
            params(id),
            8,
            keyed(id as u8),
        ));
    }
    let (mut steps, mut partial, mut drafted) = (0, 0, 0);
    while h.eng.unfinished() > 0 {
        h.records.clear();
        h.step();
        h.now += 1;
        let transfers = h.take_transfers();
        if h.records.is_empty() {
            assert!(transfers.is_empty());
            continue;
        }
        let [t] = &transfers[..] else {
            panic!("one transfer record a step");
        };
        let (mut entries, mut rows, mut taps, mut drafts) = (vec![], vec![], vec![], vec![]);
        for r in &h.records {
            let bottom = h.spec.chain_layer_start(r.chain_start, 0);
            let end = r.context_len + r.num_tokens + r.drafts.len() as u32;
            for pos in r.context_len..end {
                entries.push((r.slot, pos));
                if pos >= bottom {
                    rows.push((r.slot, pos));
                }
                if pos >= r.chain_start {
                    taps.push((r.slot, pos));
                }
            }
            if !r.drafts.is_empty() {
                drafts.push((r.slot, r.drafts.clone()));
            }
            partial += usize::from(bottom > r.context_len);
        }
        assert_eq!(t.bank_entries, entries);
        assert_eq!(
            t.bank_entry_bytes,
            18 + 17,
            "packed KV, then the packed index key"
        );
        assert_eq!(t.boundary_rows, rows);
        assert_eq!(t.taps, taps);
        assert_eq!(t.drafts, drafts);
        // The second Engram table sits beside the decoder; the first never crosses.
        let gathered: Vec<_> = entries.iter().map(|(s, p)| (*s, *p, 1)).collect();
        assert_eq!(t.gathered, gathered);
        steps += 1;
        drafted += drafts.len();
    }
    assert!(steps > 10 && partial > 0 && drafted > 0);
}

/// The oracle's draft pass (context KV recomputed densely, the pass given it
/// explicitly) is the reference's own entry point wherever that one drafts.
#[test]
fn the_oracle_drafts_as_the_reference_entry_point_does() {
    let model = fixture();
    let oracle = DsDenseOracle {
        model: &model,
        sampleable_vocab_size: VOCAB,
    };
    let tokens = TestRng(15).tokens(70);
    let dense = oracle.run(&tokens).unwrap();
    let out = model
        .forward(&mut model.new_state(), &tokens, &Default::default())
        .unwrap();
    let hidden = out.main_hidden.unwrap();
    let mut state = model.new_drafter_state();
    model.drafter_observe(&mut state, hidden.row(0)).unwrap();
    for p in 2..tokens.len() as u32 {
        let want = model
            .drafter_forward(&mut state, hidden.row(p as usize - 1), tokens[p as usize])
            .unwrap();
        let got = oracle.draft(&dense, p).unwrap();
        assert_eq!(
            bits(&got.base_logits.data),
            bits(&want.base_logits.data),
            "row {p}"
        );
        assert_eq!(bits(&got.hidden.data), bits(&want.hidden.data));
    }
}
