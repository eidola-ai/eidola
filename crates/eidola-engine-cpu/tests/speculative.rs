//! Speculative decoding statistics on small-vocabulary models derived from the fixture
//! (vocabulary cut to its first few ids), where drafts agree with the target often enough
//! to exercise every acceptance path: greedy output equals plain decoding, acceptance
//! actually happens (including whole accepted chains), and seeded speculative sampling
//! draws from exactly the target distribution.

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use common::*;
use eidola_engine::engine::CacheScope;
use eidola_engine::sampling::{self, Logits, SamplingParams};
use eidola_engine_cpu::{DenseOracle, MtpHidden};
use eidola_engine_model::ReferenceModel;

fn small(vocab: usize) -> Arc<ReferenceModel> {
    small_vocab_fixture(vocab, 1.0, &[0, 1])
}

#[test]
fn greedy_speculation_accepts_drafts_and_equals_plain_decoding() {
    let m = echo_drafter_fixture(4, &[0, 1, 2, 3], 0.05);
    let mut rng = TestRng(21);
    let prompts: Vec<Vec<u32>> = (0..6).map(|_| rng.var_tokens(2, 20, 4)).collect();
    let run = |depths: Vec<usize>, spec_on: bool| {
        let mut cfg = exec_config(256, depths);
        cfg.sampleable_vocab_size = 4;
        let mut h = Harness::new(m.clone(), cfg, sched(spec_on));
        for (i, p) in prompts.iter().enumerate() {
            h.submit(request(
                i as u64,
                p.clone(),
                SamplingParams::greedy(),
                24,
                CacheScope::Private,
            ));
        }
        h.run();
        let full_chains = h
            .records
            .iter()
            .filter(|r| !r.drafts.is_empty() && r.produced.len() == r.drafts.len() + 1)
            .count();
        let sum = h.check_all(spec_on);
        let outs: Vec<Vec<u32>> = (0..prompts.len() as u64)
            .map(|i| h.outputs[&i].clone())
            .collect();
        (outs, sum, full_chains)
    };
    let (plain, _, _) = run(vec![0, 1], false);
    for depths in [vec![0], vec![0, 1], vec![0, 1, 0]] {
        let (spec, sum, full) = run(depths.clone(), true);
        assert_eq!(spec, plain, "depths {depths:?}");
        eprintln!(
            "echo drafter, vocab 4, greedy, depths {depths:?}: accepted {}/{} drafts ({:.1} %), per depth \
             (proposed, accepted) {:?}, fully accepted chains {full}",
            sum.accepted,
            sum.drafted,
            100.0 * sum.accepted as f64 / sum.drafted as f64,
            &sum.per_depth[..depths.len()]
        );
        assert!(sum.accepted > 0 && sum.accepted < sum.drafted);
        if depths.len() > 1 {
            assert!(full > 0, "some chain is accepted whole");
        }
    }
}

/// Upper 0.1 % point of chi-square with `df` degrees of freedom (Wilson–Hilferty).
fn chi2_critical(df: usize) -> f64 {
    let k = df as f64;
    let z = 3.090_232;
    k * (1.0 - 2.0 / (9.0 * k) + z * (2.0 / (9.0 * k)).sqrt()).powi(3)
}

/// Joint distribution of the first three generated tokens (vocabulary 4, so 64 cells)
/// under speculative sampling with three chained draft depths, against the exact
/// distribution computed from the dense oracle's processed probabilities; chi-square
/// goodness of fit at p = 0.001.
#[test]
fn speculative_sampling_draws_from_the_target_distribution() {
    let m = small(4);
    let prompt = vec![1u32, 2, 3, 0, 1];
    let n = 2000u64;
    let oracle = DenseOracle {
        model: &m,
        mtp_depths: &[0, 1, 0],
        mtp_hidden: MtpHidden::Normed,
        sampleable_vocab_size: 4,
    };
    for (ci, base) in [
        SamplingParams::random(1.0, 0),
        SamplingParams {
            temperature: 0.8,
            top_k: 3,
            top_p: 0.95,
            min_p: 0.0,
            seed: 0,
        },
    ]
    .into_iter()
    .enumerate()
    {
        // Exact P(t1, t2, t3).
        let mut exact = [0.0f64; 64];
        let mut probs_after: HashMap<Vec<u32>, Vec<f64>> = HashMap::new();
        let mut probs = |seq: &[u32]| {
            probs_after
                .entry(seq.to_vec())
                .or_insert_with(|| {
                    let r = oracle.run(seq).unwrap();
                    let row = r.target(seq.len() as u32 - 1);
                    sampling::processed_probs(Logits::new(row, 4), &base)
                })
                .clone()
        };
        for cell in 0..64u32 {
            let t = [cell / 16, cell / 4 % 4, cell % 4];
            let mut seq = prompt.clone();
            let mut p = 1.0;
            for &x in &t {
                p *= probs(&seq)[x as usize];
                seq.push(x);
            }
            exact[cell as usize] = p;
        }

        let mut cfg = exec_config(64, vec![0, 1, 0]);
        cfg.sampleable_vocab_size = 4;
        cfg.record = false;
        let mut h = Harness::new(m.clone(), cfg, sched(true));
        h.checks = false; // statistics only; invariants are covered elsewhere
        for seed in 0..n {
            let p = SamplingParams {
                seed: seed + 1000 * ci as u64,
                ..base
            };
            h.submit(request(
                seed,
                prompt.clone(),
                p,
                3,
                CacheScope::Keyed(salt(1)),
            ));
            if h.eng.unfinished() >= 32 {
                h.run();
            }
        }
        h.run();
        let mut counts = [0u64; 64];
        for seed in 0..n {
            let r = &h.outputs[&seed];
            counts[(r[0] * 16 + r[1] * 4 + r[2]) as usize] += 1;
        }
        // Cells expected fewer than 5 times are pooled.
        let (mut chi, mut cells, mut pool_e, mut pool_o) = (0.0, 0usize, 0.0, 0.0);
        for c in 0..64 {
            let e = exact[c] * n as f64;
            if e >= 5.0 {
                chi += (counts[c] as f64 - e).powi(2) / e;
                cells += 1;
            } else {
                pool_e += e;
                pool_o += counts[c] as f64;
            }
        }
        if pool_e > 0.0 {
            chi += (pool_o - pool_e).powi(2) / pool_e.max(1e-9);
            cells += 1;
        }
        let s = h.eng.stats();
        let crit = chi2_critical(cells - 1);
        eprintln!(
            "config {ci}: chi-square {chi:.1} over {cells} cells (critical {crit:.1}); \
             accepted {}/{} drafts ({:.1} %)",
            s.accepted,
            s.drafted,
            100.0 * s.accepted as f64 / s.drafted as f64
        );
        assert!(s.accepted > 0 && s.accepted < s.drafted);
        assert!(chi < crit, "config {ci}: chi-square {chi} >= {crit}");
    }
}
