//! Window pools sized apart from the banks, over the DeepSeek-V4.1 CPU executor on the
//! synthetic fixture (block 8, window 48: a live window is at most `ceil(48 / 8) + 1 = 7`
//! blocks). Sequences many times longer than their window pools, their continuations and
//! regenerations, and workloads whose windows do not all fit at once: every row equals
//! the dense oracle bit for bit and every output dense generation.

mod ds_common;

use ds_common::*;
use eidola_engine::engine::{FinishReason, SchedulerConfig};
use eidola_engine::sampling::SamplingParams;

/// The fixture with 512 bank blocks (4,096 positions) and `window_blocks` in every window
/// group, the null block among them; the drafter on.
fn windowed(window_blocks: u32, sched: SchedulerConfig) -> Harness {
    let mut cfg = exec_config(512, 4);
    cfg.window_blocks = window_blocks;
    let h = Harness::new(fixture(), cfg, sched);
    for g in &h.spec.kv_groups {
        let want = match g.attention {
            eidola_engine::spec::AttentionKind::Full => 512,
            eidola_engine::spec::AttentionKind::Sliding { .. } => window_blocks,
        };
        assert_eq!(g.num_blocks, want, "{}", g.name);
    }
    h
}

/// One conversation of 700-position prompts through window pools of 38 blocks, the
/// node's rule for one seat and a 64-token step (`1 + (4 × 7 + 1) + 8`: a live window,
/// the block after it, three retained windows, and a chunk's eight blocks): a first
/// turn, a continuation resuming at its last sealed boundary, and a regeneration at the
/// first prompt's, each running hundreds of positions past what the pools could hold
/// whole, without a preemption.
#[test]
fn long_sequences_run_through_small_window_pools() {
    let g = SamplingParams::greedy();
    for spec_on in [false, true] {
        let mut h = windowed(38, chunked(64, spec_on));
        let mut rng = TestRng(60);
        let prompt = rng.tokens(700);
        assert_eq!(h.run_one(1, &prompt, g, 40, keyed(1)), 0);
        let mut cont = prompt.clone();
        cont.extend(&h.outputs[&1]);
        cont.extend(rng.tokens(150));
        // 700 + 40 - 1 positions entered KV: the last sealed boundary is 736.
        assert_eq!(h.run_one(2, &cont, g, 40, keyed(1)), 736);
        // The first prompt's regeneration point, the last boundary before its final token.
        assert_eq!(h.run_one(3, &prompt, g, 40, keyed(1)), 696);
        for id in 1..=3 {
            assert_eq!(h.finished[&id], FinishReason::Length, "request {id}");
        }
        assert_eq!(h.eng.stats().preemptions, 0);
        let sum = h.check_all(spec_on);
        eprintln!("speculative {spec_on}: {sum:?}");
        assert_eq!(sum.drafted > 0, spec_on);
    }
}

/// Six concurrent keyed sequences of 150-250 positions whose windows do not fit at once
/// (17 window blocks: two sequences' live windows and the blocks after them): the window
/// pools preempt and shed as any pool does, in whole prompts and in chunks that cut
/// triangles, and every row, every hit and every output is still the dense oracle's.
#[test]
fn window_pressure_preempts_and_resumes_exactly() {
    for (spec_on, chunk) in [(false, 64), (true, 64), (true, 11)] {
        let mut h = windowed(17, chunked(chunk, spec_on));
        let mut rng = TestRng(61);
        for id in 0..6 {
            let prompt = rng.var_tokens(150, 100);
            let p = if spec_on {
                SamplingParams::greedy()
            } else {
                params(id)
            };
            h.submit(request(id, prompt, p, 24, keyed(id as u8)));
        }
        h.run();
        let stats = h.eng.stats();
        assert!(
            stats.preemptions > 0,
            "the window pools must force preemption"
        );
        for id in 0..6 {
            assert_eq!(h.finished[&id], FinishReason::Length, "request {id}");
        }
        let sum = h.check_all(spec_on);
        eprintln!("speculative {spec_on} chunk {chunk}: {stats:?} {sum:?}");
    }
}
