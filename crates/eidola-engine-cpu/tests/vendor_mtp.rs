//! The executor's draft chain against a direct transcription of the model vendor's MTP
//! inference indexing, written independently of the executor's slot layout and of the
//! dense oracle: MTP layer `k` at anchor `x` combines the main model's (normed) hidden
//! state `h_x` with the token `t_{x+k+1}`, attends at RoPE position `x` over its own KV
//! (one row per anchor), and predicts `t_{x+k+2}`. Every layer conditions on the main
//! model's state, never on another MTP layer's output.
//!
//! A draft chain that feeds one depth's output into the next (or shifts any depth's
//! token, hidden row or position) fails this bit for bit at depth 1 and beyond.

mod common;

use std::collections::HashMap;

use common::*;
use eidola_engine::engine::CacheScope;
use eidola_engine::sampling::{self, Logits};
use eidola_engine_model::{ForwardOptions, LogitsAt, Matrix, ReferenceModel};

/// Logits of MTP layer `layer`, as draft depth `k`, at every anchor `x` with `t_{x+k+1}`
/// inside `tokens` (row `x` is anchor `x`).
fn vendor_logits(model: &ReferenceModel, layer: usize, k: usize, tokens: &[u32]) -> Matrix {
    let fwd = model
        .forward(
            tokens,
            &ForwardOptions {
                logits: LogitsAt::Last,
                capture_layers: false,
            },
        )
        .unwrap();
    let anchors: Vec<usize> = (0..tokens.len() - (k + 1)).collect();
    let next: Vec<u32> = anchors.iter().map(|&x| tokens[x + k + 1]).collect();
    let h = fwd.hidden_normed.select_rows(&anchors);
    model
        .mtp_forward(layer, &next, &h, &anchors, &LogitsAt::All)
        .unwrap()
        .logits
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn draft_chain_matches_vendor_mtp_indexing() {
    let depths = vec![0, 1, 0];
    let mut rng = TestRng(0x7e2d);
    let mut h = Harness::fixture(512, depths.clone(), sched(true));
    for i in 0..8u64 {
        let prompt = rng.var_tokens(3, 30, VOCAB);
        h.submit(request(i, prompt, params(i), 20, CacheScope::Private));
    }
    h.run();

    let mut dense: HashMap<(usize, Vec<u32>), Matrix> = HashMap::new();
    let mut rows = [0usize; 3];
    let mut greedy_drafts = [0usize; 3];
    for r in &h.records {
        for (d, s, l) in &r.drafter_logits {
            let x = (*s as usize) - 1 - d;
            let m = dense
                .entry((*d, r.tokens.clone()))
                .or_insert_with(|| vendor_logits(&h.model, depths[*d], *d, &r.tokens));
            assert_eq!(
                bits(l),
                bits(m.row(x)),
                "depth {d} (layer {}) at anchor {x}, slot {s}",
                depths[*d]
            );
            rows[*d] += 1;
        }
        // Draft `i` verified at position `p + 1 + i` was drafted from anchor `p - 1`:
        // layer `i` predicts `t_{x+i+2}`.
        if r.sampling.is_greedy() && !r.drafts.is_empty() {
            let p = (r.context_len + r.num_tokens - 1) as usize;
            for (i, &t) in r.drafts.iter().enumerate() {
                let m = dense
                    .entry((i, r.tokens.clone()))
                    .or_insert_with(|| vendor_logits(&h.model, depths[i], i, &r.tokens));
                let logits = Logits::new(m.row(p - 1), VOCAB);
                assert_eq!(
                    sampling::argmax(logits),
                    t,
                    "greedy draft {i} at {}",
                    p + 1 + i
                );
                greedy_drafts[i] += 1;
            }
        }
    }
    eprintln!("vendor-indexed drafter rows per depth {rows:?}, greedy drafts {greedy_drafts:?}");
    assert!(
        rows.iter().all(|&n| n > 50),
        "every depth checked: {rows:?}"
    );
    assert!(
        greedy_drafts.iter().all(|&n| n > 5),
        "greedy drafts at every depth: {greedy_drafts:?}"
    );
}
