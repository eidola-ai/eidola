//! Sparse attention over a sliding window plus selected global entries
//! (`inference/kernel.py` `sparse_attn_kernel`; `inference/model.py`
//! `Indexer`, `select_candidate_blocks`).
//!
//! One KV vector per position serves as key and value for every head.

use super::numeric::pinned_sum;
use super::numeric::{bf16_round, exp32};
use crate::tensor::dot;

/// Slots per block of the online softmax.
pub const ATTENTION_BLOCK: usize = 64;

/// One query's attention: every head against the listed entries.
///
/// `slots` is the canonical list for the query: the window's slots in
/// ascending position (an absent position is `None`), then the selected
/// global entries in ascending position. It is processed in blocks of
/// [`ATTENTION_BLOCK`] consecutive slots; a slot's place in the list decides
/// its block, whether or not other slots are empty. Per head, per block:
///
/// 1. `s_j = dot(q, kv_j) · scale` in the pinned order;
/// 2. the running maximum moves to `max(m, max_j s_j)`, and the running sums
///    are rescaled by `r = exp(m_old - m)`;
/// 3. `p_j = exp(s_j - m)`; the normaliser gains `Σ_j p_j`, left to right;
/// 4. the value sum gains `bf16(p_j) · kv_j` for each entry in slot order.
///
/// Finally the normaliser gains the sink, `exp(sink - m)`, which contributes
/// no value, and the output is `bf16(values / normaliser)`. A query with no
/// entry at all outputs zeros.
pub fn sparse_attend(
    q: &[f32],
    slots: &[Option<&[f32]>],
    sink: &[f32],
    scale: f32,
    out: &mut [f32],
) {
    let heads = sink.len();
    let d = q.len() / heads;
    assert_eq!(out.len(), q.len());
    let mut scores = [0.0f32; ATTENTION_BLOCK];
    for h in 0..heads {
        let qh = &q[h * d..(h + 1) * d];
        let acc = &mut out[h * d..(h + 1) * d];
        acc.fill(0.0);
        let (mut max, mut den) = (-1e30f32, 0.0f32);
        for block in slots.chunks(ATTENTION_BLOCK) {
            let mut block_max = f32::NEG_INFINITY;
            for (s, slot) in scores.iter_mut().zip(block) {
                if let Some(kv) = slot {
                    *s = dot(qh, kv) * scale;
                    block_max = block_max.max(*s);
                }
            }
            if block_max == f32::NEG_INFINITY {
                continue;
            }
            let prev = max;
            max = max.max(block_max);
            let r = exp32(prev - max);
            for a in acc.iter_mut() {
                *a *= r;
            }
            let mut block_sum = 0.0f32;
            for (s, slot) in scores.iter().zip(block) {
                if let Some(kv) = slot {
                    let p = exp32(*s - max);
                    block_sum += p;
                    let weight = bf16_round(p);
                    for (a, &v) in acc.iter_mut().zip(*kv) {
                        *a += weight * v;
                    }
                }
            }
            den = den * r + block_sum;
        }
        den += exp32(sink[h] - max);
        for a in acc.iter_mut() {
            *a = bf16_round(*a / den);
        }
    }
}

/// The total order every top-k uses: higher score first, then lower index.
/// `-0.0` and `0.0` are one score.
fn by_score_then_index(scores: &[f32], a: usize, b: usize) -> std::cmp::Ordering {
    let key = |i: usize| if scores[i] == 0.0 { 0.0 } else { scores[i] };
    key(b).total_cmp(&key(a)).then(a.cmp(&b))
}

/// Indices of the `k` best of `candidates` under the total order (higher
/// score first, then lower index), in that order.
pub fn top_k(scores: &[f32], candidates: impl Iterator<Item = usize>, k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = candidates.collect();
    order.sort_unstable_by(|&a, &b| by_score_then_index(scores, a, b));
    order.truncate(k);
    order
}

/// One global entry's index score for a query: per index head `h`,
/// `bf16(dot(q_h, key))`, rectified, times `weights[h]` and rounded to BF16;
/// the head terms are summed in the pinned order and rounded to BF16.
pub fn index_score(q: &[f32], key: &[f32], weights: &[f32]) -> f32 {
    let d = key.len();
    let terms: Vec<f32> = weights
        .iter()
        .enumerate()
        .map(|(h, &w)| {
            let s = bf16_round(dot(&q[h * d..(h + 1) * d], key));
            bf16_round(if s > 0.0 { s } else { 0.0 } * w)
        })
        .collect();
    bf16_round(pinned_sum(&terms))
}

/// The candidate blocks a query keeps: `scores[t]` for the `visible` entries
/// it can reach, grouped into blocks of `block_size`; a block scores as its
/// best entry, the block holding the newest entry is always kept, and the
/// best `top_blocks` survive. Returns one flag per visible entry.
pub fn candidate_mask(
    scores: &[f32],
    visible: usize,
    top_blocks: usize,
    block_size: usize,
) -> Vec<bool> {
    if visible == 0 {
        return Vec::new();
    }
    let blocks = visible.div_ceil(block_size);
    let mut block_scores: Vec<f32> = scores[..visible]
        .chunks(block_size)
        .map(|b| b.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)))
        .collect();
    block_scores[(visible - 1) / block_size] = f32::INFINITY;
    let kept = top_k(&block_scores, 0..blocks, top_blocks);
    let mut keep = vec![false; blocks];
    for b in kept {
        keep[b] = block_scores[b] > f32::NEG_INFINITY;
    }
    (0..visible).map(|t| keep[t / block_size]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_breaks_ties_by_lower_index() {
        let scores = [1.0, 3.0, 3.0, -0.0, 0.0, 2.0, 3.0];
        assert_eq!(top_k(&scores, 0..7, 4), [1, 2, 6, 5]);
        assert_eq!(top_k(&scores, 0..5, 9), [1, 2, 0, 3, 4]);
    }

    #[test]
    fn candidate_mask_pins_the_newest_block() {
        // Blocks of 2: scores (5,1) (4,4) (0,0) (3,-) with 7 visible entries.
        let scores = [5.0, 1.0, 4.0, 4.0, 0.0, 0.0, 3.0];
        let mask = candidate_mask(&scores, 7, 2, 2);
        // The newest block (3) is pinned; the best other block is block 0.
        assert_eq!(mask, [true, true, false, false, false, false, true]);
        let all = candidate_mask(&scores, 7, 8, 2);
        assert!(all.iter().all(|k| *k));
    }

    #[test]
    fn attention_with_one_entry_and_a_sink() {
        let q = [1.0f32, 0.0];
        let kv = [2.0f32, 4.0];
        let mut out = [0.0f32; 2];
        // One head: p = exp(0) = 1 and the sink adds exp(sink - s).
        sparse_attend(&q, &[None, Some(&kv)], &[2.0], 1.0, &mut out);
        assert_eq!(out, [1.0, 2.0]);
        sparse_attend(&q, &[None, None], &[0.0], 1.0, &mut out);
        assert_eq!(out, [0.0, 0.0]);
    }

    /// An empty slot keeps its place in the block structure but adds nothing.
    #[test]
    fn blocks_follow_slot_positions() {
        let q: Vec<f32> = (0..8).map(|i| (i as f32 * 0.7).sin()).collect();
        let kvs: Vec<Vec<f32>> = (0..70)
            .map(|t| {
                (0..8)
                    .map(|i| ((t * 8 + i) as f32 * 0.37).cos() * 3.0)
                    .collect()
            })
            .collect();
        let run = |slots: &[Option<&[f32]>]| {
            let mut out = vec![0.0; 8];
            sparse_attend(&q, slots, &[0.3], 0.5, &mut out);
            out
        };
        let packed: Vec<Option<&[f32]>> = kvs.iter().map(|k| Some(k.as_slice())).collect();
        // Leading empty slots that do not move a block boundary change nothing.
        let mut shifted: Vec<Option<&[f32]>> = vec![None; 64];
        shifted.extend(packed.iter().copied());
        let mut padded = packed.clone();
        padded.extend([None; 20]);
        assert_eq!(run(&packed), run(&padded));
        let mut whole_block_first: Vec<Option<&[f32]>> = vec![None; 64];
        whole_block_first.extend(packed.iter().copied());
        assert_eq!(run(&packed), run(&whole_block_first));
        assert_eq!(run(&shifted), run(&whole_block_first));
    }
}
