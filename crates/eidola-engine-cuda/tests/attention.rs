//! FlashInfer's FA2 sink attention over the executor's paged KV pools, against
//! the model crate's reference `attend`, for global and sliding layers,
//! prefill chunks and decode rows, on every image this device runs.

mod common;

use common::{Lcg, setup};
use eidola_engine_cuda::attention::{Attention, AttnLayer, AttnRequest, HEAD_DIM_QK, HEAD_DIM_VO};
use eidola_engine_cuda::bf16;
use eidola_engine_cuda::kv::{GroupGeometry, KvStore};
use eidola_engine_cuda::launch::dptr;
use eidola_engine_model::attention::attend;
use eidola_engine_model::config::{AttentionKind, AttentionSpec};

const NQ: u32 = 64;
const BS: u32 = 16;

struct Row {
    /// Positions already in KV before this row's queries.
    context: u32,
    queries: u32,
}

fn run(window: Option<u32>, kv_heads: u32, rows: &[Row], seed: u64) {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    let mut rng = Lcg(seed);
    let layers = 3u32;
    let layer = 1u32;
    let total_blocks: u32 = rows
        .iter()
        .map(|r| (r.context + r.queries).div_ceil(BS))
        .sum::<u32>()
        + 1;
    let geom = GroupGeometry {
        num_layers: layers,
        num_kv_heads: kv_heads,
        head_dim_qk: HEAD_DIM_QK,
        head_dim_v: HEAD_DIM_VO,
        block_size: BS,
        num_blocks: total_blocks,
    };
    let mut store = KvStore::new(gpu, vec![geom.clone()], 1, 1, 0).unwrap();
    // Physical blocks handed out in a scrambled order.
    let mut free: Vec<u32> = (1..total_blocks).collect();
    for i in (1..free.len()).rev() {
        free.swap(i, rng.below(i as u64 + 1) as usize);
    }
    let (dqk, dv, h) = (
        HEAD_DIM_QK as usize,
        HEAD_DIM_VO as usize,
        kv_heads as usize,
    );
    let mut blocks: Vec<Vec<u16>> = vec![vec![0u16; geom.block_elems()]; total_blocks as usize];
    let mut requests = Vec::new();
    // Per row: per position, K [h][dqk] and V [h][dv] as f32 (BF16 values).
    let mut kv: Vec<Vec<(Vec<f32>, Vec<f32>)>> = Vec::new();
    let mut q_rows: Vec<Vec<f32>> = Vec::new();
    let mut q_start = 0;
    for r in rows {
        let end = r.context + r.queries;
        let pages: Vec<u32> = (0..end.div_ceil(BS)).map(|_| free.pop().unwrap()).collect();
        let mut row_kv = Vec::new();
        for pos in 0..end {
            let k: Vec<f32> = (0..h * dqk)
                .map(|_| bf16::to_f32(bf16::from_f32(rng.f32() * 2.0)))
                .collect();
            let v: Vec<f32> = (0..h * dv)
                .map(|_| bf16::to_f32(bf16::from_f32(rng.f32())))
                .collect();
            let b = &mut blocks[pages[(pos / BS) as usize] as usize];
            let off = (pos % BS) as usize;
            let (ko, vo) = (geom.k_offset(layer), geom.v_offset(layer));
            for (i, &x) in k.iter().enumerate() {
                b[ko + off * h * dqk + i] = bf16::from_f32(x);
            }
            for (i, &x) in v.iter().enumerate() {
                b[vo + off * h * dv + i] = bf16::from_f32(x);
            }
            row_kv.push((k, v));
        }
        // Leave out leading pages no query can see.
        let first_visible = match window {
            Some(w) => (r.context + 1).saturating_sub(w),
            None => 0,
        };
        let first_page = first_visible / BS;
        requests.push(AttnRequest {
            q_start,
            qo_len: r.queries,
            pages: pages[first_page as usize..].to_vec(),
            kv_len: end - first_page * BS,
        });
        for _ in 0..r.queries {
            q_rows.push(
                (0..NQ as usize * dqk)
                    .map(|_| bf16::to_f32(bf16::from_f32(rng.f32() * 2.0)))
                    .collect(),
            );
        }
        q_start += r.queries;
        kv.push(row_kv);
    }
    for (b, data) in blocks.iter().enumerate().skip(1) {
        store.write_block(gpu, 0, b as u32, data).unwrap();
    }
    let sinks: Vec<f32> = match window {
        Some(_) => (0..NQ).map(|_| 0.5 + rng.f32().abs()).collect(),
        None => vec![f32::NEG_INFINITY; NQ as usize],
    };
    let dsink = s.clone_htod(&sinks).unwrap();
    let q_flat: Vec<u16> = q_rows
        .iter()
        .flatten()
        .map(|&x| bf16::from_f32(x))
        .collect();
    let dq = s.clone_htod(&q_flat).unwrap();
    let spec = AttentionSpec {
        kind: match window {
            Some(w) => AttentionKind::Sliding { window: w as usize },
            None => AttentionKind::Global,
        },
        num_q_heads: NQ as usize,
        num_kv_heads: h,
        head_dim_qk: dqk,
        head_dim_v: dv,
        rope_dim: 64,
        rope_theta: 1e4,
        has_sinks: window.is_some(),
        softmax_scale: 1.0 / (dqk as f32).sqrt(),
    };
    // Reference outputs.
    let mut want = Vec::new();
    for (ri, r) in rows.iter().enumerate() {
        for i in 0..r.queries {
            let pos = r.context + i;
            let lo = match window {
                Some(w) => (pos + 1).saturating_sub(w),
                None => 0,
            };
            let mut out = vec![0f32; NQ as usize * dv];
            let qr = &q_rows[(requests[ri].q_start + i) as usize];
            for head in 0..NQ as usize {
                let g = head / (NQ as usize / h);
                let keys: Vec<&[f32]> = (lo..=pos)
                    .map(|p| &kv[ri][p as usize].0[g * dqk..(g + 1) * dqk])
                    .collect();
                let values: Vec<&[f32]> = (lo..=pos)
                    .map(|p| &kv[ri][p as usize].1[g * dv..(g + 1) * dv])
                    .collect();
                attend(
                    &spec,
                    &qr[head * dqk..(head + 1) * dqk],
                    &keys,
                    &values,
                    window.map(|_| sinks[head]),
                    &mut out[head * dv..(head + 1) * dv],
                );
            }
            want.extend(out);
        }
    }
    let base = dptr(store.pool(0), s);
    let layer_desc = AttnLayer {
        k_base: base + 2 * geom.k_offset(layer) as u64,
        v_base: base + 2 * geom.v_offset(layer) as u64,
        block_elems: geom.block_elems() as u32,
        num_kv_heads: kv_heads,
        page_size: BS,
        window_left: window.map_or(-1, |w| w as i32 - 1),
        sink: dptr(&dsink, s),
    };
    for &arch in &su.archs {
        let attn = Attention::from_module(su.module("flashinfer_fa2_sink_paged", arch)).unwrap();
        let plan = attn.plan(gpu, &requests, NQ / kv_heads, BS).unwrap();
        let o = s
            .alloc_zeros::<u16>(q_rows.len() * NQ as usize * dv)
            .unwrap();
        unsafe { attn.run(gpu, &plan, &layer_desc, NQ, dptr(&dq, s), dptr(&o, s)) }.unwrap();
        let got = s.clone_dtoh(&o).unwrap();
        let mut worst = 0f32;
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let g = bf16::to_f32(g);
            let err = (g - w).abs();
            worst = worst.max(err);
            // BF16 output and BF16 probabilities (FA2 rounds P to BF16 for
            // the PV product); values are in [-1, 1].
            assert!(
                err <= 4e-3 + w.abs() / 128.0,
                "{arch:?} element {i}: {g} vs {w}"
            );
        }
        eprintln!(
            "{arch:?} window {window:?}, {} rows, {} queries: worst absolute error {worst:.2e}",
            rows.len(),
            q_rows.len()
        );
    }
}

#[test]
fn global_prefill_and_decode() {
    run(
        None,
        4,
        &[
            Row {
                context: 0,
                queries: 37,
            },
            Row {
                context: 300,
                queries: 1,
            },
            Row {
                context: 50,
                queries: 130,
            },
            Row {
                context: 17,
                queries: 1,
            },
        ],
        5,
    );
}

#[test]
fn sliding_prefill_and_decode() {
    run(
        Some(128),
        8,
        &[
            Row {
                context: 0,
                queries: 200,
            },
            Row {
                context: 1000,
                queries: 1,
            },
            Row {
                context: 130,
                queries: 64,
            },
            Row {
                context: 5,
                queries: 3,
            },
            Row {
                context: 127,
                queries: 1,
            },
        ],
        6,
    );
}

#[test]
fn decode_batch() {
    let rows: Vec<Row> = (0..48)
        .map(|i| Row {
            context: 1 + i * 37,
            queries: 1,
        })
        .collect();
    run(Some(128), 8, &rows, 7);
    run(None, 4, &rows, 8);
}
