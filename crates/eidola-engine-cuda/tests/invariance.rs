//! Batch invariance: a query row's output is the same bits whatever else its
//! step holds and however its sequence was chunked.
//!
//! - **Attention** (`attention_rows_are_batch_invariant`; synthetic KV at
//!   Flash's shapes, no checkpoint): four probe rows at positions `p ..= p +
//!   3`, for global layers (4 KV heads, split at 1,024 keys and merged) and
//!   sliding ones (8 KV heads, window 128, sinks, anchored), at contexts
//!   where the window first drops key 0 (126), around and past the first
//!   chunk boundary (1,000, 1,022, 1,037), before a 64-key boundary (1,084)
//!   and long ones (2,047, 4,101, 16,389). Each probe alone in its own launch is the baseline;
//!   every other scenario must give its rows bit for bit:
//!   - `+prefill`: the four as decode rows beside another sequence's
//!     300-query chunk;
//!   - `verify`: one request of four queries (a drafted decode row, `k = 3`);
//!   - `chunk300` / `chunk2048`: inside a prefill chunk of 300 / 2,048
//!     queries starting 150 / 1,000 positions before `p` (or at 0);
//!   - `chunk300@-13`: inside a 300-query chunk starting 13 before `p`, so
//!     at `p` = 126 and 1,084 the probes' query tiles cross a 64-key boundary
//!     (a row's walk must still end at its own last KV tile);
//!   - `chunk2048/passes`: the same chunk planned with a partial-row scratch
//!     of one row's chunks plus one, so its global work list runs in many
//!     passes.
//!
//!   Every row is also checked against the f32 reference `attend` (within
//!   the attention test's tolerance); the sinks are large enough that a
//!   dropped or doubled sink falls outside it. Prints one line per kind and
//!   context.
//! - **The executor** (`executor_logits_are_batch_invariant`; the truncated
//!   checkpoint, `EIDOLA_MIMO_DIR`): the logits of four positions past 1,037
//!   and past 2,100, computed as decode steps after a prefill in chunks of
//!   64, 63 and 65, as one verify-shaped chunk after chunks of 300, as decode rows each
//!   beside another sequence's 300-token chunk, inside a 2,048-token chunk,
//!   inside 300-token chunks, and after a prefix shared from another
//!   sequence's blocks (a prefix-cache hit): the same bits in every one.
//!
//! The scenarios' host side (requests, plans, budgets) is built without a
//! device too (`scenarios_plan_without_a_device`).

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use common::{Lcg, setup};
use cudarc::driver::CudaSlice;
use eidola_engine::executor::{Executor, Maintenance, SeqEntry, StepInput, TableUpdate};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::Bucket;
use eidola_engine_cuda::attention::{
    Attention, AttnLayer, AttnRequest, HEAD_DIM_QK, HEAD_DIM_VO, HostPlan, MASKED_PAGE, Partials,
    Reduction, SPLIT_KEYS,
};
use eidola_engine_cuda::bf16;
use eidola_engine_cuda::launch::dptr;
use eidola_engine_cuda::{CudaExecutor, CudaExecutorConfig, CudaGraphs, Gpu, KvBlocks, MtpHidden};
use eidola_engine_model::attention::attend;
use eidola_engine_model::config::{AttentionKind, AttentionSpec};
use eidola_engine_model::safetensors::WeightSet;

const NQ: u32 = 64;
const BS: u32 = 16;
const PROBES: u32 = 4;
/// Probe contexts: where a window of 128 first drops key 0 (126 to 129
/// probed), around and past the first 1,024-key chunk boundary, just before
/// a 64-key boundary that is not a chunk boundary (1,084), and long ones.
const CONTEXTS: [u32; 8] = [126, 1000, 1022, 1037, 1084, 2047, 4101, 16389];

fn usize_of(x: u32) -> usize {
    usize::try_from(x).unwrap()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Global,
    Sliding,
}

impl Kind {
    fn kv_heads(self) -> u32 {
        match self {
            Kind::Global => 4,
            Kind::Sliding => 8,
        }
    }
    fn group(self) -> u32 {
        NQ / self.kv_heads()
    }
    fn window(self) -> Option<u32> {
        match self {
            Kind::Global => None,
            Kind::Sliding => Some(128),
        }
    }
    fn reduction(self) -> Reduction {
        match self {
            Kind::Global => Reduction::Split,
            Kind::Sliding => Reduction::Anchored { origin: 0 },
        }
    }
    fn first_visible(self, q: u32) -> u32 {
        self.window().map_or(0, |w| (q + 1).saturating_sub(w))
    }
}

/// A step's sequences: (first query position, queries).
type Queries = Vec<(u32, u32)>;

/// How much partial-row scratch a scenario plans with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Budget {
    /// Enough for one pass of any scenario.
    OnePass,
    /// One row's chunks plus one: a 2,048-row chunk runs in many passes.
    Small,
}

/// Every scenario for probes at `p`: its name, the step's sequences, the
/// step row of the first probe, and its partial-row budget.
fn scenarios(p: u32) -> Vec<(&'static str, Queries, u32, Budget)> {
    let mut beside: Queries = (0..PROBES).map(|i| (p + i, 1)).collect();
    beside.push((p.saturating_sub(700), 300));
    let start = p.saturating_sub(1000);
    let (s150, s13) = (p.saturating_sub(150), p.saturating_sub(13));
    vec![
        ("+prefill", beside, 0, Budget::OnePass),
        ("verify", vec![(p, PROBES)], 0, Budget::OnePass),
        ("chunk300", vec![(s150, 300)], p - s150, Budget::OnePass),
        // Query tiles of 8 or 16 rows from p - 13 put probe rows in tiles
        // that cross a 64-key boundary at 1,088 (p = 1,084) or 128 (p = 126).
        ("chunk300@-13", vec![(s13, 300)], p - s13, Budget::OnePass),
        ("chunk2048", vec![(start, 2048)], p - start, Budget::OnePass),
        (
            "chunk2048/passes",
            vec![(start, 2048)],
            p - start,
            Budget::Small,
        ),
    ]
}

/// Positions the pool holds: every scenario's keys.
fn pool_positions() -> u32 {
    CONTEXTS
        .iter()
        .flat_map(|&p| scenarios(p))
        .flat_map(|(_, q, _, _)| q.into_iter().map(|(s, n)| s + n))
        .max()
        .unwrap()
}

/// The partial rows a budget allows.
fn budget_rows(b: Budget) -> u32 {
    let most_chunks = pool_positions().div_ceil(SPLIT_KEYS);
    match b {
        Budget::OnePass => 2048 * most_chunks,
        Budget::Small => most_chunks + 1,
    }
}

/// Logical page `l` lives at physical page `perm[l]`, a fixed shuffle of
/// `1 ..= pages`; physical page 0 is the null block (zeros, the masked page).
fn page_perm(pages: u32, seed: u64) -> Vec<u32> {
    let mut rng = Lcg(seed);
    let mut perm: Vec<u32> = (1..=pages).collect();
    for i in (1..perm.len()).rev() {
        let j = usize::try_from(rng.below(i as u64 + 1)).unwrap();
        perm.swap(i, j);
    }
    perm
}

/// The request for queries `start .. start + len` as the executor lists it:
/// pages from the reduction's anchor, those wholly before the first query's
/// window masked.
fn request(kind: Kind, perm: &[u32], q_start: u32, start: u32, len: u32) -> AttnRequest {
    let first = kind.first_visible(start);
    let kv_start = kind.reduction().kv_start(first, BS);
    let end = start + len;
    AttnRequest {
        q_start,
        qo_len: len,
        pages: (kv_start / BS..first / BS)
            .map(|_| MASKED_PAGE)
            .chain((first / BS..end.div_ceil(BS)).map(|l| perm[usize_of(l)]))
            .collect(),
        kv_start,
        kv_len: end - kv_start,
    }
}

fn requests(kind: Kind, perm: &[u32], queries: &[(u32, u32)]) -> Vec<AttnRequest> {
    let mut q_start = 0;
    queries
        .iter()
        .map(|&(start, len)| {
            let r = request(kind, perm, q_start, start, len);
            q_start += len;
            r
        })
        .collect()
}

/// The host side of every scenario, without a device: each plans (the
/// executor's checks), in one pass with the one-pass budget and in several
/// with the small one where its chunks need them, and the probes land in
/// the scenario's rows at their positions.
#[test]
fn scenarios_plan_without_a_device() {
    let positions = pool_positions();
    assert_eq!(positions, 16389 - 1000 + 2048);
    let perm = page_perm(positions.div_ceil(BS), 1);
    for kind in [Kind::Global, Kind::Sliding] {
        for &p in &CONTEXTS {
            for i in 0..PROBES {
                let r = requests(kind, &perm, &[(p + i, 1)]);
                HostPlan::new(
                    &r,
                    kind.group(),
                    BS,
                    kind.reduction(),
                    budget_rows(Budget::Small),
                )
                .unwrap();
            }
            for (name, queries, first, budget) in scenarios(p) {
                // The probes' rows hold positions p ..= p + 3.
                let mut pos = Vec::new();
                for &(s, n) in &queries {
                    pos.extend(s..s + n);
                }
                assert_eq!(
                    &pos[usize_of(first)..usize_of(first + PROBES)],
                    &(p..p + PROBES).collect::<Vec<_>>()[..],
                    "{name} at {p}"
                );
                let r = requests(kind, &perm, &queries);
                let plan =
                    HostPlan::new(&r, kind.group(), BS, kind.reduction(), budget_rows(budget))
                        .unwrap();
                let passes = plan.passes().len();
                match (kind, budget, name) {
                    (Kind::Global, Budget::Small, _) => assert!(passes > 1, "{name} at {p}"),
                    _ => assert_eq!(passes, 1, "{kind:?} {name} at {p}"),
                }
            }
        }
    }
}

/// A position's query row, `[64 heads][192]`, the same wherever it is used.
fn q_row(pos: u32) -> Vec<u16> {
    let mut rng = Lcg(u64::from(pos).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x51);
    (0..usize_of(NQ * HEAD_DIM_QK))
        .map(|_| bf16::from_f32(rng.f32() * 2.0))
        .collect()
}

/// Sink logits of the sliding layers: large enough that leaving them out (or
/// adding one twice) moves a row past the reference tolerance.
const SINK_LO: f32 = 4.0;

/// One layer's KV for positions `0..positions`, paged through shuffled
/// physical pages (page 0 left zero), with host copies for the reference.
struct Pool {
    kind: Kind,
    perm: Vec<u32>,
    k_host: Vec<u16>,
    v_host: Vec<u16>,
    sinks: Vec<f32>,
    k: CudaSlice<u16>,
    v: CudaSlice<u16>,
    sink: CudaSlice<f32>,
}

impl Pool {
    fn new(gpu: &Gpu, kind: Kind, positions: u32, seed: u64) -> Pool {
        Pool::with_key(gpu, kind, positions, seed, None)
    }

    /// The pool, with position `loud` (if any) holding a key and value far
    /// larger than any other: a row that sees it moves a long way.
    fn with_key(gpu: &Gpu, kind: Kind, positions: u32, seed: u64, loud: Option<u32>) -> Pool {
        let pages = positions.div_ceil(BS);
        let perm = page_perm(pages, seed);
        let mut rng = Lcg(seed ^ 0xA5A5);
        let h = kind.kv_heads();
        let (kp, vp) = (usize_of(h * HEAD_DIM_QK), usize_of(h * HEAD_DIM_VO));
        let total = usize_of(pages * BS);
        let k_host: Vec<u16> = (0..total * kp)
            .map(|_| bf16::from_f32(rng.f32() * 2.0))
            .collect();
        let v_host: Vec<u16> = (0..total * vp).map(|_| bf16::from_f32(rng.f32())).collect();
        let sinks: Vec<f32> = match kind.window() {
            Some(_) => (0..NQ).map(|_| SINK_LO + 2.0 * rng.f32().abs()).collect(),
            None => vec![f32::NEG_INFINITY; usize_of(NQ)],
        };
        let mut k_host = k_host;
        let mut v_host = v_host;
        if let Some(x) = loud {
            let x = usize_of(x);
            k_host[x * kp..(x + 1) * kp].fill(bf16::from_f32(8.0));
            v_host[x * vp..(x + 1) * vp].fill(bf16::from_f32(50.0));
        }
        // Physical page `perm[l]` holds logical page `l`; page 0 stays zero.
        let n = usize_of(BS);
        let mut k_dev = vec![0u16; (total + n) * kp];
        let mut v_dev = vec![0u16; (total + n) * vp];
        for (l, &phys) in perm.iter().enumerate() {
            let (src, dst) = (l * n, usize_of(phys * BS));
            k_dev[dst * kp..(dst + n) * kp].copy_from_slice(&k_host[src * kp..(src + n) * kp]);
            v_dev[dst * vp..(dst + n) * vp].copy_from_slice(&v_host[src * vp..(src + n) * vp]);
        }
        let s = gpu.stream();
        Pool {
            kind,
            k: s.clone_htod(&k_dev).unwrap(),
            v: s.clone_htod(&v_dev).unwrap(),
            sink: s.clone_htod(&sinks).unwrap(),
            perm,
            k_host,
            v_host,
            sinks,
        }
    }

    fn layer(&self, gpu: &Gpu) -> AttnLayer {
        let s = gpu.stream();
        let h = self.kind.kv_heads();
        AttnLayer {
            k_base: dptr(&self.k, s),
            v_base: dptr(&self.v, s),
            k_page_stride: BS * h * HEAD_DIM_QK,
            v_page_stride: BS * h * HEAD_DIM_VO,
            num_kv_heads: h,
            page_size: BS,
            window_left: self
                .kind
                .window()
                .map_or(-1, |w| i32::try_from(w).unwrap() - 1),
            sink: dptr(&self.sink, s),
        }
    }

    /// The f32 reference output of the query at `pos` (`[64][128]`).
    fn reference(&self, pos: u32) -> Vec<f32> {
        let h = usize_of(self.kind.kv_heads());
        let (dqk, dv) = (usize_of(HEAD_DIM_QK), usize_of(HEAD_DIM_VO));
        let spec = AttentionSpec {
            kind: match self.kind.window() {
                Some(w) => AttentionKind::Sliding {
                    window: usize_of(w),
                },
                None => AttentionKind::Global,
            },
            num_q_heads: usize_of(NQ),
            num_kv_heads: h,
            head_dim_qk: dqk,
            head_dim_v: dv,
            rope_dim: 64,
            rope_theta: 1e4,
            has_sinks: self.kind.window().is_some(),
            softmax_scale: 1.0 / (dqk as f32).sqrt(),
        };
        let q: Vec<f32> = q_row(pos).iter().map(|&x| bf16::to_f32(x)).collect();
        let (lo, hi) = (usize_of(self.kind.first_visible(pos)), usize_of(pos) + 1);
        let mut out = vec![0f32; usize_of(NQ) * dv];
        let per_group = usize_of(NQ) / h;
        for g in 0..h {
            let k: Vec<Vec<f32>> = (lo..hi)
                .map(|p| {
                    let o = p * h * dqk + g * dqk;
                    self.k_host[o..o + dqk]
                        .iter()
                        .map(|&x| bf16::to_f32(x))
                        .collect()
                })
                .collect();
            let v: Vec<Vec<f32>> = (lo..hi)
                .map(|p| {
                    let o = p * h * dv + g * dv;
                    self.v_host[o..o + dv]
                        .iter()
                        .map(|&x| bf16::to_f32(x))
                        .collect()
                })
                .collect();
            let keys: Vec<&[f32]> = k.iter().map(Vec::as_slice).collect();
            let values: Vec<&[f32]> = v.iter().map(Vec::as_slice).collect();
            for head in g * per_group..(g + 1) * per_group {
                attend(
                    &spec,
                    &q[head * dqk..(head + 1) * dqk],
                    &keys,
                    &values,
                    self.kind.window().map(|_| self.sinks[head]),
                    &mut out[head * dv..(head + 1) * dv],
                );
            }
        }
        out
    }
}

/// One attention call over `queries`, planned as the executor plans it with
/// `budget`, and its output rows `first .. first + n`.
#[allow(clippy::too_many_arguments)]
fn attend_rows(
    gpu: &Gpu,
    attn: &Attention,
    pool: &Pool,
    partials: &Partials,
    queries: &[(u32, u32)],
    budget: Budget,
    first: u32,
    n: u32,
) -> Vec<u16> {
    let s = gpu.stream();
    let kind = pool.kind;
    let reqs = requests(kind, &pool.perm, queries);
    let rows: u32 = queries.iter().map(|&(_, l)| l).sum();
    let q_host: Vec<u16> = queries
        .iter()
        .flat_map(|&(st, l)| (st..st + l).flat_map(q_row))
        .collect();
    let q = s.clone_htod(&q_host).unwrap();
    let o = s
        .alloc_zeros::<u16>(usize_of(rows * NQ * HEAD_DIM_VO))
        .unwrap();
    let plan = attn
        .plan(
            gpu,
            &reqs,
            kind.group(),
            BS,
            kind.reduction(),
            budget_rows(budget),
        )
        .unwrap();
    // SAFETY: Q and O hold the plan's rows; the pool holds every page the
    // requests list; the partials hold the budget's rows.
    unsafe {
        attn.run(
            gpu,
            &plan,
            &pool.layer(gpu),
            NQ,
            dptr(&q, s),
            dptr(&o, s),
            partials,
        )
    }
    .unwrap();
    let all = s.clone_dtoh(&o).unwrap();
    let w = usize_of(NQ * HEAD_DIM_VO);
    all[usize_of(first) * w..usize_of(first + n) * w].to_vec()
}

/// The count of differing BF16 elements and the largest difference.
fn diff(a: &[u16], b: &[u16]) -> (usize, f32) {
    let mut n = 0;
    let mut worst = 0f32;
    for (&x, &y) in a.iter().zip(b) {
        if x != y {
            n += 1;
            worst = worst.max((bf16::to_f32(x) - bf16::to_f32(y)).abs());
        }
    }
    (n, worst)
}

/// Rows against the reference: the worst error, and the elements outside
/// the attention test's tolerance (BF16 output and probabilities, values in
/// [-1, 1]).
fn against(got: &[u16], want: &[f32]) -> (f32, usize) {
    let (mut worst, mut outside) = (0f32, 0usize);
    for (&g, &w) in got.iter().zip(want) {
        let err = (bf16::to_f32(g) - w).abs();
        worst = worst.max(err);
        outside += usize::from(err > 4e-3 + w.abs() / 128.0);
    }
    (worst, outside)
}

#[test]
fn attention_rows_are_batch_invariant() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    let positions = pool_positions();
    let most = budget_rows(Budget::OnePass);
    let tv = s
        .alloc_zeros::<u16>(usize_of(most * NQ * HEAD_DIM_VO))
        .unwrap();
    let ts = s.alloc_zeros::<f32>(usize_of(most * NQ)).unwrap();
    let partials = Partials {
        v: dptr(&tv, s),
        s: dptr(&ts, s),
        rows: most,
        heads: NQ,
    };
    let mut failures = Vec::new();
    for (k, kind) in [Kind::Global, Kind::Sliding].into_iter().enumerate() {
        let pool = Pool::new(gpu, kind, positions, 11 + k as u64);
        for &p in &CONTEXTS {
            let want: Vec<f32> = (0..PROBES).flat_map(|i| pool.reference(p + i)).collect();
            for &arch in &su.archs {
                let attn =
                    Attention::from_module(su.module("flashinfer_fa2_sink_paged", arch)).unwrap();
                let rows = |queries: &[(u32, u32)], budget, first, n| {
                    attend_rows(gpu, &attn, &pool, &partials, queries, budget, first, n)
                };
                // Each probe alone: the baseline.
                let decode: Vec<u16> = (0..PROBES)
                    .flat_map(|i| rows(&[(p + i, 1)], Budget::OnePass, 0, 1))
                    .collect();
                let (worst, outside) = against(&decode, &want);
                if outside > 0 {
                    failures.push(format!(
                        "{kind:?} {arch:?} p={p} decode: {outside} elements outside the \
                         reference tolerance (worst {worst:.1e})"
                    ));
                }
                let mut cells = Vec::new();
                for (name, queries, first, budget) in scenarios(p) {
                    let got = rows(&queries, budget, first, PROBES);
                    let (n, worst) = diff(&decode, &got);
                    cells.push(if n == 0 {
                        format!("{name} =")
                    } else {
                        format!("{name} ≠{n} ({worst:.1e})")
                    });
                    if n != 0 {
                        failures.push(format!(
                            "{kind:?} {arch:?} p={p} {name}: {n} elements differ from the rows \
                             alone (largest {worst:.1e})"
                        ));
                    }
                }
                println!(
                    "{kind:?} {arch:?} p={p}: reference worst {worst:.1e}; {}",
                    cells.join(", ")
                );
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The window's edge is exact on the device: a row alone (and inside a
/// chunk) gives the same bits whatever the key just outside its window
/// holds, and moves when the key just inside it does, at rows where a
/// window of 128 first drops key 0 and past it.
#[test]
fn the_window_edge_is_exact() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    let rows = budget_rows(Budget::Small);
    let tv = s
        .alloc_zeros::<u16>(usize_of(rows * NQ * HEAD_DIM_VO))
        .unwrap();
    let ts = s.alloc_zeros::<f32>(usize_of(rows * NQ)).unwrap();
    let partials = Partials {
        v: dptr(&tv, s),
        s: dptr(&ts, s),
        rows,
        heads: NQ,
    };
    let kind = Kind::Sliding;
    let w = kind.window().unwrap();
    let positions = 1200;
    let base = Pool::new(gpu, kind, positions, 21);
    let mut failures = Vec::new();
    for &arch in &su.archs {
        let attn = Attention::from_module(su.module("flashinfer_fa2_sink_paged", arch)).unwrap();
        for p in [127u32, 128, 129, 130, 200, 1037, 1087] {
            // Alone, and as the 14th row of a 100-query chunk.
            let shapes: [(&[(u32, u32)], u32); 2] =
                [(&[(p, 1)], 0), (&[(p.saturating_sub(13), 100)], p.min(13))];
            for (queries, first) in shapes {
                let run = |pool: &Pool| {
                    attend_rows(
                        gpu,
                        &attn,
                        pool,
                        &partials,
                        queries,
                        Budget::Small,
                        first,
                        1,
                    )
                };
                let want = run(&base);
                if let Some(outside) = (p + 1).checked_sub(w + 1) {
                    let pool = Pool::with_key(gpu, kind, positions, 21, Some(outside));
                    if run(&pool) != want {
                        failures.push(format!("{arch:?} p={p} {queries:?}: key {outside} outside the window reached the row"));
                    }
                }
                let inside = (p + 1).saturating_sub(w);
                let pool = Pool::with_key(gpu, kind, positions, 21, Some(inside));
                if run(&pool) == want {
                    failures.push(format!("{arch:?} p={p} {queries:?}: key {inside} inside the window did not reach the row"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

// ---------------------------------------------------------------------------
// The executor on the truncated checkpoint.

const KEEP: [usize; 4] = [0, 1, 2, 5];
const SAMPLEABLE: u32 = 151_675;
/// Blocks each scenario's sequence owns in both groups.
const SEQ_BLOCKS: u32 = 200;

fn token(i: u32) -> u32 {
    (i * 7919 + 13) % 150_000
}

/// The tokens of a sequence, positions `range`; the other sequence's
/// differ.
fn tokens(range: std::ops::Range<u32>, other: bool) -> Vec<u32> {
    range
        .map(|i| token(i + if other { 50_000 } else { 0 }))
        .collect()
}

/// `slot`'s table: logical block `i` at `base + i` for `i` in `blocks`, in
/// both groups.
fn map(slot: u32, base: u32, blocks: std::ops::Range<u32>) -> Vec<TableUpdate> {
    (0..2u32)
        .flat_map(|group| {
            blocks.clone().map(move |index| TableUpdate {
                slot,
                group,
                index,
                block: base + index,
            })
        })
        .collect()
}

/// One step over `parts` (`(slot, context, tokens)` each), every row
/// sampled; the logits of every host token, in order.
fn step(
    ex: &mut CudaExecutor,
    parts: &[(u32, u32, &[u32])],
    maintenance: Vec<Maintenance>,
    table_updates: Vec<TableUpdate>,
) -> Vec<f32> {
    let mut seqs = Vec::new();
    let mut token_ids = Vec::new();
    let mut positions = Vec::new();
    for &(slot, context, toks) in parts {
        let n = u32::try_from(toks.len()).unwrap();
        seqs.push(SeqEntry {
            slot,
            token_start: u32::try_from(token_ids.len()).unwrap(),
            num_tokens: n,
            context_len: context,
            num_drafts: 0,
            sample: true,
            sampling: SamplingParams::greedy(),
        });
        token_ids.extend_from_slice(toks);
        positions.extend(context..context + n);
    }
    ex.execute(&StepInput {
        bucket: Bucket {
            max_seqs: 8,
            max_tokens: 2048,
        },
        maintenance,
        table_updates,
        seqs,
        token_ids,
        positions,
        return_logits: false,
    })
    .unwrap();
    ex.take_all_logits().unwrap()
}

/// Prefill positions `from .. to` of `slot` in chunks of `chunk`; the first
/// step resets the slot and maps `blocks` at `base`.
#[allow(clippy::too_many_arguments)]
fn prefill(
    ex: &mut CudaExecutor,
    slot: u32,
    base: u32,
    blocks: std::ops::Range<u32>,
    from: u32,
    to: u32,
    chunk: u32,
    other: bool,
) -> Vec<f32> {
    let mut logits = Vec::new();
    let mut updates = map(slot, base, blocks);
    let mut reset = vec![Maintenance::ResetSlot { slot }];
    let mut c = from;
    while c < to {
        let n = chunk.min(to - c);
        let toks = tokens(c..c + n, other);
        logits.extend(step(
            ex,
            &[(slot, c, &toks)],
            std::mem::take(&mut reset),
            std::mem::take(&mut updates),
        ));
        c += n;
    }
    logits
}

#[test]
fn executor_logits_are_batch_invariant() {
    let Some(dir) = std::env::var_os("EIDOLA_MIMO_DIR").map(PathBuf::from) else {
        eprintln!("skipping: EIDOLA_MIMO_DIR not set");
        return;
    };
    let Some(su) = setup() else { return };
    let store = Arc::new(WeightSet::open_dir(&dir).unwrap());
    let blocks = 1 + 8 * SEQ_BLOCKS;
    let cfg = CudaExecutorConfig {
        block_size: BS,
        num_blocks: KvBlocks {
            global: blocks,
            sliding: blocks,
            drafter: 0,
        },
        num_state_slots: 8,
        max_model_len: 4096,
        buckets: vec![Bucket {
            max_seqs: 8,
            max_tokens: 2048,
        }],
        sampleable_vocab_size: SAMPLEABLE,
        image: None,
        graphs: CudaGraphs::Off,
        draft_tokens: 0,
        mtp_hidden: MtpHidden::Normed,
    };
    let mut ex = CudaExecutor::new(su.gpu, &su.dir, store, Some(&KEEP), cfg).unwrap();
    ex.record_all_logits = true;
    let vocab = 152_576usize;
    let all = |b: u32| 0..b;
    for p in [1037u32, 2100] {
        let end = p + PROBES;
        let probes = tokens(p..end, false);
        let row = |logits: &[f32], i: usize| logits[i * vocab..(i + 1) * vocab].to_vec();
        let mut runs: Vec<(&str, Vec<f32>)> = Vec::new();
        let base = |slot: u32| 1 + slot * SEQ_BLOCKS;
        let need = (p + 1100).div_ceil(BS);

        // Prefill in chunks of 64, then four decode steps.
        prefill(&mut ex, 0, base(0), all(need), 0, p, 64, false);
        let mut out = Vec::new();
        for i in 0..PROBES {
            let l = step(
                &mut ex,
                &[(0, p + i, &probes[usize_of(i)..usize_of(i) + 1])],
                vec![],
                vec![],
            );
            out.extend(row(&l, 0));
        }
        runs.push(("decode after chunks of 64", out));

        // Chunks of 300, then the four as one verify-shaped chunk.
        prefill(&mut ex, 1, base(1), all(need), 0, p, 300, false);
        let l = step(&mut ex, &[(1, p, &probes)], vec![], vec![]);
        runs.push(("a 4-token chunk after chunks of 300", l));

        // Chunks of 256, then each probe beside another sequence's
        // 300-token chunk (slot 7).
        prefill(&mut ex, 2, base(2), all(need), 0, p, 256, false);
        let mut updates = map(7, base(7), all(need));
        let mut reset = vec![Maintenance::ResetSlot { slot: 7 }];
        let mut out = Vec::new();
        for i in 0..PROBES {
            let other = tokens(300 * i..300 * (i + 1), true);
            let l = step(
                &mut ex,
                &[
                    (2, p + i, &probes[usize_of(i)..usize_of(i) + 1]),
                    (7, 300 * i, &other),
                ],
                std::mem::take(&mut reset),
                std::mem::take(&mut updates),
            );
            out.extend(row(&l, 0));
        }
        runs.push(("decode beside a 300-token chunk", out));

        // The probes inside one 2,048-token chunk from p - 1,000.
        let start = p - 1000;
        prefill(&mut ex, 3, base(3), all(need), 0, start, 2048, false);
        let chunk = tokens(start..start + 2048, false);
        let l = step(&mut ex, &[(3, start, &chunk)], vec![], vec![]);
        let at = usize_of(p - start) * vocab;
        runs.push((
            "inside a 2,048-token chunk",
            l[at..at + usize_of(PROBES) * vocab].to_vec(),
        ));

        // Chunks of 63 and 65 (every chunk after the first starting off a
        // 16- and 64-key boundary), then four decode steps (slot 6).
        for (chunk, name) in [
            (63u32, "decode after chunks of 63"),
            (65, "decode after chunks of 65"),
        ] {
            prefill(&mut ex, 6, base(6), all(need), 0, p, chunk, false);
            let mut out = Vec::new();
            for i in 0..PROBES {
                let l = step(
                    &mut ex,
                    &[(6, p + i, &probes[usize_of(i)..usize_of(i) + 1])],
                    vec![],
                    vec![],
                );
                out.extend(row(&l, 0));
            }
            runs.push((name, out));
        }

        // The probes inside chunks of 300 from 0.
        let l = prefill(
            &mut ex,
            4,
            base(4),
            all(need),
            0,
            end.div_ceil(300) * 300,
            300,
            false,
        );
        let at = usize_of(p) * vocab;
        runs.push((
            "inside chunks of 300",
            l[at..at + usize_of(PROBES) * vocab].to_vec(),
        ));

        // A prefix-cache hit: slot 5 reads slot 0's full blocks below p's
        // block boundary, computes the rest itself.
        let shared = p / BS;
        let mut updates: Vec<TableUpdate> = map(5, base(0), all(shared));
        updates.extend(map(5, base(5), shared..need));
        let rest = tokens(shared * BS..p, false);
        step(
            &mut ex,
            &[(5, shared * BS, &rest)],
            vec![Maintenance::ResetSlot { slot: 5 }],
            updates,
        );
        let l = step(&mut ex, &[(5, p, &probes)], vec![], vec![]);
        runs.push(("after a shared prefix", l));

        let (name0, base_logits) = &runs[0];
        for (name, logits) in &runs[1..] {
            let differ = logits
                .iter()
                .zip(base_logits)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            println!("p={p}: {name} against {name0}: {differ} logits differ");
            assert_eq!(differ, 0, "p={p}: {name} against {name0}");
        }
        // The slots' blocks are reused at the next context (slot 5 shares
        // slot 0's, which slot 0 writes again).
        for slot in 0..8 {
            step(&mut ex, &[], vec![Maintenance::ResetSlot { slot }], vec![]);
        }
    }
}
