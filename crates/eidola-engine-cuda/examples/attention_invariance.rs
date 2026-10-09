//! Whether a query row's attention output depends on what else runs in its
//! step, under the executor's attention and under candidate kernel shapes,
//! and what each shape costs. Synthetic KV and queries at Flash's shapes; no
//! checkpoint needed.
//!
//! ```text
//! attention_invariance <kernels_dir> [--matrix] [--timing] [--iters N]
//!                      [--contexts 1037,4101,16389] [--splits 512,1024,2048]
//! ```
//!
//! With neither `--matrix` nor `--timing`, both run.
//!
//! **Matrix.** For each layer kind (global: 4 KV heads, no window, no sinks;
//! sliding: 8 KV heads, window 128, sinks) and each context `p`, four probe
//! rows at positions `p ..= p + 3` are computed five ways in one structure:
//!
//! - `decode`: each probe alone, one launch per row (the baseline);
//! - `+prefill`: the four as decode rows beside another sequence's 300-token
//!   prefill chunk in the same launch;
//! - `verify`: one request of four queries (a drafted decode row, `k = 3`);
//! - `chunk300` / `chunk2048`: inside a prefill chunk of 300 / 2,048 queries
//!   that starts 150 / 1,000 positions before `p`.
//!
//! Every scenario reads the same K, V and query values per position, so an
//! output that depends only on the row's own inputs is bit-identical to
//! `decode`. Each cell prints `=` or the count of differing BF16 elements of
//! the four rows (out of 4 x 64 heads x 128) and the largest difference;
//! `ref` is the worst error against the f32 reference `attend` and whether
//! every element is inside the attention test's tolerance. The executor's
//! own path is the positive control: if it shows no difference anywhere, the
//! matrix has no teeth and the run exits 1.
//!
//! Structures (`flashinfer_fa2_sink_paged_variants.cu` names the shapes):
//!
//! - `executor`: the executor's plan and instances, as `Attention::plan`
//!   makes them: the query tile from the step's longest packed query run,
//!   sliding page lists from the first query's first visible page.
//! - `exec-shapes+anchor`: the executor's three shapes, sliding layers
//!   anchored (page lists from a multiple of 128 positions, tiles at
//!   multiples of their width).
//! - `w1-k4` / `w1-k4+anchor`: one KV warp and 64-key tiles at every query
//!   tile (the executor's q128 shape, plus q16 and q64 instances of it).
//! - `w1-k4+anchor+splitC`: the same with global layers' KV split at
//!   multiples of `C` keys and merged by the executor image's merge kernel.
//! - `w4-k2` / `w4-k2+anchor`: four KV warps and 128-key tiles (the
//!   executor's q16 shape, plus a 64-row query tile of it).
//!
//! The query tile within a structure follows the executor's rule (16, 64 or
//! 128 rows by the step's longest packed run); a structure whose rows match
//! `decode` in every cell is invariant to it.
//!
//! **Timing.** Device time per attention call of one layer (attention plus,
//! for split structures, the merge), median of `--iters` launches enqueued
//! behind a long launch, for decode (1, 8, 32, 64 rows), verify (8 and 64
//! rows of four queries), prefill chunks (64, 364, 2,048 queries after 0 or
//! 4,096 positions), and a mixed step (64 decode rows beside a 364-query
//! chunk at 4,096). `step_us` scales a call to Flash's 48 layers (9 global,
//! 39 sliding) for the shapes both kinds run. Prints tables and one JSON line
//! per measurement (`{"section": "timing", "structure", "layer", "shape",
//! "context", "us"}` and `{"section": "matrix", ...}`).

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUevent_flags;
use eidola_engine_cuda::attention::{
    Attention, AttnLayer, AttnRequest, HEAD_DIM_QK, HEAD_DIM_VO, HostPlan, PagedKv, PagedParams,
    UintFastdiv,
};
use eidola_engine_cuda::launch::dptr;
use eidola_engine_cuda::{Gpu, Kernel, KernelDir, KernelModule, bf16};
use eidola_engine_model::attention::attend;
use eidola_engine_model::config::{AttentionKind, AttentionSpec};

const NQ: u32 = 64;
const BS: u32 = 16;
/// Anchored page lists start at a multiple of this many positions, a
/// multiple of every KV tile width (64 and 128).
const ANCHOR: u32 = 128;
const GLOBAL_LAYERS: f64 = 9.0;
const SLIDING_LAYERS: f64 = 39.0;
const EXECUTOR_IMAGE: &str = "flashinfer_fa2_sink_paged";
const VARIANT_IMAGE: &str = "flashinfer_fa2_sink_paged_variants";
const MERGE_META: &str = "eidola_fa2_merge_states_bf16_d128_meta";
/// Blocks of the persistent merge kernel (it strides over rows x heads).
const MERGE_BLOCKS: u32 = 148 * 4;

fn usize_of(x: u32) -> usize {
    usize::try_from(x).unwrap()
}

fn u32_of(x: usize) -> u32 {
    u32::try_from(x).unwrap()
}

fn i32_of(x: u32) -> i32 {
    i32::try_from(x).unwrap()
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let i = args.iter().position(|x| x == name)?;
    Some(
        args.get(i + 1)
            .unwrap_or_else(|| panic!("{name} takes a value")),
    )
}

fn list(args: &[String], name: &str, default: &str) -> Vec<u32> {
    flag(args, name)
        .unwrap_or(default)
        .split(',')
        .filter(|x| !x.is_empty())
        .map(|x| x.parse().unwrap())
        .collect()
}

struct Lcg(u64);

impl Lcg {
    fn f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// A position's query row, `[64 heads][192]`, the same wherever it is used.
fn q_row(pos: u32) -> Vec<u16> {
    let mut rng = Lcg(u64::from(pos).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x51);
    (0..usize_of(NQ * HEAD_DIM_QK))
        .map(|_| bf16::from_f32(rng.f32() * 2.0))
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Global,
    Sliding,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Global => "global",
            Kind::Sliding => "sliding",
        }
    }
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
    fn first_visible(self, q: u32) -> u32 {
        self.window().map_or(0, |w| (q + 1).saturating_sub(w))
    }
}

/// The request for queries `start .. start + len` over logical pages mapped
/// by `perm`: its pages from the first query's first visible page (the
/// executor's rule), or, anchored, from the multiple of `ANCHOR` positions at
/// or below it, so the kernel's key 0 is an absolute multiple of every tile
/// width.
fn paged_request(
    kind: Kind,
    perm: &[u32],
    q_start: u32,
    start: u32,
    len: u32,
    anchored: bool,
) -> AttnRequest {
    let first = kind.first_visible(start);
    let first = if anchored {
        first / ANCHOR * ANCHOR
    } else {
        first
    };
    let first_page = first / BS;
    let end = start + len;
    AttnRequest {
        q_start,
        qo_len: len,
        pages: (first_page..end.div_ceil(BS))
            .map(|l| perm[usize_of(l)])
            .collect(),
        kv_len: end - first_page * BS,
    }
}

/// One layer's KV for positions `0..positions`, paged through scrambled
/// physical pages, with host copies for the reference.
struct Pool {
    kind: Kind,
    k: CudaSlice<u16>,
    v: CudaSlice<u16>,
    k_host: Vec<u16>,
    v_host: Vec<u16>,
    /// Physical page of each logical page.
    perm: Vec<u32>,
    sinks: Vec<f32>,
    sink_dev: CudaSlice<f32>,
}

impl Pool {
    fn new(gpu: &Gpu, kind: Kind, positions: u32, seed: u64) -> Pool {
        let s = gpu.stream();
        let pages = positions.div_ceil(BS);
        let h = kind.kv_heads();
        let mut rng = Lcg(seed);
        // Logical page l lives at physical page perm[l]: a fixed shuffle.
        let mut perm: Vec<u32> = (0..pages).collect();
        for i in (1..perm.len()).rev() {
            rng.f32();
            let j = usize::try_from(rng.0 >> 33).unwrap() % (i + 1);
            perm.swap(i, j);
        }
        let k_pos = usize_of(h * HEAD_DIM_QK);
        let v_pos = usize_of(h * HEAD_DIM_VO);
        let total = usize_of(pages * BS);
        let mut k_host = vec![0u16; total * k_pos];
        let mut v_host = vec![0u16; total * v_pos];
        for x in &mut k_host {
            *x = bf16::from_f32(rng.f32() * 2.0);
        }
        for x in &mut v_host {
            *x = bf16::from_f32(rng.f32());
        }
        // Device layout: physical page p holds its 16 positions' K (and V).
        let mut k_dev = vec![0u16; total * k_pos];
        let mut v_dev = vec![0u16; total * v_pos];
        for (l, &p) in perm.iter().enumerate() {
            let (src, dst) = (l * usize_of(BS), usize_of(p * BS));
            let n = usize_of(BS);
            k_dev[dst * k_pos..(dst + n) * k_pos]
                .copy_from_slice(&k_host[src * k_pos..(src + n) * k_pos]);
            v_dev[dst * v_pos..(dst + n) * v_pos]
                .copy_from_slice(&v_host[src * v_pos..(src + n) * v_pos]);
        }
        let sinks: Vec<f32> = match kind.window() {
            Some(_) => (0..NQ).map(|_| 0.5 + rng.f32().abs()).collect(),
            None => vec![f32::NEG_INFINITY; usize_of(NQ)],
        };
        Pool {
            kind,
            k: s.clone_htod(&k_dev).unwrap(),
            v: s.clone_htod(&v_dev).unwrap(),
            k_host,
            v_host,
            perm,
            sink_dev: s.clone_htod(&sinks).unwrap(),
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
            sink: dptr(&self.sink_dev, s),
        }
    }

    fn request(&self, q_start: u32, start: u32, len: u32, anchored: bool) -> AttnRequest {
        paged_request(self.kind, &self.perm, q_start, start, len, anchored)
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
        let lo = usize_of(self.kind.first_visible(pos));
        let hi = usize_of(pos) + 1;
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

/// Which shapes a structure runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Family {
    /// The executor's plan and launch (`Attention::plan` / `run`).
    Executor,
    /// The executor's three shapes, launched here.
    ExecutorShapes,
    /// One KV warp, 64-key tiles.
    OneWarp,
    /// Four KV warps, 128-key tiles in 32-key stripes.
    FourWarp,
}

#[derive(Clone, Debug)]
struct Structure {
    name: String,
    family: Family,
    /// Sliding page lists and tiles anchored at absolute positions.
    anchored: bool,
    /// Global layers' KV split at multiples of this many keys (0: none).
    split: u32,
}

impl Structure {
    fn new(name: &str, family: Family, anchored: bool, split: u32) -> Structure {
        Structure {
            name: name.to_owned(),
            family,
            anchored,
            split,
        }
    }

    /// The image, entry and query tile for a step whose longest packed run
    /// picks `class` (16, 64 or 128) under the executor's rule.
    fn kernel(&self, kind: Kind, class: u32) -> (&'static str, &'static str, u32) {
        let anchored = self.anchored && kind == Kind::Sliding;
        let e = |tile: u32| -> (&'static str, &'static str, u32) {
            let entry = match tile {
                16 => "eidola_fa2_sink_paged_bf16_q16",
                64 => "eidola_fa2_sink_paged_bf16_q64",
                _ => "eidola_fa2_sink_paged_bf16_q128",
            };
            (EXECUTOR_IMAGE, entry, tile)
        };
        let v = |entry: &'static str, tile: u32| (VARIANT_IMAGE, entry, tile);
        match (self.family, anchored, class) {
            (Family::Executor, _, c) | (Family::ExecutorShapes, false, c) => e(c),
            (Family::ExecutorShapes, true, 16) => v("eidola_fa2v_anch_q16_w1x4_k2", 16),
            (Family::ExecutorShapes, true, 64) => v("eidola_fa2v_anch_q64_w4x1_k8", 64),
            (Family::ExecutorShapes, true, _) => v("eidola_fa2v_anch_q128_w4x1_k4", 128),
            (Family::OneWarp, false, 16) => v("eidola_fa2v_stock_q16_w1x1_k4", 16),
            (Family::OneWarp, false, 64) => v("eidola_fa2v_stock_q64_w4x1_k4", 64),
            (Family::OneWarp, false, _) => e(128),
            (Family::OneWarp, true, 16) => v("eidola_fa2v_anch_q16_w1x1_k4", 16),
            (Family::OneWarp, true, 64) => v("eidola_fa2v_anch_q64_w4x1_k4", 64),
            (Family::OneWarp, true, _) => v("eidola_fa2v_anch_q128_w4x1_k4", 128),
            (Family::FourWarp, false, 16) => e(16),
            (Family::FourWarp, false, _) => v("eidola_fa2v_stock_q64_w4x4_k2", 64),
            (Family::FourWarp, true, 16) => v("eidola_fa2v_anch_q16_w1x4_k2", 16),
            (Family::FourWarp, true, _) => v("eidola_fa2v_anch_q64_w4x4_k2", 64),
        }
    }
}

/// The executor's query-tile rule: the longest packed query run of the step.
fn class_of(queries: &[(u32, u32)], group: u32) -> u32 {
    let packed = queries.iter().map(|&(_, n)| n * group).max().unwrap_or(1);
    if packed <= 16 {
        16
    } else if packed <= 64 {
        64
    } else {
        128
    }
}

struct Kernels {
    executor: Attention,
    raw: Vec<(String, Kernel)>,
    merge: Kernel,
}

impl Kernels {
    fn load(gpu: &Gpu, dir: &KernelDir) -> Kernels {
        let executor =
            Attention::from_module(KernelModule::load(gpu, dir, EXECUTOR_IMAGE).unwrap()).unwrap();
        let exec_raw = KernelModule::load(gpu, dir, EXECUTOR_IMAGE).unwrap();
        let variants = KernelModule::load(gpu, dir, VARIANT_IMAGE).unwrap();
        let mut raw = Vec::new();
        for module in [&exec_raw, &variants] {
            for entry in &module.cubin().entries {
                if entry.meta == MERGE_META {
                    continue;
                }
                let kernel = module.kernel(&entry.symbol).unwrap();
                assert_eq!(
                    usize_of(kernel.meta().params_bytes),
                    std::mem::size_of::<PagedParams>(),
                    "{}",
                    entry.symbol
                );
                raw.push((entry.symbol.clone(), kernel));
            }
        }
        let merge_symbol = &exec_raw
            .cubin()
            .entry_for_meta(MERGE_META)
            .expect("merge entry")
            .symbol;
        let merge = exec_raw.kernel(merge_symbol).unwrap();
        Kernels {
            executor,
            raw,
            merge,
        }
    }

    fn raw(&self, symbol: &str) -> &Kernel {
        &self
            .raw
            .iter()
            .find(|(s, _)| s == symbol)
            .unwrap_or_else(|| panic!("{symbol}"))
            .1
    }
}

/// One attention call, uploaded and ready to launch any number of times.
struct Call<'a> {
    launch: Launch<'a>,
    /// Query rows and output (`[rows][64][128]`).
    _q: CudaSlice<u16>,
    out: CudaSlice<u16>,
}

enum Launch<'a> {
    Executor {
        attn: &'a Attention,
        plan: eidola_engine_cuda::attention::AttnPlan,
        layer: AttnLayer,
        q: u64,
        o: u64,
    },
    Raw {
        kernel: &'a Kernel,
        params: PagedParams,
        grid: [u32; 3],
        merge: Option<Merge<'a>>,
        _bufs: Vec<CudaSlice<i32>>,
        _chunk: CudaSlice<u32>,
    },
}

struct Merge<'a> {
    kernel: &'a Kernel,
    tmp_v: CudaSlice<u16>,
    tmp_s: CudaSlice<f32>,
    indptr: CudaSlice<i32>,
    out: u64,
    rows: u32,
}

impl Call<'_> {
    fn fire(&self, gpu: &Gpu) {
        let s = gpu.stream();
        match &self.launch {
            Launch::Executor {
                attn,
                plan,
                layer,
                q,
                o,
            } => unsafe { attn.run(gpu, plan, layer, NQ, *q, *o) }.unwrap(),
            Launch::Raw {
                kernel,
                params,
                grid,
                merge,
                ..
            } => {
                let mut p = *params;
                let mut args = [&mut p as *mut PagedParams as *mut std::ffi::c_void];
                // SAFETY: the single by-value argument is the kernel's
                // PagedParams (size checked at load); every address points
                // into buffers this call or its pool owns, sized for the
                // plan's rows and pages.
                unsafe { kernel.launch(s, *grid, &mut args) }.unwrap();
                if let Some(m) = merge {
                    let (v, sc, ind) = (dptr(&m.tmp_v, s), dptr(&m.tmp_s, s), dptr(&m.indptr, s));
                    // SAFETY: the merge reads `indptr[rows]` partial rows of
                    // `tmp_v` / `tmp_s` the attention launch wrote, and writes
                    // `rows` x 64 heads of `out`.
                    unsafe {
                        eidola_engine_cuda::launch!(
                            gpu,
                            m.kernel,
                            [MERGE_BLOCKS, 1, 1],
                            v,
                            sc,
                            ind,
                            m.out,
                            0u64,
                            m.rows,
                            0u64,
                            NQ
                        )
                    }
                    .unwrap();
                }
            }
        }
    }
}

/// A call's work list as FlashInfer's prefill planner lays it out with a
/// fixed split size (`PrefillSplitQOKVIndptr`): a work item per (request,
/// query tile, KV chunk), the partial outputs of request `i`'s query `q` at
/// rows `o_indptr[i] + q * chunks + c`, and the merge reading each query's
/// `chunks` partials in chunk order. Without a split there is one chunk and
/// `o_indptr` is `q_indptr`.
#[derive(Debug, Default)]
struct WorkList {
    q_indptr: Vec<i32>,
    indices: Vec<i32>,
    indptr: Vec<i32>,
    last_page_len: Vec<i32>,
    request_indices: Vec<i32>,
    qo_tile_indices: Vec<i32>,
    kv_tile_indices: Vec<i32>,
    o_indptr: Vec<i32>,
    merge_indptr: Vec<i32>,
}

impl WorkList {
    fn new(requests: &[AttnRequest], group: u32, tile: u32, split: u32) -> WorkList {
        let mut w = WorkList {
            q_indptr: vec![0],
            indptr: vec![0],
            o_indptr: vec![0],
            merge_indptr: vec![0],
            ..WorkList::default()
        };
        for (i, r) in requests.iter().enumerate() {
            w.q_indptr.push(i32_of(r.q_start + r.qo_len));
            w.indices.extend(r.pages.iter().map(|&p| i32_of(p)));
            w.indptr.push(i32_of(u32_of(w.indices.len())));
            w.last_page_len
                .push(i32_of(r.kv_len - (u32_of(r.pages.len()) - 1) * BS));
            let tiles = (r.qo_len * group).div_ceil(tile);
            let chunks = if split > 0 {
                r.kv_len.div_ceil(split)
            } else {
                1
            };
            for t in 0..tiles {
                for c in 0..chunks {
                    w.request_indices.push(i32_of(u32_of(i)));
                    w.qo_tile_indices.push(i32_of(t));
                    w.kv_tile_indices.push(i32_of(c));
                }
            }
            let last = *w.o_indptr.last().unwrap();
            w.o_indptr.push(last + i32_of(r.qo_len * chunks));
            for _ in 0..r.qo_len {
                let m = *w.merge_indptr.last().unwrap();
                w.merge_indptr.push(m + i32_of(chunks));
            }
        }
        w
    }
}

/// Upload one call of `structure` over `queries` (first position, count) of
/// separate sequences sharing the pool.
fn prepare<'a>(
    gpu: &Gpu,
    kernels: &'a Kernels,
    structure: &Structure,
    pool: &Pool,
    queries: &[(u32, u32)],
) -> Call<'a> {
    let s = gpu.stream();
    let kind = pool.kind;
    let group = kind.group();
    let anchored = structure.anchored && kind == Kind::Sliding;
    let mut requests = Vec::new();
    let mut q_host = Vec::new();
    let mut q_start = 0;
    for &(start, len) in queries {
        requests.push(pool.request(q_start, start, len, anchored));
        for pos in start..start + len {
            q_host.extend(q_row(pos));
        }
        q_start += len;
    }
    let rows = q_start;
    let q = s.clone_htod(&q_host).unwrap();
    let out = s
        .alloc_zeros::<u16>(usize_of(rows * NQ * HEAD_DIM_VO))
        .unwrap();
    let layer = pool.layer(gpu);
    // Every request is checked as the executor's plans are.
    HostPlan::new(&requests, group, BS).unwrap();
    if structure.family == Family::Executor {
        let plan = kernels.executor.plan(gpu, &requests, group, BS).unwrap();
        let (qp, op) = (dptr(&q, s), dptr(&out, s));
        return Call {
            launch: Launch::Executor {
                attn: &kernels.executor,
                plan,
                layer,
                q: qp,
                o: op,
            },
            _q: q,
            out,
        };
    }
    let class = class_of(queries, group);
    let (_, symbol, tile) = structure.kernel(kind, class);
    let kernel = kernels.raw(symbol);
    let split = if kind == Kind::Global {
        structure.split
    } else {
        0
    };
    let wl = WorkList::new(&requests, group, tile, split);
    let partials = u32::try_from(*wl.o_indptr.last().unwrap()).unwrap();
    let up = |v: &[i32]| s.clone_htod(v).unwrap();
    let bufs = vec![
        up(&wl.q_indptr),
        up(&wl.indices),
        up(&wl.indptr),
        up(&wl.last_page_len),
        up(&wl.request_indices),
        up(&wl.qo_tile_indices),
        up(&wl.kv_tile_indices),
        up(&wl.o_indptr),
    ];
    let chunk = s.clone_htod(&[split.max(1)]).unwrap();
    let h = kind.kv_heads();
    let work_items = u32_of(wl.request_indices.len());
    let merge = (split > 0).then(|| Merge {
        kernel: &kernels.merge,
        tmp_v: s
            .alloc_zeros::<u16>(usize_of(partials * NQ * HEAD_DIM_VO))
            .unwrap(),
        tmp_s: s.alloc_zeros::<f32>(usize_of(partials * NQ)).unwrap(),
        indptr: up(&wl.merge_indptr),
        out: dptr(&out, s),
        rows,
    });
    let (o_ptr, lse_ptr) = match &merge {
        Some(m) => (dptr(&m.tmp_v, s), dptr(&m.tmp_s, s)),
        None => (dptr(&out, s), 0),
    };
    let params = PagedParams {
        q: dptr(&q, s),
        paged_kv: PagedKv {
            page_size: UintFastdiv::new(BS),
            num_heads: h,
            head_dim: HEAD_DIM_QK,
            batch_size: u32_of(requests.len()),
            stride_page: layer.k_page_stride,
            stride_n: h * HEAD_DIM_QK,
            stride_h: HEAD_DIM_QK,
            v_stride_page: layer.v_page_stride,
            v_stride_n: h * HEAD_DIM_VO,
            v_stride_h: HEAD_DIM_VO,
            k_data: layer.k_base,
            v_data: layer.v_base,
            indices: dptr(&bufs[1], s),
            indptr: dptr(&bufs[2], s),
            last_page_len: dptr(&bufs[3], s),
            rope_pos_offset: 0,
        },
        q_indptr: dptr(&bufs[0], s),
        o: o_ptr,
        lse: lse_ptr,
        group_size: UintFastdiv::new(group),
        sink: layer.sink,
        sm_scale: 1.0 / f64::from(HEAD_DIM_QK).sqrt(),
        num_qo_heads: NQ,
        q_stride_n: i32_of(NQ * HEAD_DIM_QK),
        q_stride_h: i32_of(HEAD_DIM_QK),
        window_left: layer.window_left,
        request_indices: dptr(&bufs[4], s),
        qo_tile_indices: dptr(&bufs[5], s),
        kv_tile_indices: dptr(&bufs[6], s),
        o_indptr: dptr(&bufs[7], s),
        kv_chunk_size_ptr: dptr(&chunk, s),
        padded_batch_size: work_items,
        partition_kv: split > 0,
        ..PagedParams::default()
    };
    Call {
        launch: Launch::Raw {
            kernel,
            params,
            grid: [work_items, 1, h],
            merge,
            _bufs: bufs,
            _chunk: chunk,
        },
        _q: q,
        out,
    }
}

/// Run a call once and read rows `first .. first + n` of its output.
fn rows_of(gpu: &Gpu, call: &Call, first: u32, n: u32) -> Vec<u16> {
    call.fire(gpu);
    let all = gpu.stream().clone_dtoh(&call.out).unwrap();
    let w = usize_of(NQ * HEAD_DIM_VO);
    all[usize_of(first) * w..usize_of(first + n) * w].to_vec()
}

const PROBES: u32 = 4;
const SCENARIOS: [&str; 4] = ["+prefill", "verify", "chunk300", "chunk2048"];

/// The probe rows (positions `p ..= p + 3`) of every scenario: `decode`,
/// then `SCENARIOS` in order.
fn scenarios(gpu: &Gpu, kernels: &Kernels, st: &Structure, pool: &Pool, p: u32) -> Vec<Vec<u16>> {
    let mut decode = Vec::new();
    for i in 0..PROBES {
        let call = prepare(gpu, kernels, st, pool, &[(p + i, 1)]);
        decode.extend(rows_of(gpu, &call, 0, 1));
    }
    let mut out = vec![decode];
    // Four decode rows beside another sequence's 300-query chunk.
    let mut q: Vec<(u32, u32)> = (0..PROBES).map(|i| (p + i, 1)).collect();
    q.push((p - 700, 300));
    out.push(rows_of(
        gpu,
        &prepare(gpu, kernels, st, pool, &q),
        0,
        PROBES,
    ));
    out.push(rows_of(
        gpu,
        &prepare(gpu, kernels, st, pool, &[(p, PROBES)]),
        0,
        PROBES,
    ));
    out.push(rows_of(
        gpu,
        &prepare(gpu, kernels, st, pool, &[(p - 150, 300)]),
        150,
        PROBES,
    ));
    out.push(rows_of(
        gpu,
        &prepare(gpu, kernels, st, pool, &[(p - 1000, 2048)]),
        1000,
        PROBES,
    ));
    out
}

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

fn matrix(
    gpu: &Gpu,
    kernels: &Kernels,
    structures: &[Structure],
    pools: &[Pool],
    contexts: &[u32],
) -> bool {
    let per = usize_of(PROBES * NQ * HEAD_DIM_VO);
    println!("\n## Bitwise matrix: probe rows p..p+3 against `decode` (one launch per row)\n");
    println!(
        "{:<8} {:>6} {:<24} {:>22} {:>22} {:>22} {:>22} {:>16}",
        "layer",
        "p",
        "structure",
        SCENARIOS[0],
        SCENARIOS[1],
        SCENARIOS[2],
        SCENARIOS[3],
        "ref worst"
    );
    let mut control = 0usize;
    for pool in pools {
        for &p in contexts {
            let want: Vec<f32> = (0..PROBES).flat_map(|i| pool.reference(p + i)).collect();
            for st in structures {
                if st.split > 0 && pool.kind == Kind::Sliding {
                    continue;
                }
                let got = scenarios(gpu, kernels, st, pool, p);
                let mut cells = Vec::new();
                let mut json_cells = Vec::new();
                for (name, rows) in SCENARIOS.iter().zip(&got[1..]) {
                    assert_eq!(rows.len(), per);
                    let (n, worst) = diff(&got[0], rows);
                    if st.family == Family::Executor {
                        control += n;
                    }
                    cells.push(if n == 0 {
                        "=".to_owned()
                    } else {
                        format!("≠{n} ({worst:.1e})")
                    });
                    json_cells.push(format!(
                        "\"{name}\": {{\"differ\": {n}, \"max_abs\": {worst:e}}}"
                    ));
                }
                let (mut worst, mut outside) = (0f32, 0usize);
                for rows in &got {
                    for (&g, &w) in rows.iter().zip(&want) {
                        let err = (bf16::to_f32(g) - w).abs();
                        worst = worst.max(err);
                        if err > 4e-3 + w.abs() / 128.0 {
                            outside += 1;
                        }
                    }
                }
                let ok = if outside == 0 { "ok" } else { "OUT" };
                println!(
                    "{:<8} {:>6} {:<24} {:>22} {:>22} {:>22} {:>22} {:>9.1e} {ok:>6}",
                    pool.kind.name(),
                    p,
                    st.name,
                    cells[0],
                    cells[1],
                    cells[2],
                    cells[3],
                    worst
                );
                println!(
                    "{{\"section\": \"matrix\", \"structure\": \"{}\", \"layer\": \"{}\", \"p\": {p}, {}, \"ref_worst\": {worst:e}, \"ref_outside\": {outside}}}",
                    st.name,
                    pool.kind.name(),
                    json_cells.join(", ")
                );
            }
        }
    }
    println!(
        "\ncontrol: the executor's rows differ from its own decode rows in {control} elements"
    );
    control > 0
}

/// Microseconds per call, `iters` calls enqueued behind `head_start`.
fn time_us(gpu: &Gpu, iters: usize, head_start: &Call, call: &Call) -> f64 {
    call.fire(gpu);
    gpu.synchronize().unwrap();
    let ctx = gpu.context();
    let start = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    let end = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    head_start.fire(gpu);
    start.record(gpu.stream()).unwrap();
    for _ in 0..iters {
        call.fire(gpu);
    }
    end.record(gpu.stream()).unwrap();
    let ms = start.elapsed_ms(&end).unwrap();
    f64::from(ms) * 1000.0 / iters as f64
}

/// A step's sequences: (first query position, queries).
type Queries = Vec<(u32, u32)>;

/// The timed shapes: (label, context, queries).
fn shapes(contexts: &[u32]) -> Vec<(String, u32, Queries)> {
    let mut v = Vec::new();
    for &c in contexts {
        for rows in [1u32, 8, 32, 64] {
            v.push((
                format!("decode x{rows}"),
                c,
                (0..rows).map(|i| (c + 3 * i, 1)).collect(),
            ));
        }
        for rows in [8u32, 64] {
            v.push((
                format!("verify x{rows}x4"),
                c,
                (0..rows).map(|i| (c + 3 * i, 4)).collect(),
            ));
        }
    }
    for c in [0u32, 4096] {
        for n in [64u32, 364, 2048] {
            v.push((format!("prefill {n}"), c, vec![(c, n)]));
        }
    }
    let mut mixed: Queries = (0..64).map(|i| (4096 + 3 * i, 1)).collect();
    mixed.push((4096, 364));
    v.push(("mixed 64+364".to_owned(), 4096, mixed));
    v
}

fn timing(
    gpu: &Gpu,
    kernels: &Kernels,
    structures: &[Structure],
    pools: &[Pool],
    contexts: &[u32],
    iters: usize,
) {
    let executor = &structures[0];
    let head = prepare(gpu, kernels, executor, &pools[0], &[(4096, 2048)]);
    let shapes = shapes(contexts);
    println!("\n## Device time per layer call (us), mean of {iters} back-to-back calls\n");
    let mut header = format!("{:<8} {:<16} {:>6}", "layer", "shape", "ctx");
    for st in structures {
        header += &format!(" {:>22}", st.name);
    }
    println!("{header}");
    // [structure][shape] per layer kind.
    let mut table = vec![vec![vec![None; shapes.len()]; structures.len()]; pools.len()];
    for (pi, pool) in pools.iter().enumerate() {
        for (si, (label, c, queries)) in shapes.iter().enumerate() {
            let mut line = format!("{:<8} {:<16} {:>6}", pool.kind.name(), label, c);
            for (ti, st) in structures.iter().enumerate() {
                if st.split > 0 && pool.kind == Kind::Sliding {
                    line += &format!(" {:>22}", "-");
                    continue;
                }
                let call = prepare(gpu, kernels, st, pool, queries);
                let us = time_us(gpu, iters, &head, &call);
                table[pi][ti][si] = Some(us);
                line += &format!(" {us:>22.1}");
                println!(
                    "{{\"section\": \"timing\", \"structure\": \"{}\", \"layer\": \"{}\", \"shape\": \"{label}\", \"context\": {c}, \"us\": {us:.2}}}",
                    st.name,
                    pool.kind.name()
                );
            }
            println!("{line}");
        }
    }
    // Per step: 9 global + 39 sliding layers. A split structure's sliding
    // layers are its unsplit sibling's (splits apply to global layers only).
    println!("\n## Per step, 48 layers (ms): 9 x global + 39 x sliding\n");
    let mut header = format!("{:<16} {:>6}", "shape", "ctx");
    for st in structures {
        header += &format!(" {:>22}", st.name);
    }
    println!("{header}");
    let sliding_of = |ti: usize| -> usize {
        let st = &structures[ti];
        if st.split == 0 {
            return ti;
        }
        structures
            .iter()
            .position(|o| o.split == 0 && o.family == st.family && o.anchored == st.anchored)
            .unwrap_or(ti)
    };
    for (si, (label, c, _)) in shapes.iter().enumerate() {
        let mut line = format!("{label:<16} {c:>6}");
        for ti in 0..structures.len() {
            let g = table[0][ti][si];
            let s = table[1][sliding_of(ti)][si];
            match (g, s) {
                (Some(g), Some(s)) => {
                    let ms = (GLOBAL_LAYERS * g + SLIDING_LAYERS * s) / 1000.0;
                    line += &format!(" {ms:>22.3}");
                    println!(
                        "{{\"section\": \"step\", \"structure\": \"{}\", \"shape\": \"{label}\", \"context\": {c}, \"ms\": {ms:.4}}}",
                        structures[ti].name
                    );
                }
                _ => line += &format!(" {:>22}", "-"),
            }
        }
        println!("{line}");
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let Some(kernels_dir) = a.get(1) else {
        eprintln!("usage: see the module docs");
        std::process::exit(2);
    };
    let rest = &a[2..];
    let want_matrix = rest.iter().any(|x| x == "--matrix");
    let want_timing = rest.iter().any(|x| x == "--timing");
    let (want_matrix, want_timing) = if want_matrix || want_timing {
        (want_matrix, want_timing)
    } else {
        (true, true)
    };
    let iters: usize = flag(rest, "--iters").map_or(50, |x| x.parse().unwrap());
    let contexts = list(rest, "--contexts", "1037,4101,16389");
    let splits = list(rest, "--splits", "512,1024,2048");
    assert!(iters >= 1);
    assert!(
        contexts.iter().all(|&c| c >= 1000),
        "contexts start at 1000: the chunk scenarios begin 1,000 positions before p"
    );
    assert!(splits.iter().all(|&c| c > 0 && c % BS == 0));

    let gpu = Gpu::open(0).unwrap();
    let dir = KernelDir::new(kernels_dir);
    let kernels = Kernels::load(&gpu, &dir);
    // Positions every scenario and shape reaches.
    let top = contexts.iter().copied().max().unwrap().max(4096) + 2048 + 256;
    let pools = [
        Pool::new(&gpu, Kind::Global, top, 11),
        Pool::new(&gpu, Kind::Sliding, top, 12),
    ];
    let mut structures = vec![
        Structure::new("executor", Family::Executor, false, 0),
        Structure::new("exec-shapes+anchor", Family::ExecutorShapes, true, 0),
        Structure::new("w1-k4", Family::OneWarp, false, 0),
        Structure::new("w1-k4+anchor", Family::OneWarp, true, 0),
    ];
    for &c in &splits {
        structures.push(Structure::new(
            &format!("w1-k4+anchor+split{c}"),
            Family::OneWarp,
            true,
            c,
        ));
    }
    structures.push(Structure::new("w4-k2", Family::FourWarp, false, 0));
    structures.push(Structure::new("w4-k2+anchor", Family::FourWarp, true, 0));

    let mut teeth = true;
    if want_matrix {
        teeth = matrix(&gpu, &kernels, &structures, &pools, &contexts);
    }
    if want_timing {
        timing(&gpu, &kernels, &structures, &pools, &contexts, iters);
    }
    if !teeth {
        eprintln!("the executor's own rows never differed: the matrix has no teeth");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidola_engine_kernels::Manifest;

    /// With a fixed split, every (request, query tile, chunk) has one work
    /// item, each query's partials sit at `o_indptr[i] + q * chunks + c`,
    /// and the merge reads exactly those, in chunk order.
    #[test]
    fn work_list_is_flashinfers_split_layout() {
        let perm: Vec<u32> = (0..200).collect();
        let reqs = [
            paged_request(Kind::Global, &perm, 0, 1097, 3, false),
            paged_request(Kind::Global, &perm, 3, 39, 1, false),
        ];
        let w = WorkList::new(&reqs, 16, 16, 512);
        // 1,100 keys: 3 chunks; 3 queries x 16 heads: 3 tiles of 16.
        assert_eq!(w.request_indices.len(), 3 * 3 + 1);
        assert_eq!(w.o_indptr, vec![0, 9, 10]);
        assert_eq!(w.merge_indptr, vec![0, 3, 6, 9, 10]);
        assert_eq!(w.kv_tile_indices[..3], [0, 1, 2]);
        assert_eq!(w.qo_tile_indices[..4], [0, 0, 0, 1]);
        for (i, r) in reqs.iter().enumerate() {
            let chunks = i32_of(r.kv_len.div_ceil(512));
            for q in 0..r.qo_len {
                assert_eq!(
                    w.merge_indptr[usize_of(r.q_start + q)],
                    w.o_indptr[i] + i32_of(q) * chunks
                );
            }
        }
        let unsplit = WorkList::new(&reqs, 16, 16, 0);
        assert_eq!(unsplit.o_indptr, unsplit.q_indptr);
        assert_eq!(unsplit.request_indices.len(), 3 + 1);
        assert!(unsplit.kv_tile_indices.iter().all(|&c| c == 0));
    }

    /// Anchored sliding page lists start at an absolute multiple of 128
    /// positions at or below the first query's window, cover the KV exactly
    /// as the executor's plans must, and the executor's rule starts at the
    /// first visible page.
    #[test]
    fn anchored_pages_start_at_absolute_multiples() {
        let perm: Vec<u32> = (0..300).collect();
        for start in (0..3000).step_by(7) {
            for len in [1, 4, 300, 1000] {
                let fv = Kind::Sliding.first_visible(start);
                let a = paged_request(Kind::Sliding, &perm, 0, start, len, true);
                let first = start + len - a.kv_len;
                assert_eq!(first % ANCHOR, 0);
                assert!(first <= fv && fv < first + ANCHOR);
                HostPlan::new(&[a], 8, BS).unwrap();
                let e = paged_request(Kind::Sliding, &perm, 0, start, len, false);
                assert_eq!(start + len - e.kv_len, fv / BS * BS);
                let g = paged_request(Kind::Global, &perm, 0, start, len, true);
                assert_eq!(g.kv_len, start + len);
            }
        }
    }

    /// `AnchoredSink`'s widened window, as the CUDA computes it: the
    /// kernel's start for a query tile lands on the multiple of the KV tile
    /// at or below the tile's first row's first visible key, and the window
    /// only widens.
    #[test]
    fn anchored_variant_starts_tiles_on_multiples() {
        let sub = |a: u32, b: u32| a.saturating_sub(b);
        for (cta_q, cta_kv, group) in [(16, 128, 8), (64, 64, 8), (128, 64, 8), (64, 128, 16)] {
            for kv_len in [1u32, 5, 127, 128, 129, 300, 1037, 2200] {
                for qo_len in [1u32, 4, 37, 300] {
                    if qo_len > kv_len {
                        continue;
                    }
                    let wl = 127;
                    for tile in 0..(qo_len * group).div_ceil(cta_q) {
                        let q0 = tile * cta_q / group;
                        let first = sub(kv_len + q0, qo_len + wl);
                        let anchored = first / cta_kv * cta_kv;
                        let wl_eff = if first > 0 {
                            kv_len + q0 - qo_len - anchored
                        } else {
                            wl
                        };
                        assert!(wl_eff >= wl);
                        assert_eq!(sub(kv_len + q0, qo_len + wl_eff), anchored);
                    }
                }
            }
        }
    }

    /// Every structure runs one KV reduction shape at every query tile: the
    /// one-warp family 64-key tiles of one warp, the four-warp family
    /// 128-key tiles of four, and the executor's shapes the executor's; each
    /// entry exists in its image.
    #[test]
    fn structures_keep_one_shape_per_family() {
        let m = Manifest::embedded();
        for family in [Family::ExecutorShapes, Family::OneWarp, Family::FourWarp] {
            for anchored in [false, true] {
                let st = Structure::new("t", family, anchored, 0);
                for kind in [Kind::Global, Kind::Sliding] {
                    for class in [16, 64, 128] {
                        let (image, entry, tile) = st.kernel(kind, class);
                        assert!(
                            m.cubin(image, "sm_103a")
                                .and_then(|c| c.entry(entry))
                                .is_some(),
                            "{image} {entry}"
                        );
                        assert!(tile <= class.max(16), "{entry}");
                        let shape = match entry {
                            "eidola_fa2_sink_paged_bf16_q16" => "_w1x4_k2",
                            "eidola_fa2_sink_paged_bf16_q64" => "_w4x1_k8",
                            "eidola_fa2_sink_paged_bf16_q128" => "_w4x1_k4",
                            e => &e[e.len() - 8..],
                        };
                        let want = match family {
                            Family::OneWarp => vec!["_w1x1_k4", "_w4x1_k4"],
                            Family::FourWarp => vec!["_w1x4_k2", "_w4x4_k2"],
                            _ => vec!["_w1x4_k2", "_w4x1_k8", "_w4x1_k4"],
                        };
                        assert!(want.contains(&shape), "{family:?} {entry}");
                        let anchored_entry = entry.contains("_anch_");
                        assert_eq!(anchored_entry, anchored && kind == Kind::Sliding, "{entry}");
                    }
                }
            }
        }
    }

    /// The executor's tile rule.
    #[test]
    fn query_tile_class_is_the_executors() {
        assert_eq!(class_of(&[(5, 1)], 16), 16);
        assert_eq!(class_of(&[(5, 1), (9, 4)], 16), 64);
        assert_eq!(class_of(&[(5, 1), (9, 4)], 8), 64);
        assert_eq!(class_of(&[(5, 1), (9, 300)], 8), 128);
        assert_eq!(class_of(&[(5, 2)], 8), 16);
    }
}
