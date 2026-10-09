//! Per-launch device time of the expert path's kernels (the router in every
//! form, gather, SwiGLU, combine) and the fused-QKV RoPE + KV write against
//! their single-block reference forms (`engine_ops_reference.cu`), and of the
//! placement and both grouped GEMMs (no reference form; zero weights for all
//! 256 experts), at Flash's shapes on synthetic data, per token count. No
//! checkpoint needed.
//!
//! ```text
//! moe_kernel_bench <kernels_dir> [--iters N] [--tokens 1,2,4,...] [--skew S]
//!                  [--gemm-only] [--variants]
//! ```
//!
//! Each measurement enqueues `--iters` launches (default 200) behind a
//! long-running launch, so the device runs them back to back whatever the
//! host's launch rate, and divides the time between two events around them.
//! The expert layout is the executor's for that token count, placed from the
//! router's own selection. The router is timed in every form at every token
//! count (`router/token`, `router/tiled`, `router/split`, the last both of its
//! launches), with the form the executor runs marked `*`: the data
//! `EXECUTOR_ROUTER` is chosen from. `qkv` is a sliding layer's (two KV
//! heads per chunk). Prints a table and one JSON line per kernel and token
//! count (`{"kernel", "tokens", "new_us", "reference_us", "executor"}`;
//! `reference_us` is null for a kernel without a reference form).
//!
//! `--skew S` replaces the router's selection (which spreads tokens nearly
//! uniformly over the experts) with `top_k` distinct experts per token drawn
//! from a Zipf law of exponent `S` over the experts in a fixed shuffled order,
//! weights `1 / top_k`; every kernel after the router runs on that routing. At
//! `S = 1.4` 64 tokens reach about 106 experts (the hottest about 62 of the
//! 64), against about 225 for the router's: a stand-in for real decode
//! traffic, whose grouped GEMMs cost about half the synthetic routing's at
//! 64 rows. `--gemm-only` times only the placement and the grouped GEMMs.
//!
//! `--variants` also runs the grouped-GEMM variants of the
//! `deepgemm_fp8_fp4_grouped_variants` image (`moe_gemm::MoeVariant`)
//! against the executor's instances, on the same routing and on random
//! expert weights (FP4 codes uniform, UE8M0 scales 2^-2 ..= 2^2) and token
//! rows: each over a layout placed with its own block M, gate/up from the
//! gathered rows, the SwiGLU of its own output, down from that. It checks
//! every routed row of both outputs against the executor's instances
//! (identical bits, or the count of differing elements and the largest
//! absolute difference), times both launches, and reports the experts the
//! routing reaches, the blocks each instance computes, and the weight bytes
//! those experts hold per microsecond (a lower bound on the HBM traffic: A,
//! the scales of A and D come on top). It starts with a device-to-device copy
//! of 2 GiB, whose read-plus-write bandwidth is a floor on what streaming can
//! reach on the part. JSON lines for these carry `kernel`
//! `variant/<name>/<proj>` (`base` for the executor's instances),
//! `reference_us` the executor's instance's time, and `experts`, `blocks`,
//! `weight_tb_s`, `identical`, `differing`, `max_abs_diff`. At one or two
//! tokens the reached experts' weights fit in L2, so back-to-back launches
//! overstate those counts' bandwidth.

use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaSlice, CudaStream};
use eidola_engine_cuda::bf16;
use eidola_engine_cuda::engine_ops::{
    EXECUTOR_ROUTER, EngineOps, QkvArgs, RouterForm, expert_placement_aligned, psum_rows,
    psum_rows_aligned, router_scores_len,
};
use eidola_engine_cuda::launch::dptr;
use eidola_engine_cuda::moe_gemm::{
    GROUPS, MoeGemm, MoeGemmArgs, MoeGemmVariants, MoeProj, MoeTile, MoeVariant,
};
use eidola_engine_cuda::{Gpu, KernelDir, KernelModule};

const HIDDEN: u32 = 4096;
const EXPERTS: u32 = 256;
const TOP_K: u32 = 8;
const INTER: u32 = 2048;
/// Tokens of the launch that keeps the device busy while the timed launches
/// are enqueued (the reference router at this size runs for milliseconds).
const HEAD_START_TOKENS: u32 = 4096;
/// A sliding layer's fused QKV: 4 chunks of 16 Q and 2 KV heads.
const QKV_CHUNKS: u32 = 4;
const QKV_Q_HEADS: u32 = 16;
const QKV_KV_HEADS: u32 = 2;
const QKV_STRIDE: u32 = 3712;
const KV_BLOCK: u32 = 16;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// Uniform in `[0, 1)` from the state's high 53 bits.
    fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `top_k` distinct experts per token (ascending, as the router writes
/// them), each draw from a Zipf law of exponent `s` over the experts in an
/// order shuffled once from `seed`.
fn skewed_ids(tokens: u32, top_k: u32, s: f64, seed: u64) -> Vec<i32> {
    let mut rng = Lcg(seed);
    let mut order: Vec<i32> = (0..i32::try_from(EXPERTS).unwrap()).collect();
    for i in (1..order.len()).rev() {
        let j = usize::try_from((rng.next() >> 33) % (i as u64 + 1)).unwrap();
        order.swap(i, j);
    }
    let mut cdf = Vec::with_capacity(order.len());
    let mut total = 0.0f64;
    for rank in 0..order.len() {
        total += 1.0 / ((rank + 1) as f64).powf(s);
        cdf.push(total);
    }
    let mut ids = Vec::with_capacity(usize_of(tokens * top_k));
    for _ in 0..tokens {
        let mut row: Vec<i32> = Vec::with_capacity(usize_of(top_k));
        while row.len() < usize_of(top_k) {
            let u = rng.f64() * total;
            let rank = cdf.partition_point(|&c| c <= u).min(order.len() - 1);
            if !row.contains(&order[rank]) {
                row.push(order[rank]);
            }
        }
        row.sort_unstable();
        ids.extend(row);
    }
    ids
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let i = args.iter().position(|x| x == name)?;
    Some(
        args.get(i + 1)
            .unwrap_or_else(|| panic!("{name} takes a value")),
    )
}

fn usize_of(x: u32) -> usize {
    usize::try_from(x).unwrap()
}

/// Microseconds per launch of `launch`, run `iters` times behind `head_start`.
fn time_us(gpu: &Gpu, iters: usize, head_start: &dyn Fn(), launch: &dyn Fn()) -> f64 {
    launch();
    gpu.synchronize().unwrap();
    let ctx = gpu.context();
    let start = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    let end = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    head_start();
    start.record(gpu.stream()).unwrap();
    for _ in 0..iters {
        launch();
    }
    end.record(gpu.stream()).unwrap();
    let ms = start.elapsed_ms(&end).unwrap();
    f64::from(ms) * 1000.0 / iters as f64
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let Some(kernels) = a.get(1) else {
        eprintln!("usage: see the module docs");
        std::process::exit(2);
    };
    let rest = &a[2..];
    let iters: usize = flag(rest, "--iters").map_or(200, |x| x.parse().unwrap());
    let token_counts: Vec<u32> = flag(rest, "--tokens")
        .unwrap_or("1,2,4,8,16,32,64,128,513,2048")
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    let skew: Option<f64> = flag(rest, "--skew").map(|x| x.parse().unwrap());
    let gemm_only = rest.iter().any(|x| x == "--gemm-only");
    let with_variants = rest.iter().any(|x| x == "--variants");
    assert!(iters >= 1 && token_counts.iter().all(|&t| t >= 1));
    assert!(skew.is_none_or(|s| s.is_finite() && s >= 0.0));

    let gpu = Gpu::open(0).unwrap();
    let dir = KernelDir::new(kernels);
    let ops = EngineOps::load(&gpu, &dir).unwrap();
    let reference = KernelModule::load(&gpu, &dir, "engine_ops_reference").unwrap();
    let ref_router = reference.kernel("eidola_reference_router_topk").unwrap();
    let ref_gather = reference
        .kernel("eidola_reference_gather_quant_ue8m0")
        .unwrap();
    let ref_swiglu = reference
        .kernel("eidola_reference_swiglu_quant_fp8_ue8m0")
        .unwrap();
    let ref_combine = reference.kernel("eidola_reference_moe_combine").unwrap();
    let ref_qkv = reference.kernel("eidola_reference_qkv_rope_kv").unwrap();
    let moe =
        MoeGemm::from_module(KernelModule::load(&gpu, &dir, "deepgemm_fp8_fp4_grouped").unwrap())
            .unwrap();
    let s = gpu.stream();
    // Every expert's weights and scales (zeros: the time does not depend on
    // the values), so a step reads what the executor's would.
    let expert_weights = |proj: MoeProj| {
        let (n, k, g) = (usize_of(proj.n()), usize_of(proj.k()), usize_of(GROUPS));
        (
            s.alloc_zeros::<u8>(g * n * k / 2).unwrap(),
            s.alloc_zeros::<i32>(g * (k / 128) * n).unwrap(),
        )
    };
    let (gate_up_w, gate_up_sf) = expert_weights(MoeProj::GateUp);
    let (down_w, down_sf) = expert_weights(MoeProj::Down);
    let variants = with_variants.then(|| {
        let module = KernelModule::load(&gpu, &dir, MoeGemmVariants::IMAGE)
            .expect("--variants needs a kernel build with the variants image");
        (
            MoeGemmVariants::from_module(module).unwrap(),
            RandomExperts::new(s, 7),
        )
    });

    let max_tokens = token_counts
        .iter()
        .copied()
        .max()
        .unwrap()
        .max(HEAD_START_TOKENS);
    let mut rng = Lcg(1);
    let x: Vec<f32> = (0..usize_of(max_tokens * HIDDEN))
        .map(|_| rng.f32())
        .collect();
    let w: Vec<u16> = (0..usize_of(EXPERTS * HIDDEN))
        .map(|_| bf16::from_f32(rng.f32() * 0.03))
        .collect();
    let bias: Vec<f32> = (0..EXPERTS).map(|_| rng.f32() * 1e-3).collect();
    let dx = s.clone_htod(&x).unwrap();
    let dw = s.clone_htod(&w).unwrap();
    let dbias = s.clone_htod(&bias).unwrap();
    let mut ids = s.alloc_zeros::<i32>(usize_of(max_tokens * TOP_K)).unwrap();
    let mut wts = s.alloc_zeros::<f32>(usize_of(max_tokens * TOP_K)).unwrap();
    let scores = s
        .alloc_zeros::<f32>(router_scores_len(usize_of(max_tokens), usize_of(EXPERTS)).unwrap())
        .unwrap();
    let (pid, pw, psc, px, prw, pb) = (
        dptr(&ids, s),
        dptr(&wts, s),
        dptr(&scores, s),
        dptr(&dx, s),
        dptr(&dw, s),
        dptr(&dbias, s),
    );
    // The head start writes its selection into its own buffers, so the
    // routing a measurement runs on (`--skew` replaces the router's) stays.
    let head_ids = s
        .alloc_zeros::<i32>(usize_of(HEAD_START_TOKENS * TOP_K))
        .unwrap();
    let head_wts = s
        .alloc_zeros::<f32>(usize_of(HEAD_START_TOKENS * TOP_K))
        .unwrap();
    let (head_pid, head_pw) = (dptr(&head_ids, s), dptr(&head_wts, s));
    let head_start = || unsafe {
        eidola_engine_cuda::launch!(
            gpu,
            ref_router,
            [HEAD_START_TOKENS, 1, 1],
            head_pid,
            head_pw,
            px,
            prw,
            pb,
            HIDDEN,
            EXPERTS,
            TOP_K,
            1.0f32
        )
        .unwrap();
    };

    if let Some((_, experts)) = &variants {
        let copy_us = time_us(&gpu, iters.min(20), &head_start, &|| unsafe {
            experts.copy_gate_up_weights(s).unwrap();
        });
        let bytes = 2.0 * experts.gate_up_w.len() as f64;
        println!(
            "roofline: device-to-device copy of {} MiB: {copy_us:.1} us, {:.2} TB/s read + write",
            experts.gate_up_w.len() >> 20,
            bytes / copy_us / 1e6
        );
    }

    println!(
        "{:>6} {:>13} {:>12} {:>14} {:>8}",
        "tokens", "kernel", "new us", "reference us", "speedup"
    );
    let mut json = Vec::new();
    for &t in &token_counts {
        let rows = u32::try_from(psum_rows(usize_of(t), usize_of(TOP_K))).unwrap();
        let rows4 = rows.div_ceil(4) * 4;
        let mut report = |kernel: &str, executor: bool, new_us: f64, ref_us: Option<f64>| {
            let mark = if executor { "*" } else { " " };
            match ref_us {
                Some(r) => println!(
                    "{t:>6} {kernel:>12}{mark} {new_us:>12.2} {r:>14.2} {:>7.1}x",
                    r / new_us
                ),
                None => println!(
                    "{t:>6} {kernel:>12}{mark} {new_us:>12.2} {:>14} {:>8}",
                    "-", "-"
                ),
            }
            let ref_json = ref_us.map_or("null".to_string(), |r| format!("{r:.3}"));
            json.push(format!(
                "{{\"kernel\":\"{kernel}\",\"tokens\":{t},\"new_us\":{new_us:.3},\"reference_us\":{ref_json},\"executor\":{executor}}}"
            ));
        };

        // Router, every form.
        if !gemm_only {
            let router = |form: RouterForm| {
                time_us(&gpu, iters, &head_start, &|| unsafe {
                    ops.router_topk_form(
                        &gpu, form, pid, pw, psc, px, prw, pb, t, HIDDEN, EXPERTS, TOP_K, 1.0,
                    )
                    .unwrap();
                })
            };
            let per_token = router(RouterForm::PerToken);
            let tiled = router(RouterForm::Tiled);
            let split = router(RouterForm::Split);
            let old = time_us(&gpu, iters, &head_start, &|| unsafe {
                eidola_engine_cuda::launch!(
                    gpu,
                    ref_router,
                    [t, 1, 1],
                    pid,
                    pw,
                    px,
                    prw,
                    pb,
                    HIDDEN,
                    EXPERTS,
                    TOP_K,
                    1.0f32
                )
                .unwrap();
            });
            let chosen = EXECUTOR_ROUTER;
            report(
                "router/token",
                chosen == RouterForm::PerToken,
                per_token,
                Some(old),
            );
            report(
                "router/tiled",
                chosen == RouterForm::Tiled,
                tiled,
                Some(old),
            );
            report(
                "router/split",
                chosen == RouterForm::Split,
                split,
                Some(old),
            );
        }

        // The routing: the router's own selection, or the skewed draw in its
        // place.
        unsafe {
            ops.router_topk(
                &gpu, pid, pw, psc, px, prw, pb, t, HIDDEN, EXPERTS, TOP_K, 1.0,
            )
            .unwrap();
        }
        if let Some(skew) = skew {
            let host = skewed_ids(t, TOP_K, skew, u64::from(t));
            s.memcpy_htod(&host, &mut ids).unwrap();
            let weights = vec![1.0 / TOP_K as f32; host.len()];
            s.memcpy_htod(&weights, &mut wts).unwrap();
        }
        let host_ids = {
            let all = s.clone_dtoh(&ids).unwrap();
            all[..usize_of(t * TOP_K)].to_vec()
        };

        // The layout from this selection.
        let grouped = s.alloc_zeros::<i32>(usize_of(EXPERTS)).unwrap();
        let row_of = s.alloc_zeros::<i32>(usize_of(t * TOP_K)).unwrap();
        let (pgrouped, prow_of) = (dptr(&grouped, s), dptr(&row_of, s));
        let permute = time_us(&gpu, iters, &head_start, &|| unsafe {
            ops.moe_permute(&gpu, pgrouped, prow_of, pid, t, TOP_K, rows)
                .unwrap();
        });
        report("permute", true, permute, None);

        // Gather.
        let a = s.alloc_zeros::<u8>(usize_of(rows * HIDDEN)).unwrap();
        let asf = s
            .alloc_zeros::<i32>(usize_of(HIDDEN / 512 * rows4))
            .unwrap();
        let (pa, pasf) = (dptr(&a, s), dptr(&asf, s));
        let gather = || unsafe {
            ops.gather_quant_ue8m0(&gpu, pa, pasf, px, prow_of, t, TOP_K, rows, HIDDEN, rows4)
                .unwrap();
        };
        if gemm_only {
            gather();
        } else {
            let host_row_of = s.clone_dtoh(&row_of).unwrap();
            let mut row_src = vec![-1i32; usize_of(rows)];
            for (i, &r) in host_row_of.iter().enumerate() {
                row_src[usize::try_from(r).unwrap()] = i32::try_from(i / usize_of(TOP_K)).unwrap();
            }
            let row_src = s.clone_htod(&row_src).unwrap();
            let prow_src = dptr(&row_src, s);
            let new = time_us(&gpu, iters, &head_start, &gather);
            let old = time_us(&gpu, iters, &head_start, &|| unsafe {
                eidola_engine_cuda::launch!(
                    gpu,
                    ref_gather,
                    [rows, HIDDEN / 512, 1],
                    pa,
                    pasf,
                    px,
                    prow_src,
                    HIDDEN,
                    rows4
                )
                .unwrap();
            });
            report("gather", true, new, Some(old));
        }

        // SwiGLU.
        let gu = s
            .alloc_zeros::<u16>(usize_of(rows) * usize_of(2 * INTER))
            .unwrap();
        let q = s.alloc_zeros::<u8>(usize_of(rows * INTER)).unwrap();
        let qsf = s.alloc_zeros::<i32>(usize_of(INTER / 512 * rows4)).unwrap();
        let (pgu, pq, pqsf) = (dptr(&gu, s), dptr(&q, s), dptr(&qsf, s));
        if !gemm_only {
            let new = time_us(&gpu, iters, &head_start, &|| unsafe {
                ops.swiglu_quant_ue8m0(&gpu, pq, pqsf, pgu, prow_of, t, TOP_K, rows, INTER, rows4)
                    .unwrap();
            });
            let old = time_us(&gpu, iters, &head_start, &|| unsafe {
                eidola_engine_cuda::launch!(
                    gpu,
                    ref_swiglu,
                    [rows, INTER / 512, 1],
                    pq,
                    pqsf,
                    pgu,
                    INTER,
                    rows4
                )
                .unwrap();
            });
            report("swiglu", true, new, Some(old));
        }

        // Combine, over the down projection's rows.
        let edown = s
            .alloc_zeros::<u16>(usize_of(rows) * usize_of(HIDDEN))
            .unwrap();
        if !gemm_only {
            let out = s.alloc_zeros::<f32>(usize_of(t * HIDDEN)).unwrap();
            let (pedown, pout) = (dptr(&edown, s), dptr(&out, s));
            let new = time_us(&gpu, iters, &head_start, &|| unsafe {
                ops.moe_combine(&gpu, pout, pedown, prow_of, pw, t, HIDDEN, TOP_K)
                    .unwrap();
            });
            let old = time_us(&gpu, iters, &head_start, &|| unsafe {
                eidola_engine_cuda::launch!(
                    gpu,
                    ref_combine,
                    [t, 1, 1],
                    pout,
                    pedown,
                    prow_of,
                    pw,
                    HIDDEN,
                    TOP_K
                )
                .unwrap();
            });
            report("combine", true, new, Some(old));
        }

        // The grouped GEMMs over this layout: gate/up from the gathered rows
        // into the SwiGLU's input, down from the SwiGLU's output.
        let gemm = |proj, a, sfa, b, sfb, d| {
            let args = MoeGemmArgs {
                proj,
                m: rows,
                grouped_layout: pgrouped,
                a,
                sfa,
                b,
                sfb,
                d,
            };
            time_us(&gpu, iters, &head_start, &|| unsafe {
                moe.launch(&gpu, &args).unwrap();
            })
        };
        let gate_up = gemm(
            MoeProj::GateUp,
            pa,
            pasf,
            dptr(&gate_up_w, s),
            dptr(&gate_up_sf, s),
            pgu,
        );
        report("gemm/gate_up", true, gate_up, None);
        let down = gemm(
            MoeProj::Down,
            pq,
            pqsf,
            dptr(&down_w, s),
            dptr(&down_sf, s),
            dptr(&edown, s),
        );
        report("gemm/down", true, down, None);
        drop((a, asf, gu, q, qsf, edown));

        if !gemm_only {
            // Fused QKV RoPE + KV write: token i at block 1 + i / 16, slot i % 16.
            let nkv = QKV_CHUNKS * QKV_KV_HEADS;
            let block_elems = u64::from(KV_BLOCK * nkv * (192 + 128));
            let qkv = s
                .alloc_zeros::<u16>(usize_of(t * QKV_CHUNKS * QKV_STRIDE))
                .unwrap();
            let q = s
                .alloc_zeros::<u16>(usize_of(t * QKV_CHUNKS * QKV_Q_HEADS * 192))
                .unwrap();
            let blocks = t.div_ceil(KV_BLOCK) + 1;
            let pool = s
                .alloc_zeros::<u16>(usize_of(blocks) * usize::try_from(block_elems).unwrap())
                .unwrap();
            let rope = s.alloc_zeros::<f32>(usize_of(t * 64)).unwrap();
            let positions = s.clone_htod(&(0..t).collect::<Vec<u32>>()).unwrap();
            let kv_block = s
                .clone_htod(&(0..t).map(|i| 1 + i / KV_BLOCK).collect::<Vec<u32>>())
                .unwrap();
            let kv_slot = s
                .clone_htod(&(0..t).map(|i| i % KV_BLOCK).collect::<Vec<u32>>())
                .unwrap();
            let args = QkvArgs {
                qkv: dptr(&qkv, s),
                q_out: dptr(&q, s),
                pool: dptr(&pool, s),
                positions: dptr(&positions, s),
                kv_block: dptr(&kv_block, s),
                kv_slot: dptr(&kv_slot, s),
                rope: dptr(&rope, s),
                block_elems,
                k_off: 0,
                v_off: u64::from(KV_BLOCK * nkv * 192),
                chunk_stride: QKV_STRIDE,
                chunks: QKV_CHUNKS,
                q_heads_per_chunk: QKV_Q_HEADS,
                kv_heads_per_chunk: QKV_KV_HEADS,
            };
            let new = time_us(&gpu, iters, &head_start, &|| unsafe {
                ops.qkv_rope_kv(&gpu, args, t).unwrap();
            });
            let old = time_us(&gpu, iters, &head_start, &|| unsafe {
                eidola_engine_cuda::launch!(gpu, ref_qkv, [t, 1, 1], args).unwrap();
            });
            report("qkv", true, new, Some(old));
        }

        if let Some((variants, experts)) = &variants {
            let bench = VariantBench {
                gpu: &gpu,
                ops: &ops,
                moe: &moe,
                variants,
                experts,
                iters,
                head_start: &head_start,
                px,
            };
            bench.run(t, &host_ids, &mut json);
        }
    }
    println!();
    for line in json {
        println!("{line}");
    }
}

/// Random expert weights for the variants' comparison: FP4 codes uniform
/// over all sixteen values (every code is finite), UE8M0 scales 2^-2 ..= 2^2,
/// for all 256 experts of both projections.
struct RandomExperts {
    gate_up_w: CudaSlice<u8>,
    gate_up_sf: CudaSlice<i32>,
    down_w: CudaSlice<u8>,
    down_sf: CudaSlice<i32>,
    /// The copy roofline's destination, as large as the gate/up weights.
    copy_dst: CudaSlice<u8>,
}

impl RandomExperts {
    fn new(s: &Arc<CudaStream>, seed: u64) -> RandomExperts {
        let mut rng = Lcg(seed);
        let mut weights = |proj: MoeProj| {
            let (n, k, g) = (usize_of(proj.n()), usize_of(proj.k()), usize_of(GROUPS));
            let mut w = vec![0u8; g * n * k / 2];
            for chunk in w.chunks_mut(8) {
                let bytes = rng.next().to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }
            let sf: Vec<i32> = (0..g * (k / 128) * n)
                .map(|_| {
                    let r = rng.next() >> 32;
                    let e = |i: u64| 125 + u8::try_from((r >> (8 * i)) % 5).unwrap();
                    i32::from_le_bytes([e(0), e(1), e(2), e(3)])
                })
                .collect();
            (s.clone_htod(&w).unwrap(), s.clone_htod(&sf).unwrap())
        };
        let (gate_up_w, gate_up_sf) = weights(MoeProj::GateUp);
        let (down_w, down_sf) = weights(MoeProj::Down);
        let copy_dst = s.alloc_zeros::<u8>(gate_up_w.len()).unwrap();
        RandomExperts {
            gate_up_w,
            gate_up_sf,
            down_w,
            down_sf,
            copy_dst,
        }
    }

    fn weights(&self, proj: MoeProj, s: &Arc<CudaStream>) -> (u64, u64) {
        match proj {
            MoeProj::GateUp => (dptr(&self.gate_up_w, s), dptr(&self.gate_up_sf, s)),
            MoeProj::Down => (dptr(&self.down_w, s), dptr(&self.down_sf, s)),
        }
    }

    /// Enqueue a copy of the gate/up weights into `copy_dst`.
    ///
    /// # Safety
    ///
    /// Both buffers belong to `s`'s context and are `gate_up_w.len()` bytes.
    unsafe fn copy_gate_up_weights(
        &self,
        s: &Arc<CudaStream>,
    ) -> Result<(), cudarc::driver::DriverError> {
        let (src, dst) = (dptr(&self.gate_up_w, s), dptr(&self.copy_dst, s));
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(dst, src, self.gate_up_w.len(), s.cu_stream())
        }
    }
}

/// One instance set over its own layout: the executor's (`None`) or a
/// variant.
#[derive(Clone, Copy)]
struct Config {
    variant: Option<MoeVariant>,
}

impl Config {
    fn name(self) -> &'static str {
        self.variant.map_or("base", MoeVariant::name)
    }

    fn tile(self) -> MoeTile {
        self.variant.map_or(MoeTile::EXECUTOR, MoeVariant::tile)
    }
}

/// What one configuration produced at one token count: both launches'
/// times and every routed pair's rows of both outputs (BF16 bits, pair-major).
struct ConfigRun {
    gate_up_us: f64,
    down_us: f64,
    gate_up_rows: Vec<u16>,
    down_rows: Vec<u16>,
}

struct VariantBench<'a> {
    gpu: &'a Gpu,
    ops: &'a EngineOps,
    moe: &'a MoeGemm,
    variants: &'a MoeGemmVariants,
    experts: &'a RandomExperts,
    iters: usize,
    head_start: &'a dyn Fn(),
    /// The tokens' f32 rows.
    px: u64,
}

impl VariantBench<'_> {
    /// Run the executor's instances and every variant over `ids` (`t`
    /// tokens' routing), compare, report.
    fn run(&self, t: u32, ids: &[i32], json: &mut Vec<String>) {
        let mut counts = vec![0usize; usize_of(EXPERTS)];
        for &id in ids {
            counts[usize::try_from(id).unwrap()] += 1;
        }
        let reached = counts.iter().filter(|&&c| c > 0).count();
        let hottest = counts.iter().copied().max().unwrap_or(0);
        let base = self.run_config(Config { variant: None }, t, ids);
        let configs = std::iter::once(Config { variant: None })
            .chain(MoeVariant::ALL.map(|v| Config { variant: Some(v) }));
        println!(
            "{t:>6} variants: {reached} experts reached, the hottest {hottest} rows; \
             weights TB/s counts each reached expert's FP4 weights and scales once"
        );
        for config in configs {
            let run = config.variant.map(|_| self.run_config(config, t, ids));
            let run = run.as_ref().unwrap_or(&base);
            let block_m = usize_of(config.tile().block_m);
            let m_blocks: usize = counts.iter().map(|c| c.div_ceil(block_m)).sum();
            for (proj, us, base_us, rows, base_rows) in [
                (
                    MoeProj::GateUp,
                    run.gate_up_us,
                    base.gate_up_us,
                    &run.gate_up_rows,
                    &base.gate_up_rows,
                ),
                (
                    MoeProj::Down,
                    run.down_us,
                    base.down_us,
                    &run.down_rows,
                    &base.down_rows,
                ),
            ] {
                let (n, k) = (u64::from(proj.n()), u64::from(proj.k()));
                let expert_bytes = n * k / 2 + n * k / 32;
                let tb_s = (reached as u64 * expert_bytes) as f64 / us / 1e6;
                let blocks = m_blocks * usize_of(proj.n() / 128);
                let diff = compare(base_rows, rows);
                let proj_name = match proj {
                    MoeProj::GateUp => "gate_up",
                    MoeProj::Down => "down",
                };
                let kernel = format!("variant/{}/{proj_name}", config.name());
                let verdict = if diff.differing == 0 {
                    "identical".to_string()
                } else {
                    format!(
                        "{} differ, max |d| {:.3e}",
                        diff.differing, diff.max_abs_diff
                    )
                };
                println!(
                    "{t:>6} {kernel:>24}{} {us:>10.2} us {:>6.3}x {blocks:>6} blocks {tb_s:>6.2} TB/s  {verdict}",
                    if config.variant.is_none() { "*" } else { " " },
                    base_us / us,
                );
                json.push(format!(
                    "{{\"kernel\":\"{kernel}\",\"tokens\":{t},\"new_us\":{us:.3},\"reference_us\":{base_us:.3},\"executor\":{},\"experts\":{reached},\"blocks\":{blocks},\"weight_tb_s\":{tb_s:.3},\"identical\":{},\"differing\":{},\"max_abs_diff\":{}}}",
                    config.variant.is_none(),
                    diff.differing == 0,
                    diff.differing,
                    if diff.max_abs_diff.is_finite() {
                        format!("{:e}", diff.max_abs_diff)
                    } else {
                        "null".to_string()
                    },
                ));
            }
        }
    }

    /// Gather, gate/up, SwiGLU and down for `config` over a layout placed
    /// with its block M; read back every pair's rows, then time both GEMMs.
    fn run_config(&self, config: Config, t: u32, ids: &[i32]) -> ConfigRun {
        let (gpu, ops, s) = (self.gpu, self.ops, self.gpu.stream());
        let align = usize_of(config.tile().block_m);
        let rows = u32::try_from(psum_rows_aligned(usize_of(t), usize_of(TOP_K), align)).unwrap();
        let rows4 = rows.div_ceil(4) * 4;
        let (row_of, grouped) = expert_placement_aligned(ids, align);
        let d_row_of = s.clone_htod(&row_of).unwrap();
        let d_grouped = s.clone_htod(&grouped).unwrap();
        let (prow_of, pgrouped) = (dptr(&d_row_of, s), dptr(&d_grouped, s));
        let a = s.alloc_zeros::<u8>(usize_of(rows * HIDDEN)).unwrap();
        let asf = s
            .alloc_zeros::<i32>(usize_of(HIDDEN / 512 * rows4))
            .unwrap();
        let gu = s
            .alloc_zeros::<u16>(usize_of(rows) * usize_of(2 * INTER))
            .unwrap();
        let q = s.alloc_zeros::<u8>(usize_of(rows * INTER)).unwrap();
        let qsf = s.alloc_zeros::<i32>(usize_of(INTER / 512 * rows4)).unwrap();
        let down = s
            .alloc_zeros::<u16>(usize_of(rows) * usize_of(HIDDEN))
            .unwrap();
        let (pa, pasf, pgu) = (dptr(&a, s), dptr(&asf, s), dptr(&gu, s));
        let (pq, pqsf, pdown) = (dptr(&q, s), dptr(&qsf, s), dptr(&down, s));
        let args = |proj: MoeProj| {
            let (b, sfb) = self.experts.weights(proj, s);
            let (a, sfa, d) = match proj {
                MoeProj::GateUp => (pa, pasf, pgu),
                MoeProj::Down => (pq, pqsf, pdown),
            };
            MoeGemmArgs {
                proj,
                m: rows,
                grouped_layout: pgrouped,
                a,
                sfa,
                b,
                sfb,
                d,
            }
        };
        let (gate_up_args, down_args) = (args(MoeProj::GateUp), args(MoeProj::Down));
        let launch = |args: &MoeGemmArgs| unsafe {
            match config.variant {
                None => self.moe.launch(gpu, args),
                Some(v) => self.variants.launch(gpu, v, args),
            }
            .unwrap();
        };
        unsafe {
            ops.gather_quant_ue8m0(
                gpu, pa, pasf, self.px, prow_of, t, TOP_K, rows, HIDDEN, rows4,
            )
            .unwrap();
        }
        launch(&gate_up_args);
        unsafe {
            ops.swiglu_quant_ue8m0(gpu, pq, pqsf, pgu, prow_of, t, TOP_K, rows, INTER, rows4)
                .unwrap();
        }
        launch(&down_args);
        let pair_rows = |buf: &CudaSlice<u16>, width: usize| {
            let all = s.clone_dtoh(buf).unwrap();
            let mut out = Vec::with_capacity(row_of.len() * width);
            for &r in &row_of {
                let r = usize::try_from(r).unwrap();
                out.extend_from_slice(&all[r * width..(r + 1) * width]);
            }
            out
        };
        let gate_up_rows = pair_rows(&gu, usize_of(2 * INTER));
        let down_rows = pair_rows(&down, usize_of(HIDDEN));
        let gate_up_us = time_us(gpu, self.iters, self.head_start, &|| launch(&gate_up_args));
        let down_us = time_us(gpu, self.iters, self.head_start, &|| launch(&down_args));
        ConfigRun {
            gate_up_us,
            down_us,
            gate_up_rows,
            down_rows,
        }
    }
}

struct Diff {
    differing: usize,
    max_abs_diff: f32,
}

/// Elements whose BF16 bits differ, and the largest absolute difference
/// among them (infinite where either side is not finite).
fn compare(base: &[u16], other: &[u16]) -> Diff {
    assert_eq!(base.len(), other.len());
    let mut diff = Diff {
        differing: 0,
        max_abs_diff: 0.0,
    };
    for (&x, &y) in base.iter().zip(other) {
        if x != y {
            diff.differing += 1;
            let (x, y) = (bf16::to_f32(x), bf16::to_f32(y));
            let d = if x.is_finite() && y.is_finite() {
                (x - y).abs()
            } else {
                f32::INFINITY
            };
            diff.max_abs_diff = diff.max_abs_diff.max(d);
        }
    }
    diff
}
