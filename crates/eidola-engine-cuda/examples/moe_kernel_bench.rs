//! Per-launch device time of the expert path's kernels (the router in every
//! form, gather, SwiGLU, combine) and the fused-QKV RoPE + KV write against
//! their single-block reference forms (`engine_ops_reference.cu`), and of the
//! placement and both grouped GEMMs (no reference form; zero weights for all
//! 256 experts), at Flash's shapes on synthetic data, per token count. No
//! checkpoint needed.
//!
//! ```text
//! moe_kernel_bench <kernels_dir> [--iters N] [--tokens 1,2,4,...]
//! ```
//!
//! Each measurement enqueues `--iters` launches (default 200) behind a
//! long-running launch, so the device runs them back to back whatever the
//! host's launch rate, and divides the time between two events around them.
//! The expert layout is the executor's for that token count, placed from the
//! router's own selection. The router is timed in both forms at every token
//! count (`router/token`, and `router/split` over both of its launches), with
//! the form the executor runs at that count marked `*`: the data
//! `ROUTER_PER_TOKEN_MAX` is chosen from. `qkv` is a sliding layer's (two KV
//! heads per chunk). Prints a table and one JSON line per kernel and token
//! count (`{"kernel", "tokens", "new_us", "reference_us", "executor"}`;
//! `reference_us` is null for a kernel without a reference form).

use cudarc::driver::sys::CUevent_flags;
use eidola_engine_cuda::bf16;
use eidola_engine_cuda::engine_ops::psum_rows;
use eidola_engine_cuda::engine_ops::{
    EngineOps, QkvArgs, RouterForm, executor_router, router_scores_len,
};
use eidola_engine_cuda::launch::dptr;
use eidola_engine_cuda::moe_gemm::{GROUPS, MoeGemm, MoeGemmArgs, MoeProj};
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
    fn f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
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
    assert!(iters >= 1 && token_counts.iter().all(|&t| t >= 1));

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
    let ids = s.alloc_zeros::<i32>(usize_of(max_tokens * TOP_K)).unwrap();
    let wts = s.alloc_zeros::<f32>(usize_of(max_tokens * TOP_K)).unwrap();
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
    let head_start = || unsafe {
        eidola_engine_cuda::launch!(
            gpu,
            ref_router,
            [HEAD_START_TOKENS, 1, 1],
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
    };

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

        // Router, both forms.
        let router = |form: RouterForm| {
            time_us(&gpu, iters, &head_start, &|| unsafe {
                ops.router_topk_form(
                    &gpu, form, pid, pw, psc, px, prw, pb, t, HIDDEN, EXPERTS, TOP_K, 1.0,
                )
                .unwrap();
            })
        };
        let per_token = router(RouterForm::PerToken);
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
        let chosen = executor_router(t);
        report(
            "router/token",
            chosen == RouterForm::PerToken,
            per_token,
            Some(old),
        );
        report(
            "router/split",
            chosen == RouterForm::Split,
            split,
            Some(old),
        );

        // The layout from this selection.
        unsafe {
            ops.router_topk(
                &gpu, pid, pw, psc, px, prw, pb, t, HIDDEN, EXPERTS, TOP_K, 1.0,
            )
            .unwrap();
        }
        let grouped = s.alloc_zeros::<i32>(usize_of(EXPERTS)).unwrap();
        let row_of = s.alloc_zeros::<i32>(usize_of(t * TOP_K)).unwrap();
        let (pgrouped, prow_of) = (dptr(&grouped, s), dptr(&row_of, s));
        let permute = time_us(&gpu, iters, &head_start, &|| unsafe {
            ops.moe_permute(&gpu, pgrouped, prow_of, pid, t, TOP_K, rows)
                .unwrap();
        });
        report("permute", true, permute, None);
        let host_row_of = s.clone_dtoh(&row_of).unwrap();
        let mut row_src = vec![-1i32; usize_of(rows)];
        for (i, &r) in host_row_of.iter().enumerate() {
            row_src[usize::try_from(r).unwrap()] = i32::try_from(i / usize_of(TOP_K)).unwrap();
        }
        let row_src = s.clone_htod(&row_src).unwrap();
        let prow_src = dptr(&row_src, s);

        // Gather.
        let a = s.alloc_zeros::<u8>(usize_of(rows * HIDDEN)).unwrap();
        let asf = s
            .alloc_zeros::<i32>(usize_of(HIDDEN / 512 * rows4))
            .unwrap();
        let (pa, pasf) = (dptr(&a, s), dptr(&asf, s));
        let new = time_us(&gpu, iters, &head_start, &|| unsafe {
            ops.gather_quant_ue8m0(&gpu, pa, pasf, px, prow_of, t, TOP_K, rows, HIDDEN, rows4)
                .unwrap();
        });
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

        // SwiGLU.
        let gu = s
            .alloc_zeros::<u16>(usize_of(rows) * usize_of(2 * INTER))
            .unwrap();
        let q = s.alloc_zeros::<u8>(usize_of(rows * INTER)).unwrap();
        let qsf = s.alloc_zeros::<i32>(usize_of(INTER / 512 * rows4)).unwrap();
        let (pgu, pq, pqsf) = (dptr(&gu, s), dptr(&q, s), dptr(&qsf, s));
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

        // Combine, over the down projection's rows.
        let edown = s
            .alloc_zeros::<u16>(usize_of(rows) * usize_of(HIDDEN))
            .unwrap();
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
            pedown,
        );
        report("gemm/down", true, down, None);

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
    println!();
    for line in json {
        println!("{line}");
    }
}
