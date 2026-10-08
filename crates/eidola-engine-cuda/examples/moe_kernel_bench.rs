//! Per-launch device time of the expert path's router, gather and SwiGLU
//! kernels against their single-block reference forms
//! (`engine_ops_reference.cu`), at Flash's shapes on synthetic data, per token
//! count. No checkpoint needed.
//!
//! ```text
//! moe_kernel_bench <kernels_dir> [--iters N] [--tokens 1,2,4,...]
//! ```
//!
//! Each measurement enqueues `--iters` launches (default 200) behind a
//! long-running launch, so the device runs them back to back whatever the
//! host's launch rate, and divides the time between two events around them.
//! The expert layout is the executor's for that token count (masked up to 128
//! tokens, contiguous above), placed from the router's own selection. Prints a
//! table and one JSON line per kernel and token count
//! (`{"kernel", "tokens", "new_us", "reference_us"}`).

use cudarc::driver::sys::CUevent_flags;
use eidola_engine_cuda::bf16;
use eidola_engine_cuda::engine_ops::EngineOps;
use eidola_engine_cuda::launch::dptr;
use eidola_engine_cuda::{Gpu, KernelDir, KernelModule};

const HIDDEN: u32 = 4096;
const EXPERTS: u32 = 256;
const TOP_K: u32 = 8;
const INTER: u32 = 2048;
const CAP: u32 = 128;
const BLOCK_M: u32 = 128;
/// Tokens of the launch that keeps the device busy while the timed launches
/// are enqueued (the reference router at this size runs for milliseconds).
const HEAD_START_TOKENS: u32 = 4096;

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
    let s = gpu.stream();

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
    let (pid, pw, px, prw, pb) = (
        dptr(&ids, s),
        dptr(&wts, s),
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
        "{:>6} {:>8} {:>12} {:>14} {:>8}",
        "tokens", "kernel", "new us", "reference us", "speedup"
    );
    let mut json = Vec::new();
    for &t in &token_counts {
        let masked = t <= CAP;
        let (rows, cap) = if masked {
            (EXPERTS * CAP, CAP)
        } else {
            let n = t * TOP_K;
            (
                (n + n.min(EXPERTS) * (BLOCK_M - 1)).div_ceil(BLOCK_M) * BLOCK_M,
                0,
            )
        };
        let rows4 = rows.div_ceil(4) * 4;
        let mut report = |kernel: &str, new_us: f64, ref_us: f64| {
            println!(
                "{t:>6} {kernel:>8} {new_us:>12.2} {ref_us:>14.2} {:>7.1}x",
                ref_us / new_us
            );
            json.push(format!(
                "{{\"kernel\":\"{kernel}\",\"tokens\":{t},\"new_us\":{new_us:.3},\"reference_us\":{ref_us:.3}}}"
            ));
        };

        // Router.
        let new = time_us(&gpu, iters, &head_start, &|| unsafe {
            ops.router_topk(&gpu, pid, pw, px, prw, pb, t, HIDDEN, EXPERTS, TOP_K, 1.0)
                .unwrap();
        });
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
        report("router", new, old);

        // The layout from this selection.
        unsafe {
            ops.router_topk(&gpu, pid, pw, px, prw, pb, t, HIDDEN, EXPERTS, TOP_K, 1.0)
                .unwrap();
        }
        let grouped = s.alloc_zeros::<i32>(usize_of(rows.max(EXPERTS))).unwrap();
        let row_of = s.alloc_zeros::<i32>(usize_of(t * TOP_K)).unwrap();
        unsafe {
            ops.moe_permute(
                &gpu,
                dptr(&grouped, s),
                dptr(&row_of, s),
                pid,
                t,
                TOP_K,
                cap,
                rows,
            )
            .unwrap();
        }
        let host_row_of = s.clone_dtoh(&row_of).unwrap();
        let mut row_src = vec![-1i32; usize_of(rows)];
        for (i, &r) in host_row_of.iter().enumerate() {
            row_src[usize::try_from(r).unwrap()] = i32::try_from(i / usize_of(TOP_K)).unwrap();
        }
        let row_src = s.clone_htod(&row_src).unwrap();
        let (prow_of, prow_src) = (dptr(&row_of, s), dptr(&row_src, s));

        // Gather.
        let a = s.alloc_zeros::<u8>(usize_of(rows * HIDDEN)).unwrap();
        let asf = s
            .alloc_zeros::<i32>(usize_of(HIDDEN / 512 * rows4))
            .unwrap();
        let (pa, pasf) = (dptr(&a, s), dptr(&asf, s));
        let new = time_us(&gpu, iters, &head_start, &|| unsafe {
            ops.gather_quant_ue8m0(
                &gpu, pa, pasf, px, prow_of, t, TOP_K, rows, HIDDEN, rows4, cap,
            )
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
                rows4,
                cap
            )
            .unwrap();
        });
        report("gather", new, old);

        // SwiGLU.
        let gu = s
            .alloc_zeros::<u16>(usize_of(rows) * usize_of(2 * INTER))
            .unwrap();
        let q = s.alloc_zeros::<u8>(usize_of(rows * INTER)).unwrap();
        let qsf = s.alloc_zeros::<i32>(usize_of(INTER / 512 * rows4)).unwrap();
        let (pgu, pq, pqsf) = (dptr(&gu, s), dptr(&q, s), dptr(&qsf, s));
        let new = time_us(&gpu, iters, &head_start, &|| unsafe {
            ops.swiglu_quant_ue8m0(
                &gpu, pq, pqsf, pgu, prow_of, t, TOP_K, rows, INTER, rows4, cap,
            )
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
                rows4,
                cap
            )
            .unwrap();
        });
        report("swiglu", new, old);
    }
    println!();
    for line in json {
        println!("{line}");
    }
}
