//! Per-launch device time of DeepSeek-V4.1's quantised GEMMs in every form
//! (`exact`: the scalar reference; `banded`, `tolerance`: the tensor-core
//! forms of `ds_gemm_tc`), per row count, at a dense shape and the routed
//! experts' shape, on model-like synthetic operands. No checkpoint needed.
//!
//! ```text
//! ds_gemm_bench <kernels_dir> [--rows 1,2,4,...] [--iters N] [--forms exact,banded,tolerance]
//!               [--dense N,K] [--experts N,K] [--all-tiles | --tiles s8,t8,t32]
//!               [--only dense|experts] [--once]
//! ```
//!
//! Defaults: rows 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
//! 8192; the dense shape `wo_b` (5120 × 8192); the experts' fused gate and up
//! (4608 × 5120, 384 experts, 6 a token, runs aligned to 8 rows); 20
//! iterations, fewer where a launch is long (each measurement is held to
//! about two seconds). The tensor-core forms run the block shape the host
//! picks (`TokenTile::for_lin8`, `for_lin4`), or every shape with
//! `--all-tiles`.
//!
//! `--once` launches each (shape, rows, form, block shape) exactly once,
//! untimed, and prints a `launch` line naming it, in launch order: what a
//! profiler is run over (`scripts/ds_gpu_session.sh`'s `ds_gemm_profile`).
//!
//! Each measurement enqueues its launches back to back and divides the time
//! between two events around them. Reported per launch: microseconds,
//! TFLOP/s (two per multiply-add of the rows the product computes) and the
//! weight bytes it must read at least once (all of a dense weight; the
//! experts the layout reaches) per second. A table, then one JSON line per
//! measurement (`{"shape", "rows", "form", "tile", "us", "tflops",
//! "weight_tbps"}`).

#[path = "../tests/ds_gemm_data/mod.rs"]
mod ds_gemm_data;

use cudarc::driver::sys::CUevent_flags;
use ds_gemm_data::{Rng, activations, fp4_weights, fp8_weight, layout};
use eidola_engine_cuda::ds::abi::{Lin4Params, Lin8Params};
use eidola_engine_cuda::ds::gemm::{DsGemmTc, GemmForm, TcForm, TokenTile};
use eidola_engine_cuda::ds::gemm_ref::DsGemmRef;
use eidola_engine_cuda::launch::dptr;
use eidola_engine_cuda::{Gpu, KernelDir};

const GROUPS: u32 = 384;
const ACTIVE: u32 = 6;
const ROW_ALIGN: u32 = 8;

fn us(x: u32) -> usize {
    usize::try_from(x).unwrap()
}

/// Microseconds per launch: one warm launch, then launches until about two
/// seconds or `iters`, whichever comes first.
fn time_us(gpu: &Gpu, iters: u32, launch: &dyn Fn()) -> f64 {
    let ctx = gpu.context();
    let event = || {
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap()
    };
    let (start, end) = (event(), event());
    start.record(gpu.stream()).unwrap();
    launch();
    end.record(gpu.stream()).unwrap();
    let first_ms = f64::from(start.elapsed_ms(&end).unwrap());
    let n = if first_ms > 0.0 {
        (2000.0 / first_ms).clamp(1.0, f64::from(iters))
    } else {
        f64::from(iters)
    };
    let n = (1..=iters).find(|i| f64::from(*i) >= n).unwrap_or(iters);
    start.record(gpu.stream()).unwrap();
    for _ in 0..n {
        launch();
    }
    end.record(gpu.stream()).unwrap();
    f64::from(start.elapsed_ms(&end).unwrap()) * 1000.0 / f64::from(n)
}

struct Args {
    kernels: String,
    rows: Vec<u32>,
    iters: u32,
    forms: Vec<GemmForm>,
    dense: (u32, u32),
    experts: (u32, u32),
    all_tiles: bool,
    tiles: Option<Vec<TokenTile>>,
    only: Option<String>,
    once: bool,
}

fn parse() -> Args {
    let a: Vec<String> = std::env::args().collect();
    let Some(kernels) = a.get(1).cloned() else {
        eprintln!("usage: ds_gemm_bench <kernels_dir> [options]; see the module docs");
        std::process::exit(2);
    };
    let mut args = Args {
        kernels,
        rows: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        iters: 20,
        forms: GemmForm::ALL.to_vec(),
        dense: (5120, 8192),
        experts: (4608, 5120),
        all_tiles: false,
        tiles: None,
        only: None,
        once: false,
    };
    let pair = |s: &str| -> (u32, u32) {
        let v: Vec<u32> = s.split(',').map(|x| x.parse().unwrap()).collect();
        assert_eq!(v.len(), 2, "a shape is N,K");
        (v[0], v[1])
    };
    let mut i = 2;
    while i < a.len() {
        match a[i].as_str() {
            "--rows" => {
                args.rows = a[i + 1].split(',').map(|x| x.parse().unwrap()).collect();
                i += 1;
            }
            "--iters" => {
                args.iters = a[i + 1].parse().unwrap();
                i += 1;
            }
            "--forms" => {
                args.forms = a[i + 1]
                    .split(',')
                    .map(|f| {
                        *GemmForm::ALL
                            .iter()
                            .find(|g| g.name() == f)
                            .unwrap_or_else(|| panic!("unknown form {f}"))
                    })
                    .collect();
                i += 1;
            }
            "--dense" => {
                args.dense = pair(&a[i + 1]);
                i += 1;
            }
            "--experts" => {
                args.experts = pair(&a[i + 1]);
                i += 1;
            }
            "--all-tiles" => args.all_tiles = true,
            "--tiles" => {
                args.tiles = Some(
                    a[i + 1]
                        .split(',')
                        .map(|t| {
                            *TokenTile::ALL
                                .iter()
                                .find(|x| x.name() == t)
                                .unwrap_or_else(|| panic!("unknown block shape {t}"))
                        })
                        .collect(),
                );
                i += 1;
            }
            "--only" => {
                assert!(
                    a[i + 1] == "dense" || a[i + 1] == "experts",
                    "--only dense or experts"
                );
                args.only = Some(a[i + 1].clone());
                i += 1;
            }
            "--once" => args.once = true,
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    args
}

/// The (form, shape) pairs to time; `pick` is the host's choice for a
/// tensor-core form.
fn runs(args: &Args, pick: &dyn Fn(TcForm) -> TokenTile) -> Vec<(GemmForm, Option<TokenTile>)> {
    let mut out = Vec::new();
    for &form in &args.forms {
        match form.tc() {
            None => out.push((form, None)),
            Some(_) if args.tiles.is_some() => {
                for &tile in args.tiles.as_ref().unwrap() {
                    out.push((form, Some(tile)));
                }
            }
            Some(_) if args.all_tiles => {
                for tile in TokenTile::ALL {
                    out.push((form, Some(tile)));
                }
            }
            Some(f) => out.push((form, Some(pick(f)))),
        }
    }
    out
}

fn report(
    shape: &str,
    rows: u32,
    form: GemmForm,
    tile: Option<TokenTile>,
    t: f64,
    flops: f64,
    bytes: f64,
) {
    let tile = tile.map_or("-", TokenTile::name);
    let tflops = flops / t / 1e6;
    let tbps = bytes / t / 1e6;
    println!(
        "{shape:>8} {rows:>6} {:>10} {tile:>5} {t:>12.1} {tflops:>9.2} {tbps:>9.3}",
        form.name()
    );
    println!(
        "{{\"shape\":\"{shape}\",\"rows\":{rows},\"form\":\"{}\",\"tile\":\"{tile}\",\"us\":{t:.3},\"tflops\":{tflops:.4},\"weight_tbps\":{tbps:.4}}}",
        form.name()
    );
}

/// Time `launch` and report it, or with `--once` run it once and name it.
#[allow(clippy::too_many_arguments)]
fn measure(
    gpu: &Gpu,
    args: &Args,
    launches: &mut u32,
    shape: &str,
    rows: u32,
    form: GemmForm,
    tile: Option<TokenTile>,
    flops: f64,
    bytes: f64,
    launch: &dyn Fn(),
) {
    if args.once {
        launch();
        gpu.synchronize().unwrap();
        println!(
            "launch {launches} {shape} {rows} {} {}",
            form.name(),
            tile.map_or("-", TokenTile::name)
        );
        *launches += 1;
    } else {
        let t = time_us(gpu, args.iters, launch);
        report(shape, rows, form, tile, t, flops, bytes);
    }
}

fn main() {
    let args = parse();
    let mut launches = 0u32;
    let gpu = Gpu::open(0).expect("open device 0");
    let dir = KernelDir::new(&args.kernels);
    let reference = DsGemmRef::load(&gpu, &dir).expect("ds_reference");
    let tc = DsGemmTc::load(&gpu, &dir).expect("ds_gemm_tc");
    let s = gpu.stream();
    let max_rows = *args.rows.iter().max().unwrap();
    let mut g = Rng(0xbe9c);
    println!(
        "{:>8} {:>6} {:>10} {:>5} {:>12} {:>9} {:>9}",
        "shape", "rows", "form", "tile", "us", "TFLOP/s", "w TB/s"
    );

    let dense = args.only.as_deref() != Some("experts");
    let experts = args.only.as_deref() != Some("dense");

    // Dense: one weight, the activations of the largest row count.
    if dense {
        let (n, k) = args.dense;
        let (w, ws) = fp8_weight(&mut g, us(n), us(k), false);
        let (x, xs) = activations(&mut g, us(max_rows), us(k), false);
        let (dw, dws, dx, dxs) = (
            s.clone_htod(&w).unwrap(),
            s.clone_htod(&ws).unwrap(),
            s.clone_htod(&x).unwrap(),
            s.clone_htod(&xs).unwrap(),
        );
        let out = s.alloc_zeros::<u16>(us(max_rows) * us(n)).unwrap();
        for &rows in &args.rows {
            let p = Lin8Params {
                codes: dptr(&dx, s),
                act_scales: dptr(&dxs, s),
                w_codes: dptr(&dw, s),
                w_scales: dptr(&dws, s),
                out: dptr(&out, s),
                rows,
                n,
                k,
                reserved: 0,
            };
            let flops = 2.0 * f64::from(rows) * f64::from(n) * f64::from(k);
            let bytes = f64::from(n) * f64::from(k);
            for (form, tile) in runs(&args, &|f| TokenTile::for_lin8(f, rows)) {
                let launch = || {
                    // SAFETY: every array holds the largest row count's operands.
                    unsafe {
                        match (form.tc(), tile) {
                            (Some(f), Some(t)) => tc.lin8_tile(&gpu, f, t, &p),
                            _ => reference.lin8(&gpu, &p),
                        }
                    }
                    .unwrap();
                };
                measure(
                    &gpu,
                    &args,
                    &mut launches,
                    "dense",
                    rows,
                    form,
                    tile,
                    flops,
                    bytes,
                    &launch,
                );
            }
        }
        drop((dw, dws, dx, dxs, out));
    }

    // Experts: all 384 experts' weights, a layout per token count.
    if !experts {
        return;
    }
    let (n, k) = args.experts;
    let (w, ws) = fp4_weights(&mut g, us(GROUPS), us(n), us(k), false);
    let (dw, dws) = (s.clone_htod(&w).unwrap(), s.clone_htod(&ws).unwrap());
    let bound = max_rows * ACTIVE + GROUPS * ROW_ALIGN;
    let (x, xs) = activations(&mut g, us(bound), us(k), false);
    let (dx, dxs) = (s.clone_htod(&x).unwrap(), s.clone_htod(&xs).unwrap());
    let out = s.alloc_zeros::<u16>(us(bound) * us(n)).unwrap();
    for &tokens in &args.rows {
        let (ends, rows) = layout(&mut g, tokens, GROUPS, ACTIVE, ROW_ALIGN);
        // Experts the layout reaches: a run holding a row.
        let mut prev = 0u32;
        let mut reached = 0u32;
        for &e in &ends {
            if e > prev.next_multiple_of(ROW_ALIGN) {
                reached += 1;
            }
            prev = e;
        }
        let dends = s.clone_htod(&ends).unwrap();
        let p = Lin4Params {
            codes: dptr(&dx, s),
            act_scales: dptr(&dxs, s),
            w_packed: dptr(&dw, s),
            w_scales: dptr(&dws, s),
            group_end: dptr(&dends, s),
            out: dptr(&out, s),
            layout_rows: rows,
            n,
            k,
            groups: GROUPS,
            row_align: ROW_ALIGN,
            reserved: 0,
        };
        let pairs = tokens * ACTIVE;
        let flops = 2.0 * f64::from(pairs) * f64::from(n) * f64::from(k);
        let bytes = f64::from(reached) * f64::from(n) * f64::from(k) / 2.0;
        for (form, tile) in runs(&args, &|f| TokenTile::for_lin4(f, rows, GROUPS)) {
            let launch = || {
                // SAFETY: every array holds the largest layout's operands.
                unsafe {
                    match (form.tc(), tile) {
                        (Some(f), Some(t)) => tc.lin4_tile(&gpu, f, t, &p),
                        _ => reference.lin4(&gpu, &p),
                    }
                }
                .unwrap();
            };
            measure(
                &gpu,
                &args,
                &mut launches,
                "experts",
                tokens,
                form,
                tile,
                flops,
                bytes,
                &launch,
            );
        }
    }
}
