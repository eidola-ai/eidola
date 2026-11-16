//! The DeepSeek-V4.1 reference products and head (`ds_reference`, `ds_head`)
//! against the model crate's `lin8`, `lin4`, `linear_f32` and the grouped
//! projection of `attention_output`, bit for bit, on every image this device
//! runs.
//!
//! Without a device: the vector file the kernels crate's host check runs
//! against is current, and the Engram chain's host side is built and held to
//! `engram::inject`. With one (`device_suite`): the seeded cases at small
//! shapes and the released widths (the quantiser feeding `lin8` on the
//! device); a row launched alone against the same row in its batch; the
//! Engram path as the executor runs it (gather, quantiser, `lin8`, injection)
//! against `engram::inject`; and each kernel at the released model's largest
//! shapes (the whole vocabulary for the head and the Markov head), checked
//! and then timed. The suite writes `ds_gemm_ref.json`.

mod common;
mod deepseek_cases;

use std::sync::Arc;

use common::setup;
use deepseek_cases::device::{How, Kernels, ROW_KERNELS, check};
use deepseek_cases::suite::Summary;
use deepseek_cases::{GEMM_REF_KERNELS, Rng, bits_of, inject_after_projection, sweep, vals};
use eidola_engine_cuda::Gpu;
use eidola_engine_cuda::deepseek::abi;
use eidola_engine_cuda::launch::dptr;
use eidola_engine_model::deepseek::engram::{EngramWeights, SparseEngramTable, inject};
use eidola_engine_model::deepseek::linear::{Fp8Linear, lin8};
use eidola_engine_model::tensor::Matrix;

#[test]
fn vectors_are_current() {
    deepseek_cases::check_or_update("ds_gemm_ref.vectors", &GEMM_REF_KERNELS);
}

/// One Engram module over a few rows: the stored table, the hashed rows of
/// each position, the projection, the gate weights and the streams, with the
/// streams `engram::inject` leaves.
struct EngramChain {
    table_rows: usize,
    width: usize,
    lookups: usize,
    rows: usize,
    copies: usize,
    dim: usize,
    codes: Vec<u8>,
    scales: Vec<u8>,
    ids: Vec<u32>,
    w_codes: Vec<u8>,
    w_scales: Vec<u8>,
    gate: Vec<f32>,
    x: Vec<u16>,
    out: Vec<u16>,
}

fn engram_chain(g: &mut Rng, released: bool) -> EngramChain {
    let pick = |g: &mut Rng, v: usize, bound: usize| if released { v } else { 1 + g.below(bound) };
    let table_rows = 4 + g.below(40);
    let width = 32 * pick(g, 8, 2);
    let lookups = pick(g, 24, 4);
    let rows = pick(g, 8, 3);
    let copies = pick(g, 4, 4);
    let dim = pick(g, 5120, 80);
    let n = dim * (copies + 1);
    let k = lookups * width;
    let codes = g.codes(table_rows * width);
    let scales = g.scales(table_rows * width / 32);
    let ids: Vec<u32> = (0..rows * lookups)
        .map(|_| u32::try_from(g.below(table_rows)).unwrap())
        .collect();
    let w_codes = g.codes(n * k);
    let w_scales = g.scales(n.div_ceil(32) * (k / 32));
    let gate = g.f32s(copies * dim, 118, 8);
    let x: Vec<u16> = (0..rows).flat_map(|_| g.bf16s(copies * dim)).collect();
    let all: Vec<u64> = (0..table_rows as u64).collect();
    let weights = EngramWeights {
        table: Arc::new(
            SparseEngramTable::from_stored(table_rows as u64, width, &all, &codes, &scales)
                .unwrap(),
        ),
        wkv: Fp8Linear::from_bytes(n, k, &w_codes, &w_scales),
        gate_weight: Matrix::from_vec(copies, dim, gate.clone()),
    };
    let mut out = Vec::new();
    for r in 0..rows {
        let mut xr = vals(&x[r * copies * dim..(r + 1) * copies * dim]);
        let row_ids: Vec<u64> = ids[r * lookups..(r + 1) * lookups]
            .iter()
            .map(|id| u64::from(*id))
            .collect();
        inject(&weights, &row_ids, &mut xr, copies, deepseek_cases::EPS).unwrap();
        // The same through the projection's output: what the injection
        // kernel's seeded cases use as their reference.
        let mut gathered = vec![0.0f32; k];
        for (id, row) in row_ids.iter().zip(gathered.chunks_exact_mut(width)) {
            weights.table.read_row(*id, row).unwrap();
        }
        let mut kv = vec![0.0f32; n];
        lin8(&gathered, &weights.wkv, &mut kv);
        let mut again = vals(&x[r * copies * dim..(r + 1) * copies * dim]);
        inject_after_projection(&mut again, &kv, &gate, copies, deepseek_cases::EPS);
        assert!(
            xr.iter()
                .zip(&again)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "the injection's host form is not engram::inject"
        );
        out.extend(bits_of(&xr));
    }
    EngramChain {
        table_rows,
        width,
        lookups,
        rows,
        copies,
        dim,
        codes,
        scales,
        ids,
        w_codes,
        w_scales,
        gate,
        x,
        out,
    }
}

/// The chain's host side, and with it the injection cases' reference, against
/// `engram::inject`.
#[test]
fn engram_chain_is_built_without_a_device() {
    let mut g = Rng(0xe9);
    for _ in 0..64 {
        let c = engram_chain(&mut g, false);
        assert_eq!(c.out.len(), c.rows * c.copies * c.dim);
    }
}

/// Gather, quantise, project and inject on the device, as a layer does.
fn run_engram_chain(gpu: &Gpu, k: &Kernels, c: &EngramChain) {
    let s = gpu.stream();
    let u = |x: usize| u32::try_from(x).unwrap();
    let (rows, kk, n) = (u(c.rows), u(c.lookups * c.width), u(c.dim * (c.copies + 1)));
    let codes = s.clone_htod(&c.codes).unwrap();
    let scales = s.clone_htod(&c.scales).unwrap();
    let ids = s.clone_htod(&c.ids).unwrap();
    let w_codes = s.clone_htod(&c.w_codes).unwrap();
    let w_scales = s.clone_htod(&c.w_scales).unwrap();
    let gate = s.clone_htod(&c.gate).unwrap();
    let x = s.clone_htod(&c.x).unwrap();
    let row_codes = s.alloc_zeros::<u8>(c.rows * c.lookups * c.width).unwrap();
    let row_scales = s
        .alloc_zeros::<u8>(c.rows * c.lookups * c.width / 32)
        .unwrap();
    let act_codes = s.alloc_zeros::<u8>(c.rows * c.lookups * c.width).unwrap();
    let act_scales = s
        .alloc_zeros::<u8>(c.rows * c.lookups * c.width / 32)
        .unwrap();
    let kv = s
        .alloc_zeros::<u16>(c.rows * c.dim * (c.copies + 1))
        .unwrap();
    let status = s.alloc_zeros::<u32>(1).unwrap();
    // SAFETY: every array is sized for the counts passed with it.
    unsafe {
        k.ops
            .engram_gather(
                gpu,
                &abi::EngramGatherParams {
                    codes: dptr(&codes, s),
                    scales: dptr(&scales, s),
                    ids: dptr(&ids, s),
                    out_codes: dptr(&row_codes, s),
                    out_scales: dptr(&row_scales, s),
                    status: dptr(&status, s),
                    table_rows: c.table_rows as u64,
                    rows,
                    ids_per_row: u(c.lookups),
                    width: u(c.width),
                    reserved: 0,
                },
            )
            .unwrap();
        k.ops
            .quant_fp8(
                gpu,
                &abi::QuantFp8Params {
                    x: dptr(&row_codes, s),
                    x_scales: dptr(&row_scales, s),
                    codes: dptr(&act_codes, s),
                    scales: dptr(&act_scales, s),
                    rows,
                    width: kk,
                    x_format: abi::X_FP8,
                    reserved: 0,
                },
            )
            .unwrap();
        k.gemm
            .lin8(
                gpu,
                &abi::Lin8Params {
                    codes: dptr(&act_codes, s),
                    act_scales: dptr(&act_scales, s),
                    w_codes: dptr(&w_codes, s),
                    w_scales: dptr(&w_scales, s),
                    out: dptr(&kv, s),
                    rows,
                    n,
                    k: kk,
                    reserved: 0,
                },
            )
            .unwrap();
        // In place, as the executor updates the streams.
        k.ops
            .engram_inject(
                gpu,
                &abi::EngramInjectParams {
                    x: dptr(&x, s),
                    kv: dptr(&kv, s),
                    gate_weight: dptr(&gate, s),
                    out: dptr(&x, s),
                    rows,
                    dim: u(c.dim),
                    copies: u(c.copies),
                    norm_eps: deepseek_cases::EPS,
                },
            )
            .unwrap();
    }
    gpu.synchronize().unwrap();
    assert_eq!(s.clone_dtoh(&status).unwrap(), [0]);
    let got = s.clone_dtoh(&x).unwrap();
    if let Some(at) = got.iter().zip(&c.out).position(|(a, b)| a != b) {
        panic!(
            "the Engram chain differs from engram::inject at element {at} ({} rows, {} lookups \
             of {}, {} copies of {})",
            c.rows, c.lookups, c.width, c.copies, c.dim
        );
    }
}

#[test]
fn device_suite() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let mut summary = Summary::new("ds_gemm_ref", gpu);
    for (i, &arch) in su.archs.iter().enumerate() {
        let k = Kernels::load(gpu, &su.dir, arch);
        for kernel in GEMM_REF_KERNELS {
            summary.check(format!("{arch:?}/cases/{kernel}"), || {
                for wide in [false, true] {
                    for (j, case) in sweep(kernel, wide).iter().enumerate() {
                        check(
                            gpu,
                            &k,
                            case,
                            How::default(),
                            &format!("case {j} wide {wide}"),
                        );
                    }
                }
            });
            if ROW_KERNELS.contains(&kernel) {
                summary.check(format!("{arch:?}/row_alone/{kernel}"), || {
                    for wide in [false, true] {
                        for (j, case) in sweep(kernel, wide).iter().take(32).enumerate() {
                            for row in 0..case.u("rows") as usize {
                                let how = How {
                                    row: Some(row),
                                    ..How::default()
                                };
                                check(gpu, &k, case, how, &format!("row {row} alone, case {j}"));
                            }
                        }
                    }
                });
            }
        }
        summary.check(format!("{arch:?}/engram_chain"), || {
            let mut g = Rng(0xe9);
            for _ in 0..64 {
                run_engram_chain(gpu, &k, &engram_chain(&mut g, false));
            }
        });
        if i > 0 {
            continue;
        }
        summary.check(format!("{arch:?}/engram_chain/released"), || {
            run_engram_chain(gpu, &k, &engram_chain(&mut Rng(0xe9), true));
        });
        for kernel in GEMM_REF_KERNELS {
            let iters = if kernel == "head" { 5 } else { 20 };
            summary.released(gpu, &k, kernel, iters);
        }
        summary.released(gpu, &k, "markov", 20);
    }
    summary.finish();
}

/// Every kernel's case at the released model's largest shapes, with its
/// reference outputs: what the device suite checks and times.
#[test]
#[ignore = "the largest shapes: run with --release (minutes, about 8 GB)"]
fn released_cases_are_built() {
    for kernel in GEMM_REF_KERNELS.into_iter().chain(["markov"]) {
        let case =
            deepseek_cases::build(kernel, deepseek_cases::Regime::Released, &mut Rng(0xbe9c));
        assert!(!case.outputs.is_empty(), "{kernel}");
    }
    engram_chain(&mut Rng(0xe9), true);
}
