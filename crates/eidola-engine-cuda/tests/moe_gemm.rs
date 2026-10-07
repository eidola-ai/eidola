//! DeepGEMM's FP8 × MXFP4 grouped GEMM, launched with Rust-built descriptors,
//! against a host reference over the same quantized operands, in both grouped
//! layouts, on every image this device runs.

mod common;

use common::{Lcg, setup};
use eidola_engine_cuda::bf16;
use eidola_engine_cuda::launch::dptr;
use eidola_engine_cuda::moe_gemm::{BLOCK_M, GROUPS, MoeGemm, MoeGemmArgs, MoeLayout, MoeProj};
use eidola_engine_model::numeric::fp8_e4m3_to_f32;

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn e2m1(code: u8) -> f32 {
    let v = E2M1[(code & 7) as usize];
    if code & 8 != 0 { -v } else { v }
}

fn pow2(e: u8) -> f32 {
    2f32.powi(e as i32 - 127)
}

/// One group's weights: `[N][K/2]` packed FP4 and `[N][K/32]` E8M0 scales (the
/// checkpoint's layout), plus the dequantized values.
struct Expert {
    packed: Vec<u8>,
    scales: Vec<u8>,
    values: Vec<f32>,
}

fn expert(rng: &mut Lcg, n: usize, k: usize) -> Expert {
    let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next_u64() as u8).collect();
    let scales: Vec<u8> = (0..n * k / 32).map(|_| 118 + rng.below(6) as u8).collect();
    let mut values = vec![0f32; n * k];
    for r in 0..n {
        for c in 0..k {
            let byte = packed[r * k / 2 + c / 2];
            let code = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
            values[r * k + c] = e2m1(code) * pow2(scales[r * k / 32 + c / 32]);
        }
    }
    Expert {
        packed,
        scales,
        values,
    }
}

/// SFB words for one group: `[K/128][N]`, byte `j` of word `(q, n)` the scale
/// of K group `4q + j` of row `n`.
fn sfb_words(e: &Expert, n: usize, k: usize) -> Vec<i32> {
    let mut out = vec![0i32; k / 128 * n];
    for r in 0..n {
        for q in 0..k / 128 {
            let s = &e.scales[r * k / 32 + 4 * q..r * k / 32 + 4 * q + 4];
            out[q * n + r] = i32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        }
    }
    out
}

/// Activation rows quantized as the engine does (FP8 codes with one E8M0
/// scale per 128 of K): codes, scales `[row][K/128]`, dequantized values.
fn activations(rng: &mut Lcg, rows: usize, k: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    let mut codes = vec![0u8; rows * k];
    let mut scales = vec![0u8; rows * k / 128];
    let mut values = vec![0f32; rows * k];
    for r in 0..rows {
        for c in 0..k {
            let code = loop {
                let b = rng.next_u64() as u8;
                if b & 0x7f != 0x7f {
                    break b;
                }
            };
            codes[r * k + c] = code;
        }
        for q in 0..k / 128 {
            scales[r * k / 128 + q] = 120 + rng.below(4) as u8;
        }
        for c in 0..k {
            values[r * k + c] =
                fp8_e4m3_to_f32(codes[r * k + c]) * pow2(scales[r * k / 128 + c / 128]);
        }
    }
    (codes, scales, values)
}

/// SFA words `[K/512][rows']` (rows' = rows rounded up to 4) for one block of rows.
fn sfa_words(scales: &[u8], rows: usize, k: usize) -> Vec<i32> {
    let r4 = rows.div_ceil(4) * 4;
    let mut out = vec![0i32; k / 512 * r4];
    for r in 0..rows {
        for w in 0..k / 512 {
            let s = &scales[r * k / 128 + 4 * w..r * k / 128 + 4 * w + 4];
            out[w * r4 + r] = i32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        }
    }
    out
}

/// `out[i][j] = Σ_k a[i][k] · b[j][k]` in f64, rows in parallel.
fn reference(a: &[f32], b: &[f32], rows: usize, n: usize, k: usize) -> Vec<f64> {
    let mut out = vec![0f64; rows * n];
    let threads = std::thread::available_parallelism().map_or(8, |p| p.get());
    let chunk = rows.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (ci, o) in out.chunks_mut(chunk * n).enumerate() {
            s.spawn(move || {
                for (ri, orow) in o.chunks_mut(n).enumerate() {
                    let i = ci * chunk + ri;
                    let ar = &a[i * k..(i + 1) * k];
                    for (j, v) in orow.iter_mut().enumerate() {
                        let br = &b[j * k..(j + 1) * k];
                        *v = ar.iter().zip(br).map(|(&x, &y)| x as f64 * y as f64).sum();
                    }
                }
            });
        }
    });
    out
}

fn check(got: &[u16], want: &[f64], what: &str) -> f64 {
    let mut worst = 0f64;
    let scale = want.iter().fold(0f64, |m, &w| m.max(w.abs()));
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let g = bf16::to_f32(g) as f64;
        let err = (g - w).abs();
        // BF16 output rounding plus f32 accumulation across K.
        let tol = w.abs() / 256.0 + scale * 1e-5;
        worst = worst.max(err / (w.abs() + scale * 1e-5));
        assert!(err <= tol, "{what} at {i}: {g} vs {w}");
    }
    worst
}

fn run(proj: MoeProj, layout: MoeLayout) {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    let (n, k) = (proj.n() as usize, proj.k() as usize);
    let g_all = GROUPS as usize;
    let mut rng = Lcg(91 + k as u64);
    // Active groups and their row counts (one empty group in the masked case).
    let active: Vec<(usize, usize)> = vec![(3, 130), (17, 128), (200, 5), (255, 1), (40, 0)];
    let experts: Vec<Expert> = active.iter().map(|_| expert(&mut rng, n, k)).collect();

    // B and SFB for all groups; only the active ones hold data.
    let mut db = s.alloc_zeros::<u8>(g_all * n * k / 2).unwrap();
    let mut dsfb = s.alloc_zeros::<i32>(g_all * (k / 128) * n).unwrap();
    for ((g, _), e) in active.iter().zip(&experts) {
        let mut v = db.slice_mut(g * n * k / 2..(g + 1) * n * k / 2);
        s.memcpy_htod(&e.packed, &mut v).unwrap();
        let words = sfb_words(e, n, k);
        let mut v = dsfb.slice_mut(g * (k / 128) * n..(g + 1) * (k / 128) * n);
        s.memcpy_htod(&words, &mut v).unwrap();
    }

    // Rows per group, quantized.
    let rows: Vec<(Vec<u8>, Vec<u8>, Vec<f32>)> = active
        .iter()
        .map(|&(_, r)| activations(&mut rng, r, k))
        .collect();
    let wants: Vec<Vec<f64>> = rows
        .iter()
        .zip(&experts)
        .zip(&active)
        .map(|(((_, _, a), e), &(_, r))| reference(a, &e.values, r, n, k))
        .collect();

    // Lay A, SFA, D and the grouped layout out.
    let (m, a, sfa, layout_vec, offsets): (usize, Vec<u8>, Vec<i32>, Vec<i32>, Vec<usize>) =
        match layout {
            MoeLayout::Contiguous => {
                let mut offsets = Vec::new();
                let mut total = 0usize;
                for &(_, r) in &active {
                    offsets.push(total);
                    total += r.div_ceil(BLOCK_M as usize) * BLOCK_M as usize;
                }
                let mut a = vec![0u8; total * k];
                let mut scales = vec![127u8; total * k / 128];
                let mut gl = vec![-1i32; total];
                for (((g, r), off), (codes, sc, _)) in active.iter().zip(&offsets).zip(&rows) {
                    a[off * k..(off + r) * k].copy_from_slice(codes);
                    scales[off * k / 128..(off + r) * k / 128].copy_from_slice(sc);
                    for row in 0..*r {
                        gl[off + row] = *g as i32;
                    }
                }
                let sfa = sfa_words(&scales, total, k);
                (total, a, sfa, gl, offsets)
            }
            MoeLayout::Masked => {
                let cap = 2 * BLOCK_M as usize;
                let r4 = cap.div_ceil(4) * 4;
                let mut a = vec![0u8; g_all * cap * k];
                let mut sfa = vec![0i32; g_all * (k / 512) * r4];
                let mut masked = vec![0i32; g_all];
                let mut offsets = Vec::new();
                for ((g, r), (codes, sc, _)) in active.iter().zip(&rows) {
                    offsets.push(g * cap);
                    a[g * cap * k..(g * cap + r) * k].copy_from_slice(codes);
                    let mut padded = sc.clone();
                    padded.resize(cap * k / 128, 127);
                    let words = sfa_words(&padded, cap, k);
                    sfa[g * (k / 512) * r4..(g + 1) * (k / 512) * r4].copy_from_slice(&words);
                    masked[*g] = *r as i32;
                }
                (cap, a, sfa, masked, offsets)
            }
        };
    let da = s.clone_htod(&a).unwrap();
    let dsfa = s.clone_htod(&sfa).unwrap();
    let dgl = s.clone_htod(&layout_vec).unwrap();
    let d_rows = match layout {
        MoeLayout::Contiguous => m,
        MoeLayout::Masked => m * g_all,
    };
    for &arch in &su.archs {
        let gemm = MoeGemm::from_module(su.module("deepgemm_fp8_fp4_grouped", arch)).unwrap();
        let dd = s.alloc_zeros::<u16>(d_rows * n).unwrap();
        let args = MoeGemmArgs {
            layout,
            proj,
            m: m as u32,
            grouped_layout: dptr(&dgl, s),
            a: dptr(&da, s),
            sfa: dptr(&dsfa, s),
            b: dptr(&db, s),
            sfb: dptr(&dsfb, s),
            d: dptr(&dd, s),
        };
        unsafe { gemm.launch(gpu, &args) }.unwrap();
        let got = s.clone_dtoh(&dd).unwrap();
        for (i, (&(g, r), off)) in active.iter().zip(&offsets).enumerate() {
            let w = check(
                &got[off * n..(off + r) * n],
                &wants[i],
                &format!("{arch:?} {proj:?} {layout:?} group {g}"),
            );
            eprintln!(
                "{arch:?} {proj:?} {layout:?} group {g} ({r} rows): worst relative error {w:.2e}"
            );
        }
    }
}

#[test]
fn contiguous_gate_up() {
    run(MoeProj::GateUp, MoeLayout::Contiguous);
}

#[test]
fn contiguous_down() {
    run(MoeProj::Down, MoeLayout::Contiguous);
}

#[test]
fn masked_gate_up() {
    run(MoeProj::GateUp, MoeLayout::Masked);
}

#[test]
fn masked_down() {
    run(MoeProj::Down, MoeLayout::Masked);
}
