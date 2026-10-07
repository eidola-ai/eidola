//! The CUTLASS GEMMs launched with Rust-built parameters, against a host
//! reference over the same quantized inputs, on every image this device runs.

mod common;

use common::{Lcg, setup};
use eidola_engine_cuda::bf16;
use eidola_engine_cuda::gemm::{Gemm, GemmArgs, GemmKind};
use eidola_engine_cuda::launch::dptr;
use eidola_engine_model::numeric::fp8_e4m3_to_f32;

/// A finite e4m3 code (0x7f and 0xff are NaN).
fn fp8(rng: &mut Lcg) -> u8 {
    loop {
        let b = rng.next_u64() as u8;
        if b & 0x7f != 0x7f {
            return b;
        }
    }
}

#[test]
fn fp8_blockwise_matches_reference() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    for &(m, n, k) in &[
        (4u32, 256u32, 512u32),
        (128, 128, 128),
        (36, 13824, 4096),
        (1000, 4096, 2048),
        (8, 4096, 16384),
    ] {
        let mut rng = Lcg((m as u64) << 32 | n as u64 ^ k as u64);
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let (kb, nb) = (ku / 128, nu.div_ceil(128));
        let a: Vec<u8> = (0..mu * ku).map(|_| fp8(&mut rng)).collect();
        let b: Vec<u8> = (0..nu * ku).map(|_| fp8(&mut rng)).collect();
        // Scales: arbitrary positive f32, laid out [K/128][M] and [K/128][N/128].
        let sfa: Vec<f32> = (0..kb * mu)
            .map(|_| 0.01 + rng.f32().abs() * 0.02)
            .collect();
        let sfb: Vec<f32> = (0..kb * nb)
            .map(|_| 0.001 + rng.f32().abs() * 0.003)
            .collect();
        let (da, db) = (s.clone_htod(&a).unwrap(), s.clone_htod(&b).unwrap());
        let (dsfa, dsfb) = (s.clone_htod(&sfa).unwrap(), s.clone_htod(&sfb).unwrap());
        // Reference: per 128-wide K block, the exact f64 dot product, scaled.
        let af: Vec<f32> = a.iter().map(|&x| fp8_e4m3_to_f32(x)).collect();
        let bf: Vec<f32> = b.iter().map(|&x| fp8_e4m3_to_f32(x)).collect();
        let mut want = vec![0f64; mu * nu];
        let mut mag = vec![0f64; mu * nu];
        for i in 0..mu {
            for j in 0..nu {
                let (mut acc, mut abs) = (0f64, 0f64);
                for q in 0..kb {
                    let (mut dot, mut dabs) = (0f64, 0f64);
                    for t in q * 128..(q + 1) * 128 {
                        let p = af[i * ku + t] as f64 * bf[j * ku + t] as f64;
                        dot += p;
                        dabs += p.abs();
                    }
                    let scale = sfa[q * mu + i] as f64 * sfb[q * nb + j / 128] as f64;
                    acc += dot * scale;
                    abs += dabs * scale;
                }
                want[i * nu + j] = acc;
                mag[i * nu + j] = abs;
            }
        }
        for &arch in &su.archs {
            let (name, entry) = GemmKind::Fp8Blockwise.kernel();
            let module = su.module(name, arch);
            let gemm = Gemm::new(GemmKind::Fp8Blockwise, module.kernel(entry).unwrap()).unwrap();
            let dd = s.alloc_zeros::<u16>(mu * nu).unwrap();
            let args = GemmArgs {
                m,
                n,
                k,
                a: dptr(&da, s),
                b: dptr(&db, s),
                d: dptr(&dd, s),
                sfa: dptr(&dsfa, s),
                sfb: dptr(&dsfb, s),
                alpha: 1.0,
            };
            unsafe { gemm.launch(gpu, &args) }.unwrap();
            let got = s.clone_dtoh(&dd).unwrap();
            let mut worst = 0f64;
            for (idx, (&g, &w)) in got.iter().zip(&want).enumerate() {
                let g = bf16::to_f32(g) as f64;
                // BF16 output: half an ulp of rounding plus f32 accumulation.
                let tol = w.abs() / 256.0 + mag[idx] * ku as f64 * f32::EPSILON as f64;
                let err = (g - w).abs();
                worst = worst.max(err / (w.abs() + 1e-6));
                assert!(err <= tol, "{arch:?} ({m},{n},{k}) at {idx}: {g} vs {w}");
            }
            eprintln!("{arch:?} fp8 ({m},{n},{k}): worst relative error {worst:.2e}");
        }
    }
}

#[test]
fn bf16_matches_reference() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    for &(m, n, k, alpha) in &[
        (1u32, 4096u32, 8192u32, 0.707f32),
        (37, 4096, 8192, 1.0),
        (5, 152_576, 4096, 1.0),
        (64, 128, 64, 0.5),
        (300, 1024, 4096, 1.0),
    ] {
        let mut rng = Lcg((m as u64) << 40 | (n as u64) << 8 | k as u64);
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let a: Vec<u16> = (0..mu * ku).map(|_| bf16::from_f32(rng.f32())).collect();
        let b: Vec<u16> = (0..nu * ku)
            .map(|_| bf16::from_f32(rng.f32() * 0.05))
            .collect();
        let (da, db) = (s.clone_htod(&a).unwrap(), s.clone_htod(&b).unwrap());
        let af: Vec<f32> = a.iter().map(|&x| bf16::to_f32(x)).collect();
        let bf: Vec<f32> = b.iter().map(|&x| bf16::to_f32(x)).collect();
        let mut want = vec![0f64; mu * nu];
        let mut mag = vec![0f64; mu * nu];
        for i in 0..mu {
            for j in 0..nu {
                let (mut acc, mut abs) = (0f64, 0f64);
                for t in 0..ku {
                    let p = af[i * ku + t] as f64 * bf[j * ku + t] as f64;
                    acc += p;
                    abs += p.abs();
                }
                want[i * nu + j] = acc * alpha as f64;
                mag[i * nu + j] = abs * alpha as f64;
            }
        }
        for &arch in &su.archs {
            let (name, entry) = GemmKind::Bf16.kernel();
            let module = su.module(name, arch);
            let gemm = Gemm::new(GemmKind::Bf16, module.kernel(entry).unwrap()).unwrap();
            let dd = s.alloc_zeros::<f32>(mu * nu).unwrap();
            let args = GemmArgs {
                m,
                n,
                k,
                a: dptr(&da, s),
                b: dptr(&db, s),
                d: dptr(&dd, s),
                sfa: 0,
                sfb: 0,
                alpha,
            };
            unsafe { gemm.launch(gpu, &args) }.unwrap();
            let got = s.clone_dtoh(&dd).unwrap();
            let mut worst = 0f64;
            for (idx, &g) in got.iter().enumerate() {
                // f32 accumulation: error bounded by K ulp of the absolute sum.
                let err = (g as f64 - want[idx]).abs();
                let bound = mag[idx] * ku as f64 * f32::EPSILON as f64 + 1e-30;
                worst = worst.max(err / (mag[idx] * f32::EPSILON as f64 + 1e-30));
                assert!(
                    err <= bound,
                    "{arch:?} ({m},{n},{k}) at {idx}: {g} vs {}",
                    want[idx]
                );
            }
            eprintln!(
                "{arch:?} bf16 ({m},{n},{k}): worst error {worst:.1} ulp of the absolute sum"
            );
        }
    }
}
