//! The dense path's f32-scale kernels against their one-warp-per-group
//! reference forms (`engine_ops_reference.cu`) bit for bit: the activation
//! quantization (`eidola_quant_fp8_f32scale`) and the dense SwiGLU
//! (`eidola_swiglu_quant_fp8_f32scale`), every code and every scale word,
//! including the words of `m_pad`'s padding rows, which neither writes, on
//! every image this device runs, from one row to a full prefill step.

mod common;

use common::{Lcg, setup};
use eidola_engine_cuda::bf16;
use eidola_engine_cuda::engine_ops::{EngineOps, SWIGLU_MAX_WARPS, SWIGLU_PIECE};
use eidola_engine_cuda::launch::dptr;

/// Flash's hidden size and dense intermediate.
const HIDDEN: usize = 4096;
const DENSE_INTER: usize = 16_384;

/// (rows, K) of the quantization: the hidden size from one row to a full
/// step, and K of 640 (five groups: a block whose last three warps have no
/// group) and 128 (one).
const QUANT_CASES: [(usize, usize); 10] = [
    (1, HIDDEN),
    (2, HIDDEN),
    (7, HIDDEN),
    (64, HIDDEN),
    (129, HIDDEN),
    (513, HIDDEN),
    (2048, HIDDEN),
    (8192, HIDDEN),
    (33, 640),
    (300, 128),
];

/// (rows, intermediate) of the SwiGLU: the dense intermediate from one row
/// (64 items, fewer warps than a block row) to 2,048 rows (131,072 items,
/// each warp walking about 28), and narrow rows at counts whose items are not
/// a multiple of the grid's warps (3 pieces a row, one piece a row).
const SWIGLU_CASES: [(usize, usize); 9] = [
    (1, DENSE_INTER),
    (2, DENSE_INTER),
    (7, DENSE_INTER),
    (64, DENSE_INTER),
    (129, DENSE_INTER),
    (513, DENSE_INTER),
    (2048, DENSE_INTER),
    (8192, 768),
    (5001, 256),
];

const SENTINEL_BYTE: u8 = 0xa5;
const SENTINEL_WORD: u32 = 0x7fc0_5a5a;

fn u32_of(x: usize) -> u32 {
    u32::try_from(x).unwrap()
}

/// A value of magnitude around 2^`e` with a random sign and mantissa.
fn scaled(rng: &mut Lcg, e: i32) -> f32 {
    rng.f32() * 2f32.powi(e)
}

/// Rows to quantize: per 128-group magnitudes from 2^-140 (subnormal) to
/// 2^127, all-zero groups, single spikes, and groups holding an infinity or
/// NaN.
fn quant_input(rng: &mut Lcg, rows: usize, k: usize) -> Vec<f32> {
    let mut x = vec![0f32; rows * k];
    for (gi, group) in x.chunks_mut(128).enumerate() {
        match gi % 11 {
            0 => {}
            1 => group[usize::try_from(rng.below(128)).unwrap()] = scaled(rng, 10),
            2 => {
                for v in group.iter_mut() {
                    *v = scaled(rng, 0);
                }
                group[5] = f32::INFINITY;
            }
            3 => {
                for v in group.iter_mut() {
                    *v = scaled(rng, 3);
                }
                group[77] = f32::NAN;
            }
            _ => {
                let e = i32::try_from(rng.below(268)).unwrap() - 140;
                for v in group.iter_mut() {
                    *v = scaled(rng, e);
                }
            }
        }
    }
    x
}

/// Gate/up rows of the dense gate/up GEMM's output: magnitudes from 2^-8 to
/// 2^8 per row (exp overflows and underflows in SiLU), zero rows, and BF16
/// infinities and NaN.
fn swiglu_input(rng: &mut Lcg, rows: usize, inter: usize) -> Vec<u16> {
    let mut gu = vec![0u16; rows * 2 * inter];
    for (r, row) in gu.chunks_mut(2 * inter).enumerate() {
        if r % 11 == 3 {
            continue;
        }
        let e = i32::try_from(rng.below(17)).unwrap() - 8;
        for v in row.iter_mut() {
            *v = bf16::from_f32(scaled(rng, e));
        }
        if r % 13 == 5 {
            row[7] = bf16::from_f32(f32::INFINITY);
            row[inter + 200] = bf16::from_f32(f32::NEG_INFINITY);
            row[inter - 1] = bf16::from_f32(f32::NAN);
        }
    }
    gu
}

/// Every case's inputs are built, sized as the launches read them, and the
/// largest SwiGLU case gives every warp of its grid several items, without a
/// device.
#[test]
fn cases_are_built_without_a_device() {
    let mut rng = Lcg(1);
    for &(rows, k) in &QUANT_CASES[..3] {
        assert_eq!(quant_input(&mut rng, rows, k).len(), rows * k);
    }
    for &(rows, inter) in &SWIGLU_CASES[..3] {
        assert_eq!(swiglu_input(&mut rng, rows, inter).len(), rows * 2 * inter);
    }
    for &(_, inter) in &SWIGLU_CASES {
        assert!(inter.is_multiple_of(SWIGLU_PIECE as usize));
    }
    assert!(
        SWIGLU_CASES
            .iter()
            .any(|&(rows, inter)| rows * inter / SWIGLU_PIECE as usize
                > 4 * SWIGLU_MAX_WARPS as usize),
        "a case where every warp walks several items"
    );
}

/// The f32-scale quantization: every code of every row and every scale word
/// (padding rows' words left as they were) exactly as the reference.
#[test]
fn quant_matches_reference_bit_for_bit() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    for &arch in &su.archs {
        let ops = EngineOps::from_module(su.module("engine_ops", arch)).unwrap();
        let reference = su
            .module("engine_ops_reference", arch)
            .kernel("eidola_reference_quant_fp8_f32scale")
            .unwrap();
        for &(rows, k) in &QUANT_CASES {
            let mut rng = Lcg(0x9a0 ^ (rows as u64) << 8 ^ k as u64);
            let m_pad = rows.div_ceil(4) * 4 + 4;
            let dx = s.clone_htod(&quant_input(&mut rng, rows, k)).unwrap();
            let mut outs = Vec::new();
            for new in [true, false] {
                let q = s.clone_htod(&vec![SENTINEL_BYTE; rows * k]).unwrap();
                let sf = s.clone_htod(&vec![SENTINEL_WORD; k / 128 * m_pad]).unwrap();
                unsafe {
                    if new {
                        ops.quant_fp8(
                            gpu,
                            dptr(&q, s),
                            dptr(&sf, s),
                            dptr(&dx, s),
                            u32_of(rows),
                            u32_of(k),
                            u32_of(m_pad),
                        )
                        .unwrap();
                    } else {
                        eidola_engine_cuda::launch!(
                            gpu,
                            reference,
                            [u32_of(rows), u32_of(k / 128), 1],
                            dptr(&q, s),
                            dptr(&sf, s),
                            dptr(&dx, s),
                            u32_of(k),
                            u32_of(m_pad)
                        )
                        .unwrap();
                    }
                }
                outs.push((s.clone_dtoh(&q).unwrap(), s.clone_dtoh(&sf).unwrap()));
            }
            compare(&format!("quant {arch:?} {rows}x{k}"), k, m_pad, &outs);
        }
    }
}

/// The dense SwiGLU: every code and scale word exactly as the reference.
#[test]
fn dense_swiglu_matches_reference_bit_for_bit() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    for &arch in &su.archs {
        let ops = EngineOps::from_module(su.module("engine_ops", arch)).unwrap();
        let reference = su
            .module("engine_ops_reference", arch)
            .kernel("eidola_reference_swiglu_quant_fp8_f32scale")
            .unwrap();
        for &(rows, inter) in &SWIGLU_CASES {
            let mut rng = Lcg(0x5f1 ^ (rows as u64) << 8 ^ inter as u64);
            let m_pad = rows.div_ceil(4) * 4 + 4;
            let dgu = s.clone_htod(&swiglu_input(&mut rng, rows, inter)).unwrap();
            let mut outs = Vec::new();
            for new in [true, false] {
                let q = s.clone_htod(&vec![SENTINEL_BYTE; rows * inter]).unwrap();
                let sf = s
                    .clone_htod(&vec![SENTINEL_WORD; inter / 128 * m_pad])
                    .unwrap();
                unsafe {
                    if new {
                        ops.swiglu_quant_f32scale(
                            gpu,
                            dptr(&q, s),
                            dptr(&sf, s),
                            dptr(&dgu, s),
                            u32_of(rows),
                            u32_of(inter),
                            u32_of(m_pad),
                        )
                        .unwrap();
                    } else {
                        eidola_engine_cuda::launch!(
                            gpu,
                            reference,
                            [u32_of(rows), u32_of(inter / 128), 1],
                            dptr(&q, s),
                            dptr(&sf, s),
                            dptr(&dgu, s),
                            u32_of(inter),
                            u32_of(m_pad)
                        )
                        .unwrap();
                    }
                }
                outs.push((s.clone_dtoh(&q).unwrap(), s.clone_dtoh(&sf).unwrap()));
            }
            compare(
                &format!("dense swiglu {arch:?} {rows}x{inter}"),
                inter,
                m_pad,
                &outs,
            );
        }
    }
}

/// The executor's outputs (`outs[0]`) against the reference's (`outs[1]`):
/// codes row by row, scale words group by group, every word.
fn compare(what: &str, k: usize, m_pad: usize, outs: &[(Vec<u8>, Vec<u32>)]) {
    let ((new_q, new_sf), (old_q, old_sf)) = (&outs[0], &outs[1]);
    for (r, (a, b)) in new_q.chunks(k).zip(old_q.chunks(k)).enumerate() {
        assert_eq!(a, b, "{what}: row {r} codes");
    }
    for (g, (a, b)) in new_sf.chunks(m_pad).zip(old_sf.chunks(m_pad)).enumerate() {
        assert_eq!(a, b, "{what}: group {g} scales");
    }
}
