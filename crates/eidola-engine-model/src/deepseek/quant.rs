//! Run-time activation quantisation, as the reference's `kernel.py` defines
//! it (`act_quant_kernel`, `fp4_quant_kernel`).
//!
//! Every scale is computed per token vector, per block of consecutive
//! elements, from that block alone.

use super::numeric::{bf16_round, round_e2m1, round_e4m3, round_scale};

/// Elements per activation scale of the FP8 quantiser.
pub const FP8_BLOCK: usize = 32;

fn amax(block: &[f32]) -> f32 {
    block.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

/// The FP8 block scale: `2^ceil(log2(max(amax, 1e-4) / 448))`.
#[inline]
fn fp8_scale(block: &[f32]) -> f32 {
    round_scale(amax(block).max(1e-4f64 as f32), (1.0f64 / 448.0) as f32)
}

/// FP8 quantisation of one activation vector in 32-wide blocks: `codes[j]` is
/// the E4M3 value of `x[j] / scale` (not yet multiplied back) and `scales`
/// holds one power of two per block. This is what a quantised GEMM consumes.
pub fn act_quant(x: &[f32], codes: &mut [f32], scales: &mut [f32]) {
    assert!(x.len().is_multiple_of(FP8_BLOCK), "width {}", x.len());
    assert_eq!(codes.len(), x.len());
    assert_eq!(scales.len(), x.len() / FP8_BLOCK);
    for ((xb, cb), s) in x
        .chunks_exact(FP8_BLOCK)
        .zip(codes.chunks_exact_mut(FP8_BLOCK))
        .zip(scales.iter_mut())
    {
        *s = fp8_scale(xb);
        for (c, &v) in cb.iter_mut().zip(xb) {
            *c = round_e4m3(v / *s);
        }
    }
}

/// Fake FP8 quantisation in place (32-wide blocks, power-of-two scale): each
/// element becomes `bf16(e4m3(x / s) · s)`.
pub fn fake_quant_fp8(x: &mut [f32]) {
    assert!(x.len().is_multiple_of(FP8_BLOCK), "width {}", x.len());
    for block in x.chunks_exact_mut(FP8_BLOCK) {
        let s = fp8_scale(block);
        for v in block {
            *v = bf16_round(round_e4m3(*v / s) * s);
        }
    }
}

/// Fake FP4 quantisation in place with a power-of-two scale per 32 elements
/// (`2^ceil(log2(max(amax, 6·2^-126) / 6))`): the index keys and queries.
pub fn fake_quant_fp4_pow2(x: &mut [f32]) {
    assert!(x.len().is_multiple_of(32), "width {}", x.len());
    for block in x.chunks_exact_mut(32) {
        let floor = 6.0 * f32::from_bits(1 << 23);
        let s = round_scale(amax(block).max(floor), (1.0f64 / 6.0) as f32);
        for v in block {
            *v = bf16_round(round_e2m1(*v / s) * s);
        }
    }
}

/// FP4 quantisation with a power-of-two scale per 32 elements, as codes and
/// scales: `codes[j]` is the E2M1 value of `x[j] / scale` and `scales` holds
/// one power of two per block. `bf16(codes[j] · scale)` is what
/// [`fake_quant_fp4_pow2`] leaves in place of `x[j]`; this is the form a
/// packed index key is stored in.
pub fn quant_fp4_pow2(x: &[f32], codes: &mut [f32], scales: &mut [f32]) {
    assert!(x.len().is_multiple_of(32), "width {}", x.len());
    assert_eq!(codes.len(), x.len());
    assert_eq!(scales.len(), x.len() / 32);
    for ((xb, cb), s) in x
        .chunks_exact(32)
        .zip(codes.chunks_exact_mut(32))
        .zip(scales.iter_mut())
    {
        let floor = 6.0 * f32::from_bits(1 << 23);
        *s = round_scale(amax(xb).max(floor), (1.0f64 / 6.0) as f32);
        for (c, &v) in cb.iter_mut().zip(xb) {
            *c = round_e2m1(v / *s);
        }
    }
}

/// FP4 quantisation with an E4M3 scale per 16 elements, as codes and scales:
/// `bf16(codes[j] · scale)` is what [`fake_quant_fp4_e4m3`] leaves in place of
/// `x[j]`; this is the form a packed global KV entry is stored in.
pub fn quant_fp4_e4m3(x: &[f32], codes: &mut [f32], scales: &mut [f32]) {
    assert!(x.len().is_multiple_of(16), "width {}", x.len());
    assert_eq!(codes.len(), x.len());
    assert_eq!(scales.len(), x.len() / 16);
    for ((xb, cb), s) in x
        .chunks_exact(16)
        .zip(codes.chunks_exact_mut(16))
        .zip(scales.iter_mut())
    {
        *s = round_e4m3(amax(xb).max(6.0 * 2f32.powi(-9)) / 6.0);
        for (c, &v) in cb.iter_mut().zip(xb) {
            *c = round_e2m1(v / *s);
        }
    }
}

/// Fake FP4 quantisation in place with an E4M3 scale per 16 elements
/// (`e4m3(max(amax, 6·2^-9) / 6)`): the global KV. The scale is not a power
/// of two, so `x / s` is a rounded division and `amax / s` can exceed 6, where
/// the cast saturates.
pub fn fake_quant_fp4_e4m3(x: &mut [f32]) {
    assert!(x.len().is_multiple_of(16), "width {}", x.len());
    for block in x.chunks_exact_mut(16) {
        let s = round_e4m3(amax(block).max(6.0 * 2f32.powi(-9)) / 6.0);
        for v in block {
            *v = bf16_round(round_e2m1(*v / s) * s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_blocks_scale_independently() {
        let mut x = vec![0.0f32; 64];
        x[0] = 448.0;
        x[1] = 3.3;
        x[32] = 1e-6;
        let (mut codes, mut scales) = (vec![0.0; 64], vec![0.0; 2]);
        act_quant(&x, &mut codes, &mut scales);
        assert_eq!(scales, [1.0, 2f32.powi(-22)]);
        assert_eq!(codes[0], 448.0);
        assert_eq!(codes[1], 3.25);
        // Below the 1e-4 floor the block still quantises against 2^-22.
        assert_eq!(codes[32], round_e4m3(1e-6 * 2f32.powi(22)));
        let mut y = x.clone();
        fake_quant_fp8(&mut y);
        assert_eq!(y[1], 3.25);
        assert_eq!(y[0], 448.0);
    }

    #[test]
    fn fp4_e4m3_scale_can_saturate() {
        // amax / 6 = 1.1666…, which rounds down to the E4M3 value 1.125, so
        // amax / s = 6.22 and the cast saturates at 6.
        let mut x = vec![0.0f32; 16];
        x[0] = 7.0;
        x[1] = -1.0;
        fake_quant_fp4_e4m3(&mut x);
        assert_eq!(x[0], 6.0 * 1.125);
        assert_eq!(x[1], -1.125);
        let mut z = vec![0.0f32; 16];
        fake_quant_fp4_e4m3(&mut z);
        assert!(z.iter().all(|v| *v == 0.0));
    }

    /// The code-and-scale forms dequantise to exactly what the in-place
    /// quantisers leave, signed zeros included, for every quantiser a KV is
    /// stored through.
    #[test]
    fn codes_and_scales_dequantise_to_the_fake_quantised_values() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..400 {
            let magnitude = [1e-9f32, 1e-3, 0.4, 3.0, 70.0, 900.0][case % 6];
            let x: Vec<f32> = (0..64)
                .map(|i| match (case + i) % 11 {
                    0 => 0.0,
                    1 => -0.0,
                    _ => ((next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * magnitude,
                })
                .collect();
            let same = |codes: &[f32], scales: &[f32], per: usize, fake: &[f32]| {
                for (j, (c, f)) in codes.iter().zip(fake).enumerate() {
                    let v = bf16_round(c * scales[j / per]);
                    assert_eq!(v.to_bits(), f.to_bits(), "case {case} element {j}");
                }
            };
            let (mut codes, mut s8) = (vec![0.0; 64], vec![0.0; 2]);
            act_quant(&x, &mut codes, &mut s8);
            let mut fake = x.clone();
            fake_quant_fp8(&mut fake);
            same(&codes, &s8, 32, &fake);
            quant_fp4_pow2(&x, &mut codes, &mut s8);
            let mut fake = x.clone();
            fake_quant_fp4_pow2(&mut fake);
            same(&codes, &s8, 32, &fake);
            let mut s16 = vec![0.0; 4];
            quant_fp4_e4m3(&x, &mut codes, &mut s16);
            let mut fake = x.clone();
            fake_quant_fp4_e4m3(&mut fake);
            same(&codes, &s16, 16, &fake);
        }
    }

    #[test]
    fn fp4_pow2_scale() {
        let mut x = vec![0.0f32; 32];
        x[0] = 3.0;
        x[1] = 0.3;
        x[2] = -0.2;
        fake_quant_fp4_pow2(&mut x);
        // amax 3 → scale 0.5; 0.3 / 0.5 = 0.6 → 0.5; -0.2 / 0.5 = -0.4 → -0.5.
        assert_eq!(&x[..3], &[3.0, 0.25, -0.25]);
    }
}
