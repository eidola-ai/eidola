//! Rounding points and transcendentals of the DeepSeek-V4.1 forward.
//!
//! The reference keeps activations in BF16 and quantises them to FP8 or FP4 at
//! run time; every such rounding is an explicit function here. Transcendentals
//! are built from correctly rounded `+ - * /` and `sqrt` alone (never the
//! platform `exp`, `log`, `sin` or `cos`, whose last bit differs between C
//! libraries), so every IEEE host, and a device kernel that fuses no
//! multiply-add, computes the same bits.

use crate::numeric::fp8_e4m3_to_f32;

/// Round an f32 to the nearest BF16 (ties to even) and widen it back.
#[inline]
pub fn bf16_round(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

pub fn bf16_round_slice(x: &mut [f32]) {
    for v in x {
        *v = bf16_round(*v);
    }
}

/// The 127 non-negative finite E4M3 values, in code order.
const E4M3_POSITIVE: [f32; 127] = {
    let mut t = [0.0f32; 127];
    let mut i = 0;
    while i < 127 {
        t[i] = fp8_e4m3_to_f32(i as u8);
        i += 1;
    }
    t
};

const E2M1_POSITIVE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Nearest code point of an ascending `table` to `|v|`, ties to the even code,
/// with `v`'s sign. `|v|` must not exceed the table's last entry.
#[inline]
fn round_to_codes(v: f32, table: &[f32]) -> f32 {
    let a = v.abs();
    let hi = table.partition_point(|&t| t < a).min(table.len() - 1);
    let lo = hi.saturating_sub(1);
    let (d_lo, d_hi) = (a - table[lo], table[hi] - a);
    let pick_hi = d_hi < d_lo || (d_hi == d_lo && hi % 2 == 0);
    table[if pick_hi { hi } else { lo }].copysign(v)
}

/// f32 → FP8 E4M3 → f32: saturating at ±448, nearest, ties to the even code.
#[inline]
pub fn round_e4m3(v: f32) -> f32 {
    round_to_codes(v.clamp(-448.0, 448.0), &E4M3_POSITIVE)
}

/// f32 → FP4 E2M1 → f32: saturating at ±6, nearest, ties to the even code
/// (0.25 → 0, 0.75 → 1, 1.25 → 1, 1.75 → 2, 2.5 → 2, 3.5 → 4, 5 → 4).
#[inline]
pub fn round_e2m1(v: f32) -> f32 {
    round_to_codes(v.clamp(-6.0, 6.0), &E2M1_POSITIVE)
}

/// `2^ceil(log2(amax · inv))` by the reference kernel's bit arithmetic
/// (`fast_round_scale`): the f32 product's exponent, plus one when its
/// mantissa is non-zero.
#[inline]
pub fn round_scale(amax: f32, inv: f32) -> f32 {
    let bits = (amax * inv).to_bits();
    let exp = ((bits >> 23) & 0xff) as i32;
    let e = exp - 127 + i32::from(bits & 0x7f_ffff != 0);
    f32::from_bits(((e + 127) as u32) << 23)
}

const INV_LN2: f64 = f64::from_bits(0x3ff71547652b82fe);
const LN2_HI: f64 = f64::from_bits(0x3fe62e42fee00000);
const LN2_LO: f64 = f64::from_bits(0x3dea39ef35793c76);

/// `1 / k!`, correctly rounded, for the degree-13 Taylor polynomial of `e^r`.
const EXP_COEFFS: [u64; 14] = [
    0x3ff0000000000000,
    0x3ff0000000000000,
    0x3fe0000000000000,
    0x3fc5555555555555,
    0x3fa5555555555555,
    0x3f81111111111111,
    0x3f56c16c16c16c17,
    0x3f2a01a01a01a01a,
    0x3efa01a01a01a01a,
    0x3ec71de3a556c734,
    0x3e927e4fb7789f5c,
    0x3e5ae64567f544e4,
    0x3e21eed8eff8d898,
    0x3de6124613a86d09,
];

/// `e^x`, the same function as `eidola_engine::sampling::det_exp` over the
/// range this forward uses: `n = round(x / ln 2)` (half away from zero),
/// `r = (x - n·ln2_hi) - n·ln2_lo`, a degree-13 Horner polynomial, then a
/// power-of-two scale. Arguments below -708 return 0; every caller rounds the
/// result to f32, where those are 0 anyway.
pub fn det_exp(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 709.782712893384 {
        return f64::INFINITY;
    }
    if x < -708.0 {
        return 0.0;
    }
    let t = x * INV_LN2;
    let n = (t.abs() + 0.5).floor().copysign(t);
    let r = (x - n * LN2_HI) - n * LN2_LO;
    let mut p = f64::from_bits(EXP_COEFFS[13]);
    for k in (0..13).rev() {
        p = p * r + f64::from_bits(EXP_COEFFS[k]);
    }
    let n = n as i64;
    let pow2 = |e: i64| f64::from_bits(((e + 1023) as u64) << 52);
    if n > 1023 {
        p * pow2(1023) * pow2(n - 1023)
    } else {
        p * pow2(n)
    }
}

/// Natural log of a positive normal f64: `x = m·2^e` with `m` in
/// `(sqrt ½, sqrt 2]`, `s = (m - 1) / (m + 1)`,
/// `ln m = 2s·(1 + s²/3 + … + s²²/23)`, `ln x = e·ln2_hi + (ln m + e·ln2_lo)`.
pub fn det_ln(x: f64) -> f64 {
    debug_assert!(x.is_normal() && x > 0.0, "det_ln({x})");
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > std::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let e = e as f64;
    let s = (m - 1.0) / (m + 1.0);
    let z = s * s;
    let mut p = 1.0 / 23.0;
    for k in (0..=10).rev() {
        p = p * z + 1.0 / (2 * k + 1) as f64;
    }
    e * LN2_HI + (2.0 * s * p + e * LN2_LO)
}

/// 2/π.
const INV_PIO2: f64 = f64::from_bits(0x3fe45f306dc9c883);
/// π/2 in three 33-bit pieces, so `n·piece` is exact for the `n` in range.
const PIO2: [f64; 3] = [
    f64::from_bits(0x3ff921fb54400000),
    f64::from_bits(0x3dd0b4611a600000),
    f64::from_bits(0x3ba3198a2e000000),
];
/// Odd polynomial of `sin` on `[-π/4, π/4]`: the coefficients of r³ … r¹³.
const SIN_COEFFS: [f64; 6] = [
    f64::from_bits(0xbfc5555555555549),
    f64::from_bits(0x3f8111111110f8a6),
    f64::from_bits(0xbf2a01a019c161d5),
    f64::from_bits(0x3ec71de357b1fe7d),
    f64::from_bits(0xbe5ae5e68a2b9ceb),
    f64::from_bits(0x3de5d93a5acfd57c),
];
/// Even polynomial of `cos` on `[-π/4, π/4]`: the coefficients of r⁴ … r¹⁴.
const COS_COEFFS: [f64; 6] = [
    f64::from_bits(0x3fa555555555554c),
    f64::from_bits(0xbf56c16c16c15177),
    f64::from_bits(0x3efa01a019cb1590),
    f64::from_bits(0xbe927e4f809c52ad),
    f64::from_bits(0x3e21ee9ebdb4b1c4),
    f64::from_bits(0xbda8fae9be8838d4),
];

/// `(sin x, cos x)` for `|x|` up to about 1.6e6: `n = round(x·2/π)` (half away
/// from zero), `r = ((x - n·p1) - n·p2) - n·p3`, polynomials in `r` on
/// `[-π/4, π/4]`, then the quadrant `n mod 4`.
pub fn det_sin_cos(x: f64) -> (f64, f64) {
    debug_assert!(x.abs() < 1.6e6, "det_sin_cos({x})");
    let t = x * INV_PIO2;
    let n = (t.abs() + 0.5).floor().copysign(t);
    let r = ((x - n * PIO2[0]) - n * PIO2[1]) - n * PIO2[2];
    let z = r * r;
    let (mut ps, mut pc) = (SIN_COEFFS[5], COS_COEFFS[5]);
    for k in (0..5).rev() {
        ps = ps * z + SIN_COEFFS[k];
        pc = pc * z + COS_COEFFS[k];
    }
    let sin = r + (r * z) * ps;
    let cos = (1.0 - 0.5 * z) + (z * z) * pc;
    match (n as i64).rem_euclid(4) {
        0 => (sin, cos),
        1 => (cos, -sin),
        2 => (-sin, -cos),
        _ => (-cos, sin),
    }
}

/// `e^x` for an f32 argument, rounded to f32.
#[inline]
pub fn exp32(x: f32) -> f32 {
    det_exp(x as f64) as f32
}

/// `1 / (1 + e^-x)`, evaluated in f64 and rounded to f32.
#[inline]
pub fn sigmoid32(x: f32) -> f32 {
    (1.0 / (1.0 + det_exp(-(x as f64)))) as f32
}

/// `x · sigmoid(x)` as `x / (1 + e^-x)`, evaluated in f64 and rounded to f32.
#[inline]
pub fn silu32(x: f32) -> f32 {
    let x = x as f64;
    (x / (1.0 + det_exp(-x))) as f32
}

/// `ln(1 + e^x)`, or `x` itself above 20 (the reference's softplus threshold).
/// While `e^x < 2^-20` the log is taken as `e - e²/2`.
#[inline]
pub fn softplus32(x: f32) -> f32 {
    if x > 20.0 {
        return x;
    }
    let e = det_exp(x as f64);
    let v = if e < 9.5367431640625e-7 {
        e - 0.5 * e * e
    } else {
        det_ln(1.0 + e)
    };
    v as f32
}

/// `base^e` for an f32 exponent, as `det_exp(e · det_ln(base))` rounded to f32.
#[inline]
pub fn pow32(base: f64, e: f32) -> f32 {
    det_exp(e as f64 * det_ln(base)) as f32
}

/// `1 / sqrt(x)` in f32: two correctly rounded operations.
#[inline]
pub fn rsqrt32(x: f32) -> f32 {
    1.0 / x.sqrt()
}

/// f32 sum in the crate's pinned order ([`crate::tensor::dot`]'s): 16
/// interleaved partial sums over the full 16-wide chunks, a halving tree over
/// them, then the tail left to right. Under 16 terms that is left to right.
pub fn pinned_sum(x: &[f32]) -> f32 {
    let mut acc = [0.0f32; 16];
    let (chunks, rest) = x.as_chunks::<16>();
    for c in chunks {
        for j in 0..16 {
            acc[j] += c[j];
        }
    }
    let mut w = 8;
    while w > 0 {
        for j in 0..w {
            acc[j] += acc[j + w];
        }
        w /= 2;
    }
    let mut tail = 0.0f32;
    for v in rest {
        tail += v;
    }
    acc[0] + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounds_to_nearest_even() {
        let f = |bits: u32| f32::from_bits(bits);
        assert_eq!(bf16_round(1.0), 1.0);
        // Exactly half way: ties go to the even BF16 mantissa.
        assert_eq!(bf16_round(f(0x3f80_8000)), f(0x3f80_0000));
        assert_eq!(bf16_round(f(0x3f81_8000)), f(0x3f82_0000));
        assert_eq!(bf16_round(f(0x3f80_8001)), f(0x3f81_0000));
        assert_eq!(bf16_round(f(0x3f80_7fff)), f(0x3f80_0000));
        // A carry out of the mantissa moves to the next binade.
        assert_eq!(bf16_round(f(0x3fff_ffff)), 2.0);
        assert_eq!(bf16_round(-f(0x3f81_8000)), -f(0x3f82_0000));
        assert_eq!(bf16_round(f32::MAX), f32::INFINITY);
    }

    #[test]
    fn e4m3_rounding_is_nearest_even_and_saturating() {
        assert_eq!(round_e4m3(1.0625), 1.0);
        assert_eq!(round_e4m3(1.1875), 1.25);
        assert_eq!(round_e4m3(-1.1875), -1.25);
        assert_eq!(round_e4m3(1.07), 1.125);
        assert_eq!(round_e4m3(1000.0), 448.0);
        assert_eq!(round_e4m3(-1e9), -448.0);
        assert_eq!(round_e4m3(435.0), 448.0);
        // Subnormals step by 2^-9; the half-way point to the first goes to 0.
        let step = 2f32.powi(-9);
        assert_eq!(round_e4m3(0.5 * step), 0.0);
        assert_eq!(round_e4m3(1.5 * step), 2.0 * step);
        assert_eq!(round_e4m3(0.51 * step), step);
        for code in 0..127u8 {
            let v = fp8_e4m3_to_f32(code);
            assert_eq!(round_e4m3(v), v);
        }
    }

    #[test]
    fn e2m1_ties_go_to_the_even_code() {
        let cases = [
            (0.25, 0.0),
            (0.75, 1.0),
            (1.25, 1.0),
            (1.75, 2.0),
            (2.5, 2.0),
            (3.5, 4.0),
            (5.0, 4.0),
            (5.01, 6.0),
            (7.0, 6.0),
            (0.26, 0.5),
        ];
        for (x, want) in cases {
            assert_eq!(round_e2m1(x), want, "{x}");
            assert_eq!(round_e2m1(-x), -want, "-{x}");
        }
    }

    #[test]
    fn round_scale_is_the_ceiling_power_of_two() {
        let inv = (1.0f64 / 448.0) as f32;
        assert_eq!(round_scale(448.0, inv), 1.0);
        assert_eq!(round_scale(449.0, inv), 2.0);
        assert_eq!(round_scale(224.0, inv), 0.5);
        assert_eq!(round_scale(1e-4f64 as f32, inv), 2f32.powi(-22));
        let inv6 = (1.0f64 / 6.0) as f32;
        // 1.5 · fl(1/6) rounds to exactly 0.25, so the scale does not step up.
        assert_eq!(round_scale(1.5, inv6), 0.25);
        assert_eq!(round_scale(6.0 * 2f32.powi(-126), inv6), 2f32.powi(-126));
    }

    /// The bits are the contract: a device executor reproduces them.
    #[test]
    fn transcendentals_are_pinned() {
        assert_eq!(det_exp(0.0), 1.0);
        for (x, bits) in [
            (-1.0, 0x3fd78b56362cef38u64),
            (-0.5, 0x3fe368b2fc6f960a),
            (-13.37, 0x3eba31ad57ba12f4),
            (-700.25, 0x00caf5fe9a485c8e),
            (3.0, 0x403415e5bf6fb106),
        ] {
            assert_eq!(det_exp(x).to_bits(), bits, "det_exp({x})");
        }
        assert_eq!(det_exp(f64::NEG_INFINITY), 0.0);
        assert_eq!(det_ln(1.0), 0.0);
        let ulps = |a: f64, b: f64| (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs();
        let mut s = 7u64;
        for _ in 0..100_000 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            let x = 1e-6 + u * 2e5;
            assert!(ulps(det_ln(x), x.ln()) <= 2, "ln({x})");
            let a = (u - 0.5) * 2.0e6;
            let (sin, cos) = det_sin_cos(a);
            assert!(
                (sin - a.sin()).abs() < 4e-16 && (cos - a.cos()).abs() < 4e-16,
                "{a}"
            );
        }
        assert_eq!(det_sin_cos(0.0), (0.0, 1.0));
    }
}
