//! Scalar decoders for the storage formats MiMo checkpoints use.
//!
//! Every decode here is exact: each format's values are representable in f32.

/// BF16 is the upper half of an IEEE f32.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// OCP FP8 E4M3 ("e4m3fn"): bias 7, no infinities, `S.1111.111` is NaN,
/// largest finite magnitude 448.
pub fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    FP8_E4M3_TABLE[bits as usize]
}

const fn fp8_e4m3_decode(bits: u8) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((bits >> 3) & 0x0f) as i32;
    let man = (bits & 0x07) as i32;
    if exp == 0x0f && man == 0x07 {
        return f32::NAN;
    }
    let magnitude = if exp == 0 {
        // Subnormal: m/8 * 2^-6.
        man as f32 / 8.0 / 64.0
    } else {
        let frac = 1.0 + man as f32 / 8.0;
        let mut p = 1.0f32;
        let mut e = exp - 7;
        while e > 0 {
            p *= 2.0;
            e -= 1;
        }
        while e < 0 {
            p /= 2.0;
            e += 1;
        }
        frac * p
    };
    sign * magnitude
}

const FP8_E4M3_TABLE: [f32; 256] = {
    let mut t = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = fp8_e4m3_decode(i as u8);
        i += 1;
    }
    t
};

/// OCP FP4 E2M1 code points, indexed by the 4-bit code (bit 3 is the sign).
pub const E2M1_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// OCP E8M0 shared scale: `2^(bits - 127)`; 255 is NaN.
pub fn e8m0_to_f32(bits: u8) -> f32 {
    if bits == 0xff {
        return f32::NAN;
    }
    // 2^-127 is subnormal in f32 but still exact.
    let e = bits as i32 - 127;
    if e >= -126 {
        f32::from_bits(((e + 127) as u32) << 23)
    } else {
        f32::from_bits(1u32 << 22)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_known_values() {
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0);
        assert_eq!(fp8_e4m3_to_f32(0xb8), -1.0);
        assert_eq!(fp8_e4m3_to_f32(0x7e), 448.0);
        assert_eq!(fp8_e4m3_to_f32(0xfe), -448.0);
        assert_eq!(fp8_e4m3_to_f32(0x01), 2f32.powi(-9));
        assert_eq!(fp8_e4m3_to_f32(0x08), 2f32.powi(-6));
        assert_eq!(fp8_e4m3_to_f32(0x3c), 1.5);
        assert!(fp8_e4m3_to_f32(0x7f).is_nan());
        assert!(fp8_e4m3_to_f32(0xff).is_nan());
        // 0x78 is 256 (exp 15, mantissa 0) — e4m3fn has no infinity.
        assert_eq!(fp8_e4m3_to_f32(0x78), 256.0);
    }

    #[test]
    fn fp8_table_is_monotonic_over_positive_codes() {
        for b in 1u8..0x7f {
            assert!(fp8_e4m3_to_f32(b) > fp8_e4m3_to_f32(b - 1), "code {b:#x}");
        }
    }

    #[test]
    fn e8m0_values() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(118), 2f32.powi(-9));
        assert_eq!(e8m0_to_f32(130), 8.0);
        assert_eq!(e8m0_to_f32(0), 2f32.powi(-127));
        assert!(e8m0_to_f32(255).is_nan());
    }

    #[test]
    fn bf16_roundtrip() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc000), -2.0);
    }
}
