//! BF16 bit patterns on the host. Device tensors hold BF16 as `u16`.

/// Exact widening.
pub fn to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Round to nearest, ties to even, as `__float2bfloat16` does; NaN stays NaN.
pub fn from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_ties() {
        for x in [0.0f32, -0.0, 1.0, -2.5, 3.140625, f32::INFINITY] {
            assert_eq!(to_f32(from_f32(x)), x);
        }
        // 1 + 2^-8 is halfway between 1 and 1 + 2^-7: ties to even (1).
        assert_eq!(from_f32(1.0 + 1.0 / 256.0), from_f32(1.0));
        // 1 + 3 * 2^-8 is halfway between 1 + 2^-7 and 1 + 2^-6: to even.
        assert_eq!(to_f32(from_f32(1.0 + 3.0 / 256.0)), 1.0 + 4.0 / 256.0);
        assert!(to_f32(from_f32(f32::NAN)).is_nan());
    }
}
