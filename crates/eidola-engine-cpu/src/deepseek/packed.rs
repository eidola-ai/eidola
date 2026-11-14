//! The packed forms KV is stored in: the model's own low-precision formats, byte for
//! byte what a device executor holds.
//!
//! | Form | Bytes for `n` elements | Layout |
//! |---|---|---|
//! | [`Fp8`]: window KV | `n + n / 32` | `n` FP8 E4M3 codes, then one E8M0 scale per 32 |
//! | [`Fp4E4m3`]: global KV | `n / 2 + n / 16` | E2M1 codes, two per byte (element `2i` in the low nibble), then one FP8 E4M3 scale per 16 |
//! | [`Fp4Pow2`]: index key | `n / 2 + n / 32` | E2M1 codes as above, then one E8M0 scale per 32 |
//!
//! Packing quantises with the model crate's kernels (`act_quant`, `quant_fp4_e4m3`,
//! `quant_fp4_pow2`) and stores their codes and scales exactly; unpacking is
//! `bf16(code · scale)`, the expression the model's in-place quantisers end with. So a
//! value read back from a pool is, bit for bit, the value the dense forward holds.

use std::sync::OnceLock;

use eidola_engine_model::deepseek::numeric::bf16_round;
use eidola_engine_model::deepseek::quant::{act_quant, quant_fp4_e4m3, quant_fp4_pow2};
use eidola_engine_model::numeric::{E2M1_VALUES, e8m0_to_f32, fp8_e4m3_to_f32};

/// The FP8 E4M3 byte of a code point (either zero keeps its sign).
fn e4m3_byte(v: f32) -> u8 {
    static TABLE: OnceLock<Vec<(u32, u8)>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t: Vec<(u32, u8)> = (0..=255u8)
            .map(|b| (fp8_e4m3_to_f32(b), b))
            .filter(|(v, _)| v.is_finite())
            .map(|(v, b)| (v.to_bits(), b))
            .collect();
        t.sort_unstable();
        t
    });
    let i = table
        .binary_search_by_key(&v.to_bits(), |e| e.0)
        .unwrap_or_else(|_| panic!("{v} is not an FP8 E4M3 value"));
    table[i].1
}

/// The E2M1 nibble of a code point.
fn e2m1_nibble(v: f32) -> u8 {
    E2M1_VALUES
        .iter()
        .position(|c| c.to_bits() == v.to_bits())
        .unwrap_or_else(|| panic!("{v} is not an FP4 E2M1 value")) as u8
}

/// The E8M0 byte of a power-of-two scale.
fn e8m0_byte(s: f32) -> u8 {
    let b = ((s.to_bits() >> 23) & 0xff) as u8;
    assert_eq!(
        e8m0_to_f32(b).to_bits(),
        s.to_bits(),
        "{s} is not an E8M0 scale"
    );
    b
}

fn pack_nibbles(codes: &[f32], out: &mut [u8]) {
    for (byte, pair) in out.iter_mut().zip(codes.chunks_exact(2)) {
        *byte = e2m1_nibble(pair[0]) | e2m1_nibble(pair[1]) << 4;
    }
}

fn nibble(bytes: &[u8], i: usize) -> f32 {
    E2M1_VALUES[(bytes[i / 2] >> (4 * (i % 2)) & 0xf) as usize]
}

/// FP8 E4M3 codes with an E8M0 scale per 32 elements.
pub(crate) struct Fp8;

impl Fp8 {
    pub fn bytes(n: usize) -> usize {
        n + n / 32
    }

    /// Quantises `x` (the value before the model's fake FP8 quantisation) into `out`.
    pub fn pack(x: &[f32], out: &mut [u8]) {
        let n = x.len();
        let (mut codes, mut scales) = (vec![0.0; n], vec![0.0; n / 32]);
        act_quant(x, &mut codes, &mut scales);
        for (b, c) in out[..n].iter_mut().zip(&codes) {
            *b = e4m3_byte(*c);
        }
        for (b, s) in out[n..Self::bytes(n)].iter_mut().zip(&scales) {
            *b = e8m0_byte(*s);
        }
    }

    pub fn unpack(bytes: &[u8], out: &mut [f32]) {
        let n = out.len();
        for (j, o) in out.iter_mut().enumerate() {
            *o = bf16_round(fp8_e4m3_to_f32(bytes[j]) * e8m0_to_f32(bytes[n + j / 32]));
        }
    }
}

/// E2M1 codes with an FP8 E4M3 scale per 16 elements.
pub(crate) struct Fp4E4m3;

impl Fp4E4m3 {
    pub fn bytes(n: usize) -> usize {
        n / 2 + n / 16
    }

    pub fn pack(x: &[f32], out: &mut [u8]) {
        let n = x.len();
        let (mut codes, mut scales) = (vec![0.0; n], vec![0.0; n / 16]);
        quant_fp4_e4m3(x, &mut codes, &mut scales);
        pack_nibbles(&codes, &mut out[..n / 2]);
        for (b, s) in out[n / 2..Self::bytes(n)].iter_mut().zip(&scales) {
            *b = e4m3_byte(*s);
        }
    }

    pub fn unpack(bytes: &[u8], out: &mut [f32]) {
        let n = out.len();
        for (j, o) in out.iter_mut().enumerate() {
            *o = bf16_round(nibble(bytes, j) * fp8_e4m3_to_f32(bytes[n / 2 + j / 16]));
        }
    }
}

/// E2M1 codes with an E8M0 scale per 32 elements.
pub(crate) struct Fp4Pow2;

impl Fp4Pow2 {
    pub fn bytes(n: usize) -> usize {
        n / 2 + n / 32
    }

    pub fn pack(x: &[f32], out: &mut [u8]) {
        let n = x.len();
        let (mut codes, mut scales) = (vec![0.0; n], vec![0.0; n / 32]);
        quant_fp4_pow2(x, &mut codes, &mut scales);
        pack_nibbles(&codes, &mut out[..n / 2]);
        for (b, s) in out[n / 2..Self::bytes(n)].iter_mut().zip(&scales) {
            *b = e8m0_byte(*s);
        }
    }

    pub fn unpack(bytes: &[u8], out: &mut [f32]) {
        let n = out.len();
        for (j, o) in out.iter_mut().enumerate() {
            *o = bf16_round(nibble(bytes, j) * e8m0_to_f32(bytes[n / 2 + j / 32]));
        }
    }
}

#[cfg(test)]
mod tests {
    use eidola_engine_model::deepseek::quant::{
        fake_quant_fp4_e4m3, fake_quant_fp4_pow2, fake_quant_fp8,
    };

    use super::*;

    fn vectors() -> Vec<Vec<f32>> {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        };
        (0..300)
            .map(|case| {
                let magnitude = [1e-9f32, 1e-3, 0.4, 3.0, 70.0, 900.0][case % 6];
                (0..64)
                    .map(|i| match (case + i) % 13 {
                        0 => 0.0,
                        1 => -0.0,
                        _ => next() * magnitude,
                    })
                    .collect()
            })
            .collect()
    }

    /// A packed value unpacks to exactly what the model's in-place quantiser leaves,
    /// signed zeros included, in the documented number of bytes.
    #[test]
    fn packed_forms_round_trip_the_models_quantisers() {
        type Form = (
            fn(usize) -> usize,
            fn(&[f32], &mut [u8]),
            fn(&[u8], &mut [f32]),
            fn(&mut [f32]),
            usize,
        );
        let forms: [Form; 3] = [
            (Fp8::bytes, Fp8::pack, Fp8::unpack, fake_quant_fp8, 66),
            (
                Fp4E4m3::bytes,
                Fp4E4m3::pack,
                Fp4E4m3::unpack,
                fake_quant_fp4_e4m3,
                36,
            ),
            (
                Fp4Pow2::bytes,
                Fp4Pow2::pack,
                Fp4Pow2::unpack,
                fake_quant_fp4_pow2,
                34,
            ),
        ];
        for (bytes, pack, unpack, fake, want) in forms {
            assert_eq!(bytes(64), want);
            for x in vectors() {
                let mut packed = vec![0u8; bytes(64)];
                pack(&x, &mut packed);
                let mut got = vec![0.0; 64];
                unpack(&packed, &mut got);
                let mut expect = x.clone();
                fake(&mut expect);
                let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&got), bits(&expect));
            }
        }
        // The released shapes: 528 bytes of window KV, 288 + 68 of a global entry.
        assert_eq!(Fp8::bytes(512), 528);
        assert_eq!(Fp4E4m3::bytes(512) + Fp4Pow2::bytes(128), 356);
    }
}
