//! Seeded cases of the DeepSeek-V4.1 scalar kernels: each kernel's inputs
//! drawn from one generator and its outputs computed by the model crate's
//! reference functions. The GPU tests launch a case and compare every output
//! bit; the vector files (`eidola-engine-kernels`' `tests/data/ds_*.vectors`)
//! record a digest of the same outputs, which that crate's host check
//! reproduces by running the kernels' bodies (`csrc/ds_ops.cuh`) on the host
//! over inputs it draws the same way (`oracle/ds_kernels_host.cc` mirrors
//! [`build`] draw for draw).
//!
//! A case holds no NaN and no infinity: which NaN an operation produces
//! differs between processors, and a quantiser turns a NaN's sign into a code.
#![allow(dead_code)]

pub mod device;
pub mod suite;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use eidola_engine_cuda::ds::abi;
use eidola_engine_cuda::ds::ops::expert_layout;
use eidola_engine_model::deepseek::config::DsMoeSpec;
use eidola_engine_model::deepseek::forward::stream_mean;
use eidola_engine_model::deepseek::linear::{Fp4Linear, Fp8Linear, lin4, lin8, linear_f32};
use eidola_engine_model::deepseek::mhc;
use eidola_engine_model::deepseek::moe::{route, swiglu_hidden};
use eidola_engine_model::deepseek::numeric::{
    bf16_round, exp32, pinned_sum, round_e2m1, round_e4m3, round_scale, rsqrt32, sigmoid32,
};
use eidola_engine_model::deepseek::quant::{
    act_quant, fake_quant_fp4_e4m3, fake_quant_fp4_pow2, fake_quant_fp8,
};
use eidola_engine_model::deepseek::rope::{rotate, unrotate};
use eidola_engine_model::deepseek::weights::rms;
use eidola_engine_model::numeric::{E2M1_VALUES, e8m0_to_f32, fp8_e4m3_to_f32};
use eidola_engine_model::tensor::{Matrix, dot};

const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;

fn fin(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// SplitMix64, as the numerics vectors' generator.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GOLDEN);
        fin(self.0)
    }

    pub fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n as u64).unwrap()
    }

    /// An f32 of any sign with exponent field `base + (0..spread)`.
    pub fn f32_in(&mut self, base: u32, spread: u32) -> f32 {
        let r = self.next();
        let exp = base + ((r >> 32) & 0xff) as u32 % spread;
        f32::from_bits(((r >> 63) as u32) << 31 | exp << 23 | (r & 0x7f_ffff) as u32)
    }

    pub fn f32s(&mut self, n: usize, base: u32, spread: u32) -> Vec<f32> {
        (0..n).map(|_| self.f32_in(base, spread)).collect()
    }

    /// BF16 values of any sign with exponent field `base + (0..spread)`; one
    /// in 61 is a zero of either sign.
    pub fn bf16_range(&mut self, n: usize, base: u32, spread: u32) -> Vec<u16> {
        (0..n)
            .map(|_| {
                let r = self.next();
                let sign = ((r >> 63) as u16) << 15;
                if (r >> 40).is_multiple_of(61) {
                    sign
                } else {
                    let exp = base + ((r >> 32) & 0xff) as u32 % spread;
                    sign | u16::try_from(exp).unwrap() << 7 | (r & 0x7f) as u16
                }
            })
            .collect()
    }

    /// Moderate BF16 values sharing a magnitude range: 2^-27 to 2^20.
    pub fn bf16s(&mut self, n: usize) -> Vec<u16> {
        let base = 100 + n32(self.below(40));
        self.bf16_range(n, base, 8)
    }

    /// BF16 values over 40 binades, so that a sum of a few of them rounds
    /// and its order shows.
    pub fn bf16_wide(&mut self, n: usize) -> Vec<u16> {
        let base = 90 + n32(self.below(20));
        self.bf16_range(n, base, 40)
    }

    /// `n` BF16 values in 32-wide blocks, each block with its own magnitude
    /// range anywhere from the smallest normals to 2^85: what a quantiser's
    /// scale rule and floors see.
    pub fn bf16_blocks(&mut self, n: usize) -> Vec<u16> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n / 32 {
            let base = 1 + n32(self.below(200));
            out.extend(self.bf16_range(32, base, 12));
        }
        out
    }

    /// An E4M3 code that is not NaN.
    fn code(&mut self) -> u8 {
        let b = (self.next() & 0xff) as u8;
        if b & 0x7f == 0x7f { b - 1 } else { b }
    }

    pub fn codes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.code()).collect()
    }

    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() & 0xff) as u8).collect()
    }

    /// E8M0 weight scales: 2^-27 to 2^28.
    pub fn scales(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| u8::try_from(100 + self.below(56)).unwrap())
            .collect()
    }

    /// `active` distinct ascending ids below `groups`.
    fn ascending_ids(&mut self, active: usize, groups: usize) -> Vec<u32> {
        let mut at = 0;
        (0..active)
            .map(|j| {
                at += self.below(groups - at - (active - j) + 1);
                at += 1;
                n32(at - 1)
            })
            .collect()
    }
}

/// A device array of one element type.
#[derive(Clone, Debug, PartialEq)]
pub enum Buf {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    F32(Vec<f32>),
}

impl Buf {
    /// The array as digest words: integers as they are, floats as their bits.
    fn words(&self) -> Vec<u64> {
        match self {
            Buf::U8(v) => v.iter().map(|x| *x as u64).collect(),
            Buf::U16(v) => v.iter().map(|x| *x as u64).collect(),
            Buf::U32(v) => v.iter().map(|x| *x as u64).collect(),
            Buf::F32(v) => v
                .iter()
                .map(|x| {
                    assert!(!x.is_nan(), "a case holds no NaN");
                    x.to_bits() as u64
                })
                .collect(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Buf::U8(v) => v.len(),
            Buf::U16(v) => v.len(),
            Buf::U32(v) => v.len(),
            Buf::F32(v) => v.len(),
        }
    }
}

/// One output array: what the buffer holds before the launch and after it.
#[derive(Clone, Debug)]
pub struct Output {
    pub name: &'static str,
    pub before: Buf,
    pub after: Buf,
}

/// One kernel launch (or a short chain of them) with its expected outputs.
#[derive(Clone, Debug)]
pub struct Case {
    pub kernel: &'static str,
    /// Counts, strides and flags, by the parameter struct's field names.
    pub n: BTreeMap<&'static str, u32>,
    pub f: BTreeMap<&'static str, f32>,
    pub inputs: BTreeMap<&'static str, Buf>,
    pub outputs: Vec<Output>,
}

impl Case {
    fn new(kernel: &'static str) -> Case {
        Case {
            kernel,
            n: BTreeMap::new(),
            f: BTreeMap::new(),
            inputs: BTreeMap::new(),
            outputs: Vec::new(),
        }
    }

    pub fn u(&self, name: &str) -> u32 {
        *self
            .n
            .get(name)
            .unwrap_or_else(|| panic!("no count {name}"))
    }

    /// The count `name`, or 0 for a case that has none.
    pub fn u_or_zero(&self, name: &str) -> u32 {
        self.n.get(name).copied().unwrap_or(0)
    }

    pub fn has(&self, name: &str) -> bool {
        self.inputs.contains_key(name)
    }

    fn set(&mut self, name: &'static str, v: usize) {
        self.n.insert(name, u32::try_from(v).unwrap());
    }

    fn out(&mut self, name: &'static str, before: Buf, after: Buf) {
        assert_eq!(before.len(), after.len(), "{name}");
        self.outputs.push(Output {
            name,
            before,
            after,
        });
    }

    fn out_u8(&mut self, name: &'static str, after: Vec<u8>) {
        self.out(
            name,
            Buf::U8(vec![SENTINEL_U8; after.len()]),
            Buf::U8(after),
        );
    }

    fn out_u16(&mut self, name: &'static str, after: Vec<u16>) {
        self.out(
            name,
            Buf::U16(vec![SENTINEL_U16; after.len()]),
            Buf::U16(after),
        );
    }

    /// Fold every output word, in order, into `h`.
    fn digest_into(&self, h: &mut u64) {
        for o in &self.outputs {
            for w in o.after.words() {
                *h = fin(h.wrapping_add(GOLDEN).wrapping_add(w));
            }
        }
    }
}

/// What an output buffer holds before a launch, so an element the kernel
/// should not write is seen.
pub const SENTINEL_U8: u8 = 0xa5;
pub const SENTINEL_U16: u16 = 0xa5a5;
pub const SENTINEL_U32: u32 = 0xa5a5_a5a5;
pub const SENTINEL_F32: f32 = -7777.0;

/// `1e-6`, the norm epsilon of every case.
pub const EPS: f32 = 1e-6;

/// The kernels with cases, in the vector files' order: `ds_ops.vectors` then
/// `ds_gemm_ref.vectors`.
pub const OPS_KERNELS: [&str; 19] = [
    "embed",
    "quant_fp8",
    "rot_quant",
    "rot_quant_e4m3",
    "rotate",
    "rms_norm",
    "mhc_mix",
    "collapse",
    "expand",
    "stream_mean",
    "linear",
    "pool",
    "route",
    "place",
    "place_rows",
    "swiglu",
    "combine",
    "engram_gather",
    "engram_inject",
];
pub const GEMM_REF_KERNELS: [&str; 4] = ["lin8", "lin4", "grouped_dot", "head"];

fn n32(x: usize) -> u32 {
    u32::try_from(x).unwrap()
}

pub fn val(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

pub fn vals(b: &[u16]) -> Vec<f32> {
    b.iter().map(|x| val(*x)).collect()
}

/// The stored bits of a BF16-valued f32.
pub fn bits(x: f32) -> u16 {
    assert_eq!(x.to_bits() & 0xffff, 0, "{x} is not a BF16 value");
    (x.to_bits() >> 16) as u16
}

pub fn bits_of(x: &[f32]) -> Vec<u16> {
    x.iter().map(|v| bits(*v)).collect()
}

/// The stored byte of an E4M3 value.
pub fn e4m3_code(v: f32) -> u8 {
    let code = (0u8..127)
        .find(|c| fp8_e4m3_to_f32(*c) == v.abs())
        .unwrap_or_else(|| panic!("{v} is not an E4M3 value"));
    code | if v.is_sign_negative() { 0x80 } else { 0 }
}

/// The stored nibble of an E2M1 value.
pub fn e2m1_code(v: f32) -> u8 {
    let code = (0u8..8)
        .find(|c| E2M1_VALUES[*c as usize] == v.abs())
        .unwrap_or_else(|| panic!("{v} is not an E2M1 value"));
    code | if v.is_sign_negative() { 8 } else { 0 }
}

/// The stored byte of a power-of-two activation scale.
pub fn scale_byte(s: f32) -> u8 {
    assert_eq!(s.to_bits() & 0x807f_ffff, 0, "{s} is not a power of two");
    u8::try_from(s.to_bits() >> 23).unwrap()
}

/// `quant::act_quant` of whole rows, as the stored codes and scale bytes.
pub fn quant_fp8(x: &[f32]) -> (Vec<u8>, Vec<u8>) {
    let (mut codes, mut scales) = (vec![0.0; x.len()], vec![0.0; x.len() / 32]);
    act_quant(x, &mut codes, &mut scales);
    (
        codes.iter().map(|c| e4m3_code(*c)).collect(),
        scales.iter().map(|s| scale_byte(*s)).collect(),
    )
}

fn amax(block: &[f32]) -> f32 {
    block.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

fn pack_nibbles(codes: &[u8]) -> Vec<u8> {
    codes.chunks_exact(2).map(|p| p[0] | p[1] << 4).collect()
}

/// A vector's fake quantisation as packed codes and scales, in one of the
/// three formats. The FP4 codes and scales are composed here from the
/// reference's public rounding functions by the rule `quant::fake_quant_fp4_*`
/// state, and checked against those functions' output element for element.
// The f64 constants round to f32 as the reference writes them.
#[allow(clippy::cast_possible_truncation)]
pub fn fake_quant_packed(v: &[f32], format: u32) -> (Vec<u8>, Vec<u8>) {
    let mut faked = v.to_vec();
    let (codes, scales, dequant): (Vec<u8>, Vec<u8>, Vec<f32>) = match format {
        abi::KV_FP8 => {
            fake_quant_fp8(&mut faked);
            let (codes, scales) = quant_fp8(v);
            let back = codes
                .iter()
                .enumerate()
                .map(|(j, c)| {
                    bf16_round(fp8_e4m3_to_f32(*c) * f32::from_bits((scales[j / 32] as u32) << 23))
                })
                .collect();
            (codes, scales, back)
        }
        abi::KV_FP4_E8M0 => {
            fake_quant_fp4_pow2(&mut faked);
            let (mut codes, mut scales, mut back) = (Vec::new(), Vec::new(), Vec::new());
            for block in v.chunks_exact(32) {
                let floor = 6.0 * f32::from_bits(1 << 23);
                let s = round_scale(amax(block).max(floor), (1.0f64 / 6.0) as f32);
                scales.push(scale_byte(s));
                for x in block {
                    let c = round_e2m1(*x / s);
                    codes.push(e2m1_code(c));
                    back.push(bf16_round(c * s));
                }
            }
            (pack_nibbles(&codes), scales, back)
        }
        abi::KV_FP4_E4M3 => {
            fake_quant_fp4_e4m3(&mut faked);
            let (mut codes, mut scales, mut back) = (Vec::new(), Vec::new(), Vec::new());
            for block in v.chunks_exact(16) {
                let s = round_e4m3(amax(block).max(6.0 * 2f32.powi(-9)) / 6.0);
                scales.push(e4m3_code(s));
                for x in block {
                    let c = round_e2m1(*x / s);
                    codes.push(e2m1_code(c));
                    back.push(bf16_round(c * s));
                }
            }
            (pack_nibbles(&codes), scales, back)
        }
        _ => panic!("format {format}"),
    };
    for (j, (a, b)) in dequant.iter().zip(&faked).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "element {j}: codes against fake_quant"
        );
    }
    (codes, scales)
}

/// `engram::inject` from the projected keys and value on: the reference's
/// expressions over its public functions. `tests/ds_gemm_ref.rs` holds it to
/// `engram::inject` itself.
// The f64 constants round to f32 as the reference writes them.
#[allow(clippy::cast_possible_truncation)]
pub fn inject_after_projection(x: &mut [f32], kv: &[f32], gate: &[f32], hc: usize, eps: f32) {
    let dim = x.len() / hc;
    let (keys, value) = kv.split_at(hc * dim);
    let scale = (1.0 / (dim as f64).sqrt()) as f32;
    let mut terms = vec![0.0f32; dim];
    for j in 0..hc {
        let xj = &mut x[j * dim..(j + 1) * dim];
        let key = &keys[j * dim..(j + 1) * dim];
        let rstd =
            rsqrt32(dot(xj, xj) / dim as f32 + eps) * rsqrt32(dot(key, key) / dim as f32 + eps);
        for d in 0..dim {
            terms[d] = xj[d] * gate[j * dim + d] * key[d];
        }
        let dot = pinned_sum(&terms) * rstd * scale;
        // A NaN counts as +0 (`engram::inject_gathered`).
        let dot = if dot.is_nan() { 0.0 } else { dot };
        let g = sigmoid32(dot.abs().max(1e-6f64 as f32).sqrt().copysign(dot));
        for d in 0..dim {
            xj[d] = bf16_round(xj[d] + g * value[d]);
        }
    }
}

/// A ratio-2 pooled channel, as `DsModel`'s compressor computes it.
fn pooled(kv: [f32; 2], score: [f32; 2]) -> f32 {
    let max = score.iter().fold(f32::NEG_INFINITY, |m, s| m.max(*s));
    let e = [exp32(score[0] - max), exp32(score[1] - max)];
    let sum = e.iter().fold(0.0f32, |a, v| a + v);
    let mut pooled = 0.0f32;
    for i in 0..2 {
        pooled += kv[i] * (e[i] / sum);
    }
    bf16_round(pooled)
}

/// `wide` or `1 + below(bound)`.
fn sz(g: &mut Rng, wide: Option<usize>, bound: usize) -> usize {
    match wide {
        Some(w) => w,
        None => 1 + g.below(bound),
    }
}

/// The shapes a case is drawn at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regime {
    /// Small random shapes: every remainder of every grid.
    Small,
    /// The released model's widths over a few rows.
    Wide,
    /// The released model's largest shapes over a step's worth of rows. For
    /// a device only: not in the vector files.
    Released,
}

/// One case of `kernel`. Every draw is in the order the host check repeats.
pub fn build(kernel: &'static str, regime: Regime, g: &mut Rng) -> Case {
    let wide = regime != Regime::Small;
    let w = |v: usize| wide.then_some(v);
    // A size the largest shapes change.
    let w2 = |v: usize, released: usize| match regime {
        Regime::Small => None,
        Regime::Wide => Some(v),
        Regime::Released => Some(released),
    };
    let mut c = Case::new(kernel);
    match kernel {
        "embed" => {
            let vocab = sz(g, w(300), 40);
            let dim = sz(g, w(5120), 600);
            let copies = sz(g, w(4), 4);
            let rows = sz(g, w2(3, 64), 5);
            let table = g.bf16s(vocab * dim);
            let mut status = 0;
            let mut out = Vec::new();
            let tokens: Vec<u32> = (0..rows)
                .map(|_| {
                    let mut t = g.below(vocab);
                    if g.below(8) == 0 {
                        t = vocab + g.below(3);
                    }
                    for _ in 0..copies {
                        if t < vocab {
                            out.extend_from_slice(&table[t * dim..(t + 1) * dim]);
                        } else {
                            status = abi::STATUS_BAD_TOKEN;
                            out.extend(std::iter::repeat_n(0, dim));
                        }
                    }
                    n32(t)
                })
                .collect();
            c.set("vocab", vocab);
            c.set("dim", dim);
            c.set("copies", copies);
            c.set("rows", rows);
            c.inputs.insert("table", Buf::U16(table));
            c.inputs.insert("tokens", Buf::U32(tokens));
            c.out_u16("out", out);
            c.out("status", Buf::U32(vec![0]), Buf::U32(vec![status]));
        }
        "quant_fp8" => {
            let rows = sz(g, w2(2, 64), 4);
            let width = 32 * sz(g, w2(160, 480), 8);
            let fp8 = g.below(2) == 1;
            let x: Vec<f32> = if fp8 {
                let codes = g.codes(rows * width);
                let scales = g.scales(rows * width / 32);
                let x = codes
                    .iter()
                    .enumerate()
                    .map(|(j, c)| bf16_round(fp8_e4m3_to_f32(*c) * e8m0_to_f32(scales[j / 32])))
                    .collect();
                c.inputs.insert("x", Buf::U8(codes));
                c.inputs.insert("x_scales", Buf::U8(scales));
                x
            } else {
                let x = g.bf16_blocks(rows * width);
                let v = vals(&x);
                c.inputs.insert("x", Buf::U16(x));
                v
            };
            let (codes, scales) = quant_fp8(&x);
            c.set("rows", rows);
            c.set("width", width);
            c.set("x_format", fp8 as usize);
            c.out_u8("codes", codes);
            c.out_u8("scales", scales);
        }
        "rot_quant" | "rot_quant_e4m3" => {
            let rows = sz(g, w2(2, 64), 3);
            let pooled = g.below(2) == 1;
            let heads = if pooled { 1 } else { sz(g, w2(4, 64), 3) };
            let head_dim = 32 * sz(g, w(16), 4);
            let rope_dim = if wide {
                64
            } else {
                2 * (1 + g.below(head_dim / 2))
            };
            let rotation = g.below(3);
            let format = if kernel == "rot_quant_e4m3" {
                abi::KV_FP4_E4M3
            } else if g.below(2) == 0 {
                abi::KV_FP8
            } else {
                abi::KV_FP4_E8M0
            };
            let cis_positions = 4 + g.below(8);
            let cis = g.f32s(cis_positions * rope_dim, 117, 10);
            let positions: Vec<u32> = (0..rows).map(|_| n32(g.below(cis_positions))).collect();
            let width = heads * head_dim;
            let x = g.bf16_blocks(rows * width);
            let (cb, sb) = (
                abi::kv_code_bytes(format, n32(width)) as usize,
                abi::kv_scale_bytes(format, n32(width)) as usize,
            );
            let packed: Vec<(Vec<u8>, Vec<u8>)> = (0..rows)
                .map(|r| {
                    let mut v = vals(&x[r * width..(r + 1) * width]);
                    let pairs: Vec<(f32, f32)> = cis[positions[r] as usize * rope_dim..]
                        [..rope_dim]
                        .chunks_exact(2)
                        .map(|p| (p[0], p[1]))
                        .collect();
                    for head in v.chunks_exact_mut(head_dim) {
                        match n32(rotation) {
                            abi::ROT_ROTATE => rotate(head, &pairs),
                            abi::ROT_UNROTATE => unrotate(head, &pairs),
                            _ => {}
                        }
                    }
                    fake_quant_packed(&v, format)
                })
                .collect();
            c.set("rows", rows);
            c.set("heads", heads);
            c.set("head_dim", head_dim);
            c.set("rope_dim", rope_dim);
            c.set("rotation", rotation);
            c.set("format", format as usize);
            c.set("cis_positions", cis_positions);
            c.inputs.insert("x", Buf::U16(x));
            c.inputs.insert("cis", Buf::F32(cis));
            c.inputs.insert("positions", Buf::U32(positions));
            if pooled {
                let epb = 1 + g.below(5);
                let blocks = 3 + g.below(3);
                let codes_offset = g.below(4);
                let scales_offset = codes_offset + epb * cb;
                let block_bytes = scales_offset + epb * sb + g.below(4);
                let total = epb * blocks;
                let first = g.below(total);
                let mut pool = vec![SENTINEL_U8; blocks * block_bytes];
                let targets: Vec<u32> = (0..rows)
                    .map(|r| {
                        if g.below(4) == 0 {
                            return abi::SLOT_EMPTY;
                        }
                        let e = (first + r) % total;
                        let at = e / epb * block_bytes;
                        let o = e % epb;
                        pool[at + codes_offset + o * cb..][..cb].copy_from_slice(&packed[r].0);
                        pool[at + scales_offset + o * sb..][..sb].copy_from_slice(&packed[r].1);
                        n32(e)
                    })
                    .collect();
                c.set("entries_per_block", epb);
                c.set("codes_offset", codes_offset);
                c.set("scales_offset", scales_offset);
                c.set("block_bytes", block_bytes);
                c.inputs.insert("targets", Buf::U32(targets));
                c.out_u8("pool", pool);
            } else {
                c.out_u8(
                    "dst_codes",
                    packed.iter().flat_map(|p| p.0.clone()).collect(),
                );
                c.out_u8(
                    "dst_scales",
                    packed.iter().flat_map(|p| p.1.clone()).collect(),
                );
            }
        }
        "rms_norm" => {
            let rows = sz(g, w2(2, 64), 3);
            let width = sz(g, w(5120), 100);
            let x: Vec<u16> = (0..rows).flat_map(|_| g.bf16s(width)).collect();
            let weight = g.f32s(width, 118, 8);
            let out = x
                .chunks_exact(width)
                .flat_map(|row| bits_of(&rms(&vals(row), &weight, EPS)))
                .collect();
            c.set("rows", rows);
            c.set("width", width);
            c.inputs.insert("x", Buf::U16(x));
            c.inputs.insert("weight", Buf::F32(weight));
            c.out_u16("out", out);
        }
        "mhc_mix" => {
            let rows = sz(g, w2(2, 64), 2);
            let copies = sz(g, w(4), 4);
            let dim = sz(g, w(5120), 40);
            let iters = sz(g, w(20), 20);
            let n = (2 + copies) * copies;
            let x: Vec<u16> = (0..rows).flat_map(|_| g.bf16s(copies * dim)).collect();
            let weights = mhc::MhcWeights {
                projection: Matrix::from_vec(n, copies * dim, g.f32s(n * copies * dim, 120, 4)),
                base: g.f32s(n, 122, 4),
                scale: [g.f32_in(124, 3), g.f32_in(124, 3), g.f32_in(124, 3)],
            };
            let out: Vec<f32> = x
                .chunks_exact(copies * dim)
                .flat_map(|row| {
                    let m = mhc::mixes(&weights, &vals(row), copies, EPS, EPS, iters);
                    [m.pre, m.post, m.comb].concat()
                })
                .collect();
            c.set("rows", rows);
            c.set("copies", copies);
            c.set("dim", dim);
            c.set("iters", iters);
            c.inputs.insert("x", Buf::U16(x));
            c.inputs
                .insert("projection", Buf::F32(weights.projection.data));
            c.inputs.insert("base", Buf::F32(weights.base));
            c.inputs.insert("scale", Buf::F32(weights.scale.to_vec()));
            c.out(
                "out",
                Buf::F32(vec![SENTINEL_F32; out.len()]),
                Buf::F32(out),
            );
        }
        "collapse" | "expand" | "stream_mean" => {
            let rows = sz(g, w2(2, 64), 3);
            let copies = sz(g, w(4), 4);
            let dim = sz(g, w(5120), 300);
            let x: Vec<u16> = (0..rows).flat_map(|_| g.bf16_wide(copies * dim)).collect();
            c.set("rows", rows);
            c.set("copies", copies);
            c.set("dim", dim);
            let xr = |r: usize| vals(&x[r * copies * dim..(r + 1) * copies * dim]);
            match kernel {
                "collapse" => {
                    let stride = copies + g.below(3);
                    let pre = g.f32s(rows * stride, 120, 6);
                    let mut out = Vec::new();
                    for r in 0..rows {
                        let mut o = vec![0.0; dim];
                        mhc::collapse(&xr(r), &pre[r * stride..][..copies], &mut o);
                        out.extend(bits_of(&o));
                    }
                    c.set("pre_stride", stride);
                    c.inputs.insert("pre", Buf::F32(pre));
                    c.out_u16("out", out);
                }
                "expand" => {
                    let y: Vec<u16> = (0..rows).flat_map(|_| g.bf16s(dim)).collect();
                    let stride = (2 + copies) * copies + g.below(3);
                    let mixes = g.f32s(rows * stride, 120, 6);
                    let mut out = Vec::new();
                    for r in 0..rows {
                        let row = &mixes[r * stride..];
                        let m = mhc::Mixes {
                            pre: row[..copies].to_vec(),
                            post: row[copies..2 * copies].to_vec(),
                            comb: row[2 * copies..(2 + copies) * copies].to_vec(),
                        };
                        let mut o = xr(r);
                        mhc::expand(&mut o, &vals(&y[r * dim..(r + 1) * dim]), &m);
                        out.extend(bits_of(&o));
                    }
                    c.set("mix_stride", stride);
                    c.inputs.insert("y", Buf::U16(y));
                    c.inputs.insert("mixes", Buf::F32(mixes));
                    c.out_u16("out", out);
                }
                _ => {
                    let stride = dim + g.below(3);
                    let mut out = vec![SENTINEL_U16; rows * stride];
                    for r in 0..rows {
                        out[r * stride..][..dim]
                            .copy_from_slice(&bits_of(&stream_mean(&xr(r), copies)));
                    }
                    c.set("out_stride", stride);
                    c.out_u16("out", out);
                }
            }
            c.inputs.insert("x", Buf::U16(x));
        }
        "linear" | "head" | "markov" => {
            let markov = kernel == "markov";
            let head = kernel == "head" || markov;
            let rows = sz(
                g,
                if head { w(9) } else { w2(2, 64) },
                if head { 20 } else { 3 },
            );
            let n = sz(
                g,
                if head { w2(100, 129_280) } else { w2(300, 512) },
                if head { 40 } else { 70 },
            );
            let k = sz(g, w(if markov { 256 } else { 5120 }), 100);
            let (x_f32, w_f32, out_f32) = (g.below(2) == 1, g.below(2) == 1, g.below(2) == 1);
            // The whole vocabulary's weight stays BF16 (half the memory).
            let w_f32 = w_f32 && !(head && regime == Regime::Released);
            let (indexed, added) = (g.below(2) == 1, g.below(2) == 1);
            // The Markov head reads each row's input through an index and
            // adds its products to the base logits.
            let (indexed, added) = (indexed || markov, added || markov);
            let x_rows = if indexed { rows + g.below(3) } else { rows };
            let fill = |g: &mut Rng, len: usize, f32s: bool| -> (Buf, Vec<f32>) {
                if f32s {
                    let v = g.f32s(len, 115, 10);
                    (Buf::F32(v.clone()), v)
                } else {
                    let v = g.bf16s(len);
                    let f = vals(&v);
                    (Buf::U16(v), f)
                }
            };
            let (x_buf, x) = fill(g, x_rows * k, x_f32);
            let (w_buf, wv) = fill(g, n * k, w_f32);
            let weight = Matrix::from_vec(n, k, wv);
            let index: Vec<u32> = (0..rows)
                .map(|r| {
                    if indexed {
                        n32(g.below(x_rows))
                    } else {
                        n32(r)
                    }
                })
                .collect();
            let addend = if added {
                g.f32s(rows * n, 115, 14)
            } else {
                Vec::new()
            };
            let mut out = vec![0.0f32; rows * n];
            for r in 0..rows {
                let o = &mut out[r * n..(r + 1) * n];
                linear_f32(&x[index[r] as usize * k..][..k], &weight, o);
                for (j, v) in o.iter_mut().enumerate() {
                    if added {
                        *v += addend[r * n + j];
                    }
                    if !out_f32 {
                        *v = bf16_round(*v);
                    }
                }
            }
            c.set("rows", rows);
            c.set("n", n);
            c.set("k", k);
            c.set("x_dtype", x_f32 as usize);
            c.set("w_dtype", w_f32 as usize);
            c.set("out_dtype", out_f32 as usize);
            c.inputs.insert("x", x_buf);
            c.inputs.insert("w", w_buf);
            if indexed {
                c.inputs.insert("x_rows", Buf::U32(index));
            }
            if added {
                c.inputs.insert("addend", Buf::F32(addend));
            }
            if out_f32 {
                c.out(
                    "out",
                    Buf::F32(vec![SENTINEL_F32; out.len()]),
                    Buf::F32(out),
                );
            } else {
                c.out_u16("out", bits_of(&out));
            }
        }
        "pool" => {
            let entries = sz(g, w2(3, 64), 4);
            let width = sz(g, w(512), 300);
            let src = 2 * entries + g.below(3);
            let kv = g.f32s(src * width, 115, 10);
            let score = g.f32s(src * width, 122, 8);
            let first: Vec<u32> = (0..entries).map(|_| n32(g.below(src))).collect();
            let second: Vec<u32> = (0..entries).map(|_| n32(g.below(src))).collect();
            let mut out = Vec::new();
            for e in 0..entries {
                let (a, b) = (first[e] as usize * width, second[e] as usize * width);
                for ch in 0..width {
                    out.push(bits(pooled(
                        [kv[a + ch], kv[b + ch]],
                        [score[a + ch], score[b + ch]],
                    )));
                }
            }
            c.set("entries", entries);
            c.set("width", width);
            c.inputs.insert("kv", Buf::F32(kv));
            c.inputs.insert("score", Buf::F32(score));
            c.inputs.insert("first", Buf::U32(first));
            c.inputs.insert("second", Buf::U32(second));
            c.out_u16("out", out);
        }
        "route" => {
            let rows = sz(g, w2(3, 64), 3);
            let experts = sz(g, w(384), 40);
            let active = if wide { 6 } else { 1 + g.below(experts.min(8)) };
            let normalize = g.below(2) == 1 && active > 1;
            let route_scale = if wide { 1.5 } else { g.f32_in(125, 3).abs() };
            // A quarter of the cases draw scores and biases from a few
            // values, so ties decide the selection; a quarter have scores
            // small enough for the normaliser's floor to matter.
            let mode = g.below(4);
            let (ties, low) = (mode == 0, mode == 1);
            let logits: Vec<f32> = (0..rows * experts)
                .map(|_| {
                    if ties {
                        (g.below(7) as f32 - 3.0) * 0.5
                    } else if low {
                        -g.f32_in(133, 1).abs()
                    } else {
                        g.f32_in(120, 10)
                    }
                })
                .collect();
            let bias: Vec<f32> = (0..experts)
                .map(|_| {
                    if ties {
                        g.below(3) as f32 * 0.25
                    } else {
                        g.f32_in(118, 6)
                    }
                })
                .collect();
            let spec = DsMoeSpec {
                n_routed: experts,
                source_n_routed: experts,
                n_active: active,
                inter_dim: 0,
                norm_topk_prob: normalize,
                route_scale,
                swiglu_limit: 0.0,
            };
            let (mut ids, mut weights) = (Vec::new(), Vec::new());
            for row in logits.chunks_exact(experts) {
                // A router whose products with the input 1.0 are the logits.
                let router = Matrix::from_vec(experts, 1, row.to_vec());
                let routing = route(&spec, &router, &bias, &[1.0]);
                let mut pairs: Vec<(usize, f32)> =
                    routing.experts.into_iter().zip(routing.weights).collect();
                pairs.sort_by_key(|p| p.0);
                ids.extend(pairs.iter().map(|p| n32(p.0)));
                weights.extend(pairs.iter().map(|p| p.1));
            }
            c.set("rows", rows);
            c.set("experts", experts);
            c.set("active", active);
            c.set("normalize", normalize as usize);
            c.f.insert("route_scale", route_scale);
            c.inputs.insert("logits", Buf::F32(logits));
            c.inputs.insert("bias", Buf::F32(bias));
            c.out(
                "ids",
                Buf::U32(vec![SENTINEL_U32; ids.len()]),
                Buf::U32(ids),
            );
            c.out(
                "weights",
                Buf::F32(vec![SENTINEL_F32; weights.len()]),
                Buf::F32(weights),
            );
        }
        "place" => {
            let rows = sz(g, w2(64, 8192), 6);
            let active = sz(g, w(6), 4);
            let groups = if wide { 384 } else { active + g.below(12) };
            let row_align = sz(g, w(16), 4);
            let ids: Vec<u32> = (0..rows)
                .flat_map(|_| g.ascending_ids(active, groups))
                .collect();
            let weights = g.f32s(rows * active, 120, 6);
            let layout = expert_layout(&ids, n32(groups), n32(row_align));
            let layout_rows = layout.rows as usize + g.below(3);
            let mut row_weight = vec![SENTINEL_F32; layout_rows];
            for (pair, row) in layout.row_of.iter().enumerate() {
                row_weight[*row as usize] = weights[pair];
            }
            c.set("rows", rows);
            c.set("active", active);
            c.set("groups", groups);
            c.set("row_align", row_align);
            c.set("layout_rows", layout_rows);
            c.inputs.insert("ids", Buf::U32(ids));
            c.inputs.insert("weights", Buf::F32(weights));
            c.out(
                "row_of",
                Buf::U32(vec![SENTINEL_U32; rows * active]),
                Buf::U32(layout.row_of),
            );
            c.out(
                "row_weight",
                Buf::F32(vec![SENTINEL_F32; layout_rows]),
                Buf::F32(row_weight),
            );
            c.out(
                "group_end",
                Buf::U32(vec![SENTINEL_U32; groups]),
                Buf::U32(layout.group_end),
            );
        }
        "swiglu" => {
            let rows = sz(g, w2(2, 64), 3);
            let width = 32 * sz(g, w(72), 4);
            let in_stride = width + 32 * g.below(2);
            let base = 120 + n32(g.below(12));
            let gate = g.bf16_range(rows * in_stride, base, 8);
            let up = g.bf16_range(rows * in_stride, base, 8);
            let limit = if g.below(2) == 0 { 0.0 } else { 10.0 };
            let weighted = g.below(2) == 1;
            let weight = g.f32s(rows, 120, 8);
            let (mut codes, mut scales) = (Vec::new(), Vec::new());
            for r in 0..rows {
                let h = swiglu_hidden(
                    &vals(&gate[r * in_stride..][..width]),
                    &vals(&up[r * in_stride..][..width]),
                    limit,
                    weighted.then_some(weight[r]),
                );
                let (cs, ss) = quant_fp8(&h);
                codes.extend(cs);
                scales.extend(ss);
            }
            c.set("rows", rows);
            c.set("width", width);
            c.set("in_stride", in_stride);
            c.f.insert("limit", limit);
            c.inputs.insert("gate", Buf::U16(gate));
            c.inputs.insert("up", Buf::U16(up));
            if weighted {
                c.inputs.insert("weight", Buf::F32(weight));
            }
            c.out_u8("codes", codes);
            c.out_u8("scales", scales);
        }
        "rotate" => {
            // The rotation alone (`KV_BF16`): the attention queries.
            let rows = sz(g, w2(2, 64), 3);
            let heads = sz(g, w2(4, 64), 3);
            let head_dim = 32 * sz(g, w(16), 4);
            let rope_dim = if wide {
                64
            } else {
                2 * (1 + g.below(head_dim / 2))
            };
            let rotation = g.below(3);
            let cis_positions = 4 + g.below(8);
            let cis = g.f32s(cis_positions * rope_dim, 117, 10);
            let positions: Vec<u32> = (0..rows).map(|_| n32(g.below(cis_positions))).collect();
            let width = heads * head_dim;
            let x = g.bf16_blocks(rows * width);
            let mut out = Vec::new();
            for r in 0..rows {
                let mut v = vals(&x[r * width..(r + 1) * width]);
                let pairs: Vec<(f32, f32)> = cis[positions[r] as usize * rope_dim..][..rope_dim]
                    .chunks_exact(2)
                    .map(|p| (p[0], p[1]))
                    .collect();
                for head in v.chunks_exact_mut(head_dim) {
                    match n32(rotation) {
                        abi::ROT_ROTATE => rotate(head, &pairs),
                        abi::ROT_UNROTATE => unrotate(head, &pairs),
                        _ => {}
                    }
                }
                out.extend(bits_of(&v));
            }
            c.set("rows", rows);
            c.set("heads", heads);
            c.set("head_dim", head_dim);
            c.set("rope_dim", rope_dim);
            c.set("rotation", rotation);
            c.set("format", abi::KV_BF16 as usize);
            c.set("cis_positions", cis_positions);
            c.inputs.insert("x", Buf::U16(x));
            c.inputs.insert("cis", Buf::F32(cis));
            c.inputs.insert("positions", Buf::U32(positions));
            c.out_u16("dst_codes", out);
        }
        "place_rows" => {
            let rows = sz(g, w2(3, 64), 4);
            let active = sz(g, w(6), 6);
            let row_bytes = sz(g, w(160), 70);
            let layout_rows = rows * active + g.below(4);
            // Distinct layout rows; one pair in eight names a row past the layout.
            let mut free: Vec<usize> = (0..layout_rows).collect();
            let mut status = 0;
            let row_of: Vec<u32> = (0..rows * active)
                .map(|_| {
                    let to = free.swap_remove(g.below(free.len()));
                    if g.below(8) == 0 {
                        status = abi::STATUS_BAD_INDEX;
                        return n32(layout_rows + g.below(3));
                    }
                    n32(to)
                })
                .collect();
            let src = g.bytes(rows * row_bytes);
            let mut dst = vec![SENTINEL_U8; layout_rows * row_bytes];
            for (i, to) in row_of.iter().enumerate() {
                let to = *to as usize;
                if to < layout_rows {
                    dst[to * row_bytes..(to + 1) * row_bytes]
                        .copy_from_slice(&src[i / active * row_bytes..][..row_bytes]);
                }
            }
            c.set("rows", rows);
            c.set("active", active);
            c.set("row_bytes", row_bytes);
            c.set("layout_rows", layout_rows);
            c.inputs.insert("row_of", Buf::U32(row_of));
            c.inputs.insert("src", Buf::U8(src));
            c.out_u8("dst", dst);
            c.out("status", Buf::U32(vec![0]), Buf::U32(vec![status]));
        }
        "combine" => {
            let rows = sz(g, w2(3, 64), 4);
            let active = sz(g, w(6), 6);
            let dim = sz(g, w(5120), 300);
            let layout_rows = rows * active + g.below(4);
            let row_of: Vec<u32> = (0..rows * active)
                .map(|_| n32(g.below(layout_rows)))
                .collect();
            let routed = g.bf16_wide(layout_rows * dim);
            let shared = g.bf16_wide(rows * dim);
            let mut out = Vec::new();
            for r in 0..rows {
                for d in 0..dim {
                    let mut y = 0.0f32;
                    for slot in 0..active {
                        y += val(routed[row_of[r * active + slot] as usize * dim + d]);
                    }
                    out.push(bits(bf16_round(y + val(shared[r * dim + d]))));
                }
            }
            c.set("rows", rows);
            c.set("active", active);
            c.set("dim", dim);
            c.inputs.insert("row_of", Buf::U32(row_of));
            c.inputs.insert("routed", Buf::U16(routed));
            c.inputs.insert("shared", Buf::U16(shared));
            c.out_u16("out", out);
        }
        "engram_gather" => {
            let table_rows = 2 + g.below(20);
            let width = 32 * sz(g, w(8), 3);
            let rows = sz(g, w2(2, 64), 3);
            let ids_per_row = sz(g, w(24), 4);
            let codes = g.bytes(table_rows * width);
            let scales = g.bytes(table_rows * width / 32);
            let mut status = 0;
            let (mut out_codes, mut out_scales) = (Vec::new(), Vec::new());
            let ids: Vec<u32> = (0..rows * ids_per_row)
                .map(|_| {
                    let mut id = g.below(table_rows);
                    if g.below(8) == 0 {
                        id = table_rows + g.below(3);
                    }
                    if id < table_rows {
                        out_codes.extend_from_slice(&codes[id * width..(id + 1) * width]);
                        out_scales.extend_from_slice(&scales[id * width / 32..][..width / 32]);
                    } else {
                        status = abi::STATUS_BAD_INDEX;
                        out_codes.extend(std::iter::repeat_n(0, width));
                        out_scales.extend(std::iter::repeat_n(0, width / 32));
                    }
                    n32(id)
                })
                .collect();
            c.set("table_rows", table_rows);
            c.set("width", width);
            c.set("rows", rows);
            c.set("ids_per_row", ids_per_row);
            c.inputs.insert("codes", Buf::U8(codes));
            c.inputs.insert("scales", Buf::U8(scales));
            c.inputs.insert("ids", Buf::U32(ids));
            c.out_u8("out_codes", out_codes);
            c.out_u8("out_scales", out_scales);
            c.out("status", Buf::U32(vec![0]), Buf::U32(vec![status]));
        }
        "engram_inject" => {
            let rows = sz(g, w2(2, 64), 2);
            let copies = sz(g, w(4), 4);
            let dim = sz(g, w(5120), 120);
            let x: Vec<u16> = (0..rows).flat_map(|_| g.bf16s(copies * dim)).collect();
            let kv: Vec<u16> = (0..rows)
                .flat_map(|_| g.bf16s((copies + 1) * dim))
                .collect();
            let gate = g.f32s(copies * dim, 118, 8);
            // One case in four holds a NaN in a key (never in a value or the stream):
            // its gate is the one a `+0` product gives, whatever the NaN.
            let mut kv = kv;
            if g.below(4) == 0 {
                let at = g.below(copies * dim);
                kv[at] = [0x7fc0, 0xffc0, 0x7f81, 0xffa1][g.below(4)];
            }
            let mut out = Vec::new();
            for r in 0..rows {
                let mut xr = vals(&x[r * copies * dim..(r + 1) * copies * dim]);
                let kvr = vals(&kv[r * (copies + 1) * dim..(r + 1) * (copies + 1) * dim]);
                inject_after_projection(&mut xr, &kvr, &gate, copies, EPS);
                out.extend(bits_of(&xr));
            }
            c.set("rows", rows);
            c.set("copies", copies);
            c.set("dim", dim);
            c.inputs.insert("x", Buf::U16(x));
            c.inputs.insert("kv", Buf::U16(kv));
            c.inputs.insert("gate_weight", Buf::F32(gate));
            c.out_u16("out", out);
        }
        "lin8" => {
            let rows = sz(g, w2(2, 64), 3);
            let n = sz(g, w2(300, 32_768), 70);
            let k = 32 * sz(g, w2(160, 40), 4);
            let x = g.bf16_blocks(rows * k);
            let w_codes = g.codes(n * k);
            let w_scales = g.scales(n.div_ceil(32) * (k / 32));
            let weight = Fp8Linear::from_bytes(n, k, &w_codes, &w_scales);
            let xv = vals(&x);
            let mut out = Vec::new();
            for row in xv.chunks_exact(k) {
                let mut o = vec![0.0; n];
                lin8(row, &weight, &mut o);
                out.extend(bits_of(&o));
            }
            let (codes, scales) = quant_fp8(&xv);
            c.set("rows", rows);
            c.set("n", n);
            c.set("k", k);
            c.inputs.insert("x", Buf::U16(x));
            c.inputs.insert("w_codes", Buf::U8(w_codes));
            c.inputs.insert("w_scales", Buf::U8(w_scales));
            c.out_u8("codes", codes);
            c.out_u8("act_scales", scales);
            c.out_u16("out", out);
        }
        "lin4" => {
            let tokens = sz(g, w2(4, 64), 4);
            let active = sz(g, w(3), 3);
            let groups = if wide { 8 } else { active + g.below(5) };
            let row_align = 1 + g.below(4);
            let n = sz(g, w2(300, 2304), 40);
            let k = 32 * sz(g, w2(72, 160), 3);
            let ids: Vec<u32> = (0..tokens)
                .flat_map(|_| g.ascending_ids(active, groups))
                .collect();
            let x = g.bf16_blocks(tokens * k);
            let w_packed = g.bytes(groups * n * k / 2);
            let w_scales = g.scales(groups * n * (k / 32));
            let layout = expert_layout(&ids, n32(groups), n32(row_align));
            let layout_rows = layout.rows as usize + g.below(3);
            let xv = vals(&x);
            let (codes, scales) = quant_fp8(&xv);
            let kb = k / 32;
            let mut layout_codes = vec![0u8; layout_rows * k];
            let mut layout_scales = vec![0u8; layout_rows * kb];
            let mut out = vec![SENTINEL_U16; layout_rows * n];
            for (pair, row) in layout.row_of.iter().enumerate() {
                let (token, row) = (pair / active, *row as usize);
                layout_codes[row * k..][..k].copy_from_slice(&codes[token * k..][..k]);
                layout_scales[row * kb..][..kb].copy_from_slice(&scales[token * kb..][..kb]);
                let e = ids[pair] as usize;
                let weight = Fp4Linear::from_bytes(
                    n,
                    k,
                    &w_packed[e * n * k / 2..][..n * k / 2],
                    &w_scales[e * n * kb..][..n * kb],
                );
                let mut o = vec![0.0; n];
                lin4(&xv[token * k..][..k], &weight, &mut o);
                out[row * n..][..n].copy_from_slice(&bits_of(&o));
            }
            c.set("tokens", tokens);
            c.set("active", active);
            c.set("groups", groups);
            c.set("row_align", row_align);
            c.set("n", n);
            c.set("k", k);
            c.set("layout_rows", layout_rows);
            c.inputs.insert("ids", Buf::U32(ids));
            c.inputs.insert("x", Buf::U16(x));
            c.inputs.insert("w_packed", Buf::U8(w_packed));
            c.inputs.insert("w_scales", Buf::U8(w_scales));
            c.inputs.insert("codes", Buf::U8(layout_codes));
            c.inputs.insert("act_scales", Buf::U8(layout_scales));
            c.inputs.insert("group_end", Buf::U32(layout.group_end));
            c.out_u16("out", out);
        }
        "grouped_dot" => {
            let rows = sz(g, w2(2, 64), 2);
            let groups = sz(g, w2(2, 8), 3);
            let rank = sz(g, w2(70, 1024), 70);
            let k = 32 * sz(g, w(128), 4);
            let o = g.bf16_blocks(rows * groups * k);
            let w_codes = g.codes(groups * rank * k);
            let w_scales = g.scales((groups * rank).div_ceil(32) * (k / 32));
            let weight =
                Fp8Linear::from_bytes(groups * rank, k, &w_codes, &w_scales).dequantised_bf16();
            let ov = vals(&o);
            let (codes, scales) = quant_fp8(&ov);
            let mut faked = ov;
            let mut out = Vec::new();
            for row in faked.chunks_exact_mut(groups * k) {
                fake_quant_fp8(row);
                for z in 0..groups * rank {
                    let slice = &row[z / rank * k..][..k];
                    out.push(bits(bf16_round(dot(slice, weight.row(z)))));
                }
            }
            c.set("rows", rows);
            c.set("groups", groups);
            c.set("rank", rank);
            c.set("k", k);
            c.inputs.insert("x", Buf::U16(o));
            c.inputs.insert("w_codes", Buf::U8(w_codes));
            c.inputs.insert("w_scales", Buf::U8(w_scales));
            c.inputs.insert("codes", Buf::U8(codes));
            c.inputs.insert("act_scales", Buf::U8(scales));
            c.out_u16("out", out);
        }
        _ => panic!("no kernel {kernel}"),
    }
    c
}

/// The sweeps of one vector file: per kernel, `2^SMALL_LOG2` small cases and
/// `2^WIDE_LOG2` at the released widths.
pub const SEED: u64 = 0x5eed;
pub const SMALL_LOG2: u32 = 9;
pub const WIDE_LOG2: u32 = 1;

/// Every case of a sweep, in order.
pub fn sweep(kernel: &'static str, wide: bool) -> Vec<Case> {
    let mut g = Rng(SEED);
    let (log2, regime) = if wide {
        (WIDE_LOG2, Regime::Wide)
    } else {
        (SMALL_LOG2, Regime::Small)
    };
    (0..1u32 << log2)
        .map(|_| build(kernel, regime, &mut g))
        .collect()
}

/// The vector file of `kernels`: one line per sweep,
/// `s <kernel> <s|w> <seed> <log2 cases> = <digest>`, every number
/// hexadecimal. A digest starts at 0 and takes each output word `w`, case
/// after case and output after output, as `h = fin(h + GOLDEN + w)`
/// (`fin` SplitMix64's finalizer), as the numerics vectors' do.
pub fn vectors(kernels: &[&'static str]) -> String {
    let mut out = String::from(
        "# DeepSeek-V4.1 kernel vectors. Generated by eidola-engine-cuda's tests/ds_cases,\n\
         # which documents the format.\n",
    );
    for kernel in kernels {
        for wide in [false, true] {
            let mut h = 0u64;
            let cases = sweep(kernel, wide);
            for case in &cases {
                case.digest_into(&mut h);
            }
            let log2 = cases.len().trailing_zeros();
            let regime = if wide { 'w' } else { 's' };
            writeln!(out, "s {kernel} {regime} {SEED:x} {log2:x} = {h:x}").unwrap();
        }
    }
    out
}

/// Require the committed vector file to be what [`vectors`] emits, or rewrite
/// it under `EIDOLA_UPDATE_DS_VECTORS=1`.
pub fn check_or_update(file: &str, kernels: &[&'static str]) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../eidola-engine-kernels/tests/data")
        .join(file);
    let generated = vectors(kernels);
    if std::env::var_os("EIDOLA_UPDATE_DS_VECTORS").is_some() {
        std::fs::write(&path, generated).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        return;
    }
    let committed =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(
        committed == generated,
        "{} is not what the reference emits; rerun with EIDOLA_UPDATE_DS_VECTORS=1 and re-run \
         the kernels crate's host check",
        path.display()
    );
}
