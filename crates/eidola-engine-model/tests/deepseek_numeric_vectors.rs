//! Golden vectors of the DeepSeek-V4.1 numerics, emitted from the reference
//! functions for an implementation in another language to be checked against
//! bit for bit: the GPU kernels' numerics header in `eidola-engine-kernels`
//! (`csrc/ds_numeric.cuh`, checked on a host by `oracle/ds_numeric_host.cc`).
//!
//! `vectors_are_current` regenerates `ds_numeric.vectors` and requires the
//! committed file to equal it; `EIDOLA_UPDATE_DS_VECTORS=1` rewrites the file
//! instead. The ignored `exhaustive_digests_are_current` does the same for
//! `ds_numeric.exhaustive` (every f32 through each one-argument function).
//!
//! # File format
//!
//! One record per line, tokens separated by spaces, every number hexadecimal:
//!
//! - `c <name> <bits>`: a constant.
//! - `v <fn> <inputs…> = <outputs…>`: one case.
//! - `s <fn> <seed> <log2 items> = <digest>`: `2^n` cases drawn from the
//!   generator the function's sweep uses (below), all outputs folded in order
//!   into one digest.
//! - `x <fn> = <digest>` (the exhaustive file): the wrapping sum over every
//!   u32 `b` of `fin((b << 32) | out(b))`.
//!
//! f32 values are their 32 bits and f64 values their 64. A NaN output is
//! written canonically (`7fc00000`, `7ff8000000000000`): which NaN an
//! operation returns differs between processors.
//!
//! The generator is SplitMix64 (`state += 0x9e3779b97f4a7c15`, then the
//! finalizer `fin`); a digest starts at 0 and takes each output word `w` as
//! `h = fin(h + 0x9e3779b97f4a7c15 + w)`. Each sweep's draws are spelled out
//! in `sweep_item`; the host check mirrors them.

use std::fmt::Write as _;
use std::path::PathBuf;

use eidola_engine_model::deepseek::attention::top_k;
use eidola_engine_model::deepseek::linear::{Fp4Linear, Fp8Linear, lin4, lin8};
use eidola_engine_model::deepseek::moe::swiglu_hidden;
use eidola_engine_model::deepseek::numeric::{
    bf16_round, det_exp, det_ln, det_sin_cos, exp32, pinned_sum, pow32, round_e2m1, round_e4m3,
    round_scale, rsqrt32, sigmoid32, silu32, softplus32,
};
use eidola_engine_model::deepseek::quant::{
    act_quant, fake_quant_fp4_e4m3, fake_quant_fp4_pow2, fake_quant_fp8,
};
use eidola_engine_model::deepseek::rope::{RopeTable, Yarn, rotate, unrotate};
use eidola_engine_model::deepseek::weights::rms;
use eidola_engine_model::numeric::{E2M1_VALUES, e8m0_to_f32, fp8_e4m3_to_f32};
use eidola_engine_model::tensor::dot;
use rayon::prelude::*;

const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;

fn fin(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GOLDEN);
        fin(self.0)
    }
    /// A uniform draw below `n`, as `next() % n`.
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// An f32 of any sign with exponent field `base + (0..spread)` and a
    /// random mantissa, from one draw.
    fn f32_in(&mut self, base: u32, spread: u32) -> f32 {
        let r = self.next();
        let exp = base + ((r >> 32) & 0xff) as u32 % spread;
        f32::from_bits(((r >> 63) as u32) << 31 | exp << 23 | (r & 0x7f_ffff) as u32)
    }
    /// A moderate f32: magnitude in `[2^-24, 2^24)`.
    fn moderate(&mut self) -> f32 {
        self.f32_in(103, 48)
    }
    /// An f64 of any sign with exponent field `base + (0..spread)`.
    fn f64_in(&mut self, base: u64, spread: u64) -> f64 {
        let r = self.next();
        let exp = base + self.below(spread);
        f64::from_bits((r >> 63) << 63 | exp << 52 | (r & 0x000f_ffff_ffff_ffff))
    }
    /// A block of activations sharing a magnitude range: exponent fields
    /// `base .. base + 12` with `base` in `1..=230`.
    fn block(&mut self, n: usize) -> Vec<f32> {
        let base = 1 + self.below(230) as u32;
        (0..n).map(|_| self.f32_in(base, 12)).collect()
    }
    /// An E4M3 code that is not NaN.
    fn code(&mut self) -> u8 {
        let b = (self.next() & 0xff) as u8;
        if b & 0x7f == 0x7f { b - 1 } else { b }
    }
}

fn w32(x: f32) -> u64 {
    if x.is_nan() {
        0x7fc0_0000
    } else {
        x.to_bits() as u64
    }
}

fn w64(x: f64) -> u64 {
    if x.is_nan() {
        0x7ff8_0000_0000_0000
    } else {
        x.to_bits()
    }
}

fn f(bits: u64) -> f32 {
    f32::from_bits(bits as u32)
}

fn words32(x: &[f32]) -> Vec<u64> {
    x.iter().map(|v| w32(*v)).collect()
}

fn bits32(x: &[f32]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits() as u64).collect()
}

type Unary = (&'static str, fn(f32) -> f32);

/// The one-argument f32 functions, by name.
const UNARY: [Unary; 8] = [
    ("bf16_round", bf16_round),
    ("round_e4m3", round_e4m3),
    ("round_e2m1", round_e2m1),
    ("exp32", exp32),
    ("sigmoid32", sigmoid32),
    ("silu32", silu32),
    ("softplus32", softplus32_or_nan),
    ("rsqrt32", rsqrt32),
];

/// `softplus32` refuses a NaN (its logarithm's precondition); the vectors
/// record a NaN there.
fn softplus32_or_nan(x: f32) -> f32 {
    if x.is_nan() { x } else { softplus32(x) }
}

fn fp8_weight(rows: usize, cols: usize, codes: &[u8], scales: &[u8]) -> Fp8Linear {
    Fp8Linear::from_bytes(rows, cols, codes, scales)
}

/// The exact dot product of 32 E4M3 codes with 32 E4M3 codes, rounded once to
/// f32: the rule `linear`'s block product implements. Checked against `lin8`
/// on the same block, whose activations quantise to themselves at scale 1.
fn fp8_block_dot(a: &[u8], w: &[u8]) -> f32 {
    assert!(a[0] & 0x7f == 0x7e, "the block's first code is ±448");
    let int = |c: u8| (fp8_e4m3_to_f32(c) as f64 * 512.0) as i64;
    let sum: i64 = a.iter().zip(w).map(|(a, w)| int(*a) * int(*w)).sum();
    let d = (sum as f64 / 262144.0) as f32;
    let x: Vec<f32> = a.iter().map(|c| fp8_e4m3_to_f32(*c)).collect();
    let mut out = [0.0f32];
    lin8(&x, &fp8_weight(1, 32, w, &[127]), &mut out);
    assert_eq!(out[0].to_bits(), bf16_round(d).to_bits());
    d
}

/// The same for 32 E2M1 elements (16 packed bytes), against `lin4`.
fn fp4_block_dot(a: &[u8], packed: &[u8]) -> f32 {
    assert!(a[0] & 0x7f == 0x7e, "the block's first code is ±448");
    let mut sum = 0i64;
    for (j, c) in a.iter().enumerate() {
        let nibble = (packed[j / 2] >> ((j & 1) * 4)) & 0x0f;
        sum += (fp8_e4m3_to_f32(*c) as f64 * 512.0) as i64
            * (E2M1_VALUES[nibble as usize] * 2.0) as i64;
    }
    let d = (sum as f64 / 1024.0) as f32;
    let x: Vec<f32> = a.iter().map(|c| fp8_e4m3_to_f32(*c)).collect();
    let mut out = [0.0f32];
    lin4(&x, &Fp4Linear::from_bytes(1, 32, packed, &[127]), &mut out);
    assert_eq!(out[0].to_bits(), bf16_round(d).to_bits());
    d
}

/// A function's case as `(inputs, outputs)` words. `eval` computes the
/// outputs of explicit inputs; `sweep_item` draws inputs.
fn eval(name: &str, i: &[u64]) -> Vec<u64> {
    if let Some((_, func)) = UNARY.iter().find(|(n, _)| *n == name) {
        return vec![w32(func(f(i[0])))];
    }
    match name {
        "e4m3_decode" => vec![w32(fp8_e4m3_to_f32(i[0] as u8))],
        "e2m1_decode" => vec![w32(E2M1_VALUES[i[0] as usize])],
        "e8m0_decode" => vec![w32(e8m0_to_f32(i[0] as u8))],
        "round_scale" => vec![w32(round_scale(f(i[0]), f(i[1])))],
        "det_exp" => vec![w64(det_exp(f64::from_bits(i[0])))],
        "det_ln" => vec![w64(det_ln(f64::from_bits(i[0])))],
        "det_sin_cos" => {
            let (s, c) = det_sin_cos(f64::from_bits(i[0]));
            vec![w64(s), w64(c)]
        }
        "pow32" => vec![w32(pow32(f64::from_bits(i[0]), f(i[1])))],
        // 32 activations → 32 code values, then the block scale.
        "act_quant" => {
            let x: Vec<f32> = i.iter().map(|b| f(*b)).collect();
            let (mut codes, mut scales) = (vec![0.0; 32], vec![0.0; 1]);
            act_quant(&x, &mut codes, &mut scales);
            let mut out = words32(&codes);
            out.push(w32(scales[0]));
            out
        }
        "fq8" | "fq4m" | "fq4n" => {
            let mut x: Vec<f32> = i.iter().map(|b| f(*b)).collect();
            match name {
                "fq8" => fake_quant_fp8(&mut x),
                "fq4m" => fake_quant_fp4_pow2(&mut x),
                _ => fake_quant_fp4_e4m3(&mut x),
            }
            words32(&x)
        }
        // 32 activation codes, 32 weight codes → the block dot.
        "fp8_dot" => {
            let b: Vec<u8> = i.iter().map(|v| *v as u8).collect();
            vec![w32(fp8_block_dot(&b[..32], &b[32..]))]
        }
        // 32 activation codes, 16 packed weight bytes.
        "fp4_dot" => {
            let b: Vec<u8> = i.iter().map(|v| *v as u8).collect();
            vec![w32(fp4_block_dot(&b[..32], &b[32..]))]
        }
        // rows, blocks, then x[32·blocks], codes[rows · 32·blocks],
        // scales[ceil(rows / 32) · blocks] → out[rows].
        "lin8" => {
            let (rows, k) = (i[0] as usize, i[1] as usize * 32);
            let x: Vec<f32> = i[2..2 + k].iter().map(|b| f(*b)).collect();
            let rest: Vec<u8> = i[2 + k..].iter().map(|v| *v as u8).collect();
            let w = fp8_weight(rows, k, &rest[..rows * k], &rest[rows * k..]);
            let mut out = vec![0.0; rows];
            lin8(&x, &w, &mut out);
            words32(&out)
        }
        // rows, blocks, then x, packed[rows · 16·blocks], scales[rows · blocks].
        "lin4" => {
            let (rows, k) = (i[0] as usize, i[1] as usize * 32);
            let x: Vec<f32> = i[2..2 + k].iter().map(|b| f(*b)).collect();
            let rest: Vec<u8> = i[2 + k..].iter().map(|v| *v as u8).collect();
            let w = Fp4Linear::from_bytes(rows, k, &rest[..rows * k / 2], &rest[rows * k / 2..]);
            let mut out = vec![0.0; rows];
            lin4(&x, &w, &mut out);
            words32(&out)
        }
        // n, a[n], b[n].
        "dot" => {
            let n = i[0] as usize;
            let a: Vec<f32> = i[1..1 + n].iter().map(|b| f(*b)).collect();
            let b: Vec<f32> = i[1 + n..].iter().map(|b| f(*b)).collect();
            vec![w32(dot(&a, &b))]
        }
        // n, x[n].
        "pinned_sum" => {
            let x: Vec<f32> = i[1..].iter().map(|b| f(*b)).collect();
            vec![w32(pinned_sum(&x))]
        }
        // a, b, cos, sin.
        "rotate" | "unrotate" => {
            let mut x = [f(i[0]), f(i[1])];
            let cis = [(f(i[2]), f(i[3]))];
            if name == "rotate" {
                rotate(&mut x, &cis);
            } else {
                unrotate(&mut x, &cis);
            }
            words32(&x)
        }
        // position, frequency → cos, sin.
        "cis" => {
            let table = RopeTable {
                freqs: vec![f(i[1])],
            };
            let (c, s) = table.cis(i[0] as usize)[0];
            vec![w32(c), w32(s)]
        }
        // n, eps, x[n], weight[n].
        "rms" => {
            let n = i[0] as usize;
            let x: Vec<f32> = i[2..2 + n].iter().map(|b| f(*b)).collect();
            let w: Vec<f32> = i[2 + n..].iter().map(|b| f(*b)).collect();
            words32(&rms(&x, &w, f(i[1])))
        }
        // gate, up, limit, weighted, weight.
        "swiglu" => {
            let weight = (i[3] != 0).then(|| f(i[4]));
            words32(&swiglu_hidden(&[f(i[0])], &[f(i[1])], f(i[2]), weight))
        }
        // n, k, scores[n] → the chosen indices, best first.
        "top_k" => {
            let scores: Vec<f32> = i[2..].iter().map(|b| f(*b)).collect();
            top_k(&scores, 0..i[0] as usize, i[1] as usize)
                .into_iter()
                .map(|t| t as u64)
                .collect()
        }
        _ => panic!("no function {name}"),
    }
}

/// The inputs of one case of `name`'s sweep.
fn sweep_item(name: &str, g: &mut Rng) -> Vec<u64> {
    match name {
        // Any bit pattern on even items, a moderate value on odd ones.
        "bf16_round" | "round_e4m3" | "round_e2m1" | "exp32" | "sigmoid32" | "silu32"
        | "softplus32" | "rsqrt32" => {
            let raw = g.next();
            let moderate = g.moderate();
            vec![if raw >> 63 == 0 {
                raw & 0xffff_ffff
            } else {
                moderate.to_bits() as u64
            }]
        }
        "round_scale" => {
            let amax = g.f32_in(1, 254).abs();
            let inv = if g.next() & 1 == 0 {
                (1.0f64 / 448.0) as f32
            } else {
                (1.0f64 / 6.0) as f32
            };
            bits32(&[amax, inv])
        }
        // Magnitudes from 2^-30 to 2^12, so both cut-offs are crossed.
        "det_exp" => vec![g.f64_in(993, 42).to_bits()],
        // Any positive normal.
        "det_ln" => vec![g.f64_in(1, 2046).abs().to_bits()],
        // Magnitudes below 2^20.
        "det_sin_cos" => vec![g.f64_in(983, 60).to_bits()],
        "pow32" => {
            let base = g.f64_in(1013, 30).abs();
            vec![base.to_bits(), g.f32_in(110, 20).to_bits() as u64]
        }
        "act_quant" | "fq8" | "fq4m" => bits32(&g.block(32)),
        "fq4n" => bits32(&g.block(16)),
        "fp8_dot" | "fp4_dot" => {
            let mut v = vec![0x7e | (g.next() & 0x80)];
            v.extend((1..32).map(|_| g.code() as u64));
            if name == "fp8_dot" {
                v.extend((0..32).map(|_| g.code() as u64));
            } else {
                v.extend((0..16).map(|_| g.next() & 0xff));
            }
            v
        }
        "lin8" | "lin4" => {
            let rows = 1 + g.below(40);
            let blocks = 1 + g.below(4);
            let mut v = vec![rows, blocks];
            for _ in 0..blocks {
                v.extend(bits32(&g.block(32)));
            }
            let (rows, k) = (rows as usize, blocks as usize * 32);
            if name == "lin8" {
                v.extend((0..rows * k).map(|_| g.code() as u64));
                v.extend((0..rows.div_ceil(32) * (k / 32)).map(|_| 100 + g.below(56)));
            } else {
                v.extend((0..rows * k / 2).map(|_| g.next() & 0xff));
                v.extend((0..rows * (k / 32)).map(|_| 100 + g.below(56)));
            }
            v
        }
        "dot" => {
            let n = g.below(97);
            let mut v = vec![n];
            v.extend((0..2 * n).map(|_| g.moderate().to_bits() as u64));
            v
        }
        "pinned_sum" => {
            let n = g.below(97);
            let mut v = vec![n];
            v.extend((0..n).map(|_| g.moderate().to_bits() as u64));
            v
        }
        "rotate" | "unrotate" => (0..4).map(|_| g.moderate().to_bits() as u64).collect(),
        "cis" => vec![g.below(1 << 20), g.f32_in(103, 24).abs().to_bits() as u64],
        "rms" => {
            let n = 1 + g.below(48) as usize;
            let mut v = vec![n as u64, 1e-6f32.to_bits() as u64];
            v.extend(bits32(&g.block(n)));
            v.extend((0..n).map(|_| g.moderate().to_bits() as u64));
            v
        }
        "swiglu" => {
            let (gate, up) = (g.f32_in(117, 16), g.f32_in(117, 16));
            let limit = if g.next() & 1 == 0 { 0.0f32 } else { 10.0 };
            let weighted = g.next() & 1;
            let weight = g.f32_in(117, 12);
            let mut v = bits32(&[gate, up, limit]);
            v.push(weighted);
            v.push(weight.to_bits() as u64);
            v
        }
        // Few distinct BF16 scores, so ties are the rule; signed zeros and
        // NaNs of both signs among them.
        "top_k" => {
            let n = 1 + g.below(64);
            let k = 1 + g.below(n);
            let mut v = vec![n, k];
            for _ in 0..n {
                v.push(match g.below(12) {
                    0 => 0x8000_0000,
                    1 => 0x7fc0_0000,
                    2 => 0xffc0_0000,
                    r => ((r as f32) - 6.0).to_bits() as u64,
                });
            }
            v
        }
        _ => panic!("no sweep for {name}"),
    }
}

fn sweep(out: &mut String, name: &str, seed: u64, log2: u32) {
    let mut g = Rng(seed);
    let mut h = 0u64;
    for _ in 0..1u64 << log2 {
        for w in eval(name, &sweep_item(name, &mut g)) {
            h = fin(h.wrapping_add(GOLDEN).wrapping_add(w));
        }
    }
    writeln!(out, "s {name} {seed:x} {log2:x} = {h:x}").unwrap();
}

fn case(out: &mut String, name: &str, inputs: &[u64]) {
    write!(out, "v {name}").unwrap();
    for i in inputs {
        write!(out, " {i:x}").unwrap();
    }
    out.push_str(" =");
    for o in eval(name, inputs) {
        write!(out, " {o:x}").unwrap();
    }
    out.push('\n');
}

fn b(x: f32) -> u64 {
    x.to_bits() as u64
}

/// `x` and its two f32 neighbours, with both signs.
fn around(x: f32) -> Vec<u64> {
    let bits = x.abs().to_bits();
    let mut v = Vec::new();
    for near in [bits.saturating_sub(1), bits, bits + 1] {
        v.push(near as u64);
        v.push((near | 0x8000_0000) as u64);
    }
    v
}

const SPECIAL32: [u32; 16] = [
    0x0000_0000, // 0
    0x8000_0000, // -0
    0x0000_0001, // the smallest subnormal
    0x007f_ffff, // the largest subnormal
    0x0080_0000, // the smallest normal
    0x8080_0000,
    0x3f80_0000, // 1
    0xbf80_0000,
    0x7f7f_ffff, // the largest finite
    0xff7f_ffff,
    0x7f80_0000, // infinities
    0xff80_0000,
    0x7fc0_0000, // NaNs, quiet and signalling, both signs
    0xffc0_0000,
    0x7f80_0001,
    0xffa1_2345,
];

fn generate() -> String {
    let mut out = String::new();
    out.push_str("# DeepSeek-V4.1 numerics vectors. Generated by eidola-engine-model's\n");
    out.push_str("# tests/deepseek_numeric_vectors.rs, which documents the format.\n");
    for (name, bits) in [
        ("inv448", ((1.0f64 / 448.0) as f32).to_bits()),
        ("inv6", ((1.0f64 / 6.0) as f32).to_bits()),
        ("fp8_floor", (1e-4f64 as f32).to_bits()),
        ("fp4_pow2_floor", (6.0 * f32::from_bits(1 << 23)).to_bits()),
        ("fp4_e4m3_floor", (6.0 * 2f32.powi(-9)).to_bits()),
    ] {
        writeln!(out, "c {name} {bits:x}").unwrap();
    }

    for code in 0..256 {
        case(&mut out, "e4m3_decode", &[code]);
        case(&mut out, "e8m0_decode", &[code]);
    }
    for code in 0..16 {
        case(&mut out, "e2m1_decode", &[code]);
    }

    // Every one-argument function at the special values.
    for (name, _) in UNARY {
        for s in SPECIAL32 {
            case(&mut out, name, &[s as u64]);
        }
    }

    // BF16: ties to even, carries, and the overflow to infinity.
    for bits in [
        0x3f80_8000u32,
        0x3f81_8000,
        0x3f80_8001,
        0x3f80_7fff,
        0x3fff_ffff,
        0x7f7f_8000,
        0x7f7f_7fff,
        0x0000_7fff,
        0x0000_8000,
        0x0000_8001,
        0x0001_8000,
        0x007f_8000,
    ] {
        case(&mut out, "bf16_round", &[bits as u64]);
        case(&mut out, "bf16_round", &[(bits | 0x8000_0000) as u64]);
    }

    // E4M3 and E2M1: every code point, every midpoint between neighbours
    // (the ties) and the values either side, and the saturation edge.
    let e4m3: Vec<f32> = (0..127).map(|c| fp8_e4m3_to_f32(c as u8)).collect();
    let e2m1: Vec<f32> = E2M1_VALUES[..8].to_vec();
    for (name, table, beyond) in [
        ("round_e4m3", &e4m3, [448.5f32, 464.0, 1e9]),
        ("round_e2m1", &e2m1, [6.5, 7.0, 1e9]),
    ] {
        for pair in table.windows(2) {
            for x in around(pair[0]) {
                case(&mut out, name, &[x]);
            }
            for x in around((pair[0] + pair[1]) / 2.0) {
                case(&mut out, name, &[x]);
            }
        }
        for x in around(*table.last().unwrap()) {
            case(&mut out, name, &[x]);
        }
        for x in beyond {
            case(&mut out, name, &[b(x)]);
            case(&mut out, name, &[b(-x)]);
        }
    }

    // Scales: either side of every power of two, the floors, the extremes.
    let (inv448, inv6) = ((1.0f64 / 448.0) as f32, (1.0f64 / 6.0) as f32);
    for k in -40..=40 {
        for x in around(448.0 * 2f32.powi(k)) {
            if x >> 31 == 0 {
                case(&mut out, "round_scale", &[x, b(inv448)]);
            }
        }
    }
    for k in -126..=-100 {
        for x in around(6.0 * 2f32.powi(k)) {
            if x >> 31 == 0 {
                case(&mut out, "round_scale", &[x, b(inv6)]);
            }
        }
    }
    for k in -20..=20 {
        for x in around(6.0 * 2f32.powi(k)) {
            if x >> 31 == 0 {
                case(&mut out, "round_scale", &[x, b(inv6)]);
            }
        }
    }
    for amax in [
        0.0,
        1.5,
        3.0,
        1e-4f64 as f32,
        f32::MIN_POSITIVE,
        1e-45,
        f32::MAX,
        f32::INFINITY,
    ] {
        case(&mut out, "round_scale", &[b(amax), b(inv448)]);
        case(&mut out, "round_scale", &[b(amax), b(inv6)]);
    }

    // Quantiser blocks: the floors, saturation, zeros of both signs, NaN and
    // infinite elements, and ties.
    let mut blocks: Vec<Vec<f32>> = Vec::new();
    let with = |head: &[f32]| {
        let mut x = vec![0.0f32; 32];
        x[..head.len()].copy_from_slice(head);
        x
    };
    blocks.push(with(&[]));
    blocks.push(with(&[-0.0, 0.0, -0.0]));
    blocks.push(with(&[448.0, 3.3, -1.1875, 1.0625]));
    blocks.push(with(&[449.0, 3.3, -447.0]));
    blocks.push(with(&[7.0, -1.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]));
    blocks.push(with(&[3.0, 0.3, -0.2]));
    blocks.push(with(&[1e-6, -3e-7]));
    blocks.push(with(&[f32::NAN, 2.0, -5.0]));
    blocks.push(with(&[f32::INFINITY, 2.0, -5.0]));
    blocks.push(with(&[f32::MAX, -f32::MAX, 1.0]));
    blocks.push(with(&[1e-45, -1e-45, f32::MIN_POSITIVE]));
    for floor in [
        1e-4f64 as f32,
        6.0 * f32::from_bits(1 << 23),
        6.0 * 2f32.powi(-9),
    ] {
        for x in around(floor) {
            blocks.push(with(&[f(x), -floor / 3.0, floor / 7.0]));
        }
    }
    for k in [-30, -9, -1, 0, 1, 7] {
        let s = 2f32.powi(k);
        blocks.push((0..32).map(|j| (j as f32 - 15.5) * 0.25 * s).collect());
        blocks.push((0..32).map(|j| (j as f32 * 14.0 - 200.0) * s).collect());
    }
    for x in &blocks {
        let bits = bits32(x);
        for name in ["act_quant", "fq8", "fq4m"] {
            case(&mut out, name, &bits);
        }
        case(&mut out, "fq4n", &bits[..16]);
        case(&mut out, "fq4n", &bits[16..]);
    }

    // Block dots: extremes, cancellation, and sums that need the one rounding.
    let codes =
        |fill: &dyn Fn(usize) -> u8| -> Vec<u64> { (0..32).map(|j| fill(j) as u64).collect() };
    let a_sets: [Vec<u64>; 4] = [
        codes(&|_| 0x7e),
        codes(&|j| if j % 2 == 0 { 0x7e } else { 0xfe }),
        codes(&|j| if j == 0 { 0x7e } else { j as u8 }),
        codes(&|j| if j == 0 { 0xfe } else { 0x79 - 3 * j as u8 }),
    ];
    let w8_sets: [Vec<u64>; 4] = [
        codes(&|_| 0x7e),
        codes(&|j| 0x01 + j as u8),
        codes(&|j| if j % 3 == 0 { 0xfe } else { 0x7d - j as u8 }),
        codes(&|j| 0x80 | (4 * j as u8 + 1)),
    ];
    for a in &a_sets {
        for w in &w8_sets {
            let mut i = a.clone();
            i.extend(w);
            case(&mut out, "fp8_dot", &i);
        }
        for packed in [0x77u64, 0xff, 0x7f, 0x19, 0x00] {
            let mut i = a.clone();
            i.extend((0..16).map(|j| if j % 2 == 0 { packed } else { 0x5d ^ packed }));
            case(&mut out, "fp4_dot", &i);
        }
    }

    // Transcendentals: the pinned arguments, the cut-offs, the rounding
    // boundaries of the argument reduction.
    let ln2 = std::f64::consts::LN_2;
    let mut exps = vec![
        0.0, -1.0, -0.5, -13.37, -700.25, 3.0, 1e-300, -1e-300, 1.0, 88.5, -745.0,
    ];
    exps.extend([f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    for edge in [
        709.782712893384,
        -708.0,
        0.5 * ln2,
        1.5 * ln2,
        1023.5 * ln2,
        -20.0 * ln2,
    ] {
        for d in [-1i64, 0, 1] {
            let x = f64::from_bits(edge.abs().to_bits().wrapping_add_signed(d));
            exps.push(x.copysign(edge));
        }
    }
    for x in &exps {
        case(&mut out, "det_exp", &[x.to_bits()]);
    }
    let sqrt2 = std::f64::consts::SQRT_2;
    for x in [
        1.0,
        2.0,
        0.5,
        10000.0,
        160000.0,
        1e-6,
        f64::MIN_POSITIVE,
        f64::MAX,
        3.0,
        0.1,
    ] {
        case(&mut out, "det_ln", &[x.to_bits()]);
    }
    for d in [-1i64, 0, 1] {
        for edge in [sqrt2, sqrt2 / 2.0, 1.0] {
            case(&mut out, "det_ln", &[edge.to_bits().wrapping_add_signed(d)]);
        }
    }
    let pio2 = std::f64::consts::FRAC_PI_2;
    for k in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 1000.5, 65536.0, 1.0e6] {
        for d in [-1i64, 0, 1] {
            let x = f64::from_bits(
                (k * pio2)
                    .to_bits()
                    .wrapping_add_signed(d * (k > 0.0) as i64),
            );
            case(&mut out, "det_sin_cos", &[x.to_bits()]);
            case(&mut out, "det_sin_cos", &[(-x).to_bits()]);
        }
    }
    for x in [1e-300, 1e-9, 0.1, 1.0, 1.5e6, -1.5e6] {
        case(&mut out, "det_sin_cos", &[f64::to_bits(x)]);
    }
    for (name, _) in &UNARY[3..7] {
        for j in -80..=80 {
            case(&mut out, name, &[b(j as f32 * 1.25)]);
        }
        // softplus's two branch points, and exp32's f32 range.
        for edge in [20.0f32, -13.862944, 88.72284, -103.97208, -708.0, 709.78] {
            for x in around(edge) {
                case(&mut out, name, &[x]);
            }
        }
    }
    for x in [4.0f32, 2.0, 1e-20, 1e20, 0.1, 3.0] {
        case(&mut out, "rsqrt32", &[b(x)]);
    }

    // RoPE: the frequency powers and the released tables at spread positions.
    for theta in [10000.0f64, 160000.0] {
        for i in 0..32 {
            case(
                &mut out,
                "pow32",
                &[theta.to_bits(), b((2 * i) as f32 / 64.0)],
            );
        }
    }
    let yarn = Yarn {
        original_seq_len: 65536,
        factor: 16.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    for table in [
        RopeTable::new(64, 10000.0, None),
        RopeTable::new(64, 160000.0, Some(yarn)),
    ] {
        for position in [0u64, 1, 2, 127, 128, 4095, 65535, 65536, 1_048_575] {
            for freq in table.freqs.iter().step_by(3) {
                case(&mut out, "cis", &[position, b(*freq)]);
            }
        }
    }
    for q in [
        [1.0f32, 2.0, 0.5, 0.8660254],
        [-3.5, 0.25, -1.0, 0.0],
        [448.0, -448.0, 0.70710677, 0.70710677],
    ] {
        case(&mut out, "rotate", &bits32(&q));
        case(&mut out, "unrotate", &bits32(&q));
    }

    // Pinned orders at every length around the 16-wide chunks.
    for n in (0..=35).chain([47, 48, 49, 64, 80, 81]) {
        let a: Vec<f32> = (0..n)
            .map(|j| ((j * 37 % 23) as f32 - 11.0) * 1.37e-3)
            .collect();
        let c: Vec<f32> = (0..n)
            .map(|j| ((j * 13 % 29) as f32 - 14.0) * 733.1)
            .collect();
        let mut i = vec![n as u64];
        i.extend(bits32(&a));
        case(&mut out, "pinned_sum", &i);
        i.extend(bits32(&c));
        case(&mut out, "dot", &i);
    }

    for (g, u, limit, weighted, w) in [
        (1.0f32, 2.0f32, 10.0f32, 0u64, 0.0f32),
        (12.0, -15.0, 10.0, 1, 0.37),
        (-12.0, 15.0, 10.0, 1, 1.5),
        (12.0, -15.0, 0.0, 0, 0.0),
        (-0.0, 3.0, 10.0, 1, -0.0),
        (f32::NAN, 1.0, 10.0, 0, 0.0),
        (1.0, f32::NAN, 10.0, 1, 1.0),
    ] {
        case(&mut out, "swiglu", &[b(g), b(u), b(limit), weighted, b(w)]);
    }

    let scores = [
        1.0f32,
        3.0,
        3.0,
        -0.0,
        0.0,
        2.0,
        3.0,
        f32::NAN,
        -f32::NAN,
        f32::NEG_INFINITY,
    ];
    for (n, k) in [(7u64, 4u64), (5, 9), (10, 10), (10, 1), (1, 1)] {
        let mut i = vec![n, k];
        i.extend(bits32(&scores[..n as usize]));
        case(&mut out, "top_k", &i);
    }

    // Sweeps.
    for (name, _) in UNARY {
        sweep(&mut out, name, 0x11, 22);
    }
    for (name, log2) in [
        ("round_scale", 22),
        ("det_exp", 21),
        ("det_ln", 21),
        ("det_sin_cos", 21),
        ("pow32", 20),
        ("act_quant", 18),
        ("fq8", 18),
        ("fq4m", 18),
        ("fq4n", 18),
        ("fp8_dot", 18),
        ("fp4_dot", 18),
        ("lin8", 13),
        ("lin4", 13),
        ("dot", 17),
        ("pinned_sum", 17),
        ("rotate", 20),
        ("unrotate", 20),
        ("cis", 20),
        ("rms", 17),
        ("swiglu", 20),
        ("top_k", 16),
    ] {
        sweep(&mut out, name, 0x5eed, log2);
    }
    out
}

fn data(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../eidola-engine-kernels/tests/data")
        .join(file)
}

fn check_or_update(file: &str, generated: &str) {
    let path = data(file);
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

#[test]
fn vectors_are_current() {
    check_or_update("ds_numeric.vectors", &generate());
}

/// Every f32 through each one-argument function. Minutes of CPU time.
#[test]
#[ignore = "2^32 inputs per function"]
fn exhaustive_digests_are_current() {
    let mut out = String::from("# Every f32 through each one-argument function.\n");
    for (name, func) in UNARY {
        let digest = (0u32..1 << 16)
            .into_par_iter()
            .map(|hi| {
                let mut sum = 0u64;
                for lo in 0u32..1 << 16 {
                    let bits = hi << 16 | lo;
                    let word = (bits as u64) << 32 | w32(func(f32::from_bits(bits)));
                    sum = sum.wrapping_add(fin(word));
                }
                sum
            })
            .reduce(|| 0u64, u64::wrapping_add);
        writeln!(out, "x {name} = {digest:x}").unwrap();
    }
    check_or_update("ds_numeric.exhaustive", &out);
}
