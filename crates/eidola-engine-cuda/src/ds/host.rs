//! A host [`Device`]: host memory for device memory, and every kernel family computed
//! with the model crate's reference functions. It is how the executor (its planner, its
//! layouts and block images, its step order, its boundary) is checked against the CPU
//! reference executor on a machine without a GPU, and it is the statement of what each
//! launch of [`DsKernels`] must compute. Never a serving backend.
//!
//! A device address here is a host pointer into an allocation this device made; every
//! kernel reads and writes through the addresses of its struct exactly as a device
//! kernel would.

use std::any::Any;
use std::sync::OnceLock;

use eidola_engine::kv_cipher::{KvKeystream, KvNonce};
use eidola_engine::sampling::{self, Logits, SamplingParams};
use eidola_engine::secret::KvKey;
use eidola_engine_model::Matrix;
use eidola_engine_model::deepseek::attention::{candidate_mask, index_score, sparse_attend, top_k};
use eidola_engine_model::deepseek::mhc::{self, MhcWeights, Mixes};
use eidola_engine_model::deepseek::moe::swiglu_hidden;
use eidola_engine_model::deepseek::numeric::{
    bf16_round, exp32, pinned_sum, rsqrt32, sigmoid32, softplus32,
};
use eidola_engine_model::deepseek::quant::{act_quant, quant_fp4_e4m3, quant_fp4_pow2};
use eidola_engine_model::deepseek::rope::{rotate, unrotate};
use eidola_engine_model::deepseek::weights::rms;
use eidola_engine_model::numeric::{E2M1_VALUES, e8m0_to_f32, fp8_e4m3_to_f32};
use eidola_engine_model::tensor::dot;

use super::abi::*;
use super::half::{Device, SampleArgs};
use super::kernels::{DsKernels, PlaceRowsArgs};
use super::plan;
use crate::sampler::STATUS_NON_FINITE;
use crate::{CudaError, Result};

/// The host device.
#[derive(Debug, Default)]
pub struct HostDevice;

/// # Safety
///
/// `addr` is `len` initialised `T`s inside a live allocation, aligned for `T`.
unsafe fn rd<'a, T>(addr: u64, len: usize) -> &'a [T] {
    // SAFETY: the caller's.
    unsafe { std::slice::from_raw_parts(addr as *const T, len) }
}

/// # Safety
///
/// As [`rd`], and nothing else reads or writes the range while the slice lives.
#[allow(clippy::mut_from_ref)]
unsafe fn wr<'a, T>(addr: u64, len: usize) -> &'a mut [T] {
    // SAFETY: the caller's.
    unsafe { std::slice::from_raw_parts_mut(addr as *mut T, len) }
}

fn us(x: u32) -> usize {
    x as usize
}

fn widen(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|b| crate::bf16::to_f32(*b)).collect()
}

/// The 16 bits of a value the reference has already rounded to BF16.
fn narrow_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    assert!(bits & 0xffff == 0 || v.is_nan(), "{v} is not a BF16 value");
    u16::try_from(bits >> 16).expect("the high half")
}

fn store_bf16(values: &[f32], out: &mut [u16]) {
    for (o, v) in out.iter_mut().zip(values) {
        *o = narrow_bf16(*v);
    }
}

/// The E4M3 byte of a code point (either zero keeps its sign).
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
    let at = table
        .binary_search_by_key(&v.to_bits(), |e| e.0)
        .unwrap_or_else(|_| panic!("{v} is not an E4M3 value"));
    table[at].1
}

fn e2m1_nibble(v: f32) -> u8 {
    let at = E2M1_VALUES
        .iter()
        .position(|c| c.to_bits() == v.to_bits())
        .unwrap_or_else(|| panic!("{v} is not an E2M1 value"));
    u8::try_from(at).expect("a nibble")
}

/// A scale's byte: its exponent field (E8M0 for a finite power of two, 255 for the
/// infinite scale of a block holding an infinite value).
fn scale_byte(s: f32) -> u8 {
    u8::try_from((s.to_bits() >> 23) & 0xff).expect("an exponent field")
}

fn scale_of(byte: u8) -> f32 {
    f32::from_bits(u32::from(byte) << 23)
}

fn nibble(bytes: &[u8], i: usize) -> f32 {
    E2M1_VALUES[usize::from(bytes[i / 2] >> (4 * (i % 2)) & 0xf)]
}

/// The values of a packed row: `bf16(code · scale)`.
fn dequant(format: u32, width: usize, codes: &[u8], scales: &[u8]) -> Vec<f32> {
    (0..width)
        .map(|j| {
            bf16_round(match format {
                KV_FP8 => fp8_e4m3_to_f32(codes[j]) * scale_of(scales[j / 32]),
                KV_FP4_E4M3 => nibble(codes, j) * fp8_e4m3_to_f32(scales[j / 16]),
                KV_FP4_E8M0 => nibble(codes, j) * scale_of(scales[j / 32]),
                _ => panic!("format {format}"),
            })
        })
        .collect()
}

/// The addresses of entry `e` of `plane`.
fn entry_at(plane: &KvPlane, e: u32) -> (u64, u64) {
    let (block, o) = (
        u64::from(e / plane.entries_per_block),
        u64::from(e % plane.entries_per_block),
    );
    let at = plane.base + block * plane.block_bytes;
    (
        at + plane.codes_offset + o * u64::from(kv_code_bytes(plane.format, plane.width)),
        at + plane.scales_offset + o * u64::from(kv_scale_bytes(plane.format, plane.width)),
    )
}

/// # Safety
///
/// `e` is an entry of a live pool that `plane` describes.
unsafe fn read_entry(plane: &KvPlane, e: u32) -> Vec<f32> {
    let (codes, scales) = entry_at(plane, e);
    let (cb, sb) = (
        us(kv_code_bytes(plane.format, plane.width)),
        us(kv_scale_bytes(plane.format, plane.width)),
    );
    // SAFETY: the caller's.
    unsafe { dequant(plane.format, us(plane.width), rd(codes, cb), rd(scales, sb)) }
}

/// The packed codes and scales of a row in `format`.
fn pack(format: u32, x: &[f32]) -> (Vec<u8>, Vec<u8>) {
    let n = x.len();
    let mut codes = vec![0.0; n];
    let nibbles = |codes: &[f32]| {
        codes
            .chunks_exact(2)
            .map(|p| e2m1_nibble(p[0]) | e2m1_nibble(p[1]) << 4)
            .collect()
    };
    match format {
        KV_FP8 => {
            let mut scales = vec![0.0; n / 32];
            act_quant(x, &mut codes, &mut scales);
            (
                codes.iter().map(|c| e4m3_byte(*c)).collect(),
                scales.iter().map(|s| scale_byte(*s)).collect(),
            )
        }
        KV_FP4_E4M3 => {
            let mut scales = vec![0.0; n / 16];
            quant_fp4_e4m3(x, &mut codes, &mut scales);
            (
                nibbles(&codes),
                scales.iter().map(|s| e4m3_byte(*s)).collect(),
            )
        }
        KV_FP4_E8M0 => {
            let mut scales = vec![0.0; n / 32];
            quant_fp4_pow2(x, &mut codes, &mut scales);
            (
                nibbles(&codes),
                scales.iter().map(|s| scale_byte(*s)).collect(),
            )
        }
        _ => panic!("format {format}"),
    }
}

/// # Safety
///
/// `p`'s arrays hold what its fields document.
unsafe fn rotate_rows(p: &RotQuantParams, quantise: bool) {
    let width = us(p.heads * p.head_dim);
    let pairs = us(p.rope_dim / 2);
    // SAFETY: the caller's, for every array below.
    unsafe {
        for r in 0..us(p.rows) {
            let mut x = widen(rd::<u16>(p.x + (r * width * 2) as u64, width));
            if p.rotation != ROT_NONE {
                let pos = us(rd::<u32>(p.positions, us(p.rows))[r]);
                assert!(pos < us(p.cis_positions), "position {pos} past the table");
                let table = rd::<f32>(p.cis + (pos * pairs * 8) as u64, pairs * 2);
                let cis: Vec<(f32, f32)> = table.chunks_exact(2).map(|c| (c[0], c[1])).collect();
                for head in x.chunks_exact_mut(us(p.head_dim)) {
                    match p.rotation {
                        ROT_ROTATE => rotate(head, &cis),
                        ROT_UNROTATE => unrotate(head, &cis),
                        other => panic!("rotation {other}"),
                    }
                }
            }
            if !quantise {
                store_bf16(&x, wr::<u16>(p.dst_codes + (r * width * 2) as u64, width));
                continue;
            }
            let (codes, scales) = pack(p.format, &x);
            let (to_codes, to_scales) = if p.targets != 0 {
                let e = rd::<u32>(p.targets, us(p.rows))[r];
                if e == SLOT_EMPTY {
                    continue;
                }
                assert_eq!(
                    (p.dst.format, p.dst.width),
                    (p.format, p.heads * p.head_dim)
                );
                entry_at(&p.dst, e)
            } else {
                (
                    p.dst_codes + (r * codes.len()) as u64,
                    p.dst_scales + (r * scales.len()) as u64,
                )
            };
            wr::<u8>(to_codes, codes.len()).copy_from_slice(&codes);
            wr::<u8>(to_scales, scales.len()).copy_from_slice(&scales);
        }
    }
}

/// # Safety
///
/// `addr` holds `rows` rows of `k` elements of `dtype`.
unsafe fn typed_row(addr: u64, dtype: u32, row: usize, k: usize) -> Vec<f32> {
    // SAFETY: the caller's.
    unsafe {
        match dtype {
            DT_BF16 => widen(rd::<u16>(addr + (row * k * 2) as u64, k)),
            DT_F32 => rd::<f32>(addr + (row * k * 4) as u64, k).to_vec(),
            other => panic!("dtype {other}"),
        }
    }
}

/// Exact dot product of two 32-wide blocks of FP8 values, rounded once.
fn fp8_block_dot(a: &[u8], w: &[u8]) -> f32 {
    let mut s = 0.0f64;
    for (a, w) in a.iter().zip(w) {
        s += f64::from(fp8_e4m3_to_f32(*a)) * f64::from(fp8_e4m3_to_f32(*w));
    }
    // Every partial sum is exact in f64, so the order does not matter; one rounding.
    #[allow(clippy::cast_possible_truncation)]
    {
        s as f32
    }
}

impl DsKernels for HostDevice {
    unsafe fn embed(&self, p: &EmbedParams) -> Result<()> {
        let (dim, copies) = (us(p.dim), us(p.copies));
        // SAFETY: the caller's, as for every method below.
        unsafe {
            let tokens = rd::<u32>(p.tokens, us(p.rows));
            let out = wr::<u16>(p.out, us(p.rows) * copies * dim);
            for (row, &t) in out.chunks_exact_mut(copies * dim).zip(tokens) {
                if t >= p.vocab {
                    row.fill(0);
                    wr::<u32>(p.status, 1)[0] |= STATUS_BAD_TOKEN;
                    continue;
                }
                let src = rd::<u16>(p.table + (us(t) * dim * 2) as u64, dim);
                for copy in row.chunks_exact_mut(dim) {
                    copy.copy_from_slice(src);
                }
            }
        }
        Ok(())
    }

    unsafe fn quant_fp8(&self, p: &QuantFp8Params) -> Result<()> {
        let w = us(p.width);
        unsafe {
            for r in 0..us(p.rows) {
                let x = match p.x_format {
                    X_BF16 => widen(rd::<u16>(p.x + (r * w * 2) as u64, w)),
                    X_FP8 => dequant(
                        KV_FP8,
                        w,
                        rd(p.x + (r * w) as u64, w),
                        rd(p.x_scales + (r * w / 32) as u64, w / 32),
                    ),
                    other => panic!("x_format {other}"),
                };
                let (codes, scales) = pack(KV_FP8, &x);
                wr::<u8>(p.codes + (r * w) as u64, w).copy_from_slice(&codes);
                wr::<u8>(p.scales + (r * w / 32) as u64, w / 32).copy_from_slice(&scales);
            }
        }
        Ok(())
    }

    unsafe fn rot_quant(&self, p: &RotQuantParams) -> Result<()> {
        unsafe { rotate_rows(p, true) };
        Ok(())
    }

    unsafe fn rotate(&self, p: &RotQuantParams) -> Result<()> {
        unsafe { rotate_rows(p, false) };
        Ok(())
    }

    unsafe fn rms_norm(&self, p: &RmsNormParams) -> Result<()> {
        let w = us(p.width);
        unsafe {
            let weight = rd::<f32>(p.weight, w);
            for r in 0..us(p.rows) {
                let x = widen(rd::<u16>(p.x + (r * w * 2) as u64, w));
                store_bf16(&rms(&x, weight, p.eps), wr(p.out + (r * w * 2) as u64, w));
            }
        }
        Ok(())
    }

    unsafe fn mhc_mix(&self, p: &MhcMixParams) -> Result<()> {
        let (hc, width) = (us(p.copies), us(p.copies * p.dim));
        let n = us(mix_words(p.copies));
        unsafe {
            let w = MhcWeights {
                projection: Matrix::from_vec(n, width, rd::<f32>(p.projection, n * width).to_vec()),
                base: rd::<f32>(p.base, n).to_vec(),
                scale: p.scale,
            };
            for r in 0..us(p.rows) {
                let x = widen(rd::<u16>(p.x + (r * width * 2) as u64, width));
                let m = mhc::mixes(&w, &x, hc, p.norm_eps, p.eps, us(p.iters));
                let out = wr::<f32>(p.out + (r * n * 4) as u64, n);
                out[..hc].copy_from_slice(&m.pre);
                out[hc..2 * hc].copy_from_slice(&m.post);
                out[2 * hc..].copy_from_slice(&m.comb);
            }
        }
        Ok(())
    }

    unsafe fn collapse(&self, p: &CollapseParams) -> Result<()> {
        let (dim, hc) = (us(p.dim), us(p.copies));
        unsafe {
            for r in 0..us(p.rows) {
                let x = widen(rd::<u16>(p.x + (r * hc * dim * 2) as u64, hc * dim));
                let pre = rd::<f32>(p.pre + (r * us(p.pre_stride) * 4) as u64, hc);
                let mut out = vec![0.0; dim];
                mhc::collapse(&x, pre, &mut out);
                store_bf16(&out, wr(p.out + (r * dim * 2) as u64, dim));
            }
        }
        Ok(())
    }

    unsafe fn expand(&self, p: &ExpandParams) -> Result<()> {
        let (dim, hc) = (us(p.dim), us(p.copies));
        unsafe {
            for r in 0..us(p.rows) {
                let mut x = widen(rd::<u16>(p.x + (r * hc * dim * 2) as u64, hc * dim));
                let y = widen(rd::<u16>(p.y + (r * dim * 2) as u64, dim));
                let m = rd::<f32>(p.mixes + (r * us(p.mix_stride) * 4) as u64, (2 + hc) * hc);
                let mixes = Mixes {
                    pre: m[..hc].to_vec(),
                    post: m[hc..2 * hc].to_vec(),
                    comb: m[2 * hc..].to_vec(),
                };
                mhc::expand(&mut x, &y, &mixes);
                store_bf16(&x, wr(p.out + (r * hc * dim * 2) as u64, hc * dim));
            }
        }
        Ok(())
    }

    unsafe fn lin8(&self, p: &Lin8Params) -> Result<()> {
        let (n, k) = (us(p.n), us(p.k));
        let kb = k / 32;
        unsafe {
            let w = rd::<u8>(p.w_codes, n * k);
            let ws = rd::<u8>(p.w_scales, n.div_ceil(32) * kb);
            for r in 0..us(p.rows) {
                let codes = rd::<u8>(p.codes + (r * k) as u64, k);
                let scales = rd::<u8>(p.act_scales + (r * kb) as u64, kb);
                let out = wr::<u16>(p.out + (r * n * 2) as u64, n);
                for (o, out) in out.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for b in 0..kb {
                        let d = fp8_block_dot(
                            &codes[b * 32..(b + 1) * 32],
                            &w[o * k + b * 32..o * k + (b + 1) * 32],
                        );
                        acc += d * scale_of(scales[b]) * e8m0_to_f32(ws[(o / 32) * kb + b]);
                    }
                    *out = narrow_bf16(bf16_round(acc));
                }
            }
        }
        Ok(())
    }

    unsafe fn lin4(&self, p: &Lin4Params) -> Result<()> {
        let (n, k) = (us(p.n), us(p.k));
        let kb = k / 32;
        unsafe {
            let ends = rd::<u32>(p.group_end, us(p.groups));
            let mut prev = 0u32;
            for (g, &end) in ends.iter().enumerate() {
                let start = prev.next_multiple_of(p.row_align);
                prev = end;
                assert!(end <= p.layout_rows, "a run past the layout");
                let w = rd::<u8>(p.w_packed + (g * n * k / 2) as u64, n * k / 2);
                let ws = rd::<u8>(p.w_scales + (g * n * kb) as u64, n * kb);
                for r in us(start)..us(end.max(start)) {
                    let codes = rd::<u8>(p.codes + (r * k) as u64, k);
                    let scales = rd::<u8>(p.act_scales + (r * kb) as u64, kb);
                    // In units of 2^-9: an integer.
                    #[allow(clippy::cast_possible_truncation)]
                    let ints: Vec<i32> = codes
                        .iter()
                        .map(|c| (fp8_e4m3_to_f32(*c) * 512.0) as i32)
                        .collect();
                    let out = wr::<u16>(p.out + (r * n * 2) as u64, n);
                    for (o, out) in out.iter_mut().enumerate() {
                        let row = &w[o * k / 2..(o + 1) * k / 2];
                        let mut acc = 0.0f32;
                        for b in 0..kb {
                            let mut sum = 0i32;
                            for (j, a) in ints.iter().enumerate().skip(b * 32).take(32) {
                                #[allow(clippy::cast_possible_truncation)]
                                let doubled = (2.0 * nibble(row, j)) as i32;
                                sum += a * doubled;
                            }
                            #[allow(clippy::cast_precision_loss)]
                            let d = sum as f32 * (1.0 / 1024.0);
                            acc += d * scale_of(scales[b]) * e8m0_to_f32(ws[o * kb + b]);
                        }
                        *out = narrow_bf16(bf16_round(acc));
                    }
                }
            }
        }
        Ok(())
    }

    unsafe fn grouped_dot(&self, p: &GroupedDotParams) -> Result<()> {
        let (groups, rank, k) = (us(p.groups), us(p.rank), us(p.k));
        let (kb, n) = (k / 32, groups * rank);
        unsafe {
            let w = rd::<u8>(p.w_codes, n * k);
            let ws = rd::<u8>(p.w_scales, n.div_ceil(32) * kb);
            let weight: Vec<Vec<f32>> = (0..n)
                .map(|o| {
                    (0..k)
                        .map(|c| {
                            bf16_round(
                                fp8_e4m3_to_f32(w[o * k + c])
                                    * e8m0_to_f32(ws[(o / 32) * kb + c / 32]),
                            )
                        })
                        .collect()
                })
                .collect();
            for r in 0..us(p.rows) {
                let width = groups * k;
                let x = dequant(
                    KV_FP8,
                    width,
                    rd(p.codes + (r * width) as u64, width),
                    rd(p.act_scales + (r * width / 32) as u64, width / 32),
                );
                let out = wr::<u16>(p.out + (r * n * 2) as u64, n);
                for (o, out) in out.iter_mut().enumerate() {
                    let g = o / rank;
                    *out = narrow_bf16(bf16_round(dot(&x[g * k..(g + 1) * k], &weight[o])));
                }
            }
        }
        Ok(())
    }

    unsafe fn attention(&self, p: &AttnParams) -> Result<()> {
        let width = us(p.heads * p.head_dim);
        unsafe {
            let sink = rd::<f32>(p.sink, us(p.heads));
            for r in 0..us(p.rows) {
                let q = widen(rd::<u16>(p.q + (r * width * 2) as u64, width));
                let list = rd::<u32>(
                    p.slots + (r * us(p.slot_stride) * 4) as u64,
                    us(p.a_slots + p.b_slots),
                );
                let entries: Vec<Option<Vec<f32>>> = list
                    .iter()
                    .enumerate()
                    .map(|(i, &e)| {
                        let plane = if i < us(p.a_slots) { &p.a } else { &p.b };
                        (e != SLOT_EMPTY).then(|| read_entry(plane, e))
                    })
                    .collect();
                let slots: Vec<Option<&[f32]>> = entries.iter().map(|e| e.as_deref()).collect();
                let mut out = vec![0.0; width];
                sparse_attend(&q, &slots, sink, p.scale, &mut out);
                store_bf16(&out, wr(p.out + (r * width * 2) as u64, width));
            }
        }
        Ok(())
    }

    unsafe fn linear(&self, p: &LinearParams) -> Result<()> {
        let (n, k) = (us(p.n), us(p.k));
        unsafe {
            let weight: Vec<Vec<f32>> = (0..n).map(|o| typed_row(p.w, p.w_dtype, o, k)).collect();
            for r in 0..us(p.rows) {
                let from = match p.x_rows {
                    0 => r,
                    rows => us(rd::<u32>(rows, us(p.rows))[r]),
                };
                let x = typed_row(p.x, p.x_dtype, from, k);
                let mut out: Vec<f32> = weight.iter().map(|w| dot(&x, w)).collect();
                if p.addend != 0 {
                    let add = rd::<f32>(p.addend + (r * n * 4) as u64, n);
                    for (o, a) in out.iter_mut().zip(add) {
                        *o += *a;
                    }
                }
                match p.out_dtype {
                    DT_F32 => wr::<f32>(p.out + (r * n * 4) as u64, n).copy_from_slice(&out),
                    DT_BF16 => {
                        let rounded: Vec<f32> = out.iter().map(|v| bf16_round(*v)).collect();
                        store_bf16(&rounded, wr(p.out + (r * n * 2) as u64, n));
                    }
                    other => panic!("out_dtype {other}"),
                }
            }
        }
        Ok(())
    }

    unsafe fn pool(&self, p: &PoolParams) -> Result<()> {
        let w = us(p.width);
        unsafe {
            let (first, second) = (
                rd::<u32>(p.first, us(p.entries)),
                rd::<u32>(p.second, us(p.entries)),
            );
            for e in 0..us(p.entries) {
                let row = |base: u64, r: u32| rd::<f32>(base + (us(r) * w * 4) as u64, w);
                let kv = [row(p.kv, first[e]), row(p.kv, second[e])];
                let score = [row(p.score, first[e]), row(p.score, second[e])];
                let out = wr::<u16>(p.out + (e * w * 2) as u64, w);
                for (ch, out) in out.iter_mut().enumerate() {
                    let max = score[0][ch].max(score[1][ch]);
                    let weights = [exp32(score[0][ch] - max), exp32(score[1][ch] - max)];
                    let sum = weights[0] + weights[1];
                    let mut pooled = 0.0f32;
                    for m in 0..2 {
                        pooled += kv[m][ch] * (weights[m] / sum);
                    }
                    *out = narrow_bf16(bf16_round(pooled));
                }
            }
        }
        Ok(())
    }

    unsafe fn index_score(&self, p: &IndexScoreParams) -> Result<()> {
        let width = us(p.heads * p.dim);
        let epb = p.keys.entries_per_block;
        unsafe {
            let (table_row, visible) = (
                rd::<u32>(p.table_row, us(p.rows)),
                rd::<u32>(p.visible, us(p.rows)),
            );
            for r in 0..us(p.rows) {
                let q = dequant(
                    KV_FP4_E8M0,
                    width,
                    rd(p.q_codes + (r * width / 2) as u64, width / 2),
                    rd(p.q_scales + (r * width / 32) as u64, width / 32),
                );
                let dots = widen(rd::<u16>(
                    p.head_dots + (r * us(p.heads) * 2) as u64,
                    us(p.heads),
                ));
                let weights: Vec<f32> = dots
                    .iter()
                    .map(|d| bf16_round(d * p.weight_scale))
                    .collect();
                let table = rd::<i32>(
                    p.tables + (us(table_row[r]) * us(p.table_stride) * 4) as u64,
                    us(p.table_stride),
                );
                let scores = wr::<u16>(
                    p.scores + (r * us(p.score_stride) * 2) as u64,
                    us(visible[r]),
                );
                for (t, score) in scores.iter_mut().enumerate() {
                    let t = u32::try_from(t).expect("an entry");
                    let Ok(block) = u32::try_from(table[us(t / epb)]) else {
                        wr::<u32>(p.status, 1)[0] |= STATUS_BAD_INDEX;
                        *score = 0;
                        continue;
                    };
                    let key = read_entry(&p.keys, block * epb + t % epb);
                    *score = narrow_bf16(index_score(&q, &key, &weights));
                }
            }
        }
        Ok(())
    }

    unsafe fn select(&self, p: &SelectParams) -> Result<()> {
        let (words, be) = (us(p.mask_words), us(p.block_entries));
        unsafe {
            let (table_row, visible) = (
                rd::<u32>(p.table_row, us(p.rows)),
                rd::<u32>(p.visible, us(p.rows)),
            );
            for r in 0..us(p.rows) {
                let n = us(visible[r]);
                let scores = widen(rd::<u16>(p.scores + (r * us(p.score_stride) * 2) as u64, n));
                if p.mask_out != 0 {
                    let mask = wr::<u32>(p.mask_out + (r * words * 4) as u64, words);
                    mask.fill(0);
                    let keep = candidate_mask(&scores, n, us(p.top_blocks), be);
                    for (t, kept) in keep.iter().enumerate() {
                        if *kept {
                            mask[t / be / 32] |= 1 << (t / be % 32);
                        }
                    }
                }
                let allowed: Vec<usize> = match p.mask_in {
                    0 => (0..n).collect(),
                    at => {
                        let mask = rd::<u32>(at + (r * words * 4) as u64, words);
                        (0..n)
                            .filter(|t| mask[t / be / 32] >> (t / be % 32) & 1 == 1)
                            .collect()
                    }
                };
                let mut chosen = top_k(&scores, allowed.into_iter(), us(p.top_k));
                chosen.sort_unstable();
                let table = rd::<i32>(
                    p.tables + (us(table_row[r]) * us(p.table_stride) * 4) as u64,
                    us(p.table_stride),
                );
                let slots = wr::<u32>(
                    p.slots + ((r * us(p.slot_stride) + us(p.slot_first)) * 4) as u64,
                    us(p.top_k),
                );
                slots.fill(SLOT_EMPTY);
                for (slot, t) in slots.iter_mut().zip(chosen) {
                    let t = u32::try_from(t).expect("an entry");
                    match u32::try_from(table[us(t / p.entries_per_block)]) {
                        Ok(block) => *slot = block * p.entries_per_block + t % p.entries_per_block,
                        Err(_) => wr::<u32>(p.status, 1)[0] |= STATUS_BAD_INDEX,
                    }
                }
            }
        }
        Ok(())
    }

    unsafe fn route(&self, p: &RouteParams) -> Result<()> {
        let (experts, active) = (us(p.experts), us(p.active));
        unsafe {
            let bias = rd::<f32>(p.bias, experts);
            for r in 0..us(p.rows) {
                let logits = rd::<f32>(p.logits + (r * experts * 4) as u64, experts);
                let scores: Vec<f32> = logits.iter().map(|l| softplus32(*l).sqrt()).collect();
                let biased: Vec<f32> = scores.iter().zip(bias).map(|(s, b)| s + b).collect();
                let chosen = top_k(&biased, 0..experts, active);
                let mut weights: Vec<f32> = chosen.iter().map(|e| scores[*e]).collect();
                if p.normalize != 0 {
                    let sum = weights.iter().fold(0.0f32, |a, w| a + w);
                    #[allow(clippy::cast_possible_truncation)]
                    let floor = 1e-20f64 as f32;
                    for w in &mut weights {
                        *w /= sum + floor;
                    }
                }
                let mut pairs: Vec<(usize, f32)> = chosen
                    .into_iter()
                    .zip(weights.iter().map(|w| w * p.route_scale))
                    .collect();
                pairs.sort_by_key(|pair| pair.0);
                let ids = wr::<u32>(p.ids + (r * active * 4) as u64, active);
                let out = wr::<f32>(p.weights + (r * active * 4) as u64, active);
                for (i, (e, w)) in pairs.into_iter().enumerate() {
                    ids[i] = u32::try_from(e).expect("an expert");
                    out[i] = w;
                }
            }
        }
        Ok(())
    }

    unsafe fn place(&self, p: &PlaceParams) -> Result<()> {
        let pairs = us(p.rows * p.active);
        unsafe {
            let placed = plan::place(
                rd::<u32>(p.ids, pairs),
                rd::<f32>(p.weights, pairs),
                p.groups,
                p.row_align,
                p.layout_rows,
            )
            .ok_or_else(|| CudaError::new("place: the layout does not hold the step's pairs"))?;
            wr::<u32>(p.row_of, pairs).copy_from_slice(&placed.0);
            wr::<f32>(p.row_weight, us(p.layout_rows)).copy_from_slice(&placed.1);
            wr::<u32>(p.group_end, us(p.groups)).copy_from_slice(&placed.2);
        }
        Ok(())
    }

    unsafe fn place_rows(&self, p: &PlaceRowsArgs) -> Result<()> {
        let (active, bytes) = (us(p.active), us(p.row_bytes));
        unsafe {
            let row_of = rd::<u32>(p.row_of, us(p.rows) * active);
            for (i, &to) in row_of.iter().enumerate() {
                if to >= p.layout_rows {
                    wr::<u32>(p.status, 1)[0] |= STATUS_BAD_INDEX;
                    continue;
                }
                let src = rd::<u8>(p.src + (i / active * bytes) as u64, bytes);
                wr::<u8>(p.dst + (us(to) * bytes) as u64, bytes).copy_from_slice(src);
            }
        }
        Ok(())
    }

    unsafe fn swiglu(&self, p: &SwigluParams) -> Result<()> {
        let (w, stride) = (us(p.width), us(p.in_stride));
        unsafe {
            for r in 0..us(p.rows) {
                let gate = widen(rd::<u16>(p.gate + (r * stride * 2) as u64, w));
                let up = widen(rd::<u16>(p.up + (r * stride * 2) as u64, w));
                let weight = (p.weight != 0).then(|| rd::<f32>(p.weight, us(p.rows))[r]);
                let hidden = swiglu_hidden(&gate, &up, p.limit, weight);
                let (codes, scales) = pack(KV_FP8, &hidden);
                wr::<u8>(p.codes + (r * w) as u64, w).copy_from_slice(&codes);
                wr::<u8>(p.scales + (r * w / 32) as u64, w / 32).copy_from_slice(&scales);
            }
        }
        Ok(())
    }

    unsafe fn combine(&self, p: &CombineParams) -> Result<()> {
        let (dim, active) = (us(p.dim), us(p.active));
        unsafe {
            for r in 0..us(p.rows) {
                let row_of = rd::<u32>(p.row_of + (r * active * 4) as u64, active);
                let mut sum = vec![0.0f32; dim];
                for &at in row_of {
                    let routed = rd::<u16>(p.routed + (us(at) * dim * 2) as u64, dim);
                    for (a, b) in sum.iter_mut().zip(widen(routed)) {
                        *a += b;
                    }
                }
                let shared = widen(rd::<u16>(p.shared + (r * dim * 2) as u64, dim));
                let out = wr::<u16>(p.out + (r * dim * 2) as u64, dim);
                for ((o, a), b) in out.iter_mut().zip(&sum).zip(&shared) {
                    *o = narrow_bf16(bf16_round(*a + *b));
                }
            }
        }
        Ok(())
    }

    unsafe fn engram_gather(&self, p: &EngramGatherParams) -> Result<()> {
        let (w, per) = (us(p.width), us(p.ids_per_row));
        unsafe {
            let ids = rd::<u32>(p.ids, us(p.rows) * per);
            for (i, &id) in ids.iter().enumerate() {
                let codes = wr::<u8>(p.out_codes + (i * w) as u64, w);
                let scales = wr::<u8>(p.out_scales + (i * w / 32) as u64, w / 32);
                if u64::from(id) >= p.table_rows {
                    codes.fill(0);
                    scales.fill(0);
                    wr::<u32>(p.status, 1)[0] |= STATUS_BAD_INDEX;
                    continue;
                }
                codes.copy_from_slice(rd(p.codes + (us(id) * w) as u64, w));
                scales.copy_from_slice(rd(p.scales + (us(id) * w / 32) as u64, w / 32));
            }
        }
        Ok(())
    }

    unsafe fn engram_inject(&self, p: &EngramInjectParams) -> Result<()> {
        let (dim, hc) = (us(p.dim), us(p.copies));
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let (scale, floor, n) = (
            (1.0 / (dim as f64).sqrt()) as f32,
            1e-6f64 as f32,
            dim as f32,
        );
        unsafe {
            let gate_weight = rd::<f32>(p.gate_weight, hc * dim);
            for r in 0..us(p.rows) {
                let mut x = widen(rd::<u16>(p.x + (r * hc * dim * 2) as u64, hc * dim));
                let kv = widen(rd::<u16>(
                    p.kv + (r * (hc + 1) * dim * 2) as u64,
                    (hc + 1) * dim,
                ));
                let (keys, value) = kv.split_at(hc * dim);
                let mut terms = vec![0.0f32; dim];
                for j in 0..hc {
                    let xj = &mut x[j * dim..(j + 1) * dim];
                    let key = &keys[j * dim..(j + 1) * dim];
                    let rstd = rsqrt32(dot(xj, xj) / n + p.norm_eps)
                        * rsqrt32(dot(key, key) / n + p.norm_eps);
                    for d in 0..dim {
                        terms[d] = xj[d] * gate_weight[j * dim + d] * key[d];
                    }
                    let dot = pinned_sum(&terms) * rstd * scale;
                    let gate = sigmoid32(dot.abs().max(floor).sqrt().copysign(dot));
                    for d in 0..dim {
                        xj[d] = bf16_round(xj[d] + gate * value[d]);
                    }
                }
                store_bf16(&x, wr(p.out + (r * hc * dim * 2) as u64, hc * dim));
            }
        }
        Ok(())
    }
}

impl Device for HostDevice {
    fn alloc(&self, bytes: usize) -> Result<(u64, Box<dyn Any>)> {
        let buf = vec![0u64; bytes.div_ceil(8)].into_boxed_slice();
        Ok((buf.as_ptr() as u64, Box::new(buf)))
    }

    unsafe fn zero(&self, addr: u64, bytes: usize) -> Result<()> {
        // SAFETY: the caller's range, for this and every method below.
        unsafe { wr::<u8>(addr, bytes).fill(0) };
        Ok(())
    }

    unsafe fn upload(&self, addr: u64, src: &[u8]) -> Result<()> {
        unsafe { wr::<u8>(addr, src.len()).copy_from_slice(src) };
        Ok(())
    }

    unsafe fn download(&self, addr: u64, dst: &mut [u8]) -> Result<()> {
        dst.copy_from_slice(unsafe { rd::<u8>(addr, dst.len()) });
        Ok(())
    }

    unsafe fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        unsafe { wr::<u8>(dst, bytes).copy_from_slice(rd::<u8>(src, bytes)) };
        Ok(())
    }

    unsafe fn cipher(&self, addr: u64, words: u32, key: &KvKey, nonce: &KvNonce) -> Result<()> {
        KvKeystream::new(key, nonce).apply_words(unsafe { wr::<u32>(addr, us(words)) });
        Ok(())
    }

    unsafe fn sample(&self, a: &SampleArgs<'_>) -> Result<()> {
        let stride = us(a.logits_stride);
        unsafe {
            let tokens = wr::<u32>(a.tokens, a.rows.len());
            for (token, row) in tokens.iter_mut().zip(a.rows) {
                if row.logit_row >= a.logit_rows {
                    return Err(CudaError::new("sample: a logit row outside the step's"));
                }
                let logits = rd::<f32>(a.logits + (us(row.logit_row) * stride * 4) as u64, stride);
                if logits[..us(a.n)]
                    .iter()
                    .any(|l| l.is_nan() || *l == f32::INFINITY)
                {
                    wr::<u32>(a.status, 1)[0] |= STATUS_NON_FINITE;
                    continue;
                }
                let params =
                    SamplingParams::new(row.temperature, row.top_k, row.top_p, row.min_p, row.seed)
                        .map_err(|e| CudaError::new(e.to_string()))?;
                *token =
                    sampling::sample(Logits::new(logits, a.n), &params, u64::from(row.position));
            }
        }
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn kernels(&self) -> &dyn DsKernels {
        self
    }

    fn capacity(&self) -> Option<usize> {
        None
    }
}
