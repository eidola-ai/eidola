//! Operands of the DeepSeek-V4.1 quantised GEMMs for the tensor-core tests
//! and bench (`tests/ds_gemm.rs`, `examples/ds_gemm_bench.rs`): activation
//! codes and scale bytes as the quantiser leaves them, FP8 and FP4 weights as
//! stored, and expert-major layouts, either drawn like the released model's
//! (a block scaled to its largest element, values roughly normal with a few
//! outliers) or as arbitrary codes over every exponent.
#![allow(dead_code)]

/// SplitMix64.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u32) -> u32 {
        u32::try_from(self.next() % u64::from(n)).unwrap()
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        let top = u32::try_from(self.next() >> 40).unwrap();
        f32::from_bits(0x3f80_0000 | (top >> 1)) - 1.0
    }

    /// Roughly normal (a sum of four uniforms), one in 64 eight times wider.
    pub fn gauss(&mut self) -> f32 {
        let s = self.unit() + self.unit() + self.unit() + self.unit() - 2.0;
        if self.below(64) == 0 { 8.0 * s } else { s }
    }

    /// An E4M3 code that is not NaN, of any exponent.
    pub fn any_code(&mut self) -> u8 {
        let b = u8::try_from(self.next() & 0xff).unwrap();
        if b & 0x7f == 0x7f { b - 1 } else { b }
    }
}

/// The positive E4M3 values, by code (`0x00 ..= 0x7e`, ascending).
fn e4m3_values() -> Vec<f32> {
    (0u8..0x7f)
        .map(eidola_engine_model::numeric::fp8_e4m3_to_f32)
        .collect()
}

/// The nearest E4M3 code to `v` (ties either way: only codes are needed),
/// saturating at 448.
pub fn e4m3_near(table: &[f32], v: f32) -> u8 {
    let a = v.abs();
    let i = table.partition_point(|x| *x < a).min(table.len() - 1);
    let i = if i > 0 && (a - table[i - 1]) < (table[i] - a) {
        i - 1
    } else {
        i
    };
    u8::try_from(i).unwrap() | if v < 0.0 { 0x80 } else { 0 }
}

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn e2m1_near(v: f32) -> u8 {
    let a = v.abs();
    let mut best = 0;
    for (i, x) in E2M1.iter().enumerate() {
        if (a - x).abs() < (a - E2M1[best]).abs() {
            best = i;
        }
    }
    u8::try_from(best).unwrap() | if v < 0.0 { 8 } else { 0 }
}

/// `count` values in 32-wide blocks, each block scaled so its largest is
/// `top`, encoded by `encode`.
fn scaled_blocks(g: &mut Rng, count: usize, top: f32, encode: &dyn Fn(f32) -> u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(count);
    let mut block = [0.0f32; 32];
    for _ in 0..count / 32 {
        for v in &mut block {
            *v = g.gauss();
        }
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        out.extend(block.iter().map(|v| encode(v / amax * top)));
    }
    out
}

/// Activation codes `[rows, k]` and scale bytes `[rows, k / 32]`: like the
/// model's (`wide` false) or arbitrary codes over every exponent.
pub fn activations(g: &mut Rng, rows: usize, k: usize, wide: bool) -> (Vec<u8>, Vec<u8>) {
    let codes = if wide {
        (0..rows * k).map(|_| g.any_code()).collect()
    } else {
        let t = e4m3_values();
        scaled_blocks(g, rows * k, 448.0, &|v| e4m3_near(&t, v))
    };
    let scales = (0..rows * k / 32)
        .map(|_| u8::try_from(105 + g.below(40)).unwrap())
        .collect();
    (codes, scales)
}

/// An FP8 weight `[n, k]` with its tile scales `[ceil(n / 32), k / 32]`.
pub fn fp8_weight(g: &mut Rng, n: usize, k: usize, wide: bool) -> (Vec<u8>, Vec<u8>) {
    let codes = if wide {
        (0..n * k).map(|_| g.any_code()).collect()
    } else {
        let t = e4m3_values();
        scaled_blocks(g, n * k, 448.0, &|v| e4m3_near(&t, v))
    };
    let scales = (0..n.div_ceil(32) * (k / 32))
        .map(|_| u8::try_from(110 + g.below(16)).unwrap())
        .collect();
    (codes, scales)
}

/// FP4 weights of `groups` groups `[groups, n, k / 2]` (element `2i` in the
/// low nibble) with their scales `[groups, n, k / 32]`.
pub fn fp4_weights(
    g: &mut Rng,
    groups: usize,
    n: usize,
    k: usize,
    wide: bool,
) -> (Vec<u8>, Vec<u8>) {
    let nibbles = if wide {
        (0..groups * n * k)
            .map(|_| u8::try_from(g.next() & 15).unwrap())
            .collect()
    } else {
        scaled_blocks(g, groups * n * k, 6.0, &e2m1_near)
    };
    let packed = nibbles.chunks_exact(2).map(|p| p[0] | p[1] << 4).collect();
    let scales = (0..groups * n * k / 32)
        .map(|_| u8::try_from(115 + g.below(16)).unwrap())
        .collect();
    (packed, scales)
}

/// FP4 weights of `groups` groups as [`fp4_weights`], made of `distinct`
/// drawn groups repeated: the released experts' bytes without drawing every
/// one of them.
pub fn fp4_weights_tiled(
    g: &mut Rng,
    groups: usize,
    distinct: usize,
    n: usize,
    k: usize,
    wide: bool,
) -> (Vec<u8>, Vec<u8>) {
    let (w, ws) = fp4_weights(g, distinct, n, k, wide);
    let (wg, sg) = (n * k / 2, n * k / 32);
    let mut packed = Vec::with_capacity(groups * wg);
    let mut scales = Vec::with_capacity(groups * sg);
    for i in 0..groups {
        let j = i % distinct;
        packed.extend_from_slice(&w[j * wg..(j + 1) * wg]);
        scales.extend_from_slice(&ws[j * sg..(j + 1) * sg]);
    }
    (packed, scales)
}

/// An expert-major layout of `tokens` tokens each routed to `active`
/// distinct groups (popular groups more often): each group's end row, and
/// the layout's rows (its last run's end).
pub fn layout(
    g: &mut Rng,
    tokens: u32,
    groups: u32,
    active: u32,
    row_align: u32,
) -> (Vec<u32>, u32) {
    let mut counts = vec![0u32; usize::try_from(groups).unwrap()];
    for _ in 0..tokens {
        let mut picked: Vec<u32> = Vec::new();
        while u32::try_from(picked.len()).unwrap() < active {
            // Skewed: the smaller of two draws favours low ids.
            let id = g.below(groups).min(g.below(groups));
            if !picked.contains(&id) {
                picked.push(id);
            }
        }
        for id in picked {
            counts[usize::try_from(id).unwrap()] += 1;
        }
    }
    let mut end = Vec::with_capacity(counts.len());
    let mut at = 0u32;
    for c in counts {
        at = at.next_multiple_of(row_align) + c;
        end.push(at);
    }
    (end, at)
}
