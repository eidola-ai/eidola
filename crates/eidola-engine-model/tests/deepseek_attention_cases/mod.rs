//! Cases for the DeepSeek-V4.1 sparse attention, index scores, selections and
//! window slot lists, each with the result the reference functions give: the
//! inputs are the packed forms a KV pool holds, so an implementation over
//! packed entries (the GPU kernels) can be run on them unchanged.
//!
//! `tests/deepseek_attention_vectors.rs` writes a set of them to a file for
//! the kernels' host check; `eidola-engine-cuda`'s tests include this module
//! by path and run sweeps of them through the kernels.
// Included by path from another crate's tests, whose lints differ.
#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use eidola_engine_model::deepseek::attention::{candidate_mask, index_score, sparse_attend, top_k};
use eidola_engine_model::deepseek::drafter::ring_order;
use eidola_engine_model::deepseek::numeric::bf16_round;
use eidola_engine_model::numeric::{E2M1_VALUES, fp8_e4m3_to_f32};

pub const EMPTY: u32 = 0xffff_ffff;
pub const BAD_INDEX: u32 = 0x200;

/// Packed entry formats: E4M3 codes with an exponent byte per 32; E2M1
/// nibbles with an E4M3 scale per 16; E2M1 nibbles with an exponent byte per
/// 32.
pub const FP8: u32 = 0;
pub const FP4_E4M3: u32 = 1;
pub const FP4_E8M0: u32 = 2;

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
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    /// A BF16 value of either sign with magnitude in `[2^lo, 2^hi)`.
    pub fn bf16(&mut self, lo: i32, hi: i32) -> u16 {
        let r = self.next();
        let exp = (127 + lo + ((r >> 32) % (hi - lo) as u64) as i32) as u16;
        ((r >> 63) as u16) << 15 | exp << 7 | (r & 0x7f) as u16
    }
    /// An E4M3 code that is not NaN.
    pub fn code(&mut self) -> u8 {
        let b = (self.next() & 0xff) as u8;
        if b & 0x7f == 0x7f { b - 1 } else { b }
    }
}

pub fn bf16_value(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// The stored bits of a BF16 result, a NaN as `7fc0`: which NaN an operation
/// returns differs between processors.
pub fn bf16_bits(x: f32) -> u16 {
    if x.is_nan() {
        0x7fc0
    } else {
        assert_eq!(x.to_bits() & 0xffff, 0, "not a BF16 value");
        (x.to_bits() >> 16) as u16
    }
}

fn exponent_scale(byte: u8) -> f32 {
    f32::from_bits((byte as u32) << 23)
}

pub fn code_bytes(format: u32, width: usize) -> usize {
    if format == FP8 { width } else { width / 2 }
}

pub fn scale_bytes(format: u32, width: usize) -> usize {
    if format == FP4_E4M3 {
        width / 16
    } else {
        width / 32
    }
}

/// A packed entry's values, as a fake quantisation leaves them:
/// `bf16(code · scale)`.
pub fn unpack(format: u32, width: usize, entry: &[u8]) -> Vec<f32> {
    let (codes, scales) = entry.split_at(code_bytes(format, width));
    (0..width)
        .map(|d| {
            let nibble = || (codes[d / 2] >> ((d & 1) * 4)) & 0xf;
            bf16_round(match format {
                FP8 => fp8_e4m3_to_f32(codes[d]) * exponent_scale(scales[d / 32]),
                FP4_E4M3 => E2M1_VALUES[nibble() as usize] * fp8_e4m3_to_f32(scales[d / 16]),
                _ => E2M1_VALUES[nibble() as usize] * exponent_scale(scales[d / 32]),
            })
        })
        .collect()
}

/// A random packed entry (its codes, then its scales) whose values have
/// magnitudes around `2^exp`.
pub fn entry(rng: &mut Rng, format: u32, width: usize, exp: i32) -> Vec<u8> {
    let mut e: Vec<u8> = (0..code_bytes(format, width))
        .map(|_| match format {
            FP8 => rng.code(),
            _ => rng.next() as u8,
        })
        .collect();
    for _ in 0..scale_bytes(format, width) {
        e.push(match format {
            FP8 => (127 + exp - 2 + rng.below(3) as i32) as u8,
            FP4_E4M3 => ((6 + exp + rng.below(2) as i32) as u8) << 3 | rng.below(8) as u8,
            _ => (127 + exp - 1 + rng.below(2) as i32) as u8,
        });
    }
    e
}

fn nibble_set(codes: &mut [u8], d: usize, v: u8) {
    let shift = (d & 1) * 4;
    codes[d / 2] = codes[d / 2] & !(0xf << shift) | (v & 0xf) << shift;
}

fn nibble_get(codes: &[u8], d: usize) -> u8 {
    (codes[d / 2] >> ((d & 1) * 4)) & 0xf
}

/// Where the large terms of a [`cancelling`] entry sit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cancel {
    /// At the entry's two ends: each lane of the pinned order adds a large
    /// term, small ones, then the large term's negation.
    Ends,
    /// In elements `d` and `d + 8`: the lanes' sums are large and cancel in
    /// the halving tree.
    Lanes,
}

/// How many leading elements a format's large terms cover: a whole scale
/// block where a scale is what makes them large.
fn cancel_span(format: u32) -> usize {
    if format == FP4_E8M0 { 32 } else { 16 }
}

/// For a vector the cancelling entries are multiplied with: copies the
/// elements that meet the large terms so that each large product meets its
/// negation. `get` and `set` address element `d`.
fn mirror<T: Copy>(
    mode: Cancel,
    format: u32,
    width: usize,
    get: impl Fn(usize) -> T,
    mut set: impl FnMut(usize, T),
) {
    let span = cancel_span(format);
    match mode {
        Cancel::Ends => (0..span).for_each(|d| set(width - span + d, get(d))),
        Cancel::Lanes => (0..span)
            .filter(|d| d % 16 < 8)
            .for_each(|d| set(d + 8, get(d))),
    }
}

/// Rewrites a packed entry so that its dot product with a vector mirrored
/// the same way ([`mirror`]) cancels its largest terms exactly: some elements
/// become large, others their negations, and the rest stay small beside
/// them. The result is then what the small terms lost to rounding on the way,
/// so another summation order gives another value. Needs a width of 64 or
/// more.
pub fn cancelling(rng: &mut Rng, mode: Cancel, format: u32, width: usize, entry: &mut [u8]) {
    assert!(width >= 64);
    let span = cancel_span(format);
    let (codes, scales) = entry.split_at_mut(code_bytes(format, width));
    let large = |d: usize| match mode {
        Cancel::Ends => d < span || d >= width - span,
        Cancel::Lanes => d < span,
    };
    match format {
        FP8 => {
            scales.fill(127);
            for (d, c) in codes.iter_mut().enumerate() {
                // Exponent fields 13 and 14 (64 to 240) against 3 to 7 (under 2).
                let r = rng.next() as u8;
                *c = match large(d) {
                    true => r & 0x87 | (13 + (r >> 3 & 1)) << 3,
                    false => r & 0x87 | (3 + (r >> 3) % 5) << 3,
                };
            }
            let copy = codes.to_vec();
            mirror(mode, format, width, |d| copy[d], |d, c| codes[d] = c ^ 0x80);
        }
        _ => {
            let (per, big, small): (usize, u8, u8) = match format {
                FP4_E4M3 => (16, 0x70, 0x20),
                _ => (32, 139, 122),
            };
            for (b, s) in scales.iter_mut().enumerate() {
                *s = if large(b * per) { big } else { small };
            }
            for d in (0..width).filter(|d| large(*d)) {
                let v = nibble_get(codes, d) | 4;
                nibble_set(codes, d, v);
            }
            let copy = codes.to_vec();
            mirror(
                mode,
                format,
                width,
                |d| nibble_get(&copy, d),
                |d, v| nibble_set(codes, d, v ^ 8),
            );
        }
    }
}

// ---- sparse attention ----

/// One query of `sparse_attend` over packed entries.
#[derive(Clone)]
pub struct AttnCase {
    pub heads: usize,
    pub dim: usize,
    pub scale: f32,
    /// The global slots follow the last listed window slot (the drafter's
    /// list); otherwise a slot's index is its place.
    pub packed: bool,
    pub formats: [u32; 2],
    pub q: Vec<u16>,
    pub sink: Vec<f32>,
    /// The `a` words, then the `b` words: an index into that plane's
    /// `entries`, or [`EMPTY`].
    pub slots: Vec<u32>,
    pub a_slots: usize,
    pub entries: [Vec<Vec<u8>>; 2],
}

impl AttnCase {
    /// `sparse_attend`'s output, as stored BF16 bits.
    pub fn expected(&self) -> Vec<u16> {
        let values: Vec<Vec<Vec<f32>>> = (0..2)
            .map(|p| {
                self.entries[p]
                    .iter()
                    .map(|e| unpack(self.formats[p], self.dim, e))
                    .collect()
            })
            .collect();
        let (a, b) = self.slots.split_at(self.a_slots);
        let a_len = match self.packed {
            true => a.iter().rposition(|w| *w != EMPTY).map_or(0, |i| i + 1),
            false => a.len(),
        };
        let list: Vec<Option<&[f32]>> = a[..a_len]
            .iter()
            .map(|w| (*w != EMPTY).then(|| values[0][*w as usize].as_slice()))
            .chain(
                b.iter()
                    .map(|w| (*w != EMPTY).then(|| values[1][*w as usize].as_slice())),
            )
            .collect();
        let q: Vec<f32> = self.q.iter().map(|b| bf16_value(*b)).collect();
        let mut o = vec![0.0f32; q.len()];
        sparse_attend(&q, &list, &self.sink, self.scale, &mut o);
        o.iter().map(|v| bf16_bits(*v)).collect()
    }

    /// Makes every head's scores the residue of cancelled large terms (see
    /// [`cancelling`]). Head widths of 64 and more; an FP8 or E4M3-scaled
    /// plane.
    pub fn cancel(&mut self, rng: &mut Rng, mode: Cancel) {
        let dim = self.dim;
        for p in 0..2 {
            for e in &mut self.entries[p] {
                cancelling(rng, mode, self.formats[p], dim, e);
            }
        }
        for head in self.q.chunks_exact_mut(dim) {
            for (d, x) in head.iter_mut().enumerate() {
                let large = d < 16 || (mode == Cancel::Ends && d >= dim - 16);
                *x = if large {
                    rng.bf16(6, 9)
                } else {
                    rng.bf16(-2, 1)
                };
            }
            let copy = head.to_vec();
            mirror(mode, FP8, dim, |d| copy[d], |d, x| head[d] = x);
        }
    }
}

/// A query over `a_slots` window slots and `b_slots` global slots, one in
/// `empty_in` of them empty (none when 0), with a fresh entry in every listed
/// slot and the pool in another order than the list.
#[allow(clippy::too_many_arguments)]
pub fn attn_case(
    rng: &mut Rng,
    heads: usize,
    dim: usize,
    a_slots: usize,
    b_slots: usize,
    formats: [u32; 2],
    exp: i32,
    empty_in: usize,
) -> AttnCase {
    let mut entries = [Vec::new(), Vec::new()];
    let mut slots = Vec::new();
    for s in 0..a_slots + b_slots {
        let plane = (s >= a_slots) as usize;
        if empty_in > 0 && rng.below(empty_in) == 0 {
            slots.push(EMPTY);
            continue;
        }
        slots.push(entries[plane].len() as u32);
        entries[plane].push(entry(rng, formats[plane], dim, exp));
    }
    for (p, plane) in entries.iter_mut().enumerate() {
        let n = plane.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            order.swap(i, rng.below(i + 1));
        }
        // Entry `e` moves to pool index `order[e]`.
        let mut moved = plane.clone();
        for (e, to) in order.iter().enumerate() {
            moved[*to as usize] = std::mem::take(&mut plane[e]);
        }
        *plane = moved;
        let range = if p == 0 {
            0..a_slots
        } else {
            a_slots..a_slots + b_slots
        };
        for w in slots[range].iter_mut().filter(|w| **w != EMPTY) {
            *w = order[*w as usize];
        }
    }
    AttnCase {
        heads,
        dim,
        scale: bf16_round(1.0 / (dim as f32).sqrt()),
        packed: false,
        formats,
        q: (0..heads * dim).map(|_| rng.bf16(-3, 1)).collect(),
        sink: (0..heads).map(|_| bf16_value(rng.bf16(-2, 3))).collect(),
        slots,
        a_slots,
        entries,
    }
}

// ---- window slots ----

/// The window part of one row's slot list.
#[derive(Clone)]
pub struct WindowCase {
    /// 0: position order, right-aligned. 1: ring order.
    pub order: u32,
    pub window: usize,
    pub per_block: usize,
    pub pos: usize,
    /// The lowest position listed (position order only).
    pub first: usize,
    /// Block ids by block of positions; a negative word maps nothing.
    pub table: Vec<i32>,
}

impl WindowCase {
    pub fn new(order: u32, window: usize, per_block: usize, pos: usize, first: usize) -> Self {
        WindowCase {
            order,
            window,
            per_block,
            pos,
            first,
            table: (0..pos / per_block + 1)
                .map(|b| (b * 7 + 3) as i32 % 61)
                .collect(),
        }
    }

    /// The positions the reference lists, slot by slot.
    pub fn positions(&self) -> Vec<Option<usize>> {
        match self.order {
            // `forward.rs`: slot `i` is position `p + i + 1 - window`.
            0 => (0..self.window)
                .map(|i| {
                    (self.pos + i + 1)
                        .checked_sub(self.window)
                        .filter(|x| *x >= self.first)
                })
                .collect(),
            _ => {
                let mut ring: Vec<_> = ring_order(self.pos, self.window)
                    .into_iter()
                    .map(Some)
                    .collect();
                ring.resize(self.window, None);
                ring
            }
        }
    }

    /// The slot words (pool entries through the table) and the status.
    pub fn expected(&self) -> (Vec<u32>, u32) {
        let mut status = 0;
        let slots = self
            .positions()
            .iter()
            .map(|x| match x {
                None => EMPTY,
                Some(x) if self.table[x / self.per_block] < 0 => {
                    status = BAD_INDEX;
                    EMPTY
                }
                Some(x) => {
                    (self.table[x / self.per_block] as usize * self.per_block + x % self.per_block)
                        as u32
                }
            })
            .collect();
        (slots, status)
    }
}

// ---- index scores ----

/// `index_score` of every visible entry for one query.
#[derive(Clone)]
pub struct ScoreCase {
    pub heads: usize,
    pub dim: usize,
    pub weight_scale: f32,
    /// The index queries, packed FP4 with exponent scales: codes
    /// `[heads · dim / 2]`, then scales `[heads · dim / 32]`.
    pub q: Vec<u8>,
    /// BF16 `[heads]`.
    pub head_dots: Vec<u16>,
    /// The visible entries' keys, packed likewise.
    pub keys: Vec<Vec<u8>>,
}

impl ScoreCase {
    pub fn new(rng: &mut Rng, heads: usize, dim: usize, visible: usize) -> Self {
        ScoreCase {
            heads,
            dim,
            weight_scale: bf16_round(1.0 / ((heads * dim) as f32).sqrt()),
            q: entry(rng, FP4_E8M0, heads * dim, 0),
            head_dots: (0..heads).map(|_| rng.bf16(-2, 4)).collect(),
            keys: (0..visible).map(|_| entry(rng, FP4_E8M0, dim, 0)).collect(),
        }
    }

    /// Makes every head's dot the residue of cancelled large terms (see
    /// [`cancelling`]; widths of 96 and more), and, with 9 heads or more, the
    /// head sum too: heads 0 and 8 share a query and carry opposite weights
    /// far above the others'.
    pub fn cancel(&mut self, rng: &mut Rng, mode: Cancel) {
        let (heads, dim) = (self.heads, self.dim);
        assert!(dim >= 96);
        for k in &mut self.keys {
            cancelling(rng, mode, FP4_E8M0, dim, k);
        }
        let (codes, scales) = self.q.split_at_mut(heads * dim / 2);
        for h in 0..heads {
            let copy = codes.to_vec();
            mirror(
                mode,
                FP4_E8M0,
                dim,
                |d| nibble_get(&copy, h * dim + d),
                |d, v| nibble_set(codes, h * dim + d, v),
            );
            if mode == Cancel::Ends {
                scales[(h + 1) * dim / 32 - 1] = scales[h * dim / 32];
            }
        }
        if heads > 8 {
            let (lo, hi) = codes.split_at_mut(8 * dim / 2);
            hi[..dim / 2].copy_from_slice(&lo[..dim / 2]);
            let (lo, hi) = scales.split_at_mut(8 * dim / 32);
            hi[..dim / 32].copy_from_slice(&lo[..dim / 32]);
            self.head_dots[0] = rng.bf16(24, 25);
            self.head_dots[8] = self.head_dots[0] ^ 0x8000;
        }
    }

    /// Every visible entry's score, as stored BF16 bits.
    pub fn expected(&self) -> Vec<u16> {
        let iq = unpack(FP4_E8M0, self.heads * self.dim, &self.q);
        let weights: Vec<f32> = self
            .head_dots
            .iter()
            .map(|d| bf16_round(bf16_value(*d) * self.weight_scale))
            .collect();
        self.keys
            .iter()
            .map(|k| bf16_bits(index_score(&iq, &unpack(FP4_E8M0, self.dim, k), &weights)))
            .collect()
    }
}

// ---- selection ----

/// One row's top-k selection and candidate blocks.
#[derive(Clone)]
pub struct SelectCase {
    /// BF16 scores of the visible entries.
    pub scores: Vec<u16>,
    pub k: usize,
    pub top_blocks: usize,
    pub block_entries: usize,
    /// Entries per pool block.
    pub per_block: usize,
    /// The candidate blocks the row selects inside, one flag per block.
    pub mask_in: Option<Vec<bool>>,
    /// Whether the row publishes candidate blocks.
    pub mask_out: bool,
    pub table: Vec<i32>,
}

impl SelectCase {
    pub fn new(
        scores: Vec<u16>,
        k: usize,
        top_blocks: usize,
        block_entries: usize,
        per_block: usize,
    ) -> Self {
        let table = (0..scores.len().div_ceil(per_block) + 1)
            .map(|b| (b * 5 + 2) as i32 % 97)
            .collect();
        SelectCase {
            scores,
            k,
            top_blocks,
            block_entries,
            per_block,
            mask_in: None,
            mask_out: true,
            table,
        }
    }

    pub fn blocks(&self) -> usize {
        self.scores.len().div_ceil(self.block_entries)
    }

    /// Selects inside random candidate blocks (the newest always among them).
    pub fn with_pool(mut self, rng: &mut Rng) -> Self {
        let blocks = self.blocks();
        self.mask_in = Some(
            (0..blocks)
                .map(|b| b + 1 == blocks || rng.below(3) != 0)
                .collect(),
        );
        self
    }

    /// The selected entries in ascending order.
    pub fn chosen(&self) -> Vec<usize> {
        let scores: Vec<f32> = self.scores.iter().map(|b| bf16_value(*b)).collect();
        let allowed = (0..scores.len()).filter(|t| {
            self.mask_in
                .as_ref()
                .is_none_or(|m| m[t / self.block_entries])
        });
        let mut chosen = top_k(&scores, allowed, self.k);
        chosen.sort_unstable();
        chosen
    }

    /// The `k` slot words (pool entries through the table, then [`EMPTY`]),
    /// the kept candidate blocks (one flag per block) and the status.
    pub fn expected(&self) -> (Vec<u32>, Vec<bool>, u32) {
        let mut status = 0;
        let mut slots: Vec<u32> = self
            .chosen()
            .iter()
            .map(|t| match self.table[t / self.per_block] {
                b if b < 0 => {
                    status = BAD_INDEX;
                    EMPTY
                }
                b => (b as usize * self.per_block + t % self.per_block) as u32,
            })
            .collect();
        slots.resize(self.k, EMPTY);
        let scores: Vec<f32> = self.scores.iter().map(|b| bf16_value(*b)).collect();
        let kept = candidate_mask(&scores, scores.len(), self.top_blocks, self.block_entries);
        let kept = (0..self.blocks())
            .map(|b| kept[b * self.block_entries])
            .collect();
        (slots, kept, status)
    }
}

/// Scores of one kind: 0 any BF16 pattern, 1 a few values (ties everywhere,
/// both zeros), 2 ordinary values, 3 ordinary values with NaNs, infinities
/// and zeros of both signs among them, 4 one value.
pub fn scores(rng: &mut Rng, n: usize, kind: usize) -> Vec<u16> {
    const FEW: [u16; 6] = [0x0000, 0x8000, 0x3f80, 0x3f80, 0xbf80, 0x4000];
    const ODD: [u16; 10] = [
        0x7fc0, 0xffc0, 0x7f81, 0xffff, 0x7f80, 0xff80, 0x0000, 0x8000, 0x0001, 0x8001,
    ];
    (0..n)
        .map(|_| match kind {
            0 => rng.next() as u16,
            1 => FEW[rng.below(FEW.len())],
            3 if rng.below(4) == 0 => ODD[rng.below(ODD.len())],
            4 => 0x3f80,
            _ => rng.bf16(-4, 4),
        })
        .collect()
}

/// Candidate-block flags as mask words: bit `b % 32` of word `b / 32`.
pub fn mask_words(flags: &[bool], words: usize) -> Vec<u32> {
    let mut w = vec![0u32; words];
    for (b, f) in flags.iter().enumerate() {
        w[b / 32] |= (*f as u32) << (b % 32);
    }
    w
}
