//! Engram: hashed n-gram lookups added into the residual stream
//! (`inference/engram.py`; `inference/model.py` `Engram`,
//! `ParallelEngramEmbedding`).
//!
//! A position is hashed as its 2-gram … `max_ngram`-gram, each over several
//! heads; every (n-gram size, head) pair owns a prime-sized bucket range of
//! the layer's table. N-grams are hashed over a compressed vocabulary, in
//! which tokens that normalise alike share an id. That map is built from the
//! tokenizer, which this crate does not know: the caller provides it.

use std::collections::HashMap;
use std::sync::Arc;

use super::linear::{Fp8Linear, lin8};
use super::numeric::{bf16_round, pinned_sum, rsqrt32, sigmoid32};
use crate::error::{Error, Result};
use crate::numeric::{e8m0_to_f32, fp8_e4m3_to_f32};
use crate::safetensors::{Dtype, WeightSet};
use crate::tensor::{Matrix, dot};

/// The hash constants of one Engram layer.
#[derive(Debug, Clone, PartialEq)]
pub struct EngramLayerSpec {
    /// The checkpoint layer the module sits before.
    pub source_layer: usize,
    /// One odd multiplier per lookback (0 = the position's own token).
    pub multipliers: Vec<i64>,
    /// Bucket moduli, `[(max_ngram - 1) · heads]`, n-gram size major.
    pub primes: Vec<u64>,
    /// First table row of each bucket range (the running sum of the primes).
    pub offsets: Vec<u64>,
    /// Table rows: the sum of the primes.
    pub num_rows: u64,
}

fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

fn pow_mod(mut b: u64, mut e: u64, m: u64) -> u64 {
    let mut r = 1;
    b %= m;
    while e > 0 {
        if e & 1 == 1 {
            r = mul_mod(r, b, m);
        }
        b = mul_mod(b, b, m);
        e >>= 1;
    }
    r
}

/// Miller–Rabin with the bases that decide every u64.
fn is_prime(n: u64) -> bool {
    const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    if n < 2 {
        return false;
    }
    for p in BASES {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let (mut d, mut s) = (n - 1, 0);
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }
    'base: for a in BASES {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'base;
            }
        }
        return false;
    }
    true
}

/// NumPy's `SeedSequence(seed).generate_state(4, uint64)` for a seed below
/// 2^32, as the 128-bit state and increment words PCG64 is seeded with.
fn seed_sequence(seed: u32) -> [u64; 4] {
    const MULT_A: u32 = 0x931e_8875;
    const MULT_B: u32 = 0x58f3_8ded;
    const MIX_L: u32 = 0xca01_f9dd;
    const MIX_R: u32 = 0x4973_f715;
    let mut hash_const: u32 = 0x43b0_d7e5;
    let mut hashmix = |mut v: u32| {
        v ^= hash_const;
        hash_const = hash_const.wrapping_mul(MULT_A);
        v = v.wrapping_mul(hash_const);
        v ^ (v >> 16)
    };
    let mix = |x: u32, y: u32| {
        let r = MIX_L.wrapping_mul(x).wrapping_sub(MIX_R.wrapping_mul(y));
        r ^ (r >> 16)
    };
    let mut pool = [0u32; 4];
    for (i, p) in pool.iter_mut().enumerate() {
        *p = hashmix(if i == 0 { seed } else { 0 });
    }
    for src in 0..4 {
        for dst in 0..4 {
            if src != dst {
                let h = hashmix(pool[src]);
                pool[dst] = mix(pool[dst], h);
            }
        }
    }
    let mut hash_const: u32 = 0x8b51_f9dd;
    let mut words = [0u32; 8];
    for (i, w) in words.iter_mut().enumerate() {
        let mut v = pool[i % 4] ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        v = v.wrapping_mul(hash_const);
        *w = v ^ (v >> 16);
    }
    std::array::from_fn(|i| words[2 * i] as u64 | (words[2 * i + 1] as u64) << 32)
}

/// NumPy's PCG64 (`PCG64(seed)`): a 128-bit LCG with the XSL-RR output.
struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    const MULT: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

    fn new(seed: u32) -> Self {
        let w = seed_sequence(seed);
        let init = (w[0] as u128) << 64 | w[1] as u128;
        let seq = (w[2] as u128) << 64 | w[3] as u128;
        let mut g = Pcg64 {
            state: 0,
            inc: (seq << 1) | 1,
        };
        g.step();
        g.state = g.state.wrapping_add(init);
        g.step();
        g
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(Self::MULT).wrapping_add(self.inc);
    }

    fn next_u64(&mut self) -> u64 {
        self.step();
        let (hi, lo) = ((self.state >> 64) as u64, self.state as u64);
        (hi ^ lo).rotate_right((self.state >> 122) as u32)
    }

    /// `Generator.integers(0, bound, dtype=int64)` for a bound above 2^32:
    /// Lemire's multiply-and-reject on 64-bit draws.
    fn below(&mut self, bound: u64) -> u64 {
        let mut m = self.next_u64() as u128 * bound as u128;
        if (m as u64) < bound {
            let threshold = bound.wrapping_neg() % bound;
            while (m as u64) < threshold {
                m = self.next_u64() as u128 * bound as u128;
            }
        }
        (m >> 64) as u64
    }
}

/// The hash layout of every Engram layer, as the reference derives it
/// (`EngramLayout.from_args`, `compute_hash_multipliers`):
///
/// - primes: walking layers, then n-gram sizes, each head takes the next
///   prime above the previous head's, starting above `bucket_base - 1` for
///   every n-gram size and skipping primes already handed out;
/// - multipliers: for layer `l`, `2·v + 1` for `max_ngram` draws `v` below
///   `((2^63 - 1) / compressed_vocab) / 2` from NumPy's
///   `default_rng(10007·l).integers`. `token · multiplier` cannot overflow.
///
/// The multipliers are recomputed here, not stored: they are a function of
/// the config alone, and the reference's values for the released model are
/// pinned by `released_hash_multipliers`.
pub fn engram_layout(
    layer_ids: &[usize],
    max_ngram: usize,
    heads: usize,
    bucket_base: u64,
    compressed_vocab: usize,
) -> Vec<EngramLayerSpec> {
    let bound = ((i64::MAX as u64 / compressed_vocab as u64) / 2).max(1);
    let mut seen: Vec<u64> = Vec::new();
    layer_ids
        .iter()
        .map(|&layer| {
            let mut primes = Vec::new();
            for _ in 1..max_ngram {
                let mut current = bucket_base.saturating_sub(1);
                for _ in 0..heads {
                    current += 1;
                    while !is_prime(current) || seen.contains(&current) {
                        current += 1;
                    }
                    seen.push(current);
                    primes.push(current);
                }
            }
            let mut offsets = Vec::with_capacity(primes.len());
            let mut total = 0;
            for p in &primes {
                offsets.push(total);
                total += p;
            }
            let mut rng = Pcg64::new((10007 * layer) as u32);
            let multipliers = (0..max_ngram)
                .map(|_| {
                    let v = if bound > u32::MAX as u64 {
                        rng.below(bound)
                    } else {
                        unreachable_small_bound(bound)
                    };
                    (2 * v + 1) as i64
                })
                .collect();
            EngramLayerSpec {
                source_layer: layer,
                multipliers,
                primes,
                offsets,
                num_rows: total,
            }
        })
        .collect()
}

/// A compressed vocabulary above 2^30 entries would take NumPy's 32-bit draw
/// path, which no released config reaches.
fn unreachable_small_bound(bound: u64) -> u64 {
    panic!(
        "Engram multiplier bound {bound} needs NumPy's 32-bit bounded draw, which is not implemented"
    )
}

/// Token ids → table rows.
#[derive(Debug, Clone)]
pub struct EngramHasher {
    token_map: Vec<u32>,
    pad: i64,
    max_ngram: usize,
    heads: usize,
}

impl EngramHasher {
    /// `token_map[id]` is the compressed id of token `id`, as the reference's
    /// `build_compressed_token_map` assigns it from the tokenizer.
    pub fn new(
        spec: &super::config::EngramSpec,
        vocab_size: usize,
        token_map: Vec<u32>,
    ) -> Result<Self> {
        if token_map.len() != vocab_size {
            return Err(Error::Input(format!(
                "the compressed-vocabulary map has {} entries for a vocabulary of {vocab_size}",
                token_map.len()
            )));
        }
        let size = token_map.iter().max().map_or(0, |m| *m as usize + 1);
        if size != spec.compressed_vocab {
            return Err(Error::Input(format!(
                "the compressed-vocabulary map has {size} ids; the config declares {}",
                spec.compressed_vocab
            )));
        }
        let pad = *token_map
            .get(spec.pad_token as usize)
            .ok_or_else(|| Error::InvalidConfig(format!("Engram pad token {}", spec.pad_token)))?;
        Ok(EngramHasher {
            token_map,
            pad: pad as i64,
            max_ngram: spec.max_ngram,
            heads: spec.n_heads,
        })
    }

    /// The table rows of one layer for the position whose token is the last
    /// of `history` (the sequence from its start, or at least the last
    /// `max_ngram` tokens of it, with `position` the token's index).
    ///
    /// Lookback `k` reads the token `k` positions back, or the pad id before
    /// the sequence start. The running XOR of `token_k · multiplier_k` after
    /// lookback `g` is the (g+1)-gram's hash, reduced modulo each head's prime.
    pub fn rows(&self, layer: &EngramLayerSpec, history: &[u32], position: usize) -> Vec<u64> {
        let mut rows = Vec::with_capacity((self.max_ngram - 1) * self.heads);
        let mut rolling = 0i64;
        for k in 0..self.max_ngram {
            let token = if k <= position && k < history.len() {
                self.token_map[history[history.len() - 1 - k] as usize] as i64
            } else {
                self.pad
            };
            rolling ^= token.wrapping_mul(layer.multipliers[k]);
            if k > 0 {
                for h in 0..self.heads {
                    let slot = (k - 1) * self.heads + h;
                    rows.push(
                        layer.offsets[slot] + rolling.rem_euclid(layer.primes[slot] as i64) as u64,
                    );
                }
            }
        }
        rows
    }
}

/// Row access to one Engram table. The released tables are about 98 GB each,
/// so a row is read by index and never the table as a whole.
pub trait EngramTable: Send + Sync {
    fn num_rows(&self) -> u64;
    fn dim(&self) -> usize;
    /// Row `index` as real values: `e4m3(code) · 2^(scale - 127)` with one
    /// scale per 32 columns, each rounded to BF16.
    fn read_row(&self, index: u64, out: &mut [f32]) -> Result<()>;
}

fn decode_row(codes: &[u8], scales: &[u8], out: &mut [f32]) {
    for (j, (o, &c)) in out.iter_mut().zip(codes).enumerate() {
        *o = bf16_round(fp8_e4m3_to_f32(c) * e8m0_to_f32(scales[j / 32]));
    }
}

/// A table read from the memory-mapped checkpoint, one row at a time.
pub struct MappedEngramTable {
    store: Arc<WeightSet>,
    weight: String,
    scale: String,
    rows: u64,
    dim: usize,
}

impl MappedEngramTable {
    /// The table stored as `{prefix}.weight` (`[rows, dim]` FP8) and
    /// `{prefix}.scale` (`[rows, dim / 32]` E8M0).
    pub fn open(store: Arc<WeightSet>, prefix: &str, rows: u64, dim: usize) -> Result<Self> {
        let (weight, scale) = (format!("{prefix}.weight"), format!("{prefix}.scale"));
        store
            .get(&weight)?
            .expect(&weight, Dtype::F8E4M3, &[rows as usize, dim])?;
        store
            .get(&scale)?
            .expect(&scale, Dtype::F8E8M0, &[rows as usize, dim / 32])?;
        Ok(MappedEngramTable {
            store,
            weight,
            scale,
            rows,
            dim,
        })
    }
}

impl EngramTable for MappedEngramTable {
    fn num_rows(&self) -> u64 {
        self.rows
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn read_row(&self, index: u64, out: &mut [f32]) -> Result<()> {
        if index >= self.rows {
            return Err(Error::Input(format!("Engram row {index} of {}", self.rows)));
        }
        let (i, d) = (index as usize, self.dim);
        let codes = &self.store.get(&self.weight)?.data[i * d..(i + 1) * d];
        let scales = &self.store.get(&self.scale)?.data[i * (d / 32)..(i + 1) * (d / 32)];
        decode_row(codes, scales, out);
        Ok(())
    }
}

/// A table holding only some rows: a test's stand-in for a table too large to
/// have. Reading a row it does not hold is an error.
pub struct SparseEngramTable {
    rows: u64,
    dim: usize,
    held: HashMap<u64, Vec<f32>>,
}

impl SparseEngramTable {
    /// `ids[i]` is the table row whose stored bytes are row `i` of `codes`
    /// (`[n, dim]` FP8) and `scales` (`[n, dim / 32]` E8M0).
    pub fn from_stored(
        rows: u64,
        dim: usize,
        ids: &[u64],
        codes: &[u8],
        scales: &[u8],
    ) -> Result<Self> {
        if codes.len() != ids.len() * dim || scales.len() != ids.len() * (dim / 32) {
            return Err(Error::Input(
                "sparse Engram rows do not match their ids".into(),
            ));
        }
        let held = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let mut row = vec![0.0; dim];
                decode_row(
                    &codes[i * dim..(i + 1) * dim],
                    &scales[i * (dim / 32)..(i + 1) * (dim / 32)],
                    &mut row,
                );
                (id, row)
            })
            .collect();
        Ok(SparseEngramTable { rows, dim, held })
    }
}

impl EngramTable for SparseEngramTable {
    fn num_rows(&self) -> u64 {
        self.rows
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn read_row(&self, index: u64, out: &mut [f32]) -> Result<()> {
        let row = self.held.get(&index).ok_or_else(|| {
            Error::Input(format!("Engram row {index} is not among the rows held"))
        })?;
        out.copy_from_slice(row);
        Ok(())
    }
}

/// One Engram module's weights.
pub struct EngramWeights {
    pub table: Arc<dyn EngramTable>,
    /// `[dim · (hc + 1), lookups · head_dim]`: `hc` keys and one value.
    pub wkv: Fp8Linear,
    /// `q_weight · k_weight`, `[hc, dim]`: the two are only used as a product.
    pub gate_weight: Matrix,
}

/// Add one token's lookup into its stream `x` (`hc · dim`, copy-major), in
/// place.
///
/// The gathered rows go through `lin8` into one key per copy and a shared
/// value. Per copy `j`, all in f32:
/// `dot = Σ_d (x_j·w_j·key_j) · rsqrt(mean(x_j²) + eps) · rsqrt(mean(key_j²) + eps) · dim^-0.5`,
/// `gate = sigmoid(sign(dot) · sqrt(max(|dot|, 1e-6)))`, `x_j ← bf16(x_j + gate · value)`.
pub fn inject(
    w: &EngramWeights,
    rows: &[u64],
    x: &mut [f32],
    hc: usize,
    norm_eps: f32,
) -> Result<()> {
    let gathered = gather(w.table.as_ref(), rows)?;
    inject_gathered(w, &gathered, x, hc, norm_eps);
    Ok(())
}

/// The table rows `rows`, concatenated: all an Engram lookup reads of its
/// table, so the table may live apart from the layer that consumes it.
pub fn gather(table: &dyn EngramTable, rows: &[u64]) -> Result<Vec<f32>> {
    let head = table.dim();
    let mut gathered = vec![0.0f32; rows.len() * head];
    for (r, out) in rows.iter().zip(gathered.chunks_exact_mut(head)) {
        table.read_row(*r, out)?;
    }
    Ok(gathered)
}

/// [`inject`] given the rows already gathered ([`gather`]).
pub fn inject_gathered(
    w: &EngramWeights,
    gathered: &[f32],
    x: &mut [f32],
    hc: usize,
    norm_eps: f32,
) {
    let dim = x.len() / hc;
    let mut kv = vec![0.0f32; dim * (hc + 1)];
    lin8(gathered, &w.wkv, &mut kv);
    let (keys, value) = kv.split_at(hc * dim);
    let scale = (1.0 / (dim as f64).sqrt()) as f32;
    let mut terms = vec![0.0f32; dim];
    for j in 0..hc {
        let xj = &mut x[j * dim..(j + 1) * dim];
        let key = &keys[j * dim..(j + 1) * dim];
        let rstd = rsqrt32(dot(xj, xj) / dim as f32 + norm_eps)
            * rsqrt32(dot(key, key) / dim as f32 + norm_eps);
        for d in 0..dim {
            terms[d] = xj[d] * w.gate_weight.row(j)[d] * key[d];
        }
        let dot = pinned_sum(&terms) * rstd * scale;
        let gate = sigmoid32(dot.abs().max(1e-6f64 as f32).sqrt().copysign(dot));
        for d in 0..dim {
            xj[d] = bf16_round(xj[d] + gate * value[d]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NumPy 2.3.4: `np.random.PCG64(10007).random_raw(3)`.
    #[test]
    fn pcg64_matches_numpy() {
        assert_eq!(
            seed_sequence(10007),
            [
                0x77d32a348ad70d07,
                0x993ed609b84a4fa5,
                0xd5f972dec8698e49,
                0x7d79ebaebcf148de
            ]
        );
        let mut g = Pcg64::new(10007);
        assert_eq!(
            [g.next_u64(), g.next_u64(), g.next_u64()],
            [0xd2c3f84026661e33, 0x0d4fb6969eb57435, 0x62e6e1c58c3bd72b]
        );
    }

    /// DeepSeek-V4.1-Flash-1017: Engram layers 1 and 14, 2/3/4-grams, 8
    /// heads, buckets from 16,000,000, compressed vocabulary 99,092. The
    /// multipliers are the reference's (`compute_hash_multipliers` under
    /// NumPy 2.3.4); the row counts are the checkpoint's table sizes.
    #[test]
    fn released_hash_multipliers() {
        let layout = engram_layout(&[1, 14], 4, 8, 16_000_000, 99_092);
        assert_eq!(
            layout[0].multipliers,
            [
                76632096046245,
                4839876093313,
                35959672319349,
                73987337458391
            ]
        );
        assert_eq!(
            layout[1].multipliers,
            [
                67716810739261,
                51510806800915,
                30921347202721,
                82619226485591
            ]
        );
        assert_eq!(layout[0].num_rows, 384_006_168);
        assert_eq!(layout[1].num_rows, 384_016_682);
        assert_eq!(&layout[0].primes[..3], &[16000057, 16000079, 16000081]);
        assert_eq!(layout[1].primes[23], 16000889);
        assert_eq!(layout[0].offsets[1], 16000057);
    }

    #[test]
    fn primality() {
        let primes: Vec<u64> = (0..60).filter(|n| is_prime(*n)).collect();
        assert_eq!(
            primes,
            [
                2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59
            ]
        );
        assert!(is_prime(16000057) && !is_prime(16000059));
        assert!(is_prime(18446744073709551557) && !is_prime(18446744073709551556));
    }
}
