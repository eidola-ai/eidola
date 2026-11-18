//! The fast profile's blocked DeltaNet (`QwDnChunkParams`, `qwen_fast_deltanet.cuh`):
//! its definition written out in Rust ([`model`]: the host form's steps in its order,
//! each tensor-core K-step read as the host form reads it), the token recurrence it
//! regroups in f64 ([`recurrence_f64`]), and the cases every device runs through
//! [`QwenKernels::dn_chunk`](eidola_engine_cuda::qwen::kernels::QwenKernels::dn_chunk):
//! against the model (bit for bit on the host, within a bound on a GPU, whose tensor cores
//! sum a K-step their own way), against the recurrence (within a bound, measured), and
//! reproducibility (every output and state bit for bit the same however a run is cut into
//! launches, whether a position is decoded from its block's boundary or prefilled with the
//! positions after it present, and whatever else shares the launch).
#![allow(dead_code, clippy::needless_range_loop)]

use eidola_engine_cuda::qwen::QwenDevice;
use eidola_engine_cuda::qwen::abi::*;
use eidola_engine_model::qwen::deltanet::recurrent_step;
use eidola_engine_model::qwen::numeric::{bf16_round, exp32};
use eidola_engine_model::tensor::dot;

use super::{Buf, Lcg, buf, read, zeros};

pub const TILE: usize = 16;
const KD: usize = 128;
const COLS: usize = 32;

fn u(x: usize) -> u32 {
    u32::try_from(x).expect("a test width fits u32")
}

/// TF32, nearest with ties away from zero, as f32 (`cvt.rna.tf32.f32`).
pub fn tf32(x: f32) -> f32 {
    let b = x.to_bits();
    if b & 0x7f80_0000 == 0x7f80_0000 {
        return x;
    }
    f32::from_bits(b.wrapping_add(0x1000) & 0xffff_e000)
}

/// One K-step of 8 as the host form reads it: the products of TF32 operands summed from
/// +0 in K order.
fn kstep(a: impl Fn(usize) -> f32, b: impl Fn(usize) -> f32, k0: usize) -> f32 {
    let mut s = 0.0f32;
    for k in k0..k0 + 8 {
        s += tf32(a(k)) * tf32(b(k));
    }
    s
}

/// `Σ_j P_rj · Δ_j` over a tile's 16 rows in the state kernel's order: four partial sums
/// (rows `2t, 2t + 1, 2t + 8, 2t + 9` from +0), then `(p0 + p1) + (p2 + p3)`.
pub fn ztree(pr: &[f32; TILE], dj: &[f32]) -> f32 {
    let mut p = [0.0f32; 4];
    for (t, pt) in p.iter_mut().enumerate() {
        let mut a = 0.0f32;
        for j in [2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9] {
            a += pr[j] * dj[j];
        }
        *pt = a;
    }
    (p[0] + p[1]) + (p[2] + p[3])
}

/// One run of one value head: its key head's queries and keys (`[n][128]`), its values
/// (`[n][vd]`), write strengths and log decays.
pub struct HeadRun<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub beta: &'a [f32],
    pub g: &'a [f32],
    pub n: usize,
    pub vd: usize,
}

/// The blocked form of one run of one value head from state `s0` (`[128][vd]`), blocks of
/// `chunk`: every position's output (rounded to BF16) and the state after each whole block.
pub fn model(r: &HeadRun<'_>, s0: &[f32], chunk: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
    let vd = r.vd;
    let mut s = s0.to_vec();
    let mut out = vec![0.0f32; r.n * vd];
    let mut states = Vec::new();
    let mut from = 0;
    while from < r.n {
        let n = (r.n - from).min(chunk);
        let at = |x: &[f32], w: usize, row: usize, i: usize| {
            if row < n {
                x[(from + row) * w + i]
            } else {
                0.0
            }
        };
        let rowv = |x: &[f32], w: usize, row: usize| -> Vec<f32> {
            (0..w).map(|i| at(x, w, row, i)).collect()
        };
        let qs: Vec<Vec<f32>> = (0..TILE).map(|row| rowv(r.q, KD, row)).collect();
        let ks: Vec<Vec<f32>> = (0..TILE).map(|row| rowv(r.k, KD, row)).collect();
        let beta: Vec<f32> = (0..TILE).map(|row| at(r.beta, 1, row, 0)).collect();
        let g: Vec<f32> = (0..TILE).map(|row| at(r.g, 1, row, 0)).collect();
        let mut gc = [0.0f32; TILE];
        let mut acc = g[0];
        gc[0] = acc;
        for row in 1..TILE {
            acc += g[row];
            gc[row] = acc;
        }
        let mut a = [[0.0f32; TILE]; TILE];
        let mut pm = [[0.0f32; TILE]; TILE];
        for row in 0..n {
            for j in 0..=row {
                let d = exp32(gc[row] - gc[j]);
                pm[row][j] = dot(&qs[row], &ks[j]) * d;
                if j < row {
                    a[row][j] = (beta[row] * dot(&ks[row], &ks[j])) * d;
                }
            }
        }
        let eg: Vec<f32> = gc.iter().map(|&x| exp32(x)).collect();
        let c: Vec<f32> = (0..TILE).map(|row| beta[row] * eg[row]).collect();
        let egl: Vec<f32> = gc.iter().map(|&x| exp32(gc[chunk - 1] - x)).collect();
        let gamma = exp32(gc[chunk - 1]);
        let mut t = [[0.0f32; TILE]; TILE];
        for j in 0..TILE {
            for row in 0..TILE {
                t[row][j] = if row == j {
                    1.0
                } else if row > j && row < n {
                    let mut acc = a[row][j];
                    for m in j + 1..row {
                        acc += a[row][m] * t[m][j];
                    }
                    -acc
                } else {
                    0.0
                };
            }
        }
        // W (rounded to TF32) and U on the present rows, zero past them.
        let mut w = vec![[0.0f32; KD]; TILE];
        let mut uu = vec![vec![0.0f32; vd]; TILE];
        for row in 0..n {
            for i in 0..KD {
                let mut acc = 0.0f32;
                for j in 0..=row {
                    acc += (t[row][j] * c[j]) * ks[j][i];
                }
                w[row][i] = tf32(acc);
            }
            for col in 0..vd {
                let mut acc = 0.0f32;
                for j in 0..=row {
                    acc += (t[row][j] * beta[j]) * at(r.v, vd, j, col);
                }
                uu[row][col] = acc;
            }
        }
        let mut delta = vec![vec![0.0f32; vd]; TILE];
        let mut y = vec![vec![0.0f32; vd]; TILE];
        for row in 0..TILE {
            for col in 0..vd {
                let (mut x, mut yy) = (0.0f32, 0.0f32);
                for ks_ in 0..KD / 8 {
                    x += kstep(|k| w[row][k], |k| s[k * vd + col], 8 * ks_);
                    yy += kstep(|k| qs[row][k], |k| s[k * vd + col], 8 * ks_);
                }
                delta[row][col] = uu[row][col] - x;
                y[row][col] = yy;
            }
        }
        for row in 0..n {
            for col in 0..vd {
                let dj: Vec<f32> = (0..TILE).map(|j| delta[j][col]).collect();
                let z = ztree(&pm[row], &dj);
                out[(from + row) * vd + col] = bf16_round(eg[row] * y[row][col] + z);
            }
        }
        if n == chunk {
            let mut dl = vec![vec![0.0f32; vd]; TILE];
            for j in 0..n {
                for col in 0..vd {
                    dl[j][col] = egl[j] * delta[j][col];
                }
            }
            for m in 0..KD {
                for col in 0..vd {
                    let mut dd = 0.0f32;
                    for ks_ in 0..TILE / 8 {
                        dd += kstep(|k| ks[k][m], |k| dl[k][col], 8 * ks_);
                    }
                    let sv = &mut s[m * vd + col];
                    *sv = *sv * gamma + dd;
                }
            }
            states.push(s.clone());
        }
        from += n;
    }
    (out, states)
}

/// The token recurrence in f64 from `s0`: `S ← αS; δ = (v − Sᵀk)β; S ← S + kδᵀ; o = Sᵀq`
/// with `α = exp(g)`, unrounded.
pub fn recurrence_f64(r: &HeadRun<'_>, s0: &[f32]) -> Vec<f64> {
    let vd = r.vd;
    let mut s: Vec<f64> = s0.iter().map(|&x| f64::from(x)).collect();
    let mut out = vec![0.0f64; r.n * vd];
    for p in 0..r.n {
        let alpha = f64::from(r.g[p]).exp();
        let beta = f64::from(r.beta[p]);
        let q = &r.q[p * KD..(p + 1) * KD];
        let k = &r.k[p * KD..(p + 1) * KD];
        for x in s.iter_mut() {
            *x *= alpha;
        }
        for col in 0..vd {
            let mut rr = 0.0f64;
            for i in 0..KD {
                rr += s[i * vd + col] * f64::from(k[i]);
            }
            let d = (f64::from(r.v[p * vd + col]) - rr) * beta;
            for i in 0..KD {
                s[i * vd + col] += f64::from(k[i]) * d;
            }
            let mut o = 0.0f64;
            for i in 0..KD {
                o += s[i * vd + col] * f64::from(q[i]);
            }
            out[p * vd + col] = o;
        }
    }
    out
}

/// Inputs of `positions` staged positions of a launch geometry: unit-norm keys and queries
/// (queries scaled by `1 / √128`, as the prep makes them), BF16 values, BF16 write
/// strengths in (0, 1] and log decays of the gates' range.
pub struct Inputs {
    pub k_heads: usize,
    pub v_heads: usize,
    pub vd: usize,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub beta: Vec<f32>,
    pub g: Vec<f32>,
}

impl Inputs {
    pub fn new(
        g: &mut Lcg,
        positions: usize,
        (k_heads, v_heads, vd): (usize, usize, usize),
    ) -> Self {
        let unit = |g: &mut Lcg, scale: f32| -> Vec<f32> {
            let x: Vec<f32> = (0..KD).map(|_| g.f32()).collect();
            let n = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-3);
            x.iter().map(|v| v / n * scale).collect()
        };
        let mut q = Vec::with_capacity(positions * k_heads * KD);
        let mut k = Vec::with_capacity(positions * k_heads * KD);
        for _ in 0..positions * k_heads {
            q.extend(unit(g, 1.0 / (KD as f32).sqrt()));
            k.extend(unit(g, 1.0));
        }
        let v = (0..positions * v_heads * vd).map(|_| g.bf16(2.0)).collect();
        let beta = (0..positions * v_heads)
            .map(|_| bf16_round(0.05 + 0.95 * (0.5 + 0.5 * g.f32())))
            .collect();
        // Decays from about 0.4 to 1 a position, and some of exactly 1.
        let gs = (0..positions * v_heads)
            .map(|i| {
                if i % 37 == 0 {
                    0.0
                } else {
                    -0.9 * (0.5 + 0.5 * g.f32()).powi(3)
                }
            })
            .collect();
        Inputs {
            k_heads,
            v_heads,
            vd,
            q,
            k,
            v,
            beta,
            g: gs,
        }
    }

    fn state_len(&self) -> usize {
        self.v_heads * KD * self.vd
    }

    /// Value head `h`'s run over staged positions `first .. first + n`.
    pub fn head(&self, first: usize, n: usize, h: usize) -> HeadOwned {
        let kh = h / (self.v_heads / self.k_heads);
        let mut o = HeadOwned::default();
        for p in first..first + n {
            let at = (p * self.k_heads + kh) * KD;
            o.q.extend_from_slice(&self.q[at..at + KD]);
            o.k.extend_from_slice(&self.k[at..at + KD]);
            let vh = p * self.v_heads + h;
            o.v.extend_from_slice(&self.v[vh * self.vd..(vh + 1) * self.vd]);
            o.beta.push(self.beta[vh]);
            o.g.push(self.g[vh]);
        }
        o.n = n;
        o.vd = self.vd;
        o
    }
}

#[derive(Default)]
pub struct HeadOwned {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub beta: Vec<f32>,
    pub g: Vec<f32>,
    pub n: usize,
    pub vd: usize,
}

impl HeadOwned {
    pub fn run(&self) -> HeadRun<'_> {
        HeadRun {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            beta: &self.beta,
            g: &self.g,
            n: self.n,
            vd: self.vd,
        }
    }
}

/// A tail cache on a device: `ring` rows a slot of one layer's entries, for `slots` slots.
pub struct Tail {
    pub buf: Buf,
    pub ring: usize,
    floats: usize,
}

impl Tail {
    pub fn new<D: QwenDevice>(dev: &D, inp: &Inputs, slots: usize, ring: usize) -> Tail {
        let entry = chunk_entry_floats(u(inp.k_heads), u(inp.v_heads), 128, u(inp.vd));
        Tail {
            buf: zeros(dev, slots * ring * entry * 4),
            ring,
            floats: slots * ring * entry,
        }
    }

    /// Floats the cache holds.
    pub fn buf_floats(&self) -> usize {
        self.floats
    }
}

/// One run of a launch, as the executor lays one out: its relative 0 at absolute position
/// `pos0` (a block boundary) from a source state (`None`: zero), its `cached` positions
/// read from slot `slot`'s tail cache, its `new` positions staged at `first ..` (outputs
/// there), states stored after the given relative counts, and the new positions from
/// relative `keep_from` on kept in the tail cache.
#[derive(Clone)]
pub struct RunSpec {
    pub first: usize,
    pub cached: usize,
    pub new: usize,
    pub src: Option<Vec<f32>>,
    pub writes: Vec<usize>,
    pub pos0: usize,
    pub slot: usize,
    pub keep_from: usize,
}

impl RunSpec {
    /// A fresh run from position 0: no cached positions, nothing kept.
    pub fn fresh(first: usize, new: usize, src: Option<Vec<f32>>, writes: Vec<usize>) -> Self {
        RunSpec {
            first,
            cached: 0,
            new,
            src,
            writes,
            pos0: 0,
            slot: 0,
            keep_from: usize::MAX,
        }
    }
}

/// What one launch returns: every staged position's output row (zeros where none was
/// written) and each run's stored states, in its `writes` order.
pub struct Launched {
    pub out: Vec<f32>,
    pub states: Vec<Vec<Vec<f32>>>,
}

/// One `dn_chunk` launch of `runs` over `inp` at block length `chunk`, over `tail`.
pub fn launch<D: QwenDevice>(
    dev: &D,
    inp: &Inputs,
    runs: &[RunSpec],
    chunk: usize,
    tail: &Tail,
) -> Result<Launched, String> {
    launch_in(dev, inp, runs, chunk, tail, false)
}

/// [`launch`] in the form named: the fused one (`dn_chunk_fused`, for runs within the ring)
/// or the unfused one.
pub fn launch_in<D: QwenDevice>(
    dev: &D,
    inp: &Inputs,
    runs: &[RunSpec],
    chunk: usize,
    tail: &Tail,
    fused: bool,
) -> Result<Launched, String> {
    let positions = (inp.q.len() / (inp.k_heads * KD)).max(1);
    let sl = inp.state_len();
    let pad = |x: &Vec<f32>| {
        if x.is_empty() {
            vec![0.0; 4]
        } else {
            x.clone()
        }
    };
    let (qb, kb, vb) = (
        buf(dev, &pad(&inp.q)),
        buf(dev, &pad(&inp.k)),
        buf(dev, &pad(&inp.v)),
    );
    let (bb, gb) = (buf(dev, &pad(&inp.beta)), buf(dev, &pad(&inp.g)));
    let ob = zeros(dev, positions * inp.v_heads * inp.vd * 4);
    let mut seqs = Vec::new();
    let mut keep: Vec<(Option<Buf>, Vec<Buf>)> = Vec::new();
    let mut tile = 0usize;
    for r in runs {
        let count = r.cached + r.new;
        let src = r.src.as_ref().map(|s| buf(dev, s));
        let wb: Vec<Buf> = r.writes.iter().map(|_| zeros(dev, sl * 4)).collect();
        let mut d = DnSeq {
            src: src.as_ref().map_or(0, |b| b.addr),
            first: u(r.first),
            count: u(count),
            out_from: u(r.cached),
            live_after: u(count + 1),
            writes: u(r.writes.len()),
            tile: u(tile),
            pos0: u(r.pos0),
            tail_row: u(r.slot * tail.ring),
            keep_from: u32::try_from(r.keep_from).unwrap_or(u32::MAX),
            ..DnSeq::default()
        };
        for (w, (&at, b)) in r.writes.iter().zip(&wb).enumerate() {
            d.write_after[w] = u(at);
            d.write_dst[w] = b.addr;
        }
        tile += count.div_ceil(chunk);
        seqs.push(d);
        keep.push((src, wb));
    }
    let sb = buf(dev, &seqs);
    let (fw, fu, fp, ff) = chunk_scratch_floats(u(inp.vd));
    let n = tile.max(1) * inp.v_heads;
    let (w, uu, pm, fac) = (
        zeros(dev, n * fw * 4),
        zeros(dev, n * fu * 4),
        zeros(dev, n * fp * 4),
        zeros(dev, n * ff * 4),
    );
    let nk = tile.max(1) * inp.k_heads * CHUNK_QK_FLOATS * 4;
    let (qr, kr, kraw) = (zeros(dev, nk), zeros(dev, nk), zeros(dev, nk));
    let rec = zeros(dev, n * CHUNK_REC_FLOATS * 4);
    let p = DnChunkParams {
        seqs: sb.addr,
        q: qb.addr,
        k: kb.addr,
        v: vb.addr,
        beta: bb.addr,
        g: gb.addr,
        out: ob.addr,
        w: w.addr,
        u: uu.addr,
        pm: pm.addr,
        fac: fac.addr,
        qr: qr.addr,
        kr: kr.addr,
        kraw: kraw.addr,
        rec: rec.addr,
        tail: tail.buf.addr,
        state_offset: 0,
        seq_count: u(runs.len()),
        tiles: u(tile),
        k_heads: u(inp.k_heads),
        v_heads: u(inp.v_heads),
        k_dim: 128,
        v_dim: u(inp.vd),
        chunk: u(chunk),
        ring: u(tail.ring),
    };
    // SAFETY: the buffers hold what the struct names.
    if fused {
        unsafe { dev.dn_chunk_fused(&p) }.map_err(|e| e.to_string())?;
    } else {
        unsafe { dev.dn_chunk(&p) }.map_err(|e| e.to_string())?;
    }
    dev.synchronize().map_err(|e| e.to_string())?;
    Ok(Launched {
        out: read::<D, f32>(dev, &ob, positions * inp.v_heads * inp.vd),
        states: keep
            .iter()
            .map(|(_, wb)| wb.iter().map(|b| read::<D, f32>(dev, b, sl)).collect())
            .collect(),
    })
}

/// One sequence served the way the executor serves it, over slot `slot` of a tail cache:
/// its base (a boundary) and the state there, steps of new positions from the base's tail,
/// each reading the positions since the base from the cache, keeping those from the
/// boundary its committed positions reach, and storing the state there.
pub struct Session<'a> {
    pub inp: &'a Inputs,
    pub chunk: usize,
    pub slot: usize,
    pub base: usize,
    pub state: Option<Vec<f32>>,
}

impl Session<'_> {
    /// Positions `start .. start + committed + drafts` (`start` at or past the base, the
    /// cache holding `base .. start`): returns their outputs (`[n][v_heads][vd]`). The
    /// first `committed` advance the base to the last boundary they reach.
    pub fn step<D: QwenDevice>(
        &mut self,
        dev: &D,
        tail: &Tail,
        start: usize,
        committed: usize,
        drafts: usize,
    ) -> Result<Vec<f32>, String> {
        let end = start + committed + drafts;
        let host_end = start + committed;
        let new_base = host_end / self.chunk * self.chunk;
        let mut staged = inp_shape(self.inp);
        append(&mut staged, self.inp, start, end - start);
        let writes = if new_base > self.base {
            vec![new_base - self.base]
        } else {
            Vec::new()
        };
        let run = RunSpec {
            first: 0,
            cached: start - self.base,
            new: end - start,
            src: self.state.clone(),
            writes: writes.clone(),
            pos0: self.base,
            slot: self.slot,
            keep_from: new_base - self.base,
        };
        let r = launch(dev, &staged, &[run], self.chunk, tail)?;
        if !writes.is_empty() {
            self.state = Some(r.states[0][0].clone());
            self.base = new_base;
        }
        let wide = self.inp.v_heads * self.inp.vd;
        Ok(r.out[..(end - start) * wide].to_vec())
    }
}
/// The model's outputs and whole-block states of a run (every head), laid out as a launch
/// lays them: `[n][v_heads][vd]` and `[v_heads][128][vd]` per state.
pub fn model_run(
    inp: &Inputs,
    first: usize,
    n: usize,
    src: Option<&[f32]>,
    chunk: usize,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let (vh, vd) = (inp.v_heads, inp.vd);
    let hs = KD * vd;
    let mut out = vec![0.0f32; n * vh * vd];
    let mut states: Vec<Vec<f32>> = Vec::new();
    for h in 0..vh {
        let s0 = src.map_or_else(|| vec![0.0; hs], |s| s[h * hs..(h + 1) * hs].to_vec());
        let head = inp.head(first, n, h);
        let (o, st) = model(&head.run(), &s0, chunk);
        for p in 0..n {
            out[(p * vh + h) * vd..][..vd].copy_from_slice(&o[p * vd..(p + 1) * vd]);
        }
        if states.is_empty() {
            states = vec![vec![0.0; vh * hs]; st.len()];
        }
        for (b, x) in st.iter().enumerate() {
            states[b][h * hs..(h + 1) * hs].copy_from_slice(x);
        }
    }
    (out, states)
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn same(what: &str, got: &[f32], want: &[f32]) -> Result<(), String> {
    super::cases::same(what, &bits(got), &bits(want))
}

/// The largest `|got − want| / (1 + |want|)`.
fn worst(got: &[f32], want: &[f32]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs() / (1.0 + f64::from(b).abs()))
        .fold(0.0, f64::max)
}

/// Launch geometries: the released grouping (three value heads a key head) at 128-wide
/// values, and the fixture's (64-wide), each block length a serving block size gives.
const GEOMETRIES: [((usize, usize, usize), usize); 4] = [
    ((2, 6, 128), 16),
    ((2, 6, 64), 16),
    ((1, 3, 128), 4),
    ((1, 2, 32), 1),
];

/// Several runs in one launch (from zero and from a state, folding a prefix, outputs from
/// varied positions, states stored at block boundaries) against [`model`]: bit for bit
/// when `exact` (the host form). Else (a GPU, whose tensor cores sum a K-step their own
/// way, so its f32 values differ from the host form's by a few f32 units of the products'
/// magnitude) every stored state within `bound` of the model's relative to `1 + |x|`, and
/// every output within one BF16 unit of its own plus `bound` of the largest output of its
/// run: a difference at the products' scale crosses at most one BF16 rounding boundary of
/// an output of that scale, but an output that cancels to a small value (`exp(G) · Y + Z`)
/// is many of its own units from it. Returns the largest state deviation and the outputs'
/// units apart (0, 1, 2 and more).
/// Several fresh runs in one launch (from zero and from a state, storing at block
/// boundaries, keeping their last block's positions in the tail cache), then each run
/// continued by 7 positions in a second launch (its last boundary's state its source, its
/// positions since read from the cache), against [`model`]: bit for bit when `exact` (the
/// host form). Else (a GPU, whose tensor cores sum a K-step their own way, so its f32
/// values differ from the host form's by a few f32 units of the products' magnitude)
/// every stored state within `bound` of the model's relative to `1 + |x|`, and every
/// output within one BF16 unit of its own plus `bound` of the largest output of its run:
/// a difference at the products' scale crosses at most one BF16 rounding boundary of an
/// output of that scale, but an output that cancels to a small value (`exp(G) · Y + Z`) is
/// many of its own units from it. Returns the largest state deviation and the outputs'
/// units apart (0, 1, 2 and more).
pub fn against_model<D: QwenDevice>(
    dev: &D,
    seed: u64,
    exact: bool,
    bound: f64,
) -> Result<String, String> {
    let mut g = Lcg(seed);
    let mut worst_seen = 0.0f64;
    let mut ulps = [0usize; 3];
    let mut excess = 0.0f64;
    let extra = 7usize;
    for (dims, chunk) in GEOMETRIES {
        let lens = [1usize, chunk, chunk + 1, 3 * chunk + 5, 70, 2];
        let total: usize = lens.iter().map(|n| n + extra).sum();
        let inp = Inputs::new(&mut g, total, dims);
        let sl = inp.state_len();
        let wide = inp.v_heads * inp.vd;
        let what = format!("dn chunk {dims:?} at {chunk}");
        let tail = Tail::new(dev, &inp, lens.len(), chunk + 3);
        let mut runs = Vec::new();
        let mut first = 0;
        let mut firsts = Vec::new();
        for (i, &n) in lens.iter().enumerate() {
            let src: Option<Vec<f32>> =
                (i % 2 == 1).then(|| (0..sl).map(|_| g.f32() * 0.3).collect());
            let whole = n / chunk;
            // At most four stores a run (the kernel's), the last boundary's among them.
            let mut writes: Vec<usize> = (1..whole)
                .filter(|b| b % 2 == i % 2)
                .map(|b| b * chunk)
                .collect();
            writes.reverse();
            writes.truncate(3);
            writes.reverse();
            if whole > 0 {
                writes.push(whole * chunk);
            }
            runs.push(RunSpec {
                first,
                cached: 0,
                new: n,
                src,
                writes,
                pos0: 0,
                slot: i,
                keep_from: whole * chunk,
            });
            firsts.push(first);
            first += n + extra;
        }
        // The first launch reads every run's positions at its own offset.
        let got = launch(dev, &inp, &runs, chunk, &tail)?;
        let mut check_out = |gout: &[f32], wout: &[f32], what: &str| -> Result<(), String> {
            if exact {
                return same(what, gout, wout);
            }
            let scale = wout.iter().fold(0.0f64, |m, &b| m.max(f64::from(b).abs()));
            for (&a, &b) in gout.iter().zip(wout) {
                let d = (super::cases::ord16(eidola_engine_cuda::bf16::from_f32(a))
                    - super::cases::ord16(eidola_engine_cuda::bf16::from_f32(b)))
                .unsigned_abs();
                ulps[(d as usize).min(2)] += 1;
                let unit = f64::from(b).abs() * f64::from(f32::EPSILON) * 65536.0;
                let off = (f64::from(a) - f64::from(b)).abs() - unit.max(f64::MIN_POSITIVE);
                excess = excess.max(off / scale.max(f64::MIN_POSITIVE));
            }
            Ok(())
        };
        let mut second = Vec::new();
        let mut staged = inp_shape(&inp);
        for (i, r) in runs.iter().enumerate() {
            let n = r.new;
            let (out, states) = model_run(&inp, r.first, n, r.src.as_deref(), chunk);
            check_out(
                &got.out[r.first * wide..(r.first + n) * wide],
                &out,
                &format!("{what}: run {i} outputs"),
            )?;
            for (w, &at) in r.writes.iter().enumerate() {
                let want = &states[at / chunk - 1];
                if exact {
                    same(
                        &format!("{what}: run {i} state after {at}"),
                        &got.states[i][w],
                        want,
                    )?;
                } else {
                    worst_seen = worst_seen.max(worst(&got.states[i][w], want));
                }
            }
            // The continuation: from the last boundary's state, the cache holding the
            // positions since.
            let base = n / chunk * chunk;
            let src = if base == 0 {
                r.src.clone()
            } else {
                got.states[i].last().cloned()
            };
            let at = staged.q.len() / (inp.k_heads * KD);
            append(&mut staged, &inp, r.first + n, extra);
            second.push(RunSpec {
                first: at,
                cached: n - base,
                new: extra,
                src,
                writes: Vec::new(),
                pos0: base,
                slot: i,
                keep_from: usize::MAX,
            });
        }
        let got2 = launch(dev, &staged, &second, chunk, &tail)?;
        for (i, (r, c)) in runs.iter().zip(&second).enumerate() {
            let (out, _) = model_run(&inp, r.first, r.new + extra, r.src.as_deref(), chunk);
            check_out(
                &got2.out[c.first * wide..(c.first + extra) * wide],
                &out[r.new * wide..],
                &format!("{what}: run {i} continued from the cache"),
            )?;
        }
        let _ = &firsts;
    }
    if worst_seen > bound || excess > bound {
        return Err(format!(
            "states {worst_seen:e} from the model, outputs {excess:e} of their run's scale past \
             one BF16 unit (bound {bound:e}); outputs 0/1/2+ BF16 units apart: {ulps:?}"
        ));
    }
    Ok(format!(
        "states within {worst_seen:e} of the model, outputs within one BF16 unit and \
         {:e} of their run's scale; 0/1/2+ BF16 units apart: {ulps:?}",
        excess.max(0.0)
    ))
}

/// One run of `n` positions from a random state against the token recurrence, per
/// geometry: the exact profile's (`recurrent_step` in f32, rounded to BF16 as the exact
/// kernel rounds it) in BF16 units in the last place (a histogram of 0, 1, 2, 3 and more),
/// and both against the recurrence in f64, as the largest error relative to the outputs'
/// largest magnitude. The blocked form's f64 error may exceed the exact one's by at most
/// `bound`.
pub fn against_recurrence<D: QwenDevice>(
    dev: &D,
    seed: u64,
    n: usize,
    bound: f64,
) -> Result<String, String> {
    let mut g = Lcg(seed);
    let mut report = Vec::new();
    for (dims, chunk) in GEOMETRIES {
        let inp = Inputs::new(&mut g, n, dims);
        let sl = inp.state_len();
        let src: Vec<f32> = (0..sl).map(|_| g.f32() * 0.3).collect();
        let tail = Tail::new(dev, &inp, 1, 1);
        let got = launch(
            dev,
            &inp,
            &[RunSpec::fresh(0, n, Some(src.clone()), Vec::new())],
            chunk,
            &tail,
        )?;
        let (vh, vd) = (inp.v_heads, inp.vd);
        let hs = KD * vd;
        let (mut err, mut err_exact, mut mag) = (0.0f64, 0.0f64, 0.0f64);
        let mut hist = [0usize; 4];
        for h in 0..vh {
            let head = inp.head(0, n, h);
            let run = head.run();
            let want = recurrence_f64(&run, &src[h * hs..(h + 1) * hs]);
            let mut st = src[h * hs..(h + 1) * hs].to_vec();
            let mut raw = vec![0.0f32; vd];
            for p in 0..n {
                recurrent_step(
                    &mut st,
                    &run.q[p * KD..(p + 1) * KD],
                    &run.k[p * KD..(p + 1) * KD],
                    &run.v[p * vd..(p + 1) * vd],
                    exp32(run.g[p]),
                    run.beta[p],
                    &mut raw,
                );
                for c in 0..vd {
                    let w = want[p * vd + c];
                    let x = got.out[(p * vh + h) * vd + c];
                    let e = bf16_round(raw[c]);
                    err = err.max((f64::from(x) - w).abs());
                    err_exact = err_exact.max((f64::from(e) - w).abs());
                    mag = mag.max(w.abs());
                    let d = (super::cases::ord16(eidola_engine_cuda::bf16::from_f32(x))
                        - super::cases::ord16(eidola_engine_cuda::bf16::from_f32(e)))
                    .unsigned_abs();
                    hist[(d as usize).min(3)] += 1;
                }
            }
        }
        let (rel, rel_exact) = (err / mag.max(1e-30), err_exact / mag.max(1e-30));
        report.push(format!(
            "{dims:?}@{chunk}: BF16 ulp from exact 0/1/2/3+ = {hist:?}, \
             f64 error blocked {rel:.3e}, exact {rel_exact:.3e}"
        ));
        if rel > rel_exact + bound {
            return Err(format!(
                "{dims:?} at {chunk}: {rel:e} of the largest output from the f64 recurrence, \
                 the exact profile's {rel_exact:e}"
            ));
        }
    }
    Ok(format!("over {n} positions: {}", report.join("; ")))
}

/// Reproducibility, bit for bit: one sequence of `len` positions run whole (its states
/// stored at every boundary) against the same sequence served as the executor serves it
/// (`Session`: each step reading its block's earlier positions from the tail cache, so a
/// block's row is computed with 1 to `chunk` of its positions present, the rest zeros in
/// every operand)
///
/// - a position a step (decode);
/// - in chunks of 1, 3, 7, 16, 17, 64 and 65 positions;
/// - a committed position and three drafts a step, accepting none to all of them in turn
///   (the rejected ones computed again, over their cache rows);
///
/// and the whole run beside other runs in one launch, at another offset and slot.
pub fn invariant<D: QwenDevice>(dev: &D, seed: u64, len: usize) -> Result<String, String> {
    let mut g = Lcg(seed);
    let mut checked = 0usize;
    for (dims, chunk) in GEOMETRIES {
        let inp = Inputs::new(&mut g, len + 40, dims);
        let sl = inp.state_len();
        let wide = inp.v_heads * inp.vd;
        let what = format!("{dims:?} at {chunk}");
        let ring = chunk - 1 + 3;
        let bounds: Vec<usize> = (1..=len / chunk).map(|b| b * chunk).collect();
        let none = Tail::new(dev, &inp, 1, 1);
        // The whole run, once per four of its boundaries (a run stores at most four
        // states): the same outputs every time, and every boundary's state.
        let mut states = Vec::new();
        let mut full_out = Vec::new();
        for four in bounds.chunks(4).chain(bounds.is_empty().then_some(&[][..])) {
            let r = launch(
                dev,
                &inp,
                &[RunSpec::fresh(0, len, None, four.to_vec())],
                chunk,
                &none,
            )?;
            let out = r.out[..len * wide].to_vec();
            if full_out.is_empty() {
                full_out = out;
            } else {
                same(&format!("{what}: the whole run again"), &out, &full_out)?;
            }
            states.extend(r.states[0].iter().cloned());
        }
        let full_out = &full_out[..];
        let check_state = |sess: &Session<'_>, at: &str| -> Result<(), String> {
            if sess.base > 0 {
                same(
                    &format!("{what}: the state at {} {at}", sess.base),
                    sess.state.as_deref().expect("a state"),
                    &states[sess.base / chunk - 1],
                )?;
            }
            Ok(())
        };
        // Decoded, and in chunks.
        for c in [1usize, 3, 7, 16, 17, 64, 65] {
            let tail = Tail::new(dev, &inp, 1, ring);
            let mut sess = Session {
                inp: &inp,
                chunk,
                slot: 0,
                base: 0,
                state: None,
            };
            let mut start = 0;
            while start < len {
                let n = c.min(len - start);
                let out = sess.step(dev, &tail, start, n, 0)?;
                same(
                    &format!("{what}: positions {start}..{} in steps of {c}", start + n),
                    &out,
                    &full_out[start * wide..(start + n) * wide],
                )?;
                check_state(&sess, &format!("in steps of {c}"))?;
                checked += n;
                start += n;
            }
        }
        // A committed position and three drafts a step.
        {
            let tail = Tail::new(dev, &inp, 1, ring);
            let mut sess = Session {
                inp: &inp,
                chunk,
                slot: 0,
                base: 0,
                state: None,
            };
            let (mut start, mut step) = (0usize, 0usize);
            while start < len {
                let drafts = 3.min(len - start - 1);
                let out = sess.step(dev, &tail, start, 1, drafts)?;
                same(
                    &format!(
                        "{what}: positions {start}..{} with drafts",
                        start + 1 + drafts
                    ),
                    &out,
                    &full_out[start * wide..(start + 1 + drafts) * wide],
                )?;
                check_state(&sess, "with drafts")?;
                checked += 1 + drafts;
                start += 1 + (step % 4).min(drafts);
                step += 1;
            }
        }
        // Beside other runs, at another offset and slot.
        let mut crowd = inp_shape(&inp);
        let other = Inputs::new(&mut g, 23, dims);
        append(&mut crowd, &other, 0, 23);
        append(&mut crowd, &inp, 0, len);
        append(&mut crowd, &other, 0, 9);
        let src: Vec<f32> = (0..sl).map(|_| g.f32() * 0.3).collect();
        let tail = Tail::new(dev, &inp, 3, ring);
        let mut runs = vec![
            RunSpec::fresh(0, 23, Some(src.clone()), vec![]),
            RunSpec::fresh(23, len, None, bounds.iter().copied().take(4).collect()),
            RunSpec::fresh(23 + len, 9, Some(src), vec![]),
        ];
        for (i, r) in runs.iter_mut().enumerate() {
            r.slot = i;
        }
        let r = launch(dev, &crowd, &runs, chunk, &tail)?;
        same(
            &format!("{what}: beside other runs"),
            &r.out[23 * wide..(23 + len) * wide],
            full_out,
        )?;
        for (w, s) in r.states[1].iter().enumerate() {
            same(
                &format!("{what}: state {w} beside other runs"),
                s,
                &states[w],
            )?;
        }
        checked += len;
    }
    Ok(format!("{checked} positions"))
}

/// The fused form against the unfused one, bit for bit: `rows` sequences in one launch a
/// step, each served as the executor serves it over its own slot (a committed position
/// and three drafts a step, accepting none to all, so its folds take every value up to the
/// tail's), at every geometry; once in each form over its own tail cache: every step's
/// outputs, every state stored, and the two tail caches at the end.
pub fn fused_equals_unfused<D: QwenDevice>(
    dev: &D,
    seed: u64,
    rows: &[usize],
) -> Result<String, String> {
    struct Seq {
        base: usize,
        start: usize,
        state: Option<Vec<f32>>,
    }
    let mut g = Lcg(seed);
    let mut checked = 0usize;
    for (dims, chunk) in GEOMETRIES {
        for &rn in rows {
            let len = 3 * chunk + 5;
            let inp = Inputs::new(&mut g, len + rn, dims);
            let wide = inp.v_heads * inp.vd;
            // The executor's ring: a decode row's whole run (its tail, host token, drafts).
            let ring = chunk - 1 + 3 + 1 + 3;
            let what = format!("{dims:?} at {chunk}, {rn} rows");
            let tails = [
                Tail::new(dev, &inp, rn, ring),
                Tail::new(dev, &inp, rn, ring),
            ];
            let mut seqs: [Vec<Seq>; 2] = std::array::from_fn(|_| {
                (0..rn)
                    .map(|_| Seq {
                        base: 0,
                        start: 0,
                        state: None,
                    })
                    .collect()
            });
            let mut step = 0usize;
            loop {
                let live: Vec<usize> = (0..rn).filter(|&i| seqs[0][i].start < len).collect();
                if live.is_empty() {
                    break;
                }
                let mut outs = Vec::new();
                let mut stored = Vec::new();
                for form in 0..2 {
                    let mut staged = inp_shape(&inp);
                    let mut runs = Vec::new();
                    let mut spans = Vec::new();
                    for &i in &live {
                        let sq = &seqs[form][i];
                        let drafts = 3.min(len - sq.start - 1);
                        let n = 1 + drafts;
                        let host_end = sq.start + 1;
                        let new_base = host_end / chunk * chunk;
                        let first = staged.beta.len() / inp.v_heads;
                        // Sequence `i` is the inputs from position `i` on.
                        append(&mut staged, &inp, i + sq.start, n);
                        runs.push(RunSpec {
                            first,
                            cached: sq.start - sq.base,
                            new: n,
                            src: sq.state.clone(),
                            writes: if new_base > sq.base {
                                vec![new_base - sq.base]
                            } else {
                                Vec::new()
                            },
                            pos0: sq.base,
                            slot: i,
                            keep_from: new_base - sq.base,
                        });
                        spans.push((first, n, new_base));
                    }
                    let r = launch_in(dev, &staged, &runs, chunk, &tails[form], form == 1)?;
                    let mut o = Vec::new();
                    let mut st = Vec::new();
                    for (k, &i) in live.iter().enumerate() {
                        let (first, n, new_base) = spans[k];
                        o.push(r.out[first * wide..(first + n) * wide].to_vec());
                        let sq = &mut seqs[form][i];
                        if !runs[k].writes.is_empty() {
                            sq.state = Some(r.states[k][0].clone());
                            sq.base = new_base;
                            st.push(sq.state.clone().expect("a state"));
                        }
                        // Accept none to all of the drafts, by step and sequence.
                        let accepted = ((step + i) % 4).min(n - 1);
                        sq.start += 1 + accepted;
                    }
                    outs.push(o);
                    stored.push(st);
                }
                for (k, &i) in live.iter().enumerate() {
                    same(
                        &format!("{what}: step {step}, sequence {i}: outputs"),
                        &outs[1][k],
                        &outs[0][k],
                    )?;
                    checked += outs[0][k].len() / wide;
                }
                for (w, (a, b)) in stored[1].iter().zip(&stored[0]).enumerate() {
                    same(&format!("{what}: step {step}: state {w}"), a, b)?;
                }
                step += 1;
            }
            let n = tails[0].buf_floats();
            same(
                &format!("{what}: the tail caches"),
                &read::<D, f32>(dev, &tails[1].buf, n),
                &read::<D, f32>(dev, &tails[0].buf, n),
            )?;
        }
    }
    Ok(format!("{checked} positions"))
}

fn inp_shape(inp: &Inputs) -> Inputs {
    Inputs {
        k_heads: inp.k_heads,
        v_heads: inp.v_heads,
        vd: inp.vd,
        q: Vec::new(),
        k: Vec::new(),
        v: Vec::new(),
        beta: Vec::new(),
        g: Vec::new(),
    }
}

/// Appends `src`'s staged positions `from .. from + n` to `dst`.
fn append(dst: &mut Inputs, src: &Inputs, from: usize, n: usize) {
    let kw = src.k_heads * KD;
    let vw = src.v_heads * src.vd;
    dst.q.extend_from_slice(&src.q[from * kw..(from + n) * kw]);
    dst.k.extend_from_slice(&src.k[from * kw..(from + n) * kw]);
    dst.v.extend_from_slice(&src.v[from * vw..(from + n) * vw]);
    dst.beta
        .extend_from_slice(&src.beta[from * src.v_heads..(from + n) * src.v_heads]);
    dst.g
        .extend_from_slice(&src.g[from * src.v_heads..(from + n) * src.v_heads]);
}

/// The intra kernel's scratch after one launch over `runs` (`W`, `U`, `P`, the factors,
/// the rounded queries and keys, the keys, the rows of `T`, write strengths and decays, in
/// that order), for holding a device's bits to the host body's.
pub fn intra_scratch<D: QwenDevice>(
    dev: &D,
    intra: &dyn Fn(&DnChunkParams) -> Result<(), String>,
    seed: u64,
) -> Result<Vec<Vec<u32>>, String> {
    let mut g = Lcg(seed);
    let mut all = Vec::new();
    for (dims, chunk) in GEOMETRIES {
        let lens = [1usize, chunk, 2 * chunk + 3, 40];
        let total: usize = lens.iter().sum();
        let inp = Inputs::new(&mut g, total, dims);
        let mut seqs = Vec::new();
        let (mut first, mut tile) = (0usize, 0usize);
        for &n in &lens {
            seqs.push(DnSeq {
                first: u(first),
                count: u(n),
                live_after: u(n + 1),
                tile: u(tile),
                keep_from: u32::MAX,
                ..DnSeq::default()
            });
            first += n;
            tile += n.div_ceil(chunk);
        }
        let (qb, kb, vb) = (buf(dev, &inp.q), buf(dev, &inp.k), buf(dev, &inp.v));
        let (bb, gb) = (buf(dev, &inp.beta), buf(dev, &inp.g));
        let ob = zeros(dev, 8);
        let sb = buf(dev, &seqs);
        let tail = Tail::new(dev, &inp, 1, 1);
        let (fw, fu, fp, ff) = chunk_scratch_floats(u(inp.vd));
        let n = tile * inp.v_heads;
        let nk = tile * inp.k_heads;
        let lens6 = [
            n * fw,
            n * fu,
            n * fp,
            n * ff,
            nk * CHUNK_QK_FLOATS,
            nk * CHUNK_QK_FLOATS,
            nk * CHUNK_QK_FLOATS,
            n * CHUNK_REC_FLOATS,
        ];
        let bufs: Vec<Buf> = lens6.iter().map(|&l| zeros(dev, l * 4)).collect();
        let p = DnChunkParams {
            seqs: sb.addr,
            q: qb.addr,
            k: kb.addr,
            v: vb.addr,
            beta: bb.addr,
            g: gb.addr,
            out: ob.addr,
            w: bufs[0].addr,
            u: bufs[1].addr,
            pm: bufs[2].addr,
            fac: bufs[3].addr,
            qr: bufs[4].addr,
            kr: bufs[5].addr,
            kraw: bufs[6].addr,
            rec: bufs[7].addr,
            tail: tail.buf.addr,
            state_offset: 0,
            seq_count: u(lens.len()),
            tiles: u(tile),
            k_heads: u(inp.k_heads),
            v_heads: u(inp.v_heads),
            k_dim: 128,
            v_dim: u(inp.vd),
            chunk: u(chunk),
            ring: 1,
        };
        intra(&p)?;
        dev.synchronize().map_err(|e| e.to_string())?;
        for (b, &len) in bufs.iter().zip(&lens6) {
            all.push(read::<D, u32>(dev, b, len));
        }
    }
    Ok(all)
}
