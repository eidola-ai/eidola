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
        for row in 0..TILE {
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
                t[row][j] = if row < j {
                    0.0
                } else if row == j {
                    1.0
                } else {
                    let mut acc = a[row][j];
                    for m in j + 1..row {
                        acc += a[row][m] * t[m][j];
                    }
                    -acc
                };
            }
        }
        let mut w = vec![[0.0f32; KD]; TILE];
        let mut uu = vec![vec![0.0f32; vd]; TILE];
        for row in 0..TILE {
            for i in 0..KD {
                let mut acc = 0.0f32;
                for j in 0..=row {
                    acc += (t[row][j] * c[j]) * ks[j][i];
                }
                w[row][i] = acc;
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
                let (mut x0, mut x1, mut y0, mut y1) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for ks_ in 0..KD / 8 {
                    let sx = kstep(|k| w[row][k], |k| s[k * vd + col], 8 * ks_);
                    let sy = kstep(|k| qs[row][k], |k| s[k * vd + col], 8 * ks_);
                    if ks_ % 2 == 0 {
                        x0 += sx;
                        y0 += sy;
                    } else {
                        x1 += sx;
                        y1 += sy;
                    }
                }
                delta[row][col] = uu[row][col] - (x0 + x1);
                y[row][col] = y0 + y1;
            }
        }
        let mut dl = vec![vec![0.0f32; vd]; TILE];
        for row in 0..n {
            for col in 0..vd {
                let mut acc = 0.0f32;
                for j in 0..=row {
                    acc += pm[row][j] * delta[j][col];
                }
                out[(from + row) * vd + col] = bf16_round(eg[row] * y[row][col] + acc);
                dl[row][col] = egl[row] * delta[row][col];
            }
        }
        if n == chunk {
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

/// One run of a launch: staged positions `first .. first + count`, a source state (`None`:
/// zero), outputs from `out_from`, and states stored after the given counts.
#[derive(Clone)]
pub struct RunSpec {
    pub first: usize,
    pub count: usize,
    pub src: Option<Vec<f32>>,
    pub out_from: usize,
    pub writes: Vec<usize>,
}

/// What one launch returns: every staged position's output row (zeros where none was
/// written) and each run's stored states, in its `writes` order.
pub struct Launched {
    pub out: Vec<f32>,
    pub states: Vec<Vec<Vec<f32>>>,
}

/// One `dn_chunk` launch of `runs` over `inp` at block length `chunk`.
pub fn launch<D: QwenDevice>(
    dev: &D,
    inp: &Inputs,
    runs: &[RunSpec],
    chunk: usize,
) -> Result<Launched, String> {
    let positions = inp.q.len() / (inp.k_heads * KD);
    let sl = inp.state_len();
    let (qb, kb, vb) = (buf(dev, &inp.q), buf(dev, &inp.k), buf(dev, &inp.v));
    let (bb, gb) = (buf(dev, &inp.beta), buf(dev, &inp.g));
    let ob = zeros(dev, positions * inp.v_heads * inp.vd * 4);
    let mut seqs = Vec::new();
    let mut keep: Vec<(Option<Buf>, Vec<Buf>)> = Vec::new();
    let mut tile = 0usize;
    for r in runs {
        let src = r.src.as_ref().map(|s| buf(dev, s));
        let wb: Vec<Buf> = r.writes.iter().map(|_| zeros(dev, sl * 4)).collect();
        let mut d = DnSeq {
            src: src.as_ref().map_or(0, |b| b.addr),
            first: u(r.first),
            count: u(r.count),
            out_from: u(r.out_from),
            live_after: u(r.count + 1),
            writes: u(r.writes.len()),
            tile: u(tile),
            ..DnSeq::default()
        };
        for (w, (&at, b)) in r.writes.iter().zip(&wb).enumerate() {
            d.write_after[w] = u(at);
            d.write_dst[w] = b.addr;
        }
        tile += r.count.div_ceil(chunk);
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
        state_offset: 0,
        seq_count: u(runs.len()),
        tiles: u(tile),
        k_heads: u(inp.k_heads),
        v_heads: u(inp.v_heads),
        k_dim: 128,
        v_dim: u(inp.vd),
        chunk: u(chunk),
        reserved: 0,
    };
    // SAFETY: the buffers hold what the struct names.
    unsafe { dev.dn_chunk(&p) }.map_err(|e| e.to_string())?;
    dev.synchronize().map_err(|e| e.to_string())?;
    Ok(Launched {
        out: read::<D, f32>(dev, &ob, positions * inp.v_heads * inp.vd),
        states: keep
            .iter()
            .map(|(_, wb)| wb.iter().map(|b| read::<D, f32>(dev, b, sl)).collect())
            .collect(),
    })
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
/// when `exact` (the host form); else every stored state within `bound` of the model's
/// (relative to `1 + |x|`) and every output (rounded to BF16) within one BF16 unit of it.
/// Returns the largest state deviation and the outputs' units apart (0, 1, more).
pub fn against_model<D: QwenDevice>(
    dev: &D,
    seed: u64,
    exact: bool,
    bound: f64,
) -> Result<String, String> {
    let mut g = Lcg(seed);
    let mut worst_seen = 0.0f64;
    let mut ulps = [0usize; 3];
    for (dims, chunk) in GEOMETRIES {
        let lens = [1usize, chunk, chunk + 1, 3 * chunk + 5, 70, 2];
        let total: usize = lens.iter().sum();
        let inp = Inputs::new(&mut g, total, dims);
        let sl = inp.state_len();
        let mut runs = Vec::new();
        let mut first = 0;
        for (i, &n) in lens.iter().enumerate() {
            let src = (i % 2 == 1).then(|| (0..sl).map(|_| g.f32() * 0.3).collect());
            let whole = n / chunk;
            // At most four stores a run (the kernel's), spread over its boundaries.
            let mut writes: Vec<usize> = (0..=whole)
                .filter(|b| b % 2 == i % 2)
                .map(|b| b * chunk)
                .collect();
            writes.reverse();
            let mut writes: Vec<usize> =
                writes.into_iter().step_by(1 + whole / 8).take(4).collect();
            writes.reverse();
            runs.push(RunSpec {
                first,
                count: n,
                src,
                out_from: if i % 3 == 2 { n / 2 } else { 0 },
                writes,
            });
            first += n;
        }
        let got = launch(dev, &inp, &runs, chunk)?;
        let what = format!("dn chunk {dims:?} at {chunk}");
        let wide = inp.v_heads * inp.vd;
        for (i, r) in runs.iter().enumerate() {
            let (out, states) = model_run(&inp, r.first, r.count, r.src.as_deref(), chunk);
            let gout = &got.out[(r.first + r.out_from) * wide..(r.first + r.count) * wide];
            let wout = &out[r.out_from * wide..];
            if exact {
                same(&format!("{what}: run {i} outputs"), gout, wout)?;
            } else {
                for (&a, &b) in gout.iter().zip(wout) {
                    let d = (super::cases::ord16(eidola_engine_cuda::bf16::from_f32(a))
                        - super::cases::ord16(eidola_engine_cuda::bf16::from_f32(b)))
                    .unsigned_abs();
                    ulps[(d as usize).min(2)] += 1;
                }
            }
            same(
                &format!("{what}: run {i} positions before its outputs"),
                &got.out[r.first * wide..(r.first + r.out_from) * wide],
                &vec![0.0; r.out_from * wide],
            )?;
            for (w, &at) in r.writes.iter().enumerate() {
                let want = if at == 0 {
                    r.src.clone().unwrap_or_else(|| vec![0.0; sl])
                } else {
                    states[at / chunk - 1].clone()
                };
                if exact || at == 0 {
                    same(
                        &format!("{what}: run {i} state after {at}"),
                        &got.states[i][w],
                        &want,
                    )?;
                } else {
                    worst_seen = worst_seen.max(worst(&got.states[i][w], &want));
                }
            }
        }
    }
    if worst_seen > bound || ulps[2] > 0 {
        return Err(format!(
            "states {worst_seen:e} from the model (bound {bound:e}); outputs 0/1/2+ BF16 units \
             apart: {ulps:?}"
        ));
    }
    Ok(format!(
        "states within {worst_seen:e} of the model; outputs 0/1/2+ BF16 units apart: {ulps:?}"
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
        let got = launch(
            dev,
            &inp,
            &[RunSpec {
                first: 0,
                count: n,
                src: Some(src.clone()),
                out_from: 0,
                writes: Vec::new(),
            }],
            chunk,
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
/// stored at every boundary) against
///
/// - every position decoded alone: a run from its block's boundary state folding the
///   block's earlier positions (so a block's row is computed with 1 to `chunk` of its
///   positions present, the rest zeros in every operand);
/// - the sequence cut into chunks of 1, 3, 7, 16, 17, 64 and 65 positions, each chunk a
///   launch resuming from the last boundary before it (folding to its start), storing the
///   boundaries it reaches;
/// - the whole run beside other runs in one launch, at another staged offset.
pub fn invariant<D: QwenDevice>(dev: &D, seed: u64, len: usize) -> Result<String, String> {
    let mut g = Lcg(seed);
    let mut checked = 0usize;
    for (dims, chunk) in GEOMETRIES {
        let inp = Inputs::new(&mut g, len + 40, dims);
        let sl = inp.state_len();
        let wide = inp.v_heads * inp.vd;
        let what = format!("{dims:?} at {chunk}");
        let bounds: Vec<usize> = (1..=len / chunk).map(|b| b * chunk).collect();
        // The whole run, once per four of its boundaries (a run stores at most four
        // states): the same outputs every time, and every boundary's state.
        let mut states = Vec::new();
        let mut full_out = Vec::new();
        for four in bounds.chunks(4).chain(bounds.is_empty().then_some(&[][..])) {
            let r = launch(
                dev,
                &inp,
                &[RunSpec {
                    first: 0,
                    count: len,
                    src: None,
                    out_from: 0,
                    writes: four.to_vec(),
                }],
                chunk,
            )?;
            let out = r.out[..len * wide].to_vec();
            if full_out.is_empty() {
                full_out = out;
            } else {
                same(&format!("{what}: the whole run again"), &out, &full_out)?;
            }
            states.extend(r.states[0].iter().cloned());
        }
        let state_at =
            |pos: usize| -> Option<Vec<f32>> { (pos > 0).then(|| states[pos / chunk - 1].clone()) };
        let full_out = &full_out[..];
        // Every position decoded from its boundary.
        let mut runs = Vec::new();
        let mut at = Vec::new();
        let mut first = 0;
        for t in 0..len {
            let base = t / chunk * chunk;
            let n = t - base + 1;
            // Its own copy of the block's inputs, so every run has its own positions.
            runs.push((base, n, first));
            at.push(first + n - 1);
            first += n;
        }
        let mut stacked = Inputs {
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            beta: Vec::new(),
            g: Vec::new(),
            ..inp_shape(&inp)
        };
        for &(base, n, _) in &runs {
            append(&mut stacked, &inp, base, n);
        }
        let specs: Vec<RunSpec> = runs
            .iter()
            .map(|&(base, n, first)| RunSpec {
                first,
                count: n,
                src: state_at(base),
                out_from: n - 1,
                writes: Vec::new(),
            })
            .collect();
        let dec = launch(dev, &stacked, &specs, chunk)?;
        for (t, &row) in at.iter().enumerate() {
            same(
                &format!("{what}: position {t} decoded from its boundary"),
                &dec.out[row * wide..(row + 1) * wide],
                &full_out[t * wide..(t + 1) * wide],
            )?;
            checked += 1;
        }
        // Chunked prefills.
        for c in [1usize, 3, 7, 16, 17, 64, 65] {
            let mut start = 0;
            while start < len {
                let end = (start + c).min(len);
                let base = start / chunk * chunk;
                let writes: Vec<usize> = bounds
                    .iter()
                    .filter(|&&b| b > base && b <= end)
                    .map(|&b| b - base)
                    .take(4)
                    .collect();
                let r = launch(
                    dev,
                    &inp,
                    &[RunSpec {
                        first: base,
                        count: end - base,
                        src: state_at(base),
                        out_from: start - base,
                        writes: writes.clone(),
                    }],
                    chunk,
                )?;
                same(
                    &format!("{what}: positions {start}..{end} in chunks of {c}"),
                    &r.out[start * wide..end * wide],
                    &full_out[start * wide..end * wide],
                )?;
                for (w, &b) in writes.iter().enumerate() {
                    same(
                        &format!("{what}: the state at {} in chunks of {c}", base + b),
                        &r.states[0][w],
                        &states[(base + b) / chunk - 1],
                    )?;
                }
                checked += end - start;
                start = end;
            }
        }
        // Beside other runs, at another offset.
        let mut crowd = Inputs {
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            beta: Vec::new(),
            g: Vec::new(),
            ..inp_shape(&inp)
        };
        let other = Inputs::new(&mut g, 23, dims);
        append(&mut crowd, &other, 0, 23);
        append(&mut crowd, &inp, 0, len);
        append(&mut crowd, &other, 0, 9);
        let src: Vec<f32> = (0..sl).map(|_| g.f32() * 0.3).collect();
        let r = launch(
            dev,
            &crowd,
            &[
                RunSpec {
                    first: 0,
                    count: 23,
                    src: Some(src.clone()),
                    out_from: 0,
                    writes: vec![],
                },
                RunSpec {
                    first: 23,
                    count: len,
                    src: None,
                    out_from: 0,
                    writes: bounds.iter().copied().take(4).collect(),
                },
                RunSpec {
                    first: 23 + len,
                    count: 9,
                    src: Some(src),
                    out_from: 2,
                    writes: vec![],
                },
            ],
            chunk,
        )?;
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

/// The intra kernel's scratch after one launch over `runs` (`W`, `U`, `P` and the
/// factors, in that order), for holding a device's bits to the host body's.
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
                ..DnSeq::default()
            });
            first += n;
            tile += n.div_ceil(chunk);
        }
        let (qb, kb, vb) = (buf(dev, &inp.q), buf(dev, &inp.k), buf(dev, &inp.v));
        let (bb, gb) = (buf(dev, &inp.beta), buf(dev, &inp.g));
        let ob = zeros(dev, 8);
        let sb = buf(dev, &seqs);
        let (fw, fu, fp, ff) = chunk_scratch_floats(u(inp.vd));
        let n = tile * inp.v_heads;
        let bufs = [
            zeros(dev, n * fw * 4),
            zeros(dev, n * fu * 4),
            zeros(dev, n * fp * 4),
            zeros(dev, n * ff * 4),
        ];
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
            state_offset: 0,
            seq_count: u(lens.len()),
            tiles: u(tile),
            k_heads: u(inp.k_heads),
            v_heads: u(inp.v_heads),
            k_dim: 128,
            v_dim: u(inp.vd),
            chunk: u(chunk),
            reserved: 0,
        };
        intra(&p)?;
        dev.synchronize().map_err(|e| e.to_string())?;
        for (b, len) in bufs.iter().zip([fw, fu, fp, ff]) {
            all.push(read::<D, u32>(dev, b, n * len));
        }
    }
    Ok(all)
}
