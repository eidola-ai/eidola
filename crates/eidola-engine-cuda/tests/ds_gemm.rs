//! The DeepSeek-V4.1 quantised GEMMs on tensor cores (`ds_gemm_tc`) on a
//! device, against the scalar reference (`ds_reference`, itself held to the
//! model crate's `lin8` / `lin4` bit for bit by `tests/ds_gemm_ref.rs`).
//!
//! `device_suite`, on every image this device runs:
//!
//! - **The probe** (`eidola_ds_gemm_probe`): the block product both forms run,
//!   against the exact sum, over every band pair of the banded form in four
//!   families of operands (uniform, extremes, carries, cancellation), and the
//!   whole codes unbanded. A banded pair must return the exact sum every time,
//!   and no element's bits may move with its place in the fragment. The
//!   unbanded counts are what the tolerance form's block products are: they
//!   are recorded, not required.
//! - **The banded form equals the reference** byte for byte (padding rows of
//!   a layout untouched), in every block shape: small random shapes (every grid
//!   remainder, K not a multiple of the staged step, layouts with empty
//!   groups at row alignments 1, 3, 8 and 16), and the released model's
//!   dense and expert shapes from 1 row up, on model-like and arbitrary codes.
//! - **The tolerance form is within its bound**, over the same cases, and
//!   the same bits in every block shape: per output the distance in BF16
//!   units in the last place from the reference, recorded as a histogram with
//!   the largest relative error (`TOLERANCE_*`).
//! - **Batch invariance**: a row computed alone (a split block) is the same
//!   bits as in a batch (32-row tiles), and an expert's rows the same in a
//!   layout holding only them (split blocks) as in the full layout, in both
//!   forms.
//!
//! The suite writes `ds_gemm.json` (every check, the probe's counts and first
//! differing records, the tolerance measurements) under
//! `EIDOLA_DS_SUMMARY_DIR`, else the test target's temporary directory.
//! `EIDOLA_DS_PROBE_DOTS` sets the dot products per probe case (default
//! 2^33, about 8.6 billion). Skipped without a device or `EIDOLA_ENGINE_KERNELS_DIR`.

mod common;
mod ds_gemm_data;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

use common::setup;
use ds_gemm_data::{Rng, activations, fp4_weights, fp4_weights_tiled, fp8_weight, layout};
use eidola_engine_cuda::ds::abi::{Lin4Params, Lin8Params};
use eidola_engine_cuda::ds::gemm::{
    DsGemmTc, PROBE_A_E2M1, PROBE_A_E4M3, PROBE_CANCEL, PROBE_CARRY, PROBE_COUNTS, PROBE_EXTREMES,
    PROBE_RECORD_WORDS, PROBE_UNIFORM, ProbeParams, TcForm, TokenTile, W8_BANDS, X4_BANDS,
    X8_BANDS, probe_band,
};
use eidola_engine_cuda::ds::gemm_ref::DsGemmRef;
use eidola_engine_cuda::launch::{dptr, dptr_at};
use eidola_engine_cuda::{Gpu, ImageArch};
use serde_json::{Value, json};

/// The tolerance form's bound: every output within this many BF16 units in
/// the last place of the reference, or within `TOLERANCE_RMS` times its row's
/// root mean square (an output near zero after cancellation). On a B300 every
/// measured output was within it: 286 of 21,284,807 `lin8` outputs differed
/// (all but 20 by one unit; at most 16 units, 0.0188 of the row's root mean
/// square), no `lin4` output did. Indexing or band faults move outputs by far
/// more; the histogram is recorded beside it.
const TOLERANCE_ULPS: u32 = 2;
const TOLERANCE_RMS: f64 = 1.0 / 1024.0;

/// Dense FP8 weights of the released model `[n, k]` (`wq_a`, `wq_b`, `wkv`,
/// `wo_b`, the shared expert's `w1` / `w3` and `w2`, the index queries'
/// `wq_b`, Engram's `wkv`, the drafter's `main_proj`).
const DENSE: [(u32, u32); 9] = [
    (1280, 5120),
    (32_768, 1280),
    (512, 5120),
    (5120, 8192),
    (2304, 5120),
    (5120, 2304),
    (4096, 1280),
    (25_600, 6144),
    (5120, 15_360),
];
/// The routed experts: gate and up fused `[4608, 5120]`, down `[5120, 2304]`;
/// 384 experts, 6 a token.
const EXPERTS: [(u32, u32); 2] = [(4608, 5120), (5120, 2304)];
const GROUPS: u32 = 384;
const ACTIVE: u32 = 6;

struct Report {
    checks: Vec<Value>,
    measured: Vec<Value>,
    failed: usize,
}

impl Report {
    fn check(&mut self, name: impl Into<String>, body: impl FnOnce(&mut Vec<Value>)) {
        let name = name.into();
        let start = Instant::now();
        let mut measured = Vec::new();
        let outcome = catch_unwind(AssertUnwindSafe(|| body(&mut measured)));
        let seconds = start.elapsed().as_secs_f64();
        let error = outcome.err().map(|e| {
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_else(|| "panic".to_owned())
        });
        match &error {
            None => eprintln!("ok      {name} ({seconds:.1} s)"),
            Some(e) => {
                self.failed += 1;
                eprintln!("FAILED  {name}: {e}");
            }
        }
        for m in &measured {
            eprintln!("measure {m}");
        }
        self.measured.extend(measured);
        self.checks.push(
            json!({"check": name, "ok": error.is_none(), "error": error, "seconds": seconds}),
        );
    }

    fn finish(self, gpu: &Gpu) {
        let dir = std::env::var_os("EIDOLA_DS_SUMMARY_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ds_gemm.json");
        let info = gpu.info();
        let summary = json!({
            "suite": "ds_gemm",
            "device": {
                "name": info.name,
                "compute_capability": [info.compute_capability.0, info.compute_capability.1],
                "sm_count": info.sm_count,
                "driver_version": info.driver_version,
            },
            "tolerance_bound": {"ulps": TOLERANCE_ULPS, "rms": TOLERANCE_RMS},
            "passed": self.checks.len() - self.failed,
            "failed": self.failed,
            "checks": self.checks,
            "measured": self.measured,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
        eprintln!(
            "ds_gemm: {} checks, {} failed; summary in {}",
            self.checks.len(),
            self.failed,
            path.display()
        );
        assert_eq!(self.failed, 0, "{} checks failed", self.failed);
    }
}

struct Kernels {
    reference: DsGemmRef,
    tc: DsGemmTc,
}

fn u(x: usize) -> u32 {
    u32::try_from(x).unwrap()
}
fn us(x: u32) -> usize {
    usize::try_from(x).unwrap()
}

// ---- the probe ----

struct ProbeCase {
    name: String,
    a_band: (u32, u32),
    b_band: (u32, u32),
    a_format: u32,
    family: u32,
    /// The band pair's sums are exactly representable: every one must be exact.
    banded: bool,
}

fn probe_cases() -> Vec<ProbeCase> {
    let families = [
        (PROBE_UNIFORM, "uniform"),
        (PROBE_EXTREMES, "extremes"),
        (PROBE_CARRY, "carry"),
        (PROBE_CANCEL, "cancel"),
    ];
    let mut cases = Vec::new();
    for &(family, fname) in &families {
        for &w in &W8_BANDS {
            for &x in &X8_BANDS {
                cases.push(ProbeCase {
                    name: format!("fp8/w{}-{}/x{}-{}/{fname}", w.0, w.1, x.0, x.1),
                    a_band: w,
                    b_band: x,
                    a_format: PROBE_A_E4M3,
                    family,
                    banded: true,
                });
            }
        }
        for &x in &X4_BANDS {
            cases.push(ProbeCase {
                name: format!("fp4/x{}-{}/{fname}", x.0, x.1),
                a_band: (0, 15),
                b_band: x,
                a_format: PROBE_A_E2M1,
                family,
                banded: true,
            });
        }
        for (a_format, what) in [(PROBE_A_E4M3, "fp8"), (PROBE_A_E2M1, "fp4")] {
            cases.push(ProbeCase {
                name: format!("{what}/unbanded/{fname}"),
                a_band: (0, 15),
                b_band: (0, 15),
                a_format,
                family,
                banded: false,
            });
        }
    }
    cases
}

const MAX_RECORDS: u32 = 16;

fn run_probe(gpu: &Gpu, tc: &DsGemmTc, c: &ProbeCase, dots: u64) -> Value {
    let s = gpu.stream();
    let blocks = gpu.info().sm_count * 8;
    let per_trial = u64::from(blocks) * 4 * 128;
    let trials = u32::try_from(dots.div_ceil(per_trial).max(1)).unwrap();
    let counts = s.alloc_zeros::<u64>(PROBE_COUNTS).unwrap();
    let records = s
        .alloc_zeros::<u32>(us(MAX_RECORDS) * PROBE_RECORD_WORDS)
        .unwrap();
    let p = ProbeParams {
        counts: dptr(&counts, s),
        records: dptr(&records, s),
        seed: 0x5eed_0000 ^ u64::from(c.family) << 8 ^ u64::from(c.a_format),
        trials,
        max_records: MAX_RECORDS,
        a_band: probe_band(c.a_band.0, c.a_band.1),
        b_band: probe_band(c.b_band.0, c.b_band.1),
        a_format: c.a_format,
        family: c.family,
    };
    // SAFETY: both arrays are sized as the probe documents.
    unsafe { tc.probe(gpu, &p, blocks) }.unwrap();
    gpu.synchronize().unwrap();
    let n = s.clone_dtoh(&counts).unwrap();
    let r = s.clone_dtoh(&records).unwrap();
    let kept = us(u32::try_from(n[4].min(u64::from(MAX_RECORDS))).unwrap());
    let recs: Vec<Value> = r
        .chunks_exact(PROBE_RECORD_WORDS)
        .take(kept)
        .map(|w| {
            json!({
                "warp": w[0], "trial": w[1], "row": w[2], "column": w[3],
                "got": format!("{:#010x}", w[4]), "exact_rounded": format!("{:#010x}", w[5]),
                "exact_units": i64::from(w[6]) | i64::from(w[7]) << 32,
                "got_value": f32::from_bits(w[4]), "exact_value": f32::from_bits(w[5]),
            })
        })
        .collect();
    json!({
        "probe": c.name, "banded": c.banded, "checked": n[0], "differing": n[1],
        "moved": n[2], "worst_ulps": n[3], "records": recs,
    })
}

// ---- products ----

struct Lin8Case {
    rows: u32,
    n: u32,
    k: u32,
    codes: Vec<u8>,
    scales: Vec<u8>,
    w: Vec<u8>,
    ws: Vec<u8>,
}

fn lin8_case(g: &mut Rng, rows: u32, n: u32, k: u32, wide: bool) -> Lin8Case {
    let (codes, scales) = activations(g, us(rows), us(k), wide);
    let (w, ws) = fp8_weight(g, us(n), us(k), wide);
    Lin8Case {
        rows,
        n,
        k,
        codes,
        scales,
        w,
        ws,
    }
}

/// What a product's outputs are compared in: the reference's bytes, and each
/// form's.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Run {
    Reference,
    Tc(TcForm, TokenTile),
}

const SENTINEL: u16 = 0xdead;

fn run_lin8(gpu: &Gpu, k: &Kernels, c: &Lin8Case, run: Run) -> Vec<u16> {
    let s = gpu.stream();
    let codes = s.clone_htod(&c.codes).unwrap();
    let scales = s.clone_htod(&c.scales).unwrap();
    let w = s.clone_htod(&c.w).unwrap();
    let ws = s.clone_htod(&c.ws).unwrap();
    let out = s.clone_htod(&vec![SENTINEL; us(c.rows) * us(c.n)]).unwrap();
    let p = Lin8Params {
        codes: dptr(&codes, s),
        act_scales: dptr(&scales, s),
        w_codes: dptr(&w, s),
        w_scales: dptr(&ws, s),
        out: dptr(&out, s),
        rows: c.rows,
        n: c.n,
        k: c.k,
        reserved: 0,
    };
    // SAFETY: every array is sized for the counts passed with it.
    unsafe {
        match run {
            Run::Reference => k.reference.lin8(gpu, &p),
            Run::Tc(form, tile) => k.tc.lin8_tile(gpu, form, tile, &p),
        }
    }
    .unwrap();
    gpu.synchronize().unwrap();
    s.clone_dtoh(&out).unwrap()
}

struct Lin4Case {
    n: u32,
    k: u32,
    groups: u32,
    row_align: u32,
    layout_rows: u32,
    group_end: Vec<u32>,
    codes: Vec<u8>,
    scales: Vec<u8>,
    /// Shared by the cases of one shape.
    w: Rc<Vec<u8>>,
    ws: Rc<Vec<u8>>,
}

/// FP4 weights and their scales, shared by the cases of one shape.
type Weights = (Rc<Vec<u8>>, Rc<Vec<u8>>);

/// A layout's case over `weights`, or weights of its own drawn like its codes.
fn lin4_case(
    g: &mut Rng,
    ends: (Vec<u32>, u32),
    n: u32,
    k: u32,
    row_align: u32,
    wide: bool,
    weights: Option<&Weights>,
) -> Lin4Case {
    let (group_end, used) = ends;
    let groups = u(group_end.len());
    let layout_rows = used + g.below(9);
    let (codes, scales) = activations(g, us(layout_rows), us(k), wide);
    let (w, ws) = match weights {
        Some((w, ws)) => (Rc::clone(w), Rc::clone(ws)),
        None => {
            let (w, ws) = fp4_weights(g, us(groups), us(n), us(k), wide);
            (Rc::new(w), Rc::new(ws))
        }
    };
    Lin4Case {
        n,
        k,
        groups,
        row_align,
        layout_rows,
        group_end,
        codes,
        scales,
        w,
        ws,
    }
}

fn run_lin4(gpu: &Gpu, k: &Kernels, c: &Lin4Case, run: Run) -> Vec<u16> {
    let s = gpu.stream();
    let codes = s.clone_htod(&c.codes).unwrap();
    let scales = s.clone_htod(&c.scales).unwrap();
    let w = s.clone_htod(c.w.as_slice()).unwrap();
    let ws = s.clone_htod(c.ws.as_slice()).unwrap();
    let ends = s.clone_htod(&c.group_end).unwrap();
    let out = s
        .clone_htod(&vec![SENTINEL; us(c.layout_rows) * us(c.n)])
        .unwrap();
    let p = Lin4Params {
        codes: dptr(&codes, s),
        act_scales: dptr(&scales, s),
        w_packed: dptr(&w, s),
        w_scales: dptr(&ws, s),
        group_end: dptr(&ends, s),
        out: dptr(&out, s),
        layout_rows: c.layout_rows,
        n: c.n,
        k: c.k,
        groups: c.groups,
        row_align: c.row_align,
        reserved: 0,
    };
    // SAFETY: every array is sized for the counts passed with it.
    unsafe {
        match run {
            Run::Reference => k.reference.lin4(gpu, &p),
            Run::Tc(form, tile) => k.tc.lin4_tile(gpu, form, tile, &p),
        }
    }
    .unwrap();
    gpu.synchronize().unwrap();
    s.clone_dtoh(&out).unwrap()
}

// ---- comparisons ----

fn bf16(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// Distance in BF16 units in the last place (`u32::MAX` across a NaN).
fn ulps(a: u16, b: u16) -> u32 {
    let ord = |x: u16| -> i32 {
        let v = i32::from(x & 0x7fff);
        if x & 0x8000 != 0 { -v } else { v }
    };
    if bf16(a).is_nan() || bf16(b).is_nan() {
        return if bf16(a).is_nan() && bf16(b).is_nan() {
            0
        } else {
            u32::MAX
        };
    }
    ord(a).abs_diff(ord(b))
}

fn same_bytes(want: &[u16], got: &[u16], what: &str) {
    if let Some(at) = want.iter().zip(got).position(|(a, b)| a != b) {
        let differing = want.iter().zip(got).filter(|(a, b)| a != b).count();
        panic!(
            "{what}: {differing} of {} outputs differ, the first at {at} ({:#06x} for {:#06x})",
            want.len(),
            got[at],
            want[at]
        );
    }
}

/// The tolerance form's outputs against the reference's: a histogram of
/// BF16 ulps, the largest relative error, and the bound.
#[derive(Default)]
struct Tolerance {
    outputs: u64,
    histogram: [u64; 7],
    max_rel: f64,
    max_rms: f64,
    out_of_bound: u64,
}

const BUCKETS: [&str; 7] = ["0", "1", "2", "3-4", "5-8", "9-16", ">16"];

impl Tolerance {
    fn add(&mut self, want: &[u16], got: &[u16], width: usize) {
        for (w_row, g_row) in want.chunks_exact(width).zip(got.chunks_exact(width)) {
            if w_row.iter().all(|x| *x == SENTINEL) {
                continue; // a padding row: neither form writes it
            }
            let rms = (w_row
                .iter()
                .map(|x| f64::from(bf16(*x)).powi(2))
                .filter(|x| x.is_finite())
                .sum::<f64>()
                / w_row.len() as f64)
                .sqrt();
            for (&w, &g) in w_row.iter().zip(g_row) {
                self.outputs += 1;
                let d = ulps(w, g);
                let bucket = match d {
                    0 => 0,
                    1 => 1,
                    2 => 2,
                    3..=4 => 3,
                    5..=8 => 4,
                    9..=16 => 5,
                    _ => 6,
                };
                self.histogram[bucket] += 1;
                let (wv, gv) = (f64::from(bf16(w)), f64::from(bf16(g)));
                let err = (wv - gv).abs();
                if wv != 0.0 && wv.is_finite() {
                    self.max_rel = self.max_rel.max(err / wv.abs());
                }
                if rms > 0.0 && err.is_finite() {
                    self.max_rms = self.max_rms.max(err / rms);
                }
                let within = d <= TOLERANCE_ULPS || (err.is_finite() && err <= TOLERANCE_RMS * rms);
                if !within {
                    self.out_of_bound += 1;
                }
            }
        }
    }

    fn report(&self, what: &str) -> Value {
        let hist: serde_json::Map<String, Value> = BUCKETS
            .iter()
            .zip(self.histogram)
            .map(|(b, n)| ((*b).to_owned(), json!(n)))
            .collect();
        json!({
            "tolerance": what, "outputs": self.outputs, "ulp_histogram": hist,
            "max_relative_error": self.max_rel, "max_error_over_row_rms": self.max_rms,
            "out_of_bound": self.out_of_bound,
        })
    }
}

// ---- the suite ----

fn lin8_shapes(g: &mut Rng) -> Vec<(String, Lin8Case)> {
    let mut cases = Vec::new();
    for i in 0..64 {
        let rows = 1 + g.below(if i < 48 { 40 } else { 300 });
        let n = 1 + g.below(700);
        let k = 32 * (1 + g.below(24));
        let wide = i % 2 == 0;
        cases.push((
            format!("small {i}: {rows} x {n} x {k}"),
            lin8_case(g, rows, n, k, wide),
        ));
    }
    for &(n, k) in &DENSE {
        for rows in [1, 8, 9, 33] {
            cases.push((
                format!("released {n} x {k}, {rows} rows"),
                lin8_case(g, rows, n, k, false),
            ));
        }
        cases.push((
            format!("released {n} x {k}, 64 rows, arbitrary codes"),
            lin8_case(g, 64, n, k, true),
        ));
    }
    cases.push((
        "released 5120 x 8192, 2048 rows".to_owned(),
        lin8_case(g, 2048, 5120, 8192, false),
    ));
    cases
}

/// Small random layouts: every grid remainder, empty groups, several row
/// alignments.
fn lin4_small(g: &mut Rng) -> Vec<(String, Lin4Case)> {
    let mut cases = Vec::new();
    for i in 0..48 {
        let groups = 1 + g.below(24);
        let n = 1 + g.below(600);
        let k = 32 * (1 + g.below(16));
        let row_align = [1, 3, 8, 16][us(g.below(4))];
        let mut end = Vec::new();
        let mut at = 0u32;
        for _ in 0..groups {
            at = at.next_multiple_of(row_align);
            at += if g.below(4) == 0 { 0 } else { g.below(70) };
            end.push(at);
        }
        if at == 0 {
            continue;
        }
        cases.push((
            format!("small {i}: {groups} groups of {n} x {k}, align {row_align}"),
            lin4_case(g, (end, at), n, k, row_align, i % 2 == 0, None),
        ));
    }
    cases
}

/// The released experts' shapes, all 384 experts' weights (eight drawn ones
/// repeated), routed from 1 to 2,048 tokens.
fn lin4_released(g: &mut Rng) -> Vec<(String, Lin4Case)> {
    let mut cases = Vec::new();
    for &(n, k) in &EXPERTS {
        let (w, ws) = fp4_weights_tiled(g, us(GROUPS), 8, us(n), us(k), false);
        let weights = (Rc::new(w), Rc::new(ws));
        for (tokens, row_align) in [(1, 1), (7, 1), (64, 8), (512, 1), (2048, 8)] {
            let ends = layout(g, tokens, GROUPS, ACTIVE, row_align);
            cases.push((
                format!("released {n} x {k}, {tokens} tokens, align {row_align}"),
                lin4_case(g, ends, n, k, row_align, false, Some(&weights)),
            ));
        }
        let ends = layout(g, 64, GROUPS, ACTIVE, 1);
        cases.push((
            format!("released {n} x {k}, 64 tokens, arbitrary codes"),
            lin4_case(g, ends, n, k, 1, true, Some(&weights)),
        ));
    }
    cases
}

fn products(r: &mut Report, gpu: &Gpu, k: &Kernels, arch: ImageArch) {
    let mut g = Rng(0x7c_9e00);
    let lin8 = lin8_shapes(&mut g);
    let mut lin4 = lin4_small(&mut g);
    lin4.extend(lin4_released(&mut g));
    let mut tol8 = Tolerance::default();
    let mut tol4 = Tolerance::default();
    r.check(format!("{arch:?}/lin8/banded"), |_| {
        for (name, c) in &lin8 {
            let want = run_lin8(gpu, k, c, Run::Reference);
            for tile in TokenTile::ALL {
                let got = run_lin8(gpu, k, c, Run::Tc(TcForm::Banded, tile));
                same_bytes(&want, &got, &format!("{name}, {tile:?}"));
            }
        }
    });
    r.check(format!("{arch:?}/lin4/banded"), |_| {
        for (name, c) in &lin4 {
            let want = run_lin4(gpu, k, c, Run::Reference);
            for tile in TokenTile::ALL {
                let got = run_lin4(gpu, k, c, Run::Tc(TcForm::Banded, tile));
                same_bytes(&want, &got, &format!("{name}, {tile:?}"));
            }
        }
    });
    r.check(format!("{arch:?}/lin8/tolerance"), |m| {
        for (name, c) in &lin8 {
            let want = run_lin8(gpu, k, c, Run::Reference);
            let s8 = run_lin8(gpu, k, c, Run::Tc(TcForm::Tolerance, TokenTile::S8));
            for tile in [TokenTile::T8, TokenTile::T32] {
                let got = run_lin8(gpu, k, c, Run::Tc(TcForm::Tolerance, tile));
                same_bytes(&s8, &got, &format!("{name}: {tile:?} against S8"));
            }
            tol8.add(&want, &s8, us(c.n));
        }
        m.push(tol8.report(&format!("{arch:?}/lin8")));
        assert_eq!(tol8.out_of_bound, 0, "outputs out of the bound");
    });
    r.check(format!("{arch:?}/lin4/tolerance"), |m| {
        for (name, c) in &lin4 {
            let want = run_lin4(gpu, k, c, Run::Reference);
            let s8 = run_lin4(gpu, k, c, Run::Tc(TcForm::Tolerance, TokenTile::S8));
            for tile in [TokenTile::T8, TokenTile::T32] {
                let got = run_lin4(gpu, k, c, Run::Tc(TcForm::Tolerance, tile));
                same_bytes(&s8, &got, &format!("{name}: {tile:?} against S8"));
            }
            tol4.add(&want, &s8, us(c.n));
        }
        m.push(tol4.report(&format!("{arch:?}/lin4")));
        assert_eq!(tol4.out_of_bound, 0, "outputs out of the bound");
    });
}

/// A row alone (a one-row launch over the same arrays at its offset, a
/// split block) against the same row in a batch of 300 (32-row tiles); an
/// expert's rows in a layout of only them against the full layout.
fn invariance(r: &mut Report, gpu: &Gpu, k: &Kernels, arch: ImageArch) {
    for form in [TcForm::Banded, TcForm::Tolerance] {
        r.check(format!("{arch:?}/invariance/lin8/{form:?}"), |_| {
            let mut g = Rng(0x1a2b);
            let c = lin8_case(&mut g, 300, 2304, 5120, false);
            let batch = run_lin8(gpu, k, &c, Run::Tc(form, TokenTile::T32));
            let s = gpu.stream();
            let codes = s.clone_htod(&c.codes).unwrap();
            let scales = s.clone_htod(&c.scales).unwrap();
            let w = s.clone_htod(&c.w).unwrap();
            let ws = s.clone_htod(&c.ws).unwrap();
            let kb = us(c.k / 32);
            for row in [0usize, 1, 31, 32, 150, 299] {
                let out = s.alloc_zeros::<u16>(us(c.n)).unwrap();
                let p = Lin8Params {
                    codes: dptr_at(&codes, s, row * us(c.k)),
                    act_scales: dptr_at(&scales, s, row * kb),
                    w_codes: dptr(&w, s),
                    w_scales: dptr(&ws, s),
                    out: dptr(&out, s),
                    rows: 1,
                    n: c.n,
                    k: c.k,
                    reserved: 0,
                };
                // SAFETY: one row of the case's arrays, at its offset.
                unsafe { k.tc.lin8_tile(gpu, form, TokenTile::S8, &p) }.unwrap();
                gpu.synchronize().unwrap();
                let alone = s.clone_dtoh(&out).unwrap();
                same_bytes(
                    &batch[row * us(c.n)..(row + 1) * us(c.n)],
                    &alone,
                    &format!("row {row} alone"),
                );
            }
        });
        r.check(format!("{arch:?}/invariance/lin4/{form:?}"), |_| {
            let mut g = Rng(0x3c4d);
            let ends = layout(&mut g, 256, GROUPS, ACTIVE, 1);
            let (w, ws) = fp4_weights_tiled(&mut g, us(GROUPS), 8, 5120, 2304, false);
            let full = lin4_case(
                &mut g,
                ends,
                5120,
                2304,
                1,
                false,
                Some(&(Rc::new(w), Rc::new(ws))),
            );
            let all = run_lin4(gpu, k, &full, Run::Tc(form, TokenTile::T32));
            // Several experts, each alone in a layout of its own rows.
            for e in [0usize, 1, 17, 200, 383] {
                let start = if e == 0 { 0 } else { full.group_end[e - 1] };
                let end = full.group_end[e];
                if start == end {
                    continue;
                }
                let rows = us(end - start);
                let (k_, kb) = (us(full.k), us(full.k / 32));
                let only = Lin4Case {
                    n: full.n,
                    k: full.k,
                    groups: full.groups,
                    row_align: 1,
                    layout_rows: end - start,
                    group_end: (0..full.groups)
                        .map(|j| if us(j) < e { 0 } else { end - start })
                        .collect(),
                    codes: full.codes[us(start) * k_..us(end) * k_].to_vec(),
                    scales: full.scales[us(start) * kb..us(end) * kb].to_vec(),
                    w: Rc::clone(&full.w),
                    ws: Rc::clone(&full.ws),
                };
                let alone = run_lin4(gpu, k, &only, Run::Tc(form, TokenTile::S8));
                let n = us(full.n);
                same_bytes(
                    &all[us(start) * n..us(end) * n],
                    &alone[..rows * n],
                    &format!("expert {e}'s {rows} rows alone"),
                );
            }
        });
    }
}

#[test]
fn device_suite() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let dots: u64 = std::env::var("EIDOLA_DS_PROBE_DOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 33);
    let mut r = Report {
        checks: Vec::new(),
        measured: Vec::new(),
        failed: 0,
    };
    for &arch in &su.archs {
        let k = Kernels {
            reference: DsGemmRef::from_module(su.module(DsGemmRef::IMAGE, arch)).unwrap(),
            tc: DsGemmTc::from_module(su.module(DsGemmTc::IMAGE, arch)).unwrap(),
        };
        for c in probe_cases() {
            r.check(format!("{arch:?}/probe/{}", c.name), |m| {
                let v = run_probe(gpu, &k.tc, &c, dots);
                let (differing, moved) = (v["differing"].as_u64(), v["moved"].as_u64());
                m.push(v);
                assert_eq!(moved, Some(0), "an element's bits moved with its place");
                if c.banded {
                    assert_eq!(differing, Some(0), "a banded sum came back inexact");
                }
            });
        }
        products(&mut r, gpu, &k, arch);
        invariance(&mut r, gpu, &k, arch);
    }
    r.finish(gpu);
}

/// The suite's host side: every case is built, every layout is what the
/// reference takes, and the probe's cases name every band pair.
#[test]
fn cases_are_built_without_a_device() {
    let mut g = Rng(0x7c_9e00);
    let lin4 = lin4_small(&mut g);
    for (name, c) in &lin4 {
        assert_eq!(c.group_end.len(), us(c.groups), "{name}");
        assert!(c.group_end.windows(2).all(|w| w[0] <= w[1]), "{name}");
        assert!(*c.group_end.last().unwrap() <= c.layout_rows, "{name}");
        assert_eq!(c.codes.len(), us(c.layout_rows) * us(c.k), "{name}");
        assert_eq!(c.w.len(), us(c.groups) * us(c.n) * us(c.k) / 2, "{name}");
    }
    let banded = probe_cases().iter().filter(|c| c.banded).count();
    assert_eq!(
        banded,
        4 * (W8_BANDS.len() * X8_BANDS.len() + X4_BANDS.len())
    );
    // The tolerance metric: equal outputs are 0 ulps, a NaN against a number never within.
    assert_eq!(ulps(0x3f80, 0x3f80), 0);
    assert_eq!(ulps(0x3f80, 0x3f81), 1);
    assert_eq!(ulps(0x0001, 0x8001), 2);
    assert_eq!(ulps(0x7fc0, 0x3f80), u32::MAX);
}
