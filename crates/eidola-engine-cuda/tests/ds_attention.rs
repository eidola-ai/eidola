//! DeepSeek-V4.1 sparse attention and window slot lists (`ds_attention.cu`)
//! against the model crate's reference, bit for bit: `eidola_ds_attn` against
//! `attention::sparse_attend` over packed pool entries, and
//! `eidola_ds_window_slots` against the canonical list of `forward.rs` and
//! `drafter::ring_order`.
//!
//! Every case is a job (`ds_support`): on a device it runs on the exact cubin
//! and the family cubin; without one the same jobs run through the kernel
//! bodies compiled for the host, so the cases, the parameters and the expected
//! bytes are all exercised by a plain `cargo test`. On a device the tests also
//! time the kernels at the released shapes and write a JSON summary.
//!
//! ```sh
//! EIDOLA_ENGINE_KERNELS_DIR=<build output> EIDOLA_DS_BENCH_DIR=<dir> \
//!   cargo test --release -p eidola-engine-cuda --test ds_attention --test ds_index \
//!   -- --nocapture --test-threads 1
//! ```

mod common;
mod ds_support;

use ds_support::cases::{
    AttnCase, Cancel, EMPTY, FP4_E4M3, FP4_E8M0, FP8, Rng, WindowCase, attn_case,
};
use ds_support::{Arena, Call, Job, Ops, Timing, pool, run_on_device, run_on_host};
use eidola_engine_cuda::ds::abi::{
    ATTN_FIXED, ATTN_PACKED, AttnParams, STATUS_BAD_INDEX, WindowSlotsParams,
};
use eidola_engine_cuda::ds::attention::{MAX_HEADS_PER_BLOCK, heads_per_block, window_slots};

fn u32_of(x: usize) -> u32 {
    u32::try_from(x).unwrap()
}

/// The group sizes a launch of these heads is run at: the library's own, one
/// head a block, and the most.
fn groups(rows: usize, heads: usize) -> Vec<u32> {
    let mut g = vec![
        heads_per_block(u32_of(rows), u32_of(heads)),
        1,
        MAX_HEADS_PER_BLOCK,
    ];
    g.sort_unstable();
    g.dedup();
    g
}

/// One launch over `cases` as its rows (one shape), at `group` heads per
/// block: the planes as pools holding every row's entries, the slot words as
/// pool entries, rows of slots wider than the lists.
fn attn_job(label: &str, cases: &[AttnCase], per_block: [usize; 2], group: u32) -> Job {
    let c0 = &cases[0];
    let (heads, dim, a_slots) = (c0.heads, c0.dim, c0.a_slots);
    let b_slots = c0.slots.len() - a_slots;
    let stride = a_slots + b_slots + 3;
    let mut arena = Arena::default();
    let mut entries = [Vec::new(), Vec::new()];
    let (mut q, mut slots, mut want) = (Vec::<u16>::new(), Vec::new(), Vec::new());
    let sink = c0.sink.clone();
    for c in cases {
        assert_eq!(
            (
                c.heads,
                c.dim,
                c.a_slots,
                c.slots.len(),
                c.packed,
                c.formats
            ),
            (
                heads,
                dim,
                a_slots,
                a_slots + b_slots,
                c0.packed,
                c0.formats
            ),
            "{label}: rows of one launch share a shape"
        );
        let first = [u32_of(entries[0].len()), u32_of(entries[1].len())];
        for (s, w) in c.slots.iter().enumerate() {
            let plane = usize::from(s >= a_slots);
            slots.push(if *w == EMPTY { EMPTY } else { first[plane] + w });
        }
        // Words past the list are never read.
        slots.extend([0x1234_5678u32; 3]);
        for (pooled, own) in entries.iter_mut().zip(&c.entries) {
            pooled.extend(own.iter().cloned());
        }
        q.extend(&c.q);
        want.extend(c.expected());
    }
    // One sink vector serves every row of a launch.
    assert!(
        cases.iter().all(|c| c
            .sink
            .iter()
            .map(|s| s.to_bits())
            .eq(sink.iter().map(|s| s.to_bits()))),
        "{label}: rows of one launch share the sinks"
    );
    let a = pool(&mut arena, c0.formats[0], dim, per_block[0], &entries[0]);
    let b = pool(&mut arena, c0.formats[1], dim, per_block[1], &entries[1]);
    let q_at = arena.put(&q);
    let slots_at = arena.put(&slots);
    let sink_at = arena.put(&sink);
    let out_at = arena.put(&vec![0xdeadu16; q.len()]);
    let mut job = Job::new(format!("{label}, {group} heads a block"), arena);
    job.calls.push(Call::Attn(
        AttnParams {
            q: Arena::addr(q_at),
            slots: Arena::addr(slots_at),
            sink: Arena::addr(sink_at),
            out: Arena::addr(out_at),
            a,
            b,
            rows: u32_of(cases.len()),
            heads: u32_of(heads),
            head_dim: u32_of(dim),
            a_slots: u32_of(a_slots),
            b_slots: u32_of(b_slots),
            slot_stride: u32_of(stride),
            scale: c0.scale,
            layout: if c0.packed { ATTN_PACKED } else { ATTN_FIXED },
        },
        group,
    ));
    job.expect("attention output", out_at, &want);
    job
}

/// A launch's jobs at every group size.
fn attn_jobs(jobs: &mut Vec<Job>, label: &str, cases: &[AttnCase], per_block: [usize; 2]) {
    for group in groups(cases.len(), cases[0].heads) {
        jobs.push(attn_job(label, cases, per_block, group));
    }
}

/// Rows of one shape sharing their sinks.
#[allow(clippy::too_many_arguments)]
fn rows(
    rng: &mut Rng,
    n: usize,
    heads: usize,
    dim: usize,
    a: usize,
    b: usize,
    formats: [u32; 2],
    empty_in: usize,
) -> Vec<AttnCase> {
    let mut cases: Vec<AttnCase> = (0..n)
        .map(|_| attn_case(rng, heads, dim, a, b, formats, 0, empty_in))
        .collect();
    let sink = cases[0].sink.clone();
    for c in &mut cases {
        c.sink = sink.clone();
    }
    cases
}

const BACKBONE: [u32; 2] = [FP8, FP4_E4M3];

/// The edge cases, each a launch of a few rows.
fn edge_jobs() -> Vec<Job> {
    let mut rng = Rng(0xa77e_0001);
    let rng = &mut rng;
    let mut jobs = Vec::new();

    // Empty slots anywhere, lists inside one block, ending on a block edge and
    // crossing several; head counts that fill and do not fill a block's group.
    for &(heads, dim) in &[(1, 32), (4, 32), (9, 32), (17, 64)] {
        for &(a, b) in &[(3, 0), (48, 24), (64, 5), (70, 70), (128, 70)] {
            for empty_in in [0, 3] {
                let cases = rows(rng, 3, heads, dim, a, b, BACKBONE, empty_in);
                let label = format!("{heads}x{dim} window {a} + {b}, empties 1/{empty_in}");
                attn_jobs(&mut jobs, &label, &cases, [5, 3]);
            }
        }
    }
    // Every format in either plane.
    for formats in [[FP8, FP8], [FP4_E8M0, FP4_E4M3], [FP4_E4M3, FP4_E8M0]] {
        let cases = rows(rng, 2, 2, 64, 40, 40, formats, 4);
        attn_jobs(&mut jobs, &format!("formats {formats:?}"), &cases, [7, 2]);
    }
    // A window not yet full is right-aligned: its first block may be empty, and
    // it crosses the block boundary at 64 slots.
    let mut cases = rows(rng, 7, 2, 32, 128, 24, BACKBONE, 0);
    for (c, listed) in cases.iter_mut().zip([1, 2, 63, 64, 65, 127, 128]) {
        c.slots[..128 - listed].fill(EMPTY);
    }
    attn_jobs(&mut jobs, "a window filling up", &cases, [16, 4]);
    // Rows with no entry at all give zeros, beside rows with one entry and
    // with only global entries.
    let mut cases = rows(rng, 4, 3, 32, 48, 24, BACKBONE, 0);
    cases[0].slots.fill(EMPTY);
    cases[1].slots[..71].fill(EMPTY);
    cases[2].slots[..48].fill(EMPTY);
    assert!(cases[0].expected().iter().all(|b| *b == 0));
    assert!(cases[1].expected().iter().any(|b| *b != 0));
    attn_jobs(&mut jobs, "rows without entries", &cases, [4, 4]);
    let cases = rows(rng, 2, 2, 32, 0, 5, BACKBONE, 0);
    attn_jobs(&mut jobs, "no window slots", &cases, [4, 4]);
    // A selection shorter than the top-k: trailing global slots empty.
    let mut cases = rows(rng, 2, 2, 32, 48, 80, BACKBONE, 0);
    cases[0].slots[60..].fill(EMPTY);
    cases[1].slots[100..].fill(EMPTY);
    attn_jobs(&mut jobs, "a short selection", &cases, [4, 4]);
    // Sinks: none, dominant, negligible, NaN.
    let mut cases = rows(rng, 2, 4, 32, 48, 24, BACKBONE, 5);
    for c in &mut cases {
        c.sink = vec![f32::NEG_INFINITY, 40.0, -40.0, f32::NAN];
    }
    attn_jobs(&mut jobs, "sinks", &cases, [4, 4]);
    // Magnitudes: weights that underflow against a later block's maximum.
    for exp in [-6, 3, 5] {
        let mut cases: Vec<AttnCase> = (0..2)
            .map(|_| attn_case(rng, 2, 64, 70, 70, BACKBONE, exp, 5))
            .collect();
        cases[1].sink = cases[0].sink.clone();
        attn_jobs(&mut jobs, &format!("magnitudes 2^{exp}"), &cases, [4, 4]);
    }
    // An infinite scale (NaN and infinite values), and a block whose only
    // score is -inf: skipped, alone and before a later entry.
    let mut cases = rows(rng, 3, 1, 32, 130, 0, [FP8, FP8], 0);
    let at = cases[0].slots[5] as usize;
    cases[0].entries[0][at][32] = 0xff;
    for (c, with_later) in cases[1..].iter_mut().zip([false, true]) {
        c.q.fill(0xbf80);
        let keep = c.slots[70] as usize;
        c.entries[0][keep] = [vec![0x38; 32], vec![0xff]].concat();
        for (s, w) in c.slots.iter_mut().enumerate() {
            if s != 70 && !(with_later && s == 129) {
                *w = EMPTY;
            }
        }
    }
    assert!(cases[0].expected().contains(&0x7fc0));
    assert!(cases[1].expected().iter().all(|b| *b == 0));
    attn_jobs(&mut jobs, "infinite scales and -inf blocks", &cases, [9, 9]);
    // The drafter's list: the window in ring order as far as the sequence has
    // come, then the draft slots' KV (FP8, from scratch), packed behind it.
    let mut cases = rows(rng, 8, 2, 32, 128, 5, [FP8, FP8], 0);
    for (c, listed) in cases.iter_mut().zip([1, 3, 59, 60, 64, 100, 127, 128]) {
        c.packed = true;
        c.slots[listed..128].fill(EMPTY);
    }
    // An empty slot inside the listed window keeps its place.
    cases[3].slots[10] = EMPTY;
    attn_jobs(&mut jobs, "drafter lists", &cases, [16, 5]);
    jobs
}

/// Seeded sweeps: random shapes, lists and batches, half of them with scores
/// that are the residue of cancelled terms, where the summation order shows.
fn sweep_jobs(seed: u64, n: usize) -> Vec<Job> {
    let mut rng = Rng(seed);
    let rng = &mut rng;
    let mut jobs = Vec::new();
    for i in 0..n {
        let heads = [1, 2, 3, 5, 8, 9, 16, 17][rng.below(8)];
        let dim = [32, 64, 96, 128, 256][rng.below(5)];
        let a = [1, 17, 48, 64, 65, 128][rng.below(6)];
        let b = [0, 5, 24, 64, 150][rng.below(5)];
        let formats = [BACKBONE, [FP8, FP8]][rng.below(2)];
        let n_rows = 1 + rng.below(4);
        let empty_in = [0, 2, 6][rng.below(3)];
        let mut cases = rows(rng, n_rows, heads, dim, a, b, formats, empty_in);
        if dim >= 64 && i % 2 == 0 {
            for c in &mut cases {
                let mode = [Cancel::Ends, Cancel::Lanes][rng.below(2)];
                c.cancel(rng, mode);
            }
        }
        if a > 8 && rng.below(4) == 0 {
            for c in &mut cases {
                c.packed = true;
                let listed = 1 + rng.below(a);
                c.slots[listed..a].fill(EMPTY);
            }
        }
        let per_block = [1 + rng.below(20), 1 + rng.below(20)];
        let label = format!("sweep {seed:x}/{i}: {heads}x{dim}, {a} + {b}");
        let group = groups(n_rows, heads)[rng.below(groups(n_rows, heads).len())];
        jobs.push(attn_job(&label, &cases, per_block, group));
    }
    jobs
}

/// The released shapes: 64 heads of 512 over a window of 128 and 512 global
/// entries, and a drafter block's 128 + 5.
fn released_jobs() -> Vec<Job> {
    let mut rng = Rng(0xa77e_0003);
    let rng = &mut rng;
    let mut jobs = Vec::new();
    let mut cases = rows(rng, 3, 64, 512, 128, 512, BACKBONE, 0);
    cases[1].slots[..100].fill(EMPTY);
    cases[1].slots[400..].fill(EMPTY);
    cases[2].cancel(rng, Cancel::Lanes);
    for group in [1, 8] {
        jobs.push(attn_job("released backbone", &cases, [16, 8], group));
    }
    let mut cases = rows(rng, 2, 64, 512, 128, 5, [FP8, FP8], 0);
    for (c, listed) in cases.iter_mut().zip([128, 61]) {
        c.packed = true;
        c.slots[listed..128].fill(EMPTY);
        c.cancel(rng, Cancel::Ends);
    }
    jobs.push(attn_job("released drafter", &cases, [16, 5], 4));
    jobs
}

/// One launch over `cases` as its rows (one order, window and block size),
/// each row with its own table.
fn window_job(label: &str, cases: &[WindowCase]) -> Job {
    let c0 = &cases[0];
    let stride = c0.window + 4;
    let table_stride = cases.iter().map(|c| c.table.len()).max().unwrap() + 1;
    let mut arena = Arena::default();
    let (mut tables, mut want, mut status) = (Vec::new(), Vec::new(), 0);
    for c in cases {
        assert_eq!(
            (c.order, c.window, c.per_block),
            (c0.order, c0.window, c0.per_block)
        );
        let mut table = c.table.clone();
        table.resize(table_stride, -1);
        tables.extend(table);
        let (slots, bad) = c.expected();
        // The library's host form agrees with the reference's list.
        let (host, unmapped) =
            window_slots(c.order, c.window, c.pos, c.first, &c.table, c.per_block);
        assert_eq!((&host, unmapped), (&slots, bad != 0), "{label}");
        want.push(slots);
        status |= bad;
    }
    // Rows read their tables in another order than the rows'.
    let n = cases.len();
    let table_row: Vec<u32> = (0..n).map(|r| u32_of((r * 7 + 3) % n)).collect();
    let mut shuffled = vec![0i32; tables.len()];
    for (r, to) in table_row.iter().enumerate() {
        shuffled[*to as usize * table_stride..][..table_stride]
            .copy_from_slice(&tables[r * table_stride..][..table_stride]);
    }
    assert_eq!(STATUS_BAD_INDEX, ds_support::cases::BAD_INDEX);
    let positions_at = arena.put(&cases.iter().map(|c| u32_of(c.pos)).collect::<Vec<_>>());
    let first_at = arena.put(&cases.iter().map(|c| u32_of(c.first)).collect::<Vec<_>>());
    let tables_at = arena.put(&shuffled);
    let table_row_at = arena.put(&table_row);
    let slots_at = arena.put(&vec![0x5a5a_5a5au32; n * stride]);
    let status_at = arena.put(&[0u32]);
    let mut job = Job::new(label, arena);
    job.calls.push(Call::Window(WindowSlotsParams {
        positions: Arena::addr(positions_at),
        first: Arena::addr(first_at),
        tables: Arena::addr(tables_at),
        table_row: Arena::addr(table_row_at),
        slots: Arena::addr(slots_at),
        status: Arena::addr(status_at),
        rows: u32_of(n),
        window: u32_of(c0.window),
        block_positions: u32_of(c0.per_block),
        table_stride: u32_of(table_stride),
        slot_stride: u32_of(stride),
        order: c0.order,
    }));
    for (r, slots) in want.iter().enumerate() {
        job.expect(
            format!("row {r}'s window"),
            slots_at + r * stride * 4,
            slots,
        );
    }
    job.expect("status", status_at, &[status]);
    job
}

fn window_jobs() -> Vec<Job> {
    let mut rng = Rng(0x51075);
    let mut jobs = Vec::new();
    for &(window, per_block) in &[(4, 4), (48, 16), (128, 16), (128, 256), (129, 8), (300, 32)] {
        for order in 0..2 {
            // Around the window filling, deep into a sequence, and cut at a
            // first position.
            let mut cases: Vec<WindowCase> = [0, 1, window - 2, window - 1, window, window + 1]
                .into_iter()
                .chain((0..40).map(|_| rng.below(40 * window)))
                .map(|pos| WindowCase::new(order, window, per_block, pos, 0))
                .collect();
            for c in cases.iter_mut().skip(10).step_by(3) {
                c.first = c.pos.saturating_sub(rng.below(window + 2));
            }
            cases.push(WindowCase::new(
                order,
                window,
                per_block,
                3 * window,
                3 * window,
            ));
            let label = format!("window {window}, order {order}, blocks of {per_block}");
            jobs.push(window_job(&label, &cases));
            // A table word that maps nothing: inside one row's window.
            let pos = 9 * window + 5;
            cases[2] = WindowCase::new(order, window, per_block, pos, 0);
            cases[2].table[pos / per_block] = -1;
            jobs.push(window_job(&format!("{label}, an unmapped block"), &cases));
        }
    }
    jobs
}

fn all_jobs() -> Vec<Job> {
    let mut jobs = edge_jobs();
    jobs.extend(sweep_jobs(0xa77e_0002, 60));
    jobs.extend(released_jobs());
    jobs.extend(window_jobs());
    jobs
}

/// The kernel bodies, compiled for the host, on every case the device runs.
#[test]
fn bodies_match_the_reference_on_the_host() {
    let jobs = all_jobs();
    assert!(jobs.len() > 150, "{}", jobs.len());
    if run_on_host(&jobs) {
        // A longer sweep than the device needs: it is cheap here.
        assert!(run_on_host(&sweep_jobs(0xa77e_0004, 400)));
    }
}

#[test]
fn kernels_match_the_reference_on_the_device() {
    let Some(su) = common::setup() else { return };
    let jobs = all_jobs();
    for &arch in &su.archs {
        let ops = Ops::load(&su.gpu, &su.dir, arch);
        run_on_device(&su.gpu, &ops, &jobs);
        println!("{arch:?}: {} attention and window jobs match", jobs.len());
    }
}

/// A launch at the released backbone shape over `n` rows, every slot listed:
/// the rows share one pool of entries.
fn bench_job(rng: &mut Rng, n: usize, b_slots: usize, group: u32) -> Job {
    let (heads, dim, a_slots) = (64, 512, 128);
    let formats = if b_slots == 5 { [FP8, FP8] } else { BACKBONE };
    let base = attn_case(rng, heads, dim, a_slots, b_slots, formats, 0, 0);
    let mut arena = Arena::default();
    let a = pool(&mut arena, formats[0], dim, 16, &base.entries[0]);
    let b = pool(&mut arena, formats[1], dim, 8, &base.entries[1]);
    let q: Vec<u16> = (0..n * heads * dim).map(|_| rng.bf16(-3, 1)).collect();
    let stride = a_slots + b_slots;
    let mut slots = Vec::with_capacity(n * stride);
    for r in 0..n {
        // Each row lists the entries from another start.
        slots.extend((0..a_slots).map(|s| u32_of((s + r) % a_slots)));
        slots.extend((0..b_slots).map(|s| u32_of((s + 3 * r) % b_slots)));
    }
    let q_at = arena.put(&q);
    let slots_at = arena.put(&slots);
    let sink_at = arena.put(&base.sink);
    let out_at = arena.put(&vec![0u16; q.len()]);
    let mut job = Job::new("bench", arena);
    job.calls.push(Call::Attn(
        AttnParams {
            q: Arena::addr(q_at),
            slots: Arena::addr(slots_at),
            sink: Arena::addr(sink_at),
            out: Arena::addr(out_at),
            a,
            b,
            rows: u32_of(n),
            heads: u32_of(heads),
            head_dim: u32_of(dim),
            a_slots: u32_of(a_slots),
            b_slots: u32_of(b_slots),
            slot_stride: u32_of(stride),
            scale: base.scale,
            layout: ATTN_FIXED,
        },
        group,
    ));
    job
}

/// The bench's launches build without a device, at one row.
#[test]
fn bench_launches_build_without_a_device() {
    let mut rng = Rng(0xbe);
    for b_slots in [512, 5] {
        let job = bench_job(&mut rng, 1, b_slots, 1);
        let Call::Attn(p, group) = job.calls[0] else {
            unreachable!()
        };
        eidola_engine_cuda::ds::attention::check_attn(&p.clone()).unwrap();
        assert_eq!(
            eidola_engine_cuda::ds::attention::attn_grid(&p, group).unwrap(),
            [64, 1, 1]
        );
    }
}

/// Microseconds per query at the released shapes, at 1, 64 and 384 queries,
/// for the library's group size and for the others.
#[test]
fn timings_on_the_device() {
    use eidola_engine_cuda::launch::dptr;
    let Some(su) = common::setup() else { return };
    let arch = su.archs[0];
    let ops = Ops::load(&su.gpu, &su.dir, arch);
    let s = su.gpu.stream();
    let mut rng = Rng(0xbe);
    let mut timings = Vec::new();
    for (shape, b_slots) in [("backbone", 512usize), ("drafter", 5)] {
        for n in [1usize, 64, 384] {
            let auto = heads_per_block(u32_of(n), 64);
            let job = bench_job(&mut rng, n, b_slots, auto);
            let buffer = s.clone_htod(&job.arena).unwrap();
            let base = dptr(&buffer, s);
            for group in [1, 2, 4, 8] {
                let Call::Attn(p, _) = job.calls[0] else {
                    unreachable!()
                };
                let call = Call::Attn(p, group).moved_to(base);
                let iters = if n == 1 { 200 } else { 20 };
                let us = ds_support::time_us(&su.gpu, iters, &|| {
                    // SAFETY: the call's arrays lie in `buffer`, alive past the timing.
                    unsafe { ops.launch(&su.gpu, &call) }
                });
                let picked = if group == auto { ", the library's" } else { "" };
                timings.push(Timing {
                    kernel: "eidola_ds_attn",
                    shape: format!(
                        "{shape}: 64x512, 128 + {b_slots} slots, {group} heads a block{picked}"
                    ),
                    rows: u32_of(n),
                    launch_us: us,
                });
            }
        }
    }
    // The window slots of a decode step's rows.
    for n in [1usize, 64, 384] {
        let cases: Vec<WindowCase> = (0..n)
            .map(|r| WindowCase::new(0, 128, 16, 5000 + 37 * r, 0))
            .collect();
        let job = window_job("bench", &cases);
        let buffer = s.clone_htod(&job.arena).unwrap();
        let call = job.calls[0].moved_to(dptr(&buffer, s));
        let us = ds_support::time_us(&su.gpu, 200, &|| {
            // SAFETY: as above.
            unsafe { ops.launch(&su.gpu, &call) }
        });
        timings.push(Timing {
            kernel: "eidola_ds_window_slots",
            shape: "window 128, position order".into(),
            rows: u32_of(n),
            launch_us: us,
        });
    }
    ds_support::write_bench("ds_attention_bench", arch, &timings);
}
