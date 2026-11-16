//! DeepSeek-V4.1 index scores and selections (`ds_index.cu`) against the model
//! crate's reference, bit for bit: `eidola_ds_index_score` against
//! `attention::index_score` over packed keys, `eidola_ds_select` against
//! `attention::top_k` and `attention::candidate_mask`, and a layer's whole
//! chain (window slots, scores, candidate blocks, both selections, attention
//! over the lists they build) against the reference's composition.
//!
//! Every case is a job (`deepseek_support`): on a device it runs on the exact cubin
//! and the family cubin; without one the same jobs run through the kernel
//! bodies compiled for the host. On a device the tests also time the kernels
//! at the released shapes and write a JSON summary. The command is
//! `tests/deepseek_attention.rs`'s.

mod common;
mod deepseek_support;

use deepseek_support::cases::{
    AttnCase, Cancel, FP4_E4M3, FP4_E8M0, FP8, Rng, ScoreCase, SelectCase, WindowCase, bf16_value,
    entry, mask_words as pack_mask, scores,
};
use deepseek_support::{Arena, Call, Job, Ops, Timing, pool, run_on_device, run_on_host};
use eidola_engine_cuda::deepseek::abi::{
    ATTN_FIXED, AttnParams, IndexScoreParams, SelectParams, WindowSlotsParams,
};
use eidola_engine_cuda::deepseek::index::{candidate_blocks, global_slots, mask_words, score_grid};

fn u32_of(x: usize) -> u32 {
    u32::try_from(x).unwrap()
}

/// One launch over `cases` as its rows (one head count and width), over a
/// grid `width` blocks wide (the library's when 0). Every row's keys sit in
/// one pool, its bank's blocks scattered among the other rows' and named by
/// its own table. With `unmapped`, that row's table loses that block.
fn score_job(
    label: &str,
    cases: &[ScoreCase],
    per_block: usize,
    width: u32,
    unmapped: Option<(usize, usize)>,
) -> Job {
    let c0 = &cases[0];
    let (heads, dim) = (c0.heads, c0.dim);
    let n = cases.len();
    let blocks: Vec<usize> = cases
        .iter()
        .map(|c| c.keys.len().div_ceil(per_block))
        .collect();
    let table_stride = blocks.iter().max().unwrap() + 1;
    // Pool block of (row, block): the rows' blocks interleaved, from the last row's.
    let mut tables = vec![-1i32; n * table_stride];
    let mut owners = Vec::new();
    for b in 0..table_stride {
        for r in (0..n).rev().filter(|r| b < blocks[*r]) {
            tables[r * table_stride + b] = i32::try_from(owners.len()).unwrap();
            owners.push((r, b));
        }
    }
    let filler = vec![0x77u8; dim / 2 + dim / 32];
    let mut pooled = Vec::new();
    for (r, b) in &owners {
        for o in 0..per_block {
            pooled.push(
                cases[*r]
                    .keys
                    .get(b * per_block + o)
                    .unwrap_or(&filler)
                    .clone(),
            );
        }
    }
    let stride = cases.iter().map(|c| c.keys.len()).max().unwrap() + 5;
    let (mut q_codes, mut q_scales, mut head_dots) = (Vec::new(), Vec::new(), Vec::<u16>::new());
    let mut want = Vec::new();
    let mut status = 0;
    for (r, c) in cases.iter().enumerate() {
        assert_eq!(
            (c.heads, c.dim, c.weight_scale.to_bits()),
            (heads, dim, c0.weight_scale.to_bits())
        );
        let (codes, scales) = c.q.split_at(heads * dim / 2);
        q_codes.extend_from_slice(codes);
        q_scales.extend_from_slice(scales);
        head_dots.extend(&c.head_dots);
        let mut row = c.expected();
        if let Some((_, block)) = unmapped.filter(|(row, _)| *row == r) {
            // An entry the table does not map scores +0 and raises the status.
            tables[r * table_stride + block] = -3;
            for s in row.iter_mut().skip(block * per_block).take(per_block) {
                *s = 0;
            }
            status = deepseek_support::cases::BAD_INDEX;
        }
        want.push(row);
    }
    // Rows read their tables in another order than the rows'.
    let table_row: Vec<u32> = (0..n).map(|r| u32_of((r * 5 + 2) % n)).collect();
    let mut shuffled = vec![0i32; tables.len()];
    for (r, to) in table_row.iter().enumerate() {
        shuffled[*to as usize * table_stride..][..table_stride]
            .copy_from_slice(&tables[r * table_stride..][..table_stride]);
    }
    let mut arena = Arena::default();
    let keys = pool(&mut arena, FP4_E8M0, dim, per_block, &pooled);
    let q_codes_at = arena.put_bytes(&q_codes, 3);
    let q_scales_at = arena.put_bytes(&q_scales, 1);
    let head_dots_at = arena.put(&head_dots);
    let tables_at = arena.put(&shuffled);
    let table_row_at = arena.put(&table_row);
    let visible_at = arena.put(
        &cases
            .iter()
            .map(|c| u32_of(c.keys.len()))
            .collect::<Vec<_>>(),
    );
    let scores_at = arena.put(&vec![0xbeefu16; n * stride]);
    let status_at = arena.put(&[0u32]);
    let p = IndexScoreParams {
        q_codes: Arena::addr(q_codes_at),
        q_scales: Arena::addr(q_scales_at),
        head_dots: Arena::addr(head_dots_at),
        tables: Arena::addr(tables_at),
        table_row: Arena::addr(table_row_at),
        visible: Arena::addr(visible_at),
        scores: Arena::addr(scores_at),
        status: Arena::addr(status_at),
        keys,
        rows: u32_of(n),
        heads: u32_of(heads),
        dim: u32_of(dim),
        table_stride: u32_of(table_stride),
        score_stride: u32_of(stride),
        weight_scale: c0.weight_scale,
    };
    let width = if width == 0 { score_grid(&p)[0] } else { width };
    let mut job = Job::new(format!("{label}, grid {width} wide"), arena);
    job.calls.push(Call::Score(p, width));
    for (r, row) in want.iter().enumerate() {
        job.expect(format!("row {r}'s scores"), scores_at + r * stride * 2, row);
    }
    job.expect("status", status_at, &[status]);
    job
}

fn score_jobs() -> Vec<Job> {
    let mut rng = Rng(0x5c04_e001);
    let rng = &mut rng;
    let mut jobs = Vec::new();
    // Heads under one lane group (summed left to right), exactly one, one and a
    // tail, two; widths of 2 to 8 lane steps; entries around a pass of 8 and
    // around the passes a grid's blocks take in turn.
    for &(heads, dim) in &[
        (1, 32),
        (2, 32),
        (5, 64),
        (16, 32),
        (17, 96),
        (31, 32),
        (32, 128),
    ] {
        let cases: Vec<ScoreCase> = [1, 7, 8, 9, 17, 40, 131]
            .into_iter()
            .map(|visible| ScoreCase::new(rng, heads, dim, visible))
            .collect();
        for width in [0, 1, 3] {
            jobs.push(score_job(&format!("{heads}x{dim}"), &cases, 6, width, None));
        }
        // Infinite scales in the query and in a key, and a zero block.
        let mut wild = ScoreCase::new(rng, heads, dim, 11);
        wild.q[heads * dim / 2] = 0xff;
        wild.keys[0][dim / 2] = 0xff;
        wild.keys[10][..dim / 2].fill(0);
        jobs.push(score_job(
            &format!("{heads}x{dim}, infinite scales"),
            &[wild],
            4,
            2,
            None,
        ));
    }
    // Dots and head sums that are the residue of cancelled terms: the pinned
    // orders show.
    for &(heads, dim) in &[(4, 96), (9, 128), (17, 128), (32, 128)] {
        let cases: Vec<ScoreCase> = [Cancel::Ends, Cancel::Lanes, Cancel::Ends, Cancel::Lanes]
            .into_iter()
            .map(|mode| {
                let mut c = ScoreCase::new(rng, heads, dim, 60);
                c.cancel(rng, mode);
                c
            })
            .collect();
        jobs.push(score_job(
            &format!("{heads}x{dim}, cancelling"),
            &cases,
            16,
            0,
            None,
        ));
    }
    // A bank block the table does not map.
    let cases: Vec<ScoreCase> = (0..3).map(|_| ScoreCase::new(rng, 4, 32, 50)).collect();
    jobs.push(score_job("an unmapped block", &cases, 8, 2, Some((1, 3))));
    jobs
}

/// The released indexer (32 heads of 128) at the longest context: one row
/// seeing 131,072 entries beside rows seeing few.
fn released_score_job() -> Job {
    let mut rng = Rng(0x5c04_e002);
    let cases: Vec<ScoreCase> = [131_072, 1, 1000]
        .into_iter()
        .map(|visible| ScoreCase::new(&mut rng, 32, 128, visible))
        .collect();
    score_job("released indexer", &cases, 64, 0, None)
}

/// One launch over `cases` as its rows (one top-k, block count and block
/// sizes; all with a pool or none; all publishing blocks or none).
fn select_job(label: &str, cases: &[SelectCase]) -> Job {
    let c0 = &cases[0];
    let n = cases.len();
    let stride = cases.iter().map(|c| c.scores.len()).max().unwrap() + 3;
    let words = mask_words(u32_of(stride), u32_of(c0.block_entries)) as usize + 1;
    let table_stride = cases.iter().map(|c| c.table.len()).max().unwrap() + 1;
    let (slot_first, slot_stride) = (5usize, 5 + c0.k + 2);
    let (mut all_scores, mut masks, mut tables) = (Vec::new(), Vec::new(), Vec::new());
    let (mut want_slots, mut want_masks, mut status) = (Vec::new(), Vec::new(), 0);
    for c in cases {
        assert_eq!(
            (
                c.k,
                c.top_blocks,
                c.block_entries,
                c.per_block,
                c.mask_in.is_some(),
                c.mask_out
            ),
            (
                c0.k,
                c0.top_blocks,
                c0.block_entries,
                c0.per_block,
                c0.mask_in.is_some(),
                c0.mask_out
            ),
            "{label}: rows of one launch share a shape"
        );
        let mut row = c.scores.clone();
        // Scores past the visible entries are never read: the best there are.
        row.resize(stride, 0x7f80);
        all_scores.extend(row);
        let pool = c.mask_in.as_ref().map(|m| pack_mask(m, words));
        let mut table = c.table.clone();
        table.resize(table_stride, -1);
        let (slots, kept, bad) = c.expected();
        let kept = pack_mask(&kept, words);
        // The library's host forms agree with the reference.
        let (host, unmapped) = global_slots(
            &c.scores,
            pool.as_deref(),
            c.k,
            c.block_entries,
            &c.table,
            c.per_block,
        );
        assert_eq!((&host, unmapped), (&slots, bad != 0), "{label}");
        assert_eq!(
            candidate_blocks(&c.scores, c.top_blocks, c.block_entries, words),
            kept,
            "{label}"
        );
        masks.extend(pool.unwrap_or_default());
        tables.extend(table);
        want_slots.push(slots);
        want_masks.extend(kept);
        status |= bad;
    }
    let mut arena = Arena::default();
    let scores_at = arena.put(&all_scores);
    let visible_at = arena.put(
        &cases
            .iter()
            .map(|c| u32_of(c.scores.len()))
            .collect::<Vec<_>>(),
    );
    let mask_in_at = c0.mask_in.is_some().then(|| arena.put(&masks));
    let mask_out_at = c0
        .mask_out
        .then(|| arena.put(&vec![0x3333_3333u32; n * words]));
    let tables_at = arena.put(&tables);
    let table_row_at = arena.put(&(0..u32_of(n)).collect::<Vec<_>>());
    let slots_at = arena.put(&vec![0x5a5a_5a5au32; n * slot_stride]);
    let status_at = arena.put(&[0u32]);
    let mut job = Job::new(label, arena);
    job.calls.push(Call::Select(SelectParams {
        scores: Arena::addr(scores_at),
        visible: Arena::addr(visible_at),
        mask_in: mask_in_at.map_or(0, Arena::addr),
        mask_out: mask_out_at.map_or(0, Arena::addr),
        tables: Arena::addr(tables_at),
        table_row: Arena::addr(table_row_at),
        slots: Arena::addr(slots_at),
        status: Arena::addr(status_at),
        rows: u32_of(n),
        score_stride: u32_of(stride),
        mask_words: u32_of(words),
        table_stride: u32_of(table_stride),
        slot_stride: u32_of(slot_stride),
        slot_first: u32_of(slot_first),
        top_k: u32_of(c0.k),
        top_blocks: u32_of(c0.top_blocks),
        block_entries: u32_of(c0.block_entries),
        entries_per_block: u32_of(c0.per_block),
    }));
    for (r, slots) in want_slots.iter().enumerate() {
        let at = slots_at + (r * slot_stride + slot_first) * 4;
        job.expect(format!("row {r}'s selection"), at, slots);
    }
    if let Some(at) = mask_out_at {
        job.expect("candidate blocks", at, &want_masks);
    }
    job.expect("status", status_at, &[status]);
    job
}

/// Rows of one launch: every kind of score at each visible count.
fn select_rows(
    rng: &mut Rng,
    visible: &[usize],
    k: usize,
    top_blocks: usize,
    block_entries: usize,
    pooled: bool,
) -> Vec<SelectCase> {
    let mut rows = Vec::new();
    for &v in visible {
        for kind in 0..5 {
            let mut c = SelectCase::new(scores(rng, v, kind), k, top_blocks, block_entries, 16);
            if pooled {
                c = c.with_pool(rng);
                c.mask_out = false;
            }
            rows.push(c);
        }
    }
    rows
}

fn select_jobs() -> Vec<Job> {
    let mut rng = Rng(0x5e1e_c701);
    let rng = &mut rng;
    let mut jobs = Vec::new();
    // Around the k-th score with ties at it, both zeros, NaNs and infinities;
    // fewer visible entries than k, and none.
    for &(k, block_entries, top_blocks) in &[(1, 1, 1), (3, 2, 2), (24, 4, 8), (8, 8, 3)] {
        let visible = [0, 1, 2, k.max(2) - 1, k, k + 1, 3 * k + 5, 100, 257];
        for pooled in [false, true] {
            let rows = select_rows(rng, &visible, k, top_blocks, block_entries, pooled);
            let label = format!("top {k}, {top_blocks} blocks of {block_entries}, pool {pooled}");
            jobs.push(select_job(&label, &rows));
        }
    }
    // Several candidates a thread and several mask words; ties at the k-th
    // score inside one thread's run and inside one mask word.
    for &(k, block_entries, top_blocks) in &[(25, 2, 5), (513, 2, 37), (512, 8, 2048), (1, 2, 1)] {
        for pooled in [false, true] {
            let rows = select_rows(
                rng,
                &[513, 3001, 20_000],
                k,
                top_blocks,
                block_entries,
                pooled,
            );
            let label = format!("top {k}, {top_blocks} blocks of {block_entries}, pool {pooled}");
            jobs.push(select_job(&format!("{label}, long rows"), &rows));
        }
    }
    // Blocks of NaN or -inf alone are never kept.
    let rows: Vec<SelectCase> = [0xffc0u16, 0xff80, 0x7fc0, 0x3f80, 0x8000]
        .into_iter()
        .map(|fill| {
            let mut s = scores(rng, 90, 2);
            s[8..40].fill(fill);
            SelectCase::new(s, 60, 30, 4, 16)
        })
        .collect();
    assert!(rows[1].expected().1[2..10].iter().all(|kept| !kept));
    jobs.push(select_job("blocks without a score", &rows));
    // A selected entry behind a table word that maps nothing.
    let mut rows = select_rows(rng, &[200], 24, 3, 4, false);
    let middle = rows[2].chosen()[12];
    rows[2].table[middle / 16] = -7;
    jobs.push(select_job("an unmapped block", &rows));
    // No top-k and no candidate block at all.
    let rows = select_rows(rng, &[200], 0, 0, 4, false);
    jobs.push(select_job("nothing to select", &rows));
    jobs
}

/// The released selection (top 512; 2,048 candidate blocks of 8) from 1 to
/// 131,072 visible entries.
fn released_select_jobs() -> Vec<Job> {
    let mut rng = Rng(0x5e1e_c702);
    let visible = [1, 511, 512, 513, 16_384, 16_385, 131_071, 131_072];
    [false, true]
        .into_iter()
        .map(|pooled| {
            let rows: Vec<SelectCase> = visible
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let mut c = SelectCase::new(
                        scores(&mut rng, *v, [2, 1, 3, 0][i % 4]),
                        512,
                        2048,
                        8,
                        64,
                    );
                    if pooled {
                        c = c.with_pool(&mut rng);
                        c.mask_out = false;
                    }
                    c
                })
                .collect();
            select_job(&format!("released selection, pool {pooled}"), &rows)
        })
        .collect()
}

/// The shapes of a layer's chain.
struct Shape {
    heads: usize,
    dim: usize,
    window: usize,
    index_heads: usize,
    index_dim: usize,
    top_k: usize,
    top_blocks: usize,
    block_entries: usize,
}

/// A candidate source's layer and a later index source's, launch by launch,
/// for rows at `positions`: the window slots, the index scores, the candidate
/// blocks with a selection over everything visible, attention over that list;
/// then a selection inside the candidate blocks and attention over its list.
/// Each launch reads what the one before wrote, and the expected bytes are
/// the reference's functions composed the same way.
fn chain_job(label: &str, shape: &Shape, positions: &[usize], seed: u64) -> Job {
    let mut rng = Rng(seed);
    let rng = &mut rng;
    let n = positions.len();
    let (bp, epb) = (16usize, 8usize);
    let Shape {
        heads,
        dim,
        window,
        index_heads,
        index_dim,
        top_k,
        top_blocks,
        block_entries,
    } = *shape;
    // Every row its own blocks in the shared pools: window blocks by position,
    // bank blocks by entry (one entry a position).
    let (mut win_tables, mut bank_tables) = (Vec::new(), Vec::new());
    let (mut win_blocks, mut bank_blocks) = (0usize, 0usize);
    let win_stride = positions.iter().max().unwrap() / bp + 2;
    let bank_stride = (positions.iter().max().unwrap() + 1).div_ceil(epb) + 1;
    for p in positions {
        let mut t = vec![-1i32; win_stride];
        for word in &mut t[(p + 1).saturating_sub(window) / bp..=p / bp] {
            *word = i32::try_from(win_blocks).unwrap();
            win_blocks += 1;
        }
        win_tables.push(t);
        let mut t = vec![-1i32; bank_stride];
        for word in t.iter_mut().take((p + 1).div_ceil(epb)) {
            *word = i32::try_from(bank_blocks).unwrap();
            bank_blocks += 1;
        }
        bank_tables.push(t);
    }
    let win_pool: Vec<Vec<u8>> = (0..win_blocks * bp)
        .map(|_| entry(rng, FP8, dim, 0))
        .collect();
    let kv_pool: Vec<Vec<u8>> = (0..bank_blocks * epb)
        .map(|_| entry(rng, FP4_E4M3, dim, 0))
        .collect();
    let key_pool: Vec<Vec<u8>> = (0..bank_blocks * epb)
        .map(|_| entry(rng, FP4_E8M0, index_dim, 0))
        .collect();
    let base_attn =
        deepseek_support::cases::attn_case(rng, heads, dim, 0, 0, [FP8, FP4_E4M3], 0, 0);
    let words = mask_words(
        u32_of(positions.iter().max().unwrap() + 4),
        u32_of(block_entries),
    ) as usize;
    let stride = window + top_k;
    let score_stride = positions.iter().max().unwrap() + 4;

    let (mut q, mut iq_codes, mut iq_scales, mut head_dots) =
        (Vec::<u16>::new(), Vec::new(), Vec::new(), Vec::<u16>::new());
    let mut want_scores = Vec::new();
    let (mut want_lists, mut want_out, mut want_masks) = (
        [Vec::new(), Vec::new()],
        [Vec::new(), Vec::new()],
        Vec::new(),
    );
    let mut weight_scale = 0.0;
    for (r, p) in positions.iter().enumerate() {
        let visible = p + 1;
        let (window_words, bad) = WindowCase {
            table: win_tables[r].clone(),
            ..WindowCase::new(0, window, bp, *p, 0)
        }
        .expected();
        assert_eq!(bad, 0);
        let mut score = ScoreCase::new(rng, index_heads, index_dim, 0);
        score.keys = (0..visible)
            .map(|t| {
                key_pool[usize::try_from(bank_tables[r][t / epb]).unwrap() * epb + t % epb].clone()
            })
            .collect();
        weight_scale = score.weight_scale;
        let (codes, scales) = score.q.split_at(index_heads * index_dim / 2);
        iq_codes.extend_from_slice(codes);
        iq_scales.extend_from_slice(scales);
        head_dots.extend(&score.head_dots);
        let row_scores = score.expected();
        let mut select = SelectCase::new(row_scores.clone(), top_k, top_blocks, block_entries, epb);
        select.table = bank_tables[r].clone();
        let (first, kept, bad) = select.expected();
        assert_eq!(bad, 0);
        select.mask_in = Some(kept.clone());
        let (second, _, _) = select.expected();
        want_masks.extend(pack_mask(&kept, words));
        want_scores.push(row_scores);
        let row_q: Vec<u16> = (0..heads * dim).map(|_| rng.bf16(-3, 1)).collect();
        for (i, global) in [first, second].into_iter().enumerate() {
            let slots = [window_words.clone(), global].concat();
            let attn = AttnCase {
                q: row_q.clone(),
                slots: slots.clone(),
                a_slots: window,
                entries: [win_pool.clone(), kv_pool.clone()],
                ..base_attn.clone()
            };
            want_out[i].extend(attn.expected());
            want_lists[i].push(slots);
        }
        q.extend(row_q);
    }

    let mut arena = Arena::default();
    let a = pool(&mut arena, FP8, dim, bp, &win_pool);
    let b = pool(&mut arena, FP4_E4M3, dim, epb, &kv_pool);
    let keys = pool(&mut arena, FP4_E8M0, index_dim, epb, &key_pool);
    let positions_at = arena.put(&positions.iter().map(|p| u32_of(*p)).collect::<Vec<_>>());
    let first_at = arena.put(&vec![0u32; n]);
    let visible_at = arena.put(&positions.iter().map(|p| u32_of(p + 1)).collect::<Vec<_>>());
    let win_tables_at = arena.put(&win_tables.concat());
    let bank_tables_at = arena.put(&bank_tables.concat());
    let table_row_at = arena.put(&(0..u32_of(n)).collect::<Vec<_>>());
    let q_at = arena.put(&q);
    let sink_at = arena.put(&base_attn.sink);
    let iq_codes_at = arena.put_bytes(&iq_codes, 0);
    let iq_scales_at = arena.put_bytes(&iq_scales, 0);
    let head_dots_at = arena.put(&head_dots);
    let scores_at = arena.put(&vec![0xbeefu16; n * score_stride]);
    let mask_at = arena.put(&vec![0x3333_3333u32; n * words]);
    let lists_at = [0, 1].map(|_| arena.put(&vec![0x5a5a_5a5au32; n * stride]));
    let out_at = [0, 1].map(|_| arena.put(&vec![0xdeadu16; q.len()]));
    let status_at = arena.put(&[0u32]);
    let mut job = Job::new(label, arena);
    let rows = u32_of(n);
    let score = IndexScoreParams {
        q_codes: Arena::addr(iq_codes_at),
        q_scales: Arena::addr(iq_scales_at),
        head_dots: Arena::addr(head_dots_at),
        tables: Arena::addr(bank_tables_at),
        table_row: Arena::addr(table_row_at),
        visible: Arena::addr(visible_at),
        scores: Arena::addr(scores_at),
        status: Arena::addr(status_at),
        keys,
        rows,
        heads: u32_of(index_heads),
        dim: u32_of(index_dim),
        table_stride: u32_of(bank_stride),
        score_stride: u32_of(score_stride),
        weight_scale,
    };
    job.calls.push(Call::Score(score, score_grid(&score)[0]));
    for (i, list_at) in lists_at.into_iter().enumerate() {
        job.calls.push(Call::Window(WindowSlotsParams {
            positions: Arena::addr(positions_at),
            first: Arena::addr(first_at),
            tables: Arena::addr(win_tables_at),
            table_row: Arena::addr(table_row_at),
            slots: Arena::addr(list_at),
            status: Arena::addr(status_at),
            rows,
            window: u32_of(window),
            block_positions: u32_of(bp),
            table_stride: u32_of(win_stride),
            slot_stride: u32_of(stride),
            order: 0,
        }));
        // The candidate source publishes the blocks and selects over everything; the later
        // source selects inside them.
        let (mask_in, mask_out) = if i == 0 {
            (0, Arena::addr(mask_at))
        } else {
            (Arena::addr(mask_at), 0)
        };
        job.calls.push(Call::Select(SelectParams {
            scores: Arena::addr(scores_at),
            visible: Arena::addr(visible_at),
            mask_in,
            mask_out,
            tables: Arena::addr(bank_tables_at),
            table_row: Arena::addr(table_row_at),
            slots: Arena::addr(list_at),
            status: Arena::addr(status_at),
            rows,
            score_stride: u32_of(score_stride),
            mask_words: u32_of(words),
            table_stride: u32_of(bank_stride),
            slot_stride: u32_of(stride),
            slot_first: u32_of(window),
            top_k: u32_of(top_k),
            top_blocks: u32_of(top_blocks),
            block_entries: u32_of(block_entries),
            entries_per_block: u32_of(epb),
        }));
        let p = AttnParams {
            q: Arena::addr(q_at),
            slots: Arena::addr(list_at),
            sink: Arena::addr(sink_at),
            out: Arena::addr(out_at[i]),
            a,
            b,
            rows,
            heads: u32_of(heads),
            head_dim: u32_of(dim),
            a_slots: u32_of(window),
            b_slots: u32_of(top_k),
            slot_stride: u32_of(stride),
            scale: base_attn.scale,
            layout: ATTN_FIXED,
        };
        let group = eidola_engine_cuda::deepseek::attention::heads_per_block(rows, p.heads);
        job.calls.push(Call::Attn(p, group));
        job.expect(format!("slot lists {i}"), list_at, &want_lists[i].concat());
        job.expect(format!("attention output {i}"), out_at[i], &want_out[i]);
    }
    for (r, row) in want_scores.iter().enumerate() {
        job.expect(
            format!("row {r}'s scores"),
            scores_at + r * score_stride * 2,
            row,
        );
    }
    job.expect("candidate blocks", mask_at, &want_masks);
    job.expect("status", status_at, &[0u32]);
    // The two selections differ somewhere, or the pool was never exercised.
    assert!(top_blocks * block_entries >= top_k, "{label}");
    job
}

fn chain_jobs() -> Vec<Job> {
    // The synthetic fixture's shapes, then the released ones at a short context.
    let tiny = Shape {
        heads: 4,
        dim: 32,
        window: 48,
        index_heads: 2,
        index_dim: 32,
        top_k: 24,
        top_blocks: 8,
        block_entries: 4,
    };
    let released = Shape {
        heads: 64,
        dim: 512,
        window: 128,
        index_heads: 32,
        index_dim: 128,
        top_k: 512,
        top_blocks: 2048,
        block_entries: 8,
    };
    vec![
        chain_job(
            "chain, fixture shapes",
            &tiny,
            &[0, 1, 22, 47, 48, 95, 300, 301],
            0xc4a1,
        ),
        chain_job("chain, released shapes", &released, &[5, 700], 0xc4a2),
    ]
}

fn all_jobs() -> Vec<Job> {
    let mut jobs = score_jobs();
    jobs.push(released_score_job());
    jobs.extend(select_jobs());
    jobs.extend(released_select_jobs());
    jobs.extend(chain_jobs());
    jobs
}

/// The kernel bodies, compiled for the host, on every case the device runs.
#[test]
fn bodies_match_the_reference_on_the_host() {
    let jobs = all_jobs();
    assert!(jobs.len() > 50, "{}", jobs.len());
    run_on_host(&jobs);
}

#[test]
fn kernels_match_the_reference_on_the_device() {
    let Some(su) = common::setup() else { return };
    let jobs = all_jobs();
    for &arch in &su.archs {
        let ops = Ops::load(&su.gpu, &su.dir, arch);
        run_on_device(&su.gpu, &ops, &jobs);
        println!(
            "{arch:?}: {} index, selection and chain jobs match",
            jobs.len()
        );
    }
}

/// A score launch at the released indexer over `n` rows that all read one
/// bank of `visible` entries.
fn bench_score_job(rng: &mut Rng, n: usize, visible: usize) -> Job {
    let (heads, dim, epb) = (32usize, 128usize, 64usize);
    let keys: Vec<Vec<u8>> = (0..visible).map(|_| entry(rng, FP4_E8M0, dim, 0)).collect();
    let blocks = visible.div_ceil(epb);
    let table: Vec<i32> = (0..blocks).map(|b| i32::try_from(b).unwrap()).collect();
    let q = entry(rng, FP4_E8M0, heads * dim, 0);
    let (codes, scales) = q.split_at(heads * dim / 2);
    let mut arena = Arena::default();
    let keys = pool(&mut arena, FP4_E8M0, dim, epb, &keys);
    let q_codes_at = arena.put_bytes(&codes.repeat(n), 0);
    let q_scales_at = arena.put_bytes(&scales.repeat(n), 0);
    let head_dots_at = arena.put(&(0..n * heads).map(|_| rng.bf16(-2, 4)).collect::<Vec<_>>());
    let tables_at = arena.put(&table);
    let table_row_at = arena.put(&vec![0u32; n]);
    let visible_at = arena.put(&vec![u32_of(visible); n]);
    let scores_at = arena.put(&vec![0u16; n * visible]);
    let status_at = arena.put(&[0u32]);
    let p = IndexScoreParams {
        q_codes: Arena::addr(q_codes_at),
        q_scales: Arena::addr(q_scales_at),
        head_dots: Arena::addr(head_dots_at),
        tables: Arena::addr(tables_at),
        table_row: Arena::addr(table_row_at),
        visible: Arena::addr(visible_at),
        scores: Arena::addr(scores_at),
        status: Arena::addr(status_at),
        keys,
        rows: u32_of(n),
        heads: u32_of(heads),
        dim: u32_of(dim),
        table_stride: u32_of(blocks),
        score_stride: u32_of(visible),
        weight_scale: bf16_value(0x3c80),
    };
    let mut job = Job::new("bench", arena);
    job.calls.push(Call::Score(p, score_grid(&p)[0]));
    job
}

/// A selection launch at the released shapes over `n` rows of `visible`
/// entries: a candidate source's (blocks out) or a later source's (blocks in).
fn bench_select_job(rng: &mut Rng, n: usize, visible: usize, pooled: bool) -> Job {
    let rows: Vec<SelectCase> = (0..n)
        .map(|_| {
            let mut c = SelectCase::new(scores(rng, visible, 2), 512, 2048, 8, 64);
            if pooled {
                c = c.with_pool(rng);
                c.mask_out = false;
            }
            c
        })
        .collect();
    select_job("bench", &rows)
}

/// The bench's launches build without a device, at one row.
#[test]
fn bench_launches_build_without_a_device() {
    let mut rng = Rng(0xbe);
    let job = bench_score_job(&mut rng, 1, 4096);
    let Call::Score(p, width) = job.calls[0] else {
        unreachable!()
    };
    eidola_engine_cuda::deepseek::index::check_score(&p).unwrap();
    assert_eq!(width, 512);
    for pooled in [false, true] {
        let job = bench_select_job(&mut rng, 1, 4096, pooled);
        let Call::Select(p) = job.calls[0] else {
            unreachable!()
        };
        eidola_engine_cuda::deepseek::index::check_select(&p).unwrap();
    }
}

/// Microseconds per query at the released shapes, at 1, 64 and 384 queries,
/// at a short and at the longest context.
#[test]
fn timings_on_the_device() {
    use eidola_engine_cuda::launch::dptr;
    let Some(su) = common::setup() else { return };
    let arch = su.archs[0];
    let ops = Ops::load(&su.gpu, &su.dir, arch);
    let s = su.gpu.stream();
    let mut rng = Rng(0xbe);
    let mut timings = Vec::new();
    let mut time = |kernel: &'static str, shape: String, n: usize, job: Job, iters: usize| {
        let buffer = s.clone_htod(&job.arena).unwrap();
        let call = job.calls[0].moved_to(dptr(&buffer, s));
        let us = deepseek_support::time_us(&su.gpu, iters, &|| {
            // SAFETY: the call's arrays lie in `buffer`, alive past the timing.
            unsafe { ops.launch(&su.gpu, &call) }
        });
        timings.push(Timing {
            kernel,
            shape,
            rows: u32_of(n),
            launch_us: us,
        });
    };
    for visible in [4096usize, 131_072] {
        for n in [1usize, 64, 384] {
            let iters = if visible * n > 1 << 20 { 3 } else { 50 };
            let job = bench_score_job(&mut rng, n, visible);
            let shape = format!("32x128, {visible} visible entries");
            time("eidola_ds_index_score", shape, n, job, iters);
            for pooled in [false, true] {
                let job = bench_select_job(&mut rng, n, visible, pooled);
                let shape = format!(
                    "top 512 of {visible}, {}",
                    if pooled {
                        "inside candidate blocks"
                    } else {
                        "publishing 2048 blocks of 8"
                    }
                );
                time("eidola_ds_select", shape, n, job, iters.max(20));
            }
        }
    }
    deepseek_support::write_bench("ds_index_bench", arch, &timings);
}
