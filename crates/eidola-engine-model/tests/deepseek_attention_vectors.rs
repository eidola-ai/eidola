//! Golden vectors of the DeepSeek-V4.1 sparse attention, index scores, top-k
//! selections and slot lists, emitted from the reference functions for the GPU
//! kernels' bodies in `eidola-engine-kernels` (`csrc/ds_attention.cuh`,
//! `csrc/ds_index.cuh`) to be checked against bit for bit on a host
//! (`oracle/ds_attention_host.cc`).
//!
//! `vectors_are_current` regenerates `ds_attention.vectors` and requires the
//! committed file to equal it; `EIDOLA_UPDATE_DS_VECTORS=1` rewrites the file
//! instead.
//!
//! The cases and their expected results come from `deepseek_attention_cases`,
//! which `eidola-engine-cuda`'s tests share.
//!
//! # File format
//!
//! One case per line, tokens separated by spaces, every number hexadecimal. A
//! run of bytes is one token of hex digits (`-` when empty). BF16 values are
//! their 16 bits and f32 values their 32; a NaN result is written `7fc0`. An
//! empty slot is `ffffffff`. KV entries are given packed, as a pool stores
//! them: an entry's codes, then its scales (FP8: E4M3 codes and an exponent
//! byte per 32; FP4: E2M1 nibbles, element `2i` in the low nibble of byte `i`,
//! with an E4M3 scale per 16 (format 1) or an exponent byte per 32 (format 2)).
//!
//! - `attn <heads> <dim> <scale> <layout> <a format> <b format> <a slots>
//!   <b slots> <a entries per block> <b entries per block> : <q> <sink> <slots>
//!   <a entries> <b entries> = <out>`: `sparse_attend` of one query. The slot
//!   words index the entries of plane `a` (the first `a slots`) and of plane
//!   `b`. Layout 1 lists the `b` slots right after the last listed `a` slot.
//! - `win <order> <window> <positions per block> <position> <first>
//!   <table words> : <table> = <slots> <status>`: the window part of a slot
//!   list, in position order (0) or ring order (1), as pool entries through a
//!   block table.
//! - `score <heads> <dim> <weight scale> <keys per block> <visible> :
//!   <q codes> <q scales> <head dots> <key entries> = <scores>`: `index_score`
//!   of every visible entry.
//! - `select <visible> <top k> <top blocks> <block entries>
//!   <entries per block> <mask words> <mask in?> <mask out?> : <scores>
//!   <mask in> <table> = <slots> <mask out> <status>`: the selected entries
//!   ascending, as pool entries through a block table, and the candidate
//!   blocks (bit `b % 32` of word `b / 32`).

use std::fmt::Write as _;
use std::path::PathBuf;

mod deepseek_attention_cases;

use deepseek_attention_cases::{
    AttnCase, Cancel, EMPTY, FP4_E4M3, FP4_E8M0, FP8, Rng, ScoreCase, SelectCase, WindowCase,
    attn_case, mask_words, scores,
};

fn hex(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "-".into();
    }
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn words<T: std::fmt::LowerHex>(out: &mut String, items: &[T]) {
    for x in items {
        write!(out, " {x:x}").unwrap();
    }
}

fn emit_attn(out: &mut String, rng: &mut Rng, c: &AttnCase) {
    write!(
        out,
        "attn {:x} {:x} {:x} {:x} {:x} {:x} {:x} {:x} {:x} {:x} :",
        c.heads,
        c.dim,
        c.scale.to_bits(),
        c.packed as u32,
        c.formats[0],
        c.formats[1],
        c.a_slots,
        c.slots.len() - c.a_slots,
        1 + rng.below(7),
        1 + rng.below(7)
    )
    .unwrap();
    words(out, &c.q);
    words(out, &c.sink.iter().map(|s| s.to_bits()).collect::<Vec<_>>());
    words(out, &c.slots);
    for p in 0..2 {
        write!(out, " {}", hex(&c.entries[p].concat())).unwrap();
    }
    out.push_str(" =");
    words(out, &c.expected());
    out.push('\n');
}

fn attn_cases(out: &mut String) {
    let mut rng = Rng(0xa77e);
    let rng = &mut rng;
    let backbone = [FP8, FP4_E4M3];
    // Shapes: one lane step to 32, head groups of every size, lists inside one
    // block, ending on a block edge and crossing several.
    for &(heads, dim) in &[(1, 32), (2, 32), (4, 32), (9, 32), (17, 32)] {
        for &(a, b) in &[(3, 0), (48, 24), (64, 5), (70, 70), (128, 70)] {
            for empty_in in [0, 3] {
                let c = attn_case(rng, heads, dim, a, b, backbone, 0, empty_in);
                emit_attn(out, rng, &c);
            }
        }
    }
    // Scores that are the residue of cancelled terms: any other summation
    // order, or a fused multiply-add, moves them.
    for &(heads, dim) in &[(3, 64), (2, 128), (1, 512)] {
        for &(a, b) in &[(48, 24), (64, 0), (70, 70)] {
            for mode in [Cancel::Ends, Cancel::Lanes] {
                let formats = if a == 64 { [FP8, FP8] } else { backbone };
                let mut c = attn_case(rng, heads, dim, a, b, formats, 0, 4);
                c.cancel(rng, mode);
                emit_attn(out, rng, &c);
            }
        }
    }
    // The widest magnitudes: weights that underflow against a later block's
    // maximum.
    for exp in [-6, 3, 5] {
        let c = attn_case(rng, 2, 64, 70, 70, backbone, exp, 5);
        emit_attn(out, rng, &c);
    }
    // Every format in either plane.
    for formats in [[FP8, FP8], [FP4_E8M0, FP4_E4M3], [FP4_E4M3, FP4_E8M0]] {
        let c = attn_case(rng, 2, 32, 40, 40, formats, 0, 4);
        emit_attn(out, rng, &c);
    }
    // A window not yet full: right-aligned, so its first block may be empty.
    for listed in [1, 2, 63, 64, 65, 127] {
        let mut c = attn_case(rng, 2, 32, 128, 24, backbone, 0, 0);
        c.slots[..128 - listed].fill(EMPTY);
        emit_attn(out, rng, &c);
    }
    // No entry at all, one entry, and no window slots.
    let mut c = attn_case(rng, 3, 32, 48, 24, backbone, 0, 0);
    c.slots.fill(EMPTY);
    emit_attn(out, rng, &c);
    let mut c = attn_case(rng, 3, 32, 48, 24, backbone, 0, 0);
    c.slots[..71].fill(EMPTY);
    emit_attn(out, rng, &c);
    let c = attn_case(rng, 2, 32, 0, 5, backbone, 0, 0);
    emit_attn(out, rng, &c);
    // Trailing global slots empty: a selection shorter than the top-k.
    let mut c = attn_case(rng, 2, 32, 48, 80, backbone, 0, 0);
    c.slots[60..].fill(EMPTY);
    emit_attn(out, rng, &c);
    // Sinks: none (-inf), dominant, negligible, and NaN.
    for sink in [f32::NEG_INFINITY, 40.0, -40.0, f32::NAN] {
        let mut c = attn_case(rng, 2, 32, 48, 24, backbone, 0, 5);
        c.sink[1] = sink;
        emit_attn(out, rng, &c);
    }
    // The drafter's list: the window in ring order without the slots a short
    // sequence has not reached, then the draft slots (FP8, from scratch).
    for listed in [1, 3, 59, 60, 64, 100, 128] {
        let mut c = attn_case(rng, 2, 32, 128, 5, [FP8, FP8], 0, 0);
        c.packed = true;
        c.slots[listed..128].fill(EMPTY);
        emit_attn(out, rng, &c);
    }
    // The packed layout leaves an empty slot inside the window in place.
    let mut c = attn_case(rng, 2, 32, 70, 5, [FP8, FP8], 0, 0);
    c.packed = true;
    c.slots[10] = EMPTY;
    c.slots[66..70].fill(EMPTY);
    emit_attn(out, rng, &c);
    // An infinite scale: the entry's values are NaN (zero codes) or infinite.
    let mut c = attn_case(rng, 2, 32, 48, 24, backbone, 0, 0);
    let first = c.slots[5] as usize;
    c.entries[0][first][32] = 0xff;
    emit_attn(out, rng, &c);
    // A block whose only score is -inf is skipped; one after it is not.
    for with_later in [false, true] {
        let mut c = attn_case(rng, 1, 32, 130, 0, [FP8, FP8], 0, 0);
        c.q.fill(0xbf80);
        let keep = c.slots[70] as usize;
        c.entries[0][keep] = [vec![0x38; 32], vec![0xff]].concat();
        for (s, w) in c.slots.iter_mut().enumerate() {
            if s != 70 && !(with_later && s == 129) {
                *w = EMPTY;
            }
        }
        emit_attn(out, rng, &c);
    }
}

fn emit_window(out: &mut String, c: &WindowCase) {
    let (slots, status) = c.expected();
    write!(
        out,
        "win {:x} {:x} {:x} {:x} {:x} {:x} :",
        c.order,
        c.window,
        c.per_block,
        c.pos,
        c.first,
        c.table.len()
    )
    .unwrap();
    words(out, &c.table.iter().map(|w| *w as u32).collect::<Vec<_>>());
    out.push_str(" =");
    words(out, &slots);
    writeln!(out, " {status:x}").unwrap();
}

fn window_cases(out: &mut String) {
    let mut rng = Rng(0x51075);
    for &(window, per_block) in &[(4, 4), (48, 16), (128, 16), (128, 256), (129, 8), (300, 32)] {
        let edges = [0, 1, window - 2, window - 1, window, window + 1];
        for pos in edges.into_iter().chain([5 * window + 3]) {
            for order in 0..2 {
                emit_window(out, &WindowCase::new(order, window, per_block, pos, 0));
            }
        }
        for _ in 0..4 {
            let pos = rng.below(40 * window);
            for order in 0..2 {
                emit_window(out, &WindowCase::new(order, window, per_block, pos, 0));
            }
            // A window cut at a first position, at and inside its range.
            let first = pos.saturating_sub(rng.below(window + 2));
            emit_window(out, &WindowCase::new(0, window, per_block, pos, first));
        }
        let at = 3 * window;
        emit_window(out, &WindowCase::new(0, window, per_block, at, at));
        // A table word that maps nothing, inside the window and outside it.
        let pos = 9 * window + 5;
        for (order, block) in [(0, pos / per_block), (1, pos / per_block), (0, 0)] {
            let mut c = WindowCase::new(order, window, per_block, pos, 0);
            c.table[block] = -1;
            emit_window(out, &c);
        }
    }
}

fn emit_score(out: &mut String, rng: &mut Rng, c: &ScoreCase) {
    let (codes, scales) = c.q.split_at(c.heads * c.dim / 2);
    write!(
        out,
        "score {:x} {:x} {:x} {:x} {:x} : {} {}",
        c.heads,
        c.dim,
        c.weight_scale.to_bits(),
        1 + rng.below(9),
        c.keys.len(),
        hex(codes),
        hex(scales)
    )
    .unwrap();
    words(out, &c.head_dots);
    write!(out, " {} =", hex(&c.keys.concat())).unwrap();
    words(out, &c.expected());
    out.push('\n');
}

fn score_cases(out: &mut String) {
    let mut rng = Rng(0x5c04e);
    let rng = &mut rng;
    // Heads under one lane group (left to right), exactly one, one and a tail,
    // two; widths of 2 to 8 lane steps; entries around a pass of 8.
    for &(heads, dim) in &[
        (1, 32),
        (2, 32),
        (5, 64),
        (16, 32),
        (17, 32),
        (31, 32),
        (32, 128),
    ] {
        for visible in [1, 7, 8, 9, 17, 40] {
            let c = ScoreCase::new(rng, heads, dim, visible);
            emit_score(out, rng, &c);
        }
        // An infinite scale in the query and in a key, and a zero block.
        let mut c = ScoreCase::new(rng, heads, dim, 11);
        c.q[heads * dim / 2] = 0xff;
        c.keys[0][dim / 2] = 0xff;
        c.keys[10][..dim / 2].fill(0);
        emit_score(out, rng, &c);
    }
    // Dots and head sums that are the residue of cancelled terms.
    for &(heads, dim) in &[(4, 96), (9, 128), (17, 128), (32, 128)] {
        for mode in [Cancel::Ends, Cancel::Lanes] {
            let mut c = ScoreCase::new(rng, heads, dim, 20);
            c.cancel(rng, mode);
            emit_score(out, rng, &c);
        }
    }
}

fn emit_select(out: &mut String, rng: &mut Rng, c: &SelectCase) {
    let (slots, kept, status) = c.expected();
    let words_in_mask = c.blocks().div_ceil(32) + rng.below(3) + 1;
    write!(
        out,
        "select {:x} {:x} {:x} {:x} {:x} {words_in_mask:x} {:x} {:x} :",
        c.scores.len(),
        c.k,
        c.top_blocks,
        c.block_entries,
        c.per_block,
        c.mask_in.is_some() as u32,
        c.mask_out as u32
    )
    .unwrap();
    words(out, &c.scores);
    if let Some(pool) = &c.mask_in {
        words(out, &mask_words(pool, words_in_mask));
    }
    words(out, &c.table.iter().map(|w| *w as u32).collect::<Vec<_>>());
    out.push_str(" =");
    words(out, &slots);
    if c.mask_out {
        words(out, &mask_words(&kept, words_in_mask));
    }
    writeln!(out, " {status:x}").unwrap();
}

fn select_cases(out: &mut String) {
    let mut rng = Rng(0x5e1ec7);
    let rng = &mut rng;
    // Around the k-th score, with ties at it, for every kind of score.
    for kind in 0..4 {
        for &(k, block_entries, top_blocks) in &[(1, 1, 1), (3, 2, 2), (24, 4, 8), (8, 8, 3)] {
            for visible in [0, 1, 2, k.max(2) - 1, k, k + 1, 3 * k + 5, 100, 257] {
                let per_block = 1 + rng.below(40);
                let s = scores(rng, visible, kind);
                let mut c = SelectCase::new(s, k, top_blocks, block_entries, per_block);
                if visible % 2 == 1 {
                    c = c.with_pool(rng);
                }
                emit_select(out, rng, &c);
            }
        }
    }
    // Runs of several candidates a thread, and of several mask words; the
    // released top-k and block counts.
    for &(visible, k, top_blocks, block_entries, kind) in &[
        (513, 24, 8, 4, 1),
        (700, 512, 2048, 8, 2),
        (1500, 512, 40, 8, 3),
        (5000, 512, 100, 8, 1),
        (5000, 24, 2048, 2, 0),
        (20000, 512, 2048, 8, 2),
    ] {
        for pool in [false, true] {
            let s = scores(rng, visible, kind);
            let mut c = SelectCase::new(s, k, top_blocks, block_entries, 64);
            if pool {
                c = c.with_pool(rng);
                c.mask_out = false;
            }
            emit_select(out, rng, &c);
        }
    }
    // Blocks of NaN or -inf alone are never kept; a run of equal scores.
    for fill in [0xffc0u16, 0xff80, 0x7fc0, 0x3f80, 0x8000] {
        let mut s = scores(rng, 90, 2);
        s[8..40].fill(fill);
        for (k, top_blocks) in [(7, 5), (60, 30)] {
            emit_select(out, rng, &SelectCase::new(s.clone(), k, top_blocks, 4, 16));
        }
    }
    // Ties at the k-th score inside one thread's run of candidates and inside
    // one mask word: every score equal, then a few values.
    for kind in [4, 1] {
        for &(visible, k, top_blocks) in &[(3001, 25, 5), (3001, 513, 37), (4000, 1, 1)] {
            for pool in [false, true] {
                let s = scores(rng, visible, kind);
                let mut c = SelectCase::new(s, k, top_blocks, 2, 64);
                if pool {
                    c = c.with_pool(rng);
                }
                emit_select(out, rng, &c);
            }
        }
    }
    // A selected entry behind a table word that maps nothing; no top-k and no
    // candidate block at all.
    let mut c = SelectCase::new(scores(rng, 200, 2), 24, 3, 4, 16);
    let middle = c.chosen()[12];
    c.table[middle / 16] = -7;
    emit_select(out, rng, &c);
    let c = SelectCase::new(scores(rng, 200, 2), 0, 0, 4, 16);
    emit_select(out, rng, &c);
}

fn generate() -> String {
    let mut out = String::from(
        "# DeepSeek-V4.1 attention, index scores, selections and slot lists from the reference.\n\
         # Emitted by eidola-engine-model's tests/deepseek_attention_vectors.rs, which documents\n\
         # the format.\n",
    );
    attn_cases(&mut out);
    window_cases(&mut out);
    score_cases(&mut out);
    select_cases(&mut out);
    out
}

#[test]
fn vectors_are_current() {
    let generated = generate();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../eidola-engine-kernels/tests/data/ds_attention.vectors");
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
