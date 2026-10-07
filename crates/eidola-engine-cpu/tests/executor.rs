//! The executor driven directly through the seam, without the scheduler: maintenance
//! semantics (zero really zeroes, copy really copies), drafter state carried by boundary
//! taps, batch-composition independence, and panics on contract violations.

mod common;

use common::*;
use eidola_engine::executor::{
    Executor, Maintenance, SeqEntry, Slot, StepInput, StepOutput, TableUpdate,
};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::NULL_BLOCK;
use eidola_engine_cpu::{CpuExecutor, DenseRun};

fn executor() -> CpuExecutor {
    let mut cfg = exec_config(16, vec![0, 1]);
    cfg.num_state_slots = 4;
    CpuExecutor::new(fixture(), cfg)
}

fn groups(e: &CpuExecutor) -> u32 {
    e.spec().kv_groups.len() as u32
}

/// `table[slot][g][index] = block` in every group.
fn map(e: &CpuExecutor, slot: Slot, index: u32, block: u32) -> Vec<TableUpdate> {
    (0..groups(e))
        .map(|group| TableUpdate {
            slot,
            group,
            index,
            block,
        })
        .collect()
}

struct Row {
    slot: Slot,
    context_len: u32,
    tokens: Vec<u32>,
    sample: bool,
    drafts: u32,
}

fn step(rows: &[Row], maintenance: Vec<Maintenance>, table_updates: Vec<TableUpdate>) -> StepInput {
    let mut seqs = Vec::new();
    let mut token_ids = Vec::new();
    let mut positions = Vec::new();
    for r in rows {
        seqs.push(SeqEntry {
            slot: r.slot,
            token_start: token_ids.len() as u32,
            num_tokens: r.tokens.len() as u32,
            context_len: r.context_len,
            num_drafts: r.drafts,
            sample: r.sample,
            sampling: SamplingParams::greedy(),
        });
        token_ids.extend(&r.tokens);
        positions.extend(r.context_len..r.context_len + r.tokens.len() as u32);
    }
    StepInput {
        bucket: eidola_engine::spec::Bucket {
            max_seqs: 32,
            max_tokens: 256,
        },
        maintenance,
        table_updates,
        seqs,
        token_ids,
        positions,
        return_logits: true,
    }
}

fn prefill(slot: Slot, tokens: &[u32]) -> Row {
    Row {
        slot,
        context_len: 0,
        tokens: tokens.to_vec(),
        sample: false,
        drafts: 0,
    }
}

fn decode(slot: Slot, context_len: u32, token: u32, drafts: u32) -> Row {
    Row {
        slot,
        context_len,
        tokens: vec![token],
        sample: true,
        drafts,
    }
}

fn logits(out: &StepOutput, row: usize) -> &Vec<Vec<f32>> {
    &out.logits.as_ref().unwrap()[row]
}

fn dense(tokens: &[u32]) -> DenseRun {
    let m = fixture();
    eidola_engine_cpu::DenseOracle {
        model: &m,
        mtp_depths: &[0, 1],
        mtp_hidden: eidola_engine_cpu::MtpHidden::Normed,
    }
    .run(tokens)
    .unwrap()
}

const TOKS: [u32; 9] = [17, 3, 250, 9, 41, 41, 0, 128, 77];

#[test]
fn zero_really_zeroes_every_byte_and_nothing_else() {
    let mut e = executor();
    let mut updates = map(&e, 0, 0, 1);
    updates.extend(map(&e, 0, 1, 2));
    e.execute(&step(&[prefill(0, &TOKS[..8])], vec![], updates))
        .unwrap();
    for g in 0..groups(&e) {
        assert!(
            !e.block_is_zero(g, 1) && !e.block_is_zero(g, 2),
            "group {g} written"
        );
        assert!(
            e.block_is_zero(g, 3),
            "group {g}: untouched block stays zero"
        );
    }
    let mut maintenance: Vec<Maintenance> = (0..groups(&e))
        .map(|group| Maintenance::Zero { group, block: 1 })
        .collect();
    maintenance.push(Maintenance::ResetSlot { slot: 0 });
    e.execute(&step(&[], maintenance, vec![])).unwrap();
    for g in 0..groups(&e) {
        assert!(
            e.block_is_zero(g, 1),
            "group {g}: zeroed block is all-zero bytes"
        );
        assert!(!e.block_is_zero(g, 2), "group {g}: other blocks untouched");
    }
    assert_eq!(e.zero_log().len(), groups(&e) as usize);
}

/// A block copied to a fresh slot carries the drafter's boundary tap with it: the fresh
/// slot continues exactly like the original one, in the same batch, and both equal the
/// dense oracle.
#[test]
fn copied_blocks_and_boundary_taps_resume_exactly() {
    let mut e = executor();
    e.execute(&step(&[prefill(0, &TOKS[..4])], vec![], map(&e, 0, 0, 1)))
        .unwrap();
    let copy: Vec<Maintenance> = (0..groups(&e))
        .map(|group| Maintenance::Copy {
            group,
            src: 1,
            dst: 3,
        })
        .collect();
    let mut updates = map(&e, 0, 1, 2);
    updates.extend(map(&e, 1, 0, 3));
    updates.extend(map(&e, 1, 1, 4));
    let out = e
        .execute(&step(
            &[decode(0, 4, TOKS[4], 2), decode(1, 4, TOKS[4], 2)],
            copy,
            updates,
        ))
        .unwrap();
    assert_eq!(out.row(0), out.row(1));
    assert_eq!(logits(&out, 0), logits(&out, 1));
    let recs = e.take_records();
    let (a, b) = (&recs[recs.len() - 2], &recs[recs.len() - 1]);
    assert_eq!(a.tokens, b.tokens);
    assert_eq!(a.drafter_logits.len(), b.drafter_logits.len());
    for (x, y) in a.drafter_logits.iter().zip(&b.drafter_logits) {
        assert_eq!(x, y);
    }
    let d = dense(&a.tokens);
    for (pos, l) in &a.target_logits {
        assert_eq!(l.as_slice(), d.target(*pos));
    }
    for (depth, slot, l) in &a.drafter_logits {
        assert_eq!(Some(l.as_slice()), d.drafter(*depth, *slot));
    }
}

/// A row's logits do not depend on what else is in the batch, which slot or blocks it
/// uses, or how its prefix was chunked.
#[test]
fn rows_are_independent_of_batch_slot_and_chunking() {
    let mut alone = executor();
    let mut updates = map(&alone, 0, 0, 1);
    updates.extend(map(&alone, 0, 1, 2));
    updates.extend(map(&alone, 0, 2, 3));
    alone
        .execute(&step(&[prefill(0, &TOKS[..8])], vec![], updates))
        .unwrap();
    let solo = alone
        .execute(&step(&[decode(0, 8, TOKS[8], 2)], vec![], vec![]))
        .unwrap();

    let mut mixed = executor();
    let mut updates = Vec::new();
    for (i, b) in [(0, 9), (1, 5), (2, 12)] {
        updates.extend(map(&mixed, 3, i, b));
    }
    updates.extend(map(&mixed, 1, 0, 1));
    updates.extend(map(&mixed, 1, 1, 2));
    mixed
        .execute(&step(
            &[prefill(1, &[5, 6, 7, 8, 9, 10]), prefill(3, &TOKS[..3])],
            vec![],
            updates,
        ))
        .unwrap();
    mixed
        .execute(&step(
            &[
                Row {
                    slot: 3,
                    context_len: 3,
                    tokens: TOKS[3..8].to_vec(),
                    sample: false,
                    drafts: 0,
                },
                decode(1, 6, 11, 0),
            ],
            vec![],
            vec![],
        ))
        .unwrap();
    let both = mixed
        .execute(&step(
            &[decode(1, 7, 12, 0), decode(3, 8, TOKS[8], 2)],
            vec![],
            vec![],
        ))
        .unwrap();
    assert_eq!(solo.row(0), both.row(1));
    assert_eq!(logits(&solo, 0), logits(&both, 1));
}

#[test]
#[should_panic(expected = "unmapped")]
fn reading_an_unmapped_block_panics() {
    let mut e = executor();
    e.execute(&step(&[prefill(0, &TOKS[..4])], vec![], map(&e, 0, 0, 1)))
        .unwrap();
    let mut updates = map(&e, 0, 0, NULL_BLOCK);
    updates.extend(map(&e, 0, 1, 2));
    e.execute(&step(&[decode(0, 4, TOKS[4], 0)], vec![], updates))
        .unwrap();
}

#[test]
#[should_panic(expected = "zeroed or foreign")]
fn reading_a_zeroed_block_panics() {
    let mut e = executor();
    e.execute(&step(&[prefill(0, &TOKS[..4])], vec![], map(&e, 0, 0, 1)))
        .unwrap();
    let zero: Vec<Maintenance> = (0..groups(&e))
        .map(|group| Maintenance::Zero { group, block: 1 })
        .collect();
    e.execute(&step(&[decode(0, 4, TOKS[4], 0)], zero, map(&e, 0, 1, 2)))
        .unwrap();
}

#[test]
#[should_panic(expected = "write to shared block")]
fn writing_a_shared_block_panics() {
    let mut e = executor();
    let mut updates = map(&e, 0, 0, 1);
    updates.extend(map(&e, 1, 0, 1));
    e.execute(&step(&[prefill(1, &TOKS[..2])], vec![], updates))
        .unwrap();
}

#[test]
#[should_panic(expected = "no drafter state")]
fn resuming_off_a_block_boundary_panics() {
    let mut e = executor();
    e.execute(&step(&[prefill(0, &TOKS[..3])], vec![], map(&e, 0, 0, 1)))
        .unwrap();
    // Slot 1 takes over block 1 but never computed position 2 itself, and position 2
    // ends no block: there is no drafter state to continue from.
    let mut updates = map(&e, 0, 0, NULL_BLOCK);
    updates.extend(map(&e, 1, 0, 1));
    e.execute(&step(&[decode(1, 3, TOKS[3], 0)], vec![], updates))
        .unwrap();
}

#[test]
#[should_panic(expected = "drafts requested")]
fn more_drafts_than_depths_panics() {
    let mut e = executor();
    e.execute(&step(&[decode(0, 0, 5, 0)], vec![], map(&e, 0, 0, 1)))
        .unwrap();
    e.execute(&step(&[decode(0, 1, 6, 3)], vec![], vec![]))
        .unwrap();
}
