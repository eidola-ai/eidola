//! The executor driven directly through the seam, without the scheduler: maintenance
//! semantics (zero really zeroes, copy really copies, encrypt and decrypt round-trip),
//! drafter state carried by boundary taps, batch-composition independence, and panics on
//! contract violations.

mod common;

use common::*;
use eidola_engine::executor::{
    Executor, Maintenance, SeqEntry, Slot, StepInput, StepOutput, TableUpdate,
};
use eidola_engine::kv_cipher::KvNonce;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::secret::KvKey;
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
    keyed_step(rows, maintenance, table_updates, Vec::new())
}

fn keyed_step(
    rows: &[Row],
    maintenance: Vec<Maintenance>,
    table_updates: Vec<TableUpdate>,
    kv_keys: Vec<KvKey>,
) -> StepInput {
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
        kv_keys,
        table_updates,
        seqs,
        token_ids,
        positions,
        return_logits: true,
        aux: Default::default(),
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
        sampleable_vocab_size: VOCAB,
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

/// `ResetSlot` overwrites the slot's per-sequence model state (the drafter's hidden
/// states) with zeros in place, and touches no other slot.
#[test]
fn reset_slot_scrubs_model_state_and_nothing_else() {
    let mut e = executor();
    for slot in 0..4 {
        assert!(e.slot_state_is_zero(slot), "slot {slot} starts zero");
    }
    let mut updates = map(&e, 0, 0, 1);
    updates.extend(map(&e, 0, 1, 2));
    updates.extend(map(&e, 1, 0, 3));
    updates.extend(map(&e, 1, 1, 4));
    e.execute(&step(
        &[prefill(0, &TOKS[..6]), prefill(1, &TOKS[1..7])],
        vec![],
        updates,
    ))
    .unwrap();
    assert!(!e.slot_state_is_zero(0) && !e.slot_state_is_zero(1));
    e.execute(&step(&[], vec![Maintenance::ResetSlot { slot: 0 }], vec![]))
        .unwrap();
    assert!(e.slot_state_is_zero(0), "reset slot holds only zeros");
    assert!(!e.slot_state_is_zero(1), "other slots untouched");
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

/// `Encrypt` or `Decrypt` of `block` in every group, under key index 0, with a nonce per
/// group.
fn crypt_all(e: &CpuExecutor, block: u32, encrypt: bool) -> Vec<Maintenance> {
    (0..groups(e))
        .map(|group| {
            let nonce = KvNonce::from_words([group, block, 7]);
            if encrypt {
                Maintenance::Encrypt {
                    group,
                    block,
                    key: 0,
                    nonce,
                }
            } else {
                Maintenance::Decrypt {
                    group,
                    block,
                    key: 0,
                    nonce,
                }
            }
        })
        .collect()
}

fn key(n: u8) -> KvKey {
    KvKey::from_bytes([n; 32])
}

/// Encrypting a block changes essentially every stored word of it (KV, the drafter's tap,
/// the row tags) and nothing else; another key gives other ciphertext; decrypting
/// restores every bit, and the sequence then continues exactly as the dense oracle does.
#[test]
fn encrypt_then_decrypt_restores_every_bit_and_touches_nothing_else() {
    let filled = || {
        let mut e = executor();
        let mut updates = map(&e, 0, 0, 1);
        updates.extend(map(&e, 0, 1, 2));
        e.execute(&step(&[prefill(0, &TOKS[..8])], vec![], updates))
            .unwrap();
        e
    };
    let mut e = filled();
    let plain: Vec<(Vec<u32>, Vec<u32>)> = (0..groups(&e))
        .map(|g| (e.block_image(g, 1), e.block_image(g, 2)))
        .collect();
    let enc = crypt_all(&e, 1, true);
    e.execute(&keyed_step(&[], enc.clone(), vec![], vec![key(1)]))
        .unwrap();
    let mut other = filled();
    other
        .execute(&keyed_step(&[], enc, vec![], vec![key(2)]))
        .unwrap();
    for (g, (p1, p2)) in plain.iter().enumerate() {
        let g = g as u32;
        let c = e.block_image(g, 1);
        assert!(e.block_is_encrypted(g, 1));
        let same = c.iter().zip(p1).filter(|(a, b)| a == b).count();
        assert!(
            same * 1000 < c.len(),
            "group {g}: {same} of {} words unchanged",
            c.len()
        );
        assert_eq!(
            &e.block_image(g, 2),
            p2,
            "group {g}: other blocks untouched"
        );
        let c2 = other.block_image(g, 1);
        let same = c.iter().zip(&c2).filter(|(a, b)| a == b).count();
        assert!(
            same * 1000 < c.len(),
            "group {g}: keys give distinct ciphertext"
        );
    }
    e.execute(&keyed_step(
        &[],
        crypt_all(&e, 1, false),
        vec![],
        vec![key(1)],
    ))
    .unwrap();
    for (g, (p1, _)) in plain.iter().enumerate() {
        assert_eq!(&e.block_image(g as u32, 1), p1, "group {g}: bit-identical");
        assert!(!e.block_is_encrypted(g as u32, 1));
    }
    let out = e
        .execute(&step(&[decode(0, 8, TOKS[8], 2)], vec![], map(&e, 0, 2, 3)))
        .unwrap();
    let rec = e.take_records().pop().unwrap();
    let d = dense(&rec.tokens);
    for (pos, l) in &rec.target_logits {
        assert_eq!(l.as_slice(), d.target(*pos));
    }
    for (depth, slot, l) in &rec.drafter_logits {
        assert_eq!(Some(l.as_slice()), d.drafter(*depth, *slot));
    }
    assert!(!out.row(0).is_empty());
}

/// Zeroing an encrypted block leaves only zeros, and the block is plaintext again.
#[test]
fn zeroing_an_encrypted_block_scrubs_it() {
    let mut e = executor();
    e.execute(&step(&[prefill(0, &TOKS[..4])], vec![], map(&e, 0, 0, 1)))
        .unwrap();
    let mut m = crypt_all(&e, 1, true);
    m.extend((0..groups(&e)).map(|group| Maintenance::Zero { group, block: 1 }));
    e.execute(&keyed_step(&[], m, vec![], vec![key(1)]))
        .unwrap();
    for g in 0..groups(&e) {
        assert!(e.block_is_zero(g, 1) && !e.block_is_encrypted(g, 1));
    }
}

/// A read of a block left encrypted fails its tag check, like a read of a zeroed one.
#[test]
#[should_panic(expected = "zeroed or foreign")]
fn reading_an_encrypted_block_panics() {
    let mut e = executor();
    e.execute(&step(&[prefill(0, &TOKS[..4])], vec![], map(&e, 0, 0, 1)))
        .unwrap();
    let enc = crypt_all(&e, 1, true);
    e.execute(&keyed_step(
        &[decode(0, 4, TOKS[4], 0)],
        enc,
        map(&e, 0, 1, 2),
        vec![key(1)],
    ))
    .unwrap();
}

#[test]
#[should_panic(expected = "decrypted twice")]
fn decrypting_a_plaintext_block_panics() {
    let mut e = executor();
    e.execute(&step(&[prefill(0, &TOKS[..4])], vec![], map(&e, 0, 0, 1)))
        .unwrap();
    e.execute(&keyed_step(
        &[],
        crypt_all(&e, 1, false),
        vec![],
        vec![key(1)],
    ))
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

#[test]
#[should_panic(expected = "drafts requested at position 0")]
fn drafts_at_position_zero_panic() {
    let mut e = executor();
    e.execute(&step(&[decode(0, 0, 5, 1)], vec![], map(&e, 0, 0, 1)))
        .unwrap();
}

/// Eight host blocks (about 0.4 MB locked); the soft memory-lock limit raised first.
fn tiered_executor() -> CpuExecutor {
    allow_memlock();
    let mut cfg = exec_config(16, vec![0, 1]);
    cfg.num_state_slots = 4;
    cfg.host_blocks = 8;
    CpuExecutor::new(fixture(), cfg)
}

/// `op(group)` in every group.
fn all_groups(e: &CpuExecutor, op: impl Fn(u32) -> Maintenance) -> Vec<Maintenance> {
    (0..groups(e)).map(op).collect()
}

/// Demote then promote moves a block's ciphertext out and back word for word (KV, the
/// drafter's tap and the row tags), in every group; zeroing the device block between them
/// and then the host block leaves zeros; decrypting the promoted block restores the
/// plaintext bit for bit, and the sequence then continues exactly as the dense oracle.
#[test]
fn demote_and_promote_round_trip_the_ciphertext_and_decrypt_restores_it() {
    let mut e = tiered_executor();
    let mut updates = map(&e, 0, 0, 1);
    updates.extend(map(&e, 0, 1, 2));
    e.execute(&step(&[prefill(0, &TOKS[..8])], vec![], updates))
        .unwrap();
    let plain: Vec<Vec<u32>> = (0..groups(&e)).map(|g| e.block_image(g, 1)).collect();
    e.execute(&keyed_step(
        &[],
        crypt_all(&e, 1, true),
        vec![],
        vec![key(1)],
    ))
    .unwrap();
    let cipher: Vec<Vec<u32>> = (0..groups(&e)).map(|g| e.block_image(g, 1)).collect();
    // Demote block 1 to host block 3, and zero it (the order the core queues them in).
    let mut m = all_groups(&e, |group| Maintenance::Demote {
        group,
        block: 1,
        host: 3,
    });
    m.extend(all_groups(&e, |group| Maintenance::Zero {
        group,
        block: 1,
    }));
    e.execute(&step(&[], m, vec![])).unwrap();
    for g in 0..groups(&e) {
        assert_eq!(
            e.host_block_image(g, 3),
            cipher[g as usize],
            "group {g}: copied"
        );
        assert!(e.host_block_is_encrypted(g, 3));
        assert!(
            e.block_is_zero(g, 1),
            "group {g}: the device block is zeroed"
        );
        let same = e
            .host_block_image(g, 3)
            .iter()
            .zip(&plain[g as usize])
            .filter(|(a, b)| a == b)
            .count();
        assert!(
            same * 1000 < plain[g as usize].len(),
            "group {g}: plaintext on the host"
        );
    }
    // Promote it into block 1 again, decrypt, zero the host block.
    let mut m = all_groups(&e, |group| Maintenance::Promote {
        group,
        host: 3,
        block: 1,
    });
    m.extend(crypt_all(&e, 1, false));
    m.extend(all_groups(&e, |group| Maintenance::ZeroHost {
        group,
        host: 3,
    }));
    e.execute(&keyed_step(&[], m, vec![], vec![key(1)]))
        .unwrap();
    for g in 0..groups(&e) {
        assert_eq!(
            e.block_image(g, 1),
            plain[g as usize],
            "group {g}: bit-identical"
        );
        assert!(e.host_block_is_zero(g, 3) && !e.host_block_is_encrypted(g, 3));
    }
    e.execute(&step(&[decode(0, 8, TOKS[8], 2)], vec![], map(&e, 0, 2, 4)))
        .unwrap();
    let rec = e.take_records().pop().unwrap();
    let d = dense(&rec.tokens);
    for (pos, l) in &rec.target_logits {
        assert_eq!(l.as_slice(), d.target(*pos));
    }
    for (depth, slot, l) in &rec.drafter_logits {
        assert_eq!(Some(l.as_slice()), d.drafter(*depth, *slot));
    }
}

/// Demoting a block that is not encrypted is a host bug the executor refuses.
#[test]
#[should_panic(expected = "demote of a plaintext block")]
fn demoting_a_plaintext_block_panics() {
    let mut e = tiered_executor();
    e.execute(&step(&[prefill(0, &TOKS[..4])], vec![], map(&e, 0, 0, 1)))
        .unwrap();
    let m = all_groups(&e, |group| Maintenance::Demote {
        group,
        block: 1,
        host: 1,
    });
    let _ = e.execute(&step(&[], m, vec![]));
}

/// Promoting into a block that still holds data (its zero not run first) is refused.
#[test]
#[should_panic(expected = "promote into a block that is not zero")]
fn promoting_into_a_block_with_data_panics() {
    let mut e = tiered_executor();
    e.execute(&step(&[prefill(0, &TOKS[..4])], vec![], map(&e, 0, 0, 1)))
        .unwrap();
    let mut m = crypt_all(&e, 1, true);
    m.extend(all_groups(&e, |group| Maintenance::Demote {
        group,
        block: 1,
        host: 2,
    }));
    e.execute(&keyed_step(&[], m, vec![], vec![key(1)]))
        .unwrap();
    let m = all_groups(&e, |group| Maintenance::Promote {
        group,
        host: 2,
        block: 1,
    });
    let _ = e.execute(&step(&[], m, vec![]));
}
