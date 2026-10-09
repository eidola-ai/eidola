//! The CUDA executor driven directly through the seam on the truncated real
//! checkpoint (see `real_flash.rs` for the environment it needs): zero really
//! zeroes, a copied block resumes bit for bit in a fresh slot, a row's logits
//! are bit for bit independent of its batch and its chunking, the sampler
//! never returns a padded id, contract violations panic, and a malformed
//! step changes nothing.

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use common::setup;
use eidola_engine::executor::{Executor, Maintenance, SeqEntry, StepInput, TableUpdate};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::Bucket;
use eidola_engine_cuda::attention::{AttnRequest, Reduction};
use eidola_engine_cuda::model::ForwardInput;
use eidola_engine_cuda::{CudaExecutor, CudaExecutorConfig, CudaGraphs, KvBlocks, MtpHidden};
use eidola_engine_model::safetensors::WeightSet;

const KEEP: [usize; 4] = [0, 1, 2, 5];
const BS: u32 = 16;
const SAMPLEABLE: u32 = 151_675;
const VOCAB: usize = 152_576;
const VOCAB32: u32 = 152_576;

fn row(slot: u32, start: u32, context: u32, n: u32, sample: bool) -> SeqEntry {
    SeqEntry {
        slot,
        token_start: start,
        num_tokens: n,
        context_len: context,
        num_drafts: 0,
        sample,
        sampling: SamplingParams::greedy(),
    }
}

fn step(
    seqs: Vec<SeqEntry>,
    tokens: Vec<u32>,
    m: Vec<Maintenance>,
    t: Vec<TableUpdate>,
) -> StepInput {
    let positions = seqs
        .iter()
        .flat_map(|s| s.context_len..s.context_len + s.num_tokens)
        .collect();
    StepInput {
        bucket: Bucket {
            max_seqs: 4,
            max_tokens: 256,
        },
        maintenance: m,
        table_updates: t,
        seqs,
        token_ids: tokens,
        positions,
        return_logits: true,
    }
}

/// `table[slot][g][first + i] = blocks[i]` in both groups.
fn map(slot: u32, first: u32, blocks: &[u32]) -> Vec<TableUpdate> {
    (0..2)
        .flat_map(|group| {
            blocks
                .iter()
                .enumerate()
                .map(move |(i, &block)| TableUpdate {
                    slot,
                    group,
                    index: first + u32::try_from(i).unwrap(),
                    block,
                })
        })
        .collect()
}

fn panics(f: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err()
}

#[test]
fn seam_contract_on_the_truncated_checkpoint() {
    let Some(dir) = std::env::var_os("EIDOLA_MIMO_DIR").map(PathBuf::from) else {
        eprintln!("skipping: EIDOLA_MIMO_DIR not set");
        return;
    };
    let Some(su) = setup() else { return };
    let store = Arc::new(WeightSet::open_dir(&dir).unwrap());
    let cfg = CudaExecutorConfig {
        block_size: BS,
        num_blocks: KvBlocks {
            global: 32,
            sliding: 32,
            drafter: 0,
        },
        num_state_slots: 4,
        max_model_len: 1024,
        buckets: vec![Bucket {
            max_seqs: 4,
            max_tokens: 256,
        }],
        sampleable_vocab_size: SAMPLEABLE,
        image: None,
        graphs: CudaGraphs::Off,
        draft_tokens: 0,
        mtp_hidden: MtpHidden::Normed,
    };
    let mut ex = CudaExecutor::new(su.gpu, &su.dir, store, Some(&KEEP), cfg).unwrap();
    let prompt: Vec<u32> = (0..40u32).map(|i| (i * 7919 + 13) % 150_000).collect();

    // Prefill 40 tokens in slot 0 (blocks 1, 2, 3).
    let out = ex
        .execute(&step(
            vec![row(0, 0, 0, 40, true)],
            prompt.clone(),
            vec![],
            map(0, 0, &[1, 2, 3]),
        ))
        .unwrap();
    assert!(out.row(0)[0] < SAMPLEABLE);
    for g in 0..2 {
        for b in 1..=3 {
            assert!(
                !ex.block_is_zero(g, b).unwrap(),
                "group {g} block {b} written"
            );
        }
        assert!(ex.block_is_zero(g, 4).unwrap());
    }

    // Copy blocks 1 and 2 (positions 0..32) to 5 and 6, then zero block 3:
    // the copies are exact, the zero touches only its block.
    let before: Vec<Vec<u16>> = (0..2)
        .map(|g| ex.kv().read_block(ex.gpu(), g, 1).unwrap())
        .collect();
    let copies = (0..2)
        .flat_map(|group| {
            [
                Maintenance::Copy {
                    group,
                    src: 1,
                    dst: 5,
                },
                Maintenance::Copy {
                    group,
                    src: 2,
                    dst: 6,
                },
            ]
        })
        .chain((0..2).map(|group| Maintenance::Zero { group, block: 3 }))
        .collect();
    // Resume in slot 1 at the block boundary 32 from the copies, and keep
    // going in slot 0 is no longer possible (its block 3 is gone): instead
    // re-run positions 32..40 in slot 1 and compare with slot 0's prefill.
    let resumed = ex
        .execute(&step(
            vec![row(1, 0, 32, 8, true)],
            prompt[32..].to_vec(),
            copies,
            [map(1, 0, &[5, 6]), map(1, 2, &[7])].concat(),
        ))
        .unwrap();
    for g in 0..2 {
        assert!(ex.block_is_zero(g, 3).unwrap(), "zero really zeroes");
        assert_eq!(
            ex.kv().read_block(ex.gpu(), g as usize, 5).unwrap(),
            before[g as usize],
            "copy really copies"
        );
        assert!(
            !ex.block_is_zero(g, 2).unwrap(),
            "zero touches only its block"
        );
    }
    let a = &out.logits.as_ref().unwrap()[0][0];
    let b = &resumed.logits.as_ref().unwrap()[0][0];
    // One 40-token prefill against a resume at 32 from copied blocks: the
    // same bits at position 39.
    assert!(
        a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
        "logits at position 39 differ between a prefill and a resume"
    );
    assert_eq!(out.row(0), resumed.row(0), "the same greedy token");

    // Row independence: decode slot 1 alone, then together with another row.
    let alone = ex
        .execute(&step(
            vec![row(1, 0, 40, 1, true)],
            vec![17],
            vec![],
            vec![],
        ))
        .unwrap();
    // Undo by re-running the same position (it overwrites the same KV row).
    let together = ex
        .execute(&step(
            vec![row(1, 0, 40, 1, true), row(2, 1, 0, 30, true)],
            [vec![17u32], prompt[..30].to_vec()].concat(),
            vec![],
            map(2, 0, &[8, 9]),
        ))
        .unwrap();
    let a = &alone.logits.as_ref().unwrap()[0][0];
    let b = &together.logits.as_ref().unwrap()[0][0];
    // A decode row alone and beside a 30-token prefill: the same bits.
    assert!(
        a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
        "a decode row's logits move with its batch"
    );
    assert_eq!(alone.row(0), together.row(0));
    assert_eq!(a.len(), VOCAB);

    // ResetSlot empties the tables: a read through slot 1 now panics.
    ex.execute(&step(
        vec![],
        vec![],
        vec![Maintenance::ResetSlot { slot: 1 }],
        vec![],
    ))
    .unwrap();
    assert!(
        panics(|| {
            let _ = ex.execute(&step(vec![row(1, 0, 41, 1, true)], vec![5], vec![], vec![]));
        }),
        "read of an unmapped block"
    );
    // A write into a block mapped twice (shared) panics.
    assert!(
        panics(|| {
            let _ = ex.execute(&step(
                vec![row(3, 0, 0, 4, true)],
                vec![1, 2, 3, 4],
                vec![],
                map(3, 0, &[8]),
            ));
        }),
        "write to a shared block"
    );
    // A malformed step changes nothing: its maintenance and table updates
    // are validated with its rows and applied only once all of it passes.
    let tables_before: Vec<Vec<u32>> = (0..2).map(|g| ex.kv().table(3, g).to_vec()).collect();
    let device_before: Vec<Vec<i32>> = (0..2)
        .map(|g| ex.kv().read_table_row(ex.gpu(), 3, g).unwrap())
        .collect();
    let block_before = ex.kv().read_block(ex.gpu(), 0, 2).unwrap();
    let malformed = || {
        step(
            vec![row(3, 0, 0, 4, true)],
            vec![1, 2, 3, 4],
            vec![Maintenance::Zero { group: 0, block: 2 }],
            map(3, 0, &[10]),
        )
    };
    let mut bad_positions = malformed();
    bad_positions.positions[2] = 99;
    assert!(
        panics(|| {
            let _ = ex.execute(&bad_positions);
        }),
        "positions disagree"
    );
    let mut bad_token = malformed();
    bad_token.token_ids[1] = VOCAB32;
    assert!(
        ex.execute(&bad_token).is_err(),
        "token outside the vocabulary"
    );
    // A bucket outside the configured ladder, even a larger one.
    let mut bad_bucket = malformed();
    bad_bucket.bucket = Bucket {
        max_seqs: 8,
        max_tokens: 256,
    };
    let e = ex.execute(&bad_bucket).unwrap_err();
    assert!(e.to_string().contains("not one of the configured"), "{e}");
    for g in 0..2 {
        assert_eq!(
            ex.kv().table(3, g),
            &tables_before[g][..],
            "mirror untouched"
        );
        assert_eq!(
            ex.kv().read_table_row(ex.gpu(), 3, g).unwrap(),
            device_before[g],
            "device table untouched"
        );
    }
    assert_eq!(
        ex.kv().read_block(ex.gpu(), 0, 2).unwrap(),
        block_before,
        "maintenance not applied"
    );

    // The model's own entry checks every index its kernels would follow.
    let plans: Vec<_> = (0..2)
        .map(|g| {
            ex.model()
                .kernels()
                .attention
                .plan(
                    ex.gpu(),
                    &[AttnRequest {
                        q_start: 0,
                        qo_len: 1,
                        pages: vec![11],
                        kv_start: 0,
                        kv_len: 1,
                    }],
                    [16, 8][g],
                    BS,
                    [Reduction::Split, Reduction::Anchored { origin: 0 }][g],
                    1,
                )
                .unwrap()
        })
        .collect();
    let targets = vec![vec![(11u32, 0u32)]; 2];
    let input = |tokens: &'static [u32], positions: &'static [u32], logit_rows: &'static [u32]| {
        (tokens, positions, logit_rows)
    };
    let check = |(tokens, positions, logit_rows): (&[u32], &[u32], &[u32]),
                 targets: &[Vec<(u32, u32)>]| {
        ex.model().check_input(
            ex.kv(),
            &ForwardInput {
                tokens,
                positions,
                kv_targets: targets,
                plans: &plans,
                logit_rows,
            },
        )
    };
    check(input(&[5], &[0], &[0]), &targets).unwrap();
    for (bad, why) in [
        (input(&[5], &[0], &[u32::MAX]), "logit row"),
        (input(&[5], &[0], &[1]), "logit row"),
        (input(&[5], &[1 << 20], &[0]), "position"),
        (input(&[VOCAB32], &[0], &[0]), "vocabulary"),
        (input(&[5, 6], &[0, 1], &[0]), "KV targets"),
    ] {
        let e = check(bad, &targets).unwrap_err();
        assert!(e.to_string().contains(why), "{why}: {e}");
    }
    for bad in [(0, 0), (32, 0), (11, BS)] {
        let e = check(input(&[5], &[0], &[0]), &[vec![bad], vec![(11, 0)]]).unwrap_err();
        assert!(e.to_string().contains("KV target"), "{bad:?}: {e}");
    }
    // Drafts are refused (this executor drafts none), at position 0 above all.
    let mut r = row(0, 0, 0, 1, true);
    r.num_drafts = 1;
    assert!(
        panics(|| {
            let _ = ex.execute(&step(vec![r], vec![1], vec![], vec![]));
        }),
        "drafts at position 0"
    );
}
