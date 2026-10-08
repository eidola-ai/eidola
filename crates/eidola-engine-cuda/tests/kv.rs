//! Device KV maintenance through the seam's operations: zero really zeroes
//! (that block, every layer, nothing else), copy really copies, `ResetSlot`
//! empties the slot's tables and scrubs its state (and only its), and table
//! updates land on the device.

mod common;

use common::{Lcg, setup};
use eidola_engine::executor::{Maintenance, TableUpdate};
use eidola_engine_cuda::kv::{GroupGeometry, KvStore};

fn geometry() -> Vec<GroupGeometry> {
    vec![
        GroupGeometry {
            num_layers: 3,
            num_kv_heads: 4,
            head_dim_qk: 192,
            head_dim_v: 128,
            block_size: 16,
            num_blocks: 6,
        },
        GroupGeometry {
            num_layers: 5,
            num_kv_heads: 8,
            head_dim_qk: 192,
            head_dim_v: 128,
            block_size: 16,
            num_blocks: 5,
        },
    ]
}

fn filled(su: &common::Setup, store: &mut KvStore, rng: &mut Lcg) -> Vec<Vec<Vec<u16>>> {
    let mut all = Vec::new();
    for g in 0..store.geometry().len() {
        let mut blocks = Vec::new();
        for b in 0..store.geometry()[g].num_blocks {
            let data: Vec<u16> = (0..store.geometry()[g].block_elems())
                .map(|_| {
                    let r = rng.next_u64().to_le_bytes();
                    u16::from_le_bytes([r[0], r[1]]) | 1
                })
                .collect();
            store.write_block(&su.gpu, g, b, &data).unwrap();
            blocks.push(data);
        }
        all.push(blocks);
    }
    all
}

#[test]
fn zero_and_copy_touch_exactly_their_blocks() {
    let Some(su) = setup() else { return };
    let mut store = KvStore::new(&su.gpu, geometry(), 3, 8, 0).unwrap();
    let mut rng = Lcg(3);
    let mut want = filled(&su, &mut store, &mut rng);
    store
        .apply(
            &su.gpu,
            &[
                Maintenance::Zero { group: 1, block: 2 },
                Maintenance::Copy {
                    group: 0,
                    src: 4,
                    dst: 1,
                },
                // Order matters: copy a block, then zero its source.
                Maintenance::Copy {
                    group: 1,
                    src: 3,
                    dst: 4,
                },
                Maintenance::Zero { group: 1, block: 3 },
            ],
            &[],
        )
        .unwrap();
    want[1][2].fill(0);
    want[0][1] = want[0][4].clone();
    want[1][4] = want[1][3].clone();
    want[1][3].fill(0);
    for (g, blocks) in want.iter().enumerate() {
        for (b, data) in blocks.iter().enumerate() {
            assert!(
                store
                    .read_block(&su.gpu, g, u32::try_from(b).unwrap())
                    .unwrap()
                    == *data,
                "group {g} block {b}"
            );
        }
    }
}

#[test]
fn reset_slot_empties_tables_and_scrubs_only_that_slot() {
    let Some(su) = setup() else { return };
    let width = 7;
    let mut store = KvStore::new(&su.gpu, geometry(), 3, 8, width).unwrap();
    let updates: Vec<TableUpdate> = (0..3)
        .flat_map(|slot| {
            (0..2).map(move |group| TableUpdate {
                slot,
                group,
                index: slot,
                block: 1 + slot,
            })
        })
        .collect();
    store.apply(&su.gpu, &[], &updates).unwrap();
    for slot in 0..3 {
        store
            .write_state(&su.gpu, slot, &vec![1.5 + slot as f32; width])
            .unwrap();
        for g in 0..2 {
            let row = store.read_table_row(&su.gpu, slot, g).unwrap();
            let mut want = vec![0i32; 8];
            want[slot as usize] = 1 + i32::try_from(slot).unwrap();
            assert_eq!(row, want);
        }
    }
    // Reset slot 1, then map its index 5 again in group 0 in the same step.
    store
        .apply(
            &su.gpu,
            &[Maintenance::ResetSlot { slot: 1 }],
            &[TableUpdate {
                slot: 1,
                group: 0,
                index: 5,
                block: 4,
            }],
        )
        .unwrap();
    assert!(
        store
            .read_state(&su.gpu, 1)
            .unwrap()
            .iter()
            .all(|x| x.to_bits() == 0)
    );
    assert_eq!(store.read_state(&su.gpu, 0).unwrap(), vec![1.5; width]);
    assert_eq!(store.read_state(&su.gpu, 2).unwrap(), vec![3.5; width]);
    let mut want = vec![0i32; 8];
    want[5] = 4;
    assert_eq!(store.read_table_row(&su.gpu, 1, 0).unwrap(), want);
    assert_eq!(store.read_table_row(&su.gpu, 1, 1).unwrap(), vec![0; 8]);
    assert_eq!(store.read_table_row(&su.gpu, 2, 1).unwrap()[2], 3);
    assert_eq!(store.table(1, 0)[5], 4);
}

#[test]
fn shared_and_unmapped_blocks_are_refused() {
    let Some(su) = setup() else { return };
    let mut store = KvStore::new(&su.gpu, geometry(), 2, 8, 0).unwrap();
    let share = |slot| TableUpdate {
        slot,
        group: 0,
        index: 0,
        block: 3,
    };
    store.apply(&su.gpu, &[], &[share(0), share(1)]).unwrap();
    assert_eq!(store.block_of(0, 0, 5), 3);
    let shared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        store.writable_block(0, 0, 5)
    }));
    assert!(shared.is_err(), "a write into a shared block must panic");
    let unmapped =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| store.block_of(0, 0, 16)));
    assert!(
        unmapped.is_err(),
        "a read through an unmapped entry must panic"
    );
    store
        .apply(&su.gpu, &[Maintenance::ResetSlot { slot: 1 }], &[])
        .unwrap();
    assert_eq!(store.writable_block(0, 0, 5), 3);
}
