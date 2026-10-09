//! `eidola_copy_rows`, the drafted step's gathers and scatters, against its
//! reference form: the same copies done on the host, word for word (a copy
//! has no arithmetic to round, so the reference is the host gather itself).
//! On every image this device runs: rows of one word (token ids) and of a
//! hidden state's 4,096, items across every buffer slot, the grid's second
//! dimension for wide rows; an item naming a buffer slot or a row out of
//! range copies nothing and raises `STATUS_BAD_INDEX`, the others still
//! land; and the launch geometry is the one `copy_grid` gives.

mod common;

use common::{Lcg, setup};
use eidola_engine_cuda::engine_ops::{
    COPY_BUFFERS, COPY_THREADS, CopyArgs, EngineOps, STATUS_BAD_INDEX, copy_grid,
};
use eidola_engine_cuda::launch::dptr;

/// One item: source buffer and row, destination buffer and row.
type Item = (usize, usize, usize, usize);

/// A case's host side: the buffers' contents, the item list (`items` valid
/// items, then two out of range) and the buffers the valid items leave.
/// Needs no device, so the run without one builds every case
/// (`cases_are_built_without_a_device`).
fn plan(width: usize, rows: [usize; 4], items: usize, seed: u64) -> Plan {
    let mut rng = Lcg(seed);
    let host: Vec<Vec<u32>> = rows
        .iter()
        .map(|&r| {
            (0..r * width)
                .map(|_| u32::try_from(rng.next_u64() >> 32).unwrap())
                .collect()
        })
        .collect();
    // Items: distinct destinations, never another item's source.
    let mut dsts = std::collections::HashSet::new();
    let mut srcs = Vec::new();
    let mut list = Vec::new();
    let mut draws = 0u32;
    while list.len() < items {
        draws += 1;
        assert!(draws < 1_000_000, "no item list of {items} found");
        let below = |rng: &mut Lcg, n: usize| usize::try_from(rng.below(n as u64)).unwrap();
        let (sb, db) = (below(&mut rng, 4), below(&mut rng, 4));
        let (sr, dr) = (below(&mut rng, rows[sb]), below(&mut rng, rows[db]));
        if dsts.contains(&(sb, sr)) || srcs.contains(&(db, dr)) || !dsts.insert((db, dr)) {
            continue;
        }
        srcs.push((sb, sr));
        list.push((sb, sr, db, dr));
    }
    // One item past its buffer's rows and one naming an unused slot: refused.
    list.push((0, rows[0], 1, 0));
    list.push((COPY_BUFFERS, 0, 2, 0));
    // The reference: every in-range item, in any order (none overlaps).
    let mut want = host.clone();
    for &(sb, sr, db, dr) in &list[..items] {
        let row = host[sb][sr * width..(sr + 1) * width].to_vec();
        want[db][dr * width..(dr + 1) * width].copy_from_slice(&row);
    }
    Plan { host, list, want }
}

struct Plan {
    host: Vec<Vec<u32>>,
    list: Vec<Item>,
    want: Vec<Vec<u32>>,
}

/// Buffers of `rows[b]` rows each; items copy row to row among them.
fn case(width: usize, rows: [usize; 4], items: usize, seed: u64) {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    let Plan {
        mut host,
        list,
        want,
    } = plan(width, rows, items, seed);
    let col = |f: &dyn Fn(&Item) -> usize| -> Vec<u32> {
        list.iter().map(|i| u32::try_from(f(i)).unwrap()).collect()
    };
    let arrays: Vec<_> = [col(&|i| i.0), col(&|i| i.1), col(&|i| i.2), col(&|i| i.3)]
        .iter()
        .map(|v| s.clone_htod(v).unwrap())
        .collect();
    for &arch in &su.archs {
        let ops = EngineOps::from_module(su.module("engine_ops", arch)).unwrap();
        let bufs: Vec<_> = host.iter().map(|b| s.clone_htod(b).unwrap()).collect();
        let status = s.alloc_zeros::<u32>(1).unwrap();
        let mut args = CopyArgs {
            src_buf: dptr(&arrays[0], s),
            src_row: dptr(&arrays[1], s),
            dst_buf: dptr(&arrays[2], s),
            dst_row: dptr(&arrays[3], s),
            status: dptr(&status, s),
            width: u32::try_from(width).unwrap(),
            items: u32::try_from(list.len()).unwrap(),
            ..CopyArgs::default()
        };
        for (b, buf) in bufs.iter().enumerate() {
            args.buf[b] = dptr(buf, s);
            args.rows[b] = rows[b] as u64;
        }
        // SAFETY: the item arrays hold `items` words each; every buffer's
        // rows are its own.
        unsafe { ops.copy_rows(gpu, args).unwrap() };
        for (b, buf) in bufs.iter().enumerate() {
            assert_eq!(
                s.clone_dtoh(buf).unwrap(),
                want[b],
                "{arch:?} width {width}: buffer {b}"
            );
        }
        assert_eq!(
            s.clone_dtoh(&status).unwrap(),
            vec![STATUS_BAD_INDEX],
            "{arch:?}"
        );
    }
    host.clear();
}

/// Every case: width, rows per buffer, valid items, seed.
const CASES: [(usize, [usize; 4], usize, u64); 3] = [
    (1, [64, 9, 300, 17], 120, 1),
    (4096, [40, 12, 33, 8], 50, 2),
    // More than one block of threads per row, the last partial.
    (COPY_THREADS as usize * 3 + 5, [7, 7, 7, 7], 12, 3),
];

#[test]
fn token_rows() {
    let (w, r, i, s) = CASES[0];
    case(w, r, i, s);
}

#[test]
fn hidden_rows() {
    let (w, r, i, s) = CASES[1];
    case(w, r, i, s);
}

#[test]
fn uneven_wide_rows() {
    let (w, r, i, s) = CASES[2];
    case(w, r, i, s);
}

/// The cases' host side runs without a device, so a setup that cannot be
/// built (an item list that cannot be drawn) fails here rather than on the
/// GPU host.
#[test]
fn cases_are_built_without_a_device() {
    for (w, r, i, s) in CASES {
        let p = plan(w, r, i, s);
        assert_eq!(p.list.len(), i + 2);
    }
}

/// One block row per item; the row's words over blocks of `COPY_THREADS`.
#[test]
fn the_grid_covers_every_word_once() {
    assert_eq!(copy_grid(5, 1).unwrap(), [5, 1, 1]);
    assert_eq!(copy_grid(5, 4096).unwrap(), [5, 16, 1]);
    assert_eq!(copy_grid(3, 257).unwrap(), [3, 2, 1]);
    assert_eq!(copy_grid(0, 4096).unwrap(), [0, 16, 1]);
    assert!(copy_grid(1, 0).is_err());
    assert!(copy_grid(1, 65_536 * COPY_THREADS).is_err());
}
