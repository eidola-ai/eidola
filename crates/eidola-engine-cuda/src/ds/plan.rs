//! The step planner: pure functions from a step's rows to the order, counts and index
//! lists its launches take. Nothing here touches a device.
//!
//! **Row order.** The kernels compute rows `0 .. rows` of their arrays, so the positions
//! a layer computes must be a prefix. A position's *depth* is the number of chain layers
//! that compute it (`x >= chain_layer_start(chain_start, j)`, which shrinks as `j`
//! grows), and the positions the head reads are among those of full depth. Ordering by
//! (head first, depth descending, then the reference executor's order: rows in step
//! order, positions ascending) makes every set a prefix: the head's rows, each chain
//! layer's rows, the boundary rows (the bottom layer's), and every position (the
//! encoder's rows, and the bank entries that cross). Both halves use the one order, so a
//! row index means the same position on each.

use eidola_engine::executor::Slot;

use super::abi::SLOT_EMPTY;

/// One row of a step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanRow {
    pub slot: Slot,
    /// The first host token's position.
    pub c: u32,
    /// The last host token's position.
    pub p: u32,
    pub sample: bool,
    /// Where the chain's top layer starts.
    pub chain_start: u32,
}

/// One position of a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Item {
    /// Index of its row in the step.
    pub row: u32,
    pub slot: Slot,
    pub pos: u32,
    /// Chain layers that compute it: layers `0 .. depth`, counted from the bottom.
    pub depth: u32,
    /// The head reads it.
    pub top: bool,
    /// Its index in the reference executor's order.
    pub index: u32,
}

/// A step's positions in device row order and the row counts of every launch group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepPlan {
    pub items: Vec<Item>,
    /// Per chain layer (0 = bottom): the rows it computes, a prefix.
    pub chain_rows: Vec<u32>,
    /// Rows the head reads, a prefix.
    pub top_rows: u32,
    /// Per step row: the device row of its last host position.
    pub last_row: Vec<u32>,
}

/// The first position chain layer `j` of `depth` computes in a row whose top layer
/// starts at `chain_start` (`ModelSpec::chain_layer_start`).
pub fn chain_layer_start(chain_start: u32, depth: u32, window: u32, j: u32) -> u32 {
    chain_start.saturating_sub((depth - 1 - j) * (window - 1))
}

impl StepPlan {
    /// The plan of `rows` under a chain of `depth` layers of `window`.
    pub fn new(rows: &[PlanRow], depth: u32, window: u32) -> StepPlan {
        let mut items = Vec::new();
        for (ri, r) in rows.iter().enumerate() {
            assert!(r.c <= r.p, "a row has at least one host token");
            for pos in r.c..=r.p {
                let computed = (0..depth)
                    .filter(|j| pos >= chain_layer_start(r.chain_start, depth, window, *j))
                    .count();
                let top = r.sample && pos >= r.p;
                let depth_here = u32::try_from(computed).expect("at most the chain's depth");
                assert!(
                    !top || depth_here == depth,
                    "slot {}: sampling at {pos}, where the chain's top layer does not run",
                    r.slot
                );
                items.push(Item {
                    row: u32::try_from(ri).expect("rows of a step"),
                    slot: r.slot,
                    pos,
                    depth: depth_here,
                    top,
                    index: u32::try_from(items.len()).expect("positions of a step"),
                });
            }
        }
        items.sort_by_key(|it| (!it.top, depth - it.depth, it.index));
        let count = |f: &dyn Fn(&Item) -> bool| {
            let n = items.iter().filter(|it| f(it)).count();
            assert!(
                items[..n].iter().all(f),
                "a launch group's rows are a prefix of the step's order"
            );
            u32::try_from(n).expect("positions of a step")
        };
        let chain_rows = (0..depth)
            .map(|j| count(&|it: &Item| it.depth > j))
            .collect();
        let top_rows = count(&|it: &Item| it.top);
        let mut last_row = vec![u32::MAX; rows.len()];
        for (at, it) in items.iter().enumerate() {
            if it.pos == rows[it.row as usize].p {
                last_row[it.row as usize] = u32::try_from(at).expect("positions of a step");
            }
        }
        StepPlan {
            items,
            chain_rows,
            top_rows,
            last_row,
        }
    }

    /// Every position: the encoder's rows.
    pub fn rows(&self) -> u32 {
        u32::try_from(self.items.len()).expect("positions of a step")
    }

    /// Rows whose state crosses to the decoder: those the chain's bottom layer computes.
    pub fn boundary_rows(&self) -> u32 {
        self.chain_rows.first().copied().unwrap_or(0)
    }

    /// Rows whose output chain layer `j` must produce: those the next layer computes, or
    /// the head's.
    pub fn emit_rows(&self, j: usize) -> u32 {
        self.chain_rows.get(j + 1).copied().unwrap_or(self.top_rows)
    }

    /// Layer-positions the chain computes.
    pub fn chain_work(&self) -> u64 {
        self.chain_rows.iter().map(|n| u64::from(*n)).sum()
    }

    /// The device row of position `pos` of step row `row`.
    pub fn row_of(&self, row: u32, pos: u32) -> Option<u32> {
        self.items
            .iter()
            .position(|it| it.row == row && it.pos == pos)
            .map(|at| u32::try_from(at).expect("positions of a step"))
    }
}

/// One member of a pooled entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Member {
    /// A row of this step.
    Row(u32),
    /// The slot's open entry, left by an earlier step at `pos`.
    State { slot: Slot, pos: u32 },
}

/// The ratio-2 pooling of one step: the entries that close, in device row order of their
/// closing positions, and the positions whose projections the slot keeps.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PoolPlan {
    /// Per closing entry: its earlier member.
    pub first: Vec<Member>,
    /// Per closing entry: the row of its closing position.
    pub second: Vec<u32>,
    /// Per closing entry: the slot and the closing position.
    pub closes: Vec<(Slot, u32)>,
    /// `(row, slot, pos)`: projections kept as per-slot state.
    pub keep: Vec<(u32, Slot, u32)>,
}

/// The pooling of `plan` at `ratio` 2, keeping each row's last `cap` positions.
pub fn pool_plan(plan: &StepPlan, rows: &[PlanRow], cap: u32) -> PoolPlan {
    let mut out = PoolPlan::default();
    for (at, it) in plan.items.iter().enumerate() {
        let at = u32::try_from(at).expect("positions of a step");
        let r = &rows[it.row as usize];
        if it.pos % 2 == 1 {
            out.first.push(if it.pos > r.c {
                Member::Row(
                    plan.row_of(it.row, it.pos - 1)
                        .expect("a row is contiguous"),
                )
            } else {
                Member::State {
                    slot: it.slot,
                    pos: it.pos - 1,
                }
            });
            out.second.push(at);
            out.closes.push((it.slot, it.pos));
        }
        if it.pos + cap > r.p {
            out.keep.push((at, it.slot, it.pos));
        }
    }
    out
}

/// The window part of the attention slot lists of `items`: per row `stride` words, slot
/// `i` the entry of position `pos + i + 1 - window` (empty before position 0), the rest
/// empty. `entry` maps a slot and a position to its entry in the window group.
pub fn window_slots(
    items: &[Item],
    window: u32,
    stride: u32,
    entry: impl Fn(Slot, u32) -> u32,
) -> Vec<u32> {
    assert!(stride >= window);
    let mut out = vec![SLOT_EMPTY; items.len() * stride as usize];
    for (row, it) in out.chunks_exact_mut(stride as usize).zip(items) {
        for (i, word) in row[..window as usize].iter_mut().enumerate() {
            let i = u32::try_from(i).expect("a window slot");
            if let Some(at) = (it.pos + i + 1).checked_sub(window) {
                *word = entry(it.slot, at);
            }
        }
    }
    out
}

/// Rows the expert-major layout can need for `rows` positions routed to `active` of
/// `groups` experts, each expert's run started on a multiple of `row_align`: every pair,
/// and the padding before each run that holds one.
pub fn layout_rows(rows: u32, active: u32, groups: u32, row_align: u32) -> Option<u32> {
    let pairs = rows.checked_mul(active)?;
    let runs = pairs.min(groups);
    pairs.checked_add(runs.checked_mul(row_align.checked_sub(1)?)?)
}

/// The expert-major placement (`PlaceParams`): per pair its layout row, per layout row
/// its routing weight, and each expert's end row. Pairs of one expert keep their (row,
/// slot) order; a run starts at the previous run's end rounded up to `row_align`.
pub fn place(
    ids: &[u32],
    weights: &[f32],
    groups: u32,
    row_align: u32,
    layout: u32,
) -> Option<(Vec<u32>, Vec<f32>, Vec<u32>)> {
    let mut counts = vec![0u32; groups as usize];
    for id in ids {
        *counts.get_mut(*id as usize)? += 1;
    }
    let (mut start, mut end) = (vec![0u32; groups as usize], vec![0u32; groups as usize]);
    let mut at = 0u32;
    for g in 0..groups as usize {
        at = at.next_multiple_of(row_align);
        start[g] = at;
        at += counts[g];
        end[g] = at;
    }
    if at > layout {
        return None;
    }
    let mut row_of = vec![0u32; ids.len()];
    let mut row_weight = vec![0.0f32; layout as usize];
    let mut next = start;
    for (i, id) in ids.iter().enumerate() {
        let row = next[*id as usize];
        next[*id as usize] += 1;
        row_of[i] = row;
        row_weight[row as usize] = weights[i];
    }
    Some((row_of, row_weight, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(slot: Slot, c: u32, p: u32, sample: bool, chain_start: u32) -> PlanRow {
        PlanRow {
            slot,
            c,
            p,
            sample,
            chain_start,
        }
    }

    /// A triangle: the top layer from `chain_start`, each layer below from `window - 1`
    /// earlier, the head at the last position.
    #[test]
    fn a_triangle_orders_deepest_first() {
        // Depth 4, window 4: layer starts 11, 14, 17, 20 for chain_start 20.
        let rows = [row(3, 8, 23, true, 20)];
        let plan = StepPlan::new(&rows, 4, 4);
        assert_eq!(plan.rows(), 16);
        assert_eq!(plan.chain_rows, vec![13, 10, 7, 4]);
        assert_eq!(plan.top_rows, 1);
        assert_eq!(plan.boundary_rows(), 13);
        assert_eq!(plan.items[0].pos, 23);
        assert_eq!(
            plan.items[1..4].iter().map(|i| i.pos).collect::<Vec<_>>(),
            vec![20, 21, 22]
        );
        assert_eq!(
            plan.items[13..].iter().map(|i| i.pos).collect::<Vec<_>>(),
            vec![8, 9, 10]
        );
        assert_eq!(
            (0..4).map(|j| plan.emit_rows(j)).collect::<Vec<_>>(),
            [10, 7, 4, 1]
        );
        assert_eq!(plan.chain_work(), 34);
        assert_eq!(plan.last_row, vec![0]);
        for (j, n) in plan.chain_rows.iter().enumerate() {
            let start = chain_layer_start(20, 4, 4, u32::try_from(j).unwrap());
            let want: Vec<u32> = (8..=23).filter(|p| *p >= start).collect();
            let mut got: Vec<u32> = plan.items[..*n as usize].iter().map(|i| i.pos).collect();
            got.sort_unstable();
            assert_eq!(got, want, "layer {j}");
        }
    }

    /// Every launch group is a prefix for mixed rows: decode rows, encoder-only chunks,
    /// triangles cut anywhere, rows that do not sample.
    #[test]
    fn every_group_is_a_prefix_for_mixed_steps() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = |n: u32| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            u32::try_from((state >> 33) % u64::from(n)).unwrap()
        };
        for _ in 0..500 {
            let (depth, window) = (1 + next(5), 2 + next(9));
            let rows: Vec<PlanRow> = (0..1 + next(5))
                .map(|slot| {
                    let c = next(60);
                    let p = c + next(40);
                    let sample = next(2) == 0;
                    // A sampling row's top layer runs at its last position.
                    let chain_start = if sample {
                        c + next(p - c + 1)
                    } else {
                        c + next(p - c + 30)
                    };
                    row(slot, c, p, sample, chain_start)
                })
                .collect();
            let plan = StepPlan::new(&rows, depth, window);
            let total: u32 = rows.iter().map(|r| r.p - r.c + 1).sum();
            assert_eq!(plan.rows(), total);
            for j in 0..depth {
                let n = plan.chain_rows[j as usize] as usize;
                for (at, it) in plan.items.iter().enumerate() {
                    let r = &rows[it.row as usize];
                    let computes = it.pos >= chain_layer_start(r.chain_start, depth, window, j);
                    assert_eq!(computes, at < n, "layer {j} row {at}");
                }
                assert!(plan.emit_rows(j as usize) <= plan.chain_rows[j as usize]);
            }
            for (at, it) in plan.items.iter().enumerate() {
                let r = &rows[it.row as usize];
                assert_eq!(it.top, r.sample && it.pos == r.p);
                assert_eq!(it.top, at < plan.top_rows as usize);
            }
            for (ri, r) in rows.iter().enumerate() {
                let at = plan.last_row[ri] as usize;
                assert_eq!((plan.items[at].row as usize, plan.items[at].pos), (ri, r.p));
            }
            // Within a group the reference executor's order is kept.
            assert!(plan.items.windows(2).all(|w| (w[0].top, w[0].depth)
                != (w[1].top, w[1].depth)
                || w[0].index < w[1].index));
        }
    }

    #[test]
    fn pooling_pairs_rows_and_the_slots_open_entry() {
        // Row 0 starts on an odd position: its first entry closes on the slot's state.
        let rows = [row(0, 5, 9, false, 9), row(1, 2, 2, true, 2)];
        let plan = StepPlan::new(&rows, 2, 3);
        let pool = pool_plan(&plan, &rows, 3);
        let at = |r: u32, pos: u32| plan.row_of(r, pos).unwrap();
        let mut closes: Vec<_> = pool
            .closes
            .iter()
            .zip(&pool.first)
            .zip(&pool.second)
            .map(|((c, f), s)| (*c, *f, *s))
            .collect();
        closes.sort_by_key(|c| c.0);
        assert_eq!(
            closes,
            vec![
                ((0, 5), Member::State { slot: 0, pos: 4 }, at(0, 5)),
                ((0, 7), Member::Row(at(0, 6)), at(0, 7)),
                ((0, 9), Member::Row(at(0, 8)), at(0, 9)),
            ]
        );
        let mut keep = pool.keep.clone();
        keep.sort_by_key(|k| (k.1, k.2));
        assert_eq!(
            keep,
            vec![
                (at(0, 7), 0, 7),
                (at(0, 8), 0, 8),
                (at(0, 9), 0, 9),
                (at(1, 2), 1, 2)
            ]
        );
    }

    #[test]
    fn window_slots_are_right_aligned() {
        let rows = [row(2, 0, 4, true, 0)];
        let plan = StepPlan::new(&rows, 1, 3);
        let slots = window_slots(&plan.items, 3, 5, |slot, pos| slot * 100 + pos);
        let of = |pos: u32| {
            let r = plan.row_of(0, pos).unwrap() as usize;
            slots[r * 5..(r + 1) * 5].to_vec()
        };
        let e = SLOT_EMPTY;
        assert_eq!(of(0), vec![e, e, 200, e, e]);
        assert_eq!(of(1), vec![e, 200, 201, e, e]);
        assert_eq!(of(4), vec![202, 203, 204, e, e]);
    }

    #[test]
    fn placement_is_expert_major_and_stable() {
        // Three rows of two experts each, ascending ids within a row.
        let ids = [0, 2, 2, 3, 0, 3];
        let weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let (row_of, row_weight, end) = place(&ids, &weights, 4, 1, 6).unwrap();
        assert_eq!(row_of, vec![0, 2, 3, 4, 1, 5]);
        assert_eq!(row_weight, vec![0.1, 0.5, 0.2, 0.3, 0.4, 0.6]);
        assert_eq!(end, vec![2, 2, 4, 6]);

        let bound = layout_rows(3, 2, 4, 4).unwrap();
        assert_eq!(bound, 6 + 4 * 3);
        let (row_of, _, end) = place(&ids, &weights, 4, 4, bound).unwrap();
        assert_eq!(row_of, vec![0, 4, 5, 8, 1, 9]);
        assert_eq!(end, vec![2, 4, 6, 10]);
        assert!(place(&ids, &weights, 4, 4, 9).is_none());
        assert!(place(&[4], &[1.0], 4, 1, 1).is_none());
        assert_eq!(layout_rows(u32::MAX, 2, 4, 1), None);
    }
}
