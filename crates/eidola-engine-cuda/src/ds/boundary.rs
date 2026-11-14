//! What crosses between the two halves: typed host values, staged through host memory in
//! both directions (between two GPUs of one confidential VM there is no other path this
//! executor assumes).
//!
//! - [`ToEncoder`] (decoder → encoder): the Engram rows gathered from the tables the
//!   decoder's device holds, as stored (codes and scales), for every position of the
//!   step. The tables never cross.
//! - [`ToDecoder`] (encoder → decoder): one bank entry of the split layer for every
//!   position of the step, in its stored form; and a boundary row (the residual copies
//!   and the collapse weights entering the split layer) for exactly the positions the
//!   chain's bottom layer computes.
//!
//! [`ToDecoder::check`] holds a value to that set, which is the one the CPU reference
//! executor's `ToDecoder::check` enforces; [`StepTransfer`] is a step's crossing in the
//! reference executor's order, for comparing the two.

use eidola_engine::executor::Slot;

use super::plan::StepPlan;
use crate::{CudaError, Result};

/// Engram rows of one table for every position of a step, in device row order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GatheredRows {
    /// `[rows, lookups · head_dim]` E4M3.
    pub codes: Vec<u8>,
    /// `[rows, lookups · head_dim / 32]` E8M0.
    pub scales: Vec<u8>,
}

/// Decoder → encoder.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToEncoder {
    /// Per Engram layer of the config: the rows of a table the decoder's device holds,
    /// `None` for a table beside the encoder.
    pub gathered: Vec<Option<GatheredRows>>,
}

/// Encoder → decoder. Rows are in device row order: every position has a bank entry,
/// and the first [`ToDecoder::rows`] of them a boundary row.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToDecoder {
    /// Every position of the step.
    pub positions: Vec<(Slot, u32)>,
    /// Bytes of one bank entry: its KV codes, KV scales, index-key codes and index-key
    /// scales.
    pub entry_bytes: usize,
    /// `[positions, entry_bytes]`.
    pub entries: Vec<u8>,
    /// Boundary rows.
    pub rows: u32,
    /// BF16 `[rows, copies · dim]`: the residual copies.
    pub streams: Vec<u16>,
    /// `[rows, copies]`: the collapse weights the split layer applies.
    pub pre_mix: Vec<f32>,
}

impl ToDecoder {
    /// Refuses a value that is not exactly what `plan` declares: an entry for every
    /// position, in the plan's order, and a boundary row for each position the chain's
    /// bottom layer computes and no other.
    pub fn check(&self, plan: &StepPlan, stream_width: usize, copies: usize) -> Result<()> {
        let want: Vec<(Slot, u32)> = plan.items.iter().map(|it| (it.slot, it.pos)).collect();
        if self.positions != want {
            return Err(CudaError::new(
                "the crossing's bank entries are not the step's positions",
            ));
        }
        let n = self.positions.len();
        if self.entries.len() != n * self.entry_bytes {
            return Err(CudaError::new("the crossing's bank entries are misshapen"));
        }
        let rows = self.rows as usize;
        if self.rows != plan.boundary_rows()
            || self.streams.len() != rows * stream_width
            || self.pre_mix.len() != rows * copies
        {
            return Err(CudaError::new(
                "the crossing's boundary rows are not those the chain computes",
            ));
        }
        Ok(())
    }
}

/// What crossed between the halves in one step, in the reference executor's order (rows
/// in step order, positions ascending).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepTransfer {
    /// Decoder → encoder: gathered Engram rows, as `(slot, position, Engram layer)`.
    pub gathered: Vec<(Slot, u32, usize)>,
    /// Encoder → decoder: bank entries, as `(slot, position)`.
    pub bank_entries: Vec<(Slot, u32)>,
    /// Bytes of one bank entry.
    pub bank_entry_bytes: usize,
    /// Encoder → decoder: boundary rows, as `(slot, position)`.
    pub boundary_rows: Vec<(Slot, u32)>,
    /// Layer-positions the chain computed.
    pub chain_work: u64,
}

impl StepTransfer {
    /// The transfer of a step from the values that crossed.
    pub fn of(plan: &StepPlan, to_encoder: &ToEncoder, to_decoder: &ToDecoder) -> StepTransfer {
        let mut order: Vec<usize> = (0..plan.items.len()).collect();
        order.sort_by_key(|at| plan.items[*at].index);
        let at = |i: &usize| (plan.items[*i].slot, plan.items[*i].pos);
        let mut gathered = Vec::new();
        for (layer, rows) in to_encoder.gathered.iter().enumerate() {
            if rows.is_some() {
                gathered.extend(order.iter().map(|i| {
                    let (slot, pos) = at(i);
                    (slot, pos, layer)
                }));
            }
        }
        StepTransfer {
            gathered,
            bank_entries: order.iter().map(|i| to_decoder.positions[*i]).collect(),
            bank_entry_bytes: if to_decoder.positions.is_empty() {
                0
            } else {
                to_decoder.entry_bytes
            },
            boundary_rows: order
                .iter()
                .filter(|i| **i < to_decoder.rows as usize)
                .map(at)
                .collect(),
            chain_work: plan.chain_work(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::plan::PlanRow;
    use super::*;

    fn crossing(plan: &StepPlan) -> ToDecoder {
        let rows = plan.boundary_rows();
        ToDecoder {
            positions: plan.items.iter().map(|it| (it.slot, it.pos)).collect(),
            entry_bytes: 3,
            entries: vec![0; plan.items.len() * 3],
            rows,
            streams: vec![0; rows as usize * 8],
            pre_mix: vec![0.0; rows as usize * 2],
        }
    }

    #[test]
    fn only_the_declared_set_crosses() {
        let rows = [PlanRow {
            slot: 1,
            c: 0,
            p: 9,
            sample: false,
            chain_start: 8,
        }];
        // Depth 2, window 3: the bottom layer from 6, the top from 8.
        let plan = StepPlan::new(&rows, 2, 3);
        assert_eq!(plan.boundary_rows(), 4);
        let good = crossing(&plan);
        good.check(&plan, 8, 2).unwrap();

        let mut extra = good.clone();
        extra.rows += 1;
        extra.streams.extend([0; 8]);
        extra.pre_mix.extend([0.0; 2]);
        assert!(
            extra.check(&plan, 8, 2).is_err(),
            "a row the chain does not compute"
        );

        let mut missing = good.clone();
        missing.positions.pop();
        missing.entries.truncate(27);
        assert!(
            missing.check(&plan, 8, 2).is_err(),
            "a position without its entry"
        );

        let mut swapped = good.clone();
        swapped.positions.swap(0, 1);
        assert!(swapped.check(&plan, 8, 2).is_err());

        let t = StepTransfer::of(
            &plan,
            &ToEncoder {
                gathered: vec![None, Some(GatheredRows::default())],
            },
            &good,
        );
        assert_eq!(t.bank_entries, (0..10).map(|p| (1, p)).collect::<Vec<_>>());
        assert_eq!(t.boundary_rows, (6..10).map(|p| (1, p)).collect::<Vec<_>>());
        assert_eq!(t.gathered, (0..10).map(|p| (1, p, 1)).collect::<Vec<_>>());
        assert_eq!((t.bank_entry_bytes, t.chain_work), (3, 6));
    }
}
