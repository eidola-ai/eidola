//! A [`Device`] with no memory behind it: allocations are addresses, copies and zeros
//! do nothing, read-backs are zeros, and a launch runs what its wrapper checks before a
//! device launch (the struct's preconditions and the grid) and nothing else.
//!
//! With a half laid out without its checkpoint ([`super::weights::HalfWeights::load`]
//! with no store) it answers, on any machine and at any scale, what only arithmetic
//! decides: how many bytes of each kind a configuration puts on a device, whether every
//! count and offset fits the kernels' integers, and whether every launch of a step is
//! one its wrapper accepts.

use std::any::Any;
use std::cell::Cell;

use eidola_engine::kv_cipher::KvNonce;
use eidola_engine::secret::KvKey;

use super::abi::*;
use super::half::{Device, SampleArgs};
use super::kernels::DsKernels;
use super::{attention, gemm_ref, head, index, ops};
use crate::{CudaError, Result};

/// The dry device. Addresses start well above zero and are 256-byte aligned, as a
/// device's allocations are.
#[derive(Debug)]
pub struct DryDevice {
    next: Cell<u64>,
    allocated: Cell<u64>,
    launches: Cell<u64>,
    capacity: Option<usize>,
}

impl DryDevice {
    /// A device that refuses to allocate past `capacity` bytes, when given.
    pub fn new(capacity: Option<usize>) -> DryDevice {
        DryDevice {
            next: Cell::new(1 << 32),
            allocated: Cell::new(0),
            launches: Cell::new(0),
            capacity,
        }
    }

    /// Bytes allocated so far (each allocation rounded up to 256).
    pub fn allocated(&self) -> u64 {
        self.allocated.get()
    }

    /// Launches accepted so far.
    pub fn launches(&self) -> u64 {
        self.launches.get()
    }
}

impl Device for DryDevice {
    fn alloc(&self, bytes: usize) -> Result<(u64, Box<dyn Any>)> {
        let size = (bytes as u64).next_multiple_of(256);
        let total = self.allocated.get() + size;
        if self.capacity.is_some_and(|c| total > c as u64) {
            return Err(CudaError::new(format!(
                "out of device memory: {total} bytes of {}",
                self.capacity.unwrap_or(0)
            )));
        }
        let addr = self.next.get();
        self.next.set(addr + size);
        self.allocated.set(total);
        Ok((addr, Box::new(())))
    }

    unsafe fn zero(&self, _: u64, _: usize) -> Result<()> {
        Ok(())
    }

    unsafe fn upload(&self, _: u64, _: &[u8]) -> Result<()> {
        Ok(())
    }

    unsafe fn download(&self, _: u64, dst: &mut [u8]) -> Result<()> {
        dst.fill(0);
        Ok(())
    }

    unsafe fn copy(&self, _: u64, _: u64, _: usize) -> Result<()> {
        Ok(())
    }

    unsafe fn cipher(&self, _: u64, _: u32, _: &KvKey, _: &KvNonce) -> Result<()> {
        Ok(())
    }

    unsafe fn sample(&self, _: &SampleArgs<'_>) -> Result<()> {
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn kernels(&self) -> &dyn DsKernels {
        self
    }

    fn capacity(&self) -> Option<usize> {
        self.capacity
    }
}

/// A family as its wrapper treats it before launching: nothing for no rows, else its
/// checks and its grid.
macro_rules! families {
    ($($name:ident: $params:ty, $rows:ident => $grid:expr),* $(,)?) => {
        impl DsKernels for DryDevice {
            $(
                unsafe fn $name(&self, p: &$params) -> Result<()> {
                    if p.$rows == 0 {
                        return Ok(());
                    }
                    let grid: fn(&$params) -> Result<[u32; 3]> = $grid;
                    grid(p)?;
                    self.launches.set(self.launches.get() + 1);
                    Ok(())
                }
            )*
        }
    };
}

families! {
    embed: EmbedParams, rows => ops::embed_grid,
    quant_fp8: QuantFp8Params, rows => ops::quant_fp8_grid,
    rot_quant: RotQuantParams, rows => ops::rot_quant_grid,
    rms_norm: RmsNormParams, rows => ops::rms_norm_grid,
    mhc_mix: MhcMixParams, rows => ops::mhc_mix_grid,
    collapse: CollapseParams, rows => ops::collapse_grid,
    expand: ExpandParams, rows => ops::expand_grid,
    lin8: Lin8Params, rows => gemm_ref::lin8_grid,
    lin4: Lin4Params, layout_rows => gemm_ref::lin4_grid,
    grouped_dot: GroupedDotParams, rows => gemm_ref::grouped_dot_grid,
    attention: AttnParams, rows => |p| {
        attention::check_attn(p)?;
        attention::attn_grid(p, attention::heads_per_block(p.rows, p.heads))
    },
    linear: LinearParams, rows => ops::linear_grid,
    head: LinearParams, rows => head::head_grid,
    pool: PoolParams, entries => ops::pool_grid,
    index_score: IndexScoreParams, rows => |p| {
        index::check_score(p)?;
        Ok(index::score_grid(p))
    },
    select: SelectParams, rows => |p| {
        index::check_select(p)?;
        Ok([p.rows, 1, 1])
    },
    route: RouteParams, rows => ops::route_grid,
    place: PlaceParams, groups => ops::place_grid,
    place_rows: PlaceRowsParams, rows => ops::place_rows_grid,
    swiglu: SwigluParams, rows => ops::swiglu_grid,
    combine: CombineParams, rows => ops::combine_grid,
    engram_gather: EngramGatherParams, rows => ops::engram_gather_grid,
    engram_inject: EngramInjectParams, rows => ops::engram_inject_grid,
}
