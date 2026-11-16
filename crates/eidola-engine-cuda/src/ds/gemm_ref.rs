//! Launch wrappers for the DeepSeek-V4.1 reference products
//! (`ds_reference.cu`): the quantised GEMMs as scalar integer block dots,
//! which define what a tensor-core form must equal, and the attention
//! output's grouped projection.

use super::abi::{GroupedDotParams, Lin4Params, Lin8Params};
use super::ops::{ELEMENT_THREADS, LINEAR_CHAINS, addr, bind, grid, require, wrapper};
use crate::module::{Kernel, KernelDir, KernelModule};
use crate::{Gpu, Result, launch};

fn blocks(items: u32, per_block: u32) -> u64 {
    u64::from(items.div_ceil(per_block))
}

/// `eidola_ds_reference_lin8`'s grid, after its checks.
pub fn lin8_grid(p: &Lin8Params) -> Result<[u32; 3]> {
    let w = "ds lin8";
    addr(p.codes, 1, w)?;
    addr(p.act_scales, 1, w)?;
    addr(p.w_codes, 1, w)?;
    addr(p.w_scales, 1, w)?;
    addr(p.out, 2, w)?;
    require(p.n > 0, "ds lin8: no output")?;
    require(
        p.k > 0 && p.k.is_multiple_of(32),
        "ds lin8: inputs not whole 32-wide blocks",
    )?;
    require(p.reserved == 0, "ds lin8: reserved")?;
    grid(blocks(p.n, ELEMENT_THREADS), p.rows.into(), 1, w)
}

/// `eidola_ds_reference_lin4`'s grid.
pub fn lin4_grid(p: &Lin4Params) -> Result<[u32; 3]> {
    let w = "ds lin4";
    addr(p.codes, 1, w)?;
    addr(p.act_scales, 1, w)?;
    addr(p.w_packed, 1, w)?;
    addr(p.w_scales, 1, w)?;
    addr(p.group_end, 4, w)?;
    addr(p.out, 2, w)?;
    require(p.n > 0 && p.groups > 0, "ds lin4: no output or group")?;
    require(
        p.k > 0 && p.k.is_multiple_of(32),
        "ds lin4: inputs not whole 32-wide blocks",
    )?;
    require(p.row_align > 0 && p.reserved == 0, "ds lin4: row alignment")?;
    grid(blocks(p.n, ELEMENT_THREADS), p.layout_rows.into(), 1, w)
}

/// `eidola_ds_grouped_dot`'s grid.
pub fn grouped_dot_grid(p: &GroupedDotParams) -> Result<[u32; 3]> {
    let w = "ds grouped dot";
    addr(p.codes, 1, w)?;
    addr(p.act_scales, 1, w)?;
    addr(p.w_codes, 1, w)?;
    addr(p.w_scales, 1, w)?;
    addr(p.out, 2, w)?;
    require(p.rank > 0, "ds grouped dot: no output")?;
    require(
        p.k > 0 && p.k.is_multiple_of(32),
        "ds grouped dot: inputs not whole 32-wide blocks",
    )?;
    require(
        p.groups.checked_mul(p.rank).is_some() && p.groups.checked_mul(p.k).is_some(),
        "ds grouped dot: row width",
    )?;
    grid(
        blocks(p.rank, 16 * LINEAR_CHAINS),
        p.groups.into(),
        p.rows.into(),
        w,
    )
}

/// The reference products' kernels.
pub struct DsGemmRef {
    _module: KernelModule,
    lin8: Kernel,
    lin4: Kernel,
    grouped_dot: Kernel,
}

impl DsGemmRef {
    pub const IMAGE: &'static str = "ds_reference";

    pub fn load(gpu: &Gpu, dir: &KernelDir) -> Result<DsGemmRef> {
        DsGemmRef::from_module(KernelModule::load(gpu, dir, Self::IMAGE)?)
    }

    pub fn from_module(m: KernelModule) -> Result<DsGemmRef> {
        m.expect_image(Self::IMAGE)?;
        Ok(DsGemmRef {
            lin8: bind::<Lin8Params>(&m, "eidola_ds_reference_lin8", ELEMENT_THREADS)?,
            lin4: bind::<Lin4Params>(&m, "eidola_ds_reference_lin4", ELEMENT_THREADS)?,
            grouped_dot: bind::<GroupedDotParams>(&m, "eidola_ds_grouped_dot", ELEMENT_THREADS)?,
            _module: m,
        })
    }

    wrapper!(
        /// `linear::lin8` after its quantiser.
        lin8, Lin8Params, lin8_grid, rows, lin8
    );
    wrapper!(
        /// `linear::lin4` after its quantiser, over an expert-major layout
        /// (`group_end` as the placement wrote it). Padding rows' outputs
        /// are not written.
        lin4, Lin4Params, lin4_grid, layout_rows, lin4
    );
    wrapper!(
        /// The attention output's grouped projection (`wo_a`): the pinned
        /// dot of the fake-FP8 output and the stored FP8 weight, both
        /// dequantised on load.
        grouped_dot, GroupedDotParams, grouped_dot_grid, rows, grouped_dot
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: u64 = 0x1000;

    #[test]
    fn grids_and_refusals() {
        let lin8 = Lin8Params {
            codes: A,
            act_scales: A,
            w_codes: A,
            w_scales: A,
            out: A,
            rows: 3,
            n: 32_768,
            k: 1280,
            reserved: 0,
        };
        assert_eq!(lin8_grid(&lin8).unwrap(), [128, 3, 1]);
        assert!(lin8_grid(&Lin8Params { k: 1270, ..lin8 }).is_err());
        assert!(lin8_grid(&Lin8Params { n: 0, ..lin8 }).is_err());
        assert!(lin8_grid(&Lin8Params { out: A + 1, ..lin8 }).is_err());
        assert!(
            lin8_grid(&Lin8Params {
                w_scales: 0,
                ..lin8
            })
            .is_err()
        );
        let lin4 = Lin4Params {
            codes: A,
            act_scales: A,
            w_packed: A,
            w_scales: A,
            group_end: A,
            out: A,
            layout_rows: 54_000,
            n: 2304,
            k: 5120,
            groups: 384,
            row_align: 16,
            reserved: 0,
        };
        assert_eq!(lin4_grid(&lin4).unwrap(), [9, 54_000, 1]);
        assert!(
            lin4_grid(&Lin4Params {
                layout_rows: 65_536,
                ..lin4
            })
            .is_err()
        );
        assert!(
            lin4_grid(&Lin4Params {
                row_align: 0,
                ..lin4
            })
            .is_err()
        );
        assert!(
            lin4_grid(&Lin4Params {
                group_end: A + 2,
                ..lin4
            })
            .is_err()
        );
        let grouped = GroupedDotParams {
            codes: A,
            act_scales: A,
            w_codes: A,
            w_scales: A,
            out: A,
            rows: 2,
            groups: 8,
            rank: 1024,
            k: 4096,
        };
        assert_eq!(grouped_dot_grid(&grouped).unwrap(), [16, 8, 2]);
        assert_eq!(
            grouped_dot_grid(&GroupedDotParams {
                rank: 1025,
                ..grouped
            })
            .unwrap(),
            [17, 8, 2]
        );
        assert!(grouped_dot_grid(&GroupedDotParams { k: 4100, ..grouped }).is_err());
        assert!(
            grouped_dot_grid(&GroupedDotParams {
                groups: 0,
                ..grouped
            })
            .is_err()
        );
    }
}
