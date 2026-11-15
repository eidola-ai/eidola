//! The launch wrapper of the DeepSeek-V4.1 output head and Markov head
//! (`ds_head.cu`): `linear::linear_f32` laid out weight row by weight row, a
//! lane group per vocabulary entry carrying 8 rows of logits, so a launch
//! reads the weight once per 8 rows.

use super::abi::LinearParams;
use super::ops::{ELEMENT_THREADS, HEAD_CHAINS, bind, check_linear, grid, wrapper};
use crate::module::{Kernel, KernelDir, KernelModule};
use crate::{Gpu, Result, launch};

/// `eidola_ds_head`'s grid, after its checks.
pub fn head_grid(p: &LinearParams) -> Result<[u32; 3]> {
    let w = "ds head";
    check_linear(p, w)?;
    grid(
        u64::from(p.n.div_ceil(16)),
        u64::from(p.rows.div_ceil(HEAD_CHAINS)),
        1,
        w,
    )
}

pub struct DsHead {
    _module: KernelModule,
    head: Kernel,
}

impl DsHead {
    pub const IMAGE: &'static str = "ds_head";

    pub fn load(gpu: &Gpu, dir: &KernelDir) -> Result<DsHead> {
        DsHead::from_module(KernelModule::load(gpu, dir, Self::IMAGE)?)
    }

    pub fn from_module(m: KernelModule) -> Result<DsHead> {
        m.expect_image(Self::IMAGE)?;
        Ok(DsHead {
            head: bind::<LinearParams>(&m, "eidola_ds_head", ELEMENT_THREADS)?,
            _module: m,
        })
    }

    wrapper!(
        /// Logits of every row: `out[r][n] = dot(x[x_rows[r]], w[n])`, plus
        /// `addend[r][n]` (the Markov head over the base logits).
        head, LinearParams, head_grid, rows, head
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_is_a_group_per_weight_row_and_a_block_row_per_eight_rows() {
        let p = LinearParams {
            x: 0x1000,
            w: 0x1000,
            out: 0x1000,
            rows: 9,
            n: 129_280,
            k: 5120,
            x_dtype: crate::ds::abi::DT_BF16,
            w_dtype: crate::ds::abi::DT_F32,
            out_dtype: crate::ds::abi::DT_F32,
            ..Default::default()
        };
        assert_eq!(head_grid(&p).unwrap(), [8080, 2, 1]);
        assert_eq!(
            head_grid(&LinearParams { rows: 8, ..p }).unwrap(),
            [8080, 1, 1]
        );
        assert!(head_grid(&LinearParams { out: 0x1002, ..p }).is_err());
        assert!(head_grid(&LinearParams { n: 0, ..p }).is_err());
    }
}
