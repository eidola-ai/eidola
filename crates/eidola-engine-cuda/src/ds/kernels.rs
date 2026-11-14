//! The kernel launches the DeepSeek-V4.1 executor makes, as one trait.
//!
//! Each method is one kernel family of `csrc/ds_abi.cuh`, taking that family's parameter
//! struct ([`super::abi`]) with every address a device address of the half that launches
//! it. The executor is written against this trait only, so the same step runs over the
//! device kernels ([`Linked`]) and over the host reference ([`super::host`]), which
//! computes each family with the model crate's functions and is what the executor is
//! checked against where there is no GPU.
//!
//! A launch is queued on the half's stream; nothing is read back.

use super::abi::*;
use crate::{CudaError, Result};

/// One routed activation row per (row, slot) pair, copied into the expert-major layout:
/// row `r` of `src` goes to row `row_of[r · active + s]` of `dst`, for every `s`, as
/// `row_bytes` bytes. Not a struct of the ABI header: on the device it is
/// `eidola_copy_rows` over `rows · active` items.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlaceRowsArgs {
    /// `[rows, row_bytes]`.
    pub src: u64,
    /// `[layout_rows, row_bytes]`.
    pub dst: u64,
    /// `[rows, active]` layout rows (`PlaceParams::row_of`).
    pub row_of: u64,
    /// One status word, OR-ed into: a layout row at or past `layout_rows` copies
    /// nothing and raises `STATUS_BAD_INDEX`.
    pub status: u64,
    pub rows: u32,
    pub active: u32,
    pub row_bytes: u32,
    pub layout_rows: u32,
}

/// The kernel families of the forward.
///
/// # Safety
///
/// Every method launches a kernel over the addresses its struct names. The caller
/// guarantees that each lies inside a live allocation of this half's device, holds what
/// the struct's field documents for the counts given, and stays allocated until the
/// launch has completed; that no output overlaps an input unless the struct allows it;
/// and that every host-checked index (positions, strides, entry targets) is in range.
// One contract for every method: the trait's `# Safety`.
#[allow(clippy::missing_safety_doc)]
pub trait DsKernels {
    unsafe fn embed(&self, p: &EmbedParams) -> Result<()>;
    unsafe fn quant_fp8(&self, p: &QuantFp8Params) -> Result<()>;
    unsafe fn rot_quant(&self, p: &RotQuantParams) -> Result<()>;
    /// The rotation of [`RotQuantParams`] with no quantisation (the attention queries):
    /// row `r`'s rotated BF16 values go to row `r` of `dst_codes`, as BF16
    /// `[rows, heads · head_dim]`; `format`, `dst_scales`, `targets` and `dst` are unused.
    unsafe fn rotate(&self, p: &RotQuantParams) -> Result<()>;
    unsafe fn rms_norm(&self, p: &RmsNormParams) -> Result<()>;
    unsafe fn mhc_mix(&self, p: &MhcMixParams) -> Result<()>;
    unsafe fn collapse(&self, p: &CollapseParams) -> Result<()>;
    unsafe fn expand(&self, p: &ExpandParams) -> Result<()>;
    unsafe fn lin8(&self, p: &Lin8Params) -> Result<()>;
    unsafe fn lin4(&self, p: &Lin4Params) -> Result<()>;
    unsafe fn grouped_dot(&self, p: &GroupedDotParams) -> Result<()>;
    unsafe fn attention(&self, p: &AttnParams) -> Result<()>;
    unsafe fn linear(&self, p: &LinearParams) -> Result<()>;
    unsafe fn pool(&self, p: &PoolParams) -> Result<()>;
    unsafe fn index_score(&self, p: &IndexScoreParams) -> Result<()>;
    unsafe fn select(&self, p: &SelectParams) -> Result<()>;
    unsafe fn route(&self, p: &RouteParams) -> Result<()>;
    unsafe fn place(&self, p: &PlaceParams) -> Result<()>;
    unsafe fn place_rows(&self, p: &PlaceRowsArgs) -> Result<()>;
    unsafe fn swiglu(&self, p: &SwigluParams) -> Result<()>;
    unsafe fn combine(&self, p: &CombineParams) -> Result<()>;
    unsafe fn engram_gather(&self, p: &EngramGatherParams) -> Result<()>;
    unsafe fn engram_inject(&self, p: &EngramInjectParams) -> Result<()>;
}

/// The device kernels. No launch wrapper is bound yet: every method refuses, naming the
/// family, so an executor built over a device fails at its first launch instead of
/// computing anything. Binding a family is replacing its body with the wrapper's call.
#[derive(Debug, Default)]
pub struct Linked;

fn not_linked(family: &str) -> CudaError {
    CudaError::new(format!("ds kernel not linked: {family}"))
}

macro_rules! unlinked {
    ($($name:ident: $params:ty),* $(,)?) => {
        impl DsKernels for Linked {
            $(
                unsafe fn $name(&self, _: &$params) -> Result<()> {
                    Err(not_linked(stringify!($name)))
                }
            )*
        }

        /// The families [`DsKernels`] names, in declaration order.
        pub const FAMILIES: &[&str] = &[$(stringify!($name)),*];
    };
}

unlinked! {
    embed: EmbedParams,
    quant_fp8: QuantFp8Params,
    rot_quant: RotQuantParams,
    rotate: RotQuantParams,
    rms_norm: RmsNormParams,
    mhc_mix: MhcMixParams,
    collapse: CollapseParams,
    expand: ExpandParams,
    lin8: Lin8Params,
    lin4: Lin4Params,
    grouped_dot: GroupedDotParams,
    attention: AttnParams,
    linear: LinearParams,
    pool: PoolParams,
    index_score: IndexScoreParams,
    select: SelectParams,
    route: RouteParams,
    place: PlaceParams,
    place_rows: PlaceRowsArgs,
    swiglu: SwigluParams,
    combine: CombineParams,
    engram_gather: EngramGatherParams,
    engram_inject: EngramInjectParams,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unbound_family_refuses_by_name() {
        // SAFETY: nothing is launched.
        let e = unsafe { Linked.embed(&EmbedParams::default()) }.unwrap_err();
        assert_eq!(e.to_string(), "cuda: ds kernel not linked: embed");
        assert_eq!(FAMILIES.len(), 23);
    }
}
