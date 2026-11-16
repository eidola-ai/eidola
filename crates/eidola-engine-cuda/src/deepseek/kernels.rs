//! The kernel launches the DeepSeek-V4.1 executor makes, as one trait.
//!
//! Each method is one kernel family of `csrc/ds_abi.cuh`, taking that family's parameter
//! struct ([`super::abi`]) with every address a device address of the half that launches
//! it. The executor is written against this trait only, so the same step runs over the
//! device kernels ([`super::half::CudaDevice`], through the launch wrappers) and over
//! the host reference ([`super::host`]), which computes each family with the model
//! crate's functions and is what the executor is checked against where there is no GPU.
//!
//! A launch is queued on the half's stream; nothing is read back.

use super::abi::*;
use crate::Result;

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
    /// Every format, [`KV_BF16`] (the rotation alone) included.
    unsafe fn rot_quant(&self, p: &RotQuantParams) -> Result<()>;
    unsafe fn rms_norm(&self, p: &RmsNormParams) -> Result<()>;
    unsafe fn mhc_mix(&self, p: &MhcMixParams) -> Result<()>;
    unsafe fn collapse(&self, p: &CollapseParams) -> Result<()>;
    unsafe fn expand(&self, p: &ExpandParams) -> Result<()>;
    unsafe fn lin8(&self, p: &Lin8Params) -> Result<()>;
    unsafe fn lin4(&self, p: &Lin4Params) -> Result<()>;
    unsafe fn grouped_dot(&self, p: &GroupedDotParams) -> Result<()>;
    unsafe fn attention(&self, p: &AttnParams) -> Result<()>;
    unsafe fn linear(&self, p: &LinearParams) -> Result<()>;
    /// [`DsKernels::linear`] laid out weight row by weight row: the output head.
    unsafe fn head(&self, p: &LinearParams) -> Result<()>;
    unsafe fn pool(&self, p: &PoolParams) -> Result<()>;
    unsafe fn index_score(&self, p: &IndexScoreParams) -> Result<()>;
    unsafe fn select(&self, p: &SelectParams) -> Result<()>;
    unsafe fn stream_mean(&self, p: &StreamMeanParams) -> Result<()>;
    unsafe fn route(&self, p: &RouteParams) -> Result<()>;
    /// Both placement launches; `scratch` holds `groups` words.
    unsafe fn place(&self, p: &PlaceParams) -> Result<()>;
    unsafe fn place_rows(&self, p: &PlaceRowsParams) -> Result<()>;
    unsafe fn swiglu(&self, p: &SwigluParams) -> Result<()>;
    unsafe fn combine(&self, p: &CombineParams) -> Result<()>;
    unsafe fn engram_gather(&self, p: &EngramGatherParams) -> Result<()>;
    unsafe fn engram_inject(&self, p: &EngramInjectParams) -> Result<()>;
}

/// The families [`DsKernels`] names, in declaration order.
pub const FAMILIES: &[&str] = &[
    "embed",
    "quant_fp8",
    "rot_quant",
    "rms_norm",
    "mhc_mix",
    "collapse",
    "expand",
    "lin8",
    "lin4",
    "grouped_dot",
    "attention",
    "linear",
    "head",
    "pool",
    "index_score",
    "select",
    "stream_mean",
    "route",
    "place",
    "place_rows",
    "swiglu",
    "combine",
    "engram_gather",
    "engram_inject",
];
