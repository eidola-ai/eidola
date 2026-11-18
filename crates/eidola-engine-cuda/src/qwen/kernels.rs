//! The kernel launches the Qwen3.8 executor makes, as one trait, and each family's grid.
//!
//! Each method of [`QwenKernels`] is one kernel family of `csrc/qwen_abi.cuh`, taking that
//! family's parameter struct ([`super::abi`]) with every address an address of the device
//! that launches it. The executor is written against this trait only, so the same step
//! runs over the device kernels ([`super::cuda::CudaDevice`]) and over host memory with
//! the kernels' own bodies compiled for the host ([`super::host::HostDevice`]), which is
//! how the executor is held to the CPU reference executor where there is no GPU.
//!
//! Every family's `*_grid` function checks the struct's counts and widths against what its
//! kernel body is written for and returns the grid: a pure function of the row count and
//! the widths, tested without a device. A launch over no rows launches nothing.

use eidola_engine::sampling::Stream;

use super::abi::*;
use super::attention::Fa2Args;
pub use crate::mimo::engine_ops::{COPY_BUFFERS, CopyArgs, STATUS_BAD_INDEX, copy_grid};
pub use crate::sampler::AcceptLaunch;
use crate::{CudaError, Result};

/// One `eidola_sample` launch over device rows (`crate::sampler`): for each of
/// `num_rows` sample rows at `rows` (`SampleRow`s), the processed distribution over the
/// first `n` logits of its logit row into `probs[r · n ..]` (f64) and, with `draw`, a
/// token drawn from it with that stream into `tokens[r]` (a greedy row takes the
/// argmax). Non-finite logits raise `STATUS_NON_FINITE` in `status`.
#[derive(Clone, Copy, Debug)]
pub struct SampleArgs {
    pub logits: u64,
    pub logits_stride: u64,
    pub n: u32,
    pub rows: u64,
    pub num_rows: u32,
    pub draw: Option<Stream>,
    pub probs: u64,
    pub tokens: u64,
    pub status: u64,
}

/// The kernel families of the forward.
///
/// # Safety
///
/// Every method launches a kernel over the addresses its struct names. The caller
/// guarantees that each lies inside a live allocation of this device, holds what the
/// struct's field documents for the counts given, and stays allocated until the launch has
/// completed; that no output overlaps an input unless the struct allows it; and that every
/// host-checked index (positions, strides, targets, table words) is in range.
#[allow(clippy::missing_safety_doc)]
pub trait QwenKernels {
    unsafe fn embed(&self, p: &EmbedParams) -> Result<()>;
    unsafe fn norm(&self, p: &NormParams) -> Result<()>;
    unsafe fn quant(&self, p: &QuantParams) -> Result<()>;
    /// The FP8 linear: exact (`lin8_rows`) in the exact profile, the tensor-core form in
    /// the fast one.
    unsafe fn lin8(&self, p: &Lin8Params) -> Result<()>;
    /// The BF16 linear: exact (`linear_bf16_rows`) in the exact profile, the tensor-core
    /// form in the fast one for the large linears (the head and the drafter's input
    /// projection), which name an f32 scratch of `rows · n` values in `fast`; `None` for
    /// the exact form in both profiles.
    unsafe fn linear(&self, p: &LinearParams, fast: Option<u64>) -> Result<()>;
    /// The fast profile's BF16 GEMM alone (CUTLASS's, the large linears' tensor-core
    /// form): `p.rows × p.n` f32 values into `p.out`, dense (`x_stride = k`,
    /// `out_stride = n`, `out_f32 = 1`), unrounded; the caller rounds what it reads
    /// (`eidola_qwen_round`). An exact-profile GPU refuses it; the host device computes the
    /// exact linear's values (already BF16, which a rounding leaves as they are).
    unsafe fn gemm_bf16(&self, p: &LinearParams) -> Result<()>;
    unsafe fn gate(&self, p: &GateParams) -> Result<()>;
    unsafe fn kv_write(&self, p: &KvWriteParams) -> Result<()>;
    /// The exact attention core (`attend`).
    unsafe fn attention(&self, p: &AttnParams) -> Result<()>;
    unsafe fn dn_prep(&self, p: &DnPrepParams) -> Result<()>;
    unsafe fn dn_rec(&self, p: &DnRecParams) -> Result<()>;
    unsafe fn dn_norm(&self, p: &DnNormParams) -> Result<()>;
    unsafe fn widen(&self, p: &WidenParams) -> Result<()>;
    unsafe fn round(&self, p: &RoundParams) -> Result<()>;
    /// Row copies between up to [`COPY_BUFFERS`] buffers (`eidola_copy_rows`): every
    /// buffer and row index bounded, an item out of range copying nothing and raising
    /// [`STATUS_BAD_INDEX`].
    unsafe fn copy_rows(&self, a: &CopyArgs) -> Result<()>;
    /// The device sampler ([`SampleArgs`]).
    unsafe fn sample(&self, a: &SampleArgs) -> Result<()>;
    /// Chain acceptance (`eidola_chain_accept`).
    unsafe fn accept(&self, a: &AcceptLaunch) -> Result<()>;
    /// One FA2 head-dim-256 pass and its merge (the fast profile's attention).
    unsafe fn fa2(&self, a: &Fa2Args) -> Result<()>;
}

fn refuse(what: &str) -> CudaError {
    CudaError::new(format!("qwen kernel: {what}"))
}

fn ensure(ok: bool, what: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(refuse(what)) }
}

/// The driver's grid limit in y.
const MAX_GRID_Y: u32 = 65_535;

fn rows_grid(x: u32, rows: u32) -> Result<[u32; 3]> {
    ensure(rows <= MAX_GRID_Y, "rows past the grid's y limit")?;
    Ok([x.max(1), rows, 1])
}

const ELEMENT: u32 = 256;
const BLOCK_THREADS: u32 = 128;
const LIN_ROWS: u32 = 8;
/// 128-wide blocks a CTA of the quantiser takes, a warp each (`kQuantWarps`).
const QUANT_WARPS: u32 = BLOCK_THREADS / 32;
/// Rows a quantiser CTA covers (`kQuantRows`).
const QUANT_ROWS: u32 = 4;

pub fn embed_grid(p: &EmbedParams) -> Result<[u32; 3]> {
    ensure(
        p.tokens != 0 && p.table != 0 && p.out != 0 && p.status != 0,
        "embed: null",
    )?;
    ensure(p.out_stride >= p.width && p.width > 0, "embed: stride")?;
    rows_grid(p.width.div_ceil(ELEMENT), p.rows)
}

pub fn norm_grid(p: &NormParams) -> Result<[u32; 3]> {
    ensure(p.x != 0 && p.weight != 0 && p.out != 0, "norm: null")?;
    ensure(p.n > 0 && p.units > 0, "norm: empty unit")?;
    ensure(p.n <= NORM_MAX, "norm: a unit past the staged width")?;
    ensure(
        p.x_stride >= (p.units - 1) * p.x_unit_stride + p.n,
        "norm: input stride",
    )?;
    ensure(
        p.out_stride >= (p.units - 1) * p.out_unit_stride + p.n,
        "norm: output stride",
    )?;
    if p.add != 0 {
        ensure(
            p.units == 1 && p.add_stride >= p.n,
            "norm: a residual add over one unit",
        )?;
    }
    if p.rotary > 0 {
        ensure(
            p.rope_cos != 0 && p.rope_sin != 0 && p.positions != 0,
            "norm: rotation tables",
        )?;
        ensure(
            p.rotary.is_multiple_of(2) && p.rotary <= p.n && p.n <= ROPE_MAX,
            "norm: rotary width",
        )?;
    }
    ensure(p.eps.is_finite() && p.eps > 0.0, "norm: epsilon")?;
    ensure(p.units <= 0x7fff_ffff, "norm: units")?;
    rows_grid(p.units, p.rows)
}

pub fn quant_grid(p: &QuantParams) -> Result<[u32; 3]> {
    ensure(p.x != 0 && p.codes != 0 && p.scales != 0, "quant: null")?;
    ensure(
        p.k > 0 && p.k.is_multiple_of(BLOCK),
        "quant: whole 128-blocks",
    )?;
    ensure(p.x_stride >= p.k && p.m_pad >= p.rows, "quant: strides")?;
    ensure(
        p.x.is_multiple_of(8) && p.x_stride.is_multiple_of(4) && p.codes.is_multiple_of(4),
        "quant: a lane's 8-byte load and 4-byte store aligned",
    )?;
    rows_grid(
        (p.k / BLOCK).div_ceil(QUANT_WARPS),
        p.rows.div_ceil(QUANT_ROWS),
    )
}

pub fn lin8_grid(p: &Lin8Params) -> Result<[u32; 3]> {
    ensure(
        p.a != 0 && p.a_scales != 0 && p.w != 0 && p.w_scales != 0 && p.out != 0,
        "lin8: null",
    )?;
    ensure(
        p.n > 0 && p.k > 0 && p.n.is_multiple_of(BLOCK) && p.k.is_multiple_of(BLOCK),
        "lin8: whole 128-blocks",
    )?;
    ensure(p.m_pad >= p.rows && p.out_stride >= p.n, "lin8: strides")?;
    ensure(p.reserved == 0, "lin8: reserved")?;
    let y = p.rows.div_ceil(LIN_ROWS);
    ensure(y <= MAX_GRID_Y, "lin8: rows")?;
    Ok([p.n.div_ceil(BLOCK_THREADS), y.max(1), 1])
}

pub fn linear_grid(p: &LinearParams) -> Result<[u32; 3]> {
    ensure(p.x != 0 && p.w != 0 && p.out != 0, "linear: null")?;
    ensure(p.n > 0 && p.k > 0, "linear: empty")?;
    ensure(p.x_stride >= p.k && p.out_stride >= p.n, "linear: strides")?;
    ensure(p.out_f32 <= 1, "linear: output type")?;
    let y = p.rows.div_ceil(LIN_ROWS);
    ensure(y <= MAX_GRID_Y, "linear: rows")?;
    Ok([p.n.div_ceil(BLOCK_THREADS), y.max(1), 1])
}

/// Outputs (a warp each) and rows of a small-N GEMM CTA (`qwen_fast.cuh`).
const SMALL_WARPS: u32 = 4;
const SMALL_ROWS: u32 = 4;

/// The fast profile's small-N BF16 GEMM (`eidola_qwen_fast_gemm_bf16`): f32 outputs,
/// dense rows of 16-byte chunks; CTAs of four outputs by four rows.
pub fn small_gemm_grid(p: &LinearParams) -> Result<[u32; 3]> {
    ensure(p.x != 0 && p.w != 0 && p.out != 0, "small gemm: null")?;
    ensure(p.n > 0 && p.k > 0, "small gemm: empty")?;
    ensure(
        p.out_f32 == 1 && p.out_stride >= p.n,
        "small gemm: f32 rows",
    )?;
    ensure(
        p.k.is_multiple_of(8)
            && p.x_stride.is_multiple_of(8)
            && p.x_stride >= p.k
            && p.x.is_multiple_of(16)
            && p.w.is_multiple_of(16),
        "small gemm: rows of whole 16-byte chunks, aligned",
    )?;
    let y = p.rows.div_ceil(SMALL_ROWS);
    ensure(y <= MAX_GRID_Y, "small gemm: rows")?;
    Ok([p.n.div_ceil(SMALL_WARPS), y.max(1), 1])
}

pub fn gate_grid(p: &GateParams) -> Result<[u32; 3]> {
    ensure(p.a != 0 && p.g != 0 && p.out != 0, "gate: null")?;
    ensure(
        p.width > 0 && p.unit > 0 && p.width.is_multiple_of(p.unit),
        "gate: units",
    )?;
    ensure(p.kind <= GATE_SIGMOID, "gate: kind")?;
    ensure(
        p.a_stride >= p.width
            && p.out_stride >= p.width
            && p.g_stride >= (p.width / p.unit - 1) * p.g_unit_stride + p.unit,
        "gate: strides",
    )?;
    rows_grid(p.width.div_ceil(ELEMENT), p.rows)
}

/// The fast profile's gates (`eidola_qwen_fast_gate`): 256 threads of 8 elements.
pub fn fast_gate_grid(p: &GateParams) -> Result<[u32; 3]> {
    gate_grid(p)?;
    rows_grid(p.width.div_ceil(ELEMENT * 8), p.rows)
}

pub fn kv_write_grid(p: &KvWriteParams) -> Result<[u32; 3]> {
    ensure(
        p.k != 0 && p.v != 0 && p.pool != 0 && p.targets != 0,
        "kv write: null",
    )?;
    ensure(
        p.width > 0 && p.k_stride >= p.width && p.v_stride >= p.width,
        "kv write: strides",
    )?;
    ensure(
        p.layer < p.layers && p.block_size > 0 && p.reserved == 0,
        "kv write: layout",
    )?;
    rows_grid(p.width.div_ceil(ELEMENT), p.rows)
}

pub fn attention_grid(p: &AttnParams) -> Result<[u32; 3]> {
    ensure(
        p.q != 0 && p.pool != 0 && p.tables != 0 && p.positions != 0 && p.out != 0 && p.scores != 0,
        "attention: null",
    )?;
    ensure(
        p.kv_heads > 0 && p.q_heads.is_multiple_of(p.kv_heads),
        "attention: whole query groups",
    )?;
    ensure(
        p.head_dim > 0 && p.head_dim <= ELEMENT,
        "attention: head dim at most 256",
    )?;
    ensure(
        p.q_stride >= (p.q_heads - 1) * p.q_head_stride + p.head_dim
            && p.out_stride >= p.q_heads * p.head_dim,
        "attention: strides",
    )?;
    ensure(
        p.layer < p.layers && p.block_size > 0 && p.score_stride > 0,
        "attention: layout",
    )?;
    ensure(p.scale.is_finite(), "attention: scale")?;
    rows_grid(p.q_heads, p.rows)
}

pub fn dn_prep_grid(p: &DnPrepParams) -> Result<[u32; 3]> {
    ensure(
        [
            p.conv_in,
            p.window,
            p.ab,
            p.conv_w,
            p.exp_a_log,
            p.dt_bias,
            p.q,
            p.k,
            p.v,
            p.beta,
            p.alpha,
        ]
        .iter()
        .all(|&a| a != 0),
        "dn prep: null",
    )?;
    ensure(p.k_dim == KEY_DIM, "dn prep: key dim 128")?;
    ensure(
        p.k_heads > 0 && p.v_heads.is_multiple_of(p.k_heads) && p.v_dim > 0,
        "dn prep: heads",
    )?;
    ensure(p.reserved == 0, "dn prep: reserved")?;
    rows_grid(p.k_heads, p.positions)
}

/// Value columns a block of the recurrence computes (`kDnCols`): block `x` runs sequence
/// `x / slices` over columns `DN_COLS · (x % slices) ..` of value head `y`.
const DN_COLS: u32 = 32;

pub fn dn_rec_grid(p: &DnRecParams) -> Result<[u32; 3]> {
    ensure(
        [p.seqs, p.q, p.k, p.v, p.beta, p.alpha, p.out]
            .iter()
            .all(|&a| a != 0),
        "dn rec: null",
    )?;
    ensure(p.k_dim == KEY_DIM, "dn rec: key dim 128")?;
    ensure(
        p.v_dim > 0 && p.v_dim <= BLOCK_THREADS,
        "dn rec: value dim at most 128",
    )?;
    ensure(
        p.k_heads > 0 && p.v_heads.is_multiple_of(p.k_heads),
        "dn rec: heads",
    )?;
    ensure(p.reserved == 0, "dn rec: reserved")?;
    ensure(p.v_heads <= MAX_GRID_Y, "dn rec: heads")?;
    let slices = p.v_dim.div_ceil(DN_COLS);
    let x = p
        .seq_count
        .max(1)
        .checked_mul(slices)
        .ok_or_else(|| refuse("dn rec: sequences"))?;
    Ok([x, p.v_heads, 1])
}

pub fn dn_norm_grid(p: &DnNormParams) -> Result<[u32; 3]> {
    ensure(
        p.o != 0 && p.z != 0 && p.weight != 0 && p.out != 0,
        "dn norm: null",
    )?;
    ensure(
        p.v_heads > 0 && p.v_dim > 0 && p.z_stride >= p.v_heads * p.v_dim,
        "dn norm: widths",
    )?;
    ensure(
        p.v_dim <= BLOCK_THREADS,
        "dn norm: value dim at most 128 (a thread per staged value)",
    )?;
    ensure(
        p.eps.is_finite() && p.eps > 0.0 && p.reserved == 0,
        "dn norm: epsilon",
    )?;
    rows_grid(p.v_heads, p.rows)
}

pub fn widen_grid(p: &WidenParams) -> Result<[u32; 3]> {
    ensure(p.x != 0 && p.out != 0, "widen: null")?;
    ensure(
        p.width > 0 && p.x_stride >= p.width && p.out_stride >= p.width,
        "widen: strides",
    )?;
    rows_grid(p.width.div_ceil(ELEMENT), p.rows)
}

pub fn round_grid(p: &RoundParams) -> Result<[u32; 3]> {
    ensure(p.x != 0 && p.out != 0, "round: null")?;
    ensure(
        p.width > 0 && p.x_stride >= p.width && p.out_stride >= p.width,
        "round: strides",
    )?;
    ensure(p.out_f32 <= 1 && p.reserved == 0, "round: output type")?;
    rows_grid(p.width.div_ceil(ELEMENT), p.rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lin8() -> Lin8Params {
        Lin8Params {
            a: 8,
            a_scales: 8,
            w: 8,
            w_scales: 8,
            out: 8,
            rows: 9,
            n: 256,
            k: 384,
            m_pad: 12,
            out_stride: 256,
            reserved: 0,
        }
    }

    #[test]
    fn lin8_grids_cover_outputs_and_row_groups() {
        assert_eq!(lin8_grid(&lin8()).unwrap(), [2, 2, 1]);
        let mut p = lin8();
        p.rows = 0;
        assert_eq!(lin8_grid(&p).unwrap(), [2, 1, 1]);
        for bad in [
            Lin8Params { n: 200, ..lin8() },
            Lin8Params { k: 100, ..lin8() },
            Lin8Params { m_pad: 8, ..lin8() },
            Lin8Params {
                out_stride: 255,
                ..lin8()
            },
            Lin8Params { w: 0, ..lin8() },
            Lin8Params {
                reserved: 1,
                ..lin8()
            },
        ] {
            assert!(lin8_grid(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn quant_grid_is_a_warp_per_block_and_refuses_unaligned_lanes() {
        let q = QuantParams {
            x: 256,
            codes: 256,
            scales: 256,
            rows: 3,
            k: 5120,
            x_stride: 5120,
            m_pad: 4,
        };
        assert_eq!(quant_grid(&q).unwrap(), [10, 1, 1]);
        assert_eq!(quant_grid(&QuantParams { k: 640, ..q }).unwrap(), [2, 1, 1]);
        assert_eq!(
            quant_grid(&QuantParams {
                rows: 9,
                m_pad: 12,
                ..q
            })
            .unwrap(),
            [10, 3, 1]
        );
        for bad in [
            QuantParams { k: 100, ..q },
            QuantParams { x: 260, ..q },
            QuantParams {
                x_stride: 5122,
                ..q
            },
            QuantParams { codes: 258, ..q },
            QuantParams { m_pad: 2, ..q },
        ] {
            assert!(quant_grid(&bad).is_err(), "{bad:?}");
        }
    }

    fn norm() -> NormParams {
        NormParams {
            x: 8,
            weight: 8,
            out: 8,
            rows: 3,
            units: 4,
            n: 256,
            x_stride: 2048,
            x_unit_stride: 512,
            out_stride: 1024,
            out_unit_stride: 256,
            eps: 1e-6,
            ..NormParams::default()
        }
    }

    #[test]
    fn norm_grid_is_units_by_rows_and_refuses_what_its_body_cannot_run() {
        assert_eq!(norm_grid(&norm()).unwrap(), [4, 3, 1]);
        let rotated = NormParams {
            rotary: 64,
            rope_cos: 8,
            rope_sin: 8,
            positions: 8,
            ..norm()
        };
        assert_eq!(norm_grid(&rotated).unwrap(), [4, 3, 1]);
        for bad in [
            NormParams {
                x_stride: 1791,
                ..norm()
            },
            NormParams {
                out_stride: 1023,
                ..norm()
            },
            NormParams {
                add: 8,
                add_stride: 256,
                ..norm()
            },
            NormParams {
                rope_cos: 0,
                ..rotated
            },
            NormParams {
                rotary: 63,
                ..rotated
            },
            NormParams {
                n: 1024,
                x_stride: 8192,
                x_unit_stride: 1024,
                out_stride: 4096,
                out_unit_stride: 1024,
                ..rotated
            },
            NormParams { eps: 0.0, ..norm() },
            NormParams {
                n: NORM_MAX + 16,
                units: 1,
                x_stride: NORM_MAX + 16,
                out_stride: NORM_MAX + 16,
                ..norm()
            },
        ] {
            assert!(norm_grid(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn deltanet_grids_hold_the_compiled_key_dim() {
        let rec = DnRecParams {
            seqs: 8,
            q: 8,
            k: 8,
            v: 8,
            beta: 8,
            alpha: 8,
            out: 8,
            seq_count: 5,
            k_heads: 16,
            v_heads: 48,
            k_dim: 128,
            v_dim: 128,
            ..DnRecParams::default()
        };
        assert_eq!(dn_rec_grid(&rec).unwrap(), [20, 48, 1]);
        assert_eq!(
            dn_rec_grid(&DnRecParams { v_dim: 64, ..rec }).unwrap(),
            [10, 48, 1]
        );
        assert_eq!(
            dn_rec_grid(&DnRecParams { v_dim: 33, ..rec }).unwrap(),
            [10, 48, 1]
        );
        assert_eq!(
            dn_rec_grid(&DnRecParams {
                seq_count: 0,
                ..rec
            })
            .unwrap(),
            [4, 48, 1]
        );
        assert!(dn_rec_grid(&DnRecParams { k_dim: 64, ..rec }).is_err());
        assert!(dn_rec_grid(&DnRecParams { v_dim: 256, ..rec }).is_err());
        assert!(dn_rec_grid(&DnRecParams { v_heads: 47, ..rec }).is_err());
    }
}
