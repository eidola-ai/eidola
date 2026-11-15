//! Launch wrappers for the DeepSeek-V4.1 quantised GEMMs on tensor cores
//! (`ds_gemm_tc.cu`), and the choice of form a caller makes between them and
//! the scalar reference ([`super::gemm_ref`]).
//!
//! Every form takes the reference's own parameter structs ([`Lin8Params`],
//! [`Lin4Params`]) and layouts: activation codes and scale bytes as the
//! quantiser writes them, weights as stored (FP8 codes with a scale per
//! 32 × 32 tile; packed FP4 with a scale per row and 32), the expert-major
//! layout as the placement writes it, for any `row_align`. Nothing is stored
//! twice. The forms:
//!
//! - [`GemmForm::Exact`]: the scalar integer reference (`ds_reference.cu`),
//!   the definition.
//! - [`GemmForm::Banded`]: the tensor-core form whose block products are
//!   exact by construction (exponent bands whose sums every adder that is
//!   exact on representable sums returns exactly; `ds_gemm_tc.cuh` gives the
//!   argument). Bit for bit the reference's, provided the probe
//!   ([`DsGemmTc::probe`]) finds the tensor cores exact on such sums.
//! - [`GemmForm::Tolerance`]: one tensor-core block product per block, the
//!   hardware's rounding of the 32-term dot; everything after it the
//!   reference's. Deterministic and batch-invariant; held to the reference
//!   within the measured tolerance (`tests/ds_gemm.rs`).
//!
//! A launch's blocks have one of three shapes ([`TokenTile`]), chosen from the
//! row count alone (and for `lin4` the group count); no bit of a row depends
//! on the shape (a row's arithmetic is the same operations in the same order
//! over its own operands in every shape, and the probe holds an MMA element's
//! bits to its place in the fragment).

use super::abi::{Lin4Params, Lin8Params};
use super::gemm_ref::DsGemmRef;
use super::ops::{addr, bind, grid, require};
use crate::module::{Kernel, KernelDir, KernelModule};
use crate::{Gpu, Result, launch};

/// Threads of every tensor-core GEMM block and of the probe.
pub const TC_THREADS: u32 = 128;
/// Activation and weight codes are copied 16 bytes at a time.
pub const TC_OPERAND_ALIGN: u64 = 16;
/// `lin8` runs split blocks ([`TokenTile::S8`]) up to this many rows: there
/// the product streams the weights, and only many small blocks with deep
/// staging keep enough of them in flight.
pub const SPLIT_MAX_ROWS: u32 = 32;
/// `lin4` runs split blocks while the layout holds at most this many rows per
/// group on average.
pub const SPLIT_MAX_ROWS_PER_GROUP: u32 = 8;
/// Past the split blocks: `lin8`'s tolerance form runs 32-row tiles above this
/// many rows (8-row tiles below), its banded form 8-row tiles at every count;
/// `lin4`'s banded form runs 32-row tiles above this many rows per group on
/// average, its tolerance form at every count past the split blocks. Measured
/// on a B300 (`eidola-engine-cuda/AGENTS.md` → DeepSeek-V4.1).
pub const TOLERANCE_T8_MAX_ROWS: u32 = 128;
pub const BANDED_T8_MAX_ROWS_PER_GROUP: u32 = 64;

/// How a quantised GEMM is computed. Every form takes the same parameter
/// structs and layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GemmForm {
    /// The scalar integer reference: the definition, slow.
    Exact,
    /// Tensor cores, exact block products by exponent bands.
    Banded,
    /// Tensor cores, the hardware's block products.
    Tolerance,
}

impl GemmForm {
    pub const ALL: [GemmForm; 3] = [GemmForm::Exact, GemmForm::Banded, GemmForm::Tolerance];

    /// Whether the form's outputs are the reference's bit for bit (the banded
    /// form on hardware where the probe passes).
    pub fn exact(self) -> bool {
        !matches!(self, GemmForm::Tolerance)
    }

    pub fn name(self) -> &'static str {
        match self {
            GemmForm::Exact => "exact",
            GemmForm::Banded => "banded",
            GemmForm::Tolerance => "tolerance",
        }
    }
}

/// The shape of a tensor-core GEMM's blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TokenTile {
    /// 8 rows by 16 outputs, the four warps sharing out each step's blocks
    /// and their contributions added in block order: the split block.
    S8,
    /// 8 rows by 64 outputs, a warp per 16.
    T8,
    /// 32 rows by 128 outputs, a warp per 32.
    T32,
}

impl TokenTile {
    pub const ALL: [TokenTile; 3] = [TokenTile::S8, TokenTile::T8, TokenTile::T32];

    /// The shape `lin8` runs `rows` rows in.
    pub fn for_lin8(form: TcForm, rows: u32) -> TokenTile {
        if rows <= SPLIT_MAX_ROWS {
            TokenTile::S8
        } else if form == TcForm::Banded || rows <= TOLERANCE_T8_MAX_ROWS {
            TokenTile::T8
        } else {
            TokenTile::T32
        }
    }

    /// The shape `lin4` runs a layout of `layout_rows` rows over `groups`
    /// groups in.
    pub fn for_lin4(form: TcForm, layout_rows: u32, groups: u32) -> TokenTile {
        let per_group = layout_rows.div_ceil(groups.max(1));
        if per_group <= SPLIT_MAX_ROWS_PER_GROUP {
            TokenTile::S8
        } else if form == TcForm::Banded && per_group <= BANDED_T8_MAX_ROWS_PER_GROUP {
            TokenTile::T8
        } else {
            TokenTile::T32
        }
    }

    pub fn tokens(self) -> u32 {
        match self {
            TokenTile::S8 | TokenTile::T8 => 8,
            TokenTile::T32 => 32,
        }
    }

    pub fn features(self) -> u32 {
        match self {
            TokenTile::S8 => 16,
            TokenTile::T8 => 64,
            TokenTile::T32 => 128,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            TokenTile::S8 => "s8",
            TokenTile::T8 => "t8",
            TokenTile::T32 => "t32",
        }
    }

    fn index(self) -> usize {
        match self {
            TokenTile::S8 => 0,
            TokenTile::T8 => 1,
            TokenTile::T32 => 2,
        }
    }
}

fn blocks(items: u32, per_block: u32) -> u64 {
    u64::from(items.div_ceil(per_block))
}

/// The tensor-core `lin8`'s grid for `tile`, after its checks.
pub fn lin8_tc_grid(p: &Lin8Params, tile: TokenTile) -> Result<[u32; 3]> {
    let w = "ds lin8 (tensor cores)";
    addr(p.codes, TC_OPERAND_ALIGN, w)?;
    addr(p.act_scales, 1, w)?;
    addr(p.w_codes, TC_OPERAND_ALIGN, w)?;
    addr(p.w_scales, 1, w)?;
    addr(p.out, 2, w)?;
    require(p.n > 0, "ds lin8 (tensor cores): no output")?;
    require(
        p.k > 0 && p.k.is_multiple_of(32),
        "ds lin8 (tensor cores): inputs not whole 32-wide blocks",
    )?;
    require(p.reserved == 0, "ds lin8 (tensor cores): reserved")?;
    grid(
        blocks(p.n, tile.features()),
        blocks(p.rows, tile.tokens()),
        1,
        w,
    )
}

/// The tensor-core `lin4`'s grid for `tile`, after its checks.
pub fn lin4_tc_grid(p: &Lin4Params, tile: TokenTile) -> Result<[u32; 3]> {
    let w = "ds lin4 (tensor cores)";
    addr(p.codes, TC_OPERAND_ALIGN, w)?;
    addr(p.act_scales, 1, w)?;
    addr(p.w_packed, TC_OPERAND_ALIGN, w)?;
    addr(p.w_scales, 1, w)?;
    addr(p.group_end, 4, w)?;
    addr(p.out, 2, w)?;
    require(
        p.n > 0 && p.groups > 0,
        "ds lin4 (tensor cores): no output or group",
    )?;
    require(
        p.k > 0 && p.k.is_multiple_of(32),
        "ds lin4 (tensor cores): inputs not whole 32-wide blocks",
    )?;
    require(
        p.row_align > 0 && p.reserved == 0,
        "ds lin4 (tensor cores): row alignment",
    )?;
    grid(
        blocks(p.n, tile.features()),
        blocks(p.layout_rows, tile.tokens()),
        1,
        w,
    )
}

/// The tensor-core forms ([`GemmForm::Banded`], [`GemmForm::Tolerance`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TcForm {
    Banded,
    Tolerance,
}

impl GemmForm {
    /// The tensor-core form, or `None` for the reference.
    pub fn tc(self) -> Option<TcForm> {
        match self {
            GemmForm::Exact => None,
            GemmForm::Banded => Some(TcForm::Banded),
            GemmForm::Tolerance => Some(TcForm::Tolerance),
        }
    }
}

impl TcForm {
    fn index(self) -> usize {
        match self {
            TcForm::Banded => 0,
            TcForm::Tolerance => 1,
        }
    }
}

/// `DsGemmProbeParams` (`csrc/ds_gemm_tc.cuh`): the probe of the block
/// product both tensor-core forms run.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProbeParams {
    /// `[5]` u64: dot products checked; those not equal to the exact sum
    /// rounded once; elements whose bits moved with their place in the
    /// fragment; the largest distance in f32 units in the last place; records
    /// claimed.
    pub counts: u64,
    /// `[max_records][8]` u32: global warp, trial, row, column, the bits
    /// returned, the bits of the rounded exact sum, the exact sum's low and
    /// high words (units of 2^-18, or 2^-10 for an E2M1 A).
    pub records: u64,
    pub seed: u64,
    /// Trials per warp; a trial checks 128 dot products.
    pub trials: u32,
    pub max_records: u32,
    /// E4M3 exponent fields `lo | hi << 8` the A (weight) codes are drawn from.
    pub a_band: u32,
    /// The same for B (activations).
    pub b_band: u32,
    /// [`PROBE_A_E4M3`] or [`PROBE_A_E2M1`].
    pub a_format: u32,
    /// `PROBE_UNIFORM` ..= `PROBE_CANCEL`.
    pub family: u32,
}

const _: () = {
    assert!(std::mem::size_of::<ProbeParams>() == 48);
    assert!(std::mem::align_of::<ProbeParams>() == 8);
    assert!(std::mem::offset_of!(ProbeParams, counts) == 0);
    assert!(std::mem::offset_of!(ProbeParams, records) == 8);
    assert!(std::mem::offset_of!(ProbeParams, seed) == 16);
    assert!(std::mem::offset_of!(ProbeParams, trials) == 24);
    assert!(std::mem::offset_of!(ProbeParams, max_records) == 28);
    assert!(std::mem::offset_of!(ProbeParams, a_band) == 32);
    assert!(std::mem::offset_of!(ProbeParams, b_band) == 36);
    assert!(std::mem::offset_of!(ProbeParams, a_format) == 40);
    assert!(std::mem::offset_of!(ProbeParams, family) == 44);
};

pub const PROBE_A_E4M3: u32 = 0;
pub const PROBE_A_E2M1: u32 = 1;
/// Codes uniform over the band, a sixteenth of them zero.
pub const PROBE_UNIFORM: u32 = 0;
/// Each code the band's largest or smallest magnitude, either sign.
pub const PROBE_EXTREMES: u32 = 1;
/// Every code the band's largest positive magnitude.
pub const PROBE_CARRY: u32 = 2;
/// Large products cancelling in pairs, leaving a residue.
pub const PROBE_CANCEL: u32 = 3;
/// Words of [`ProbeParams::counts`] and of one record.
pub const PROBE_COUNTS: usize = 5;
pub const PROBE_RECORD_WORDS: usize = 8;

/// An exponent-field band `[lo, hi]` as the probe takes it.
pub const fn probe_band(lo: u32, hi: u32) -> u32 {
    lo | hi << 8
}

/// The E4M3 exponent-field bands of the banded form (`dstc::*_band_*`): FP8
/// weights, activations against FP8 weights, activations against FP4
/// weights.
pub const W8_BANDS: [(u32, u32); 2] = [(0, 8), (9, 15)];
pub const X8_BANDS: [(u32, u32); 3] = [(0, 5), (6, 10), (11, 15)];
pub const X4_BANDS: [(u32, u32); 2] = [(0, 8), (9, 15)];

/// `eidola_ds_gemm_probe`'s checks; its grid is `blocks` blocks.
pub fn probe_grid(p: &ProbeParams, blocks: u32) -> Result<[u32; 3]> {
    let w = "ds gemm probe";
    addr(p.counts, 8, w)?;
    if p.max_records > 0 {
        addr(p.records, 4, w)?;
    }
    let band_ok = |b: u32| {
        let (lo, hi) = (b & 0xff, b >> 8);
        lo <= hi && hi <= 15
    };
    require(
        band_ok(p.a_band) && band_ok(p.b_band),
        "ds gemm probe: band",
    )?;
    require(
        p.a_format <= PROBE_A_E2M1 && p.family <= PROBE_CANCEL,
        "ds gemm probe: format or family",
    )?;
    grid(blocks.into(), 1, 1, w)
}

/// The tensor-core GEMMs and the probe.
pub struct DsGemmTc {
    _module: KernelModule,
    /// `[form][tile]`.
    lin8: [[Kernel; 3]; 2],
    lin4: [[Kernel; 3]; 2],
    probe: Kernel,
}

impl DsGemmTc {
    pub const IMAGE: &'static str = "ds_gemm_tc";

    pub fn load(gpu: &Gpu, dir: &KernelDir) -> Result<DsGemmTc> {
        DsGemmTc::from_module(KernelModule::load(gpu, dir, Self::IMAGE)?)
    }

    pub fn from_module(m: KernelModule) -> Result<DsGemmTc> {
        m.expect_image(Self::IMAGE)?;
        let entries = |product: &str, form: &str| -> Result<[Kernel; 3]> {
            let bind_one = |tile: TokenTile| {
                let symbol = format!("eidola_ds_{product}_{form}_{}", tile.name());
                if product == "lin8" {
                    bind::<Lin8Params>(&m, &symbol, TC_THREADS)
                } else {
                    bind::<Lin4Params>(&m, &symbol, TC_THREADS)
                }
            };
            Ok([
                bind_one(TokenTile::S8)?,
                bind_one(TokenTile::T8)?,
                bind_one(TokenTile::T32)?,
            ])
        };
        Ok(DsGemmTc {
            lin8: [entries("lin8", "banded")?, entries("lin8", "tolerance")?],
            lin4: [entries("lin4", "banded")?, entries("lin4", "tolerance")?],
            probe: bind::<ProbeParams>(&m, "eidola_ds_gemm_probe", TC_THREADS)?,
            _module: m,
        })
    }

    /// `linear::lin8` after its quantiser, in `form`, in the shape the row
    /// count picks ([`TokenTile::for_lin8`]).
    ///
    /// # Safety
    ///
    /// Every address must be valid for the arrays [`Lin8Params`] describes
    /// (`csrc/ds_abi.cuh`) until the launch completes.
    pub unsafe fn lin8(&self, gpu: &Gpu, form: TcForm, p: &Lin8Params) -> Result<()> {
        unsafe { self.lin8_tile(gpu, form, TokenTile::for_lin8(form, p.rows), p) }
    }

    /// [`DsGemmTc::lin8`] in a given shape (the tests and the bench).
    ///
    /// # Safety
    ///
    /// As [`DsGemmTc::lin8`].
    pub unsafe fn lin8_tile(
        &self,
        gpu: &Gpu,
        form: TcForm,
        tile: TokenTile,
        p: &Lin8Params,
    ) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        let grid = lin8_tc_grid(p, tile)?;
        unsafe { launch!(gpu, self.lin8[form.index()][tile.index()], grid, *p) }
    }

    /// `linear::lin4` after its quantiser over an expert-major layout, in
    /// `form`, in the shape the layout picks ([`TokenTile::for_lin4`]). Padding rows'
    /// outputs are not written.
    ///
    /// # Safety
    ///
    /// Every address must be valid for the arrays [`Lin4Params`] describes
    /// until the launch completes, and `group_end` must hold a layout of
    /// `layout_rows` rows.
    pub unsafe fn lin4(&self, gpu: &Gpu, form: TcForm, p: &Lin4Params) -> Result<()> {
        unsafe {
            self.lin4_tile(
                gpu,
                form,
                TokenTile::for_lin4(form, p.layout_rows, p.groups),
                p,
            )
        }
    }

    /// [`DsGemmTc::lin4`] in a given shape.
    ///
    /// # Safety
    ///
    /// As [`DsGemmTc::lin4`].
    pub unsafe fn lin4_tile(
        &self,
        gpu: &Gpu,
        form: TcForm,
        tile: TokenTile,
        p: &Lin4Params,
    ) -> Result<()> {
        if p.layout_rows == 0 {
            return Ok(());
        }
        let grid = lin4_tc_grid(p, tile)?;
        unsafe { launch!(gpu, self.lin4[form.index()][tile.index()], grid, *p) }
    }

    /// The probe over `blocks` blocks of four warps.
    ///
    /// # Safety
    ///
    /// `counts` must hold [`PROBE_COUNTS`] u64 words and `records`
    /// `max_records` records of [`PROBE_RECORD_WORDS`] u32 words, both valid
    /// until the launch completes.
    pub unsafe fn probe(&self, gpu: &Gpu, p: &ProbeParams, blocks: u32) -> Result<()> {
        let grid = probe_grid(p, blocks)?;
        unsafe { launch!(gpu, self.probe, grid, *p) }
    }
}

/// The quantised GEMMs in every form: the reference and the tensor-core
/// image, for a caller that chooses the form at run time.
pub struct DsGemms<'a> {
    pub reference: &'a DsGemmRef,
    pub tc: &'a DsGemmTc,
}

impl DsGemms<'_> {
    /// `lin8` in `form`.
    ///
    /// # Safety
    ///
    /// As [`DsGemmTc::lin8`].
    pub unsafe fn lin8(&self, gpu: &Gpu, form: GemmForm, p: &Lin8Params) -> Result<()> {
        match form.tc() {
            None => unsafe { self.reference.lin8(gpu, p) },
            Some(f) => unsafe { self.tc.lin8(gpu, f, p) },
        }
    }

    /// `lin4` in `form`.
    ///
    /// # Safety
    ///
    /// As [`DsGemmTc::lin4`].
    pub unsafe fn lin4(&self, gpu: &Gpu, form: GemmForm, p: &Lin4Params) -> Result<()> {
        match form.tc() {
            None => unsafe { self.reference.lin4(gpu, p) },
            Some(f) => unsafe { self.tc.lin4(gpu, f, p) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: u64 = 0x1000;

    fn lin8() -> Lin8Params {
        Lin8Params {
            codes: A,
            act_scales: A + 1,
            w_codes: A,
            w_scales: A + 3,
            out: A,
            rows: 3,
            n: 32_768,
            k: 1280,
            reserved: 0,
        }
    }

    fn lin4() -> Lin4Params {
        Lin4Params {
            codes: A,
            act_scales: A + 1,
            w_packed: A,
            w_scales: A + 1,
            group_end: A,
            out: A,
            layout_rows: 54_000,
            n: 4608,
            k: 5120,
            groups: 384,
            row_align: 1,
            reserved: 0,
        }
    }

    #[test]
    fn tiles_follow_the_row_count() {
        use TcForm::{Banded, Tolerance};
        for form in [Banded, Tolerance] {
            assert_eq!(TokenTile::for_lin8(form, 1), TokenTile::S8);
            assert_eq!(TokenTile::for_lin8(form, SPLIT_MAX_ROWS), TokenTile::S8);
            assert_eq!(TokenTile::for_lin8(form, SPLIT_MAX_ROWS + 1), TokenTile::T8);
            // One token routed to six of 384 experts; 512 tokens.
            assert_eq!(TokenTile::for_lin4(form, 6, 384), TokenTile::S8);
            assert_eq!(TokenTile::for_lin4(form, 3072, 384), TokenTile::S8);
        }
        assert_eq!(TokenTile::for_lin4(Banded, 3073, 384), TokenTile::T8);
        assert_eq!(TokenTile::for_lin4(Tolerance, 3073, 384), TokenTile::T32);
        assert_eq!(TokenTile::for_lin8(Banded, 8192), TokenTile::T8);
        assert_eq!(
            TokenTile::for_lin8(Tolerance, TOLERANCE_T8_MAX_ROWS),
            TokenTile::T8
        );
        assert_eq!(
            TokenTile::for_lin8(Tolerance, TOLERANCE_T8_MAX_ROWS + 1),
            TokenTile::T32
        );
        assert_eq!(TokenTile::for_lin4(Banded, 64 * 384, 384), TokenTile::T8);
        assert_eq!(
            TokenTile::for_lin4(Banded, 64 * 384 + 384, 384),
            TokenTile::T32
        );
        assert_eq!(
            TokenTile::for_lin4(Tolerance, 64 * 384, 384),
            TokenTile::T32
        );
        // No group is no different from one.
        assert_eq!(TokenTile::for_lin4(Banded, 4, 0), TokenTile::S8);
        for tile in TokenTile::ALL {
            // Whole 16-output MMA tiles.
            assert_eq!(tile.features() % 16, 0);
        }
    }

    #[test]
    fn lin8_grids_and_refusals() {
        let p = lin8();
        assert_eq!(lin8_tc_grid(&p, TokenTile::S8).unwrap(), [2048, 1, 1]);
        assert_eq!(lin8_tc_grid(&p, TokenTile::T8).unwrap(), [512, 1, 1]);
        assert_eq!(lin8_tc_grid(&p, TokenTile::T32).unwrap(), [256, 1, 1]);
        let p = Lin8Params {
            rows: 8192,
            n: 5120,
            k: 8192,
            ..lin8()
        };
        assert_eq!(lin8_tc_grid(&p, TokenTile::T32).unwrap(), [40, 256, 1]);
        assert_eq!(
            lin8_tc_grid(&Lin8Params { n: 1, ..p }, TokenTile::T8).unwrap(),
            [1, 1024, 1]
        );
        for bad in [
            Lin8Params { k: 1270, ..p },
            Lin8Params { k: 0, ..p },
            Lin8Params { n: 0, ..p },
            Lin8Params { reserved: 1, ..p },
            Lin8Params { codes: A + 8, ..p },
            Lin8Params {
                w_codes: A + 4,
                ..p
            },
            Lin8Params { out: A + 1, ..p },
            Lin8Params { w_scales: 0, ..p },
            Lin8Params { act_scales: 0, ..p },
            // Past the driver's grid: 32 rows a block.
            Lin8Params {
                rows: u32::MAX,
                ..p
            },
        ] {
            assert!(lin8_tc_grid(&bad, TokenTile::T8).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn lin4_grids_and_refusals() {
        let p = lin4();
        assert_eq!(lin4_tc_grid(&p, TokenTile::T32).unwrap(), [36, 1688, 1]);
        assert_eq!(lin4_tc_grid(&p, TokenTile::T8).unwrap(), [72, 6750, 1]);
        assert_eq!(lin4_tc_grid(&p, TokenTile::S8).unwrap(), [288, 6750, 1]);
        // Any alignment of the runs is served (a tile then meets several groups).
        for row_align in [1, 3, 8, 128] {
            assert!(lin4_tc_grid(&Lin4Params { row_align, ..p }, TokenTile::T8).is_ok());
        }
        for bad in [
            Lin4Params { row_align: 0, ..p },
            Lin4Params { groups: 0, ..p },
            Lin4Params { n: 0, ..p },
            Lin4Params { k: 5100, ..p },
            Lin4Params { reserved: 1, ..p },
            Lin4Params {
                group_end: A + 2,
                ..p
            },
            Lin4Params {
                w_packed: A + 8,
                ..p
            },
            Lin4Params { codes: A + 1, ..p },
            Lin4Params { out: A + 1, ..p },
        ] {
            assert!(lin4_tc_grid(&bad, TokenTile::T32).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn probe_refusals() {
        let p = ProbeParams {
            counts: A,
            records: A,
            seed: 7,
            trials: 4,
            max_records: 16,
            a_band: probe_band(0, 8),
            b_band: probe_band(11, 15),
            a_format: PROBE_A_E4M3,
            family: PROBE_CANCEL,
        };
        assert_eq!(probe_grid(&p, 1184).unwrap(), [1184, 1, 1]);
        assert!(probe_grid(&ProbeParams { counts: A + 4, ..p }, 1).is_err());
        assert!(
            probe_grid(
                &ProbeParams {
                    a_band: probe_band(9, 8),
                    ..p
                },
                1
            )
            .is_err()
        );
        assert!(
            probe_grid(
                &ProbeParams {
                    b_band: probe_band(0, 16),
                    ..p
                },
                1
            )
            .is_err()
        );
        assert!(probe_grid(&ProbeParams { family: 4, ..p }, 1).is_err());
        assert!(
            probe_grid(
                &ProbeParams {
                    records: 0,
                    max_records: 0,
                    ..p
                },
                1
            )
            .is_ok()
        );
    }

    /// The band tables partition the E4M3 exponent fields 0..=15, in order.
    #[test]
    fn bands_cover_every_exponent_once() {
        for bands in [&W8_BANDS[..], &X8_BANDS[..], &X4_BANDS[..]] {
            let mut next = 0;
            for &(lo, hi) in bands {
                assert_eq!(lo, next);
                assert!(hi >= lo);
                next = hi + 1;
            }
            assert_eq!(next, 16);
        }
    }
}
