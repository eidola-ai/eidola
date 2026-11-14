//! Launch wrappers for the DeepSeek-V4.1 scalar kernels (`ds_ops.cu`,
//! `ds_ops_div.cu`): each takes its kernel's parameter struct ([`abi`]),
//! refuses one outside the kernel's limits before anything reaches the
//! device, and launches over the grid the kernel's body is written for
//! (`eidola-engine-kernels`' `csrc/ds_ops.cuh`, mirrored by the `*_grid`
//! functions here, which run without a device).
//!
//! The device addresses and what they hold are the caller's: a wrapper checks
//! that a required address is not null and is aligned for its element type,
//! and every count, stride and flag; it cannot see a buffer's length.

use super::abi::{self, *};
use crate::module::{Kernel, KernelDir, KernelModule};
use crate::{CudaError, Gpu, Result, launch};

/// Threads of a block, per kernel family (`dsk::k*Threads`).
pub const ELEMENT_THREADS: u32 = 256;
pub const BLOCK_THREADS: u32 = 128;
pub const MIX_THREADS: u32 = 1024;
pub const ROUTE_THREADS: u32 = 128;
/// Outputs a lane group of the plain product carries, and rows one of the head's.
pub const LINEAR_CHAINS: u32 = 4;
pub const HEAD_CHAINS: u32 = 8;
/// Residual copies the collapse, expand, stream mean and embedding take.
pub const MAX_COPIES: u32 = 8;
/// Residual copies the mix kernel takes: its lane groups fit one block.
pub const MAX_MIX_COPIES: u32 = 6;
/// Experts and active experts the router takes.
pub const MAX_EXPERTS: u32 = 1024;
pub const MAX_ACTIVE: u32 = 16;

// The mix's lane groups (one per coefficient, and the squared norm) fit its block.
const _: () = assert!(abi::mix_words(MAX_MIX_COPIES) < MIX_THREADS / 16);

/// The largest grid the driver takes.
const GRID_X_MAX: u64 = (1 << 31) - 1;
const GRID_YZ_MAX: u64 = 65_535;

pub(super) fn require(ok: bool, what: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(CudaError::new(what))
    }
}

/// A required device address: not null, aligned to `align` bytes.
pub(super) fn addr(a: u64, align: u64, what: &str) -> Result<()> {
    require(a != 0, &format!("{what}: null address"))?;
    require(
        a.is_multiple_of(align),
        &format!("{what}: not {align}-byte aligned"),
    )
}

/// An optional device address: null, or aligned.
pub(super) fn opt_addr(a: u64, align: u64, what: &str) -> Result<()> {
    if a == 0 { Ok(()) } else { addr(a, align, what) }
}

/// Bytes of one element of a `DT_*` type.
fn dt_bytes(dtype: u32, what: &str) -> Result<u64> {
    match dtype {
        abi::DT_BF16 => Ok(2),
        abi::DT_F32 => Ok(4),
        _ => Err(CudaError::new(format!("{what}: element type {dtype}"))),
    }
}

/// A grid of `x × y × z` blocks, refused past the driver's limits.
pub(super) fn grid(x: u64, y: u64, z: u64, what: &str) -> Result<[u32; 3]> {
    require(
        (1..=GRID_X_MAX).contains(&x)
            && (1..=GRID_YZ_MAX).contains(&y)
            && (1..=GRID_YZ_MAX).contains(&z),
        &format!("{what}: grid {x} x {y} x {z}"),
    )?;
    Ok([
        crate::narrow(x, what)?,
        crate::narrow(y, what)?,
        crate::narrow(z, what)?,
    ])
}

fn blocks(items: u32, per_block: u32) -> u64 {
    u64::from(items.div_ceil(per_block))
}

/// `eidola_ds_embed`'s grid, after its checks.
pub fn embed_grid(p: &EmbedParams) -> Result<[u32; 3]> {
    let w = "ds embed";
    addr(p.table, 2, w)?;
    addr(p.tokens, 4, w)?;
    addr(p.out, 2, w)?;
    addr(p.status, 4, w)?;
    require(p.dim > 0 && p.vocab > 0, "ds embed: empty table")?;
    require((1..=MAX_COPIES).contains(&p.copies), "ds embed: copies")?;
    grid(blocks(p.dim, ELEMENT_THREADS), p.rows.into(), 1, w)
}

/// `eidola_ds_quant_fp8`'s grid.
pub fn quant_fp8_grid(p: &QuantFp8Params) -> Result<[u32; 3]> {
    let w = "ds quant";
    require(
        p.width > 0 && p.width.is_multiple_of(32),
        "ds quant: width not whole 32-wide blocks",
    )?;
    match p.x_format {
        abi::X_BF16 => addr(p.x, 2, w)?,
        abi::X_FP8 => {
            addr(p.x, 1, w)?;
            addr(p.x_scales, 1, w)?;
        }
        _ => return Err(CudaError::new("ds quant: input format")),
    }
    addr(p.codes, 1, w)?;
    addr(p.scales, 1, w)?;
    require(p.reserved == 0, "ds quant: reserved")?;
    grid(blocks(p.width / 32, BLOCK_THREADS), p.rows.into(), 1, w)
}

/// A plane a kernel writes whole entries of `format` and `width` into: its
/// code and scale regions lie inside a block and apart.
pub fn check_plane(plane: &KvPlane, format: u32, width: u32, what: &str) -> Result<()> {
    addr(plane.base, 1, what)?;
    require(
        plane.format == format && plane.width == width && plane.reserved == 0,
        &format!("{what}: the plane is not the launch's format and width"),
    )?;
    require(
        plane.entries_per_block > 0,
        &format!("{what}: a plane block holds no entry"),
    )?;
    let entries = u64::from(plane.entries_per_block);
    let codes = entries * u64::from(abi::kv_code_bytes(format, width));
    let scales = entries * u64::from(abi::kv_scale_bytes(format, width));
    let end = |start: u64, len: u64| start.checked_add(len).filter(|e| *e <= plane.block_bytes);
    let (Some(codes_end), Some(scales_end)) = (
        end(plane.codes_offset, codes),
        end(plane.scales_offset, scales),
    ) else {
        return Err(CudaError::new(format!(
            "{what}: the plane's entries pass its block"
        )));
    };
    require(
        codes_end <= plane.scales_offset || scales_end <= plane.codes_offset,
        &format!("{what}: the plane's codes and scales overlap"),
    )
}

/// The grid of `eidola_ds_rot_quant` (`KV_FP8`, `KV_FP4_E8M0`) or
/// `eidola_ds_rot_quant_e4m3` (`KV_FP4_E4M3`).
pub fn rot_quant_grid(p: &RotQuantParams) -> Result<[u32; 3]> {
    let w = "ds rot-quant";
    addr(p.x, 2, w)?;
    require(p.heads > 0, "ds rot-quant: no head")?;
    require(
        p.head_dim > 0 && p.head_dim.is_multiple_of(32),
        "ds rot-quant: head width not whole 32-wide blocks",
    )?;
    require(
        p.rope_dim.is_multiple_of(2) && p.rope_dim <= p.head_dim,
        "ds rot-quant: rotated width",
    )?;
    require(p.format <= abi::KV_FP4_E8M0, "ds rot-quant: format")?;
    match p.rotation {
        abi::ROT_NONE => {}
        abi::ROT_ROTATE | abi::ROT_UNROTATE => {
            addr(p.cis, 4, w)?;
            addr(p.positions, 4, w)?;
            require(p.cis_positions > 0, "ds rot-quant: empty table")?;
        }
        _ => return Err(CudaError::new("ds rot-quant: rotation")),
    }
    let width = p
        .heads
        .checked_mul(p.head_dim)
        .ok_or_else(|| CudaError::new("ds rot-quant: row width"))?;
    if p.targets != 0 {
        addr(p.targets, 4, w)?;
        require(p.heads == 1, "ds rot-quant: a pool entry is one vector")?;
        check_plane(&p.dst, p.format, p.head_dim, w)?;
    } else {
        addr(p.dst_codes, 1, w)?;
        addr(p.dst_scales, 1, w)?;
    }
    require(p.reserved == 0, "ds rot-quant: reserved")?;
    grid(blocks(width / 32, BLOCK_THREADS), p.rows.into(), 1, w)
}

/// `eidola_ds_rms_norm`'s grid.
pub fn rms_norm_grid(p: &RmsNormParams) -> Result<[u32; 3]> {
    let w = "ds rms norm";
    addr(p.x, 2, w)?;
    addr(p.weight, 4, w)?;
    addr(p.out, 2, w)?;
    require(p.width > 0, "ds rms norm: width 0")?;
    require(p.eps.is_finite() && p.reserved == 0, "ds rms norm: eps")?;
    grid(p.rows.into(), 1, 1, w)
}

/// `eidola_ds_mhc_mix`'s grid.
pub fn mhc_mix_grid(p: &MhcMixParams) -> Result<[u32; 3]> {
    let w = "ds mhc mix";
    addr(p.x, 2, w)?;
    addr(p.projection, 4, w)?;
    addr(p.base, 4, w)?;
    addr(p.out, 4, w)?;
    require(
        (1..=MAX_MIX_COPIES).contains(&p.copies),
        "ds mhc mix: copies",
    )?;
    require(
        p.dim > 0 && p.dim.checked_mul(p.copies).is_some(),
        "ds mhc mix: width",
    )?;
    require(p.iters >= 1, "ds mhc mix: no Sinkhorn step")?;
    require(
        p.norm_eps.is_finite() && p.eps.is_finite() && p.reserved == 0,
        "ds mhc mix: eps",
    )?;
    grid(p.rows.into(), 1, 1, w)
}

/// `eidola_ds_collapse`'s grid.
pub fn collapse_grid(p: &CollapseParams) -> Result<[u32; 3]> {
    let w = "ds collapse";
    addr(p.x, 2, w)?;
    addr(p.pre, 4, w)?;
    addr(p.out, 2, w)?;
    require(p.dim > 0, "ds collapse: width 0")?;
    require((1..=MAX_COPIES).contains(&p.copies), "ds collapse: copies")?;
    require(p.pre_stride >= p.copies, "ds collapse: weight stride")?;
    grid(blocks(p.dim, ELEMENT_THREADS), p.rows.into(), 1, w)
}

/// `eidola_ds_expand`'s grid.
pub fn expand_grid(p: &ExpandParams) -> Result<[u32; 3]> {
    let w = "ds expand";
    addr(p.x, 2, w)?;
    addr(p.y, 2, w)?;
    addr(p.mixes, 4, w)?;
    addr(p.out, 2, w)?;
    require(p.dim > 0, "ds expand: width 0")?;
    require((1..=MAX_COPIES).contains(&p.copies), "ds expand: copies")?;
    require(
        p.mix_stride >= abi::mix_words(p.copies),
        "ds expand: mix stride",
    )?;
    grid(blocks(p.dim, ELEMENT_THREADS), p.rows.into(), 1, w)
}

/// `eidola_ds_stream_mean`'s grid.
pub fn stream_mean_grid(p: &StreamMeanParams) -> Result<[u32; 3]> {
    let w = "ds stream mean";
    addr(p.x, 2, w)?;
    addr(p.out, 2, w)?;
    require(p.dim > 0, "ds stream mean: width 0")?;
    require(
        (1..=MAX_COPIES).contains(&p.copies),
        "ds stream mean: copies",
    )?;
    require(p.out_stride >= p.dim, "ds stream mean: output stride")?;
    grid(blocks(p.dim, ELEMENT_THREADS), p.rows.into(), 1, w)
}

/// The checks every plain product shares.
pub(super) fn check_linear(p: &LinearParams, w: &str) -> Result<()> {
    addr(p.x, dt_bytes(p.x_dtype, w)?, w)?;
    addr(p.w, dt_bytes(p.w_dtype, w)?, w)?;
    addr(p.out, dt_bytes(p.out_dtype, w)?, w)?;
    opt_addr(p.x_rows, 4, w)?;
    opt_addr(p.addend, 4, w)?;
    require(p.n > 0 && p.k > 0, &format!("{w}: empty weight"))
}

/// `eidola_ds_linear`'s grid.
pub fn linear_grid(p: &LinearParams) -> Result<[u32; 3]> {
    let w = "ds linear";
    check_linear(p, w)?;
    grid(blocks(p.n, 16 * LINEAR_CHAINS), p.rows.into(), 1, w)
}

/// `eidola_ds_pool`'s grid.
pub fn pool_grid(p: &PoolParams) -> Result<[u32; 3]> {
    let w = "ds pool";
    addr(p.kv, 4, w)?;
    addr(p.score, 4, w)?;
    addr(p.first, 4, w)?;
    addr(p.second, 4, w)?;
    addr(p.out, 2, w)?;
    require(p.width > 0, "ds pool: width 0")?;
    grid(blocks(p.width, ELEMENT_THREADS), p.entries.into(), 1, w)
}

/// `eidola_ds_route`'s grid.
pub fn route_grid(p: &RouteParams) -> Result<[u32; 3]> {
    let w = "ds route";
    addr(p.logits, 4, w)?;
    addr(p.bias, 4, w)?;
    addr(p.ids, 4, w)?;
    addr(p.weights, 4, w)?;
    require((1..=MAX_EXPERTS).contains(&p.experts), "ds route: experts")?;
    require(
        (1..=MAX_ACTIVE.min(p.experts)).contains(&p.active),
        "ds route: active experts",
    )?;
    require(p.normalize <= 1 && p.reserved == 0, "ds route: flags")?;
    require(p.route_scale.is_finite(), "ds route: scale")?;
    grid(p.rows.into(), 1, 1, w)
}

/// Rows an expert-major layout of `pairs` routed pairs can reach: every run
/// but the last may be followed by up to `row_align - 1` padding rows.
pub fn layout_rows_bound(pairs: u64, groups: u32, row_align: u32) -> u64 {
    let runs = pairs.min(u64::from(groups));
    pairs + runs.saturating_sub(1) * u64::from(row_align.saturating_sub(1))
}

/// The grid of both placement launches (`eidola_ds_place_count`, then
/// `eidola_ds_place`). `scratch` holds `groups` words.
pub fn place_grid(p: &PlaceParams) -> Result<[u32; 3]> {
    let w = "ds place";
    addr(p.ids, 4, w)?;
    addr(p.weights, 4, w)?;
    addr(p.row_of, 4, w)?;
    addr(p.row_weight, 4, w)?;
    addr(p.group_end, 4, w)?;
    addr(p.scratch, 4, w)?;
    require(p.active > 0 && p.groups > 0, "ds place: no pair or group")?;
    require(
        p.row_align > 0 && p.reserved == 0,
        "ds place: row alignment",
    )?;
    let pairs = u64::from(p.rows) * u64::from(p.active);
    require(
        u64::from(p.layout_rows) >= layout_rows_bound(pairs, p.groups, p.row_align),
        "ds place: the layout cannot hold every placement of its pairs",
    )?;
    grid(blocks(p.groups, BLOCK_THREADS), 1, 1, w)
}

/// `eidola_ds_swiglu`'s grid.
pub fn swiglu_grid(p: &SwigluParams) -> Result<[u32; 3]> {
    let w = "ds swiglu";
    addr(p.gate, 2, w)?;
    addr(p.up, 2, w)?;
    opt_addr(p.weight, 4, w)?;
    addr(p.codes, 1, w)?;
    addr(p.scales, 1, w)?;
    require(
        p.width > 0 && p.width.is_multiple_of(32),
        "ds swiglu: width not whole 32-wide blocks",
    )?;
    require(p.in_stride >= p.width, "ds swiglu: input stride")?;
    require(!p.limit.is_nan(), "ds swiglu: limit")?;
    grid(blocks(p.width / 32, BLOCK_THREADS), p.rows.into(), 1, w)
}

/// `eidola_ds_combine`'s grid.
pub fn combine_grid(p: &CombineParams) -> Result<[u32; 3]> {
    let w = "ds combine";
    addr(p.routed, 2, w)?;
    addr(p.shared, 2, w)?;
    addr(p.row_of, 4, w)?;
    addr(p.out, 2, w)?;
    require(p.dim > 0 && p.reserved == 0, "ds combine: width 0")?;
    grid(blocks(p.dim, ELEMENT_THREADS), p.rows.into(), 1, w)
}

/// `eidola_ds_engram_gather`'s grid.
pub fn engram_gather_grid(p: &EngramGatherParams) -> Result<[u32; 3]> {
    let w = "ds engram gather";
    addr(p.codes, 1, w)?;
    addr(p.scales, 1, w)?;
    addr(p.ids, 4, w)?;
    addr(p.out_codes, 1, w)?;
    addr(p.out_scales, 1, w)?;
    addr(p.status, 4, w)?;
    require(
        p.width > 0 && p.width.is_multiple_of(32),
        "ds engram gather: width not whole 32-wide blocks",
    )?;
    require(p.reserved == 0, "ds engram gather: reserved")?;
    grid(
        blocks(p.width / 32, BLOCK_THREADS),
        p.ids_per_row.into(),
        p.rows.into(),
        w,
    )
}

/// `eidola_ds_engram_inject`'s grid.
pub fn engram_inject_grid(p: &EngramInjectParams) -> Result<[u32; 3]> {
    let w = "ds engram inject";
    addr(p.x, 2, w)?;
    addr(p.kv, 2, w)?;
    addr(p.gate_weight, 4, w)?;
    addr(p.out, 2, w)?;
    require(p.dim > 0, "ds engram inject: width 0")?;
    require(p.norm_eps.is_finite(), "ds engram inject: eps")?;
    grid(p.copies.into(), p.rows.into(), 1, w)
}

/// An expert-major layout ([`expert_layout`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertLayout {
    /// Each pair's layout row.
    pub row_of: Vec<u32>,
    /// Each group's end row.
    pub group_end: Vec<u32>,
    /// Rows the layout uses: the last group's end.
    pub rows: u32,
}

/// The host form of the placement kernels (`PlaceParams`): pairs of one
/// expert keep their order, a run starts on the next multiple of `row_align`
/// at or after the previous run's end, and `group_end[g]` is the end of
/// group `g`'s pairs. `ids` lists every pair's expert, below `groups`.
pub fn expert_layout(ids: &[u32], groups: u32, row_align: u32) -> ExpertLayout {
    assert!(row_align > 0, "row alignment 0");
    let mut count = vec![0u32; groups as usize];
    for id in ids {
        count[*id as usize] += 1;
    }
    let mut next = Vec::with_capacity(count.len());
    let mut group_end = Vec::with_capacity(count.len());
    let mut end = 0u32;
    for c in &count {
        let start = end.next_multiple_of(row_align);
        next.push(start);
        end = start + c;
        group_end.push(end);
    }
    let row_of = ids
        .iter()
        .map(|id| {
            let row = next[*id as usize];
            next[*id as usize] += 1;
            row
        })
        .collect();
    ExpertLayout {
        row_of,
        group_end,
        rows: end,
    }
}

/// Resolve `symbol` and hold its launch contract to `P` and `threads`.
pub(super) fn bind<P>(m: &KernelModule, symbol: &str, threads: u32) -> Result<Kernel> {
    let kernel = m.kernel(symbol)?;
    let meta = kernel.meta();
    if meta.params_bytes as usize != std::mem::size_of::<P>() || meta.block != [threads, 1, 1] {
        return Err(CudaError::new(format!("{symbol}: launch contract")));
    }
    Ok(kernel)
}

/// The scalar kernels of both images.
pub struct DsOps {
    _exact: KernelModule,
    _div: KernelModule,
    embed: Kernel,
    quant_fp8: Kernel,
    rot_quant: Kernel,
    rot_quant_e4m3: Kernel,
    rms_norm: Kernel,
    mhc_mix: Kernel,
    collapse: Kernel,
    expand: Kernel,
    stream_mean: Kernel,
    linear: Kernel,
    pool: Kernel,
    route: Kernel,
    place_count: Kernel,
    place: Kernel,
    swiglu: Kernel,
    combine: Kernel,
    engram_gather: Kernel,
    engram_inject: Kernel,
}

impl DsOps {
    /// The image of the kernels that neither divide nor take a square root.
    pub const IMAGE: &'static str = "ds_ops";
    /// The image of the kernels that do.
    pub const IMAGE_DIV: &'static str = "ds_ops_div";

    pub fn load(gpu: &Gpu, dir: &KernelDir) -> Result<DsOps> {
        DsOps::from_modules(
            KernelModule::load(gpu, dir, Self::IMAGE)?,
            KernelModule::load(gpu, dir, Self::IMAGE_DIV)?,
        )
    }

    pub fn from_modules(exact: KernelModule, div: KernelModule) -> Result<DsOps> {
        exact.expect_image(Self::IMAGE)?;
        div.expect_image(Self::IMAGE_DIV)?;
        Ok(DsOps {
            embed: bind::<EmbedParams>(&exact, "eidola_ds_embed", ELEMENT_THREADS)?,
            quant_fp8: bind::<QuantFp8Params>(&exact, "eidola_ds_quant_fp8", BLOCK_THREADS)?,
            rot_quant: bind::<RotQuantParams>(&exact, "eidola_ds_rot_quant", BLOCK_THREADS)?,
            rot_quant_e4m3: bind::<RotQuantParams>(
                &div,
                "eidola_ds_rot_quant_e4m3",
                BLOCK_THREADS,
            )?,
            rms_norm: bind::<RmsNormParams>(&div, "eidola_ds_rms_norm", ELEMENT_THREADS)?,
            mhc_mix: bind::<MhcMixParams>(&div, "eidola_ds_mhc_mix", MIX_THREADS)?,
            collapse: bind::<CollapseParams>(&exact, "eidola_ds_collapse", ELEMENT_THREADS)?,
            expand: bind::<ExpandParams>(&exact, "eidola_ds_expand", ELEMENT_THREADS)?,
            stream_mean: bind::<StreamMeanParams>(&div, "eidola_ds_stream_mean", ELEMENT_THREADS)?,
            linear: bind::<LinearParams>(&exact, "eidola_ds_linear", ELEMENT_THREADS)?,
            pool: bind::<PoolParams>(&div, "eidola_ds_pool", ELEMENT_THREADS)?,
            route: bind::<RouteParams>(&div, "eidola_ds_route", ROUTE_THREADS)?,
            place_count: bind::<PlaceParams>(&exact, "eidola_ds_place_count", BLOCK_THREADS)?,
            place: bind::<PlaceParams>(&exact, "eidola_ds_place", BLOCK_THREADS)?,
            swiglu: bind::<SwigluParams>(&div, "eidola_ds_swiglu", BLOCK_THREADS)?,
            combine: bind::<CombineParams>(&exact, "eidola_ds_combine", ELEMENT_THREADS)?,
            engram_gather: bind::<EngramGatherParams>(
                &exact,
                "eidola_ds_engram_gather",
                BLOCK_THREADS,
            )?,
            engram_inject: bind::<EngramInjectParams>(
                &div,
                "eidola_ds_engram_inject",
                ELEMENT_THREADS,
            )?,
            _exact: exact,
            _div: div,
        })
    }
}

/// One wrapper: the grid function's checks, nothing launched for no rows.
macro_rules! wrapper {
    ($(#[$doc:meta])* $name:ident, $params:ty, $grid:path, $rows:ident, $kernel:ident) => {
        $(#[$doc])*
        ///
        /// # Safety
        ///
        /// Every address must be valid for the arrays the struct's fields
        /// describe (`csrc/ds_abi.cuh`) until the launch completes.
        pub unsafe fn $name(&self, gpu: &Gpu, p: &$params) -> Result<()> {
            if p.$rows == 0 {
                return Ok(());
            }
            let grid = $grid(p)?;
            unsafe { launch!(gpu, self.$kernel, grid, *p) }
        }
    };
}
pub(super) use wrapper;

impl DsOps {
    wrapper!(
        /// Embedding into the residual copies.
        embed, EmbedParams, embed_grid, rows, embed
    );
    wrapper!(
        /// `quant::act_quant`: codes and scale bytes.
        quant_fp8, QuantFp8Params, quant_fp8_grid, rows, quant_fp8
    );
    wrapper!(
        /// `weights::rms`.
        rms_norm, RmsNormParams, rms_norm_grid, rows, rms_norm
    );
    wrapper!(
        /// `mhc::mixes`.
        mhc_mix, MhcMixParams, mhc_mix_grid, rows, mhc_mix
    );
    wrapper!(
        /// `mhc::collapse`.
        collapse, CollapseParams, collapse_grid, rows, collapse
    );
    wrapper!(
        /// `mhc::expand`.
        expand, ExpandParams, expand_grid, rows, expand
    );
    wrapper!(
        /// `forward::stream_mean`.
        stream_mean, StreamMeanParams, stream_mean_grid, rows, stream_mean
    );
    wrapper!(
        /// `linear::linear_f32` / `linear_bf16`, a block of threads per 64 outputs of a row.
        linear, LinearParams, linear_grid, rows, linear
    );
    wrapper!(
        /// The ratio-2 compressor's pooling.
        pool, PoolParams, pool_grid, entries, pool
    );
    wrapper!(
        /// `moe::route` after the logits.
        route, RouteParams, route_grid, rows, route
    );
    wrapper!(
        /// `moe::swiglu_hidden`, then `quant::act_quant`.
        swiglu, SwigluParams, swiglu_grid, rows, swiglu
    );
    wrapper!(
        /// The feed-forward sum.
        combine, CombineParams, combine_grid, rows, combine
    );
    wrapper!(
        /// Engram table rows as stored.
        engram_gather, EngramGatherParams, engram_gather_grid, rows, engram_gather
    );
    wrapper!(
        /// `engram::inject` after its projection.
        engram_inject, EngramInjectParams, engram_inject_grid, rows, engram_inject
    );

    /// Rotation and fake quantisation into packed codes and scales, by the
    /// entry of the launch's format.
    ///
    /// # Safety
    ///
    /// As every wrapper here; with `targets`, the rows' entries must be
    /// distinct and inside the plane's pool.
    pub unsafe fn rot_quant(&self, gpu: &Gpu, p: &RotQuantParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        let grid = rot_quant_grid(p)?;
        let kernel = if p.format == abi::KV_FP4_E4M3 {
            &self.rot_quant_e4m3
        } else {
            &self.rot_quant
        };
        unsafe { launch!(gpu, kernel, grid, *p) }
    }

    /// The expert-major layout ([`expert_layout`]) of a step's routed pairs,
    /// in two launches over `scratch`.
    ///
    /// # Safety
    ///
    /// As every wrapper here; every id must be below `groups`.
    pub unsafe fn place(&self, gpu: &Gpu, p: &PlaceParams) -> Result<()> {
        let grid = place_grid(p)?;
        unsafe {
            launch!(gpu, self.place_count, grid, *p)?;
            launch!(gpu, self.place, grid, *p)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An address aligned for anything.
    const A: u64 = 0x1000;

    #[test]
    fn grids_cover_every_unit_once() {
        let embed = EmbedParams {
            table: A,
            tokens: A,
            out: A,
            status: A,
            rows: 7,
            dim: 5120,
            copies: 4,
            vocab: 129_280,
        };
        assert_eq!(embed_grid(&embed).unwrap(), [20, 7, 1]);
        assert_eq!(
            embed_grid(&EmbedParams { dim: 5121, ..embed }).unwrap(),
            [21, 7, 1]
        );
        let quant = QuantFp8Params {
            x: A,
            codes: A,
            scales: A,
            rows: 3,
            width: 15_360,
            ..Default::default()
        };
        // 480 blocks of 32, 128 a thread block.
        assert_eq!(quant_fp8_grid(&quant).unwrap(), [4, 3, 1]);
        let linear = LinearParams {
            x: A,
            w: A,
            out: A,
            rows: 5,
            n: 384,
            k: 5120,
            ..Default::default()
        };
        assert_eq!(linear_grid(&linear).unwrap(), [6, 5, 1]);
        assert_eq!(
            linear_grid(&LinearParams { n: 385, ..linear }).unwrap(),
            [7, 5, 1]
        );
        let gather = EngramGatherParams {
            codes: A,
            scales: A,
            ids: A,
            out_codes: A,
            out_scales: A,
            status: A,
            table_rows: 1 << 30,
            rows: 9,
            ids_per_row: 24,
            width: 256,
            reserved: 0,
        };
        assert_eq!(engram_gather_grid(&gather).unwrap(), [1, 24, 9]);
        let mix = MhcMixParams {
            x: A,
            projection: A,
            base: A,
            out: A,
            rows: 11,
            dim: 5120,
            copies: 4,
            iters: 20,
            ..Default::default()
        };
        assert_eq!(mhc_mix_grid(&mix).unwrap(), [11, 1, 1]);
    }

    #[test]
    fn out_of_range_parameters_are_refused() {
        let quant = QuantFp8Params {
            x: A,
            codes: A,
            scales: A,
            rows: 3,
            width: 64,
            ..Default::default()
        };
        quant_fp8_grid(&quant).unwrap();
        assert!(quant_fp8_grid(&QuantFp8Params { width: 48, ..quant }).is_err());
        assert!(quant_fp8_grid(&QuantFp8Params { width: 0, ..quant }).is_err());
        assert!(quant_fp8_grid(&QuantFp8Params { x: A + 1, ..quant }).is_err());
        assert!(quant_fp8_grid(&QuantFp8Params { codes: 0, ..quant }).is_err());
        assert!(
            quant_fp8_grid(&QuantFp8Params {
                x_format: 2,
                ..quant
            })
            .is_err()
        );
        // Stored FP8 rows need their scales.
        assert!(
            quant_fp8_grid(&QuantFp8Params {
                x_format: abi::X_FP8,
                ..quant
            })
            .is_err()
        );
        assert!(
            quant_fp8_grid(&QuantFp8Params {
                rows: 65_536,
                ..quant
            })
            .is_err()
        );

        let plane = KvPlane {
            base: A,
            block_bytes: 8 * (64 + 2),
            codes_offset: 0,
            scales_offset: 8 * 64,
            entries_per_block: 8,
            format: abi::KV_FP8,
            width: 64,
            reserved: 0,
        };
        let rot = RotQuantParams {
            x: A,
            cis: A,
            positions: A,
            targets: A,
            dst: plane,
            rows: 2,
            heads: 1,
            head_dim: 64,
            rope_dim: 32,
            rotation: abi::ROT_ROTATE,
            format: abi::KV_FP8,
            cis_positions: 100,
            ..Default::default()
        };
        assert_eq!(rot_quant_grid(&rot).unwrap(), [1, 2, 1]);
        assert!(rot_quant_grid(&RotQuantParams { heads: 2, ..rot }).is_err());
        assert!(
            rot_quant_grid(&RotQuantParams {
                rope_dim: 31,
                ..rot
            })
            .is_err()
        );
        assert!(
            rot_quant_grid(&RotQuantParams {
                rope_dim: 66,
                ..rot
            })
            .is_err()
        );
        assert!(rot_quant_grid(&RotQuantParams { rotation: 3, ..rot }).is_err());
        assert!(rot_quant_grid(&RotQuantParams { format: 3, ..rot }).is_err());
        assert!(rot_quant_grid(&RotQuantParams { cis: 0, ..rot }).is_err());
        assert!(
            rot_quant_grid(&RotQuantParams {
                cis_positions: 0,
                ..rot
            })
            .is_err()
        );
        // A plane of another format or width, one whose entries pass its
        // block, and one whose codes and scales overlap.
        for bad in [
            KvPlane {
                format: abi::KV_FP4_E8M0,
                ..plane
            },
            KvPlane { width: 32, ..plane },
            KvPlane {
                block_bytes: 8 * 64 + 15,
                ..plane
            },
            KvPlane {
                scales_offset: 8 * 64 - 1,
                ..plane
            },
            KvPlane {
                entries_per_block: 0,
                ..plane
            },
            KvPlane { base: 0, ..plane },
        ] {
            assert!(
                rot_quant_grid(&RotQuantParams { dst: bad, ..rot }).is_err(),
                "{bad:?}"
            );
        }
        // Dense output needs both destinations.
        assert!(rot_quant_grid(&RotQuantParams { targets: 0, ..rot }).is_err());

        let route = RouteParams {
            logits: A,
            bias: A,
            ids: A,
            weights: A,
            rows: 1,
            experts: 384,
            active: 6,
            normalize: 1,
            route_scale: 1.5,
            reserved: 0,
        };
        route_grid(&route).unwrap();
        assert!(
            route_grid(&RouteParams {
                experts: 1025,
                ..route
            })
            .is_err()
        );
        assert!(
            route_grid(&RouteParams {
                active: 17,
                ..route
            })
            .is_err()
        );
        assert!(route_grid(&RouteParams { active: 0, ..route }).is_err());
        assert!(
            route_grid(&RouteParams {
                experts: 4,
                ..route
            })
            .is_err()
        );
        assert!(
            route_grid(&RouteParams {
                logits: A + 2,
                ..route
            })
            .is_err()
        );

        let mix = MhcMixParams {
            x: A,
            projection: A,
            base: A,
            out: A,
            rows: 1,
            dim: 64,
            copies: 4,
            iters: 20,
            ..Default::default()
        };
        mhc_mix_grid(&mix).unwrap();
        assert!(mhc_mix_grid(&MhcMixParams { copies: 7, ..mix }).is_err());
        assert!(mhc_mix_grid(&MhcMixParams { iters: 0, ..mix }).is_err());
        assert!(
            mhc_mix_grid(&MhcMixParams {
                eps: f32::NAN,
                ..mix
            })
            .is_err()
        );

        let expand = ExpandParams {
            x: A,
            y: A,
            mixes: A,
            out: A,
            rows: 1,
            dim: 64,
            copies: 4,
            mix_stride: 24,
        };
        expand_grid(&expand).unwrap();
        assert!(
            expand_grid(&ExpandParams {
                mix_stride: 23,
                ..expand
            })
            .is_err()
        );
        assert!(
            expand_grid(&ExpandParams {
                copies: 9,
                ..expand
            })
            .is_err()
        );

        let linear = LinearParams {
            x: A,
            w: A,
            out: A,
            rows: 1,
            n: 8,
            k: 8,
            x_dtype: abi::DT_F32,
            ..Default::default()
        };
        linear_grid(&linear).unwrap();
        assert!(linear_grid(&LinearParams { x: A + 2, ..linear }).is_err());
        assert!(
            linear_grid(&LinearParams {
                w_dtype: 2,
                ..linear
            })
            .is_err()
        );
        assert!(linear_grid(&LinearParams { k: 0, ..linear }).is_err());
        assert!(
            linear_grid(&LinearParams {
                addend: A + 1,
                ..linear
            })
            .is_err()
        );

        let place = PlaceParams {
            ids: A,
            weights: A,
            row_of: A,
            row_weight: A,
            group_end: A,
            scratch: A,
            rows: 4,
            active: 6,
            groups: 384,
            row_align: 16,
            layout_rows: 24 + 23 * 15,
            reserved: 0,
        };
        assert_eq!(place_grid(&place).unwrap(), [3, 1, 1]);
        assert!(
            place_grid(&PlaceParams {
                layout_rows: 24 + 23 * 15 - 1,
                ..place
            })
            .is_err()
        );
        assert!(
            place_grid(&PlaceParams {
                scratch: 0,
                ..place
            })
            .is_err()
        );
        assert!(
            place_grid(&PlaceParams {
                row_align: 0,
                ..place
            })
            .is_err()
        );

        let swiglu = SwigluParams {
            gate: A,
            up: A,
            weight: 0,
            codes: A,
            scales: A,
            rows: 1,
            width: 2304,
            in_stride: 2304,
            limit: 10.0,
        };
        assert_eq!(swiglu_grid(&swiglu).unwrap(), [1, 1, 1]);
        assert!(
            swiglu_grid(&SwigluParams {
                in_stride: 2303,
                ..swiglu
            })
            .is_err()
        );
        assert!(
            swiglu_grid(&SwigluParams {
                width: 2300,
                ..swiglu
            })
            .is_err()
        );
        assert!(
            swiglu_grid(&SwigluParams {
                limit: f32::NAN,
                ..swiglu
            })
            .is_err()
        );
    }

    /// The layout's rule, and its bound reached.
    #[test]
    fn expert_layout_follows_the_rule() {
        // Rows of two pairs: experts (0, 2), (2, 3), (0, 3).
        let l = expert_layout(&[0, 2, 2, 3, 0, 3], 5, 4);
        assert_eq!(l.row_of, [0, 4, 5, 8, 1, 9]);
        assert_eq!(l.group_end, [2, 4, 6, 10, 12]);
        assert_eq!(l.rows, 12);
        let l = expert_layout(&[1, 1, 1], 3, 1);
        assert_eq!((l.row_of, l.group_end), (vec![0, 1, 2], vec![0, 3, 3]));
        // One pair in each of three groups, aligned to 16: the bound.
        let l = expert_layout(&[0, 1, 2], 3, 16);
        assert_eq!(u64::from(l.rows), layout_rows_bound(3, 3, 16));
        assert_eq!(layout_rows_bound(0, 3, 16), 0);
        assert_eq!(layout_rows_bound(6, 2, 16), 21);
    }
}
