//! Launch wrappers for the executor's own kernels (`engine_ops.cu`). Every
//! operand is a device address; the callers own the buffers and their sizes.

use crate::module::KernelDir;

use crate::module::{Kernel, KernelModule};
use crate::{Gpu, Result, launch};

/// `EidolaQkvArgs`: the fused-QKV RoPE and KV-write kernel's parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct QkvArgs {
    pub qkv: u64,
    pub q_out: u64,
    pub pool: u64,
    pub positions: u64,
    pub kv_block: u64,
    pub kv_slot: u64,
    pub rope: u64,
    pub block_elems: u64,
    pub k_off: u64,
    pub v_off: u64,
    pub chunk_stride: u32,
    pub chunks: u32,
    pub q_heads_per_chunk: u32,
    pub kv_heads_per_chunk: u32,
}

const _: () = assert!(std::mem::size_of::<QkvArgs>() == 96);

/// Buffers one `eidola_copy_rows` launch may name.
pub const COPY_BUFFERS: usize = 8;

/// The status bit `eidola_copy_rows` raises for an item naming a buffer or a
/// row out of range (the item copies nothing).
pub const STATUS_BAD_INDEX: u32 = 4;

/// `EidolaCopyArgs`: one `eidola_copy_rows` launch. Buffer `b` is
/// `buf[b]`, holding `rows[b]` rows of `width` 32-bit words (0 rows for an
/// unused buffer); item `i` copies row `src_row[i]` of buffer `src_buf[i]`
/// to row `dst_row[i]` of buffer `dst_buf[i]`. The four item arrays are
/// device addresses of `items` words each.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CopyArgs {
    pub buf: [u64; COPY_BUFFERS],
    pub rows: [u64; COPY_BUFFERS],
    pub src_buf: u64,
    pub src_row: u64,
    pub dst_buf: u64,
    pub dst_row: u64,
    pub status: u64,
    pub width: u32,
    pub items: u32,
}

const _: () = assert!(std::mem::size_of::<CopyArgs>() == 176);

pub struct EngineOps {
    _module: KernelModule,
    embed: Kernel,
    rmsnorm: Kernel,
    quant: Kernel,
    add_f32: Kernel,
    add_bf16: Kernel,
    qkv: Kernel,
    swiglu_f32s: Kernel,
    swiglu_ue8m0: Kernel,
    router: Kernel,
    router_scores: Kernel,
    router_select: Kernel,
    permute_count: Kernel,
    permute: Kernel,
    gather_quant: Kernel,
    combine: Kernel,
    gather_bf16: Kernel,
    copy: Kernel,
}

impl EngineOps {
    pub fn load(gpu: &Gpu, dir: &KernelDir) -> Result<EngineOps> {
        EngineOps::from_module(KernelModule::load(gpu, dir, "engine_ops")?)
    }

    pub fn from_module(m: KernelModule) -> Result<EngineOps> {
        m.expect_image("engine_ops")?;
        let qkv = m.kernel("eidola_qkv_rope_kv")?;
        if qkv.meta().params_bytes as usize != std::mem::size_of::<QkvArgs>() {
            return Err(crate::CudaError::new(
                "eidola_qkv_rope_kv: argument struct size",
            ));
        }
        let copy = m.kernel("eidola_copy_rows")?;
        if copy.meta().params_bytes as usize != std::mem::size_of::<CopyArgs>() {
            return Err(crate::CudaError::new(
                "eidola_copy_rows: argument struct size",
            ));
        }
        Ok(EngineOps {
            copy,
            embed: m.kernel("eidola_embed")?,
            rmsnorm: m.kernel("eidola_rmsnorm_f32")?,
            quant: m.kernel("eidola_quant_fp8_f32scale")?,
            add_f32: m.kernel("eidola_add_f32")?,
            add_bf16: m.kernel("eidola_add_bf16")?,
            qkv,
            swiglu_f32s: m.kernel("eidola_swiglu_quant_fp8_f32scale")?,
            swiglu_ue8m0: m.kernel("eidola_swiglu_quant_fp8_ue8m0")?,
            router: m.kernel("eidola_router_topk")?,
            router_scores: m.kernel("eidola_router_scores")?,
            router_select: m.kernel("eidola_router_select")?,
            permute_count: m.kernel("eidola_moe_permute_count")?,
            permute: m.kernel("eidola_moe_permute")?,
            gather_quant: m.kernel("eidola_gather_quant_ue8m0")?,
            combine: m.kernel("eidola_moe_combine")?,
            gather_bf16: m.kernel("eidola_gather_rows_bf16")?,
            _module: m,
        })
    }
}

// SAFETY (every method): the argument lists match the kernels' signatures in
// `engine_ops.cu`; the device addresses are the caller's, sized as each
// kernel's comment there requires.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
impl EngineOps {
    pub unsafe fn embed(
        &self,
        gpu: &Gpu,
        out: u64,
        table: u64,
        tokens: u64,
        hidden: u32,
        rows: u32,
    ) -> Result<()> {
        require(hidden > 0, "embed: hidden 0")?;
        if rows == 0 {
            return Ok(());
        }
        unsafe { launch!(gpu, self.embed, [rows, 1, 1], out, table, tokens, hidden) }
    }

    /// RMSNorm of `rows` f32 rows (`x_stride` apart) into f32 and/or BF16
    /// outputs (0 for none).
    pub unsafe fn rmsnorm(
        &self,
        gpu: &Gpu,
        out: u64,
        out_bf16: u64,
        x: u64,
        x_stride: u32,
        weight: u64,
        hidden: u32,
        eps: f32,
        rows: u32,
    ) -> Result<()> {
        require(
            hidden > 0 && x_stride >= hidden,
            "rmsnorm: row stride below the width",
        )?;
        if rows == 0 {
            return Ok(());
        }
        unsafe {
            launch!(
                gpu,
                self.rmsnorm,
                [rows, 1, 1],
                out,
                out_bf16,
                x,
                x_stride,
                weight,
                hidden,
                eps
            )
        }
    }

    pub unsafe fn quant_fp8(
        &self,
        gpu: &Gpu,
        q: u64,
        sf: u64,
        x: u64,
        rows: u32,
        k: u32,
        m_pad: u32,
    ) -> Result<()> {
        let grid = quant_grid(rows, k, m_pad, q, x)?;
        if grid[0] == 0 {
            return Ok(());
        }
        unsafe { launch!(gpu, self.quant, grid, q, sf, x, k, m_pad) }
    }

    pub unsafe fn add_f32(&self, gpu: &Gpu, h: u64, d: u64, n: u64) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let blocks = n.div_ceil(256).min(65_535) as u32;
        unsafe { launch!(gpu, self.add_f32, [blocks, 1, 1], h, d, n) }
    }

    pub unsafe fn add_bf16(
        &self,
        gpu: &Gpu,
        h: u64,
        d: u64,
        hidden: u32,
        d_stride: u32,
        rows: u32,
    ) -> Result<()> {
        require(d_stride >= hidden, "add_bf16: row stride below the width")?;
        if rows == 0 {
            return Ok(());
        }
        unsafe { launch!(gpu, self.add_bf16, [rows, 1, 1], h, d, hidden, d_stride) }
    }

    pub unsafe fn qkv_rope_kv(&self, gpu: &Gpu, args: QkvArgs, tokens: u32) -> Result<()> {
        let grid = qkv_grid(tokens, &args)?;
        if grid[0] == 0 {
            return Ok(());
        }
        unsafe { launch!(gpu, self.qkv, grid, args) }
    }

    pub unsafe fn swiglu_quant_f32scale(
        &self,
        gpu: &Gpu,
        q: u64,
        sf: u64,
        gu: u64,
        rows: u32,
        inter: u32,
        m_pad: u32,
    ) -> Result<()> {
        let launch = swiglu_f32scale_grid(rows, inter, m_pad, q, gu)?;
        if launch.items == 0 {
            return Ok(());
        }
        // The kernel walks the items of `launch.rows` rows.
        unsafe {
            launch!(
                gpu,
                self.swiglu_f32s,
                launch.grid,
                q,
                sf,
                gu,
                inter,
                m_pad,
                launch.rows
            )
        }
    }

    /// SwiGLU + UE8M0 quantization of the expert layout's routed rows: the
    /// `tokens * top_k` rows `row_of` names, in a layout of `rows` rows.
    /// Padding rows are left as they are.
    pub unsafe fn swiglu_quant_ue8m0(
        &self,
        gpu: &Gpu,
        q: u64,
        sf: u64,
        gu: u64,
        row_of: u64,
        tokens: u32,
        top_k: u32,
        rows: u32,
        inter: u32,
        rows4: u32,
    ) -> Result<()> {
        swiglu_operands(q, gu, rows, rows4)?;
        let grid = swiglu_grid(tokens, top_k, inter, rows)?;
        if grid[0] == 0 {
            return Ok(());
        }
        unsafe {
            launch!(
                gpu,
                self.swiglu_ue8m0,
                grid,
                q,
                sf,
                gu,
                row_of,
                inter,
                rows4
            )
        }
    }

    /// Router top-k of `tokens` rows, in the executor's form for that count
    /// ([`executor_router`]). `scores` is the split form's scratch:
    /// [`router_scores_len`] f32.
    pub unsafe fn router_topk(
        &self,
        gpu: &Gpu,
        ids: u64,
        weights: u64,
        scores: u64,
        x: u64,
        router: u64,
        bias: u64,
        tokens: u32,
        hidden: u32,
        experts: u32,
        top_k: u32,
        scaling: f32,
    ) -> Result<()> {
        unsafe {
            self.router_topk_form(
                gpu,
                executor_router(tokens),
                ids,
                weights,
                scores,
                x,
                router,
                bias,
                tokens,
                hidden,
                experts,
                top_k,
                scaling,
            )
        }
    }

    /// Router top-k in a given form. Both forms compute the same ids and
    /// weights bit for bit; the executor goes through [`Self::router_topk`],
    /// the tests and the kernel bench through this. Only the split form
    /// reads `scores`.
    pub unsafe fn router_topk_form(
        &self,
        gpu: &Gpu,
        form: RouterForm,
        ids: u64,
        weights: u64,
        scores: u64,
        x: u64,
        router: u64,
        bias: u64,
        tokens: u32,
        hidden: u32,
        experts: u32,
        top_k: u32,
        scaling: f32,
    ) -> Result<()> {
        router_preconditions(scores, x, router, hidden, experts, top_k)?;
        match form {
            RouterForm::PerToken => {
                let grid = router_grid(tokens)?;
                if tokens == 0 {
                    return Ok(());
                }
                unsafe {
                    launch!(
                        gpu,
                        self.router,
                        grid,
                        ids,
                        weights,
                        x,
                        router,
                        bias,
                        hidden,
                        experts,
                        top_k,
                        scaling
                    )
                }
            }
            RouterForm::Split => {
                let grid = router_scores_grid(tokens, experts)?;
                if tokens == 0 {
                    return Ok(());
                }
                unsafe {
                    launch!(
                        gpu,
                        self.router_scores,
                        grid,
                        scores,
                        x,
                        router,
                        bias,
                        tokens,
                        hidden,
                        experts
                    )?;
                    launch!(
                        gpu,
                        self.router_select,
                        router_select_grid(tokens),
                        ids,
                        weights,
                        scores,
                        tokens,
                        experts,
                        top_k,
                        scaling
                    )
                }
            }
        }
    }

    /// Expert-major placement of `tokens × top_k` routed pairs
    /// ([`expert_placement`] is its host form): `row_of` per pair, and the
    /// psum layout's grouped layout, 256 words, every run within
    /// `rows_bound` rows. `scratch` is [`PERMUTE_SCRATCH_WORDS`] words: above
    /// one block, `eidola_moe_permute_count` writes every block's counts
    /// there and `eidola_moe_permute` (the same grid) reads them, so no
    /// block waits for another; with one block the placement is one launch
    /// and reads no scratch.
    pub unsafe fn moe_permute(
        &self,
        gpu: &Gpu,
        grouped_layout: u64,
        row_of: u64,
        topk_ids: u64,
        scratch: u64,
        tokens: u32,
        top_k: u32,
        rows_bound: u32,
    ) -> Result<()> {
        let grid = permute_grid(tokens, top_k, rows_bound, scratch)?;
        if grid[0] > 1 {
            unsafe {
                launch!(
                    gpu,
                    self.permute_count,
                    grid,
                    scratch,
                    topk_ids,
                    tokens,
                    top_k
                )?;
            }
        }
        unsafe {
            launch!(
                gpu,
                self.permute,
                grid,
                grouped_layout,
                row_of,
                topk_ids,
                scratch,
                tokens,
                top_k
            )
        }
    }

    /// FP8 + UE8M0 quantization of each routed (token, slot) pair's token
    /// row into the row `row_of` names, in a layout of `rows` rows. Padding
    /// rows are left as they are.
    pub unsafe fn gather_quant_ue8m0(
        &self,
        gpu: &Gpu,
        a: u64,
        sf: u64,
        x: u64,
        row_of: u64,
        tokens: u32,
        top_k: u32,
        rows: u32,
        k: u32,
        rows4: u32,
    ) -> Result<()> {
        require(
            sfa_layout_ok(rows, rows4),
            "gather_quant: the scale layout must cover the rows",
        )?;
        require(
            x.is_multiple_of(16) && a.is_multiple_of(4),
            "gather_quant: token rows must be 16-byte and codes 4-byte aligned",
        )?;
        let grid = gather_grid(tokens, top_k, k, rows)?;
        if grid[0] == 0 {
            return Ok(());
        }
        unsafe {
            launch!(
                gpu,
                self.gather_quant,
                grid,
                a,
                sf,
                x,
                row_of,
                top_k,
                k,
                rows4
            )
        }
    }

    pub unsafe fn moe_combine(
        &self,
        gpu: &Gpu,
        out: u64,
        d: u64,
        row_of: u64,
        topk_w: u64,
        tokens: u32,
        hidden: u32,
        top_k: u32,
    ) -> Result<()> {
        require(
            (1..=crate::support::MAX_TOP_K).contains(&(top_k as usize)),
            "combine: 1..=8 experts per token",
        )?;
        require(
            out.is_multiple_of(16) && d.is_multiple_of(16),
            "combine: output and expert rows must be 16-byte aligned",
        )?;
        let grid = combine_grid(tokens, hidden)?;
        if grid[0] == 0 {
            return Ok(());
        }
        unsafe {
            launch!(
                gpu,
                self.combine,
                grid,
                out,
                d,
                row_of,
                topk_w,
                hidden,
                top_k
            )
        }
    }

    pub unsafe fn gather_rows_bf16(
        &self,
        gpu: &Gpu,
        out: u64,
        x: u64,
        rows: u64,
        n: u32,
        hidden: u32,
    ) -> Result<()> {
        require(hidden > 0, "gather_rows: hidden 0")?;
        if n == 0 {
            return Ok(());
        }
        unsafe { launch!(gpu, self.gather_bf16, [n, 1, 1], out, x, rows, hidden) }
    }

    /// `args.items` row copies of `args.width` words ([`CopyArgs`]). The
    /// kernel bounds every buffer and row index against `args.rows` and
    /// raises [`STATUS_BAD_INDEX`] in `args.status` for an item out of
    /// range; the item arrays themselves must hold `items` words each, and
    /// no item's destination row may be another item's source or
    /// destination.
    pub unsafe fn copy_rows(&self, gpu: &Gpu, args: CopyArgs) -> Result<()> {
        let grid = copy_grid(args.items, args.width)?;
        if grid[0] == 0 {
            return Ok(());
        }
        unsafe { launch!(gpu, self.copy, grid, args) }
    }
}

/// Threads per block of `eidola_copy_rows`: one per word of a row.
pub const COPY_THREADS: u32 = 256;

/// The row copy's grid: one block row per item, and one thread per word of
/// the row.
pub fn copy_grid(items: u32, width: u32) -> Result<[u32; 3]> {
    require(width > 0, "copy_rows: rows of no words")?;
    require(
        items <= MAX_GRID_X,
        "copy_rows: too many items for one launch",
    )?;
    let blocks = width.div_ceil(COPY_THREADS);
    require(blocks <= 65_535, "copy_rows: rows too wide for one launch")?;
    Ok([items, blocks, 1])
}

/// The driver's grid limit in x.
const MAX_GRID_X: u32 = (1 << 31) - 1;

/// Blocks per cluster of the one-token router (its launch contract records
/// it, and the launch sets it from there): block rank `b` scores experts
/// `32b .. 32b + 31`.
pub const ROUTER_CLUSTER: u32 = 8;
/// Threads per block of `eidola_router_topk`: one warp per expert.
pub const ROUTER_THREADS: u32 = 1024;
/// The widest row the router takes.
pub const ROUTER_MAX_HIDDEN: u32 = 4096;
/// The router's row width is whole chunks of this many elements (the split
/// form streams rows through shared memory a chunk at a time).
pub const ROUTER_CHUNK: u32 = 128;
/// Tokens per block of `eidola_router_scores`.
pub const ROUTER_SCORES_TILE: u32 = 8;
/// Experts per block of `eidola_router_scores`.
pub const ROUTER_SCORES_EXPERTS: u32 = 16;
/// Threads per block of `eidola_router_scores`: four warps, four experts each.
pub const ROUTER_SCORES_THREADS: u32 = 128;
/// Dynamic shared memory of `eidola_router_scores`: eight stages of one chunk
/// of the tile's rows (f32) and the block's weight rows (BF16).
pub const ROUTER_SCORES_SMEM: u32 =
    8 * ROUTER_CHUNK * (ROUTER_SCORES_TILE * 4 + ROUTER_SCORES_EXPERTS * 2);
/// Tokens per block of `eidola_router_select`: one warp each.
pub const ROUTER_SELECT_WARPS: u32 = 4;

/// Which router kernels a launch runs. Both give identical ids and weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouterForm {
    /// `eidola_router_topk`: one cluster of [`ROUTER_CLUSTER`] blocks per
    /// token, which scores the token's experts and selects.
    PerToken,
    /// `eidola_router_scores` then `eidola_router_select`: every (token,
    /// expert)'s score into a scratch buffer, from blocks of
    /// [`ROUTER_SCORES_TILE`] tokens × [`ROUTER_SCORES_EXPERTS`] experts, then
    /// one warp per token selecting.
    Split,
}

/// The most tokens the executor routes in the one-token form; above it, the
/// split form.
///
/// On a B300 (`moe_kernel_bench`, MiMo-V2.6-Flash shapes, µs per layer) the
/// one-token form costs 20.5 at 1 token, 20.9 at 8, 21.27 at 9, 21.26 at 10
/// and 21.37 at 12; in a second run 22.27 at 14 and 22.36 at 15, then 37.70
/// at 16 (38.91 in the first), 57.3 at 32 and 93.3 at 64, growing with the
/// token count from there: it steps up once its clusters no longer all run at
/// once. The split form costs 25.6 at 1 token, 26.6 at 8, 27.52 at 9, 26.83
/// at 10 and 27.54 at 12; 26.64 at 14 and 15 and 26.63 at 16 in the second
/// run; 27.6 at 64, 49.7 at 256 and 825 at 8192, below the one-token form from
/// 16 tokens on. Through 15 tokens the one-token form saves 4–6 µs a layer,
/// about a quarter of a millisecond of a decode step over the 47 expert
/// layers, so the bound is the last count before the step.
pub const ROUTER_PER_TOKEN_MAX: u32 = 15;

/// The form the executor routes `tokens` rows in. A function of the token
/// count alone, so a captured decode rung, its replays and the same step run
/// eagerly all launch the same kernels.
pub fn executor_router(tokens: u32) -> RouterForm {
    if tokens <= ROUTER_PER_TOKEN_MAX {
        RouterForm::PerToken
    } else {
        RouterForm::Split
    }
}

/// What a router call must satisfy whichever form runs it, checked before
/// either form launches: past these, only each form's own grid limit can
/// refuse a call, so whether an argument is accepted never depends on the
/// form a token count picks. `scores` is the split form's scratch; the
/// one-token form does not read it, but the executor passes the same scratch
/// to both.
pub fn router_preconditions(
    scores: u64,
    x: u64,
    router: u64,
    hidden: u32,
    experts: u32,
    top_k: u32,
) -> Result<()> {
    require(
        (1..=ROUTER_MAX_HIDDEN).contains(&hidden)
            && hidden.is_multiple_of(ROUTER_CHUNK)
            && (1..=crate::support::EXPERTS).contains(&(experts as usize))
            && (2..=crate::support::MAX_TOP_K).contains(&(top_k as usize))
            && top_k <= experts,
        // The kernel always renormalizes the selection; the reference does
        // not for a single expert, so one expert per token is not served.
        "router: hidden whole 128-wide chunks up to 4096, at most 256 experts and 2..=8 per token",
    )?;
    require(
        x.is_multiple_of(16) && router.is_multiple_of(16),
        "router: rows and weights must be 16-byte aligned",
    )?;
    require(
        scores.is_multiple_of(4),
        "router: the scores scratch must be 4-byte aligned",
    )
}

/// The one-token form's grid: one cluster of [`ROUTER_CLUSTER`] blocks per
/// token. The split form has two grids ([`router_scores_grid`],
/// [`router_select_grid`]).
pub fn router_grid(tokens: u32) -> Result<[u32; 3]> {
    let blocks = tokens
        .checked_mul(ROUTER_CLUSTER)
        .filter(|&b| b <= MAX_GRID_X)
        .ok_or_else(|| crate::CudaError::new("router: too many tokens for one launch"))?;
    Ok([blocks, 1, 1])
}

/// The driver's grid limit in y.
const MAX_GRID_Y: u32 = 65_535;

/// `eidola_router_scores`' grid: (expert blocks of [`ROUTER_SCORES_EXPERTS`],
/// token tiles of [`ROUTER_SCORES_TILE`]), the last of each partial.
pub fn router_scores_grid(tokens: u32, experts: u32) -> Result<[u32; 3]> {
    let tiles = tokens.div_ceil(ROUTER_SCORES_TILE);
    require(
        tiles <= MAX_GRID_Y,
        "router: too many tokens for one launch",
    )?;
    Ok([experts.div_ceil(ROUTER_SCORES_EXPERTS), tiles, 1])
}

/// `eidola_router_select`'s grid: one warp per token,
/// [`ROUTER_SELECT_WARPS`] to a block.
pub fn router_select_grid(tokens: u32) -> [u32; 3] {
    [tokens.div_ceil(ROUTER_SELECT_WARPS), 1, 1]
}

/// f32 elements of the split form's scratch for `tokens` rows: every
/// (token, expert)'s choice, then every score. `None` on overflow.
pub fn router_scores_len(tokens: usize, experts: usize) -> Option<usize> {
    tokens.checked_mul(experts)?.checked_mul(2)
}

/// Elements of a token's row each `eidola_moe_combine` block covers: 128
/// threads of 8 consecutive elements.
pub const COMBINE_WIDTH: u32 = 1024;

/// The combine's grid: one block per token and [`COMBINE_WIDTH`] elements of
/// the row, which must be whole.
pub fn combine_grid(tokens: u32, hidden: u32) -> Result<[u32; 3]> {
    require(
        hidden > 0 && hidden.is_multiple_of(COMBINE_WIDTH) && hidden / COMBINE_WIDTH <= 65_535,
        "combine: hidden must be whole 1024-wide pieces",
    )?;
    require(
        tokens <= MAX_GRID_X,
        "combine: too many tokens for one launch",
    )?;
    Ok([tokens, hidden / COMBINE_WIDTH, 1])
}

/// Threads per block of `eidola_qkv_rope_kv`.
pub const QKV_THREADS: u32 = 256;

/// Consecutive output elements each `eidola_qkv_rope_kv` thread computes
/// (one 16-byte BF16 vector).
pub const QKV_VEC: u32 = 8;

/// The fused-QKV kernel's grid: one thread per [`QKV_VEC`] consecutive
/// output elements of a token (its chunks' Q and K heads of 192 and V heads
/// of 128, all whole vectors), one block row per token. Every load and store
/// is a 16-byte vector, so the QKV rows, the Q output, the pool and the RoPE
/// table must be 16-byte aligned, and the chunk stride, the block's elements
/// and the K and V offsets whole vectors.
pub fn qkv_grid(tokens: u32, args: &QkvArgs) -> Result<[u32; 3]> {
    let per_chunk = args
        .q_heads_per_chunk
        .checked_add(args.kv_heads_per_chunk)
        .and_then(|h| h.checked_mul(192))
        .and_then(|qk| {
            args.kv_heads_per_chunk
                .checked_mul(128)
                .and_then(|v| qk.checked_add(v))
        });
    let per_chunk = per_chunk.filter(|&n| n > 0 && args.chunk_stride >= n);
    require(
        args.chunks > 0 && per_chunk.is_some(),
        "qkv_rope_kv: chunk stride below its heads",
    )?;
    let vec = u64::from(QKV_VEC);
    require(
        [args.qkv, args.q_out, args.pool, args.rope]
            .iter()
            .all(|a| a.is_multiple_of(16))
            && [
                u64::from(args.chunk_stride),
                args.block_elems,
                args.k_off,
                args.v_off,
            ]
            .iter()
            .all(|n| n.is_multiple_of(vec)),
        "qkv_rope_kv: buffers must be 16-byte aligned and strides and offsets whole 8-element vectors",
    )?;
    let blocks = per_chunk
        .and_then(|n| n.checked_mul(args.chunks))
        .map(|n| (n / QKV_VEC).div_ceil(QKV_THREADS))
        .filter(|&b| b <= 65_535)
        .ok_or_else(|| crate::CudaError::new("qkv_rope_kv: too many heads per token"))?;
    require(
        tokens <= MAX_GRID_X,
        "qkv_rope_kv: too many tokens for one launch",
    )?;
    Ok([tokens, blocks, 1])
}

/// Every routed (token, slot) pair lands in its own row of the expert
/// layout, so its `rows` must hold them all: the pair count.
fn routed_pairs(tokens: u32, top_k: u32, rows: u32) -> Result<u32> {
    require(
        (1..=crate::support::MAX_TOP_K).contains(&(top_k as usize)),
        "expert rows: 1..=8 experts per token",
    )?;
    let pairs = tokens
        .checked_mul(top_k)
        .filter(|&p| p <= MAX_GRID_X)
        .ok_or_else(|| crate::CudaError::new("expert rows: too many pairs for one launch"))?;
    require(
        pairs <= rows,
        "expert rows: the layout holds fewer rows than pairs",
    )?;
    Ok(pairs)
}

/// Elements of a row one warp of either SwiGLU kernel covers per item: two
/// 128-wide groups, 32 lanes of 8 consecutive elements (`kSwigluPiece`).
pub const SWIGLU_PIECE: u32 = 256;

/// Warps per block of both SwiGLU kernels (`kSwigluWarps`).
pub const SWIGLU_WARPS: u32 = 4;

/// Most warps one SwiGLU launch runs (`kSwigluMaxWarps`): 8 blocks of
/// [`SWIGLU_WARPS`] on each of 148 SMs, which such a part holds at once (the
/// kernels' `__launch_bounds__` keep 8 blocks within an SM's registers), so
/// every warp starts at once and walks its share of the items.
pub const SWIGLU_MAX_WARPS: u32 = 148 * 8 * SWIGLU_WARPS;

/// Most items one SwiGLU launch walks: a warp's item index plus the grid's
/// warps stays within `u32`.
const SWIGLU_MAX_ITEMS: u32 = i32::MAX as u32;

/// A SwiGLU launch: its grid, the rows (the dense form) or routed pairs
/// (the expert form) the kernel is given, and the items (row pieces of
/// [`SWIGLU_PIECE`] elements) its warps walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwigluLaunch {
    pub grid: [u32; 3],
    pub rows: u32,
    pub items: u32,
}

/// One warp per item up to [`SWIGLU_MAX_WARPS`], whole blocks, over `rows`
/// rows of `pieces` items.
fn swiglu_launch(rows: u32, pieces: u32, what: &str) -> Result<SwigluLaunch> {
    let items = rows
        .checked_mul(pieces)
        .filter(|&n| n <= SWIGLU_MAX_ITEMS)
        .ok_or_else(|| crate::CudaError::new(what))?;
    Ok(SwigluLaunch {
        grid: [items.min(SWIGLU_MAX_WARPS).div_ceil(SWIGLU_WARPS), 1, 1],
        rows,
        items,
    })
}

/// The dense SwiGLU's launch (f32 scales): `rows × inter / SWIGLU_PIECE`
/// items, the intermediate whole pieces; gate/up rows 16-byte and codes
/// 8-byte aligned.
pub fn swiglu_f32scale_grid(
    rows: u32,
    inter: u32,
    m_pad: u32,
    q: u64,
    gu: u64,
) -> Result<SwigluLaunch> {
    require(
        inter > 0 && inter.is_multiple_of(SWIGLU_PIECE) && m_pad >= rows,
        "swiglu (f32 scales): intermediate must be whole 256-wide pieces and m_pad cover the rows",
    )?;
    require(
        gu.is_multiple_of(16) && q.is_multiple_of(8),
        "swiglu (f32 scales): gate/up rows must be 16-byte and codes 8-byte aligned",
    )?;
    swiglu_launch(
        rows,
        inter / SWIGLU_PIECE,
        "swiglu (f32 scales): too many rows for one launch",
    )
}

/// Elements of a routed row each `eidola_swiglu_quant_fp8_ue8m0` block
/// covers: two 512-wide scale words, 128 threads of 8 consecutive elements.
pub const SWIGLU_WIDTH: u32 = 1024;

/// The expert SwiGLU's grid: one block per routed (token, slot) pair and
/// [`SWIGLU_WIDTH`] elements of the intermediate, which must be whole.
pub fn swiglu_grid(tokens: u32, top_k: u32, inter: u32, rows: u32) -> Result<[u32; 3]> {
    require(
        inter > 0 && inter.is_multiple_of(SWIGLU_WIDTH) && inter / SWIGLU_WIDTH <= 65_535,
        "swiglu (UE8M0): intermediate must be whole 1024-wide pieces",
    )?;
    let pairs = routed_pairs(tokens, top_k, rows)?;
    Ok([pairs, inter / SWIGLU_WIDTH, 1])
}

/// The expert SwiGLU's operands: the scale layout covers the rows, gate/up
/// rows are 16-byte and codes 8-byte aligned.
pub fn swiglu_operands(q: u64, gu: u64, rows: u32, rows4: u32) -> Result<()> {
    require(
        sfa_layout_ok(rows, rows4),
        "swiglu (UE8M0): the scale layout must cover the rows",
    )?;
    require(
        gu.is_multiple_of(16) && q.is_multiple_of(8),
        "swiglu (UE8M0): gate/up rows must be 16-byte and codes 8-byte aligned",
    )
}

/// Warps per block of `eidola_quant_fp8_f32scale`, one per 128-wide group.
pub const QUANT_WARPS: u32 = 4;

/// The f32-scale activation quantization's grid: one block per row and
/// [`QUANT_WARPS`] 128-wide groups of K, which must be whole groups; rows
/// 16-byte and codes 4-byte aligned, and `m_pad` covering the rows.
pub fn quant_grid(rows: u32, k: u32, m_pad: u32, q: u64, x: u64) -> Result<[u32; 3]> {
    require(
        k > 0 && k.is_multiple_of(128) && m_pad >= rows,
        "quant_fp8: K must be whole 128-wide groups and m_pad cover the rows",
    )?;
    require(
        x.is_multiple_of(16) && q.is_multiple_of(4),
        "quant_fp8: rows must be 16-byte and codes 4-byte aligned",
    )?;
    require(
        rows <= MAX_GRID_X,
        "quant_fp8: too many rows for one launch",
    )?;
    let blocks = (k / 128).div_ceil(QUANT_WARPS);
    require(blocks <= 65_535, "quant_fp8: K too wide for one launch")?;
    Ok([rows, blocks, 1])
}

/// Routed pairs per block of `eidola_moe_permute` (`kPermuteChunk`), up to
/// [`PERMUTE_MAX_BLOCKS`].
pub const PERMUTE_CHUNK: u32 = 512;

/// Most blocks of `eidola_moe_permute` (`kPermuteMaxBlocks`): the scratch
/// holds a count per (expert, block), each expert's read in 16-byte loads.
pub const PERMUTE_MAX_BLOCKS: u32 = 128;

/// Most pairs `eidola_moe_permute` places in one launch of one block (640
/// tokens of 8). On a B300 the one block costs 8.2 µs at 4,104 pairs and
/// 14.3 at 8,192, the two-launch form a flat 10.2–10.3 µs from 1,024 pairs
/// to 8,192: they cross near 5,500 pairs (interpolating the one block
/// linearly), so the bound sits just below, where one block still wins.
pub const PERMUTE_ONE_BLOCK_PAIRS: u32 = 5120;

/// Words of `eidola_moe_permute`'s scratch: a count per (expert, block).
pub const PERMUTE_SCRATCH_WORDS: usize = crate::support::EXPERTS * PERMUTE_MAX_BLOCKS as usize;

/// The placement's grid: one block up to [`PERMUTE_ONE_BLOCK_PAIRS`] (one
/// launch; it writes the grouped layout even with no pairs), else one block
/// per [`PERMUTE_CHUNK`] pairs, at most [`PERMUTE_MAX_BLOCKS`] (a count
/// launch, then the placement).
/// The layout's rows must hold every run, and the scratch be 16-byte aligned.
pub fn permute_grid(tokens: u32, top_k: u32, rows_bound: u32, scratch: u64) -> Result<[u32; 3]> {
    require(
        (1..=crate::support::MAX_TOP_K).contains(&(top_k as usize)),
        "permute: 1..=8 experts per token",
    )?;
    let pairs = tokens
        .checked_mul(top_k)
        .ok_or_else(|| crate::CudaError::new("permute: too many pairs"))?;
    require(
        rows_bound as usize >= psum_rows(tokens as usize, top_k as usize)
            && i32::try_from(rows_bound).is_ok(),
        "permute: the layout's rows must hold every run",
    )?;
    require(
        scratch != 0 && scratch.is_multiple_of(16),
        "permute: scratch must be 16-byte aligned",
    )?;
    let blocks = if pairs <= PERMUTE_ONE_BLOCK_PAIRS {
        1
    } else {
        pairs.div_ceil(PERMUTE_CHUNK).min(PERMUTE_MAX_BLOCKS)
    };
    Ok([blocks, 1, 1])
}

/// The expert gather's grid: one block per token and 512-wide word of `k`,
/// which stores the word into the rows of all the token's pairs (each
/// quantized once, not once per pair).
pub fn gather_grid(tokens: u32, top_k: u32, k: u32, rows: u32) -> Result<[u32; 3]> {
    require(
        k > 0 && k.is_multiple_of(512) && k / 512 <= 65_535,
        "gather_quant: K must be whole 512-wide words",
    )?;
    routed_pairs(tokens, top_k, rows)?;
    Ok([tokens, k / 512, 1])
}

/// Rows per block of the grouped GEMMs: in the psum layout every expert's
/// run starts on a multiple of it.
pub const EXPERT_BLOCK_ROWS: usize = crate::moe_gemm::BLOCK_M as usize;

/// Rows the psum expert layout can need for `tokens × top_k` routed pairs
/// over the 256 experts: every pair, plus up to 127 padding rows for each
/// expert some pair reaches, rounded up to a whole block.
pub fn psum_rows(tokens: usize, top_k: usize) -> usize {
    let n = tokens * top_k;
    let experts = crate::support::EXPERTS;
    (n + n.min(experts) * (EXPERT_BLOCK_ROWS - 1)).div_ceil(EXPERT_BLOCK_ROWS) * EXPERT_BLOCK_ROWS
}

/// `eidola_moe_permute` on the host: the placement it computes for
/// `topk_ids` (the router's ids, `top_k` per token). Returns `row_of` (-1 for
/// an id outside the 256 experts, which the kernel leaves unwritten) and the
/// psum layout's grouped layout (256 words): expert `e`'s run starts at the
/// previous run's end rounded up to [`EXPERT_BLOCK_ROWS`], and its word is
/// the end of its pairs. Pairs keep their order within an expert.
pub fn expert_placement(topk_ids: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let experts = crate::support::EXPERTS;
    let expert = |id: i32| usize::try_from(id).ok().filter(|&e| e < experts);
    let mut counts = vec![0usize; experts];
    for &id in topk_ids {
        if let Some(e) = expert(id) {
            counts[e] += 1;
        }
    }
    let mut next = vec![0usize; experts];
    let mut grouped = vec![0i32; experts];
    let mut start = 0usize;
    for e in 0..experts {
        next[e] = start;
        grouped[e] = i32::try_from(start + counts[e]).expect("an expert layout within i32 rows");
        start += counts[e].div_ceil(EXPERT_BLOCK_ROWS) * EXPERT_BLOCK_ROWS;
    }
    let row_of = topk_ids
        .iter()
        .map(|&id| match expert(id) {
            Some(e) => {
                next[e] += 1;
                i32::try_from(next[e] - 1).expect("an expert layout within i32 rows")
            }
            None => -1,
        })
        .collect();
    (row_of, grouped)
}

fn require(ok: bool, what: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(crate::CudaError::new(what))
    }
}

/// The packed UE8M0 scale layout (`sfa_index` in `engine_ops.cu`):
/// `[K/512][rows4]`.
fn sfa_layout_ok(rows: u32, rows4: u32) -> bool {
    rows4 >= rows && rows4.is_multiple_of(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one-token form: one cluster per token; the grid is the token
    /// count times the cluster, a whole number of clusters.
    #[test]
    fn per_token_router_grid_is_one_cluster_per_token() {
        for tokens in [0u32, 1, 2, 7, 64, 128, 513, 8192] {
            let g = router_grid(tokens).unwrap();
            assert_eq!(g, [tokens * ROUTER_CLUSTER, 1, 1], "{tokens}");
        }
        assert_eq!(
            ROUTER_THREADS / 32 * ROUTER_CLUSTER,
            256,
            "one warp per expert"
        );
        assert!(router_grid(u32::MAX / ROUTER_CLUSTER + 1).is_err());
        assert!(router_grid((1 << 31) / ROUTER_CLUSTER).is_err());
        assert!(router_grid((1 << 31) / ROUTER_CLUSTER - 1).is_ok());
    }

    /// The split form's scores grid covers every (token, expert) pair once:
    /// whole expert blocks and token tiles, the last of each partial, none
    /// empty; its selection grid has a warp for every token and no block
    /// without one. The scratch holds a choice and a score per pair.
    #[test]
    fn split_router_grids_cover_every_pair_once() {
        assert_eq!(
            ROUTER_SCORES_THREADS / 32 * 4,
            ROUTER_SCORES_EXPERTS,
            "four experts per warp"
        );
        assert_eq!(
            ROUTER_SCORES_TILE * 4,
            32,
            "one chain per lane after the butterflies"
        );
        assert_eq!(ROUTER_SCORES_SMEM, 64 * 1024);
        for experts in [1u32, 15, 16, 17, 255, 256] {
            for tokens in [0u32, 1, 2, 7, 8, 9, 15, 16, 17, 64, 128, 513, 2048, 8192] {
                let g = router_scores_grid(tokens, experts).unwrap();
                assert_eq!(g[2], 1);
                assert!(g[0] * ROUTER_SCORES_EXPERTS >= experts, "{experts}");
                assert!((g[0] - 1) * ROUTER_SCORES_EXPERTS < experts, "{experts}");
                assert!(g[1] * ROUTER_SCORES_TILE >= tokens, "{tokens}");
                assert!(
                    g[1] == 0 || (g[1] - 1) * ROUTER_SCORES_TILE < tokens,
                    "{tokens}: an empty tile"
                );
                let s = router_select_grid(tokens);
                assert_eq!(s[1..], [1, 1]);
                assert!(s[0] * ROUTER_SELECT_WARPS >= tokens, "{tokens}");
                assert!(
                    s[0] == 0 || (s[0] - 1) * ROUTER_SELECT_WARPS < tokens,
                    "{tokens}: a block without a token"
                );
            }
        }
        assert_eq!(router_scores_grid(64, 256).unwrap(), [16, 8, 1]);
        assert_eq!(router_scores_grid(513, 256).unwrap(), [16, 65, 1]);
        assert_eq!(router_select_grid(513), [129, 1, 1]);
        let last = 65_535 * ROUTER_SCORES_TILE;
        assert!(router_scores_grid(last, 256).is_ok());
        assert!(router_scores_grid(last + 1, 256).is_err(), "past the grid");
        assert_eq!(router_scores_len(64, 256), Some(2 * 64 * 256));
        assert_eq!(router_scores_len(usize::MAX / 2, 256), None);
    }

    /// Every router precondition holds whichever form runs: the split form's
    /// scratch alignment too, which the one-token form does not read, so a
    /// misaligned scratch is refused at every token count before any launch.
    #[test]
    fn router_preconditions_do_not_depend_on_the_form() {
        let ok = |scores, x, router, hidden, experts, top_k| {
            router_preconditions(scores, x, router, hidden, experts, top_k).is_ok()
        };
        assert!(ok(0, 0, 0, 4096, 256, 8));
        assert!(ok(4, 16, 32, 128, 2, 2));
        assert!(!ok(2, 0, 0, 4096, 256, 8), "scores scratch unaligned");
        assert!(!ok(0, 8, 0, 4096, 256, 8), "rows unaligned");
        assert!(!ok(0, 0, 8, 4096, 256, 8), "weights unaligned");
        assert!(!ok(0, 0, 0, 4097, 256, 8), "hidden past the staged row");
        assert!(!ok(0, 0, 0, 4000, 256, 8), "hidden not whole chunks");
        assert!(!ok(0, 0, 0, 0, 256, 8), "no hidden");
        assert!(!ok(0, 0, 0, 4096, 257, 8), "experts");
        assert!(!ok(0, 0, 0, 4096, 256, 9), "top_k");
        assert!(!ok(0, 0, 0, 4096, 256, 1), "one expert per token");
        assert!(!ok(0, 0, 0, 4096, 4, 8), "top_k past the experts");
    }

    /// The executor routes up to [`ROUTER_PER_TOKEN_MAX`] tokens in the
    /// one-token form and more in the split form: every rung of the decode
    /// ladder, and of each drafted ladder (rows of `1 + width` tokens), runs
    /// the form its token count picks, the same whether captured or eager.
    #[test]
    fn executor_router_picks_the_form_by_token_count() {
        assert_eq!(ROUTER_PER_TOKEN_MAX, 15);
        for tokens in 0..=ROUTER_PER_TOKEN_MAX {
            assert_eq!(executor_router(tokens), RouterForm::PerToken, "{tokens}");
        }
        for tokens in [16u32, 17, 64, 513, 8192, u32::MAX] {
            assert_eq!(executor_router(tokens), RouterForm::Split, "{tokens}");
        }
        // Decode rungs of one token per row; drafted rungs of four (width 3).
        let form = |rows: u32, width: u32| executor_router(rows * (1 + width));
        for rows in [1, 2, 4, 8] {
            assert_eq!(form(rows, 0), RouterForm::PerToken, "{rows}");
        }
        assert_eq!(form(16, 0), RouterForm::Split);
        assert_eq!(form(1, 3), RouterForm::PerToken);
        assert_eq!(form(2, 3), RouterForm::PerToken);
        assert_eq!(form(3, 3), RouterForm::PerToken);
        assert_eq!(form(4, 3), RouterForm::Split);
    }

    /// One block per token and 1024 elements; widths that are not whole
    /// pieces are refused.
    #[test]
    fn combine_grid_is_one_block_per_token_and_piece() {
        for tokens in [0u32, 1, 2, 7, 64, 128, 513, 8192] {
            assert_eq!(combine_grid(tokens, 4096).unwrap(), [tokens, 4, 1]);
        }
        assert_eq!(combine_grid(3, 1024).unwrap(), [3, 1, 1]);
        assert!(combine_grid(1, 0).is_err());
        assert!(combine_grid(1, 4095).is_err());
        assert!(combine_grid(1, 4096 + 512).is_err());
        assert!(combine_grid(1 << 31, 4096).is_err());
    }

    /// One thread per 8 consecutive output elements of a token: Flash's
    /// global (one KV head per chunk) and sliding (two) layers, and the
    /// refusals.
    #[test]
    fn qkv_grid_is_one_thread_per_vector() {
        let args = |kv: u32, stride: u32| QkvArgs {
            chunks: 4,
            chunk_stride: stride,
            q_heads_per_chunk: 16,
            kv_heads_per_chunk: kv,
            ..QkvArgs::default()
        };
        // Global: 4 x (17 x 192 + 128) = 13,568 elements, 1,696 vectors, 7
        // blocks of 256. Sliding: 4 x (18 x 192 + 256) = 14,848 elements,
        // 1,856 vectors, 8 blocks.
        for tokens in [0u32, 1, 7, 64, 8192] {
            assert_eq!(qkv_grid(tokens, &args(1, 3456)).unwrap(), [tokens, 7, 1]);
            assert_eq!(qkv_grid(tokens, &args(2, 3712)).unwrap(), [tokens, 8, 1]);
        }
        for (kv, stride) in [(1u32, 3456u32), (2, 3712)] {
            let [_, blocks, _] = qkv_grid(1, &args(kv, stride)).unwrap();
            let elements = 4 * ((16 + kv) * 192 + kv * 128);
            assert!(elements.is_multiple_of(QKV_VEC));
            let vectors = elements / QKV_VEC;
            assert!(blocks * QKV_THREADS >= vectors);
            assert!((blocks - 1) * QKV_THREADS < vectors);
        }
        assert!(
            qkv_grid(1, &args(1, 3391)).is_err(),
            "stride below the heads"
        );
        assert!(qkv_grid(1, &args(1, 3392)).is_ok());
        // A count that is not whole blocks rounds up: 3 x 3,392 = 10,176
        // elements, 1,272 vectors.
        let three = QkvArgs {
            chunks: 3,
            ..args(1, 3392)
        };
        assert_eq!(qkv_grid(2, &three).unwrap(), [2, 5, 1]);
        assert!(
            qkv_grid(
                1,
                &QkvArgs {
                    chunks: 0,
                    ..args(1, 3456)
                }
            )
            .is_err()
        );
        assert!(qkv_grid(1, &args(u32::MAX, u32::MAX)).is_err(), "overflow");
        assert!(qkv_grid(1 << 31, &args(1, 3456)).is_err());
        // Every vector access is 16 bytes: the buffers aligned, the strides
        // and offsets whole vectors.
        assert!(
            qkv_grid(1, &args(1, 3460)).is_err(),
            "stride not whole vectors"
        );
        let ok = QkvArgs {
            qkv: 256,
            q_out: 512,
            pool: 1024,
            rope: 2048,
            block_elems: 80,
            k_off: 8,
            v_off: 16,
            ..args(1, 3456)
        };
        assert!(qkv_grid(1, &ok).is_ok());
        for bad in [
            QkvArgs { qkv: 264, ..ok },
            QkvArgs { q_out: 8, ..ok },
            QkvArgs { pool: 1032, ..ok },
            QkvArgs { rope: 2052, ..ok },
            QkvArgs {
                block_elems: 84,
                ..ok
            },
            QkvArgs { k_off: 4, ..ok },
            QkvArgs { v_off: 18, ..ok },
        ] {
            assert!(qkv_grid(1, &bad).is_err(), "{bad:?}");
        }
    }

    /// The SwiGLU: a warp per 256-element piece of each routed pair's row;
    /// the gather: one block per token and 512-wide word. Neither depends on
    /// the layout's row count (every pair plus the padding of each expert
    /// reached): launches scale with the tokens, not the layout's padding.
    #[test]
    fn expert_row_grids_follow_the_tokens() {
        for tokens in [1u32, 2, 7, 64, 128, 129, 513, 8192] {
            let rows = u32::try_from(psum_rows(tokens as usize, 8)).unwrap();
            assert_eq!(
                gather_grid(tokens, 8, 4096, rows).unwrap(),
                [tokens, 8, 1],
                "gather, {tokens} tokens"
            );
            assert_eq!(
                swiglu_grid(tokens, 8, 2048, rows).unwrap(),
                [tokens * 8, 2, 1],
                "swiglu, {tokens} tokens"
            );
        }
        assert_eq!(gather_grid(0, 8, 4096, 0).unwrap(), [0, 8, 1]);
        assert_eq!(swiglu_grid(0, 8, 2048, 0).unwrap(), [0, 2, 1]);
        for grid in [gather_grid, swiglu_grid] {
            assert!(grid(2, 8, 2048, 15).is_err(), "more pairs than rows");
            assert!(grid(2, 8, 2048, 16).is_ok());
            assert!(grid(1, 0, 2048, 8).is_err(), "top_k 0");
            assert!(grid(1, 9, 2048, 9).is_err(), "top_k 9");
            assert!(grid(1, 8, 1000, 8).is_err(), "not whole words");
            assert!(grid(1, 8, 0, 8).is_err(), "no words");
            assert!(grid(u32::MAX / 8 + 1, 8, 2048, u32::MAX).is_err());
            assert!(grid(1 << 28, 8, 2048, u32::MAX).is_err(), "past the grid");
        }
        assert!(gather_grid(1, 8, 512, 8).is_ok());
        assert!(swiglu_grid(1, 8, 512, 8).is_err(), "half a piece");
        assert!(swiglu_grid(1, 8, 1536, 8).is_err(), "not whole pieces");
    }

    /// The dense SwiGLU launches one warp per item (a 256-element piece of a
    /// row) in whole blocks of 4 warps up to the most warps a 148-SM part
    /// holds at once, then the same grid whatever the items: every warp
    /// walks its share, and every item has a warp, whatever the count.
    #[test]
    fn swiglu_launches_a_warp_per_item_up_to_a_resident_grid() {
        assert_eq!(SWIGLU_MAX_WARPS, 4736);
        // Rows × 64 items at Flash's 16,384.
        for (rows, blocks) in [
            (1u32, 16u32),
            (4, 64),
            (73, 1168),
            (74, SWIGLU_MAX_WARPS / 4),
            (2048, SWIGLU_MAX_WARPS / 4),
        ] {
            let launch = swiglu_f32scale_grid(rows, 16_384, rows, 0, 0).unwrap();
            assert_eq!(launch.items, rows * 64, "{rows} rows");
            assert_eq!(launch.rows, rows);
            assert_eq!(launch.grid, [blocks, 1, 1], "{rows} rows");
            let warps = launch.grid[0] * SWIGLU_WARPS;
            assert!(warps >= launch.items.min(SWIGLU_MAX_WARPS));
            assert!(warps < launch.items.min(SWIGLU_MAX_WARPS) + SWIGLU_WARPS);
        }
        // Three items: one block, its last warp idle.
        let three = swiglu_f32scale_grid(3, 256, 4, 0, 0).unwrap();
        assert_eq!((three.grid, three.items), ([1, 1, 1], 3));
        assert_eq!(swiglu_f32scale_grid(0, 256, 0, 0, 0).unwrap().items, 0);
        // Items past i32 are refused (a warp's next item must not wrap).
        assert!(swiglu_f32scale_grid(1 << 24, 1 << 15, 1 << 24, 0, 0).is_err());
        assert!(swiglu_f32scale_grid(1 << 23, 1 << 15, 1 << 23, 0, 0).is_ok());
    }

    /// The dense SwiGLU and the expert SwiGLU refuse what their kernels
    /// cannot run: partial pieces, `m_pad` short of the rows, a scale layout
    /// short of the rows, and gate/up rows or codes off their 16- and 8-byte
    /// vectors.
    #[test]
    fn swiglu_refusals() {
        let ok = |q, gu| swiglu_f32scale_grid(8, 16_384, 8, q, gu);
        assert!(ok(0, 0).is_ok());
        assert!(ok(8, 16).is_ok());
        assert!(ok(0, 8).is_err(), "gate/up rows off 16 bytes");
        assert!(ok(4, 0).is_err(), "codes off 8 bytes");
        assert!(
            swiglu_f32scale_grid(8, 128, 8, 0, 0).is_err(),
            "half a piece"
        );
        assert!(
            swiglu_f32scale_grid(8, 384, 8, 0, 0).is_err(),
            "a piece and a half"
        );
        assert!(swiglu_f32scale_grid(8, 0, 8, 0, 0).is_err(), "no pieces");
        assert!(
            swiglu_f32scale_grid(8, 256, 7, 0, 0).is_err(),
            "m_pad below the rows"
        );
        assert!(swiglu_operands(0, 0, 8, 8).is_ok());
        assert!(swiglu_operands(8, 16, 8, 8).is_ok());
        assert!(
            swiglu_operands(0, 8, 8, 8).is_err(),
            "gate/up rows off 16 bytes"
        );
        assert!(swiglu_operands(4, 0, 8, 8).is_err(), "codes off 8 bytes");
        assert!(swiglu_operands(0, 0, 8, 4).is_err(), "rows4 below the rows");
        assert!(
            swiglu_operands(0, 0, 8, 10).is_err(),
            "rows4 not whole words"
        );
    }

    /// The f32-scale quantization: one block per row and four 128-wide
    /// groups, the last block's spare warps idle; refused: partial groups,
    /// `m_pad` short of the rows, rows off 16 bytes and codes off 4.
    #[test]
    fn quant_grid_is_a_warp_per_group() {
        assert_eq!(quant_grid(7, 4096, 8, 0, 0).unwrap(), [7, 8, 1]);
        assert_eq!(quant_grid(1, 16_384, 4, 0, 0).unwrap(), [1, 32, 1]);
        assert_eq!(
            quant_grid(3, 640, 4, 0, 0).unwrap(),
            [3, 2, 1],
            "five groups"
        );
        assert_eq!(quant_grid(3, 128, 4, 0, 0).unwrap(), [3, 1, 1]);
        assert_eq!(quant_grid(0, 4096, 0, 0, 0).unwrap(), [0, 8, 1]);
        assert!(quant_grid(1, 4096, 4, 4, 16).is_ok());
        assert!(quant_grid(1, 200, 4, 0, 0).is_err(), "partial group");
        assert!(quant_grid(1, 0, 4, 0, 0).is_err(), "no groups");
        assert!(quant_grid(8, 128, 4, 0, 0).is_err(), "m_pad below the rows");
        assert!(quant_grid(1, 128, 4, 0, 8).is_err(), "rows off 16 bytes");
        assert!(quant_grid(1, 128, 4, 2, 0).is_err(), "codes off 4 bytes");
        assert!(
            quant_grid(1, 128 * 4 * 65_536, 4, 0, 0).is_err(),
            "too wide"
        );
    }

    /// The placement: one block per 512 pairs, at least one and at most 128;
    /// refused: top_k outside 1..=8, a layout short of the worst placement,
    /// and a missing or misaligned scratch.
    #[test]
    fn permute_grid_follows_the_pairs() {
        let scratch = 256;
        for (tokens, blocks) in [
            (0u32, 1u32),
            (1, 1),
            (64, 1),
            (65, 1),
            (513, 1),
            (640, 1),
            (641, 11),
            (2048, 32),
            (8192, 128),
            (8193, 128),
            (32_768, 128),
        ] {
            let rows = u32::try_from(psum_rows(tokens as usize, 8)).unwrap();
            assert_eq!(
                permute_grid(tokens, 8, rows, scratch).unwrap(),
                [blocks, 1, 1],
                "{tokens} tokens"
            );
        }
        let rows = |t: u32| u32::try_from(psum_rows(t as usize, 8)).unwrap();
        assert!(
            permute_grid(200, 8, rows(200) - 1, scratch).is_err(),
            "rows"
        );
        assert!(permute_grid(1, 0, rows(1), scratch).is_err(), "top_k 0");
        assert!(permute_grid(1, 9, rows(1), scratch).is_err(), "top_k 9");
        assert!(permute_grid(1, 8, rows(1), 0).is_err(), "no scratch");
        assert!(
            permute_grid(1, 8, rows(1), 8).is_err(),
            "scratch off 16 bytes"
        );
        assert!(
            permute_grid(1, 8, u32::MAX, scratch).is_err(),
            "rows past i32"
        );
        assert!(permute_grid(u32::MAX / 8 + 1, 8, u32::MAX, scratch).is_err());
        assert_eq!(PERMUTE_SCRATCH_WORDS, 256 * 128);
    }

    /// `eidola_moe_permute`'s arithmetic on the host, for `blocks` blocks:
    /// each block's run and each warp's eighth of it as the kernel cuts
    /// them, per-(warp, expert) counts, each block's first row inside each
    /// expert's run from the counts of the blocks before it, and the pairs
    /// walked in order inside each warp.
    fn permute_emulated(ids: &[i32], blocks: usize) -> (Vec<i32>, Vec<i32>) {
        const WARPS: usize = 8;
        let n = ids.len();
        let expert = |id: i32| usize::try_from(id).ok().filter(|&e| e < 256);
        let run = n.div_ceil(blocks);
        let cut = |b: usize| {
            let lo = (b * run).min(n);
            (lo, (lo + run).min(n))
        };
        let warp_cut = |b: usize, w: usize| {
            let (blo, bhi) = cut(b);
            let seg = (bhi - blo).div_ceil(WARPS);
            let lo = (blo + w * seg).min(bhi);
            (lo, (lo + seg).min(bhi))
        };
        // counts[b][w][e]
        let mut counts = vec![vec![vec![0usize; 256]; WARPS]; blocks];
        for (b, block) in counts.iter_mut().enumerate() {
            for (w, warp) in block.iter_mut().enumerate() {
                let (lo, hi) = warp_cut(b, w);
                for &id in &ids[lo..hi] {
                    if let Some(e) = expert(id) {
                        warp[e] += 1;
                    }
                }
            }
        }
        // before[b][e]: expert e's pairs in the blocks before b.
        let mut before = vec![vec![0usize; 256]; blocks + 1];
        for b in 0..blocks {
            for e in 0..256 {
                before[b + 1][e] = before[b][e] + counts[b].iter().map(|w| w[e]).sum::<usize>();
            }
        }
        let mut grouped = vec![0i32; 256];
        let mut starts = vec![0usize; 256];
        let mut start = 0;
        for e in 0..256 {
            let total = before[blocks][e];
            starts[e] = start;
            grouped[e] = i32::try_from(start + total).unwrap();
            start += total.div_ceil(128) * 128;
        }
        let mut row_of = vec![-1i32; n];
        for b in 0..blocks {
            for w in 0..WARPS {
                let mut next: Vec<usize> = (0..256)
                    .map(|e| {
                        starts[e]
                            + before[b][e]
                            + counts[b][..w].iter().map(|x| x[e]).sum::<usize>()
                    })
                    .collect();
                let (lo, hi) = warp_cut(b, w);
                for i in lo..hi {
                    if let Some(e) = expert(ids[i]) {
                        row_of[i] = i32::try_from(next[e]).unwrap();
                        next[e] += 1;
                    }
                }
            }
        }
        (row_of, grouped)
    }

    /// The kernel's block and warp cuts with the counts of the blocks before
    /// each are the host placement, word for word, at every grid the host
    /// picks and at block counts that leave short and empty runs.
    #[test]
    fn permute_blocks_place_as_the_host_does() {
        for (tokens, spread) in [
            (1usize, 256usize),
            (64, 256),
            (65, 256),
            (129, 9),
            (513, 256),
            (2048, 12),
            (8193, 256),
        ] {
            let ids = ids(tokens as u64 * 7 + spread as u64, tokens, 8, spread);
            let mut stray = ids.clone();
            for i in (0..stray.len()).step_by(37) {
                stray[i] = if i % 2 == 0 { -1 } else { 256 };
            }
            let rows = u32::try_from(psum_rows(tokens, 8)).unwrap();
            let grid = permute_grid(u32::try_from(tokens).unwrap(), 8, rows, 16).unwrap()[0];
            for ids in [&ids, &stray] {
                let want = expert_placement(ids);
                for blocks in [grid as usize, 1, 2, 3, 128] {
                    assert_eq!(
                        permute_emulated(ids, blocks),
                        want,
                        "{tokens} tokens over {spread}, {blocks} blocks"
                    );
                }
            }
        }
    }

    /// DeepGEMM's psum scheduler (`sched::Scheduler::get_next_block` for
    /// `MGroupedContiguousWithPsumLayout`, with `get_aligned_effective_m_in_block`
    /// as `ensure_zero_padding = false` makes it) over a grouped layout:
    /// every block it computes, as (group, first row, rows computed). Group
    /// `g`'s rows run from the previous end rounded up to 128 to its own end;
    /// the scheduler subtracts those in `u32`, so an end below the previous
    /// run's rounded end would wrap: refused here.
    fn psum_blocks(ends: &[i32]) -> Vec<(usize, usize, usize)> {
        let mut blocks = Vec::new();
        let mut first = 0usize;
        for (g, &end) in ends.iter().enumerate() {
            let end = usize::try_from(end).unwrap();
            assert!(
                end >= first,
                "group {g} ends at {end}, before its start {first}"
            );
            let n = (end - first).div_ceil(128);
            for b in 0..n {
                let row = first + b * 128;
                let rows = if b + 1 == n {
                    (end - row).div_ceil(16) * 16
                } else {
                    128
                };
                blocks.push((g, row, rows));
            }
            first = end.div_ceil(128) * 128;
        }
        blocks
    }

    /// Router-like ids: `top_k` distinct experts per token, drawn from the
    /// first `spread` experts.
    fn ids(seed: u64, tokens: usize, top_k: usize, spread: usize) -> Vec<i32> {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            usize::try_from(state >> 33).unwrap()
        };
        let mut out = Vec::with_capacity(tokens * top_k);
        for _ in 0..tokens {
            let mut row: Vec<i32> = Vec::with_capacity(top_k);
            while row.len() < top_k {
                let e = i32::try_from(next() % spread).unwrap();
                if !row.contains(&e) {
                    row.push(e);
                }
            }
            row.sort_unstable();
            out.extend(row);
        }
        out
    }

    /// The psum placement is what DeepGEMM's psum scheduler expects: every
    /// routed row lies in exactly one block the scheduler computes, of its own
    /// expert; no two blocks overlap (no expert's output lands in another's
    /// rows); nothing is computed past the layout's bound; and the blocks are
    /// exactly each expert's ceil(count / 128), so the work follows the
    /// routed rows. Pairs keep their order within an expert.
    #[test]
    fn psum_placement_matches_the_psum_scheduler() {
        for (tokens, spread) in [
            (1usize, 256usize),
            (16, 256),
            (129, 256),
            (256, 256),
            (513, 256),
            (2048, 256),
            (300, 8),
            (129, 9),
            (4096, 12),
        ] {
            let top_k = 8;
            let ids = ids(tokens as u64 * 31 + spread as u64, tokens, top_k, spread);
            let (row_of, ends) = expert_placement(&ids);
            let bound = psum_rows(tokens, top_k);
            let blocks = psum_blocks(&ends);
            let mut owner = vec![None; bound];
            for &(g, first, rows) in &blocks {
                for (r, o) in owner.iter_mut().enumerate().skip(first).take(rows) {
                    assert!(o.is_none(), "{tokens}/{spread}: row {r} computed twice");
                    *o = Some(g);
                }
                assert!(
                    first + rows <= bound,
                    "{tokens}/{spread}: block past the bound"
                );
            }
            let mut counts = vec![0usize; 256];
            let mut last = vec![None; 256];
            for (i, (&id, &r)) in ids.iter().zip(&row_of).enumerate() {
                let (e, r) = (usize::try_from(id).unwrap(), usize::try_from(r).unwrap());
                assert_eq!(owner[r], Some(e), "{tokens}/{spread}: pair {i} at row {r}");
                assert!(
                    last[e].is_none_or(|l| l < r),
                    "pairs out of order in expert {e}"
                );
                last[e] = Some(r);
                counts[e] += 1;
            }
            let want: usize = counts.iter().map(|c| c.div_ceil(128)).sum();
            assert_eq!(blocks.len(), want, "{tokens}/{spread}: blocks");
        }
    }

    /// The psum bound holds the placements that pad the most: the pairs
    /// spread over as many experts as they can reach, each run padded to a
    /// whole block.
    #[test]
    fn psum_rows_hold_the_worst_placement() {
        for tokens in [1usize, 2, 16, 17, 32, 33, 129, 513, 8192] {
            let top_k = 8;
            let n = tokens * top_k;
            let experts = n.min(256);
            let mut ids = Vec::with_capacity(n);
            for i in 0..n {
                ids.push(i32::try_from(i % experts).unwrap());
            }
            let (_, ends) = expert_placement(&ids);
            let last = usize::try_from(*ends.last().unwrap()).unwrap();
            let used = last.div_ceil(128) * 128;
            assert!(used <= psum_rows(tokens, top_k), "{tokens}: {used} rows");
            assert!(psum_rows(tokens, top_k).is_multiple_of(128));
        }
        assert_eq!(
            psum_rows(129, 8),
            33_664,
            "1,032 pairs + 256 × 127, in whole blocks"
        );
    }

    /// An id outside the experts is placed nowhere and counted for none.
    #[test]
    fn stray_ids_are_placed_nowhere() {
        let ids = [3, 7, 3, -1, 256, 7, 3, 0];
        let (row_of, ends) = expert_placement(&ids);
        assert_eq!(row_of, [128, 256, 129, -1, -1, 257, 130, 0]);
        assert_eq!(
            (ends[0], ends[2], ends[3], ends[7], ends[255]),
            (1, 128, 131, 258, 384)
        );
    }
}
