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
    router_tiled: Kernel,
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
            router_tiled: m.kernel("eidola_router_topk_tiled")?,
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
        require(
            k > 0 && k.is_multiple_of(128) && m_pad >= rows,
            "quant_fp8: K must be whole 128-wide groups and m_pad cover the rows",
        )?;
        if rows == 0 {
            return Ok(());
        }
        unsafe { launch!(gpu, self.quant, [rows, k / 128, 1], q, sf, x, k, m_pad) }
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
        require(
            inter > 0 && inter.is_multiple_of(128) && m_pad >= rows,
            "swiglu (f32 scales): intermediate must be whole 128-wide groups and m_pad cover the rows",
        )?;
        if rows == 0 {
            return Ok(());
        }
        unsafe {
            launch!(
                gpu,
                self.swiglu_f32s,
                [rows, inter / 128, 1],
                q,
                sf,
                gu,
                inter,
                m_pad
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
        cap: u32,
    ) -> Result<()> {
        require(
            inter > 0 && inter.is_multiple_of(512) && sfa_layout_ok(rows, rows4, cap),
            "swiglu (UE8M0): intermediate must be whole 512-wide words and the scale layout cover the rows",
        )?;
        let grid = pair_grid(tokens, top_k, inter, rows)?;
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
                rows4,
                cap
            )
        }
    }

    /// Router top-k of `tokens` rows, in the form [`RouterForm::for_tokens`]
    /// picks.
    pub unsafe fn router_topk(
        &self,
        gpu: &Gpu,
        ids: u64,
        weights: u64,
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
                RouterForm::for_tokens(tokens),
                ids,
                weights,
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
    /// the tests and the kernel bench through this.
    pub unsafe fn router_topk_form(
        &self,
        gpu: &Gpu,
        form: RouterForm,
        ids: u64,
        weights: u64,
        x: u64,
        router: u64,
        bias: u64,
        tokens: u32,
        hidden: u32,
        experts: u32,
        top_k: u32,
        scaling: f32,
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
        let grid = router_grid(form, tokens)?;
        if grid[0] == 0 {
            return Ok(());
        }
        unsafe {
            match form {
                RouterForm::PerToken => launch!(
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
                ),
                RouterForm::Tiled => launch!(
                    gpu,
                    self.router_tiled,
                    grid,
                    ids,
                    weights,
                    x,
                    router,
                    bias,
                    tokens,
                    hidden,
                    experts,
                    top_k,
                    scaling
                ),
            }
        }
    }

    pub unsafe fn moe_permute(
        &self,
        gpu: &Gpu,
        grouped_layout: u64,
        row_of: u64,
        topk_ids: u64,
        tokens: u32,
        top_k: u32,
        cap: u32,
        rows_bound: u32,
    ) -> Result<()> {
        require(
            (1..=crate::support::MAX_TOP_K).contains(&(top_k as usize))
                && (if cap == 0 {
                    rows_bound > 0
                } else {
                    cap >= tokens
                }),
            "permute: at most 8 experts per token; a masked expert holds every token",
        )?;
        unsafe {
            launch!(
                gpu,
                self.permute,
                [1, 1, 1],
                grouped_layout,
                row_of,
                topk_ids,
                tokens,
                top_k,
                cap,
                rows_bound
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
        cap: u32,
    ) -> Result<()> {
        require(
            k > 0 && k.is_multiple_of(512) && sfa_layout_ok(rows, rows4, cap),
            "gather_quant: K must be whole 512-wide words and the scale layout cover the rows",
        )?;
        let grid = pair_grid(tokens, top_k, k, rows)?;
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
                rows4,
                cap
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

/// Blocks per cluster of both router forms (the kernels' launch contracts
/// record it, and the launch sets it from there): block rank `b` scores
/// experts `32b .. 32b + 31`.
pub const ROUTER_CLUSTER: u32 = 8;
/// Threads per block of `eidola_router_topk`: one warp per expert.
pub const ROUTER_THREADS: u32 = 1024;
/// Threads per block of `eidola_router_topk_tiled`: four warps, eight experts
/// each.
pub const ROUTER_TILED_THREADS: u32 = 128;
/// Tokens per cluster of `eidola_router_topk_tiled`.
pub const ROUTER_TILE: u32 = 8;
/// The widest row the router takes.
pub const ROUTER_MAX_HIDDEN: u32 = 4096;
/// The router's row width is whole chunks of this many elements (the tiled
/// form streams rows through shared memory a chunk at a time).
pub const ROUTER_CHUNK: u32 = 128;

/// The largest token count the router runs in its one-token form; larger
/// counts take the tiled form.
///
/// Chosen from both forms' device time per launch on a B300
/// (`moe_kernel_bench`, MiMo-V2.6-Flash shapes). The one-token form costs
/// 20.5 µs at 1–6 tokens, 21.6 µs at 8–12, 38.3 µs at 16 and 38.9 µs at 24,
/// then 57 µs at 32 and 92 µs at 64, because each cluster reads the whole
/// 2 MiB router weight for its one token. The tiled form reads it once per
/// eight tokens and costs 43–50 µs from 1 to 64 tokens. The one-token form is
/// the faster up to 24 tokens and the tiled from 32. Both forms give
/// identical ids and weights, so this moves only time; `moe_kernel_bench`
/// prints both forms at every token count to retune it.
pub const ROUTER_PER_TOKEN_MAX: u32 = 24;

/// Which router kernel a launch runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouterForm {
    /// `eidola_router_topk`: one cluster per token.
    PerToken,
    /// `eidola_router_topk_tiled`: one cluster per [`ROUTER_TILE`] tokens.
    Tiled,
}

impl RouterForm {
    /// The form the executor runs for a step of `tokens` rows: a function of
    /// the token count alone, so a decode graph's rung fixes it.
    pub fn for_tokens(tokens: u32) -> RouterForm {
        if tokens <= ROUTER_PER_TOKEN_MAX {
            RouterForm::PerToken
        } else {
            RouterForm::Tiled
        }
    }
}

/// The router's grid: one cluster of [`ROUTER_CLUSTER`] blocks per token
/// (one-token form) or per [`ROUTER_TILE`] tokens, the last tile partial
/// (tiled form).
pub fn router_grid(form: RouterForm, tokens: u32) -> Result<[u32; 3]> {
    let clusters = match form {
        RouterForm::PerToken => tokens,
        RouterForm::Tiled => tokens.div_ceil(ROUTER_TILE),
    };
    let blocks = clusters
        .checked_mul(ROUTER_CLUSTER)
        .filter(|&b| b <= MAX_GRID_X)
        .ok_or_else(|| crate::CudaError::new("router: too many tokens for one launch"))?;
    Ok([blocks, 1, 1])
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

/// The fused-QKV kernel's grid: one thread per output element of a token
/// (its chunks' Q and K heads of 192 and V heads of 128), one block row per
/// token.
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
    let blocks = per_chunk
        .and_then(|n| n.checked_mul(args.chunks))
        .map(|n| n.div_ceil(QKV_THREADS))
        .filter(|&b| b <= 65_535)
        .ok_or_else(|| crate::CudaError::new("qkv_rope_kv: too many heads per token"))?;
    require(
        tokens <= MAX_GRID_X,
        "qkv_rope_kv: too many tokens for one launch",
    )?;
    Ok([tokens, blocks, 1])
}

/// The grid of the pair-indexed expert kernels (gather, SwiGLU): one block
/// per routed (token, slot) pair and 512-wide word of `width`. Every pair
/// lands in its own row, so the layout's `rows` must hold them all.
pub fn pair_grid(tokens: u32, top_k: u32, width: u32, rows: u32) -> Result<[u32; 3]> {
    require(
        (1..=crate::support::MAX_TOP_K).contains(&(top_k as usize)),
        "expert rows: 1..=8 experts per token",
    )?;
    require(
        width > 0 && width.is_multiple_of(512) && width / 512 <= 65_535,
        "expert rows: width must be whole 512-wide words",
    )?;
    let pairs = tokens
        .checked_mul(top_k)
        .filter(|&p| p <= MAX_GRID_X)
        .ok_or_else(|| crate::CudaError::new("expert rows: too many pairs for one launch"))?;
    require(
        pairs <= rows,
        "expert rows: the layout holds fewer rows than pairs",
    )?;
    Ok([pairs, width / 512, 1])
}

fn require(ok: bool, what: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(crate::CudaError::new(what))
    }
}

/// The packed UE8M0 scale layouts (`sfa_index` in `engine_ops.cu`): `[K/512][rows4]`
/// for the contiguous layout, `[rows / cap][K/512][cap]` for the masked one.
fn sfa_layout_ok(rows: u32, rows4: u32, cap: u32) -> bool {
    if cap == 0 {
        rows4 >= rows && rows4.is_multiple_of(4)
    } else {
        rows.is_multiple_of(cap) && cap.is_multiple_of(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one-token form: one cluster per token; the grid is the token
    /// count times the cluster, a whole number of clusters.
    #[test]
    fn per_token_router_grid_is_one_cluster_per_token() {
        for tokens in [0u32, 1, 2, 7, 64, 128, 513, 8192] {
            let g = router_grid(RouterForm::PerToken, tokens).unwrap();
            assert_eq!(g, [tokens * ROUTER_CLUSTER, 1, 1], "{tokens}");
        }
        assert_eq!(
            ROUTER_THREADS / 32 * ROUTER_CLUSTER,
            256,
            "one warp per expert"
        );
        let per_token = |t| router_grid(RouterForm::PerToken, t);
        assert!(per_token(u32::MAX / ROUTER_CLUSTER + 1).is_err());
        assert!(per_token((1 << 31) / ROUTER_CLUSTER).is_err());
        assert!(per_token((1 << 31) / ROUTER_CLUSTER - 1).is_ok());
    }

    /// The tiled form: one cluster per [`ROUTER_TILE`] tokens, the last one
    /// partial, so the clusters cover every token and no cluster is empty.
    #[test]
    fn tiled_router_grid_covers_every_token_once() {
        assert_eq!(
            ROUTER_TILED_THREADS / 32 * 8 * ROUTER_CLUSTER,
            256,
            "eight experts per warp"
        );
        for tokens in [0u32, 1, 2, 7, 8, 9, 15, 16, 17, 64, 128, 513, 2048, 8192] {
            let g = router_grid(RouterForm::Tiled, tokens).unwrap();
            assert_eq!(g[1..], [1, 1]);
            assert!(g[0].is_multiple_of(ROUTER_CLUSTER), "{tokens}");
            let clusters = g[0] / ROUTER_CLUSTER;
            assert!(
                clusters * ROUTER_TILE >= tokens,
                "{tokens}: a token uncovered"
            );
            assert!(
                clusters == 0 || (clusters - 1) * ROUTER_TILE < tokens,
                "{tokens}: an empty cluster"
            );
        }
        assert_eq!(router_grid(RouterForm::Tiled, 513).unwrap(), [65 * 8, 1, 1]);
        let tiled = |t| router_grid(RouterForm::Tiled, t);
        let last = (MAX_GRID_X / ROUTER_CLUSTER) * ROUTER_TILE;
        assert!(tiled(last).is_ok());
        assert!(tiled(last + 1).is_err(), "past the grid");
    }

    /// The form follows the token count alone: the one-token form up to the
    /// threshold, the tiled form above it, so every rung of a decode ladder
    /// has one form.
    #[test]
    fn router_form_switches_at_the_threshold() {
        for tokens in 0..=ROUTER_PER_TOKEN_MAX {
            assert_eq!(
                RouterForm::for_tokens(tokens),
                RouterForm::PerToken,
                "{tokens}"
            );
        }
        for tokens in [ROUTER_PER_TOKEN_MAX + 1, 32, 64, 128, 513, 8192, u32::MAX] {
            assert_eq!(
                RouterForm::for_tokens(tokens),
                RouterForm::Tiled,
                "{tokens}"
            );
        }
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

    /// One thread per output element of a token: Flash's global (one KV head
    /// per chunk) and sliding (two) layers, and the refusals.
    #[test]
    fn qkv_grid_is_one_thread_per_element() {
        let args = |kv: u32, stride: u32| QkvArgs {
            chunks: 4,
            chunk_stride: stride,
            q_heads_per_chunk: 16,
            kv_heads_per_chunk: kv,
            ..QkvArgs::default()
        };
        // Global: 4 x (17 x 192 + 128) = 13,568 elements, 53 blocks of 256.
        // Sliding: 4 x (18 x 192 + 256) = 14,848 elements, 58 blocks.
        for tokens in [0u32, 1, 7, 64, 8192] {
            assert_eq!(qkv_grid(tokens, &args(1, 3456)).unwrap(), [tokens, 53, 1]);
            assert_eq!(qkv_grid(tokens, &args(2, 3712)).unwrap(), [tokens, 58, 1]);
        }
        for (kv, stride) in [(1u32, 3456u32), (2, 3712)] {
            let [_, blocks, _] = qkv_grid(1, &args(kv, stride)).unwrap();
            let elements = 4 * ((16 + kv) * 192 + kv * 128);
            assert!(blocks * QKV_THREADS >= elements);
            assert!((blocks - 1) * QKV_THREADS < elements);
        }
        assert!(
            qkv_grid(1, &args(1, 3391)).is_err(),
            "stride below the heads"
        );
        assert!(qkv_grid(1, &args(1, 3392)).is_ok());
        // A count that is not whole blocks rounds up: 3 x 3,392 = 10,176.
        let three = QkvArgs {
            chunks: 3,
            ..args(1, 3392)
        };
        assert_eq!(qkv_grid(2, &three).unwrap(), [2, 40, 1]);
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
    }

    /// One block per routed pair and 512-wide word, whatever the layout's
    /// row count: decode launches scale with the tokens, not the experts'
    /// capacity.
    #[test]
    fn pair_grid_is_one_block_per_pair_and_word() {
        let masked_rows = 256 * 128;
        for tokens in [1u32, 2, 7, 64, 128] {
            assert_eq!(
                pair_grid(tokens, 8, 4096, masked_rows).unwrap(),
                [tokens * 8, 8, 1],
                "gather, {tokens} tokens"
            );
            assert_eq!(
                pair_grid(tokens, 8, 2048, masked_rows).unwrap(),
                [tokens * 8, 4, 1],
                "swiglu, {tokens} tokens"
            );
        }
        // Contiguous layouts hold every pair plus padding.
        for tokens in [129u32, 513, 8192] {
            let n = (tokens * 8) as usize;
            let rows = u32::try_from((n + n.min(256) * 127).div_ceil(128) * 128).unwrap();
            assert_eq!(
                pair_grid(tokens, 8, 4096, rows).unwrap(),
                [tokens * 8, 8, 1]
            );
        }
        assert_eq!(pair_grid(0, 8, 4096, 0).unwrap(), [0, 8, 1]);
        assert!(pair_grid(2, 8, 4096, 15).is_err(), "more pairs than rows");
        assert!(pair_grid(2, 8, 4096, 16).is_ok());
        assert!(pair_grid(1, 0, 4096, 8).is_err(), "top_k 0");
        assert!(pair_grid(1, 9, 4096, 9).is_err(), "top_k 9");
        assert!(pair_grid(1, 8, 1000, 8).is_err(), "not whole words");
        assert!(pair_grid(1, 8, 0, 8).is_err(), "no words");
        assert!(pair_grid(u32::MAX / 8 + 1, 8, 4096, u32::MAX).is_err());
        assert!(
            pair_grid(1 << 28, 8, 4096, u32::MAX).is_err(),
            "past the grid"
        );
    }
}
