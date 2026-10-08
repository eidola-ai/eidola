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
    permute: Kernel,
    gather_quant: Kernel,
    combine: Kernel,
    gather_bf16: Kernel,
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
        Ok(EngineOps {
            embed: m.kernel("eidola_embed")?,
            rmsnorm: m.kernel("eidola_rmsnorm_f32")?,
            quant: m.kernel("eidola_quant_fp8_f32scale")?,
            add_f32: m.kernel("eidola_add_f32")?,
            add_bf16: m.kernel("eidola_add_bf16")?,
            qkv,
            swiglu_f32s: m.kernel("eidola_swiglu_quant_fp8_f32scale")?,
            swiglu_ue8m0: m.kernel("eidola_swiglu_quant_fp8_ue8m0")?,
            router: m.kernel("eidola_router_topk")?,
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
        require(
            args.chunks > 0
                && args.chunk_stride
                    >= (args.q_heads_per_chunk + args.kv_heads_per_chunk) * 192
                        + args.kv_heads_per_chunk * 128,
            "qkv_rope_kv: chunk stride below its heads",
        )?;
        if tokens == 0 {
            return Ok(());
        }
        unsafe { launch!(gpu, self.qkv, [tokens, 1, 1], args) }
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
        require(
            (1..=ROUTER_MAX_HIDDEN).contains(&hidden)
                && (1..=crate::support::EXPERTS).contains(&(experts as usize))
                && (2..=crate::support::MAX_TOP_K).contains(&(top_k as usize))
                && top_k <= experts,
            // The kernel always renormalizes the selection; the reference does
            // not for a single expert, so one expert per token is not served.
            "router: hidden at most 4096, at most 256 experts and 2..=8 per token",
        )?;
        let grid = router_grid(tokens)?;
        if grid[0] == 0 {
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
            hidden > 0 && (1..=crate::support::MAX_TOP_K).contains(&(top_k as usize)),
            "combine: 1..=8 experts per token",
        )?;
        if tokens == 0 {
            return Ok(());
        }
        unsafe {
            launch!(
                gpu,
                self.combine,
                [tokens, 1, 1],
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
}

/// The driver's grid limit in x.
const MAX_GRID_X: u32 = (1 << 31) - 1;

/// Blocks per token of `eidola_router_topk`: its cluster size (the kernel's
/// launch contract records it, and the launch sets it from there).
pub const ROUTER_CLUSTER: u32 = 8;
/// Threads per block of `eidola_router_topk`: one warp per expert.
pub const ROUTER_THREADS: u32 = 1024;
/// The widest row the router stages in shared memory.
pub const ROUTER_MAX_HIDDEN: u32 = 4096;

/// The router's grid: one cluster of [`ROUTER_CLUSTER`] blocks per token.
pub fn router_grid(tokens: u32) -> Result<[u32; 3]> {
    let blocks = tokens
        .checked_mul(ROUTER_CLUSTER)
        .filter(|&b| b <= MAX_GRID_X)
        .ok_or_else(|| crate::CudaError::new("router: too many tokens for one launch"))?;
    Ok([blocks, 1, 1])
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

    /// One cluster per token: the grid is the token count times the cluster,
    /// a whole number of clusters, and a function of the token count alone
    /// (a decode graph's rung).
    #[test]
    fn router_grid_is_one_cluster_per_token() {
        for tokens in [0u32, 1, 2, 7, 64, 128, 513, 8192] {
            let g = router_grid(tokens).unwrap();
            assert_eq!(g, [tokens * ROUTER_CLUSTER, 1, 1], "{tokens}");
            assert!(g[0].is_multiple_of(ROUTER_CLUSTER));
            assert_eq!(g[0] / ROUTER_CLUSTER, tokens);
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
