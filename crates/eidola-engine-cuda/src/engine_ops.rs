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

    pub unsafe fn swiglu_quant_ue8m0(
        &self,
        gpu: &Gpu,
        q: u64,
        sf: u64,
        gu: u64,
        rows: u32,
        inter: u32,
        rows4: u32,
        cap: u32,
    ) -> Result<()> {
        require(
            inter > 0 && inter.is_multiple_of(512) && sfa_layout_ok(rows, rows4, cap),
            "swiglu (UE8M0): intermediate must be whole 512-wide words and the scale layout cover the rows",
        )?;
        if rows == 0 {
            return Ok(());
        }
        unsafe {
            launch!(
                gpu,
                self.swiglu_ue8m0,
                [rows, inter / 512, 1],
                q,
                sf,
                gu,
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
            hidden > 0
                && (1..=crate::support::EXPERTS as u32).contains(&experts)
                && (2..=crate::support::MAX_TOP_K as u32).contains(&top_k)
                && top_k <= experts,
            // The kernel always renormalizes the selection; the reference does
            // not for a single expert, so one expert per token is not served.
            "router: at most 256 experts and 2..=8 per token",
        )?;
        if tokens == 0 {
            return Ok(());
        }
        unsafe {
            launch!(
                gpu,
                self.router,
                [tokens, 1, 1],
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
        row_src: u64,
        topk_ids: u64,
        tokens: u32,
        top_k: u32,
        cap: u32,
        rows_bound: u32,
    ) -> Result<()> {
        require(
            (1..=crate::support::MAX_TOP_K as u32).contains(&top_k)
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
                row_src,
                topk_ids,
                tokens,
                top_k,
                cap,
                rows_bound
            )
        }
    }

    pub unsafe fn gather_quant_ue8m0(
        &self,
        gpu: &Gpu,
        a: u64,
        sf: u64,
        x: u64,
        row_src: u64,
        rows: u32,
        k: u32,
        rows4: u32,
        cap: u32,
    ) -> Result<()> {
        require(
            k > 0 && k.is_multiple_of(512) && sfa_layout_ok(rows, rows4, cap),
            "gather_quant: K must be whole 512-wide words and the scale layout cover the rows",
        )?;
        if rows == 0 {
            return Ok(());
        }
        unsafe {
            launch!(
                gpu,
                self.gather_quant,
                [rows, k / 512, 1],
                a,
                sf,
                x,
                row_src,
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
            hidden > 0 && (1..=crate::support::MAX_TOP_K as u32).contains(&top_k),
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
