//! The MiMo-V2.6 target forward on the device: kernels, weights, scratch, and
//! one batched forward over a step's tokens.
//!
//! Per layer:
//!
//! 1. `x = RMSNorm(h)` (f32), quantized to FP8 (f32 per-128 scales) for the
//!    fused QKV GEMM (CUTLASS blockwise, BF16 out).
//! 2. Partial RoPE on Q and K; K and V written into the layer's KV group
//!    through the block tables; paged attention with sinks / window.
//! 3. `h += o_proj(attn) · value_scale` (CUTLASS BF16 GEMM, f32 out, the value
//!    scale as the GEMM's alpha).
//! 4. `x = RMSNorm(h)`; dense SwiGLU (two FP8 GEMMs) or the routed experts
//!    (router top-k, expert-major placement, two DeepGEMM grouped GEMMs,
//!    weighted combine in ascending expert order), added into `h`.
//!
//! The residual stream `h` stays f32 throughout.

use crate::module::KernelDir;
use cudarc::driver::CudaSlice;
use eidola_engine_model::ModelConfig;
use eidola_engine_model::attention::rope_cos_sin;
use eidola_engine_model::config::{AttentionSpec, FfnKind};
use eidola_engine_model::safetensors::WeightSet;

use crate::attention::{Attention, AttnLayer, AttnPlan};
use crate::device::ImageArch;
use crate::engine_ops::{EngineOps, QkvArgs};
use crate::gemm::{Gemm, GemmArgs, GemmKind};
use crate::kv::{GroupGeometry, KvStore};
use crate::launch::{dptr, dptr_at};
use crate::module::{ImageSource, KernelModule};
use crate::moe_gemm::{BLOCK_M, MoeGemm, MoeGemmArgs, MoeLayout, MoeProj};
use crate::sampler::Sampler;
use crate::weights::{Ffn, ModelWeights};
use crate::{CudaError, Gpu, Result};

/// Every kernel the executor launches.
pub struct Kernels {
    pub ops: EngineOps,
    pub fp8_gemm: Gemm,
    pub bf16_gemm: Gemm,
    pub moe: MoeGemm,
    pub attention: Attention,
    pub sampler: Sampler,
    _modules: Vec<KernelModule>,
}

impl Kernels {
    /// Load every image for `arch` (the device's own by default).
    pub fn load(gpu: &Gpu, dir: &KernelDir, arch: Option<ImageArch>) -> Result<Kernels> {
        let arch = arch
            .or(gpu.image_arch())
            .ok_or_else(|| CudaError::new("no kernel image for this device"))?;
        let load = |name: &str| KernelModule::load_from(gpu, dir, name, ImageSource::Cubin(arch));
        let gemm = |kind: GemmKind| -> Result<(Gemm, KernelModule)> {
            let (name, _) = kind.kernel();
            let m = load(name)?;
            Ok((Gemm::from_module(kind, &m)?, m))
        };
        let (fp8_gemm, fp8_module) = gemm(GemmKind::Fp8Blockwise)?;
        let (bf16_gemm, bf16_module) = gemm(GemmKind::Bf16)?;
        Ok(Kernels {
            ops: EngineOps::from_module(load("engine_ops")?)?,
            moe: MoeGemm::from_module(load("deepgemm_fp8_fp4_grouped")?)?,
            attention: Attention::from_module(load("flashinfer_fa2_sink_paged")?)?,
            sampler: Sampler::from_module(load("sampling")?)?,
            fp8_gemm,
            bf16_gemm,
            _modules: vec![fp8_module, bf16_module],
        })
    }
}

/// Where each target layer's KV lives: `(group, layer within the group)`.
#[derive(Clone, Debug)]
pub struct LayerKv {
    pub group: usize,
    pub layer_in_group: u32,
}

/// One forward's token-level inputs (already validated by the executor).
pub struct ForwardInput<'a> {
    pub tokens: &'a [u32],
    /// RoPE (and KV) position of each token.
    pub positions: &'a [u32],
    /// Per group, per token: the physical block and offset its KV goes to.
    pub kv_targets: &'a [Vec<(u32, u32)>],
    /// Per group, the attention work list.
    pub plans: &'a [AttnPlan],
    /// Tokens whose logits are wanted, in output order.
    pub logit_rows: &'a [u32],
}

/// Device scratch sized for at most `max_tokens` tokens and `max_logit_rows`
/// logit rows per forward.
struct Scratch {
    max_tokens: usize,
    tokens: CudaSlice<u32>,
    positions: CudaSlice<u32>,
    kv_block: CudaSlice<u32>,
    kv_slot: CudaSlice<u32>,
    h: CudaSlice<f32>,
    x: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    xq: CudaSlice<u8>,
    xsf: CudaSlice<f32>,
    qkv: CudaSlice<u16>,
    q: CudaSlice<u16>,
    attn: CudaSlice<u16>,
    gu: CudaSlice<u16>,
    ffn_out: CudaSlice<u16>,
    // MoE
    topk_ids: CudaSlice<i32>,
    topk_w: CudaSlice<f32>,
    row_of: CudaSlice<i32>,
    grouped: CudaSlice<i32>,
    row_src: CudaSlice<i32>,
    ea: CudaSlice<u8>,
    esf: CudaSlice<i32>,
    egu: CudaSlice<u16>,
    eact: CudaSlice<u8>,
    eact_sf: CudaSlice<i32>,
    edown: CudaSlice<u16>,
    // Head
    logit_rows: CudaSlice<u32>,
    sel: CudaSlice<u16>,
    pub logits: CudaSlice<f32>,
}

/// Floats per position in a RoPE table (32 cos, 32 sin).
const ROPE_TABLE_WIDTH: usize = 64;

/// Masked layout (decode-sized batches): rows per expert.
const MASKED_CAP: usize = 128;

fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

/// Rows the contiguous expert layout can need for `tokens` tokens.
fn contiguous_rows(tokens: usize, top_k: usize, experts: usize) -> usize {
    let n = tokens * top_k;
    round_up(
        n + n.min(experts) * (BLOCK_M as usize - 1),
        BLOCK_M as usize,
    )
}

pub struct GpuModel {
    pub weights: ModelWeights,
    pub kernels: Kernels,
    pub layer_kv: Vec<LayerKv>,
    scratch: Scratch,
    /// Per distinct RoPE θ: `[max_len][32 cos | 32 sin]`.
    rope: Vec<(f32, CudaSlice<f32>)>,
    max_logit_rows: usize,
    /// When set, a host copy of `h` after every layer (the last forward's).
    pub capture_layers: bool,
    pub captured: Vec<Vec<f32>>,
}

/// Element counts of every scratch buffer, from the retained layers and the
/// step capacity; checked arithmetic throughout.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ScratchSizes {
    toks: usize,
    per_group: usize,
    tph: usize,
    tpk: usize,
    xsf: usize,
    qkv: usize,
    q: usize,
    attn: usize,
    gu: usize,
    topk: usize,
    eah: usize,
    esf: usize,
    egu: usize,
    eact: usize,
    eact_sf: usize,
    sel: usize,
    logits: usize,
    erows: usize,
    experts: usize,
    rows: usize,
}

impl ScratchSizes {
    pub(crate) fn new(
        c: &ModelConfig,
        groups: usize,
        qkv_cols: usize,
        max_tokens: usize,
        max_logit_rows: usize,
        max_len: usize,
    ) -> Result<ScratchSizes> {
        let (h, v) = (c.hidden_size, c.vocab_size);
        // Every buffer size is computed with checked arithmetic, and the whole
        // set before anything is allocated.
        let overflow = || CudaError::new("scratch sizes overflow for this step capacity");
        let prod = |xs: &[usize]| -> Result<usize> {
            xs.iter()
                .try_fold(1usize, |acc, &x| acc.checked_mul(x))
                .map(|n| n.max(1))
                .ok_or_else(overflow)
        };
        let up = |x: usize, m: usize| -> Result<usize> {
            x.max(1).div_ceil(m).checked_mul(m).ok_or_else(overflow)
        };
        let tp = up(max_tokens, 4)?;
        let nq = c
            .layers
            .iter()
            .map(|l| l.attention.num_q_heads)
            .max()
            .unwrap_or(0);
        // Only the retained layers' FFNs size their buffers: a dense-only
        // selection needs no expert scratch, an expert-only one no dense.
        let has = |k: FfnKind| c.layers.iter().any(|l| l.ffn == k);
        let dense_i = if has(FfnKind::Dense) {
            c.dense_intermediate_size
        } else {
            0
        };
        let kmax = h.max(dense_i).max(prod(&[nq, 128])?);
        let (top_k, experts, inter) = match &c.moe {
            Some(m) if has(FfnKind::Moe) => (m.top_k, m.num_experts, m.intermediate_size),
            _ => (0, 0, 0),
        };
        let erows = if experts > 0 {
            let n = prod(&[max_tokens, top_k])?;
            let padded = prod(&[n.min(experts), BLOCK_M as usize - 1])?
                .checked_add(n)
                .ok_or_else(overflow)?;
            up(padded, BLOCK_M as usize)?.max(prod(&[experts, MASKED_CAP])?)
        } else {
            1
        };
        let erows4 = up(erows, 4)?;
        let rows = max_logit_rows.max(1);
        let sz = [
            max_tokens.max(1),
            prod(&[groups, max_tokens])?,
            prod(&[tp, h])?,
            prod(&[tp, kmax])?,
            prod(&[kmax / 128, tp])?,
            prod(&[tp, qkv_cols])?,
            prod(&[max_tokens, nq, 192])?,
            prod(&[max_tokens, nq, 128])?,
            prod(&[tp, 2, dense_i])?,
            prod(&[max_tokens, top_k])?,
            prod(&[erows, h])?,
            prod(&[h / 512, erows4])?,
            prod(&[erows, 2, inter])?,
            prod(&[erows, inter])?,
            prod(&[inter / 512, erows4])?,
            prod(&[rows, h])?,
            prod(&[rows, v])?,
            prod(&[max_len, ROPE_TABLE_WIDTH])?,
        ];
        let [
            toks,
            per_group,
            tph,
            tpk,
            xsf,
            qkv,
            q,
            attn,
            gu,
            topk,
            eah,
            esf,
            egu,
            eact,
            eact_sf,
            sel,
            logits,
            _rope_len,
        ] = sz;
        Ok(ScratchSizes {
            toks,
            per_group,
            tph,
            tpk,
            xsf,
            qkv,
            q,
            attn,
            gu,
            topk,
            eah,
            esf,
            egu,
            eact,
            eact_sf,
            sel,
            logits,
            erows,
            experts,
            rows,
        })
    }

    /// Bytes across the expert buffers (`ea`, `esf`, `egu`, `eact`,
    /// `eact_sf`, `edown`).
    #[cfg(test)]
    fn expert_bytes(&self) -> usize {
        self.eah + self.esf * 4 + self.egu * 2 + self.eact + self.eact_sf * 4 + self.eah * 2
    }
}

impl GpuModel {
    pub fn new(
        gpu: &Gpu,
        weights: ModelWeights,
        kernels: Kernels,
        layer_kv: Vec<LayerKv>,
        max_tokens: usize,
        max_logit_rows: usize,
        max_len: usize,
    ) -> Result<GpuModel> {
        let c = &weights.config;
        let s = gpu.stream();
        let groups = layer_kv.iter().map(|l| l.group + 1).max().unwrap_or(0);
        let qkv_cols = weights
            .layers
            .iter()
            .map(|l| l.attention.qkv.linear.n as usize)
            .max()
            .unwrap_or(0);
        let ScratchSizes {
            toks,
            per_group,
            tph,
            tpk,
            xsf,
            qkv,
            q,
            attn,
            gu,
            topk,
            eah,
            esf,
            egu,
            eact,
            eact_sf,
            sel,
            logits,
            erows,
            experts,
            rows,
        } = ScratchSizes::new(c, groups, qkv_cols, max_tokens, max_logit_rows, max_len)?;
        let scratch = Scratch {
            max_tokens,
            tokens: s.alloc_zeros(toks)?,
            positions: s.alloc_zeros(toks)?,
            kv_block: s.alloc_zeros(per_group)?,
            kv_slot: s.alloc_zeros(per_group)?,
            h: s.alloc_zeros(tph)?,
            x: s.alloc_zeros(tph)?,
            proj: s.alloc_zeros(tph)?,
            xq: s.alloc_zeros(tpk)?,
            xsf: s.alloc_zeros(xsf)?,
            qkv: s.alloc_zeros(qkv)?,
            q: s.alloc_zeros(q)?,
            attn: s.alloc_zeros(attn)?,
            gu: s.alloc_zeros(gu)?,
            ffn_out: s.alloc_zeros(tph)?,
            topk_ids: s.alloc_zeros(topk)?,
            topk_w: s.alloc_zeros(topk)?,
            row_of: s.alloc_zeros(topk)?,
            grouped: s.alloc_zeros(erows.max(experts))?,
            row_src: s.alloc_zeros(erows)?,
            ea: s.alloc_zeros(eah)?,
            esf: s.alloc_zeros(esf)?,
            egu: s.alloc_zeros(egu)?,
            eact: s.alloc_zeros(eact)?,
            eact_sf: s.alloc_zeros(eact_sf)?,
            edown: s.alloc_zeros(eah)?,
            logit_rows: s.alloc_zeros(rows)?,
            sel: s.alloc_zeros(sel)?,
            logits: s.alloc_zeros(logits)?,
        };
        let mut rope = Vec::new();
        for l in &c.layers {
            let spec = &l.attention;
            if rope.iter().any(|(t, _): &(f32, _)| *t == spec.rope_theta) {
                continue;
            }
            rope.push((spec.rope_theta, s.clone_htod(&rope_table(spec, max_len))?));
        }
        Ok(GpuModel {
            weights,
            kernels,
            layer_kv,
            scratch,
            rope,
            max_logit_rows,
            capture_layers: false,
            captured: Vec::new(),
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.weights.config
    }

    pub fn max_tokens(&self) -> usize {
        self.scratch.max_tokens
    }

    pub fn logits(&self) -> &CudaSlice<f32> {
        &self.scratch.logits
    }

    /// Run the target over `input`, writing KV through `kv` and leaving the
    /// logits of `input.logit_rows` in [`GpuModel::logits`].
    pub fn forward(&mut self, gpu: &Gpu, kv: &KvStore, input: &ForwardInput<'_>) -> Result<()> {
        let t = input.tokens.len();
        if t == 0 {
            return Ok(());
        }
        if t > self.scratch.max_tokens || input.logit_rows.len() > self.max_logit_rows {
            return Err(CudaError::new(format!(
                "forward of {t} tokens / {} logit rows exceeds the scratch",
                input.logit_rows.len()
            )));
        }
        let s = gpu.stream().clone();
        let c = self.weights.config.clone();
        let h = c.hidden_size;
        let tp = round_up(t, 4);
        let eps = c.rms_norm_eps;
        let sc = &mut self.scratch;
        let k = &self.kernels;
        let ops = &k.ops;

        // Inputs.
        s.memcpy_htod(input.tokens, &mut sc.tokens.slice_mut(..t))?;
        s.memcpy_htod(input.positions, &mut sc.positions.slice_mut(..t))?;
        for (g, targets) in input.kv_targets.iter().enumerate() {
            let blocks: Vec<u32> = targets.iter().map(|x| x.0).collect();
            let slots: Vec<u32> = targets.iter().map(|x| x.1).collect();
            let mt = sc.max_tokens;
            s.memcpy_htod(&blocks, &mut sc.kv_block.slice_mut(g * mt..g * mt + t))?;
            s.memcpy_htod(&slots, &mut sc.kv_slot.slice_mut(g * mt..g * mt + t))?;
        }
        // Padding rows of every M-padded activation stay zero.
        s.memset_zeros(&mut sc.x)?;
        s.memset_zeros(&mut sc.h)?;

        let p = |b: &CudaSlice<f32>| dptr(b, &s);
        unsafe {
            ops.embed(
                gpu,
                p(&sc.h),
                dptr(&self.weights.embed, &s),
                dptr(&sc.tokens, &s),
                h as u32,
                t as u32,
            )?;
        }
        if self.capture_layers {
            self.captured.clear();
        }
        for (li, lw) in self.weights.layers.iter().enumerate() {
            let spec = &c.layers[li].attention;
            let lkv = &self.layer_kv[li];
            let geom = &kv.geometry()[lkv.group];
            let nq = spec.num_q_heads as u32;
            // 1. Norm, quantize, QKV.
            unsafe {
                ops.rmsnorm(
                    gpu,
                    p(&sc.x),
                    0,
                    p(&sc.h),
                    h as u32,
                    dptr(&lw.input_norm, &s),
                    h as u32,
                    eps,
                    t as u32,
                )?;
                ops.quant_fp8(
                    gpu,
                    dptr(&sc.xq, &s),
                    p(&sc.xsf),
                    p(&sc.x),
                    tp as u32,
                    h as u32,
                    tp as u32,
                )?;
                let qkv = &lw.attention.qkv;
                k.fp8_gemm.launch(
                    gpu,
                    &GemmArgs {
                        m: tp as u32,
                        n: qkv.linear.n,
                        k: qkv.linear.k,
                        a: dptr(&sc.xq, &s),
                        b: dptr(&qkv.linear.w, &s),
                        d: dptr(&sc.qkv, &s),
                        sfa: p(&sc.xsf),
                        sfb: dptr(&qkv.linear.scale, &s),
                        alpha: 1.0,
                    },
                )?;
                // 2. RoPE, KV write, attention.
                let rope = &self
                    .rope
                    .iter()
                    .find(|(th, _)| *th == spec.rope_theta)
                    .expect("table per theta")
                    .1;
                let mt = sc.max_tokens;
                ops.qkv_rope_kv(
                    gpu,
                    QkvArgs {
                        qkv: dptr(&sc.qkv, &s),
                        q_out: dptr(&sc.q, &s),
                        pool: dptr(kv.pool(lkv.group), &s),
                        positions: dptr(&sc.positions, &s),
                        kv_block: dptr_at(&sc.kv_block, &s, lkv.group * mt),
                        kv_slot: dptr_at(&sc.kv_slot, &s, lkv.group * mt),
                        rope: dptr(rope, &s),
                        block_elems: geom.block_elems() as u64,
                        k_off: geom.k_offset(lkv.layer_in_group) as u64,
                        v_off: geom.v_offset(lkv.layer_in_group) as u64,
                        chunk_stride: qkv.chunk_stride,
                        chunks: qkv.chunks,
                        q_heads_per_chunk: qkv.q_heads_per_chunk,
                        kv_heads_per_chunk: qkv.kv_heads_per_chunk,
                    },
                    t as u32,
                )?;
                let pool = dptr(kv.pool(lkv.group), &s);
                k.attention.run(
                    gpu,
                    &input.plans[lkv.group],
                    &AttnLayer {
                        k_base: pool + 2 * geom.k_offset(lkv.layer_in_group) as u64,
                        v_base: pool + 2 * geom.v_offset(lkv.layer_in_group) as u64,
                        block_elems: geom.block_elems() as u32,
                        num_kv_heads: geom.num_kv_heads,
                        page_size: geom.block_size,
                        window_left: spec.window().map_or(-1, |w| w as i32 - 1),
                        sink: dptr(&lw.attention.sinks, &s),
                    },
                    nq,
                    dptr(&sc.q, &s),
                    dptr(&sc.attn, &s),
                )?;
                // 3. o_proj with the value scale, added into h.
                k.bf16_gemm.launch(
                    gpu,
                    &GemmArgs {
                        m: t as u32,
                        n: h as u32,
                        k: nq * 128,
                        a: dptr(&sc.attn, &s),
                        b: dptr(&lw.attention.o_proj, &s),
                        d: p(&sc.proj),
                        sfa: 0,
                        sfb: 0,
                        alpha: c.attention_value_scale.unwrap_or(1.0),
                    },
                )?;
                ops.add_f32(gpu, p(&sc.h), p(&sc.proj), (t * h) as u64)?;
                // 4. FFN.
                ops.rmsnorm(
                    gpu,
                    p(&sc.x),
                    0,
                    p(&sc.h),
                    h as u32,
                    dptr(&lw.post_attention_norm, &s),
                    h as u32,
                    eps,
                    t as u32,
                )?;
                match &lw.ffn {
                    Ffn::Dense(d) => {
                        ops.quant_fp8(
                            gpu,
                            dptr(&sc.xq, &s),
                            p(&sc.xsf),
                            p(&sc.x),
                            tp as u32,
                            h as u32,
                            tp as u32,
                        )?;
                        k.fp8_gemm.launch(
                            gpu,
                            &GemmArgs {
                                m: tp as u32,
                                n: d.gate_up.n,
                                k: d.gate_up.k,
                                a: dptr(&sc.xq, &s),
                                b: dptr(&d.gate_up.w, &s),
                                d: dptr(&sc.gu, &s),
                                sfa: p(&sc.xsf),
                                sfb: dptr(&d.gate_up.scale, &s),
                                alpha: 1.0,
                            },
                        )?;
                        ops.swiglu_quant_f32scale(
                            gpu,
                            dptr(&sc.xq, &s),
                            p(&sc.xsf),
                            dptr(&sc.gu, &s),
                            tp as u32,
                            d.inter,
                            tp as u32,
                        )?;
                        k.fp8_gemm.launch(
                            gpu,
                            &GemmArgs {
                                m: tp as u32,
                                n: d.down.n,
                                k: d.down.k,
                                a: dptr(&sc.xq, &s),
                                b: dptr(&d.down.w, &s),
                                d: dptr(&sc.ffn_out, &s),
                                sfa: p(&sc.xsf),
                                sfb: dptr(&d.down.scale, &s),
                                alpha: 1.0,
                            },
                        )?;
                        ops.add_bf16(
                            gpu,
                            p(&sc.h),
                            dptr(&sc.ffn_out, &s),
                            h as u32,
                            h as u32,
                            t as u32,
                        )?;
                    }
                    Ffn::Moe(m) => {
                        ops.router_topk(
                            gpu,
                            dptr(&sc.topk_ids, &s),
                            p(&sc.topk_w),
                            p(&sc.x),
                            dptr(&m.router, &s),
                            dptr(&m.bias, &s),
                            t as u32,
                            h as u32,
                            m.experts,
                            m.top_k,
                            m.scaling,
                        )?;
                        let masked = t <= MASKED_CAP;
                        let (layout, rows, cap) = if masked {
                            (
                                MoeLayout::Masked,
                                m.experts as usize * MASKED_CAP,
                                MASKED_CAP as u32,
                            )
                        } else {
                            let r = contiguous_rows(t, m.top_k as usize, m.experts as usize);
                            (MoeLayout::Contiguous, r, 0)
                        };
                        let rows4 = round_up(rows, 4) as u32;
                        ops.moe_permute(
                            gpu,
                            dptr(&sc.grouped, &s),
                            dptr(&sc.row_of, &s),
                            dptr(&sc.row_src, &s),
                            dptr(&sc.topk_ids, &s),
                            t as u32,
                            m.top_k,
                            cap,
                            rows as u32,
                        )?;
                        ops.gather_quant_ue8m0(
                            gpu,
                            dptr(&sc.ea, &s),
                            dptr(&sc.esf, &s),
                            p(&sc.x),
                            dptr(&sc.row_src, &s),
                            rows as u32,
                            h as u32,
                            rows4,
                            cap,
                        )?;
                        let gemm_m = if masked {
                            MASKED_CAP as u32
                        } else {
                            rows as u32
                        };
                        k.moe.launch(
                            gpu,
                            &MoeGemmArgs {
                                layout,
                                proj: MoeProj::GateUp,
                                m: gemm_m,
                                grouped_layout: dptr(&sc.grouped, &s),
                                a: dptr(&sc.ea, &s),
                                sfa: dptr(&sc.esf, &s),
                                b: dptr(&m.gate_up, &s),
                                sfb: dptr(&m.gate_up_sf, &s),
                                d: dptr(&sc.egu, &s),
                            },
                        )?;
                        ops.swiglu_quant_ue8m0(
                            gpu,
                            dptr(&sc.eact, &s),
                            dptr(&sc.eact_sf, &s),
                            dptr(&sc.egu, &s),
                            rows as u32,
                            m.inter,
                            rows4,
                            cap,
                        )?;
                        k.moe.launch(
                            gpu,
                            &MoeGemmArgs {
                                layout,
                                proj: MoeProj::Down,
                                m: gemm_m,
                                grouped_layout: dptr(&sc.grouped, &s),
                                a: dptr(&sc.eact, &s),
                                sfa: dptr(&sc.eact_sf, &s),
                                b: dptr(&m.down, &s),
                                sfb: dptr(&m.down_sf, &s),
                                d: dptr(&sc.edown, &s),
                            },
                        )?;
                        ops.moe_combine(
                            gpu,
                            p(&sc.proj),
                            dptr(&sc.edown, &s),
                            dptr(&sc.row_of, &s),
                            p(&sc.topk_w),
                            t as u32,
                            h as u32,
                            m.top_k,
                        )?;
                        ops.add_f32(gpu, p(&sc.h), p(&sc.proj), (t * h) as u64)?;
                    }
                }
            }
            if self.capture_layers {
                self.captured.push(s.clone_dtoh(&sc.h.slice(..t * h))?);
            }
        }
        // Head: final norm of the wanted rows, BF16, lm_head (f32 logits).
        let n = input.logit_rows.len();
        if n > 0 {
            s.memcpy_htod(input.logit_rows, &mut sc.logit_rows.slice_mut(..n))?;
            unsafe {
                ops.rmsnorm(
                    gpu,
                    p(&sc.x),
                    0,
                    p(&sc.h),
                    h as u32,
                    dptr(&self.weights.final_norm, &s),
                    h as u32,
                    eps,
                    t as u32,
                )?;
                ops.gather_rows_bf16(
                    gpu,
                    dptr(&sc.sel, &s),
                    p(&sc.x),
                    dptr(&sc.logit_rows, &s),
                    n as u32,
                    h as u32,
                )?;
                k.bf16_gemm.launch(
                    gpu,
                    &GemmArgs {
                        m: n as u32,
                        n: c.vocab_size as u32,
                        k: h as u32,
                        a: dptr(&sc.sel, &s),
                        b: dptr(&self.weights.lm_head, &s),
                        d: p(&sc.logits),
                        sfa: 0,
                        sfb: 0,
                        alpha: 1.0,
                    },
                )?;
            }
        }
        Ok(())
    }
}

/// `[max_len][32 cos | 32 sin]` for one layer's RoPE, computed with the
/// reference's own `rope_cos_sin` (so tables are bit-identical to the oracle's).
pub fn rope_table(spec: &AttentionSpec, max_len: usize) -> Vec<f32> {
    let half = spec.rope_dim / 2;
    let mut t = Vec::with_capacity(max_len * spec.rope_dim);
    for pos in 0..max_len {
        let (cos, sin) = rope_cos_sin(spec, pos);
        t.extend_from_slice(&cos[..half]);
        t.extend_from_slice(&sin[..half]);
    }
    t
}

/// Group target layers by (attention kind, KV shape), in the canonical
/// order: global groups before sliding ones, each kind in first-layer order.
/// The order is the configuration's, not the layer selection's.
pub fn group_layers(config: &ModelConfig) -> (Vec<AttentionSpec>, Vec<LayerKv>) {
    let same = |k: &AttentionSpec, a: &AttentionSpec| {
        k.kind == a.kind
            && k.num_kv_heads == a.num_kv_heads
            && k.head_dim_qk == a.head_dim_qk
            && k.head_dim_v == a.head_dim_v
    };
    let mut keys: Vec<AttentionSpec> = Vec::new();
    for l in &config.layers {
        if !keys.iter().any(|k| same(k, &l.attention)) {
            keys.push(l.attention.clone());
        }
    }
    keys.sort_by_key(|k| {
        matches!(
            k.kind,
            eidola_engine_model::config::AttentionKind::Sliding { .. }
        )
    });
    let mut counts = vec![0u32; keys.len()];
    let out = config
        .layers
        .iter()
        .map(|l| {
            let g = keys
                .iter()
                .position(|k| same(k, &l.attention))
                .expect("every layer's key was collected");
            counts[g] += 1;
            LayerKv {
                group: g,
                layer_in_group: counts[g] - 1,
            }
        })
        .collect();
    (keys, out)
}

/// The KV geometry of each group.
pub fn group_geometry(
    keys: &[AttentionSpec],
    layer_kv: &[LayerKv],
    block_size: u32,
    num_blocks: &[u32],
) -> Vec<GroupGeometry> {
    keys.iter()
        .enumerate()
        .map(|(g, a)| GroupGeometry {
            num_layers: layer_kv.iter().filter(|l| l.group == g).count() as u32,
            num_kv_heads: a.num_kv_heads as u32,
            head_dim_qk: a.head_dim_qk as u32,
            head_dim_v: a.head_dim_v as u32,
            block_size,
            num_blocks: num_blocks[g],
        })
        .collect()
}

/// The checkpoint's configuration, whole or with `keep_layers` of its layers.
pub fn read_config(store: &WeightSet, keep_layers: Option<&[usize]>) -> Result<ModelConfig> {
    let mut config = store
        .model_config()
        .map_err(|e| CudaError::new(e.to_string()))?;
    if let Some(keep) = keep_layers {
        crate::support::check_layers(keep, config.source_num_layers)?;
        config = config
            .truncated(keep)
            .map_err(|e| CudaError::new(e.to_string()))?;
    }
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidola_engine_model::config::AttentionKind;

    fn flash(keep: &[usize]) -> ModelConfig {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../eidola-engine-model/tests/data/flash-mopd.config.json"
        );
        ModelConfig::from_file(std::path::Path::new(path))
            .unwrap()
            .truncated(keep)
            .unwrap()
    }

    /// KV groups come in the canonical order (global, then sliding) whatever
    /// kind the first retained layer is.
    #[test]
    fn groups_are_in_canonical_order() {
        // Flash: layer 0 global, 1..=4 sliding, 5 global.
        for keep in [&[0, 1, 2, 5][..], &[1, 5], &[1, 2, 5]] {
            let (keys, kv) = group_layers(&flash(keep));
            assert_eq!(keys.len(), 2, "{keep:?}");
            assert_eq!(keys[0].kind, AttentionKind::Global, "{keep:?}");
            assert!(
                matches!(keys[1].kind, AttentionKind::Sliding { .. }),
                "{keep:?}"
            );
            for (l, kv) in flash(keep).layers.iter().zip(&kv) {
                assert_eq!(keys[kv.group].kind, l.attention.kind, "{keep:?}");
            }
        }
        let (_, kv) = group_layers(&flash(&[1, 5]));
        assert_eq!((kv[0].group, kv[1].group), (1, 0));
        let (keys, _) = group_layers(&flash(&[1, 2]));
        assert!(matches!(
            keys[..],
            [AttentionSpec {
                kind: AttentionKind::Sliding { .. },
                ..
            }]
        ));
    }

    /// Scratch is sized by the retained layers' FFNs: a dense-only selection
    /// allocates no expert buffers, an expert-only one no dense ones.
    #[test]
    fn scratch_follows_the_retained_layers() {
        let size = |keep: &[usize]| {
            let (keys, kv) = group_layers(&flash(keep));
            let groups = kv.iter().map(|l| l.group + 1).max().unwrap();
            assert_eq!(groups, keys.len());
            ScratchSizes::new(&flash(keep), groups, 12_800, 8192, 64, 4096).unwrap()
        };
        let (dense, moe, both) = (size(&[0]), size(&[1, 2]), size(&[0, 1]));
        assert_eq!(dense.experts, 0);
        assert!(dense.expert_bytes() < 1 << 16, "{dense:?}");
        assert!(moe.expert_bytes() > 1 << 30, "{moe:?}");
        assert_eq!(moe.expert_bytes(), both.expert_bytes());
        assert_eq!(moe.gu, 1, "no dense layer, no SwiGLU buffer");
        assert_eq!(dense.gu, both.gu);
        assert!(dense.gu > 1);
    }
}
