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

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use eidola_engine_kernels::ArtifactDir;
use eidola_engine_model::ModelConfig;
use eidola_engine_model::attention::rope_cos_sin;
use eidola_engine_model::config::AttentionSpec;
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
    pub fn load(gpu: &Gpu, dir: &ArtifactDir<'static>, arch: Option<ImageArch>) -> Result<Kernels> {
        let arch = arch
            .or(gpu.image_arch())
            .ok_or_else(|| CudaError::new("no kernel image for this device"))?;
        let load = |name: &str| KernelModule::load_from(gpu, dir, name, ImageSource::Cubin(arch));
        let gemm = |kind: GemmKind| -> Result<(Gemm, KernelModule)> {
            let (name, entry) = kind.kernel();
            let m = load(name)?;
            Ok((Gemm::new(kind, m.kernel(entry)?)?, m))
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
        let (h, v) = (c.hidden_size, c.vocab_size);
        let tp = round_up(max_tokens.max(1), 4);
        let groups = layer_kv.iter().map(|l| l.group + 1).max().unwrap_or(0);
        let qkv_cols = weights
            .layers
            .iter()
            .map(|l| l.attention.qkv.linear.n as usize)
            .max()
            .unwrap_or(0);
        let nq = c
            .layers
            .iter()
            .map(|l| l.attention.num_q_heads)
            .max()
            .unwrap_or(0);
        let dense_i = c.dense_intermediate_size;
        let kmax = h.max(dense_i).max(nq * 128);
        let (top_k, experts, inter) = c
            .moe
            .as_ref()
            .map_or((0, 0, 0), |m| (m.top_k, m.num_experts, m.intermediate_size));
        let erows = if experts > 0 {
            contiguous_rows(max_tokens, top_k, experts).max(experts * MASKED_CAP)
        } else {
            1
        };
        let erows4 = round_up(erows, 4);
        let a = |n: usize| n.max(1);
        let scratch = Scratch {
            max_tokens,
            tokens: s.alloc_zeros(a(max_tokens))?,
            positions: s.alloc_zeros(a(max_tokens))?,
            kv_block: s.alloc_zeros(a(groups * max_tokens))?,
            kv_slot: s.alloc_zeros(a(groups * max_tokens))?,
            h: s.alloc_zeros(tp * h)?,
            x: s.alloc_zeros(tp * h)?,
            proj: s.alloc_zeros(tp * h)?,
            xq: s.alloc_zeros(tp * kmax)?,
            xsf: s.alloc_zeros(kmax / 128 * tp)?,
            qkv: s.alloc_zeros(tp * qkv_cols)?,
            q: s.alloc_zeros(a(max_tokens * nq * 192))?,
            attn: s.alloc_zeros(a(max_tokens * nq * 128))?,
            gu: s.alloc_zeros(tp * 2 * dense_i)?,
            ffn_out: s.alloc_zeros(tp * h)?,
            topk_ids: s.alloc_zeros(a(max_tokens * top_k))?,
            topk_w: s.alloc_zeros(a(max_tokens * top_k))?,
            row_of: s.alloc_zeros(a(max_tokens * top_k))?,
            grouped: s.alloc_zeros(erows.max(experts))?,
            row_src: s.alloc_zeros(erows)?,
            ea: s.alloc_zeros(erows * h)?,
            esf: s.alloc_zeros(a(h / 512 * erows4))?,
            egu: s.alloc_zeros(erows * 2 * inter.max(1))?,
            eact: s.alloc_zeros(erows * inter.max(1))?,
            eact_sf: s.alloc_zeros(a(inter / 512 * erows4))?,
            edown: s.alloc_zeros(erows * h)?,
            logit_rows: s.alloc_zeros(a(max_logit_rows))?,
            sel: s.alloc_zeros(a(max_logit_rows) * h)?,
            logits: s.alloc_zeros(a(max_logit_rows) * v)?,
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

/// Group target layers by (attention kind, KV shape) in first-layer order,
/// as the CPU reference executor does.
pub fn group_layers(config: &ModelConfig) -> (Vec<AttentionSpec>, Vec<LayerKv>) {
    let mut keys: Vec<AttentionSpec> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    let mut out = Vec::new();
    for l in &config.layers {
        let a = &l.attention;
        let g = match keys.iter().position(|k| {
            k.kind == a.kind
                && k.num_kv_heads == a.num_kv_heads
                && k.head_dim_qk == a.head_dim_qk
                && k.head_dim_v == a.head_dim_v
        }) {
            Some(g) => g,
            None => {
                keys.push(a.clone());
                counts.push(0);
                keys.len() - 1
            }
        };
        out.push(LayerKv {
            group: g,
            layer_in_group: counts[g],
        });
        counts[g] += 1;
    }
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

/// Load the model (whole or with `keep_layers` of the checkpoint) onto `gpu`.
pub fn load_weights(
    gpu: &Gpu,
    store: Arc<WeightSet>,
    keep_layers: Option<&[usize]>,
) -> Result<ModelWeights> {
    let mut config = store
        .model_config()
        .map_err(|e| CudaError::new(e.to_string()))?;
    if let Some(keep) = keep_layers {
        config = config
            .truncated(keep)
            .map_err(|e| CudaError::new(e.to_string()))?;
    }
    ModelWeights::load(gpu, store, config)
}
