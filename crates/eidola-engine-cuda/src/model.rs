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
use eidola_engine::spec::NULL_BLOCK;
use eidola_engine_model::ModelConfig;
use eidola_engine_model::attention::rope_cos_sin;
use eidola_engine_model::config::{AttentionSpec, FfnKind};
use eidola_engine_model::safetensors::WeightSet;

use crate::attention::{Attention, AttnLayer, AttnPlan, PlanShape, PlanView};
use crate::device::ImageArch;
use crate::engine_ops::{EngineOps, QkvArgs};
use crate::gemm::{Gemm, GemmArgs, GemmKind};
use crate::kv::{GroupGeometry, KvStore};
use crate::launch::{dptr, dptr_at};
use crate::module::{ImageSource, KernelModule};
use crate::moe_gemm::{BLOCK_M, MoeGemm, MoeGemmArgs, MoeLayout, MoeProj};
use crate::sampler::Sampler;
use crate::weights::{Ffn, ModelWeights};
use crate::{CudaError, Gpu, Result, narrow};

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

/// [`ForwardInput`]'s host values with its plans as shapes: what
/// [`GpuModel::check_parts`] checks.
pub(crate) struct InputParts<'a> {
    pub tokens: &'a [u32],
    pub positions: &'a [u32],
    pub kv_targets: &'a [Vec<(u32, u32)>],
    pub plans: &'a [PlanShape],
    pub logit_rows: &'a [u32],
}

/// Where a forward's launches read their per-step values: device
/// addresses, and the counts that fix the launch shapes.
pub(crate) struct Indirect {
    /// Token rows every row-wise launch covers.
    pub tokens: usize,
    pub token_ids: u64,
    pub positions: u64,
    /// Per group: `tokens` block ids, then `tokens` offsets in the block.
    pub kv_block: Vec<u64>,
    pub kv_slot: Vec<u64>,
    /// Per group: the attention work list.
    pub plans: Vec<PlanView>,
    /// `num_logit_rows` token rows whose logits the head computes.
    pub logit_rows: u64,
    pub num_logit_rows: usize,
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
const MASKED_CAP32: u32 = 128;
const _: () = assert!(MASKED_CAP32 as usize == MASKED_CAP);

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
    pub(crate) weights: ModelWeights,
    pub(crate) kernels: Kernels,
    pub(crate) layer_kv: Vec<LayerKv>,
    scratch: Scratch,
    /// Per distinct RoPE θ: `[max_len][32 cos | 32 sin]`.
    rope: Vec<(f32, CudaSlice<f32>)>,
    max_logit_rows: usize,
    /// Positions the RoPE tables cover.
    max_len: usize,
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
    /// The model over its weights, with `layer_kv` from [`group_layers`] of
    /// the same configuration (the executor's construction).
    pub(crate) fn new(
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
            max_len,
            capture_layers: false,
            captured: Vec::new(),
        })
    }

    pub fn weights(&self) -> &ModelWeights {
        &self.weights
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    /// Where each layer's KV lives.
    pub fn layer_kv(&self) -> &[LayerKv] {
        &self.layer_kv
    }

    pub fn config(&self) -> &ModelConfig {
        &self.weights.config
    }

    pub fn max_tokens(&self) -> usize {
        self.scratch.max_tokens
    }

    /// KV groups the retained layers use.
    pub fn num_groups(&self) -> usize {
        self.layer_kv.iter().map(|l| l.group + 1).max().unwrap_or(0)
    }

    pub fn logits(&self) -> &CudaSlice<f32> {
        &self.scratch.logits
    }

    /// Check every host value of `input` that a kernel would use as an index
    /// or count, against the scratch, the vocabulary, the RoPE tables and
    /// `kv`'s pools, without touching the device. [`GpuModel::forward`] runs
    /// it first; the executor runs it before a step changes any state.
    pub fn check_input(&self, kv: &KvStore, input: &ForwardInput<'_>) -> Result<()> {
        let shapes: Vec<PlanShape> = input.plans.iter().map(AttnPlan::shape).collect();
        self.check_parts(
            kv,
            &InputParts {
                tokens: input.tokens,
                positions: input.positions,
                kv_targets: input.kv_targets,
                plans: &shapes,
                logit_rows: input.logit_rows,
            },
        )
    }

    /// [`GpuModel::check_input`] over host plans' shapes: what the executor
    /// checks before a step changes any state, whichever path then runs it.
    pub(crate) fn check_parts(&self, kv: &KvStore, input: &InputParts<'_>) -> Result<()> {
        let bad = |what: String| Err(CudaError::new(format!("forward input: {what}")));
        let t = input.tokens.len();
        if t > self.scratch.max_tokens || input.logit_rows.len() > self.max_logit_rows {
            return bad(format!(
                "{t} tokens / {} logit rows exceed the scratch",
                input.logit_rows.len()
            ));
        }
        if input.positions.len() != t {
            return bad(format!(
                "{} positions for {t} tokens",
                input.positions.len()
            ));
        }
        let vocab = self.weights.config.vocab_size;
        if let Some(&tok) = input.tokens.iter().find(|&&x| x as usize >= vocab) {
            return bad(format!("token {tok} outside the vocabulary of {vocab}"));
        }
        if let Some(&p) = input
            .positions
            .iter()
            .find(|&&p| p as usize >= self.max_len)
        {
            return bad(format!(
                "position {p} past the RoPE tables' {}",
                self.max_len
            ));
        }
        if let Some(&r) = input.logit_rows.iter().find(|&&r| r as usize >= t) {
            return bad(format!("logit row {r} of {t} tokens"));
        }
        let geometry = kv.geometry();
        let groups = self.num_groups();
        if geometry.len() != groups
            || input.kv_targets.len() != groups
            || input.plans.len() != groups
        {
            return bad(format!(
                "{} KV pools, {} target lists, {} plans for {groups} groups",
                geometry.len(),
                input.kv_targets.len(),
                input.plans.len()
            ));
        }
        // The pools must be the ones these layers were grouped for: each
        // layer's KV shape is its group's, and its index inside the group's
        // block exists.
        for (l, lkv) in self.weights.config.layers.iter().zip(&self.layer_kv) {
            let (a, geom) = (&l.attention, &geometry[lkv.group]);
            if geom.num_kv_heads as usize != a.num_kv_heads
                || geom.head_dim_qk as usize != a.head_dim_qk
                || geom.head_dim_v as usize != a.head_dim_v
                || lkv.layer_in_group >= geom.num_layers
            {
                return bad(format!(
                    "layer {} does not fit KV group {} ({geom:?})",
                    l.index, lkv.group
                ));
            }
        }
        for (g, (geom, targets)) in geometry.iter().zip(input.kv_targets).enumerate() {
            if targets.len() != t {
                return bad(format!(
                    "group {g}: {} KV targets for {t} tokens",
                    targets.len()
                ));
            }
            // Block 0 is the null block: never written.
            if let Some(&(b, o)) = targets
                .iter()
                .find(|&&(b, o)| b == NULL_BLOCK || b >= geom.num_blocks || o >= geom.block_size)
            {
                return bad(format!("group {g}: KV target block {b} offset {o}"));
            }
            let plan = &input.plans[g];
            if plan.page_size != geom.block_size
                || plan.q_rows as usize != t
                || plan.max_page.is_some_and(|p| p >= geom.num_blocks)
            {
                return bad(format!(
                    "group {g}: a plan for pages of {} over {} rows reading block {:?}",
                    plan.page_size, plan.q_rows, plan.max_page
                ));
            }
        }
        Ok(())
    }

    /// Run the target over `input`, writing KV through `kv` and leaving the
    /// logits of `input.logit_rows` in [`GpuModel::logits`]. The input is
    /// checked whole ([`GpuModel::check_input`]) before anything is launched.
    pub fn forward(&mut self, gpu: &Gpu, kv: &KvStore, input: &ForwardInput<'_>) -> Result<()> {
        self.check_input(kv, input)?;
        let t = input.tokens.len();
        if t == 0 {
            return Ok(());
        }
        let s = gpu.stream().clone();
        let sc = &mut self.scratch;
        let mt = sc.max_tokens;

        // Inputs.
        s.memcpy_htod(input.tokens, &mut sc.tokens.slice_mut(..t))?;
        s.memcpy_htod(input.positions, &mut sc.positions.slice_mut(..t))?;
        for (g, targets) in input.kv_targets.iter().enumerate() {
            let blocks: Vec<u32> = targets.iter().map(|x| x.0).collect();
            let slots: Vec<u32> = targets.iter().map(|x| x.1).collect();
            s.memcpy_htod(&blocks, &mut sc.kv_block.slice_mut(g * mt..g * mt + t))?;
            s.memcpy_htod(&slots, &mut sc.kv_slot.slice_mut(g * mt..g * mt + t))?;
        }
        let n = input.logit_rows.len();
        if n > 0 {
            s.memcpy_htod(input.logit_rows, &mut sc.logit_rows.slice_mut(..n))?;
        }
        // Padding rows of every M-padded activation stay zero.
        s.memset_zeros(&mut sc.x)?;
        s.memset_zeros(&mut sc.h)?;

        let groups = input.kv_targets.len();
        let src = Indirect {
            tokens: t,
            token_ids: dptr(&sc.tokens, &s),
            positions: dptr(&sc.positions, &s),
            kv_block: (0..groups)
                .map(|g| dptr_at(&sc.kv_block, &s, g * mt))
                .collect(),
            kv_slot: (0..groups)
                .map(|g| dptr_at(&sc.kv_slot, &s, g * mt))
                .collect(),
            plans: input.plans.iter().map(|p| p.view(gpu)).collect(),
            logit_rows: dptr(&sc.logit_rows, &s),
            num_logit_rows: n,
        };
        let capture = self.capture_layers;
        // SAFETY: every address is this model's scratch, just filled with
        // the checked input, or a plan `check_input` accepted.
        unsafe { self.launch(gpu, kv, &src, capture) }
    }

    /// Zero the first `rows` rows (padded to whole groups of four) of the
    /// M-padded activations the next forward reads past its tokens: what
    /// [`GpuModel::forward`] does to the whole buffers, over only the rows a
    /// forward of `rows` tokens reaches.
    pub(crate) fn zero_padding_rows(&mut self, gpu: &Gpu, rows: usize) -> Result<()> {
        let s = gpu.stream();
        let n = round_up(rows, 4)
            .checked_mul(self.weights.config.hidden_size)
            .filter(|&n| n <= self.scratch.x.len() && n <= self.scratch.h.len())
            .ok_or_else(|| CudaError::new(format!("{rows} rows exceed the scratch")))?;
        s.memset_zeros(&mut self.scratch.x.slice_mut(..n))?;
        s.memset_zeros(&mut self.scratch.h.slice_mut(..n))?;
        Ok(())
    }

    /// The forward's launches, from the embedding to the head, reading every
    /// per-step value through `src`'s device addresses: nothing here copies
    /// from the host (unless `capture` asks for the per-layer copies),
    /// allocates, or reads back, so with fixed addresses and counts the
    /// sequence can be recorded once and replayed.
    ///
    /// # Safety
    ///
    /// When the launches run, `src`'s addresses must hold `src.tokens` token
    /// ids inside the vocabulary, as many positions inside the RoPE tables,
    /// per group as many KV targets inside the group's pool (pad block
    /// included), each group's work list over the same rows with pages
    /// inside the pool, and `src.num_logit_rows` logit rows below
    /// `src.tokens`; `src.tokens` and `src.num_logit_rows` within the
    /// scratch.
    pub(crate) unsafe fn launch(
        &mut self,
        gpu: &Gpu,
        kv: &KvStore,
        src: &Indirect,
        capture: bool,
    ) -> Result<()> {
        let t = src.tokens;
        if t == 0 {
            return Ok(());
        }
        if t > self.scratch.max_tokens
            || src.num_logit_rows > self.max_logit_rows
            || src.kv_block.len() != self.num_groups()
            || src.kv_slot.len() != self.num_groups()
            || src.plans.len() != self.num_groups()
        {
            return Err(CudaError::new(format!(
                "forward launch: {t} tokens, {} logit rows, {} / {} / {} group inputs",
                src.num_logit_rows,
                src.kv_block.len(),
                src.kv_slot.len(),
                src.plans.len()
            )));
        }
        let s = gpu.stream().clone();
        let c = self.weights.config.clone();
        let h = c.hidden_size;
        let tp = round_up(t, 4);
        // Every count the kernels take, as the u32 they take it as.
        let (h32, t32, tp32): (u32, u32, u32) = (
            narrow(h, "hidden size")?,
            narrow(t, "step tokens")?,
            narrow(tp, "padded step tokens")?,
        );
        let eps = c.rms_norm_eps;
        let sc = &mut self.scratch;
        let k = &self.kernels;
        let ops = &k.ops;

        let p = |b: &CudaSlice<f32>| dptr(b, &s);
        unsafe {
            ops.embed(
                gpu,
                p(&sc.h),
                dptr(&self.weights.embed, &s),
                src.token_ids,
                h32,
                t32,
            )?;
        }
        if capture {
            self.captured.clear();
        }
        for (li, lw) in self.weights.layers.iter().enumerate() {
            let spec = &c.layers[li].attention;
            let lkv = &self.layer_kv[li];
            let geom = &kv.geometry()[lkv.group];
            let nq: u32 = narrow(spec.num_q_heads, "query heads")?;
            // 1. Norm, quantize, QKV.
            unsafe {
                ops.rmsnorm(
                    gpu,
                    p(&sc.x),
                    0,
                    p(&sc.h),
                    h32,
                    dptr(&lw.input_norm, &s),
                    h32,
                    eps,
                    t32,
                )?;
                ops.quant_fp8(gpu, dptr(&sc.xq, &s), p(&sc.xsf), p(&sc.x), tp32, h32, tp32)?;
                let qkv = &lw.attention.qkv;
                k.fp8_gemm.launch(
                    gpu,
                    &GemmArgs {
                        m: tp32,
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
                ops.qkv_rope_kv(
                    gpu,
                    QkvArgs {
                        qkv: dptr(&sc.qkv, &s),
                        q_out: dptr(&sc.q, &s),
                        pool: dptr(kv.pool(lkv.group), &s),
                        positions: src.positions,
                        kv_block: src.kv_block[lkv.group],
                        kv_slot: src.kv_slot[lkv.group],
                        rope: dptr(rope, &s),
                        block_elems: geom.block_elems() as u64,
                        k_off: geom.k_offset(lkv.layer_in_group) as u64,
                        v_off: geom.v_offset(lkv.layer_in_group) as u64,
                        chunk_stride: qkv.chunk_stride,
                        chunks: qkv.chunks,
                        q_heads_per_chunk: qkv.q_heads_per_chunk,
                        kv_heads_per_chunk: qkv.kv_heads_per_chunk,
                    },
                    t32,
                )?;
                let pool = dptr(kv.pool(lkv.group), &s);
                k.attention.run_view(
                    gpu,
                    &src.plans[lkv.group],
                    &AttnLayer {
                        k_base: pool + 2 * geom.k_offset(lkv.layer_in_group) as u64,
                        v_base: pool + 2 * geom.v_offset(lkv.layer_in_group) as u64,
                        block_elems: narrow(geom.block_elems(), "KV block elements")?,
                        num_kv_heads: geom.num_kv_heads,
                        page_size: geom.block_size,
                        window_left: match spec.window() {
                            Some(w) => narrow::<i32, _>(w, "window")? - 1,
                            None => -1,
                        },
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
                        m: t32,
                        n: h32,
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
                    h32,
                    dptr(&lw.post_attention_norm, &s),
                    h32,
                    eps,
                    t32,
                )?;
                match &lw.ffn {
                    Ffn::Dense(d) => {
                        ops.quant_fp8(
                            gpu,
                            dptr(&sc.xq, &s),
                            p(&sc.xsf),
                            p(&sc.x),
                            tp32,
                            h32,
                            tp32,
                        )?;
                        k.fp8_gemm.launch(
                            gpu,
                            &GemmArgs {
                                m: tp32,
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
                            tp32,
                            d.inter,
                            tp32,
                        )?;
                        k.fp8_gemm.launch(
                            gpu,
                            &GemmArgs {
                                m: tp32,
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
                        ops.add_bf16(gpu, p(&sc.h), dptr(&sc.ffn_out, &s), h32, h32, t32)?;
                    }
                    Ffn::Moe(m) => {
                        ops.router_topk(
                            gpu,
                            dptr(&sc.topk_ids, &s),
                            p(&sc.topk_w),
                            p(&sc.x),
                            dptr(&m.router, &s),
                            dptr(&m.bias, &s),
                            t32,
                            h32,
                            m.experts,
                            m.top_k,
                            m.scaling,
                        )?;
                        let masked = t <= MASKED_CAP;
                        let (layout, rows, cap) = if masked {
                            (
                                MoeLayout::Masked,
                                m.experts as usize * MASKED_CAP,
                                MASKED_CAP32,
                            )
                        } else {
                            let r = contiguous_rows(t, m.top_k as usize, m.experts as usize);
                            (MoeLayout::Contiguous, r, 0)
                        };
                        let rows4: u32 = narrow(round_up(rows, 4), "padded expert rows")?;
                        let rows32: u32 = narrow(rows, "expert rows")?;
                        ops.moe_permute(
                            gpu,
                            dptr(&sc.grouped, &s),
                            dptr(&sc.row_of, &s),
                            dptr(&sc.topk_ids, &s),
                            t32,
                            m.top_k,
                            cap,
                            rows32,
                        )?;
                        ops.gather_quant_ue8m0(
                            gpu,
                            dptr(&sc.ea, &s),
                            dptr(&sc.esf, &s),
                            p(&sc.x),
                            dptr(&sc.row_of, &s),
                            t32,
                            m.top_k,
                            rows32,
                            h32,
                            rows4,
                            cap,
                        )?;
                        let gemm_m = if masked { MASKED_CAP32 } else { rows32 };
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
                            dptr(&sc.row_of, &s),
                            t32,
                            m.top_k,
                            rows32,
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
                            t32,
                            h32,
                            m.top_k,
                        )?;
                        ops.add_f32(gpu, p(&sc.h), p(&sc.proj), (t * h) as u64)?;
                    }
                }
            }
            if capture {
                self.captured.push(s.clone_dtoh(&sc.h.slice(..t * h))?);
            }
        }
        // Head: final norm of the wanted rows, BF16, lm_head (f32 logits).
        let n = src.num_logit_rows;
        let n32: u32 = narrow(n, "logit rows")?;
        if n > 0 {
            unsafe {
                ops.rmsnorm(
                    gpu,
                    p(&sc.x),
                    0,
                    p(&sc.h),
                    h32,
                    dptr(&self.weights.final_norm, &s),
                    h32,
                    eps,
                    t32,
                )?;
                ops.gather_rows_bf16(gpu, dptr(&sc.sel, &s), p(&sc.x), src.logit_rows, n32, h32)?;
                k.bf16_gemm.launch(
                    gpu,
                    &GemmArgs {
                        m: n32,
                        n: narrow(c.vocab_size, "vocabulary")?,
                        k: h32,
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
) -> Result<Vec<GroupGeometry>> {
    keys.iter()
        .enumerate()
        .map(|(g, a)| {
            Ok(GroupGeometry {
                num_layers: narrow(
                    layer_kv.iter().filter(|l| l.group == g).count(),
                    "layers in a group",
                )?,
                num_kv_heads: narrow(a.num_kv_heads, "KV heads")?,
                head_dim_qk: narrow(a.head_dim_qk, "QK head dim")?,
                head_dim_v: narrow(a.head_dim_v, "V head dim")?,
                block_size,
                num_blocks: num_blocks[g],
            })
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
