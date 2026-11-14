//! Checkpoint tensors → f32 reference weights.
//!
//! Dense tensors (attention, norms, dense FFNs, router, embeddings, head) are
//! dequantised once at load. Routed experts stay in the memory-mapped files
//! and are dequantised per forward call, so a model with hundreds of experts
//! per layer never holds them all in f32 at once.

use std::sync::Arc;

use crate::config::{AttentionSpec, FfnKind, ModelConfig};
use crate::dequant::{QkvSharding, QkvWeight, deinterleave_qkv, fp8_block_dequant, mxfp4_dequant};
use crate::error::{Error, Result};
use crate::safetensors::{Dtype, WeightSet};
use crate::tensor::Matrix;

#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Fold `attention_value_scale` into the `o_proj` columns at load instead
    /// of multiplying V at run time. Attention output is linear in V, so the
    /// two are the same function; the unfolded path exists to test that.
    pub fold_value_scale: bool,
    /// Load the `model.mtp.layers.*` draft layers (as many as the checkpoint
    /// ships, or the config declares when it says so).
    pub load_mtp: bool,
    /// Rank-major chunk count of the fused QKV. `None` reads `tp_size` from
    /// the checkpoint metadata, else falls back to the config's
    /// `num_key_value_heads` (the convention vLLM uses).
    pub qkv_chunks: Option<usize>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            fold_value_scale: true,
            load_mtp: true,
            qkv_chunks: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// `[q_size + k_size + v_size, hidden]`, rows in unsharded `[Q | K | V]`
    /// head order.
    pub qkv: Matrix,
    /// `[hidden, num_q_heads · head_dim_v]`; already multiplied by the value
    /// scale when it was folded.
    pub o_proj: Matrix,
    pub sinks: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct DenseFfnWeights {
    pub gate: Matrix,
    pub up: Matrix,
    pub down: Matrix,
}

/// Where one routed expert's tensors live; dequantised on demand.
#[derive(Debug, Clone)]
pub struct ExpertSource {
    prefix: String,
}

#[derive(Debug, Clone)]
pub struct MoeWeights {
    /// Router `[num_experts, hidden]` (stored BF16, used in f32).
    pub router: Matrix,
    pub correction_bias: Vec<f32>,
    pub experts: Vec<ExpertSource>,
}

#[derive(Debug, Clone)]
pub enum FfnWeights {
    Dense(DenseFfnWeights),
    Moe(MoeWeights),
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub input_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub attention: AttentionWeights,
    pub ffn: FfnWeights,
}

#[derive(Debug, Clone)]
pub struct MtpWeights {
    pub enorm: Vec<f32>,
    pub hnorm: Vec<f32>,
    /// `[hidden, 2·hidden]`, input is `[enorm(embed) ‖ hnorm(h_prev)]`.
    pub eh_proj: Matrix,
    pub input_norm: Vec<f32>,
    /// Stored as `pre_mlp_layernorm`.
    pub post_attention_norm: Vec<f32>,
    pub attention: AttentionWeights,
    pub ffn: DenseFfnWeights,
    pub final_norm: Vec<f32>,
}

pub struct ModelWeights {
    pub config: ModelConfig,
    pub embed: Matrix,
    pub final_norm: Vec<f32>,
    pub lm_head: Matrix,
    pub layers: Vec<LayerWeights>,
    pub mtp: Vec<MtpWeights>,
    /// Whether `o_proj` already carries `attention_value_scale`.
    pub value_scale_folded: bool,
    pub qkv_chunks: usize,
    store: Arc<WeightSet>,
}

impl std::fmt::Debug for ModelWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelWeights")
            .field("layers", &self.layers.len())
            .field("mtp", &self.mtp.len())
            .field("value_scale_folded", &self.value_scale_folded)
            .field("qkv_chunks", &self.qkv_chunks)
            .finish()
    }
}

/// Dequantised expert, `[intermediate, hidden]` / `[hidden, intermediate]`.
pub struct ExpertWeights {
    pub gate: Matrix,
    pub up: Matrix,
    pub down: Matrix,
}

impl ModelWeights {
    pub fn load(store: Arc<WeightSet>, config: ModelConfig, opts: &LoadOptions) -> Result<Self> {
        let qkv_chunks = match opts.qkv_chunks {
            Some(c) => c,
            None => match store.metadata("tp_size") {
                Some(s) => s.trim().parse().map_err(|_| {
                    Error::Input(format!(
                        "checkpoint metadata tp_size {s:?} is not an integer"
                    ))
                })?,
                None => config.num_key_value_heads,
            },
        };
        let loader = Loader {
            store: &store,
            config: &config,
            qkv_chunks,
        };
        let h = config.hidden_size;
        let v = config.vocab_size;

        let embed = loader.float_matrix("model.embed_tokens.weight", v, h)?;
        let lm_head = loader.float_matrix("lm_head.weight", v, h)?;
        let final_norm = loader.float_vec("model.norm.weight", h)?;

        let fold = opts.fold_value_scale && config.attention_value_scale.is_some();
        let mut layers = Vec::with_capacity(config.layers.len());
        for spec in &config.layers {
            let p = format!("model.layers.{}", spec.source_index);
            let attention = loader.attention(&p, &spec.attention, fold)?;
            let ffn = match spec.ffn {
                FfnKind::Dense => {
                    FfnWeights::Dense(loader.dense_ffn(&p, config.dense_intermediate_size)?)
                }
                FfnKind::Moe => FfnWeights::Moe(loader.moe(&p)?),
            };
            layers.push(LayerWeights {
                input_norm: loader.float_vec(&format!("{p}.input_layernorm.weight"), h)?,
                post_attention_norm: loader
                    .float_vec(&format!("{p}.post_attention_layernorm.weight"), h)?,
                attention,
                ffn,
            });
        }

        let mut mtp = Vec::new();
        if opts.load_mtp {
            let shipped = (0..)
                .take_while(|i| store.contains(&format!("model.mtp.layers.{i}.eh_proj.weight")))
                .count();
            let count = config.mtp.declared_layers.unwrap_or(shipped);
            if count > shipped {
                return Err(Error::MissingTensor(format!(
                    "model.mtp.layers.{shipped}.* (config declares {count} MTP layers)"
                )));
            }
            for i in 0..count {
                let p = format!("model.mtp.layers.{i}");
                mtp.push(MtpWeights {
                    enorm: loader.float_vec(&format!("{p}.enorm.weight"), h)?,
                    hnorm: loader.float_vec(&format!("{p}.hnorm.weight"), h)?,
                    eh_proj: loader.float_matrix(&format!("{p}.eh_proj.weight"), h, 2 * h)?,
                    input_norm: loader.float_vec(&format!("{p}.input_layernorm.weight"), h)?,
                    post_attention_norm: loader
                        .float_vec(&format!("{p}.pre_mlp_layernorm.weight"), h)?,
                    attention: loader.attention(&p, &config.mtp.attention, fold)?,
                    ffn: loader.dense_ffn(&p, config.mtp.intermediate_size)?,
                    final_norm: loader.float_vec(&format!("{p}.final_layernorm.weight"), h)?,
                });
            }
        }

        Ok(ModelWeights {
            embed,
            final_norm,
            lm_head,
            layers,
            mtp,
            value_scale_folded: fold,
            qkv_chunks,
            config,
            store,
        })
    }

    /// Dequantise one routed expert of layer `layer` (position in this model).
    pub fn expert(&self, layer: usize, expert: usize) -> Result<ExpertWeights> {
        let FfnWeights::Moe(moe) = &self.layers[layer].ffn else {
            return Err(Error::Input(format!("layer {layer} has no experts")));
        };
        let spec = self
            .config
            .moe
            .as_ref()
            .ok_or_else(|| Error::Input("config has no MoE block".into()))?;
        let src = &moe.experts[expert];
        let loader = Loader {
            store: &self.store,
            config: &self.config,
            qkv_chunks: self.qkv_chunks,
        };
        let (h, i) = (self.config.hidden_size, spec.intermediate_size);
        Ok(ExpertWeights {
            gate: loader.expert_linear(&format!("{}.gate_proj", src.prefix), i, h)?,
            up: loader.expert_linear(&format!("{}.up_proj", src.prefix), i, h)?,
            down: loader.expert_linear(&format!("{}.down_proj", src.prefix), h, i)?,
        })
    }

    pub fn store(&self) -> &WeightSet {
        &self.store
    }
}

struct Loader<'a> {
    store: &'a WeightSet,
    config: &'a ModelConfig,
    qkv_chunks: usize,
}

impl Loader<'_> {
    fn float_vec(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        let t = self.store.get(name)?;
        if t.numel() != len || t.shape.len() != 1 {
            return Err(Error::layout(
                name,
                format!("shape {:?}, expected [{len}]", t.shape),
            ));
        }
        t.to_f32(name)
    }

    fn float_matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Matrix> {
        let t = self.store.get(name)?;
        if t.shape != [rows, cols] {
            return Err(Error::layout(
                name,
                format!("shape {:?}, expected [{rows}, {cols}]", t.shape),
            ));
        }
        Ok(Matrix::from_vec(rows, cols, t.to_f32(name)?))
    }

    fn fp8_block(&self) -> [usize; 2] {
        self.config
            .quant
            .as_ref()
            .map(|q| q.fp8_block)
            .unwrap_or([128, 128])
    }

    /// A `{prefix}.weight` linear that is either float or FP8 with
    /// `{prefix}.weight_scale_inv`.
    fn linear(&self, prefix: &str, rows: usize, cols: usize) -> Result<Matrix> {
        let wname = format!("{prefix}.weight");
        let sname = format!("{prefix}.weight_scale_inv");
        let w = self.store.get(&wname)?;
        if w.dtype != Dtype::F8E4M3 {
            return self.float_matrix(&wname, rows, cols);
        }
        if w.shape != [rows, cols] {
            return Err(Error::layout(
                &wname,
                format!("shape {:?}, expected [{rows}, {cols}]", w.shape),
            ));
        }
        let s = self.store.get(&sname)?;
        if s.dtype != Dtype::F32 || s.shape.len() != 2 {
            return Err(Error::layout(
                &sname,
                format!("{:?} {:?}", s.dtype, s.shape),
            ));
        }
        let scales = s.to_f32(&sname)?;
        fp8_block_dequant(
            &wname,
            w.data,
            rows,
            cols,
            &scales,
            [s.shape[0], s.shape[1]],
            self.fp8_block(),
        )
    }

    fn attention(
        &self,
        prefix: &str,
        spec: &AttentionSpec,
        fold: bool,
    ) -> Result<AttentionWeights> {
        let h = self.config.hidden_size;
        let wname = format!("{prefix}.self_attn.qkv_proj.weight");
        let w = self.store.get(&wname)?;
        if w.shape != [spec.qkv_rows(), h] {
            return Err(Error::layout(
                &wname,
                format!("shape {:?}, expected [{}, {h}]", w.shape, spec.qkv_rows()),
            ));
        }
        let sharding = QkvSharding {
            chunks: self.qkv_chunks,
        };
        let qkv = if w.dtype == Dtype::F8E4M3 {
            let sname = format!("{prefix}.self_attn.qkv_proj.weight_scale_inv");
            let s = self.store.get(&sname)?;
            if s.dtype != Dtype::F32 || s.shape.len() != 2 {
                return Err(Error::layout(
                    &sname,
                    format!("{:?} {:?}", s.dtype, s.shape),
                ));
            }
            let scales = s.to_f32(&sname)?;
            deinterleave_qkv(
                &wname,
                QkvWeight::Fp8 {
                    q: w.data,
                    scales: &scales,
                    scale_shape: [s.shape[0], s.shape[1]],
                },
                h,
                spec,
                sharding,
                self.fp8_block(),
            )?
        } else {
            deinterleave_qkv(
                &wname,
                QkvWeight::Float(w.to_f32(&wname)?),
                h,
                spec,
                sharding,
                self.fp8_block(),
            )?
        };

        let mut o_proj = self.linear(&format!("{prefix}.self_attn.o_proj"), h, spec.o_size())?;
        if fold && let Some(s) = self.config.attention_value_scale {
            for x in &mut o_proj.data {
                *x *= s;
            }
        }
        let sinks = if spec.has_sinks {
            Some(self.float_vec(
                &format!("{prefix}.self_attn.attention_sink_bias"),
                spec.num_q_heads,
            )?)
        } else {
            let n = format!("{prefix}.self_attn.attention_sink_bias");
            if self.store.contains(&n) {
                return Err(Error::layout(
                    &n,
                    "checkpoint has sinks the config does not enable for this layer",
                ));
            }
            None
        };
        Ok(AttentionWeights { qkv, o_proj, sinks })
    }

    fn dense_ffn(&self, prefix: &str, inter: usize) -> Result<DenseFfnWeights> {
        let h = self.config.hidden_size;
        Ok(DenseFfnWeights {
            gate: self.linear(&format!("{prefix}.mlp.gate_proj"), inter, h)?,
            up: self.linear(&format!("{prefix}.mlp.up_proj"), inter, h)?,
            down: self.linear(&format!("{prefix}.mlp.down_proj"), h, inter)?,
        })
    }

    fn moe(&self, prefix: &str) -> Result<MoeWeights> {
        let spec = self
            .config
            .moe
            .as_ref()
            .ok_or_else(|| Error::InvalidConfig("MoE layer without MoE config".into()))?;
        let h = self.config.hidden_size;
        let router =
            self.float_matrix(&format!("{prefix}.mlp.gate.weight"), spec.num_experts, h)?;
        let correction_bias = self.float_vec(
            &format!("{prefix}.mlp.gate.e_score_correction_bias"),
            spec.num_experts,
        )?;
        let mut experts = Vec::with_capacity(spec.num_experts);
        for e in 0..spec.num_experts {
            let p = format!("{prefix}.mlp.experts.{e}");
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let n = format!("{p}.{proj}.weight");
                if !self.store.contains(&n) {
                    return Err(Error::MissingTensor(n));
                }
            }
            experts.push(ExpertSource { prefix: p });
        }
        Ok(MoeWeights {
            router,
            correction_bias,
            experts,
        })
    }

    /// An expert projection: MXFP4 (`.weight` U8 `[rows, cols/2]` +
    /// `.weight_scale` U8 `[rows, cols/32]`), FP8 block-scaled, or float.
    fn expert_linear(&self, prefix: &str, rows: usize, cols: usize) -> Result<Matrix> {
        let wname = format!("{prefix}.weight");
        let w = self.store.get(&wname)?;
        if w.dtype != Dtype::U8 {
            return self.linear(prefix, rows, cols);
        }
        let block = self
            .config
            .quant
            .as_ref()
            .and_then(|q| q.mxfp4_block)
            .ok_or_else(|| {
                Error::layout(
                    &wname,
                    "packed U8 expert but the config declares no MXFP4 storage",
                )
            })?;
        let sname = format!("{prefix}.weight_scale");
        let s = self.store.get(&sname)?;
        w.expect(&wname, Dtype::U8, &[rows, cols / 2])?;
        s.expect(&sname, Dtype::U8, &[rows, cols / block])?;
        mxfp4_dequant(&wname, w.data, s.data, rows, cols, block)
    }
}
