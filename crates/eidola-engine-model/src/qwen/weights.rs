//! Checkpoint tensors → the Qwen3.8 and Qwen3.5 reference weights.
//!
//! The large weights (every projection, FP8 or BF16 as the checkpoint stores
//! them, the embedding, the head and the drafter's input projection) stay in
//! the memory-mapped checkpoint and are decoded one row at a time as a product
//! reads them; Qwen3.8-27B in f32 would be about 108 GB. Norm weights, the
//! DeltaNet's convolution, gates and small projections, and every FP8 block
//! scale are decoded at load; one of those stored in F32 is held in BF16, as the
//! released stacks hold it. A tied head is the embedding's own tensor
//! (`QwenConfig::tie_word_embeddings`), and a tied checkpoint that also
//! carries an `lm_head.weight` is refused. Vision tensors are never read.

use std::sync::Arc;

use super::config::{LayerKind, QwenConfig};
use super::linear::{Bf16Matrix, Linear};
use super::numeric::bf16_round_slice;
use crate::error::{Error, Result};
use crate::safetensors::{Dtype, WeightSet};
use crate::tensor::Matrix;

const PREFIX: &str = "model.language_model";

/// A zero-centred RMSNorm's effective weight `1 + w` (f32), as the reference
/// computes it.
#[derive(Clone, Debug)]
pub struct ZcNorm(pub Vec<f32>);

pub struct MlpWeights {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

pub struct DeltaNetWeights {
    pub in_proj_qkv: Linear,
    pub in_proj_z: Linear,
    /// `[value heads, hidden]`, BF16 values.
    pub in_proj_b: Matrix,
    pub in_proj_a: Matrix,
    /// `[conv_dim, conv_kernel]`.
    pub conv: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    /// The gated norm's plain weight, `[value head dim]`.
    pub norm: Vec<f32>,
    pub out_proj: Linear,
}

pub struct AttentionWeights {
    /// `[heads · 2 · head_dim, hidden]`: per head `h`, rows `2h·d..(2h+1)·d`
    /// are its query and the next `d` its output gate.
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: ZcNorm,
    pub k_norm: ZcNorm,
}

pub enum MixerWeights {
    DeltaNet(DeltaNetWeights),
    Attention(AttentionWeights),
}

pub struct LayerWeights {
    pub input_norm: ZcNorm,
    pub post_norm: ZcNorm,
    pub mixer: MixerWeights,
    pub mlp: MlpWeights,
}

pub struct MtpWeights {
    pub pre_fc_norm_embedding: ZcNorm,
    pub pre_fc_norm_hidden: ZcNorm,
    /// `[hidden, 2 · hidden]`, input `[normed embedding ‖ normed hidden]`.
    pub fc: Bf16Matrix,
    pub layer: LayerWeights,
    pub norm: ZcNorm,
}

pub struct QwenWeights {
    pub config: QwenConfig,
    pub store: Arc<WeightSet>,
    pub embed: Bf16Matrix,
    /// The output head: `lm_head.weight`, or with a tied head the embedding's
    /// tensor itself.
    pub head: Bf16Matrix,
    pub final_norm: ZcNorm,
    pub layers: Vec<LayerWeights>,
    pub mtp: Vec<MtpWeights>,
}

/// A small tensor in f32, holding BF16 values. One stored in F32 (Qwen3.5's `A_log` and
/// the DeltaNet's gated-norm weight) is rounded to BF16, ties to even: the released
/// stacks build the model in its configuration's dtype (`bfloat16`) and load every
/// parameter into it, Transformers' `from_pretrained` and vLLM's loader alike.
fn small(store: &WeightSet, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let t = store.get(name)?;
    let stored = if t.dtype == Dtype::F32 {
        Dtype::F32
    } else {
        Dtype::Bf16
    };
    t.expect(name, stored, shape)?;
    let mut v = t.to_f32(name)?;
    if stored == Dtype::F32 {
        bf16_round_slice(&mut v);
    }
    if let Some(bad) = v.iter().find(|x| !x.is_finite()) {
        return Err(Error::layout(name, format!("value {bad} is not finite")));
    }
    Ok(v)
}

fn zc_norm(store: &WeightSet, name: &str, n: usize) -> Result<ZcNorm> {
    Ok(ZcNorm(
        small(store, name, &[n])?
            .into_iter()
            .map(|w| 1.0 + w)
            .collect(),
    ))
}

/// A large projection in the checkpoint's format.
fn linear(
    store: &Arc<WeightSet>,
    prefix: &str,
    rows: usize,
    cols: usize,
    c: &QwenConfig,
) -> Result<Linear> {
    Linear::open(store, prefix, rows, cols, c.weights)
}

fn mlp(store: &Arc<WeightSet>, p: &str, c: &QwenConfig) -> Result<MlpWeights> {
    let (h, i) = (c.hidden_size, c.intermediate_size);
    Ok(MlpWeights {
        gate: linear(store, &format!("{p}.mlp.gate_proj"), i, h, c)?,
        up: linear(store, &format!("{p}.mlp.up_proj"), i, h, c)?,
        down: linear(store, &format!("{p}.mlp.down_proj"), h, i, c)?,
    })
}

fn attention(store: &Arc<WeightSet>, p: &str, c: &QwenConfig) -> Result<AttentionWeights> {
    let a = &c.attention;
    let h = c.hidden_size;
    let p = format!("{p}.self_attn");
    Ok(AttentionWeights {
        q_proj: linear(
            store,
            &format!("{p}.q_proj"),
            2 * a.num_q_heads * a.head_dim,
            h,
            c,
        )?,
        k_proj: linear(
            store,
            &format!("{p}.k_proj"),
            a.num_kv_heads * a.head_dim,
            h,
            c,
        )?,
        v_proj: linear(
            store,
            &format!("{p}.v_proj"),
            a.num_kv_heads * a.head_dim,
            h,
            c,
        )?,
        o_proj: linear(
            store,
            &format!("{p}.o_proj"),
            h,
            a.num_q_heads * a.head_dim,
            c,
        )?,
        q_norm: zc_norm(store, &format!("{p}.q_norm.weight"), a.head_dim)?,
        k_norm: zc_norm(store, &format!("{p}.k_norm.weight"), a.head_dim)?,
    })
}

fn deltanet(store: &Arc<WeightSet>, p: &str, c: &QwenConfig) -> Result<DeltaNetWeights> {
    let d = &c.deltanet;
    let h = c.hidden_size;
    let p = format!("{p}.linear_attn");
    let nv = d.num_v_heads;
    let bf16_matrix = |name: &str, rows: usize| -> Result<Matrix> {
        Ok(Matrix::from_vec(rows, h, small(store, name, &[rows, h])?))
    };
    Ok(DeltaNetWeights {
        in_proj_qkv: linear(store, &format!("{p}.in_proj_qkv"), d.conv_dim(), h, c)?,
        in_proj_z: linear(store, &format!("{p}.in_proj_z"), d.value_dim(), h, c)?,
        in_proj_b: bf16_matrix(&format!("{p}.in_proj_b.weight"), nv)?,
        in_proj_a: bf16_matrix(&format!("{p}.in_proj_a.weight"), nv)?,
        conv: small(
            store,
            &format!("{p}.conv1d.weight"),
            &[d.conv_dim(), 1, d.conv_kernel],
        )?,
        a_log: small(store, &format!("{p}.A_log"), &[nv])?,
        dt_bias: small(store, &format!("{p}.dt_bias"), &[nv])?,
        norm: small(store, &format!("{p}.norm.weight"), &[d.v_head_dim])?,
        out_proj: linear(store, &format!("{p}.out_proj"), h, d.value_dim(), c)?,
    })
}

fn layer(store: &Arc<WeightSet>, p: &str, kind: LayerKind, c: &QwenConfig) -> Result<LayerWeights> {
    let h = c.hidden_size;
    Ok(LayerWeights {
        input_norm: zc_norm(store, &format!("{p}.input_layernorm.weight"), h)?,
        post_norm: zc_norm(store, &format!("{p}.post_attention_layernorm.weight"), h)?,
        mixer: match kind {
            LayerKind::DeltaNet => MixerWeights::DeltaNet(deltanet(store, p, c)?),
            LayerKind::Attention => MixerWeights::Attention(attention(store, p, c)?),
        },
        mlp: mlp(store, p, c)?,
    })
}

impl QwenWeights {
    pub fn load(store: Arc<WeightSet>, config: QwenConfig) -> Result<Self> {
        let (v, h) = (config.vocab_size, config.hidden_size);
        let mut layers = Vec::with_capacity(config.layers.len());
        for spec in &config.layers {
            let p = format!("{PREFIX}.layers.{}", spec.checkpoint_index);
            layers.push(layer(&store, &p, spec.kind, &config)?);
        }
        let mut mtp = Vec::with_capacity(config.mtp_layers);
        for i in 0..config.mtp_layers {
            mtp.push(MtpWeights {
                pre_fc_norm_embedding: zc_norm(&store, "mtp.pre_fc_norm_embedding.weight", h)?,
                pre_fc_norm_hidden: zc_norm(&store, "mtp.pre_fc_norm_hidden.weight", h)?,
                fc: Bf16Matrix::open(&store, "mtp.fc.weight", h, 2 * h)?,
                layer: layer(
                    &store,
                    &format!("mtp.layers.{i}"),
                    LayerKind::Attention,
                    &config,
                )?,
                norm: zc_norm(&store, "mtp.norm.weight", h)?,
            });
        }
        let embed_name = format!("{PREFIX}.embed_tokens.weight");
        let head = if config.tie_word_embeddings {
            if store.contains("lm_head.weight") {
                return Err(Error::layout(
                    "lm_head.weight",
                    "the head is tied to the embedding, and the checkpoint has one of its own",
                ));
            }
            Bf16Matrix::open(&store, &embed_name, v, h)?
        } else {
            Bf16Matrix::open(&store, "lm_head.weight", v, h)?
        };
        Ok(QwenWeights {
            embed: Bf16Matrix::open(&store, &embed_name, v, h)?,
            head,
            final_norm: zc_norm(&store, &format!("{PREFIX}.norm.weight"), h)?,
            layers,
            mtp,
            config,
            store,
        })
    }
}
