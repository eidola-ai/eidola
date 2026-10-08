//! MiMo-V2 `config.json` → a validated, typed model description.
//!
//! The per-layer table is derived from the config's own fields
//! (`hybrid_layer_pattern`, `moe_layer_freq`, the `swa_*` overrides), never
//! from a hard-coded ratio or index list. Anything the reference forward does
//! not implement is rejected here, loudly, rather than computed differently.

use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::error::{Error, Result};

/// The subset of `config.json` the text model reads. Unknown fields
/// (vision, audio, processor) are ignored; every field that changes the text
/// forward is either honoured or rejected in [`ModelConfig::from_raw`].
#[derive(Debug, Clone, Deserialize)]
pub struct RawConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: Option<usize>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub v_head_dim: usize,
    pub swa_num_attention_heads: Option<usize>,
    pub swa_num_key_value_heads: Option<usize>,
    pub swa_head_dim: Option<usize>,
    pub swa_v_head_dim: Option<usize>,
    pub rope_theta: f64,
    pub swa_rope_theta: Option<f64>,
    pub rope_parameters: Option<Value>,
    pub rope_scaling: Option<Value>,
    #[serde(default = "one")]
    pub partial_rotary_factor: f64,
    pub sliding_window: Option<usize>,
    pub sliding_window_size: Option<usize>,
    #[serde(default)]
    pub add_full_attention_sink_bias: bool,
    #[serde(default)]
    pub add_swa_attention_sink_bias: bool,
    pub attention_value_scale: Option<f64>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f64,
    pub attention_projection_layout: Option<String>,
    pub attention_chunk_size: Option<usize>,
    pub hybrid_layer_pattern: Vec<u8>,
    pub hybrid_block_size: Option<Value>,
    pub moe_layer_freq: Option<Vec<u8>>,
    pub n_routed_experts: Option<usize>,
    pub n_shared_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub n_group: Option<usize>,
    pub topk_group: Option<usize>,
    #[serde(default = "yes")]
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: Option<f64>,
    pub scoring_func: Option<String>,
    pub topk_method: Option<String>,
    pub moe_router_dtype: Option<String>,
    pub layernorm_epsilon: f64,
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
    pub num_nextn_predict_layers: Option<usize>,
    pub quantization_config: Option<RawQuantConfig>,
}

fn one() -> f64 {
    1.0
}
fn yes() -> bool {
    true
}

/// Every key of `quantization_config` is read: an unknown one is refused, so
/// no quantization setting is dropped unseen.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawQuantConfig {
    pub quant_method: String,
    pub fmt: Option<String>,
    pub weight_block_size: Option<Vec<usize>>,
    pub store_dtype: Option<String>,
    pub mxfp4_block_size: Option<usize>,
    #[serde(default)]
    pub ignored_layers: Vec<String>,
    pub activation_scheme: Option<String>,
}

/// Global (full causal) or sliding-window attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    Global,
    /// Each query attends to itself and the `window - 1` positions before it.
    Sliding {
        window: usize,
    },
}

/// Everything attention needs for one layer.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionSpec {
    pub kind: AttentionKind,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim_qk: usize,
    pub head_dim_v: usize,
    /// Rotated dims are `[0, rope_dim)` of each Q/K head (NeoX rotate-half);
    /// the rest pass through unrotated.
    pub rope_dim: usize,
    pub rope_theta: f32,
    /// A learned per-query-head logit appended to each softmax row and
    /// dropped after normalisation.
    pub has_sinks: bool,
    /// Softmax temperature, `head_dim_qk^-0.5`.
    pub softmax_scale: f32,
}

impl AttentionSpec {
    pub fn q_size(&self) -> usize {
        self.num_q_heads * self.head_dim_qk
    }
    pub fn k_size(&self) -> usize {
        self.num_kv_heads * self.head_dim_qk
    }
    pub fn v_size(&self) -> usize {
        self.num_kv_heads * self.head_dim_v
    }
    pub fn qkv_rows(&self) -> usize {
        self.q_size() + self.k_size() + self.v_size()
    }
    /// Width of the attention output fed to `o_proj`.
    pub fn o_size(&self) -> usize {
        self.num_q_heads * self.head_dim_v
    }
    pub fn group_size(&self) -> usize {
        self.num_q_heads / self.num_kv_heads
    }
    pub fn window(&self) -> Option<usize> {
        match self.kind {
            AttentionKind::Global => None,
            AttentionKind::Sliding { window } => Some(window),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnKind {
    /// SwiGLU with the config's `intermediate_size`.
    Dense,
    /// Routed experts only; see [`MoeSpec`].
    Moe,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerSpec {
    /// Position of the layer in this (possibly truncated) model.
    pub index: usize,
    /// Index in the checkpoint's tensor names (`model.layers.{source_index}`).
    pub source_index: usize,
    pub attention: AttentionSpec,
    pub ffn: FfnKind,
}

/// The routed-expert block: sigmoid scores, selection on scores plus
/// `e_score_correction_bias`, weights from the unbiased scores.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeSpec {
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
}

/// The multi-token-prediction draft layers shipped inside the checkpoint
/// (`model.mtp.layers.{i}`).
#[derive(Debug, Clone, PartialEq)]
pub struct MtpSpec {
    /// The count the config declares; `None` when the config leaves it null
    /// even though the checkpoint may still ship the weights.
    pub declared_layers: Option<usize>,
    /// MTP layers use the sliding-window attention geometry.
    pub attention: AttentionSpec,
    pub intermediate_size: usize,
}

/// Checkpoint storage formats.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantSpec {
    /// FP8 e4m3 weights with F32 `weight_scale_inv` per `[block_rows, block_cols]` tile.
    pub fp8_block: [usize; 2],
    /// Routed experts stored as packed E2M1 with E8M0 scales per this many K elements.
    pub mxfp4_block: Option<usize>,
    /// Modules stored unquantized (BF16) despite the FP8 method.
    pub ignored_layers: Vec<String>,
    /// How activations are quantized for the FP8 GEMMs, as the checkpoint
    /// declares it (`dynamic`: per-token scales computed at run time). The
    /// reference forward runs activations unquantized and does not read it.
    pub activation_scheme: Option<String>,
}

/// A validated MiMo-V2 text model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub hidden_size: usize,
    /// Rows of `embed_tokens` / `lm_head`. Checkpoints pad this past the
    /// tokenizer's last real token; logits for padded ids are computed like
    /// any other and are the caller's to mask.
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    /// V is multiplied by this before attention (applied either literally or
    /// folded into `o_proj`; see `LoadOptions::fold_value_scale`).
    pub attention_value_scale: Option<f32>,
    pub dense_intermediate_size: usize,
    pub moe: Option<MoeSpec>,
    pub layers: Vec<LayerSpec>,
    pub mtp: MtpSpec,
    pub quant: Option<QuantSpec>,
    pub max_position_embeddings: usize,
    /// Layer count of the full checkpoint (before any truncation).
    pub source_num_layers: usize,
    /// The top-level `num_key_value_heads`. Checkpoints that do not record
    /// the rank-major chunk count of their fused QKV were saved with this
    /// many chunks.
    pub num_key_value_heads: usize,
}

impl ModelConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        Self::from_json_bytes(&bytes, &path.display().to_string())
    }

    pub fn from_json_bytes(bytes: &[u8], what: &str) -> Result<Self> {
        let raw: RawConfig = serde_json::from_slice(bytes).map_err(|e| Error::Json {
            what: what.to_string(),
            source: e,
        })?;
        Self::from_raw(&raw)
    }

    pub fn from_raw(raw: &RawConfig) -> Result<Self> {
        let unsupported = |m: String| Err(Error::UnsupportedConfig(m));
        let invalid = |m: String| Err(Error::InvalidConfig(m));

        if raw.model_type != "mimo_v2" {
            return unsupported(format!(
                "model_type {:?} (expected \"mimo_v2\")",
                raw.model_type
            ));
        }
        if !raw.architectures.is_empty()
            && !raw.architectures.iter().any(|a| a == "MiMoV2ForCausalLM")
        {
            return unsupported(format!("architectures {:?}", raw.architectures));
        }
        if raw.hidden_act != "silu" {
            return unsupported(format!("hidden_act {:?}", raw.hidden_act));
        }
        if raw.attention_bias {
            return unsupported("attention_bias = true".into());
        }
        if raw.attention_dropout != 0.0 {
            return unsupported(format!("attention_dropout {}", raw.attention_dropout));
        }
        if raw.tie_word_embeddings {
            return unsupported("tie_word_embeddings = true".into());
        }
        match raw.attention_projection_layout.as_deref() {
            Some("fused_qkv") => {}
            other => {
                return unsupported(format!(
                    "attention_projection_layout {other:?} (only \"fused_qkv\" checkpoints are supported)"
                ));
            }
        }
        if raw.hybrid_block_size.as_ref().is_some_and(|v| !v.is_null()) {
            return unsupported("hybrid_block_size is set".into());
        }
        if raw.n_shared_experts.unwrap_or(0) != 0 {
            return unsupported(format!("n_shared_experts {:?}", raw.n_shared_experts));
        }
        if let Some(rs) = &raw.rope_scaling
            && !rs.is_null()
        {
            check_default_rope(rs, "rope_scaling")?;
        }

        // RoPE: the HF reference reads the partial factor and the global theta
        // from `rope_parameters` for the rotary tables but from the top level
        // for the attention split; refuse configs where those disagree.
        let mut partial = raw.partial_rotary_factor;
        let mut global_theta = raw.rope_theta;
        if let Some(rp) = &raw.rope_parameters
            && !rp.is_null()
        {
            check_default_rope(rp, "rope_parameters")?;
            if let Some(p) = rp.get("partial_rotary_factor").and_then(Value::as_f64) {
                if p != partial {
                    return invalid(format!(
                        "rope_parameters.partial_rotary_factor {p} != partial_rotary_factor {partial}"
                    ));
                }
                partial = p;
            }
            if let Some(t) = rp.get("rope_theta").and_then(Value::as_f64) {
                if t != global_theta {
                    return invalid(format!(
                        "rope_parameters.rope_theta {t} != rope_theta {global_theta}"
                    ));
                }
                global_theta = t;
            }
        }
        let swa_theta = raw.swa_rope_theta.unwrap_or(global_theta);

        let window = match (raw.sliding_window, raw.sliding_window_size) {
            (Some(a), Some(b)) if a != b => {
                return invalid(format!("sliding_window {a} != sliding_window_size {b}"));
            }
            (Some(a), _) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };

        let n = raw.num_hidden_layers;
        if raw.hybrid_layer_pattern.len() != n {
            return invalid(format!(
                "hybrid_layer_pattern has {} entries for {n} layers",
                raw.hybrid_layer_pattern.len()
            ));
        }
        let moe_freq = match &raw.moe_layer_freq {
            Some(v) => v.clone(),
            None => vec![0; n],
        };
        if moe_freq.len() != n {
            return invalid(format!(
                "moe_layer_freq has {} entries for {n} layers",
                moe_freq.len()
            ));
        }

        let global = attention_spec(
            AttentionKind::Global,
            raw.num_attention_heads,
            raw.num_key_value_heads,
            raw.head_dim,
            raw.v_head_dim,
            partial,
            global_theta,
            raw.add_full_attention_sink_bias,
        )?;
        let sliding = match window {
            Some(w) if w > 0 => Some(attention_spec(
                AttentionKind::Sliding { window: w },
                raw.swa_num_attention_heads
                    .unwrap_or(raw.num_attention_heads),
                raw.swa_num_key_value_heads
                    .unwrap_or(raw.num_key_value_heads),
                raw.swa_head_dim.unwrap_or(raw.head_dim),
                raw.swa_v_head_dim.unwrap_or(raw.v_head_dim),
                partial,
                swa_theta,
                raw.add_swa_attention_sink_bias,
            )?),
            _ => None,
        };

        let moe = if moe_freq.iter().any(|&f| f != 0) {
            let num_experts = raw.n_routed_experts.ok_or_else(|| {
                Error::InvalidConfig("MoE layers without n_routed_experts".into())
            })?;
            let top_k = raw.num_experts_per_tok.ok_or_else(|| {
                Error::InvalidConfig("MoE layers without num_experts_per_tok".into())
            })?;
            let intermediate_size = raw.moe_intermediate_size.ok_or_else(|| {
                Error::InvalidConfig("MoE layers without moe_intermediate_size".into())
            })?;
            match raw.scoring_func.as_deref() {
                Some("sigmoid") => {}
                other => return unsupported(format!("scoring_func {other:?}")),
            }
            match raw.topk_method.as_deref() {
                Some("noaux_tc") => {}
                other => return unsupported(format!("topk_method {other:?}")),
            }
            // Group-limited routing is a no-op exactly when every group is kept.
            let n_group = raw.n_group.unwrap_or(1);
            let topk_group = raw.topk_group.unwrap_or(1);
            if n_group != topk_group {
                return unsupported(format!(
                    "group-limited routing (n_group {n_group}, topk_group {topk_group})"
                ));
            }
            match raw.moe_router_dtype.as_deref() {
                None | Some("bfloat16") | Some("float32") => {}
                other => return unsupported(format!("moe_router_dtype {other:?}")),
            }
            if top_k == 0 || top_k > num_experts {
                return invalid(format!("top-{top_k} of {num_experts} experts"));
            }
            Some(MoeSpec {
                num_experts,
                top_k,
                intermediate_size,
                norm_topk_prob: raw.norm_topk_prob,
                routed_scaling_factor: positive_f32(
                    "routed_scaling_factor",
                    raw.routed_scaling_factor.unwrap_or(1.0),
                )?,
            })
        } else {
            None
        };

        let mut layers = Vec::with_capacity(n);
        for (i, (&pattern, &freq)) in raw.hybrid_layer_pattern.iter().zip(&moe_freq).enumerate() {
            let attention = match pattern {
                0 => global.clone(),
                1 => sliding.clone().ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "layer {i} is sliding-window but no sliding_window is set"
                    ))
                })?,
                p => return invalid(format!("hybrid_layer_pattern[{i}] = {p}")),
            };
            let ffn = match freq {
                0 => FfnKind::Dense,
                1 => FfnKind::Moe,
                f => return invalid(format!("moe_layer_freq[{i}] = {f}")),
            };
            layers.push(LayerSpec {
                index: i,
                source_index: i,
                attention,
                ffn,
            });
        }

        // MTP layers use the SWA geometry and a dense FFN of `intermediate_size`.
        let mtp_attention = sliding.clone().unwrap_or_else(|| global.clone());

        let quant = match &raw.quantization_config {
            None => None,
            Some(q) => Some(quant_spec(q)?),
        };

        let rms_norm_eps = positive_f32("layernorm_epsilon", raw.layernorm_epsilon)?;
        let attention_value_scale = raw
            .attention_value_scale
            .map(|s| positive_f32("attention_value_scale", s))
            .transpose()?;

        Ok(ModelConfig {
            hidden_size: raw.hidden_size,
            vocab_size: raw.vocab_size,
            rms_norm_eps,
            attention_value_scale,
            dense_intermediate_size: raw.intermediate_size,
            moe,
            layers,
            mtp: MtpSpec {
                declared_layers: raw.num_nextn_predict_layers,
                attention: mtp_attention,
                intermediate_size: raw.intermediate_size,
            },
            quant,
            max_position_embeddings: raw.max_position_embeddings,
            source_num_layers: n,
            num_key_value_heads: raw.num_key_value_heads,
        })
    }

    /// A model made of only the listed checkpoint layers, in the given order,
    /// each keeping its own attention type and FFN kind.
    pub fn truncated(&self, keep: &[usize]) -> Result<Self> {
        if keep.is_empty() {
            return Err(Error::Input("truncation keeps no layers".into()));
        }
        let mut layers = Vec::with_capacity(keep.len());
        for (pos, &src) in keep.iter().enumerate() {
            let base = self
                .layers
                .iter()
                .find(|l| l.source_index == src)
                .ok_or_else(|| Error::Input(format!("layer {src} is not in this model")))?;
            if keep[..pos].contains(&src) {
                return Err(Error::Input(format!("layer {src} listed twice")));
            }
            layers.push(LayerSpec {
                index: pos,
                ..base.clone()
            });
        }
        Ok(ModelConfig {
            layers,
            ..self.clone()
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Checkpoint indices of the global-attention layers, in order.
    pub fn global_layer_indices(&self) -> Vec<usize> {
        self.layers
            .iter()
            .filter(|l| l.attention.kind == AttentionKind::Global)
            .map(|l| l.source_index)
            .collect()
    }
}

fn check_default_rope(v: &Value, what: &str) -> Result<()> {
    for key in ["rope_type", "type"] {
        if let Some(t) = v.get(key)
            && t.as_str() != Some("default")
        {
            return Err(Error::UnsupportedConfig(format!("{what}.{key} = {t}")));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn attention_spec(
    kind: AttentionKind,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim_qk: usize,
    head_dim_v: usize,
    partial_rotary_factor: f64,
    theta: f64,
    has_sinks: bool,
) -> Result<AttentionSpec> {
    if num_kv_heads == 0 || !num_q_heads.is_multiple_of(num_kv_heads) {
        return Err(Error::InvalidConfig(format!(
            "{num_q_heads} query heads do not group over {num_kv_heads} KV heads"
        )));
    }
    if !(0.0..=1.0).contains(&partial_rotary_factor) {
        return Err(Error::InvalidConfig(format!(
            "partial_rotary_factor {partial_rotary_factor}"
        )));
    }
    // Python's int(head_dim * factor): truncation in double precision.
    let rope_dim = (head_dim_qk as f64 * partial_rotary_factor) as usize;
    if !rope_dim.is_multiple_of(2) || rope_dim > head_dim_qk {
        return Err(Error::InvalidConfig(format!(
            "rotary dim {rope_dim} from head_dim {head_dim_qk} × {partial_rotary_factor}"
        )));
    }
    if head_dim_qk == 0 {
        return Err(Error::InvalidConfig("head_dim 0".into()));
    }
    let rope_theta = positive_f32("rope theta", theta)?;
    let softmax_scale = positive_f32("softmax scale", (head_dim_qk as f64).powf(-0.5))?;
    Ok(AttentionSpec {
        kind,
        num_q_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_v,
        rope_dim,
        rope_theta,
        has_sinks,
        softmax_scale,
    })
}

/// A config float as the forward stores it: converted to f32 first, then required to
/// be finite and strictly positive. Validating the f64 instead would let a value that
/// is fine in double precision but overflows (`rope_theta: 1e40` becomes infinity) or
/// underflows (an epsilon of `1e-50` becomes 0) in f32 through, silently computing a
/// different model.
fn positive_f32(name: &str, value: f64) -> Result<f32> {
    let stored = value as f32;
    if stored.is_finite() && stored > 0.0 {
        Ok(stored)
    } else {
        Err(Error::InvalidConfig(format!(
            "{name} {value} is {stored} as f32; it must be finite and positive"
        )))
    }
}

fn quant_spec(q: &RawQuantConfig) -> Result<QuantSpec> {
    if q.quant_method != "fp8" {
        return Err(Error::UnsupportedConfig(format!(
            "quant_method {:?}",
            q.quant_method
        )));
    }
    match q.fmt.as_deref() {
        None | Some("e4m3") => {}
        other => return Err(Error::UnsupportedConfig(format!("fp8 fmt {other:?}"))),
    }
    let fp8_block = match q.weight_block_size.as_deref() {
        Some([r, c]) if *r > 0 && *c > 0 => [*r, *c],
        other => {
            return Err(Error::UnsupportedConfig(format!(
                "weight_block_size {other:?} (block-wise scales required)"
            )));
        }
    };
    let mxfp4_block = match q.store_dtype.as_deref() {
        None => None,
        Some("mxfp4") => {
            let b = q.mxfp4_block_size.unwrap_or(32);
            if b != 32 {
                return Err(Error::UnsupportedConfig(format!("mxfp4_block_size {b}")));
            }
            Some(b)
        }
        Some(other) => return Err(Error::UnsupportedConfig(format!("store_dtype {other:?}"))),
    };
    Ok(QuantSpec {
        fp8_block,
        mxfp4_block,
        ignored_layers: q.ignored_layers.clone(),
        activation_scheme: q.activation_scheme.clone(),
    })
}
