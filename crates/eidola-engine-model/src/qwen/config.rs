//! Qwen3.8 configuration → a validated, typed model description.
//!
//! The checkpoint's `config.json` nests the text model under `text_config`
//! beside the vision tower and the quantisation block. Every key at every
//! level is read and an unknown key is refused. The per-layer table comes from
//! `layer_types`, never from a hard-coded ratio; `full_attention_interval`
//! must agree with it.

use serde::Deserialize;

use crate::error::{Error, Result};

/// Rows and columns of one FP8 weight-scale block, and the width of one
/// activation-scale block.
pub const QUANT_BLOCK: usize = 128;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct HfConfig {
    architectures: Vec<String>,
    model_type: String,
    #[allow(dead_code)]
    transformers_version: Option<String>,
    tie_word_embeddings: bool,
    language_model_only: Option<bool>,
    image_token_id: Option<u32>,
    video_token_id: Option<u32>,
    #[allow(dead_code)]
    vision_start_token_id: Option<u32>,
    #[allow(dead_code)]
    vision_end_token_id: Option<u32>,
    quantization_config: HfQuant,
    text_config: HfText,
    vision_config: Option<HfVision>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct HfQuant {
    quant_method: String,
    fmt: String,
    activation_scheme: String,
    weight_block_size: Vec<usize>,
    /// Names the loader must not quantise. The checkpoint's own tensors decide
    /// which weights are FP8 (a weight with a `weight_scale_inv` sibling); the
    /// list also names modules this dense model does not have.
    #[allow(dead_code)]
    modules_to_not_convert: Vec<String>,
}

/// The vision tower's description. Read so that no key goes unseen; the text
/// forward uses none of it.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct HfVision {
    deepstack_visual_indexes: Vec<usize>,
    depth: usize,
    hidden_act: String,
    hidden_size: usize,
    in_channels: usize,
    initializer_range: Option<f64>,
    intermediate_size: usize,
    model_type: String,
    num_heads: usize,
    num_position_embeddings: usize,
    out_hidden_size: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    temporal_patch_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct HfRope {
    rope_type: String,
    rope_theta: f64,
    partial_rotary_factor: f64,
    mrope_section: Vec<usize>,
    mrope_interleaved: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct HfText {
    model_type: String,
    dtype: String,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    layer_types: Vec<String>,
    full_attention_interval: usize,
    hidden_act: String,
    rms_norm_eps: f64,
    tie_word_embeddings: bool,
    max_position_embeddings: usize,
    // Full attention.
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    attention_bias: bool,
    attention_dropout: f64,
    attn_output_gate: bool,
    partial_rotary_factor: f64,
    rope_parameters: HfRope,
    // Gated DeltaNet.
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    output_gate_type: String,
    mamba_ssm_dtype: String,
    // Drafter.
    mtp_num_hidden_layers: usize,
    mtp_use_dedicated_embeddings: bool,
    // Read and unused by the forward.
    #[allow(dead_code)]
    initializer_range: Option<f64>,
    #[allow(dead_code)]
    use_cache: Option<bool>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    #[allow(dead_code)]
    pad_token_id: Option<u32>,
}

/// What a layer's token mixer is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated DeltaNet linear attention (`linear_attention`).
    DeltaNet,
    /// Gated full attention (`full_attention`).
    Attention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerSpec {
    /// Index of the layer in the checkpoint (`model.language_model.layers.{i}`).
    pub checkpoint_index: usize,
    pub kind: LayerKind,
}

/// Gated full attention's geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionSpec {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Leading dimensions of each head that RoPE rotates:
    /// `int(head_dim · partial_rotary_factor)`.
    pub rotary_dim: usize,
    pub rope_theta: f64,
    /// `head_dim^-½`, rounded to f32.
    pub softmax_scale: f32,
}

impl AttentionSpec {
    /// Query heads per KV head; query head `h` reads KV head `h / group`.
    pub fn group(&self) -> usize {
        self.num_q_heads / self.num_kv_heads
    }
}

/// Gated DeltaNet's geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaNetSpec {
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
    /// Width of the causal depthwise convolution (taps, current token last).
    pub conv_kernel: usize,
}

impl DeltaNetSpec {
    pub fn key_dim(&self) -> usize {
        self.num_k_heads * self.k_head_dim
    }
    pub fn value_dim(&self) -> usize {
        self.num_v_heads * self.v_head_dim
    }
    /// Channels of the convolution: queries, keys, values.
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    /// Value heads per key head; value head `h` reads key head `h / group`.
    pub fn group(&self) -> usize {
        self.num_v_heads / self.num_k_heads
    }
    /// f32 elements of one layer's recurrent state.
    pub fn state_len(&self) -> usize {
        self.num_v_heads * self.k_head_dim * self.v_head_dim
    }
}

/// A validated Qwen3.8 text model.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    /// The layers this model runs, in order.
    pub layers: Vec<LayerSpec>,
    /// The checkpoint's layer count (layers may be truncated away).
    pub checkpoint_layers: usize,
    pub attention: AttentionSpec,
    pub deltanet: DeltaNetSpec,
    /// MTP draft layers (`mtp.layers.{i}`), each of the full-attention kind.
    pub mtp_layers: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    /// Token ids of the vision placeholders (refused: vision is not served).
    pub vision_token_ids: Vec<u32>,
    /// The config describes a vision tower (never loaded here).
    pub has_vision: bool,
}

fn positive_f32(v: f64, name: &str) -> Result<f32> {
    let f = v as f32;
    if !(f.is_finite() && f > 0.0) {
        return Err(Error::InvalidConfig(format!(
            "{name} = {v} is not a positive finite f32"
        )));
    }
    Ok(f)
}

fn unsupported(what: impl Into<String>) -> Error {
    Error::UnsupportedConfig(what.into())
}

fn invalid(what: impl Into<String>) -> Error {
    Error::InvalidConfig(what.into())
}

fn require(cond: bool, err: impl FnOnce() -> Error) -> Result<()> {
    if cond { Ok(()) } else { Err(err()) }
}

impl QwenConfig {
    pub fn from_json_bytes(bytes: &[u8], what: &str) -> Result<Self> {
        let c: HfConfig = serde_json::from_slice(bytes).map_err(|e| Error::Json {
            what: what.to_string(),
            source: e,
        })?;
        Self::from_hf(c)
    }

    fn from_hf(c: HfConfig) -> Result<Self> {
        require(c.model_type == "qwen3_5", || {
            unsupported(format!("model_type {:?}", c.model_type))
        })?;
        require(
            c.architectures == ["Qwen3_5ForConditionalGeneration"],
            || unsupported(format!("architectures {:?}", c.architectures)),
        )?;
        let q = &c.quantization_config;
        require(q.quant_method == "fp8", || {
            unsupported(format!("quant_method {:?}", q.quant_method))
        })?;
        require(q.fmt == "e4m3", || unsupported(format!("fmt {:?}", q.fmt)))?;
        require(q.activation_scheme == "dynamic", || {
            unsupported(format!("activation_scheme {:?}", q.activation_scheme))
        })?;
        require(q.weight_block_size == [QUANT_BLOCK, QUANT_BLOCK], || {
            unsupported(format!("weight_block_size {:?}", q.weight_block_size))
        })?;

        let t = &c.text_config;
        require(t.model_type == "qwen3_5_text", || {
            unsupported(format!("text model_type {:?}", t.model_type))
        })?;
        require(t.dtype == "bfloat16", || {
            unsupported(format!("dtype {:?}", t.dtype))
        })?;
        require(!c.tie_word_embeddings && !t.tie_word_embeddings, || {
            unsupported("tied word embeddings")
        })?;
        require(t.hidden_act == "silu", || {
            unsupported(format!("hidden_act {:?}", t.hidden_act))
        })?;
        require(!t.attention_bias, || unsupported("attention_bias"))?;
        require(t.attention_dropout == 0.0, || {
            unsupported(format!("attention_dropout {}", t.attention_dropout))
        })?;
        require(t.attn_output_gate, || {
            unsupported("attention without its output gate")
        })?;
        // `swish` names the DeltaNet's output gate, a SiLU of `z`. The full
        // attention's gate is a sigmoid whatever this says.
        require(t.output_gate_type == "swish", || {
            unsupported(format!("output_gate_type {:?}", t.output_gate_type))
        })?;
        require(t.mamba_ssm_dtype == "float32", || {
            unsupported(format!("mamba_ssm_dtype {:?}", t.mamba_ssm_dtype))
        })?;
        require(!t.mtp_use_dedicated_embeddings, || {
            unsupported("MTP with its own embeddings")
        })?;

        let r = &t.rope_parameters;
        require(r.rope_type == "default", || {
            unsupported(format!("rope_type {:?}", r.rope_type))
        })?;
        require(r.mrope_interleaved, || {
            unsupported("non-interleaved M-RoPE")
        })?;
        require(r.partial_rotary_factor == t.partial_rotary_factor, || {
            invalid(format!(
                "partial_rotary_factor {} at the top level, {} in rope_parameters",
                t.partial_rotary_factor, r.partial_rotary_factor
            ))
        })?;
        positive_f32(r.rope_theta, "rope_theta")?;
        positive_f32(r.partial_rotary_factor, "partial_rotary_factor")?;
        require(r.partial_rotary_factor <= 1.0, || {
            invalid("partial_rotary_factor above 1")
        })?;
        // `int(head_dim * partial_rotary_factor)`, a truncation in double precision.
        let rotary_dim = (t.head_dim as f64 * r.partial_rotary_factor) as usize;
        require(rotary_dim > 0 && rotary_dim.is_multiple_of(2), || {
            invalid(format!("rotary width {rotary_dim}"))
        })?;
        // Text positions are equal in all three M-RoPE streams, so the sections
        // only have to tile the frequencies.
        require(
            r.mrope_section.len() == 3 && r.mrope_section.iter().sum::<usize>() * 2 == rotary_dim,
            || {
                invalid(format!(
                    "mrope_section {:?} does not tile {} frequencies",
                    r.mrope_section,
                    rotary_dim / 2
                ))
            },
        )?;

        require(t.num_hidden_layers == t.layer_types.len(), || {
            invalid(format!(
                "num_hidden_layers {} with {} layer_types",
                t.num_hidden_layers,
                t.layer_types.len()
            ))
        })?;
        let mut layers = Vec::with_capacity(t.layer_types.len());
        for (i, kind) in t.layer_types.iter().enumerate() {
            let kind = match kind.as_str() {
                "linear_attention" => LayerKind::DeltaNet,
                "full_attention" => LayerKind::Attention,
                other => return Err(unsupported(format!("layer type {other:?}"))),
            };
            let by_interval =
                t.full_attention_interval > 0 && (i + 1).is_multiple_of(t.full_attention_interval);
            require((kind == LayerKind::Attention) == by_interval, || {
                invalid(format!(
                    "layer {i} is {kind:?}, full_attention_interval {} says otherwise",
                    t.full_attention_interval
                ))
            })?;
            layers.push(LayerSpec {
                checkpoint_index: i,
                kind,
            });
        }

        require(t.hidden_size.is_multiple_of(QUANT_BLOCK), || {
            unsupported(format!("hidden_size {} is not whole blocks", t.hidden_size))
        })?;
        require(t.intermediate_size.is_multiple_of(QUANT_BLOCK), || {
            unsupported(format!("intermediate_size {}", t.intermediate_size))
        })?;
        require(t.vocab_size > 0, || invalid("vocab_size 0"))?;

        require(
            t.num_attention_heads > 0
                && t.num_key_value_heads > 0
                && t.num_attention_heads.is_multiple_of(t.num_key_value_heads),
            || {
                invalid(format!(
                    "{} query heads over {} KV heads",
                    t.num_attention_heads, t.num_key_value_heads
                ))
            },
        )?;
        require(
            t.head_dim.is_multiple_of(2) && rotary_dim <= t.head_dim,
            || invalid(format!("head_dim {}", t.head_dim)),
        )?;
        let attention = AttentionSpec {
            num_q_heads: t.num_attention_heads,
            num_kv_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            rotary_dim,
            rope_theta: r.rope_theta,
            softmax_scale: (t.head_dim as f64).powf(-0.5) as f32,
        };

        require(
            t.linear_num_key_heads > 0
                && t.linear_num_value_heads
                    .is_multiple_of(t.linear_num_key_heads),
            || {
                invalid(format!(
                    "{} value heads over {} key heads",
                    t.linear_num_value_heads, t.linear_num_key_heads
                ))
            },
        )?;
        require(t.linear_conv_kernel_dim >= 1, || invalid("conv kernel 0"))?;
        require(
            t.linear_key_head_dim > 0 && t.linear_value_head_dim > 0,
            || invalid("DeltaNet head dim 0"),
        )?;
        let deltanet = DeltaNetSpec {
            num_k_heads: t.linear_num_key_heads,
            num_v_heads: t.linear_num_value_heads,
            k_head_dim: t.linear_key_head_dim,
            v_head_dim: t.linear_value_head_dim,
            conv_kernel: t.linear_conv_kernel_dim,
        };
        for (what, n) in [
            ("DeltaNet query/key/value width", deltanet.conv_dim()),
            ("DeltaNet value width", deltanet.value_dim()),
            (
                "attention query width",
                2 * t.num_attention_heads * t.head_dim,
            ),
            ("attention KV width", t.num_key_value_heads * t.head_dim),
            ("attention output width", t.num_attention_heads * t.head_dim),
        ] {
            require(n.is_multiple_of(QUANT_BLOCK), || {
                unsupported(format!("{what} {n} is not whole blocks"))
            })?;
        }

        let mut vision_token_ids: Vec<u32> = [c.image_token_id, c.video_token_id]
            .into_iter()
            .flatten()
            .collect();
        vision_token_ids.sort_unstable();
        let has_vision = c.vision_config.is_some() && c.language_model_only != Some(true);
        let _ = c.language_model_only;

        Ok(QwenConfig {
            vocab_size: t.vocab_size,
            hidden_size: t.hidden_size,
            intermediate_size: t.intermediate_size,
            rms_norm_eps: positive_f32(t.rms_norm_eps, "rms_norm_eps")?,
            max_position_embeddings: t.max_position_embeddings,
            checkpoint_layers: layers.len(),
            layers,
            attention,
            deltanet,
            mtp_layers: t.mtp_num_hidden_layers,
            bos_token_id: t.bos_token_id,
            eos_token_id: t.eos_token_id,
            vision_token_ids,
            has_vision,
        })
    }

    /// Keep only the listed checkpoint layers, in the order given. Every layer
    /// is self-contained (its state is its own), so any non-empty subset is a
    /// model.
    pub fn truncated(mut self, keep: &[usize]) -> Result<Self> {
        if keep.is_empty() {
            return Err(Error::Input("a truncation keeps no layer".into()));
        }
        let mut layers = Vec::with_capacity(keep.len());
        for &i in keep {
            let spec = self
                .layers
                .iter()
                .find(|l| l.checkpoint_index == i)
                .ok_or_else(|| Error::Input(format!("layer {i} is not in this model")))?;
            if layers.iter().any(|l: &LayerSpec| l.checkpoint_index == i) {
                return Err(Error::Input(format!("layer {i} kept twice")));
            }
            layers.push(*spec);
        }
        self.layers = layers;
        Ok(self)
    }

    /// The model without its MTP layers.
    pub fn without_mtp(mut self) -> Self {
        self.mtp_layers = 0;
        self
    }

    /// Positions of the layers of one kind among [`Self::layers`].
    pub fn layers_of(&self, kind: LayerKind) -> impl Iterator<Item = usize> + '_ {
        self.layers
            .iter()
            .enumerate()
            .filter(move |(_, l)| l.kind == kind)
            .map(|(i, _)| i)
    }
}
