//! DeepSeek-V4.1 configuration → a validated, typed model description.
//!
//! Two JSON shapes describe the same model and both are read: the checkpoint's
//! `config.json` (Hugging Face names, the text model nested under
//! `text_config`) and the flat file the released inference code takes
//! (`inference/config.json`). Every key of either is read; an unknown key is
//! refused. The per-layer table comes from the config's own lists
//! (`compress_ratios`, the KV-source and index-source layers, the candidate
//! source, the Engram layers), never from a hard-coded pattern.

use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use super::engram::{EngramLayerSpec, engram_layout};
use crate::error::{Error, Result};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct HfConfig {
    #[serde(default)]
    architectures: Vec<String>,
    model_type: String,
    dtype: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    transformers_version: Option<String>,
    #[allow(dead_code)]
    bos_token_id: Option<u32>,
    #[allow(dead_code)]
    eos_token_id: Option<u32>,
    #[allow(dead_code)]
    pad_token_id: Option<u32>,
    image_token_id: Option<u32>,
    quantization_config: HfQuant,
    text_config: HfText,
    vision_config: Option<HfVision>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct HfQuant {
    quant_method: String,
    activation_scheme: String,
    weight_block_size: Vec<usize>,
    scale_fmt: String,
    expert_dtype: String,
}

/// The vision tower's description. Read so that no key goes unseen; the text
/// forward uses none of it.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct HfVision {
    model_type: Option<String>,
    num_hidden_layers: usize,
    hidden_size: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    patch_size: usize,
    rope_theta: f64,
    downsample_ratio: usize,
    max_image_tokens: usize,
    min_pixels: usize,
    max_wh_ratio: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct HfRopeScaling {
    rope_type: String,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
    original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct HfText {
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    moe_intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    qk_rope_head_dim: usize,
    q_lora_rank: usize,
    o_lora_rank: usize,
    o_groups: usize,
    hidden_act: String,
    swiglu_limit: f64,
    rms_norm_eps: f64,
    attention_bias: bool,
    attention_dropout: f64,
    #[allow(dead_code)]
    initializer_range: Option<f64>,
    #[allow(dead_code)]
    use_cache: Option<bool>,
    tie_word_embeddings: bool,
    max_position_embeddings: usize,
    rope_theta: f64,
    rope_scaling: HfRopeScaling,
    n_routed_experts: usize,
    n_shared_experts: usize,
    num_experts_per_tok: usize,
    scoring_func: String,
    topk_method: String,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
    sliding_window: usize,
    compress_ratios: Vec<usize>,
    compress_rope_theta: f64,
    kv_source_layer_ids: Vec<usize>,
    index_source_layer_ids: Vec<usize>,
    index_n_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    candidate_source_layer_id: i64,
    candidate_topk_blocks: usize,
    candidate_block_size: usize,
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f64,
    engram_layer_ids: Vec<usize>,
    engram_num_embeddings: Vec<u64>,
    engram_max_ngram_size: usize,
    engram_vocab_size: u64,
    engram_n_heads: usize,
    engram_head_dim: usize,
    engram_pad_token_id: u32,
    engram_compressed_vocab_size: usize,
    num_nextn_predict_layers: usize,
    dspark_block_size: usize,
    dspark_noise_token_id: u32,
    dspark_target_layer_ids: Vec<usize>,
    dspark_markov_rank: usize,
    dspark_n_routed_experts: usize,
    dspark_num_experts_per_tok: usize,
}

fn one() -> f64 {
    1.0
}
fn yes() -> bool {
    true
}
fn minus_one() -> i64 {
    -1
}

/// The released inference code's flat argument file (`ModelArgs` field names).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct InferenceConfig {
    #[allow(dead_code)]
    max_batch_size: Option<usize>,
    max_seq_len: Option<usize>,
    #[allow(dead_code)]
    temperature: Option<f64>,
    dtype: String,
    expert_dtype: Option<String>,
    vocab_size: usize,
    dim: usize,
    moe_inter_dim: usize,
    n_layers: usize,
    n_mtp_layers: usize,
    n_heads: usize,
    n_routed_experts: usize,
    n_shared_experts: usize,
    n_activated_experts: usize,
    score_func: String,
    #[serde(default = "one")]
    gate_temp: f64,
    #[serde(default = "yes")]
    norm_topk_prob: bool,
    route_scale: f64,
    swiglu_limit: f64,
    q_lora_rank: usize,
    head_dim: usize,
    rope_head_dim: usize,
    norm_eps: f64,
    o_groups: usize,
    o_lora_rank: usize,
    window_size: usize,
    compress_ratios: Vec<usize>,
    kv_source_layers: Vec<usize>,
    index_source_layers: Vec<usize>,
    compress_rope_theta: f64,
    original_seq_len: usize,
    rope_theta: f64,
    rope_factor: f64,
    beta_fast: f64,
    beta_slow: f64,
    index_n_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    #[serde(default = "minus_one")]
    candidate_source_layer: i64,
    #[serde(default)]
    candidate_topk_blocks: usize,
    #[serde(default)]
    candidate_block_size: usize,
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f64,
    #[serde(default)]
    engram_layer_ids: Vec<usize>,
    #[serde(default)]
    engram_num_embeddings: Vec<u64>,
    #[serde(default)]
    engram_max_ngram_size: usize,
    #[serde(default)]
    engram_vocab_size: u64,
    #[serde(default)]
    engram_n_heads: usize,
    #[serde(default)]
    engram_head_dim: usize,
    #[serde(default)]
    engram_pad_id: u32,
    #[serde(default)]
    engram_compressed_vocab_size: usize,
    #[serde(default)]
    vision_n_layers: usize,
    #[allow(dead_code)]
    vision_dim: Option<usize>,
    #[allow(dead_code)]
    vision_n_heads: Option<usize>,
    #[allow(dead_code)]
    vision_inter_dim: Option<usize>,
    #[allow(dead_code)]
    vision_patch_size: Option<usize>,
    #[allow(dead_code)]
    vision_rope_theta: Option<f64>,
    #[allow(dead_code)]
    vision_downsample_ratio: Option<usize>,
    #[allow(dead_code)]
    vision_max_n_token: Option<usize>,
    #[allow(dead_code)]
    vision_min_pixels: Option<usize>,
    #[allow(dead_code)]
    vision_max_wh_ratio: Option<f64>,
    image_token_id: Option<u32>,
    #[serde(default)]
    dspark_block_size: usize,
    #[serde(default)]
    dspark_noise_token_id: u32,
    #[serde(default)]
    dspark_target_layer_ids: Vec<usize>,
    #[serde(default)]
    dspark_markov_rank: usize,
    #[serde(default)]
    dspark_n_routed_experts: usize,
    #[serde(default)]
    dspark_n_activated_experts: usize,
}

/// Both shapes reduced to one set of fields.
#[derive(Debug, Clone, PartialEq)]
struct Fields {
    vocab_size: usize,
    dim: usize,
    moe_inter_dim: usize,
    n_layers: usize,
    n_mtp_layers: usize,
    n_heads: usize,
    n_routed_experts: usize,
    n_activated_experts: usize,
    norm_topk_prob: bool,
    route_scale: f64,
    swiglu_limit: f64,
    q_lora_rank: usize,
    head_dim: usize,
    rope_head_dim: usize,
    norm_eps: f64,
    o_groups: usize,
    o_lora_rank: usize,
    window_size: usize,
    compress_ratios: Vec<usize>,
    kv_source_layers: Vec<usize>,
    index_source_layers: Vec<usize>,
    compress_rope_theta: f64,
    original_seq_len: usize,
    rope_theta: f64,
    rope_factor: f64,
    beta_fast: f64,
    beta_slow: f64,
    index_n_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    candidate_source_layer: i64,
    candidate_topk_blocks: usize,
    candidate_block_size: usize,
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f64,
    engram_layer_ids: Vec<usize>,
    engram_num_embeddings: Vec<u64>,
    engram_max_ngram_size: usize,
    engram_vocab_size: u64,
    engram_n_heads: usize,
    engram_head_dim: usize,
    engram_pad_id: u32,
    engram_compressed_vocab_size: usize,
    dspark_block_size: usize,
    dspark_noise_token_id: u32,
    dspark_target_layer_ids: Vec<usize>,
    dspark_markov_rank: usize,
    dspark_n_routed_experts: usize,
    dspark_n_activated_experts: usize,
    image_token_id: Option<u32>,
    has_vision: bool,
    max_positions: Option<usize>,
}

/// Which RoPE table a layer rotates with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeKind {
    /// `rope_theta`, no YaRN: layers with compress ratio 0, and the drafter.
    Plain,
    /// `compress_rope_theta` with YaRN: every rotation of a layer whose
    /// compress ratio is above 0.
    Compressed,
}

/// One backbone layer's place in the sharing structure.
#[derive(Debug, Clone, PartialEq)]
pub struct DsLayerSpec {
    /// Position in this (possibly truncated) model.
    pub index: usize,
    /// Index in the checkpoint's tensor names (`layers.{source_index}`).
    pub source_index: usize,
    /// 0: sliding window only. `r > 0`: the window plus selected entries of a
    /// global KV in which one entry stands for `r` consecutive tokens.
    pub compress_ratio: usize,
    pub rope: RopeKind,
    /// This layer computes a global KV and its index keys from its own input.
    pub is_kv_source: bool,
    /// This layer runs an indexer.
    pub is_index_source: bool,
    /// The layer (index in this table) whose global KV this layer reads.
    pub kv_source: Option<usize>,
    /// The layer (index in this table) whose top-k selection this layer uses;
    /// itself when it is an index source.
    pub index_source: Option<usize>,
    /// This layer's indexer also publishes the candidate blocks.
    pub is_candidate_source: bool,
    /// This layer's indexer selects only inside the candidate blocks.
    pub uses_candidates: bool,
    /// Index into [`EngramSpec::layers`] when an Engram module runs before
    /// this block.
    pub engram: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DsAttentionSpec {
    pub n_heads: usize,
    pub head_dim: usize,
    /// The last `rope_dim` elements of a head (and of an index head) rotate.
    pub rope_dim: usize,
    pub q_lora_rank: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    /// A query sees itself and the `window - 1` positions before it.
    pub window: usize,
    /// `head_dim^-0.5`.
    pub softmax_scale: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexerSpec {
    pub n_heads: usize,
    pub head_dim: usize,
    pub top_k: usize,
    /// `head_dim^-0.5 · n_heads^-0.5` as a BF16 value, multiplied into the
    /// per-head score weights.
    pub weight_scale: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateSpec {
    pub top_blocks: usize,
    pub block_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DsMoeSpec {
    /// Routed experts in use (the first `n_routed` of the checkpoint's).
    pub n_routed: usize,
    /// Routed experts the checkpoint stores per layer.
    pub source_n_routed: usize,
    pub n_active: usize,
    pub inter_dim: usize,
    pub norm_topk_prob: bool,
    pub route_scale: f32,
    /// Clamp of the SwiGLU branches (both sides of the up branch, the gate
    /// branch from above); 0 disables it.
    pub swiglu_limit: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RopeSpec {
    pub theta: f64,
    pub compress_theta: f64,
    pub original_seq_len: usize,
    pub factor: f32,
    pub beta_fast: f64,
    pub beta_slow: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngramSpec {
    /// Every Engram layer the checkpoint has, in layer order. The hash layout
    /// of one depends on those before it, so truncation never drops entries
    /// here; a kept layer refers to its entry by index.
    pub layers: Vec<EngramLayerSpec>,
    pub max_ngram: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub pad_token: u32,
    pub compressed_vocab: usize,
}

impl EngramSpec {
    /// Table rows looked up per token per layer.
    pub fn lookups(&self) -> usize {
        (self.max_ngram - 1) * self.n_heads
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DrafterSpec {
    /// Draft blocks (`mtp.{i}`).
    pub blocks: usize,
    /// Draft slots computed per call.
    pub slots: usize,
    pub noise_token: u32,
    /// The backbone layers (indices in this table) whose entering streams
    /// feed the drafter.
    pub target_layers: Vec<usize>,
    pub markov_rank: usize,
    pub moe: DsMoeSpec,
}

/// A validated DeepSeek-V4.1 text model.
#[derive(Debug, Clone, PartialEq)]
pub struct DsConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub norm_eps: f32,
    pub attention: DsAttentionSpec,
    pub rope: RopeSpec,
    pub indexer: IndexerSpec,
    pub candidates: Option<CandidateSpec>,
    pub moe: DsMoeSpec,
    pub layers: Vec<DsLayerSpec>,
    pub engram: Option<EngramSpec>,
    /// `None` when the config has no drafter or a truncation dropped one of
    /// its target layers.
    pub drafter: Option<DrafterSpec>,
    /// The id of an image position. The text forward refuses it.
    pub image_token_id: Option<u32>,
    /// The config describes a vision tower (never loaded here).
    pub has_vision: bool,
    pub max_positions: Option<usize>,
    /// Layer count of the full checkpoint.
    pub source_num_layers: usize,
    fields: Fields,
}

fn positive_f32(v: f64, name: &str) -> Result<f32> {
    let f = v as f32;
    if !f.is_finite() || f <= 0.0 {
        return Err(Error::InvalidConfig(format!(
            "{name} = {v} is not a positive finite f32"
        )));
    }
    Ok(f)
}

fn inv_sqrt(n: usize) -> f64 {
    1.0 / (n as f64).sqrt()
}

impl DsConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        Self::from_json_bytes(&bytes, &path.display().to_string())
    }

    /// Either shape; the nested `text_config` marks the checkpoint's.
    pub fn from_json_bytes(bytes: &[u8], what: &str) -> Result<Self> {
        let json = |e| Error::Json {
            what: what.to_string(),
            source: e,
        };
        let value: Value = serde_json::from_slice(bytes).map_err(json)?;
        let fields = if value.get("text_config").is_some() {
            fields_from_hf(serde_json::from_value(value).map_err(json)?)?
        } else {
            fields_from_inference(serde_json::from_value(value).map_err(json)?)?
        };
        let keep: Vec<usize> = (0..fields.n_layers).collect();
        Self::build(fields, &keep)
    }

    /// Keep a subset of checkpoint layers (ascending source indices). The
    /// subset must still be a model: every kept layer with a compress ratio
    /// needs its KV source and an index source among the kept layers.
    pub fn truncated(&self, keep: &[usize]) -> Result<Self> {
        let mut fields = self.fields.clone();
        fields.n_routed_experts = self.moe.n_routed;
        if let Some(d) = &self.drafter {
            fields.dspark_n_routed_experts = d.moe.n_routed;
        }
        let mut out = Self::build(fields, keep)?;
        out.moe.source_n_routed = self.moe.source_n_routed;
        if let (Some(d), Some(s)) = (&mut out.drafter, &self.drafter) {
            d.moe.source_n_routed = s.moe.source_n_routed;
        }
        Ok(out)
    }

    /// Use only the first `backbone` routed experts of each backbone layer and
    /// the first `drafter` of each draft block: a smaller model made of the
    /// checkpoint's own tensors, for tests that cannot hold every expert.
    pub fn with_routed_experts(&self, backbone: usize, drafter: usize) -> Result<Self> {
        let mut out = self.clone();
        let cut = |moe: &mut DsMoeSpec, n: usize| {
            if n < moe.n_active || n > moe.source_n_routed {
                return Err(Error::Input(format!(
                    "{n} routed experts (the checkpoint has {}, {} are active per token)",
                    moe.source_n_routed, moe.n_active
                )));
            }
            moe.n_routed = n;
            Ok(())
        };
        cut(&mut out.moe, backbone)?;
        if let Some(d) = &mut out.drafter {
            cut(&mut d.moe, drafter)?;
        }
        Ok(out)
    }

    fn build(f: Fields, keep: &[usize]) -> Result<Self> {
        let invalid = |m: String| Err(Error::InvalidConfig(m));
        if keep.is_empty()
            || keep.windows(2).any(|w| w[0] >= w[1])
            || *keep.last().unwrap() >= f.n_layers
        {
            return Err(Error::Input(format!(
                "kept layers {keep:?} must be ascending indices below {}",
                f.n_layers
            )));
        }
        if f.compress_ratios.len() != f.n_layers + f.n_mtp_layers {
            return invalid(format!(
                "compress_ratios has {} entries for {} layers and {} draft blocks",
                f.compress_ratios.len(),
                f.n_layers,
                f.n_mtp_layers
            ));
        }
        if f.compress_ratios[f.n_layers..].iter().any(|r| *r != 0) {
            return Err(Error::UnsupportedConfig(
                "a draft block with a compress ratio".into(),
            ));
        }
        for (name, v) in [
            ("hidden size", f.dim),
            ("q_lora_rank", f.q_lora_rank),
            ("head_dim", f.head_dim),
            ("index_head_dim", f.index_head_dim),
            ("moe intermediate size", f.moe_inter_dim),
            ("o_groups · o_lora_rank", f.o_groups * f.o_lora_rank),
        ] {
            if v == 0 || !v.is_multiple_of(32) {
                return invalid(format!(
                    "{name} = {v} is not a positive multiple of the 32-wide quantisation block"
                ));
            }
        }
        if f.o_groups == 0 || !f.n_heads.is_multiple_of(f.o_groups) {
            return invalid(format!(
                "{} heads do not split into {} output groups",
                f.n_heads, f.o_groups
            ));
        }
        if f.rope_head_dim == 0
            || !f.rope_head_dim.is_multiple_of(2)
            || f.rope_head_dim > f.head_dim
            || f.rope_head_dim > f.index_head_dim
        {
            return invalid(format!("rope head dim {}", f.rope_head_dim));
        }
        if f.hc_mult == 0 || f.hc_sinkhorn_iters == 0 || f.window_size == 0 || f.index_topk == 0 {
            return invalid(
                "hc_mult, hc_sinkhorn_iters, the window and index_topk must be positive".into(),
            );
        }
        if f.n_activated_experts == 0 || f.n_activated_experts > f.n_routed_experts {
            return invalid(format!(
                "{} of {} experts active",
                f.n_activated_experts, f.n_routed_experts
            ));
        }
        if f.swiglu_limit < 0.0 || !(f.swiglu_limit as f32).is_finite() {
            return invalid(format!("swiglu_limit {}", f.swiglu_limit));
        }
        let listed = |list: &[usize], what: &str| -> Result<()> {
            for &l in list {
                if l >= f.n_layers || f.compress_ratios[l] == 0 {
                    return Err(Error::InvalidConfig(format!(
                        "{what} layer {l} has no compress ratio"
                    )));
                }
            }
            Ok(())
        };
        listed(&f.kv_source_layers, "KV-source")?;
        listed(&f.index_source_layers, "index-source")?;

        let candidate_source = usize::try_from(f.candidate_source_layer).ok();
        let candidates = match candidate_source {
            None => None,
            Some(c) => {
                if !f.index_source_layers.contains(&c) || !f.kv_source_layers.contains(&c) {
                    return invalid(format!(
                        "candidate source layer {c} is not a KV and index source"
                    ));
                }
                if f.candidate_block_size == 0 || f.candidate_topk_blocks == 0 {
                    return invalid("candidate block size and count must be positive".into());
                }
                // The newest block may hold a single entry. Fewer candidates
                // than the top-k would leave the selection short, which the
                // reference fills with entries outside the pool.
                if (f.candidate_topk_blocks - 1) * f.candidate_block_size + 1 < f.index_topk {
                    return Err(Error::UnsupportedConfig(format!(
                        "{} candidate blocks of {} can hold fewer than index_topk = {} entries",
                        f.candidate_topk_blocks, f.candidate_block_size, f.index_topk
                    )));
                }
                Some(CandidateSpec {
                    top_blocks: f.candidate_topk_blocks,
                    block_size: f.candidate_block_size,
                })
            }
        };

        let engram = if f.engram_layer_ids.is_empty() {
            None
        } else {
            if f.engram_max_ngram_size < 2 || f.engram_n_heads == 0 {
                return invalid("Engram needs n-grams of at least 2 tokens and a head".into());
            }
            let cols = (f.engram_max_ngram_size - 1) * f.engram_n_heads * f.engram_head_dim;
            if cols == 0 || !f.engram_head_dim.is_multiple_of(32) {
                return invalid(format!("Engram head dim {}", f.engram_head_dim));
            }
            if f.engram_layer_ids.iter().any(|l| *l >= f.n_layers)
                || f.engram_layer_ids.windows(2).any(|w| w[0] >= w[1])
            {
                return invalid(format!("Engram layers {:?}", f.engram_layer_ids));
            }
            if f.engram_compressed_vocab_size == 0 {
                return invalid("engram_compressed_vocab_size is 0".into());
            }
            let layers = engram_layout(
                &f.engram_layer_ids,
                f.engram_max_ngram_size,
                f.engram_n_heads,
                f.engram_vocab_size,
                f.engram_compressed_vocab_size,
            );
            if f.engram_num_embeddings.len() != layers.len() {
                return invalid(
                    "engram_num_embeddings does not list one table per Engram layer".into(),
                );
            }
            for (l, &rows) in layers.iter().zip(&f.engram_num_embeddings) {
                if l.num_rows != rows {
                    return invalid(format!(
                        "Engram layer {}: the hash buckets cover {} rows, the config declares {rows}",
                        l.source_layer, l.num_rows
                    ));
                }
            }
            Some(EngramSpec {
                layers,
                max_ngram: f.engram_max_ngram_size,
                n_heads: f.engram_n_heads,
                head_dim: f.engram_head_dim,
                pad_token: f.engram_pad_id,
                compressed_vocab: f.engram_compressed_vocab_size,
            })
        };

        let mut layers: Vec<DsLayerSpec> = Vec::with_capacity(keep.len());
        for (index, &src) in keep.iter().enumerate() {
            let ratio = f.compress_ratios[src];
            let is_kv_source = f.kv_source_layers.contains(&src);
            let is_index_source = f.index_source_layers.contains(&src);
            let (mut kv_source, mut index_source) = (None, None);
            if ratio > 0 {
                kv_source = if is_kv_source {
                    Some(index)
                } else {
                    layers.iter().rposition(|l| l.is_kv_source)
                };
                index_source = if is_index_source {
                    Some(index)
                } else {
                    layers.iter().rposition(|l| l.is_index_source)
                };
                let ok = match (kv_source, index_source) {
                    (Some(k), Some(i)) => {
                        let same_ratio = k == index || layers[k].compress_ratio == ratio;
                        let same_bank = i == index || layers[i].kv_source == Some(k);
                        same_ratio && same_bank && i >= k
                    }
                    _ => false,
                };
                if !ok {
                    return invalid(format!(
                        "layer {src} (compress ratio {ratio}) has no KV source and index source of its own ratio among the layers before it"
                    ));
                }
            }
            let is_candidate_source = candidate_source == Some(src);
            let uses_candidates = is_index_source && candidate_source.is_some_and(|c| c < src);
            if uses_candidates {
                let source = layers.iter().position(|l| l.is_candidate_source);
                if source.is_none() || kv_source != layers[source.unwrap()].kv_source {
                    return invalid(format!(
                        "layer {src} selects inside candidate blocks, but the candidate source is not a kept layer over the same global KV"
                    ));
                }
            }
            layers.push(DsLayerSpec {
                index,
                source_index: src,
                compress_ratio: ratio,
                rope: if ratio > 0 {
                    RopeKind::Compressed
                } else {
                    RopeKind::Plain
                },
                is_kv_source,
                is_index_source,
                kv_source,
                index_source,
                is_candidate_source,
                uses_candidates,
                engram: f.engram_layer_ids.iter().position(|l| *l == src),
            });
        }

        let moe = DsMoeSpec {
            n_routed: f.n_routed_experts,
            source_n_routed: f.n_routed_experts,
            n_active: f.n_activated_experts,
            inter_dim: f.moe_inter_dim,
            norm_topk_prob: f.norm_topk_prob,
            route_scale: positive_f32(f.route_scale, "route scale")?,
            swiglu_limit: f.swiglu_limit as f32,
        };
        let drafter = if f.dspark_block_size == 0 || f.n_mtp_layers == 0 {
            None
        } else {
            if f.dspark_target_layer_ids.is_empty()
                || f.dspark_target_layer_ids.iter().any(|l| *l >= f.n_layers)
            {
                return invalid(format!(
                    "drafter target layers {:?}",
                    f.dspark_target_layer_ids
                ));
            }
            if !(f.dim * f.dspark_target_layer_ids.len()).is_multiple_of(32)
                || f.dspark_markov_rank == 0
            {
                return invalid("drafter input width or Markov rank".into());
            }
            let n_routed = if f.dspark_n_routed_experts > 0 {
                f.dspark_n_routed_experts
            } else {
                f.n_routed_experts
            };
            let n_active = if f.dspark_n_activated_experts > 0 {
                f.dspark_n_activated_experts
            } else {
                f.n_activated_experts
            };
            if n_active > n_routed {
                return invalid(format!("{n_active} of {n_routed} drafter experts active"));
            }
            let targets: Option<Vec<usize>> = f
                .dspark_target_layer_ids
                .iter()
                .map(|t| keep.iter().position(|k| k == t))
                .collect();
            targets.map(|target_layers| DrafterSpec {
                blocks: f.n_mtp_layers,
                slots: f.dspark_block_size,
                noise_token: f.dspark_noise_token_id,
                target_layers,
                markov_rank: f.dspark_markov_rank,
                moe: DsMoeSpec {
                    n_routed,
                    source_n_routed: n_routed,
                    n_active,
                    ..moe.clone()
                },
            })
        };

        let weight_scale = super::numeric::bf16_round(
            (inv_sqrt(f.index_head_dim) * inv_sqrt(f.index_n_heads)) as f32,
        );
        Ok(DsConfig {
            vocab_size: f.vocab_size,
            dim: f.dim,
            hc_mult: f.hc_mult,
            hc_sinkhorn_iters: f.hc_sinkhorn_iters,
            hc_eps: positive_f32(f.hc_eps, "hc_eps")?,
            norm_eps: positive_f32(f.norm_eps, "norm eps")?,
            attention: DsAttentionSpec {
                n_heads: f.n_heads,
                head_dim: f.head_dim,
                rope_dim: f.rope_head_dim,
                q_lora_rank: f.q_lora_rank,
                o_groups: f.o_groups,
                o_lora_rank: f.o_lora_rank,
                window: f.window_size,
                softmax_scale: inv_sqrt(f.head_dim) as f32,
            },
            rope: RopeSpec {
                theta: positive_f32(f.rope_theta, "rope_theta").map(|_| f.rope_theta)?,
                compress_theta: positive_f32(f.compress_rope_theta, "compress_rope_theta")
                    .map(|_| f.compress_rope_theta)?,
                original_seq_len: f.original_seq_len,
                factor: positive_f32(f.rope_factor, "rope factor")?,
                beta_fast: positive_f32(f.beta_fast, "beta_fast").map(|_| f.beta_fast)?,
                beta_slow: positive_f32(f.beta_slow, "beta_slow").map(|_| f.beta_slow)?,
            },
            indexer: IndexerSpec {
                n_heads: f.index_n_heads,
                head_dim: f.index_head_dim,
                top_k: f.index_topk,
                weight_scale,
            },
            candidates,
            moe,
            layers,
            engram,
            drafter,
            image_token_id: f.image_token_id,
            has_vision: f.has_vision,
            max_positions: f.max_positions,
            source_num_layers: f.n_layers,
            fields: f,
        })
    }
}

fn unsupported<T>(m: String) -> Result<T> {
    Err(Error::UnsupportedConfig(m))
}

fn fields_from_hf(c: HfConfig) -> Result<Fields> {
    let t = c.text_config;
    if c.model_type != "deepseek_v41" || t.model_type != "deepseek_v41_text" {
        return unsupported(format!(
            "model_type {:?} / {:?}",
            c.model_type, t.model_type
        ));
    }
    if !c.architectures.is_empty()
        && !c
            .architectures
            .iter()
            .any(|a| a == "DeepseekV41ForCausalLM")
    {
        return unsupported(format!("architectures {:?}", c.architectures));
    }
    if c.dtype.as_deref().is_some_and(|d| d != "bfloat16") {
        return unsupported(format!("activation dtype {:?}", c.dtype));
    }
    let q = &c.quantization_config;
    if q.quant_method != "fp8"
        || q.activation_scheme != "dynamic"
        || q.weight_block_size != [32, 32]
        || q.scale_fmt != "ue8m0"
        || q.expert_dtype != "fp4"
    {
        return unsupported(format!("quantization_config {q:?}"));
    }
    if t.hidden_act != "silu"
        || t.attention_bias
        || t.attention_dropout != 0.0
        || t.tie_word_embeddings
    {
        return unsupported(
            "hidden_act, attention_bias, attention_dropout or tie_word_embeddings".into(),
        );
    }
    if t.num_key_value_heads != 1 {
        return unsupported(format!("num_key_value_heads {}", t.num_key_value_heads));
    }
    if t.rope_scaling.rope_type != "yarn" {
        return unsupported(format!("rope_type {:?}", t.rope_scaling.rope_type));
    }
    if t.topk_method != "noaux_tc" {
        return unsupported(format!("topk_method {:?}", t.topk_method));
    }
    check_common(&t.scoring_func, t.n_shared_experts)?;
    Ok(Fields {
        vocab_size: t.vocab_size,
        dim: t.hidden_size,
        moe_inter_dim: t.moe_intermediate_size,
        n_layers: t.num_hidden_layers,
        n_mtp_layers: t.num_nextn_predict_layers,
        n_heads: t.num_attention_heads,
        n_routed_experts: t.n_routed_experts,
        n_activated_experts: t.num_experts_per_tok,
        norm_topk_prob: t.norm_topk_prob,
        route_scale: t.routed_scaling_factor,
        swiglu_limit: t.swiglu_limit,
        q_lora_rank: t.q_lora_rank,
        head_dim: t.head_dim,
        rope_head_dim: t.qk_rope_head_dim,
        norm_eps: t.rms_norm_eps,
        o_groups: t.o_groups,
        o_lora_rank: t.o_lora_rank,
        window_size: t.sliding_window,
        compress_ratios: t.compress_ratios,
        kv_source_layers: t.kv_source_layer_ids,
        index_source_layers: t.index_source_layer_ids,
        compress_rope_theta: t.compress_rope_theta,
        original_seq_len: t.rope_scaling.original_max_position_embeddings,
        rope_theta: t.rope_theta,
        rope_factor: t.rope_scaling.factor,
        beta_fast: t.rope_scaling.beta_fast,
        beta_slow: t.rope_scaling.beta_slow,
        index_n_heads: t.index_n_heads,
        index_head_dim: t.index_head_dim,
        index_topk: t.index_topk,
        candidate_source_layer: t.candidate_source_layer_id,
        candidate_topk_blocks: t.candidate_topk_blocks,
        candidate_block_size: t.candidate_block_size,
        hc_mult: t.hc_mult,
        hc_sinkhorn_iters: t.hc_sinkhorn_iters,
        hc_eps: t.hc_eps,
        engram_layer_ids: t.engram_layer_ids,
        engram_num_embeddings: t.engram_num_embeddings,
        engram_max_ngram_size: t.engram_max_ngram_size,
        engram_vocab_size: t.engram_vocab_size,
        engram_n_heads: t.engram_n_heads,
        engram_head_dim: t.engram_head_dim,
        engram_pad_id: t.engram_pad_token_id,
        engram_compressed_vocab_size: t.engram_compressed_vocab_size,
        dspark_block_size: t.dspark_block_size,
        dspark_noise_token_id: t.dspark_noise_token_id,
        dspark_target_layer_ids: t.dspark_target_layer_ids,
        dspark_markov_rank: t.dspark_markov_rank,
        dspark_n_routed_experts: t.dspark_n_routed_experts,
        dspark_n_activated_experts: t.dspark_num_experts_per_tok,
        image_token_id: c.image_token_id,
        has_vision: c.vision_config.is_some(),
        max_positions: Some(t.max_position_embeddings),
    })
}

fn check_common(scoring: &str, shared: usize) -> Result<()> {
    if scoring != "sqrtsoftplus" {
        return unsupported(format!("router score function {scoring:?}"));
    }
    if shared != 1 {
        return unsupported(format!("{shared} shared experts"));
    }
    Ok(())
}

fn fields_from_inference(c: InferenceConfig) -> Result<Fields> {
    if c.dtype != "fp8" || c.expert_dtype.as_deref() != Some("fp4") {
        return unsupported(format!(
            "weight dtype {:?}, expert dtype {:?}",
            c.dtype, c.expert_dtype
        ));
    }
    if c.gate_temp != 1.0 {
        return unsupported(format!("gate_temp {}", c.gate_temp));
    }
    check_common(&c.score_func, c.n_shared_experts)?;
    Ok(Fields {
        vocab_size: c.vocab_size,
        dim: c.dim,
        moe_inter_dim: c.moe_inter_dim,
        n_layers: c.n_layers,
        n_mtp_layers: c.n_mtp_layers,
        n_heads: c.n_heads,
        n_routed_experts: c.n_routed_experts,
        n_activated_experts: c.n_activated_experts,
        norm_topk_prob: c.norm_topk_prob,
        route_scale: c.route_scale,
        swiglu_limit: c.swiglu_limit,
        q_lora_rank: c.q_lora_rank,
        head_dim: c.head_dim,
        rope_head_dim: c.rope_head_dim,
        norm_eps: c.norm_eps,
        o_groups: c.o_groups,
        o_lora_rank: c.o_lora_rank,
        window_size: c.window_size,
        compress_ratios: c.compress_ratios,
        kv_source_layers: c.kv_source_layers,
        index_source_layers: c.index_source_layers,
        compress_rope_theta: c.compress_rope_theta,
        original_seq_len: c.original_seq_len,
        rope_theta: c.rope_theta,
        rope_factor: c.rope_factor,
        beta_fast: c.beta_fast,
        beta_slow: c.beta_slow,
        index_n_heads: c.index_n_heads,
        index_head_dim: c.index_head_dim,
        index_topk: c.index_topk,
        candidate_source_layer: c.candidate_source_layer,
        candidate_topk_blocks: c.candidate_topk_blocks,
        candidate_block_size: c.candidate_block_size,
        hc_mult: c.hc_mult,
        hc_sinkhorn_iters: c.hc_sinkhorn_iters,
        hc_eps: c.hc_eps,
        engram_layer_ids: c.engram_layer_ids,
        engram_num_embeddings: c.engram_num_embeddings,
        engram_max_ngram_size: c.engram_max_ngram_size,
        engram_vocab_size: c.engram_vocab_size,
        engram_n_heads: c.engram_n_heads,
        engram_head_dim: c.engram_head_dim,
        engram_pad_id: c.engram_pad_id,
        engram_compressed_vocab_size: c.engram_compressed_vocab_size,
        dspark_block_size: c.dspark_block_size,
        dspark_noise_token_id: c.dspark_noise_token_id,
        dspark_target_layer_ids: c.dspark_target_layer_ids,
        dspark_markov_rank: c.dspark_markov_rank,
        dspark_n_routed_experts: c.dspark_n_routed_experts,
        dspark_n_activated_experts: c.dspark_n_activated_experts,
        image_token_id: c.image_token_id,
        has_vision: c.vision_n_layers > 0,
        max_positions: c.max_seq_len,
    })
}
