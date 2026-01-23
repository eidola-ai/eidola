//! Configuration for Llama models.

use serde::Deserialize;

/// Configuration for a Llama model, loaded from config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    /// Vocabulary size.
    pub vocab_size: usize,

    /// Hidden size (embedding dimension).
    pub hidden_size: usize,

    /// Intermediate size for feed-forward network.
    pub intermediate_size: usize,

    /// Number of hidden layers (transformer blocks).
    pub num_hidden_layers: usize,

    /// Number of attention heads.
    pub num_attention_heads: usize,

    /// Number of key-value heads (for grouped-query attention).
    /// If not specified, defaults to num_attention_heads (standard MHA).
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// RMS normalization epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    /// Maximum sequence length.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// Rope theta for positional encoding.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// Beginning of sequence token ID.
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: u32,

    /// End of sequence token ID.
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_max_position_embeddings() -> usize {
    2048
}

fn default_rope_theta() -> f64 {
    10000.0
}

fn default_bos_token_id() -> u32 {
    1
}

fn default_eos_token_id() -> u32 {
    2
}

impl LlamaConfig {
    /// Returns the number of key-value heads, defaulting to num_attention_heads if not set.
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Returns the head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Creates a config for TinyLlama-1.1B with default values.
    #[cfg(test)]
    pub fn tiny_llama_1b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 2048,
            intermediate_size: 5632,
            num_hidden_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: Some(4),
            rms_norm_eps: 1e-5,
            max_position_embeddings: 2048,
            rope_theta: 10000.0,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }
}
