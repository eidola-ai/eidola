//! Configuration for Qwen3 models.

use serde::Deserialize;
use serde::de::Deserializer;

/// Configuration for a Qwen3 model, loaded from config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
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

    /// Head dimension (explicit in Qwen3, unlike Llama where it's derived).
    #[serde(default)]
    pub head_dim: Option<usize>,

    /// RMS normalization epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    /// Maximum sequence length.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// Rope theta for positional encoding.
    /// Qwen3 uses 1,000,000 for extended context (vs TinyLlama's 10,000).
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// Whether to tie word embeddings with lm_head.
    /// Qwen3 always uses false (separate embed_tokens and lm_head).
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    /// Beginning of sequence token ID.
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: u32,

    /// End of sequence token IDs.
    /// Qwen3 configs often have multiple EOS tokens: [151645, 151643]
    /// where 151645 is <|im_end|> and 151643 is <|endoftext|>.
    #[serde(
        default = "default_eos_token_ids",
        deserialize_with = "deserialize_eos_token_ids",
        rename = "eos_token_id"
    )]
    pub eos_token_ids: Vec<u32>,
}

/// Deserialize eos_token_id which can be either a single u32 or an array of u32.
fn deserialize_eos_token_ids<'de, D>(deserializer: D) -> Result<Vec<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum EosTokenId {
        Single(u32),
        Multiple(Vec<u32>),
    }

    match EosTokenId::deserialize(deserializer)? {
        EosTokenId::Single(id) => Ok(vec![id]),
        EosTokenId::Multiple(ids) => Ok(ids),
    }
}

fn default_rms_norm_eps() -> f64 {
    1e-6 // Qwen3 uses 1e-6 (vs Llama's 1e-5)
}

fn default_max_position_embeddings() -> usize {
    32768
}

fn default_rope_theta() -> f64 {
    1_000_000.0 // Qwen3 uses 1M (vs Llama's 10K)
}

fn default_tie_word_embeddings() -> bool {
    false // Qwen3 always uses separate lm_head
}

fn default_bos_token_id() -> u32 {
    151643 // Qwen3 specific
}

fn default_eos_token_ids() -> Vec<u32> {
    // Qwen3 uses both <|im_end|> (151645) and <|endoftext|> (151643)
    vec![151645, 151643]
}

impl Qwen3Config {
    /// Returns the number of key-value heads, defaulting to num_attention_heads if not set.
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Returns the head dimension.
    /// Qwen3 can have explicit head_dim in config, otherwise derive from hidden_size.
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Returns the primary EOS token ID (first in the list).
    /// This is typically <|im_end|> (151645) for Qwen3 chat models.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_ids.first().copied().unwrap_or(151645)
    }

    /// Returns all EOS token IDs.
    /// Qwen3 models typically stop on both <|im_end|> and <|endoftext|>.
    pub fn all_eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Creates a config for Qwen3-8B with default values.
    #[cfg(test)]
    pub fn qwen3_8b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 4096,
            intermediate_size: 12288,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            head_dim: Some(128),
            rms_norm_eps: 1e-6,
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: false,
            bos_token_id: 151643,
            eos_token_ids: vec![151645, 151643],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_config_defaults() {
        let config = Qwen3Config::qwen3_8b();
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_kv_heads(), 8);
        assert_eq!(config.head_dim(), 128);
        assert!(!config.tie_word_embeddings);
    }

    #[test]
    fn test_qwen3_config_deserialization_single_eos() {
        // Test with single eos_token_id (older format)
        let config_json = r#"{
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 151936,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": false,
            "bos_token_id": 151643,
            "eos_token_id": 151645
        }"#;

        let config: Qwen3Config = serde_json::from_str(config_json).unwrap();

        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.intermediate_size, 12288);
        assert_eq!(config.num_hidden_layers, 36);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads(), 8);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.rope_theta, 1_000_000.0);
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.eos_token_id(), 151645);
        assert_eq!(config.all_eos_token_ids(), &[151645]);
    }

    #[test]
    fn test_qwen3_config_deserialization_array_eos() {
        // Test with array eos_token_id (Qwen3 format)
        let config_json = r#"{
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 151936,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": false,
            "bos_token_id": 151643,
            "eos_token_id": [151645, 151643]
        }"#;

        let config: Qwen3Config = serde_json::from_str(config_json).unwrap();

        assert_eq!(config.eos_token_id(), 151645);
        assert_eq!(config.all_eos_token_ids(), &[151645, 151643]);
    }
}
