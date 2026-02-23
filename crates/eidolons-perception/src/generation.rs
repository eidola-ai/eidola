//! Configuration for text generation.

/// Configuration for text generation.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate.
    pub max_new_tokens: usize,
    /// Temperature for sampling (higher = more random).
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f32,
    /// Token ID that signals end of generation (primary).
    pub eos_token_id: u32,
    /// Additional EOS token IDs that should also stop generation.
    /// Useful for models like Qwen3 that have multiple stop tokens.
    pub additional_eos_ids: Vec<u32>,
    /// Whether to include the EOS token in output.
    pub include_eos: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 1024,
            temperature: 0.7,
            top_p: 0.9,
            eos_token_id: 2,
            additional_eos_ids: Vec::new(),
            include_eos: false,
        }
    }
}

impl GenerationConfig {
    /// Returns true if the given token is an EOS token.
    pub fn is_eos_token(&self, token_id: u32) -> bool {
        token_id == self.eos_token_id || self.additional_eos_ids.contains(&token_id)
    }
}
