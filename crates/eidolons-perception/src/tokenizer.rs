//! Tokenizer wrapper for TinyLlama models.
//!
//! Provides a simple interface for encoding text to tokens and decoding tokens back to text,
//! using the HuggingFace tokenizers library.

use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// Special tokens for TinyLlama chat format.
pub const BOS_TOKEN: &str = "<s>";
pub const EOS_TOKEN: &str = "</s>";
pub const BOS_TOKEN_ID: u32 = 1;
pub const EOS_TOKEN_ID: u32 = 2;

/// Wrapper around the HuggingFace tokenizer for TinyLlama models.
#[derive(Debug)]
pub struct TinyLlamaTokenizer {
    tokenizer: Tokenizer,
}

impl TinyLlamaTokenizer {
    /// Loads a tokenizer from a tokenizer.json file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the tokenizer.json file
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer file cannot be read or parsed.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self { tokenizer })
    }

    /// Encodes text into token IDs.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to encode
    /// * `add_bos` - Whether to prepend the BOS token
    ///
    /// # Returns
    ///
    /// Vector of token IDs.
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))?;

        let mut ids: Vec<u32> = encoding.get_ids().to_vec();

        if add_bos {
            ids.insert(0, BOS_TOKEN_ID);
        }

        Ok(ids)
    }

    /// Decodes token IDs back into text.
    ///
    /// # Arguments
    ///
    /// * `ids` - The token IDs to decode
    /// * `skip_special_tokens` - Whether to skip special tokens in output
    ///
    /// # Returns
    ///
    /// The decoded text string.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Returns the BOS token ID.
    pub fn bos_token_id(&self) -> u32 {
        BOS_TOKEN_ID
    }

    /// Returns the EOS token ID.
    pub fn eos_token_id(&self) -> u32 {
        EOS_TOKEN_ID
    }

    /// Formats a prompt for the TinyLlama chat format.
    ///
    /// TinyLlama uses the following chat template:
    /// ```text
    /// <|system|>
    /// You are a helpful assistant.</s>
    /// <|user|>
    /// {user_message}</s>
    /// <|assistant|>
    /// ```
    pub fn format_chat_prompt(user_message: &str) -> String {
        format!(
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
            user_message
        )
    }
}

/// Loads a tokenizer from the HuggingFace hub cache.
///
/// # Arguments
///
/// * `repo` - The HuggingFace API repo handle
///
/// # Returns
///
/// A loaded TinyLlamaTokenizer.
pub async fn load_tokenizer(repo: &hf_hub::api::tokio::ApiRepo) -> Result<TinyLlamaTokenizer> {
    let tokenizer_path = repo
        .get("tokenizer.json")
        .await
        .context("Failed to download tokenizer.json")?;

    TinyLlamaTokenizer::from_file(&tokenizer_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_format() {
        let formatted = TinyLlamaTokenizer::format_chat_prompt("Hello!");
        assert!(formatted.contains("<|system|>"));
        assert!(formatted.contains("<|user|>"));
        assert!(formatted.contains("Hello!"));
        assert!(formatted.contains("<|assistant|>"));
    }
}
