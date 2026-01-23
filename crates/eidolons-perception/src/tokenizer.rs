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

/// Role for multi-turn chat formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    /// User message
    User,
    /// Assistant message
    Assistant,
}

/// A message in a multi-turn conversation for formatting.
#[derive(Debug, Clone)]
pub struct FormatMessage<'a> {
    /// The role of the message sender
    pub role: ChatRole,
    /// The message content
    pub content: &'a str,
}

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

    /// Formats a multi-turn conversation for TinyLlama.
    ///
    /// Uses the ChatML format:
    /// ```text
    /// <|system|>
    /// You are a helpful assistant.</s>
    /// <|user|>
    /// {first user message}</s>
    /// <|assistant|>
    /// {first assistant response}</s>
    /// <|user|>
    /// {second user message}</s>
    /// <|assistant|>
    /// ```
    ///
    /// The final `<|assistant|>` tag prompts the model to generate a response.
    pub fn format_multi_turn_prompt(messages: &[FormatMessage<'_>]) -> String {
        let mut formatted = String::new();

        // Start with system prompt
        formatted.push_str("<|system|>\nYou are a helpful assistant.</s>\n");

        // Add each message in the conversation
        for msg in messages {
            match msg.role {
                ChatRole::User => {
                    formatted.push_str("<|user|>\n");
                    formatted.push_str(msg.content);
                    formatted.push_str("</s>\n");
                }
                ChatRole::Assistant => {
                    formatted.push_str("<|assistant|>\n");
                    formatted.push_str(msg.content);
                    formatted.push_str("</s>\n");
                }
            }
        }

        // Add assistant prefix for model to continue
        formatted.push_str("<|assistant|>\n");
        formatted
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

    #[test]
    fn test_multi_turn_format_single_message() {
        let messages = vec![FormatMessage {
            role: ChatRole::User,
            content: "Hello!",
        }];
        let formatted = TinyLlamaTokenizer::format_multi_turn_prompt(&messages);

        assert!(formatted.contains("<|system|>"));
        assert!(formatted.contains("<|user|>\nHello!</s>"));
        assert!(formatted.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_multi_turn_matches_single_turn_format() {
        // Single user message through multi-turn should match single-turn exactly
        let single_turn = TinyLlamaTokenizer::format_chat_prompt("Hello!");

        let messages = vec![FormatMessage {
            role: ChatRole::User,
            content: "Hello!",
        }];
        let multi_turn = TinyLlamaTokenizer::format_multi_turn_prompt(&messages);

        println!("=== Single-turn format ===");
        println!("{:?}", single_turn);
        println!("=== Multi-turn format ===");
        println!("{:?}", multi_turn);

        assert_eq!(
            single_turn, multi_turn,
            "Single message multi-turn should match single-turn format exactly"
        );
    }

    #[test]
    fn test_multi_turn_format_conversation() {
        let messages = vec![
            FormatMessage {
                role: ChatRole::User,
                content: "What is 2+2?",
            },
            FormatMessage {
                role: ChatRole::Assistant,
                content: "2+2 equals 4.",
            },
            FormatMessage {
                role: ChatRole::User,
                content: "And 3+3?",
            },
        ];
        let formatted = TinyLlamaTokenizer::format_multi_turn_prompt(&messages);

        // Check structure
        assert!(formatted.contains("<|system|>\nYou are a helpful assistant.</s>"));
        assert!(formatted.contains("<|user|>\nWhat is 2+2?</s>"));
        assert!(formatted.contains("<|assistant|>\n2+2 equals 4.</s>"));
        assert!(formatted.contains("<|user|>\nAnd 3+3?</s>"));
        assert!(formatted.ends_with("<|assistant|>\n"));

        // Check order (user1 before assistant, assistant before user2)
        let user1_pos = formatted.find("What is 2+2?").unwrap();
        let asst_pos = formatted.find("2+2 equals 4.").unwrap();
        let user2_pos = formatted.find("And 3+3?").unwrap();
        assert!(user1_pos < asst_pos);
        assert!(asst_pos < user2_pos);
    }
}
