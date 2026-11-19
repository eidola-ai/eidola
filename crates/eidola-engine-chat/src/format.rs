//! The chat formats this crate implements, one per model architecture.

use crate::tokenizer::{
    CONFIG_FILE, GENERATION_CONFIG_FILE, PINNED_DEEPSEEK_V4_1_TOKENIZER_SHA256,
    PINNED_QWEN3_8_TOKENIZER_SHA256, PINNED_TOKENIZER_SHA256,
};

/// A model architecture's prompt and output format. The caller picks it from
/// the model it serves; everything format-specific is reached through it or
/// through the format's own module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChatFormat {
    /// MiMo-V2.6 (Flash and Pro): the model's Jinja template
    /// ([`crate::mimo::ChatTemplate`]) and `<tool_call>` markup.
    MimoV2_6,
    /// DeepSeek-V4.1: the prompt encoder and DSML tool calls of
    /// [`crate::deepseek`].
    DeepSeekV4_1,
    /// Qwen3.8 (27B, text only): the model's Jinja template
    /// ([`crate::qwen::ChatTemplate`]) and `<tool_call>` markup with each tag
    /// on its own line.
    Qwen3_8,
}

impl ChatFormat {
    /// The file in the model directory that names the end-of-sequence ids
    /// (`eos_token_id`).
    pub fn eos_config_file(self) -> &'static str {
        match self {
            ChatFormat::MimoV2_6 | ChatFormat::Qwen3_8 => GENERATION_CONFIG_FILE,
            ChatFormat::DeepSeekV4_1 => CONFIG_FILE,
        }
    }

    /// The `tokenizer.json` hashes accepted for this format.
    pub fn pinned_tokenizer_sha256(self) -> &'static [&'static str] {
        match self {
            ChatFormat::MimoV2_6 => PINNED_TOKENIZER_SHA256,
            ChatFormat::DeepSeekV4_1 => PINNED_DEEPSEEK_V4_1_TOKENIZER_SHA256,
            ChatFormat::Qwen3_8 => PINNED_QWEN3_8_TOKENIZER_SHA256,
        }
    }
}
