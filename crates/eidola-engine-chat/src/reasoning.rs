//! What a format's reasoning parser yields: [`crate::mimo::reasoning`] for
//! MiMo-V2.6's `<think>` blocks, [`crate::deepseek::output`] for DeepSeek-V4.1's.

/// Reasoning and content produced by one step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReasoningDelta {
    pub reasoning: String,
    pub content: String,
}
