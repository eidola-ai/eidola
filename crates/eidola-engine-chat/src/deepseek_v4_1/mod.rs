//! The DeepSeek-V4.1 chat format (text only).
//!
//! The format is defined by the model repository's reference encoder,
//! `encoding/encoding.py` in `deepseek-ai/DeepSeek-V4.1-Flash-1017` (MIT), not
//! by a template file, so it is compiled in:
//!
//! 1. [`encode`] renders a [`ChatInput`](crate::ChatInput) to a
//!    [`Prompt`](crate::Prompt) whose text is byte-identical to the
//!    reference's `encode_messages`, with [`EncodeOptions`] carrying the
//!    thinking mode and the reasoning effort.
//! 2. [`Tokenizer::encode_prompt`](crate::Tokenizer::encode_prompt) encodes
//!    it; request data spells no added token.
//! 3. [`OutputConfig::deepseek_v4_1`](crate::OutputConfig::deepseek_v4_1)
//!    makes an [`OutputParser`](crate::OutputParser) split the generated text
//!    into reasoning, content and DSML tool calls ([`output`]).
//!
//! Images are not served: a request carrying an image part is refused with
//! [`ChatError::UnsupportedContent`](crate::ChatError::UnsupportedContent).

pub mod encoding;
pub mod output;

pub use encoding::{EncodeOptions, ReasoningEffort, encode, tool_schemas};

/// Starts every prompt.
pub const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
/// Ends an assistant turn; the model's end-of-sequence token.
pub const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
pub const SYSTEM_TOKEN: &str = "<｜System｜>";
pub const USER_TOKEN: &str = "<｜User｜>";
pub const ASSISTANT_TOKEN: &str = "<｜Assistant｜>";
pub const LATEST_REMINDER_TOKEN: &str = "<｜latest_reminder｜>";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";
/// The placeholder the reference writes for an image, which no text may spell.
pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

pub(crate) const CALLS_OPEN: &str = "<｜DSML｜ calls>";
pub(crate) const CALLS_CLOSE: &str = "</｜DSML｜ calls>";
pub(crate) const INVOKE_OPEN: &str = "<｜DSML｜ invoke";
pub(crate) const INVOKE_CLOSE: &str = "</｜DSML｜ invoke>";
pub(crate) const PARAMETER_OPEN: &str = "<｜DSML｜ parameter";
pub(crate) const PARAMETER_CLOSE: &str = "</｜DSML｜ parameter>";
/// What the reference writes between the answer and its tool calls.
pub(crate) const CALLS_SEPARATOR: &str = "\n\n";
