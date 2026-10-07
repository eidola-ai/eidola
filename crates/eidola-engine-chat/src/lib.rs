//! Chat layer for the MiMo-V2.6 model family (Flash and Pro): tokenizer,
//! chat template, and streaming reasoning / tool-call output parsing.
//!
//! A request flows through it in this order:
//!
//! 1. [`ChatInput::from_json`] parses the request's `messages` and `tools`
//!    (order-preserving, Python value semantics), and
//!    [`ChatInput::normalize_tool_call_arguments`] decodes history tool-call
//!    arguments sent as JSON strings.
//! 2. [`ChatTemplate::render`] produces the prompt, byte-identical to Python
//!    `transformers.apply_chat_template` with the model's own template.
//! 3. [`MimoTokenizer::encode`] turns it into token ids.
//! 4. As the engine samples ids (stopping at [`MimoTokenizer::is_eos`] or a
//!    length limit), [`OutputParser`] turns them into OpenAI deltas —
//!    `reasoning_content`, `content`, `tool_calls` — and maps the
//!    [`FinishReason`].
//!
//! The model files (`chat_template.jinja`, `tokenizer.json`,
//! `generation_config.json`) are read from the model directory; the template
//! and tokenizer must match pinned SHA-256 hashes.

pub mod args;
pub mod error;
pub mod json;
pub mod output;
pub mod pysem;
pub mod reasoning;
pub mod template;
pub mod tokenizer;
pub mod tool_call;

pub use args::{ArgumentTyping, ToolSchemas};
pub use error::ChatError;
pub use output::{ChatDelta, FinishReason, OutputConfig, OutputParser, StopCause};
pub use template::{ChatInput, ChatTemplate, RenderOptions};
pub use tokenizer::{Detokenizer, MimoTokenizer};
pub use tool_call::{CallIdSource, PrefixedCallIds, ToolCall};
