//! Chat layer of the inference engine: tokenizer, prompt rendering, and
//! streaming reasoning / tool-call output parsing, for each model architecture
//! the engine serves ([`ChatFormat`]): MiMo-V2.6 (Flash and Pro) and
//! DeepSeek-V4.1.
//!
//! A MiMo request flows through it in this order:
//!
//! 1. [`ChatInput::from_json`] parses the request's `messages` and `tools`
//!    (order-preserving, Python value semantics), and
//!    [`ChatInput::normalize_tool_call_arguments`] decodes history tool-call
//!    arguments sent as JSON strings.
//! 2. [`ChatTemplate::render_prompt`] produces the prompt, its text
//!    byte-identical to Python `transformers.apply_chat_template` with the
//!    model's own template, and says which parts of it are request data.
//! 3. [`MimoTokenizer::encode_prompt`] turns it into token ids. Only the
//!    template's own text becomes an added token (`<|im_end|>`, `<think>`, …);
//!    request data that spells one is encoded as text.
//! 4. As the engine samples ids (stopping at [`MimoTokenizer::is_eos`] or a
//!    length limit), [`OutputParser`] turns them into OpenAI deltas —
//!    `reasoning_content`, `content`, `tool_calls` — and maps the
//!    [`FinishReason`].
//!
//! The model files (`chat_template.jinja`, `tokenizer.json`,
//! `generation_config.json`) are read from the model directory; the template
//! and tokenizer must match pinned SHA-256 hashes.
//!
//! A DeepSeek-V4.1 request takes the same steps through [`deepseek_v4_1`]:
//! [`deepseek_v4_1::encode`] in place of the template (the format is compiled
//! in, and takes tool-call arguments as the request sent them), a
//! [`Tokenizer`] loaded with [`Tokenizer::for_format`], and an
//! [`OutputParser`] configured with [`OutputConfig::deepseek_v4_1`].

pub mod args;
pub mod deepseek_v4_1;
pub mod error;
pub mod format;
pub mod json;
pub mod output;
pub mod pysem;
pub mod reasoning;
pub mod template;
pub mod tokenizer;
pub mod tool_call;

pub use args::{ArgumentTyping, ToolSchemas};
pub use error::ChatError;
pub use format::ChatFormat;
pub use output::{ChatDelta, FinishReason, OutputConfig, OutputParser, StopCause};
pub use template::{ChatInput, ChatTemplate, Prompt, RenderOptions};
pub use tokenizer::{Detokenizer, MimoTokenizer, Tokenizer};
pub use tool_call::{CallIdSource, PrefixedCallIds, ToolCall, UnparsedText};
