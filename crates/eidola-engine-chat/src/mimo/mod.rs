//! The MiMo-V2.6 chat format (Flash and Pro): the model's own Jinja template
//! ([`template`]), its `<think>` reasoning ([`reasoning`], shared with
//! Qwen3.8) and its compact `<tool_call>` markup ([`tool_call`], the shared
//! parser in its [`Markup::Compact`](crate::tool_call::xml::Markup) form).
//!
//! The tokenizer is the crate's [`Tokenizer`](crate::Tokenizer) loaded through
//! its MiMo constructors ([`MimoTokenizer`]), and an
//! [`OutputParser`](crate::OutputParser) configured with
//! [`OutputConfig::new`](crate::OutputConfig::new) runs this module's parsers.

pub mod template;

pub use crate::reasoning;
pub use crate::tool_call::xml as tool_call;

pub use crate::tokenizer::MimoTokenizer;
pub use template::{ChatTemplate, RenderOptions};
