//! The Qwen3.8 chat format (Qwen3.8-27B, text only): the model's own Jinja
//! template ([`template`]), `<think>` reasoning ([`reasoning`], shared with
//! MiMo-V2.6) and `<tool_call>` markup with each tag on its own line
//! ([`tool_call`], the shared parser in its
//! [`Markup::Newlines`](crate::tool_call::xml::Markup) form).
//!
//! The tokenizer is the crate's [`Tokenizer`](crate::Tokenizer) loaded for
//! [`ChatFormat::Qwen3_8`](crate::ChatFormat) (`tokenizer.json` and
//! `generation_config.json`), and an [`OutputParser`](crate::OutputParser)
//! configured with [`OutputConfig::qwen3_8`](crate::OutputConfig::qwen3_8)
//! runs this module's parsers.

pub mod template;

pub use crate::reasoning;
pub use crate::tool_call::xml as tool_call;
pub use template::{ChatTemplate, RenderOptions};
