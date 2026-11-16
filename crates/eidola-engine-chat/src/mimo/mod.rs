//! The MiMo-V2.6 chat format (Flash and Pro): the model's own Jinja template
//! ([`template`]), its `<think>` reasoning ([`reasoning`]) and its
//! `<tool_call>` markup ([`tool_call`]).
//!
//! The tokenizer is the crate's [`Tokenizer`](crate::Tokenizer) loaded through
//! its MiMo constructors ([`MimoTokenizer`]), and an
//! [`OutputParser`](crate::OutputParser) configured with
//! [`OutputConfig::new`](crate::OutputConfig::new) runs this module's parsers.

pub mod reasoning;
pub mod template;
pub mod tool_call;

pub use crate::tokenizer::MimoTokenizer;
pub use template::{ChatTemplate, RenderOptions};
