//! Chat layer for the MiMo-V2.6 model family: tokenizer, chat template, and
//! streaming reasoning / tool-call output parsing.

pub mod error;
pub mod json;
pub mod template;

pub use error::ChatError;
pub use template::{ChatInput, ChatTemplate, RenderOptions};
