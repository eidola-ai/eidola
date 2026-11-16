//! The request's chat input and the prompt a chat format renders from it,
//! shared by every format.

use std::ops::Range;

use sha2::{Digest, Sha256};

use crate::error::ChatError;
use crate::json::{self, Json};

/// A rendered prompt: its text, and which parts of it the request supplied.
///
/// Everything the template writes from the request (a role, message content,
/// reasoning, a tool call's name, argument names and values, tool definitions)
/// is data; the rest is the template's own text, which alone may spell the
/// model's added tokens ([`crate::mimo::MimoTokenizer::encode_prompt`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt {
    text: String,
    data: Vec<Range<usize>>,
}

impl Prompt {
    /// A prompt from its text and data spans (ascending, disjoint, non-empty,
    /// on character boundaries).
    pub(crate) fn from_parts(text: String, data: Vec<Range<usize>>) -> Prompt {
        Prompt { text, data }
    }

    /// The prompt text, byte-identical to the reference rendering's.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The byte ranges of [`Self::text`] that are request data: ascending,
    /// disjoint, non-empty, on character boundaries.
    pub fn data_spans(&self) -> &[Range<usize>] {
        &self.data
    }

    pub fn into_text(self) -> String {
        self.text
    }

    /// Cuts the prompt to its first `len` bytes.
    pub(crate) fn truncate(&mut self, len: usize) {
        self.text.truncate(len);
        self.data.retain_mut(|span| {
            span.end = span.end.min(len);
            span.start < span.end
        });
    }
}

/// OpenAI-shaped chat input: the `messages` array and the optional `tools`
/// array of a chat-completions request.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatInput {
    pub messages: Json,
    pub tools: Option<Json>,
}

impl ChatInput {
    /// Parses the request's `messages` and `tools` JSON texts, preserving key
    /// order (which `tojson` reproduces in the prompt).
    pub fn from_json(messages: &str, tools: Option<&str>) -> Result<ChatInput, ChatError> {
        let messages =
            json::parse(messages).map_err(|e| ChatError::InvalidInput(format!("messages: {e}")))?;
        let tools = match tools {
            None => None,
            Some(text) => Some(
                json::parse(text).map_err(|e| ChatError::InvalidInput(format!("tools: {e}")))?,
            ),
        };
        Ok(ChatInput { messages, tools })
    }
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}
