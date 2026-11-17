//! Tool calls as every chat format's output parser produces them: the call, the
//! source of its ids, one streaming step, a complete output's result, and what
//! becomes of text that is not a call. The `<tool_call><function=…>` markup
//! MiMo-V2.6 and Qwen3.8 write is parsed by [`xml`]; DeepSeek-V4.1's DSML by
//! [`crate::deepseek::output`].

pub mod xml;

/// What becomes of output text that is not a call ([`xml`]'s
/// module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UnparsedText {
    /// Content, except whitespace-only text next to a call: the serving rule.
    #[default]
    Content,
    /// SGLang's: dropped, but for undeclared functions' blocks and the text
    /// before them. For differential tests against SGLang only.
    Sglang,
}

/// A parsed call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    /// Position among this response's calls (OpenAI's `index`).
    pub index: usize,
    /// OpenAI's `id`.
    pub id: String,
    pub name: String,
    /// The arguments object as JSON text (Python `json.dumps` layout).
    pub arguments: String,
}

impl ToolCall {
    /// The OpenAI `tool_calls[]` entry (also valid as a streaming delta).
    pub fn to_openai(&self) -> serde_json::Value {
        serde_json::json!({
            "index": self.index,
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        })
    }
}

/// Source of tool-call ids.
pub trait CallIdSource: Send {
    fn call_id(&mut self, index: usize) -> String;
}

/// Ids of the form `{prefix}{index}`. The caller supplies a prefix that is
/// unique per response (e.g. `call_` plus random characters), so ids never
/// repeat across a conversation.
#[derive(Clone, Debug)]
pub struct PrefixedCallIds(pub String);

impl CallIdSource for PrefixedCallIds {
    fn call_id(&mut self, index: usize) -> String {
        format!("{}{index}", self.0)
    }
}

impl<F: FnMut(usize) -> String + Send> CallIdSource for F {
    fn call_id(&mut self, index: usize) -> String {
        self(index)
    }
}

/// The complete-output result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsedToolCalls {
    pub content: String,
    /// `(name, arguments)` pairs in order.
    pub calls: Vec<(String, String)>,
}

/// One step of streaming output.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolCallDelta {
    pub content: String,
    pub calls: Vec<ToolCall>,
}

/// Length of the longest suffix of `text` that is a proper prefix of `tag`.
pub(crate) fn partial_suffix(text: &str, tag: &str) -> usize {
    let max = text.len().min(tag.len() - 1);
    (1..=max)
        .rev()
        .find(|&n| {
            text.is_char_boundary(text.len() - n) && tag.starts_with(&text[text.len() - n..])
        })
        .unwrap_or(0)
}
