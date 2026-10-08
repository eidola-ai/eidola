//! The wire contract: the strict request the gateway forwards, and the response shapes.
//!
//! The request is the server's strict chat-completions subset (`eidola-server`'s
//! `types.rs`: the same fields, nullability and `deny_unknown_fields`), plus one field the
//! gateway adds, `cache_key`. Anything else is refused, so the engine's own parameter
//! surface is not reachable from outside.
//!
//! The strict types only *validate* `messages` and `tools`. The prompt is rendered from
//! the same body parsed again by the chat crate's order-preserving JSON parser, because
//! `serde_json::Value` sorts object keys and rounds large integers, and the template
//! prints both (`eidola-engine-chat/AGENTS.md`).

use base64::Engine as _;
use eidola_engine::secret::CacheKey;
use eidola_engine_chat::json::{self as chat_json, Json};
use serde::Deserialize;
use zeroize::Zeroize;

use crate::error::ApiError;

/// Most stop sequences a request may carry (OpenAI's limit).
pub const MAX_STOP_SEQUENCES: usize = 4;

/// A chat completion request, as the gateway forwards it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// The client's prefix-cache key: 32 bytes, base64url without padding. Content-class:
    /// never logged, and turned into an engine salt as soon as it is decoded.
    #[serde(default)]
    pub cache_key: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: Role,
    #[serde(default)]
    pub content: Option<MessageContent>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default)]
    pub detail: Option<String>,
}

/// A request that passed every check that needs no model: what the pipeline runs.
pub struct ValidRequest {
    /// `messages` and `tools`, key order and number spelling as sent.
    pub messages: Json,
    pub tools: Option<Json>,
    /// Whether tool calls are parsed out of the output (`tools` given and `tool_choice`
    /// not `"none"`).
    pub parse_tools: bool,
    pub max_completion_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: bool,
    pub include_usage: bool,
    pub stop: Vec<String>,
    pub cache_key: Option<CacheKey>,
}

impl std::fmt::Debug for ValidRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ValidRequest(<content>)")
    }
}

/// Parses and validates a request body for the model `model_id`. Refusals happen here,
/// before anything is rendered, tokenized or scheduled.
pub fn parse_request(body: &[u8], model_id: &str) -> Result<ValidRequest, ApiError> {
    let mut req: ChatCompletionRequest =
        serde_json::from_slice(body).map_err(ApiError::from_serde)?;
    // Decode (and scrub) the key first, so no early return leaves it in memory.
    let cache_key = req.cache_key.take().map(decode_cache_key).transpose()?;

    if req.model != model_id {
        return Err(ApiError::ModelNotFound);
    }
    if req.messages.is_empty() {
        return Err(ApiError::invalid("messages must not be empty"));
    }
    for m in &req.messages {
        if let Some(MessageContent::Parts(parts)) = &m.content
            && parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. }))
        {
            return Err(ApiError::invalid("this model accepts text only"));
        }
    }
    if req.max_completion_tokens == Some(0) {
        return Err(ApiError::invalid(
            "max_completion_tokens must be at least 1",
        ));
    }
    let stop = match req.stop.take() {
        None => Vec::new(),
        Some(StopSequence::Single(s)) => vec![s],
        Some(StopSequence::Multiple(v)) => v,
    };
    if stop.len() > MAX_STOP_SEQUENCES {
        return Err(ApiError::invalid("stop accepts at most 4 sequences"));
    }
    if stop.iter().any(String::is_empty) {
        return Err(ApiError::invalid("stop sequences must not be empty"));
    }
    let tools_given = req.tools.as_ref().is_some_and(|t| !t.is_empty());
    let parse_tools = match &req.tool_choice {
        None => tools_given,
        Some(serde_json::Value::String(s)) if s == "auto" => tools_given,
        Some(serde_json::Value::String(s)) if s == "none" => false,
        Some(_) => {
            return Err(ApiError::invalid(
                "tool_choice supports only \"auto\" and \"none\"",
            ));
        }
    };

    // The same body, parsed order-preserving for the template.
    let text = std::str::from_utf8(body).map_err(|_| ApiError::invalid("body is not UTF-8"))?;
    let mut doc =
        chat_json::parse(text).map_err(|_| ApiError::invalid("body is not valid JSON"))?;
    let mut take = |key: &str| -> Option<Json> {
        let Json::Object(fields) = &mut doc else {
            return None;
        };
        let i = fields.iter().position(|(k, _)| k == key)?;
        Some(fields.swap_remove(i).1)
    };
    let messages = take("messages").expect("validated above");
    let tools = take("tools").filter(|t| !matches!(t, Json::Null));
    if let Some(Json::Str(mut s)) = take("cache_key") {
        s.zeroize();
    }

    Ok(ValidRequest {
        messages,
        tools,
        parse_tools,
        max_completion_tokens: req.max_completion_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stream: req.stream,
        include_usage: req.stream_options.is_some_and(|o| o.include_usage),
        stop,
        cache_key,
    })
}

fn decode_cache_key(mut text: String) -> Result<CacheKey, ApiError> {
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(text.as_bytes());
    text.zeroize();
    let mut bytes = decoded.map_err(|_| ApiError::invalid(CACHE_KEY_SHAPE))?;
    let result = <[u8; 32]>::try_from(bytes.as_slice())
        .map(CacheKey::from_bytes)
        .map_err(|_| ApiError::invalid(CACHE_KEY_SHAPE));
    bytes.zeroize();
    // `from_bytes` copied the array it was given; that copy is gone with this frame and
    // the decoded buffer is scrubbed above.
    result
}

const CACHE_KEY_SHAPE: &str = "cache_key must be 32 bytes, base64url without padding";
