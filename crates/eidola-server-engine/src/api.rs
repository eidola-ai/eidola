//! The wire contract: the strict request the gateway forwards, and the response shapes.
//!
//! The request is the gateway's strict chat-completions subset (`eidola-server-gateway`'s
//! `types.rs`: the same fields, nullability and `deny_unknown_fields`), plus one field the
//! gateway adds, `cache_key`. Every request asks for a stream with
//! `stream_options.include_usage`, the only answer this node gives. Anything else is
//! refused, so the engine's own parameter
//! surface is not reachable from outside. A continuation (`continue_final_message`, with
//! `add_generation_prompt: false`) is part of the subset: its prompt ends inside the
//! final assistant message, at the field it continues.
//!
//! The strict types only *validate* `messages` and `tools`. The prompt is rendered from
//! the same body parsed again by the chat crate's order-preserving JSON parser, because
//! `serde_json::Value` sorts object keys and rounds large integers, and the template
//! prints both (`eidola-engine-chat/AGENTS.md`).

use base64::Engine as _;
use eidola_common::engine_protocol::{
    CACHE_KEY_BYTES, CACHE_KEY_TEXT_LEN, CachedPrefix, ContinueField, FinalContent, FinalMessage,
    SubsetError, ToolChoice, check_cached_prefix, check_continuation, check_max_completion_tokens,
    check_stop, check_tool_choice,
};
use eidola_engine::secret::CacheKey;
use eidola_engine_chat::json::{self as chat_json, Json};
use serde::Deserialize;
use zeroize::Zeroize;

use crate::error::ApiError;

/// Most stop sequences a request may carry, and the longest one in UTF-8 bytes. Stop
/// sequences are delimiters (`"\n\n"`, `"Observation:"`, an end marker); Eidola's own
/// chat path sends none, and its local proxy relays only what a local caller sets. The
/// byte cap is far above any delimiter while bounding the matcher
/// (`pipeline::StopMatcher`, a pattern copy plus a `usize` table per byte, plus the
/// held-back text) to a few kilobytes per request, so a body cannot turn its 32 MiB into
/// hundreds of MiB of matcher state. Both live in `eidola_common::engine_protocol` with
/// the rest of the subset's rules, which the gateway applies before routing.
pub use eidola_common::engine_protocol::{MAX_STOP_BYTES, MAX_STOP_SEQUENCES};

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
    /// Whether the prompt ends with a new assistant turn; with a continuation it may only
    /// be `false` (`eidola_common::engine_protocol::check_continuation`).
    #[serde(default)]
    pub add_generation_prompt: Option<bool>,
    /// Continue the final assistant message rather than answer it: `true` (its content)
    /// or the name of the field to continue. Read by `check_continuation`.
    #[serde(default)]
    pub continue_final_message: Option<serde_json::Value>,
    /// The client's prefix-cache key: 32 bytes, base64url without padding. Content-class:
    /// never logged, and turned into the engine's lineage as soon as it is decoded. Its text is
    /// scrubbed whenever it is dropped, including when a later field fails to parse and
    /// serde drops the partly built request.
    #[serde(default)]
    pub cache_key: Option<CacheKeyText>,
    /// The declared cached prefix, `{"cached_tokens", "new_tokens"}` exactly
    /// (`eidola_common::engine_protocol::CachedPrefix`): the request is served only if
    /// the prefix cache holds that many of its prompt's leading tokens under its key and
    /// the prompt has at most that many more.
    #[serde(default)]
    pub cached_prefix: Option<serde_json::Value>,
}

/// The cache key's JSON text, zeroed when dropped. Deserialized straight from a string, so
/// the text is never held by an unscrubbed owner.
pub struct CacheKeyText(String);

impl CacheKeyText {
    fn as_str(&self) -> &str {
        &self.0
    }
}

impl Drop for CacheKeyText {
    fn drop(&mut self) {
        self.0.zeroize();
        #[cfg(test)]
        tests::record_scrub(&self.0);
    }
}

impl std::fmt::Debug for CacheKeyText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CacheKeyText(<redacted>)")
    }
}

impl<'de> Deserialize<'de> for CacheKeyText {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d).map(CacheKeyText)
    }
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
    /// An assistant message's reasoning, which the template renders before its content.
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

impl Message {
    /// This message as the continuation rule reads a final message.
    fn as_final(&self) -> FinalMessage<'_> {
        FinalMessage {
            is_assistant: self.role == Role::Assistant,
            content: match &self.content {
                None => FinalContent::Absent,
                Some(MessageContent::Text(text)) => FinalContent::Text(text),
                Some(MessageContent::Parts(_)) => FinalContent::Parts,
            },
            has_tool_calls: self.tool_calls.as_ref().is_some_and(|c| !c.is_empty()),
            reasoning_content: self.reasoning_content.as_deref(),
        }
    }
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

/// A content part. Every variant denies unknown fields: the template renders the parts
/// from the raw body, and it treats an `image_url`, `image`, `audio` or `video` key on a
/// part as multimodal content, so a key this type does not name must never get through.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
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
    pub stop: Vec<String>,
    pub cache_key: Option<CacheKey>,
    /// The final message's field the prompt continues, or `None` for a new assistant
    /// turn.
    pub continuation: Option<ContinueField>,
    /// The declared cached prefix, checked against the tokenized prompt and then by the
    /// engine at admission.
    pub cached_prefix: Option<CachedPrefix>,
}

impl std::fmt::Debug for ValidRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ValidRequest(<content>)")
    }
}

/// Parses and validates a request body for the model `model_id`. Refusals happen here,
/// before anything is rendered, tokenized or scheduled.
pub fn parse_request(body: &[u8], model_id: &str) -> Result<ValidRequest, ApiError> {
    // The value count and depth first, with nothing allocated: both parses below
    // allocate per value, and the host-memory budget per request counts at most
    // `MAX_REQUEST_JSON_VALUES` of them.
    eidola_common::engine_protocol::check_request_json(body)
        .map_err(|e| ApiError::invalid(e.to_string()))?;
    let mut req: ChatCompletionRequest =
        serde_json::from_slice(body).map_err(ApiError::from_serde)?;
    // Decode (and scrub) the key first, so no early return leaves it in memory.
    let cache_key = req.cache_key.take().map(decode_cache_key).transpose()?;

    if req.model != model_id {
        return Err(ApiError::ModelNotFound);
    }
    // Every answer is a stream carrying its usage, which is what the gateway bills by
    // and what bounds how long it waits on this node; it asks for one whatever its
    // own client asked for, so a request that does not is not the gateway's.
    if !req.stream || !req.stream_options.as_ref().is_some_and(|o| o.include_usage) {
        return Err(ApiError::invalid(STREAM_REQUIRED));
    }
    // The subset's rules, by the functions the gateway refuses with before routing.
    let invalid = |e: SubsetError| ApiError::invalid(e.to_string());
    if req.messages.is_empty() {
        return Err(invalid(SubsetError::EmptyMessages));
    }
    for m in &req.messages {
        if let Some(MessageContent::Parts(parts)) = &m.content
            && parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. }))
        {
            return Err(invalid(SubsetError::ImageContent));
        }
    }
    check_max_completion_tokens(req.max_completion_tokens).map_err(invalid)?;
    let stop = match req.stop.take() {
        None => Vec::new(),
        Some(StopSequence::Single(s)) => vec![s],
        Some(StopSequence::Multiple(v)) => v,
    };
    check_stop(&stop).map_err(invalid)?;
    let tools_given = req.tools.as_ref().is_some_and(|t| !t.is_empty());
    let tool_choice = check_tool_choice(req.tool_choice.as_ref()).map_err(invalid)?;
    // "none" is honoured by not showing the model any tools (below), so there is
    // nothing to parse either.
    let parse_tools = tool_choice == ToolChoice::Auto && tools_given;
    let continuation = check_continuation(
        req.continue_final_message.as_ref(),
        req.add_generation_prompt,
        req.messages.last().map(Message::as_final),
    )
    .map_err(invalid)?;
    let cached_prefix = req
        .cached_prefix
        .as_ref()
        .map(|v| CachedPrefix::from_value(v).ok_or_else(|| ApiError::invalid(CACHED_PREFIX_SHAPE)))
        .transpose()?;
    check_cached_prefix(cached_prefix, cache_key.is_some()).map_err(invalid)?;

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
    // `tool_choice: "none"` renders the prompt without the tool definitions, so the model
    // is never offered a tool it may not call (this changes the prompt, and with it the
    // reusable cache prefix, relative to the same request under "auto").
    let tools = if tool_choice == ToolChoice::None {
        None
    } else {
        tools
    };
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
        stop,
        cache_key,
        continuation,
        cached_prefix,
    })
}

/// Decodes the key into a fixed stack buffer (no heap copy, no by-value array), hands it
/// to [`CacheKey::from_buffer`], which scrubs it, and scrubs the text (dropping it) and the
/// buffer's slack. Copies outside this crate's reach remain: the request body's bytes in the HTTP
/// stack's shared read buffers, and serde_json's scratch buffer when the key's JSON string
/// uses escapes.
fn decode_cache_key(text: CacheKeyText) -> Result<CacheKey, ApiError> {
    // Room for any 43-character input; a longer one is refused before decoding.
    let mut buf = [0u8; 48];
    let decoded = if text.as_str().len() == CACHE_KEY_TEXT_LEN {
        base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode_slice(text.as_str().as_bytes(), &mut buf)
    } else {
        Ok(0)
    };
    drop(text);
    let result = match decoded {
        Ok(CACHE_KEY_BYTES) => {
            let key: &mut [u8; 32] = (&mut buf[..32]).try_into().expect("32 bytes");
            Ok(CacheKey::from_buffer(key))
        }
        _ => Err(ApiError::invalid(CACHE_KEY_SHAPE)),
    };
    buf.zeroize();
    result
}

const STREAM_REQUIRED: &str =
    "this node answers only a stream: stream must be true, with stream_options.include_usage";

const CACHE_KEY_SHAPE: &str = "cache_key must be 32 bytes, base64url without padding";

const CACHED_PREFIX_SHAPE: &str =
    "cached_prefix must be {\"cached_tokens\": <u32>, \"new_tokens\": <u32>} and nothing else";

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    thread_local! {
        /// Per dropped `CacheKeyText` on this thread: its capacity, and whether every byte
        /// of it read zero after the scrub.
        static SCRUBS: RefCell<Vec<(usize, bool)>> = const { RefCell::new(Vec::new()) };
    }

    /// Called by `CacheKeyText::drop` after zeroing, before the buffer is freed.
    pub(super) fn record_scrub(s: &String) {
        let cap = s.capacity();
        // SAFETY: the allocation is live (the `String` has not been dropped yet), `cap`
        // bytes long, and every byte of it was just written by `zeroize`, which zeroes
        // the whole capacity.
        let bytes = unsafe { std::slice::from_raw_parts(s.as_ptr(), cap) };
        let zero = bytes.iter().all(|b| *b == 0);
        SCRUBS.with(|v| v.borrow_mut().push((cap, zero)));
    }

    fn take_scrubs() -> Vec<(usize, bool)> {
        SCRUBS.with(|v| std::mem::take(&mut *v.borrow_mut()))
    }

    const KEY: &str = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";

    /// A field after `cache_key` failing to parse drops the partly built request: the key
    /// text is scrubbed then too, and the request is refused as invalid.
    #[test]
    fn a_failure_after_the_cache_key_scrubs_it() {
        take_scrubs();
        for body in [
            format!(r#"{{"cache_key":"{KEY}","model":5}}"#),
            format!(r#"{{"cache_key":"{KEY}","unknown":1}}"#),
            format!(r#"{{"cache_key":"{KEY}","messages":[{{"role":"nobody"}}]}}"#),
            format!(r#"{{"cache_key":"{KEY}","cache_key":"{KEY}"}}"#),
        ] {
            let e = parse_request(body.as_bytes(), "m").unwrap_err();
            assert!(matches!(e, ApiError::InvalidRequest(_)), "{body}: {e}");
            let scrubs = take_scrubs();
            assert!(!scrubs.is_empty(), "{body}: the key text was not dropped");
            assert!(
                scrubs.iter().all(|&(cap, zero)| cap >= KEY.len() && zero),
                "{body}: {scrubs:?}"
            );
        }
    }

    /// A body that is not JSON is refused by the shape scan before anything is parsed,
    /// so no key text is ever allocated.
    #[test]
    fn a_malformed_body_never_holds_the_key() {
        take_scrubs();
        let body = format!(r#"{{"cache_key":"{KEY}","#);
        let e = parse_request(body.as_bytes(), "m").unwrap_err();
        assert!(matches!(e, ApiError::InvalidRequest(_)), "{e}");
        assert!(take_scrubs().is_empty());
    }

    /// The accepted path scrubs it too.
    #[test]
    fn a_decoded_cache_key_is_scrubbed() {
        take_scrubs();
        let body = format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],"stream":true,"stream_options":{{"include_usage":true}},"cache_key":"{KEY}"}}"#
        );
        let req = parse_request(body.as_bytes(), "m").unwrap();
        assert!(req.cache_key.is_some());
        let scrubs = take_scrubs();
        assert_eq!(scrubs.len(), 1, "{scrubs:?}");
        assert!(scrubs[0].1);
    }

    /// The decoder accepts exactly the texts the shared shape rule
    /// (`eidola_common::engine_protocol::is_cache_key_text`) accepts, which is
    /// what the gateway checks before a request is paid for: every final
    /// character after a valid prefix, and lengths, alphabets and padding
    /// around it.
    #[test]
    fn the_decoder_agrees_with_the_shared_shape_rule() {
        let prefix = &KEY[..42];
        let mut texts: Vec<String> = (0u8..=255)
            .filter_map(|b| char::from_u32(b.into()))
            .map(|c| format!("{prefix}{c}"))
            .collect();
        texts.extend([
            String::new(),
            prefix.to_string(),
            format!("{KEY}A"),
            format!("{prefix}="),
            format!("+{}", &KEY[1..]),
            format!("/{}", &KEY[1..]),
            "_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJK".to_string(),
            "_-0123456789abcdefghijklmnopqrstuvwxyzABCDE".to_string(),
            "_-0123456789abcdefghijklmnopqrstuvwxyzABCDF".to_string(),
            "_-0123456789abcdefghijklmnopqrstuvwxyzABCDA".to_string(),
        ]);
        let mut accepted = 0;
        for text in texts {
            let shared = eidola_common::engine_protocol::is_cache_key_text(&text);
            let decoded = decode_cache_key(CacheKeyText(text.clone())).is_ok();
            assert_eq!(decoded, shared, "{text:?}");
            accepted += usize::from(decoded);
        }
        // Sixteen canonical final characters after the zero prefix, and the
        // two mixed-alphabet keys ending in `E` and `A`.
        assert_eq!(accepted, 18);
    }
}
