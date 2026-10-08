//! Core API types for chat completions.
//!
//! These types follow the de facto standard format used by most LLM gateways.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// A chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    /// ID of the model to use.
    pub model: String,

    /// A list of messages comprising the conversation.
    pub messages: Vec<Message>,

    /// The maximum number of completion tokens to generate.
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,

    /// Sampling temperature between 0 and 2.
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Nucleus sampling parameter.
    #[serde(default)]
    pub top_p: Option<f32>,

    /// Whether to stream partial responses.
    #[serde(default)]
    pub stream: bool,

    /// Streaming options. The OpenAI-compatible field; we accept it for
    /// API parity with clients that already set it (e.g. an SDK setting
    /// `include_usage: true` to capture token counts in the final chunk).
    /// Note: the server overrides `include_usage` to `true` for any
    /// streaming request before forwarding upstream — usage is required
    /// for accurate per-token refunds and isn't a client choice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Up to 4 sequences where the API will stop generating.
    #[serde(default)]
    pub stop: Option<StopSequence>,

    /// Tool (function) definitions the model may call.
    ///
    /// **Opaque pass-through.** Each entry is forwarded upstream verbatim:
    /// the server never executes a tool and has no reason to understand the
    /// JSON Schema inside `function.parameters`, while modelling it would
    /// mean rejecting (via this struct's `deny_unknown_fields`) every
    /// provider extension a client legitimately sends. Keeping the entries
    /// as raw `Value`s is also what lets the pricing contract measure
    /// exactly the bytes that go on the wire — see
    /// `handlers::chargeable_prompt_tokens_for`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,

    /// How the model should choose among `tools` (`"none"` / `"auto"` /
    /// `"required"`, or a `{"type":"function", …}` object). Opaque
    /// pass-through for the same reason as [`Self::tools`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,

    /// The client's prefix-cache key: 32 bytes, base64url without padding
    /// (`eidola_common::engine_protocol`). An Eidola-hosted engine scopes
    /// prefix-cache reuse to requests carrying the same key; without one,
    /// nothing a request computes is reused by any other.
    ///
    /// **Secret, and never serialized.** It is held decoded in a
    /// scrubbed-on-drop buffer, prints redacted, and `Serialize` skips it, so
    /// re-serializing a request for the Tinfoil upstream (which has no use
    /// for it) cannot carry it there. The one writer is
    /// `engine_trust::protocol::engine_request_body`.
    // `skip_serializing_if` with a predicate that always skips, rather than
    // `skip_serializing`, so the OpenAPI schema (which drops a
    // `skip_serializing` field) still documents the field clients may send.
    #[serde(default, skip_serializing_if = "never_serialized")]
    #[schema(value_type = Option<String>, min_length = 43, max_length = 43, pattern = "^[A-Za-z0-9_-]{43}$")]
    pub cache_key: Option<CacheKey>,
}

/// The `skip_serializing_if` predicate of a member that is never serialized.
fn never_serialized<T>(_: &T) -> bool {
    true
}

/// A client's decoded prefix-cache key.
///
/// Deserialized from its wire text, which must satisfy
/// `eidola_common::engine_protocol::is_cache_key_text`; the text is scrubbed
/// once decoded, and a refusal names the rule, never the value. Copies out of
/// reach here: the request body's bytes in the HTTP stack's buffers, and
/// `serde_json`'s scratch buffer when the JSON string uses escapes.
#[derive(Clone)]
pub struct CacheKey(Box<zeroize::Zeroizing<[u8; eidola_common::engine_protocol::CACHE_KEY_BYTES]>>);

impl CacheKey {
    /// The key's bytes.
    pub fn as_bytes(&self) -> &[u8; eidola_common::engine_protocol::CACHE_KEY_BYTES] {
        &self.0
    }
}

impl std::fmt::Debug for CacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CacheKey(<redacted>)")
    }
}

/// Refuses, always: a request's `cache_key` is skipped when the request
/// serializes, and this impl exists only because that skip is spelled as a
/// predicate. Nothing may serialize a key by accident.
impl Serialize for CacheKey {
    fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom("a cache key is never serialized"))
    }
}

impl<'de> Deserialize<'de> for CacheKey {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use base64::Engine as _;
        use eidola_common::engine_protocol::{CACHE_KEY_BYTES, is_cache_key_text};

        let text = zeroize::Zeroizing::new(String::deserialize(d)?);
        if !is_cache_key_text(&text) {
            return Err(serde::de::Error::custom(
                "cache_key must be 32 bytes, base64url without padding",
            ));
        }
        let mut key = Box::new(zeroize::Zeroizing::new([0u8; CACHE_KEY_BYTES]));
        // The decoder wants room for its length estimate; the slack is
        // scrubbed with the buffer.
        let mut buf = zeroize::Zeroizing::new([0u8; CACHE_KEY_BYTES + 3]);
        let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode_slice(text.as_bytes(), &mut buf[..]);
        if decoded != Ok(CACHE_KEY_BYTES) {
            return Err(serde::de::Error::custom(
                "cache_key must be 32 bytes, base64url without padding",
            ));
        }
        key.copy_from_slice(&buf[..CACHE_KEY_BYTES]);
        Ok(Self(key))
    }
}

/// Deserialize a real client body through the server's strict request type.
///
/// This test-only bridge lets app-core's HTTP harness compose its captured
/// body with the same `ChatCompletionRequest` deserialization production uses,
/// including strict nested `Message` fields.
#[cfg(feature = "test-support")]
#[doc(hidden)]
pub fn test_chat_completion_request_is_accepted(
    body: serde_json::Value,
) -> Result<(), serde_json::Error> {
    serde_json::from_value::<ChatCompletionRequest>(body).map(|_| ())
}

/// OpenAI-compatible streaming options.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    /// Include token-usage statistics in the final stream chunk. The
    /// server forces this on for upstream calls so it can compute
    /// accurate refunds; the field exists here only to round-trip
    /// honest clients.
    #[serde(default)]
    pub include_usage: bool,
}

/// Stop sequence can be a single string or array of strings.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

impl StopSequence {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopSequence::Single(s) => vec![s],
            StopSequence::Multiple(v) => v,
        }
    }
}

/// A message in the conversation.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Message {
    /// The role of the message author.
    pub role: Role,

    /// The content of the message.
    ///
    /// Nullable since tool calling: an assistant message that only called
    /// tools carries `"content": null`. The key is deliberately **always
    /// serialized** (no `skip_serializing_if`) — several chat templates
    /// require it to exist, and clients send the explicit `null` for that
    /// reason, so dropping it on the way upstream would change the request.
    #[serde(default)]
    pub content: Option<MessageContent>,

    /// An optional name for the participant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tool calls the assistant made, replayed by the client verbatim on the
    /// follow-up request.
    ///
    /// **Opaque pass-through**, like [`ChatCompletionRequest::tools`]: the
    /// client is required to replay the provider's own call objects
    /// unchanged (ids and any provider extension fields intact), so the
    /// server must not normalize them through a narrower struct.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,

    /// The id of the tool call this message answers (`role: "tool"` only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// UTF-8 byte length of the message's text content — 0 when the content
    /// is absent or `null` (an assistant message that only called tools).
    pub fn content_byte_len(&self) -> usize {
        self.content.as_ref().map(|c| c.byte_len()).unwrap_or(0)
    }
}

/// The role of a message author.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    /// The result of a tool call, keyed by `tool_call_id`.
    Tool,
}

/// Message content can be a simple string or array of content parts (for multimodal).
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract plain text from the content.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            MessageContent::Text(s) => Some(s),
            MessageContent::Parts(parts) => {
                // Return first text part if any
                parts.iter().find_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
            }
        }
    }

    /// Total byte length of all text content (for token estimation).
    pub fn byte_len(&self) -> usize {
        match self {
            MessageContent::Text(s) => s.len(),
            MessageContent::Parts(parts) => parts
                .iter()
                .map(|p| match p {
                    ContentPart::Text { text } => text.len(),
                    ContentPart::ImageUrl { .. } => 0,
                })
                .sum(),
        }
    }
}

/// A content part within a multimodal message.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text content.
    Text { text: String },

    /// Image content via URL.
    ImageUrl { image_url: ImageUrl },
}

/// An image URL reference.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageUrl {
    /// The URL of the image, or a base64-encoded data URI.
    pub url: String,

    /// Optional detail level for the image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionResponse {
    /// Unique identifier for the completion.
    pub id: String,

    /// The object type (always "chat.completion").
    pub object: String,

    /// Unix timestamp of when the completion was created.
    pub created: u64,

    /// The model used for completion.
    pub model: String,

    /// List of completion choices.
    pub choices: Vec<Choice>,

    /// Usage statistics for the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl ChatCompletionResponse {
    pub fn new(id: String, model: String, choices: Vec<Choice>, usage: Option<Usage>) -> Self {
        Self {
            id,
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model,
            choices,
            usage,
        }
    }
}

/// A completion choice.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Choice {
    /// The index of this choice.
    pub index: u32,

    /// The generated message.
    pub message: AssistantMessage,

    /// The reason the model stopped generating.
    pub finish_reason: Option<FinishReason>,
}

/// An assistant message in a response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct AssistantMessage {
    pub role: Role,
    pub content: Option<String>,

    /// Reasoning ("thinking") output from models that emit it. Two
    /// spellings are in the wild: OpenAI o-series and many compatible
    /// gateways use `reasoning_content`; vLLM uses `reasoning`. We
    /// faithfully round-trip whichever the upstream sent so clients can
    /// pick. Both stay `None` for non-thinking models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,

    /// Tool calls the model asked for. Relayed to the client **verbatim**
    /// (raw `Value`s) so the ids and any provider extension fields survive —
    /// the client replays these objects unchanged on its follow-up request,
    /// and a narrower struct here would silently drop what it doesn't model,
    /// exactly the defect the `reasoning*` fields above were added to fix.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

/// The reason the model stopped generating.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    /// The model stopped to call tools. Without this variant the whole
    /// completion (blocking) or chunk (SSE) fails to deserialize, so a
    /// tool-calling response never reaches the client at all.
    ToolCalls,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// A streaming chat completion chunk.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionChunk {
    /// Unique identifier for the completion.
    pub id: String,

    /// The object type (always "chat.completion.chunk").
    pub object: String,

    /// Unix timestamp of when the chunk was created.
    pub created: u64,

    /// The model used for completion.
    pub model: String,

    /// List of completion choices (deltas).
    pub choices: Vec<ChunkChoice>,

    /// Usage statistics (included in the final chunk by some providers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl ChatCompletionChunk {
    pub fn new(id: String, model: String, choices: Vec<ChunkChoice>) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model,
            choices,
            usage: None,
        }
    }
}

/// A choice delta in a streaming chunk.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChunkChoice {
    /// The index of this choice.
    pub index: u32,

    /// The delta (partial update) for this choice.
    pub delta: ChunkDelta,

    /// The reason the model stopped generating (only in final chunk).
    pub finish_reason: Option<FinishReason>,
}

/// A delta update in a streaming response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Reasoning ("thinking") deltas — same dual spelling as
    /// `AssistantMessage`. Without these fields here, serde silently
    /// drops the upstream `reasoning_content` / `reasoning` keys during
    /// deserialization and the client only ever sees `delta.content`.
    /// Round-trip whatever the upstream emits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,

    /// Streamed tool-call deltas, relayed **verbatim** as raw `Value`s.
    ///
    /// Nothing about a streamed tool call is guaranteed to arrive whole: the
    /// id, the function name and the `arguments` string all arrive in
    /// fragments keyed by the entry's `index`, and the client reassembles
    /// them. The proxy therefore must not model, reorder, or normalize these
    /// entries — including the streaming-only `index` framing key, which the
    /// client's accumulator needs. Same rule as `reasoning*` above: a field
    /// this struct does not name is dropped on re-serialization.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

/// A list of available models.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelsResponse {
    /// The list of models.
    pub data: Vec<Model>,
}

/// A model descriptor.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Model {
    /// The model identifier (e.g. "openai/gpt-4o").
    pub id: String,

    /// Human-readable display name.
    pub name: String,

    /// Short description of the model's capabilities.
    pub description: String,

    /// Maximum context window size in tokens.
    pub context_length: u64,

    /// The largest completion this model may be asked for.
    ///
    /// Separate from `context_length` on purpose: a context window is what a
    /// request may *contain*, not what one response may *be*. `null` means
    /// undeclared — a client must treat that as "unknown" and keep whatever
    /// default it already applies, never as zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,

    /// Which public output-budget ladder this model draws from.
    pub output_budget_class: OutputBudgetClass,

    /// Who runs the inference for this model.
    pub hosting: ModelHosting,

    /// What this model can do.
    pub capabilities: ModelCapabilities,

    /// Pricing in integer credits per 1k tokens.
    pub pricing: ModelPricing,
}

/// One capability leaf.
///
/// An object rather than a bare boolean, so a capability can grow a sibling
/// key later — an effort ladder, a weights measurement — without a type change
/// breaking a client already deployed against this shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct Capability {
    /// Whether the model supports it.
    pub supported: bool,
}

impl Capability {
    /// A leaf carrying `supported`.
    pub const fn new(supported: bool) -> Self {
        Self { supported }
    }
}

/// Who runs a model's inference, as the catalog declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelHosting {
    /// Tinfoil's inference service, reached through its router enclave.
    Tinfoil,
    /// Eidola's own inference engine, in deployments this gateway build pins
    /// by measurement and by weights hash.
    Eidola,
}

/// Prompt-cache reuse across requests.
///
/// `supported` means requests to this model may carry a `cache_key`, and that
/// requests carrying the same key can reuse each other's computed prompt
/// prefix. The two bounds are the serving deployments' measured retention:
/// a prefix nothing has used for `idle_ttl_secs`, or any prefix older than
/// `max_age_secs`, is never reused and is zeroed. Both are absent when
/// `supported` is false.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct PromptCacheCapability {
    /// Whether requests may carry a `cache_key`.
    pub supported: bool,
    /// Longest a cached prefix stays reusable without being used, in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_ttl_secs: Option<u64>,
    /// Longest a cached prefix stays reusable at all, in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_age_secs: Option<u64>,
}

impl PromptCacheCapability {
    /// No prompt-cache reuse.
    pub const fn unsupported() -> Self {
        Self {
            supported: false,
            idle_ttl_secs: None,
            max_age_secs: None,
        }
    }
}

/// The exact weights a model is served from, when this gateway pins them.
///
/// `supported` means this gateway build accepts only deployments serving the
/// weights hashed here, and tells each one so on every request. `sha256` is
/// the engine's weights hash (a SHA-256 over the manifest of every file the
/// engine reads from its weights directory); `repo` and `revision` say where
/// those files were taken from. All three are absent when `supported` is
/// false.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct PinnedWeightsCapability {
    /// Whether the weights are pinned by this gateway.
    pub supported: bool,
    /// The weights hash, lowercase hex.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// The source repository, `<owner>/<name>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// The source repository's revision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

impl PinnedWeightsCapability {
    /// Weights this gateway does not pin.
    pub const fn unsupported() -> Self {
        Self {
            supported: false,
            sha256: None,
            repo: None,
            revision: None,
        }
    }
}

/// A kind of content a model accepts or produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Text,
    Image,
    Audio,
}

/// Which public output-budget ladder a model draws from.
///
/// A *class*, not a number: the ladder's rungs are policy that can move
/// without the catalog moving, and keeping the selection public and
/// per-model is what keeps an output budget from becoming a per-user value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum OutputBudgetClass {
    /// A model that answers directly; the budget covers the answer alone.
    Standard,
    /// A model that thinks before it answers, so the same answer needs a
    /// materially larger budget — spend the whole of a standard one on
    /// reasoning and the response carries no answer at all.
    Reasoning,
}

/// What a model can do, as the catalog declares it.
///
/// Only axes that actually vary across the models this server sells are
/// carried. An axis every model shares tells a client nothing and is one more
/// assertion that has to stay true. `prompt_cache` and `pinned_weights` are
/// the axes on which an Eidola-hosted row differs from a Tinfoil-hosted one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct ModelCapabilities {
    /// Whether the model accepts a `tools` request field.
    ///
    /// Named after the wire rather than after an adjective: the question a
    /// client actually has is "will this request be accepted", and a field
    /// name answers it where "supports tool use" does not.
    pub tool_calling: Capability,

    /// Whether the model produces reasoning content before its answer.
    pub reasoning: Capability,

    /// The content kinds the model accepts.
    pub input_modalities: Vec<Modality>,

    /// The content kinds the model produces.
    pub output_modalities: Vec<Modality>,

    /// Whether, and for how long, requests may reuse each other's prompt
    /// prefix (`cache_key`).
    pub prompt_cache: PromptCacheCapability,

    /// The weights this gateway pins for the model, and where they came from.
    pub pinned_weights: PinnedWeightsCapability,
}

/// Pricing for a model in scaled integer credits per token (or per request).
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelPricing {
    pub per_prompt_token: ScaledPrice,
    pub per_completion_token: ScaledPrice,
    /// Per-request pricing (for models like Whisper or TTS that charge per request
    /// rather than per token). When present, `per_prompt_token` and
    /// `per_completion_token` are zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_request: Option<ScaledPrice>,
}

/// A price expressed as an integer value with a fixed scale factor.
///
/// Actual credits per unit = `value / scale_factor`.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ScaledPrice {
    pub value: u64,
    pub scale_factor: u64,
}

/// An error response in OpenAI format, optionally including a refund token.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    pub error: ErrorDetail,

    /// Refund token for unspent credits (present when an error occurs after
    /// the ACT nullifier has been recorded).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refund: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

impl ErrorResponse {
    pub fn new(message: impl Into<String>, error_type: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: error_type.into(),
                code: None,
            },
            refund: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_request() {
        let json = r#"{
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.model, "gpt-4o");
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].role, Role::User);
        assert!(matches!(
            &request.messages[0].content,
            Some(MessageContent::Text(t)) if t == "Hello!"
        ));
        assert!(!request.stream);
    }

    #[test]
    fn test_parse_request_with_all_options() {
        let json = r#"{
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"}
            ],
            "max_completion_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": true,
            "stop": ["END"]
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.max_completion_tokens, Some(100));
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.top_p, Some(0.9));
        assert!(request.stream);
        assert!(matches!(&request.stop, Some(StopSequence::Multiple(v)) if v == &["END"]));
    }

    #[test]
    fn test_parse_stop_single_string() {
        let json = r#"{
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": "STOP"
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        match request.stop.unwrap() {
            StopSequence::Single(s) => assert_eq!(s, "STOP"),
            _ => panic!("expected Single variant"),
        }
    }

    #[test]
    fn test_parse_stop_array() {
        let json = r#"{
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": ["END", "STOP", "DONE"]
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        match request.stop.unwrap() {
            StopSequence::Multiple(v) => {
                assert_eq!(v, vec!["END", "STOP", "DONE"]);
            }
            _ => panic!("expected Multiple variant"),
        }
    }

    #[test]
    fn test_stop_sequence_into_vec() {
        let single = StopSequence::Single("STOP".to_string());
        assert_eq!(single.into_vec(), vec!["STOP"]);

        let multiple = StopSequence::Multiple(vec!["A".to_string(), "B".to_string()]);
        assert_eq!(multiple.into_vec(), vec!["A", "B"]);
    }

    #[test]
    fn test_parse_multimodal_message() {
        let json = r#"{
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
                ]
            }]
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        match &request.messages[0].content {
            Some(MessageContent::Parts(parts)) => {
                assert_eq!(parts.len(), 2);
                assert!(
                    matches!(&parts[0], ContentPart::Text { text } if text == "What's in this image?")
                );
                assert!(matches!(
                    &parts[1],
                    ContentPart::ImageUrl { image_url } if image_url.url == "https://example.com/img.png"
                ));
            }
            _ => panic!("expected Parts variant"),
        }
    }

    #[test]
    fn test_parse_image_with_detail() {
        let json = r#"{
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png", "detail": "high"}}
                ]
            }]
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();

        match &request.messages[0].content {
            Some(MessageContent::Parts(parts)) => match &parts[0] {
                ContentPart::ImageUrl { image_url } => {
                    assert_eq!(image_url.detail, Some("high".to_string()));
                }
                _ => panic!("expected ImageUrl"),
            },
            _ => panic!("expected Parts"),
        }
    }

    #[test]
    fn test_message_content_as_text() {
        let text_content = MessageContent::Text("Hello".to_string());
        assert_eq!(text_content.as_text(), Some("Hello"));

        let parts_content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "First".to_string(),
            },
            ContentPart::Text {
                text: "Second".to_string(),
            },
        ]);
        assert_eq!(parts_content.as_text(), Some("First")); // Returns first text

        let image_only = MessageContent::Parts(vec![ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://example.com".to_string(),
                detail: None,
            },
        }]);
        assert_eq!(image_only.as_text(), None);
    }

    #[test]
    fn test_serialize_response() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-4o".to_string(),
            choices: vec![Choice {
                index: 0,
                message: AssistantMessage {
                    role: Role::Assistant,
                    content: Some("Hello!".to_string()),
                    reasoning_content: None,
                    reasoning: None,
                    tool_calls: None,
                },
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"id\":\"chatcmpl-123\""));
        assert!(json.contains("\"object\":\"chat.completion\""));
        assert!(json.contains("\"finish_reason\":\"stop\""));
    }

    #[test]
    fn test_serialize_chunk() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1234567890,
            model: "gpt-4o".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some(Role::Assistant),
                    content: None,
                    reasoning_content: None,
                    reasoning: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("\"object\":\"chat.completion.chunk\""));
        assert!(json.contains("\"role\":\"assistant\""));
        // content should be omitted when None (skip_serializing_if)
        assert!(!json.contains("\"content\":null"));
    }

    #[test]
    fn test_reject_unknown_fields_in_request() {
        let json = r#"{
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "foo": "bar"
        }"#;
        let err = serde_json::from_str::<ChatCompletionRequest>(json).unwrap_err();
        assert!(err.to_string().contains("unknown field"));
    }

    #[test]
    fn test_reject_unknown_fields_in_message() {
        let json = r#"{
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi", "foo": "bar"}]
        }"#;
        let err = serde_json::from_str::<ChatCompletionRequest>(json).unwrap_err();
        assert!(err.to_string().contains("unknown field"));
    }

    #[test]
    fn app_core_request_shape_is_accepted() {
        let plain_messages = vec![serde_json::json!({
            "role": "user",
            "content": "Hello."
        })];
        let tool_messages = vec![
            serde_json::json!({"role": "user", "content": "Use the calculator."}),
            serde_json::json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "provider_extra": {"trace": "abc"},
                    "function": {"name": "calc", "arguments": "{\"expr\":\"2+2\"}"}
                }]
            }),
            serde_json::json!({"role": "tool", "tool_call_id": "call_1", "content": "4"}),
        ];
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Evaluate arithmetic.",
                "parameters": {"type": "object", "properties": {}}
            }
        })];

        let bodies = [
            eidola_common::chat_completion_request_body(
                "test-model",
                &plain_messages,
                256,
                &[],
                false,
                false,
                None,
            ),
            eidola_common::chat_completion_request_body(
                "test-model",
                &plain_messages,
                256,
                &[],
                true,
                false,
                None,
            ),
            eidola_common::chat_completion_request_body(
                "test-model",
                &plain_messages,
                256,
                &[],
                true,
                true,
                None,
            ),
            eidola_common::chat_completion_request_body(
                "test-model",
                &tool_messages,
                256,
                &tools,
                false,
                false,
                None,
            ),
            eidola_common::chat_completion_request_body(
                "test-model",
                &tool_messages,
                256,
                &tools,
                true,
                false,
                None,
            ),
            eidola_common::chat_completion_request_body(
                "test-model",
                &tool_messages,
                256,
                &tools,
                true,
                true,
                None,
            ),
        ];

        for body in bodies {
            serde_json::from_value::<ChatCompletionRequest>(body)
                .expect("the body app-core sends must remain accepted by the strict server type");
        }
    }

    // -----------------------------------------------------------------
    // Tool calling: the three request shapes, and the two response shapes
    //
    // The server is a stateless proxy: what it parses it must forward
    // upstream unchanged, and what upstream sends it must relay to the
    // client unchanged. Each test below therefore asserts the *round trip*
    // (parse → re-serialize), which is exactly what `backend.rs` does with
    // `.json(request)` on the way up and `serde_json::to_string(&chunk)` on
    // the way down.
    // -----------------------------------------------------------------

    /// The full tool-bearing request: a `tools` array, an assistant message
    /// with `tool_calls` and `content: null`, and a `role: "tool"` result.
    /// Before this landed, every one of these 422'd at the extractor.
    const TOOL_REQUEST: &str = r#"{
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "what is 2+2?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "index": 0,
                 "function": {"name": "calc", "arguments": "{\"expr\":\"2+2\"}"},
                 "provider_extra": {"trace": "abc"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "4"}
        ],
        "tools": [
            {"type": "function", "function": {
                "name": "calc",
                "description": "Evaluate arithmetic.",
                "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}
            }}
        ],
        "tool_choice": "auto"
    }"#;

    #[test]
    fn tool_bearing_request_parses() {
        let request: ChatCompletionRequest = serde_json::from_str(TOOL_REQUEST).unwrap();

        assert_eq!(request.messages.len(), 3);

        // Assistant message: null content, verbatim call objects.
        let assistant = &request.messages[1];
        assert_eq!(assistant.role, Role::Assistant);
        assert!(assistant.content.is_none());
        assert_eq!(assistant.content_byte_len(), 0);
        let calls = assistant.tool_calls.as_ref().expect("tool_calls parsed");
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[0]["function"]["name"], "calc");

        // Tool result message.
        let tool = &request.messages[2];
        assert_eq!(tool.role, Role::Tool);
        assert_eq!(tool.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(tool.content_byte_len(), 1);

        // Request-level tool advertisement.
        assert_eq!(request.tools.as_ref().unwrap().len(), 1);
        assert_eq!(request.tool_choice.as_ref().unwrap(), "auto");
    }

    #[test]
    fn tool_bearing_request_forwards_upstream_unchanged() {
        let request: ChatCompletionRequest = serde_json::from_str(TOOL_REQUEST).unwrap();
        // `backend.rs` forwards with `.json(request)` — compare the
        // re-serialized value against the parsed original.
        let forwarded: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&request).unwrap()).unwrap();
        let original: serde_json::Value = serde_json::from_str(TOOL_REQUEST).unwrap();

        // Opaque pass-through: provider extension fields and the streaming
        // `index` key inside a call object survive the proxy.
        assert_eq!(
            forwarded["messages"][1]["tool_calls"], original["messages"][1]["tool_calls"],
            "tool_calls must be forwarded verbatim"
        );
        assert_eq!(forwarded["tools"], original["tools"]);
        assert_eq!(forwarded["tool_choice"], original["tool_choice"]);
        assert_eq!(forwarded["messages"][2], original["messages"][2]);

        // The explicit null content survives: several chat templates require
        // the key to exist on an assistant tool-call message.
        assert!(forwarded["messages"][1].get("content").is_some());
        assert!(forwarded["messages"][1]["content"].is_null());
    }

    #[test]
    fn message_without_content_key_is_accepted() {
        // Some clients omit `content` entirely rather than sending null.
        let json = r#"{
            "model": "gpt-4o",
            "messages": [{"role": "assistant", "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "n", "arguments": "{}"}}
            ]}]
        }"#;
        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(request.messages[0].content.is_none());
        assert_eq!(request.messages[0].content_byte_len(), 0);
    }

    #[test]
    fn request_without_tools_serializes_exactly_as_before() {
        // The pre-tool-calling shape must be untouched on the wire: no
        // `tools` / `tool_choice` / `tool_calls` / `tool_call_id` keys.
        let json = r#"{"model": "m", "messages": [{"role": "user", "content": "hi"}]}"#;
        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let out = serde_json::to_string(&request).unwrap();
        assert!(!out.contains("tools"));
        assert!(!out.contains("tool_choice"));
        assert!(!out.contains("tool_calls"));
        assert!(!out.contains("tool_call_id"));
    }

    #[test]
    fn blocking_response_relays_tool_calls_verbatim() {
        // The upstream's blocking answer: `finish_reason: "tool_calls"` plus
        // call objects carrying a provider extension field.
        let upstream = r#"{
            "id": "chatcmpl-1", "object": "chat.completion", "created": 1,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "calc", "arguments": "{\"expr\":\"2+2\"}"},
                     "provider_extra": {"trace": "abc"}}
                ]},
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        }"#;
        let parsed: ChatCompletionResponse = serde_json::from_str(upstream).unwrap();
        assert!(matches!(
            parsed.choices[0].finish_reason,
            Some(FinishReason::ToolCalls)
        ));

        let relayed: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&parsed).unwrap()).unwrap();
        let original: serde_json::Value = serde_json::from_str(upstream).unwrap();
        assert_eq!(
            relayed["choices"][0]["message"]["tool_calls"],
            original["choices"][0]["message"]["tool_calls"],
            "the client replays these objects verbatim — nothing may be dropped"
        );
        assert_eq!(relayed["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn streamed_tool_call_deltas_relay_verbatim() {
        // A streamed call arrives in fragments keyed by `index`; the client
        // reassembles them, so the proxy must relay each delta unchanged.
        let deltas = [
            r#"{"id":"c","object":"chat.completion.chunk","created":1,"model":"m",
                "choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[
                    {"index":0,"id":"call_1","type":"function",
                     "function":{"name":"ca","arguments":""},"provider_extra":{"t":1}}
                ]},"finish_reason":null}]}"#,
            r#"{"id":"c","object":"chat.completion.chunk","created":1,"model":"m",
                "choices":[{"index":0,"delta":{"tool_calls":[
                    {"index":0,"function":{"name":"lc","arguments":"{\"expr\":"}}
                ]},"finish_reason":null}]}"#,
            r#"{"id":"c","object":"chat.completion.chunk","created":1,"model":"m",
                "choices":[{"index":0,"delta":{"tool_calls":[
                    {"index":0,"function":{"arguments":"\"2+2\"}"}}
                ]},"finish_reason":"tool_calls"}]}"#,
        ];

        for delta in deltas {
            let chunk: ChatCompletionChunk = serde_json::from_str(delta).unwrap();
            let relayed: serde_json::Value =
                serde_json::from_str(&serde_json::to_string(&chunk).unwrap()).unwrap();
            let original: serde_json::Value = serde_json::from_str(delta).unwrap();
            assert_eq!(
                relayed["choices"][0]["delta"]["tool_calls"],
                original["choices"][0]["delta"]["tool_calls"],
                "streamed tool_calls must relay verbatim (index framing included)"
            );
            assert_eq!(
                relayed["choices"][0]["finish_reason"],
                original["choices"][0]["finish_reason"]
            );
        }
    }

    // -----------------------------------------------------------------
    // cache_key: accepted, held secret, never forwarded by serialization
    // -----------------------------------------------------------------

    /// 32 bytes 0x00..0x1f, base64url without padding.
    const CACHE_KEY: &str = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8";

    fn with_cache_key(key: &str) -> String {
        format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],"cache_key":"{key}"}}"#
        )
    }

    #[test]
    fn a_well_formed_cache_key_is_accepted_and_decoded() {
        let request: ChatCompletionRequest =
            serde_json::from_str(&with_cache_key(CACHE_KEY)).unwrap();
        let key = request.cache_key.as_ref().expect("cache_key parsed");
        let expected: Vec<u8> = (0u8..32).collect();
        assert_eq!(key.as_bytes().as_slice(), expected.as_slice());
    }

    #[test]
    fn a_malformed_cache_key_is_refused_without_echoing_it() {
        let secretish = "SECRETSECRETSECRETSECRETSECRETSECRETSECRE";
        for bad in [
            secretish,                                     // 41 characters
            &format!("{}=", &CACHE_KEY[..42]),             // padding
            &format!("{}B", &CACHE_KEY[..42]),             // non-canonical final character
            "+/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", // standard alphabet
        ] {
            let err = serde_json::from_str::<ChatCompletionRequest>(&with_cache_key(bad))
                .unwrap_err()
                .to_string();
            assert!(err.contains("cache_key must be 32 bytes"), "{err}");
            assert!(!err.contains(bad), "the refusal quoted the key: {err}");
        }
        let err = serde_json::from_str::<ChatCompletionRequest>(
            r#"{"model":"m","messages":[],"cache_key":5}"#,
        );
        assert!(err.is_err());
    }

    #[test]
    fn a_cache_key_never_prints() {
        let request: ChatCompletionRequest =
            serde_json::from_str(&with_cache_key(CACHE_KEY)).unwrap();
        let printed = format!("{request:?}");
        assert!(printed.contains("CacheKey(<redacted>)"), "{printed}");
        assert!(!printed.contains(CACHE_KEY));
    }

    /// The request re-serializes for the Tinfoil upstream without the key:
    /// `backend.rs` forwards `.json(request)`, and the key is not something
    /// that upstream may see.
    #[test]
    fn a_cache_key_is_never_serialized() {
        let request: ChatCompletionRequest =
            serde_json::from_str(&with_cache_key(CACHE_KEY)).unwrap();
        let forwarded = serde_json::to_string(&request).unwrap();
        assert!(!forwarded.contains("cache_key"), "{forwarded}");
        assert!(!forwarded.contains(CACHE_KEY), "{forwarded}");
        let cloned = serde_json::to_string(&request.clone()).unwrap();
        assert!(!cloned.contains("cache_key"));
        // And the key alone refuses to serialize.
        assert!(serde_json::to_string(request.cache_key.as_ref().unwrap()).is_err());
    }

    #[test]
    fn the_shared_body_with_a_cache_key_is_accepted() {
        let body = eidola_common::chat_completion_request_body(
            "test-model",
            &[serde_json::json!({"role": "user", "content": "Hello."})],
            256,
            &[],
            true,
            true,
            Some(CACHE_KEY),
        );
        let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        assert!(request.cache_key.is_some());
    }

    #[test]
    fn test_serialize_error_response() {
        let error = ErrorResponse::new("Something went wrong", "internal_error");

        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("\"message\":\"Something went wrong\""));
        assert!(json.contains("\"type\":\"internal_error\""));
    }
}
