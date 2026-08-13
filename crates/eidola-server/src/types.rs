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

    /// Pricing in integer credits per 1k tokens.
    pub pricing: ModelPricing,
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
            ),
            eidola_common::chat_completion_request_body(
                "test-model",
                &plain_messages,
                256,
                &[],
                true,
                false,
            ),
            eidola_common::chat_completion_request_body(
                "test-model",
                &plain_messages,
                256,
                &[],
                true,
                true,
            ),
            eidola_common::chat_completion_request_body(
                "test-model",
                &tool_messages,
                256,
                &tools,
                false,
                false,
            ),
            eidola_common::chat_completion_request_body(
                "test-model",
                &tool_messages,
                256,
                &tools,
                true,
                false,
            ),
            eidola_common::chat_completion_request_body(
                "test-model",
                &tool_messages,
                256,
                &tools,
                true,
                true,
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

    #[test]
    fn test_serialize_error_response() {
        let error = ErrorResponse::new("Something went wrong", "internal_error");

        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("\"message\":\"Something went wrong\""));
        assert!(json.contains("\"type\":\"internal_error\""));
    }
}
