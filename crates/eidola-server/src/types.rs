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

    /// Up to 4 sequences where the API will stop generating.
    #[serde(default)]
    pub stop: Option<StopSequence>,
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
    pub content: MessageContent,

    /// An optional name for the participant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// The role of a message author.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
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
}

/// The reason the model stopped generating.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
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
            MessageContent::Text(t) if t == "Hello!"
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
            MessageContent::Parts(parts) => {
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
            MessageContent::Parts(parts) => match &parts[0] {
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
    fn test_serialize_error_response() {
        let error = ErrorResponse::new("Something went wrong", "internal_error");

        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("\"message\":\"Something went wrong\""));
        assert!(json.contains("\"type\":\"internal_error\""));
    }
}
