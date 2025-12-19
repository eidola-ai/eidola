//! Anthropic Claude API types for the Messages API.
//!
//! Note: Some fields in these types are not directly used but are required for
//! proper JSON deserialization of API responses.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// A request to the Anthropic Messages API.
#[derive(Debug, Clone, Serialize)]
pub struct MessagesRequest {
    /// The model to use (e.g., "claude-sonnet-4-20250514").
    pub model: String,

    /// The maximum number of tokens to generate.
    pub max_tokens: u32,

    /// The conversation messages.
    pub messages: Vec<Message>,

    /// Optional system prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// Sampling temperature (0.0 to 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Nucleus sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Top-k sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    /// Whether to stream the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Optional metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

/// Request metadata.
#[derive(Debug, Clone, Serialize)]
pub struct Metadata {
    /// An external identifier for the user.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

/// A message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// The role (user or assistant).
    pub role: Role,

    /// The message content.
    pub content: MessageContent,
}

/// Message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// Message content can be a string or array of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A content block within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content.
    Text { text: String },

    /// Image content.
    Image { source: ImageSource },
}

/// Image source specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Base64-encoded image data.
    Base64 { media_type: String, data: String },

    /// Image from URL.
    Url { url: String },
}

/// A response from the Anthropic Messages API.
#[derive(Debug, Clone, Deserialize)]
pub struct MessagesResponse {
    /// Unique message identifier.
    pub id: String,

    /// Object type (always "message").
    #[serde(rename = "type")]
    pub response_type: String,

    /// The role (always "assistant").
    pub role: Role,

    /// The generated content blocks.
    pub content: Vec<ResponseContentBlock>,

    /// The model that generated the response.
    pub model: String,

    /// The reason generation stopped.
    pub stop_reason: Option<StopReason>,

    /// The stop sequence that triggered stopping, if any.
    pub stop_sequence: Option<String>,

    /// Token usage statistics.
    pub usage: Usage,
}

/// A content block in a response.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentBlock {
    /// Text content.
    Text { text: String },
}

/// The reason generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    PauseTurn,
    Refusal,
}

/// Token usage statistics.
#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
}

// ============================================================================
// Streaming types
// ============================================================================

/// A server-sent event from the Anthropic streaming API.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Start of the message.
    MessageStart { message: MessageStartData },

    /// Start of a content block.
    ContentBlockStart {
        index: u32,
        content_block: ContentBlockStartData,
    },

    /// Incremental update to a content block.
    ContentBlockDelta { index: u32, delta: ContentDelta },

    /// End of a content block.
    ContentBlockStop { index: u32 },

    /// Final message metadata.
    MessageDelta {
        delta: MessageDeltaData,
        usage: Usage,
    },

    /// End of the message stream.
    MessageStop,

    /// Ping event (keepalive).
    Ping,

    /// Error event.
    Error { error: ErrorData },
}

/// Data in a message_start event.
#[derive(Debug, Clone, Deserialize)]
pub struct MessageStartData {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: Role,
    pub model: String,
    pub usage: Usage,
}

/// Data in a content_block_start event.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockStartData {
    Text { text: String },
}

/// Delta update for a content block.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    TextDelta { text: String },
}

/// Delta data in a message_delta event.
#[derive(Debug, Clone, Deserialize)]
pub struct MessageDeltaData {
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
}

/// Error data.
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorData {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

/// An error response from Anthropic.
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorResponse {
    #[serde(rename = "type")]
    pub response_type: String,
    pub error: ErrorData,
}
