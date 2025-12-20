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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_simple_request() {
        let request = MessagesRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("Hello!".to_string()),
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            metadata: None,
        };

        let json = serde_json::to_string(&request).unwrap();

        assert!(json.contains("\"model\":\"claude-sonnet-4-20250514\""));
        assert!(json.contains("\"max_tokens\":1024"));
        assert!(json.contains("\"role\":\"user\""));
        // Optional fields should not appear
        assert!(!json.contains("\"system\""));
        assert!(!json.contains("\"temperature\""));
    }

    #[test]
    fn test_serialize_request_with_system() {
        let request = MessagesRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("Hi".to_string()),
            }],
            system: Some("You are a helpful assistant.".to_string()),
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            stop_sequences: Some(vec!["END".to_string()]),
            stream: Some(true),
            metadata: None,
        };

        let json = serde_json::to_string(&request).unwrap();

        assert!(json.contains("\"system\":\"You are a helpful assistant.\""));
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"stop_sequences\":[\"END\"]"));
        assert!(json.contains("\"stream\":true"));
    }

    #[test]
    fn test_serialize_multimodal_content() {
        let request = MessagesRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Blocks(vec![
                    ContentBlock::Text {
                        text: "What's in this image?".to_string(),
                    },
                    ContentBlock::Image {
                        source: ImageSource::Base64 {
                            media_type: "image/png".to_string(),
                            data: "iVBORw0KGgo=".to_string(),
                        },
                    },
                ]),
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            metadata: None,
        };

        let json = serde_json::to_string(&request).unwrap();

        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"type\":\"image\""));
        assert!(json.contains("\"type\":\"base64\""));
        assert!(json.contains("\"media_type\":\"image/png\""));
    }

    #[test]
    fn test_serialize_metadata() {
        let request = MessagesRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("Hi".to_string()),
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            metadata: Some(Metadata {
                user_id: Some("user-123".to_string()),
            }),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"user_id\":\"user-123\""));
    }

    #[test]
    fn test_parse_response() {
        let json = r#"{
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        }"#;

        let response: MessagesResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.id, "msg_01XFDUDYJgAACzvnptvVoYEL");
        assert_eq!(response.role, Role::Assistant);
        assert_eq!(response.content.len(), 1);
        assert!(matches!(
            &response.content[0],
            ResponseContentBlock::Text { text } if text == "Hello!"
        ));
        assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
    }

    #[test]
    fn test_parse_response_max_tokens() {
        let json = r#"{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Truncated..."}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "max_tokens",
            "stop_sequence": null,
            "usage": {"input_tokens": 10, "output_tokens": 100}
        }"#;

        let response: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.stop_reason, Some(StopReason::MaxTokens));
    }

    #[test]
    fn test_parse_response_with_cache_tokens() {
        let json = r#"{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 500
            }
        }"#;

        let response: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.usage.cache_creation_input_tokens, Some(1000));
        assert_eq!(response.usage.cache_read_input_tokens, Some(500));
    }

    // ========================================================================
    // Stream event parsing tests
    // ========================================================================

    #[test]
    fn test_parse_stream_message_start() {
        let json = r#"{
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 10, "output_tokens": 0}
            }
        }"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();

        match event {
            StreamEvent::MessageStart { message } => {
                assert_eq!(message.id, "msg_123");
                assert_eq!(message.role, Role::Assistant);
            }
            _ => panic!("expected MessageStart"),
        }
    }

    #[test]
    fn test_parse_stream_content_block_delta() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"}
        }"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();

        match event {
            StreamEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                match delta {
                    ContentDelta::TextDelta { text } => assert_eq!(text, "Hello"),
                }
            }
            _ => panic!("expected ContentBlockDelta"),
        }
    }

    #[test]
    fn test_parse_stream_message_delta() {
        let json = r#"{
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"input_tokens": 10, "output_tokens": 20}
        }"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();

        match event {
            StreamEvent::MessageDelta { delta, usage } => {
                assert_eq!(delta.stop_reason, Some(StopReason::EndTurn));
                assert_eq!(usage.output_tokens, 20);
            }
            _ => panic!("expected MessageDelta"),
        }
    }

    #[test]
    fn test_parse_stream_ping() {
        let json = r#"{"type": "ping"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, StreamEvent::Ping));
    }

    #[test]
    fn test_parse_stream_message_stop() {
        let json = r#"{"type": "message_stop"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, StreamEvent::MessageStop));
    }

    #[test]
    fn test_parse_stream_error() {
        let json = r#"{
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Server overloaded"}
        }"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();

        match event {
            StreamEvent::Error { error } => {
                assert_eq!(error.error_type, "overloaded_error");
                assert_eq!(error.message, "Server overloaded");
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn test_parse_error_response() {
        let json = r#"{
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "messages: field required"
            }
        }"#;

        let response: ErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.error.error_type, "invalid_request_error");
        assert_eq!(response.error.message, "messages: field required");
    }
}
