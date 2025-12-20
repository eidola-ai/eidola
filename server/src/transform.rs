//! Transformations between OpenAI and Anthropic API formats.

use crate::anthropic;
use crate::openai;

/// Default max tokens if not specified in request.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Convert an OpenAI chat completion request to an Anthropic messages request.
pub fn openai_to_anthropic(
    request: openai::ChatCompletionRequest,
) -> Result<anthropic::MessagesRequest, TransformError> {
    let mut system_prompt: Option<String> = None;
    let mut messages: Vec<anthropic::Message> = Vec::new();

    for msg in request.messages {
        match msg.role {
            openai::Role::System => {
                // Anthropic requires system as a separate parameter, not in messages.
                // Concatenate multiple system messages if present.
                let text = extract_text_content(&msg.content)?;
                if let Some(existing) = system_prompt {
                    system_prompt = Some(format!("{}\n\n{}", existing, text));
                } else {
                    system_prompt = Some(text);
                }
            }
            openai::Role::User => {
                messages.push(anthropic::Message {
                    role: anthropic::Role::User,
                    content: convert_content(&msg.content)?,
                });
            }
            openai::Role::Assistant => {
                messages.push(anthropic::Message {
                    role: anthropic::Role::Assistant,
                    content: convert_content(&msg.content)?,
                });
            }
        }
    }

    // Anthropic requires at least one message
    if messages.is_empty() {
        return Err(TransformError::EmptyMessages);
    }

    // Build metadata if user ID provided
    let metadata = request.user.map(|user_id| anthropic::Metadata {
        user_id: Some(user_id),
    });

    Ok(anthropic::MessagesRequest {
        model: map_model_name(&request.model),
        max_tokens: request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        messages,
        system: system_prompt,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: None,
        stop_sequences: request.stop.map(|s| s.into_vec()),
        stream: if request.stream { Some(true) } else { None },
        metadata,
    })
}

/// Convert message content from OpenAI to Anthropic format.
fn convert_content(
    content: &openai::MessageContent,
) -> Result<anthropic::MessageContent, TransformError> {
    match content {
        openai::MessageContent::Text(text) => Ok(anthropic::MessageContent::Text(text.clone())),
        openai::MessageContent::Parts(parts) => {
            let blocks: Result<Vec<_>, _> = parts.iter().map(convert_content_part).collect();
            Ok(anthropic::MessageContent::Blocks(blocks?))
        }
    }
}

/// Convert a single content part from OpenAI to Anthropic format.
fn convert_content_part(
    part: &openai::ContentPart,
) -> Result<anthropic::ContentBlock, TransformError> {
    match part {
        openai::ContentPart::Text { text } => {
            Ok(anthropic::ContentBlock::Text { text: text.clone() })
        }
        openai::ContentPart::ImageUrl { image_url } => {
            let source = parse_image_url(&image_url.url)?;
            Ok(anthropic::ContentBlock::Image { source })
        }
    }
}

/// Parse an image URL or data URI into Anthropic image source format.
fn parse_image_url(url: &str) -> Result<anthropic::ImageSource, TransformError> {
    if url.starts_with("data:") {
        // Parse data URI: data:image/png;base64,<data>
        let without_prefix = url
            .strip_prefix("data:")
            .ok_or(TransformError::InvalidImageUrl)?;
        let (media_type, rest) = without_prefix
            .split_once(";base64,")
            .ok_or(TransformError::InvalidImageUrl)?;

        Ok(anthropic::ImageSource::Base64 {
            media_type: media_type.to_string(),
            data: rest.to_string(),
        })
    } else {
        // Regular URL
        Ok(anthropic::ImageSource::Url {
            url: url.to_string(),
        })
    }
}

/// Extract text from message content.
fn extract_text_content(content: &openai::MessageContent) -> Result<String, TransformError> {
    match content {
        openai::MessageContent::Text(text) => Ok(text.clone()),
        openai::MessageContent::Parts(parts) => {
            let texts: Vec<&str> = parts
                .iter()
                .filter_map(|p| match p {
                    openai::ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            if texts.is_empty() {
                Err(TransformError::NoTextContent)
            } else {
                Ok(texts.join("\n"))
            }
        }
    }
}

/// Map OpenAI-style model names to Anthropic model names.
fn map_model_name(model: &str) -> String {
    // Allow pass-through of Anthropic model names
    if model.starts_with("claude-") {
        return model.to_string();
    }

    // Map common aliases
    match model {
        "gpt-4o" | "gpt-4" => "claude-sonnet-4-20250514".to_string(),
        "gpt-4o-mini" | "gpt-3.5-turbo" => "claude-3-5-haiku-20241022".to_string(),
        "o1" | "o1-preview" => "claude-sonnet-4-20250514".to_string(),
        _ => model.to_string(), // Pass through unknown models
    }
}

/// Convert an Anthropic response to OpenAI format.
pub fn anthropic_to_openai(
    response: anthropic::MessagesResponse,
    original_model: &str,
) -> openai::ChatCompletionResponse {
    // Extract text content from response
    let content: Option<String> = response
        .content
        .iter()
        .map(|block| match block {
            anthropic::ResponseContentBlock::Text { text } => text.clone(),
        })
        .reduce(|a, b| format!("{}{}", a, b));

    let finish_reason = response.stop_reason.map(|r| match r {
        anthropic::StopReason::EndTurn | anthropic::StopReason::StopSequence => {
            openai::FinishReason::Stop
        }
        anthropic::StopReason::MaxTokens => openai::FinishReason::Length,
        anthropic::StopReason::Refusal => openai::FinishReason::ContentFilter,
        _ => openai::FinishReason::Stop,
    });

    let choices = vec![openai::Choice {
        index: 0,
        message: openai::AssistantMessage {
            role: openai::Role::Assistant,
            content,
        },
        finish_reason,
    }];

    let usage = Some(openai::Usage {
        prompt_tokens: response.usage.input_tokens,
        completion_tokens: response.usage.output_tokens,
        total_tokens: response.usage.input_tokens + response.usage.output_tokens,
    });

    openai::ChatCompletionResponse::new(response.id, original_model.to_string(), choices, usage)
}

/// Convert Anthropic stream events to OpenAI chunk format.
pub struct StreamTransformer {
    message_id: Option<String>,
    model: String,
}

impl StreamTransformer {
    pub fn new(model: String) -> Self {
        Self {
            message_id: None,
            model,
        }
    }

    /// Transform an Anthropic stream event to an OpenAI chunk.
    /// Returns None for events that don't produce output (like ping).
    pub fn transform(
        &mut self,
        event: anthropic::StreamEvent,
    ) -> Option<openai::ChatCompletionChunk> {
        match event {
            anthropic::StreamEvent::MessageStart { message } => {
                self.message_id = Some(message.id.clone());
                // Send initial chunk with role
                Some(openai::ChatCompletionChunk::new(
                    message.id,
                    self.model.clone(),
                    vec![openai::ChunkChoice {
                        index: 0,
                        delta: openai::ChunkDelta {
                            role: Some(openai::Role::Assistant),
                            content: None,
                        },
                        finish_reason: None,
                    }],
                ))
            }
            anthropic::StreamEvent::ContentBlockDelta { delta, .. } => {
                let text = match delta {
                    anthropic::ContentDelta::TextDelta { text } => text,
                };
                let id = self
                    .message_id
                    .clone()
                    .unwrap_or_else(|| "msg_unknown".to_string());
                Some(openai::ChatCompletionChunk::new(
                    id,
                    self.model.clone(),
                    vec![openai::ChunkChoice {
                        index: 0,
                        delta: openai::ChunkDelta {
                            role: None,
                            content: Some(text),
                        },
                        finish_reason: None,
                    }],
                ))
            }
            anthropic::StreamEvent::MessageDelta { delta, .. } => {
                let finish_reason = delta.stop_reason.map(|r| match r {
                    anthropic::StopReason::EndTurn | anthropic::StopReason::StopSequence => {
                        openai::FinishReason::Stop
                    }
                    anthropic::StopReason::MaxTokens => openai::FinishReason::Length,
                    anthropic::StopReason::Refusal => openai::FinishReason::ContentFilter,
                    _ => openai::FinishReason::Stop,
                });
                let id = self
                    .message_id
                    .clone()
                    .unwrap_or_else(|| "msg_unknown".to_string());
                Some(openai::ChatCompletionChunk::new(
                    id,
                    self.model.clone(),
                    vec![openai::ChunkChoice {
                        index: 0,
                        delta: openai::ChunkDelta {
                            role: None,
                            content: None,
                        },
                        finish_reason,
                    }],
                ))
            }
            // These events don't produce OpenAI output
            anthropic::StreamEvent::ContentBlockStart { .. }
            | anthropic::StreamEvent::ContentBlockStop { .. }
            | anthropic::StreamEvent::MessageStop
            | anthropic::StreamEvent::Ping => None,
            anthropic::StreamEvent::Error { .. } => None,
        }
    }
}

/// Errors that can occur during transformation.
#[derive(Debug)]
pub enum TransformError {
    EmptyMessages,
    NoTextContent,
    InvalidImageUrl,
}

impl std::fmt::Display for TransformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformError::EmptyMessages => write!(f, "messages array cannot be empty"),
            TransformError::NoTextContent => write!(f, "message must contain text content"),
            TransformError::InvalidImageUrl => write!(f, "invalid image URL or data URI"),
        }
    }
}

impl std::error::Error for TransformError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic;
    use crate::openai;

    // ========================================================================
    // Helper constructors for tests
    // ========================================================================

    fn user_message(text: &str) -> openai::Message {
        openai::Message {
            role: openai::Role::User,
            content: openai::MessageContent::Text(text.to_string()),
            name: None,
        }
    }

    fn system_message(text: &str) -> openai::Message {
        openai::Message {
            role: openai::Role::System,
            content: openai::MessageContent::Text(text.to_string()),
            name: None,
        }
    }

    fn assistant_message(text: &str) -> openai::Message {
        openai::Message {
            role: openai::Role::Assistant,
            content: openai::MessageContent::Text(text.to_string()),
            name: None,
        }
    }

    fn simple_request(messages: Vec<openai::Message>) -> openai::ChatCompletionRequest {
        openai::ChatCompletionRequest {
            model: "gpt-4o".to_string(),
            messages,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: false,
            stop: None,
            user: None,
        }
    }

    // ========================================================================
    // openai_to_anthropic tests
    // ========================================================================

    #[test]
    fn test_simple_user_message() {
        let request = simple_request(vec![user_message("Hello, world!")]);
        let result = openai_to_anthropic(request).unwrap();

        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, anthropic::Role::User);
        assert!(matches!(
            &result.messages[0].content,
            anthropic::MessageContent::Text(t) if t == "Hello, world!"
        ));
        assert!(result.system.is_none());
    }

    #[test]
    fn test_system_message_extraction() {
        let request = simple_request(vec![
            system_message("You are a helpful assistant."),
            user_message("Hello!"),
        ]);
        let result = openai_to_anthropic(request).unwrap();

        assert_eq!(result.system, Some("You are a helpful assistant.".to_string()));
        assert_eq!(result.messages.len(), 1); // System message not in messages
    }

    #[test]
    fn test_multiple_system_messages_concatenated() {
        let request = simple_request(vec![
            system_message("First instruction."),
            system_message("Second instruction."),
            user_message("Hello!"),
        ]);
        let result = openai_to_anthropic(request).unwrap();

        assert_eq!(
            result.system,
            Some("First instruction.\n\nSecond instruction.".to_string())
        );
    }

    #[test]
    fn test_conversation_history() {
        let request = simple_request(vec![
            system_message("Be concise."),
            user_message("What is 2+2?"),
            assistant_message("4"),
            user_message("And 3+3?"),
        ]);
        let result = openai_to_anthropic(request).unwrap();

        assert_eq!(result.messages.len(), 3);
        assert_eq!(result.messages[0].role, anthropic::Role::User);
        assert_eq!(result.messages[1].role, anthropic::Role::Assistant);
        assert_eq!(result.messages[2].role, anthropic::Role::User);
    }

    #[test]
    fn test_empty_messages_error() {
        let request = simple_request(vec![]);
        let result = openai_to_anthropic(request);

        assert!(matches!(result, Err(TransformError::EmptyMessages)));
    }

    #[test]
    fn test_only_system_message_error() {
        let request = simple_request(vec![system_message("System only")]);
        let result = openai_to_anthropic(request);

        // System messages are extracted, leaving empty messages array
        assert!(matches!(result, Err(TransformError::EmptyMessages)));
    }

    #[test]
    fn test_default_max_tokens() {
        let request = simple_request(vec![user_message("Hello")]);
        let result = openai_to_anthropic(request).unwrap();

        assert_eq!(result.max_tokens, DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn test_custom_max_tokens() {
        let mut request = simple_request(vec![user_message("Hello")]);
        request.max_tokens = Some(100);
        let result = openai_to_anthropic(request).unwrap();

        assert_eq!(result.max_tokens, 100);
    }

    #[test]
    fn test_temperature_passthrough() {
        let mut request = simple_request(vec![user_message("Hello")]);
        request.temperature = Some(0.7);
        let result = openai_to_anthropic(request).unwrap();

        assert_eq!(result.temperature, Some(0.7));
    }

    #[test]
    fn test_stop_sequence_single() {
        let mut request = simple_request(vec![user_message("Hello")]);
        request.stop = Some(openai::StopSequence::Single("STOP".to_string()));
        let result = openai_to_anthropic(request).unwrap();

        assert_eq!(result.stop_sequences, Some(vec!["STOP".to_string()]));
    }

    #[test]
    fn test_stop_sequence_multiple() {
        let mut request = simple_request(vec![user_message("Hello")]);
        request.stop = Some(openai::StopSequence::Multiple(vec![
            "END".to_string(),
            "STOP".to_string(),
        ]));
        let result = openai_to_anthropic(request).unwrap();

        assert_eq!(
            result.stop_sequences,
            Some(vec!["END".to_string(), "STOP".to_string()])
        );
    }

    #[test]
    fn test_user_id_mapping() {
        let mut request = simple_request(vec![user_message("Hello")]);
        request.user = Some("user-123".to_string());
        let result = openai_to_anthropic(request).unwrap();

        assert!(result.metadata.is_some());
        assert_eq!(
            result.metadata.unwrap().user_id,
            Some("user-123".to_string())
        );
    }

    #[test]
    fn test_stream_flag() {
        let mut request = simple_request(vec![user_message("Hello")]);
        request.stream = true;
        let result = openai_to_anthropic(request).unwrap();

        assert_eq!(result.stream, Some(true));
    }

    // ========================================================================
    // Model name mapping tests
    // ========================================================================

    #[test]
    fn test_model_mapping_gpt4o() {
        assert_eq!(map_model_name("gpt-4o"), "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_model_mapping_gpt4() {
        assert_eq!(map_model_name("gpt-4"), "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_model_mapping_gpt4o_mini() {
        assert_eq!(map_model_name("gpt-4o-mini"), "claude-3-5-haiku-20241022");
    }

    #[test]
    fn test_model_mapping_gpt35_turbo() {
        assert_eq!(map_model_name("gpt-3.5-turbo"), "claude-3-5-haiku-20241022");
    }

    #[test]
    fn test_model_passthrough_claude() {
        assert_eq!(
            map_model_name("claude-3-opus-20240229"),
            "claude-3-opus-20240229"
        );
    }

    #[test]
    fn test_model_passthrough_unknown() {
        assert_eq!(map_model_name("some-other-model"), "some-other-model");
    }

    // ========================================================================
    // Image URL parsing tests
    // ========================================================================

    #[test]
    fn test_parse_image_url_regular() {
        let result = parse_image_url("https://example.com/image.png").unwrap();

        assert!(matches!(
            result,
            anthropic::ImageSource::Url { url } if url == "https://example.com/image.png"
        ));
    }

    #[test]
    fn test_parse_image_url_data_uri() {
        let result = parse_image_url("data:image/png;base64,iVBORw0KGgo=").unwrap();

        match result {
            anthropic::ImageSource::Base64 { media_type, data } => {
                assert_eq!(media_type, "image/png");
                assert_eq!(data, "iVBORw0KGgo=");
            }
            _ => panic!("expected Base64 variant"),
        }
    }

    #[test]
    fn test_parse_image_url_data_uri_jpeg() {
        let result = parse_image_url("data:image/jpeg;base64,/9j/4AAQ").unwrap();

        match result {
            anthropic::ImageSource::Base64 { media_type, data } => {
                assert_eq!(media_type, "image/jpeg");
                assert_eq!(data, "/9j/4AAQ");
            }
            _ => panic!("expected Base64 variant"),
        }
    }

    #[test]
    fn test_parse_image_url_invalid_data_uri() {
        // Missing base64 marker
        let result = parse_image_url("data:image/png,notbase64");
        assert!(matches!(result, Err(TransformError::InvalidImageUrl)));
    }

    // ========================================================================
    // Multimodal content tests
    // ========================================================================

    #[test]
    fn test_multimodal_text_and_image() {
        let message = openai::Message {
            role: openai::Role::User,
            content: openai::MessageContent::Parts(vec![
                openai::ContentPart::Text {
                    text: "What's in this image?".to_string(),
                },
                openai::ContentPart::ImageUrl {
                    image_url: openai::ImageUrl {
                        url: "https://example.com/cat.jpg".to_string(),
                        detail: None,
                    },
                },
            ]),
            name: None,
        };

        let request = openai::ChatCompletionRequest {
            model: "gpt-4o".to_string(),
            messages: vec![message],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: false,
            stop: None,
            user: None,
        };

        let result = openai_to_anthropic(request).unwrap();

        match &result.messages[0].content {
            anthropic::MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert!(matches!(&blocks[0], anthropic::ContentBlock::Text { text } if text == "What's in this image?"));
                assert!(matches!(&blocks[1], anthropic::ContentBlock::Image { .. }));
            }
            _ => panic!("expected Blocks variant"),
        }
    }

    // ========================================================================
    // anthropic_to_openai response tests
    // ========================================================================

    #[test]
    fn test_response_conversion() {
        let anthropic_response = anthropic::MessagesResponse {
            id: "msg_123".to_string(),
            response_type: "message".to_string(),
            role: anthropic::Role::Assistant,
            content: vec![anthropic::ResponseContentBlock::Text {
                text: "Hello!".to_string(),
            }],
            model: "claude-sonnet-4-20250514".to_string(),
            stop_reason: Some(anthropic::StopReason::EndTurn),
            stop_sequence: None,
            usage: anthropic::Usage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };

        let result = anthropic_to_openai(anthropic_response, "gpt-4o");

        assert_eq!(result.id, "msg_123");
        assert_eq!(result.model, "gpt-4o");
        assert_eq!(result.choices.len(), 1);
        assert_eq!(result.choices[0].message.content, Some("Hello!".to_string()));
        assert!(matches!(
            result.choices[0].finish_reason,
            Some(openai::FinishReason::Stop)
        ));

        let usage = result.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_response_stop_reason_max_tokens() {
        let anthropic_response = anthropic::MessagesResponse {
            id: "msg_123".to_string(),
            response_type: "message".to_string(),
            role: anthropic::Role::Assistant,
            content: vec![anthropic::ResponseContentBlock::Text {
                text: "Truncated...".to_string(),
            }],
            model: "claude-sonnet-4-20250514".to_string(),
            stop_reason: Some(anthropic::StopReason::MaxTokens),
            stop_sequence: None,
            usage: anthropic::Usage {
                input_tokens: 10,
                output_tokens: 100,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };

        let result = anthropic_to_openai(anthropic_response, "gpt-4o");

        assert!(matches!(
            result.choices[0].finish_reason,
            Some(openai::FinishReason::Length)
        ));
    }

    #[test]
    fn test_response_multiple_content_blocks_concatenated() {
        let anthropic_response = anthropic::MessagesResponse {
            id: "msg_123".to_string(),
            response_type: "message".to_string(),
            role: anthropic::Role::Assistant,
            content: vec![
                anthropic::ResponseContentBlock::Text {
                    text: "First ".to_string(),
                },
                anthropic::ResponseContentBlock::Text {
                    text: "Second".to_string(),
                },
            ],
            model: "claude-sonnet-4-20250514".to_string(),
            stop_reason: Some(anthropic::StopReason::EndTurn),
            stop_sequence: None,
            usage: anthropic::Usage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };

        let result = anthropic_to_openai(anthropic_response, "gpt-4o");

        assert_eq!(
            result.choices[0].message.content,
            Some("First Second".to_string())
        );
    }

    // ========================================================================
    // StreamTransformer tests
    // ========================================================================

    #[test]
    fn test_stream_transformer_message_start() {
        let mut transformer = StreamTransformer::new("gpt-4o".to_string());

        let event = anthropic::StreamEvent::MessageStart {
            message: anthropic::MessageStartData {
                id: "msg_123".to_string(),
                message_type: "message".to_string(),
                role: anthropic::Role::Assistant,
                model: "claude-sonnet-4-20250514".to_string(),
                usage: anthropic::Usage {
                    input_tokens: 10,
                    output_tokens: 0,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            },
        };

        let chunk = transformer.transform(event).unwrap();

        assert_eq!(chunk.id, "msg_123");
        assert_eq!(chunk.model, "gpt-4o");
        assert_eq!(chunk.choices[0].delta.role, Some(openai::Role::Assistant));
        assert!(chunk.choices[0].delta.content.is_none());
    }

    #[test]
    fn test_stream_transformer_content_delta() {
        let mut transformer = StreamTransformer::new("gpt-4o".to_string());

        // First, start the message to set the ID
        transformer.transform(anthropic::StreamEvent::MessageStart {
            message: anthropic::MessageStartData {
                id: "msg_123".to_string(),
                message_type: "message".to_string(),
                role: anthropic::Role::Assistant,
                model: "claude-sonnet-4-20250514".to_string(),
                usage: anthropic::Usage {
                    input_tokens: 10,
                    output_tokens: 0,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            },
        });

        let event = anthropic::StreamEvent::ContentBlockDelta {
            index: 0,
            delta: anthropic::ContentDelta::TextDelta {
                text: "Hello".to_string(),
            },
        };

        let chunk = transformer.transform(event).unwrap();

        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
        assert!(chunk.choices[0].delta.role.is_none());
    }

    #[test]
    fn test_stream_transformer_ping_returns_none() {
        let mut transformer = StreamTransformer::new("gpt-4o".to_string());
        let result = transformer.transform(anthropic::StreamEvent::Ping);
        assert!(result.is_none());
    }

    #[test]
    fn test_stream_transformer_message_stop_returns_none() {
        let mut transformer = StreamTransformer::new("gpt-4o".to_string());
        let result = transformer.transform(anthropic::StreamEvent::MessageStop);
        assert!(result.is_none());
    }

    #[test]
    fn test_stream_transformer_message_delta_with_stop_reason() {
        let mut transformer = StreamTransformer::new("gpt-4o".to_string());

        // Start message first
        transformer.transform(anthropic::StreamEvent::MessageStart {
            message: anthropic::MessageStartData {
                id: "msg_123".to_string(),
                message_type: "message".to_string(),
                role: anthropic::Role::Assistant,
                model: "claude-sonnet-4-20250514".to_string(),
                usage: anthropic::Usage {
                    input_tokens: 10,
                    output_tokens: 0,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            },
        });

        let event = anthropic::StreamEvent::MessageDelta {
            delta: anthropic::MessageDeltaData {
                stop_reason: Some(anthropic::StopReason::EndTurn),
                stop_sequence: None,
            },
            usage: anthropic::Usage {
                input_tokens: 10,
                output_tokens: 20,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };

        let chunk = transformer.transform(event).unwrap();

        assert!(matches!(
            chunk.choices[0].finish_reason,
            Some(openai::FinishReason::Stop)
        ));
    }
}
