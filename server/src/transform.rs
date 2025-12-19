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
