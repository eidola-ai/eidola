//! OpenAPI documentation for the Eidolons API.

use utoipa::OpenApi;

use crate::openai::{
    AssistantMessage, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice,
    ChunkChoice, ChunkDelta, ContentPart, ErrorDetail, ErrorResponse, FinishReason, ImageUrl,
    Message, MessageContent, Role, StopSequence, Usage,
};

/// OpenAPI documentation for the Eidolons API.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "Eidolons API",
        description = "OpenAI-compatible proxy API for AI providers",
        version = "0.1.0"
    ),
    paths(openapi_paths::health, openapi_paths::chat_completions),
    components(schemas(
        ChatCompletionRequest,
        ChatCompletionResponse,
        ChatCompletionChunk,
        Message,
        Role,
        MessageContent,
        ContentPart,
        ImageUrl,
        StopSequence,
        Choice,
        ChunkChoice,
        ChunkDelta,
        AssistantMessage,
        FinishReason,
        Usage,
        ErrorResponse,
        ErrorDetail,
    ))
)]
pub struct ApiDoc;

// Dummy functions for utoipa path documentation.
// These are never called - they exist only to provide OpenAPI endpoint metadata.
#[allow(dead_code)]
mod openapi_paths {
    use crate::openai::{ChatCompletionRequest, ChatCompletionResponse, ErrorResponse};

    /// Health check endpoint.
    #[utoipa::path(
        get,
        path = "/health",
        responses(
            (status = 200, description = "Server is healthy", body = String, example = json!({"status": "ok"}))
        )
    )]
    pub fn health() {}

    /// Create a chat completion.
    #[utoipa::path(
        post,
        path = "/v1/chat/completions",
        request_body = ChatCompletionRequest,
        responses(
            (status = 200, description = "Chat completion response", body = ChatCompletionResponse),
            (status = 400, description = "Invalid request", body = ErrorResponse),
            (status = 502, description = "Upstream provider error", body = ErrorResponse)
        )
    )]
    pub fn chat_completions() {}
}
