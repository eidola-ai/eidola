//! Generate OpenAPI specification for the Eidolons API.
//!
//! This binary outputs the OpenAPI JSON specification to stdout.
//! It is used by the build system to generate the committed openapi.json file.

use eidolons_server::openai::{
    AssistantMessage, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice,
    ChunkChoice, ChunkDelta, ContentPart, ErrorDetail, ErrorResponse, FinishReason, ImageUrl,
    Message, MessageContent, Role, StopSequence, Usage,
};
use utoipa::OpenApi;

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
struct ApiDoc;

// Dummy functions for utoipa path documentation.
#[allow(dead_code)]
mod openapi_paths {
    use super::*;

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

fn main() {
    let spec = ApiDoc::openapi()
        .to_pretty_json()
        .expect("Failed to serialize OpenAPI spec");
    println!("{}", spec);
}
