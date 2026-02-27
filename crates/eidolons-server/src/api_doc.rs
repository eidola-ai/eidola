//! OpenAPI documentation for the Eidolons API.

use utoipa::OpenApi;

use crate::auth::AuthMethod;
use crate::backend::TeeType;
use crate::response::{
    AttestationStatus, AuthorizationInfo, BackendAttestation, DataExposure, EidolonsResponse,
    EidolonsStreamMetadata, PrivacyMetadata, ProxyAttestation, TransportInfo, VerificationMetadata,
};
use crate::types::{
    AssistantMessage, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice,
    ChunkChoice, ChunkDelta, ContentPart, ErrorDetail, ErrorResponse, FinishReason, ImageUrl,
    Message, MessageContent, Model, ModelsResponse, Role, StopSequence, Usage,
};

/// OpenAPI documentation for the Eidolons API.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "Eidolons API",
        description = "Privacy-transparent AI proxy API with inline verification metadata",
        version = "0.1.0"
    ),
    paths(openapi_paths::health, openapi_paths::list_models, openapi_paths::chat_completions),
    components(schemas(
        // Request types
        ChatCompletionRequest,
        Message,
        Role,
        MessageContent,
        ContentPart,
        ImageUrl,
        StopSequence,
        // OpenAI response types (internal, used by EidolonsResponse)
        ChatCompletionResponse,
        ChatCompletionChunk,
        Choice,
        ChunkChoice,
        ChunkDelta,
        AssistantMessage,
        FinishReason,
        Usage,
        // Eidolons response types
        EidolonsResponse,
        EidolonsStreamMetadata,
        // Privacy metadata
        PrivacyMetadata,
        AuthorizationInfo,
        AuthMethod,
        DataExposure,
        TransportInfo,
        // Verification metadata
        VerificationMetadata,
        ProxyAttestation,
        AttestationStatus,
        BackendAttestation,
        TeeType,
        // Model listing types
        ModelsResponse,
        Model,
        // Error types
        ErrorResponse,
        ErrorDetail,
    ))
)]
pub struct ApiDoc;

// Dummy functions for utoipa path documentation.
// These are never called - they exist only to provide OpenAPI endpoint metadata.
#[allow(dead_code)]
mod openapi_paths {
    use crate::response::EidolonsResponse;
    use crate::types::{ChatCompletionRequest, ErrorResponse, ModelsResponse};

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
    ///
    /// Proxies the request to the configured backend and returns a response
    /// enriched with privacy and verification metadata.
    #[utoipa::path(
        post,
        path = "/v1/chat/completions",
        request_body = ChatCompletionRequest,
        responses(
            (status = 200, description = "Chat completion response with privacy and verification metadata", body = EidolonsResponse),
            (status = 400, description = "Invalid request", body = ErrorResponse),
            (status = 401, description = "Authentication failed", body = ErrorResponse),
            (status = 502, description = "Upstream provider error", body = ErrorResponse)
        )
    )]
    pub fn chat_completions() {}

    /// List available models.
    #[utoipa::path(
        get,
        path = "/v1/models",
        responses(
            (status = 200, description = "List of available models", body = ModelsResponse),
            (status = 502, description = "Upstream provider error", body = ErrorResponse)
        )
    )]
    pub fn list_models() {}
}
