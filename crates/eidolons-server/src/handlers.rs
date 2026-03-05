//! Top-level HTTP handlers: health, models, chat completions.

use std::convert::Infallible;

use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::response::Sse;
use axum::response::sse::{Event, KeepAlive};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::error;

use crate::AppState;
use crate::auth::{self, TokenAuth};
use crate::backend::{BackendStreamEvent, ChatBackend};
use crate::error::ServerError;
use crate::response::{
    EidolonsResponse, EidolonsStreamMetadata, build_privacy_metadata, build_verification_metadata,
};
use crate::types::{ChatCompletionRequest, ErrorResponse, ModelsResponse};

/// Health check endpoint.
#[utoipa::path(
    get,
    path = "/health",
    tag = "Public",
    responses(
        (status = 200, description = "Server is healthy", body = String, example = json!({"status": "ok"}))
    )
)]
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

/// List available models.
#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "Public",
    responses(
        (status = 200, description = "List of available models", body = ModelsResponse),
        (status = 502, description = "Upstream provider error", body = ErrorResponse)
    )
)]
pub async fn list_models(
    State(state): State<AppState>,
) -> Result<Json<ModelsResponse>, ServerError> {
    let models = state.backend.list_models().await.map_err(|e| {
        error!("Failed to list models: {}", e);
        e
    })?;
    Ok(Json(models))
}

/// Create a chat completion.
///
/// Proxies the request to the configured backend and returns a response
/// enriched with privacy and verification metadata.
#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    tag = "Unlinked",
    request_body = ChatCompletionRequest,
    responses(
        (status = 200, description = "Chat completion response with privacy and verification metadata", body = EidolonsResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 401, description = "Authentication failed", body = ErrorResponse),
        (status = 502, description = "Upstream provider error", body = ErrorResponse)
    )
)]
pub async fn chat_completions(
    TokenAuth(auth_context): TokenAuth,
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ServerError> {
    if request.stream {
        handle_streaming_request(state, &request, &auth_context).await
    } else {
        handle_non_streaming_request(&state, &request, &auth_context).await
    }
}

/// Handle a non-streaming chat completion request.
async fn handle_non_streaming_request(
    state: &AppState,
    request: &ChatCompletionRequest,
    auth_context: &auth::AuthContext,
) -> Result<axum::response::Response, ServerError> {
    let backend_response = state.backend.send(request).await.map_err(|e| {
        error!("Backend error: {}", e);
        e
    })?;

    let meta = &backend_response.meta;
    let is_tee = meta.tee_type.is_some();

    let backend_attestation = if is_tee {
        if let Some(chat_id) = &meta.chat_id {
            state
                .attestation
                .fetch_signature(chat_id, &meta.backend_model)
                .await
        } else {
            None
        }
    } else {
        None
    };

    let privacy = build_privacy_metadata(auth_context, is_tee, &meta.provider);
    let verification = build_verification_metadata(backend_attestation);

    let eidolons_response =
        EidolonsResponse::from_completion(backend_response.response, privacy, verification);

    Ok(Json(eidolons_response).into_response())
}

/// Handle a streaming chat completion request.
async fn handle_streaming_request(
    state: AppState,
    request: &ChatCompletionRequest,
    auth_context: &auth::AuthContext,
) -> Result<axum::response::Response, ServerError> {
    let mut upstream_rx = state.backend.send_stream(request).await.map_err(|e| {
        error!("Failed to start stream: {}", e);
        e
    })?;

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    let auth_context = auth_context.clone();

    tokio::spawn(async move {
        while let Some(event_result) = upstream_rx.recv().await {
            match event_result {
                Ok(BackendStreamEvent::Chunk(chunk)) => {
                    let json_str = serde_json::to_string(&chunk).unwrap();
                    let event = Event::default().data(json_str);
                    if tx.send(Ok(event)).await.is_err() {
                        return; // Client disconnected
                    }
                }
                Ok(BackendStreamEvent::Done(meta)) => {
                    let is_tee = meta.tee_type.is_some();

                    let backend_attestation = if is_tee {
                        if let Some(chat_id) = &meta.chat_id {
                            state
                                .attestation
                                .fetch_signature(chat_id, &meta.backend_model)
                                .await
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    let privacy = build_privacy_metadata(&auth_context, is_tee, &meta.provider);
                    let verification = build_verification_metadata(backend_attestation);

                    let stream_meta = EidolonsStreamMetadata::new(
                        meta.chat_id.unwrap_or_default(),
                        privacy,
                        verification,
                    );

                    // Send metadata event
                    let json_str = serde_json::to_string(&stream_meta).unwrap();
                    let event = Event::default().data(json_str);
                    let _ = tx.send(Ok(event)).await;

                    // Send [DONE]
                    let done_event = Event::default().data("[DONE]");
                    let _ = tx.send(Ok(done_event)).await;
                    return;
                }
                Err(e) => {
                    error!("Stream error: {}", e);
                    return;
                }
            }
        }
    });

    let stream = ReceiverStream::new(rx);
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}
