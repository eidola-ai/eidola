//! Eidolons Server - An OpenAI-compatible proxy for AI providers.
//!
//! This server accepts requests in OpenAI Chat Completions API format and
//! proxies them to configured upstream AI providers (currently Anthropic Claude).

use eidolons_server::{anthropic, openai, proxy, transform};
use utoipa::OpenApi;

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody, combinators::BoxBody};
use hyper::body::Frame;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info, warn};

use crate::openai::{
    AssistantMessage, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice,
    ChunkChoice, ChunkDelta, ContentPart, ErrorDetail, ErrorResponse, FinishReason, ImageUrl,
    Message, MessageContent, Role, StopSequence, Usage,
};
use crate::proxy::AnthropicClient;
use crate::transform::StreamTransformer;

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
// These are never called - they exist only to provide OpenAPI endpoint metadata.
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

/// Server configuration.
struct Config {
    /// Address to bind the server to.
    bind_addr: SocketAddr,

    /// Anthropic API key.
    anthropic_api_key: String,
}

impl Config {
    /// Load configuration from environment variables.
    fn from_env() -> Result<Self, String> {
        let bind_addr = std::env::var("BIND_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:8080".to_string())
            .parse()
            .map_err(|e| format!("invalid BIND_ADDR: {}", e))?;

        let anthropic_api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY environment variable is required")?;

        Ok(Config {
            bind_addr,
            anthropic_api_key,
        })
    }
}

/// Shared application state.
struct AppState {
    client: AnthropicClient,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Install the pure-Rust crypto provider for TLS (must be done before any TLS operations)
    rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider())
        .expect("failed to install rustls crypto provider");

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("eidolons_server=info".parse().unwrap())
                .add_directive("hyper=warn".parse().unwrap()),
        )
        .init();

    // Load configuration
    let config = Config::from_env().map_err(|e| {
        error!("Configuration error: {}", e);
        e
    })?;

    info!("Starting Eidolons server on {}", config.bind_addr);

    // Create shared state
    let state = Arc::new(AppState {
        client: AnthropicClient::new(config.anthropic_api_key),
    });

    // Bind TCP listener
    let listener = TcpListener::bind(config.bind_addr).await?;
    info!("Listening on http://{}", config.bind_addr);

    // Accept connections
    loop {
        let (stream, remote_addr) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let state = Arc::clone(&state);

        tokio::spawn(async move {
            let service = service_fn(move |req| {
                let state = Arc::clone(&state);
                async move { handle_request(req, state).await }
            });

            if let Err(e) = http1::Builder::new().serve_connection(io, service).await
                && !e.is_incomplete_message()
            {
                warn!("Connection error from {}: {}", remote_addr, e);
            }
        });
    }
}

/// Handle an incoming HTTP request.
async fn handle_request(
    req: Request<hyper::body::Incoming>,
    state: Arc<AppState>,
) -> Result<Response<BoxBody<Bytes, Infallible>>, Infallible> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    info!("{} {}", method, path);

    let response = match (method, path.as_str()) {
        // Health check endpoint
        (Method::GET, "/health") => json_response(StatusCode::OK, r#"{"status":"ok"}"#),

        // OpenAPI specification
        (Method::GET, "/openapi.json") => {
            let spec = ApiDoc::openapi()
                .to_json()
                .expect("OpenAPI spec serialization failed");
            json_response(StatusCode::OK, &spec)
        }

        // OpenAI-compatible chat completions endpoint
        (Method::POST, "/v1/chat/completions") => handle_chat_completions(req, state).await,

        // 404 for everything else
        _ => {
            let error = ErrorResponse::new("not found", "invalid_request_error");
            json_response(
                StatusCode::NOT_FOUND,
                &serde_json::to_string(&error).unwrap(),
            )
        }
    };

    Ok(response)
}

/// Handle the chat completions endpoint.
async fn handle_chat_completions(
    req: Request<hyper::body::Incoming>,
    state: Arc<AppState>,
) -> Response<BoxBody<Bytes, Infallible>> {
    // Read request body
    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            error!("Failed to read request body: {}", e);
            let error = ErrorResponse::new("failed to read request body", "invalid_request_error");
            return json_response(
                StatusCode::BAD_REQUEST,
                &serde_json::to_string(&error).unwrap(),
            );
        }
    };

    // Parse OpenAI request
    let openai_request: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(req) => req,
        Err(e) => {
            error!("Failed to parse request: {}", e);
            let error = ErrorResponse::new(
                format!("invalid request body: {}", e),
                "invalid_request_error",
            );
            return json_response(
                StatusCode::BAD_REQUEST,
                &serde_json::to_string(&error).unwrap(),
            );
        }
    };

    let is_streaming = openai_request.stream;
    let original_model = openai_request.model.clone();

    // Transform to Anthropic format
    let anthropic_request = match transform::openai_to_anthropic(openai_request) {
        Ok(req) => req,
        Err(e) => {
            error!("Failed to transform request: {}", e);
            let error = ErrorResponse::new(e.to_string(), "invalid_request_error");
            return json_response(
                StatusCode::BAD_REQUEST,
                &serde_json::to_string(&error).unwrap(),
            );
        }
    };

    if is_streaming {
        handle_streaming_request(&state, anthropic_request, &original_model).await
    } else {
        handle_non_streaming_request(&state, anthropic_request, &original_model).await
    }
}

/// Handle a non-streaming chat completion request.
async fn handle_non_streaming_request(
    state: &AppState,
    request: anthropic::MessagesRequest,
    original_model: &str,
) -> Response<BoxBody<Bytes, Infallible>> {
    match state.client.send(&request).await {
        Ok(response) => {
            let openai_response = transform::anthropic_to_openai(response, original_model);
            json_response(
                StatusCode::OK,
                &serde_json::to_string(&openai_response).unwrap(),
            )
        }
        Err(e) => {
            error!("Upstream error: {}", e);
            let (status, error) = match &e {
                proxy::ProxyError::Upstream {
                    status,
                    error_type,
                    message,
                } => {
                    let http_status =
                        StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY);
                    (http_status, ErrorResponse::new(message, error_type))
                }
                _ => (
                    StatusCode::BAD_GATEWAY,
                    ErrorResponse::new(e.to_string(), "upstream_error"),
                ),
            };
            json_response(status, &serde_json::to_string(&error).unwrap())
        }
    }
}

/// Handle a streaming chat completion request.
async fn handle_streaming_request(
    state: &AppState,
    request: anthropic::MessagesRequest,
    original_model: &str,
) -> Response<BoxBody<Bytes, Infallible>> {
    // Start streaming from Anthropic
    let mut upstream_rx = match state.client.send_stream(&request).await {
        Ok(rx) => rx,
        Err(e) => {
            error!("Failed to start stream: {}", e);
            let (status, error) = match &e {
                proxy::ProxyError::Upstream {
                    status,
                    error_type,
                    message,
                } => {
                    let http_status =
                        StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY);
                    (http_status, ErrorResponse::new(message, error_type))
                }
                _ => (
                    StatusCode::BAD_GATEWAY,
                    ErrorResponse::new(e.to_string(), "upstream_error"),
                ),
            };
            return json_response(status, &serde_json::to_string(&error).unwrap());
        }
    };

    // Create channel for transformed SSE output
    let (tx, rx) = mpsc::channel::<Result<Frame<Bytes>, Infallible>>(32);
    let model = original_model.to_string();

    // Spawn task to transform and forward events
    tokio::spawn(async move {
        let mut transformer = StreamTransformer::new(model);

        while let Some(event_result) = upstream_rx.recv().await {
            match event_result {
                Ok(event) => {
                    if let Some(chunk) = transformer.transform(event) {
                        let sse_data =
                            format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());
                        if tx
                            .send(Ok(Frame::data(Bytes::from(sse_data))))
                            .await
                            .is_err()
                        {
                            break; // Client disconnected
                        }
                    }
                }
                Err(e) => {
                    error!("Stream error: {}", e);
                    break;
                }
            }
        }

        // Send final [DONE] marker
        let _ = tx
            .send(Ok(Frame::data(Bytes::from("data: [DONE]\n\n"))))
            .await;
    });

    // Create streaming response
    let stream = ReceiverStream::new(rx);
    let body = StreamBody::new(stream);

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .body(BodyExt::boxed(body))
        .unwrap()
}

/// Create a JSON response with the given status and body.
fn json_response(status: StatusCode, body: &str) -> Response<BoxBody<Bytes, Infallible>> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body.to_string())).boxed())
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openapi_spec_generation() {
        let spec = ApiDoc::openapi();

        // Verify basic info
        assert_eq!(spec.info.title, "Eidolons API");
        assert_eq!(spec.info.version, "0.1.0");

        // Verify paths exist
        assert!(
            spec.paths.paths.contains_key("/health"),
            "missing /health path"
        );
        assert!(
            spec.paths.paths.contains_key("/v1/chat/completions"),
            "missing /v1/chat/completions path"
        );

        // Verify schemas exist
        let schemas = spec.components.as_ref().unwrap();
        assert!(
            schemas.schemas.contains_key("ChatCompletionRequest"),
            "missing ChatCompletionRequest schema"
        );
        assert!(
            schemas.schemas.contains_key("ChatCompletionResponse"),
            "missing ChatCompletionResponse schema"
        );

        // Verify JSON serialization works
        let json = spec.to_json().expect("failed to serialize to JSON");
        assert!(json.contains("Eidolons API"));
        assert!(json.contains("/v1/chat/completions"));
    }
}
