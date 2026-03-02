//! Eidolons Server - A privacy-transparent AI proxy.
//!
//! This server accepts requests in OpenAI Chat Completions API format and
//! proxies them to upstream AI providers via RedPill.ai, enriching responses
//! with inline privacy metadata and cryptographic verification.

use eidolons_server::api_doc::ApiDoc;
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

use eidolons_server::attestation::AttestationClient;
use eidolons_server::auth::{self, AnyValidator, NoopValidator};
use eidolons_server::backend::{BackendStreamEvent, ChatBackend, RedPillBackend};
use eidolons_server::error::ServerError;
use eidolons_server::response::{
    EidolonsResponse, EidolonsStreamMetadata, SSE_DONE, build_privacy_metadata,
    build_verification_metadata, sse_data,
};
use eidolons_server::types::{ChatCompletionRequest, ErrorResponse};

/// Server configuration.
struct Config {
    /// Address to bind the server to.
    bind_addr: SocketAddr,

    /// RedPill API key (used for both chat completions and attestation).
    redpill_api_key: String,

    /// RedPill base URL (default: https://api.redpill.ai/v1).
    redpill_base_url: Option<String>,

    /// Authorization mode: "privacy_pass", "none" (default: "none" for development).
    auth_mode: String,
}

impl Config {
    /// Load configuration from environment variables.
    fn from_env() -> Result<Self, String> {
        let bind_addr = std::env::var("BIND_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:8080".to_string())
            .parse()
            .map_err(|e| format!("invalid BIND_ADDR: {}", e))?;

        let redpill_api_key = std::env::var("REDPILL_API_KEY")
            .map_err(|_| "REDPILL_API_KEY environment variable is required")?;

        let redpill_base_url = std::env::var("REDPILL_BASE_URL").ok();

        let auth_mode = std::env::var("AUTH_MODE").unwrap_or_else(|_| "none".to_string());

        Ok(Config {
            bind_addr,
            redpill_api_key,
            redpill_base_url,
            auth_mode,
        })
    }
}

/// Shared application state.
struct AppState {
    backend: RedPillBackend,
    validator: AnyValidator,
    attestation: AttestationClient,
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
    info!("Auth mode: {}", config.auth_mode);

    // Build the token validator
    let validator = match config.auth_mode.as_str() {
        "none" => {
            warn!("Running with no authentication (development mode)");
            AnyValidator::Noop(NoopValidator)
        }
        other => {
            return Err(format!("unknown AUTH_MODE: {} (expected: none)", other).into());
        }
    };

    // Create shared state
    let state = Arc::new(AppState {
        backend: RedPillBackend::new(
            config.redpill_api_key.clone(),
            config.redpill_base_url.clone(),
        ),
        validator,
        attestation: AttestationClient::new(config.redpill_api_key, config.redpill_base_url),
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

        // List available models
        (Method::GET, "/v1/models") => handle_list_models(state).await,

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

/// Handle the list models endpoint.
async fn handle_list_models(state: Arc<AppState>) -> Response<BoxBody<Bytes, Infallible>> {
    match state.backend.list_models().await {
        Ok(models) => json_response(StatusCode::OK, &serde_json::to_string(&models).unwrap()),
        Err(e) => {
            error!("Failed to list models: {}", e);
            error_response(&e)
        }
    }
}

/// Handle the chat completions endpoint.
async fn handle_chat_completions(
    req: Request<hyper::body::Incoming>,
    state: Arc<AppState>,
) -> Response<BoxBody<Bytes, Infallible>> {
    // Authenticate the request
    let auth_context = match auth::authenticate(&req, &state.validator).await {
        Ok(ctx) => ctx,
        Err(e) => return error_response(&e),
    };

    // Read request body
    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            error!("Failed to read request body: {}", e);
            return error_response(&ServerError::BadRequest {
                message: "failed to read request body".to_string(),
            });
        }
    };

    // Parse OpenAI request
    let request: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(req) => req,
        Err(e) => {
            return error_response(&ServerError::BadRequest {
                message: format!("invalid request body: {}", e),
            });
        }
    };

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
) -> Response<BoxBody<Bytes, Infallible>> {
    let backend_response = match state.backend.send(request).await {
        Ok(resp) => resp,
        Err(e) => {
            error!("Backend error: {}", e);
            return error_response(&e);
        }
    };

    let meta = &backend_response.meta;
    let is_tee = meta.tee_type.is_some();

    // Fetch attestation for TEE models (best-effort)
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

    json_response(
        StatusCode::OK,
        &serde_json::to_string(&eidolons_response).unwrap(),
    )
}

/// Handle a streaming chat completion request.
async fn handle_streaming_request(
    state: Arc<AppState>,
    request: &ChatCompletionRequest,
    auth_context: &auth::AuthContext,
) -> Response<BoxBody<Bytes, Infallible>> {
    let mut upstream_rx = match state.backend.send_stream(request).await {
        Ok(rx) => rx,
        Err(e) => {
            error!("Failed to start stream: {}", e);
            return error_response(&e);
        }
    };

    // Create channel for SSE output
    let (tx, rx) = mpsc::channel::<Result<Frame<Bytes>, Infallible>>(32);
    let auth_context = auth_context.clone();

    tokio::spawn(async move {
        while let Some(event_result) = upstream_rx.recv().await {
            match event_result {
                Ok(BackendStreamEvent::Chunk(chunk)) => {
                    let data = sse_data(&chunk);
                    if tx.send(Ok(Frame::data(Bytes::from(data)))).await.is_err() {
                        return; // Client disconnected
                    }
                }
                Ok(BackendStreamEvent::Done(meta)) => {
                    let is_tee = meta.tee_type.is_some();

                    // Fetch attestation for TEE models (best-effort)
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
                    let data = sse_data(&stream_meta);
                    let _ = tx.send(Ok(Frame::data(Bytes::from(data)))).await;

                    // Send [DONE]
                    let _ = tx
                        .send(Ok(Frame::data(Bytes::from(SSE_DONE.to_string()))))
                        .await;
                    return;
                }
                Err(e) => {
                    error!("Stream error: {}", e);
                    return;
                }
            }
        }
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

/// Convert a `ServerError` into an HTTP response.
fn error_response(err: &ServerError) -> Response<BoxBody<Bytes, Infallible>> {
    let status = err.status_code();
    let body = err.to_error_response();
    json_response(status, &serde_json::to_string(&body).unwrap())
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
            spec.paths.paths.contains_key("/v1/models"),
            "missing /v1/models path"
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

        // Verify JSON serialization works
        let json = spec.to_json().expect("failed to serialize to JSON");
        assert!(json.contains("Eidolons API"));
        assert!(json.contains("/v1/chat/completions"));
    }
}
