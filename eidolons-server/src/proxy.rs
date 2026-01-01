//! HTTP client for proxying requests to Anthropic.

use futures_util::StreamExt;
use reqwest::Client;
use tokio::sync::mpsc;

use crate::anthropic::{ErrorResponse, MessagesRequest, MessagesResponse, StreamEvent};

/// The Anthropic API base URL.
const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";

/// The Anthropic API version header value.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic API client.
pub struct AnthropicClient {
    client: Client,
    api_key: String,
}

impl AnthropicClient {
    /// Create a new Anthropic client with the given API key.
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("failed to build HTTP client");

        Self { client, api_key }
    }

    /// Send a non-streaming request to Anthropic.
    pub async fn send(&self, request: &MessagesRequest) -> Result<MessagesResponse, ProxyError> {
        let response = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(request)
            .send()
            .await
            .map_err(|e| ProxyError::Network(e.to_string()))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ProxyError::Network(e.to_string()))?;

        if status.is_success() {
            serde_json::from_slice(&body).map_err(|e| {
                tracing::error!("Failed to parse Anthropic response: {}", e);
                tracing::debug!("Response body: {}", String::from_utf8_lossy(&body));
                ProxyError::Parse(e.to_string())
            })
        } else {
            // Try to parse as Anthropic error response
            if let Ok(error) = serde_json::from_slice::<ErrorResponse>(&body) {
                Err(ProxyError::Upstream {
                    status: status.as_u16(),
                    error_type: error.error.error_type,
                    message: error.error.message,
                })
            } else {
                Err(ProxyError::Upstream {
                    status: status.as_u16(),
                    error_type: "unknown".to_string(),
                    message: String::from_utf8_lossy(&body).to_string(),
                })
            }
        }
    }

    /// Send a streaming request to Anthropic.
    /// Returns a receiver that yields parsed stream events.
    pub async fn send_stream(
        &self,
        request: &MessagesRequest,
    ) -> Result<mpsc::Receiver<Result<StreamEvent, ProxyError>>, ProxyError> {
        let response = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(request)
            .send()
            .await
            .map_err(|e| ProxyError::Network(e.to_string()))?;

        let status = response.status();

        if !status.is_success() {
            let body = response
                .bytes()
                .await
                .map_err(|e| ProxyError::Network(e.to_string()))?;

            if let Ok(error) = serde_json::from_slice::<ErrorResponse>(&body) {
                return Err(ProxyError::Upstream {
                    status: status.as_u16(),
                    error_type: error.error.error_type,
                    message: error.error.message,
                });
            } else {
                return Err(ProxyError::Upstream {
                    status: status.as_u16(),
                    error_type: "unknown".to_string(),
                    message: String::from_utf8_lossy(&body).to_string(),
                });
            }
        }

        // Create channel for streaming events
        let (tx, rx) = mpsc::channel(32);

        // Spawn task to process the stream
        let stream = response.bytes_stream();
        tokio::spawn(async move {
            let mut stream = std::pin::pin!(stream);
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        buffer.push_str(&String::from_utf8_lossy(&chunk));

                        // Process complete SSE events from the buffer
                        while let Some(event) = extract_sse_event(&mut buffer) {
                            if let Some(parsed) = parse_sse_event(&event)
                                && tx.send(Ok(parsed)).await.is_err()
                            {
                                return; // Receiver dropped
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(ProxyError::Network(e.to_string()))).await;
                        return;
                    }
                }
            }
        });

        Ok(rx)
    }
}

/// Extract a complete SSE event from the buffer, if available.
fn extract_sse_event(buffer: &mut String) -> Option<String> {
    // SSE events are separated by double newlines
    if let Some(pos) = buffer.find("\n\n") {
        let event = buffer[..pos].to_string();
        *buffer = buffer[pos + 2..].to_string();
        Some(event)
    } else {
        None
    }
}

/// Parse an SSE event string into a StreamEvent.
fn parse_sse_event(event: &str) -> Option<StreamEvent> {
    let mut data: Option<&str> = None;

    for line in event.lines() {
        if let Some(value) = line.strip_prefix("data: ") {
            data = Some(value);
        }
    }

    let data = data?;

    // Parse the JSON data
    match serde_json::from_str::<StreamEvent>(data) {
        Ok(event) => Some(event),
        Err(e) => {
            tracing::warn!("Failed to parse SSE event: {} - data: {}", e, data);
            None
        }
    }
}

/// Errors from the proxy.
#[derive(Debug)]
pub enum ProxyError {
    /// Network error communicating with upstream.
    Network(String),

    /// Error parsing response.
    Parse(String),

    /// Error from the upstream API.
    Upstream {
        status: u16,
        error_type: String,
        message: String,
    },
}

impl std::fmt::Display for ProxyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProxyError::Network(msg) => write!(f, "network error: {}", msg),
            ProxyError::Parse(msg) => write!(f, "parse error: {}", msg),
            ProxyError::Upstream {
                status,
                error_type,
                message,
            } => write!(
                f,
                "upstream error ({}): {}: {}",
                status, error_type, message
            ),
        }
    }
}

impl std::error::Error for ProxyError {}
