//! Chat completion backend trait and RedPill.ai implementation.

use std::time::Instant;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, mpsc};
use utoipa::ToSchema;

use crate::error::ServerError;
use crate::types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Model, ModelPricing,
    ModelsResponse, ScaledPrice,
};

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// The type of Trusted Execution Environment backing a model.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum TeeType {
    /// Intel TDX + NVIDIA H100 Confidential Computing (Phala-hosted via RedPill).
    PhalaGpu,
}

/// Metadata about a backend's execution of a request.
#[derive(Debug, Clone)]
pub struct BackendMeta {
    /// The backend provider name (e.g., "redpill").
    pub provider: String,

    /// The chat completion ID returned by the backend (for attestation lookup).
    pub chat_id: Option<String>,

    /// The actual model name used by the backend.
    pub backend_model: String,

    /// Whether this model runs inside a TEE.
    pub tee_type: Option<TeeType>,

    /// Token usage statistics (from the response or final streaming chunk).
    pub usage: Option<crate::types::Usage>,
}

/// A completed (non-streaming) backend response.
pub struct BackendResponse {
    /// The OpenAI-format completion response.
    pub response: ChatCompletionResponse,

    /// Metadata about this execution.
    pub meta: BackendMeta,
}

/// Events emitted by a streaming backend.
pub enum BackendStreamEvent {
    /// A content chunk (standard OpenAI format).
    Chunk(ChatCompletionChunk),

    /// The stream has completed. Carries final metadata.
    Done(BackendMeta),
}

// ---------------------------------------------------------------------------
// ChatBackend trait
// ---------------------------------------------------------------------------

/// Trait for chat completion backends.
///
/// Uses RPITIT (stable since Rust 1.75) instead of `async_trait`.
pub trait ChatBackend: Send + Sync {
    /// Send a non-streaming chat completion request.
    fn send(
        &self,
        request: &ChatCompletionRequest,
    ) -> impl std::future::Future<Output = Result<BackendResponse, ServerError>> + Send;

    /// List available models.
    fn list_models(
        &self,
    ) -> impl std::future::Future<Output = Result<ModelsResponse, ServerError>> + Send;

    /// Send a streaming chat completion request.
    ///
    /// Returns a receiver that yields `Chunk` events followed by a final `Done` event.
    fn send_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> impl std::future::Future<
        Output = Result<mpsc::Receiver<Result<BackendStreamEvent, ServerError>>, ServerError>,
    > + Send;
}

// ---------------------------------------------------------------------------
// RedPill.ai backend
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// RedPill upstream response types (internal)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RedPillModelsResponse {
    data: Vec<RedPillModel>,
}

#[derive(Debug, Deserialize)]
struct RedPillModel {
    id: String,
    name: String,
    #[allow(dead_code)]
    created: u64,
    description: String,
    context_length: u64,
    pricing: RedPillPricing,
}

#[derive(Debug, Deserialize)]
struct RedPillPricing {
    prompt: String,
    completion: String,
    #[serde(flatten)]
    other: std::collections::HashMap<String, String>,
}

/// Default pricing markup factor.
pub const DEFAULT_PRICING_MARKUP: f64 = 1.5;

/// Fixed scale factor for pricing: credits per token = value / PRICING_SCALE_FACTOR.
pub const PRICING_SCALE_FACTOR: u64 = 1_000_000;

/// Convert a USD-per-token price string to scaled integer credits,
/// applying a markup factor and rounding up.
fn usd_to_scaled_credits(usd_per_token: &str, markup: f64) -> u64 {
    let usd: f64 = usd_per_token.parse().unwrap_or(0.0);
    // credits/token = usd * 1e6 (USD→µ$) * markup
    // scaled value  = credits/token * scale_factor
    (usd * 1e6 * markup * PRICING_SCALE_FACTOR as f64).ceil() as u64
}

impl RedPillModel {
    /// Returns true if all pricing fields beyond prompt/completion are "0" (or absent).
    fn has_only_known_pricing(&self) -> bool {
        self.pricing
            .other
            .values()
            .all(|v| v == "0" || v.is_empty())
    }

    fn into_model(self, markup: f64) -> Model {
        let p = &self.pricing;
        Model {
            id: self.id,
            name: self.name,
            description: self.description,
            context_length: self.context_length,
            pricing: ModelPricing {
                per_prompt_token: ScaledPrice {
                    value: usd_to_scaled_credits(&p.prompt, markup),
                    scale_factor: PRICING_SCALE_FACTOR,
                },
                per_completion_token: ScaledPrice {
                    value: usd_to_scaled_credits(&p.completion, markup),
                    scale_factor: PRICING_SCALE_FACTOR,
                },
            },
        }
    }
}

// ---------------------------------------------------------------------------
// RedPill.ai backend
// ---------------------------------------------------------------------------

/// RedPill.ai backend.
///
/// Sends OpenAI-format requests to RedPill's API, which supports both
/// frontier providers (OpenAI, Anthropic, etc.) and confidential Phala GPU
/// TEE models (phala/* prefix).
/// How long to cache the model list before refreshing.
const MODEL_CACHE_TTL_SECS: u64 = 300; // 5 minutes

pub struct RedPillBackend {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    pricing_markup: f64,
    models_cache: RwLock<Option<(Instant, ModelsResponse)>>,
}

impl RedPillBackend {
    pub fn new(api_key: String, base_url: Option<String>, pricing_markup: Option<f64>) -> Self {
        let client = reqwest::Client::builder()
            .build()
            .expect("failed to build HTTP client");

        Self {
            client,
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.redpill.ai/v1".to_string()),
            pricing_markup: pricing_markup.unwrap_or(DEFAULT_PRICING_MARKUP),
            models_cache: RwLock::new(None),
        }
    }

    /// Determine if a model runs inside a TEE based on its name.
    fn tee_type_for_model(model: &str) -> Option<TeeType> {
        if model.starts_with("phala/") {
            Some(TeeType::PhalaGpu)
        } else {
            None
        }
    }
}

impl RedPillBackend {
    /// Look up a model by ID, using a cached model list.
    pub async fn lookup_model(&self, model_id: &str) -> Result<Option<Model>, ServerError> {
        // Check cache
        {
            let cache = self.models_cache.read().await;
            if let Some((fetched_at, ref models)) = *cache
                && fetched_at.elapsed().as_secs() < MODEL_CACHE_TTL_SECS
            {
                return Ok(models.data.iter().find(|m| m.id == model_id).cloned());
            }
        }

        // Cache miss or stale — refresh
        let models = ChatBackend::list_models(self).await?;
        let result = models.data.iter().find(|m| m.id == model_id).cloned();
        *self.models_cache.write().await = Some((Instant::now(), models));
        Ok(result)
    }
}

impl ChatBackend for RedPillBackend {
    async fn list_models(&self) -> Result<ModelsResponse, ServerError> {
        let url = format!("{}/models", self.base_url);

        let response = self
            .client
            .get(&url)
            .header("authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .map_err(|e| ServerError::Network(e.to_string()))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(e.to_string()))?;

        if !status.is_success() {
            if let Ok(err) = serde_json::from_slice::<crate::types::ErrorResponse>(&body) {
                return Err(ServerError::Backend {
                    status: status.as_u16(),
                    error_type: err.error.error_type,
                    message: err.error.message,
                });
            }
            return Err(ServerError::Backend {
                status: status.as_u16(),
                error_type: "unknown".to_string(),
                message: String::from_utf8_lossy(&body).to_string(),
            });
        }

        let upstream: RedPillModelsResponse = serde_json::from_slice(&body).map_err(|e| {
            tracing::error!("Failed to parse models response: {}", e);
            ServerError::Parse(e.to_string())
        })?;

        let markup = self.pricing_markup;
        Ok(ModelsResponse {
            data: upstream
                .data
                .into_iter()
                .filter(|m| m.has_only_known_pricing())
                .map(|m| m.into_model(markup))
                .collect(),
        })
    }

    async fn send(&self, request: &ChatCompletionRequest) -> Result<BackendResponse, ServerError> {
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(request)
            .send()
            .await
            .map_err(|e| ServerError::Network(e.to_string()))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(e.to_string()))?;

        if !status.is_success() {
            if let Ok(err) = serde_json::from_slice::<crate::types::ErrorResponse>(&body) {
                return Err(ServerError::Backend {
                    status: status.as_u16(),
                    error_type: err.error.error_type,
                    message: err.error.message,
                });
            }
            return Err(ServerError::Backend {
                status: status.as_u16(),
                error_type: "unknown".to_string(),
                message: String::from_utf8_lossy(&body).to_string(),
            });
        }

        let completion: ChatCompletionResponse = serde_json::from_slice(&body).map_err(|e| {
            tracing::error!("Failed to parse backend response: {}", e);
            tracing::debug!("Response body: {}", String::from_utf8_lossy(&body));
            ServerError::Parse(e.to_string())
        })?;

        let meta = BackendMeta {
            provider: "redpill".to_string(),
            chat_id: Some(completion.id.clone()),
            backend_model: completion.model.clone(),
            tee_type: Self::tee_type_for_model(&request.model),
            usage: completion.usage.clone(),
        };

        Ok(BackendResponse {
            response: completion,
            meta,
        })
    }

    async fn send_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<mpsc::Receiver<Result<BackendStreamEvent, ServerError>>, ServerError> {
        let url = format!("{}/chat/completions", self.base_url);

        // Ensure stream=true in the forwarded request
        let mut stream_request = request.clone();
        stream_request.stream = true;

        let response = self
            .client
            .post(&url)
            .header("authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&stream_request)
            .send()
            .await
            .map_err(|e| ServerError::Network(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body = response
                .bytes()
                .await
                .map_err(|e| ServerError::Network(e.to_string()))?;

            if let Ok(err) = serde_json::from_slice::<crate::types::ErrorResponse>(&body) {
                return Err(ServerError::Backend {
                    status: status.as_u16(),
                    error_type: err.error.error_type,
                    message: err.error.message,
                });
            }
            return Err(ServerError::Backend {
                status: status.as_u16(),
                error_type: "unknown".to_string(),
                message: String::from_utf8_lossy(&body).to_string(),
            });
        }

        let (tx, rx) = mpsc::channel(32);
        let model = request.model.clone();
        let tee_type = Self::tee_type_for_model(&request.model);

        tokio::spawn(async move {
            let stream = response.bytes_stream();
            let mut stream = std::pin::pin!(stream);
            let mut buffer = String::new();
            let mut chat_id: Option<String> = None;
            let mut backend_model = model.clone();
            let mut final_usage: Option<crate::types::Usage> = None;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        buffer.push_str(&String::from_utf8_lossy(&chunk));

                        while let Some(data) = extract_sse_data(&mut buffer) {
                            if data == "[DONE]" {
                                // Stream complete — we'll send Done below
                                break;
                            }
                            match serde_json::from_str::<ChatCompletionChunk>(&data) {
                                Ok(chunk) => {
                                    if chat_id.is_none() {
                                        chat_id = Some(chunk.id.clone());
                                    }
                                    backend_model.clone_from(&chunk.model);
                                    if chunk.usage.is_some() {
                                        final_usage.clone_from(&chunk.usage);
                                    }
                                    if tx.send(Ok(BackendStreamEvent::Chunk(chunk))).await.is_err()
                                    {
                                        return; // Client disconnected
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "Failed to parse SSE chunk: {} - data: {}",
                                        e,
                                        data
                                    );
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(ServerError::Network(e.to_string()))).await;
                        return;
                    }
                }
            }

            // Send final Done event with metadata
            let meta = BackendMeta {
                provider: "redpill".to_string(),
                chat_id,
                backend_model,
                tee_type,
                usage: final_usage,
            };
            let _ = tx.send(Ok(BackendStreamEvent::Done(meta))).await;
        });

        Ok(rx)
    }
}

/// Extract the data payload from a single SSE event in the buffer.
fn extract_sse_data(buffer: &mut String) -> Option<String> {
    let pos = buffer.find("\n\n")?;
    let event_block = buffer[..pos].to_string();
    *buffer = buffer[pos + 2..].to_string();

    for line in event_block.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            return Some(data.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_only_known_pricing() {
        let model = |other: Vec<(&str, &str)>| RedPillModel {
            id: "test".into(),
            name: "test".into(),
            created: 0,
            description: "".into(),
            context_length: 0,
            pricing: RedPillPricing {
                prompt: "0.00000004".into(),
                completion: "0.00000015".into(),
                other: other
                    .into_iter()
                    .map(|(k, v)| (k.into(), v.into()))
                    .collect(),
            },
        };

        // All zeros — allowed
        assert!(model(vec![("image", "0"), ("request", "0")]).has_only_known_pricing());

        // Empty other — allowed
        assert!(model(vec![]).has_only_known_pricing());

        // Non-zero known field — filtered out
        assert!(!model(vec![("image", "0.001")]).has_only_known_pricing());

        // Non-zero unknown field — filtered out
        assert!(!model(vec![("mystery_fee", "0.005")]).has_only_known_pricing());
    }

    #[test]
    fn test_usd_to_scaled_credits() {
        // phala/gpt-oss-20b with 1.5x markup, scale_factor = 1_000_000
        // $0.00000004/token → 0.06 credits/token → 60_000 scaled
        assert_eq!(usd_to_scaled_credits("0.00000004", 1.5), 60_000);
        // $0.00000015/token → 0.225 credits/token → 225_000 scaled
        assert_eq!(usd_to_scaled_credits("0.00000015", 1.5), 225_000);

        // Zero price
        assert_eq!(usd_to_scaled_credits("0", 1.5), 0);

        // Ceil rounding: $1e-16/token * 1e6 * 1.5 * 1e6 = 0.15 → ceil → 1
        assert_eq!(usd_to_scaled_credits("0.0000000000000001", 1.5), 1);

        // Invalid input falls back to 0
        assert_eq!(usd_to_scaled_credits("", 1.5), 0);
        assert_eq!(usd_to_scaled_credits("not_a_number", 1.5), 0);
    }

    #[test]
    fn test_tee_type_for_model() {
        assert!(matches!(
            RedPillBackend::tee_type_for_model("phala/deepseek-v3"),
            Some(TeeType::PhalaGpu)
        ));
        assert!(RedPillBackend::tee_type_for_model("openai/gpt-4o").is_none());
        assert!(RedPillBackend::tee_type_for_model("anthropic/claude-sonnet-4").is_none());
    }

    #[test]
    fn test_extract_sse_data() {
        let mut buffer = "data: {\"id\":\"123\"}\n\ndata: [DONE]\n\n".to_string();

        let first = extract_sse_data(&mut buffer);
        assert_eq!(first, Some("{\"id\":\"123\"}".to_string()));

        let second = extract_sse_data(&mut buffer);
        assert_eq!(second, Some("[DONE]".to_string()));

        let third = extract_sse_data(&mut buffer);
        assert!(third.is_none());
    }

    #[test]
    fn test_extract_sse_data_with_event_type() {
        let mut buffer = "event: message\ndata: {\"hello\":true}\n\n".to_string();
        let data = extract_sse_data(&mut buffer);
        assert_eq!(data, Some("{\"hello\":true}".to_string()));
    }

    #[test]
    fn test_extract_sse_data_partial() {
        let mut buffer = "data: partial".to_string();
        let data = extract_sse_data(&mut buffer);
        assert!(data.is_none());
        assert_eq!(buffer, "data: partial"); // Buffer unchanged
    }
}
