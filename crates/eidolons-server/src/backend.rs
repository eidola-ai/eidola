//! Chat completion backend trait and Tinfoil implementation.
//!
//! All Tinfoil models run inside confidential enclaves (AMD SEV-SNP / Intel TDX
//! / NVIDIA CC). The model catalog is hardcoded — only pricing can be overridden
//! at runtime via `TINFOIL_PRICING_OVERRIDES`.

use std::collections::HashMap;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
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
    /// Tinfoil confidential enclave (AMD SEV-SNP / Intel TDX / NVIDIA CC).
    TinfoilEnclave,
}

/// Metadata about a backend's execution of a request.
#[derive(Debug, Clone)]
pub struct BackendMeta {
    /// The backend provider name (e.g., "tinfoil").
    pub provider: String,

    /// The chat completion ID returned by the backend.
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
// Static model catalog
// ---------------------------------------------------------------------------

/// Default pricing markup factor.
pub const DEFAULT_PRICING_MARKUP: f64 = 1.5;

/// Fixed scale factor for pricing: credits per token = value / PRICING_SCALE_FACTOR.
pub const PRICING_SCALE_FACTOR: u64 = 1_000_000;

/// A static model catalog entry with hardcoded pricing.
struct CatalogEntry {
    id: &'static str,
    name: &'static str,
    description: &'static str,
    context_length: u64,
    /// USD per million input tokens (0.0 for per-request models).
    input_per_m: f64,
    /// USD per million output tokens (0.0 for per-request models).
    output_per_m: f64,
    /// USD per request (0.0 for token-based models).
    per_request_usd: f64,
}

/// Hardcoded model catalog. Model identifiers and descriptions are bound to
/// the image contents; only pricing can be overridden at runtime via
/// `TINFOIL_PRICING_OVERRIDES`.
const MODEL_CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        id: "kimi-k2-5",
        name: "Kimi K2.5",
        description: "Unified vision and language model for visual inputs, design-to-code, and multi-agent orchestration",
        context_length: 256_000,
        input_per_m: 1.5,
        output_per_m: 5.25,
        per_request_usd: 0.0,
    },
    CatalogEntry {
        id: "deepseek-r1-0528",
        name: "DeepSeek R1",
        description: "Latest reasoning model with significantly enhanced depth and performance approaching top-tier models",
        context_length: 128_000,
        input_per_m: 1.5,
        output_per_m: 5.25,
        per_request_usd: 0.0,
    },
    CatalogEntry {
        id: "gpt-oss-120b",
        name: "GPT-OSS 120B",
        description: "Open-weight model designed for powerful reasoning, agentic tasks, and versatile use cases",
        context_length: 128_000,
        input_per_m: 0.75,
        output_per_m: 1.25,
        per_request_usd: 0.0,
    },
    CatalogEntry {
        id: "gpt-oss-safeguard-120b",
        name: "GPT-OSS Safeguard 120B",
        description: "Safety reasoning model for content classification and trust & safety applications",
        context_length: 128_000,
        input_per_m: 0.50,
        output_per_m: 1.0,
        per_request_usd: 0.0,
    },
    CatalogEntry {
        id: "qwen3-vl-30b",
        name: "Qwen3-VL 30B",
        description: "Advanced vision-language model for image understanding",
        context_length: 256_000,
        input_per_m: 1.25,
        output_per_m: 4.0,
        per_request_usd: 0.0,
    },
    CatalogEntry {
        id: "nomic-embed-text",
        name: "Nomic Embed Text",
        description: "Open-source text embedding model that outperforms OpenAI models on key benchmarks",
        context_length: 8_192,
        input_per_m: 0.05,
        output_per_m: 0.0,
        per_request_usd: 0.0,
    },
    CatalogEntry {
        id: "voxtral-small-24b",
        name: "Voxtral Small 24B",
        description: "Audio-capable model built on Mistral Small 3 for transcription, translation, and spoken queries",
        context_length: 32_000,
        input_per_m: 0.20,
        output_per_m: 0.60,
        per_request_usd: 0.0,
    },
    CatalogEntry {
        id: "whisper-large-v3-turbo",
        name: "Whisper Large V3 Turbo",
        description: "High-performance speech recognition and transcription model",
        context_length: 0,
        input_per_m: 0.0,
        output_per_m: 0.0,
        per_request_usd: 0.01,
    },
    CatalogEntry {
        id: "qwen3-tts",
        name: "Qwen3-TTS 1.7B",
        description: "Expressive text-to-speech model with voice cloning, voice design, and multilingual support",
        context_length: 0,
        input_per_m: 0.0,
        output_per_m: 0.0,
        per_request_usd: 0.01,
    },
    CatalogEntry {
        id: "llama3-3-70b",
        name: "Llama 3.3 70B",
        description: "High-performance multilingual language model optimized for speed",
        context_length: 128_000,
        input_per_m: 1.75,
        output_per_m: 2.75,
        per_request_usd: 0.0,
    },
];

/// Convert USD per million tokens to scaled integer credits, applying markup.
///
/// The 1e6 (USD→µ$) and /1e6 (per-M→per-token) factors cancel, leaving:
/// `scaled = usd_per_million * markup * PRICING_SCALE_FACTOR`
fn usd_per_m_to_scaled_credits(usd_per_million: f64, markup: f64) -> u64 {
    (usd_per_million * markup * PRICING_SCALE_FACTOR as f64).ceil() as u64
}

/// Convert USD per request to scaled integer credits, applying markup.
fn usd_per_req_to_scaled_credits(usd_per_request: f64, markup: f64) -> u64 {
    // credits/request = USD * 1e6 (USD→µ$) * markup
    // scaled = credits/request * PRICING_SCALE_FACTOR
    (usd_per_request * 1e6 * markup * PRICING_SCALE_FACTOR as f64).ceil() as u64
}

/// Runtime pricing override for a single model, parsed from `TINFOIL_PRICING_OVERRIDES`.
///
/// Example JSON: `{"kimi-k2-5": {"input": 1.5, "output": 5.25}}`
#[derive(Debug, Deserialize)]
struct PricingOverride {
    /// Override: USD per million input tokens.
    input: Option<f64>,
    /// Override: USD per million output tokens.
    output: Option<f64>,
    /// Override: USD per request.
    request: Option<f64>,
}

// ---------------------------------------------------------------------------
// Tinfoil backend
// ---------------------------------------------------------------------------

/// Tinfoil inference backend.
///
/// Sends OpenAI-format requests to Tinfoil's API. All Tinfoil models run
/// inside confidential enclaves with attestation-verified TLS.
pub struct TinfoilBackend {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    /// Static model list built from `MODEL_CATALOG` with optional pricing overrides.
    models: Vec<Model>,
}

impl TinfoilBackend {
    pub fn new(
        client: reqwest::Client,
        api_key: String,
        base_url: Option<String>,
        pricing_markup: Option<f64>,
    ) -> Self {
        let markup = pricing_markup.unwrap_or(DEFAULT_PRICING_MARKUP);
        let overrides = Self::parse_pricing_overrides();
        let models = Self::build_model_list(markup, &overrides);

        Self {
            client,
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://inference.tinfoil.sh/v1".to_string()),
            models,
        }
    }

    fn parse_pricing_overrides() -> HashMap<String, PricingOverride> {
        match std::env::var("TINFOIL_PRICING_OVERRIDES") {
            Ok(json) => match serde_json::from_str(&json) {
                Ok(overrides) => {
                    tracing::info!("Loaded pricing overrides from TINFOIL_PRICING_OVERRIDES");
                    overrides
                }
                Err(e) => {
                    tracing::warn!("Failed to parse TINFOIL_PRICING_OVERRIDES: {}", e);
                    HashMap::new()
                }
            },
            Err(_) => HashMap::new(),
        }
    }

    fn build_model_list(markup: f64, overrides: &HashMap<String, PricingOverride>) -> Vec<Model> {
        MODEL_CATALOG
            .iter()
            .map(|entry| {
                let ovr = overrides.get(entry.id);

                let pricing = if entry.per_request_usd > 0.0 {
                    let per_req = ovr.and_then(|o| o.request).unwrap_or(entry.per_request_usd);
                    ModelPricing {
                        per_prompt_token: ScaledPrice {
                            value: 0,
                            scale_factor: PRICING_SCALE_FACTOR,
                        },
                        per_completion_token: ScaledPrice {
                            value: 0,
                            scale_factor: PRICING_SCALE_FACTOR,
                        },
                        per_request: Some(ScaledPrice {
                            value: usd_per_req_to_scaled_credits(per_req, markup),
                            scale_factor: PRICING_SCALE_FACTOR,
                        }),
                    }
                } else {
                    let input = ovr.and_then(|o| o.input).unwrap_or(entry.input_per_m);
                    let output = ovr.and_then(|o| o.output).unwrap_or(entry.output_per_m);
                    ModelPricing {
                        per_prompt_token: ScaledPrice {
                            value: usd_per_m_to_scaled_credits(input, markup),
                            scale_factor: PRICING_SCALE_FACTOR,
                        },
                        per_completion_token: ScaledPrice {
                            value: usd_per_m_to_scaled_credits(output, markup),
                            scale_factor: PRICING_SCALE_FACTOR,
                        },
                        per_request: None,
                    }
                };

                Model {
                    id: entry.id.to_string(),
                    name: entry.name.to_string(),
                    description: entry.description.to_string(),
                    context_length: entry.context_length,
                    pricing,
                }
            })
            .collect()
    }

    /// Look up a model by ID from the static catalog.
    pub fn lookup_model(&self, model_id: &str) -> Option<Model> {
        self.models.iter().find(|m| m.id == model_id).cloned()
    }
}

impl ChatBackend for TinfoilBackend {
    async fn list_models(&self) -> Result<ModelsResponse, ServerError> {
        Ok(ModelsResponse {
            data: self.models.clone(),
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
            provider: "tinfoil".to_string(),
            chat_id: Some(completion.id.clone()),
            backend_model: completion.model.clone(),
            tee_type: Some(TeeType::TinfoilEnclave),
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
                provider: "tinfoil".to_string(),
                chat_id,
                backend_model,
                tee_type: Some(TeeType::TinfoilEnclave),
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
    fn test_usd_per_m_to_scaled_credits() {
        // kimi-k2-5 input: $1.5/M tokens with 1.5x markup
        // The 1e6 factors (USD→µ$ and /M→/token) cancel:
        // scaled = 1.5 * 1.5 * 1_000_000 = 2_250_000
        assert_eq!(usd_per_m_to_scaled_credits(1.5, 1.5), 2_250_000);

        // gpt-oss-120b input: $0.75/M with 1.5x markup
        assert_eq!(usd_per_m_to_scaled_credits(0.75, 1.5), 1_125_000);

        // Zero price
        assert_eq!(usd_per_m_to_scaled_credits(0.0, 1.5), 0);

        // nomic-embed-text input: $0.05/M with 1.5x markup
        // 0.05 * 1.5 = 0.075; ceil(0.075 * 1e6) = 75_000
        // (may be 75_001 due to f64 representation — ceil rounds up any epsilon)
        let nomic = usd_per_m_to_scaled_credits(0.05, 1.5);
        assert!(nomic == 75_000 || nomic == 75_001, "got {nomic}");
    }

    #[test]
    fn test_usd_per_req_to_scaled_credits() {
        // whisper: $0.01/req with 1.5x markup
        // credits/req = 0.01 * 1e6 * 1.5 = 15_000
        // scaled = 15_000 * 1e6 = 15_000_000_000
        assert_eq!(usd_per_req_to_scaled_credits(0.01, 1.5), 15_000_000_000);

        // Zero
        assert_eq!(usd_per_req_to_scaled_credits(0.0, 1.5), 0);
    }

    #[test]
    fn test_model_catalog_completeness() {
        let overrides = HashMap::new();
        let models = TinfoilBackend::build_model_list(1.5, &overrides);
        assert_eq!(models.len(), MODEL_CATALOG.len());

        // Verify all models have non-empty fields
        for model in &models {
            assert!(!model.id.is_empty());
            assert!(!model.name.is_empty());
            assert!(!model.description.is_empty());
        }
    }

    #[test]
    fn test_pricing_overrides() {
        let mut overrides = HashMap::new();
        overrides.insert(
            "kimi-k2-5".to_string(),
            PricingOverride {
                input: Some(2.0),
                output: Some(6.0),
                request: None,
            },
        );

        let models = TinfoilBackend::build_model_list(1.0, &overrides);
        let kimi = models.iter().find(|m| m.id == "kimi-k2-5").unwrap();

        // With 1.0x markup and $2.0/M input override
        assert_eq!(kimi.pricing.per_prompt_token.value, 2_000_000);
        assert_eq!(kimi.pricing.per_completion_token.value, 6_000_000);
        assert!(kimi.pricing.per_request.is_none());
    }

    #[test]
    fn test_per_request_model_pricing() {
        let overrides = HashMap::new();
        let models = TinfoilBackend::build_model_list(1.0, &overrides);

        let whisper = models
            .iter()
            .find(|m| m.id == "whisper-large-v3-turbo")
            .unwrap();
        assert!(whisper.pricing.per_request.is_some());
        assert_eq!(whisper.pricing.per_prompt_token.value, 0);
        assert_eq!(whisper.pricing.per_completion_token.value, 0);

        // $0.01 * 1e6 * 1.0 * 1e6 = 10_000_000_000
        assert_eq!(
            whisper.pricing.per_request.as_ref().unwrap().value,
            10_000_000_000
        );
    }

    #[test]
    fn test_per_request_override() {
        let mut overrides = HashMap::new();
        overrides.insert(
            "whisper-large-v3-turbo".to_string(),
            PricingOverride {
                input: None,
                output: None,
                request: Some(0.02),
            },
        );

        let models = TinfoilBackend::build_model_list(1.0, &overrides);
        let whisper = models
            .iter()
            .find(|m| m.id == "whisper-large-v3-turbo")
            .unwrap();

        // $0.02 * 1e6 * 1.0 * 1e6 = 20_000_000_000
        assert_eq!(
            whisper.pricing.per_request.as_ref().unwrap().value,
            20_000_000_000
        );
    }

    #[test]
    fn test_lookup_model() {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let backend = TinfoilBackend::new(reqwest::Client::new(), String::new(), None, Some(1.5));

        assert!(backend.lookup_model("kimi-k2-5").is_some());
        assert!(backend.lookup_model("nonexistent").is_none());
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
