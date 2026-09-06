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
    Capability, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Modality,
    Model, ModelCapabilities, ModelPricing, ModelsResponse, OutputBudgetClass, ScaledPrice,
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
///
/// Chunk and Done carry the [`std::time::Instant`] at which the upstream
/// reader took them off the socket. Timing must use that stamp, not the
/// receive time on the handler side: both the backend and downstream
/// channels are bounded, so a slow client backpressures the pipeline and
/// arrival at the handler measures delivery delay, not upstream
/// generation. (A persistently slow client eventually throttles the
/// socket itself via TCP flow control — that residual coupling is
/// inherent to bounded buffering and accepted.)
pub enum BackendStreamEvent {
    /// A content chunk (standard OpenAI format), stamped with its arrival
    /// off the upstream socket.
    Chunk(ChatCompletionChunk, std::time::Instant),

    /// The stream has completed. Carries final metadata, stamped with the
    /// moment the upstream stream ended.
    Done(BackendMeta, std::time::Instant),
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

/// Validate that the pricing markup covers the contract's safe cost factor.
///
/// The shared client/server pricing contract (`eidola-common`) charges the
/// prompt-side byte term at no less than `bytes / N`, where `N =
/// SAFE_COST_FACTOR_NUM / SAFE_COST_FACTOR_DEN` — and since BPE tokenizers
/// never produce more tokens than content bytes, the worst-case
/// actual/charged token ratio on that term is exactly `N`. Break-even on
/// dynamic costs therefore requires `PRICING_MARKUP >= N`; a markup below
/// the factor silently opens a loss window, so startup refuses it outright.
///
/// Future refinement (out of scope for now): when the configured markup
/// exceeds the factor, the server could publish a larger dynamic factor to
/// clients via the `/models` payload instead of both sides relying on the
/// compiled-in constants.
pub fn validate_pricing_markup(markup: f64) -> Result<(), String> {
    let floor =
        eidola_common::SAFE_COST_FACTOR_NUM as f64 / eidola_common::SAFE_COST_FACTOR_DEN as f64;
    // `is_nan` check: a NaN markup must fail too, and `<` alone lets it pass.
    if markup.is_nan() || markup < floor {
        return Err(format!(
            "PRICING_MARKUP ({markup}) must be >= the safe cost factor \
             {}/{} = {floor}: the pricing contract charges prompt bytes at \
             1/{floor} of their worst-case token count, so a lower markup \
             would sell tokens below cost",
            eidola_common::SAFE_COST_FACTOR_NUM,
            eidola_common::SAFE_COST_FACTOR_DEN,
        ));
    }
    Ok(())
}

/// Fixed scale factor for pricing: credits per token = value / PRICING_SCALE_FACTOR.
pub const PRICING_SCALE_FACTOR: u64 = 1_000_000;

/// A static model catalog entry: what we sell, at what price, and what it can
/// do.
///
/// The capability fields are transcribed from the upstream model list, which
/// publishes them per model — they are not a convention invented here. The two
/// output-limit fields are the exception and are declared by us (see
/// `max_output_tokens`).
struct CatalogEntry {
    id: &'static str,
    name: &'static str,
    description: &'static str,
    context_length: u64,
    /// Whether the model accepts a `tools` request field.
    tool_calling: bool,
    /// Whether the model produces reasoning content before its answer.
    reasoning: bool,
    /// Content kinds the model accepts / produces.
    input_modalities: &'static [Modality],
    output_modalities: &'static [Modality],
    /// The largest completion we are prepared to ask this model for.
    ///
    /// **Declared by us, not transcribed.** No upstream we serve publishes a
    /// maximum-output figure for any of these models, so this is a ceiling we
    /// commit to rather than a fact we were handed: a length the model has
    /// been seen to reach comfortably, kept well inside its context window so
    /// a long prompt still leaves room to answer. Budget ladders live *under*
    /// this number; none may cross it.
    max_output_tokens: Option<u64>,
    /// Which public output-budget ladder the model draws from. Declared
    /// rather than derived from `reasoning`, so a model can be moved between
    /// ladders without the catalog having to lie about what it does.
    output_budget_class: OutputBudgetClass,
    /// USD per million input tokens (0.0 for per-request models).
    input_per_m: f64,
    /// USD per million output tokens (0.0 for per-request models).
    output_per_m: f64,
    /// USD per request (0.0 for token-based models).
    per_request_usd: f64,
}

/// Hardcoded model catalog. Model identifiers, descriptions and capabilities
/// are bound to the image contents; only pricing can be overridden at runtime
/// via `TINFOIL_PRICING_OVERRIDES`.
///
/// **This stays a build input rather than a boot-time mirror of the upstream
/// list.** The catalog is what an enclave *publishes* about the models it
/// sells, and clients shape requests from it — whether an agent is offered
/// tools, most of all. Mirroring upstream at boot would make that an
/// unmeasured input, so a changed upstream could silently alter what a client
/// is told a model can do without any measurement moving. Same argument the
/// GUI makes about translations being build inputs rather than runtime loads.
///
/// **Upstream is authoritative for what a model *is*; this list is
/// authoritative for what we sell.** The two are deliberately different sizes
/// — see `NOT_SOLD_UPSTREAM_MODELS`, which records every deliberate omission
/// so the gap stays a decision instead of decaying into an oversight.
///
/// Prices are Tinfoil's list prices in USD per million tokens (per request for
/// the per-request models), before `PRICING_MARKUP`.
///
/// **Cached input is not modeled.** Tinfoil quotes a discounted cached-input
/// price for some of these models — `glm-5-2` at $0.375/M against its $1.50
/// full input price, for one — but the pricing contract (`eidola-common`, and
/// the `ModelPricing` shape clients read from `/v1/models`) has a single
/// prompt-token rate with no cache-hit term. Charging every prompt token at
/// the full input price therefore over-covers a cached turn rather than
/// under-covering it, which is the safe direction for both the client's hold
/// and the server's charge; modeling the discount would be a wire-visible
/// contract change on both sides.
const MODEL_CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        id: "glm-5-2",
        name: "GLM-5.2",
        description: "Advanced language model with strong reasoning and multilingual capabilities",
        context_length: 393_216,
        tool_calling: true,
        reasoning: true,
        input_modalities: &[Modality::Text],
        output_modalities: &[Modality::Text],
        max_output_tokens: Some(32_768),
        output_budget_class: OutputBudgetClass::Reasoning,
        input_per_m: 1.5,
        output_per_m: 5.25,
        per_request_usd: 0.0,
    },
    CatalogEntry {
        id: "deepseek-v4-flash",
        name: "DeepSeek V4 Flash",
        description: "Efficient mixture-of-experts model with a 1M-token context window, speculative decoding, and tool calling",
        context_length: 1_048_576,
        tool_calling: true,
        reasoning: true,
        input_modalities: &[Modality::Text],
        output_modalities: &[Modality::Text],
        max_output_tokens: Some(32_768),
        output_budget_class: OutputBudgetClass::Reasoning,
        input_per_m: 0.70,
        output_per_m: 1.90,
        per_request_usd: 0.0,
    },
    CatalogEntry {
        id: "gemma4-31b",
        name: "Gemma 4 31B",
        description: "Lightweight and efficient language model from Google for versatile use cases",
        context_length: 262_144,
        tool_calling: true,
        reasoning: true,
        input_modalities: &[Modality::Text, Modality::Image],
        output_modalities: &[Modality::Text],
        max_output_tokens: Some(16_384),
        output_budget_class: OutputBudgetClass::Reasoning,
        input_per_m: 0.40,
        output_per_m: 1.0,
        per_request_usd: 0.0,
    },
    CatalogEntry {
        id: "kimi-k3",
        name: "Kimi K3",
        description: "Multimodal mixture-of-experts model with hybrid attention for long-context reasoning, coding, and agentic workflows",
        context_length: 262_144,
        tool_calling: true,
        reasoning: true,
        input_modalities: &[Modality::Text, Modality::Image],
        output_modalities: &[Modality::Text],
        max_output_tokens: Some(32_768),
        output_budget_class: OutputBudgetClass::Reasoning,
        input_per_m: 4.0,
        output_per_m: 20.0,
        per_request_usd: 0.0,
    },
    CatalogEntry {
        id: "gpt-oss-120b",
        name: "GPT-OSS 120B",
        description: "Open-weight model designed for powerful reasoning, agentic tasks, and versatile use cases",
        context_length: 131_072,
        tool_calling: true,
        reasoning: true,
        input_modalities: &[Modality::Text],
        output_modalities: &[Modality::Text],
        max_output_tokens: Some(32_768),
        output_budget_class: OutputBudgetClass::Reasoning,
        input_per_m: 0.15,
        output_per_m: 0.60,
        per_request_usd: 0.0,
    },
    CatalogEntry {
        id: "gpt-oss-safeguard-120b",
        name: "GPT-OSS Safeguard 120B",
        description: "Safety reasoning model for content classification and trust & safety applications",
        context_length: 131_072,
        tool_calling: true,
        reasoning: true,
        input_modalities: &[Modality::Text],
        output_modalities: &[Modality::Text],
        max_output_tokens: Some(8_192),
        output_budget_class: OutputBudgetClass::Reasoning,
        input_per_m: 0.15,
        output_per_m: 0.60,
        per_request_usd: 0.0,
    },
    CatalogEntry {
        id: "voxtral-small-24b",
        name: "Voxtral Small 24B",
        description: "Audio-capable model built on Mistral Small 3 for transcription, translation, and spoken queries",
        context_length: 32_768,
        // The one row in this catalog that declares **no** tool calling, and
        // the reason a declaration is worth carrying at all: offering it tools
        // costs a paid round that can only fail.
        tool_calling: false,
        reasoning: false,
        input_modalities: &[Modality::Text],
        output_modalities: &[Modality::Text],
        max_output_tokens: Some(4_096),
        output_budget_class: OutputBudgetClass::Standard,
        input_per_m: 0.20,
        output_per_m: 0.60,
        per_request_usd: 0.0,
    },
    CatalogEntry {
        id: "llama3-3-70b",
        name: "Llama 3.3 70B",
        description: "High-performance multilingual language model optimized for speed",
        context_length: 131_072,
        tool_calling: true,
        reasoning: false,
        input_modalities: &[Modality::Text],
        output_modalities: &[Modality::Text],
        max_output_tokens: Some(8_192),
        output_budget_class: OutputBudgetClass::Standard,
        input_per_m: 1.75,
        output_per_m: 2.75,
        per_request_usd: 0.0,
    },
];

/// An upstream model this catalog deliberately does not carry.
pub struct UnsoldModel {
    /// The upstream model id.
    pub id: &'static str,
    /// Why it is not sold.
    pub reason: &'static str,
}

/// The upstream ids we have decided **not** to sell, and why.
///
/// This exists so that "deliberately not sold" and "we missed it" are
/// different things in the tree. The catalog is strictly smaller than the
/// upstream list, and without a record of the gap any check that compares the
/// two spends its life reporting the same known differences until somebody
/// mutes it. This list is what gives such a check teeth: anything upstream
/// publishes that appears in neither the catalog nor here is genuinely new.
///
/// Every entry today is the same call — this server exposes exactly one
/// inference route, `POST /v1/chat/completions`, so a model that serves some
/// other endpoint is not something a conversation can be routed to. Listing
/// one anyway would sell a selection that can only fail at the point of use,
/// and no downstream filter is as strong as never offering it.
pub const NOT_SOLD_UPSTREAM_MODELS: &[UnsoldModel] = &[
    UnsoldModel {
        id: "nomic-embed-text",
        reason: "embeddings only (/v1/embeddings); no embeddings route is exposed",
    },
    UnsoldModel {
        id: "whisper-large-v3-turbo",
        reason: "transcription only (/v1/audio/transcriptions); no audio route is exposed",
    },
    UnsoldModel {
        id: "doc-upload",
        reason: "document conversion (/v1/convert/file); no conversion route is exposed",
    },
    UnsoldModel {
        id: "websearch",
        reason: "an upstream-hosted tool, not a model a conversation is routed to",
    },
    UnsoldModel {
        id: "qwen3-tts",
        reason: "speech synthesis (/v1/audio/speech); no audio route is exposed",
    },
    UnsoldModel {
        id: "voxtral-tts",
        reason: "speech synthesis (/v1/audio/speech); no audio route is exposed",
    },
    UnsoldModel {
        id: "voxtral-mini-4b-realtime",
        reason: "realtime sessions (/v1/realtime); no realtime route is exposed",
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
/// Example JSON: `{"kimi-k3": {"input": 4.0, "output": 20.0}}`
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
    /// The attesting upstream client, held in a swappable cell so the
    /// `upstream_trust` refresh task can hot-swap in a client built over a
    /// new allowed-measurement set without rebuilding the backend. Each
    /// request reads the current client lock-free via `load()`. See
    /// `src/upstream_trust`.
    client: std::sync::Arc<arc_swap::ArcSwap<reqwest::Client>>,
    api_key: String,
    base_url: String,
    /// Static model list built from `MODEL_CATALOG` with optional pricing overrides.
    models: Vec<Model>,
}

impl TinfoilBackend {
    pub fn new(
        client: std::sync::Arc<arc_swap::ArcSwap<reqwest::Client>>,
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
        Self::build_list_from(MODEL_CATALOG, markup, overrides)
    }

    /// The catalog → wire projection, over an explicit catalog slice so the
    /// per-request pricing shape can be exercised against a synthetic row.
    /// The shape outlives any one model that uses it (transcription is priced
    /// per request, and nothing says the next thing we sell will not be), so
    /// it is kept and tested rather than deleted along with the last row that
    /// happened to need it.
    fn build_list_from(
        catalog: &[CatalogEntry],
        markup: f64,
        overrides: &HashMap<String, PricingOverride>,
    ) -> Vec<Model> {
        catalog
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
                    max_output_tokens: entry.max_output_tokens,
                    output_budget_class: entry.output_budget_class,
                    capabilities: ModelCapabilities {
                        tool_calling: Capability::new(entry.tool_calling),
                        reasoning: Capability::new(entry.reasoning),
                        input_modalities: entry.input_modalities.to_vec(),
                        output_modalities: entry.output_modalities.to_vec(),
                    },
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

    #[tracing::instrument(skip_all, name = "upstream.chat", err)]
    async fn send(&self, request: &ChatCompletionRequest) -> Result<BackendResponse, ServerError> {
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .load()
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
            // Privacy: serde messages quote offending values, and this body
            // is inference content — only the category summary may reach the
            // log path or the `Parse` message (which `Display`s into logs
            // via `err` instrumentation).
            let summary = crate::error::parse_error_summary(&e);
            tracing::error!("failed to parse backend response: {summary}");
            ServerError::Parse(summary)
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

    #[tracing::instrument(skip_all, name = "upstream.chat_stream", err)]
    async fn send_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<mpsc::Receiver<Result<BackendStreamEvent, ServerError>>, ServerError> {
        let url = format!("{}/chat/completions", self.base_url);

        // Ensure stream=true in the forwarded request, and force
        // include_usage on so the upstream emits a final usage chunk.
        // We need usage to compute the per-token refund; without it we'd
        // default to a zero refund and effectively bill the client the
        // worst-case `charge_credits` for every streaming request. This
        // is server policy, not a client choice — we override whatever
        // the caller set.
        let mut stream_request = request.clone();
        stream_request.stream = true;
        stream_request.stream_options = Some(crate::types::StreamOptions {
            include_usage: true,
        });

        let response = self
            .client
            .load()
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
                        // One stamp per socket read: every SSE event
                        // extracted from this buffer batch arrived
                        // together, and stamping here (before any
                        // channel send can block on backpressure) is
                        // what keeps the timing instruments measuring
                        // the upstream rather than the client.
                        let received_at = std::time::Instant::now();
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
                                    if tx
                                        .send(Ok(BackendStreamEvent::Chunk(chunk, received_at)))
                                        .await
                                        .is_err()
                                    {
                                        return; // Client disconnected
                                    }
                                }
                                Err(e) => {
                                    // Privacy: serde messages quote offending
                                    // values, and this chunk is inference
                                    // content — log only the category
                                    // summary.
                                    tracing::warn!(
                                        "failed to parse SSE chunk: {}",
                                        crate::error::parse_error_summary(&e)
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
            let _ = tx
                .send(Ok(BackendStreamEvent::Done(
                    meta,
                    std::time::Instant::now(),
                )))
                .await;
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
    fn markup_below_safe_cost_factor_is_rejected() {
        // The startup gate: a markup under the contract's safe cost factor
        // (3/2 = 1.5) would sell prompt tokens below cost.
        assert!(validate_pricing_markup(1.2).is_err());
        assert!(validate_pricing_markup(0.0).is_err());
        assert!(validate_pricing_markup(f64::NAN).is_err());
        assert!(validate_pricing_markup(1.5).is_ok());
        assert!(validate_pricing_markup(DEFAULT_PRICING_MARKUP).is_ok());
        assert!(validate_pricing_markup(2.0).is_ok());
    }

    #[test]
    fn test_usd_per_m_to_scaled_credits() {
        // glm-5-2 input: $1.5/M tokens with 1.5x markup
        // The 1e6 factors (USD→µ$ and /M→/token) cancel:
        // scaled = 1.5 * 1.5 * 1_000_000 = 2_250_000
        assert_eq!(usd_per_m_to_scaled_credits(1.5, 1.5), 2_250_000);

        // gpt-oss-120b input: $0.15/M with 1.5x markup
        // scaled = 0.15 * 1.5 * 1_000_000 = 225_000
        assert_eq!(usd_per_m_to_scaled_credits(0.15, 1.5), 225_000);

        // Zero price
        assert_eq!(usd_per_m_to_scaled_credits(0.0, 1.5), 0);

        // A rate with no exact binary representation: $0.05/M with 1.5x markup.
        // 0.05 * 1.5 = 0.075; ceil(0.075 * 1e6) = 75_000
        // (may be 75_001 due to f64 representation — ceil rounds up any epsilon)
        let epsilon_rate = usd_per_m_to_scaled_credits(0.05, 1.5);
        assert!(
            epsilon_rate == 75_000 || epsilon_rate == 75_001,
            "got {epsilon_rate}"
        );
    }

    #[test]
    fn test_usd_per_req_to_scaled_credits() {
        // $0.01/req with 1.5x markup
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
            assert!(
                !model.capabilities.input_modalities.is_empty(),
                "{} declares no input modality",
                model.id
            );
            assert!(
                !model.capabilities.output_modalities.is_empty(),
                "{} declares no output modality",
                model.id
            );
        }
    }

    /// Every catalog row reaches the wire **whole**: same window, same
    /// capabilities, same declared output ceiling, prices multiplied by the
    /// markup and by nothing else.
    ///
    /// The table this walks is the catalog itself rather than a second
    /// hand-written copy of the same numbers beside it, and that is the whole
    /// change. Two static copies in one file agree by construction: they catch
    /// a stray edit to one of them and nothing else. What they cannot catch is
    /// the failure the pinning was written for — an upstream value moving out
    /// from under a row that was correct when it was transcribed — because
    /// both copies are equally stale. Detecting *that* needs a live second
    /// copy, so it belongs to a scheduled check against the published list,
    /// reading `NOT_SOLD_UPSTREAM_MODELS` to know which absences are
    /// deliberate. Left here is the half a unit test can actually prove, and
    /// the numbers now exist once in the tree instead of twice.
    #[test]
    fn the_catalog_reaches_the_wire_whole() {
        let models = TinfoilBackend::build_model_list(DEFAULT_PRICING_MARKUP, &HashMap::new());
        assert_eq!(models.len(), MODEL_CATALOG.len(), "catalog size");

        for entry in MODEL_CATALOG {
            let id = entry.id;
            let model = models
                .iter()
                .find(|m| m.id == id)
                .unwrap_or_else(|| panic!("{id} missing from the published list"));

            assert_eq!(model.name, entry.name, "{id} name");
            assert_eq!(model.description, entry.description, "{id} description");
            assert_eq!(model.context_length, entry.context_length, "{id} context");
            assert_eq!(
                model.max_output_tokens, entry.max_output_tokens,
                "{id} output ceiling"
            );
            assert_eq!(
                model.output_budget_class, entry.output_budget_class,
                "{id} budget class"
            );
            assert_eq!(
                model.capabilities.tool_calling.supported, entry.tool_calling,
                "{id} tool calling"
            );
            assert_eq!(
                model.capabilities.reasoning.supported, entry.reasoning,
                "{id} reasoning"
            );
            assert_eq!(
                model.capabilities.input_modalities, entry.input_modalities,
                "{id} input modalities"
            );
            assert_eq!(
                model.capabilities.output_modalities, entry.output_modalities,
                "{id} output modalities"
            );

            if entry.per_request_usd > 0.0 {
                assert_eq!(
                    model.pricing.per_request.as_ref().map(|p| p.value),
                    Some(usd_per_req_to_scaled_credits(
                        entry.per_request_usd,
                        DEFAULT_PRICING_MARKUP
                    )),
                    "{id} request price"
                );
            } else {
                assert!(model.pricing.per_request.is_none(), "{id} request price");
                assert_eq!(
                    model.pricing.per_prompt_token.value,
                    usd_per_m_to_scaled_credits(entry.input_per_m, DEFAULT_PRICING_MARKUP),
                    "{id} prompt price"
                );
                assert_eq!(
                    model.pricing.per_completion_token.value,
                    usd_per_m_to_scaled_credits(entry.output_per_m, DEFAULT_PRICING_MARKUP),
                    "{id} completion price"
                );
            }
        }
    }

    /// This server registers exactly one inference route, so every row it
    /// sells has to be something a conversation can actually be routed to.
    ///
    /// The strongest form of "never offer a choice we know to be impossible"
    /// is not to list it: a model with no endpoint here never reaches a picker
    /// to be filtered out of, and no downstream filter is as reliable as the
    /// absence of the row. Every model that fails this is recorded in
    /// `NOT_SOLD_UPSTREAM_MODELS` instead.
    #[test]
    fn every_catalog_row_is_a_chat_model() {
        for entry in MODEL_CATALOG {
            let id = entry.id;
            assert!(
                entry.context_length > 0,
                "{id} declares no context window, so it cannot be a chat model"
            );
            assert!(
                entry.input_modalities.contains(&Modality::Text)
                    && entry.output_modalities.contains(&Modality::Text),
                "{id} does not take text in and give text back"
            );
            assert_eq!(
                entry.per_request_usd, 0.0,
                "{id} is priced per request, which no chat model here is"
            );
        }
    }

    /// The deliberate-omission record is what tells "not sold" from "missed",
    /// so it must not disagree with the catalog or repeat itself.
    #[test]
    fn the_omission_record_is_disjoint_from_the_catalog() {
        for unsold in NOT_SOLD_UPSTREAM_MODELS {
            assert!(
                !MODEL_CATALOG.iter().any(|e| e.id == unsold.id),
                "{} is both sold and recorded as not sold",
                unsold.id
            );
            assert!(
                !unsold.reason.trim().is_empty(),
                "{} is omitted without a reason",
                unsold.id
            );
        }
        for (i, unsold) in NOT_SOLD_UPSTREAM_MODELS.iter().enumerate() {
            assert!(
                !NOT_SOLD_UPSTREAM_MODELS[i + 1..]
                    .iter()
                    .any(|o| o.id == unsold.id),
                "{} is recorded twice",
                unsold.id
            );
        }
        for (i, entry) in MODEL_CATALOG.iter().enumerate() {
            assert!(
                !MODEL_CATALOG[i + 1..].iter().any(|o| o.id == entry.id),
                "{} is listed twice",
                entry.id
            );
        }
    }

    /// A declared output ceiling is a promise about one response, and a
    /// response shares the window with its prompt — so a ceiling at or above
    /// the window promises a turn with no room to ask anything.
    #[test]
    fn every_declared_output_ceiling_leaves_room_for_a_prompt() {
        for entry in MODEL_CATALOG {
            let Some(ceiling) = entry.max_output_tokens else {
                continue;
            };
            assert!(
                ceiling < entry.context_length,
                "{}'s output ceiling ({ceiling}) does not fit inside its \
                 context window ({})",
                entry.id,
                entry.context_length
            );
        }
    }

    #[test]
    fn test_pricing_overrides() {
        let mut overrides = HashMap::new();
        overrides.insert(
            "kimi-k3".to_string(),
            PricingOverride {
                input: Some(2.0),
                output: Some(6.0),
                request: None,
            },
        );

        let models = TinfoilBackend::build_model_list(1.0, &overrides);
        let kimi = models.iter().find(|m| m.id == "kimi-k3").unwrap();

        // With 1.0x markup and $2.0/M input override
        assert_eq!(kimi.pricing.per_prompt_token.value, 2_000_000);
        assert_eq!(kimi.pricing.per_completion_token.value, 6_000_000);
        assert!(kimi.pricing.per_request.is_none());
    }

    /// The per-request pricing shape, exercised against a **synthetic** row.
    ///
    /// Nothing in the catalog is priced per request any more — the models that
    /// were are all served by endpoints this server does not expose. The shape
    /// stays because it is contract, not because a particular row needs it,
    /// and a fixture is the honest way to keep testing a shape that no
    /// shipped row currently wears.
    const PER_REQUEST_FIXTURE: &[CatalogEntry] = &[CatalogEntry {
        id: "per-request-fixture",
        name: "Per-Request Fixture",
        description: "A synthetic row priced per request rather than per token",
        context_length: 8_192,
        tool_calling: false,
        reasoning: false,
        input_modalities: &[Modality::Text],
        output_modalities: &[Modality::Text],
        max_output_tokens: None,
        output_budget_class: OutputBudgetClass::Standard,
        input_per_m: 0.0,
        output_per_m: 0.0,
        per_request_usd: 0.01,
    }];

    #[test]
    fn test_per_request_model_pricing() {
        let overrides = HashMap::new();
        let models = TinfoilBackend::build_list_from(PER_REQUEST_FIXTURE, 1.0, &overrides);

        let model = models
            .iter()
            .find(|m| m.id == "per-request-fixture")
            .unwrap();
        assert!(model.pricing.per_request.is_some());
        assert_eq!(model.pricing.per_prompt_token.value, 0);
        assert_eq!(model.pricing.per_completion_token.value, 0);

        // $0.01 * 1e6 * 1.0 * 1e6 = 10_000_000_000
        assert_eq!(
            model.pricing.per_request.as_ref().unwrap().value,
            10_000_000_000
        );
    }

    #[test]
    fn test_per_request_override() {
        let mut overrides = HashMap::new();
        overrides.insert(
            "per-request-fixture".to_string(),
            PricingOverride {
                input: None,
                output: None,
                request: Some(0.02),
            },
        );

        let models = TinfoilBackend::build_list_from(PER_REQUEST_FIXTURE, 1.0, &overrides);
        let model = models
            .iter()
            .find(|m| m.id == "per-request-fixture")
            .unwrap();

        // $0.02 * 1e6 * 1.0 * 1e6 = 20_000_000_000
        assert_eq!(
            model.pricing.per_request.as_ref().unwrap().value,
            20_000_000_000
        );
    }

    #[test]
    fn test_lookup_model() {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let backend = TinfoilBackend::new(
            std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(reqwest::Client::new())),
            String::new(),
            None,
            Some(1.5),
        );

        assert!(backend.lookup_model("kimi-k3").is_some());
        assert!(backend.lookup_model("nonexistent").is_none());
        // Retired upstream ids must not linger in the catalog: a client that
        // still asks for one should get an honest 404 rather than a price.
        assert!(backend.lookup_model("kimi-k2-6").is_none());
        assert!(backend.lookup_model("deepseek-v4-pro").is_none());
        // Nor may a model this server has no route for. These are live
        // upstream ids; not selling them is the decision, and a 404 is what
        // that decision has to look like from outside.
        for unsold in NOT_SOLD_UPSTREAM_MODELS {
            assert!(
                backend.lookup_model(unsold.id).is_none(),
                "{} is recorded as not sold but is priced",
                unsold.id
            );
        }
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
