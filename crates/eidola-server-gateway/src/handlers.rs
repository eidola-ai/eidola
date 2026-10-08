//! Top-level HTTP handlers: health, models, chat completions.

use std::time::{Duration, Instant};

use anonymous_credit_tokens::{Scalar, SpendProof, credit_to_scalar, scalar_to_credit};
use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request, State};
use axum::response::IntoResponse;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use opentelemetry::KeyValue;
use rand_core::OsRng;
use serde::de::DeserializeOwned;
use tokio::sync::mpsc;
use tracing::{error, warn};

use crate::AppState;
use crate::auth::{ActSpend, AuthContext, AuthMethod, TokenAuth};
use crate::backend::{BackendStreamEvent, ChatBackend, PRICING_SCALE_FACTOR};
use crate::credentials;
use crate::db;
use crate::engine_trust::protocol::ValidatedRequest;
use crate::error::ServerError;
use crate::padding::{self, Cadence, StreamEvent};
use crate::response::{
    EidolaResponse, EidolaStreamMetadata, RefundInfo, build_privacy_metadata,
    build_verification_metadata,
};
use crate::types::{
    ChatCompletionChunk, ChatCompletionRequest, ErrorResponse, Model, ModelHosting, ModelsResponse,
    Usage,
};

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

// ---------------------------------------------------------------------------
// Billing helpers
// ---------------------------------------------------------------------------

/// Extract the shared pricing contract's inputs from a request and compute
/// its chargeable prompt tokens.
///
/// The walk itself is [`eidola_common::prompt_charge`] — the single
/// implementation the client calls too, which is what makes "both sides
/// compute the same function of the same request" structural rather than a
/// convention two crates maintain in parallel. The only server-side work is
/// handing it the request's `messages` and `tools` arrays as JSON.
///
/// The `to_value` conversion is deliberate and cheap in context: this runs
/// two or three times per chat completion, against a request the server
/// already fully re-serializes to forward upstream, on the far side of an
/// LLM call. Reconstructing the walk over the typed struct to avoid it would
/// reintroduce exactly the drift the consolidation removed.
pub(crate) fn chargeable_prompt_tokens_for(request: &ChatCompletionRequest) -> u64 {
    let Ok(value) = serde_json::to_value(request) else {
        // Unreachable in practice (the request was deserialized from JSON and
        // every field is serializable). Falling back to the message count
        // alone keeps the constants' floor rather than charging nothing.
        return eidola_common::chargeable_prompt_tokens(0, request.messages.len() as u64);
    };
    let messages = value
        .get("messages")
        .and_then(|m| m.as_array())
        .map(|m| m.as_slice())
        .unwrap_or(&[]);
    let tools = value
        .get("tools")
        .and_then(|t| t.as_array())
        .map(Vec::as_slice);
    eidola_common::prompt_charge(messages, tools).chargeable_prompt_tokens()
}

/// The effective completion-token ceiling for a request: its
/// `max_completion_tokens`, falling back to the model's context length.
pub(crate) fn effective_max_completion(request: &ChatCompletionRequest, model: &Model) -> u64 {
    request
        .max_completion_tokens
        .map(|t| t as u64)
        .unwrap_or(model.context_length)
}

/// Compute the worst-case cost in credits for a request — the pre-flight
/// minimum hold.
///
/// For per-request models (e.g., Whisper, TTS), returns the flat per-request
/// price. For token-based models, the prompt side is the shared client/server
/// pricing contract (`eidola_common::chargeable_prompt_tokens`: a content-byte
/// term at the safe cost factor plus per-message and per-request constants),
/// and the completion side uses `max_completion_tokens` (or context_length).
/// The client sizes its hold with the identical function of the identical
/// request, so the client and server go/no-go decisions agree bit-for-bit.
fn worst_case_cost(request: &ChatCompletionRequest, model: &Model) -> u128 {
    // Per-request pricing: flat cost regardless of token count.
    if let Some(ref per_req) = model.pricing.per_request {
        return (per_req.value as u128).div_ceil(per_req.scale_factor as u128);
    }

    let sf = PRICING_SCALE_FACTOR as u128;

    // Prompt: the shared contract formula.
    let prompt_rate = model.pricing.per_prompt_token.value as u128;
    let prompt_credits = (chargeable_prompt_tokens_for(request) as u128 * prompt_rate).div_ceil(sf);

    // Completion: use max_completion_tokens or fall back to context_length.
    let completion_rate = model.pricing.per_completion_token.value as u128;
    let completion_credits =
        (effective_max_completion(request, model) as u128 * completion_rate).div_ceil(sf);

    prompt_credits + completion_credits
}

/// Compute the actual cost in credits from usage data, clamped to the
/// pricing contract.
///
/// The prompt component charges `min(actual_prompt_tokens,
/// chargeable_prompt_tokens(...))` — the contract's cap, which guarantees
/// the charge never exceeds the hold both sides computed pre-flight. The
/// completion component is bounded by the request's effective
/// max-completion ceiling anyway (the model stops there), but is clamped
/// defensively against a misbehaving upstream usage report.
fn actual_cost(
    usage: &Usage,
    model: &Model,
    chargeable_prompt_tokens: u64,
    max_completion_tokens: u64,
) -> u128 {
    // Per-request pricing: flat cost regardless of actual token usage.
    if let Some(ref per_req) = model.pricing.per_request {
        return (per_req.value as u128).div_ceil(per_req.scale_factor as u128);
    }

    let sf = PRICING_SCALE_FACTOR as u128;
    let charged_prompt = (usage.prompt_tokens as u64).min(chargeable_prompt_tokens);
    let charged_completion = (usage.completion_tokens as u64).min(max_completion_tokens);
    let prompt_cost = charged_prompt as u128 * model.pricing.per_prompt_token.value as u128;
    let completion_cost =
        charged_completion as u128 * model.pricing.per_completion_token.value as u128;
    // Ceiling division for each component, then sum
    let prompt_credits = prompt_cost.div_ceil(sf);
    let completion_credits = completion_cost.div_ceil(sf);
    prompt_credits + completion_credits
}

/// What a completed request costs: its usage priced and clamped to the
/// contract ([`actual_cost`]), or the whole charge when the upstream reported
/// no usage. One rule for both hosting paths and both transports.
pub(crate) fn settled_cost(
    usage: Option<&Usage>,
    model: &Model,
    chargeable_prompt_tokens: u64,
    max_completion_tokens: u64,
    charge_credits: u128,
) -> u128 {
    usage
        .map(|u| actual_cost(u, model, chargeable_prompt_tokens, max_completion_tokens))
        .unwrap_or(charge_credits)
}

/// Who answers a request for `model`, as its privacy metadata names them.
fn provider_for(model: &Model) -> &'static str {
    match model.hosting {
        ModelHosting::Tinfoil => "tinfoil",
        ModelHosting::Eidola => crate::engine_router::PROVIDER,
    }
}

/// Send a non-streaming request to the upstream that hosts `model`.
async fn dispatch(
    state: &AppState,
    request: &ValidatedRequest,
    model: &Model,
) -> Result<crate::backend::BackendResponse, ServerError> {
    match model.hosting {
        ModelHosting::Tinfoil => state.backend.send(request.request()).await,
        ModelHosting::Eidola => engines(state)?.send(request).await,
    }
}

/// Send a streaming request to the upstream that hosts `model`.
async fn dispatch_stream(
    state: &AppState,
    request: &ValidatedRequest,
    model: &Model,
) -> Result<mpsc::Receiver<Result<BackendStreamEvent, ServerError>>, ServerError> {
    match model.hosting {
        ModelHosting::Tinfoil => state.backend.send_stream(request.request()).await,
        ModelHosting::Eidola => engines(state)?.send_stream(request).await,
    }
}

/// The engine router. A build that lists an Eidola-hosted model pins it, and
/// a gateway pinning any model starts with a router, so this refuses only
/// what cannot be routed anyway.
fn engines(state: &AppState) -> Result<&crate::engine_router::EngineRouter, ServerError> {
    state.engines.as_ref().ok_or_else(|| {
        ServerError::ServiceUnavailable("no engine is available for this model".to_string())
    })
}

/// Issue a refund token, returning `refund_credits` to the client.
///
/// `refund_credits` is the number of credits to return (i.e., the `t` parameter
/// in the ACT spec — the resulting token will have `c - s + t` credits).
///
/// The refund token is also stored in the nullifier row so the client can
/// recover it via `POST /v1/credentials/refund` if the response is lost.
async fn issue_refund_async(
    state: &AppState,
    spend_proof: &SpendProof<128>,
    issuer_key_hash: &[u8; 32],
    refund_credits: u128,
) -> Result<RefundInfo, ServerError> {
    let t = credit_to_scalar::<128>(refund_credits)
        .map_err(|e| ServerError::Internal(format!("invalid refund amount: {e:?}")))?;

    let cache = state.credential_key_cache.read().await;
    let key = cache
        .get(issuer_key_hash)
        .ok_or_else(|| ServerError::Internal("issuer key not in cache for refund".to_string()))?;

    let refund = key
        .secret_key
        .refund(&key.params, spend_proof, t, OsRng)
        .map_err(|e| ServerError::Internal(format!("refund issuance failed: {e:?}")))?;

    let refund_cbor = refund
        .to_cbor()
        .map_err(|e| ServerError::Internal(format!("refund CBOR encoding failed: {e:?}")))?;

    // Best-effort store in DB for client recovery. Failure here is not fatal
    // — the refund is still returned in the response.
    let key_id = hex::encode(issuer_key_hash);
    let nullifier_bytes = spend_proof.nullifier().as_bytes().to_vec();
    if let Err(e) =
        db::store_refund_token(&state.db_pool, &key_id, &nullifier_bytes, &refund_cbor).await
    {
        warn!("Failed to store refund token for recovery: {e}");
    }

    Ok(RefundInfo {
        refund: URL_SAFE_NO_PAD.encode(&refund_cbor),
        issuer_key_id: key_id,
    })
}

/// Build an HTTP error response that includes a refund token.
fn error_response_with_refund(
    error: &ServerError,
    refund: Option<RefundInfo>,
) -> axum::response::Response {
    let status = error.status_code();
    let mut body = error.to_error_response();
    body.refund = refund.map(|r| serde_json::to_value(r).unwrap());
    (status, Json(body)).into_response()
}

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

/// Cryptographically verify the spend proof.
///
/// Checks challenge_digest, loads the issuer key, validates request_context,
/// and verifies the proof itself. Does NOT record the nullifier — errors here
/// mean the ACT is invalid or malformed, so no refund is needed.
async fn verify_spend_proof(state: &AppState, act: &ActSpend) -> Result<(), ServerError> {
    let master_key = &state.credential_master_key;

    // Verify the challenge_digest matches our expected TokenChallenge.
    let expected_digest = credentials::compute_challenge_digest();
    if act.challenge_digest != expected_digest {
        return Err(ServerError::Unauthorized {
            message: "invalid challenge_digest in token".to_string(),
        });
    }

    // Ensure the issuer key is loaded into the cache.
    credentials::load_key_for_spending(
        &state.credential_key_cache,
        master_key,
        &state.db_pool,
        &act.issuer_key_hash,
    )
    .await?;

    // Verify the spend proof's request_context matches what we expect.
    let cache = state.credential_key_cache.read().await;
    let key = cache.get(&act.issuer_key_hash).ok_or_else(|| {
        ServerError::Internal("issuer key evicted from cache unexpectedly".to_string())
    })?;

    if act.spend_proof.context() != key.request_context_scalar {
        return Err(ServerError::Unauthorized {
            message: "invalid request_context in spend proof".to_string(),
        });
    }

    // Verify the spend proof by calling refund with t=0 (discards the result).
    key.secret_key
        .refund::<128>(&key.params, &act.spend_proof, Scalar::ZERO, OsRng)
        .map_err(|_| ServerError::Unauthorized {
            message: "invalid spend proof".to_string(),
        })?;

    Ok(())
}

/// Validate the model and charge amount against the request.
///
/// Called after the nullifier is recorded. Errors here require a full refund.
fn validate_request(
    state: &AppState,
    act: &ActSpend,
    request: &ChatCompletionRequest,
) -> Result<(Model, u128), ServerError> {
    // Decode the charge amount from the spend proof.
    let charge_credits = scalar_to_credit::<128>(&act.spend_proof.charge()).map_err(|_| {
        ServerError::BadRequest {
            message: "invalid charge amount in spend proof".to_string(),
        }
    })?;

    // Look up the model and validate pricing.
    let model =
        state
            .backend
            .lookup_model(&request.model)
            .ok_or_else(|| ServerError::BadRequest {
                message: "unknown model".to_string(),
            })?;

    // Pre-flight go/no-go: the presented charge must cover the worst-case
    // cost (the shared contract's minimum hold).
    check_sufficient_charge(charge_credits, request, &model)?;

    Ok((model, charge_credits))
}

/// Pre-flight go/no-go: reject a spend that presents less than the
/// worst-case cost — the same formula the client used to size its hold —
/// before anything is sent upstream. Pure so it is directly unit-testable.
fn check_sufficient_charge(
    charge_credits: u128,
    request: &ChatCompletionRequest,
    model: &Model,
) -> Result<(), ServerError> {
    let wc = worst_case_cost(request, model);
    if charge_credits < wc {
        return Err(ServerError::PaymentRequired {
            message: format!(
                "insufficient charge: {} credits provided, {} required (worst case)",
                charge_credits, wc
            ),
            available: charge_credits as i64,
        });
    }
    Ok(())
}

/// Create a chat completion.
///
/// Requires an ACT (Anonymous Credit Token) for authorization. The spend proof
/// is verified, the nullifier is recorded, and a refund token is issued with
/// any unspent credits.
#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    tag = "Unlinked",
    request_body = ChatCompletionRequest,
    responses(
        (status = 200, description = "Chat completion response with privacy and verification metadata", body = EidolaResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 401, description = "Authentication failed", body = ErrorResponse),
        (status = 402, description = "Insufficient charge amount", body = ErrorResponse),
        (status = 409, description = "Credential already spent", body = ErrorResponse),
        (status = 502, description = "Upstream provider error", body = ErrorResponse),
        (status = 503, description = "No upstream is available for the model", body = ErrorResponse)
    )
)]
pub async fn chat_completions(
    TokenAuth(act): TokenAuth,
    State(state): State<AppState>,
    ChatRequest(validated): ChatRequest,
) -> Result<axum::response::Response, ServerError> {
    let request = validated.request();
    // The counter's `model` label must come from the catalog, never the
    // caller's string (the fixed-list rule bounding label cardinality):
    // resolve it up front, with anything unresolved collapsing to `other`.
    let model_label = state
        .backend
        .lookup_model(&request.model)
        .map(|m| m.id)
        .unwrap_or_else(|| "other".to_string());
    let stream = request.stream;

    let result = chat_completions_phases(&state, &act, &validated).await;

    // An `Ok` from the phases can still be an error *response* — a
    // refund-bearing 4xx/5xx built after the credential was spent — so
    // the status label reads the response, not the `Result`. `rejected`
    // is the pre-flight refusals (invalid or duplicate credential,
    // unknown model, insufficient charge). A streaming request counts
    // `ok` once the SSE response opens; how the stream itself ends is
    // `CHAT_STREAM_OUTCOME`'s job.
    let status = match &result {
        Ok(response) if response.status().is_success() => "ok",
        Ok(_) => "error",
        Err(_) => "rejected",
    };
    crate::telemetry::metrics::CHAT_REQUESTS.add(
        1,
        &[
            KeyValue::new("model", model_label),
            KeyValue::new("stream", if stream { "true" } else { "false" }),
            KeyValue::new("status", status),
        ],
    );

    // Every answer from here is sized to a bucket (a JSON body: a completion,
    // or an error) or framed (an event stream): see `padding`.
    let response = result.unwrap_or_else(IntoResponse::into_response);
    Ok(padding::pad_json_response(response).await)
}

async fn chat_completions_phases(
    state: &AppState,
    act: &ActSpend,
    validated: &ValidatedRequest,
) -> Result<axum::response::Response, ServerError> {
    let request = validated.request();
    // Phase 1: Verify the ACT cryptographically. Errors here mean the token
    // is invalid/malformed — no nullifier recorded, no refund needed.
    verify_spend_proof(state, act).await?;

    // Phase 2: Record the nullifier. After this succeeds, the credential is
    // consumed and we MUST issue a refund on every subsequent code path.
    let key_id = hex::encode(act.issuer_key_hash);
    let nullifier = act.spend_proof.nullifier();
    let nullifier_bytes = nullifier.as_bytes().to_vec();
    let recorded = db::record_nullifier(&state.db_pool, &key_id, &nullifier_bytes).await?;
    if !recorded {
        return Err(ServerError::Conflict {
            message: "credential already spent (duplicate nullifier)".to_string(),
        });
    }

    // --- POINT OF NO RETURN: nullifier is recorded ---
    // From here on, we MUST issue a refund on any error.

    // Phase 3: Validate the request (model, charge amount). On failure, issue
    // a full refund of the charge amount back to the client.
    let (model, charge_credits) = match validate_request(state, act, request) {
        Ok(v) => v,
        Err(e) => {
            // Decode charge for the refund. If this also fails, fall back to
            // zero refund (returns blind remaining value c - s).
            let refund_credits = scalar_to_credit::<128>(&act.spend_proof.charge()).unwrap_or(0);
            warn!("Request validation failed after nullifier recorded, issuing full refund: {e}");
            let refund = issue_refund_async(
                state,
                &act.spend_proof,
                &act.issuer_key_hash,
                refund_credits,
            )
            .await;
            return Ok(error_response_with_refund(&e, refund.ok()));
        }
    };

    // Phase 4: Handle the request.
    if request.stream {
        handle_streaming_request(state.clone(), validated, act, &model, charge_credits).await
    } else {
        handle_non_streaming_request(state, validated, act, &model, charge_credits).await
    }
}

/// Handle a non-streaming chat completion request.
async fn handle_non_streaming_request(
    state: &AppState,
    validated: &ValidatedRequest,
    act: &ActSpend,
    model: &Model,
    charge_credits: u128,
) -> Result<axum::response::Response, ServerError> {
    let request = validated.request();
    let auth_context = AuthContext {
        method: AuthMethod::AnonymousCredential,
    };

    // Make the backend request. On error, issue a full refund.
    let dispatched_at = Instant::now();
    let backend_response = match dispatch(state, validated, model).await {
        Ok(resp) => resp,
        Err(e) => {
            // Known error — backend didn't charge. Full refund.
            warn!("Backend error, issuing full refund: {}", e);
            let refund = issue_refund_async(
                state,
                &act.spend_proof,
                &act.issuer_key_hash,
                charge_credits,
            )
            .await;
            return Ok(error_response_with_refund(&e, refund.ok()));
        }
    };
    crate::telemetry::metrics::CHAT_DURATION.record(
        dispatched_at.elapsed().as_secs_f64(),
        &[KeyValue::new("model", model.id.clone())],
    );

    // Record token usage metrics (safe for unlinked layer: only model + counts).
    if let Some(usage) = &backend_response.meta.usage {
        let model_attr = KeyValue::new("model", model.id.clone());
        crate::telemetry::metrics::CHAT_TOKENS.add(
            usage.prompt_tokens as u64,
            &[model_attr.clone(), KeyValue::new("type", "prompt")],
        );
        crate::telemetry::metrics::CHAT_TOKENS.add(
            usage.completion_tokens as u64,
            &[model_attr, KeyValue::new("type", "completion")],
        );
    }

    // Compute actual cost (clamped to the pricing contract) and refund.
    let chargeable_prompt = chargeable_prompt_tokens_for(request);
    let max_completion = effective_max_completion(request, model);
    // No usage → charge worst case.
    let cost = settled_cost(
        backend_response.meta.usage.as_ref(),
        model,
        chargeable_prompt,
        max_completion,
        charge_credits,
    );

    let refund_credits = charge_credits.saturating_sub(cost);
    let refund_info = match issue_refund_async(
        state,
        &act.spend_proof,
        &act.issuer_key_hash,
        refund_credits,
    )
    .await
    {
        Ok(info) => Some(info),
        Err(e) => {
            error!("CRITICAL: failed to issue refund: {}", e);
            // We were charged, so fall back to refunding 0 (blind remaining value).
            match issue_refund_async(state, &act.spend_proof, &act.issuer_key_hash, 0).await {
                Ok(info) => Some(info),
                Err(e2) => {
                    error!("CRITICAL: failed to issue fallback zero refund: {}", e2);
                    None
                }
            }
        }
    };

    let meta = &backend_response.meta;
    let is_tee = meta.tee_type.is_some();

    let privacy = build_privacy_metadata(&auth_context, is_tee, &meta.provider);
    let verification = build_verification_metadata(None);

    let eidola_response = EidolaResponse::from_completion(
        backend_response.response,
        privacy,
        verification,
        refund_info,
    );

    Ok(Json(eidola_response).into_response())
}

/// Per-stream timing accumulator behind the streaming instruments.
///
/// Lives for exactly one stream and reports once, when the stream terminates.
/// Everything it holds is a duration or a count: it never sees content, and no
/// value it produces is attributed to an individual request — each one lands
/// in a histogram bucket or a counter keyed only by model.
struct StreamTiming {
    model_id: String,
    dispatched_at: Instant,
    first_chunk_at: Option<Instant>,
    last_chunk_at: Option<Instant>,
    chunks: u64,
    /// Output chunks stamped strictly after `first_chunk_at`. Chunk stamps
    /// have socket-read resolution — every SSE event parsed out of one
    /// read shares one stamp — so this, not `chunks`, is what says how
    /// much of the stream the observed timing window actually covers (see
    /// `observations`).
    chunks_after_first: u64,
    max_gap: Duration,
}

impl StreamTiming {
    fn new(model_id: String, dispatched_at: Instant) -> Self {
        Self {
            model_id,
            dispatched_at,
            first_chunk_at: None,
            last_chunk_at: None,
            chunks: 0,
            chunks_after_first: 0,
            max_gap: Duration::ZERO,
        }
    }

    fn model_attr(&self) -> [KeyValue; 1] {
        [KeyValue::new("model", self.model_id.clone())]
    }

    /// Note an output-bearing chunk, at the instant the upstream reader
    /// took it off the socket (see [`BackendStreamEvent`] — handler-side
    /// arrival would additionally fold this process's channel buffering
    /// into every gap; the socket stamp removes that, though a client
    /// slow enough to exhaust the bounded buffers still throttles the
    /// socket itself). Records time-to-first-token on the first one and
    /// tracks the largest inter-chunk gap after that.
    fn on_chunk(&mut self, at: Instant) {
        match self.last_chunk_at {
            None => {
                self.first_chunk_at = Some(at);
                crate::telemetry::metrics::CHAT_TTFT.record(
                    at.duration_since(self.dispatched_at).as_secs_f64(),
                    &self.model_attr(),
                );
            }
            Some(previous) => {
                self.max_gap = self.max_gap.max(at.duration_since(previous));
                if self.first_chunk_at.is_some_and(|first| at > first) {
                    self.chunks_after_first += 1;
                }
            }
        }
        self.last_chunk_at = Some(at);
        self.chunks += 1;
    }

    /// The max-gap and output-rate observations this stream yields, or
    /// `None` where no meaningful observation exists. Pure, so the
    /// arithmetic is unit-testable apart from the instruments.
    ///
    /// Both need real inter-chunk timing to exist: when every output chunk
    /// shared one socket-read stamp (a coalesced read — common for short
    /// responses), no timing was observed at all, and recording anyway
    /// would fabricate a perfectly smooth zero gap and an unbounded rate.
    ///
    /// The rate's window opens at the first stamp, so it covers generating
    /// only the chunks stamped after it — `chunks_after_first` of
    /// `chunks`. The numerator is scaled to that share of the total
    /// tokens; computing the share from stamped-after counts rather than
    /// chunk ordinals is what keeps a coalesced burst honest (499 chunks
    /// in one read followed by 1 more is 1/500 of the tokens across the
    /// window, not 499/500). Assumes tokens spread roughly evenly across
    /// chunks — exact in the common one-token-per-chunk case.
    fn observations(&self, completion_tokens: Option<u64>) -> (Option<f64>, Option<f64>) {
        if self.chunks < 2 || self.chunks_after_first == 0 {
            return (None, None);
        }
        let (Some(first), Some(last)) = (self.first_chunk_at, self.last_chunk_at) else {
            return (None, None);
        };

        let gap = Some(self.max_gap.as_secs_f64());

        let generation = last.duration_since(first).as_secs_f64();
        let rate = match completion_tokens {
            Some(tokens) if tokens > 0 && generation > 0.0 => {
                let tokens_in_window =
                    tokens as f64 * self.chunks_after_first as f64 / self.chunks as f64;
                Some(tokens_in_window / generation)
            }
            _ => None,
        };

        (gap, rate)
    }

    /// Report the stream's outcome. Called exactly once, on every terminal
    /// path, at the moment the terminal condition is observed — before
    /// refund settlement and metadata delivery, which are our own
    /// post-stream work: the duration histogram is documented as
    /// dispatch-to-final-chunk, and recording it later would let a slow
    /// database read as an upstream generation regression. `ended_at` is
    /// the upstream-side stamp of that terminal event where one exists
    /// (the Done event carries one); error paths pass their observation
    /// time.
    ///
    /// The output rate is measured across the *generation* window (first chunk
    /// to last), not the whole stream: time-to-first-token is already its own
    /// histogram, and leaving prefill in the denominator would make the rate
    /// move for two unrelated reasons at once.
    fn finish(self, reason: &'static str, completion_tokens: Option<u64>, ended_at: Instant) {
        let attrs = self.model_attr();

        crate::telemetry::metrics::CHAT_STREAM_OUTCOME.add(
            1,
            &[
                KeyValue::new("model", self.model_id.clone()),
                KeyValue::new("reason", reason),
            ],
        );
        crate::telemetry::metrics::CHAT_STREAM_DURATION.record(
            ended_at.duration_since(self.dispatched_at).as_secs_f64(),
            &attrs,
        );

        let (gap, rate) = self.observations(completion_tokens);
        if let Some(gap) = gap {
            crate::telemetry::metrics::CHAT_INTER_TOKEN_GAP_MAX.record(gap, &attrs);
        }
        if let Some(rate) = rate {
            crate::telemetry::metrics::CHAT_OUTPUT_RATE.record(rate, &attrs);
        }
    }
}

/// Whether a streamed chunk carries generated output — content, reasoning,
/// or tool-call fragments. The role preamble, the bare finish-reason chunk,
/// and the usage-only final chunk (which we force via `include_usage`) carry
/// none, and must not feed [`StreamTiming`]: they would start the
/// time-to-first-token clock before the first generated token and stretch
/// the generation window the output rate and max-gap instruments measure.
fn carries_output(chunk: &ChatCompletionChunk) -> bool {
    chunk.choices.iter().any(|choice| {
        let delta = &choice.delta;
        delta.content.as_deref().is_some_and(|s| !s.is_empty())
            || delta
                .reasoning_content
                .as_deref()
                .is_some_and(|s| !s.is_empty())
            || delta.reasoning.as_deref().is_some_and(|s| !s.is_empty())
            || delta.tool_calls.as_ref().is_some_and(|t| !t.is_empty())
    })
}

/// Handle a streaming chat completion request.
async fn handle_streaming_request(
    state: AppState,
    validated: &ValidatedRequest,
    act: &ActSpend,
    model: &Model,
    charge_credits: u128,
) -> Result<axum::response::Response, ServerError> {
    let request = validated.request();
    let auth_context = AuthContext {
        method: AuthMethod::AnonymousCredential,
    };

    // Stream timing is measured from upstream dispatch, not from request
    // arrival: everything before this point is our own verification work,
    // which the HTTP-level histogram already covers.
    let dispatched_at = Instant::now();
    let mut upstream_rx = match dispatch_stream(&state, validated, model).await {
        Ok(rx) => rx,
        Err(e) => {
            // Known error — upstream didn't process any tokens. Full refund.
            warn!("Stream start error, issuing full refund: {}", e);
            let refund = issue_refund_async(
                &state,
                &act.spend_proof,
                &act.issuer_key_hash,
                charge_credits,
            )
            .await;
            return Ok(error_response_with_refund(&e, refund.ok()));
        }
    };

    // Events for the client, framed by `padding` into fixed-size writes on a
    // fixed tick (see `padded_sse_response`).
    let (tx, rx) = mpsc::channel::<StreamEvent>(32);

    // Clone/copy values for the spawned task.
    let issuer_key_hash = act.issuer_key_hash;
    // We need to serialize the spend proof for the spawned task.
    let spend_proof_cbor = match act.spend_proof.to_cbor() {
        Ok(cbor) => cbor,
        Err(e) => {
            // Can't serialize spend proof for the spawned task — issue full refund now.
            error!("spend proof re-encode failed: {e:?}");
            let refund = issue_refund_async(
                &state,
                &act.spend_proof,
                &act.issuer_key_hash,
                charge_credits,
            )
            .await;
            let err = ServerError::Internal(format!("spend proof re-encode failed: {e:?}"));
            return Ok(error_response_with_refund(&err, refund.ok()));
        }
    };
    let model_id = model.id.clone();
    let task_model = model.clone();
    // Contract clamp inputs, computed from the request before the task takes
    // over (the spawned task never sees the request itself).
    let chargeable_prompt = chargeable_prompt_tokens_for(request);
    let max_completion = effective_max_completion(request, model);
    let provider = provider_for(model);

    tokio::spawn(async move {
        /// Re-parse the spend proof and issue a refund with the given amount.
        /// Returns None only if the cryptographic operations themselves fail.
        async fn try_refund(
            state: &AppState,
            spend_proof_cbor: &[u8],
            issuer_key_hash: &[u8; 32],
            refund_credits: u128,
        ) -> Option<RefundInfo> {
            let proof = match SpendProof::<128>::from_cbor(spend_proof_cbor) {
                Ok(p) => p,
                Err(e) => {
                    error!("CRITICAL: failed to re-parse spend proof for refund: {e:?}");
                    return None;
                }
            };
            match issue_refund_async(state, &proof, issuer_key_hash, refund_credits).await {
                Ok(info) => Some(info),
                Err(e) => {
                    // No amount in the log: `refund_credits` is a function
                    // of the prompt's chargeable bytes and the upstream's
                    // token counts — the same content-derived quantity the
                    // `PaymentRequired` redaction keeps out of logs.
                    error!(
                        "CRITICAL: failed to issue refund: {}, retrying with zero",
                        e
                    );
                    // Fall back to a zero refund (returns blind remaining value
                    // c - s) so the client doesn't lose the credential entirely.
                    match issue_refund_async(state, &proof, issuer_key_hash, 0).await {
                        Ok(info) => Some(info),
                        Err(e2) => {
                            error!("CRITICAL: failed to issue fallback zero refund: {}", e2);
                            None
                        }
                    }
                }
            }
        }

        /// Send a metadata SSE event containing a refund, then [DONE].
        async fn send_metadata_event(
            tx: &mpsc::Sender<StreamEvent>,
            refund_info: Option<RefundInfo>,
            privacy: crate::response::PrivacyMetadata,
            verification: crate::response::VerificationMetadata,
            chat_id: String,
        ) {
            let stream_meta =
                EidolaStreamMetadata::new(chat_id, privacy, verification, refund_info);
            let json_str = serde_json::to_string(&stream_meta).unwrap();
            let _ = tx.send(StreamEvent::Json(json_str)).await;
            let _ = tx.send(StreamEvent::Done).await;
        }

        let mut final_usage: Option<Usage> = None;
        let mut timing = StreamTiming::new(model_id.clone(), dispatched_at);

        while let Some(event_result) = upstream_rx.recv().await {
            match event_result {
                Ok(BackendStreamEvent::Chunk(chunk, at)) => {
                    if carries_output(&chunk) {
                        timing.on_chunk(at);
                    }
                    // Capture usage from the final chunk if present.
                    if chunk.usage.is_some() {
                        final_usage.clone_from(&chunk.usage);
                    }
                    let json_str = serde_json::to_string(&chunk).unwrap();
                    if tx.send(StreamEvent::Json(json_str)).await.is_err() {
                        // Client disconnected — we were likely billed for tokens
                        // already streamed but don't know how much. Refund 0
                        // (returns blind remaining value c - s). The client can't
                        // receive this, but we issue it for consistency.
                        warn!("Client disconnected mid-stream, issuing zero refund");
                        timing.finish("client_disconnect", None, at);
                        let _ = try_refund(&state, &spend_proof_cbor, &issuer_key_hash, 0).await;
                        return;
                    }
                }
                Ok(BackendStreamEvent::Done(meta, at)) => {
                    let is_tee = meta.tee_type.is_some();

                    // Prefer usage from the final chunk, then from meta.
                    if final_usage.is_none() {
                        final_usage = meta.usage.clone();
                    }

                    timing.finish(
                        "done",
                        final_usage.as_ref().map(|u| u.completion_tokens as u64),
                        at,
                    );

                    // Record token usage metrics (safe: only model + counts).
                    if let Some(usage) = &final_usage {
                        let model_attr = KeyValue::new("model", model_id.clone());
                        crate::telemetry::metrics::CHAT_TOKENS.add(
                            usage.prompt_tokens as u64,
                            &[model_attr.clone(), KeyValue::new("type", "prompt")],
                        );
                        crate::telemetry::metrics::CHAT_TOKENS.add(
                            usage.completion_tokens as u64,
                            &[model_attr, KeyValue::new("type", "completion")],
                        );
                    }

                    let privacy = build_privacy_metadata(&auth_context, is_tee, &meta.provider);
                    let verification = build_verification_metadata(None);

                    // Compute the refund from usage, with the charge clamped
                    // to the pricing contract (same as the blocking path).
                    let cost = settled_cost(
                        final_usage.as_ref(),
                        &task_model,
                        chargeable_prompt,
                        max_completion,
                        charge_credits,
                    );

                    let refund_credits = charge_credits.saturating_sub(cost);
                    let refund_info =
                        try_refund(&state, &spend_proof_cbor, &issuer_key_hash, refund_credits)
                            .await;

                    send_metadata_event(
                        &tx,
                        refund_info,
                        privacy,
                        verification,
                        meta.chat_id.unwrap_or_default(),
                    )
                    .await;
                    return;
                }
                Err(e) => {
                    // Some chunks may have been delivered and billed; we don't
                    // know the actual cost. Refund 0 (blind remaining value).
                    error!("Stream error, issuing zero refund: {}", e);
                    timing.finish("upstream_error", None, Instant::now());
                    let refund_info =
                        try_refund(&state, &spend_proof_cbor, &issuer_key_hash, 0).await;
                    let privacy = build_privacy_metadata(&auth_context, true, provider);
                    let verification = build_verification_metadata(None);
                    send_metadata_event(&tx, refund_info, privacy, verification, String::new())
                        .await;
                    return;
                }
            }
        }

        // upstream_rx closed without a Done event (unexpected). Chunks may
        // have been delivered and billed; we don't know the cost. Refund 0.
        warn!("Upstream channel closed without Done event, issuing zero refund");
        timing.finish("channel_closed", None, Instant::now());
        let refund_info = try_refund(&state, &spend_proof_cbor, &issuer_key_hash, 0).await;
        let privacy = build_privacy_metadata(&auth_context, true, provider);
        let verification = build_verification_metadata(None);
        send_metadata_event(&tx, refund_info, privacy, verification, String::new()).await;
    });

    Ok(padding::padded_sse_response(rx, Cadence::CLIENT_FACING))
}

/// `Json<T>` wrapper that logs the rejection reason at warn level on
/// failure, before returning the same response axum would have returned.
///
/// Why we need this: `Json<T>` rejections fail the request before the
/// handler runs (the extractor runs first), so handler-level logging
/// can't see them. With `#[serde(deny_unknown_fields)]` on
/// `ChatCompletionRequest`, an unrecognized field — for instance a
/// client sending an OpenAI-extension key the server hasn't added —
/// becomes a 422 with no log entry, and the client sees an opaque
/// "(422): unknown error". This wrapper makes those failures visible
/// to operators.
///
/// **Privacy:** the rejection message produced by axum + serde **can
/// echo client-authored body values** — `deny_unknown_fields` quotes the
/// unrecognized field name verbatim, and serde's data errors quote the
/// offending scalar (`invalid type: string "<the whole string>",
/// expected u32`). Only the rejection *class* reaches the log path; the
/// full detail still goes to the client in the rejection response — its
/// own data, over its own attested connection.
///
/// Before any of that, the body's JSON shape is held to the limits an
/// Eidola-hosted engine enforces (`eidola_common::engine_protocol::
/// check_request_json`: at most `MAX_REQUEST_JSON_VALUES` values, nested at
/// most `MAX_REQUEST_JSON_DEPTH` deep), with nothing parsed yet: a body the
/// node would refuse for its size in values is refused here first, with a
/// 400.
pub struct LoggedJson<T>(pub T);

impl<S, T> FromRequest<S> for LoggedJson<T>
where
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = axum::response::Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let (parts, bytes) = checked_body::<T, S>(req, state).await?;
        parse_logged(parts, bytes, state).await.map(Self)
    }
}

/// The body's bytes, held to the JSON shape limits (see [`LoggedJson`]).
// The error is the refusal itself, built once and returned as the extractor's
// rejection.
#[allow(clippy::result_large_err)]
async fn checked_body<T, S: Send + Sync>(
    req: Request,
    state: &S,
) -> Result<(axum::http::request::Parts, axum::body::Bytes), axum::response::Response> {
    let (parts, body) = req.into_parts();
    let bytes = axum::body::Bytes::from_request(Request::from_parts(parts.clone(), body), state)
        .await
        .map_err(|rejection| {
            warn!(
                payload_type = std::any::type_name::<T>(),
                "request body rejected: bytes error"
            );
            rejection.into_response()
        })?;
    // A malformed body goes on to axum's parse, which refuses it as it always has
    // (and, since the scan counts values as it goes, holds no more of them before
    // the error than the cap).
    use eidola_common::engine_protocol::{JsonShapeError, check_request_json};
    if let Err(shape @ (JsonShapeError::TooManyValues | JsonShapeError::TooDeep)) =
        check_request_json(&bytes)
    {
        let class = if shape == JsonShapeError::TooDeep {
            "too deep"
        } else {
            "too many values"
        };
        warn!(
            payload_type = std::any::type_name::<T>(),
            "request body rejected: {class} error"
        );
        return Err(ServerError::BadRequest {
            message: shape.to_string(),
        }
        .into_response());
    }
    Ok((parts, bytes))
}

/// axum's `Json` over `bytes`: its value, or its refusal, logged by class.
#[allow(clippy::result_large_err)]
async fn parse_logged<T, S>(
    parts: axum::http::request::Parts,
    bytes: axum::body::Bytes,
    state: &S,
) -> Result<T, axum::response::Response>
where
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
{
    let req = Request::from_parts(parts, axum::body::Body::from(bytes));
    match Json::<T>::from_request(req, state).await {
        Ok(Json(value)) => Ok(value),
        Err(rejection) => {
            // Class only — the rejection's message can quote body
            // values (see the type-level privacy note).
            let class = match &rejection {
                JsonRejection::JsonDataError(_) => "data",
                JsonRejection::JsonSyntaxError(_) => "syntax",
                JsonRejection::MissingJsonContentType(_) => "missing content-type",
                JsonRejection::BytesRejection(_) => "bytes",
                _ => "other",
            };
            warn!(
                payload_type = std::any::type_name::<T>(),
                "request body rejected: {class} error"
            );
            Err(rejection.into_response())
        }
    }
}

/// The chat request: [`LoggedJson`]'s checks and refusals, keeping the body's
/// bytes with the strict parse. A request bound for an Eidola-hosted engine
/// is forwarded from its bytes (`engine_trust::protocol::ValidatedRequest`),
/// whose only constructor parses exactly those bytes.
///
/// The content type and the JSON syntax are checked first by axum's own
/// extractor over a parse that keeps nothing; a body that fails any check is
/// refused by axum's parse of the strict type, exactly as before.
pub struct ChatRequest(pub ValidatedRequest);

impl<S: Send + Sync> FromRequest<S> for ChatRequest {
    type Rejection = axum::response::Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let (parts, bytes) = checked_body::<ChatCompletionRequest, S>(req, state).await?;
        let shape_ok = Json::<serde::de::IgnoredAny>::from_request(
            Request::from_parts(parts.clone(), axum::body::Body::from(bytes.clone())),
            state,
        )
        .await
        .is_ok();
        if shape_ok && let Ok(validated) = ValidatedRequest::from_bytes(bytes.clone()) {
            return Ok(Self(validated));
        }
        parse_logged::<ChatCompletionRequest, S>(parts, bytes.clone(), state).await?;
        // axum accepted a body the strict parse refused; both parse the same
        // bytes with the same type, so this is not reached.
        ValidatedRequest::from_bytes(bytes).map(Self).map_err(|_| {
            ServerError::BadRequest {
                message: "invalid request body".to_string(),
            }
            .into_response()
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The extractor's result over `body`: the value, or the refusal's status and text.
    async fn extract(body: String) -> Result<serde_json::Value, (axum::http::StatusCode, String)> {
        let req = Request::builder()
            .method("POST")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body))
            .unwrap();
        match LoggedJson::<serde_json::Value>::from_request(req, &()).await {
            Ok(LoggedJson(v)) => Ok(v),
            Err(response) => {
                let status = response.status();
                let text = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                Err((status, String::from_utf8_lossy(&text).into_owned()))
            }
        }
    }

    /// The extractor's result as status and text, for comparing refusals.
    async fn refusal_of(response: axum::response::Response) -> (axum::http::StatusCode, String) {
        let status = response.status();
        let text = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, String::from_utf8_lossy(&text).into_owned())
    }

    fn chat_request(body: &str, content_type: Option<&str>) -> Request {
        let mut builder = Request::builder().method("POST");
        if let Some(content_type) = content_type {
            builder = builder.header("content-type", content_type);
        }
        builder
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    /// The chat extractor keeps the exact body bytes with the strict parse,
    /// and refuses everything the plain extractor refuses, identically.
    #[tokio::test]
    async fn the_chat_extractor_keeps_the_bytes_and_refuses_as_before() {
        let valid = r#"{"model":"m", "messages":[{"role":"user","content":"hi"}]}"#;
        let ChatRequest(validated) =
            ChatRequest::from_request(chat_request(valid, Some("application/json")), &())
                .await
                .unwrap_or_else(|_| panic!("accepted"));
        assert_eq!(validated.raw(), valid.as_bytes());
        assert_eq!(validated.request().model, "m");

        let deep = format!("{}{}", "[".repeat(65), "]".repeat(65));
        let cases: Vec<(String, Option<&str>)> = vec![
            (
                r#"{"model":"m","messages":[],"extra":1}"#.into(),
                Some("application/json"),
            ),
            (
                r#"{"model":"m","messages":"#.into(),
                Some("application/json"),
            ),
            (
                r#"{"model":7,"messages":[]}"#.into(),
                Some("application/json"),
            ),
            (valid.into(), None),
            (valid.into(), Some("text/plain")),
            (deep, Some("application/json")),
        ];
        for (body, content_type) in cases {
            let chat = match ChatRequest::from_request(chat_request(&body, content_type), &()).await
            {
                Ok(_) => panic!("accepted {body:?}"),
                Err(response) => refusal_of(response).await,
            };
            let plain = match LoggedJson::<ChatCompletionRequest>::from_request(
                chat_request(&body, content_type),
                &(),
            )
            .await
            {
                Ok(_) => panic!("the plain extractor accepted {body:?}"),
                Err(response) => refusal_of(response).await,
            };
            assert_eq!(chat, plain, "{body:?} {content_type:?}");
        }
        // A suffixed JSON media type is JSON to both.
        assert!(
            ChatRequest::from_request(chat_request(valid, Some("application/vnd.x+json")), &())
                .await
                .is_ok()
        );
    }

    /// A body past the engine's JSON value or depth limit is refused with a 400 before
    /// anything is parsed; one at the limits, and a malformed one, go on as before.
    #[tokio::test]
    async fn a_body_past_the_json_shape_limits_is_refused() {
        use eidola_common::engine_protocol::{MAX_REQUEST_JSON_DEPTH, MAX_REQUEST_JSON_VALUES};
        let bad = axum::http::StatusCode::BAD_REQUEST;
        let values = |n: usize| format!("[{}0]", "0,".repeat(n - 2));
        assert!(extract(values(MAX_REQUEST_JSON_VALUES)).await.is_ok());
        let (status, text) = extract(values(MAX_REQUEST_JSON_VALUES + 1))
            .await
            .unwrap_err();
        assert_eq!(status, bad);
        assert!(text.contains("JSON values"), "{text}");
        let nested = |d: usize| format!("{}{}", "[".repeat(d), "]".repeat(d));
        assert!(extract(nested(MAX_REQUEST_JSON_DEPTH)).await.is_ok());
        let (status, text) = extract(nested(MAX_REQUEST_JSON_DEPTH + 1))
            .await
            .unwrap_err();
        assert_eq!(status, bad);
        assert!(text.contains("deeper"), "{text}");
        // Malformed JSON is axum's refusal, unchanged.
        let (status, _) = extract("[1,".into()).await.unwrap_err();
        assert_eq!(status, bad);
    }
    use crate::types::{
        Capability, Modality, ModelCapabilities, ModelHosting, ModelPricing, OutputBudgetClass,
        PinnedWeightsCapability, PromptCacheCapability, ScaledPrice,
    };

    /// A token-priced model with easy integer math at `PRICING_SCALE_FACTOR`:
    /// 1 credit per prompt token, 2 credits per completion token.
    pub(crate) fn test_model() -> Model {
        Model {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            description: String::new(),
            context_length: 8192,
            max_output_tokens: Some(4096),
            output_budget_class: OutputBudgetClass::Standard,
            hosting: ModelHosting::Tinfoil,
            capabilities: ModelCapabilities {
                tool_calling: Capability::new(true),
                reasoning: Capability::new(false),
                input_modalities: vec![Modality::Text],
                output_modalities: vec![Modality::Text],
                prompt_cache: PromptCacheCapability::unsupported(),
                pinned_weights: PinnedWeightsCapability::unsupported(),
            },
            pricing: ModelPricing {
                per_prompt_token: ScaledPrice {
                    value: PRICING_SCALE_FACTOR,
                    scale_factor: PRICING_SCALE_FACTOR,
                },
                per_completion_token: ScaledPrice {
                    value: 2 * PRICING_SCALE_FACTOR,
                    scale_factor: PRICING_SCALE_FACTOR,
                },
                per_request: None,
            },
        }
    }

    /// Parse a request from the exact JSON shape a client sends, so byte
    /// counting exercises the real deserialization path.
    fn request(contents: &[&str], max_completion_tokens: u32) -> ChatCompletionRequest {
        let messages: Vec<serde_json::Value> = contents
            .iter()
            .map(|c| serde_json::json!({"role": "user", "content": c}))
            .collect();
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        }))
        .expect("valid request")
    }

    /// Parse a chunk from the exact JSON shape the upstream sends.
    fn chunk(json: serde_json::Value) -> ChatCompletionChunk {
        serde_json::from_value(json).expect("valid chunk")
    }

    #[test]
    fn carries_output_ignores_non_token_chunks() {
        // Role preamble with an empty content string.
        assert!(!carries_output(&chunk(serde_json::json!({
            "id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
        }))));
        // Bare finish-reason chunk.
        assert!(!carries_output(&chunk(serde_json::json!({
            "id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }))));
        // Forced usage-only final chunk (empty choices).
        assert!(!carries_output(&chunk(serde_json::json!({
            "id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        }))));
    }

    #[test]
    fn carries_output_accepts_content_reasoning_and_tool_calls() {
        for delta in [
            serde_json::json!({"content": "Hi"}),
            serde_json::json!({"reasoning_content": "hmm"}),
            serde_json::json!({"reasoning": "hmm"}),
            serde_json::json!({"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]}),
        ] {
            assert!(
                carries_output(&chunk(serde_json::json!({
                    "id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": null}]
                }))),
                "delta should count as output: {delta}"
            );
        }
    }

    /// Drive `StreamTiming` with synthetic socket-read stamps and check the
    /// pure `observations` math — the instruments themselves record to the
    /// global (no-op in tests) meter and aren't asserted here.
    #[test]
    fn stream_observations_scale_to_the_observed_window() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);

        // Per-token streaming, one stamp per chunk: window spans 2 of 3
        // chunks → rate = 3 tokens * 2/3 over 20 ms = 100 tok/s; max gap
        // is the larger interval.
        let mut timing = StreamTiming::new("m".into(), t0);
        timing.on_chunk(at(100));
        timing.on_chunk(at(108));
        timing.on_chunk(at(120));
        let (gap, rate) = timing.observations(Some(3));
        assert_eq!(gap, Some(0.012));
        assert_eq!(rate, Some(100.0));

        // A coalesced burst: 4 chunks share one read, 1 more arrives 30 ms
        // later. The window covers generating 1 of 5 chunks, so the rate is
        // 500 * 1/5 / 0.030 — not 500 * 4/5 / 0.030.
        let mut timing = StreamTiming::new("m".into(), t0);
        for _ in 0..4 {
            timing.on_chunk(at(100));
        }
        timing.on_chunk(at(130));
        let (gap, rate) = timing.observations(Some(500));
        assert_eq!(gap, Some(0.030));
        assert!((rate.unwrap() - 500.0 / 5.0 / 0.030).abs() < 1e-9);
    }

    /// Streams that yield no usable inter-chunk timing must yield no
    /// observation at all — recording would fabricate a perfectly smooth
    /// zero gap and an unbounded rate.
    #[test]
    fn stream_observations_skip_unobservable_windows() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);

        // Everything in one coalesced read: one shared stamp.
        let mut timing = StreamTiming::new("m".into(), t0);
        for _ in 0..3 {
            timing.on_chunk(at(100));
        }
        assert_eq!(timing.observations(Some(50)), (None, None));

        // A single chunk.
        let mut timing = StreamTiming::new("m".into(), t0);
        timing.on_chunk(at(100));
        assert_eq!(timing.observations(Some(50)), (None, None));

        // No chunks at all.
        let timing = StreamTiming::new("m".into(), t0);
        assert_eq!(timing.observations(Some(50)), (None, None));

        // Real timing but zero (or absent) reported tokens: the gap stands,
        // the rate does not — 0 tok/s would read as a stall.
        let mut timing = StreamTiming::new("m".into(), t0);
        timing.on_chunk(at(100));
        timing.on_chunk(at(110));
        assert_eq!(timing.observations(Some(0)), (Some(0.010), None));
        let mut timing = StreamTiming::new("m".into(), t0);
        timing.on_chunk(at(100));
        timing.on_chunk(at(110));
        assert_eq!(timing.observations(None), (Some(0.010), None));
    }

    #[test]
    fn worst_case_cost_uses_the_shared_contract() {
        // One 12-byte message, max 100 completion tokens.
        // chargeable prompt = ceil(12*2/3) + 8*1 + 32 = 8 + 8 + 32 = 48.
        // wc = 48 * 1 credit + 100 * 2 credits = 248.
        let req = request(&["hello world!"], 100);
        assert_eq!(chargeable_prompt_tokens_for(&req), 48);
        assert_eq!(worst_case_cost(&req, &test_model()), 248);
    }

    #[test]
    fn preflight_rejects_hold_below_minimum() {
        let req = request(&["hello world!"], 100);
        let model = test_model();
        let wc = worst_case_cost(&req, &model);

        // One credit short → 402 PaymentRequired, before anything upstream.
        let err = check_sufficient_charge(wc - 1, &req, &model)
            .expect_err("hold below the minimum must be rejected");
        assert!(
            matches!(err, ServerError::PaymentRequired { .. }),
            "got {err:?}"
        );

        // Exactly the minimum → accepted.
        check_sufficient_charge(wc, &req, &model).expect("exact minimum hold is sufficient");
    }

    #[test]
    fn preflight_rejects_old_bytes_as_tokens_hold_for_tiny_messages() {
        // The defect the contract fixes: 40 one-byte messages have 40
        // content bytes, but the chat template adds per-message tokens the
        // old bytes-as-tokens hold never covered. The old-style hold
        // (bytes × prompt_rate + max_completion × completion_rate = 240)
        // must now fail pre-flight instead of under-funding the charge.
        let contents: Vec<&str> = vec!["x"; 40];
        let req = request(&contents, 100);
        let model = test_model();
        let old_style_hold = 40 + 100 * 2;
        assert!(
            check_sufficient_charge(old_style_hold, &req, &model).is_err(),
            "bytes-as-tokens hold must be below the contract minimum"
        );
    }

    #[test]
    fn token_dense_usage_is_clamped_to_the_contract() {
        // Same request as above: chargeable prompt = 48, max completion 100.
        let req = request(&["hello world!"], 100);
        let model = test_model();
        let chargeable = chargeable_prompt_tokens_for(&req);
        let max_completion = effective_max_completion(&req, &model);

        // Upstream reports 1000 actual prompt tokens (> chargeable): the
        // prompt component is charged at the clamp, not the actual count.
        let usage = Usage {
            prompt_tokens: 1000,
            completion_tokens: 50,
            total_tokens: 1050,
        };
        let cost = actual_cost(&usage, &model, chargeable, max_completion);
        assert_eq!(cost, 48 + 50 * 2, "prompt clamped to 48, completion real");

        // And the refund reflects the clamped charge.
        let wc = worst_case_cost(&req, &model);
        assert_eq!(wc.saturating_sub(cost), 248 - 148);
    }

    #[test]
    fn completion_usage_is_clamped_defensively() {
        // The model already stops at max_completion_tokens; a usage report
        // claiming more is clamped anyway.
        let req = request(&["hello world!"], 100);
        let model = test_model();
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 500,
            total_tokens: 510,
        };
        let cost = actual_cost(
            &usage,
            &model,
            chargeable_prompt_tokens_for(&req),
            effective_max_completion(&req, &model),
        );
        assert_eq!(cost, 10 + 100 * 2);
    }

    #[test]
    fn per_request_pricing_ignores_token_clamps() {
        let mut model = test_model();
        model.pricing.per_request = Some(ScaledPrice {
            value: 5 * PRICING_SCALE_FACTOR,
            scale_factor: PRICING_SCALE_FACTOR,
        });
        let req = request(&["hi"], 100);
        assert_eq!(worst_case_cost(&req, &model), 5);
        let usage = Usage {
            prompt_tokens: 1_000_000,
            completion_tokens: 1_000_000,
            total_tokens: 2_000_000,
        };
        assert_eq!(actual_cost(&usage, &model, 1, 1), 5);
    }

    /// Tripwire against client/server drift: the client sizes its hold from
    /// the (role, content) string pairs it sends (summing `content.len()`
    /// and counting messages — see `prepare_turn` in eidola-app-core); the
    /// server recomputes from the parsed `ChatCompletionRequest`. Both must
    /// feed identical inputs into the one shared formula.
    #[test]
    fn client_and_server_prompt_terms_agree() {
        // Multi-byte UTF-8 included: byte length, not char count, is the input.
        let contents = ["hello", "héllo wörld", "日本語のテキスト", ""];

        // Client side: raw content strings, exactly as app-core computes.
        let client_bytes: u64 = contents.iter().map(|c| c.len() as u64).sum();
        let client_term =
            eidola_common::chargeable_prompt_tokens(client_bytes, contents.len() as u64);

        // Server side: the parsed request.
        let req = request(&contents, 100);
        let server_term = chargeable_prompt_tokens_for(&req);

        assert_eq!(client_term, server_term);
    }

    // -----------------------------------------------------------------
    // The pricing contract's tool-calling extension
    // -----------------------------------------------------------------

    /// The cross-crate pricing fixture: one user post, one assistant tool
    /// call, one tool result, and one advertised tool.
    ///
    /// The same logical request is pinned three times — here (the server's
    /// walk over the parsed struct), in `eidola-common`'s
    /// `cross_crate_tool_round_fixture` (the arithmetic), and in
    /// `eidola-app-core`'s `prompt_charge_matches_the_shared_contract_fixture`
    /// (the client's walk over its `serde_json::Value` messages). All three
    /// must produce 230 chargeable prompt tokens; change one and the other
    /// two fail.
    const TOOL_ROUND_FIXTURE: &str = r#"{
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "what is 2+2?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "calc", "arguments": "{\"expr\":\"2+2\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "4"}
        ],
        "tools": [
            {"type": "function", "function": {
                "name": "calc",
                "description": "Evaluate arithmetic.",
                "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}
            }}
        ]
    }"#;

    #[test]
    fn tool_round_fixture_charges_the_pinned_contract_value() {
        let req: ChatCompletionRequest = serde_json::from_str(TOOL_ROUND_FIXTURE).unwrap();
        // Content bytes: 12 ("what is 2+2?") + 0 (null) + 1 ("4") = 13.
        // Tool call:     93 (the whole entry, compact JSON).
        // Tool schema:   154 (compact JSON).
        // ceil(260*2/3) + 8*3 + 32 = 174 + 24 + 32 = 230.
        let call_bytes =
            eidola_common::json_text_bytes(&req.messages[1].tool_calls.as_ref().unwrap()[0]);
        assert_eq!(
            call_bytes, 93,
            "the fixture's call entry must stay 93 bytes"
        );
        let tool_bytes = eidola_common::json_text_bytes(&req.tools.as_ref().unwrap()[0]);
        assert_eq!(tool_bytes, 154, "the fixture's schema must stay 154 bytes");
        assert_eq!(chargeable_prompt_tokens_for(&req), 230);
    }

    #[test]
    fn tool_call_arguments_are_charged() {
        // The margin leak this closes: an assistant message whose whole
        // payload is a tool call has no `content` string at all.
        let with_call: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "assistant", "content": null, "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": "calc", "arguments": "{\"expr\":\"2+2\"}"}}
            ]}],
        }))
        .unwrap();
        let without: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "assistant", "content": null}],
        }))
        .unwrap();

        // The whole 88-byte entry ⇒ ceil(88*2/3) = 59 more tokens.
        assert_eq!(
            chargeable_prompt_tokens_for(&with_call) - chargeable_prompt_tokens_for(&without),
            59
        );
    }

    #[test]
    fn tool_call_framing_bytes_are_charged() {
        // The proxy forwards the *whole* call entry upstream, and the chat
        // template — which the proxy cannot see — decides what it renders
        // into the prompt. Mistral's renders the call id; Qwen's and
        // Llama's don't. Charging only `function.{name,arguments}` would let
        // the id, the `type`, and any provider extension ride free while
        // still counting toward upstream `prompt_tokens`, which the clamp
        // then discards. So the whole entry is measured.
        let with_long_id = |id: &str| -> ChatCompletionRequest {
            serde_json::from_value(serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "assistant", "content": null, "tool_calls": [
                    {"id": id, "type": "function",
                     "function": {"name": "calc", "arguments": "{}"}}
                ]}],
            }))
            .unwrap()
        };

        let short = with_long_id("c");
        let realistic = with_long_id("call_9tK2mQz7Xb4LpR1vN8sYcE0d");
        assert!(
            chargeable_prompt_tokens_for(&realistic) > chargeable_prompt_tokens_for(&short),
            "a longer call id is more forwarded bytes and must cost more"
        );

        // A provider extension field on the entry counts too — it is
        // forwarded verbatim by the same pass-through rule.
        let with_extension: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "assistant", "content": null, "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": "calc", "arguments": "{}"},
                 "provider_extra": {"trace": "0123456789abcdef"}}
            ]}],
        }))
        .unwrap();
        assert!(
            chargeable_prompt_tokens_for(&with_extension) > chargeable_prompt_tokens_for(&short),
            "a forwarded provider extension must not ride free"
        );
    }

    #[test]
    fn advertised_tool_schemas_are_charged() {
        let base = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        });
        let mut with_tools = base.clone();
        with_tools["tools"] = serde_json::json!([
            {"type": "function", "function": {
                "name": "calc",
                "description": "Evaluate arithmetic.",
                "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}
            }}
        ]);

        let plain: ChatCompletionRequest = serde_json::from_value(base).unwrap();
        let tooled: ChatCompletionRequest = serde_json::from_value(with_tools).unwrap();

        // 2 content bytes alone ⇒ ceil(4/3) = 2 byte-term tokens; with the
        // 154-byte schema ⇒ ceil(312/3) = 104. The schema costs 102 tokens.
        assert_eq!(
            chargeable_prompt_tokens_for(&tooled) - chargeable_prompt_tokens_for(&plain),
            102
        );
    }

    #[test]
    fn tool_result_content_is_charged_once_as_ordinary_content() {
        // `role: "tool"` content is an ordinary content string — pinned so
        // the tool-call accounting can never start double-counting it.
        let as_tool: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "tool", "tool_call_id": "c", "content": "a result"}],
        }))
        .unwrap();
        let as_user: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "a result"}],
        }))
        .unwrap();
        assert_eq!(
            chargeable_prompt_tokens_for(&as_tool),
            chargeable_prompt_tokens_for(&as_user)
        );
    }

    #[test]
    fn a_tool_less_request_charges_exactly_what_it_did_before() {
        // Regression guard for the overwhelming majority of traffic: no
        // `tools`, no `tool_calls` ⇒ the pre-tool-calling value, unchanged.
        let req = request(&["hello world!"], 100);
        assert_eq!(chargeable_prompt_tokens_for(&req), 48);
    }

    #[test]
    fn the_clamp_rises_with_the_tool_bytes_it_charges() {
        // The contract's third role: `charged_prompt = min(actual,
        // chargeable)`. The tools schema really is in the upstream prompt,
        // so charging its bytes must also *raise* the cap — otherwise the
        // server would keep clamping real tool-round usage down to a
        // content-only ceiling and eat the difference.
        let req: ChatCompletionRequest = serde_json::from_str(TOOL_ROUND_FIXTURE).unwrap();
        let model = test_model();
        let chargeable = chargeable_prompt_tokens_for(&req);
        assert_eq!(chargeable, 230);

        // Upstream reports 150 real prompt tokens — above a content-only
        // ceiling (13 bytes ⇒ ceil(26/3) + 24 + 32 = 65), below the extended
        // one, so it is now charged in full instead of being clamped away.
        let usage = Usage {
            prompt_tokens: 150,
            completion_tokens: 10,
            total_tokens: 160,
        };
        let cost = actual_cost(&usage, &model, chargeable, 100);
        assert_eq!(cost, 150 + 10 * 2);
    }
}
