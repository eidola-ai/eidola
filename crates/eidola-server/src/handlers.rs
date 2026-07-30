//! Top-level HTTP handlers: health, models, chat completions.

use std::convert::Infallible;

use anonymous_credit_tokens::{Scalar, SpendProof, credit_to_scalar, scalar_to_credit};
use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request, State};
use axum::response::IntoResponse;
use axum::response::Sse;
use axum::response::sse::{Event, KeepAlive};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use opentelemetry::KeyValue;
use rand_core::OsRng;
use serde::de::DeserializeOwned;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, warn};

use crate::AppState;
use crate::auth::{ActSpend, AuthContext, AuthMethod, TokenAuth};
use crate::backend::{BackendStreamEvent, ChatBackend, PRICING_SCALE_FACTOR};
use crate::credentials;
use crate::db;
use crate::error::ServerError;
use crate::response::{
    EidolaResponse, EidolaStreamMetadata, RefundInfo, build_privacy_metadata,
    build_verification_metadata,
};
use crate::types::{ChatCompletionRequest, ErrorResponse, Model, ModelsResponse, Usage};

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

/// Byte length of a JSON value measured as text: its own bytes when it is a
/// string (the OpenAI shape for `arguments`), otherwise its compact JSON
/// serialization.
///
/// Shared rule with the client (`eidola-app-core`'s `json_text_bytes`) — see
/// [`chargeable_prompt_tokens_for`].
fn json_text_bytes(value: &serde_json::Value) -> u64 {
    match value.as_str() {
        Some(s) => s.len() as u64,
        None => serde_json::to_string(value).map(|s| s.len()).unwrap_or(0) as u64,
    }
}

/// Extract the shared pricing contract's inputs from a request and compute
/// its chargeable prompt tokens.
///
/// The inputs are gathered through [`eidola_common::PromptCharge`], which
/// defines *what counts*: every message (its per-message allowance plus its
/// `content` bytes), every assistant `tool_calls` entry (its compact JSON
/// serialization), and every advertised `tools` entry (likewise). The client
/// walks its own request shape through the same API and must arrive at the
/// same numbers — the tripwire is
/// `client_and_server_prompt_terms_agree` below.
fn chargeable_prompt_tokens_for(request: &ChatCompletionRequest) -> u64 {
    let mut charge = eidola_common::PromptCharge::new();

    for message in &request.messages {
        charge.add_message(message.content_byte_len() as u64);
        for call in message.tool_calls.iter().flatten() {
            // The *whole* entry: the proxy forwards it verbatim (id, type
            // and provider extensions included) and cannot know which of
            // those fields the upstream chat template renders into the
            // prompt, so it charges what it forwards.
            charge.add_tool_call(json_text_bytes(call));
        }
    }

    // The advertised tool schemas: re-injected into the prompt by the chat
    // template on every round, so charged once per request.
    for tool in request.tools.iter().flatten() {
        charge.add_tool_definition(json_text_bytes(tool));
    }

    charge.chargeable_prompt_tokens()
}

/// The effective completion-token ceiling for a request: its
/// `max_completion_tokens`, falling back to the model's context length.
fn effective_max_completion(request: &ChatCompletionRequest, model: &Model) -> u64 {
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
                message: format!("unknown model: {}", request.model),
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
        (status = 502, description = "Upstream provider error", body = ErrorResponse)
    )
)]
pub async fn chat_completions(
    TokenAuth(act): TokenAuth,
    State(state): State<AppState>,
    LoggedJson(request): LoggedJson<ChatCompletionRequest>,
) -> Result<axum::response::Response, ServerError> {
    // Phase 1: Verify the ACT cryptographically. Errors here mean the token
    // is invalid/malformed — no nullifier recorded, no refund needed.
    verify_spend_proof(&state, &act).await?;

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
    let (model, charge_credits) = match validate_request(&state, &act, &request) {
        Ok(v) => v,
        Err(e) => {
            // Decode charge for the refund. If this also fails, fall back to
            // zero refund (returns blind remaining value c - s).
            let refund_credits = scalar_to_credit::<128>(&act.spend_proof.charge()).unwrap_or(0);
            warn!("Request validation failed after nullifier recorded, issuing full refund: {e}");
            let refund = issue_refund_async(
                &state,
                &act.spend_proof,
                &act.issuer_key_hash,
                refund_credits,
            )
            .await;
            return Ok(error_response_with_refund(&e, refund.ok()));
        }
    };

    // Phase 4: Handle the request.
    let result = if request.stream {
        handle_streaming_request(state, &request, &act, &model, charge_credits).await
    } else {
        handle_non_streaming_request(&state, &request, &act, &model, charge_credits).await
    };

    let status = if result.is_ok() { "ok" } else { "error" };
    crate::telemetry::metrics::CHAT_REQUESTS.add(
        1,
        &[
            KeyValue::new("model", request.model.clone()),
            KeyValue::new("stream", if request.stream { "true" } else { "false" }),
            KeyValue::new("status", status),
        ],
    );

    result
}

/// Handle a non-streaming chat completion request.
async fn handle_non_streaming_request(
    state: &AppState,
    request: &ChatCompletionRequest,
    act: &ActSpend,
    model: &Model,
    charge_credits: u128,
) -> Result<axum::response::Response, ServerError> {
    let auth_context = AuthContext {
        method: AuthMethod::AnonymousCredential,
    };

    // Make the backend request. On error, issue a full refund.
    let backend_response = match state.backend.send(request).await {
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
    let cost = backend_response
        .meta
        .usage
        .as_ref()
        .map(|u| actual_cost(u, model, chargeable_prompt, max_completion))
        .unwrap_or(charge_credits); // No usage → charge worst case

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

/// Handle a streaming chat completion request.
async fn handle_streaming_request(
    state: AppState,
    request: &ChatCompletionRequest,
    act: &ActSpend,
    model: &Model,
    charge_credits: u128,
) -> Result<axum::response::Response, ServerError> {
    let auth_context = AuthContext {
        method: AuthMethod::AnonymousCredential,
    };

    let mut upstream_rx = match state.backend.send_stream(request).await {
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

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);

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
                    error!(
                        "CRITICAL: failed to issue refund ({} credits): {}, retrying with zero",
                        refund_credits, e
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
            tx: &mpsc::Sender<Result<Event, Infallible>>,
            refund_info: Option<RefundInfo>,
            privacy: crate::response::PrivacyMetadata,
            verification: crate::response::VerificationMetadata,
            chat_id: String,
        ) {
            let stream_meta =
                EidolaStreamMetadata::new(chat_id, privacy, verification, refund_info);
            let json_str = serde_json::to_string(&stream_meta).unwrap();
            let event = Event::default().data(json_str);
            let _ = tx.send(Ok(event)).await;
            let done_event = Event::default().data("[DONE]");
            let _ = tx.send(Ok(done_event)).await;
        }

        let mut final_usage: Option<Usage> = None;

        while let Some(event_result) = upstream_rx.recv().await {
            match event_result {
                Ok(BackendStreamEvent::Chunk(chunk)) => {
                    // Capture usage from the final chunk if present.
                    if chunk.usage.is_some() {
                        final_usage.clone_from(&chunk.usage);
                    }
                    let json_str = serde_json::to_string(&chunk).unwrap();
                    let event = Event::default().data(json_str);
                    if tx.send(Ok(event)).await.is_err() {
                        // Client disconnected — we were likely billed for tokens
                        // already streamed but don't know how much. Refund 0
                        // (returns blind remaining value c - s). The client can't
                        // receive this, but we issue it for consistency.
                        warn!("Client disconnected mid-stream, issuing zero refund");
                        let _ = try_refund(&state, &spend_proof_cbor, &issuer_key_hash, 0).await;
                        return;
                    }
                }
                Ok(BackendStreamEvent::Done(meta)) => {
                    let is_tee = meta.tee_type.is_some();

                    // Prefer usage from the final chunk, then from meta.
                    if final_usage.is_none() {
                        final_usage = meta.usage.clone();
                    }

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
                    let cost = final_usage
                        .as_ref()
                        .map(|u| actual_cost(u, &task_model, chargeable_prompt, max_completion))
                        .unwrap_or(charge_credits);

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
                    let refund_info =
                        try_refund(&state, &spend_proof_cbor, &issuer_key_hash, 0).await;
                    let privacy = build_privacy_metadata(&auth_context, true, "tinfoil");
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
        let refund_info = try_refund(&state, &spend_proof_cbor, &issuer_key_hash, 0).await;
        let privacy = build_privacy_metadata(&auth_context, true, "tinfoil");
        let verification = build_verification_metadata(None);
        send_metadata_event(&tx, refund_info, privacy, verification, String::new()).await;
    });

    let stream = ReceiverStream::new(rx);
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
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
/// **Privacy:** the rejection error message produced by axum + serde is
/// **structural only** — it names the offending field and the kind of
/// error ("unknown field `foo`", "missing field `model`", "expected u32
/// at line N column M") and does not include the user's prompt or any
/// other request body content. We log only that error string. Body
/// bytes are owned by axum's extractor and never reach the log path.
pub struct LoggedJson<T>(pub T);

impl<S, T> FromRequest<S> for LoggedJson<T>
where
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = JsonRejection;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(Self(value)),
            Err(rejection) => {
                // Field-shape diagnostic only — the source error from
                // axum/serde names the field but never echoes the body
                // value. Safe to log.
                warn!(
                    target = std::any::type_name::<T>(),
                    "request body rejected: {rejection}"
                );
                Err(rejection)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelPricing, ScaledPrice};

    /// A token-priced model with easy integer math at `PRICING_SCALE_FACTOR`:
    /// 1 credit per prompt token, 2 credits per completion token.
    fn test_model() -> Model {
        Model {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            description: String::new(),
            context_length: 8192,
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
        let call_bytes = json_text_bytes(&req.messages[1].tool_calls.as_ref().unwrap()[0]);
        assert_eq!(
            call_bytes, 93,
            "the fixture's call entry must stay 93 bytes"
        );
        let tool_bytes = json_text_bytes(&req.tools.as_ref().unwrap()[0]);
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
