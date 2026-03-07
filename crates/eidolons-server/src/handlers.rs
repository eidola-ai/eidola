//! Top-level HTTP handlers: health, models, chat completions.

use std::convert::Infallible;

use anonymous_credit_tokens::{Scalar, SpendProof, credit_to_scalar, scalar_to_credit};
use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::response::Sse;
use axum::response::sse::{Event, KeepAlive};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rand_core::OsRng;
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
    EidolonsResponse, EidolonsStreamMetadata, RefundInfo, build_privacy_metadata,
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

/// Compute the worst-case cost in credits for a request.
///
/// Uses 1-byte-per-token estimate for prompt size plus `max_completion_tokens`
/// (or context_length) for completion, each at their respective per-token rate.
fn worst_case_cost(request: &ChatCompletionRequest, model: &Model) -> u128 {
    let sf = PRICING_SCALE_FACTOR as u128;

    // Prompt: estimate 1 token per byte of message content.
    let prompt_bytes: usize = request.messages.iter().map(|m| m.content.byte_len()).sum();
    let prompt_rate = model.pricing.per_prompt_token.value as u128;
    let prompt_credits = (prompt_bytes as u128 * prompt_rate).div_ceil(sf);

    // Completion: use max_completion_tokens or fall back to context_length.
    let max_completion = request
        .max_completion_tokens
        .map(|t| t as u64)
        .unwrap_or(model.context_length);
    let completion_rate = model.pricing.per_completion_token.value as u128;
    let completion_credits = (max_completion as u128 * completion_rate).div_ceil(sf);

    prompt_credits + completion_credits
}

/// Compute the actual cost in credits from usage data.
fn actual_cost(usage: &Usage, model: &Model) -> u128 {
    let sf = PRICING_SCALE_FACTOR as u128;
    let prompt_cost = usage.prompt_tokens as u128 * model.pricing.per_prompt_token.value as u128;
    let completion_cost =
        usage.completion_tokens as u128 * model.pricing.per_completion_token.value as u128;
    // Ceiling division for each component, then sum
    let prompt_credits = prompt_cost.div_ceil(sf);
    let completion_credits = completion_cost.div_ceil(sf);
    prompt_credits + completion_credits
}

/// Issue a refund token, returning `refund_credits` to the client.
///
/// `refund_credits` is the number of credits to return (i.e., the `t` parameter
/// in the ACT spec — the resulting token will have `c - s + t` credits).
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

    Ok(RefundInfo {
        refund: URL_SAFE_NO_PAD.encode(&refund_cbor),
        issuer_key_id: hex::encode(issuer_key_hash),
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

/// Verify the spend proof, validate the model and pricing, check the charge
/// amount, and record the nullifier. Returns the model info and charge amount.
///
/// After this function returns `Ok`, the nullifier has been recorded and a
/// refund MUST be issued to the client on any subsequent error.
async fn authorize_spend(
    state: &AppState,
    act: &ActSpend,
    request: &ChatCompletionRequest,
) -> Result<(Model, u128), ServerError> {
    let master_key = state.credential_master_key.as_ref().ok_or_else(|| {
        ServerError::ServiceUnavailable("credential verification is not configured".to_string())
    })?;

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
    {
        let cache = state.credential_key_cache.read().await;
        let key = cache.get(&act.issuer_key_hash).ok_or_else(|| {
            ServerError::Internal("issuer key evicted from cache unexpectedly".to_string())
        })?;

        // Check that the spend proof's ctx matches the expected request_context scalar.
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
    }

    // Look up the model and validate pricing.
    let model = state
        .backend
        .lookup_model(&request.model)
        .await?
        .ok_or_else(|| ServerError::BadRequest {
            message: format!("unknown model: {}", request.model),
        })?;

    // Decode the charge amount from the spend proof.
    let charge_credits = scalar_to_credit::<128>(&act.spend_proof.charge()).map_err(|_| {
        ServerError::BadRequest {
            message: "invalid charge amount in spend proof".to_string(),
        }
    })?;

    // Check that the charge covers the worst-case cost.
    let wc = worst_case_cost(request, &model);
    if charge_credits < wc {
        return Err(ServerError::PaymentRequired {
            message: format!(
                "insufficient charge: {} credits provided, {} required (worst case)",
                charge_credits, wc
            ),
            available: charge_credits as i64,
        });
    }

    // Atomically record the nullifier.
    let key_id = hex::encode(act.issuer_key_hash);
    let nullifier = act.spend_proof.nullifier();
    let nullifier_bytes = nullifier.as_bytes().to_vec();
    let recorded = db::record_nullifier(&state.db_pool, &key_id, &nullifier_bytes).await?;
    if !recorded {
        return Err(ServerError::Conflict {
            message: "credential already spent (duplicate nullifier)".to_string(),
        });
    }

    Ok((model, charge_credits))
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
        (status = 200, description = "Chat completion response with privacy and verification metadata", body = EidolonsResponse),
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
    Json(request): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ServerError> {
    let (model, charge_credits) = authorize_spend(&state, &act, &request).await?;

    // --- POINT OF NO RETURN: nullifier is recorded ---
    // From here on, we MUST issue a refund on any error.

    if request.stream {
        handle_streaming_request(state, &request, &act, &model, charge_credits).await
    } else {
        handle_non_streaming_request(&state, &request, &act, &model, charge_credits).await
    }
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

    // Compute actual cost and refund.
    let cost = backend_response
        .meta
        .usage
        .as_ref()
        .map(|u| actual_cost(u, model))
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

    let eidolons_response = EidolonsResponse::from_completion(
        backend_response.response,
        privacy,
        verification,
        refund_info,
    );

    Ok(Json(eidolons_response).into_response())
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
    let model_pricing = model.pricing.clone();

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
                        "CRITICAL: failed to issue refund ({} credits): {}",
                        refund_credits, e
                    );
                    None
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
                EidolonsStreamMetadata::new(chat_id, privacy, verification, refund_info);
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

                    // Compute refund based on usage.
                    let cost = final_usage
                        .as_ref()
                        .map(|u| {
                            let sf = PRICING_SCALE_FACTOR as u128;
                            let pc = u.prompt_tokens as u128
                                * model_pricing.per_prompt_token.value as u128;
                            let cc = u.completion_tokens as u128
                                * model_pricing.per_completion_token.value as u128;
                            pc.div_ceil(sf) + cc.div_ceil(sf)
                        })
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
                    let privacy = build_privacy_metadata(&auth_context, false, "redpill");
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
        let privacy = build_privacy_metadata(&auth_context, false, "redpill");
        let verification = build_verification_metadata(None);
        send_metadata_event(&tx, refund_info, privacy, verification, String::new()).await;
    });

    let stream = ReceiverStream::new(rx);
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}
