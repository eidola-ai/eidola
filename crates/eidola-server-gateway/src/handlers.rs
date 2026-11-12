//! Top-level HTTP handlers: health, models, chat completions.

use std::time::{Duration, Instant, SystemTime};

use anonymous_credit_tokens::scalar_to_credit;
use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request, State};
use axum::response::IntoResponse;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use eidola_common::spend_credentials::{ACT_L, partial_returns};
use opentelemetry::KeyValue;
use serde::de::DeserializeOwned;
use tokio::sync::mpsc;
use tracing::{error, warn};

use crate::AppState;
use crate::auth::{ActSpend, AuthContext, AuthMethod};
use crate::backend::{BackendStreamEvent, PRICING_SCALE_FACTOR, ResolvedModel};
use crate::credentials::{self, KeyUse, RefundStage, SignedRefund};
use crate::engine_router::assemble::Assembler;
use crate::engine_trust::protocol::ValidatedRequest;
use crate::error::ServerError;
use crate::padding::{self, Cadence, StreamEvent};
use crate::response::{
    EidolaResponse, EidolaStreamMetadata, RefundInfo, build_privacy_metadata,
    build_verification_metadata,
};
use crate::types::{
    ChatCompletionChunk, ChatCompletionRequest, ErrorResponse, Model, ModelsResponse, Usage,
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
///
/// One entry per weights hash, with its tags, source, per-file manifest,
/// lifecycle and the engine deployments that serve it (`docs/model-identity.md`).
#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "Public",
    responses(
        (status = 200, description = "List of available models", body = ModelsResponse)
    )
)]
pub async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(state.catalog.list())
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
/// `max_completion_tokens`, falling back to the model's declared
/// `max_output_tokens`, and to its context length only for a row that
/// declares none.
///
/// A request that reaches pricing never has the field absent when its model
/// declares a ceiling: `chat_completions` writes the declared ceiling into a
/// request that named none ([`ValidatedRequest::default_completion_ceiling`]),
/// so the node is asked for exactly what was priced. The fallbacks keep this
/// total for any caller.
pub(crate) fn effective_max_completion(request: &ChatCompletionRequest, model: &Model) -> u64 {
    request
        .max_completion_tokens
        .map(|t| t as u64)
        .or(model.max_output_tokens)
        .unwrap_or(model.context_length)
}

/// Refuse a request asking for more completion than the model's declared
/// ceiling.
///
/// **Refused, not clamped.** The client sized its hold on what it sent, and
/// the hold, the server's pre-flight minimum and the settlement cap are all
/// the same function of that request; quietly lowering the field would
/// answer a different request from the one priced and leave the caller
/// unable to tell why its answer stopped short. A refusal after the
/// nullifier is recorded refunds the whole charge, as every validation
/// refusal does, and is padded like every other response. With the ceiling
/// written into a request that named none, every accepted request's
/// completion side is the declared ceiling at most, whoever the client is.
fn check_completion_ceiling(
    request: &ChatCompletionRequest,
    model: &Model,
) -> Result<(), ServerError> {
    match (request.max_completion_tokens, model.max_output_tokens) {
        (Some(asked), Some(declared)) if u64::from(asked) > declared => {
            Err(ServerError::BadRequest {
                message: "max_completion_tokens exceeds this model's max_output_tokens".to_string(),
            })
        }
        _ => Ok(()),
    }
}

/// Compute the worst-case cost in credits for a request — the pre-flight
/// minimum hold.
///
/// For a model priced per request, returns the flat per-request price. For token-based models, the prompt side is the shared client/server
/// pricing contract (`eidola_common::chargeable_prompt_tokens`: a content-byte
/// term at the safe cost factor plus per-message and per-request constants),
/// and the completion side uses `max_completion_tokens` (or context_length).
/// The client sizes its hold with the identical function of the identical
/// request, so the client and server go/no-go decisions agree bit-for-bit.
pub(crate) fn worst_case_cost(request: &ChatCompletionRequest, model: &Model) -> u128 {
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

/// What a request that ended early costs: the prompt and the output the engine
/// had generated by the last chunk the gateway received from it, priced and
/// clamped to the contract like any usage ([`actual_cost`]); before any chunk
/// arrived, the prompt alone (at the contract's cap, the most it may be
/// charged), since the engine has begun on it.
///
/// **Generated, not delivered.** The count is the engine's running one, which
/// rides on every delta chunk the gateway receives; it covers output still in
/// the gateway's buffers toward a client that has gone. A client cannot make
/// output it caused free by leaving: the engine is released at once, and what
/// it produced up to then is charged. What the engine produced after its last
/// chunk reached the gateway (at most the steps in flight before the cancel
/// reaches it) is not counted: the gateway's count is the one it can vouch
/// for.
pub(crate) fn early_cost(
    usage: Option<&Usage>,
    model: &Model,
    chargeable_prompt_tokens: u64,
    max_completion_tokens: u64,
) -> u128 {
    let prompt_only = Usage {
        prompt_tokens: u32::try_from(chargeable_prompt_tokens).unwrap_or(u32::MAX),
        completion_tokens: 0,
        total_tokens: 0,
    };
    actual_cost(
        usage.unwrap_or(&prompt_only),
        model,
        chargeable_prompt_tokens,
        max_completion_tokens,
    )
}

/// What a completed request costs: its usage priced and clamped to the
/// contract ([`actual_cost`]), or the whole charge when the upstream reported
/// no usage. One rule for both transports.
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

/// Send a request to the engines that serve its model. Every request is a
/// stream to the engine, whatever its client asked for: a blocking answer is
/// assembled here from the stream (`engine_router::assemble`).
async fn dispatch_stream(
    state: &AppState,
    request: &ValidatedRequest,
) -> Result<mpsc::Receiver<Result<BackendStreamEvent, ServerError>>, ServerError> {
    engines(state)?.send_stream(request).await
}

/// The engine router. Every catalog row is pinned, and a gateway pinning any
/// model starts with a router unless its deployment supplies no
/// `ENGINE_TOKEN`; without one nothing can be routed, and the request is
/// refused as unavailable.
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
/// The refund token is also stored in the nullifier row, sealed under the
/// client's refund recovery key (`act.refund_key`), so the client can recover
/// it via `POST /v1/credentials/refund` if the response is lost.
///
/// The refund is signed under the spent credential's own key
/// (`credentials::sign_refund` takes the key from `act`, and nothing else).
pub(crate) async fn issue_refund_async(
    state: &AppState,
    act: &ActSpend,
    refund_credits: u128,
) -> Result<RefundInfo, ServerError> {
    let (refund_cbor, key_id) = {
        let cache = state.credential_key_cache.read().await;
        credentials::sign_refund(&cache, act, refund_credits, SystemTime::now())?
    };

    // Best-effort store in DB for client recovery, sealed under the client's
    // key: the token carries the refund amount, and Postgres is outside the
    // enclave. Failure here is not fatal — the refund is still returned in the
    // response.
    if let Err(e) = credentials::store_sealed_refund(state, act, &key_id, &refund_cbor).await {
        warn!("Failed to store refund token for recovery: {e}");
    }

    Ok(RefundInfo {
        refund: URL_SAFE_NO_PAD.encode(&refund_cbor),
        issuer_key_id: key_id,
    })
}

/// The longest this gateway lets a paid request run: a request still running
/// then is ended, and settled as a request that ended early. A resource bound
/// on the gateway's own work (an engine slot, a connection, a task), not part
/// of the payment contract: whether a provisional refund is final is decided
/// by whether a request is still being served (`credentials::LiveSpends`),
/// not by the clock. Thirty minutes is far past any completion a catalogued
/// model's declared ceiling allows at the engine's speed, so only a stalled
/// request or a client that will not read reaches it.
pub(crate) const MAX_REQUEST_DURATION: Duration = Duration::from_secs(30 * 60);

/// How long past its deadline (or, for a stream cut off there, past the cut)
/// a stream offers its terminal metadata to a client that is not reading: the
/// refund is already stored, so this bounds only how long the task waits to
/// hand it over in band, on every ending.
const DEADLINE_DELIVERY_GRACE: Duration = Duration::from_secs(5);

/// How long a stream's client may take nothing at all before the stream is
/// ended as if it had gone.
///
/// The client-facing stream writes a frame every tick (50 ms) while its
/// client reads, so this is a client, and every buffer between it and this
/// server, that has accepted nothing for that long. A client that stops
/// reading but keeps its connection open would otherwise hold its engine's
/// admission slot to the request's deadline, at the price of what was
/// generated before the backpressure reached the engine: a few cheap requests
/// could hold every slot. A minute is long past a reader's hiccups (TCP
/// retransmissions, a brief loss of signal); the cost of a shorter one is a
/// paused reader's answer cut off, settled on what was generated like any
/// early ending.
pub(crate) const CLIENT_STALL_LIMIT: Duration = Duration::from_secs(60);

/// The answer to a request ended at its deadline
/// ([`MAX_REQUEST_DURATION`] after its nullifier was recorded).
fn request_deadline_error() -> ServerError {
    ServerError::Backend {
        status: 504,
        error_type: "timeout".to_string(),
        message: "the request reached its deadline and was ended".to_string(),
    }
}

/// Build an HTTP error response that carries a request's refunds, one per
/// input in input order.
fn error_response_with_refunds(
    error: &ServerError,
    refunds: Vec<RefundInfo>,
) -> axum::response::Response {
    let status = error.status_code();
    let mut body = error.to_error_response();
    body.refunds = refunds
        .into_iter()
        .map(|r| serde_json::to_value(r).unwrap())
        .collect();
    (status, Json(body)).into_response()
}

/// The inputs of one paid request, in the order it presented them, each with
/// the provisional refund it was recorded with, and the share each one holds.
#[derive(Clone)]
pub(crate) struct PaidInputs {
    acts: Vec<ActSpend>,
    shares: Vec<u128>,
}

impl PaidInputs {
    /// The request's hold: the sum of its shares, already checked not to
    /// overflow ([`check_presented_spends`]).
    fn hold(&self) -> u128 {
        self.shares.iter().sum()
    }
}

/// Settle a request charged `charge` across its inputs: the charge, clamped
/// to the hold, is consumed in input order
/// (`eidola_common::spend_credentials::partial_returns`), and each input's
/// refund is signed under its own key and stored over its own provisional
/// refund. A charge of nothing returns every share whole; a charge of the
/// hold returns nothing but each credential's untouched balance `c − s`.
///
/// An input whose refund cannot be signed is retried alone with `t = 0`,
/// which still returns its untouched balance; its partial return never moves
/// to another input, which would cross keys. The refunds come back in input
/// order, or none at all if any input's could not be signed even at `t = 0`:
/// `refunds[i]` answers input `i`, so a gap is never closed up. Every refund
/// that was signed is stored, and the client recovers each input by its own
/// nullifier.
pub(crate) async fn settle_inputs(
    state: &AppState,
    inputs: &PaidInputs,
    charge: u128,
) -> Vec<RefundInfo> {
    let returns = partial_returns(&inputs.shares, charge.min(inputs.hold()))
        .unwrap_or_else(|| vec![0; inputs.shares.len()]);
    let mut refunds = Vec::with_capacity(inputs.acts.len());
    let mut complete = true;
    for (act, t) in inputs.acts.iter().zip(returns) {
        match issue_refund_async(state, act, t).await {
            Ok(info) => refunds.push(info),
            // No amount in the log: `t` is a function of the prompt's
            // chargeable bytes and the upstream's token counts, the same
            // content-derived quantity the `PaymentRequired` redaction keeps
            // out of logs.
            Err(e) if t > 0 => {
                error!("CRITICAL: failed to issue refund: {e}, retrying with zero");
                match issue_refund_async(state, act, 0).await {
                    Ok(info) => refunds.push(info),
                    Err(e2) => {
                        error!("CRITICAL: failed to issue fallback zero refund: {e2}");
                        complete = false;
                    }
                }
            }
            Err(e) => {
                error!("CRITICAL: failed to issue zero refund: {e}");
                complete = false;
            }
        }
    }
    if complete { refunds } else { Vec::new() }
}

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

/// The cheap checks of one presented spend, before any proof is verified:
/// the challenge digest, the key it names (loaded, and still accepting
/// spends), the proof's request context against that key's, and a share of
/// at least one credit, which it returns. A zero share pays for nothing and
/// is a degree of freedom a client could be recognised by, so it is refused.
async fn check_spend_statics(state: &AppState, act: &ActSpend) -> Result<u128, ServerError> {
    let master_key = &state.credential_master_key;

    // Verify the challenge_digest matches our expected TokenChallenge.
    let expected_digest = credentials::compute_challenge_digest();
    if act.challenge_digest != expected_digest {
        return Err(ServerError::Unauthorized {
            message: "invalid challenge_digest in token".to_string(),
        });
    }

    // Ensure the issuer key is loaded into the cache and still accepting
    // spends. The deadline is checked again under the lock this proof is
    // verified under, before the nullifier is recorded.
    credentials::load_key(
        &state.credential_key_cache,
        master_key,
        &state.db_pool,
        &act.issuer_key_hash,
        KeyUse::Spend,
    )
    .await?;

    // Verify the spend proof's request_context matches what we expect.
    {
        let cache = state.credential_key_cache.read().await;
        let key = credentials::usable_key(
            &cache,
            &act.issuer_key_hash,
            KeyUse::Spend,
            SystemTime::now(),
        )?;
        credentials::check_presented_context(key, act)?;
    }

    let share = scalar_to_credit::<ACT_L>(&act.spend_proof.charge()).map_err(|_| {
        ServerError::BadRequest {
            message: "invalid charge amount in spend proof".to_string(),
        }
    })?;
    if share == 0 {
        return Err(ServerError::BadRequest {
            message: "a spend's share must be at least one credit".to_string(),
        });
    }
    Ok(share)
}

/// Verify one spend that passed [`check_spend_statics`], under the key it
/// names, re-checked under the lock it is verified under (the expensive
/// step), returning the refund of nothing (`t = 0`) that verification signs:
/// the spend's provisional refund. Nothing is recorded.
///
/// The refund signed at settlement verifies the proof once more: the ACT
/// crate verifies only as part of signing a refund for a fixed amount, and
/// this request's amount is not known until its response is settled
/// (`credentials::verify_and_refund`). The `t = 0` refund this verification
/// signs is stored with the nullifier, so a crash before settlement costs the
/// spend its hold and never the credential's untouched balance.
async fn verify_checked_spend(
    state: &AppState,
    act: &ActSpend,
) -> Result<SignedRefund, ServerError> {
    let cache = state.credential_key_cache.read().await;
    let key = credentials::usable_key(
        &cache,
        &act.issuer_key_hash,
        KeyUse::Spend,
        SystemTime::now(),
    )?;
    credentials::verify_presented_spend(key, act)
}

/// Every presented spend of a request, checked and verified, before anything
/// is recorded: the cheap checks for every input first ([`check_spend_statics`],
/// the hold's sum, distinct nullifiers), then each proof, stopping at the
/// first that does not verify. Returns each input's share and the `t = 0`
/// refund its verification signed (its provisional refund), in input order.
///
/// **Distinct nullifiers, per key.** Two inputs with one nullifier are two
/// spends of one credential instance, which ACT lets a client produce (with
/// different shares) and which is exactly the fork nullifiers exist to stop;
/// the whole request is refused. The atomic insert would refuse it too.
pub(crate) async fn check_presented_spends(
    state: &AppState,
    acts: &[ActSpend],
) -> Result<(Vec<u128>, Vec<SignedRefund>), ServerError> {
    let mut shares = Vec::with_capacity(acts.len());
    for act in acts {
        shares.push(check_spend_statics(state, act).await?);
    }
    if eidola_common::spend_credentials::hold(&shares).is_none() {
        return Err(ServerError::BadRequest {
            message: "the shares of this request overflow".to_string(),
        });
    }
    for (i, act) in acts.iter().enumerate() {
        let repeated = acts[i + 1..].iter().any(|other| {
            other.issuer_key_hash == act.issuer_key_hash
                && other.spend_proof.nullifier().as_bytes()
                    == act.spend_proof.nullifier().as_bytes()
        });
        if repeated {
            return Err(ServerError::BadRequest {
                message: "a request presents the same credential twice".to_string(),
            });
        }
    }
    let mut provisionals = Vec::with_capacity(acts.len());
    for act in acts {
        provisionals.push(verify_checked_spend(state, act).await?);
    }
    Ok((shares, provisionals))
}

/// Validate the model and the request's hold (the sum of its inputs'
/// shares) against the request.
///
/// Called after the nullifiers are recorded. Errors here require full
/// refunds. `resolution` is the request's `model`, resolved at the edge
/// (`Catalog::resolve`).
fn validate_request(
    hold: u128,
    request: &ChatCompletionRequest,
    resolution: Result<&ResolvedModel, crate::models::ModelRefError>,
) -> Result<ResolvedModel, ServerError> {
    // The model, resolved to its weights hash, and its pricing.
    let resolved = resolution.map_err(ServerError::ModelReference)?.clone();

    // No completion past the declared ceiling, whatever the client.
    check_completion_ceiling(request, &resolved.model)?;

    // The presented hold must be exactly the worst-case cost (the shared
    // contract's hold), neither less nor more.
    check_hold_equals_worst_case(hold, request, &resolved.model)?;

    Ok(resolved)
}

/// Pre-flight go/no-go: refuse a spend whose hold is not exactly the
/// worst-case cost — the same formula the client used to size its hold —
/// before anything is sent upstream. Pure so it is directly unit-testable.
///
/// **Exactly, not at least.** A hold below the worst case could be charged
/// more than it holds; a hold above it is a number the client chose beyond
/// what the request needs, a degree of freedom it could be recognised by, and
/// with an exact hold a request settled without reported usage is charged
/// the worst case and never more. Refused after the nullifier is recorded,
/// like every validation refusal, so it is answered with a full refund.
fn check_hold_equals_worst_case(
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
    if charge_credits > wc {
        // No amount in the message: `BadRequest` is logged as it reads.
        return Err(ServerError::BadRequest {
            message: "the hold exceeds this request's worst-case cost; it must equal it"
                .to_string(),
        });
    }
    Ok(())
}

/// Create a chat completion.
///
/// Paid with one or two ACT (Anonymous Credit Token) spends in the body's
/// `credentials` member. Every spend proof is verified, every nullifier is
/// recorded, all or none, each with a provisional refund of nothing, and when
/// the request ends each is replaced by its input's final refund: the charge
/// is consumed in input order, and each input's unspent credits return under
/// its own issuer key.
///
/// `model` is `tag`, `tag@sha256:<hex>` or `sha256:<hex>` (a full SHA-256 in
/// lowercase hex), resolved to the model's weights hash before anything else.
/// A digest, when present, decides alone; a bare tag resolves through this
/// build's catalog. The response's `model` is `tag@sha256:<hex>` (or
/// `sha256:<hex>` for a hash named alone).
#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    tag = "Unlinked",
    request_body = ChatCompletionRequest,
    responses(
        (status = 200, description = "Chat completion response with privacy and verification metadata", body = EidolaResponse),
        (status = 400, description = "Invalid request; a malformed model reference (`invalid_model_reference`: not `tag`, `tag@sha256:<hex>` or `sha256:<hex>`), a malformed refund recovery key, more than two credentials, one credential presented twice, a share of zero, or a credential in a header (refused before any credential is spent); or a hold above the worst-case cost (refunded in full)", body = ErrorResponse),
        (status = 401, description = "Authentication failed", body = ErrorResponse),
        (status = 402, description = "A hold below the worst-case cost (refunded in full)", body = ErrorResponse),
        (status = 404, description = "The model reference names no model (`model_not_found`); refunded in full", body = ErrorResponse),
        (status = 409, description = "A credential already spent; none of the request's credentials is recorded", body = ErrorResponse),
        (status = 410, description = "The model reference names a retired model (`model_retired`); refunded in full", body = ErrorResponse),
        (status = 502, description = "Inference engine error", body = ErrorResponse),
        (status = 503, description = "No engine is available for the model", body = ErrorResponse)
    )
)]
pub async fn chat_completions(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    ChatRequest(mut validated, reading): ChatRequest,
) -> Result<axum::response::Response, ServerError> {
    // The model reference is resolved to a weights hash here, at the edge:
    // the request is rewritten to name the hash, so routing, the node,
    // billing, logs and telemetry see only the hash, never the tag the
    // client sent. A reference that does not resolve is refused after the
    // credential is spent, with a full refund, like every validation
    // refusal.
    let resolution = state.catalog.resolve(&validated.request().model);
    if let Ok(resolved) = &resolution {
        validated.resolve_model(&resolved.model.sha256);
        // A request that names no completion ceiling asks for the model's
        // declared one — written into the request, so pricing, settlement
        // and the body the node receives all read one number. Left absent,
        // the node would fill the rest of its window, past what was priced.
        validated.default_completion_ceiling(resolved.model.max_output_tokens);
    }
    // The counter's `model` label must come from the catalog, never the
    // caller's string (the fixed-list rule bounding label cardinality): the
    // resolved hash, with anything unresolved collapsing to `other`.
    let model_label = resolution
        .as_ref()
        .map(|r| r.model.sha256.clone())
        .unwrap_or_else(|_| "other".to_string());
    let stream = validated.request().stream;

    let presented = crate::auth::refuse_header_credentials(&headers)
        .and_then(|()| crate::auth::presented_spends(&validated.request().credentials.0));
    let result = match presented {
        Ok(acts) => chat_completions_phases(&state, acts, validated, resolution, reading).await,
        Err(e) => Err(e),
    };

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

    result
}

async fn chat_completions_phases(
    state: &AppState,
    acts: Vec<ActSpend>,
    validated: ValidatedRequest,
    resolution: Result<ResolvedModel, crate::models::ModelRefError>,
    reading: BodyPermit,
) -> Result<axum::response::Response, ServerError> {
    // Phase 1: Check and verify every input. Errors here mean a token is
    // invalid/malformed — no nullifier recorded, no refund needed.
    let (shares, provisionals) = check_presented_spends(state, &acts).await?;
    // The request is paid for: it stops counting against the memory set
    // aside for unverified large bodies.
    drop(reading);

    // Phases 2 to 4, in a task that outlives this handler.
    run_paid_request(
        state.clone(),
        validated,
        resolution,
        record_inputs(state.clone(), acts, shares, provisionals),
    )
    .await
}

/// Phase 2: Record every nullifier, all or none, each with its provisional
/// refund (`t = 0`) stored beside it. After this succeeds, every input is
/// consumed and a refund MUST be issued for each on every subsequent code
/// path; the final ones overwrite the provisional ones.
///
/// Every input enters this process's live set first, so there is no instant
/// at which a provisional refund is stored and recovery could take it for one
/// a crash left behind; each leaves when the last holder of its input lets
/// go, after its final refund's write has been attempted.
async fn record_inputs(
    state: AppState,
    acts: Vec<ActSpend>,
    shares: Vec<u128>,
    provisionals: Vec<SignedRefund>,
) -> Result<PaidInputs, ServerError> {
    let lives: Vec<_> = acts
        .iter()
        .map(|act| std::sync::Arc::new(state.live_spends.serve(act)))
        .collect();
    let recording: Vec<(&ActSpend, &SignedRefund)> = acts.iter().zip(&provisionals).collect();
    let recorded = credentials::record_spends(&state, &recording, RefundStage::Provisional).await?;
    let Some(recorded) = recorded else {
        return Err(ServerError::Conflict {
            message: "credential already spent (duplicate nullifier)".to_string(),
        });
    };
    Ok(PaidInputs {
        acts: acts
            .into_iter()
            .zip(recorded)
            .zip(lives)
            .map(|((act, sealed), live)| ActSpend {
                provisional: Some(std::sync::Arc::new(sealed)),
                live: Some(live),
                ..act
            })
            .collect(),
        shares,
    })
}

/// Record a verified request's inputs (`recording`) and serve it, in a task
/// of its own, answering with what the task returns.
///
/// **The task outlives its caller.** A client that goes away drops the
/// handler's future wherever it is waiting; were recording or anything after
/// it awaited there, the nullifiers could be recorded with nothing left to
/// settle them, and recovery would declare each provisional refund final: the
/// whole hold charged for nothing. So recording, the validation refusal, the
/// dispatch and every settlement run here for both transports, and a request
/// whose handler has gone still ends with its final refunds stored for
/// recovery. A request whose handler went away before its engine answered is
/// dispatched to its answer like any other: a refusal settles as a full
/// refund, and an opened stream finds its client gone at once, releases the
/// engine and settles as that early ending ([`forward_stream`], and for a
/// blocking request [`collect_answer`], which learns its client has gone from
/// this handler's [`ClientWatch`]). The deadline bounds both.
pub(crate) async fn run_paid_request(
    state: AppState,
    validated: ValidatedRequest,
    resolution: Result<ResolvedModel, crate::models::ModelRefError>,
    recording: impl Future<Output = Result<PaidInputs, ServerError>> + Send + 'static,
) -> Result<axum::response::Response, ServerError> {
    let (client, present) = tokio::sync::oneshot::channel();
    let client = ClientWatch(client);
    let answer = tokio::spawn(async move {
        let deadline = tokio::time::Instant::now() + MAX_REQUEST_DURATION;
        let inputs = recording.await?;

        // --- POINT OF NO RETURN: every nullifier is recorded ---
        // From here on, we MUST issue a refund for every input on any error,
        // and the request ends by `deadline` whatever the engine does.
        let request = validated.request();

        // Phase 3: Validate the request (model, hold). On failure, every
        // input's share is returned in full.
        let resolved = match validate_request(
            inputs.hold(),
            request,
            resolution.as_ref().map_err(|e| *e),
        ) {
            Ok(resolved) => resolved,
            Err(e) => {
                warn!(
                    "Request validation failed after nullifier recorded, issuing full refund: {e}"
                );
                let refunds = settle_inputs(&state, &inputs, 0).await;
                return Ok(error_response_with_refunds(&e, refunds));
            }
        };

        // Phase 4: Handle the request.
        if request.stream {
            handle_streaming_request(state.clone(), &validated, inputs, &resolved, deadline).await
        } else {
            handle_non_streaming_request(&state, &validated, &inputs, &resolved, deadline, client)
                .await
        }
    })
    .await;
    // Held until the task has answered: dropped earlier only with this
    // handler, when its client went away.
    drop(present);
    answer.map_err(|_| ServerError::Internal("the request's task failed".to_string()))?
}

/// What a blocking request's task watches to learn that its client has gone:
/// the handler holds the other end until the task answers, and a client that
/// goes away drops the handler, and that end with it.
pub(crate) struct ClientWatch(tokio::sync::oneshot::Sender<std::convert::Infallible>);

impl ClientWatch {
    /// Resolves once the handler, and with it the client, has gone.
    async fn gone(&mut self) {
        self.0.closed().await
    }
}

/// Open the engine's stream for a paid request. A refusal, or no engine
/// answering, is settled here as a full refund; not opened by the deadline,
/// the engine may have begun, and the whole hold is charged. Either way the
/// error is the request's answer, its refunds in it.
async fn open_upstream(
    state: &AppState,
    validated: &ValidatedRequest,
    inputs: &PaidInputs,
    deadline: tokio::time::Instant,
) -> Opened {
    match tokio::time::timeout_at(deadline, dispatch_stream(state, validated)).await {
        Ok(Ok(rx)) => Opened::Stream(rx),
        Ok(Err(e)) => {
            // Nothing was generated: the engine refused it or never took it.
            warn!(
                "Engine error before the answer opened, issuing full refund: {}",
                e
            );
            let refunds = settle_inputs(state, inputs, 0).await;
            Opened::Answered(error_response_with_refunds(&e, refunds))
        }
        Err(_) => {
            warn!("Request reached its deadline before its answer opened, issuing zero refund");
            let refunds = settle_inputs(state, inputs, inputs.hold()).await;
            Opened::Answered(error_response_with_refunds(
                &request_deadline_error(),
                refunds,
            ))
        }
    }
}

/// What opening a paid request's engine stream came to.
enum Opened {
    Stream(mpsc::Receiver<Result<BackendStreamEvent, ServerError>>),
    /// Settled without a stream; this is the request's answer.
    Answered(axum::response::Response),
}

/// Handle a non-streaming chat completion request: the engine's stream,
/// assembled into one answer ([`collect_answer`]) and settled on its usage. An
/// answer that ends early is settled on what was generated
/// ([`early_cost`]), by the rule a stream follows, and its error is the
/// response, carrying the refunds.
async fn handle_non_streaming_request(
    state: &AppState,
    validated: &ValidatedRequest,
    inputs: &PaidInputs,
    resolved: &ResolvedModel,
    deadline: tokio::time::Instant,
    mut client: ClientWatch,
) -> Result<axum::response::Response, ServerError> {
    let request = validated.request();
    let model = &resolved.model;
    let charge_credits = inputs.hold();
    let auth_context = AuthContext {
        method: AuthMethod::AnonymousCredential,
    };
    let chargeable_prompt = chargeable_prompt_tokens_for(request);
    let max_completion = effective_max_completion(request, model);

    let dispatched_at = Instant::now();
    let upstream_rx = match open_upstream(state, validated, inputs, deadline).await {
        Opened::Stream(rx) => rx,
        Opened::Answered(answer) => return Ok(answer),
    };
    let mut assembler = Assembler::new();
    let meta = match collect_answer(upstream_rx, &mut client, deadline, &mut assembler).await {
        Ok(meta) => meta,
        Err((reason, error)) => {
            // The engine was released before this.
            warn!(
                "blocking request ended early ({}), settling on what was generated",
                reason.label()
            );
            let cost = early_cost(assembler.usage(), model, chargeable_prompt, max_completion);
            let refunds = settle_inputs(state, inputs, cost).await;
            return Ok(error_response_with_refunds(&error, refunds));
        }
    };
    crate::telemetry::metrics::CHAT_DURATION.record(
        dispatched_at.elapsed().as_secs_f64(),
        &[KeyValue::new("model", model.sha256.clone())],
    );

    // Record token usage metrics (safe for unlinked layer: only model + counts).
    if let Some(usage) = assembler.usage() {
        let model_attr = KeyValue::new("model", model.sha256.clone());
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
    // No usage → charge worst case.
    let cost = settled_cost(
        assembler.usage(),
        model,
        chargeable_prompt,
        max_completion,
        charge_credits,
    );
    let refunds = settle_inputs(state, inputs, cost).await;

    let backend_response = assembler.finish(meta);
    let meta = &backend_response.meta;
    let is_tee = meta.tee_type.is_some();

    let privacy = build_privacy_metadata(&auth_context, is_tee, &meta.provider);
    let verification = build_verification_metadata(None);

    // The response names the model as the client's reference resolved:
    // `tag@sha256:<hex>`, or `sha256:<hex>` for a hash named alone.
    let mut completion = backend_response.response;
    completion.model.clone_from(&resolved.echo);
    let eidola_response =
        EidolaResponse::from_completion(completion, privacy, verification, refunds);

    Ok(Json(eidola_response).into_response())
}

/// Read the engine's stream for a blocking request into `assembler` until it
/// ends, and say how: the engine's `Done`, or an early ending with the error
/// that answers it. Every chunk is kept, the engine's running count among
/// them, so [`Assembler::usage`] is what was generated by the last chunk
/// received whichever way it ended.
///
/// The client going away ([`ClientWatch`]) and the deadline are watched while
/// the engine works, and on any early ending the upstream receiver is dropped
/// before this returns, which ends the engine's work at once (the relay
/// closes its connection, and the node cancels the request). A blocking
/// client has no stream to stop reading: its answer is written once, after
/// settlement, so nothing it does after asking holds the engine.
pub(crate) async fn collect_answer(
    mut upstream_rx: mpsc::Receiver<Result<BackendStreamEvent, ServerError>>,
    client: &mut ClientWatch,
    deadline: tokio::time::Instant,
    assembler: &mut Assembler,
) -> Result<crate::backend::BackendMeta, (EarlyEnding, ServerError)> {
    loop {
        let next = tokio::select! {
            biased;
            () = client.gone() => Err(EarlyEnding::ClientGone),
            () = tokio::time::sleep_until(deadline) => Err(EarlyEnding::Deadline),
            event = upstream_rx.recv() => Ok(event),
        };
        match next {
            Err(EarlyEnding::Deadline) => {
                return Err((EarlyEnding::Deadline, request_deadline_error()));
            }
            Err(early) => {
                return Err((
                    early,
                    ServerError::Network("the client went away".to_string()),
                ));
            }
            Ok(None) => {
                return Err((
                    EarlyEnding::ChannelClosed,
                    ServerError::Network(
                        "the engine's stream ended before it completed".to_string(),
                    ),
                ));
            }
            Ok(Some(Ok(BackendStreamEvent::Chunk(chunk, _)))) => {
                if assembler.push(&chunk).is_err() {
                    return Err((
                        EarlyEnding::AnswerTooLarge,
                        ServerError::Backend {
                            status: 502,
                            error_type: "response_too_large".to_string(),
                            message: "the answer exceeded what the gateway assembles".to_string(),
                        },
                    ));
                }
            }
            Ok(Some(Ok(BackendStreamEvent::Done(meta, _)))) => return Ok(meta),
            Ok(Some(Err(e))) => return Err((EarlyEnding::UpstreamError, e)),
        }
    }
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

/// Whether a streamed chunk is the engine's running count and nothing else:
/// `usage`, on choices that say nothing (no role, output, finish reason or
/// `continuable`). The node sends one for a step that generated tokens and
/// gave out no text (a tool call still being read, held-back text); it is
/// kept for billing and never offered to the client. The final usage chunk
/// has no choices and is not one of these.
fn carries_only_a_count(chunk: &ChatCompletionChunk) -> bool {
    chunk.usage.is_some()
        && !chunk.choices.is_empty()
        && chunk.choices.iter().all(|choice| {
            let delta = &choice.delta;
            choice.finish_reason.is_none()
                && choice.continuable.is_none()
                && delta.role.is_none()
                && delta.content.is_none()
                && delta.reasoning_content.is_none()
                && delta.reasoning.is_none()
                && delta.tool_calls.is_none()
        })
}

/// Handle a streaming chat completion request.
async fn handle_streaming_request(
    state: AppState,
    validated: &ValidatedRequest,
    inputs: PaidInputs,
    resolved: &ResolvedModel,
    deadline: tokio::time::Instant,
) -> Result<axum::response::Response, ServerError> {
    let model = &resolved.model;
    let charge_credits = inputs.hold();
    let request = validated.request();
    let auth_context = AuthContext {
        method: AuthMethod::AnonymousCredential,
    };

    // Stream timing is measured from upstream dispatch, not from request
    // arrival: everything before this point is our own verification work,
    // which the HTTP-level histogram already covers.
    let dispatched_at = Instant::now();
    let upstream_rx = match open_upstream(&state, validated, &inputs, deadline).await {
        Opened::Stream(rx) => rx,
        Opened::Answered(answer) => return Ok(answer),
    };

    // Events for the client, framed by `padding` into fixed-size writes on a
    // fixed tick, the waiting bytes bounded (see `padding::EventSender::send`).
    let (tx, padded) = padding::channel(Cadence::CLIENT_FACING);

    // Move values into the spawned task. The task refunds every input
    // itself, each under the key that input names.
    let spent = inputs;
    let model_id = model.sha256.clone();
    let echo = resolved.echo.clone();
    let task_model = model.clone();
    // Contract clamp inputs, computed from the request before the task takes
    // over (the spawned task never sees the request itself).
    let chargeable_prompt = chargeable_prompt_tokens_for(request);
    let max_completion = effective_max_completion(request, model);
    let provider = crate::engine_router::PROVIDER;

    tokio::spawn(async move {
        // Every terminal metadata write is abandoned by then.
        let delivery_until = deadline + DEADLINE_DELIVERY_GRACE;
        let mut timing = StreamTiming::new(model_id.clone(), dispatched_at);
        let (ending, final_usage) =
            forward_stream(upstream_rx, &tx, deadline, &echo, &mut timing).await;
        match ending {
            StreamEnding::Done(meta, at) => {
                let is_tee = meta.tee_type.is_some();
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
                let refunds = settle_inputs(&state, &spent, cost).await;

                send_metadata_event(
                    &tx,
                    refunds,
                    privacy,
                    verification,
                    meta.chat_id.unwrap_or_default(),
                    true,
                    delivery_until,
                )
                .await;
            }
            StreamEnding::Early(reason) => {
                // Settled on what the engine generated up to its last chunk
                // (`early_cost`), consumed from the inputs in input order
                // like any charge; the engine was released before this.
                warn!(
                    "stream ended early ({}), settling on what was generated",
                    reason.label()
                );
                timing.finish(reason.label(), None, Instant::now());
                let cost = early_cost(
                    final_usage.as_ref(),
                    &task_model,
                    chargeable_prompt,
                    max_completion,
                );
                let refunds = settle_inputs(&state, &spent, cost).await;
                if reason != EarlyEnding::ClientGone {
                    // A client cut off at the deadline or for not reading is
                    // offered its refund in band for the grace alone.
                    let until =
                        if matches!(reason, EarlyEnding::Deadline | EarlyEnding::ClientStalled) {
                            tokio::time::Instant::now() + DEADLINE_DELIVERY_GRACE
                        } else {
                            delivery_until
                        };
                    let privacy = build_privacy_metadata(&auth_context, true, provider);
                    let verification = build_verification_metadata(None);
                    // No `[DONE]`: the answer did not finish, and the
                    // client must not take what arrived for a whole one.
                    send_metadata_event(
                        &tx,
                        refunds,
                        privacy,
                        verification,
                        String::new(),
                        false,
                        until,
                    )
                    .await;
                }
            }
        }
    });

    Ok(padding::padded_sse_response(padded))
}

/// Send the stream's metadata event, carrying its refunds, then `[DONE]` when
/// `finished` (the engine said it was done). A stream that ended early gets
/// the metadata and no `[DONE]`, so a client reads it as the interrupted
/// answer it is and still finds its refund in band.
///
/// Abandoned at `until`, whatever ending it closes: a client that holds its
/// connection open without reading does not hold the task past the request's
/// deadline (and its grace), and the refund is stored before this runs.
#[allow(clippy::too_many_arguments)]
async fn send_metadata_event(
    tx: &padding::EventSender,
    refunds: Vec<RefundInfo>,
    privacy: crate::response::PrivacyMetadata,
    verification: crate::response::VerificationMetadata,
    chat_id: String,
    finished: bool,
    until: tokio::time::Instant,
) {
    let stream_meta = EidolaStreamMetadata::new(chat_id, privacy, verification, refunds);
    let json_str = serde_json::to_string(&stream_meta).unwrap();
    let _ = tokio::time::timeout_at(until, async {
        let _ = tx.send(StreamEvent::Json(json_str)).await;
        if finished {
            let _ = tx.send(StreamEvent::Done).await;
        }
    })
    .await;
}

/// Why a request ended before the engine said it was done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EarlyEnding {
    /// The client went away (its stream was dropped).
    ClientGone,
    /// The client stopped reading ([`CLIENT_STALL_LIMIT`]).
    ClientStalled,
    /// The request reached its deadline ([`MAX_REQUEST_DURATION`]).
    Deadline,
    /// The engine failed partway.
    UpstreamError,
    /// The engine's channel closed without a `Done`.
    ChannelClosed,
    /// A blocking answer grew past what the gateway assembles
    /// (`engine_router::assemble::MAX_ASSEMBLED_BYTES`).
    AnswerTooLarge,
}

impl EarlyEnding {
    /// The fixed label the stream-outcome counter and the log use.
    fn label(self) -> &'static str {
        match self {
            EarlyEnding::ClientGone => "client_disconnect",
            EarlyEnding::ClientStalled => "client_stalled",
            EarlyEnding::Deadline => "deadline",
            EarlyEnding::UpstreamError => "upstream_error",
            EarlyEnding::ChannelClosed => "channel_closed",
            EarlyEnding::AnswerTooLarge => "answer_too_large",
        }
    }
}

/// How a forwarded stream ended.
enum StreamEnding {
    /// The engine finished.
    Done(crate::backend::BackendMeta, Instant),
    /// It ended early; the engine has been released.
    Early(EarlyEnding),
}

/// Forward the engine's stream to the client until it ends, and say how it
/// ended, with the last usage the engine reported (its running count, or its
/// final one).
///
/// The client going away, the client no longer reading
/// ([`CLIENT_STALL_LIMIT`]) and the deadline are watched while the engine is
/// still working (a long prefill sends nothing) and while a chunk waits for
/// room in the client's stream, and on any of them, or on any
/// early ending, the upstream receiver is dropped before this returns, which
/// ends the engine's work at once (the relay closes its connection, and the
/// node cancels the request in its scheduler). The engine's running count
/// rides on every delta chunk, and alone on a chunk with an empty delta for a
/// step that gave out no text ([`carries_only_a_count`], kept and not
/// forwarded); it is kept before the client is offered the chunk, so what was
/// generated is counted whether or not it was delivered; only the final usage
/// chunk (no choices) reaches the client, as the stream always carried it. Every chunk names the model as `echo`, the client's
/// reference resolved.
async fn forward_stream(
    mut upstream_rx: mpsc::Receiver<Result<BackendStreamEvent, ServerError>>,
    tx: &padding::EventSender,
    deadline: tokio::time::Instant,
    echo: &str,
    timing: &mut StreamTiming,
) -> (StreamEnding, Option<Usage>) {
    let mut final_usage: Option<Usage> = None;
    loop {
        let next = tokio::select! {
            biased;
            () = tx.closed() => Err(EarlyEnding::ClientGone),
            () = tokio::time::sleep_until(deadline) => Err(EarlyEnding::Deadline),
            () = tx.stalled(CLIENT_STALL_LIMIT) => Err(EarlyEnding::ClientStalled),
            event = upstream_rx.recv() => Ok(event),
        };
        let event = match next {
            Ok(Some(event)) => event,
            Ok(None) => return (StreamEnding::Early(EarlyEnding::ChannelClosed), final_usage),
            Err(early) => return (StreamEnding::Early(early), final_usage),
        };
        match event {
            Ok(BackendStreamEvent::Chunk(mut chunk, at)) => {
                echo.clone_into(&mut chunk.model);
                if carries_output(&chunk) {
                    timing.on_chunk(at);
                }
                if chunk.usage.is_some() {
                    final_usage.clone_from(&chunk.usage);
                    if carries_only_a_count(&chunk) {
                        continue;
                    }
                    if !chunk.choices.is_empty() {
                        chunk.usage = None;
                    }
                }
                let json_str = serde_json::to_string(&chunk).unwrap();
                let sent = tokio::select! {
                    biased;
                    sent = tx.send(StreamEvent::Json(json_str)) => sent.map_err(|_| EarlyEnding::ClientGone),
                    // A client reading too slowly to take the stream holds
                    // the request past its deadline no more than an engine
                    // does, and one that has stopped reading holds it no
                    // longer than the stall limit.
                    () = tokio::time::sleep_until(deadline) => Err(EarlyEnding::Deadline),
                    () = tx.stalled(CLIENT_STALL_LIMIT) => Err(EarlyEnding::ClientStalled),
                };
                if let Err(early) = sent {
                    return (StreamEnding::Early(early), final_usage);
                }
            }
            Ok(BackendStreamEvent::Done(meta, at)) => {
                // Prefer usage from the chunks, then from meta.
                if final_usage.is_none() {
                    final_usage.clone_from(&meta.usage);
                }
                return (StreamEnding::Done(meta, at), final_usage);
            }
            Err(e) => {
                error!("Stream error: {}", e);
                return (StreamEnding::Early(EarlyEnding::UpstreamError), final_usage);
            }
        }
    }
}

/// `Json<T>` wrapper that logs the rejection's class at warn level on
/// failure and answers it with a fixed message for that class.
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
/// expected u32`). So neither the log nor the client gets it: the log
/// records the class, and the client gets a fixed message for the class
/// (`refusal`), with the rejection's own status. Sent to the client, the
/// detail would be its own data over its own attested connection, but the
/// response's length would carry the echoed value to anyone watching the
/// ciphertext, padding or not (a bucket is a function of length).
///
/// A body longer than [`crate::MAX_REQUEST_BODY_BYTES`] is refused (413), and
/// one past [`crate::UNGATED_REQUEST_BODY_BYTES`] is read only under a
/// [`BodyPermit`] (`checked_body`).
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
        let (parts, bytes, _permit) = checked_body::<T, S>(req, state).await?;
        parse_logged(parts, bytes, state).await.map(Self)
    }
}

/// The large request bodies being read and parsed right now, at most
/// [`crate::large_body_permits`].
static LARGE_BODIES: std::sync::LazyLock<tokio::sync::Semaphore> =
    std::sync::LazyLock::new(|| tokio::sync::Semaphore::new(crate::large_body_permits()));

/// How long a large request body may take to arrive: its permit is one of
/// few, and a sender that stalls must not keep it.
const LARGE_BODY_READ_TIMEOUT: Duration = Duration::from_secs(300);

/// The hold a large request body has on the memory set aside for them
/// (`None` for a body within [`crate::UNGATED_REQUEST_BODY_BYTES`]): taken
/// before the body is read and kept while it and its parse are held
/// unverified.
pub struct BodyPermit(#[allow(dead_code)] Option<tokio::sync::SemaphorePermit<'static>>);

/// The body's bytes, within the size limit and held to the JSON shape limits
/// (see [`LoggedJson`]).
///
/// **Size.** A body that declares a length of at most
/// [`crate::UNGATED_REQUEST_BODY_BYTES`], or declares none, is read up to that
/// many bytes. One that declares more, up to [`crate::MAX_REQUEST_BODY_BYTES`],
/// takes a [`BodyPermit`] first and is refused (503) when none is free, so
/// the memory unverified large bodies and their parses take is bounded; one
/// that declares more than that is refused (413) unread.
// The error is the refusal itself, built once and returned as the extractor's
// rejection.
#[allow(clippy::result_large_err)]
async fn checked_body<T, S: Send + Sync>(
    req: Request,
    state: &S,
) -> Result<(axum::http::request::Parts, axum::body::Bytes, BodyPermit), axum::response::Response> {
    use axum::http::StatusCode;
    let too_large = || {
        warn!(
            payload_type = std::any::type_name::<T>(),
            "request body rejected: too large"
        );
        refusal(
            StatusCode::PAYLOAD_TOO_LARGE,
            "the request body is too large",
        )
    };
    let unreadable = |status| {
        warn!(
            payload_type = std::any::type_name::<T>(),
            "request body rejected: bytes error"
        );
        refusal(status, "the request body could not be read")
    };
    let (parts, body) = req.into_parts();
    let declared = parts
        .headers
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    let (bytes, permit) = match declared {
        Some(len) if len > crate::MAX_REQUEST_BODY_BYTES as u64 => return Err(too_large()),
        Some(len) if len > crate::UNGATED_REQUEST_BODY_BYTES as u64 => {
            let Ok(permit) = LARGE_BODIES.try_acquire() else {
                warn!(
                    payload_type = std::any::type_name::<T>(),
                    "request body rejected: no large-body permit free"
                );
                return Err(ServerError::ServiceUnavailable(
                    "too many large requests are being read; try again".to_string(),
                )
                .into_response());
            };
            let read = axum::body::to_bytes(body, len as usize);
            match tokio::time::timeout(LARGE_BODY_READ_TIMEOUT, read).await {
                Ok(Ok(bytes)) => (bytes, Some(permit)),
                Ok(Err(_)) => return Err(unreadable(StatusCode::BAD_REQUEST)),
                Err(_) => return Err(unreadable(StatusCode::REQUEST_TIMEOUT)),
            }
        }
        _ => {
            let read =
                axum::body::Bytes::from_request(Request::from_parts(parts.clone(), body), state);
            match read.await {
                Ok(bytes) => (bytes, None),
                Err(rejection) if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE => {
                    return Err(too_large());
                }
                Err(rejection) => return Err(unreadable(rejection.status())),
            }
        }
    };
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
    Ok((parts, bytes, BodyPermit(permit)))
}

/// A request over a body [`checked_body`] has already read and held to the
/// size limits, for axum's own extractors to parse: the router's limit for
/// unread bodies does not apply to it.
fn read_body(parts: axum::http::request::Parts, bytes: axum::body::Bytes) -> Request {
    let mut req = Request::from_parts(parts, axum::body::Body::from(bytes));
    axum::extract::DefaultBodyLimit::disable().apply(&mut req);
    req
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
    match Json::<T>::from_request(read_body(parts, bytes), state).await {
        Ok(Json(value)) => Ok(value),
        Err(rejection) => {
            // Class only — the rejection's message can quote body
            // values (see the type-level privacy note).
            let (class, message) = match &rejection {
                JsonRejection::JsonDataError(_) => (
                    "data",
                    "the request body does not match the expected schema",
                ),
                JsonRejection::JsonSyntaxError(_) => {
                    ("syntax", "the request body is not valid JSON")
                }
                JsonRejection::MissingJsonContentType(_) => (
                    "missing content-type",
                    "expected a request with `Content-Type: application/json`",
                ),
                JsonRejection::BytesRejection(_) => ("bytes", "the request body could not be read"),
                _ => ("other", "invalid request body"),
            };
            warn!(
                payload_type = std::any::type_name::<T>(),
                "request body rejected: {class} error"
            );
            Err(refusal(rejection.status(), message))
        }
    }
}

/// A body refusal: `status`, and a fixed message that quotes nothing from the
/// request.
fn refusal(status: axum::http::StatusCode, message: &'static str) -> axum::response::Response {
    (
        status,
        Json(ErrorResponse::new(message, "invalid_request_error")),
    )
        .into_response()
}

/// The chat request: [`LoggedJson`]'s checks and refusals, keeping the body's
/// bytes with the strict parse. A request bound for an Eidola-hosted engine
/// is forwarded from its bytes (`engine_trust::protocol::ValidatedRequest`),
/// whose only constructor parses exactly those bytes.
///
/// The content type and the JSON syntax are checked first by axum's own
/// extractor over a parse that keeps nothing; a body that fails any check is
/// refused by axum's parse of the strict type, exactly as before.
pub struct ChatRequest(pub ValidatedRequest, pub BodyPermit);

impl<S: Send + Sync> FromRequest<S> for ChatRequest {
    type Rejection = axum::response::Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let (parts, bytes, permit) = checked_body::<ChatCompletionRequest, S>(req, state).await?;
        let shape_ok = Json::<serde::de::IgnoredAny>::from_request(
            read_body(parts.clone(), bytes.clone()),
            state,
        )
        .await
        .is_ok();
        if shape_ok && let Ok(validated) = ValidatedRequest::from_bytes(bytes.clone()) {
            return Ok(Self(validated, permit));
        }
        parse_logged::<ChatCompletionRequest, S>(parts, bytes.clone(), state).await?;
        // axum accepted a body the strict parse refused; both parse the same
        // bytes with the same type, so this is not reached.
        ValidatedRequest::from_bytes(bytes)
            .map(|validated| Self(validated, permit))
            .map_err(|_| {
                ServerError::BadRequest {
                    message: "invalid request body".to_string(),
                }
                .into_response()
            })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    /// A request's inputs as a test builds them, without recording anything.
    /// A [`ClientWatch`] and the end its handler holds.
    pub(crate) fn client_watch() -> (
        super::ClientWatch,
        tokio::sync::oneshot::Receiver<std::convert::Infallible>,
    ) {
        let (watch, present) = tokio::sync::oneshot::channel();
        (super::ClientWatch(watch), present)
    }

    pub(crate) fn paid_inputs(acts: Vec<super::ActSpend>, shares: Vec<u128>) -> super::PaidInputs {
        super::PaidInputs { acts, shares }
    }

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
        let ChatRequest(validated, _) =
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

    /// The size limits on the shaped router, at their boundaries: a body that
    /// declares its length is read up to `MAX_REQUEST_BODY_BYTES` (24 MiB,
    /// what a node reads less what forwarding can add) and refused one byte
    /// past it; one that declares none is read up to
    /// `UNGATED_REQUEST_BODY_BYTES`; a large body is read only under a
    /// permit, and refused while none is free. Every refusal says what it is
    /// and is padded like any other response.
    #[tokio::test]
    async fn request_bodies_are_held_to_the_size_limits() {
        use axum::http::StatusCode;
        use tower::ServiceExt;
        async fn body_only(LoggedJson(_): LoggedJson<serde_json::Value>) -> &'static str {
            "ok"
        }
        let (router, _) = crate::build_router()
            .route("/body", axum::routing::post(body_only))
            .with_state(crate::credentials::tests::unreachable_state(
                Default::default(),
            ))
            .split_for_parts();
        let app = padding::shape(router);
        let post = async |len: usize, declared: bool| {
            let mut request = Request::builder()
                .method("POST")
                .uri("/body")
                .header("content-type", "application/json");
            if declared {
                request = request.header("content-length", len);
            }
            let body = format!("\"{}\"", "x".repeat(len - 2));
            let response = app
                .clone()
                .oneshot(request.body(axum::body::Body::from(body)).unwrap())
                .await
                .unwrap();
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert!(bytes.len().is_power_of_two() && bytes.len() >= padding::MIN_BODY_BUCKET);
            (status, String::from_utf8_lossy(&bytes).into_owned())
        };

        const MAX: usize = 24 * 1024 * 1024;
        const UNGATED: usize = 2 * 1024 * 1024;
        assert_eq!(crate::MAX_REQUEST_BODY_BYTES, MAX);
        assert_eq!(crate::UNGATED_REQUEST_BODY_BYTES, UNGATED);
        const {
            assert!(
                MAX + eidola_common::engine_protocol::FORWARDED_BODY_GROWTH_BYTES
                    <= eidola_common::engine_protocol::MAX_REQUEST_BODY_BYTES
            )
        };

        assert_eq!(post(MAX, true).await.0, StatusCode::OK);
        let (status, text) = post(MAX + 1, true).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert!(text.contains("the request body is too large"), "{text}");

        // Without a declared length, the ungated limit is the limit.
        assert_eq!(post(UNGATED, false).await.0, StatusCode::OK);
        let (status, text) = post(UNGATED + 1, false).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert!(text.contains("the request body is too large"), "{text}");

        // With every permit taken, a large body is refused and a small one
        // is still read.
        let permits = crate::large_body_permits();
        assert_eq!(permits, 10);
        let held = LARGE_BODIES.acquire_many(permits as u32).await.unwrap();
        let (status, text) = post(UNGATED + 1, true).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(text.contains("too many large requests"), "{text}");
        assert_eq!(post(UNGATED, true).await.0, StatusCode::OK);
        drop(held);
        assert_eq!(post(UNGATED + 1, true).await.0, StatusCode::OK);
        assert_eq!(LARGE_BODIES.available_permits(), permits);
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
        // A body past the value limit is refused for it whatever its strings
        // hold: one that is not UTF-8 never reaches a parser that allocates
        // per value.
        let mut hostile = values(MAX_REQUEST_JSON_VALUES + 1).into_bytes();
        let end = hostile.len() - 1;
        hostile.splice(end..end, *b",\"\xff\"");
        let req = Request::builder()
            .method("POST")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(hostile))
            .unwrap();
        let Err(response) = LoggedJson::<serde_json::Value>::from_request(req, &()).await else {
            panic!("accepted");
        };
        let (status, text) = refusal_of(response).await;
        assert_eq!(status, bad);
        assert!(text.contains("JSON values"), "{text}");
    }
    use crate::types::{
        Capability, Modality, ModelCapabilities, ModelPricing, OutputBudgetClass,
        PinnedWeightsCapability, PromptCacheCapability, ScaledPrice,
    };

    /// A token-priced model with easy integer math at `PRICING_SCALE_FACTOR`:
    /// 1 credit per prompt token, 2 credits per completion token.
    pub(crate) fn test_model() -> Model {
        Model {
            id: "test-model".to_string(),
            sha256: "3".repeat(64),
            tags: vec![],
            architecture: "MimoV26Flash".to_string(),
            weights: crate::types::WeightsProvenance {
                repo: "example/fixture".to_string(),
                revision: "4".repeat(40),
                files: vec![],
            },
            lifecycle: crate::types::ModelLifecycle {
                status: crate::types::LifecycleStatus::Current,
                retire_after: None,
            },
            deployments: vec![],
            name: "Test Model".to_string(),
            description: String::new(),
            context_length: 8192,
            max_output_tokens: Some(4096),
            output_budget_class: OutputBudgetClass::Standard,
            capabilities: ModelCapabilities {
                tool_calling: Capability::new(true),
                reasoning: Capability::new(false),
                input_modalities: vec![Modality::Text],
                output_modalities: vec![Modality::Text],
                prompt_cache: PromptCacheCapability::unsupported(),
                pinned_weights: PinnedWeightsCapability {
                    supported: true,
                    sha256: Some("3".repeat(64)),
                    repo: Some("example/test-model".to_string()),
                    revision: Some("4".repeat(40)),
                },
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

    /// **A stream that ends early is charged for what was generated**: the
    /// engine's last running count, priced and clamped by the contract; before
    /// any count, the prompt alone at the contract's cap. Never nothing (a
    /// client could otherwise generate freely and leave), never the whole
    /// hold for what was not produced.
    #[test]
    fn an_early_ending_is_charged_for_what_was_generated() {
        let req = request(&["hello world!"], 100);
        let model = test_model();
        let chargeable = chargeable_prompt_tokens_for(&req);
        let max_completion = effective_max_completion(&req, &model);
        let hold = worst_case_cost(&req, &model);
        let usage = |prompt: u32, completion: u32| Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        };
        // The test model charges one credit a prompt token, two a completion
        // token.
        let prompt = u32::try_from(chargeable).unwrap();
        assert_eq!(
            early_cost(Some(&usage(prompt, 7)), &model, chargeable, max_completion),
            u128::from(prompt) + 14
        );
        assert_eq!(
            early_cost(None, &model, chargeable, max_completion),
            u128::from(prompt),
            "nothing reported yet: the prompt, which the engine has begun on"
        );
        // Clamped like any usage, so never past the hold.
        assert_eq!(
            early_cost(
                Some(&usage(prompt + 1000, 1_000_000)),
                &model,
                chargeable,
                max_completion
            ),
            hold
        );
        assert!(early_cost(Some(&usage(prompt, 1)), &model, chargeable, max_completion) < hold);
    }

    /// The forwarding loop, driven directly: a channel standing in for the
    /// engine's relay, and a padded stream standing in for the client.
    mod forwarding {
        use super::*;
        use crate::backend::BackendMeta;

        const CADENCE: Cadence = Cadence {
            frames: eidola_common::stream_shape::Cadence::new(256, Duration::from_millis(5)),
            max_backlog_bytes: 1 << 20,
        };

        fn delta(text: &str, running: Option<(u32, u32)>) -> BackendStreamEvent {
            let mut chunk = serde_json::json!({
                "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}],
            });
            if let Some((p, c)) = running {
                chunk["usage"] = serde_json::json!({
                    "prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
                });
            }
            BackendStreamEvent::Chunk(serde_json::from_value(chunk).unwrap(), Instant::now())
        }

        fn timing() -> StreamTiming {
            StreamTiming::new("m".to_string(), Instant::now())
        }

        /// **The client leaving releases the engine at once and keeps the
        /// count**: the loop returns on the disconnect, even with the engine
        /// sending nothing, having dropped its receiver (which is what makes
        /// the relay close the engine's connection), and with the engine's
        /// last running count, which the early charge is computed from.
        #[tokio::test]
        async fn a_client_that_leaves_releases_the_engine_and_keeps_the_count() {
            let (up_tx, up_rx) = mpsc::channel(8);
            let (tx, padded) = padding::channel(CADENCE);
            up_tx.send(Ok(delta("a", Some((30, 3))))).await.unwrap();
            up_tx.send(Ok(delta("b", Some((30, 9))))).await.unwrap();
            let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
            let mut timing = timing();
            let forward = forward_stream(up_rx, &tx, deadline, "m", &mut timing);
            tokio::pin!(forward);
            // Both chunks are taken, then the engine goes quiet (a prefill, a
            // slow step) and the client leaves.
            let early = tokio::select! {
                ended = &mut forward => Some(ended),
                () = tokio::time::sleep(Duration::from_millis(100)) => None,
            };
            assert!(early.is_none(), "nothing ended yet");
            drop(padded);
            let (ending, usage) = tokio::time::timeout(Duration::from_secs(5), forward)
                .await
                .expect("the disconnect is noticed without waiting for the engine");
            assert!(matches!(
                ending,
                StreamEnding::Early(EarlyEnding::ClientGone)
            ));
            assert_eq!(usage.map(|u| u.completion_tokens), Some(9));
            assert!(up_tx.is_closed(), "the engine's receiver was dropped");
        }

        /// **A client that stops reading releases the engine within the
        /// stall limit and keeps the count**: nothing takes the stream's
        /// frames, so the loop ends as `ClientStalled` once the limit has
        /// passed, not at the deadline, with the engine's receiver dropped and
        /// its last running count kept for the early charge.
        #[tokio::test(start_paused = true)]
        async fn a_client_that_stops_reading_releases_the_engine_and_keeps_the_count() {
            let (up_tx, up_rx) = mpsc::channel(8);
            // Held, never read.
            let (tx, _padded) = padding::channel(CADENCE);
            up_tx.send(Ok(delta("a", Some((30, 3))))).await.unwrap();
            up_tx.send(Ok(delta("b", Some((30, 7))))).await.unwrap();
            let deadline = tokio::time::Instant::now() + MAX_REQUEST_DURATION;
            let started = tokio::time::Instant::now();
            let mut timing = timing();
            let (ending, usage) = forward_stream(up_rx, &tx, deadline, "m", &mut timing).await;
            assert!(matches!(
                ending,
                StreamEnding::Early(EarlyEnding::ClientStalled)
            ));
            let waited = started.elapsed();
            assert!(
                waited >= CLIENT_STALL_LIMIT
                    && waited < CLIENT_STALL_LIMIT + Duration::from_secs(1),
                "{waited:?}"
            );
            assert_eq!(usage.map(|u| u.completion_tokens), Some(7));
            assert!(up_tx.is_closed(), "the engine's receiver was dropped");
        }

        /// **A slow reader is not a stalled one, nor a quiet engine**: a
        /// client taking the stream's frames at their rate keeps the stream
        /// going while the engine sends nothing for longer than the stall
        /// limit (a long prefill: the frames are padding), and then while
        /// the engine outruns it, each send waiting for room; the stream ends
        /// as the engine's answer does.
        #[tokio::test(start_paused = true)]
        async fn a_slow_reader_is_not_taken_for_a_stalled_one() {
            use futures_util::StreamExt;
            let (up_tx, up_rx) = mpsc::channel(8);
            let (tx, padded) = padding::channel(CADENCE);
            let reader = tokio::spawn(padded.map(|f| f.unwrap().len()).fold(0, async |n, l| n + l));
            // Eight times what the stream holds waiting, so the sends wait
            // on the reader for most of the run.
            let text = "x".repeat(4096);
            let chunks = 8 * CADENCE.max_backlog_bytes / text.len();
            let engine = tokio::spawn(async move {
                tokio::time::sleep(2 * CLIENT_STALL_LIMIT).await;
                for i in 0..chunks {
                    let c = u32::try_from(i).unwrap();
                    up_tx.send(Ok(delta(&text, Some((30, c))))).await.unwrap();
                }
                let meta = BackendMeta {
                    provider: "eidola".to_string(),
                    chat_id: Some("c".to_string()),
                    backend_model: "m".to_string(),
                    tee_type: None,
                    usage: None,
                };
                let _ = up_tx
                    .send(Ok(BackendStreamEvent::Done(meta, Instant::now())))
                    .await;
            });
            let deadline = tokio::time::Instant::now() + MAX_REQUEST_DURATION;
            let started = tokio::time::Instant::now();
            let mut timing = timing();
            let (ending, _) = forward_stream(up_rx, &tx, deadline, "m", &mut timing).await;
            assert!(
                matches!(ending, StreamEnding::Done(..)),
                "not cut as stalled"
            );
            assert!(
                started.elapsed() > 4 * CLIENT_STALL_LIMIT,
                "the reader set the pace: {:?}",
                started.elapsed()
            );
            engine.await.unwrap();
            drop(tx);
            assert!(reader.await.unwrap() > 8 * CADENCE.max_backlog_bytes);
        }

        /// The deadline ends the loop the same way, engine quiet or not.
        #[tokio::test]
        async fn the_deadline_releases_the_engine_and_keeps_the_count() {
            let (up_tx, up_rx) = mpsc::channel(8);
            let (tx, _padded) = padding::channel(CADENCE);
            up_tx.send(Ok(delta("a", Some((30, 4))))).await.unwrap();
            let deadline = tokio::time::Instant::now() + Duration::from_millis(100);
            let mut timing = timing();
            let (ending, usage) = tokio::time::timeout(
                Duration::from_secs(5),
                forward_stream(up_rx, &tx, deadline, "m", &mut timing),
            )
            .await
            .expect("the deadline ends it");
            assert!(matches!(ending, StreamEnding::Early(EarlyEnding::Deadline)));
            assert_eq!(usage.map(|u| u.completion_tokens), Some(4));
            assert!(up_tx.is_closed());
        }

        /// **A blocking client that leaves releases the engine at once and
        /// keeps the count**: its handler dropped (here, the end it holds),
        /// the collection returns with the engine sending nothing, its
        /// receiver dropped, and the engine's last running count kept for the
        /// early charge.
        #[tokio::test]
        async fn a_blocking_client_that_leaves_releases_the_engine_and_keeps_the_count() {
            let (up_tx, up_rx) = mpsc::channel(8);
            up_tx.send(Ok(delta("a", Some((30, 3))))).await.unwrap();
            up_tx.send(Ok(delta("b", Some((30, 9))))).await.unwrap();
            let (mut client, present) = client_watch();
            let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
            let mut assembler = Assembler::new();
            let ended = {
                let collect = collect_answer(up_rx, &mut client, deadline, &mut assembler);
                tokio::pin!(collect);
                let early = tokio::select! {
                    ended = &mut collect => Some(ended.is_ok()),
                    () = tokio::time::sleep(Duration::from_millis(100)) => None,
                };
                assert!(early.is_none(), "nothing ended yet");
                drop(present);
                tokio::time::timeout(Duration::from_secs(5), collect)
                    .await
                    .expect("the departure is noticed without waiting for the engine")
            };
            assert!(matches!(ended, Err((EarlyEnding::ClientGone, _))));
            assert_eq!(assembler.usage().map(|u| u.completion_tokens), Some(9));
            assert!(up_tx.is_closed(), "the engine's receiver was dropped");
        }

        /// **A blocking request at its deadline releases the engine and keeps
        /// the count**, answered as the deadline.
        #[tokio::test(start_paused = true)]
        async fn a_blocking_request_at_its_deadline_releases_the_engine_and_keeps_the_count() {
            let (up_tx, up_rx) = mpsc::channel(8);
            up_tx.send(Ok(delta("a", Some((30, 4))))).await.unwrap();
            let (mut client, _present) = client_watch();
            let deadline = tokio::time::Instant::now() + MAX_REQUEST_DURATION;
            let mut assembler = Assembler::new();
            let ended = collect_answer(up_rx, &mut client, deadline, &mut assembler).await;
            assert_eq!(tokio::time::Instant::now(), deadline);
            let Err((EarlyEnding::Deadline, error)) = ended else {
                panic!("not ended by the deadline");
            };
            assert_eq!(error.status_code(), axum::http::StatusCode::GATEWAY_TIMEOUT);
            assert_eq!(assembler.usage().map(|u| u.completion_tokens), Some(4));
            assert!(up_tx.is_closed());
        }

        /// **A blocking answer is bounded**: the chunk that takes it past
        /// `MAX_ASSEMBLED_BYTES` ends it, the engine released and the count
        /// before it kept.
        #[tokio::test]
        async fn a_blocking_answer_past_its_bound_releases_the_engine() {
            use crate::engine_router::assemble::MAX_ASSEMBLED_BYTES;
            let (up_tx, up_rx) = mpsc::channel(8);
            let half = "x".repeat(MAX_ASSEMBLED_BYTES / 2 + 1);
            up_tx.send(Ok(delta(&half, Some((30, 5))))).await.unwrap();
            up_tx.send(Ok(delta(&half, Some((30, 6))))).await.unwrap();
            let (mut client, _present) = client_watch();
            let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
            let mut assembler = Assembler::new();
            let ended = collect_answer(up_rx, &mut client, deadline, &mut assembler).await;
            assert!(matches!(ended, Err((EarlyEnding::AnswerTooLarge, _))));
            assert!(up_tx.is_closed());
        }

        /// **An early ending carries its refund and no `[DONE]`**, so a
        /// client reads it as the interrupted answer it is; a finished one
        /// ends with `[DONE]` as always.
        #[tokio::test]
        async fn an_early_ending_sends_its_refund_and_no_done() {
            use futures_util::StreamExt;
            for finished in [false, true] {
                let (tx, padded) = padding::channel(CADENCE);
                let auth = AuthContext {
                    method: AuthMethod::AnonymousCredential,
                };
                let refund = RefundInfo {
                    refund: "r".to_string(),
                    issuer_key_id: "k".to_string(),
                };
                send_metadata_event(
                    &tx,
                    vec![refund],
                    build_privacy_metadata(&auth, true, "p"),
                    build_verification_metadata(None),
                    String::new(),
                    finished,
                    tokio::time::Instant::now() + Duration::from_secs(5),
                )
                .await;
                drop(tx);
                let bytes: Vec<u8> = padded.map(|frame| frame.unwrap().to_vec()).concat().await;
                let text = String::from_utf8(bytes).unwrap();
                assert!(text.contains("\"refunds\""), "{text}");
                assert_eq!(text.contains("[DONE]"), finished, "{text}");
            }
        }

        /// A chunk carrying the running count alone (a step that generated
        /// tokens and gave out no text) is kept for billing and never reaches
        /// the client: a stream cut after it is charged by its count, and the
        /// client's bytes are those of the stream without it.
        #[tokio::test]
        async fn a_count_alone_is_kept_and_never_forwarded() {
            use futures_util::StreamExt;
            let count_alone = |p: u32, c: u32| {
                let chunk = serde_json::json!({
                    "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": null}],
                    "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c},
                });
                BackendStreamEvent::Chunk(serde_json::from_value(chunk).unwrap(), Instant::now())
            };
            let run = async |with_counts: bool| {
                let (up_tx, up_rx) = mpsc::channel(8);
                let (tx, padded) = padding::channel(CADENCE);
                up_tx.send(Ok(delta("a", Some((30, 3))))).await.unwrap();
                if with_counts {
                    up_tx.send(Ok(count_alone(30, 40))).await.unwrap();
                    up_tx.send(Ok(count_alone(30, 80))).await.unwrap();
                }
                // The engine's connection breaks while a tool call is read.
                drop(up_tx);
                let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
                let mut timing = timing();
                let (ending, usage) = forward_stream(up_rx, &tx, deadline, "m", &mut timing).await;
                assert!(matches!(
                    ending,
                    StreamEnding::Early(EarlyEnding::ChannelClosed)
                ));
                drop(tx);
                let bytes: Vec<u8> = padded.map(|frame| frame.unwrap().to_vec()).concat().await;
                // The events, without the padding between them (the number
                // of padding frames follows the clock).
                let events: Vec<String> = String::from_utf8(bytes)
                    .unwrap()
                    .lines()
                    .filter(|line| line.starts_with("data:"))
                    .map(|line| line.trim_end().to_string())
                    .collect();
                (usage.map(|u| u.completion_tokens), events)
            };
            let (count, events) = run(true).await;
            assert_eq!(count, Some(80), "the cut is charged by the last count");
            let (plain_count, plain_events) = run(false).await;
            assert_eq!(plain_count, Some(3));
            assert_eq!(events.len(), 1, "{events:?}");
            assert_eq!(events, plain_events, "the client's events are unchanged");
        }

        /// A completed stream forwards no running count to the client: only
        /// the final usage chunk (no choices) carries usage, as it always did.
        #[tokio::test]
        async fn the_running_count_never_reaches_the_client() {
            use futures_util::StreamExt;
            let (up_tx, up_rx) = mpsc::channel(8);
            let (tx, padded) = padding::channel(CADENCE);
            up_tx.send(Ok(delta("a", Some((30, 3))))).await.unwrap();
            let final_usage = serde_json::json!({
                "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": [],
                "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
            });
            up_tx
                .send(Ok(BackendStreamEvent::Chunk(
                    serde_json::from_value(final_usage).unwrap(),
                    Instant::now(),
                )))
                .await
                .unwrap();
            let meta = BackendMeta {
                provider: "eidola".to_string(),
                chat_id: Some("c".to_string()),
                backend_model: "m".to_string(),
                tee_type: None,
                usage: None,
            };
            up_tx
                .send(Ok(BackendStreamEvent::Done(meta, Instant::now())))
                .await
                .unwrap();
            let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
            let mut timing = timing();
            let (ending, usage) = forward_stream(up_rx, &tx, deadline, "m", &mut timing).await;
            assert!(matches!(ending, StreamEnding::Done(..)));
            assert_eq!(usage.map(|u| u.completion_tokens), Some(5));
            drop(tx);
            let bytes: Vec<u8> = padded.map(|frame| frame.unwrap().to_vec()).concat().await;
            let text = String::from_utf8(bytes).unwrap();
            assert_eq!(text.matches("\"usage\"").count(), 1, "{text}");
            assert!(text.contains("\"completion_tokens\":5"));
            assert!(!text.contains("\"completion_tokens\":3"));
        }
    }

    #[test]
    fn preflight_requires_the_hold_to_equal_the_worst_case() {
        let req = request(&["hello world!"], 100);
        let model = test_model();
        let wc = worst_case_cost(&req, &model);

        // One credit short → 402 PaymentRequired, before anything upstream.
        let err = check_hold_equals_worst_case(wc - 1, &req, &model)
            .expect_err("hold below the minimum must be rejected");
        assert!(
            matches!(err, ServerError::PaymentRequired { .. }),
            "got {err:?}"
        );

        // One credit over → refused too, and the log line names no amount.
        let err = check_hold_equals_worst_case(wc + 1, &req, &model)
            .expect_err("a hold above the worst case must be rejected");
        assert!(matches!(err, ServerError::BadRequest { .. }), "got {err:?}");
        assert!(!err.to_string().contains(&(wc + 1).to_string()));

        // Exactly the worst case → accepted.
        check_hold_equals_worst_case(wc, &req, &model).expect("the exact hold is accepted");
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
            check_hold_equals_worst_case(old_style_hold, &req, &model).is_err(),
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

    /// **No completion past the declared ceiling, and an absent field is the
    /// declared ceiling.** A request asking for more than the model's
    /// `max_output_tokens` is refused rather than clamped — its hold was sized
    /// on what it sent — and one naming none is priced at the declared
    /// ceiling, never at the context window.
    #[test]
    fn the_completion_side_is_held_to_the_declared_ceiling() {
        let model = test_model(); // declares 4096, context 8192
        assert!(check_completion_ceiling(&request(&["hi"], 4_096), &model).is_ok());
        assert!(check_completion_ceiling(&request(&["hi"], 1), &model).is_ok());
        let refused = check_completion_ceiling(&request(&["hi"], 4_097), &model).unwrap_err();
        assert_eq!(refused.status_code(), axum::http::StatusCode::BAD_REQUEST);

        let absent: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap();
        assert!(check_completion_ceiling(&absent, &model).is_ok());
        assert_eq!(effective_max_completion(&absent, &model), 4_096);
        assert_eq!(
            worst_case_cost(&absent, &model),
            worst_case_cost(&request(&["hi"], 4_096), &model),
            "an absent field is priced as the declared ceiling"
        );

        // A row declaring nothing keeps the window as its last resort.
        let mut undeclared = test_model();
        undeclared.max_output_tokens = None;
        assert!(check_completion_ceiling(&request(&["hi"], 8_000), &undeclared).is_ok());
        assert_eq!(effective_max_completion(&absent, &undeclared), 8_192);
    }

    /// The two halves are wired into the handler: the default is applied
    /// before anything reads the request, and the refusal sits beside the
    /// charge check it precedes. A source scan, because the handler needs a
    /// database and what is asserted is that a call exists in a place.
    #[test]
    fn the_handler_applies_the_ceiling_before_pricing() {
        let source = include_str!("handlers.rs");
        let body_of = |name: &str| {
            let start = source
                .find(&format!("fn {name}("))
                .unwrap_or_else(|| panic!("{name} exists"));
            let end = start + source[start..].find("\n}\n").expect("a closing brace");
            &source[start..end]
        };
        let handler = body_of("chat_completions");
        let default = handler
            .find("default_completion_ceiling(")
            .expect("chat_completions gives an absent ceiling the declared one");
        let first_read = handler
            .find("chat_completions_phases(")
            .expect("and then runs the phases");
        assert!(
            default < first_read,
            "the default precedes every read of the request"
        );

        let validate = body_of("validate_request");
        let check = validate
            .find("check_completion_ceiling(")
            .expect("validate_request refuses a ceiling past the declared one");
        let charge = validate
            .find("check_hold_equals_worst_case(")
            .expect("and checks the charge");
        assert!(check < charge, "{validate}");
    }

    /// **A request's charge never depends on whether its prompt was served
    /// from an engine's prefix cache.** Two usage reports in the exact shape
    /// an Eidola engine sends (`eidola-server-engine`'s `usage()`), alike but
    /// for `prompt_tokens_details.cached_tokens` — none of the prompt cached,
    /// and nearly all of it — settle to one charge, for a token-priced model
    /// and a per-request-priced one, with and without the contract's clamp
    /// biting. A discount for a cache hit would make the charge, which the
    /// refund reveals, say whether this prompt's prefix was seen before.
    #[test]
    fn the_settled_charge_ignores_prefix_cache_hits() {
        let engine_usage = |cached: u32| -> Usage {
            serde_json::from_value(serde_json::json!({
                "prompt_tokens": 900,
                "completion_tokens": 250,
                "total_tokens": 1_150,
                "prompt_tokens_details": {"cached_tokens": cached},
            }))
            .expect("the engine's usage shape parses")
        };
        let cold = engine_usage(0);
        let warm = engine_usage(896);

        let token_priced = test_model();
        let mut per_request = test_model();
        per_request.pricing.per_request = Some(ScaledPrice {
            value: 7 * PRICING_SCALE_FACTOR,
            scale_factor: PRICING_SCALE_FACTOR,
        });
        for model in [&token_priced, &per_request] {
            // Chargeable prompt above the reported count (the usual case) and
            // below it (the clamp charging the contract's figure).
            for chargeable in [2_000, 500] {
                let settle =
                    |usage: &Usage| settled_cost(Some(usage), model, chargeable, 4_096, 99_999);
                assert_eq!(
                    settle(&cold),
                    settle(&warm),
                    "{} with chargeable {chargeable}",
                    model.id
                );
            }
        }
        // And the figure is the full price of every prompt token reported.
        assert_eq!(
            settled_cost(Some(&warm), &token_priced, 2_000, 4_096, 99_999),
            900 + 2 * 250
        );
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
