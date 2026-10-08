//! The HTTP surface: plain HTTP behind the enclave's TLS/attestation shim.
//!
//! | Route | Auth | Purpose |
//! |---|---|---|
//! | `POST /v1/chat/completions` | gateway token + weights hash | the strict chat subset, streaming or not |
//! | `GET /v1/engine/info` | gateway token | model id, weights hash, build, executor |
//! | `GET /healthz` | none | content-free liveness/readiness |
//!
//! A chat request is refused, in this order and before anything is admitted: a missing
//! or wrong gateway token; a missing or different `X-Eidola-Weights-Sha256` (checked
//! before the body is read); a body that is too large, malformed, outside the subset, or
//! for another model; a full admission bound. Only then is the prompt rendered,
//! tokenized and submitted.

use std::convert::Infallible;
use std::sync::Arc;

use axum::Router;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use eidola_engine::engine::SubmitError;
use eidola_engine::secret::SaltDeriver;
use eidola_engine_chat::ChatDelta;
use futures_util::stream::{self, Stream, StreamExt};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::api::{self, ValidRequest};
use crate::auth::GatewayToken;
use crate::error::ApiError;
use crate::model::LoadedModel;
use crate::pipeline::{self, Decoder, Prepared};
use crate::worker::{Admission, EngineHandle, Output, RequestGuard};

/// The request header carrying the weights hash the gateway expects.
pub const WEIGHTS_HEADER: &str = "x-eidola-weights-sha256";

/// Largest accepted request body.
pub const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

/// Shared state of every handler.
pub struct AppState {
    pub model_id: String,
    pub executor: &'static str,
    pub max_model_len: u32,
    pub model: Arc<LoadedModel>,
    pub token: GatewayToken,
    pub salts: SaltDeriver,
    pub engine: EngineHandle,
    pub admission: Arc<Admission>,
}

/// The node's router.
pub fn router(state: Arc<AppState>) -> Router {
    let authenticated = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/engine/info", get(engine_info))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_gateway,
        ));
    Router::new()
        .route("/healthz", get(healthz))
        .merge(authenticated)
        .with_state(state)
}

async fn require_gateway(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let auth = request
        .headers()
        .get(header::AUTHORIZATION)
        .map(|v| v.as_bytes());
    if !state.token.accepts(auth) {
        return ApiError::Unauthorized.into_response();
    }
    next.run(request).await
}

async fn healthz(State(state): State<Arc<AppState>>) -> Response {
    if state.engine.is_healthy() {
        (StatusCode::OK, "ok").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "unavailable").into_response()
    }
}

/// The engine build's identity. `git_sha` is the `EIDOLA_GIT_SHA` value the build was
/// given (absent when the build was not given one); nothing is read from the build
/// environment implicitly, so the binary stays a function of its inputs.
pub fn build_info() -> Value {
    json!({
        "crate": env!("CARGO_PKG_NAME"),
        "version": env!("CARGO_PKG_VERSION"),
        "git_sha": option_env!("EIDOLA_GIT_SHA"),
    })
}

async fn engine_info(State(state): State<Arc<AppState>>) -> Response {
    axum::Json(json!({
        "model": state.model_id,
        "weights_sha256": state.model.weights_hash(),
        "build": build_info(),
        "executor": state.executor,
    }))
    .into_response()
}

fn check_weights_header(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    let value = headers
        .get(WEIGHTS_HEADER)
        .ok_or(ApiError::WeightsHashRequired)?;
    let matches = value
        .to_str()
        .is_ok_and(|v| v.eq_ignore_ascii_case(state.model.weights_hash()));
    if matches {
        Ok(())
    } else {
        Err(ApiError::WeightsHashMismatch)
    }
}

async fn chat_completions(State(state): State<Arc<AppState>>, request: Request) -> Response {
    match chat(state, request).await {
        Ok(r) => r,
        Err(e) => e.into_response(),
    }
}

async fn chat(state: Arc<AppState>, request: Request) -> Result<Response, ApiError> {
    check_weights_header(&state, request.headers())?;
    let body = axum::body::to_bytes(request.into_body(), MAX_BODY_BYTES)
        .await
        .map_err(|_| ApiError::PayloadTooLarge)?;
    let req = api::parse_request(&body, &state.model_id)?;
    drop(body);
    let permit = state.admission.try_acquire().ok_or(ApiError::Overloaded)?;

    let stream = req.stream;
    let include_usage = req.include_usage;
    let id = state.engine.next_id();
    let prep_state = state.clone();
    // The permit bounds the work, so the work owns it: it moves into the blocking task
    // (which runs to completion even if this handler is dropped by a disconnect), comes
    // back with the prepared request, and goes on into the engine with the submission.
    // It is released only when the work it admitted has ended.
    let (prepared, permit) = tokio::task::spawn_blocking(move || {
        let prepared = prepare(&prep_state, id, req);
        prepared.map(|p| (p, permit))
    })
    .await
    .map_err(|_| ApiError::Internal("request preparation panicked"))??;
    let Prepared {
        request,
        prompt_tokens,
        output,
        stop,
    } = prepared;

    let (guard, mut events) = state
        .engine
        .submit(request, permit)
        .map_err(|_| ApiError::Unavailable)?;
    match events.recv().await {
        Some(Output::Accepted) => {}
        Some(Output::Rejected(e)) => return Err(submit_error(e, prompt_tokens)),
        _ => return Err(ApiError::Unavailable),
    }

    let response = ResponseMeta {
        id: pipeline::random_id("chatcmpl-"),
        created: unix_now(),
        model: state.model_id.clone(),
        prompt_tokens,
    };
    let call_prefix = pipeline::random_id("call_");
    let run = Run {
        state,
        guard,
        events,
        response,
        cached_prompt_tokens: 0,
    };
    if stream {
        Ok(stream_response(
            run,
            output,
            stop,
            call_prefix,
            include_usage,
        ))
    } else {
        complete_response(run, output, stop, call_prefix).await
    }
}

fn prepare(state: &AppState, id: u64, req: ValidRequest) -> Result<Prepared, ApiError> {
    pipeline::prepare(&state.model, &state.salts, state.max_model_len, id, req)
}

fn submit_error(e: SubmitError, prompt_tokens: u32) -> ApiError {
    match e {
        SubmitError::TooLong | SubmitError::ExceedsCapacity => ApiError::ContextLengthExceeded(
            format!("the prompt ({prompt_tokens} tokens) does not fit this node"),
        ),
        SubmitError::EmptyPrompt => ApiError::invalid("the prompt is empty"),
        SubmitError::ZeroMaxTokens => ApiError::invalid("max_completion_tokens must be at least 1"),
        SubmitError::InvalidToken | SubmitError::DuplicateId => {
            ApiError::Internal("the engine refused the request")
        }
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

struct ResponseMeta {
    id: String,
    created: u64,
    model: String,
    prompt_tokens: u32,
}

/// One admitted request in flight.
struct Run {
    state: Arc<AppState>,
    guard: RequestGuard,
    events: mpsc::UnboundedReceiver<Output>,
    response: ResponseMeta,
    cached_prompt_tokens: u32,
}

/// `usage` with `prompt_tokens_details.cached_tokens`: informational only (nothing here
/// prices; the gateway's charge never depends on it).
fn usage(prompt_tokens: u32, completion_tokens: u32, cached: u32) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached},
    })
}

/// What reading the next engine event produced.
enum Polled {
    Step(pipeline::Step),
    Failed(ApiError),
}

impl Run {
    async fn next(&mut self, decoder: &mut Decoder) -> Polled {
        match self.events.recv().await {
            Some(Output::Tokens {
                tokens,
                cached_prompt_tokens,
                finish,
            }) => {
                if let Some(c) = cached_prompt_tokens {
                    self.cached_prompt_tokens = c;
                }
                if finish.is_some() {
                    self.guard.finished();
                }
                match decoder.push(&tokens, finish) {
                    Ok(step) => {
                        if step.stopped_by_sequence {
                            self.guard.cancel();
                        }
                        Polled::Step(step)
                    }
                    Err(e) => Polled::Failed(e),
                }
            }
            Some(Output::Failed) | None => Polled::Failed(ApiError::Unavailable),
            Some(Output::Accepted) | Some(Output::Rejected(_)) => {
                Polled::Failed(ApiError::Internal("unexpected engine message"))
            }
        }
    }
}

async fn complete_response(
    mut run: Run,
    output: eidola_engine_chat::OutputConfig,
    stop: Vec<String>,
    call_prefix: String,
) -> Result<Response, ApiError> {
    let mut decoder = Decoder::new(run.state.model.clone(), output, stop, call_prefix);
    let mut all = ChatDelta::default();
    let reason = loop {
        match run.next(&mut decoder).await {
            Polled::Step(step) => {
                all.reasoning_content
                    .push_str(&step.delta.reasoning_content);
                all.content.push_str(&step.delta.content);
                all.tool_calls.extend(step.delta.tool_calls);
                if let Some(reason) = step.finish {
                    break reason;
                }
            }
            Polled::Failed(e) => return Err(e),
        }
    };
    let mut message = json!({"role": "assistant", "content": Value::Null});
    if !all.content.is_empty() {
        message["content"] = Value::String(all.content);
    }
    if !all.reasoning_content.is_empty() {
        message["reasoning_content"] = Value::String(all.reasoning_content);
    }
    if !all.tool_calls.is_empty() {
        message["tool_calls"] = all
            .tool_calls
            .iter()
            .map(|c| {
                let mut v = c.to_openai();
                if let Some(o) = v.as_object_mut() {
                    o.remove("index");
                }
                v
            })
            .collect();
    }
    let r = &run.response;
    Ok(axum::Json(json!({
        "id": r.id,
        "object": "chat.completion",
        "created": r.created,
        "model": r.model,
        "choices": [{"index": 0, "message": message, "finish_reason": reason.as_str()}],
        "usage": usage(r.prompt_tokens, decoder.completion_tokens(), run.cached_prompt_tokens),
    }))
    .into_response())
}

fn chunk(r: &ResponseMeta, choices: Value, usage: Option<Value>) -> Event {
    let mut c = json!({
        "id": r.id,
        "object": "chat.completion.chunk",
        "created": r.created,
        "model": r.model,
        "choices": choices,
    });
    if let Some(u) = usage {
        c["usage"] = u;
    }
    Event::default().data(c.to_string())
}

/// SSE: a role chunk, one chunk per non-empty delta, a finish chunk, the usage chunk
/// (`choices: []`) when `stream_options.include_usage`, then `[DONE]`. Dropping the
/// stream (the client went away) drops the request's guard, which cancels it in the
/// engine.
fn stream_response(
    run: Run,
    output: eidola_engine_chat::OutputConfig,
    stop: Vec<String>,
    call_prefix: String,
    include_usage: bool,
) -> Response {
    struct S {
        run: Run,
        decoder: Decoder,
        started: bool,
        finished: bool,
        include_usage: bool,
    }
    let decoder = Decoder::new(run.state.model.clone(), output, stop, call_prefix);
    let state = S {
        run,
        decoder,
        started: false,
        finished: false,
        include_usage,
    };
    let events = stream::unfold(state, |mut s| async move {
        if s.finished {
            return None;
        }
        let mut out: Vec<Event> = Vec::new();
        if !s.started {
            // The role chunk goes out at once, before the first token is computed.
            s.started = true;
            out.push(chunk(
                &s.run.response,
                json!([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]),
                None,
            ));
            return Some((stream::iter(out.into_iter().map(Ok::<_, Infallible>)), s));
        }
        match s.run.next(&mut s.decoder).await {
            Polled::Step(step) => {
                if !step.delta.is_empty() {
                    out.push(chunk(
                        &s.run.response,
                        json!([{"index": 0, "delta": step.delta.to_openai_delta(), "finish_reason": null}]),
                        None,
                    ));
                }
                if let Some(reason) = step.finish {
                    out.push(chunk(
                        &s.run.response,
                        json!([{"index": 0, "delta": {}, "finish_reason": reason.as_str()}]),
                        None,
                    ));
                    if s.include_usage {
                        let r = &s.run.response;
                        out.push(chunk(
                            r,
                            json!([]),
                            Some(usage(
                                r.prompt_tokens,
                                s.decoder.completion_tokens(),
                                s.run.cached_prompt_tokens,
                            )),
                        ));
                    }
                    out.push(Event::default().data("[DONE]"));
                    s.finished = true;
                }
            }
            Polled::Failed(e) => {
                tracing::warn!("stream ended early: {e}");
                out.push(Event::default().event("error").data(e.to_body().to_string()));
                s.finished = true;
            }
        }
        Some((stream::iter(out.into_iter().map(Ok::<_, Infallible>)), s))
    })
    .flatten();
    sse(events)
}

fn sse<S>(events: S) -> Response
where
    S: Stream<Item = Result<Event, Infallible>> + Send + 'static,
{
    Sse::new(events).into_response()
}
