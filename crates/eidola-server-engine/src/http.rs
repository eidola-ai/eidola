//! The HTTP surface: plain HTTP behind the enclave's TLS/attestation shim.
//!
//! | Route | Auth | Purpose |
//! |---|---|---|
//! | `POST /v1/chat/completions` | gateway token + weights hash | the strict chat subset, answered as a stream |
//! | `GET /v1/engine/info` | gateway token | model id, weights hash, build, executor, device |
//! | `GET /healthz` | none | content-free liveness/readiness |
//!
//! A chat request is refused, in this order and before anything is admitted: a missing
//! or wrong gateway token; a missing or different `X-Eidola-Weights-Sha256` (checked
//! before the body is read); a full read bound (checked before the body is read); a body
//! that is too large, malformed, outside the subset, or for another model; a full
//! admission bound. Only then is the prompt rendered, tokenized and submitted.
//!
//! **Every response is shaped** (`eidola_common::stream_shape`): a chat answer, always a
//! stream, is written as frames of one size at [`STREAM_CADENCE`] from the moment it
//! opens, and every other one, refusals included, is padded to a size bucket ([`shape_response`],
//! around every route). The host and the network between this node and the gateway see
//! how long a response ran and a padded one's bucket, not the size or timing of its
//! steps.
//!
//! **Memory before the engine is bounded.** A request holds a slot of the read bound
//! (`AppState::reading`, as many slots as the admission bound) from before its first
//! body byte is read until it holds an admission permit or is refused, so at most
//! `EIDOLA_ENGINE_MAX_REQUESTS` bodies (each at most [`MAX_BODY_BYTES`]) and their parses
//! exist outside admission at once, and at most as many parsed requests inside it. With
//! no read slot free a request is refused at once (`overloaded`), unread.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::Router;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use eidola_common::engine_protocol::STREAM_CADENCE;
use eidola_common::stream_shape::{Framer, StreamEvent, pad_body};
use eidola_engine::engine::SubmitError;
use eidola_engine::secret::SaltDeriver;
use futures_util::stream::Stream;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TryRecvError;
use tokio::time::{Interval, MissedTickBehavior};

use crate::api::{self, ValidRequest};
use crate::auth::GatewayToken;
use crate::config::WeightsStorage;
use crate::error::ApiError;
use crate::model::LoadedModel;
use crate::pipeline::{self, Decoder, Prepared};
use crate::worker::{Admission, EngineHandle, Output, RequestGuard};

/// The request header carrying the weights hash the gateway expects.
pub const WEIGHTS_HEADER: &str = eidola_common::engine_protocol::WEIGHTS_HEADER;

/// Largest accepted request body.
pub const MAX_BODY_BYTES: usize = eidola_common::engine_protocol::MAX_REQUEST_BODY_BYTES;

/// The device an executor runs on, as `/v1/engine/info` reports it: the hardware and
/// the kernel image, nothing about any request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceReport {
    /// The driver's device name (`NVIDIA B300 SXM6 AC`).
    pub name: String,
    /// `major.minor`.
    pub compute_capability: String,
    /// The kernel image the device runs (`sm_103a`), when the build has one for it.
    pub image: Option<&'static str>,
    pub sm_count: u32,
    pub memory_bytes: u64,
}

/// Shared state of every handler.
pub struct AppState {
    pub model_id: String,
    pub executor: &'static str,
    /// The executor's device; `None` for the CPU executor.
    pub device: Option<DeviceReport>,
    pub max_model_len: u32,
    pub model: Arc<LoadedModel>,
    pub token: GatewayToken,
    pub salts: SaltDeriver,
    pub engine: EngineHandle,
    pub admission: Arc<Admission>,
    /// The read bound: requests whose body is being read or parsed (module docs).
    pub reading: Arc<Admission>,
}

/// The most event bytes a streamed response holds waiting for frames before it stops
/// taking the engine's output.
///
/// A stream reads the engine's output for it once a tick, so this is what the request
/// holds when the engine runs faster than [`STREAM_CADENCE`] carries: five seconds of
/// frames, which bounds the latency it queues, and memory per stream rather than the
/// whole answer. Past it the request's event buffer fills, and the engine cuts it off as
/// it cuts off a reader that has stopped (`worker::EVENT_BUFFER`). One event larger than
/// this on its own is still taken, and cut across frames.
pub const STREAM_BACKLOG_BYTES: usize = 1 << 20;

/// The node's router, every response shaped ([`shape_response`]).
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
        .layer(middleware::from_fn(shape_response))
}

/// Every response but a stream (which its frames shape), padded with trailing spaces to
/// its size bucket (`eidola_common::stream_shape::body_bucket`), with a matching
/// `Content-Length`. Around every route and the fallback, so it covers what no handler
/// sees too: an authentication refusal, a 404, a 405. Trailing whitespace leaves a JSON
/// body's value unchanged; the gateway reads `/healthz` without it.
async fn shape_response(request: Request, next: Next) -> Response {
    let response = next.run(request).await;
    let status = response.status();
    let stream = response
        .headers()
        .get(header::CONTENT_TYPE)
        .is_some_and(|v| v.as_bytes().starts_with(b"text/event-stream"));
    if stream
        || status.is_informational()
        || status == StatusCode::NO_CONTENT
        || status == StatusCode::NOT_MODIFIED
    {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    let bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes.to_vec(),
        // Every body here is in memory; one that cannot be read is a defect, answered
        // the same padded way.
        Err(_) => {
            let failed = ApiError::Internal("the response could not be read");
            parts.status = failed.status();
            parts.headers.insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("application/json"),
            );
            failed.to_body().to_string().into_bytes()
        }
    };
    let padded = pad_body(bytes);
    parts
        .headers
        .insert(header::CONTENT_LENGTH, padded.len().into());
    Response::from_parts(parts, axum::body::Body::from(padded))
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
        match state.model.storage() {
            WeightsStorage::VerifiedReadonly => (StatusCode::OK, "ok").into_response(),
            WeightsStorage::DevWritable => {
                (StatusCode::OK, "ok; weights-storage=dev-writable").into_response()
            }
        }
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
        "weights_storage": state.model.storage().as_str(),
        "build": build_info(),
        "executor": state.executor,
        "device": state.device.as_ref().map(|d| json!({
            "name": d.name,
            "compute_capability": d.compute_capability,
            "image": d.image,
            "sm_count": d.sm_count,
            "memory_bytes": d.memory_bytes,
        })),
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
    // Held from before the first body byte until the request is admitted or refused, so
    // the bodies and parses outside admission are bounded too.
    // `Overloaded` only where nothing has been admitted: a gateway sends it to another
    // node (`eidola_common::engine_protocol::refused_before_admission`).
    let reading = state.reading.try_acquire().ok_or(ApiError::Overloaded)?;
    let body = axum::body::to_bytes(request.into_body(), MAX_BODY_BYTES)
        .await
        .map_err(|_| ApiError::PayloadTooLarge)?;
    let req = api::parse_request(&body, &state.model_id)?;
    drop(body);
    let permit = state.admission.try_acquire().ok_or(ApiError::Overloaded)?;
    drop(reading);

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
        window_bound,
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
    let decoder =
        Decoder::new(state.model.clone(), output, stop, call_prefix).window_bound(window_bound);
    let run = Run {
        _state: state,
        guard,
        events,
        response,
        cached_prompt_tokens: 0,
    };
    Ok(stream_response(run, decoder))
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
    /// Held for the life of the request, so the node it runs in outlives it.
    _state: Arc<AppState>,
    guard: RequestGuard,
    events: mpsc::Receiver<Output>,
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
    /// What one engine event (or the channel's end, `None`) produced.
    fn on_output(&mut self, output: Option<Output>, decoder: &mut Decoder) -> Polled {
        match output {
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

/// A finishing choice, carrying `continuable` on a `length` ending: whether the
/// reasoning and content it gave out can be sent back as a continued final message
/// that renders exactly the text the model generated (`OutputParser::resumes_exactly`).
/// A client offers continuation only where it is `true`.
fn finish_choice(mut choice: Value, continuable: Option<bool>) -> Value {
    if let Some(c) = continuable {
        choice["continuable"] = Value::Bool(c);
    }
    choice
}

fn chunk(r: &ResponseMeta, choices: Value, usage: Option<Value>) -> StreamEvent {
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
    StreamEvent::Json(c.to_string())
}

/// The only answer to an admitted request (`api::parse_request` refuses any request that
/// does not ask for a stream with `stream_options.include_usage`): a role chunk, one
/// chunk per non-empty delta, a finish chunk, the usage chunk (`choices: []`), then
/// `[DONE]`; a failure partway ends it with an `event: error`. Dropping the stream (the
/// client went away) drops the request's guard, which cancels it in the engine. A
/// client's blocking request is the gateway's to assemble from this.
///
/// The role chunk and every delta chunk also carry the running `usage` (the prompt, and the completion tokens decoded so far, held-back text
/// included), and a step that moved the count while giving out no delta (a tool call
/// still being read, a held stop prefix or partial tag, a skipped special token) sends
/// a chunk with an empty delta carrying the count alone: the gateway, which bills,
/// keeps the last count it received, so a request it cuts short (its client went
/// away, or it reached its deadline) is charged for what was generated rather than
/// for nothing or for everything. The gateway forwards only the final usage chunk to
/// its client, and never a chunk that carries the count alone.
///
/// **Written in frames** ([`Shaped`]): the events go into a [`Framer`], and the body is
/// one frame of [`STREAM_CADENCE`]'s size per tick, from the moment the response opens
/// (the role chunk is in the first) until the frame that carries the last event.
fn stream_response(run: Run, decoder: Decoder) -> Response {
    let mut events = Events {
        run,
        decoder,
        reported: 0,
    };
    let mut framer = Framer::new(STREAM_CADENCE);
    // The role chunk goes out at once, before the first token is computed.
    let running = events.running_usage();
    framer.push(&chunk(
        &events.run.response,
        json!([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]),
        Some(running),
    ));
    let mut interval = tokio::time::interval(STREAM_CADENCE.tick());
    // A late tick moves the schedule rather than writing two frames back to back.
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut response = axum::body::Body::from_stream(Shaped {
        events,
        framer,
        interval,
    })
    .into_response();
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    headers.insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-cache, no-transform"),
    );
    response
}

/// A streamed request's events, as the engine's output turns into them.
struct Events {
    run: Run,
    decoder: Decoder,
    /// The completion count the last chunk sent carried.
    reported: u32,
}

impl Events {
    /// The running `usage`.
    fn running_usage(&mut self) -> Value {
        self.reported = self.decoder.completion_tokens();
        usage(
            self.run.response.prompt_tokens,
            self.decoder.completion_tokens(),
            self.run.cached_prompt_tokens,
        )
    }

    /// Queue on `framer` the events one engine event (or the channel's end) produced.
    fn on_output(&mut self, output: Option<Output>, framer: &mut Framer) {
        match self.run.on_output(output, &mut self.decoder) {
            Polled::Step(step) => {
                if !step.delta.is_empty() {
                    let running = self.running_usage();
                    framer.push(&chunk(
                        &self.run.response,
                        json!([{"index": 0, "delta": step.delta.to_openai_delta(), "finish_reason": null}]),
                        Some(running),
                    ));
                } else if step.finish.is_none() && self.decoder.completion_tokens() != self.reported
                {
                    // The count moved and nothing was given out: the count alone.
                    let running = self.running_usage();
                    framer.push(&chunk(
                        &self.run.response,
                        json!([{"index": 0, "delta": {}, "finish_reason": null}]),
                        Some(running),
                    ));
                }
                if let Some(reason) = step.finish {
                    framer.push(&chunk(
                        &self.run.response,
                        json!([finish_choice(
                            json!({"index": 0, "delta": {}, "finish_reason": reason.as_str()}),
                            step.continuable
                        )]),
                        None,
                    ));
                    let r = &self.run.response;
                    framer.push(&chunk(
                        r,
                        json!([]),
                        Some(usage(
                            r.prompt_tokens,
                            self.decoder.completion_tokens(),
                            self.run.cached_prompt_tokens,
                        )),
                    ));
                    framer.push(&StreamEvent::Done);
                }
            }
            Polled::Failed(e) => {
                tracing::warn!("stream ended early: {e}");
                framer.push(&StreamEvent::Error(e.to_body().to_string()));
            }
        }
    }
}

/// The body of a streamed response: one frame per tick.
///
/// Each tick takes whatever the engine has produced since the last (up to
/// [`STREAM_BACKLOG_BYTES`] waiting), then writes one frame, so a write's size and time
/// depend on nothing the engine did. Writes nothing between ticks, and nothing more once
/// the frame carrying `[DONE]` or the error is out. A reader that stops reading stops
/// the ticks, and so the reading of the engine's output: the engine cuts the request off
/// once its buffer is full, as before.
struct Shaped {
    events: Events,
    framer: Framer,
    interval: Interval,
}

impl Stream for Shaped {
    type Item = Result<Vec<u8>, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.framer.done() && this.framer.backlog() == 0 {
            return Poll::Ready(None);
        }
        if this.interval.poll_tick(cx).is_pending() {
            return Poll::Pending;
        }
        while !this.framer.done() && this.framer.backlog() < STREAM_BACKLOG_BYTES {
            let output = match this.events.run.events.try_recv() {
                Ok(output) => Some(output),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => None,
            };
            this.events.on_output(output, &mut this.framer);
        }
        Poll::Ready(Some(Ok(this.framer.frame())))
    }
}
