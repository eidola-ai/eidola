//! The router end to end: fake engines speaking the node's protocol, each
//! behind an in-process shim that serves real attestation (SEV-SNP reports
//! over a mock AMD root, with GPU evidence items when asked), reached through
//! the same attesting-client factory production uses. Only the roots differ:
//! production verifies TDX quotes to Intel's root, which no test can sign
//! for, so these models are pinned to the shims' SEV-SNP measurements.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use tinfoil_shim_mock::{RunningShim, ShimConfig};
use tinfoil_verifier::AllowedMeasurement;

use super::*;
use crate::engine_trust::{PinnedWeights, PromptCachePolicy};
use crate::types::{
    CacheKey, Capability, Model, ModelCapabilities, ModelPricing, OutputBudgetClass,
    PinnedWeightsCapability, PromptCacheCapability, ScaledPrice,
};
use eidola_common::engine_protocol::error_type as kind;
use eidola_common::stream_shape::{Framer, StreamEvent, pad_body};

const TOKEN: &str = "test-gateway-token-0123456789";

const MODEL_A: &str = "model-a";
const WEIGHTS_A: &str = "a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1";
const DEPLOYMENT_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const MEASUREMENT_A: [u8; 48] = [0xa0; 48];

const MODEL_B: &str = "model-b";
const WEIGHTS_B: &str = "b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2";
const DEPLOYMENT_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const MEASUREMENT_B: [u8; 48] = [0xb0; 48];

const PROMPT_TOKENS: u32 = 7;
const COMPLETION_TOKENS: u32 = 5;

fn pinned(id: &'static str, sha256: &'static str) -> PinnedModel {
    PinnedModel::fixture(
        id,
        PinnedWeights {
            sha256,
            repo: "example/model",
            revision: "0123456789abcdef0123456789abcdef01234567",
        },
        PromptCachePolicy {
            enabled: true,
            idle_ttl_secs: 900,
            max_age_secs: 7200,
        },
    )
}

fn pin(measurement: [u8; 48], gpus: Option<u32>) -> AllowedMeasurement {
    let mut pin = AllowedMeasurement::sev_snp(tinfoil_verifier::SevSnpPin {
        measurement: hex::encode(measurement),
        host_data: hex::encode([0u8; 32]),
        guest_policy: format!("{:#x}", tinfoil_shim_mock::MOCK_GUEST_POLICY),
    });
    pin.expected_gpus = gpus;
    pin
}

/// Model A pinned to deployment A, model B to deployment B, each attesting
/// its own measurement.
fn models() -> Vec<EngineModel> {
    vec![
        EngineModel::fixture(
            pinned(MODEL_A, WEIGHTS_A),
            vec![(DEPLOYMENT_A.into(), pin(MEASUREMENT_A, None))],
        ),
        EngineModel::fixture(
            pinned(MODEL_B, WEIGHTS_B),
            vec![(DEPLOYMENT_B.into(), pin(MEASUREMENT_B, None))],
        ),
    ]
}

fn fast_config() -> RouterConfig {
    RouterConfig {
        breaker: BreakerConfig {
            failure_threshold: 2,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_millis(400),
        },
        placement_refresh: Duration::from_secs(3600),
        probe_interval: Duration::from_secs(3600),
        probe_timeout: Duration::from_secs(10),
        request_attempts: 3,
        stream_headers_limit: Duration::from_secs(10),
        stream_idle_limit: Duration::from_secs(2),
    }
}

/// The one directory every shim's persistent roots live in, created once so
/// shims started concurrently never race to create it.
fn cert_dir() -> std::path::PathBuf {
    static DIR: OnceLock<std::path::PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let dir = std::env::temp_dir().join(format!(
            "eidola-gateway-engine-router-{}",
            std::process::id()
        ));
        tinfoil_shim_mock::ensure_persistent_material(&dir).expect("shim roots");
        dir
    })
    .clone()
}

/// How a fake engine answers chat requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Serve,
    Overloaded,
    Fail500,
    /// Refuse every chat request as the node does a wrong weights header.
    WeightsMismatch,
    /// Stream a role chunk, then the node's `event: error` frame.
    StreamError,
    /// Stream a role chunk, then end without `[DONE]`.
    StreamTruncated,
    /// Answer `engine_unavailable` (503), as the node does when its engine
    /// stops after the request was submitted.
    UnavailableAfterSubmit,
    /// Stream a chunk that does not parse among good ones, then `[DONE]`.
    MalformedChunk,
    /// Stream an event type the node never sends, then the rest.
    UnknownEvent,
    /// Stream a role chunk, then wait for `release`.
    Hold,
    /// Answer nothing, headers included, until `release`; then serve.
    Stall,
    /// Stream the role chunk's frame, then write nothing at all, the
    /// connection left open: a node that has stopped.
    Wedge,
    /// [`Mode::StreamError`], the role chunk carrying a running count of
    /// [`GENERATED`] completion tokens.
    CountThenError,
    /// [`Mode::Wedge`], the role chunk carrying a running count of
    /// [`GENERATED`] completion tokens.
    CountThenWedge,
    /// Refuse every chat request as the node does a declared cached prefix
    /// its cache does not hold (409 `cached_prefix_missing`).
    PrefixMissing,
    /// Refuse every chat request as the node does a prompt past its
    /// declaration (409 `prompt_budget_exceeded`).
    OverBudget,
}

/// The completion tokens a counting mode's engine reports before it fails.
const GENERATED: u32 = 3;

/// One chat request as an engine received it.
#[derive(Clone)]
struct Received {
    authorization: Option<String>,
    weights: Option<String>,
    body: Vec<u8>,
}

struct EngineState {
    model: String,
    weights: String,
    storage: String,
    health: String,
    mode: Mutex<Mode>,
    healthy: AtomicBool,
    chats: Mutex<Vec<Received>>,
    health_checks: AtomicUsize,
    release: tokio::sync::Notify,
    /// Held streams whose connection went away before they were released.
    abandoned: AtomicUsize,
}

/// A fake engine behind its shim.
struct Engine {
    state: Arc<EngineState>,
    shim: Option<RunningShim>,
    base_url: String,
    _server: tokio::task::JoinHandle<()>,
}

impl Engine {
    fn chats(&self) -> Vec<Received> {
        self.state.chats.lock().unwrap().clone()
    }

    fn set_mode(&self, mode: Mode) {
        *self.state.mode.lock().unwrap() = mode;
    }

    fn row(&self, model: &str, deployment: &str) -> EnginePlacementRow {
        EnginePlacementRow {
            model_id: model.into(),
            deployment: deployment.into(),
            base_url: self.base_url.clone(),
            enabled: true,
        }
    }
}

/// What a fake engine reports about itself.
struct Identity {
    model: &'static str,
    weights: &'static str,
    storage: &'static str,
    health: &'static str,
    measurement: [u8; 48],
    gpus: u32,
}

impl Identity {
    fn of(model: &'static str) -> Self {
        let (weights, measurement) = match model {
            MODEL_A => (WEIGHTS_A, MEASUREMENT_A),
            _ => (WEIGHTS_B, MEASUREMENT_B),
        };
        Self {
            model,
            weights,
            storage: "verified-readonly",
            health: "ok",
            measurement,
            gpus: 0,
        }
    }
}

async fn engine(identity: Identity) -> Engine {
    let cert_dir = cert_dir();
    let state = Arc::new(EngineState {
        model: identity.model.into(),
        weights: identity.weights.into(),
        storage: identity.storage.into(),
        health: identity.health.into(),
        mode: Mutex::new(Mode::Serve),
        healthy: AtomicBool::new(true),
        chats: Mutex::new(Vec::new()),
        health_checks: AtomicUsize::new(0),
        release: tokio::sync::Notify::new(),
        abandoned: AtomicUsize::new(0),
    });
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/engine/info", get(info))
        .route("/v1/chat/completions", post(chat))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let shim = tinfoil_shim_mock::start(ShimConfig {
        upstream_url: format!("http://{addr}"),
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        measurement: identity.measurement.to_vec(),
        host_data: [0; 32],
        cert_dir,
        gpus: identity.gpus,
    })
    .await
    .unwrap();
    Engine {
        state,
        base_url: format!("https://127.0.0.1:{}", shim.local_addr.port()),
        shim: Some(shim),
        _server: server,
    }
}

fn authorized(state: &EngineState, headers: &HeaderMap) -> bool {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v == format!("Bearer {TOKEN}"))
        && !state.model.is_empty()
}

/// A body padded to its size bucket, as the node pads every response that
/// is not a stream.
fn padded(status: StatusCode, content_type: &'static str, body: Vec<u8>) -> Response {
    (
        status,
        [(axum::http::header::CONTENT_TYPE, content_type)],
        pad_body(body),
    )
        .into_response()
}

fn padded_json(status: StatusCode, value: serde_json::Value) -> Response {
    padded(status, "application/json", value.to_string().into_bytes())
}

fn error(status: StatusCode, error_type: &str) -> Response {
    padded_json(
        status,
        serde_json::json!({
            "error": {"message": "refused", "type": error_type, "code": null}
        }),
    )
}

async fn healthz(State(state): State<Arc<EngineState>>) -> Response {
    state.health_checks.fetch_add(1, Ordering::Relaxed);
    if state.healthy.load(Ordering::Relaxed) {
        padded(
            StatusCode::OK,
            "text/plain",
            state.health.clone().into_bytes(),
        )
    } else {
        padded(
            StatusCode::SERVICE_UNAVAILABLE,
            "text/plain",
            b"unavailable".to_vec(),
        )
    }
}

async fn info(State(state): State<Arc<EngineState>>, headers: HeaderMap) -> Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error");
    }
    padded_json(
        StatusCode::OK,
        serde_json::json!({
            "model": state.model,
            "weights_sha256": state.weights,
            "weights_storage": state.storage,
            "build": {"crate": "fake", "version": "0", "git_sha": null},
            "executor": "cuda",
            "device": null,
        }),
    )
}

fn chunk(
    state: &EngineState,
    choices: serde_json::Value,
    usage: Option<serde_json::Value>,
) -> Written {
    let mut body = serde_json::json!({
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": state.model,
        "choices": choices,
    });
    if let Some(usage) = usage {
        body["usage"] = usage;
    }
    Written::Event(StreamEvent::Json(body.to_string()))
}

fn usage() -> serde_json::Value {
    serde_json::json!({
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": COMPLETION_TOKENS,
        "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
        "prompt_tokens_details": {"cached_tokens": 0},
    })
}

async fn chat(
    State(state): State<Arc<EngineState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error");
    }
    let header = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
    };
    state.chats.lock().unwrap().push(Received {
        authorization: header("authorization"),
        weights: header(WEIGHTS_HEADER_NAME),
        body: body.to_vec(),
    });
    let mode = *state.mode.lock().unwrap();
    if mode == Mode::Stall {
        state.release.notified().await;
    }
    match mode {
        Mode::Overloaded => return error(StatusCode::SERVICE_UNAVAILABLE, kind::OVERLOADED),
        Mode::Fail500 => return error(StatusCode::INTERNAL_SERVER_ERROR, kind::INTERNAL_ERROR),
        Mode::WeightsMismatch => {
            return error(StatusCode::PRECONDITION_FAILED, kind::WEIGHTS_HASH_MISMATCH);
        }
        Mode::UnavailableAfterSubmit => {
            return error(StatusCode::SERVICE_UNAVAILABLE, kind::ENGINE_UNAVAILABLE);
        }
        Mode::PrefixMissing => return error(StatusCode::CONFLICT, kind::CACHED_PREFIX_MISSING),
        Mode::OverBudget => return error(StatusCode::CONFLICT, kind::PROMPT_BUDGET_EXCEEDED),
        _ => {}
    }
    let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
    // The node answers only streams that report their usage.
    if request["stream"] != true || request["stream_options"]["include_usage"] != true {
        return error(StatusCode::BAD_REQUEST, kind::INVALID_REQUEST);
    }
    let counted = matches!(mode, Mode::CountThenError | Mode::CountThenWedge);
    let role = chunk(
        &state,
        serde_json::json!([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]),
        counted.then(|| {
            serde_json::json!({
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": GENERATED,
                "total_tokens": PROMPT_TOKENS + GENERATED,
                "prompt_tokens_details": {"cached_tokens": 0},
            })
        }),
    );
    let rest: Vec<Written> = match mode {
        Mode::StreamError | Mode::CountThenError => vec![Written::Event(StreamEvent::Error(
            serde_json::json!({"error": {"message": "the engine stopped", "type": "engine_unavailable", "code": null}})
                .to_string(),
        ))],
        Mode::StreamTruncated | Mode::Wedge | Mode::CountThenWedge => vec![],
        Mode::MalformedChunk | Mode::UnknownEvent => {
            let odd = if mode == Mode::MalformedChunk {
                Written::Event(StreamEvent::Json(
                    r#"{"id":"chatcmpl-fake","choices":"#.to_string(),
                ))
            } else {
                // A well-formed chunk, so only its event type is wrong. No
                // node writes one, so it goes out between frames.
                let Written::Event(StreamEvent::Json(data)) = chunk(
                    &state,
                    serde_json::json!([{"index": 0, "delta": {"content": "x"}, "finish_reason": null}]),
                    None,
                ) else {
                    unreachable!()
                };
                Written::Raw(format!("event: progress\ndata: {data}\n\n").into_bytes())
            };
            vec![
                odd,
                chunk(
                    &state,
                    serde_json::json!([{"index": 0, "delta": {"content": "hello"}, "finish_reason": null}]),
                    None,
                ),
                chunk(&state, serde_json::json!([]), Some(usage())),
                Written::Event(StreamEvent::Done),
            ]
        }
        _ => vec![
            chunk(
                &state,
                serde_json::json!([{"index": 0, "delta": {"content": "hello"}, "finish_reason": null}]),
                None,
            ),
            chunk(
                &state,
                serde_json::json!([{"index": 0, "delta": {}, "finish_reason": "stop"}]),
                None,
            ),
            chunk(&state, serde_json::json!([]), Some(usage())),
            Written::Event(StreamEvent::Done),
        ],
    };
    let hold = mode == Mode::Hold;
    let wedge = matches!(mode, Mode::Wedge | Mode::CountThenWedge);
    shaped_stream(role, hold, wedge, state, rest)
}

const WEIGHTS_HEADER_NAME: &str = eidola_common::engine_protocol::WEIGHTS_HEADER;

/// What a fake engine writes into its stream.
enum Written {
    /// An event, framed as the node frames it.
    Event(StreamEvent),
    /// Bytes no node writes, sent as they are between two frames.
    Raw(Vec<u8>),
}

/// The stream as the node writes it: frames at the node's cadence from the
/// moment it opens, the role chunk in the first; for a held stream, padding
/// until released; then the rest, ending after the frame that carries its
/// last event (or, with no `[DONE]`, once everything is out).
fn shaped_stream(
    role: Written,
    hold: bool,
    wedge: bool,
    state: Arc<EngineState>,
    rest: Vec<Written>,
) -> Response {
    /// Counts a held stream dropped (its connection gone) before release.
    struct Abandon(Option<Arc<EngineState>>);
    impl Drop for Abandon {
        fn drop(&mut self) {
            if let Some(state) = self.0.take() {
                state
                    .abandoned
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }
    }
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Written>();
    let _ = tx.send(role);
    tokio::spawn(async move {
        if hold {
            let mut abandon = Abandon(Some(state.clone()));
            tokio::select! {
                () = state.release.notified() => abandon.0 = None,
                () = tx.closed() => return,
            }
        }
        for written in rest {
            let _ = tx.send(written);
        }
    });
    let cadence = eidola_common::engine_protocol::STREAM_CADENCE;
    let interval = tokio::time::interval(cadence.tick());
    let body = futures_util::stream::unfold(
        (rx, Framer::new(cadence), interval, false, false),
        move |(mut rx, mut framer, mut interval, mut closed, mut started)| async move {
            if wedge && started {
                std::future::pending::<()>().await;
            }
            if framer.backlog() == 0 && (framer.done() || closed) {
                return None;
            }
            started = true;
            interval.tick().await;
            let mut out = Vec::new();
            loop {
                match rx.try_recv() {
                    Ok(Written::Event(event)) => framer.push(&event),
                    Ok(Written::Raw(raw)) => {
                        out.extend_from_slice(&framer.frame());
                        out.extend_from_slice(&raw);
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        closed = true;
                        break;
                    }
                }
            }
            out.extend_from_slice(&framer.frame());
            Some((
                Ok::<_, std::convert::Infallible>(bytes::Bytes::from(out)),
                (rx, framer, interval, closed, started),
            ))
        },
    );
    (
        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
        axum::body::Body::from_stream(body),
    )
        .into_response()
}

fn router_for(engines: &[&Engine], models: Vec<EngineModel>) -> EngineRouter {
    router_with(engines, models, fast_config())
}

fn router_with(
    engines: &[&Engine],
    models: Vec<EngineModel>,
    config: RouterConfig,
) -> EngineRouter {
    let shim = engines[0].shim.as_ref().unwrap();
    let mut tls_roots = rustls::RootCertStore::empty();
    tls_roots
        .add(rustls::pki_types::CertificateDer::from(
            shim.tls_ca_der.clone(),
        ))
        .unwrap();
    EngineRouter::new(
        models,
        EngineToken::new(TOKEN.into()).unwrap(),
        attesting_client_factory(AttestationRoots {
            tls_roots,
            trusted_ark_der: Some(shim.ark_der.clone()),
            trusted_ask_der: Some(shim.ask_der.clone()),
        }),
        config,
    )
}

fn request_text(model: &str, key: Option<&str>, stream: bool, extra: &str) -> String {
    let key = key
        .map(|k| format!(r#","cache_key":"{k}""#))
        .unwrap_or_default();
    format!(
        r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}],"max_completion_tokens":16,"stream":{stream}{key}{extra}}}"#
    )
}

fn request(model: &str, key: Option<&str>, stream: bool) -> ValidatedRequest {
    ValidatedRequest::from_bytes(bytes::Bytes::from(request_text(model, key, stream, ""))).unwrap()
}

/// A well-formed cache key's bytes for `i`.
fn key_bytes(i: u64) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[..8].copy_from_slice(&i.to_le_bytes());
    bytes[31] = 0x5a;
    bytes
}

/// A well-formed cache key's text for `i`.
fn key_text(i: u64) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(key_bytes(i))
}

/// The first `i` whose key ranks the upstream at `first` ahead of every other
/// upstream at `others`. Rendezvous weights hash each upstream's base URL,
/// which carries a port the OS picked, so which engine a fixed key reaches
/// differs from run to run; a test that needs a given engine tried first
/// chooses its key this way.
fn key_ranking_first(router: &EngineRouter, model: &str, first: &str, others: &[&str]) -> u64 {
    let ups: Vec<Arc<Upstream>> = std::iter::once(first)
        .chain(others.iter().copied())
        .map(|url| upstream(router, model, url))
        .collect();
    (0..)
        .find(|&i| {
            let ranked = rendezvous::rank_by_key(&CacheKey::from_bytes(key_bytes(i)), ups.clone());
            ranked[0].base_url == first
        })
        .unwrap()
}

/// Which engine served the last request: the one whose chat count grew.
fn served_by(engines: &[&Engine], before: &[usize]) -> usize {
    let grew: Vec<usize> = engines
        .iter()
        .enumerate()
        .filter(|(i, e)| e.chats().len() > before[*i])
        .map(|(i, _)| i)
        .collect();
    assert_eq!(grew.len(), 1, "exactly one engine served the request");
    grew[0]
}

fn counts(engines: &[&Engine]) -> Vec<usize> {
    engines.iter().map(|e| e.chats().len()).collect()
}

fn upstream(router: &EngineRouter, model: &str, base_url: &str) -> Arc<Upstream> {
    router.inner.upstreams.load()[model]
        .iter()
        .find(|u| u.base_url == base_url)
        .unwrap()
        .clone()
}

/// A request's whole answer as the router streams it: the content of its
/// chunks, and the `Done` that ends it.
struct Answered {
    content: String,
    meta: crate::backend::BackendMeta,
}

trait SendWhole {
    async fn send(&self, request: &ValidatedRequest) -> Result<Answered, ServerError>;
}

impl SendWhole for EngineRouter {
    async fn send(&self, request: &ValidatedRequest) -> Result<Answered, ServerError> {
        let mut rx = self.send_stream(request).await?;
        let mut content = String::new();
        while let Some(event) = rx.recv().await {
            match event? {
                BackendStreamEvent::Chunk(chunk, _) => {
                    for choice in &chunk.choices {
                        content.push_str(choice.delta.content.as_deref().unwrap_or(""));
                    }
                }
                BackendStreamEvent::Done(meta, _) => return Ok(Answered { content, meta }),
            }
        }
        Err(ServerError::Network(
            "the stream ended before it completed".to_string(),
        ))
    }
}

fn is_unavailable(result: &Result<Answered, ServerError>) -> bool {
    matches!(result, Err(ServerError::ServiceUnavailable(_)))
}

/// A new upstream takes nothing until a probe has verified it; then a
/// request reaches it with the gateway's token, the compiled-in weights hash,
/// and exactly the body `engine_request_body` builds, asking for a stream
/// with its usage whatever stream options the client sent.
#[tokio::test]
async fn traffic_waits_for_a_probe_and_carries_the_pinned_headers() {
    let a = engine(Identity::of(MODEL_A)).await;
    let router = router_for(&[&a], models());
    router
        .apply_placement(&[a.row(MODEL_A, DEPLOYMENT_A)])
        .await;
    assert!(is_unavailable(
        &router.send(&request(MODEL_A, None, true)).await
    ));
    assert!(a.chats().is_empty());

    router.probe_due().await;
    let key = key_text(1);
    let sent = request(MODEL_A, Some(&key), true);
    let response = router.send(&sent).await.unwrap();
    assert_eq!(response.meta.provider, PROVIDER);
    assert!(matches!(
        response.meta.tee_type,
        Some(TeeType::EidolaEngine)
    ));
    assert_eq!(response.content, "hello");

    let chats = a.chats();
    assert_eq!(chats.len(), 1);
    assert_eq!(
        chats[0].authorization.as_deref(),
        Some(format!("Bearer {TOKEN}").as_str())
    );
    assert_eq!(chats[0].weights.as_deref(), Some(WEIGHTS_A));
    assert_eq!(chats[0].body, engine_request_body(&sent).to_vec());
    let body: serde_json::Value = serde_json::from_slice(&chats[0].body).unwrap();
    assert_eq!(body["stream"], true);
    assert_eq!(
        body["stream_options"],
        serde_json::json!({"include_usage": true})
    );
    assert_eq!(body["cache_key"], key.as_str());
}

/// Every request carrying one cache key lands on one engine, and keys spread
/// over the engines.
#[tokio::test]
async fn a_conversation_stays_on_its_engine() {
    let engines = [
        engine(Identity::of(MODEL_A)).await,
        engine(Identity::of(MODEL_A)).await,
        engine(Identity::of(MODEL_A)).await,
    ];
    let refs: Vec<&Engine> = engines.iter().collect();
    let router = router_for(&refs, models());
    let rows: Vec<EnginePlacementRow> = engines
        .iter()
        .map(|e| e.row(MODEL_A, DEPLOYMENT_A))
        .collect();
    router.apply_placement(&rows).await;
    router.probe_due().await;

    let mut used = std::collections::BTreeSet::new();
    for i in 0..12 {
        let key = key_text(i);
        let mut homes = std::collections::BTreeSet::new();
        for stream in [false, true, false] {
            let before = counts(&refs);
            if stream {
                let mut rx = router
                    .send_stream(&request(MODEL_A, Some(&key), true))
                    .await
                    .unwrap();
                while rx.recv().await.is_some() {}
            } else {
                router
                    .send(&request(MODEL_A, Some(&key), true))
                    .await
                    .unwrap();
            }
            homes.insert(served_by(&refs, &before));
        }
        assert_eq!(homes.len(), 1, "key {i} moved between engines");
        used.extend(homes);
    }
    assert!(used.len() >= 2, "twelve keys all on one of three engines");
}

/// Without a key, a request goes to the engine with fewer requests in flight.
#[tokio::test]
async fn without_a_key_the_least_loaded_engine_is_chosen() {
    let engines = [
        engine(Identity::of(MODEL_A)).await,
        engine(Identity::of(MODEL_A)).await,
    ];
    let refs: Vec<&Engine> = engines.iter().collect();
    let router = router_for(&refs, models());
    let rows: Vec<EnginePlacementRow> = engines
        .iter()
        .map(|e| e.row(MODEL_A, DEPLOYMENT_A))
        .collect();
    router.apply_placement(&rows).await;
    router.probe_due().await;
    // One request held open on one engine.
    for e in &engines {
        e.set_mode(Mode::Hold);
    }
    let before = counts(&refs);
    let mut held = router
        .send_stream(&request(MODEL_A, None, true))
        .await
        .unwrap();
    held.recv().await.unwrap().unwrap();
    let busy = served_by(&refs, &before);
    let idle = 1 - busy;

    // While it is in flight, every request goes to the other engine, however
    // the tie-breaking rotation turns.
    engines[idle].set_mode(Mode::Serve);
    for _ in 0..4 {
        let before = counts(&refs);
        router.send(&request(MODEL_A, None, true)).await.unwrap();
        assert_eq!(served_by(&refs, &before), idle);
    }

    // Released, the two are equal again and take turns.
    engines[busy].state.release.notify_one();
    while held.recv().await.is_some() {}
    engines[busy].set_mode(Mode::Serve);
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..4 {
        let before = counts(&refs);
        router.send(&request(MODEL_A, None, true)).await.unwrap();
        seen.insert(served_by(&refs, &before));
    }
    assert_eq!(seen.len(), 2);
}

/// **A dropped receiver releases the engine at once**, even while the engine
/// is sending nothing (a long prefill): the relay watches for its handler
/// going away, closes the connection, and the engine sees its request go
/// (which the node turns into a cancel in its scheduler).
#[tokio::test]
async fn a_dropped_receiver_closes_the_engine_connection_without_waiting_for_output() {
    let a = engine(Identity::of(MODEL_A)).await;
    let router = router_for(&[&a], models());
    router
        .apply_placement(&[a.row(MODEL_A, DEPLOYMENT_A)])
        .await;
    router.probe_due().await;
    a.set_mode(Mode::Hold);
    let mut held = router
        .send_stream(&request(MODEL_A, None, true))
        .await
        .unwrap();
    held.recv().await.unwrap().unwrap();
    drop(held);
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while a.state.abandoned.load(std::sync::atomic::Ordering::SeqCst) == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the engine's stream was never released"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// A streaming request whose handler is dropped while the engine has not yet
/// answered (the client went away mid-dispatch) is still settled with its
/// final refund, and its spend leaves the live set: the dispatch and the
/// settlement run in the request's own task.
#[test]
fn a_stream_whose_handler_is_dropped_before_it_opens_is_settled() {
    // Signing a refund needs more stack than a test thread has.
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(a_stream_dropped_before_it_opens_is_settled())
        })
        .unwrap()
        .join()
        .unwrap();
}

async fn a_stream_dropped_before_it_opens_is_settled() {
    use crate::credentials::tests::{Cache, DAY, add_key, spend};
    use crate::credentials::verification_count;

    let a = engine(Identity::of(MODEL_A)).await;
    let router = router_for(&[&a], models());
    router
        .apply_placement(&[a.row(MODEL_A, DEPLOYMENT_A)])
        .await;
    router.probe_due().await;
    a.set_mode(Mode::Stall);

    let validated = request(MODEL_A, None, true);
    let model = catalog_model();
    let hold = crate::handlers::tests::worst_case_cost(validated.request(), &model).unwrap();
    let mut keys = Cache::new();
    let hash = add_key(&mut keys, std::time::SystemTime::now() + DAY);
    let (act, _) = spend(&keys[&hash], 1_000_000, hold);
    let proof = act.spend_proof.clone();
    let state = crate::AppState::new(
        crate::backend::Catalog::new(None).unwrap(),
        Some(router),
        crate::db::create_pool(
            "postgres://nobody@127.0.0.1:1/none?sslmode=disable",
            None,
            None,
        )
        .unwrap(),
        None,
        [0u8; 32],
        Arc::new(tokio::sync::RwLock::new(keys)),
    );
    crate::handlers::check_presented_spends(&state, std::slice::from_ref(&act))
        .await
        .unwrap();
    assert_eq!(verification_count::of(&proof), 1);

    let live = Arc::new(state.live_spends.serve(&act));
    let recording = async move {
        Ok(crate::handlers::tests::paid_inputs(
            vec![crate::auth::ActSpend {
                live: Some(live),
                ..act
            }],
            vec![hold],
        ))
    };
    let mut handler = Box::pin(crate::handlers::run_paid_request(
        state.clone(),
        validated,
        Ok(crate::backend::ResolvedModel {
            model,
            echo: MODEL_A.to_string(),
        }),
        recording,
    ));
    let until = std::time::Instant::now() + Duration::from_secs(10);
    while a.chats().is_empty() {
        assert!(
            std::time::Instant::now() < until,
            "the engine was never asked"
        );
        tokio::select! {
            _ = &mut handler => panic!("answered before the engine did"),
            () = tokio::time::sleep(Duration::from_millis(10)) => {}
        }
    }
    // The client goes away while the engine has answered nothing.
    drop(handler);
    assert_eq!(verification_count::of(&proof), 1);
    a.state.release.notify_one();

    let key_id = hex::encode(hash);
    let nullifier = proof.nullifier().as_bytes().to_vec();
    while verification_count::of(&proof) < 2 || state.live_spends.is_live(&key_id, &nullifier) {
        assert!(
            std::time::Instant::now() < until,
            "the request was never settled"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Only enabled rows naming a pinned model and one of its pinned deployments,
/// at an `https` URL, become upstreams; anything else is never contacted.
#[tokio::test]
async fn only_pinned_enabled_placement_is_routed() {
    let a = engine(Identity::of(MODEL_A)).await;
    let router = router_for(&[&a], models());
    let mut disabled = a.row(MODEL_A, DEPLOYMENT_A);
    disabled.enabled = false;
    let mut plain = a.row(MODEL_A, DEPLOYMENT_A);
    plain.base_url = plain.base_url.replace("https://", "http://");
    let rows = [
        disabled,
        a.row(MODEL_A, DEPLOYMENT_B),
        a.row("unpinned-model", DEPLOYMENT_A),
        plain,
    ];
    router.apply_placement(&rows).await;
    router.probe_due().await;
    assert!(is_unavailable(
        &router.send(&request(MODEL_A, None, true)).await
    ));
    assert_eq!(a.state.health_checks.load(Ordering::Relaxed), 0);
    assert!(a.chats().is_empty());

    router
        .apply_placement(&[a.row(MODEL_A, DEPLOYMENT_A)])
        .await;
    router.probe_due().await;
    router.send(&request(MODEL_A, None, true)).await.unwrap();
    assert_eq!(a.chats().len(), 1);
}

/// Each model's upstreams attest against that model's pins: an engine whose
/// measurement is model B's, placed under model A, never gets a request (not
/// even its health check, which the attesting client would carry only over an
/// attested connection), and serves model B.
#[tokio::test]
async fn an_engine_is_attested_against_its_models_pins() {
    let b = engine(Identity::of(MODEL_B)).await;
    let router = router_for(&[&b], models());
    router
        .apply_placement(&[b.row(MODEL_A, DEPLOYMENT_A)])
        .await;
    router.probe_due().await;
    assert!(!upstream(&router, MODEL_A, &b.base_url).routable());
    assert!(is_unavailable(
        &router.send(&request(MODEL_A, None, true)).await
    ));
    assert_eq!(b.state.health_checks.load(Ordering::Relaxed), 0);
    assert!(b.chats().is_empty());

    router
        .apply_placement(&[b.row(MODEL_B, DEPLOYMENT_B)])
        .await;
    router.probe_due().await;
    router.send(&request(MODEL_B, None, true)).await.unwrap();
    assert_eq!(b.chats().len(), 1);
}

/// A pin's `expected_gpus` is enforced on every handshake: exactly that many
/// nonce-bound GPU evidence items, no fewer and no more.
#[tokio::test]
async fn gpu_evidence_must_match_the_pin() {
    let gpus = |n| async move {
        engine(Identity {
            gpus: n,
            ..Identity::of(MODEL_A)
        })
        .await
    };
    let (two, one, three) = (gpus(2).await, gpus(1).await, gpus(3).await);
    let models = vec![EngineModel::fixture(
        pinned(MODEL_A, WEIGHTS_A),
        vec![(DEPLOYMENT_A.into(), pin(MEASUREMENT_A, Some(2)))],
    )];
    let router = router_for(&[&two, &one, &three], models);
    let rows = [
        two.row(MODEL_A, DEPLOYMENT_A),
        one.row(MODEL_A, DEPLOYMENT_A),
        three.row(MODEL_A, DEPLOYMENT_A),
    ];
    router.apply_placement(&rows).await;
    router.probe_due().await;
    assert!(upstream(&router, MODEL_A, &two.base_url).routable());
    assert!(!upstream(&router, MODEL_A, &one.base_url).routable());
    assert!(!upstream(&router, MODEL_A, &three.base_url).routable());
    for _ in 0..4 {
        router.send(&request(MODEL_A, None, true)).await.unwrap();
    }
    assert_eq!(two.chats().len(), 4);
}

/// An engine that reports other weights, another model, writable storage, or
/// a development health answer fails its probe and never takes traffic; one
/// whose chat endpoint refuses the pinned weights header is opened at once
/// and its request goes to another engine.
#[tokio::test]
async fn a_misregistered_engine_never_takes_traffic() {
    let wrong_weights = engine(Identity {
        weights: WEIGHTS_B,
        ..Identity::of(MODEL_A)
    })
    .await;
    let wrong_model = engine(Identity {
        model: MODEL_B,
        ..Identity::of(MODEL_A)
    })
    .await;
    let writable = engine(Identity {
        storage: "dev-writable",
        ..Identity::of(MODEL_A)
    })
    .await;
    let dev_health = engine(Identity {
        health: "ok; weights-storage=dev-writable",
        ..Identity::of(MODEL_A)
    })
    .await;
    let refuses = engine(Identity::of(MODEL_A)).await;
    let good = engine(Identity::of(MODEL_A)).await;
    let bad = [&wrong_weights, &wrong_model, &writable, &dev_health];
    let all = [
        &wrong_weights,
        &wrong_model,
        &writable,
        &dev_health,
        &refuses,
        &good,
    ];
    let router = router_for(&all, models());
    let rows: Vec<EnginePlacementRow> = all.iter().map(|e| e.row(MODEL_A, DEPLOYMENT_A)).collect();
    router.apply_placement(&rows).await;
    router.probe_due().await;
    for e in bad {
        assert!(!upstream(&router, MODEL_A, &e.base_url).routable());
    }
    refuses.set_mode(Mode::WeightsMismatch);
    // The first key tries the refusing engine first, so its refusal is seen.
    let first = key_ranking_first(&router, MODEL_A, &refuses.base_url, &[&good.base_url]);
    for i in first..first + 8 {
        router
            .send(&request(MODEL_A, Some(&key_text(i)), true))
            .await
            .unwrap();
    }
    for e in bad {
        assert!(e.chats().is_empty());
    }
    assert_eq!(refuses.chats().len(), 1, "opened on its first refusal");
    assert!(!upstream(&router, MODEL_A, &refuses.base_url).routable());
    assert_eq!(good.chats().len(), 8);
}

/// Consecutive failures open an upstream: its keys go to their next engine.
/// Once its backoff has passed and a probe succeeds, they come back.
#[tokio::test]
async fn failures_open_an_engine_and_a_probe_restores_it() {
    let engines = [
        engine(Identity::of(MODEL_A)).await,
        engine(Identity::of(MODEL_A)).await,
    ];
    let refs: Vec<&Engine> = engines.iter().collect();
    let router = router_for(&refs, models());
    let rows: Vec<EnginePlacementRow> = engines
        .iter()
        .map(|e| e.row(MODEL_A, DEPLOYMENT_A))
        .collect();
    router.apply_placement(&rows).await;
    router.probe_due().await;

    let key = key_text(42);
    let before = counts(&refs);
    router
        .send(&request(MODEL_A, Some(&key), true))
        .await
        .unwrap();
    let home = served_by(&refs, &before);
    let other = 1 - home;

    // A server error is the engine's answer: returned, not retried, counted.
    engines[home].set_mode(Mode::Fail500);
    for _ in 0..2 {
        let result = router.send(&request(MODEL_A, Some(&key), true)).await;
        assert!(matches!(
            result,
            Err(ServerError::Backend { status: 500, .. })
        ));
    }
    let before = counts(&refs);
    router
        .send(&request(MODEL_A, Some(&key), true))
        .await
        .unwrap();
    assert_eq!(
        served_by(&refs, &before),
        other,
        "opened after two failures"
    );

    // Still failing when the backoff passes: the probe fails, it stays open.
    engines[home].state.healthy.store(false, Ordering::Relaxed);
    engines[home].set_mode(Mode::Serve);
    tokio::time::sleep(Duration::from_millis(150)).await;
    router.probe_due().await;
    let before = counts(&refs);
    router
        .send(&request(MODEL_A, Some(&key), true))
        .await
        .unwrap();
    assert_eq!(served_by(&refs, &before), other);

    // Healthy again: after the doubled backoff, a probe restores it.
    engines[home].state.healthy.store(true, Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(250)).await;
    router.probe_due().await;
    let before = counts(&refs);
    router
        .send(&request(MODEL_A, Some(&key), true))
        .await
        .unwrap();
    assert_eq!(served_by(&refs, &before), home);
}

/// An engine that could not have run the request passes it on within the
/// same request: at capacity (not counted against its health), or gone.
#[tokio::test]
async fn an_overloaded_or_vanished_engine_passes_the_request_on() {
    let mut engines = [
        engine(Identity::of(MODEL_A)).await,
        engine(Identity::of(MODEL_A)).await,
    ];
    let refs: Vec<&Engine> = engines.iter().collect();
    let router = router_for(&refs, models());
    let rows: Vec<EnginePlacementRow> = engines
        .iter()
        .map(|e| e.row(MODEL_A, DEPLOYMENT_A))
        .collect();
    router.apply_placement(&rows).await;
    router.probe_due().await;

    let key = key_text(7);
    let before = counts(&refs);
    router
        .send(&request(MODEL_A, Some(&key), true))
        .await
        .unwrap();
    let home = served_by(&refs, &before);
    let other = 1 - home;

    engines[home].set_mode(Mode::Overloaded);
    for _ in 0..3 {
        let before = counts(&refs);
        router
            .send(&request(MODEL_A, Some(&key), true))
            .await
            .unwrap();
        assert_eq!(engines[other].chats().len(), before[other] + 1);
    }
    let home_url = engines[home].base_url.clone();
    assert!(upstream(&router, MODEL_A, &home_url).routable());

    // Its shim goes away: no attested connection, so the request moves on.
    engines[home].set_mode(Mode::Serve);
    engines[home].shim = None;
    let before_other = engines[other].chats().len();
    router
        .send(&request(MODEL_A, Some(&key), true))
        .await
        .unwrap();
    assert_eq!(engines[other].chats().len(), before_other + 1);
}

/// A request a node would refuse is refused here, and no engine sees it,
/// even when none is available.
#[tokio::test]
async fn a_request_outside_the_node_subset_never_reaches_an_engine() {
    let a = engine(Identity::of(MODEL_A)).await;
    let router = router_for(&[&a], models());
    // The extras each put a parsed request outside the node's subset: a
    // tool choice it cannot honour, and continuations its rule refuses (a
    // final message that is not an assistant's, a generation prompt turned
    // off with nothing continued, a field it does not continue).
    let refusals = [
        r#","tool_choice":"required""#,
        r#","continue_final_message":true"#,
        r#","add_generation_prompt":false"#,
        r#","continue_final_message":"tool_calls""#,
    ]
    .map(|extra| {
        ValidatedRequest::from_bytes(bytes::Bytes::from(request_text(
            MODEL_A, None, false, extra,
        )))
        .unwrap()
    });
    for refused in &refusals {
        assert!(matches!(
            router.send(refused).await,
            Err(ServerError::BadRequest { .. })
        ));
    }
    router
        .apply_placement(&[a.row(MODEL_A, DEPLOYMENT_A)])
        .await;
    router.probe_due().await;
    for refused in &refusals {
        assert!(matches!(
            router.send(refused).await,
            Err(ServerError::BadRequest { .. })
        ));
        assert!(matches!(
            router.send_stream(refused).await,
            Err(ServerError::BadRequest { .. })
        ));
    }
    assert!(a.chats().is_empty());
}

/// The usage an engine reports is what settles the charge, on both
/// transports, through the handler's own settlement; a stream that fails or
/// ends early is an error, which the handler settles as one.
#[tokio::test]
async fn the_engines_usage_settles_the_charge() {
    let a = engine(Identity::of(MODEL_A)).await;
    let router = router_for(&[&a], models());
    router
        .apply_placement(&[a.row(MODEL_A, DEPLOYMENT_A)])
        .await;
    router.probe_due().await;

    let model = catalog_model();
    let sent = request(MODEL_A, None, true);
    let charge = 1_000_000u128;
    let price = crate::handlers::RequestPrice::of(sent.request(), &model);
    let settle =
        |usage: Option<&crate::types::Usage>| crate::handlers::settled_cost(usage, &price, charge);
    // The fee, 1 credit per prompt token, 2 per completion token.
    let expected = eidola_common::pricing::REQUEST_FEE_CREDITS as u128
        + PROMPT_TOKENS as u128
        + 2 * COMPLETION_TOKENS as u128;

    let response = router.send(&sent).await.unwrap();
    assert_eq!(settle(response.meta.usage.as_ref()), expected);

    let mut rx = router
        .send_stream(&request(MODEL_A, None, true))
        .await
        .unwrap();
    let mut text = String::new();
    let mut done = None;
    while let Some(event) = rx.recv().await {
        match event.unwrap() {
            BackendStreamEvent::Chunk(chunk, _) => {
                for choice in &chunk.choices {
                    text.push_str(choice.delta.content.as_deref().unwrap_or(""));
                }
            }
            BackendStreamEvent::Done(meta, _) => done = Some(meta),
        }
    }
    assert_eq!(text, "hello");
    let meta = done.expect("a completed stream ends with Done");
    assert_eq!(meta.provider, PROVIDER);
    assert_eq!(settle(meta.usage.as_ref()), expected);
    // The forwarded stream request asked for usage.
    let body: serde_json::Value = serde_json::from_slice(&a.chats()[1].body).unwrap();
    assert_eq!(body["stream_options"]["include_usage"], true);

    for mode in [Mode::StreamError, Mode::StreamTruncated] {
        a.set_mode(mode);
        let mut rx = router
            .send_stream(&request(MODEL_A, None, true))
            .await
            .unwrap();
        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }
        assert!(matches!(
            events.first(),
            Some(Ok(BackendStreamEvent::Chunk(..)))
        ));
        // The node's error frame is the engine's own error; a stream that
        // just stops is a broken connection.
        match (mode, events.last()) {
            (Mode::StreamError, Some(Err(ServerError::Backend { error_type, .. }))) => {
                assert_eq!(error_type, "engine_unavailable")
            }
            (Mode::StreamTruncated, Some(Err(ServerError::Network(_)))) => {}
            (mode, last) => panic!("{mode:?} ended with {:?}", last.map(|e| e.is_ok())),
        }
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(BackendStreamEvent::Done(..)))),
            "{mode:?}"
        );
    }
}

/// Requests ride a pooled, already attested connection: one handshake (one
/// attestation, and on a GPU node one round of fresh SPDM reports) serves the
/// probe and every request after it.
#[tokio::test]
async fn pooled_connections_are_attested_once() {
    let a = engine(Identity::of(MODEL_A)).await;
    let router = router_for(&[&a], models());
    router
        .apply_placement(&[a.row(MODEL_A, DEPLOYMENT_A)])
        .await;
    router.probe_due().await;
    for _ in 0..5 {
        router.send(&request(MODEL_A, None, true)).await.unwrap();
    }
    assert_eq!(a.chats().len(), 5);
    assert_eq!(a.shim.as_ref().unwrap().attestations(), 1);
}

/// Re-reading placement keeps a known upstream (its client, connections and
/// health) and treats a changed one as new: unverified until probed.
#[tokio::test]
async fn placement_refresh_keeps_known_upstreams() {
    let a = engine(Identity::of(MODEL_A)).await;
    let router = router_for(&[&a], models());
    let row = a.row(MODEL_A, DEPLOYMENT_A);
    router.apply_placement(std::slice::from_ref(&row)).await;
    router.probe_due().await;
    let first = upstream(&router, MODEL_A, &a.base_url);
    assert!(first.routable());

    router.apply_placement(std::slice::from_ref(&row)).await;
    let again = upstream(&router, MODEL_A, &a.base_url);
    assert!(Arc::ptr_eq(&first, &again));
    assert!(again.routable());

    // The same URL under another of the model's deployments is a new upstream.
    let two_deployments = vec![EngineModel::fixture(
        pinned(MODEL_A, WEIGHTS_A),
        vec![
            (DEPLOYMENT_A.into(), pin(MEASUREMENT_A, None)),
            (DEPLOYMENT_B.into(), pin(MEASUREMENT_A, None)),
        ],
    )];
    let router = router_for(&[&a], two_deployments);
    router.apply_placement(std::slice::from_ref(&row)).await;
    router.probe_due().await;
    let before = upstream(&router, MODEL_A, &a.base_url);
    router
        .apply_placement(&[a.row(MODEL_A, DEPLOYMENT_B)])
        .await;
    let after = upstream(&router, MODEL_A, &a.base_url);
    assert!(!Arc::ptr_eq(&before, &after));
    assert!(!after.routable());

    // Removed from placement, it is gone.
    router.apply_placement(&[]).await;
    assert!(router.inner.upstreams.load().get(MODEL_A).is_none());
}

/// A token-priced catalog row with easy arithmetic: 1 credit per prompt
/// token, 2 per completion token.
fn catalog_model() -> Model {
    let price = |value| ScaledPrice {
        value,
        scale_factor: crate::backend::PRICING_SCALE_FACTOR,
    };
    Model {
        id: MODEL_A.into(),
        sha256: MODEL_A.into(),
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
        name: "A".into(),
        description: String::new(),
        context_length: 8192,
        max_output_tokens: None,
        output_budget_class: OutputBudgetClass::Standard,
        capabilities: ModelCapabilities {
            tool_calling: Capability::new(true),
            reasoning: Capability::new(true),
            input_modalities: vec![],
            output_modalities: vec![],
            prompt_cache: PromptCacheCapability::unsupported(),
            pinned_weights: PinnedWeightsCapability {
                supported: true,
                sha256: Some(WEIGHTS_A.into()),
                repo: None,
                revision: None,
            },
        },
        pricing: ModelPricing {
            per_prompt_token: price(crate::backend::PRICING_SCALE_FACTOR),
            per_completion_token: price(2 * crate::backend::PRICING_SCALE_FACTOR),
            per_cached_prompt_token: Some(price(crate::backend::PRICING_SCALE_FACTOR / 4)),
            per_request: None,
        },
    }
}

/// The production source reads the table `schema.sql` creates. Needs a
/// database with the schema applied (`DATABASE_URL`).
#[tokio::test]
#[ignore]
async fn placement_reads_the_schemas_table() {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = crate::db::create_pool(&url, None, None).unwrap();
    let model = format!("placement-test-{}", uuid::Uuid::new_v4());
    let client = pool.get().await.unwrap();
    for (url, enabled) in [("https://b.test", true), ("https://a.test", false)] {
        client
            .execute(
                "INSERT INTO engine_placement (model_id, deployment, base_url, enabled) \
                 VALUES ($1, $2, $3, $4)",
                &[&model, &DEPLOYMENT_A, &url, &enabled],
            )
            .await
            .unwrap();
    }
    let rows = placement::PostgresPlacement(pool.clone())
        .load()
        .await
        .unwrap();
    client
        .execute(
            "DELETE FROM engine_placement WHERE model_id = $1",
            &[&model],
        )
        .await
        .unwrap();
    let ours: Vec<EnginePlacementRow> = rows.into_iter().filter(|r| r.model_id == model).collect();
    let row = |url: &str, enabled| EnginePlacementRow {
        model_id: model.clone(),
        deployment: DEPLOYMENT_A.into(),
        base_url: url.into(),
        enabled,
    };
    assert_eq!(
        ours,
        vec![row("https://a.test", false), row("https://b.test", true)]
    );
}

/// An answer the node may give after the engine ran the request (its engine
/// stopped: `engine_unavailable`), or any answer that is not one of the
/// node's pre-admission refusals, is returned and never sent to another
/// engine; it counts against the upstream's health.
#[tokio::test]
async fn an_answer_after_admission_is_returned_not_retried() {
    let engines = [
        engine(Identity::of(MODEL_A)).await,
        engine(Identity::of(MODEL_A)).await,
    ];
    let refs: Vec<&Engine> = engines.iter().collect();
    let router = router_for(&refs, models());
    let rows: Vec<EnginePlacementRow> = engines
        .iter()
        .map(|e| e.row(MODEL_A, DEPLOYMENT_A))
        .collect();
    router.apply_placement(&rows).await;
    router.probe_due().await;

    let key = key_text(9);
    let before = counts(&refs);
    router
        .send(&request(MODEL_A, Some(&key), true))
        .await
        .unwrap();
    let home = served_by(&refs, &before);
    let other = 1 - home;

    engines[home].set_mode(Mode::UnavailableAfterSubmit);
    for stream in [false, true] {
        let before = counts(&refs);
        let result = if stream {
            router
                .send_stream(&request(MODEL_A, Some(&key), true))
                .await
                .map(|_| ())
        } else {
            router
                .send(&request(MODEL_A, Some(&key), true))
                .await
                .map(|_| ())
        };
        match result {
            Err(ServerError::Backend {
                status: 503,
                error_type,
                ..
            }) => assert_eq!(error_type, kind::ENGINE_UNAVAILABLE),
            other => panic!("expected the engine's 503, got ok={}", other.is_ok()),
        }
        assert_eq!(engines[home].chats().len(), before[home] + 1);
        assert_eq!(
            engines[other].chats().len(),
            before[other],
            "retried elsewhere"
        );
    }
    assert!(
        !upstream(&router, MODEL_A, &engines[home].base_url).routable(),
        "two failures open it"
    );
}

/// Anything in a stream the relay cannot read ends it with an error and
/// counts against the upstream: a later `[DONE]` never turns skipped output
/// into a completed, billed response.
#[tokio::test]
async fn an_unreadable_stream_event_ends_the_stream_as_an_error() {
    for mode in [Mode::MalformedChunk, Mode::UnknownEvent] {
        let a = engine(Identity::of(MODEL_A)).await;
        let router = router_for(&[&a], models());
        router
            .apply_placement(&[a.row(MODEL_A, DEPLOYMENT_A)])
            .await;
        router.probe_due().await;
        a.set_mode(mode);
        for _ in 0..2 {
            let mut rx = router
                .send_stream(&request(MODEL_A, None, true))
                .await
                .unwrap();
            let mut events = Vec::new();
            while let Some(event) = rx.recv().await {
                events.push(event);
            }
            assert!(
                matches!(events.last(), Some(Err(ServerError::Parse(_)))),
                "{mode:?}"
            );
            assert!(
                !events
                    .iter()
                    .any(|e| matches!(e, Ok(BackendStreamEvent::Done(..)))),
                "{mode:?}"
            );
            let text: String = events
                .iter()
                .filter_map(|e| match e {
                    Ok(BackendStreamEvent::Chunk(c, _)) => {
                        c.choices.first().and_then(|c| c.delta.content.clone())
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(text, "", "{mode:?}: nothing after the bad event is relayed");
        }
        assert!(
            !upstream(&router, MODEL_A, &a.base_url).routable(),
            "{mode:?}: counted as failures"
        );
    }
}

/// **A node that stops answering is let go of at once, not at the request's
/// deadline.** Response headers that do not come within the bound are no
/// response: the request goes to the next engine, and the silent one is
/// counted as failing. An open stream that delivers nothing, not even a
/// padding frame, for the idle bound ends as an error of that engine, which
/// the handler settles on what was generated (`early_cost`).
#[tokio::test(flavor = "multi_thread")]
async fn a_silent_engine_is_given_up_on_within_its_bounds() {
    let engines = [
        engine(Identity::of(MODEL_A)).await,
        engine(Identity::of(MODEL_A)).await,
    ];
    let refs: Vec<&Engine> = engines.iter().collect();
    let config = RouterConfig {
        stream_headers_limit: Duration::from_millis(500),
        stream_idle_limit: Duration::from_millis(500),
        ..fast_config()
    };
    let router = router_with(&refs, models(), config);
    let rows: Vec<EnginePlacementRow> = engines
        .iter()
        .map(|e| e.row(MODEL_A, DEPLOYMENT_A))
        .collect();
    router.apply_placement(&rows).await;
    router.probe_due().await;
    let (stalled, other) = (&engines[0], &engines[1]);
    let key = key_text(key_ranking_first(
        &router,
        MODEL_A,
        &stalled.base_url,
        &[&other.base_url],
    ));

    // No headers: the next engine serves it.
    stalled.set_mode(Mode::Stall);
    let done = tokio::time::timeout(Duration::from_secs(5), async {
        let mut rx = router
            .send_stream(&request(MODEL_A, Some(&key), true))
            .await
            .unwrap();
        let mut done = false;
        while let Some(event) = rx.recv().await {
            done |= matches!(event, Ok(BackendStreamEvent::Done(..)));
        }
        done
    })
    .await
    .expect("the silent engine is given up on long before this");
    assert!(done, "served by the other engine");
    assert_eq!(stalled.chats().len(), 1);
    assert_eq!(other.chats().len(), 1);
    stalled.state.release.notify_waiters();

    // An open stream that goes silent: an error of its own, counted.
    other.set_mode(Mode::Wedge);
    let only_other = router_with(&refs, models(), config);
    only_other
        .apply_placement(&[other.row(MODEL_A, DEPLOYMENT_A)])
        .await;
    only_other.probe_due().await;
    for _ in 0..2 {
        let events = tokio::time::timeout(Duration::from_secs(5), async {
            let mut rx = only_other
                .send_stream(&request(MODEL_A, None, true))
                .await
                .unwrap();
            let mut events = Vec::new();
            while let Some(event) = rx.recv().await {
                events.push(event);
            }
            events
        })
        .await
        .expect("the silent stream is ended long before this");
        assert!(
            matches!(events.first(), Some(Ok(BackendStreamEvent::Chunk(..)))),
            "the role chunk came through"
        );
        assert!(
            matches!(events.last(), Some(Err(ServerError::Network(_)))),
            "ends as a network error of the engine"
        );
    }
    assert!(
        !upstream(&only_other, MODEL_A, &other.base_url).routable(),
        "counted as failures"
    );
}

/// A paid request ended early is settled on what was generated, through the
/// chat handler's own settlement: an engine that fails partway or falls
/// silent is charged the prompt and the output it reported by its last chunk,
/// the rest of the hold refunded in the stream's metadata event, and the
/// stream ends without `[DONE]`; an engine that sends no response headers is
/// no engine, and with none other to try the request is refused and charged
/// the request fee alone.
#[test]
fn a_request_ended_early_is_settled_on_what_was_generated() {
    // Signing a refund needs more stack than a test thread has.
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(a_request_ended_early_is_settled())
        })
        .unwrap()
        .join()
        .unwrap();
}

async fn a_request_ended_early_is_settled() {
    let model = catalog_model();
    let hold =
        crate::handlers::tests::worst_case_cost(request(MODEL_A, None, true).request(), &model)
            .unwrap();
    // The fee, 1 credit a prompt token, 2 a completion token.
    let fee = eidola_common::pricing::REQUEST_FEE_CREDITS as u128;
    let generated = fee + PROMPT_TOKENS as u128 + 2 * GENERATED as u128;
    for (mode, status, charged) in [
        (Mode::CountThenError, 200, generated),
        (Mode::CountThenWedge, 200, generated),
        (Mode::Stall, 503, fee),
    ] {
        // An engine and router of its own, so no ending opens the breaker
        // for the next.
        let a = engine(Identity::of(MODEL_A)).await;
        let config = RouterConfig {
            stream_headers_limit: Duration::from_millis(500),
            stream_idle_limit: Duration::from_millis(500),
            ..fast_config()
        };
        let router = router_with(&[&a], models(), config);
        router
            .apply_placement(&[a.row(MODEL_A, DEPLOYMENT_A)])
            .await;
        router.probe_due().await;
        a.set_mode(mode);
        let started = std::time::Instant::now();
        let (answered, finished, credits) = paid_request(&router, hold).await;
        assert_eq!(answered, status, "{mode:?}");
        assert!(!finished, "{mode:?}: an early ending is no finished answer");
        assert_eq!(credits, CREDITS - charged, "{mode:?}");
        assert_eq!(a.chats().len(), 1, "{mode:?}: the engine was asked");
        if mode != Mode::CountThenError {
            // Given up on at its bound, not sooner.
            assert!(started.elapsed() >= Duration::from_millis(500), "{mode:?}");
        }
        a.state.release.notify_waiters();
    }
}

const CREDITS: u128 = 1_000_000;

/// What a client completes its refund with.
struct Completion {
    pre_refund: anonymous_credit_tokens::PreRefund,
    params: anonymous_credit_tokens::Params,
    spend_proof: anonymous_credit_tokens::SpendProof<{ eidola_common::spend_credentials::ACT_L }>,
    public: anonymous_credit_tokens::PublicKey,
}

type Handler = std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<axum::response::Response, ServerError>>>,
>;

/// A request through `router`, paid with a hold of `hold`, as the chat
/// handler runs it: the handler's future, what completes its refund, the
/// state it runs on, and its spend's key id and nullifier.
async fn paid_handler(
    router: &EngineRouter,
    hold: u128,
) -> (Handler, Completion, crate::AppState, String, Vec<u8>) {
    paid_handler_for(router, request(MODEL_A, None, true), hold).await
}

/// [`paid_handler`] for `validated`.
async fn paid_handler_for(
    router: &EngineRouter,
    validated: ValidatedRequest,
    hold: u128,
) -> (Handler, Completion, crate::AppState, String, Vec<u8>) {
    use crate::credentials::tests::{Cache, DAY, add_key, spend};

    let model = catalog_model();
    let mut keys = Cache::new();
    let hash = add_key(&mut keys, std::time::SystemTime::now() + DAY);
    let (act, pre_refund) = spend(&keys[&hash], CREDITS, hold);
    let completion = Completion {
        pre_refund,
        params: keys[&hash].params.clone(),
        spend_proof: act.spend_proof.clone(),
        public: keys[&hash].secret_key.public().clone(),
    };
    let nullifier = act.spend_proof.nullifier().as_bytes().to_vec();
    let state = crate::AppState::new(
        crate::backend::Catalog::new(None).unwrap(),
        Some(router.clone()),
        crate::db::create_pool(
            "postgres://nobody@127.0.0.1:1/none?sslmode=disable",
            None,
            None,
        )
        .unwrap(),
        None,
        [0u8; 32],
        Arc::new(tokio::sync::RwLock::new(keys)),
    );
    crate::handlers::check_presented_spends(&state, std::slice::from_ref(&act))
        .await
        .unwrap();
    let live = Arc::new(state.live_spends.serve(&act));
    let recording = async move {
        Ok(crate::handlers::tests::paid_inputs(
            vec![crate::auth::ActSpend {
                live: Some(live),
                ..act
            }],
            vec![hold],
        ))
    };
    let handler = Box::pin(crate::handlers::run_paid_request(
        state.clone(),
        validated,
        Ok(crate::backend::ResolvedModel {
            model,
            echo: MODEL_A.to_string(),
        }),
        recording,
    ));
    (handler, completion, state, hex::encode(hash), nullifier)
}

/// Pay for and answer one request through `router`, as the chat handler
/// does, and return the answer's status, whether its stream ended with
/// `[DONE]`, and the credits the client holds after completing its refund:
/// the one the stream's metadata event carried, or a refusal's.
async fn paid_request(router: &EngineRouter, hold: u128) -> (u16, bool, u128) {
    let (status, finished, credits, _) =
        paid_request_for(router, request(MODEL_A, None, true), hold).await;
    (status, finished, credits)
}

/// [`paid_request`] for `validated`, with the answer's body: a refusal's error
/// body, or a stream's metadata event.
async fn paid_request_for(
    router: &EngineRouter,
    validated: ValidatedRequest,
    hold: u128,
) -> (u16, bool, u128, serde_json::Value) {
    use base64::Engine as _;
    let (handler, c, _, _, _) = paid_handler_for(router, validated, hold).await;
    let (status, finished, body) = tokio::time::timeout(Duration::from_secs(10), async {
        let response = handler.await.unwrap();
        let status = response.status().as_u16();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        if status != 200 {
            return (
                status,
                false,
                serde_json::from_slice(unpadded(&bytes)).unwrap(),
            );
        }
        let mut reader = eidola_common::stream_shape::StreamReader::new();
        reader.push(&bytes);
        let (mut finished, mut metadata) = (false, serde_json::Value::Null);
        while let Some(event) = reader.next_event().unwrap() {
            match event {
                StreamEvent::Done => finished = true,
                StreamEvent::Json(json) => {
                    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
                    if value["object"] == "eidola.chat.completion.metadata" {
                        metadata = value;
                    }
                }
                StreamEvent::Error(_) => {}
            }
        }
        (status, finished, metadata)
    })
    .await
    .expect("answered within the bounds");
    let refund = body["refunds"][0]["refund"].as_str().unwrap();
    let refund = anonymous_credit_tokens::Refund::from_cbor(
        &base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(refund)
            .unwrap(),
    )
    .unwrap();
    let token = c
        .pre_refund
        .to_credit_token::<{ eidola_common::spend_credentials::ACT_L }>(
            &c.params,
            &c.spend_proof,
            &refund,
            &c.public,
        )
        .unwrap();
    let credits = anonymous_credit_tokens::scalar_to_credit::<
        { eidola_common::spend_credentials::ACT_L },
    >(&token.credits())
    .unwrap();
    (status, finished, credits, body)
}

/// **A node's refusal of a declared cached prefix reaches the client as the
/// node made it, and no other engine is tried.** Both refusals (`409`
/// `cached_prefix_missing`, `409` `prompt_budget_exceeded`) come back from the
/// router with their status and type and the gateway's own message, the
/// engine the key ranks second never sees the request, and the refusing
/// engine is not counted as failing.
#[tokio::test]
async fn a_declared_prefix_refusal_is_returned_not_retried() {
    let engines = [
        engine(Identity::of(MODEL_A)).await,
        engine(Identity::of(MODEL_A)).await,
    ];
    let refs: Vec<&Engine> = engines.iter().collect();
    let router = router_for(&refs, models());
    let rows: Vec<EnginePlacementRow> = engines
        .iter()
        .map(|e| e.row(MODEL_A, DEPLOYMENT_A))
        .collect();
    router.apply_placement(&rows).await;
    router.probe_due().await;
    let (home, other) = (&engines[0], &engines[1]);
    let key = key_text(key_ranking_first(
        &router,
        MODEL_A,
        &home.base_url,
        &[&other.base_url],
    ));
    let declared = |key: &str| {
        ValidatedRequest::from_bytes(bytes::Bytes::from(request_text(
            MODEL_A,
            Some(key),
            true,
            r#","cached_prefix":{"cached_tokens":256,"new_tokens":40}"#,
        )))
        .unwrap()
    };
    for (mode, expected) in [
        (Mode::PrefixMissing, kind::CACHED_PREFIX_MISSING),
        (Mode::OverBudget, kind::PROMPT_BUDGET_EXCEEDED),
    ] {
        home.set_mode(mode);
        // More refusals than the breaker's threshold: none is a failure.
        for _ in 0..3 {
            let before = other.chats().len();
            match router.send_stream(&declared(&key)).await {
                Err(ServerError::Backend {
                    status: 409,
                    error_type,
                    message,
                }) => {
                    assert_eq!(error_type, expected);
                    assert_ne!(message, "refused", "the node's message is not relayed");
                }
                other => panic!(
                    "{mode:?}: expected the node's 409, got ok={}",
                    other.is_ok()
                ),
            }
            assert_eq!(other.chats().len(), before, "{mode:?}: tried elsewhere");
        }
        assert!(
            upstream(&router, MODEL_A, &home.base_url).routable(),
            "{mode:?}: counted against the engine"
        );
    }
    // What the home engine received was the declaration, as priced.
    let sent: serde_json::Value = serde_json::from_slice(&home.chats()[0].body).unwrap();
    assert_eq!(
        sent["cached_prefix"],
        serde_json::json!({"cached_tokens": 256, "new_tokens": 40})
    );
}

/// **A paid request refused before an engine produced anything is charged
/// the request fee and nothing more**, through the chat handler's own
/// settlement: each of the node's two declared-prefix refusals, relayed with
/// its status and type; a request outside the node's subset (a declaration
/// without a `cache_key`), refused before routing; an engine's error before
/// it answered; and no engine at all. The declared hold is the declaration's.
#[test]
fn a_refused_paid_request_is_charged_the_fee() {
    // Signing a refund needs more stack than a test thread has.
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(a_refused_paid_request())
        })
        .unwrap()
        .join()
        .unwrap();
}

async fn a_refused_paid_request() {
    let fee = eidola_common::pricing::REQUEST_FEE_CREDITS as u128;
    let model = catalog_model();
    let key = key_text(3);
    let declaration = r#","cached_prefix":{"cached_tokens":256,"new_tokens":40}"#;
    let declared = request_text(MODEL_A, Some(&key), true, declaration);
    let keyless = request_text(MODEL_A, None, true, declaration);
    let plain = request_text(MODEL_A, None, true, "");
    let hold_of = |text: &str| {
        crate::handlers::tests::worst_case_cost(
            ValidatedRequest::from_bytes(bytes::Bytes::from(text.to_string()))
                .unwrap()
                .request(),
            &model,
        )
        .unwrap()
    };
    // 256 cached at 0.25, 40 new at 1, 16 completion at 2.
    assert_eq!(hold_of(&declared), fee + 64 + 40 + 32);

    for (case, mode, text, status, error_type) in [
        (
            "prefix missing",
            Some(Mode::PrefixMissing),
            &declared,
            409,
            kind::CACHED_PREFIX_MISSING,
        ),
        (
            "over budget",
            Some(Mode::OverBudget),
            &declared,
            409,
            kind::PROMPT_BUDGET_EXCEEDED,
        ),
        (
            "outside the subset",
            Some(Mode::Serve),
            &keyless,
            400,
            "invalid_request_error",
        ),
        (
            "an engine error",
            Some(Mode::Fail500),
            &plain,
            500,
            kind::INTERNAL_ERROR,
        ),
        ("no engine", None, &plain, 503, "service_unavailable"),
    ] {
        let a = engine(Identity::of(MODEL_A)).await;
        let router = router_for(&[&a], models());
        if let Some(mode) = mode {
            router
                .apply_placement(&[a.row(MODEL_A, DEPLOYMENT_A)])
                .await;
            router.probe_due().await;
            a.set_mode(mode);
        }
        let validated = ValidatedRequest::from_bytes(bytes::Bytes::from(text.clone())).unwrap();
        let (answered, finished, credits, body) =
            paid_request_for(&router, validated, hold_of(text)).await;
        assert_eq!(answered, status, "{case}");
        assert!(!finished, "{case}");
        assert_eq!(body["error"]["type"], error_type, "{case}");
        assert_eq!(credits, CREDITS - fee, "{case}: charged the fee alone");
        if case == "outside the subset" {
            assert!(a.chats().is_empty(), "{case}: no engine saw it");
        }
    }
}

/// A request whose client goes away while its engine works releases the
/// engine at once (its stream, dropped with the handler, finds its client
/// gone) and is still settled: the spend leaves the live set once its final
/// refund is written.
#[test]
fn a_request_whose_client_leaves_releases_its_engine() {
    // Signing a refund needs more stack than a test thread has.
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(a_client_leaves())
        })
        .unwrap()
        .join()
        .unwrap();
}

async fn a_client_leaves() {
    let a = engine(Identity::of(MODEL_A)).await;
    let router = router_for(&[&a], models());
    router
        .apply_placement(&[a.row(MODEL_A, DEPLOYMENT_A)])
        .await;
    router.probe_due().await;
    a.set_mode(Mode::Hold);
    let hold = crate::handlers::tests::worst_case_cost(
        request(MODEL_A, None, true).request(),
        &catalog_model(),
    )
    .unwrap();
    let (handler, _, state, key_id, nullifier) = paid_handler(&router, hold).await;
    let until = std::time::Instant::now() + Duration::from_secs(10);
    // The stream opens; the engine holds its answer open.
    let stream = tokio::time::timeout(Duration::from_secs(10), handler)
        .await
        .expect("the stream opened")
        .unwrap();
    assert_eq!(stream.status(), StatusCode::OK);
    while a.chats().is_empty() {
        assert!(
            std::time::Instant::now() < until,
            "the engine was never asked"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // The client leaves, and its stream with it.
    drop(stream);
    while a.state.abandoned.load(Ordering::SeqCst) == 0
        || state.live_spends.is_live(&key_id, &nullifier)
    {
        assert!(
            std::time::Instant::now() < until,
            "the engine was never released, or the request never settled"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// A cache key whose ranking over the upstreams at `order` is exactly that
/// order.
fn key_ranking(router: &EngineRouter, model: &str, order: &[&str]) -> u64 {
    let ups: Vec<Arc<Upstream>> = order
        .iter()
        .map(|url| upstream(router, model, url))
        .collect();
    (0..)
        .find(|&i| {
            let ranked = rendezvous::rank_by_key(&CacheKey::from_bytes(key_bytes(i)), ups.clone());
            ranked
                .iter()
                .map(|u| u.base_url.as_str())
                .eq(order.iter().copied())
        })
        .unwrap()
}

/// Three engines a key ranks stalled first, `second` next and `third` last,
/// with headers given up on quickly.
async fn ranked_behind_a_stall() -> ([Engine; 3], EngineRouter, String) {
    let engines = [
        engine(Identity::of(MODEL_A)).await,
        engine(Identity::of(MODEL_A)).await,
        engine(Identity::of(MODEL_A)).await,
    ];
    let refs: Vec<&Engine> = engines.iter().collect();
    let config = RouterConfig {
        stream_headers_limit: Duration::from_millis(500),
        ..fast_config()
    };
    let router = router_with(&refs, models(), config);
    let rows: Vec<EnginePlacementRow> = engines
        .iter()
        .map(|e| e.row(MODEL_A, DEPLOYMENT_A))
        .collect();
    router.apply_placement(&rows).await;
    router.probe_due().await;
    let order: Vec<&str> = engines.iter().map(|e| e.base_url.as_str()).collect();
    let key = key_text(key_ranking(&router, MODEL_A, &order));
    engines[0].set_mode(Mode::Stall);
    (engines, router, key)
}

/// **A half-open engine takes one request at a time, decided at the
/// attempt.** Requests planned while its slot was free, all reaching it at
/// once (each having waited on an engine ranked ahead that never answers),
/// send it exactly one; the rest go on to the next engine. Once the one it
/// took is served, it is healthy and takes every request.
#[tokio::test(flavor = "multi_thread")]
async fn a_half_open_engine_takes_one_request_however_many_arrive() {
    let ([stalled, recovering, next], router, key) = ranked_behind_a_stall().await;
    let config = fast_config().breaker;
    {
        let up = upstream(&router, MODEL_A, &recovering.base_url);
        up.breaker().trip(Instant::now(), &config);
        up.breaker().on_probe(true, Instant::now(), &config);
        assert!(up.routable(), "half-open, its slot free");
    }
    recovering.set_mode(Mode::Hold);
    let requests: Vec<_> = (0..4)
        .map(|_| {
            let router = router.clone();
            let key = key.clone();
            tokio::spawn(async move {
                router
                    .send_stream(&request(MODEL_A, Some(&key), true))
                    .await
                    .unwrap()
            })
        })
        .collect();
    let mut streams = Vec::new();
    for r in requests {
        streams.push(r.await.unwrap());
    }
    assert_eq!(
        stalled.chats().len(),
        4,
        "each tried the stalled engine first"
    );
    assert_eq!(recovering.chats().len(), 1, "one request at a time");
    assert_eq!(next.chats().len(), 3, "the rest went on");
    let up = upstream(&router, MODEL_A, &recovering.base_url);
    assert!(!up.routable(), "its slot is taken");

    // The one it took is served: healthy, every request admitted.
    recovering.state.release.notify_waiters();
    for mut rx in streams {
        while rx.recv().await.is_some() {}
    }
    stalled.state.release.notify_waiters();
    assert!(up.routable());
    let guards: Vec<_> = (0..3).map(|_| up.admit().expect("healthy")).collect();
    drop(guards);
}

/// **A planned fallback is rechecked when its turn comes.** A request whose
/// plan held a healthy engine behind one that never answers does not reach
/// it once it has opened in the meantime: with nothing else to try, it is
/// the answer no engine could take.
#[tokio::test(flavor = "multi_thread")]
async fn a_fallback_that_opened_while_the_request_waited_is_skipped() {
    let ([stalled, fallback, last], router, key) = ranked_behind_a_stall().await;
    last.set_mode(Mode::Stall);
    let pending = {
        let router = router.clone();
        let key = key.clone();
        tokio::spawn(async move {
            router
                .send_stream(&request(MODEL_A, Some(&key), true))
                .await
                .map(|_| ())
        })
    };
    // Trip the fallback as soon as the request is waiting on the first
    // engine, well inside the headers limit that moves it on.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while stalled.chats().is_empty() {
        assert!(
            std::time::Instant::now() < deadline,
            "the request never reached the first engine"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(stalled.chats().len(), 1, "waiting on the first engine");
    upstream(&router, MODEL_A, &fallback.base_url)
        .breaker()
        .trip(Instant::now(), &fast_config().breaker);
    let outcome = pending.await.unwrap();
    assert!(
        matches!(outcome, Err(ServerError::ServiceUnavailable(_))),
        "no engine took it"
    );
    assert_eq!(fallback.chats().len(), 0, "the opened engine was skipped");
    assert_eq!(last.chats().len(), 1, "the one after it was still tried");
    stalled.state.release.notify_waiters();
    last.state.release.notify_waiters();
}
