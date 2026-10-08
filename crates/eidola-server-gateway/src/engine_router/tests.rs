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
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use tinfoil_shim_mock::{RunningShim, ShimConfig};
use tinfoil_verifier::AllowedMeasurement;

use super::*;
use crate::engine_trust::{PinnedWeights, PromptCachePolicy};
use crate::types::{
    Capability, Model, ModelCapabilities, ModelHosting, ModelPricing, OutputBudgetClass,
    PinnedWeightsCapability, PromptCacheCapability, ScaledPrice,
};
use eidola_common::engine_protocol::error_type as kind;

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
    let mut pin = AllowedMeasurement::sev_snp(hex::encode(measurement));
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
}

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

fn error(status: StatusCode, error_type: &str) -> Response {
    (
        status,
        axum::Json(serde_json::json!({
            "error": {"message": "refused", "type": error_type, "code": null}
        })),
    )
        .into_response()
}

async fn healthz(State(state): State<Arc<EngineState>>) -> Response {
    state.health_checks.fetch_add(1, Ordering::Relaxed);
    if state.healthy.load(Ordering::Relaxed) {
        state.health.clone().into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "unavailable").into_response()
    }
}

async fn info(State(state): State<Arc<EngineState>>, headers: HeaderMap) -> Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error");
    }
    axum::Json(serde_json::json!({
        "model": state.model,
        "weights_sha256": state.weights,
        "weights_storage": state.storage,
        "build": {"crate": "fake", "version": "0", "git_sha": null},
        "executor": "cuda",
        "device": null,
    }))
    .into_response()
}

fn chunk(
    state: &EngineState,
    choices: serde_json::Value,
    usage: Option<serde_json::Value>,
) -> Event {
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
    Event::default().data(body.to_string())
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
    match mode {
        Mode::Overloaded => return error(StatusCode::SERVICE_UNAVAILABLE, kind::OVERLOADED),
        Mode::Fail500 => return error(StatusCode::INTERNAL_SERVER_ERROR, kind::INTERNAL_ERROR),
        Mode::WeightsMismatch => {
            return error(StatusCode::PRECONDITION_FAILED, kind::WEIGHTS_HASH_MISMATCH);
        }
        Mode::UnavailableAfterSubmit => {
            return error(StatusCode::SERVICE_UNAVAILABLE, kind::ENGINE_UNAVAILABLE);
        }
        _ => {}
    }
    let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
    if request["stream"] != serde_json::json!(true) {
        return axum::Json(serde_json::json!({
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": state.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }],
            "usage": usage(),
        }))
        .into_response();
    }
    let role = chunk(
        &state,
        serde_json::json!([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]),
        None,
    );
    let rest: Vec<Event> = match mode {
        Mode::StreamError => vec![Event::default().event("error").data(
            serde_json::json!({"error": {"message": "the engine stopped", "type": "engine_unavailable", "code": null}})
                .to_string(),
        )],
        Mode::StreamTruncated => vec![],
        Mode::MalformedChunk | Mode::UnknownEvent => {
            let odd = if mode == Mode::MalformedChunk {
                Event::default().data(r#"{"id":"chatcmpl-fake","choices":"#)
            } else {
                Event::default().event("progress").data("{}")
            };
            vec![
                odd,
                chunk(
                    &state,
                    serde_json::json!([{"index": 0, "delta": {"content": "hello"}, "finish_reason": null}]),
                    None,
                ),
                chunk(&state, serde_json::json!([]), Some(usage())),
                Event::default().data("[DONE]"),
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
            Event::default().data("[DONE]"),
        ],
    };
    let hold = mode == Mode::Hold;
    let state = state.clone();
    let events = async_stream(role, hold, state, rest);
    Sse::new(events).into_response()
}

const WEIGHTS_HEADER_NAME: &str = eidola_common::engine_protocol::WEIGHTS_HEADER;

/// The role chunk at once; then, for a held stream, nothing until released;
/// then the rest.
fn async_stream(
    role: Event,
    hold: bool,
    state: Arc<EngineState>,
    rest: Vec<Event>,
) -> impl futures_util::Stream<Item = Result<Event, std::convert::Infallible>> {
    let head = futures_util::stream::once(async move { Ok(role) });
    let tail = futures_util::stream::once(async move {
        if hold {
            state.release.notified().await;
        }
        futures_util::stream::iter(rest.into_iter().map(Ok))
    })
    .flatten();
    head.chain(tail)
}

fn router_for(engines: &[&Engine], models: Vec<EngineModel>) -> EngineRouter {
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
        fast_config(),
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

/// A well-formed cache key's text for `i`.
fn key_text(i: u64) -> String {
    use base64::Engine as _;
    let mut bytes = [0u8; 32];
    bytes[..8].copy_from_slice(&i.to_le_bytes());
    bytes[31] = 0x5a;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
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

fn is_unavailable(result: &Result<BackendResponse, ServerError>) -> bool {
    matches!(result, Err(ServerError::ServiceUnavailable(_)))
}

/// A new upstream takes nothing until a probe has verified it; then a
/// request reaches it with the gateway's token, the compiled-in weights hash,
/// and exactly the body `engine_request_body` builds.
#[tokio::test]
async fn traffic_waits_for_a_probe_and_carries_the_pinned_headers() {
    let a = engine(Identity::of(MODEL_A)).await;
    let router = router_for(&[&a], models());
    router
        .apply_placement(&[a.row(MODEL_A, DEPLOYMENT_A)])
        .await;
    assert!(is_unavailable(
        &router.send(&request(MODEL_A, None, false)).await
    ));
    assert!(a.chats().is_empty());

    router.probe_due().await;
    let key = key_text(1);
    let sent = request(MODEL_A, Some(&key), false);
    let response = router.send(&sent).await.unwrap();
    assert_eq!(response.meta.provider, PROVIDER);
    assert!(matches!(
        response.meta.tee_type,
        Some(TeeType::EidolaEngine)
    ));
    assert_eq!(
        response.response.choices[0].message.content.as_deref(),
        Some("hello")
    );

    let chats = a.chats();
    assert_eq!(chats.len(), 1);
    assert_eq!(
        chats[0].authorization.as_deref(),
        Some(format!("Bearer {TOKEN}").as_str())
    );
    assert_eq!(chats[0].weights.as_deref(), Some(WEIGHTS_A));
    assert_eq!(chats[0].body, engine_request_body(&sent).to_vec());
    let body: serde_json::Value = serde_json::from_slice(&chats[0].body).unwrap();
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
                    .send(&request(MODEL_A, Some(&key), false))
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
        router.send(&request(MODEL_A, None, false)).await.unwrap();
        assert_eq!(served_by(&refs, &before), idle);
    }

    // Released, the two are equal again and take turns.
    engines[busy].state.release.notify_one();
    while held.recv().await.is_some() {}
    engines[busy].set_mode(Mode::Serve);
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..4 {
        let before = counts(&refs);
        router.send(&request(MODEL_A, None, false)).await.unwrap();
        seen.insert(served_by(&refs, &before));
    }
    assert_eq!(seen.len(), 2);
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
        &router.send(&request(MODEL_A, None, false)).await
    ));
    assert_eq!(a.state.health_checks.load(Ordering::Relaxed), 0);
    assert!(a.chats().is_empty());

    router
        .apply_placement(&[a.row(MODEL_A, DEPLOYMENT_A)])
        .await;
    router.probe_due().await;
    router.send(&request(MODEL_A, None, false)).await.unwrap();
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
        &router.send(&request(MODEL_A, None, false)).await
    ));
    assert_eq!(b.state.health_checks.load(Ordering::Relaxed), 0);
    assert!(b.chats().is_empty());

    router
        .apply_placement(&[b.row(MODEL_B, DEPLOYMENT_B)])
        .await;
    router.probe_due().await;
    router.send(&request(MODEL_B, None, false)).await.unwrap();
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
        router.send(&request(MODEL_A, None, false)).await.unwrap();
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
    for i in 0..8 {
        router
            .send(&request(MODEL_A, Some(&key_text(i)), false))
            .await
            .unwrap();
    }
    for e in bad {
        assert!(e.chats().is_empty());
    }
    assert!(refuses.chats().len() <= 1, "opened on its first refusal");
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
        .send(&request(MODEL_A, Some(&key), false))
        .await
        .unwrap();
    let home = served_by(&refs, &before);
    let other = 1 - home;

    // A server error is the engine's answer: returned, not retried, counted.
    engines[home].set_mode(Mode::Fail500);
    for _ in 0..2 {
        let result = router.send(&request(MODEL_A, Some(&key), false)).await;
        assert!(matches!(
            result,
            Err(ServerError::Backend { status: 500, .. })
        ));
    }
    let before = counts(&refs);
    router
        .send(&request(MODEL_A, Some(&key), false))
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
        .send(&request(MODEL_A, Some(&key), false))
        .await
        .unwrap();
    assert_eq!(served_by(&refs, &before), other);

    // Healthy again: after the doubled backoff, a probe restores it.
    engines[home].state.healthy.store(true, Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(250)).await;
    router.probe_due().await;
    let before = counts(&refs);
    router
        .send(&request(MODEL_A, Some(&key), false))
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
        .send(&request(MODEL_A, Some(&key), false))
        .await
        .unwrap();
    let home = served_by(&refs, &before);
    let other = 1 - home;

    engines[home].set_mode(Mode::Overloaded);
    for _ in 0..3 {
        let before = counts(&refs);
        router
            .send(&request(MODEL_A, Some(&key), false))
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
        .send(&request(MODEL_A, Some(&key), false))
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
    let refused = ValidatedRequest::from_bytes(bytes::Bytes::from(request_text(
        MODEL_A,
        None,
        false,
        r#","tool_choice":"required""#,
    )))
    .unwrap();
    assert!(matches!(
        router.send(&refused).await,
        Err(ServerError::BadRequest { .. })
    ));
    router
        .apply_placement(&[a.row(MODEL_A, DEPLOYMENT_A)])
        .await;
    router.probe_due().await;
    assert!(matches!(
        router.send(&refused).await,
        Err(ServerError::BadRequest { .. })
    ));
    assert!(matches!(
        router.send_stream(&refused).await,
        Err(ServerError::BadRequest { .. })
    ));
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
    let sent = request(MODEL_A, None, false);
    let charge = 1_000_000u128;
    let settle = |usage: Option<&crate::types::Usage>| {
        crate::handlers::settled_cost(
            usage,
            &model,
            crate::handlers::chargeable_prompt_tokens_for(sent.request()),
            crate::handlers::effective_max_completion(sent.request(), &model),
            charge,
        )
    };
    // 1 credit per prompt token, 2 per completion token.
    let expected = PROMPT_TOKENS as u128 + 2 * COMPLETION_TOKENS as u128;

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
        router.send(&request(MODEL_A, None, false)).await.unwrap();
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
        name: "A".into(),
        description: String::new(),
        context_length: 8192,
        max_output_tokens: None,
        output_budget_class: OutputBudgetClass::Standard,
        hosting: ModelHosting::Eidola,
        capabilities: ModelCapabilities {
            tool_calling: Capability::new(true),
            reasoning: Capability::new(true),
            input_modalities: vec![],
            output_modalities: vec![],
            prompt_cache: PromptCacheCapability::unsupported(),
            pinned_weights: PinnedWeightsCapability::unsupported(),
        },
        pricing: ModelPricing {
            per_prompt_token: price(crate::backend::PRICING_SCALE_FACTOR),
            per_completion_token: price(2 * crate::backend::PRICING_SCALE_FACTOR),
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
        .send(&request(MODEL_A, Some(&key), false))
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
                .send(&request(MODEL_A, Some(&key), false))
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

/// The parser takes complete events only, joins multi-line data, passes over
/// data-less blocks as the format defines, and refuses a block that is not
/// UTF-8 rather than repairing it.
#[test]
fn the_stream_parser_reads_events_and_refuses_bad_bytes() {
    let mut buffer =
        b": keep-alive\n\nevent: error\ndata: a\ndata: b\n\ndata: [DONE]\n\ndata: par".to_vec();
    assert_eq!(
        next_event(&mut buffer),
        Ok(Some(SseEvent {
            event: Some("error".into()),
            data: "a\nb".into()
        }))
    );
    assert_eq!(
        next_event(&mut buffer),
        Ok(Some(SseEvent {
            event: None,
            data: "[DONE]".into()
        }))
    );
    assert_eq!(next_event(&mut buffer), Ok(None));
    assert_eq!(buffer, b"data: par");

    let mut bad = b"data: \xff\xfe\n\n".to_vec();
    assert_eq!(next_event(&mut bad), Err(Unreadable));
}
