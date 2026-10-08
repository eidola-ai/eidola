//! Routing requests for Eidola-hosted models to the engine deployments this
//! gateway pins.
//!
//! The pieces, and where trust comes from in each:
//!
//! - **Which engines exist** — `placement`: operator-edited rows in the
//!   gateway's Postgres, refreshed periodically and filtered to the
//!   deployments this build pins. Availability only, never trust.
//! - **Whether an engine is the one it claims to be** — the attesting client:
//!   every upstream's `reqwest::Client` is `tinfoil_verifier`'s, built over
//!   its model's compiled-in pins (`engine_trust::accepted_deployments`, TDX
//!   pins with `expected_gpus`, so the verifier also requires exactly that
//!   many NVIDIA GPU evidence items bound to each handshake's nonce). Every
//!   new connection is attested; requests on a pooled connection ride on its
//!   attestation, and every GPU-enclave handshake makes the node collect fresh
//!   SPDM reports, so keep-alive pooling is what keeps handshakes rare. One
//!   client per upstream, not one shared by the model's upstreams, because the
//!   verifier addresses its inline attestation request to the host of the base
//!   URL it was built with.
//! - **Whether an engine serves the pinned weights** — a probe before it takes
//!   traffic, and again periodically: `/healthz` must answer `ok` and
//!   `/v1/engine/info` must report the pinned model id and weights hash on
//!   `verified-readonly` storage. Every chat request also carries the pinned
//!   hash, which the node checks before reading the body.
//! - **Whether an engine is well** — `breaker`, local to this process.
//! - **Which engine a request goes to** — `rendezvous`: the client's cache key
//!   when it sent one (so a conversation stays on the engine holding its
//!   prefix cache), the least-loaded engine otherwise.
//! - **What a request may be** — `subset`: the node's accepted subset, checked
//!   before anything is sent.
//!
//! A request is tried on at most [`RouterConfig::request_attempts`] engines,
//! in rank order, moving on only when the engine gave no response (no
//! attested connection, or a connection that closed under the request), a
//! refusal before the body is read, or "overloaded". An engine's answer, an
//! error included, is never retried elsewhere.
//! The response comes back in the same shapes the Tinfoil backend produces
//! (`BackendResponse`, `BackendStreamEvent`), so billing settles it the same
//! way: usage → charge.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use futures_util::StreamExt;
use reqwest::header::{CONTENT_TYPE, HeaderMap};
use tinfoil_verifier::AllowedMeasurement;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::backend::{BackendMeta, BackendResponse, BackendStreamEvent, TeeType};
use crate::db::EnginePlacementRow;
use crate::engine_trust::protocol::{
    EngineToken, ValidatedRequest, engine_request_body, engine_request_headers,
};
use crate::engine_trust::{self, PinnedModel};
use crate::error::ServerError;
use crate::types::{ChatCompletionChunk, ChatCompletionResponse, ErrorResponse};

mod breaker;
pub mod placement;
mod rendezvous;
pub mod subset;

use breaker::Breaker;
pub use breaker::BreakerConfig;
use placement::PlacementSource;

/// The provider name an engine-served response carries in its privacy
/// metadata (`eidola-engine-tee`).
pub const PROVIDER: &str = "eidola-engine";

/// A pinned model as the router serves it.
pub struct EngineModel {
    pinned: PinnedModel,
    /// The config hash of every accepted deployment.
    deployments: Vec<String>,
    /// Every accepted deployment's pin: what each upstream's client attests
    /// against.
    pins: Vec<AllowedMeasurement>,
}

impl EngineModel {
    /// Every model this build pins, with its deployments.
    pub fn compiled() -> Result<Vec<Self>, String> {
        engine_trust::PINNED_MODELS
            .iter()
            .map(|pinned| {
                let accepted = engine_trust::accepted_deployments(pinned.id())?;
                if accepted.is_empty() {
                    return Err(format!("{} is pinned without a deployment", pinned.id()));
                }
                Ok(Self {
                    pinned: *pinned,
                    deployments: accepted.iter().map(|d| d.config_sha256.clone()).collect(),
                    pins: accepted.into_iter().map(|d| d.pin).collect(),
                })
            })
            .collect()
    }

    /// A model with explicit deployments, for tests of the router.
    #[cfg(test)]
    pub(crate) fn fixture(
        pinned: PinnedModel,
        deployments: Vec<(String, AllowedMeasurement)>,
    ) -> Self {
        Self {
            pinned,
            deployments: deployments.iter().map(|(d, _)| d.clone()).collect(),
            pins: deployments.into_iter().map(|(_, p)| p).collect(),
        }
    }
}

/// What a client for one upstream is built from.
pub struct ClientSpec {
    /// The upstream's model's pins.
    pub pins: Vec<AllowedMeasurement>,
    /// The upstream's base URL.
    pub base_url: String,
}

/// Builds the client for one upstream.
pub type ClientFactory = Arc<
    dyn Fn(ClientSpec) -> Pin<Box<dyn Future<Output = Result<reqwest::Client, String>> + Send>>
        + Send
        + Sync,
>;

/// The roots an attesting client verifies an engine against.
pub struct AttestationRoots {
    /// TLS roots for the engine's certificate (the bundled WebPKI roots in
    /// production).
    pub tls_roots: rustls::RootCertStore,
    /// SEV-SNP chain overrides; `None` in production. TDX quotes always
    /// verify to the built-in Intel root.
    pub trusted_ark_der: Option<Vec<u8>>,
    pub trusted_ask_der: Option<Vec<u8>>,
}

/// The factory production uses: `tinfoil_verifier::attesting_client` over the
/// upstream's model's pins, addressed to the upstream.
pub fn attesting_client_factory(roots: AttestationRoots) -> ClientFactory {
    let roots = Arc::new(roots);
    Arc::new(move |spec: ClientSpec| {
        let roots = roots.clone();
        Box::pin(async move {
            tinfoil_verifier::attesting_client(tinfoil_verifier::AttestingClientConfig {
                allowed_measurements: &spec.pins,
                inference_base_url: &spec.base_url,
                trusted_ark_der: roots.trusted_ark_der.as_deref(),
                trusted_ask_der: roots.trusted_ask_der.as_deref(),
                snp_min_tcb: None,
                snp_observer: None,
                attestation_observer: None,
                tls_roots: roots.tls_roots.clone(),
            })
            .await
            .map_err(|e| e.to_string())
        })
    })
}

/// The router's tuning.
#[derive(Debug, Clone, Copy)]
pub struct RouterConfig {
    pub breaker: BreakerConfig,
    /// How often placement is re-read.
    pub placement_refresh: Duration,
    /// How often upstreams are probed (an open one only once its backoff has
    /// passed).
    pub probe_interval: Duration,
    /// How long a probe may take.
    pub probe_timeout: Duration,
    /// How many engines one request may be tried on.
    pub request_attempts: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            breaker: BreakerConfig {
                failure_threshold: 3,
                initial_backoff: Duration::from_secs(5),
                max_backoff: Duration::from_secs(300),
            },
            placement_refresh: Duration::from_secs(30),
            probe_interval: Duration::from_secs(10),
            probe_timeout: Duration::from_secs(15),
            request_attempts: 3,
        }
    }
}

/// One engine a model's requests can go to.
pub struct Upstream {
    pub(crate) base_url: String,
    deployment: String,
    client: reqwest::Client,
    in_flight: AtomicUsize,
    breaker: std::sync::Mutex<Breaker>,
    probing: AtomicBool,
}

impl Upstream {
    fn new(placement: placement::Placement, client: reqwest::Client) -> Self {
        Self {
            base_url: placement.base_url,
            deployment: placement.deployment,
            client,
            in_flight: AtomicUsize::new(0),
            breaker: std::sync::Mutex::new(Breaker::new()),
            probing: AtomicBool::new(false),
        }
    }

    /// An upstream with a plain client, for tests of ranking.
    #[cfg(test)]
    pub(crate) fn fixture(base_url: &str) -> Self {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        Self::new(
            placement::Placement {
                deployment: String::new(),
                base_url: base_url.to_string(),
            },
            reqwest::Client::new(),
        )
    }

    /// Requests in flight.
    pub(crate) fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    /// Count a request in flight until the returned guard drops.
    pub(crate) fn begin(self: &Arc<Self>) -> InFlight {
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        InFlight(self.clone())
    }

    fn breaker(&self) -> std::sync::MutexGuard<'_, Breaker> {
        self.breaker.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn routable(&self) -> bool {
        self.breaker().routable()
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }
}

/// A request in flight on an upstream.
pub(crate) struct InFlight(Arc<Upstream>);

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Routes requests for Eidola-hosted models. Cheap to clone.
#[derive(Clone)]
pub struct EngineRouter {
    inner: Arc<Inner>,
}

struct Inner {
    models: BTreeMap<String, EngineModel>,
    upstreams: ArcSwap<BTreeMap<String, Vec<Arc<Upstream>>>>,
    token: EngineToken,
    factory: ClientFactory,
    config: RouterConfig,
    rotation: AtomicUsize,
    placement: tokio::sync::Mutex<()>,
}

/// Where a request goes and what it carries.
struct Plan<'a> {
    model: &'a EngineModel,
    /// The upstreams to try, in order.
    upstreams: Vec<Arc<Upstream>>,
    headers: HeaderMap,
    body: bytes::Bytes,
}

/// What one attempt on one upstream came to.
enum Attempt {
    /// A successful response, its body unread.
    Response(reqwest::Response),
    /// The engine did not run the request; try the next.
    Next,
    /// The request failed in a way another engine would not change.
    Fail(ServerError),
}

impl EngineRouter {
    /// A router over `models`, with no upstreams until placement is applied.
    pub fn new(
        models: Vec<EngineModel>,
        token: EngineToken,
        factory: ClientFactory,
        config: RouterConfig,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                models: models
                    .into_iter()
                    .map(|m| (m.pinned.id().to_string(), m))
                    .collect(),
                upstreams: ArcSwap::from_pointee(BTreeMap::new()),
                token,
                factory,
                config,
                rotation: AtomicUsize::new(0),
                placement: tokio::sync::Mutex::new(()),
            }),
        }
    }

    /// Replace the upstreams with what `rows` admits. An upstream already
    /// known (same model, base URL and deployment) is kept with its client,
    /// pooled connections and health; a new one gets a fresh client and takes
    /// no traffic until a probe passes.
    pub async fn apply_placement(&self, rows: &[EnginePlacementRow]) {
        let _serial = self.inner.placement.lock().await;
        let (usable, skipped) = placement::admissible(rows, |model| {
            self.inner
                .models
                .get(model)
                .map(|m| m.deployments.as_slice())
        });
        if !skipped.is_empty() {
            debug!(
                skipped = skipped.len(),
                "engine placement: rows not usable by this gateway"
            );
        }
        let current = self.inner.upstreams.load_full();
        let mut next: BTreeMap<String, Vec<Arc<Upstream>>> = BTreeMap::new();
        for (model_id, placements) in usable {
            let model = &self.inner.models[&model_id];
            let mut list = Vec::with_capacity(placements.len());
            for placement in placements {
                let known = current.get(&model_id).and_then(|known| {
                    known
                        .iter()
                        .find(|u| {
                            u.base_url == placement.base_url && u.deployment == placement.deployment
                        })
                        .cloned()
                });
                if let Some(upstream) = known {
                    list.push(upstream);
                    continue;
                }
                let spec = ClientSpec {
                    pins: model.pins.clone(),
                    base_url: placement.base_url.clone(),
                };
                match (self.inner.factory)(spec).await {
                    Ok(client) => {
                        info!(
                            model = %model_id,
                            upstream = %placement.base_url,
                            "engine placement: new upstream, probing before traffic"
                        );
                        list.push(Arc::new(Upstream::new(placement, client)));
                    }
                    Err(e) => warn!(
                        model = %model_id,
                        upstream = %placement.base_url,
                        "engine placement: no client for upstream: {e}"
                    ),
                }
            }
            next.insert(model_id, list);
        }
        self.inner.upstreams.store(Arc::new(next));
    }

    /// Read placement from `source` and apply it, then probe what is due. A
    /// failed read keeps the current placement.
    pub async fn refresh(&self, source: &dyn PlacementSource) {
        match source.load().await {
            Ok(rows) => self.apply_placement(&rows).await,
            Err(e) => warn!("engine placement: read failed, keeping the current placement: {e}"),
        }
        self.probe_due().await;
    }

    /// Probe every upstream whose probe is due, concurrently.
    pub async fn probe_due(&self) {
        let now = Instant::now();
        let snapshot = self.inner.upstreams.load_full();
        let mut probes = Vec::new();
        for (model_id, list) in snapshot.iter() {
            let model = &self.inner.models[model_id];
            for upstream in list {
                if upstream.breaker().probe_due(now)
                    && !upstream.probing.swap(true, Ordering::AcqRel)
                {
                    probes.push(self.probe_and_record(model, upstream.clone()));
                }
            }
        }
        futures_util::future::join_all(probes).await;
    }

    async fn probe_and_record(&self, model: &EngineModel, upstream: Arc<Upstream>) {
        let outcome = match tokio::time::timeout(
            self.inner.config.probe_timeout,
            self.probe(model, &upstream),
        )
        .await
        {
            Ok(outcome) => outcome,
            Err(_) => Err("the probe timed out".to_string()),
        };
        let was_routable = upstream.routable();
        upstream
            .breaker()
            .on_probe(outcome.is_ok(), Instant::now(), &self.inner.config.breaker);
        match outcome {
            Ok(()) if !was_routable => info!(
                model = %model.pinned.id(),
                upstream = %upstream.base_url,
                "engine upstream verified; taking traffic"
            ),
            Ok(()) => {}
            Err(reason) => warn!(
                model = %model.pinned.id(),
                upstream = %upstream.base_url,
                "engine upstream failed its probe; no traffic until a later probe passes: {reason}"
            ),
        }
        upstream.probing.store(false, Ordering::Release);
    }

    /// `/healthz` answers `ok`, and `/v1/engine/info` reports the pinned model
    /// and weights on verified read-only storage. Both over the upstream's
    /// attesting client, so the connection they use is attested too.
    async fn probe(&self, model: &EngineModel, upstream: &Upstream) -> Result<(), String> {
        let health = upstream
            .client
            .get(upstream.url("/healthz"))
            .send()
            .await
            .map_err(|e| format!("unreachable: {}", error_chain(&e)))?;
        let status = health.status();
        let body = health
            .text()
            .await
            .map_err(|e| format!("health body: {}", error_chain(&e)))?;
        // A development node answers `ok; weights-storage=dev-writable`.
        if !status.is_success() || body != "ok" {
            return Err(format!("unhealthy ({status})"));
        }

        #[derive(serde::Deserialize)]
        struct Info {
            model: String,
            weights_sha256: String,
            weights_storage: String,
        }
        let response = upstream
            .client
            .get(upstream.url("/v1/engine/info"))
            .headers(engine_request_headers(&self.inner.token, &model.pinned))
            .send()
            .await
            .map_err(|e| format!("unreachable: {}", error_chain(&e)))?;
        if !response.status().is_success() {
            return Err(format!("engine info refused ({})", response.status()));
        }
        let info: Info = response
            .json()
            .await
            .map_err(|_| "engine info unreadable".to_string())?;
        if info.model != model.pinned.id() {
            return Err("it serves another model".to_string());
        }
        if !info
            .weights_sha256
            .eq_ignore_ascii_case(model.pinned.weights().sha256)
        {
            return Err("it serves weights other than the pinned ones".to_string());
        }
        if info.weights_storage != "verified-readonly" {
            return Err("its weights are not on verified read-only storage".to_string());
        }
        Ok(())
    }

    /// Re-read placement and probe on their periods, until the process ends.
    pub fn spawn(&self, source: Arc<dyn PlacementSource>) -> tokio::task::JoinHandle<()> {
        let router = self.clone();
        tokio::spawn(async move {
            let config = router.inner.config;
            let mut placement = tokio::time::interval(config.placement_refresh);
            let mut probes = tokio::time::interval(config.probe_interval);
            placement.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            probes.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    _ = placement.tick() => router.refresh(source.as_ref()).await,
                    _ = probes.tick() => router.probe_due().await,
                }
            }
        })
    }

    /// The upstreams a request tries, in order, with the headers and body it
    /// sends. The subset check comes first, so a request a node would refuse
    /// is refused here whether or not an engine is available.
    fn plan(&self, request: &ValidatedRequest) -> Result<Plan<'_>, ServerError> {
        subset::check(request)?;
        let parsed = request.request();
        let model = self.inner.models.get(&parsed.model).ok_or_else(|| {
            ServerError::ServiceUnavailable("model is not served by an engine".to_string())
        })?;
        let snapshot = self.inner.upstreams.load();
        let routable: Vec<Arc<Upstream>> = snapshot
            .get(&parsed.model)
            .into_iter()
            .flatten()
            .filter(|u| u.routable())
            .cloned()
            .collect();
        let mut ranked = match &parsed.cache_key {
            Some(key) => rendezvous::rank_by_key(key, routable),
            None => rendezvous::rank_by_load(
                routable,
                self.inner.rotation.fetch_add(1, Ordering::Relaxed),
            ),
        };
        ranked.truncate(self.inner.config.request_attempts.max(1));
        if ranked.is_empty() {
            return Err(no_engine());
        }
        let headers = engine_request_headers(&self.inner.token, &model.pinned);
        // The forwarded copy is an ordinary buffer the HTTP stack owns, as on
        // the node; the canonical body it is copied from is scrubbed.
        let body = bytes::Bytes::copy_from_slice(&engine_request_body(request));
        Ok(Plan {
            model,
            upstreams: ranked,
            headers,
            body,
        })
    }

    /// Send `request` to one upstream and classify the outcome, recording it
    /// on the upstream's breaker.
    async fn attempt(
        &self,
        upstream: &Upstream,
        headers: &HeaderMap,
        body: &bytes::Bytes,
    ) -> Attempt {
        let config = &self.inner.config.breaker;
        let sent = upstream
            .client
            .post(upstream.url("/v1/chat/completions"))
            .headers(headers.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(body.clone())
            .send()
            .await;
        let response = match sent {
            Ok(response) => response,
            // No response at all: no attested connection could be made, or a
            // pooled one closed under the request (an engine restarting, say).
            // Nothing reached the client and only a response's usage settles
            // a charge, so whatever the engine did, another may run it; the
            // cost of a repeat is the engine's compute, never the client's.
            Err(e) => {
                upstream.breaker().on_failure(Instant::now(), config);
                warn!(
                    upstream = %upstream.base_url,
                    "engine upstream gave no response: {}",
                    error_chain(&e)
                );
                return Attempt::Next;
            }
        };
        let status = response.status();
        if status.is_success() {
            return Attempt::Response(response);
        }
        let body = response.bytes().await.unwrap_or_default();
        let (error_type, message) = match serde_json::from_slice::<ErrorResponse>(&body) {
            Ok(e) => (e.error.error_type, e.error.message),
            Err(_) => (
                "unknown".to_string(),
                "the engine refused the request".to_string(),
            ),
        };
        match status.as_u16() {
            // Refused before the body was read: the gateway's token, the
            // pinned weights, the model or the route are not what this
            // upstream serves. Misconfigured; open it and move on.
            401 | 404 | 412 | 428 => {
                upstream.breaker().trip(Instant::now(), config);
                warn!(
                    upstream = %upstream.base_url,
                    status = status.as_u16(),
                    "engine upstream refused the gateway; opened"
                );
                Attempt::Next
            }
            // At capacity is load, not ill health; a stopped engine is.
            503 => {
                if error_type != "overloaded" {
                    upstream.breaker().on_failure(Instant::now(), config);
                }
                Attempt::Next
            }
            s => {
                if s >= 500 {
                    upstream.breaker().on_failure(Instant::now(), config);
                }
                Attempt::Fail(ServerError::Backend {
                    status: s,
                    error_type,
                    message,
                })
            }
        }
    }

    /// A non-streaming request.
    #[tracing::instrument(skip_all, name = "engine.chat", err)]
    pub async fn send(&self, request: &ValidatedRequest) -> Result<BackendResponse, ServerError> {
        let Plan {
            upstreams,
            headers,
            body,
            ..
        } = self.plan(request)?;
        for upstream in upstreams {
            let _in_flight = upstream.begin();
            let response = match self.attempt(&upstream, &headers, &body).await {
                Attempt::Response(response) => response,
                Attempt::Next => continue,
                Attempt::Fail(e) => return Err(e),
            };
            let config = &self.inner.config.breaker;
            let bytes = match response.bytes().await {
                Ok(bytes) => bytes,
                Err(e) => {
                    upstream.breaker().on_failure(Instant::now(), config);
                    return Err(ServerError::Network(e.without_url().to_string()));
                }
            };
            let completion: ChatCompletionResponse = match serde_json::from_slice(&bytes) {
                Ok(completion) => completion,
                Err(e) => {
                    upstream.breaker().on_failure(Instant::now(), config);
                    return Err(ServerError::Parse(crate::error::parse_error_summary(&e)));
                }
            };
            upstream.breaker().on_success();
            let meta = BackendMeta {
                provider: PROVIDER.to_string(),
                chat_id: Some(completion.id.clone()),
                backend_model: completion.model.clone(),
                tee_type: Some(TeeType::EidolaEngine),
                usage: completion.usage.clone(),
            };
            return Ok(BackendResponse {
                response: completion,
                meta,
            });
        }
        Err(no_engine())
    }

    /// A streaming request: chunks as the engine sends them, then `Done` with
    /// the usage the engine reported.
    #[tracing::instrument(skip_all, name = "engine.chat_stream", err)]
    pub async fn send_stream(
        &self,
        request: &ValidatedRequest,
    ) -> Result<mpsc::Receiver<Result<BackendStreamEvent, ServerError>>, ServerError> {
        let Plan {
            model,
            upstreams,
            headers,
            body,
        } = self.plan(request)?;
        for upstream in upstreams {
            let in_flight = upstream.begin();
            let response = match self.attempt(&upstream, &headers, &body).await {
                Attempt::Response(response) => response,
                Attempt::Next => continue,
                Attempt::Fail(e) => return Err(e),
            };
            let (tx, rx) = mpsc::channel(32);
            tokio::spawn(relay_stream(
                response,
                upstream,
                in_flight,
                model.pinned.id().to_string(),
                self.inner.config.breaker,
                tx,
            ));
            return Ok(rx);
        }
        Err(no_engine())
    }
}

/// The refusal when no engine could take a request.
fn no_engine() -> ServerError {
    ServerError::ServiceUnavailable("no engine is available for this model".to_string())
}

/// An error and its sources, for an operator log line. Engine URLs and
/// attestation failures are operator data; nothing here is request content.
fn error_chain(e: &reqwest::Error) -> String {
    let mut text = e.to_string();
    let mut source = std::error::Error::source(e);
    while let Some(s) = source {
        text.push_str(": ");
        text.push_str(&s.to_string());
        source = s.source();
    }
    text
}

/// One server-sent event.
struct SseEvent {
    event: Option<String>,
    data: String,
}

/// Take the next complete event (terminated by a blank line) off `buffer`.
fn next_event(buffer: &mut Vec<u8>) -> Option<SseEvent> {
    loop {
        let end = buffer.windows(2).position(|w| w == b"\n\n")?;
        let block: Vec<u8> = buffer.drain(..end + 2).collect();
        let text = String::from_utf8_lossy(&block[..end]);
        let mut event = None;
        let mut data: Option<String> = None;
        for line in text.lines() {
            let line = line.strip_suffix('\r').unwrap_or(line);
            if let Some(value) = line.strip_prefix("event:") {
                event = Some(value.trim_start().to_string());
            } else if let Some(value) = line.strip_prefix("data:") {
                let value = value.strip_prefix(' ').unwrap_or(value);
                match &mut data {
                    Some(d) => {
                        d.push('\n');
                        d.push_str(value);
                    }
                    None => data = Some(value.to_string()),
                }
            }
        }
        // A block with no data (a comment, a keep-alive) is not an event.
        if let Some(data) = data {
            return Some(SseEvent { event, data });
        }
    }
}

/// Relay an engine's SSE stream as backend events. A stream that ends without
/// `[DONE]`, or with the node's `event: error` frame, is an error (the
/// handler then settles it as a failed stream); one that completes reports
/// the engine's usage in `Done`.
async fn relay_stream(
    response: reqwest::Response,
    upstream: Arc<Upstream>,
    _in_flight: InFlight,
    model: String,
    breaker: BreakerConfig,
    tx: mpsc::Sender<Result<BackendStreamEvent, ServerError>>,
) {
    let mut stream = std::pin::pin!(response.bytes_stream());
    let mut buffer: Vec<u8> = Vec::new();
    let mut chat_id: Option<String> = None;
    let mut backend_model = model;
    let mut usage = None;
    let fail = |e: ServerError| -> Result<BackendStreamEvent, ServerError> {
        upstream.breaker().on_failure(Instant::now(), &breaker);
        Err(e)
    };
    while let Some(read) = stream.next().await {
        let bytes = match read {
            Ok(bytes) => bytes,
            Err(e) => {
                let _ = tx
                    .send(fail(ServerError::Network(e.without_url().to_string())))
                    .await;
                return;
            }
        };
        // One stamp per socket read, before any send can wait on the client.
        let received_at = Instant::now();
        buffer.extend_from_slice(&bytes);
        while let Some(event) = next_event(&mut buffer) {
            if event.event.as_deref() == Some("error") {
                let (error_type, message) = match serde_json::from_str::<ErrorResponse>(&event.data)
                {
                    Ok(e) => (e.error.error_type, e.error.message),
                    Err(_) => ("unknown".to_string(), "the engine failed".to_string()),
                };
                let _ = tx
                    .send(fail(ServerError::Backend {
                        status: 502,
                        error_type,
                        message,
                    }))
                    .await;
                return;
            }
            if event.data == "[DONE]" {
                upstream.breaker().on_success();
                let meta = BackendMeta {
                    provider: PROVIDER.to_string(),
                    chat_id,
                    backend_model,
                    tee_type: Some(TeeType::EidolaEngine),
                    usage,
                };
                let _ = tx
                    .send(Ok(BackendStreamEvent::Done(meta, Instant::now())))
                    .await;
                return;
            }
            match serde_json::from_str::<ChatCompletionChunk>(&event.data) {
                Ok(chunk) => {
                    if chat_id.is_none() {
                        chat_id = Some(chunk.id.clone());
                    }
                    backend_model.clone_from(&chunk.model);
                    if chunk.usage.is_some() {
                        usage.clone_from(&chunk.usage);
                    }
                    if tx
                        .send(Ok(BackendStreamEvent::Chunk(chunk, received_at)))
                        .await
                        .is_err()
                    {
                        // The client went away; dropping the response closes
                        // the connection, which cancels the node's request.
                        return;
                    }
                }
                Err(e) => warn!(
                    "failed to parse an engine SSE chunk: {}",
                    crate::error::parse_error_summary(&e)
                ),
            }
        }
    }
    let _ = tx
        .send(fail(ServerError::Network(
            "the engine's stream ended before it completed".to_string(),
        )))
        .await;
}

#[cfg(test)]
mod tests;
