//! The proxy's **space-free turn**: resolve, open, pay, call, record, settle.
//!
//! ## Why this is a sibling of the chore runner rather than a rider on it
//!
//! [`crate::utility`] is the existing precedent for a raw completion with no
//! space behind it: it routes through the backend registry, takes the
//! zero-spend path on an engine-backed reference, and provisions and settles
//! an ACT on a remote one. Everything about *paying* is the same shape here.
//!
//! What differs is evidence, and it differs on purpose in both directions. A
//! chore writes **no request rows and records no attestations** — its own
//! module says so — because a chore has no action to hang a forensic trail
//! from and the harness made it on its own behalf. A proxied request is the
//! opposite on both counts: a person's tool asked for it, and the Record is
//! where that person goes to see what left this machine. A proxy that paid
//! like a turn and recorded like a chore would be the one configuration in
//! which Eidola spends money and leaves nothing behind.
//!
//! So the split is: **resolution is shared verbatim**
//! ([`crate::Inner::resolve_utility_target`] — pure, no engine start, no
//! network, so an unresolvable reference degrades before anything is started)
//! and **opening is this module's own**, adding the attestation observer, the
//! provider row, the connection row and the request row that the chore runner
//! deliberately skips.
//!
//! ## Two allowlists, one rule
//!
//! The proxy **constructs** the upstream request; it never forwards one. That
//! is stated twice, because a downstream tool controls two things:
//!
//! - **Headers** — [`UpstreamHeaders`]. An enumerated set, so a header a
//!   future tool invents is dropped rather than passed.
//! - **Body fields** — [`ProxyChatRequest::from_json`]. Also an enumerated
//!   set, and not merely for symmetry: the Eidola server deserializes
//!   `deny_unknown_fields`, so a body forwarded verbatim would turn one
//!   unrecognised key from a downstream SDK into a 400 for the whole request.
//!   Reading the fields we know keeps a request working instead.
//!
//! The outer shape of what goes upstream is
//! [`eidola_common::chat_completion_request_body`] — the same construction the
//! app's own turns use, so there is one definition of what an Eidola chat
//! request looks like rather than two.

use std::sync::{Arc, Mutex};

use serde_json::Value;
use uuid::Uuid;

use super::{LocalExposure, ProxySettings};
use crate::changes::Change;
use crate::error::AppError;
use crate::{
    EidolaResolved, Inner, ModelInfo, backends, db, estimate_charge_credits, fetch_models,
    find_event_boundary, flush_attestations, local_models, now_ms, parse_server_error_message,
    process_refund, recover_refund,
};

/// The completion ceiling a request that named none gets.
///
/// The same number the app's own turns fall back to when a backend declares no
/// context length — one rule, so a proxied request and an in-app one ask for
/// the same budget from the same model.
const DEFAULT_MAX_COMPLETION_TOKENS: u32 = 4096;

/// Whether the proxy puts a `traceparent` on the upstream request.
///
/// **The default is off, and it is a decision rather than an omission.** The
/// Eidola server treats an inbound `traceparent` whose sampled flag is set as a
/// request to record that request at per-request granularity, exporting its
/// spans instead of only aggregates. Generic client-side OpenTelemetry
/// instrumentation injects one on *every* outbound call, so a tool with OTel
/// switched on — pointed at this proxy — would opt every one of its requests
/// out of the aggregate-only guarantee without anyone deciding to, and would
/// carry its own trace ids across requests, which is a self-linking primitive
/// on an otherwise anonymous surface.
///
/// Tracing is therefore **the app's** decision about its own traffic, never a
/// header a downstream tool can set. This build has no such setting — the
/// control belongs with a Developer settings pane that does not exist yet —
/// so [`Inner::upstream_tracing`] answers [`TraceUpstream::Off`] and the
/// enabled arm is reached only by its own tests. *Removal trigger for that
/// note: a tracing setting landing, at which point exactly one call site
/// changes.*
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceUpstream {
    /// Send no trace context.
    Off,
    /// Send a `traceparent`, minting a **fresh** trace id.
    On,
}

/// Mint a W3C `traceparent` for one attempt.
///
/// **A fresh id per attempt, retries included.** Reusing the id of a failed
/// attempt is the obvious implementation and the wrong one: it links the two
/// requests together for whoever reads the traces, which is the linkage this
/// whole surface exists to avoid — and a retry is precisely the moment a
/// client is most likely to be identifiable by the pair.
fn mint_traceparent() -> String {
    use rand_core::RngCore;
    let mut trace_id = [0u8; 16];
    let mut span_id = [0u8; 8];
    rand_core::OsRng.fill_bytes(&mut trace_id);
    rand_core::OsRng.fill_bytes(&mut span_id);
    let hex = |bytes: &[u8]| {
        use std::fmt::Write;
        let mut out = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            let _ = write!(out, "{b:02x}");
        }
        out
    };
    // version 00, sampled flag set — an unsampled traceparent asks the server
    // for nothing, so sending one would be a header with no meaning.
    format!("00-{}-{}-01", hex(&trace_id), hex(&span_id))
}

/// The complete set of headers the proxy puts on an upstream request.
///
/// **This is an allowlist, and that is the point.** Denylisting the two
/// headers we know to be dangerous today would pass the third one a tool
/// invents tomorrow; enumerating what the flow requires means anything else is
/// dropped without anyone having to think of it first.
///
/// The members, and why each is here:
///
/// - **`Content-Type: application/json`** — the request has a JSON body.
/// - **`Accept: text/event-stream`** — streaming only, and the same header the
///   app's own streaming turns send.
/// - **`Authorization`** — the ACT this app spends, or an external backend's
///   own key. **Never the downstream's.** A downstream `Authorization`
///   authenticates the tool *to the proxy* and is consumed there; the two
///   credentials share a header name and nothing else.
/// - **`traceparent`** — only when [`TraceUpstream::On`], and then freshly
///   minted here. See [`TraceUpstream`].
///
/// What is deliberately absent: everything else. `User-Agent`, `X-Request-Id`,
/// `Cookie`, `tracestate`, a vendor's own `x-*` — a header the proxy did not
/// decide to send does not go.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpstreamHeaders {
    /// `Some` iff the route carries a credential.
    pub authorization: Option<String>,
    pub streaming: bool,
    pub trace: TraceUpstream,
}

impl UpstreamHeaders {
    /// The headers, in the order they are set. `traceparent` is minted here
    /// rather than carried on the struct, so each call — each *attempt* — gets
    /// its own.
    pub fn to_pairs(&self) -> Vec<(&'static str, String)> {
        let mut pairs = vec![("Content-Type", "application/json".to_string())];
        if self.streaming {
            pairs.push(("Accept", "text/event-stream".to_string()));
        }
        if let Some(auth) = &self.authorization {
            pairs.push(("Authorization", auth.clone()));
        }
        if self.trace == TraceUpstream::On {
            pairs.push(("traceparent", mint_traceparent()));
        }
        pairs
    }

    /// What the Record shows about this request's headers: every name, and
    /// every value **except** the credential, which is replaced by its scheme.
    ///
    /// The names are the evidence that the allowlist did its job — a reader
    /// can see for themselves that nothing of their tool's travelled. The
    /// value of `Authorization` is a spend proof or an API key and is the one
    /// thing that must not land in a durable local log.
    pub fn for_record(&self) -> String {
        let redacted: Vec<Value> = self
            .to_pairs()
            .into_iter()
            .map(|(name, value)| {
                let shown = if name == "Authorization" {
                    value
                        .split_once(' ')
                        .map(|(scheme, _)| format!("{scheme} <redacted>"))
                        .unwrap_or_else(|| "<redacted>".to_string())
                } else {
                    value
                };
                Value::Array(vec![Value::String(name.to_string()), Value::String(shown)])
            })
            .collect();
        Value::Array(redacted).to_string()
    }
}

/// A chat-completion request as it arrived from a downstream tool, read by
/// **allowlist**.
///
/// Every field here is one the Eidola server's own request type accepts. A key
/// the downstream sent that is not on this list is dropped — never forwarded,
/// and never a refusal either: a client that adds a field Eidola has no
/// opinion on should keep working.
#[derive(Clone, Debug, Default)]
pub struct ProxyChatRequest {
    pub model: String,
    pub messages: Vec<Value>,
    pub max_completion_tokens: Option<u32>,
    pub temperature: Option<Value>,
    pub top_p: Option<Value>,
    pub stop: Option<Value>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,
    pub stream: bool,
}

impl ProxyChatRequest {
    /// Read a downstream body. Refuses only what makes the request
    /// unanswerable: a missing or non-string `model`, and a missing or
    /// non-array `messages`.
    pub fn from_json(body: &Value) -> Result<Self, AppError> {
        let object = body.as_object().ok_or_else(|| AppError::Config {
            message: "the request body must be a JSON object".into(),
        })?;
        let model = object
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| AppError::Config {
                message: "`model` is required and must be a string".into(),
            })?
            .to_string();
        let messages = object
            .get("messages")
            .and_then(Value::as_array)
            .ok_or_else(|| AppError::Config {
                message: "`messages` is required and must be an array".into(),
            })?
            .clone();
        Ok(ProxyChatRequest {
            model,
            messages,
            max_completion_tokens: object
                .get("max_completion_tokens")
                // `max_tokens` is the older spelling half the ecosystem still
                // sends. Reading it is not a second construction — it lands in
                // the one field the outer shape carries.
                .or_else(|| object.get("max_tokens"))
                .and_then(Value::as_u64)
                .map(|n| u32::try_from(n).unwrap_or(u32::MAX)),
            temperature: object.get("temperature").cloned(),
            top_p: object.get("top_p").cloned(),
            stop: object.get("stop").cloned(),
            tools: object.get("tools").and_then(Value::as_array).cloned(),
            tool_choice: object.get("tool_choice").cloned(),
            stream: object
                .get("stream")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }

    /// The tool schemas as the pricing contract wants them — an empty slice
    /// when none were sent, which is what omits the field entirely.
    fn tool_schemas(&self) -> &[Value] {
        self.tools.as_deref().unwrap_or(&[])
    }
}

/// An opened, **recording** route: everything one HTTP call needs, plus the
/// forensic identity of where it is going.
struct ProxyRoute {
    client: reqwest::Client,
    base_url: String,
    /// The model id the backend's own API expects.
    wire_model: String,
    /// The canonical `<model>@<backend>` string the caller named.
    canonical: String,
    backend_id: String,
    /// `(prompt_rate, completion_rate, scale_factor)` — `Some` only for a
    /// route that bills.
    pricing: Option<(u128, u128, u128)>,
    /// An external backend's own bearer key, when it has one.
    external_auth: Option<String>,
    /// The connection row this request will be recorded against — present
    /// only where an attestation was verified, which is the whole point of
    /// recording it.
    connection_id: Option<String>,
    /// The largest completion the backend declares this model may be asked
    /// for; `None` when nothing was declared, which is never read as a zero.
    declared_max_output: Option<u64>,
    /// Held for the life of the call so the engine is not evicted underneath
    /// it.
    #[allow(dead_code)]
    engine_lease: Option<local_models::EngineLease>,
}

/// What one non-streaming proxied request produced.
pub struct ProxyChatResponse {
    pub status: u16,
    /// The upstream's own answer, with the credential artifact removed and
    /// the model named as the caller named it.
    pub body: Value,
}

/// One event on its way to a downstream tool.
pub enum ProxyStreamEvent {
    /// The upstream accepted the request. Sent exactly once, before any
    /// chunk, so the surface above can commit to a `200` and a body only when
    /// there is really going to be one.
    Open,
    /// One SSE event's bytes, terminator included — ready to write.
    Chunk(Vec<u8>),
}

impl Inner {
    /// Whether this build traces its own upstream requests.
    ///
    /// Always [`TraceUpstream::Off`]: there is no tracing setting in this
    /// build, and a proxied request must never be traced on a downstream
    /// tool's say-so (see [`TraceUpstream`]). *Removal trigger: the Developer
    /// settings pane's "trace my requests" control, which this then reads —
    /// one call site, covering the app's own requests and proxied ones alike.*
    fn upstream_tracing(&self) -> TraceUpstream {
        TraceUpstream::Off
    }

    /// The models the proxy offers, as OpenAI-shaped ids.
    ///
    /// Only exposed backends contribute, and an engine-backed backend
    /// contributes what the exposure setting says: every downloaded model, or
    /// only those with an engine already running.
    pub(crate) async fn proxy_models(&self) -> Result<Vec<ModelInfo>, AppError> {
        let settings = self.proxy_settings().await?;
        let running: std::collections::HashSet<String> =
            self.running_engines().into_iter().map(|e| e.id).collect();
        let mut out = Vec::new();
        for backend_id in &settings.backends {
            // One dead backend must not blank the whole listing — the same
            // rule the GUI's per-backend catalog slots take.
            let Ok(models) = self.backend_models(backend_id).await else {
                continue;
            };
            let engine_backed = db::get_backend(&self.db_conn().await?, backend_id)
                .await?
                .and_then(|row| backends::BackendKind::parse(&row.kind))
                .map(|kind| kind.is_engine_backed())
                .unwrap_or(false);
            for model in models {
                if engine_backed
                    && settings.local_exposure == LocalExposure::Loaded
                    && !running.contains(&model.id)
                {
                    continue;
                }
                out.push(model);
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out.dedup_by(|a, b| a.id == b.id);
        Ok(out)
    }

    /// Resolve a downstream model reference against the proxy's own
    /// permissions.
    ///
    /// **Exposure is checked before anything is opened.** A model on a backend
    /// the reader did not expose is refused by name, ahead of any engine start
    /// and ahead of any credential — the same shape (and the same reason) as
    /// the stale-selection refusal on the turn path: the only thing worse than
    /// this failure would be quietly answering from somewhere the reader did
    /// not agree to.
    async fn resolve_proxy_target(
        &self,
        settings: &ProxySettings,
        model_ref: &str,
    ) -> Result<crate::utility::UtilityTarget, AppError> {
        let mref = backends::parse_model_ref(model_ref);
        if !settings.backends.contains(&mref.backend_id) {
            return Err(AppError::ModelUnavailable {
                model: model_ref.to_string(),
            });
        }
        let conn = self.db_conn().await?;
        self.resolve_utility_target(&conn, model_ref, "proxy").await
    }

    /// Open the route: lease or start the engine, build the client, verify and
    /// **record** the attestation, read the catalog's pricing when the call
    /// bills.
    async fn open_proxy_route(
        &self,
        settings: &ProxySettings,
        target: &crate::utility::UtilityTarget,
    ) -> Result<ProxyRoute, AppError> {
        let backend = &target.backend;
        let now = now_ms();
        match target.kind {
            backends::BackendKind::Local | backends::BackendKind::LlamaCpp => {
                let leased = self.local.lease_engine(&backend.id, &target.model);
                let (engine_url, lease) = match leased {
                    Some((url, _ctx, lease)) => (url, lease),
                    None => {
                        // The exposure setting is a *permission to start an
                        // engine*, so it is enforced here rather than only in
                        // the listing: a tool that names a model the listing
                        // withheld must not be able to start it anyway.
                        if settings.local_exposure == LocalExposure::Loaded {
                            return Err(AppError::ModelUnavailable {
                                model: target.canonical.clone(),
                            });
                        }
                        if target.kind == backends::BackendKind::LlamaCpp && !backend.auto_start {
                            return Err(AppError::NotConfigured {
                                message: format!(
                                    "`{}` is not loaded and backend `{}` has auto-start disabled",
                                    target.canonical, backend.id
                                ),
                            });
                        }
                        self.load_local_model(&target.canonical).await?;
                        self.local
                            .lease_engine(&backend.id, &target.model)
                            .map(|(url, _ctx, lease)| (url, lease))
                            .ok_or_else(|| AppError::LocalModel {
                                message: format!(
                                    "`{}` was unloaded while starting",
                                    target.canonical
                                ),
                            })?
                    }
                };
                Ok(ProxyRoute {
                    client: self.plain_client()?,
                    base_url: engine_url,
                    wire_model: target.canonical.clone(),
                    canonical: target.canonical.clone(),
                    backend_id: backend.id.clone(),
                    pricing: None,
                    external_auth: None,
                    connection_id: None,
                    declared_max_output: None,
                    engine_lease: Some(lease),
                })
            }
            backends::BackendKind::OpenAi => {
                let base_url = backend
                    .base_url
                    .clone()
                    .ok_or_else(|| AppError::NotConfigured {
                        message: format!("backend `{}` has no base URL", backend.id),
                    })?;
                Ok(ProxyRoute {
                    client: self.plain_client()?,
                    base_url,
                    wire_model: target.model.clone(),
                    canonical: target.canonical.clone(),
                    backend_id: backend.id.clone(),
                    pricing: None,
                    external_auth: backend.api_key.as_ref().map(|k| format!("Bearer {k}")),
                    connection_id: None,
                    declared_max_output: None,
                    engine_lease: None,
                })
            }
            backends::BackendKind::Eidola => {
                let eidola = EidolaResolved::from_row(Some(backend))?;
                let db_conn = self.db_conn().await?;
                let provider_id =
                    db::ensure_provider(&db_conn, &backend.id, "inference", now).await?;
                // **The observer is what makes this a recording route.** A
                // chore's client is built without one, so the handshake is
                // verified and then forgotten; here it becomes an
                // `attestation` row and the `connection` row the request is
                // recorded against.
                let log: Arc<Mutex<Vec<tinfoil_verifier::VerifiedAttestation>>> =
                    Arc::new(Mutex::new(Vec::new()));
                let log_clone = log.clone();
                let observer: Option<tinfoil_verifier::AttestationObserver> = Some(Arc::new(
                    move |att: tinfoil_verifier::VerifiedAttestation| {
                        log_clone.lock().unwrap().push(att);
                    },
                ));
                let client = self.build_client(&eidola, observer).await?;
                let models = fetch_models(&client, &eidola.base_url).await?;
                let connection_id =
                    flush_attestations(&log, &db_conn, &provider_id, &eidola.base_url, now).await?;
                if connection_id.is_some() {
                    self.bus.emit(Change::Record);
                }
                let entry = models
                    .data
                    .iter()
                    .find(|m| m.id == target.model)
                    .ok_or_else(|| AppError::ModelUnavailable {
                        model: target.canonical.clone(),
                    })?;
                Ok(ProxyRoute {
                    client,
                    base_url: eidola.base_url.clone(),
                    wire_model: target.model.clone(),
                    canonical: target.canonical.clone(),
                    backend_id: backend.id.clone(),
                    pricing: Some((
                        entry.pricing.per_prompt_token.value as u128,
                        entry.pricing.per_completion_token.value as u128,
                        entry.pricing.per_prompt_token.scale_factor as u128,
                    )),
                    external_auth: None,
                    connection_id,
                    declared_max_output: entry.max_output_tokens,
                    engine_lease: None,
                })
            }
        }
    }

    /// The completion ceiling this request asks for.
    ///
    /// The downstream's value when it named one, clamped to what the backend
    /// declares the model will produce — a typo asking for four billion tokens
    /// otherwise computes a hold nothing in the wallet can cover, and fails as
    /// a funding error rather than as the arithmetic mistake it is. An
    /// undeclared ceiling is never read as a zero, so nothing is clamped where
    /// nothing was declared.
    fn proxy_completion_budget(
        request: &ProxyChatRequest,
        declared_max_output: Option<u64>,
    ) -> u32 {
        let asked = request
            .max_completion_tokens
            .unwrap_or(DEFAULT_MAX_COMPLETION_TOKENS);
        match declared_max_output {
            Some(declared) => asked.min(u32::try_from(declared).unwrap_or(u32::MAX)),
            None => asked,
        }
    }

    /// Build the upstream body: the shared outer shape, plus the sampling
    /// fields the downstream actually sent.
    ///
    /// `chat_completion_request_body` is the one construction of an Eidola
    /// chat request; the allowlisted extras are inserted over it rather than
    /// re-spelled beside it, so the two can never disagree about the shape
    /// they share.
    fn proxy_request_body(
        request: &ProxyChatRequest,
        wire_model: &str,
        max_completion_tokens: u32,
        stream: bool,
        bills: bool,
    ) -> Value {
        let mut body = eidola_common::chat_completion_request_body(
            wire_model,
            &request.messages,
            max_completion_tokens,
            request.tool_schemas(),
            stream,
            // The Eidola server forces `include_usage` upstream regardless
            // (accurate refunds depend on it); a local engine only reports
            // usage when asked. The same rule the app's own turns take.
            stream && !bills,
        );
        if let Some(object) = body.as_object_mut() {
            for (key, value) in [
                ("temperature", &request.temperature),
                ("top_p", &request.top_p),
                ("stop", &request.stop),
                ("tool_choice", &request.tool_choice),
            ] {
                if let Some(value) = value {
                    object.insert(key.to_string(), value.clone());
                }
            }
        }
        body
    }

    /// Record one proxied exchange. `action_id` is always `None` — there is no
    /// action, and that is the shape rather than an omission: the Record's
    /// request row is nullable on `action_id` precisely so an exchange that
    /// produced nothing persistable still lands.
    #[allow(clippy::too_many_arguments)]
    async fn record_proxy_request(
        &self,
        route: &ProxyRoute,
        headers: &UpstreamHeaders,
        request_body: &Value,
        response_status: Option<u16>,
        response_body: Vec<u8>,
        error: Option<String>,
        credential_nonce: Option<String>,
        request_at: i64,
        response_at: i64,
    ) {
        let Ok(conn) = self.db_conn().await else {
            return;
        };
        let entry = db::Request {
            id: Uuid::now_v7().to_string(),
            connection_id: route.connection_id.clone(),
            action_id: None,
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            request_headers: Some(headers.for_record()),
            request_body: Some(request_body.to_string().into_bytes()),
            response_status: response_status.map(i64::from),
            response_headers: None,
            response_body: Some(response_body),
            request_at,
            response_at: Some(response_at),
            duration_ms: Some(response_at - request_at),
            error,
            credential_nonce,
            created_at: now_ms(),
            backend_id: Some(route.backend_id.clone()),
        };
        // Best effort, and deliberately so: a request that already happened is
        // not undone by failing to write it down, and raising here would turn
        // a bookkeeping failure into a lost answer the caller already paid for.
        if db::insert_request(&conn, &entry).await.is_ok() {
            self.bus.emit(Change::Record);
        }
    }

    /// One non-streaming proxied completion.
    pub(crate) async fn proxy_chat(
        &self,
        request: ProxyChatRequest,
    ) -> Result<ProxyChatResponse, AppError> {
        let settings = self.proxy_settings().await?;
        let target = self.resolve_proxy_target(&settings, &request.model).await?;
        let route = self.open_proxy_route(&settings, &target).await?;
        let max_completion_tokens =
            Self::proxy_completion_budget(&request, route.declared_max_output);
        let body = Self::proxy_request_body(
            &request,
            &route.wire_model,
            max_completion_tokens,
            false,
            route.pricing.is_some(),
        );

        let cfg = self.load_config();
        let now = now_ms();
        let db_conn = self.db_conn().await?;
        let mut spend = None;
        let auth_value = match route.pricing {
            None => route.external_auth.clone(),
            Some(pricing) => {
                let charge = estimate_charge_credits(
                    &request.messages,
                    request.tool_schemas(),
                    max_completion_tokens,
                    pricing,
                );
                if charge == 0 {
                    return Err(AppError::Credential {
                        message: "computed charge is zero — model pricing may be missing".into(),
                    });
                }
                let (prep, auth) = self.acquire_spend(&cfg, &db_conn, charge, now).await?;
                spend = Some(prep);
                Some(auth)
            }
        };
        let nonce = spend.as_ref().map(|s| s.cred.nonce.clone());
        let headers = UpstreamHeaders {
            authorization: auth_value.clone(),
            streaming: false,
            trace: self.upstream_tracing(),
        };

        let request_at = now_ms();
        let mut outbound = route
            .client
            .post(format!("{}/v1/chat/completions", route.base_url))
            .json(&body);
        for (name, value) in headers.to_pairs() {
            // `json()` already set Content-Type; setting it again is
            // idempotent and keeps the allowlist the single enumeration.
            outbound = outbound.header(name, value);
        }

        let response = match outbound.send().await {
            Ok(response) => response,
            Err(e) => {
                let error = AppError::from_request(e);
                self.settle_proxy_refund(&db_conn, &spend, &auth_value, &route, None)
                    .await;
                self.record_proxy_request(
                    &route,
                    &headers,
                    &body,
                    None,
                    Vec::new(),
                    Some(error.to_string()),
                    nonce,
                    request_at,
                    now_ms(),
                )
                .await;
                return Err(error);
            }
        };
        let status = response.status();
        let text = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                let error = AppError::Network {
                    message: format!(
                        "failed to read the response: {}",
                        crate::error::request_error_text(e)
                    ),
                };
                self.settle_proxy_refund(&db_conn, &spend, &auth_value, &route, None)
                    .await;
                self.record_proxy_request(
                    &route,
                    &headers,
                    &body,
                    Some(status.as_u16()),
                    Vec::new(),
                    Some(error.to_string()),
                    nonce,
                    request_at,
                    now_ms(),
                )
                .await;
                return Err(error);
            }
        };
        let response_at = now_ms();
        let mut parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        self.settle_proxy_refund(&db_conn, &spend, &auth_value, &route, parsed.get("refund"))
            .await;
        self.record_proxy_request(
            &route,
            &headers,
            &body,
            Some(status.as_u16()),
            text.as_bytes().to_vec(),
            None,
            nonce,
            request_at,
            response_at,
        )
        .await;

        if !status.is_success() {
            return Err(AppError::Server {
                status: status.as_u16(),
                message: parse_server_error_message(&text),
            });
        }

        if let Some(object) = parsed.as_object_mut() {
            // **The credential artifact never reaches downstream.** Eidola's
            // answer carries a `refund` this app consumes to mint the
            // credential's successor; it is wallet material, not part of any
            // OpenAI response, and a tool that saw it could do nothing with it
            // but keep a copy.
            object.remove("refund");
            // The caller asked for a model by the name this app gave it; the
            // answer names it back. (Only an external OpenAI backend's wire id
            // differs from the canonical one — an engine's alias *is* the
            // canonical id, and an Eidola model is spelled bare either way.)
            object.insert("model".to_string(), Value::String(route.canonical.clone()));
        }
        Ok(ProxyChatResponse {
            status: status.as_u16(),
            body: parsed,
        })
    }

    /// One streaming proxied completion. Events reach `sender`; the returned
    /// `Result` is the terminal outcome.
    pub(crate) async fn proxy_chat_stream(
        &self,
        request: ProxyChatRequest,
        sender: tokio::sync::mpsc::UnboundedSender<ProxyStreamEvent>,
    ) -> Result<(), AppError> {
        use futures_util::StreamExt;

        let settings = self.proxy_settings().await?;
        let target = self.resolve_proxy_target(&settings, &request.model).await?;
        let route = self.open_proxy_route(&settings, &target).await?;
        let max_completion_tokens =
            Self::proxy_completion_budget(&request, route.declared_max_output);
        let body = Self::proxy_request_body(
            &request,
            &route.wire_model,
            max_completion_tokens,
            true,
            route.pricing.is_some(),
        );

        let cfg = self.load_config();
        let now = now_ms();
        let db_conn = self.db_conn().await?;
        let mut spend = None;
        let auth_value = match route.pricing {
            None => route.external_auth.clone(),
            Some(pricing) => {
                let charge = estimate_charge_credits(
                    &request.messages,
                    request.tool_schemas(),
                    max_completion_tokens,
                    pricing,
                );
                if charge == 0 {
                    return Err(AppError::Credential {
                        message: "computed charge is zero — model pricing may be missing".into(),
                    });
                }
                let (prep, auth) = self.acquire_spend(&cfg, &db_conn, charge, now).await?;
                spend = Some(prep);
                Some(auth)
            }
        };
        let nonce = spend.as_ref().map(|s| s.cred.nonce.clone());
        let headers = UpstreamHeaders {
            authorization: auth_value.clone(),
            streaming: true,
            trace: self.upstream_tracing(),
        };

        let request_at = now_ms();
        let mut outbound = route
            .client
            .post(format!("{}/v1/chat/completions", route.base_url))
            .json(&body);
        for (name, value) in headers.to_pairs() {
            outbound = outbound.header(name, value);
        }

        let response = match outbound.send().await {
            Ok(response) => response,
            Err(e) => {
                let error = AppError::from_request(e);
                self.settle_proxy_refund(&db_conn, &spend, &auth_value, &route, None)
                    .await;
                self.record_proxy_request(
                    &route,
                    &headers,
                    &body,
                    None,
                    Vec::new(),
                    Some(error.to_string()),
                    nonce,
                    request_at,
                    now_ms(),
                )
                .await;
                return Err(error);
            }
        };
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            self.settle_proxy_refund(&db_conn, &spend, &auth_value, &route, None)
                .await;
            self.record_proxy_request(
                &route,
                &headers,
                &body,
                Some(status.as_u16()),
                text.as_bytes().to_vec(),
                None,
                nonce,
                request_at,
                now_ms(),
            )
            .await;
            return Err(AppError::Server {
                status: status.as_u16(),
                message: parse_server_error_message(&text),
            });
        }

        // Only now is there going to be a `200` downstream.
        let _ = sender.send(ProxyStreamEvent::Open);

        let mut byte_stream = response.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut raw: Vec<u8> = Vec::new();
        let mut read_error: Option<AppError> = None;
        while let Some(chunk) = byte_stream.next().await {
            let bytes = match chunk {
                Ok(bytes) => bytes,
                Err(e) => {
                    read_error = Some(AppError::Network {
                        message: format!("stream read failed: {e}"),
                    });
                    break;
                }
            };
            raw.extend_from_slice(&bytes);
            buf.extend_from_slice(&bytes);
            while let Some(pos) = find_event_boundary(&buf) {
                let event: Vec<u8> = buf.drain(..pos).collect();
                let boundary_len = if buf.starts_with(b"\r\n\r\n") { 4 } else { 2 };
                let terminator: Vec<u8> = buf.drain(..boundary_len.min(buf.len())).collect();
                let mut out = forward_sse_event(&event);
                out.extend_from_slice(&terminator);
                if sender.send(ProxyStreamEvent::Chunk(out)).is_err() {
                    // Nobody is listening any more. The turn is upstream and
                    // paid for either way, so the pump stops and the settle
                    // below still runs — a caller loses its delivery, never
                    // this app's accounting.
                    break;
                }
            }
        }
        // Whatever is left is a partial event; forward it so a downstream
        // parser sees exactly the bytes the upstream sent.
        if !buf.is_empty() {
            let _ = sender.send(ProxyStreamEvent::Chunk(forward_sse_event(&buf)));
        }

        // SSE carries no inline refund — the token has no body to ride — so a
        // streaming spend always settles through the recovery endpoint, the
        // same as every streaming turn this app makes.
        self.settle_proxy_refund(&db_conn, &spend, &auth_value, &route, None)
            .await;
        self.record_proxy_request(
            &route,
            &headers,
            &body,
            Some(status.as_u16()),
            raw,
            read_error.as_ref().map(ToString::to_string),
            nonce,
            request_at,
            now_ms(),
        )
        .await;
        match read_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Settle a proxied call's credential: the inline refund when the response
    /// carried one, otherwise the recovery endpoint. A no-op for the
    /// zero-spend backends, and best-effort throughout — a wallet hiccup must
    /// not turn a delivered answer into a failure.
    async fn settle_proxy_refund(
        &self,
        db_conn: &turso::Connection,
        spend: &Option<crate::SpendPrep>,
        auth_value: &Option<String>,
        route: &ProxyRoute,
        inline: Option<&Value>,
    ) {
        let Some(spend) = spend else { return };
        let refund_obj = match inline {
            Some(object) => Some(object.clone()),
            None => match auth_value {
                Some(auth) => recover_refund(&route.client, &route.base_url, auth)
                    .await
                    .ok(),
                None => None,
            },
        };
        let Some(refund_obj) = refund_obj else {
            eprintln!("warning: a proxied call's credential refund could not be recovered");
            return;
        };
        let applied = process_refund(
            &refund_obj,
            &spend.params,
            &spend.spend_proof,
            &spend.pre_refund,
            &spend.public_key,
            db_conn,
            &spend.pre_cred_id,
            spend.cred.generation + 1,
            now_ms(),
        )
        .await;
        match applied {
            Ok(()) => self.bus.emit(Change::Wallet),
            Err(e) => eprintln!("warning: a proxied call's refund failed to apply: {e}"),
        }
    }
}

/// One SSE event on its way downstream.
///
/// **Byte-identical unless there is a credential in it.** The answer a model
/// gave is not this app's to reformat, so an event whose payloads carry
/// nothing of ours is passed through exactly as it arrived. The one thing that
/// may not travel is a `refund` — wallet material this app consumes — and
/// removing it is the only reason an event is ever rebuilt. (Today the Eidola
/// server puts no refund in a stream, which is *why* streaming spends settle
/// through the recovery endpoint; this is the outbound half of the same
/// allowlist discipline the headers take, so a server that later grew one
/// could not leak it through here by default.)
fn forward_sse_event(event: &[u8]) -> Vec<u8> {
    let Ok(text) = std::str::from_utf8(event) else {
        return event.to_vec();
    };
    let carries_refund = text.lines().any(|line| {
        line.trim_end_matches('\r')
            .strip_prefix("data:")
            .map(str::trim_start)
            .and_then(|payload| serde_json::from_str::<Value>(payload).ok())
            .is_some_and(|value| value.get("refund").is_some())
    });
    if !carries_refund {
        return event.to_vec();
    }
    let mut out = String::with_capacity(text.len());
    for (index, line) in text.split('\n').enumerate() {
        if index > 0 {
            out.push('\n');
        }
        let carriage = line.ends_with('\r');
        let bare = line.trim_end_matches('\r');
        match bare
            .strip_prefix("data:")
            .map(str::trim_start)
            .and_then(|payload| serde_json::from_str::<Value>(payload).ok())
        {
            Some(mut value) if value.get("refund").is_some() => {
                if let Some(object) = value.as_object_mut() {
                    object.remove("refund");
                }
                out.push_str("data: ");
                out.push_str(&value.to_string());
            }
            _ => out.push_str(bare),
        }
        if carriage {
            out.push('\r');
        }
    }
    out.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(auth: Option<&str>, streaming: bool, trace: TraceUpstream) -> UpstreamHeaders {
        UpstreamHeaders {
            authorization: auth.map(str::to_string),
            streaming,
            trace,
        }
    }

    #[test]
    fn the_upstream_header_set_is_exactly_the_allowlist() {
        let names = |h: &UpstreamHeaders| -> Vec<&'static str> {
            h.to_pairs().into_iter().map(|(name, _)| name).collect()
        };
        assert_eq!(
            names(&headers(None, false, TraceUpstream::Off)),
            vec!["Content-Type"],
            "a local route carries nothing but the body's own type"
        );
        assert_eq!(
            names(&headers(Some("Bearer x"), true, TraceUpstream::Off)),
            vec!["Content-Type", "Accept", "Authorization"],
        );
        assert_eq!(
            names(&headers(Some("Bearer x"), true, TraceUpstream::On)),
            vec!["Content-Type", "Accept", "Authorization", "traceparent"],
            "and nothing else is reachable — the set is enumerated, not filtered"
        );
    }

    #[test]
    fn tracing_is_off_and_mints_a_fresh_id_when_it_is_not() {
        let off = headers(None, false, TraceUpstream::Off);
        assert!(
            !off.to_pairs()
                .iter()
                .any(|(name, _)| *name == "traceparent"),
            "a downstream tool cannot ask this app to trace"
        );

        let on = headers(None, false, TraceUpstream::On);
        let first = on
            .to_pairs()
            .into_iter()
            .find(|(name, _)| *name == "traceparent")
            .expect("traceparent")
            .1;
        let second = on
            .to_pairs()
            .into_iter()
            .find(|(name, _)| *name == "traceparent")
            .expect("traceparent")
            .1;
        assert_ne!(
            first, second,
            "each attempt mints its own id — a retry that reused one would link the two"
        );
        assert!(
            first.starts_with("00-") && first.ends_with("-01"),
            "{first}"
        );
        assert_eq!(first.len(), 55, "version-traceid-spanid-flags");
    }

    #[test]
    fn the_record_shows_the_names_and_never_the_credential() {
        let recorded = headers(Some("Bearer super-secret"), true, TraceUpstream::Off).for_record();
        assert!(recorded.contains("Authorization"), "{recorded}");
        assert!(recorded.contains("Bearer <redacted>"), "{recorded}");
        assert!(
            !recorded.contains("super-secret"),
            "a spend proof must not land in a durable local log: {recorded}"
        );
        assert!(recorded.contains("text/event-stream"), "{recorded}");
    }

    #[test]
    fn a_downstream_body_is_read_by_allowlist() {
        let body = serde_json::json!({
            "model": "m@local",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5,
            "top_p": 0.9,
            "stop": ["END"],
            "tools": [{"type": "function"}],
            "tool_choice": "auto",
            "stream": true,
            "max_tokens": 128,
            "logit_bias": {"1": 2},
            "x-vendor-extension": "whatever a future SDK sends"
        });
        let request = ProxyChatRequest::from_json(&body).expect("read");
        assert_eq!(request.model, "m@local");
        assert!(request.stream);
        assert_eq!(request.max_completion_tokens, Some(128), "`max_tokens` too");
        assert_eq!(request.tool_schemas().len(), 1);

        // The unknown fields are simply not represented — there is nowhere for
        // them to be forwarded from.
        let upstream = Inner::proxy_request_body(&request, "m", 128, true, false);
        let object = upstream.as_object().expect("object");
        let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "max_completion_tokens",
                "messages",
                "model",
                "stop",
                "stream",
                "stream_options",
                "temperature",
                "tool_choice",
                "tools",
                "top_p",
            ],
            "no downstream key this app did not decide to send"
        );
        assert_eq!(
            object["model"], "m",
            "the backend's own id goes on the wire"
        );
    }

    #[test]
    fn a_body_without_the_two_required_fields_is_refused() {
        assert!(ProxyChatRequest::from_json(&serde_json::json!("not an object")).is_err());
        assert!(
            ProxyChatRequest::from_json(&serde_json::json!({"messages": []})).is_err(),
            "no model"
        );
        assert!(
            ProxyChatRequest::from_json(&serde_json::json!({"model": "m"})).is_err(),
            "no messages"
        );
    }

    #[test]
    fn an_sse_event_is_forwarded_byte_for_byte() {
        let event = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}";
        assert_eq!(
            forward_sse_event(event),
            event.to_vec(),
            "nothing this app does may alter what the model said"
        );
        assert_eq!(forward_sse_event(b"data: [DONE]"), b"data: [DONE]".to_vec());
        // A non-UTF-8 event still travels.
        assert_eq!(forward_sse_event(&[0xff, 0xfe]), vec![0xff, 0xfe]);
    }

    #[test]
    fn an_sse_event_carrying_a_refund_loses_only_the_refund() {
        let out = forward_sse_event(b"data: {\"id\":\"x\",\"refund\":\"credential-material\"}");
        let text = String::from_utf8(out).expect("utf-8");
        assert!(!text.contains("credential-material"), "{text}");
        assert!(text.contains("\"id\":\"x\""), "{text}");
        assert!(text.starts_with("data: "), "{text}");
    }

    #[test]
    fn the_completion_budget_honours_the_ask_and_the_declaration() {
        let asking = |n: Option<u32>| ProxyChatRequest {
            model: "m".into(),
            messages: vec![],
            max_completion_tokens: n,
            ..Default::default()
        };
        assert_eq!(
            Inner::proxy_completion_budget(&asking(None), None),
            DEFAULT_MAX_COMPLETION_TOKENS,
            "a request that named none gets the app's own fallback"
        );
        assert_eq!(
            Inner::proxy_completion_budget(&asking(Some(100)), Some(500)),
            100
        );
        assert_eq!(
            Inner::proxy_completion_budget(&asking(Some(u32::MAX)), Some(500)),
            500,
            "a typo cannot ask for a hold nothing could cover"
        );
        assert_eq!(
            Inner::proxy_completion_budget(&asking(Some(9_999)), None),
            9_999,
            "an undeclared ceiling is not a zero"
        );
    }
}
