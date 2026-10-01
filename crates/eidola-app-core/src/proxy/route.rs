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

use super::LocalExposure;
use crate::changes::Change;
use crate::error::AppError;
use crate::peer_read::MAX_RESPONSE_BYTES;
use crate::recorded::{RecordedBody, recorded_answer, recorded_cut_answer, recorded_request};
use crate::{
    ChargePricing, EidolaResolved, Inner, ModelInfo, backends, db, estimate_charge_credits,
    fetch_models, find_event_boundary, flush_attestations, is_event_stream, local_models, now_ms,
    parse_server_error_message, process_refund, recover_refund,
};

/// The completion ceiling a request that named none gets.
///
/// The same number the app's own turns fall back to when a backend declares no
/// context length — one rule, so a proxied request and an in-app one ask for
/// the same budget from the same model.
const DEFAULT_MAX_COMPLETION_TOKENS: u32 = 4096;

/// How long one backend has to answer `/v1/models` before it is treated as
/// unavailable for this listing.
///
/// A catalog read is a small request a healthy backend answers in
/// milliseconds; a local engine's is loopback. Ten seconds is generous enough
/// that a slow-but-working backend still contributes and short enough that a
/// wedged one costs a listing rather than the endpoint. It is deliberately
/// unrelated to a completion's own budget: a completion may legitimately take
/// minutes, and a bound covering both would be no bound at all.
const MODEL_LIST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// The per-backend catalog deadline in force, in milliseconds —
/// [`MODEL_LIST_TIMEOUT`] unless a test has shortened it. The seam moves the
/// number, never the mechanism, and it is compiled only for tests: see
/// `http::set_header_read_timeout_for_test` for why a
/// `#[doc(hidden)] pub fn` over a process-global atomic is not a test-only
/// seam at all.
#[cfg(feature = "test-support")]
static MODEL_LIST_TIMEOUT_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn model_list_timeout() -> std::time::Duration {
    #[cfg(feature = "test-support")]
    {
        let ms = MODEL_LIST_TIMEOUT_MS.load(std::sync::atomic::Ordering::Relaxed);
        if ms > 0 {
            return std::time::Duration::from_millis(ms);
        }
    }
    MODEL_LIST_TIMEOUT
}

/// Test-only: shorten the per-backend catalog deadline. `0` restores it.
#[cfg(feature = "test-support")]
#[doc(hidden)]
pub fn set_model_list_timeout_for_test(millis: u64) {
    MODEL_LIST_TIMEOUT_MS.store(millis, std::sync::atomic::Ordering::Relaxed);
}

/// One blocking upstream answer, read under [`MAX_RESPONSE_BYTES`].
type CappedBody = crate::peer_read::BoundedBody;

impl RecordedBody {
    /// The bytes to record for a proxied stream, with the cap's note when they
    /// are not all of them and the note for how the stream ended.
    ///
    /// **Neither transport has an unqualified seal to reach for**: an ending
    /// is not optional information about a body — see
    /// [`RecordedBody::seal_blocking`] for the blocking twin.
    ///
    /// `read_to_end` is whether the upstream read ran to the upstream's own
    /// end — false where the caller's departure, a transport failure or an
    /// oversized event stopped it — because only then is `received` the
    /// response's size rather than how far this app got.
    fn seal_stream(self, delivery: StreamDelivery, read_to_end: bool) -> Vec<u8> {
        let mut out = self.seal_response(read_to_end);
        if let Some(note) = delivery.note() {
            out.extend_from_slice(note.as_bytes());
        }
        out
    }
}

/// Read a blocking answer, bounded **as the bytes arrive**.
///
/// The reader is [`crate::peer_read::read_bounded`] — the crate's one rule for
/// reading from a peer — and what differs here is only the **ending**: an
/// oversized API answer is refused, while a proxied completion keeps what it
/// read, records it as the truncation it is, and answers the caller a gateway
/// failure, because the Record is evidence and a body that stopped at this
/// app's ceiling is a fact about the exchange. The ceiling is
/// [`MAX_RESPONSE_BYTES`] rather than the retention cap: what the Record keeps
/// and what this app may hold to answer one caller are different numbers, and
/// only the second bounds the process.
async fn read_capped_body(
    response: reqwest::Response,
) -> Result<CappedBody, crate::peer_read::ReadFailure> {
    crate::peer_read::read_bounded(response, MAX_RESPONSE_BYTES).await
}

/// How a proxied stream ended, as the Record has to state it.
///
/// **The cap was the first face of "a partial must never claim to be whole";
/// this is the second.** A stream has three endings and two of them leave the
/// row saying something untrue if nothing names them: an upstream read *ended
/// on purpose* looks byte-for-byte like an upstream that finished, and a
/// delivery that stopped short looks like one that did not. Neither is visible
/// in `received` versus `kept`, which is why the cap's own marker cannot cover
/// it.
///
/// The note deliberately states **facts, not a ratio**. What reaches the caller
/// is *forwarded* bytes — an event carrying a refund is rebuilt on the way out
/// — so "M of N bytes delivered" would mix two counts that are not the same
/// unit. What is true and useful is which of the three endings happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamDelivery {
    /// The upstream closed on its own and everything it sent went downstream.
    Complete,
    /// The caller went away; the upstream was read to its end anyway, because
    /// a hold had to settle from the refund in its tail. What is recorded is
    /// the whole upstream answer — and is **not** what was delivered.
    CallerGoneReadOn,
    /// The caller went away and the upstream read ended with it. What is
    /// recorded stops where this app stopped reading, which is not where the
    /// upstream had stopped talking.
    CallerGoneReadEnded,
}

/// Which ending a stream had, from the two facts the loop records.
///
/// A pure decision so the Record's claim can be pinned without staging a
/// caller that vanishes mid-stream — an interleaving the in-memory transport
/// cannot schedule deterministically.
fn stream_delivery(downstream_gone: bool, read_ended_early: bool) -> StreamDelivery {
    match (downstream_gone, read_ended_early) {
        (false, _) => StreamDelivery::Complete,
        (true, false) => StreamDelivery::CallerGoneReadOn,
        (true, true) => StreamDelivery::CallerGoneReadEnded,
    }
}

impl StreamDelivery {
    fn note(self) -> Option<&'static str> {
        match self {
            StreamDelivery::Complete => None,
            StreamDelivery::CallerGoneReadOn => Some(
                "\n\n[eidola: the caller disconnected before the end. The upstream response above \
                 was read in full and is not what was delivered.]\n",
            ),
            // No cause is claimed for the read's end: a departure ends it on
            // a route with nothing to settle, and a transport failure or an
            // oversized event can end it while a drain was running — each
            // with the same consequence, which is all this note states.
            StreamDelivery::CallerGoneReadEnded => Some(
                "\n\n[eidola: the caller disconnected, and the upstream read did not run to its \
                 end. The response above stops where this app stopped reading, not where the \
                 upstream stopped sending.]\n",
            ),
        }
    }
}

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
/// - **`Accept`** — `text/event-stream` on a streaming request (the same header
///   the app's own streaming turns send), `application/json` otherwise. Stated
///   on both because reqwest inserts an `Accept: */*` of its own where the
///   request carries none: an unstated header is not an absent one.
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
///
/// **Including the ones nobody wrote here.** An enumeration is only true if
/// nothing underneath it adds a header of its own, and `plain_http_client`'s
/// builder installs `User-Agent: eidola-app-core/<version>` — so a proxied
/// route is built from [`local_models::proxy_http_client`] instead, which sets
/// none. What is left on the wire besides this list is HTTP's own framing
/// (`Host`, `Content-Length`): the protocol, not a fact about this app.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpstreamHeaders {
    /// `Some` iff the route carries a credential.
    pub authorization: Option<String>,
    pub streaming: bool,
    pub trace: TraceUpstream,
}

/// The header set actually put on **one attempt's** wire.
///
/// **Minted once, so the Record shows what travelled.** `traceparent` is a new
/// id per attempt, which is right — and it made the description
/// ([`UpstreamHeaders`]) and the thing itself two different values: the request
/// builder minted one set and the Record row minted a second, so the recorded
/// trace id had never been on any wire and could correlate with nothing. The
/// attempt's headers are therefore materialised into this, and both the request
/// and the row read that one value; `for_record` lives here rather than on the
/// description because redacting *a set* is what the Record actually needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SentHeaders(Vec<(&'static str, String)>);

impl SentHeaders {
    /// The pairs, in the order they are set.
    pub fn pairs(&self) -> &[(&'static str, String)] {
        &self.0
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
            .0
            .iter()
            .map(|(name, value)| {
                let shown = if *name == "Authorization" {
                    value
                        .split_once(' ')
                        .map(|(scheme, _)| format!("{scheme} <redacted>"))
                        .unwrap_or_else(|| "<redacted>".to_string())
                } else {
                    value.clone()
                };
                Value::Array(vec![Value::String(name.to_string()), Value::String(shown)])
            })
            .collect();
        Value::Array(redacted).to_string()
    }
}

impl UpstreamHeaders {
    /// Mint this attempt's header set. Call **once per attempt** — that is the
    /// whole point of [`SentHeaders`].
    pub fn materialize(&self) -> SentHeaders {
        SentHeaders(self.to_pairs())
    }

    /// The headers, in the order they are set. `traceparent` is minted here
    /// rather than carried on the struct, so each call — each *attempt* — gets
    /// its own; [`UpstreamHeaders::materialize`] is the one caller, so "each
    /// call" and "each attempt" cannot come apart.
    fn to_pairs(&self) -> Vec<(&'static str, String)> {
        let mut pairs = vec![("Content-Type", "application/json".to_string())];
        // **Always named, because otherwise something else names it.** reqwest
        // inserts `Accept: */*` when the request carries none, so leaving the
        // blocking transport's unstated did not mean "no `Accept`" — it meant
        // one this app did not decide, outside the enumeration the Record shows.
        // Saying what each transport actually wants is both truer and shorter.
        pairs.push((
            "Accept",
            if self.streaming {
                "text/event-stream".to_string()
            } else {
                "application/json".to_string()
            },
        ));
        if let Some(auth) = &self.authorization {
            pairs.push(("Authorization", auth.clone()));
        }
        if self.trace == TraceUpstream::On {
            pairs.push(("traceparent", mint_traceparent()));
        }
        pairs
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
    /// Whether the caller asked for a usage chunk (`stream_options.
    /// include_usage`). Read as a field of its own rather than forwarded as an
    /// object, because the outer shape expresses exactly this one option and
    /// the allowlist discipline says a key nobody decided to send is not sent.
    pub include_usage: bool,
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
            // `max_tokens` is the older spelling half the ecosystem still sends.
            // Reading it is not a second construction — it lands in the one
            // field the outer shape carries.
            //
            // **Each spelling is converted before the fallback is taken**, so a
            // field that is present but not a number — `null` above all, which
            // SDKs emit for "unset" — reads as absent. Falling back on the *raw*
            // value instead let `"max_completion_tokens": null` win the `or`,
            // turn into `None`, and discard the caller's own `max_tokens` cap for
            // the 4096 default: more output produced, and billed, than asked for.
            max_completion_tokens: object
                .get("max_completion_tokens")
                .and_then(Value::as_u64)
                .or_else(|| object.get("max_tokens").and_then(Value::as_u64))
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
            include_usage: object
                .get("stream_options")
                .and_then(|options| options.get("include_usage"))
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
    pricing: Option<ChargePricing>,
    /// An external backend's own bearer key, when it has one.
    external_auth: Option<String>,
    /// The connection row this request will be recorded against — present
    /// only where an attestation was verified, which is the whole point of
    /// recording it. **Not final at open**: the completion can ride a *new*
    /// TLS connection (the server may close the listing one), and that
    /// handshake's attestation is flushed after the send — see
    /// [`ProxyRoute::flush_new_attestations`].
    connection_id: Option<String>,
    /// The observer's capture buffer, and what a flush needs to turn it into
    /// rows. `None` for a route that verifies no enclave (engine-backed and
    /// external backends), where there is never anything to flush.
    attestations: Option<AttestationSink>,
    /// The largest completion the backend declares this model may be asked
    /// for; `None` when nothing was declared, which is never read as a zero.
    declared_max_output: Option<u64>,
    /// Held for the life of the call so the engine is not evicted underneath
    /// it.
    #[allow(dead_code)]
    engine_lease: Option<local_models::EngineLease>,
}

/// A completion that has passed its preflight: a route, the body it will send,
/// and the hold it took if the route bills.
struct ProxyPreflight {
    route: ProxyRoute,
    body: Value,
    db_conn: turso::Connection,
    spend: Option<crate::SpendPrep>,
    auth_value: Option<String>,
}

/// Everything a post-send attestation flush needs.
struct AttestationSink {
    log: Arc<Mutex<Vec<tinfoil_verifier::VerifiedAttestation>>>,
    provider_id: String,
    base_url: String,
}

impl ProxyRoute {
    /// Persist any attestation captured since the last flush, adopting the
    /// connection it opened.
    ///
    /// **The turn path's `flush_new_attestations`, and it is here for the same
    /// reason.** A route opens by fetching the catalog, which verifies a
    /// handshake and writes its connection row — but the completion that
    /// follows may travel on a *different* TLS connection, because the server
    /// is free to close the one the listing used. That second handshake's
    /// attestation would otherwise sit in the observer and never be persisted,
    /// while the Record named the listing's connection as the one that carried
    /// the prompt. Both are wrong in the same way: the Record has to identify
    /// the connection the request actually went down.
    ///
    /// Returns whether a row was written, so the caller can announce it.
    async fn flush_new_attestations(&mut self, db_conn: &turso::Connection) -> bool {
        let Some(sink) = self.attestations.as_ref() else {
            return false;
        };
        match flush_attestations(
            &sink.log,
            db_conn,
            &sink.provider_id,
            &sink.base_url,
            now_ms(),
        )
        .await
        {
            Ok(Some(connection_id)) => {
                self.connection_id = Some(connection_id);
                true
            }
            // Nothing new to flush, or the write failed. A failed flush is
            // best-effort like every other Record write on this path: it costs
            // the connection row, never the answer the caller paid for.
            _ => false,
        }
    }
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
        // **The registry is authoritative for engine membership; the scan only
        // decorates.** The status menu learned this rule against a teardown;
        // here it is a listing, and the disagreement it prevents is worse,
        // because the *route* already obeys it: `open_proxy_route` leases from
        // this same registry and never consults the filesystem, so an engine
        // whose `.gguf` was renamed or deleted — or whose directory stopped
        // being readable — goes on serving requests perfectly well while a scan
        // -derived listing hides it. That is the one thing this surface must
        // not do: `/v1/models` is a capability statement, and a model the proxy
        // will serve has to be in it.
        //
        // **And it lists exactly what the lease would take** — ready, not merely
        // present (`reserve_engine` inserts the entry before the subprocess is
        // up), and of the incarnation the row below was authorized as
        // (`LocalRuntime::leasable_engines`, the lease's own predicate). A
        // snapshot taken by backend id alone, before the authorizing read,
        // published a retired incarnation's engine across a remove-and-re-add:
        // a model in the listing that every request for it was refused.
        // The registry rows first — local reads, and what decides how each
        // backend is treated below.
        //
        // **Each one is the exposed row, read with its permission, and it is the
        // row that gets scanned.** The settings snapshot names ids; reading a
        // row by id and then scanning by id again let a remove-and-re-add
        // between the two send the catalog request to the replacement's URL
        // with the replacement's key — a destination the reader never exposed —
        // and hand its answer to the caller. `db::exposed_backend` is the
        // resolve chapter's read, and `backend_models_for_row` scans exactly
        // what it returned.
        let mut plans = Vec::new();
        let conn = self.db_conn().await?;
        for backend_id in &settings.backends {
            // Read **before** the row, as the resolve chapter reads it: the
            // epoch must be no newer than the configuration it vouches for.
            let epoch = self.local.backend_epoch(backend_id);
            let Some(row) = db::exposed_backend(&conn, backend_id).await? else {
                continue;
            };
            let engine_backed = backends::BackendKind::parse(&row.kind)
                .map(|kind| kind.is_engine_backed())
                .unwrap_or(false);
            let loaded_only = offers_running_engines_only(settings.local_exposure, row.auto_start);
            plans.push((row, engine_backed, loaded_only, epoch));
        }
        // The gap between the authorizing read and the scan it permits.
        #[cfg(feature = "test-support")]
        crate::subspace_driver::pause_in_window(&self.proxy_authorized_window).await;

        // **Every catalog is asked at once, and each one is asked with a
        // deadline.** One dead backend must not blank the whole listing — the
        // rule the GUI's per-backend catalog slots take — but `.ok()` only
        // isolates a future that *resolves*, and `plain_http_client` sets no
        // request timeout: a backend that accepts the connection and then says
        // nothing left this `await` outstanding for ever, so `/v1/models` never
        // answered at all and every healthy backend's models went with it.
        // Awaiting them sequentially also made the endpoint's latency the
        // *sum* of every backend's, which is the same defect measured in
        // seconds rather than in forever.
        //
        // [`MODEL_LIST_TIMEOUT`] is the per-backend bound; a timeout reads as
        // an unavailable backend, which is exactly what it is.
        let scans = futures_util::future::join_all(plans.iter().map(|(row, _, _, _)| async {
            tokio::time::timeout(model_list_timeout(), self.backend_models_for_row(row))
                .await
                .ok()
                .and_then(|r| r.ok())
        }))
        .await;

        let mut out = Vec::new();
        for ((row, engine_backed, loaded_only, epoch), scanned) in plans.iter().zip(scans) {
            let (backend_id, engine_backed, loaded_only) = (&row.id, *engine_backed, *loaded_only);
            if engine_backed && loaded_only {
                // The scan is the *decoration* here, not the membership: it
                // supplies a context length and capabilities where it has
                // them, and where it does not the engine still answers for
                // itself. A scan that failed outright therefore blanks nothing.
                for (slug, context_tokens) in self.local.leasable_engines(backend_id, *epoch) {
                    let id = local_models::engine_model_id(backend_id, &slug);
                    let decorated = scanned
                        .as_ref()
                        .and_then(|models| models.iter().find(|m| m.id == id))
                        .cloned();
                    out.push(decorated.unwrap_or_else(|| engine_model_info(id, context_tokens)));
                }
                continue;
            }
            let Some(models) = scanned else {
                continue;
            };
            out.extend(models);
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
    ///
    /// **And it is checked against the row the request will be sent to, in the
    /// same read.** A settings snapshot names ids, and an id is not an
    /// incarnation: `remove_backend` + `insert_backend` puts a different base
    /// URL and a different key behind the same name, and the removal is what
    /// drops the exposure. Authorizing from the snapshot and then resolving the
    /// live row is therefore two questions about two different things, with a
    /// request's own network and engine latency in between — so the permission
    /// and the row come back from one statement ([`db::exposed_backend`]) and
    /// this reads neither `settings.backends` nor anything else taken earlier.
    /// It is the decide-at-the-write rule read from the other side: the read
    /// that authorizes is the read that resolves.
    ///
    /// A backend that is not exposed, one that was removed, and one that was
    /// disabled all answer the same refusal, deliberately. They are one fact to
    /// the caller — this proxy will not serve that model — and a key holder is
    /// owed no way to tell a backend that exists from one that does not.
    async fn resolve_proxy_target(
        &self,
        model_ref: &str,
    ) -> Result<(crate::utility::UtilityTarget, local_models::BackendEpoch), AppError> {
        // The gap this stages is the one the doc above is about: a settings
        // snapshot is already in hand, and the read below is what decides.
        #[cfg(feature = "test-support")]
        crate::subspace_driver::pause_in_window(&self.proxy_resolve_window).await;
        let mref = backends::parse_model_ref(model_ref);
        // Read **before** the row, the ordering `BackendEpoch` requires: the
        // epoch has to be no newer than the configuration it vouches for, or a
        // retirement between the two would be vouched for by the value read
        // after it. This is the third chapter of the exposure story — the
        // incarnation the load will act on — and it travels with the row it was
        // read beside.
        let epoch = self.local.backend_epoch(&mref.backend_id);
        let conn = self.db_conn().await?;
        let Some(backend) = db::exposed_backend(&conn, &mref.backend_id).await? else {
            return Err(AppError::ModelUnavailable {
                model: model_ref.to_string(),
            });
        };
        Ok((
            crate::utility::utility_target_for(backend, mref, "proxy")?,
            epoch,
        ))
    }

    /// Open the route: lease or start the engine, build the client, verify and
    /// **record** the attestation, read the catalog's pricing when the call
    /// bills.
    async fn open_proxy_route(
        &self,
        target: &crate::utility::UtilityTarget,
        epoch: local_models::BackendEpoch,
    ) -> Result<ProxyRoute, AppError> {
        let backend = &target.backend;
        let now = now_ms();
        match target.kind {
            backends::BackendKind::Local | backends::BackendKind::LlamaCpp => {
                // The gap this stages is the one the load's doc is about: an
                // authorized row is already in hand, and the id behind it can
                // be replaced before anything acts on it.
                #[cfg(feature = "test-support")]
                crate::subspace_driver::pause_in_window(&self.proxy_authorized_window).await;
                // **Leased only from the incarnation this request was
                // authorized against** (`lease_authorized_engine`). A ready
                // engine under this slug is not evidence it is the exposed
                // backend's: a remove-and-re-add that loads the same slug
                // leaves the replacement's engine under this key.
                let leased = self
                    .local
                    .lease_authorized_engine(&backend.id, &target.model, epoch);
                let (engine_url, lease) = match leased {
                    Some((url, _ctx, lease)) => (url, lease),
                    None => {
                        // The exposure setting is a *permission to start an
                        // engine*, so it is enforced here rather than only in
                        // the listing: a tool that names a model the listing
                        // withheld must not be able to start it anyway.
                        //
                        // **And it is read here, at the branch that acts on
                        // it.** A `ProxySettings` snapshot taken when the
                        // request arrived says what the permission was a whole
                        // resolve-and-catalog ago; a reader who takes the
                        // permission back in that window would still have a
                        // subprocess started on their machine, which is the one
                        // thing the setting exists to prevent. Same rule as the
                        // backend's own exposure (`db::exposed_backend`): the
                        // read that authorizes is the read that decides.
                        if self.proxy_local_exposure().await? == LocalExposure::Loaded {
                            return Err(AppError::ModelUnavailable {
                                model: target.canonical.clone(),
                            });
                        }
                        // **The backend's own permissions are read here too,
                        // for the same reason.** The resolved row carries two
                        // things: an *identity and configuration* (kind, URL,
                        // key, models directory, engine path — carried, and
                        // made safe to carry by the epoch) and *permissions*.
                        // Of the permissions, `enabled` retires the backend's
                        // engines and bumps its epoch, so the load and the
                        // lease already refuse a withdrawn one; the proxy's
                        // exposure of the backend and its `auto_start` move
                        // nothing of the kind, so the row's copies were a stale
                        // licence to start a subprocess. One read of the
                        // exposed row answers both, at the decision.
                        let starts_on_demand = {
                            let conn = self.db_conn().await?;
                            match db::exposed_backend(&conn, &backend.id).await? {
                                None => {
                                    return Err(AppError::ModelUnavailable {
                                        model: target.canonical.clone(),
                                    });
                                }
                                Some(current) => current.auto_start,
                            }
                        };
                        if target.kind == backends::BackendKind::LlamaCpp && !starts_on_demand {
                            return Err(AppError::NotConfigured {
                                message: format!(
                                    "`{}` is not loaded and backend `{}` has auto-start disabled",
                                    target.canonical, backend.id
                                ),
                            });
                        }
                        // **The row the request was authorized against, not
                        // a fresh read of its id.** See
                        // `Inner::load_authorized_engine`: a remove-and-re-add
                        // in the window this request has already opened puts a
                        // different models directory and a different engine
                        // behind the same name, and a load that resolves by id
                        // would start *that* one and lease it the prompt.
                        self.load_authorized_engine(backend, &target.model, epoch)
                            .await?;
                        self.local
                            .lease_authorized_engine(&backend.id, &target.model, epoch)
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
                    // Not `plain_client`: its builder adds a `User-Agent`, which
                    // would travel outside the enumerated header set the Record
                    // shows the reader (`local_models::proxy_http_client`).
                    client: self.proxy_client()?,
                    base_url: engine_url,
                    wire_model: target.canonical.clone(),
                    canonical: target.canonical.clone(),
                    backend_id: backend.id.clone(),
                    pricing: None,
                    external_auth: None,
                    // Written once a send is attempted, never at open — see
                    // `attach_plain_connection`.
                    connection_id: None,
                    attestations: None,
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
                    // Not `plain_client`: its builder adds a `User-Agent`, which
                    // would travel outside the enumerated header set the Record
                    // shows the reader (`local_models::proxy_http_client`).
                    client: self.proxy_client()?,
                    base_url,
                    wire_model: target.model.clone(),
                    canonical: target.canonical.clone(),
                    backend_id: backend.id.clone(),
                    pricing: None,
                    external_auth: backend.api_key.as_ref().map(|k| format!("Bearer {k}")),
                    // Written once a send is attempted, never at open — see
                    // `attach_plain_connection`.
                    connection_id: None,
                    attestations: None,
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
                // **The same deadline the listing takes.** Opening a route is
                // its own catalog fetch — pricing has to be read before a hold
                // can be computed — and it sat outside every bound: an endpoint
                // that accepted the connection and then said nothing left the
                // completion, and the proxy connection behind it, pending for
                // ever without the chat request ever being made. One deadline
                // per catalog read, wherever the read happens.
                let models = match tokio::time::timeout(
                    model_list_timeout(),
                    fetch_models(&client, &eidola.base_url),
                )
                .await
                {
                    Ok(models) => models?,
                    Err(_) => {
                        return Err(AppError::Network {
                            message: format!(
                                "`{}` did not answer its model catalog in time",
                                backend.id
                            ),
                        });
                    }
                };
                let connection_id =
                    flush_attestations(&log, &db_conn, &provider_id, &eidola.base_url, now).await?;
                if connection_id.is_some() {
                    self.bus.emit(Change::Record);
                }
                // The completion's own handshake, as the plain test client can
                // never produce one: observed after the catalog's was recorded.
                #[cfg(feature = "test-support")]
                if let Some(hash) = self
                    .proxy_planted_handshake
                    .lock()
                    .expect("planted handshake lock poisoned")
                    .take()
                {
                    log.lock()
                        .unwrap()
                        .push(tinfoil_verifier::VerifiedAttestation {
                            platform: tinfoil_verifier::Platform::SevSnp,
                            matched_measurement: tinfoil_verifier::MatchedMeasurement::SevSnp(
                                "a".repeat(96),
                            ),
                            attestation_hash: hash,
                            attestation_doc: b"report".to_vec(),
                            pcr_digest: "b".repeat(96),
                            peer_spki_hash: "c".repeat(64),
                        });
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
                    pricing: Some(ChargePricing::from_catalog(&entry.pricing)),
                    external_auth: None,
                    connection_id,
                    attestations: Some(AttestationSink {
                        log,
                        provider_id,
                        base_url: eidola.base_url.clone(),
                    }),
                    declared_max_output: entry.max_output_tokens,
                    engine_lease: None,
                })
            }
        }
    }

    /// Give a plain route its `connection` row — **once a send is about to be
    /// attempted, and not before.**
    ///
    /// A `connection` row is a statement that a transport was used. Written at
    /// route open, it outran the one failure that sends nothing at all: the
    /// request build, which refuses on a user-typed key that is not a header
    /// value or a base URL that does not parse — and the Record then attached
    /// that refusal to a connection nothing ever opened. So the row is written
    /// between the build and the send, which is the earliest moment the claim
    /// is true; a pre-send failure keeps its destination in its own error text
    /// instead ([`unsent_error`]). An attested route needs none of this — its
    /// rows come from handshakes, which are evidence of a transport by
    /// construction. Best-effort like every other Record write here: a failed
    /// insert costs the row, never the request.
    async fn attach_plain_connection(&self, route: &mut ProxyRoute) {
        if route.attestations.is_some() || route.connection_id.is_some() {
            return;
        }
        match self
            .record_plain_connection(&route.backend_id, &route.base_url, now_ms())
            .await
        {
            Ok(id) => route.connection_id = Some(id),
            Err(e) => eprintln!("warning: a proxied request's destination was not recorded: {e}"),
        }
    }

    /// Record the destination of a route **no enclave vouches for**: a
    /// `connection` row with the URL this request is about to be sent to, and
    /// no attestation.
    ///
    /// **The Record's question is where the prompt went, and it has to be
    /// answerable after every mutation.** An attested route answers it through
    /// the `connection` row its handshake writes; a plain route wrote none, so
    /// the request row's only path to a URL was its `backend_id` — a foreign
    /// key to a row the reader can edit, or remove and re-add with a different
    /// URL under the same name, and an engine's address is a port that changes
    /// on every start. After any of those, the trail could no longer say where
    /// an earlier prompt left the machine, which is the question it exists to
    /// answer. The schema already admits the honest shape — `attestation_hash`
    /// is nullable and `clearnet` is the transport a plain HTTP connection is —
    /// so nothing is claimed that did not happen: a row with no attestation is
    /// a statement that nothing was verified.
    async fn record_plain_connection(
        &self,
        backend_id: &str,
        base_url: &str,
        now: i64,
    ) -> Result<String, AppError> {
        let conn = self.db_conn().await?;
        let provider_id = db::ensure_provider(&conn, backend_id, "inference", now).await?;
        crate::insert_plain_connection(&conn, &provider_id, base_url, now).await
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
    ) -> Value {
        let mut body = eidola_common::chat_completion_request_body(
            wire_model,
            &request.messages,
            max_completion_tokens,
            request.tool_schemas(),
            stream,
            // **The caller's option, not a rule about billing.** This asked for
            // usage on every route that does not bill — which is *both*
            // zero-spend kinds, an engine and an external OpenAI backend, since
            // neither carries pricing. On an external backend that is a field
            // the downstream never sent: a strict OpenAI-compatible server that
            // does not implement it rejects every proxied stream, and a lenient
            // one emits a usage event nobody asked for.
            //
            // The app's own turns ask because they *consume* usage — the token
            // counts go on the `inference` row. Nothing on this path reads
            // `usage` at all: a billed route settles from the refund the server
            // hands back, and a zero-spend route settles nothing. So the honest
            // rule here is the pass-through one: ask exactly when the caller
            // asked, on every route kind. (A billed route gets a usage chunk
            // regardless, because the Eidola server forces `include_usage`
            // upstream for its own refunds — asking or not asking changes
            // nothing there.)
            stream && request.include_usage,
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
        headers: &SentHeaders,
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
            // **The retention cap's request-side twin, and the durable half.**
            // A caller may send `MAX_REQUEST_BYTES` and repeat it, and every
            // exchange wrote the reconstructed body down in full — a few dozen
            // calls adding a gigabyte to the profile database and its WAL, with
            // nothing pruning `request` rows to take it back. The prompt still
            // travels upstream whole; what is bounded is what is *kept*, and a
            // row that keeps less says so in the payload itself.
            request_body: Some(recorded_request(request_body)),
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

    /// Everything a completion does **before anything is sent** — and the one
    /// door its refusals leave through.
    ///
    /// A request row per completion is the Record's own contract, and these
    /// exits were where it broke: each was a `?` between authentication and
    /// the first `record_proxy_request`, so an authenticated tool asking for a
    /// model and being refused left nothing behind. They are every exit there
    /// is before a send — what was refused, and why it wrote no row:
    ///
    /// | Exit | What refuses |
    /// | --- | --- |
    /// | `resolve_proxy_target` | the backend is not exposed, was removed or disabled; the profile's read failed; a target that cannot be built |
    /// | `open_proxy_route` | an engine the exposure setting does not permit starting, auto-start off, a load that failed, an engine unloaded while starting; a catalog that did not answer, did not attest, or does not list the model |
    /// | `db_conn` | the profile's database cannot be opened |
    /// | the zero-charge refusal | catalog pricing that computes no charge |
    /// | `acquire_spend` | no account, no credential to spend, the wallet read or write failed |
    ///
    /// **Structural rather than remembered:** [`Self::preflighted`] is the only
    /// caller, and it records whatever this returns as an error — so an exit
    /// added here later is recorded without anyone adding a line for it, and
    /// none of the rows above is written per site. Past this point the request
    /// has a route and a hold, and the transport's own arms record it.
    async fn proxy_preflight(
        &self,
        request: &ProxyChatRequest,
        stream: bool,
    ) -> Result<ProxyPreflight, AppError> {
        let (target, epoch) = self.resolve_proxy_target(&request.model).await?;
        let route = self.open_proxy_route(&target, epoch).await?;
        let max_completion_tokens =
            Self::proxy_completion_budget(request, route.declared_max_output);
        let body =
            Self::proxy_request_body(request, &route.wire_model, max_completion_tokens, stream);

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
        Ok(ProxyPreflight {
            route,
            body,
            db_conn,
            spend,
            auth_value,
        })
    }

    /// [`Self::proxy_preflight`], with every refusal written into the Record
    /// on its way out.
    async fn preflighted(
        &self,
        request: &ProxyChatRequest,
        stream: bool,
    ) -> Result<ProxyPreflight, AppError> {
        let arrived_at = now_ms();
        // Boxed: the preflight's state (an engine load, a catalog fetch, a
        // credential acquisition) is a large future, and nested inline in each
        // transport's it took a debug build's worker thread past its stack.
        let result = Box::pin(self.proxy_preflight(request, stream)).await;
        if let Err(error) = &result {
            self.record_proxy_refusal(request, stream, error, arrived_at)
                .await;
        }
        result
    }

    /// The Record row for a completion refused before anything was sent.
    ///
    /// **Unattached in every column that would claim something happened.** No
    /// `connection_id` — no transport carried this request (a catalog fetch
    /// may have opened one, but this prompt did not go down it); no
    /// `backend_id` — the refusal may be that there is no such backend, or
    /// none this proxy may name, and the column is a reference to a row; no
    /// request headers, response or credential, because none were sent, read
    /// or spent. What it does keep is what the tool asked for: the request in
    /// the outer shape every proxied body takes, naming the model as the
    /// caller named it — the wire name was never resolved — at the size the
    /// caller asked for, and the refusal in the caller's own words with the
    /// fact the reader most needs beside it.
    async fn record_proxy_refusal(
        &self,
        request: &ProxyChatRequest,
        stream: bool,
        error: &AppError,
        arrived_at: i64,
    ) {
        let Ok(conn) = self.db_conn().await else {
            return;
        };
        let asked = Self::proxy_request_body(
            request,
            &request.model,
            request
                .max_completion_tokens
                .unwrap_or(DEFAULT_MAX_COMPLETION_TOKENS),
            stream,
        );
        let refused_at = now_ms();
        let entry = db::Request {
            id: Uuid::now_v7().to_string(),
            connection_id: None,
            action_id: None,
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            request_headers: None,
            request_body: Some(recorded_request(&asked)),
            response_status: None,
            response_headers: None,
            response_body: None,
            request_at: arrived_at,
            response_at: Some(refused_at),
            duration_ms: Some(refused_at - arrived_at),
            error: Some(format!("{error} — refused before anything was sent")),
            credential_nonce: None,
            created_at: refused_at,
            backend_id: None,
        };
        if db::insert_request(&conn, &entry).await.is_ok() {
            self.bus.emit(Change::Record);
        }
    }

    /// One non-streaming proxied completion.
    pub(crate) async fn proxy_chat(
        &self,
        request: ProxyChatRequest,
    ) -> Result<ProxyChatResponse, AppError> {
        let ProxyPreflight {
            mut route,
            body,
            db_conn,
            spend,
            auth_value,
        } = self.preflighted(&request, false).await?;
        let nonce = spend.as_ref().map(|s| s.cred.nonce.clone());
        // Materialised **once** for this attempt: the wire and the Record row
        // then carry the same `traceparent`, which is the whole point of the
        // set being a value rather than a recipe.
        let headers = UpstreamHeaders {
            authorization: auth_value.clone(),
            streaming: false,
            trace: self.upstream_tracing(),
        }
        .materialize();

        let request_at = now_ms();
        // **A failure here is past the hold, so it settles like every other
        // one.** The build can refuse — an external backend's key is user-typed
        // and may not be a header value — and a `?` would have returned with a
        // credential left `spending` and nothing in the Record to say why.
        let outbound = match build_upstream_request(&route, &body, &headers) {
            Ok(outbound) => outbound,
            Err(error) => {
                self.settle_proxy_refund(&db_conn, &spend, &auth_value, &route, None)
                    .await;
                self.record_proxy_request(
                    &route,
                    &headers,
                    &body,
                    None,
                    Vec::new(),
                    Some(unsent_error(&route, &error)),
                    nonce,
                    request_at,
                    now_ms(),
                )
                .await;
                return Err(error);
            }
        };
        // A send is about to be attempted, so a plain route's destination is
        // now a connection this request used.
        self.attach_plain_connection(&mut route).await;

        let response = match route.client.execute(outbound).await {
            Ok(response) => response,
            Err(e) => {
                let error = AppError::from_request(e);
                // **A send that failed may still have opened a connection.** The
                // handshake completes before a byte of the request is written,
                // so an attempt that dies before the response head can have
                // verified a fresh enclave the catalog fetch never reached —
                // and the row below would name the catalog's connection as the
                // one this prompt went down. Flushed before the settlement,
                // whose own recovery call may handshake again on a connection
                // that is not this request's.
                if route.flush_new_attestations(&db_conn).await {
                    self.bus.emit(Change::Record);
                }
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
        // The completion may have opened a connection of its own; the Record
        // must name the one it actually went down.
        if route.flush_new_attestations(&db_conn).await {
            self.bus.emit(Change::Record);
        }
        let status = response.status();
        let answer = match read_capped_body(response).await {
            Ok(answer) => answer,
            Err(failure) => {
                let error = failure.error;
                self.settle_proxy_refund(&db_conn, &spend, &auth_value, &route, None)
                    .await;
                self.record_proxy_request(
                    &route,
                    &headers,
                    &body,
                    Some(status.as_u16()),
                    recorded_cut_answer(&failure.partial),
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
        let text = answer.text();
        // **A body that does not parse is not an answer.** Coercing it to JSON
        // `null` and keeping the upstream's `2xx` would hand a downstream tool
        // an apparent success carrying a fabricated body — a truncated
        // response or an intermediary's HTML error page reads as "the model
        // said nothing". The exchange is still recorded and the hold still
        // settles below; only the *answer* becomes the gateway failure it is.
        //
        // **And a body the ceiling stopped is refused whatever it parses as**,
        // which is not the same claim. "A fragment does not parse" is true of
        // most oversized JSON and false of the case that matters: a *complete*
        // object followed by enough whitespace to cross the ceiling parses
        // perfectly, because trailing whitespace is valid — so the caller was
        // handed a success while the Record row beside it said the read had
        // stopped at this app's ceiling, the two describing one exchange
        // differently. `over_ceiling` is the app's own decision to stop reading
        // and is therefore asked *before* the parse rather than inferred from
        // it — the `whole_text` rule (`peer_read`), on the one surface that
        // keeps what it read instead of refusing outright.
        let parsed: Option<Value> = (!answer.over_ceiling)
            .then(|| serde_json::from_str::<Value>(&text).ok())
            .flatten();
        // **And syntax is not shape.** `{}`, `null`, and an upstream's error
        // document all parse, so a check that ended at `from_str` handed a
        // downstream tool a `200` carrying nothing it could read as an answer —
        // with this app's own `model` inserted into it, which makes the
        // fabrication look more like a completion rather than less. The floor
        // is [`is_completion`]: exactly what this app's own turn path requires
        // of a blocking answer, so the proxy refuses nothing its chat would
        // have accepted.
        //
        // **A refusal this app made belongs in the row it made it about.** The
        // upstream's status is the upstream's claim; a `2xx` this proxy would
        // not accept is recorded as an error too, or the Record shows the
        // exchange as the success the caller was explicitly not given. A
        // non-2xx needs no such column — its status already says what happened.
        //
        // **And the ceiling refuses whatever the status.** A non-2xx is passed
        // through as the upstream's own answer only when it arrived whole; a
        // fragment of one is this app's refusal like any other
        // (`answer_past_ceiling`).
        let refusal = if answer.over_ceiling {
            Some(crate::peer_read::answer_past_ceiling(&route.backend_id))
        } else {
            status
                .is_success()
                .then(|| match parsed.as_ref() {
                    None => Some(malformed_json_answer(&route.backend_id, status.as_u16())),
                    Some(body) if !is_completion(body) => {
                        Some(shapeless_answer(&route.backend_id, status.as_u16()))
                    }
                    Some(_) => None,
                })
                .flatten()
        };
        self.settle_proxy_refund(
            &db_conn,
            &spend,
            &auth_value,
            &route,
            parsed.as_ref().and_then(|body| body.get("refund")),
        )
        .await;
        self.record_proxy_request(
            &route,
            &headers,
            &body,
            Some(status.as_u16()),
            recorded_answer(&answer),
            refusal.as_ref().map(ToString::to_string),
            nonce,
            request_at,
            response_at,
        )
        .await;

        if answer.over_ceiling
            && let Some(refusal) = refusal
        {
            return Err(refusal);
        }
        if !status.is_success() {
            return Err(AppError::Server {
                status: status.as_u16(),
                message: parse_server_error_message(&text),
            });
        }

        // This is the arm `refusal` was built for — the status is a success and
        // there is no answer — so the caller is told exactly what the row says,
        // by carrying the same value rather than rebuilding it. The settlement
        // above ran first on purpose: a body this app will not accept can still
        // carry the credential's successor, and the class rule is that whatever
        // the server *hands* us settles, whatever else the body turns out to be.
        if let Some(refusal) = refusal {
            return Err(refusal);
        }
        // A success that reached here parsed and passed the shape check, so the
        // `else` is unreachable — and refuses rather than inventing a body.
        let Some(mut parsed) = parsed else {
            return Err(malformed_json_answer(&route.backend_id, status.as_u16()));
        };
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
        sender: tokio::sync::mpsc::Sender<ProxyStreamEvent>,
    ) -> Result<(), AppError> {
        let ProxyPreflight {
            mut route,
            body,
            db_conn,
            spend,
            auth_value,
        } = self.preflighted(&request, true).await?;
        let nonce = spend.as_ref().map(|s| s.cred.nonce.clone());
        // Materialised **once** for this attempt: the wire and the Record row
        // then carry the same `traceparent`, which is the whole point of the
        // set being a value rather than a recipe.
        let headers = UpstreamHeaders {
            authorization: auth_value.clone(),
            streaming: true,
            trace: self.upstream_tracing(),
        }
        .materialize();

        let request_at = now_ms();
        // Past the hold, so a refusal settles it — the blocking transport's
        // arm, for the same reason.
        let outbound = match build_upstream_request(&route, &body, &headers) {
            Ok(outbound) => outbound,
            Err(error) => {
                self.settle_proxy_refund(&db_conn, &spend, &auth_value, &route, None)
                    .await;
                self.record_proxy_request(
                    &route,
                    &headers,
                    &body,
                    None,
                    Vec::new(),
                    Some(unsent_error(&route, &error)),
                    nonce,
                    request_at,
                    now_ms(),
                )
                .await;
                return Err(error);
            }
        };
        // A send is about to be attempted, so a plain route's destination is
        // now a connection this request used.
        self.attach_plain_connection(&mut route).await;

        let response = match route.client.execute(outbound).await {
            Ok(response) => response,
            Err(e) => {
                let error = AppError::from_request(e);
                // **A send that failed may still have opened a connection.** The
                // handshake completes before a byte of the request is written,
                // so an attempt that dies before the response head can have
                // verified a fresh enclave the catalog fetch never reached —
                // and the row below would name the catalog's connection as the
                // one this prompt went down. Flushed before the settlement,
                // whose own recovery call may handshake again on a connection
                // that is not this request's.
                if route.flush_new_attestations(&db_conn).await {
                    self.bus.emit(Change::Record);
                }
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
        // Same as the blocking transport: the prompt may have travelled down a
        // connection the catalog fetch never opened.
        if route.flush_new_attestations(&db_conn).await {
            self.bus.emit(Change::Record);
        }
        let status = response.status();
        if !status.is_success() {
            // Bounded like every other blocking read on this path: an error
            // body is a body, and an intermediary's is the one most likely to
            // be enormous.
            // **A body that could not be read is a read failure, not an empty
            // answer.** Defaulting it turned an upstream that reset mid-body
            // into a clean read of nothing: the caller got the upstream's status
            // (or a shape refusal) instead of the transport failure that really
            // happened, and the Record row carried no error at all. The blocking
            // transport's arm, for the same reason: settle, record the failure,
            // return it.
            let answer = match read_capped_body(response).await {
                Ok(answer) => answer,
                Err(failure) => {
                    let error = failure.error;
                    self.settle_proxy_refund(&db_conn, &spend, &auth_value, &route, None)
                        .await;
                    self.record_proxy_request(
                        &route,
                        &headers,
                        &body,
                        Some(status.as_u16()),
                        recorded_cut_answer(&failure.partial),
                        Some(error.to_string()),
                        nonce,
                        request_at,
                        now_ms(),
                    )
                    .await;
                    return Err(error);
                }
            };
            let text = answer.text();
            // **A stream that never opened can still carry its refund.** The
            // server spends the credential before it dispatches, so every
            // failure after the nullifier is recorded — request validation,
            // `send_stream`, a spend-proof re-encode — answers with a
            // refund-bearing JSON error body rather than an SSE stream
            // (`eidola-server/src/handlers.rs`: `error_response_with_refund`).
            // Persisting that token for recovery is best-effort there, so the
            // in-band value is again the only one that can answer for the arm
            // where persistence failed. Third door, same rule: the refund the
            // server *hands* us settles; recovery is what absence falls back
            // to. This is the arm the blocking transport already covered by
            // reading `refund` off the parsed body before it looks at the
            // status.
            // A body the ceiling stopped is never parsed (the `whole_text`
            // rule) and is refused whatever its status.
            let inline = (!answer.over_ceiling)
                .then(|| serde_json::from_str::<Value>(&text).ok())
                .flatten()
                .and_then(|body| body.get("refund").cloned());
            let refusal = answer
                .over_ceiling
                .then(|| crate::peer_read::answer_past_ceiling(&route.backend_id));
            self.settle_proxy_refund(&db_conn, &spend, &auth_value, &route, inline.as_ref())
                .await;
            self.record_proxy_request(
                &route,
                &headers,
                &body,
                Some(status.as_u16()),
                recorded_answer(&answer),
                refusal.as_ref().map(ToString::to_string),
                nonce,
                request_at,
                now_ms(),
            )
            .await;
            return Err(refusal.unwrap_or_else(|| AppError::Server {
                status: status.as_u16(),
                message: parse_server_error_message(&text),
            }));
        }

        // **A `200` is not an answer until it is the *shape* that was asked
        // for.** Opening downstream commits this response to
        // `200 text/event-stream`, and the head cannot be taken back — so a
        // backend that ignored `stream: true` and answered a normal JSON
        // completion, or an intermediary that answered an HTML page, would
        // have its body forwarded as an unterminated SSE fragment under a
        // status saying everything went well. That is the blocking transport's
        // malformed-2xx rule (a `2xx` that is not JSON is a gateway failure)
        // read on the other transport, where the wrong shape is *not* JSON.
        //
        // Strict when the header is present, permissive when it is absent: the
        // two cases this exists for both name a content type (`application/json`
        // and `text/html`), while a compliant SSE server always sets one, so
        // refusing an unlabelled body would only break a server that is already
        // unusual without catching anything the named cases do not.
        //
        // **And this is a fourth arm of the refund class**: a backend that
        // ignored `stream: true` may well have answered with a whole
        // completion, refund and all — the credential is spent either way, so
        // the body is read for its token before this fails.
        if !is_event_stream(response.headers()) {
            // **A body that could not be read is a read failure, not an empty
            // answer.** Defaulting it turned an upstream that reset mid-body
            // into a clean read of nothing: the caller got the upstream's status
            // (or a shape refusal) instead of the transport failure that really
            // happened, and the Record row carried no error at all. The blocking
            // transport's arm, for the same reason: settle, record the failure,
            // return it.
            let answer = match read_capped_body(response).await {
                Ok(answer) => answer,
                Err(failure) => {
                    let error = failure.error;
                    self.settle_proxy_refund(&db_conn, &spend, &auth_value, &route, None)
                        .await;
                    self.record_proxy_request(
                        &route,
                        &headers,
                        &body,
                        Some(status.as_u16()),
                        recorded_cut_answer(&failure.partial),
                        Some(error.to_string()),
                        nonce,
                        request_at,
                        now_ms(),
                    )
                    .await;
                    return Err(error);
                }
            };
            let text = answer.text();
            let inline = (!answer.over_ceiling)
                .then(|| serde_json::from_str::<Value>(&text).ok())
                .flatten()
                .and_then(|body| body.get("refund").cloned());
            // The refusal is this app's, so the row carries it: the upstream
            // said `200` and nothing was forwarded, which a row holding only
            // that status would present as an answered request. The ceiling,
            // where it stopped the read, is the refusal that names it.
            let refusal = if answer.over_ceiling {
                crate::peer_read::answer_past_ceiling(&route.backend_id)
            } else {
                malformed_stream_answer(&route.backend_id, status.as_u16())
            };
            self.settle_proxy_refund(&db_conn, &spend, &auth_value, &route, inline.as_ref())
                .await;
            self.record_proxy_request(
                &route,
                &headers,
                &body,
                Some(status.as_u16()),
                recorded_answer(&answer),
                Some(refusal.to_string()),
                nonce,
                request_at,
                now_ms(),
            )
            .await;
            return Err(refusal);
        }

        // Only now is there going to be a `200` downstream. **A caller that
        // has already gone is noticed here rather than at the first chunk**:
        // an upstream that answers and closes with nothing in it would
        // otherwise leave the ending classified as a complete delivery to a
        // caller who was not there.
        let mut downstream_gone = false;
        deliver(&sender, ProxyStreamEvent::Open, &mut downstream_gone).await;

        let mut byte_stream = response.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        // What the Record will keep, and how much really arrived — the two are
        // not the same number, and the row says so when they differ.
        let mut raw = RecordedBody::default();
        let mut read_error: Option<AppError> = None;
        // The refund the upstream closed the stream with, if it sent one.
        let mut inline_refund: Option<Value> = None;
        // **The caller has gone, and what that ends depends on what is owed.**
        // A spending route's refund arrives at the *end* of the stream, so the
        // upstream is drained to reach it — delivery is what a vanished caller
        // loses, never this app's accounting. With nothing to settle there is
        // nothing at the end worth reading, so the read stops with the caller.
        // Whether the loop above stopped because the caller had gone rather
        // than because the upstream had. The Record says which.
        let mut read_ended_early = false;
        let settling = spend.is_some();
        loop {
            // **The caller is watched while nothing arrives**, or a silent
            // upstream would hold this read — and the connection and engine
            // lease behind it — long after there was anyone to deliver to.
            // See [`next_chunk`] for why a spending route waits anyway.
            let chunk = match next_chunk(&mut byte_stream, &sender, settling).await {
                NextChunk::Chunk(chunk) => chunk,
                NextChunk::UpstreamEnded => break,
                NextChunk::CallerGone => {
                    downstream_gone = true;
                    read_ended_early = true;
                    break;
                }
            };
            let bytes = match chunk {
                Ok(bytes) => bytes,
                Err(e) => {
                    read_error = Some(AppError::Network {
                        message: format!("stream read failed: {e}"),
                    });
                    break;
                }
            };
            raw.push(&bytes);
            buf.extend_from_slice(&bytes);
            let mut oversized = false;
            while let Some((pos, boundary_len)) = find_event_boundary(&buf) {
                // The frame ceiling is asked of a complete event **before** it
                // is drained and forwarded (`peer_read::event_past_ceiling`).
                if crate::peer_read::event_past_ceiling(Some(pos), 0) {
                    oversized = true;
                    break;
                }
                let event: Vec<u8> = buf.drain(..pos).collect();
                let terminator: Vec<u8> = buf.drain(..boundary_len).collect();
                let (mut out, refund) = forward_sse_event(&event, &route.canonical);
                if refund.is_some() {
                    inline_refund = refund;
                }
                if downstream_gone {
                    // Still parsed, because the refund is in here somewhere;
                    // no longer sent, because there is nobody to send it to.
                    continue;
                }
                out.extend_from_slice(&terminator);
                // **Awaited, on a bounded queue** — that is what makes a
                // client which stops reading stop being served rather than be
                // buffered at, and it is what bounds what this process holds
                // for one connection. The pressure travels outward: the pump
                // waits on the queue, the queue waits on hyper, hyper waits on
                // the socket, and the only place that can relieve it is the
                // caller reading. Nothing waits on anything behind it.
                //
                // **It always terminates, and the taxonomy is three lines:** a
                // caller that goes away drops hyper's body, which drops the
                // receiver, which fails this send immediately — forwarding
                // ends here and now. A caller that stalls without going away
                // holds the pump, which is deliberate backpressure and delays
                // the settlement rather than losing it: the hold stays
                // `spending` with a live request behind it, which is exactly
                // what `LiveSpend` keeps recovery out of. And a caller that
                // went away on a route with a hold to settle leaves the drain
                // below running to the refund, on a body the cap bounds.
                deliver(&sender, ProxyStreamEvent::Chunk(out), &mut downstream_gone).await;
            }
            // **What is left over is an event still being accumulated**, and
            // it is the one buffer on this path with no ceiling of its own: the
            // retention cap bounds the Record and the queue bounds delivery,
            // while a backend that never terminates an event grows this until
            // the process dies. That residual half is measured after the drain;
            // the complete half — an oversized event whose boundary arrived in
            // the same read, which the drain would otherwise have taken whole —
            // is refused before it (`oversized`). The residual overshoot is a
            // single transport chunk, the transport's own bound rather than ours.
            if oversized || crate::peer_read::event_past_ceiling(None, buf.len()) {
                read_error = Some(crate::peer_read::oversized_event(&route.backend_id));
                // Refused, so not forwarded: the tail below exists to hand a
                // downstream parser the bytes the upstream really sent, and
                // these are the bytes this app has just declined to accept.
                buf.clear();
                break;
            }
            if downstream_gone && !settling {
                read_ended_early = true;
                break;
            }
        }
        // Whatever is left is a partial event; forward it so a downstream
        // parser sees exactly the bytes the upstream sent — through the same
        // door, because this send can fail for the same reason and its failure
        // is the same fact: an unterminated tail nobody received is not a
        // complete delivery, and discarding the result sealed the row as one.
        if !buf.is_empty() {
            let (out, refund) = forward_sse_event(&buf, &route.canonical);
            if refund.is_some() {
                inline_refund = refund;
            }
            deliver(&sender, ProxyStreamEvent::Chunk(out), &mut downstream_gone).await;
        }

        // **The in-band token first, recovery only for its absence.** The
        // server closes an Eidola stream with a metadata event carrying the
        // refund, and sends it even when its own persistence of that token
        // failed — the one case where recovery can never answer. See
        // [`forward_sse_event`].
        self.settle_proxy_refund(
            &db_conn,
            &spend,
            &auth_value,
            &route,
            inline_refund.as_ref(),
        )
        .await;
        self.record_proxy_request(
            &route,
            &headers,
            &body,
            Some(status.as_u16()),
            {
                // The read reached the upstream's own end only if nothing
                // stopped it — neither the caller's departure nor a failure.
                let read_to_end = !read_ended_early && read_error.is_none();
                raw.seal_stream(stream_delivery(downstream_gone, !read_to_end), read_to_end)
            },
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

/// Whether an engine-backed backend contributes only the models whose engine
/// is already running.
///
/// **`auto_start` is the backend's own rule and exposure does not override
/// it.** A `llamacpp` backend with auto-start off refuses a request-triggered
/// load before any spawn (app-core's local-model doctrine), and a proxied
/// request *is* a request — so under `Downloaded` such a backend's unloaded
/// models would be advertised and then deterministically refused at open, with
/// the listing promising what the route can never serve. Exposure is a
/// permission the reader grants the proxy *over* a backend; it is not a licence
/// to break that backend's own configuration. So a backend that will not start
/// an engine on demand takes the running-engine filter whatever the exposure
/// setting says, which is what keeps the listing and the route agreeing.
/// What the proxy can say about an engine the filesystem scan cannot name.
///
/// The registry has no display metadata — that lives in a `.meta.json` sidecar
/// beside the file that may be gone — so this is the id the route answers to
/// plus the context length the engine was actually started with. Zero pricing
/// is not a claim: an engine-backed model is free by construction, which is
/// exactly what `plain_model_info` says for every other locally-served row.
fn engine_model_info(id: String, context_tokens: u32) -> ModelInfo {
    ModelInfo {
        id,
        context_length: u64::from(context_tokens),
        max_output_tokens: None,
        output_budget_class: None,
        capabilities: Default::default(),
        prompt_credits_per_token: 0.0,
        completion_credits_per_token: 0.0,
        request_credits: None,
    }
}

fn offers_running_engines_only(exposure: LocalExposure, starts_on_demand: bool) -> bool {
    exposure == LocalExposure::Loaded || !starts_on_demand
}

/// Build the upstream request so its header set is **exactly** the allowlist.
///
/// "These headers and nothing else" was untrue in three ways at once, and only
/// one of them was written anywhere: `RequestBuilder::header` **appends**, so
/// `json()`'s `Content-Type` and the allowlist's became two of them on the
/// wire; reqwest adds an `Accept: */*` of its own, which on a streaming request
/// sat beside `Accept: text/event-stream` and left the upstream to choose;
/// and the client's builder added a `User-Agent`. The claim is not something to
/// re-audit each time a dependency moves, so the request is **built and then
/// its headers replaced**: whatever a client or a body helper put there is
/// cleared, and the enumeration becomes a fact. What hyper adds afterwards is
/// framing (`Host`, `Content-Length`) — the protocol, not a claim about this
/// app — and the body is serialized here rather than by `json()` for the same
/// reason: a helper that sets a header is a second author of the set.
fn build_upstream_request(
    route: &ProxyRoute,
    body: &Value,
    headers: &SentHeaders,
) -> Result<reqwest::Request, AppError> {
    let mut request = route
        .client
        .post(format!("{}/v1/chat/completions", route.base_url))
        .body(body.to_string())
        .build()
        .map_err(|e| AppError::Network {
            message: format!(
                "building the upstream request: {}",
                crate::error::request_error_text(e)
            ),
        })?;
    let out = request.headers_mut();
    out.clear();
    for (name, value) in headers.pairs() {
        let name = reqwest::header::HeaderName::from_bytes(name.as_bytes()).map_err(|_| {
            AppError::Config {
                message: format!("`{name}` is not a usable header name"),
            }
        })?;
        let value =
            reqwest::header::HeaderValue::from_str(value).map_err(|_| AppError::Config {
                message: format!("`{name}` carries a value that cannot travel in a header"),
            })?;
        out.insert(name, value);
    }
    Ok(request)
}

/// Send one event downstream, and let a failed send be the fact it is.
///
/// **The one door out of the forwarding loop**, so no send can be made whose
/// result nobody reads: the ending a stream is recorded as is decided from
/// `downstream_gone` (see [`stream_delivery`]), and a discarded failure seals a
/// delivery that did not happen as `Complete`. The `Open` event, every
/// forwarded event and the unterminated tail all go through here for that
/// reason — the tail was the one that did not, and it is exactly the send most
/// likely to meet a caller who has already gone.
async fn deliver(
    sender: &tokio::sync::mpsc::Sender<ProxyStreamEvent>,
    event: ProxyStreamEvent,
    downstream_gone: &mut bool,
) {
    if *downstream_gone {
        return;
    }
    if sender.send(event).await.is_err() {
        *downstream_gone = true;
    }
}

/// What ended one turn of the forwarding loop's wait.
enum NextChunk<T> {
    /// The upstream produced something — a chunk, or the failure of one.
    Chunk(T),
    /// The upstream closed.
    UpstreamEnded,
    /// The caller went away while nothing was arriving.
    CallerGone,
}

/// Wait for the next chunk — **watching the caller while nothing arrives.**
///
/// A vanished caller is otherwise noticed only at a send, which never comes if
/// the upstream has gone quiet: a backend that answers its head and then says
/// nothing, or stalls between chunks, left this `await` outstanding for as long
/// as it cared to, holding the upstream connection and any engine lease behind
/// it while nobody was left to receive a byte. So with **nothing to settle**
/// the wait watches the receiver too, and the caller's departure ends the read
/// the same way a send failure would.
///
/// With a hold to settle it does not: a spending route's refund arrives at the
/// *end* of the stream, so the drain is deliberate and delivery is what a
/// vanished caller loses, never this app's accounting. `biased` so a chunk
/// already in hand is always preferred to noticing the departure.
async fn next_chunk<S, T>(
    stream: &mut S,
    sender: &tokio::sync::mpsc::Sender<ProxyStreamEvent>,
    settling: bool,
) -> NextChunk<T>
where
    S: futures_util::Stream<Item = T> + Unpin,
{
    use futures_util::StreamExt;

    let arrived = if settling {
        stream.next().await
    } else {
        tokio::select! {
            biased;
            chunk = stream.next() => chunk,
            () = sender.closed() => return NextChunk::CallerGone,
        }
    };
    match arrived {
        Some(chunk) => NextChunk::Chunk(chunk),
        None => NextChunk::UpstreamEnded,
    }
}

/// The gateway failure a `2xx` whose body is not JSON becomes.
///
/// **Built once and used twice**, because the caller's answer and the Record's
/// `error` column have to be the same sentence: a row carrying only `200` shows
/// a refusal this app made as the success it was not, and a row whose wording
/// drifts from what the tool was told is worse than either.
fn malformed_json_answer(backend_id: &str, status: u16) -> AppError {
    AppError::Network {
        message: format!("`{backend_id}` answered {status} with a body that is not JSON"),
    }
}

/// The Record's sentence for a request that was never sent.
///
/// Nothing was opened, so there is no `connection` row to say where it would
/// have gone — and the question "where would this prompt have left the
/// machine" still deserves an answer, so a plain route's refusal carries its
/// destination in its own words. What reaches the *caller* is the error alone:
/// the backend's URL is the reader's configuration, not the tool's business.
/// An attested route says nothing extra — its build cannot fail on a URL the
/// catalog fetch has already used, and the connection it is recorded against
/// is one that handshake really opened.
fn unsent_error(route: &ProxyRoute, error: &AppError) -> String {
    if route.attestations.is_some() {
        return error.to_string();
    }
    format!(
        "{error} — nothing was sent; the request would have gone to {}",
        route.base_url
    )
}

/// Whether a parsed `2xx` body is a completion at all.
///
/// **The bar is this app's own, not a stricter one invented here.** The turn
/// path reads a blocking answer by walking `choices` as an array and taking the
/// first element's `message` (`lib.rs`), so a body with no `choices` array
/// carries nothing that path would call an answer — while everything above that
/// floor (an empty `choices`, a `message` with no `content`, unknown fields) is
/// something it reads without complaint, and the proxy must too. Matching the
/// floor exactly is what keeps this from refusing completions the app's own
/// chat accepts.
///
/// `Value::get` answers `None` for anything that is not an object, so `null`, a
/// bare array and a string are refused by the same line.
fn is_completion(body: &Value) -> bool {
    body.get("choices").is_some_and(Value::is_array)
}

/// A `2xx` whose body is JSON but not a completion — `{}`, `null`, or an
/// upstream's error document answered with a success status.
fn shapeless_answer(backend_id: &str, status: u16) -> AppError {
    AppError::Network {
        message: format!(
            "`{backend_id}` answered {status} with JSON that is not a chat completion"
        ),
    }
}

/// The streaming twin: a `2xx` that is not server-sent events.
fn malformed_stream_answer(backend_id: &str, status: u16) -> AppError {
    AppError::Network {
        message: format!(
            "`{backend_id}` answered {status} to a streaming request with a body that is not \
             server-sent events"
        ),
    }
}

/// One SSE event on its way downstream, and the refund it was carrying.
///
/// **Byte-identical unless one of two fields is this app's business.** The
/// answer a model gave is not this app's to reformat, so an event whose
/// payloads carry neither is passed through exactly as it arrived; those two
/// are the only reasons an event is ever rebuilt, and they are the outbound
/// half of the same allowlist discipline the headers take:
///
/// - **`refund`** — wallet material this app consumes, which may not travel.
/// - **`model`** — the caller asked for `<model>@<backend>` and the wire
///   request carries the backend's own spelling, so a chunk answers with the
///   *unqualified* id. The blocking transport rewrites that field for exactly
///   this reason: a tool that stores what came back and reuses it names a model
///   with no backend, which resolves to the default one — a request answered by
///   a different backend than the one it was answered by last time, and, where
///   two backends serve the same name, a display that conflates them. The field
///   is rewritten only where the payload already has one, so nothing is added
///   to an event that never claimed a model.
///
/// **And the token is taken, not merely dropped.** The Eidola server closes a
/// stream with a metadata event (`object == "eidola.chat.completion.metadata"`)
/// carrying the refund, immediately before `[DONE]` — and it sends that event
/// *even when its own best-effort persistence of the token failed*, which is
/// exactly the case where the recovery endpoint can never answer. Discarding
/// the token and asking recovery for it would then strand the credential in
/// `spending` for good, with the one usable copy having arrived in-band and
/// been thrown away. So the filter hands it back and recovery becomes the
/// fallback for its *absence* — the order the blocking path already keeps.
/// The object handed back is `RefundInfo` (`{refund, issuer_key_id}`), the
/// same shape [`process_refund`] reads from a blocking body.
fn forward_sse_event(event: &[u8], canonical: &str) -> (Vec<u8>, Option<Value>) {
    let Ok(text) = std::str::from_utf8(event) else {
        return (event.to_vec(), None);
    };
    // **Fields are split by the same scanner that split the events.** An event
    // is a run of lines, and `str::lines()` knows only two of the format's
    // three line endings — so `id: 1\rdata: {…}` arrived as one line with no
    // `data:` prefix and its payload was invisible: no model rewrite, and no
    // refund found in the metadata event that settles the credential.
    let lines = crate::split_event_lines(text);
    // **And the event's data is assembled before it is parsed** — one value,
    // the `data:` fields joined with `\n`, which is what the format says an
    // event carries (`sse_event_data`). Parsing field by field read a payload
    // the sender split across lines as fragments that are each not JSON, and
    // the event went out untouched: no model rewrite, no refund.
    let Some(mut value) = crate::sse_event_data(text)
        .and_then(|data| serde_json::from_str::<Value>(data.trim_start()).ok())
    else {
        return (event.to_vec(), None);
    };
    let refund = value.get("refund").cloned();
    let misnames = value
        .get("model")
        .and_then(Value::as_str)
        .is_some_and(|model| model != canonical);
    if refund.is_none() && !misnames {
        return (event.to_vec(), None);
    }
    if let Some(object) = value.as_object_mut() {
        object.remove("refund");
        if object.contains_key("model") {
            object.insert("model".to_string(), Value::String(canonical.to_string()));
        }
    }
    // Rebuilt line by line with the terminator each line actually carried, so
    // an event this app rewrites is framed the way the sender framed it. The
    // rewritten value takes the place of the **first** data field and the rest
    // are dropped: it is one value, and re-serialised JSON has no newline in it
    // to split across fields again. Every other field keeps its place.
    let mut out = String::with_capacity(text.len());
    let mut written = false;
    for (line, terminator) in &lines {
        if crate::sse_data_field(line).is_some() {
            if written {
                continue;
            }
            written = true;
            out.push_str("data: ");
            out.push_str(&value.to_string());
        } else {
            out.push_str(line);
        }
        out.push_str(terminator);
    }
    (out.into_bytes(), refund)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recorded::RECORD_BODY_MAX_BYTES;

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
            h.materialize()
                .pairs()
                .iter()
                .map(|(name, _)| *name)
                .collect()
        };
        assert_eq!(
            names(&headers(None, false, TraceUpstream::Off)),
            vec!["Content-Type", "Accept"],
            "a local route carries the body's own type and what it will take back"
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
            !off.materialize()
                .pairs()
                .iter()
                .any(|(name, _)| *name == "traceparent"),
            "a downstream tool cannot ask this app to trace"
        );

        let on = headers(None, false, TraceUpstream::On);
        let trace_of = |sent: &SentHeaders| -> String {
            sent.pairs()
                .iter()
                .find(|(name, _)| *name == "traceparent")
                .expect("traceparent")
                .1
                .clone()
        };
        let attempt = on.materialize();
        let first = trace_of(&attempt);
        let second = trace_of(&on.materialize());
        assert_ne!(
            first, second,
            "each attempt mints its own id — a retry that reused one would link the two"
        );
        // **And within one attempt the wire and the Record agree.** The id is
        // minted per materialization, so a second `to_pairs` for the row's sake
        // would record a trace that had never travelled and could correlate
        // with nothing.
        assert!(
            attempt.for_record().contains(&first),
            "the row shows the id this attempt actually sent: {}",
            attempt.for_record()
        );
        assert_eq!(
            trace_of(&attempt),
            first,
            "and the set is a value, not a recipe"
        );
        assert!(
            first.starts_with("00-") && first.ends_with("-01"),
            "{first}"
        );
        assert_eq!(first.len(), 55, "version-traceid-spanid-flags");
    }

    #[test]
    fn the_record_shows_the_names_and_never_the_credential() {
        let recorded = headers(Some("Bearer super-secret"), true, TraceUpstream::Off)
            .materialize()
            .for_record();
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
            "stream_options": {"include_usage": true, "something_else": 1},
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
        let upstream = Inner::proxy_request_body(&request, "m", 128, true);
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
        assert_eq!(
            object["stream_options"],
            serde_json::json!({"include_usage": true}),
            "the one option the outer shape expresses, rebuilt rather than \
             forwarded — `something_else` is a key nobody decided to send"
        );
    }

    /// **Asking for usage is the caller's decision on every route kind.**
    ///
    /// It used to be `stream && !bills`, which is a fact about billing standing
    /// in for a fact about the caller — and it swept up external OpenAI
    /// backends, which carry no pricing either, adding a field the downstream
    /// never sent to a server that may not implement it.
    #[test]
    fn a_usage_chunk_is_asked_for_only_when_the_caller_asked() {
        let ask = |stream_options: Value, stream: bool| {
            let mut body = serde_json::json!({
                "model": "m@acme",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": stream,
            });
            if !stream_options.is_null() {
                body["stream_options"] = stream_options;
            }
            let request = ProxyChatRequest::from_json(&body).expect("read");
            Inner::proxy_request_body(&request, "m", 128, stream)
                .get("stream_options")
                .cloned()
        };
        assert_eq!(ask(Value::Null, true), None, "nothing asked, nothing added");
        assert_eq!(
            ask(serde_json::json!({"include_usage": false}), true),
            None,
            "and `false` is an answer, not an absence"
        );
        assert_eq!(
            ask(serde_json::json!({"include_usage": true}), true),
            Some(serde_json::json!({"include_usage": true})),
            "asked for, so sent"
        );
        assert_eq!(
            ask(serde_json::json!({"include_usage": true}), false),
            None,
            "the option means nothing off a stream, so it never travels on one"
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

    /// The floor is the turn path's, and a floor has two edges: what it lets
    /// through matters as much as what it stops. Everything the app's own chat
    /// reads without complaint is accepted here, or the proxy would be holding
    /// a policy the rest of the app does not.
    #[test]
    fn a_completion_is_an_object_with_choices_and_nothing_stricter() {
        assert!(is_completion(&serde_json::json!({"choices": []})), "empty");
        assert!(
            is_completion(&serde_json::json!({"choices": [{"message": {}}], "extra": 1})),
            "a choice with no content, and fields this app does not know"
        );
        assert!(!is_completion(&serde_json::json!({})));
        assert!(!is_completion(&serde_json::json!(null)));
        assert!(
            !is_completion(&serde_json::json!({"error": {"message": "no"}})),
            "an error document answered with a success status"
        );
        assert!(
            !is_completion(&serde_json::json!({"choices": {"message": {}}})),
            "`choices` that is not an array"
        );
        assert!(
            !is_completion(&serde_json::json!([{"choices": []}])),
            "array"
        );
    }

    #[test]
    fn an_sse_event_is_forwarded_byte_for_byte() {
        let event = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}";
        let (out, refund) = forward_sse_event(event, "m@acme");
        assert_eq!(
            out,
            event.to_vec(),
            "nothing this app does may alter what the model said"
        );
        assert!(refund.is_none());
        assert_eq!(
            forward_sse_event(b"data: [DONE]", "m@acme").0,
            b"data: [DONE]".to_vec()
        );
        // A non-UTF-8 event still travels.
        assert_eq!(
            forward_sse_event(&[0xff, 0xfe], "m@acme").0,
            vec![0xff, 0xfe]
        );
        // And an event already naming the model the caller asked for is left
        // exactly as it arrived — the rewrite is a correction, not a pass.
        let named = br#"data: {"model":"m@acme","choices":[]}"#;
        assert_eq!(forward_sse_event(named, "m@acme").0, named.to_vec());
    }

    /// **A chunk names the model the caller asked for.** The caller named
    /// `<model>@<backend>` and the wire request carries the backend's own
    /// spelling, so a chunk answers with the unqualified id — which a tool that
    /// stores and reuses it turns into a request for the *default* backend, and
    /// which conflates two backends serving one name in any display. The
    /// blocking transport rewrites that field; this is the streaming twin.
    #[test]
    fn a_streamed_chunk_names_the_model_the_caller_asked_for() {
        let (out, refund) = forward_sse_event(
            br#"data: {"id":"c1","model":"m","choices":[{"delta":{"content":"hi"}}]}"#,
            "m@acme",
        );
        let text = String::from_utf8(out).expect("utf-8");
        assert!(text.contains(r#""model":"m@acme""#), "{text}");
        assert!(
            text.contains(r#""content":"hi""#),
            "what the model said is untouched: {text}"
        );
        assert!(refund.is_none());

        // Nothing is *added*: an event that never claimed a model does not
        // acquire one, and `[DONE]` is not JSON at all.
        let bare = br#"data: {"choices":[{"delta":{"content":"hi"}}]}"#;
        assert_eq!(forward_sse_event(bare, "m@acme").0, bare.to_vec());

        // And the two rewrites compose on one event: the terminal metadata
        // carries a refund, and a chunk carrying both loses one and gains the
        // other.
        let (out, refund) = forward_sse_event(
            br#"data: {"model":"m","refund":{"refund":"material","issuer_key_id":"ab"}}"#,
            "m@acme",
        );
        let text = String::from_utf8(out).expect("utf-8");
        assert!(!text.contains("material"), "{text}");
        assert!(text.contains(r#""model":"m@acme""#), "{text}");
        assert_eq!(refund.expect("the refund")["refund"], "material");
    }

    #[test]
    fn an_sse_event_carrying_a_refund_yields_it_and_loses_it() {
        // The server's real terminal event: Eidola's own metadata, with the
        // refund nested as `RefundInfo` — the shape `process_refund` reads.
        let (out, refund) = forward_sse_event(
            br#"data: {"object":"eidola.chat.completion.metadata","id":"x","refund":{"refund":"credential-material","issuer_key_id":"ab"}}"#,
            "m@eidola",
        );
        let text = String::from_utf8(out).expect("utf-8");
        assert!(!text.contains("credential-material"), "{text}");
        assert!(text.contains("\"id\":\"x\""), "{text}");
        assert!(text.starts_with("data: "), "{text}");

        // **And the token is handed back rather than dropped** — the server
        // sends it even when its own persistence failed, which is the one case
        // where recovery can never answer.
        let refund = refund.expect("the refund the event carried");
        assert_eq!(refund["refund"], "credential-material");
        assert_eq!(refund["issuer_key_id"], "ab");
    }

    /// **The cap bounds what is held, not only what is written.** Sealing
    /// alone would keep the Record row honest while the process still carried
    /// the whole answer in memory for the length of the stream — which is the
    /// half a caller naming its own ceiling against an external backend can
    /// spend. So the ceiling is enforced as the bytes arrive, and the seal is
    /// what makes the result honest about it.
    #[test]
    fn a_recorded_body_never_holds_more_than_its_cap() {
        let mut body = RecordedBody::default();
        let chunk = vec![b'x'; 64 * 1024];
        for _ in 0..40 {
            body.push(&chunk);
            assert!(
                body.kept.len() <= RECORD_BODY_MAX_BYTES,
                "retention is bounded as the stream runs, not at the end: {} bytes",
                body.kept.len()
            );
        }
        let received = body.received;
        assert_eq!(received, 40 * chunk.len(), "and it counts what really came");

        let sealed = body.seal_stream(StreamDelivery::Complete, true);
        let text = String::from_utf8_lossy(&sealed);
        assert!(
            text.contains(&format!("{received}-byte response")),
            "a partial says how much it is not: {}",
            &text[text.len().saturating_sub(200)..]
        );
        // The note is the only thing past the cap, and it names itself.
        assert!(sealed.len() > RECORD_BODY_MAX_BYTES);
        assert!(sealed.len() < RECORD_BODY_MAX_BYTES + 1024);
    }

    /// A one-shot loopback HTTP server answering with `body`. The reads this
    /// module bounds happen on a `reqwest::Response`, so the only honest way to
    /// hold the bound is against a real one.
    async fn serving_once(body: Vec<u8>) -> std::net::SocketAddr {
        use tokio::io::AsyncWriteExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            use tokio::io::AsyncReadExt;

            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            // Read the request first: answering over unread bytes and closing
            // resets the connection, which the client meets as a read failure.
            let mut request = Vec::new();
            let mut byte = [0u8; 1];
            while !request.ends_with(b"\r\n\r\n") {
                match socket.read(&mut byte).await {
                    Ok(1) => request.push(byte[0]),
                    _ => break,
                }
            }
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(head.as_bytes()).await;
            let _ = socket.write_all(&body).await;
            let _ = socket.shutdown().await;
        });
        addr
    }

    /// **The ceiling bounds what the read holds, not only what is written
    /// down.** The retention cap keeps a `request` row honest while the process
    /// still buffers the whole answer — which is the half a caller naming its
    /// own ceiling against a backend that declares none can spend. So the read
    /// stops as the bytes arrive, and the Record says that is why it stopped.
    #[tokio::test]
    async fn a_blocking_answer_is_bounded_as_it_arrives() {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let client = reqwest::Client::builder().build().expect("client");

        let addr = serving_once(vec![b'x'; MAX_RESPONSE_BYTES + 4096]).await;
        let response = client
            .get(format!("http://{addr}/"))
            .send()
            .await
            .expect("send");
        let answer = read_capped_body(response).await.expect("read");
        assert!(
            answer.bytes.len() <= MAX_RESPONSE_BYTES,
            "the read is bounded while it runs, not after: {} bytes",
            answer.bytes.len()
        );
        assert!(answer.over_ceiling, "and it knows why it stopped");
        let row = String::from_utf8_lossy(&recorded_answer(&answer)).to_string();
        assert!(
            row.contains("keeps the first"),
            "the retention cap still speaks"
        );
        assert!(
            row.contains("Nothing past the ceiling was kept"),
            "and a read this app ended is not an upstream that finished"
        );
        // **One boundary on the row.** The crossing chunk is cut at the
        // ceiling and its remainder dropped unexamined, so the count every
        // note states is the ceiling itself — never a larger "received" figure
        // beside a note saying nothing past the ceiling was read.
        assert!(
            row.contains(&format!("of the {MAX_RESPONSE_BYTES} bytes this app read")),
            "the cap's count is the ceiling's: {}",
            &row[row.len().saturating_sub(600)..]
        );
        assert!(
            !row.contains("never read"),
            "and no note claims more than it knows"
        );
        // **And the row never names a prefix as the size.** The read stopped
        // at the ceiling, so what it counted is how far it got: the cap's note
        // states a lower bound, and nothing on the row says the response was
        // that many bytes — or that everything above is all that was received.
        assert!(
            row.contains("at least that large"),
            "the count is stated as the lower bound it is: {}",
            &row[row.len().saturating_sub(600)..]
        );
        assert!(
            !row.contains(&format!("{}-byte response", answer.bytes.len())),
            "a prefix is not reported as the response's size"
        );
        assert!(
            !row.contains("everything this app received"),
            "and the ceiling's note claims nothing about what the cap kept"
        );

        // An ordinary answer is read whole and claims nothing.
        let addr = serving_once(br#"{"ok":true}"#.to_vec()).await;
        let response = client
            .get(format!("http://{addr}/"))
            .send()
            .await
            .expect("send");
        let answer = read_capped_body(response).await.expect("read");
        assert_eq!(answer.text(), r#"{"ok":true}"#);
        assert!(!answer.over_ceiling);
        assert_eq!(recorded_answer(&answer), br#"{"ok":true}"#.to_vec());
    }

    /// **Every send downstream updates the ending the Record will state.**
    /// The delivery state is what `stream_delivery` reads, so a send whose
    /// result is discarded seals a delivery that did not happen as a complete
    /// one — which is what the unterminated final event did, the send most
    /// likely of all to meet a caller who has already gone.
    #[tokio::test]
    async fn a_send_to_a_caller_that_has_gone_is_not_a_delivery() {
        let (sender, receiver) = tokio::sync::mpsc::channel::<ProxyStreamEvent>(4);
        let mut downstream_gone = false;

        deliver(
            &sender,
            ProxyStreamEvent::Chunk(b"data: hi\n\n".to_vec()),
            &mut downstream_gone,
        )
        .await;
        assert!(!downstream_gone, "a caller that is there receives it");

        drop(receiver);
        deliver(
            &sender,
            ProxyStreamEvent::Chunk(b"data: tail".to_vec()),
            &mut downstream_gone,
        )
        .await;
        assert!(
            downstream_gone,
            "a send that could not land is not a delivery"
        );
        assert_eq!(
            stream_delivery(downstream_gone, false),
            StreamDelivery::CallerGoneReadOn,
            "and the Record says so rather than calling it complete"
        );
    }

    /// **A caller that goes while the upstream is silent still ends the read.**
    /// The departure is otherwise noticed only at a send, and a backend that
    /// answers its head and then says nothing never produces one — so the read
    /// held the upstream connection, and any engine lease behind it, for as
    /// long as that backend cared to stay quiet, with nobody left to deliver a
    /// byte to.
    #[tokio::test]
    async fn a_caller_that_goes_while_the_upstream_is_silent_ends_the_read() {
        use std::time::Duration;

        let (sender, receiver) = tokio::sync::mpsc::channel::<ProxyStreamEvent>(1);
        drop(receiver);
        let mut silent = futures_util::stream::pending::<Vec<u8>>();

        // The wait is bounded here because the defect's own shape is a wait
        // that never ends: without the cure this fails rather than hangs, and
        // a hang says nothing.
        let ended = tokio::time::timeout(
            Duration::from_secs(5),
            next_chunk(&mut silent, &sender, false),
        )
        .await
        .expect("the read ends with the caller rather than waiting out a silent upstream");
        assert!(matches!(ended, NextChunk::CallerGone));

        // A route with a hold to settle waits anyway: its refund is at the end
        // of the stream, and delivery is what a vanished caller loses.
        let settling = tokio::time::timeout(
            Duration::from_millis(50),
            next_chunk(&mut silent, &sender, true),
        )
        .await;
        assert!(
            settling.is_err(),
            "a spending route reads on to the refund in the tail"
        );

        // And the upstream's own two endings are unchanged, caller or no.
        let mut done = futures_util::stream::empty::<Vec<u8>>();
        assert!(matches!(
            next_chunk(&mut done, &sender, false).await,
            NextChunk::UpstreamEnded
        ));
        let mut one = futures_util::stream::iter(vec![vec![1u8]]);
        assert!(
            matches!(
                next_chunk(&mut one, &sender, false).await,
                NextChunk::Chunk(_)
            ),
            "a chunk in hand is preferred to noticing the departure"
        );
    }

    /// **The cap was the first face of "a partial must never claim to be
    /// whole"; the ending is the second.** Neither of the two ways a stream
    /// stops short is visible in `received` versus `kept`: an upstream read
    /// ended on purpose looks byte-for-byte like an upstream that finished, and
    /// a delivery that stopped short looks like one that did not. Recorded with
    /// no marker, a deliberate partial reads as a complete answer — and on the
    /// zero-spend exit that is exactly what happened, since the read ends the
    /// moment the caller does and `received == kept`.
    #[test]
    fn a_stream_that_stopped_short_says_which_way_it_stopped() {
        assert_eq!(stream_delivery(false, false), StreamDelivery::Complete);
        assert_eq!(
            stream_delivery(true, false),
            StreamDelivery::CallerGoneReadOn,
            "the caller went, the read went on to the refund in the tail"
        );
        assert_eq!(
            stream_delivery(true, true),
            StreamDelivery::CallerGoneReadEnded,
            "nothing to settle, so the read ended with the caller"
        );

        // A complete delivery adds nothing; either partial names itself.
        let whole = |ending| {
            let mut body = RecordedBody::default();
            body.push(
                b"data: hello

",
            );
            String::from_utf8(
                body.seal_stream(ending, ending != StreamDelivery::CallerGoneReadEnded),
            )
            .expect("utf-8")
        };
        assert_eq!(
            whole(StreamDelivery::Complete),
            "data: hello

"
        );
        assert!(
            whole(StreamDelivery::CallerGoneReadEnded).contains("stops where this app stopped"),
            "a read this app ended is not the upstream's ending"
        );
        assert!(
            whole(StreamDelivery::CallerGoneReadOn).contains("is not what was delivered"),
            "a whole upstream answer nobody received says so"
        );

        // And the two markers compose: an over-cap body that was also cut
        // short carries both facts.
        let mut big = RecordedBody::default();
        big.push(&vec![b'x'; RECORD_BODY_MAX_BYTES + 4096]);
        let big_received = big.received;
        let sealed =
            String::from_utf8_lossy(&big.seal_stream(StreamDelivery::CallerGoneReadEnded, false))
                .to_string();
        assert!(sealed.contains("keeps the first"), "the cap still speaks");
        assert!(
            sealed.contains("stops where this app stopped"),
            "and so does the ending"
        );
        assert!(
            sealed.contains("at least that large")
                && !sealed.contains(&format!("{big_received}-byte response")),
            "a read that ended early states its count as a lower bound: {sealed}"
        );

        // **And the two never contradict each other.** The cap note claims
        // only what the cap knows — the rest was received — so a row whose
        // ending says the caller went away is not also told the bytes reached
        // them. The caller-gone-but-read-on ending is the sharpest case: the
        // refund's drain is exactly a read past a caller who has gone.
        let mut drained = RecordedBody::default();
        drained.push(&vec![b'x'; RECORD_BODY_MAX_BYTES + 4096]);
        let sealed =
            String::from_utf8_lossy(&drained.seal_stream(StreamDelivery::CallerGoneReadOn, true))
                .to_string();
        assert!(
            sealed.contains("was received and not retained"),
            "the cap says what it knows: {sealed}"
        );
        assert!(
            !sealed.contains("delivered and not retained"),
            "and never that bytes the ending says nobody received were delivered: {sealed}"
        );
    }

    /// **`null` is not a cap, so the legacy spelling still counts.** SDKs emit
    /// `null` for an unset field; reading it as present discarded the caller's
    /// `max_tokens` for the 4096 default — more output, and more billed, than
    /// asked for. Each spelling is converted before the fallback is taken.
    #[test]
    fn a_null_completion_cap_falls_back_to_the_legacy_one() {
        let cap = |body: Value| {
            ProxyChatRequest::from_json(&body)
                .expect("read")
                .max_completion_tokens
        };
        let base = |extra: Value| {
            let mut body = serde_json::json!({"model": "m", "messages": []});
            body.as_object_mut()
                .expect("object")
                .extend(extra.as_object().expect("object").clone());
            body
        };
        assert_eq!(
            cap(base(
                serde_json::json!({"max_completion_tokens": null, "max_tokens": 128})
            )),
            Some(128),
            "null first, the legacy cap behind it"
        );
        assert_eq!(
            cap(base(
                serde_json::json!({"max_tokens": null, "max_completion_tokens": 64})
            )),
            Some(64),
            "the current spelling wins wherever it is a number"
        );
        assert_eq!(
            cap(base(
                serde_json::json!({"max_completion_tokens": 64, "max_tokens": 128})
            )),
            Some(64),
            "and outranks the legacy one when both are set"
        );
        assert_eq!(
            cap(base(
                serde_json::json!({"max_completion_tokens": null, "max_tokens": null})
            )),
            None,
            "both null is no cap at all, so the budget's default applies"
        );
    }

    #[test]
    fn a_response_that_is_not_server_sent_events_is_not_taken_for_one() {
        use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};

        let with = |value: &str| {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_str(value).expect("header"));
            is_event_stream(&headers)
        };
        assert!(with("text/event-stream"));
        assert!(
            with("text/event-stream; charset=utf-8"),
            "parameters are the sender's business"
        );
        assert!(
            with("Text/Event-Stream"),
            "the media type is case-insensitive"
        );
        // The two shapes this exists for.
        assert!(
            !with("application/json"),
            "a backend that ignored `stream: true`"
        );
        assert!(!with("text/html"), "an intermediary's error page");
        // Permissive where the rule cannot help: an unlabelled body is left
        // alone rather than refused, since a compliant SSE server always sets
        // the header and the cases above both name one.
        assert!(is_event_stream(&HeaderMap::new()));
    }

    #[test]
    fn a_backend_that_will_not_start_an_engine_offers_only_what_runs() {
        // The managed `local` singleton always starts on demand, so the
        // exposure setting is the whole answer for it.
        assert!(offers_running_engines_only(LocalExposure::Loaded, true));
        assert!(!offers_running_engines_only(
            LocalExposure::Downloaded,
            true
        ));
        // A `llamacpp` backend with auto-start off refuses the load, so
        // "all downloaded" cannot mean "all downloaded" there — advertising
        // them would promise what the route deterministically refuses.
        assert!(offers_running_engines_only(LocalExposure::Loaded, false));
        assert!(
            offers_running_engines_only(LocalExposure::Downloaded, false),
            "exposure is a permission over a backend, not a licence to break its own rule"
        );
    }

    /// The route the flush is about: an eidola-shaped one whose observer has
    /// captured a handshake the listing never wrote a row for.
    fn route_with_pending_attestation(provider_id: &str, base_url: &str) -> ProxyRoute {
        // The route holds a client it never uses here; building one needs the
        // provider `AppCore::new` installs in production. Idempotent.
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let log: Arc<Mutex<Vec<tinfoil_verifier::VerifiedAttestation>>> =
            Arc::new(Mutex::new(Vec::new()));
        log.lock()
            .unwrap()
            .push(tinfoil_verifier::VerifiedAttestation {
                platform: tinfoil_verifier::Platform::SevSnp,
                matched_measurement: tinfoil_verifier::MatchedMeasurement::SevSnp("a".repeat(96)),
                attestation_hash: "second-handshake".into(),
                attestation_doc: b"report".to_vec(),
                pcr_digest: "b".repeat(96),
                peer_spki_hash: "c".repeat(64),
            });
        ProxyRoute {
            client: reqwest::Client::builder()
                .build()
                .expect("a client with no TLS work to do"),
            base_url: base_url.to_string(),
            wire_model: "m".into(),
            canonical: "m".into(),
            backend_id: "eidola".into(),
            pricing: None,
            external_auth: None,
            // What the *listing's* handshake wrote.
            connection_id: Some("listing-connection".into()),
            attestations: Some(AttestationSink {
                log,
                provider_id: provider_id.to_string(),
                base_url: base_url.to_string(),
            }),
            declared_max_output: None,
            engine_lease: None,
        }
    }

    #[tokio::test]
    async fn a_second_handshake_is_recorded_and_the_request_follows_it() {
        // **The completion can ride a connection the catalog fetch never
        // opened** — the server is free to close the listing's. That
        // handshake's attestation would otherwise sit in the observer
        // unpersisted while the Record named the listing's connection as the
        // one that carried the prompt.
        let dir = tempfile::tempdir().expect("tempdir");
        let db = crate::db::open(dir.path()).await.expect("open");
        let conn = crate::db::connect(&db).await.expect("connect");
        let provider_id = crate::db::ensure_provider(&conn, "eidola", "inference", 1)
            .await
            .expect("provider");

        let mut route = route_with_pending_attestation(&provider_id, "https://example.invalid");
        assert!(
            route.flush_new_attestations(&conn).await,
            "a captured handshake is a row to write"
        );
        assert_ne!(
            route.connection_id.as_deref(),
            Some("listing-connection"),
            "the request now names the connection that actually carried it"
        );
        let attestations = crate::db::list_attestations(&conn, 10, 0)
            .await
            .expect("attestations");
        assert!(
            attestations.iter().any(|a| a.hash == "second-handshake"),
            "the second handshake is in the Record"
        );

        // Idempotent: nothing captured since means nothing to adopt, and the
        // connection the request is attached to does not move.
        let adopted = route.connection_id.clone();
        assert!(!route.flush_new_attestations(&conn).await);
        assert_eq!(route.connection_id, adopted);
    }

    #[tokio::test]
    async fn a_route_that_verifies_no_enclave_has_nothing_to_flush() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = crate::db::open(dir.path()).await.expect("open");
        let conn = crate::db::connect(&db).await.expect("connect");
        let mut route = route_with_pending_attestation("p", "https://example.invalid");
        route.attestations = None;
        route.connection_id = None;
        assert!(
            !route.flush_new_attestations(&conn).await,
            "an engine-backed or external route never has a handshake to record"
        );
        assert_eq!(route.connection_id, None);
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
