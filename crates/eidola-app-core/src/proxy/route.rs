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

/// The most of one response body a `request` row keeps.
///
/// **A bound the Record needs and the delivery does not.** What travels
/// downstream is streamed through and forgotten; what is *retained* is retained
/// per connection until the stream ends, so a caller free to name any ceiling
/// against any exposed backend could otherwise make one request cost this
/// process the whole answer in memory — and then the same bytes again in the
/// database. A megabyte is far past any answer a person reads and far short of
/// a size worth holding: an SSE transcript of a 4k-token completion is tens of
/// kilobytes.
///
/// The turn path's own `response_buf` has no such bound today. It is the same
/// shape and a different exposure — the app issues those requests itself, one
/// per turn, against a ceiling it set — so it is recorded as a known twin
/// rather than changed from here.
const RECORD_BODY_MAX_BYTES: usize = 1 << 20;

/// The most one server-sent event may accumulate before it is refused.
///
/// **The third buffer, and the one two ceilings did not cover.** A stream is
/// bounded in what it *retains* (`RECORD_BODY_MAX_BYTES`) and in what it may
/// queue for the caller (`STREAM_QUEUE_EVENTS`), and neither bounds the frame
/// accumulator: bytes go into `buf` and only come out when
/// [`find_event_boundary`] finds the blank line that ends an event. A backend
/// that sends one enormous event — or never sends a terminator at all — grows
/// that buffer until the process is out of memory, with the Record's cap and
/// the queue both looking perfectly healthy.
///
/// A megabyte is orders of magnitude past any real event: a completion chunk is
/// hundreds of bytes and the terminal metadata event carrying a refund is a few
/// kilobytes. What passes it is not an event this app can forward, so the
/// stream ends and the Record says why.
const MAX_SSE_EVENT_BYTES: usize = 1 << 20;

/// The most of one upstream answer this app will read at all.
///
/// **The retention cap's other half, and the one the process feels.**
/// [`RECORD_BODY_MAX_BYTES`] bounds what a `request` row keeps; it cannot bound
/// what reading the answer costs, because a blocking transport buffers the
/// whole body before anything can seal it. An exposed backend declares no
/// ceiling of its own and a downstream caller names its own `max_tokens`, so
/// one request against a backend that answers without end — or an intermediary
/// streaming an enormous error page — would cost this process the whole answer
/// in memory, times however many callers the listener admits.
///
/// Eight megabytes is far past any completion a model produces (a 128k-token
/// answer is around half a megabyte) and far short of a size worth holding. A
/// body that reaches it is not an answer this app can use — it will not parse —
/// so the read stops there, the exchange is recorded as the truncation it is,
/// and the caller gets the gateway failure.
const MAX_RESPONSE_BYTES: usize = 8 << 20;

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
/// number, never the mechanism.
static MODEL_LIST_TIMEOUT_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn model_list_timeout() -> std::time::Duration {
    match MODEL_LIST_TIMEOUT_MS.load(std::sync::atomic::Ordering::Relaxed) {
        0 => MODEL_LIST_TIMEOUT,
        ms => std::time::Duration::from_millis(ms),
    }
}

/// Test-only: shorten the per-backend catalog deadline. `0` restores it.
#[doc(hidden)]
pub fn set_model_list_timeout_for_test(millis: u64) {
    MODEL_LIST_TIMEOUT_MS.store(millis, std::sync::atomic::Ordering::Relaxed);
}

/// A response body on its way to the Record, kept to [`RECORD_BODY_MAX_BYTES`].
///
/// **Truncation is recorded as truncation.** A Record row holding the first
/// megabyte of a larger answer and saying nothing would be a partial claiming
/// to be whole, which is the one thing this trail may never be — a reader opens
/// it to see what left this machine. So the seal states both numbers, in the
/// payload itself, in a form no upstream could have sent by accident.
#[derive(Default)]
struct RecordedBody {
    kept: Vec<u8>,
    received: usize,
}

impl RecordedBody {
    fn push(&mut self, bytes: &[u8]) {
        self.received += bytes.len();
        let room = RECORD_BODY_MAX_BYTES.saturating_sub(self.kept.len());
        if room > 0 {
            self.kept.extend_from_slice(&bytes[..room.min(bytes.len())]);
        }
    }

    /// The bytes to record for a stream, with the note when they are not all
    /// of them and the note for how the stream ended.
    ///
    /// **Neither transport has an unqualified `seal` to reach for**: an ending
    /// is not optional information about a body, and a blocking read has an
    /// ending of its own now that it is bounded — see [`RecordedBody::seal_blocking`].
    fn seal_stream(self, delivery: StreamDelivery) -> Vec<u8> {
        let mut out = seal_recorded_body(self.kept, self.received);
        if let Some(note) = delivery.note() {
            out.extend_from_slice(note.as_bytes());
        }
        out
    }

    /// The bytes to record for a blocking answer, with the note when the
    /// ceiling — not the upstream — is why the read stopped.
    fn seal_blocking(self, read: BodyRead) -> Vec<u8> {
        let mut out = seal_recorded_body(self.kept, self.received);
        if let Some(note) = read.note() {
            out.extend_from_slice(note.as_bytes());
        }
        out
    }
}

/// How a blocking answer's read ended.
///
/// The same rule the stream's [`StreamDelivery`] states, on the transport that
/// now needs it for the same reason: a read this app ended at its own ceiling
/// looks byte-for-byte like an upstream that finished, so a row carrying no
/// marker would be a partial claiming to be whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BodyRead {
    /// The upstream closed on its own and everything it sent was read.
    Complete,
    /// [`MAX_RESPONSE_BYTES`] stopped the read. What is recorded stops where
    /// this app stopped reading, not where the upstream stopped sending.
    CeilingReached,
}

impl BodyRead {
    fn note(self) -> Option<String> {
        match self {
            BodyRead::Complete => None,
            BodyRead::CeilingReached => Some(format!(
                "\n\n[eidola: the read stopped at the {MAX_RESPONSE_BYTES}-byte ceiling this app \
                 holds for one answer. What is above is everything this app received; the rest \
                 was never read, and no answer was passed on.]\n"
            )),
        }
    }
}

/// One blocking upstream answer, read under [`MAX_RESPONSE_BYTES`].
type CappedBody = crate::peer_read::BoundedBody;

/// What the Record keeps of a blocking answer, stating both the retention cap
/// and the ceiling where either applied.
fn recorded(answer: &CappedBody) -> Vec<u8> {
    let mut kept = RecordedBody::default();
    kept.push(&answer.bytes);
    // The ceiling can stop the read part-way through a chunk, so more arrived
    // than was kept to parse; the Record states the larger number.
    kept.received = answer.received;
    kept.seal_blocking(if answer.over_ceiling {
        BodyRead::CeilingReached
    } else {
        BodyRead::Complete
    })
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
async fn read_capped_body(response: reqwest::Response) -> Result<CappedBody, AppError> {
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
            StreamDelivery::CallerGoneReadEnded => Some(
                "\n\n[eidola: the caller disconnected and the upstream read ended with it. The \
                 response above stops where this app stopped reading, not where the upstream \
                 stopped sending.]\n",
            ),
        }
    }
}

/// Take a whole body down to what the Record keeps, saying so if it did.
fn seal_recorded_body(mut kept: Vec<u8>, received: usize) -> Vec<u8> {
    if kept.len() > RECORD_BODY_MAX_BYTES {
        kept.truncate(RECORD_BODY_MAX_BYTES);
    }
    if received <= kept.len() {
        return kept;
    }
    let note = format!(
        "\n\n[eidola: this Record entry keeps the first {} bytes of a {}-byte response. The rest \
         was delivered and not retained.]\n",
        kept.len(),
        received
    );
    kept.extend_from_slice(note.as_bytes());
    kept
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

impl UpstreamHeaders {
    /// The headers, in the order they are set. `traceparent` is minted here
    /// rather than carried on the struct, so each call — each *attempt* — gets
    /// its own.
    pub fn to_pairs(&self) -> Vec<(&'static str, String)> {
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
        // **Ready, not merely present**: `reserve_engine` inserts the entry
        // before the subprocess is up and `lease_engine` refuses it until it
        // is, so listing a warming engine would be the same disagreement read
        // the other way round.
        let ready: Vec<local_models::RunningEngine> = self
            .running_engines()
            .into_iter()
            .filter(|engine| engine.ready)
            .collect();
        // The registry rows first — local reads, and what decides how each
        // backend is treated below.
        let mut plans = Vec::new();
        for backend_id in &settings.backends {
            let row = db::get_backend(&self.db_conn().await?, backend_id).await?;
            let engine_backed = row
                .as_ref()
                .and_then(|row| backends::BackendKind::parse(&row.kind))
                .map(|kind| kind.is_engine_backed())
                .unwrap_or(false);
            let starts_on_demand = row.map(|row| row.auto_start).unwrap_or(false);
            plans.push((
                backend_id,
                engine_backed,
                offers_running_engines_only(settings.local_exposure, starts_on_demand),
            ));
        }

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
        let scans = futures_util::future::join_all(plans.iter().map(|(backend_id, _, _)| async {
            tokio::time::timeout(model_list_timeout(), self.backend_models(backend_id))
                .await
                .ok()
                .and_then(|r| r.ok())
        }))
        .await;

        let mut out = Vec::new();
        for ((backend_id, engine_backed, loaded_only), scanned) in plans.iter().zip(scans) {
            let (engine_backed, loaded_only) = (*engine_backed, *loaded_only);
            if engine_backed && loaded_only {
                // The scan is the *decoration* here, not the membership: it
                // supplies a context length and capabilities where it has
                // them, and where it does not the engine still answers for
                // itself. A scan that failed outright therefore blanks nothing.
                for engine in ready.iter().filter(|e| &&e.backend_id == backend_id) {
                    let decorated = scanned
                        .as_ref()
                        .and_then(|models| models.iter().find(|m| m.id == engine.id))
                        .cloned();
                    out.push(decorated.unwrap_or_else(|| engine_model_info(engine)));
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
        let mut route = self.open_proxy_route(&settings, &target).await?;
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
                    Some(error.to_string()),
                    nonce,
                    request_at,
                    now_ms(),
                )
                .await;
                return Err(error);
            }
        };

        let response = match route.client.execute(outbound).await {
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
        // The completion may have opened a connection of its own; the Record
        // must name the one it actually went down.
        if route.flush_new_attestations(&db_conn).await {
            self.bus.emit(Change::Record);
        }
        let status = response.status();
        let answer = match read_capped_body(response).await {
            Ok(answer) => answer,
            Err(error) => {
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
        let text = answer.text();
        // **A body that does not parse is not an answer.** Coercing it to JSON
        // `null` and keeping the upstream's `2xx` would hand a downstream tool
        // an apparent success carrying a fabricated body — a truncated
        // response or an intermediary's HTML error page reads as "the model
        // said nothing". The exchange is still recorded and the hold still
        // settles below; only the *answer* becomes the gateway failure it is.
        // A body the ceiling stopped lands here too, and by the same route: it
        // is a fragment, so it does not parse, so it is not an answer.
        let parsed: Option<Value> = serde_json::from_str(&text).ok();
        // **A refusal this app made belongs in the row it made it about.** The
        // upstream's status is the upstream's claim; a `2xx` this proxy would
        // not accept is recorded as an error too, or the Record shows the
        // exchange as the success the caller was explicitly not given. A
        // non-2xx needs no such column — its status already says what happened.
        let refusal = (status.is_success() && parsed.is_none())
            .then(|| malformed_json_answer(&route.backend_id, status.as_u16()));
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
            recorded(&answer),
            refusal.as_ref().map(ToString::to_string),
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

        // This is the arm `refusal` was built for — the status is a success and
        // the body did not parse — so the caller is told what the row says.
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
        let settings = self.proxy_settings().await?;
        let target = self.resolve_proxy_target(&settings, &request.model).await?;
        let mut route = self.open_proxy_route(&settings, &target).await?;
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
                    Some(error.to_string()),
                    nonce,
                    request_at,
                    now_ms(),
                )
                .await;
                return Err(error);
            }
        };

        let response = match route.client.execute(outbound).await {
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
            let answer = read_capped_body(response).await.unwrap_or_default();
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
            let inline = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|body| body.get("refund").cloned());
            self.settle_proxy_refund(&db_conn, &spend, &auth_value, &route, inline.as_ref())
                .await;
            self.record_proxy_request(
                &route,
                &headers,
                &body,
                Some(status.as_u16()),
                recorded(&answer),
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
        // **And this is a fifth arm of the refund class**: a backend that
        // ignored `stream: true` may well have answered with a whole
        // completion, refund and all — the credential is spent either way, so
        // the body is read for its token before this fails.
        if !is_event_stream(response.headers()) {
            let answer = read_capped_body(response).await.unwrap_or_default();
            let text = answer.text();
            let inline = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|body| body.get("refund").cloned());
            // The refusal is this app's, so the row carries it: the upstream
            // said `200` and nothing was forwarded, which a row holding only
            // that status would present as an answered request.
            let refusal = malformed_stream_answer(&route.backend_id, status.as_u16());
            self.settle_proxy_refund(&db_conn, &spend, &auth_value, &route, inline.as_ref())
                .await;
            self.record_proxy_request(
                &route,
                &headers,
                &body,
                Some(status.as_u16()),
                recorded(&answer),
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
            while let Some(pos) = find_event_boundary(&buf) {
                let event: Vec<u8> = buf.drain(..pos).collect();
                let boundary_len = if buf.starts_with(b"\r\n\r\n") { 4 } else { 2 };
                let terminator: Vec<u8> = buf.drain(..boundary_len.min(buf.len())).collect();
                let (mut out, refund) = forward_sse_event(&event);
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
            // the process dies. Checked after the drain, so what is measured is
            // an event with no boundary in it rather than a chunk that happened
            // to straddle one; the residual is a single transport chunk of
            // overshoot, which is the transport's own bound rather than ours.
            if buf.len() > MAX_SSE_EVENT_BYTES {
                read_error = Some(AppError::Network {
                    message: format!(
                        "`{}` sent a single event past the {MAX_SSE_EVENT_BYTES}-byte ceiling \
                         this app will hold, or never ended one",
                        route.backend_id
                    ),
                });
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
            let (out, refund) = forward_sse_event(&buf);
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
            raw.seal_stream(stream_delivery(downstream_gone, read_ended_early)),
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
fn engine_model_info(engine: &local_models::RunningEngine) -> ModelInfo {
    ModelInfo {
        id: engine.id.clone(),
        context_length: u64::from(engine.context_tokens),
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
    headers: &UpstreamHeaders,
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
    for (name, value) in headers.to_pairs() {
        let name = reqwest::header::HeaderName::from_bytes(name.as_bytes()).map_err(|_| {
            AppError::Config {
                message: format!("`{name}` is not a usable header name"),
            }
        })?;
        let value =
            reqwest::header::HeaderValue::from_str(&value).map_err(|_| AppError::Config {
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

/// The streaming twin: a `2xx` that is not server-sent events.
fn malformed_stream_answer(backend_id: &str, status: u16) -> AppError {
    AppError::Network {
        message: format!(
            "`{backend_id}` answered {status} to a streaming request with a body that is not \
             server-sent events"
        ),
    }
}

/// Whether a response's own headers say it is server-sent events.
///
/// The media type only — parameters (`; charset=utf-8`) are the sender's
/// business — and an **absent** header answers `true`, which is the permissive
/// half of the rule stated at the call site.
fn is_event_stream(headers: &reqwest::header::HeaderMap) -> bool {
    let Some(value) = headers.get(reqwest::header::CONTENT_TYPE) else {
        return true;
    };
    value
        .to_str()
        .map(|v| {
            v.split(';')
                .next()
                .unwrap_or("")
                .trim()
                .eq_ignore_ascii_case("text/event-stream")
        })
        .unwrap_or(false)
}

/// One SSE event on its way downstream, and the refund it was carrying.
///
/// **Byte-identical unless there is a credential in it.** The answer a model
/// gave is not this app's to reformat, so an event whose payloads carry
/// nothing of ours is passed through exactly as it arrived. The one thing that
/// may not travel is a `refund` — wallet material this app consumes — and
/// removing it is the only reason an event is ever rebuilt. This is the
/// outbound half of the same allowlist discipline the headers take.
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
fn forward_sse_event(event: &[u8]) -> (Vec<u8>, Option<Value>) {
    let Ok(text) = std::str::from_utf8(event) else {
        return (event.to_vec(), None);
    };
    let refund = text.lines().find_map(|line| {
        line.trim_end_matches('\r')
            .strip_prefix("data:")
            .map(str::trim_start)
            .and_then(|payload| serde_json::from_str::<Value>(payload).ok())
            .and_then(|value| value.get("refund").cloned())
    });
    if refund.is_none() {
        return (event.to_vec(), None);
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
    (out.into_bytes(), refund)
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
        let (out, refund) = forward_sse_event(event);
        assert_eq!(
            out,
            event.to_vec(),
            "nothing this app does may alter what the model said"
        );
        assert!(refund.is_none());
        assert_eq!(
            forward_sse_event(b"data: [DONE]").0,
            b"data: [DONE]".to_vec()
        );
        // A non-UTF-8 event still travels.
        assert_eq!(forward_sse_event(&[0xff, 0xfe]).0, vec![0xff, 0xfe]);
    }

    #[test]
    fn an_sse_event_carrying_a_refund_yields_it_and_loses_it() {
        // The server's real terminal event: Eidola's own metadata, with the
        // refund nested as `RefundInfo` — the shape `process_refund` reads.
        let (out, refund) = forward_sse_event(
            br#"data: {"object":"eidola.chat.completion.metadata","id":"x","refund":{"refund":"credential-material","issuer_key_id":"ab"}}"#,
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

        let sealed = body.seal_stream(StreamDelivery::Complete);
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
        let row = String::from_utf8_lossy(&recorded(&answer)).to_string();
        assert!(
            row.contains("keeps the first"),
            "the retention cap still speaks"
        );
        assert!(
            row.contains("never read"),
            "and a read this app ended is not an upstream that finished"
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
        assert_eq!(recorded(&answer), br#"{"ok":true}"#.to_vec());
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
            String::from_utf8(body.seal_stream(ending)).expect("utf-8")
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
        let sealed = String::from_utf8_lossy(&big.seal_stream(StreamDelivery::CallerGoneReadEnded))
            .to_string();
        assert!(sealed.contains("keeps the first"), "the cap still speaks");
        assert!(
            sealed.contains("stops where this app stopped"),
            "and so does the ending"
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
