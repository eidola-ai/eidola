//! In-process mock-upstream harness for driving `AppCore::chat` /
//! `chat_stream` (and the account / credential HTTP paths they depend on)
//! against a deterministic, no-network fixture.
//!
//! ## Why this exists
//!
//! `chat` / `chat_stream` need a live upstream: the attesting HTTP client,
//! OpenAI-compatible `/v1/models` + `/v1/chat/completions`, anonymous-credit
//! issuance (`/v1/keys`, `/v1/account/credentials`), and refund recovery
//! (`/v1/credentials/refund`). Two things made these paths untestable in
//! process: (1) the real client performs per-handshake SEV-SNP/TDX enclave
//! attestation over TLS, which no in-process mock can satisfy cheaply; and
//! (2) credentials must be *cryptographically spendable* — a stubbed body
//! won't decode into a `CreditToken` the client can spend.
//!
//! ## How it works
//!
//! * **Attestation bypass.** `AppCore::with_test_http_client` (a `#[doc(hidden)]`
//!   test seam on the production type) injects a plain-HTTP `reqwest::Client`,
//!   so `Inner::build_client` returns it instead of constructing the attesting
//!   client. Tests point `base_url` at this mock over `http://`; no TLS, no
//!   attestation, no shim-mock subprocess. (The alternative — running
//!   `tinfoil-shim-mock` as a TLS subprocess, as
//!   `tinfoil-verifier`'s `mock_attesting_client_e2e` does — is heavier and
//!   still cannot abort an SSE stream mid-event on demand.)
//!
//! * **Real issuance crypto.** The mock holds a freshly generated ACT issuer
//!   `PrivateKey` and reuses the *same* `anonymous-credit-tokens` primitives
//!   the production server (`crates/eidola-server/src/credentials.rs`) uses —
//!   `issue` for `/v1/account/credentials`, `refund` for the inline chat refund
//!   and `/v1/credentials/refund`. The server's issuance handler is glued to
//!   postgres and can't be called as a library, so this reimplements only the
//!   crypto (key gen, request-context derivation, issue/refund), byte-for-byte
//!   matching `credentials.rs`. Credentials minted here decode and spend in the
//!   real client path.
//!
//! * **Raw HTTP/1.1 server.** A bare `tokio::net::TcpListener` (no axum) frames
//!   one request/response per connection so SSE streaming and mid-stream
//!   connection drops are fully under test control. The blocking/streaming chat
//!   behaviour is selected per test via [`ChatBehavior`].

#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anonymous_credit_tokens::{
    IssuanceRequest, Params, PrivateKey, Scalar, SpendProof, credit_to_scalar, scalar_to_credit,
};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use eidola_app_core::AppCore;
use rand_core::OsRng;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

// Domain-separator components: must match the client default
// (`config::DEFAULT_DOMAIN_SEPARATOR`) and the server
// (`credentials.rs`): `ACT-v1:eidola:inference:production:2026-03-05`.
const DS_ORG: &str = "eidola";
const DS_SERVICE: &str = "inference";
const DS_DEPLOYMENT: &str = "production";
const DS_VERSION: &str = "2026-03-05";

const ISSUER_NAME: &str = "eidola";
const ORIGIN_INFO: &str = "inference";

/// The model the mock advertises. Pricing is deliberately simple so the
/// client's integer charge math is easy to reason about: 1 credit per prompt
/// byte and 1 credit per completion-token of the 4096-cap hold.
pub const MODEL: &str = "gemma4-31b";

/// How the mock should respond to `POST /v1/chat/completions`. Selected per
/// test; each maps to a row in the `tests/bus.rs` exit-point table.
#[derive(Clone, Copy, Debug)]
pub enum ChatBehavior {
    /// 200 JSON completion with inline `refund` (blocking happy path).
    OkBlocking,
    /// 200 JSON completion **without** an inline refund (forces the body-refund
    /// fallback to go through `/v1/credentials/refund`).
    OkBlockingNoInlineRefund,
    /// 200 SSE stream: content + reasoning deltas, usage, `[DONE]`.
    OkStreaming,
    /// A plain success in **whichever transport asked** — SSE for a streaming
    /// request, JSON for a blocking one. One behaviour for a test that must
    /// exercise both twins against one upstream, which is otherwise impossible:
    /// `OkBlocking` answers a streamed ask with a body its parser reads as no
    /// content at all.
    OkEitherTransport,
    /// As `OkStreaming`, but the model mimics the per-message header
    /// scaffolding: its content starts with a header-shaped line. Exercises
    /// strip-on-receipt.
    OkStreamingWithHeader,
    /// As `OkStreamingWithHeader`, but the mimicked header is **split across
    /// deltas** (mid-handle, mid-separator, and across the blank line) — what a
    /// real token stream looks like, and what the incremental strip must cope
    /// with.
    OkStreamingWithSplitHeader,
    /// 200 SSE stream that the server aborts mid-event (writes a partial event,
    /// then drops the TCP connection). Exercises the mid-SSE read failure arm.
    StreamingMidAbort,
    /// 200 SSE stream that ends **cleanly** after real content but never says
    /// it is over: well-formed events, a proper end to the chunked body, and
    /// no `[DONE]` and no terminal `finish_reason` anywhere.
    ///
    /// Deliberately not [`ChatBehavior::StreamingMidAbort`], which dies inside
    /// a frame and so fails in the transport's own decoder. Nothing fails here
    /// — the bytes are valid to the last one — which is exactly why the turn
    /// used to persist as a completed answer: the only thing wrong with it is
    /// that the upstream never claimed it was finished.
    StreamingEndsWithoutDone,
    /// The first chat request answers ordinarily; every later one is
    /// [`ChatBehavior::StreamingEndsWithoutDone`]. Whichever transport asked.
    ///
    /// How a test gets a **real answer already in the transcript** and then a
    /// regeneration whose stream dies — the case where reading the cut text as
    /// a completion would supersede the answer it was meant to improve.
    OkThenStreamEndsWithoutDone,
    /// Non-2xx JSON error body (e.g. 500). Exercises the non-2xx arm of both
    /// `chat` and `chat_stream`.
    Non2xx(u16),
    /// Accept the request, then drop the connection before sending any
    /// response bytes (network error after send).
    DropBeforeResponse,
    /// Blocking: the first `n` chat requests answer with a tool call for
    /// [`TOOL_NAME`]; request `n + 1` answers normally. `n = 0` is an ordinary
    /// completion; a large `n` drives the loop into its round cap.
    ToolRoundsBlocking(u64),
    /// SSE twin of [`ChatBehavior::ToolRoundsBlocking`]. The tool call is split
    /// across four deltas (name in two pieces, arguments in two), so the
    /// client's incremental assembly is exercised rather than a single
    /// pre-assembled chunk.
    ToolRoundsStreaming(u64),
    /// Blocking: a tool call with **no `id`** — structurally unusable, so the
    /// turn must fail honestly rather than panic or silently drop it.
    ToolCallMalformed,
    /// Blocking: the first request answers with a tool call whose
    /// `function.arguments` is not valid JSON; the second answers normally.
    /// The bad arguments are a *model* mistake, so the loop reports them back
    /// as a tool error and carries on.
    ToolCallBadArguments,
    /// Blocking: `tool_calls` is present and non-null but **not an array** (an
    /// object here). Structurally unusable — there is no call to execute and
    /// none to persist — so it must fail the turn, not read as "no tools".
    ToolCallsNotAnArray,
    /// SSE: a `delta.tool_calls` that is present and non-null but not an array
    /// (a string here). The streaming twin of
    /// [`ChatBehavior::ToolCallsNotAnArray`].
    ToolCallsNotAnArrayStreaming,
    /// Blocking: an ordinary completion that spells `tool_calls` explicitly as
    /// `null`. Some providers always emit the key; `null` means "no tools" and
    /// must stay an ordinary success.
    ToolCallsNull,
    /// SSE twin of [`ChatBehavior::ToolRoundsStreaming`] whose deltas also
    /// carry **provider-specific** fields, at the tool-call level and inside
    /// `function`, one of them restated with a different value in a later
    /// chunk. Pins that streamed calls preserve provider fields (as blocking
    /// calls do) and that the merge rule is last-wins.
    ToolRoundsStreamingWithExtras(u64),
    /// Blocking: the model answers with a single call to the **decline** tool
    /// (`eidola_app_core::decline::DeclineTool`), carrying [`DECLINE_REASON`].
    /// The agent-side decline checkpoint should end the turn here — no
    /// inference, no post.
    DeclineBlocking,
    /// SSE twin of [`ChatBehavior::DeclineBlocking`] — the streaming path a
    /// driven `respond_stream_as` turn takes.
    DeclineStreaming,
    /// Blocking: reject any request whose body carries a `tools` field with
    /// llama.cpp's own 500 (`"tools param requires --jinja flag"`), and answer
    /// normally otherwise.
    ///
    /// This is a real, verified upstream shape, not an invented one: llama.cpp
    /// returns exactly that 500 without `--jinja`, and *with* `--jinja` returns
    /// a 500 template-render crash when the model's tool block uses Jinja
    /// filters it lacks. Either way the endpoint speaks chat completions
    /// perfectly well and rejects only the `tools` field — which is the case
    /// backend kind cannot predict.
    RejectTools,
    /// As [`ChatBehavior::RejectTools`], but only for the named **wire model**;
    /// every other model on the same host answers a `tools`-bearing request
    /// normally. One backend serves many models with many chat templates, which
    /// is why the client's tool-capability memo is keyed by the pair.
    RejectToolsForModel(&'static str),
    /// Reject a `tools`-bearing request the way **an Eidola server too old to
    /// know the field** does, and answer normally otherwise.
    ///
    /// This is the deployed shape, not an invented one. The server's request
    /// type is `deny_unknown_fields`, so an unknown member fails in the
    /// `LoggedJson` extractor, whose `Rejection` is axum's `JsonRejection`
    /// (`crates/eidola-server/src/handlers.rs`). Axum renders `JsonDataError`
    /// as `(422, String)` — **`text/plain`, not JSON** — and the handler body
    /// never runs, so no `refund` rides the response and the ACT nullifier is
    /// never recorded.
    ///
    /// The distinction from [`ChatBehavior::RejectTools`] is the whole point:
    /// llama.cpp answers a JSON-bodied 500, the old enclave answers a
    /// plain-text 422, and a client that decides "server error vs transport
    /// error" by whether the body parses handles only the first.
    RejectToolsUnparseable,
    /// Blocking: reject a request advertising the turn's **own** navigation
    /// tools (recognized by `list_branches`) with llama.cpp's 500, and answer
    /// every other request with a tool call for [`TOOL_NAME`], forever.
    ///
    /// A *content-dependent* rejection is a real llama.cpp shape: with
    /// `--jinja` it renders each advertised tool's schema and crashes on
    /// filters the model's template lacks, so which tools are advertised
    /// decides the outcome. It is also the only shape that reaches the turn
    /// loop's true worst case — the degrade probe *plus* a full run of nominal
    /// rounds — since the loop only continues if the toolless retry, which
    /// still carries the consumer's own tools, is accepted.
    RejectAutoToolsThenToolRounds,
    /// Blocking: the next chat request answers with **one round requesting
    /// every call currently in [`MockConfig::tool_script`]**, consuming the
    /// script; later requests answer normally.
    ///
    /// Exists for the task-21 navigation tools, whose arguments are post
    /// handles that only exist once the fixture space has been built — so the
    /// script is set at runtime rather than at mock construction.
    ToolScript,
    /// **The reasoning model that never starts answering**, in whichever
    /// transport asked: reasoning all the way to the completion ceiling, no
    /// content at all, `finish_reason: "length"`, and honest usage saying the
    /// whole budget went.
    ///
    /// One behaviour across both transports on purpose — the classification is
    /// a property of the turn, not of how its bytes arrived, and two behaviours
    /// could drift apart while both stayed green.
    ReasoningOnlyLength,
    /// A **partial** answer stopped at the ceiling: real content, then
    /// `finish_reason: "length"`. Whichever transport asked. The turn is
    /// ordinary — the text is worth keeping — but it must not be called
    /// finished.
    PartialAnswerLength,
    /// The first chat request answers ordinarily; every later one is
    /// [`ChatBehavior::ReasoningOnlyLength`]. Whichever transport asked.
    ///
    /// This is how a test gets a *readable answer already in the transcript*
    /// and then a regeneration that hits the ceiling — the reported case, and
    /// the one where the classification decides whether a real answer survives.
    OkThenReasoningOnlyLength,
}

/// The reason the decline behaviours state; asserted on the persisted
/// `decision` action's text block.
pub const DECLINE_REASON: &str = "Nothing to add here.";

/// The decline tool's name — matches `eidola_app_core::decline::DECLINE_TOOL_NAME`.
pub const DECLINE_TOOL: &str = "decline";

/// The tool name the tool-calling behaviours call — matches
/// `eidola_app_core::tools::EchoTool`.
pub const TOOL_NAME: &str = "echo";

/// The engine slug the may-decline router tests register against the mock
/// (`test_register_loaded_local_model("local", ROUTER_SLUG, mock.port())`).
pub const ROUTER_SLUG: &str = "router";

/// The qualified model reference a space's `router_model` is set to in the
/// router tests. Engine-backed backends send the canonical qualified id as the
/// wire model, so this is exactly what the mock sees in the request body — the
/// key it dispatches the router arm on.
pub const ROUTER_MODEL: &str = "router@local";

/// A **remote** (eidola-backend) router model, advertised by the mock's
/// catalog alongside [`MODEL`] with the same trivial pricing. Lets the tests
/// exercise the router's spend path (a remote router bills a normal
/// inference) while still dispatching separately from the turns.
pub const ROUTER_REMOTE_MODEL: &str = "router-remote";

/// The head of `eidola_app_core::summaries::SUMMARY_SYSTEM_PROMPT`. Branch
/// summaries share the router's *model*, so the mock tells the two chores apart
/// by their system prompt, not by the wire model.
pub const SUMMARY_PROMPT_HEAD: &str = "You summarize one branch";

/// How the mock answers a **branch summary** request (see [`SummaryBehavior`]),
/// which is any chat request whose system message starts with
/// [`SUMMARY_PROMPT_HEAD`]. Dispatched ahead of [`RouterBehavior`], so one mock
/// serves turns, the router, and the summarizer in one test.
#[derive(Clone, Debug)]
pub enum SummaryBehavior {
    /// Not summary-aware: answered like any other chat request. The default,
    /// and what every pre-summary test sees.
    Passthrough,
    /// 200 JSON completion whose assistant content is exactly this string.
    Reply(String),
    /// Non-2xx error body — the summarizer is reachable but refuses.
    Fail(u16),
}

/// How the mock answers a chat request for [`ROUTER_MODEL`] — the may-decline
/// router's scripted seam. Every arm keeps the *turn* behaviour
/// ([`ChatBehavior`]) untouched, so one mock serves both roles in one test.
#[derive(Clone, Debug)]
pub enum RouterBehavior {
    /// Not router-aware: a request for the router model is answered like any
    /// other chat request. The default, and what every pre-router test sees.
    Passthrough,
    /// 200 JSON completion whose assistant content is exactly this string —
    /// the scripted routing decision (e.g. `{"notify": [1]}`), or deliberate
    /// garbage for the degrade-on-malformed-output row.
    Reply(String),
    /// Non-2xx error body — the router is reachable but refuses. Exercises
    /// degrade-on-failure.
    Fail(u16),
    /// Send a 200 response head promising a body, then drop the connection
    /// without writing it: the request was **accepted**, so a remote router's
    /// credential hold is already placed, but the body read fails. Exercises
    /// the refund-settlement path on a post-send body-read failure.
    DropBody,
}

/// The `(tool_name, arguments_json)` calls [`ChatBehavior::ToolScript`] asks
/// for. Shared + interior-mutable so a test can fill it in after building its
/// fixture (see [`tool_script`]).
pub type ToolScript = Arc<std::sync::Mutex<Vec<(String, String)>>>;

/// A fresh, empty [`ToolScript`].
pub fn tool_script() -> ToolScript {
    Arc::new(std::sync::Mutex::new(Vec::new()))
}

/// Whether the refund endpoints (inline + recovery) actually mint a successor
/// credential, or fail. Lets refund-recovery-succeeds vs -fails be selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefundMode {
    /// Refund endpoints return a valid successor credential.
    Succeed,
    /// `/v1/credentials/refund` returns 500; inline refunds are omitted.
    Fail,
    /// The first `n` calls to `/v1/credentials/refund` return 500 and every
    /// later one succeeds; inline refunds are omitted throughout.
    ///
    /// A **transient** settlement failure — the case that separates "the hold
    /// could not be settled" from "the hold was settled on a retry". The round
    /// that took the hold fails to settle it and carries on best-effort; the
    /// next `begin_next_round` then settles it on its own last-chance attempt,
    /// which is a durable commit that must announce itself.
    FailFirst(u64),
}

/// Mock upstream configuration.
#[derive(Clone)]
pub struct MockConfig {
    pub chat: ChatBehavior,
    /// How a request for [`ROUTER_MODEL`] is answered (see [`RouterBehavior`]).
    pub router: RouterBehavior,
    /// How a branch-summary request is answered (see [`SummaryBehavior`]).
    pub summary: SummaryBehavior,
    pub refund: RefundMode,
    /// Account available balance returned by `/v1/account/balances`.
    pub balance: i64,
    /// When set, `GET /v1/models` responds with this HTTP status instead of the
    /// pricing catalog — simulating a `prepare_turn` setup failure (the fetch
    /// the original PR #218 screenshot failed on) before the turn's own error
    /// wrapping exists.
    pub models_status: Option<u16>,
    /// The calls [`ChatBehavior::ToolScript`] serves (ignored otherwise).
    pub tool_script: ToolScript,
    /// What `GET /v1/models` declares about [`MODEL`]'s tool calling.
    ///
    /// `None` — the default — omits the `capabilities` object entirely, which
    /// is the shape a server too old to publish capabilities sends and the
    /// shape every backend that cannot declare anything sends. That is what
    /// keeps the client on the learned path, so every test written before
    /// declarations existed keeps exercising exactly what it always did.
    ///
    /// `Some(_)` publishes a leaf. Acting on it also needs
    /// `trust_declared_capabilities_for_test`, since a test necessarily
    /// reaches this mock through a base-URL override and an override is a
    /// hint, never a declaration.
    pub declared_tool_calling: Option<bool>,
    /// How long a chat request is held before it is answered.
    ///
    /// A real model request takes time — that is the whole reason a surface
    /// needs a pending state, and the reason two clicks can overlap. The mock
    /// answers instantly, which makes an in-flight turn unobservable, so a test
    /// about *concurrency* asks for a window it can act inside.
    pub chat_delay_ms: u64,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            chat: ChatBehavior::OkBlocking,
            router: RouterBehavior::Passthrough,
            summary: SummaryBehavior::Passthrough,
            refund: RefundMode::Succeed,
            balance: 10_000_000,
            models_status: None,
            tool_script: tool_script(),
            declared_tool_calling: None,
            chat_delay_ms: 0,
        }
    }
}

/// A running mock upstream. Holds the listening task; dropping it stops the
/// server. `base_url` is the `http://127.0.0.1:PORT` origin to point the
/// client at.
pub struct MockServer {
    pub base_url: String,
    /// Number of `POST /v1/chat/completions` requests received.
    chat_hits: Arc<AtomicU64>,
    /// The parsed JSON body of every `POST /v1/chat/completions`, in arrival
    /// order — lets tests assert exactly what context the client sent
    /// upstream (e.g. regenerate's upstream-only thread).
    chat_bodies: Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
    /// Whether each `POST /v1/chat/completions` carried an `Authorization`
    /// header, in arrival order — local turns must send none (no spend).
    chat_auths: Arc<std::sync::Mutex<Vec<bool>>>,
    /// The raw `Authorization` header of each `POST /v1/chat/completions`,
    /// in arrival order — lets tests distinguish a spend's `PrivateToken`
    /// from an external backend's `Bearer` key.
    chat_auth_values: Arc<std::sync::Mutex<Vec<Option<String>>>>,
    /// Number of `POST /v1/credentials/refund` requests received.
    refund_hits: Arc<AtomicU64>,
    /// Number of `GET /v1/models` requests received — the catalogue fetch a
    /// turn's *setup* makes, and so the earliest thing on the wire that says
    /// preparation got as far as building a client. Tests that assert a turn
    /// was refused before it did any work read this rather than `chat_hits`,
    /// which only notices once the turn is fully prepared.
    models_hits: Arc<AtomicU64>,
    _task: tokio::task::JoinHandle<()>,
}

impl MockServer {
    pub fn chat_hits(&self) -> u64 {
        self.chat_hits.load(Ordering::SeqCst)
    }
    pub fn refund_hits(&self) -> u64 {
        self.refund_hits.load(Ordering::SeqCst)
    }
    /// Catalogue fetches (see `models_hits`).
    pub fn models_hits(&self) -> u64 {
        self.models_hits.load(Ordering::SeqCst)
    }
    /// The recorded chat request bodies (see `chat_bodies`).
    pub fn chat_bodies(&self) -> Vec<serde_json::Value> {
        self.chat_bodies.lock().unwrap().clone()
    }
    /// Per-chat-request `Authorization` presence (see `chat_auths`).
    pub fn chat_auths(&self) -> Vec<bool> {
        self.chat_auths.lock().unwrap().clone()
    }
    /// Per-chat-request raw `Authorization` values (see `chat_auth_values`).
    pub fn chat_auth_values(&self) -> Vec<Option<String>> {
        self.chat_auth_values.lock().unwrap().clone()
    }
    /// The loopback port the mock listens on — used by local-model tests to
    /// register a fake "loaded engine" at the mock's address.
    pub fn port(&self) -> u16 {
        self.base_url
            .rsplit(':')
            .next()
            .and_then(|p| p.parse().ok())
            .expect("mock base_url always carries a port")
    }
}

// ===========================================================================
// Participant-aware upstream rendering (roles + headers)
// ===========================================================================

/// The label of the seeded default agent — `db::default_agent_label` over the
/// compiled-in `DEFAULT_MODEL`, which the harness's [`MODEL`] matches.
pub const DEFAULT_AGENT_LABEL: &str = "Gemma4 31b";

/// The shared human participant's label. Viewpoint-neutral on the wire since
/// task 64 — the GUI still *displays* "You" for a human labelled `user`, so the
/// reader's experience is unchanged while the wire stops asserting second
/// person at a participant who may not be the reader.
pub const HUMAN_LABEL: &str = "User";

/// The rendering-protocol note app-core appends to every turn's system
/// message. Pinned here verbatim (the same discipline as pinning the rendered
/// message bytes) — a change to `HEADER_PROTOCOL_NOTE` in `lib.rs` must be a
/// deliberate edit here too.
pub const HEADER_PROTOCOL_NOTE: &str = "Each message in this conversation begins with a one-line header identifying the post, its \
     author, and when it was written: `#<handle> · <author> · <UTC timestamp>`. Handles are \
     assigned by the client; never write a header line yourself — reply with your message text \
     only.";

/// The note app-core appends to the system message of any turn that carries a
/// trailing volatile message at all — a roster, a map, or both. Pinned verbatim,
/// same discipline as [`HEADER_PROTOCOL_NOTE`].
pub const TRAILING_BLOCK_NOTE: &str = "The last message is client-generated metadata about this \
     space, not a post by any participant. No reply is due to it, and it ends by naming the post \
     you are answering.";

/// The thread-map protocol note app-core appends to the system message of a
/// turn whose space has branches (task 21), after [`TRAILING_BLOCK_NOTE`].
pub const THREAD_MAP_NOTE: &str = "This space is threaded: the conversation above is one branch of \
     it, and other branches exist. A `<thread-map>` block in that message lists them.";

/// Appended after [`THREAD_MAP_NOTE`] only when the navigation tools actually
/// attach (i.e. the backend can carry a `tools` field).
pub const THREAD_MAP_TOOLS_NOTE: &str = "When a map entry looks relevant to what you are writing, \
     read it with the navigation tools (`list_branches`, `read_thread`, `read_post`); otherwise \
     answer from the conversation you were given — most replies need none.";

/// The task-64 identity line: the sentence that tells a model which
/// participant it is. Written out long-hand (a byte pin, not a
/// reimplementation) — it leads the notes, i.e. sits directly after the
/// charter.
pub fn identity_line(label: &str) -> String {
    format!("You are \"{label}\" in this conversation.")
}

/// The exact `system` message content a turn sends: the responding
/// participant's effective system prompt (when it has one), then the identity
/// line, then the rendering-protocol note.
///
/// `responder` is the responding participant's effective label in the space —
/// the identity line's subject.
pub fn system_message(prompt: Option<&str>, responder: &str) -> String {
    system_message_with(prompt, responder, &[])
}

/// [`system_message`] plus the extra notes a turn appends (the thread-map
/// notes), joined by blank lines exactly as `prepare_turn` does.
pub fn system_message_with(prompt: Option<&str>, responder: &str, extra_notes: &[&str]) -> String {
    let identity = identity_line(responder);
    let mut notes = vec![identity.as_str(), HEADER_PROTOCOL_NOTE];
    notes.extend_from_slice(extra_notes);
    match prompt {
        Some(p) => {
            let mut s = p.to_string();
            for n in &notes {
                s.push_str("\n\n");
                s.push_str(n);
            }
            s
        }
        None => notes.join("\n\n"),
    }
}

/// The exact task-64 roster block a turn prepends to its trailing volatile
/// message when the space is multi-party (`has_map || participants > 2`).
///
/// `members` is `(label, kind, is_you)` in membership order — written out
/// long-hand, which is the point: a byte pin, not a reimplementation.
pub fn roster(members: &[(&str, &str, bool)]) -> String {
    let mut out = String::from("Participants in this conversation:\n");
    for (label, kind, you) in members {
        let you = if *you { " (you)" } else { "" };
        out.push_str(&format!("- \"{label}\" — {kind}{you}\n"));
    }
    out.push_str(
        "\nEach participant answers for itself; weigh others' posts on their merits rather than \
         deferring to them.",
    );
    out
}

/// The whole trailing volatile message: the roster (when present), the thread
/// map (when present), and the `Respond to #h.` pointer that always closes it,
/// separated by blank lines.
///
/// The pointer belongs to the *message*, not to the map, so every shape of the
/// block ends the same way — a roster-only block included.
pub fn trailing(roster: Option<&str>, map: Option<&str>, respond_to: &str) -> String {
    let mut sections: Vec<&str> = Vec::new();
    sections.extend(roster);
    sections.extend(map);
    if sections.is_empty() {
        return String::new();
    }
    format!("{}\n\nRespond to #{respond_to}.", sections.join("\n\n"))
}

/// The exact `<memory>` section a turn appends to its system message when the
/// responding participant holds memory blocks (task 35).
///
/// `blocks` is `(name, scope_label, text)` where `scope_label` is rendered
/// verbatim (`"core"` / `"this space"`) — the expected bytes written out
/// long-hand, which is the point: a byte pin, not a reimplementation.
pub fn memory_section(blocks: &[(&str, &str, &str)]) -> String {
    let mut out = String::from(
        "<memory>\nNotes you wrote for yourself in earlier turns. They are not part of the \
         conversation and no one else is shown them. Core notes travel with you; the rest are \
         about this space.\n",
    );
    for (name, scope, text) in blocks {
        out.push_str(&format!("\n--- {name} ({scope}) ---\n{text}\n"));
    }
    out.push_str("</memory>");
    out
}

/// The exact `<thread-map>` message content a turn appends.
///
/// `forks` is `(anchor_line, entry_lines)` — the expected bytes written out
/// long-hand, which is the point: this is a byte pin, not a reimplementation.
pub fn thread_map(forks: &[(String, Vec<String>)]) -> String {
    let mut out = String::from(
        "<thread-map>\nBranches of this space that the conversation above does not contain. Each \
         line: handle · author · posts · last activity — opening line; a branch you have posted \
         in also says so.\n",
    );
    for (anchor, entries) in forks {
        out.push('\n');
        out.push_str(anchor);
        out.push('\n');
        for e in entries {
            out.push_str("  ");
            out.push_str(e);
            out.push('\n');
        }
    }
    out.push_str("</thread-map>");
    out
}

/// One thread-map entry line: `#handle · author · posts · when — opening`,
/// with the task-33 `· you participated, <mine>` segment when the responding
/// participant posted in that branch.
pub fn map_entry(
    item_id: &str,
    author: &str,
    posts: &str,
    when: &str,
    mine: Option<&str>,
    opening: &str,
) -> String {
    let mine = mine
        .map(|m| format!(" · you participated, {m}"))
        .unwrap_or_default();
    format!(
        "#{} · {author} · {posts} · {when}{mine} — {opening}",
        eidola_app_core::post_handle(item_id)
    )
}

/// A thread-map entry plus the LLM-written summary line under it. The summary
/// carries its own indent because [`thread_map`] only prefixes an entry's first
/// line.
#[allow(clippy::too_many_arguments)]
pub fn map_entry_summarized(
    item_id: &str,
    author: &str,
    posts: &str,
    when: &str,
    mine: Option<&str>,
    opening: &str,
    summary: &str,
) -> String {
    format!(
        "{}\n      {summary}",
        map_entry(item_id, author, posts, when, mine, opening)
    )
}

/// The exact rendered upstream content of a post: its
/// `#<handle> · <label> · <stamp>` header line, a blank line, then the text.
///
/// `at_ms` is the post's **current generation's** creation time, which is what
/// the header stamps (task 64). Tests read it from the space's post tree — see
/// [`Stamps`] — because it is a real wall-clock value, and pinning it is
/// exactly how the pins prove the stamp does not drift between turns.
pub fn headed(item_id: &str, label: &str, at_ms: i64, text: &str) -> String {
    let handle = eidola_app_core::post_handle(item_id);
    let stamp = eidola_app_core::post_stamp(at_ms);
    if text.is_empty() {
        format!("#{handle} · {label} · {stamp}")
    } else {
        format!("#{handle} · {label} · {stamp}\n\n{text}")
    }
}

/// Creation times of a space's posts, by item id — the header stamps a turn
/// will render.
///
/// Captured from `get_space_tree`, i.e. from each item's **current
/// generation**, which is what the header carries. So a snapshot taken before
/// an edit and one taken after legitimately disagree about that item, and a
/// test asserting pre-edit bytes must hold the pre-edit snapshot: that is the
/// semantics, not an accident.
#[derive(Clone, Debug, Default)]
pub struct Stamps(std::collections::HashMap<String, i64>);

impl Stamps {
    /// Capture the space's current post stamps.
    pub fn of(core: &AppCore, space_id: &str) -> Self {
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(space_id.to_string()))
            .expect("post tree");
        Self(
            tree.into_iter()
                .map(|n| (n.item_id, n.created_at))
                .collect(),
        )
    }

    /// The stamp time of `item_id`. Panics when the item is not in the
    /// snapshot — a header can only be pinned against a post that exists.
    pub fn at(&self, item_id: &str) -> i64 {
        *self
            .0
            .get(item_id)
            .unwrap_or_else(|| panic!("no post stamp captured for item {item_id}"))
    }

    /// [`headed`] over this snapshot's stamp for `item_id`.
    pub fn headed(&self, item_id: &str, label: &str, text: &str) -> String {
        headed(item_id, label, self.at(item_id), text)
    }
}

/// `(role, content)` pairs of a recorded chat request body's `messages` array.
pub fn flat_messages(body: &serde_json::Value) -> Vec<(String, String)> {
    body["messages"]
        .as_array()
        .expect("messages array")
        .iter()
        .map(|m| {
            (
                m["role"].as_str().unwrap_or_default().to_string(),
                m["content"].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect()
}

/// The mock issuer: an ACT keypair plus the params/context the server derives.
struct Issuer {
    key: PrivateKey,
    params: Params,
    key_id_hex: String,
    request_context_scalar: Scalar,
}

impl Issuer {
    fn new() -> Self {
        let key = PrivateKey::random(OsRng);
        let public_key_cbor = key.public().to_cbor().expect("encode public key");
        let key_hash: [u8; 32] = Sha256::digest(&public_key_cbor).into();
        let key_id_hex = hex::encode(key_hash);
        let params = Params::new(DS_ORG, DS_SERVICE, DS_DEPLOYMENT, DS_VERSION);
        // request_context = issuer_name || origin_info || key_hash, then
        // SHA-256 → scalar (mirrors `credentials.rs`).
        let mut ctx = Vec::new();
        ctx.extend_from_slice(ISSUER_NAME.as_bytes());
        ctx.extend_from_slice(ORIGIN_INFO.as_bytes());
        ctx.extend_from_slice(&key_hash);
        let ctx_hash: [u8; 32] = Sha256::digest(&ctx).into();
        let request_context_scalar = Scalar::from_bytes_mod_order(ctx_hash);
        Self {
            key,
            params,
            key_id_hex,
            request_context_scalar,
        }
    }

    fn public_key_b64(&self) -> String {
        URL_SAFE_NO_PAD.encode(self.key.public().to_cbor().expect("encode public key"))
    }

    /// Issue a credential for a base64url CBOR `IssuanceRequest` and credit
    /// amount. Returns the base64url CBOR `IssuanceResponse`.
    fn issue(&self, issuance_request_b64: &str, credits: i64) -> Option<String> {
        let req_cbor = URL_SAFE_NO_PAD.decode(issuance_request_b64).ok()?;
        let req = IssuanceRequest::from_cbor(&req_cbor).ok()?;
        let credit_scalar = credit_to_scalar::<128>(credits as u128).ok()?;
        let resp = self
            .key
            .issue::<128>(
                &self.params,
                &req,
                credit_scalar,
                self.request_context_scalar,
                OsRng,
            )
            .ok()?;
        Some(URL_SAFE_NO_PAD.encode(resp.to_cbor().ok()?))
    }

    /// Produce a refund for a parsed spend proof, refunding the full charge
    /// (no work performed). Returns the base64url CBOR `Refund`.
    fn refund_for(&self, spend_proof: &SpendProof<128>) -> Option<String> {
        if spend_proof.context() != self.request_context_scalar {
            return None;
        }
        let charge = scalar_to_credit::<128>(&spend_proof.charge()).ok()?;
        let t = credit_to_scalar::<128>(charge).ok()?;
        let refund = self
            .key
            .refund::<128>(&self.params, spend_proof, t, OsRng)
            .ok()?;
        Some(URL_SAFE_NO_PAD.encode(refund.to_cbor().ok()?))
    }

    /// Parse a `PrivateToken token="..."` header into its embedded spend proof
    /// (mirrors the server's `TokenAuth` extractor).
    fn spend_proof_from_auth(auth: &str) -> Option<SpendProof<128>> {
        let payload = auth
            .strip_prefix("PrivateToken token=\"")
            .and_then(|s| s.strip_suffix('"'))?;
        let bytes = URL_SAFE_NO_PAD.decode(payload).ok()?;
        if bytes.len() < 67 {
            return None;
        }
        // [token_type(2)][challenge_digest(32)][issuer_key_id(32)][spend_proof(..)]
        SpendProof::<128>::from_cbor(&bytes[66..]).ok()
    }
}

/// Start the mock upstream on an ephemeral loopback port.
pub async fn start(config: MockConfig) -> MockServer {
    let issuer = Arc::new(Issuer::new());
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let base_url = format!("http://{addr}");

    let chat_hits = Arc::new(AtomicU64::new(0));
    let chat_bodies: Arc<std::sync::Mutex<Vec<serde_json::Value>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let chat_auths: Arc<std::sync::Mutex<Vec<bool>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let chat_auth_values: Arc<std::sync::Mutex<Vec<Option<String>>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let refund_hits = Arc::new(AtomicU64::new(0));
    let models_hits = Arc::new(AtomicU64::new(0));

    let task = {
        let issuer = issuer.clone();
        let chat_hits = chat_hits.clone();
        let chat_bodies = chat_bodies.clone();
        let chat_auths = chat_auths.clone();
        let chat_auth_values = chat_auth_values.clone();
        let refund_hits = refund_hits.clone();
        let models_hits = models_hits.clone();
        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let issuer = issuer.clone();
                let config = config.clone();
                let chat_hits = chat_hits.clone();
                let chat_bodies = chat_bodies.clone();
                let chat_auths = chat_auths.clone();
                let chat_auth_values = chat_auth_values.clone();
                let refund_hits = refund_hits.clone();
                let models_hits = models_hits.clone();
                tokio::spawn(async move {
                    let _ = handle_conn(
                        stream,
                        issuer,
                        config,
                        chat_hits,
                        chat_bodies,
                        chat_auths,
                        chat_auth_values,
                        refund_hits,
                        models_hits,
                    )
                    .await;
                });
            }
        })
    };

    MockServer {
        base_url,
        chat_hits,
        chat_bodies,
        chat_auths,
        chat_auth_values,
        refund_hits,
        models_hits,
        _task: task,
    }
}

/// A parsed HTTP/1.1 request: method, path, auth header, body.
struct Req {
    method: String,
    path: String,
    auth: Option<String>,
    body: Vec<u8>,
}

async fn read_request(stream: &mut TcpStream) -> Option<Req> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    // Read until end of headers.
    let head_end = loop {
        let n = stream.read(&mut tmp).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos + 4;
        }
        if buf.len() > 64 * 1024 {
            return None;
        }
    };

    let head = std::str::from_utf8(&buf[..head_end]).ok()?;
    let mut lines = head.split("\r\n");
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();

    let mut content_length = 0usize;
    let mut auth = None;
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            let name = name.trim().to_ascii_lowercase();
            let value = value.trim();
            if name == "content-length" {
                content_length = value.parse().unwrap_or(0);
            } else if name == "authorization" {
                auth = Some(value.to_string());
            }
        }
    }

    let mut body = buf[head_end..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut tmp).await.ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }

    Some(Req {
        method,
        path,
        auth,
        body,
    })
}

#[allow(clippy::too_many_arguments)]
async fn handle_conn(
    mut stream: TcpStream,
    issuer: Arc<Issuer>,
    config: MockConfig,
    chat_hits: Arc<AtomicU64>,
    chat_bodies: Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
    chat_auths: Arc<std::sync::Mutex<Vec<bool>>>,
    chat_auth_values: Arc<std::sync::Mutex<Vec<Option<String>>>>,
    refund_hits: Arc<AtomicU64>,
    models_hits: Arc<AtomicU64>,
) -> std::io::Result<()> {
    // Each connection serves at most one request (reqwest opens a fresh
    // connection per request here; HTTP/1.1 keep-alive is unnecessary for the
    // test and one-request-per-connection keeps the framing trivial).
    let Some(req) = read_request(&mut stream).await else {
        return Ok(());
    };

    let path = req.path.as_str();

    // Route. Paths are matched without the `?...` query (none used here).
    match (req.method.as_str(), path) {
        ("GET", "/v1/models") => {
            models_hits.fetch_add(1, Ordering::SeqCst);
            match config.models_status {
                Some(status) => {
                    write_json(&mut stream, status, r#"{"error":"models unavailable"}"#).await?;
                }
                None => {
                    write_json(&mut stream, 200, &models_body(&config)).await?;
                }
            }
        }
        ("GET", "/v1/keys") => {
            write_json(&mut stream, 200, &keys_body(&issuer)).await?;
        }
        ("GET", "/v1/account/balances") => {
            let body = serde_json::json!({
                "available": config.balance,
                "pools": [{ "amount": config.balance, "source": "mock", "expires_at": null }],
            });
            write_json(&mut stream, 200, &body.to_string()).await?;
        }
        ("POST", "/v1/account/credentials") => {
            let parsed: serde_json::Value =
                serde_json::from_slice(&req.body).unwrap_or(serde_json::Value::Null);
            let ir = parsed.get("issuance_request").and_then(|v| v.as_str());
            let credits = parsed.get("credits").and_then(|v| v.as_i64()).unwrap_or(0);
            match ir.and_then(|ir| issuer.issue(ir, credits)) {
                Some(issuance_response) => {
                    let body = serde_json::json!({
                        "issuance_response": issuance_response,
                        "issuer_key_id": issuer.key_id_hex,
                        "credits": credits,
                        "ledger_entry_id": uuid_like(),
                    });
                    write_json(&mut stream, 200, &body.to_string()).await?;
                }
                None => {
                    write_json(&mut stream, 400, &error_body("issuance failed")).await?;
                }
            }
        }
        ("POST", "/v1/credentials/refund") => {
            let attempt = refund_hits.fetch_add(1, Ordering::SeqCst) + 1;
            handle_refund(&mut stream, &issuer, &config, req.auth.as_deref(), attempt).await?;
        }
        ("POST", "/v1/chat/completions") => {
            // 1-based index of this chat request — the tool-calling
            // behaviours script their answer per round.
            let hit = chat_hits.fetch_add(1, Ordering::SeqCst) + 1;
            let parsed = serde_json::from_slice::<serde_json::Value>(&req.body).ok();
            let is_router = parsed
                .as_ref()
                .and_then(|b| b.get("model"))
                .and_then(|m| m.as_str())
                .is_some_and(|m| m == ROUTER_MODEL || m == ROUTER_REMOTE_MODEL);
            if let Some(body) = parsed {
                chat_bodies.lock().unwrap().push(body);
            }
            chat_auths.lock().unwrap().push(req.auth.is_some());
            chat_auth_values.lock().unwrap().push(req.auth.clone());
            let parsed: serde_json::Value =
                serde_json::from_slice(&req.body).unwrap_or(serde_json::Value::Null);
            // A branch-summary request is recognized by its system prompt (it
            // shares the router's model) and answered first, so a test can
            // script the summarizer, the router, and the turns independently.
            let is_summary = parsed
                .get("messages")
                .and_then(|m| m.as_array())
                .and_then(|a| a.first())
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .is_some_and(|c| c.starts_with(SUMMARY_PROMPT_HEAD));
            match (is_summary, &config.summary) {
                (true, SummaryBehavior::Reply(content)) => {
                    let body = serde_json::json!({
                        "choices": [{ "message": { "role": "assistant", "content": content } }],
                        "usage": { "prompt_tokens": 9, "completion_tokens": 5 },
                    });
                    return write_json(&mut stream, 200, &body.to_string()).await;
                }
                (true, SummaryBehavior::Fail(status)) => {
                    return write_json(&mut stream, *status, &error_body("summarizer unavailable"))
                        .await;
                }
                _ => {}
            }
            // A request for the router model is answered by the router arm
            // (unless the mock is not router-aware), leaving `ChatBehavior`
            // free to script the *turns* in the same test.
            match (is_router, &config.router) {
                (true, RouterBehavior::Reply(content)) => {
                    let body = serde_json::json!({
                        "choices": [{ "message": { "role": "assistant", "content": content } }],
                        "usage": { "prompt_tokens": 7, "completion_tokens": 3 },
                    });
                    write_json(&mut stream, 200, &body.to_string()).await?;
                }
                (true, RouterBehavior::Fail(status)) => {
                    write_json(&mut stream, *status, &error_body("router unavailable")).await?;
                }
                (true, RouterBehavior::DropBody) => {
                    // Headers promise 64 bytes; none arrive and the connection
                    // closes, so the client's body read errors after the
                    // request was accepted.
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\n\
                              Content-Type: application/json\r\n\
                              Content-Length: 64\r\n\
                              Connection: close\r\n\r\n",
                        )
                        .await?;
                    stream.flush().await?;
                }
                _ => {
                    handle_chat(
                        &mut stream,
                        &issuer,
                        &config,
                        req.auth.as_deref(),
                        hit,
                        &parsed,
                    )
                    .await?;
                }
            }
        }
        _ => {
            write_json(&mut stream, 404, &error_body("not found")).await?;
        }
    }
    Ok(())
}

async fn handle_refund(
    stream: &mut TcpStream,
    issuer: &Issuer,
    config: &MockConfig,
    auth: Option<&str>,
    // 1-based index of this recovery call.
    attempt: u64,
) -> std::io::Result<()> {
    let failing = match config.refund {
        RefundMode::Fail => true,
        RefundMode::FailFirst(n) => attempt <= n,
        RefundMode::Succeed => false,
    };
    if failing {
        return write_json(stream, 500, &error_body("refund unavailable")).await;
    }
    let refund = auth
        .and_then(Issuer::spend_proof_from_auth)
        .and_then(|sp| issuer.refund_for(&sp));
    match refund {
        Some(refund_b64) => {
            let body = serde_json::json!({
                "refund": { "refund": refund_b64, "issuer_key_id": issuer.key_id_hex },
            });
            write_json(stream, 200, &body.to_string()).await
        }
        None => write_json(stream, 500, &error_body("refund proof invalid")).await,
    }
}

async fn handle_chat(
    stream: &mut TcpStream,
    issuer: &Issuer,
    config: &MockConfig,
    auth: Option<&str>,
    hit: u64,
    request: &serde_json::Value,
) -> std::io::Result<()> {
    if config.chat_delay_ms > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(config.chat_delay_ms)).await;
    }
    // Compute an inline refund object once (shared by the blocking happy path).
    let inline_refund = if config.refund == RefundMode::Succeed {
        auth.and_then(Issuer::spend_proof_from_auth)
            .and_then(|sp| issuer.refund_for(&sp))
            .map(|refund_b64| {
                serde_json::json!({ "refund": refund_b64, "issuer_key_id": issuer.key_id_hex })
            })
    } else {
        None
    };

    match config.chat {
        ChatBehavior::OkBlocking | ChatBehavior::OkBlockingNoInlineRefund => {
            // `reasoning_content` mirrors the SSE stream's `delta.reasoning`:
            // the blocking path recovers the model's thinking from the
            // aggregated `message` object, so both transports persist a
            // `thinking` block symmetrically.
            let mut body = serde_json::json!({
                "choices": [{ "message": {
                    "role": "assistant",
                    "content": "Hello from the mock.",
                    "reasoning_content": "thinking…",
                } }],
                "usage": { "prompt_tokens": 11, "completion_tokens": 5 },
            });
            if matches!(config.chat, ChatBehavior::OkBlocking)
                && let Some(refund) = inline_refund
            {
                body["refund"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::Non2xx(status) => {
            write_json(stream, status, &error_body("upstream model error")).await
        }
        ChatBehavior::DropBeforeResponse => {
            // Drop the connection without writing anything: the client's
            // `send().await` succeeds (request was sent) but reading the body /
            // the response itself fails — exercising the network-error arm.
            // Actually `send()` returns once headers are read; with no bytes at
            // all reqwest surfaces a transport error from `send`.
            Ok(())
        }
        ChatBehavior::OkStreaming => write_sse_stream(stream, true, &[STREAM_CONTENT]).await,
        ChatBehavior::OkStreamingWithHeader => {
            write_sse_stream(
                stream,
                true,
                &[&format!("{MIMICKED_HEADER}\n\n{STREAM_CONTENT}")],
            )
            .await
        }
        ChatBehavior::OkStreamingWithSplitHeader => {
            // Chopped the way a token stream chops: mid-handle, mid-separator,
            // and with the blank line arriving attached to the first body token.
            let (handle, label) = MIMICKED_HEADER
                .split_once(" \u{b7} ")
                .expect("the mimicked header carries the separator");
            let (h1, h2) = handle.split_at(4);
            write_sse_stream(
                stream,
                true,
                &[
                    h1,
                    h2,
                    " \u{b7}",
                    " ",
                    label,
                    &format!("\n\n{STREAM_CONTENT}"),
                ],
            )
            .await
        }
        ChatBehavior::StreamingMidAbort => write_sse_stream(stream, false, &[STREAM_CONTENT]).await,
        ChatBehavior::StreamingEndsWithoutDone => {
            write_sse_unterminated_stream(stream, &[STREAM_CONTENT]).await
        }
        ChatBehavior::OkThenStreamEndsWithoutDone if hit == 1 => {
            if request.get("stream").and_then(|s| s.as_bool()) == Some(true) {
                return write_sse_stream(stream, true, &[STREAM_CONTENT]).await;
            }
            let mut body = serde_json::json!({
                "choices": [{ "message": {
                    "role": "assistant",
                    "content": "Hello from the mock.",
                } }],
                "usage": { "prompt_tokens": 11, "completion_tokens": 5 },
            });
            if let Some(refund) = inline_refund {
                body["refund"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::OkThenStreamEndsWithoutDone => {
            write_sse_unterminated_stream(stream, &[STREAM_CONTENT]).await
        }
        ChatBehavior::ToolRoundsBlocking(rounds) => {
            if hit <= rounds {
                let mut body = tool_call_body(&tool_call_object(hit, &tool_arguments(hit)));
                if let Some(refund) = inline_refund {
                    body["refund"] = refund;
                }
                write_json(stream, 200, &body.to_string()).await
            } else {
                let mut body = serde_json::json!({
                    "choices": [{ "message": {
                        "role": "assistant",
                        "content": TOOL_FINAL_CONTENT,
                    } }],
                    "usage": { "prompt_tokens": 11, "completion_tokens": 5 },
                });
                if let Some(refund) = inline_refund {
                    body["refund"] = refund;
                }
                write_json(stream, 200, &body.to_string()).await
            }
        }
        ChatBehavior::ToolCallMalformed => {
            // No `id` on the call — nothing to execute, nothing to persist.
            let call = serde_json::json!({
                "type": "function",
                "function": { "name": TOOL_NAME, "arguments": "{}" },
            });
            let mut body = tool_call_body(&call);
            if let Some(refund) = inline_refund {
                body["refund"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::ToolCallBadArguments => {
            let mut body = if hit == 1 {
                tool_call_body(&tool_call_object(1, "{not json"))
            } else {
                serde_json::json!({
                    "choices": [{ "message": {
                        "role": "assistant",
                        "content": TOOL_FINAL_CONTENT,
                    } }],
                    "usage": { "prompt_tokens": 11, "completion_tokens": 5 },
                })
            };
            if let Some(refund) = inline_refund {
                body["refund"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::ToolRoundsStreaming(rounds) => {
            if hit <= rounds {
                write_sse_tool_stream(stream, hit, false).await
            } else {
                write_sse_stream(stream, true, &[TOOL_FINAL_CONTENT]).await
            }
        }
        ChatBehavior::ToolRoundsStreamingWithExtras(rounds) => {
            if hit <= rounds {
                write_sse_tool_stream(stream, hit, true).await
            } else {
                write_sse_stream(stream, true, &[TOOL_FINAL_CONTENT]).await
            }
        }
        ChatBehavior::ToolCallsNotAnArray => {
            let mut body = serde_json::json!({
                "choices": [{ "message": {
                    "role": "assistant",
                    "content": serde_json::Value::Null,
                    // An object, not an array — structurally unusable.
                    "tool_calls": { "id": tool_call_id(1), "function": {
                        "name": TOOL_NAME, "arguments": "{}" } },
                } }],
                "usage": { "prompt_tokens": 11, "completion_tokens": 5 },
            });
            if let Some(refund) = inline_refund {
                body["refund"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::ToolCallsNull => {
            let mut body = serde_json::json!({
                "choices": [{ "message": {
                    "role": "assistant",
                    "content": TOOL_FINAL_CONTENT,
                    "tool_calls": serde_json::Value::Null,
                } }],
                "usage": { "prompt_tokens": 11, "completion_tokens": 5 },
            });
            if let Some(refund) = inline_refund {
                body["refund"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::ToolCallsNotAnArrayStreaming => write_sse_bad_tool_calls_stream(stream).await,
        ChatBehavior::DeclineBlocking => {
            let mut body = tool_call_body(&decline_call_object());
            if let Some(refund) = inline_refund {
                body["refund"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::DeclineStreaming => write_sse_decline_stream(stream).await,
        ChatBehavior::RejectTools | ChatBehavior::RejectToolsForModel(_) => {
            // `RejectTools` refuses the field for every model on this host;
            // `RejectToolsForModel` for exactly one.
            let refused_model = match config.chat {
                ChatBehavior::RejectToolsForModel(m) => Some(m),
                _ => None,
            };
            let wire_model = request
                .get("model")
                .and_then(|m| m.as_str())
                .unwrap_or_default();
            if request.get("tools").is_some()
                && refused_model.is_none_or(|refused| refused == wire_model)
            {
                return write_json(
                    stream,
                    500,
                    &error_body("tools param requires --jinja flag"),
                )
                .await;
            }
            // **The retry is answered in the transport that asked.** Both
            // doors reach this behaviour — the sub-space driver only ever
            // streams — and answering a stream with a completion body sends no
            // `data:` frames at all, so the turn read as an empty answer that
            // nothing complained about.
            if request.get("stream").and_then(|s| s.as_bool()) == Some(true) {
                return write_sse_stream(stream, true, &[STREAM_CONTENT]).await;
            }
            let mut body = serde_json::json!({
                "choices": [{ "message": {
                    "role": "assistant",
                    "content": "Hello from the mock.",
                } }],
                "usage": { "prompt_tokens": 11, "completion_tokens": 5 },
            });
            if let Some(refund) = inline_refund {
                body["refund"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::OkThenReasoningOnlyLength if hit == 1 => {
            if request.get("stream").and_then(|s| s.as_bool()) == Some(true) {
                return write_sse_stream(stream, true, &[STREAM_CONTENT]).await;
            }
            let mut body = serde_json::json!({
                "choices": [{ "message": {
                    "role": "assistant",
                    "content": "Hello from the mock.",
                } }],
                "usage": { "prompt_tokens": 11, "completion_tokens": 5 },
            });
            if let Some(refund) = inline_refund {
                body["refund"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::ReasoningOnlyLength | ChatBehavior::OkThenReasoningOnlyLength => {
            if request.get("stream").and_then(|s| s.as_bool()) == Some(true) {
                return write_sse_length_stream(stream, &[]).await;
            }
            let mut body = serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": TRUNCATED_REASONING,
                    },
                    "finish_reason": "length",
                }],
                "usage": { "prompt_tokens": 11, "completion_tokens": CEILING_TOKENS },
            });
            if let Some(refund) = inline_refund {
                body["refund"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::PartialAnswerLength => {
            if request.get("stream").and_then(|s| s.as_bool()) == Some(true) {
                return write_sse_length_stream(stream, &[STREAM_CONTENT]).await;
            }
            let mut body = serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": STREAM_CONTENT,
                        "reasoning_content": TRUNCATED_REASONING,
                    },
                    "finish_reason": "length",
                }],
                "usage": { "prompt_tokens": 11, "completion_tokens": CEILING_TOKENS },
            });
            if let Some(refund) = inline_refund {
                body["refund"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::OkEitherTransport => {
            if request.get("stream").and_then(|s| s.as_bool()) == Some(true) {
                return write_sse_stream(stream, true, &[STREAM_CONTENT]).await;
            }
            let mut body = serde_json::json!({
                "choices": [{ "message": {
                    "role": "assistant",
                    "content": "Hello from the mock.",
                } }],
                "usage": { "prompt_tokens": 11, "completion_tokens": 5 },
            });
            if let Some(refund) = inline_refund {
                body["refund"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::RejectToolsUnparseable => {
            if request.get("tools").is_some() {
                // The handler body never runs, so: no `refund` in the
                // response, and the ACT nullifier is left unrecorded — which
                // is what lets `/v1/credentials/refund` issue a full one.
                return write_plain_text(stream, 422, UNKNOWN_FIELD_REJECTION).await;
            }
            // Answer in whichever transport asked, so one behaviour covers the
            // blocking twin and the streaming one.
            if request.get("stream").and_then(|s| s.as_bool()) == Some(true) {
                return write_sse_stream(stream, true, &[STREAM_CONTENT]).await;
            }
            let mut body = serde_json::json!({
                "choices": [{ "message": {
                    "role": "assistant",
                    "content": "Hello from the mock.",
                } }],
                "usage": { "prompt_tokens": 11, "completion_tokens": 5 },
            });
            if let Some(refund) = inline_refund {
                body["refund"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::RejectAutoToolsThenToolRounds => {
            let advertises_auto_tools = request
                .get("tools")
                .and_then(|t| t.as_array())
                .is_some_and(|tools| {
                    tools
                        .iter()
                        .any(|t| t["function"]["name"].as_str() == Some("list_branches"))
                });
            if advertises_auto_tools {
                return write_json(
                    stream,
                    500,
                    &error_body("tools param requires --jinja flag"),
                )
                .await;
            }
            let mut body = tool_call_body(&tool_call_object(hit, &tool_arguments(hit)));
            if let Some(refund) = inline_refund {
                body["refund"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::ToolScript => {
            // Consume the script: the round that serves it asks for every
            // scripted call at once (so one turn can exercise several tools),
            // and every later request answers normally.
            let script: Vec<(String, String)> =
                std::mem::take(&mut *config.tool_script.lock().unwrap());
            let mut body = if script.is_empty() {
                serde_json::json!({
                    "choices": [{ "message": {
                        "role": "assistant",
                        "content": TOOL_FINAL_CONTENT,
                    } }],
                    "usage": { "prompt_tokens": 11, "completion_tokens": 5 },
                })
            } else {
                let calls: Vec<serde_json::Value> = script
                    .iter()
                    .enumerate()
                    .map(|(i, (name, arguments))| {
                        serde_json::json!({
                            "id": tool_call_id(i as u64 + 1),
                            "type": "function",
                            "function": { "name": name, "arguments": arguments },
                        })
                    })
                    .collect();
                serde_json::json!({
                    "choices": [{ "message": {
                        "role": "assistant",
                        "content": serde_json::Value::Null,
                        "tool_calls": calls,
                    } }],
                    "usage": { "prompt_tokens": 11, "completion_tokens": 5 },
                })
            };
            if let Some(refund) = inline_refund {
                body["refund"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
    }
}

/// SSE stream whose single delta carries a non-null, non-array `tool_calls`.
async fn write_sse_bad_tool_calls_stream(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(sse_head().as_bytes()).await?;
    stream.flush().await?;
    let delta = serde_json::json!({
        "choices": [{ "delta": { "tool_calls": "definitely not an array" } }]
    });
    stream.write_all(&sse_event(&delta.to_string())).await?;
    stream
        .write_all(&sse_event(
            &serde_json::json!({"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":5}})
                .to_string(),
        ))
        .await?;
    stream.write_all(&sse_event("[DONE]")).await?;
    stream.write_all(b"0\r\n\r\n").await?;
    stream.flush().await?;
    Ok(())
}

/// The answer text the tool-calling behaviours give once they stop calling.
pub const TOOL_FINAL_CONTENT: &str = "Done, the tool said so.";

/// The `text` argument round `n`'s tool call carries.
pub fn tool_arguments(round: u64) -> String {
    serde_json::json!({ "text": format!("round-{round}") }).to_string()
}

/// The call id round `n`'s tool call carries.
pub fn tool_call_id(round: u64) -> String {
    format!("call_{round}")
}

/// The echoed result the harness's tool produces for round `n`.
pub fn tool_result_text(round: u64) -> String {
    format!("round-{round}")
}

fn tool_call_object(round: u64, arguments: &str) -> serde_json::Value {
    serde_json::json!({
        "id": tool_call_id(round),
        "type": "function",
        "function": { "name": TOOL_NAME, "arguments": arguments },
    })
}

fn decline_arguments() -> String {
    serde_json::json!({ "reason": DECLINE_REASON }).to_string()
}

fn decline_call_object() -> serde_json::Value {
    serde_json::json!({
        "id": "call_decline",
        "type": "function",
        "function": { "name": DECLINE_TOOL, "arguments": decline_arguments() },
    })
}

/// SSE stream carrying one whole `decline` call (the assembly path itself is
/// pinned by the four-delta echo stream; this arm is about the checkpoint).
async fn write_sse_decline_stream(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(sse_head().as_bytes()).await?;
    stream.flush().await?;
    let mut call = decline_call_object();
    call["index"] = serde_json::json!(0);
    let deltas = vec![
        serde_json::json!({"choices":[{"delta":{"tool_calls":[call]}}]}),
        serde_json::json!({"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":5}}),
    ];
    for d in deltas {
        stream.write_all(&sse_event(&d.to_string())).await?;
    }
    stream.write_all(&sse_event("[DONE]")).await?;
    stream.write_all(b"0\r\n\r\n").await?;
    stream.flush().await?;
    Ok(())
}

fn tool_call_body(call: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "choices": [{ "message": {
            "role": "assistant",
            "content": serde_json::Value::Null,
            "tool_calls": [call],
        } }],
        "usage": { "prompt_tokens": 11, "completion_tokens": 5 },
    })
}

/// SSE stream whose single tool call arrives in **four** deltas: the id plus
/// the first half of the function name, the second half of the name, then the
/// arguments string in two pieces. Nothing is a complete JSON value on its own,
/// which is exactly what the client's per-index accumulator has to survive.
async fn write_sse_tool_stream(
    stream: &mut TcpStream,
    round: u64,
    extras: bool,
) -> std::io::Result<()> {
    stream.write_all(sse_head().as_bytes()).await?;
    stream.flush().await?;

    let args = tool_arguments(round);
    let (args_a, args_b) = args.split_at(args.len() / 2);
    let (name_a, name_b) = TOOL_NAME.split_at(2);

    // Chunk 1 introduces the call plus provider fields at both levels
    // (including a nested object, which no concatenating merge could survive);
    // chunk 3 restates two of them with new values, pinning last-wins.
    let (extra_1, fn_extra_1, extra_3, fn_extra_3) = if extras {
        (
            serde_json::json!({ "provider_tag": "alpha", "trace": { "span": "s1" } }),
            serde_json::json!({ "cache_key": "k1" }),
            serde_json::json!({ "provider_tag": "beta" }),
            serde_json::json!({ "cache_key": "k2" }),
        )
    } else {
        (
            serde_json::json!({}),
            serde_json::json!({}),
            serde_json::json!({}),
            serde_json::json!({}),
        )
    };

    let mut call_1 = serde_json::json!({
        "index": 0, "id": tool_call_id(round), "type": "function"
    });
    merge_into(&mut call_1, &extra_1);
    let mut fn_1 = serde_json::json!({ "name": name_a });
    merge_into(&mut fn_1, &fn_extra_1);
    call_1["function"] = fn_1;

    let mut call_3 = serde_json::json!({ "index": 0 });
    merge_into(&mut call_3, &extra_3);
    let mut fn_3 = serde_json::json!({ "arguments": args_a });
    merge_into(&mut fn_3, &fn_extra_3);
    call_3["function"] = fn_3;

    let deltas = vec![
        serde_json::json!({"choices":[{"delta":{"tool_calls":[call_1]}}]}),
        serde_json::json!({"choices":[{"delta":{"tool_calls":[
            {"index":0,"function":{"name":name_b}}]}}]}),
        serde_json::json!({"choices":[{"delta":{"tool_calls":[call_3]}}]}),
        serde_json::json!({"choices":[{"delta":{"tool_calls":[
            {"index":0,"function":{"arguments":args_b}}]}}]}),
        serde_json::json!({"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":5}}),
    ];
    for d in deltas {
        stream.write_all(&sse_event(&d.to_string())).await?;
    }
    stream.write_all(&sse_event("[DONE]")).await?;
    stream.write_all(b"0\r\n\r\n").await?;
    stream.flush().await?;
    Ok(())
}

fn merge_into(target: &mut serde_json::Value, extra: &serde_json::Value) {
    let (Some(t), Some(e)) = (target.as_object_mut(), extra.as_object()) else {
        return;
    };
    for (k, v) in e {
        t.insert(k.clone(), v.clone());
    }
}

fn sse_head() -> &'static str {
    "HTTP/1.1 200 OK\r\n\
     Content-Type: text/event-stream\r\n\
     Transfer-Encoding: chunked\r\n\
     Connection: close\r\n\r\n"
}

/// One chunked SSE `data:` event.
fn sse_event(payload: &str) -> Vec<u8> {
    let event = format!("data: {payload}\n\n");
    let mut out = format!("{:x}\r\n", event.len()).into_bytes();
    out.extend_from_slice(event.as_bytes());
    out.extend_from_slice(b"\r\n");
    out
}

/// The streaming mock's answer text.
pub const STREAM_CONTENT: &str = "Hello from the stream.";

/// What a model that spent its whole budget thinking has to show for it.
pub const TRUNCATED_REASONING: &str = "still working through it…";

/// The completion ceiling the truncating behaviours report as consumed — the
/// client's own `max_completion_tokens` for a model with room for it, which is
/// what makes the response honest rather than merely shaped right.
pub const CEILING_TOKENS: i64 = 4096;

/// An SSE stream that ends at the **completion ceiling**: reasoning deltas,
/// then whatever `content_chunks` holds (possibly nothing), then a terminal
/// chunk carrying `finish_reason: "length"` and the spent budget, then
/// `[DONE]`.
///
/// The reason lands on its own trailing chunk with an empty delta, which is
/// where real providers put it — a client that only reads chunks carrying
/// content would miss it entirely.
async fn write_sse_length_stream(
    stream: &mut TcpStream,
    content_chunks: &[&str],
) -> std::io::Result<()> {
    let head = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Transfer-Encoding: chunked\r\n\
                Connection: close\r\n\r\n";
    stream.write_all(head.as_bytes()).await?;
    stream.flush().await?;

    let reasoning = serde_json::json!({
        "choices": [{ "delta": { "reasoning": TRUNCATED_REASONING }, "finish_reason": null }]
    });
    stream.write_all(&sse_event(&reasoning.to_string())).await?;
    stream.flush().await?;

    for chunk in content_chunks {
        let content = serde_json::json!({
            "choices": [{ "delta": { "content": chunk }, "finish_reason": null }]
        });
        stream.write_all(&sse_event(&content.to_string())).await?;
        stream.flush().await?;
    }

    let terminal = serde_json::json!({
        "choices": [{ "delta": {}, "finish_reason": "length" }],
        "usage": { "prompt_tokens": 11, "completion_tokens": CEILING_TOKENS }
    });
    stream.write_all(&sse_event(&terminal.to_string())).await?;
    stream.write_all(&sse_event("[DONE]")).await?;
    stream.write_all(b"0\r\n\r\n").await?;
    stream.flush().await?;
    Ok(())
}

/// A header-shaped first line a model might mimic (see
/// `ChatBehavior::OkStreamingWithHeader`). Deliberately *not* a real handle of
/// anything — strip-on-receipt is shape-based, not identity-based.
pub const MIMICKED_HEADER: &str = "#a2c3d4e · Gemma4 31b";

/// Write a chunked SSE stream. When `complete`, emits reasoning + one content
/// delta per entry of `content_chunks`, a usage chunk, and `[DONE]`. When not
/// complete, emits one partial event and then drops the connection mid-stream
/// (simulating an abort).
/// An SSE stream that **ends without ever saying it is over**: reasoning, then
/// `content_chunks`, then a proper end to the chunked body — and no `[DONE]`,
/// no `finish_reason`, on any chunk.
///
/// Every byte is well-formed, so nothing in the transport complains; the only
/// thing missing is the upstream's claim to have finished. That is what
/// separates it from [`ChatBehavior::StreamingMidAbort`], which dies inside a
/// frame and fails in the decoder.
async fn write_sse_unterminated_stream(
    stream: &mut TcpStream,
    content_chunks: &[&str],
) -> std::io::Result<()> {
    let head = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Transfer-Encoding: chunked\r\n\
                Connection: close\r\n\r\n";
    stream.write_all(head.as_bytes()).await?;
    stream.flush().await?;

    let frame = |payload: String| -> Vec<u8> {
        let event = format!("data: {payload}\n\n");
        let mut out = format!("{:x}\r\n", event.len()).into_bytes();
        out.extend_from_slice(event.as_bytes());
        out.extend_from_slice(b"\r\n");
        out
    };

    let reasoning = serde_json::json!({
        "choices": [{ "delta": { "reasoning": "thinking…" } }]
    });
    stream.write_all(&frame(reasoning.to_string())).await?;
    stream.flush().await?;

    for chunk in content_chunks {
        let content = serde_json::json!({
            "choices": [{ "delta": { "content": chunk } }]
        });
        stream.write_all(&frame(content.to_string())).await?;
        stream.flush().await?;
    }

    // The body ends properly. No `[DONE]`, and nothing ever named a
    // `finish_reason` — the connection simply stopped having more to say.
    stream.write_all(b"0\r\n\r\n").await?;
    stream.flush().await?;
    Ok(())
}

async fn write_sse_stream(
    stream: &mut TcpStream,
    complete: bool,
    content_chunks: &[&str],
) -> std::io::Result<()> {
    let head = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Transfer-Encoding: chunked\r\n\
                Connection: close\r\n\r\n";
    stream.write_all(head.as_bytes()).await?;
    stream.flush().await?;

    let send_event = |payload: String| -> Vec<u8> {
        let event = format!("data: {payload}\n\n");
        let mut out = format!("{:x}\r\n", event.len()).into_bytes();
        out.extend_from_slice(event.as_bytes());
        out.extend_from_slice(b"\r\n");
        out
    };

    // First a reasoning delta, then content.
    let reasoning = serde_json::json!({
        "choices": [{ "delta": { "reasoning": "thinking…" } }]
    });
    stream.write_all(&send_event(reasoning.to_string())).await?;
    stream.flush().await?;

    if !complete {
        // Abort: drop the connection mid-stream without a terminating chunk.
        return Ok(());
    }

    for chunk in content_chunks {
        let content = serde_json::json!({
            "choices": [{ "delta": { "content": chunk } }]
        });
        stream.write_all(&send_event(content.to_string())).await?;
        stream.flush().await?;
    }

    let usage = serde_json::json!({
        "choices": [],
        "usage": { "prompt_tokens": 11, "completion_tokens": 5 }
    });
    stream.write_all(&send_event(usage.to_string())).await?;

    stream.write_all(&send_event("[DONE]".to_string())).await?;

    // Terminating zero-length chunk.
    stream.write_all(b"0\r\n\r\n").await?;
    stream.flush().await?;
    Ok(())
}

fn models_body(config: &MockConfig) -> String {
    let mut primary = serde_json::json!({
        "id": MODEL,
        "context_length": 8192u64,
        "pricing": {
            "per_prompt_token": { "value": 1u64, "scale_factor": 1u64 },
            "per_completion_token": { "value": 1u64, "scale_factor": 1u64 }
        }
    });
    if let Some(supported) = config.declared_tool_calling {
        primary["capabilities"] = serde_json::json!({
            "tool_calling": { "supported": supported },
            "reasoning": { "supported": false },
            "input_modalities": ["text"],
            "output_modalities": ["text"],
        });
        primary["max_output_tokens"] = serde_json::json!(4096u64);
        primary["output_budget_class"] = serde_json::json!("standard");
    }
    serde_json::json!({
        "data": [primary, {
            "id": ROUTER_REMOTE_MODEL,
            "context_length": 8192u64,
            "pricing": {
                "per_prompt_token": { "value": 1u64, "scale_factor": 1u64 },
                "per_completion_token": { "value": 1u64, "scale_factor": 1u64 }
            }
        }]
    })
    .to_string()
}

fn keys_body(issuer: &Issuer) -> String {
    let ds = format!("ACT-v1:{DS_ORG}:{DS_SERVICE}:{DS_DEPLOYMENT}:{DS_VERSION}");
    // Reproduce the production `/v1/keys` shape during a rotation grace
    // window: a just-retired key (out of its `issue_from..issue_until` issuing
    // window but still within `accept_until`) sorts *ahead* of the current
    // key. It carries a different public key, so a client that selected by
    // domain separator alone — every key shares one — would pick the decoy and
    // fail proof verification against the current issuer (the original bug).
    // The client must skip the out-of-window decoy and choose the current key.
    let decoy_pk = PrivateKey::random(OsRng)
        .public()
        .to_cbor()
        .expect("encode decoy public key");
    let decoy_key_hash: [u8; 32] = Sha256::digest(&decoy_pk).into();
    serde_json::json!({
        "data": [
            {
                "id": hex::encode(decoy_key_hash),
                "public_key": URL_SAFE_NO_PAD.encode(&decoy_pk),
                "domain_separator": ds,
                "issue_from": "2026-01-01T00:00:00Z",
                "issue_until": "2026-02-01T00:00:00Z",
                "accept_until": "2030-01-01T00:00:00Z",
            },
            {
                "id": issuer.key_id_hex,
                "public_key": issuer.public_key_b64(),
                "domain_separator": ds,
                "issue_from": "2026-02-01T00:00:00Z",
                "issue_until": "2030-01-01T00:00:00Z",
                "accept_until": "2030-01-01T00:00:00Z",
            }
        ]
    })
    .to_string()
}

fn error_body(message: &str) -> String {
    serde_json::json!({ "error": { "message": message } }).to_string()
}

fn uuid_like() -> String {
    // Deterministic-enough opaque id; the client only stores it.
    format!("{:016x}", rand_core::RngCore::next_u64(&mut OsRng))
}

async fn write_json(stream: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Status",
    };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {len}\r\n\
         Connection: close\r\n\r\n{body}",
        len = body.len(),
    );
    stream.write_all(response.as_bytes()).await?;
    stream.flush().await
}

/// The exact bytes axum's `JsonDataError` rejection produces for an unknown
/// body field: status 422, `text/plain`, and the rejection's `Display` text.
/// Mirrors `axum_core::__define_rejection!`'s `(status, body_text)` response.
pub const UNKNOWN_FIELD_REJECTION: &str = "Failed to deserialize the JSON body into the target type: unknown field `tools`, \
     expected one of `model`, `messages`, `max_completion_tokens`, `temperature`, `top_p`, \
     `stop`, `stream`, `stream_options`";

/// Write a `text/plain` response — what an axum extractor rejection is, and
/// what [`ChatBehavior::RejectToolsUnparseable`] answers with.
async fn write_plain_text(stream: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    let reason = match status {
        400 => "Bad Request",
        422 => "Unprocessable Entity",
        _ => "Status",
    };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: text/plain; charset=utf-8\r\n\
         Content-Length: {len}\r\n\
         Connection: close\r\n\r\n{body}",
        len = body.len(),
    );
    stream.write_all(response.as_bytes()).await?;
    stream.flush().await
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

// ---------------------------------------------------------------------------
// AppCore construction wired to the mock
// ---------------------------------------------------------------------------

/// Build a mock upstream plus an `AppCore` wired to it: a plain-HTTP client
/// (no attestation, via the `with_test_http_client` seam) with `base_url`
/// pointed at the mock. The mock listener is spawned on the core's own tokio
/// runtime so it shares the runtime that will drive the chat. Returns the mock,
/// the core, and the tempdir backing its config + data (kept alive by the
/// caller). This is the canonical entry point for chat-path tests.
pub fn core_for(config: MockConfig) -> (MockServer, AppCore, tempfile::TempDir) {
    // The injected plain client is built before `AppCore` (which installs the
    // rustls provider), so install it here first. Idempotent across tests.
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().expect("tempdir");
    let config_dir = dir.path().to_path_buf();
    let data_dir = dir.path().join("data");
    let client = reqwest::Client::builder()
        .build()
        .expect("plain http client");
    let core = AppCore::with_test_http_client(config_dir, data_dir, client).expect("open core");
    let mock = core.runtime().block_on(async { start(config).await });
    core.runtime()
        .block_on(core.set_base_url(mock.base_url.clone()))
        .expect("set base url");
    (mock, core, dir)
}

/// Re-open an `AppCore` over a directory a previous one used — the **restart**
/// seam.
///
/// The single-writer lock is held for the life of an `AppCore`, so the caller
/// must have dropped the previous one. Everything durable is on disk, so what
/// comes back is the same profile with none of the previous process's memory:
/// exactly the state a guard that is supposed to survive a restart has to be
/// held against.
pub fn reopen_core(dir: &tempfile::TempDir, base_url: &str) -> AppCore {
    let config_dir = dir.path().to_path_buf();
    let data_dir = dir.path().join("data");
    let client = reqwest::Client::builder()
        .build()
        .expect("plain http client");
    let core = AppCore::with_test_http_client(config_dir, data_dir, client).expect("reopen core");
    core.runtime()
        .block_on(core.set_base_url(base_url.to_string()))
        .expect("set base url");
    core
}

/// Configure account credentials so auto-provisioning can reach the balance /
/// allocate endpoints. The actual basic-auth password is never verified by the
/// mock, so any non-empty values work.
pub fn with_account(core: &AppCore) {
    core.set_account_credentials(uuid_account_id(), "mock-secret".into())
        .expect("set account credentials");
}

fn uuid_account_id() -> String {
    // The client serializes this verbatim into the Basic auth username; it must
    // be a syntactically valid UUID only if the *mock* parses it — ours does
    // not, so a fixed string suffices.
    "00000000-0000-0000-0000-000000000001".into()
}
