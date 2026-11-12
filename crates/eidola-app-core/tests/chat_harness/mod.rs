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
//!   the production server (`crates/eidola-server-gateway/src/credentials.rs`) uses —
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
use eidola_common::issuer_calendar::{ISSUER_CALENDAR, IssuerCalendar};
use eidola_common::spend_credentials::ACT_L;
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
    /// A **complete** 200 JSON completion followed by enough trailing
    /// whitespace to cross the blocking read's ceiling.
    ///
    /// The padded-complete-object shape, not a truncation: most oversized JSON
    /// truncates into something that does not parse, so a mock that merely sent
    /// a huge body would be indistinguishable from the cured behaviour. Here
    /// the retained prefix parses perfectly, which is exactly why the ceiling
    /// has to be asked about rather than inferred from the parse. Asked by a
    /// spending request, which the gateway answers with a stream, it is a
    /// stream whose data passes the read ceiling.
    OkBlockingPaddedPastCeiling,
    /// A **non-2xx** whose body is a complete JSON error document followed by
    /// enough whitespace to cross the read ceiling — in whichever transport
    /// asked. The status is the only thing that differs from
    /// [`ChatBehavior::OkBlockingPaddedPastCeiling`], which is the point: a
    /// reader that lets an unsuccessful status excuse a body from parsing must
    /// not let it excuse a fragment from the ceiling.
    Non2xxPaddedPastCeiling(u16),
    /// A `200` whose body is valid JSON and not a completion: an error document
    /// answered with a success status, carrying a refund.
    ///
    /// Syntax is not shape. This parses perfectly and has no `choices`, so a
    /// reader that stops at `from_str` hands a downstream tool an apparent
    /// success carrying nothing it can read as an answer. The refund rides it
    /// because the credential is spent either way — a body too shapeless to
    /// answer with can still hold the only copy of the token — and it is minted
    /// whatever [`RefundMode`] says, for the same reason
    /// [`ChatBehavior::StreamingWithMetadataRefund`] does.
    OkBlockingNotACompletion,
    /// A `200` whose body is not JSON at all — a truncated answer, or an
    /// intermediary's HTML error page. What a client must never read as a
    /// successful completion.
    OkNonJsonBody,
    /// 200 SSE stream: content + reasoning deltas, usage, `[DONE]`.
    OkStreaming,
    /// The same stream with **bare carriage returns** as line endings.
    ///
    /// Valid event-stream — the format takes `\r\n`, `\n` and `\r` — and the
    /// one a reader recognising only the first two never splits at all: every
    /// event piles into one frame until a ceiling or EOF. A backend choosing
    /// this is unusual, which is exactly why nothing else would notice.
    ///
    /// Its events are **multi-field** (`id:` ahead of `data:`) and name the
    /// model in the upstream's own spelling, so a consumer that separates
    /// events correctly and then parses their fields with `str::lines()` is
    /// caught too — see [`write_sse_stream_with_terminator`].
    OkStreamingBareCarriageReturns,
    /// The **real server's** streaming shape: content, a usage chunk, then a
    /// terminal metadata event (`object == "eidola.chat.completion.metadata"`)
    /// carrying the refund, then `[DONE]`.
    ///
    /// The refund rides that event **whatever [`RefundMode`] says**, because
    /// that is what the server does: its persistence of the token is
    /// best-effort and its failure is exactly what makes the recovery endpoint
    /// unable to answer, while the in-band copy still goes out. Pairing this
    /// with `RefundMode::Fail` is therefore the honest model of the one case
    /// where the in-band token is the *only* copy.
    StreamingWithMetadataRefund,
    /// [`ChatBehavior::StreamingWithMetadataRefund`] with **every JSON payload
    /// split across several `data:` fields** — one per line of its
    /// pretty-printed form — and each chunk naming the model in the upstream's
    /// own spelling ([`STREAM_WIRE_MODEL`]).
    ///
    /// Valid event-stream: an event's data is its `data:` fields joined with
    /// `\n`, and pretty-printed JSON joined back that way is the same JSON. A
    /// reader that parses field by field sees fragments that are each not JSON
    /// and silently skips them — the deltas, the model rewrite, and the refund in
    /// the metadata event all go missing without anything failing.
    StreamingSplitDataFields,
    /// [`ChatBehavior::StreamingWithMetadataRefund`] framed the way the Eidola
    /// server frames every stream it answers: through its own
    /// `eidola_server_gateway::padding::Framer`, as writes of exactly
    /// `FRAME_BYTES`, each the events waiting plus padding (a comment event,
    /// trailing whitespace on a JSON line, or blank lines after `[DONE]`), an
    /// event larger than a frame carried across frames, and frames of nothing
    /// but padding between the content deltas.
    StreamingPadded,
    /// [`ChatBehavior::StreamingPadded`] with **more padding than the read
    /// ceiling** between its content deltas — what a slow answer looks like
    /// when the frames keep coming at the frame rate. Ten megabytes of padding,
    /// spelled here rather than imported: the mock must not learn this app's
    /// ceilings.
    StreamingPaddedPastReadCeiling,
    /// A padded stream whose **answer is just under the read ceiling** —
    /// eight megabytes less four kilobytes of events that carry data — and
    /// whose last frames (several of nothing but padding, then the tail)
    /// arrive coalesced into one transport chunk, as an intermediary that
    /// buffers may deliver them. A reader that counts a transport chunk
    /// before telling its padding from its data crosses the ceiling here;
    /// one that counts only data does not.
    StreamingPaddedNearReadCeiling,
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
    /// A response head naming `status` and `content_type`, a `Content-Length`
    /// the body never reaches, and then the connection dropped — an upstream
    /// that reset **after** its head and **before** any stream could open.
    ///
    /// The body read fails in the transport, which is the whole point: a reader
    /// that defaults a failed read to an empty body turns this into a clean
    /// answer of nothing under whatever status the head claimed.
    HeadThenCut {
        status: u16,
        content_type: &'static str,
    },
    /// 200 SSE stream that the server aborts mid-event (writes a partial event,
    /// then drops the TCP connection). Exercises the mid-SSE read failure arm.
    StreamingMidAbort,
    /// A `200 text/event-stream` that never terminates an event: it writes far
    /// more than the proxy's per-event ceiling with no blank line anywhere, so
    /// nothing can ever be drained out of the frame accumulator. The shape a
    /// buffer with no ceiling of its own dies on.
    StreamingUnterminatedFlood,
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
    /// **The gateway cut the stream short** (its client left, its deadline,
    /// an engine failure): content deltas, no terminal `finish_reason`, then
    /// the metadata event carrying the refund the gateway settled on (at
    /// [`MockConfig::settled_charge`]), and no `[DONE]`. The answer did not
    /// finish; its refund is in band.
    StreamingCutShortWithRefund,
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
    /// A **refund-bearing** non-2xx, the shape the server answers with when a
    /// streaming request fails after the nullifier is recorded but before the
    /// SSE opens — request validation, `send_stream`, a spend-proof re-encode
    /// (`eidola-server-gateway/src/handlers.rs`: `error_response_with_refund`). The
    /// credential is spent and the only copy of its refund may be in this body,
    /// because the server's own persistence of the token is best-effort.
    Non2xxWithRefund(u16),
    /// A `200` SSE stream carrying **more bytes than the Record keeps**, so a
    /// truncated recording is what a reader must be told about rather than
    /// handed silently — in events each well under the frame ceiling, so the
    /// stream itself is one a reader accepts.
    OkStreamingOversized,
    /// A `200` SSE stream whose one content event is **larger than the frame
    /// ceiling and properly terminated**. Not
    /// [`ChatBehavior::StreamingUnterminatedFlood`]: the blank line that ends
    /// it arrives, so a reader that measures only what is left after draining
    /// complete events takes this one whole.
    StreamingOneOversizedEvent,
    /// A `200` SSE stream of **ordinary, well-framed events that never stops
    /// coming** — far more content than any completion, each event small, and
    /// a `[DONE]` only at the very end.
    ///
    /// Not [`ChatBehavior::StreamingUnterminatedFlood`]: every event here ends
    /// properly, so a frame-buffer ceiling never trips. What has to end it is a
    /// ceiling on the stream as a whole, because a turn assembles the deltas
    /// into an answer in memory and a stream without end is an answer without
    /// end. The total is spelled here rather than imported — the mock is the
    /// peer and must not learn this app's ceilings.
    StreamingPastReadCeiling,
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
    /// (`crates/eidola-server-gateway/src/handlers.rs`). Axum renders `JsonDataError`
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
    /// and then a regeneration that hits the ceiling — the case where the
    /// generation switcher is what keeps the readable answer.
    OkThenReasoningOnlyLength,
    /// **An answer cut off at the ceiling, and its continuation**, in whichever
    /// transport asked. A request without `continue_final_message` stops at
    /// the ceiling as `cut` says ([`Cut`]); one with it is answered from inside
    /// the field it continues: the rest of the reasoning
    /// ([`CONTINUED_REASONING`]) and then the answer ([`CONTINUED_ANSWER`]) for
    /// `"reasoning_content"`, the rest of the answer ([`CONTINUED_TAIL`]) for
    /// `true`. With `again`, the first continuation (the turn's second chat
    /// request) stops at the ceiling too, before any answer when it continued
    /// the reasoning — so a continuation is continued.
    ContinueAfterLength { cut: Cut, again: bool },
    /// A reasoning-only cut whose continuation calls the harness's tool (one
    /// round, [`TOOL_NAME`]), after which the ordinary request that follows
    /// the tool round answers [`TOOL_FINAL_CONTENT`]. Whichever transport
    /// asked.
    ContinueIntoToolRound,
    /// An answer opening with a mimicked header line ([`MIMICKED_HEADER`]) and
    /// then cut off at the ceiling, which the engine reports `continuable` —
    /// the strip this app applies is what makes the stored text differ.
    /// Whichever transport asked.
    HeaderThenLength,
}

/// Where a [`ChatBehavior::ContinueAfterLength`] answer stops the first time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cut {
    /// Reasoning only: [`TRUNCATED_REASONING`], no answer.
    Reasoning,
    /// Mid-answer: [`TRUNCATED_REASONING`], then [`STREAM_CONTENT`].
    Answer,
    /// Mid-answer, where the engine's parser dropped what it was cut inside of
    /// (an unfinished tool call): the same text as [`Cut::Answer`], reported
    /// `continuable: false`.
    Dropped,
}

/// The rest of a continued reasoning.
pub const CONTINUED_REASONING: &str = " and so: the moon.";
/// The answer a continued reasoning reaches.
pub const CONTINUED_ANSWER: &str = "The tides follow the moon.";
/// The rest of a continued answer.
pub const CONTINUED_TAIL: &str = " Then it carries on.";

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

/// A model the mock's catalog prices **per request** — the flat-pricing shape
/// (zero token rates, the price in `per_request`).
pub const FLAT_MODEL: &str = "flat-priced";
/// The flat price, in credits, [`FLAT_MODEL`] costs per request.
pub const FLAT_PRICE: u64 = 7;

/// The idle TTL [`supported_prompt_cache`] declares, in seconds.
pub const CACHE_IDLE_TTL_SECS: i64 = 900;
/// The maximum key age [`supported_prompt_cache`] declares, in seconds.
pub const CACHE_MAX_AGE_SECS: i64 = 7200;

/// The `prompt_cache` leaf an Eidola-hosted catalog row carries: its engine
/// reuses a prompt prefix between requests sharing a `cache_key`, under the
/// engine core's own default retention.
pub fn supported_prompt_cache() -> serde_json::Value {
    serde_json::json!({
        "supported": true,
        "idle_ttl_secs": CACHE_IDLE_TTL_SECS,
        "max_age_secs": CACHE_MAX_AGE_SECS,
    })
}

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
    /// A **complete** JSON completion whose assistant content is this string,
    /// followed by megabytes of trailing whitespace — so the first bytes parse
    /// perfectly while the body as a whole is past the ceiling this app reads.
    /// The shape a caller that discards `over_ceiling` accepts as a whole
    /// answer: truncation lands in the padding and the object still parses.
    /// Asked by a spending request, which the gateway answers with a stream,
    /// it is a stream whose data passes that ceiling.
    ReplyPaddedPastCeiling(String),
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

/// The model ids the mock's catalog leaves out (see
/// [`MockConfig::catalog_omits`]). Shared + interior-mutable so a row can be
/// retired mid-test, after something has already selected it.
pub type CatalogOmissions = Arc<std::sync::Mutex<Vec<String>>>;

/// A fresh catalog with nothing left out.
pub fn catalog_omissions() -> CatalogOmissions {
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
    /// **The server whose own persistence of the refund token failed.** Every
    /// arm that carries a refund in band carries it — the server sends the
    /// token whether or not it managed to store it — and
    /// `/v1/credentials/refund` answers the permanent `404` the real endpoint
    /// gives for a token it never stored.
    ///
    /// The one configuration in which the in-band copy is the *only* copy: a
    /// client that discards it and asks recovery instead strands the credential
    /// in `spending` for good, which `Fail` cannot model (it omits the in-band
    /// token too) and `Succeed` cannot detect (recovery would answer).
    NotStored,
    /// **The request is still being served, then its gateway crashed.** The
    /// first `n` calls to `/v1/credentials/refund` answer with the gateway's
    /// provisional refund: a refund of nothing (`t = 0`), marked
    /// `"provisional": true`, as the real endpoint does for a spend whose
    /// request may still end. Every later call answers the same refund marked
    /// `"provisional": false`, as the real endpoint does once no request is
    /// serving the spend any more and the provisional refund is its last
    /// word. Inline refunds are omitted, as for `Fail`.
    ProvisionalFirst(u64),
    /// The first `n` calls to `/v1/credentials/refund` succeed and every
    /// later one never answers: a recovery that stalls part-way through a
    /// request's inputs. Inline refunds are omitted.
    StallAfter(u64),
    /// **The gateway settled a stream it cut short on what was generated.**
    /// `/v1/credentials/refund` answers a final refund returning the hold less
    /// `n` credits (the prompt and the output generated before the cut), as
    /// the real endpoint does for a stream that ended early (its client went
    /// away, it reached its deadline, or its engine failed). Inline refunds
    /// are omitted, as for `Fail`.
    ChargedEarly(u128),
}

/// The weights hash the mock catalog publishes for [`MODEL`] (see
/// [`MockConfig::published_sha256`]).
pub type PublishedSha = Arc<std::sync::Mutex<Option<String>>>;

/// A one-shot non-2xx for the next chat request (see
/// [`MockConfig::fail_next_chat`]).
pub type FailNextChat = Arc<std::sync::Mutex<Option<u16>>>;

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
    /// Model ids `GET /v1/models` leaves out of an otherwise healthy catalog.
    ///
    /// Shared and interior-mutable so a test can retire a row **after** a
    /// selection has been made against it, which is the only order the real
    /// case ever happens in: a catalog row goes away and nothing un-picks the
    /// participant, the space router or the recorded model a regeneration
    /// replays. Empty by default, so every other test sees the full catalog.
    pub catalog_omits: CatalogOmissions,
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
    /// The `capabilities.prompt_cache` leaf `GET /v1/models` publishes for
    /// [`MODEL`], verbatim. `None` — the default — publishes none, which is
    /// every model the server sells today; [`supported_prompt_cache`] is the
    /// leaf an Eidola-hosted row carries.
    pub declared_prompt_cache: Option<serde_json::Value>,
    /// The `max_output_tokens` `GET /v1/models` declares for [`MODEL`].
    /// `None` — the default — declares none (unless `declared_tool_calling`
    /// publishes its standard row), which leaves a turn on the fallback
    /// ceiling every test written before declarations were read expects.
    pub declared_max_output: Option<u64>,
    /// The weights hash `GET /v1/models` publishes for [`MODEL`], with
    /// [`MODEL`] as its tag. Shared and interior-mutable so a test can move
    /// the tag onto other weights between turns, as a release does. `None` —
    /// the default — publishes none, the shape of a gateway older than
    /// hash-keyed models.
    pub published_sha256: PublishedSha,
    /// Weights `GET /v1/models` also lists, under no tag: the row a release
    /// leaves behind when it moves [`MODEL`]'s tag to other weights while the
    /// earlier ones are still served. Otherwise [`MODEL`]'s row. `None` — the
    /// default — lists none.
    pub untagged_sha256: PublishedSha,
    /// When set, the **next** chat request is answered with this non-2xx
    /// status (an error document) and the setting clears, whatever `chat`
    /// says — so a test can fail one turn between ordinary ones.
    pub fail_next_chat: FailNextChat,
    /// Armed by a test to make the mock **stop listening** once it has served
    /// the next `GET /v1/models` — so the turn that fetched the catalog then
    /// meets a refused connection for its completion, the one failure that
    /// provably sends nothing.
    pub close_switch: CloseSwitch,
    /// List [`FLAT_MODEL`], the flat-priced entry, in the catalog. Opt-in so
    /// the listings every other test pins are unchanged.
    pub list_flat_model: bool,
    /// How long a chat request is held before it is answered.
    ///
    /// A real model request takes time — that is the whole reason a surface
    /// needs a pending state, and the reason two clicks can overlap. The mock
    /// answers instantly, which makes an in-flight turn unobservable, so a test
    /// about *concurrency* asks for a window it can act inside.
    pub chat_delay_ms: u64,
    /// How long a completion answered as the gateway's stream runs after its
    /// response headers and before its first delta: the time a model takes to
    /// generate, which a stream's opening does not wait for.
    pub answer_ms: u64,
    /// When set, the issuance response names this issuer key id instead of the
    /// key that actually signed it. `None` — the default — names the real key.
    pub issued_key_id: Option<String>,
    /// When set, every refund (in band and recovered) names this issuer key id
    /// instead of the key that actually signed it. `None` names the real key.
    pub refund_key_id: Option<String>,
    /// The charge a completed request is settled at, consumed from its inputs
    /// in input order as the gateway consumes it (clamped to the hold).
    /// `None` — the default — charges nothing: every input's share returns
    /// whole. Applies to the in-band refunds of a completed request; a
    /// refusal or a recovery returns every share whole as before.
    pub settled_charge: Option<u128>,
    /// When set, issuance signs under this request context instead of the one
    /// every credential under the key shares. `None` uses the shared one.
    pub issuance_context: Option<Scalar>,
    /// Credits added to what the client asked for at issuance. Zero issues
    /// exactly the requested amount.
    pub issuance_extra_credits: i64,
    /// The current issuer key's `(issue_from, issue_until, accept_until)` as
    /// `GET /v1/keys` advertises them. `None` — the default — is the current
    /// epoch's windows on [`MockConfig::issuer_calendar`], so no test written
    /// before key deadlines mattered can meet one. A test about a key's
    /// deadlines sets them relative to now ([`KeyWindow::epoch_ending_at`]),
    /// with the calendar that puts them on one.
    pub key_window: Option<KeyWindow>,
    /// The issuer calendar the mock lists windows on and the client checks
    /// them against (`AppCore::test_set_issuer_calendar`, applied by
    /// [`core_for`]). `None` is the production calendar.
    pub issuer_calendar: Option<IssuerCalendar>,
    /// Switch the mock to a second, real issuer key: listed as current and
    /// signing issuances, in the same epoch as the first. What a server
    /// rotating within an epoch (or singling a client out) looks like.
    pub key_swap: KeySwap,
    /// Stop listing the previous epoch's key: a listing that leaves out the
    /// one entry a key reused from it would otherwise collide with.
    pub omit_previous_key: Arc<std::sync::atomic::AtomicBool>,
    /// How `POST /v1/account/credentials` misbehaves, switchable mid-test
    /// (see [`IssuanceKnobs`]). The default answers every request.
    pub issuance: IssuanceKnobs,
    /// Answer a spending request that asked for a stream with whatever the
    /// behaviour writes, as a backend that ignored `stream: true` does.
    /// `false` — the default — answers it as the gateway does: every
    /// completion a behaviour writes goes out as a stream.
    pub ignores_stream: bool,
}

/// See [`MockConfig::key_swap`].
#[derive(Clone, Default)]
pub struct KeySwap(Arc<KeySwapState>);

#[derive(Default)]
struct KeySwapState {
    on: std::sync::atomic::AtomicBool,
    second: std::sync::OnceLock<Issuer>,
}

impl KeySwap {
    /// From now on, list and issue under the second key.
    pub fn swap(&self) {
        self.0.on.store(true, Ordering::SeqCst);
    }

    /// The second key's id.
    pub fn second_key_id(&self) -> String {
        self.second().key_id_hex.clone()
    }

    fn second(&self) -> &Issuer {
        self.0.second.get_or_init(Issuer::new)
    }

    /// The issuer whose key has id `key_id`: the first, or the second if it
    /// was ever made. A refund is signed under the key its input names.
    fn issuer_for<'a>(&'a self, first: &'a Issuer, key_id: &str) -> Option<&'a Issuer> {
        if first.key_id_hex == key_id {
            return Some(first);
        }
        self.0
            .second
            .get()
            .filter(|second| second.key_id_hex == key_id)
    }

    /// The issuer that lists and issues: the second one once swapped.
    fn active<'a>(&'a self, first: &'a Issuer) -> &'a Issuer {
        if self.0.on.load(Ordering::SeqCst) {
            self.second()
        } else {
            first
        }
    }
}

/// Switches on the mock's issuance endpoint, shared with the running server so
/// a test can flip one between two requests.
#[derive(Clone, Default)]
pub struct IssuanceKnobs(Arc<IssuanceKnobState>);

#[derive(Default)]
struct IssuanceKnobState {
    /// How many upcoming issuances are recorded (debited and stored, as the
    /// real server does) and then answered with nothing: the connection is
    /// closed before a response is written — a lost response.
    lose: std::sync::atomic::AtomicU32,
    /// How many upcoming issuances are signed and then held, neither debited
    /// nor stored, and answered with nothing: an original attempt that
    /// stalled before its debit, which lands only when a test calls
    /// [`MockServer::land_held_issuances`].
    hold: std::sync::atomic::AtomicU32,
    /// When non-zero, every issuance not already stored is refused with this
    /// status before anything is debited.
    refuse_with: std::sync::atomic::AtomicU32,
    /// Answer every retry — a request whose idempotency key is already
    /// stored — with `503`, while fresh issuances still go through: a lost
    /// response whose recovery keeps failing.
    fail_retries: std::sync::atomic::AtomicBool,
    /// Accept every issuance request and never answer it.
    stall: std::sync::atomic::AtomicBool,
    /// Refuse every issuance request with `401` before anything else, as the
    /// gateway's `BasicAuth` does for a wrong secret.
    unauthorized: std::sync::atomic::AtomicBool,
}

impl IssuanceKnobs {
    /// Lose the responses of the next `n` issuances (see [`IssuanceKnobState`]).
    pub fn lose_next(&self, n: u32) {
        self.0.lose.store(n, Ordering::SeqCst);
    }
    /// Hold the next `n` issuances before their debit (see
    /// [`IssuanceKnobState`]).
    pub fn hold_next(&self, n: u32) {
        self.0.hold.store(n, Ordering::SeqCst);
    }
    /// Refuse every fresh issuance with `status`; `None` answers again.
    pub fn refuse_with(&self, status: Option<u16>) {
        self.0
            .refuse_with
            .store(status.map(u32::from).unwrap_or(0), Ordering::SeqCst);
    }
    /// Fail every retry (see [`IssuanceKnobState`]); `false` answers again.
    pub fn fail_retries(&self, on: bool) {
        self.0.fail_retries.store(on, Ordering::SeqCst);
    }
    /// Refuse every issuance as unauthenticated (see [`IssuanceKnobState`]).
    pub fn unauthorized(&self, on: bool) {
        self.0.unauthorized.store(on, Ordering::SeqCst);
    }
    /// Stall every issuance request (see [`IssuanceKnobState`]).
    pub fn stall(&self, on: bool) {
        self.0.stall.store(on, Ordering::SeqCst);
    }
    fn take_hold(&self) -> bool {
        self.0
            .hold
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok()
    }
    fn take_loss(&self) -> bool {
        self.0
            .lose
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok()
    }
    fn refusal(&self) -> Option<u16> {
        match self.0.refuse_with.load(Ordering::SeqCst) {
            0 => None,
            status => Some(status as u16),
        }
    }
}

/// An issuer key's advertised window, as the ISO 8601 strings `/v1/keys`
/// carries (see [`MockConfig::key_window`]).
#[derive(Clone, Debug)]
pub struct KeyWindow {
    pub issue_from: String,
    pub issue_until: String,
    pub accept_until: String,
}

impl KeyWindow {
    /// The windows of the epoch that ends issuing at `issue_until`, on a
    /// calendar of `epoch_secs`-long epochs aligned to it: the pair a test
    /// about a key's deadlines needs, since the client refuses any key whose
    /// windows are not on its calendar.
    pub fn epoch_ending_at(issue_until: i64, epoch_secs: u64) -> (Self, IssuerCalendar) {
        let until = u64::try_from(issue_until).expect("a deadline after 1970");
        let calendar = IssuerCalendar {
            anchor_unix_secs: until % epoch_secs,
            epoch_secs,
        };
        let epoch = calendar.epoch_at(until).expect("on the calendar") - 1;
        (Self::of_epoch(&calendar, epoch), calendar)
    }

    /// Epoch `epoch`'s windows on `calendar`.
    pub fn of_epoch(calendar: &IssuerCalendar, epoch: u64) -> Self {
        let w = calendar.windows(epoch).expect("epoch in range");
        let secs = |s: u64| i64::try_from(s).expect("in range");
        Self::from_unix_secs(
            secs(w.issue_from),
            secs(w.issue_until),
            secs(w.accept_until),
        )
    }

    /// A window given as unix seconds.
    pub fn from_unix_secs(issue_from: i64, issue_until: i64, accept_until: i64) -> Self {
        use eidola_server_gateway::helpers::unix_to_iso;
        Self {
            issue_from: unix_to_iso(issue_from),
            issue_until: unix_to_iso(issue_until),
            accept_until: unix_to_iso(accept_until),
        }
    }
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            chat: ChatBehavior::OkBlocking,
            ignores_stream: false,
            router: RouterBehavior::Passthrough,
            summary: SummaryBehavior::Passthrough,
            refund: RefundMode::Succeed,
            balance: 10_000_000,
            models_status: None,
            catalog_omits: catalog_omissions(),
            tool_script: tool_script(),
            declared_tool_calling: None,
            declared_prompt_cache: None,
            declared_max_output: None,
            published_sha256: PublishedSha::default(),
            untagged_sha256: PublishedSha::default(),
            fail_next_chat: FailNextChat::default(),
            close_switch: CloseSwitch::default(),
            list_flat_model: false,
            chat_delay_ms: 0,
            answer_ms: 0,
            issued_key_id: None,
            refund_key_id: None,
            settled_charge: None,
            issuance_context: None,
            issuance_extra_credits: 0,
            key_window: None,
            issuance: IssuanceKnobs::default(),
            issuer_calendar: None,
            key_swap: KeySwap::default(),
            omit_previous_key: Arc::default(),
        }
    }
}

/// The issuer key id a refund names: the real one unless the test asked for
/// another ([`MockConfig::refund_key_id`]).
fn refund_key_id(issuer: &Issuer, config: &MockConfig) -> String {
    config
        .refund_key_id
        .clone()
        .unwrap_or_else(|| issuer.key_id_hex.clone())
}

/// See [`MockConfig::close_switch`].
#[derive(Clone, Default)]
pub struct CloseSwitch(Arc<CloseSwitchState>);

#[derive(Default)]
struct CloseSwitchState {
    armed: std::sync::atomic::AtomicBool,
    tx: std::sync::Mutex<
        Option<tokio::sync::mpsc::UnboundedSender<tokio::sync::oneshot::Sender<()>>>,
    >,
}

impl CloseSwitch {
    /// Close the listener after the next catalog fetch.
    pub fn arm(&self) {
        self.0.armed.store(true, Ordering::SeqCst);
    }

    fn install(&self, tx: tokio::sync::mpsc::UnboundedSender<tokio::sync::oneshot::Sender<()>>) {
        *self.0.tx.lock().unwrap() = Some(tx);
    }

    async fn trip_if_armed(&self) {
        if !self.0.armed.swap(false, Ordering::SeqCst) {
            return;
        }
        let (ack, closed) = tokio::sync::oneshot::channel();
        let sent = self
            .0
            .tx
            .lock()
            .unwrap()
            .as_ref()
            .map(|tx| tx.send(ack).is_ok());
        if sent == Some(true) {
            let _ = closed.await;
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
    /// The exact bytes of every `POST /v1/chat/completions` body, in arrival
    /// order — for the assertions that are about bytes rather than JSON
    /// values (a body that must be byte-for-byte what it was before).
    chat_raw_bodies: Arc<std::sync::Mutex<Vec<Vec<u8>>>>,
    /// Whether each `POST /v1/chat/completions` carried an `Authorization`
    /// header, in arrival order — local turns must send none (no spend).
    chat_auths: Arc<std::sync::Mutex<Vec<bool>>>,
    /// The raw `Authorization` header of each `POST /v1/chat/completions`,
    /// in arrival order — an external backend's `Bearer` key; a spend never
    /// travels in a header.
    chat_auth_values: Arc<std::sync::Mutex<Vec<Option<String>>>>,
    /// The `token` of each `POST /v1/chat/completions` body's `credentials`
    /// elements, in input order, per request in arrival order — empty for a
    /// request that spent nothing (a local or external turn).
    chat_spends: Arc<std::sync::Mutex<Vec<Vec<String>>>>,
    /// Number of `POST /v1/credentials/refund` requests received.
    refund_hits: Arc<AtomicU64>,
    /// The mock issuer, for what it remembers about spends.
    issuer: Arc<Issuer>,
    /// Number of `GET /v1/models` requests received — the catalogue fetch a
    /// turn's *setup* makes, and so the earliest thing on the wire that says
    /// preparation got as far as building a client. Tests that assert a turn
    /// was refused before it did any work read this rather than `chat_hits`,
    /// which only notices once the turn is fully prepared.
    models_hits: Arc<AtomicU64>,
    /// The issuance switches the server was started with.
    issuance: IssuanceKnobs,
    _task: tokio::task::JoinHandle<()>,
}

impl MockServer {
    pub fn chat_hits(&self) -> u64 {
        self.chat_hits.load(Ordering::SeqCst)
    }
    pub fn refund_hits(&self) -> u64 {
        self.refund_hits.load(Ordering::SeqCst)
    }
    /// How many distinct spends arrived carrying a refund recovery key (a
    /// spend without one is refused, as the real server refuses it).
    pub fn keyed_spends(&self) -> usize {
        self.issuer.keyed_spends()
    }
    /// Spends whose hold was not exactly the request's worst-case cost (each
    /// refused, as the gateway refuses it).
    /// Spending chat requests that did not ask for a stream: each refused,
    /// as the gateway refuses it, before anything is taken.
    pub fn blocking_asks(&self) -> u64 {
        self.issuer.blocking_asks.load(Ordering::SeqCst)
    }

    pub fn hold_mismatches(&self) -> u64 {
        self.issuer.hold_mismatches.load(Ordering::SeqCst)
    }
    /// The issuance endpoint's switches (see [`MockConfig::issuance`]).
    pub fn issuance_knobs(&self) -> IssuanceKnobs {
        self.issuance.clone()
    }
    /// Issuance requests received, retries included.
    pub fn issuance_requests(&self) -> u64 {
        self.issuer.issuance_requests.load(Ordering::SeqCst)
    }
    /// Issuances that debited the account: a retry answered from what was
    /// stored is not one.
    pub fn issuance_debits(&self) -> u64 {
        self.issuer.issuance_debits.load(Ordering::SeqCst)
    }
    /// The credits converted by every debiting issuance.
    pub fn issuance_debited_credits(&self) -> i64 {
        self.issuer.issuance_debited_credits.load(Ordering::SeqCst) as i64
    }
    /// Debit and store every issuance [`IssuanceKnobs::hold_next`] held, as
    /// a stalled original reaching its debit late would.
    pub fn land_held_issuances(&self) {
        let held: Vec<_> = std::mem::take(&mut *self.issuer.held.lock().unwrap());
        for held in held {
            self.issuer.record_issuance(held);
        }
    }
    /// Catalogue fetches (see `models_hits`).
    pub fn models_hits(&self) -> u64 {
        self.models_hits.load(Ordering::SeqCst)
    }
    /// The recorded chat request bodies (see `chat_bodies`).
    pub fn chat_bodies(&self) -> Vec<serde_json::Value> {
        self.chat_bodies.lock().unwrap().clone()
    }
    /// The recorded chat request bodies as sent (see `chat_raw_bodies`).
    pub fn chat_raw_bodies(&self) -> Vec<Vec<u8>> {
        self.chat_raw_bodies.lock().unwrap().clone()
    }
    /// The `cache_key` each chat request carried, in arrival order.
    pub fn chat_cache_keys(&self) -> Vec<Option<String>> {
        self.chat_bodies()
            .iter()
            .map(|b| {
                b.get("cache_key")
                    .and_then(|k| k.as_str())
                    .map(str::to_owned)
            })
            .collect()
    }
    /// Per-chat-request `Authorization` presence (see `chat_auths`).
    pub fn chat_auths(&self) -> Vec<bool> {
        self.chat_auths.lock().unwrap().clone()
    }
    /// Per-chat-request raw `Authorization` values (see `chat_auth_values`).
    pub fn chat_auth_values(&self) -> Vec<Option<String>> {
        self.chat_auth_values.lock().unwrap().clone()
    }
    /// Per-chat-request first-input spend tokens (see `chat_spends`).
    pub fn chat_spends(&self) -> Vec<Option<String>> {
        self.chat_inputs()
            .into_iter()
            .map(|tokens| tokens.into_iter().next())
            .collect()
    }
    /// Per-chat-request spend tokens, every input in input order (see
    /// `chat_spends`).
    pub fn chat_inputs(&self) -> Vec<Vec<String>> {
        self.chat_spends.lock().unwrap().clone()
    }
    /// Per-chat-request shares, every input in input order: what each spend
    /// proof revealed, read off the wire the way the server reads it.
    pub fn chat_shares(&self) -> Vec<Vec<u128>> {
        self.chat_inputs()
            .iter()
            .map(|tokens| {
                tokens
                    .iter()
                    .filter_map(|token| Presented::from_token(token))
                    .filter_map(|input| scalar_to_credit::<ACT_L>(&input.proof.charge()).ok())
                    .collect()
            })
            .collect()
    }
    /// Per-chat-request nullifiers (hex), every input in input order: what
    /// the gateway records, so no two inputs anywhere may share one.
    pub fn chat_nullifiers(&self) -> Vec<Vec<String>> {
        self.chat_inputs()
            .iter()
            .map(|tokens| {
                tokens
                    .iter()
                    .filter_map(|token| Presented::from_token(token))
                    .map(|input| hex::encode(input.proof.nullifier().as_bytes()))
                    .collect()
            })
            .collect()
    }
    /// Per-chat-request issuer key ids, every input in input order.
    pub fn chat_input_keys(&self) -> Vec<Vec<String>> {
        self.chat_inputs()
            .iter()
            .map(|tokens| {
                tokens
                    .iter()
                    .filter_map(|token| Presented::from_token(token))
                    .map(|input| input.key_id)
                    .collect()
            })
            .collect()
    }
    /// The hold each `POST /v1/chat/completions` presented (the sum of its
    /// inputs' shares), in arrival order — `None` for a request that spent
    /// nothing. This is the hold the client sized, read off the wire the way
    /// the server reads it.
    pub fn chat_charges(&self) -> Vec<Option<u128>> {
        self.chat_shares()
            .into_iter()
            .map(|shares| (!shares.is_empty()).then(|| shares.iter().sum()))
            .collect()
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

/// The label of the harness profile's default agent — `db::default_agent_label`
/// over [`MODEL`], the model [`core_for`] seats it on.
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

/// An issuance signed and ready to debit: its idempotency key, the digest a
/// retry must match, the response body, and the credits it converts.
struct SignedIssuance {
    idem: String,
    request_digest: String,
    body: String,
    credits: i64,
}

/// The mock issuer: an ACT keypair plus the params/context the server derives.
struct Issuer {
    key: PrivateKey,
    params: Params,
    key_id_hex: String,
    request_context_scalar: Scalar,
    /// The refund recovery key each spend arrived with, by nullifier: the
    /// key the real server seals that spend's stored refund under, and so
    /// the only one its recovery endpoint opens it for.
    spend_keys: std::sync::Mutex<std::collections::HashMap<Vec<u8>, [u8; 32]>>,
    /// Idempotent issuance, as the real server keeps it: idempotency key →
    /// (the request it answered, the response body it was given).
    issuances: std::sync::Mutex<std::collections::HashMap<String, (String, String)>>,
    /// Issuances signed but not yet debited (see [`IssuanceKnobs::hold_next`]).
    held: std::sync::Mutex<Vec<SignedIssuance>>,
    /// Every `POST /v1/account/credentials` received.
    issuance_requests: AtomicU64,
    /// Every issuance that debited (a fresh one, not a stored answer).
    issuance_debits: AtomicU64,
    /// The credits all those debits converted.
    issuance_debited_credits: AtomicU64,
    /// A public key listed as the previous epoch's, for an epoch the mock
    /// never listed a current key for: stable for the mock's life.
    decoy_public_key: Vec<u8>,
    /// The key (id, base64 public key) the mock listed as current for each
    /// epoch, first listing wins — what it lists as that epoch's key later.
    listed: std::sync::Mutex<std::collections::HashMap<u64, (String, String)>>,
    /// Spends whose hold was not exactly the request's worst-case cost.
    hold_mismatches: AtomicU64,
    /// Spending chat requests that did not ask for a stream, which the
    /// gateway refuses.
    blocking_asks: AtomicU64,
}

impl Issuer {
    /// Debit for a signed issuance and store its response, once per
    /// idempotency key, as the real server's `record_credential_issuance`.
    fn record_issuance(&self, signed: SignedIssuance) {
        let mut issuances = self.issuances.lock().unwrap();
        if issuances.contains_key(&signed.idem) {
            return;
        }
        self.issuance_debits.fetch_add(1, Ordering::SeqCst);
        self.issuance_debited_credits
            .fetch_add(signed.credits.max(0) as u64, Ordering::SeqCst);
        issuances.insert(signed.idem, (signed.request_digest, signed.body));
    }

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
            spend_keys: std::sync::Mutex::new(std::collections::HashMap::new()),
            issuances: std::sync::Mutex::new(std::collections::HashMap::new()),
            held: std::sync::Mutex::new(Vec::new()),
            issuance_requests: AtomicU64::new(0),
            issuance_debits: AtomicU64::new(0),
            issuance_debited_credits: AtomicU64::new(0),
            decoy_public_key: PrivateKey::random(OsRng)
                .public()
                .to_cbor()
                .expect("encode decoy public key"),
            listed: std::sync::Mutex::new(std::collections::HashMap::new()),
            hold_mismatches: AtomicU64::new(0),
            blocking_asks: AtomicU64::new(0),
        }
    }

    /// Remember the refund recovery key a spend (or a recovery that found no
    /// spend) arrived with, and say whether `key` is the one on record: what
    /// decides whether the real server's stored copy opens.
    fn spend_key_matches(&self, proof: &SpendProof<ACT_L>, key: [u8; 32]) -> bool {
        let nullifier = proof.nullifier().as_bytes().to_vec();
        let mut keys = self.spend_keys.lock().unwrap();
        *keys.entry(nullifier).or_insert(key) == key
    }

    /// How many distinct spends arrived with a refund recovery key.
    fn keyed_spends(&self) -> usize {
        self.spend_keys.lock().unwrap().len()
    }

    fn public_key_b64(&self) -> String {
        URL_SAFE_NO_PAD.encode(self.key.public().to_cbor().expect("encode public key"))
    }

    /// Issue a credential for a base64url CBOR `IssuanceRequest` and credit
    /// amount. Returns the base64url CBOR `IssuanceResponse`.
    fn issue(
        &self,
        issuance_request_b64: &str,
        credits: i64,
        context: Option<Scalar>,
    ) -> Option<String> {
        let req_cbor = URL_SAFE_NO_PAD.decode(issuance_request_b64).ok()?;
        let req = IssuanceRequest::from_cbor(&req_cbor).ok()?;
        let credit_scalar = credit_to_scalar::<ACT_L>(credits as u128).ok()?;
        let resp = self
            .key
            .issue::<ACT_L>(
                &self.params,
                &req,
                credit_scalar,
                context.unwrap_or(self.request_context_scalar),
                OsRng,
            )
            .ok()?;
        Some(URL_SAFE_NO_PAD.encode(resp.to_cbor().ok()?))
    }

    /// Produce a refund returning `t` credits of a parsed spend proof's
    /// charge. Returns the base64url CBOR `Refund`.
    fn refund_of(&self, spend_proof: &SpendProof<ACT_L>, t: u128) -> Option<String> {
        if spend_proof.context() != self.request_context_scalar {
            return None;
        }
        let t = credit_to_scalar::<ACT_L>(t).ok()?;
        let refund = self
            .key
            .refund::<ACT_L>(&self.params, spend_proof, t, OsRng)
            .ok()?;
        Some(URL_SAFE_NO_PAD.encode(refund.to_cbor().ok()?))
    }
}

/// One input a request presented: its spend proof and the issuer key id its
/// Token names.
struct Presented {
    proof: SpendProof<ACT_L>,
    key_id: String,
}

impl Presented {
    /// Parse a `credentials` element's `token` (mirrors the server's
    /// `auth::presented_spends`). Like the server, it reads only the one
    /// spelling, padded base64url, decoded here with the `base64` crate's
    /// canonical padded engine and a re-encoding check rather than the shared
    /// `eidola_common::spend_credentials` rule the client writes with, so the
    /// client's spelling is checked against an independent reading.
    fn from_token(token: &str) -> Option<Self> {
        let bytes = base64::engine::general_purpose::URL_SAFE
            .decode(token)
            .ok()
            .filter(|bytes| base64::engine::general_purpose::URL_SAFE.encode(bytes) == token)?;
        if bytes.len() < 67 {
            return None;
        }
        // [token_type(2)][challenge_digest(32)][issuer_key_id(32)][spend_proof(..)]
        Some(Self {
            proof: SpendProof::<ACT_L>::from_cbor(&bytes[66..]).ok()?,
            key_id: hex::encode(&bytes[34..66]),
        })
    }

    fn share(&self) -> Option<u128> {
        scalar_to_credit::<ACT_L>(&self.proof.charge()).ok()
    }
}

/// One refund per input, in input order, each signed under the key its input
/// names and returning that input's partial return when the request is
/// charged `charge`, consumed in input order as the gateway consumes it
/// (clamped to the hold). A charge of nothing returns every share whole.
/// `None` if any input's refund could not be signed.
fn refunds_value(
    issuer: &Issuer,
    config: &MockConfig,
    inputs: &[Presented],
    charge: u128,
) -> Option<serde_json::Value> {
    let shares: Vec<u128> = inputs.iter().map(Presented::share).collect::<Option<_>>()?;
    let hold = eidola_common::spend_credentials::hold(&shares)?;
    let returns = eidola_common::spend_credentials::partial_returns(&shares, charge.min(hold))?;
    let refunds = inputs
        .iter()
        .zip(returns)
        .map(|(input, t)| {
            let signer = config.key_swap.issuer_for(issuer, &input.key_id)?;
            let refund = signer.refund_of(&input.proof, t)?;
            Some(serde_json::json!({
                "refund": refund,
                "issuer_key_id": refund_key_id(signer, config),
            }))
        })
        .collect::<Option<Vec<_>>>()?;
    (!refunds.is_empty()).then_some(serde_json::Value::Array(refunds))
}

/// Start the mock upstream on an ephemeral loopback port.
pub async fn start(config: MockConfig) -> MockServer {
    let (close_tx, mut close_rx) =
        tokio::sync::mpsc::unbounded_channel::<tokio::sync::oneshot::Sender<()>>();
    config.close_switch.install(close_tx);
    let issuer = Arc::new(Issuer::new());
    let issuance = config.issuance.clone();
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let base_url = format!("http://{addr}");

    let chat_hits = Arc::new(AtomicU64::new(0));
    let chat_bodies: Arc<std::sync::Mutex<Vec<serde_json::Value>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let chat_raw_bodies: Arc<std::sync::Mutex<Vec<Vec<u8>>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let chat_auths: Arc<std::sync::Mutex<Vec<bool>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let chat_auth_values: Arc<std::sync::Mutex<Vec<Option<String>>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let chat_spends: Arc<std::sync::Mutex<Vec<Vec<String>>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let refund_hits = Arc::new(AtomicU64::new(0));
    let models_hits = Arc::new(AtomicU64::new(0));

    let task = {
        let issuer = issuer.clone();
        let chat_hits = chat_hits.clone();
        let chat_bodies = chat_bodies.clone();
        let chat_raw_bodies = chat_raw_bodies.clone();
        let chat_auths = chat_auths.clone();
        let chat_auth_values = chat_auth_values.clone();
        let chat_spends = chat_spends.clone();
        let refund_hits = refund_hits.clone();
        let models_hits = models_hits.clone();
        tokio::spawn(async move {
            loop {
                let accepted = tokio::select! {
                    accepted = listener.accept() => accepted,
                    Some(ack) = close_rx.recv() => {
                        // Stop listening *before* acknowledging, so the next
                        // connection attempt is refused rather than queued.
                        drop(listener);
                        let _ = ack.send(());
                        break;
                    }
                };
                let Ok((stream, _)) = accepted else {
                    break;
                };
                let issuer = issuer.clone();
                let config = config.clone();
                let chat_hits = chat_hits.clone();
                let chat_bodies = chat_bodies.clone();
                let chat_raw_bodies = chat_raw_bodies.clone();
                let chat_auths = chat_auths.clone();
                let chat_auth_values = chat_auth_values.clone();
                let chat_spends = chat_spends.clone();
                let refund_hits = refund_hits.clone();
                let models_hits = models_hits.clone();
                tokio::spawn(
                    GATEWAY_STREAM.scope(std::cell::Cell::new(None), async move {
                        let _ = handle_conn(
                            stream,
                            issuer,
                            config,
                            chat_hits,
                            chat_bodies,
                            chat_raw_bodies,
                            chat_auths,
                            chat_auth_values,
                            chat_spends,
                            refund_hits,
                            models_hits,
                        )
                        .await;
                    }),
                );
            }
        })
    };

    MockServer {
        base_url,
        chat_hits,
        chat_bodies,
        chat_raw_bodies,
        chat_auths,
        chat_auth_values,
        chat_spends,
        refund_hits,
        issuer,
        models_hits,
        issuance,
        _task: task,
    }
}

/// A parsed HTTP/1.1 request: method, path, auth header, body.
struct Req {
    method: String,
    path: String,
    /// The `Authorization` header (an external backend's key; never a spend).
    auth: Option<String>,
    /// Whether the request carried the refund-key header the gateway refuses.
    refund_key_header: bool,
    /// The body's inputs, in input order: each `credentials` element's
    /// `token` text and its refund recovery key, decoded (`None` when absent
    /// or not exactly the wire form — the real server refuses both alike).
    /// Empty when the member is not an array of one or two
    /// `{token, refund_key}` elements.
    spends: Vec<WireSpend>,
    /// Whether the body carried a `credentials` member at all.
    has_credentials: bool,
    body: Vec<u8>,
}

/// The body's inputs, read independently of the client's writer: the
/// `credentials` member must be an array of one or two objects, each with
/// exactly the members `token` and `refund_key`, as the gateway reads it.
/// One `credentials` element as read: its `token` text and its refund
/// recovery key, decoded (`None` when absent or not exactly the wire form).
type WireSpend = (String, Option<[u8; 32]>);

fn body_spend(body: &[u8]) -> (bool, Vec<WireSpend>) {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) else {
        return (false, Vec::new());
    };
    let Some(member) = value.get("credentials") else {
        return (false, Vec::new());
    };
    let elements = match member.as_array() {
        Some(elements) if (1..=2).contains(&elements.len()) => elements,
        _ => return (true, Vec::new()),
    };
    let mut spends = Vec::new();
    for element in elements {
        let Some(object) = element.as_object().filter(|o| o.len() == 2) else {
            return (true, Vec::new());
        };
        let Some(token) = object.get("token").and_then(|t| t.as_str()) else {
            return (true, Vec::new());
        };
        let key = object
            .get("refund_key")
            .and_then(|k| k.as_str())
            .and_then(|k| eidola_common::refund_key::decode_refund_key(k.as_bytes()));
        spends.push((token.to_owned(), key));
    }
    (true, spends)
}

/// `raw` without its `credentials` member: what a test asserting bytes
/// compares, since the member is the spend's and changes every request. The
/// member's values (padded base64url, hex) hold no `]`, so the array ends at
/// the first one.
fn without_credentials(raw: &[u8]) -> Vec<u8> {
    let Ok(text) = std::str::from_utf8(raw) else {
        return raw.to_vec();
    };
    let Some(start) = text.find("\"credentials\":[") else {
        return raw.to_vec();
    };
    let Some(close) = text[start..].find(']') else {
        return raw.to_vec();
    };
    let mut end = start + close + 1;
    let mut start = start;
    if text[end..].starts_with(',') {
        end += 1;
    } else if text[..start].ends_with(',') {
        start -= 1;
    }
    format!("{}{}", &text[..start], &text[end..]).into_bytes()
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
    let mut refund_key_header = false;
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            let name = name.trim().to_ascii_lowercase();
            let value = value.trim();
            if name == "content-length" {
                content_length = value.parse().unwrap_or(0);
            } else if name == "authorization" {
                auth = Some(value.to_string());
            } else if name == "x-eidola-refund-key" {
                refund_key_header = true;
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

    let (has_credentials, spends) = body_spend(&body);
    Some(Req {
        method,
        path,
        auth,
        refund_key_header,
        spends,
        has_credentials,
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
    chat_raw_bodies: Arc<std::sync::Mutex<Vec<Vec<u8>>>>,
    chat_auths: Arc<std::sync::Mutex<Vec<bool>>>,
    chat_auth_values: Arc<std::sync::Mutex<Vec<Option<String>>>>,
    chat_spends: Arc<std::sync::Mutex<Vec<Vec<String>>>>,
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
            config.close_switch.trip_if_armed().await;
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
            write_json(&mut stream, 200, &keys_body(&issuer, &config)).await?;
        }
        ("GET", "/v1/account/balances") => {
            let body = serde_json::json!({
                "available": config.balance,
                "pools": [{ "amount": config.balance, "source": "mock", "expires_at": null }],
            });
            write_json(&mut stream, 200, &body.to_string()).await?;
        }
        ("POST", "/v1/account/credentials") => {
            // Its own boxed future: every route's locals share this one, which
            // runs on the core's worker threads beside the turn it answers.
            Box::pin(handle_issuance(&mut stream, &issuer, &config, &req.body)).await?;
        }
        ("POST", "/v1/credentials/refund") => {
            let attempt = refund_hits.fetch_add(1, Ordering::SeqCst) + 1;
            // The real endpoint refuses a credential in a header, and a
            // recovery without its key, and opens a stored refund only for the
            // key the spend arrived with.
            if req.auth.is_some() || req.refund_key_header {
                return write_json(&mut stream, 400, &error_body("credential in a header")).await;
            }
            // Exactly one input: each is recovered by its own nullifier.
            let [(token, key)] = req.spends.as_slice() else {
                return write_json(&mut stream, 400, &error_body("exactly one credential")).await;
            };
            let input = Presented::from_token(token);
            match (&input, key) {
                (_, None) => {
                    return write_json(&mut stream, 400, &error_body("refund key required")).await;
                }
                (Some(input), Some(key)) if !issuer.spend_key_matches(&input.proof, *key) => {
                    return write_json(&mut stream, 404, &error_body("refund token not found"))
                        .await;
                }
                _ => {}
            }
            handle_refund(&mut stream, &issuer, &config, input, attempt).await?;
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
            if let Some(mut body) = parsed.clone() {
                if let Some(object) = body.as_object_mut() {
                    object.remove("credentials");
                }
                chat_bodies.lock().unwrap().push(body);
            }
            chat_raw_bodies
                .lock()
                .unwrap()
                .push(without_credentials(&req.body));
            chat_auths.lock().unwrap().push(req.auth.is_some());
            chat_auth_values.lock().unwrap().push(req.auth.clone());
            chat_spends
                .lock()
                .unwrap()
                .push(req.spends.iter().map(|(token, _)| token.clone()).collect());
            // A spend travels in the body and nowhere else: one in a header,
            // or a `credentials` member that is not one or two well-formed
            // spends, is refused before anything is taken, as the gateway
            // refuses it.
            if req.has_credentials && (req.auth.is_some() || req.refund_key_header) {
                return write_json(&mut stream, 400, &error_body("credential in a header")).await;
            }
            // **The gateway serves chat only as a stream**: a spending
            // request that does not ask for one is refused before anything
            // is taken, as the gateway refuses it, and counted, so a test can
            // say none was sent. Every completion it is answered with is a
            // stream ([`GATEWAY_STREAM`]).
            if req.has_credentials {
                let streams = parsed
                    .as_ref()
                    .and_then(|b| b.get("stream"))
                    .and_then(|s| s.as_bool())
                    == Some(true);
                if !streams {
                    issuer.blocking_asks.fetch_add(1, Ordering::SeqCst);
                    return write_json(
                        &mut stream,
                        400,
                        &error_body("a chat completion must ask for a stream"),
                    )
                    .await;
                }
                GATEWAY_STREAM
                    .with(|on| on.set((!config.ignores_stream).then_some(config.answer_ms)));
            }
            let inputs: Option<Vec<Presented>> = req
                .spends
                .iter()
                .map(|(token, _)| Presented::from_token(token))
                .collect();
            let inputs = inputs.unwrap_or_default();
            if req.has_credentials && (inputs.is_empty() || inputs.len() != req.spends.len()) {
                return write_json(&mut stream, 401, &error_body("malformed credentials")).await;
            }
            // A spend without its refund recovery key, a share of zero, and
            // one credential presented twice are refused before anything is
            // taken, as the real server refuses them.
            for (input, (_, key)) in inputs.iter().zip(&req.spends) {
                match key {
                    Some(key) => {
                        issuer.spend_key_matches(&input.proof, *key);
                    }
                    None => {
                        return write_json(&mut stream, 400, &error_body("refund key required"))
                            .await;
                    }
                }
                if input.share().is_none_or(|share| share == 0) {
                    return write_json(&mut stream, 400, &error_body("a share of zero")).await;
                }
            }
            if inputs.len() == 2
                && inputs[0].key_id == inputs[1].key_id
                && inputs[0].proof.nullifier().as_bytes() == inputs[1].proof.nullifier().as_bytes()
            {
                return write_json(&mut stream, 400, &error_body("one credential twice")).await;
            }
            let parsed: serde_json::Value =
                serde_json::from_slice(&req.body).unwrap_or(serde_json::Value::Null);
            // **The hold is exactly the worst case**, as the gateway requires:
            // computed here from the body as it arrived and the catalog row it
            // names, through the shared contract, against the sum of the
            // inputs' shares. A spend holding anything else is refused after
            // it is taken, with full refunds, as the gateway refuses it (and
            // counted, so a test can say none was).
            let hold = eidola_common::spend_credentials::hold(
                &inputs
                    .iter()
                    .filter_map(Presented::share)
                    .collect::<Vec<_>>(),
            );
            if !inputs.is_empty()
                && let Some(worst) = worst_case_cost(&config, &parsed)
                && hold != Some(worst)
            {
                issuer.hold_mismatches.fetch_add(1, Ordering::SeqCst);
                let mut body: serde_json::Value =
                    serde_json::from_str(&error_body("the hold must equal the worst-case cost"))
                        .expect("error body");
                if let Some(refunds) = refunds_value(&issuer, &config, &inputs, 0) {
                    body["refunds"] = refunds;
                }
                return write_json(&mut stream, 400, &body.to_string()).await;
            }
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
                (true, SummaryBehavior::ReplyPaddedPastCeiling(_)) if gateway_stream() => {
                    // A stream, which padding does not lengthen: past the
                    // ceiling is an answer whose data passes it.
                    let piece = "y".repeat(64 * 1024);
                    let pieces: Vec<&str> = std::iter::repeat_n(piece.as_str(), 48).collect();
                    return write_sse_stream(&mut stream, true, &pieces).await;
                }
                (true, SummaryBehavior::ReplyPaddedPastCeiling(content)) => {
                    let body = serde_json::json!({
                        "choices": [{ "message": { "role": "assistant", "content": content } }],
                        "usage": { "prompt_tokens": 9, "completion_tokens": 5 },
                    });
                    let padded = format!("{body}{}", " ".repeat(3 * 1024 * 1024));
                    return write_json(&mut stream, 200, &padded).await;
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
                    handle_chat(&mut stream, &issuer, &config, &inputs, hit, &parsed).await?;
                }
            }
        }
        _ => {
            write_json(&mut stream, 404, &error_body("not found")).await?;
        }
    }
    Ok(())
}

/// `POST /v1/account/credentials`: idempotent issuance, as the real server
/// answers it (see [`IssuanceKnobs`] for the misbehaviours a test can ask for).
async fn handle_issuance(
    stream: &mut TcpStream,
    issuer: &Issuer,
    config: &MockConfig,
    body: &[u8],
) -> std::io::Result<()> {
    issuer.issuance_requests.fetch_add(1, Ordering::SeqCst);
    if config.issuance.0.unauthorized.load(Ordering::SeqCst) {
        return write_json(stream, 401, &error_body("invalid credentials")).await;
    }
    if config.issuance.0.stall.load(Ordering::SeqCst) {
        tokio::time::sleep(std::time::Duration::from_secs(120)).await;
        return Ok(());
    }
    let parsed: serde_json::Value = serde_json::from_slice(body).unwrap_or(serde_json::Value::Null);
    let ir = parsed.get("issuance_request").and_then(|v| v.as_str());
    let credits = parsed.get("credits").and_then(|v| v.as_i64()).unwrap_or(0);
    let named_key = parsed.get("issuer_key_id").and_then(|v| v.as_str());
    let Some(idem) = parsed
        .get("idempotency_key")
        .and_then(|v| v.as_str())
        .filter(|k| k.len() == 64)
        .map(str::to_string)
    else {
        return write_json(stream, 400, &error_body("idempotency_key required")).await;
    };
    // What a retry must agree on to be answered from the store.
    let request_digest = format!("{}|{credits}|{}", ir.unwrap_or(""), named_key.unwrap_or(""));
    let stored = issuer.issuances.lock().unwrap().get(&idem).cloned();
    if stored.is_some() && config.issuance.0.fail_retries.load(Ordering::SeqCst) {
        return write_json(stream, 503, &error_body("issuance unavailable")).await;
    }
    if let Some((digest, body)) = stored {
        if digest != request_digest {
            return write_json(stream, 409, &error_body("idempotency key reused")).await;
        }
        return write_json(stream, 200, &body).await;
    }
    if let Some(status) = config.issuance.refusal() {
        return write_json(stream, status, &error_body("issuance refused")).await;
    }
    // The key listed as current — the second one once a test swapped it.
    let issuing = config.key_swap.active(issuer);
    if named_key != Some(issuing.key_id_hex.as_str()) {
        return write_json(stream, 409, &error_body("issuer key is not current")).await;
    }
    match ir.and_then(|ir| {
        issuing.issue(
            ir,
            credits + config.issuance_extra_credits,
            config.issuance_context,
        )
    }) {
        Some(issuance_response) => {
            let body = serde_json::json!({
                "issuance_response": issuance_response,
                "issuer_key_id": config
                    .issued_key_id
                    .clone()
                    .unwrap_or_else(|| issuing.key_id_hex.clone()),
                "credits": credits,
                "ledger_entry_id": uuid_like(),
            })
            .to_string();
            let signed = SignedIssuance {
                idem,
                request_digest,
                body: body.clone(),
                credits,
            };
            if config.issuance.take_hold() {
                // Signed, not yet debited; the client hears nothing.
                issuer.held.lock().unwrap().push(signed);
                return Ok(());
            }
            issuer.record_issuance(signed);
            if config.issuance.take_loss() {
                // Debited and stored; the client hears nothing.
                return Ok(());
            }
            write_json(stream, 200, &body).await?;
        }
        None => {
            write_json(stream, 400, &error_body("issuance failed")).await?;
        }
    }
    Ok(())
}

async fn handle_refund(
    stream: &mut TcpStream,
    issuer: &Issuer,
    config: &MockConfig,
    input: Option<Presented>,
    // 1-based index of this recovery call.
    attempt: u64,
) -> std::io::Result<()> {
    let failing = match config.refund {
        RefundMode::Fail => true,
        RefundMode::FailFirst(n) => attempt <= n,
        RefundMode::Succeed | RefundMode::ProvisionalFirst(_) | RefundMode::ChargedEarly(_) => {
            false
        }
        RefundMode::StallAfter(n) if attempt > n => {
            std::future::pending::<()>().await;
            unreachable!()
        }
        RefundMode::StallAfter(_) => false,
        RefundMode::NotStored => {
            return write_json(stream, 404, &error_body("refund token not found")).await;
        }
    };
    if failing {
        return write_json(stream, 500, &error_body("refund unavailable")).await;
    }
    let stored_provisional = matches!(config.refund, RefundMode::ProvisionalFirst(_));
    let provisional = matches!(config.refund, RefundMode::ProvisionalFirst(n) if attempt <= n);
    // The stored provisional refund returns nothing; a stream cut short is
    // charged what it generated; any other returns the share whole. Signed
    // under the key the input names.
    let refunds = input.and_then(|input| {
        let charge = if stored_provisional {
            input.share()?
        } else if let RefundMode::ChargedEarly(charged) = config.refund {
            charged
        } else {
            0
        };
        refunds_value(issuer, config, std::slice::from_ref(&input), charge)
    });
    match refunds {
        Some(refunds) => {
            let body = serde_json::json!({
                "refunds": refunds,
                "provisional": provisional,
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
    inputs: &[Presented],
    hit: u64,
    request: &serde_json::Value,
) -> std::io::Result<()> {
    if config.chat_delay_ms > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(config.chat_delay_ms)).await;
    }
    let fail = config
        .fail_next_chat
        .lock()
        .expect("fail-next lock poisoned")
        .take();
    if let Some(status) = fail {
        return write_json(stream, status, &error_body("upstream model error")).await;
    }
    // The inline refunds of a completed request, computed once (shared by
    // the blocking happy path): one per input, settled at the configured
    // charge.
    let inline_refund = if matches!(config.refund, RefundMode::Succeed | RefundMode::NotStored) {
        refunds_value(issuer, config, inputs, config.settled_charge.unwrap_or(0))
    } else {
        None
    };
    // Every share whole: what a refusal carries.
    let full_refunds = || refunds_value(issuer, config, inputs, 0);
    // What a completed stream's metadata carries, whatever the refund mode.
    let completed_refunds =
        || refunds_value(issuer, config, inputs, config.settled_charge.unwrap_or(0));

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
                body["refunds"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::OkBlockingPaddedPastCeiling if gateway_stream() => {
            // The gateway's answer is a stream, which padding does not
            // lengthen: what passes the ceiling is an answer whose data does.
            let piece = "y".repeat(64 * 1024);
            let pieces: Vec<&str> = std::iter::repeat_n(piece.as_str(), 160).collect();
            write_sse_stream(stream, true, &pieces).await
        }
        ChatBehavior::OkBlockingPaddedPastCeiling => {
            let body = serde_json::json!({
                "choices": [{ "message": {
                    "role": "assistant",
                    "content": "Hello from the mock.",
                } }],
                "usage": { "prompt_tokens": 11, "completion_tokens": 5 },
            });
            // Past `proxy::route::MAX_RESPONSE_BYTES` (8 MiB) — spelled here
            // rather than imported, because the mock is the *peer* and must not
            // learn this app's ceilings.
            let padded = format!("{body}{}", " ".repeat(9 * 1024 * 1024));
            write_json(stream, 200, &padded).await
        }
        ChatBehavior::Non2xxPaddedPastCeiling(status) => {
            let padded = format!(
                "{}{}",
                error_body("upstream model error"),
                " ".repeat(9 * 1024 * 1024)
            );
            write_json(stream, status, &padded).await
        }
        ChatBehavior::OkBlockingNotACompletion => {
            let mut body: serde_json::Value =
                serde_json::from_str(&error_body("upstream model error")).expect("error body");
            if let Some(refunds) = full_refunds() {
                body["refunds"] = refunds;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::Non2xx(status) => {
            write_json(stream, status, &error_body("upstream model error")).await
        }
        ChatBehavior::Non2xxWithRefund(status) => {
            let mut body: serde_json::Value =
                serde_json::from_str(&error_body("stream start failed")).expect("error body");
            if let Some(refunds) = full_refunds() {
                body["refunds"] = refunds;
            }
            write_json(stream, status, &body.to_string()).await
        }
        ChatBehavior::StreamingPastReadCeiling => {
            // 160 × 64 KiB of content: ten megabytes, past an eight-megabyte
            // ceiling, in events no larger than any frame bound.
            let piece = "y".repeat(64 * 1024);
            let pieces: Vec<&str> = std::iter::repeat_n(piece.as_str(), 160).collect();
            write_sse_stream(stream, true, &pieces).await
        }
        ChatBehavior::OkStreamingOversized => {
            // Three large content deltas — together far past what a Record row
            // keeps, each well under any frame bound, nothing else unusual.
            let piece = "x".repeat(400_000);
            write_sse_stream(stream, true, &[&piece, &piece, &piece]).await
        }
        ChatBehavior::StreamingOneOversizedEvent => {
            // One content delta past a one-megabyte frame ceiling, ended
            // properly. Spelled here rather than imported: the mock is the
            // peer and must not learn this app's ceilings.
            let big = "z".repeat(1_200_000);
            write_sse_stream(stream, true, &[&big]).await
        }
        ChatBehavior::DropBeforeResponse => {
            // Drop the connection without writing anything: the client's
            // `send().await` succeeds (request was sent) but reading the body /
            // the response itself fails — exercising the network-error arm.
            // Actually `send()` returns once headers are read; with no bytes at
            // all reqwest surfaces a transport error from `send`.
            Ok(())
        }
        ChatBehavior::OkNonJsonBody => {
            write_raw(
                stream,
                200,
                "text/html",
                "<html><body>502 Bad Gateway</body></html>",
            )
            .await
        }
        ChatBehavior::OkStreaming => write_sse_stream(stream, true, &[STREAM_CONTENT]).await,
        ChatBehavior::OkStreamingBareCarriageReturns => {
            write_sse_stream_with_terminator(stream, &[STREAM_CONTENT], "\r").await
        }
        ChatBehavior::StreamingWithMetadataRefund => {
            let refund = completed_refunds();
            write_sse_stream_with_metadata(stream, &[STREAM_CONTENT], refund).await
        }
        ChatBehavior::StreamingPadded | ChatBehavior::StreamingPaddedPastReadCeiling => {
            let refund = completed_refunds();
            let idle_frames = if matches!(config.chat, ChatBehavior::StreamingPaddedPastReadCeiling)
            {
                // Ten megabytes of padding frames across the deltas below.
                10 * 1024 * 1024 / eidola_server_gateway::padding::FRAME_BYTES / 3
            } else {
                2
            };
            write_padded_sse_stream(stream, &PADDED_CONTENT, refund, idle_frames).await
        }
        ChatBehavior::StreamingPaddedNearReadCeiling => write_padded_near_ceiling(stream).await,
        ChatBehavior::StreamingSplitDataFields => {
            let refund = completed_refunds();
            write_sse_stream_split_data(stream, &[STREAM_CONTENT], refund).await
        }
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
        ChatBehavior::HeadThenCut {
            status,
            content_type,
        } => {
            let head = format!(
                "HTTP/1.1 {status} OK\r\nContent-Type: {content_type}\r\n\
                 Content-Length: 4096\r\nConnection: close\r\n\r\n{{\"error\":"
            );
            stream.write_all(head.as_bytes()).await?;
            stream.flush().await
        }
        ChatBehavior::StreamingUnterminatedFlood => write_sse_flood(stream).await,
        ChatBehavior::StreamingEndsWithoutDone => {
            write_sse_unterminated_stream(stream, &[STREAM_CONTENT]).await
        }
        ChatBehavior::StreamingCutShortWithRefund => {
            // Settled on what was generated: `settled_charge`, or nothing.
            let refund = completed_refunds();
            write_sse_cut_short(stream, &[STREAM_CONTENT], refund).await
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
                body["refunds"] = refund;
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
                    body["refunds"] = refund;
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
                    body["refunds"] = refund;
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
                body["refunds"] = refund;
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
                body["refunds"] = refund;
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
                body["refunds"] = refund;
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
                body["refunds"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::ToolCallsNotAnArrayStreaming => write_sse_bad_tool_calls_stream(stream).await,
        ChatBehavior::DeclineBlocking => {
            let mut body = tool_call_body(&decline_call_object());
            if let Some(refund) = inline_refund {
                body["refunds"] = refund;
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
                body["refunds"] = refund;
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
                body["refunds"] = refund;
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
                    "continuable": true,
                }],
                "usage": { "prompt_tokens": 11, "completion_tokens": CEILING_TOKENS },
            });
            if let Some(refund) = inline_refund {
                body["refunds"] = refund;
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
                    "continuable": true,
                }],
                "usage": { "prompt_tokens": 11, "completion_tokens": CEILING_TOKENS },
            });
            if let Some(refund) = inline_refund {
                body["refunds"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::ContinueAfterLength { cut, again } => {
            let (reasoning, content, finish) = match request.get("continue_final_message") {
                None => match cut {
                    Cut::Reasoning => (TRUNCATED_REASONING, "", "length"),
                    Cut::Answer | Cut::Dropped => (TRUNCATED_REASONING, STREAM_CONTENT, "length"),
                },
                Some(serde_json::Value::String(field)) if field == "reasoning_content" => {
                    if again && hit == 2 {
                        (CONTINUED_REASONING, "", "length")
                    } else {
                        (CONTINUED_REASONING, CONTINUED_ANSWER, "stop")
                    }
                }
                Some(_) => (
                    "",
                    CONTINUED_TAIL,
                    if again && hit == 2 { "length" } else { "stop" },
                ),
            };
            let exact = !(cut == Cut::Dropped && request.get("continue_final_message").is_none());
            write_ending(
                stream,
                request,
                inline_refund,
                reasoning,
                content,
                finish,
                exact,
            )
            .await
        }
        ChatBehavior::HeaderThenLength => {
            let content = format!("{MIMICKED_HEADER}\n\nThe tides are");
            write_ending(stream, request, inline_refund, "", &content, "length", true).await
        }
        ChatBehavior::ContinueIntoToolRound => {
            let streamed = request.get("stream").and_then(|s| s.as_bool()) == Some(true);
            if request.get("continue_final_message").is_some() {
                if streamed {
                    return write_sse_tool_stream(stream, 1, false).await;
                }
                let mut body = tool_call_body(&tool_call_object(1, &tool_arguments(1)));
                if let Some(refund) = inline_refund {
                    body["refunds"] = refund;
                }
                return write_json(stream, 200, &body.to_string()).await;
            }
            if hit == 1 {
                write_ending(
                    stream,
                    request,
                    inline_refund,
                    TRUNCATED_REASONING,
                    "",
                    "length",
                    true,
                )
                .await
            } else {
                write_ending(
                    stream,
                    request,
                    inline_refund,
                    "",
                    TOOL_FINAL_CONTENT,
                    "stop",
                    true,
                )
                .await
            }
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
                body["refunds"] = refund;
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
                body["refunds"] = refund;
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
                body["refunds"] = refund;
            }
            write_json(stream, 200, &body.to_string()).await
        }
        ChatBehavior::ToolScript => {
            // Consume the script: the round that serves it asks for every
            // scripted call at once (so one turn can exercise several tools),
            // and every later request answers normally.
            let script: Vec<(String, String)> =
                std::mem::take(&mut *config.tool_script.lock().unwrap());
            // **Whichever transport asked.** A turn driven for a named
            // participant streams (`respond_stream_as`), which is the only
            // door a delegated room's own agents take — so a blocking-only arm
            // would answer those a body their parser reads as no content at
            // all, and no scripted call would ever be made.
            if request.get("stream").and_then(|s| s.as_bool()) == Some(true) {
                return write_sse_script_stream(stream, &script).await;
            }
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
                body["refunds"] = refund;
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

/// The SSE twin of the [`ChatBehavior::ToolScript`] body: every scripted call
/// in one delta, or the plain answer when the script is spent.
///
/// One delta per call rather than the split-across-deltas shape
/// [`write_sse_tool_stream`] uses — that stream exists to exercise the client's
/// accumulator, and this one exists to deliver arguments a test wrote.
async fn write_sse_script_stream(
    stream: &mut TcpStream,
    script: &[(String, String)],
) -> std::io::Result<()> {
    stream.write_all(sse_head().as_bytes()).await?;
    stream.flush().await?;
    if script.is_empty() {
        stream
            .write_all(&sse_event(
                &serde_json::json!({"choices":[{"delta":{"content":TOOL_FINAL_CONTENT}}]})
                    .to_string(),
            ))
            .await?;
    } else {
        let calls: Vec<serde_json::Value> = script
            .iter()
            .enumerate()
            .map(|(i, (name, arguments))| {
                serde_json::json!({
                    "index": i,
                    "id": tool_call_id(i as u64 + 1),
                    "type": "function",
                    "function": { "name": name, "arguments": arguments },
                })
            })
            .collect();
        stream
            .write_all(&sse_event(
                &serde_json::json!({"choices":[{"delta":{"tool_calls":calls}}]}).to_string(),
            ))
            .await?;
    }
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

/// The name an upstream gives the model in its own chunks — deliberately
/// **not** the canonical id, so a forwarded chunk still carrying it is a
/// chunk nothing rewrote.
pub const STREAM_WIRE_MODEL: &str = "upstream-wire-name";

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
        "choices": [{ "delta": {}, "finish_reason": "length", "continuable": true }],
        "usage": { "prompt_tokens": 11, "completion_tokens": CEILING_TOKENS }
    });
    stream.write_all(&sse_event(&terminal.to_string())).await?;
    stream.write_all(&sse_event("[DONE]")).await?;
    stream.write_all(b"0\r\n\r\n").await?;
    stream.flush().await?;
    Ok(())
}

/// Answer in whichever transport asked: `reasoning`, then `content`, ending
/// with `finish`. A `length` ending reports the whole ceiling spent, as the
/// truncating behaviours do, with the engine's `continuable` verdict (`exact`).
/// A stream carries the reasoning as one delta and the content as two, so the
/// client's accumulation is exercised.
async fn write_ending(
    stream: &mut TcpStream,
    request: &serde_json::Value,
    inline_refund: Option<serde_json::Value>,
    reasoning: &str,
    content: &str,
    finish: &str,
    exact: bool,
) -> std::io::Result<()> {
    let completion_tokens = if finish == "length" {
        CEILING_TOKENS
    } else {
        5
    };
    if request.get("stream").and_then(|s| s.as_bool()) != Some(true) {
        let mut body = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                },
                "finish_reason": finish,
            }],
            "usage": { "prompt_tokens": 11, "completion_tokens": completion_tokens },
        });
        if finish == "length" {
            body["choices"][0]["continuable"] = serde_json::json!(exact);
        }
        if let Some(refund) = inline_refund {
            body["refunds"] = refund;
        }
        return write_json(stream, 200, &body.to_string()).await;
    }
    stream.write_all(sse_head().as_bytes()).await?;
    let mut deltas = Vec::new();
    if !reasoning.is_empty() {
        deltas.push(serde_json::json!({"reasoning_content": reasoning}));
    }
    if !content.is_empty() {
        let half = content.floor_char_boundary(content.len() / 2);
        let (a, b) = content.split_at(half);
        for piece in [a, b].into_iter().filter(|p| !p.is_empty()) {
            deltas.push(serde_json::json!({"content": piece}));
        }
    }
    for delta in deltas {
        let chunk = serde_json::json!({"choices": [{"delta": delta, "finish_reason": null}]});
        stream.write_all(&sse_event(&chunk.to_string())).await?;
    }
    let mut terminal = serde_json::json!({
        "choices": [{ "delta": {}, "finish_reason": finish }],
        "usage": { "prompt_tokens": 11, "completion_tokens": completion_tokens }
    });
    if finish == "length" {
        terminal["choices"][0]["continuable"] = serde_json::json!(exact);
    }
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

/// The server's real streaming close: the ordinary chunks, then one metadata
/// event carrying the refund, then `[DONE]`. See
/// [`ChatBehavior::StreamingWithMetadataRefund`].
/// A response with an arbitrary content type and body — for the shapes that
/// are not JSON at all.
async fn write_raw(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &str,
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} OK\r\nContent-Type: {content_type}\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(body.as_bytes()).await?;
    stream.flush().await
}

async fn write_sse_stream_with_metadata(
    stream: &mut TcpStream,
    content_chunks: &[&str],
    refund: Option<serde_json::Value>,
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

    let mut metadata = serde_json::json!({
        "object": "eidola.chat.completion.metadata",
        "id": "chatcmpl-mock",
    });
    if let Some(refund) = refund {
        metadata["refunds"] = refund;
    }
    stream.write_all(&send_event(metadata.to_string())).await?;
    stream.write_all(&send_event("[DONE]".to_string())).await?;
    stream.write_all(b"0\r\n\r\n").await?;
    stream.flush().await?;
    Ok(())
}

/// [`ChatBehavior::StreamingCutShortWithRefund`]: deltas, the metadata event
/// with its refund, and no `[DONE]`.
async fn write_sse_cut_short(
    stream: &mut TcpStream,
    content_chunks: &[&str],
    refund: Option<serde_json::Value>,
) -> std::io::Result<()> {
    let head = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Transfer-Encoding: chunked\r\n\
                Connection: close\r\n\r\n";
    stream.write_all(head.as_bytes()).await?;
    let send_event = |payload: String| -> Vec<u8> {
        let event = format!("data: {payload}\n\n");
        let mut out = format!("{:x}\r\n", event.len()).into_bytes();
        out.extend_from_slice(event.as_bytes());
        out.extend_from_slice(b"\r\n");
        out
    };
    for chunk in content_chunks {
        let content = serde_json::json!({ "choices": [{ "delta": { "content": chunk } }] });
        stream.write_all(&send_event(content.to_string())).await?;
    }
    let mut metadata = serde_json::json!({
        "object": "eidola.chat.completion.metadata",
        "id": "",
    });
    if let Some(refund) = refund {
        metadata["refunds"] = refund;
    }
    stream.write_all(&send_event(metadata.to_string())).await?;
    stream.write_all(b"0\r\n\r\n").await?;
    stream.flush().await
}

/// The content [`ChatBehavior::StreamingPadded`] streams, one delta each;
/// joined, [`STREAM_CONTENT`].
pub const PADDED_CONTENT: [&str; 3] = ["Hello ", "from the ", "stream."];

/// The metadata stream framed by the Eidola server's own framer, with
/// `idle_frames` frames of nothing but padding after each content delta. See
/// [`ChatBehavior::StreamingPadded`].
async fn write_padded_sse_stream(
    stream: &mut TcpStream,
    content_chunks: &[&str],
    refund: Option<serde_json::Value>,
    idle_frames: usize,
) -> std::io::Result<()> {
    use eidola_server_gateway::padding::{Framer, StreamEvent};
    let head = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Transfer-Encoding: chunked\r\n\
                Connection: close\r\n\r\n";
    stream.write_all(head.as_bytes()).await?;
    stream.flush().await?;

    let mut framer = Framer::new(eidola_server_gateway::padding::Cadence::CLIENT_FACING.frames);
    // One write per frame, as the server's ticks write them.
    async fn tick(stream: &mut TcpStream, framer: &mut Framer) -> std::io::Result<()> {
        let frame = framer.frame();
        let mut out = format!("{:x}\r\n", frame.len()).into_bytes();
        out.extend_from_slice(&frame);
        out.extend_from_slice(b"\r\n");
        stream.write_all(&out).await?;
        stream.flush().await
    }

    tick(stream, &mut framer).await?;
    for chunk in content_chunks {
        let content = serde_json::json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": STREAM_WIRE_MODEL,
            "choices": [{ "index": 0, "delta": { "content": chunk }, "finish_reason": null }]
        });
        framer.push(&StreamEvent::Json(content.to_string()));
        tick(stream, &mut framer).await?;
        for _ in 0..idle_frames {
            tick(stream, &mut framer).await?;
        }
    }
    let finish = serde_json::json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "model": STREAM_WIRE_MODEL,
        "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }]
    });
    framer.push(&StreamEvent::Json(finish.to_string()));
    let usage = serde_json::json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "model": STREAM_WIRE_MODEL,
        "choices": [],
        "usage": { "prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16 }
    });
    framer.push(&StreamEvent::Json(usage.to_string()));
    let mut metadata = serde_json::json!({
        "object": "eidola.chat.completion.metadata",
        "id": "chatcmpl-mock",
        // Larger than a frame, as the server's privacy and verification
        // metadata is, so the event crosses frames.
        "privacy": { "note": "m".repeat(eidola_server_gateway::padding::FRAME_BYTES) },
    });
    if let Some(refund) = refund {
        metadata["refunds"] = refund;
    }
    framer.push(&StreamEvent::Json(metadata.to_string()));
    framer.push(&StreamEvent::Done);
    while framer.backlog() > 0 {
        tick(stream, &mut framer).await?;
    }
    stream.write_all(b"0\r\n\r\n").await?;
    stream.flush().await?;
    Ok(())
}

/// A blocking completion as the gateway answers it: a stream framed by the
/// gateway's own framer, each choice's reasoning, content and tool calls as
/// deltas, its ending, the usage chunk, the metadata event carrying the
/// completion's `refunds`, then `[DONE]`.
async fn write_gateway_stream(
    stream: &mut TcpStream,
    completion: &serde_json::Value,
) -> std::io::Result<()> {
    use eidola_server_gateway::padding::{Framer, StreamEvent};
    let head = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Transfer-Encoding: chunked\r\n\
                Connection: close\r\n\r\n";
    stream.write_all(head.as_bytes()).await?;
    let answer_ms = GATEWAY_STREAM
        .try_with(std::cell::Cell::get)
        .ok()
        .flatten()
        .unwrap_or(0);
    if answer_ms > 0 {
        stream.flush().await?;
        tokio::time::sleep(std::time::Duration::from_millis(answer_ms)).await;
    }
    let chunk = |choices: serde_json::Value| {
        let mut chunk = serde_json::json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "choices": choices,
        });
        if let Some(model) = completion.get("model") {
            chunk["model"] = model.clone();
        }
        chunk
    };
    let mut events = Vec::new();
    for (i, choice) in completion["choices"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        let message = &choice["message"];
        let mut deltas = Vec::new();
        for key in ["reasoning_content", "reasoning", "content"] {
            if let Some(text) = message.get(key).and_then(|v| v.as_str())
                && !text.is_empty()
            {
                deltas.push(serde_json::json!({ key: text }));
            }
        }
        match message.get("tool_calls") {
            None | Some(serde_json::Value::Null) => {}
            Some(serde_json::Value::Array(calls)) => {
                let calls: Vec<serde_json::Value> = calls
                    .iter()
                    .enumerate()
                    .map(|(index, call)| {
                        let mut call = call.clone();
                        call["index"] = index.into();
                        call
                    })
                    .collect();
                deltas.push(serde_json::json!({ "tool_calls": calls }));
            }
            Some(other) => deltas.push(serde_json::json!({ "tool_calls": other })),
        }
        for delta in deltas {
            events.push(chunk(serde_json::json!([{ "index": i, "delta": delta }])));
        }
        let mut ending = serde_json::json!({ "index": i, "delta": {} });
        for key in ["finish_reason", "continuable"] {
            if let Some(value) = choice.get(key) {
                ending[key] = value.clone();
            }
        }
        events.push(chunk(serde_json::json!([ending])));
    }
    if let Some(usage) = completion.get("usage") {
        let mut tail = chunk(serde_json::json!([]));
        tail["usage"] = usage.clone();
        events.push(tail);
    }
    let mut metadata = serde_json::json!({
        "object": "eidola.chat.completion.metadata",
        "id": "chatcmpl-mock",
    });
    if let Some(refunds) = completion.get("refunds") {
        metadata["refunds"] = refunds.clone();
    }
    events.push(metadata);

    let mut framer = Framer::new(eidola_server_gateway::padding::Cadence::CLIENT_FACING.frames);
    for event in events {
        framer.push(&StreamEvent::Json(event.to_string()));
    }
    framer.push(&StreamEvent::Done);
    while framer.backlog() > 0 {
        let frame = framer.frame();
        let mut out = format!("{:x}\r\n", frame.len()).into_bytes();
        out.extend_from_slice(&frame);
        out.extend_from_slice(b"\r\n");
        stream.write_all(&out).await?;
    }
    stream.write_all(b"0\r\n\r\n").await?;
    stream.flush().await
}

/// See [`ChatBehavior::StreamingPaddedNearReadCeiling`]. The sizes are the
/// server's own encoding (`data: ` + JSON + a blank line), summed here.
async fn write_padded_near_ceiling(stream: &mut TcpStream) -> std::io::Result<()> {
    use eidola_server_gateway::padding::{Framer, StreamEvent};
    // Spelled here rather than imported: the mock is the peer.
    const TARGET: usize = 8 * 1024 * 1024 - 4096;
    let head = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Transfer-Encoding: chunked\r\n\
                Connection: close\r\n\r\n";
    stream.write_all(head.as_bytes()).await?;

    let chunk = |content: &str| {
        serde_json::json!({
            "choices": [{ "index": 0, "delta": { "content": content }, "finish_reason": null }]
        })
        .to_string()
    };
    let encoded = |json: &str| json.len() + "data: ".len() + 2;
    let tail = vec![
        serde_json::json!({ "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }] })
            .to_string(),
        serde_json::json!({
            "choices": [],
            "usage": { "prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16 }
        })
        .to_string(),
        serde_json::json!({ "object": "eidola.chat.completion.metadata", "id": "chatcmpl-mock" })
            .to_string(),
    ];
    let tail_bytes: usize =
        tail.iter().map(|j| encoded(j)).sum::<usize>() + "data: [DONE]\n\n".len();
    let overhead = encoded(&chunk(""));
    let piece = 64 * 1024;
    let mut contents = Vec::new();
    let mut total = tail_bytes;
    while total + overhead + piece <= TARGET {
        contents.push("y".repeat(piece));
        total += overhead + piece;
    }
    contents.push("y".repeat(TARGET - total - overhead));

    let mut framer = Framer::new(eidola_server_gateway::padding::Cadence::CLIENT_FACING.frames);
    let chunked = |frames: &[u8]| {
        let mut out = format!("{:x}\r\n", frames.len()).into_bytes();
        out.extend_from_slice(frames);
        out.extend_from_slice(b"\r\n");
        out
    };
    for content in &contents {
        framer.push(&StreamEvent::Json(chunk(content)));
        while framer.backlog() > 0 {
            stream.write_all(&chunked(&framer.frame())).await?;
        }
    }
    // The last frames, coalesced: padding, then the tail.
    let mut last = Vec::new();
    for _ in 0..8 {
        last.extend_from_slice(&framer.frame());
    }
    for json in tail {
        framer.push(&StreamEvent::Json(json));
    }
    framer.push(&StreamEvent::Done);
    while framer.backlog() > 0 {
        last.extend_from_slice(&framer.frame());
    }
    stream.write_all(&chunked(&last)).await?;
    stream.write_all(b"0\r\n\r\n").await?;
    stream.flush().await
}

/// The metadata stream with each JSON payload split across one `data:` field
/// per line of its pretty-printed form. See
/// [`ChatBehavior::StreamingSplitDataFields`].
async fn write_sse_stream_split_data(
    stream: &mut TcpStream,
    content_chunks: &[&str],
    refund: Option<serde_json::Value>,
) -> std::io::Result<()> {
    let head = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Transfer-Encoding: chunked\r\n\
                Connection: close\r\n\r\n";
    stream.write_all(head.as_bytes()).await?;
    stream.flush().await?;

    let send_event = |payload: &serde_json::Value| -> Vec<u8> {
        let pretty = serde_json::to_string_pretty(payload).expect("serialize");
        let mut event = String::new();
        for line in pretty.lines() {
            event.push_str("data: ");
            event.push_str(line);
            event.push('\n');
        }
        event.push('\n');
        let mut out = format!("{:x}\r\n", event.len()).into_bytes();
        out.extend_from_slice(event.as_bytes());
        out.extend_from_slice(b"\r\n");
        out
    };

    for chunk in content_chunks {
        let content = serde_json::json!({
            "model": STREAM_WIRE_MODEL,
            "choices": [{ "delta": { "content": chunk } }]
        });
        stream.write_all(&send_event(&content)).await?;
        stream.flush().await?;
    }
    let usage = serde_json::json!({
        "model": STREAM_WIRE_MODEL,
        "choices": [],
        "usage": { "prompt_tokens": 11, "completion_tokens": 5 }
    });
    stream.write_all(&send_event(&usage)).await?;
    let mut metadata = serde_json::json!({
        "object": "eidola.chat.completion.metadata",
        "id": "chatcmpl-mock",
    });
    if let Some(refund) = refund {
        metadata["refunds"] = refund;
    }
    stream.write_all(&send_event(&metadata)).await?;
    let done = "data: [DONE]\n\n";
    stream
        .write_all(format!("{:x}\r\n{done}\r\n", done.len()).as_bytes())
        .await?;
    stream.write_all(b"0\r\n\r\n").await?;
    stream.flush().await?;
    Ok(())
}

/// A stream that opens as server-sent events and then never ends an event.
///
/// Every byte is `data:` content with no blank line after it, so
/// `find_event_boundary` never succeeds and everything written stays in the
/// reader's frame accumulator. Deliberately more than the proxy's per-event
/// ceiling, which is the whole point: the ceiling is what has to end this.
async fn write_sse_flood(stream: &mut TcpStream) -> std::io::Result<()> {
    let head = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Transfer-Encoding: chunked\r\n\
                Connection: close\r\n\r\n";
    stream.write_all(head.as_bytes()).await?;
    stream.flush().await?;

    let filler = "x".repeat(64 * 1024);
    stream.write_all(b"6\r\ndata: \r\n").await?;
    for _ in 0..24 {
        let framed = format!("{:x}\r\n{filler}\r\n", filler.len());
        stream.write_all(framed.as_bytes()).await?;
        stream.flush().await?;
    }
    stream.write_all(b"0\r\n\r\n").await?;
    stream.flush().await
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

/// [`write_sse_stream`]'s happy path with a line ending the caller names.
///
/// Only the terminator differs — same events, same order, same `[DONE]` — so a
/// test asserting the ordinary outcome over this writer is asserting that the
/// *framing* was understood rather than that some other path was taken.
///
/// **Every event carries a second field**, an `id:` line ahead of its `data:`
/// one, because a one-field event hides half the defect: separating events at
/// the right byte and then splitting their *fields* the wrong way leaves a
/// single-field event looking perfectly fine, while `id: 1\rdata: {…}` collapses
/// into one line with no `data:` prefix — the payload invisible, no delta
/// delivered, no `[DONE]` seen, and, on the proxy, no refund found.
async fn write_sse_stream_with_terminator(
    stream: &mut TcpStream,
    content_chunks: &[&str],
    eol: &str,
) -> std::io::Result<()> {
    let head = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Transfer-Encoding: chunked\r\n\
                Connection: close\r\n\r\n";
    stream.write_all(head.as_bytes()).await?;
    stream.flush().await?;

    let send_event = |id: usize, payload: String| -> Vec<u8> {
        let event = format!("id: {id}{eol}data: {payload}{eol}{eol}");
        let mut out = format!("{:x}\r\n", event.len()).into_bytes();
        out.extend_from_slice(event.as_bytes());
        out.extend_from_slice(b"\r\n");
        out
    };

    // Each payload names the model the *upstream* calls it, which is what makes
    // a per-event assertion possible downstream: the proxy rewrites that field
    // to the canonical id, and it can only do so on a payload it found.
    let reasoning = serde_json::json!({
        "model": STREAM_WIRE_MODEL,
        "choices": [{ "delta": { "reasoning": "thinking…" } }]
    });
    stream
        .write_all(&send_event(0, reasoning.to_string()))
        .await?;
    stream.flush().await?;

    for (index, chunk) in content_chunks.iter().enumerate() {
        let content = serde_json::json!({
            "model": STREAM_WIRE_MODEL,
            "choices": [{ "delta": { "content": chunk } }]
        });
        stream
            .write_all(&send_event(index + 1, content.to_string()))
            .await?;
        stream.flush().await?;
    }

    let next = content_chunks.len() + 1;
    let usage = serde_json::json!({
        "model": STREAM_WIRE_MODEL,
        "choices": [],
        "usage": { "prompt_tokens": 11, "completion_tokens": 5 }
    });
    stream
        .write_all(&send_event(next, usage.to_string()))
        .await?;
    stream
        .write_all(&send_event(next + 1, "[DONE]".to_string()))
        .await?;
    stream.write_all(b"0\r\n\r\n").await?;
    stream.flush().await?;
    Ok(())
}

/// The worst-case cost of `request` under the catalog row it names, as the
/// gateway computes it: the flat per-request price, or the prompt priced by
/// the shared contract plus the completion ceiling (the request's own, or the
/// row's declared one) at the completion rate, each rounded up. `None` for a
/// model the catalog does not list.
fn worst_case_cost(config: &MockConfig, request: &serde_json::Value) -> Option<u128> {
    let catalog: serde_json::Value = serde_json::from_str(&models_body(config)).ok()?;
    let model = request.get("model")?.as_str()?;
    let row = catalog
        .get("data")?
        .as_array()?
        .iter()
        .find(|row| row.get("id").and_then(|id| id.as_str()) == Some(model))?;
    let price = |key: &str| -> Option<(u128, u128)> {
        let p = row.get("pricing")?.get(key)?;
        Some((
            u128::from(p.get("value")?.as_u64()?),
            u128::from(p.get("scale_factor")?.as_u64()?),
        ))
    };
    if let Some((value, scale)) = price("per_request") {
        return Some(value.div_ceil(scale));
    }
    let (prompt_rate, scale) = price("per_prompt_token")?;
    let (completion_rate, _) = price("per_completion_token")?;
    let messages = request.get("messages")?.as_array()?;
    let tools = request
        .get("tools")
        .and_then(|t| t.as_array())
        .map(Vec::as_slice);
    let prompt =
        u128::from(eidola_common::prompt_charge(messages, tools).chargeable_prompt_tokens());
    let completion = request
        .get("max_completion_tokens")
        .and_then(|m| m.as_u64())
        .or_else(|| row.get("max_output_tokens").and_then(|m| m.as_u64()))
        .or_else(|| row.get("context_length").and_then(|m| m.as_u64()))?;
    Some(
        (prompt * prompt_rate).div_ceil(scale)
            + (u128::from(completion) * completion_rate).div_ceil(scale),
    )
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
    if let Some(leaf) = &config.declared_prompt_cache {
        if primary.get("capabilities").is_none() {
            primary["capabilities"] = serde_json::json!({});
        }
        primary["capabilities"]["prompt_cache"] = leaf.clone();
    }
    if let Some(ceiling) = config.declared_max_output {
        primary["max_output_tokens"] = serde_json::json!(ceiling);
    }
    if let Some(sha256) = config
        .published_sha256
        .lock()
        .expect("published hash lock poisoned")
        .clone()
    {
        primary["sha256"] = serde_json::json!(sha256);
        primary["tags"] = serde_json::json!([MODEL]);
    }
    let untagged = config
        .untagged_sha256
        .lock()
        .expect("untagged hash lock poisoned")
        .clone()
        .map(|sha256| {
            let mut row = primary.clone();
            row["id"] = serde_json::json!(format!("sha256:{sha256}"));
            row["sha256"] = serde_json::json!(sha256);
            row["tags"] = serde_json::json!([]);
            row
        });
    let router = serde_json::json!({
        "id": ROUTER_REMOTE_MODEL,
        "context_length": 8192u64,
        "pricing": {
            "per_prompt_token": { "value": 1u64, "scale_factor": 1u64 },
            "per_completion_token": { "value": 1u64, "scale_factor": 1u64 }
        }
    });
    // **A model priced per request, not per token** — the shape the server's
    // catalog gives a flat-priced model: zero token rates, and the price in
    // `per_request`. A client that drops the flat price holds zero and refuses.
    let flat = serde_json::json!({
        "id": FLAT_MODEL,
        "context_length": 8192u64,
        "pricing": {
            "per_prompt_token": { "value": 0u64, "scale_factor": 1_000_000u64 },
            "per_completion_token": { "value": 0u64, "scale_factor": 1_000_000u64 },
            "per_request": { "value": FLAT_PRICE * 1_000_000, "scale_factor": 1_000_000u64 }
        }
    });
    let omitted = config
        .catalog_omits
        .lock()
        .expect("catalog omissions lock poisoned")
        .clone();
    let listed: Vec<serde_json::Value> = [primary, router]
        .into_iter()
        .chain(config.list_flat_model.then_some(flat))
        .chain(untagged)
        .filter(|entry| {
            let id = entry["id"].as_str().unwrap_or_default();
            !omitted.iter().any(|o| o == id)
        })
        .collect();
    serde_json::json!({ "data": listed }).to_string()
}

fn keys_body(issuer: &Issuer, config: &MockConfig) -> String {
    let ds = format!("ACT-v1:{DS_ORG}:{DS_SERVICE}:{DS_DEPLOYMENT}:{DS_VERSION}");
    // Reproduce the production `/v1/keys` shape: the previous epoch's key
    // (out of its issuing window but still within `accept_until`) sorts
    // *ahead* of the current key. It carries a different public key, so a
    // client that selected by domain separator alone — every key shares one
    // — would pick the decoy and fail proof verification against the current
    // issuer (the original bug). The client must skip the out-of-window
    // decoy and choose the current key.
    //
    // Like an honest server, the mock remembers which key it listed for each
    // epoch: the previous epoch's entry is the key it listed as current then
    // (the decoy only for an epoch it never listed). So a mock that keeps one
    // key across a boundary lists that key for two epochs, as a dishonest
    // server would; a test wanting the next epoch accepted gives it a new key
    // ([`KeySwap`]).
    let calendar = config.issuer_calendar.unwrap_or(ISSUER_CALENDAR);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("after 1970")
        .as_secs();
    let window = config.key_window.clone().unwrap_or_else(|| {
        KeyWindow::of_epoch(&calendar, calendar.epoch_at(now).expect("on the calendar"))
    });
    let current_epoch = eidola_app_core::iso_to_ms(&window.issue_from)
        .ok()
        .and_then(|ms| u64::try_from(ms.div_euclid(1000)).ok())
        .and_then(|secs| calendar.epoch_at(secs))
        .unwrap_or(0);
    let current = config.key_swap.active(issuer);
    let previous = {
        let mut listed = issuer.listed.lock().unwrap();
        listed
            .entry(current_epoch)
            .or_insert_with(|| (current.key_id_hex.clone(), current.public_key_b64()));
        listed.get(&current_epoch.wrapping_sub(1)).cloned()
    };
    let (previous_id, previous_pk) = previous.unwrap_or_else(|| {
        let hash: [u8; 32] = Sha256::digest(&issuer.decoy_public_key).into();
        (
            hex::encode(hash),
            URL_SAFE_NO_PAD.encode(&issuer.decoy_public_key),
        )
    });
    let decoy = KeyWindow::of_epoch(&calendar, current_epoch.saturating_sub(1));
    let previous_entry = serde_json::json!({
        "id": previous_id,
        "public_key": previous_pk,
        "domain_separator": ds,
        "issue_from": decoy.issue_from,
        "issue_until": decoy.issue_until,
        "accept_until": decoy.accept_until,
    });
    let current_entry = serde_json::json!({
        "id": current.key_id_hex,
        "public_key": current.public_key_b64(),
        "domain_separator": ds,
        "issue_from": window.issue_from,
        "issue_until": window.issue_until,
        "accept_until": window.accept_until,
    });
    let data = if config.omit_previous_key.load(Ordering::SeqCst) {
        vec![current_entry]
    } else {
        vec![previous_entry, current_entry]
    };
    serde_json::json!({ "data": data }).to_string()
}

fn error_body(message: &str) -> String {
    serde_json::json!({ "error": { "message": message } }).to_string()
}

fn uuid_like() -> String {
    // Deterministic-enough opaque id; the client only stores it.
    format!("{:016x}", rand_core::RngCore::next_u64(&mut OsRng))
}

tokio::task_local! {
    /// Whether the chat request this connection is answering is one the
    /// Eidola gateway answers: a spend, asking for a stream. The gateway
    /// serves chat only as a stream, so every completion such a request is
    /// answered with is written as the gateway writes one
    /// ([`write_gateway_stream`]), whatever behaviour wrote it.
    ///
    /// The value is how long such a stream runs before its first delta
    /// ([`MockConfig::answer_ms`]).
    static GATEWAY_STREAM: std::cell::Cell<Option<u64>>;
}

/// Whether this connection answers as the gateway does ([`GATEWAY_STREAM`]).
fn gateway_stream() -> bool {
    GATEWAY_STREAM
        .try_with(std::cell::Cell::get)
        .is_ok_and(|on| on.is_some())
}

async fn write_json(stream: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    if status == 200
        && gateway_stream()
        && let Ok(completion) = serde_json::from_str::<serde_json::Value>(body)
        && completion
            .get("choices")
            .and_then(|c| c.as_array())
            .is_some_and(|choices| choices.iter().all(|c| c.get("message").is_some()))
    {
        return write_gateway_stream(stream, &completion).await;
    }
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        402 => "Payment Required",
        404 => "Not Found",
        409 => "Conflict",
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
/// A backend that answers **every** request with a `307` pointing at a second
/// listener, and that listener — which records whether anything reached it and
/// answers whatever does with an ordinary success.
///
/// The fixture for the redirect rule: a client that follows replays the
/// request body — the whole prompt — to the `Location`, an origin nobody
/// configured. Returns the redirecting backend's base URL and the flag the
/// other origin sets. Plain threads rather than the mock's runtime, because a
/// test that holds a redirect policy has to run on an **ordinary** core (the
/// harness's injected client would answer for the one under test) and must
/// not depend on that core's runtime to serve its peer.
pub fn redirecting_upstream() -> (String, Arc<std::sync::atomic::AtomicBool>) {
    use std::io::{Read, Write};

    fn read_head_and_body(stream: &mut std::net::TcpStream) {
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            match stream.read(&mut byte) {
                Ok(1) => head.push(byte[0]),
                _ => return,
            }
        }
        let text = String::from_utf8_lossy(&head).to_string();
        let length: usize = text
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.trim()
                    .eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse().ok())?
            })
            .unwrap_or(0);
        let mut rest = vec![0u8; length];
        let _ = stream.read_exact(&mut rest);
    }

    let elsewhere = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let elsewhere_port = elsewhere.local_addr().expect("addr").port();
    let reached = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let saw = Arc::clone(&reached);
    std::thread::spawn(move || {
        while let Ok((mut stream, _)) = elsewhere.accept() {
            saw.store(true, Ordering::SeqCst);
            read_head_and_body(&mut stream);
            let body = r#"{"choices":[{"message":{"role":"assistant","content":"elsewhere"}}]}"#;
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });

    let backend = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let backend_port = backend.local_addr().expect("addr").port();
    std::thread::spawn(move || {
        while let Ok((mut stream, _)) = backend.accept() {
            read_head_and_body(&mut stream);
            let _ = write!(
                stream,
                "HTTP/1.1 307 Temporary Redirect\r\n\
                 Location: http://127.0.0.1:{elsewhere_port}/v1/chat/completions\r\n\
                 Content-Length: 0\r\nConnection: close\r\n\r\n"
            );
        }
    });
    (format!("http://127.0.0.1:{backend_port}"), reached)
}

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
    if let Some(calendar) = config.issuer_calendar {
        core.test_set_issuer_calendar(calendar);
    }
    let mock = core.runtime().block_on(async { start(config).await });
    core.runtime()
        .block_on(core.set_base_url(mock.base_url.clone()))
        .expect("set base url");
    seat_default_agent_on_mock(&core);
    (mock, core, dir)
}

/// Re-seat the Default template's one agent on the mock's [`MODEL`], labelled
/// [`DEFAULT_AGENT_LABEL`], so a space created from it is answered over the
/// mock. The compiled default is a local model, which a harness profile has no
/// file or engine for; `tests/template_seed.rs` and `tests/bus.rs` hold the
/// real seed.
fn seat_default_agent_on_mock(core: &AppCore) {
    core.runtime()
        .block_on(core.update_template(
            eidola_app_core::DEFAULT_TEMPLATE_ID.into(),
            None,
            None,
            Some(vec![eidola_app_core::NewTemplateParticipant {
                label: DEFAULT_AGENT_LABEL.into(),
                model_ref: Some(MODEL.into()),
                system_prompt: Some(SEEDED_AGENT_SYSTEM_PROMPT.into()),
                notify_policy: "human".into(),
            }]),
        ))
        .expect("seat the default agent on the mock's model");
}

/// The generic prompt the seeded default agent carries
/// (`db::DEFAULT_AGENT_SYSTEM_PROMPT`), kept on the re-seated agent.
const SEEDED_AGENT_SYSTEM_PROMPT: &str =
    "You are a helpful assistant. Answer clearly and concisely.";

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
/// A request context no other credential shares, for
/// [`MockConfig::issuance_context`].
#[allow(dead_code)]
pub fn distinct_context() -> Scalar {
    Scalar::from(7u64)
}

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
